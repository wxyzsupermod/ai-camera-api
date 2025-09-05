from __future__ import annotations
"""Background / motion model with calibration.

Strategy:
 1. Calibration phase (collect frames for N seconds) builds per-pixel mean and
    variance using Welford's online algorithm (memory efficient).
 2. Detection phase: absolute difference to mean compared against
        dynamic_threshold = base_threshold + std_factor * sqrt(var).
    This suppresses false positives from gently moving foliage / sensor noise.
 3. Model update: only pixels NOT currently flagged as motion are
    incorporated back into the running mean / variance using an exponential
    moving average (learning rate = alpha) so slow illumination drift is tracked.

The existing pipeline expected a simple background frame. We encapsulate the
enhanced logic here while keeping a similar interface (detect returns boxes
and a binary mask for visualization / debugging).
"""
from typing import List, Tuple
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


class BackgroundModel:
    def __init__(self, shape: Tuple[int, int], *, threshold: int, std_factor: float, alpha: float, morph_kernel: int, min_area: int, max_area: int):
        self.h, self.w = shape
        self.threshold = threshold
        self.std_factor = std_factor
        self.alpha = alpha
        self.morph_kernel = morph_kernel
        self.min_area = min_area
        self.max_area = max_area
        # Calibration accumulators
        self._count = 0
        self._mean = np.zeros((self.h, self.w), dtype=np.float32)
        self._M2 = np.zeros((self.h, self.w), dtype=np.float32)
        self.ready = False

    def reset(self):
        self._count = 0
        self._mean.fill(0)
        self._M2.fill(0)
        self.ready = False

    def add_calibration_frame(self, frame: np.ndarray):
        """Accumulate a frame during calibration (grayscale uint8)."""
        f = frame.astype(np.float32)
        self._count += 1
        delta = f - self._mean
        self._mean += delta / self._count
        delta2 = f - self._mean
        self._M2 += delta * delta2

    def finalize(self):
        if self._count >= 2:
            # variance estimate
            var = self._M2 / (self._count - 1)
            # Avoid zero variance (would produce overly sensitive pixels)
            var[var < 4.0] = 4.0  # minimal noise floor (≈ std 2)
            self._var = var
            self.ready = True
        else:  # Fallback: treat single frame as mean, large variance
            self._var = np.full_like(self._mean, 25.0)  # std 5
            self.ready = True

    def _dynamic_mask(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        x0, y0, x1, y1 = roi
        roi_slice = frame[y0:y1, x0:x1].astype(np.float32)
        mean_roi = self._mean[y0:y1, x0:x1]
        var_roi = self._var[y0:y1, x0:x1]
        std_roi = np.sqrt(var_roi)
        diff = np.abs(roi_slice - mean_roi)
        dyn_thresh = self.threshold + self.std_factor * std_roi
        motion = diff > dyn_thresh
        mask = np.zeros_like(frame, dtype=np.uint8)
        mask[y0:y1, x0:x1][motion] = 255
        return mask

    def detect(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        if not self.ready:
            raise RuntimeError('BackgroundModel not calibrated')
        mask = self._dynamic_mask(frame, roi)
        # Morphology cleanup
        if self.morph_kernel > 1 and cv2 is not None:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.dilate(mask, k, iterations=1)
        boxes: List[Tuple[int, int, int, int]] = []
        if cv2 is not None:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if not c.size:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if area < self.min_area or area > self.max_area:
                    continue
                boxes.append((x, y, w, h))
        else:  # very coarse fallback: single bounding box around all motion
            ys, xs = np.where(mask > 0)
            if xs.size:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                area = w * h
                if self.min_area <= area <= self.max_area:
                    boxes.append((x_min, y_min, w, h))
        return boxes, mask

    def update(self, frame: np.ndarray, mask: np.ndarray):
        """Update mean/var for background where mask == 0 (no motion)."""
        if not self.ready:
            return
        stable = mask == 0
        if not np.any(stable):  # nothing to update
            return
        f = frame.astype(np.float32)
        # Update mean with EMA
        self._mean[stable] = (1 - self.alpha) * self._mean[stable] + self.alpha * f[stable]
        # Update variance towards new squared diff (EMA as well)
        diff = f[stable] - self._mean[stable]
        cur_var = (diff * diff)
        self._var[stable] = (1 - self.alpha) * self._var[stable] + self.alpha * cur_var
        # Clamp minimal variance
        np.maximum(self._var, 4.0, out=self._var)
