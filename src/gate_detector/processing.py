from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .config import DetectionConfig

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

def process_frame(gray: np.ndarray, roi_px: Tuple[int, int, int, int], bg: np.ndarray, cfg: DetectionConfig) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], np.ndarray]:
    x0, y0, x1, y1 = roi_px
    roi_slice = gray[y0:y1, x0:x1]
    bg_roi = bg[y0:y1, x0:x1]
    diff_roi = cv2.absdiff(roi_slice, bg_roi) if cv2 is not None else np.abs(roi_slice.astype(np.int16) - bg_roi.astype(np.int16)).astype(np.uint8)
    update_mask = diff_roi < (cfg.threshold * 0.5)
    bg_roi[update_mask] = (cfg.alpha * roi_slice[update_mask] + (1 - cfg.alpha) * bg_roi[update_mask]).astype(np.uint8)
    _, thresh = cv2.threshold(diff_roi, cfg.threshold, 255, cv2.THRESH_BINARY) if cv2 is not None else (None, (diff_roi > cfg.threshold).astype(np.uint8) * 255)
    if cfg.morph_kernel > 1 and cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_kernel, cfg.morph_kernel))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)
        thresh = cv2.dilate(thresh, k, iterations=1)
    boxes: List[Tuple[int, int, int, int]] = []
    if cv2 is not None:
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if not c.size:
                continue
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < cfg.min_area or area > cfg.max_area:
                continue
            boxes.append((x + x0, y + y0, w, h))
    else:
        ys, xs = np.where(thresh > 0)
        if xs.size > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            boxes.append((x_min + x0, y_min + y0, x_max - x_min + 1, y_max - y_min + 1))
    return bg, boxes, thresh
