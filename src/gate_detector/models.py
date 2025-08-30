from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import numpy as np

@dataclass
class Candidate:
    centers: Deque[Tuple[float, float]]
    first_ts: float
    last_ts: float

    def add(self, c: Tuple[float, float], ts: float):
        self.centers.append(c)
        self.last_ts = ts

    def speed(self) -> Optional[float]:
        if len(self.centers) < 2:
            return None
        (x0, y0), (x1, y1) = self.centers[-2], self.centers[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def direction_stability(self) -> float:
        if len(self.centers) < 3:
            return 0.0
        pts = np.array(self.centers)
        pts_centered = pts - pts.mean(axis=0)
        if pts_centered.shape[0] < 2:
            return 0.0
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_vec = eigvecs[:, np.argmax(eigvals)]
        perp_vec = np.array([-major_vec[1], major_vec[0]])
        deviations = np.abs(pts_centered @ perp_vec)
        span = np.max(pts_centered @ major_vec) - np.min(pts_centered @ major_vec)
        if span <= 1e-6:
            return 0.0
        return float(np.mean(deviations) / (span + 1e-9))

class RingBuffer:
    def __init__(self, size: int):
        from typing import Deque as _Deque
        import numpy as _np
        self.size = size
        self.buf: _Deque[Tuple[float, _np.ndarray]] = deque(maxlen=size)

    def add(self, ts: float, frame):
        if self.size == 0:
            return
        self.buf.append((ts, frame.copy()))

    def dump(self) -> List[Tuple[float, np.ndarray]]:
        return list(self.buf)
