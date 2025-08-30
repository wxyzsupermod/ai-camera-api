from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from .config import DetectionConfig
from .models import Candidate

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_event(cfg: DetectionConfig, event_ts: float, frame: np.ndarray, candidate: Optional[Candidate], pre_frames: List[Tuple[float, np.ndarray]], post_frames: List[Tuple[float, np.ndarray]], debug_masks: Optional[List[np.ndarray]] = None):
    ts_iso = datetime.fromtimestamp(event_ts, tz=timezone.utc).isoformat().replace(':', '-')
    out_dir = Path(cfg.output_dir) / ts_iso
    ensure_dir(out_dir)
    import cv2
    cv2.imwrite(str(out_dir / 'frame.jpg'), frame)
    for i, (ts, f) in enumerate(pre_frames):
        cv2.imwrite(str(out_dir / f'pre_{i:02d}.jpg'), f)
    for i, (ts, f) in enumerate(post_frames):
        cv2.imwrite(str(out_dir / f'post_{i:02d}.jpg'), f)
    if cfg.save_debug_masks and debug_masks:
        for i, m in enumerate(debug_masks):
            cv2.imwrite(str(out_dir / f'mask_{i:02d}.png'), m)
    meta = {
        'event_timestamp_utc': ts_iso,
        'candidate_centers': list(candidate.centers) if candidate else [],
        'first_candidate_ts': getattr(candidate, 'first_ts', None),
        'last_candidate_ts': getattr(candidate, 'last_ts', None),
        'config': asdict(cfg),
    }
    with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
