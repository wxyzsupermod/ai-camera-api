from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

@dataclass
class DetectionConfig:
    frame_size: Tuple[int, int]
    target_fps: int
    exposure_us: int
    analogue_gain: float
    roi: Tuple[float, float, float, float]
    threshold: int
    alpha: float
    morph_kernel: int
    min_area: int
    max_area: int
    speed_px_min: float
    speed_px_max: float
    frames_to_confirm: int
    cooldown_seconds: float
    pre_frames: int
    post_frames: int
    output_dir: str
    preview: bool
    log_level: str
    save_debug_masks: bool

def load_config(path: str) -> DetectionConfig:
    if yaml is None:
        raise RuntimeError("Missing dependency 'pyyaml'. Install with: pip install pyyaml")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    return DetectionConfig(
        frame_size=tuple(data.get('frame_size', [640, 480])),
        target_fps=int(data.get('target_fps', 120)),
        exposure_us=int(data.get('exposure_us', 3000)),
        analogue_gain=float(data.get('analogue_gain', 1.0)),
        roi=tuple(data.get('roi', [0, 0, 1, 1])),
        threshold=int(data.get('threshold', 30)),
        alpha=float(data.get('alpha', 0.02)),
        morph_kernel=int(data.get('morph_kernel', 3)),
        min_area=int(data.get('min_area', 25)),
        max_area=int(data.get('max_area', 2500)),
        speed_px_min=float(data.get('speed_px_min', 2)),
        speed_px_max=float(data.get('speed_px_max', 100)),
        frames_to_confirm=int(data.get('frames_to_confirm', 3)),
        cooldown_seconds=float(data.get('cooldown_seconds', 0.5)),
        pre_frames=int(data.get('pre_frames', 0)),
        post_frames=int(data.get('post_frames', 0)),
        output_dir=str(data.get('output_dir', 'events')),
        preview=bool(data.get('preview', False)),
        log_level=str(data.get('log_level', 'info')),
        save_debug_masks=bool(data.get('save_debug_masks', False)),
    )

def save_config(cfg: DetectionConfig, path: str):  # pragma: no cover - side effect I/O
    if yaml is None:
        raise RuntimeError("Missing dependency 'pyyaml'. Install with: pip install pyyaml")
    data = {
        'frame_size': list(cfg.frame_size),
        'target_fps': cfg.target_fps,
        'exposure_us': cfg.exposure_us,
        'analogue_gain': cfg.analogue_gain,
        'roi': list(cfg.roi),
        'threshold': cfg.threshold,
        'alpha': cfg.alpha,
        'morph_kernel': cfg.morph_kernel,
        'min_area': cfg.min_area,
        'max_area': cfg.max_area,
        'speed_px_min': cfg.speed_px_min,
        'speed_px_max': cfg.speed_px_max,
        'frames_to_confirm': cfg.frames_to_confirm,
        'cooldown_seconds': cfg.cooldown_seconds,
        'pre_frames': cfg.pre_frames,
        'post_frames': cfg.post_frames,
        'output_dir': cfg.output_dir,
        'preview': cfg.preview,
        'log_level': cfg.log_level,
        'save_debug_masks': cfg.save_debug_masks,
    }
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)
