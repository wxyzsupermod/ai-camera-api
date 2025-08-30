from __future__ import annotations
import time
from typing import Optional
from .config import DetectionConfig
import sys, os

try:
    from picamera2 import Picamera2  # type: ignore
except ImportError:  # pragma: no cover
    # Attempt dynamic path injection for Debian-based system installs
    Picamera2 = None  # type: ignore
    potential_paths = [
        '/usr/lib/python3/dist-packages',  # Debian/RPi OS site-packages
        '/usr/local/lib/python3/dist-packages',
    ]
    for p in potential_paths:
        if p not in sys.path and os.path.isdir(p):
            sys.path.append(p)
            try:
                from picamera2 import Picamera2  # type: ignore
                break
            except Exception:  # pragma: no cover
                continue

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

from .logging_utils import log


def init_camera(cfg: DetectionConfig, *, manual_exposure: bool = True) -> Optional[Picamera2]:  # type: ignore
    if Picamera2 is None:
        log('Picamera2 not available; running synthetic mode only.', 'error', cfg_level=cfg.log_level)
        return None
    cam = Picamera2()
    w, h = cfg.frame_size
    try:
        video_config = cam.create_video_configuration(main={'size': (w, h), 'format': 'RGB888'})
        cam.configure(video_config)
    except Exception as e:  # pragma: no cover
        log(f'Primary camera config failed ({e}); trying default format', 'warn', cfg_level=cfg.log_level)
        try:
            video_config = cam.create_video_configuration(main={'size': (w, h)})
            cam.configure(video_config)
        except Exception as e2:
            log(f'Fallback camera config failed: {e2}', 'error', cfg_level=cfg.log_level)
            return None
    try:
        cam.start()
    except Exception as e:
        log(f'Camera start failed: {e}', 'error', cfg_level=cfg.log_level); return None
    try:
        frame_duration_us = int(1_000_000 / cfg.target_fps)
        cam.set_controls({'FrameDurationLimits': (frame_duration_us, frame_duration_us)})
    except Exception as e:
        log(f'Setting FrameDurationLimits failed (continuing): {e}', 'warn', cfg_level=cfg.log_level)
    if manual_exposure:
        try:
            cam.set_controls({'ExposureTime': cfg.exposure_us, 'AnalogueGain': cfg.analogue_gain})
        except Exception as e:
            log(f'Setting exposure/gain failed (continuing auto for now): {e}', 'warn', cfg_level=cfg.log_level)
    else:
        log('Auto exposure warmup enabled (will lock after warmup).', cfg_level=cfg.log_level)
    time.sleep(0.2)
    return cam


def capture_frame(cam: Optional[Picamera2], cfg: DetectionConfig):  # type: ignore
    if cam is None:
        raise RuntimeError('Camera not initialized (or running in synthetic test mode).')
    arr = cam.capture_array('main')
    if cv2 is not None:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        import numpy as np
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
    return gray
