"""HTTP server for detector (clean rewrite).

Endpoints:
  GET  /                       -> UI (static/index.html)
  GET  /static/<file>          -> static assets
  GET  /snapshot               -> latest JPEG frame (single)
  GET  /stream                 -> MJPEG stream
  GET  /events/latest          -> latest event meta.json
  GET  /health                 -> uptime ok
  GET  /status                 -> detection enabled flag
  GET  /config                 -> current config JSON
  PUT  /config                 -> update subset of config fields (JSON body)
  POST /control/start|stop     -> enable / disable detection
  POST /calibration/start      -> begin frame capture (query ?frames=N)
  GET  /calibration/status     -> calibration capture progress
  POST /calibration/apply      -> derive min/max area from captured frames
  GET  /diagnostics            -> basic runtime info
"""

from __future__ import annotations
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Any, List
from urllib.parse import urlparse, parse_qs

import numpy as np

from .logging_utils import log
from .config import save_config

STATIC_DIR = Path(__file__).parent / 'static'
INDEX_FILE = STATIC_DIR / 'index.html'


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.last_event_dir: Optional[str] = None
        self.start_time = time.time()
        self.detection_enabled: bool = False
        self.cfg: Optional[Any] = None
        self.cfg_path: Optional[str] = None
        # Calibration capture
        self.calib_active: bool = False
        self.calib_target: int = 0
        self.calib_frames: List[np.ndarray] = []


class DetectorHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    shared: SharedState = None  # type: ignore
    stop_flag = None  # type: ignore

    # ---------- helpers ----------
    def log_message(self, format, *args):  # silence
        return

    def _send_bytes(self, code: int, body: bytes, content_type: str = 'text/plain', extra_headers=None):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _send_json(self, obj, code: int = 200):
        self._send_bytes(code, json.dumps(obj).encode(), 'application/json')

    # ---------- verbs ----------
    def do_OPTIONS(self):  # CORS
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,PUT,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith('/static/'):
                rel = path[len('/static/') :]
                target = STATIC_DIR / rel
                if not target.exists() or not target.is_file():
                    self._send_bytes(404, b'not found'); return
                mime = 'text/plain'
                if target.suffix == '.html': mime = 'text/html'
                elif target.suffix == '.js': mime = 'application/javascript'
                elif target.suffix == '.css': mime = 'text/css'
                self._send_bytes(200, target.read_bytes(), mime); return
            if path == '/':
                if INDEX_FILE.exists():
                    self._send_bytes(200, INDEX_FILE.read_bytes(), 'text/html'); return
                self._send_bytes(500, b'UI missing'); return
            if path == '/snapshot':
                with self.shared.lock:
                    data = self.shared.latest_jpeg
                if not data:
                    self._send_bytes(503, b'No frame yet'); return
                self._send_bytes(200, data, 'image/jpeg'); return
            if path == '/events/latest':
                with self.shared.lock:
                    d = self.shared.last_event_dir
                if not d:
                    self._send_bytes(204, b''); return
                meta = Path(d) / 'meta.json'
                if not meta.exists():
                    self._send_bytes(404, b''); return
                self._send_bytes(200, meta.read_bytes(), 'application/json'); return
            if path == '/stream':
                boundary = 'FRAME'
                self.send_response(200)
                self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={boundary}')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                try:
                    while not self.stop_flag.is_set():
                        with self.shared.lock:
                            data = self.shared.latest_jpeg
                        if data:
                            self.wfile.write(b'--' + boundary.encode() + b'\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n')
                            self.wfile.write(data + b'\r\n')
                        time.sleep(0.07)
                except BrokenPipeError:
                    pass
                return
            if path == '/health':
                self._send_json({'ok': True, 'uptime': time.time() - self.shared.start_time}); return
            if path == '/status':
                self._send_json({'detection_enabled': self.shared.detection_enabled}); return
            if path == '/config':
                with self.shared.lock:
                    cfg = self.shared.cfg
                if not cfg:
                    self._send_json({'error': 'no config loaded'}, 503); return
                keys = ['frame_size','target_fps','exposure_us','analogue_gain','roi','threshold','alpha','morph_kernel','min_area','max_area','speed_px_min','speed_px_max','frames_to_confirm','cooldown_seconds','pre_frames','post_frames','output_dir','preview','log_level','save_debug_masks']
                self._send_json({k: getattr(cfg, k) for k in keys}); return
            if path == '/calibration/status':
                with self.shared.lock:
                    st = {'active': self.shared.calib_active, 'captured': len(self.shared.calib_frames), 'target': self.shared.calib_target}
                self._send_json(st); return
            if path == '/diagnostics':
                import sys
                self._send_json({'python': sys.version, 'detection_enabled': self.shared.detection_enabled, 'uptime': time.time() - self.shared.start_time}); return
            self._send_bytes(404, b'not found')
        except Exception as e:  # pragma: no cover
            import traceback
            log(f'HTTP GET error {path}: {e}', 'error')
            try: self._send_bytes(500, b'Internal Error')
            except Exception: pass

    def do_PUT(self):  # config update
        if self.path != '/config':
            self._send_bytes(404, b'not found'); return
        length = int(self.headers.get('Content-Length', '0') or 0)
        raw = self.rfile.read(length) if length > 0 else b'{}'
        try:
            data = json.loads(raw.decode() or '{}')
        except json.JSONDecodeError:
            self._send_json({'error': 'invalid json'}, 400); return
        with self.shared.lock:
            cfg = self.shared.cfg; path_cfg = self.shared.cfg_path
            if not cfg or not path_cfg:
                self._send_json({'error': 'config not loaded'}, 503); return
            updated = {}
            for k, v in data.items():
                if k == 'roi' and isinstance(v, list) and len(v) == 4:
                    cfg.roi = tuple(float(x) for x in v); updated['roi'] = cfg.roi
                elif hasattr(cfg, k):
                    setattr(cfg, k, v); updated[k] = getattr(cfg, k)
            try: save_config(cfg, path_cfg)
            except Exception as e: self._send_json({'error': f'failed to save: {e}'}, 500); return
        self._send_json({'updated': updated})

    def do_POST(self):  # control & calibration
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == '/control/start':
            with self.shared.lock: self.shared.detection_enabled = True
            self._send_json({'detection_enabled': True}); return
        if path == '/control/stop':
            with self.shared.lock: self.shared.detection_enabled = False
            self._send_json({'detection_enabled': False}); return
        if path == '/calibration/start':
            frames = int(qs.get('frames', ['120'])[0])
            if frames <= 0 or frames > 5000:
                self._send_json({'error': 'frames out of range'}, 400); return
            with self.shared.lock:
                self.shared.calib_active = True
                self.shared.calib_target = frames
                self.shared.calib_frames.clear()
            self._send_json({'ok': True, 'target': frames}); return
        if path == '/calibration/apply':
            with self.shared.lock:
                cfg = self.shared.cfg
                frames = list(self.shared.calib_frames)
                self.shared.calib_active = False
            if not cfg:
                self._send_json({'error': 'no config'}, 503); return
            if not frames:
                self._send_json({'error': 'no frames captured'}, 400); return
            w, h = cfg.frame_size
            rx0, ry0, rx1, ry1 = cfg.roi
            x0, y0, x1, y1 = int(rx0 * w), int(ry0 * h), int(rx1 * w), int(ry1 * h)
            areas: List[int] = []
            for fr in frames:
                roi_arr = fr[y0:y1, x0:x1]
                a = int(np.count_nonzero(roi_arr))
                if a > 0: areas.append(a)
            if not areas:
                self._send_json({'error': 'no motion areas detected'}, 400); return
            areas.sort()
            def pct(p: float):
                idx = int(p * (len(areas) - 1)); return areas[idx]
            new_min = max(5, int(pct(0.10) * 0.8))
            new_max = int(pct(0.90) * 1.2)
            with self.shared.lock:
                cfg.min_area = new_min; cfg.max_area = new_max
                try: save_config(cfg, self.shared.cfg_path)  # type: ignore
                except Exception as e: log(f'Failed to save calibrated config: {e}', 'error', cfg_level=getattr(cfg,'log_level','info'))
            self._send_json({'min_area': new_min, 'max_area': new_max, 'samples': len(areas)}); return
        self._send_bytes(404, b'not found')


def start_http_server(host: str, port: int, shared: SharedState, stop_flag):
    try:
        DetectorHTTPHandler.shared = shared
        DetectorHTTPHandler.stop_flag = stop_flag
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        server = ThreadingHTTPServer((host, port), DetectorHTTPHandler)
    except OSError as e:
        log(f'HTTP bind failed: {e}', 'error'); return
    log(f'HTTP on http://{host}:{port} (endpoints: / /snapshot /stream /events/latest /health /status /control/start /control/stop /config /calibration/start /calibration/status /calibration/apply /diagnostics)')
    def run():
        try:
            server.serve_forever()
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()
