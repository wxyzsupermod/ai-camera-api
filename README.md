Drone Gate Passage Detection
============================

High–framerate Raspberry Pi 5 + Camera Module 3 motion trigger for fast, small drones passing through a gate. Captures timestamped frames (optionally short pre/post frame sequences) when a small fast object crosses a defined Region of Interest (ROI). No counting or identity, just event capture.

## Features
* High FPS capture (target 120 FPS @ 640x480 or cropped mode) using Picamera2.
* ROI mask (only analyze gate opening).
* Lightweight background / frame differencing tuned for small, fast objects.
* Size + speed + straight‑line motion heuristics to reduce false triggers (e.g. leaves, shadows).
* Cooldown to avoid duplicate events.
* Optional pre/post buffering (ring buffer) for context frames.
* Minimal dependencies (numpy, opencv-python-headless, pyyaml, picamera2).

## Hardware & Setup Tips
* Ensure strong, even lighting; use high shutter speed (short exposure) to freeze motion (e.g. ExposureTime 1500–4000 µs at 120 FPS).
* Lock auto‑exposure & auto‑white-balance after initial convergence to avoid background model churn.
* Angle camera slightly downward to keep the sky (fast brightness changes) out of ROI.
* Physically mask (tape / 3D print) extraneous regions if they cause noise (moving grass, reflective surfaces).
* Consider a contrasting backdrop behind the gate to increase object contrast.

## Installation
Enable camera:
```bash
sudo raspi-config nonint do_camera 0  # (or use GUI raspi-config)
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-yaml
```
If OpenCV package above is outdated or missing modules, fallback to pip (slower build):
```bash
python3 -m pip install --upgrade pip
pip install opencv-python-headless numpy pyyaml
```

Clone / copy this project, then install Python deps (if not using system packages):
```bash
pip install -r requirements.txt
```

## Configuration
Edit `config.yaml`:
* `frame_size`: Sensor output resolution (smaller => faster). Start 640x480.
* `roi`: Fractional (x0,y0,x1,y1) gate rectangle within frame.
* `min_area` / `max_area`: Pixel area bounds for candidate object (after morphology) at chosen resolution.
* `speed_px_min` / `speed_px_max`: Per‑frame center displacement bounds.
* `frames_to_confirm`: Consecutive frames needed before triggering.
* `cooldown_seconds`: Minimum gap between events.
* `pre_frames` / `post_frames`: Context frames saved around detection.
* `output_dir`: Where events (JPEG + JSON) are written.

## Run
```bash
python3 src/drone_gate_detector.py --config config.yaml
```
Stop with Ctrl+C. By default detection is idle until you enable it.

Start the HTTP server (includes a minimal built‑in web UI at `/`):
```bash
python3 src/drone_gate_detector.py --config config.yaml --http-port 8080
```
Then:
* Visit `http://<host>:8080/` for a simple control + stream page (Start/Stop/Snapshot)
* `GET /stream` MJPEG stream (overlay frame)
* `GET /snapshot` latest processed frame (JPEG)
* `GET /control/start` enables detection / event saving (JSON)
* `GET /control/stop` pauses detection (stream & snapshot still available)
* `GET /status` returns JSON detection_enabled state
* `GET /events/latest` returns latest event meta JSON
* `GET /diagnostics` runtime / environment info

Use `--auto-start` to begin detecting immediately (legacy behavior).
Events are saved under `events/` only while detection is enabled.

## Output
For each event: `events/<UTC_ISO_TIMESTAMP>/frame.jpg` plus `meta.json` (timestamp(s), detection metrics, centers path, config snapshot).

## Tuning Workflow
1. Run with `--preview` to optionally display (if a display is attached / X forwarding) for tuning thresholds.
2. Adjust `threshold` first to isolate only the drone (white blob on diff mask) while ignoring noise.
3. Adjust `min_area`, `max_area` to bracket blob size (inspect console logs).
4. Adjust `speed_px_min/max` by logging reported speeds.
5. Verify consecutive frames logic: ensure `frames_to_confirm` not too high (a very fast drone might appear only 3–5 frames at 120 FPS).
6. Set `cooldown_seconds` just longer than maximum expected transit time.

## Background / Motion Model (Adaptive)
At startup the detector now performs a short calibration phase (default `calibration_seconds: 2.0`). It collects grayscale frames and builds per‑pixel mean and variance using Welford's online method. During detection a pixel is considered motion if:

	|I - mean| > threshold + std_factor * std

Config additions:
* `std_factor` (default 2.5): multiplies per‑pixel standard deviation to derive adaptive threshold component (suppresses moving grass / sensor flicker).
* `calibration_seconds` (default 2.0): duration of initial statistics collection before motion detection begins.

Only pixels not currently flagged as motion update the background (EMA with `alpha`) to follow slow illumination drift without absorbing fast moving drones.

The legacy simple frame differencing path remains for fallback / compatibility but is superseded by the adaptive model in normal operation.

## Performance Notes
* Keep Python loop lean: disable debug overlays once tuned.
* If CPU bound, reduce resolution further (e.g. 512x384, 480x360) or set grayscale only pipeline.
* Consider overclocking GPU core modestly (at your risk) or using a cropped mode focusing only on gate to increase effective FPS.
* If still dropping frames, switch to raw ring buffer using Picamera2 callback thread and process asynchronously (future enhancement).

## Future Enhancements (Not Yet Implemented)
* Asynchronous producer (camera) / consumer (detector) queues.
* Optional TensorRT / tiny model classifier to filter non‑drone objects.
* Multi‑gate support (multiple ROIs).
* Adaptive thresholding for varying light.

## License
MIT (add LICENSE file if needed).
