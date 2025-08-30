"""Synthetic test (offline) for drone_gate_detector core logic.

This doesn't validate real camera interaction; it exercises the processing pipeline
using the synthetic motion path (enabled via --synthetic).
"""
import subprocess
import sys
from pathlib import Path


def test_synthetic_run():
    # Run a few seconds synthetic to ensure no crashes and events produced.
    cmd = [sys.executable, "src/drone_gate_detector.py", "--config", "config.yaml", "--synthetic", "--auto-start"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Let it run for ~3 seconds
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.terminate()
    # Check events directory for at least one event
    events = list(Path("events").glob("*/frame.jpg"))
    assert events, "No event frames saved during synthetic test"
