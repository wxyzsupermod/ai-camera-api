#!/usr/bin/env python3
"""Wrapper module preserving original entry point.

The implementation was refactored into the package `gate_detector` for
maintainability. Import and delegate to `gate_detector.cli.main`.
"""
import os
import sys
from pathlib import Path
import glob

# Add the virtual environment to the Python path dynamically
script_dir = Path(__file__).parent
venv_base = script_dir.parent / ".venv" / "lib"

# Find the Python version directory dynamically
python_dirs = glob.glob(str(venv_base / "python*"))
if python_dirs:
    venv_path = Path(python_dirs[0]) / "site-packages"
    if venv_path.exists():
        sys.path.insert(0, str(venv_path))

from gate_detector.cli import main  # noqa: F401

if __name__ == '__main__':  # pragma: no cover
    main()
