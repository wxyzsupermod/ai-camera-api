#!/usr/bin/env python3
"""Wrapper module preserving original entry point.

The implementation was refactored into the package `gate_detector` for
maintainability. Import and delegate to `gate_detector.cli.main`.
"""
from gate_detector.cli import main  # noqa: F401

if __name__ == '__main__':  # pragma: no cover
    main()
