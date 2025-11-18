"""Shared utilities and constants for activity detection analysis."""

import os
import platform
from pathlib import Path

DEFAULT_SAMPLING_FREQUENCY = 50
DEFAULT_LOWCUT = 0.3
DEFAULT_HIGHCUT = 20

DEFAULT_WINDOW_SIZE = 300
DEFAULT_OVERLAP = 0.5

VISUALIZATIONS_DIR = Path("visualizations")


def detect_platform():
    system = platform.system()
    machine = platform.machine()
    
    is_apple_silicon = (system == 'Darwin' and machine == 'arm64')
    is_macos = (system == 'Darwin')
    is_linux = (system == 'Linux')
    is_windows = (system == 'Windows')
    
    try:
        cpu_count = os.cpu_count() or 4
    except:
        cpu_count = 4
    
    optimal_n_jobs = max(1, cpu_count - 1)
    
    return {
        'is_apple_silicon': is_apple_silicon,
        'is_macos': is_macos,
        'is_linux': is_linux,
        'is_windows': is_windows,
        'cpu_count': cpu_count,
        'optimal_n_jobs': optimal_n_jobs,
        'platform_name': f"{system} {machine}"
    }


def ensure_visualizations_dir():
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    return VISUALIZATIONS_DIR


