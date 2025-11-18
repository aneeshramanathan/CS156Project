"""Shared utilities and constants for activity detection analysis."""

import platform
import os
from pathlib import Path
import numpy as np

# Sampling frequency (Hz)
DEFAULT_SAMPLING_FREQUENCY = 50

# Filter parameters
DEFAULT_LOWCUT = 0.3
DEFAULT_HIGHCUT = 20

# Windowing defaults
# Task 7's expanded sweep showed 300 samples (~6s) consistently yielded the
# highest accuracy/F1 for the EdgeImpulse dataset, so we adopt it as default.
DEFAULT_WINDOW_SIZE = 300
DEFAULT_OVERLAP = 0.5

# Visualization output directory
VISUALIZATIONS_DIR = Path("visualizations")


def detect_platform():
    """Detect the current platform and return optimization settings."""
    system = platform.system()
    machine = platform.machine()
    
    is_apple_silicon = (system == 'Darwin' and machine == 'arm64')
    is_macos = (system == 'Darwin')
    is_linux = (system == 'Linux')
    is_windows = (system == 'Windows')
    
    # Get CPU count for thread optimization
    try:
        cpu_count = os.cpu_count() or 4
    except:
        cpu_count = 4
    
    # Use all cores except 1 to leave one for system
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
    """Create visualizations directory if it doesn't exist."""
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    return VISUALIZATIONS_DIR


