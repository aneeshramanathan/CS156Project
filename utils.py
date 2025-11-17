"""Shared utilities and constants for activity detection analysis."""

import platform
import os
from pathlib import Path

# Sampling frequency (Hz)
DEFAULT_SAMPLING_FREQUENCY = 50

# Filter parameters
DEFAULT_LOWCUT = 0.3
DEFAULT_HIGHCUT = 20

# Windowing defaults
DEFAULT_WINDOW_SIZE = 150
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


def get_optimized_model_params():
    """Get optimized model parameters based on platform."""
    platform_info = detect_platform()
    
    # Use cpu_count - 1 threads for all models
    n_jobs = platform_info['optimal_n_jobs']
    
    if platform_info['is_apple_silicon']:
        # Optimized for Apple Silicon - reduce complexity for faster training
        return {
            'random_forest': {
                'n_estimators': 50,  # Reduced from 100
                'max_depth': 12,     # Reduced from 15
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': n_jobs
            },
            'adaboost': {
                'n_estimators': 30,  # Reduced from 50
                'learning_rate': 1.0,
                'random_state': 42
            },
            'decision_tree': {
                'max_depth': 8,       # Reduced from 10
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42
            }
        }
    else:
        # Standard parameters for other platforms
        return {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': n_jobs
            },
            'adaboost': {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': 42
            },
            'decision_tree': {
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42
            }
        }


def get_device():
    """
    Auto-detect and return the best available device for PyTorch.
    Returns: 'mps' for Apple Silicon, 'cuda' for NVIDIA GPU, 'cpu' as fallback
    """
    try:
        import torch
        
        if torch.backends.mps.is_available():
            return torch.device('mps'), 'MPS (Apple Silicon GPU)'
        elif torch.cuda.is_available():
            return torch.device('cuda'), f'CUDA (NVIDIA GPU: {torch.cuda.get_device_name(0)})'
        else:
            return torch.device('cpu'), 'CPU'
    except ImportError:
        return None, 'CPU (PyTorch not installed)'


def ensure_visualizations_dir():
    """Create visualizations directory if it doesn't exist."""
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    return VISUALIZATIONS_DIR

