"""Task 4: Windowing strategies."""

import numpy as np
import pandas as pd
from src.utils import DEFAULT_SAMPLING_FREQUENCY, DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP


def create_windows(signal, window_size=DEFAULT_WINDOW_SIZE, overlap=DEFAULT_OVERLAP):
    """
    Create sliding windows from a signal.
    
    Args:
        signal: Input signal array
        window_size: Size of each window in samples
        overlap: Overlap ratio between consecutive windows (0 to 1)
    
    Returns:
        Array of windows
    """
    step_size = int(window_size * (1 - overlap))
    windows = []
    
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        windows.append(window)
    
    return np.array(windows)


def analyze_window_sizes(signal, window_sizes, overlap=DEFAULT_OVERLAP, fs=DEFAULT_SAMPLING_FREQUENCY):
    """
    Analyze multiple window sizes and return statistics.
    
    Args:
        signal: Input signal array
        window_sizes: List of window sizes to test
        overlap: Overlap ratio
        fs: Sampling frequency
    
    Returns:
        DataFrame with window analysis results
    """
    window_analysis = []
    
    for ws in window_sizes:
        windows = create_windows(signal, ws, overlap)
        
        window_analysis.append({
            'window_size': ws,
            'n_windows': len(windows),
            'coverage': (len(windows) * ws * (1 - overlap) + ws * overlap) / len(signal),
            'time_duration_sec': ws / fs,
        })
    
    return pd.DataFrame(window_analysis)

