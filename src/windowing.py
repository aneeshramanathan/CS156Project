"""Task 4: Windowing strategies."""

import numpy as np
import pandas as pd

from src.utils import DEFAULT_SAMPLING_FREQUENCY, DEFAULT_OVERLAP, DEFAULT_WINDOW_SIZE


def create_windows(signal, window_size=DEFAULT_WINDOW_SIZE, overlap=DEFAULT_OVERLAP):
    """Create sliding windows from a signal."""
    step_size = int(window_size * (1 - overlap))
    windows = []
    
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        windows.append(window)
    
    return np.array(windows)


def analyze_window_sizes(signal, window_sizes, overlap=DEFAULT_OVERLAP, fs=DEFAULT_SAMPLING_FREQUENCY):
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

