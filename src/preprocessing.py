"""Task 3: Signal preprocessing functions."""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from src.utils import DEFAULT_SAMPLING_FREQUENCY, DEFAULT_LOWCUT, DEFAULT_HIGHCUT


def preprocess_signal(signal_array, fs=DEFAULT_SAMPLING_FREQUENCY, 
                      lowcut=DEFAULT_LOWCUT, highcut=DEFAULT_HIGHCUT, 
                      interpolate_missing=True):
    """
    Preprocess sensor signals with filtering and interpolation.
    
    Args:
        signal_array: Input signal
        fs: Sampling frequency
        lowcut: Low cutoff frequency for bandpass filter
        highcut: High cutoff frequency for bandpass filter
        interpolate_missing: Whether to interpolate NaN values
    
    Returns:
        Preprocessed signal
    """
    signal_copy = signal_array.copy()
    
    if interpolate_missing and np.any(np.isnan(signal_copy)):
        valid_indices = ~np.isnan(signal_copy)
        if np.sum(valid_indices) > 1:
            x = np.arange(len(signal_copy))
            interpolator = interp1d(x[valid_indices], signal_copy[valid_indices], 
                                   kind='linear', fill_value='extrapolate')
            signal_copy = interpolator(x)
    
    if np.any(np.isnan(signal_copy)):
        signal_copy = np.nan_to_num(signal_copy, nan=np.nanmean(signal_copy))
    
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    try:
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        filtered_signal = scipy_signal.filtfilt(b, a, signal_copy)
    except:
        b, a = scipy_signal.butter(4, high, btype='low')
        filtered_signal = scipy_signal.filtfilt(b, a, signal_copy)
    
    # Robust outlier handling: softly clip extreme values instead of
    # flattening them to the mean. This preserves the overall shape and
    # variance of the signal while removing implausible spikes.
    mean = np.mean(filtered_signal)
    std = np.std(filtered_signal)
    if std > 0:
        limit = 5 * std
        diff = filtered_signal - mean
        filtered_signal = mean + np.clip(diff, -limit, limit)
    
    return filtered_signal


def preprocess_dataset(signal_data):
    """Apply preprocessing to entire dataset."""
    preprocessed_data = []
    
    for item in signal_data:
        df = item['df'].copy()
        sensor_cols = item['sensor_cols']
        
        for col in sensor_cols:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            df[f'{col}_preprocessed'] = preprocess_signal(numeric_data.values)
        
        preprocessed_data.append({
            'df': df,
            'sensor_cols': sensor_cols,
            'preprocessed_cols': [f'{col}_preprocessed' for col in sensor_cols],
            'label_col': item['label_col'],
            'participant': item['participant']
        })
    
    return preprocessed_data

