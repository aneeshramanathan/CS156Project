"""Task 3: Signal preprocessing functions."""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from src.utils import DEFAULT_HIGHCUT, DEFAULT_LOWCUT, DEFAULT_SAMPLING_FREQUENCY


TIME_LIKE_TOKENS = ('time', 'timestamp', 'seconds_elapsed')
LABEL_LIKE_TOKENS = ('activity', 'label', 'annotation')


def _should_use_sensor_column(col_name: str) -> bool:
    """Filter out metadata/time columns that masquerade as sensors."""
    lowered = col_name.lower()
    if any(token in lowered for token in TIME_LIKE_TOKENS):
        return False
    if any(token in lowered for token in LABEL_LIKE_TOKENS):
        return False
    return True


def preprocess_signal(signal_array, fs=DEFAULT_SAMPLING_FREQUENCY, 
                      lowcut=DEFAULT_LOWCUT, highcut=DEFAULT_HIGHCUT, 
                      interpolate_missing=True):
    """Preprocess sensor signals with filtering and interpolation."""
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
    
    # Softly clip extreme values to preserve signal shape while removing spikes
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
        raw_sensor_cols = item['sensor_cols']
        sensor_cols = [col for col in raw_sensor_cols if _should_use_sensor_column(col)]
        
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


def normalize_sequence_for_dl(sequence, method='standardize'):
    sequence = np.asarray(sequence)
    original_shape = sequence.shape
    sequence_flat = sequence.flatten() if sequence.ndim > 1 else sequence
    
    if method == 'standardize':
        mean = np.mean(sequence_flat)
        std = np.std(sequence_flat)
        if std < 1e-8:
            std = 1.0
        normalized = (sequence - mean) / std
        params = {'mean': mean, 'std': std, 'method': 'standardize'}
    elif method == 'minmax':
        min_val = np.min(sequence_flat)
        max_val = np.max(sequence_flat)
        if max_val - min_val < 1e-8:
            normalized = sequence - min_val
        else:
            normalized = (sequence - min_val) / (max_val - min_val)
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def normalize_sequences_for_dl(sequences, method='standardize', fit_on_train=True, 
                               normalization_params=None):
    if isinstance(sequences, list):
        all_sequences = np.concatenate([np.asarray(seq).flatten() for seq in sequences])
    else:
        all_sequences = np.asarray(sequences).flatten()
    
    if fit_on_train:
        if method == 'standardize':
            mean = np.mean(all_sequences)
            std = np.std(all_sequences)
            if std < 1e-8:
                std = 1.0
            normalization_params = {'mean': mean, 'std': std, 'method': 'standardize'}
        elif method == 'minmax':
            min_val = np.min(all_sequences)
            max_val = np.max(all_sequences)
            normalization_params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    
    normalized_sequences = []
    for seq in (sequences if isinstance(sequences, list) else [sequences]):
        seq = np.asarray(seq)
        if normalization_params['method'] == 'standardize':
            normalized = (seq - normalization_params['mean']) / normalization_params['std']
        else:  # minmax
            min_val = normalization_params['min']
            max_val = normalization_params['max']
            if max_val - min_val < 1e-8:
                normalized = seq - min_val
            else:
                normalized = (seq - min_val) / (max_val - min_val)
        normalized_sequences.append(normalized)
    
    if not isinstance(sequences, list):
        normalized_sequences = normalized_sequences[0]
    
    return normalized_sequences, normalization_params


def prepare_dl_input(sequences, normalize=True, method='standardize'):
    sequences = np.asarray(sequences)
    params = None
    
    if normalize:
        # Normalize each feature channel independently
        if sequences.ndim == 3:  # (n_samples, seq_len, n_features)
            n_features = sequences.shape[2]
            normalized_seqs = np.zeros_like(sequences)
            feature_params = []
            
            for feat_idx in range(n_features):
                feat_data = sequences[:, :, feat_idx]
                norm_feat, feat_param = normalize_sequence_for_dl(feat_data, method=method)
                normalized_seqs[:, :, feat_idx] = norm_feat
                feature_params.append(feat_param)
            
            params = {'feature_params': feature_params, 'method': method}
            sequences = normalized_seqs
        else:
            sequences, params = normalize_sequences_for_dl(sequences, method=method)
    
    return sequences, params
