"""Task 5: Feature extraction functions."""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.utils import DEFAULT_OVERLAP, DEFAULT_SAMPLING_FREQUENCY, DEFAULT_WINDOW_SIZE, detect_platform

# Minimum fraction of majority label required in a window (drops ambiguous transition windows)
MIN_LABEL_FRACTION = 0.8

AXIS_TOKENS = {'x', 'y', 'z'}
SENSOR_HINTS = (
    'acc', 'accelerometer', 'gyr', 'gyro', 'gyroscope', 'mag', 'magnetometer',
    'gravity', 'totacc', 'speed', 'velocity'
)


def extract_time_domain_features(window):
    features = {}
    
    window = np.asarray(window)
    mean = np.mean(window)
    std = np.std(window)
    
    features['mean'] = mean
    features['std'] = std
    features['rms'] = np.sqrt(np.mean(window**2))
    
    zero_crossings = np.sum(np.diff(np.sign(window)) != 0)
    features['zcr'] = zero_crossings / len(window)
    
    features['min'] = np.min(window)
    features['max'] = np.max(window)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(window)
    
    if std > 0:
        centered = (window - mean) / std
        features['skewness'] = np.mean(centered**3)
        features['kurtosis'] = np.mean(centered**4) - 3.0
    else:
        features['skewness'] = 0.0
        features['kurtosis'] = 0.0
    
    features['peak_to_peak'] = np.ptp(window)
    features['mad'] = np.mean(np.abs(window - mean))
    
    return features


def _entropy(pk):
    pk = np.asarray(pk)
    pk = pk[pk > 0]
    if len(pk) == 0:
        return 0.0
    return -np.sum(pk * np.log(pk))


def extract_frequency_domain_features(window, fs=DEFAULT_SAMPLING_FREQUENCY):
    features = {}
    
    N = len(window)
    fft_vals = np.fft.fft(window)
    fft_mag = np.abs(fft_vals[:N//2])
    fft_freq = np.fft.fftfreq(N, 1/fs)[:N//2]
    
    if len(fft_mag) > 0:
        dominant_freq_idx = np.argmax(fft_mag)
        features['dominant_freq'] = fft_freq[dominant_freq_idx]
    else:
        features['dominant_freq'] = 0.0
    
    features['spectral_energy'] = np.sum(fft_mag**2)
    
    psd = fft_mag**2
    psd_sum = np.sum(psd)
    if psd_sum > 0:
        psd_norm = psd / psd_sum
        psd_norm = psd_norm[psd_norm > 0]
        features['spectral_entropy'] = _entropy(psd_norm) if len(psd_norm) > 0 else 0.0
    else:
        features['spectral_entropy'] = 0.0
    
    fft_sum = np.sum(fft_mag)
    if fft_sum > 0 and len(fft_mag) > 0:
        features['spectral_centroid'] = np.sum(fft_freq * fft_mag) / fft_sum
        features['spectral_spread'] = np.sqrt(np.sum(((fft_freq - features['spectral_centroid'])**2) * fft_mag) / fft_sum)
        
        cumsum_fft = np.cumsum(fft_mag)
        threshold = 0.85 * fft_sum
        rolloff_indices = np.where(cumsum_fft >= threshold)[0]
        if len(rolloff_indices) > 0:
            features['spectral_rolloff'] = fft_freq[rolloff_indices[0]]
        else:
            features['spectral_rolloff'] = fft_freq[-1] if len(fft_freq) > 0 else 0.0
    else:
        features['spectral_centroid'] = 0.0
        features['spectral_spread'] = 0.0
        features['spectral_rolloff'] = 0.0
    
    return features


def extract_all_features(window, fs=DEFAULT_SAMPLING_FREQUENCY):
    time_features = extract_time_domain_features(window)
    freq_features = extract_frequency_domain_features(window, fs)
    all_features = {**time_features, **freq_features}
    return all_features


def _sanitize_feature_prefix(name: str) -> str:
    sanitized = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in str(name))
    sanitized = sanitized.strip('_').lower()
    return sanitized or 'sensor'


def _tokenize_name(name: str):
    return [tok for tok in name.replace('-', '_').split('_') if tok]


def _infer_group_and_axis(col_name: str):
    base_name = col_name.replace('_preprocessed', '')
    tokens = _tokenize_name(base_name.lower())
    
    axis = None
    for tok in tokens:
        if tok in AXIS_TOKENS:
            axis = tok
            break
    if axis is None:
        axis = 'scalar'
    
    sensor_tokens = [tok for tok in tokens if tok != axis]
    sensor_hint = next((tok for tok in reversed(sensor_tokens) if tok in SENSOR_HINTS), None)
    if sensor_hint:
        group = sensor_hint
    elif sensor_tokens:
        group = sensor_tokens[-1]
    else:
        group = base_name.lower()
    
    return group, axis, base_name


def _group_sensor_columns(preprocessed_cols):
    groups = {}
    for col in preprocessed_cols:
        group_hint, axis_hint, _ = _infer_group_and_axis(col)
        group_key = _sanitize_feature_prefix(group_hint)
        axis_key = _sanitize_feature_prefix(axis_hint)
        
        if not axis_key:
            axis_key = 'axis'
        
        group = groups.setdefault(group_key, [])
        axis_label = axis_key
        existing_axes = {axis for axis, _ in group}
        counter = 2
        while axis_label in existing_axes:
            axis_label = f"{axis_key}{counter}"
            counter += 1
        
        group.append((axis_label, col))
    
    return groups


def _add_prefixed_features(target_dict, prefix, features_dict):
    for key, value in features_dict.items():
        target_dict[f"{prefix}_{key}"] = float(value)


def _extract_features_from_sensor_group(signals_dict, labels, group_name, participant,
                                        window_size, overlap, fs=DEFAULT_SAMPLING_FREQUENCY):
    """Extract features from a multi-axis sensor group within sliding windows."""
    if labels is None or len(labels) == 0 or not signals_dict:
        return []
    
    sanitized_group = _sanitize_feature_prefix(group_name)
    axis_names = list(signals_dict.keys())
    if not axis_names:
        return []
    
    label_array = np.asarray(labels)
    axis_arrays = {axis: np.asarray(signal) for axis, signal in signals_dict.items()}
    series_length = min(len(label_array), *(len(arr) for arr in axis_arrays.values()))
    if series_length < window_size:
        return []
    
    step_size = max(1, int(window_size * (1 - overlap)))
    results = []
    
    for start in range(0, series_length - window_size + 1, step_size):
        end = start + window_size
        window_labels = label_array[start:end]
        
        unique, counts = np.unique(window_labels, return_counts=True)
        if len(counts) == 0:
            continue
        max_count = counts.max()
        if max_count / len(window_labels) < MIN_LABEL_FRACTION:
            continue
        majority_label = unique[np.argmax(counts)]
        
        features = {}
        axis_windows = {axis: axis_arrays[axis][start:end] for axis in axis_names}
        
        for axis in axis_names:
            axis_features = extract_all_features(axis_windows[axis], fs)
            _add_prefixed_features(features, f"{sanitized_group}_{axis}", axis_features)
        
        if len(axis_names) > 1:
            window_matrix = np.column_stack([axis_windows[axis] for axis in axis_names])
            magnitude = np.linalg.norm(window_matrix, axis=1)
            mag_features = extract_all_features(magnitude, fs)
            _add_prefixed_features(features, f"{sanitized_group}_mag", mag_features)
            
            jerk = np.diff(window_matrix, axis=0)
            if jerk.size > 0:
                jerk_mag = np.linalg.norm(jerk, axis=1)
                jerk_features = extract_time_domain_features(jerk_mag)
                _add_prefixed_features(features, f"{sanitized_group}_jerk", jerk_features)
                features[f"{sanitized_group}_jerk_energy"] = float(np.sum(jerk_mag ** 2))
            
            corr_matrix = np.corrcoef(window_matrix, rowvar=False)
            if corr_matrix.ndim == 2:
                for i in range(len(axis_names)):
                    for j in range(i + 1, len(axis_names)):
                        corr_val = corr_matrix[i, j]
                        if np.isnan(corr_val):
                            corr_val = 0.0
                        features[f"{sanitized_group}_corr_{axis_names[i]}_{axis_names[j]}"] = float(corr_val)
            
            energy_per_axis = np.sum(window_matrix ** 2, axis=0)
            total_energy = float(np.sum(energy_per_axis)) + 1e-12
            for axis, energy in zip(axis_names, energy_per_axis):
                features[f"{sanitized_group}_{axis}_energy_frac"] = float(energy / total_energy)
        
        features['sensor_channel'] = sanitized_group
        results.append({
            'features': features,
            'label': majority_label,
            'participant': participant
        })
    
    return results


def extract_features_from_dataset(preprocessed_data, window_size=DEFAULT_WINDOW_SIZE, 
                                   overlap=DEFAULT_OVERLAP, use_multiprocessing=True):
    platform_info = detect_platform()
    
    if use_multiprocessing:
        cpu_count = platform_info['cpu_count']
        max_workers = max(1, cpu_count - 1)
        
        print(f"Using multiprocessing with {max_workers} workers for feature extraction...")
        
        all_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for item in preprocessed_data:
                df = item['df']
                label_col = item['label_col']
                participant = item['participant']
                preprocessed_cols = item['preprocessed_cols']
                
                labels = df[label_col].values if label_col in df.columns else None
                grouped_cols = _group_sensor_columns(preprocessed_cols)
                
                for group_name, axis_cols in grouped_cols.items():
                    signals_dict = {}
                    for axis_label, col in axis_cols:
                        if col in df.columns:
                            signals_dict[axis_label] = df[col].values
                    if not signals_dict:
                        continue
                    future = executor.submit(
                        _extract_features_from_sensor_group,
                        signals_dict,
                        labels,
                        group_name,
                        participant,
                        window_size,
                        overlap,
                        DEFAULT_SAMPLING_FREQUENCY
                    )
                    futures.append(future)
            
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"  Processed {completed}/{total} sensor groups...", end='\r')
        
        print(f"\n  Completed processing all {total} sensor groups")
        
    else:
        all_results = []
        for item in preprocessed_data:
            df = item['df']
            label_col = item['label_col']
            participant = item['participant']
            preprocessed_cols = item['preprocessed_cols']
            
            labels = df[label_col].values if label_col in df.columns else None
            
            grouped_cols = _group_sensor_columns(preprocessed_cols)
            for group_name, axis_cols in grouped_cols.items():
                signals_dict = {}
                for axis_label, col in axis_cols:
                    if col in df.columns:
                        signals_dict[axis_label] = df[col].values
                if not signals_dict:
                    continue
                results = _extract_features_from_sensor_group(
                    signals_dict,
                    labels,
                    group_name,
                    participant,
                    window_size,
                    overlap,
                    DEFAULT_SAMPLING_FREQUENCY
                )
                all_results.extend(results)
    
    all_features = [r['features'] for r in all_results]
    all_labels = [r['label'] for r in all_results]
    all_participants = [r['participant'] for r in all_results]
    
    features_df = pd.DataFrame(all_features)
    features_df['label'] = all_labels
    features_df['participant'] = all_participants
    
    if 'sensor_channel' in features_df.columns:
        unique_channels = sorted(features_df['sensor_channel'].astype(str).unique())
        channel_to_idx = {ch: idx for idx, ch in enumerate(unique_channels)}
        features_df['sensor_channel_idx'] = features_df['sensor_channel'].map(channel_to_idx).astype(int)
    
    return features_df

