"""Task 5: Feature extraction functions."""

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import DEFAULT_SAMPLING_FREQUENCY, DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP, detect_platform


def extract_time_domain_features(window):
    """Extract time-domain features from a signal window."""
    features = {}
    
    features['mean'] = np.mean(window)
    features['std'] = np.std(window)
    features['rms'] = np.sqrt(np.mean(window**2))
    
    zero_crossings = np.sum(np.diff(np.sign(window)) != 0)
    features['zcr'] = zero_crossings / len(window)
    
    features['min'] = np.min(window)
    features['max'] = np.max(window)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(window)
    features['skewness'] = pd.Series(window).skew()
    features['kurtosis'] = pd.Series(window).kurtosis()
    
    features['peak_to_peak'] = np.ptp(window)
    features['mad'] = np.mean(np.abs(window - features['mean']))
    
    return features


def extract_frequency_domain_features(window, fs=DEFAULT_SAMPLING_FREQUENCY):
    """Extract frequency-domain features from a signal window."""
    features = {}
    
    N = len(window)
    fft_vals = fft(window)
    fft_mag = np.abs(fft_vals[:N//2])
    fft_freq = fftfreq(N, 1/fs)[:N//2]
    
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
        features['spectral_entropy'] = entropy(psd_norm) if len(psd_norm) > 0 else 0.0
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
    """Extract all features from a window."""
    time_features = extract_time_domain_features(window)
    freq_features = extract_frequency_domain_features(window, fs)
    all_features = {**time_features, **freq_features}
    return all_features


def _extract_features_from_sensor_column(item, col, window_size, overlap):
    """Helper function to extract features from a single sensor column (for parallelization)."""
    from windowing import create_windows
    
    df = item['df']
    label_col = item['label_col']
    participant = item['participant']
    
    signal = df[col].values
    labels = df[label_col].values if label_col in df.columns else None
    
    if labels is None:
        return []
    
    step_size = int(window_size * (1 - overlap))
    results = []
    
    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        window_labels = labels[i:i + window_size]
        
        unique, counts = np.unique(window_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        
        features = extract_all_features(window)
        features['sensor_channel'] = col
        
        results.append({
            'features': features,
            'label': majority_label,
            'participant': participant
        })
    
    return results


def extract_features_from_dataset(preprocessed_data, window_size=DEFAULT_WINDOW_SIZE, 
                                   overlap=DEFAULT_OVERLAP, use_multithreading=True):
    """
    Extract features from entire dataset with optional multithreading.
    
    Args:
        preprocessed_data: List of preprocessed data items
        window_size: Size of sliding windows
        overlap: Overlap ratio between windows
        use_multithreading: Whether to use multithreading (default: True)
    """
    platform_info = detect_platform()
    
    if use_multithreading:
        cpu_count = platform_info['cpu_count']
        max_workers = max(1, cpu_count - 1)  # Use all cores except 1 to leave one for system
        
        print(f"Using multithreading with {max_workers} workers for feature extraction...")
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for item in preprocessed_data:
                preprocessed_cols = item['preprocessed_cols']
                for col in preprocessed_cols:
                    future = executor.submit(_extract_features_from_sensor_column, 
                                           item, col, window_size, overlap)
                    futures.append(future)
            
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"  Processed {completed}/{total} sensor columns...", end='\r')
        
        print(f"\n  Completed processing all {total} sensor columns")
        
    else:
        all_results = []
        for item in preprocessed_data:
            preprocessed_cols = item['preprocessed_cols']
            for col in preprocessed_cols:
                results = _extract_features_from_sensor_column(item, col, window_size, overlap)
                all_results.extend(results)
    
    all_features = [r['features'] for r in all_results]
    all_labels = [r['label'] for r in all_results]
    all_participants = [r['participant'] for r in all_results]
    
    features_df = pd.DataFrame(all_features)
    features_df['label'] = all_labels
    features_df['participant'] = all_participants
    
    return features_df

