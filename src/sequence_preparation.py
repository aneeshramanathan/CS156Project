import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import prepare_dl_input
from src.utils import DEFAULT_OVERLAP, DEFAULT_WINDOW_SIZE, DEFAULT_SAMPLING_FREQUENCY
from src.feature_extraction import _group_sensor_columns

# Minimum fraction of majority label required in a window
MIN_LABEL_FRACTION = 0.8


def create_sequences_from_preprocessed_data(preprocessed_data, window_size=DEFAULT_WINDOW_SIZE, 
                                           overlap=DEFAULT_OVERLAP, normalize=True):
    # First pass: collect all unique feature names across all participants
    all_feature_names_set = set()
    for item in preprocessed_data:
        preprocessed_cols = item['preprocessed_cols']
        for col in preprocessed_cols:
            feat_name = col.replace('_preprocessed', '')
            all_feature_names_set.add(feat_name)
    
    # Sort for consistent ordering
    all_feature_names = sorted(list(all_feature_names_set))
    n_features = len(all_feature_names)
    
    if n_features == 0:
        raise ValueError("No features found in preprocessed data")
    
    print(f"Found {n_features} unique sensor channels across all participants")
    
    # Create mapping from feature name to index
    feature_to_idx = {name: idx for idx, name in enumerate(all_feature_names)}
    
    all_sequences = []
    all_labels = []
    all_participants = []
    
    # Second pass: create sequences with consistent feature ordering
    for item in preprocessed_data:
        df = item['df']
        label_col = item['label_col']
        participant = item['participant']
        preprocessed_cols = item['preprocessed_cols']
        
        # Get labels
        if label_col not in df.columns:
            continue
        labels = df[label_col].values
        
        # Create feature matrix with consistent ordering
        # Initialize with zeros for missing features
        sensor_matrix = np.zeros((len(df), n_features))
        
        for col in preprocessed_cols:
            if col in df.columns:
                feat_name = col.replace('_preprocessed', '')
                if feat_name in feature_to_idx:
                    feat_idx = feature_to_idx[feat_name]
                    sensor_matrix[:, feat_idx] = df[col].values
        
        min_length = min(len(labels), len(sensor_matrix))
        
        if min_length < window_size:
            continue
        
        # Create sliding windows
        step_size = max(1, int(window_size * (1 - overlap)))
        
        for start in range(0, min_length - window_size + 1, step_size):
            end = start + window_size
            
            # Get window labels
            window_labels = labels[start:end]
            unique, counts = np.unique(window_labels, return_counts=True)
            
            if len(counts) == 0:
                continue
            
            max_count = counts.max()
            if max_count / len(window_labels) < MIN_LABEL_FRACTION:
                continue
            
            majority_label = unique[np.argmax(counts)]
            
            # Extract window sequence: (window_size, n_features)
            window_sequence = sensor_matrix[start:end, :]
            
            all_sequences.append(window_sequence)
            all_labels.append(majority_label)
            all_participants.append(participant)
    
    if not all_sequences:
        raise ValueError("No sequences created from preprocessed data")
    
    # Convert to numpy arrays - now all sequences have the same shape
    X_sequences = np.array(all_sequences)  # (n_samples, window_size, n_features)
    y_sequences = np.array(all_labels)
    participants = np.array(all_participants)
    
    print(f"Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
    
    # Normalize sequences for deep learning
    normalization_params = None
    if normalize:
        X_sequences, normalization_params = prepare_dl_input(X_sequences, normalize=True, method='standardize')
    
    return X_sequences, y_sequences, participants, all_feature_names, normalization_params


def prepare_sequences_for_ann(preprocessed_data, window_size=DEFAULT_WINDOW_SIZE, 
                              overlap=DEFAULT_OVERLAP, test_size=0.2, random_state=42,
                              output_dir=None, normalize=True):
    print("\n" + "="*60)
    print("PREPARING SEQUENCES FOR ANN MODELS")
    print("="*60)
    
    # Create sequences
    print("Creating sequences from preprocessed sensor data...")
    X_sequences, y_sequences, participants, feature_names, norm_params = create_sequences_from_preprocessed_data(
        preprocessed_data, window_size=window_size, overlap=overlap, normalize=normalize
    )
    
    print(f"Created {len(X_sequences)} sequences")
    print(f"Sequence shape: {X_sequences.shape} (n_samples, window_size={window_size}, n_features={X_sequences.shape[2]})")
    print(f"Number of features per timestep: {X_sequences.shape[2]}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_sequences)
    print(f"Classes: {le.classes_}")
    
    # Split by participant to avoid data leakage
    unique_participants = np.unique(participants)
    train_participants, test_participants = train_test_split(
        unique_participants, test_size=test_size, random_state=random_state
    )
    
    train_mask = np.isin(participants, train_participants)
    test_mask = np.isin(participants, test_participants)
    
    X_train = X_sequences[train_mask]
    X_test = X_sequences[test_mask]
    y_train = y_encoded[train_mask]
    y_test = y_encoded[test_mask]
    
    print(f"\nTrain set: {len(X_train)} sequences from {len(train_participants)} participants")
    print(f"Test set: {len(X_test)} sequences from {len(test_participants)} participants")
    
    # Save to .npy files
    if output_dir is None:
        output_dir = Path("data/processed")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "X_seq_train.npy"
    test_path = output_dir / "X_seq_test.npy"
    y_train_path = output_dir / "y_seq_train.npy"
    y_test_path = output_dir / "y_seq_test.npy"
    
    np.save(train_path, X_train)
    np.save(test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    
    print(f"\nSaved sequences to:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {y_train_path}")
    print(f"  - {y_test_path}")
    
    if norm_params:
        print(f"\nNormalization: {norm_params.get('method', 'standardize')}")
        if 'feature_params' in norm_params:
            print(f"  Normalized {len(norm_params['feature_params'])} feature channels independently")
    
    print("\n[OK] Sequence preparation for ANN models complete")
    
    return {
        'X_train_path': str(train_path),
        'X_test_path': str(test_path),
        'y_train_path': str(y_train_path),
        'y_test_path': str(y_test_path),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': le,
        'normalization_params': norm_params,
        'feature_names': feature_names
    }

