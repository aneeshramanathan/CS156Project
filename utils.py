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
        
        # Check for Apple Silicon (MPS) first
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps'), 'MPS (Apple Silicon GPU)'
        
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            
            # Check compute capability and verify GPU actually works
            try:
                compute_cap = torch.cuda.get_device_capability(0)
                compute_cap_str = f"{compute_cap[0]}.{compute_cap[1]}"
                
                # Try to create a test tensor to verify GPU actually works
                # Suppress warnings temporarily during test
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_tensor = torch.zeros(1).cuda()
                    # Try a simple operation to ensure kernels are available
                    _ = test_tensor * 2
                    del test_tensor
                    torch.cuda.empty_cache()
                
                # GPU works! Return it even if there's a compute capability warning
                return torch.device('cuda'), f'CUDA (NVIDIA GPU: {device_name}, CUDA {cuda_version}, Compute {compute_cap_str}, {device_count} device(s))'
            except RuntimeError as e:
                error_msg = str(e)
                # GPU detected but actually not compatible (can't create tensors or no kernels)
                print(f"\n⚠️  ERROR: GPU detected but PyTorch cannot use it.")
                print(f"   GPU: {device_name}")
                if 'compute_cap_str' in locals():
                    print(f"   Compute Capability: {compute_cap_str}")
                print(f"   Error: {error_msg}")
                
                if "no kernel image is available" in error_msg.lower():
                    print(f"\n   This means PyTorch doesn't have compiled kernels for your GPU's compute capability.")
                    print(f"   Your RTX 5070 Ti (compute 12.0) requires PyTorch nightly build.")
                    print(f"\n   SOLUTION: Install PyTorch nightly:")
                    print(f"   pip uninstall torch torchvision torchaudio")
                    print(f"   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121")
                    print(f"\n   Falling back to CPU for now...")
                else:
                    print(f"   Try installing PyTorch nightly: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121")
                
                return torch.device('cpu'), f'CPU (GPU incompatible - {error_msg[:50]}...)'
        else:
            # Provide diagnostic info if CUDA is not available
            cuda_available = torch.cuda.is_available()
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"⚠️  Warning: PyTorch was compiled with CUDA {torch.version.cuda}, but CUDA is not available.")
                print(f"   This might mean:")
                print(f"   1. NVIDIA drivers are not installed or outdated")
                print(f"   2. CUDA toolkit is not properly installed")
                print(f"   3. GPU is not accessible")
            else:
                print(f"⚠️  Warning: PyTorch was installed without CUDA support (CPU-only version).")
                print(f"   To use GPU with CUDA 13, install PyTorch with CUDA 12.1 (compatible):")
                print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                print(f"   Or for CUDA 12.4: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            return torch.device('cpu'), 'CPU'
    except ImportError:
        return None, 'CPU (PyTorch not installed)'


def ensure_visualizations_dir():
    """Create visualizations directory if it doesn't exist."""
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    return VISUALIZATIONS_DIR


# ============================================================================
# NumPy-based replacements for scikit-learn utilities
# ============================================================================

def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    """
    Split arrays into random train and test subsets (NumPy implementation).
    
    Args:
        X: Feature array
        y: Target array
        test_size: Proportion of dataset to include in test split
        random_state: Random seed
        stratify: Array of labels for stratified splitting
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    if stratify is not None:
        # Stratified split: maintain class distribution
        unique_classes, class_indices = np.unique(stratify, return_inverse=True)
        train_indices = []
        test_indices = []
        
        for class_idx in range(len(unique_classes)):
            class_mask = class_indices == class_idx
            class_indices_list = np.where(class_mask)[0]
            
            n_class_test = int(len(class_indices_list) * test_size)
            np.random.shuffle(class_indices_list)
            
            test_indices.extend(class_indices_list[:n_class_test])
            train_indices.extend(class_indices_list[n_class_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    else:
        # Random split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


class StandardScaler:
    """StandardScaler replacement using NumPy."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.scale_ = None
    
    def fit(self, X):
        """Compute mean and std for later scaling."""
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Handle zero variance features
        self.scale_ = np.where(self.std_ > 0, self.std_, 1.0)
        return self
    
    def transform(self, X):
        """Perform standardization by centering and scaling."""
        X = np.asarray(X)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("This StandardScaler instance is not fitted yet.")
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)


class LabelEncoder:
    """LabelEncoder replacement using NumPy."""
    
    def __init__(self):
        self.classes_ = None
    
    def fit(self, y):
        """Fit label encoder."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        return self
    
    def transform(self, y):
        """Transform labels to normalized encoding."""
        if self.classes_ is None:
            raise ValueError("This LabelEncoder instance is not fitted yet.")
        y = np.asarray(y)
        # Create mapping from class to index
        mapping = {cls: idx for idx, cls in enumerate(self.classes_)}
        return np.array([mapping[val] for val in y])
    
    def fit_transform(self, y):
        """Fit label encoder and return encoded labels."""
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        """Transform labels back to original encoding."""
        if self.classes_ is None:
            raise ValueError("This LabelEncoder instance is not fitted yet.")
        y = np.asarray(y)
        return np.array([self.classes_[idx] for idx in y])


# ============================================================================
# NumPy-based classification metrics
# ============================================================================

def accuracy_score(y_true, y_pred):
    """Compute accuracy classification score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='macro', zero_division=0, labels=None):
    """
    Compute precision score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'macro' for macro-averaged precision, None for per-class scores
        zero_division: Value to return when there is a zero division
        labels: Optional list of labels to include (for per-class scores)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    
    precisions = []
    
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        
        if tp + fp == 0:
            precisions.append(zero_division)
        else:
            precisions.append(tp / (tp + fp))
    
    if average == 'macro':
        return np.mean(precisions) if precisions else zero_division
    else:
        return np.array(precisions)


def recall_score(y_true, y_pred, average='macro', zero_division=0, labels=None):
    """
    Compute recall score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'macro' for macro-averaged recall, None for per-class scores
        zero_division: Value to return when there is a zero division
        labels: Optional list of labels to include (for per-class scores)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    
    recalls = []
    
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        
        if tp + fn == 0:
            recalls.append(zero_division)
        else:
            recalls.append(tp / (tp + fn))
    
    if average == 'macro':
        return np.mean(recalls) if recalls else zero_division
    else:
        return np.array(recalls)


def f1_score(y_true, y_pred, average='macro', zero_division=0, labels=None):
    """
    Compute F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'macro' for macro-averaged F1, None for per-class scores
        zero_division: Value to return when there is a zero division
        labels: Optional list of labels to include (for per-class scores)
    """
    prec = precision_score(y_true, y_pred, average=average, zero_division=zero_division, labels=labels)
    rec = recall_score(y_true, y_pred, average=average, zero_division=zero_division, labels=labels)
    
    if average == 'macro':
        if prec + rec == 0:
            return zero_division
        return 2 * (prec * rec) / (prec + rec)
    else:
        # Return per-class F1 scores
        f1_scores = []
        for p, r in zip(prec, rec):
            if p + r == 0:
                f1_scores.append(zero_division)
            else:
                f1_scores.append(2 * (p * r) / (p + r))
        return np.array(f1_scores)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of label indices to include
    
    Returns:
        Confusion matrix as 2D numpy array
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    # Create mapping from label to index
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1
    
    return cm


def classification_report(y_true, y_pred, labels=None, target_names=None, zero_division=0):
    """
    Build a text report showing the main classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of label indices to include
        target_names: Optional list of label names
        zero_division: Value to return when there is a zero division
    
    Returns:
        String report
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    
    if target_names is None:
        target_names = [f'Class {i}' for i in labels]
    elif len(target_names) != len(labels):
        # Filter target_names to match labels
        target_names = [target_names[i] if i < len(target_names) else f'Class {i}' 
                       for i in labels]
    
    # Calculate per-class metrics
    precisions = precision_score(y_true, y_pred, average=None, zero_division=zero_division, labels=labels)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=zero_division, labels=labels)
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=zero_division, labels=labels)
    
    # Calculate support (number of true instances per class)
    supports = []
    for label in labels:
        supports.append(np.sum(y_true == label))
    
    # Build report
    report_lines = []
    report_lines.append("              precision    recall  f1-score   support\n")
    report_lines.append("\n")
    
    for i, label in enumerate(labels):
        label_name = target_names[i] if i < len(target_names) else f'Class {label}'
        prec = precisions[i] if i < len(precisions) else zero_division
        rec = recalls[i] if i < len(recalls) else zero_division
        f1 = f1_scores[i] if i < len(f1_scores) else zero_division
        support = supports[i]
        
        report_lines.append(f"{label_name:>15}        {prec:.2f}      {rec:.2f}      {f1:.2f}        {support}\n")
    
    # Macro averages
    macro_prec = np.mean(precisions) if len(precisions) > 0 else zero_division
    macro_rec = np.mean(recalls) if len(recalls) > 0 else zero_division
    macro_f1 = np.mean(f1_scores) if len(f1_scores) > 0 else zero_division
    total_support = np.sum(supports)
    
    report_lines.append("\n")
    report_lines.append(f"{'    avg / total':>15}        {macro_prec:.2f}      {macro_rec:.2f}      {macro_f1:.2f}        {total_support}\n")
    
    return ''.join(report_lines)

