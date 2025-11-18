"""Task 7: Advanced evaluation functions."""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from feature_extraction import extract_features_from_dataset
from modeling import train_classical_models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utils import (
    DEFAULT_SAMPLING_FREQUENCY, detect_platform
)


def evaluate_standard_split(model, X_train, y_train, X_test, y_test):
    """Evaluate scikit-learn model with standard 80/20 split."""
    # Scikit-learn model evaluation
    y_pred = model.predict(X_test)
    y_true = y_test
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred
    }


def evaluate_loso(X, y, groups, model_class=None, model_params=None):
    """
    Evaluate using Leave-One-Subject-Out cross-validation with scikit-learn models.
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Group labels (participant IDs)
        model_class: Model class to use (defaults to RandomForestClassifier)
        model_params: Dictionary of model parameters
    
    Returns:
        Dictionary with LOSO metrics
    """
    platform_info = detect_platform()
    if model_class is None:
        model_class = RandomForestClassifier
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'n_jobs': platform_info["optimal_n_jobs"],
            'random_state': 42
        }
    
    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)
    
    print(f"\nNumber of unique participants: {n_participants}")
    
    if n_participants < 2:
        print("\n⚠️ WARNING: LOSO requires at least 2 participants.")
        print(f"Found only {n_participants} participant(s): {unique_participants}")
        print("\nSkipping LOSO evaluation. Using simplified evaluation instead...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use a simple scikit-learn model for evaluation
        model_cv = model_class(**model_params)
        model_cv.fit(X_scaled, y)
        y_pred = model_cv.predict(X_scaled)
        loso_accuracy = accuracy_score(y, y_pred)
        
        return {
            'accuracy': loso_accuracy,
            'precision': loso_accuracy,
            'recall': loso_accuracy,
            'f1_score': loso_accuracy,
            'mean_accuracy': loso_accuracy,
            'std_accuracy': 0.0,
            'predictions': None,
            'true_labels': None
        }
    
    print(f"Participants: {unique_participants}")
    print(f"Using scikit-learn {model_class.__name__}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Manual LOSO implementation: iterate over unique groups
    unique_groups = np.unique(groups)
    loso_scores = []
    loso_predictions = []
    loso_true_labels = []
    
    print("Running LOSO evaluation (may take a while)...\n")
    
    fold = 1
    for test_group in unique_groups:
        # Split: test on current group, train on all others
        test_mask = groups == test_group
        train_mask = ~test_mask
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        X_train_fold = X_scaled[train_idx]
        X_test_fold = X_scaled[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]
        
        # Train scikit-learn model
        model_fold = model_class(**model_params)
        model_fold.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        y_pred_fold = model_fold.predict(X_test_fold)
        
        acc = accuracy_score(y_test_fold, y_pred_fold)
        loso_scores.append(acc)
        loso_predictions.extend(y_pred_fold)
        loso_true_labels.extend(y_test_fold)
        
        test_participant = groups[test_idx][0]
        print(f"Fold {fold}/{n_participants} (Participant {test_participant}): Accuracy = {acc:.4f}")
        fold += 1
    
    loso_predictions = np.array(loso_predictions)
    loso_true_labels = np.array(loso_true_labels)
    
    loso_accuracy = accuracy_score(loso_true_labels, loso_predictions)
    loso_precision = precision_score(loso_true_labels, loso_predictions, average='macro', zero_division=0)
    loso_recall = recall_score(loso_true_labels, loso_predictions, average='macro', zero_division=0)
    loso_f1 = f1_score(loso_true_labels, loso_predictions, average='macro', zero_division=0)
    
    print("\n=== LOSO OVERALL RESULTS ===")
    print(f"Mean Accuracy: {np.mean(loso_scores):.4f} (+/- {np.std(loso_scores):.4f})")
    print(f"Overall Accuracy: {loso_accuracy:.4f}")
    print(f"Overall Precision (macro): {loso_precision:.4f}")
    print(f"Overall Recall (macro): {loso_recall:.4f}")
    print(f"Overall F1-Score (macro): {loso_f1:.4f}")
    
    return {
        'accuracy': loso_accuracy,
        'precision': loso_precision,
        'recall': loso_recall,
        'f1_score': loso_f1,
        'mean_accuracy': np.mean(loso_scores),
        'std_accuracy': np.std(loso_scores),
        'predictions': loso_predictions,
        'true_labels': loso_true_labels
    }


def compare_window_sizes(preprocessed_data, window_sizes, feature_cols_func, 
                        label_encoder, scaler):
    """
    Compare performance across different window sizes.
    
    Args:
        preprocessed_data: Preprocessed dataset
        window_sizes: List of window sizes to test
        feature_cols_func: Function to get feature columns from features_df
        label_encoder: Fitted LabelEncoder
        scaler: Fitted StandardScaler
    
    Returns:
        DataFrame with window size comparison results
    """
    platform_info = detect_platform()
    
    window_results = []
    
    for ws in window_sizes:
        print(f"Testing window size: {ws} samples ({ws/50:.2f} seconds)")
        
        features_ws = extract_features_from_dataset(preprocessed_data, window_size=ws, overlap=0.5)
        
        X_ws = features_ws[feature_cols_func(features_ws)].fillna(0).values
        y_ws = label_encoder.fit_transform(features_ws['label'])
        
        X_ws_scaled = scaler.fit_transform(X_ws)
        
        X_train_ws, X_test_ws, y_train_ws, y_test_ws = train_test_split(
            X_ws_scaled, y_ws, test_size=0.2, random_state=42, stratify=y_ws
        )
        
        # Use scikit-learn RandomForest model with class weighting for imbalanced data
        rf_ws = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=platform_info["optimal_n_jobs"],
            random_state=42
        )
        rf_ws.fit(X_train_ws, y_train_ws)
        
        y_pred_ws = rf_ws.predict(X_test_ws)
        
        acc = accuracy_score(y_test_ws, y_pred_ws)
        prec = precision_score(y_test_ws, y_pred_ws, average='macro', zero_division=0)
        rec = recall_score(y_test_ws, y_pred_ws, average='macro', zero_division=0)
        f1 = f1_score(y_test_ws, y_pred_ws, average='macro', zero_division=0)
        
        window_results.append({
            'Window Size': ws,
            'Duration (s)': ws / DEFAULT_SAMPLING_FREQUENCY,
            'N Samples': len(X_ws),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, N={len(X_ws)}\n")
    
    return pd.DataFrame(window_results)


def generate_confusion_matrix(y_true, y_pred, class_names):
    """Generate confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(class_names) > len(unique_classes):
        class_names_filtered = [class_names[i] for i in unique_classes]
    else:
        class_names_filtered = class_names
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm, cm_normalized, class_names_filtered


def generate_classification_report(y_true, y_pred, class_names):
    """Generate classification report."""
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(class_names) > len(unique_classes):
        target_names = [class_names[i] for i in unique_classes]
    else:
        target_names = class_names
    
    report = classification_report(
        y_true, y_pred,
        labels=unique_classes,
        target_names=target_names,
        zero_division=0
    )
    
    return report

