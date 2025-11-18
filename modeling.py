"""Task 6: Classical ML models using scikit-learn.

This module implements classical ML models using scikit-learn:
Decision Tree, SVM, Naive Bayes, Random Forest, AdaBoost, and XGBoost.

These models perform per-window activity classification: given hand-crafted
features from a single sensor window, they predict the **current** activity
for that window (not a next-activity or sequence prediction task).
"""

import numpy as np
import pandas as pd
import time
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import (
    DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP, detect_platform
)




def train_classical_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate classical ML models using scikit-learn (Task 6).
    Implements Decision Tree, SVM, Naive Bayes, Random Forest, AdaBoost, and XGBoost.
    Returns: results DataFrame and trained models dictionary.
    """
    platform_info = detect_platform()
    
    print(f"\nPlatform detected: {platform_info['platform_name']}")
    print(f"Using CPU-based scikit-learn models for classical ML algorithms")
    
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"Input features: {input_size}, Number of classes: {num_classes}")
    
    # Compute class weights to handle imbalanced dataset (7 cycling, 3 sitting, 2 walking)
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    # Define scikit-learn models with appropriate hyperparameters
    models_config = {
        'Decision Tree': {
            'model_class': DecisionTreeClassifier,
            'params': {
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': class_weight_dict,
                'random_state': 42
            }
        },
        'SVM': {
            'model_class': LinearSVC,
            'params': {
                'C': 1.0,
                'max_iter': 2000,
                'class_weight': class_weight_dict,
                'random_state': 42,
                'dual': False  # Faster for n_samples > n_features
            }
        },
        'Naive Bayes': {
            'model_class': GaussianNB,
            'params': {}
        },
        'Random Forest': {
            'model_class': RandomForestClassifier,
            'params': {
                'n_estimators': 300,
                'max_depth': None,  # No depth limit - let min_samples_* regularize
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': class_weight_dict,
                'n_jobs': platform_info["optimal_n_jobs"],
                'random_state': 42
            }
        },
        'AdaBoost': {
            'model_class': AdaBoostClassifier,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'class_weight': class_weight_dict
        },
    }
    
    results = []
    trained_models = {}
    
    # ---------------------------------------------------------------------
    # Scikit-learn classical models
    # Note: Using scikit-learn for classical ML algorithms (Decision Tree, SVM, etc.)
    # is faster and more accurate than neural network approximations. These algorithms
    # are optimized C/Cython implementations that solve the problem directly rather than
    # using iterative gradient descent.
    # ---------------------------------------------------------------------
    for name, config in models_config.items():
        print(f"\nTraining {name}...")
        
        model = config['model_class'](**config['params'])
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Train Time (s)': train_time,
            'Pred Time (s)': pred_time
        })
        
        trained_models[name] = model
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Prediction time: {pred_time:.4f}s")

    # ---------------------------------------------------------------------
    # XGBoost model using the same features
    # ---------------------------------------------------------------------
    print("\nTraining XGBoost (true gradient-boosted trees)...")
    start_time = time.time()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=platform_info["optimal_n_jobs"],
        random_state=42,
    )
    # Apply class weights as sample weights for XGBoost
    sample_weights = np.array([class_weight_dict[y] for y in y_train])
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = xgb.predict(X_test)
    pred_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append({
        "Model": "XGBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Train Time (s)": train_time,
        "Pred Time (s)": pred_time,
    })

    trained_models["XGBoost"] = xgb  # type: ignore

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Prediction time: {pred_time:.4f}s")
    
    results_df = pd.DataFrame(results)
    return results_df, trained_models
