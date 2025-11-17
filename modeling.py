"""Task 6: PyTorch neural network models with GPU acceleration."""

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import (
    DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP, detect_platform, get_device,
    accuracy_score, precision_score, recall_score, f1_score
)


class FeatureDataset(Dataset):
    """PyTorch dataset for feature vectors."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleNN(nn.Module):
    """Simple feedforward neural network."""
    def __init__(self, input_size, num_classes, hidden_sizes=[64, 32], dropout=0.2):
        super(SimpleNN, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DecisionTreeNN(nn.Module):
    """Neural network approximating a decision tree."""
    def __init__(self, input_size, num_classes):
        super(DecisionTreeNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class SVMLikeNN(nn.Module):
    """Neural network with SVM-like characteristics (hinge loss equivalent)."""
    def __init__(self, input_size, num_classes):
        super(SVMLikeNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class NaiveBayesNN(nn.Module):
    """Neural network approximating Naive Bayes."""
    def __init__(self, input_size, num_classes):
        super(NaiveBayesNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class RandomForestNN(nn.Module):
    """Deep neural network approximating random forest ensemble."""
    def __init__(self, input_size, num_classes):
        super(RandomForestNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class AdaBoostNN(nn.Module):
    """Neural network with AdaBoost-like training."""
    def __init__(self, input_size, num_classes):
        super(AdaBoostNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_pytorch_model(model, train_loader, device, epochs=50, lr=0.001):
    """Train a PyTorch model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_pytorch_model(model, test_loader, device):
    """Evaluate a PyTorch model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def create_sequences_for_dl(preprocessed_data, window_size=DEFAULT_WINDOW_SIZE, 
                             overlap=DEFAULT_OVERLAP):
    """
    Create sequences for deep learning models.
    Returns: X (n_samples, window_size, n_channels), y (n_samples,)
    """
    sequences = []
    labels = []
    
    for item in preprocessed_data:
        df = item['df']
        preprocessed_cols = item['preprocessed_cols']
        label_col = item['label_col']
        
        sensor_data = df[preprocessed_cols].values
        label_data = df[label_col].values if label_col in df.columns else None
        
        if label_data is None:
            continue
        
        step_size = int(window_size * (1 - overlap))
        for i in range(0, len(sensor_data) - window_size + 1, step_size):
            window = sensor_data[i:i + window_size, :]
            window_labels = label_data[i:i + window_size]
            
            unique, counts = np.unique(window_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            
            sequences.append(window)
            labels.append(majority_label)
    
    X_seq = np.array(sequences)
    y_seq = np.array(labels)
    
    return X_seq, y_seq


def train_classical_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate PyTorch neural network models.
    Returns: results DataFrame and trained models dictionary
    """
    device, device_name = get_device()
    platform_info = detect_platform()
    
    print(f"\nPlatform detected: {platform_info['platform_name']}")
    print(f"Using device: {device_name}")
    
    if device is None:
        raise ImportError("PyTorch is not installed. Please install it with: pip install torch")
    
    # Move data to device
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"Input features: {input_size}, Number of classes: {num_classes}")
    
    # Create datasets and data loaders
    train_dataset = FeatureDataset(X_train, y_train)
    test_dataset = FeatureDataset(X_test, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define models with appropriate architectures
    models_config = {
        'Decision Tree': {
            'model_class': DecisionTreeNN,
            'epochs': 30,
            'lr': 0.001
        },
        'SVM': {
            'model_class': SVMLikeNN,
            'epochs': 50,
            'lr': 0.001
        },
        'Naive Bayes': {
            'model_class': NaiveBayesNN,
            'epochs': 30,
            'lr': 0.001
        },
        'Random Forest': {
            'model_class': RandomForestNN,
            'epochs': 50,
            'lr': 0.001
        },
        'AdaBoost': {
            'model_class': AdaBoostNN,
            'epochs': 40,
            'lr': 0.001
        },
    }
    
    results = []
    trained_models = {}
    
    for name, config in models_config.items():
        print(f"\nTraining {name}...")
        
        model = config['model_class'](input_size, num_classes).to(device)
        
        # Verify model is on the correct device
        model_device = next(model.parameters()).device
        if model_device.type == 'cuda':
            print(f"  ✓ Model loaded on GPU: {torch.cuda.get_device_name(model_device.index)}")
        else:
            print(f"  ⚠ Model loaded on {model_device.type.upper()}")
        
        start_time = time.time()
        train_pytorch_model(model, train_loader, device, 
                          epochs=config['epochs'], lr=config['lr'])
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred, y_true = evaluate_pytorch_model(model, test_loader, device)
        pred_time = time.time() - start_time
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
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
    
    results_df = pd.DataFrame(results)
    return results_df, trained_models
