"""All visualization and plotting functions."""

# Set non-interactive backend for matplotlib to avoid threading/multiprocessing issues
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import VISUALIZATIONS_DIR, DEFAULT_SAMPLING_FREQUENCY

plt.style.use('default')
sns.set_palette('husl')


def plot_dataset_exploration(data_list, metadata, output_path):
    """Task 1: Plot dataset exploration summary."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    participant_files = {}
    for item in data_list:
        p_id = item['participant']
        participant_files[p_id] = participant_files.get(p_id, 0) + 1
    
    axes[0].bar(range(len(participant_files)), list(participant_files.values()))
    axes[0].set_xlabel('Participant Index')
    axes[0].set_ylabel('Number of Files')
    axes[0].set_title('Files per Participant')
    axes[0].grid(True, alpha=0.3)
    
    if metadata['labels']:
        label_counts = pd.Series(metadata['labels']).value_counts().head(10)
        axes[1].barh(range(len(label_counts)), label_counts.values)
        axes[1].set_yticks(range(len(label_counts)))
        axes[1].set_yticklabels(label_counts.index)
        axes[1].set_xlabel('Count')
        axes[1].set_title('Top 10 Activity Labels')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_annotated_signals(data_list, n_samples, output_path):
    """Task 2: Plot signals around labeled events."""
    signal_data = []
    
    for item in data_list:
        if 'dataframe' in item:
            df = item['dataframe']
            sensor_cols = [col for col in df.columns if any(x in col.lower() for x in ['acc', 'gyr', 'sensor', 'x', 'y', 'z'])]
            label_cols = [col for col in df.columns if any(x in col.lower() for x in ['label', 'activity', 'class'])]
            
            if sensor_cols and label_cols:
                signal_data.append({
                    'df': df,
                    'sensor_cols': sensor_cols,
                    'label_col': label_cols[0],
                    'participant': item['participant']
                })
    
    if not signal_data:
        return signal_data
    
    n_plots = min(n_samples, len(signal_data))
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4*n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, item in enumerate(signal_data[:n_plots]):
        df = item['df']
        sensor_cols = item['sensor_cols'][:3]
        label_col = item['label_col']
        
        for col in sensor_cols:
            axes[idx].plot(df[col].values, label=col, alpha=0.7)
        
        if label_col in df.columns:
            labels = df[label_col].values
            unique_labels = pd.unique(labels)
            
            for i, label in enumerate(unique_labels[:5]):
                mask = labels == label
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    for start, end in zip(indices[::100], indices[1::100]):
                        axes[idx].axvspan(start, end, alpha=0.2, label=f'Activity: {label}' if i < 5 else '')
        
        axes[idx].set_xlabel('Time (samples)')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_title(f'Participant {item["participant"]} - Sensor Signals with Activity Labels')
        axes[idx].legend(loc='upper right', fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return signal_data


def plot_preprocessing_comparison(original_signal, preprocessed_signal, output_path):
    """Task 3: Plot before/after preprocessing comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    axes[0].plot(original_signal, alpha=0.7, color='blue')
    axes[0].set_title('Original Signal (with missing data)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(preprocessed_signal, alpha=0.7, color='green')
    axes[1].set_title('Preprocessed Signal (filtered + interpolated)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    zoom_start = len(original_signal) // 4
    zoom_end = zoom_start + 500
    axes[2].plot(original_signal[zoom_start:zoom_end], alpha=0.7, label='Original', color='blue')
    axes[2].plot(preprocessed_signal[zoom_start:zoom_end], alpha=0.7, label='Preprocessed', color='green')
    axes[2].set_title('Comparison (Zoomed)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (samples)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_windowing_strategies(window_analysis_df, sample_signal, output_path):
    """Task 4: Plot windowing strategy analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(window_analysis_df['window_size'], window_analysis_df['n_windows'], 
                    'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Window Size (samples)', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Windows', fontweight='bold')
    axes[0, 0].set_title('Window Count vs Window Size', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(window_analysis_df['window_size'], window_analysis_df['time_duration_sec'], 
                    'o-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('Window Size (samples)', fontweight='bold')
    axes[0, 1].set_ylabel('Time Duration (seconds)', fontweight='bold')
    axes[0, 1].set_title('Window Duration vs Window Size', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    signal_segment = sample_signal[:1000]
    axes[1, 0].plot(signal_segment, alpha=0.5, color='gray', label='Signal')
    
    for ws in [50, 150, 300]:
        step = int(ws * 0.5)
        for i in range(0, len(signal_segment) - ws, step):
            if i == 0:
                axes[1, 0].axvspan(i, i+ws, alpha=0.2, label=f'Window size {ws}')
            else:
                axes[1, 0].axvspan(i, i+ws, alpha=0.2)
    
    axes[1, 0].set_xlabel('Time (samples)', fontweight='bold')
    axes[1, 0].set_ylabel('Amplitude', fontweight='bold')
    axes[1, 0].set_title('Window Segmentation Examples', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    norm_windows = window_analysis_df['n_windows'] / window_analysis_df['n_windows'].max()
    norm_duration = window_analysis_df['time_duration_sec'] / window_analysis_df['time_duration_sec'].max()
    
    x = np.arange(len(window_analysis_df))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, norm_windows, width, label='Normalized Window Count', alpha=0.8)
    axes[1, 1].bar(x + width/2, norm_duration, width, label='Normalized Duration', alpha=0.8)
    axes[1, 1].set_xlabel('Window Size Index', fontweight='bold')
    axes[1, 1].set_ylabel('Normalized Value', fontweight='bold')
    axes[1, 1].set_title('Trade-off: Sample Count vs Time Resolution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(window_analysis_df['window_size'], rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_distributions(features_df, output_path):
    """Task 5: Plot feature distributions."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    feature_cols = ['mean', 'std', 'rms', 'zcr', 'dominant_freq', 
                   'spectral_energy', 'spectral_entropy', 'skewness', 'kurtosis']
    
    for idx, col in enumerate(feature_cols):
        if col in features_df.columns:
            axes[idx].hist(features_df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(features_df, feature_cols, output_path):
    """Task 5: Plot feature importance using NumPy-based ANOVA F-score."""
    import numpy as np
    from utils import LabelEncoder
    
    X = features_df[feature_cols].fillna(0).values
    y = LabelEncoder().fit_transform(features_df['label'])
    
    # Calculate feature importance using ANOVA F-score (between-class / within-class variance)
    unique_labels = np.unique(y)
    n_features = X.shape[1]
    importance_scores = np.zeros(n_features)
    
    for feat_idx in range(n_features):
        feature_values = X[:, feat_idx]
        
        # Calculate between-class variance
        overall_mean = np.mean(feature_values)
        between_var = 0.0
        for label in unique_labels:
            class_mask = y == label
            class_mean = np.mean(feature_values[class_mask])
            class_size = np.sum(class_mask)
            between_var += class_size * (class_mean - overall_mean) ** 2
        
        # Calculate within-class variance
        within_var = 0.0
        for label in unique_labels:
            class_mask = y == label
            class_values = feature_values[class_mask]
            if len(class_values) > 1:
                within_var += np.sum((class_values - np.mean(class_values)) ** 2)
        
        # F-score: between-class variance / within-class variance
        if within_var > 0:
            importance_scores[feat_idx] = between_var / within_var
        else:
            importance_scores[feat_idx] = 0.0
    
    # Normalize importance scores to sum to 1 (similar to sklearn's feature_importances_)
    if np.sum(importance_scores) > 0:
        importance_scores = importance_scores / np.sum(importance_scores)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    top_features = importance_df.head(15)
    axes[0].barh(range(len(top_features)), top_features['importance'])
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'])
    axes[0].set_xlabel('Importance Score', fontweight='bold')
    axes[0].set_title('Top 15 Most Important Features', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    time_domain = ['mean', 'std', 'rms', 'zcr', 'min', 'max', 'range', 'median', 'skewness', 'kurtosis', 'peak_to_peak', 'mad']
    freq_domain = ['dominant_freq', 'spectral_energy', 'spectral_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_rolloff']
    
    time_importance = importance_df[importance_df['feature'].isin(time_domain)]['importance'].sum()
    freq_importance = importance_df[importance_df['feature'].isin(freq_domain)]['importance'].sum()
    
    categories = ['Time Domain', 'Frequency Domain']
    importance_values = [time_importance, freq_importance]
    colors_highlight = ['#FF6B6B', '#4ECDC4']
    
    axes[1].bar(categories, importance_values, color=colors_highlight, edgecolor='black', alpha=0.8)
    axes[1].set_ylabel('Total Importance', fontweight='bold')
    axes[1].set_title('Feature Importance by Domain', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return importance_df


def plot_model_comparison(results_df, output_path):
    """Task 6: Plot model comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[0, 0].bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)
    
    axes[0, 0].set_xlabel('Model', fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontweight='bold')
    axes[0, 0].set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.1])
    
    axes[0, 1].bar(results_df['Model'], results_df['Train Time (s)'], alpha=0.8, edgecolor='black')
    axes[0, 1].set_xlabel('Model', fontweight='bold')
    axes[0, 1].set_ylabel('Time (seconds)', fontweight='bold')
    axes[0, 1].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    sorted_results = results_df.sort_values('F1-Score', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_results)))
    axes[1, 0].barh(sorted_results['Model'], sorted_results['F1-Score'], color=colors, edgecolor='black')
    axes[1, 0].set_xlabel('F1-Score (Macro)', fontweight='bold')
    axes[1, 0].set_title('Model Ranking by F1-Score', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].set_xlim([0, 1.1])
    
    for i, v in enumerate(sorted_results['F1-Score']):
        axes[1, 0].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
    
    axes[1, 1].scatter(results_df['Train Time (s)'], results_df['Accuracy'], 
                       s=200, alpha=0.6, edgecolor='black', linewidth=2)
    
    for i, model in enumerate(results_df['Model']):
        axes[1, 1].annotate(model, 
                           (results_df['Train Time (s)'].iloc[i], results_df['Accuracy'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    axes[1, 1].set_xlabel('Training Time (seconds)', fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy', fontweight='bold')
    axes[1, 1].set_title('Accuracy vs Training Time Trade-off', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_evaluation_comparison(evaluation_comparison_df, output_path):
    """Task 7: Plot standard split vs LOSO comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, evaluation_comparison_df.iloc[0, 1:], width, label='Standard Split', alpha=0.8)
    ax.bar(x + width/2, evaluation_comparison_df.iloc[1, 1:], width, label='LOSO', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Standard Split vs LOSO Evaluation Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_window_size_comparison(window_results_df, output_path):
    """Task 7: Plot window size comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(window_results_df['Window Size'], window_results_df['Accuracy'], 
                    'o-', label='Accuracy', linewidth=2)
    axes[0, 0].plot(window_results_df['Window Size'], window_results_df['Precision'], 
                    's-', label='Precision', linewidth=2)
    axes[0, 0].plot(window_results_df['Window Size'], window_results_df['Recall'], 
                    '^-', label='Recall', linewidth=2)
    axes[0, 0].plot(window_results_df['Window Size'], window_results_df['F1-Score'], 
                    'd-', label='F1-Score', linewidth=2)
    axes[0, 0].set_xlabel('Window Size (samples)', fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontweight='bold')
    axes[0, 0].set_title('Performance Metrics vs Window Size', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(window_results_df)))
    axes[0, 1].bar(window_results_df['Window Size'].astype(str), window_results_df['F1-Score'], 
                   color=colors, edgecolor='black')
    axes[0, 1].set_xlabel('Window Size (samples)', fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score', fontweight='bold')
    axes[0, 1].set_title('F1-Score by Window Size', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.1])
    
    axes[1, 0].scatter(window_results_df['N Samples'], window_results_df['F1-Score'], 
                       s=200, alpha=0.6, edgecolor='black', linewidth=2)
    for i, ws in enumerate(window_results_df['Window Size']):
        axes[1, 0].annotate(f"{ws}", 
                           (window_results_df['N Samples'].iloc[i], window_results_df['F1-Score'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 0].set_xlabel('Number of Samples', fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score', fontweight='bold')
    axes[1, 0].set_title('F1-Score vs Sample Count', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    best_idx = window_results_df['F1-Score'].idxmax()
    best_window = window_results_df.loc[best_idx]
    
    categories = ['Small\n(25-75)', 'Medium\n(100-200)', 'Large\n(250-300)']
    f1_scores = [
        window_results_df[window_results_df['Window Size'].between(25, 75)]['F1-Score'].mean(),
        window_results_df[window_results_df['Window Size'].between(100, 200)]['F1-Score'].mean(),
        window_results_df[window_results_df['Window Size'].between(250, 300)]['F1-Score'].mean()
    ]
    colors_highlight = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    axes[1, 1].barh(categories, f1_scores, color=colors_highlight, edgecolor='black')
    axes[1, 1].set_xlabel('F1-Score', fontweight='bold')
    axes[1, 1].set_title('Performance by Window Size Category', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].set_xlim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, cm_normalized, class_names, output_path):
    """Task 7: Plot confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel('Predicted Label', fontweight='bold')
    axes[0].set_ylabel('True Label', fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel('Predicted Label', fontweight='bold')
    axes[1].set_ylabel('True Label', fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_analysis(error_pairs, output_path):
    """Task 7: Plot error analysis."""
    from collections import Counter
    
    error_counts = Counter(error_pairs)
    most_common_errors = error_counts.most_common(10)
    
    if most_common_errors:
        error_labels = [f"{true_cls}→{pred_cls}" for (true_cls, pred_cls), _ in most_common_errors]
        error_values = [count for _, count in most_common_errors]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(error_labels, error_values, color='coral', edgecolor='black')
        ax.set_xlabel('Number of Misclassifications', fontweight='bold')
        ax.set_title('Top 10 Misclassification Patterns', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

