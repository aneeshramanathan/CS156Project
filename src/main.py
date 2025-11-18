"""Main script to run all activity detection analysis tasks."""

# Set non-interactive matplotlib backend early to avoid multiprocessing/threading issues
import matplotlib
matplotlib.use('Agg')  # Must be set before any matplotlib imports

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import (
    DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP, DEFAULT_SAMPLING_FREQUENCY,
    VISUALIZATIONS_DIR, ensure_visualizations_dir
)
from src.data_loading import download_dataset, load_activity_data, create_dataset_summary
from src.preprocessing import preprocess_signal, preprocess_dataset
from src.windowing import create_windows, analyze_window_sizes
from src.feature_extraction import extract_features_from_dataset
from src.modeling import train_classical_models
from src.evaluation import (
    evaluate_standard_split, evaluate_loso, compare_window_sizes,
    generate_confusion_matrix, generate_classification_report
)
from src.visualization import (
    plot_dataset_exploration, plot_annotated_signals, plot_preprocessing_comparison,
    plot_windowing_strategies, plot_feature_distributions, plot_feature_importance,
    plot_model_comparison, plot_evaluation_comparison, plot_window_size_comparison,
    plot_confusion_matrix, plot_error_analysis
)


def main():
    """Run all tasks sequentially."""
    
    output_dir = ensure_visualizations_dir()
    print("="*60)
    print("ACTIVITY DETECTION ANALYSIS - COMPREHENSIVE PIPELINE")
    print("="*60)
    
    # ============================================================================
    # Task 1: Dataset Exploration
    # ============================================================================
    print("\n" + "="*60)
    print("TASK 1: Dataset Exploration")
    print("="*60)
    
    dataset_path = download_dataset()
    data_list, metadata = load_activity_data(dataset_path)
    summary_table, unique_labels = create_dataset_summary(data_list, metadata)
    
    print("\n=== DATASET SUMMARY TABLE ===")
    print(summary_table.to_string(index=False))
    
    if unique_labels:
        print(f"\nActivity types: {sorted(unique_labels)[:10]}")
    
    plot_dataset_exploration(data_list, metadata, output_dir / 'task1_dataset_exploration.png')
    print("\n[OK] Task 1 Complete: Dataset summary created")
    
    # ============================================================================
    # Task 2: Annotated Signal Exploration
    # ============================================================================
    print("\n" + "="*60)
    print("TASK 2: Annotated Signal Exploration")
    print("="*60)
    
    signal_data = plot_annotated_signals(data_list, n_samples=3, 
                                         output_path=output_dir / 'task2_annotated_signals.png')
    
    if not signal_data:
        print("No signal data with labels found. Creating synthetic example...")
        signal_data = []
        for participant_id in range(3):
            t = np.linspace(0, 10, 1000)
            activities = ['walking', 'running', 'standing', 'sitting'] * 250
            np.random.seed(42 + participant_id)
            noise_level = 0.1 * (1 + participant_id * 0.2)
            synthetic_df = pd.DataFrame({
                'timestamp': t,
                'acc_x': np.sin(2*np.pi*t) + np.random.normal(0, noise_level, len(t)),
                'acc_y': np.cos(2*np.pi*t) + np.random.normal(0, noise_level, len(t)),
                'acc_z': np.sin(4*np.pi*t) + np.random.normal(0, noise_level, len(t)),
                'activity': activities[:len(t)]
            })
            signal_data.append({
                'df': synthetic_df,
                'sensor_cols': ['acc_x', 'acc_y', 'acc_z'],
                'label_col': 'activity',
                'participant': f'synthetic_user_{participant_id}'
            })
        plot_annotated_signals(data_list, n_samples=3, 
                              output_path=output_dir / 'task2_annotated_signals.png')
    
    print("\n[OK] Task 2 Complete: Signal visualization finished")
    
    # ============================================================================
    # Task 3: Signal Preprocessing
    # ============================================================================
    print("\n" + "="*60)
    print("TASK 3: Signal Preprocessing")
    print("="*60)
    
    if signal_data:
        sample_data = signal_data[0]['df']
        sensor_cols = signal_data[0]['sensor_cols'][:3]
        test_signal = sample_data[sensor_cols[0]].values.copy()
        missing_indices = np.random.choice(len(test_signal), 
                                          size=int(0.05*len(test_signal)), replace=False)
        test_signal[missing_indices] = np.nan
        preprocessed = preprocess_signal(test_signal)
        
        plot_preprocessing_comparison(test_signal, preprocessed, 
                                     output_path=output_dir / 'task3_preprocessing_comparison.png')
        
        print(f"Missing data points: {np.sum(np.isnan(test_signal))}")
        print(f"After preprocessing: {np.sum(np.isnan(preprocessed))}")
        print(f"Original signal std: {np.nanstd(test_signal):.4f}")
        print(f"Preprocessed signal std: {np.std(preprocessed):.4f}")
    
    preprocessed_data = preprocess_dataset(signal_data)
    print("\n[OK] Task 3 Complete: Signal preprocessing finished")
    
    # ============================================================================
    # Task 4: Windowing Strategies
    # ============================================================================
    print("\n" + "="*60)
    print("TASK 4: Windowing Strategies")
    print("="*60)
    
    if preprocessed_data:
        sample_signal = preprocessed_data[0]['df'][preprocessed_data[0]['preprocessed_cols'][0]].values
        window_sizes = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]
        window_analysis_df = analyze_window_sizes(sample_signal, window_sizes, 
                                                   overlap=DEFAULT_OVERLAP, 
                                                   fs=DEFAULT_SAMPLING_FREQUENCY)
        
        print("\n=== WINDOWING ANALYSIS ===")
        print(window_analysis_df.to_string(index=False))
        
        plot_windowing_strategies(window_analysis_df, sample_signal, 
                                 output_path=output_dir / 'task4_windowing_strategies.png')
    
    print("\n[OK] Task 4 Complete: Windowing analysis finished")
    
    # ============================================================================
    # Task 5: Feature Extraction & Analysis
    # ============================================================================
    print("\n" + "="*60)
    print("TASK 5: Feature Extraction & Analysis")
    print("="*60)
    
    print("Extracting features from all data...")
    features_df = extract_features_from_dataset(preprocessed_data, 
                                                window_size=DEFAULT_WINDOW_SIZE, 
                                                overlap=DEFAULT_OVERLAP)
    
    print(f"\nExtracted {len(features_df)} feature vectors")
    print(f"Features: {len([c for c in features_df.columns if c not in ['label', 'participant', 'sensor_channel']])}")
    
    # Start with all numeric feature columns (excluding label/metadata)
    feature_cols = [col for col in features_df.columns 
                   if col not in ['label', 'participant', 'sensor_channel']]
    
    sample_features = features_df[feature_cols[:9]].head(100)
    plot_feature_distributions(sample_features, 
                              output_path=output_dir / 'task5_feature_distributions.png')
    
    importance_df = plot_feature_importance(features_df, feature_cols, 
                                           output_path=output_dir / 'task5_feature_importance.png')
    
    print("\n=== TOP 15 MOST IMPORTANT FEATURES ===")
    print(importance_df.head(15).to_string(index=False))
    
    # Feature selection: keep only the most discriminative features for modeling.
    # This reduces noise and focuses the classifiers on the strongest signals.
    top_feature_count = 30
    top_features = importance_df.head(top_feature_count)['feature'].tolist()
    preserved_cols = {'sensor_channel_idx'}
    feature_cols = [col for col in features_df.columns 
                   if (col in top_features) or (col in preserved_cols)]
    
    print("\n[OK] Task 5 Complete: Feature extraction finished")
    
    # ============================================================================
    # Task 6: Classical ML Modeling
    # ============================================================================
    print("\n" + "="*60)
    print("TASK 6: Classical ML Modeling")
    print("="*60)
    
    # Each row in features_df is a single sensor window; y is the **current**
    # activity label for that window (activity classification, not
    # next-activity prediction).
    X = features_df[feature_cols].fillna(0).values
    y = features_df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y_encoded))}")
    
    results_df, trained_models = train_classical_models(X_train, y_train, X_test, y_test)
    
    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(results_df.to_string(index=False))
    
    plot_model_comparison(results_df, output_path=output_dir / 'task6_model_comparison.png')
    
    # Choose best model for downstream evaluation (Task 7) based on macro F1
    model_names = [name for name in trained_models.keys()]
    best_rows = results_df[results_df['Model'].isin(model_names)]
    best_model_name = best_rows.loc[best_rows['F1-Score'].idxmax(), 'Model']
    best_model = trained_models[best_model_name]
    print(f"\nBest model (by macro F1-Score): {best_model_name}")
    
    print("\n[OK] Task 6 Complete: Classical ML modeling finished")
    
    # ============================================================================
    # Task 7: Advanced Evaluation
    # ============================================================================
    print("\n" + "="*60)
    print("TASK 7: Advanced Evaluation")
    print("="*60)
    
    # Evaluate activity classification performance (per-window current activity)
    # using both a standard train/test split and LOSO cross-validation.
    X_full = features_df[feature_cols].fillna(0).values
    y_full = le.fit_transform(features_df['label'])
    groups = features_df['participant'].values
    
    try:
        best_model_params = best_model.get_params()
    except Exception:
        best_model_params = None
    loso_results = evaluate_loso(
        X_full,
        y_full,
        groups,
        model_class=best_model.__class__,
        model_params=best_model_params
    )
    
    standard_results = evaluate_standard_split(best_model, X_train, y_train, X_test, y_test)
    
    evaluation_comparison = pd.DataFrame({
        'Evaluation Method': ['Standard Split (80/20)', 'LOSO Cross-Validation'],
        'Accuracy': [standard_results['accuracy'], loso_results['accuracy']],
        'Precision (macro)': [standard_results['precision'], loso_results['precision']],
        'Recall (macro)': [standard_results['recall'], loso_results['recall']],
        'F1-Score (macro)': [standard_results['f1_score'], loso_results['f1_score']]
    })
    
    print("\n=== EVALUATION METHOD COMPARISON ===")
    print(evaluation_comparison.to_string(index=False))
    
    plot_evaluation_comparison(evaluation_comparison, 
                               output_path=output_dir / 'task7_evaluation_comparison.png')
    
    # Task 7: compare at least 10 window sizes for both standard split and LOSO,
    # mirroring the Task 4 analysis.
    window_sizes_test = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300]
    window_results_df = compare_window_sizes(
        preprocessed_data,
        window_sizes_test,
        lambda df: [c for c in feature_cols if c in df.columns],
        le
    )
    
    print("\n=== WINDOW SIZE COMPARISON TABLE ===")
    print(window_results_df.to_string(index=False))
    
    plot_window_size_comparison(window_results_df, 
                                output_path=output_dir / 'task7_window_size_comparison.png')
    
    # Evaluate best model
    y_pred_best = best_model.predict(X_test)
    cm, cm_normalized, class_names = generate_confusion_matrix(y_test, y_pred_best, le.classes_)
    
    plot_confusion_matrix(cm, cm_normalized, class_names, 
                         output_path=output_dir / 'task7_confusion_matrix.png')
    
    print("\n=== PER-CLASS PERFORMANCE REPORT ===")
    report = generate_classification_report(y_test, y_pred_best, le.classes_)
    print(report)
    
    misclassified = y_test != y_pred_best
    error_pairs = []
    for true_label, pred_label in zip(y_test[misclassified], y_pred_best[misclassified]):
        try:
            error_pairs.append((le.classes_[true_label], le.classes_[pred_label]))
        except IndexError:
            error_pairs.append((f"Class_{true_label}", f"Class_{pred_label}"))
    
    if error_pairs:
        error_counts = Counter(error_pairs)
        most_common_errors = error_counts.most_common(10)
        print("\nMost common classification errors (true -> predicted):")
        for (true_class, pred_class), count in most_common_errors:
            print(f"  true={true_class}, predicted={pred_class}: {count} times")
        
        plot_error_analysis(error_pairs, output_path=output_dir / 'task7_error_analysis.png')
    
    print("\n[OK] Task 7 Complete: Advanced evaluation finished")
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "="*60)
    print("ALL TASKS COMPLETE!")
    print("="*60)
    print(f"\nVisualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - task1_dataset_exploration.png")
    print("  - task2_annotated_signals.png")
    print("  - task3_preprocessing_comparison.png")
    print("  - task4_windowing_strategies.png")
    print("  - task5_feature_distributions.png")
    print("  - task5_feature_importance.png")
    print("  - task6_model_comparison.png")
    print("  - task7_evaluation_comparison.png")
    print("  - task7_window_size_comparison.png")
    print("  - task7_confusion_matrix.png")
    print("  - task7_error_analysis.png")
    print("  - data/processed/X_seq_train.npy, data/processed/X_seq_test.npy (sequence data for deep activity classification models)")
    print("  - data/processed/y_seq_train.npy, data/processed/y_seq_test.npy (sequence labels for activity classification)")
    print("="*60)


if __name__ == "__main__":
    main()

