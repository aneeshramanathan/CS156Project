## Human Activity Classification Pipeline

### Overview

This project implements an end-to-end **activity classification** pipeline using raw sensor data (accelerometer, gyroscope, GPS) from the Edge Impulse Activity Detection dataset.  

Given multi-sensor recordings labeled as `cycling`, `walking`, or `sitting`, the pipeline:

- Loads and merges sensor streams per session
- Preprocesses signals
- Segments data into sliding windows
- Extracts time/frequency features
- Trains and compares classical ML models
- Evaluates with both standard train/test split and LOSO
- Produces visualizations and error analysis

The main task is **per-window classification of the current activity**.

---

## Project Structure

CS156Project/
  data/                      # Local cache of Kaggle activity sessions
  visualizations/            # Generated PNGs for Tasks 1–7
  data_loading.py            # Dataset download + loading/merging
  preprocessing.py           # Signal preprocessing
  windowing.py               # Sliding window utilities
  feature_extraction.py      # Time/frequency feature extraction
  modeling.py                # Classical ML models (sklearn + XGBoost)
  evaluation.py              # Standard split + LOSO + window-size eval
  visualization.py           # All plotting functions
  utils.py                   # Constants + platform detection
  main.py                    # Orchestrates the full pipeline
  requirements.txt           # Dependencies
  README.md                  # This file---

## Installation
```bash
python -m venv venv
```
# Windows
```bash
venv\Scripts\activate
```
# macOS / Linux
```bash
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```
Dependencies are standard: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `kagglehub`.

---

## Running

From the project root:
python main.py

This will:

- Download and cache the Kaggle dataset into `data/` (or reuse existing cache)
- Run the full analysis pipeline:
  1. **Dataset exploration** (summary table, label distribution)
  2. **Annotated signal plots** (sensor traces with activity regions)
  3. **Preprocessing** (band-pass filtering, interpolation, robust clipping)
  4. **Windowing analysis** (multiple window sizes)
  5. **Feature extraction** (time + frequency features per window)
  6. **Classical ML modeling**  
     - Decision Tree, Linear SVM, Naive Bayes, Random Forest, AdaBoost, XGBoost  
     - Stratified 80/20 split, label encoding, feature scaling  
     - Class weighting for imbalance, metrics via scikit-learn
  7. **Advanced evaluation**  
     - LOSO cross-validation across participants  
     - Window-size performance comparison  
     - Confusion matrix + error analysis (most common misclassifications)

All plots are saved under `visualizations/`, and metrics are printed to the console.

---

## Outputs

After `python main.py` finishes, you should have:

- In `visualizations/`:
  - `task1_dataset_exploration.png`
  - `task2_annotated_signals.png`
  - `task3_preprocessing_comparison.png`
  - `task4_windowing_strategies.png`
  - `task5_feature_distributions.png`
  - `task5_feature_importance.png`
  - `task6_model_comparison.png`
  - `task7_evaluation_comparison.png`
  - `task7_window_size_comparison.png`
  - `task7_confusion_matrix.png`
  - `task7_error_analysis.png`

- In the console:
  - Dataset summary
  - Feature importance ranking
  - Per-model metrics for the classical models
  - Standard vs LOSO metrics
  - Window-size comparison table
  - Per-class classification report
  - Top misclassification patterns (true → predicted)

---

## Tweaking the Pipeline

A few obvious knobs:

- `DEFAULT_WINDOW_SIZE`, `DEFAULT_OVERLAP` in `utils.py`  
- Filter band (`DEFAULT_LOWCUT`, `DEFAULT_HIGHCUT`) and clipping logic in `preprocessing.py`  
- Majority-label threshold `MIN_LABEL_FRACTION` in `feature_extraction.py`  
- Model hyperparameters in `modeling.py`  
- Which features are used (`feature_cols` selection) in `main.py`