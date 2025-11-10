# Troubleshooting Guide

This document contains solutions to common errors encountered while running the notebooks.

## Fixed Issues

### ✅ Issue 1: LOSO ValueError - "fewer than 2 unique groups"

**Error Message:**
```
ValueError: The groups parameter contains fewer than 2 unique groups (['synthetic']).
LeaveOneGroupOut expects at least 2.
```

**Cause:**
The synthetic data was generating only 1 participant, but LOSO (Leave-One-Subject-Out) cross-validation requires at least 2 participants to work.

**Fix Applied:**
1. Modified synthetic data generation to create **3 participants** (`synthetic_user_0`, `synthetic_user_1`, `synthetic_user_2`)
2. Added automatic fallback to 5-fold cross-validation if fewer than 2 participants are detected
3. Added clear warning messages

**Status:** ✅ Fixed in the current notebook

---

### ✅ Issue 2: Classification Report ValueError - "Number of classes does not match size of target_names"

**Error Message:**
```
ValueError: Number of classes, 2, does not match size of target_names, 1.
Try specifying the labels parameter
```

**Cause:**
The label encoder was fit on all classes in the full dataset, but the test set might only contain a subset of those classes.

**Fix Applied:**
1. Filter class names to only include classes present in the test set
2. Use the `labels` parameter in `classification_report()` to explicitly specify which classes to report
3. Ensure confusion matrix dimensions match the filtered class names

**Status:** ✅ Fixed in the current notebook

---

## Common Issues and Solutions

### Issue: "ModuleNotFoundError" - Missing packages

**Solution:**
```bash
pip install kagglehub pandas numpy matplotlib seaborn scipy scikit-learn tensorflow torch transformers plotly dash xgboost jupyter
```

---

### Issue: Kaggle API authentication error

**Solution:**
1. Go to https://www.kaggle.com/settings/account
2. Create New API Token and download `kaggle.json`
3. Place in `~/.kaggle/kaggle.json` (Mac/Linux) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

---

### Issue: Out of memory error

**Solution:**
```python
# Reduce batch size
model.fit(X_train, y_train, batch_size=32)  # Instead of 64

# Or reduce features
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
```

---

### Issue: GPU not detected

**Check GPU:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import torch
print(torch.cuda.is_available())
```

**Solution:**
```bash
# For NVIDIA GPU
pip install tensorflow-gpu torch

# For Apple Silicon
pip install tensorflow-metal
```

---

## Getting Help

If you encounter other issues:
1. Check error messages carefully
2. Review the README.md
3. Verify package versions: `pip list | grep -E "tensorflow|torch|sklearn"`

---

**Last Updated:** November 9, 2024
