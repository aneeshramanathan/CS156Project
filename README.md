# Human Behavior Prediction and Habit Modeling System

## Project Overview

This project develops an AI system that predicts a user's next action based on their recent sequence of movements using deep learning models (LSTM, GRU, Transformers). The system learns patterns and routines to provide personalized predictions for applications in fitness tracking, health monitoring, and smart home automation.

## Repository Structure

```
CS156Project/
├── README.md                              # This file
├── activity_detection_analysis.ipynb      # Assignment tasks (Tasks 1-7)
├── behavior_prediction_system.ipynb       # Main project notebook
└── [Generated files after running]
```

## Two Notebooks Explained

### 1. `activity_detection_analysis.ipynb` (Assignment Requirements)

This notebook fulfills all **7 assignment tasks** with classical ML and evaluation:

- **Task 1**: Dataset Exploration (participants, demographics, labels)
- **Task 2**: Signal plots and pattern analysis
- **Task 3**: Signal preprocessing (filtering, interpolation)
- **Task 4**: Windowing strategies (≥10 window sizes)
- **Task 5**: Feature extraction (time & frequency domain)
- **Task 6**: Classical ML models (DT, SVM, NB, RF, AdaBoost, XGBoost)
- **Task 7**: Advanced evaluation (80/20 split + LOSO cross-validation)

**Points Distribution**: 10.0 points total (as per assignment rubric)

### 2. `behavior_prediction_system.ipynb` (Main Project)

This is your **actual project** focused on behavior prediction and habit modeling:

#### Key Features:

1. **Next-Activity Prediction**
   - Predict what user will do next based on recent activity sequence
   - Uses sequences of 10 previous activities to predict the 11th

2. **Deep Learning Models**
   - **LSTM**: Captures long-term dependencies in behavior sequences
   - **GRU**: More efficient variant with comparable performance
   - **Transformer**: State-of-the-art attention mechanism

3. **Transition Detection**
   - Analyzes activity transitions (e.g., walking → sitting)
   - Creates transition probability matrices
   - Identifies common behavior patterns

4. **Routine Modeling**
   - Extracts user-specific patterns
   - Learns daily routines
   - Identifies repeated behavior sequences

5. **Interactive Dashboard**
   - Activity timeline visualization
   - Distribution patterns
   - Common transitions
   - Real-time predictions
   - Saved as `dashboard.html`

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install kagglehub pandas numpy matplotlib seaborn scipy scikit-learn tensorflow torch transformers plotly dash xgboost jupyter
```

Or use the notebooks' built-in installation cells.

## Quick Start

### Option 1: Run the Assignment Notebook

```bash
jupyter notebook activity_detection_analysis.ipynb
```

Then execute all cells sequentially. This will:
- Download the dataset automatically
- Complete all 7 assignment tasks
- Generate visualizations and results
- Save evaluation metrics

### Option 2: Run the Behavior Prediction Notebook

```bash
jupyter notebook behavior_prediction_system.ipynb
```

This will:
- Download and prepare sequence data
- Train LSTM, GRU, and Transformer models
- Analyze activity transitions
- Extract user patterns
- Generate interactive dashboard
- Save trained models

## Dataset

**Source**: [Activity Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/edgeimpulse/activity-detection)

The dataset contains accelerometer and gyroscope sensor data from users performing various activities. If the real dataset doesn't contain labeled sequences, the notebook will automatically generate synthetic data that models realistic daily routines.

### Activities Tracked:
- Sleeping
- Walking
- Running
- Sitting
- Standing
- Eating
- Working
- Exercising
- And more...

## Model Architecture

### LSTM Model
```
Embedding → LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(64) → Dense(n_classes)
```

### GRU Model
```
Embedding → GRU(64) → Dropout → GRU(32) → Dropout → Dense(64) → Dense(n_classes)
```

### Transformer Model
```
Embedding + Positional Encoding → Multi-Head Attention → Feed Forward →
Global Pooling → Dense(64) → Dense(n_classes)
```

## Results

Expected performance (may vary based on dataset):

| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| LSTM        | 85-92%   | 84-91%    | 83-90% | 84-91%   |
| GRU         | 84-91%   | 83-90%    | 82-89% | 83-90%   |
| Transformer | 86-93%   | 85-92%    | 84-91% | 85-92%   |

*Note: Results depend on data quality and sequence patterns*

## Generated Files

After running the notebooks, you'll have:

### Assignment Notebook Output:
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
- `X_seq_train.npy`, `X_seq_test.npy` (deep learning data)
- `y_seq_train.npy`, `y_seq_test.npy` (labels)

### Behavior Prediction Output:
- `lstm_model.h5` - Trained LSTM model
- `gru_model.h5` - Trained GRU model
- `transformer_model.h5` - Trained Transformer model
- `label_encoder.pkl` - Activity label encoder
- `transition_matrix.npy` - Activity transition probabilities
- `activity_names.txt` - List of activities
- `model_comparison_results.csv` - Performance metrics
- `dashboard.html` - Interactive visualization dashboard
- Various PNG visualizations

## Usage Examples

### Predict Next Activity

```python
from tensorflow import keras
import pickle
import numpy as np

# Load model and encoder
model = keras.models.load_model('lstm_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Recent activity sequence
recent_activities = ['walking', 'walking', 'standing', 'sitting',
                     'sitting', 'eating', 'standing', 'walking',
                     'walking', 'running']

# Encode and predict
encoded = le.transform(recent_activities).reshape(1, -1)
probs = model.predict(encoded)[0]

# Get top 3 predictions
top_3 = np.argsort(probs)[-3:][::-1]
for idx in top_3:
    activity = le.classes_[idx]
    probability = probs[idx]
    print(f"{activity}: {probability*100:.1f}%")
```

### Analyze User Patterns

```python
import numpy as np

# Load transition matrix
trans_matrix = np.load('transition_matrix.npy')
with open('activity_names.txt', 'r') as f:
    activities = f.read().split('\n')

# Find most likely transition from 'walking'
walking_idx = activities.index('walking')
next_probs = trans_matrix[walking_idx]
most_likely = activities[np.argmax(next_probs)]

print(f"After walking, user most likely: {most_likely}")
```

## Applications

### 1. Fitness Apps
- Predict user's workout routine
- Suggest activities based on patterns
- Track consistency and progress

### 2. Health Monitoring
- Detect anomalies in daily patterns
- Alert caregivers for elderly/patients
- Monitor recovery progress

### 3. Smart Home Automation
- Automate lights, temperature based on predicted activities
- Prepare coffee when user predicted to wake up
- Lock doors when user predicted to sleep

### 4. Personal Assistants
- Proactive calendar suggestions
- Reminder timing optimization
- Context-aware notifications

## Customization

### Change Sequence Length

```python
SEQUENCE_LENGTH = 15  # Instead of 10
X_seq, y_seq, user_ids = create_prediction_sequences(
    sequences,
    sequence_length=SEQUENCE_LENGTH,
    stride=3
)
```

### Add More Deep Learning Models

```python
# Example: Bidirectional LSTM
def build_bilstm_model(sequence_length, n_classes):
    model = keras.Sequential([
        layers.Embedding(n_classes, 32, input_length=sequence_length),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

### Personalize for Single User

```python
# Filter data for specific user
user_sequences = [seq for seq in sequences if seq['participant'] == 'user_0']
X_user, y_user, _ = create_prediction_sequences(user_sequences, sequence_length=10)

# Train personalized model
personal_model = build_lstm_model(10, n_classes)
personal_model.fit(X_user, y_user, epochs=50, validation_split=0.2)
```

## Troubleshooting

### Issue: Dataset not found
**Solution**: The notebook will automatically generate synthetic data if the real dataset isn't available. You can also manually download from Kaggle.

### Issue: Out of memory
**Solution**:
```python
# Reduce batch size
model.fit(X_train, y_train, batch_size=32)  # Instead of 64

# Or reduce sequence length
SEQUENCE_LENGTH = 5  # Instead of 10
```

### Issue: Low accuracy
**Solution**:
- Increase sequence length (more context)
- Add more training data
- Tune hyperparameters (learning rate, units, layers)
- Try ensemble methods

### Issue: GPU not detected
**Solution**:
```bash
# For TensorFlow
pip install tensorflow-gpu

# For PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Future Enhancements

- [ ] Real-time prediction pipeline
- [ ] Mobile app integration
- [ ] Online learning for personalization
- [ ] Multi-sensor fusion (heart rate, GPS)
- [ ] Anomaly detection for health monitoring
- [ ] Web dashboard with live updates
- [ ] Edge device deployment (Raspberry Pi, smartphones)
- [ ] Transfer learning across users

## Team Contributions

Add your team member contributions here:

- **Member 1**: [Name] - Data preprocessing, LSTM model
- **Member 2**: [Name] - GRU model, evaluation
- **Member 3**: [Name] - Transformer model, dashboard
- **Member 4**: [Name] - Pattern analysis, documentation

## References

1. Activity Recognition Dataset: [Kaggle Link](https://www.kaggle.com/datasets/edgeimpulse/activity-detection)
2. LSTM: Hochreiter & Schmidhuber (1997) - Long Short-Term Memory
3. GRU: Cho et al. (2014) - Learning Phrase Representations using RNN
4. Transformer: Vaswani et al. (2017) - Attention Is All You Need
5. HAR: Human Activity Recognition with Smartphones - UCI ML Repository

## License

This project is for educational purposes as part of CS156.

## Contact

For questions or issues, please open an issue in the repository.

---

**Happy Predicting! 🚀**
