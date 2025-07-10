CNN-LSTM-MHA Forex Prediction
Pure Historical Data Training for Binary Direction Prediction
🎯 Overview
This is a pure implementation of CNN-LSTM with Multi-Head Attention for forex direction prediction. The system uses only historical data without artificial augmentation or complex tricks.
Key Features:

✅ Pure Historical Data: No SMOTE, no class balancing, no data augmentation
✅ Binary Classification: Simple 0=Down, 1=Up prediction
✅ Clean Architecture: CNN → LSTM → Multi-Head Attention → Dense
✅ Timestamp Experiments: Each training creates a separate timestamped folder
✅ Simple Output: Probability [0,1] representing market direction


🏗️ Architecture
Input(60,15) → CNN(64→128) → LSTM(128→64) → Multi-Head Attention(8 heads) → Dense(64→32→1) → Sigmoid
Model Components:

Input: 60 timesteps × 15 features (EUR/USD/GBP × OHLCV)
CNN: Feature extraction with BatchNorm
LSTM: Temporal processing with dropout
Multi-Head Attention: 8 attention heads for pattern recognition
Dense: Final classification layers
Output: Sigmoid activation for [0,1] probability


📁 Project Structure
📁 PROJECT_ROOT/
├── 📜 main.py                    # Pure training pipeline
├── 📜 config.py                  # Simple configuration
├── 📜 data_processor.py          # Pure preprocessing
├── 📜 cnn_lstm_mha.py           # Pure CNN-LSTM-MHA model
├── 📜 requirements.txt           # Minimal dependencies
│
├── 📁 data/                      # Input data
│   ├── EURUSD_1H.csv
│   ├── GBPUSD_1H.csv
│   └── USDJPY_1H.csv
│
└── 📁 experiments/               # Training results (auto-created)
    ├── 📁 20250704_183045/
    │   ├── best_model.keras
    │   ├── final_model.keras
    │   ├── training_log.txt
    │   ├── predictions.csv
    │   └── summary.txt
    └── 📁 20250704_191223/
        └── ...

🚀 Usage
1. Install Dependencies
bashpip install -r requirements.txt
2. Prepare Data
Place your CSV files in the data/ folder:

EURUSD_1H.csv
GBPUSD_1H.csv
USDJPY_1H.csv

Expected CSV format:
Local time,Open,High,Low,Close,Volume
01.01.2019 00:00:00.000 GMT+0000,1.20137,1.20158,1.20026,1.20106,6885.930
3. Run Training
bashpython main.py
Each training run creates a new timestamped experiment folder automatically.

📊 Data Processing
OHLC Processing:

Percentage Change: Convert prices to returns
Z-Score Normalization: Using training set statistics only
No Data Leakage: Validation uses training statistics

Volume Processing:

7SD Capping: Remove extreme outliers
Min-Max Scaling: Scale to [0,1] using training statistics

Target Variable:

Binary: 0 = Price goes down, 1 = Price goes up
Simple: No artificial balancing or augmentation


🎯 Training Philosophy
Pure Approach:

Accept Natural Distribution: Use data as-is without forcing balance
Historical Truth: Learn from real market behavior
No Artificial Boosting: No SMOTE, class weights, or tricks
Quality Architecture: Focus on model design rather than data manipulation

Expected Performance:

Accuracy: 52-58% (realistic for forex prediction)
Prediction Range: Natural distribution across [0,1]
Signal Balance: Based on actual market conditions


📈 Results
Output Files:

best_model.keras - Best model during training
final_model.keras - Final trained model
training_log.txt - Training metrics
predictions.csv - Validation predictions
summary.txt - Experiment summary

Key Metrics:

Accuracy: Overall direction prediction accuracy
Precision/Recall: Binary classification metrics
Prediction Range: Min/max probability values
Signal Distribution: SELL/HOLD/BUY breakdown


⚙️ Configuration
Edit config.py to customize:
python# Model Architecture
CNN_FILTERS = [64, 128]
LSTM_UNITS = [128, 64]
MHA_HEADS = 8

# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# Data Periods
TRAIN_START = '2019-01-01'
TRAIN_END = '2020-12-31'
VAL_START = '2021-01-01'
VAL_END = '2021-01-31'