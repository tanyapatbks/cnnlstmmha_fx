import os
import datetime

class Config:
    def __init__(self):
        # Experiment Settings - Auto-generate timestamp-based folder
        self.EXPERIMENT_NAME = self.generate_experiment_name()
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_NAME}"
        
        # Data Settings
        self.WINDOW_SIZE = 120
        self.TOTAL_FEATURES = 27  # 3 currencies × 9 features (OHLCV + RSI + MACD)
        self.ALL_CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        # Indicators Settings
        self.RSI_PERIOD = 14
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # Data Split Periods (Fixed periods)
        self.TRAIN_START = '2018-01-01'
        self.TRAIN_END = '2021-12-31'
        self.VAL_START = '2022-03-01'
        self.VAL_END = '2022-03-31'
        
        # Model Architecture Parameters
        self.CNN_FILTERS = [64, 128]
        self.CNN_KERNEL_SIZE = 3
        self.LSTM_UNITS = [128, 64]
        self.MHA_HEADS = 16
        self.MHA_KEY_DIM = 64
        self.DENSE_UNITS = [64, 32]
        self.DROPOUT_RATE = 0.2
        
        # Training Parameters (Simple & Stable)
        self.LEARNING_RATE = 0.00001
        self.BATCH_SIZE = 32
        self.EPOCHS = 20
        self.EARLY_STOPPING_PATIENCE = 50
        self.REDUCE_LR_PATIENCE = 10
        
        # File Paths
        self.DATA_PATH = 'data/'
        
        # Create experiment directory
        self.create_experiment_directory()
        
    def generate_experiment_name(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return timestamp
    
    def create_experiment_directory(self):
        os.makedirs(self.EXPERIMENT_PATH, exist_ok=True)
        print(f"📁 Experiment: {self.EXPERIMENT_NAME}")
        
    def print_summary(self):
        print("⚙️ CONFIGURATION")
        print("-" * 40)
        print(f"Experiment: {self.EXPERIMENT_NAME}")
        print(f"Input Shape: ({self.WINDOW_SIZE}, {self.TOTAL_FEATURES})")
        print(f"Architecture: CNN{self.CNN_FILTERS} + LSTM{self.LSTM_UNITS} + MHA({self.MHA_HEADS})")
        print(f"Training: LR={self.LEARNING_RATE}, Batch={self.BATCH_SIZE}, Epochs={self.EPOCHS}")
        print(f"Data: {self.TRAIN_START} to {self.VAL_END}")
        print(f"Indicators: RSI({self.RSI_PERIOD}), MACD({self.MACD_FAST},{self.MACD_SLOW},{self.MACD_SIGNAL})")
        print("-" * 40)