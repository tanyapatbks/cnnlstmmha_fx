import os
import datetime

class Config:
    def __init__(self):
        # Experiment Settings - Auto-generate timestamp-based folder
        self.EXPERIMENT_NAME = self.generate_experiment_name()
        self.EXPERIMENT_PATH = f"experiments/{self.EXPERIMENT_NAME}"
        
        # Data Settings
        self.WINDOW_SIZE = 60
        self.TOTAL_FEATURES = 15  # 3 currencies × 5 features (OHLCV)
        self.ALL_CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        # Data Split Periods (Fixed periods)
        self.TRAIN_START = '2019-01-01'
        self.TRAIN_END = '2020-12-31'
        self.VAL_START = '2021-01-01'
        self.VAL_END = '2021-01-31'
        
        # Model Architecture Parameters
        self.CNN_FILTERS = [64, 128]
        self.CNN_KERNEL_SIZE = 3
        self.LSTM_UNITS = [128, 64]
        self.MHA_HEADS = 8
        self.MHA_KEY_DIM = 64
        self.DENSE_UNITS = [64, 32]
        self.DROPOUT_RATE = 0.3
        
        # Training Parameters (Simple & Stable)
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.EARLY_STOPPING_PATIENCE = 20
        self.REDUCE_LR_PATIENCE = 8
        
        # File Paths
        self.DATA_PATH = 'data/'
        
        # Create experiment directory
        self.create_experiment_directory()
        
    def generate_experiment_name(self):
        """Generate experiment name based on current timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return timestamp
    
    def create_experiment_directory(self):
        """Create experiment directory"""
        os.makedirs(self.EXPERIMENT_PATH, exist_ok=True)
        print(f"📁 Experiment: {self.EXPERIMENT_NAME}")
        
    def print_summary(self):
        """Print configuration summary"""
        print("⚙️ CONFIGURATION")
        print("-" * 40)
        print(f"Experiment: {self.EXPERIMENT_NAME}")
        print(f"Input Shape: ({self.WINDOW_SIZE}, {self.TOTAL_FEATURES})")
        print(f"Architecture: CNN{self.CNN_FILTERS} + LSTM{self.LSTM_UNITS} + MHA({self.MHA_HEADS})")
        print(f"Training: LR={self.LEARNING_RATE}, Batch={self.BATCH_SIZE}, Epochs={self.EPOCHS}")
        print(f"Data: {self.TRAIN_START} to {self.VAL_END}")
        print("-" * 40)