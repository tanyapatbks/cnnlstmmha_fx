import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def load_currency_data(self):
        """Load raw currency data from CSV files"""
        raw_data = {}
        
        for pair in self.config.ALL_CURRENCY_PAIRS:
            file_path = f"{self.config.DATA_PATH}{pair}_1H.csv"
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Load data
            df = pd.read_csv(file_path)
            df['Local time'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
            df.set_index('Local time', inplace=True)
            df.sort_index(inplace=True)
            
            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            raw_data[pair] = df
            
        return raw_data
    
    def preprocess_ohlc(self, ohlc_data, train_start, train_end):
        """
        Simple OHLC preprocessing: percentage change + z-score normalization
        """
        # Calculate percentage change
        ohlc_pct = ohlc_data.pct_change().dropna()
        
        # Get training period for statistics
        train_start_dt = pd.to_datetime(train_start, utc=True)
        train_end_dt = pd.to_datetime(train_end, utc=True)
        train_mask = (ohlc_pct.index >= train_start_dt) & (ohlc_pct.index <= train_end_dt)
        train_ohlc = ohlc_pct[train_mask]
        
        # Z-score normalization using training statistics
        scaler = StandardScaler()
        scaler.fit(train_ohlc)
        ohlc_normalized = pd.DataFrame(
            scaler.transform(ohlc_pct),
            index=ohlc_pct.index,
            columns=ohlc_pct.columns
        )
        
        return ohlc_normalized
    
    def preprocess_volume(self, volume_data, train_start, train_end):
        """
        Simple Volume preprocessing: 7SD capping + min-max scaling
        """
        # Get training period for statistics
        train_start_dt = pd.to_datetime(train_start, utc=True)
        train_end_dt = pd.to_datetime(train_end, utc=True)
        train_mask = (volume_data.index >= train_start_dt) & (volume_data.index <= train_end_dt)
        train_volume = volume_data[train_mask]
        
        # 7SD capping using training statistics
        train_mean = train_volume.mean()
        train_std = train_volume.std()
        cap_upper = train_mean + (7 * train_std)
        cap_lower = train_mean - (7 * train_std)
        volume_capped = np.clip(volume_data, cap_lower, cap_upper)
        
        # Min-max scaling using training statistics
        train_volume_capped = volume_capped[train_mask]
        scaler = MinMaxScaler()
        scaler.fit(train_volume_capped.values.reshape(-1, 1))
        volume_scaled = pd.Series(
            scaler.transform(volume_capped.values.reshape(-1, 1)).flatten(),
            index=volume_data.index
        )
        
        return volume_scaled
    
    def create_unified_features(self, processed_data):
        """
        Combine all currency features into unified matrix
        """
        feature_list = []
        
        for pair in self.config.ALL_CURRENCY_PAIRS:
            pair_data = processed_data[pair]
            
            # Add OHLC features
            for col in ['Open', 'High', 'Low', 'Close']:
                feature_list.append(pair_data[col])
            
            # Add Volume feature
            feature_list.append(pair_data['Volume'])
        
        # Combine all features
        unified_features = pd.concat(feature_list, axis=1)
        unified_features.columns = [f"{pair}_{col}" for pair in self.config.ALL_CURRENCY_PAIRS 
                                   for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        unified_features.dropna(inplace=True)
        
        return unified_features
    
    def prepare_binary_target(self, close_prices):
        """
        Simple binary target: 1 if price goes up, 0 if price goes down
        """
        price_change = close_prices.pct_change().shift(-1)
        binary_target = (price_change > 0).astype(int)
        binary_target.dropna(inplace=True)
        
        return binary_target
    
    def create_sequences(self, features, target):
        """
        Create sequences for training
        """
        # Align features and target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        X, y = [], []
        
        for i in range(self.config.WINDOW_SIZE, len(features)):
            X.append(features.iloc[i-self.config.WINDOW_SIZE:i].values)
            y.append(target.iloc[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """
        Complete data preparation pipeline
        """
        print("📊 Loading and preprocessing data...")
        
        # Load raw data
        raw_data = self.load_currency_data()
        
        # Process each currency pair
        processed_data = {}
        for pair in self.config.ALL_CURRENCY_PAIRS:
            pair_data = raw_data[pair].copy()
            
            # Process OHLC
            ohlc_normalized = self.preprocess_ohlc(
                pair_data[['Open', 'High', 'Low', 'Close']], 
                self.config.TRAIN_START, 
                self.config.TRAIN_END
            )
            
            # Process Volume
            volume_scaled = self.preprocess_volume(
                pair_data['Volume'], 
                self.config.TRAIN_START, 
                self.config.TRAIN_END
            )
            
            # Combine processed data
            processed_pair = pd.concat([ohlc_normalized, volume_scaled.rename('Volume')], axis=1)
            processed_data[pair] = processed_pair
        
        # Create unified features
        unified_features = self.create_unified_features(processed_data)
        
        # Prepare target (use EURUSD as primary target)
        target = self.prepare_binary_target(raw_data['EURUSD']['Close'])
        
        # Create sequences
        X, y = self.create_sequences(unified_features, target)
        
        # Split into train/validation
        train_start = pd.to_datetime(self.config.TRAIN_START, utc=True)
        train_end = pd.to_datetime(self.config.TRAIN_END, utc=True)
        val_start = pd.to_datetime(self.config.VAL_START, utc=True)
        val_end = pd.to_datetime(self.config.VAL_END, utc=True)
        
        # Get timestamps for sequences
        timestamps = unified_features.index[self.config.WINDOW_SIZE:]
        
        # Create masks
        train_mask = (timestamps >= train_start) & (timestamps <= train_end)
        val_mask = (timestamps >= val_start) & (timestamps <= val_end)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        print(f"Training data: {X_train.shape}")
        print(f"Validation data: {X_val.shape}")
        print(f"Target distribution - UP: {y_train.mean():.1%}, DOWN: {(1-y_train.mean()):.1%}")
        
        return X_train, y_train, X_val, y_val