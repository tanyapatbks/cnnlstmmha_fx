import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization,
    MultiHeadAttention, LayerNormalization, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

class PureCNNLSTMMHA:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()
        
    def build_model(self):
        # Input layer
        inputs = Input(shape=(self.config.WINDOW_SIZE, self.config.TOTAL_FEATURES))
        
        # CNN Feature Extraction
        x = Conv1D(self.config.CNN_FILTERS[0], self.config.CNN_KERNEL_SIZE, 
                   padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        
        x = Conv1D(self.config.CNN_FILTERS[1], self.config.CNN_KERNEL_SIZE, 
                   padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # LSTM Temporal Processing
        x = LSTM(self.config.LSTM_UNITS[0], return_sequences=True, 
                 dropout=self.config.DROPOUT_RATE)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(self.config.LSTM_UNITS[1], return_sequences=True, 
                 dropout=self.config.DROPOUT_RATE)(x)
        x = BatchNormalization()(x)
        
        # Multi-Head Attention
        mha_out = MultiHeadAttention(
            num_heads=self.config.MHA_HEADS, 
            key_dim=self.config.MHA_KEY_DIM, 
            dropout=0.1
        )(x, x)
        
        # Residual Connection + Layer Normalization
        attended = LayerNormalization()(x + mha_out)
        
        # Attention Pooling
        attention_weights = Dense(1, activation='tanh')(attended)
        attention_weights = Lambda(lambda x: tf.nn.softmax(x, axis=1))(attention_weights)
        pooled = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([attended, attention_weights])
        
        # Dense layers
        output = Dense(self.config.DENSE_UNITS[0], activation='relu')(pooled)
        output = BatchNormalization()(output)
        output = Dropout(self.config.DROPOUT_RATE)(output)
        
        output = Dense(self.config.DENSE_UNITS[1], activation='relu')(output)
        output = Dropout(0.2)(output)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(output)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        print("🎯 Training model...")
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=self.config.REDUCE_LR_PATIENCE,
                factor=0.5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=f"{self.config.EXPERIMENT_PATH}/best_model.keras",
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self):
        model_path = f"{self.config.EXPERIMENT_PATH}/final_model.keras"
        self.model.save(model_path)
        return model_path

def evaluate_predictions(predictions, y_actual):
    pred_binary = (predictions > 0.5).astype(int).flatten()
    actual_binary = y_actual.astype(int)
    
    accuracy = (pred_binary == actual_binary).mean()
    
    # Calculate confusion matrix
    tp = ((pred_binary == 1) & (actual_binary == 1)).sum()
    tn = ((pred_binary == 0) & (actual_binary == 0)).sum()
    fp = ((pred_binary == 1) & (actual_binary == 0)).sum()
    fn = ((pred_binary == 0) & (actual_binary == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'predictions_range': [predictions.min(), predictions.max()],
        'predictions_mean': predictions.mean()
    }