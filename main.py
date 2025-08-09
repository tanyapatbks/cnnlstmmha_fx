import warnings
import datetime
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from cnn_lstm_mha import PureCNNLSTMMHA, evaluate_predictions

def save_results(config, history, predictions, y_val, metrics):
    # Save training log
    log_path = f"{config.EXPERIMENT_PATH}/training_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Experiment: {config.EXPERIMENT_NAME}\n")
        f.write(f"Started: {datetime.datetime.now()}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final train loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"Final val loss: {history.history['val_loss'][-1]:.6f}\n")
        f.write(f"Best val loss: {min(history.history['val_loss']):.6f}\n")
        f.write(f"Final train accuracy: {history.history['accuracy'][-1]:.3f}\n")
        f.write(f"Final val accuracy: {history.history['val_accuracy'][-1]:.3f}\n")
        f.write(f"Best val accuracy: {max(history.history['val_accuracy']):.3f}\n")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'actual': y_val,
        'predicted_prob': predictions.flatten(),
        'predicted_class': (predictions > 0.5).astype(int).flatten()
    })
    pred_df.to_csv(f"{config.EXPERIMENT_PATH}/predictions.csv", index=False)
    
    # Save summary
    summary_path = f"{config.EXPERIMENT_PATH}/summary.txt"
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Experiment: {config.EXPERIMENT_NAME}\n")
        f.write(f"Architecture: CNN-LSTM-MHA\n")
        f.write(f"Task: Binary Classification [0,1]\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n")
        f.write(f"Recall: {metrics['recall']:.3f}\n")
        f.write(f"Prediction Range: [{metrics['predictions_range'][0]:.3f}, {metrics['predictions_range'][1]:.3f}]\n")
        f.write(f"Prediction Mean: {metrics['predictions_mean']:.3f}\n")
        
        # Signal distribution
        signals = []
        for prob in predictions.flatten():
            if prob < 0.4:
                signals.append("SELL")
            elif prob > 0.6:
                signals.append("BUY")
            else:
                signals.append("HOLD")
        
        signal_counts = pd.Series(signals).value_counts()
        f.write(f"\nSIGNAL DISTRIBUTION\n")
        f.write("-" * 20 + "\n")
        total = len(signals)
        for signal in ['SELL', 'HOLD', 'BUY']:
            count = signal_counts.get(signal, 0)
            f.write(f"{signal}: {count} ({count/total*100:.1f}%)\n")

def main():
    print("🚀 PURE CNN-LSTM-MHA FOREX PREDICTION")
    print("=" * 50)
    print("Pure Historical Data Training")
    print("Binary Classification: 0=Down, 1=Up")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        config.print_summary()
        
        # Prepare data
        processor = DataProcessor(config)
        X_train, y_train, X_val, y_val = processor.prepare_data()
        
        # Build and train model
        model = PureCNNLSTMMHA(config)
        print(f"Model parameters: {model.model.count_params():,}")
        
        # Train
        history = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        predictions = model.predict(X_val)
        metrics = evaluate_predictions(predictions, y_val)
        
        # Save model
        model_path = model.save_model()
        
        # Save results
        save_results(config, history, predictions, y_val, metrics)
        
        # Print final results
        print("\n" + "=" * 50)
        print("✅ TRAINING COMPLETED")
        print("=" * 50)
        print(f"Experiment: {config.EXPERIMENT_NAME}")
        print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"Prediction Range: [{metrics['predictions_range'][0]:.3f}, {metrics['predictions_range'][1]:.3f}]")
        print(f"Results saved: {config.EXPERIMENT_PATH}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()