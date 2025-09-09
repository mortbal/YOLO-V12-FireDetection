from ultralytics import YOLO
import os
import shutil
import yaml
from datetime import datetime

def add_training_callbacks(model):
    """Add callbacks for training status monitoring"""
    
    def on_train_start(trainer):
        """Called when training starts"""
        print("[STATUS] TRAINING_STARTED")
        print(f"[INFO] Starting training for {trainer.epochs} epochs")
    
    def on_train_epoch_start(trainer):
        """Called at the start of each epoch"""
        epoch = trainer.epoch + 1
        print(f"[STATUS] EPOCH_STARTED:{epoch}")
    
    def on_train_epoch_end(trainer):
        """Called at the end of each epoch"""
        epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        progress = (epoch / total_epochs) * 100
        
        print(f"[STATUS] EPOCH_COMPLETED:{epoch}")
        print(f"[PROGRESS] {progress:.1f}")
        print(f"[INFO] Epoch {epoch}/{total_epochs} completed")
        
        # Extract metrics if available
        try:
            if hasattr(trainer, 'metrics') and trainer.metrics:
                if hasattr(trainer.metrics, 'box') and hasattr(trainer.metrics.box, 'map'):
                    map_value = trainer.metrics.box.map
                    if hasattr(map_value, 'item'):
                        print(f"[METRIC] MAP:{map_value.item():.4f}")
                    else:
                        print(f"[METRIC] MAP:{float(map_value):.4f}")
        except Exception:
            pass
    
    def on_train_end(trainer):
        """Called when training completes successfully"""
        print("[STATUS] TRAINING_COMPLETED")
        print("[INFO] Training finished successfully")
    
    def on_val_start(trainer):
        """Called when validation starts"""
        print("[STATUS] VALIDATION_STARTED")
    
    def on_val_end(trainer):
        """Called when validation ends"""
        print("[STATUS] VALIDATION_COMPLETED")
    
    # Add callbacks to model
    model.add_callback('on_train_start', on_train_start)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    model.add_callback('on_train_epoch_end', on_train_epoch_end)
    model.add_callback('on_train_end', on_train_end)
    model.add_callback('on_val_start', on_val_start)
    model.add_callback('on_val_end', on_val_end)

if __name__ == "__main__":
    try:
        print("[STATUS] INITIALIZING")
        
        # Read all configuration from config file
        config_path = 'YoloAssets/train_config_minimal.yaml'
        print(f"[INFO] Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get model path from config
        model_file = config.get('model', 'yolo11n.pt')
        model_type = config.get('model_type', 'base')
        epochs = config.get('epochs', 100)
        
        print(f"[INFO] Using model: {model_file} (type: {model_type})")
        print(f"[INFO] Training for {epochs} epochs")
        
        # Check if model file exists
        if not os.path.exists(model_file):
            print(f"[ERROR] Model file not found: {model_file}")
            # Try to find the model file in the expected location based on type
            if model_type == 'base':
                alt_path = f"YoloAssets/BaseModels/{os.path.basename(model_file)}"
                if os.path.exists(alt_path):
                    model_file = alt_path
                    print(f"[INFO] Found model at alternative path: {model_file}")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_file}")
        
        print("[STATUS] LOADING_MODEL")
        model = YOLO(model_file)
        print("[STATUS] MODEL_LOADED")
        
        # Add callbacks for status monitoring
        add_training_callbacks(model)

        print("[STATUS] TRAINING_STARTING")
        model.train(cfg=config_path)

        print("[STATUS] SUCCESS")
        print(f"[INFO] Training completed successfully")

    except KeyboardInterrupt:
        print("[STATUS] CANCELLED")
        print("[ERROR] Training cancelled by user")
    except Exception as e:
        print("[STATUS] FAILED")
        print(f"[ERROR] {str(e)}")