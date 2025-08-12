from ultralytics import YOLO
import os
import shutil
import yaml
import argparse
from datetime import datetime

def clear_training_directory():
    """Clear YoloAssets/runs/train directory"""
    train_dir = "YoloAssets/runs/train"
    if os.path.exists(train_dir):
        try:
            shutil.rmtree(train_dir)
            print(f"Cleared training directory: {train_dir}")
        except Exception as e:
            print(f"Warning: Could not clear training directory: {e}")

def save_trained_model_to_archive(model_name, epochs):
    """Archive training results to TrainedModels/{model}_E{epochs}_{datetime}"""
    current_time = datetime.now()
    datetime_str = current_time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_name}_E{epochs}_{datetime_str}"
    
    source_path = os.path.join(os.getcwd(), 'YoloAssets/runs/train/fire_detection_model')
    dest_base = os.path.join(os.getcwd(), 'TrainedModels')
    dest_path = os.path.join(dest_base, folder_name)
    
    os.makedirs(dest_base, exist_ok=True)
    
    if os.path.exists(source_path):
        try:
            shutil.copytree(source_path, dest_path)
            print(f"\n✅ Training archived to: {dest_path}")
            return dest_path
        except Exception as e:
            print(f"❌ Error archiving training: {e}")
            return None
    else:
        print(f"❌ Training folder not found: {source_path}")
        return None

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
        
        parser = argparse.ArgumentParser(description='Train YOLO model for fire detection')
        parser.add_argument('model', help='YOLO model to use (e.g., yolo12n, yolo12s, yolo12m, yolo12l, yolo12x)')
        args = parser.parse_args()
        
        model_name = args.model
        model_file = f"{model_name}.pt"
        
        print(f"[INFO] Using model: {model_file}")
        clear_training_directory()
        
        print("[STATUS] LOADING_MODEL")
        model = YOLO(model_file)
        print("[STATUS] MODEL_LOADED")
        
        # Add callbacks for status monitoring
        add_training_callbacks(model)
        
        # Read config to get epochs info
        with open('YoloAssets/train_config_minimal.yaml', 'r') as f:
            config = yaml.safe_load(f)
        epochs = config.get('epochs', 1)
        print(f"[INFO] Configured for {epochs} epochs")

        print("[STATUS] TRAINING_STARTING")
        model.train(cfg='YoloAssets/train_config_minimal.yaml')

        print("[STATUS] VALIDATION_STARTING")
        metrics = model.val()

        print("[STATUS] ARCHIVING")
        archive_path = save_trained_model_to_archive(model_name, epochs)
        
        if archive_path:
            print("[STATUS] SUCCESS")
            print(f"[INFO] Training completed successfully")
        else:
            print("[STATUS] FAILED")
            print("[ERROR] Failed to archive training results")

    except KeyboardInterrupt:
        print("[STATUS] CANCELLED")
        print("[ERROR] Training cancelled by user")
    except Exception as e:
        print("[STATUS] FAILED")
        print(f"[ERROR] {str(e)}")