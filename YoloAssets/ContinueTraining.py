from ultralytics import YOLO
import os
import shutil
import yaml
import argparse
from datetime import datetime
import glob

def find_latest_checkpoint():
    """Find the latest checkpoint in the training directory"""
    training_dir = "YoloAssets/runs/train/fire_detection_model/weights"
    
    if not os.path.exists(training_dir):
        return None
    
    # Look for last.pt (most recent checkpoint) first
    last_checkpoint = os.path.join(training_dir, "last.pt")
    if os.path.exists(last_checkpoint):
        return last_checkpoint
    
    # If last.pt doesn't exist, look for best.pt
    best_checkpoint = os.path.join(training_dir, "best.pt")
    if os.path.exists(best_checkpoint):
        return best_checkpoint
    
    # Look for any epoch checkpoint files (epoch*.pt)
    epoch_checkpoints = glob.glob(os.path.join(training_dir, "epoch*.pt"))
    if epoch_checkpoints:
        # Sort by modification time and return the latest
        epoch_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return epoch_checkpoints[0]
    
    return None

def get_checkpoint_info(checkpoint_path):
    """Get information about a checkpoint"""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        epoch = checkpoint.get('epoch', 0)
        best_fitness = checkpoint.get('best_fitness', 0)
        
        return {
            'epoch': epoch,
            'best_fitness': best_fitness,
            'path': checkpoint_path
        }
    except Exception as e:
        print(f"[WARNING] Could not read checkpoint info: {e}")
        return {'epoch': 0, 'best_fitness': 0, 'path': checkpoint_path}

def save_continued_model_to_archive(model_name, original_epochs, continued_epochs, start_epoch):
    """Archive continued training results"""
    # Note: Training results are automatically saved by YOLO in the proper location
    # No need to copy files as YOLO manages training outputs automatically
    current_time = datetime.now()
    datetime_str = current_time.strftime("%Y%m%d_%H%M%S")
    total_epochs = original_epochs + continued_epochs
    folder_name = f"{model_name}_E{total_epochs}_Continued_from_{start_epoch}_{datetime_str}"
    
    print(f"\n✅ Continued training completed: {folder_name}")
    print(f"[INFO] Training results are managed by YOLO in the training directory")
    return folder_name

def add_training_callbacks(model, start_epoch=0):
    """Add callbacks for training status monitoring with epoch offset"""
    
    def on_train_start(trainer):
        """Called when training starts"""
        print("[STATUS] TRAINING_STARTED")
        print(f"[INFO] Resuming training from epoch {start_epoch + 1} for {trainer.epochs} more epochs")
    
    def on_train_epoch_start(trainer):
        """Called at the start of each epoch"""
        current_epoch = trainer.epoch + 1 + start_epoch
        print(f"[STATUS] EPOCH_STARTED:{current_epoch}")
    
    def on_train_epoch_end(trainer):
        """Called at the end of each epoch"""
        current_epoch = trainer.epoch + 1 + start_epoch
        total_epochs = trainer.epochs + start_epoch
        progress = (current_epoch / total_epochs) * 100
        
        print(f"[STATUS] EPOCH_COMPLETED:{current_epoch}")
        print(f"[PROGRESS] {progress:.1f}")
        print(f"[INFO] Epoch {current_epoch}/{total_epochs} completed")
        
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
        print("[INFO] Continued training finished successfully")
    
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
        
        parser = argparse.ArgumentParser(description='Continue training YOLO model for fire detection')
        parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from (optional, will auto-detect if not provided)')
        parser.add_argument('--epochs', type=int, default=50, help='Number of additional epochs to train (default: 50)')
        parser.add_argument('--model', type=str, help='Original model name for archiving purposes (e.g., yolo12n)')
        
        args = parser.parse_args()
        
        # Find checkpoint to resume from
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            checkpoint_path = find_latest_checkpoint()
            
        if not checkpoint_path:
            print("[ERROR] No checkpoint found to resume from")
            print("[ERROR] Please run initial training first or provide a valid checkpoint path")
            print("[STATUS] FAILED")
            exit(1)
        
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            print("[STATUS] FAILED")
            exit(1)
        
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        
        # Get checkpoint information
        checkpoint_info = get_checkpoint_info(checkpoint_path)
        start_epoch = checkpoint_info['epoch']
        
        print(f"[INFO] Checkpoint was at epoch: {start_epoch}")
        print(f"[INFO] Will train for {args.epochs} additional epochs")
        
        print("[STATUS] LOADING_MODEL")
        model = YOLO(checkpoint_path)
        print("[STATUS] MODEL_LOADED")
        
        # Add callbacks for status monitoring
        add_training_callbacks(model, start_epoch)
        
        # Read config to update epochs and get original configuration
        config_path = 'YoloAssets/train_config_minimal.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        original_epochs = config.get('epochs', 100)
        
        # Update config for continued training
        config['epochs'] = args.epochs  # Additional epochs to train
        config['resume'] = True  # Enable resume mode in YOLO
        
        # Write temporary config for continued training
        continue_config_path = 'YoloAssets/train_config_continue.yaml'
        with open(continue_config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        print("[STATUS] TRAINING_STARTING")
        print(f"[INFO] Training configuration updated for continued training")
        
        # Continue training
        model.train(cfg=continue_config_path, resume=True)
        
        print("[STATUS] VALIDATION_STARTING")
        metrics = model.val()
        
        print("[STATUS] ARCHIVING")
        model_name = args.model if args.model else "unknown_model"
        archive_path = save_continued_model_to_archive(
            model_name, 
            original_epochs, 
            args.epochs, 
            start_epoch
        )
        
        # Clean up temporary config
        if os.path.exists(continue_config_path):
            os.remove(continue_config_path)
        
        if archive_path:
            print("[STATUS] SUCCESS")
            print(f"[INFO] Continued training completed successfully")
            print(f"[INFO] Total epochs trained: {start_epoch + args.epochs}")
        else:
            print("[STATUS] FAILED")
            print("[ERROR] Failed to archive continued training results")
            
    except KeyboardInterrupt:
        print("[STATUS] CANCELLED")
        print("[ERROR] Continued training cancelled by user")
        # Clean up temporary config if it exists
        continue_config_path = 'YoloAssets/train_config_continue.yaml'
        if os.path.exists(continue_config_path):
            os.remove(continue_config_path)
    except Exception as e:
        print("[STATUS] FAILED")
        print(f"[ERROR] {str(e)}")
        # Clean up temporary config if it exists
        continue_config_path = 'YoloAssets/train_config_continue.yaml'
        if os.path.exists(continue_config_path):
            os.remove(continue_config_path)