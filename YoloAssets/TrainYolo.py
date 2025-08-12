#!/usr/bin/env python3
"""
YOLO Training Script with Command Line Arguments
Usage: python TrainYolo.py --Epoch 100 --Model 12 --Size m
"""

from ultralytics import YOLO
import os
import shutil
import argparse
from datetime import datetime
import sys
import json
import socket
import threading
from ultralytics.utils.callbacks import base

def validate_arguments(args):
    """Validate command line arguments"""
    # Validate YOLO version
    if args.Model not in [10, 11, 12]:
        raise ValueError(f"Invalid YOLO model version: {args.Model}. Acceptable values are: 10, 11, 12")
    
    # Validate model size
    if args.Size not in ['n', 's', 'm', 'l', 'x']:
        raise ValueError(f"Invalid model size: {args.Size}. Acceptable values are: n, s, m, l, x")
    
    # Validate epochs
    if args.Epoch < 1:
        raise ValueError(f"Invalid epoch count: {args.Epoch}. Must be greater than 0")
    
    return True

def get_model_filename(yolo_version, model_size):
    """Generate the model filename based on version and size"""
    return f"yolo{yolo_version}{model_size}.pt"

def generate_model_name(yolo_version, model_size, epochs):
    """Generate model name in the format: Yolo{version}{size}_E{epochs}_{datetime}"""
    current_time = datetime.now()
    datetime_str = current_time.strftime("%Y%m%d_%H%M%S")
    return f"Yolo{yolo_version}{model_size}_E{epochs}_{datetime_str}"

def save_best_model(model_name, source_path="runs/train/fire_detection_model/weights/best.pt"):
    """Save the best trained model to YoloAssets/BestModels/{ModelName}"""
    # Create the BestModels directory structure
    best_models_dir = os.path.join("YoloAssets", "BestModels", model_name)
    os.makedirs(best_models_dir, exist_ok=True)
    
    # Copy best.pt model
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Best model file not found at: {source_path}")
    
    dest_model_path = os.path.join(best_models_dir, "best.pt")
    shutil.copy2(source_path, dest_model_path)
    
    # Copy data.yaml (dataset config)
    dataset_config_source = "dataset/data.yaml"
    if os.path.exists(dataset_config_source):
        shutil.copy2(dataset_config_source, os.path.join(best_models_dir, "data.yaml"))
    else:
        # Fallback to YoloAssets/dataset/data.yaml
        dataset_config_source = "YoloAssets/dataset/data.yaml"
        if os.path.exists(dataset_config_source):
            shutil.copy2(dataset_config_source, os.path.join(best_models_dir, "data.yaml"))
        else:
            print("Warning: data.yaml not found in expected locations")
    
    # Create a training info file
    info_file_path = os.path.join(best_models_dir, "training_info.txt")
    with open(info_file_path, 'w') as f:
        f.write(f"Training Information\n")
        f.write(f"==================\n\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model File: best.pt\n")
        f.write(f"Dataset Config: data.yaml\n\n")
        f.write(f"Training completed successfully!\n")
    
    print(f"\nBest model saved to: {best_models_dir}")
    print("Files saved: best.pt, data.yaml, training_info.txt")
    return best_models_dir

class ProgressReporter:
    """Class to handle real-time progress reporting to the dashboard"""
    
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        self.socket = None
        
    def connect(self):
        """Connect to the dashboard server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            return True
        except Exception as e:
            print(f"Could not connect to dashboard server: {e}")
            return False
    
    def send_update(self, data):
        """Send training update to dashboard"""
        if self.socket:
            try:
                message = json.dumps(data) + '\n'
                self.socket.send(message.encode())
            except Exception as e:
                print(f"Error sending update: {e}")
    
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()

def add_training_callbacks(model, progress_reporter=None):
    """Add callbacks to YOLO model for progress reporting"""
    
    def on_train_epoch_end(trainer):
        """Called at the end of each training epoch"""
        if progress_reporter:
            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            
            # Get metrics from trainer
            metrics = {}
            if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                metrics['loss'] = float(trainer.tloss)
            if hasattr(trainer, 'metrics') and trainer.metrics:
                if hasattr(trainer.metrics, 'box') and hasattr(trainer.metrics.box, 'map'):
                    metrics['map'] = float(trainer.metrics.box.map)
            
            progress = (epoch / total_epochs) * 100
            
            update_data = {
                'type': 'epoch_end',
                'epoch': epoch,
                'total_epochs': total_epochs,
                'progress': round(progress, 1),
                **metrics
            }
            
            progress_reporter.send_update(update_data)
        
        print(f"Epoch {epoch}/{trainer.epochs} completed - Best fitness: {trainer.best_fitness}")
    
    def on_train_start(trainer):
        """Called at the start of training"""
        if progress_reporter:
            progress_reporter.send_update({
                'type': 'train_start',
                'total_epochs': trainer.epochs,
                'message': 'Training started...'
            })
        print("Training started...")
    
    def on_train_end(trainer):
        """Called at the end of training"""
        if progress_reporter:
            progress_reporter.send_update({
                'type': 'train_end',
                'message': 'Training completed!'
            })
        print("Training completed!")
    
    # Add callbacks to model
    model.add_callback('on_train_epoch_end', on_train_epoch_end)
    model.add_callback('on_train_start', on_train_start)
    model.add_callback('on_train_end', on_train_end)

def main():
    """Main training function"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="YOLO Fire Detection Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--Model", 
        type=int, 
        required=True,
        help="YOLO version (10, 11, or 12)"
    )
    
    parser.add_argument(
        "--Size", 
        type=str, 
        required=True,
        help="Model size (n, s, m, l, or x)"
    )
    
    parser.add_argument(
        "--Epoch", 
        type=int, 
        required=True,
        help="Number of training epochs (must be >= 1)"
    )
    
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640,
        help="Image size for training"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use for training (cuda, cpu, or mps)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_arguments(args)
        
        # Generate model filename and name
        model_filename = get_model_filename(args.Model, args.Size)
        model_name = generate_model_name(args.Model, args.Size, args.Epoch)
        
        print("="*60)
        print("YOLO Fire Detection Training")
        print("="*60)
        print(f"Model: YOLO v{args.Model} ({args.Size})")
        print(f"Model File: {model_filename}")
        print(f"Epochs: {args.Epoch}")
        print(f"Batch Size: {args.batch}")
        print(f"Image Size: {args.imgsz}")
        print(f"Device: {args.device}")
        print(f"Output Model Name: {model_name}")
        print("="*60)
        
        # Check if model file exists
        model_paths_to_check = [
            model_filename,
            os.path.join("YoloAssets", model_filename),
            model_filename.replace(f"yolo{args.Model}", f"yolov{args.Model}")  # Alternative naming
        ]
        
        model_path = None
        for path in model_paths_to_check:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print(f"Model file not found. Tried paths:")
            for path in model_paths_to_check:
                print(f"  - {path}")
            print(f"\nYOLO will attempt to download {model_filename} automatically...")
            model_path = model_filename
        else:
            print(f"Using model file: {model_path}")
        
        # Load the YOLO model
        print(f"\nLoading YOLO model: {model_path}")
        model = YOLO(model_path)
        
        # Determine dataset path
        dataset_paths = [
            "dataset/data.yaml",
            "YoloAssets/dataset/data.yaml"
        ]
        
        dataset_path = None
        for path in dataset_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path is None:
            raise FileNotFoundError("Dataset configuration file (data.yaml) not found in expected locations")
        
        print(f"Using dataset: {dataset_path}")
        
        # Set up progress reporter for real-time updates
        progress_reporter = ProgressReporter()
        connected_to_dashboard = progress_reporter.connect()
        
        if connected_to_dashboard:
            print("Connected to dashboard for real-time updates")
        else:
            print("Dashboard connection failed - proceeding without real-time updates")
        
        # Add training callbacks to model
        add_training_callbacks(model, progress_reporter if connected_to_dashboard else None)
        
        # Train the model
        print(f"\nStarting training for {args.Epoch} epochs...")
        print(f"Model: {model_filename}")

        results = model.train(
            data=dataset_path,
            epochs=args.Epoch,
            imgsz=args.imgsz,
            batch=args.batch,
            name='fire_detection_model',
            device=args.device,
            project='runs/train'
        )
        
        # Close progress reporter connection
        if connected_to_dashboard:
            progress_reporter.close()
        
        # Validate the model
        print("\nValidating model...")
        metrics = model.val()
        
        # Print training results
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        print(f"Training results saved in: {os.path.join(os.getcwd(), 'runs/train/fire_detection_model')}")
        print(f"Best model weights saved in: {os.path.join(os.getcwd(), 'runs/train/fire_detection_model/weights/best.pt')}")
        
        # Save model to custom location with proper naming
        try:
            saved_location = save_best_model(model_name)
            print(f"\nModel successfully saved to: {saved_location}")
        except Exception as e:
            print(f"Error saving model to custom location: {str(e)}")
            return 1
        
        print("\n" + "="*60)
        print("TRAINING PROCESS COMPLETED SUCCESSFULLY!")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())