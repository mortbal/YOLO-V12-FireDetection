from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
import subprocess
import sys
import os
import yaml
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import base64
import torch
from ultralytics import YOLO

# Flask configuration
static_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../static"))
app = Flask(__name__, static_folder=static_dir, static_url_path='/static')

socketio = SocketIO(app, cors_allowed_origins="*")

# Training state variables
training_start_time = None
epoch_start_times = {}
last_epoch_duration = None
total_elapsed_timer = None

# Live detection state variables
live_detection_active = False
live_detection_model = None
live_detection_thread = None

frontend_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../frontend"))

@app.route("/")
def serve_index():
    return send_from_directory(frontend_dir, "index.html")

def update_training_config(epochs, model_info=None):
    """Update training config with epochs and model information"""
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../YoloAssets/train_config_minimal.yaml")
    )
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update epochs
        config['epochs'] = epochs
        
        # Update model information if provided
        if model_info:
            model_type = model_info.get('type', 'base')
            model_name = model_info.get('name', 'yolo11n.pt')
            
            if model_type == 'base':
                # For base models, use the model name directly from BaseModels folder
                config['model'] = os.path.join('YoloAssets/BaseModels', model_name)
            elif model_type == 'trained':
                # For trained models, use the best.pt from the trained model folder
                config['model'] = os.path.join('YoloAssets/TrainedModels', model_name, 'best.pt')
            elif model_type == 'checkpoint':
                # For checkpoints, use the last.pt or best.pt from checkpoints folder
                config['model'] = os.path.join('YoloAssets/TrainCheckpoints', model_name, 'last.pt')
            
            config['model_type'] = model_type
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True
    except Exception as e:
        print(f"Error updating training config: {e}")
        return False

@app.route("/train", methods=["POST"])
def run_training():
    
    try:
        data = request.get_json()
        selected_model = data.get("model", "")
        epochs = data.get("epochs", 100)
        
        # Parse the selected model format: "type:modelname"
        if ":" in selected_model:
            model_type, model_name = selected_model.split(":", 1)
        else:
            # Fallback for backward compatibility
            yolo_version = data.get("yolo_version", 12)
            model_size = data.get("model_size", "n")
            model_name = f"yolo{yolo_version}{model_size}.pt"
            model_type = "base"
        
        # Prepare model info for config update
        model_info = {
            'type': model_type,
            'name': model_name
        }
        
        # Update training config with epochs and model info
        if not update_training_config(epochs, model_info):
            return jsonify({"status": "error", "message": "Failed to update training config"}), 500
        
        script_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../YoloAssets/TrainYolo.py")
        )
        
        
        global training_start_time
        training_start_time = time.time()
        
        start_total_elapsed_timer()
        
        emit_status_update("Preparing for train : Loading YOLO...")
        
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=os.path.join(os.path.dirname(__file__), "../..")
        )
        
        def monitor_regular_training_output():
            global training_start_time, epoch_start_times, last_epoch_duration
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:

                    if "[STATUS] INITIALIZING" in line:
                        emit_status_update("Preparing for train : Initializing...")
                        
                    elif "[STATUS] TRAINING_STARTING" in line:
                        emit_status_update("Preparing for train : Warming up...")
                        
                    elif "Fast image access" in line:
                        emit_status_update("Preparing for train : Discovering Dataset...")
                        
                    elif "[STATUS] TRAINING_STARTED" in line:
                        emit_status_update("Training")
                        total_elapsed = time.time() - training_start_time
                        socketio.emit('total_elapsed_update', {
                            'total_elapsed_seconds': total_elapsed
                        })
                        
                    elif "G      " in line and "/" in line and "%" in line:
                        try:
                            parts = line.split()
                            
                            epoch_info = parts[0] if "/" in parts[0] else None
                            current_epoch, total_epochs = map(int, epoch_info.split('/')) if epoch_info else (0, 0)
                            
                            gpu_memory = parts[1] if len(parts) > 1 and 'G' in parts[1] else None
                            
                            progress_percent = 0
                            current_batch = 0
                            total_batches = 0
                            iteration_speed = None
                            elapsed_time = None
                            remaining_time = None
                            
                            for i, part in enumerate(parts):
                                if '%' in part and '|' in part:
                                    progress_percent = float(part.split('%')[0])
                                    
                                    if i + 1 < len(parts) and '/' in parts[i + 1]:
                                        batch_info = parts[i + 1].replace('|', '').strip()
                                        if '/' in batch_info:
                                            current_batch, total_batches = map(int, batch_info.split('/'))
                                    
                                    for j in range(i, len(parts)):
                                        if '[' in parts[j] and '<' in parts[j]:
                                            time_part = parts[j].replace('[', '').replace(']', '')
                                            if '<' in time_part:
                                                elapsed_time, remaining_time = time_part.split('<')
                                                remaining_time = remaining_time.replace(',', '').strip()
                                        elif 's/it]' in parts[j]:
                                            iteration_speed = parts[j].replace('s/it]', '').replace(',', '').strip()
                                    break
                            
                            
                            if current_epoch not in epoch_start_times:
                                epoch_start_times[current_epoch] = time.time()
                            
                            total_elapsed = time.time() - training_start_time
                            
                            if remaining_time and ('?' in str(remaining_time)):
                                remaining_time = '00:00'
                            
                            total_remaining = None
                            if remaining_time and remaining_time in ['00:00', '0:00'] and elapsed_time:
                                epoch_duration_seconds = parse_time_to_seconds(elapsed_time)
                                if epoch_duration_seconds > 0:
                                    last_epoch_duration = epoch_duration_seconds
                                    remaining_epochs = total_epochs - current_epoch
                                    total_remaining = remaining_epochs * last_epoch_duration
                            
                            socketio.emit('training_update', {
                                'epoch': current_epoch,
                                'total_epochs': total_epochs,
                                'progress': progress_percent,
                                'batch_progress': progress_percent,
                                'current_batch': current_batch,
                                'total_batches': total_batches,
                                'gpu_memory': gpu_memory,
                                'iteration_speed': iteration_speed,
                                'elapsed_time': elapsed_time,
                                'remaining_time': remaining_time,
                                'total_elapsed_seconds': total_elapsed,
                                'total_remaining_seconds': total_remaining
                            })
                            
                        except Exception as e:
                            pass
                        
                    elif "[STATUS] MODEL_LOADED" in line:
                        pass
                    elif "[STATUS] SUCCESS" in line:
                        total_elapsed = time.time() - training_start_time if training_start_time else 0
                        stop_total_elapsed_timer()
                        emit_status_update("Training Complete!")
                        socketio.emit('training_complete', {
                            'message': 'Training completed successfully!',
                            'total_time': format_seconds_to_time(total_elapsed)
                        })
                        
                    elif "[STATUS] FAILED" in line:
                        total_elapsed = time.time() - training_start_time if training_start_time else 0
                        stop_total_elapsed_timer()
                        emit_status_update("Training Failed")
                        socketio.emit('training_error', {
                            'error': 'Training process failed',
                            'total_time': format_seconds_to_time(total_elapsed)
                        })
        
        monitor_thread = threading.Thread(target=monitor_regular_training_output)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return jsonify({
            "status": "training_started", 
            "model": model_name,
            "epochs": epochs,
            "message": f"Started training {model_name} for {epochs} epochs"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Timer and status utility functions

def parse_time_to_seconds(time_str):
    """Parse time strings to seconds"""
    if not time_str or time_str == '-':
        return 0
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0]) or 0
            seconds = int(parts[1]) or 0
            return minutes * 60 + seconds
    except:
        pass
    return 0

def format_seconds_to_time(seconds):
    """Format seconds to time string"""
    if seconds <= 0:
        return "00:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def start_total_elapsed_timer():
    """Start elapsed time broadcast timer"""
    global total_elapsed_timer, training_start_time
    
    if total_elapsed_timer:
        total_elapsed_timer.cancel()
    
    def send_elapsed_update():
        global total_elapsed_timer, training_start_time
        if training_start_time:
            total_elapsed = time.time() - training_start_time
            socketio.emit('total_elapsed_update', {
                'total_elapsed_seconds': total_elapsed,
                'formatted_time': format_seconds_to_time(total_elapsed)
            })
            
            total_elapsed_timer = threading.Timer(1.0, send_elapsed_update)
            total_elapsed_timer.daemon = True
            total_elapsed_timer.start()
    
    send_elapsed_update()

def stop_total_elapsed_timer():
    """Stop elapsed time timer"""
    global total_elapsed_timer
    if total_elapsed_timer:
        total_elapsed_timer.cancel()
        total_elapsed_timer = None

def emit_status_update(status_message):
    """Send status to clients"""
    try:
        socketio.emit('training_status_update', {'status': status_message})
    except Exception as e:
        pass

@socketio.on('connect')
def handle_connect():
    pass

@socketio.on('disconnect')
def handle_disconnect():
    pass

@app.route("/browse_folder", methods=["POST"])
def browse_folder():
    """Handle folder browsing requests"""
    try:
        data = request.get_json()
        folder_type = data.get('type', '')
        
        # Create a hidden tkinter root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring to front
        
        # Open folder dialog
        if folder_type == 'test':
            title = "Select Test Files Folder"
            initial_dir = "../TestFiles"
        elif folder_type == 'result':
            title = "Select Results Output Folder"
            initial_dir = "../ResultFiles"
        else:
            title = "Select Folder"
            initial_dir = ".."
        
        # Make sure initial directory exists
        if not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")  # Default to home directory
        
        folder_path = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir
        )
        
        # Clean up the tkinter root
        root.destroy()
        
        if folder_path:
            return jsonify({
                "status": "success",
                "folder_path": folder_path
            })
        else:
            return jsonify({
                "status": "cancelled",
                "message": "No folder selected"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to open folder dialog: {str(e)}"
        }), 500

@app.route("/get_all_models", methods=["GET"])
def get_all_models():
    """Get list of all available models from three sources"""
    try:
        result = {
            "baseModels": [],
            "trainedModels": [],
            "checkpointModels": []
        }
        
        # 1. Base models from YoloAssets/BaseModels
        base_models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../YoloAssets/BaseModels"))
        if os.path.exists(base_models_dir):
            for item in os.listdir(base_models_dir):
                if item.endswith('.pt'):
                    result["baseModels"].append(item)
        
        # 2. Trained models from YoloAssets/TrainedModels (folders)
        trained_models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../YoloAssets/TrainedModels"))
        if os.path.exists(trained_models_dir):
            for item in os.listdir(trained_models_dir):
                model_path = os.path.join(trained_models_dir, item)
                if os.path.isdir(model_path):
                    result["trainedModels"].append(item)
        
        # 3. Checkpoint models from YoloAssets/TrainCheckpoints (folders)
        checkpoint_models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../YoloAssets/TrainCheckpoints"))
        if os.path.exists(checkpoint_models_dir):
            for item in os.listdir(checkpoint_models_dir):
                model_path = os.path.join(checkpoint_models_dir, item)
                if os.path.isdir(model_path):
                    result["checkpointModels"].append(item)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"baseModels": [], "trainedModels": [], "checkpointModels": []})

@app.route("/get_models", methods=["GET"])
def get_models():
    """Get list of available trained models (for backward compatibility)"""
    try:
        # Look for models in TrainedModels directory
        models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../YoloAssets/TrainedModels"))
        
        if not os.path.exists(models_dir):
            return jsonify([])
        
        model_names = []
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path):
                model_names.append(item)
        
        return jsonify(model_names)
        
    except Exception as e:
        return jsonify([])

@app.route("/run_detection", methods=["POST"])
def run_detection():
    """Handle detection requests"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        test_folder = data.get('test_folder')
        output_folder = data.get('output_folder')
        
        if not model_name or not test_folder or not output_folder:
            return jsonify({
                "status": "error",
                "message": "Missing required parameters"
            }), 400
        
        # Build model path
        model_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), 
            f"../../TrainedModels/{model_name}/weights/best.pt"
        ))
        
        
        if not os.path.exists(model_path):
            return jsonify({
                "status": "error",
                "message": f"Model not found: {model_name} at path: {model_path}"
            }), 404
        
        # Start real detection process
        socketio.emit('detection_update', {
            'message': f'🚀 Starting detection with model: {model_name}'
        })
        
        socketio.emit('detection_update', {
            'message': f'📁 Processing files from: {test_folder}'
        })
        
        # Run actual detection script
        def run_detection_process():
            try:
                script_path = os.path.normpath(os.path.join(
                    os.path.dirname(__file__), 
                    "../../YoloAssets/RunDetection.py"
                ))
                
                # Build command arguments with GPU optimization
                cmd = [
                    sys.executable, script_path,
                    '--model', model_path,
                    '--source', test_folder,
                    '--output', output_folder,
                    '--conf', '0.5',
                    '--device', 'cuda',  # Force GPU usage
                    '--imgsz', '640',    # Standard image size
                    '--half',            # Use FP16 for faster inference
                    '--batch-size', '1'  # Batch size
                ]
                
                socketio.emit('detection_update', {
                    'message': '🔄 Launching detection script...'
                })
                
                # Start subprocess
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    cwd=os.path.join(os.path.dirname(__file__), "../..")
                )
                
                # Monitor output in real-time
                processed_files = 0
                total_files = 0
                
                for line in iter(process.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        print(f"Detection: {line}")  # Debug logging
                        
                        # Parse status messages
                        if "[STATUS] INITIALIZING" in line:
                            socketio.emit('detection_update', {
                                'message': '⚙️ Initializing detection system...'
                            })
                        elif "[STATUS] LOADING_MODEL" in line:
                            socketio.emit('detection_update', {
                                'message': '🤖 Loading AI model...'
                            })
                        elif "[STATUS] MODEL_LOADED" in line:
                            socketio.emit('detection_update', {
                                'message': '✅ Model loaded successfully'
                            })
                        elif "[INFO] Using device:" in line:
                            device = line.split("Using device: ")[1]
                            socketio.emit('detection_update', {
                                'message': f'⚡ Using device: {device}'
                            })
                        elif "[INFO] GPU:" in line:
                            gpu_info = line.split("GPU: ")[1]
                            socketio.emit('detection_update', {
                                'message': f'🎮 GPU: {gpu_info}'
                            })
                        elif "[INFO] GPU Memory:" in line:
                            memory_info = line.split("GPU Memory: ")[1]
                            socketio.emit('detection_update', {
                                'message': f'💾 GPU Memory: {memory_info}'
                            })
                        elif "[INFO] Using FP16" in line:
                            socketio.emit('detection_update', {
                                'message': '🚀 FP16 half-precision enabled for faster inference'
                            })
                        elif "[STATUS] DETECTION_STARTED" in line:
                            socketio.emit('detection_update', {
                                'message': 'Detection started!'
                            })
                        elif "[STATUS] DETECTION_COMPLETE" in line:
                            socketio.emit('detection_complete', {
                                'message': f'🎉 Detection completed! Processed {processed_files} files.',
                                'model_path': model_path,
                                'test_folder': test_folder,
                                'output_folder': output_folder,
                                'processed_files': processed_files
                            })
                            return
                        elif "[STATUS] DETECTION_FAILED" in line:
                            socketio.emit('detection_error', {
                                'error': 'Detection process failed. Check the logs for details.'
                            })
                            return
                        elif "[STATUS] DETECTION_CANCELLED" in line:
                            socketio.emit('detection_error', {
                                'error': 'Detection was cancelled.'
                            })
                            return
                        
                        # Parse file count
                        elif "Found" in line and "files to process" in line:
                            try:
                                total_files = int(line.split("Found ")[1].split(" files")[0])
                                socketio.emit('detection_update', {
                                    'message': f'📊 Found {total_files} files to process'
                                })
                            except:
                                pass
                        
                        # Parse progress updates
                        elif "[PROGRESS] Overall progress:" in line:
                            try:
                                # Extract progress percentage and file numbers
                                progress_part = line.split("Overall progress: ")[1]
                                percentage = float(progress_part.split("%")[0])
                                file_info = progress_part.split("(")[1].split(")")[0]
                                current, total = map(int, file_info.split("/"))
                                processed_files = current
                                
                                socketio.emit('detection_update', {
                                    'message': f'📈 Progress: {percentage:.1f}% ({current}/{total} files)'
                                })
                            except:
                                pass
                        
                        # Parse individual file processing
                        elif "[INFO] Processing:" in line:
                            filename = line.split("Processing: ")[1]
                            socketio.emit('detection_update', {
                                'message': f'🔍 Processing: {filename}'
                            })
                        elif "[INFO] Saved:" in line:
                            filename = line.split("Saved: ")[1]
                            socketio.emit('detection_update', {
                                'message': f'💾 Saved: {filename}'
                            })
                        elif "[ERROR]" in line:
                            socketio.emit('detection_update', {
                                'message': f'❌ {line}'
                            })
                        elif "[WARNING]" in line:
                            socketio.emit('detection_update', {
                                'message': f'⚠️ {line}'
                            })
                
                # If we reach here, process ended without explicit status
                process.wait()
                if process.returncode == 0:
                    socketio.emit('detection_complete', {
                        'message': f'✅ Detection completed! Check output folder: {output_folder}',
                        'model_path': model_path,
                        'test_folder': test_folder,
                        'output_folder': output_folder,
                        'processed_files': processed_files
                    })
                else:
                    socketio.emit('detection_error', {
                        'error': f'Detection script exited with error code: {process.returncode}'
                    })
                    
            except Exception as e:
                print(f"Detection error: {str(e)}")
                socketio.emit('detection_error', {
                    'error': f'Failed to run detection: {str(e)}'
                })
        
        detection_thread = threading.Thread(target=run_detection_process)
        detection_thread.daemon = True
        detection_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Detection started",
            "model_path": model_path,
            "test_folder": test_folder,
            "output_folder": output_folder
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/start_live_detection", methods=["POST"])
def start_live_detection():
    """Start live camera detection"""
    global live_detection_active, live_detection_model, live_detection_thread
    
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        camera_index = data.get('camera_index', 0)
        
        if not model_name:
            return jsonify({
                "status": "error",
                "message": "No model selected"
            }), 400
        
        # Build model path
        model_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), 
            f"../../TrainedModels/{model_name}/weights/best.pt"
        ))
        
        if not os.path.exists(model_path):
            return jsonify({
                "status": "error",
                "message": f"Model not found: {model_name}"
            }), 404
        
        # Stop any existing detection
        if live_detection_active:
            stop_live_detection_process()
        
        # Load model
        live_detection_model = YOLO(model_path)
        live_detection_active = True
        
        # Start detection thread
        live_detection_thread = threading.Thread(
            target=run_live_detection_process, 
            args=(camera_index,)
        )
        live_detection_thread.daemon = True
        live_detection_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Live detection started",
            "model_name": model_name,
            "camera_index": camera_index
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/stop_live_detection", methods=["POST"])
def stop_live_detection():
    """Stop live camera detection"""
    global live_detection_active
    
    try:
        stop_live_detection_process()
        
        return jsonify({
            "status": "success",
            "message": "Live detection stopped"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

def stop_live_detection_process():
    """Helper function to stop live detection"""
    global live_detection_active, live_detection_model, live_detection_thread
    
    live_detection_active = False
    live_detection_model = None
    
    if live_detection_thread:
        live_detection_thread.join(timeout=2)
        live_detection_thread = None

def run_live_detection_process(camera_index):
    """Run live detection on camera feed using YOLO native streaming"""
    global live_detection_active, live_detection_model
    
    try:
        socketio.emit('live_detection_started', {
            'message': 'Initializing live detection...'
        })
        
        detection_counts = {'fire': 0, 'smoke': 0, 'other': 0}
        frame_count = 0
        
        # Use YOLO native streaming with camera source
        print(f"[INFO] Starting YOLO stream on camera {camera_index}")
        
        # Configure device
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {device}")
        
        # Start YOLO streaming inference
        results_generator = live_detection_model(
            source=camera_index,           # Camera index (0, 1, 2, etc.)
            stream=True,                   # Enable streaming mode
            conf=0.5,                      # Confidence threshold
            device=device,                 # GPU/CPU device
            imgsz=640,                     # Input image size
            verbose=False,                 # Disable verbose output
            classes=None,                  # Detect all classes
            save=False,                    # Don't save files
            show=False                     # Don't display windows
        )
        
        socketio.emit('live_detection_started', {
            'message': 'Live detection started successfully'
        })
        
        # Process streaming results
        for result in results_generator:
            if not live_detection_active:
                break
            
            frame_count += 1
            
            # Get original image with detections drawn
            annotated_frame = result.plot()  # YOLO draws bounding boxes automatically
            
            # Extract detection information
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Map class IDs to names
                    class_names = ['fire', 'other', 'smoke']
                    class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    }
                    detections.append(detection)
                    
                    # Update counts
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
            # Process every frame for better responsiveness
            if frame_count % 1 == 0:  # Process all frames
                # Convert frame to base64 for streaming
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send processed frame and detection data
                socketio.emit('live_detection_frame', {
                    'frame': frame_base64,
                    'detections': detections,
                    'counts': detection_counts,
                    'timestamp': time.time(),
                    'frame_count': frame_count
                })
            
            # Small delay for better performance
            time.sleep(0.01)  # ~100 FPS max, but limited by camera/processing speed
        
        socketio.emit('live_detection_stopped', {
            'message': 'Live detection stopped',
            'total_detections': detection_counts,
            'frames_processed': frame_count
        })
        
    except Exception as e:
        print(f"[ERROR] Live detection error: {str(e)}")
        socketio.emit('live_detection_error', {
            'error': f'Live detection error: {str(e)}'
        })

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)

