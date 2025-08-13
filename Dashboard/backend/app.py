from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
import subprocess
import sys
import os
import yaml
import threading
import time
from datetime import datetime

# Flask configuration
static_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../static"))
app = Flask(__name__, static_folder=static_dir, static_url_path='/static')

socketio = SocketIO(app, cors_allowed_origins="*")

# Training state variables
training_start_time = None
epoch_start_times = {}
last_epoch_duration = None
total_elapsed_timer = None
training_process = None
is_training_paused = False
current_model_info = None

frontend_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../frontend"))

@app.route("/")
def serve_index():
    return send_from_directory(frontend_dir, "index.html")

def update_training_config(epochs):
    """Update epochs in training config"""
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../YoloAssets/train_config_minimal.yaml")
    )
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['epochs'] = epochs
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        return True
    except Exception as e:
        return False

@app.route("/train", methods=["POST"])
def run_training():
    
    try:
        data = request.get_json()
        yolo_version = data.get("yolo_version", 12)
        model_size = data.get("model_size", "n")
        epochs = data.get("epochs", 100)
        
        model_name = f"yolo{yolo_version}{model_size}"
        
        if not update_training_config(epochs):
            return jsonify({"status": "error", "message": "Failed to update training config"}), 500
        
        script_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../YoloAssets/TrainYolo.py")
        )
        
        
        global training_start_time, training_process, current_model_info
        training_start_time = time.time()
        current_model_info = {
            'yolo_version': yolo_version,
            'model_size': model_size,
            'epochs': epochs,
            'model_name': model_name
        }
        
        start_total_elapsed_timer()
        
        emit_status_update("Preparing for train : Loading YOLO...")
        
        training_process = subprocess.Popen(
            [sys.executable, script_path, model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=os.path.join(os.path.dirname(__file__), "../..")
        )
        
        def monitor_regular_training_output():
            global training_start_time, epoch_start_times, last_epoch_duration
            for line in iter(training_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    # Print all output to console
                    print(f"[TrainYolo] {line}")
                    
                    # Continue with existing processing logic

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

@app.route("/pause", methods=["POST"])
def pause_training():
    """Pause the current training process"""
    global training_process, is_training_paused
    
    try:
        if training_process is None:
            return jsonify({"status": "error", "message": "No training process running"}), 400
        
        if is_training_paused:
            return jsonify({"status": "error", "message": "Training is already paused"}), 400
        
        # Terminate the current training process gracefully
        training_process.terminate()
        
        # Wait for process to terminate
        try:
            training_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            training_process.kill()
            training_process.wait()
        
        is_training_paused = True
        stop_total_elapsed_timer()
        
        emit_status_update("Training Paused")
        socketio.emit('training_paused', {
            'message': 'Training has been paused. You can resume from the last checkpoint.'
        })
        
        return jsonify({
            "status": "paused",
            "message": "Training paused successfully"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/resume", methods=["POST"])
def resume_training():
    """Resume training from the last checkpoint"""
    global training_process, is_training_paused, current_model_info, training_start_time
    
    try:
        if not is_training_paused:
            return jsonify({"status": "error", "message": "No paused training to resume"}), 400
        
        if current_model_info is None:
            return jsonify({"status": "error", "message": "No training information available"}), 400
        
        data = request.get_json() if request.is_json else {}
        additional_epochs = data.get("epochs", 50)  # Default to 50 more epochs
        
        script_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../YoloAssets/ContinueTraining.py")
        )
        
        # Reset training state for resumption
        training_start_time = time.time()
        is_training_paused = False
        
        start_total_elapsed_timer()
        emit_status_update("Resuming training from checkpoint...")
        
        training_process = subprocess.Popen(
            [
                sys.executable, script_path,
                "--model", current_model_info['model_name'],
                "--epochs", str(additional_epochs)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=os.path.join(os.path.dirname(__file__), "../..")
        )
        
        def monitor_continue_training_output():
            global training_start_time, epoch_start_times, last_epoch_duration
            for line in iter(training_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    # Print all output to console
                    print(f"[ContinueTraining] {line}")
                    
                    # Process the same status messages as regular training
                    if "[STATUS] INITIALIZING" in line:
                        emit_status_update("Resuming training : Loading checkpoint...")
                        
                    elif "[STATUS] TRAINING_STARTING" in line:
                        emit_status_update("Resuming training : Warming up...")
                        
                    elif "[STATUS] TRAINING_STARTED" in line:
                        emit_status_update("Training (Resumed)")
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
                        
                    elif "[STATUS] SUCCESS" in line:
                        total_elapsed = time.time() - training_start_time if training_start_time else 0
                        stop_total_elapsed_timer()
                        emit_status_update("Training Complete!")
                        socketio.emit('training_complete', {
                            'message': 'Resumed training completed successfully!',
                            'total_time': format_seconds_to_time(total_elapsed)
                        })
                        
                    elif "[STATUS] FAILED" in line:
                        total_elapsed = time.time() - training_start_time if training_start_time else 0
                        stop_total_elapsed_timer()
                        emit_status_update("Training Failed")
                        socketio.emit('training_error', {
                            'error': 'Resumed training process failed',
                            'total_time': format_seconds_to_time(total_elapsed)
                        })
        
        monitor_thread = threading.Thread(target=monitor_continue_training_output)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return jsonify({
            "status": "training_resumed",
            "additional_epochs": additional_epochs,
            "message": f"Resumed training for {additional_epochs} additional epochs"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/check_unfinished", methods=["GET"])
def check_unfinished_training():
    """Check if there's an unfinished training run that can be resumed"""
    try:
        import glob
        import torch
        
        training_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../YoloAssets/runs/train/fire_detection_model/weights")
        )
        
        if not os.path.exists(training_dir):
            return jsonify({"has_unfinished": False})
        
        # Look for checkpoints in order of preference
        checkpoints = []
        
        # Check for last.pt (most recent)
        last_checkpoint = os.path.join(training_dir, "last.pt")
        if os.path.exists(last_checkpoint):
            checkpoints.append(last_checkpoint)
        
        # Check for best.pt
        best_checkpoint = os.path.join(training_dir, "best.pt")
        if os.path.exists(best_checkpoint):
            checkpoints.append(best_checkpoint)
        
        # Check for epoch checkpoints
        epoch_checkpoints = glob.glob(os.path.join(training_dir, "epoch*.pt"))
        if epoch_checkpoints:
            epoch_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            checkpoints.extend(epoch_checkpoints)
        
        if not checkpoints:
            return jsonify({"has_unfinished": False})
        
        # Get info from the best available checkpoint
        checkpoint_path = checkpoints[0]
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 0)
            best_fitness = checkpoint.get('best_fitness', 0)
            
            # Read training config to get total epochs
            config_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "../../YoloAssets/train_config_minimal.yaml")
            )
            
            total_epochs = 100  # default
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                total_epochs = config.get('epochs', 100)
            
            # Check if training was completed
            is_completed = epoch >= total_epochs
            
            checkpoint_info = {
                "has_unfinished": not is_completed,
                "checkpoint_path": checkpoint_path,
                "current_epoch": epoch,
                "total_epochs": total_epochs,
                "completion_percentage": (epoch / total_epochs * 100) if total_epochs > 0 else 0,
                "best_fitness": float(best_fitness) if best_fitness else 0,
                "checkpoint_type": os.path.basename(checkpoint_path),
                "last_modified": os.path.getmtime(checkpoint_path)
            }
            
            return jsonify(checkpoint_info)
            
        except Exception as e:
            # If we can't read the checkpoint, assume it exists but we can't get details
            return jsonify({
                "has_unfinished": True,
                "checkpoint_path": checkpoint_path,
                "current_epoch": 0,
                "total_epochs": 100,
                "completion_percentage": 0,
                "best_fitness": 0,
                "checkpoint_type": os.path.basename(checkpoint_path),
                "error": f"Could not read checkpoint details: {str(e)}"
            })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/clear_training", methods=["POST"])
def clear_training_data():
    """Clear existing training data to start fresh"""
    try:
        training_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../YoloAssets/runs/train")
        )
        
        if os.path.exists(training_dir):
            import shutil
            shutil.rmtree(training_dir)
            print(f"Cleared training directory: {training_dir}")
        
        return jsonify({
            "status": "cleared",
            "message": "Training data cleared successfully"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/list_trained_models", methods=["GET"])
def list_trained_models():
    """List all available trained models from TrainedModels folder"""
    try:
        trained_models_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../TrainedModels")
        )
        
        models = []
        
        if not os.path.exists(trained_models_dir):
            return jsonify({"models": models})
        
        # Scan TrainedModels directory for model folders
        for item in os.listdir(trained_models_dir):
            model_path = os.path.join(trained_models_dir, item)
            
            if os.path.isdir(model_path):
                # Look for best.pt file in the model folder
                best_model_path = os.path.join(model_path, "best.pt")
                
                if os.path.exists(best_model_path):
                    # Extract model information from folder name
                    # Expected format: ModelName_E100_20250812_141258
                    parts = item.split('_')
                    model_info = {
                        "folder_name": item,
                        "model_path": best_model_path,
                        "display_name": item,
                        "created_date": os.path.getctime(model_path)
                    }
                    
                    # Try to extract more readable info from folder name
                    if len(parts) >= 2:
                        model_name = parts[0]  # e.g., Yolo12n
                        epochs = parts[1] if parts[1].startswith('E') else 'Unknown'
                        
                        if len(parts) >= 4:
                            date_part = parts[2]
                            time_part = parts[3]
                            
                            try:
                                # Format date and time more readably
                                from datetime import datetime
                                dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                                formatted_date = dt.strftime("%Y-%m-%d %H:%M")
                                
                                model_info["display_name"] = f"{model_name} ({epochs}) - {formatted_date}"
                                model_info["model_type"] = model_name
                                model_info["epochs"] = epochs
                                model_info["training_date"] = formatted_date
                            except:
                                pass
                    
                    models.append(model_info)
        
        # Sort models by creation date (newest first)
        models.sort(key=lambda x: x["created_date"], reverse=True)
        
        return jsonify({"models": models})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/run_detection", methods=["POST"])
def run_detection():
    """Run detection on test files using selected model"""
    try:
        data = request.get_json()
        model_path = data.get("model_path")
        test_folder = data.get("test_folder")
        output_folder = data.get("output_folder")
        
        if not all([model_path, test_folder, output_folder]):
            return jsonify({"status": "error", "message": "Missing required parameters"}), 400
        
        # Validate paths exist
        if not os.path.exists(model_path):
            return jsonify({"status": "error", "message": "Selected model not found"}), 400
            
        if not os.path.exists(test_folder):
            return jsonify({"status": "error", "message": "Test folder not found"}), 400
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        script_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../YoloAssets/RunDetection.py")
        )
        
        # Run detection script
        global training_process  # Reuse the global process variable
        training_process = subprocess.Popen(
            [
                sys.executable, script_path,
                "--model", model_path,
                "--source", test_folder,
                "--output", output_folder
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=os.path.join(os.path.dirname(__file__), "../..")
        )
        
        def monitor_detection_output():
            for line in iter(training_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    print(f"[RunDetection] {line}")
                    
                    # Emit detection progress updates
                    socketio.emit('detection_update', {
                        'message': line
                    })
                    
                    if "[STATUS] DETECTION_COMPLETE" in line:
                        socketio.emit('detection_complete', {
                            'message': 'Detection completed successfully!',
                            'output_folder': output_folder
                        })
                    elif "[STATUS] DETECTION_FAILED" in line:
                        socketio.emit('detection_error', {
                            'error': 'Detection process failed'
                        })
        
        monitor_thread = threading.Thread(target=monitor_detection_output)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return jsonify({
            "status": "detection_started",
            "message": "Detection started successfully",
            "output_folder": output_folder
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


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
