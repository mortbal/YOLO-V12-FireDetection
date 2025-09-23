from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO
import os
from shared_types import UpdateType

# Import managers
from training_manager import TrainingManager
from detection_manager import DetectionManager
from live_detection import LiveDetectionManager
from file_manager import FileManager

# Global root directory - set to project root (two levels up from backend)
rootDir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))

# Flask configuration
static_dir = os.path.normpath(os.path.join(rootDir, "Dashboard/static"))
app = Flask(__name__, static_folder=static_dir, static_url_path='/static')

socketio = SocketIO(app, cors_allowed_origins="*")

# Array for log filtering
log_filter = [UpdateType.STATUS, UpdateType.DEBUG, UpdateType.VERBOSE, UpdateType.ERROR, UpdateType.WARNING]

# Boolean for printing to chrome console
log_to_chrome = False

# Debug logging flag
DEBUG_LOGGING = False

def log_update(update_type: UpdateType, update_message: str):
    # Log function for updates with type and message
    # Only show DEBUG messages if DEBUG_LOGGING is enabled
    if update_type == UpdateType.DEBUG and not DEBUG_LOGGING:
        return

    if update_type in log_filter:
        print(f"[{update_type.value}] {update_message}")
        if log_to_chrome:
            emit_to_frontend('console_log', {
                'type': update_type.value,
                'message': update_message
            })

# Centralized SocketIO emit function
def emit_to_frontend(event_name, data):
    # Central function for all SocketIO emissions - makes debugging easier
    print(f"[EMIT] {event_name}: {data}")
    socketio.emit(event_name, data)

# Update callback for training updates
def training_update_callback(yolo_update):
    # Callback for training updates - just print for now
    print(f"[UPDATE_RECEIVED] Type: {yolo_update.update_type.value}, Data: {yolo_update.data}")

# Initialize managers after function definitions
training_manager = TrainingManager(emit_to_frontend, log_update)
detection_manager = DetectionManager(emit_to_frontend, log_update)
live_detection_manager = LiveDetectionManager(emit_to_frontend, log_update)
file_manager = FileManager(emit_to_frontend, log_update)

# Routes
@app.route("/")
def serve_index():
     # Serve main page
    frontend_dir = os.path.normpath(os.path.join(rootDir, "Dashboard/frontend"))
    return send_from_directory(frontend_dir, "index.html")

# Training routes
@app.route("/train", methods=["POST"])
def run_training():
    # Start training - route to training manager
    data = request.get_json()
    selected_model = data.get("model", "")
    epochs = data.get("epochs", 100)
    result = training_manager.start_training(selected_model, epochs, training_update_callback)


@app.route("/cancel_training", methods=["POST"])
def cancel_training():
    # Cancel training - route to training manager
    try:
        result = training_manager.cancel_training()
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Detection routes
@app.route("/run_detection", methods=["POST"])
def run_detection():
    # Run batch detection - route to detection manager
    data = request.get_json()
    model_name = data.get('model_name')
    test_folder = data.get('test_folder')
    output_folder = data.get('output_folder')
    
    result = detection_manager.run_detection(model_name, test_folder, output_folder)
    if result["status"] == "success":
        log_update(UpdateType.STATUS, f"Detection started successfully: {result['message']}")
        return jsonify(result)
    else:
        log_update(UpdateType.ERROR, f"Detection failed: {result['message']}")
        return jsonify(result), 400 if "Missing required" in result["message"] else 404


# Live detection routes
@app.route("/start_live_detection", methods=["POST"])
def start_live_detection():
    # Start live detection - route to live detection manager
    data = request.get_json()
    model_name = data.get('model_name')
    camera_index = data.get('camera_index', 0)
    if not model_name:
        return jsonify({"status": "error", "message": "No model selected"}), 400
    # Build model path
    model_path = os.path.normpath(os.path.join(
        rootDir,
        f"TrainedModels/{model_name}/weights/best.pt"
    ))
    if not os.path.exists(model_path):
        return jsonify({"status": "error", "message": f"Model not found: {model_name}"}), 404
    result = live_detection_manager.start_live_detection(model_path, camera_index)
    if result["status"] == "success":
        result["model_name"] = model_name
        return jsonify(result)
    else:
        return jsonify(result), 500


@app.route("/stop_live_detection", methods=["POST"])
def stop_live_detection():
    # Stop live detection - route to live detection manager
    try:
        result = live_detection_manager.stop_live_detection()
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# File management routes
@app.route("/browse_folder", methods=["POST"])
def browse_folder():
    # Browse for folder - route to file manager
    try:
        data = request.get_json()
        folder_type = data.get('type', '')

        result = file_manager.browse_folder(folder_type)

        if result["status"] == "success":
            return jsonify(result)
        elif result["status"] == "cancelled":
            return jsonify(result)
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/get_all_models", methods=["GET"])
def get_all_models():
    # Get all models - route to file manager
    try:
        result = file_manager.get_all_models()
        return jsonify(result)
    except Exception as e:
        return jsonify({"baseModels": [], "trainedModels": [], "checkpointModels": []})

@app.route("/get_models", methods=["GET"])
def get_models():
    # Get trained models - route to file manager (backward compatibility)
    try:
        result = file_manager.get_trained_models()
        return jsonify(result)
    except Exception as e:
        return jsonify([])

# State management routes
@app.route("/get_app_state", methods=["GET"])
def get_app_state():
    # Get application state from training manager
    try:
        training_state = training_manager.get_state()
        live_detection_state = live_detection_manager.get_status()
        detection_state = detection_manager.get_status()

        # Combine states
        app_state = training_state.copy()
        app_state['live_detection'] = live_detection_state
        app_state['detection'] = detection_state
        app_state['debug_logging'] = DEBUG_LOGGING

        return jsonify(app_state)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/set_debug_logging", methods=["POST"])
def set_debug_logging():
    #Enable/disable debug logging centrally in app.py
    global DEBUG_LOGGING
    try:
        data = request.get_json()
        DEBUG_LOGGING = data.get('enabled', False)

        # No need to update managers - they use log_update which checks DEBUG_LOGGING

        print(f"Debug logging {'enabled' if DEBUG_LOGGING else 'disabled'}")
        return jsonify({"status": "success", "debug_logging": DEBUG_LOGGING})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# SocketIO event handlers (minimal)
@socketio.on('connect')
def handle_connect():
    # Handle client connection
    pass

@socketio.on('disconnect')
def handle_disconnect():
    # Handle client disconnection
    pass

if __name__ == "__main__":

    # Test manager initialization
    print("✅ Managers initialized successfully")
    print(f"   - Training Manager: {type(training_manager).__name__}")
    print(f"   - Detection Manager: {type(detection_manager).__name__}")
    print(f"   - Live Detection Manager: {type(live_detection_manager).__name__}")
    print(f"   - File Manager: {type(file_manager).__name__}")
    print()

    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False, log_output=True)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        input("Press Enter to exit...")