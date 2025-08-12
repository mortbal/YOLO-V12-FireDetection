from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
import sys
import threading
import time
import subprocess
import json
import socket

# Add parent directory to path to import training modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, 
           template_folder='../frontend',
           static_folder='../static')
app.config['SECRET_KEY'] = 'fire_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.current_config = None
        self.training_thread = None
        self.progress_server = None
        self.progress_socket = None
        
    def start_training(self, config):
        if self.is_training:
            return False
            
        self.is_training = True
        self.current_config = config
        
        # Start progress server to receive updates from TrainYolo.py
        self.progress_server = threading.Thread(target=self._start_progress_server)
        self.progress_server.start()
        
        # Start actual training
        self.training_thread = threading.Thread(target=self._run_training, args=(config,))
        self.training_thread.start()
        return True
    
    def _start_progress_server(self):
        """Start a server to receive progress updates from TrainYolo.py"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', 5001))
            server_socket.listen(1)
            server_socket.settimeout(1)  # Non-blocking with timeout
            
            print("Progress server started on port 5001")
            
            while self.is_training:
                try:
                    client_socket, address = server_socket.accept()
                    self.progress_socket = client_socket
                    
                    # Handle incoming progress updates
                    buffer = ""
                    while self.is_training:
                        try:
                            data = client_socket.recv(1024).decode()
                            if not data:
                                break
                            
                            buffer += data
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                if line.strip():
                                    try:
                                        update_data = json.loads(line.strip())
                                        self._handle_progress_update(update_data)
                                    except json.JSONDecodeError:
                                        pass
                        except socket.timeout:
                            continue
                        except Exception as e:
                            print(f"Error receiving data: {e}")
                            break
                    
                    client_socket.close()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_training:
                        print(f"Progress server error: {e}")
                    break
            
            server_socket.close()
            print("Progress server stopped")
            
        except Exception as e:
            print(f"Failed to start progress server: {e}")
    
    def _handle_progress_update(self, update_data):
        """Handle progress updates from TrainYolo.py"""
        if update_data.get('type') == 'epoch_end':
            # Emit training update to frontend
            socketio.emit('training_update', {
                'epoch': update_data.get('epoch', 0),
                'total_epochs': update_data.get('total_epochs', 0),
                'loss': update_data.get('loss', 0),
                'map': update_data.get('map', 0),
                'progress': update_data.get('progress', 0)
            })
        elif update_data.get('type') == 'train_start':
            socketio.emit('training_started', {'message': 'Training started'})
        elif update_data.get('type') == 'train_end':
            socketio.emit('training_complete', {'message': 'Training completed'})
        
    def _run_training(self, config):
        try:
            # Build command to run TrainYolo.py
            # Go up from backend folder to Dashboard folder, then up to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            yolo_script_path = os.path.join(project_root, 'YoloAssets', 'TrainYolo.py')
            
            command = [
                sys.executable,  # Python executable
                yolo_script_path,
                '--Model', str(config['yolo_version']),
                '--Size', config['model_size'], 
                '--Epoch', str(config['epochs'])
            ]
            
            print(f"Starting training with command: {' '.join(command)}")
            
            # Run the training script with real-time output
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                cwd=project_root  # Set working directory to project root
            )
            
            # Read output line by line in real-time
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(f"[TrainYolo] {line}")  # Print to Flask console
                    output_lines.append(line)
                    
                    # Emit console output to frontend (optional)
                    socketio.emit('console_output', {'line': line})
            
            # Wait for process to complete
            process.wait()
            stdout = '\n'.join(output_lines)
            stderr = ""
            
            if process.returncode == 0:
                print("Training completed successfully")
                socketio.emit('training_complete', {'message': 'Training completed successfully'})
            else:
                print(f"Training failed: {stderr}")
                socketio.emit('training_error', {'error': f'Training failed: {stderr}'})
                
        except Exception as e:
            print(f"Training error: {e}")
            socketio.emit('training_error', {'error': str(e)})
        finally:
            self.is_training = False
            if self.progress_socket:
                self.progress_socket.close()

training_manager = TrainingManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start-training', methods=['POST'])
def start_training():
    try:
        config = request.json
        
        # Validate configuration
        if not all(key in config for key in ['yolo_version', 'model_size', 'epochs']):
            return jsonify({'error': 'Missing configuration parameters'}), 400
            
        if config['epochs'] < 1 or config['epochs'] > 1000:
            return jsonify({'error': 'Invalid epoch count'}), 400
            
        # Start training
        if training_manager.start_training(config):
            return jsonify({'status': 'Training started', 'config': config})
        else:
            return jsonify({'error': 'Training already in progress'}), 409
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-training', methods=['POST'])
def stop_training():
    training_manager.is_training = False
    return jsonify({'status': 'Training stopped'})

@app.route('/api/training-status')
def training_status():
    return jsonify({
        'is_training': training_manager.is_training,
        'config': training_manager.current_config
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'is_training': training_manager.is_training})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)