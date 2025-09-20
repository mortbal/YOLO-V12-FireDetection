# Training Manager Module
# Handles YOLO model training orchestration and monitoring

import subprocess
import sys
import os
import yaml
import threading
import time
from datetime import datetime
import re
from yolo_output_processor import YoloOutputProcessor
from shared_types import UpdateType

class TrainingManager:
    def __init__(self, emit_function, log_update):
        self.emit_function = emit_function
        self.log_update = log_update

        # Training state
        self.training_process = None
        self.training_start_time = None
        self.total_elapsed_timer = None
        self.epoch_start_times = {}
        self.last_epoch_duration = None

        # YOLO output processor
        self.yolo_processor = YoloOutputProcessor()

        # Training state for persistence
        self.app_state = {
            'training_status': 'idle',
            'training_progress': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'last_training_message': '',
            'training_start_time': None,
        }

    def start_training(self, selected_model: str, epochs: int, update_callback=None):
        """
        Start YOLO model training

        Args:
            selected_model: Model selection string (format: "type:modelname")
            epochs: Number of training epochs
            update_callback: Callback function for training updates

        Returns:
            dict: Status response
        """
        try:
            # Parse the selected model format: "type:modelname"
            if ":" in selected_model:
                model_type, model_name = selected_model.split(":", 1)
            else:
                # Fallback for backward compatibility
                model_name = f"yolo12n.pt"
                model_type = "base"

            # Prepare model info for config update
            model_info = {'type': model_type, 'name': model_name}

            # Update training config
            if not self._update_training_config(epochs, model_info):
                return {"status": "error", "message": "Failed to update training config"}

            script_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "../../YoloAssets/TrainYolo.py")
            )

            if not os.path.exists(script_path):
                return {"status": "error", "message": f"Training script not found: {script_path}"}

            # Set training start time
            self.training_start_time = time.time()
            self.yolo_processor.set_training_start_time(self.training_start_time)

            # Update app state
            self.app_state['training_status'] = 'starting'
            self.app_state['total_epochs'] = epochs
            self.app_state['current_epoch'] = 0
            self.app_state['training_progress'] = 0
            self.app_state['last_training_message'] = f"Starting training {model_name} for {epochs} epochs"
            self.app_state['training_start_time'] = self.training_start_time

            # Start elapsed time timer
            self._start_total_elapsed_timer()

            # Emit initial status
            self.emit_function('training_status_update', {'status': 'Preparing for train : Loading YOLO...'})

            # Start training process
            self.training_process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                cwd=os.path.join(os.path.dirname(__file__), "../..")
            )

            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_training_output, args=(update_callback,))
            monitor_thread.daemon = True
            monitor_thread.start()

            return {
                "status": "training_started",
                "model": model_name,
                "epochs": epochs,
                "message": f"Started training {model_name} for {epochs} epochs"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def cancel_training(self):
        """
        Cancel currently running training

        Returns:
            dict: Status response
        """
        try:
            if self.training_process and self.training_process.poll() is None:
                # Process is still running, terminate it
                self.training_process.terminate()
                self.training_process.wait(timeout=5)

            # Stop the timer
            if self.total_elapsed_timer:
                self.total_elapsed_timer.cancel()
                self.total_elapsed_timer = None

            # Update app state
            self.app_state['training_status'] = 'cancelled'
            self.app_state['last_training_message'] = 'Training cancelled by user'

            # Reset training process
            self.training_process = None

            # Emit cancellation status
            self.emit_function('training_complete', {
                'status': 'cancelled',
                'message': 'Training cancelled by user'
            })

            return {"status": "success", "message": "Training cancelled successfully"}

        except Exception as e:
            return {"status": "error", "message": f"Failed to cancel training: {str(e)}"}

    def get_state(self):
        """Get current training state"""
        return self.app_state.copy()


    def _monitor_training_output(self, update_callback=None):
        """Monitor training process output"""
        for line in iter(self.training_process.stdout.readline, ''):
            line = line.strip()
            if line:
                self.log_update(UpdateType.DEBUG, f"Training output: {line}")

                # Process line through YOLO output processor
                yolo_update = self.yolo_processor.process_line(line)

                if yolo_update:
                    # Handle the structured update
                    self._handle_training_update(yolo_update)

                    # Call external update callback if provided
                    if update_callback:
                        update_callback(yolo_update)

                # Handle special cases not covered by processor
                if "[INFO] Training completed successfully" in line:
                    self._rename_training_folder_to_finished()

    def _handle_training_update(self, yolo_update):
        """Handle training updates internally"""
        if yolo_update.update_type == UpdateType.STATUS:
            status_data = yolo_update.data

            # Check if this is training progress data (has epoch info)
            if 'epoch' in status_data and 'total_epochs' in status_data:
                # This is training progress
                self.app_state['current_epoch'] = status_data.get('epoch', 0)
                self.app_state['total_epochs'] = status_data.get('total_epochs', 0)
                self.app_state['training_progress'] = (status_data.get('epoch', 0) / max(status_data.get('total_epochs', 1), 1)) * 100
                self.app_state['last_training_message'] = f"Training epoch {status_data.get('epoch', 0)}/{status_data.get('total_epochs', 0)}"

                # Emit training update to frontend
                training_data = {
                    'epoch': status_data.get('epoch', 0),
                    'total_epochs': status_data.get('total_epochs', 0),
                    'progress': status_data.get('progress_percent', 0),
                    'batch_progress': status_data.get('progress_percent', 0),
                    'current_batch': status_data.get('current_batch', 0),
                    'total_batches': status_data.get('total_batches', 0),
                    'gpu_memory': status_data.get('gpu_memory'),
                    'iteration_speed': status_data.get('iteration_speed'),
                    'elapsed_time': status_data.get('elapsed_time'),
                    'remaining_time': status_data.get('remaining_time'),
                    'total_elapsed_seconds': status_data.get('total_elapsed_seconds', 0)
                }

                self.emit_function('training_update', training_data)

            else:
                # This is a regular status update
                self.app_state['training_status'] = status_data.get('status', 'unknown')
                self.app_state['last_training_message'] = status_data.get('message', '')

                # Emit status update to frontend
                self.emit_function('training_status_update', {'status': status_data.get('message', '')})

                # Handle specific status changes
                if status_data.get('status') == 'completed':
                    total_elapsed = time.time() - self.training_start_time if self.training_start_time else 0
                    self._stop_total_elapsed_timer()
                    self.emit_function('training_complete', {
                        'message': 'Training completed successfully!',
                        'total_time': self._format_seconds_to_time(total_elapsed)
                    })
                elif status_data.get('status') == 'failed':
                    total_elapsed = time.time() - self.training_start_time if self.training_start_time else 0
                    self._stop_total_elapsed_timer()
                    self.emit_function('training_error', {
                        'error': 'Training process failed',
                        'total_time': self._format_seconds_to_time(total_elapsed)
                    })

    def _update_training_config(self, epochs, model_info=None):
        """Update training config with epochs and model information"""
        config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../YoloAssets/train_config_minimal.yaml")
        )

        try:
            if not os.path.exists(config_path):
                return False

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Update epochs
            config['epochs'] = epochs

            # Generate timestamp for folder name
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Update model information if provided
            if model_info:
                model_type = model_info.get('type', 'base')
                model_name = model_info.get('name', 'yolo11n.pt')

                # Set model path based on type
                if model_type == 'base':
                    model_path = os.path.join('YoloAssets/BaseModels', model_name)
                    config['resume'] = False
                elif model_type == 'trained':
                    if model_name.startswith('finished_run_'):
                        model_path = os.path.join('YoloAssets/Trains', model_name, 'weights/best.pt')
                    else:
                        model_path = os.path.join('YoloAssets/TrainedModels', model_name, 'best.pt')
                    config['resume'] = True
                elif model_type == 'checkpoint':
                    model_path = os.path.join('YoloAssets/Trains', model_name, 'weights/last.pt')
                    config['resume'] = True

                config['model'] = model_path
                if 'model_type' in config:
                    del config['model_type']

            # Generate folder name
            folder_name = f"ongoing_run_{timestamp}"
            config['name'] = folder_name

            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

            return True
        except Exception as e:
            self.log_update(UpdateType.ERROR, f"Error updating training config: {e}")
            return False

    def _rename_training_folder_to_finished(self):
        """Rename training folder from ongoing to finished"""
        try:
            config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../YoloAssets/train_config_minimal.yaml"))
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            folder_name = config.get('name', '')
            new_folder_name = folder_name.replace('ongoing_', 'finished_')

            trains_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../YoloAssets/Trains"))
            old_path = os.path.join(trains_dir, folder_name)
            new_path = os.path.join(trains_dir, new_folder_name)

            if os.path.exists(old_path) and not os.path.exists(new_path):
               os.rename(old_path, new_path)
        except Exception as e:
            self.log_update(UpdateType.ERROR, f"Error renaming training folder: {e}")

    def _start_total_elapsed_timer(self):
        """Start elapsed time broadcast timer"""
        if self.total_elapsed_timer:
            self.total_elapsed_timer.cancel()

        def send_elapsed_update():
            if self.training_start_time:
                total_elapsed = time.time() - self.training_start_time
                self.emit_function('total_elapsed_update', {
                    'total_elapsed_seconds': total_elapsed,
                    'formatted_time': self._format_seconds_to_time(total_elapsed)
                })

                self.total_elapsed_timer = threading.Timer(1.0, send_elapsed_update)
                self.total_elapsed_timer.daemon = True
                self.total_elapsed_timer.start()

        send_elapsed_update()

    def _stop_total_elapsed_timer(self):
        """Stop elapsed time timer"""
        if self.total_elapsed_timer:
            self.total_elapsed_timer.cancel()
            self.total_elapsed_timer = None

    def _format_seconds_to_time(self, seconds):
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