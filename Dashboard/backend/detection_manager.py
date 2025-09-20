"""
Detection Manager Module
Handles batch detection processing using YOLO models
"""

import subprocess
import sys
import os
import threading
from shared_types import UpdateType

class DetectionManager:
    def __init__(self, emit_function, log_update):
        self.emit_function = emit_function
        self.log_update = log_update
        self.detection_process = None

    def run_detection(self, model_name: str, test_folder: str, output_folder: str):
        """
        Run batch detection on a folder of images

        Args:
            model_name: Name of the trained model
            test_folder: Path to folder containing test images
            output_folder: Path to output folder for results

        Returns:
            dict: Status response
        """
        try:
            if not model_name or not test_folder or not output_folder:
                return {
                    "status": "error",
                    "message": "Missing required parameters"
                }

            # Build model path
            model_path = os.path.normpath(os.path.join(
                os.path.dirname(__file__),
                f"../../TrainedModels/{model_name}/weights/best.pt"
            ))

            if not os.path.exists(model_path):
                return {
                    "status": "error",
                    "message": f"Model not found: {model_name} at path: {model_path}"
                }

            # Start detection process in background thread
            detection_thread = threading.Thread(
                target=self._run_detection_process,
                args=(model_path, test_folder, output_folder)
            )
            detection_thread.daemon = True
            detection_thread.start()

            return {
                "status": "success",
                "message": "Detection started",
                "model_path": model_path,
                "test_folder": test_folder,
                "output_folder": output_folder
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _run_detection_process(self, model_path: str, test_folder: str, output_folder: str):
        """Run the actual detection process"""
        try:
            # Start real detection process
            self.emit_function('detection_update', {
                'message': f'Starting detection with model'
            })

            self.emit_function('detection_update', {
                'message': f'Processing files from: {test_folder}'
            })

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

            self.emit_function('detection_update', {
                'message': 'Launching detection script...'
            })

            # Start subprocess
            self.detection_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                cwd=os.path.join(os.path.dirname(__file__), "../..")
            )

            # Monitor output in real-time
            processed_files = 0

            for line in iter(self.detection_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    self.log_update(UpdateType.DEBUG, f"Detection: {line}")

                    # Parse status messages
                    if "[STATUS] INITIALIZING" in line:
                        self.emit_function('detection_update', {
                            'message': 'Initializing detection system...'
                        })
                    elif "[STATUS] LOADING_MODEL" in line:
                        self.emit_function('detection_update', {
                            'message': 'Loading AI model...'
                        })
                    elif "[STATUS] MODEL_LOADED" in line:
                        self.emit_function('detection_update', {
                            'message': 'Model loaded successfully'
                        })
                    elif "[INFO] Using device:" in line:
                        device = line.split("Using device: ")[1]
                        self.emit_function('detection_update', {
                            'message': f'Using device: {device}'
                        })
                    elif "[INFO] GPU:" in line:
                        gpu_info = line.split("GPU: ")[1]
                        self.emit_function('detection_update', {
                            'message': f'GPU: {gpu_info}'
                        })
                    elif "[INFO] GPU Memory:" in line:
                        memory_info = line.split("GPU Memory: ")[1]
                        self.emit_function('detection_update', {
                            'message': f'GPU Memory: {memory_info}'
                        })
                    elif "[INFO] Using FP16" in line:
                        self.emit_function('detection_update', {
                            'message': 'FP16 half-precision enabled for faster inference'
                        })
                    elif "[STATUS] DETECTION_STARTED" in line:
                        self.emit_function('detection_update', {
                            'message': 'Detection started!'
                        })
                    elif "[STATUS] DETECTION_COMPLETE" in line:
                        self.emit_function('detection_complete', {
                            'message': f'Detection completed! Processed {processed_files} files.',
                            'model_path': model_path,
                            'test_folder': test_folder,
                            'output_folder': output_folder,
                            'processed_files': processed_files
                        })
                        return
                    elif "[STATUS] DETECTION_FAILED" in line:
                        self.emit_function('detection_error', {
                            'error': 'Detection process failed. Check the logs for details.'
                        })
                        return
                    elif "[STATUS] DETECTION_CANCELLED" in line:
                        self.emit_function('detection_error', {
                            'error': 'Detection was cancelled.'
                        })
                        return

                    # Parse file count
                    elif "Found" in line and "files to process" in line:
                        try:
                            total_files = int(line.split("Found ")[1].split(" files")[0])
                            self.emit_function('detection_update', {
                                'message': f'Found {total_files} files to process'
                            })
                        except:
                            pass

                    # Parse progress updates
                    elif "[PROGRESS] Overall progress:" in line:
                        try:
                            progress_part = line.split("Overall progress: ")[1]
                            percentage = float(progress_part.split("%")[0])
                            file_info = progress_part.split("(")[1].split(")")[0]
                            current, total = map(int, file_info.split("/"))
                            processed_files = current
                            self.emit_function('detection_update', {
                                'message': f'Progress: {percentage:.1f}% ({current}/{total} files)'
                            })
                        except:
                            pass

                    # Parse individual file processing
                    elif "[INFO] Processing:" in line:
                        filename = line.split("Processing: ")[1]
                        self.emit_function('detection_update', {
                            'message': f'Processing: {filename}'
                        })
                    elif "[INFO] Saved:" in line:
                        filename = line.split("Saved: ")[1]
                        self.emit_function('detection_update', {
                            'message': f'Saved: {filename}'
                        })
                    elif "[ERROR]" in line:
                        self.emit_function('detection_update', {
                            'message': f'{line}'
                        })
                    elif "[WARNING]" in line:
                        self.emit_function('detection_update', {
                            'message': f'{line}'
                        })

            # If we reach here, process ended without explicit status
            self.detection_process.wait()
            if self.detection_process.returncode == 0:
                self.emit_function('detection_complete', {
                    'message': f'Detection completed! Check output folder: {output_folder}',
                    'model_path': model_path,
                    'test_folder': test_folder,
                    'output_folder': output_folder,
                    'processed_files': processed_files
                })
            else:
                self.emit_function('detection_error', {
                    'error': f'Detection script exited with error code: {self.detection_process.returncode}'
                })

        except Exception as e:
            error_msg = f"Failed to run detection: {str(e)}"
            self.log_update(UpdateType.ERROR, f"Detection error: {error_msg}")
            self.emit_function('detection_error', {
                'error': error_msg
            })

    def cancel_detection(self):
        """Cancel currently running detection"""
        try:
            if self.detection_process and self.detection_process.poll() is None:
                self.detection_process.terminate()
                self.detection_process.wait(timeout=5)
            return {"status": "success", "message": "Detection cancelled"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_status(self):
        """Get detection status"""
        return {
            'running': self.detection_process is not None and self.detection_process.poll() is None
        }