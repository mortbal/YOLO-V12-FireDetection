"""
Live Detection Module
Handles real-time camera detection using YOLO models
"""

import threading
import time
import cv2
import numpy as np
import base64
import torch
from ultralytics import YOLO
from shared_types import UpdateType

class LiveDetectionManager:
    def __init__(self, emit_function, log_update):
        self.emit_function = emit_function
        self.log_update = log_update

        # Live detection state variables
        self.live_detection_active = False
        self.live_detection_model = None
        self.live_detection_thread = None

    def start_live_detection(self, model_path: str, camera_index: int = 0):
        """
        Start live camera detection

        Args:
            model_path: Path to the YOLO model file
            camera_index: Camera index (0, 1, 2, etc.)

        Returns:
            dict: Status response
        """
        try:
            # Stop any existing detection
            if self.live_detection_active:
                self.stop_live_detection()

            # Load model
            self.live_detection_model = YOLO(model_path)
            self.live_detection_active = True

            # Start detection thread
            self.live_detection_thread = threading.Thread(
                target=self._run_live_detection_process,
                args=(camera_index,)
            )
            self.live_detection_thread.daemon = True
            self.live_detection_thread.start()

            return {
                "status": "success",
                "message": "Live detection started",
                "camera_index": camera_index
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def stop_live_detection(self):
        """
        Stop live camera detection

        Returns:
            dict: Status response
        """
        try:
            self._stop_live_detection_process()
            return {
                "status": "success",
                "message": "Live detection stopped"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _stop_live_detection_process(self):
        """Helper function to stop live detection"""
        self.live_detection_active = False
        self.live_detection_model = None

        if self.live_detection_thread:
            self.live_detection_thread.join(timeout=2)
            self.live_detection_thread = None

    def _run_live_detection_process(self, camera_index):
        """Run live detection on camera feed using YOLO native streaming"""
        try:
            self.emit_function('live_detection_started', {
                'message': 'Initializing live detection...'
            })

            detection_counts = {'fire': 0, 'smoke': 0, 'other': 0}
            frame_count = 0

            # Use YOLO native streaming with camera source
            self.log_update(UpdateType.DEBUG, f"Starting YOLO stream on camera {camera_index}")

            # Configure device
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.log_update(UpdateType.DEBUG, f"Using device: {device}")

            # Start YOLO streaming inference
            results_generator = self.live_detection_model(
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

            self.emit_function('live_detection_started', {
                'message': 'Live detection started successfully'
            })

            # Process streaming results
            for result in results_generator:
                if not self.live_detection_active:
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
                    self.emit_function('live_detection_frame', {
                        'frame': frame_base64,
                        'detections': detections,
                        'counts': detection_counts,
                        'timestamp': time.time(),
                        'frame_count': frame_count
                    })

                # Small delay for better performance
                time.sleep(0.01)  # ~100 FPS max, but limited by camera/processing speed

            self.emit_function('live_detection_stopped', {
                'message': 'Live detection stopped',
                'total_detections': detection_counts,
                'frames_processed': frame_count
            })

        except Exception as e:
            error_msg = f"Live detection error: {str(e)}"
            self.log_update(UpdateType.ERROR, error_msg)
            self.emit_function('live_detection_error', {
                'error': error_msg
            })

    def is_active(self):
        """Check if live detection is currently active"""
        return self.live_detection_active

    def get_status(self):
        """Get current live detection status"""
        return {
            'active': self.live_detection_active,
            'model_loaded': self.live_detection_model is not None,
            'thread_running': self.live_detection_thread is not None and self.live_detection_thread.is_alive()
        }