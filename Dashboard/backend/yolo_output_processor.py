# YOLO Output Processor
# Handles parsing and processing of YOLO training output lines

import re
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from shared_types import UpdateType



@dataclass
class YoloUpdate:
    update_type: UpdateType
    data: Dict[str, Any]
    raw_line: str
    timestamp: float

class YoloOutputProcessor:
    def __init__(self):
        self.training_start_time = None

    def set_training_start_time(self, start_time: float):
        # Set the training start time for elapsed calculations
        self.training_start_time = start_time

    def process_line(self, line: str) -> Optional[YoloUpdate]:
        # Process a single line of YOLO output and return structured update
        # Args: line: Raw output line from YOLO training
        # Returns: YoloUpdate object or None if line should be ignored
        line = line.strip()
        if not line:
            return None

        timestamp = time.time()

        # Process different types of lines
        if "[STATUS]" in line:
            return self._process_status_line(line, timestamp)
        elif "[DEBUG]" in line:
            return self._process_debug_line(line, timestamp)
        elif "it/s" in line and "%" in line:
            return self._process_training_progress_line(line, timestamp)
        elif "Fast image access" in line:
            return YoloUpdate(
                update_type=UpdateType.STATUS,
                data={"status": "discovering_dataset", "message": "Discovering dataset..."},
                raw_line=line,
                timestamp=timestamp
            )
        else:
            # Return as verbose for any other line
            return YoloUpdate(
                update_type=UpdateType.VERBOSE,
                data={"message": line},
                raw_line=line,
                timestamp=timestamp
            )

    def _process_status_line(self, line: str, timestamp: float) -> YoloUpdate:
        # Process [STATUS] lines
        status_map = {
            "[STATUS] INITIALIZING": {"status": "initializing", "message": "Initializing..."},
            "[STATUS] TRAINING_STARTING": {"status": "warming_up", "message": "Warming up..."},
            "[STATUS] VALIDATION_STARTING": {"status": "validating", "message": "Running validation..."},
            "[STATUS] TRAINING_STARTED": {"status": "training", "message": "Training in progress..."},
            "[STATUS] MODEL_LOADED": {"status": "model_loaded", "message": "Model loaded successfully"},
            "[STATUS] SUCCESS": {"status": "completed", "message": "Training completed successfully!"},
            "[STATUS] FAILED": {"status": "failed", "message": "Training failed"},
        }

        for status_key, data in status_map.items():
            if status_key in line:
                return YoloUpdate(
                    update_type=UpdateType.STATUS,
                    data=data,
                    raw_line=line,
                    timestamp=timestamp
                )

        # Unknown status line
        return YoloUpdate(
            update_type=UpdateType.STATUS,
            data={"status": "unknown", "message": line},
            raw_line=line,
            timestamp=timestamp
        )

    def _process_debug_line(self, line: str, timestamp: float) -> YoloUpdate:
        # Process [DEBUG] lines
        return YoloUpdate(
            update_type=UpdateType.DEBUG,
            data={"message": line},
            raw_line=line,
            timestamp=timestamp
        )


    def _process_training_progress_line(self, line: str, timestamp: float) -> Optional[YoloUpdate]:
        # Process training progress lines like:
        # 1/100      12.4G      2.141      3.998      2.014        223        640: 9% ━─────────── 159/1702 4.6it/s 38.0s<5:37
        try:
            parts = line.split()

            # Extract epoch info (first part like "1/100")
            epoch_info = parts[0] if "/" in parts[0] else None
            if not epoch_info:
                return None

            current_epoch, total_epochs = map(int, epoch_info.split('/'))

            # Initialize parsed data
            progress_data = {
                'epoch': current_epoch,
                'total_epochs': total_epochs,
                'progress_percent': 0,
                'gpu_memory': None,
                'current_batch': 0,
                'total_batches': 0,
                'iteration_speed': None,
                'elapsed_time': None,
                'remaining_time': None,
                'total_elapsed_seconds': time.time() - self.training_start_time if self.training_start_time else 0
            }

            # Parse each part of the line
            for part in parts:
                # GPU memory (like "12.4G")
                if 'G' in part and self._is_gpu_memory(part):
                    progress_data['gpu_memory'] = part

                # Progress percentage (like "9%")
                elif '%' in part:
                    try:
                        progress_data['progress_percent'] = float(part.replace('%', ''))
                    except ValueError:
                        pass

                # Batch info (like "159/1702")
                elif '/' in part and part != epoch_info:
                    try:
                        batch_parts = part.split('/')
                        if len(batch_parts) == 2 and batch_parts[0].isdigit() and batch_parts[1].isdigit():
                            progress_data['current_batch'] = int(batch_parts[0])
                            progress_data['total_batches'] = int(batch_parts[1])
                    except ValueError:
                        pass

                # Iteration speed (like "4.6it/s")
                elif 'it/s' in part:
                    progress_data['iteration_speed'] = part

                # Time info (like "38.0s<5:37")
                elif '<' in part and ('s' in part or ':' in part):
                    try:
                        elapsed_time, remaining_time = part.split('<')
                        progress_data['elapsed_time'] = elapsed_time
                        progress_data['remaining_time'] = remaining_time
                    except ValueError:
                        pass

            return YoloUpdate(
                update_type=UpdateType.STATUS,
                data=progress_data,
                raw_line=line,
                timestamp=timestamp
            )

        except Exception as e:
            return YoloUpdate(
                update_type=UpdateType.DEBUG,
                data={"error": str(e), "line": line},
                raw_line=line,
                timestamp=timestamp
            )

    def _is_gpu_memory(self, part: str) -> bool:
        # Check if a part represents GPU memory (like '12.4G')
        try:
            # Remove 'G' and check if remaining is a valid number
            memory_str = part.replace('G', '')
            float(memory_str)
            return True
        except ValueError:
            return False