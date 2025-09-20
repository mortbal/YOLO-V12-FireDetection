"""
File Manager Module
Handles file system operations and model discovery
"""

import os
import tkinter as tk
from tkinter import filedialog
from shared_types import UpdateType

class FileManager:
    def __init__(self, emit_to_frontend, log_update):
        self.emit_to_frontend = emit_to_frontend
        self.log_update = log_update

    def browse_folder(self, folder_type: str):
        """
        Open folder browser dialog

        Args:
            folder_type: Type of folder ('test', 'result', or other)

        Returns:
            dict: Status response with folder path
        """
        try:
            # Create a hidden tkinter root window
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.attributes('-topmost', True)  # Bring to front

            # Set dialog properties based on folder type
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
                return {
                    "status": "success",
                    "folder_path": folder_path
                }
            else:
                return {
                    "status": "cancelled",
                    "message": "No folder selected"
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to open folder dialog: {str(e)}"
            }

    def get_all_models(self):
        """
        Get list of all available models from three sources

        Returns:
            dict: Dictionary with baseModels, trainedModels, and checkpointModels lists
        """
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

            # 3. Models from YoloAssets/Trains (ongoing and finished training)
            trains_models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../YoloAssets/Trains"))
            if os.path.exists(trains_models_dir):
                for item in os.listdir(trains_models_dir):
                    model_path = os.path.join(trains_models_dir, item)
                    if os.path.isdir(model_path):
                        if item.startswith('ongoing_run_'):
                            result["checkpointModels"].append(item)
                        elif item.startswith('finished_run_'):
                            result["trainedModels"].append(item)

            return result

        except Exception as e:
            self.log_update(UpdateType.ERROR, f"Error getting models: {e}")
            return {"baseModels": [], "trainedModels": [], "checkpointModels": []}