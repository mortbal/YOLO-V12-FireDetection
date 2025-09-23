"""
File Manager Module
Handles file system operations and model discovery
"""

import os
from shared_types import UpdateType

class FileManager:
    def __init__(self, emit_to_frontend, log_update):
        self.emit_to_frontend = emit_to_frontend
        self.log_update = log_update


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