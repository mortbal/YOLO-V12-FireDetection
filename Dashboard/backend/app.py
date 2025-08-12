from flask import Flask, send_from_directory, request
import subprocess
import sys
import os

app = Flask(__name__)

# Path to frontend index.html
frontend_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../Frontend"))
index_file = "index.html"

@app.route("/")
def serve_index():
    return send_from_directory(frontend_dir, index_file)

@app.route("/train", methods=["POST"])
def run_training():
    model_name = request.form.get("model", "yolo12n")  # default if not provided

    script_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../YoloAssets/TrainYolo.py")
    )

    process = subprocess.Popen(
        [sys.executable, script_path, model_name],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    return {"status": "training_started", "model": model_name}

if __name__ == "__main__":
    # Run Flask server
    app.run(host="0.0.0.0", port=5000, debug=True)
