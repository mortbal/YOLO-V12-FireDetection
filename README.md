# 🔥 Fire Detection System

A real-time fire and smoke detection system powered by YOLOv11/YOLOv12 for enhanced safety and early warning.

## ⚡ Features

- **Real-time Detection** - Live webcam and video feed processing
- **High Accuracy** - YOLOv11/YOLOv12 based object detection
- **Multi-format Support** - PyTorch and ONNX inference
- **3-Class Detection** - Fire, Smoke, and Other objects
- **Optimized Performance** - Fast inference with configurable confidence

## 🚀 Quick Start

```bash
# Train the model
python Train.py

# Real-time detection (webcam)
python DetectPT.py --source 0

# Video file detection
python DetectPT.py --source video.mp4 --conf 0.5

# ONNX inference
python DetectFireOnnx.py
```

## 📁 Project Structure

```
├── Train.py              # Model training script
├── DetectPT.py           # PyTorch real-time detection
├── DetectFireOnnx.py     # ONNX optimized inference
├── TestModel.py          # Model performance testing
├── dataset/              # Training data (YOLO format)
│   ├── data.yaml         # Dataset configuration
│   ├── train/            # Training images & labels
│   ├── valid/            # Validation set
│   └── test/             # Test set
└── runs/                 # Training outputs & models
```

## 🎯 Detection Classes

| Class | ID | Description |
|-------|----|----|
| Fire  | 0  | Active flames |
| Other | 1  | General objects |
| Smoke | 2  | Smoke plumes |

## ⚙️ Requirements

- Python 3.8+
- Ultralytics YOLO
- OpenCV
- ONNX Runtime
- PyTorch

## 📊 Model Performance

- **Input Size**: 640x640
- **Architecture**: YOLOv11n/YOLOv12n (nano)
- **Batch Size**: 16
- **Optimized for**: Speed and accuracy balance

## 🔧 Configuration

Update `dataset/data.yaml` for custom datasets:

```yaml
path: ./dataset
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['fire', 'other', 'smoke']
```

## 📈 Usage Examples

```bash
# High confidence detection
python DetectPT.py --source 0 --conf 0.8

# Process video file
python DetectPT.py --source fire_video.mp4

# Test model accuracy
python TestModel.py
```

## 🛠️ Development

Built for safety-critical applications with focus on:
- Low latency inference
- High detection accuracy
- Minimal false positives
- Real-time performance

---

**⚠️ Safety Notice**: This system is designed for early fire detection assistance. Always follow proper fire safety protocols and use professional fire detection systems for critical applications.