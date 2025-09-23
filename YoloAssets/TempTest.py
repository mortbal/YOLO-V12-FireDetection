#!/usr/bin/env python3
from ultralytics import YOLO
import os
import torch
import numpy as np
import cv2

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def custom_nested_nms(boxes, scores, classes, conf_threshold=0.01, size_ratio_threshold=0.7):
    """
    Custom NMS that preserves nested objects with significant size differences
    Optimized for fire-in-smoke detection scenarios
    """
    if len(boxes) == 0:
        return []

    # Filter by confidence first
    valid_indices = [i for i, score in enumerate(scores) if score >= conf_threshold]
    if not valid_indices:
        return []

    # Sort by confidence (highest first)
    sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)

    keep = []
    suppressed = set()

    for i in sorted_indices:
        if i in suppressed:
            continue

        keep.append(i)
        box_i = boxes[i]
        class_i = classes[i]

        # Check against all remaining boxes
        for j in sorted_indices:
            if j <= i or j in suppressed:
                continue

            box_j = boxes[j]
            class_j = classes[j]

            # Always keep all boxes - no suppression for nested fire/smoke detection
            # This preserves all detections regardless of overlap or class

    return keep

def main():
    """
    Run trained YOLO model on test files with predefined paths
    """
    # Get script directory and build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    model_path = os.path.join(script_dir, "Trains", "yolov9-c-fire.pt")
    test_files_dir = os.path.join(project_root, "DetectionTest", "TestFiles")
    output_dir = os.path.join(project_root, "DetectionTest", "ResultFiles")

    print("[STATUS] INITIALIZING TempTest")

    # Validate inputs
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    if not os.path.exists(test_files_dir):
        print(f"[ERROR] Test files directory not found: {test_files_dir}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading model: {model_path}")
    print("[STATUS] LOADING_MODEL")

    # Configure device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    if device.startswith('cuda') and torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Load model
    model = YOLO(model_path)

    print("[STATUS] MODEL_LOADED")
    print(f"[INFO] Model loaded successfully, using {device} for inference")

    print("[STATUS] DETECTION_STARTED")
    print(f"[INFO] Processing test files from: {test_files_dir}")
    print(f"[INFO] Output directory: {output_dir}")

    # Run inference with disabled NMS to get all raw detections
    # Get all images/videos in test directory
    import glob
    test_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.mp4', '*.avi', '*.mov']:
        test_files.extend(glob.glob(os.path.join(test_files_dir, ext)))
        test_files.extend(glob.glob(os.path.join(test_files_dir, ext.upper())))

    # Remove duplicates
    test_files = list(set(test_files))

    print(f"[INFO] Found {len(test_files)} files to process")

    # Process each file individually for custom NMS
    all_results = []
    for test_file in test_files:
        print(f"[INFO] Processing: {os.path.basename(test_file)}")

        # Process with optimized parameters for fire-in-smoke detection
        if test_file.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"[INFO] Processing entire video: {os.path.basename(test_file)}")
            raw_results = model.predict(
                source=test_file,
                conf=0.01,                   # Even lower confidence to catch more detections
                iou=0.6,                    # Allow some overlap but not extreme
                device=device,
                imgsz=640,
                save=False,
                verbose=False,
                stream=True
            )
        else:
            image = cv2.imread(test_file)
            if image is None:
                print(f"[ERROR] Could not read image: {os.path.basename(test_file)}")
                continue

            raw_results = model.predict(
                source=image,
                conf=0.01,                   # Even lower confidence to catch more detections
                iou=0.6,                    # Allow some overlap but not extreme
                device=device,
                imgsz=640,
                save=False,
                verbose=False
            )

        # For videos, create output video writer
        if test_file.lower().endswith(('.mp4', '.avi', '.mov')):
            filename = os.path.basename(test_file)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_detected.mp4")

            # Get video properties
            cap = cv2.VideoCapture(test_file)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        for result in raw_results:
            frame_count += 1
            if result.boxes is not None and len(result.boxes) > 0:
                # Extract detection data
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                # Apply custom nested-aware NMS
                keep_indices = custom_nested_nms(boxes, scores, classes,
                                               conf_threshold=0.01,   # Match YOLO conf
                                               size_ratio_threshold=0.7)  # Better for fire-in-smoke

                if keep_indices:
                    result.boxes = result.boxes[keep_indices]
                else:
                    result.boxes = None

            # Save result
            if test_file.lower().endswith(('.mp4', '.avi', '.mov')):
                # Write frame to video
                if result.boxes is not None and len(result.boxes) > 0:
                    annotated = result.plot(labels=True, conf=True, line_width=2)
                    out.write(annotated)
                else:
                    out.write(result.orig_img)

                # Show progress percentage (update same line)
                progress = (frame_count / total_frames) * 100
                print(f"\r[INFO] Processing {os.path.basename(test_file)}: {progress:.1f}% ({frame_count}/{total_frames})", end='', flush=True)
            else:
                # Save image
                filename = os.path.basename(test_file)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_detected{ext}")

                if result.boxes is not None and len(result.boxes) > 0:
                    annotated = result.plot(labels=True, conf=True, line_width=2)
                    cv2.imwrite(output_path, annotated)
                else:
                    cv2.imwrite(output_path, result.orig_img)

            all_results.append(result)

        # Close video writer
        if test_file.lower().endswith(('.mp4', '.avi', '.mov')):
            out.release()
            print()  # New line after progress updates

    results = all_results

    # Count and report results
    total_detections = 0
    processed_files = 0

    for result in results:
        processed_files += 1
        if result.boxes is not None:
            detections_in_file = len(result.boxes)
            total_detections += detections_in_file
            filename = os.path.basename(result.path)
        else:
            filename = os.path.basename(result.path)

    # Summary
    print(f"\n[INFO] Detection completed!")
    print(f"[INFO] Processed files: {processed_files}")
    print(f"[INFO] Total detections: {total_detections}")
    print(f"[INFO] Results saved to: {output_dir}")
    print("[STATUS] DETECTION_COMPLETE")

if __name__ == "__main__":
    main()