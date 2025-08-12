#!/usr/bin/env python3
"""
Script to process media files (mp4, jpeg, png) from TestSample folder
and run YOLO detection, outputting results to TestResults folder.
"""

import os
import cv2
import glob
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def get_model_name():
    """Get the model name from the model file path."""
    return "FireDetection_Epoch10"

def create_output_filename(input_file, model_name):
    """
    Create output filename in format: {OriginalFileName}_{ModelName}_{DateTime}.{OriginalFormat}
    
    Args:
        input_file (str): Path to input file
        model_name (str): Name of the model
        
    Returns:
        str: Output filename
    """
    input_path = Path(input_file)
    original_name = input_path.stem
    original_ext = input_path.suffix
    
    # Get current datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output filename
    output_filename = f"{original_name}_{model_name}_{current_time}{original_ext}"
    return output_filename

def process_image(model, input_path, output_path):
    """
    Process a single image file with YOLO detection.
    
    Args:
        model: YOLO model instance
        input_path (str): Path to input image
        output_path (str): Path to save output image
    """
    try:
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image {input_path}")
            return False
        
        # Run YOLO detection
        results = model.predict(image, conf=0.25)
        
        # Draw detections on image
        annotated_image = results[0].plot()
        
        # Save output image
        cv2.imwrite(output_path, annotated_image)
        print(f"Processed image: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")
        return False

def process_video(model, input_path, output_path):
    """
    Process a single video file with YOLO detection.
    
    Args:
        model: YOLO model instance
        input_path (str): Path to input video
        output_path (str): Path to save output video
    """
    try:
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print(f"Processing video: {input_path} ({total_frames} frames)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection on frame
            results = model.predict(frame, conf=0.25)
            
            # Draw detections on frame
            annotated_frame = results[0].plot()
            
            # Write frame to output video
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Release everything
        cap.release()
        out.release()
        
        print(f"Processed video: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing video {input_path}: {e}")
        return False

def process_test_samples():
    """
    Process all media files in TestSample folder and save results to TestResults folder.
    """
    # Define folder paths
    test_sample_folder = "TestSample"
    test_results_folder = "TestResults"
    
    # Check if TestSample folder exists
    if not os.path.exists(test_sample_folder):
        print(f"Error: {test_sample_folder} folder not found!")
        print("Please create the TestSample folder and add your media files.")
        return
    
    # Create TestResults folder if it doesn't exist
    os.makedirs(test_results_folder, exist_ok=True)
    
    # Load YOLO model
    try:
        model = YOLO('LastTrainData/Epoch10/best.pt')
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Make sure the model file exists at 'LastTrainData/Epoch10/best.pt'")
        return
    
    # Get model name
    model_name = get_model_name()
    
    # Define supported file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    # Find all media files in TestSample folder
    media_files = []
    for ext in image_extensions + video_extensions:
        pattern = os.path.join(test_sample_folder, f"*{ext}")
        media_files.extend(glob.glob(pattern, flags=glob.IGNORECASE))
    
    if not media_files:
        print(f"No supported media files found in {test_sample_folder} folder")
        print(f"Supported formats: {image_extensions + video_extensions}")
        return
    
    print(f"Found {len(media_files)} media files to process")
    
    # Process each file
    processed_count = 0
    for media_file in media_files:
        print(f"\nProcessing: {media_file}")
        
        # Create output filename
        output_filename = create_output_filename(media_file, model_name)
        output_path = os.path.join(test_results_folder, output_filename)
        
        # Get file extension
        file_ext = Path(media_file).suffix.lower()
        
        # Process based on file type
        success = False
        if file_ext in [ext.lower() for ext in image_extensions]:
            success = process_image(model, media_file, output_path)
        elif file_ext in [ext.lower() for ext in video_extensions]:
            success = process_video(model, media_file, output_path)
        
        if success:
            processed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {processed_count}/{len(media_files)} files")
    print(f"Results saved in {test_results_folder} folder")

if __name__ == "__main__":
    process_test_samples()