from ultralytics import YOLO
import cv2
import os
from datetime import datetime

def list_available_models():
    """List all available trained models in LastTrainData folder"""
    lasttraindata_path = os.path.join(os.getcwd(), 'LastTrainData')
    
    if not os.path.exists(lasttraindata_path):
        print("LastTrainData folder not found!")
        return []
    
    subfolders = [f for f in os.listdir(lasttraindata_path) 
                  if os.path.isdir(os.path.join(lasttraindata_path, f))]
    
    if not subfolders:
        print("No trained models found in LastTrainData folder!")
        return []
    
    print("Available trained models:")
    print("-" * 30)
    for i, folder in enumerate(subfolders, 1):
        print(f"{i}. {folder}")
    
    return subfolders

def select_model(available_models):
    """Let user select a model by number"""
    while True:
        try:
            choice = int(input(f"\nSelect model (1-{len(available_models)}): "))
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number")

def test_model_on_video(model_folder):
    """Run selected model on TestVideo.mp4 and save results"""
    # Load the model
    model_path = os.path.join(os.getcwd(), 'LastTrainData', model_folder, 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check if test video exists
    video_path = os.path.join(os.getcwd(), 'TestVideo.mp4')
    if not os.path.exists(video_path):
        print(f"Test video not found: {video_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video filename with datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"ResultVideo_{model_folder}_{current_time}.mp4"
    output_path = os.path.join(os.getcwd(), output_filename)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video...")
    print(f"Output will be saved as: {output_filename}")
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        results = model.predict(frame, conf=0.25)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Show progress
        if frame_count % 30 == 0:  # Print progress every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"\nVideo processing completed!")
    print(f"Result saved as: {output_filename}")

def main():
    print("Fire Detection Model Tester")
    print("=" * 40)
    
    # List available models
    available_models = list_available_models()
    
    if not available_models:
        return
    
    # Let user select model
    selected_model = select_model(available_models)
    print(f"\nSelected model: {selected_model}")
    
    # Test the model on video
    test_model_on_video(selected_model)

if __name__ == "__main__":
    main()