from ultralytics import YOLO
import os
import argparse
import torch

def main():
    try:
        print("[STATUS] INITIALIZING")
        
        parser = argparse.ArgumentParser(description='Run fire detection using YOLO native capabilities')
        parser.add_argument('--model', type=str, required=True, help='Path to trained model file (.pt)')
        parser.add_argument('--source', type=str, required=True, help='Path to source folder/file containing test data')
        parser.add_argument('--output', type=str, required=True, help='Path to output folder for results')
        parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
        parser.add_argument('--device', type=str, default='', help='Device to run on (cuda, cpu, or auto-detect)')
        parser.add_argument('--imgsz', type=int, default=640, help='Input image size (default: 640)')
        parser.add_argument('--half', action='store_true', help='Use FP16 half-precision inference')
        parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing (default: 1)')
        
        args = parser.parse_args()
        
        # Validate inputs
        if not os.path.exists(args.model):
            print(f"[ERROR] Model file not found: {args.model}")
            print("[STATUS] DETECTION_FAILED")
            return
        
        if not os.path.exists(args.source):
            print(f"[ERROR] Source path not found: {args.source}")
            print("[STATUS] DETECTION_FAILED")
            return
        
        # Create output folder
        os.makedirs(args.output, exist_ok=True)
        
        print(f"[INFO] Loading model: {args.model}")
        print("[STATUS] LOADING_MODEL")
        
        # Configure device
        if args.device == '':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        print(f"[INFO] Using device: {device}")
        if device.startswith('cuda') and torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Load model
        model = YOLO(args.model)
        
        print("[STATUS] MODEL_LOADED")
        print(f"[INFO] Model loaded successfully, using {device} for inference")
        
        print("[STATUS] DETECTION_STARTED")
        print(f"[INFO] Processing source: {args.source}")
        print(f"[INFO] Output directory: {args.output}")
        
        # Run inference using YOLO's native capabilities
        # This handles directories, videos, images, and all formats automatically!
        results = model(
            source=args.source,              # Input source (file, dir, URL, etc.)
            conf=args.conf,                  # Confidence threshold
            device='cuda:0',                   # Device to use
            imgsz=args.imgsz,               # Image size
            half=args.half,                 # FP16 inference
            batch=args.batch_size,          # Batch size
            save=True,                      # Save annotated images/videos
            project=args.output,            # Project directory
            name='results',                 # Experiment name
            exist_ok=True,                  # Overwrite existing results
            show_labels=True,               # Show labels
            show_conf=True,                 # Show confidence scores
            save_txt=False,                 # Don't save label files
            save_crop=False,                # Don't save cropped detections
            show=False,                     # Don't display results
            verbose=True                    # Enable verbose output
        )
        
        # Count results
        total_detections = 0
        processed_files = 0
        
        for result in results:
            processed_files += 1
            if result.boxes is not None:
                detections_in_image = len(result.boxes)
                total_detections += detections_in_image
                print(f"[INFO] Processed: {result.path} - {detections_in_image} detections")
            else:
                print(f"[INFO] Processed: {result.path} - 0 detections")
        
        # Summary
        print(f"\n[INFO] Detection completed!")
        print(f"[INFO] Processed files: {processed_files}")
        print(f"[INFO] Total detections: {total_detections}")
        print(f"[INFO] Results saved to: {os.path.join(args.output, 'results')}")
        
        print("[STATUS] DETECTION_COMPLETE")
        
    except KeyboardInterrupt:
        print("\n[STATUS] DETECTION_CANCELLED")
        print("[INFO] Detection cancelled by user")
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        print("[STATUS] DETECTION_FAILED")

if __name__ == "__main__":
    main()