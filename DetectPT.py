from ultralytics import YOLO
import cv2
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='video source (0 for webcam or video file path)')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    return parser.parse_args()

# Load the trained model
try:
    model = YOLO('runs/detect/fire_detection_model/weights/best.pt')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit()

def detect_fire():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    prev_time = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Run detection on frame
            results = model.predict(frame, conf=args.conf)
            
            # Get detection results
            boxes = results[0].boxes
            if len(boxes) > 0:
                # Alert if fire detected
                cv2.putText(frame, 'FIRE DETECTED!', (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show confidence for each detection
                for box in boxes:
                    conf = float(box.conf)
                    cv2.putText(frame, f'Conf: {conf:.2f}', 
                              (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw FPS on frame
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Fire Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_fire()
