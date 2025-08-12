from ultralytics import YOLO
import os
import shutil

def save_model_to_custom_location(epoch_name="Epoch10"):
    """Save best model and required files to custom location"""
    custom_save_path = os.path.join(os.getcwd(), 'LastTrainData', epoch_name)
    os.makedirs(custom_save_path, exist_ok=True)
    
    # Copy best.pt model
    best_model_path = os.path.join(os.getcwd(), 'runs/train/fire_detection_model/weights/best.pt')
    shutil.copy2(best_model_path, os.path.join(custom_save_path, 'best.pt'))
    
    # Copy data.yaml (dataset config)
    shutil.copy2('dataset/data.yaml', os.path.join(custom_save_path, 'data.yaml'))
    
    print(f"\nModel and config files saved to: {custom_save_path}")
    print("Files saved: best.pt, data.yaml")
if __name__ == "__main__":
    try:

        # Load the YOLO model
        # You can use 'yolo12n.pt' (smallest), 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt', or 'yolo12x.pt' (largest)
        model = YOLO('yolo12n.pt')

        # Train the model
        model.train(
            data='dataset/data.yaml',
            epochs=1,
            imgsz=640,
            batch=16,
            name='fire_detection_model'
        )

        # Validate the model
        print("Validating model...")
        metrics = model.val()

        # Print training results location
        print("\nTraining completed!")
        print(f"Training results saved in: {os.path.join(os.getcwd(), 'runs/train/fire_detection_model')}")
        print(f"Best model weights saved in: {os.path.join(os.getcwd(), 'runs/train/fire_detection_model/weights/best.pt')}")

        # Export the model (optional)
        # print("\nExporting model to ONNX format...")
        # model.export(format='onnx')
        
        # Save model to custom location
        save_model_to_custom_location()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
