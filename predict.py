from ultralytics import YOLO

def run_prediction():
    # 1. Naye retrained model ke weights load karein
    model = YOLO('runs/detect/train/weights/best.pt')

    # 2. Prediction run karein
    results = model.predict(
        source='tested_images',   # Single image path (e.g. 'test_car.jpg') ya poora folder path
        conf=0.70,              # Standard 50% confidence threshold
        save=True,              # Output image par bounding boxes draw karke save karega
        device=0                # GPU speedup ke liye
    )

    print("\nInference Complete!")
    print("Result images dekhne ke liye yeh folder kholein: runs/detect/predict/")

if __name__ == '__main__':
    run_prediction()