from ultralytics import YOLO

def start_training():
    model = YOLO('yolov8n.pt')

    model.train(
        data='data.yaml',
        epochs=60,         # 50 se barha kar 60 kar diya hai
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        
        # --- AUGMENTATION PARAMETERS ---
        degrees=15.0,     # 15 degrees rotation
        flipud=0.5,       # Upside-Down flip (aerial views ke liye best)
        fliplr=0.5,       # Left-Right flip
        mosaic=1.0,       # 4 images combination
        mixup=0.1,        # Blend images
        scale=0.5,        # Zoom-in / Zoom-out (+- 50%)
        hsv_h=0.015,      # Hue adjustment
        hsv_s=0.7,        # Saturation adjustment
        hsv_v=0.4         # Brightness adjustment
    )

if __name__ == '__main__':
    start_training()