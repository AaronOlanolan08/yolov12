import torch
from ultralytics import YOLO

def main():
    # Path to your custom YOLOv12 config
    config_path = 'ultralytics/cfg/models/v12/yolov12.yaml'
    # Dummy dataset (COCO128 is a small default for quick tests)
    data_path = 'ultralytics/cfg/datasets/coco128.yaml'

    # Create model from config
    model = YOLO(config_path)

    # Quick test train (1 epoch, small batch, small imgsz)
    results = model.train(
        data=data_path,
        epochs=1,
        imgsz=320,
        batch=2,
        device='cpu' if not torch.cuda.is_available() else 0,
        workers=0,
        verbose=True
    )
    print('Test train complete. Results:', results)

if __name__ == '__main__':
    main()
