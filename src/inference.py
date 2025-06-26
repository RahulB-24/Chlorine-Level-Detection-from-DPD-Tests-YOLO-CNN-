# inference.py
import torch
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image, ImageDraw
import os
import sys

# Paths
YOLO_MODEL_PATH = 'models/test_tube_detector/weights/best.pt'
CNN_MODEL_PATH = 'models/cnn_chlorine_best3.pth'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Load CNN model
cnn_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 1)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
cnn_model.eval()

# Prediction Function
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    results = yolo_model(image)[0]

    draw = ImageDraw.Draw(image)

    if results.boxes is None or len(results.boxes) == 0:
        print("⚠️ No test tube detected. Using full image for prediction.")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = cnn_model(input_tensor).item()
        draw.text((10, 10), f"{pred:.2f} PPM (Full Image)", fill="orange")
    else:
        # Select the box with the largest area
        boxes = results.boxes.xyxy.cpu().numpy()
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        best_idx = areas.index(max(areas))
        best_box = boxes[best_idx]
        x1, y1, x2, y2 = map(int, best_box[:4])
        cropped = image.crop((x1, y1, x2, y2))
        input_tensor = transform(cropped).unsqueeze(0)
        with torch.no_grad():
            pred = cnn_model(input_tensor).item()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{pred:.2f} PPM", fill="red")

    # Save output image
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    image.save(output_path)
    print(f"✅ Saved: {output_path}, Prediction: {pred:.2f} PPM")

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
