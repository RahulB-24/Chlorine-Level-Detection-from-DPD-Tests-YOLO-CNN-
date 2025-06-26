# src/app.py
import torch
from ultralytics import YOLO
from torchvision import transforms, models
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import os

app = Flask(__name__)
CORS(app)

YOLO_MODEL_PATH = r'C:\Users\etern\Desktop\test tube\models\test_tube_detector\weights\best.pt'
CNN_MODEL_PATH = r'C:\Users\etern\Desktop\test tube\models\cnn_chlorine_best3.pth'
OUTPUT_DIR = 'src/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
yolo_model = YOLO(YOLO_MODEL_PATH)

cnn_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 1)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    results = yolo_model(image)[0]
    draw = ImageDraw.Draw(image)

    if results.boxes is None or len(results.boxes) == 0:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = cnn_model(input_tensor).item()
        draw.text((10, 10), f"{pred:.2f} PPM (Full Image)", fill="orange")
    else:
        boxes = results.boxes.xyxy.cpu().numpy()
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        best_idx = areas.index(max(areas))
        x1, y1, x2, y2 = map(int, boxes[best_idx])
        cropped = image.crop((x1, y1, x2, y2))
        input_tensor = transform(cropped).unsqueeze(0)
        with torch.no_grad():
            pred = cnn_model(input_tensor).item()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{pred:.2f} PPM", fill="red")

    output_path = os.path.join(OUTPUT_DIR, file.filename)
    image.save(output_path)

    return jsonify({
        "prediction": round(pred, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
