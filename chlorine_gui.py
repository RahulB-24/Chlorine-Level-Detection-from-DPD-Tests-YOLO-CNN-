import os
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights


# === CONFIGURATION ===
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_WEIGHTS = "models/test_tube_detector/weights/best.pt"
    CNN_WEIGHTS = "models/cnn_weights.pth"
    IMAGE_SIZE = 224
    SAVE_DIR = "inference_results"
    CONFIDENCE_PRECISION = 3
    YOLO_CONF_THRESH = 0.3


def load_cnn_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 1)
    )

    checkpoint = torch.load(Config.CNN_WEIGHTS, map_location=Config.DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Loaded CNN weights with best validation R² = {checkpoint.get('best_val_r2', 'N/A')}")
    else:
        raise RuntimeError("❌ Invalid checkpoint format or missing 'model_state_dict'.")

    model.to(Config.DEVICE)
    model.eval()
    preprocess = ResNet50_Weights.DEFAULT.transforms()
    return model, preprocess


def draw_and_save(image, box, ppm, save_path):
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red", width=4)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.text((box[0] + 5, box[1] + 5), f"{ppm} PPM", fill="yellow", font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    print(f"📸 Saved result to: {save_path}")


class WaterDroplet:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.r = random.uniform(5, 10)
        self.speed = random.uniform(0.8, 2.5)
        self.id = None
        self.alpha = random.uniform(0.1, 0.4)

    def update(self):
        self.y += self.speed
        if self.y - self.r > self.height:
            self.y = -self.r
            self.x = random.uniform(0, self.width)
        # Draw with alpha by using fill with stipple workaround
        if self.id:
            self.canvas.delete(self.id)
        # Stipple patterns simulate transparency
        stipple_patterns = ['gray12', 'gray25', 'gray50', 'gray75']
        # Choose stipple based on alpha (lower alpha = lighter stipple)
        if self.alpha < 0.15:
            stipple = stipple_patterns[0]
        elif self.alpha < 0.25:
            stipple = stipple_patterns[1]
        elif self.alpha < 0.35:
            stipple = stipple_patterns[2]
        else:
            stipple = stipple_patterns[3]
        self.id = self.canvas.create_oval(self.x - self.r, self.y - self.r,
                                          self.x + self.r, self.y + self.r,
                                          fill="#00b4d8", outline="", stipple=stipple)


class ChlorineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chlorine Detection")
        self.root.geometry("900x700")
        self.root.configure(bg="#e6f0f7")

        # Load models
        self.yolo_model = YOLO(Config.YOLO_WEIGHTS)
        self.cnn_model, self.preprocess = load_cnn_model()

        # GUI Elements
        self.create_widgets()

        # Water droplets animation
        self.droplets = []
        self.init_droplets()
        self.animate_droplets()

    def create_widgets(self):
        # Canvas for background + water animation
        self.bg_canvas = tk.Canvas(self.root, width=900, height=700, bg="#e6f0f7", highlightthickness=0)
        self.bg_canvas.place(x=0, y=0)

        # Frame for widgets
        self.frame = tk.Frame(self.root, bg="#e6f0f7")
        self.frame.place(relx=0.5, rely=0.05, anchor="n")

        # Upload button
        self.upload_btn = tk.Button(self.frame, text="Upload Image", font=("Helvetica", 16, "bold"),
                                    bg="#007acc", fg="white", padx=15, pady=10,
                                    command=self.upload_image)
        self.upload_btn.pack(pady=10)

        # Image display
        self.image_label = tk.Label(self.root, bg="#e6f0f7")
        self.image_label.place(relx=0.5, rely=0.25, anchor="n")

        # Status Label
        self.status_label = tk.Label(self.root, text="", font=("Helvetica", 18, "bold"),
                                     bg="#e6f0f7", fg="#004080")
        self.status_label.place(relx=0.5, rely=0.65, anchor="n")

        # Chlorine Scale Canvas
        self.scale_canvas = tk.Canvas(self.root, width=600, height=80, bg="#e6f0f7", highlightthickness=0)
        self.scale_canvas.place(relx=0.5, rely=0.8, anchor="n")
        self.draw_scale()

    def draw_scale(self):
        c = self.scale_canvas
        c.delete("all")
        c.create_line(50, 40, 550, 40, width=4, fill="#007acc")
        for i in range(6):
            x = 50 + i * 100
            c.create_line(x, 30, x, 50, width=3, fill="#004080")
            c.create_text(x, 60, text=str(i), font=("Helvetica", 14, "bold"), fill="#004080")

        self.pointer = c.create_polygon(50, 20, 60, 10, 70, 20, fill="#ff4500")

    def move_pointer(self, ppm):
        ppm = max(0, min(5, ppm))
        x = 50 + (ppm * 100)
        c = self.scale_canvas
        c.coords(self.pointer, x - 10, 20, x, 10, x + 10, 20)

    def init_droplets(self):
        for _ in range(60):
            droplet = WaterDroplet(self.bg_canvas, 900, 700)
            self.droplets.append(droplet)

    def animate_droplets(self):
        self.bg_canvas.delete("all")
        for drop in self.droplets:
            drop.update()
        self.root.after(50, self.animate_droplets)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not file_path:
            return

        # Display the original image
        self.display_image(file_path)

        # Run detection + prediction asynchronously
        threading.Thread(target=self.run_detection, args=(file_path,), daemon=True).start()

    def display_image(self, path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((600, 400))
        self.photo_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo_img)

    def run_detection(self, image_path):
        self.status_label.config(text="Running chlorine test...")
        time.sleep(1.5)  # For animation effect

        original_img = Image.open(image_path).convert("RGB")
        yolo_results = self.yolo_model(image_path, conf=Config.YOLO_CONF_THRESH)[0]

        boxes = yolo_results.boxes
        if len(boxes) == 0:
            self.status_label.config(text="⚠️ No test tube detected.")
            return

        # Extract boxes and confidences, pick the one with highest confidence
        boxes_xy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()

        best_idx = scores.argmax()
        x1, y1, x2, y2 = boxes_xy[best_idx].astype(int)

        # Crop detected test tube region
        cropped = original_img.crop((x1, y1, x2, y2))

        # Preprocess for CNN
        input_tensor = self.preprocess(cropped).unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            output = self.cnn_model(input_tensor)
            raw_ppm = output.item()
            clamped_ppm = max(0.0, min(5.0, raw_ppm))
            ppm = round(clamped_ppm, Config.CONFIDENCE_PRECISION)

        message = f"Detected Chlorine: {ppm} PPM"
        if ppm < 2:
            add_amt = round(2 - ppm, 2)
            message += f"\nAdd {add_amt} PPM chlorine to reach 2."
        elif ppm > 2:
            dil_amt = round(ppm - 2, 2)
            message += f"\nDilute to reduce chlorine by {dil_amt} PPM."

        self.status_label.config(text=message)
        self.move_pointer(ppm)

        # Draw bounding box + ppm text and save
        save_path = os.path.join(Config.SAVE_DIR, f"pred_{os.path.basename(image_path)}")
        draw_and_save(original_img.copy(), (x1, y1, x2, y2), ppm, save_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChlorineApp(root)
    root.mainloop()
