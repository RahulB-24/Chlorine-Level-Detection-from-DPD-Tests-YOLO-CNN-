# Chlorine Level Detection from DPD Tests (YOLO + CNN)

A full-stack AI-powered web application that detects chlorine levels (0–5) from images using a hybrid deep learning approach. This project automates chlorine level detection in water using the DPD (N,N-diethyl-p-phenylenediamine) test method. A YOLO model detects the DPD color region from a test image, which is then classified using a CNN into chlorine levels (0–5 ppm). The web interface allows users to upload an image and get instant predictions with visual feedback.

---

## 📌 Features

- 🔍 YOLOv8 for region-of-interest detection
- 🧠 CNN for classification of chlorine level (0 to 5)
- 🌐 React.js frontend with full-screen preview and animation
- 🎨 Animated chlorine & water background for enhanced UX
- 📤 Upload any image and receive predictions in real time
- 📊 Prediction displayed as an animated slider

---

## ⚙️ How It Works (End-to-End)

1. **Image Upload (Frontend)**  
   User uploads an image through the React.js interface. The image is sent to the backend via an API call.

2. **Detection & Cropping (Backend - YOLOv8)**  
   The Flask backend receives the image and uses a pre-trained YOLOv8 model (`yolov8n.pt`) to detect and crop the region where chlorine is likely to be found (e.g., bottles or containers).

3. **Classification (Backend - CNN)**  
   The cropped image is passed into a trained CNN model that classifies the chlorine level from 0 to 5. The prediction is converted to an integer score.

4. **Response (Backend to Frontend)**  
   The prediction score is returned as JSON. The frontend updates the UI by showing the original image, overlay effects, and an interactive slider indicating the chlorine level.

---

## 🛠️ Tech Stack

### 🔧 Backend
- Python 3.10+
- Flask (API server)
- PyTorch (YOLOv8)
- TensorFlow/Keras (CNN classification)
- OpenCV (image processing)

### 🌐 Frontend
- React.js
- Axios (HTTP requests)
- Custom CSS (no Tailwind)
- Animations and slider display

---

## 📁 Project Structure

chlorine-detection/  
├── frontend/             → React app (UI)  
├── models/               → Trained CNN weights and YOLO weights  
├── src/  
│   ├── inference.py      → Main backend inference logic  
│   ├── train_cnn.py      → CNN training script  
│   ├── yolo_crop.py      → Crops images using YOLO  
│   └── outputs/          → Stores YOLO output images (ignored by Git)  
├── yolov8n.pt            → YOLOv8 model weights  
├── requirements.txt      → Backend dependencies  
├── data.yaml             → YOLO training config  
├── chlorine_gui.py       → Optional GUI launcher  
└── .gitignore, README.md  

---

## 🚀 How to Run the Project

### 🔌 Prerequisites
- Python 3.10+
- Node.js & npm
- Git installed

---

### 🐍 Backend Setup

1. Create a virtual environment  
   `python -m venv venv`

2. Activate the environment  
   - On macOS/Linux: `source venv/bin/activate`  
   - On Windows: `venv\Scripts\activate`

3. Install dependencies  
   `pip install -r requirements.txt`

4. Start the backend server  
   `python src/inference.py`  
   Runs on: `http://127.0.0.1:5000`

---

### 🌐 Frontend Setup

1. Navigate to the frontend directory  
   `cd frontend`

2. Install React dependencies  
   `npm install`

3. Start the React development server  
   `npm start`  
   Runs on: `http://localhost:3000`

---

## 🔁 Full Workflow

- User uploads an image from the browser
- Flask backend receives it and:
  - Uses YOLO to crop the chlorine container region
  - Passes it to CNN to classify the level
- Returns prediction to frontend
- Frontend displays it with image + animated slider

---

## 📦 Deployment

- **Backend** can be deployed via Render, Railway, or Heroku  
- **Frontend** can be hosted on Vercel or Netlify  
- Make sure to replace the API endpoint in React with your deployed backend URL


---

## 🛡️ License

Licensed under the **MIT License**.  
You are free to use, modify, and distribute this project with attribution.

---

## 🙋‍♂️ Author

**Rahul Balachandar**  
GitHub: https://github.com/RahulB-24  
Email: rahulbalachandar24@gmail.com
