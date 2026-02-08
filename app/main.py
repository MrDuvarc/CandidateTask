from fastapi import FastAPI
import uvicorn
from ultralytics import YOLO

app = FastAPI(title="Object Detection API", version="1.0")

#Model Loading
print("Loading model...")
try:
    #Relative path and task specification for ONNX model
    model = YOLO("yolov8m.pt", task="detect")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

#Health Check Endpoint
@app.get("/")
def home():
    return {"message": "Object Detection API is running!", "model_ready": model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 