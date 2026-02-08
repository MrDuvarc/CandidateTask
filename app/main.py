from fastapi import FastAPI, UploadFile, File
import uvicorn
from ultralytics import YOLO
from starlette.responses import StreamingResponse
import io
from PIL import Image

app = FastAPI(title="Object Detection API", version="1.0")

#Model Loading
print("Loading model...")
try:
    # Initialize the model weights and specify the detection task    
    model = YOLO("yolov8m.pt", task="detect")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

#Health Check Endpoint
@app.get("/")
def home():
    return {"message": "Object Detection API is running!", "model_ready": model is not None}


#Detection Endpoint
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded."}
    
    Image_bytes = await file.read()
    image = Image.open(io.BytesIO(Image_bytes))

    results = model(image)

    #Returns BGR numpy array, convert from BGR to RGB 
    plotted_image_bgr = results[0].plot()
    plotted_image_rgb= Image.fromarray(plotted_image_bgr[..., ::-1])
    
    #Save the result to a memory buffer
    img_byte_arr = io.BytesIO()
    plotted_image_rgb.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    #Return the image directly
    return StreamingResponse(img_byte_arr, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 