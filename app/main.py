from fastapi import FastAPI, UploadFile, File, Path, HTTPException
import uvicorn
from ultralytics import YOLO
import io
from PIL import Image, ImageDraw
import base64
from typing import Optional, List
from pydantic import BaseModel
from pathlib import Path as SysPath

app = FastAPI(title="Object Detection API", version="1.0")

# Response Models
class DetectedObject(BaseModel):
    label: str
    x: int
    y: int
    width: int
    height: int
    confidence: float

class DetectResponse(BaseModel):
    image: str  # base64-encoded image with detections drawn
    objects: List[DetectedObject]
    count: int


# Model Loading
BASE_DIR = SysPath(__file__).resolve().parent # Path to the exported ONNX model
MODEL_PATH = BASE_DIR / "yolov8m.onnx"

print("Loading model from:", MODEL_PATH)

model = None
model_error = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH), task="detect")
    print("Model loaded successfully.")
except Exception as e:
    model_error = str(e)
    print("Model loading failed:", model_error)
    model = None


# Health Check Endpoint
@app.get("/")
def home():
    return {
        "message": "Object Detection API is running!",
        "model_ready": model is not None,
        "model_path": str(MODEL_PATH),
        "model_error": model_error
    }


# Helpers
def img_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_objects(results, label_filter: Optional[str] = None) -> List[DetectedObject]:
    r0 = results[0]
    names = r0.names
    boxes = r0.boxes

    objects: List[DetectedObject] = []

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls_id = int(b.cls.item())
            label = str(names.get(cls_id, cls_id))
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

            if label_filter and label.lower() != label_filter.lower():
                continue

            objects.append(
                DetectedObject(
                    label=label,
                    x=int(round(x1)),
                    y=int(round(y1)),
                    width=int(round(x2 - x1)),
                    height=int(round(y2 - y1)),
                    confidence=round(conf, 4),
                )
            )

    return objects


def draw_boxes(image: Image.Image, results, label_filter: Optional[str] = None) -> Image.Image:
    r0 = results[0]
    names = r0.names
    boxes = r0.boxes
    draw = ImageDraw.Draw(image)

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls_id = int(b.cls.item())
            label = str(names.get(cls_id, cls_id))
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

            if label_filter and label.lower() != label_filter.lower():
                continue

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text(
                (int(x1), max(0, int(y1) - 12)),
                f"{label} {conf:.2f}",
                fill="red"
            )

    return image


# Task Endpoints
@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model not loaded")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)

    rendered = draw_boxes(image.copy(), results, label_filter=None)
    objects = extract_objects(results, label_filter=None)

    return {
        "image": img_to_base64(rendered),
        "objects": objects,
        "count": len(objects),
    }


@app.post("/detect/{label}", response_model=DetectResponse)
async def detect_with_label(
    label: str = Path(..., description="Filter detections by label (e.g. person)"),
    file: UploadFile = File(...),
):
    if model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model not loaded")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)

    rendered = draw_boxes(image.copy(), results, label_filter=label)
    objects = extract_objects(results, label_filter=label)

    return {
        "image": img_to_base64(rendered),
        "objects": objects,
        "count": len(objects),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)