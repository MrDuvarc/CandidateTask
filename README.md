YOLOv8 ONNX Object Detection Microservice

Overview
This project implements a Dockerized microservice for object detection using YOLOv8 model exported to ONNX format.
The microservice provides a RESTful API implemented with FastAPI and executes model inference through ONNX Runtime.
The API accepts an image file and returns detected objects along with a Base64-encoded rendered image.

Features
-YOLOv8 model exported to ONNX
-ONNX Runtime inference (CPU)
-FastAPI-based REST API
-Optional label-based filtering
-Base64-encoded rendered output image
-Containerized with Docker and Docker Compose

Model
-The model used is YOLOv8m exported to ONNX format.
-The export was performed using the Ultralytics CLI: yolo export model=yolov8m.pt format=onnx
-The resulting file (yolov8m.onnx) is stored inside the app loaded at application startup.
-Inference is executed using ONNX Runtime inside the container.

Why ONNX?
-Provides a representation of the model graph, allowing inference without requiring the original training framework.
-Instead of loading the full Ultralytics stack, inference is executed using ONNX Runtime which reduces runtime overload.
-Models are static and self contained, making them more predictable and easier to deploy.
-Runtime applies graph optimizations such as operator fusion and constant folding, improving inference efficiency.

API Endpoints
Health Check
-GET /
+Returns service status and model state.

Detect All Objects
-POST /detect
+Accepts multipart/form-data with the following field: file → image file

Detect by Label
-POST /detect/{label}
POST /detect/person ==> Returns only detections matching the provided label.

Response Format
The API returns a JSON object with:
-image ==> Base64 encoded JPEG image with bounding boxes
-objects ==> List of detected objects
-count ==> Total number of detections

Each detected object contains:
-label
-x
-y
-width
-height
-confidence

Running the Application
+Requirements
-Docker
-Docker Compose
(No local Python environment is required)

+Build and Start
-From the project root directory: docker compose up --build.
-User should go to http://localhost:8000/docs after the build.
-After the site loads POST/detect ==> Try it out ==> Choose a file ==>Execute
-The output will be visible under Responses.
-Also please don't open yolov8m.onnx file before build time. Somehow VS Code can damage it's structure and an 500 Internal Server Error may occur.

+The service will be available at:
http://localhost:8000
Swagger documentation:
http://localhost:8000/docs

Docker Details
The application runs inside a Python 3.10 slim container.
The container:
-Installs dependencies from requirements.txt
-Copies application source code
-Loads ONNX model at startup
-Runs Uvicorn server on port 8000
-Port 8000 is exposed to the host system.

Testing
-Added 3 images to the project file. The API is manually tested with those images. The results are:
+TestImage1:
0: 640x640 21 persons, 1 car, 1 bus, 2 backpacks, 2 handbags, 449.4ms yolo-api  | Speed: 20.3ms preprocess, 449.4ms inference, 67.3ms postprocess per image at shape (1, 3, 640, 640)
+TestImage2:
0: 640x640 1 person, 1 bench, 1 book, 477.1ms yolo-api  | Speed: 8.2ms preprocess, 477.1ms inference, 40.2ms postprocess per image at shape (1, 3, 640, 640)
+TestImage3:
0: 640x640 5 bananas, 11 apples, 5 oranges, 419.0ms yolo-api  | Speed: 7.0ms preprocess, 419.0ms inference, 32.8ms postprocess per image at shape (1, 3, 640, 640)

Project Structure
app/
---main.py
---yolov8m.onnx
Dockerfile
docker-compose.yml
requirements.txt
README.md
LICENSE
.gitignore
.dockerignore

Version Control and Git Workflow
This project was developed using Git with a structured and incremental workflow.
-Feature-based development approach
-Clear and semantic commit messages
-Logical repository organization
-Conventional commit messages used such as: feat, chore, docs

Repository Structure
-Application source code isolated in app/
-Docker configuration clearly separated
-Sensitive files excluded via .gitignore and .dockerignore

Notes
-The ONNX model must exist inside the app directory.
-The API returns Base64-encoded images as asked.
-The project can be executed consistently across different devices because Docker encapsulates the entire runtime environment; including dependencies, configurations, and system settings.

License
This project is licensed under the MIT License. See the LICENSE file for details.