from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import cv2
import time
from ultralytics import YOLO

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize camera on startup
    global capture
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend for Windows
    if not capture.isOpened():
        raise RuntimeError("Could not open camera")
    
    # Camera warm-up
    time.sleep(2.0)
    capture.read()  # Dummy read to initialize
    
    yield
    
    # Cleanup on shutdown
    capture.release()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = YOLO("AUG_ZJU.pt")

def generate_frames():
    while True:
        try:
            success, frame = capture.read()
            if not success:
                print("Frame capture error, retrying...")
                time.sleep(0.1)
                continue

            results = model(frame)
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

        except Exception as e:
            print(f"Stream error: {str(e)}")
            time.sleep(1)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )