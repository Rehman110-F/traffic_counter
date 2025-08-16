from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from detection import run_detection, video_setup
from pathlib import Path
import threading
from detection import count_objects, count_lock

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite dev server URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
video_path =  Path("input/input5.mp4")
output_path = Path("output/output_video.mp4")
cap , out= video_setup(video_path , output_path)

def video_thread():
    run_detection(cap, out)

threading.Thread(target=video_thread, daemon=True).start()
@app.get("/")
def index():
    print("hello")
    return {"message": "Hello World"}

@app.get("/counts")
def get_counts():
    with count_lock:
        # Return cumulative counts for each class
        return {cls: len(ids) for cls, ids in count_objects.items()}