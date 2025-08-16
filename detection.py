import cv2
from ultralytics import YOLO
import time
import numpy as np
import threading
from collections import defaultdict
from versions import device_setup
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Setup
# -----------------------------
device, dtype = device_setup()

# YOLO model (let Ultralytics pick GPU if available)
model = YOLO("yolov8n.pt")

# DeepSORT tuned for more stable IDs (you can tweak further)
tracker = DeepSort(
    max_age=35,              # keep tracks longer during occlusion
    n_init=3,                # require more hits before confirming (reduces false IDs)
    max_cosine_distance=0.25,# stricter appearance matching
    embedder="mobilenet",    # keep built-in lightweight embedder (or swap for stronger if available)
    nn_budget=150
)

# Allowed classes (set to None or empty set to allow all)
ALLOWED = {"person", "car", "bus", "truck", "motorbike", "bicycle"}

# Smoothing config (EMA)
SMOOTH_ALPHA = 0.50   # higher = smoother but slower to react (0..1)
MIN_BOX_AREA = 300    # ignore very small boxes as false detections (pixels^2)

# Keep EMA state and last seen frame index for cleanup
smooth_boxes = {}           # {track_id: np.array([x1,y1,x2,y2], dtype=float)}
last_seen_frame = {}        # {track_id: frame_idx}
frame_idx = 0
count_objects = {}   # we use this to count to each object
count_lock = threading.Lock()

def video_setup(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    return cap, out


def detector(cap, out, imgsz=416, conf=0.5):
    """
    - Run YOLO per frame (no skipping) at moderate imgsz.
    - Convert YOLO XYXY -> XYWH for deep_sort_realtime.
    - Smooth drawn boxes using EMA on track.to_ltrb() (Kalman-smoothed from tracker).
    - Keep tracker.update_tracks() called every frame.
    """
    global frame_idx, smooth_boxes, last_seen_frame , count_objects

    prev_t = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        classes_names = []  # this will for the all the classes detected per frame
        # YOLO detection (one result per frame)
        #if frame_idx % 2 == 0:
        yres = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]


        # Build detections for DeepSORT (x, y, w, h)
        detections = []
        for xyxy, cls, score in zip(yres.boxes.xyxy, yres.boxes.cls, yres.boxes.conf):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            w, h = x2 - x1, y2 - y1
            cls_name = model.names[int(cls)]
            classes_names.append(cls_name)
            if ALLOWED and cls_name not in ALLOWED:
                continue
            # ignore very small boxes (likely false positives)
            if w * h < MIN_BOX_AREA:
                continue
            detections.append(([x1, y1, w, h], float(score.item()), cls_name))

        # Update tracker every frame (even if detections empty)
        tracks = tracker.update_tracks(detections, frame=frame)

        # Build a set of active track IDs for this frame
        active_ids = set()

        # Draw tracks using Kalman-filtered boxes, but apply EMA smoothing for stability
        for track in tracks:
            if not track.is_confirmed():
                # optionally track.unconfirmed tracks can be ignored
                continue



            track_id = track.track_id
            active_ids.add(track_id)

            # count per id okay
            class_name = track.det_class
            with count_lock:
                if class_name not in count_objects:
                    count_objects[class_name] = set()
                count_objects[class_name].add(track_id)  # we  have got the counts per objects with difference in their ids

            # get Kalman-filtered ltrb from track (floats)
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            if ltrb is None:
                continue

            x1f, y1f, x2f, y2f = ltrb
            # Convert to numpy array for smoothing
            curr_box = np.array([x1f, y1f, x2f, y2f], dtype=float)

            # Initialize EMA if first time
            if track_id not in smooth_boxes:
                smooth_boxes[track_id] = curr_box.copy()
            else:
                prev_box = smooth_boxes[track_id]
                # EMA: new = alpha * prev + (1-alpha) * curr
                smooth_boxes[track_id] = SMOOTH_ALPHA * prev_box + (1.0 - SMOOTH_ALPHA) * curr_box

            # update last seen
            last_seen_frame[track_id] = frame_idx

            # use the smoothed box to draw
            sx1, sy1, sx2, sy2 = smooth_boxes[track_id].astype(int)

            # Keep coordinates inside frame bounds
            h_frame, w_frame = frame.shape[:2]
            sx1 = max(0, min(sx1, w_frame - 1))
            sy1 = max(0, min(sy1, h_frame - 1))
            sx2 = max(0, min(sx2, w_frame - 1))
            sy2 = max(0, min(sy2, h_frame - 1))

            # Skip if extremely small after smoothing
            if (sx2 - sx1) * (sy2 - sy1) < MIN_BOX_AREA:
                continue

            # Draw rectangle and ID
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (40, 200, 40), 2)
            cv2.putText(frame, f"ID:{track_id}", (sx1, max(15, sy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 140, 255), 2)

        # Cleanup smoothing state for tracks not seen for a while
        # Remove entries not in active_ids and not seen for N frames
        CLEANUP_FRAMES = 150
        to_delete = []
        for tid, last in last_seen_frame.items():
            if tid not in active_ids and (frame_idx - last) > CLEANUP_FRAMES:
                to_delete.append(tid)
        for tid in to_delete:
            last_seen_frame.pop(tid, None)
            smooth_boxes.pop(tid, None)

        # FPS display (instantaneous)
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_t))
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ---- Display Object Counts ----

        y_offset = 124
        for cls_name, ids in count_objects.items():
            object_count = len(ids)
            cv2.putText(frame, f"{cls_name}: {object_count}", (12, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 233, 124), 2)
            y_offset += 30


        # Show and save
        cv2.namedWindow("DeepSORT Tracking - Smoothed", cv2.WINDOW_NORMAL)  # allow manual resize
        cv2.resizeWindow("DeepSORT Tracking - Smoothed", 820, 620)  # force window size #3840x2160
        cv2.imshow("DeepSORT Tracking - Smoothed", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def run_detection(cap, out):
    # Use these defaults; reduce imgsz if CPU too slow (320 or 384)
    detector(cap, out, imgsz=416, conf=0.5)

