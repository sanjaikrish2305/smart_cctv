"""
Combined Wild-Animal + Weapon + Violence Detector
- TF-Hub SSD MobileNet V2 (COCO) for animals
- YOLOv8 (Ultralytics) for weapons; optional custom gun model support
- Optical-flow based violence detection
- Email (and optional Telegram) alerts with evidence and location
"""

import os
import time
import collections
import cv2
import numpy as np
import smtplib
from email.message import EmailMessage
import requests

# Optional: ultralytics for YOLO weapon detection
from ultralytics import YOLO

import tensorflow_hub as hub

# ---------------- CONFIG ----------------
SENDER_EMAIL = "cctvcamers705@gmail.com"
SENDER_PASS  = "dlpx nefx xgco pylu"
RECEIVER_EMAIL = "sanjaisan2325@gmail.com"

# Telegram optional
TELEGRAM_BOT_TOKEN = None   # "123:ABC..."
TELEGRAM_CHAT_ID = None     # "123456789"

CONF_THRESHOLD = 0.45
ALERT_COOLDOWN = 30           # seconds
SAVE_DIR = "evidence"
os.makedirs(SAVE_DIR, exist_ok=True)

# Violence detection params
FLOW_BUFFER_SIZE = 16
MAG_MEAN_THRESHOLD = 2.0
MAG_VAR_THRESHOLD  = 1.5
VIOLENCE_DETECTION_COUNT = 3

# TF-Hub SSD model (COCO)
TFHUB_MODEL_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

# YOLOv8 model: default (yolov8n) or a custom weapon model
YOLO_MODEL_PATH = "yolov8n.pt"             # uses ultralytics default if local weight not found
CUSTOM_GUN_MODEL_PATH = None               # e.g., "gun_yolo.pt" if you trained one

# COCO class names (80 classes) — note indexing differences handled later
COCO_CLASSES = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
 "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
 "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
 "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
 "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
 "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
 "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
 "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
 "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
 "teddy bear","hair drier","toothbrush"
]

# Target sets
TARGET_ANIMALS = {"elephant","bear","zebra","giraffe","cow","horse","sheep","dog","cat"}
# Knife is in COCO; gun needs custom YOLO to be reliable
TARGET_WEAPONS_COCO = {"knife"}     # detected by COCO/TF-Hub
TARGET_WEAPONS_YOLO = {"knife","gun","pistol","rifle","firearm"}  # YOLO names may vary
TARGET_LABELS_COCO = TARGET_ANIMALS.union(TARGET_WEAPONS_COCO)

# ---------------- helper functions ----------------
def get_location():
    try:
        res = requests.get("https://ipinfo.io", timeout=5).json()
        city = res.get("city", "Unknown")
        region = res.get("region", "Unknown")
        country = res.get("country", "Unknown")
        coords = res.get("loc", "Unknown")
        map_link = f"https://www.google.com/maps?q={coords}"
        return f"{city}, {region}, {country} (coords: {coords})\nMap: {map_link}"
    except Exception as e:
        return f"Location lookup failed: {e}"

def send_email_alert(subject, body, attachment_path=None):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg.set_content(body)
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                file_data = f.read()
                file_name = os.path.basename(attachment_path)
            msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASS)
            smtp.send_message(msg)
        print("📩 Email sent.")
        return True
    except Exception as e:
        print("❌ Email send failed:", e)
        return False

def send_telegram_message(text):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram send failed:", e)
        return False

# ---------------- load models ----------------
print("Loading TF-Hub SSD MobileNet V2 model (COCO)...")
detector_tf = hub.load(TFHUB_MODEL_HANDLE)
print("TF detector loaded.")

# Load YOLOv8 model (ultralytics)
print("Loading YOLOv8 model...")
yolo_model = None
try:
    if CUSTOM_GUN_MODEL_PATH:
        yolo_model = YOLO(CUSTOM_GUN_MODEL_PATH)
        print("Loaded custom YOLO gun model:", CUSTOM_GUN_MODEL_PATH)
    else:
        yolo_model = YOLO(YOLO_MODEL_PATH)  # will download if not present
        print("Loaded YOLO model:", YOLO_MODEL_PATH)
except Exception as e:
    print("⚠ Failed to load YOLO model:", e)
    yolo_model = None

# -------------- TF-Hub detector wrapper ----------------
def run_tfhub_detector(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (320, 320))
    img_input = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    outputs = detector_tf(img_input)
    boxes = outputs["detection_boxes"].numpy()[0]
    classes = outputs["detection_classes"].numpy()[0].astype(np.int32)
    scores = outputs["detection_scores"].numpy()[0]
    h, w = frame.shape[:2]
    boxes_xyxy = []
    for (ymin, xmin, ymax, xmax) in boxes:
        x1 = int(xmin * w); y1 = int(ymin * h)
        x2 = int(xmax * w); y2 = int(ymax * h)
        boxes_xyxy.append((x1,y1,x2,y2))
    return boxes_xyxy, classes, scores

# -------------- Violence detector (optical flow) --------------
class ViolenceDetector:
    def __init__(self, buf_size=FLOW_BUFFER_SIZE):
        self.gray_buf = collections.deque(maxlen=buf_size)
        self.mag_means = collections.deque(maxlen=buf_size-1)
    def add_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.gray_buf.append(gray)
        if len(self.gray_buf) >= 2:
            prev = self.gray_buf[-2]; curr = self.gray_buf[-1]
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5,3,15,3,5,1.2,0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mean_mag = float(np.mean(mag)); var_mag = float(np.var(mag))
            self.mag_means.append((mean_mag, var_mag))
            return mean_mag, var_mag
        return None, None
    def detect_violence(self, mean_threshold=MAG_MEAN_THRESHOLD, var_threshold=MAG_VAR_THRESHOLD, window=4):
        if len(self.mag_means) < window: return False
        recent = list(self.mag_means)[-window:]
        mean_vals = [m for m,v in recent]; var_vals = [v for m,v in recent]
        return (np.mean(mean_vals) > mean_threshold) and (np.mean(var_vals) > var_threshold)

# -------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

violence_detector = ViolenceDetector()
last_alert_time = 0.0
violence_positive_count = 0

print("Starting camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    now = time.time()
    annotated = frame.copy()
    detected_targets = []

    # 1) TF-Hub detections (COCO) — animals & knife (COCO)
    boxes, classes, scores = run_tfhub_detector(frame)
    for (box, cls, score) in zip(boxes, classes, scores):
        if score < 0.30: continue
        idx = int(cls) - 1
        if idx < 0 or idx >= len(COCO_CLASSES): continue
        label = COCO_CLASSES[idx]
        x1,y1,x2,y2 = box
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,200,0), 2)
        cv2.putText(annotated, f"{label}:{score:.2f}", (x1, max(20,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)
        if label in TARGET_LABELS_COCO and score >= CONF_THRESHOLD:
            detected_targets.append((label, float(score), (x1,y1,x2,y2), "COCO"))

    # 2) YOLO detections for weapons (and optionally gun)
    if yolo_model is not None:
        try:
            yolo_results = yolo_model(frame, imgsz=640, conf=0.3, verbose=False)
            # ultralytics returns list of results (one per image)
            for res in yolo_results:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    # get class name safely
                    names = res.names if hasattr(res, "names") else yolo_model.names
                    label = names.get(cls_id, str(cls_id)).lower() if isinstance(names, dict) else str(names[cls_id]).lower()
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0].numpy()
                    x1,y1,x2,y2 = map(int, xyxy[:4])
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(annotated, f"{label}:{conf:.2f}", (x1, max(20,y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    # Check weapon labels from YOLO
                    for wlabel in TARGET_WEAPONS_YOLO:
                        if wlabel in label and conf >= CONF_THRESHOLD:
                            detected_targets.append((label, float(conf), (x1,y1,x2,y2), "YOLO"))
        except Exception as e:
            print("YOLO detection failed:", e)

    # 3) Violence via optical flow
    mean_mag, var_mag = violence_detector.add_frame(frame)
    is_violence_window = violence_detector.detect_violence()
    if is_violence_window:
        violence_positive_count += 1
    else:
        violence_positive_count = max(0, violence_positive_count - 0.5)
    violence_detected = violence_positive_count >= VIOLENCE_DETECTION_COUNT
    if violence_detected:
        cv2.putText(annotated, "VIOLENCE DETECTED", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # 4) Handle alerts
    alert_triggered = False
    alert_info = []
    if detected_targets:
        for label, conf, bbox, src in detected_targets:
            alert_info.append(f"{label} (conf={conf:.2f}) bbox={bbox} src={src}")
        alert_triggered = True
    if violence_detected:
        alert_info.append("Violence/fight behaviour detected by motion analysis")
        alert_triggered = True

    if alert_triggered and (now - last_alert_time) >= ALERT_COOLDOWN:
        last_alert_time = now
        ts = int(now)
        fname = os.path.join(SAVE_DIR, f"evidence_{ts}.jpg")
        cv2.imwrite(fname, annotated)
        location_info = get_location()
        subject = "🚨 ALERT: Suspicious activity detected!"
        body = "Detections:\n" + "\n".join(alert_info) + f"\n\nLocation:\n{location_info}\n\nEvidence: {fname}"
        send_email_alert(subject, body, fname)
        if TELeGRAM_OK := (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
            send_telegram_message(subject + "\n" + "\n".join(alert_info) + f"\n{location_info}")
        print("Alert sent:", alert_info)

    # 5) overlay debugging info
    if mean_mag is not None:
        cv2.putText(annotated, f"Flow mean:{mean_mag:.2f} var:{(var_mag or 0):.2f}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(annotated, f"ViolenceCount:{violence_positive_count:.1f}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Smart Surveillance", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
