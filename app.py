import datetime
import glob
import os
import queue
import threading
import time
import csv
import shutil
import zipfile
import io
import cv2 
import numpy as np
import uuid
from functools import wraps
from PIL import ImageFont, ImageDraw, Image
from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
    send_file,
    session,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "cctv_dashboard_secret_key_change_as_needed"

# ==========================================
# 1. AUTHENTICATION & ROLE-BASED SYSTEM
# ==========================================
USERS_DB = {
    "admin": {"password": "admin1234", "role": "admin"},
    "user": {"password": "user1234", "role": "user"}
}

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            if request.path.startswith('/api/'):
                return jsonify({"status": "error", "message": "กรุณาเข้าสู่ระบบก่อนใช้งาน (Unauthorized)"}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_only(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin':
            return jsonify({"status": "error", "message": "Permission Denied: Admin level required"}), 403
        return f(*args, **kwargs)
    return decorated

# URL สำหรับเชื่อมกล้อง RTSP (หรือใส่ 0 หากใช้ Webcam)
HIKVISION_RTSP_URL = 0
# -------------------------------------------------------------
model = YOLO('yolov8n.pt')

PERSON_CLASS_ID = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

known_face_hists = []
known_face_names = []

is_paused = False

# ==========================================
# THAI TEXT DRAWING ENGINE (PIL)
# ==========================================
def draw_text_pil(img, text, position=(20, 15), font_size=18, color=(255, 255, 0)):
    """ ฟังก์ชันสำหรับเขียนข้อความภาษาไทยลงบนภาพด้วย Pillow """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("tahoma.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("angsana.ttf", font_size + 4)
        except:
            font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# ACTIVE USERS TRACKER
# ==========================================
active_sessions = {}
kicked_sessions = set()
session_lock = threading.Lock()

def clean_expired_sessions():
    while True:
        now = time.time()
        with session_lock:
            expired_keys = [sid for sid, data in active_sessions.items() if now - data["last_seen"] > 120]
            for sid in expired_keys:
                del active_sessions[sid]
        time.sleep(10)

# ==========================================
# DUAL-LINE CONFIGURATION
# ==========================================
line1_coords = {"x1": 0.32, "y1": 0.0, "x2": 0.32, "y2": 1.0}
line2_coords = {"x1": 0.40, "y1": 0.0, "x2": 0.40, "y2": 1.0}

data_lock = threading.Lock()

BASE_PROJECT_DIR = r"D:\ESP32_Project"
REPORT_DIR = os.path.join(BASE_PROJECT_DIR, "สรุปผลย้อนหลังรายสัปดาห์")
SAVE_DIR = os.path.join(BASE_PROJECT_DIR, "รูปภาพปัจจุบัน (วันนี้)")

for folder in [REPORT_DIR, SAVE_DIR]:
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            print(f"[Warning] Cannot create folder: {e}")

def get_current_week_str():
    now = datetime.datetime.now()
    year, week_num, _ = now.isocalendar()
    return f"{year}-W{week_num:02d}"

def get_current_date_str():
    return datetime.datetime.now().strftime("%Y-%m-%d")

# ตัวแปรสถิติรายวัน 24 ชั่วโมง (Reset ทุกเที่ยงคืน)
daily_stats = {
    "date": get_current_date_str(),
    "person_in": 0,
    "person_out": 0
}

weekly_stats = {
    "week": get_current_week_str(),
    "person_in": 0,
    "person_out": 0
}

face_count = 0
person_count = 0
total_in = 0
total_out = 0
visit_logs = []

tracked_positions = {}
armed_ids = set()            
counted_ids = set()          

last_known_names = {}
last_draw_boxes = [] 

frame_count = 0
frame_skip_interval = 3

def load_known_faces():
    """ โหลดใบหน้าต้นแบบจากโฟลเดอร์ known_faces """
    global known_face_hists, known_face_names
    with data_lock:
        known_face_hists = []
        known_face_names = []

        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)

        for file in os.listdir(KNOWN_FACES_DIR):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                raw_name = os.path.splitext(file)[0]
                person_name = raw_name.split('_')[0]
                
                img_path = os.path.join(KNOWN_FACES_DIR, file)
                img = cv2.imread(img_path)

                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        crop = img[y : y + h, x : x + w]
                    else:
                        crop = img

                    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv_crop], [0, 1], None, [50, 60], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                    known_face_hists.append(hist)
                    known_face_names.append(person_name)

load_known_faces()

def analyze_unrecognized_reason(face_bgr):
    """ วิเคราะห์สาเหตุที่ไม่สามารถระบุตัวตนใบหน้าได้ """
    h, w = face_bgr.shape[:2]
    if w < 45 or h < 45:
        return "Unknown (ใบหน้าเล็ก/ไกล)"
        
    gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_face)
    if avg_brightness < 45:
        return "Unknown (แสงมืดเกินไป)"
    elif avg_brightness > 215:
        return "Unknown (ย้อนแสง/แสงจ้า)"

    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)
    if len(eyes) < 2:
        return "Unknown (ก้มหน้า/เอียงหน้า)"

    return "Unknown (ไม่พบข้อมูลในระบบ)"

def recognize_face(face_bgr):
    if len(known_face_hists) == 0: 
        return analyze_unrecognized_reason(face_bgr), 0.0
        
    hsv_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    hist_face = cv2.calcHist([hsv_face], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_face, hist_face, 0, 1, cv2.NORM_MINMAX)

    best_match_name = "Unknown"
    max_score = 0.0
    for template_hist, name in zip(known_face_hists, known_face_names):
        score = cv2.compareHist(hist_face, template_hist, cv2.HISTCMP_CORREL)
        if score > 0.78 and score > max_score:
            max_score = score
            best_match_name = name

    if best_match_name == "Unknown":
        best_match_name = analyze_unrecognized_reason(face_bgr)

    return best_match_name, max_score

def cross_product(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def is_intersecting(p1, p2, p3, p4):
    cp1 = cross_product(p1, p2, p3)
    cp2 = cross_product(p1, p2, p4)
    cp3 = cross_product(p3, p4, p1)
    cp4 = cross_product(p3, p4, p2)

    if ((cp1 > 0 and cp2 < 0) or (cp1 < 0 and cp2 > 0)) and \
       ((cp3 > 0 and cp4 < 0) or (cp3 < 0 and cp4 > 0)):
        return True
    return False

# ==========================================
# CSV REPORT GENERATOR (DAILY & WEEKLY)
# ==========================================
def save_daily_report_to_csv(target_date):
    """ สร้างไฟล์ CSV สรุปสถิติจำนวนการเข้า-ออก ประจำวัน (24 ชม.) แยกตามรายชื่อบุคคล """
    csv_file_path = os.path.join(REPORT_DIR, f"person_summary_daily_{target_date}.csv")
    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as f:
        f.write("sep=,\n")
        writer = csv.writer(f)
        
        writer.writerow([
            "ชื่อบุคคล / สถานะระบุตัวตน", 
            "เข้าวันนี้ (IN)", 
            "ออกวันนี้ (OUT)", 
            "รวมจำนวนเข้า-ออกวันนี้", 
            "วันที่ (Date)"
        ])
        
        person_counts = {}
        with data_lock:
            for log in visit_logs:
                display_name = log.get("display_name", log.get("name", "Unknown"))
                direction = log.get("direction", "")
                
                if display_name not in person_counts:
                    person_counts[display_name] = {"IN": 0, "OUT": 0}
                
                if direction in ["IN", "OUT"]:
                    person_counts[display_name][direction] += 1

        for name, counts in person_counts.items():
            in_count = counts["IN"]
            out_count = counts["OUT"]
            total_today = in_count + out_count
            
            writer.writerow([
                name,
                f"{in_count} ครั้ง",
                f"{out_count} ครั้ง",
                f"{total_today} ครั้ง",
                target_date
            ])

        total_p_in = daily_stats.get("person_in", 0)
        total_p_out = daily_stats.get("person_out", 0)
        writer.writerow([])
        writer.writerow([
            "--- สรุปรวมประจำวัน ---",
            f"{total_p_in} ครั้ง",
            f"{total_p_out} ครั้ง",
            f"{total_p_in + total_p_out} ครั้ง",
            target_date
        ])

def archive_old_captures_to_drive_d():
    try:
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        if os.path.exists(SAVE_DIR):
            for filename in os.listdir(SAVE_DIR):
                if filename.endswith(".jpg") and today_str not in filename:
                    parts = filename.split('_')
                    file_date_str = None
                    for part in parts:
                        if len(part) == 8 and part.isdigit():
                            file_date_str = f"{part[:4]}-{part[4:6]}-{part[6:]}"
                            break
                    if file_date_str:
                        date_folder = os.path.join(REPORT_DIR, f"captures_{file_date_str}")
                        if not os.path.exists(date_folder):
                            os.makedirs(date_folder)
                        shutil.move(os.path.join(SAVE_DIR, filename), os.path.join(date_folder, filename))
    except Exception as e:
        print(f"[Archive Error] {e}")

def check_and_reset_stats():
    """ ตรวจสอบการเปลี่ยนวัน (24 ชม. เที่ยงคืน reset 1 ใหม่) และเปลี่ยนสัปดาห์ """
    global daily_stats, weekly_stats, total_in, total_out, armed_ids, counted_ids, tracked_positions, visit_logs, last_known_names
    current_date = get_current_date_str()
    current_week = get_current_week_str()
    archive_old_captures_to_drive_d()

    with data_lock:
        # 1. Reset รายวัน เมื่อผ่านเที่ยงคืน (24 ชม.)
        if current_date != daily_stats["date"]:
            save_daily_report_to_csv(daily_stats["date"])
            daily_stats = {"date": current_date, "person_in": 0, "person_out": 0}
            total_in = 0
            total_out = 0
            visit_logs.clear()
            armed_ids.clear()
            counted_ids.clear()
            tracked_positions.clear()
            last_known_names.clear()

        # 2. Reset สัปดาห์
        if current_week != weekly_stats["week"]:
            weekly_stats = {"week": current_week, "person_in": 0, "person_out": 0}

class RTSPVideoWriter:
    def __init__(self, src):
        self.src = src
        if isinstance(self.src, int):
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|framedrop|strict;experimental"
            )
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.q = queue.Queue(maxsize=1)
        self.stopped = False
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.1)
                if isinstance(self.src, int):
                    self.cap.open(self.src, cv2.CAP_DSHOW)
                else:
                    self.cap.open(self.src, cv2.CAP_FFMPEG)
                continue
                
            ret, frame = self.cap.read() if isinstance(self.src, int) else self.cap.retrieve() if self.cap.grab() else (False, None)
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

    def read(self):
        try: return True, self.q.get(timeout=0.2)
        except queue.Empty: return False, None

cam_stream = RTSPVideoWriter(HIKVISION_RTSP_URL)

def generate_raw_frames():
    while True:
        success, frame = cam_stream.read()
        if not success or frame is None:
            time.sleep(0.01)
            continue
        display_frame = cv2.resize(frame, (1280, 720))
        ret, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

def generate_frames():
    global face_count, person_count, total_in, total_out, visit_logs, is_paused
    global line1_coords, line2_coords, tracked_positions, armed_ids, counted_ids, daily_stats, weekly_stats
    global frame_count, frame_skip_interval, last_known_names, last_draw_boxes

    target_classes = [PERSON_CLASS_ID]

    while True:
        check_and_reset_stats()

        if is_paused:
            time.sleep(0.1)
            continue

        success, frame = cam_stream.read()
        if not success or frame is None:
            time.sleep(0.001)
            continue

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]
        
        l1_start = (int(line1_coords["x1"] * frame_w), int(line1_coords["y1"] * frame_h))
        l1_end   = (int(line1_coords["x2"] * frame_w), int(line1_coords["y2"] * frame_h))

        l2_start = (int(line2_coords["x1"] * frame_w), int(line2_coords["y1"] * frame_h))
        l2_end   = (int(line2_coords["x2"] * frame_w), int(line2_coords["y2"] * frame_h))

        if frame_count % frame_skip_interval != 0:
            for item in last_draw_boxes:
                x1, y1, x2, y2, color, display_label = item
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                rgb_color = (color[2], color[1], color[0])
                frame = draw_text_pil(frame, display_label, position=(x1, max(5, y1 - 25)), font_size=16, color=rgb_color)

            display_frame = cv2.resize(frame, (960, 540))
            ret, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            continue

        clean_frame = frame.copy()

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=target_classes,
            conf=0.35,
            imgsz=320,
            verbose=False
        )

        current_persons = 0
        current_faces = 0
        active_track_ids = set()
        current_draw_boxes = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                if box.id is None: continue

                track_id = int(box.id[0].item())
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                curr_pos = (cx, cy)
                active_track_ids.add(track_id)

                label_name = "Person"

                if cls_id == PERSON_CLASS_ID:
                    current_persons += 1
                    person_crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                    if person_crop.size > 0:
                        gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray_crop, 
                            scaleFactor=1.1, 
                            minNeighbors=3, 
                            minSize=(25, 25)
                        )
                        if len(faces) > 0:
                            current_faces += 1
                            fx, fy, fw, fh = faces[0]
                            face_crop = person_crop[fy:fy + fh, fx:fx + fw]
                            if face_crop.size > 0:
                                name, score = recognize_face(face_crop)
                                label_name = name
                        else:
                            label_name = "Unknown (ก้มหน้า/เอียงหน้า)"

                with data_lock:
                    if track_id in tracked_positions:
                        prev_pos = tracked_positions[track_id]["pos"]

                        if cls_id == PERSON_CLASS_ID:
                            if track_id not in armed_ids:
                                if is_intersecting(prev_pos, curr_pos, l1_start, l1_end):
                                    armed_ids.add(track_id)

                            if track_id in armed_ids and track_id not in counted_ids:
                                if is_intersecting(prev_pos, curr_pos, l2_start, l2_end):
                                    cp = cross_product(l2_start, l2_end, curr_pos)
                                    if cp > 0:
                                        total_in += 1
                                        daily_stats["person_in"] += 1
                                        weekly_stats["person_in"] += 1
                                        log_visit(label_name, "IN", clean_frame)
                                    else:
                                        total_out += 1
                                        daily_stats["person_out"] += 1
                                        weekly_stats["person_out"] += 1
                                        log_visit(label_name, "OUT", clean_frame)

                                    counted_ids.add(track_id)
                                    save_daily_report_to_csv(daily_stats["date"])

                    tracked_positions[track_id] = {"pos": curr_pos, "label": label_name}

                color = (0, 255, 0)
                if "Unknown" in label_name:
                    color = (0, 165, 255)

                status_tag = ""
                if track_id in counted_ids: status_tag = " [COUNTED]"
                elif track_id in armed_ids: status_tag = " [ARMED]"

                display_label = f"ID:{track_id} {label_name}{status_tag}"
                current_draw_boxes.append((x1, y1, x2, y2, color, display_label))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                rgb_color = (color[2], color[1], color[0])
                frame = draw_text_pil(frame, display_label, position=(x1, max(5, y1 - 25)), font_size=16, color=rgb_color)

        last_draw_boxes = current_draw_boxes

        with data_lock:
            tracked_positions = {tid: val for tid, val in tracked_positions.items() if tid in active_track_ids}
            armed_ids = {tid for tid in armed_ids if tid in active_track_ids}
            counted_ids = {tid for tid in counted_ids if tid in active_track_ids}
            last_known_names = {tid: val for tid, val in last_known_names.items() if tid in active_track_ids}

            person_count = current_persons
            face_count = current_faces

        display_frame = cv2.resize(frame, (960, 540))
        ret, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

def auto_clean_old_captures():
    while True:
        try:
            now = time.time()
            retention_period = 30 * 24 * 60 * 60
            if os.path.exists(REPORT_DIR):
                for item in os.listdir(REPORT_DIR):
                    item_path = os.path.join(REPORT_DIR, item)
                    if os.path.isdir(item_path) and item.startswith("captures_"):
                        if (now - os.path.getmtime(item_path)) > retention_period:
                            shutil.rmtree(item_path)
                    elif os.path.isfile(item_path) and (item.startswith("person_summary_") or item.startswith("weekly_summary_")) and item.endswith(".csv"):
                        if (now - os.path.getmtime(item_path)) > retention_period:
                            os.remove(item_path)
        except Exception as e:
            print(f"[Auto-Clean Error] {e}")
        time.sleep(3600)

def log_visit(name, direction, frame):
    global visit_logs
    now = datetime.datetime.now()
    timestamp_file = now.strftime("%Y%m%d_%H%M%S")
    time_str = now.strftime("%H:%M:%S")
    date_time_display = now.strftime("%Y-%m-%d %H:%M:%S")
    hour_str = now.strftime("%H:00")

    clean_name = name.split(" (")[0]
    display_name = name
    if clean_name in ["Unknown", "Person"]:
        unk_num = sum(1 for item in visit_logs if clean_name in item["display_name"]) + 1
        reason = ""
        if "(" in name:
            reason = f" ({name.split('(')[1].replace(')', '')})"
        display_name = f"{clean_name}_{unk_num}{reason}"

    filename_only = f"{direction}_{timestamp_file}.jpg"

    saved_frame = frame.copy()
    cv2.rectangle(saved_frame, (10, 10), (750, 48), (0, 0, 0), -1)
    stamp_text = f"{date_time_display} | [{direction}] {display_name}"
    saved_frame = draw_text_pil(saved_frame, stamp_text, position=(20, 15), font_size=20, color=(255, 255, 0))

    filename = os.path.join(SAVE_DIR, filename_only)
    cv2.imwrite(filename, saved_frame)

    visit_logs.append({
        "id": len(visit_logs) + 1,
        "name": name,
        "display_name": display_name,
        "direction": direction,
        "time": time_str,
        "hour": hour_str,
        "filename": filename_only,
    })

# ==========================================
# PAGE TEMPLATES (LOGIN & DASHBOARD)
# ==========================================
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - CCTV Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style> body { background: radial-gradient(circle at top left, #0b1329, #030712); font-family: 'Plus Jakarta Sans', sans-serif; } </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4">
    <div class="w-full max-w-md bg-slate-900/90 border border-slate-800/80 p-8 rounded-3xl shadow-2xl text-white">
        <div class="text-center mb-6">
            <div class="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 mb-3">
                <i class="fa-solid fa-shield-halved text-2xl"></i>
            </div>
            <h1 class="text-2xl font-black bg-gradient-to-r from-cyan-400 via-sky-400 to-indigo-400 bg-clip-text text-transparent">CCTV AI COMMAND CENTER</h1>
            <p class="text-xs text-slate-400 mt-1">กรอกชื่อผู้ใช้และรหัสผ่านเพื่อเข้าสู่ระบบ</p>
        </div>
        {% if error %}
        <div class="bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs p-3 rounded-xl mb-4 text-center font-medium">{{ error }}</div>
        {% endif %}
        <form action="/login" method="POST" class="space-y-4">
            <div>
                <label class="block text-xs font-semibold text-slate-300 mb-1.5">Username</label>
                <input type="text" name="username" required placeholder="admin หรือ user" class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none focus:border-cyan-500">
            </div>
            <div>
                <label class="block text-xs font-semibold text-slate-300 mb-1.5">Password</label>
                <input type="password" name="password" required placeholder="กรอกรหัสผ่าน" class="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none focus:border-cyan-500">
            </div>
            <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-extrabold py-3 rounded-xl text-xs transition mt-2">LOGIN</button>
        </form>
    </div>
</body>
</html>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Dual-Line Vision Command Center</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background: radial-gradient(circle at 10% 10%, #0d1527, #030712 90%); color: #f8fafc; font-family: 'Plus Jakarta Sans', sans-serif; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
        .cyber-card { background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4); }
        canvas { touch-action: none; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); } ::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.3); border-radius: 4px; }
    </style>
</head>
<body class="min-h-screen p-4 md:p-6 lg:p-8 flex flex-col items-center">
    <!-- Header System -->
    <header class="w-full max-w-7xl mb-6 flex flex-col sm:flex-row justify-between items-center gap-4 cyber-card p-4 rounded-3xl border border-slate-800 z-30">
        <div class="flex items-center gap-3.5">
            <div class="bg-gradient-to-tr from-cyan-500 via-sky-500 to-indigo-500 p-3 rounded-2xl text-slate-950 font-black text-xl shadow-lg">
                <i class="fa-solid fa-eye"></i>
            </div>
            <div>
                <h1 class="text-xl md:text-2xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-300 bg-clip-text text-transparent">AI VISION COMMAND CENTER</h1>
                <div class="flex items-center gap-2 mt-0.5">
                    <span class="text-[10px] font-bold text-slate-400 uppercase">ROLE:</span>
                    <span class="text-[10px] px-2 py-0.5 rounded-full font-black uppercase border border-amber-500/40 bg-amber-500/10 text-amber-300">{{ session.get('role', 'user') }}</span>
                    <span class="text-xs text-slate-500 font-mono">({{ session.get('username') }})</span>
                </div>
            </div>
        </div>
        <div class="flex items-center gap-2.5 w-full sm:w-auto justify-end flex-wrap">
            {% if session.get('role') == 'admin' %}
            <button onclick="openFaceManagerModal()" class="px-3.5 py-2 rounded-2xl bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 text-amber-300 font-semibold text-xs transition active:scale-95 cursor-pointer">
                <i class="fa-solid fa-id-card-clip mr-1"></i> จัดการใบหน้า
            </button>
            {% endif %}
            
            <button onclick="openActiveUsersModal()" class="px-3.5 py-2 rounded-2xl bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 font-semibold text-xs transition active:scale-95 cursor-pointer">
                <i class="fa-solid fa-users-viewfinder"></i> ผู้ใช้งานขณะนี้: <b id="active-users-count" class="font-mono text-white text-sm">1</b> คน
            </button>
            
            <!-- Realtime Clock Widget -->
            <div class="px-4 py-2 rounded-2xl bg-slate-950/80 border border-slate-800 flex items-center gap-3">
                <span id="status-dot" class="h-2.5 w-2.5 rounded-full bg-emerald-400"></span>
                <span id="clock" class="font-mono text-xs font-bold text-slate-200">--:--:--</span>
            </div>
            
            <a href="/logout" class="bg-rose-500/10 text-rose-400 border border-rose-500/20 p-2.5 rounded-2xl text-xs font-semibold"><i class="fa-solid fa-right-from-bracket"></i></a>
        </div>
    </header>

    <main class="max-w-7xl w-full grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2 flex flex-col gap-6">
            <!-- Video Box & Fixed Z-Index Canvas -->
            <div id="video-container" class="cyber-card rounded-3xl p-2 bg-slate-950 aspect-video relative overflow-hidden flex items-center justify-center border border-slate-800/90 shadow-2xl">
                <img id="video-stream" src="/video_feed" class="w-full h-full object-cover rounded-2xl select-none">
                <canvas id="line-canvas" class="absolute inset-2 w-[calc(100%-1rem)] h-[calc(100%-1rem)] rounded-2xl {% if session.get('role') == 'admin' %}cursor-crosshair{% else %}pointer-events-none{% endif %} z-0"></canvas>
            </div>
            {% if session.get('role') == 'admin' %}
            <div class="cyber-card px-4 py-3 rounded-2xl border border-slate-800/80 text-xs text-slate-400 flex justify-between items-center">
                <span><b>ลากปกติ:</b> ตั้งเส้น 2 (COUNT) | <b>Shift + ลาก:</b> ตั้งเส้น 1 (TRIGGER)</span>
                <button onclick="resetLinesDefault()" class="text-[11px] bg-slate-800 text-slate-200 px-3 py-1.5 rounded-xl border border-slate-700">รีเซ็ตพิกัดเส้น</button>
            </div>
            {% endif %}
        </div>

        <div class="flex flex-col gap-4">
            <!-- คลิกเพื่อดูรายชื่อพนักงานที่ผ่านเข้า-ออก ประจำวัน 24 ชม. -->
            <div class="grid grid-cols-2 gap-3.5">
                <div onclick="openPersonListModal('IN')" class="cyber-card p-4 rounded-3xl border-emerald-500/30 bg-emerald-500/5 cursor-pointer hover:bg-emerald-500/10 transition group">
                    <div class="flex justify-between items-center">
                        <span class="text-[10px] font-black text-emerald-400 uppercase">PERSON IN (24h)</span>
                        <i class="fa-solid fa-circle-info text-emerald-400 text-xs opacity-70 group-hover:opacity-100"></i>
                    </div>
                    <div class="text-4xl font-mono font-black text-emerald-400 mt-2" id="total-in">0</div>
                    <span class="text-[9px] text-slate-400 mt-1 block">คลิกเพื่อดูรายชื่อทั้งหมด</span>
                </div>

                <div onclick="openPersonListModal('OUT')" class="cyber-card p-4 rounded-3xl border-rose-500/30 bg-rose-500/5 cursor-pointer hover:bg-rose-500/10 transition group">
                    <div class="flex justify-between items-center">
                        <span class="text-[10px] font-black text-rose-400 uppercase">PERSON OUT (24h)</span>
                        <i class="fa-solid fa-circle-info text-rose-400 text-xs opacity-70 group-hover:opacity-100"></i>
                    </div>
                    <div class="text-4xl font-mono font-black text-rose-400 mt-2" id="total-out">0</div>
                    <span class="text-[9px] text-slate-400 mt-1 block">คลิกเพื่อดูรายชื่อทั้งหมด</span>
                </div>
            </div>

            <div class="cyber-card p-4 rounded-3xl border-slate-800">
                <span class="text-[10px] font-black text-cyan-400 uppercase"><i class="fa-solid fa-users"></i> บุคคลในจอขณะนี้</span>
                <div class="text-3xl font-mono font-bold text-white mt-1.5" id="person-count">0</div>
            </div>

            <div class="cyber-card p-3 rounded-3xl border-slate-800 flex items-center gap-2">
                {% if session.get('role') == 'admin' %}
                <button id="reset-cam-btn" onclick="resetCamera()" class="flex-1 bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 py-3 rounded-2xl text-xs font-bold border border-amber-500/30 transition active:scale-95">
                    <i class="fa-solid fa-arrows-rotate mr-1"></i> รีเซ็ตกล้อง
                </button>
                {% endif %}
                <button onclick="takeSnapshot()" class="flex-1 bg-emerald-500/10 text-emerald-300 py-3 rounded-2xl text-xs font-bold border border-emerald-500/30">จับภาพ</button>
                <button onclick="toggleFullscreen()" class="flex-1 bg-slate-800 text-slate-200 py-3 rounded-2xl text-xs font-bold border border-slate-700">เต็มจอ</button>
            </div>

            <div class="cyber-card p-4 rounded-3xl border-slate-800 flex flex-col gap-3">
                <div class="flex justify-between items-center border-b border-slate-800 pb-2.5">
                    <h3 class="text-xs font-extrabold text-slate-200 uppercase"><i class="fa-solid fa-list-check text-cyan-400"></i> Real-time Activity Log</h3>
                </div>
                <div class="h-44 overflow-y-auto pr-1">
                    <table class="w-full text-left text-[11px] font-mono">
                        <thead class="text-slate-500 text-[9px] uppercase border-b border-slate-800/50 sticky top-0 bg-slate-900/95">
                            <tr><th class="py-1">เวลา</th><th class="py-1">ทิศทาง</th><th class="py-1">รายละเอียด / สาเหตุที่ไม่พบชื่อ</th></tr>
                        </thead>
                        <tbody id="log-table-body" class="divide-y divide-slate-800/40 text-slate-300">
                            <tr><td colspan="3" class="text-center py-6 text-slate-600">ยังไม่มีรายการบันทึก...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="cyber-card p-4 rounded-3xl border-slate-800 flex flex-col gap-3">
                <a href="/summary" class="w-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 font-bold py-3 px-4 rounded-2xl text-xs flex justify-between items-center">
                    <span><i class="fa-solid fa-chart-pie"></i> สรุปผลย้อนหลังรายสัปดาห์</span><i class="fa-solid fa-chevron-right"></i>
                </a>
                <a href="/gallery" class="w-full bg-slate-950/60 text-slate-200 font-bold py-3 px-4 rounded-2xl text-xs flex justify-between items-center border border-slate-800/80">
                    <span><i class="fa-solid fa-images text-emerald-400"></i> คลังรูปภาพวันนี้</span>
                    <span class="bg-emerald-500/20 text-emerald-300 text-[10px] font-mono px-2.5 py-0.5 rounded-full" id="all-captures-count">0</span>
                </a>
            </div>
        </div>
    </main>

    <!-- MODAL: PERSON DETAIL LIST (24 HOUR LOG) -->
    <div id="person-list-modal" class="fixed inset-0 bg-black/80 backdrop-blur-md z-[80] hidden flex items-center justify-center p-4" onclick="closePersonListModal()">
        <div class="cyber-card max-w-xl w-full rounded-3xl p-6 border border-slate-800 shadow-2xl relative" onclick="event.stopPropagation()">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3 mb-4">
                <div>
                    <h3 id="modal-person-title" class="text-base font-extrabold text-white">รายชื่อบุคคลที่เดินทางผ่าน (24 ชั่วโมงวันนี้)</h3>
                    <p class="text-xs text-slate-400 mt-0.5">รีเซ็ตนับ 1 ใหม่ทุกเที่ยงคืน (00:00 น.)</p>
                </div>
                <button onclick="closePersonListModal()" class="text-slate-400 hover:text-white text-lg"><i class="fa-solid fa-xmark"></i></button>
            </div>
            
            <div class="overflow-y-auto max-h-72 pr-1">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="text-slate-400 uppercase text-[10px] bg-slate-950/80 border-b border-slate-800 sticky top-0">
                        <tr>
                            <th class="py-2.5 px-3">ชื่อบุคคล / เหตุผลที่ไม่ระบุตัวตน</th>
                            <th class="py-2.5 px-3 text-center">ทิศทาง</th>
                            <th class="py-2.5 px-3 text-center">เวลาสแกน</th>
                            <th class="py-2.5 px-3 text-center">รูปอ้างอิง</th>
                        </tr>
                    </thead>
                    <tbody id="person-list-table-body" class="divide-y divide-slate-800/50 text-slate-200">
                        <tr><td colspan="4" class="text-center py-6 text-slate-500">กำลังโหลดข้อมูล...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- MODAL: FACE MANAGER -->
    <div id="face-manager-modal" class="fixed inset-0 bg-black/80 backdrop-blur-md z-[80] hidden flex items-center justify-center p-4" onclick="closeFaceManagerModal()">
        <div class="cyber-card max-w-lg w-full rounded-3xl p-6 border border-slate-800 shadow-2xl relative" onclick="event.stopPropagation()">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3 mb-3">
                <div class="flex items-center gap-2">
                    <i class="fa-solid fa-id-card-clip text-amber-400 text-lg"></i>
                    <h3 class="text-base font-extrabold text-white">ระบบจัดการใบหน้า (Known Faces)</h3>
                </div>
                <button onclick="closeFaceManagerModal()" class="text-slate-400 hover:text-white text-lg"><i class="fa-solid fa-xmark"></i></button>
            </div>

            <div class="bg-slate-950/70 p-4 rounded-2xl border border-slate-800 mb-4 space-y-3">
                <h4 class="text-xs font-bold text-amber-300 uppercase"><i class="fa-solid fa-camera mr-1"></i> ลงทะเบียนใบหน้า (ถ่าย 5 รูปอัตโนมัติ)</h4>
                
                <div class="relative aspect-video rounded-xl overflow-hidden bg-black border border-slate-800 flex items-center justify-center">
                    <img id="face-preview-stream" src="" class="w-full h-full object-cover rounded-xl">
                    <div id="snap-flash-overlay" class="absolute inset-0 bg-white opacity-0 pointer-events-none transition-opacity duration-150 z-20"></div>

                    <div id="countdown-overlay" class="absolute inset-0 bg-black/60 backdrop-blur-xs hidden flex flex-col items-center justify-center text-center p-4 z-10">
                        <span id="guide-instruction" class="text-amber-300 font-extrabold text-base mb-2 uppercase tracking-wide">🔴 กำลังถ่ายรูปอัตโนมัติ...</span>
                        <span id="countdown-num" class="text-6xl font-black text-rose-500 animate-pulse">2</span>
                    </div>
                </div>

                <div>
                    <input type="text" id="person-name" required placeholder="กรอกชื่อภาษาอังกฤษ (เช่น Marisa)" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-amber-400">
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <button type="button" id="snap-btn" onclick="startPhotoEnrollmentProcess()" class="bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-400 hover:to-orange-500 text-slate-950 font-black py-2.5 rounded-xl text-xs transition flex items-center justify-center gap-1.5 shadow-lg active:scale-95 cursor-pointer">
                        <i class="fa-solid fa-camera text-sm"></i> 📸 ถ่ายรูปภาพ (5 รูปอัตโนมัติ)
                    </button>
                    <label class="bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold py-2.5 rounded-xl text-xs transition flex items-center justify-center gap-1.5 cursor-pointer border border-slate-700 active:scale-95">
                        <i class="fa-solid fa-upload"></i> อัปโหลดไฟล์รูป
                        <input type="file" id="person-img" accept="image/*" class="hidden" onchange="uploadFaceFile()">
                    </label>
                </div>
            </div>

            <h4 class="text-xs font-bold text-slate-400 uppercase mb-2"><i class="fa-solid fa-users mr-1"></i> รายชื่อใบหน้าในระบบ</h4>
            <div class="overflow-y-auto max-h-40 pr-1">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="text-slate-500 text-[10px] uppercase border-b border-slate-800 bg-slate-950">
                        <tr><th class="py-2 px-3">รูปถ่าย</th><th class="py-2 px-3">ชื่อ</th><th class="py-2 px-3 text-center">จัดการ</th></tr>
                    </thead>
                    <tbody id="known-faces-table-body" class="divide-y divide-slate-800/50 text-slate-200">
                        <tr><td colspan="3" class="text-center py-6 text-slate-500">กำลังโหลด...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- MODAL: IMAGE VIEWER -->
    <div id="image-viewer-modal" class="fixed inset-0 bg-black/90 backdrop-blur-md z-[90] hidden flex items-center justify-center p-4" onclick="closeImageViewerModal()">
        <div class="cyber-card max-w-lg w-full rounded-3xl p-5 border border-slate-800 shadow-2xl relative flex flex-col items-center" onclick="event.stopPropagation()">
            <div class="w-full flex justify-between items-center border-b border-slate-800 pb-3 mb-4">
                <h3 id="image-viewer-title" class="text-sm font-extrabold text-amber-400 uppercase">แสดงรูปภาพใบหน้า</h3>
                <button onclick="closeImageViewerModal()" class="text-slate-400 hover:text-white text-lg"><i class="fa-solid fa-xmark"></i></button>
            </div>
            <div class="w-full flex items-center justify-center max-h-[70vh] overflow-hidden rounded-2xl bg-black/50 border border-slate-800 p-2">
                <img id="image-viewer-img" src="" class="w-48 h-48 object-cover rounded-2xl border-2 border-amber-400 shadow-xl">
            </div>
            <button onclick="closeImageViewerModal()" class="mt-4 px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-bold border border-slate-700 transition">ปิดหน้าต่าง</button>
        </div>
    </div>

    <!-- MODAL: ACTIVE USERS -->
    <div id="active-users-modal" class="fixed inset-0 bg-black/80 backdrop-blur-md z-[80] hidden flex items-center justify-center p-4" onclick="closeActiveUsersModal()">
        <div class="cyber-card max-w-lg w-full rounded-3xl p-6 border border-slate-800 shadow-2xl relative" onclick="event.stopPropagation()">
            <div class="flex justify-between items-center border-b border-slate-800 pb-4 mb-4">
                <h3 class="text-base font-extrabold text-white">ผู้ใช้งานที่กำลังออนไลน์ (Active Users)</h3>
                <button onclick="closeActiveUsersModal()" class="text-slate-400 hover:text-white text-lg"><i class="fa-solid fa-xmark"></i></button>
            </div>
            <div class="overflow-x-auto max-h-60 overflow-y-auto">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="text-slate-400 uppercase text-[10px] bg-slate-950/80 border-b border-slate-800">
                        <tr><th class="py-2 px-3">ผู้ใช้งาน</th><th class="py-2 px-3">สิทธิ์</th><th class="py-2 px-3">IP Address</th><th class="py-2 px-3 text-right">เวลา</th>{% if session.get('role') == 'admin' %}<th class="py-2 px-3 text-center">จัดการ</th>{% endif %}</tr>
                    </thead>
                    <tbody id="active-users-table-body" class="divide-y divide-slate-800/50 text-slate-200">
                        <tr><td colspan="5" class="text-center py-6 text-slate-500">กำลังโหลด...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('line-canvas');
        const ctx = canvas.getContext('2d');
        const userRole = "{{ session.get('role', 'user') }}";
        let isDrawing = false;
        let activeLineTarget = 1;

        let line1 = { x1: 0.32, y1: 0.0, x2: 0.32, y2: 1.0 };
        let line2 = { x1: 0.40, y1: 0.0, x2: 0.40, y2: 1.0 };

        function resizeCanvas() { canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight; drawOverlayLines(); }
        window.addEventListener('resize', resizeCanvas);

        function drawOverlayLines() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // ถ้าไม่ใช่ Admin ให้ข้ามการวาดเส้นซ้อนบนหน้าจอ
            if (userRole !== 'admin') return;

            const l1_px1 = line1.x1 * canvas.width, l1_py1 = line1.y1 * canvas.height, l1_px2 = line1.x2 * canvas.width, l1_py2 = line1.y2 * canvas.height;
            ctx.beginPath(); ctx.moveTo(l1_px1, l1_py1); ctx.lineTo(l1_px2, l1_py2);
            ctx.strokeStyle = '#f97316'; ctx.lineWidth = 3; ctx.stroke();

            const l2_px1 = line2.x1 * canvas.width, l2_py1 = line2.y1 * canvas.height, l2_px2 = line2.x2 * canvas.width, l2_py2 = line2.y2 * canvas.height;
            ctx.beginPath(); ctx.moveTo(l2_px1, l2_py1); ctx.lineTo(l2_px2, l2_py2);
            ctx.strokeStyle = '#facc15'; ctx.lineWidth = 2; ctx.stroke();
        }

        function getMousePos(e) {
            const rect = canvas.getBoundingClientRect();
            return {
                x: (e.clientX - rect.left) / rect.width,
                y: (e.clientY - rect.top) / rect.height
            };
        }

        if (userRole === 'admin') {
            canvas.addEventListener('mousedown', (e) => {
                isDrawing = true;
                activeLineTarget = e.shiftKey ? 1 : 2;
                const pos = getMousePos(e);

                if (activeLineTarget === 1) {
                    line1.x1 = pos.x; line1.y1 = pos.y; line1.x2 = pos.x; line1.y2 = pos.y;
                } else {
                    line2.x1 = pos.x; line2.y1 = pos.y; line2.x2 = pos.x; line2.y2 = pos.y;
                }
                drawOverlayLines();
            });

            canvas.addEventListener('mousemove', (e) => {
                if (!isDrawing) return;
                const pos = getMousePos(e);
                if (activeLineTarget === 1) {
                    line1.x2 = pos.x; line1.y2 = pos.y;
                } else {
                    line2.x2 = pos.x; line2.y2 = pos.y;
                }
                drawOverlayLines();
            });

            canvas.addEventListener('mouseup', () => {
                if (isDrawing) {
                    isDrawing = false;
                    fetch('/set_lines', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ line1: line1, line2: line2 })
                    });
                }
            });
        }

        function resetLinesDefault() {
            if (userRole !== 'admin') return;
            line1 = { x1: 0.32, y1: 0.0, x2: 0.32, y2: 1.0 };
            line2 = { x1: 0.40, y1: 0.0, x2: 0.40, y2: 1.0 };
            drawOverlayLines();
            fetch('/set_lines', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ line1: line1, line2: line2 })
            });
        }

        function openPersonListModal(filterDir) {
            const modal = document.getElementById('person-list-modal');
            const title = document.getElementById('modal-person-title');
            title.innerText = filterDir === 'IN' ? 'รายชื่อบุคคลที่เดินเข้า (PERSON IN - 24 ชม.)' : 'รายชื่อบุคคลที่เดินออก (PERSON OUT - 24 ชม.)';
            modal.classList.remove('hidden');

            fetch('/get_visit_logs').then(res => res.json()).then(logs => {
                const tbody = document.getElementById('person-list-table-body');
                const filtered = logs.filter(l => l.direction === filterDir);
                if (!filtered || filtered.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" class="text-center py-6 text-slate-500">ไม่มีรายการสแกนในทิศทางนี้</td></tr>';
                    return;
                }
                tbody.innerHTML = filtered.map(item => `
                    <tr class="hover:bg-slate-800/40 transition">
                        <td class="py-2.5 px-3 font-bold text-white">${item.display_name}</td>
                        <td class="py-2.5 px-3 text-center"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${item.direction === 'IN' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}">${item.direction}</span></td>
                        <td class="py-2.5 px-3 text-center text-slate-400">${item.time}</td>
                        <td class="py-2.5 px-3 text-center"><a href="/captures/${item.filename}" target="_blank" class="text-cyan-400 hover:underline"><i class="fa-solid fa-image mr-1"></i>ดูรูปถ่าย</a></td>
                    </tr>
                `).join('');
            });
        }

        function closePersonListModal() {
            document.getElementById('person-list-modal').classList.add('hidden');
        }

        function openFaceManagerModal() {
            document.getElementById('face-manager-modal').classList.remove('hidden');
            document.getElementById('face-preview-stream').src = "/raw_video_feed";
            fetchKnownFacesList();
        }

        function closeFaceManagerModal() {
            document.getElementById('face-manager-modal').classList.add('hidden');
            document.getElementById('face-preview-stream').src = "";
        }

        function viewFaceImage(imgUrl, name) {
            document.getElementById('image-viewer-img').src = imgUrl;
            document.getElementById('image-viewer-title').innerText = `รูปต้นแบบใบหน้า: ${name}`;
            document.getElementById('image-viewer-modal').classList.remove('hidden');
        }

        function closeImageViewerModal() {
            document.getElementById('image-viewer-modal').classList.add('hidden');
            document.getElementById('image-viewer-img').src = "";
        }

        function fetchKnownFacesList() {
            fetch('/api/known_faces')
                .then(res => res.json())
                .then(faces => {
                    const tbody = document.getElementById('known-faces-table-body');
                    if (!faces || faces.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="3" class="text-center py-6 text-slate-500">ยังไม่มีข้อมูลใบหน้าในระบบ</td></tr>';
                        return;
                    }
                    tbody.innerHTML = faces.map(f => {
                        const imgUrl = `/known_faces/${f.filename}?t=${new Date().getTime()}`;
                        return `
                        <tr class="hover:bg-slate-800/40 transition">
                            <td class="py-2 px-3">
                                <img src="${imgUrl}" 
                                     onclick="viewFaceImage('${imgUrl}', '${f.name}')" 
                                     title="คลิกเพื่อดูรูปขนาดใหญ่"
                                     class="w-10 h-10 object-cover rounded-xl border border-slate-700 hover:border-amber-400 cursor-pointer transition transform hover:scale-105 shadow-md">
                            </td>
                            <td class="py-2 px-3 font-bold text-white">${f.name}</td>
                            <td class="py-2 px-3 text-center flex items-center justify-center gap-1.5">
                                <button onclick="editKnownFace('${f.filename}', '${f.name}')" class="bg-amber-500/20 hover:bg-amber-500/40 text-amber-300 border border-amber-500/30 text-[10px] px-2 py-1 rounded-lg font-bold transition">
                                    <i class="fa-solid fa-pen-to-square mr-1"></i>แก้ไข
                                </button>
                                <button onclick="deleteKnownFace('${f.filename}')" class="bg-rose-500/20 hover:bg-rose-500/40 text-rose-300 border border-rose-500/30 text-[10px] px-2 py-1 rounded-lg font-bold transition">
                                    <i class="fa-solid fa-trash-can mr-1"></i>ลบ
                                </button>
                            </td>
                        </tr>
                    `}).join('');
                });
        }

        function editKnownFace(oldFilename, oldName) {
            const newName = prompt(`แก้ไขชื่อใบหน้าสำหรับ "${oldName}":`, oldName);
            if (!newName || newName.trim() === '' || newName.trim() === oldName) return;

            fetch('/api/rename_face', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ old_filename: oldFilename, new_name: newName.trim() })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    fetchKnownFacesList();
                    alert(data.message);
                } else {
                    alert(data.message || 'เกิดข้อผิดพลาดในการแก้ไขชื่อ');
                }
            });
        }

        function playBeepSound() {
            try {
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = 'sine';
                osc.frequency.value = 880; 
                gain.gain.value = 0.3;
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.15); 
            } catch(e) {}
        }

        async function startPhotoEnrollmentProcess() {
            const baseName = document.getElementById('person-name').value.trim();
            if (!baseName) {
                alert('กรุณากรอกชื่อในช่องด้านบนก่อนเริ่มถ่ายรูป!');
                return;
            }

            const overlay = document.getElementById('countdown-overlay');
            const guideEl = document.getElementById('guide-instruction');
            const numEl = document.getElementById('countdown-num');
            const snapBtn = document.getElementById('snap-btn');
            const flashEl = document.getElementById('snap-flash-overlay');

            const poses = [
                { label: "📸 รูปที่ 1/5: มองหน้าตรง 😐" },
                { label: "📸 รูปที่ 2/5: ค่อยๆ หันหน้าไปทางซ้าย 👈" },
                { label: "📸 รูปที่ 3/5: ค่อยๆ หันหน้าไปทางขวา 👉" },
                { label: "📸 รูปที่ 4/5: เงยหน้าขึ้นเล็กน้อย 👆" },
                { label: "📸 รูปที่ 5/5: ก้มหน้าลงเล็กน้อย 👇" }
            ];

            snapBtn.disabled = true;
            overlay.classList.remove('hidden');

            for (let i = 0; i < poses.length; i++) {
                guideEl.innerText = poses[i].label;

                for (let sec = 2; sec > 0; sec--) {
                    numEl.innerText = sec;
                    await new Promise(r => setTimeout(r, 1000));
                }

                numEl.innerText = "⚡ แชะ!";
                
                flashEl.classList.remove('opacity-0');
                flashEl.classList.add('opacity-80');
                playBeepSound();

                setTimeout(() => {
                    flashEl.classList.remove('opacity-80');
                    flashEl.classList.add('opacity-0');
                }, 150);

                try {
                    await fetch('/api/enroll_single_photo', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name: baseName, index: i + 1 })
                    });
                } catch (e) {
                    console.error("Capture Error:", e);
                }

                await new Promise(r => setTimeout(r, 500));
            }

            guideEl.innerText = "⏳ กำลังบันทึกข้อมูลใบหน้า...";
            numEl.innerText = "";
            await new Promise(r => setTimeout(r, 800));

            overlay.classList.add('hidden');
            snapBtn.disabled = false;
            document.getElementById('person-name').value = '';
            fetchKnownFacesList();
            alert(`✅ บันทึกรูปถ่ายใบหน้าของ '${baseName}' ครบทั้ง 5 รูปเรียบร้อยแล้ว!`);
        }

        function uploadFaceFile() {
            const nameInput = document.getElementById('person-name').value.trim();
            const fileInput = document.getElementById('person-img').files[0];

            if (!nameInput) {
                alert('กรุณากรอกชื่อภาษาอังกฤษในช่องด้านบนก่อนเลือกไฟล์รูป');
                document.getElementById('person-img').value = '';
                return;
            }
            if (!fileInput) return;

            const formData = new FormData();
            formData.append('name', nameInput);
            formData.append('file', fileInput);

            fetch('/api/add_face', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    document.getElementById('person-name').value = '';
                    document.getElementById('person-img').value = '';
                    fetchKnownFacesList();
                    alert(data.message);
                } else {
                    alert(data.message || 'เกิดข้อผิดพลาด');
                }
            });
        }

        function deleteKnownFace(filename) {
            if (!confirm(`คุณแน่ใจหรือไม่ว่าต้องการลบใบหน้านี้ออกจากระบบ?`)) return;
            fetch('/api/delete_face', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: filename })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    fetchKnownFacesList();
                } else {
                    alert(data.message || 'เกิดข้อผิดพลาด');
                }
            });
        }

        function sendHeartbeat() {
            fetch('/api/heartbeat', { method: 'POST' })
                .then(res => {
                    if (res.status === 401) { alert('คุณถูกลบออกจากระบบโดย Admin'); window.location.href = '/login'; return; }
                    return res.json();
                })
                .then(data => { if (data && data.active_users !== undefined) document.getElementById('active-users-count').innerText = data.active_users; });
        }
        sendHeartbeat();
        setInterval(sendHeartbeat, 5000);

        function openActiveUsersModal() { document.getElementById('active-users-modal').classList.remove('hidden'); fetchActiveUsersList(); }
        function closeActiveUsersModal() { document.getElementById('active-users-modal').classList.add('hidden'); }

        function fetchActiveUsersList() {
            fetch('/api/active_users_list').then(res => res.json()).then(users => {
                const tbody = document.getElementById('active-users-table-body');
                if (!users || users.length === 0) { tbody.innerHTML = '<tr><td colspan="5" class="text-center py-6 text-slate-500">ไม่มีผู้ใช้ออนไลน์</td></tr>'; return; }
                tbody.innerHTML = users.map(u => `
                    <tr class="hover:bg-slate-800/40">
                        <td class="py-2.5 px-3 font-bold text-white">${u.username}</td>
                        <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded text-[9px] font-extrabold bg-sky-500/20 text-sky-300 border border-sky-500/30">${u.role}</span></td>
                        <td class="py-2.5 px-3 text-slate-400 font-mono">${u.ip}</td>
                        <td class="py-2.5 px-3 text-right text-slate-400 font-mono">${u.last_seen_str}</td>
                        ${userRole === 'admin' ? `<td class="py-2.5 px-3 text-center">${u.role !== 'admin' ? `<button onclick="kickUser('${u.session_id}')" class="bg-rose-500/20 text-rose-300 px-2 py-1 rounded text-[10px]">เตะออก</button>` : '-'}</td>` : ''}
                    </tr>
                `).join('');
            });
        }

        function kickUser(sid) {
            if (!confirm('ต้องการลบผู้ใช้งานนี้ออกจากระบบ?')) return;
            fetch('/api/kick_user', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sid }) })
            .then(res => res.json()).then(() => fetchActiveUsersList());
        }

        function fetchLogsTable() {
            fetch('/get_visit_logs').then(res => res.json()).then(logs => {
                const tbody = document.getElementById('log-table-body');
                if (!logs || logs.length === 0) { tbody.innerHTML = '<tr><td colspan="3" class="text-center py-6 text-slate-600">ยังไม่มีรายการบันทึก...</td></tr>'; return; }
                const recentLogs = logs.slice(-10).reverse();
                tbody.innerHTML = recentLogs.map(log => `
                    <tr class="hover:bg-slate-800/40 transition">
                        <td class="py-1.5 text-slate-400">${log.time}</td>
                        <td class="py-1.5"><span class="text-emerald-400 font-bold bg-emerald-500/10 border border-emerald-500/30 px-1.5 py-0.5 rounded">${log.direction}</span></td>
                        <td class="py-1.5 font-semibold text-slate-200 truncate max-w-[160px]">${log.display_name}</td>
                    </tr>
                `).join('');
            });
        }

        function fetchMetrics() {
            fetch('/get_count').then(res => res.json()).then(data => {
                document.getElementById('person-count').innerText = data.person_count;
                document.getElementById('total-in').innerText = data.daily_stats.person_in;
                document.getElementById('total-out').innerText = data.daily_stats.person_out;
                document.getElementById('all-captures-count').innerText = data.all_captures_count;

                if (!isDrawing && data.line1_coords && data.line2_coords) {
                    line1 = data.line1_coords;
                    line2 = data.line2_coords;
                    drawOverlayLines();
                }
            });
        }

        function updateClock() { 
            const clockEl = document.getElementById('clock');
            if (clockEl) {
                clockEl.innerText = new Date().toLocaleTimeString('th-TH'); 
            }
        }

        updateClock();
        setInterval(updateClock, 1000);
        setInterval(fetchMetrics, 2000);
        setInterval(fetchLogsTable, 4000);

        setTimeout(() => { resizeCanvas(); fetchLogsTable(); }, 500);

        function resetCamera() {
            const btn = document.getElementById('reset-cam-btn');
            if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1"></i> กำลังรีเซ็ต...';

            fetch('/reset_camera', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'success') {
                        const img = document.getElementById('video-stream');
                        if (img) img.src = "/video_feed?t=" + new Date().getTime();
                    } else {
                        alert("เกิดข้อผิดพลาดในการรีเซ็ต: " + data.message);
                    }
                })
                .finally(() => {
                    if (btn) btn.innerHTML = '<i class="fa-solid fa-arrows-rotate mr-1"></i> รีเซ็ตกล้อง';
                });
        }

        function takeSnapshot() { fetch('/manual_snapshot', { method: 'POST' }); }
        function toggleFullscreen() { const container = document.getElementById('video-container'); if (!document.fullscreenElement) container.requestFullscreen(); else document.exitFullscreen(); }
    </script>
</body>
</html>
"""

SUMMARY_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>รายงานสรุปผลย้อนหลังรายวัน/สัปดาห์</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style> body { background: #030712; color: #f8fafc; font-family: 'Plus Jakarta Sans', sans-serif; } </style>
</head>
<body class="p-6 min-h-screen">
    <div class="max-w-5xl mx-auto space-y-6">
        <div class="flex justify-between items-center border-b border-slate-800 pb-4">
            <div>
                <h1 class="text-xl font-bold text-cyan-400"><i class="fa-solid fa-chart-pie mr-2"></i>รายงานสรุปผลย้อนหลัง</h1>
                <p class="text-xs text-slate-400 mt-1">รายการไฟล์ CSV บันทึกสถิติประจำวัน (24 ชม.) และสัปดาห์</p>
            </div>
            <a href="/" class="bg-slate-800 text-slate-200 px-4 py-2 rounded-xl text-xs font-bold border border-slate-700 hover:bg-slate-700 transition">
                <i class="fa-solid fa-arrow-left mr-1.5"></i>กลับหน้าหลัก
            </a>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden p-4">
            <table class="w-full text-left text-xs font-mono">
                <thead class="text-slate-400 uppercase text-[10px] border-b border-slate-800 bg-slate-950">
                    <tr>
                        <th class="py-3 px-4">ชื่อไฟล์รายงาน CSV</th>
                        <th class="py-3 px-4 text-center">ดาวน์โหลด</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800/60 text-slate-200">
                    {% for report in reports %}
                    <tr class="hover:bg-slate-800/40 transition">
                        <td class="py-3 px-4 font-bold text-white"><i class="fa-solid fa-file-csv text-emerald-400 mr-2"></i>{{ report }}</td>
                        <td class="py-3 px-4 text-center">
                            <a href="/download_report/{{ report }}" class="bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 px-3 py-1.5 rounded-xl font-bold transition">
                                <i class="fa-solid fa-download mr-1"></i>ดาวน์โหลด
                            </a>
                        </td>
                    </tr>
                    {% else %}
                    <tr><td colspan="2" class="text-center py-8 text-slate-500">ยังไม่มีไฟล์รายงานสรุปผลในระบบ</td></tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

GALLERY_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>คลังรูปภาพบันทึกประจำวัน</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style> body { background: #030712; color: #f8fafc; font-family: 'Plus Jakarta Sans', sans-serif; } </style>
</head>
<body class="p-6 min-h-screen">
    <div class="max-w-7xl mx-auto space-y-6">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-800 pb-4 gap-4">
            <div>
                <h1 class="text-xl font-bold text-cyan-400"><i class="fa-solid fa-images mr-2"></i>คลังรูปภาพบันทึกวันนี้</h1>
                <p class="text-xs text-slate-400 mt-1">ตัวกรองปัจจุบัน: <b class="text-white">{{ active_filter }}</b></p>
            </div>
            <div class="flex items-center gap-2 flex-wrap">
                <button onclick="downloadSelectedImages()" class="bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/30 px-3.5 py-2 rounded-xl text-xs font-bold transition cursor-pointer">
                    <i class="fa-solid fa-file-arrow-down mr-1.5"></i>ดาวน์โหลดรูปที่เลือก (<span id="download-count">0</span>)
                </button>
                {% if session.get('role') == 'admin' %}
                <button onclick="toggleSelectAll()" class="bg-slate-800 hover:bg-slate-700 text-slate-200 px-3.5 py-2 rounded-xl text-xs font-bold border border-slate-700 transition cursor-pointer">
                    <i class="fa-solid fa-check-double mr-1.5"></i>เลือกทั้งหมด
                </button>
                <button onclick="deleteSelectedImages()" class="bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/30 px-3.5 py-2 rounded-xl text-xs font-bold transition cursor-pointer">
                    <i class="fa-solid fa-trash-can mr-1.5"></i>ลบรูปที่เลือก (<span id="selected-count">0</span>)
                </button>
                <button onclick="deleteAllImages()" class="bg-rose-600 hover:bg-rose-700 text-white px-3.5 py-2 rounded-xl text-xs font-bold transition shadow-lg cursor-pointer">
                    <i class="fa-solid fa-dumpster-fire mr-1.5"></i>ลบทั้งหมด
                </button>
                {% endif %}
                <a href="/" class="bg-slate-800 text-slate-200 px-4 py-2 rounded-xl text-xs font-bold border border-slate-700 hover:bg-slate-700 transition">
                    <i class="fa-solid fa-arrow-left mr-1.5"></i>กลับหน้าหลัก
                </a>
            </div>
        </div>

        <div class="flex gap-2">
            <a href="/gallery" class="px-3.5 py-1.5 rounded-xl text-xs font-bold {% if active_filter == 'ALL' %}bg-cyan-500 text-slate-950{% else %}bg-slate-900 text-slate-400 border border-slate-800{% endif %}">ทั้งหมด</a>
            <a href="/gallery?filter=IN" class="px-3.5 py-1.5 rounded-xl text-xs font-bold {% if active_filter == 'IN' %}bg-emerald-500 text-slate-950{% else %}bg-slate-900 text-slate-400 border border-slate-800{% endif %}">PERSON IN</a>
            <a href="/gallery?filter=OUT" class="px-3.5 py-1.5 rounded-xl text-xs font-bold {% if active_filter == 'OUT' %}bg-rose-500 text-slate-950{% else %}bg-slate-900 text-slate-400 border border-slate-800{% endif %}">PERSON OUT</a>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {% for img in images %}
            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden p-2 group relative">
                <input type="checkbox" value="{{ img }}" onchange="updateSelectedCount()" class="img-checkbox absolute top-4 left-4 w-5 h-5 accent-cyan-500 z-10 cursor-pointer">
                
                <div class="relative overflow-hidden rounded-xl bg-black cursor-pointer" onclick="openLightbox('/captures/{{ img }}', '{{ img }}')">
                    <img src="/captures/{{ img }}" class="w-full aspect-video object-cover rounded-xl group-hover:scale-105 transition duration-300 shadow-md">
                    <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition duration-200 flex items-center justify-center text-white gap-2">
                        <span class="bg-black/60 px-2.5 py-1 rounded-lg text-xs font-bold"><i class="fa-solid fa-magnifying-glass-plus mr-1"></i>ดูรูปใหญ่</span>
                    </div>
                </div>

                <div class="p-2 flex items-center justify-between gap-1">
                    <div class="text-[11px] text-slate-300 font-mono truncate flex-1" title="{{ img }}">
                        {{ img }}
                    </div>
                    <a href="/captures/{{ img }}" download="{{ img }}" class="bg-slate-800 hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-400 p-1.5 rounded-lg text-xs transition border border-slate-700" title="ดาวน์โหลดรูปนี้">
                        <i class="fa-solid fa-download"></i>
                    </a>
                </div>
            </div>
            {% else %}
            <div class="col-span-full text-center py-12 text-slate-500 text-xs">ไม่พบรูปภาพในหมวดหมู่นี้</div>
            {% endfor %}
        </div>
    </div>

    <!-- LIGHTBOX MODAL -->
    <div id="lightbox-modal" class="fixed inset-0 bg-black/90 backdrop-blur-md z-[100] hidden flex flex-col items-center justify-center p-4" onclick="closeLightbox()">
        <div class="cyber-card max-w-4xl w-full rounded-3xl p-4 border border-slate-800 shadow-2xl relative flex flex-col items-center" onclick="event.stopPropagation()">
            <div class="w-full flex justify-between items-center border-b border-slate-800 pb-3 mb-3">
                <h3 id="lightbox-title" class="text-xs font-mono font-bold text-cyan-400 truncate">รูปภาพ</h3>
                <div class="flex items-center gap-2">
                    <a id="lightbox-download-btn" href="" download class="bg-cyan-500/20 hover:bg-cyan-500/40 text-cyan-300 border border-cyan-500/30 px-3 py-1 rounded-xl text-xs font-bold transition">
                        <i class="fa-solid fa-download mr-1"></i>ดาวน์โหลด
                    </a>
                    <button onclick="closeLightbox()" class="text-slate-400 hover:text-white text-lg px-2"><i class="fa-solid fa-xmark"></i></button>
                </div>
            </div>
            <div class="w-full flex items-center justify-center max-h-[75vh] overflow-hidden rounded-2xl bg-slate-950 p-2 border border-slate-800/80">
                <img id="lightbox-img" src="" class="max-h-[70vh] max-w-full object-contain rounded-xl">
            </div>
        </div>
    </div>

    <script>
        let allSelected = false;

        function openLightbox(src, title) {
            document.getElementById('lightbox-img').src = src;
            document.getElementById('lightbox-title').innerText = title;
            document.getElementById('lightbox-download-btn').href = src;
            document.getElementById('lightbox-download-btn').setAttribute('download', title);
            document.getElementById('lightbox-modal').classList.remove('hidden');
        }

        function closeLightbox() {
            document.getElementById('lightbox-modal').classList.add('hidden');
            document.getElementById('lightbox-img').src = "";
        }

        function updateSelectedCount() {
            const checkboxes = document.querySelectorAll('.img-checkbox:checked');
            const count = checkboxes.length;
            const selCountEl = document.getElementById('selected-count');
            const dlCountEl = document.getElementById('download-count');
            if (selCountEl) selCountEl.innerText = count;
            if (dlCountEl) dlCountEl.innerText = count;
        }

        function toggleSelectAll() {
            allSelected = !allSelected;
            const checkboxes = document.querySelectorAll('.img-checkbox');
            checkboxes.forEach(cb => cb.checked = allSelected);
            updateSelectedCount();
        }

        function getSelectedFilenames() {
            const checkboxes = document.querySelectorAll('.img-checkbox:checked');
            return Array.from(checkboxes).map(cb => cb.value);
        }

        function downloadSelectedImages() {
            const selected = getSelectedFilenames();
            if (selected.length === 0) {
                alert('กรุณาติ๊กเลือกรูปภาพที่ต้องการดาวน์โหลดก่อน!');
                return;
            }

            fetch('/api/download_captures_zip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filenames: selected })
            })
            .then(res => {
                if (!res.ok) throw new Error("เกิดข้อผิดพลาดในการดาวน์โหลด");
                return res.blob();
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `captures_${new Date().toISOString().slice(0,10)}.zip`;
                document.body.appendChild(a);
                a.click();
                a.remove();
            })
            .catch(err => alert(err.message));
        }

        function deleteSelectedImages() {
            const selected = getSelectedFilenames();
            if (selected.length === 0) {
                alert('กรุณาติ๊กเลือกรูปภาพที่ต้องการลบ!');
                return;
            }

            if (!confirm(`คุณแน่ใจหรือไม่ว่าต้องการลบรูปภาพจำนวน ${selected.length} รูป?`)) return;

            fetch('/api/delete_captures', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filenames: selected })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    window.location.reload();
                } else {
                    alert(data.message || 'เกิดข้อผิดพลาดในการลบรูป');
                }
            });
        }

        function deleteAllImages() {
            const allCheckboxes = document.querySelectorAll('.img-checkbox');
            const allFiles = Array.from(allCheckboxes).map(cb => cb.value);

            if (allFiles.length === 0) {
                alert('ไม่มีรูปภาพให้ลบ!');
                return;
            }

            if (!confirm(`⚠️ คำเตือน: คุณต้องการลบรูปภาพทั้งหมดจำนวน ${allFiles.length} รูปใช่หรือไม่?`)) return;

            fetch('/api/delete_captures', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filenames: allFiles })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    window.location.reload();
                } else {
                    alert(data.message || 'เกิดข้อผิดพลาดในการลบรูปภาพ');
                }
            });
        }
    </script>
</body>
</html>
"""

# ==========================================
# API ROUTING & AUTH
# ==========================================
@app.route("/raw_video_feed")
def raw_video_feed():
    return Response(generate_raw_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/known_faces")
@requires_auth
def get_known_faces():
    faces = []
    if os.path.exists(KNOWN_FACES_DIR):
        for f in os.listdir(KNOWN_FACES_DIR):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(f)[0]
                faces.append({"name": name, "filename": f})
    return jsonify(faces)

@app.route("/known_faces/<filename>")
@requires_auth
def serve_known_face_img(filename):
    return send_from_directory(KNOWN_FACES_DIR, filename)

@app.route("/api/enroll_single_photo", methods=["POST"])
@requires_auth
@admin_only
def enroll_single_photo():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "ไม่ได้รับข้อมูล JSON"}), 400

    name = data.get("name", "").strip()
    index = data.get("index", 1)

    if not name:
        return jsonify({"status": "error", "message": "กรุณากรอกชื่อผู้ใช้งาน"}), 400

    clean_name = secure_filename(name)
    
    pose_names = [
        "Frontal View (หน้าตรง)", 
        "Left View (มองซ้าย)", 
        "Right View (มองขวา)", 
        "Tilt Up View (เงยหน้า)", 
        "Tilt Down View (ก้มหน้า)"
    ]
    pose_label = pose_names[index-1] if index <= len(pose_names) else f"Pose_{index}"

    success, frame = cam_stream.read()
    if not success or frame is None:
        return jsonify({"status": "error", "message": "ไม่สามารถดึงภาพจากกล้องได้"}), 500

    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        (fx, fy, f_w, f_h) = faces[0]
        pad_w = int(f_w * 0.25)
        pad_h = int(f_h * 0.35)

        x1 = max(0, fx - pad_w)
        y1 = max(0, fy - pad_h)
        x2 = min(fw, fx + f_w + pad_w)
        y2 = min(fh, fy + f_h + pad_h)
        face_img = frame[y1:y2, x1:x2]
    else:
        results = model(frame, classes=[PERSON_CLASS_ID], conf=0.2, verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
            head_h = int((py2 - py1) * 0.45)
            face_img = frame[py1:min(fh, py1 + head_h), px1:px2]
        else:
            face_img = frame

    if face_img.size > 0:
        face_img_resized = cv2.resize(face_img, (200, 200))
        cv2.rectangle(face_img_resized, (0, 0), (200, 22), (0, 0, 0), -1)
        face_img_resized = draw_text_pil(face_img_resized, f"{pose_label}", position=(5, 2), font_size=13, color=(0, 255, 255))
        
        filename = f"{clean_name}_{index}.jpg"
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        cv2.imwrite(filepath, face_img_resized)

    load_known_faces()
    return jsonify({"status": "success", "message": f"ถ่ายรูปที่ {index} สำเร็จ"})

@app.route("/api/add_face", methods=["POST"])
@requires_auth
@admin_only
def add_face():
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({"status": "error", "message": "ข้อมูลไม่ครบถ้วน"}), 400

    file = request.files['file']
    name = request.form['name'].strip()

    if not file or not name:
        return jsonify({"status": "error", "message": "กรุณากรอกชื่อและเลือกไฟล์รูป"}), 400

    clean_name = secure_filename(name)
    filename = f"{clean_name}.jpg"
    filepath = os.path.join(KNOWN_FACES_DIR, filename)

    file.save(filepath)
    load_known_faces()

    return jsonify({"status": "success", "message": f"อัปโหลดใบหน้า '{name}' สำเร็จ!"})

@app.route("/api/rename_face", methods=["POST"])
@requires_auth
@admin_only
def rename_face():
    data = request.get_json()
    old_filename = data.get("old_filename")
    new_name = data.get("new_name", "").strip()

    if not old_filename or not new_name:
        return jsonify({"status": "error", "message": "ข้อมูลไม่ครบถ้วน"}), 400

    clean_new_name = secure_filename(new_name)
    ext = os.path.splitext(old_filename)[1]
    new_filename = f"{clean_new_name}{ext}"

    old_filepath = os.path.join(KNOWN_FACES_DIR, old_filename)
    new_filepath = os.path.join(KNOWN_FACES_DIR, new_filename)

    if not os.path.exists(old_filepath):
        return jsonify({"status": "error", "message": "ไม่พบไฟล์ใบหน้าเดิมในระบบ"}), 404

    try:
        os.rename(old_filepath, new_filepath)
        load_known_faces()
        return jsonify({"status": "success", "message": f"เปลี่ยนชื่อเป็น '{new_name}' เรียบร้อยแล้ว!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"เกิดข้อผิดพลาดในการเปลี่ยนชื่อ: {str(e)}"}), 500

@app.route("/api/delete_face", methods=["POST"])
@requires_auth
@admin_only
def delete_face():
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"status": "error", "message": "ไม่พบชื่อไฟล์"}), 400

    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        load_known_faces()
        return jsonify({"status": "success", "message": "ลบใบหน้าเรียบร้อยแล้ว"})

    return jsonify({"status": "error", "message": "ไม่พบไฟล์ในระบบ"}), 404

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username in USERS_DB and USERS_DB[username]["password"] == password:
            session['logged_in'] = True
            session['username'] = username
            session['role'] = USERS_DB[username]["role"]
            session['session_id'] = f"{username}_{uuid.uuid4().hex[:8]}"
            return redirect(url_for('index'))
        else:
            error = "Username หรือ Password ไม่ถูกต้อง"

    return render_template_string(LOGIN_TEMPLATE, error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/")
@requires_auth
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/heartbeat", methods=["POST"])
@requires_auth
def heartbeat():
    session_id = session.get('session_id')
    username = session.get('username', 'Guest')
    user_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    now = time.time()

    if not session_id:
        session_id = f"{username}_{uuid.uuid4().hex[:8]}"
        session['session_id'] = session_id

    with session_lock:
        if session_id in kicked_sessions:
            kicked_sessions.remove(session_id)
            if session_id in active_sessions:
                del active_sessions[session_id]
            session.clear()
            return jsonify({"status": "kicked", "message": "คุณถูกลบออกจากระบบโดย Admin"}), 401

        active_sessions[session_id] = {
            "session_id": session_id,
            "username": username,
            "role": session.get('role', 'user'),
            "ip": user_ip,
            "last_seen": now,
            "last_seen_str": datetime.datetime.now().strftime("%H:%M:%S")
        }
        active_count = len(active_sessions)
    return jsonify({"status": "ok", "active_users": active_count})

@app.route("/api/active_users_list")
@requires_auth
def active_users_list():
    with session_lock:
        users_list = list(active_sessions.values())
    return jsonify(users_list)

@app.route("/api/kick_user", methods=["POST"])
@requires_auth
@admin_only
def kick_user():
    data = request.get_json()
    target_session_id = data.get("session_id")

    if not target_session_id:
        return jsonify({"status": "error", "message": "ไม่พบข้อมูลเซสชัน"}), 400

    with session_lock:
        if target_session_id in active_sessions:
            if active_sessions[target_session_id]["role"] == "admin":
                return jsonify({"status": "error", "message": "ไม่สามารถลบบัญชี Admin ได้"}), 403

            kicked_sessions.add(target_session_id)
            del active_sessions[target_session_id]
            return jsonify({"status": "success", "message": "เตะผู้ใช้เรียบร้อยแล้ว"})

    return jsonify({"status": "error", "message": "ไม่พบผู้ใช้ในระบบออนไลน์"}), 404

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/reset_camera", methods=["POST"])
@requires_auth
@admin_only
def reset_camera():
    global cam_stream
    try:
        cam_stream.stopped = True
        if hasattr(cam_stream, 'cap') and cam_stream.cap is not None:
            cam_stream.cap.release()
        
        time.sleep(0.5)
        cam_stream = RTSPVideoWriter(HIKVISION_RTSP_URL)
        return jsonify({"status": "success", "message": "รีเซ็ตสัญญาณกล้องเรียบร้อยแล้ว"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/set_lines", methods=["POST"])
@requires_auth
@admin_only
def set_lines():
    global line1_coords, line2_coords
    data = request.get_json()
    with data_lock:
        if "line1" in data:
            line1_coords["x1"] = float(data["line1"].get("x1", 0.32))
            line1_coords["y1"] = float(data["line1"].get("y1", 0.0))
            line1_coords["x2"] = float(data["line1"].get("x2", 0.32))
            line1_coords["y2"] = float(data["line1"].get("y2", 1.0))
        if "line2" in data:
            line2_coords["x1"] = float(data["line2"].get("x1", 0.40))
            line2_coords["y1"] = float(data["line2"].get("y1", 0.0))
            line2_coords["x2"] = float(data["line2"].get("x2", 0.40))
            line2_coords["y2"] = float(data["line2"].get("y2", 1.0))
    return jsonify({"status": "success"})

@app.route("/manual_snapshot", methods=["POST"])
def manual_snapshot():
    success, frame = cam_stream.read()
    if not success or frame is None:
        return jsonify({"status": "error"}), 500

    now = datetime.datetime.now()
    timestamp_file = now.strftime("%Y%m%d_%H%M%S")
    date_time_display = now.strftime("%Y-%m-%d %H:%M:%S")
    hour_str = now.strftime("%H:00")
    filename_only = f"MANUAL_SNAPSHOT_{timestamp_file}.jpg"
    filename = os.path.join(SAVE_DIR, filename_only)

    saved_frame = frame.copy()
    cv2.rectangle(saved_frame, (10, 10), (520, 48), (0, 0, 0), -1)
    stamp_text = f"{date_time_display} | [MANUAL] Manual Snapshot"
    saved_frame = draw_text_pil(saved_frame, stamp_text, position=(20, 15), font_size=20, color=(255, 255, 0))

    cv2.imwrite(filename, saved_frame)

    with data_lock:
        visit_logs.append({
            "id": len(visit_logs) + 1,
            "name": "Manual",
            "display_name": "Manual Snapshot",
            "direction": "MANUAL",
            "time": now.strftime("%H:%M:%S"),
            "hour": hour_str,
            "filename": filename_only,
        })
    return jsonify({"status": "success"})

@app.route("/get_count")
def get_count():
    all_files = glob.glob(os.path.join(SAVE_DIR, "*.jpg"))
    with data_lock:
        return jsonify({
            "person_count": person_count,
            "total_in": total_in,
            "total_out": total_out,
            "all_captures_count": len(all_files),
            "line1_coords": line1_coords,
            "line2_coords": line2_coords,
            "daily_stats": daily_stats,
            "weekly_stats": weekly_stats
        })

@app.route("/get_visit_logs")
def get_visit_logs():
    with data_lock:
        return jsonify(visit_logs)

@app.route("/gallery")
@requires_auth
def gallery():
    active_filter = request.args.get('filter', 'ALL').upper()
    all_files = []
    if os.path.exists(SAVE_DIR):
        files = sorted(os.listdir(SAVE_DIR), reverse=True)
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                if active_filter == 'ALL':
                    all_files.append(f)
                elif active_filter in f:
                    all_files.append(f)
    return render_template_string(GALLERY_TEMPLATE, images=all_files, active_filter=active_filter)

@app.route("/captures/<filename>")
@requires_auth
def serve_capture_image(filename):
    return send_from_directory(SAVE_DIR, filename, as_attachment=False)

@app.route("/api/download_captures_zip", methods=["POST"])
@requires_auth
def download_captures_zip():
    data = request.get_json()
    filenames = data.get("filenames", [])

    if not filenames:
        return jsonify({"status": "error", "message": "ไม่ได้เลือกรูปภาพ"}), 400

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in filenames:
            safe_fname = secure_filename(fname)
            filepath = os.path.join(SAVE_DIR, safe_fname)
            if os.path.exists(filepath):
                zf.write(filepath, safe_fname)

    memory_file.seek(0)
    zip_filename = f"captures_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name=zip_filename)

@app.route("/api/delete_captures", methods=["POST"])
@requires_auth
@admin_only
def delete_captures():
    data = request.get_json()
    filenames = data.get("filenames", [])
    
    if not filenames:
        return jsonify({"status": "error", "message": "ไม่ได้เลือกรูปภาพ"}), 400

    deleted_count = 0
    with data_lock:
        for fname in filenames:
            safe_fname = secure_filename(fname)
            filepath = os.path.join(SAVE_DIR, safe_fname)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    print(f"[Delete Error] {e}")

        global visit_logs
        visit_logs = [log for log in visit_logs if log.get("filename") not in filenames]

    return jsonify({"status": "success", "message": f"ลบรูปภาพเรียบร้อยแล้ว {deleted_count} รูป"})

@app.route("/summary")
@requires_auth
def summary():
    reports = []
    if os.path.exists(REPORT_DIR):
        for item in sorted(os.listdir(REPORT_DIR), reverse=True):
            if (item.startswith("person_summary_") or item.startswith("weekly_summary_") or item.startswith("daily_summary_")) and item.endswith(".csv"):
                reports.append(item)
    return render_template_string(SUMMARY_TEMPLATE, reports=reports)

@app.route("/download_report/<filename>")
@requires_auth
def download_report(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    clean_thread = threading.Thread(target=auto_clean_old_captures, daemon=True)
    clean_thread.start()
    
    session_clean_thread = threading.Thread(target=clean_expired_sessions, daemon=True)
    session_clean_thread.start()
    
    app.run(host="0.0.0.0", port=5000, debug=False)