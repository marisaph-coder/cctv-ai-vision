import os
import urllib.request
import cv2
import numpy as np

# IP Address ของ ESP32-CAM
stream_url = "http://192.168.30.197:81/stream"

# ชื่อไฟล์โมเดล
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"

# ตรวจสอบว่ามีไฟล์โมเดลอยู่ในโฟลเดอร์หรือไม่
if not os.path.exists(model_file) or not os.path.exists(config_file):
    print("ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ deploy.prototxt และ res10_300x300_ssd_iter_140000.caffemodel ในโฟลเดอร์")
    exit()

# โหลดโมเดลตรวจจับใบหน้าโดยใช้ cv2.dnn.readNet
net = cv2.dnn.readNet(model_file, config_file)

# เชื่อมต่อกับกล้อง ESP32-CAM
try:
    stream = urllib.request.urlopen(stream_url)
except Exception as e:
    print("ไม่สามารถเชื่อมต่อกล้อง ESP32-CAM ได้:", e)
    exit()

bytes_buffer = b""
print("เชื่อมต่อสำเร็จ! กำลังตรวจจับใบหน้า (กด 'q' เพื่อปิด)")

while True:
    bytes_buffer += stream.read(1024)
    a = bytes_buffer.find(b"\xff\xd8")
    b = bytes_buffer.find(b"\xff\xd9")

    if a != -1 and b != -1:
        jpg = bytes_buffer[a : b + 2]
        bytes_buffer = bytes_buffer[b + 2 :]

        frame = cv2.imdecode(
            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        if frame is not None:
            (h, w) = frame.shape[:2]

            # แปลงภาพเพื่อส่งเข้าโมเดล DNN
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0),
            )
            net.setInput(blob)
            detections = net.forward()

            # แสดงผลกรอบตรวจจับใบหน้า
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # กรองเอาเฉพาะค่าความมั่นใจมากกว่า 50%
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    cv2.rectangle(
                        frame, (startX, startY), (endX, endY), (0, 255, 0), 2
                    )
                    text = f"Face: {confidence * 100:.1f}%"
                    cv2.putText(
                        frame,
                        text,
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("ESP32-CAM: Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()