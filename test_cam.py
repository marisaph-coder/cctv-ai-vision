import cv2

# ใส่ IP Address ของ ESP32-CAM (ปรับตัวเลขตามที่คุณเห็นใน Serial Monitor)
stream_url = "http://192.168.30.197:81/stream"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("ไม่สามารถเชื่อมต่อกล้องได้ เช็ก IP Address อีกครั้ง")
    exit()

print("เชื่อมต่อสำเร็จ! กด 'q' ที่คีย์บอร์ดเพื่อปิดหน้าต่าง")

while True:
    ret, frame = cap.read()
    if not ret:
        print("สัญญาณภาพขาดหาย")
        break

    cv2.imshow("ESP32-CAM Live Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()