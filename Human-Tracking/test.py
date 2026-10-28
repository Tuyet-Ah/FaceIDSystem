import cv2

from ultralytics import YOLO

# Tải model YOLO
model = YOLO("weights/yolo26s.pt")

# # Mở tệp video
cap = cv2.VideoCapture("data/YTSave.to_YouTube_Media_tOoRl4yecko_Chain-Robbery-Incident-in-India-Caught-in-CCTV-cameras-CCTV-Footage_002_360p.mp4.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Chạy theo dõi với trạng thái được duy trì giữa các khung hình
        results = model.track(frame, persist=True, tracker = "weights/tracktrack_persist.yaml", device = "cpu")  # Sử dụng CPU để theo dõi

        # Trực quan hóa kết quả
        annotated_frame = results[0].plot()
        cv2.imshow("Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()