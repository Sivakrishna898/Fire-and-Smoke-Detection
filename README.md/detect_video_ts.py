import sys
import cv2
import torch

# Use local YOLOv5 code (offline)
sys.path.append("yolov5")

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

device = select_device("")
model = DetectMultiBackend("weights/best.pt", device=device)
stride, names = model.stride, model.names

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
 
if not cap.isOpened():
    print("❌ Webcam not opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = letterbox(frame, 640, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR->RGB
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Fire & Smoke Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
