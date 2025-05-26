import cv2
import torch
from facenet_pytorch import MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)

detector = mtcnn
cam = cv2.VideoCapture(2)

if not cam.isOpened():
    print("Error: Camera not found.")
    exit()

SCALE = 0.5

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    boxes, _ = detector.detect(small_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = int(x1 / SCALE), int(y1 / SCALE)
            x2, y2 = int(x2 / SCALE), int(y2 / SCALE)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()