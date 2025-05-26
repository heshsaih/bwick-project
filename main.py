import cv2
import torch
from facenet_pytorch import MTCNN
from deepface import DeepFace
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)

detector = mtcnn
cam = cv2.VideoCapture(2)

if not cam.isOpened():
    IOError("Could not open camera. Please check the camera index.")
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
            x1, y1, x2, y2 = [int(coord / SCALE) for coord in box]
            face_img = frame[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, (224, 224))
            try:
                result = DeepFace.analyze(
                    img_path=face_img,
                    actions=['race'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                res = result[0] if isinstance(result, list) else result
                race = res['dominant_race']
                confidence = res['race'][race]

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{race} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"Error analyzing face: {e}")

    cv2.imshow('Race Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()