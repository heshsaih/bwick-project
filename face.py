import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from deepface import DeepFace
from queue import Queue
from threading import Thread

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(keep_all=True, device=device)

# Initialize queues for task management
task_queue = Queue()
result_queue = Queue()

def analyzer_worker_faces():
    while True:
        faces = task_queue.get()
        if faces is None:
            break
        parsed = []
        for face in faces:
            try:
                res = DeepFace.analyze(
                    img_path=face,
                    actions=['race'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                result = res[0] if isinstance(res, list) else res
                label = result['dominant_race']
                confidence = result['race'].get(label, 0)
            except Exception as e:
                print(f"Analyzer error for one face: {e}")
                label, confidence = None, 0
            parsed.append((label, confidence))
        result_queue.put(parsed)
        task_queue.task_done()

# Start the face analysis thread
analyzer = Thread(target=analyzer_worker_faces, daemon=True)
analyzer.start()

dam = cv2.VideoCapture(2)
if not dam.isOpened():
    raise IOError("Could not open camera. Check the camera index.")

# Constants for processing image frames
SCALE = 0.5
ANALYZE_INTERVAL = 0.25  # Analyze every 0.25 seconds
frame_count = 0
last_results = []

# Constants for processing audio input

print("Starting race detector. Press 'q' to quit.")

# Main loop for reading frames from the camera
while True:
    ret, frame = dam.read()
    if not ret:
        break
    frame_count += 1

    small = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)
    boxes, _ = detector.detect(small)

    coords = []
    faces_to_analyze = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(c/SCALE) for c in box]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            coords.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_small = cv2.resize(face_crop, (64,64), interpolation=cv2.INTER_NEAREST)
            faces_to_analyze.append(face_small)

    if frame_count % ANALYZE_INTERVAL == 0 and faces_to_analyze and task_queue.empty():
        task_queue.put(faces_to_analyze)

    try:
        last_results = result_queue.get_nowait()
    except:
        pass

    display_results = []
    for i in range(len(coords)):
        if i < len(last_results):
            display_results.append(last_results[i])
        else:
            display_results.append((None,0))

    for (x1, y1, x2, y2), (race, conf) in zip(coords, display_results):
        label = f"{race} ({conf:.0f}%)" if race else "Unknown"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('Race Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
dam.release()
cv2.destroyAllWindows()
task_queue.put(None)
analyzer.join()
