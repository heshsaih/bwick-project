import os
import cv2 
from deepface import DeepFace

DATASET_DIR = "dataset"

CLASSES = ["white", "black", "asian"]

correct = 0
total = 0

errors = []

for true_label in CLASSES:
    class_dir = os.path.join(DATASET_DIR, true_label)
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"Error reading image {fpath}")
            continue
        
        try:
            res = DeepFace.analyze(
                img_path = img,
                actions = ["race"],
                enforce_detection = False,
                detector_backend = 'mtcnn'
            )
            r = res[0] if isinstance(res, list) else res
            pred = r["dominant_race"].lower()
        except Exception as e:
            print(f"Error analyzing image {fpath}: {e}")
            continue
            
        total += 1
        if pred == true_label:
            correct += 1
        else:
            errors.append((fpath, true_label, pred))

accuracy = 100 * correct / total if total else 0
print(f"Total images: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")

#print("Errors:")
#for e in errors:
    #print(f"File: {e[0]}, True label: {e[1]}, Predicted: {e[2]}")