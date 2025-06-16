import os
import cv2
import torch
from facenet_pytorch import MTCNN
from deepface import DeepFace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(keep_all=True, device=device)

DATASET_DIR = "dataset"

CLASSES = ["white", "black", "asian"]

results = {
    "total": 0,
    "correct": 0,
    "white": {
        "predicted": 0,
        "hit": 0,
        "errors": []
    },
    "black": {
        "predicted": 0,
        "hit": 0,
        "errors": []
    },
    "asian": {
        "predicted": 0,
        "hit": 0,
        "errors": []
    },
    "else": {
        "predicted": 0,
        "hit": 0,
        "errors": []
    }
}


for true_label in CLASSES:
    class_dir = os.path.join(DATASET_DIR, true_label)
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"Error reading image {fpath}")
            continue
        print(f"Analizing file: {fpath}...")

        try:
            res = DeepFace.analyze(
                img_path=img,
                actions=["race"],
                enforce_detection=False,
                detector_backend='mtcnn'
            )
            r = res[0] if isinstance(res, list) else res
            pred = r["dominant_race"].lower()
        except Exception as e:
            print(f"Error analyzing image {fpath}: {e}")
            continue

        results["total"] += 1
        if pred in CLASSES:
            results[pred]["predicted"] += 1
        else:
            results["else"]["predicted"] += 1
        if pred == true_label:
            results["correct"] += 1
            results[pred]["hit"] += 1
        elif pred in CLASSES:
            results[pred]["errors"].append({
                "fpath": fpath,
                "actual": true_label,
                "predicted": pred
            })
        else:
            results["else"]["errors"].append({
                "fpath": fpath,
                "actual": true_label,
                "predicted": pred
            })

print("Results:")
total = results["total"]
print(f"Total analyzed images: {total}")
correct = results["correct"]
print(f"Correct predictions: {correct}")
accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}%")
print("Statistics for different races:\n")
for race in results.keys():
    if race in CLASSES:
        print(f"{race}: ")
        total_predictions = results[race]["predicted"]
        print(f"\tTotal predictions: {total_predictions}")
        correct_predictions = results[race]["hit"]
        print(f"\tCorrect predictions: {correct_predictions}")
        race_accuracy = (correct_predictions / total_predictions) * 100
        print(f"\tPrediction accuracy for the race: {race_accuracy:.2f}%")
        print("Errors: ")
        for error in results[race]["errors"]:
            err_fpath = error["fpath"]
            print(f"\n\t\tFile: {err_fpath}")
            err_predicted = error["predicted"]
            print(f"\t\tPredicted: {err_predicted}")
            err_actual = error["actual"]
            print(f"\t\tActual: {err_actual}")
        print("")
    elif race == "else":
        print(f"{race}: ")
        total_misses = results[race]["predicted"]
        print(f"\tTotal misses: {total_misses}")
        percentage = (total_misses / total) * 100
        print(f"\tPercentage of misses: {percentage:.2f}%")
        print("Errors: ")
        for error in results[race]["errors"]:
            err_fpath = error["fpath"]
            print(f"\n\t\tFile: {err_fpath}")
            err_predicted = error["predicted"]
            print(f"\t\tPredicted: {err_predicted}")
            err_actual = error["actual"]
            print(f"\t\tActual: {err_actual}")
