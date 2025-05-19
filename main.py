import cv2

cam = cv2.VideoCapture(0)
i = 12

if not cam.isOpened():
    print("Error: Camera not found.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
