import cv2
import datetime
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture("test_vdo.mp4")
model = YOLO("yolov5nu.pt")

while True:
    start = datetime.datetime.now()
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    detactions = model(frame)[0]

    for data in detactions.boxes.data.tolist():
        confidencs = data[4]

        if float(confidencs) < 0.2 or float(data[5]) != 0.0:
            continue

        xmin, ymin, xman, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin), (xman, ymax), (0, 0, 255), 2)
   
    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} millisec.")

    fps = f"FPS: {1/total:.2f}"
    print(fps)
    
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
