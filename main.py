import cv2
import datetime
import numpy as np
import urllib.request
from ultralytics import YOLO

url = 'http://192.168.31.36/stream'
led_on = "http://192.168.31.36/led_on"
led_off = "http://192.168.31.36/led_off"

video_cap = cv2.VideoCapture("./test_vdo.mp4")
model = YOLO("yolov5nu.pt")

mode = 0  # 0 image from url, 1 test video

# while((1 + 1) == 2):
#     imageURL = urllib.request.urlopen(url)
#     imageByte = np.asarray(bytearray(imageURL.read()), dtype=np.uint8)
#     imgProc = cv2.imdecode(imageByte, -1)
#     results = model(imgProc)[0]
#
#     print(results)
#
#     cv2.imshow('Test Image Proc', imgProc)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

while((1 + 1) == 2):
    start = datetime.datetime.now()

    if mode != 0:
        ret, imgProc = video_cap.read()
        detactions = model(imgProc)[0]
    else:
        imageURL = urllib.request.urlopen(url)
        imageByte = np.asarray(bytearray(imageURL.read()), dtype=np.uint8)
        imgProc = cv2.imdecode(imageByte, -1)
        detactions = model(imgProc)[0]

    for data in detactions.boxes.data.tolist():
        confidencs = data[4]

        if float(confidencs) < 0.8:
            continue

        xmin, ymin, xman, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(imgProc, (xmin, ymin), (xman, ymax), (0, 0, 255), 2)
    
    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} millisec.")

    fps = f"FPS: {1/total:.2f}"
    print(fps)
    
    cv2.putText(imgProc, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    cv2.imshow("Detect human.exe", imgProc)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_cap.release()
cv2.destroyAllWindows()