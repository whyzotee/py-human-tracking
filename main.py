import cv2
import datetime
import requests
import numpy as np
from ultralytics import YOLO

url = 'http://192.168.31.36/stream'
led_on = "http://192.168.31.36/led_on"
led_off = "http://192.168.31.36/led_off"

isled_on = False

model = YOLO("yolov5nu.pt")

while (1 + 1) == 2:
    start = datetime.datetime.now()

    imageURL = requests.get(url)
    imageArray = np.frombuffer(bytearray(imageURL.content), dtype=np.uint8)
    imgDecode = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)

    # detactions = model(imgDecode)[0]
    detactions = model.predict(source=imgDecode, imgsz=320)[0]

    for data in detactions.boxes.data.tolist():
        confidencs = data[4]

        if float(confidencs) < 0.8 or float(data[5]) != 0.0:
            continue

        xmin, ymin, xman, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(imgDecode, (xmin, ymin), (xman, ymax), (0, 0, 255), 2)

    if len(detactions.boxes.data.tolist()) > 0 and detactions.boxes.data.tolist()[0][5] == 0.0:
        if detactions.boxes.data.tolist()[0][4] < 0.8:
            continue

        if not isled_on:
            requests.get(led_on)
            isled_on = True
    else:
        if isled_on:
            requests.get(led_off)
            isled_on = False

    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} millisec.")

    fps = f"FPS: {1 / total:.2f}"
    print(fps)

    cv2.putText(imgDecode, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 8)

    cv2.imshow("Detect human.exe", imgDecode)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
