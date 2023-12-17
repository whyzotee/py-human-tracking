import cv2
import requests
import numpy as np
from ultralytics import YOLO

url = 'http://192.168.31.36/stream'
led_on = "http://192.168.31.36/led_on"
led_off = "http://192.168.31.36/led_off"

isled_on = False

model = YOLO("yolov5nu.pt")

while (1 + 1) == 2:
    imageURL = requests.get(url)
    imageArray = np.frombuffer(bytearray(imageURL.content), dtype=np.uint8)
    imgDecode = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)

    results = model.predict(source=imgDecode, imgsz=256)[0]
    data = results.boxes.data.tolist()

    if len(data) > 0 and data[0][5] == 0.0:
        if data[0][4] < 0.8:
            continue
        if not isled_on:
            requests.get(led_on)
            isled_on = True
    else:
        if isled_on:
            requests.get(led_off)
            isled_on = False
