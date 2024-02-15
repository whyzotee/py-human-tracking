import cv2
import requests
import numpy as np
import RPi.GPIO as GPIO
from ultralytics import YOLO

relay = 37
model = YOLO("yolov5nu.pt")
# cap = cv2.VideoCapture('http://192.168.4.1/stream')
url = 'http://192.168.4.1/snapshot'


GPIO.setmode(GPIO.BOARD)
GPIO.setup(relay, GPIO.OUT)

while True:
    # ret, imgDecode = cap.read()

    imageURL = requests.get(url)
    imageArray = np.frombuffer(bytearray(imageURL.content), dtype=np.uint8)
    imgDecode = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)

    results = model.predict(source=imgDecode, imgsz=256)[0]
    data = results.boxes.data.tolist()

    if len(data) > 0 and data[0][5] == 0.0:
        if data[0][4] < 0.8:
            continue
        GPIO.output(relay, GPIO.HIGH)
    else:
        GPIO.output(relay, GPIO.LOW)

GPIO.cleanup()
print("End Loop")
