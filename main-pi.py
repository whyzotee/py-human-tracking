import cv2
import numpy as np
import RPi.GPIO as GPIO
from ultralytics import YOLO

relay = 37
model = YOLO("yolov5nu.pt")
cap = cv2.VideoCapture('http://192.168.4.1/stream')


GPIO.setmode(GPIO.BOARD)
GPIO.setup(relay, GPIO.OUT)

while cap.isOpened():
    ret, imgDecode = cap.read()

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
