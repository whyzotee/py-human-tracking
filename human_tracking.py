import cv2
import requests
import numpy as np
import cvlib as cv
import urllib.request
from cvlib.object_detection import draw_bbox

url = 'http://192.168.31.36/stream'
led_on = "http://192.168.31.36/led_on"
led_off = "http://192.168.31.36/led_off"

def detect_people(image):
    bbox, label, conf = cv.detect_common_objects(image, model="yolov3-tiny")
    draw_bbox(image, bbox, label, conf)
    return label

def main():
    cv2.namedWindow("Hello World", cv2.WINDOW_NORMAL)

    while True:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)

        lables = detect_people(img)

        cv2.imshow('Hello World', img)

        if "person" in lables:
            requests.get(led_on)
        else:
            requests.get(led_off)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
