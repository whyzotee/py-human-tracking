from datetime import datetime

import cv2
import torch
import datetime
import requests
import torchvision
import numpy as np

url = 'http://192.168.31.36/stream'
led_on = "http://192.168.31.36/led_on"
led_off = "http://192.168.31.36/led_off"

model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

while (1 + 1) == 2:
    start = datetime.datetime.now()

    imageStream = requests.get(url, stream=True)
    imageStream.raise_for_status()
    imageStream = imageStream.raw
    imageByte = np.asarray(bytearray(imageStream.read()), dtype=np.uint8)
    imgReal = cv2.imdecode(imageByte, cv2.IMREAD_COLOR)

    image_resize = cv2.resize(imgReal, (640, 480))
    img = image_resize.copy()
    imgTransform = torchvision.transforms.ToTensor()
    image = imgTransform(image_resize)

    with torch.no_grad():
        ypred = model([image])

        bbox, scores, labels = ypred[0]["boxes"], ypred[0]["scores"], ypred[0]["labels"]
        nums = torch.argwhere(scores > 0.7).shape[0]
        for i in range(nums):
            x, y, w, h = bbox[i].numpy().astype('int')
            cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 5)

    end: datetime = datetime.datetime.now()
    total: float = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} millisec.")

    fps = f"FPS: {1 / total:.2f}"
    print(fps)

    cv2.putText(image_resize, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    cv2.imshow('Image Proc Test', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
