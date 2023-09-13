# we are going to predt hwo is our model performing
# we imported all this lime from the documentation
from ultralytics import YOLO
import numpy as np

model = YOLO(
    '/home/golu/PycharmProjects/weatherImageClassificationYOLOv8/runs/classify/train2/weights/last.pt')  # load a custom model

results = model(
    '/home/golu/PycharmProjects/weatherImageClassificationYOLOv8/weather/weatherDataset/test/rain/rain182.jpg')

name_dict = results[0].names
probs = results[0].probs.tolist()
print(name_dict)
print(probs)
print(name_dict[np.argmax(probs)])  # we are taking the maximum probablity of which the model thinks is the value
