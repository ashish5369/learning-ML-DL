import os
import cv2 as cv

img_path = os.path.join(r'/home/golu/PycharmProjects/openCV/pexels-pixabay-33787.jpg')

img = cv.imread(img_path)
img = cv.resize(img, (500, 500))

new_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('images', img)
cv.imshow('image2', new_img)
# converting to grayscale is one of the most useful
cv.waitKey(0)
