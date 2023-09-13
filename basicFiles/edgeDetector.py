import cv2 as cv
import os
import numpy as np

img = cv.imread(os.path.join(
    r'/home/golu/PycharmProjects/openCV/opencv-python-course-computer-vision-master/05_threshold/code/bear.jpg'))
cv.imshow('image', img)
# different edge detetcor works differnty in different images
k_size = 3
blur_img = cv.GaussianBlur(img, (k_size, k_size), 0)
img_canny = cv.Canny(blur_img, 200, 250)
cv.imshow('canny', img_canny)

cv.waitKey(0)
