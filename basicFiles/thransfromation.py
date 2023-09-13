import cv2 as cv
import os
import numpy as np

img = cv.imread(os.path.join(
    r'/home/golu/PycharmProjects/openCV/opencv-python-course-computer-vision-master/05_threshold/code/bear.jpg'))
cv.imshow('image', img)
k_size = 3
blur_img = cv.GaussianBlur(img, (k_size, k_size), 0)
img_canny = cv.Canny(blur_img, 200, 250)
cv.imshow('canny', img_canny)

img_dilate = cv.dilate(img_canny, np.ones((3, 3), dtype=np.int8))
cv.imshow('dilate', img_dilate)
# dilating looks like we are making everything ot loo thicker , we use to make the  all the white images or corners to dilate

img_erode = cv.erode(img_canny, np.ones((3, 3), dtype=np.int8))
cv.imshow('erode imag',img_erode)
#erode is the ope
cv.waitKey(0)
