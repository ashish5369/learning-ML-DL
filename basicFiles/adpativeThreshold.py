import cv2 as cv
import os

img_path = os.path.join(
    r'/home/golu/PycharmProjects/openCV/opencv-python-course-computer-vision-master/05_threshold/code/handwritten.png')
img = cv.imread(img_path)
img = cv.resize(img, (640, 640))
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img_gray, 80, 255, cv.THRESH_BINARY)
# this kind of image can be distincted from the background using simple or global threshold we need to use the
# adpative thresholding for this kind of image we can use multiple values of thresholding, for 80 it might give some
# result and for 60 it might give some better or differnt result but in reality we cannot use any one of, we ownt be
# able to get all the text , if we reduce it by certaint the lights are going to go away and out text wont be
# visible, and we cannot pass it to OCR as well them , no one is perfect , we have to use a combination of the
# threhold and for that purpose we need to use adaptive threshold

adapThresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 30)
# we'll have many small regions and we'll create multiple values of threshold, and we'll be using it to find the
# perfect text
cv.imshow('thresholdImage', thresh)
cv.imshow('adaptiveThresholdImage', adapThresh)
# we can clearly see that here we have all the text clearly, and we are able to read it and we can pass  it to ocr as
# well


# cv.imshow('Image', img_gray)
# cv.imshow('Image1', img)

cv.waitKey(0)
