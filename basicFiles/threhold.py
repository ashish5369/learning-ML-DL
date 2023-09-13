import cv2 as cv


import os

img_path = os.path.join(
    r'/home/golu/PycharmProjects/openCV/opencv-python-course-computer-vision-master/05_threshold/code/bear.jpg')
img = cv.imread(img_path)
img = cv.resize(img, (500, 500))
# the basic idea behind the use of threshold is to create a binary image

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.blur(img_gray, (10, 10))
ret, thresh = cv.threshold(blur, 70, 255, cv.THRESH_BINARY) #this is global threshold but this does not work in all the situations
# the pixel below 80 wll be taken to 0 and above 80 will be given 1 or 255 which means white
# there are different types of thresholding, and we are going to use the thresh binary  here in
# ret stores the threshold value
# print(ret)
cv.imshow('threshImg', thresh)

# threshold----used for semantic segmentation, one of the biggest use case is in case of separating the background
# from he foregrounds like in the image we have used we have made the background separate from the foreground and
# this is one of the test cases of the thresholding or the openCV is not the best method to separate
# background, but it gets the work done foe most of the situation

cv.waitKey(0)
