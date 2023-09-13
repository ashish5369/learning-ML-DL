import cv2 as cv
import os

img = cv.imread(os.pa+th.join(
    r'/home/golu/PycharmProjects/openCV/opencv-python-course-computer-vision-master/08_contours/code/birds.jpg'))
cv.imshow("img", img)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, threshImg = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV)
# this will take anything lower thn 127 to 255 and above 127 to 0
cv.imshow('threshold img', threshImg)
# we weork wiht threshold images in contours, here we absolutely everything n black and white and whatever we ant
# todetct we have to make that objetct in white we need obecjts to be detected white in contours thus we use inverse
# threshold
contours, hierarchy = cv.findContours(threshImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# in contours we need the image to be one dimenssional so we have to use a gray image for finding the threshold and
# that thresh image will be used in finding the contours we'll have contours for all the white objects inthe image
# and we are going to have thid ontours for all the white objects and if we have 2 3 birds connected and it sieems to
# be one object so we iwll have a single contour for all the 3 birsds as this will seem like a single contour
# CONTOURS FOR ISOLATED WHITE REGIONS
for cnt in contours:
    if cv.contourArea(cnt) > 200:
        cv.drawContours(img, cnt, -1, (0, 255, 0), 1)

        # the contour is drawn above thewhite images and we are drawing on the top of the original image only and here we are
        # drawing the contours and i t loooks like a border to all thw white images which are like the obejct of the image,
        # that we actually want to detect we are only taking the bigger contours not the smaller ones as they are not so
        # singnificant so contours only greater than 200 will be considered
        x1, y1, w, h = cv.boundingRect(cnt)
        cv.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        # and this way we have found the bounding rect using the contour adn that bounding rect is what we are seeing
        # in the image and this acts like a object detector,so a imagel liket htis so simple

# cv.imshow("contours", contours)
cv.imshow("image", img)

cv.waitKey(0)
