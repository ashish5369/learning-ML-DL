import cv2 as cv
from PIL import Image
from util import get_limits

# install pillow as this will help us getting the boudnging box
yellow = (0, 255, 255)  # value of yellow in BGR colorspace
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # generally we use BGR colorspace , and we have each pixel as a combination if blue-green and red but here we are
    # going to use HSV colorSpace --hue saturation value we are mostly going to work wih hue channel and here we are
    # going to have the most of the information of the color but we cannot have a single color in our programm as
    # this is nt right as we can have yellow of Multiple hades and variances that is we need to have a range of colors

    cv.imshow('frame', frame)
    hsvImage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=yellow)
    mask = cv.inRange(hsvImage, lowerLimit, upperLimit)
    # here we are going to give an image adn color range and this is goign to return us the
    # co-ordinates of that color in that range
    # cv.imshow('mask', mask)
    mask_ = Image.fromarray(mask)
    # converting the array as our image is an array in open cv we are converting it to a image
    bbox = mask_.getbbox()  # this function gets the bounding box and this is a feature of pillow
    if bbox is not None:
        x1, y1, w, h = bbox
        cv.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 5)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
