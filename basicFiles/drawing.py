import cv2 as cv
import os

img_path = os.path.join(
    r'/home/golu/PycharmProjects/openCV/opencv-python-course-computer-vision-master/07_drawing/code/whiteboard.png')
img = cv.imread(img_path)
print(img.shape)
# line
cv.line(img, (100, 150), (300, 450), (0, 255, 0), 3)

# rectangle
cv.rectangle(img, (200, 350), (450, 600), (0, 0, 255), -1)
# for thickness the minus means we'll fill the rectangle

# circle
cv.circle(img, (800, 200), 75, (255, 0, 0), 5)

# text
cv.putText(img, "hey guys this is shapes", (392, 323), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

cv.imshow('img', img)
cv.waitKey(0)
