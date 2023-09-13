import cv2 as cv
import os

img_path = os.path.join(r'/home/golu/PycharmProjects/openCV/pexels-pixabay-33787.jpg')

img = cv.imread(img_path)

print(img.shape)
cv.imshow("image", img)
resiezed_img = cv.resize(img, (640, 640))

cv.imshow("resized image", resiezed_img)
print(resiezed_img.shape)

cv.waitKey(0)
