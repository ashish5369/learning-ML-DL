import cv2 as cv
import os

img_path = os.path.join(r'/home/golu/PycharmProjects/openCV/pexels-pixabay-33787.jpg')
img = cv.imread(img_path)

img = cv.resize(img, (500, 500))
cv.imshow("image", img)
print(img.shape)

cropped_img = img[200:400, 200:400]  # here we are specifyong the interval of the heights and the widths
cv.imshow("cropped_img", cropped_img)
print(img.shape)

cv.waitKey(0)
