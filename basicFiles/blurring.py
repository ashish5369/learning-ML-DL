import cv2 as cv
import os

img_path = os.path.join(r'/home/golu/PycharmProjects/openCV/pexels-pixabay-33787.jpg')

img = cv.imread(img_path)
k_size = 7
# this defines the value of the nearnby blocks so we can select them according to our need, it signifies the bigger the area we are going to take to calculate the blur

blur_img = cv.blur(img, (k_size, k_size))
cv.imshow('blur', blur_img)
cv.waitKey(0)

######blur is used to make the images with noise , they can help in removing the noise from the images , adn depending on the type of noise we can apply
#different type of blur to make it better and remove oise
