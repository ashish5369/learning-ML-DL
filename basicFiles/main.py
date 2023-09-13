import cv2
import cv2 as cv
import os

#image path
img_path=os.path.join(r'/home/golu/PycharmProjects/openCV/pexels-pixabay-33787.jpg')

#read image
img=cv.imread(img_path)
cv.imshow('img',img)

#to write image

cv.imwrite(r'/home/golu/PycharmProjects/openCV/output.jpg',img)
cv.waitKey(0)#this number is the time for which the

#import video
video_path=os.path.join(r'')

video=cv.VideoCapture(video_path)
#visualize the video

ret = True
while ret:
    ret,frame=video.read()
    if ret:
        cv.imshow('frame',frame)
    #so it will read the video as frames and till there are frames the ret is going to be true and moment when there are no frames left the value of ret is going to be false
        cv2.waitKey(40) # as we have 25fps and which is 40ms so we are asking the him to wait 40 after every frame

video.release()
cv.destroyAllWindows()








