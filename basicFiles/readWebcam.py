import cv2
import cv2 as cv

webcam = cv.VideoCapture(0)

while True:  # here we ar enot defining ret as for webcam the frames ar always true , we always have some frames
    ret, frame = webcam.read()

    cv.imshow("frame", frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):  # this statement means that we'll stop the capture when 'q' is pressed
        break

webcam.release()
cv.destroyAllWindows()
