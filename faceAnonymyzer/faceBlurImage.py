import cv2 as cv
import os
import mediapipe as mp

output_path = os.path.join(r"/home/golu/PycharmProjects/openCV/faceAnamolyzer")

if not os.path.exists(output_path):
    os.makedirs(output_path)
img = cv.imread(
    os.path.join(
        r"/home/golu/PycharmProjects/openCV/faceAnamolyzer/jake-nackos-IF9TK5Uy-KI-unsplash.jpg"
    )
)
print(img.shape)
img = cv.resize(img, (1000, 600))
# cv.imshow('image',img)
H, W, _ = img.shape
# detect faces
mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:
    # this model perfroms which are in range odf 2 to 5 metre ,model selection refers to distance the object is from
    # the camera
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(
        img_rgb
    )  # this .process gives all the data about the image that we are processing
    # print(out.detections)#prints all the detection
    # this prints and gives x y w and h which are like the bounding value of the face which this
    # model is capturing
    # and this wont detct a face of animal
    # print(out.detections)
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            cv.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)

            # blur face
            img[y1 : y1 + h, x1 : x1 + w, :] = cv.blur(
                img[y1 : y1 + h, x1 : x1 + w, :], (30, 30)
            )
            # cv.imshow("img", img)

cv.imshow("image", img)
# save the image
cv.imwrite(os.path.join(output_path, "output.jpg"), img)

cv.waitKey(0)
# read image
