import cv2 as cv
import os
import mediapipe as mp
import argparse


def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    # this .process gives all the data about the image that we are processing
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
            # cv.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)

            # blur face
            img[y1 : y1 + h, x1 : x1 + w, :] = cv.blur(
                img[y1 : y1 + h, x1 : x1 + w, :], (30, 30)
            )
            # cv.imshow("img", img)

    return img


args = argparse.ArgumentParser()
args.add_argument("--mode", default="webcam")
# this asks user what he wants to do , if user wants to use and image it will use and image and if user want ot us
# use an video we will specify video here

args.add_argument("--filePath", default=None)
args = args.parse_args()
output_path = os.path.join(r"/home/golu/PycharmProjects/openCV/faceAnamolyzer")

if not os.path.exists(output_path):
    os.makedirs(output_path)

# detect faces
mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:

    if args.mode in ["image"]:
        img = cv.imread(args.filePath)

        img = process_img(img, face_detection)

    elif args.mode in ["video"]:

        cap = cv.VideoCapture(args.filePath)
        ret, frame = cap.read()
        output_video = cv.VideoWriter(
            os.path.join(output_path, "output.mp4"),
            cv.VideoWriter_fourcc(*"MP4V"),
            25,
            (frame.shape[1], frame.shape[0]),
        )
        # cv.VideoWriter_fourcc(*"MP4V") this is the video codec that we are going to use , we have to enter the
        # path, codec, frames,width and height of video
        while ret:
            img = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ["webcam"]:
        cap = cv.VideoCapture(0)
        ret, frame = cap.read()

        while ret:
            frame = process_img(frame, face_detection)
            cv.imshow("frame", frame)
            cv.waitKey(25)

            ret, frame = cap.read()
        cap.release()
# save the image
cv.imwrite(os.path.join(output_path, "output.jpg"), img)

cv.waitKey(0)
