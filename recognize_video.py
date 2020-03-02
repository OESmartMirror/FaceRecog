# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import eel
import pyzbar.pyzbar as pyzbar

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", default='.\\face_detection_model\\',
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", default='.\\openface_nn4.small2.v1.t7',
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", default='.\\output\\recognizer.pickle',
                help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", default='.\\output\\le.pickle',
                help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# set up detection results array
detectionresults = []

# load serialized face detector from disk
# print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model from disk
# print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
print("sleeping for a sec[eel]")
time.sleep(1.0)


def Detect_face(frame):
    # grab the frame from the threaded video stream
    # print("detection started")
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            Recognize(face)


def Recognize(face):
    # construct a blob for the face ROI, then pass the blob
    # through our face embedding model to obtain the 128-d
    # quantification of the face
    # print("recognition started")
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    print(vec)
    # perform classification to recognize the face
    preds = recognizer.predict_proba(vec)[0]
    j = np.argmax(preds)
    proba = preds[j]
    name = le.classes_[j]
    add_to_output_buffer(name, proba)


def add_to_output_buffer(name, proba):
    # print('adding result to output buffer')
    detectionresults.append([name, proba])

def read_from_output_buffer():
    if len(detectionresults) == 0:
        return "No user seen"
    else:
        detectionresults.sort(key=lambda qi: qi[1])

        print(detectionresults[-1])
        return detectionresults[-1]


@eel.expose
def loop_recog_for(num_of_frames):
    detectionresults.clear()
    print("looping for " + str(num_of_frames) + " frames")
    for x in range(num_of_frames):
        frame = vs.read()
        Detect_face(frame)

    intermediate = read_from_output_buffer()
    if intermediate == "No user seen":
        return "No user seen"
    else:
        result = intermediate[0]
        return result


def read_qr_from(frame):
    QRCode = pyzbar.decode(frame)
    if QRCode is not []:
        return QRCode.data

@eel.expose
def perform_recognition_flow_for(num_of_frames):
    for _ in range(num_of_frames):
        frame = vs.read()

        qr_result = read_qr_from(frame)

        if qr_result is not []:
            return qr_result
        else:
            Detect_face(frame)

        intermediate = read_from_output_buffer()
        if intermediate == "No user seen":
            return "No user seen"
        else:
            result = intermediate[0]
            return result


@eel.expose
def mock_recog(num_of_frames):
    if num_of_frames > 1:
        print("it is recece!")
        return "recece"
    else:
        print("it's not recece")
        return "not recece"


print("starting eel")
eel.init('web')
eel.start('main.html')








# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
