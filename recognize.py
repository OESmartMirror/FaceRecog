# USAGE
# python recognize.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy
import numpy as np
import argparse
import imutils
import pickle
import cv2
import pandas as pd

import path
import os

from PIL import Image




def convert_to_rgb(imagePath):
    image_to_convert = Image.open(imagePath)
    rgb_im = image_to_convert.convert('RGB')
    rgb_im.save(imagePath)


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

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

old_embeddings_file = open('output/embeddings.pickle', 'rb')
old_embeddings = pickle.load(old_embeddings_file)

new_embedding_file = open('output/embeddings2.pickle', 'rb')
new_embeddings = pickle.load(new_embedding_file)

distances = []

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())


# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions

imagePaths = list(path.list_images(".\\Dataset\\"))

for (idx, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(idx + 1, len(imagePaths)))
    pic_name = imagePath.split(os.path.sep)[-1]
    convert_to_rgb(imagePath)
    #img = Image.open(imagePath)
    img = cv2.imread(imagePath)
    # image = cv2.imread(args["image"])
    # #image = imutils.resize(image, width=600)
    image = imutils.resize(img, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face

            for item in new_embeddings:
                names = [item[0], pic_name]
                distance = numpy.linalg.norm(item[1] - vec)
                distances.append([names, distance])


            #min_dist = distances[0]

            #for item in distances:
            #    if item[1] < min_dist[1]:
            #         min_dist = item

            #print(min_dist)


            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated
            # probability
            text: str = "{}: {:.2f}% {}".format(name, proba * 100, pic_name)
            print(text)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # show the output image
            cv2.imshow("Image", image)
            cv2.waitKey(0)

print(distances)
filname = 'old_version_test_output.csv'
file = open(filname, 'w+')
df_distances = []

for item in distances:
    name1 = item[0][0]
    name2 = item[0][1]
    distance = item[1]
    df_distances.append([name1,name2,distance])

dataframe = pd.DataFrame(df_distances)
df_sorted = dataframe.sort_values(2)
df_sorted = np.round(df_sorted, decimals=3)
df_sorted.to_csv(filname, sep=';', float_format=None, header=False, index=False)
