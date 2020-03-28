import time
import numpy
import pandas as pd
import path
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


def print_perf(string, start, finish):
    print(f'{string} : took {round((finish - start), 2)} seconds')


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", default='.\\dataset\\',
                help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", default='.\\output\\embeddings.pickle',
                help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", default='.\\face_detection_model\\',
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", default='.\\openface_nn4.small2.v1.t7',
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
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

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
datasetPaths = list(path.list_images(args["dataset"]))

# initialize our lists of extracted facial embeddings and
# corresponding people names
dataset_embeddings = []
dataset_known_names = []
dataset_collection = []
# initialize the total number of faces processed
total = 0

start_all = time.perf_counter()

start_dataset= time.perf_counter()
# loop over the image paths
for (i, imagePath) in enumerate(datasetPaths):
    # extract the person name from the image path
    #print("[INFO] processing image {}/{}".format(i + 1, len(datasetPaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # add the name of the person + corresponding face
            # embedding to their respective lists
            dataset_known_names.append(name)
            dataset_embeddings.append(vec.flatten())
            dataset_collection.append([name, vec.flatten()])
            total += 1

end_dataset = time.perf_counter()
print_perf(f'[PERF] Processing dataset images', start_dataset, end_dataset)
# dump the facial embeddings + names to disk
dataset_data = {"embeddings": dataset_embeddings, "names": dataset_known_names}

#print(dataset_collection)
file = open('output/dataset_embeddings.pickle', 'wb')
pickle.dump(dataset_collection, file)
file.close()

f = open(args["embeddings"], "wb")
f.write(pickle.dumps(dataset_data))
f.close()

eval_paths = list(path.list_images('.\\Eval\\'))
eval_embeddings = []
eval_known_names = []
eval_collection = []
start_eval = time.perf_counter()
for (i, imagePath) in enumerate(eval_paths):
    # extract the person name from the image path
    # print("[INFO] processing image {}/{}".format(i + 1, len(datasetPaths)))
    name = imagePath.split(os.path.sep)[-2]
    pic_name = imagePath.split(os.path.sep)[-1]
    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # add the name of the person + corresponding face
            # embedding to their respective lists
            eval_known_names.append(name)
            eval_embeddings.append(vec.flatten())
            eval_collection.append([pic_name, vec.flatten()])
            total += 1
end_eval = time.perf_counter()
print_perf(f'[PERF] Processing Eval images', start_eval, end_eval)
file = open('output/eval_embeddings.pickle', 'wb')
pickle.dump(eval_collection, file)
file.close()


dataset_embedding_file = open('output/dataset_embeddings.pickle', 'rb')
dataset_embeddings = pickle.load(dataset_embedding_file)

eval_embeddings_file = open('output/eval_embeddings.pickle', 'rb')
eval_embeddings = pickle.load(eval_embeddings_file)

distances = []

for dataset_item in dataset_embeddings:
    for eval_item in eval_embeddings:
        names = [dataset_item[0], eval_item[0]]
        distance = numpy.linalg.norm(dataset_item[1] - eval_item[1])
        distances.append([names, distance])

# print(distances)
filname = 'test_128_output.csv'
file = open(filname, 'w+')
df_distances = []

for item in distances:
    name1 = item[0][0]
    name2 = item[0][1]
    distance = item[1]
    df_distances.append([name1, name2, distance])

dataframe = pd.DataFrame(df_distances)
df_sorted = dataframe.sort_values(2)
df_sorted = np.round(df_sorted, decimals=3)
df_sorted.to_csv(filname, sep=';', float_format=None, header=False, index=False)
end_all = time.perf_counter()
print_perf(f'[PERF] Execution - {total} images', start_all, end_all)
