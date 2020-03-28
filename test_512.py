import io
from base64 import b64encode
import torch
import pandas as pd
import path
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import time
import random
import pickle
import numpy as np


def write_collection_to_pickle(collection, filename):
    if not collection:
        raise ValueError('[ERROR] write_collection_to_pickle : empty input')

    with open(filename, 'wb', ) as file:
        pickle.dump(collection, file, protocol=4, fix_imports=True)


def read_collection_from_pickle(filename):
    if not filename:
        print("[ERROR] read_collection_from_pickle : no input")
    print(f'[INFO] Deserializing file: {filename}')
    infile = open(filename, 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict


def print_perf(string, start, finish):
    print(f'{string} : took {round((finish - start), 2)} seconds')


def get_min_distance(collection):
    if not collection:
        raise ValueError('empty input')

    min_dist = collection[0]

    for item in collection:
        if item[1] < min_dist[1]:
            min_dist = item
    return min_dist


def get_min_distance_of(collection):
    min_dist = collection[0]

    for item in collection:

        if item[1] < min_dist[1]:
            min_dist = item
    return min_dist


def sort_based_on_distance(collection):
    if not collection:
        raise ValueError('empty input')
    sorted = collection.sort(key=lambda x: x[1])
    return sorted


def convert_to_rgb_jpg(imagePaths):
    start = time.perf_counter()
    for (i, imagePath) in enumerate(imagePaths):
        print(f'[INFO] processing image {i + 1}/{len(imagePaths)}')

        image_to_convert = Image.open(imagePath)
        directory = imagePath.split(os.path.sep)[2]
        processed = imagePath.split(os.path.sep)[-1]

        checkpath = os.path.join(".", "Processed", directory, )
        if not os.path.exists(checkpath):
            os.makedirs(checkpath)

        new_file = os.path.join(".", "Processed", directory, processed.split('.')[0] + ".jpg")
        rgb_im = image_to_convert.convert('RGB')
        rgb_im.save(new_file)
        # print("file processed and saved as" + new_file)
    finish = time.perf_counter()
    print_perf("[PERF] Processing reference images", start, finish)


def extract_embeddings(local_processed_path):
    local_embeddings = []
    persons = []
    cropped_images = []
    names = []

    for (i, imagePath) in enumerate(local_processed_path):
        # extract the person name from the image path
        # print("[INFO] processing image {}/{}".format(i + 1, len(local_processed_path)))
        name = imagePath.split(os.path.sep)[-2]

        img = Image.open(imagePath)

        # perform face detection on the image
        face_start = time.perf_counter()

        try:
            x = mtcnn(img)
            if x is not None:
                names.append(name)
                cropped_images.append(x)
        except:
            print(f"something went wrong at image {i}, path: {imagePath}")

        face_finish = time.perf_counter()
        # print_perf(f'[PERF] Face detection - {name}', face_start, face_finish)

    extract_start = time.perf_counter()

    # enqueue the detected faces for tensor extraction
    img_cropped = torch.stack(cropped_images).to(device)

    # calculate tensors from faces
    local_embeddings = resnet(img_cropped).detach().cpu()

    for (i, imagePath) in enumerate(local_embeddings):
        persons.append([names[i], local_embeddings[i]])

    extract_finish = time.perf_counter()
    print_perf("[PERF] Embedding extraction", extract_start, extract_finish)

    return persons


def extract_eval_embeddings(local_processed_path):
    local_embeddings = []
    persons = []
    cropped_images = []
    names = []
    extract_start = time.perf_counter()
    for (i, imagePath) in enumerate(local_processed_path):
        # extract the person name from the image path
        # print("[INFO] processing image {}/{}".format(i + 1, len(local_processed_path)))
        name = imagePath.split(os.path.sep)[-2]
        pic_name = imagePath.split(os.path.sep)[-1]
        img = Image.open(imagePath)

        # perform face detection on the image
        face_start = time.perf_counter()

        x = mtcnn(img)
        if x is not None:
            names.append(pic_name)
            cropped_images.append(x)

        face_finish = time.perf_counter()
        # print_perf(f'[PERF] Face detection - {name}', face_start, face_finish)



    # enqueue the detected faces for tensor extraction
    img_cropped = torch.stack(cropped_images).to(device)

    # calculate tensors from faces
    local_embeddings = resnet(img_cropped).detach().cpu()

    for (i, imagePath) in enumerate(local_embeddings):
        persons.append([names[i], local_embeddings[i]])

    extract_finish = time.perf_counter()
    print_perf("[PERF] Processing Eval images", extract_start, extract_finish)

    return persons


def calculate_distances(reference, evaluation):
    distances = []
    for reference_item in reference:
        for eval_item in evaluation:
            eval_item_tensor = eval_item[1]
            names = ([reference_item[0], eval_item[0]])
            distance = (reference_item[1] - eval_item_tensor).norm().item()
            distances.append([names, distance, eval_item_tensor])
    return distances


def calculate_distances_2(reference, evaluation):
    distances = []
    for reference_item in reference:
        for eval_item in evaluation:
            eval_item_tensor = eval_item[1]
            name1 = reference_item[0]
            name2 = eval_item[0]
            distance = (reference_item[1] - eval_item_tensor).norm().item()
            distances.append([name1, name2, distance])
    return distances


def extract_tensor_from_file(file):
    eval_ims = []
    img_eval = Image.open(file)
    eval_ims.append(mtcnn(img_eval))
    container = torch.stack(eval_ims).to(device)
    tensor = resnet(container).detach().cpu()
    return tensor


def extract_tensor_from_image(img):
    x = mtcnn(img)
    if x is not None:
        container = [x]
        stacked = torch.stack(container).to(device)
        tensor = resnet(stacked).detach().cpu()
        return tensor
    else:
        return -1


def convert_images_to_RGB_jpeg(path, type):
    conv_eval_start = time.perf_counter()
    for (i, file_ref_path) in enumerate(path):
        old_file_path = file_ref_path
        image_to_convert = Image.open(old_file_path)
        rgb_im = image_to_convert.convert('RGB')
        new_file_path = file_ref_path.replace("png", "jpg")
        rgb_im.save(new_file_path)
        if not old_file_path == new_file_path:
            os.remove(file_ref_path)
    conv_eval_finish = time.perf_counter()
    print_perf(f'[PERF] Processing {type} images', conv_eval_start, conv_eval_finish)


def process_dataset_images():
    start_dataset = time.perf_counter()
    global new_dataset_paths, reference_data, data_sate
    convert_images_to_RGB_jpeg(dataset_paths, 'dataset')
    new_dataset_paths = list(path.list_images(".\\Dataset\\"))
    reference_data = extract_embeddings(new_dataset_paths)
    data_sate = new_dataset_paths
    end_dataset = time.perf_counter()
    print_perf('[PERF] Processing Dataset images', start_dataset, end_dataset)


def process_eval_images():
    start_eval = time.perf_counter()
    global new_eval_paths, eval_data, eval_sate
    convert_images_to_RGB_jpeg(eval_paths, 'evaluation')
    new_eval_paths = list(path.list_images(".\\Eval\\"))
    eval_data = extract_embeddings(new_eval_paths)
    eval_sate = new_eval_paths
    end_eval = time.perf_counter()
    print_perf('[PERF] Processing Eval images', start_eval, end_eval)


def execute_func_with_probability(func, probability):
    if not 0 < probability <= 100:
        print("incorrect probability given")
    else:
        if random.randint(0, 100) <= probability:
            func()



def save_reference_embeddings():
    write_collection_to_pickle(reference_data, pickled_dataset_path)


def pre_flight_check():
    global data_sate, eval_state, eval_data, reference_data
    # if there's no saved data from previous runs, initialize data to current folder structure
    if os.path.isfile(pickled_data_state_path):
        data_sate = read_collection_from_pickle(pickled_data_state_path)
    else:
        data_sate = dataset_paths
    if os.path.isfile(pickled_eval_state_path):
        eval_state = read_collection_from_pickle(pickled_eval_state_path)
    else:
        eval_state = eval_paths
    if os.path.isfile(pickled_eval_path):

        if not eval_state == eval_paths:
            # if changes were made to the images, re-process them
            process_eval_images()
            write_collection_to_pickle(eval_data, pickled_eval_path)

        # Save the current state of images
        write_collection_to_pickle(eval_state, pickled_eval_state_path)

        # Fill operational data from previous saved state
        eval_data = read_collection_from_pickle(pickled_eval_path)
    else:
        # In the case of missing pickle file, process images and create a new file to speed up consecutive runs
        process_eval_images()
        write_collection_to_pickle(eval_data, pickled_eval_path)
        write_collection_to_pickle(eval_state, pickled_eval_state_path)
    if os.path.isfile(pickled_dataset_path):

        if not data_sate == dataset_paths:
            process_dataset_images()
            write_collection_to_pickle(reference_data, pickled_dataset_path)

        write_collection_to_pickle(data_sate, pickled_data_state_path)
        reference_data = read_collection_from_pickle(pickled_dataset_path)

    else:
        # In the case of missing pickle file, process images and create a new file to speed up consecutive runs
        process_dataset_images()
        write_collection_to_pickle(reference_data, pickled_dataset_path)
        write_collection_to_pickle(data_sate, pickled_data_state_path)


n_to_n_eval_mode = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize the Face Detection CNN
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
              device=device)
# Initialize the Face Recognition CNN
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# container for storing the list containing all the path strings in ./Dataset/*
dataset_sate = []
# Same as above, but for the ./Eval/* directory
eval_state = []

# container for storing the [label, Tensor] tuple calculated from images in the Dataset directory
reference_data = []
# same as above, but for the Eval directory
eval_data = []

# baked in path strings for serialized data
pickled_dataset_path = './dataset.pickle'
pickled_data_state_path = './data_state.pickle'
pickled_eval_path = './eval.pickle'
pickled_eval_state_path = './eval_state.pickle'

# fill appropriate containers with current images in specified directories
dataset_paths = list(path.list_images(".\\Dataset\\"))
eval_paths = list(path.list_images(".\\Eval\\"))

# check states from previous runs, and initialize operative data accordingly
pre_flight_check()


filename = 'test_512_output.csv'

start_all = time.perf_counter()
if n_to_n_eval_mode:
    file = open(filename, 'w+')
    distances_between_people = calculate_distances(reference_data, reference_data)
    for row in distances_between_people:
        name1 = row[0]
        name2 = row[1]
        dist = row[2]
        file.write(f'{name1};{name2};{dist}\n')
    file.close()
else:
    # process_eval_images()
    distances_between_people = calculate_distances_2(reference_data, eval_data)
    dataframe = pd.DataFrame(distances_between_people)
    df_sorted = dataframe.sort_values(2)
    df_sorted = np.round(df_sorted, decimals=3)


    print(f'[INFO] Writing to file: {filename}')
    start = time.perf_counter()
    df_sorted.to_csv(filename, sep=';', float_format=None, header=False, index=False)
    end = time.perf_counter()
    print_perf("[PERF] Writing to file", start, end)

total = len(eval_data) + len(reference_data)
end_all = time.perf_counter()
print_perf(f'[PERF] Execution - {total}', start_all, end_all)
