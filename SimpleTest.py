import io
from base64 import b64encode
import torch
import path
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from imutils.video import VideoStream
import imutils
import cv2
import time
import eel
import pyqrcode
import random
import json
import pickle


def write_collection_to_pickle(collection, filename):
    if not collection:
        raise ValueError('[ERROR] write_collection_to_pickle : empty input')

    with open(filename, 'wb',) as file:
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

        x = mtcnn(img)
        if x is not None:
            names.append(name)
            cropped_images.append(x)

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


def calculate_distances(reference, evaluation):
    distances = []
    for reference_item in reference:
        for eval_item in evaluation:
            eval_item_tensor = eval_item[1]
            names = ([reference_item[0], eval_item[0]])
            distance = (reference_item[1] - eval_item_tensor).norm().item()
            distances.append([names, distance, eval_item_tensor])
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


def get_image_from_camera(vs):
    frame = vs.read()
    return frame


def write_to_json(collection):
    if not collection:
        raise ValueError('empty input')

    with open('dataset.json', 'w', encoding='utf-8') as file:
        json.dump(collection, file, ensure_ascii=False, indent=4)


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

@eel.expose
def generate_setup_qr_code():
    img = pyqrcode.create('https://SmartMirror.net/setup')
    buffers = io.BytesIO()
    img.png(buffers, scale=8)
    encoded = b64encode(buffers.getvalue()).decode("ascii")
    return "data:image/png;base64, " + encoded


@eel.expose
def loop_recog_for(num_of_frames):
    eval_data.clear()
    success = False
    loopcounter = 0
    while not success:

        for i in range(num_of_frames):
            frame = get_image_from_camera(vs)
            tensor = extract_tensor_from_image(frame)
            eval_data.append(['_', tensor])

        distances_between_people = calculate_distances(reference_data, eval_data)

        result = get_min_distance(distances_between_people)
        recognized_name = result[0][0]
        recognized_tensor = result[2]
        conf = round((1 - (result[1] / 2)), 2) * 100
        if conf < 60:

            loopcounter += 1
            eel.sleep(0.5)

            if loopcounter == 5:
                print(f'[INFO] No user seen')
                return -1
        else:
            print(f'[RESULT]  "{recognized_name}"  {conf}% confidence')
            # print(f'[RESULT]  "{result[0][0]}"  {result[1]}')
            success = True
            if 72 < conf < 90 and random.randint(1, 3) == 2:
                reference_data.append([recognized_name, recognized_tensor])
                print("[INFO] Face added to references")
            return [recognized_name]

vs = VideoStream(src=0).start()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('[INFO] Running on device: {}'.format(device))

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
              device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


dataset_sate = []
reference_data = []
eval_data = []
pickled_dataset_path = './dataset.pickle'
pickled_data_state_path = 'data_state.pickle'
dataset_paths = list(path.list_images(".\\Dataset\\"))
eval_paths = list(path.list_images(".\\Eval\\"))

if os.path.isfile(pickled_data_state_path):
    data_sate = read_collection_from_pickle(pickled_data_state_path)
else:
    data_sate = dataset_paths


def process_dataset_images():
    global new_dataset_paths, reference_data, data_sate
    convert_images_to_RGB_jpeg(dataset_paths, 'dataset')
    new_dataset_paths = list(path.list_images(".\\Dataset\\"))
    reference_data = extract_embeddings(new_dataset_paths)
    data_sate = new_dataset_paths


if os.path.isfile(pickled_dataset_path):

    if not data_sate == dataset_paths:

        process_dataset_images()
        write_collection_to_pickle(reference_data, pickled_dataset_path)

    write_collection_to_pickle(data_sate, pickled_data_state_path)
    reference_data = read_collection_from_pickle(pickled_dataset_path)

else:
    process_dataset_images()

    write_collection_to_pickle(reference_data, pickled_dataset_path)
    write_collection_to_pickle(data_sate, pickled_data_state_path)


#convert_images_to_RGB_jpeg(dataset_paths)
#convert_images_to_RGB_jpeg(eval_paths)

#new_dataset_paths = list(path.list_images(".\\Dataset\\"))
#new_eval_paths = list(path.list_images(".\\Eval\\"))







#eval_data = extract_embeddings(new_eval_paths)

#frame = get_image_from_camera()
#embedding = extract_tensor_from_image(frame)
#for row in distances_between_people:
#    name1 = row[0][0]
#    name2 = row[0][1]
#    dist = row[1]
#    print(f'{name1};{name2};{dist}')
print('starting eel')
eel.init('web')
eel.start('main.html')

while True:
    result = loop_recog_for(3)
    eel.sleep(3)