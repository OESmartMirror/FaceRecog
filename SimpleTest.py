import torch
import path
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from imutils.video import VideoStream
import imutils
import cv2
import time


def print_perf(string, start, finish):
    print(f'{string} : took {round((finish - start), 2)} seconds')


def return_closest_tensor(collection):
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
        print("[INFO] processing image {}/{}".format(i + 1, len(local_processed_path)))
        name = imagePath.split(os.path.sep)[-2]

        img = Image.open(imagePath)
        names.append(name)
        # perform face detection on the image
        face_start = time.perf_counter()

        cropped_images.append(mtcnn(img))

        face_finish = time.perf_counter()
        # print_perf(f'[PERF] Face detection - {name}', face_start, face_finish)

    extract_start = time.perf_counter()

    # enqueue the detected faces for tensor extraction
    img_cropped = torch.stack(cropped_images).to(device)

    # calculate tensors from faces
    local_embeddings = resnet(img_cropped).detach().cpu()

    for (i, imagePath) in enumerate(local_processed_path):
        persons.append([names[i], local_embeddings[i]])

    extract_finish = time.perf_counter()
    print_perf("[PERF] Embedding extraction", extract_start, extract_finish)

    return persons


def calculate_distances(reference, evaluation):
    distances = []
    for reference_item in reference:
        for eval_item in evaluation:
            names = ([reference_item[0], eval_item[0]])
            distance = (reference_item[1] - eval_item[1]).norm().item()
            distances.append([names, distance])
    return distances


def extract_tensor_from_file(file):
    eval_ims = []
    img_eval = Image.open(file)
    eval_ims.append(mtcnn(img_eval))
    container = torch.stack(eval_ims).to(device)
    tensor = resnet(container).detach().cpu()
    return tensor


def extract_tensor_from_image(img):
    container = [mtcnn(img)]
    stacked = torch.stack(container).to(device)
    tensor = resnet(stacked).detach().cpu()
    return tensor


def get_image_from_camera():
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    frame = vs.read()
    return frame


def convert_images_to_RGB_jpeg(path):
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
    print_perf("[PERF] Processing evaluation images", conv_eval_start, conv_eval_finish)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('[INFO] Running on device: {}'.format(device))

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
              device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

dataset_paths = list(path.list_images(".\\Dataset\\"))
eval_paths = list(path.list_images(".\\Eval\\"))


convert_images_to_RGB_jpeg(dataset_paths)
convert_images_to_RGB_jpeg(eval_paths)

new_dataset_paths = list(path.list_images(".\\Dataset\\"))
new_eval_paths = list(path.list_images(".\\Eval\\"))


reference_data = extract_embeddings(new_dataset_paths)

eval_data = extract_embeddings(new_eval_paths)

if True:
    # frame = get_image_from_camera()
    # embedding = extract_tensor_from_image(frame)
    distances_between_people = calculate_distances(reference_data, eval_data)

    for row in distances_between_people:
        name1 = row[0][0]
        name2 = row[0][1]
        dist = row[1]
        print(f'{name1};{name2};{dist}')

    result = return_closest_tensor(distances_between_people)
    conf = round((1 - (result[1] / 2)), 2) * 100
    print(
        f'[RESULT] The smallest distance can be found between: "{result[0][0]}" and "{result[0][1]}" , with the distance of {round(result[1], 2)}, equating to {conf}% confidence in recognition')
