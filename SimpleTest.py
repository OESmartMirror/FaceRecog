import torch
import path
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


def return_closest_tensor(collection):
    if not collection:
        raise ValueError('empty input')

    min_dist = collection[0]

    for item in collection:
        if item[1] < min_dist[1]:
            min_dist = item
    return min_dist


def convert_to_rgb(imagePath):
    image_to_convert = Image.open(imagePath)
    rgb_im = image_to_convert.convert('RGB')
    rgb_im.save(imagePath)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

imagePaths = list(path.list_images(".\\dataset\\"))
datapath = './Eval/test_1/test1.jpg'

embeddings = []
cropped_images = []
person = []
names = []
eval_ims = []
distances = []

for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    # print("[INFO] processing image {}/{}".format(i + 1,
    #   len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    convert_to_rgb(imagePath)

    img = Image.open(imagePath)
    names.append(name)
    # perform face detection on the image
    cropped_images.append(mtcnn(img))


img_cropped = torch.stack(cropped_images).to(device)
embeddings = resnet(img_cropped).detach().cpu()


for i in range(14):
    person.append([names[i], embeddings[i]])


convert_to_rgb(datapath)
img_eval = Image.open(datapath)
eval_ims.append(mtcnn(img_eval))
pre_recog = torch.stack(eval_ims).to(device)
embedding = resnet(pre_recog).detach().cpu()


for item in person:
    distances.append([item[0], (item[1] - embedding[0]).norm().item()])

result = return_closest_tensor(distances)
probability = (1 - (result[1] / 4)) * 100
print("result: \n " + result[0] + ' ' + str(probability) + "%")
