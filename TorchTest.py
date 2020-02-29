from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from imutils.video import VideoStream
import imutils
import time
import cv2
from PIL import Image

#vs = VideoStream(src=0).start()
#time.sleep(2.0)

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#resnet = InceptionResnetV1(pretrained='vggface2').eval()

def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('./dataset')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        #print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

#dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
#print(pd.DataFrame(dists, columns=names, index=names))
#print(dists)



file = open("torch_test_output.txt", 'w')
for i in embeddings:
    file.write(str(i))
    file.write(str("\n"))
file.close()


eval_dataset = datasets.ImageFolder('./Eval')
eval_dataset.idx_to_class = {i:c for c, i in eval_dataset.class_to_idx.items()}
eval_loader = DataLoader(eval_dataset, collate_fn=collate_fn, num_workers=workers)

eval_aligned = []
eval_names = []
for x, y in eval_loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        #print('Face detected with probability: {:8f}'.format(prob))
        eval_aligned.append(x_aligned)
        eval_names.append(dataset.idx_to_class[y])

eval_aligned = torch.stack(eval_aligned).to(device)
eval_embeddings = resnet(eval_aligned).detach().cpu()

#dists = [[[(e1 - e2).norm().item(), ]for e2 in eval_embeddings] for e1 in embeddings]
dists = []
for i in range(14):
    dists += [(embeddings[i] - eval_embeddings).norm().item(), i+1]
print(dists)
