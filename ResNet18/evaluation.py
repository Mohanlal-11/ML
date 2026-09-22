import torch

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from architecture import ResNet18

classes = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments required for the resnet inference.')
    parser.add_argument('--weights', type=str, default='store/weights/resnet18_CIFAR10.pt', help='Path to model weights file (.pth).')
    parser.add_argument('--images', type=str, default="store/cifar10_images", help='path to input images directory')

    args = parser.parse_args()

    input_img_dir = args.images
    wts = args.weights

    model = ResNet18(num_classes=len(classes))
    model.load_state_dict(torch.load(wts, map_location=torch.device('cpu')), strict=True)
    model.eval()
    image_files = [os.path.join(input_img_dir, f) for f in os.listdir(input_img_dir) if f.endswith('.jpg')]

    correct = 0
    for image_file in tqdm(image_files, desc='Evaluation'):
        img = cv2.imread(image_file)
        
        inp_img = torch.from_numpy(np.transpose(img, (2,0,1))).unsqueeze(0)
        with torch.no_grad():
            prediction = model(inp_img/255.0)
            # print(f'torch model prediction : {prediction}')
        probabilities = torch.softmax(prediction, dim=1)
        score, cls_idx = torch.max(probabilities, dim=1)
        score = round(score.item(), 4)

        label_name = image_file.split("/")[-1].split("_")[0]
        # print(f'label name : {label_name}')
        labels = torch.tensor([classes.index(label_name)])
        correct += cls_idx.eq(labels).sum().item()
        cls_name = classes[cls_idx]
        # print(f'predicted class : {cls_name, score}, ground truth : {image_file.split("/")[-1][:-6]}')

    accuracy = (correct/len(image_files))* 100
    print('#'*80)
    print(f'[INFO] Accuracy : {accuracy:.2f}%')
    print('#'*80)