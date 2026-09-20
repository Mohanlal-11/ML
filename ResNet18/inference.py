import torch

import os
import cv2
from pathlib import Path
import numpy as np
import time
import argparse

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

def calOperation(model, input_dims, device):
    import thop

    x = torch.rand(input_dims).to(device)
    macs, params = thop.profile(model, inputs=(x,))
    print(f"MACs: {macs}")
    print(f"GFLOPs: {(macs * 2) / 1e9}")
    print(f"Parameters: {params/1e6} M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments required for the resnet inference.')
    parser.add_argument('--weights', type=str, default='store/weights/resnet18_CIFAR10.pt', help='Path to model weights file (.pth).')
    parser.add_argument('--images', type=str, default="store/cifar10_images", help='path to input images directory')
    parser.add_argument('--save_dir', type=str, default='results', help='name of directory to save results')
    parser.add_argument('--no_cuda', action='store_true', help='to use cpu')

    args = parser.parse_args()

    input_img_dir = args.images
    wts = args.weights

    save_dir = Path('store/INFERENCE')
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = save_dir/args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    model = ResNet18(num_classes=len(classes))
    model.load_state_dict(torch.load(wts, map_location=torch.device('cpu')), strict=True)

    image_files = [os.path.join(input_img_dir, f) for f in os.listdir(input_img_dir) if f.endswith('.jpg')]
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Device : {device}')

    model.to(device).eval()
    calOperation(model, (1,3,32,32), device)

    total_time = 0
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        pltimg = cv2.resize(img, (320,320))
        
        inp_img = torch.from_numpy(np.transpose(img, (2,0,1))).unsqueeze(0).to(device)
        with torch.no_grad():
            time1 = time.time()
            prediction = model(inp_img/255.0)
            model_time = (time.time()-time1)*1000
            if i!=0:
                total_time+=model_time
            # print(f'torch model prediction : {prediction}')
        probabilities = torch.softmax(prediction, dim=1)
        score, cls_idx = torch.max(probabilities, dim=1)
        score = round(score.item(), 4)

        cls_name = classes[cls_idx]
        # print(f'predicted class : {cls_name, score}, ground truth : {image_file.split("/")[-1].split("_")[0]}')

        pltimg = cv2.putText(pltimg, f'Pred class : {cls_name}', (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)   
        pltimg = cv2.putText(pltimg, f'Pred score: {cls_name} : {score}', (1, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
        pltimg = cv2.putText(pltimg, f'GT : {image_file.split("/")[-1].split("_")[0]}', (5, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
        cv2.imwrite(f'{save_dir}/{image_file.split("/")[-1]}', pltimg)
        print(f'[INFO] The result is saved as {save_dir}/{image_file.split("/")[-1]} with shape {pltimg.shape} so that we can see the result clearly.')

    avg_model_time = total_time/(len(image_files)-1)
    print(f'[INFO] Average inference time of model per image for {len(image_files)-1} images : {avg_model_time:.3f}ms.')