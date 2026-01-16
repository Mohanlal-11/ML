import pytorch_nndct 
from pytorch_nndct.apis import torch_quantizer
import torch
from torch.utils import data
from torchvision import transforms

from tqdm import tqdm
import argparse 
import os
import sys
import yaml
from pathlib import Path
import cv2

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import yolo
from yolo.model.transform import Transformer

@torch.no_grad()
def test(args, model, d_test, device):
    # -------------------------------------------------------------------------- #
    print("evaluating...")
    eval_output, iter_eval = yolo.evaluate(model, d_test, device, args, evaluation=True)
    print(f'[INFO] Evaluation result : \n {eval_output}')
    return eval_output


def main():
    parser = argparse.ArgumentParser(description="Input required arguments for the quantization, testing and deployment of yolov3 model.")
    parser.add_argument('-cb', '--batch_size', type=int, default=8, help="Argument for the batch size in calibration dataloader.")
    parser.add_argument('-w', '--weight', type=str, default="weights/yolov5_nano-1000.pth", help="Argument for the pretrained weight of yolo model in '.pt' format.")
    parser.add_argument('-m', '--mode', type=str, default="float", help="Argument for the different required during quantization i.e. 'calib', 'test'.")
    parser.add_argument('-d', '--device', type=str, default='gpu', help="Argument for the device used during quantization and deployment.")
    parser.add_argument('-s', '--input_size', nargs="+", type=int, default=[320, 320], help="Argument for the input size to model.")
    parser.add_argument("--score_thres", type=float, default=0.1)
    parser.add_argument("--nms_thres", type=float, default=0.6)
    parser.add_argument('-f', '--result_dir', type=str, default="Deployment/quantized_result_nano", help="Argument for the directory where the results of quantization will be save.")
    parser.add_argument('-i', '--deploy', action='store_true')
    parser.add_argument("--calib_data", type=str)   
    parser.add_argument("--fuse", action="store_true")
    
    arg = parser.parse_args()

    arg.dataset = "coco"
    arg.file_root = f"/{arg.calib_data}/images/train2017"
    arg.ann_file = f"/{arg.calib_data}/annotations/instances_train2017_PersonOnly_16.json"
    arg.results = os.path.join(os.path.dirname(arg.weight), "results.json")
    
    arg.iters = -1
    
    arg.model_size = "nano"
    arg.kwargs = {"img_sizes": arg.input_size}

    if arg.device == "gpu":
        device = torch.device('cuda')
    elif arg.device == 'cpu':
        device = torch.device('cpu')
        
    if arg.deploy:
        arg.batch_size = 1

    image_resizer = transforms.Resize((arg.input_size, arg.input_size))

    cuda = device == 'cuda'
    arg.amp = False
    if cuda and torch.__version__ >= "1.6.0":
        capability = torch.cuda.get_device_capability()[0]
        if capability >= 7: # 7 refers to RTX series GPUs
            arg.amp = True
            print("Automatic mixed precision (AMP) is enabled!")

    dataset_test = yolo.datasets(arg.dataset, arg.file_root, arg.ann_file, train=True) # set train=True for eval
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_test = yolo.GroupedBatchSampler(
        sampler_test, dataset_test.aspect_ratios, arg.batch_size)
    
    arg.num_workers = 0
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_sampler=batch_sampler_test, num_workers=arg.num_workers,  
        collate_fn=yolo.collate_wrapper, pin_memory=True)

    d_test = yolo.DataPrefetcher(data_loader_test) if cuda else data_loader_test

    yolo.setup_seed(3)
    
    model_sizes = {"nano":(0.33, 0.25), "small": (0.33, 0.5), "medium": (0.67, 0.75), "large": (1, 1), "extreme": (1.33, 1.25)}
    num_classes = len(d_test.dataset.classes)
    model = yolo.YOLOv5(num_classes, model_sizes[arg.model_size], **arg.kwargs).to(device)
    model.head.eval_with_loss = False
    
    checkpoint = torch.load(arg.weight, map_location=device)

    strides = (8,16,32)
    transformer = Transformer(arg.input_size[0], arg.input_size[1], stride=max(strides))

    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"][0])
        print(checkpoint["eval_info"])
    else:
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in checkpoint.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")

        print(f'[INFO] The len of wts keys in filtered wts and own model is : {len(filtered_state_dict.keys()), len(model_state_dict.keys())}')
        model.load_state_dict(filtered_state_dict, strict=False)
        # model.load_state_dict(checkpoint)

    if arg.fuse:
        model = model.fuse()
    
    quant_model = model
    if arg.mode == 'float':
       test(arg, model=model, d_test=d_test, device=device)

    else:
        # input = torch.rand(1,3,640, 640)
        for i, data in enumerate(d_test):
            # print(input.shape, input.dtype)
            dataLoader_imgs = data.images
            dataLoader_targets = data.targets
            input,_,_,_,_ = transformer(dataLoader_imgs, dataLoader_targets)
            quantizer = torch_quantizer(arg.mode, model, (input), output_dir=arg.result_dir, device=device)
            quant_model = quantizer.quant_model
            if arg.batch_size == 1:
                break

        test(arg, quant_model, d_test, device)

    if arg.mode == 'calib':
        quantizer.export_quant_config()
        
    if arg.mode == 'test' and arg.deploy:
        quantizer.export_xmodel(arg.result_dir, deploy_check=True)
        
        
if __name__ == "__main__":
    main() 
            
        