import pytorch_nndct 
from pytorch_nndct.apis import torch_quantizer
import torch

from tqdm.auto import tqdm
import argparse 
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from yolov3.yolov3_model import YOLOv3
from yolov3.yolov3imagefolder import CocoImageFolder
from yolov3.yolov3lossfunc import Yolov3Loss
from tqdm import tqdm

def model_evaluation(quantized_model,
                     test_dataloader,
                     device,
                     loss_function,
                     deploy=False):
    size = [608//32, 608//16, 608//8]
    num_anchors = 3
    num_coordinate = 85 # 8 = 5+num_class; num_classs = 80
    quantized_model.to(device)
    quantized_model.eval()
    total_loss = 0
    with torch.inference_mode():
        for batch, (image, label) in tqdm(enumerate(test_dataloader), desc="Running evaluation"):
            image = image.to(device)
            batch_size = image.shape[0]
            stage1_out, stage2_out, stage3_out = quantized_model(image)
            stage1_out = stage1_out.permute(0,2,3,1).view(batch_size, size[0], size[0], num_anchors, num_coordinate)
            stage1_confidence = stage1_out[..., 0:1]
            stage1_x_center = stage1_out[..., 1:2]
            stage1_y_center = stage1_out[..., 2:3]
            stage1_width = stage1_out[..., 3:4]
            stage1_height = stage1_out[..., 4:5]
            stage1_class = stage1_out[..., 5:]
            large_scale_prediction = torch.cat((stage1_confidence, stage1_x_center, stage1_y_center, stage1_width, stage1_height, stage1_class), dim=-1)

            stage2_out = stage2_out.permute(0,2,3,1).view(batch_size, size[1], size[1], num_anchors, num_coordinate)
            stage2_confidence = stage2_out[..., 0:1]
            stage2_x_center = stage2_out[..., 1:2]
            stage2_y_center = stage2_out[..., 2:3]
            stage2_width = stage2_out[..., 3:4]
            stage2_height = stage2_out[..., 4:5]
            stage2_class = stage2_out[..., 5:]
            medium_scale_prediction = torch.cat((stage2_confidence, stage2_x_center, stage2_y_center, stage2_width, stage2_height, stage2_class), dim=-1)

            stage3_out = stage3_out.permute(0,2,3,1).view(batch_size, size[2], size[2], num_anchors, num_coordinate)
            stage3_confidence = stage3_out[..., 0:1]
            stage3_x_center = stage3_out[..., 1:2]
            stage3_y_center = stage3_out[..., 2:3]
            stage3_width = stage3_out[..., 3:4]
            stage3_height = stage3_out[..., 4:5]
            stage3_class = stage3_out[..., 5:]
            small_scale_prediction = torch.cat((stage3_confidence, stage3_x_center, stage3_y_center, stage3_width, stage3_height, stage3_class), dim=-1)
        #     loss = (
        #     loss_function(predictions=large_scale_prediction, labels=label[0].to(device), scale="large", device=device)
        #     + loss_function(predictions=medium_scale_prediction, labels=label[1].to(device), scale="medium", device=device)
        #     + loss_function(predictions=small_scale_prediction, labels=label[2].to(device), scale="small", device=device)
        # )
            
        #     total_loss += loss.item()
           
            if deploy:
                return
            
        total_loss /= (3*len(test_dataloader))
        print(f'[INFO] total_loss: {total_loss}')

def main():
    parser = argparse.ArgumentParser(description="Input required arguments for the quantization, testing and deployment of yolov3 model.")
    parser.add_argument('-tb', "--test_batch", type=int, default=8, help="Argument for the batch size in test dataloader.")
    parser.add_argument('-cb', '--calib_batch', type=int, default=8, help="Argument for the batch size in calibration dataloader.")
    parser.add_argument('-w', '--weight', type=str, default="weights/yolov3_wts_reference.pt", help="Argument for the pretrained weight of yolo model in '.pt' format.")
    parser.add_argument('-m', '--mode', type=str, default="float", help="Argument for the different required during quantization i.e. 'calib', 'test'.")
    parser.add_argument('-d', '--device', type=str, default='gpu', help="Argument for the device used during quantization and deployment.")
    parser.add_argument('-s', '--input_size', type=int, default=608, help="Argument for the input size to model.")
    parser.add_argument('-n', '--num_class', type=int, default=80, help="Argument for number of class in which model is trained.")
    parser.add_argument('-f', '--result_dir', type=str, default="./quantized_result", help="Argument for the directory where the results of quantization will be save.")
    parser.add_argument('-i', '--deploy', action='store_true')
    
    arg = parser.parse_args()
    
    if arg.device == "gpu":
        device = torch.device('cuda')
    elif arg.device == 'cpu':
        device = torch.device('cpu')
        
    if arg.deploy:
        arg.test_batch = 1
    
    anchors_list = [
        [116,90, 156,198, 373,326],
        [30,61, 62,45, 59,119],
        [10,13, 16,30, 33,23]]
        
    calibration_dataset = CocoImageFolder(root="data/calibration", image_size=arg.input_size)
    test_dataset = CocoImageFolder(root="data/test", image_size=arg.input_size)
    
    calibration_dataloader = torch.utils.data.DataLoader(dataset=calibration_dataset,
                                                            batch_size=arg.calib_batch,
                                                            shuffle=True,
                                                            num_workers=os.cpu_count())
    
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                            batch_size=arg.test_batch,
                                                            shuffle=False,
                                                            num_workers=os.cpu_count())
    
    model = YOLOv3(input_size=arg.input_size, anchors=anchors_list, num_classes=arg.num_class)
    wts = torch.load(f=arg.weight)
    new_wts = {my_para_name:wts[pretrained_para_name] for my_para_name, pretrained_para_name in zip(model.state_dict().keys(), wts.keys())}
    model.load_state_dict(new_wts, strict=True)
    
    loss = Yolov3Loss(anchors=anchors_list, input_size=arg.input_size)
    
    if not arg.mode == 'float':
        input = torch.rand(1,3,608,608)
        # input, labels = next(iter(calibration_dataloader))
        # for batch, (input, labels) in enumerate(calibration_dataloader):
            # print(f'[INFO] shape of input : {input.shape}, {type(input)}')
        quantizer = torch_quantizer(arg.mode, model, (input), output_dir=arg.result_dir, device=device)
        model = quantizer.quant_model
        
    model_evaluation(quantized_model=model, test_dataloader=test_dataloader, device=device, loss_function=loss, deploy=arg.deploy)
    
    if arg.mode == 'calib':
        quantizer.export_quant_config()
        
    if arg.deploy:
        quantizer.export_xmodel(arg.result_dir, deploy_check=True)
        
        
if __name__ == "__main__":
    main() 
            
        