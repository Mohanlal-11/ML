import torch
from pytorch_nndct.apis import Inspector
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from yolov3.yolov3_model import YOLOv3

def inspection_model():
    anchors_list = [
        [116,90, 156,198, 373,326],
        [30,61, 62,45, 59,119],
        [10,13, 16,30, 33,23]]
    
    target = "DPUCZDX8G_ISA1_B4096"
    # Initialize inspector with target3
    inspector = Inspector(target)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wts = torch.load("weights/yolov3_wts_reference.pt", map_location=torch.device(device))
    
    # Load the model to be inspected
    model = YOLOv3(input_size=608, anchors=anchors_list, num_classes=80)
    new_wts = {my_para_name:wts[pretrained_para_name] for my_para_name, pretrained_para_name in zip(model.state_dict().keys(), wts.keys())}
    model.load_state_dict(new_wts, strict=True)

    # Random input
    dummy_input = torch.randn(1,3,608,608)

    # Start to inspect
    inspector.inspect(model, (dummy_input,), device=device, output_dir="./code/inspection_results", image_format="svg") 
    
if __name__ =="__main__":
    inspection_model()