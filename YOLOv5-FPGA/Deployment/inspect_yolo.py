import torch
from pytorch_nndct.apis import Inspector

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import yolo

def inspection_model():
    target = "DPUCZDX8G_ISA1_B4096"
    # Initialize inspector with target
    inspector = Inspector(target)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model to be inspected
    model = yolo.YOLOv5(80, img_sizes=320, score_thresh=0.3)
    # model.eval()
    # Random input
    dummy_input = torch.randn(1,3,320,320)

    # Start to inspect
    inspector.inspect(model, (dummy_input,), device=device, output_dir="Deployment/inspection_results", image_format="svg") 
    
if __name__ =="__main__":
    inspection_model()