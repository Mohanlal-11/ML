import torch
import yolo
from PIL import Image
from torchvision import transforms
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
import cv2
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wts", type=str, default='weights/yolov5small-45.pth', help='path to pretrained weight of yolov5.')
    parser.add_argument("--input_size", type=int, default=320, help="input size of model.")
    parser.add_argument("--score_thres", type=int, default=0.3, help="score threshold for post processing.")
    parser.add_argument("--img_path", type=str, default="/media/logictronix01/ML-41/LLVIP_reAnno/LLVIP_reannotated_all/test", help='path to folder that contains test images')
    parser.add_argument("--save_dir", type=str, default="yolov5_inference", help='path to folder to save inference results.')
    parser.add_argument("--json", type=str, default='/media/logictronix01/ML-41/LLVIP_reAnno/LLVIP_reannotated_all/annotations/test_LLVIP_all_reannotated_dataset_yolov5.json', help='path to json file of test/val image.')
    parser.add_argument("--gt", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    args = parser.parse_args()

    # COCO dataset, 80 classes
    # classes = (
    #     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    #     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    #     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    #     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    #     "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    #     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    #     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    #     "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    #     "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    #     "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
    
    #LLVIP dataset
    classes = ("person",)
    
    num_classes = len(classes)
    images_path = [os.path.join(args.img_path, name) for name in os.listdir(args.img_path) if name.endswith(('.jpg', '.jpeg', '.png'))]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.json, "r") as fp:
        json_data = json.load(fp)
       
    gt_img_names = json_data["images"]
    gt_img_anno = json_data["annotations"]
    
    img_info = {img["file_name"]:img["id"] for img in gt_img_names}
    anno_info = {}
    for anno in gt_img_anno:
        anno_info[anno["id"]]=[]
    
    for ann in gt_img_anno:
        anno_info[ann["id"]].append(ann["bbox"])
    # print(anno_info, len(anno_info))
    model = yolo.YOLOv5(num_classes, img_sizes=args.input_size, score_thresh=args.score_thres)
    model.eval()

    checkpoint = torch.load(args.wts)
    if args.pretrain:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint["model"])

    for im_path in tqdm(images_path[:200], desc="Inference"):
        im_name_only = im_path.split('/')[-1]
        if args.gt:
            if im_name_only in img_info.keys():
                img_id = img_info[im_name_only]
                gt_box = anno_info[img_id]
                # print(im_name_only, gt_box)
        else:
            gt_box=None
                
        res_save_path = os.path.join(save_dir, im_name_only)
        img = Image.open(im_path).convert("RGB")
        img = transforms.ToTensor()(img)

        model.head.merge = False

        images = [img]
        results, losses = model(images)

        for res in results:
            boxes = res["boxes"].cpu().detach().tolist()
            labels = res["labels"].cpu().tolist()
            scores = res["scores"].cpu().tolist()
            img = img.permute(1,2,0)
            img = img.numpy()*255.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(img.shape)
            img = cv2.UMat(img)
            
            for i, box in enumerate(boxes):
                # print(box)
                img = cv2.rectangle(img=img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0,255,0), thickness=2)
                img = cv2.putText(img, classes[labels[i]]+":"+str(scores[i]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 1, cv2.LINE_AA)
                if args.gt:
                    for box in gt_box:
                        xmin, ymin = box[0], box[1]
                        xmax = box[0]+box[2]
                        ymax = box[1]+box[3]
                        img = cv2.rectangle(img=img, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), color=(0,0,255), thickness=2)

            cv2.imwrite(res_save_path, img)
        # yolo.show(images, results, gt_box, classes, save=res_save_path)