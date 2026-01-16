import torch

from PIL import Image
from torchvision import transforms
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
import cv2
import numpy as np
import datetime
import pickle

import yolo
from yolo.model.transform import Transformer
from yolo.model import box_ops


strides = (8,16,32)
# anchors = [
#             [[10, 13], [16, 30], [33, 23]],
#             [[30, 61], [62, 45], [59, 119]],
#             [[116, 90], [156, 198], [373, 326]]
#         ]
anchors = [
            [[3.6, 6.9], [7.0, 19.0], [14.5, 39.1]],
            [[19.4, 44.9], [22.1, 67.6], [46.8, 106.6]],
            [[92.6, 142.3], [151.4, 126.8], [108.6, 177.6]],
        ]
detections = 100


def inference(preds, image_shapes, scale_factors, max_size, score_thresh, nms_thresh, merge=True):
    anchors_tens = torch.tensor(anchors)
    ids, ps, boxes = [], [], []
    for pred, stride, wh in zip(preds, strides, anchors_tens): # 3.54s
        pred = torch.sigmoid(pred)
        n, y, x, a = torch.where(pred[..., 4] > score_thresh)
        p = pred[n, y, x, a]
        
        xy = torch.stack((x, y), dim=1)
        xy = (2 * p[:, :2] - 0.5 + xy) * stride
        wh = 4 * p[:, 2:4] ** 2 * wh[a]
        box = torch.cat((xy, wh), dim=1)
        
        ids.append(n)
        ps.append(p)
        boxes.append(box)
        
    ids = torch.cat(ids)
    ps = torch.cat(ps)
    boxes = torch.cat(boxes)
    
    boxes = box_ops.cxcywh2xyxy(boxes)
    # print(boxes)
    logits = ps[:, [4]] * ps[:, 5:]
    indices, labels = torch.where(logits > score_thresh) # 4.94s
    ids, boxes, scores = ids[indices], boxes[indices], logits[indices, labels]
    
    results = []
    for i, im_s in enumerate(image_shapes): # 20.97s
        keep = torch.where(ids == i)[0] # 3.11s
        box, label, score = boxes[keep], labels[keep], scores[keep]
        #ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1] # 0.27s
        #keep = torch.where((ws >= self.min_size) & (hs >= self.min_size))[0] # 3.33s
        #boxes, objectness, logits = boxes[keep], objectness[keep], logits[keep] # 0.36s
        
        if len(box) > 0:
            box[:, 0].clamp_(0, im_s[1]) # 0.39s
            box[:, 1].clamp_(0, im_s[0]) #~
            box[:, 2].clamp_(0, im_s[1]) #~
            box[:, 3].clamp_(0, im_s[0]) #~
            
            keep = box_ops.batched_nms(box, score, label, nms_thresh, max_size) # 4.43s
            keep = keep[:detections]
            
            nms_box, nms_label = box[keep], label[keep]
            if merge: # slightly increase AP, decrease speed ~14%
                mask = nms_label[:, None] == label[None]
                iou = (box_ops.box_iou(nms_box, box) * mask) > nms_thresh # 1.84s
                weights = iou * score[None] # 0.14s
                nms_box = torch.mm(weights, box) / weights.sum(1, keepdim=True) # 0.55s
                
            box, label, score = nms_box / scale_factors[i], nms_label, score[keep] # 0.30s
        results.append(dict(boxes=box, labels=label, scores=score)) # boxes format: (xmin, ymin, xmax, ymax)
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wts", type=str, default='weights/yolov5_nano-1000.pth', help='path to pretrained weight of yolov5.')
    parser.add_argument("--input_size", type=int, default=320, help="input size of model.")
    parser.add_argument("--score_thres", type=float, default=0.3, help="score threshold for post processing.")
    parser.add_argument("--nms_thres", type=float, default=0.3)
    parser.add_argument("--img_path", type=str, default="test_img", help='path to folder that contains test images')
    parser.add_argument("--model_size", type=str, default="small", help="model size")
    parser.add_argument("--save_dir", type=str, default="yolov5_inference", help='path to folder to save inference results.')
    parser.add_argument("--gt", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--coco", action='store_true')
    args = parser.parse_args()

    # COCO dataset, 80 classes
    if args.coco:
        classes = (
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
    
    #LLVIP dataset
    else:
        classes = ("person",)
    
    num_classes = len(classes)
    images_path = [os.path.join(args.img_path, name) for name in os.listdir(args.img_path) if name.endswith(('.jpg', '.jpeg', '.png'))]
    
    save_folder = Path("INFERENCE")
    save_dir = save_folder/f"{datetime.datetime.now().strftime('%Y-%m-%d')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_sizes = {"nano":(0.33, 0.25), "small": (0.33, 0.5), "medium": (0.67, 0.75), "large": (1, 1), "extreme": (1.33, 1.25)}
    model = yolo.YOLOv5(num_classes, model_size=model_sizes[args.model_size], img_sizes=args.input_size, score_thresh=args.score_thres)

    trainable = 0
    nontrainable = 0
    for par in model.parameters():
        if par.requires_grad:
            trainable += par.numel()
        else:
            nontrainable += par.numel()

    print(f'Model has total trainable parameters {trainable}')
    print(f'Model has total non-trainable parameters {nontrainable}')

    checkpoint = torch.load(args.wts)
    # print(checkpoint.keys())
    if args.pretrain:
        model_state_dict = model.state_dict()
        # print(len(model_state_dict.keys()))

        # Filter out keys that don't match shape
        filtered_state_dict = {}
        for k, v in checkpoint.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")

        print(f'[INFO] The len of wts keys in filtered wts and own model is : {len(filtered_state_dict.keys()), len(model_state_dict.keys())}')
        model.load_state_dict(filtered_state_dict, strict=False)
        # model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint["model"])

    if isinstance(args.input_size, int):
        img_sizes = (args.input_size, args.input_size)
    transformer = Transformer(
        min_size=img_sizes[0], max_size=img_sizes[1], stride=max(strides))
    model.eval()
    transformer.eval()
    for im_path in tqdm(images_path[:200], desc="Inference"):
        im_name_only = im_path.split('/')[-1]
        res_save_path = os.path.join(save_dir, im_name_only)
        # img = Image.open(im_path).convert("RGB")
        ori_img = cv2.imread(im_path)
        resized_img = cv2.resize(ori_img, (args.input_size, args.input_size))
        img = transforms.ToTensor()(ori_img)

        model.head.merge = False

        images = [img]
        images, targets, scale_factors, image_shapes, _ = transformer(images, targets=None)
        predictions = model(images, targets)
        # for i, prd in enumerate(predictions):
        #     # print(prd.shape)
        #     with open(f"yolov5_preds_{i}.pkl", "wb") as fw:
        #         pickle.dump(prd.detach().cpu().numpy(), fw)
        max_size = max(images.shape[2:])
        results = inference(preds=predictions, image_shapes=image_shapes, scale_factors=scale_factors,max_size=max_size, score_thresh=args.score_thres, nms_thresh=args.nms_thres)

        for res in results:
            boxes = res["boxes"].cpu().detach().tolist()
            labels = res["labels"].cpu().tolist()
            scores = res["scores"].cpu().tolist()
            # img = img.permute(1,2,0)
            # img = img.numpy()*255.0
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # # print(img.shape)
            # img = cv2.UMat(img)
            
            for i, box in enumerate(boxes):
                # print(box)
                resized_img = cv2.rectangle(img=resized_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0,255,0), thickness=2)
                resized_img = cv2.putText(resized_img, classes[labels[i]]+":"+str(scores[i]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(res_save_path, resized_img)
        # yolo.show(images, results, gt_box, classes, save=res_save_path)