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
    parser.add_argument(
        "--wts",
        type=str,
        default="Mohan/YOLOV5_Thermal_pretrained_from_ori_repo/pretrained_wts/yolov5small.pth",
        help="path to pretrained weight of yolov5.",
    )
    parser.add_argument(
        "--input_size", type=int, default=672, help="input size of model."
    )
    parser.add_argument(
        "--score_thres",
        type=int,
        default=0.3,
        help="score threshold for post processing.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="path to folder that contains test images",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="yolov5_inference",
        help="path to folder to save inference results.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="LLVIP_reannotated_all/annotations/test_LLVIP_all_reannotated_dataset_yolov5.json",
        help="path to json file of test/val image.",
    )
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

    # LLVIP dataset
    classes = ("person",)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = len(classes)
    # images_path = [
    #     os.path.join(args.img_path, name)
    #     for name in os.listdir(args.img_path)
    #     if name.endswith((".jpg", ".jpeg", ".png"))
    # ]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json, "r") as fp:
        json_data = json.load(fp)

    gt_img_names = json_data["images"]
    gt_img_anno = json_data["annotations"]

    img_info = {img["file_name"]: img["id"] for img in gt_img_names}
    anno_info = {}
    for anno in gt_img_anno:
        anno_info[anno["image_id"]] = []

    for ann in gt_img_anno:
        anno_info[ann["image_id"]].append(ann["bbox"])
    # print(anno_info, len(anno_info))
    model = yolo.YOLOv5(
        num_classes, img_sizes=args.input_size, score_thresh=args.score_thres
    )
    model.eval()

    checkpoint = torch.load(args.wts, map_location=device)
    if args.pretrain:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint["model"])
    model.to(device)
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"cannot open the video")
        exit()

    frame_id = 0
    while True:
        # image = cv.imread("data/pedestrain1.jpg", cv.IMREAD_COLOR)
        ret, img = cap.read()
        img = cv2.resize(img, (args.input_size, args.input_size))
        
        img_copy = img.copy()
        # cv2.imshow("ori", img)
        # key = cv2.waitKey(30)
        # if key==ord('q'):
        #     break
        if not ret:
            print("video ended.")
            exit()

        img = transforms.ToTensor()(img)
        img=img.to(device)
        model.head.merge = False

        images = [img]
        results, losses = model(images)

        for res in results:
            boxes = res["boxes"].cpu().detach().tolist()
            labels = res["labels"].cpu().tolist()
            scores = res["scores"].cpu().tolist()

            for i, box in enumerate(boxes):
                # print(box)
                img_copy = cv2.rectangle(
                    img=img_copy,
                    pt1=(int(box[0]), int(box[1])),
                    pt2=(int(box[2]), int(box[3])),
                    color=(0, 255, 0),
                    thickness=2,
                )
                img_copy = cv2.putText(
                    img_copy,
                    classes[labels[i]] + ":" + str(scores[i]),
                    (int(box[0]), int(box[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
        cv2.imshow("from video", img_copy)

        key = cv2.waitKey(3) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
