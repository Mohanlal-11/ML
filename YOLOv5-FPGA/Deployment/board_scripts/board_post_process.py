import numpy as np

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from board_utils import sigmoid, cxcywh2xyxy, nms

strides = (8,16,32)
anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]
detections = 100

def inference_own(predictions, score_thres, nms_thres):
    anchors_arr = np.array(anchors)
    # print(anchors_arr.shape)
    ids, ps, boxes = [], [], []
    for pred, stride, wh in zip(predictions, strides, anchors_arr):
        pred = sigmoid(pred)
        # print(pred.shape)
        n,y,x,a = np.where(pred[..., 4]>score_thres)
        # print(f'{n}\n,{y}\n,{x}\n,{a}\n')
        selected_pred = pred[n, y, x, a]
        # print(f'selected preds:{selected_pred.shape}')
        selected_topleft = np.stack((x,y), axis=1)
        # print(f'selected topleft:{selected_topleft}')
        bbox_cxcy = (2*selected_pred[:, :2] - 0.5 + selected_topleft) * stride
        bbox_wh = 4 * selected_pred[:, 2:4] ** 2 * wh[a]
        bbox = np.concatenate((bbox_cxcy, bbox_wh), axis=-1)

        ids.append(n)
        ps.append(selected_pred)
        boxes.append(bbox)

    ids = np.concatenate(ids, axis=0)
    ps = np.concatenate(ps, axis=0)
    boxes = np.concatenate(boxes, axis=0)
    # print(f'selected boxes:{boxes.shape, boxes.dtype}')

    boxes = cxcywh2xyxy(boxes)
    # print(boxes.shape, boxes.dtype)
    # print(ps[:, [4]].shape, ps[:, 4].shape)
    logits = ps[:, [4]] * ps[:, 5:]
    indices, labels = np.where(logits > score_thres) # 4.94s
    ids, boxes, scores = ids[indices], boxes[indices], logits[indices, labels]
    indx = nms(boxes, scores, nms_thres)
    # print(indices, labels)
    final_boxes, final_scores, final_cls_ids = boxes[indx], scores[indx], labels[indx]
    # print(final_boxes, final_scores, final_cls_ids)
    return final_boxes, final_scores, final_cls_ids


if __name__ == "__main__":
    import pickle
    import cv2

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
    
    predictions = []
    for i in range(3):
        with open(f'yolov5_preds_{i}.pkl', 'rb') as fp:
            predictions.append(pickle.load(fp))

    res = inference_own(predictions, 0.3, 0.4)
    boxes, scores, cls_ids = res
    image = cv2.imread("test_one/000000317863.jpg")
    resized_img = cv2.resize(image, (320,320))
    for i, box in enumerate(boxes):
        resized_img = cv2.rectangle(img=resized_img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0,255,0), thickness=2)
        resized_img = cv2.putText(resized_img, classes[cls_ids[i]]+":"+str(scores[i]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255, 0, 0), 1, cv2.LINE_AA)
        
    cv2.imshow("vis", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()