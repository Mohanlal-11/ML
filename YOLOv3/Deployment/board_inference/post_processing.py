import os
import pickle
import cv2 as cv
import numpy as np
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from tools import nms, decode, softmax

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
W,H = 608,608

def post_proces(pred_boxes, grid):
    assert type(pred_boxes) == list, "All the predictions should be in list i.e. List[torch.tensor]."
    decoded_boxes = []
    for scale, preds in enumerate(pred_boxes):
        decoded_prediction = decode(preds, grid, scale)
        decoded_boxes.append(decoded_prediction)
    # print(f'decoding finished.')
    final_large_boxes = nms(decoded_boxes[0])
    final_medium_boxes = nms(decoded_boxes[1])
    final_small_boxes = nms(decoded_boxes[2])
    # print(f'nms finished')
    return [final_large_boxes, final_medium_boxes, final_small_boxes]

def display_output(final_boxes, re_img, ori_image, size, im_name):
    for id, boxes in enumerate(final_boxes):
        for box in boxes:
            confidence = box[4]
            x_c = (box[0]/size[id])
            y_c = (box[1]/size[id])
            w = (box[2]/608)
            h = (box[3]/608)
            labels = box[5:]
            label = np.argmax(softmax(np.array(labels), axis=-1))

            x_min = (x_c - (w/2))*ori_image.shape[1]
            y_min = (y_c - (h/2))*ori_image.shape[0]
            x_max = (x_c + (w/2))*ori_image.shape[1]
            y_max = (y_c + (h/2))*ori_image.shape[0]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = max(0, x_max)
            y_max = max(0, y_max)
            
            if confidence > 0.65:
                print([x_c, y_c, w, h, label])
                ori_image = cv.rectangle(ori_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0,0), 3)
                ori_image = cv.putText(ori_image, f"Class: {CLASSES[label]}", (int(x_min), int(y_min)-30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv.LINE_AA)
                ori_image = cv.putText(ori_image, f"Confidence: {str(confidence)}", (int(x_min), int(y_min)-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv.LINE_AA)
                # cv.imwrite(f"prediction/{im_name}", ori_image)
                # print(f'[INFO] The predictions on image {im_name} is saved in folder "prediction".')
                cv.imshow("visualization", ori_image)
                k = cv.waitKey(0)
                if k == ord('q'):
                    break
        cv.destroyAllWindows()
    
def main():
    with open('yolov3_model_preds_large.pkl', 'rb') as f:
        large_boxes = pickle.load(f)
    
    with open('yolov3_model_preds_medium.pkl', 'rb') as f:
        medium_boxes = pickle.load(f)
    with open('yolov3_model_preds_small.pkl', 'rb') as f:
        small_boxes = pickle.load(f)
    print(large_boxes.shape, medium_boxes.shape, small_boxes.shape)
    print(f'type and dtype of boxes : {type(large_boxes),large_boxes.dtype}')
    ori_img = cv.imread("img/pedestrain1.jpg")
    re_img = cv.resize(ori_img, (608,608))
    pred_boxes = [(np.transpose(large_boxes, (0,2,3,1))),np.transpose(medium_boxes, (0,2,3,1)),np.transpose(small_boxes, (0,2,3,1))]    
    grid_size = [W//32,W//16,W//8]
    results = post_proces(pred_boxes=pred_boxes, grid=grid_size)
    display_output(results, re_img, ori_image=ori_img, size=grid_size, im_name="pedestrain1.jpg")
    
if __name__ == "__main__":
    main()
        
    