import os
import numpy as np

anchors = [
            [116,90, 156,198, 373,326],
            [30,61, 62,45, 59,119],
            [10,13, 16,30, 33,23]]
nms_conf = 0.5
nms_iou = 0.3

def create_grid(grid_size):
    grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    return grid_x, grid_y

def set_grid_xy_np(grid_size):
    grid_x, grid_y = create_grid(grid_size)
    grid_x = np.broadcast_to((np.reshape(grid_x, (1, grid_size, grid_size,1))), (1,grid_size, grid_size,3))
    grid_y = np.broadcast_to((np.reshape(grid_y, (1, grid_size, grid_size, 1))), (1,grid_size,grid_size,3))
    return grid_x, grid_y

def calculate_ious(bbox1, bbox2):
    x1 = bbox1[..., 0:1] - bbox1[..., 2:3]/2
    y1 = bbox1[..., 1:2] - bbox1[..., 3:4]/2
    x2 = bbox1[..., 0:1] + bbox1[..., 2:3]/2
    y2 = bbox1[..., 1:2] + bbox1[..., 3:4]/2
    
    X1 = bbox2[..., 0:1] - (bbox2[..., 2:3] / 2)
    Y1 = bbox2[..., 1:2] - (bbox2[..., 3:4] / 2)
    X2 = bbox2[..., 0:1] + (bbox2[..., 2:3] / 2)
    Y2 = bbox2[..., 1:2] + (bbox2[..., 3:4] / 2)

    overlapped_xmin = np.maximum(x1,X1)
    overlapped_xmax = np.minimum(x2,X2)
    overlapped_ymin = np.maximum(y1,Y1)
    overlapped_ymax = np.minimum(y2,Y2)
    
    overlapped_width = np.maximum((overlapped_xmax-overlapped_xmin), 0)
    overlapped_height = np.maximum((overlapped_ymax-overlapped_ymin), 0)
    intersection_area = overlapped_width*overlapped_height
    
    bbox1_area = abs((x2-x1) * (y2-y1))
    bbox2_area = abs((X2-X1) * (Y2-Y1)) 
    
    union_area = bbox1_area + bbox2_area - intersection_area
    IoU = intersection_area/union_area
    return IoU

def nms(decoded_bboxes):
    bboxes = decoded_bboxes.reshape(-1, decoded_bboxes.shape[4])
    bboxes = bboxes.tolist()
    bboxes_obj = [box for box in bboxes if box[4]>nms_conf]
    bboxes_obj = sorted(bboxes_obj, key=lambda x:x[4], reverse=True)
    selected_bboxes = []
    
    while bboxes_obj:
        chosen_bbox = bboxes_obj.pop(0)
        bboxes_obj = [bbox for bbox in bboxes_obj if np.argmax(np.array(bbox[5:]))!=np.argmax(np.array(chosen_bbox[5:])) or calculate_ious(bbox1=np.array(bbox[0:4]), bbox2=np.array(chosen_bbox[0:4]))<nms_iou]
        selected_bboxes.append(chosen_bbox)
    
    return selected_bboxes

def decode(pred_boxes, grids, scale):
    # print(f'shape of preds received in decode: {pred_boxes.shape}')
    trans_pred = pred_boxes.reshape(1,grids[scale], grids[scale], 3, 85)
    trans_pred = np.concatenate((trans_pred[..., 0:1],trans_pred[..., 1:2],trans_pred[..., 2:3],trans_pred[..., 3:4],trans_pred[..., 4:5],trans_pred[..., 5:]), axis=-1)
  
    conf = trans_pred[..., 4]
    x_c = trans_pred[..., 0]
    y_c = trans_pred[..., 1]
    wid = trans_pred[..., 2]
    hei = trans_pred[..., 3]
    cls_prob = trans_pred[..., 5:]
    
    scale_anchor = np.array(anchors[scale])
    scale_anchor_width = np.broadcast_to((np.reshape(scale_anchor[::2], (1,1,1,3))), (1,grids[scale], grids[scale], 3))
    scale_anchor_height = np.broadcast_to((np.reshape(scale_anchor[1::2], (1,1,1,3))), (1, grids[scale], grids[scale], 3))
    
    grid_x, grid_y = set_grid_xy_np(grid_size=grids[scale])
    conf = sigmoid(conf)
    x_center = sigmoid(x_c) + grid_x
    y_center = sigmoid(y_c) + grid_y
    width = np.exp(wid) * scale_anchor_width
    height = np.exp(hei) * scale_anchor_height
    
    coordinates = np.stack((x_center, y_center, width, height, conf), axis=-1)
    coordinates = np.concatenate((coordinates, cls_prob), axis=-1)
    return coordinates
    
def sigmoid(logit_values):
    return 1 / (1+np.exp(-logit_values))

def softmax(logits, axis):
    shifted_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True)) #Here, the input logits are subtracted with max of input for numerical stability.
                                                                            #This is because if input are too large then its exponential value will to inf, so if we did this then all inputs 
                                                                            # which are small than max value will be -ve and its exp will be too small (e.g.: 0.01, 0.23) and max value will be '0' whose exp will be max i.e. '1'.
    softmax_value = shifted_logits / np.sum(shifted_logits, axis=axis, keepdims=True)
    return softmax_value
