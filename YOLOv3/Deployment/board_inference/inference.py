import xir 
import vart
import threading

import os
import sys
import time
from typing import List

import cv2 as cv
import numpy as np

from tools import nms, decode, softmax

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_dpu_subgraphs(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "Given graph is None. It should not be None."
    
    root_subgraph = graph.get_root_subgraph()
    assert root_subgraph is not None, "Failed to get the root subgraph of given graph."
    if root_subgraph.is_leaf:
        return []
    
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs)>0
    
    return [subgraph for subgraph in child_subgraphs if subgraph.has_attr("device") and subgraph.get_attr("device").upper() == "DPU"]
    
def pre_process_image(image_path,W,H,scale):
    input_scale = 2**scale
    image_ori = cv.imread(image_path)
    image = cv.cvtColor(image_ori, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (W, H))
    
    image = image.astype(np.float32) / 255.0
    # image  = image*input_scale
    return image, image_ori
    
def runYolo(runner, imgs):
    input_tensor = runner.get_input_tensors()
    output_tensor = runner.get_output_tensors()
    
    output_fixpos = output_tensor[0].get_attr("fix_point")
    # print(f'output fixpos: {output_fixpos}')
    print(f'[INFO] The input and output tensors are: {input_tensor}, \n {output_tensor}')
    input_shape = tuple(input_tensor[0].dims)
    
    output_shape1 = tuple(output_tensor[0].dims) #for large scale
    output_shape2 = tuple(output_tensor[1].dims)
    output_shape3 = tuple(output_tensor[2].dims)
    
    print(f'[INFO] Input and output shapes are: {input_shape} \n {output_shape1, output_shape2, output_shape3}')
    
    input_data = [np.empty(input_shape, dtype=np.float32, order="C")]
    # print(f'input data: {input_data[0].shape}, \n {input_data[0]}')
    output_data1 = np.empty(output_shape1, dtype=np.float32, order="C")
    output_data2 = np.empty(output_shape2, dtype=np.float32, order="C")
    output_data3 = np.empty(output_shape3, dtype=np.float32, order="C")
    
    output_data = [output_data1, output_data2, output_data3] #output buffer for large, medium and small scale predictions.
    
    input_data[0] = imgs.reshape(input_shape[1], input_shape[2], input_shape[3])
    
    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)
    
    return [output_data[0], output_data[1], output_data[2]]

def post_process_output(pred_boxes, grid):
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

def display_output(final_boxes, ori_image, size, im_name):
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
        cv.destroyAllWindows()

def main(argv):
    print("====================Model inference is begining===================!")
    
    graphs = xir.Graph.deserialize(argv[1])
    subgraphs = get_dpu_subgraphs(graphs)
    assert len(subgraphs) == 1, "The length of subgraphs must be 'one' because that ensures all graphs/layers can be executable in 'DPU'."
    
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")
    
    input_tensors = dpu_runner.get_input_tensors()
    input_shape = tuple(input_tensors[0].dims)
    H = input_shape[1]
    W = input_shape[2]
    print(f'model input size : {H,W}')
    input_fixpos = dpu_runner.get_input_tensors()[0].get_attr("fix_point")
    # print(f'input fix position: {input_fixpos}')
    # output_fixpos = dpu_runner.get_output_tensors()[0].get_attr("fix_point")
    grid_size = [W//32, W//16, W//8]
        
    images_dir = argv[2]
    images_paths = [os.path.join(images_dir, img_path) for img_path in os.listdir(images_dir) if img_path.endswith((".jpg", ".jpeg", ".png"))]
    
    for im_path in images_paths:
        im_name = im_path.split("/")[-1]
        processed_image, original_image = pre_process_image(im_path,W,H, input_fixpos) 
        print(f'input and ori img : {processed_image.shape, original_image.shape}')
        prediction = runYolo(dpu_runner, processed_image)
        
        print(f'[INFO] The shape of predictions : {prediction[0].shape, prediction[1].shape, prediction[2].shape}')
        # prediction = [np.transpose(prediction[0], (0, 3, 1, 2)), np.transpose(prediction[1], (0, 3, 1, 2)), np.transpose(prediction[2], (0, 3, 1, 2))]
        boxes = post_process_output(prediction, grid_size)
        
        display_output(boxes, original_image, grid_size, im_name)
        

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python3 inference.py <yolov3.xmodel> <path/to/image/directory>"
    main(sys.argv)
