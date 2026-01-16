import xir 
import vart
import threading

import os
import sys
import time
from typing import List

import cv2 as cv
import numpy as np

from board_post_process import inference_own

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
    
def pre_process_image(image_path,W,H):
    image_ori = cv.imread(image_path)
    image = cv.cvtColor(image_ori, cv.COLOR_BGR2RGB)
    image = cv.resize(image_ori, (H, W))
    
    image = image.astype(np.float32) / 255.0
    return image, image_ori
    
def runYolo(runner, imgs):
    # print(imgs, imgs.dtype)
    input_tensor = runner.get_input_tensors()
    output_tensor = runner.get_output_tensors()
    
    # print(f'[INFO] The input and output tensors are: {input_tensor}, \n {output_tensor}')
    input_shape = tuple(input_tensor[0].dims)
    
    output_shape1 = tuple(output_tensor[0].dims)
    output_shape2 = tuple(output_tensor[1].dims)
    output_shape3 = tuple(output_tensor[2].dims)
    
    # print(f'[INFO] Input and output shapes are: {input_shape} \n {output_shape1, output_shape2, output_shape3}')
    
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

def display(res, image, fps):
    boxes, scores, cls_ids = res
    for i, box in enumerate(boxes):
        image = cv.rectangle(img=image, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0,255,0), thickness=2)
        image = cv.putText(image, f"FPS: {round(fps)}", (10,20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)
        image = cv.putText(image, CLASSES[cls_ids[i]]+":"+str(scores[i]), (int(box[0]), int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 
            1, (255, 0, 0), 1, cv.LINE_AA)
    cv.imshow("visualize", image)
    cv.waitKey(0)
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
    # print(f'model input size : {(H,W)}')
        
    images_dir = argv[2]
    images_paths = [os.path.join(images_dir, img_path) for img_path in os.listdir(images_dir) if img_path.endswith((".jpg", ".jpeg", ".png"))]
    
    for im_path in images_paths:
        # im_name = im_path.split("/")[-1]
        pre_time1 = time.perf_counter()
        processed_image, original_image = pre_process_image(im_path,W,H)
        pre_time2 = time.perf_counter()
        pre_time = (pre_time2-pre_time1)*1000
        print(f'Time taken by pre-processing: {pre_time:.3f}ms.')

        # print(f'input and ori img : {processed_image.shape, original_image.shape}')
        model_time1 = time.perf_counter()
        prediction = runYolo(dpu_runner, processed_image)
        model_time2 = time.perf_counter()
        model_time = (model_time2-model_time1)*1000
        print(f'Time taken by model: {model_time:.3f}ms.')

        post_time1 = time.perf_counter()
        result = inference_own(predictions=prediction, score_thres=0.3, nms_thres=0.4)
        post_time2 = time.perf_counter()
        post_time = (post_time2-post_time1)*1000
        print(f'Time taken by post processing : {post_time:.3f}ms.')

        elapsed_time = pre_time + model_time + post_time
        print(f'elapsed time: {elapsed_time} ms.') 
        FPS = 1000/elapsed_time
        # print(f'FPS : {FPS:.3f}') # since elapsed time is in milisecond.
        display(result, processed_image, FPS)
        print('DONE!!')
        
if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python3 inference_yolov5.py <yolov5.xmodel> <path/to/image/directory>"
    main(sys.argv)
