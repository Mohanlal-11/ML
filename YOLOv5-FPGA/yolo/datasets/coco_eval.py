import copy
import torch
import numpy as np

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]
            
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                         for iou_type in iou_types}
        self.has_results = False
        self.metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large", "AR_maxdet1", "AR_maxdet10", "AR_maxdet100", "AR_small", "AR_medium", "AR_large"]
    def accumulate(self, coco_results):
        if len(coco_results) == 0:
            return
            
        image_ids = list(set([res["image_id"] for res in coco_results]))
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_dt = self.coco_gt.loadRes(coco_results) if coco_results else COCO() # use the method loadRes

            coco_eval.cocoDt = coco_dt 
            coco_eval.params.imgIds = image_ids # ids of images to be evaluated
            coco_eval.evaluate() # 15.4s
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            coco_eval.accumulate() # 3s
            
            precisions = coco_eval.eval["precision"]
            eval_results = {}

            recalls = coco_eval.eval['recall']        # [T, K, A, M]

            aps = coco_eval.stats[:12]
            for k, v in zip(self.metric_names, aps):
                eval_results[k] = v
                
            # Use area range index 0 (all), maxDets index -1 (usually 100)
            precision_vals = precisions[:, :, :, 0, -1]
            recall_vals = recalls[:, :, 0, -1]

            # Filter out invalid entries (-1)
            valid_precision = precision_vals[precision_vals > -1]
            valid_recall = recall_vals[recall_vals > -1]

            if valid_precision.size > 0 and valid_recall.size > 0:
                mean_precision = np.mean(valid_precision)
                mean_recall = np.mean(valid_recall)
                f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-6)
                # logger.info(f"F1 Score: {f1_score:.4f}")
                eval_results["Mean Precision"] = float(mean_precision)
                eval_results["Mean Recall"] = float(mean_recall)
                eval_results["F1-score"] = float(f1_score)
            else:
                # logger.warning("Unable to compute F1 Score due to insufficient valid precision/recall values.")
                eval_results["F1-score"] = 0.0
            
            print(f'[INFO] Evaluation result: {eval_results}')
                
        self.has_results = True
    
    def summarize(self):
        if self.has_results:
            for iou_type in self.iou_types:
                print("IoU metric: {}".format(iou_type))
                self.coco_eval[iou_type].summarize()
        else:
            print("evaluation has no results")

            
def prepare_for_coco(predictions, ann_labels):
    coco_results = []
    for image_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        # convert to coco bbox format: xmin, ymin, w, h
        boxes = prediction["boxes"]
        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)

        boxes = boxes.tolist()

        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        labels = [ann_labels[l] for l in labels]

        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

