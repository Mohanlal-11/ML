import json
import sys
import time
import logging
from tqdm import tqdm
import datetime
from pathlib import Path

import torch

from .model import box_ops
from .model.transform import Transformer

from . import distributed
from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass

logger = logging.getLogger()

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

def train_one_epoch(model, optimizer, data_loader, device, epoch, args, ema, mosaic):    
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters
    num_iters = epoch * len(data_loader)

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    o_m = Meter("optimizer")
    e_m = Meter("ema")
    model.train()
    A = time.time()
    bar = tqdm(data_loader)
    for i, data in enumerate(bar):
        T = time.time()
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            args.accumulate = max(1, round(r * (64 / args.batch_size - 1) + 1))
            for j, p in enumerate(optimizer.param_groups):
                init = 0.1 if j == 0 else 0 # 0: biases
                p["lr"] = r * (args.lr_epoch - init) + init
                p["momentum"] = r * (args.momentum - 0.9) + 0.9
                   
        images = data.images
        targets = data.targets

        transformer = Transformer(min_size=args.input_size[0], max_size=args.input_size[1], stride=max(strides), mosaic=mosaic)
        transformer.to(device).train()
        images, targets, _, _, _ = transformer(images, targets)

        S = time.time()
        if args.amp:
            with torch.cuda.amp.autocast():
                losses = model(images.to(device), targets)
        else:
            losses = model(images.to(device), targets)
            
        # if num_iters % args.print_freq == 0:
        losses_list = [l.item() for l in losses.values()]
        info = f"[Train] Epoch-{epoch}: Loss_box:%f Loss_obj:%f Loss_cls:%f"%(losses_list[0], losses_list[1], losses_list[2])
        bar.set_description(info)
        logger.info(info)
            
        losses = {k: v * args.batch_size for k, v in losses.items()}
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)

        #losses_reduced = distributed.reduce_dict(losses)
        #total_loss_reduced = sum(losses_reduced.values())

        if not torch.isfinite(total_loss):
            logger.info("Loss is {}, stopping training".format(total_loss.item()))
            logger.info("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))
            sys.exit(1)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        if num_iters % args.accumulate == 0:
            S = time.time()
            optimizer.step()
            optimizer.zero_grad()
            o_m.update(time.time() - S)
            
            S = time.time()
            ema.update(model)
            e_m.update(time.time() - S)

        t_m.update(time.time() - T)
        
        num_iters += 1
        if i >= iters - 1:
            break
           
    A = time.time() - A
    logger.info("iter: {:.1f}, total: {:.1f}, model: {:.1f}, ".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    logger.info("backward: {:.1f}, optimizer: {:.1f}, ema: {:.1f}".format(1000*b_m.avg,1000*o_m.avg,1000*e_m.avg))
    return (m_m.sum + b_m.sum + o_m.sum + e_m.sum) / iters
            

def evaluate(model, data_loader, device, args, generate=True, evaluation=True, merge=False):
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, args, merge)
      
    output = ""
    if distributed.get_rank() == 0 and evaluation:
        dataset = data_loader.dataset
        coco_evaluator = CocoEvaluator(dataset.coco)

        results = json.load(open(args.results))

        S = time.time()
        coco_evaluator.accumulate(results)
        print("accumulate: {:.1f}s".format(time.time() - S))
        
        # collect the output of builtin function "print"
        temp = sys.stdout
        sys.stdout = TextArea()

        coco_evaluator.summarize()

        output = sys.stdout
        sys.stdout = temp
        
    if hasattr(args, "distributed") and args.distributed:
        torch.distributed.barrier()
    return output, iter_eval
    

def inference(preds, image_shapes, scale_factors, box_map, max_size, score_thresh, nms_thresh, device, merge=True):
    anchors_tens = torch.tensor(anchors)
    anchors_tens = anchors_tens.to(device)
    # print(anchors_tens.device, preds[0].device, device)
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
            # print("in nms",box.shape)
            box[:, ::2] /= box_map[i][1]
            box[:, 1::2] /= box_map[i][0] 
        results.append(dict(boxes=box, labels=label, scores=score)) # boxes format: (xmin, ymin, xmax, ymax)
        
    return results
  
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args, merge):
    iters = len(data_loader) if args.iters < 0 else args.iters
    ann_labels = data_loader.dataset.ann_labels
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, data in enumerate(tqdm(data_loader,desc="Evaluation")):
        T = time.time()
        
        images = data.images
        targets = data.targets

        transformer = Transformer(min_size=args.input_size[0], max_size=args.input_size[1], stride=max(strides))
        transformer.to(device).eval()
        images, targets, scale_factors, image_shapes, box_map = transformer(images, targets)
        # print(f'box map:{box_map}')
        # print(f'gen res : {images.shape}')
        max_size = max(images.shape[2:])
        # print(f'max size : {max_size}')
        # print(f'image shapes: {image_shapes}')
        #torch.cuda.synchronize()
        S = time.time()
        if args.amp:
            with torch.cuda.amp.autocast():
                preds = model(images.to(device), targets)
                if args.batch_size == 1:
                    return
        else:
            preds = model(images.to(device), targets)
            if args.batch_size == 1:
                return
            
        outputs = inference(preds, image_shapes, scale_factors, box_map, max_size, args.score_thres, args.nms_thres, device, merge)
        m_m.update(time.time() - S)
        
        # if losses and i % 10 == 0:
        #     print("{}\t".format(i), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))
            
        outputs = [{k: v.cpu() for k, v in out.items()} for out in outputs]
        predictions = {tgt["image_id"].item(): out for tgt, out in zip(targets, outputs)}
        coco_results.extend(prepare_for_coco(predictions, ann_labels))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    
    S = time.time()
    all_results = distributed.all_gather(coco_results)
    print("all gather: {:.1f}s".format(time.time() - S))
    
    merged_results = []
    for res in all_results:
        merged_results.extend(res)
        
    if distributed.get_rank() == 0:
        json.dump(merged_results, open(args.results, "w"))
        
    return m_m.sum / iters
    
