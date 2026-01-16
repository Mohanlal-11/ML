# -----------------------------------------------------------------------------------------------------
import torch
import math
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from yolo.datasets import utils


torch.set_printoptions(precision=1)

def parse_dataset(ds, img_sizes):
    info = {}
    # print(ds)
    ids = [int(id_) for id_ in ds.ids]
    # print(f'ids: {ids}')
    min_size, max_size = img_sizes
    factor = lambda s: min(min_size / min(s), max_size / max(s))
    scales, pixels = [], []
    for id_ in ids: 
        v = ds.coco.imgs[id_]
        h, w = v["height"], v["width"]
        scale = factor((h, w))
        scales.append(scale)
        pixels.append(round(scale * h) * round(scale * w))

    boxes = torch.cat([ds.get_target(id_)["boxes"] * scale
                       for id_, scale in zip(ds.ids, scales)])
    
    info["boxes"] = boxes
    info["scales"] = scales
    info["pixels"] = pixels
    return info


def wh_iou(wh1, wh2):
    area1 = torch.prod(wh1, dim=1)
    area2 = torch.prod(wh2, dim=1)
    wh = torch.min(wh1[:, None], wh2[None])
    inter = torch.prod(wh, dim=2)
    return inter / (area1[:, None] + area2 - inter)


def kmeans(boxes, k, n=100):
    num = len(boxes)
    last_clusters = torch.zeros((num,))
    rng = torch.randperm(num)[:k]
    clusters = boxes[rng]

    for _ in range(n):
        #iou = wh_iou(boxes, clusters)
        #avg_iou = torch.max(iou, dim=1)[0].mean().item()
        #print("{:.1f} ".format(100 * avg_iou), end="")
        
        nearest_clusters = matched_fn(boxes, clusters)[1]

        if (last_clusters == nearest_clusters).all():
            #print("nice ", end="")
            break

        for i in range(k):
            order = nearest_clusters == i
            clusters[i] = torch.median(boxes[order], dim=0)[0]
        last_clusters = nearest_clusters
            
    iou = wh_iou(boxes, clusters)
    avg_iou = torch.max(iou, dim=1)[0].mean().item()
    return clusters, avg_iou


def matched_fn(wh1, wh2):
    ratios = wh1[:, None] / wh2[None]
    max_ratios = torch.max(ratios, 1 / ratios).max(2)[0]
    return max_ratios.min(1)


def lazy_fn(boxes, k=3, n=5, iters=200, thresh=4):
    results = []
    for _ in range(n):
        clusters, avg_iou = kmeans(boxes, k, iters)
        area = clusters.prod(1)
        order = area.sort()[1]
        clusters = clusters[order]
        
        values = matched_fn(boxes, clusters)[0]
        bpr = (values < thresh).float().mean()
        
        left = len(boxes) * (1 - bpr)
        results.append((clusters, round(100 * avg_iou, 2), round(100 * bpr.item(), 2)))
    return results


def auto_anchors(ds, img_sizes=[320, 416], levels=3, ks=3, stride=4, **kwargs):
    if isinstance(img_sizes, int):
        img_sizes = [img_sizes, img_sizes]
    if isinstance(ks, int):
        ks = [ks] * levels
    assert len(ks) == levels, "len(ks) != levels"
    print(img_sizes, stride)
    
    info = parse_dataset(ds, img_sizes)
    boxes = info["boxes"]
    wh = torch.stack((boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), dim=1)
    
    new_boxes = wh[(wh >= 2).all(1)]
    print(f"len(new_boxes)={len(new_boxes)}")
    areas = new_boxes.prod(1)

    chunked_areas = areas.sort()[0].chunk(levels)
    print("before split:", [len(x) for x in chunked_areas])
    sep = lambda x: math.ceil(x.sqrt().item() / stride) * stride
    sizes = [0] + [sep(chunked_areas[i][-1]) for i in range(levels - 1)] + [10000]
    #sizes = [0, 72, 180, 10000]
    print("split size:", sizes)
    
    orders = [(areas >= s ** 2) & (areas < sizes[i + 1] ** 2)
              for i, s in enumerate(sizes[:-1])]
    print("after split:", [x.sum().item() for x in orders])

    anchors = []
    for i in range(levels):
        boxes_part = new_boxes[orders[i]]
    
        res = lazy_fn(boxes_part, ks[i], **kwargs)
        res.sort(key=lambda x: x[2])
        # print(f"\nlevel {i + 1}:")
        # print(*res[-1])
        anchors.append(res[-1][0].tolist())
    
    with open('anchors.txt', 'w') as fw:
        fw.write(str(anchors))
        print(f'[INFO] The generated anchors are saved to file anchors.txt')
        
    print("\ntotal: [")
    for i, anchor in enumerate(anchors):
        print("    [", end="")
        for j, anc in enumerate(anchor):
            print([round(a, 1) for a in anc], end=", " if j < len(anchor) - 1 else "")
        print("],")
    print("]")

if __name__ == "__main__":
    input_img_dim = [320, 320]
    ds = utils.datasets("coco", "/media/logictronix01/ML_WORKSPACE/coco2017_dataset", "/media/logictronix01/ML_WORKSPACE/coco2017_dataset/annotations/instances_train2017_PersonOnly_16.json", train=False)
    auto_anchors(ds, input_img_dim)
