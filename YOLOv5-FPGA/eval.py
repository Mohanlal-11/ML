import argparse
import os
import time
import torch
import yolo

    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda: yolo.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    
    args.amp = False
    if cuda and torch.__version__ >= "1.6.0":
        capability = torch.cuda.get_device_capability()[0]
        if capability >= 7: # 7 refers to RTX series GPUs
            args.amp = True
            print("Automatic mixed precision (AMP) is enabled!")
            
    # ---------------------- prepare data loader ------------------------------- #
    
    DALI = cuda & yolo.DALI & (args.dataset == "coco")
    
    if DALI:
        print("Nvidia DALI is utilized!")
        d_test = yolo.DALICOCODataLoader(
            args.file_root, args.ann_file, args.batch_size, collate_fn=yolo.collate_wrapper)
    else:
        dataset_test = yolo.datasets(args.dataset, args.file_root, args.ann_file, train=True) # set train=True for eval
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        batch_sampler_test = yolo.GroupedBatchSampler(
            sampler_test, dataset_test.aspect_ratios, args.batch_size)
        
        args.num_workers = 0
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_sampler=batch_sampler_test, num_workers=args.num_workers,  
            collate_fn=yolo.collate_wrapper, pin_memory=cuda)

        d_test = yolo.DataPrefetcher(data_loader_test) if cuda else data_loader_test
    
    # -------------------------------------------------------------------------- #

    yolo.setup_seed(3)
    
    model_sizes = {"nano":(0.33, 0.25), "small": (0.33, 0.5), "medium": (0.67, 0.75), "large": (1, 1), "extreme": (1.33, 1.25)}
    num_classes = len(d_test.dataset.classes)
    model = yolo.YOLOv5(num_classes, model_sizes[args.model_size], **args.kwargs).to(device)
    model.head.eval_with_loss = args.eval_with_loss
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"][0])
        print(checkpoint["eval_info"])
    else:
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in checkpoint.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")

        print(f'[INFO] The len of wts keys in filtered wts and own model is : {len(filtered_state_dict.keys()), len(model_state_dict.keys())}')
        model.load_state_dict(filtered_state_dict, strict=False)

    model.fuse()
    print("evaluating...")
    B = time.time()
    eval_output, iter_eval = yolo.evaluate(model, d_test, device, args, evaluation=args.evaluation)
    B = time.time() - B
    print(eval_output)
    print("\ntotal time of this evaluation: {:.2f} s, speed: {:.2f} FPS".format(B, args.batch_size / iter_eval))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args([]) # [] is needed when using Jupyter Notebook.
    
    args.use_cuda = True
    
    args.dataset = "coco"
    args.file_root = "coco2017_dataset/images/train2017"
    args.ann_file = "coco2017_dataset/annotations/instances_train2017_PersonOnly_16.json"
    args.ckpt_path = "weights/yolov5_nano-1000.pth"
    args.results = os.path.join(os.path.dirname(args.ckpt_path), "results.json")
    
    args.batch_size = 32
    args.iters = -1
    
    args.model_size = "nano"
    args.input_size = [320,320]
    args.score_thres, args.nms_thres = 0.1,0.6
    args.kwargs = {"img_sizes": args.input_size} # mAP 34.6 FPS 451
    #args.kwargs = {"img_sizes": 672, "score_thresh": 0.001, "detections": 300} # mAP 36.1. take more(2x-4x) time in total
    args.evaluation = True
    args.eval_with_loss = False
    
    main(args)
    
    