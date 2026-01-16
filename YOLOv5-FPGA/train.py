import argparse
import math
import os
import re
import time
import datetime
import logging
from pathlib import Path
import torch

import yolo

train_logs = Path('logs')
time_stamp = f'train_logs_{datetime.datetime.now().strftime("%Y-%m%d_%H-%M-%S")}'
train_logs = train_logs / time_stamp
train_logs.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(train_logs / 'training.log'),
                    logging.StreamHandler()
                ])
logger = logging.getLogger()  

def main(args):    
    # Prepare for distributed training
    yolo.init_distributed_mode(args)
    begin_time = time.time()
    print(time.asctime(time.localtime(begin_time)))
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda: yolo.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    
    # Automatic mixed precision
    args.amp = False
    if cuda and torch.__version__ >= "1.6.0":
        capability = torch.cuda.get_device_capability()[0]
        if capability >= 7: # 7 refers to RTX series GPUs, e.g. 2080Ti, 2080, Titan RTX
            args.amp = True
            logger.info("Automatic mixed precision (AMP) is enabled!")
        
    # ---------------------- prepare data loader ------------------------------- #
    
    # NVIDIA DALI, much faster data loader.
    DALI = cuda & yolo.DALI & args.dali & (args.dataset == "coco")
    # The code below is for COCO 2017 dataset
    # If you're using VOC dataset or COCO 2012 dataset, remember to revise the code
    splits = ("train2017", "train2017")
    file_roots = [os.path.join(args.data_dir, "images", x) for x in splits]
    ann_files = [os.path.join(args.data_dir, "annotations/instances_{}_PersonOnly_16.json".format(x)) for x in splits]
    print(f'Image root and annotation file root: {file_roots, ann_files}')
    if DALI:
        # Currently only support COCO dataset; support distributed training
        
        # DALICOCODataLoader behaves like PyTorch's DataLoader.
        # It consists of Dataset, DataLoader and DataPrefetcher. Thus it outputs CUDA tensor.
        logger.info("Nvidia DALI is utilized!")
        d_train = yolo.DALICOCODataLoader(
            file_roots[0], ann_files[0], args.batch_size, collate_fn=yolo.collate_wrapper,
            drop_last=True, shuffle=True, device_id=args.gpu, world_size=args.world_size)
        
        d_test = yolo.DALICOCODataLoader(
            file_roots[1], ann_files[1], args.batch_size, collate_fn=yolo.collate_wrapper, 
            device_id=args.gpu, world_size=args.world_size)
    else:
        # transforms = yolo.RandomAffine((0, 0), (0.1, 0.1), (0.9, 1.1), (0, 0, 0, 0))
        dataset_train = yolo.datasets(args.dataset, file_roots[0], ann_files[0], train=True)
        dataset_test = yolo.datasets(args.dataset, file_roots[1], ann_files[1], train=True) # set train=True for eval

        if args.distributed:
            sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
            sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        batch_sampler_train = yolo.GroupedBatchSampler(
            sampler_train, dataset_train.aspect_ratios, args.batch_size, drop_last=True)
        batch_sampler_test = yolo.GroupedBatchSampler(
            sampler_test, dataset_test.aspect_ratios, args.batch_size)

        # args.num_workers = min(os.cpu_count() // 2, 8, args.batch_size if args.batch_size > 1 else 0)
        args.num_workers = 0
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
            collate_fn=yolo.collate_wrapper, pin_memory=cuda)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_sampler=batch_sampler_test, num_workers=args.num_workers,  
            collate_fn=yolo.collate_wrapper, pin_memory=cuda)

        # cuda version of DataLoader, it behaves like DataLoader, but faster
        # DataLoader's pin_memroy should be True
        d_train = yolo.DataPrefetcher(data_loader_train) if cuda else data_loader_train
        d_test = yolo.DataPrefetcher(data_loader_test) if cuda else data_loader_test
        
    # args.warmup_iters = max(1000, 3 * len(d_train))
    args.warmup_iters = args.warmup_epoch*len(d_train)
    
    # -------------------------------------------------------------------------- #

    print(args)
    yolo.setup_seed(args.seed)
    
    model_sizes = {"nano":(0.33, 0.25), "small": (0.33, 0.5), "medium": (0.67, 0.75), "large": (1, 1), "extreme": (1.33, 1.25)}
    num_classes = len(d_train.dataset.classes)
    print(f'Number of classes in train dataset:{num_classes}')
    model = yolo.YOLOv5(num_classes, model_sizes[args.model_size], img_sizes=args.input_size).to(device)
    # model.transformer.mosaic = args.mosaic
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    params = {"conv_weights": [], "biases": [], "others": []}
    for n, p in model_without_ddp.named_parameters():
        if p.requires_grad:
            if p.dim() == 4:
                params["conv_weights"].append(p)
            elif ".bias" in n:
                params["biases"].append(p)
            else:
                params["others"].append(p)

    args.accumulate = max(1, round(64 / args.batch_size))
    wd = args.weight_decay * args.batch_size * args.accumulate / 64
    optimizer = torch.optim.SGD(params["biases"], lr=args.lr, momentum=args.momentum, nesterov=True)
    optimizer.add_param_group({"params": params["conv_weights"], "weight_decay": wd})
    optimizer.add_param_group({"params": params["others"]})

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)

    lr_lambda = lambda x: math.cos(math.pi * x / ((x // args.period + 1) * args.period) / 2) ** 2 * 0.9 + 0.1

    logger.info("Optimizer param groups: ")
    logger.info(", ".join("{} {}".format(len(v), k) for k, v in params.items()))
    del params
    if cuda: torch.cuda.empty_cache()
       
    ema = yolo.ModelEMA(model)
    ema_without_ddp = ema.ema.module if args.distributed else ema.ema
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device) # load last checkpoint
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        ema_without_ddp.load_state_dict(checkpoint["ema"][0])
        ema.updates = checkpoint["ema"][1]
        del checkpoint
        if cuda: torch.cuda.empty_cache()
        logger.info(f"Training resume from last weight from {args.resume}")

    if args.pretrained:
        model_state_dict = model.state_dict()
        state_dict = torch.load(args.pretrained)
        state_dict = state_dict["model"]
        # Filter out keys that don't match shape
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")
        model.load_state_dict(filtered_state_dict, strict=False) 
        logger.info(f"Using pretrained weights from path {args.pretrained}") 

    if args.pretrained is None and args.resume is None:
        logger.info("Training from scratch.")  
    since = time.time()
    logger.info("Already trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
    best = 0 
    for epoch in range(start_epoch, args.epochs):
        # logger.info("Epoch: {} \t".format(epoch + 1))
        
        if not DALI and args.distributed:
            sampler_train.set_epoch(epoch)
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        logger.info("lr_epoch: {:.4f}, factor: {:.4f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = yolo.train_one_epoch(model, optimizer, d_train, device, epoch, args, ema, args.mosaic)
        A = time.time() - A
        
        last_wts_path = os.path.join(args.ckpt_path.split('/')[0], "model_last_wts.pth")
        yolo.save_ckpt(
            model_without_ddp, optimizer, epoch+1, last_wts_path,
            val=False, ema=(ema_without_ddp.state_dict(), ema.updates))
        
        if (epoch+1)%args.val_epoch==0:
            B = time.time()
            eval_output, iter_eval = yolo.evaluate(ema.ema, d_test, device, args)
            B = time.time() - B
            # print(type(eval_output), eval_output)
            trained_epoch = epoch + 1
            if yolo.get_rank() == 0:
                logger.info("Training: {:.2f} s, Evaluation: {:.2f} s".format(A, B))
                yolo.collect_gpu_info("yolov5s", [args.batch_size / iter_train, args.batch_size / iter_eval])
                logger.info(eval_output.get_AP())
                
                # AP = eval_output.get_AP()
                # if type(AP) == dict and eval_output.get_AP()['bbox AP'] > best:
                yolo.save_ckpt(
                    model_without_ddp, optimizer, trained_epoch, args.ckpt_path,
                    eval_info=str(eval_output), ema=(ema_without_ddp.state_dict(), ema.updates))

                # It will create many checkpoint files during training, so delete some.
                ckpts = yolo.find_ckpts(args.ckpt_path)
                remaining = 60
                if len(ckpts) > remaining:
                    for i in range(len(ckpts) - remaining):
                        os.system("rm {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    logger.info("Total time of this training: {:.2f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true") # whether use the GPU
    
    parser.add_argument("--dataset", default="coco") # style of dataset, choice: ["coco", "voc"]
    parser.add_argument("--data-dir", default="") # root directory of the dataset
    parser.add_argument("--dali", action="store_true") # NVIDIA's DataLoader, faster but without random affine
    parser.add_argument("--ckpt-path") # ckpts name
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--results") # path where to save the evaluation results
    parser.add_argument("--pretrained", type=str, default="")
    
    # you may not train the model for 273 epochs once, and want to split it into several tasks.
    # set epochs={the target epoch of each training task}
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--val_epoch", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    # total epochs. iterations=500000, true batch size=64, so total epochs=272.93
    parser.add_argument("--period", type=int, default=273) 
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=-1) # max iterations per epoch, -1 denotes an entire epoch
    
    parser.add_argument("--seed", type=int, default=3) # random seed
    parser.add_argument("--model-size", default="nano") # choice: ["small", "medium", "large", "extreme"]
    parser.add_argument('--input_size', nargs="+", type=int, default=[320, 320]) # range of input images' max_size during training
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)

    parser.add_argument("--score_thres", type=float, default=0.1)
    parser.add_argument("--nms_thres", type=float, default=0.6)
    parser.add_argument("--warmup_epoch", type=int, default=20)
    
    parser.add_argument("--mosaic", action="store_true") # mosaic data augmentaion, increasing ~2% AP, a little slow
    parser.add_argument("--print-freq", type=int, default=100) # frequency of printing losses during training
    parser.add_argument("--world-size", type=int, default=1) # total number of processes
    parser.add_argument("--dist-url", default="env://") # distributed initial method
    
    parser.add_argument("--root") # gpu cloud platform special
    args = parser.parse_args()
    
    if args.ckpt_path is None:
        ckpt_dir = Path("weights")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        args.ckpt_path = os.path.join(ckpt_dir, f"yolov5_{args.model_size}.pth")
        
    if args.results is None:
        eval_dir = Path("Evaluation Results")
        eval_dir.mkdir(parents=True,exist_ok=True)
        args.results = os.path.join(eval_dir, "results.json")
    print(args.results)
        
    begin_time = time.time()
    print("{}.txt".format(int(begin_time)))
    
    main(args)
    
    # for gpu rent platform
    if yolo.get_rank() == 0:
        if args.root:
            os.system("mv {}data/logs/log.txt {}data/logs/{}.txt".format(args.root, args.root, int(begin_time)))
            prefix, ext = os.path.splitext(args.ckpt_path)
            os.system("cp {}-{}{} {}data/ckpts/".format(prefix, args.epochs, ext, args.root)) # copy last checkpoint
            #os.system("{}root/shutdown.sh".format(args.root)) # for jikecloud only
    print("All over!")
    
    
