import math
import sys
import time
import copy

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

class Args:
    def __init__(self):
        self.k_folds = 5
        self.epochs = 3 # Number of epochs PER FOLD
        self.num_classes = 4 # Background + 3 classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 4
        self.workers = 0
        self.lr = 0.02
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.print_freq = 50
        self.amp = False
        self.seed = 42
        self.data_path = './data'

def kfold_train(args, model, data_loader):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    indices = torch.arange(len(data_loader)).tolist()
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed) 
    
    all_fold_evaluators = []
    fold_results_summary = {}

    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"\n{'='*20} Start Fold {fold+1}/{args.k_folds} {'='*20}")

        dataset_train = Subset(data_loader, train_indices)
        dataset_val = Subset(data_loader, val_indices)
        
        train_data_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.workers, collate_fn=utils.collate_fn 
        )
        val_data_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.workers, collate_fn=utils.collate_fn
        )

        model.to(device)
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
        
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        
        best_map = -1.0
        best_evaluator = None

        for epoch in range(args.epochs):
            print(f"\n--- Fold {fold+1}, Epoch {epoch+1}/{args.epochs} ---")
            
            train_one_epoch(model, optimizer, train_data_loader, device, epoch, args.print_freq, scaler)
            
            print("Running Evaluation for current epoch...")
            coco_evaluator = evaluate(model, val_data_loader, device=device)
            
            current_map = coco_evaluator.stats['bbox'][0]
            
            if current_map > best_map:
                best_map = current_map
                best_evaluator = copy.deepcopy(coco_evaluator)

        print(f"\nFold {fold+1} finished. Best mAP: {best_map:.4f}")

        if best_evaluator:
            all_fold_evaluators.append(best_evaluator)
            fold_results_summary[f'Fold_{fold+1}_Best_mAP'] = best_map
            
    print(f"\n\n{'='*20} K-Fold Summary (N={args.k_folds}) {'='*20}")
    
    if fold_results_summary:
        maps = list(fold_results_summary.values())
        mean_map = np.mean(maps)
        std_map = np.std(maps)
        
        print("\nIndividual Fold Results:")
        for k, v in fold_results_summary.items():
            print(f"- {k}: {v:.4f}")
        
        print(f"\n--- Final Cross-Validation Result ---")
        print(f"Mean Best mAP: {mean_map:.4f} \u00B1 {std_map:.4f}")
        
        for i, evaluator in enumerate(all_fold_evaluators):
            print(f"--- Full Summary for Fold {i+1} ---")
            evaluator.summarize()

    return all_fold_evaluators