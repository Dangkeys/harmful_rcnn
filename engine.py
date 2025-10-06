import math
import os
import sys
import time
import copy
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from tqdm.auto import tqdm

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} Train", leave=False, total=len(data_loader))
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    
    # --- Metrics for TQDM ---
    avg_loss = utils.SmoothedValue(window_size=print_freq, fmt="{value:.4f}")
    avg_lr = utils.SmoothedValue(window_size=1, fmt="{value:.6f}")

    # metric_logger.log_every(data_loader, print_freq, header)
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.amp.autocast('cuda', enabled=scaler is not None):
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
        
        # --- Update and display TQDM postfix metrics ---
        avg_loss.update(losses_reduced.item())
        avg_lr.update(optimizer.param_groups[0]["lr"])
        
        # Format the loss dictionary for display
        loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict_reduced.items()])
        
        pbar.set_postfix(loss=avg_loss.fmt.format(value=avg_loss.avg), 
                         lr=avg_lr.fmt.format(value=avg_lr.avg),
                         raw_losses=f"({loss_str})")

    #     metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
    #     metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # return metric_logger

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
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # Use tqdm to wrap the data_loader for evaluation

    pbar = tqdm(data_loader, desc=f"Evaluation", leave=False, total=len(data_loader))

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # --- Metrics for TQDM ---
    avg_model_time = utils.SmoothedValue(window_size=pbar.miniters, fmt="{avg:.3f}s")
    avg_eval_time = utils.SmoothedValue(window_size=pbar.miniters, fmt="{avg:.3f}s")

    for images, targets in pbar:
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
        # --- Update and display TQDM postfix metrics ---
        avg_model_time.update(model_time)
        avg_eval_time.update(evaluator_time)
        
        pbar.set_postfix(model_t=avg_model_time.fmt.format(avg=avg_model_time.avg),
                         eval_t=avg_eval_time.fmt.format(avg=avg_eval_time.avg))

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

def kfold_train(args, model, dataset):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    indices = torch.arange(len(dataset)).tolist()
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed) 
    
    all_fold_evaluators = []
    kfold_data = {'summary': {}}

    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"\n{'='*20} Start Fold {fold+1}/{args.k_folds} {'='*20}")

        dataset_train = Subset(dataset, train_indices)
        dataset_val = Subset(dataset, val_indices)
        
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
        
        scaler = torch.amp.GradScaler('cuda') if args.amp else None
        
        best_map = -1.0
        best_evaluator = None
        best_metrics = {}

        for epoch in range(args.epochs):
            print(f"\n--- Fold {fold+1}, Epoch {epoch+1}/{args.epochs} ---")
            
            train_one_epoch(model, optimizer, train_data_loader, device, epoch, args.print_freq, scaler)
            
            print("Running Evaluation for current epoch...")
            coco_evaluator = evaluate(model, val_data_loader, device=device)
            
            current_map = coco_evaluator.coco_eval['bbox'].stats[0] 
            
            if current_map > best_map:
                best_map = current_map
                best_evaluator = copy.deepcopy(coco_evaluator)
                best_metrics = extract_coco_metrics(coco_evaluator) 

        print(f"\nFold {fold+1} finished. Best mAP: {best_map:.4f}")

        if best_evaluator:
            all_fold_evaluators.append(best_evaluator)
            kfold_data['summary'][f'Fold {fold+1}'] = best_metrics
            
    print(f"\n\n{'='*20} K-Fold Summary (N={args.k_folds}) {'='*20}")
    
    if kfold_data['summary']:
        maps = [d['mAP50-95'] for d in kfold_data['summary'].values()] # Use mAP50-95 for overall mean
        mean_map = np.mean(maps)
        std_map = np.std(maps)
        
        print("\nIndividual Fold Results:")
        for k, v in kfold_data['summary'].items():
            print(f"- {k}: {v:.4f}")
        
        print(f"\n--- Final Cross-Validation Result ---")
        print(f"Mean Best mAP: {mean_map:.4f} \u00B1 {std_map:.4f}")
        
        for i, evaluator in enumerate(all_fold_evaluators):
            print(f"--- Full Summary for Fold {i+1} ---")
            evaluator.summarize()

    return kfold_data
# Helper function to get the mAP and AR metrics from CocoEvaluator
def extract_coco_metrics(coco_evaluator, iou_type='bbox'):
    """Extracts key COCO metrics from a CocoEvaluator object."""
    stats = coco_evaluator.coco_eval[iou_type].stats

    AP50 = stats[1]
    AR100 = stats[8] # AR @ maxDets=100
    
    # Simplified F1-score placeholder (F1 is not standard COCO output):
    f1_proxy = (2 * AP50 * AR100) / (AP50 + AR100) if (AP50 + AR100) > 0 else 0 

    metrics = {
        'mAP50-95': stats[0],
        'mAP50': stats[1],
        'mAP75': stats[2],
        'AR100': stats[8],
        'f1_score': f1_proxy
    }
    return metrics

def plot_graph_kfold(kfold_data: dict):
    summary_df = pd.DataFrame.from_dict(kfold_data['summary'], orient='index')
    summary_df.index = [i for i in range(1, len(summary_df) + 1)] 
    summary_df.index.name = 'Fold'

    print("\n--- K-Fold Metrics Summary ---")
    print(summary_df)

    mean_metrics = summary_df.mean()
    std_metrics = summary_df.std()

    print("\n--- Average Metrics ---")
    print(f"Mean mAP50: {mean_metrics['mAP50']:.4f} +/- {std_metrics['mAP50']:.4f}")
    print(f"Mean mAP50-95: {mean_metrics['mAP50-95']:.4f} +/- {std_metrics['mAP50-95']:.4f}")


    # 3. Plotting the mAP50 results per fold
    plt.figure(figsize=(10, 6))

    summary_df['mAP50'].plot(
        kind='bar', 
        capsize=5, 
        color='skyblue'
    )

    # Draw a line for the average mAP50
    plt.axhline(mean_metrics['mAP50'], color='red', linestyle='--', label=f'Average mAP50: {mean_metrics["mAP50"]:.4f}')

    plt.title('Validation mAP@0.5 Across K-Folds')
    plt.xlabel('Fold Number')
    plt.ylabel('mAP@0.5')
    plt.xticks(rotation=0)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    plt.tight_layout()
    SAVE_FILE_NAME_1 = 'kfold_map50_results.png'
    plt.savefig(SAVE_FILE_NAME_1) 
    print(f"\nPlot saved successfully as: {os.path.abspath(SAVE_FILE_NAME_1)}")


    # --- PLOT 2: mAP@0.5:0.95 (New Plot) ---
    plt.figure(figsize=(10, 6))

    summary_df['mAP50-95'].plot(
        kind='bar', 
        capsize=5, 
        color='lightcoral' # Use a different color
    )

    # Draw a line for the average mAP50-95
    plt.axhline(mean_metrics['mAP50-95'], color='darkred', linestyle='--', label=f'Average mAP50-95: {mean_metrics["mAP50-95"]:.4f}')

    plt.title('Validation mAP@0.5:0.95 Across K-Folds')
    plt.xlabel('Fold Number')
    plt.ylabel('mAP@0.5:0.95')
    plt.xticks(rotation=0)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    plt.tight_layout()
    SAVE_FILE_NAME_2 = 'kfold_map50-95_results.png'
    plt.savefig(SAVE_FILE_NAME_2) 
    print(f"\nPlot saved successfully as: {os.path.abspath(SAVE_FILE_NAME_2)}")

    # --- PLOT 3: F1 Score ---
    plt.figure(figsize=(10, 6))

    summary_df['f1_score'].plot(
        kind='bar', 
        capsize=5, 
        color='darkgreen' # New color for F1 score
    )

    # Draw a line for the average F1 Score
    plt.axhline(mean_metrics['f1_score'], color='blue', linestyle='--', label=f'Average F1 Score: {mean_metrics["f1_score"]:.4f}')

    plt.title('Validation F1 Score Across K-Folds')
    plt.xlabel('Fold Number')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=0)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    plt.tight_layout()
    SAVE_FILE_NAME_3 = 'kfold_f1_score_results.png'
    plt.savefig(SAVE_FILE_NAME_3) 
    print(f"Plot saved successfully as: {os.path.abspath(SAVE_FILE_NAME_3)}")

    plt.show()