
from __future__ import annotations
import argparse
import json
import os

from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm.auto import tqdm


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.ops import nms

from utils.dataloader import CellInstanceDataset
from utils.model import MaskRCNN
from utils.engine import train_one_epoch, evaluate, masks_to_rles
from utils.utils import plot_loss_ap, set_seed, collate_fn, visualize_predictions

loss_lr_history = []
ap_history = []


def train(args):  
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}, {torch.cuda.get_device_name(0)}")

    complete_dataset = CellInstanceDataset(args.data_root, split="train")
    train_len = int(0.9 * len(complete_dataset))
    val_len = len(complete_dataset) - train_len
    train_dataset, val_dataset = random_split(complete_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn,)

    model = MaskRCNN(num_classes=5)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)  
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.8*args.epochs), int(0.9*args.epochs)], gamma=0.1)

    os.makedirs(args.out_dir, exist_ok=True)
    best_ap = 0.0
    patience_cntr = 0
    for epoch in range(1, args.epochs + 1):
        
        train_avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch:03d} Train  Loss {train_avg_loss:.4f}")
        loss_lr_history.append(train_avg_loss) 

        ap = evaluate(model, val_loader, device, args.score_thresh, args.mask_thresh, args.out_dir, epoch)
        ap_history.append(ap)

        # every 10 epoch save model
        if epoch % 10 == 0:
            torch.save(model.state_dict(), Path(args.out_dir) / f"epoch{epoch}_model.pth")
            print(f"model save in  {Path(args.out_dir)}/epoch{epoch}_model.pth")
            plot_loss_ap(loss_lr_history, ap_history, save_path=f"{args.out_dir}/epoch{epoch}")

        # save best ap model
        if ap >= best_ap:
            best_ap = ap
            patience_cntr = 0
            torch.save(model.state_dict(), Path(args.out_dir) / "model_best.pth")
            print(f"epoch{epoch} best ap {best_ap} save in  {Path(args.out_dir)} / model_best.pth")
        else:                         
            patience_cntr += 1

        if patience_cntr >= args.patience:
            print(f"Early-stopping triggered (patience={args.patience}) at epoch {epoch}")
            break

        lr_scheduler.step()

    torch.save(model.state_dict(), Path(args.out_dir) / "model_final.pth")

def predict(args):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, {torch.cuda.get_device_name(0)}")

    test_dataset = CellInstanceDataset(args.data_root, split="test")
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    model = MaskRCNN(num_classes=5)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    SCALES          = [0.75, 1.0, 1.25]        # TTA scale
    TTA_NMS_THRESH  = 0.50                     # NMS thresh

    results: List[Dict[str, Any]] = []
    viz_count = 0

    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Test"):
            img      = images[0].to(device)          # (C,H,W)
            H, W     = img.shape[-2:]
            img_name = names[0]
            image_id = filename_to_id(img_name, args.data_root)

            all_boxes, all_scores, all_labels, all_masks = [], [], [], []
            for scale in SCALES:
                img_s = F.interpolate(
                    img.unsqueeze(0), scale_factor=scale,
                    mode="bilinear", align_corners=False
                )                           # (1,C,Hs,Ws)

                out = model(img_s)[0]

                boxes = out["boxes"] / scale
                masks = F.interpolate(
                    out["masks"],
                    size=(H, W), mode="bilinear", align_corners=False
                )

                all_boxes.append(boxes.cpu())
                all_scores.append(out["scores"].cpu())
                all_labels.append(out["labels"].cpu())
                all_masks.append(masks.cpu())

            boxes   = torch.cat(all_boxes)
            scores  = torch.cat(all_scores)
            labels  = torch.cat(all_labels)
            masks   = torch.cat(all_masks)

            keep_idx = []
            for cls in labels.unique():
                idx  = torch.where(labels == cls)[0]
                kept = nms(boxes[idx], scores[idx], TTA_NMS_THRESH)
                keep_idx.append(idx[kept])
            keep_idx = torch.cat(keep_idx)

            boxes, scores, labels, masks = (
                boxes[keep_idx], scores[keep_idx],
                labels[keep_idx], masks[keep_idx]
            )

            bin_masks = (masks.squeeze(1) > args.mask_thresh)    # mask_thresh
            res = masks_to_rles(
                bin_masks, scores, labels, image_id,
                score_thr=args.score_thresh                     # score_thresh
            )
            results.extend(res)

            if viz_count < 10:
                visualize_predictions(
                    phase="test",
                    image_tensor=img.cpu(),
                    gt_boxes=None,
                    pred_boxes=boxes,
                    pred_scores=scores,
                    iou_threshold=args.mask_thresh,
                    save_path=f"{args.out_dir}/{viz_count}"
                )
                viz_count += 1

    with open(args.save_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp)
    print(f"Saved submission to {args.save_path}")

    


def filename_to_id(filename, data_root) -> int:

    mapping_path = Path(data_root) / "test_image_name_to_ids.json"

    if not hasattr(filename_to_id, "_cache"):
        with open(mapping_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        if isinstance(data, list):
            data = {d["file_name"]: d["id"] for d in data}

        filename_to_id._cache = data   

    return int(filename_to_id._cache[filename])



def parse_args(): 
    parser = argparse.ArgumentParser(description="Mask R‑CNN HW runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=str, required=True)
    common.add_argument("--workers", type=int, default=4)
    common.add_argument("--mask-thresh", type=float, default=0.8)
    common.add_argument("--score-thresh", type=float, default=0.5)
    common.add_argument("--out-dir", type=str, default="outputs")

    # train
    tr = sub.add_parser("train", parents=[common])
    tr.add_argument("--epochs", type=int, default=25)
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--patience", type=int, default=20)

    # predict 
    pr = sub.add_parser("predict", parents=[common])
    pr.add_argument("--weights", type=str, required=True)
    pr.add_argument("--save-path", type=str, default="test-results.json")

    return parser.parse_args()


def main():  
    args = parse_args()
    if args.cmd == "train":
        train(args)
        plot_loss_ap(loss_lr_history, ap_history, save_path=f"{args.out_dir}/final")
    else:  # predict
        predict(args)


if __name__ == "__main__":
    main()
