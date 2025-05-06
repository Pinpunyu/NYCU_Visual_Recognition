import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm

from utils.dataloader import CellInstanceDataset
from utils.model import MaskRCNN
from utils.engine import masks_to_rles
from utils.utils import collate_fn, visualize_predictions

import torchvision.transforms.functional as TF
import random

def apply_color_jitter(image_tensor):
    # image_tensor: (C,H,W), float32, range [0,1]
    brightness = 0.2
    contrast = 0.2
    saturation = 0.2

    if random.random() < 0.5:  
        image_tensor = TF.adjust_brightness(image_tensor, 1 + (random.uniform(-brightness, brightness)))
        image_tensor = TF.adjust_contrast(image_tensor, 1 + (random.uniform(-contrast, contrast)))
        image_tensor = TF.adjust_saturation(image_tensor, 1 + (random.uniform(-saturation, saturation)))
    return image_tensor


def filename_to_id(filename, data_root) -> int:
    mapping_path = Path(data_root) / "test_image_name_to_ids.json"
    if not hasattr(filename_to_id, "_cache"):
        with open(mapping_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, list):
            data = {d["file_name"]: d["id"] for d in data}
        filename_to_id._cache = data
    return int(filename_to_id._cache[filename])


def ensemble_predict(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, {torch.cuda.get_device_name(0)}")

    # Dataset
    test_dataset = CellInstanceDataset(args.data_root, split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn
    )

    # Load models
    models = []
    for weight_path in args.weights_list:
        model = MaskRCNN(num_classes=5)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded: {weight_path}")

    SCALES = [0.75, 1.0, 1.25]
    TTA_NMS_THRESH = 0.50
    results: List[Dict[str, Any]] = []
    viz_count = 0

    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Ensemble Test"):
            img = images[0].to(device)
            H, W = img.shape[-2:]
            img_name = names[0]
            image_id = filename_to_id(img_name, args.data_root)

            all_boxes, all_scores, all_labels, all_masks = [], [], [], []

            for model in models:
                for scale in SCALES:
                    img_s = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False)
                    img_s = img_s.squeeze(0)  # → (C,H,W) for transform
                    img_aug = apply_color_jitter(img_s)
                    img_aug = img_aug.unsqueeze(0)  # back to (1,C,H,W)

                    output = model(img_aug)[0]
                    boxes = output["boxes"] / scale
                    masks = F.interpolate(output["masks"], size=(H, W), mode="bilinear", align_corners=False)

                    all_boxes.append(boxes.cpu())
                    all_scores.append(output["scores"].cpu())
                    all_labels.append(output["labels"].cpu())
                    all_masks.append(masks.cpu())


            # Concatenate all results
            boxes = torch.cat(all_boxes)
            scores = torch.cat(all_scores)
            labels = torch.cat(all_labels)
            masks = torch.cat(all_masks)

            # Apply class-wise NMS
            keep_idx = []
            for cls in labels.unique():
                idx = torch.where(labels == cls)[0]
                kept = nms(boxes[idx], scores[idx], TTA_NMS_THRESH)
                keep_idx.append(idx[kept])
            keep_idx = torch.cat(keep_idx)

            boxes, scores, labels, masks = (
                boxes[keep_idx], scores[keep_idx],
                labels[keep_idx], masks[keep_idx]
            )

            bin_masks = (masks.squeeze(1) > args.mask_thresh)
            res = masks_to_rles(
                bin_masks, scores, labels, image_id,
                score_thr=args.score_thresh
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


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Mask R-CNN Tester")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--weights-list", type=str, nargs="+", required=True, help="List of model weight paths")
    parser.add_argument("--save-path", type=str, default="ensemble-results.json")
    parser.add_argument("--out-dir", type=str, default="outputs_ensemble")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mask-thresh", type=float, default=0.8)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ensemble_predict(args)
