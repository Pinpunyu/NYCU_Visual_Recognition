from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from utils.utils import visualize_predictions
from torchmetrics.detection.mean_ap import MeanAveragePrecision



class MetricLogger:  
    def __init__(self):
        self.history: List[Tuple[float, float]] = []

    def log(self, loss: float, lr: float):
        self.history.append((loss, lr))

def masks_to_rles(masks, scores, labels, image_id, score_thr) :
    """Convert predicted masks to COCO result dicts with RLE encoded masks."""

    keep = scores >= score_thr
    masks = masks[keep].cpu().numpy()
    scores = scores[keep].cpu().tolist()
    labels = labels[keep].cpu().tolist()
    records = []

    for m, s, c in zip(masks, scores, labels):
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")

        y, x = np.where(m)
        ymin, ymax = y.min(), y.max()
        xmin, xmax = x.min(), x.max()
        bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
        records.append({
                "image_id": int(image_id),
                "category_id": int(c),
                "segmentation": rle,
                "score": float(s),
                "bbox": bbox,
            }
        )
    return records


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
        for images, targets in pbar:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total = sum(loss_dict.values())

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            total_loss += total.item()

            pbar.set_postfix(
                loss=f"{total.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    
    return total_loss / len(train_loader)



def evaluate(model, loader, device, score_thresh, mask_thresh, output_dir, epoch):  # noqa: D401
    
    model.eval()
    metric = MeanAveragePrecision(iou_type="segm")

    viz_count = 0 

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Evaluate"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                keep = outputs[i]["scores"] >= score_thresh     # 0.5
                bin_masks = (outputs[i]["masks"].squeeze(1) > mask_thresh)  # 0.8

                pred = {
                    "scores": outputs[i]["scores"][keep].cpu(),
                    "labels": outputs[i]["labels"][keep].cpu(),
                    "masks":  bin_masks[keep].to(torch.uint8).cpu()
                }
                gt = {
                    "labels": targets[i]["labels"].cpu(),
                    "masks": targets[i]["masks"].to(torch.uint8).cpu()
                }

                metric.update([pred], [gt])

                if viz_count < 5 and epoch % 10 == 0:  # every 5  epoch vis 6 picture
                    visualize_predictions(
                        phase="val",
                        image_tensor=images[i].cpu(),
                        gt_boxes=targets[i]["boxes"].cpu(),
                        pred_boxes=outputs[i]["boxes"].cpu(),
                        pred_scores=outputs[i]["scores"].cpu(),
                        iou_threshold=mask_thresh,
                        save_path=f"{output_dir}/epoch{epoch}_{viz_count}"
                    )
                    viz_count += 1


    res = metric.compute()
    print(f"Epoch {epoch} Evaluation Results: {res['map'].item()}")
    return res["map"].item() 

