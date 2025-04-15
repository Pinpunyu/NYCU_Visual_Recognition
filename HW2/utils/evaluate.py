from tqdm import tqdm
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def evaluate(model, valid_loader, device):
    model.eval()
    results = []
    metric = MeanAveragePrecision()

    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Validating")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            image_ids = [int(t['image_id'].item()) for t in targets]

            outputs = model(images) 

            for output, target in zip(outputs, targets):
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()

                # Apply NMS
                keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                preds = {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels
                }

                gts = {
                    "boxes": target['boxes'].cpu(),
                    "labels": target['labels'].cpu()
                }

                metric.update([preds], [gts])

                image_id = int(target['image_id'].item())
                for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
                    result = {
                        "image_id": image_id,
                        "bbox": [round(float(x), 2) for x in box.tolist()],
                        "score": round(float(score), 4),
                        "category_id": int(label) + 1
                    }
                    results.append(result)

    map_score = metric.compute()
    return results, map_score