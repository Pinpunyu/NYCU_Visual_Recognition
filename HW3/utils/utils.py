import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision.ops as ops


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch): 
    return tuple(zip(*batch))


def plot_loss_ap(losses, ap_history, save_path="training"):

    epochs = range(1, len(ap_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    loss_path = f"{save_path}_loss.png"
    plt.savefig(loss_path)
    print(f"Saved loss plot to {loss_path}")
    # plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(epochs, ap_history, label="AP", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Average Precision (AP)")
    plt.title("Validation AP")
    plt.grid(True)
    plt.legend()
    ap_path = f"{save_path}_ap.png"
    plt.savefig(ap_path)
    print(f"Saved AP plot to {ap_path}")
    # plt.show()


def visualize_predictions(phase, image_tensor, gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5, save_path="./output"):


    image_tensor = (image_tensor * 255).byte() if image_tensor.max() <= 1 else image_tensor.byte()
    image = image_tensor.clone()

    if gt_boxes is not None and gt_boxes.numel() != 0:
        image = draw_bounding_boxes(image, gt_boxes, colors="green", width=2)



    if pred_boxes is None or pred_boxes.numel() == 0:
        plt.imshow(to_pil_image(image))
        plt.axis("off")
        plt.savefig(f"{save_path}_{phase}_pred.png")
        return

    if gt_boxes is not None and gt_boxes.numel() != 0:
        iou_matrix = ops.box_iou(pred_boxes, gt_boxes)
        max_ious, _ = iou_matrix.max(dim=1)
        labels = ["P" if iou >= iou_threshold else "N" for iou in max_ious]
    else:
        labels = ["P"] * pred_boxes.size(0)
    label_colors = ["yellow" if l == "P" else "red" for l in labels]

    image = draw_bounding_boxes(image, pred_boxes, labels=labels, colors=label_colors, width=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(to_pil_image(image))
    plt.title("Green: GT | Yellow: P | Red: N")
    plt.axis("off")
    plt.savefig(f"{save_path}_{phase}_pred.png")
    plt.close()
