from pathlib import Path
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import skimage.io as skio  # tif reader
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torchvision.ops import boxes as box_ops
from torchvision import transforms
from torchvision.transforms import v2 as T    


class CellInstanceDataset(Dataset):

    _IMG_FILE = "image.tif"
    _CLS_TMPL = "class{}.tif"

    def __init__(self, root, split = "train", ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split

        if split == "train":
            self.transforms = T.Compose([
                T.ToImage(),                                 
                # T.RandomHorizontalFlip(p=0.5),
                # T.RandomVerticalFlip(p=0.5),
                # T.RandomRotation(degrees=15),                
                T.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
                #T.GaussianBlur(kernel_size=3, sigma=(0.1,1.0)),
                T.ToDtype(torch.float32, scale=True)         # 0-255→0-1 float32
            ])


        else:
            self.transforms = T.Compose([
                T.ToImage(),                             
                T.ToDtype(torch.float32, scale=True),
            ])

 

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train/val/test")

        if split == "test":
            self.img_files = sorted((self.root / "test_release").glob("*.tif"))
        else:
            self.img_dirs = sorted((self.root / "train").iterdir())

    def _read_image(self, path: Path) -> Image.Image:
       
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")          
        return img


    def _load_masks(self, img_dir):
        """Return (instance_masks, class_labels)."""

        masks = []
        labels= []
        for cls_id in range(1, 5):
            m_path = img_dir / self._CLS_TMPL.format(cls_id)
            if not m_path.exists():
                continue
            mask_arr = skio.imread(m_path)
            unique_inst = np.unique(mask_arr)
            # 0 is background
            for inst_id in unique_inst[unique_inst != 0]:
                instance_mask = (mask_arr == inst_id).astype(np.uint8)
                masks.append(instance_mask)
                labels.append(cls_id)
        if not masks:
            # fallback – empty image
            return np.empty((0, 0, 0), dtype=np.uint8), np.empty((0,), dtype=np.int64)
        stacked = np.stack(masks, axis=0)
        return stacked, np.array(labels, dtype=np.int64)

    # annotations
    def __getitem__(self, idx: int):
        if self.split == "test":
            img_path = self.img_files[idx]
            image = self._read_image(img_path)
            
            image = self.transforms(image)
            return image, str(img_path.name)
        

        img_dir = self.img_dirs[idx]
        image = self._read_image(img_dir / self._IMG_FILE)
        masks_np, labels_np = self._load_masks(img_dir)

        # build target dict (COCO‑style)
        masks_tensor = torch.as_tensor(masks_np, dtype=torch.uint8)
        boxes_tensor = self._masks_to_boxes(masks_tensor)


        target: Dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": torch.as_tensor(labels_np, dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": torch.tensor([idx]),
            "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]),
            # "iscrowd": torch.zeros((labels_np.shape[0],), dtype=torch.int64),
            "iscrowd": torch.zeros((len(labels_np),), dtype=torch.int64),
        }

        image, target = self.transforms(image, target)
        return image, target
    


    


    def __len__(self) -> int:  
        return len(self.img_files) if self.split == "test" else len(self.img_dirs)

    def _masks_to_boxes(self, masks):  
        """Convert binary masks to tight bounding boxes."""
        if masks.numel() == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        boxes = box_ops.masks_to_boxes(masks)
        return boxes

