import glob
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os


def collate_fn(batch):
    return tuple(zip(*batch))

class RecognitionData(Dataset):
    def __init__(self, img_dir, json_path=''):
        self.mode = img_dir.split('/')[-1]
        self.img_dir = img_dir
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.img_list = data['images']
            self.anno_dict = self._build_annotation_dict(data['annotations'])
            self.cat_to_label = self._build_cat_to_label(data['categories']) # category_id → 0~9
            self.label_to_cat = {v: k for k, v in self.cat_to_label.items()}  # 0~9 → category_id
    
    def _build_annotation_dict(self, annotations):
        anno_dict = {}
        for anno in annotations:
            img_id = anno['image_id']
            if img_id not in anno_dict:
                anno_dict[img_id] = []
            anno_dict[img_id].append(anno)
        return anno_dict

    def _build_cat_to_label(self, categories):
        cat_ids = sorted([cat['id'] for cat in categories])
        return {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_info = self.img_list[index]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)

        annos = self.anno_dict.get(img_id, [])
        boxes = []
        labels = []
        for anno in annos:
            x, y, w, h = anno['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_to_label[anno['category_id']])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        return img, target
        
        
    