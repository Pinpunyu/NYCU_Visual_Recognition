import os
import json
import csv
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
from collections import defaultdict
from utils.model import Faster_RCNN


def generate_pred_csv(pred_json_path, output_csv_path, test_image_dir):
    with open(pred_json_path, 'r') as f:
        predictions = json.load(f)

    grouped = defaultdict(list)
    for pred in predictions:
        grouped[pred['image_id']].append((pred['bbox'][0], pred['category_id']))

    image_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(test_image_dir) if f.endswith('.png')]

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'pred_label'])

        for image_id in sorted(image_ids):
            digits = sorted(grouped[image_id], key=lambda x: x[0]) 
            if not digits:
                writer.writerow([image_id, -1])
            else:
                try:
                    number = ''.join(str(c - 1) for _, c in digits)
                    writer.writerow([image_id, int(number)])
                except ValueError:
                    writer.writerow([image_id, -1])


def test(model, model_path, test_dir, result_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
    results = []
    vis_save_path = os.path.join(result_path, 'vis')
    os.makedirs(vis_save_path, exist_ok=True)

    with torch.no_grad():
        for idx, image_file in enumerate(tqdm(image_files, desc="Testing")):
            image_path = os.path.join(test_dir, image_file)
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            output = model(img_tensor)[0]
            image_id = int(os.path.splitext(image_file)[0])

            vis_preds = []

            for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                if score > 0.7:
                    x_min, y_min, x_max, y_max = box.tolist()
                    bbox = [round(x_min, 2), round(y_min, 2),
                            round(x_max - x_min, 2), round(y_max - y_min, 2)]
                    results.append({
                        "image_id": image_id,
                        "bbox": bbox,
                        "score": round(score.item(), 4),
                        "category_id": int(label.item()) + 1
                    })
                    vis_preds.append({
                        "bbox": bbox,
                        "score": score.item(),
                        "category_id": int(label.item()) + 1
                    })

    os.makedirs(result_path, exist_ok=True)

    pred_json_path = os.path.join(result_path, 'pred.json')
    with open(pred_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Task1 - Saved pred.json to {pred_json_path}")

    pred_csv_path = os.path.join(result_path, 'pred.csv')
    generate_pred_csv(pred_json_path, pred_csv_path, test_dir)
    print(f"Task2 - Saved pred.csv to {pred_csv_path}")
    print(f"Visualization images saved to {vis_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./nycu-hw2-data')
    parser.add_argument('--checkpoint', type=str, default='./result/resnet50_v2_epoch20/ckpt/best_model.pth')
    parser.add_argument('--result_path', type=str, default='./result/resnet50_v2_epoch20')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = os.path.join(args.data_path, 'test')

    model_wrapper = Faster_RCNN(num_classes=11, backbone='resnet50_fpn_v2', pretrained=True)
    model = model_wrapper.model

    test(model, args.checkpoint, test_dir, args.result_path, device)
