import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--pred_dir', type=str, default='output/l1_ssim_tv/patch128')
    opt = parser.parse_args()

    pred_dir = opt.pred_dir
    output_file = 'pred.npz'

    result_dict = {}

    file_list = sorted(os.listdir(pred_dir), key=lambda x: int(x.split('.')[0]))

    for file_name in tqdm(file_list, desc="Packing into npz"):
        if not file_name.endswith('.png'):
            continue
        file_path = os.path.join(pred_dir, file_name)
        
        image = Image.open(file_path).convert('RGB')
        image_np = np.array(image)  # (H, W, 3)

        image_np = image_np.transpose(2, 0, 1)  # (3, H, W)

        result_dict[file_name] = image_np

    np.savez(output_file, **result_dict)
    print(f"Saved {len(result_dict)} images to {output_file}")

