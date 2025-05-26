import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import os
import torch.nn as nn 
from   torchvision.utils import save_image

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F

from utils.dataset_utils import RainSnowTestDataset

class TotalVariationLoss(nn.Module):
    def forward(self, x):
        diff_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return torch.mean(diff_h) + torch.mean(diff_v)

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)


def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def tta_inference(model, input_):
    """
    Perform Test-Time Augmentation (TTA) by averaging predictions over several flipped and rotated versions.
    """
    inputs = [
        input_,
        torch.flip(input_, dims=[3]),         # horizontal flip
        torch.flip(input_, dims=[2]),         # vertical flip
        torch.rot90(input_, k=1, dims=[2,3]), # rotate 90
        torch.rot90(input_, k=2, dims=[2,3]), # rotate 180
        torch.rot90(input_, k=3, dims=[2,3])  # rotate 270
    ]
    outputs = []

    for x in inputs:
        out = model(x)
        # Reverse augmentation
        if x is inputs[1]:
            out = torch.flip(out, dims=[3])
        elif x is inputs[2]:
            out = torch.flip(out, dims=[2])
        elif x is inputs[3]:
            out = torch.rot90(out, k=3, dims=[2,3])
        elif x is inputs[4]:
            out = torch.rot90(out, k=2, dims=[2,3])
        elif x is inputs[5]:
            out = torch.rot90(out, k=1, dims=[2,3])
        outputs.append(out)

    return torch.mean(torch.stack(outputs), dim=0)


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument("--test_root",   type=str, default="data/test/degraded/", help="test degraded images")
    parser.add_argument('--output_path', type=str, default="output/l1_ssim_tv/patch128", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="l1_ssim_tv/epoch200/epoch=199-step=160000.ckpt", help='checkpoint save path')
    parser.add_argument("--num_workers", type=int, default=4)
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path = "ckpt/" + testopt.ckpt_name
    print(ckpt_path)

    os.makedirs(testopt.output_path, exist_ok=True)

    model = PromptIRModel.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()

    test_dataset = RainSnowTestDataset(testopt.test_root)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=testopt.num_workers, pin_memory=True)

    for names, degraded in tqdm(test_dataloader, total=len(test_dataloader)):
        degraded = degraded.cuda()
        with torch.no_grad():
            restored = tta_inference(model, degraded)
            # restored = model(degraded)
        for img_t, name in zip(restored, names):
            save_path = os.path.join(testopt.output_path, name)
            save_image(torch.clamp(img_t, 0, 1.0), save_path)

    print(f"Done! All images saved to {testopt.output_path}")