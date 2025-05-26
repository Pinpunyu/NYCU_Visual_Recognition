import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.dataset_utils import RainSnowTrainDataset
from torch.utils.data import DataLoader, random_split
from torchmetrics import StructuralSimilarityIndexMeasure

class TotalVariationLoss(nn.Module):
    def forward(self, x):
        diff_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return torch.mean(diff_h) + torch.mean(diff_v)


class PromptIRModel(pl.LightningModule):
    def __init__(self, l1_w=1.0, ssim_w=1.0, tv_w=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = PromptIR(decoder=True)
        self.loss_l1 = nn.L1Loss()
        self.loss_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.loss_tv = TotalVariationLoss()
        self.val_losses = []

    def forward(self, x):
        base = self.backbone(x)
        return torch.clamp(base, 0, 1)

    def _compute_losses(self, output, target):
        l1 = self.loss_l1(output, target)
        ssim = 1 - self.loss_ssim(output, target)
        tv = self.loss_tv(output)
        total = self.hparams.l1_w * l1 + self.hparams.ssim_w * ssim + self.hparams.tv_w * tv
        return total, l1, ssim, tv

    def training_step(self, batch, batch_idx):
        _, degraded, clean = batch
        pred = self(degraded)
        loss, l1, ssim, tv = self._compute_losses(pred, clean)
        self.log_dict({'train/loss': loss, 'train/l1': l1, 'train/ssim': ssim, 'train/tv': tv})
        return loss

    def validation_step(self, batch, batch_idx):
        _, degraded, clean = batch
        pred = self(degraded)
        loss, l1, ssim, tv = self._compute_losses(pred, clean)
        self.val_losses.append(loss)
        self.log_dict({'val/loss': loss, 'val/l1': l1, 'val/ssim': ssim, 'val/tv': tv}, prog_bar=True)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True)
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=15, max_epochs=opt.epochs)
        return {"optimizer": optimizer, "lr_scheduler": { "scheduler": scheduler, "interval": "epoch", "frequency": 1, "name": "cosine_lr" }}



def main():
    print("Options\n", opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    fullset = RainSnowTrainDataset(opt)
    val_len   = int(len(fullset) * 0.1)
    train_len = len(fullset) - val_len
    trainset, valset = random_split(fullset, [train_len, val_len], generator=torch.Generator().manual_seed(14))
    valset.dataset.phase = "val"
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=opt.num_workers)
    valloader  = DataLoader(valset,  batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    model = PromptIRModel()

    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 10,save_top_k=-1)
    ckpt_best = ModelCheckpoint(dirpath=opt.ckpt_dir, monitor="avg_val_loss", mode="min", save_top_k=10, filename="best-{epoch:02d}-{avg_val_loss:.4f}")


    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, strategy="ddp_find_unused_parameters_true", logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()



