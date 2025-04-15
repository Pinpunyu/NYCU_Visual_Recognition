import os
import torch
import torchvision
import json
import argparse
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from utils.model import Faster_RCNN
from utils.dataloader import RecognitionData, collate_fn
from utils.train import train_one_epoch
from utils.evaluate import evaluate
from test import test

def save_training_settings(args, device, model_config, optim_config, sched_config):
    train_settings = {
        "model": model_config,
        "optimizer": optim_config,
        "scheduler": sched_config,
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "resume": args.resume,
            "prefetch_factor": args.prefetch_factor,
            "num_workers": args.num_workers
        },
        "path": {
            "data_path": args.data_path,
            "result_path": args.result_path
        },
        "device": str(device)
    }

    os.makedirs(args.result_path, exist_ok=True)
    with open(os.path.join(args.result_path, 'train_settings.json'), 'w') as f:
        json.dump(train_settings, f, indent=4)
    print(f"Training settings saved to {args.result_path}/train_settings.json")


def load_checkpoint(model, optimizer, checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded: {checkpoint_path}, Starting from epoch {start_epoch}")

    return model, optimizer, start_epoch

def save_checkpoint(model, optimizer, epoch, checkpoint_dir='./result/ckpt', filename='last_model.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def visualization(dir_root ,log_path):

    df = pd.read_csv(log_path)

    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dir_root}/records/loss_curve.png')

    plt.figure()
    plt.plot(df['epoch'], df['val_mAP'], label='Validation mAP', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dir_root}/records/map_curve.png')



def main(args):

    logdir = f'{args.result_path}/records'
    os.makedirs(logdir, exist_ok=True)

    if args.prefetch_factor > 0:
        train_loader = DataLoader(RecognitionData(f'{args.data_path}/train', f'{args.data_path}/train.json'),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch_factor, collate_fn=collate_fn)
        valid_loader = DataLoader(RecognitionData(f'{args.data_path}/valid', f'{args.data_path}/valid.json'),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch_factor, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(RecognitionData(f'{args.data_path}/train', f'{args.data_path}/train.json'),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        valid_loader = DataLoader(RecognitionData(f'{args.data_path}/valid', f'{args.data_path}/valid.json'),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    model_config = {
        "num_classes": 11,
        "backbone": args.backbone,
        "pretrained": True,
        "freeze_backbone": False,
        "freeze_rpn": False,
        "use_custom_head": False
    }
    model_wrapper = Faster_RCNN(**model_config)
    model = model_wrapper.model

    optim_config = {
        "type": "Adam",
        "lr": args.lr,
        "weight_decay": 0.0001
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_config["lr"], weight_decay=optim_config["weight_decay"])

    sched_config = {
        "type": "CosineAnnealingLR",
        "T_max": args.epochs,
        "eta_min": 0.000001
    }
    scheduler = CosineAnnealingLR(optimizer, T_max=sched_config["T_max"], eta_min=sched_config["eta_min"])


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}, {torch.cuda.get_device_name(0)}")
    model.to(device)

    save_training_settings(args, device, model_config, optim_config, sched_config)

    ckpt_dir = os.path.join(args.result_path, 'ckpt')
    last_model_path = os.path.join(ckpt_dir, 'last_model.pth')
    start_epoch = 0

    if args.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, last_model_path)
        print(f"Resuming training from epoch {start_epoch}")

    print(f"Start training from epoch {start_epoch}")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_mAP": []
    }

    best_score = -1
    patience_counter = 0
    best_model_path = os.path.join(args.result_path, 'ckpt', 'epoch_best.pth')

    for epoch in range(args.epochs):
        
        #train
        train_avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        print(f"Training Epoch {epoch+1}, Loss: {train_avg_loss:.4f}")
        
        #validate
        val_predictions, map_score = evaluate(model, valid_loader, device)
        val_score = map_score['map'].item()
        print(f"Validation Epoch {epoch+1}, mAP: {val_score:.4f}")

        # save training data
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_avg_loss)
        history["val_mAP"].append(val_score)


        val_json_path = os.path.join(args.result_path, f'valid_epoch_{epoch+1}.json')
        with open(val_json_path, 'w') as f:
            json.dump(val_predictions, f, indent=2)


        save_checkpoint(model, optimizer, epoch, ckpt_dir, filename='last_model.pth')

        if val_score > best_score:
            patience_counter = 0
            best_score = val_score
            save_checkpoint(model, optimizer, epoch, ckpt_dir, filename='best_model.pth')
            print(f"Best model updated at epoch {epoch+1} with mAP {val_score} ")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
        
    
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(args.result_path, 'records', 'training_log.csv'), index=False)
        print(f"Saved training log to {args.result_path}/records/training_log.csv")
        visualization(args.result_path ,f"{args.result_path}/records/training_log.csv")


    # Test prediction
    test_dir = os.path.join(args.data_path, 'test')
    best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model, optimizer, _ = load_checkpoint(model, optimizer, best_model_path)
        print(f"Loaded best model from {best_model_path}")

    test(model, best_model_path, test_dir, args.result_path, device)
    print(f" Testing Finish.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_path", type=str, default="./nycu-hw2-data")
    parser.add_argument("--backbone", type=str, default="resnet50_fpn_v2")
    parser.add_argument("--result_path", type=str, default="./result/resnext50_epoch20")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--resume", type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    main(args)

    
    '''
    load_model
    load_op
    load_data
    
    for epoch:
        train
        validat

    test
    '''