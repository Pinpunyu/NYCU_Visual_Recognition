import torch
from tqdm import tqdm

def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, targets in pbar:

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # total_loss += loss.item()
        total_loss += loss.detach().item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    return total_loss / len(train_loader)