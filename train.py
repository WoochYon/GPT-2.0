import torch
import torch.nn.functional as F

def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)

def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    total_loss, total_count = 0.0, 0

    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

        if max_steps is not None and step + 1 >= max_steps:
            break

    return total_loss / total_count
