
import torch

def evaluate_mae(model, data_loader):
    model.eval()
    mae = 0
    n = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            pred_val = pred.mean()
            mae += torch.abs(pred_val - y).sum().item()
            n += y.size(0)
    return mae / n
