
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import RockDataset
from models.stae_unet import STAEUNet
from utils.loss import mse_loss
from utils.metrics import evaluate_mae

def train_model(train_set, val_set, epochs=10, batch_size=2, lr=1e-3):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = STAEUNet(in_channels=4, out_channels=1, base_channels=32).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = mse_loss(pred.mean(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_mae = evaluate_mae(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Val MAE: {val_mae:.4f}")

    return model
