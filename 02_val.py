from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

root = Path(r"e:\desktop_wj\PRML_HOMEWORK\HOMEWORK3")
data = np.load(root / "minimal_lstm" / "data_lookback18.npz")

x_train = torch.tensor(data["x_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32)
x_val = torch.tensor(data["x_val"], dtype=torch.float32)
y_val = torch.tensor(data["y_val"], dtype=torch.float32)

batch_size = 64
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, 64, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(self.drop(o[:, -1, :]))


def rmse_loss(pred, y):
    return torch.sqrt(torch.mean((pred - y) ** 2) + 1e-8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(x_train.shape[-1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

for _ in range(30):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = rmse_loss(model(xb), yb)
        loss.backward()
        opt.step()

model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        p = model(xb.to(device)).cpu().numpy()
        preds.append(p)
        trues.append(yb.numpy())

pred = np.vstack(preds).reshape(-1)
true = np.vstack(trues).reshape(-1)
rmse = mean_squared_error(true, pred) ** 0.5
r2 = r2_score(true, pred)

print("val RMSE:", round(float(rmse), 6))
print("val R2:", round(float(r2), 6))
