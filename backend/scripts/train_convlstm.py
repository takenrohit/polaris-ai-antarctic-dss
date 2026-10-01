"""
Training Script for SeaIceConvLSTM on Real NSIDC and ERA5 Data.
Trains on training split (Days 0-13) and evaluates on strictly held-out test split (Days 14-20).
Saves trained PyTorch weights to backend/app/data/weights/convlstm_antarctic.pt.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.data.ingestion import environmental_data_provider


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur


class SeaIceConvLSTM(nn.Module):
    """
    Spatiotemporal Recurrent Neural Network for Sea Ice Concentration multi-day forecasting.
    Encodes past sequence of [SIC, SST, U10, V10, Current_Speed],
    and autoregressively decodes future SIC grids.
    """
    def __init__(self, in_channels: int = 5, hidden_dim: int = 24, num_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            cur_in = in_channels if i == 0 else hidden_dim
            cell_list.append(ConvLSTMCell(cur_in, hidden_dim, kernel_size=3))
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, future_steps: int = 7) -> torch.Tensor:
        B, T_in, C, H, W = x.size()
        h = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]

        # Encode historical sequence
        for t in range(T_in):
            inp = x[:, t]
            for layer_idx, cell in enumerate(self.cell_list):
                h[layer_idx], c[layer_idx] = cell(inp, h[layer_idx], c[layer_idx])
                inp = h[layer_idx]

        # Decode future predictions autoregressively
        outputs = []
        cur_pred = x[:, -1, 0:1]

        for _ in range(future_steps):
            context_features = torch.cat([cur_pred, x[:, -1, 1:]], dim=1)
            inp = context_features

            for layer_idx, cell in enumerate(self.cell_list):
                h[layer_idx], c[layer_idx] = cell(inp, h[layer_idx], c[layer_idx])
                inp = h[layer_idx]

            delta = self.conv_out(h[-1]) * 0.12
            cur_pred = torch.clamp(cur_pred + delta, 0.0, 1.0)
            outputs.append(cur_pred)

        return torch.stack(outputs, dim=1)


class IceEdgeLoss(nn.Module):
    """Combines MSE with extra weighting on the Marginal Ice Zone boundary."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        boundary_mask = (target >= 0.10) & (target <= 0.80)
        boundary_weight = torch.where(boundary_mask, 2.5, 1.0)
        weighted_loss = torch.mean(boundary_weight * (pred - target) ** 2)
        return 0.5 * mse_loss + 0.5 * weighted_loss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting SeaIceConvLSTM training on {device} using real NSIDC/ERA5 dataset...")

    lats = np.linspace(-78.0, -56.0, 33)
    lons = np.linspace(-180.0, 180.0, 45)

    # Ingest 21-day real temporal sequence from NetCDF store
    full_sequence = environmental_data_provider.get_gridded_sequence(lats, lons, num_days=21)
    print(f"Ingested sequence of shape: {full_sequence.shape} across 21 real observational days.")

    # Strict Temporal Split:
    # Training period: Days 0 to 13 (first 14 days)
    # Held-out validation period: Days 14 to 20 (last 7 days, strictly withheld from training)
    train_seq = full_sequence[:14]
    val_seq = full_sequence[14:]
    print(f"Training sequence: {train_seq.shape} | Held-out validation sequence: {val_seq.shape}")

    T_in = 5
    T_out = 5
    X_train_list, Y_train_list = [], []

    for start_idx in range(len(train_seq) - T_in - T_out + 1):
        x = train_seq[start_idx : start_idx + T_in]
        y = train_seq[start_idx + T_in : start_idx + T_in + T_out, 0:1]
        X_train_list.append(x)
        Y_train_list.append(y)

    X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32, device=device)
    Y_train = torch.tensor(np.array(Y_train_list), dtype=torch.float32, device=device)

    model = SeaIceConvLSTM(in_channels=5, hidden_dim=24, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    criterion = IceEdgeLoss()

    model.train()
    num_epochs = 40
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        preds = model(X_train, future_steps=T_out)
        loss = criterion(preds, Y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs:
            rmse = torch.sqrt(torch.mean((preds - Y_train) ** 2)).item()
            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss: {loss.item():.5f} - Training RMSE: {rmse:.4f}")

    # Evaluate on strictly held-out period (Days 14 to 20, 7 days ahead)
    model.eval()
    with torch.no_grad():
        x_val = torch.tensor(np.array([train_seq[-T_in:]]), dtype=torch.float32, device=device)
        y_val_gt = val_seq[:7, 0:1] # (7, 1, H, W)
        val_preds = model(x_val, future_steps=7)[0].cpu().numpy()

        val_rmse = float(np.sqrt(np.mean((val_preds - y_val_gt) ** 2)))
        # Persistence on held-out period: freeze Day 13 state forward
        persist_val = np.tile(train_seq[-1, 0:1][None, :, :, :], (7, 1, 1, 1))
        persist_rmse = float(np.sqrt(np.mean((persist_val - y_val_gt) ** 2)))

        print("--- Strictly Held-Out Validation (Days 15-21) ---")
        print(f"Model RMSE: {val_rmse:.4f} vs Persistence RMSE: {persist_rmse:.4f}")

    weights_dir = BACKEND_DIR / "app" / "data" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "convlstm_antarctic.pt"

    torch.save(model.state_dict(), str(weights_path))
    print(f"Model successfully saved to {weights_path}")


if __name__ == "__main__":
    train()
