"""
Use Case: Multivariate Time-Series Anomaly Detection — Solar Plants
--------------------------------------------------------------------
Problem : Detect anomalous telemetry patterns without labelled failure data.
Approach: LSTM Encoder-Decoder Autoencoder → Reconstruction error thresholding.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_solar_telemetry(n_days=200, seed=42):
    rng    = np.random.default_rng(seed)
    n      = n_days * 96  # 15-min intervals
    t      = np.linspace(0, n_days * 2 * np.pi, n)
    irr    = np.clip(500 * np.sin(t / n_days * np.pi) + rng.normal(0, 30, n), 0, 1000)
    temp   = 25 + 10 * np.sin(t / n_days * np.pi) + rng.normal(0, 2, n)
    wind   = np.clip(5 + 3 * np.sin(t * 0.5) + rng.normal(0, 1, n), 0, 20)
    gen    = 0.15 * irr + 0.05 * temp - 0.02 * wind + rng.normal(0, 10, n)
    volt   = 48 + 0.002 * irr + rng.normal(0, 0.5, n)
    data   = np.stack([irr, temp, wind, gen, volt], axis=1)
    # Inject anomaly in last 10%
    start  = int(0.9 * n)
    data[start:start+100, 0] *= 0.3
    data[start:start+100, 3] *= 0.4
    data[start:start+100, 4] += 8
    return data


def create_windows(data, window=48):
    return np.stack([data[i:i+window] for i in range(len(data)-window)])


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, latent_dim=32, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(n_features, latent_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, latent_dim, num_layers, batch_first=True)
        self.output  = nn.Linear(latent_dim, n_features)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        dec_in     = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(dec_in, (h, c))
        return self.output(decoded)


def train(windows, epochs=30, batch_size=64):
    X      = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
    model  = LSTMAutoencoder(windows.shape[2]).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit   = nn.MSELoss()
    for ep in range(epochs):
        loss_sum = sum(
            (opt.zero_grad() or True) and
            (loss := crit(model(b), b)) and
            (loss.backward() or True) and
            (opt.step() or True) and
            loss.item()
            for (b,) in loader
        )
        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1}/{epochs} Loss: {loss_sum/len(loader):.4f}")
    return model


def detect(model, windows, pct=95):
    model.eval()
    X = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        err = ((X - model(X))**2).mean(dim=(1,2)).cpu().numpy()
    thr   = np.percentile(err[:int(0.9*len(err))], pct)
    flags = err > thr
    return {"threshold": round(thr,4), "n_anomalies": int(flags.sum()),
            "anomaly_rate": round(flags.mean(),4), "anomaly_indices": np.where(flags)[0].tolist()}


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    data   = generate_solar_telemetry(200)
    scaler = StandardScaler()
    wins   = create_windows(scaler.fit_transform(data), 48)
    split  = int(0.9 * len(wins))
    print(f"Training on {split} windows...")
    model  = train(wins[:split], epochs=30)
    res    = detect(model, wins)
    print(f"Threshold: {res['threshold']} | Anomalies: {res['n_anomalies']} | Rate: {res['anomaly_rate']:.1%}")
    print(f"First anomaly at index: {res['anomaly_indices'][0] if res['anomaly_indices'] else None}")
