from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.representation.base import RepresentationModel

class AutoencoderModule(nn.Module):
    """Deep PyTorch Autoencoder architecture for scientific composition embeddings."""
    def __init__(self, input_dim: int = 24, latent_dim: int = 16, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
            
        # Encoder
        enc_layers = []
        in_d = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_d, h))
            enc_layers.append(nn.BatchNorm1d(h))
            enc_layers.append(nn.SiLU())
            in_d = h
        enc_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder
        dec_layers = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_d, h))
            dec_layers.append(nn.BatchNorm1d(h))
            dec_layers.append(nn.SiLU())
            in_d = h
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


class CompositionalAutoencoder(RepresentationModel):
    """
    Self-Supervised Compositional Autoencoder for Materials Science representations.
    """
    def __init__(
        self,
        input_dim: int = 24,
        latent_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        device: Optional[str] = None
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = AutoencoderModule(input_dim, latent_dim, self.hidden_dims).to(self.device)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, data: np.ndarray, epochs: int = 15, batch_size: int = 32) -> Dict[str, Any]:
        """Trains the autoencoder on normalized input scientific features."""
        data_norm = self.scaler.fit_transform(data)
        self.is_fitted = True
        
        tensor_data = torch.tensor(data_norm, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        self.model.train()
        history = []
        
        for ep in range(epochs):
            total_loss = 0.0
            for batch in loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                x_rec, z = self.model(x)
                loss = criterion(x_rec, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x)
                
            mean_loss = total_loss / len(data)
            history.append(mean_loss)
            
        return {
            "final_loss": history[-1] if history else 0.0,
            "history": history,
            "epochs": epochs
        }

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Extracts latent embeddings for the input data."""
        self.model.eval()
        if not self.is_fitted:
            data_norm = self.scaler.fit_transform(data)
            self.is_fitted = True
        else:
            data_norm = self.scaler.transform(data)
            
        tensor_data = torch.tensor(data_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, z = self.model(tensor_data)
            
        return z.cpu().numpy()

    def get_reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Calculates sample-wise mean squared reconstruction error for anomaly detection."""
        self.model.eval()
        data_norm = self.scaler.transform(data) if self.is_fitted else self.scaler.fit_transform(data)
        tensor_data = torch.tensor(data_norm, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            x_rec, _ = self.model(tensor_data)
            diff = (x_rec - tensor_data).cpu().numpy()
            mse_per_sample = np.mean(diff ** 2, axis=1)
            
        return mse_per_sample

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "scaler_mean": self.scaler.mean_ if self.is_fitted else None,
            "scaler_scale": self.scaler.scale_ if self.is_fitted else None,
            "is_fitted": self.is_fitted
        }, str(p))

    def load(self, path: Union[str, Path]) -> None:
        p = Path(path)
        ckpt = torch.load(str(p), map_location=self.device)
        self.input_dim = ckpt["input_dim"]
        self.latent_dim = ckpt["latent_dim"]
        self.hidden_dims = ckpt["hidden_dims"]
        self.model = AutoencoderModule(self.input_dim, self.latent_dim, self.hidden_dims).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        
        if ckpt.get("is_fitted", False) and ckpt.get("scaler_mean") is not None:
            self.scaler.mean_ = ckpt["scaler_mean"]
            self.scaler.scale_ = ckpt["scaler_scale"]
            self.is_fitted = True
