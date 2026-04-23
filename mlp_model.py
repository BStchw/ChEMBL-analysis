from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smiles_to_morgan_fp(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> Optional[np.ndarray]:
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
        useChirality=use_chirality,
    )
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def dataframe_to_fingerprints(
    df: pd.DataFrame,
    *,
    smiles_col: str = "canonical_smiles",
    target_col: str = "y",
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    fps = []
    ys = []
    kept_rows = []

    for idx, row in df.iterrows():
        fp = smiles_to_morgan_fp(
            row[smiles_col],
            radius=radius,
            n_bits=n_bits,
        )
        if fp is None:
            continue

        target = row[target_col]
        if pd.isna(target):
            continue

        fps.append(fp)
        ys.append(float(target))
        kept_rows.append(idx)

    if not fps:
        raise ValueError("No valid molecules after fingerprint featurization.")

    valid_df = df.loc[kept_rows].reset_index(drop=True)
    X = np.stack(fps).astype(np.float32)
    y = np.array(ys, dtype=np.float32).reshape(-1, 1)
    return X, y, valid_df


def make_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_t, y_t)


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    ds = make_tensor_dataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: Iterable[int] = (512, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims)
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_mlp_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)


@dataclass
class EpochResult:
    loss: float
    rmse: float
    mae: float
    r2: float


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.train()

    all_true = []
    all_pred = []
    running_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        all_true.append(y_batch.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    metrics = regression_metrics(y_true, y_pred)

    return EpochResult(
        loss=running_loss / max(n_samples, 1),
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.eval()

    all_true = []
    all_pred = []
    running_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        all_true.append(y_batch.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    metrics = regression_metrics(y_true, y_pred)

    return EpochResult(
        loss=running_loss / max(n_samples, 1),
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_res = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_res = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_res.loss,
            "train_rmse": train_res.rmse,
            "train_mae": train_res.mae,
            "train_r2": train_res.r2,
            "val_loss": val_res.loss,
            "val_rmse": val_res.rmse,
            "val_mae": val_res.mae,
            "val_r2": val_res.r2,
        }
        history.append(row)

        if val_res.loss < best_val_loss:
            best_val_loss = val_res.loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_res.loss:.4f} | "
                f"val_loss={val_res.loss:.4f} | "
                f"train_rmse={train_res.rmse:.4f} | "
                f"val_rmse={val_res.rmse:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    return model, history_df
