from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GINEConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


HYBRIDIZATION_VALUES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot_hybridization(atom: Chem.rdchem.Atom) -> List[float]:
    value = atom.GetHybridization()
    return [1.0 if value == h else 0.0 for h in HYBRIDIZATION_VALUES]


def atom_to_features(atom: Chem.rdchem.Atom) -> List[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetTotalNumHs()),
    ] + one_hot_hybridization(atom)


BOND_TYPE_VALUES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREO_VALUES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]

EDGE_FEATURE_DIM = len(BOND_TYPE_VALUES) + 2 + len(BOND_STEREO_VALUES)


def one_hot_value(value, allowed_values) -> List[float]:
    return [1.0 if value == allowed else 0.0 for allowed in allowed_values]

def bond_to_features(bond: Chem.rdchem.Bond) -> List[float]:
    return (
        one_hot_value(bond.GetBondType(), BOND_TYPE_VALUES)
        + [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]
        + one_hot_value(bond.GetStereo(), BOND_STEREO_VALUES)
    )

def smiles_to_data(smiles: str, target: Optional[float] = None) -> Optional[Data]:
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [atom_to_features(atom) for atom in mol.GetAtoms()]
    if not node_features:
        return None

    x = torch.tensor(node_features, dtype=torch.float)

    edge_pairs = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_features = bond_to_features(bond)

        edge_pairs.append((i, j))
        edge_features.append(bond_features)

        edge_pairs.append((j, i))
        edge_features.append(bond_features)

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if target is not None and not pd.isna(target):
        data.y = torch.tensor([[float(target)]], dtype=torch.float)

    return data


def dataframe_to_graphs(
    df: pd.DataFrame,
    *,
    smiles_col: str = "canonical_smiles",
    target_col: str = "y",
) -> Tuple[List[Data], pd.DataFrame]:
    graphs: List[Data] = []
    kept_rows = []

    for idx, row in df.iterrows():
        target = row[target_col] if target_col in row else None
        data = smiles_to_data(row[smiles_col], target=target)
        if data is None:
            continue
        graphs.append(data)
        kept_rows.append(idx)

    if not graphs:
        raise ValueError("No valid molecules after graph featurization.")

    valid_df = df.loc[kept_rows].reset_index(drop=True)
    return graphs, valid_df


def make_graph_dataloader(
    graphs: List[Data],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


class GCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.lin1 = nn.Linear(hidden_dim, 32)
        self.lin2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x


class GINERegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "add",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if pooling not in {"mean", "add", "max", "mean_add_max"}:
            raise ValueError(
                "pooling must be one of: 'mean', 'add', 'max', 'mean_add_max'"
            )

        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        self.node_encoder = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(
                GINEConv(
                    nn=mlp,
                    edge_dim=edge_dim,
                )
            )

            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.batch_norms.append(nn.Identity())

        if pooling == "mean_add_max":
            pooled_dim = hidden_dim * 3
        else:
            pooled_dim = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch)

        if self.pooling == "add":
            return global_add_pool(x, batch)

        if self.pooling == "max":
            return global_max_pool(x, batch)

        mean_pool = global_mean_pool(x, batch)
        add_pool = global_add_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        return torch.cat([mean_pool, add_pool, max_pool], dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = self.node_encoder(x)
        x = torch.relu(x)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        x = self._pool(x, batch)
        return self.head(x)

def init_linear_weights(module: nn.Module) -> None:
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

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        preds = model(batch)
        y = batch.y.view(-1, 1)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        all_true.append(y.detach().cpu().numpy())
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

    for batch in loader:
        batch = batch.to(device)

        preds = model(batch)
        y = batch.y.view(-1, 1)
        loss = criterion(preds, y)

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        all_true.append(y.detach().cpu().numpy())
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
    best_epoch = None
    best_val_loss = float("inf")
    best_val_rmse = float("inf")

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

        if val_res.rmse < best_val_rmse:
            best_epoch = epoch
            best_val_loss = val_res.loss
            best_val_rmse = val_res.rmse
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
    history_df.attrs["best_epoch"] = best_epoch
    history_df.attrs["best_val_loss"] = best_val_loss
    history_df.attrs["best_val_rmse"] = best_val_rmse

    return model, history_df
