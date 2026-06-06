from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from gnn_model import GINERegressor, smiles_to_data


def load_gine_model(model_path: str | Path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    model = GINERegressor(
        input_dim=checkpoint["input_dim"],
        edge_dim=checkpoint["edge_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        pooling=checkpoint["pooling"],
        batch_norm=checkpoint["batch_norm"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, device


@torch.no_grad()
def predict_pic50(smiles: str, model, device) -> float:
    data = smiles_to_data(smiles)

    if data is None:
        raise ValueError("Invalid SMILES or molecule could not be featurized.")

    loader = DataLoader([data], batch_size=1)
    batch = next(iter(loader)).to(device)

    pred = model(batch).view(-1).item()
    return float(pred)


def pic50_to_ic50_nm(pic50: float) -> float:
    return 10 ** (9 - pic50)


if __name__ == "__main__":
    model_path = Path("models/gine_egfr_chembl203.pt")

    model, checkpoint, device = load_gine_model(model_path)

    smiles = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    pred_pic50 = predict_pic50(smiles, model, device)
    pred_ic50_nm = pic50_to_ic50_nm(pred_pic50)

    print("Target:", checkpoint["target_name"])
    print("SMILES:", smiles)
    print("Predicted pIC50:", pred_pic50)
    print("Approx. IC50 [nM]:", pred_ic50_nm)