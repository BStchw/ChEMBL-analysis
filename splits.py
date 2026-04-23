from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame



def _validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split sizes must sum to 1.0, got train={train_size}, val={val_size}, test={test_size} (sum={total})."
        )
    for name, value in {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }.items():
        if value <= 0 or value >= 1:
            raise ValueError(f"{name} must be in (0, 1), got {value}.")



def murcko_scaffold_from_smiles(smiles: str, include_chirality: bool = False) -> Optional[str]:
    """Return the Bemis-Murcko scaffold for a SMILES string.

    Returns None if the SMILES cannot be parsed.
    """
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=include_chirality,
    )
    return scaffold or None



def add_scaffold_column(
    df: pd.DataFrame,
    *,
    smiles_col: str = "canonical_smiles",
    scaffold_col: str = "scaffold",
    include_chirality: bool = False,
    drop_invalid_smiles: bool = True,
) -> pd.DataFrame:
    """Add a Murcko scaffold column to a pandas DataFrame.

    Parameters
    ----------
    df:
        Input dataframe with at least a SMILES column.
    smiles_col:
        Name of the SMILES column.
    scaffold_col:
        Name of the scaffold output column.
    include_chirality:
        Whether chirality should be included in scaffold generation.
    drop_invalid_smiles:
        If True, rows with invalid SMILES are dropped. Otherwise the scaffold will be None.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found in dataframe.")

    out = df.copy()
    out[scaffold_col] = out[smiles_col].apply(
        lambda s: murcko_scaffold_from_smiles(s, include_chirality=include_chirality)
    )

    if drop_invalid_smiles:
        out = out[out[scaffold_col].notna()].copy()

    return out



def random_split(
    df: pd.DataFrame,
    *,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> SplitResult:
    """Create a random 3-way split.

    For typical dataset sizes this approximates the requested fractions exactly
    enough for baseline experiments while remaining robust for smaller tables.
    """
    _validate_split_sizes(train_size, val_size, test_size)

    n = len(df)
    if n < 3:
        raise ValueError("Need at least 3 rows to create train/val/test splits.")

    indices = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    n_train = int(round(n * train_size))
    n_val = int(round(n * val_size))

    # Keep all three splits non-empty when possible.
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        else:
            n_val -= 1

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return SplitResult(
        train_df=df.iloc[train_idx].reset_index(drop=True),
        val_df=df.iloc[val_idx].reset_index(drop=True),
        test_df=df.iloc[test_idx].reset_index(drop=True),
    )



def _scaffold_groups(
    df: pd.DataFrame,
    *,
    scaffold_col: str = "scaffold",
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, scaffold in df[scaffold_col].items():
        groups.setdefault(str(scaffold), []).append(idx)
    return groups



def scaffold_split(
    df: pd.DataFrame,
    *,
    smiles_col: str = "canonical_smiles",
    scaffold_col: str = "scaffold",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    include_chirality: bool = False,
    drop_invalid_smiles: bool = True,
) -> SplitResult:
    """Create a scaffold split based on RDKit Bemis-Murcko scaffolds.

    Whole scaffold groups are kept in exactly one split, so the final ratios are
    approximate rather than exact.
    """
    _validate_split_sizes(train_size, val_size, test_size)

    if scaffold_col not in df.columns:
        work_df = add_scaffold_column(
            df,
            smiles_col=smiles_col,
            scaffold_col=scaffold_col,
            include_chirality=include_chirality,
            drop_invalid_smiles=drop_invalid_smiles,
        )
    else:
        work_df = df.copy()
        if drop_invalid_smiles:
            work_df = work_df[work_df[scaffold_col].notna()].copy()

    if work_df.empty:
        raise ValueError("No rows left after scaffold generation/filtering.")

    target_train = int(round(len(work_df) * train_size))
    target_val = int(round(len(work_df) * val_size))

    groups = _scaffold_groups(work_df, scaffold_col=scaffold_col)
    if len(groups) < 3:
        raise ValueError(
            f"Scaffold split needs at least 3 distinct scaffold groups; got {len(groups)}. "            "For very small or structurally homogeneous datasets use random split or choose a richer dataset."
        )

    # Deterministic tie-breaking for groups of the same size.
    rng = np.random.RandomState(seed)
    shuffled_keys = list(groups.keys())
    rng.shuffle(shuffled_keys)
    shuffled_items = [(key, groups[key]) for key in shuffled_keys]
    sorted_groups = sorted(shuffled_items, key=lambda kv: len(kv[1]), reverse=True)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for _, group_indices in sorted_groups:
        if len(train_indices) + len(group_indices) <= target_train:
            train_indices.extend(group_indices)
        elif len(val_indices) + len(group_indices) <= target_val:
            val_indices.extend(group_indices)
        else:
            test_indices.extend(group_indices)

    # Edge-case guard: if one bucket ended up empty because of large scaffold groups,
    # move the smallest available group from the largest bucket.
    buckets = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    if any(len(v) == 0 for v in buckets.values()):
        group_list = [group_indices for _, group_indices in sorted_groups]

        def _bucket_name_for_index(index: int) -> str:
            if index in train_indices:
                return "train"
            if index in val_indices:
                return "val"
            return "test"

        for empty_name, bucket in buckets.items():
            if bucket:
                continue
            donor_name = max(buckets, key=lambda name: len(buckets[name]))
            donor_members = set(buckets[donor_name])
            donor_groups = [g for g in group_list if set(g).issubset(donor_members)]
            if not donor_groups:
                continue
            smallest_group = min(donor_groups, key=len)
            buckets[donor_name] = [i for i in buckets[donor_name] if i not in smallest_group]
            buckets[empty_name] = list(smallest_group)

        train_indices, val_indices, test_indices = (
            buckets["train"],
            buckets["val"],
            buckets["test"],
        )

    train_df = work_df.loc[sorted(train_indices)].reset_index(drop=True)
    val_df = work_df.loc[sorted(val_indices)].reset_index(drop=True)
    test_df = work_df.loc[sorted(test_indices)].reset_index(drop=True)

    return SplitResult(train_df=train_df, val_df=val_df, test_df=test_df)



def split_summary(split: SplitResult) -> pd.DataFrame:
    """Return a compact summary table for train/val/test sizes."""
    sizes = {
        "train": len(split.train_df),
        "val": len(split.val_df),
        "test": len(split.test_df),
    }
    total = sum(sizes.values())
    return pd.DataFrame(
        {
            "split": list(sizes.keys()),
            "n_rows": list(sizes.values()),
            "fraction": [value / total if total else np.nan for value in sizes.values()],
        }
    )



def scaffold_overlap_report(
    split: SplitResult,
    *,
    scaffold_col: str = "scaffold",
) -> Dict[str, int]:
    """Check whether scaffold groups leaked across splits."""
    train_scaffolds = set(split.train_df[scaffold_col].dropna()) if scaffold_col in split.train_df.columns else set()
    val_scaffolds = set(split.val_df[scaffold_col].dropna()) if scaffold_col in split.val_df.columns else set()
    test_scaffolds = set(split.test_df[scaffold_col].dropna()) if scaffold_col in split.test_df.columns else set()

    return {
        "train_val_overlap": len(train_scaffolds & val_scaffolds),
        "train_test_overlap": len(train_scaffolds & test_scaffolds),
        "val_test_overlap": len(val_scaffolds & test_scaffolds),
    }
