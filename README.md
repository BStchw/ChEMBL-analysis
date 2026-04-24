# ChEMBL-analysis

Projekt kursowy poświęcony analizie danych z **ChEMBL** i budowie prostych modeli uczenia maszynowego do przewidywania aktywności biologicznej cząsteczek na podstawie ich struktury.

Repozytorium obejmuje cały pipeline:
- eksploracyjną analizę danych (EDA),
- przygotowanie czystego zbioru do modelowania,
- dwa rodzaje splitów danych (**random** i **scaffold**),
- dwa modele bazowe:
  - **MLP** na fingerprintach Morgan,
  - **GNN (GCN)** na grafowej reprezentacji cząsteczek.

## Cel projektu

Celem projektu przewidywanie aktywności biologicznej związków chemicznych oraz porównanie:
- klasycznego podejścia opartego o fingerprinty,
- podejścia grafowego,
- łatwiejszego **random split** z bardziej wymagającym **scaffold split**.

## Zakres prac

Projekt został zbudowany wokół następującego workflow:

1. **EDA danych ChEMBL**
   - analiza braków danych,
   - rozkładów `standard_type`, `standard_units`, `standard_relation`,
   - identyfikacja duplikatów i problemów jakości danych.

2. **Przygotowanie datasetu**
   - filtrowanie do jednego typu aktywności (domyślnie `IC50`),
   - ograniczenie do jednego targetu białkowego,
   - konwersja wartości do wspólnej skali,
   - transformacja do **pIC50**,
   - agregacja wielokrotnych pomiarów do jednej etykiety na cząsteczkę.

3. **Podział danych**
   - **random split** 80/10/10,
   - **scaffold split** 80/10/10 (na scaffoldach Bemisa–Murcko).

4. **Modele bazowe**
   - **MLP** na fingerprintach Morgan,
   - **GNN** z warstwami `GCNConv` i `global_mean_pool`.

## Struktura repozytorium

W repozytorium znajdują się obecnie notebooki i pliki Pythona odpowiadające kolejnym etapom pipeline’u: `EDA_ChEMBL.ipynb`, `data_preparation.py`, `prepare_dataset.ipynb`, `splits.py`, `mlp_model.py`, `train_mlp.ipynb`, `gnn_model.py`, `train_gnn.ipynb` oraz katalog `prepared_data/`.

```text
ChEMBL-analysis/
├── EDA_ChEMBL.ipynb         # eksploracyjna analiza danych
├── data_preparation.py      # funkcje do filtrowania i budowy datasetu
├── prepare_dataset.ipynb    # przygotowanie finalnego zbioru do modelowania
├── splits.py                # random split i scaffold split
├── mlp_model.py             # model MLP + fingerprinting Morgan
├── train_mlp.ipynb          # trening i ewaluacja MLP
├── gnn_model.py             # model GNN (GCN) dla grafów molekularnych
├── train_gnn.ipynb          # trening i ewaluacja GNN
├── prepared_data/           # zapisane splity i gotowe pliki CSV
└── README.md
```

## Przygotowanie danych

Dane są filtrowane tak, aby otrzymać możliwie spójny problem regresyjny:
- pojedynczy target białkowy,
- jeden typ aktywności (`IC50`),
- tylko rekordy z `standard_relation = "="`,
- wspólne jednostki aktywności,
- poprawne struktury (`canonical_smiles`),
- agregacja powtarzających się pomiarów.

Docelowy target regresyjny to:

```text
pIC50 = 9 - log10(IC50 w nM)
```

Dzięki temu rozkład wartości jest stabilniejszy niż przy surowym `IC50`.

## Modele

### 1. MLP baseline
Wejście:
- **Morgan fingerprint** (2048 bitów)

Architektura:
- `2048 -> 512 -> 128 -> 1`
- aktywacja `ReLU`
- loss: `MSE`
- optimizer: `Adam`

### 2. GNN baseline
Wejście:
- graf cząsteczki:
  - węzły = atomy,
  - krawędzie = wiązania,
  - cechy węzłów = minimalny zestaw cech atomowych.

Architektura:
- `GCNConv(input, 64) -> ReLU`
- `GCNConv(64, 64) -> ReLU`
- `GCNConv(64, 64) -> ReLU`
- `global_mean_pool`
- `Linear(64, 32) -> ReLU`
- `Linear(32, 1)`

## Wyniki bazowe

Na podstawie dotychczasowych eksperymentów:

### MLP
- **random split**: RMSE ≈ **0.708**, R² ≈ **0.640**
- **scaffold split**: RMSE ≈ **0.931**, R² ≈ **0.353**

### GNN
- **random split**: RMSE ≈ **1.070**, R² ≈ **0.178**
- **scaffold split**: RMSE ≈ **1.078**, R² ≈ **0.132**

### Wniosek
W tej wersji projektu **MLP na fingerprintach Morgan działa lepiej niż prosty baseline GNN**. Jednocześnie scaffold split okazał się wyraźnie trudniejszy niż random split, co jest zgodne z oczekiwaniami dla danych chemoinformatycznych.

## Jak uruchomić projekt

### 1. EDA
Uruchom:
- `EDA_ChEMBL.ipynb`

### 2. Przygotowanie datasetu
Uruchom:
- `prepare_dataset.ipynb`

Notebook zapisuje gotowe pliki do katalogu `prepared_data/`, m.in.:
- `chembl_ic50_model_dataset.csv`
- `train_random.csv`
- `val_random.csv`
- `test_random.csv`
- `train_scaffold.csv`
- `val_scaffold.csv`
- `test_scaffold.csv`
- `train_random_tiny.csv`

### 3. Trening MLP
Uruchom:
- `train_mlp.ipynb`

### 4. Trening GNN
Uruchom:
- `train_gnn.ipynb`

## Wymagania

Projekt korzysta m.in. z:
- Python
- Jupyter Notebook
- PySpark
- pandas
- RDKit
- PyTorch
- PyTorch Geometric
- matplotlib


## Źródło danych

Dataset https://www.ebi.ac.uk/chembl/api/data