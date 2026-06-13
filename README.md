# ChEMBL Analysis – EGFR Activity Prediction

Projekt zrealizowany w ramach przedmiotu **„Warsztaty sztucznej inteligencji”**.  
Celem projektu jest przygotowanie danych z bazy ChEMBL oraz budowa modeli regresyjnych przewidujących aktywność biologiczną cząsteczek względem receptora **EGFR** (`CHEMBL203`).

Projekt obejmuje pełny pipeline:

- eksploracyjną analizę danych,
- czyszczenie i przygotowanie danych ChEMBL,
- transformację wartości `IC50` do `pIC50`,
- podział danych metodą random split i scaffold split,
- model bazowy MLP oparty na fingerprintach Morgan,
- model grafowy GINE,
- zapis wytrenowanego modelu GINE,
- predykcję dla pojedynczej cząsteczki zapisanej jako SMILES,
- prosty interfejs użytkownika zbudowany w Streamlit.

## Finalna funkcjonalność

Aplikacja przyjmuje zapis cząsteczki w formacie **SMILES** i zwraca:

- przewidywaną wartość `pIC50`,
- przybliżoną wartość `IC50` w nanomolach,
- podstawowe informacje o wykorzystanym modelu.

Uruchomienie aplikacji:

```bash
streamlit run app.py
```

## Cel predykcji

Model przewiduje aktywność związku chemicznego względem:

- **Target:** Epidermal growth factor receptor
- **Skrót:** EGFR
- **ChEMBL ID:** `CHEMBL203`
- **Typ aktywności:** `IC50`
- **Problem:** regresja

Wartość docelowa jest obliczana według wzoru:

```text
pIC50 = 9 - log10(IC50 [nM])
```

Większa wartość `pIC50` oznacza większą przewidywaną aktywność związku.

## Przygotowanie danych

Dane są filtrowane w celu utworzenia możliwie spójnego problemu regresyjnego. Pipeline obejmuje między innymi:

- wybór jednego targetu białkowego,
- ograniczenie danych do pomiarów `IC50`,
- wybór rekordów ze `standard_relation = "="`,
- ujednolicenie jednostek,
- odrzucenie brakujących lub niepoprawnych struktur,
- walidację zapisów `canonical_smiles`,
- transformację `IC50` do `pIC50`,
- agregację wielokrotnych pomiarów tej samej cząsteczki.

## Podział danych

W projekcie wykorzystano dwa sposoby podziału danych:

### Random split

Losowy podział cząsteczek w proporcji:

- 80% – zbiór treningowy,
- 10% – zbiór walidacyjny,
- 10% – zbiór testowy.

Random split jest prostszy, ponieważ podobne strukturalnie cząsteczki mogą znaleźć się w różnych częściach zbioru.

### Scaffold split

Podział oparty na scaffoldach Bemisa–Murcko. Cząsteczki posiadające ten sam główny szkielet chemiczny trafiają do tej samej części zbioru.

Scaffold split lepiej sprawdza zdolność modelu do generalizacji na nowe rodziny struktur chemicznych i zwykle stanowi trudniejsze zadanie niż random split.

## Modele

### MLP baseline

Model bazowy wykorzystuje fingerprinty Morgan jako liczbową reprezentację cząsteczki.

**Wejście:**

- Morgan fingerprint,
- promień: 2,
- długość: 2048 bitów.

**Architektura:**

```text
2048 -> 512 -> 128 -> 1
```

Model wykorzystuje:

- aktywację ReLU,
- funkcję straty MSE,
- optymalizator Adam.

MLP pełni rolę klasycznego baseline'u, z którym można porównywać modele grafowe.

### Finalny model GINE

Do finalnej predykcji wykorzystano model **GINE** (`Graph Isomorphism Network with Edge Features`).

W odróżnieniu od prostego GCN model GINE wykorzystuje zarówno:

- cechy atomów,
- cechy wiązań chemicznych.

**Przykładowe cechy atomów:**

- liczba atomowa,
- stopień atomu,
- ładunek formalny,
- aromatyczność,
- przynależność do pierścienia,
- liczba atomów wodoru,
- hybrydyzacja.

**Przykładowe cechy wiązań:**

- typ wiązania,
- sprzężenie,
- przynależność do pierścienia,
- stereochemia.

**Architektura finalnego modelu:**

- liniowy encoder cech atomowych,
- 3 warstwy `GINEConv`,
- `BatchNorm`,
- aktywacja ReLU,
- dropout `0.1`,
- połączenie `mean`, `add` i `max pooling`,
- głowica regresyjna zakończona pojedynczą wartością `pIC50`.

Wytrenowany checkpoint znajduje się w:

```text
models/gine_egfr_chembl203.pt
```

## Wyniki finalnego modelu GINE

Wyniki zapisane w notebooku treningowym:

| Split | RMSE | MAE | R² |
|---|---:|---:|---:|
| Random | 0.918 | 0.686 | 0.544 |
| Scaffold | 0.963 | 0.749 | 0.393 |

Model GINE został wybrany do finalnej aplikacji ze względu na grafową reprezentację cząsteczek oraz możliwość wykorzystania informacji o atomach i wiązaniach. MLP pozostaje ważnym modelem bazowym i w części eksperymentów może osiągać lepsze wyniki.

## Struktura repozytorium

```text
ChEMBL-analysis/
├── EDA_ChEMBL.ipynb
├── data_preparation.py
├── prepare_dataset.ipynb
├── splits.py
├── mlp_model.py
├── train_mlp.ipynb
├── gnn_model.py
├── train_gnn.ipynb
├── predict_gnn.ipynb
├── predict_smiles.py
├── app.py
├── models/
│   └── gine_egfr_chembl203.pt
├── prepared_data/
│   ├── chembl_ic50_model_dataset.csv
│   ├── train_random.csv
│   ├── val_random.csv
│   ├── test_random.csv
│   ├── train_scaffold.csv
│   ├── val_scaffold.csv
│   ├── test_scaffold.csv
│   └── train_random_tiny.csv
└── README.md
```

## Instalacja

Zalecany Python: **3.11**.

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/BStchw/ChEMBL-analysis.git
cd ChEMBL-analysis
```

### 2. Utworzenie środowiska wirtualnego

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalacja bibliotek potrzebnych do aplikacji

```bash
python -m pip install --upgrade pip
pip install streamlit numpy pandas rdkit torch torch-geometric
```

Do uruchamiania notebooków i całego pipeline'u mogą być również potrzebne:

```bash
pip install jupyter matplotlib scikit-learn pyspark
```

## Uruchomienie aplikacji Streamlit

W katalogu głównym projektu wykonaj:

```bash
streamlit run app.py
```

Aplikacja:

1. wczyta checkpoint modelu z katalogu `models/`,
2. przyjmie SMILES podany przez użytkownika,
3. zamieni cząsteczkę na graf,
4. wykona predykcję `pIC50`,
5. przeliczy wynik na przybliżone `IC50` w nM.

## Predykcja bez interfejsu

Przykładową predykcję można uruchomić również z terminala:

```bash
python predict_smiles.py
```

Skrypt wykorzystuje przykładową cząsteczkę SMILES i wypisuje:

- nazwę targetu,
- podany SMILES,
- przewidywane `pIC50`,
- przybliżone `IC50` w nM.

## Notebooki

Zalecana kolejność pracy:

1. `EDA_ChEMBL.ipynb` – eksploracyjna analiza danych,
2. `prepare_dataset.ipynb` – przygotowanie zbioru i splitów,
3. `train_mlp.ipynb` – trening modelu MLP,
4. `train_gnn.ipynb` – trening i ewaluacja modeli grafowych,
5. `predict_gnn.ipynb` – testowanie predykcji zapisanego modelu.

Notebooki można uruchomić poleceniem:

```bash
jupyter notebook
```

## Technologie

Projekt wykorzystuje między innymi:

- Python,
- pandas,
- NumPy,
- PySpark,
- RDKit,
- PyTorch,
- PyTorch Geometric,
- Streamlit,
- Jupyter Notebook,
- Matplotlib.

## Źródło danych

Dane pochodzą z bazy [ChEMBL](https://www.ebi.ac.uk/chembl/), zawierającej informacje o bioaktywnych cząsteczkach i ich aktywności względem targetów biologicznych.

