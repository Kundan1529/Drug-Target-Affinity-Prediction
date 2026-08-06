# Drug Target Affinity Prediction using Graph Neural Networks and ESM2

<p align="center">
Predicting drug–target binding affinity using Graph Neural Networks (GINEConv) and ESM2 protein embeddings with an interaction-aware fusion network.
</p>

---

## Overview

Drug–Target Affinity (DTA) prediction is a fundamental task in computational drug discovery, where the objective is to estimate the binding strength between a drug molecule and a target protein.

This project implements an end-to-end deep learning framework that combines:

- **Graph Neural Networks (GINEConv)** for learning molecular representations from chemical graphs.
- **ESM2 Protein Language Model** embeddings for capturing protein sequence semantics.
- **Interaction-aware fusion** using element-wise multiplication and absolute difference between drug and protein embeddings.
- A deep Multi-Layer Perceptron (MLP) for affinity prediction.

Unlike traditional approaches that rely on handcrafted descriptors, this model learns expressive representations directly from molecular structures and protein sequences.

Everything — data loading, EDA, featurization, model, training, evaluation, inference, and explainability — is contained in a single notebook: [`drug_target_affinity_prediction.ipynb`](drug_target_affinity_prediction.ipynb), stored with all execution outputs so the reported results can be inspected without rerunning.

---

# Model Architecture

```
       Drug SMILES                        Protein Sequence
            │                                    │
            ▼                                    ▼
  RDKit Graph Construction                ESM2 Protein Encoder
            │                              (frozen, 150M)
            ▼                                    │
      Molecular Graph                       mean pooling
            │                                    │
            ▼                                    ▼
   Graph Neural Network                 Protein Embedding
      (3 × GINEConv)                         (640-d)
            │                                    │
            ▼                                    │
      Drug Embedding                             │
          (128-d)                                │
            │                                    │
            └────────► project both to 256 ◄─────┘
                            │
                            ▼
                   Interaction Features
                   ─────────────────────
                   • Drug Embedding
                   • Protein Embedding
                   • Drug × Protein
                   • |Drug − Protein|
                        (1024-d)
                            │
                            ▼
                      Fusion Network
                   512 → 512 → 256 → 1
                            │
                            ▼
                    Affinity Prediction
                          (pKd)
```

---

# Features

- Graph-based molecular representation using **RDKit**
- **GINEConv** Graph Neural Network with edge (bond) features in message passing
- Precomputed, cached **ESM2** protein embeddings
- Interaction-aware feature fusion
- Batch Normalization
- SiLU activation functions in the fusion network
- AdamW optimizer
- **ReduceLROnPlateau** learning rate scheduler
- Mixed precision training (AMP) with gradient clipping
- Automatic checkpointing with resume support
- Early stopping on validation loss
- Evaluation using multiple regression metrics
- Model explainability with **GNNExplainer**

---

# Dataset

## Davis Drug–Target Affinity Dataset

The Davis dataset contains experimentally measured binding affinities between kinase inhibitors and protein targets. It is loaded from the official [DeepDTA](https://github.com/hkmztrk/DeepDTA) repository so results remain comparable to published DTA baselines.

### Statistics

| Property | Value |
|----------|------:|
| Drugs | 68 |
| Proteins | 442 |
| Total Interactions | 30,056 |
| Mean pKd | 5.452 |
| Std | 0.895 |
| Range | 5.000 – 10.796 |

Binding affinity values are converted from **Kd (nM)** to **pKd** using:

$$pKd = -\log_{10}\left(\frac{Kd}{10^{9}}\right)$$

### Splits

The official Davis folds (`train_fold_setting1` / `test_fold_setting1`) are used. Of the five training folds, fold 0 is held out for validation and the remaining four are used for training.

| Split | Samples |
|-------|--------:|
| Train | 20,036 |
| Validation | 5,010 |
| Test | 5,010 |

---

# Drug Encoder

Each drug molecule is parsed with RDKit and converted into a molecular graph, where atoms become nodes and bonds become edges.

### Node Features

- Atom Type (one-hot: C, N, O, S, F, P, Cl, Br, I, H, Unknown)
- Hybridization (one-hot: SP, SP2, SP3, SP3D, SP3D2)
- Atomic Number
- Degree
- Formal Charge
- Number of Hydrogens
- Total Valence
- Aromaticity
- Ring Membership

### Edge Features

- Bond Type (one-hot: single, double, triple, aromatic)
- Conjugation
- Ring Membership

The graph is processed by three stacked **GINEConv** layers — chosen because they consume edge features directly during message passing — each followed by BatchNorm, ReLU, and dropout (0.2), then reduced to a 128-d molecule vector by global mean pooling.

---

# Protein Encoder

Protein sequences are represented using **ESM2** (`facebook/esm2_t30_150M_UR50D`), a transformer-based protein language model pretrained on millions of protein sequences.

The encoder is kept **frozen**. Sequences are tokenized (max length 1024), passed through ESM2 once, mean-pooled over the attention mask into a 640-d vector, and cached to disk. Since Davis contains only 442 unique proteins, computing these embeddings a single time and reusing them across every epoch keeps training fast and avoids fine-tuning a large language model on a small dataset.

---

# Interaction-aware Fusion

Instead of simply concatenating the drug and protein embeddings, the model explicitly models their interactions.

Both modalities are first projected into a shared 256-d space. The final fusion vector concatenates:

```
Drug Embedding

Protein Embedding

Drug × Protein

|Drug − Protein|
```

These complementary representations help the network learn both similarity and interaction patterns between drugs and proteins. The resulting 1024-d vector passes through an MLP (512 → 512 → 256 → 1) with BatchNorm, SiLU, and dropout (0.3).

**Total trainable parameters: 1,324,161**

---

# Training Configuration

| Setting | Value |
|---------|-------|
| Loss | MSE |
| Optimizer | AdamW (lr `1e-3`, weight decay `1e-4`) |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 3, min lr `1e-6`) |
| Batch Size | 64 |
| Epochs | 50 |
| Early Stopping | patience 7 on validation loss |
| Mixed Precision | AMP (`GradScaler` + `autocast`) |
| Gradient Clipping | max norm 1.0 |
| Dropout | 0.2 (GNN) / 0.3 (fusion) |
| Seed | 42 |
| Hardware | NVIDIA Tesla T4 (Kaggle) |

Training ran the full 50 epochs without triggering early stopping, finishing at a train loss of 0.3295 and validation loss of 0.2890. `best_model.pt` tracks the best validation loss, while `latest.pt` stores the model, optimizer, scheduler, history, and best loss every epoch so training can be resumed after an interruption.

---

# Model Performance

## Davis Dataset (official test fold, 5,010 samples)

| Metric | Score |
|---------|------:|
| **MSE** | **0.3114** |
| **RMSE** | **0.5580** |
| **MAE** | **0.3112** |
| **R² Score** | **0.6115** |
| **Pearson Correlation** | **0.7882** |
| **Concordance Index (CI)** | **0.8657** |

Diagnostic plots in the notebook include predicted-vs-actual scatter against the identity line and a residual distribution histogram. Metrics are also written to `checkpoints/test_results.json`.

---

# Explainability

Because a fused GNN + protein language model is otherwise a black box, the notebook wraps the trained drug encoder and runs PyTorch Geometric's **GNNExplainer** (200 epochs, graph-level regression mode) to produce node and edge importance masks. This identifies which atoms, bonds, and molecular substructures drive a given affinity prediction, with the molecule rendered via RDKit for visual inspection.

---

# Inference

Predicting affinity for a new drug–protein pair is a single call:

```python
affinity = predict_affinity(
    smiles="CCO",
    sequence="MTEITAAMVKELRESTGAGMMDCKNALSETQHE...",
)
```

Internally: SMILES → RDKit → molecular graph, sequence → ESM2 → 640-d embedding, then both through the trained model. Batch predictions can be exported to `checkpoints/predictions.csv`.

---

# Project Structure

```
Drug-Target-Affinity-Prediction
│
├── drug_target_affinity_prediction.ipynb    # full pipeline, with outputs
│
├── README.md
│
└── LICENSE
```

Running the notebook generates the following locally (not tracked in the repository):

```
DTI_Project/
├── processed/
│   ├── graphs/drug_graphs.pt              # molecular graphs
│   └── protein_embeddings/                # cached ESM2 embeddings
│       └── esm2_embeddings.pt
├── artifacts/
├── models/
└── figures/

checkpoints/
├── best_model.pt                          # best validation checkpoint
├── latest.pt                              # resumable training state
├── history.json                           # per-epoch losses and lr
├── test_results.json                      # final test metrics
└── sample_explanation.pt                  # GNNExplainer output

DeepDTA/                                   # cloned Davis dataset
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Kundan1529/Drug-Target-Affinity-Prediction.git
```

Create and activate a virtual environment

Linux / macOS

```bash
python -m venv venv && source venv/bin/activate
```

Windows

```bash
python -m venv venv && venv\Scripts\activate
```

Install dependencies

```bash
pip install torch torch-geometric rdkit transformers accelerate biopython
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy tqdm
```

---

# Usage

The notebook was written for **Kaggle with a Tesla T4 GPU** and installs its own dependencies in the opening cells. It runs on any CUDA machine, and falls back to CPU — though ESM2 embedding generation and training will be considerably slower.

Open the notebook and run the cells in order. It downloads the Davis dataset itself:

```bash
git clone https://github.com/hkmztrk/DeepDTA.git
```

One path is environment-specific and needs editing outside Kaggle:

```python
DATASET_ROOT = Path("/kaggle/working/DeepDTA/data/davis")
```

The notebook then automatically:

- Loads and explores the dataset
- Builds molecular graphs from SMILES
- Generates and caches ESM2 protein embeddings
- Trains the model with checkpointing and early stopping
- Evaluates on the official test fold
- Runs inference and explainability

---

# Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- Hugging Face Transformers
- RDKit
- Biopython
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn

---

# Future Improvements

- Graph Attention Networks (GAT)
- Jumping Knowledge Networks
- Cross-Attention Drug–Protein Fusion
- Multi-task Learning
- Protein Structure Integration
- Hyperparameter Optimization using Optuna
- Refactoring the notebook into a modular `src/` package with `train.py` and `evaluate.py`
- Benchmarking on additional datasets (KIBA, BindingDB)

---

# Citation

If you use this repository in your research, please consider citing the original Davis dataset and ESM2 paper.

- Davis, M. I. et al. *Comprehensive analysis of kinase inhibitor selectivity.* Nature Biotechnology (2011)
- Öztürk, H. et al. *DeepDTA: deep drug–target binding affinity prediction.* Bioinformatics (2018)
- Lin, Z. et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science (2023)
- Hu, W. et al. *Strategies for Pre-training Graph Neural Networks.* ICLR (2020)

---

# Acknowledgements

- Davis Drug–Target Affinity Dataset
- PyTorch Geometric
- RDKit
- Hugging Face Transformers
- FAIR ESM2

---

# Contact

If you have any questions, suggestions, or would like to collaborate, feel free to open an issue or reach out through GitHub.

---

## License

This project is released under an **Educational Use License** — see [LICENSE](LICENSE) for the full terms.

You are free to use, modify, and share it for **educational, academic, and non-commercial research** purposes with attribution. Commercial use requires prior written permission.

The model is intended for research and learning only. It is **not validated for clinical, diagnostic, or therapeutic use** and must not be relied upon for medical decision-making or as a substitute for experimental validation.

The Davis dataset, ESM2 weights, and all third-party libraries remain under their own licenses.
