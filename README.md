# Drug Target Affinity Prediction using Graph Neural Networks and ESM2

<p align="center">
  <img src="images/pipeline.png" width="900">
</p>

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

---

# Model Architecture

```
                    Drug SMILES
                         │
                         ▼
                RDKit Graph Construction
                         │
                         ▼
                  Molecular Graph
                         │
                         ▼
                Graph Neural Network
                   (3 × GINEConv)
                         │
                         ▼
                  Drug Embedding
                         │
                         │
Protein Sequence         │
        │                │
        ▼                │
ESM2 Protein Encoder      │
        │                │
        ▼                │
Protein Embedding         │
        │                │
────────┴────────────────┘
             │
             ▼
     Interaction Features
     ─────────────────────
     • Drug Embedding
     • Protein Embedding
     • Drug × Protein
     • |Drug − Protein|
             │
             ▼
        Fusion Network
             │
             ▼
     Affinity Prediction
```

---

# Features

- Graph-based molecular representation using **RDKit**
- **GINEConv** Graph Neural Network
- Precomputed **ESM2** protein embeddings
- Interaction-aware feature fusion
- Batch Normalization
- SiLU activation functions
- AdamW optimizer
- Cosine Annealing Learning Rate Scheduler
- Automatic checkpointing
- Mixed precision training
- Evaluation using multiple regression metrics

---

# Dataset

## Davis Drug–Target Affinity Dataset

The Davis dataset contains experimentally measured binding affinities between kinase inhibitors and protein targets.

### Statistics

| Property | Value |
|----------|------:|
| Drugs | 68 |
| Proteins | 442 |
| Total Interactions | 30,056 |

Binding affinity values are converted from **Kd (nM)** to **pKd** using:

\[
pKd = -\log_{10}\left(\frac{Kd}{10^9}\right)
\]

---

# Drug Encoder

Each drug molecule is converted into a molecular graph.

### Node Features

- Atom Type
- Degree
- Formal Charge
- Hybridization
- Aromaticity
- Number of Hydrogens
- Chirality

### Edge Features

- Bond Type
- Bond Stereo
- Ring Information
- Conjugation

The graph is processed using three stacked **GINEConv** layers followed by global pooling.

---

# Protein Encoder

Protein sequences are represented using **ESM2**, a transformer-based protein language model pretrained on millions of protein sequences.

Instead of training the transformer from scratch, precomputed embeddings are used during training for computational efficiency.

---

# Interaction-aware Fusion

Instead of simply concatenating the drug and protein embeddings, the model explicitly models their interactions.

The final fusion vector consists of:

```
Drug Embedding

Protein Embedding

Drug × Protein

|Drug − Protein|
```

These complementary representations help the network learn both similarity and interaction patterns between drugs and proteins.

---

# Model Performance

## Davis Dataset

| Metric | Score |
|---------|------:|
| **MSE** | **0.3401** |
| **RMSE** | **0.5832** |
| **MAE** | **0.3303** |
| **R² Score** | **0.5757** |
| **Pearson Correlation** | **0.7646** |
| **Concordance Index (CI)** | **0.8469** |

---

# Project Structure

```
Drug-Target-Affinity-Prediction
│
├── data/
│
├── checkpoints/
│
├── notebooks/
│
├── src/
│   ├── dataset.py
│   ├── graph.py
│   ├── protein.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── checkpoint.py
│   └── utils.py
│
├── images/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Kundan1529/Drug-Target-Affinity-Prediction.git

cd Drug-Target-Affinity-Prediction
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Linux / macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run

```bash
python train.py
```

Training automatically

- Loads the dataset
- Builds molecular graphs
- Loads ESM2 embeddings
- Trains the model
- Saves checkpoints
- Logs training history

---

# Evaluation

Run

```bash
python evaluate.py
```

The evaluation reports

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Pearson Correlation
- R² Score
- Concordance Index

---

# Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- Hugging Face Transformers
- RDKit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# Future Improvements

- Graph Attention Networks (GAT)
- Jumping Knowledge Networks
- Cross-Attention Drug–Protein Fusion
- Multi-task Learning
- Protein Structure Integration
- Hyperparameter Optimization using Optuna
- Explainable AI for Drug–Target Interaction Prediction

---

# Citation

If you use this repository in your research, please consider citing the original Davis dataset and ESM2 paper.

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

This project is released under the **MIT License**.
