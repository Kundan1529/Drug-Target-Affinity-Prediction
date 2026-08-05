# Drug–Target Affinity Prediction

Predicting drug–protein binding affinity on the **Davis** benchmark with a two-branch deep learning model: a **Graph Neural Network (GINE)** over molecular graphs for the drug, and a frozen **ESM2 protein language model** for the target.

The full pipeline — data loading, exploratory analysis, featurization, model, training, evaluation, inference, and explainability — lives in a single notebook: [`drug_target_affinity_prediction.ipynb`](drug_target_affinity_prediction.ipynb).

---

## Why this problem

Experimentally screening every candidate drug against every candidate protein is prohibitively slow and expensive. Drug–Target Interaction (DTI) prediction estimates binding affinity computationally, which is useful for narrowing screening libraries, repurposing existing drugs, and personalized medicine.

This project treats it as a **regression** task: given a drug (SMILES) and a protein (amino acid sequence), predict the binding affinity as **pKd**.

---

## Approach

```
   SMILES string                        Amino acid sequence
        │                                       │
        ▼                                       ▼
   RDKit molecule                        ESM2 (frozen)
   atoms → nodes                    esm2_t30_150M_UR50D
   bonds → edges                            │
        │                              mean pooling
        ▼                                    │
  GINEConv × 3  (128-d)                      ▼
  global mean pool                    640-d embedding
        │                                    │
        └──────────► project both to 256 ◄───┘
                            │
        concat [ drug, protein, drug ⊙ protein, |drug − protein| ]  → 1024-d
                            │
                  MLP 512 → 512 → 256 → 1
                            │
                            ▼
                  predicted binding affinity (pKd)
```

**Drug branch.** Each molecule is parsed with RDKit and converted into a PyTorch Geometric graph. Atoms carry one-hot element and hybridization features plus atomic number, degree, formal charge, hydrogen count, valence, aromaticity, and ring membership. Bonds carry one-hot bond type plus conjugation and ring flags. Three `GINEConv` layers (which consume edge features directly during message passing — the reason for choosing GINE over plain GIN/GCN) with BatchNorm, ReLU, and dropout, followed by global mean pooling, produce a 128-d molecule embedding.

**Protein branch.** Sequences are encoded once by the pretrained, frozen `facebook/esm2_t30_150M_UR50D`, mean-pooled over the attention mask into a 640-d vector, and cached to disk. Since Davis has only 442 unique proteins, embeddings are computed a single time and reused across every epoch — this keeps training fast and avoids fine-tuning a large PLM on a small dataset.

**Fusion.** Both modalities are projected into a shared 256-d space, then combined as `[drug, protein, drug ⊙ protein, |drug − protein|]`. The elementwise product and absolute difference give the MLP explicit pairwise interaction signal rather than making it learn one from concatenation alone.

---

## Dataset

The **Davis** kinase inhibitor dataset, taken from the official [DeepDTA](https://github.com/hkmztrk/DeepDTA) repository (`data/davis`), so results are comparable to published DTI baselines.

| | |
|---|---|
| Drugs | 68 (SMILES) |
| Proteins | 442 (sequences) |
| Drug–protein pairs | ~30,000 (dense affinity matrix) |
| Label | Kd in nM, converted to pKd via `-log10(Kd / 1e9)` |
| Splits | official `train_fold_setting1` / `test_fold_setting1` folds |

The notebook downloads the dataset directly:

```bash
git clone https://github.com/hkmztrk/DeepDTA.git
```

Of the five official training folds, **fold 0 is held out as validation** and the remaining four are used for training; the official test fold is untouched until final evaluation.

---

## What the notebook covers

| Section | Contents |
|---|---|
| 1–2 | Setup, dependencies, seeding, device, project directories |
| 3 | Loading the Davis ligands, proteins, affinity matrix, and folds |
| 4 | Dataset EDA — dimensions, SMILES/sequence length distributions, affinity distribution, missing values, duplicates |
| 5 | Molecular descriptors via RDKit (MolWt, LogP, TPSA, rotatable bonds, H-donors/acceptors, rings) and their correlations |
| 6 | Protein sequence analysis — length distribution, amino acid composition, non-standard residue check |
| 7–8 | Molecular graph construction, from a basic version to a richer one-hot chemical featurization |
| 9 | ESM2 embedding generation and caching |
| 10–11 | `Dataset` classes and the official fold-based train/valid/test split |
| 12 | Custom collate function and PyTorch Geometric `DataLoader` batching of graphs + embeddings + labels |
| 13–14 | `DrugEncoder` (GINE) and the `DTIModel` fusion network |
| 15–16 | Training utilities and the training loop |
| 17 | Evaluation on the official test fold |
| 18 | End-to-end inference for an arbitrary SMILES + sequence pair |
| 19 | Explainability with `GNNExplainer` |

---

## Training setup

| Setting | Value |
|---|---|
| Loss | MSE |
| Optimizer | AdamW, lr `1e-3`, weight decay `1e-4` |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 3, min lr `1e-6`) |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | patience 7 on validation loss |
| Mixed precision | AMP via `torch.amp` (`GradScaler` + `autocast`) |
| Regularization | dropout 0.2 (GNN) / 0.3 (fusion), BatchNorm, gradient clipping |
| Seed | 42 |
| Hardware | Kaggle Tesla T4 |

Checkpointing is built in: `latest.pt` is written every epoch (model, optimizer, scheduler, history, best loss) so training can resume after an interruption, and `best_model.pt` tracks the best validation loss.

---

## Evaluation

The model is scored on the official Davis test fold with the metrics standard in the DTI literature:

- **MSE**, **RMSE**, **MAE**
- **Pearson correlation**
- **R²**
- **Concordance Index (CI)** — the ranking metric most commonly reported for DTA benchmarks

Diagnostics include a predicted-vs-actual scatter plot against the identity line and a residual histogram. Results are written to `checkpoints/test_results.json`.

> **Note:** the notebook in this repository is stored without execution outputs, so no metric values are quoted here. Run the notebook end to end to reproduce them.

---

## Explainability

Because a fused GNN + PLM model is otherwise a black box, Section 19 wraps the drug encoder and runs PyTorch Geometric's `GNNExplainer` (200 epochs, regression / graph-level mode) to produce node and edge importance masks. This surfaces which atoms, bonds, and substructures drive a given affinity prediction, and the molecule is rendered with RDKit's `Draw` for visual inspection.

---

## Inference

Once trained, predicting affinity for a new pair is three steps:

```python
affinity = predict_affinity(
    smiles="CCO",
    sequence="MTEITAAMVKELRESTGAGMMDCKNALSETQHE...",
)
```

Internally: SMILES → RDKit → graph, sequence → ESM2 → 640-d embedding, then both through the trained `DTIModel`. Batch predictions can be written out to `checkpoints/predictions.csv`.

---

## Running it

The notebook was written for **Kaggle with a Tesla T4 GPU** and installs its own dependencies in the first cells. It runs anywhere with a CUDA GPU (it falls back to CPU, though ESM2 embedding generation and training will be slow).

```bash
pip install torch torch-geometric rdkit transformers accelerate biopython
pip install pandas numpy matplotlib seaborn scikit-learn scipy tqdm
```

Then open the notebook and run the cells in order. One path is environment-specific and may need editing outside Kaggle:

```python
DATASET_ROOT = Path("/kaggle/working/DeepDTA/data/davis")
```

Generated artifacts (graphs, cached embeddings, checkpoints, figures) are written under `DTI_Project/` and `checkpoints/`.

---

## Stack

PyTorch · PyTorch Geometric · RDKit · Hugging Face Transformers (ESM2) · Biopython · scikit-learn · pandas · NumPy · Matplotlib · Seaborn

---

## References

- Davis et al., *Comprehensive analysis of kinase inhibitor selectivity*, Nature Biotechnology (2011)
- Öztürk et al., *DeepDTA: deep drug–target binding affinity prediction*, Bioinformatics (2018) — [github.com/hkmztrk/DeepDTA](https://github.com/hkmztrk/DeepDTA)
- Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM2), Science (2023)
- Hu et al., *Strategies for Pre-training Graph Neural Networks* (GINE), ICLR (2020)
