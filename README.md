# Aitchison Compositional Graph Embeddings (AICoG)

> Official PyTorch implementation of **"Aitchison Embeddings for Learning Compositional Graph Representations"**, accepted at **ICML 2026**.

AICoG is a role-based graph embedding framework that represents each node as a **composition over latent archetypal factors** and compares nodes using **Aitchison geometry** — the canonical geometry for compositional data. Compositions are embedded via **isometric log-ratio (ILR)** coordinates, which preserve Aitchison distances while enabling unconstrained optimization in Euclidean space. The result is an embedding model that is **interpretable by construction**, supports **subcompositional coherence** (principled component restriction without retraining), and matches or exceeds strong baselines on link prediction and node classification.

---

## Table of Contents

- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Usage](#usage)
- [Command-Line Arguments](#command-line-arguments)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Citation](#citation)
---

## Highlights

- **Compositional node representations.** Each node is a point on the simplex $\Delta^{K-1}$ — a graded mixture over $K$ latent archetypes.
- **Aitchison geometry via ILR.** Compositions are embedded into $\mathbb{R}^{K-1}$ through an isometric log-ratio transformation, so Euclidean distances in the embedding space exactly equal Aitchison distances on the simplex.
- **Expressive equivalence.** AICoG matches the expressive power of unconstrained Euclidean latent distance models while endowing distances with invariant compositional semantics.
- **Interpretability by construction.** Distances quantify *relative trade-offs* between archetypes, not arbitrary directions. Supports both fixed (Helmert) and learned ILR bases.
- **Subcompositional coherence.** Components can be removed and the composition reclosed without retraining, while preserving a well-defined geometry — useful for probing archetype influence.
- **Strong empirical performance.** Competitive with NODE2VEC, ROLE2VEC, NETMF, MMSBM, MNMF, HM-LDM, and SLIM-RAA on link prediction and node classification across 7 benchmark networks.

---

## Method Overview

Given an undirected graph $G = (V, E)$, AICoG learns:

1. A composition $\mathbf{z}_i \in \Delta^{K-1}$ for every node $i$, parameterized via softmax over unconstrained logits.
2. ILR coordinates $\mathbf{x}_i = \log(\mathbf{z}_i)^\top \mathbf{V} \in \mathbb{R}^{K-1}$, where $\mathbf{V}$ is an orthonormal basis of the contrast space (Helmert or learned).
3. Node-specific bias terms $\gamma_i$ capturing degree heterogeneity.

Edges are modeled with a Bernoulli likelihood whose log-odds are

$$\eta_{ij} = -\| \mathbf{x}_i - \mathbf{x}_j \|_2 + \gamma_i + \gamma_j .$$

Training uses uniform negative sampling of non-edges to keep the per-iteration cost at $\mathcal{O}(|E|)$.

---

## Installation

The code was developed with **Python 3.8.3** and **PyTorch 1.9.0**, and is also compatible with the versions pinned in `requirements.txt`.

```bash
git clone https://github.com/<your-username>/AICoG.git
cd AICoG
pip install -r requirements.txt
```

GPU support (CUDA) is recommended but not required.

---

## Repository Structure

```
AICoG/
├── main.py                 # Training / evaluation entry point
├── AICoG.py                # LSM model: ILR embeddings, likelihood, link prediction
├── spectral_clustering.py  # Spectral initialization for latent positions
├── requirements.txt        # Python dependencies
├── dataset_format.txt      # Description of expected dataset files
└── datasets/
    └── <dataset_name>/
        ├── sparse_i.txt           # Edge row indices    (i < j)
        ├── sparse_j.txt           # Edge column indices (i < j)
        ├── sparse_i_rem.txt       # [LP only] held-out positive edges, rows
        ├── sparse_j_rem.txt       # [LP only] held-out positive edges, columns
        ├── non_sparse_i.txt       # [LP only] sampled negative edges, rows
        ├── non_sparse_j.txt       # [LP only] sampled negative edges, columns
        └── labels.txt             # [Classification only] one integer label per node
```

---

## Datasets

AICoG expects the **upper-triangular sparse representation** of an undirected adjacency matrix. See `dataset_format.txt` for full details.

**Network input** (always required):

| File | Description |
|------|-------------|
| `sparse_i.txt` | Row indices $i$ of edges, with $i < j$ |
| `sparse_j.txt` | Column indices $j$ of edges, with $i < j$ |

**Link prediction** (set `--LP True`): the residual graph is stored in `sparse_i.txt`/`sparse_j.txt`, and the held-out edges plus negative samples in:

| File | Description |
|------|-------------|
| `sparse_i_rem.txt`, `sparse_j_rem.txt` | Held-out positive edges ($i < j$) |
| `non_sparse_i.txt`, `non_sparse_j.txt` | Sampled negative edges ($i < j$) |

**Node classification** (set `--clas True`): provide a `labels.txt` with one integer label per line, in node order $0$ to $N-1$. Use `-1` to mark unlabeled nodes.

The paper evaluates on **Cora**, **Citeseer**, **DBLP**, **AstroPh**, **GrQc**, **HepTh**, and **LastFM**.

---

## Usage

### Train embeddings (no downstream task)

```bash
python main.py --K 9 --dataset cora --LP False --clas False
```

This trains AICoG on the full graph and saves the model state dict to
`./datasets/cora/params_AICoG_cora_9_0.pt`.

### Link prediction

```bash
python main.py --K 9 --dataset cora --LP True --clas False
```

Reports ROC-AUC and PR-AUC on the held-out edges.

### Node classification

```bash
python main.py --K 9 --dataset cora_with_labels --LP False --clas True
```

Trains a logistic regression classifier on the learned embeddings and reports Micro-F1 and Macro-F1 (with regularization tuned on a validation split).

### Simplex Euclidean baseline (ablation)

To learn embeddings directly on the simplex *without* the ILR transformation:

```bash
python main.py --K 9 --dataset cora --euclidean True
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | `int` | `5000` | Number of training epochs |
| `--scaling_epochs` | `int` | `500` | Epochs for learning the initial scale of random effects |
| `--cuda` | `bool` | `True` | Enable CUDA training if available |
| `--LP` | `bool` | `False` | Run link prediction evaluation |
| `--clas` | `bool` | `False` | Run node classification evaluation |
| `--K` | `int` | `9` | Number of compositional components ($K$). The ILR/embedding dimension is $K-1$. |
| `--lr` | `float` | `0.01` | Adam learning rate |
| `--dataset` | `str` | `cora` | Dataset folder name under `./datasets/` |
| `--euclidean` | `bool` | `False` | Use the Simplex-Euclidean baseline (no ILR projection) |

> For node classification, use a `_with_labels` dataset variant. For link prediction, use the standard dataset (with held-out edge files).

---

## Reproducing Paper Results

The paper reports results averaged over five runs with embedding sizes $D \in \{8, 16, 32, 64\}$, corresponding to $K \in \{9, 17, 33, 65\}$.

**Link prediction** (Table 2 in the paper):

```bash
python main.py --K 9  --dataset cora --LP True --clas False
python main.py --K 17 --dataset cora --LP True --clas False
python main.py --K 33 --dataset cora --LP True --clas False
python main.py --K 65 --dataset cora --LP True --clas False
```

**Node classification** (Table 3 in the paper):

```bash
python main.py --K 9  --dataset cora_with_labels --LP False --clas True
python main.py --K 17 --dataset cora_with_labels --LP False --clas True
python main.py --K 33 --dataset cora_with_labels --LP False --clas True
python main.py --K 65 --dataset cora_with_labels --LP False --clas True
```

Across runs, AICoG achieves competitive or favorable performance against all baselines while providing intrinsically interpretable role representations.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{nakis2026aicog,
  title     = {Aitchison Embeddings for Learning Compositional Graph Representations},
  author    = {Nakis, Nikolaos and Kosma, Chrysoula and Promponas, Panagiotis and
                Chatzianastasis, Michail and Nikolentzos, Giannis},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---
