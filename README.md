# MolPerturbDiff

MolPerturbDiff is a structure-conditioned diffusion framework designed to predict drug-induced transcriptional responses from chemical structure. Cellular responses to small molecules arise from complex, context-dependent regulatory mechanisms. Traditional approaches often rely on predefined targets or direct regression, which limits their ability to capture global gene-level perturbations and the inherent uncertainty in cellular responses. MolPerturbDiff addresses this challenge by modeling gene-wise expression changes as stochastic perturbations conditioned on molecular structure.

---

![MolPerturbDiff Workflow](images/Graphical%20Abstract.png)

## Features

- **Structure-conditioned diffusion model:** Predicts post-treatment gene expression changes given pre-treatment profiles.
- **Gene-wise gating mechanism:** Links chemical embeddings (from SMILES) to the direction and relative strength of gene regulation.
- **Conditional diffusion process:** Captures uncertainty and heterogeneity in cellular responses.
- **Pretrained molecular encoder:** SMILES encoder trained on an extended chemical space and frozen during diffusion training.
- **Large-scale generalization:** Trained on transcriptional perturbation data across diverse cell types, enabling accurate prediction for previously unseen molecules.
- **Interpretable and scalable:** Provides structure-aware, phenotype-driven virtual drug screening.

---

## Project Structure

The project is organized into modular Python scripts. Currently, the modules are **independent** and do not automatically call each other. A `run.py` script provides a full end-to-end pipeline.

| File | Description |
|------|-------------|
| `dataset.py` | Handles data loading, SMILES tokenization, Δ (delta) computation between pre/post-treatment gene expression, and dataset classes. |
| `model.py` | Defines neural network architectures: SMILES encoder and conditional diffusion model. |
| `pretrain.py` | Performs contrastive pretraining of the SMILES encoder on molecular SMILES data. |
| `train.py` | Trains the conditional diffusion model on Δ gene expression data. |
| `predict.py` | Generates post-treatment gene expression predictions from pre-treatment profiles and molecular SMILES using a trained model. |
| `run.py` | End-to-end script integrating all stages: data loading, SMILES encoder pretraining, Δ computation, diffusion model training, and post-treatment prediction. |

---

## Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- NumPy
- Pandas
- scikit-learn
- tqdm

Install dependencies via:

```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn tqdm
```

---

## Usage

### 1. Data Preparation

- **Expression data:** `expression.csv` containing pre- and post-treatment gene expression values with metadata columns:  
  `Sample_id, Batch_id, Condition_id, Replicate_id, Molecule_id, Treatment_status`  
- **Molecule data:** `molecules.csv` with `Molecule_id`, `Canonical_SMILES`, and `Source` columns.

### 2. Pretrain SMILES Encoder

Pretraining learns molecular embeddings from SMILES sequences.  
- Input: `molecules.csv`  
- Output: `encoder_pretrain_contrastive.pt`  

```bash
python pretrain.py
```

### 3. Build Δ Pairs

Computes the gene expression difference between post-treatment and DMSO pre-treatment samples (Δ).  
- Input: `expression.csv`  
- Output: `pairs_fold{n}.pkl`, `delta_norm_fold{n}.pt`  

This is integrated in `run.py` or can be executed standalone.

### 4. Train Diffusion Model

Trains the conditional diffusion model to predict Δ given pre-treatment expression and molecular embedding.  
- Input: Δ pairs and pretrained encoder  
- Output: `checkpoints/final_model.pt`  

```bash
python train.py
```

### 5. Predict Post-treatment Expression

Generates post-treatment gene expression profiles for new molecules.  
- Input: Pre-treatment expression and molecule SMILES  
- Output: `out.csv`  

```bash
python predict.py
```

### 6. Full Pipeline

`run.py` integrates all stages:

```bash
python run.py
```

It executes the entire workflow:  
1. Load expression and molecule data  
2. Pretrain SMILES encoder (or load existing checkpoint)  
3. Build Δ pairs from Pre/Post DMSO matching  
4. Train the diffusion model  
5. Predict post-treatment gene expression for new molecules  

---

## Notes

- The modular scripts (`dataset.py`, `model.py`, `pretrain.py`, `train.py`, `predict.py`) are independent to allow debugging or standalone usage.  
- `run.py` demonstrates the complete workflow for convenience and reproducibility.  
- Hyperparameters for diffusion, training, and encoder pretraining are set in the scripts but can be adjusted for specific datasets.  

---

## Reference

This framework implements the approach described in the abstract:

> Predicting drug-induced transcriptional responses from chemical structure remains a central challenge in phenotype-driven drug discovery, as cellular responses arise from complex and context-dependent regulatory mechanisms. Existing approaches often rely on predefined targets or direct regression, limiting their ability to capture global gene-level perturbations and uncertainty. We propose MolPerturbDiff, a structure-conditioned diffusion framework that models drug responses as stochastic gene expression perturbations. Given a pre-treatment gene expression profile, MolPerturbDiff predicts gene-wise expression changes conditioned on molecular structure encoded from SMILES representations. A gene-wise structural gating mechanism explicitly links chemical embeddings to the direction and relative strength of transcriptional regulation, while a conditional diffusion process captures uncertainty and heterogeneity in cellular responses. The molecular encoder is pre-trained on an extended chemical space and frozen during diffusion training. Trained on large-scale transcriptional perturbation data across diverse cell types, MolPerturbDiff demonstrates robust generalization to previously unseen small molecules, accurately modeling relative gene expression changes while avoiding extreme predictions. The framework effectively bridges chemical structure and transcriptional phenotype without relying on predefined molecular targets. MolPerturbDiff provides a scalable and interpretable approach for phenotype-driven virtual drug screening by simulating structure-dependent cellular transcriptional responses. This work highlights the potential of diffusion-based generative models as a foundation for future virtual cell perturbation systems and structure-aware drug discovery.
