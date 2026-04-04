# RuVision — AISEHack Phase 2: India in the Haze

**Team:** RuVision
**Members:** Raj Shah · Saptarshi Misra · Hiteshri Shastri
**Track:** Pollution Forecasting (Contributed by IIT Delhi)
**Competition:** AISEHack Phase 2 — India in the Haze (PM2.5 Forecasting)
**Current Score:** 0.8403 | **Rank:** 24

---

## Problem Statement

Forecast PM2.5 concentrations across India on a 140×124 grid (25 km resolution) **16 hours ahead**, using only a **10-hour lookback window**. Inputs include meteorological variables, emission inventories, and historical PM2.5 from WRF-Chem 2016 simulations.

The evaluation metric is a weighted combination of:
- **NormGlobalSMAPE** — overall prediction accuracy
- **NormEpisodeSMAPE** — accuracy specifically on pollution spike events
- **NormEpisodeCorr** — spatial pattern correlation during episodes

---

## Approach

### Architecture (v6)

```
Input (B × 10 × 20 × 140 × 124)
        ↓
ConvLSTM (2-layer, dim=64)       ← temporal encoding
        ↓
UNet Encoder (4 levels, 32→512ch)
  + CBAM attention on all skips  ← spatial attention
  + 2× ResBlocks at bottleneck
        ↓
UNet Decoder
        ↓
Persistence Residual Head        ← output = last PM2.5 + Δ
        ↓
Output (B × 16 × 140 × 124)
```

**20 input channels:** cpm25 + 8 meteorological variables (pblh, u10, v10, rain, t2, q2, psfc, swdown) + 7 emission variables (SO2, NMVOC_e, PM25, NOx, NH3, NMVOC_finn, bio) + sin/cos hour encoding + episode-frequency prior + cluster ID

**Test-time augmentation (TTA):** original + horizontal flip, averaged at inference.

### Novel Contributions

1. **Cluster-aware spatial loss weighting** — KMeans (K=12) on seasonal PM2.5 + lat/lon identifies 12 pollution regimes. Each cluster gets a data-driven loss weight [0.5, 3.0] based on mean PM2.5, episode frequency, and variance. The IGP belt receives proportionally higher gradient signal.

2. **Competition-exact training loss** — Smooth SMAPE with pseudo-Huber approximation (δ=0.1) using the exact competition denominator `0.5·(|y|+|ŷ|)`, plus Pearson correlation loss on episodic grid points:
   ```
   L = α·GlobalSMAPE(spatial_w) + β·EpisodeSMAPE + γ·(1 − EpisodeCorr)
   ```
   Episode weight ramps 1.0 → 4.0 over 28 epochs (curriculum).

3. **Episode-frequency prior channel** — STL-decomposed residual episode maps fed as an explicit input channel, letting the model learn *where* episodes are likely to recur.

---

## Results

| Version | Kaggle Score | Notes |
|---------|-------------|-------|
| Persistence baseline | ~0.62 | Predict t+1 = t for all 16 steps |
| v4 (binary IGP mask) | ~0.78 | Uniform IGP up-weighting |
| v5 (wrong SMAPE denom) | ~0.81 | LR instability from 2× denominator error |
| **v6 (current best)** | **0.8403** | Cluster weights + exact loss + CBAM |
| v7 (in progress) | TBD | Best of 4 experiments combined |

---

## Repository Structure

```
├── training_pipeline_v6.ipynb     # Best submitted pipeline (score 0.8403)
├── exp1_simplified_arch.ipynb     # Experiment: smaller architecture variants
├── exp2_loss_function.ipynb       # Experiment: alternative loss functions
├── exp3_clusters_dbscan.ipynb     # Experiment: clustering strategy variants
├── exp4_loss_weights.ipynb        # Experiment: loss component weight tuning
├── requirements.txt               # Python dependencies
├── LICENSE                        # ANRF Open License
└── README.md
```

---

## How to Run

All notebooks are designed to run on **Kaggle** with GPU (T4 or P100).

1. Add the competition dataset `aisehack-theme-2` to your Kaggle notebook
2. Open `training_pipeline_v6.ipynb`
3. Run all cells — training takes ~45 min on T4
4. Final cell saves `preds.npy` (shape: 218×140×124×16) for submission

For experiments (Exp1–4), set the variant variable at the top of each notebook before running:
- `ARCH_VARIANT = 'A'` or `'B'` for Exp1
- `LOSS_VARIANT = 'A'/'B'/'C'/'D'` for Exp2
- `CLUSTER_VARIANT = 'A'/'B'/'C'/'D'/'E'` for Exp3
- `LOSS_WEIGHTS_VARIANT = 'A'/'B'/'C'/'D'/'E'` for Exp4

**Kaggle Notebooks:** [Add link here]

---

## Dependencies

```
torch>=2.0
numpy
scikit-learn
scipy
statsmodels
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Model Checkpoints

Best model checkpoint (v6) is available on:
- **Kaggle Hub:** [Add link here]
- **HuggingFace:** [Add link here]

Licensed under ANRF Open License (see `LICENSE`).

---

## What Did Not Work

| Approach | Why It Failed |
|---|---|
| Binary IGP mask for spatial weighting | Too coarse — no geographic nuance, inconsistent episode metrics |
| Standard SMAPE denominator `\|y\|+\|ŷ\|+ε` | 2× larger than competition definition → LR instability |
| Pure 2D ConvNet (no temporal encoder) | Missed episode onset dynamics, score ~0.71 |
| Persistence-only baseline | Strong autocorrelation but insufficient for episode scores |

---

## AI Tools Disclosure

Claude (Anthropic) was used for architecture design, loss function derivation, and notebook code generation. Prompts and chain-of-thought are documented and available on request, as per AISEHack guidelines.

---

## Honor Code

We, the team **RuVision** — Raj Shah, Saptarshi Misra, Hiteshri Shastri — have made our submissions wholly based on our efforts and have not taken help from third parties or members not part of the team.

---

## License

This project is licensed under the **ANRF Open License**.
See [`LICENSE`](./LICENSE) for full terms.
Download: https://anrfonline.in/ANRF/AbstractFilePath?FileType=E&FileName=OL_AISE.pdf&PathKey=DOCUMENT_TEMPLATE
