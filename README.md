# RuVision — AISEHack 2026 Phase 2: India in the Haze (PM2.5)

**Team:** Raj Shah, Saptarshi Misra, Hiteshri Shastri  
**Final Score:** 0.8636 (leaderboard) | Top score: 0.8968  
**Theme:** PM2.5 concentration forecasting over India — 16-hour ahead forecast from 10-hour lookback

---

## Problem

Forecast PM2.5 concentration across a 140×124 spatial grid of India (25 km resolution) 16 hours ahead, using only 10 hours of historical observations. Evaluation uses a weighted combination of NormGlobalSMAPE, NormEpisodeCorr, and NormEpisodeSMAPE.

---

## Model Architecture (v7)

A simplified spatiotemporal encoder–decoder:

```
Input (B, 10, 20, H, W)
    ↓
ConvLSTM  — 1 layer, hidden dim = 48
    ↓
t_proj    — 1×1 Conv → 24 ch
    ↓
UNet Encoder — 2 levels, base = 24 ch (→ 48 ch bottleneck)
    ↓
Bottleneck  — ConvBNGELU + 2× ResBlock (96 ch)
    ↓
UNet Decoder — transposed conv + skip concat (no CBAM)
    ↓
Head        — Conv → 16 ch forecast
    ↓
+ Persistence residual (last observed PM2.5)
Output (B, 16, H, W)
```

**Key design choices:**
- No CBAM attention — reduces overfitting on ~2000 training samples
- 0.6M parameters (vs 5M in v6 baseline)
- Persistence residual head — model learns the delta from last observation
- 20 input channels: cpm25 + 8 met vars + 7 emission vars + sin/cos hour + ep_freq_prior + cluster_id

---

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight_decay=1e-4 |
| LR schedule | OneCycleLR, max_lr=3e-4 |
| Epochs | 28 |
| Batch size | 16 |
| Data | WRF-Chem 2016, 4 months (Apr/Jul/Oct/Dec) |
| Train hours | 600 per month |
| Augmentation | Random E-W flip (p=0.5) |
| Gradient clip | 1.0 |

**Spatial clustering:** KMeans K=16 on seasonal PM2.5 + lat/lon features → per-cluster loss weights in [0.5, 3.0] based on mean PM2.5, episode frequency, and variance.

**Loss function:**
```
L = α · GlobalSMAPE(spatial_w) + β · EpisodeSMAPE + γ · (1 − EpisodeCorr)
α = 1.0,  β = 3.0,  γ = 0.15
```
GlobalSMAPE uses smooth pseudo-Huber SMAPE (δ=0.1) with competition-exact denominator `0.5·(|y|+|ŷ|)`. Episode terms computed only over STL-detected pollution episode points.

**Inference:** TTA with E-W flip average.

---

## Experiment Results

All experiments run from the v6 baseline (score: 0.8403). Each tests one change in isolation.

| Experiment | Change | Score |
|---|---|---|
| v6 baseline | — | 0.8403 |
| Exp1-A | 1-layer LSTM, 2-level UNet, CBAM at bottleneck only | 0.8519 |
| **Exp1-B** | 1-layer LSTM, 2-level UNet, **no CBAM** | **0.8542** |
| Exp2-A | 3-stage curriculum loss (MAE → EpSMAPE → full) | 0.8399 |
| Exp2-B | Fixed-denominator SMAPE | 0.8270 |
| Exp2-C | Huber + EpCorr | 0.8314 |
| Exp2-D | Log-cosh SMAPE | NaN loss |
| Exp3-A | KMeans K=8 | 0.8416 |
| **Exp3-B** | **KMeans K=16** | **0.8449** |
| Exp3-C | DBSCAN clustering | 0.8412 |
| Exp3-D | ep_freq-only weights | 0.8422 |
| Exp3-E | Wider weight range [0.2, 4.2] | 0.8392 |
| Exp4-A | α=1.0, β=2.0, γ=0.30 | 0.8601 |
| Exp4-B | α=1.0, β=1.0, γ=0.60 | 0.8520 |
| Exp4-C | α=0.5, β=2.0, γ=0.50 | 0.8593 |
| **Exp4-D** | **α=1.0, β=3.0, γ=0.15** | **0.8608** |
| Exp4-E | α=0.5, β=1.0, γ=0.50 | 0.8555 |
| Exp5 | Lighter arch (base=16, dim=32) | 0.8632 |
| Exp6-A | β=4.0, γ=0.15 | 0.8628 |
| **v7** | **Exp1-B + Exp3-B + Exp4-D combined** | **0.8636** |
| v7-regional | 6-block regional ensemble with cosine blending | 0.8552 |

**What didn't work:** Alternative loss formulations (Exp2), wider cluster weight range (Exp3-E), further architecture/loss fine-tuning beyond v7 (Exp5/6), regional ensemble.

---

## Repository Structure

```
├── training_pipeline_v6.ipynb        # Baseline (score: 0.8403)
├── training_pipeline_v7.ipynb        # Best submission (score: 0.8636)
├── training_pipeline_v7_regional.ipynb  # Regional ensemble experiment
├── inference.ipynb                   # Clean inference-only notebook
├── exp1_simplified_arch.ipynb        # Architecture ablation (variants A/B)
├── exp2_loss_function.ipynb          # Loss function variants (A/B/C/D)
├── exp3_clusters_dbscan.ipynb        # Clustering variants (A/B/C/D/E)
├── exp4_loss_weights.ipynb           # Loss weight variants (A/B/C/D/E)
├── exp5_lighter_arch.ipynb           # Further arch simplification
├── exp6_loss_weight_finetune.ipynb   # Further loss weight fine-tuning
├── requirements.txt                  # Python dependencies
├── GENAI_USAGE.md                    # GenAI tool disclosure
└── LICENSE                           # ANRF Open License
```

---

## Model Checkpoint

The v7 final checkpoint is available at:

> **[Upload link to be added — Kaggle Hub / HuggingFace]**

To use: download `v7_final.pt` and run `inference.ipynb`.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `numpy`, `scikit-learn`, `statsmodels`, `scipy`

---

## License

This codebase and model checkpoints are released under the **ANRF Open License**.  
See `LICENSE` for full terms.

---

## Citation / Acknowledgements

- ConvLSTM: Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" (NeurIPS 2015)
- CBAM: Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018) — evaluated and found to overfit at this data scale
- WRF-Chem data provided by AISEHack / ANRF
