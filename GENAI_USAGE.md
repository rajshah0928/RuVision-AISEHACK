# GenAI Tool Usage Disclosure

## Tool Used

**Claude (Sonnet) by Anthropic** — accessed via claude.ai

---

## How It Was Used

Claude was used as a coding and experiment design assistant throughout Phase 2. All model decisions were validated empirically on the leaderboard before being accepted. No model weights, training data, or external code were sourced through Claude — it was used purely for code generation, analysis, and documentation.

### Specific uses:

**1. Codebase generation**  
Claude generated the training pipeline notebooks (v6 → v7) based on our architectural and loss function specifications. We described the desired changes (e.g., "remove CBAM, reduce to 1 LSTM layer, 2 UNet levels") and Claude implemented them as complete, runnable notebooks. All hyperparameters and design choices were made by the team.

**2. Experiment design**  
We described the problem (score 0.8403, top is 0.8968, need to improve) and Claude helped structure 4 parallel experiment tracks (architecture, loss function, clustering, loss weights) with specific variants per track (Exp1–Exp4). The hypothesis for each experiment was our own; Claude helped translate hypotheses into code.

**3. Loss function derivation**  
We provided the competition metric formula and the loss function design document. Claude helped implement the explicit α·GlobalSMAPE + β·EpSMAPE + γ·(1-EpCorr) formulation and verified it matched the competition's evaluation metric structure.

**4. Results analysis**  
After each experiment run, we provided leaderboard scores to Claude, which helped identify the winning direction (higher β, simpler arch, K=16) and suggested further experiments in those directions (Exp5, Exp6).

**5. Submission documentation**  
Claude generated this README, the GENAI_USAGE.md, the inference notebook, and the final presentation based on our experiment results and descriptions of the work.

---

## What Claude Did NOT Do

- Claude did not choose hyperparameters — all α, β, γ, K, dim, base_ch values were our decisions, validated on the leaderboard
- Claude did not access any external data, APIs, or third-party code
- Claude did not run any experiments — all training was done by us on Kaggle GPU notebooks
- Claude did not interpret the competition data or decide which features to use — those decisions were carried over from Phase 1

---

## Prompt Workflow (high level)

```
1. Provided v6 notebook → asked Claude to analyse and suggest experiment directions
2. Received 4 experiment notebook designs → ran on Kaggle, collected scores
3. Provided scores back → Claude identified winning directions
4. Asked Claude to build combined v7 notebook (Exp1-B + Exp3-B + Exp4-D)
5. Ran v7 → score 0.8636 → asked Claude to build regional ensemble variant
6. Ran regional → score 0.8552 → decided to lock in v7 as final
7. Asked Claude to generate submission documentation (README, presentation, inference notebook)
```

---

## Team Sign-off

We confirm that all modelling decisions, experimental choices, and results reported are our own work. Claude was used as a productivity tool for code generation and documentation, equivalent to using GitHub Copilot or similar assistants.

**Raj Shah, Saptarshi Misra, Hiteshri Shastri — Team RuVision**  
ANRF AISEHack 2026, Phase 2, IIIT Hyderabad
