# FedFree — Federated Learning Free-Rider Defense

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isaac-sun/fedfree/blob/main/notebook.ipynb)

**Per-Class Shapley Value Detection for Free-Rider Attacks in Federated Learning**

A PyTorch-based FL framework that defends against free-rider attacks using two-phase per-class Shapley value analysis, built on DistilBERT + LoRA (PEFT) and evaluated on Yahoo Answers.

---

## Overview

In federated learning, free-rider clients submit meaningless updates while benefiting from the global model. Traditional detection fails against **disguised free-riders** who mimic honest participation patterns.

FedFree introduces a **two-phase defense** based on per-class Shapley values:

| Phase | Method | What It Catches |
|-------|--------|-----------------|
| **Phase 1** — Positive-Sum Filter | Σ max(SV<sub>i,c</sub>, 0) ≈ 0 → flag | Trivial free-riders (noise/zero updates) |
| **Phase 2** — Variance Fingerprinting | Var<sub>c</sub>(SV<sub>i,c</sub>) → low vs high | Disguised free-riders (flat contribution profiles) |

**Key insight**: Honest clients contribute **unevenly** across classes (high variance), while free-riders have **flat** profiles (low variance).

---

## Architecture

```
DistilBERT-base-uncased (66M, frozen)
  ├── LoRA Adapters (r=8, α=16) via PEFT
  │     └── Target modules: q_lin, k_lin, v_lin, out_lin
  └── Classifier: Dropout(0.3) → Linear(768, 10)
```

- **Trainable parameters**: ~0.3M (0.5% of total)
- **FL setup**: 10 clients, IID partition, 30 rounds, 2 local epochs
- **Attackers**: 40% (4/10) — noise or disguise strategies
- **Shapley**: 10 Monte Carlo permutations per round

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py          # Run baseline + defense experiments
python plot_defense.py  # Generate IEEE-style visualizations
```

**Google Colab**: [Click the badge above](https://colab.research.google.com/github/isaac-sun/fedfree/blob/main/notebook.ipynb) or open `notebook.ipynb`.

---

## Project Structure

```
fedfree/
├── config.py              Hyperparameters (dataclass Config)
├── main.py                Training script (baseline + defense)
├── plot_defense.py        5 IEEE-style publication charts
├── notebook.ipynb         Google Colab notebook
├── data/
│   └── yahoo_answers.py   Yahoo Answers 10-class loader + tokenization
├── models/
│   └── lora_model.py      DistilBERT + PEFT LoRA classifier
├── fl/
│   ├── client.py          Client-side local training (AdamW)
│   ├── server.py          Server-side coordination & evaluation
│   └── fedavg.py          FedAvg aggregation with momentum
├── attacks/
│   └── __init__.py        noise_attack / disguise_attack
├── defense/
│   ├── shapley.py         Monte Carlo per-class Shapley estimator
│   └── controller.py      Two-phase DefenseController
└── utils/
    ├── metrics.py          Macro-F1 evaluation
    ├── seed.py             Reproducibility
    └── export.py           CSV export for visualization
```

---

## Configuration

All hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clients` | 10 | Total FL clients |
| `num_rounds` | 30 | Communication rounds |
| `local_epochs` | 2 | Local training epochs per round |
| `local_lr` | 5e-4 | Client learning rate |
| `server_lr` | 0.7 | FedAvg aggregation step |
| `participation_ratio` | 0.8 | Clients selected per round |
| `malicious_ratio` | 0.4 | Attacker proportion |
| `lora_r` | 8 | LoRA rank |
| `num_mc_samples` | 10 | Monte Carlo permutations for Shapley |
| `defense_pos_sum_threshold` | 0.01 | Phase 1 detection threshold |
| `defense_var_threshold` | 0.001 | Phase 2 detection threshold |

---

## Visualization Gallery

`plot_defense.py` generates five publication-quality charts (300 DPI PDF + PNG):

| Figure | Type | Content |
|--------|------|---------|
| **fig1** | Scatter | Defense separation: positive_sum × variance |
| **fig2** | Bar Chart | Variance fingerprint: honest vs attacker |
| **fig3** | Line Plot | Convergence: baseline vs defense Macro-F1 |
| **fig4** | Stacked Bar | Per-round defense phase classification |
| **fig5** | Contour | Decision boundary with density estimation |

All charts use IEEE-compliant styling (Times New Roman, serif fonts, proper legend placement).

---

## Outputs

| File | Description |
|------|-------------|
| `results/defense_history.csv` | Per-round per-client SV metrics |
| `results/plots/fig*.pdf` | Publication-quality figures |
| `results/plots/fig*.png` | Raster versions |

---

## Dependencies

- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- PEFT ≥ 0.5
- Datasets ≥ 2.12
- scikit-learn, NumPy, Pandas
- Matplotlib, Seaborn

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fedfree2024,
  author = {FedFree Contributors},
  title = {FedFree: Federated Learning Free-Rider Defense via Per-Class Shapley},
  year = {2024},
  url = {https://github.com/isaac-sun/fedfree}
}
```

## License

MIT
