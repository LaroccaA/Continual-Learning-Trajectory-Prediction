# Adaptive Detection of Cyber-Attacks on AIS Streams for Maritime Security

**Master's Thesis — University of Bologna, Computer Engineering**  
Author: Andrea La Rocca | Supervisor: Michele Colajanni | Co-supervisor: Isabella Marasco | A.Y. 2025/2026

---

## Summary

This repository contains the complete implementation and results for a Continual Learning (CL) framework that detects spoofing attacks on AIS (Automatic Identification System) maritime traffic data. The system comprises two independent LSTM models retrained daily over 63 consecutive days: a **trajectory predictor** (regression of next-step speed and course) and a **binary attack classifier** that consumes the predictor's error channels. The proposed method, **Class-Balancing Reservoir Sampling (CBRS)** with a 6-class buffer (clean + 5 attack types), is compared against Elastic Weight Consolidation (EWC), Averaged Gradient Episodic Memory (A-GEM), a lower bound (sequential training without CL), and an upper bound (joint training).

**Key result:** CBRS with multiclass buffer (20,000) achieves **F1 = 0.901** (BWT = −0.042, FM = 0.057) vs. upper bound 0.958, maintaining per-attack recall ≥ 0.74 across all 63 days.

---

## Problem & Motivation

AIS is the primary collision-avoidance and vessel-tracking system for maritime traffic. It is unencrypted and unauthenticated, making it vulnerable to spoofing attacks (position, speed, course manipulation). Traditional anomaly detection assumes static attack distributions; real attackers appear intermittently. This work frames detection as **Continual Learning** on a non-stationary stream where attacks appear/disappear according to a realistic calendar, measuring not just learning but **retention** during absence.

---

## Framework Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAILY LOOP (×63 days)                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Load day's AIS messages (cleaned, resampled to 1-min,      │
│     split into 30-step windows → sequences)                    │
│                                                                 │
│  2. TRAJECTORY PREDICTOR (LSTM, hidden=32, 2 layers)           │
│     Input: 30-step history (lat, lon, SOG, COG_sin/cos,        │
│            heading_sin/cos)                                    │
│     Output: next-step SOG, COG_sin, COG_cos                    │
│     Loss: MSE | CL: own buffer/EWC/A-GEM instance              │
│                                                                 │
│  3. Compute 3 error channels from predictor:                   │
│     • Position error (haversine distance, meters)              │
│     • Speed error (|SOG_pred − SOG_declared|, knots)           │
│     • Direction error (angle between COG vectors, degrees)     │
│                                                                 │
│  4. ATTACK CLASSIFIER (LSTM, hidden=32, 2 layers + MLP head)   │
│     Inputs concatenated:                                       │
│       • 30-step history (separate LSTM)                        │
│       • 5 declared values (lat, lon, SOG, COG_sin, COG_cos)    │
│       • 3 error channels                                       │
│     Output: binary logit (clean vs. attacked)                  │
│     Loss: BCEWithLogitsLoss | CL: own buffer/EWC/A-GEM         │
│                                                                 │
│  5. CLeAN adaptive normalization inside both models            │
│     (EMA for extreme-value tracking + learnable rescale)       │
│                                                                 │
│  6. Evaluate on 20,000 fixed-size test sequences/day           │
│     (80/20 train/test split, deterministic per day)            │
│     Test calendar is CUMULATIVE: once an attack appears,       │
│     it stays in test until day 63.                             │
└─────────────────────────────────────────────────────────────────┘
```

### Continual Learning Strategies (applied independently to each model)

| Strategy | Mechanism | Buffer |
|----------|-----------|--------|
| **CBRS (proposed)** | Class-Balancing Reservoir Sampling (6 classes: clean + 5 attacks) | 20,000 |
| **EWC** | Quadratic penalty accumulating Fisher (S₀, S₁) over days | None |
| **A-GEM** | Gradient projection against random buffer sample | 20,000 |
| **NESSUNA** (lower bound) | Sequential fine-tuning, no memory/penalty | None |
| **Joint** (upper bound) | Single training on all data (no temporal constraint) | N/A |

---

## Dataset

- **Source:** MarineCadastre.gov (NOAA & BOEM), U.S. Coast Guard AIS receiver network
- **Period:** 63 consecutive days, 2025-01-01 to 2025-03-04
- **Raw messages:** ~68.1M cleaned messages, 14,675 distinct vessels (MMSI)
- **Geographic scope:** Entire U.S. territorial waters / EEZ coverage (no geographic clipping)
- **Vessel types:** All (no filtering by type)
- **Download:** `https://marinecadastre.gov/ais/` (CC0 1.0 Universal)

### Cleaning Pipeline (static, run once)
1. Remove duplicate rows (overlapping receivers)
2. Drop rows with missing essential fields
3. Remove AIS sentinel values: SOG=102.3 kn, COG=360°, Heading=511° (sensor unavailable)
4. Remove stationary vessels (SOG < 1 kn)
5. Keep only vessels with ≥30 valid messages in the 63-day period
6. Convert COG/Heading to sin/cos (resolve 0°/360° discontinuity)
7. Save as 9 weekly Parquet files (`ais_settimana_1.parquet` … `ais_settimana_9.parquet`)

### Sequence Construction (per day, per model)
1. Segment each vessel's track at gaps > 30 min
2. Resample to 1-min grid (linear interpolation for lat/lon/SOG; slerp for sin/cos)
3. Sliding windows: 30 history steps → predict step 31
4. Regression target: SOG + COG_sin + COG_cos (position via dead reckoning)

---

## Synthetic Attacks & Calendar

Five attack types injected at 10% of daily sequences during active blocks:

| Attack | Alteration | Magnitude | Detection Channel |
|--------|------------|-----------|-------------------|
| **Teleport** | Position jump | 550 m – 16 km random direction | Position |
| **Speed Spoofing** | SOG × factor | 1.3× – 3.5× (sometimes inverted) | Speed |
| **Ghost Ship** | Vessel stops, SOG falsified | Mix of true SOG (~0) and pre-stop SOG | Position / Speed |
| **Silent Drift** | Position accumulates | 22 – 166 m/min constant direction | Position |
| **Course Spoofing** | COG rotation | 25° – 150° random sign | Direction |

**Intermittent Calendar (training):**
```
Days 0–6     : speed_spoofing
Days 7–8     : quiet
Days 9–14    : teleport
Days 15–17   : quiet
Days 18–27   : silent_drift (10-day block)
Days 28–29   : quiet
Days 30–33   : speed_spoofing + teleport
Days 34–36   : quiet
Days 37–41   : ghost_ship
Days 42–43   : quiet
Days 44–48   : course_spoofing
Days 49–50   : quiet
Days 51–54   : silent_drift + course_spoofing
Day 55       : quiet
Days 56–57   : speed_spoofing + ghost_ship
Days 58–62   : quiet
```

**Test calendar is cumulative:** each attack enters test at its first training appearance and remains until day 63. By day 63, test set is 50% attacked (5 attacks × 10%).

---

## Experimental Protocol

- **Models:** Two independent LSTMs (hidden=32, 2 layers, seq_len=30)
- **Optimizer:** Adam, lr=0.001, batch=1024
- **Epochs/day:** Regression=5, Classifier=20
- **Test set:** 20,000 sequences/day (fixed, deterministic 80/20 split)
- **CL buffer (CBRS):** 20,000 samples, 6-class balanced
- **EWC λ:** 50 (fixed, not tuned)
- **Seed:** 42 (single run per configuration — see Limitations)
- **Evaluation:** Daily on all prior test sets → retention matrices (63×63)
- **Primary metric:** **F1 finale** = nanmean of LAST ROW of retention matrix (final model on all 63 test days)

---

## Results (from `results/*.json`)

| Configuration | Strategy | F1 Finale | BWT | FM | FWT |
|--------------|----------|-----------|-----|-----|-----|
| Reservoir Puro, buffer 500 | BR | 0.261 | −0.360 | 0.587 | −0.368 |
| Reservoir Puro, buffer 500 | EWC | 0.475 | −0.128 | 0.313 | −0.133 |
| Reservoir Puro, buffer 500 | A-GEM | 0.0003 | −0.530 | 0.788 | −0.591 |
| CBRS Binario, buffer 500 | BR | 0.479 | −0.287 | 0.423 | −0.213 |
| CBRS Binario, buffer 500 | A-GEM | 0.147 | −0.515 | 0.727 | −0.326 |
| Binario, buffer 20,000 | BR | 0.867 | −0.064 | 0.080 | −0.030 |
| **Multiclasse, buffer 20,000 (FINAL)** | **BR** | **0.901** | **−0.042** | **0.057** | **−0.017** |
| Multiclasse, buffer 20,000 | A-GEM | 0.095 | −0.597 | 0.826 | −0.291 |
| Multiclasse, buffer 10,000 | BR | 0.846 | −0.089 | 0.117 | −0.051 |
| Multiclasse, buffer 10,000 | A-GEM | 0.562 | −0.175 | 0.380 | −0.240 |
| EWC (all configs) | EWC | 0.475 | −0.128 | 0.313 | −0.133 |
| Lower bound (NESSUNA) | — | 1.1e−05 | −0.434 | 0.752 | −0.565 |
| Upper bound (Joint) | — | 0.958 | — | — | — |

### Binary Buffer Sweep (BR)
| Buffer | F1 Finale |
|--------|-----------|
| 500 | 0.479 |
| 1,000 | 0.620 |
| 2,000 | 0.727 |
| 3,000 | 0.761 |
| 5,000 | 0.774 |
| 10,000 | 0.824 |
| 20,000 | 0.867 |

### Binary Buffer Sweep (A-GEM)
| Buffer | F1 Finale |
|--------|-----------|
| 500 | 0.147 |
| 1,000 | 0.226 |
| 2,000 | 0.166 |
| 3,000 | 0.082 |

*All values match thesis (capitolo4.tex Table 10) within tolerance 0.001.*

---

## Repository Structure

```
.
├── run_experiment.py          # Main orchestrator (CLI)
├── requirements.txt              # Python dependencies
├── .gitignore
├── LICENSE
├── README.md                     # This file
├── functions/                    # Core package
│   ├── __init__.py
│   ├── loader.py                 # AISStreamLoader (Parquet I/O)
│   ├── processing.py             # ContinualPreprocessor (scaling, sequences)
│   ├── model.py                  # TrajectoryLSTM (regression)
│   ├── classifier_model.py       # AttackClassifierLSTM
│   ├── classifier_dataset.py     # build_daily_labeled_dataset (attacks + calendar)
│   ├── attack_calendar.py        # Training/test calendar construction
│   ├── regression_features.py    # compute_regression_error_features
│   ├── attacks.py                # 5 synthetic attack implementations
│   ├── cl_methods.py             # ReservoirBuffer, ClassBalancingReservoirBuffer, EWCHelper, A-GEM
│   ├── cl_metrics_classifier.py  # BWT/FM/FWT computation
│   ├── evaluation.py             # evaluate_day (MSE + ADE via dead reckoning)
│   ├── clean_normalizer.py       # CLeAN normalization (EMA + learnable)
│   ├── prediction.py             # Inference helpers
│   └── detection_metrics.py      # Binary classification metrics
├── results/                   # Final JSON/CSV results + logs
│   ├── *_classifier_metrics.json # Per-day retention entries
│   ├── *_classifier_metrics.log  # Training logs
│   ├── *.json                    # Aggregated metrics (CLF_F1_finale, etc.)
│   ├── *.csv                     # Attack calendar, test composition
│   ├── analysis_*/                # Per-run attack-day CSV tables
│   └── sweep_buffer_*/           # Buffer sweep sub-results
├── figure/             # Thesis figures (4 configs × 7 metrics)
│   ├── 01_reservoir_puro/
│   ├── 02_cbrs_binario_buffer500/
│   ├── 03_cbrs_binario_buffer20000/
│   └── 04_cbrs_6vie_buffer20000/
├── final_weights/            # Final-day weights (<1 MB each)
│   ├── cbrs_multiclasse_b20000_*   # Final multiclass BR/AGEM
│   ├── ewc_*            # EWC reference
│   ├── lower_bound_*             # Lower bound
│   ├── cbrs_binario_b20000_*     # Binary BR 20k
│   ├── cbrs_binario_b500_*              # Binary CBRS
│   ├── reservoir_puro_*             # Reservoir Puro
│   ├── cbrs_multiclasse_b10000_*         # Multiclass 10k
│   └── upper_bound_CLF.pth       # Upper bound classifier
├── notebooks/
│   └── static_preprocessing.ipynb # Static cleaning reference
├── logs/
│   └── run_logs_v2/              # Hyperparameter search logs
└── docs/
    ├── PIPELINE.md               # Detailed pipeline & data flow
    ├── RESULTS_PROVENANCE.md     # Thesis number → source file mapping
    ├── HARDCODED_PATHS.md        # Absolute/~ paths in code (unchanged)
    └── DATA.md                   # Large data artifacts & regeneration
```

---

## Reproduction

### Environment
```bash
python >= 3.10
torch >= 2.0
numpy, polars, pandas, matplotlib, scipy, joblib, psutil (optional)
```

### Expected Project Layout (Hard-coded Paths)
The scripts expect the project at `~/VM` with data at `~/VM/data/processed/`. Two options:

**Option A: Symlink (recommended)**
```bash
ln -s /path/to/this/repo ~/VM
# Place cleaned Parquet files at ~/VM/data/processed/ais_settimana_1.parquet ... 9
```

**Option B: Modify DATA_PATH in run_experiment.py**
```python
# Line 61: DATA_PATH = os.path.join(HERE, 'data', 'processed')
# Change to your actual data directory
```

### Run Commands (reconstructed from logs)

```bash
# Final multiclass CBRS (BR) — buffer 20,000, 6 classes
python run_experiment.py \
  --run_id cbrs_multiclasse_b20000 --method BR \
  --reg_lr 0.001 --reg_hidden_size 32 --reg_num_layers 2 \
  --reg_buffer_size 500 --reg_ewc_lambda 50 --reg_batch_size 1024 --reg_epochs 5 \
  --clf_lr 0.001 --clf_hidden_size 32 --clf_num_layers 2 \
  --clf_buffer_size 20000 --clf_ewc_lambda 50 --clf_batch_size 1024 --clf_epochs 20 \
  --seq_len 30 --eval_every 5 --checkpoint_every 1 --save_checkpoints

# EWC reference (same for all configs)
python run_experiment.py \
  --run_id ewc --method EWC \
  --reg_lr 0.001 --reg_hidden_size 32 --reg_num_layers 2 \
  --reg_buffer_size 500 --reg_ewc_lambda 50 --reg_batch_size 1024 --reg_epochs 5 \
  --clf_lr 0.001 --clf_hidden_size 32 --clf_num_layers 2 \
  --clf_buffer_size 500 --clf_ewc_lambda 50 --clf_batch_size 1024 --clf_epochs 20 \
  --seq_len 30 --eval_every 5

# Lower bound
python run_experiment.py \
  --run_id lower_bound --method NESSUNA \
  --reg_lr 0.001 --reg_hidden_size 32 --reg_num_layers 2 \
  --reg_buffer_size 500 --reg_ewc_lambda 50 --reg_batch_size 1024 --reg_epochs 5 \
  --clf_lr 0.001 --clf_hidden_size 32 --clf_num_layers 2 \
  --clf_buffer_size 500 --clf_ewc_lambda 50 --clf_batch_size 1024 --clf_epochs 20 \
  --seq_len 30 --eval_every 5

# Upper bound (separate script, joint training)
python upper_bound.py  # in archivio_non_git/superseded_runs/
```

### Generate Figures
```bash
python grafici.py  # Reads results/, writes figure/
```

---

## Limitations

1. **Single seed** — All results from one initialization (seed=42). No variance estimates.
2. **EWC λ not tuned** — Fixed at 50; λ sweep (10, 50, 200, 1000) showed F1 < 0.001 for λ≠50.
3. **Hard-coded paths** — Scripts assume `~/VM/data/processed/` and `~/VM/results/`. See `docs/HARDCODED_PATHS.md`.
4. **Italian comments/logs** — Code comments, docstrings, and log messages are in Italian.
5. **No geographic split** — Entire U.S. EEZ treated as one domain.
6. **Synthetic attacks only** — No real AIS attack labels available.
7. **Classifier is binary** — Detects "attacked vs clean" only; attack-type identification is implicit via buffer balancing.

---

## Citation

```bibtex
@mastersthesis{larocca2026ais,
  title     = {Rilevamento Adattivo di Cyber-Attacchi ai Flussi AIS per la Sicurezza Navale},
  author    = {La Rocca, Andrea},
  school    = {University of Bologna},
  year      = {2026},
  note      = {Supervisor: Michele Colajanni, Co-supervisor: Isabella Marasco}
}
```

Thesis PDF: `tesi_pdf/main.pdf` (in UNIBO archive)  
Repository: [GitHub URL to be added after push]

---

## License

See `LICENSE` (MIT License).

---

## Acknowledgments

- MarineCadastre.gov (NOAA/BOEM) for open AIS data
- Isabella Marasco for CLeAN normalization (arXiv:2603.17548)
- Chrysakis & Moens (ICML 2020) for Class-Balancing Reservoir Sampling
- Michele Colajanni for supervision and experimental design guidance