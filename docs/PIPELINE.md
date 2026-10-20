# Pipeline & Data Flow Documentation

This document describes the complete data flow from raw AIS messages to final CL metrics, matching the implementation in `run_experiment.py` and the `functions/` package.

---

## 1. Static Preprocessing (Run Once)

**Script:** `notebooks/static_preprocessing.ipynb` (reference)  
**Output:** `data/processed/ais_settimana_1.parquet` … `ais_settimana_9.parquet`

### Steps:
1. **Load raw daily CSVs** from MarineCadastre (63 files, ~100M rows)
2. **Deduplicate** by (MMSI, timestamp, lat, lon, SOG, COG, Heading)
3. **Drop rows** with missing essential fields (MMSI, timestamp, lat, lon, SOG, COG)
4. **Filter AIS sentinels:**
   - SOG = 102.3 knots → sensor unavailable
   - COG = 360° → sensor unavailable
   - Heading = 511° → sensor unavailable
   - Lat/Lon outside valid geographic bounds
   - MMSI not in vessel range (990000000–999999999 are aids-to-navigation)
5. **Remove stationary:** SOG < 1 knot
6. **Minimum track length:** Keep only MMSI with ≥30 valid messages across 63 days
7. **Trigonometric encoding:** COG → (sin, cos), Heading → (sin, cos)
8. **Save** as 9 weekly Parquet files (columnar, ~140 MB each)

**Result:** 68,145,342 cleaned messages, 14,675 vessels.

---

## 2. Daily Sequence Construction (Per Model, Per Day)

**Module:** `functions/processing.py` → `ContinualPreprocessor`

### 2.1 Regression Preprocessor (Shared)
For each day:
1. Load day's Parquet via `AISStreamLoader`
2. `partial_fit_scalers()` — update `StandardScaler` on features, `MinMaxScaler` on targets
3. `transform_X()` → scaled features
4. `create_sequences(seq_len=30)` → sliding windows
   - Segment at gaps > 30 min
   - Resample to 1-min grid (linear interp for lat/lon/SOG; slerp for sin/cos)
   - Windows: X = 30 steps × 7 features, y = next-step (SOG, COG_sin, COG_cos)
   - Cache to `cache_sequenze/raw_{day}_sl30_gt30.npz`
5. `create_sequences_raw()` → same but unscaled (for classifier dataset)

### 2.2 Train/Test Split (80/20, Deterministic)
```python
rng = np.random.default_rng(1000 + day_idx)
idx = rng.permutation(n_sequences)
n_train = int(0.8 * n)
train_idx, test_idx = idx[:n_train], idx[n_train:]
```
Same split used for regression AND classifier (independently).

---

## 3. Regression Training (Daily)

**Module:** `functions/model.py` → `TrajectoryLSTM`

### Architecture
- Input: 7 features (lat, lon, SOG, COG_sin, COG_cos, heading_sin, heading_cos)
- LSTM: hidden=32, num_layers=2, batch_first=True
- Output head: Linear(32 → 3) → SOG, COG_sin, COG_cos
- CLeAN normalization: `use_clean=True` → internal `CleanNormalizer` on inputs + targets (MinMax only for targets, invertible)

### Loss & Optimizer
- Loss: `MSELoss()` on normalized targets
- Optimizer: Adam(lr=0.001)
- Batch: 1024, Epochs: 5/day

### Continual Learning (per strategy)
| Strategy | Mechanism |
|----------|-----------|
| **BR** | `ReservoirBuffer(500)` → replay at each batch |
| **EWC** | `EWCHelper(model, λ=50)` → `consolidate()` daily, `ewc_loss()` added |
| **AGEM** | `ReservoirBuffer(500)` → `project_agem_gradient()` per batch |
| **NESSUNA** | Standard SGD |

### Post-Training
- If BR/AGEM: `buffer_reg.add(X_train, y_train_raw)`
- If EWC: `ewc_reg.consolidate(train_dataset)` — accumulates S₀, S₁

---

## 4. Classifier Dataset Construction (Daily)

**Module:** `functions/classifier_dataset.py` → `build_daily_labeled_dataset()`

### Inputs
- Raw sequences from regression preprocessor (`X_raw_2`, `y_raw_phys`, last_pos, dest, gaps)
- `piano_train` / `piano_test` from `attack_calendar.py`

### Attack Injection (Training Split)
For each sequence in train split:
- With prob 0.1 per active attack type: inject attack
- Attack magnitude sampled per-type range
- Label = 1 (attacked), attack_type = specific type, magnitude = sampled value
- Clean sequences: label = 0, attack_type = "clean", magnitude = 0

### Test Split
- Fixed size: 20,000 sequences/day
- Cumulative calendar: all attacks seen so far + clean
- Deterministic seed: 2000 + day_idx

### Outputs
```python
clf_train = {
    "X": [...],                    # 30×7 history
    "labels": [0/1],               # binary
    "attack_type": ["clean", "teleport", ...],
    "magnitude": [0.0, 12.3, ...],
    "declared_lat/lon/SOG/COG_sin/COG_cos": [...],
    "last_pos": [...], "gaps": [...]
}
clf_test = {same structure, 20000 entries}
```

---

## 5. Classifier Training (Daily)

**Module:** `functions/classifier_model.py` → `AttackClassifierLSTM`

### Architecture
- **Temporal encoder:** LSTM(hidden=32, num_layers=2) on 30×7 history
- **Static encoder:** CleanNormalizer on 8 static features (5 declared + 3 errors)
- **Fusion:** Concatenate final LSTM hidden + static encoded → MLP(2 layers) → 1 logit
- CLeAN on both branches (independent from regression)

### Error Features (from Regression)
**Module:** `functions/regression_features.py` → `compute_regression_error_features()`
1. Predict next-step SOG, COG_sin, COG_cos
2. Dead reckoning → predicted lat/lon
3. Errors:
   - `pos_err` = haversine(predicted, declared) [meters]
   - `speed_err` = |SOG_pred - SOG_decl| [knots]
   - `dir_err` = angle(COG_pred, COG_decl) [degrees]

### Loss & Optimizer
- Loss: `BCEWithLogitsLoss()` (numerically stable logit → prob)
- Optimizer: Adam(lr=0.001)
- Batch: 1024, Epochs: 20/day

### Continual Learning (independent instances from regression)
| Strategy | Buffer | Mechanism |
|----------|--------|-----------|
| **BR** | `ClassBalancingReservoirBuffer(20000, label_index=9)` | 6-class balanced replay |
| **EWC** | None | `EWCHelper(λ=50)` |
| **AGEM** | `ClassBalancingReservoirBuffer(20000, label_index=9)` | Gradient projection |
| **NESSUNA** | None | Standard SGD |

### Buffer Storage (BR/AGEM)
```python
combo_train = torch.cat([extra_train_t (8), y_train_t (1), attack_id (1)], dim=1)
# attack_id: 0=clean, 1=teleport, 2=speed_spoofing, 3=ghost_ship, 4=silent_drift, 5=course_spoofing
buffer_clf.add(X_train_t, combo_train)
```

---

## 6. Daily Evaluation (All Prior Days)

**Module:** `functions/evaluation.py` → `evaluate_day()`

### Regression Metrics (per eval_day)
- MSE (normalized)
- ADE (meters) via Dead Reckoning:
  ```
  For each step: lat += SOG * cos(COG) * dt, lon += SOG * sin(COG) * dt
  Compare final position to ground truth
  ```

### Classifier Metrics (per eval_day)
- Accuracy, Precision, Recall, F1 (binary, threshold=0 on logit)
- Per-attack recall/F1 (via `attack_type` in test set)
- Stored in `classifier_metrics_log` → `*_classifier_metrics.json`

### Retention Matrices (63×63)
- `R_mse[checkpoint_day, eval_day]`
- `R_ade[checkpoint_day, eval_day]`
- `R_f1[checkpoint_day, eval_day]`

---

## 7. CL Metrics Computation

**Module:** `functions/cl_metrics_classifier.py` → `compute_cl_metrics_classifier(R_f1)`

Given final model (row 62) evaluated on all test days (cols 0..62):

- **F1 Finale** = nanmean(R_f1[-1, :])  ← PRIMARY METRIC
- **BWT** = mean(R_f1[-1, j] - R_f1[j, j]) for j < 62
- **FM** = mean(R_f1[-1, j] - min(R_f1[j:, j])) for j < 62
- **FWT** = mean(R_f1[j, j] - baseline) for j ≥ 3, baseline = mean(R_f1[0:3, 0:3] diag)

Same formulas for regression (on MSE, lower=better).

---

## 8. Checkpointing & Artifacts

| Artifact | Location | When |
|----------|----------|------|
| Regression weights | `checkpoints_v2/{run_id}_{method}_REG_day{d}.pth` | Every `checkpoint_every` days |
| Regression preprocessor | `checkpoints_v2/{run_id}_{method}_REG_day{d}_prep.pkl` | Same |
| Classifier weights | `checkpoints_v2/{run_id}_{method}_CLF_day{d}.pth` | Same |
| Final regression | `models_v2/{run_id}_{method}_REG.pth` | End of run |
| Final classifier | `models_v2/{run_id}_{method}_CLF.pth` | End of run |
| Retention matrices | `models_v2/{run_id}_{method}_R_{mse,ade,f1}.npy` | End of run |
| Classifier metrics log | `results/{run_id}_{method}_classifier_metrics.json` | Daily (incremental) |
| Aggregated results | `results/{run_id}_{method}.json` | End of run |
| Raw predictions | `previsioni_classificatore/{run_id}_{method}_giorno{d:02d}.parquet` | Daily |

---

## 9. Figure Generation

**Script:** `grafici.py`  
**Input:** `results/*_classifier_metrics.json`, `upper_bound.json`, `results/analysis_*/recall_per_attacco_giorno.csv`  
**Output:** `figure/{01,02,03,04}_*/`

### Figure Types
1. **Per-attack × metric** (20 per config): Recall, F1, Acc, Prec over days (diagonal)
2. **F1 vs buffer size** (Type 2): Sweep curves
3. **F1 over time with calendar** (Type 4): 3 lines (BR/EWC/AGEM) + lower/upper bounds, attack blocks annotated

---

## 10. Key Implementation Decisions

1. **Two independent models** — No shared weights; regression never sees attack labels
2. **Opzione B** — Classifier uses regression's **updated** weights each day (train + eval)
3. **CLeAN inside models** — Not external; regression targets use MinMax-only (invertible)
4. **6-class CBRS** — Buffer balances clean + 5 attack types separately
5. **EWC object persistence** — Single `EWCHelper` per model, `consolidate()` accumulates S₀/S₁
6. **Fixed test size** — 20,000/day regardless of traffic volume
7. **Cumulative test calendar** — Enables forgetting measurement
8. **Deterministic splits** — Seed = 1000+day (regression), 2000+day (classifier)
9. **Batch evaluation** — `EVAL_BATCH_SIZE=8192` to avoid OOM on 20k sequences
10. **Daily evaluation** — Not just every `eval_every`; needed for retention curves