# Large Data Artifacts

This document lists all data artifacts **excluded from the repository** (size > 25 MB or regenerable), their origin, and how to regenerate or re-download them.

---

## Excluded from Repository (Size Policy)

| Artifact | Size | Location (UNIBO Archive) | In Repo? | Regeneration |
|----------|------|--------------------------|----------|--------------|
| Cleaned AIS Parquet (9 weekly files) | 1.3 GB | `archivio_non_git/big_binary/data/processed/` | ❌ | Run `notebooks/static_preprocessing.ipynb` on raw CSVs |
| Sequence cache (`cache_sequenze/`) | 11 GB | `archivio_non_git/big_binary/cache_sequenze/` | ❌ | Automatic on first `run_experiment.py` run |
| Daily predictions (`previsioni_classificatore/`) | 3.5 GB | `archivio_non_git/big_binary/previsioni_classificatore/` | ❌ | Automatic during experiment runs |
| Per-day checkpoints (`checkpoints_v2/`) | 193 MB | `archivio_non_git/big_binary/checkpoints_v2/` | ❌ | Automatic during experiment runs |
| All model weights (`models_v2/`) | 6 MB | `archivio_non_git/big_binary/models_v2/` | ❌ (final weights in `final_weights/`) | Automatic at end of runs |
| BACKUP_COMPLETO (full old backup) | 407 MB | `archivio_non_git/backup_completo/` | ❌ | N/A (historical) |
| Reference papers (PDFs) | 76 MB | `documenti/Paper/` | ❌ | Download from publishers/DOI |
| Archives (.tar.gz, .zip) | ~2 GB | `archivio_non_git/archives/` | ❌ | N/A (duplicates) |
| Old architectures | 1.3 GB | `archivio_non_git/old_architectures/` | ❌ | N/A (historical) |

---

## In Repository (Small, Essential)

| Artifact | Size | Location | Purpose |
|----------|------|----------|---------|
| Final run JSON/CSV | ~13 MB | `results/` | All thesis numbers |
| Final figures (PNG) | ~22 MB | `figure/` | Thesis figures |
| Final-day weights | ~1.6 MB | `final_weights/` | 27 files (<1 MB each) |
| Training logs | ~0.5 MB | `logs/run_logs_v2/` | Hyperparameter search history |
| Code & notebooks | <1 MB | `run_experiment.py`, `functions/`, `notebooks/` | Reproducibility |

---

## Raw AIS Data (Source)

**Source:** MarineCadastre.gov — Vessel Traffic Data (AIS)
- **Agencies:** NOAA (National Oceanic and Atmospheric Administration) + BOEM (Bureau of Ocean Energy Management)
- **Collector:** U.S. Coast Guard Nationwide AIS Receiver Network
- **Coverage:** U.S. Territorial Waters + EEZ
- **License:** CC0 1.0 Universal (public domain)
- **Download URL:** `https://marinecadastre.gov/ais/`
- **Period used:** 2025-01-01 to 2025-03-04 (63 consecutive days)
- **Format:** Daily CSV files (one per day, ~100 MB each compressed)

### Download Instructions
```bash
# Option 1: Web interface
# Visit https://marinecadastre.gov/ais/ → Data Access → Download by Year/Month

# Option 2: Programmatic (if available)
# Check MarineCadastre API documentation for bulk download

# Expected files after download:
raw_2025_01_01.csv
raw_2025_01_02.csv
...
raw_2025_03_04.csv
```

---

## Static Preprocessing (Regenerates Cleaned Parquet)

**Script:** `notebooks/static_preprocessing.ipynb` (reference implementation)

### Input
- Raw daily CSVs from MarineCadastre (63 files)
- Place in `data/raw/` or update notebook paths

### Processing Steps (matches thesis §3.2)
1. Load all 63 daily CSVs
2. Deduplicate by (MMSI, timestamp, lat, lon, SOG, COG, Heading)
3. Drop rows with missing essential fields
4. Filter AIS sentinel values:
   - SOG = 102.3 knots
   - COG = 360°
   - Heading = 511°
   - Invalid lat/lon
   - MMSI ≥ 990000000 (stations/aids-to-navigation)
5. Remove SOG < 1 knot (stationary)
6. Keep MMSI with ≥30 valid messages total
7. COG → sin/cos, Heading → sin/cos
8. Save as 9 weekly Parquet files:
   ```
   data/processed/
   ├── ais_settimana_1.parquet   # Days 0–6
   ├── ais_settimana_2.parquet   # Days 7–13
   ├── ais_settimana_3.parquet   # Days 14–20
   ├── ais_settimana_4.parquet   # Days 21–27
   ├── ais_settimana_5.parquet   # Days 28–34
   ├── ais_settimana_6.parquet   # Days 35–41
   ├── ais_settimana_7.parquet   # Days 42–48
   ├── ais_settimana_8.parquet   # Days 49–55
   └── ais_settimana_9.parquet   # Days 56–62
   ```

### Expected Output
- 68,145,342 cleaned messages
- 14,675 distinct MMSI
- No missing values
- All numeric columns: latitude, longitude, SOG, COG_sin, COG_cos, heading_sin, heading_cos

---

## Sequence Cache (Auto-generated)

**Location:** `cache_sequenze/` (created by `ContinualPreprocessor.create_sequences()`)

### Files
```
cache_sequenze/
├── raw_0_sl30_gt30.npz      # Day 0 sequences
├── raw_1_sl30_gt30.npz      # Day 1
...
└── raw_62_sl30_gt30.npz     # Day 62
```

### Each .npz contains
- `X`: (N, 30, 7) — 30-step history windows
- `y`: (N, 3) — next-step SOG, COG_sin, COG_cos (normalized)
- `last_pos`: (N, 2) — last known lat/lon
- `dest`: (N, 2) — ground truth next lat/lon
- `gaps`: (N,) — gap indicator

### Regeneration
- Delete `cache_sequenze/` and re-run `run_experiment.py`
- First run will rebuild all 63 files (~30 min)

---

## Daily Predictions (Auto-generated)

**Location:** `previsioni_classificatore/`

### Files
```
previsioni_classificatore/
├── {run_id}_{method}_giorno00.parquet
├── {run_id}_{method}_giorno01.parquet
...
└── {run_id}_{method}_giorno62.parquet
```

### Each Parquet contains (per sequence evaluated)
| Column | Type | Description |
|--------|------|-------------|
| checkpoint_day | int32 | Model checkpoint day |
| eval_day | int32 | Test set day |
| logit | float32 | Classifier raw output |
| etichetta | int8 | True label (0/1) |
| attacco | string | "clean" or attack type |
| magnitudo | float32 | Attack magnitude |

### Regeneration
- Delete `previsioni_classificatore/` and re-run experiment
- Written daily during evaluation phase

---

## Checkpoints (Auto-generated)

**Location:** `checkpoints_v2/`

### Files (per checkpoint day)
```
checkpoints_v2/
├── {run_id}_{method}_REG_day{d}.pth      # Regression weights
├── {run_id}_{method}_REG_day{d}_prep.pkl # Preprocessor state (joblib)
└── {run_id}_{method}_CLF_day{d}.pth      # Classifier weights
```

### Regeneration
- Set `--checkpoint_every 1 --save_checkpoints` in `run_experiment.py`
- Re-run experiment

---

## Final Weights (In Repository)

**Location:** `final_weights/`

### Files (27 files, <1 MB each)
```
final_weights/
├── cbrs_multiclasse_b20000_BR_CLF.pth      # Final multiclass BR classifier
├── cbrs_multiclasse_b20000_BR_REG.pth      # Final multiclass BR regression
├── cbrs_multiclasse_b20000_AGEM_CLF.pth    # Final multiclass A-GEM classifier
├── cbrs_multiclasse_b20000_AGEM_REG.pth    # Final multiclass A-GEM regression
├── ewc_EWC_CLF.pth              # EWC reference classifier
├── ewc_EWC_REG.pth              # EWC reference regression
├── lower_bound_NESSUNA_CLF.pth              # Lower bound classifier
├── lower_bound_NESSUNA_REG.pth              # Lower bound regression
├── cbrs_binario_b20000_BR_CLF.pth        # Binary BR 20k classifier
├── cbrs_binario_b20000_BR_REG.pth        # Binary BR 20k regression
├── cbrs_binario_b500_BR_CLF.pth                 # Binary CBRS classifier
├── cbrs_binario_b500_BR_REG.pth                 # Binary CBRS regression
├── cbrs_binario_b500_AGEM_CLF.pth               # Binary CBRS A-GEM classifier
├── cbrs_binario_b500_AGEM_REG.pth               # Binary CBRS A-GEM regression
├── reservoir_puro_BR_CLF.pth                # Reservoir Puro BR classifier
├── reservoir_puro_BR_REG.pth                # Reservoir Puro BR regression
├── reservoir_puro_EWC_CLF.pth               # Reservoir Puro EWC classifier
├── reservoir_puro_EWC_REG.pth               # Reservoir Puro EWC regression
├── reservoir_puro_AGEM_CLF.pth              # Reservoir Puro A-GEM classifier
├── reservoir_puro_AGEM_REG.pth              # Reservoir Puro A-GEM regression
├── cbrs_multiclasse_b10000_BR_CLF.pth            # Multiclass 10k BR classifier
├── cbrs_multiclasse_b10000_BR_REG.pth            # Multiclass 10k BR regression
├── cbrs_multiclasse_b10000_AGEM_CLF.pth          # Multiclass 10k A-GEM classifier
├── cbrs_multiclasse_b10000_AGEM_REG.pth          # Multiclass 10k A-GEM regression
└── upper_bound_CLF.pth                   # Upper bound classifier
```

---

## Results & Figures (In Repository)

### Results (`results/`)
| File | Size | Description |
|------|------|-------------|
| `*_classifier_metrics.json` | 400–440 KB each | Per-day retention entries (all metrics) |
| `*_classifier_metrics.log` | ~16 KB each | Training console output |
| `*.json` (aggregated) | ~600 B each | CLF_F1_finale, BWT, FM, FWT + config |
| `*.csv` | ~2 KB each | Attack calendar, test composition |
| `analysis_*/recall_per_attacco_giorno.csv` | ~20 KB each | Per-attack per-day recall for figures |
| `analysis_*/forgetting_per_attacco.csv` | ~7 KB each | Per-attack forgetting |
| `analysis_*/specificita_giorno.csv` | ~2 KB each | Per-day specificity |

### Figures (`figure/`)
```
figure/
├── 01_reservoir_puro/
│   ├── 01_per_attacco_e_metrica/   # 20 PNGs (5 attacks × 4 metrics)
│   ├── 04a_F1_nel_tempo_con_calendario.png
│   ├── 04b_recall_nel_tempo_con_calendario.png
│   ├── 04c_precision_nel_tempo_con_calendario.png
│   ├── 04d_accuracy_nel_tempo_con_calendario.png
│   ├── 05_BWT_nel_tempo_con_calendario.png
│   ├── 05_FM_nel_tempo_con_calendario.png
│   └── 05_FWT_nel_tempo_con_calendario.png
├── 02_cbrs_binario_buffer500/      # Same structure
├── 03_cbrs_binario_buffer20000/    # Same (no A-GEM)
└── 04_cbrs_6vie_buffer20000/       # Same (final config)
```
Total: ~110 PNGs, ~22 MB

---

## Reproducibility Checklist

- [ ] Download raw AIS CSVs from `https://marinecadastre.gov/ais/` (63 days, 2025-01-01 to 2025-03-04)
- [ ] Run `notebooks/static_preprocessing.ipynb` → produces `data/processed/ais_settimana_1-9.parquet`
- [ ] Symlink repo to `~/VM` or update `grafici.py` line 41
- [ ] Run key experiments (see `README.md` Reproduction section)
- [ ] Run `python grafici.py` → regenerates `figure/`
- [ ] Verify `results/cbrs_multiclasse_b10000_20000_BR.json` → `CLF_F1_finale` ≈ 0.901