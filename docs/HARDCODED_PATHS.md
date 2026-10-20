# Hard-coded Paths in Codebase

This document lists every absolute path, `~` expansion, or fixed directory assumption in the code. **No paths have been modified** — they are documented here for reproducibility.

---

## run_experiment_v2.py

| Line | Path | Type | Notes |
|------|------|------|-------|
| 60 | `HERE = os.path.dirname(os.path.abspath(__file__))` | Relative anchor | Script directory |
| 61 | `DATA_PATH = os.path.join(HERE, 'data', 'processed')` | **Relative to script** | Expects `data/processed/ais_settimana_*.parquet` |
| 62 | `RESULTS_DIR = os.path.join(HERE, 'results_v2')` | **Relative to script** | Output directory |
| 63 | `MODELS_DIR = os.path.join(HERE, 'models_v2')` | **Relative to script** | Final weights |
| 64 | `CHECKPOINTS_DIR = os.path.join(HERE, 'checkpoints_v2')` | **Relative to script** | Per-day checkpoints |
| 65 | `PREDICTIONS_DIR = os.path.join(HERE, 'previsioni_classificatore')` | **Relative to script** | Raw predictions (Parquet) |
| 185 | `cache_dir=os.path.join(HERE, 'cache_sequenze')` | **Relative to script** | Sequence cache (11 GB) |

**All paths are relative to the script location** — if you place the repo at `~/VM`, everything resolves correctly.

---

## functions/loader.py

| Line | Path | Type | Notes |
|------|------|------|-------|
| `AISStreamLoader.__init__` | `data_path` parameter | Passed from `run_experiment_v2.py` | Defaults to `DATA_PATH` above |

---

## functions/processing.py

| Line | Path | Type | Notes |
|------|------|------|-------|
| `ContinualPreprocessor.__init__` | `cache_dir` parameter | Passed from `run_experiment_v2.py` | Defaults to `cache_sequenze` |

---

## functions/attack_calendar.py

| Line | Path | Type | Notes |
|------|------|------|-------|
| `salva_calendari_csv()` | `RESULTS_DIR` parameter | Passed from `run_experiment_v2.py` | Writes `calendario_attacchi.csv`, `composizione_test.csv` |

---

## grafici.py

| Line | Path | Type | Notes |
|------|------|------|-------|
| 41 | `BASE = os.path.expanduser('~/VM')` | **Absolute ~ expansion** | **Hard-coded assumption: project at ~/VM** |
| 42 | `OUT = os.path.join(BASE, 'grafici_per_prof')` | Derived from BASE | Output figures |

> **Action required:** If repo not at `~/VM`, either symlink or change line 41.

---

## organizza_risultati.py (in archivio_non_git/superseded_runs/)

| Line | Path | Type | Notes |
|------|------|------|-------|
| Multiple | `os.path.expanduser('~/VM/results_v2/...')` | **Absolute ~ expansion** | Reads from `~/VM/results_v2/analisi_*/` |

---

## genera_tutto_capitolo4.py (in archivio_non_git/superseded_runs/)

| Line | Path | Type | Notes |
|------|------|------|-------|
| Multiple | `os.path.expanduser('~/VM/...')` | **Absolute ~ expansion** | Various I/O |

---

## Static Preprocessing Notebook

| Cell | Path | Type | Notes |
|------|------|------|-------|
| Multiple | `/home/andrealarocca/Scrivania/Tesi/VM/data/raw/...` | **Absolute path** | Original author's machine only |

---

## Summary for Reproduction

### Option 1: Use Standard Layout (Recommended)
```bash
# Place this repo at ~/VM
ln -s /path/to/Tesi_clean_repo ~/VM

# Download/place cleaned Parquet files at:
~/VM/data/processed/
  ais_settimana_1.parquet
  ais_settimana_2.parquet
  ...
  ais_settimana_9.parquet

# Run from repo root:
cd ~/VM
python run_experiment_v2.py --run_id TEST --method BR ...
```

### Option 2: Custom Location
```bash
# Place repo anywhere, e.g., /my/project
cd /my/project

# Option A: Symlink data
ln -s /path/to/parquet/files /my/project/data/processed

# Option B: Modify DATA_PATH in run_experiment_v2.py line 61
# DATA_PATH = "/my/custom/data/processed"

# Option C: Set environment and modify grafici.py line 41
# BASE = "/my/project"
```

### Paths That Work Automatically (Relative)
All paths in `run_experiment_v2.py` and `functions/` are relative to `__file__` — they work from any location if the directory structure is preserved.

### Paths Requiring `~/VM` (Hard-coded)
- `grafici.py` line 41: `BASE = os.path.expanduser('~/VM')`
- `organizza_risultati.py` (superseded)
- `genera_tutto_capitolo4.py` (superseded)

---

## Directory Structure Expected by Code

```
~/VM/                          # or symlinked repo root
├── run_experiment_v2.py
├── grafici.py
├── functions/
│   ├── __init__.py
│   ├── loader.py
│   ├── processing.py
│   ├── model.py
│   ├── classifier_model.py
│   ├── classifier_dataset.py
│   ├── attack_calendar.py
│   ├── regression_features.py
│   ├── attacks.py
│   ├── cl_methods.py
│   ├── cl_metrics_classifier.py
│   ├── evaluation.py
│   ├── clean_normalizer.py
│   ├── prediction.py
│   └── detection_metrics.py
├── data/
│   └── processed/
│       ├── ais_settimana_1.parquet
│       ├── ais_settimana_2.parquet
│       ...
│       └── ais_settimana_9.parquet
├── cache_sequenze/            # Created automatically (11 GB)
├── results_v2/                # Created automatically
├── models_v2/                 # Created automatically
├── checkpoints_v2/            # Created automatically (193 MB)
├── previsioni_classificatore/ # Created automatically (3.5 GB)
└── grafici_per_prof/          # Created by grafici.py
```

---

## Large Artifacts (Not in Repo)

| Artifact | Size | Location | Regeneration |
|----------|------|----------|--------------|
| Cleaned Parquet (9 files) | 1.3 GB | `data/processed/` | Run `static_preprocessing.ipynb` on raw MarineCadastre CSVs |
| Sequence cache | 11 GB | `cache_sequenze/` | Automatic on first run |
| Daily predictions | 3.5 GB | `previsioni_classificatore/` | Automatic during runs |
| Per-day checkpoints | 193 MB | `checkpoints_v2/` | Automatic during runs |
| All model weights | 6 MB | `models_v2/` | Automatic at end of runs |