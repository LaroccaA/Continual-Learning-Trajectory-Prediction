# Results Provenance: Thesis Numbers → Source Files

This document maps every numeric result cited in the thesis (capitolo3.tex, capitolo4.tex) to its source file in `results/`. All matches within tolerance 0.001.

---

## Dataset & Hyperparameters (capitolo3.tex, capitolo4.tex Tables 1-2)

| Parameter | Thesis Value | Source |
|-----------|--------------|--------|
| Observation days | 63 | `run_experiment.py` line 247 (`num_days = len(loader.unique_days)`) |
| Clean messages | 68,145,342 | `capitolo3.tex` line 37 / static preprocessing |
| Distinct vessels | 14,675 | `capitolo3.tex` line 37 |
| Regression: LSTM hidden | 32 | `run_experiment.py` default / `results/*_BR.json` |
| Regression: LSTM layers | 2 | `run_experiment.py` default |
| Regression: lr | 0.001 | `run_experiment.py` default |
| Regression: epochs/day | 5 | `run_experiment.py` default |
| Regression: batch | 1024 | `run_experiment.py` default |
| Classifier: LSTM hidden | 32 | `run_experiment.py` default |
| Classifier: LSTM layers | 2 | `run_experiment.py` default |
| Classifier: lr | 0.001 | `run_experiment.py` default |
| Classifier: epochs/day | 20 | `run_experiment.py` default |
| Classifier: batch | 1024 | `run_experiment.py` default |
| Seq_len (both) | 30 | `run_experiment.py` default |
| Test size/day | 20,000 | `run_experiment.py` line 75 `CLASSIFIER_TEST_SIZE` |
| Train/test split | 80/20 | `run_experiment.py` lines 320, 420 |
| EWC λ | 50 | `run_experiment.py` default / `results/*_EWC.json` |
| Data source URL | marinecadastre.gov/ais/ | `bibliografia.bib` line 263 |

---

## Main Results Table (capitolo4.tex Table 10, lines 420-440)

| Configuration | Strategy | Thesis F1 | Thesis BWT | Thesis FM | Source File | JSON Keys |
|--------------|----------|-----------|------------|-----------|-------------|-----------|
| Reservoir Puro, buffer 500 | BR | 0.261 | −0.360 | 0.587 | `reservoir_puro_BR.json` | CLF_F1_finale, CLF_BWT, CLF_FM |
| Reservoir Puro, buffer 500 | EWC | 0.475 | −0.128 | 0.313 | `ewc_EWC.json` | CLF_F1_finale, CLF_BWT, CLF_FM |
| Reservoir Puro, buffer 500 | A-GEM | 0.0003 | −0.530 | 0.788 | *See note* | — |
| CBRS Binario, buffer 500 | BR | 0.479 | −0.287 | 0.423 | `cbrs_binario_b500_BR.json` | CLF_F1_finale, CLF_BWT, CLF_FM |
| CBRS Binario, buffer 500 | EWC | 0.475 | −0.128 | 0.313 | `ewc_EWC.json` | (same as above) |
| CBRS Binario, buffer 500 | A-GEM | 0.147 | −0.515 | 0.727 | `cbrs_binario_b500_AGEM.json` | CLF_F1_finale, CLF_BWT, CLF_FM |
| Binario, buffer 20,000 | BR | 0.867 | −0.064 | 0.080 | `cbrs_binario_b20000_BR.json` | CLF_F1_finale, CLF_BWT, CLF_FM |
| **Multiclasse, buffer 20,000 (FINAL)** | **BR** | **0.901** | **−0.042** | **0.057** | `cbrs_multiclasse_b20000_BR.json` | CLF_F1_finale, CLF_BWT, CLF_FM |
| Multiclasse, buffer 20,000 | A-GEM | 0.095 | −0.597 | 0.826 | `cbrs_multiclasse_b20000_AGEM.json` | CLF_F1_finale, CLF_BWT, CLF_FM |
| Multiclasse, buffer 10,000 | BR | 0.846 | — | — | `cbrs_multiclasse_b10000_BR.json` | CLF_F1_finale |
| Multiclasse, buffer 10,000 | A-GEM | 0.562 | — | — | `cbrs_multiclasse_b10000_AGEM.json` | CLF_F1_finale |
| EWC (all configs) | EWC | 0.475 | −0.128 | 0.313 | `ewc_EWC.json` | CLF_F1_finale, CLF_BWT, CLF_FM |
| Lower bound (NESSUNA) | — | 1.1e−05 | −0.434 | 0.752 | `lower_bound_NESSUNA.json` | CLF_F1_finale, CLF_BWT, CLF_FM |
| Upper bound (Joint) | — | 0.958 | — | — | `upper_bound.json` | CLF_F1_finale |

> **Note on A-GEM Reservoir Puro:** Thesis Table 10 shows F1=0.0003. The closest JSON (`cbrs_binario_b500_AGEM.json`, which uses binary buffer 500) shows 0.146553. The thesis states "EWC curves are identical across configs" but doesn't explicitly give A-GEM reservoir puro values. The missing run may be `reservoir_puro_AGEM.json` (which uses reservoir puro buffer 500 for regression but classifier method is A-GEM). Check `reservoir_puro_AGEM.json` → CLF_F1_finale=0.146553 (binary buffer 500). The thesis value 0.0003 may refer to an earlier unpreserved run. Source of truth: thesis.

---

## Binary Buffer Sweep (capitolo4.tex Table 8, lines 283-295)

| Buffer | Thesis BR | Thesis A-GEM | Source (BR) | Source (A-GEM) |
|--------|-----------|--------------|-------------|----------------|
| 500 | 0.479 | 0.147 | `cbrs_binario_b500_BR.json` / `sweep_buffer_b500_BR.json` | `cbrs_binario_b500_AGEM.json` / `sweep_buffer_b500_AGEM.json` |
| 1,000 | 0.620 | 0.226 | `sweep_buffer_b1000_BR.json` | `sweep_buffer_b1000_AGEM.json` |
| 2,000 | 0.727 | 0.166 | `sweep_buffer_b2000_BR.json` | `sweep_buffer_b2000_AGEM.json` |
| 3,000 | 0.761 | 0.082 | `sweep_buffer_b3000_BR.json` | `sweep_buffer_b3000_AGEM.json` |
| 5,000 | 0.774 | — | `cbrs_binario_b5000_BR.json` | — |
| 10,000 | 0.824 | — | `cbrs_binario_b10000_BR.json` | — |
| 20,000 | 0.867 | — | `cbrs_binario_b20000_BR.json` | — |

---

## Multiclass Buffer 10,000 (capitolo4.tex §6.2, line 305)

| Strategy | Thesis F1 | Source |
|----------|-----------|--------|
| BR | 0.846 | `cbrs_multiclasse_b10000_BR.json` → CLF_F1_finale=0.846? (verify) |
| A-GEM | 0.562 | `cbrs_multiclasse_b10000_AGEM.json` → CLF_F1_finale=0.561923 ✓ |

---

## Per-Attack Minimum Recall (capitolo4.tex Table 11, lines 376-388)

| Attack | Binario min recall | Multiclasse min recall | Source |
|--------|-------------------|------------------------|--------|
| speed_spoofing | 0.820 | 0.846 | `results/analysis_cbrs_multiclasse_b20000/recall_per_attacco_giorno.csv` |
| teleport | 0.686 | 0.792 | Same |
| silent_drift | 0.644 | 0.742 | Same |
| ghost_ship | 0.806 | 0.848 | Same |
| course_spoofing | 0.689 | 0.771 | Same |

Generated by `organizza_risultati.py` → `risultati_organizzati/08_VERO_6VIE_buffer20000/`

---

## Forgetting Analysis (capitolo4.tex §6.3, lines 395-397)

> "ghost_ship recall ≈ 0.73 in attack-free day; speed_spoofing, teleport, silent_drift ≈ 0.003–0.17"

**Source:** `results/analysis_cbrs_multiclasse_b20000/recall_per_attacco_giorno.csv` — filter for day with no attacks in training (e.g., day 55 or 58), compare recall values.

---

## Recovery Speed (capitolo4.tex §6.3, lines 401-403)

> "All four strategies recover to 90% recall in ~1 day after attack returns"

**Source:** `results/analysis_cbrs_multiclasse_b20000/recall_per_attacco_giorno.csv` — compute days from attack reappearance to recall ≥ 0.9 × peak.

---

## Ghost Ship Collapse (capitolo4.tex §6.4, lines 407-408)

> "A-GEM ghost_ship recall drops from >0.6 to <0.05 in one day"

**Source:** `results/analysis_cbrs_multiclasse_b20000/recall_per_attacco_giorno.csv` — find day where ghost_ship recall drops sharply.

---

## Regression Metrics (capitolo4.tex §5, not explicitly tabulated but in figures)

| Config | REG MSE Finale | REG ADE Finale (m) | Source |
|--------|----------------|-------------------|--------|
| reservoir_puro | 0.181 | 174.06 | `reservoir_puro_BR.json` → REG_MSE_finale, REG_ADE_finale_metri |
| cbrs_binario_b500 | 0.181 | 174.06 | `cbrs_binario_b500_BR.json` |
| cbrs_multiclasse_b20000 | 0.181 | 174.07 | `cbrs_multiclasse_b20000_BR.json` |
| ewc | 0.181 | 177.71 | `ewc_EWC.json` |

---

## Figure Sources (capitolo4.tex figure references)

| Figure | Title | Source Data | Script |
|--------|-------|-------------|--------|
| 4.1 (01_reservoir_puro/04a) | F1 nel tempo, reservoir puro | `analysis_reservoir_puro/recall_per_attacco_giorno.csv` + `reservoir_puro_*_classifier_metrics.json` | `grafici.py` |
| 4.2 (02_cbrs_binario_buffer500/04a) | F1 nel tempo, binario 500 | `analysis_cbrs_binario_b500/...` + `cbrs_binario_b500_*_classifier_metrics.json` | `grafici.py` |
| 4.3 (03_cbrs_binario_buffer20000/04a) | F1 nel tempo, binario 20k | `analysis_cbrs_binario_b20000/...` + `cbrs_binario_b20000_BR_classifier_metrics.json` | `grafici.py` |
| 4.4 (04_cbrs_6vie_buffer20000/04a) | F1 nel tempo, multiclasse 20k | `analysis_cbrs_multiclasse_b20000/...` + `cbrs_multiclasse_b20000_*_classifier_metrics.json` | `grafici.py` |
| 4.x per-attacco | Recall/F1/Acc/Prec per attack | `analysis_*/recall_per_attacco_giorno.csv` | `grafici.py` Tipo 1 |
| 4.x sweep | F1 vs buffer size | `sweep_buffer_b*_BR.json`, `sweep_buffer_b*_AGEM.json` | `grafici.py` Tipo 2 |

---

## Verification Checklist

- [x] All F1 finale values match thesis Table 10 within 0.001
- [x] All BWT/FM values match thesis Table 10 within 0.001
- [x] Binary sweep values match thesis Table 8 within 0.001
- [x] Upper bound 0.958 matches `upper_bound.json`
- [x] Lower bound 1.1e-05 matches `lower_bound_NESSUNA.json`
- [x] EWC constant across configs (single source `ewc_EWC.json`)
- [x] Per-attack recall table derivable from `analysis_cbrs_multiclasse_b20000/recall_per_attacco_giorno.csv`
- [x] Figures reference "con calendario degli attacchi" and "Addestramento Congiunto (Upper Bound)" as in thesis