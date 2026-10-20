"""
Metriche di Continual Learning applicate al CLASSIFICATORE — analoghe a
quelle già usate per la regressione (BWT/FM/FWT), ma su metriche di
classificazione (accuracy, precision, recall, F1) invece che su MSE.

Riusa la stessa struttura a matrice di retention R[i,j] = metrica sul
giorno j (usando SOLO il suo 20% di test) quando il training è arrivato
al giorno i.
"""

import numpy as np


def binary_classification_metrics(y_true, y_pred_logit, threshold=0.5):
    """
    Calcola accuracy/precision/recall/F1 da logit grezzi (prima della
    sigmoid) e etichette vere. Nessuna libreria esterna, stesso stile
    già usato in detection_metrics.py per l'anomaly detection.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = 1.0 / (1.0 + np.exp(-np.asarray(y_pred_logit)))
    y_pred = (prob >= threshold).astype(int)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (TP + TN) / max(TP + FP + TN + FN, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
    }


def compute_cl_metrics_classifier(R_f1, n_baseline=3):
    """
    BWT/FM/FWT sulla matrice di retention del classificatore, usando F1
    come metrica di riferimento (più informativa di accuracy quando le
    classi sono bilanciate 50/50, ma robusta anche se non lo fossero).

    Stessa identica logica già usata per la regressione (compute_metrics
    in run_experiment.py), solo applicata a una matrice di F1 invece che
    di MSE. Qui PIÙ ALTO è meglio (F1), quindi il segno delle formule è
    invertito rispetto alla versione basata su loss (dove più basso è
    meglio) — vedi i commenti su ciascuna riga.
    """
    num_days = R_f1.shape[0]

    # BWT: quanto il modello finale è migliore (F1 più alto) sui giorni
    # passati rispetto a quando li aveva appena imparati. POSITIVO qui
                                                                        
    # negativo significa miglioramento) — perché F1 più alto = meglio.
    bwt_vals = [R_f1[-1, j] - R_f1[j, j] for j in range(num_days - 1)
                if not np.isnan(R_f1[-1, j]) and not np.isnan(R_f1[j, j])]
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0

    # FM: quanto il modello è peggiorato (F1 più basso) rispetto al suo
    # picco storico su quel giorno.
    fm_vals = []
    for j in range(num_days - 1):
        storico = R_f1[j:, j]
        storico = storico[~np.isnan(storico)]
        if len(storico) and not np.isnan(R_f1[-1, j]):
            fm_vals.append(np.max(storico) - R_f1[-1, j])
    fm = float(np.mean(fm_vals)) if fm_vals else 0.0

    n_base = min(n_baseline, num_days)
    baseline_points = [R_f1[i, i] for i in range(n_base) if not np.isnan(R_f1[i, i])]
    baseline = float(np.mean(baseline_points)) if baseline_points else np.nan

    fwt_vals = [R_f1[j, j] - baseline for j in range(n_base, num_days)
                if not np.isnan(R_f1[j, j])]
    fwt = float(np.mean(fwt_vals)) if fwt_vals else 0.0

    return {
        "BWT": round(bwt, 6),
        "FM": round(fm, 6),
        "FWT": round(fwt, 6),
        "F1_finale": round(float(np.nanmean(R_f1[-1, :])), 6),
    }
