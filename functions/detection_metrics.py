"""
detection_metrics.py
====================
Metriche di anomaly detection: errore di Dead Reckoning come score,
Precision/Recall/F1/AUC-ROC, e tracking della Recall nel tempo (Recall@giorno).
"""

import numpy as np


EARTH_RADIUS_M = 6371000.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Distanza in metri tra due punti geografici (vettorizzata).
    lat/lon in gradi decimali.
    """
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return EARTH_RADIUS_M * c


def compute_anomaly_scores(predicted_dest, declared_dest):
    """
    Score di anomalia POSIZIONALE: distanza di Dead Reckoning tra la
    posizione predetta dal modello e quella dichiarata nel messaggio AIS.

    Copre gli attacchi che alterano la posizione dichiarata (teleport,
    silent_drift). È STRUTTURALMENTE cieco agli attacchi che alterano
    solo il SOG dichiarato lasciando la posizione coerente (speed_spoofing):
    per quelli serve compute_speed_anomaly_scores().

    Parametri:
        predicted_dest : (N, 2) lat/lon predette dal modello (Dead Reckoning)
        declared_dest   : (N, 2) lat/lon dichiarate nel messaggio (eventualmente attaccate)

    Restituisce:
        scores : (N,) distanza in metri — più alto = più anomalo
    """
    return haversine_distance(
        predicted_dest[:, 0], predicted_dest[:, 1],
        declared_dest[:, 0],  declared_dest[:, 1]
    )


def compute_speed_anomaly_scores(predicted_sog, declared_sog):
    """
    Score di anomalia sulla VELOCITÀ: scarto assoluto tra il SOG previsto
    dal modello (dal contesto recente, mai attaccato) e il SOG dichiarato
    nel messaggio (eventualmente falsificato).

    Canale complementare a compute_anomaly_scores(): copre esattamente
    l'attacco speed_spoofing, che altera solo il SOG dichiarato lasciando
    la posizione coerente — invisibile al canale posizionale per
    costruzione. Il modello calcola già predicted_sog come parte della sua
    normale uscita cinematica (predict_positions lo restituisce sempre
    come secondo valore); qui viene solo confrontato con quello dichiarato,
    nessun modello nuovo, nessun riaddestramento.

    Parametri:
        predicted_sog : (N,) SOG previsto dal modello, in nodi
        declared_sog  : (N,) SOG dichiarato nel messaggio, in nodi
                        (eventualmente attaccato)

    Restituisce:
        scores : (N,) scarto assoluto in nodi — più alto = più anomalo
    """
    return np.abs(predicted_sog - declared_sog)


def compute_heading_anomaly_scores(predicted_cog_sin, predicted_cog_cos,
                                     declared_cog_sin, declared_cog_cos):
    """
    Score di anomalia sulla DIREZIONE: angolo, in gradi, tra il COG
    previsto dal modello e quello dichiarato nel messaggio.

    Terzo canale, complementare a posizione e velocità: copre
    course_spoofing, che altera solo il COG dichiarato lasciando
    posizione e SOG coerenti — invisibile agli altri due canali per
    costruzione, esattamente come speed_spoofing lo è al canale
    posizionale. Il COG previsto è già calcolato dal modello come parte
    della sua uscita cinematica (kinematics_pred[:,1:3]); qui viene solo
    confrontato con quello dichiarato.

    Parametri:
        predicted_cog_sin/cos : (N,) componenti del COG previsto dal modello
        declared_cog_sin/cos  : (N,) componenti del COG dichiarato
                                (eventualmente attaccato)

    Restituisce:
        scores : (N,) angolo in gradi tra i due vettori, in [0,180] —
                 più alto = più anomalo (180 = direzione opposta)
    """
    dot = predicted_cog_sin * declared_cog_sin + predicted_cog_cos * declared_cog_cos
    # Normalizzazione difensiva: i vettori dovrebbero già essere unitari
    # (cog_sin²+cog_cos²=1 per costruzione), ma dopo interpolazioni o
    # predizioni del modello potrebbero non esserlo esattamente.
    norm_pred = np.sqrt(predicted_cog_sin**2 + predicted_cog_cos**2)
    norm_decl = np.sqrt(declared_cog_sin**2 + declared_cog_cos**2)
    cos_angle = dot / (norm_pred * norm_decl + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_threshold(scores_clean, percentile=99):
    """
    Calibra la soglia di anomalia sul percentile della distribuzione
    di errori di sequenze PULITE (training/validation, mai di attacco).
    """
    return float(np.percentile(scores_clean, percentile))


def precision_recall_f1(labels, scores, threshold):
    """
    Calcola Precision, Recall, F1 per una soglia fissa.

    Parametri:
        labels    : (N,) 0=normale, 1=attacco (ground truth)
        scores    : (N,) score di anomalia (errore Dead Reckoning in metri)
        threshold : soglia oltre la quale si dichiara anomalia

    Restituisce: dict con TP, FP, TN, FN, precision, recall, f1
    """
    predictions = (scores > threshold).astype(int)

    TP = int(np.sum((predictions == 1) & (labels == 1)))
    FP = int(np.sum((predictions == 1) & (labels == 0)))
    TN = int(np.sum((predictions == 0) & (labels == 0)))
    FN = int(np.sum((predictions == 0) & (labels == 1)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "threshold": round(threshold, 2),
    }


def roc_auc(labels, scores):
    """
    Calcola l'AUC-ROC manualmente (senza sklearn) tramite il metodo
    rank-based (equivalente al test di Mann-Whitney U).

    Parametri:
        labels : (N,) 0/1
        scores : (N,) score continuo

    Restituisce: AUC in [0,1], 0.5 = random, 1.0 = perfetto
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Gestione dei pari merito (tie) con rank medio
    sorted_scores = scores[order]
    sorted_ranks  = ranks[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            sorted_ranks[i:j] = np.mean(sorted_ranks[i:j])
        i = j
    ranks[order] = sorted_ranks

    sum_ranks_pos = np.sum(ranks[labels == 1])
    auc = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    return float(auc)


def precision_recall_curve(labels, scores, n_thresholds=50):
    """
    Calcola la curva Precision-Recall su una griglia di soglie
    derivata dal range degli score osservati.

    Restituisce: dict con liste "thresholds", "precisions", "recalls", "f1s"
    """
    lo, hi = float(np.min(scores)), float(np.max(scores))
    thresholds = np.linspace(lo, hi, n_thresholds)

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        m = precision_recall_f1(labels, scores, t)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

    return {
        "thresholds": thresholds.tolist(),
        "precisions": precisions,
        "recalls":    recalls,
        "f1s":        f1s,
    }


def best_threshold_by_f1(labels, scores, n_thresholds=100):
    """
    Trova la soglia che massimizza F1 su una griglia di candidati.
    Utile come alternativa al percentile fisso quando si vuole
    il miglior compromesso assoluto.
    """
    curve = precision_recall_curve(labels, scores, n_thresholds=n_thresholds)
    best_idx = int(np.argmax(curve["f1s"]))
    return {
        "threshold": curve["thresholds"][best_idx],
        "precision": curve["precisions"][best_idx],
        "recall":    curve["recalls"][best_idx],
        "f1":        curve["f1s"][best_idx],
    }
