"""
Calcola l'errore di regressione da usare come feature per il classificatore
(Opzione B). Riusa ESATTAMENTE le stesse funzioni già validate per
l'anomaly detection — predict_positions() e i tre canali di score — non
introduce nessun calcolo nuovo, solo un riuso diverso dello stesso segnale:
prima usato come soglia post-hoc, ora usato come input della rete.
"""

import numpy as np

from functions.prediction import predict_positions
from functions.detection_metrics import (
    compute_anomaly_scores,
    compute_speed_anomaly_scores,
    compute_heading_anomaly_scores,
)


def compute_regression_error_features(regression_model, regression_preprocessor,
                                        X_context, declared_lat, declared_lon,
                                        declared_sog, declared_cog_sin, declared_cog_cos,
                                        last_pos, gaps, device, batch_size=8192):
    """
    Per un batch di sequenze, calcola i 3 errori tra quanto il modello di
    regressione si aspetterebbe e quanto viene dichiarato nel messaggio
    da giudicare.

    FIX PERFORMANCE: batch_size passa esplicitamente a predict_positions,
    che altrimenti userebbe il suo default (1024) indipendentemente da
    come il chiamante ha già suddiviso i dati — causando una doppia
    suddivisione annidata quando questa funzione viene chiamata da un
    loop già a batch, con migliaia di DataLoader ricreati inutilmente.

    Parametri:
        regression_model, regression_preprocessor : il modello di
            regressione COSÌ COM'È in questo momento (il checkpoint del
            giorno corrente nella simulazione CL, non necessariamente il
            modello finale — coerente con la logica "tempo reale" di
            tutto il progetto)
        X_context      : (N, 30, 7) contesto grezzo, identico a quello
                         usato dalla regressione
        declared_lat/lon/sog/cog_sin/cog_cos : (N,) i 5 valori dichiarati
                         nel messaggio da giudicare (quelli che un attacco
                         può aver alterato)
        last_pos       : (N, 2) ultima posizione nota (per il Dead Reckoning)
        gaps           : (N,) minuti trascorsi
        device         : 'cuda' o 'cpu'
        batch_size     : dimensione del batch per l'inferenza interna

    Restituisce:
        (N, 3) array con, in ordine: errore posizione (metri), errore
        velocità (nodi), errore direzione (gradi) — stessa unità fisica
        dei tre canali di score già usati per l'anomaly detection.
    """
    predicted_coords, kinematics_pred = predict_positions(
        regression_model, regression_preprocessor, X_context,
        last_pos, gaps, device=device, batch_size=batch_size
    )

    declared_dest = np.column_stack([declared_lat, declared_lon])
    position_error = compute_anomaly_scores(predicted_coords, declared_dest)

    speed_error = compute_speed_anomaly_scores(kinematics_pred[:, 0], declared_sog)

    heading_error = compute_heading_anomaly_scores(
        kinematics_pred[:, 1], kinematics_pred[:, 2],
        declared_cog_sin, declared_cog_cos
    )

    return np.column_stack([position_error, speed_error, heading_error]).astype(np.float32)
