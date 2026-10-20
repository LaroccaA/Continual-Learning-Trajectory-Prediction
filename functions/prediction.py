"""
prediction.py
=============
Wrapper di inferenza che applica il modello + Dead Reckoning a livello di
singola sequenza (non aggregato come evaluate_day del modulo CL), per poter
calcolare uno score di anomalia per ogni sequenza nel test set sintetico.

La logica di Dead Reckoning è IDENTICA a quella in
LSTM_CF_FIX/functions/evaluation.py — qui viene solo esposta a livello
di singolo campione invece che come MSE/ADE aggregati.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def predict_positions(model, preprocessor, X_np, last_pos, gaps,
                       device='cuda', batch_size=1024):
    """
    Esegue l'inferenza del modello e applica il Dead Reckoning per ottenere
    la posizione PREDETTA per ogni sequenza (non aggregata in una media).

    Parametri:
        model        : TrajectoryLSTM già addestrato
        preprocessor : ContinualPreprocessor con lo scaler corrente
        X_np         : (N, seq_len, n_features) sequenze GREZZE — la
                       normalizzazione avviene internamente al modello
                       (model.clean_x), mai qui. Passare raw_input=True
                       al modello è necessario, non opzionale: senza,
                       il modello tratterebbe valori fisici (lat~40,
                       lon~14, ecc.) come se fossero già in [0,1].
        last_pos     : (N, 2) ultima posizione nota (lat, lon) reale
        gaps         : (N,) gap in minuti tra last_pos e il punto predetto
        device       : 'cuda' o 'cpu'
        batch_size   : batch size per l'inferenza

    Restituisce:
        predicted_coords : (N, 2) posizioni predette via Dead Reckoning
        kinematics_pred  : (N, 3) SOG/COG_sin/COG_cos predetti (scala reale)
    """
    if len(X_np) == 0:
        return np.empty((0, 2)), np.empty((0, 3))

    loader_eval = DataLoader(
        TensorDataset(torch.FloatTensor(X_np)),
        batch_size=batch_size, shuffle=False
    )
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch_X in loader_eval:
            # raw_input=True: X è grezzo, la normalizzazione (model.clean_x)
            # avviene qui dentro con le statistiche CORRENTI, in eval mode
            # senza aggiornarle. Ometterlo produce predizioni silenziosamente
            # sbagliate: nessun errore, solo numeri senza senso.
            preds = model(batch_X[0].to(device), raw_input=True)
            all_preds.append(preds.cpu().numpy())

    preds_scaled = np.vstack(all_preds)
    # inverse_transform restituisce SEMPRE un tensore PyTorch (mai numpy),
    # anche con input numpy. Senza .numpy(), l'aritmetica successiva con
    # array numpy (last_pos, gaps) mescola tensori e array causando errori
                                                                     
    kinematics_pred = preprocessor.scaler_y.inverse_transform(preds_scaled).numpy()

    sog_pred     = kinematics_pred[:, 0]
    cog_sin_pred = kinematics_pred[:, 1]
    cog_cos_pred = kinematics_pred[:, 2]

    # --- Dead Reckoning, identico a evaluation.py del progetto CL ---
    dt_hours = gaps / 60.0
    distance_nm = sog_pred * dt_hours
    delta_lat = (distance_nm * cog_cos_pred) / 60.0
    lat_rad = np.radians(last_pos[:, 0])
    delta_lon = (distance_nm * cog_sin_pred) / (60.0 * np.cos(lat_rad))

    predicted_coords = np.zeros_like(last_pos)
    predicted_coords[:, 0] = last_pos[:, 0] + delta_lat
    predicted_coords[:, 1] = last_pos[:, 1] + delta_lon

    return predicted_coords, kinematics_pred
