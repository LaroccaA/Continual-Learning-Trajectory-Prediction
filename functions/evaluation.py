import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Raggio della Terra in metri
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def evaluate_day(model, preprocessor, X_np, y_np, last_pos, dest_real, gaps, device='cuda', batch_size=1024):
    """
    MODIFICA (CLeAN): X_np qui è GREZZO (non normalizzato) — la
    normalizzazione avviene dentro il forward del modello tramite
    CLeANNormalizer, passando raw_input=True. y_np resta normalizzato
    con CLeAN min-max EMA (invertibile), coerentemente con
    ContinualPreprocessor.create_sequences().
    """

    if len(X_np) == 0: return None, None

    loader_eval = DataLoader(TensorDataset(torch.FloatTensor(X_np)), batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch_X in loader_eval:
            preds = model(batch_X[0].to(device), raw_input=True)
            all_preds.append(preds.cpu().numpy())

    preds_scaled = np.vstack(all_preds)

    # Riportiamo tutto in scala reale tramite CLeANNormalizer.inverse_transform
    # (solo inversione min-max EMA, il layer scale/bias non viene applicato a y)
    kinematics_pred = preprocessor.scaler_y.inverse_transform(preds_scaled).numpy()
    kinematics_real = preprocessor.scaler_y.inverse_transform(y_np).numpy()

    sog_pred = kinematics_pred[:, 0]
    cog_sin_pred = kinematics_pred[:, 1]
    cog_cos_pred = kinematics_pred[:, 2]
    sog_real = kinematics_real[:, 0]

    # Applicazione rigorosa del Dead Reckoning
    dt_hours = gaps / 60.0
    distance_nm = sog_pred * dt_hours
    delta_lat = (distance_nm * cog_cos_pred) / 60.0
    lat_rad = np.radians(last_pos[:, 0])
    delta_lon = (distance_nm * cog_sin_pred) / (60.0 * np.cos(lat_rad))

    predicted_coords = np.zeros_like(last_pos)
    predicted_coords[:, 0] = last_pos[:, 0] + delta_lat
    predicted_coords[:, 1] = last_pos[:, 1] + delta_lon

    mse_loss = np.mean((preds_scaled - y_np) ** 2)

    error_metri = haversine_distance(predicted_coords[:, 0], predicted_coords[:, 1],
                                     dest_real[:, 0], dest_real[:, 1])
    ade_metri = np.mean(error_metri)

    return mse_loss, ade_metri
