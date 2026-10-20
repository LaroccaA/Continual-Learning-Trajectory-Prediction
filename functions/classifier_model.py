"""
Modello di CLASSIFICAZIONE per il rilevamento di attacchi — separato dal
modello di regressione (TrajectoryLSTM), pesi propri, non condivisi.

INPUT del classificatore (Opzione B, confermata dalla relatrice via email
l'11/08/2026): il classificatore vede sia i dati grezzi sia un segnale
calcolato dal modello di regressione.

    1. Contesto grezzo: le stesse 30 righe x 7 feature usate dalla
       regressione (lat, lon, sog, cog_sin, cog_cos, heading_sin, heading_cos)
    2. Messaggio da giudicare: i 5 valori dichiarati nel passo successivo
       che un attacco può aver alterato — lat, lon, sog, cog_sin, cog_cos
    3. Errore di regressione: quanto quei 5 valori dichiarati si discostano
       da quello che il modello di regressione avrebbe previsto lui stesso,
       guardando lo stesso contesto — 3 numeri (errore posizione in metri,
       errore velocità in nodi, errore direzione in gradi), calcolati
       riusando le stesse funzioni già validate per l'anomaly detection
       (predict_positions, compute_anomaly_scores, compute_speed_anomaly_scores,
       compute_heading_anomaly_scores in detection_metrics.py) — nessun
       calcolo nuovo, solo un riuso come FEATURE invece che come soglia
       post-hoc.

Il "messaggio da giudicare" e l'errore di regressione entrano nella rete
come feature aggiuntive concatenate all'ultimo stato nascosto della LSTM
di contesto, prima della testa di classificazione finale.
"""

import torch
import torch.nn as nn

from functions.clean_normalizer import CLeANNormalizer


# Dimensione delle feature aggiuntive (non temporali):
#   5 valori dichiarati da giudicare (lat, lon, sog, cog_sin, cog_cos)
# + 3 errori di regressione (posizione, velocità, direzione)
N_EXTRA_FEATURES = 5 + 3


class AttackClassifierLSTM(nn.Module):
    """
    Architettura a due rami, poi uniti:

      ramo temporale:  30x7 contesto grezzo --> LSTM --> stato nascosto
      ramo statico:    5 valori dichiarati + 3 errori regressione (grezzi)
                        --> normalizzati internamente (CLeAN separato,
                            NON condiviso col modello di regressione)

    I due rami si concatenano e passano per una piccola testa MLP che
    produce un unico logit (classificazione binaria: attacco / non attacco).
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 extra_features=N_EXTRA_FEATURES,
                 use_clean=True, clean_eta=0.99):
        super().__init__()

        self.use_clean = use_clean
        if use_clean:
            # Normalizzatore per il CONTESTO (30x7), separato da quello
                                                                           
            self.clean_x = CLeANNormalizer(n_features=input_size, eta=clean_eta,
                                            full_pipeline=True)
            # Normalizzatore per le feature statiche (i 5 dichiarati + i 3
            # errori di regressione), anch'esso indipendente.
            self.clean_extra = CLeANNormalizer(n_features=extra_features,
                                                eta=clean_eta, full_pipeline=True)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        head_input_size = hidden_size + extra_features
        self.head = nn.Sequential(
            nn.Linear(head_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),   # un solo logit: attacco / non attacco
        )

    def forward(self, x_context, x_extra, raw_input=False):
        """
        Parametri:
            x_context : (batch, 30, 7) contesto grezzo (stesso formato
                        della regressione)
            x_extra   : (batch, 8) feature statiche grezze — 5 valori
                        dichiarati + 3 errori di regressione, in unità
                        fisiche (non normalizzate)
            raw_input : se True, normalizza internamente sia x_context sia
                        x_extra (stesso schema del modello di regressione:
                        raw_input=True in training/inferenza sui dati
                        grezzi, False se già normalizzati esternamente —
                        caso raro, mantenuto per coerenza di interfaccia)

        Restituisce:
            logit : (batch, 1) — passare per sigmoid() per ottenere una
                    probabilità in [0,1]; usare BCEWithLogitsLoss in
                    training (più stabile numericamente della sigmoid+BCE
                    separate).
        """
        if self.use_clean and raw_input:
            x_context = self.clean_x(x_context)
            x_extra = self.clean_extra(x_extra)

        _, (h_n, _) = self.lstm(x_context)
        h_last = h_n[-1]  # (batch, hidden_size) — ultimo layer, ultimo timestep

        combined = torch.cat([h_last, x_extra], dim=1)
        logit = self.head(combined)
        return logit
