"""
clean_normalizer.py
====================
Adattamento di EMANet (normalizers.py, repo ufficiale CLeAN di
Marasco, Evangelista, Loli Piccolomini, Colajanni — arXiv:2603.17548)
al caso AIS/LSTM di questa tesi.

Sostituisce il MinMaxScaler con partial_fit (monotono crescente, non
dimentica mai gli outlier storici come il placeholder SOG=99.9) con una
normalizzazione adattiva basata su Exponential Moving Average.

Logica replicata esattamente da normalizers.py (classe EMANet):
- Le statistiche min/max vengono calcolate sul BATCH corrente dentro
  forward(), quando il modulo è in training mode.
- running = running * eta + batch * (1 - eta)
- In eval mode si usano le running stats correnti, senza aggiornarle.

ADATTAMENTO (non presente nel repo originale, che fa solo classificazione
binaria): qui serve anche poter invertire la normalizzazione per il Dead
Reckoning. Per questo CLeANNormalizer ha un parametro full_pipeline:
- full_pipeline=True  (input X): minmax EMA + scale/bias appresi, come EMANet
- full_pipeline=False (target y): solo minmax EMA, invertibile esattamente
"""

import numpy as np
import torch
import torch.nn as nn


class CLeANNormalizer(nn.Module):
    def __init__(self, n_features, eta=0.99, eps=1e-6, full_pipeline=True):
        super().__init__()
        self.n_features = n_features
        self.momentum = eta
        self.eps = eps
        self.full_pipeline = full_pipeline

        self.register_buffer("running_mx", torch.zeros(n_features))
        self.register_buffer("running_Mx", torch.ones(n_features))

        if full_pipeline:
            self.scale = nn.Parameter(torch.ones(n_features))
            self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.copy()).float()

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.n_features)

        if self.training:
            batch_mx = x_flat.min(dim=0)[0]
            batch_Mx = x_flat.max(dim=0)[0]
            with torch.no_grad():
                self.running_mx.copy_(
                    self.running_mx * self.momentum + batch_mx * (1 - self.momentum)
                )
                self.running_Mx.copy_(
                    self.running_Mx * self.momentum + batch_Mx * (1 - self.momentum)
                )

        denom = (self.running_Mx - self.running_mx).clamp(min=self.eps)
        norm_x = (x_flat - self.running_mx) / denom

        if self.full_pipeline:
            norm_x = self.scale * norm_x + self.bias

        return norm_x.reshape(orig_shape)

    def minmax_normalize(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.copy()).float()
        denom = (self.running_Mx - self.running_mx).clamp(min=self.eps)
        return (x - self.running_mx) / denom

    @torch.no_grad()
    def update_statistics(self, X_chunk):
        """
        Aggiorna manualmente le running stats con un chunk di dati grezzi,
        senza passare per forward(). Usata da scaler_y nel preprocessor
        (il target non passa mai dentro un forward del modello).
        """
        if isinstance(X_chunk, np.ndarray):
            X_chunk = torch.from_numpy(X_chunk.copy()).float()
        X_flat = X_chunk.reshape(-1, self.n_features)

        batch_mx = X_flat.min(dim=0)[0]
        batch_Mx = X_flat.max(dim=0)[0]

        self.running_mx.copy_(
            self.running_mx * self.momentum + batch_mx * (1 - self.momentum)
        )
        self.running_Mx.copy_(
            self.running_Mx * self.momentum + batch_Mx * (1 - self.momentum)
        )

    def inverse_transform(self, x_normalized):
        """
        Inverte SOLO la normalizzazione min-max (mai scale/bias, che non
        sono invertibili in forma chiusa). Usare solo su normalizer con
        full_pipeline=False (scaler_y).
        """
        if isinstance(x_normalized, np.ndarray):
            x_normalized = torch.from_numpy(x_normalized.copy()).float()
        denom = (self.running_Mx - self.running_mx).clamp(min=self.eps)
        return x_normalized * denom + self.running_mx
