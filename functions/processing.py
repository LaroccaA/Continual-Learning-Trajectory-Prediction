import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from functions.clean_normalizer import CLeANNormalizer


def _resample_one_group(segment_df, feature_cols, seq_len, gap_threshold):
    """
    Elabora UN singolo (mmsi, segment_id) fino a PRIMA della costruzione
    delle finestre: dedup, resample 1min + interpolazione temporale,
    normalizzazione dei vettori direzione, calcolo gap.

    Puramente deterministica: dipende solo da (segment_df, feature_cols,
    seq_len, gap_threshold). NON tocca mai lo scaler né alcuno stato del
    preprocessor — è per questo che il suo output può essere cachato senza
    reintrodurre il bug della cache originale (che invece cachava dati
    GIÀ NORMALIZZATI con uno scaler poi superato).

    Restituisce gli array pre-finestra, oppure None se il segmento è troppo
    corto per produrre anche una sola sequenza.
    """
    segment_df = segment_df.set_index('base_date_time')
    segment_df = segment_df[~segment_df.index.duplicated(keep='first')]
    segment_df = segment_df.resample('1min').interpolate(method='time').dropna()
    segment_df = segment_df.reset_index()

    n = len(segment_df)
    if n <= seq_len:
        return None

    mag_cog = np.sqrt(segment_df['cog_sin']**2 + segment_df['cog_cos']**2)
    mag_cog = np.where(mag_cog == 0, 1, mag_cog)
    segment_df['cog_sin'] = segment_df['cog_sin'] / mag_cog
    segment_df['cog_cos'] = segment_df['cog_cos'] / mag_cog

    if 'heading_sin' in segment_df.columns and 'heading_cos' in segment_df.columns:
        mag_hdg = np.sqrt(segment_df['heading_sin']**2 + segment_df['heading_cos']**2)
        mag_hdg = np.where(mag_hdg == 0, 1, mag_hdg)
        segment_df['heading_sin'] = segment_df['heading_sin'] / mag_hdg
        segment_df['heading_cos'] = segment_df['heading_cos'] / mag_hdg

    segment_df['gap'] = segment_df['base_date_time'].diff().dt.total_seconds() / 60.0

    return (
        segment_df[feature_cols].values,                                  
        segment_df[['sog', 'cog_sin', 'cog_cos']].values,                    
        segment_df[['latitude', 'longitude']].values,                   
        segment_df['gap'].values,                                     
    )


def _build_windows(pre, seq_len, gap_threshold):
    """
    Costruisce le finestre scorrevoli dagli array pre-finestra.
    Vettorizzato con numpy, nessun loop Python: equivalente esatto del
    ciclo "for i in range(seq_len, n)" della versione originale.
    """
    vals_raw, targets_raw, coords, gaps = pre
    n = len(vals_raw)
    if n <= seq_len:
        return None

    X_windows = np.lib.stride_tricks.sliding_window_view(vals_raw, seq_len, axis=0)
    X_windows = np.moveaxis(X_windows, -1, 1)
    X_windows = X_windows[:-1]

    gaps_windows = np.lib.stride_tricks.sliding_window_view(gaps, seq_len)
    gaps_windows = gaps_windows[1:]

    valid_mask = ~np.any(gaps_windows > gap_threshold, axis=1)
    if not np.any(valid_mask):
        return None

    return (
        X_windows[valid_mask],
        targets_raw[seq_len:n][valid_mask],
        coords[seq_len-1:n-1][valid_mask],
        coords[seq_len:n][valid_mask],
        gaps[seq_len:n][valid_mask],
    )


class ContinualPreprocessor:
    """
    MODIFICHE rispetto alla versione originale:

    1. FIX CACHE (invariato nella sostanza): le sequenze non vengono MAI
       cachate in forma normalizzata. La normalizzazione di y avviene
       sempre con lo stato CORRENTE di scaler_y, ad ogni chiamata.

    2. CLeAN (Marasco et al., arXiv:2603.17548): scaler_y è
       CLeANNormalizer(full_pipeline=False) — solo min-max EMA, invertibile
       esattamente, necessario per il Dead Reckoning. La normalizzazione
       dell'INPUT X avviene dentro TrajectoryLSTM.forward() via model.clean_x.

    3. PERFORMANCE — parallelizzazione: il resample+interpolate per-gruppo è
       obbligatoriamente per-gruppo (pandas solleva NotImplementedError se si
       tenta la versione raggruppata). I gruppi, indipendenti tra loro,
       vengono distribuiti sui core con joblib.

    4. PERFORMANCE — cache degli array GREZZI (novità): la parte costosa
       (resample, interpolazione, gap) è deterministica sui dati del giorno
       e NON dipende dallo scaler. Viene quindi calcolata una sola volta per
       giorno e salvata su disco in forma NON normalizzata. Ad ogni
       rivalutazione dello stesso giorno si ricaricano gli array grezzi e si
       riapplica scaler_y CORRENTE.

       Differenza cruciale rispetto al bug originale: quella cache salvava
       le sequenze GIÀ NORMALIZZATE, che restavano congelate a uno scaler
       superato causando il disallineamento di scala. Qui si cachea solo
       ciò che è invariante; tutto ciò che dipende dallo scaler è sempre
       ricalcolato. Il fix del bug è quindi pienamente preservato.
    """

    # Numero di core usati da joblib. Con piu' run in parallelo sulla stessa
    # macchina va limitato (es. 3 per processo su 8 core), altrimenti ogni
    # processo tenta di prendersi tutti i core e si ostacolano a vicenda.
    # Impostabile con la variabile d'ambiente CL_N_JOBS.
    N_JOBS_DEFAULT = int(os.environ.get('CL_N_JOBS', '-1'))

    def __init__(self, feature_cols, clean_eta=0.99, cache_dir=None):
        self.feature_cols = feature_cols
        self.scaler_X = CLeANNormalizer(n_features=len(feature_cols), eta=clean_eta,
                                         full_pipeline=True)
        self.scaler_y = CLeANNormalizer(n_features=3, eta=clean_eta,
                                         full_pipeline=False)
        self.is_fitted = False
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def partial_fit_scalers(self, df):
        self.scaler_y.update_statistics(df[['sog', 'cog_sin', 'cog_cos']].values)
        self.is_fitted = True

    def transform_X(self, df):
        """Pass-through: X è normalizzato dentro il modello, non qui."""
        return df.copy()

                                                                        
    # Cache degli array grezzi (non normalizzati, quindi sempre validi)
                                                                        

    def _cache_path(self, cache_key, seq_len, gap_threshold):
        filename = f"raw_{cache_key}_sl{seq_len}_gt{gap_threshold}.npz"
        return os.path.join(self.cache_dir, filename)

    def _load_pre_arrays(self, path):
        with np.load(path) as z:
            offsets = z['offsets']
            vals    = z['vals']
            targets = z['targets']
            coords  = z['coords']
            gaps    = z['gaps']
        segment_arrays = []
        for i in range(len(offsets) - 1):
            a, b = offsets[i], offsets[i+1]
            segment_arrays.append((vals[a:b], targets[a:b], coords[a:b], gaps[a:b]))
        return segment_arrays

    def _save_pre_arrays(self, path, segment_arrays):
        """Scrittura ATOMICA: scrive su un file temporaneo univoco per
        processo e poi lo rinomina. Senza questo, due run in parallelo che
        calcolano lo stesso giorno potrebbero far leggere a un terzo
        processo un file scritto a meta'. os.replace e' atomico su POSIX."""
        lengths = [len(p[0]) for p in segment_arrays]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        tmp = f"{path}.tmp{os.getpid()}"
        np.savez(
            tmp,
            offsets=offsets,
            vals=np.concatenate([p[0] for p in segment_arrays]),
            targets=np.concatenate([p[1] for p in segment_arrays]),
            coords=np.concatenate([p[2] for p in segment_arrays]),
            gaps=np.concatenate([p[3] for p in segment_arrays]),
        )
        os.replace(tmp + ".npz", path)

    def _get_pre_arrays(self, df, seq_len, gap_threshold, n_jobs, cache_key):
        use_cache = (self.cache_dir is not None and cache_key is not None)

        if use_cache:
            path = self._cache_path(cache_key, seq_len, gap_threshold)
            if os.path.exists(path):
                try:
                    return self._load_pre_arrays(path)
                except Exception:
                    pass  # cache corrotta: si ricalcola da zero

        df = df.sort_values(['mmsi', 'base_date_time']).copy()
        df['time_diff'] = df.groupby('mmsi')['base_date_time'].diff().dt.total_seconds() / 60.0
        df['segment_id'] = df.groupby('mmsi')['time_diff'].transform(
            lambda s: (s > gap_threshold).cumsum()
        )

        groups = [g.copy() for _, g in df.groupby(['mmsi', 'segment_id'], sort=True)]
        if len(groups) == 0:
            return []

        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_resample_one_group)(g, self.feature_cols, seq_len, gap_threshold)
            for g in groups
        )
        segment_arrays = [r for r in results if r is not None]

        if use_cache and len(segment_arrays) > 0:
            try:
                self._save_pre_arrays(path, segment_arrays)
            except Exception:
                pass  # se il disco è pieno si prosegue comunque senza cache

        return segment_arrays

                                                                        

    def _build_all_sequences(self, df, seq_len, gap_threshold, n_jobs, cache_key):
        """
        Logica comune a create_sequences() e create_sequences_raw(): calcola
        tutti gli array grezzi (X, y, posizioni, gap), SENZA normalizzare y.
        Le due varianti pubbliche differiscono solo nell'ultimo passo.
        """
        empty_result = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        if len(df) == 0:
            return empty_result

        if n_jobs is None:
            n_jobs = self.N_JOBS_DEFAULT
        segment_arrays = self._get_pre_arrays(df, seq_len, gap_threshold, n_jobs, cache_key)
        if len(segment_arrays) == 0:
            return empty_result

        blocks = [_build_windows(pre, seq_len, gap_threshold) for pre in segment_arrays]
        blocks = [b for b in blocks if b is not None]
        if len(blocks) == 0:
            return empty_result

        X_raw          = np.concatenate([b[0] for b in blocks]).astype(np.float32)
        y_raw          = np.concatenate([b[1] for b in blocks]).astype(np.float32)
        last_pos_real  = np.concatenate([b[2] for b in blocks])
        dest_real      = np.concatenate([b[3] for b in blocks])
        gaps_out       = np.concatenate([b[4] for b in blocks])

        return X_raw, y_raw, last_pos_real, dest_real, gaps_out

    def create_sequences(self, df, seq_len=30, gap_threshold=30, n_jobs=None,
                          cache_key=None):
        """
        Costruisce le sequenze del giorno. y viene SEMPRE normalizzato con
        lo scaler_y corrente; X resta grezzo (normalizzato nel modello).
        Uso: training e valutazione (loss/ADE), dove serve y in scala
        normalizzata per calcolare la loss contro l'output del modello.

        cache_key: identificativo del giorno (es. l'indice). Se valorizzato
        e cache_dir è impostata, la parte deterministica (resample +
        interpolazione) viene riusata invece di essere ricalcolata.
        """
        X_raw, y_raw, last_pos_real, dest_real, gaps_out = self._build_all_sequences(
            df, seq_len, gap_threshold, n_jobs, cache_key
        )
        if len(X_raw) == 0:
            return X_raw, y_raw, last_pos_real, dest_real, gaps_out

        # Normalizzazione SEMPRE fresca, mai cachata
        y_norm = self.scaler_y(y_raw).numpy().astype(np.float32)
        return X_raw, y_norm, last_pos_real, dest_real, gaps_out

    def create_sequences_raw(self, df, seq_len=30, gap_threshold=30, n_jobs=None,
                              cache_key=None):
        """
        Come create_sequences(), ma restituisce y in UNITÀ FISICHE (SOG in
        nodi, cog_sin/cos non normalizzati), senza mai passare da scaler_y.
        Usato dall'anomaly detection (attacks.py): gli attacchi sintetici
        alterano SOG/posizione in scala fisica reale (es. "SOG * 3", "offset
        di 2 gradi"), operazioni che non hanno senso su valori normalizzati
        in [0,1]. X è comunque grezzo in entrambe le varianti — qui cambia
        solo se y torna in scala fisica o normalizzata.

        Stessa cache di create_sequences() (condivide _get_pre_arrays):
        chiamare l'una o l'altra sullo stesso giorno non duplica il lavoro
        di preprocessing, cambia solo l'ultimo passo su y.
        """
        return self._build_all_sequences(df, seq_len, gap_threshold, n_jobs, cache_key)
