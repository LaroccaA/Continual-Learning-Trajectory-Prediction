"""
Costruisce il dataset ETICHETTATO per il classificatore, per un singolo
giorno: split 80/20, con attacchi sintetici iniettati secondo il PIANO DEL
GIORNO (quali attacchi, con quale frazione) — non più tutti e cinque
sempre presenti in quantità identica, ma secondo il calendario a
intermittenza definito in attack_calendar.py.

REVISIONE rispetto alla versione precedente (superata):
  - il TRAIN usa l'intera porzione 80% del giorno, con gli attacchi
    presenti secondo il piano di quel giorno (può essere anche vuoto,
    giorno di quiete: nessun addestramento del classificatore quel giorno)
  - il TEST ha ora una dimensione ASSOLUTA FISSA (di default 20.000
    sequenze), non più una percentuale del giorno — necessario perché
    altrimenti, con la composizione del test che cambia nel tempo (dal
    90/10 iniziale al 50/50 finale), il numero di sequenze per attacco
    varierebbe da un giorno all'altro rendendo le metriche non
    confrontabili
  - ogni sequenza attaccata riceve una magnitudo campionata
    casualmente (vedi attacks.py) e la magnitudo usata viene
    conservata nel campo "magnitude" per poter essere analizzata dopo

Confermato dalla relatrice via email (11/08/2026): i dati AIS non hanno
una colonna "attacco" — è lo studente a doverla costruire.
"""

import numpy as np

from functions.attacks import apply_attack


def _empty_split():
    return {
        "X": np.empty((0, 30, 7), dtype=np.float32),
        "declared_lat": np.empty(0), "declared_lon": np.empty(0),
        "declared_sog": np.empty(0), "declared_cog_sin": np.empty(0),
        "declared_cog_cos": np.empty(0),
        "last_pos": np.empty((0, 2)), "gaps": np.empty(0),
        "labels": np.empty(0, dtype=int),
        "attack_type": np.array([], dtype=object),
        "magnitude": np.empty(0),
    }


def _costruisci_parte_pulita(X_raw, y_raw, last_pos, dest_real, gaps, idx):
    return {
        "X": X_raw[idx].astype(np.float32),
        "declared_lat": dest_real[idx][:, 0].copy(),
        "declared_lon": dest_real[idx][:, 1].copy(),
        "declared_sog": y_raw[idx][:, 0].copy(),
        "declared_cog_sin": y_raw[idx][:, 1].copy(),
        "declared_cog_cos": y_raw[idx][:, 2].copy(),
        "last_pos": last_pos[idx].copy(),
        "gaps": gaps[idx].copy(),
        "labels": np.zeros(len(idx), dtype=int),
        "attack_type": np.array(["clean"] * len(idx), dtype=object),
        "magnitude": np.zeros(len(idx)),
    }


def _costruisci_parte_attaccata(X_raw, y_raw, last_pos, dest_real, gaps,
                                  idx, nome_attacco, rng):
    X_a = X_raw[idx].astype(np.float32)                            
    dest_in = dest_real[idx]
    y_in = y_raw[idx]
    lp_a = last_pos[idx]
    gaps_a = gaps[idx]

    dest_out, y_out, magnitudo = apply_attack(
        nome_attacco, lp_a, dest_in, y_in,
        seed=int(rng.integers(0, 2**31 - 1))
    )

    return {
        "X": X_a,
        "declared_lat": dest_out[:, 0].copy(),
        "declared_lon": dest_out[:, 1].copy(),
        "declared_sog": y_out[:, 0].copy(),
        "declared_cog_sin": y_out[:, 1].copy(),
        "declared_cog_cos": y_out[:, 2].copy(),
        "last_pos": lp_a.copy(),
        "gaps": gaps_a.copy(),
        "labels": np.ones(len(idx), dtype=int),
        "attack_type": np.array([nome_attacco] * len(idx), dtype=object),
        "magnitude": magnitudo,
    }


def _concatena_pezzi(pezzi):
    pezzi = [p for p in pezzi if len(p["labels"]) > 0]
    if not pezzi:
        return _empty_split()
    return {
        chiave: (np.concatenate([p[chiave] for p in pezzi], axis=0)
                 if pezzi[0][chiave].ndim > 0 else pezzi[0][chiave])
        for chiave in pezzi[0]
    }


def _build_split(X_raw, y_raw, last_pos, dest_real, gaps, idx_pool,
                   piano_attacchi, rng, dimensione_fissa=None):
    """
    Costruisce una porzione (train o test) a partire da un pool di indici
    disponibili e da un piano {attacco: frazione}.

    Se dimensione_fissa è None, usa TUTTO il pool disponibile (caso
    train: non si limita la quantità di dati di addestramento).
    Se dimensione_fissa è un intero, la porzione risultante avrà
    ESATTAMENTE quella dimensione totale (caso test: dimensione
    costante indipendentemente da quanto è grande il pool del 20%),
    campionando senza reinserimento dal pool.
    """
    pool_size = len(idx_pool)
    if pool_size == 0:
        return _empty_split()

    if dimensione_fissa is not None and pool_size < dimensione_fissa:
        import warnings
        warnings.warn(
            f"Pool disponibile ({pool_size}) più piccolo della dimensione "
            f"fissa richiesta ({dimensione_fissa}): il test di questo giorno "
            f"avrà {pool_size} sequenze invece di {dimensione_fissa}, "
            f"rompendo la comparabilità a dimensione costante con gli altri "
            f"giorni. Verificare il volume di traffico di questo giorno.",
            RuntimeWarning
        )

    n_totale = dimensione_fissa if dimensione_fissa is not None else pool_size
    n_totale = min(n_totale, pool_size)  # non si può chiedere più di quanto c'è

    perm_locale = rng.permutation(pool_size)[:n_totale]
    pool_selezionato = idx_pool[perm_locale]

    # alloca, dentro la porzione selezionata, quante sequenze vanno a
    # ciascun attacco attivo secondo il piano del giorno
    cursore = 0
    pezzi = []
    for nome_attacco, frazione in piano_attacchi.items():
        k = int(round(frazione * n_totale))
        k = min(k, n_totale - cursore)                                  
        if k <= 0:
            continue
        idx_attacco = pool_selezionato[cursore:cursore + k]
        pezzi.append(_costruisci_parte_attaccata(
            X_raw, y_raw, last_pos, dest_real, gaps, idx_attacco,
            nome_attacco, rng
        ))
        cursore += k

    idx_pulito = pool_selezionato[cursore:]
    pezzi.append(_costruisci_parte_pulita(
        X_raw, y_raw, last_pos, dest_real, gaps, idx_pulito
    ))

    return _concatena_pezzi(pezzi)


def build_daily_labeled_dataset(X_raw, y_raw, last_pos, dest_real, gaps,
                                  piano_train, piano_test,
                                  train_fraction=0.8, test_size=20000,
                                  seed=42):
    """
    Da un giorno di sequenze GREZZE, crea un dataset etichettato diviso
    in train e test, secondo i piani di attacco forniti per quel giorno
    (uno per il train, uno per il test — provengono da
    attack_calendar.build_calendario_training/test).

    Parametri:
        X_raw, y_raw, last_pos, dest_real, gaps : output di
            preprocessor.create_sequences_raw(df_giorno)
        piano_train : {attacco: frazione} per l'addestramento di QUESTO
                      giorno (può essere vuoto: giorno di quiete)
        piano_test  : {attacco: frazione} per il test di QUESTO giorno
                      (cumulativo: contiene tutti gli attacchi già
                      comparsi in addestramento fino ad ora)
        train_fraction : frazione del giorno riservata al pool di train
                         (il resto è il pool da cui si campiona il test)
        test_size   : dimensione ASSOLUTA e fissa del test, indipendente
                      dal giorno — permette di confrontare le metriche
                      nel tempo nonostante la composizione cambi
        seed        : per riproducibilità dello split e degli attacchi

    Restituisce due dizionari (train, test), ciascuno con:
        X, declared_lat, declared_lon, declared_sog, declared_cog_sin,
        declared_cog_cos, last_pos, gaps, labels, attack_type, magnitude
    """
    rng = np.random.default_rng(seed)
    N = len(X_raw)
    if N == 0:
        vuoto = _empty_split()
        return vuoto, vuoto

    idx = rng.permutation(N)
    n_train_pool = int(round(N * train_fraction))
    idx_train_pool, idx_test_pool = idx[:n_train_pool], idx[n_train_pool:]

    train = _build_split(X_raw, y_raw, last_pos, dest_real, gaps,
                          idx_train_pool, piano_train, rng,
                          dimensione_fissa=None)
    test = _build_split(X_raw, y_raw, last_pos, dest_real, gaps,
                         idx_test_pool, piano_test, rng,
                         dimensione_fissa=test_size)

    return train, test
