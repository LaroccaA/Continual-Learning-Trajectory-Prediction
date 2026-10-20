"""
attacks.py
==========
Generazione di attacchi sintetici di spoofing su traiettorie AIS reali.

I 5 attacchi implementati:
    - teleport         : salto di posizione
    - speed_spoofing   : SOG dichiarato incoerente con lo spostamento reale
    - ghost_ship        : la nave smette di muoversi ma continua a trasmettere
    - silent_drift      : deriva graduale e crescente della posizione dichiarata
    - course_spoofing   : virata dichiarata incompatibile con la cinematica

REVISIONE (magnitudo variabile): ogni attacco riceve ora una magnitudo
campionata CASUALMENTE PER CIASCUN ESEMPIO, entro un intervallo che va da
un valore sottile (difficile da distinguere da rumore/errore GPS ordinario)
a uno più marcato ma mai fisicamente impossibile — un attaccante reale non
usa sempre la stessa intensità, e un range troppo ampio (i valori originali
del progetto, es. teleport a 2 gradi fissi = 13.320 km/h implicita) rende
l'attacco banale da riconoscere fin dal primo giorno.

La magnitudo usata per ciascun esempio viene RESTITUITA insieme all'esempio
attaccato, per poter essere salvata e analizzata dopo (recall per fascia di
intensità), senza dover rieseguire nulla.

Ogni funzione riceve una sequenza pulita (X_raw, y_raw, last_pos, dest_real, gaps)
così come restituita da ContinualPreprocessor.create_sequences_raw(), e restituisce
la versione attaccata SOLO sul target (y_raw, dest_real) — la sequenza di contesto
(X_raw) rimane pulita, simulando uno scenario in cui la nave trasmette normalmente
e poi un singolo messaggio viene falsificato.
"""

import numpy as np


# Indici delle feature nel vettore X_raw / y_raw, coerenti con
# ContinualPreprocessor.feature_cols = ['latitude','longitude','sog',
#                                        'cog_sin','cog_cos','heading_sin','heading_cos']
IDX_LAT  = 0
IDX_LON  = 1
IDX_SOG  = 2
IDX_COG_SIN = 3
IDX_COG_COS = 4

# y_raw ha invece ordine ['sog','cog_sin','cog_cos']
IDX_Y_SOG     = 0
IDX_Y_COGSIN  = 1
IDX_Y_COGCOS  = 2


                                                                               
# INTERVALLI DI MAGNITUDO — un attaccante non usa sempre la stessa intensità
                                                                               
 
# "log_scale": True  -> campionamento uniforme sul LOGARITMO del valore, così
#   i valori sottili (che sono quelli davvero interessanti da studiare) non
#   restano una minoranza schiacciata contro il fondo dell'intervallo, come
#   accadrebbe con un campionamento uniforme lineare su un intervallo che
#   copre più ordini di grandezza.
# "log_scale": False -> campionamento uniforme lineare, adatto quando
#   l'intervallo copre meno di un ordine di grandezza.
 
# Le unità fisiche corrispondenti (per riferimento, non usate nel codice):
#   teleport       0.005-0.15 gradi  ->  550 m - 16 km di salto istantaneo
#   silent_drift   0.0002-0.0015 gradi/passo -> 22-166 m/min, ovvero
#                  440 m - 3,3 km accumulati su una sequenza di 30 minuti
#   ghost_ship     0.05-0.40  frazione di messaggi con SOG dichiarato
#                  incoerente con la posizione congelata
#   speed_spoofing 1.3-3.5   fattore moltiplicativo della velocità dichiarata
#   course_spoofing 25-150 gradi di virata dichiarata rispetto al reale
                                                                               

MAGNITUDE_RANGES = {
    "teleport":        {"low": 0.005,  "high": 0.15,  "log_scale": True},
    "silent_drift":    {"low": 0.0002, "high": 0.0015, "log_scale": True},
    "ghost_ship":      {"low": 0.05,   "high": 0.40,  "log_scale": False},
    "speed_spoofing":  {"low": 1.3,    "high": 3.5,   "log_scale": False},
    "course_spoofing": {"low": 25.0,   "high": 150.0, "log_scale": False},
}


def sample_magnitudes(attack_name, n, rng):
    """
    Campiona n valori di magnitudo per l'attacco richiesto, entro il suo
    intervallo, con la scala (lineare o logaritmica) definita sopra.
    """
    spec = MAGNITUDE_RANGES[attack_name]
    if spec["log_scale"]:
        log_low, log_high = np.log(spec["low"]), np.log(spec["high"])
        return np.exp(rng.uniform(log_low, log_high, size=n))
    return rng.uniform(spec["low"], spec["high"], size=n)


def _resolve_magnitude(attack_name, n, magnitude, rng):
    """
    Punto unico da cui ogni funzione di attacco ottiene le magnitudo da
    applicare: se `magnitude` è None, le campiona; se è uno scalare, lo
    replica per tutti gli n esempi (utile per test manuali o retrocompatibilità);
    se è già un array, lo usa così com'è (utile per riprodurre esattamente
    una condizione già vista, per esempio in fase di analisi).
    """
    if magnitude is None:
        return sample_magnitudes(attack_name, n, rng)
    if np.isscalar(magnitude):
        return np.full(n, float(magnitude))
    magnitude = np.asarray(magnitude, dtype=float)
    assert len(magnitude) == n, (
        f"magnitude ha lunghezza {len(magnitude)}, attesi {n} esempi"
    )
    return magnitude


def teleport_attack(dest_real, y_raw, magnitude=None, seed=None):
    """
    Sposta la posizione dichiarata di un offset in direzione casuale,
    con ampiezza (in gradi) pari alla magnitudo di ciascun esempio.

    Parametri:
        dest_real : (N, 2) array di lat/lon reali
        y_raw     : (N, 3) target sog/cog_sin/cog_cos (non modificato)
        magnitude : None (campiona), scalare (fisso per tutti), o
                    array (N,) di ampiezze in gradi già decise

    Restituisce:
        dest_attacked : (N, 2) posizioni con teleport applicato
        y_attacked    : (N, 3) invariato (l'attacco è solo posizionale)
        magnitude_usata : (N,) i valori di magnitudo effettivamente usati
    """
    rng = np.random.default_rng(seed)
    dest_attacked = dest_real.copy()
    N = len(dest_real)
    magnitude_usata = _resolve_magnitude("teleport", N, magnitude, rng)

    angle = rng.uniform(0, 2*np.pi, size=N)
    dest_attacked[:, 0] += magnitude_usata * np.cos(angle)       
    dest_attacked[:, 1] += magnitude_usata * np.sin(angle)       

    dest_attacked[:, 0] = np.clip(dest_attacked[:, 0], -90.0, 90.0)
    dest_attacked[:, 1] = np.clip(dest_attacked[:, 1], -180.0, 180.0)

    return dest_attacked, y_raw.copy(), magnitude_usata


def speed_spoofing_attack(dest_real, y_raw, magnitude=None, invert_fraction=0.3, seed=None):
    """
    Altera il SOG dichiarato di un fattore moltiplicativo, lasciando
    la posizione fisica plausibile ma incoerente con la velocità dichiarata.

    La magnitudo campionata è il fattore moltiplicativo prima dell'eventuale
    inversione (che simula un rallentamento anziché un'accelerazione) — è
    il fattore, non la direzione, a costituire l'intensità dell'attacco.

    Parametri:
        dest_real       : (N, 2) posizioni reali (non modificate)
        y_raw           : (N, 3) target sog/cog_sin/cog_cos
        magnitude       : None (campiona), scalare, o array (N,) di fattori
        invert_fraction : frazione di casi in cui il fattore viene invertito
                          (1/fattore), per simulare un rallentamento anziché
                          un'accelerazione — è la direzione dell'inganno,
                          non la sua intensità

    Restituisce:
        dest_attacked   : invariato
        y_attacked      : SOG alterato
        magnitude_usata : (N,) il fattore PRIMA dell'eventuale inversione
                          (l'intensità dell'attacco, indipendente dalla
                          direzione)
    """
    rng = np.random.default_rng(seed)
    y_attacked = y_raw.copy()
    N = len(y_raw)
    magnitude_usata = _resolve_magnitude("speed_spoofing", N, magnitude, rng)

    factors = magnitude_usata.copy()
    invert_mask = rng.random(N) < invert_fraction
    factors[invert_mask] = 1.0 / factors[invert_mask]

    y_attacked[:, IDX_Y_SOG] = y_raw[:, IDX_Y_SOG] * factors
    y_attacked[:, IDX_Y_SOG] = np.clip(y_attacked[:, IDX_Y_SOG], 0.1, 99.8)

    return dest_real.copy(), y_attacked, magnitude_usata


def ghost_ship_attack(last_pos, dest_real, y_raw, magnitude=None, seed=None):
    """
    La nave smette fisicamente di muoversi (resta sull'ultima posizione nota)
    ma continua a trasmettere messaggi AIS regolari. La magnitudo è la
    frazione di casi in cui il SOG dichiarato resta incoerente con la
    fermata (nave ferma ma dice di muoversi): più alta, più l'attacco è
    riconoscibile; più bassa, più è silenzioso — il resto dei casi dichiara
    correttamente SOG≈0, coerente con la posizione congelata, e in quei
    casi non c'è nessuna bugia esplicita nel singolo messaggio, solo nella
    storia (la posizione che non cambia più).

    Parametri:
        last_pos  : (N, 2) ultima posizione nota prima del target
        dest_real : (N, 2) posizione reale successiva
        y_raw     : (N, 3) target sog/cog_sin/cog_cos
        magnitude : None (campiona), scalare, o array (N,) di frazioni [0,1]

    Restituisce:
        dest_attacked   : posizione congelata su last_pos
        y_attacked      : SOG forzato vicino a 0, cog invariato
        magnitude_usata : (N,) le frazioni effettivamente usate
    """
    dest_attacked = last_pos.copy()
    y_attacked = y_raw.copy()

    rng = np.random.default_rng(seed)
    N = len(y_raw)
    magnitude_usata = _resolve_magnitude("ghost_ship", N, magnitude, rng)

    fake_active = rng.random(N) < magnitude_usata
    y_attacked[:, IDX_Y_SOG] = np.where(
        fake_active,
        y_raw[:, IDX_Y_SOG],
        rng.uniform(0.0, 0.5, size=N)
    )

    return dest_attacked, y_attacked, magnitude_usata


def silent_drift_attack(dest_real, y_raw, magnitude=None, clean_prefix=10, seed=None):
    """
    Deriva graduale: i primi `clean_prefix` punti della sequenza sono puliti,
    poi inizia un offset crescente linearmente nel tempo, con incremento per
    passo pari alla magnitudo di ciascun esempio.

    Parametri:
        dest_real     : (N, 2) posizioni reali, N = passi temporali in sequenza
        y_raw         : (N, 3) target
        magnitude     : None (campiona), scalare, o array (N,) di gradi/passo
        clean_prefix  : numero di passi iniziali senza drift

    Restituisce:
        dest_attacked   : posizioni con drift cumulativo crescente
        y_attacked      : invariato (il drift è solo posizionale)
        magnitude_usata : (N,) l'incremento per passo effettivamente usato
    """
    rng = np.random.default_rng(seed)
    dest_attacked = dest_real.copy()
    N = len(dest_real)
    magnitude_usata = _resolve_magnitude("silent_drift", N, magnitude, rng)

    direction = rng.uniform(0, 2*np.pi)  # direzione di drift fissa per questa traiettoria
    steps = np.maximum(0, np.arange(N) - clean_prefix)
    cumulative_offset = steps * magnitude_usata

    dest_attacked[:, 0] += cumulative_offset * np.cos(direction)
    dest_attacked[:, 1] += cumulative_offset * np.sin(direction)

    dest_attacked[:, 0] = np.clip(dest_attacked[:, 0], -90.0, 90.0)
    dest_attacked[:, 1] = np.clip(dest_attacked[:, 1], -180.0, 180.0)

    return dest_attacked, y_raw.copy(), magnitude_usata


def course_spoofing_attack(dest_real, y_raw, magnitude=None, seed=None):
    """
    Altera il COG dichiarato con una virata la cui ampiezza (in gradi) è la
    magnitudo di ciascun esempio, lasciando posizione e SOG coerenti.

    Riferimento in letteratura: Pohontu, Vertan, Ciocioi & Popa (RRIA)
    documentano proprio "sudden, multiple changes ... course over ground
    (COG)" come pattern di una "maneuvering spoofed vessel", con
    precedenti reali documentati nel Mar Nero nel 2021 (Thomas & Chiego,
    2022) — non è una minaccia inventata per lo studio.

    Parametri:
        dest_real : (N, 2) posizioni reali (non modificate)
        y_raw     : (N, 3) target sog/cog_sin/cog_cos
        magnitude : None (campiona), scalare, o array (N,) di gradi di virata

    Restituisce:
        dest_attacked   : invariato
        y_attacked      : cog_sin/cog_cos ruotati della virata
        magnitude_usata : (N,) i gradi di virata effettivamente usati
    """
    rng = np.random.default_rng(seed)
    y_attacked = y_raw.copy()
    N = len(y_raw)
    magnitude_usata = _resolve_magnitude("course_spoofing", N, magnitude, rng)

    true_cog_rad = np.arctan2(y_raw[:, IDX_Y_COGSIN], y_raw[:, IDX_Y_COGCOS])

    sign = rng.choice([-1, 1], size=N)
    flipped_cog_rad = true_cog_rad + np.radians(magnitude_usata * sign)

    y_attacked[:, IDX_Y_COGSIN] = np.sin(flipped_cog_rad)
    y_attacked[:, IDX_Y_COGCOS] = np.cos(flipped_cog_rad)

    return dest_real.copy(), y_attacked, magnitude_usata


                                                                               
# REGISTRY — utility per applicare un attacco per nome
                                                                               

ATTACK_REGISTRY = {
    "teleport":        teleport_attack,
    "speed_spoofing":  speed_spoofing_attack,
    "ghost_ship":      ghost_ship_attack,
    "silent_drift":    silent_drift_attack,
    "course_spoofing": course_spoofing_attack,
}


def apply_attack(attack_name, last_pos, dest_real, y_raw, magnitude=None, seed=None):
    """
    Applica l'attacco richiesto per nome. Gestisce le firme diverse
    delle singole funzioni (ghost_ship richiede anche last_pos).

    Restituisce SEMPRE tre valori:
        dest_attacked, y_attacked, magnitude_usata
    """
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Attacco sconosciuto: {attack_name}. "
                          f"Disponibili: {list(ATTACK_REGISTRY.keys())}")

    fn = ATTACK_REGISTRY[attack_name]

    if attack_name == "ghost_ship":
        return fn(last_pos, dest_real, y_raw, magnitude=magnitude, seed=seed)
    else:
        return fn(dest_real, y_raw, magnitude=magnitude, seed=seed)
