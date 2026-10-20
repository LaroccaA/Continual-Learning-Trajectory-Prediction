"""
ORCHESTRATORE UNIFICATO — due modelli separati, addestrati in parallelo
giorno per giorno, ciascuno con la propria istanza di Continual Learning
(buffer/EWC/A-GEM indipendenti, pesi indipendenti).

Confermato dalla relatrice via email (11/08/2026):
  - due modelli distinti, pesi propri (non condivisi)
  - entrambi confrontati sulle stesse tre tecniche CL (BR, EWC, A-GEM)
  - split giornaliero 80% train / 20% test, per entrambi i modelli
  - valutazione ogni giorno anche su tutti i test set dei giorni precedenti
  - il classificatore riceve, oltre ai dati grezzi, un errore calcolato
    dal modello di regressione (Opzione B)

REGOLA SU QUALE STATO DELLA REGRESSIONE USARE (scelta esplicita, da
segnalare come tale): ogni giorno si aggiorna PRIMA la regressione, POI
si usa la regressione COSÌ AGGIORNATA per calcolare le feature di errore
del classificatore — sia per il training di quel giorno sia per la
valutazione di TUTTI i giorni (passati e corrente) a quel checkpoint.
Una sola regola, non due diverse per training/valutazione: più semplice
e più facile da giustificare in tesi.
"""

import os
import gc
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

import polars as pl

from functions.loader import AISStreamLoader
from functions.processing import ContinualPreprocessor
from functions.model import TrajectoryLSTM
from functions.evaluation import evaluate_day
from functions.classifier_model import AttackClassifierLSTM
from functions.classifier_dataset import build_daily_labeled_dataset
from functions.attack_calendar import (
    build_calendario_training, build_calendario_test, salva_calendari_csv
)
from functions.regression_features import compute_regression_error_features
from functions.cl_methods import (
    ReservoirBuffer, ClassBalancingReservoirBuffer, EWCHelper,
    store_grad, overwrite_grad, project_agem_gradient
)
from functions.cl_metrics_classifier import (
    binary_classification_metrics, compute_cl_metrics_classifier
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, 'data', 'processed')
RESULTS_DIR = os.path.join(HERE, 'results_v2')
MODELS_DIR = os.path.join(HERE, 'models_v2')
CHECKPOINTS_DIR = os.path.join(HERE, 'checkpoints_v2')
PREDICTIONS_DIR = os.path.join(HERE, 'previsioni_classificatore')
for d in (RESULTS_DIR, MODELS_DIR, CHECKPOINTS_DIR, PREDICTIONS_DIR):
    os.makedirs(d, exist_ok=True)

# Dimensione ASSOLUTA e fissa del test set del classificatore, uguale per
# ogni giorno indipendentemente da quanto è grande il pool disponibile.
# Necessaria perché con il calendario a intermittenza la composizione del
# test cambia nel tempo (dal 90/10 iniziale fino al 50/50 finale): se la
# dimensione totale variasse anch'essa, il numero di sequenze per singolo
# attacco non sarebbe più confrontabile da un giorno all'altro.
CLASSIFIER_TEST_SIZE = 20000

FEATURES = ['latitude', 'longitude', 'sog', 'cog_sin', 'cog_cos',
            'heading_sin', 'heading_cos']

# Dimensione del batch usato SOLO in valutazione (mai in training, dove
# batch_size è un iperparametro). Un valore fisso e generoso: la
# valutazione è comunque dentro torch.no_grad(), quindi consuma meno
# memoria del training a parità di batch. Riduce la chiamata "tutto il
# test set in un colpo solo" che causava OutOfMemoryError su un test set
# di ~280.000 sequenze.
EVAL_BATCH_SIZE = 8192


def build_regression_batch_tensor(X_np, y_np):
    return torch.FloatTensor(X_np), torch.FloatTensor(y_np)


def build_extra_tensor(split, err):
    """Le 5 feature 'dichiarate' + le 3 di errore = le 8 feature statiche
    date in ingresso al classificatore, oltre al contesto temporale."""
    declared = np.column_stack([
        split["declared_lat"], split["declared_lon"], split["declared_sog"],
        split["declared_cog_sin"], split["declared_cog_cos"],
    ]).astype(np.float32)
    return torch.FloatTensor(np.concatenate([declared, err], axis=1))


def run_experiment_v2(cfg: dict) -> dict:
    """
    cfg deve contenere, SEPARATI per i due modelli (Opzione B: ricerche
    indipendenti, confermata dall'utente):
        run_id, method  ('BR'|'EWC'|'AGEM' — applicato a ENTRAMBI i modelli,
                         è la tecnica CL, non un iperparametro del singolo
                         modello)
        reg_lr, reg_hidden_size, reg_num_layers, reg_buffer_size,
        reg_ewc_lambda, reg_batch_size
        clf_lr, clf_hidden_size, clf_num_layers, clf_buffer_size,
        clf_ewc_lambda, clf_batch_size
        epochs (condiviso, FISSO A 1, non cercato)
        seq_len (condiviso, strutturale — entrambi i modelli guardano la
                stessa finestra di contesto di 30 righe)
        num_days (opzionale, limita i giorni per test rapidi)
        eval_every (ogni quanti giorni salvare i checkpoint su disco —
                   la VALUTAZIONE resta giornaliera, indipendentemente
                   da questo parametro)
        n_jobs, save_checkpoints
    """
    run_id      = cfg["run_id"]
    method      = cfg["method"]

    reg_lr          = cfg["reg_lr"]
    reg_hidden_size = cfg["reg_hidden_size"]
    reg_num_layers  = cfg["reg_num_layers"]
    reg_buffer_size = cfg["reg_buffer_size"]
    reg_ewc_lambda  = cfg["reg_ewc_lambda"]
    reg_batch_size  = cfg["reg_batch_size"]

    clf_lr          = cfg["clf_lr"]
    clf_hidden_size = cfg["clf_hidden_size"]
    clf_num_layers  = cfg["clf_num_layers"]
    clf_buffer_size = cfg["clf_buffer_size"]
    clf_ewc_lambda  = cfg["clf_ewc_lambda"]
    clf_batch_size  = cfg["clf_batch_size"]

    # Disaccoppiate: la regressione e il classificatore sono due task
                                                                      
    # per cui debbano condividere lo stesso numero di epoche ottimale —
    # specialmente per attacchi sottili (es. silent_drift), dove il
    # classificatore potrebbe aver bisogno di più passaggi per imparare
    # un confine di decisione stabile.
    reg_epochs  = cfg.get("reg_epochs", cfg.get("epochs", 1))
    clf_epochs  = cfg.get("clf_epochs", cfg.get("epochs", 1))
    seq_len     = cfg["seq_len"]
    eval_every  = cfg.get("eval_every", 5)
    n_jobs      = cfg.get("n_jobs")
    save_checkpoints = cfg.get("save_checkpoints", False)
    skip_classifier_training = cfg.get("skip_classifier_training", False)
    skip_regression_eval = cfg.get("skip_regression_eval", False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)

                                                                        
    # dei tre flag diventa vero, quindi NEEDS_MEMORY resta False e ogni
    # blocco "if USE_EWC / if USE_AGEM / if NEEDS_MEMORY" più sotto viene
    # saltato. Il modello si addestra giorno per giorno con la sola
    # discesa del gradiente ordinaria, senza memoria, senza penalità,
    # senza vincolo — la linea di base "Sequential Fine-Tuning" priva di
    # qualunque meccanismo di Continual Learning.
    USE_BR   = (method == "BR")
    USE_EWC  = (method == "EWC")
    USE_AGEM = (method == "AGEM")
    NEEDS_MEMORY = USE_BR or USE_AGEM

                                                                        
    # Due modelli, pesi indipendenti, iperparametri architetturali propri
                                                                        
    regression_model = TrajectoryLSTM(
        input_size=len(FEATURES), hidden_size=reg_hidden_size,
        num_layers=reg_num_layers, use_clean=True
    ).to(device)

    classifier_model = AttackClassifierLSTM(
        input_size=len(FEATURES), hidden_size=clf_hidden_size,
        num_layers=clf_num_layers, use_clean=True
    ).to(device)

    regression_preprocessor = ContinualPreprocessor(
        FEATURES, cache_dir=os.path.join(HERE, 'cache_sequenze')
    )

    opt_reg = torch.optim.Adam(regression_model.parameters(), lr=reg_lr)
    opt_clf = torch.optim.Adam(classifier_model.parameters(), lr=clf_lr)
    criterion_reg = nn.MSELoss()
    criterion_clf = nn.BCEWithLogitsLoss()

    # Istanze CL SEPARATE per i due modelli — stato indipendente, e ora
    # anche dimensione del buffer indipendente.
     
                                                                        
    # singolo tipo di attacco (6 classi: pulito + 5 attacchi), non solo
    # pulito/attaccato — l'obiettivo del sistema è identificare non solo
    # CHE c'è un attacco, ma QUALE, quindi il buffer non deve lasciare
    # che un attacco più rappresentato ne penalizzi altri.
     
    # Mappatura stabile nome->intero, stesso ordine di ATTACK_REGISTRY in
    # attacks.py. 0 è riservato al pulito. Questa codifica esiste SOLO
    # per la logica interna di popolamento del buffer: non viene mai
    # letta come feature dal modello (ref_extra/ref_y più sotto restano
    # invariati, leggono sempre e solo le prime 9 colonne di combo).
    ATTACK_TYPE_TO_ID = {
        "clean": 0,
        "teleport": 1,
        "speed_spoofing": 2,
        "ghost_ship": 3,
        "silent_drift": 4,
        "course_spoofing": 5,
    }

    # BR e A-GEM usano buffer_clf tramite la stessa interfaccia
    # (add/get_batch/seen): nessun'altra modifica necessaria al loro
    # codice di training.
     
    # buffer_reg resta un ReservoirBuffer normale: la regressione non ha
    # un concetto discreto di classe da bilanciare.
    buffer_reg = ReservoirBuffer(size=reg_buffer_size) if NEEDS_MEMORY else None
    buffer_clf = (ClassBalancingReservoirBuffer(size=clf_buffer_size, label_index=9)
                  if NEEDS_MEMORY else None)
    ewc_reg = None
    ewc_clf = None
    if USE_EWC:
        # L'oggetto viene creato UNA SOLA VOLTA, fuori dal ciclo dei
        # giorni, così la sua "storia" si accumula per l'intera durata
        # del run invece di essere ricreata (e quindi persa) ogni giorno.
        ewc_reg = EWCHelper(regression_model, importance=reg_ewc_lambda)
        ewc_clf = EWCHelper(classifier_model, importance=clf_ewc_lambda)

    # VERIFICA A SCHERMO, non solo nel codice sorgente: qualunque persona
    # che legga il log di un run può confermare a colpo d'occhio quale
    # modalità di bilanciamento è stata usata, senza dover ispezionare il
    # file .py in esecuzione in quel momento.
    print("=" * 70)
    print(f"BUFFER CLASSIFICATORE: ClassBalancingReservoirBuffer a "
          f"{len(ATTACK_TYPE_TO_ID)} CLASSI (label_index=9)")
    print(f"  Classi: {ATTACK_TYPE_TO_ID}")
    print("=" * 70)



    loader = AISStreamLoader(DATA_PATH, max_days=cfg.get("num_days"))
    num_days = len(loader.unique_days)

    # Il calendario delle iniezioni — vedi functions/attack_calendar.py per
    # la logica completa. Costruito una sola volta, uguale per ogni run
    # (non dipende dal metodo CL né dagli iperparametri): il calendario_test
    # è CUMULATIVO, un attacco entra alla sua prima comparsa in training e
    # non esce più, anche quando il training non lo mostra più.
    calendario_training = build_calendario_training(num_days)
    calendario_test = build_calendario_test(num_days)
    if not skip_classifier_training:
        salva_calendari_csv(RESULTS_DIR, num_days)

    # checkpoint_every, separato da eval_every: controlla SOLO ogni quanti
    # giorni si salvano i PESI su disco (.pth) — la valutazione resta
    # comunque giornaliera per costruzione, indipendentemente da questo.
    # Costo del salvataggio quasi nullo (modelli piccoli, hidden=32):
    # con checkpoint_every=1 si ottengono 63 checkpoint invece di ~14,
    # utile per analisi future (magnitudo diverse, Recall@giorno vero)
    # senza dover rifare l'intero addestramento.
    checkpoint_every = cfg.get("checkpoint_every") or eval_every
    checkpoint_days = [d for d in range(num_days)
                       if d % checkpoint_every == 0 or d == num_days - 1]

    R_mse = np.full((num_days, num_days), np.nan)       # retention regressione (MSE normalizzato)
    R_ade = np.full((num_days, num_days), np.nan)        # retention regressione (ADE in metri, Dead Reckoning)
    R_f1  = np.full((num_days, num_days), np.nan)        # retention classificatore (F1, per BWT/FM/FWT)
    classifier_metrics_log = []   # TUTTE le metriche del classificatore, non solo F1

    # Test set del 20% di OGNI giorno passato, tenuti per la valutazione
    # cumulativa — chiave: indice del giorno
    regression_test_sets = {}   # day -> (X_test_clean, y_test_clean, ...)
    classifier_test_sets = {}   # day -> dict del test set etichettato

    log_path = os.path.join(RESULTS_DIR, f"{run_id}_{method}.log")
    logf = open(log_path, 'w')

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line)
        logf.write(line + "\n")
        logf.flush()

    log(f"Avvio run_id={run_id} method={method} giorni={num_days}")

    for current_day in range(num_days):
        df_day, current_date = loader.get_day(current_day)

        # Fit dello scaler_y della regressione sul giorno corrente
        # (invariato rispetto alla pipeline precedente)
        regression_preprocessor.partial_fit_scalers(df_day)
        df_scaled = regression_preprocessor.transform_X(df_day)

        X_raw, y_raw_norm, last_pos_all, dest_all, gaps_all =\
            regression_preprocessor.create_sequences(
                df_scaled, seq_len=seq_len, cache_key=current_day, n_jobs=n_jobs
            )
                                                                         
        # classificatore — riusa la stessa cache della regressione
        X_raw_2, y_raw_phys, lp_raw, dest_raw, gaps_raw =\
            regression_preprocessor.create_sequences_raw(
                df_day, seq_len=seq_len, cache_key=current_day, n_jobs=n_jobs
            )

        if len(X_raw) == 0:
            log(f"  Giorno {current_day} ({current_date}): nessuna sequenza, salto")
            continue

                                                                      
        # SPLIT INDIPENDENTE 80/20 per la REGRESSIONE (solo dati puliti)
                                                                      
        rng_day = np.random.default_rng(1000 + current_day)
        n = len(X_raw)
        idx = rng_day.permutation(n)
        n_train_reg = int(round(n * 0.8))
        idx_train_reg, idx_test_reg = idx[:n_train_reg], idx[n_train_reg:]

        X_train_reg = X_raw[idx_train_reg]
        y_train_reg = y_raw_norm[idx_train_reg]

                                                                    
        # valutare la SUA ritenzione (BWT/FM/FWT) — non per calcolare le
        # feature di errore del classificatore (che usa solo i pesi del
        # modello, non i suoi test set salvati). Quando la configurazione
        # di regressione è già fissata (traccia classificatore: i suoi
        # iperparametri non sono più in ricerca), rivalutarla ogni giorno
        # su tutti i precedenti è puro spreco — stessa identica curva
        # ricalcolata ad ogni run, mai usata per scegliere nulla.
        # Con 3 processi in parallelo, questi test set costano ~15GB/processo
        # (~45GB in totale) — causa diretta degli OOM (-9) osservati nella
        # traccia classificatore.
        if not skip_regression_eval:
            X_test_reg = X_raw[idx_test_reg]
            y_test_reg = y_raw_norm[idx_test_reg]
            last_pos_test_reg = last_pos_all[idx_test_reg]
            dest_test_reg = dest_all[idx_test_reg]
            gaps_test_reg = gaps_all[idx_test_reg]

            regression_test_sets[current_day] = {
                "X": X_test_reg, "y": y_test_reg,
                "last_pos": last_pos_test_reg, "dest": dest_test_reg,
                "gaps": gaps_test_reg,
            }

                                                                      
        # TRAINING REGRESSIONE — invariato nella logica, solo sull'80%
                                                                      
        regression_model.train()
        X_t, y_t = build_regression_batch_tensor(X_train_reg, y_train_reg)
        train_loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=reg_batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )

        for epoch in range(reg_epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                if USE_BR and buffer_reg is not None and buffer_reg.seen > 0:
                    ref = buffer_reg.get_batch(batch_size=batch_X.size(0))
                    if ref is not None:
                        ref_y = regression_preprocessor.scaler_y.minmax_normalize(ref[1]).to(device)
                        batch_X = torch.cat((batch_X, ref[0].to(device)), dim=0)
                        batch_y = torch.cat((batch_y, ref_y), dim=0)

                opt_reg.zero_grad()
                output = regression_model(batch_X, raw_input=True)
                loss = criterion_reg(output, batch_y)

                if USE_EWC and ewc_reg is not None:
                    loss = loss + ewc_reg.ewc_loss()

                loss.backward()

                if USE_AGEM and buffer_reg is not None and buffer_reg.seen > 0:
                    ref = buffer_reg.get_batch(batch_size=batch_X.size(0))
                    if ref is not None:
                        ref_y = regression_preprocessor.scaler_y.minmax_normalize(ref[1]).to(device)
                        cur_grad = store_grad(regression_model)
                        if cur_grad is not None:
                            regression_model.zero_grad()
                            criterion_reg(
                                regression_model(ref[0].to(device), raw_input=True), ref_y
                            ).backward()
                            ref_grad = store_grad(regression_model)
                            overwrite_grad(regression_model, cur_grad)
                            if ref_grad is not None:
                                project_agem_gradient(regression_model, ref_grad)

                opt_reg.step()

        if NEEDS_MEMORY:
            y_train_raw_for_buffer = regression_preprocessor.scaler_y.inverse_transform(y_t)
            buffer_reg.add(X_t, y_train_raw_for_buffer)

        if USE_EWC:
                                                                       
            # nella storia dell'oggetto, senza sostituire quella dei
            # giorni precedenti — fedele alla formulazione originale.
            ewc_reg.consolidate(TensorDataset(X_t, y_t), device)

                                                                      
        # DATASET ETICHETTATO PER IL CLASSIFICATORE (split 80/20 proprio,
        # attacchi iniettati su metà delle sequenze — vedi
        # classifier_dataset.py). Saltato interamente se
                                                                   
        # regressione): non serve costruire dati che non verranno usati.
                                                                      
        if not skip_classifier_training:
            piano_train_oggi = calendario_training.get(current_day, {})
            piano_test_oggi = calendario_test.get(current_day, {})
            clf_train, clf_test = build_daily_labeled_dataset(
                X_raw_2, y_raw_phys, lp_raw, dest_raw, gaps_raw,
                piano_train=piano_train_oggi, piano_test=piano_test_oggi,
                train_fraction=0.8, test_size=CLASSIFIER_TEST_SIZE,
                seed=2000 + current_day
            )
            classifier_test_sets[current_day] = clf_test
            if piano_train_oggi:
                log(f"  Giorno {current_day} — attacchi in addestramento: "
                    f"{piano_train_oggi}")
            else:
                log(f"  Giorno {current_day} — quiete (nessun attacco in addestramento)")

                                                                      
        # BLOCCO CLASSIFICATORE — saltato interamente se
                                                                      
        # degli iperparametri della regressione: il classificatore non
        # serve in quella fase e la sua assenza fa risparmiare tempo).
        # In tutti gli altri casi (ricerca del classificatore, run
        # finali) questo blocco gira sempre.
                                                                      
        if not skip_classifier_training:
            # Feature di errore dal modello di regressione APPENA
                                                                                
            regression_model.eval()
            with torch.no_grad():
                extra_train = compute_regression_error_features(
                    regression_model, regression_preprocessor,
                    clf_train["X"], clf_train["declared_lat"], clf_train["declared_lon"],
                    clf_train["declared_sog"], clf_train["declared_cog_sin"],
                    clf_train["declared_cog_cos"], clf_train["last_pos"], clf_train["gaps"],
                    device, batch_size=EVAL_BATCH_SIZE
                )

            extra_train_t = build_extra_tensor(clf_train, extra_train)
            X_clf_train_t = torch.FloatTensor(clf_train["X"])
            y_clf_train_t = torch.FloatTensor(clf_train["labels"]).unsqueeze(1)

            # TRAINING CLASSIFICATORE — stessa logica CL della regressione,
            # istanze SEPARATE di buffer/EWC
            classifier_model.train()
            clf_loader = DataLoader(
                TensorDataset(X_clf_train_t, extra_train_t, y_clf_train_t),
                batch_size=clf_batch_size, shuffle=True, num_workers=0, pin_memory=True
            )

            for epoch in range(clf_epochs):
                for batch_X, batch_extra, batch_y in clf_loader:
                    batch_X = batch_X.to(device)
                    batch_extra = batch_extra.to(device)
                    batch_y = batch_y.to(device)

                    if USE_BR and buffer_clf is not None and buffer_clf.seen > 0:
                        ref = buffer_clf.get_batch(batch_size=batch_X.size(0))
                        if ref is not None:
                            ref_X, ref_combo = ref[0].to(device), ref[1].to(device)
                            ref_extra = ref_combo[:, :8]
                            ref_y = ref_combo[:, 8:9]
                            batch_X = torch.cat((batch_X, ref_X), dim=0)
                            batch_extra = torch.cat((batch_extra, ref_extra), dim=0)
                            batch_y = torch.cat((batch_y, ref_y), dim=0)

                    opt_clf.zero_grad()
                    logit = classifier_model(batch_X, batch_extra, raw_input=True)
                    loss = criterion_clf(logit, batch_y)

                    if USE_EWC and ewc_clf is not None:
                        loss = loss + ewc_clf.ewc_loss()

                    loss.backward()

                    if USE_AGEM and buffer_clf is not None and buffer_clf.seen > 0:
                        ref = buffer_clf.get_batch(batch_size=batch_X.size(0))
                        if ref is not None:
                            ref_X, ref_combo = ref[0].to(device), ref[1].to(device)
                            ref_extra = ref_combo[:, :8]
                            ref_y = ref_combo[:, 8:9]
                            cur_grad = store_grad(classifier_model)
                            if cur_grad is not None:
                                classifier_model.zero_grad()
                                criterion_clf(
                                    classifier_model(ref_X, ref_extra, raw_input=True), ref_y
                                ).backward()
                                ref_grad = store_grad(classifier_model)
                                overwrite_grad(classifier_model, cur_grad)
                                if ref_grad is not None:
                                    project_agem_gradient(classifier_model, ref_grad)

                    opt_clf.step()

            if NEEDS_MEMORY:
                # nel buffer del classificatore salviamo insieme extra+label
                # (colonna 8 = extra, colonna 9 = label), così get_batch()
                # restituisce un'unica coppia (X, combo) coerente col resto
                combo_train = torch.cat([extra_train_t, y_clf_train_t], dim=1)
                attack_id_train = torch.tensor(
                    [ATTACK_TYPE_TO_ID[a] for a in clf_train["attack_type"]],
                    dtype=torch.float32
                ).unsqueeze(1)
                combo_train = torch.cat([combo_train, attack_id_train], dim=1)
                buffer_clf.add(X_clf_train_t, combo_train)

            if USE_EWC:
                ewc_clf.consolidate(
                    TensorDataset(X_clf_train_t, extra_train_t, y_clf_train_t), device,
                    forward_fn=lambda model, batch: model(batch[0], batch[1], raw_input=True),
                    loss_fn=lambda output, batch: nn.functional.binary_cross_entropy_with_logits(output, batch[2]),
                )

                                                                      
        # VALUTAZIONE — solo ai giorni-checkpoint, su TUTTI i test set
        # (proprio + tutti quelli passati), sia regressione sia classificatore
         
                                                                   
        # valutati A BATCH (EVAL_BATCH_SIZE), non in un'unica chiamata su
        # tutto il test set — con ~280.000 sequenze per test set, una
        # singola chiamata chiedeva alla GPU ~23GB e andava in
        # OutOfMemoryError. La regressione riusa evaluate_day() (già
        # esistente, già batchata, calcola MSE E ade in metri via Dead
        # Reckoning — non solo l'MSE normalizzato).
                                                                      
                                                                      
        # VALUTAZIONE — OGNI GIORNO (non solo ai checkpoint), su TUTTI i
        # test set dei giorni precedenti + quello del giorno corrente.
        # Richiesto esplicitamente: serve la curva di catastrophic
        # forgetting giorno per giorno, non solo ogni eval_every giorni.
         
        # Il SALVATAGGIO SU DISCO dei pesi (.pth) resta invece legato a
        # eval_every, per non scrivere 63 x 2 modelli x 2 file ogni run —
        # sono due cose distinte: la valutazione (in RAM, serve per le
        # metriche CL) e il checkpoint su disco (serve solo per riusare
        # il modello dopo, es. nell'anomaly detection).
                                                                      
        regression_model.eval()
        classifier_model.eval()

        eval_days = list(range(current_day + 1))   # TUTTI i giorni fin qui
        log(f"  Giorno {current_day} — valutazione su {len(eval_days)} giorni")

        # Accumulo delle PREVISIONI GREZZE per il checkpoint di questo
        # giorno — non solo le metriche aggregate. Da logit + etichetta +
        # attacco + magnitudo si ricava qualunque metrica a posteriori
        # (recall per attacco, per fascia di magnitudo, a soglia diversa,
        # specificità, media geometrica...) senza dover mai più
        # riaddestrare. Un file per checkpoint_day, scritto a fine ciclo.
        previsioni_oggi = {
            "eval_day": [], "logit": [], "etichetta": [],
            "attacco": [], "magnitudo": [],
        }

        for eval_day in eval_days:
            # --- regressione: MSE + ADE in metri (Dead Reckoning) ---
            # Con skip_regression_eval=True, regression_test_sets resta
                                                                        
            # None, questo blocco si salta da solo, nessun'altra modifica
            # necessaria qui.
            ts_r = regression_test_sets.get(eval_day)
            if ts_r is not None and len(ts_r["X"]) > 0:
                mse, ade_m = evaluate_day(
                    regression_model, regression_preprocessor,
                    ts_r["X"], ts_r["y"], ts_r["last_pos"], ts_r["dest"],
                    ts_r["gaps"], device=device, batch_size=EVAL_BATCH_SIZE
                )
                R_mse[current_day, eval_day] = mse
                R_ade[current_day, eval_day] = ade_m

                                                                    
            # Saltato del tutto se skip_classifier_training=True: non
            # avrebbe senso valutare un classificatore mai addestrato.
            if skip_classifier_training:
                continue
            ts_c = classifier_test_sets.get(eval_day)
            if ts_c is not None and len(ts_c["labels"]) > 0:
                n_c = len(ts_c["labels"])

                                                                          
                # test, non una per ogni blocco — prima si ricreava un
                # intero DataLoader per ogni fetta da 8192, moltiplicato
                # per ~2000 coppie giorno/giorno-valutato in un run a 63
                # giorni: decine di migliaia di DataLoader inutili,
                # overhead enorme rispetto al calcolo GPU effettivo.
                # predict_positions gestisce già il proprio batching
                # internamente in modo efficiente.
                with torch.no_grad():
                    err_feat = compute_regression_error_features(
                        regression_model, regression_preprocessor,
                        ts_c["X"], ts_c["declared_lat"], ts_c["declared_lon"],
                        ts_c["declared_sog"], ts_c["declared_cog_sin"],
                        ts_c["declared_cog_cos"], ts_c["last_pos"], ts_c["gaps"],
                        device, batch_size=EVAL_BATCH_SIZE
                    )
                    declared = np.column_stack([
                        ts_c["declared_lat"], ts_c["declared_lon"],
                        ts_c["declared_sog"], ts_c["declared_cog_sin"],
                        ts_c["declared_cog_cos"],
                    ]).astype(np.float32)
                    extra_full = np.concatenate([declared, err_feat], axis=1)

                    # Solo il forward del classificatore resta a batch (la
                    # sua memoria GPU scala col batch, va comunque limitata),
                    # ma senza richiamare la regressione ad ogni fetta.
                    all_logits = []
                    for start in range(0, n_c, EVAL_BATCH_SIZE):
                        end = min(start + EVAL_BATCH_SIZE, n_c)
                        sl = slice(start, end)
                        Xc_b = torch.FloatTensor(ts_c["X"][sl]).to(device)
                        extra_b = torch.FloatTensor(extra_full[sl]).to(device)
                        logit_b = classifier_model(Xc_b, extra_b, raw_input=True)
                        all_logits.append(logit_b.cpu().numpy().ravel())

                all_logits = np.concatenate(all_logits)
                m = binary_classification_metrics(ts_c["labels"], all_logits)
                R_f1[current_day, eval_day] = m["f1"]

                classifier_metrics_log.append({
                    "checkpoint_day": current_day,
                    "eval_day": eval_day,
                    **m,
                })

                # Previsioni grezze: una riga per sequenza valutata.
                n_righe = len(all_logits)
                previsioni_oggi["eval_day"].append(
                    np.full(n_righe, eval_day, dtype=np.int32))
                previsioni_oggi["logit"].append(all_logits.astype(np.float32))
                previsioni_oggi["etichetta"].append(
                    ts_c["labels"].astype(np.int8))
                previsioni_oggi["attacco"].append(ts_c["attack_type"])
                previsioni_oggi["magnitudo"].append(
                    ts_c["magnitude"].astype(np.float32))

        if not skip_classifier_training and len(previsioni_oggi["logit"]) > 0:
            tabella = pl.DataFrame({
                "checkpoint_day": np.full(
                    sum(len(a) for a in previsioni_oggi["logit"]),
                    current_day, dtype=np.int32),
                "eval_day": np.concatenate(previsioni_oggi["eval_day"]),
                "logit": np.concatenate(previsioni_oggi["logit"]),
                "etichetta": np.concatenate(previsioni_oggi["etichetta"]),
                "attacco": np.concatenate(previsioni_oggi["attacco"]),
                "magnitudo": np.concatenate(previsioni_oggi["magnitudo"]),
            })
            percorso_previsioni = os.path.join(
                PREDICTIONS_DIR, f"{run_id}_{method}_giorno{current_day:02d}.parquet")
            tabella.write_parquet(percorso_previsioni)
            del tabella
        del previsioni_oggi

        if save_checkpoints and current_day in checkpoint_days:
            torch.save(regression_model.state_dict(),
                       os.path.join(CHECKPOINTS_DIR, f"{run_id}_{method}_REG_day{current_day}.pth"))
            import joblib
            joblib.dump(regression_preprocessor,
                        os.path.join(CHECKPOINTS_DIR, f"{run_id}_{method}_REG_day{current_day}_prep.pkl"))
            if not skip_classifier_training:
                torch.save(classifier_model.state_dict(),
                           os.path.join(CHECKPOINTS_DIR, f"{run_id}_{method}_CLF_day{current_day}.pth"))

        # Salvataggio incrementale ad ogni giorno (non solo a fine run):
        # se il run si interrompe, non si perdono le valutazioni già fatte.
        if not skip_classifier_training:
            with open(os.path.join(RESULTS_DIR, f"{run_id}_{method}_classifier_metrics.json"), 'w') as f:
                json.dump(classifier_metrics_log, f, indent=2)

        del df_day, df_scaled, X_raw, y_raw_norm
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        if _HAS_PSUTIL:
            ram_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
            log(f"  RAM del processo dopo il giorno {current_day}: {ram_gb:.1f} GB "
                f"(stima teorica a fine run: ~41 GB su 104 disponibili)")

                                                                  
                                   
                                                                  
    torch.save(regression_model.state_dict(), os.path.join(MODELS_DIR, f"{run_id}_{method}_REG.pth"))

    risultato = {
        "run_id": run_id, "method": method,
        "reg_lr": reg_lr, "reg_hidden_size": reg_hidden_size,
        "reg_num_layers": reg_num_layers, "reg_buffer_size": reg_buffer_size,
        "reg_ewc_lambda": reg_ewc_lambda, "reg_batch_size": reg_batch_size,
        "reg_epochs": reg_epochs, "clf_epochs": clf_epochs, "seq_len": seq_len,
    }

    if not skip_regression_eval:
        metrics_reg = compute_metrics_regression(R_mse)
        ade_finale = float(np.nanmean(R_ade[-1, :])) if not np.all(np.isnan(R_ade[-1, :])) else None
        np.save(os.path.join(MODELS_DIR, f"{run_id}_{method}_R_mse.npy"), R_mse)
        np.save(os.path.join(MODELS_DIR, f"{run_id}_{method}_R_ade.npy"), R_ade)
        risultato.update({
            **{f"REG_{k}": v for k, v in metrics_reg.items()},
            "REG_ADE_finale_metri": ade_finale,
        })

    if not skip_classifier_training:
        metrics_clf = compute_cl_metrics_classifier(R_f1)
        torch.save(classifier_model.state_dict(), os.path.join(MODELS_DIR, f"{run_id}_{method}_CLF.pth"))
        np.save(os.path.join(MODELS_DIR, f"{run_id}_{method}_R_f1.npy"), R_f1)
        with open(os.path.join(RESULTS_DIR, f"{run_id}_{method}_classifier_metrics.json"), 'w') as f:
            json.dump(classifier_metrics_log, f, indent=2)

        risultato.update({
            "clf_lr": clf_lr, "clf_hidden_size": clf_hidden_size,
            "clf_num_layers": clf_num_layers, "clf_buffer_size": clf_buffer_size,
            "clf_ewc_lambda": clf_ewc_lambda, "clf_batch_size": clf_batch_size,
            **{f"CLF_{k}": v for k, v in metrics_clf.items()},
        })

    with open(os.path.join(RESULTS_DIR, f"{run_id}_{method}.json"), 'w') as f:
        json.dump(risultato, f, indent=2)

    # VERIFICA EMPIRICA FINALE (non solo dichiarativa): ispeziona il
    # contenuto REALE del buffer a fine run, non solo cosa il codice
    # dichiara di voler fare. Se qualcosa nella patch fosse comunque
    # sbagliato, questo lo renderebbe visibile qui, senza dover fidarsi
    # della sola stampa di avvio.
    if NEEDS_MEMORY and buffer_clf is not None and buffer_clf.num_stored > 0:
        ids_nel_buffer = buffer_clf.buffer_y[:buffer_clf.num_stored, 9].numpy().astype(int)
        nomi_id = {v: k for k, v in ATTACK_TYPE_TO_ID.items()}
        log("VERIFICA EMPIRICA composizione finale del buffer_clf:")
        for classe_id in sorted(set(ids_nel_buffer)):
            n = int((ids_nel_buffer == classe_id).sum())
            nome = nomi_id.get(classe_id, f"id_sconosciuto_{classe_id}")
            log(f"    {nome:16s}: {n:5d} slot ({100*n/buffer_clf.num_stored:.1f}%)")

    log(f"Run completata: {risultato}")
    logf.close()
    return risultato


def compute_metrics_regression(R_mse):
    """Stessa logica già usata in run_experiment.py per la regressione
    (BWT/FM/FWT su MSE, dove più basso è meglio)."""
    num_days = R_mse.shape[0]
    bwt_vals = [R_mse[-1, j] - R_mse[j, j] for j in range(num_days - 1)
                if not np.isnan(R_mse[-1, j]) and not np.isnan(R_mse[j, j])]
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0

    fm_vals = []
    for j in range(num_days - 1):
        storico = R_mse[j:, j]
        storico = storico[~np.isnan(storico)]
        if len(storico) and not np.isnan(R_mse[-1, j]):
            fm_vals.append(R_mse[-1, j] - np.min(storico))
    fm = float(np.mean(fm_vals)) if fm_vals else 0.0

    n_base = min(3, num_days)
    baseline_points = [R_mse[i, i] for i in range(n_base) if not np.isnan(R_mse[i, i])]
    baseline = float(np.mean(baseline_points)) if baseline_points else np.nan
    fwt_vals = [R_mse[j, j] - baseline for j in range(n_base, num_days)
                if not np.isnan(R_mse[j, j])]
    fwt = float(np.mean(fwt_vals)) if fwt_vals else 0.0

    return {
        "BWT": round(bwt, 6), "FM": round(fm, 6), "FWT": round(fwt, 6),
        "MSE_finale": round(float(np.nanmean(R_mse[-1, :])), 6),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--method", required=True, choices=["BR", "EWC", "AGEM", "NONE"])

                                                                       
                                                                     
    # (protocollo di addestramento cambiato: 80% del giorno, non 100%)
    p.add_argument("--reg_lr", type=float, default=0.0001)
    p.add_argument("--reg_hidden_size", type=int, default=32)
    p.add_argument("--reg_num_layers", type=int, default=2)
    p.add_argument("--reg_buffer_size", type=int, default=500)
    p.add_argument("--reg_batch_size", type=int, default=1024)
    p.add_argument("--reg_ewc_lambda", type=float, default=10)

    # Iperparametri CLASSIFICATORE — nessun valore "ottimale" pregresso,
    # questi default sono solo un punto di partenza ragionevole, DA CERCARE
    p.add_argument("--clf_lr", type=float, default=0.0001)
    p.add_argument("--clf_hidden_size", type=int, default=32)
    p.add_argument("--clf_num_layers", type=int, default=2)
    p.add_argument("--clf_buffer_size", type=int, default=500)
    p.add_argument("--clf_batch_size", type=int, default=1024)
    p.add_argument("--clf_ewc_lambda", type=float, default=10)

    p.add_argument("--reg_epochs", type=int, default=5,
                    help="epoche per la regressione — già trovato ottimale a 5")
    p.add_argument("--clf_epochs", type=int, default=5,
                    help="epoche per il classificatore — DA VERIFICARE "
                         "indipendentemente, mai cercato separatamente finora")
    p.add_argument("--seq_len", type=int, default=30)  # condiviso, strutturale
    p.add_argument("--num_days", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--checkpoint_every", type=int, default=None,
                    help="ogni quanti giorni salvare i PESI su disco (.pth). "
                         "Se omesso, usa lo stesso valore di --eval_every "
                         "(comportamento di prima). Impostare a 1 per avere "
                         "un checkpoint per ogni giorno — costo quasi nullo "
                         "con modelli piccoli, utile per analisi future "
                         "senza dover riaddestrare.")
    p.add_argument("--n_jobs", type=int, default=None)
    p.add_argument("--save_checkpoints", action="store_true")
    p.add_argument("--skip_classifier_training", action="store_true",
                    help="Disattiva interamente il classificatore (training, "
                         "valutazione, salvataggio) — usato SOLO durante la "
                         "ricerca degli iperparametri della sola regressione, "
                         "per risparmiare tempo.")
    p.add_argument("--skip_regression_eval", action="store_true",
                    help="La regressione continua ad addestrarsi normalmente "
                         "(serve per le feature del classificatore, Opzione B) "
                         "ma non viene più valutata sui test set cumulativi — "
                         "niente test set salvati in RAM, niente evaluate_day() "
                         "ripetuta ogni giorno. Usato quando la configurazione "
                         "di regressione è già fissata (traccia classificatore): "
                         "rivalutarla lì è puro spreco, stessa curva identica "
                         "ricalcolata ad ogni run, mai usata per scegliere nulla.")
    args = p.parse_args()


    cfg = vars(args)
    risultato = run_experiment_v2(cfg)
    print(json.dumps(risultato, indent=2))
