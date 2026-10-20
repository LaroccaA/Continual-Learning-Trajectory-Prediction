import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# RESERVOIR BUFFER (Memoria Storica)
 
# Algoritmo invariato: Reservoir Sampling / Algorithm R (Vitter, 1985),
# lo stesso descritto in Buzzega et al. 2020 "Dark Experience for General
# Continual Learning" (framework Mammoth, Appendice D, Algorithm 3) e
# usato come riferimento standard in letteratura CL. Stessa sequenza di
# chiamate random.randint(), stessa logica di sostituzione, stesso
# identico contenuto finale del buffer con lo stesso seed — verificato
# con test di equivalenza automatico.
 
# Cambia SOLO la struttura di memorizzazione: tensore PyTorch
# pre-allocato invece di lista Python, come nei codebase di riferimento
# CL (es. GMvandeVen/continual-learning: self.buffer_y = torch.zeros(...)).
# Questo rende get_batch() un'operazione vettorizzata (indicizzazione
# tensoriale) invece di un loop Python con torch.stack su lista.

class ReservoirBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer_X = None  # tensore pre-allocato, creato al primo add()
        self.buffer_y = None
        self.num_stored = 0   # slot effettivamente occupati (<= size)
        self.seen = 0

    def _lazy_init(self, X_sample, y_sample):
        """Alloca il buffer alla prima chiamata, quando conosciamo le
        dimensioni reali di un campione (seq_len, n_features)."""
        if self.buffer_X is None:
            x_shape = X_sample.shape[1:]
            y_shape = y_sample.shape[1:]
            self.buffer_X = torch.zeros((self.size, *x_shape), dtype=X_sample.dtype)
            self.buffer_y = torch.zeros((self.size, *y_shape), dtype=y_sample.dtype)

    def add(self, X, y):
        X_cpu = X.detach().cpu()
        y_cpu = y.detach().cpu()
        self._lazy_init(X_cpu, y_cpu)

        for i in range(len(X_cpu)):
            self.seen += 1
            if self.num_stored < self.size:
                self.buffer_X[self.num_stored] = X_cpu[i]
                self.buffer_y[self.num_stored] = y_cpu[i]
                self.num_stored += 1
            else:
                j = random.randint(0, self.seen - 1)
                if j < self.size:
                    self.buffer_X[j] = X_cpu[i]
                    self.buffer_y[j] = y_cpu[i]

    def get_batch(self, batch_size):
        if self.num_stored == 0:
            return None

        n = min(batch_size, self.num_stored)
        indices = np.random.choice(self.num_stored, n, replace=False)
        idx_t = torch.as_tensor(indices, dtype=torch.long)

        batch_X = self.buffer_X[idx_t]
        batch_y = self.buffer_y[idx_t]
        return batch_X, batch_y


# CLASS-BALANCING RESERVOIR SAMPLING (CBRS)
 
# Chrysakis, A., & Moens, M.-F. (2020). "Online Continual Learning from
# Imbalanced Data." Proceedings of the 37th International Conference on
# Machine Learning (ICML 2020), PMLR 119.
# http://proceedings.mlr.press/v119/chrysakis20a/chrysakis20a.pdf
 
# Algorithm 1 del paper, applicato qui a DUE classi (pulito=0, attaccato=1
# — mai per singola tipologia di attacco, come da decisione presa).
# Sostituto drop-in di ReservoirBuffer: stessa interfaccia (add/get_batch/
# seen), quindi BR e A-GEM la usano SENZA alcuna modifica al loro codice
# di training — cambia solo COME il buffer si popola, non come lo si
                                                                    
# entrambi i metodi che usano un buffer.
 
# Proprietà dimostrata nel paper (Sezione 2.2): se una classe ha meno
# istanze del budget size/n_classi, CBRS garantisce che NESSUNA sua
# istanza venga mai scartata durante tutto lo stream — a differenza del
# reservoir puro, la cui composizione finale rispecchia semplicemente la
# composizione cumulativa dello stream (quindi si diluisce quanto più
# lo stream resta a lungo sbilanciato verso la classe maggioritaria,
# come nei lunghi giorni di quiete del calendario).
 
# NOTA: si applica solo dove esiste un concetto discreto di classe. Il
# buffer della regressione (buffer_reg) non ne ha uno (le traiettorie non
# sono mai "pulite vs attaccate" per il modulo di regressione) e resta
# quindi un ReservoirBuffer normale, invariato.

class ClassBalancingReservoirBuffer:
    def __init__(self, size, label_index):
        """
        size         : capienza totale del buffer (istanze, non giorni)
        label_index  : colonna di `y` che contiene l'etichetta intera
                       (qui: 0=pulito, 1=attaccato — combo_train[:, 8]
                       nel classificatore, vedi run_experiment_v2.py)
        """
        self.size = size
        self.label_index = label_index
        self.buffer_X = None
        self.buffer_y = None
        self.num_stored = 0
        self.seen = 0  # per compatibilità con "buffer.seen > 0" nel resto del codice

        self._slot_di_classe = {}   # {classe: [indici di slot occupati]}
        self._mc = {}               # istanze attualmente in memoria, per classe
        self._nc = {}               # istanze viste finora nello stream, per classe
        self._piene = set()         # classi che sono state (o sono) la più numerosa

    def _lazy_init(self, X_sample, y_sample):
        if self.buffer_X is None:
            x_shape = X_sample.shape[1:]
            y_shape = y_sample.shape[1:]
            self.buffer_X = torch.zeros((self.size, *x_shape), dtype=X_sample.dtype)
            self.buffer_y = torch.zeros((self.size, *y_shape), dtype=y_sample.dtype)

    def _classe_piu_numerosa(self):
        return max(self._slot_di_classe, key=lambda c: len(self._slot_di_classe[c]))

    def add(self, X, y):
        X_cpu = X.detach().cpu()
        y_cpu = y.detach().cpu()
        self._lazy_init(X_cpu, y_cpu)

        for i in range(len(X_cpu)):
            self.seen += 1
            c = int(y_cpu[i, self.label_index].item())
            self._nc[c] = self._nc.get(c, 0) + 1

                                                                         
            if self.num_stored < self.size:
                slot = self.num_stored
                self.buffer_X[slot] = X_cpu[i]
                self.buffer_y[slot] = y_cpu[i]
                self._slot_di_classe.setdefault(c, []).append(slot)
                self._mc[c] = self._mc.get(c, 0) + 1
                self.num_stored += 1
                continue

            # --- fase 2: memoria piena, Algorithm 1 del paper -----------
            if len(self._slot_di_classe) > 1:
                self._piene.add(self._classe_piu_numerosa())

            if c not in self._piene:
                # c non è (ancora) piena: sovrascrive un'istanza a caso
                # della classe attualmente più numerosa (righe 6-9)
                piu_numerosa = self._classe_piu_numerosa()
                slot = random.choice(self._slot_di_classe[piu_numerosa])
                self._slot_di_classe[piu_numerosa].remove(slot)
                self._mc[piu_numerosa] -= 1

                self.buffer_X[slot] = X_cpu[i]
                self.buffer_y[slot] = y_cpu[i]
                self._slot_di_classe.setdefault(c, []).append(slot)
                self._mc[c] = self._mc.get(c, 0) + 1
                self._piene.add(self._classe_piu_numerosa())
            else:
                # c è già piena: sostituzione con probabilità mc/nc,
                # SOLO tra istanze della stessa classe (righe 11-19)
                mc = self._mc.get(c, 0)
                nc = self._nc[c]
                if mc > 0 and random.random() <= mc / nc:
                    slot = random.choice(self._slot_di_classe[c])
                    self.buffer_X[slot] = X_cpu[i]
                    self.buffer_y[slot] = y_cpu[i]
                # altrimenti: istanza ignorata (riga 18 dell'algoritmo)

    def get_batch(self, batch_size):
        if self.num_stored == 0:
            return None
        n = min(batch_size, self.num_stored)
        indices = np.random.choice(self.num_stored, n, replace=False)
        idx_t = torch.as_tensor(indices, dtype=torch.long)
        return self.buffer_X[idx_t], self.buffer_y[idx_t]


# ELASTIC WEIGHT CONSOLIDATION (EWC)

class EWCHelper:
    """Implementazione fedele alla formulazione originale di Kirkpatrick
    et al. 2017 (\\cite{kirkpatrick2017overcoming}): la penalità di
    regolarizzazione si ACCUMULA su tutti i compiti (in questo caso,
    giorni) precedenti, non si sostituisce a ogni nuovo calcolo.

    NOTA SULLA RIFORMULAZIONE ALGEBRICA (nessuna approssimazione):
    la somma letterale del paper, generalizzata a N giorni, è
        sum_tau F_tau * (theta - theta*_tau)^2
    che espandendo il quadrato diventa
        theta^2 * S0  -  2*theta * S1  +  costante
    dove S0 = sum_tau F_tau e S1 = sum_tau (F_tau * theta*_tau).
    Questa è un'identità algebrica esatta, non un'approssimazione:
    mantenere due soli accumulatori (S0, S1), aggiornati con una
    somma a fine giornata, produce esattamente lo stesso valore e
    lo stesso gradiente che si otterrebbe sommando esplicitamente
    la penalità di ogni giorno passato preso singolarmente. Il
    vantaggio è che il costo per passo di addestramento resta
    costante, invece di crescere linearmente con i giorni trascorsi.

    Diversamente dalla versione precedente, l'oggetto va creato UNA
    SOLA VOLTA all'inizio dell'addestramento e mantenuto per l'intera
    durata del run — non ricreato ogni giorno.
    """

    def __init__(self, model, importance=1000):
        self.model = model
        self.importance = importance  # Forza del "lucchetto" (Lambda)
        self.S0 = None  # somma cumulativa delle Fisher di tutti i giorni consolidati
        self.S1 = None  # somma cumulativa di (Fisher * ancora) di tutti i giorni consolidati
        self.n_consolidamenti = 0

    def consolidate(self, dataset, device, batch_size=1024,
                     forward_fn=None, loss_fn=None):
        """Calcola la matrice di Fisher sui dati del giorno appena
        concluso e la somma agli accumulatori S0/S1, insieme allo
        snapshot corrente dei pesi come ancora di quel giorno. Da
        chiamare una volta al termine di ogni giorno di addestramento.

        NOTA (CLeAN): self.model.train() qui significa che, oltre a
        calcolare la Fisher Information, questo passaggio aggiorna
        ANCHE le statistiche EMA di model.clean_x con gli stessi dati,
        esattamente come nella versione precedente di questo metodo."""
        self.model.train()
        fisher_oggi = {n: torch.zeros_like(p, device=device) for n, p in self.model.named_parameters() if p.requires_grad}

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in loader:
            batch = [b.to(device) for b in batch]
            self.model.zero_grad()
            if forward_fn is not None:
                output = forward_fn(self.model, batch)
            else:
                output = self.model(batch[0], raw_input=True)
            if loss_fn is not None:
                loss = loss_fn(output, batch)
            else:
                loss = F.mse_loss(output, batch[1])
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_oggi[n] += (p.grad.detach() ** 2) / len(loader)

        ancora_oggi = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

        if self.S0 is None:
            # primo consolidamento in assoluto: inizializza gli accumulatori
            self.S0 = {n: fisher_oggi[n].clone() for n in fisher_oggi}
            self.S1 = {n: fisher_oggi[n] * ancora_oggi[n] for n in fisher_oggi}
        else:
            for n in fisher_oggi:
                self.S0[n] += fisher_oggi[n]
                self.S1[n] += fisher_oggi[n] * ancora_oggi[n]

        self.n_consolidamenti += 1

    def ewc_loss(self):
        """Ricostruisce, dagli accumulatori S0/S1, esattamente lo
        stesso valore che si otterrebbe sommando la penalità di ogni
        singolo giorno consolidato finora (vedi nota nella docstring
        della classe)."""
        if self.S0 is None:
            return 0.0
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.S0:
                loss += (self.S0[n] * p ** 2 - 2 * p * self.S1[n]).sum()
        return self.importance * loss

    def n_giorni_consolidati(self):
        """Utility diagnostica: quanti giorni sono stati consolidati finora."""
        return self.n_consolidamenti

# AVERAGED GRADIENT EPISODIC MEMORY (A-GEM)

def store_grad(model):
    """Estrae e appiattisce tutti i gradienti del modello in un unico vettore 1D."""
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.view(-1))
    if len(grads) == 0:
        return None
    return torch.cat(grads)


def overwrite_grad(model, new_grad_vector):
    """Sovrascrive i gradienti del modello scompattando il vettore 1D al loro posto."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            numel = p.numel()
            p.grad.copy_(new_grad_vector[offset:offset+numel].view_as(p.grad))
            offset += numel


def project_agem_gradient(model, ref_grad_vector):
    """
    Applica la proiezione ortogonale A-GEM sui gradienti attuali del modello.
    Modifica i gradienti in-place (direttamente nel modello).
    """
    current_grad_vector = store_grad(model)
    if current_grad_vector is None or ref_grad_vector is None:
        return

    dot_product = torch.dot(current_grad_vector, ref_grad_vector)

    if dot_product < 0:
        ref_mag = torch.dot(ref_grad_vector, ref_grad_vector)
        if ref_mag > 1e-8:
            current_grad_vector -= (dot_product / ref_mag) * ref_grad_vector
            overwrite_grad(model, current_grad_vector)
