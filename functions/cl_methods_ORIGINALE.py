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


# ELASTIC WEIGHT CONSOLIDATION (EWC)

class EWCHelper:
    def __init__(self, model, importance=1000):
        self.model = model
        self.importance = importance  # Forza del "lucchetto" (Lambda)
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {}

    def compute_fisher(self, dataset, device, batch_size=1024,
                        forward_fn=None, loss_fn=None):
        # NOTA (CLeAN): self.model.train() qui significa che, oltre a
        # calcolare la Fisher Information, questo passaggio aggiorna ANCHE
        # le statistiche EMA di model.clean_x con gli stessi dati.
         
        # SCELTA METODOLOGICA: batch_size allineato a quello di training
                                                                     
        # ~1.4M sequenze/giorno, batch_size=64 avrebbe richiesto ~22.000
        # iterazioni extra al giorno solo per la stima della Fisher
                                                                     
        # completo). Un batch più grande cambia leggermente il valore
        # numerico della stima (nota in letteratura EWC), ma non la
                                                                     
        # quadrato pesata sul numero di batch. Da documentare in tesi
        # come scelta esplicita, non un valore di default trascurabile.
         
                                                                      
        # multipli e loss diversa): forward_fn/loss_fn opzionali. Se
        # omessi, il comportamento è IDENTICO a prima — model(batch[0],
        # raw_input=True) e F.mse_loss(output, batch[1]) — verificato
        # con test di equivalenza, nessuna modifica al percorso già
        # usato dalla regressione.
        self.model.train()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in self.model.named_parameters() if p.requires_grad}

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in loader:
            batch = [b.to(device) for b in batch]
            self.model.zero_grad()
            if forward_fn is not None:
                output = forward_fn(self.model, batch)
            else:
                output = self.model(batch[0], raw_input=True)  # CLeAN: X qui è grezzo
            if loss_fn is not None:
                loss = loss_fn(output, batch)
            else:
                loss = F.mse_loss(output, batch[1])
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += (p.grad.detach() ** 2) / len(loader)

        self.fisher = fisher

    def ewc_loss(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n])**2).sum()
        return self.importance * loss


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
