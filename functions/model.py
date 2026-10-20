import torch
import torch.nn as nn

from functions.clean_normalizer import CLeANNormalizer


# Fulcro della funzione init è, oltre alla creazione del modello,
# anche la suddivisione dei compiti in due stadi sequenziali.
# Il primo cioè self.lstm si occupa di andare ad estrarre le feature temporali,
# andando ad elaborare la finestra dei dati AIS della nave, modellandone l'inerzia e le dinamiche non lineari,
# comprimendo il tutto in uno spazio latente a 64 dimensioni.
# Il secondo stadio è self.fc proietta questo spazio latente a 64 dimensioni in uno spazio a 3 dimensioni,
# ovvero la velocità e le componenti vettoriali dell'angolo di rotta.
 
# MODIFICA (CLeAN / EMANet, repo ufficiale Marasco et al.): il normalizzatore
# CLeANNormalizer (clean_x) è integrato dentro forward(). Le sue statistiche
# si aggiornano automaticamente quando il modello è in model.train() e
# riceve un batch — esattamente come EMANet nel repo originale, nessuna
# chiamata manuale necessaria. In model.eval() (usato durante la valutazione
# CL sui giorni passati) le statistiche non vengono aggiornate.


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 use_clean=True, clean_eta=0.99):
        super(TrajectoryLSTM, self).__init__()

        self.use_clean = use_clean
        if use_clean:
            self.clean_x = CLeANNormalizer(n_features=input_size, eta=clean_eta,
                                            full_pipeline=True)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # La rete prevede 3 valori: SOG (Velocità), COG_sin e COG_cos (Direzione)
        # Questo è il nostro "Vettore di Spostamento"
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x, raw_input=False):
        """
        Parametri:
            x         : se use_clean=True e raw_input=True, x sono dati GREZZI
                        (N, seq_len, n_features). clean_x normalizza
                        internamente. Se use_clean=False, x deve già essere
                        pre-normalizzato esternamente.
            raw_input : flag esplicito per evitare normalizzazioni doppie.
        """
        if self.use_clean and raw_input:
            x = self.clean_x(x)

        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


                                                                               
# successivamente, attraverso l'operazione [:, -1, :], estraiamo esclusivamente
# l'ultimo hidden state della sequenza.
# Questo vettore viene infine passato allo strato lineare per la regressione dei parametri cinematici.
