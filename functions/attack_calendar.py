"""
attack_calendar.py
===================
Definisce QUANDO ciascun attacco è presente nel flusso di addestramento,
sostituendo lo schema precedente (tutti e cinque gli attacchi sempre
presenti, in quantità identica ogni giorno) con uno a intermittenza,
pensato per rendere possibile misurare la dimenticanza: un attacco deve
poter sparire per settimane, mentre il modello continua ad aggiornarsi
su altro, perché ci sia qualcosa da dimenticare.

Il calendario è definito come una lista di blocchi (giorno_inizio,
giorno_fine_incluso, {attacco: frazione, ...}) — un blocco vuoto significa
quiete totale in quei giorni. Da questa lista derivano automaticamente:

  - il calendario giorno per giorno per l'ADDESTRAMENTO (esattamente
    quello che il blocco dice, giorno per giorno)
  - il calendario giorno per giorno per il TEST (un attacco, una volta
    comparso per la prima volta in addestramento, resta nel test per
    tutti i giorni successivi — anche quando non è più presente nel
    training)

Se num_days è più corto di 63 (per esempio in un test rapido con
--num_days), i blocchi oltre quel limite vengono semplicemente ignorati.
"""

import csv
import os

                                                                       
# IL CALENDARIO — (giorno_inizio, giorno_fine_incluso, {attacco: frazione})
                                                                       
                                                                   
# Quando due attacchi sono attivi insieme, le frazioni si sommano (es.
# due attacchi al 10% ciascuno = 20% totale attaccato quel giorno).
 
# Le assenze prolungate (es. speed_spoofing assente dal 34 al 55, mentre
# altri attacchi sono attivi nel mezzo) sono il punto centrale del
# disegno: è lì che il modello ha una vera occasione di dimenticare,
# perché continua ad aggiornarsi su altro mentre quell'attacco è assente.
 
# silent_drift riceve dieci giorni consecutivi alla prima comparsa
# (18-27): le rotte navali durano giorni, un attacco a deriva graduale
# che si esaurisse in una sola giornata sarebbe inverosimile.
                                                                       

CALENDARIO_BLOCCHI = [
    (0,  6,  {"speed_spoofing": 0.10}),
    (7,  8,  {}),
    (9,  14, {"teleport": 0.10}),
    (15, 17, {}),
    (18, 27, {"silent_drift": 0.10}),
    (28, 29, {}),
    (30, 33, {"speed_spoofing": 0.10, "teleport": 0.10}),
    (34, 36, {}),
    (37, 41, {"ghost_ship": 0.10}),
    (42, 43, {}),
    (44, 48, {"course_spoofing": 0.10}),
    (49, 50, {}),
    (51, 54, {"silent_drift": 0.10, "course_spoofing": 0.10}),
    (55, 55, {}),
    (56, 57, {"speed_spoofing": 0.10, "ghost_ship": 0.10}),
    (58, 62, {}),
]

NUM_GIORNI_PROGETTATO = 63


def build_calendario_training(num_days=NUM_GIORNI_PROGETTATO):
    """
    Restituisce {giorno: {attacco: frazione, ...}} per ogni giorno da 0 a
    num_days-1. Un giorno di quiete ha dizionario vuoto.
    """
    calendario = {g: {} for g in range(num_days)}
    for g_inizio, g_fine, attacchi in CALENDARIO_BLOCCHI:
        for g in range(g_inizio, min(g_fine, num_days - 1) + 1):
            calendario[g] = dict(attacchi)
    return calendario


def prima_apparizione(calendario_training):
    """
    {attacco: primo_giorno_in_cui_compare}, calcolato scorrendo il
    calendario di training in ordine cronologico.
    """
    prime = {}
    for g in sorted(calendario_training):
        for att in calendario_training[g]:
            if att not in prime:
                prime[att] = g
    return prime


def build_calendario_test(num_days=NUM_GIORNI_PROGETTATO, frazione_per_attacco=0.10):
    """
    Restituisce {giorno: {attacco: frazione, ...}} per il TEST: un
    attacco entra nel dizionario del giorno G se G >= giorno della sua
    prima comparsa in addestramento, e ci resta per tutti i giorni
    successivi — indipendentemente dal fatto che in quei giorni
    l'addestramento lo stia ancora mostrando o meno.
    """
    cal_training = build_calendario_training(num_days)
    prime = prima_apparizione(cal_training)

    calendario_test = {}
    for g in range(num_days):
        attivi = [att for att, g0 in prime.items() if g >= g0]
        calendario_test[g] = {att: frazione_per_attacco for att in attivi}
    return calendario_test


def salva_calendari_csv(results_dir, num_days=NUM_GIORNI_PROGETTATO):
    """
    Scrive due file CSV leggibili — uno per il training, uno per il
    test — con una riga per giorno e una colonna per attacco, in modo
    da avere un riferimento diretto di cosa è stato iniettato quando,
    utile sia per il controllo manuale sia per includerlo in tesi.
    """
    cal_training = build_calendario_training(num_days)
    cal_test = build_calendario_test(num_days)
    nomi_attacchi = sorted({a for d in cal_training.values() for a in d}
                            | {a for d in cal_test.values() for a in d})

    percorso_training = os.path.join(results_dir, "calendario_attacchi.csv")
    with open(percorso_training, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["giorno"] + nomi_attacchi + ["totale_attaccato"])
        for g in range(num_days):
            riga = cal_training[g]
            valori = [riga.get(a, 0.0) for a in nomi_attacchi]
            w.writerow([g] + valori + [sum(valori)])

    percorso_test = os.path.join(results_dir, "composizione_test.csv")
    with open(percorso_test, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["giorno"] + nomi_attacchi + ["totale_attaccato", "totale_pulito"])
        for g in range(num_days):
            riga = cal_test[g]
            valori = [riga.get(a, 0.0) for a in nomi_attacchi]
            totale = sum(valori)
            w.writerow([g] + valori + [totale, 1.0 - totale])

    return percorso_training, percorso_test
