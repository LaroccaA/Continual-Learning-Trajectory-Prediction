GRAFICI PER LA RELATRICE — INDICE
======================================================================

Struttura delle cartelle:

01_reservoir_puro/
    01_per_attacco_e_metrica/   20 grafici (5 attacchi x 4 metriche)
    04_F1_nel_tempo_con_calendario.png

02_cbrs_binario_buffer500/
    01_per_attacco_e_metrica/   20 grafici
    04_F1_nel_tempo_con_calendario.png

03_cbrs_binario_buffer20000/
    01_per_attacco_e_metrica/   20 grafici
    04_F1_nel_tempo_con_calendario.png
    (A-GEM assente: mai testato a questo buffer in modalita' binaria)

04_cbrs_6vie_buffer20000/
    01_per_attacco_e_metrica/   20 grafici
    04_F1_nel_tempo_con_calendario.png

05_F1_vs_dimensione_buffer/
    F1_vs_buffer_CBRS_binario.png   (7 valori: 500-20000)
    F1_vs_buffer_CBRS_6vie.png      (2 valori: 10000, 20000)

In ogni cartella, EWC riusa sempre la stessa curva (da CBRS binario,
buffer 500), perche' non dipende in alcun modo dal buffer o dal tipo
di bilanciamento -- dimostrato con valori identici in ogni stadio
della sperimentazione.

In ogni cartella compare anche una quarta linea grigia tratteggiata,
"Nessuna Strategia (Lower Bound)": il run lower_bound (method=NESSUNA),
senza alcun meccanismo di Continual Learning, riusato in tutte le
tecniche allo stesso modo di EWC, perché non dipende dal buffer o dal
tipo di bilanciamento.

Tutti i grafici di tipo "per attacco e metrica" e "F1 nel tempo" usano
la DIAGONALE delle rispettive matrici di ritenzione (checkpoint_day ==
eval_day) -- non metriche aggregate su piu' giorni.

Confronto a parita' di buffer (20000): cartelle 03 e 04.
