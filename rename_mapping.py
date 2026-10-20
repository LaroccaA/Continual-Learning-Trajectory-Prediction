#!/usr/bin/env python3
"""
Comprehensive renaming script to align all names with thesis terminology.
Run from repo root: python3 rename_mapping.py
"""

import os
import re
import shutil
from pathlib import Path

REPO = Path("/home/andrealarocca/Scrivania/Tesi_clean_repo")

# ============================================================
# MAPPING: vecchi nomi -> nuovi nomi (tesi)
# Ordine importante: prima le stringhe più lunghe per evitare collisioni
# ============================================================

# Directory rinomine
DIR_RENAME = {
    "risultati": "risultati",
    "figure": "figure",
    "pesi_finali": "pesi_finali",
}

# File rinomine (chiave = path relativo vecchio, valore = path relativo nuovo)
FILE_RENAME = {
    "run_experiment.py": "run_experiment.py",
    "README.md": "README.md",  # stesso nome ma aggiorniamo contenuto
}

# Sostituzioni nei contenuti (regex pattern -> replacement)
# Usiamo word boundaries dove possibile
CONTENT_REPLACEMENTS = [
    # ===== RUN ID PRINCIPALI =====
    (r'\bFINAL_NUOVO\b', 'reservoir_puro'),
    (r'\bFINAL_CBRS\b', 'cbrs_binario_b500'),
    (r'\bBUFFER_GRANDE_20000\b', 'cbrs_binario_b20000'),
    (r'\bVERIFICATO_6VIE_20000\b', 'cbrs_multiclasse_b20000'),
    (r'\bVERIFICATO_6VIE\b', 'cbrs_multiclasse_b10000'),
    (r'\bEWC_CORRETTO\b', 'ewc'),
    (r'\bLOWER_BOUND\b', 'lower_bound'),
    (r'\bUPPER_BOUND\b', 'upper_bound'),
    (r'\bFINAL_CBRS_ATTACCO\b', 'cbrs_attacco'),
    
    # ===== SWEEP BUFFER =====
    (r'\bSWEEP_BUFFER500\b', 'sweep_buffer_b500'),
    (r'\bSWEEP_BUFFER1000\b', 'sweep_buffer_b1000'),
    (r'\bSWEEP_BUFFER2000\b', 'sweep_buffer_b2000'),
    (r'\bSWEEP_BUFFER3000\b', 'sweep_buffer_b3000'),
    (r'\bSWEEP_BUFFER5000\b', 'sweep_buffer_b5000'),
    (r'\bSWEEP_BUFFER10000\b', 'sweep_buffer_b10000'),
    (r'\bSWEEP_BUFFER20000\b', 'sweep_buffer_b20000'),
    
    # ===== SWEEP LAMBDA =====
    (r'\bSWEEP_LAMBDA10\b', 'sweep_lambda_10'),
    (r'\bSWEEP_LAMBDA50\b', 'sweep_lambda_50'),
    (r'\bSWEEP_LAMBDA200\b', 'sweep_lambda_200'),
    (r'\bSWEEP_LAMBDA1000\b', 'sweep_lambda_1000'),
    
    # ===== BUFFER GRANDE (altri) =====
    (r'\bBUFFER_GRANDE_5000\b', 'cbrs_binario_b5000'),
    (r'\bBUFFER_GRANDE_10000\b', 'cbrs_binario_b10000'),
    
    # ===== verifica_6vie =====
    (r'\bVERIFICA_6VIE\b', 'verifica_6vie'),
    
    # ===== METODI CL =====
    (r'\bBR\b', 'BR'),
    (r'\bEWC\b', 'EWC'),
    (r'\bAGEM\b', 'AGEM'),
    (r'\bNONE\b', 'NESSUNA'),
    
    # ===== PATH DIRECTORY =====
    (r'risultati', 'risultati'),
    (r'figure', 'figure'),
    (r'pesi_finali', 'pesi_finali'),
    (r'run_experiment_v2\.py', 'run_experiment.py'),
    
    # ===== TITOLI GRAFICI / LEGEND =====
    (r'Reservoir Puro', 'Reservoir Puro'),
    (r'CBRS Binario, buffer 500', 'CBRS CBRS Binario, buffer 500'),
    (r'CBRS Binario, buffer 20000', 'CBRS CBRS Binario, buffer 20000'),
    (r'CBRS Multiclasse, buffer 20000', 'CBRS CBRS Multiclasse, buffer 20000'),
    (r'CBRS Multiclasse, buffer 10000', 'CBRS CBRS Multiclasse, buffer 10000'),
    (r'Nessuna strategia \(lower bound\)', 'Nessuna Strategia (Lower Bound)'),
    (r'Addestramento congiunto \(upper bound\)', 'Addestramento Congiunto (Upper Bound)'),
    (r'Buffer Replay \(BR\)', 'Buffer Replay (BR)'),
    (r'A-GEM', 'A-GEM'),
]

# File da NON modificare (es. file binary, .pth, .npy, .parquet, .png, .zip, .tar.gz)
SKIP_EXTENSIONS = {'.pth', '.npy', '.parquet', '.png', '.jpg', '.jpeg', '.pdf', '.zip', '.tar.gz', '.pkl', '.ipynb'}

def should_process(filepath):
    """Check if file should be processed for content replacement."""
    return filepath.suffix not in SKIP_EXTENSIONS and filepath.is_file()

def update_file_content(filepath):
    """Update content of a single file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content
        for pattern, repl in CONTENT_REPLACEMENTS:
            content = re.sub(pattern, repl, content)
        if content != original:
            filepath.write_text(content, encoding='utf-8')
            return True
    except Exception as e:
        print(f"  ERRORE lettura/scrittura {filepath}: {e}")
    return False

def main():
    print("=" * 60)
    print("RENOMAZIONE COMPLETA - ALLINEAMENTO TESI")
    print("=" * 60)
    
    # 1. RINOMINA DIRECTORY
    print("\n1. RINOMINA DIRECTORY")
    for old, new in DIR_RENAME.items():
        old_path = REPO / old
        new_path = REPO / new
        if old_path.exists():
            print(f"  {old} -> {new}")
            shutil.move(str(old_path), str(new_path))
        else:
            print(f"  {old} non trovato, skip")
    
    # 2. RINOMINA FILE PRINCIPALI
    print("\n2. RINOMINA FILE PRINCIPALI")
    for old, new in FILE_RENAME.items():
        old_path = REPO / old
        new_path = REPO / new
        if old_path.exists() and old != new:
            print(f"  {old} -> {new}")
            shutil.move(str(old_path), str(new_path))
    
    # 3. RINOMINA FILE IN risultati/ (ex risultati)
    print("\n3. RINOMINA FILE IN risultati/")
    risultati_dir = REPO / "risultati"
    if risultati_dir.exists():
        for f in risultati_dir.rglob("*"):
            if f.is_file():
                name = f.name
                new_name = name
                # Applica mapping ai nomi file
                for pattern, repl in CONTENT_REPLACEMENTS:
                    # Solo per i pattern che sono run_id completi
                    if pattern.startswith(r'\b') and pattern.endswith(r'\b'):
                        clean_pattern = pattern[2:-2]  # rimuovi \b
                        if clean_pattern in name:
                            new_name = new_name.replace(clean_pattern, repl)
                if new_name != name:
                    new_path = f.parent / new_name
                    print(f"  {f.relative_to(REPO)} -> {new_path.relative_to(REPO)}")
                    f.rename(new_path)
    
    # 4. RINOMINA FILE IN figure/ (ex figure)
    print("\n4. RINOMINA CARTELLE IN figure/")
    figure_dir = REPO / "figure"
    if figure_dir.exists():
        # Le sottocartelle 01_reservoir_puro, 02_cbrs_binario_buffer500, etc.
        # Le teniamo come sono (già allineate ai titoli tesi)
        pass
    
    # 5. AGGIORNA CONTENUTO DI TUTTI I FILE DI TESTO
    print("\n5. AGGIORNAMENTO CONTENUTO FILE")
    count = 0
    for filepath in REPO.rglob("*"):
        if should_process(filepath):
            if update_file_content(filepath):
                count += 1
                print(f"  Aggiornato: {filepath.relative_to(REPO)}")
    print(f"\nFile aggiornati: {count}")
    
    # 6. VERIFICA
    print("\n6. VERIFICA RAPIDA")
    # Controlla che non ci siano più vecchi nomi nei .py
    for py_file in REPO.rglob("*.py"):
        content = py_file.read_text(encoding='utf-8')
        for pattern, _ in CONTENT_REPLACEMENTS:
            clean = pattern[2:-2] if pattern.startswith(r'\b') else pattern
            if clean in content and clean not in ['BR', 'EWC', 'AGEM', 'NESSUNA']:
                print(f"  ATTENZIONE: '{clean}' ancora presente in {py_file.relative_to(REPO)}")
    
    print("\n" + "=" * 60)
    print("RENOMAZIONE COMPLETATA")
    print("=" * 60)

if __name__ == "__main__":
    main()