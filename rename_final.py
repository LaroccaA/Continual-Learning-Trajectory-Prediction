#!/usr/bin/env python3
"""
Final cleanup: rename analisi_* directories, update CSV references, fix remaining docs.
"""

import os
import re
import shutil
from pathlib import Path

REPO = Path("/home/andrealarocca/Scrivania/Tesi_clean_repo")

# Rinominare directory analisi_*
ANALISI_RENAME = {
    "analisi_FINAL_NUOVO": "analisi_reservoir_puro",
    "analisi_FINAL_CBRS": "analisi_cbrs_binario_b500",
    "analisi_FINAL_CBRS_ATTACCO": "analisi_cbrs_attacco",
    "analisi_BUFFER_GRANDE_20000": "analisi_cbrs_binario_b20000",
    "analisi_EWC_CORRETTO": "analisi_ewc",
    "analisi_LOWER_BOUND": "analisi_lower_bound",
    "analisi_VERIFICATO_6VIE": "analisi_cbrs_multiclasse_b10000",
    "analisi_VERIFICATO_6VIE_20000": "analisi_cbrs_multiclasse_b20000",
}

# Sostituzioni per CSV e file vari
CSV_REPLACEMENTS = {
    "analisi_FINAL_NUOVO": "analisi_reservoir_puro",
    "analisi_FINAL_CBRS": "analisi_cbrs_binario_b500",
    "analisi_FINAL_CBRS_ATTACCO": "analisi_cbrs_attacco",
    "analisi_BUFFER_GRANDE_20000": "analisi_cbrs_binario_b20000",
    "analisi_EWC_CORRETTO": "analisi_ewc",
    "analisi_LOWER_BOUND": "analisi_lower_bound",
    "analisi_VERIFICATO_6VIE": "analisi_cbrs_multiclasse_b10000",
    "analisi_VERIFICATO_6VIE_20000": "analisi_cbrs_multiclasse_b20000",
    "FINAL_NUOVO": "reservoir_puro",
    "FINAL_CBRS": "cbrs_binario_b500",
    "FINAL_CBRS_ATTACCO": "cbrs_attacco",
    "BUFFER_GRANDE_20000": "cbrs_binario_b20000",
    "EWC_CORRETTO": "ewc",
    "LOWER_BOUND": "lower_bound",
    "VERIFICATO_6VIE": "cbrs_multiclasse_b10000",
    "VERIFICATO_6VIE_20000": "cbrs_multiclasse_b20000",
}

# Doc replacements finali
DOC_FINAL_REPLACEMENTS = {
    # RESULTS_PROVENANCE.md
    r'`analisi_FINAL_NUOVO/': '`analisi_reservoir_puro/',
    r'`analisi_FINAL_CBRS/': '`analisi_cbrs_binario_b500/',
    r'`analisi_FINAL_CBRS_ATTACCO/': '`analisi_cbrs_attacco/',
    r'`analisi_BUFFER_GRANDE_20000/': '`analisi_cbrs_binario_b20000/',
    r'`analisi_EWC_CORRETTO/': '`analisi_ewc/',
    r'`analisi_LOWER_BOUND/': '`analisi_lower_bound/',
    r'`analisi_VERIFICATO_6VIE/': '`analisi_cbrs_multiclasse_b10000/',
    r'`analisi_VERIFICATO_6VIE_20000/': '`analisi_cbrs_multiclasse_b20000/',
    
    # File references in RESULTS_PROVENANCE.md
    r'`FINAL_NUOVO_\*_classifier_metrics\.json`': '`reservoir_puro_*_classifier_metrics.json`',
    r'`FINAL_CBRS_\*_classifier_metrics\.json`': '`cbrs_binario_b500_*_classifier_metrics.json`',
    r'`BUFFER_GRANDE_20000_BR_classifier_metrics\.json`': '`cbrs_binario_b20000_BR_classifier_metrics.json`',
    r'`VERIFICATO_6VIE_20000_\*_classifier_metrics\.json`': '`cbrs_multiclasse_b20000_*_classifier_metrics.json`',
    r'`SWEEP_BUFFER\*_BR\.json`': '`sweep_buffer_b*_BR.json`',
    r'`SWEEP_BUFFER\*_AGEM\.json`': '`sweep_buffer_b*_AGEM.json`',
    
    # DATA.md
    r'`risultati/VERIFICATO_6VIE_20000_BR\.json`': '`risultati/cbrs_multiclasse_b20000_BR.json`',
    r'`risultati/VERIFICATO_6VIE_20000_AGEM\.json`': '`risultati/cbrs_multiclasse_b20000_AGEM.json`',
    r'`risultati/VERIFICATO_6VIE_20000_BR\.json`': '`risultati/cbrs_multiclasse_b20000_BR.json`',
    
    # README.md - checkpoints
    r'VERIFICATO_6VIE_20000_\*': 'cbrs_multiclasse_b20000_*',
    r'EWC_CORRETTO_\*': 'ewc_*',
    r'LOWER_BOUND_\*': 'lower_bound_*',
    r'BUFFER_GRANDE_20000_\*': 'cbrs_binario_b20000_*',
    r'FINAL_CBRS_\*': 'cbrs_binario_b500_*',
    r'FINAL_NUOVO_\*': 'reservoir_puro_*',
    r'VERIFICATO_6VIE_\*': 'cbrs_multiclasse_b10000_*',
    r'UPPER_BOUND_CLF\.pth': 'upper_bound_CLF.pth',
}

def rename_analisi_dirs():
    """Rename analisi_* directories in risultati/"""
    print("Rinomina directory analisi_* in risultati/")
    risultati_dir = REPO / "risultati"
    for old, new in ANALISI_RENAME.items():
        old_path = risultati_dir / old
        new_path = risultati_dir / new
        if old_path.exists():
            print(f"  {old} -> {new}")
            shutil.move(str(old_path), str(new_path))
        else:
            print(f"  MANCANTE: {old}")

def update_csv_files():
    """Update references in CSV files"""
    print("\nAggiornamento file CSV...")
    count = 0
    for csv_file in REPO.rglob("*.csv"):
        try:
            content = csv_file.read_text(encoding='utf-8')
            original = content
            for old, new in CSV_REPLACEMENTS.items():
                content = content.replace(old, new)
            if content != original:
                csv_file.write_text(content, encoding='utf-8')
                print(f"  Aggiornato: {csv_file.relative_to(REPO)}")
                count += 1
        except Exception as e:
            print(f"  Errore su {csv_file}: {e}")
    print(f"  Aggiornati {count} file CSV")

def update_docs():
    """Final documentation updates"""
    print("\nAggiornamento finale documentazione...")
    
    # RESULTS_PROVENANCE.md
    prov = REPO / "docs" / "RESULTS_PROVENANCE.md"
    if prov.exists():
        content = prov.read_text(encoding='utf-8')
        for pattern, repl in DOC_FINAL_REPLACEMENTS.items():
            content = re.sub(pattern, repl, content)
        prov.write_text(content, encoding='utf-8')
        print("  docs/RESULTS_PROVENANCE.md aggiornato")
    
    # DATA.md
    data_md = REPO / "docs" / "DATA.md"
    if data_md.exists():
        content = data_md.read_text(encoding='utf-8')
        for old, new in CSV_REPLACEMENTS.items():
            content = content.replace(old, new)
        data_md.write_text(content, encoding='utf-8')
        print("  docs/DATA.md aggiornato")
    
    # README.md - any remaining
    readme = REPO / "README.md"
    if readme.exists():
        content = readme.read_text(encoding='utf-8')
        for pattern, repl in DOC_FINAL_REPLACEMENTS.items():
            content = re.sub(pattern, repl, content)
        readme.write_text(content, encoding='utf-8')
        print("  README.md aggiornato")

def main():
    print("=" * 60)
    print("FINAL CLEANUP: analisi_* + CSV + DOCS")
    print("=" * 60)
    
    rename_analisi_dirs()
    update_csv_files()
    update_docs()
    
    print("\n" + "=" * 60)
    print("FINAL CLEANUP COMPLETATO")
    print("=" * 60)

if __name__ == "__main__":
    main()