#!/usr/bin/env python3
"""
Phase 2: Rename checkpoint files, update documentation, fix remaining references.
"""

import os
import re
import shutil
from pathlib import Path

REPO = Path("/home/andrealarocca/Scrivania/Tesi_clean_repo")

# Mapping per file checkpoint (vecchio -> nuovo)
CHECKPOINT_RENAME = {
    # FINAL_NUOVO -> reservoir_puro
    "FINAL_NUOVO_BR_CLF.pth": "reservoir_puro_BR_CLF.pth",
    "FINAL_NUOVO_BR_REG.pth": "reservoir_puro_BR_REG.pth",
    "FINAL_NUOVO_BR_R_ade.npy": "reservoir_puro_BR_R_ade.npy",
    "FINAL_NUOVO_BR_R_f1.npy": "reservoir_puro_BR_R_f1.npy",
    "FINAL_NUOVO_BR_R_mse.npy": "reservoir_puro_BR_R_mse.npy",
    "FINAL_NUOVO_AGEM_CLF.pth": "reservoir_puro_AGEM_CLF.pth",
    "FINAL_NUOVO_AGEM_REG.pth": "reservoir_puro_AGEM_REG.pth",
    "FINAL_NUOVO_AGEM_R_ade.npy": "reservoir_puro_AGEM_R_ade.npy",
    "FINAL_NUOVO_AGEM_R_f1.npy": "reservoir_puro_AGEM_R_f1.npy",
    "FINAL_NUOVO_AGEM_R_mse.npy": "reservoir_puro_AGEM_R_mse.npy",
    "FINAL_NUOVO_EWC_CLF.pth": "reservoir_puro_EWC_CLF.pth",
    "FINAL_NUOVO_EWC_REG.pth": "reservoir_puro_EWC_REG.pth",
    "FINAL_NUOVO_EWC_R_ade.npy": "reservoir_puro_EWC_R_ade.npy",
    "FINAL_NUOVO_EWC_R_f1.npy": "reservoir_puro_EWC_R_f1.npy",
    "FINAL_NUOVO_EWC_R_mse.npy": "reservoir_puro_EWC_R_mse.npy",
    
    # FINAL_CBRS -> cbrs_binario_b500
    "FINAL_CBRS_BR_CLF.pth": "cbrs_binario_b500_BR_CLF.pth",
    "FINAL_CBRS_BR_REG.pth": "cbrs_binario_b500_BR_REG.pth",
    "FINAL_CBRS_BR_R_ade.npy": "cbrs_binario_b500_BR_R_ade.npy",
    "FINAL_CBRS_BR_R_f1.npy": "cbrs_binario_b500_BR_R_f1.npy",
    "FINAL_CBRS_BR_R_mse.npy": "cbrs_binario_b500_BR_R_mse.npy",
    "FINAL_CBRS_AGEM_CLF.pth": "cbrs_binario_b500_AGEM_CLF.pth",
    "FINAL_CBRS_AGEM_REG.pth": "cbrs_binario_b500_AGEM_REG.pth",
    "FINAL_CBRS_AGEM_R_ade.npy": "cbrs_binario_b500_AGEM_R_ade.npy",
    "FINAL_CBRS_AGEM_R_f1.npy": "cbrs_binario_b500_AGEM_R_f1.npy",
    "FINAL_CBRS_AGEM_R_mse.npy": "cbrs_binario_b500_AGEM_R_mse.npy",
    "FINAL_CBRS_EWC_CLF.pth": "cbrs_binario_b500_EWC_CLF.pth",
    "FINAL_CBRS_EWC_REG.pth": "cbrs_binario_b500_EWC_REG.pth",
    "FINAL_CBRS_EWC_R_ade.npy": "cbrs_binario_b500_EWC_R_ade.npy",
    "FINAL_CBRS_EWC_R_f1.npy": "cbrs_binario_b500_EWC_R_f1.npy",
    "FINAL_CBRS_EWC_R_mse.npy": "cbrs_binario_b500_EWC_R_mse.npy",
    
    # FINAL_CBRS_ATTACCO -> cbrs_attacco (SUPERSEDED)
    "FINAL_CBRS_ATTACCO_BR_CLF.pth": "cbrs_attacco_BR_CLF.pth",
    "FINAL_CBRS_ATTACCO_BR_REG.pth": "cbrs_attacco_BR_REG.pth",
    "FINAL_CBRS_ATTACCO_BR_R_ade.npy": "cbrs_attacco_BR_R_ade.npy",
    "FINAL_CBRS_ATTACCO_BR_R_f1.npy": "cbrs_attacco_BR_R_f1.npy",
    "FINAL_CBRS_ATTACCO_BR_R_mse.npy": "cbrs_attacco_BR_R_mse.npy",
    "FINAL_CBRS_ATTACCO_AGEM_CLF.pth": "cbrs_attacco_AGEM_CLF.pth",
    "FINAL_CBRS_ATTACCO_AGEM_REG.pth": "cbrs_attacco_AGEM_REG.pth",
    "FINAL_CBRS_ATTACCO_AGEM_R_ade.npy": "cbrs_attacco_AGEM_R_ade.npy",
    "FINAL_CBRS_ATTACCO_AGEM_R_f1.npy": "cbrs_attacco_AGEM_R_f1.npy",
    "FINAL_CBRS_ATTACCO_AGEM_R_mse.npy": "cbrs_attacco_AGEM_R_mse.npy",
    "FINAL_CBRS_ATTACCO_EWC_CLF.pth": "cbrs_attacco_EWC_CLF.pth",
    "FINAL_CBRS_ATTACCO_EWC_REG.pth": "cbrs_attacco_EWC_REG.pth",
    "FINAL_CBRS_ATTACCO_EWC_R_ade.npy": "cbrs_attacco_EWC_R_ade.npy",
    "FINAL_CBRS_ATTACCO_EWC_R_f1.npy": "cbrs_attacco_EWC_R_f1.npy",
    "FINAL_CBRS_ATTACCO_EWC_R_mse.npy": "cbrs_attacco_EWC_R_mse.npy",
    
    # BUFFER_GRANDE_* -> cbrs_binario_b*
    "BUFFER_GRANDE_20000_BR_CLF.pth": "cbrs_binario_b20000_BR_CLF.pth",
    "BUFFER_GRANDE_20000_BR_REG.pth": "cbrs_binario_b20000_BR_REG.pth",
    "BUFFER_GRANDE_20000_BR_R_ade.npy": "cbrs_binario_b20000_BR_R_ade.npy",
    "BUFFER_GRANDE_20000_BR_R_f1.npy": "cbrs_binario_b20000_BR_R_f1.npy",
    "BUFFER_GRANDE_20000_BR_R_mse.npy": "cbrs_binario_b20000_BR_R_mse.npy",
    "BUFFER_GRANDE_10000_BR_CLF.pth": "cbrs_binario_b10000_BR_CLF.pth",
    "BUFFER_GRANDE_10000_BR_REG.pth": "cbrs_binario_b10000_BR_REG.pth",
    "BUFFER_GRANDE_10000_BR_R_ade.npy": "cbrs_binario_b10000_BR_R_ade.npy",
    "BUFFER_GRANDE_10000_BR_R_f1.npy": "cbrs_binario_b10000_BR_R_f1.npy",
    "BUFFER_GRANDE_10000_BR_R_mse.npy": "cbrs_binario_b10000_BR_R_mse.npy",
    "BUFFER_GRANDE_5000_BR_CLF.pth": "cbrs_binario_b5000_BR_CLF.pth",
    "BUFFER_GRANDE_5000_BR_REG.pth": "cbrs_binario_b5000_BR_REG.pth",
    "BUFFER_GRANDE_5000_BR_R_ade.npy": "cbrs_binario_b5000_BR_R_ade.npy",
    "BUFFER_GRANDE_5000_BR_R_f1.npy": "cbrs_binario_b5000_BR_R_f1.npy",
    "BUFFER_GRANDE_5000_BR_R_mse.npy": "cbrs_binario_b5000_BR_R_mse.npy",
    
    # VERIFICATO_6VIE_20000 -> cbrs_multiclasse_b20000
    "VERIFICATO_6VIE_20000_BR_CLF.pth": "cbrs_multiclasse_b20000_BR_CLF.pth",
    "VERIFICATO_6VIE_20000_BR_REG.pth": "cbrs_multiclasse_b20000_BR_REG.pth",
    "VERIFICATO_6VIE_20000_BR_R_ade.npy": "cbrs_multiclasse_b20000_BR_R_ade.npy",
    "VERIFICATO_6VIE_20000_BR_R_f1.npy": "cbrs_multiclasse_b20000_BR_R_f1.npy",
    "VERIFICATO_6VIE_20000_BR_R_mse.npy": "cbrs_multiclasse_b20000_BR_R_mse.npy",
    "VERIFICATO_6VIE_20000_AGEM_CLF.pth": "cbrs_multiclasse_b20000_AGEM_CLF.pth",
    "VERIFICATO_6VIE_20000_AGEM_REG.pth": "cbrs_multiclasse_b20000_AGEM_REG.pth",
    "VERIFICATO_6VIE_20000_AGEM_R_ade.npy": "cbrs_multiclasse_b20000_AGEM_R_ade.npy",
    "VERIFICATO_6VIE_20000_AGEM_R_f1.npy": "cbrs_multiclasse_b20000_AGEM_R_f1.npy",
    "VERIFICATO_6VIE_20000_AGEM_R_mse.npy": "cbrs_multiclasse_b20000_AGEM_R_mse.npy",
    
    # VERIFICATO_6VIE -> cbrs_multiclasse_b10000
    "VERIFICATO_6VIE_BR_CLF.pth": "cbrs_multiclasse_b10000_BR_CLF.pth",
    "VERIFICATO_6VIE_BR_REG.pth": "cbrs_multiclasse_b10000_BR_REG.pth",
    "VERIFICATO_6VIE_BR_R_ade.npy": "cbrs_multiclasse_b10000_BR_R_ade.npy",
    "VERIFICATO_6VIE_BR_R_f1.npy": "cbrs_multiclasse_b10000_BR_R_f1.npy",
    "VERIFICATO_6VIE_BR_R_mse.npy": "cbrs_multiclasse_b10000_BR_R_mse.npy",
    "VERIFICATO_6VIE_AGEM_CLF.pth": "cbrs_multiclasse_b10000_AGEM_CLF.pth",
    "VERIFICATO_6VIE_AGEM_REG.pth": "cbrs_multiclasse_b10000_AGEM_REG.pth",
    "VERIFICATO_6VIE_AGEM_R_ade.npy": "cbrs_multiclasse_b10000_AGEM_R_ade.npy",
    "VERIFICATO_6VIE_AGEM_R_f1.npy": "cbrs_multiclasse_b10000_AGEM_R_f1.npy",
    "VERIFICATO_6VIE_AGEM_R_mse.npy": "cbrs_multiclasse_b10000_AGEM_R_mse.npy",
    
    # EWC_CORRETTO -> ewc
    "EWC_CORRETTO_EWC_CLF.pth": "ewc_EWC_CLF.pth",
    "EWC_CORRETTO_EWC_REG.pth": "ewc_EWC_REG.pth",
    "EWC_CORRETTO_EWC_R_ade.npy": "ewc_EWC_R_ade.npy",
    "EWC_CORRETTO_EWC_R_f1.npy": "ewc_EWC_R_f1.npy",
    "EWC_CORRETTO_EWC_R_mse.npy": "ewc_EWC_R_mse.npy",
    
    # LOWER_BOUND -> lower_bound
    "LOWER_BOUND_NONE_CLF.pth": "lower_bound_NESSUNA_CLF.pth",
    "LOWER_BOUND_NONE_REG.pth": "lower_bound_NESSUNA_REG.pth",
    "LOWER_BOUND_NONE_R_ade.npy": "lower_bound_NESSUNA_R_ade.npy",
    "LOWER_BOUND_NONE_R_f1.npy": "lower_bound_NESSUNA_R_f1.npy",
    "LOWER_BOUND_NONE_R_mse.npy": "lower_bound_NESSUNA_R_mse.npy",
    
    # UPPER_BOUND -> upper_bound
    "UPPER_BOUND_CLF.pth": "upper_bound_CLF.pth",
}

# Sostituzioni per documentazione
DOC_REPLACEMENTS = [
    # README.md - sezione checkpoints_final
    (r'VERIFICATO_6VIE_20000_\*', 'cbrs_multiclasse_b20000_*'),
    (r'EWC_CORRETTO_\*', 'ewc_*'),
    (r'LOWER_BOUND_\*', 'lower_bound_*'),
    (r'BUFFER_GRANDE_20000_\*', 'cbrs_binario_b20000_*'),
    (r'FINAL_CBRS_\*', 'cbrs_binario_b500_*'),
    (r'FINAL_NUOVO_\*', 'reservoir_puro_*'),
    (r'VERIFICATO_6VIE_\*', 'cbrs_multiclasse_b10000_*'),
    (r'UPPER_BOUND_CLF\.pth', 'upper_bound_CLF.pth'),
    
    # Titoli nelle tabelle/legende
    (r'Reservoir Puro', 'Reservoir Puro'),
    (r'CBRS Binario, buffer 500', 'CBRS Binario, buffer 500'),
    (r'CBRS Binario, buffer 20000', 'CBRS Binario, buffer 20000'),
    (r'CBRS Multiclasse, buffer 20000', 'CBRS Multiclasse, buffer 20000'),
    (r'CBRS Multiclasse, buffer 10000', 'CBRS Multiclasse, buffer 10000'),
    (r'Nessuna Strategia \(Lower Bound\)', 'Nessuna Strategia (Lower Bound)'),
    (r'Addestramento Congiunto \(Upper Bound\)', 'Addestramento Congiunto (Upper Bound)'),
]

# Mapping per RESULTS_PROVENANCE.md e altri doc
PROVENANCE_REPLACEMENTS = [
    # File JSON references
    (r'`FINAL_NUOVO_BR\.json`', '`reservoir_puro_BR.json`'),
    (r'`FINAL_NUOVO_AGEM\.json`', '`reservoir_puro_AGEM.json`'),
    (r'`FINAL_NUOVO_EWC\.json`', '`reservoir_puro_EWC.json`'),
    (r'`EWC_CORRETTO_EWC\.json`', '`ewc_EWC.json`'),
    (r'`FINAL_CBRS_BR\.json`', '`cbrs_binario_b500_BR.json`'),
    (r'`FINAL_CBRS_AGEM\.json`', '`cbrs_binario_b500_AGEM.json`'),
    (r'`FINAL_CBRS_EWC\.json`', '`cbrs_binario_b500_EWC.json`'),
    (r'`BUFFER_GRANDE_20000_BR\.json`', '`cbrs_binario_b20000_BR.json`'),
    (r'`VERIFICATO_6VIE_20000_BR\.json`', '`cbrs_multiclasse_b20000_BR.json`'),
    (r'`VERIFICATO_6VIE_20000_AGEM\.json`', '`cbrs_multiclasse_b20000_AGEM.json`'),
    (r'`VERIFICATO_6VIE_BR\.json`', '`cbrs_multiclasse_b10000_BR.json`'),
    (r'`VERIFICATO_6VIE_AGEM\.json`', '`cbrs_multiclasse_b10000_AGEM.json`'),
    (r'`LOWER_BOUND_NONE\.json`', '`lower_bound_NESSUNA.json`'),
    (r'`SWEEP_BUFFER500_BR\.json`', '`sweep_buffer_b500_BR.json`'),
    (r'`SWEEP_BUFFER500_AGEM\.json`', '`sweep_buffer_b500_AGEM.json`'),
    (r'`SWEEP_BUFFER1000_BR\.json`', '`sweep_buffer_b1000_BR.json`'),
    (r'`SWEEP_BUFFER1000_AGEM\.json`', '`sweep_buffer_b1000_AGEM.json`'),
    (r'`SWEEP_BUFFER2000_BR\.json`', '`sweep_buffer_b2000_BR.json`'),
    (r'`SWEEP_BUFFER2000_AGEM\.json`', '`sweep_buffer_b2000_AGEM.json`'),
    (r'`SWEEP_BUFFER3000_BR\.json`', '`sweep_buffer_b3000_BR.json`'),
    (r'`SWEEP_BUFFER3000_AGEM\.json`', '`sweep_buffer_b3000_AGEM.json`'),
    (r'`BUFFER_GRANDE_5000_BR\.json`', '`cbrs_binario_b5000_BR.json`'),
    (r'`BUFFER_GRANDE_10000_BR\.json`', '`cbrs_binario_b10000_BR.json`'),
    (r'`BUFFER_GRANDE_20000_BR\.json`', '`cbrs_binario_b20000_BR.json`'),
]

def rename_checkpoints():
    """Rename checkpoint files in pesi_finali/"""
    print("Rinomina file checkpoint in pesi_finali/")
    pesi_dir = REPO / "pesi_finali"
    count = 0
    for old_name, new_name in CHECKPOINT_RENAME.items():
        old_path = pesi_dir / old_name
        new_path = pesi_dir / new_name
        if old_path.exists():
            old_path.rename(new_path)
            print(f"  {old_name} -> {new_name}")
            count += 1
        else:
            print(f"  MANCANTE: {old_name}")
    print(f"Rinominati {count} file checkpoint")

def update_documentation():
    """Update README.md, docs/*.md"""
    print("\nAggiornamento documentazione...")
    
    # README.md
    readme = REPO / "README.md"
    if readme.exists():
        content = readme.read_text(encoding='utf-8')
        for pattern, repl in DOC_REPLACEMENTS:
            content = re.sub(pattern, repl, content)
        readme.write_text(content, encoding='utf-8')
        print("  README.md aggiornato")
    
    # docs/RESULTS_PROVENANCE.md
    prov = REPO / "docs" / "RESULTS_PROVENANCE.md"
    if prov.exists():
        content = prov.read_text(encoding='utf-8')
        for pattern, repl in PROVENANCE_REPLACEMENTS:
            content = re.sub(pattern, repl, content)
        prov.write_text(content, encoding='utf-8')
        print("  docs/RESULTS_PROVENANCE.md aggiornato")
    
    # docs/DATA.md - paths to checkpoints
    data_md = REPO / "docs" / "DATA.md"
    if data_md.exists():
        content = data_md.read_text(encoding='utf-8')
        for old, new in CHECKPOINT_RENAME.items():
            content = content.replace(old, new)
        data_md.write_text(content, encoding='utf-8')
        print("  docs/DATA.md aggiornato")
    
    # docs/HARDCODED_PATHS.md
    hardcoded = REPO / "docs" / "HARDCODED_PATHS.md"
    if hardcoded.exists():
        content = hardcoded.read_text(encoding='utf-8')
        content = content.replace("results_v2", "risultati")
        content = content.replace("grafici_per_prof", "figure")
        content = content.replace("checkpoints_final", "pesi_finali")
        content = content.replace("run_experiment_v2.py", "run_experiment.py")
        hardcoded.write_text(content, encoding='utf-8')
        print("  docs/HARDCODED_PATHS.md aggiornato")
    
    # docs/PIPELINE.md
    pipeline = REPO / "docs" / "PIPELINE.md"
    if pipeline.exists():
        content = pipeline.read_text(encoding='utf-8')
        content = content.replace("results_v2", "risultati")
        content = content.replace("grafici_per_prof", "figure")
        content = content.replace("checkpoints_final", "pesi_finali")
        content = content.replace("run_experiment_v2.py", "run_experiment.py")
        pipeline.write_text(content, encoding='utf-8')
        print("  docs/PIPELINE.md aggiornato")

def update_json_files():
    """Update JSON files in risultati/ that contain old run_ids internally"""
    print("\nAggiornamento run_id dentro file JSON...")
    risultati_dir = REPO / "risultati"
    count = 0
    for json_file in risultati_dir.rglob("*.json"):
        try:
            content = json_file.read_text(encoding='utf-8')
            original = content
            # Sostituisci run_id all'interno
            for old, new in CHECKPOINT_RENAME.items():
                # Estrai run_id dal nome file (es. FINAL_NUOVO_BR_CLF.pth -> FINAL_NUOVO_BR)
                run_id_old = old.replace("_CLF.pth", "").replace("_REG.pth", "").replace("_R_ade.npy", "").replace("_R_f1.npy", "").replace("_R_mse.npy", "")
                run_id_new = new.replace("_CLF.pth", "").replace("_REG.pth", "").replace("_R_ade.npy", "").replace("_R_f1.npy", "").replace("_R_mse.npy", "")
                content = content.replace(f'"run_id": "{run_id_old}"', f'"run_id": "{run_id_new}"')
                content = content.replace(f'"run_id": "{run_id_old}', f'"run_id": "{run_id_new}')
            
            # Sostituzioni generali per run_id noti
            replacements = {
                "FINAL_NUOVO": "reservoir_puro",
                "FINAL_CBRS": "cbrs_binario_b500",
                "BUFFER_GRANDE_20000": "cbrs_binario_b20000",
                "VERIFICATO_6VIE_20000": "cbrs_multiclasse_b20000",
                "VERIFICATO_6VIE": "cbrs_multiclasse_b10000",
                "EWC_CORRETTO": "ewc",
                "LOWER_BOUND": "lower_bound",
                "UPPER_BOUND": "upper_bound",
                "FINAL_CBRS_ATTACCO": "cbrs_attacco",
                "SWEEP_BUFFER500": "sweep_buffer_b500",
                "SWEEP_BUFFER1000": "sweep_buffer_b1000",
                "SWEEP_BUFFER2000": "sweep_buffer_b2000",
                "SWEEP_BUFFER3000": "sweep_buffer_b3000",
                "SWEEP_BUFFER5000": "sweep_buffer_b5000",
                "SWEEP_BUFFER10000": "sweep_buffer_b10000",
                "SWEEP_BUFFER20000": "sweep_buffer_b20000",
                "SWEEP_LAMBDA10": "sweep_lambda_10",
                "SWEEP_LAMBDA50": "sweep_lambda_50",
                "SWEEP_LAMBDA200": "sweep_lambda_200",
                "SWEEP_LAMBDA1000": "sweep_lambda_1000",
                "BUFFER_GRANDE_5000": "cbrs_binario_b5000",
                "BUFFER_GRANDE_10000": "cbrs_binario_b10000",
            }
            for old, new in replacements.items():
                content = content.replace(f'"{old}"', f'"{new}"')
            
            if content != original:
                json_file.write_text(content, encoding='utf-8')
                count += 1
        except Exception as e:
            print(f"  Errore su {json_file}: {e}")
    print(f"  Aggiornati {count} file JSON")

def main():
    print("=" * 60)
    print("PHASE 2: CHECKPOINT + DOCUMENTAZIONE + JSON INTERNI")
    print("=" * 60)
    
    rename_checkpoints()
    update_documentation()
    update_json_files()
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETATA")
    print("=" * 60)

if __name__ == "__main__":
    main()