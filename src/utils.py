"""
================================================================================
MODUL: src/utils.py
DESKRIPSI: Utilitas Pembantu, Replikasi Riset (Seed), & Persistensi Model
================================================================================

Tujuan Ilmiah / Konsep Teori:
-----------------------------
1. Reproducibility (Replikasi Hasil Riset):
   - Dalam Data Mining & Machine Learning, proses seperti pengacakan data (shuffling)
     dan inisialisasi bobot model melibatkan generator bilangan acak (Pseudo-Random Number Generator).
   - Fungsi `seed_everything()` mengunci nilai acak (default seed = 42) pada library standard random,
     numpy, dan hash Python. Hal ini menjamin siapa pun yang menjalankan eksperimen akan
     memperoleh nilai metrik dan hasil yang 100% identik dan dapat diuji ulang (reproducible).

2. Centralized Configuration Management:
   - Fungsi `load_config()` membaca berkas `config.yaml` agar parameter eksperimen (paths,
     hyperparameter, threshold) tidak di-hardcode di dalam kode program.
"""

import os
import random
import yaml
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Membaca berkas konfigurasi YAML dan mengembalikannya dalam bentuk Python dictionary.
    Mendukung pencarian path relatif terhadap direktori modul maupun root project.
    """
    if not os.path.exists(config_path):
        # Cari di tingkat root direktori project
        alt_path = os.path.join(os.path.dirname(__file__), "..", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Berkas konfigurasi tidak ditemukan di path: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42) -> None:
    """
    Mengunci generator bilangan acak secara global untuk menjamin konsistensi replikasi riset.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_seed(seed: int = 42) -> None:
    """Alias untuk fungsi seed_everything."""
    seed_everything(seed)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Membaca dataset berformat CSV ke dalam objek pandas DataFrame.
    """
    if not os.path.exists(filepath):
        alt_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset tidak ditemukan di path: {filepath}")

    df = pd.read_csv(filepath)
    return df


def save_model(model_obj: Any, filepath: str) -> None:
    """
    Menyimpan objek model/pipeline Machine Learning ke disk menggunakan Joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_obj, filepath)


def load_model(filepath: str) -> Any:
    """
    Memuat objek model/pipeline Machine Learning dari disk menggunakan Joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Berkas model tidak ditemukan di path: {filepath}")
    return joblib.load(filepath)
