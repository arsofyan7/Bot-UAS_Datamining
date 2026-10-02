"""
================================================================================
SKRIP: scripts/validate_dataset.py
DESKRIPSI: Skrip Validasi Skema Dataset & Deteksi Kebocoran Data (Data Leakage)
================================================================================

Tujuan Ilmiah / Konsep Teori:
-----------------------------
Skrip ini bertujuan memverifikasi integritas dataset sebelum proses pelatihan model.
Pengecekan meliputi:
1. Skema Kolom Wajib (8 Kolom): Memastikan konsistensi format CSV.
2. Null Check & ID Unik: Memastikan tidak ada data kosong pada fitur esensial.
3. Validasi Split: Nilai split harus salah satu dari {'train', 'val', 'test', 'oos-test'}.
4. Deteksi Kebocoran Data (Data Leakage): Memeriksa apakah ada kalimat pada data uji
   (val/test/oos) yang persis sama dengan data latih (train). Data leakage adalah kesalahan
   fatal dalam machine learning yang menyebabkan model tampak memiliki performa tinggi semu.

Penggunaan:
    python scripts/validate_dataset.py [path_ke_file_dataset.csv]
"""

import sys
import os
import csv
import argparse

# 8 Kolom Standar Wajib Sesuai Desain Riset
REQUIRED_COLUMNS = [
    'id',
    'utterance',
    'normalized_utterance',
    'intent',
    'domain',
    'source',
    'split',
    'boundary_case_notes'
]

# Himpunan Nilai Split yang Valid
VALID_SPLITS = {'train', 'val', 'test', 'oos-test'}


# -----------------------------------------------------------------------------
# FUNGSI LOGGING TERMINAL DENGAN KODE WARNA ANSI
# -----------------------------------------------------------------------------
def log_info(msg: str):
    print(f"[INFO] {msg}")


def log_success(msg: str):
    print(f"\033[92m[SUCCESS] {msg}\033[0m")


def log_warning(msg: str):
    print(f"\033[93m[WARNING] {msg}\033[0m")


def log_error(msg: str):
    print(f"\033[91m[ERROR] {msg}\033[0m")


# -----------------------------------------------------------------------------
# FUNGSI UTAMA VALIDASI DATASET
# -----------------------------------------------------------------------------
def validate_dataset(filepath: str) -> bool:
    """
    Mengeksekusi 4 tahapan validasi dataset secara berurutan.
    Mengembalikan True jika seluruh pengujian lulus, atau False jika ada error.
    """
    log_info(f"Memulai validasi dataset: {filepath}")

    if not os.path.exists(filepath):
        log_error(f"File tidak ditemukan di path: {filepath}")
        return False

    rows = []
    try:
        # Membaca file CSV menggunakan modul standar Python csv.DictReader
        with open(filepath, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers is None:
                log_error("File CSV kosong atau tidak memiliki baris header.")
                return False
            for row in reader:
                rows.append(row)
    except Exception as e:
        log_error(f"Gagal membaca file CSV: {e}")
        return False

    is_valid = True
    total_rows = len(rows)
    log_info(f"Total baris data ditemukan: {total_rows}")

    if total_rows == 0:
        log_error("Dataset tidak memiliki baris data (0 baris).")
        return False

    # -------------------------------------------------------------------------
    # TAHAP 1: VALIDASI SKEMA KOLOM WAJIB
    # -------------------------------------------------------------------------
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in headers]
    extra_cols = [col for col in headers if col not in REQUIRED_COLUMNS]

    if missing_cols:
        log_error(f"Kolom wajib yang hilang: {missing_cols}")
        is_valid = False
    else:
        log_success("Semua skema kolom wajib terpenuhi.")

    if extra_cols:
        log_warning(f"Kolom tambahan ditemukan (opsional/non-standar): {extra_cols}")

    if not is_valid:
        return False

    # -------------------------------------------------------------------------
    # TAHAP 2: VALIDASI NILAI NULL, UNIK ID, & DISTRIBUSI SPLIT
    # -------------------------------------------------------------------------
    essential_cols = ['id', 'utterance', 'intent', 'split']
    seen_ids = set()
    duplicate_ids = []
    split_counts = {}
    unique_intents = set()
    unique_domains = set()

    train_utterances = set()
    val_test_rows = []

    for idx, row in enumerate(rows, start=1):
        # Pengecekan nilai kosong pada kolom esensial
        for col in essential_cols:
            val = (row.get(col) or "").strip()
            if not val:
                log_error(f"Baris #{idx}: Kolom esensial '{col}' bernilai kosong.")
                is_valid = False

        # Pengecekan keunikan ID
        row_id = (row.get('id') or "").strip()
        if row_id in seen_ids:
            duplicate_ids.append(row_id)
        seen_ids.add(row_id)

        # Validasi kategori split
        split_val = (row.get('split') or "").strip()
        if split_val not in VALID_SPLITS:
            log_error(f"Baris #{idx} (ID: {row_id}): Nilai split '{split_val}' tidak valid. Harus salah satu dari {list(VALID_SPLITS)}")
            is_valid = False
        else:
            split_counts[split_val] = split_counts.get(split_val, 0) + 1

        intent_val = (row.get('intent') or "").strip()
        domain_val = (row.get('domain') or "").strip()
        if intent_val:
            unique_intents.add(intent_val)
        if domain_val:
            unique_domains.add(domain_val)

        # Simpan teks untuk pengecekan kebocoran data
        utt = (row.get('utterance') or "").strip().lower()
        if split_val == 'train':
            train_utterances.add(utt)
        elif split_val in {'val', 'test', 'oos-test'}:
            val_test_rows.append((row_id, split_val, utt, row.get('utterance', '')))

    if duplicate_ids:
        log_error(f"Ditemukan {len(duplicate_ids)} ID duplikat: {duplicate_ids[:5]}...")
        is_valid = False
    else:
        log_success("Semua ID bersifat unik.")

    if is_valid and split_counts:
        log_success(f"Distribusi split valid: {split_counts}")

    # -------------------------------------------------------------------------
    # TAHAP 3: PENGECEKAN KEBOCORAN DATA (DATA LEAKAGE CHECK)
    # -------------------------------------------------------------------------
    leaked_rows = [item for item in val_test_rows if item[2] in train_utterances]
    if leaked_rows:
        log_warning(f"Terdeteksi potensi Data Leakage! {len(leaked_rows)} utterance di val/test persis sama dengan train:")
        for r_id, r_split, _, orig_utt in leaked_rows[:5]:
            log_warning(f"  - [{r_split.upper()}] ID: {r_id} | Utterance: \"{orig_utt}\"")
        if len(leaked_rows) > 5:
            log_warning(f"  ...dan {len(leaked_rows) - 5} baris lainnya.")
    else:
        log_success("Tidak ditemukan Data Leakage (duplikasi teks persis) antara Train dan Val/Test.")

    log_info(f"Ringkasan: {len(unique_intents)} Unique Intents, {len(unique_domains)} Unique Domains.")

    # -------------------------------------------------------------------------
    # STATUS AKHIR
    # -------------------------------------------------------------------------
    if is_valid:
        log_success("SELURUH VALIDASI BERHASIL! Dataset siap digunakan untuk eksperimen.")
    else:
        log_error("VALIDASI GAGAL! Harap perbaiki kesalahan di atas sebelum melanjutkan.")

    return is_valid


def main():
    """Entry point eksekusi skrip dari command-line interface."""
    parser = argparse.ArgumentParser(description="Validasi Dataset Intent Classification NLP")
    parser.add_argument(
        "filepath",
        nargs="?",
        default="data/templates/dataset_template.csv",
        help="Path ke file dataset CSV (default: data/templates/dataset_template.csv)"
    )
    args = parser.parse_args()

    success = validate_dataset(args.filepath)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
