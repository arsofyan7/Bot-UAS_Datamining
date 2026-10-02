"""
Dataset Validation Script
=========================
Validates the CSV schema, split integrity, null values, and checks for data leakage
between train, validation, and test splits.

Usage:
    python scripts/validate_dataset.py [path_to_csv]
"""

import sys
import os
import csv
import argparse

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

VALID_SPLITS = {'train', 'val', 'test', 'oos-test'}


def log_info(msg: str):
    print(f"[INFO] {msg}")


def log_success(msg: str):
    print(f"\033[92m[SUCCESS] {msg}\033[0m")


def log_warning(msg: str):
    print(f"\033[93m[WARNING] {msg}\033[0m")


def log_error(msg: str):
    print(f"\033[91m[ERROR] {msg}\033[0m")


def validate_dataset(filepath: str) -> bool:
    log_info(f"Memulai validasi dataset: {filepath}")

    if not os.path.exists(filepath):
        log_error(f"File tidak ditemukan di path: {filepath}")
        return False

    rows = []
    try:
        with open(filepath, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers is None:
                log_error("File CSV kosong atau tidak memiliki header.")
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

    # 1. Validasi Kolom Wajib
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

    # 2. Validasi Nilai Kosong & Unik ID
    essential_cols = ['id', 'utterance', 'intent', 'split']
    seen_ids = set()
    duplicate_ids = []
    split_counts = {}
    unique_intents = set()
    unique_domains = set()

    train_utterances = set()
    val_test_rows = []

    for idx, row in enumerate(rows, start=1):
        # Null check
        for col in essential_cols:
            val = (row.get(col) or "").strip()
            if not val:
                log_error(f"Baris #{idx}: Kolom esensial '{col}' bernilai kosong.")
                is_valid = False

        row_id = (row.get('id') or "").strip()
        if row_id in seen_ids:
            duplicate_ids.append(row_id)
        seen_ids.add(row_id)

        # Split check
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

        # Track for leakage check
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

    # 3. Pengecekan Kebocoran Data (Data Leakage)
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

    if is_valid:
        log_success("SELURUH VALIDASI BERHASIL! Dataset siap digunakan untuk eksperimen.")
    else:
        log_error("VALIDASI GAGAL! Harap perbaiki kesalahan di atas sebelum melanjutkan.")

    return is_valid


def main():
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
