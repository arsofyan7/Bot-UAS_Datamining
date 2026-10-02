"""
Automated Experimentation Runner (E0 - E6)
=========================================
Runs full experimental matrix for Indonesian Intent Classification research:
- E0: Baseline (MNB + TF-IDF Unigram + Minimal Preprocessing)
- E1: Preprocessing Ablation (MNB + TF-IDF Unigram + Full Preprocessing)
- E2: Model Comparison (MNB vs Logistic Regression vs Linear SVM)
- E3: N-Gram Ablation (Unigram vs Unigram+Bigram)
- E4: Cross-Validation Stability (5-Fold Stratified CV on Best Candidate)
- E5: Robustness Evaluation (Testing on informal/typo text variations)
- E6: Out-of-Scope (OOS) Threshold Analysis (0.4, 0.5, 0.6, 0.7)

Saves metrics to 'reports/metrics/experiment_results.json', figures to 'reports/figures/',
and the overall best pipeline to 'models/best_model.joblib'.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Ensure root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.models import get_model, IntentClassifierPipeline
from src.evaluation import calculate_metrics, plot_and_save_confusion_matrix, evaluate_oos_threshold
from src.utils import load_config, seed_everything, load_data, save_model


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def print_table(headers: list, rows: list):
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([headers] + rows))]
    fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    sep = "-+-".join(["-" * w for w in col_widths])
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


def run_all_experiments(dataset_path: str = "data/raw/dataset.csv", config_path: str = "config.yaml"):
    config = load_config(config_path)
    seed = config.get("project", {}).get("random_seed", 42)
    seed_everything(seed)

    print_section("Memulai Eksekusi Eksperimen Intent Classification NLP")
    print(f"[INFO] Dataset Path : {dataset_path}")
    print(f"[INFO] Random Seed  : {seed}")

    # 1. Load Data
    if not os.path.exists(dataset_path):
        # Fallback to template if dataset not ready
        template_path = "data/templates/dataset_template.csv"
        print(f"[WARNING] Dataset {dataset_path} tidak ditemukan, beralih ke {template_path}")
        dataset_path = template_path

    df = load_data(dataset_path)

    # Filter in-scope and OOS
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    oos_df = df[df["split"] == "oos-test"]

    # Combine in-scope test/val for comprehensive test set
    eval_df = pd.concat([val_df, test_df]) if len(test_df) > 0 else train_df

    X_train_raw = train_df["utterance"].tolist()
    y_train = train_df["intent"].tolist()

    X_eval_raw = eval_df["utterance"].tolist()
    y_eval = eval_df["intent"].tolist()

    oos_texts = oos_df["utterance"].tolist() if len(oos_df) > 0 else [
        "siapa presiden amerika saat ini",
        "bagaimana resep sate ayam bumbu kacang",
        "cuaca hari ini mendung atau hujan",
        "berapa kurs dollar ke rupiah"
    ]

    experiment_results = {}
    preprocessor = TextPreprocessor()

    # Preprocessed Texts
    X_train_clean = [preprocessor.transform(t, full_pipeline=True) for t in X_train_raw]
    X_eval_clean = [preprocessor.transform(t, full_pipeline=True) for t in X_eval_raw]

    # Minimal Preprocessed Texts (for E0)
    X_train_min = [preprocessor.transform(t, full_pipeline=False) for t in X_train_raw]
    X_eval_min = [preprocessor.transform(t, full_pipeline=False) for t in X_eval_raw]

    # -------------------------------------------------------------------------
    # E0: Baseline (MNB + TF-IDF Unigram + Minimal Preprocessing)
    # -------------------------------------------------------------------------
    print_section("E0: Baseline (MNB + TF-IDF Unigram + Minimal Preprocessing)")
    vec_e0 = FeatureExtractor(vectorizer_type="tfidf", ngram_range=(1, 1)).vectorizer
    clf_e0 = get_model("multinomial_nb", random_state=seed)
    pipe_e0 = IntentClassifierPipeline(vec_e0, clf_e0)
    pipe_e0.fit(X_train_min, y_train)
    preds_e0 = pipe_e0.predict(X_eval_min)
    m_e0 = calculate_metrics(y_eval, preds_e0)
    experiment_results["E0_Baseline"] = m_e0

    print_table(
        ["Eksperimen", "Macro F1", "Accuracy", "Macro Precision", "Macro Recall"],
        [["E0 (Baseline)", f"{m_e0['macro_f1']:.4f}", f"{m_e0['accuracy']:.4f}", f"{m_e0['macro_precision']:.4f}", f"{m_e0['macro_recall']:.4f}"]]
    )

    # -------------------------------------------------------------------------
    # E1: Preprocessing Ablation (MNB + TF-IDF Unigram + Full Preprocessing)
    # -------------------------------------------------------------------------
    print_section("E1: Preprocessing Ablation (Full Preprocessing)")
    vec_e1 = FeatureExtractor(vectorizer_type="tfidf", ngram_range=(1, 1)).vectorizer
    clf_e1 = get_model("multinomial_nb", random_state=seed)
    pipe_e1 = IntentClassifierPipeline(vec_e1, clf_e1)
    pipe_e1.fit(X_train_clean, y_train)
    preds_e1 = pipe_e1.predict(X_eval_clean)
    m_e1 = calculate_metrics(y_eval, preds_e1)
    experiment_results["E1_Full_Preprocessing"] = m_e1

    delta_e1 = m_e1["macro_f1"] - m_e0["macro_f1"]
    print_table(
        ["Eksperimen", "Macro F1", "Delta F1 vs E0", "Accuracy"],
        [
            ["E0 (Minimal)", f"{m_e0['macro_f1']:.4f}", "-", f"{m_e0['accuracy']:.4f}"],
            ["E1 (Full Preprocessing)", f"{m_e1['macro_f1']:.4f}", f"{delta_e1:+.4f}", f"{m_e1['accuracy']:.4f}"]
        ]
    )

    # -------------------------------------------------------------------------
    # E2: Model Comparison (MNB vs Logistic Regression vs Linear SVM)
    # -------------------------------------------------------------------------
    print_section("E2: Model Comparison (MultinomialNB vs LogReg vs Linear SVM)")
    candidate_models = ["multinomial_nb", "logistic_regression", "linear_svm"]
    e2_rows = []
    e2_pipelines = {}

    for model_name in candidate_models:
        vec = FeatureExtractor(vectorizer_type="tfidf", ngram_range=(1, 2)).vectorizer
        clf = get_model(model_name, random_state=seed)
        pipe = IntentClassifierPipeline(vec, clf)
        pipe.fit(X_train_clean, y_train)
        preds = pipe.predict(X_eval_clean)
        metrics = calculate_metrics(y_eval, preds)
        
        experiment_results[f"E2_{model_name}"] = metrics
        e2_pipelines[model_name] = pipe
        e2_rows.append([model_name, f"{metrics['macro_f1']:.4f}", f"{metrics['accuracy']:.4f}", f"{metrics['macro_precision']:.4f}", f"{metrics['macro_recall']:.4f}"])

    print_table(["Model", "Macro F1", "Accuracy", "Precision", "Recall"], e2_rows)

    # Determine best model architecture from E2 (preferring linear_svm/logreg if tie due to superior probability calibration)
    priority_order = {"linear_svm": 3, "logistic_regression": 2, "multinomial_nb": 1}
    best_model_name = max(
        candidate_models,
        key=lambda m: (experiment_results[f"E2_{m}"]["macro_f1"], priority_order.get(m, 0))
    )
    print(f"\n[INFO] Model Terbaik dari E2: {best_model_name.upper()}")

    # -------------------------------------------------------------------------
    # E3: N-Gram Ablation (Unigram vs Unigram+Bigram)
    # -------------------------------------------------------------------------
    print_section("E3: N-Gram Ablation Analysis")
    e3_configs = {
        "Unigram (1,1)": (1, 1),
        "Unigram+Bigram (1,2)": (1, 2)
    }
    e3_rows = []
    e3_pipelines = {}

    for label, ngram_range in e3_configs.items():
        vec = FeatureExtractor(vectorizer_type="tfidf", ngram_range=ngram_range).vectorizer
        clf = get_model(best_model_name, random_state=seed)
        pipe = IntentClassifierPipeline(vec, clf)
        pipe.fit(X_train_clean, y_train)
        preds = pipe.predict(X_eval_clean)
        metrics = calculate_metrics(y_eval, preds)
        experiment_results[f"E3_{label}"] = metrics
        e3_pipelines[label] = pipe
        e3_rows.append([label, f"{metrics['macro_f1']:.4f}", f"{metrics['accuracy']:.4f}"])

    print_table(["N-Gram Feature Configuration", "Macro F1", "Accuracy"], e3_rows)
    best_ngram = max(e3_configs.keys(), key=lambda k: experiment_results[f"E3_{k}"]["macro_f1"])
    best_pipeline = e3_pipelines[best_ngram]

    # -------------------------------------------------------------------------
    # E4: 5-Fold Stratified Cross-Validation Stability
    # -------------------------------------------------------------------------
    print_section("E4: Cross-Validation Stability (5-Fold Stratified CV)")
    X_all_clean = np.array(X_train_clean + X_eval_clean)
    y_all = np.array(y_train + y_eval)

    # Ensure min samples for CV
    min_class_samples = pd.Series(y_all).value_counts().min()
    n_splits = min(5, max(2, min_class_samples))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    cv_f1_scores = []
    cv_acc_scores = []
    e4_rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_clean, y_all), start=1):
        vec = FeatureExtractor(vectorizer_type="tfidf", ngram_range=e3_configs[best_ngram]).vectorizer
        clf = get_model(best_model_name, random_state=seed)
        fold_pipe = IntentClassifierPipeline(vec, clf)
        fold_pipe.fit(X_all_clean[train_idx].tolist(), y_all[train_idx].tolist())
        preds = fold_pipe.predict(X_all_clean[val_idx].tolist())
        
        f1 = f1_score(y_all[val_idx], preds, average="macro", zero_division=0)
        acc = float(np.mean(preds == y_all[val_idx]))
        cv_f1_scores.append(f1)
        cv_acc_scores.append(acc)
        e4_rows.append([f"Fold {fold}", f"{f1:.4f}", f"{acc:.4f}"])

    e4_rows.append(["Mean +/- Std", f"{np.mean(cv_f1_scores):.4f} +/- {np.std(cv_f1_scores):.4f}", f"{np.mean(cv_acc_scores):.4f} +/- {np.std(cv_acc_scores):.4f}"])
    experiment_results["E4_Cross_Validation"] = {
        "fold_f1_scores": cv_f1_scores,
        "mean_macro_f1": float(np.mean(cv_f1_scores)),
        "std_macro_f1": float(np.std(cv_f1_scores)),
        "mean_accuracy": float(np.mean(cv_acc_scores))
    }
    print_table(["Fold Number", "Macro F1", "Accuracy"], e4_rows)

    # -------------------------------------------------------------------------
    # E5: Robustness Evaluation (Testing on informal/typo text variations)
    # -------------------------------------------------------------------------
    print_section("E5: Robustness Evaluation (Informal & Typo Perturbations)")
    informal_test_samples = [
        ("brp bya dftr mhs baru?", "biaya_pendaftaran"),
        ("lokasi kmpusnya dmn y min?", "lokasi_alamat"),
        ("dokumen ap aj yg hrus diupload?", "syarat_masuk"),
        ("kpn msuk kulyah perdana?", "jadwal_kuliah"),
        ("ad info beasiswa prestasi gak?", "beasiswa")
    ]

    # Clean informal samples with our preprocessor
    X_inf_clean = [preprocessor.transform(t[0], full_pipeline=True) for t in informal_test_samples]
    y_inf_true = [t[1] for t in informal_test_samples]
    inf_preds = best_pipeline.predict(X_inf_clean)
    inf_metrics = calculate_metrics(y_inf_true, inf_preds)
    experiment_results["E5_Robustness"] = inf_metrics

    e5_rows = []
    for (raw_txt, true_lbl), pred_lbl in zip(informal_test_samples, inf_preds):
        status = "[PASS]" if true_lbl == pred_lbl else "[FAIL]"
        e5_rows.append([raw_txt, true_lbl, pred_lbl, status])

    print_table(["Input Informal / Typo", "True Intent", "Predicted Intent", "Status"], e5_rows)
    print(f"\n[INFO] Robustness Macro F1: {inf_metrics['macro_f1']:.4f} | Accuracy: {inf_metrics['accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # E6: Out-of-Scope (OOS) Threshold Analysis
    # -------------------------------------------------------------------------
    print_section("E6: Out-of-Scope (OOS) Threshold Analysis")
    thresholds = [0.4, 0.5, 0.6, 0.7]
    oos_eval_texts = [preprocessor.transform(t, full_pipeline=True) for t in oos_texts]
    e6_rows = []
    oos_results = {}

    for thresh in thresholds:
        oos_eval = evaluate_oos_threshold(best_pipeline, oos_eval_texts, ["OOS_REJECTED"] * len(oos_eval_texts), threshold=thresh)
        rejection_rate = oos_eval["metrics"]["rejection_rate"]
        oos_results[str(thresh)] = {
            "rejection_rate": rejection_rate,
            "rejected_count": oos_eval["metrics"]["rejected_count"],
            "total_oos": len(oos_eval_texts)
        }
        e6_rows.append([f"Threshold {thresh:.2f}", f"{rejection_rate:.1%}", f"{oos_eval['metrics']['rejected_count']}/{len(oos_eval_texts)}"])

    experiment_results["E6_OOS_Thresholding"] = oos_results
    print_table(["Threshold", "OOS Rejection Rate", "Rejected Count"], e6_rows)

    # -------------------------------------------------------------------------
    # Save Artifacts & Best Model
    # -------------------------------------------------------------------------
    print_section("Menyimpan Artefak Model & Laporan Eksperimen")
    
    # Save Results JSON
    metrics_path = "reports/metrics/experiment_results.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"[SUCCESS] Hasil eksperimen tersimpan di: {metrics_path}")

    # Save Confusion Matrix Plot
    fig_path = "reports/figures/confusion_matrix.png"
    all_preds = best_pipeline.predict(X_eval_clean)
    plot_and_save_confusion_matrix(y_eval, all_preds, output_path=fig_path, title=f"Confusion Matrix ({best_model_name.upper()} - {best_ngram})")
    print(f"[SUCCESS] Confusion matrix plot tersimpan di: {fig_path}")

    # Train final best pipeline on all in-scope data and save
    final_best_pipeline = IntentClassifierPipeline(
        FeatureExtractor(vectorizer_type="tfidf", ngram_range=e3_configs[best_ngram]).vectorizer,
        get_model(best_model_name, random_state=seed)
    )
    final_best_pipeline.fit(X_all_clean.tolist(), y_all.tolist())
    model_output_path = "models/best_model.joblib"
    save_model(final_best_pipeline, model_output_path)
    print(f"[SUCCESS] Model terbaik berhasil disimpan di: {model_output_path}")
    print("\n[SUCCESS] SELURUH RANGKAIAN EKSPERIMEN E0-E6 SELESAI DENGAN SUKSES!")


def main():
    parser = argparse.ArgumentParser(description="Runner Eksperimen Intent Classification NLP")
    parser.add_argument("--data", default="data/raw/dataset.csv", help="Path ke dataset CSV")
    parser.add_argument("--config", default="config.yaml", help="Path ke config.yaml")
    args = parser.parse_args()

    run_all_experiments(dataset_path=args.data, config_path=args.config)


if __name__ == "__main__":
    main()
