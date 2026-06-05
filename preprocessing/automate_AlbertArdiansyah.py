"""
automate_Nama-siswa.py
======================
Script otomatisasi preprocessing dataset Telco Customer Churn.
Menjalankan seluruh tahapan preprocessing secara otomatis dan
menyimpan hasil ke folder telcocustomerchurn_preprocessing/.

Sumber dataset:
    https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Penggunaan:
    python automate_Nama-siswa.py

Output:
    telcocustomerchurn_preprocessing/
    ├── telcocustomerchurn_preprocessing.csv
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    └── y_test.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Konstanta ──────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

RAW_PATH        = "telcocustomerchurn_raw.csv"
OUTPUT_DIR      = "telcocustomerchurn_preprocessing"
OUTPUT_FULL     = os.path.join(OUTPUT_DIR, "telcocustomerchurn_preprocessing.csv")
OUTPUT_X_TRAIN  = os.path.join(OUTPUT_DIR, "X_train.csv")
OUTPUT_X_TEST   = os.path.join(OUTPUT_DIR, "X_test.csv")
OUTPUT_Y_TRAIN  = os.path.join(OUTPUT_DIR, "y_train.csv")
OUTPUT_Y_TEST   = os.path.join(OUTPUT_DIR, "y_test.csv")

RANDOM_STATE    = 42
TEST_SIZE       = 0.2

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
]

MULTI_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

SCALE_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ── Fungsi-fungsi Preprocessing ────────────────────────────────────────────

def load_data(url: str) -> pd.DataFrame:
    """Memuat dataset dari URL dan menyimpan salinan raw-nya."""
    print(f"[1/6] Memuat dataset dari:\n      {url}")
    df = pd.read_csv(url)
    df.to_csv(RAW_PATH, index=False)
    print(f"      ✅ {df.shape[0]} baris x {df.shape[1]} kolom dimuat.")
    print(f"      ✅ Dataset raw disimpan → {RAW_PATH}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values:
      - Hapus kolom customerID.
      - Konversi TotalCharges ke numerik (string kosong → NaN).
      - Imputasi NaN TotalCharges dengan median.
    """
    print("[2/6] Menangani missing values …")
    df = df.copy()

    # Hapus kolom ID
    df.drop(columns=["customerID"], inplace=True)

    # Konversi TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isnull().sum()

    # Imputasi median
    median_val = df["TotalCharges"].median()
    df["TotalCharges"].fillna(median_val, inplace=True)

    print(f"      ✅ customerID dihapus.")
    print(f"      ✅ TotalCharges: {n_missing} nilai kosong diimputasi dengan median ({median_val:.2f}).")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding fitur kategorik:
      - Label Encoding untuk kolom biner.
      - One-Hot Encoding untuk kolom multi-kategori.
    """
    print("[3/6] Encoding fitur kategorik …")
    df = df.copy()

    # Label Encoding
    le = LabelEncoder()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    print(f"      ✅ Label Encoding diterapkan pada {len(BINARY_COLS)} kolom biner.")

    # One-Hot Encoding
    existing_multi = [c for c in MULTI_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_multi, drop_first=False)
    print(f"      ✅ One-Hot Encoding diterapkan pada {len(existing_multi)} kolom multi-kategori.")
    print(f"      ✅ Jumlah kolom setelah encoding: {df.shape[1]}")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan StandardScaler pada kolom numerik kontinyu."""
    print("[4/6] Feature scaling (StandardScaler) …")
    df = df.copy()
    scaler = StandardScaler()
    existing_scale = [c for c in SCALE_COLS if c in df.columns]
    df[existing_scale] = scaler.fit_transform(df[existing_scale])
    print(f"      ✅ StandardScaler diterapkan pada: {existing_scale}")
    return df


def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """Memisahkan fitur dan label, lalu melakukan train-test split (stratified)."""
    print("[5/6] Train-test split …")
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"      ✅ Training : {X_train.shape[0]} sampel ({(1-test_size)*100:.0f}%)")
    print(f"      ✅ Testing  : {X_test.shape[0]} sampel ({test_size*100:.0f}%)")
    return X_train, X_test, y_train, y_test


def save_outputs(
    df_full: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Menyimpan semua hasil ke folder output."""
    print(f"[6/6] Menyimpan output ke folder '{OUTPUT_DIR}/' …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_full.to_csv(OUTPUT_FULL, index=False)
    X_train.to_csv(OUTPUT_X_TRAIN, index=False)
    X_test.to_csv(OUTPUT_X_TEST, index=False)
    y_train.to_csv(OUTPUT_Y_TRAIN, index=False)
    y_test.to_csv(OUTPUT_Y_TEST, index=False)

    print(f"      ✅ {OUTPUT_FULL}")
    print(f"      ✅ {OUTPUT_X_TRAIN}")
    print(f"      ✅ {OUTPUT_X_TEST}")
    print(f"      ✅ {OUTPUT_Y_TRAIN}")
    print(f"      ✅ {OUTPUT_Y_TEST}")


# ── Pipeline Utama ──────────────────────────────────────────────────────────

def run_preprocessing(url: str = DATASET_URL) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan.

    Returns
    -------
    df_clean : pd.DataFrame
        Dataset yang sudah diproses dan siap dilatih.
    """
    print("=" * 55)
    print("   AUTOMATE PREPROCESSING - Telco Customer Churn")
    print("=" * 55)

    df_raw            = load_data(url)
    df_no_missing     = handle_missing_values(df_raw)
    df_encoded        = encode_features(df_no_missing)
    df_scaled         = scale_features(df_encoded)
    X_train, X_test, y_train, y_test = split_data(df_scaled)
    save_outputs(df_scaled, X_train, X_test, y_train, y_test)

    print()
    print("=" * 55)
    print("   PREPROCESSING SELESAI ✅")
    print(f"   Dataset siap latih: {df_scaled.shape[0]} baris x {df_scaled.shape[1]} kolom")
    print("=" * 55)
    return df_scaled


# ── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_preprocessing()
