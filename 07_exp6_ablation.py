# =============================================================================
# Program      : 07_exp6_ablation.py
# Version      : 1.0
# Description  : Experiment 6 — Ablation Study and Feature Contribution.
#
#                Decomposes the contribution of each HOS feature component
#                (fine-structure kurtosis, envelope kurtosis, bispectrum
#                diagonal) to clean and distorted detection performance by
#                training XGBoost on feature subsets and evaluating on the
#                clean eval set and the MP3-32kbps distorted eval set.
#
# INPUT (downloaded from FTP):
#                  hos_features_train.h5          — 25,380 × 2,496
#                  hos_features_eval_clean.h5     — 71,237 × 2,496
#                  hos_features_eval_mp3_032.h5   — 5,000  × 2,496
#                Google Drive:
#                  ASVspoof2019.LA.cm.train.trn.txt
#                  ASVspoof2019.LA.cm.eval.trl.txt
#
#                FEATURE LAYOUT (2,496-dim, 104 dims per channel):
#                  ch*104 + 0:4   kurt_fine  (mean, var, p10, p90)
#                  ch*104 + 4:8   kurt_env   (mean, var, p10, p90)
#                  ch*104 + 8:104 bispec     (32 lags × 3 pooling stats)
#
#                ABLATION VARIANTS (6 subsets + 1 full):
#                  V1: Kurt fine-structure only      (24×4 =   96 dims)
#                  V2: Kurt envelope only            (24×4 =   96 dims)
#                  V3: Kurt fine + envelope combined (24×8 =  192 dims)
#                  V4: Bispectrum diagonal only      (24×96 = 2304 dims)
#                  V5: Kurt fine + bispectrum        (24×100= 2400 dims)
#                  V6: Kurt env  + bispectrum        (24×100= 2400 dims)
#                  V7: Full feature set              (2496 dims) — reference
#
#                PIPELINE:
#                  Step 1  Download HDF5 files from FTP
#                  Step 2  Parse protocols
#                  Step 3  Load all three HDF5 files into memory
#                  Step 4  For each variant: build index mask, subset features,
#                          train XGBoost, evaluate on clean + MP3-32, record EER
#                  Step 5  Generate figures and save JSON
#                  Step 6  Upload all outputs to FTP
#
# OUTPUT FILES (uploaded to FTP PROJECT_DIR):
#                  exp6_ablation.json
#                      Per-variant EER (clean + MP3-32), feature dims,
#                      degradation ratio per variant
#                  fig_06_01_ablation_eer_bar.png
#                      Grouped bar chart: clean EER and MP3-32 EER per variant
#                  fig_06_02_ablation_degradation.png
#                      Degradation ratio (rho) per variant for MP3-32
#                  fig_06_03_ablation_summary_table.png
#                      Visual table of all ablation results
#
# GPU Required : NO
# Dependencies : numpy, scipy, matplotlib, h5py, xgboost, tqdm, ftplib
#
# Change Log   :
#   v1.0  2026-06-03  Initial version
# =============================================================================

# !pip install numpy scipy matplotlib h5py xgboost tqdm

# =============================================================================
# SECTION 0 — Imports and Configuration
# =============================================================================

import os
import json
import time
import ftplib
import warnings
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import xgboost as xgb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- FTP Configuration ---
FTP_HOST        = "173.225.103.246"
FTP_PORT        = 2121
FTP_USER        = "guest"
FTP_PASS        = "guest"
FTP_PROJECT_DIR = "."

# --- Paths ---
DRIVE_TRAIN_PROTO = ("/content/drive/MyDrive/datasets/"
                     "ASVspoof2019_LA_cm_protocols/"
                     "ASVspoof2019.LA.cm.train.trn.txt")
DRIVE_EVAL_PROTO  = ("/content/drive/MyDrive/datasets/"
                     "ASVspoof2019_LA_cm_protocols/"
                     "ASVspoof2019.LA.cm.eval.trl.txt")
LOCAL_DIR         = "/content/exp6_work"
os.makedirs(LOCAL_DIR, exist_ok=True)

# --- Feature layout constants ---
N_CHANNELS      = 24
DIMS_PER_CHAN   = 104
N_FEAT_TOTAL    = N_CHANNELS * DIMS_PER_CHAN   # 2496

# --- Build index arrays for each feature subset ---
# Per channel: dims 0–3 kurt_fine, 4–7 kurt_env, 8–103 bispec
IDX_KURT_FINE = np.array([
    ch * DIMS_PER_CHAN + offset
    for ch in range(N_CHANNELS)
    for offset in range(0, 4)
], dtype=np.int32)   # 96 dims

IDX_KURT_ENV = np.array([
    ch * DIMS_PER_CHAN + offset
    for ch in range(N_CHANNELS)
    for offset in range(4, 8)
], dtype=np.int32)   # 96 dims

IDX_BISPEC = np.array([
    ch * DIMS_PER_CHAN + offset
    for ch in range(N_CHANNELS)
    for offset in range(8, DIMS_PER_CHAN)
], dtype=np.int32)   # 2304 dims

IDX_FULL = np.arange(N_FEAT_TOTAL, dtype=np.int32)   # 2496 dims

# Ablation variants: (name, index_array, n_dims, description)
VARIANTS = [
    ("V1_kurt_fine",
     IDX_KURT_FINE,
     len(IDX_KURT_FINE),
     "Kurtosis fine-structure only"),
    ("V2_kurt_env",
     IDX_KURT_ENV,
     len(IDX_KURT_ENV),
     "Kurtosis envelope only"),
    ("V3_kurt_combined",
     np.concatenate([IDX_KURT_FINE, IDX_KURT_ENV]),
     len(IDX_KURT_FINE) + len(IDX_KURT_ENV),
     "Kurtosis fine + envelope"),
    ("V4_bispec",
     IDX_BISPEC,
     len(IDX_BISPEC),
     "Bispectrum diagonal only"),
    ("V5_fine_bispec",
     np.concatenate([IDX_KURT_FINE, IDX_BISPEC]),
     len(IDX_KURT_FINE) + len(IDX_BISPEC),
     "Kurtosis fine + bispectrum"),
    ("V6_env_bispec",
     np.concatenate([IDX_KURT_ENV, IDX_BISPEC]),
     len(IDX_KURT_ENV) + len(IDX_BISPEC),
     "Kurtosis envelope + bispectrum"),
    ("V7_full",
     IDX_FULL,
     N_FEAT_TOTAL,
     "Full feature set (reference)"),
]

# --- XGBoost hyperparameters (same as Exp 5) ---
XGB_PARAMS = dict(
    n_estimators    = 500,
    max_depth       = 6,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    use_label_encoder=False,
    eval_metric     = 'logloss',
    random_state    = 42,
    n_jobs          = -1,
    tree_method     = 'hist',
)

# --- Plot style ---
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'figure.dpi'       : 300,
})

C_CLEAN = '#1565C0'
C_MP3   = '#B71C1C'
C_REF   = '#FF8F00'   # reference variant (V7 full)

# =============================================================================
# SECTION 1 — FTP Helpers
# =============================================================================

def get_ftp():
    ftp = ftplib.FTP()
    ftp.connect(FTP_HOST, FTP_PORT, timeout=30)
    ftp.login(FTP_USER, FTP_PASS)
    if FTP_PROJECT_DIR != ".":
        ftp.cwd(FTP_PROJECT_DIR)
    return ftp

def upload(local_path, remote_name, retries=3):
    for attempt in range(retries):
        try:
            ftp = get_ftp()
            with open(local_path, "rb") as f:
                ftp.storbinary(f"STOR {remote_name}", f)
            ftp.quit()
            print(f"  FTP ✓ {remote_name}")
            return
        except Exception as e:
            print(f"  FTP attempt {attempt+1} failed: {e}")
            time.sleep(2)
    print(f"  FTP FAILED: {remote_name}")

def download_from_ftp(remote_name, local_path, validate_h5=False):
    if os.path.exists(local_path):
        if validate_h5:
            try:
                with h5py.File(local_path, 'r') as hf:
                    n = len(hf.keys())
                print(f"Local found: {remote_name} ({n} utterances). Skipping.")
                return
            except Exception:
                print(f"Local HDF5 corrupt. Re-downloading {remote_name}.")
                os.remove(local_path)
        else:
            print(f"Local found: {remote_name}. Skipping download.")
            return
    print(f"Downloading {remote_name} from FTP ...")
    ftp = get_ftp()
    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {remote_name}", f.write)
    ftp.quit()
    print(f"Downloaded: {remote_name}  "
          f"({os.path.getsize(local_path)/1e6:.1f} MB)")

# =============================================================================
# SECTION 2 — Protocol Parsing
# =============================================================================

def parse_protocol(txt_path):
    labels = {}
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            labels[parts[1]] = 0 if parts[4] == "bonafide" else 1
    bon = sum(1 for v in labels.values() if v == 0)
    spo = sum(1 for v in labels.values() if v == 1)
    print(f"  {os.path.basename(txt_path)}: "
          f"{len(labels)} utterances (bon={bon}, spo={spo})")
    return labels

# =============================================================================
# SECTION 3 — HDF5 Loading
# =============================================================================

def load_h5(h5_path, labels_dict, desc="Loading"):
    X_list, y_list, ids_list = [], [], []
    with h5py.File(h5_path, 'r') as hf:
        for utt_id in tqdm(hf.keys(), desc=desc, ncols=80):
            label = labels_dict.get(utt_id, -1)
            if label == -1:
                continue
            X_list.append(hf[utt_id]['feat'][:])
            y_list.append(label)
            ids_list.append(utt_id)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"  {desc}: {X.shape}  "
          f"(bon={(y==0).sum()}, spo={(y==1).sum()})")
    return X, y

# =============================================================================
# SECTION 4 — EER / tDCF
# =============================================================================

def compute_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100.0
    return float(eer)

def compute_min_tdcf(y_true, scores,
                     p_spoof=0.05, c_miss=1.0, c_fa=10.0):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr  = 1 - tpr
    tdcf = c_miss * fnr * p_spoof + c_fa * fpr * (1 - p_spoof)
    return float(np.min(tdcf))

# =============================================================================
# SECTION 5 — Train and Evaluate One Variant
# =============================================================================

def run_variant(variant_name, idx_array, n_dims, description,
                X_train, y_train, X_eval_clean, y_eval_clean,
                X_eval_mp3, y_eval_mp3,
                checkpoint_dir):
    """
    Trains XGBoost on feature subset, evaluates on clean and MP3-32.
    Saves/loads model checkpoint to avoid retraining on resume.
    Returns dict of results.
    """
    model_path = os.path.join(checkpoint_dir, f"xgb_{variant_name}.json")

    # Subset features
    X_tr  = X_train[:, idx_array]
    X_cl  = X_eval_clean[:, idx_array]
    X_mp3 = X_eval_mp3[:, idx_array]

    # Train or load
    if os.path.exists(model_path):
        print(f"  [{variant_name}] Loading cached model ...")
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
    else:
        print(f"  [{variant_name}] Training on {n_dims} dims ...")
        n_pos  = int(y_train.sum())
        n_neg  = len(y_train) - n_pos
        scale  = n_neg / max(n_pos, 1)
        clf = xgb.XGBClassifier(
            scale_pos_weight=scale, **XGB_PARAMS)
        clf.fit(X_tr, y_train,
                eval_set=[(X_tr, y_train)],
                verbose=False)
        clf.save_model(model_path)
        # Upload model checkpoint to FTP
        try:
            ftp = get_ftp()
            with open(model_path, "rb") as f:
                ftp.storbinary(f"STOR {os.path.basename(model_path)}", f)
            ftp.quit()
        except Exception as e:
            print(f"    FTP model upload failed: {e}")

    # Score
    sc_clean = clf.predict_proba(X_cl)[:, 1]
    sc_mp3   = clf.predict_proba(X_mp3)[:, 1]

    eer_clean = compute_eer(y_eval_clean, sc_clean)
    eer_mp3   = compute_eer(y_eval_mp3,   sc_mp3)
    tdcf_clean= compute_min_tdcf(y_eval_clean, sc_clean)
    rho_mp3   = eer_mp3 / max(eer_clean, 0.01)

    print(f"  [{variant_name}]  dims={n_dims:>5}  "
          f"EER_clean={eer_clean:>6.2f}%  "
          f"EER_mp3={eer_mp3:>6.2f}%  "
          f"rho={rho_mp3:>5.2f}x")

    return {
        'variant'     : variant_name,
        'description' : description,
        'n_dims'      : n_dims,
        'eer_clean'   : round(eer_clean, 3),
        'eer_mp3_032' : round(eer_mp3, 3),
        'tdcf_clean'  : round(tdcf_clean, 4),
        'rho_mp3_032' : round(rho_mp3, 3),
    }

# =============================================================================
# SECTION 6 — Figures
# =============================================================================

def fig_ablation_eer_bar(results_list, out_path):
    """
    Figure 1: Grouped bar chart — clean EER and MP3-32 EER per variant.
    V7 (full) highlighted in orange as reference.
    """
    names      = [r['variant'].replace('V', 'V').replace('_', '\n', 1)
                  for r in results_list]
    short_names= [r['variant'].split('_')[0] for r in results_list]
    descs      = [r['description'] for r in results_list]
    eer_clean  = [r['eer_clean']   for r in results_list]
    eer_mp3    = [r['eer_mp3_032'] for r in results_list]

    x = np.arange(len(results_list))
    w = 0.35

    # Colour V7 differently as reference
    def bar_color(base_color, i):
        return C_REF if i == len(results_list) - 1 else base_color

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_clean = ax.bar(x - w/2, eer_clean, w,
                        color=[bar_color(C_CLEAN, i)
                               for i in range(len(results_list))],
                        alpha=0.85, edgecolor='white', linewidth=0.5,
                        label='Clean EER')
    bars_mp3   = ax.bar(x + w/2, eer_mp3, w,
                        color=[bar_color(C_MP3, i)
                               for i in range(len(results_list))],
                        alpha=0.85, edgecolor='white', linewidth=0.5,
                        hatch='//', label='MP3-32 EER')

    # Value labels
    for bar, val in zip(bars_clean, eer_clean):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=7.5, color='#1A237E')
    for bar, val in zip(bars_mp3, eer_mp3):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=7.5, color='#B71C1C')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['variant'].split('_')[0]}\n({r['n_dims']}d)"
         for r in results_list],
        fontsize=8)
    ax.set_ylabel("EER (%)")
    ax.set_title("Ablation Study — EER per Feature Subset\n"
                 "(Clean and MP3-32 kbps; V7 = full feature set reference)")
    ax.set_ylim(0, max(eer_mp3) * 1.18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Legend with description mapping
    clean_patch = mpatches.Patch(color=C_CLEAN, alpha=0.85,
                                  label='Clean EER')
    mp3_patch   = mpatches.Patch(color=C_MP3, alpha=0.85,
                                  hatch='//', label='MP3-32 EER')
    ref_patch   = mpatches.Patch(color=C_REF, alpha=0.85,
                                  label='V7: Full set (reference)')
    ax.legend(handles=[clean_patch, mp3_patch, ref_patch],
              fontsize=9, loc='upper left')

    # Variant description annotation below x-axis
    for i, r in enumerate(results_list):
        ax.text(i, -5.5, r['description'],
                ha='center', va='top', fontsize=6.5,
                rotation=15, color='#424242')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")


def fig_ablation_degradation(results_list, out_path):
    """
    Figure 2: Degradation ratio rho = EER_mp3 / EER_clean per variant.
    Dashed line at rho=1. Lower is more robust.
    """
    rhos  = [r['rho_mp3_032'] for r in results_list]
    short = [r['variant'].split('_')[0] for r in results_list]
    colors= [C_REF if i == len(results_list)-1 else '#78909C'
             for i in range(len(results_list))]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(results_list)), rhos,
                  color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2, rho + 0.02,
                f'{rho:.2f}×', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5,
               label='No degradation (1.0×)')

    ax.set_xticks(range(len(results_list)))
    ax.set_xticklabels(
        [f"{r['variant'].split('_')[0]}\n({r['n_dims']}d)"
         for r in results_list],
        fontsize=8)
    ax.set_ylabel(r"Degradation Ratio $\rho$ "
                  r"(EER$_{\mathrm{MP3}}$ / EER$_{\mathrm{clean}}$)")
    ax.set_title("MP3-32 kbps Robustness per Feature Subset\n"
                 "(Lower $\\rho$ = more robust to codec distortion)")
    ax.set_ylim(0, max(rhos) * 1.2)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")


def fig_ablation_summary_table(results_list, out_path):
    """
    Figure 3: Visual summary table rendered as a matplotlib figure.
    Rows: variants. Columns: dims, EER_clean, EER_mp3, rho.
    Cells colour-coded: green=good, red=poor.
    """
    col_labels = ['Variant', 'Description', 'Dims',
                  'EER Clean (%)', 'EER MP3-32 (%)', 'ρ (MP3-32)']
    rows = []
    for r in results_list:
        rows.append([
            r['variant'].split('_')[0],
            r['description'],
            str(r['n_dims']),
            f"{r['eer_clean']:.2f}",
            f"{r['eer_mp3_032']:.2f}",
            f"{r['rho_mp3_032']:.2f}×",
        ])

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#1565C0')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style data rows — alternate shading, highlight V7 reference
    eer_cleans = [r['eer_clean']   for r in results_list]
    eer_mp3s   = [r['eer_mp3_032'] for r in results_list]
    rhos       = [r['rho_mp3_032'] for r in results_list]

    for i, r in enumerate(results_list):
        row_idx = i + 1
        base_color = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
        if r['variant'] == 'V7_full':
            base_color = '#FFF8E1'   # light amber for reference row

        for j in range(len(col_labels)):
            table[(row_idx, j)].set_facecolor(base_color)

        # Colour EER_clean cell
        ec = r['eer_clean']
        ec_norm = (ec - min(eer_cleans)) / max(
            max(eer_cleans) - min(eer_cleans), 1e-6)
        table[(row_idx, 3)].set_facecolor(
            plt.cm.RdYlGn_r(ec_norm * 0.7 + 0.15))

        # Colour EER_mp3 cell
        em = r['eer_mp3_032']
        em_norm = (em - min(eer_mp3s)) / max(
            max(eer_mp3s) - min(eer_mp3s), 1e-6)
        table[(row_idx, 4)].set_facecolor(
            plt.cm.RdYlGn_r(em_norm * 0.7 + 0.15))

        # Colour rho cell
        rh = r['rho_mp3_032']
        rh_norm = (rh - min(rhos)) / max(max(rhos) - min(rhos), 1e-6)
        table[(row_idx, 5)].set_facecolor(
            plt.cm.RdYlGn_r(rh_norm * 0.7 + 0.15))

    ax.set_title("Ablation Study — Summary Table\n"
                 "(Green = better performance / lower degradation; "
                 "Red = worse)",
                 fontsize=10, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")

# =============================================================================
# SECTION 7 — Main
# =============================================================================

def main():
    print("=" * 65)
    print("06_exp6_ablation_v01.py")
    print("Experiment 6 — Ablation Study and Feature Contribution")
    print("=" * 65)

    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # ------------------------------------------------------------------
    # Step 1: Download HDF5 files from FTP
    # ------------------------------------------------------------------
    h5_train = os.path.join(LOCAL_DIR, "hos_features_train.h5")
    h5_clean = os.path.join(LOCAL_DIR, "hos_features_eval_clean.h5")
    h5_mp3   = os.path.join(LOCAL_DIR, "hos_features_eval_mp3_032.h5")
    ckpt_dir = os.path.join(LOCAL_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    download_from_ftp("hos_features_train.h5",        h5_train, validate_h5=True)
    download_from_ftp("hos_features_eval_clean.h5",   h5_clean, validate_h5=True)
    download_from_ftp("hos_features_eval_mp3_032.h5", h5_mp3,   validate_h5=True)

    # ------------------------------------------------------------------
    # Step 2: Parse protocols
    # ------------------------------------------------------------------
    print("\nParsing protocols ...")
    train_labels = parse_protocol(DRIVE_TRAIN_PROTO)
    eval_labels  = parse_protocol(DRIVE_EVAL_PROTO)

    # ------------------------------------------------------------------
    # Step 3: Load all three HDF5 files into memory
    # ------------------------------------------------------------------
    print("\nLoading train features ...")
    X_train, y_train = load_h5(h5_train, train_labels, "Train")

    print("\nLoading clean eval features ...")
    X_clean, y_clean = load_h5(h5_clean, eval_labels, "Clean eval")

    print("\nLoading MP3-32 eval features ...")
    # MP3-32 HDF5 uses the same utterance IDs as the eval protocol
    X_mp3, y_mp3 = load_h5(h5_mp3, eval_labels, "MP3-32 eval")

    print(f"\nData loaded:")
    print(f"  Train : {X_train.shape}")
    print(f"  Clean : {X_clean.shape}")
    print(f"  MP3-32: {X_mp3.shape}")

    # ------------------------------------------------------------------
    # Step 4: Run ablation variants
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Running ablation variants ...")
    print(f"  {'Variant':<25}  {'Dims':>5}  "
          f"{'EER_clean':>10}  {'EER_mp3':>9}  {'rho':>7}")
    print("  " + "-" * 65)

    # Check for existing results to resume
    results_path = os.path.join(LOCAL_DIR, "exp6_ablation.json")
    done_variants = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        done_variants = {r['variant']: r for r in saved.get('variants', [])}
        print(f"  Resuming: {len(done_variants)} variants already done.")

    all_results = []
    for (vname, idx_arr, n_dims, desc) in VARIANTS:
        if vname in done_variants:
            r = done_variants[vname]
            print(f"  [{vname}] Cached — "
                  f"EER_clean={r['eer_clean']:.2f}%  "
                  f"EER_mp3={r['eer_mp3_032']:.2f}%  "
                  f"rho={r['rho_mp3_032']:.2f}x")
            all_results.append(r)
            continue

        result = run_variant(
            vname, idx_arr, n_dims, desc,
            X_train, y_train,
            X_clean, y_clean,
            X_mp3,   y_mp3,
            ckpt_dir)
        all_results.append(result)

        # Save progress after each variant
        with open(results_path, 'w') as f:
            json.dump({'experiment': 'Exp6_Ablation',
                       'variants': all_results}, f, indent=2)
        upload(results_path, "exp6_ablation.json")

    # ------------------------------------------------------------------
    # Step 5: Print final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("ABLATION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Variant':<25}  {'Dims':>5}  "
          f"{'EER_clean':>10}  {'EER_mp3':>9}  {'rho':>7}")
    print("  " + "-" * 65)
    for r in all_results:
        marker = " ← ref" if r['variant'] == 'V7_full' else ""
        print(f"  {r['description']:<25}  {r['n_dims']:>5}  "
              f"{r['eer_clean']:>9.2f}%  "
              f"{r['eer_mp3_032']:>8.2f}%  "
              f"{r['rho_mp3_032']:>6.2f}×{marker}")

    # Best clean EER, best robustness
    best_clean = min(all_results[:-1],   # exclude V7 from best-subset search
                     key=lambda r: r['eer_clean'])
    best_robust = min(all_results[:-1],
                      key=lambda r: r['rho_mp3_032'])
    print(f"\n  Best clean EER (subset):     "
          f"{best_clean['variant']} ({best_clean['eer_clean']:.2f}%)")
    print(f"  Most robust subset (MP3-32): "
          f"{best_robust['variant']} (rho={best_robust['rho_mp3_032']:.2f}×)")

    # ------------------------------------------------------------------
    # Step 6: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures ...")

    fig_ablation_eer_bar(
        all_results,
        os.path.join(LOCAL_DIR, "fig_06_01_ablation_eer_bar.png"))

    fig_ablation_degradation(
        all_results,
        os.path.join(LOCAL_DIR, "fig_06_02_ablation_degradation.png"))

    fig_ablation_summary_table(
        all_results,
        os.path.join(LOCAL_DIR, "fig_06_03_ablation_summary_table.png"))

    # ------------------------------------------------------------------
    # Step 7: Upload all outputs to FTP
    # ------------------------------------------------------------------
    print("\nUploading to FTP ...")
    for fname in [
        "exp6_ablation.json",
        "fig_06_01_ablation_eer_bar.png",
        "fig_06_02_ablation_degradation.png",
        "fig_06_03_ablation_summary_table.png",
    ]:
        upload(os.path.join(LOCAL_DIR, fname), fname)

    print("\n" + "=" * 65)
    print("Experiment 6 complete.")
    print("=" * 65)


# =============================================================================
if __name__ == "__main__":
    main()