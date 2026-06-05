# =============================================================================
# Program      : 03_exp3_clean_performance.py
# Version      : 1.0
# Description  : Experiment 3 — In-Distribution Detection Performance.
#
#                Evaluates the trained HOS-XGBoost system on the full
#                ASVspoof 2019 LA evaluation partition (clean speech),
#                reports per-system EER across all 17 spoofing systems,
#                extracts XGBoost feature importance to identify which
#                HOS dimensions and cochlear channels drive decisions,
#                and produces a comparison table against published baselines.
#
# INPUT (downloaded from FTP):
#                  hos_features_eval_clean.h5  — 71,237 utterances × 2,496 dims
#                  xgb_model.json              — trained XGBoost model
#                Google Drive:
#                  ASVspoof2019.LA.cm.eval.trl.txt  — eval protocol
#                    Format: SPEAKER_ID  UTT_ID  -  SYSTEM_ID  LABEL
#                    SYSTEM_ID = A01..A19 (spoof) or '-' (bonafide)
#
#                FEATURE LAYOUT (2,496-dim, channel-interleaved, 104 dims/ch):
#                  ch*104 + 0:4   kurtosis fine-structure (mean,var,p10,p90)
#                  ch*104 + 4:8   kurtosis envelope       (mean,var,p10,p90)
#                  ch*104 + 8:104 bispectrum diagonal     (32 lags × 3 stats)
#
#                PIPELINE:
#                  Step 1  Download HDF5 and model from FTP
#                  Step 2  Parse eval protocol → labels + system IDs
#                  Step 3  Load features from HDF5
#                  Step 4  Compute pooled EER and min-tDCF
#                  Step 5  Compute per-system EER (17 spoof systems)
#                  Step 6  Extract XGBoost feature importance (gain metric)
#                          Map top features back to channel / HOS type
#                  Step 7  Generate figures
#                  Step 8  Save JSON and upload all outputs to FTP
#
# OUTPUT FILES (uploaded to FTP PROJECT_DIR):
#                  exp3_clean_performance.json
#                      Pooled EER, min-tDCF, per-system EER,
#                      top-50 feature importances with channel/type labels
#                  fig_03_01_persystem_eer.png
#                      Per-system EER bar chart (17 spoof systems + pooled)
#                  fig_03_02_feature_importance.png
#                      Top-30 features by XGBoost gain, coloured by HOS type
#                  fig_03_03_importance_by_channel.png
#                      Aggregated importance per channel and per HOS type
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
import scipy.stats as st
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
DRIVE_PROTOCOL  = ("/content/drive/MyDrive/datasets/"
                   "ASVspoof2019_LA_cm_protocols/"
                   "ASVspoof2019.LA.cm.eval.trl.txt")
LOCAL_DIR       = "/content/exp3_work"
os.makedirs(LOCAL_DIR, exist_ok=True)

# --- Feature layout (must match 05_exp5_hos_robustness_v01.py) ---
N_CHANNELS      = 24
DIMS_PER_CHAN   = 104
N_FEAT_TOTAL    = N_CHANNELS * DIMS_PER_CHAN   # 2496

# Offsets within each channel block
KURT_FINE_OFFSET = 0    # dims 0–3: mean, var, p10, p90
KURT_ENV_OFFSET  = 4    # dims 4–7: mean, var, p10, p90
BISPEC_OFFSET    = 8    # dims 8–103: 32 lags × 3 stats (mean, p10, p90)
N_BISPEC_LAGS    = 32

# Pooling stat labels
POOL_STATS = ['mean', 'var', 'p10', 'p90']

# Published baseline EERs for comparison table
PUBLISHED_BASELINES = {
    'LFCC-GMM (ASVspoof 2019)'  : 8.09,
    'RawNet2'                   : 4.01,
    'AASIST'                    : 0.83,
    'HOS-XGBoost (this work)'   : None,   # filled from results
}

# --- ERB channel centre frequencies ---
def _hz_from_erb_rate(erb_rate):
    return (10 ** (erb_rate / 21.366) - 1) / 0.004368

def get_channel_centres(n=N_CHANNELS, fmin=80.0, fmax=8000.0):
    erb_min = 21.366 * np.log10(1 + 0.004368 * fmin)
    erb_max = 21.366 * np.log10(1 + 0.004368 * fmax)
    erb_rates = np.linspace(erb_min, erb_max, n)
    return np.array([_hz_from_erb_rate(r) for r in erb_rates])

CHANNEL_CENTRES_HZ = get_channel_centres()

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

C_POOLED  = '#1565C0'
C_SYSTEM  = '#90CAF9'
C_BON     = '#1565C0'

# HOS type colours for feature importance
TYPE_COLORS = {
    'kurt_fine' : '#1565C0',
    'kurt_env'  : '#B71C1C',
    'bispec'    : '#2E7D32',
}

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
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"Downloaded: {remote_name}  ({size_mb:.1f} MB)")

# =============================================================================
# SECTION 2 — Protocol Parsing
# =============================================================================

def parse_eval_protocol(txt_path):
    """
    Returns:
      labels  : {utt_id: int}  0=bonafide, 1=spoof
      systems : {utt_id: str}  system ID e.g. 'A07', '-' for bonafide
    """
    labels, systems = {}, {}
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id    = parts[1]
            system_id = parts[3]   # '-' for bonafide, 'A07' etc for spoof
            label_str = parts[4]
            labels[utt_id]  = 0 if label_str == "bonafide" else 1
            systems[utt_id] = system_id

    bon = sum(1 for v in labels.values() if v == 0)
    spo = sum(1 for v in labels.values() if v == 1)
    unique_systems = sorted(set(v for v in systems.values() if v != '-'))
    print(f"Eval protocol: {len(labels)} utterances  "
          f"(bonafide={bon}, spoof={spo})")
    print(f"Spoof systems: {unique_systems}")
    return labels, systems

# =============================================================================
# SECTION 3 — Load Features
# =============================================================================

def load_features_from_h5(h5_path, labels_dict):
    """
    Loads full 2,496-dim feature vectors from HDF5.
    Returns X (N, 2496), y (N,), utt_ids list.
    """
    X_list, y_list, ids_list = [], [], []
    with h5py.File(h5_path, 'r') as hf:
        utt_ids = list(hf.keys())
        for utt_id in tqdm(utt_ids, desc="Loading features", ncols=80):
            label = labels_dict.get(utt_id, -1)
            if label == -1:
                continue
            feat = hf[utt_id]['feat'][:]
            X_list.append(feat)
            y_list.append(label)
            ids_list.append(utt_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Loaded: {X.shape}  "
          f"(bonafide={(y==0).sum()}, spoof={(y==1).sum()})")
    return X, y, ids_list

# =============================================================================
# SECTION 4 — EER and min-tDCF
# =============================================================================

def compute_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100.0
    return float(eer), float(thresholds[idx])

def compute_min_tdcf(y_true, scores,
                     p_spoof=0.05, c_miss=1.0, c_fa=10.0):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr  = 1 - tpr
    tdcf = c_miss * fnr * p_spoof + c_fa * fpr * (1 - p_spoof)
    return float(np.min(tdcf))

# =============================================================================
# SECTION 5 — Feature Importance Mapping
# =============================================================================

def dim_to_label(dim_idx):
    """
    Maps a flat feature dimension index (0–2495) to a human-readable label.
    Returns (channel_idx, freq_hz, hos_type, stat_label, full_label).

    Layout per channel (104 dims):
      0–3   : kurt_fine  (mean, var, p10, p90)
      4–7   : kurt_env   (mean, var, p10, p90)
      8–39  : bispec_mean  (lags 0–31)
      40–71 : bispec_p10   (lags 0–31)
      72–103: bispec_p90   (lags 0–31)
    """
    ch        = dim_idx // DIMS_PER_CHAN
    offset    = dim_idx  % DIMS_PER_CHAN
    freq_hz   = float(CHANNEL_CENTRES_HZ[ch])

    if offset < 4:
        hos_type   = 'kurt_fine'
        stat_label = f"kurt_fine_{POOL_STATS[offset]}"
    elif offset < 8:
        hos_type   = 'kurt_env'
        stat_label = f"kurt_env_{POOL_STATS[offset - 4]}"
    else:
        # Bispectrum: 32 mean lags, then 32 p10, then 32 p90
        b_offset = offset - 8      # 0–95
        lag_idx  = b_offset % N_BISPEC_LAGS
        stat_idx = b_offset // N_BISPEC_LAGS
        bstat    = ['mean', 'p10', 'p90'][stat_idx]
        hos_type   = 'bispec'
        stat_label = f"bispec_{bstat}_lag{lag_idx}"

    full_label = f"ch{ch}({freq_hz:.0f}Hz)_{stat_label}"
    return ch, freq_hz, hos_type, stat_label, full_label

def get_importance_table(clf, top_n=50):
    """
    Returns list of dicts with feature importance info,
    sorted by gain (descending).
    """
    scores = clf.get_booster().get_score(importance_type='gain')
    # XGBoost names features as 'f0', 'f1', ...
    rows = []
    for fname, gain in scores.items():
        dim_idx = int(fname[1:])   # strip 'f'
        ch, freq, hos_type, stat_label, full_label = dim_to_label(dim_idx)
        rows.append({
            'dim_idx'    : dim_idx,
            'gain'       : float(gain),
            'channel'    : ch,
            'freq_hz'    : round(freq, 1),
            'hos_type'   : hos_type,
            'stat_label' : stat_label,
            'full_label' : full_label,
        })
    rows.sort(key=lambda r: r['gain'], reverse=True)
    return rows[:top_n]

def aggregate_importance_by_channel_and_type(clf):
    """
    Returns:
      by_channel : (24,) array of summed gain per channel
      by_type    : dict {hos_type: summed_gain}
    """
    scores = clf.get_booster().get_score(importance_type='gain')
    by_channel = np.zeros(N_CHANNELS)
    by_type    = {'kurt_fine': 0.0, 'kurt_env': 0.0, 'bispec': 0.0}

    for fname, gain in scores.items():
        dim_idx = int(fname[1:])
        ch, _, hos_type, _, _ = dim_to_label(dim_idx)
        by_channel[ch] += gain
        by_type[hos_type] += gain

    # Normalise to sum to 1
    total = by_channel.sum()
    by_channel /= max(total, 1e-12)
    total_type = sum(by_type.values())
    by_type = {k: v / max(total_type, 1e-12) for k, v in by_type.items()}
    return by_channel, by_type

# =============================================================================
# SECTION 6 — Figures
# =============================================================================

def fig_persystem_eer(per_system_eers, pooled_eer, out_path):
    """
    Figure 1: Per-system EER bar chart.
    Systems sorted by EER ascending. Pooled EER shown as dashed line.
    """
    # Sort by EER
    sorted_items = sorted(per_system_eers.items(), key=lambda x: x[1])
    sys_ids = [s for s, _ in sorted_items]
    eers    = [e for _, e in sorted_items]

    # Colour: highlight systems with EER > pooled
    colors = ['#EF9A9A' if e > pooled_eer else '#90CAF9' for e in eers]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(range(len(sys_ids)), eers, color=colors,
                  edgecolor='white', linewidth=0.5)

    # Annotate bar tops
    for i, (bar, eer_val) in enumerate(zip(bars, eers)):
        ax.text(bar.get_x() + bar.get_width()/2,
                eer_val + 0.3,
                f'{eer_val:.1f}', ha='center', va='bottom',
                fontsize=7.5, color='#1A237E')

    ax.axhline(pooled_eer, color='#1565C0', lw=1.8, ls='--',
               label=f'Pooled EER = {pooled_eer:.2f}%')

    ax.set_xticks(range(len(sys_ids)))
    ax.set_xticklabels(sys_ids, fontsize=9)
    ax.set_xlabel("Spoofing System ID")
    ax.set_ylabel("EER (%)")
    ax.set_title("Per-System EER — ASVspoof 2019 LA Evaluation Set\n"
                 "HOS-XGBoost (blue = below pooled EER, red = above)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(eers) * 1.15)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")


def fig_feature_importance(importance_rows, out_path, top_n=30):
    """
    Figure 2: Top-N features by XGBoost gain, coloured by HOS type.
    """
    rows   = importance_rows[:top_n]
    labels = [r['full_label'] for r in rows]
    gains  = [r['gain']       for r in rows]
    colors = [TYPE_COLORS[r['hos_type']] for r in rows]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars = ax.barh(range(top_n), gains[::-1],
                   color=colors[::-1],
                   edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.set_xlabel("XGBoost Feature Importance (Gain)")
    ax.set_title(f"Top {top_n} Features by Gain — HOS-XGBoost")

    # Legend
    patches = [mpatches.Patch(color=c, label=t)
               for t, c in TYPE_COLORS.items()]
    type_labels = {
        'kurt_fine': 'Kurtosis fine-structure',
        'kurt_env' : 'Kurtosis envelope',
        'bispec'   : 'Bispectrum diagonal',
    }
    patches = [mpatches.Patch(color=TYPE_COLORS[t], label=type_labels[t])
               for t in TYPE_COLORS]
    ax.legend(handles=patches, fontsize=9, loc='lower right')
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")


def fig_importance_by_channel(by_channel, by_type, out_path):
    """
    Figure 3: Two-panel figure.
    Top: Aggregated normalised importance per channel (bar, log-freq x-axis).
    Bottom: Pie chart of importance by HOS type.
    """
    freqs = CHANNEL_CENTRES_HZ

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                              gridspec_kw={'width_ratios': [3, 1]})

    # Panel 1: per-channel importance
    ax = axes[0]
    ax.bar(range(N_CHANNELS), by_channel,
           color=C_BON, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.set_xticks(range(N_CHANNELS))
    freq_labels = [f"{int(round(f/10)*10)}" for f in freqs]
    ax.set_xticklabels(freq_labels, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel("Channel Centre Frequency (Hz)")
    ax.set_ylabel("Normalised Importance (Gain)")
    ax.set_title("Feature Importance by Cochlear Channel")
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Panel 2: pie by HOS type
    ax2 = axes[1]
    type_labels_full = {
        'kurt_fine': 'Kurtosis\nfine-str.',
        'kurt_env' : 'Kurtosis\nenvelope',
        'bispec'   : 'Bispectrum\ndiagonal',
    }
    pie_vals   = [by_type[t] for t in TYPE_COLORS]
    pie_labels = [type_labels_full[t] for t in TYPE_COLORS]
    pie_colors = [TYPE_COLORS[t] for t in TYPE_COLORS]
    wedges, texts, autotexts = ax2.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 8})
    for at in autotexts:
        at.set_fontsize(8)
    ax2.set_title("By HOS Type", fontsize=10)

    fig.suptitle("XGBoost Feature Importance Analysis",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")

# =============================================================================
# SECTION 7 — Main
# =============================================================================

def main():
    print("=" * 65)
    print("03_exp3_clean_performance_v01.py")
    print("Experiment 3 — In-Distribution Detection Performance")
    print("=" * 65)

    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # ------------------------------------------------------------------
    # Step 1: Download files from FTP
    # ------------------------------------------------------------------
    h5_local    = os.path.join(LOCAL_DIR, "hos_features_eval_clean.h5")
    model_local = os.path.join(LOCAL_DIR, "xgb_model.json")

    download_from_ftp("hos_features_eval_clean.h5", h5_local,
                      validate_h5=True)
    download_from_ftp("xgb_model.json", model_local)

    # ------------------------------------------------------------------
    # Step 2: Parse eval protocol
    # ------------------------------------------------------------------
    labels, systems = parse_eval_protocol(DRIVE_PROTOCOL)

    # ------------------------------------------------------------------
    # Step 3: Load features
    # ------------------------------------------------------------------
    print("\nLoading features from HDF5 ...")
    X, y, utt_ids = load_features_from_h5(h5_local, labels)

    # Build system-ID array aligned with X
    sys_ids_aligned = [systems.get(uid, '-') for uid in utt_ids]

    # ------------------------------------------------------------------
    # Step 4: Load model and score
    # ------------------------------------------------------------------
    print("\nLoading XGBoost model ...")
    clf = xgb.XGBClassifier()
    clf.load_model(model_local)
    print(f"Model loaded. Features expected: {clf.n_features_in_}")

    print("Scoring eval set ...")
    scores = clf.predict_proba(X)[:, 1]   # P(spoof)

    # ------------------------------------------------------------------
    # Step 5: Pooled EER and min-tDCF
    # ------------------------------------------------------------------
    eer_pooled, threshold = compute_eer(y, scores)
    tdcf_pooled           = compute_min_tdcf(y, scores)

    print(f"\n{'='*65}")
    print(f"POOLED RESULTS")
    print(f"{'='*65}")
    print(f"  EER         : {eer_pooled:.4f}%")
    print(f"  min-tDCF    : {tdcf_pooled:.6f}")
    print(f"  Threshold   : {threshold:.4f}")

    # ------------------------------------------------------------------
    # Step 6: Per-system EER
    # ------------------------------------------------------------------
    print(f"\nPer-system EER:")
    unique_sys = sorted(set(s for s in sys_ids_aligned if s != '-'))
    per_system_eers = {}

    print(f"  {'System':<8}  {'N_utts':>7}  {'EER (%)':>9}")
    print("  " + "-" * 30)

    for sys_id in unique_sys:
        # Indices for this system + all bonafide
        mask = np.array([
            (s == sys_id or labels.get(uid, -1) == 0)
            for s, uid in zip(sys_ids_aligned, utt_ids)
        ])
        y_sys  = y[mask]
        sc_sys = scores[mask]
        if len(np.unique(y_sys)) < 2:
            continue
        eer_sys, _ = compute_eer(y_sys, sc_sys)
        per_system_eers[sys_id] = round(eer_sys, 3)
        marker = " ←" if eer_sys > eer_pooled else ""
        print(f"  {sys_id:<8}  {mask.sum():>7}  {eer_sys:>8.2f}%{marker}")

    # Summary
    best_sys  = min(per_system_eers, key=per_system_eers.get)
    worst_sys = max(per_system_eers, key=per_system_eers.get)
    print(f"\n  Best  system: {best_sys}  ({per_system_eers[best_sys]:.2f}%)")
    print(f"  Worst system: {worst_sys}  ({per_system_eers[worst_sys]:.2f}%)")

    # ------------------------------------------------------------------
    # Step 7: Feature importance
    # ------------------------------------------------------------------
    print("\nExtracting feature importance ...")
    importance_rows        = get_importance_table(clf, top_n=50)
    by_channel, by_type    = aggregate_importance_by_channel_and_type(clf)

    print(f"\nTop-10 features by gain:")
    print(f"  {'Dim':>5}  {'Gain':>10}  {'Ch':>3}  {'Freq':>6}  "
          f"{'Type':<12}  Stat")
    print("  " + "-" * 60)
    for r in importance_rows[:10]:
        print(f"  {r['dim_idx']:>5}  {r['gain']:>10.2f}  "
              f"{r['channel']:>3}  {r['freq_hz']:>6.0f}  "
              f"{r['hos_type']:<12}  {r['stat_label']}")

    print(f"\nImportance by HOS type (normalised):")
    for t, v in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:<15} : {v*100:.1f}%")

    top_ch = int(np.argmax(by_channel))
    print(f"\nMost important channel: {top_ch} "
          f"({CHANNEL_CENTRES_HZ[top_ch]:.0f} Hz)  "
          f"importance={by_channel[top_ch]*100:.1f}%")

    # ------------------------------------------------------------------
    # Step 8: Save results JSON
    # ------------------------------------------------------------------
    PUBLISHED_BASELINES['HOS-XGBoost (this work)'] = round(eer_pooled, 4)

    results = {
        'experiment'       : 'Exp3_CleanPerformance',
        'n_utterances'     : int(len(y)),
        'n_bonafide'       : int((y == 0).sum()),
        'n_spoof'          : int((y == 1).sum()),
        'pooled_eer'       : round(eer_pooled, 4),
        'min_tdcf'         : round(tdcf_pooled, 6),
        'threshold'        : round(float(threshold), 4),
        'per_system_eer'   : per_system_eers,
        'best_system'      : best_sys,
        'worst_system'     : worst_sys,
        'published_baselines': PUBLISHED_BASELINES,
        'top50_importance' : importance_rows,
        'importance_by_type': {k: round(v, 4) for k, v in by_type.items()},
        'importance_by_channel': [round(float(v), 4)
                                   for v in by_channel],
    }

    json_path = os.path.join(LOCAL_DIR, "exp3_clean_performance.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: exp3_clean_performance.json")

    # ------------------------------------------------------------------
    # Step 9: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures ...")

    fig_persystem_eer(
        per_system_eers, eer_pooled,
        os.path.join(LOCAL_DIR, "fig_03_01_persystem_eer.png"))

    fig_feature_importance(
        importance_rows,
        os.path.join(LOCAL_DIR, "fig_03_02_feature_importance.png"),
        top_n=30)

    fig_importance_by_channel(
        by_channel, by_type,
        os.path.join(LOCAL_DIR, "fig_03_03_importance_by_channel.png"))

    # ------------------------------------------------------------------
    # Step 10: Upload all to FTP
    # ------------------------------------------------------------------
    print("\nUploading to FTP ...")
    for fname in [
        "exp3_clean_performance.json",
        "fig_03_01_persystem_eer.png",
        "fig_03_02_feature_importance.png",
        "fig_03_03_importance_by_channel.png",
    ]:
        upload(os.path.join(LOCAL_DIR, fname), fname)

    print("\n" + "=" * 65)
    print("Experiment 3 complete.")
    print(f"  Pooled EER  : {eer_pooled:.4f}%")
    print(f"  min-tDCF    : {tdcf_pooled:.6f}")
    print(f"  Best system : {best_sys}  ({per_system_eers[best_sys]:.2f}%)")
    print(f"  Worst system: {worst_sys}  ({per_system_eers[worst_sys]:.2f}%)")
    print(f"\nComparison:")
    for name, val in PUBLISHED_BASELINES.items():
        marker = " ← this work" if "this work" in name else ""
        print(f"  {name:<35} {val:>6.2f}%{marker}")
    print("=" * 65)


# =============================================================================
if __name__ == "__main__":
    main()