# =============================================================================
# Program      : 04_exp4_generalisation.py
# Version      : 1.0
# Description  : Experiment 4 — Zero-Shot Generalisation to Unseen Vocoders.
#
#                Evaluates the HOS-XGBoost system trained on ASVspoof 2019 LA
#                (no retraining) on LibriSeVoc mini, which contains 500
#                ground-truth utterances and 500 utterances each from six
#                neural vocoders entirely unseen during training:
#                DiffWave, MelGAN, Parallel WaveGAN, WaveGrad,
#                WaveNet, WaveRNN.
#
# INPUT:
#                  Google Drive (downloaded):
#                    xgb_model.json              — trained HOS-XGBoost model
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/LibriSeVoc_mini.zip
#
#                DATASET STRUCTURE (inside zip):
#                  LibriSeVoc/gt/                — 500 bonafide wav files
#                  LibriSeVoc/diffwave/          — 500 spoof wav files
#                  LibriSeVoc/melgan/            — 500 spoof wav files
#                  LibriSeVoc/parallel_wave_gan/ — 500 spoof wav files
#                  LibriSeVoc/wavegrad/          — 500 spoof wav files
#                  LibriSeVoc/wavenet/           — 500 spoof wav files
#                  LibriSeVoc/wavernn/           — 500 spoof wav files
#                  Total: 3500 wav files
#
#                PIPELINE:
#                  Step 1  Mount Drive; copy zip to local disk; unzip
#                  Step 2  Download xgb_model.json from Google Drive
#                  Step 3  Extract HOS features for all 3500 utterances
#                          (saved to HDF5 with close→upload→reopen checkpointing)
#                  Step 4  Compute pooled EER across all vocoders
#                  Step 5  Compute per-vocoder EER (each vocoder vs gt)
#                  Step 6  Compare against ASVspoof 2019 LA per-system EER
#                          to assess generalisation gap
#                  Step 7  Generate figures and save JSON
#                  Step 8  Upload all outputs to Google Drive
#
# OUTPUT FILES (uploaded to Google Drive PROJECT_DIR):
#                  exp4_generalisation.json
#                      Per-vocoder EER, pooled EER, generalisation gap
#                  fig_04_01_pervocoder_eer.png
#                      Per-vocoder EER bar chart with ASVspoof comparison
#                  fig_04_02_generalisation_scatter.png
#                      Scatter: ASVspoof per-system EER vs LibriSeVoc
#                      per-vocoder EER showing generalisation profile
#
# GPU Required : NO
# Dependencies : numpy, scipy, matplotlib, h5py, xgboost, soundfile,
#                librosa, tqdm
#
# Change Log   :
#   v1.0  2026-06-03  Initial version
# =============================================================================

# !pip install numpy scipy matplotlib h5py xgboost soundfile librosa tqdm

# =============================================================================
# SECTION 0 — Imports and Configuration
# =============================================================================

import os
import json
import time
import shutil
import zipfile
import warnings
import numpy as np
import scipy.signal as ss
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import xgboost as xgb
import soundfile as sf
import librosa
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Google Drive Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/Subband_Kurtosis/"  # Persistent storage

# --- Paths ---
DRIVE_ZIP      = "/content/drive/MyDrive/datasets/LibriSeVoc_mini.zip"
LOCAL_WORK_DIR = "/content/exp4_work"
LOCAL_AUDIO    = os.path.join(LOCAL_WORK_DIR, "LibriSeVoc")
LOCAL_CKPT     = os.path.join(LOCAL_WORK_DIR, "checkpoints")
LOCAL_OUT      = os.path.join(LOCAL_WORK_DIR, "outputs")

for d in [LOCAL_WORK_DIR, LOCAL_CKPT, LOCAL_OUT]:
    os.makedirs(d, exist_ok=True)

# --- Dataset structure ---
GT_DIR    = "gt"
VOCODER_DIRS = [
    "diffwave",
    "melgan",
    "parallel_wave_gan",
    "wavegrad",
    "wavenet",
    "wavernn",
]

# Human-readable vocoder labels for figures
VOCODER_LABELS = {
    "diffwave"         : "DiffWave",
    "melgan"           : "MelGAN",
    "parallel_wave_gan": "Parallel WaveGAN",
    "wavegrad"         : "WaveGrad",
    "wavenet"          : "WaveNet",
    "wavernn"          : "WaveRNN",
}

# ASVspoof 2019 LA per-system EER from Exp 3 (for comparison scatter)
ASVSPOOF_PERSYS_EER = {
    'A07': 1.54,  'A08': 4.54,  'A09': 5.15,  'A10': 13.54,
    'A11': 18.42, 'A12': 12.98, 'A13': 11.35, 'A14': 7.25,
    'A15': 17.96, 'A16': 4.40,  'A17': 38.32, 'A18': 15.90,
    'A19': 7.23,
}
ASVSPOOF_POOLED_EER = 13.83

# --- Feature extraction config (must match Exp 5) ---
SR              = 16000
N_CHANNELS      = 24
FMIN, FMAX      = 80.0, 8000.0
FRAME_LEN_SEC   = 0.200
HOP_LEN_SEC     = 0.100
BISPEC_LAGS     = 32
H5_UPLOAD_EVERY = 200   # checkpoint every 200 utterances

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

C_LIBRISEVOC = '#1565C0'
C_ASVSPOOF   = '#B71C1C'
C_GT         = '#FF8F00'

# =============================================================================
# SECTION 1 — Google Drive Helpers
# =============================================================================

def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_path, remote_name, retries=3):
    """Copy a local file to Google Drive project folder."""
    ensure_project_dir()
    dest_path = os.path.join(PROJECT_DIR, remote_name)
    for attempt in range(retries):
        try:
            shutil.copy2(local_path, dest_path)
            print(f"  [DRIVE OK] {local_path}  →  {dest_path}")
            return
        except Exception as e:
            print(f"  [DRIVE FAIL] attempt {attempt+1}: {e}")
            time.sleep(2)
    print(f"  [DRIVE FAILED] {remote_name}")

def load_from_drive(remote_name, local_path):
    """Copy a file from Google Drive project folder to local path."""
    if os.path.exists(local_path):
        print(f"Local found: {remote_name}. Skipping download.")
        return
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_name)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_path)
            print(f"  [DRIVE OK] {src_path}  →  {local_path}")
            size_mb = os.path.getsize(local_path) / 1e6
            print(f"Downloaded: {remote_name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  [DRIVE FAIL] copy from {src_path}: {e}")
    else:
        print(f"  [DRIVE MISSING] {src_path} not found")

def list_drive_files():
    """List files in the Google Drive project directory."""
    ensure_project_dir()
    try:
        return [f for f in os.listdir(PROJECT_DIR) if os.path.isfile(os.path.join(PROJECT_DIR, f))]
    except Exception as e:
        print(f"  [DRIVE] Could not list files: {e}")
        return []

# =============================================================================
# SECTION 2 — Dataset Preparation
# =============================================================================

def prepare_dataset():
    """Copy zip from Drive → local, unzip."""
    zip_local = os.path.join(LOCAL_WORK_DIR, "LibriSeVoc_mini.zip")
    marker    = os.path.join(LOCAL_WORK_DIR, ".unzip_done")

    if os.path.exists(marker):
        wavs = list(Path(LOCAL_AUDIO).rglob("*.wav"))
        if wavs:
            print(f"Dataset ready: {len(wavs)} wav files.")
            return
        else:
            print("Marker found but no wav files. Re-unzipping.")
            os.remove(marker)

    if not os.path.exists(zip_local):
        print(f"Copying zip from Drive "
              f"({os.path.getsize(DRIVE_ZIP)/1e6:.0f} MB) ...")
        shutil.copy2(DRIVE_ZIP, zip_local)
        print("Copy complete.")

    print("Unzipping LibriSeVoc_mini.zip ...")
    with zipfile.ZipFile(zip_local, 'r') as z:
        z.extractall(LOCAL_WORK_DIR)
    Path(marker).touch()

    # Verify
    wavs = list(Path(LOCAL_AUDIO).rglob("*.wav"))
    print(f"Unzip complete: {len(wavs)} wav files.")

def build_file_list():
    """
    Returns list of (wav_path, label, vocoder_dir) tuples.
    label: 0=bonafide (gt), 1=spoof (vocoder dirs).
    """
    entries = []

    # Ground truth (bonafide)
    gt_path = os.path.join(LOCAL_AUDIO, GT_DIR)
    gt_files = sorted([f for f in os.listdir(gt_path)
                       if f.endswith('.wav')])
    for fname in gt_files:
        entries.append((os.path.join(gt_path, fname), 0, GT_DIR))

    # Vocoders (spoof)
    for vdir in VOCODER_DIRS:
        vpath = os.path.join(LOCAL_AUDIO, vdir)
        if not os.path.isdir(vpath):
            print(f"  Warning: {vdir} not found at {vpath}")
            continue
        vfiles = sorted([f for f in os.listdir(vpath)
                         if f.endswith('.wav')])
        for fname in vfiles:
            entries.append((os.path.join(vpath, fname), 1, vdir))

    # Summary
    from collections import Counter
    counts = Counter(e[2] for e in entries)
    print(f"File list: {len(entries)} total")
    for d, n in sorted(counts.items()):
        print(f"  {d:<25} {n} files  "
              f"({'bonafide' if d == GT_DIR else 'spoof'})")
    return entries

# =============================================================================
# SECTION 3 — Gammatone Filterbank (identical to Exp 5)
# =============================================================================

def _hz_from_erb_rate(erb_rate):
    return (10 ** (erb_rate / 21.366) - 1) / 0.004368

def make_filterbank(n=N_CHANNELS, fmin=FMIN, fmax=FMAX, sr=SR):
    erb_min = 21.366 * np.log10(1 + 0.004368 * fmin)
    erb_max = 21.366 * np.log10(1 + 0.004368 * fmax)
    erb_rates = np.linspace(erb_min, erb_max, n)
    centres = np.array([_hz_from_erb_rate(r) for r in erb_rates])
    filters = []
    for fc in centres:
        bw    = 1.019 * 24.7 * (4.37 * fc / 1000.0 + 1.0)
        low   = max(50.0,         fc - 0.5 * bw)
        high  = min(sr/2.0 - 100, fc + 0.5 * bw)
        low_n  = np.clip(low  / (sr/2.0), 0.001, 0.999)
        high_n = np.clip(high / (sr/2.0), 0.001, 0.999)
        if low_n >= high_n:
            high_n = min(0.999, low_n + 0.01)
        b, a = ss.butter(4, [low_n, high_n], btype='band')
        filters.append((b, a))
    return filters, centres

_FILTERBANK, _CENTRES = make_filterbank()

# =============================================================================
# SECTION 4 — HOS Feature Extraction (identical to Exp 5)
# =============================================================================

def hilbert_decompose(signal):
    analytic = ss.hilbert(signal)
    envelope = np.abs(analytic)
    fine     = np.real(analytic / (envelope + 1e-12))
    return envelope, fine

def excess_kurtosis(x):
    if len(x) < 4:
        return 0.0
    m, s = np.mean(x), np.std(x)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)

def bispectrum_diagonal_slice(x, n_lags=BISPEC_LAGS):
    n = len(x)
    if n < 2 * n_lags:
        return np.zeros(n_lags)
    nfft   = max(256, 2 * n_lags)
    X      = np.fft.rfft(x, n=nfft)
    n_use  = min(n_lags, len(X) // 2)
    B_diag = np.zeros(n_use, dtype=complex)
    for k in range(n_use):
        if 2 * k < len(X):
            B_diag[k] = X[k] ** 2 * np.conj(X[2 * k])
    result = np.abs(B_diag[:n_lags]) if len(B_diag) >= n_lags \
             else np.pad(np.abs(B_diag), (0, n_lags - len(B_diag)))
    return result

def extract_hos_features(wav_path):
    """
    Loads wav, extracts 2,496-dim HOS feature vector.
    Identical pipeline to Exp 5.
    """
    wav, sr = sf.read(wav_path, dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])

    frame_len = int(FRAME_LEN_SEC * SR)
    hop_len   = int(HOP_LEN_SEC   * SR)
    n_frames  = max(1, (len(wav) - frame_len) // hop_len + 1)

    kurt_fine_frames = np.zeros((N_CHANNELS, n_frames))
    kurt_env_frames  = np.zeros((N_CHANNELS, n_frames))
    bispec_frames    = np.zeros((N_CHANNELS, n_frames, BISPEC_LAGS))

    for ch_idx, (b, a) in enumerate(_FILTERBANK):
        filtered       = ss.lfilter(b, a, wav)
        envelope, fine = hilbert_decompose(filtered)
        for fr in range(n_frames):
            s = fr * hop_len
            e = s + frame_len
            kurt_fine_frames[ch_idx, fr] = excess_kurtosis(fine[s:e])
            kurt_env_frames[ch_idx, fr]  = excess_kurtosis(envelope[s:e])
            bispec_frames[ch_idx, fr]    = bispectrum_diagonal_slice(
                fine[s:e], BISPEC_LAGS)

    def pool_1d(arr):
        return np.array([np.mean(arr), np.var(arr),
                         np.percentile(arr, 10),
                         np.percentile(arr, 90)], dtype=np.float32)

    def pool_bispec(arr):
        return np.concatenate([
            np.mean(arr, axis=0),
            np.percentile(arr, 10, axis=0),
            np.percentile(arr, 90, axis=0)
        ]).astype(np.float32)

    feats = []
    for ch in range(N_CHANNELS):
        feats.append(pool_1d(kurt_fine_frames[ch]))
        feats.append(pool_1d(kurt_env_frames[ch]))
        feats.append(pool_bispec(bispec_frames[ch]))

    return np.concatenate(feats).astype(np.float32)

# =============================================================================
# SECTION 5 — Batch Feature Extraction with HDF5 Checkpointing
# =============================================================================

def extract_all_features(entries, h5_path):
    """
    Extracts HOS features for all entries.
    Resumes from HDF5 checkpoint if it exists on Drive.
    Returns (X, y, vocoder_labels) arrays.
    """
    # Recovery from Drive
    if not os.path.exists(h5_path):
        remote = os.path.basename(h5_path)
        drive_files = list_drive_files()
        if remote in drive_files:
            print(f"Recovering {remote} from Google Drive ...")
            load_from_drive(remote, h5_path)
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as hf:
                    n = len(hf.keys())
                print(f"  Recovered: {n} utterances")

    # Load done IDs
    done_ids = set()
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as hf:
                done_ids = set(hf.keys())
            print(f"Resuming: {len(done_ids)} / {len(entries)} done")
        except Exception:
            print("HDF5 corrupt — starting fresh")
            os.remove(h5_path)

    todo = [(path, label, vdir) for path, label, vdir in entries
            if os.path.basename(path) not in done_ids]
    print(f"To extract: {len(todo)} utterances")

    errors       = 0
    since_upload = 0

    hf = h5py.File(h5_path, 'a')
    try:
        for i, (wav_path, label, vdir) in enumerate(
                tqdm(todo, desc="Features", ncols=90)):
            uid = os.path.basename(wav_path)
            try:
                feat = extract_hos_features(wav_path)
                grp  = hf.create_group(uid)
                grp.create_dataset('feat',   data=feat)
                grp.create_dataset('label',  data=label)
                grp.create_dataset('vocoder',
                                   data=np.bytes_(vdir.encode()))
                since_upload += 1
            except Exception as e:
                errors += 1
                tqdm.write(f"  Skip {uid}: {e}")

            # Safe checkpoint: close → upload → reopen
            if since_upload >= H5_UPLOAD_EVERY:
                hf.flush()
                hf.close()
                remote = os.path.basename(h5_path)
                try:
                    save_to_drive(h5_path, remote)
                    tqdm.write(f"[H5 checkpoint] {remote} — "
                               f"{len(done_ids) + i + 1} utterances")
                except Exception as e:
                    tqdm.write(f"[H5 checkpoint] Drive upload failed: {e}")
                hf = h5py.File(h5_path, 'a')
                since_upload = 0
    finally:
        hf.flush()
        hf.close()

    print(f"Extraction done. Errors: {errors}")

    # Load everything
    X, y, voc = [], [], []
    with h5py.File(h5_path, 'r') as hf:
        for uid in hf.keys():
            X.append(hf[uid]['feat'][:])
            y.append(int(hf[uid]['label'][()]))
            voc.append(hf[uid]['vocoder'][()].decode())

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            voc)

# =============================================================================
# SECTION 6 — EER Computation
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
# SECTION 7 — Figures
# =============================================================================

def fig_persystem_eer(per_vocoder_eers, pooled_eer, out_path):
    """
    Bar chart of per-vocoder EER, sorted ascending.
    Pooled EER shown as dashed line.
    ASVspoof pooled EER shown as dotted reference.
    """
    sorted_items = sorted(per_vocoder_eers.items(), key=lambda x: x[1])
    voc_ids  = [VOCODER_LABELS.get(v, v) for v, _ in sorted_items]
    eers     = [e for _, e in sorted_items]
    colors   = ['#EF9A9A' if e > pooled_eer else '#90CAF9'
                for e in eers]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(voc_ids)), eers, color=colors,
                  edgecolor='white', linewidth=0.5)

    for bar, eer_val in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width()/2,
                eer_val + 0.4,
                f'{eer_val:.1f}', ha='center', va='bottom',
                fontsize=8.5, color='#1A237E')

    ax.axhline(pooled_eer, color=C_LIBRISEVOC, lw=1.8, ls='--',
               label=f'LibriSeVoc pooled EER = {pooled_eer:.2f}%')
    ax.axhline(ASVSPOOF_POOLED_EER, color=C_ASVSPOOF, lw=1.5, ls=':',
               label=f'ASVspoof 2019 LA pooled EER = {ASVSPOOF_POOLED_EER:.2f}%')

    ax.set_xticks(range(len(voc_ids)))
    ax.set_xticklabels(voc_ids, fontsize=9)
    ax.set_ylabel("EER (%)")
    ax.set_title("Per-Vocoder EER — LibriSeVoc Mini (Zero-Shot)\n"
                 "HOS-XGBoost trained on ASVspoof 2019 LA only")
    ax.set_ylim(0, max(eers) * 1.2)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")


def fig_generalisation_scatter(per_vocoder_eers, out_path):
    """
    Scatter: x = ASVspoof per-system EER (known), y = not directly comparable
    so instead: scatter of LibriSeVoc per-vocoder EERs vs vocoder type,
    with ASVspoof pooled and LibriSeVoc pooled as reference lines.
    Annotated with vocoder names.
    """
    vocoder_names = [VOCODER_LABELS.get(v, v)
                     for v in per_vocoder_eers.keys()]
    eers = list(per_vocoder_eers.values())

    # Colour by vocoder family
    family_colors = {
        'DiffWave'          : '#1565C0',   # diffusion
        'WaveGrad'          : '#1565C0',   # diffusion
        'MelGAN'            : '#2E7D32',   # GAN
        'Parallel WaveGAN'  : '#2E7D32',   # GAN
        'WaveNet'           : '#B71C1C',   # autoregressive
        'WaveRNN'           : '#B71C1C',   # autoregressive
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, eer in zip(vocoder_names, eers):
        color = family_colors.get(name, '#757575')
        ax.scatter(eer, 0, s=200, color=color, zorder=3,
                   edgecolors='white', linewidth=0.8)
        ax.annotate(f"{name}\n({eer:.1f}%)",
                    (eer, 0),
                    textcoords='offset points',
                    xytext=(0, 15 if eers.index(eer) % 2 == 0 else -25),
                    ha='center', fontsize=8, color=color)

    pooled = np.mean(eers)
    ax.axvline(pooled, color=C_LIBRISEVOC, lw=1.8, ls='--',
               label=f'LibriSeVoc pooled = {pooled:.2f}%')
    ax.axvline(ASVSPOOF_POOLED_EER, color=C_ASVSPOOF, lw=1.5, ls=':',
               label=f'ASVspoof 2019 LA = {ASVSPOOF_POOLED_EER:.2f}%')

    # Family legend
    family_patches = [
        mpatches.Patch(color='#1565C0', label='Diffusion (DiffWave, WaveGrad)'),
        mpatches.Patch(color='#2E7D32', label='GAN (MelGAN, Parallel WaveGAN)'),
        mpatches.Patch(color='#B71C1C', label='Autoregressive (WaveNet, WaveRNN)'),
    ]
    leg1 = ax.legend(handles=family_patches, loc='upper left',
                     fontsize=8, title='Vocoder family')
    ax.add_artist(leg1)
    ax.legend(fontsize=9, loc='upper right')

    ax.set_xlabel("EER (%)")
    ax.set_title("Zero-Shot Generalisation Profile — LibriSeVoc Mini\n"
                 "HOS-XGBoost trained on ASVspoof 2019 LA")
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlim(0, max(eers) * 1.15)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")

# =============================================================================
# SECTION 8 — Main
# =============================================================================

def main():
    print("=" * 65)
    print("04_exp4_generalisation.py")
    print("Experiment 4 — Zero-Shot Generalisation to Unseen Vocoders")
    print("=" * 65)

    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # ------------------------------------------------------------------
    # Step 1: Prepare dataset
    # ------------------------------------------------------------------
    print("\nStep 1: Preparing LibriSeVoc dataset ...")
    prepare_dataset()
    entries = build_file_list()

    # ------------------------------------------------------------------
    # Step 2: Download XGBoost model from Google Drive
    # ------------------------------------------------------------------
    print("\nStep 2: Downloading model from Google Drive ...")
    model_local = os.path.join(LOCAL_CKPT, "xgb_model.json")
    load_from_drive("xgb_model.json", model_local)

    clf = xgb.XGBClassifier()
    clf.load_model(model_local)
    print(f"Model loaded. Features expected: {clf.n_features_in_}")

    # ------------------------------------------------------------------
    # Step 3: Extract HOS features
    # ------------------------------------------------------------------
    print("\nStep 3: Extracting HOS features ...")
    h5_path = os.path.join(LOCAL_CKPT, "hos_features_librisevoc.h5")
    X, y, voc_labels = extract_all_features(entries, h5_path)
    print(f"Features: {X.shape}  "
          f"(bonafide={(y==0).sum()}, spoof={(y==1).sum()})")

    # Final upload of complete HDF5
    save_to_drive(h5_path, "hos_features_librisevoc.h5")

    # ------------------------------------------------------------------
    # Step 4: Score all utterances
    # ------------------------------------------------------------------
    print("\nScoring ...")
    scores = clf.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Step 5: Pooled EER
    # ------------------------------------------------------------------
    eer_pooled, _  = compute_eer(y, scores)
    tdcf_pooled    = compute_min_tdcf(y, scores)
    print(f"\nPooled EER  : {eer_pooled:.4f}%")
    print(f"min-tDCF    : {tdcf_pooled:.6f}")
    print(f"(ASVspoof 2019 LA pooled EER: {ASVSPOOF_POOLED_EER:.2f}%)")
    print(f"Generalisation gap: "
          f"{eer_pooled - ASVSPOOF_POOLED_EER:+.2f} pp")

    # ------------------------------------------------------------------
    # Step 6: Per-vocoder EER
    # ------------------------------------------------------------------
    print(f"\nPer-vocoder EER (each vocoder vs gt bonafide):")
    print(f"  {'Vocoder':<25}  {'N_spo':>6}  {'EER (%)':>9}")
    print("  " + "-" * 45)

    bon_mask   = (y == 0)
    bon_scores = scores[bon_mask]
    bon_y      = y[bon_mask]

    per_vocoder_eers = {}
    voc_labels_arr   = np.array(voc_labels)

    for vdir in VOCODER_DIRS:
        spo_mask   = np.array([v == vdir for v in voc_labels])
        if not spo_mask.any():
            continue
        spo_scores = scores[spo_mask]
        spo_y      = y[spo_mask]

        # Combine bonafide + this vocoder
        combined_scores = np.concatenate([bon_scores, spo_scores])
        combined_y      = np.concatenate([bon_y,      spo_y])

        eer_voc, _ = compute_eer(combined_y, combined_scores)
        per_vocoder_eers[vdir] = round(eer_voc, 3)

        label = VOCODER_LABELS.get(vdir, vdir)
        print(f"  {label:<25}  {spo_mask.sum():>6}  {eer_voc:>8.2f}%")

    best_voc  = min(per_vocoder_eers, key=per_vocoder_eers.get)
    worst_voc = max(per_vocoder_eers, key=per_vocoder_eers.get)
    print(f"\n  Best  vocoder: {VOCODER_LABELS[best_voc]} "
          f"({per_vocoder_eers[best_voc]:.2f}%)")
    print(f"  Worst vocoder: {VOCODER_LABELS[worst_voc]} "
          f"({per_vocoder_eers[worst_voc]:.2f}%)")

    # ------------------------------------------------------------------
    # Step 7: Save results JSON
    # ------------------------------------------------------------------
    results = {
        'experiment'          : 'Exp4_Generalisation',
        'dataset'             : 'LibriSeVoc_mini',
        'n_utterances'        : int(len(y)),
        'n_bonafide'          : int((y == 0).sum()),
        'n_spoof'             : int((y == 1).sum()),
        'pooled_eer'          : round(eer_pooled, 4),
        'min_tdcf'            : round(tdcf_pooled, 6),
        'asvspoof_pooled_eer' : ASVSPOOF_POOLED_EER,
        'generalisation_gap_pp': round(eer_pooled - ASVSPOOF_POOLED_EER, 3),
        'per_vocoder_eer'     : {
            VOCODER_LABELS.get(k, k): v
            for k, v in per_vocoder_eers.items()
        },
        'best_vocoder'  : VOCODER_LABELS[best_voc],
        'worst_vocoder' : VOCODER_LABELS[worst_voc],
    }

    json_path = os.path.join(LOCAL_OUT, "exp4_generalisation.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: exp4_generalisation.json")

    # ------------------------------------------------------------------
    # Step 8: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures ...")

    fig_persystem_eer(
        per_vocoder_eers, eer_pooled,
        os.path.join(LOCAL_OUT, "fig_04_01_pervocoder_eer.png"))

    fig_generalisation_scatter(
        per_vocoder_eers,
        os.path.join(LOCAL_OUT, "fig_04_02_generalisation_scatter.png"))

    # ------------------------------------------------------------------
    # Step 9: Upload all to Google Drive
    # ------------------------------------------------------------------
    print("\nUploading to Google Drive ...")
    for fname in [
        "exp4_generalisation.json",
        "fig_04_01_pervocoder_eer.png",
        "fig_04_02_generalisation_scatter.png",
    ]:
        save_to_drive(os.path.join(LOCAL_OUT, fname), fname)

    # --- Sync to ensure all writes are flushed ---
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Experiment 4 complete.")
    print(f"  LibriSeVoc pooled EER : {eer_pooled:.2f}%")
    print(f"  ASVspoof pooled EER   : {ASVSPOOF_POOLED_EER:.2f}%")
    print(f"  Generalisation gap    : "
          f"{eer_pooled - ASVSPOOF_POOLED_EER:+.2f} pp")
    print(f"  Best vocoder  : {VOCODER_LABELS[best_voc]}  "
          f"({per_vocoder_eers[best_voc]:.2f}%)")
    print(f"  Worst vocoder : {VOCODER_LABELS[worst_voc]}  "
          f"({per_vocoder_eers[worst_voc]:.2f}%)")
    print("=" * 65)


# =============================================================================
if __name__ == "__main__":
    main()