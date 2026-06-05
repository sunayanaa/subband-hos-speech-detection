# =============================================================================
# Program      : 02_exp2_bispectrum_validation.py
# Version      : 1.0
# Description  : Experiment 2 — Bispectrum Estimation and Bicoherence Analysis.
#
#                Validates that real speech exhibits non-zero quadratic phase
#                coupling (bicoherence) while synthesised speech approaches
#                zero, confirming the theoretical basis of the bispectrum
#                diagonal feature used in the HOS detector.
#
# INPUT:
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/ASVspoof-2019-LA.zip
#                    /content/drive/MyDrive/datasets/
#                      ASVspoof2019_LA_cm_protocols/
#                        ASVspoof2019.LA.cm.eval.trl.txt
#
#                PIPELINE:
#                  Step 1  Mount Drive; copy zip to local disk; unzip if needed
#                  Step 2  Parse eval protocol; select 100 bonafide + 100 spoof
#                          utterances stratified across all 13 eval systems
#                  Step 3  For a representative mid-frequency channel (~1 kHz):
#                          compute the full 2D bispectrum magnitude via the
#                          indirect method (3rd-order cumulant → 2D FFT,
#                          lag window L=64) for both classes
#                  Step 4  Compute bicoherence index per channel (all 24
#                          gammatone channels) for all 200 utterances
#                  Step 5  Estimate variance of the direct diagonal bispectrum
#                          estimate vs frame length (128, 256, 512 ms) to
#                          justify the 200 ms design choice
#                  Step 6  Generate figures; save JSON; upload to FTP
#
# OUTPUT FILES (uploaded to FTP PROJECT_DIR):
#                  exp2_bispectrum.json
#                      Per-channel bicoherence indices per class per system,
#                      frame-length variance analysis results
#                  fig_02_01_bispectrum_bonafide.png
#                      2D bispectrum magnitude (indirect method, ~1 kHz ch),
#                      averaged over 100 bonafide utterances
#                  fig_02_02_bispectrum_spoof.png
#                      2D bispectrum magnitude, averaged over 100 spoof
#                  fig_02_03_bicoherence_boxplot.png
#                      Per-channel bicoherence index, bonafide vs spoof,
#                      all 24 channels, coloured by synthesis system group
#                  fig_02_04_frame_length_variance.png
#                      Estimation variance of diagonal bispectrum slice
#                      vs frame length (128/256/512 ms), justifies 200 ms
#
# GPU Required : NO
# Dependencies : numpy, scipy, matplotlib, soundfile, librosa, tqdm, ftplib
#
# Change Log   :
#   v1.0  2026-06-03  Initial version
# =============================================================================

# !pip install numpy scipy matplotlib soundfile librosa tqdm

# =============================================================================
# SECTION 0 — Imports and Configuration
# =============================================================================

import os
import json
import time
import ftplib
import shutil
import zipfile
import warnings
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.signal as ss
from scipy.fft import dct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- FTP Configuration ---
FTP_HOST        = "173.225.103.246"
FTP_PORT        = 2121
FTP_USER        = "guest"
FTP_PASS        = "guest"
FTP_PROJECT_DIR = "."

# --- Paths ---
DRIVE_DIR       = "/content/drive/MyDrive/datasets"
DRIVE_ZIP       = os.path.join(DRIVE_DIR, "ASVspoof-2019-LA.zip")
DRIVE_PROTOCOL  = os.path.join(DRIVE_DIR,
                   "ASVspoof2019_LA_cm_protocols",
                   "ASVspoof2019.LA.cm.eval.trl.txt")
LOCAL_WORK_DIR  = "/content/exp2_work"
LOCAL_AUDIO_DIR = os.path.join(LOCAL_WORK_DIR, "audio")
LOCAL_OUT_DIR   = os.path.join(LOCAL_WORK_DIR, "outputs")

for d in [LOCAL_WORK_DIR, LOCAL_OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Analysis configuration ---
SR              = 16000
N_CHANNELS      = 24
FMIN, FMAX      = 80.0, 8000.0
N_SAMPLES_PER_CLASS = 100      # 100 bonafide + 100 spoof
RANDOM_SEED         = 42

# Mid-frequency channel for 2D bispectrum (~1 kHz)
TARGET_FREQ_HZ  = 1000.0

# Indirect bispectrum parameters
CUMULANT_LAG    = 64     # lag window length for 3rd-order cumulant
BISPECTRUM_NFFT = 128    # 2D FFT size for bispectrum

# Direct diagonal bispectrum parameters
BISPEC_LAGS     = 32

# Frame lengths to compare (in seconds)
FRAME_LENGTHS_SEC = [0.128, 0.200, 0.256, 0.512]
HOP_LEN_SEC       = 0.100

# Synthesis system groupings for colour coding in figures
SYSTEM_GROUPS = {
    'statistical_parametric': ['A07', 'A08', 'A09', 'A10'],
    'waveform_concat'        : ['A11', 'A12', 'A13'],
    'neural_waveform'        : ['A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
}
GROUP_COLORS = {
    'statistical_parametric': '#1565C0',
    'waveform_concat'        : '#2E7D32',
    'neural_waveform'        : '#B71C1C',
    'bonafide'               : '#FF8F00',
}

# --- Plot style ---
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'figure.dpi'       : 300,
})

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

# =============================================================================
# SECTION 2 — Dataset Preparation
# =============================================================================

def prepare_audio():
    """
    Ensures ASVspoof 2019 LA eval audio is available locally.
    Copies zip from Drive → local, unzips, returns eval flac directory.
    """
    zip_local = os.path.join(LOCAL_WORK_DIR, "ASVspoof-2019-LA.zip")
    marker    = os.path.join(LOCAL_WORK_DIR, ".unzip_done")

    # Check if audio already unzipped and present
    if os.path.exists(marker):
        result = os.popen(
            f"find {LOCAL_WORK_DIR} -name '*.flac' -maxdepth 8 | head -1"
        ).read().strip()
        if result:
            print(f"Audio already present: {result[:60]}...")
        else:
            print("Marker found but no flac files. Re-unzipping.")
            os.remove(marker)

    if not os.path.exists(marker):
        # Copy zip from Drive
        if not os.path.exists(zip_local):
            print(f"Copying zip from Drive ({os.path.getsize(DRIVE_ZIP)/1e9:.2f} GB)...")
            shutil.copy2(DRIVE_ZIP, zip_local)
            print("Copy complete.")
        else:
            print("Zip already on local disk.")

        # Unzip
        print("Unzipping dataset ...")
        with zipfile.ZipFile(zip_local, 'r') as z:
            z.extractall(LOCAL_WORK_DIR)
        Path(marker).touch()
        print("Unzip complete.")

    # Discover eval flac directory
    for dirpath, dirnames, filenames in os.walk(LOCAL_WORK_DIR):
        if 'eval' in dirpath.lower() and any(f.endswith('.flac')
                                              for f in filenames):
            print(f"Eval audio found: {dirpath}")
            return dirpath

    # Fallback: find any directory with flac files containing LA_E_
    for dirpath, _, filenames in os.walk(LOCAL_WORK_DIR):
        la_e_files = [f for f in filenames
                      if f.startswith('LA_E_') and f.endswith('.flac')]
        if la_e_files:
            print(f"Eval audio found: {dirpath} ({len(la_e_files)} files)")
            return dirpath

    raise RuntimeError("Could not find eval flac directory after unzip.")

# =============================================================================
# SECTION 3 — Protocol Parsing and Utterance Selection
# =============================================================================

def parse_eval_protocol(txt_path):
    """Returns {utt_id: (label, system_id)}."""
    entries = {}
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id    = parts[1]
            system_id = parts[3]   # '-' for bonafide
            label     = 0 if parts[4] == "bonafide" else 1
            entries[utt_id] = (label, system_id)
    bon = sum(1 for v in entries.values() if v[0] == 0)
    spo = sum(1 for v in entries.values() if v[0] == 1)
    systems = sorted(set(v[1] for v in entries.values() if v[1] != '-'))
    print(f"Protocol: {len(entries)} utterances (bon={bon}, spo={spo})")
    print(f"Systems: {systems}")
    return entries

def select_utterances(entries, n_per_class, flac_dir, seed=RANDOM_SEED):
    """
    Select n_per_class bonafide and n_per_class spoof utterances.
    Spoof utterances stratified across all systems (as equal as possible).
    Only selects utterances whose flac file actually exists.
    Returns two lists: bon_ids, spo_ids (each list of utt_id strings).
    """
    rng = random.Random(seed)

    # Bonafide pool
    bon_pool = [uid for uid, (lbl, _) in entries.items() if lbl == 0]
    bon_pool = [uid for uid in bon_pool
                if os.path.exists(os.path.join(flac_dir, uid + ".flac"))]
    rng.shuffle(bon_pool)
    bon_selected = bon_pool[:n_per_class]

    # Spoof pool — stratified by system
    by_system = defaultdict(list)
    for uid, (lbl, sys) in entries.items():
        if lbl == 1:
            fpath = os.path.join(flac_dir, uid + ".flac")
            if os.path.exists(fpath):
                by_system[sys].append(uid)

    systems   = sorted(by_system.keys())
    n_systems = len(systems)
    per_sys   = max(1, n_per_class // n_systems)
    remainder = n_per_class - per_sys * n_systems

    spo_selected = []
    for i, sys in enumerate(systems):
        pool = by_system[sys]
        rng.shuffle(pool)
        take = per_sys + (1 if i < remainder else 0)
        spo_selected.extend(pool[:take])

    spo_selected = spo_selected[:n_per_class]

    print(f"Selected: {len(bon_selected)} bonafide, "
          f"{len(spo_selected)} spoof "
          f"({n_systems} systems, ~{per_sys} each)")
    return bon_selected, spo_selected

# =============================================================================
# SECTION 4 — Gammatone Filterbank
# =============================================================================

def _hz_from_erb_rate(erb_rate):
    return (10 ** (erb_rate / 21.366) - 1) / 0.004368

def get_channel_centres(n=N_CHANNELS, fmin=FMIN, fmax=FMAX):
    erb_min = 21.366 * np.log10(1 + 0.004368 * fmin)
    erb_max = 21.366 * np.log10(1 + 0.004368 * fmax)
    return np.array([_hz_from_erb_rate(r)
                     for r in np.linspace(erb_min, erb_max, n)])

CHANNEL_CENTRES_HZ = get_channel_centres()

def make_filterbank(n=N_CHANNELS, fmin=FMIN, fmax=FMAX, sr=SR):
    centres = get_channel_centres(n, fmin, fmax)
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

_FILTERBANK, _ = make_filterbank()

def get_target_channel(target_hz=TARGET_FREQ_HZ):
    """Return channel index closest to target_hz."""
    return int(np.argmin(np.abs(CHANNEL_CENTRES_HZ - target_hz)))

TARGET_CHANNEL = get_target_channel()
print(f"Target channel: {TARGET_CHANNEL} "
      f"({CHANNEL_CENTRES_HZ[TARGET_CHANNEL]:.0f} Hz)")

# =============================================================================
# SECTION 5 — Audio Loading and Subband Filtering
# =============================================================================

def load_wav(flac_path):
    wav, sr = sf.read(flac_path, dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    # Pre-emphasis
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
    return wav

def get_fine_structure(wav, ch_idx):
    """Filter to channel, return fine-structure via Hilbert."""
    b, a     = _FILTERBANK[ch_idx]
    filtered = ss.lfilter(b, a, wav)
    analytic = ss.hilbert(filtered)
    envelope = np.abs(analytic)
    fine     = np.real(analytic / (envelope + 1e-12))
    return fine, envelope

# =============================================================================
# SECTION 6 — Indirect Bispectrum (full 2D, for visualisation)
# =============================================================================

def compute_indirect_bispectrum_2d(signal, lag=CUMULANT_LAG,
                                    nfft=BISPECTRUM_NFFT):
    """
    Computes the 2D bispectrum magnitude via the indirect method:
      1. Estimate the 3rd-order cumulant c3(t1, t2) for lags |t| <= L
      2. 2D FFT of c3 → bispectrum B(f1, f2)
    Returns magnitude array of shape (nfft, nfft).
    """
    n = len(signal)
    x = signal - np.mean(signal)

    # 3rd-order cumulant estimate
    c3 = np.zeros((2 * lag + 1, 2 * lag + 1))
    for t1 in range(-lag, lag + 1):
        for t2 in range(-lag, lag + 1):
            # c3(t1, t2) = E[x(n) x(n+t1) x(n+t2)]
            i_min = max(0, -t1, -t2)
            i_max = min(n - 1, n - 1 - t1, n - 1 - t2)
            if i_max > i_min:
                c3[t1 + lag, t2 + lag] = np.mean(
                    x[i_min:i_max] *
                    x[i_min + t1:i_max + t1] *
                    x[i_min + t2:i_max + t2]
                )

    # 2D FFT
    B = np.fft.fft2(c3, s=(nfft, nfft))
    return np.abs(B[:nfft//2, :nfft//2])

def compute_class_mean_bispectrum(utt_ids, flac_dir, ch_idx,
                                   n_frames_per_utt=5, desc=""):
    """
    Computes mean 2D bispectrum magnitude across utterances.
    Uses up to n_frames_per_utt random frames per utterance for speed.
    Returns (nfft//2, nfft//2) array.
    """
    frame_len = int(0.200 * SR)   # 200 ms
    accum     = None
    count     = 0

    for uid in tqdm(utt_ids, desc=desc, ncols=80):
        try:
            wav  = load_wav(os.path.join(flac_dir, uid + ".flac"))
            fine, _ = get_fine_structure(wav, ch_idx)
            n_frames = max(1, (len(fine) - frame_len) // (frame_len // 2))
            frame_starts = np.linspace(
                0, max(0, len(fine) - frame_len),
                min(n_frames, n_frames_per_utt), dtype=int)

            for start in frame_starts:
                segment = fine[start:start + frame_len]
                if len(segment) < frame_len:
                    continue
                B = compute_indirect_bispectrum_2d(segment)
                if accum is None:
                    accum = B.copy()
                else:
                    accum += B
                count += 1
        except Exception as e:
            print(f"  Skip {uid}: {e}")

    if accum is None or count == 0:
        return np.zeros((BISPECTRUM_NFFT//2, BISPECTRUM_NFFT//2))
    return accum / count

# =============================================================================
# SECTION 7 — Bicoherence Index
# =============================================================================

def bicoherence_index(signal, n_lags=BISPEC_LAGS, frame_len=None):
    """
    Computes the bicoherence index for a signal using the direct method.
    bic(f) = |E[X(f)^2 X*(2f)]| / E[|X(f)|^2 |X(2f)|]
    Averaged over frames. Returns scalar mean bicoherence across lags.
    """
    if frame_len is None:
        frame_len = int(0.200 * SR)

    hop   = frame_len // 2
    nfft  = max(256, 2 * n_lags)
    n     = len(signal)

    num_acc = np.zeros(n_lags, dtype=complex)
    den_acc = np.zeros(n_lags, dtype=float)
    n_frames = 0

    start = 0
    while start + frame_len <= n:
        seg = signal[start:start + frame_len]
        X   = np.fft.rfft(seg, n=nfft)
        for k in range(min(n_lags, len(X) // 2)):
            if 2 * k < len(X):
                num_acc[k] += X[k] ** 2 * np.conj(X[2 * k])
                den_acc[k] += (np.abs(X[k]) ** 2) * np.abs(X[2 * k]) + 1e-12
        n_frames += 1
        start += hop

    if n_frames == 0:
        return 0.0
    bic = np.abs(num_acc) / (den_acc + 1e-12)
    return float(np.mean(bic))

def compute_bicoherence_all_channels(utt_ids, flac_dir, desc=""):
    """
    Computes mean bicoherence index across all 24 channels for each utterance.
    Returns (N, 24) array.
    """
    results = []
    for uid in tqdm(utt_ids, desc=desc, ncols=80):
        try:
            wav = load_wav(os.path.join(flac_dir, uid + ".flac"))
            row = []
            for ch in range(N_CHANNELS):
                fine, _ = get_fine_structure(wav, ch)
                bic = bicoherence_index(fine)
                row.append(bic)
            results.append(row)
        except Exception as e:
            print(f"  Skip {uid}: {e}")
            results.append([0.0] * N_CHANNELS)
    return np.array(results, dtype=np.float32)

# =============================================================================
# SECTION 8 — Frame Length Variance Analysis
# =============================================================================

def frame_length_variance_analysis(utt_ids, flac_dir,
                                    frame_lengths=FRAME_LENGTHS_SEC,
                                    n_utts=20):
    """
    For each frame length, computes the variance of the diagonal bispectrum
    estimate across utterances at the target channel.
    Returns dict {frame_len_sec: variance_array (N_BISPEC_LAGS,)}.
    """
    results = {}
    sample_ids = utt_ids[:n_utts]

    for fl_sec in frame_lengths:
        fl_samples = int(fl_sec * SR)
        hop        = fl_samples // 2
        nfft       = max(256, 2 * BISPEC_LAGS)

        utterance_means = []
        for uid in tqdm(sample_ids,
                        desc=f"Frame {int(fl_sec*1000)}ms",
                        ncols=80, leave=False):
            try:
                wav  = load_wav(os.path.join(flac_dir, uid + ".flac"))
                fine, _ = get_fine_structure(wav, TARGET_CHANNEL)

                frame_bispec = []
                start = 0
                while start + fl_samples <= len(fine):
                    seg = fine[start:start + fl_samples]
                    X   = np.fft.rfft(seg, n=nfft)
                    B_diag = np.array([
                        abs(X[k]**2 * np.conj(X[2*k]))
                        if 2*k < len(X) else 0.0
                        for k in range(BISPEC_LAGS)
                    ])
                    frame_bispec.append(B_diag)
                    start += hop

                if frame_bispec:
                    utterance_means.append(np.mean(frame_bispec, axis=0))
            except Exception:
                pass

        if utterance_means:
            arr = np.array(utterance_means)
            # Variance across utterances (averaged over lag bins)
            results[fl_sec] = float(np.mean(np.var(arr, axis=0)))
        else:
            results[fl_sec] = 0.0

    return results

# =============================================================================
# SECTION 9 — Figures
# =============================================================================

def fig_bispectrum_2d(B_matrix, title, out_path):
    """2D bispectrum magnitude (log scale)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    B_plot = B_matrix + 1e-10   # avoid log(0)
    im = ax.imshow(
        B_plot,
        origin='lower',
        aspect='auto',
        norm=LogNorm(vmin=B_plot.min(), vmax=B_plot.max()),
        cmap='hot',
        extent=[0, SR/2/1000, 0, SR/2/1000]
    )
    ax.set_xlabel("Frequency $f_1$ (kHz)")
    ax.set_ylabel("Frequency $f_2$ (kHz)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Bispectrum magnitude (log scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")


def fig_bicoherence_boxplot(bon_bic, spo_bic, entries,
                             spo_ids, out_path):
    """
    Per-channel bicoherence boxplot, bonafide vs spoof.
    Spoof coloured by synthesis system group.
    """
    freqs = CHANNEL_CENTRES_HZ

    # Map spoof utterances to group
    spo_groups = []
    for uid in spo_ids:
        sys_id = entries[uid][1]
        group  = 'neural_waveform'   # default
        for grp, sids in SYSTEM_GROUPS.items():
            if sys_id in sids:
                group = grp
                break
        spo_groups.append(group)

    fig, ax = plt.subplots(figsize=(13, 4.5))

    x = np.arange(N_CHANNELS)
    w = 0.35

    # Bonafide: mean ± std
    bon_mean = bon_bic.mean(axis=0)
    bon_std  = bon_bic.std(axis=0)
    ax.plot(x, bon_mean, 'o-', color=GROUP_COLORS['bonafide'],
            lw=1.8, ms=4, label='Bonafide', zorder=3)
    ax.fill_between(x, bon_mean - bon_std, bon_mean + bon_std,
                    color=GROUP_COLORS['bonafide'], alpha=0.2)

    # Spoof by group: mean ± std
    for grp, color in GROUP_COLORS.items():
        if grp == 'bonafide':
            continue
        mask = np.array([g == grp for g in spo_groups])
        if not mask.any():
            continue
        grp_bic  = spo_bic[mask]
        grp_mean = grp_bic.mean(axis=0)
        grp_std  = grp_bic.std(axis=0)
        label_map = {
            'statistical_parametric': 'Spoof: Statistical param. (A07–A10)',
            'waveform_concat'        : 'Spoof: Waveform concat. (A11–A13)',
            'neural_waveform'        : 'Spoof: Neural waveform (A14–A19)',
        }
        ax.plot(x, grp_mean, 's--', color=color, lw=1.5, ms=4,
                label=label_map.get(grp, grp), zorder=2)
        ax.fill_between(x, grp_mean - grp_std, grp_mean + grp_std,
                        color=color, alpha=0.10)

    freq_labels = [f"{int(round(f/10)*10)}" for f in freqs]
    ax.set_xticks(x)
    ax.set_xticklabels(freq_labels, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel("Channel Centre Frequency (Hz)")
    ax.set_ylabel("Mean Bicoherence Index")
    ax.set_title("Per-Channel Bicoherence Index: Bonafide vs Spoof by Synthesis Type")
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")


def fig_frame_length_variance(variance_results, out_path):
    """
    Estimation variance vs frame length bar chart.
    Marks 200 ms design choice.
    """
    fl_ms  = [int(fl * 1000) for fl in sorted(variance_results.keys())]
    variances = [variance_results[fl/1000]
                 for fl in fl_ms]

    colors = ['#B71C1C' if fl == 200 else '#90CAF9' for fl in fl_ms]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(fl_ms)), variances,
                  color=colors, edgecolor='white', linewidth=0.5)

    for bar, v in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(range(len(fl_ms)))
    ax.set_xticklabels([f"{fl} ms" for fl in fl_ms], fontsize=9)
    ax.set_ylabel("Estimation Variance\n(mean across lag bins)")
    ax.set_title("Bispectrum Diagonal Estimation Variance\nvs Frame Length\n"
                 "(Red = design choice: 200 ms)")
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.basename(out_path)}")

# =============================================================================
# SECTION 10 — Main
# =============================================================================

def main():
    print("=" * 65)
    print("02_exp2_bispectrum_validation_v01.py")
    print("Experiment 2 — Bispectrum Estimation and Bicoherence Analysis")
    print("=" * 65)

    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # ------------------------------------------------------------------
    # Step 1: Prepare audio (unzip from Drive if needed)
    # ------------------------------------------------------------------
    print("\nStep 1: Preparing audio ...")
    flac_dir = prepare_audio()

    # ------------------------------------------------------------------
    # Step 2: Parse protocol and select utterances
    # ------------------------------------------------------------------
    print("\nStep 2: Parsing protocol and selecting utterances ...")
    entries = parse_eval_protocol(DRIVE_PROTOCOL)
    bon_ids, spo_ids = select_utterances(
        entries, N_SAMPLES_PER_CLASS, flac_dir)

    # System IDs for spoof utterances
    spo_system_ids = [entries[uid][1] for uid in spo_ids]
    print(f"Systems represented: {sorted(set(spo_system_ids))}")

    # ------------------------------------------------------------------
    # Step 3: Compute 2D bispectrum (indirect method, target channel)
    # ------------------------------------------------------------------
    print(f"\nStep 3: Computing 2D bispectrum at channel {TARGET_CHANNEL} "
          f"({CHANNEL_CENTRES_HZ[TARGET_CHANNEL]:.0f} Hz)...")
    print("  Bonafide ...")
    B_bon = compute_class_mean_bispectrum(
        bon_ids, flac_dir, TARGET_CHANNEL,
        desc="Bispectrum [bonafide]")

    print("  Spoof ...")
    B_spo = compute_class_mean_bispectrum(
        spo_ids, flac_dir, TARGET_CHANNEL,
        desc="Bispectrum [spoof]")

    # ------------------------------------------------------------------
    # Step 4: Compute bicoherence index — all 24 channels
    # ------------------------------------------------------------------
    print("\nStep 4: Computing bicoherence index across all channels ...")
    bon_bic = compute_bicoherence_all_channels(
        bon_ids, flac_dir, desc="Bicoherence [bonafide]")
    spo_bic = compute_bicoherence_all_channels(
        spo_ids, flac_dir, desc="Bicoherence [spoof]")

    print(f"\nBicoherence summary (mean across utterances):")
    print(f"  {'Ch':>3}  {'Freq(Hz)':>8}  "
          f"{'Bon':>8}  {'Spo':>8}  {'Ratio(bon/spo)':>14}")
    print("  " + "-" * 50)
    for ch in range(N_CHANNELS):
        bon_m = bon_bic[:, ch].mean()
        spo_m = spo_bic[:, ch].mean()
        ratio = bon_m / max(spo_m, 1e-12)
        print(f"  {ch:>3}  {CHANNEL_CENTRES_HZ[ch]:>8.0f}  "
              f"{bon_m:>8.4f}  {spo_m:>8.4f}  {ratio:>14.3f}")

    # ------------------------------------------------------------------
    # Step 5: Frame length variance analysis
    # ------------------------------------------------------------------
    print("\nStep 5: Frame length variance analysis ...")
    variance_results = frame_length_variance_analysis(
        bon_ids + spo_ids, flac_dir,
        frame_lengths=FRAME_LENGTHS_SEC, n_utts=20)

    print(f"\nVariance vs frame length:")
    for fl_sec, var in sorted(variance_results.items()):
        marker = " ← design choice" if int(fl_sec * 1000) == 200 else ""
        print(f"  {int(fl_sec*1000):>4} ms : {var:.5f}{marker}")

    # ------------------------------------------------------------------
    # Step 6: Save results JSON
    # ------------------------------------------------------------------
    results = {
        'experiment'             : 'Exp2_BispectrumValidation',
        'n_bonafide'             : len(bon_ids),
        'n_spoof'                : len(spo_ids),
        'target_channel'         : TARGET_CHANNEL,
        'target_freq_hz'         : float(CHANNEL_CENTRES_HZ[TARGET_CHANNEL]),
        'channel_centres_hz'     : [float(f) for f in CHANNEL_CENTRES_HZ],
        'bicoherence_bonafide'   : {
            'mean_per_channel' : bon_bic.mean(axis=0).tolist(),
            'std_per_channel'  : bon_bic.std(axis=0).tolist(),
        },
        'bicoherence_spoof'      : {
            'mean_per_channel' : spo_bic.mean(axis=0).tolist(),
            'std_per_channel'  : spo_bic.std(axis=0).tolist(),
        },
        'bicoherence_ratio_bon_over_spo': [
            float(bon_bic[:, ch].mean() /
                  max(spo_bic[:, ch].mean(), 1e-12))
            for ch in range(N_CHANNELS)
        ],
        'frame_length_variance'  : {
            str(int(k*1000)) + 'ms': round(v, 6)
            for k, v in variance_results.items()
        },
    }

    json_path = os.path.join(LOCAL_OUT_DIR, "exp2_bispectrum.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: exp2_bispectrum.json")

    # ------------------------------------------------------------------
    # Step 7: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures ...")

    freq_label = f"{CHANNEL_CENTRES_HZ[TARGET_CHANNEL]:.0f}"

    fig_bispectrum_2d(
        B_bon,
        f"2D Bispectrum Magnitude — Bonafide Speech\n"
        f"(Indirect method, channel {TARGET_CHANNEL}, {freq_label} Hz, "
        f"averaged over {len(bon_ids)} utterances)",
        os.path.join(LOCAL_OUT_DIR, "fig_02_01_bispectrum_bonafide.png"))

    fig_bispectrum_2d(
        B_spo,
        f"2D Bispectrum Magnitude — Synthetic Speech\n"
        f"(Indirect method, channel {TARGET_CHANNEL}, {freq_label} Hz, "
        f"averaged over {len(spo_ids)} utterances)",
        os.path.join(LOCAL_OUT_DIR, "fig_02_02_bispectrum_spoof.png"))

    fig_bicoherence_boxplot(
        bon_bic, spo_bic, entries, spo_ids,
        os.path.join(LOCAL_OUT_DIR, "fig_02_03_bicoherence_boxplot.png"))

    fig_frame_length_variance(
        variance_results,
        os.path.join(LOCAL_OUT_DIR, "fig_02_04_frame_length_variance.png"))

    # ------------------------------------------------------------------
    # Step 8: Upload all to FTP
    # ------------------------------------------------------------------
    print("\nUploading to FTP ...")
    for fname in [
        "exp2_bispectrum.json",
        "fig_02_01_bispectrum_bonafide.png",
        "fig_02_02_bispectrum_spoof.png",
        "fig_02_03_bicoherence_boxplot.png",
        "fig_02_04_frame_length_variance.png",
    ]:
        upload(os.path.join(LOCAL_OUT_DIR, fname), fname)

    print("\n" + "=" * 65)
    print("Experiment 2 complete.")
    best_ch = int(np.argmax([
        bon_bic[:, ch].mean() / max(spo_bic[:, ch].mean(), 1e-12)
        for ch in range(N_CHANNELS)
    ]))
    print(f"  Best bicoherence ratio channel: {best_ch} "
          f"({CHANNEL_CENTRES_HZ[best_ch]:.0f} Hz)  "
          f"ratio={bon_bic[:, best_ch].mean() / max(spo_bic[:, best_ch].mean(), 1e-12):.3f}")
    print(f"  Lowest variance frame length: "
          f"{min(variance_results, key=variance_results.get)*1000:.0f} ms  "
          f"var={min(variance_results.values()):.5f}")
    print("=" * 65)


# =============================================================================
if __name__ == "__main__":
    main()