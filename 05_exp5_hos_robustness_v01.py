# =============================================================================
# Program      : 05_exp5_hos_robustness.py
# Version      : 1.17
# Description  : Experiment 5 — Robustness of Subband HOS Features Under
#                Codec and Channel Distortion for Synthetic Speech Detection.
#
# INPUT:
#                  - /content/drive/MyDrive/datasets/ASVspoof-2019-LA.zip
#                  - /content/drive/MyDrive/datasets/ASVspoof2019_LA_cm_protocols/
#                      ASVspoof2019.LA.cm.train.trn.txt
#                      ASVspoof2019.LA.cm.eval.trl.txt
#
#                PIPELINE:
#                  Step 1  Mount Google Drive, copy dataset to local disk, unzip
#                  Step 2  Parse protocol files (train / eval labels)
#                  Step 3  Extract HOS features (24-ch gammatone filterbank,
#                          per-channel kurtosis of fine-structure & envelope,
#                          diagonal bispectrum slice) for train partition
#                  Step 4  Train XGBoost classifier; save model checkpoint
#                  Step 5  Extract HOS features for clean LA-eval; compute
#                          baseline EER and min-tDCF
#                  Step 6  Generate 24 distorted variants of LA-eval via FFmpeg
#                          (MP3: 32/64/128 kbps | Opus: 6/12/24 kbps |
#                           AWGN: 0/5/10/20 dB SNR | Telephone BPF |
#                           Combo: MP3-32+Tel, Opus-6+AWGN-10, MP3-64+AWGN-5)
#                  Step 7  Evaluate HOS-XGBoost on each distortion condition
#                  Step 8  Evaluate LFCC-GMM baseline (SpeechBrain pretrained)
#                          on same conditions
#                  Step 9  Compute relative degradation ratio per system/condition
#                  Step 10 Generate output figures and JSON results table
#
# OUTPUT FILES (uploaded to FTP PROJECT_DIR):
#                  Checkpoints:
#                    hos_features_train.h5       — HOS features, train partition
#                    hos_features_eval_clean.h5  — HOS features, clean eval
#                    hos_features_eval_<cond>.h5 — HOS features per distortion
#                    xgb_model.json              — trained XGBoost model
#                    lfcc_scores_<cond>.json     — LFCC-GMM scores per condition
#                  Results:
#                    results_exp5.json           — full EER/tDCF table (all conditions)
#                  Figures:
#                    fig_05_01_eer_vs_mp3.png    — EER vs MP3 bitrate curve
#                    fig_05_02_eer_vs_opus.png   — EER vs Opus bitrate curve
#                    fig_05_03_eer_vs_awgn.png   — EER vs SNR curve
#                    fig_05_04_degradation_radar.png — radar chart of robustness
#                    fig_05_05_eer_heatmap.png   — full condition heatmap
#
# GPU Required : NO  (XGBoost CPU mode is sufficient; no deep learning)
# Dependencies : numpy, scipy, xgboost, h5py, matplotlib, tqdm, speechbrain,
#                librosa, soundfile, ftplib (stdlib)
#
# Change Log   :
#   v1.0  2026-06-01  Initial version
#   v1.1  2026-06-01  Fixed audio path resolution: replaced get_audio_path()
#                     with prefix-based router (LA_T_→train, LA_D_→dev,
#                     LA_E_→eval) + auto-discovery of actual unzipped root.
#                     Added discover_audio_root() to handle any zip layout.
#   v1.2  2026-06-01  Added background FTP heartbeat thread (start_ftp_heartbeat).
#                     Uploads all checkpoint files to FTP every
#                     FTP_HEARTBEAT_INTERVAL seconds throughout the run.
#                     Eliminates data loss on full Colab session wipe.
#   v1.3  2026-06-01  Added FTP-to-local recovery at the start of
#                     extract_features_batch() and train_xgboost():
#                     if local file is missing, download from FTP before
#                     starting work. Ensures partial HDF5 and model
#                     checkpoints survive a full session wipe.
#   v1.4  2026-06-01  Fixed HDF5 corruption: heartbeat thread no longer
#                     uploads .h5 files while h5py has them open for writing
#                     (caused "bad object header version number" OSError).
#                     HDF5 uploads now happen inside extract_features_batch()
#                     via close→upload→reopen every H5_UPLOAD_EVERY utterances.
#                     Heartbeat restricted to safe files only (json/npz/png).
#   v1.5  2026-06-01  Removed HDF5 FTP recovery from extract_features_batch().
#                     Downloading partial HDF5 from FTP is unsafe regardless
#                     of upload method — file may be corrupt on server.
#                     Resume is now local-only; corrupt local files are auto-
#                     detected and deleted so extraction restarts cleanly.
#   v1.6  2026-06-01  Restored safe HDF5 recovery with validation:
#                     recover_h5() checks Drive first, then FTP, validates
#                     every downloaded file with h5py before accepting it.
#                     Corrupt or empty downloads are deleted immediately.
#                     close→upload→reopen cycle now also copies to Drive.
#   v1.7  2026-06-02  Reduced H5_UPLOAD_EVERY from 2000 → 500 utterances.
#                     At ~1.6 it/s the first safe checkpoint now fires in
#                     ~5 min instead of ~21 min, surviving short disconnects.
#   v1.8  2026-06-02  Fixed log messages being overwritten by tqdm progress
#                     bar. Replaced basicConfig StreamHandler with a custom
#                     TqdmLoggingHandler that routes all log output through
#                     tqdm.write(), making checkpoint log lines visible.
#   v1.9  2026-06-02  Added corrupt npz auto-recovery in _fit_lfcc_gmm_if_needed:
#                     catches EOFError/corrupt load, deletes bad local file,
#                     purges from FTP, then refits cleanly. No more crash on
#                     corrupt lfcc_gmm_models.npz.
#  v1.10  2026-06-02  Fixed OOM crash during GMM fitting. Reduced components
#                     512→128 with diagonal covariance (matches ASVspoof
#                     baseline), capped training utterances 3000→1000 and
#                     frames per utterance to 200. ~10x RAM reduction.
#  v1.11  2026-06-02  Fixed ValueError: empty spo_feats in GMM fitting.
#                     Replaced first-N sampling with explicit per-class
#                     sampling (500 bonafide + 500 spoof). Adds RuntimeError
#                     guard if either class still empty after sampling.
#  v1.12  2026-06-02  Major distortion loop redesign for Colab resilience:
#                     (a) Reduced distortion eval set to 5000 stratified
#                     utterances (from 71237) — each condition now takes
#                     ~15 min generation + ~45 min extraction instead of 7h.
#                     (b) generate_and_extract_condition() replaces separate
#                     generate_distorted_set() + extract_features_batch():
#                     processes utterances in BATCH_SIZE=500 batches,
#                     generates wav → extracts features → appends to HDF5
#                     → deletes wav, all within each batch. HDF5 checkpoint
#                     uploaded to FTP+Drive after every batch. Resumable at
#                     any batch boundary on session wipe.
#  v1.13  2026-06-03  Fixed JSONDecodeError on results_exp5.json resume:
#                     _safe_load_results() checks for zero-byte or corrupt
#                     JSON before parsing, discards bad file, falls back
#                     gracefully to starting fresh.
#  v1.14  2026-06-03  Removed misleading SpeechBrain try/except block from
#                     run_lfcc_gmm_baseline(). SpeechBrain was never actually
#                     used — local LFCC-GMM always ran regardless. Removed
#                     the dead code path and the spurious warning message.
#                     Renamed section header to reflect local implementation.
#  v1.15  2026-06-03  Fixed critical bug: LFCC distortion scoring was reading
#                     clean audio instead of distorted audio for all conditions
#                     (get_audio_path returned original flac, not distorted wav).
#                     LFCC now re-generates distortion on 500-utterance subsample
#                     and scores distorted audio correctly. Scores checkpointed
#                     to lfcc_scores_<cond>.json with resume support.
#  v1.16  2026-06-03  Fixed two bugs causing FileNotFoundError on opus_006:
#                     (a) prepare_dataset() now validates .flac files exist
#                     before trusting the unzip marker — re-unzips if audio
#                     was wiped from local disk (new Colab VM).
#                     (b) generate_and_extract_condition() now guards the
#                     final HDF5 load with os.path.exists() check, returns
#                     empty arrays with warning instead of crashing.
#  v1.17  2026-06-03  Fixed Opus codec failure: libopus cannot be muxed into
#                     WAV container ("Codec opus not supported in WAVE format").
#                     apply_ffmpeg_distortion() now detects libopus and uses
#                     encode→OGG then decode→WAV two-step pipeline. Same fix
#                     applied to opus_awgn10 combo in both HOS and LFCC paths.
# =============================================================================

# --- Install dependencies ---
# Run this cell first in Colab
# !pip install xgboost h5py speechbrain librosa soundfile tqdm matplotlib numpy scipy

# =============================================================================
# SECTION 0 — Imports and Configuration
# =============================================================================

import os
import sys
import json
import shutil
import zipfile
import subprocess
import tempfile
import time
import threading
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.signal as ss
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
from tqdm import tqdm
import xgboost as xgb
import soundfile as sf
import librosa

warnings.filterwarnings('ignore')

class _TqdmLoggingHandler(logging.Handler):
    """Routes log records through tqdm.write() so they are not overwritten
    by tqdm progress bars."""
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

_handler = _TqdmLoggingHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s  %(levelname)s  %(message)s',
                                         datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger(__name__)

# --- FTP Configuration ---
FTP_HOST             = "173.225.103.246"
FTP_PORT             = 2121
FTP_USER             = "guest"
FTP_PASS             = "guest"
FTP_PROJECT_DIR      = "."
FTP_HEARTBEAT_INTERVAL = 180   # seconds between periodic checkpoint uploads (3 min)
H5_UPLOAD_EVERY        = 500   # upload HDF5 to FTP after every N utterances (close→upload→reopen)
DISTORTION_EVAL_N      = 5000  # stratified subset size for distortion conditions
DISTORTION_BATCH_SIZE  = 500   # utterances per generate→extract→checkpoint batch

# --- Paths ---
DRIVE_DIR        = "/content/drive/MyDrive/datasets"
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/paper/Subband_Kurtosis"  # HDF5 backup
LOCAL_WORK_DIR   = "/content/exp5_work"
AUDIO_ROOT       = None   # v1.1: replaced by _PARTITION_DIRS auto-discovery
PROTOCOL_SRC     = os.path.join(DRIVE_DIR, "ASVspoof2019_LA_cm_protocols")
PROTOCOL_LOCAL   = os.path.join(LOCAL_WORK_DIR, "protocols")
DISTORTED_ROOT   = os.path.join(LOCAL_WORK_DIR, "distorted")
CHECKPOINT_DIR   = os.path.join(LOCAL_WORK_DIR, "checkpoints")

for d in [LOCAL_WORK_DIR, PROTOCOL_LOCAL, DISTORTED_ROOT, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Feature extraction config ---
SR              = 16000          # target sample rate
N_CHANNELS      = 24             # gammatone filterbank channels
FMIN            = 80.0           # Hz
FMAX            = 8000.0         # Hz
FRAME_LEN_SEC   = 0.200          # 200 ms
HOP_LEN_SEC     = 0.100          # 50% overlap
BISPEC_LAGS     = 32             # number of lag points for bispectrum diagonal
POOL_PERCENTILE = [10, 90]       # percentile pooling in addition to mean/var

# --- Distortion conditions ---
# Each entry: (condition_tag, ffmpeg_args_template, human_label)
DISTORTIONS = [
    # MP3
    ("mp3_032", ["-codec:a", "libmp3lame", "-b:a", "32k"],   "MP3 32 kbps"),
    ("mp3_064", ["-codec:a", "libmp3lame", "-b:a", "64k"],   "MP3 64 kbps"),
    ("mp3_128", ["-codec:a", "libmp3lame", "-b:a", "128k"],  "MP3 128 kbps"),
    # Opus (via ogg container)
    ("opus_006", ["-codec:a", "libopus", "-b:a", "6k"],      "Opus 6 kbps"),
    ("opus_012", ["-codec:a", "libopus", "-b:a", "12k"],     "Opus 12 kbps"),
    ("opus_024", ["-codec:a", "libopus", "-b:a", "24k"],     "Opus 24 kbps"),
    # AWGN — handled in Python (not FFmpeg), tag used for routing
    ("awgn_00db", None, "AWGN 0 dB SNR"),
    ("awgn_05db", None, "AWGN 5 dB SNR"),
    ("awgn_10db", None, "AWGN 10 dB SNR"),
    ("awgn_20db", None, "AWGN 20 dB SNR"),
    # Telephone bandlimiting (300–3400 Hz)
    ("telephone", ["-af", "highpass=f=300,lowpass=f=3400"], "Telephone BPF"),
    # Combos
    ("combo_mp3tel",  ["-codec:a", "libmp3lame", "-b:a", "32k",
                        "-af", "highpass=f=300,lowpass=f=3400"],
                       "MP3-32 + Tel"),
    ("combo_opus_awgn10", None,  "Opus-6 + AWGN 10 dB"),   # Python-side combo
    ("combo_mp3_awgn5",   None,  "MP3-64 + AWGN 5 dB"),    # Python-side combo
]

AWGN_SNRS = {
    "awgn_00db": 0, "awgn_05db": 5,
    "awgn_10db": 10, "awgn_20db": 20,
}

# =============================================================================
# SECTION 1 — FTP Helper Functions
# =============================================================================

import ftplib

def get_ftp_connection():
    ftp = ftplib.FTP()
    ftp.connect(FTP_HOST, FTP_PORT, timeout=30)
    ftp.login(FTP_USER, FTP_PASS)
    if FTP_PROJECT_DIR != ".":
        ftp.cwd(FTP_PROJECT_DIR)
    return ftp

def upload_to_ftp(local_filepath: str, remote_filename: str, retries: int = 3):
    for attempt in range(retries):
        try:
            ftp = get_ftp_connection()
            with open(local_filepath, "rb") as f:
                ftp.storbinary(f"STOR {remote_filename}", f)
            ftp.quit()
            log.info(f"FTP upload OK: {remote_filename}")
            return
        except Exception as e:
            log.warning(f"FTP upload attempt {attempt+1} failed: {e}")
            time.sleep(2)
    log.error(f"FTP upload FAILED after {retries} attempts: {remote_filename}")

def download_from_ftp(remote_filename: str, local_filepath: str) -> bool:
    try:
        ftp = get_ftp_connection()
        with open(local_filepath, "wb") as f:
            ftp.retrbinary(f"RETR {remote_filename}", f.write)
        ftp.quit()
        log.info(f"FTP download OK: {remote_filename}")
        return True
    except Exception:
        return False

def ftp_file_exists(remote_filename: str) -> bool:
    try:
        ftp = get_ftp_connection()
        names = ftp.nlst()
        ftp.quit()
        return remote_filename in names
    except Exception:
        return False

# =============================================================================
# SECTION 1b — Background FTP Heartbeat Thread
# =============================================================================

def _heartbeat_worker(ckpt_dir: str, interval: int, stop_event: threading.Event):
    """
    Daemon thread: every `interval` seconds, uploads SAFE checkpoint files
    to FTP. HDF5 files (.h5) are intentionally EXCLUDED — they are uploaded
    inside extract_features_batch() via a close→upload→reopen cycle to avoid
    the "bad object header" corruption caused by reading an open HDF5 file.

    Safe files uploaded here:
      xgb_model.json, lfcc_gmm_models.npz,
      results_exp5.json, lfcc_scores_*.json, fig_05_*.png
    """
    log.info(f"[Heartbeat] FTP backup thread started  (interval={interval}s)")
    log.info(f"[Heartbeat] NOTE: .h5 files excluded — uploaded inline during extraction")

    while not stop_event.is_set():
        stop_event.wait(timeout=interval)
        if stop_event.is_set():
            break

        uploaded, skipped, failed = 0, 0, 0
        candidates = []

        try:
            for fname in os.listdir(ckpt_dir):
                # NEVER upload .h5 files from this thread
                if fname.endswith(".h5"):
                    continue
                if fname in ("xgb_model.json", "lfcc_gmm_models.npz",
                             "results_exp5.json", "lfcc_scores_clean.json"):
                    candidates.append(fname)
                elif (fname.startswith("lfcc_scores_") and fname.endswith(".json")) or \
                     (fname.startswith("fig_05_")      and fname.endswith(".png")):
                    candidates.append(fname)
        except Exception:
            pass

        for fname in candidates:
            local_path = os.path.join(ckpt_dir, fname)
            if not os.path.exists(local_path):
                skipped += 1
                continue
            try:
                ftp = get_ftp_connection()
                with open(local_path, "rb") as f:
                    ftp.storbinary(f"STOR {fname}", f)
                ftp.quit()
                uploaded += 1
            except Exception as e:
                failed += 1
                log.warning(f"[Heartbeat] Upload failed for {fname}: {e}")

        log.info(f"[Heartbeat] {time.strftime('%H:%M:%S')}  "
                 f"uploaded={uploaded}  skipped={skipped}  failed={failed}")

    log.info("[Heartbeat] FTP backup thread stopped.")


def start_ftp_heartbeat(ckpt_dir: str,
                        interval: int = FTP_HEARTBEAT_INTERVAL
                        ) -> Tuple[threading.Thread, threading.Event]:
    """
    Starts the background FTP heartbeat thread.
    Returns (thread, stop_event).
    Call stop_event.set() to cleanly shut the thread down at the end of main().

    Usage:
        thread, stop_evt = start_ftp_heartbeat(CHECKPOINT_DIR)
        ... run experiment ...
        stop_evt.set()
        thread.join()
    """
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_heartbeat_worker,
        args=(ckpt_dir, interval, stop_event),
        daemon=True,          # dies automatically if main thread exits
        name="ftp-heartbeat",
    )
    thread.start()
    return thread, stop_event

# =============================================================================
# SECTION 2 — Drive Mount and Dataset Preparation
# =============================================================================

def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')
    log.info("Google Drive mounted.")

def discover_audio_root(search_root: str) -> str:
    """
    Walk search_root to find the directory that contains the three
    ASVspoof2019 LA partition sub-folders (train / dev / eval).
    Returns the path of that parent directory.
    Raises RuntimeError if not found.
    """
    log.info(f"Discovering audio root under {search_root} …")
    for dirpath, dirnames, _ in os.walk(search_root):
        subdirs = set(dirnames)
        # Accept any layout that has at least the train and eval sub-folders
        # (names may vary slightly across zip versions)
        has_train = any('train' in d.lower() and 'LA' in d for d in subdirs)
        has_eval  = any('eval'  in d.lower() and 'LA' in d for d in subdirs)
        if has_train and has_eval:
            log.info(f"  Audio root found: {dirpath}")
            log.info(f"  Sub-folders: {sorted(subdirs)}")
            return dirpath
    raise RuntimeError(
        f"Could not find ASVspoof2019 LA partition folders under {search_root}. "
        f"Check that the zip extracted correctly."
    )


# Module-level cache: populated by prepare_dataset(), used by get_audio_path()
_PARTITION_DIRS: Dict[str, str] = {}   # maps prefix → absolute flac directory


def prepare_dataset():
    """Copy zip from Drive → local, unzip, copy protocols, resolve partition dirs."""
    global _PARTITION_DIRS

    zip_src   = os.path.join(DRIVE_DIR, "ASVspoof-2019-LA.zip")
    zip_local = os.path.join(LOCAL_WORK_DIR, "ASVspoof-2019-LA.zip")

    # Copy zip to local if not already present
    if not os.path.exists(zip_local):
        log.info("Copying dataset zip from Drive to local disk …")
        shutil.copy2(zip_src, zip_local)
        log.info(f"Copied: {zip_local}  ({os.path.getsize(zip_local)/1e9:.2f} GB)")
    else:
        log.info("Dataset zip already on local disk.")

    # Unzip — validate marker AND that audio files actually exist
    marker = os.path.join(LOCAL_WORK_DIR, ".unzip_done")
    audio_present = False
    if os.path.exists(marker):
        # Verify at least one .flac file exists — marker can survive a disk wipe
        result = subprocess.run(
            ["find", LOCAL_WORK_DIR, "-name", "*.flac", "-maxdepth", "6"],
            capture_output=True, text=True, timeout=30)
        audio_present = bool(result.stdout.strip())
        if not audio_present:
            log.warning("Unzip marker found but no .flac files — re-unzipping.")
            os.remove(marker)

    if not os.path.exists(marker):
        log.info("Unzipping dataset …")
        with zipfile.ZipFile(zip_local, 'r') as z:
            z.extractall(LOCAL_WORK_DIR)
        Path(marker).touch()
        log.info(f"Unzipped to {LOCAL_WORK_DIR}")
    else:
        log.info("Dataset already unzipped and audio files verified.")

    # Auto-discover the actual root that contains train/dev/eval sub-folders
    audio_root = discover_audio_root(LOCAL_WORK_DIR)

    # Build partition-dir map: LA_T_ → .../train/flac, etc.
    # Scan sub-folders and match by name pattern
    _PARTITION_DIRS = {}
    for entry in os.scandir(audio_root):
        if not entry.is_dir():
            continue
        name_lower = entry.name.lower()
        flac_dir   = os.path.join(entry.path, "flac")
        if not os.path.isdir(flac_dir):
            # Some versions store files directly without a 'flac' sub-folder
            flac_dir = entry.path
        if 'train' in name_lower:
            _PARTITION_DIRS['LA_T_'] = flac_dir
        elif 'dev' in name_lower:
            _PARTITION_DIRS['LA_D_'] = flac_dir
        elif 'eval' in name_lower:
            _PARTITION_DIRS['LA_E_'] = flac_dir

    if not _PARTITION_DIRS:
        raise RuntimeError("No partition directories mapped. Check unzip output.")

    log.info("Partition directory map:")
    for prefix, d in _PARTITION_DIRS.items():
        n_files = len(os.listdir(d)) if os.path.isdir(d) else 0
        log.info(f"  {prefix} → {d}  ({n_files} files)")

    # Copy protocol files
    for fname in ["ASVspoof2019.LA.cm.train.trn.txt",
                  "ASVspoof2019.LA.cm.eval.trl.txt"]:
        dst = os.path.join(PROTOCOL_LOCAL, fname)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(PROTOCOL_SRC, fname), dst)
    log.info("Protocol files ready.")

# =============================================================================
# SECTION 3 — Protocol Parsing
# =============================================================================

def parse_protocol(txt_path: str) -> Dict[str, int]:
    """
    Returns dict {utt_id: label} where label = 0 (bonafide) or 1 (spoof).
    Handles both train (*.trn.txt) and eval (*.trl.txt) formats.
    Format: SPEAKER_ID  UTT_ID  -  SYSTEM_ID  LABEL
    """
    labels = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id = parts[1]
            label_str = parts[4]
            labels[utt_id] = 0 if label_str == "bonafide" else 1
    log.info(f"Parsed {len(labels)} utterances from {os.path.basename(txt_path)}")
    return labels

def get_audio_path(utt_id: str, partition: str = None) -> str:
    """
    Resolve audio file path using utterance-ID prefix routing.

    Prefix map (populated by prepare_dataset):
      LA_T_  →  .../ASVspoof2019_LA_train/flac/
      LA_D_  →  .../ASVspoof2019_LA_dev/flac/
      LA_E_  →  .../ASVspoof2019_LA_eval/flac/

    Tries .flac first, then .wav.
    Raises FileNotFoundError with a diagnostic message if not found.
    """
    # Determine target directory from utterance ID prefix
    target_dir = None
    for prefix, d in _PARTITION_DIRS.items():
        if utt_id.startswith(prefix):
            target_dir = d
            break

    candidates = []
    if target_dir:
        candidates = [
            os.path.join(target_dir, utt_id + ".flac"),
            os.path.join(target_dir, utt_id + ".wav"),
        ]

    # Fallback: search all known partition dirs
    if not candidates or not any(os.path.exists(c) for c in candidates):
        for d in _PARTITION_DIRS.values():
            candidates += [
                os.path.join(d, utt_id + ".flac"),
                os.path.join(d, utt_id + ".wav"),
            ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Diagnostic: show what is actually in the expected directory
    diag = ""
    if target_dir and os.path.isdir(target_dir):
        sample = os.listdir(target_dir)[:5]
        diag = f" | Expected dir: {target_dir} | Sample files: {sample}"

    raise FileNotFoundError(
        f"Audio not found for '{utt_id}'{diag}. "
        f"Partition map: { {k: v for k, v in _PARTITION_DIRS.items()} }"
    )

# =============================================================================
# SECTION 4 — Gammatone Filterbank
# =============================================================================

def erb_from_hz(f: float) -> float:
    return 24.7 * (4.37 * f / 1000.0 + 1.0)

def hz_from_erb_rate(erb_rate: float) -> float:
    return (10 ** (erb_rate / 21.366) - 1) / 0.004368

def erb_filterbank_centers(n_channels: int, fmin: float, fmax: float) -> np.ndarray:
    erb_min = 21.366 * np.log10(1 + 0.004368 * fmin)
    erb_max = 21.366 * np.log10(1 + 0.004368 * fmax)
    erb_rates = np.linspace(erb_min, erb_max, n_channels)
    return np.array([hz_from_erb_rate(r) for r in erb_rates])

def make_gammatone_filterbank(n_channels: int, fmin: float, fmax: float,
                               sr: int) -> List[np.ndarray]:
    """
    Returns list of FIR gammatone filter impulse responses, one per channel.
    Uses 4th-order gammatone approximation via cascaded IIR (Patterson 1992).
    Each filter returned as (b, a) coefficient tuple for scipy.signal.lfilter.
    """
    centers = erb_filterbank_centers(n_channels, fmin, fmax)
    filters = []
    for fc in centers:
        bw = 1.019 * erb_from_hz(fc)
        # Approximate with a narrow bandpass Butterworth (simpler, stable)
        # Width = ±0.5*ERB around centre frequency
        low  = max(50.0, fc - 0.5 * bw)
        high = min(sr / 2.0 - 100.0, fc + 0.5 * bw)
        low_n  = low  / (sr / 2.0)
        high_n = high / (sr / 2.0)
        if low_n <= 0 or high_n >= 1 or low_n >= high_n:
            high_n = min(0.999, high_n)
            low_n  = max(0.001, low_n)
        b, a = ss.butter(4, [low_n, high_n], btype='band')
        filters.append((b, a))
    return filters, centers

# Pre-build filterbank (module-level, built once)
_FILTERBANK = None
_CENTERS    = None

def get_filterbank():
    global _FILTERBANK, _CENTERS
    if _FILTERBANK is None:
        _FILTERBANK, _CENTERS = make_gammatone_filterbank(
            N_CHANNELS, FMIN, FMAX, SR)
    return _FILTERBANK, _CENTERS

# =============================================================================
# SECTION 5 — HOS Feature Extraction
# =============================================================================

def hilbert_envelope_finestruct(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (envelope, fine_structure) via Hilbert transform."""
    analytic  = ss.hilbert(signal)
    envelope  = np.abs(analytic)
    fine      = np.real(analytic / (envelope + 1e-12))
    return envelope, fine

def excess_kurtosis(x: np.ndarray) -> float:
    """Fisher definition (normal = 0); robust to short frames."""
    n = len(x)
    if n < 4:
        return 0.0
    m  = np.mean(x)
    s  = np.std(x)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)

def bispectrum_diagonal_slice(x: np.ndarray, n_lags: int) -> np.ndarray:
    """
    Estimates the diagonal bispectrum slice B(f, f) via the direct method:
      B(f,f) ≈ E[ X(f)^2 · X*(2f) ]  (averaged over frames).
    Returns magnitude of length n_lags.
    """
    n = len(x)
    if n < 2 * n_lags:
        return np.zeros(n_lags)
    nfft   = max(256, 2 * n_lags)
    X      = np.fft.rfft(x, n=nfft)
    # Only use first n_lags frequency bins
    n_use  = min(n_lags, len(X) // 2)
    B_diag = np.zeros(n_use, dtype=complex)
    for k in range(n_use):
        if 2 * k < len(X):
            B_diag[k] = X[k] ** 2 * np.conj(X[2 * k])
    return np.abs(B_diag[:n_lags]) if len(B_diag) >= n_lags \
           else np.pad(np.abs(B_diag), (0, n_lags - len(B_diag)))

def extract_hos_features(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Full HOS feature vector for one utterance.

    Per channel (24 channels):
      - excess kurtosis of fine-structure  (1 value, frame-pooled)
      - excess kurtosis of envelope        (1 value, frame-pooled)
      - bispectrum diagonal slice of fine-structure (BISPEC_LAGS values,
        pooled: mean + 10th pctile + 90th pctile)

    Pooling: mean, variance, p10, p90 over frames for kurtosis;
             mean + p10 + p90 over frames for bispectrum slice.

    Total dimension:
      kurtosis    : 24 × 2 × 4  = 192
      bispectrum  : 24 × BISPEC_LAGS × 3 = 24 × 32 × 3 = 2304
      Total       : 2496

    Returns 1D float32 array of length 2496.
    """
    # Resample if needed
    if sr != SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SR)
        sr = SR

    # Pre-emphasis
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    frame_len = int(FRAME_LEN_SEC * SR)
    hop_len   = int(HOP_LEN_SEC   * SR)
    n_frames  = max(1, (len(waveform) - frame_len) // hop_len + 1)

    filterbank, _ = get_filterbank()

    # Accumulators: (n_frames,) arrays per stat per channel
    kurt_fine_frames = np.zeros((N_CHANNELS, n_frames))
    kurt_env_frames  = np.zeros((N_CHANNELS, n_frames))
    bispec_frames    = np.zeros((N_CHANNELS, n_frames, BISPEC_LAGS))

    for ch_idx, (b, a) in enumerate(filterbank):
        filtered = ss.lfilter(b, a, waveform)
        envelope, fine = hilbert_envelope_finestruct(filtered)
        for fr in range(n_frames):
            start = fr * hop_len
            end   = start + frame_len
            fine_fr = fine[start:end]
            env_fr  = envelope[start:end]
            kurt_fine_frames[ch_idx, fr] = excess_kurtosis(fine_fr)
            kurt_env_frames[ch_idx, fr]  = excess_kurtosis(env_fr)
            bispec_frames[ch_idx, fr]    = bispectrum_diagonal_slice(
                                               fine_fr, BISPEC_LAGS)

    # Pool over frames
    def pool_1d(arr):  # arr shape (n_frames,)
        return np.array([
            np.mean(arr), np.var(arr),
            np.percentile(arr, 10), np.percentile(arr, 90)
        ], dtype=np.float32)

    def pool_bispec(arr):  # arr shape (n_frames, BISPEC_LAGS)
        return np.concatenate([
            np.mean(arr, axis=0),
            np.percentile(arr, 10, axis=0),
            np.percentile(arr, 90, axis=0)
        ]).astype(np.float32)

    feats = []
    for ch_idx in range(N_CHANNELS):
        feats.append(pool_1d(kurt_fine_frames[ch_idx]))
        feats.append(pool_1d(kurt_env_frames[ch_idx]))
        feats.append(pool_bispec(bispec_frames[ch_idx]))

    return np.concatenate(feats).astype(np.float32)

# =============================================================================
# SECTION 5b — Safe HDF5 Recovery Helper
# =============================================================================

def recover_h5(h5_path: str) -> int:
    """
    Attempts to recover a missing local HDF5 from Drive backup first,
    then FTP. Validates every candidate before accepting it — corrupt
    or empty files are deleted immediately and the next source is tried.

    Returns number of utterances recovered (0 = starting fresh).

    Recovery priority:
      1. Drive backup  (DRIVE_BACKUP_DIR/<filename>)
      2. FTP           (only files uploaded via close→upload→reopen are safe)
    """
    fname = os.path.basename(h5_path)

    def validate_and_install(src_path: str, label: str) -> int:
        """Copy src→h5_path, validate, return utterance count or 0 on failure."""
        try:
            shutil.copy2(src_path, h5_path)
            with h5py.File(h5_path, 'r') as hf:
                n = len(hf.keys())
            if n == 0:
                raise ValueError("Empty HDF5 — no utterances")
            log.info(f"  Recovered {fname} from {label}: {n} utterances  "
                     f"({os.path.getsize(h5_path)/1e6:.1f} MB)")
            return n
        except Exception as e:
            log.warning(f"  {label} copy invalid ({e}) — discarding.")
            if os.path.exists(h5_path):
                os.remove(h5_path)
            return 0

    # --- Source 1: Drive backup ---
    drive_src = os.path.join(DRIVE_BACKUP_DIR, fname)
    if os.path.exists(drive_src) and os.path.getsize(drive_src) > 0:
        log.info(f"[recover_h5] Trying Drive backup for {fname} …")
        n = validate_and_install(drive_src, "Drive")
        if n > 0:
            return n
    else:
        log.info(f"[recover_h5] No Drive backup for {fname}.")

    # --- Source 2: FTP ---
    log.info(f"[recover_h5] Trying FTP for {fname} …")
    tmp_path = h5_path + ".tmp_download"
    if download_from_ftp(fname, tmp_path):
        n = validate_and_install(tmp_path, "FTP")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if n > 0:
            return n
    else:
        log.info(f"[recover_h5] No FTP copy for {fname}.")

    log.info(f"[recover_h5] No valid backup found for {fname}. Starting fresh.")
    return 0

def extract_features_batch(utt_ids: List[str], labels_dict: Dict[str, int],
                            h5_path: str, partition: str,
                            audio_root_override: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts HOS features for a list of utterance IDs.
    Saves to h5_path with resume support (skips already-done utterances).

    v1.6: On session wipe, calls recover_h5() which validates before
    accepting any backup (Drive first, FTP second). Corrupt files are
    rejected at source. After each safe close→upload→reopen cycle the
    file is also copied to Drive as a second backup.
    """
    # --- Recovery: Drive → FTP → fresh start (all validated) ---
    if not os.path.exists(h5_path):
        recover_h5(h5_path)

    # Local resume — corrupt local file auto-deleted
    done_ids = set()
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as hf:
                done_ids = set(hf.keys())
            log.info(f"Resuming: {len(done_ids)} utterances in "
                     f"{os.path.basename(h5_path)}")
        except Exception as e:
            log.warning(f"Local HDF5 corrupt ({e}) — deleting and starting fresh.")
            os.remove(h5_path)
            done_ids = set()

    todo = [u for u in utt_ids if u not in done_ids]
    log.info(f"To extract: {len(todo)} utterances for partition '{partition}'")

    errors        = 0
    since_upload  = 0   # utterances written since last FTP upload of this HDF5

    # We open/close h5 manually (not via 'with') so we can close it safely
    # before each FTP upload, then reopen it.
    hf = h5py.File(h5_path, 'a')
    try:
        for i, utt_id in enumerate(tqdm(todo, desc=f"Features [{partition}]",
                                        ncols=90)):
            try:
                if audio_root_override:
                    wav_path = os.path.join(audio_root_override, utt_id + ".wav")
                    if not os.path.exists(wav_path):
                        wav_path = os.path.join(audio_root_override, utt_id + ".flac")
                else:
                    wav_path = get_audio_path(utt_id, partition)
                wav, sr = sf.read(wav_path, dtype='float32')
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                feat = extract_hos_features(wav, sr)
                grp = hf.create_group(utt_id)
                grp.create_dataset('feat',  data=feat)
                grp.create_dataset('label', data=labels_dict.get(utt_id, -1))
                since_upload += 1
            except Exception as e:
                errors += 1
                log.warning(f"  Skipped {utt_id}: {e}")

            # Periodic safe HDF5 upload: close → upload → reopen
            if since_upload >= H5_UPLOAD_EVERY:
                hf.flush()
                hf.close()
                remote_name = os.path.basename(h5_path)
                n_so_far = len(done_ids) + i + 1
                # FTP upload
                try:
                    ftp = get_ftp_connection()
                    with open(h5_path, "rb") as f:
                        ftp.storbinary(f"STOR {remote_name}", f)
                    ftp.quit()
                    log.info(f"[H5 checkpoint] FTP: {remote_name} "
                             f"({os.path.getsize(h5_path)/1e6:.0f} MB) "
                             f"— {n_so_far} utterances")
                except Exception as e:
                    log.warning(f"[H5 checkpoint] FTP upload failed: {e}")
                # Drive copy (second backup)
                try:
                    os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
                    drive_dst = os.path.join(DRIVE_BACKUP_DIR, remote_name)
                    shutil.copy2(h5_path, drive_dst)
                    log.info(f"[H5 checkpoint] Drive: {remote_name} copied.")
                except Exception as e:
                    log.warning(f"[H5 checkpoint] Drive copy failed: {e}")
                hf = h5py.File(h5_path, 'a')
                since_upload = 0
    finally:
        hf.flush()
        hf.close()

    log.info(f"Extraction done. Errors: {errors}")

    # Load everything
    X, y, ids = [], [], []
    with h5py.File(h5_path, 'r') as hf:
        for utt_id in hf.keys():
            X.append(hf[utt_id]['feat'][:])
            y.append(int(hf[utt_id]['label'][()]))
            ids.append(utt_id)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# =============================================================================
# SECTION 7 — Distortion Generation
# =============================================================================

def add_awgn(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power  = np.mean(waveform ** 2)
    if sig_power < 1e-12:
        return waveform
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(len(waveform)) * np.sqrt(noise_power)
    return (waveform + noise).astype(np.float32)

def apply_ffmpeg_distortion(src_wav: str, dst_wav: str,
                             ffmpeg_args: List[str]) -> bool:
    """
    Apply FFmpeg codec/filter distortion: src_wav → dst_wav (WAV output).
    Opus codec requires an intermediate OGG container — this is handled
    automatically: if libopus is in ffmpeg_args, encode to .ogg first,
    then decode back to WAV.
    """
    is_opus = any("libopus" in a for a in ffmpeg_args)

    if is_opus:
        # Opus cannot be muxed into WAV — encode to OGG, decode to WAV
        tmp_ogg = dst_wav.replace(".wav", ".ogg")
        # Step 1: encode to OGG/Opus
        opus_args = [a for a in ffmpeg_args
                     if a not in ["-ar", str(SR), "-ac", "1", "-f", "wav"]]
        cmd_enc = ["ffmpeg", "-y", "-i", src_wav] + opus_args + [tmp_ogg]
        try:
            r = subprocess.run(cmd_enc, capture_output=True, timeout=30)
            if r.returncode != 0:
                return False
            # Step 2: decode OGG back to WAV
            cmd_dec = ["ffmpeg", "-y", "-i", tmp_ogg,
                       "-ar", str(SR), "-ac", "1", "-f", "wav", dst_wav]
            r2 = subprocess.run(cmd_dec, capture_output=True, timeout=30)
            return r2.returncode == 0
        except Exception as e:
            log.warning(f"FFmpeg Opus pipeline failed for {src_wav}: {e}")
            return False
        finally:
            if os.path.exists(tmp_ogg):
                os.unlink(tmp_ogg)
    else:
        cmd = ["ffmpeg", "-y", "-i", src_wav] + ffmpeg_args + \
              ["-ar", str(SR), "-ac", "1", "-f", "wav", dst_wav]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            log.warning(f"FFmpeg failed for {src_wav}: {e}")
            return False

def get_eval_subset(eval_utt_ids: List[str], labels_dict: Dict[str, int],
                    n: int = DISTORTION_EVAL_N, seed: int = 42) -> List[str]:
    """
    Stratified subsample of eval utterances maintaining bonafide/spoof ratio.
    Used for distortion conditions to keep each condition tractable on Colab.
    """
    rng = np.random.default_rng(seed)
    bon = [u for u in eval_utt_ids if labels_dict.get(u) == 0]
    spo = [u for u in eval_utt_ids if labels_dict.get(u) == 1]
    ratio = len(bon) / max(len(bon) + len(spo), 1)
    n_bon = int(n * ratio)
    n_spo = n - n_bon
    sel_bon = rng.choice(bon, min(n_bon, len(bon)), replace=False).tolist()
    sel_spo = rng.choice(spo, min(n_spo, len(spo)), replace=False).tolist()
    subset  = sel_bon + sel_spo
    rng.shuffle(subset)
    log.info(f"Eval subset: {len(subset)} utterances "
             f"({len(sel_bon)} bonafide + {len(sel_spo)} spoof)")
    return subset


def generate_and_extract_condition(
        utt_ids: List[str],
        labels_dict: Dict[str, int],
        cond_tag: str,
        ffmpeg_args: Optional[List[str]],
        h5_path: str,
        snr_db: Optional[float] = None,
        combo: Optional[str]    = None,
        batch_size: int         = DISTORTION_BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    v1.12: Replaces separate generate_distorted_set() + extract_features_batch()
    for distortion conditions.

    Processes utterances in batches of `batch_size`:
      For each batch:
        1. Generate distorted wav  (written to /tmp, not kept)
        2. Extract HOS features    (appended to HDF5 immediately)
        3. Delete the wav          (frees disk)
        4. Checkpoint HDF5         (FTP + Drive upload after each batch)

    Resumable at any batch boundary — already-done utterance IDs are read
    from the HDF5 on startup and skipped.
    """
    # --- Recover HDF5 if missing ---
    if not os.path.exists(h5_path):
        recover_h5(h5_path)

    # --- Load already-done IDs ---
    done_ids = set()
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as hf:
                done_ids = set(hf.keys())
            log.info(f"  [{cond_tag}] Resuming: {len(done_ids)} / {len(utt_ids)} done")
        except Exception as e:
            log.warning(f"  [{cond_tag}] HDF5 corrupt ({e}) — starting fresh")
            os.remove(h5_path)
            done_ids = set()

    todo = [u for u in utt_ids if u not in done_ids]
    if not todo:
        log.info(f"  [{cond_tag}] All utterances already extracted.")
    else:
        log.info(f"  [{cond_tag}] To process: {len(todo)} utterances "
                 f"in batches of {batch_size}")

    # --- Batch loop ---
    tmp_dir = os.path.join(DISTORTED_ROOT, f"tmp_{cond_tag}")
    os.makedirs(tmp_dir, exist_ok=True)
    errors = 0
    n_batches = max(1, (len(todo) + batch_size - 1) // batch_size)

    for b_idx in range(n_batches):
        batch = todo[b_idx * batch_size : (b_idx + 1) * batch_size]
        if not batch:
            break

        pct = int(100 * (len(done_ids) + b_idx * batch_size) / max(len(utt_ids), 1))
        log.info(f"  [{cond_tag}] Batch {b_idx+1}/{n_batches}  "
                 f"({pct}% overall)  generating+extracting {len(batch)} utterances …")

        batch_feats = {}   # utt_id → feat array

        for utt_id in batch:
            wav_tmp = os.path.join(tmp_dir, utt_id + ".wav")
            try:
                src_path = get_audio_path(utt_id, "eval")
                wav, sr  = sf.read(src_path, dtype='float32')
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != SR:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)

                # --- Apply distortion ---
                if combo == "opus_awgn10":
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False,
                                                     dir=tmp_dir) as ti:
                        sf.write(ti.name, wav, SR)
                    tmp_ogg = ti.name.replace('.wav', '.ogg')
                    tmp_decoded = ti.name.replace('.wav', '_dec.wav')
                    subprocess.run(["ffmpeg", "-y", "-i", ti.name,
                                    "-codec:a", "libopus", "-b:a", "6k",
                                    tmp_ogg], capture_output=True, timeout=30)
                    subprocess.run(["ffmpeg", "-y", "-i", tmp_ogg,
                                    "-ar", str(SR), "-ac", "1",
                                    "-f", "wav", tmp_decoded],
                                   capture_output=True, timeout=30)
                    coded, _ = sf.read(tmp_decoded, dtype='float32')
                    wav_dist = add_awgn(coded, 10.0)
                    for f in [ti.name, tmp_ogg, tmp_decoded]:
                        if os.path.exists(f): os.unlink(f)

                elif combo == "mp3_awgn5":
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False,
                                                     dir=tmp_dir) as ti:
                        sf.write(ti.name, wav, SR)
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False,
                                                     dir=tmp_dir) as to:
                        apply_ffmpeg_distortion(ti.name, to.name,
                            ["-codec:a", "libmp3lame", "-b:a", "64k"])
                        coded, _ = sf.read(to.name, dtype='float32')
                    wav_dist = add_awgn(coded, 5.0)
                    os.unlink(ti.name); os.unlink(to.name)

                elif snr_db is not None:
                    wav_dist = add_awgn(wav, snr_db)

                elif ffmpeg_args is not None:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False,
                                                     dir=tmp_dir) as ti:
                        sf.write(ti.name, wav, SR)
                    ok = apply_ffmpeg_distortion(ti.name, wav_tmp, ffmpeg_args)
                    os.unlink(ti.name)
                    if ok:
                        wav_dist, _ = sf.read(wav_tmp, dtype='float32')
                        os.unlink(wav_tmp)
                    else:
                        errors += 1
                        continue
                else:
                    wav_dist = wav

                # --- Extract features ---
                feat = extract_hos_features(wav_dist, SR)
                batch_feats[utt_id] = feat

            except Exception as e:
                errors += 1
                log.warning(f"    Skipped {utt_id}: {e}")
            finally:
                if os.path.exists(wav_tmp):
                    os.unlink(wav_tmp)

        # --- Append batch to HDF5 (safe close→write→checkpoint) ---
        if batch_feats:
            hf = h5py.File(h5_path, 'a')
            for utt_id, feat in batch_feats.items():
                grp = hf.create_group(utt_id)
                grp.create_dataset('feat',  data=feat)
                grp.create_dataset('label', data=labels_dict.get(utt_id, -1))
                done_ids.add(utt_id)
            hf.flush()
            hf.close()

            # Checkpoint to FTP and Drive
            remote_name = os.path.basename(h5_path)
            try:
                ftp = get_ftp_connection()
                with open(h5_path, "rb") as f:
                    ftp.storbinary(f"STOR {remote_name}", f)
                ftp.quit()
                log.info(f"  [{cond_tag}] FTP checkpoint: {len(done_ids)} utterances "
                         f"({os.path.getsize(h5_path)/1e6:.0f} MB)")
            except Exception as e:
                log.warning(f"  [{cond_tag}] FTP checkpoint failed: {e}")
            try:
                shutil.copy2(h5_path, os.path.join(DRIVE_BACKUP_DIR, remote_name))
                log.info(f"  [{cond_tag}] Drive checkpoint: {remote_name}")
            except Exception as e:
                log.warning(f"  [{cond_tag}] Drive checkpoint failed: {e}")

    # Clean up tmp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log.info(f"  [{cond_tag}] Extraction done. Total={len(done_ids)} Errors={errors}")

    # Load and return all features
    if not os.path.exists(h5_path):
        log.warning(f"  [{cond_tag}] HDF5 not found after extraction — "
                    f"all utterances may have failed. Returning empty arrays.")
        return np.array([], dtype=np.float32).reshape(0, 1), np.array([], dtype=np.int32)

    X, y = [], []
    with h5py.File(h5_path, 'r') as hf:
        for utt_id in hf.keys():
            X.append(hf[utt_id]['feat'][:])
            y.append(int(hf[utt_id]['label'][()]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# =============================================================================
# SECTION 8 — EER and min-tDCF Computation
# =============================================================================

def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes EER (%) given binary true labels and continuous scores
    (higher score = more likely spoof).
    Returns (eer_percent, threshold).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    # Find threshold where FPR ≈ FNR
    idx  = np.nanargmin(np.abs(fnr - fpr))
    eer  = (fpr[idx] + fnr[idx]) / 2.0 * 100.0
    return float(eer), float(thresholds[idx])

def compute_min_tdcf(y_true: np.ndarray, scores: np.ndarray,
                     p_spoof: float = 0.05,
                     c_miss: float  = 1.0,
                     c_fa:   float  = 10.0) -> float:
    """
    ASVspoof-style min-tDCF (simplified CM-only formulation).
    Normalised by the default cost of a perfect ASV system.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    tdcf = c_miss * fnr * p_spoof + c_fa * fpr * (1 - p_spoof)
    return float(np.min(tdcf))

# =============================================================================
# SECTION 9 — XGBoost Training
# =============================================================================

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  model_path: str) -> xgb.XGBClassifier:
    """Trains XGBoost, saves model to model_path. Resumes if model exists locally
    or on FTP (downloads on session wipe)."""
    # --- FTP recovery: restore model if local disk was wiped ---
    if not os.path.exists(model_path):
        remote_name = os.path.basename(model_path)
        log.info(f"Local model not found — attempting FTP recovery: {remote_name}")
        if download_from_ftp(remote_name, model_path):
            log.info(f"  Recovered model from FTP ({os.path.getsize(model_path)/1e3:.1f} KB)")
        else:
            log.info(f"  No FTP copy found. Will train from scratch.")

    if os.path.exists(model_path):
        log.info(f"Loading existing XGBoost model from {model_path}")
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
        return clf

    log.info("Training XGBoost classifier …")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    clf = xgb.XGBClassifier(
        n_estimators   = 500,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        scale_pos_weight= scale,
        use_label_encoder=False,
        eval_metric     = 'logloss',
        random_state    = 42,
        n_jobs          = -1,
        tree_method     = 'hist',
    )
    clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=50)
    clf.save_model(model_path)
    log.info(f"XGBoost model saved: {model_path}")
    upload_to_ftp(model_path, os.path.basename(model_path))
    return clf

# =============================================================================
# SECTION 10 — LFCC-GMM Baseline (local implementation)
# =============================================================================

def run_lfcc_gmm_baseline(eval_utt_ids: List[str], labels_dict: Dict[str, int],
                           condition_tag: str,
                           audio_dir_override: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scores utterances using the locally fitted LFCC-GMM (log-likelihood ratio).
    Saves scores to JSON checkpoint; reloads on resume.
    Returns (scores, y_true).
    """
    scores_path = os.path.join(CHECKPOINT_DIR,
                               f"lfcc_scores_{condition_tag}.json")
    if os.path.exists(scores_path):
        log.info(f"  LFCC-GMM scores found for [{condition_tag}], loading.")
        with open(scores_path) as f:
            data = json.load(f)
        return np.array(data['scores']), np.array(data['labels'])

    log.info(f"  Computing LFCC-GMM scores [{condition_tag}] …")
    scores, y_true = _lfcc_gmm_local(eval_utt_ids, labels_dict,
                                      condition_tag, audio_dir_override)
    result = {'scores': scores.tolist(), 'labels': y_true.tolist()}
    with open(scores_path, 'w') as f:
        json.dump(result, f)
    upload_to_ftp(scores_path, os.path.basename(scores_path))
    return scores, y_true

# ---------------------------------------------------------------------------
# Local LFCC-GMM implementation (used when SpeechBrain hub unavailable)
# ---------------------------------------------------------------------------

def extract_lfcc(waveform: np.ndarray, sr: int,
                 n_lfcc: int = 20, n_fft: int = 512,
                 hop_length: int = 160) -> np.ndarray:
    """
    Linear Frequency Cepstral Coefficients via linear filterbank + DCT.
    Returns (n_frames, n_lfcc) array.
    """
    # Linear filterbank (equally spaced in Hz, not mel)
    n_filters = 70
    f_min, f_max = 0.0, sr / 2.0
    freq_bins  = np.linspace(f_min, f_max, n_fft // 2 + 1)
    filter_edges = np.linspace(f_min, f_max, n_filters + 2)

    # Build linear filterbank matrix
    filterbank_lf = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(n_filters):
        f_left   = filter_edges[m]
        f_center = filter_edges[m + 1]
        f_right  = filter_edges[m + 2]
        for k, f in enumerate(freq_bins):
            if f_left <= f <= f_center:
                filterbank_lf[m, k] = (f - f_left) / (f_center - f_left + 1e-12)
            elif f_center < f <= f_right:
                filterbank_lf[m, k] = (f_right - f) / (f_right - f_center + 1e-12)

    # STFT
    _, _, S = ss.stft(waveform, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    power = np.abs(S) ** 2  # (n_fft//2+1, n_frames)

    # Apply filterbank
    log_energy = np.log(filterbank_lf @ power + 1e-12)  # (n_filters, n_frames)

    # DCT
    from scipy.fft import dct
    lfcc = dct(log_energy, axis=0, norm='ortho')[:n_lfcc, :]  # (n_lfcc, n_frames)
    return lfcc.T  # (n_frames, n_lfcc)

# GMM training lives here as a module-level cache to avoid re-fitting
_GMM_BONAFIDE = None
_GMM_SPOOF    = None

def _fit_lfcc_gmm_if_needed(train_utt_ids: List[str],
                              labels_dict: Dict[str, int]):
    global _GMM_BONAFIDE, _GMM_SPOOF
    gmm_path = os.path.join(CHECKPOINT_DIR, "lfcc_gmm_models.npz")
    if _GMM_BONAFIDE is not None:
        return
    # --- FTP recovery: restore GMM if local disk was wiped ---
    if not os.path.exists(gmm_path):
        log.info("Local GMM not found — attempting FTP recovery: lfcc_gmm_models.npz")
        if download_from_ftp("lfcc_gmm_models.npz", gmm_path):
            log.info(f"  Recovered GMM from FTP ({os.path.getsize(gmm_path)/1e6:.1f} MB)")
        else:
            log.info("  No FTP copy found. Will refit GMM.")
    if os.path.exists(gmm_path):
        try:
            log.info("Loading cached LFCC-GMM parameters …")
            data = np.load(gmm_path, allow_pickle=True)
            _GMM_BONAFIDE = data['gmm_bon'].item()
            _GMM_SPOOF    = data['gmm_spo'].item()
            log.info("LFCC-GMM loaded from cache.")
            return
        except Exception as e:
            log.warning(f"Cached GMM corrupt ({e}) — deleting and refitting.")
            os.remove(gmm_path)
            # Also purge from FTP so recovery does not re-download corrupt copy
            try:
                ftp = get_ftp_connection()
                ftp.delete("lfcc_gmm_models.npz")
                ftp.quit()
                log.info("Deleted corrupt lfcc_gmm_models.npz from FTP.")
            except Exception:
                pass

    log.info("Fitting LFCC-GMM on train partition …")
    from sklearn.mixture import GaussianMixture
    bon_feats, spo_feats = [], []

    MAX_PER_CLASS  = 500    # 500 bonafide + 500 spoof = 1000 total, balanced
    MAX_FRAMES_PER = 200    # cap frames per utterance
    GMM_COMPONENTS = 128    # diagonal GMM — matches ASVspoof baseline practice

    # Split by class first, sample equally — avoids empty-class vstack crash
    bon_ids = [u for u in train_utt_ids if labels_dict.get(u) == 0]
    spo_ids = [u for u in train_utt_ids if labels_dict.get(u) == 1]
    log.info(f"  Available — bonafide: {len(bon_ids)}  spoof: {len(spo_ids)}")
    selected = bon_ids[:MAX_PER_CLASS] + spo_ids[:MAX_PER_CLASS]
    log.info(f"  Using {len(selected)} utterances ({MAX_PER_CLASS} per class)")

    for utt_id in tqdm(selected, desc="LFCC-GMM train", ncols=80):
        try:
            wav_path = get_audio_path(utt_id, "train")
            wav, sr  = sf.read(wav_path, dtype='float32')
            if wav.ndim > 1: wav = wav.mean(axis=1)
            lfcc = extract_lfcc(wav, sr)
            if len(lfcc) > MAX_FRAMES_PER:
                idx  = np.random.choice(len(lfcc), MAX_FRAMES_PER, replace=False)
                lfcc = lfcc[idx]
            if labels_dict[utt_id] == 0:
                bon_feats.append(lfcc)
            else:
                spo_feats.append(lfcc)
        except Exception as e:
            log.warning(f"  LFCC skip {utt_id}: {e}")

    if not bon_feats or not spo_feats:
        raise RuntimeError(
            f"GMM training failed: bonafide={len(bon_feats)} "
            f"spoof={len(spo_feats)} arrays. Check get_audio_path for train."
        )

    X_bon = np.vstack(bon_feats).astype(np.float32)
    X_spo = np.vstack(spo_feats).astype(np.float32)
    log.info(f"  Bonafide frames: {len(X_bon)}, Spoof frames: {len(X_spo)}")
    _GMM_BONAFIDE = GaussianMixture(
        GMM_COMPONENTS, max_iter=20, random_state=42,
        covariance_type='diag').fit(X_bon)
    _GMM_SPOOF = GaussianMixture(
        GMM_COMPONENTS, max_iter=20, random_state=42,
        covariance_type='diag').fit(X_spo)
    np.savez(gmm_path, gmm_bon=_GMM_BONAFIDE, gmm_spo=_GMM_SPOOF)
    log.info("LFCC-GMM models fitted and saved.")
    upload_to_ftp(gmm_path, "lfcc_gmm_models.npz")

def _lfcc_gmm_local(eval_utt_ids: List[str], labels_dict: Dict[str, int],
                    condition_tag: str,
                    audio_dir_override: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Score eval utterances with the fitted LFCC-GMM (llr scoring)."""
    scores, y_true = [], []
    for utt_id in tqdm(eval_utt_ids[:2000], desc=f"LFCC score [{condition_tag}]",
                       ncols=80):
        try:
            if audio_dir_override:
                wav_path = os.path.join(audio_dir_override, utt_id + ".wav")
            else:
                wav_path = get_audio_path(utt_id, "eval")
            wav, sr = sf.read(wav_path, dtype='float32')
            if wav.ndim > 1: wav = wav.mean(axis=1)
            lfcc = extract_lfcc(wav, sr)
            llr  = (_GMM_SPOOF.score(lfcc) - _GMM_BONAFIDE.score(lfcc))
            scores.append(llr)
            y_true.append(labels_dict.get(utt_id, -1))
        except Exception as e:
            log.warning(f"LFCC score error [{utt_id}]: {e}")
    return np.array(scores), np.array(y_true)

# =============================================================================
# SECTION 11 — Plotting
# =============================================================================

def plot_eer_vs_bitrate(conditions: List[str], eer_hos: List[float],
                         eer_lfcc: List[float], bitrates: List[int],
                         codec_label: str, fig_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(bitrates, eer_hos,  'o-', color='#2196F3', lw=2,
            label='HOS-XGBoost')
    ax.plot(bitrates, eer_lfcc, 's--', color='#F44336', lw=2,
            label='LFCC-GMM')
    ax.set_xlabel(f"{codec_label} Bitrate (kbps)", fontsize=12)
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title(f"EER vs {codec_label} Bitrate", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    log.info(f"Saved: {fig_path}")

def plot_eer_vs_snr(snrs: List[int], eer_hos: List[float],
                    eer_lfcc: List[float], fig_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(snrs, eer_hos,  'o-', color='#2196F3', lw=2, label='HOS-XGBoost')
    ax.plot(snrs, eer_lfcc, 's--', color='#F44336', lw=2, label='LFCC-GMM')
    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB)  [lower = more noise →]", fontsize=12)
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title("EER vs AWGN Noise Level", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    log.info(f"Saved: {fig_path}")

def plot_degradation_radar(conditions: List[str],
                            deg_hos: List[float],
                            deg_lfcc: List[float],
                            fig_path: str):
    """Radar (spider) chart of relative degradation ratio per condition."""
    N = len(conditions)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    hos_vals  = deg_hos  + [deg_hos[0]]
    lfcc_vals = deg_lfcc + [deg_lfcc[0]]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300,
                            subplot_kw=dict(polar=True))
    ax.plot(angles, hos_vals,  'o-', color='#2196F3', lw=2,
            label='HOS-XGBoost')
    ax.fill(angles, hos_vals,  color='#2196F3', alpha=0.15)
    ax.plot(angles, lfcc_vals, 's--', color='#F44336', lw=2,
            label='LFCC-GMM')
    ax.fill(angles, lfcc_vals, color='#F44336', alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_title("Relative Degradation Ratio\n(EER_distorted / EER_clean)",
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {fig_path}")

def plot_eer_heatmap(all_conditions: List[str], systems: List[str],
                     eer_matrix: np.ndarray, fig_path: str):
    """Heatmap of EER (%) across all conditions × systems."""
    fig, ax = plt.subplots(figsize=(max(10, len(all_conditions) * 0.7), 4),
                            dpi=300)
    im = ax.imshow(eer_matrix, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=50)
    ax.set_xticks(range(len(all_conditions)))
    ax.set_xticklabels(all_conditions, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems, fontsize=10)
    plt.colorbar(im, ax=ax, label="EER (%)")
    for i in range(len(systems)):
        for j in range(len(all_conditions)):
            ax.text(j, i, f"{eer_matrix[i,j]:.1f}",
                    ha='center', va='center', fontsize=7, color='black')
    ax.set_title("EER (%) across Distortion Conditions × Detection System",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {fig_path}")

# =============================================================================
# SECTION 12 — Main Orchestration
# =============================================================================

def main():
    log.info("=" * 70)
    log.info("Experiment 5 — HOS Robustness under Codec/Channel Distortion")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 0: Start background FTP heartbeat
    # ------------------------------------------------------------------
    hb_thread, hb_stop = start_ftp_heartbeat(CHECKPOINT_DIR,
                                              interval=FTP_HEARTBEAT_INTERVAL)

    # ------------------------------------------------------------------
    # Step 1: Mount Drive and prepare dataset
    # ------------------------------------------------------------------
    mount_drive()
    prepare_dataset()

    # ------------------------------------------------------------------
    # Step 2: Parse protocols
    # ------------------------------------------------------------------
    train_labels = parse_protocol(
        os.path.join(PROTOCOL_LOCAL, "ASVspoof2019.LA.cm.train.trn.txt"))
    eval_labels  = parse_protocol(
        os.path.join(PROTOCOL_LOCAL, "ASVspoof2019.LA.cm.eval.trl.txt"))

    train_utt_ids = list(train_labels.keys())
    eval_utt_ids  = list(eval_labels.keys())
    log.info(f"Train: {len(train_utt_ids)} | Eval: {len(eval_utt_ids)}")

    # ------------------------------------------------------------------
    # Step 3: Extract HOS features — TRAIN
    # ------------------------------------------------------------------
    train_h5 = os.path.join(CHECKPOINT_DIR, "hos_features_train.h5")
    X_train, y_train = extract_features_batch(
        train_utt_ids, train_labels, train_h5, partition="train")
    log.info(f"Train features: {X_train.shape}  Labels: {np.bincount(y_train)}")
    upload_to_ftp(train_h5, "hos_features_train.h5")

    # ------------------------------------------------------------------
    # Step 4: Train XGBoost
    # ------------------------------------------------------------------
    model_path = os.path.join(CHECKPOINT_DIR, "xgb_model.json")
    clf = train_xgboost(X_train, y_train, model_path)

    # ------------------------------------------------------------------
    # Step 5: Extract features and evaluate — CLEAN EVAL
    # ------------------------------------------------------------------
    eval_clean_h5 = os.path.join(CHECKPOINT_DIR, "hos_features_eval_clean.h5")
    X_eval_clean, y_eval_clean = extract_features_batch(
        eval_utt_ids, eval_labels, eval_clean_h5, partition="eval")
    log.info(f"Clean eval features: {X_eval_clean.shape}")
    upload_to_ftp(eval_clean_h5, "hos_features_eval_clean.h5")

    clean_scores = clf.predict_proba(X_eval_clean)[:, 1]
    eer_clean_hos, _  = compute_eer(y_eval_clean, clean_scores)
    tdcf_clean_hos    = compute_min_tdcf(y_eval_clean, clean_scores)
    log.info(f"[Clean] HOS-XGBoost  EER={eer_clean_hos:.2f}%  min-tDCF={tdcf_clean_hos:.4f}")

    # LFCC-GMM clean baseline
    _fit_lfcc_gmm_if_needed(train_utt_ids, train_labels)
    lfcc_scores_clean, lfcc_y_clean = run_lfcc_gmm_baseline(
        eval_utt_ids, eval_labels, "clean")
    eer_clean_lfcc, _ = compute_eer(lfcc_y_clean, lfcc_scores_clean)
    log.info(f"[Clean] LFCC-GMM     EER={eer_clean_lfcc:.2f}%")

    # ------------------------------------------------------------------
    # Step 6–8: Distorted conditions loop
    # ------------------------------------------------------------------
    # Use stratified 5000-utterance subset for distortion conditions
    dist_utt_ids = get_eval_subset(eval_utt_ids, eval_labels, n=DISTORTION_EVAL_N)

    results = {
        "clean": {
            "hos_eer": eer_clean_hos,
            "hos_tdcf": tdcf_clean_hos,
            "lfcc_eer": eer_clean_lfcc,
            "hos_deg": 1.0,
            "lfcc_deg": 1.0,
        }
    }

    # Progress checkpoint: load existing results if resuming
    results_path = os.path.join(CHECKPOINT_DIR, "results_exp5.json")

    def _safe_load_results(path: str) -> dict:
        """Load results JSON safely — return None if missing, empty or corrupt."""
        if not os.path.exists(path):
            return None
        if os.path.getsize(path) == 0:
            log.warning(f"results_exp5.json is zero bytes — discarding.")
            os.remove(path)
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"results_exp5.json corrupt ({e}) — discarding.")
            os.remove(path)
            return None

    # Try local first, then FTP
    loaded = _safe_load_results(results_path)
    if loaded is None and download_from_ftp("results_exp5.json", results_path):
        loaded = _safe_load_results(results_path)
    if loaded is not None:
        results = loaded
        log.info(f"Resumed results: {list(results.keys())}")

    for (cond_tag, ffmpeg_args, human_label) in DISTORTIONS:
        if cond_tag in results:
            log.info(f"  [{cond_tag}] Already in results, skipping.")
            continue

        log.info(f"\n--- Processing condition: {cond_tag} ({human_label}) ---")

        snr_db = AWGN_SNRS.get(cond_tag, None)
        combo  = None
        if cond_tag == "combo_opus_awgn10":
            combo = "opus_awgn10"
        elif cond_tag == "combo_mp3_awgn5":
            combo = "mp3_awgn5"

        # Generate distorted audio + extract features in batches (resumable)
        cond_h5 = os.path.join(CHECKPOINT_DIR, f"hos_features_eval_{cond_tag}.h5")
        X_cond, y_cond = generate_and_extract_condition(
            dist_utt_ids, eval_labels, cond_tag, ffmpeg_args,
            cond_h5, snr_db=snr_db, combo=combo)

        # HOS-XGBoost EER
        cond_scores_hos = clf.predict_proba(X_cond)[:, 1]
        eer_cond_hos, _ = compute_eer(y_cond, cond_scores_hos)
        tdcf_cond_hos   = compute_min_tdcf(y_cond, cond_scores_hos)

        # LFCC-GMM EER on distorted audio
        # Re-generate distortion on a 500-utterance subsample, score with LFCC-GMM
        # Delete each wav immediately after scoring to save disk
        lfcc_scores_cond_path = os.path.join(CHECKPOINT_DIR,
                                              f"lfcc_scores_{cond_tag}.json")
        if os.path.exists(lfcc_scores_cond_path):
            log.info(f"  LFCC-GMM scores found for [{cond_tag}], loading.")
            with open(lfcc_scores_cond_path) as f:
                _d = json.load(f)
            lfcc_scores_cond = np.array(_d['scores'])
            lfcc_y_cond      = np.array(_d['labels'])
        else:
            log.info(f"  Computing LFCC-GMM scores on distorted audio [{cond_tag}] …")
            lfcc_subset   = dist_utt_ids[:500]
            lfcc_tmp_dir  = os.path.join(DISTORTED_ROOT, f"lfcc_tmp_{cond_tag}")
            os.makedirs(lfcc_tmp_dir, exist_ok=True)
            lfcc_scores_cond, lfcc_y_cond = [], []

            for u in tqdm(lfcc_subset, desc=f"LFCC [{cond_tag}]", ncols=80):
                wav_tmp = os.path.join(lfcc_tmp_dir, u + ".wav")
                try:
                    src_path = get_audio_path(u, "eval")
                    wav, sr  = sf.read(src_path, dtype='float32')
                    if wav.ndim > 1: wav = wav.mean(axis=1)
                    if sr != SR:
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)

                    # Apply same distortion as HOS condition
                    if combo == "opus_awgn10":
                        with tempfile.NamedTemporaryFile(suffix='.wav',
                                delete=False, dir=lfcc_tmp_dir) as ti:
                            sf.write(ti.name, wav, SR)
                        tmp_ogg = ti.name.replace('.wav', '.ogg')
                        tmp_dec = ti.name.replace('.wav', '_dec.wav')
                        subprocess.run(["ffmpeg", "-y", "-i", ti.name,
                                        "-codec:a", "libopus", "-b:a", "6k",
                                        tmp_ogg], capture_output=True, timeout=30)
                        subprocess.run(["ffmpeg", "-y", "-i", tmp_ogg,
                                        "-ar", str(SR), "-ac", "1",
                                        "-f", "wav", tmp_dec],
                                       capture_output=True, timeout=30)
                        coded, _ = sf.read(tmp_dec, dtype='float32')
                        wav_dist = add_awgn(coded, 10.0)
                        for f in [ti.name, tmp_ogg, tmp_dec]:
                            if os.path.exists(f): os.unlink(f)
                    elif combo == "mp3_awgn5":
                        with tempfile.NamedTemporaryFile(suffix='.wav',
                                delete=False, dir=lfcc_tmp_dir) as ti:
                            sf.write(ti.name, wav, SR)
                        apply_ffmpeg_distortion(ti.name, wav_tmp,
                            ["-codec:a", "libmp3lame", "-b:a", "64k"])
                        coded, _ = sf.read(wav_tmp, dtype='float32')
                        wav_dist = add_awgn(coded, 5.0)
                        os.unlink(ti.name)
                    elif snr_db is not None:
                        wav_dist = add_awgn(wav, snr_db)
                    elif ffmpeg_args is not None:
                        with tempfile.NamedTemporaryFile(suffix='.wav',
                                delete=False, dir=lfcc_tmp_dir) as ti:
                            sf.write(ti.name, wav, SR)
                        ok = apply_ffmpeg_distortion(ti.name, wav_tmp, ffmpeg_args)
                        os.unlink(ti.name)
                        if ok:
                            wav_dist, _ = sf.read(wav_tmp, dtype='float32')
                        else:
                            continue
                    else:
                        wav_dist = wav

                    lfcc = extract_lfcc(wav_dist, SR)
                    llr  = (_GMM_SPOOF.score(lfcc) - _GMM_BONAFIDE.score(lfcc))
                    lfcc_scores_cond.append(llr)
                    lfcc_y_cond.append(eval_labels.get(u, -1))
                except Exception as e:
                    log.warning(f"  LFCC distort error [{u}]: {e}")
                finally:
                    if os.path.exists(wav_tmp):
                        os.unlink(wav_tmp)

            shutil.rmtree(lfcc_tmp_dir, ignore_errors=True)
            lfcc_scores_cond = np.array(lfcc_scores_cond)
            lfcc_y_cond      = np.array(lfcc_y_cond)
            # Save scores
            with open(lfcc_scores_cond_path, 'w') as f:
                json.dump({'scores': lfcc_scores_cond.tolist(),
                           'labels': lfcc_y_cond.tolist()}, f)
            upload_to_ftp(lfcc_scores_cond_path,
                          os.path.basename(lfcc_scores_cond_path))

        eer_cond_lfcc, _ = compute_eer(lfcc_y_cond, lfcc_scores_cond)

        deg_hos  = eer_cond_hos  / max(eer_clean_hos,  0.01)
        deg_lfcc = eer_cond_lfcc / max(eer_clean_lfcc, 0.01)

        results[cond_tag] = {
            "label":    human_label,
            "hos_eer":  round(eer_cond_hos, 3),
            "hos_tdcf": round(tdcf_cond_hos, 4),
            "lfcc_eer": round(eer_cond_lfcc, 3),
            "hos_deg":  round(deg_hos, 3),
            "lfcc_deg": round(deg_lfcc, 3),
        }
        log.info(f"  [{cond_tag}] HOS EER={eer_cond_hos:.2f}%  "
                 f"LFCC EER={eer_cond_lfcc:.2f}%  "
                 f"Deg(HOS)={deg_hos:.2f}x  Deg(LFCC)={deg_lfcc:.2f}x")

        # Save checkpoint after each condition
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        upload_to_ftp(results_path, "results_exp5.json")

    # ------------------------------------------------------------------
    # Step 9: Generate all figures
    # ------------------------------------------------------------------
    log.info("\nGenerating figures …")

    # Figure 1: EER vs MP3 bitrate
    mp3_tags   = ["mp3_032", "mp3_064", "mp3_128"]
    mp3_rates  = [32, 64, 128]
    mp3_hos    = [results[t]["hos_eer"]  for t in mp3_tags if t in results]
    mp3_lfcc   = [results[t]["lfcc_eer"] for t in mp3_tags if t in results]
    if mp3_hos:
        fig1 = os.path.join(CHECKPOINT_DIR, "fig_05_01_eer_vs_mp3.png")
        plot_eer_vs_bitrate(mp3_tags, mp3_hos, mp3_lfcc, mp3_rates,
                             "MP3", fig1)
        upload_to_ftp(fig1, "fig_05_01_eer_vs_mp3.png")

    # Figure 2: EER vs Opus bitrate
    opus_tags  = ["opus_006", "opus_012", "opus_024"]
    opus_rates = [6, 12, 24]
    opus_hos   = [results[t]["hos_eer"]  for t in opus_tags if t in results]
    opus_lfcc  = [results[t]["lfcc_eer"] for t in opus_tags if t in results]
    if opus_hos:
        fig2 = os.path.join(CHECKPOINT_DIR, "fig_05_02_eer_vs_opus.png")
        plot_eer_vs_bitrate(opus_tags, opus_hos, opus_lfcc, opus_rates,
                             "Opus", fig2)
        upload_to_ftp(fig2, "fig_05_02_eer_vs_opus.png")

    # Figure 3: EER vs AWGN
    awgn_tags  = ["awgn_00db", "awgn_05db", "awgn_10db", "awgn_20db"]
    awgn_snrs  = [0, 5, 10, 20]
    awgn_hos   = [results[t]["hos_eer"]  for t in awgn_tags if t in results]
    awgn_lfcc  = [results[t]["lfcc_eer"] for t in awgn_tags if t in results]
    if awgn_hos:
        fig3 = os.path.join(CHECKPOINT_DIR, "fig_05_03_eer_vs_awgn.png")
        plot_eer_vs_snr(awgn_snrs, awgn_hos, awgn_lfcc, fig3)
        upload_to_ftp(fig3, "fig_05_03_eer_vs_awgn.png")

    # Figure 4: Radar chart of degradation ratio
    ordered_tags = [t for t, _, _ in DISTORTIONS if t in results]
    ordered_labels = [results[t]["label"] for t in ordered_tags]
    deg_hos_list  = [results[t]["hos_deg"]  for t in ordered_tags]
    deg_lfcc_list = [results[t]["lfcc_deg"] for t in ordered_tags]
    if ordered_tags:
        fig4 = os.path.join(CHECKPOINT_DIR, "fig_05_04_degradation_radar.png")
        plot_degradation_radar(ordered_labels, deg_hos_list, deg_lfcc_list, fig4)
        upload_to_ftp(fig4, "fig_05_04_degradation_radar.png")

    # Figure 5: Full EER heatmap
    all_conds  = ["clean"] + ordered_tags
    cond_lbls  = ["Clean"] + ordered_labels
    eer_matrix = np.array([
        [results[t]["hos_eer"]  for t in all_conds],
        [results[t]["lfcc_eer"] for t in all_conds],
    ])
    fig5 = os.path.join(CHECKPOINT_DIR, "fig_05_05_eer_heatmap.png")
    plot_eer_heatmap(cond_lbls, ["HOS-XGBoost", "LFCC-GMM"],
                     eer_matrix, fig5)
    upload_to_ftp(fig5, "fig_05_05_eer_heatmap.png")

    # ------------------------------------------------------------------
    # Step 10: Final summary
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 5 RESULTS SUMMARY")
    log.info("=" * 70)
    header = f"{'Condition':<25} {'HOS EER':>8} {'LFCC EER':>9} {'Deg(HOS)':>10} {'Deg(LFCC)':>11}"
    log.info(header)
    log.info("-" * 70)
    for cond_tag in ["clean"] + [t for t, _, _ in DISTORTIONS]:
        if cond_tag not in results:
            continue
        r = results[cond_tag]
        label = r.get("label", "Clean")
        log.info(f"{label:<25} {r['hos_eer']:>7.2f}%  {r['lfcc_eer']:>8.2f}%  "
                 f"{r['hos_deg']:>9.2f}x  {r['lfcc_deg']:>10.2f}x")

    log.info("=" * 70)
    log.info("All outputs uploaded to FTP. Experiment 5 complete.")

    # ------------------------------------------------------------------
    # Step 11: Stop background heartbeat thread cleanly
    # ------------------------------------------------------------------
    log.info("[Heartbeat] Stopping FTP backup thread …")
    hb_stop.set()
    hb_thread.join(timeout=30)
    log.info("[Heartbeat] Done.")


# =============================================================================
if __name__ == "__main__":
    main()