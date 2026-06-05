# =============================================================================
# Program      : 06_exp5_figures.py
# Version      : 1.0
# Description  : Generates all publication-quality figures for Experiment 5
#                (Subband HOS Robustness under Codec/Channel Distortion).
#
# INPUT:
#                  - results_exp5.json  (downloaded from Google Drive or local)
#                    Fields per condition: hos_eer, lfcc_eer, hos_deg,
#                    lfcc_deg, hos_tdcf, label
#
# OUTPUT FIGURES (saved locally + uploaded to Google Drive):
#                  fig_06_01_eer_vs_mp3.png
#                      EER (%) vs MP3 bitrate for HOS-XGBoost and LFCC-GMM
#                  fig_06_02_eer_vs_opus.png
#                      EER (%) vs Opus bitrate
#                  fig_06_03_eer_vs_awgn.png
#                      EER (%) vs AWGN SNR (dB)
#                  fig_06_04_degradation_bar.png
#                      Grouped bar chart: HOS deg vs LFCC deg across all
#                      14 conditions, colour-coded by distortion regime
#                  fig_06_05_eer_heatmap.png
#                      Heatmap of EER (%) — rows: systems, cols: conditions
#                  fig_06_06_regime_scatter.png
#                      Scatter plot: HOS deg (x) vs LFCC deg (y), one point
#                      per condition, annotated, regime boundaries marked
#
# GPU Required : NO
# Dependencies : matplotlib, numpy, json
#
# Change Log   :
#   v1.0  2026-06-03  Initial version
# =============================================================================

# !pip install matplotlib numpy

# =============================================================================
# SECTION 0 — Imports and Configuration
# =============================================================================

import os
import json
import time
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# --- Google Drive Configuration ---
PROJECT_DIR = "/content/drive/MyDrive/paper/Subband_Kurtosis/"  # Persistent storage

# --- Paths ---
LOCAL_DIR  = "/content/exp5_figures"
os.makedirs(LOCAL_DIR, exist_ok=True)

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
    'lines.linewidth'  : 2.0,
    'lines.markersize' : 7,
})

# System colours — consistent across all figures
C_HOS  = '#1565C0'   # deep blue  — HOS-XGBoost
C_LFCC = '#B71C1C'   # deep red   — LFCC-GMM

# Regime colours for bar chart
C_CODEC = '#1976D2'  # blue
C_AWGN  = '#E53935'  # red
C_TEL   = '#43A047'  # green
C_COMBO = '#FB8C00'  # orange

# =============================================================================
# SECTION 1 — Google Drive Helpers
# =============================================================================

def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_path: str, remote_name: str, retries: int = 3):
    """Copy a local file to Google Drive project folder."""
    ensure_project_dir()
    dest_path = os.path.join(PROJECT_DIR, remote_name)
    for attempt in range(retries):
        try:
            shutil.copy2(local_path, dest_path)
            print(f"  Drive upload OK: {remote_name}")
            return
        except Exception as e:
            print(f"  Drive attempt {attempt+1} failed: {e}")
            time.sleep(2)
    print(f"  Drive upload FAILED: {remote_name}")

def load_from_drive(remote_name: str, local_path: str) -> bool:
    """Copy a file from Google Drive project folder to local path."""
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_name)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_path)
            print(f"  Drive download OK: {remote_name}")
            return True
        except Exception as e:
            print(f"  Drive download failed: {e}")
            return False
    else:
        print(f"  Drive file not found: {remote_name}")
        return False

def download_results() -> dict:
    """Download results_exp5.json from Google Drive."""
    local = os.path.join(LOCAL_DIR, "results_exp5.json")
    # Try local first
    if os.path.exists(local):
        print("Loading results from local file.")
        with open(local) as f:
            return json.load(f)
    # Download from Google Drive
    print("Downloading results_exp5.json from Google Drive ...")
    if load_from_drive("results_exp5.json", local):
        with open(local) as f:
            return json.load(f)
    else:
        raise RuntimeError("Could not load results_exp5.json from Drive or local.")

# =============================================================================
# SECTION 2 — Data Preparation
# =============================================================================

# Canonical condition order matching DISTORTIONS list in experiment script
CONDITION_ORDER = [
    "mp3_032", "mp3_064", "mp3_128",
    "opus_006", "opus_012", "opus_024",
    "awgn_00db", "awgn_05db", "awgn_10db", "awgn_20db",
    "telephone",
    "combo_mp3tel", "combo_opus_awgn10", "combo_mp3_awgn5",
]

REGIME_MAP = {
    "mp3_032"        : "Codec",
    "mp3_064"        : "Codec",
    "mp3_128"        : "Codec",
    "opus_006"       : "Codec",
    "opus_012"       : "Codec",
    "opus_024"       : "Codec",
    "awgn_00db"      : "AWGN",
    "awgn_05db"      : "AWGN",
    "awgn_10db"      : "AWGN",
    "awgn_20db"      : "AWGN",
    "telephone"      : "Telephone",
    "combo_mp3tel"   : "Combo",
    "combo_opus_awgn10": "Combo",
    "combo_mp3_awgn5": "Combo",
}

REGIME_COLORS = {
    "Codec"     : C_CODEC,
    "AWGN"      : C_AWGN,
    "Telephone" : C_TEL,
    "Combo"     : C_COMBO,
}

SHORT_LABELS = {
    "mp3_032"          : "MP3\n32k",
    "mp3_064"          : "MP3\n64k",
    "mp3_128"          : "MP3\n128k",
    "opus_006"         : "Opus\n6k",
    "opus_012"         : "Opus\n12k",
    "opus_024"         : "Opus\n24k",
    "awgn_00db"        : "AWGN\n0dB",
    "awgn_05db"        : "AWGN\n5dB",
    "awgn_10db"        : "AWGN\n10dB",
    "awgn_20db"        : "AWGN\n20dB",
    "telephone"        : "Tel\nBPF",
    "combo_mp3tel"     : "MP3+\nTel",
    "combo_opus_awgn10": "Opus+\nAWGN",
    "combo_mp3_awgn5"  : "MP3+\nAWGN",
}

def extract_series(results: dict, key: str) -> list:
    return [results[c][key] for c in CONDITION_ORDER if c in results]

def save_and_upload(fig, fname: str):
    path = os.path.join(LOCAL_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fname}")
    save_to_drive(path, fname)

# =============================================================================
# SECTION 3 — Figure 1: EER vs MP3 Bitrate
# =============================================================================

def fig_eer_vs_mp3(results: dict):
    tags   = ["mp3_032", "mp3_064", "mp3_128"]
    rates  = [32, 64, 128]
    hos    = [results[t]["hos_eer"]  for t in tags]
    lfcc   = [results[t]["lfcc_eer"] for t in tags]
    clean_hos  = results["clean"]["hos_eer"]
    clean_lfcc = results["clean"]["lfcc_eer"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.axhline(clean_hos,  color=C_HOS,  lw=1.2, ls=':', alpha=0.6,
               label='HOS clean baseline')
    ax.axhline(clean_lfcc, color=C_LFCC, lw=1.2, ls=':', alpha=0.6,
               label='LFCC clean baseline')
    ax.plot(rates, hos,  'o-', color=C_HOS,  label='HOS-XGBoost')
    ax.plot(rates, lfcc, 's--', color=C_LFCC, label='LFCC-GMM')

    for r, h, l in zip(rates, hos, lfcc):
        ax.annotate(f'{h:.1f}', (r, h), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color=C_HOS)
        ax.annotate(f'{l:.1f}', (r, l), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, color=C_LFCC)

    ax.set_xlabel("MP3 Bitrate (kbps)")
    ax.set_ylabel("EER (%)")
    ax.set_title("EER vs MP3 Bitrate")
    ax.set_xticks(rates)
    ax.set_ylim(0, 75)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    save_and_upload(fig, "fig_06_01_eer_vs_mp3.png")

# =============================================================================
# SECTION 4 — Figure 2: EER vs Opus Bitrate
# =============================================================================

def fig_eer_vs_opus(results: dict):
    tags  = ["opus_006", "opus_012", "opus_024"]
    rates = [6, 12, 24]
    hos   = [results[t]["hos_eer"]  for t in tags]
    lfcc  = [results[t]["lfcc_eer"] for t in tags]
    clean_hos  = results["clean"]["hos_eer"]
    clean_lfcc = results["clean"]["lfcc_eer"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.axhline(clean_hos,  color=C_HOS,  lw=1.2, ls=':', alpha=0.6,
               label='HOS clean baseline')
    ax.axhline(clean_lfcc, color=C_LFCC, lw=1.2, ls=':', alpha=0.6,
               label='LFCC clean baseline')
    ax.plot(rates, hos,  'o-',  color=C_HOS,  label='HOS-XGBoost')
    ax.plot(rates, lfcc, 's--', color=C_LFCC, label='LFCC-GMM')

    for r, h, l in zip(rates, hos, lfcc):
        ax.annotate(f'{h:.1f}', (r, h), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color=C_HOS)
        ax.annotate(f'{l:.1f}', (r, l), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, color=C_LFCC)

    ax.set_xlabel("Opus Bitrate (kbps)")
    ax.set_ylabel("EER (%)")
    ax.set_title("EER vs Opus Bitrate")
    ax.set_xticks(rates)
    ax.set_ylim(0, 75)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    save_and_upload(fig, "fig_06_02_eer_vs_opus.png")

# =============================================================================
# SECTION 5 — Figure 3: EER vs AWGN SNR
# =============================================================================

def fig_eer_vs_awgn(results: dict):
    tags = ["awgn_00db", "awgn_05db", "awgn_10db", "awgn_20db"]
    snrs = [0, 5, 10, 20]
    hos  = [results[t]["hos_eer"]  for t in tags]
    lfcc = [results[t]["lfcc_eer"] for t in tags]
    clean_hos  = results["clean"]["hos_eer"]
    clean_lfcc = results["clean"]["lfcc_eer"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.axhline(clean_hos,  color=C_HOS,  lw=1.2, ls=':', alpha=0.6,
               label='HOS clean baseline')
    ax.axhline(clean_lfcc, color=C_LFCC, lw=1.2, ls=':', alpha=0.6,
               label='LFCC clean baseline')
    ax.plot(snrs, hos,  'o-',  color=C_HOS,  label='HOS-XGBoost')
    ax.plot(snrs, lfcc, 's--', color=C_LFCC, label='LFCC-GMM')

    for s, h, l in zip(snrs, hos, lfcc):
        ax.annotate(f'{h:.1f}', (s, h), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color=C_HOS)
        ax.annotate(f'{l:.1f}', (s, l), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, color=C_LFCC)

    ax.set_xlabel("SNR (dB)  [left = more noise]")
    ax.set_ylabel("EER (%)")
    ax.set_title("EER vs AWGN Noise Level")
    ax.set_xticks(snrs)
    ax.invert_xaxis()
    ax.set_ylim(0, 75)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    save_and_upload(fig, "fig_06_03_eer_vs_awgn.png")

# =============================================================================
# SECTION 6 — Figure 4: Degradation Bar Chart (all 14 conditions)
# =============================================================================

def fig_degradation_bar(results: dict):
    tags    = [t for t in CONDITION_ORDER if t in results]
    labels  = [SHORT_LABELS[t] for t in tags]
    hos_deg = [results[t]["hos_deg"]  for t in tags]
    lfc_deg = [results[t]["lfcc_deg"] for t in tags]
    colors  = [REGIME_COLORS[REGIME_MAP[t]] for t in tags]

    n   = len(tags)
    x   = np.arange(n)
    w   = 0.35

    fig, ax = plt.subplots(figsize=(13, 4.5))

    bars_hos  = ax.bar(x - w/2, hos_deg, w, color=colors, alpha=0.85,
                       edgecolor='white', linewidth=0.5, label='HOS-XGBoost')
    bars_lfcc = ax.bar(x + w/2, lfc_deg, w, color=colors, alpha=0.45,
                       edgecolor='white', linewidth=0.5,
                       hatch='//', label='LFCC-GMM')

    # Value labels on bars
    for bar in bars_hos:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.04,
                f'{h:.2f}', ha='center', va='bottom', fontsize=6.5,
                color='#1A237E')
    for bar in bars_lfcc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.04,
                f'{h:.2f}', ha='center', va='bottom', fontsize=6.5,
                color='#B71C1C')

    ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5,
               label='No degradation (1.0×)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Relative Degradation Ratio\n(EER$_{distorted}$ / EER$_{clean}$)")
    ax.set_title("Degradation Ratio across All Distortion Conditions")
    ax.set_ylim(0, 4.5)

    # Regime legend patches
    regime_patches = [
        mpatches.Patch(color=C_CODEC, label='Codec (MP3/Opus)'),
        mpatches.Patch(color=C_AWGN,  label='AWGN'),
        mpatches.Patch(color=C_TEL,   label='Telephone BPF'),
        mpatches.Patch(color=C_COMBO, label='Combination'),
    ]
    system_patches = [
        mpatches.Patch(facecolor='grey', alpha=0.85, label='HOS-XGBoost (solid)'),
        mpatches.Patch(facecolor='grey', alpha=0.45,
                       hatch='//', label='LFCC-GMM (hatched)'),
        mpatches.Patch(facecolor='none', edgecolor='black',
                       linestyle='--', label='No degradation (1.0×)'),
    ]
    leg1 = ax.legend(handles=regime_patches, loc='upper left',
                     fontsize=8, title='Distortion regime', title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=system_patches, loc='upper right',
              fontsize=8, title='System', title_fontsize=8)

    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    save_and_upload(fig, "fig_06_04_degradation_bar.png")

# =============================================================================
# SECTION 7 — Figure 5: EER Heatmap
# =============================================================================

def fig_eer_heatmap(results: dict):
    tags   = ["clean"] + [t for t in CONDITION_ORDER if t in results]
    xlabels = ["Clean"] + [SHORT_LABELS[t].replace('\n', ' ') for t in tags[1:]]
    hos_row  = [results[t]["hos_eer"]  for t in tags]
    lfcc_row = [results[t]["lfcc_eer"] for t in tags]
    matrix   = np.array([hos_row, lfcc_row])

    fig, ax = plt.subplots(figsize=(15, 2.8))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=55)

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=40, ha='right', fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['HOS-XGBoost', 'LFCC-GMM'], fontsize=10)

    for i in range(2):
        for j in range(len(tags)):
            val = matrix[i, j]
            color = 'white' if val > 38 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=7.5, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='EER (%)', shrink=0.8)
    ax.set_title("EER (%) — All Conditions × Systems", pad=10)
    fig.tight_layout()
    save_and_upload(fig, "fig_06_05_eer_heatmap.png")

# =============================================================================
# SECTION 8 — Figure 6: Regime Scatter (HOS deg vs LFCC deg)
# =============================================================================

def fig_regime_scatter(results: dict):
    fig, ax = plt.subplots(figsize=(6, 5.5))

    for tag in CONDITION_ORDER:
        if tag not in results:
            continue
        r       = results[tag]
        regime  = REGIME_MAP[tag]
        color   = REGIME_COLORS[regime]
        x_val   = r["hos_deg"]
        y_val   = r["lfcc_deg"]
        label   = SHORT_LABELS[tag].replace('\n', ' ')

        ax.scatter(x_val, y_val, color=color, s=80, zorder=3,
                   edgecolors='white', linewidth=0.5)
        # Offset annotations to avoid overlap
        offsets = {
            "MP3 32k": (6, 4), "MP3 64k": (6, -10), "MP3 128k": (6, 4),
            "Opus 6k": (6, 4), "Opus 12k": (-40, 6), "Opus 24k": (6, -10),
            "AWGN 0dB": (-52, 4), "AWGN 5dB": (6, -10),
            "AWGN 10dB": (6, 4), "AWGN 20dB": (6, -10),
            "Tel BPF": (6, 4),
            "MP3+ Tel": (6, -10), "Opus+ AWGN": (-58, 4),
            "MP3+ AWGN": (6, 4),
        }
        dx, dy = offsets.get(label, (6, 4))
        ax.annotate(label, (x_val, y_val),
                    textcoords='offset points', xytext=(dx, dy),
                    fontsize=7.5, color=color)

    # Diagonal: equal degradation
    lim = 4.0
    ax.plot([1, lim], [1, lim], 'k--', lw=0.8, alpha=0.4,
            label='Equal degradation')
    ax.fill_between([1, lim], [1, 1], [1, lim], alpha=0.04, color='blue',
                    label='HOS degrades more')
    ax.fill_between([1, lim], [1, lim], [lim, lim], alpha=0.04, color='red',
                    label='LFCC degrades more')

    ax.text(3.2, 1.15, 'HOS more\nsensitive', fontsize=8,
            color='#1565C0', alpha=0.7)
    ax.text(1.05, 3.4, 'LFCC more\nsensitive', fontsize=8,
            color='#B71C1C', alpha=0.7)

    # Regime legend
    patches = [mpatches.Patch(color=c, label=r)
               for r, c in REGIME_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, loc='lower right')

    ax.set_xlabel("HOS-XGBoost Degradation Ratio (×)")
    ax.set_ylabel("LFCC-GMM Degradation Ratio (×)")
    ax.set_title("Distortion Sensitivity: HOS vs LFCC")
    ax.set_xlim(0.8, lim)
    ax.set_ylim(0.5, lim)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    save_and_upload(fig, "fig_06_06_regime_scatter.png")

# =============================================================================
# SECTION 9 — Main
# =============================================================================

def main():
    print("=" * 60)
    print("06_exp5_figures.py — Generating Experiment 5 figures")
    print("=" * 60)

    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    results = download_results()
    print(f"Loaded results for {len(results)} conditions: "
          f"{list(results.keys())}")

    print("\n[1/6] EER vs MP3 bitrate ...")
    fig_eer_vs_mp3(results)

    print("[2/6] EER vs Opus bitrate ...")
    fig_eer_vs_opus(results)

    print("[3/6] EER vs AWGN SNR ...")
    fig_eer_vs_awgn(results)

    print("[4/6] Degradation bar chart ...")
    fig_degradation_bar(results)

    print("[5/6] EER heatmap ...")
    fig_eer_heatmap(results)

    print("[6/6] Regime scatter plot ...")
    fig_regime_scatter(results)

    # --- Sync to ensure all writes are flushed ---
    print("\n[SYNC] Flushing file system buffers...")
    os.sync()
    print("[SYNC] Complete.")

    print("\n" + "=" * 60)
    print("All 6 figures saved and uploaded to Google Drive.")
    print(f"Local dir: {LOCAL_DIR}")
    print("=" * 60)

# =============================================================================
if __name__ == "__main__":
    main()