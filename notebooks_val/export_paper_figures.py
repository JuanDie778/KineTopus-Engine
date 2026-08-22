#!/usr/bin/env python3
"""
=============================================================================
KineTopus Engine — Publication-Quality Pipeline Autopsy Figures
=============================================================================
Generates a 4-panel composite figure for the paper's Section 3 
(Mathematical Framework), demonstrating each pipeline layer on real 
BTC-USD daily data.

Panel (a): Power Spectral Density (FFT Layer 1)
Panel (b): Discrete→Smooth Transmutation + Analytical Derivatives (Spline Layer 2)
Panel (c): Topological Residual + Gaussian White Noise Diagnostic (Spline QA)
Panel (d): Dual CUSUM Sismograph + Regime Shift Detection (Layer 3)

Output: paper/figures/fig_pipeline_autopsy.png (300 DPI, ~8×10 inches)
=============================================================================
Author: Juan Diego Ussa Aponte
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ui.market_loader import MarketLoader
from src.quant_engine.sensor import SpectralAnalyzer
from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.nervous import RegimeShiftDetector

# ═══════════════════════════════════════════════════════════════════════════
# 0. GLOBAL STYLE — Publication-grade, journal-neutral palette
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'lines.linewidth': 0.9,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
    'axes.grid': True,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'dejavuserif',
})

# ── Color Palette (paper-friendly: high contrast, greyscale-safe) ────────
C_DISCRETE = '#adb5bd'     # Grey for raw discrete data
C_SMOOTH   = '#1e3a8a'     # Deep blue for smooth manifold
C_VEL      = '#b91c1c'     # Dark red for velocity
C_ACCEL    = '#166534'     # Forest green for acceleration
C_SPOS     = '#2563eb'     # Blue for S+
C_SNEG     = '#dc2626'     # Red for S-
C_THRESH   = '#111827'     # Near-black for threshold line
C_SHIFT    = '#f59e0b'     # Amber for regime shift markers
C_PSD_LINE = '#1e3a8a'     # Deep blue for PSD curve
C_PSD_FILL = '#bfdbfe'     # Light blue for PSD fill
C_RESID    = '#6b7280'     # Medium grey for residuals
C_GAUSS    = '#dc2626'     # Red for Gaussian overlay
C_REGIME_BAND = '#e5e7eb'  # Light grey for regime bands

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA ACQUISITION — BTC-USD Daily (5-year window)
# ═══════════════════════════════════════════════════════════════════════════
print("[1/5] Downloading BTC-USD daily data (5y)...")
import yfinance as yf
df = yf.download('BTC-USD', period='5y', interval='1d', progress=False)
if hasattr(df.columns, 'droplevel'):
    df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
print(f"      -> {len(df)} daily candles loaded ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

# ── Pipeline Layer 0: Normalization ──────────────────────────────────────
log_returns, volumen_z, precio_raw, dt_val = MarketLoader.prepare_quant_input(df)
t = np.arange(len(log_returns), dtype=np.float64) * dt_val
N = len(t)

# ═══════════════════════════════════════════════════════════════════════════
# 2. PIPELINE EXECUTION — Layers 1-3
# ═══════════════════════════════════════════════════════════════════════════
print("[2/5] Executing pipeline layers 1-3...")

# -- Layer 1: FFT Spectral Analysis (bypass multiprocessing on Windows) ----
sensor = SpectralAnalyzer(top_k=5)  # top_k=5 for richer PSD visualization
# Compute per-feature to avoid ProcessPoolExecutor spawn issue on Windows
fft_results = {}
fft_results[0] = sensor.analyze(log_returns.reshape(-1, 1), dt=dt_val)[0]
fft_results[1] = sensor.analyze(volumen_z.reshape(-1, 1), dt=dt_val)[0]

# Full PSD for plotting (recompute for full spectrum)
series_centered = log_returns - np.mean(log_returns)
fft_full = np.fft.rfft(series_centered)
frequencies = np.fft.rfftfreq(N, d=dt_val)
psd = np.abs(fft_full) ** 2
psd[0] = 0.0  # Remove DC

# Get dominant periods for annotation
periodos_retornos = fft_results[0]['periods']
amplitudes_retornos = fft_results[0]['amplitudes']
periodos_volumen = fft_results[1]['periods']

print(f"      -> Dominant periods (returns): {periodos_retornos[:3].round(1)} candles")

# ── Layer 2: Spline Smoothing ────────────────────────────────────────────
blender = ContinuousBlender(tolerance=0.0050)
blender.fit(t, log_returns, periodos_retornos, feature_idx=0)
blender.fit(t, volumen_z, periodos_volumen, feature_idx=1)

r_smooth, r_dot, r_ddot = blender.compute_continuous(0, t)

# Topological residual
residual = log_returns - r_smooth
res_mean = np.mean(residual)
res_std = np.std(residual)
print(f"      -> Spline residual: mean={res_mean:.6f}, std={res_std:.6f}")

# ── Layer 3: CUSUM Regime Detection ──────────────────────────────────────
detector = RegimeShiftDetector(threshold=5.0, drift=1.0)
cusum_report = detector.detect(log_returns, r_smooth)
shift_idx = cusum_report['shift_indices']
s_pos = cusum_report['s_pos']
s_neg = cusum_report['s_neg']
print(f"      -> CUSUM detected {len(shift_idx)} structural regime shifts")

# ═══════════════════════════════════════════════════════════════════════════
# 3. FIGURE COMPOSITION — 4-Panel Autopsy
# ═══════════════════════════════════════════════════════════════════════════
print("[3/5] Composing publication figure...")

fig = plt.figure(figsize=(7.5, 9.5))  # Journal single-column width

# Use GridSpec for precise layout control
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1.3, 0.8, 1.1],
                       hspace=0.38, left=0.11, right=0.96, top=0.97, bottom=0.04)

# ─── PANEL (a): Power Spectral Density ───────────────────────────────────
ax_a = fig.add_subplot(gs[0])

# Convert frequency to period for x-axis (more intuitive)
valid = frequencies > 0
periods_axis = 1.0 / frequencies[valid]
psd_valid = psd[valid]

# Plot PSD in log-log space
ax_a.semilogy(periods_axis, psd_valid, color=C_PSD_LINE, linewidth=0.7, alpha=0.85)
ax_a.fill_between(periods_axis, psd_valid, alpha=0.15, color=C_PSD_FILL)

# Annotate dominant peaks with arrows
for i, (T, A) in enumerate(zip(periodos_retornos[:3], amplitudes_retornos[:3])):
    if T > 0 and T < max(periods_axis):
        peak_psd = A**2
        label = f'$T_{i+1} \\approx {T:.0f}$ d'
        ax_a.annotate(label,
                      xy=(T, peak_psd),
                      xytext=(T * 1.5, peak_psd * 3),
                      fontsize=7.5, fontweight='bold', color=C_PSD_LINE,
                      arrowprops=dict(arrowstyle='->', color=C_PSD_LINE, lw=0.8),
                      ha='left')
        ax_a.plot(T, peak_psd, 'v', color=C_SHIFT, markersize=6, zorder=5)

ax_a.set_xlabel('Period (daily candles)', fontsize=8)
ax_a.set_ylabel('Power Spectral Density', fontsize=8)
ax_a.set_title('(a)  Layer 1 — Spectral Decomposition (FFT)', fontweight='bold', loc='left')
ax_a.set_xlim(1, N // 2)
ax_a.set_xscale('log')

# --- PANEL (b): Discrete->Smooth->Velocity->Acceleration -----------------
# Use a focused window for clarity (last 500 candles)
W = 500
t_w = t[-W:]

# Create 3 vertically stacked subplots within the gs[1] slot
gs_b = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.15)

# Sub-panel b1: Position (raw discrete vs smooth spline)
ax_b1 = fig.add_subplot(gs_b[0])
ax_b1.scatter(t_w, log_returns[-W:], s=0.8, c=C_DISCRETE, alpha=0.4, 
              label='Discrete $r_t$ (market noise)', zorder=1, rasterized=True)
ax_b1.plot(t_w, r_smooth[-W:], color=C_SMOOTH, linewidth=1.1, 
           label='$C^2$ Spline $r_{\\mathrm{smooth}}(t)$', zorder=3)
ax_b1.set_ylabel('$r(t)$', fontsize=8)
ax_b1.set_title('(b)  Layer 2 -- Topological Transmutation: Discrete Noise to $C^2$ Smooth Manifold', 
               fontweight='bold', loc='left', fontsize=9.5)
ax_b1.legend(loc='upper right', framealpha=0.9, fontsize=7, ncol=2)
ax_b1.tick_params(labelbottom=False)

# Sub-panel b2: Velocity (1st analytical derivative)
ax_b2 = fig.add_subplot(gs_b[1], sharex=ax_b1)
ax_b2.plot(t_w, r_dot[-W:], color=C_VEL, linewidth=0.8, alpha=0.85)
ax_b2.axhline(y=0, color='#9ca3af', linewidth=0.4, linestyle='--', alpha=0.5)
ax_b2.set_ylabel('$\\dot{r}(t)$', fontsize=8, color=C_VEL)
ax_b2.tick_params(axis='y', labelcolor=C_VEL)
ax_b2.tick_params(labelbottom=False)
# Label annotation
ax_b2.text(0.01, 0.92, 'Velocity (Capital Inertia)', transform=ax_b2.transAxes,
           fontsize=7, color=C_VEL, style='italic', va='top')

# Sub-panel b3: Acceleration (2nd analytical derivative)
ax_b3 = fig.add_subplot(gs_b[2], sharex=ax_b1)
ax_b3.plot(t_w, r_ddot[-W:], color=C_ACCEL, linewidth=0.7, alpha=0.8)
ax_b3.axhline(y=0, color='#9ca3af', linewidth=0.4, linestyle='--', alpha=0.5)
ax_b3.set_ylabel('$\\ddot{r}(t)$', fontsize=8, color=C_ACCEL)
ax_b3.set_xlabel(f'Time index (candles, last {W} shown)', fontsize=8)
ax_b3.tick_params(axis='y', labelcolor=C_ACCEL)
# Label annotation
ax_b3.text(0.01, 0.92, 'Acceleration (Force on Capital)', transform=ax_b3.transAxes,
           fontsize=7, color=C_ACCEL, style='italic', va='top')

# ─── PANEL (c): Topological Residual + Gaussian Diagnostic ──────────────
ax_c = fig.add_subplot(gs[2])

# Histogram of residual
n_bins = 80
counts, bins, patches = ax_c.hist(residual, bins=n_bins, density=True, 
                                   color=C_RESID, alpha=0.55, edgecolor='white', 
                                   linewidth=0.3, label='Empirical $\\epsilon_t$')

# Theoretical Gaussian overlay
x_gauss = np.linspace(residual.min(), residual.max(), 300)
y_gauss = norm.pdf(x_gauss, loc=res_mean, scale=res_std)
ax_c.plot(x_gauss, y_gauss, color=C_GAUSS, linewidth=1.3, 
          label=f'$\\mathcal{{N}}(\\mu={res_mean:.4f},\\, \\sigma={res_std:.4f})$')

# Annotate statistics
stats_text = (f'$\\mu = {res_mean:.5f}$\n'
              f'$\\sigma = {res_std:.5f}$\n'
              f'$N = {N}$')
ax_c.text(0.97, 0.95, stats_text, transform=ax_c.transAxes, fontsize=7,
          verticalalignment='top', horizontalalignment='right',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=C_RESID, alpha=0.85))

ax_c.set_xlabel('Residual $\\epsilon_t = r_t - r_{\\mathrm{smooth}}(t)$', fontsize=8)
ax_c.set_ylabel('Probability density', fontsize=8)
ax_c.set_title('(c)  Spline Diagnostic — Topological Residual White Noise Verification', 
               fontweight='bold', loc='left')
ax_c.legend(loc='upper left', framealpha=0.9, fontsize=7.5)

# ─── PANEL (d): CUSUM Sismograph ────────────────────────────────────────
ax_d = fig.add_subplot(gs[3])

# Plot S+ and S-
ax_d.plot(t, s_pos, color=C_SPOS, linewidth=0.7, alpha=0.8, label='$S^+(t)$ (positive tension)')
ax_d.plot(t, s_neg, color=C_SNEG, linewidth=0.7, alpha=0.8, label='$S^-(t)$ (negative tension)')

# Threshold line
ax_d.axhline(y=5.0, color=C_THRESH, linewidth=1.0, linestyle='--', alpha=0.7,
             label=f'Threshold $H = 5.0$')

# Regime shift markers
for idx in shift_idx:
    ax_d.axvline(x=t[idx], color=C_SHIFT, linewidth=0.5, alpha=0.4, linestyle='-')

# Mark shift points on the threshold line
if len(shift_idx) > 0:
    ax_d.scatter([t[i] for i in shift_idx], [5.0]*len(shift_idx), 
                 marker='x', s=25, c=C_SHIFT, linewidths=1.2, zorder=5,
                 label=f'Regime shifts ($n={len(shift_idx)}$)')

# Subtle regime bands
boundaries = [0] + list(shift_idx) + [N]
for i in range(len(boundaries) - 1):
    if i % 2 == 1:
        ax_d.axvspan(t[boundaries[i]], t[min(boundaries[i+1], N-1)], 
                     alpha=0.06, color=C_REGIME_BAND)

ax_d.set_xlabel('Time index (daily candles)', fontsize=8)
ax_d.set_ylabel('CUSUM accumulator value', fontsize=8)
ax_d.set_title(f'(d)  Layer 3 — Dual CUSUM Sismograph ($k=1.0$, $H=5.0$, {len(shift_idx)} regime shifts detected)',
               fontweight='bold', loc='left')
ax_d.legend(loc='upper right', framealpha=0.9, ncol=2, fontsize=7)
ax_d.set_xlim(t[0], t[-1])
ax_d.set_ylim(bottom=0)

# ═══════════════════════════════════════════════════════════════════════════
# 4. EXPORT
# ═══════════════════════════════════════════════════════════════════════════
output_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(output_dir, exist_ok=True)

composite_path = os.path.join(output_dir, 'fig_pipeline_autopsy.png')
fig.savefig(composite_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[4/5] Saved composite figure -> {os.path.abspath(composite_path)}")

print(f"[5/5] Export complete. {len(shift_idx)} regime shifts, {N} candles processed.")
print(f"      Dominant periods: {periodos_retornos[:3].round(1)} candles")
print(f"      Residual stats: mean={res_mean:.6f}, std={res_std:.6f}")
print(f"      -> Upload '{composite_path}' to Overleaf figures/ directory")

plt.close('all')
