# Cosmetic export from D:\华为杯\论文整理\paper_revision_work\revision_sources\04_Q2_NEW_FIGURE_DATA\q2_missing_pattern_public_baseline_bundle\package_q2_missing_pattern_results.py
# Read saved plot values only; no metrics are recalculated.
from __future__ import annotations
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'Arial', 'SimHei'], 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'figure.titlesize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.bbox': None, 'savefig.dpi': 300})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
MODELS = ('TFN', 'MulT', 'MISA')
SEEDS = (42, 43, 44)
MODALITIES = ('text', 'audio', 'vision')
LOCATIONS = ('early', 'middle', 'late')
RHOS = (0.1, 0.2, 0.3, 0.4, 0.5)
METRICS = ('accuracy', 'macro_f1', 'mae', 'pearson')
EXTRA_METRICS = ('selection_score',)
ALL_METRICS = METRICS + EXTRA_METRICS
PASTELS = {'text': '#BFDCE6', 'audio': '#EEE7B0', 'vision': '#E9C9CC'}
LINES = {'text': '#4F7F95', 'audio': '#B28D32', 'vision': '#B66F7A'}
DARK = '#333333'
GRID = '#D9D9D9'
ACCENT = '#F0B36D'
LABELS = {'text': '文本缺失', 'audio': '音频缺失', 'vision': '视觉缺失'}
METRIC_LABELS = {'accuracy': ('准确率（↑）', '准确率'), 'macro_f1': ('宏平均 F1（↑）', '宏平均 F1'), 'mae': ('平均绝对误差（MAE，↓）', '平均绝对误差'), 'pearson': ('皮尔逊相关系数（↑）', '皮尔逊相关系数')}

HERE=Path(__file__).resolve().parent.parent
DATA=HERE/'figure_data'
OUT=HERE/'figures/q2'

def make_figure(stats: pd.DataFrame, clean: dict[str, tuple[float, float]], out_base: Path) -> None:
    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 120 * mm), facecolor='white')
    gs = fig.add_gridspec(2, 2, left=0.105, right=0.97, bottom=0.14, top=0.82, wspace=0.34, hspace=0.58)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    x = np.asarray(RHOS, dtype=float)
    panel_order = ('accuracy', 'macro_f1', 'mae', 'pearson')
    panel_ids = ('a', 'b', 'c', 'd')
    for ax, metric, panel in zip(axes, panel_order, panel_ids):
        ax.set_facecolor('white')
        ax.grid(axis='y', color=GRID, linewidth=0.45, zorder=0)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(DARK)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=DARK, length=2.5, width=0.55, pad=2)
        ax.set_axisbelow(True)
        for modality in MODALITIES:
            part = stats[(stats.modality == modality) & (stats.metric == metric)].sort_values('rho')
            if len(part) != 5:
                raise RuntimeError(f'Expected five ratios for {modality}/{metric}')
            mean = part['mean'].to_numpy(float)
            sd = part['sd'].to_numpy(float)
            line_color = LINES[modality]
            band_color = PASTELS[modality]
            ax.fill_between(x, mean - sd, mean + sd, color=band_color, alpha=0.35, linewidth=0, zorder=1)
            ax.plot(x, mean, color=line_color, linewidth=2.4, marker='o', markersize=4.5, markerfacecolor=line_color, markeredgecolor='white', markeredgewidth=0.45, label=LABELS[modality], zorder=3)
        center, _sd = clean[metric]
        ax.axhline(center, color='#777777', linewidth=0.85, linestyle=(0, (3, 2)), zorder=2)
        ax.set_xticks(x, [f'{r:.1f}' for r in x])
        ax.set_xlabel('缺失比例 ρ', labelpad=3)
        ax.set_ylabel(METRIC_LABELS[metric][0], labelpad=3)
        ax.set_xlim(0.07, 0.53)
        all_vals = stats.loc[stats.metric == metric, ['mean', 'sd']]
        lows = np.r_[all_vals['mean'].to_numpy() - all_vals['sd'].to_numpy(), center]
        highs = np.r_[all_vals['mean'].to_numpy() + all_vals['sd'].to_numpy(), center]
        span = float(highs.max() - lows.min())
        pad = max(span * 0.12, 0.008)
        ax.set_ylim(float(lows.min() - pad), float(highs.max() + pad))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_title(f'({panel}) {METRIC_LABELS[metric][1]}', loc='left', fontsize=8.2, fontweight='bold', color=DARK, pad=7)
    handles = [Line2D([0], [0], color=LINES[m], marker='o', linewidth=2.4, markersize=4.5, markerfacecolor=LINES[m], markeredgecolor='white', label=LABELS[m]) for m in MODALITIES]
    handles.append(Line2D([0], [0], color='#777777', linewidth=0.85, linestyle=(0, (3, 2)), label='完整输入（三 seed 均值）'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 0.91), ncol=4, frameon=False, fontsize=7.1, handlelength=2.0, columnspacing=1.6)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix('.png'), dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(out_base.with_suffix('.pdf'), facecolor='white', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(out_base.with_suffix('.svg'), facecolor='white', bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)

stats=pd.read_csv(DATA/'fig11_q2_modality_missing_ratio_source.csv')
ref=json.loads((DATA/'p2_clean_plot_reference.json').read_text(encoding='utf-8'))
clean={m:(ref[m]['mean'],ref[m]['std']) for m in METRICS}
make_figure(stats,clean,OUT/'fig11_q2_modality_missing_degradation')
