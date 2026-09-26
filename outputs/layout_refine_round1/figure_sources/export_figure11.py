# Cosmetic export from D:\华为杯\论文整理\paper_revision_work\paper\scripts\build_revision_figures.py
# Read saved plot values only; no metrics are recalculated.
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'Arial'], 'axes.unicode_minus': False, 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out', 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300, 'savefig.bbox': None})
METRICS = ['accuracy', 'macro_f1', 'mae', 'pearson']
MODS = ['text', 'audio', 'vision', 'text+audio', 'text+vision', 'audio+vision']
LOC = ['early', 'middle', 'late']
MOD_NAMES = ['文本', '语音', '视觉', '文本＋语音', '文本＋视觉', '语音＋视觉']
LOSS_NAMES = ['准确率变化', '宏平均 F1 变化', 'MAE 变化', '皮尔逊相关变化']
SEEDS = {42, 43, 44}

HERE=Path(__file__).resolve().parent.parent
DATA=HERE/'figure_data'
OUT=HERE/'figures/q2'
def save(fig,name):
 for ext in ['pdf','png','svg']:
  fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.04)
 plt.close(fig)

def figure12():
    summary = pd.read_csv(DATA / 'fig12_q2_performance_change.csv')
    fig, axs = plt.subplots(2, 2, figsize=(183 / 25.4, 116 / 25.4))
    fig.subplots_adjust(left=0.125, right=0.918, bottom=0.135, top=0.945, wspace=0.49, hspace=0.32)
    color = LinearSegmentedColormap.from_list('fig12_RdBu_white_center', ['#B2182B', '#EF8A62', '#FFFFFF', '#67A9CF', '#2166AC'], N=257)
    row_labels = ['文本', '语音', '视觉', '文本+语音', '文本+视觉', '语音+视觉']
    limits = {}
    for i, (ax, m) in enumerate(zip(axs.flat, METRICS)):
        part = summary[summary.metric == m]
        v = part.pivot(index='modality', columns='location', values='mean_improvement').reindex(index=MODS, columns=LOC).to_numpy()
        bound = max(float(np.max(abs(v))), np.finfo(float).eps)
        limits[m] = bound
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        im = ax.imshow(v, cmap=color, norm=norm, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(3), ['前段', '中段', '后段'], fontsize=8)
        ax.set_yticks(range(6), row_labels, fontsize=8)
        ax.tick_params(length=0, pad=4)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
        ax.grid(which='minor', color='#89939A', linewidth=0.45)
        ax.tick_params(which='minor', length=0)
        for y in range(6):
            for x in range(3):
                val = v[y, x]
                rounded = round(float(val), 3)
                label = f'{rounded:+.3f}' if rounded else '0.000'
                rgb = np.asarray(color(norm(val))[:3])
                linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
                ax.text(x, y, label, ha='center', va='center', fontsize=8, color='white' if luminance < 0.38 else '#30343B')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#66727B')
            spine.set_linewidth(0.65)
        ax.set_title(f'({chr(97 + i)}) {LOSS_NAMES[i]}', loc='left', fontweight='bold', fontsize=9, pad=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
        cb.set_ticks([-bound, 0, bound])
        cb.set_ticklabels([f'{-bound:+.3f}', '0', f'{bound:+.3f}'])
        cb.ax.tick_params(labelsize=7, length=2, width=0.6, pad=2)
        cb.outline.set_visible(True)
        cb.outline.set_edgecolor('#66727B')
        cb.outline.set_linewidth(0.6)
    save(fig, 'fig12_q2_modality_location_absolute_degradation')

figure12()
