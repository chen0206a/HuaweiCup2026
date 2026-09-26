# Cosmetic export from D:\华为杯\论文整理\paper_revision_work\paper\scripts\plot_fig13_q2_modality_location_gain.py
# Read saved plot values only; no metrics are recalculated.
from __future__ import annotations
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'], 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 8, 'figure.titlesize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})
CATEGORICAL = ['#2166AC', '#B2182B', '#1B7837', '#F1A340', '#762A83', '#666666']
CATEGORICAL_EXTENDED = ['#2166AC', '#B2182B', '#1B7837', '#F1A340', '#762A83', '#666666', '#4393C3', '#D6604D', '#5AAE61', '#B35806', '#9970AB', '#999999']
DIVERGING = ['#2166AC', '#F7F7F7', '#B2182B']
SEQUENTIAL = ['#F7FBFF', '#6BAED6', '#08306B']
ACCENT_RED = '#B2182B'
GREY = '#999999'
BLACK = '#222222'
mpl.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.bbox': 'tight', 'savefig.dpi': 300})
mpl.use('Agg')
import hashlib
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from q2_figure_style import performance_cmap
MODELS = ('B0-WCE', 'B5-P2')
SEEDS = (42, 43, 44)
LOCATIONS = ('early', 'middle', 'late')
SINGLE = ('text', 'audio', 'vision')
DOUBLE = ('text+audio', 'text+vision', 'audio+vision')
METRICS = ('macro_f1', 'pearson', 'mae', 'accuracy')
TITLES = ('宏平均 F1', '皮尔逊相关', 'MAE 改善', '准确率')
CN_ROWS = {'text': '文本', 'audio': '语音', 'vision': '视觉', 'text+audio': '文本+语音', 'text+vision': '文本+视觉', 'audio+vision': '语音+视觉'}
CN_LOCS = ('前段', '中段', '后段')

HERE=Path(__file__).resolve().parent.parent
DATA=HERE/'figure_data'
OUT=HERE/'figures/q2'
BASE=OUT/'fig13_q2_p2_minus_b0_modality_location_gain'

def make_figure(matrices: dict[str, dict[str, np.ndarray]], values: pd.DataFrame) -> float:
    mpl.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'Arial', 'SimHei'], 'axes.unicode_minus': False, 'savefig.bbox': None})
    all_values = values[[f'delta_{m}' for m in METRICS]].to_numpy(dtype=float)
    limit = math.ceil(float(np.max(np.abs(all_values))) / 0.002) * 0.002
    cmap = LinearSegmentedColormap.from_list('fig13_RdBu_white_center', ['#B2182B', '#EF8A62', '#FFFFFF', '#67A9CF', '#2166AC'], N=257)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig = plt.figure(figsize=(183 / 25.4, 95 / 25.4), facecolor='white')
    grid = fig.add_gridspec(2, 4, left=0.155, right=0.902, bottom=0.185, top=0.865, wspace=0.1, hspace=0.23)
    for row_index, (group_name, modalities) in enumerate((('single', SINGLE), ('double', DOUBLE))):
        for col_index, (metric, title) in enumerate(zip(METRICS, TITLES)):
            ax = fig.add_subplot(grid[row_index, col_index])
            matrix = matrices[group_name][metric]
            im = ax.pcolormesh(np.arange(4), np.arange(4), matrix, cmap=cmap, norm=norm, shading='flat', edgecolors='#89939A', linewidth=0.45)
            ax.set_xlim(0, 3)
            ax.set_ylim(3, 0)
            ax.set_aspect('auto')
            ax.set_xticks(np.arange(3) + 0.5, CN_LOCS)
            ax.tick_params(axis='x', length=0, pad=3, labelsize=7.7, colors='#30343B')
            if col_index == 0:
                ax.set_yticks(np.arange(3) + 0.5, [CN_ROWS[m] for m in modalities])
                ax.tick_params(axis='y', length=0, pad=4, labelsize=8, colors='#30343B')
            else:
                ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#66727B')
                spine.set_linewidth(0.65)
            if row_index == 0:
                position = ax.get_position()
                fig.text((position.x0 + position.x1) / 2, 0.917, title, ha='center', va='center', fontsize=9, fontweight='bold', color='#30343B')
            for i in range(3):
                for j in range(3):
                    shown = f'{matrix[i, j]:+.3f}'
                    rgb = np.asarray(cmap(norm(matrix[i, j]))[:3])
                    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
                    ax.text(j + 0.5, i + 0.5, shown, ha='center', va='center', fontsize=7.6, color='white' if luminance < 0.38 else '#30343B')
    second_top = grid[1, 0].get_position(fig).y1
    fig.text(0.014, 0.917, '(a) 单模态缺失', ha='left', va='center', fontsize=8.5, fontweight='bold', color='#30343B')
    fig.text(0.014, second_top + 0.023, '(b) 双模态缺失', ha='left', va='bottom', fontsize=8.5, fontweight='bold', color='#30343B')
    color_axis = fig.add_axes((0.924, 0.235, 0.013, 0.565))
    cb = fig.colorbar(im, cax=color_axis)
    ticks = [-limit, -limit / 2, 0, limit / 2, limit]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{v:+.3f}' if v else '0' for v in ticks])
    cb.ax.tick_params(labelsize=7, length=2, width=0.6, pad=2)
    cb.outline.set_visible(True)
    cb.outline.set_edgecolor('#66727B')
    cb.outline.set_linewidth(0.6)
    fig.text(0.931, 0.838, '性能\n增益', ha='center', va='bottom', fontsize=7.5, color='#30343B')
    for suffix in ['pdf', 'png', 'svg']:
        fig.savefig(BASE.with_suffix('.' + suffix), dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return limit

values=pd.read_csv(DATA/'fig13_q2_modality_location_gain_values.csv')
matrices={}
for group,order in [('single',SINGLE),('double',DOUBLE)]:
 part=values[values['group']==group]
 matrices[group]={m:part.pivot(index='modalities',columns='location',values=f'delta_{m}').reindex(index=order,columns=LOCATIONS).to_numpy() for m in METRICS}
make_figure(matrices,values)
