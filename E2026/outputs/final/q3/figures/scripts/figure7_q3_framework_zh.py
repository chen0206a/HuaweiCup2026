"""Build the Chinese three-stage Q3 method overview as editable draw.io vectors."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


E_ROOT = Path(__file__).resolve().parents[5]
OUT = E_ROOT / "outputs" / "final" / "q3" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = OUT / "figure7_q3_framework_zh.drawio"

W, H = 1900, 930
INK = "#333333"
BORDER = "#747A80"
GRID = "#D7DADF"
BLUE = "#2F6FAE"
BLUE_SOFT = "#A9D3E9"
YELLOW_SOFT = "#EFD98B"
PINK_SOFT = "#EFB7B2"
BLUE_PALE = "#EDF7FB"
YELLOW_PALE = "#FCF7DF"
PINK_PALE = "#FBECEB"
ORANGE = "#E6A04B"
WHITE = "#FFFFFF"
PAPER = "#FAFAFA"
GRAY = "#737B82"
LIGHT_GRAY = "#F3F4F5"
GREEN = "#4F8062"
GREEN_SOFT = "#E7F2E9"

mx = Element("mxfile", {"host": "app.diagrams.net", "modified": "2026-09-24T00:00:00.000Z", "agent": "Codex", "version": "24.7.17", "type": "device"})
diagram = SubElement(mx, "diagram", {"id": "q3-framework-zh", "name": "图7 中文方法总览"})
model = SubElement(diagram, "mxGraphModel", {"dx": str(W), "dy": str(H), "grid": "0", "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1", "pageScale": "1", "pageWidth": str(W), "pageHeight": str(H), "math": "0", "shadow": "0"})
root = SubElement(model, "root")
SubElement(root, "mxCell", {"id": "0"})
SubElement(root, "mxCell", {"id": "1", "parent": "0"})
ART = []


def box(cid, value, x, y, w, h, *, fill=WHITE, stroke=BORDER, fs=24, bold=False, align="center", rounded=False, dashed=False, color=INK, sw=1.2, valign="middle"):
    ART.append(("box", value, x, y, w, h, fill, stroke, fs, bold, align, rounded, dashed, color, sw, valign))
    style = ["html=0", "whiteSpace=wrap", "overflow=hidden", "fontFamily=Microsoft YaHei", f"fontSize={fs}", f"fontColor={color}", f"fontStyle={1 if bold else 0}", f"align={align}", f"verticalAlign={valign}", "spacing=4", f"fillColor={fill}", f"strokeColor={stroke}", f"strokeWidth={sw}"]
    if rounded: style += ["rounded=1", "arcSize=10"]
    if dashed: style += ["dashed=1", "dashPattern=5 4"]
    cell = SubElement(root, "mxCell", {"id": cid, "value": value, "style": ";".join(style) + ";", "vertex": "1", "parent": "1"})
    SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
    return cid


def label(cid, value, x, y, w, h, *, fs=24, bold=False, align="center", color=INK):
    return box(cid, value, x, y, w, h, fill="none", stroke="none", fs=fs, bold=bold, align=align, color=color, sw=0)


def arrow(cid, x1, y1, x2, y2, *, color=BLUE, sw=2.1, head=True, dashed=False):
    ART.append(("arrow", x1, y1, x2, y2, color, sw, head, dashed))
    for suffix, x, y in (("src", x1, y1), ("dst", x2, y2)):
        anchor = SubElement(root, "mxCell", {"id": f"{cid}_{suffix}", "value": "", "style": "fillColor=none;strokeColor=none;opacity=0;", "vertex": "1", "parent": "1"})
        SubElement(anchor, "mxGeometry", {"x": str(x), "y": str(y), "width": "0.1", "height": "0.1", "as": "geometry"})
    style = ["html=1", "edgeStyle=none", "rounded=0", f"strokeColor={color}", f"strokeWidth={sw}", "endArrow=block" if head else "endArrow=none", "endFill=1"]
    if dashed: style += ["dashed=1", "dashPattern=5 4"]
    cell = SubElement(root, "mxCell", {"id": cid, "value": "", "style": ";".join(style) + ";", "edge": "1", "parent": "1", "source": f"{cid}_src", "target": f"{cid}_dst"})
    geo = SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    SubElement(geo, "mxPoint", {"x": str(x1), "y": str(y1), "as": "sourcePoint"})
    SubElement(geo, "mxPoint", {"x": str(x2), "y": str(y2), "as": "targetPoint"})


def line(cid, x1, y1, x2, y2, *, color=GRID, sw=1.1, dashed=False):
    arrow(cid, x1, y1, x2, y2, color=color, sw=sw, head=False, dashed=dashed)


# Global scaffold. No model identifier, experiment notation, or actual case values appear in the figure.
label("fig_title", "多模态情感预测与层级证据归因框架", 200, 14, 1500, 48, fs=35, bold=True)
stages = [
    ("s1", "输入与特征表示", 30, 82, 440),
    ("s2", "情感预测与 HEAF 解释分析", 510, 82, 930),
    ("s3", "解释结果输出", 1480, 82, 390),
]
for i, (cid, title, x, y, w) in enumerate(stages, 1):
    box(cid+"_head", "", x, y, w, 58, fill=PAPER, stroke=GRID, rounded=True, sw=1.1)
    box(cid+"_number", str(i), x+13, y+10, 39, 39, fill=BLUE, stroke=BLUE, fs=24, bold=True, color=WHITE, rounded=True, sw=1.0)
    label(cid+"_name", title, x+59, y+5, w-69, 49, fs=27, bold=True, align="left")
line("divide_1", 490, 158, 490, 866, color=GRID, sw=1.4)
line("divide_2", 1460, 158, 1460, 866, color=GRID, sw=1.4)
arrow("stage_flow_1", 470, 111, 505, 111, sw=2.2)
arrow("stage_flow_2", 1440, 111, 1475, 111, sw=2.2)


# Stage 1 follows the reference's parallel input-to-feature grammar.
label("input_heading", "三模态输入", 52, 164, 395, 38, fs=26, bold=True, align="left")
input_x = [50, 187, 324]
modality = [
    ("t", "文本", BLUE_SOFT, BLUE_PALE, "xₜ", "fₜ", "文本特征"),
    ("a", "音频", YELLOW_SOFT, YELLOW_PALE, "xₐ", "fₐ", "音频特征"),
    ("v", "视觉", PINK_SOFT, PINK_PALE, "xᵥ", "fᵥ", "视觉特征"),
]
for i, (key, title, fill, pale, xsym, fsym, feature_name) in enumerate(modality):
    x = input_x[i]
    box(f"input_{key}", "", x, 210, 125, 212, fill=WHITE, stroke=GRID, rounded=True, sw=1.1)
    box(f"input_{key}_accent", "", x+4, 214, 117, 8, fill=fill, stroke="none", rounded=True)
    label(f"input_{key}_title", title, x+6, 219, 113, 35, fs=26, bold=True)
    box(f"feature_{key}", "", x, 520, 125, 116, fill=pale, stroke=GRID, rounded=True, sw=1.1)
    box(f"feature_{key}_accent", "", x+4, 524, 117, 7, fill=fill, stroke="none", rounded=True)
    label(f"feature_{key}_symbol", fsym, x+5, 529, 115, 47, fs=31, bold=True)
    label(f"feature_{key}_name", feature_name, x+5, 575, 115, 42, fs=23)
    arrow(f"input_to_feature_{key}", x+62.5, 425, x+62.5, 514, color=BORDER, sw=2.3)
    arrow(f"feature_to_merge_{key}", x+62.5, 641, x+62.5, 702, color=BLUE, sw=1.9)
    label(f"input_{key}_symbol", xsym, x+7, 381, 111, 31, fs=23, bold=True)

# Text sample, audio waveform, and a neutral video-frame glyph are editable primitives.
label("text_sample", "“我很喜欢\n这部电影。”", 52, 270, 121, 85, fs=18)
for i, height in enumerate((13, 26, 39, 22, 45, 29, 51, 25, 40, 20, 31)):
    cx = 200 + i*10
    box(f"audio_wave_{i}", "", cx, 300-height/2, 4, height, fill=ORANGE, stroke="none", rounded=True)
label("audio_cues", "语调 · 能量\n节奏", 192, 328, 115, 46, fs=17)
box("video_screen", "", 342, 255, 89, 69, fill=WHITE, stroke=BORDER, rounded=True, sw=1.2)
box("video_head", "", 376, 267, 22, 21, fill="#E7EAED", stroke=BORDER, rounded=True, sw=1.0)
box("video_body", "", 364, 293, 47, 22, fill="#E7EAED", stroke=BORDER, rounded=True, sw=1.0)
label("video_cues", "表情 · 动作\n场景", 327, 326, 119, 52, fs=16)

box("multimodal_bar", "", 50, 708, 399, 72, fill=WHITE, stroke=BORDER, rounded=True, sw=1.2)
for i, (fill, pale) in enumerate(((BLUE_SOFT, BLUE_PALE), (YELLOW_SOFT, YELLOW_PALE), (PINK_SOFT, PINK_PALE))):
    box(f"modal_segment_{i}", "", 56+i*129, 715, 126, 57, fill=pale, stroke="none")
    box(f"modal_segment_accent_{i}", "", 56+i*129, 715, 126, 7, fill=fill, stroke="none")
label("multimodal_label", "多模态表示", 97, 719, 305, 47, fs=27, bold=True)
label("stage1_note", "输入经特征表示进入预测与解释", 52, 807, 395, 36, fs=21, color=GRAY)


# Stage 2 upper branch: model-level sentiment prediction.
box("prediction_panel", "", 530, 185, 890, 309, fill=PAPER, stroke=GRID, rounded=True, sw=1.2)
label("prediction_heading", "情感预测", 549, 197, 830, 41, fs=29, bold=True, align="left")
pred_inputs = [("p_t", "文本特征", BLUE_SOFT, BLUE_PALE, 255), ("p_a", "音频特征", YELLOW_SOFT, YELLOW_PALE, 322), ("p_v", "视觉特征", PINK_SOFT, PINK_PALE, 389)]
for cid, title, fill, pale, y in pred_inputs:
    box(cid, title, 550, y, 145, 49, fill=pale, stroke=GRID, fs=22, rounded=True, sw=1.0)
    box(cid+"_accent", "", 552, y+3, 7, 43, fill=fill, stroke="none", rounded=True)
    arrow(cid+"_edge", 697, y+24, 724, y+24, color=BLUE, sw=1.5, head=False)
line("prediction_bus", 725, 279, 725, 413, color=BLUE, sw=2.0)
arrow("bus_to_pool", 727, 346, 756, 346, sw=2.2)
box("pool", "池化表示提取\n均值池化\n注意力残差池化", 760, 296, 186, 100, fill=WHITE, stroke=BLUE, fs=19, bold=True, rounded=True, sw=1.4)
arrow("pool_to_fusion", 950, 346, 974, 346, sw=2.2)
box("fusion", "多模态融合", 978, 296, 163, 100, fill=WHITE, stroke=BLUE, fs=24, bold=True, rounded=True, sw=1.4)
arrow("fusion_to_split", 1145, 346, 1170, 346, sw=2.1)
line("output_split", 1171, 288, 1171, 408, color=BLUE, sw=2.0)
arrow("split_to_class", 1173, 288, 1196, 288, sw=2.0)
arrow("split_to_intensity", 1173, 408, 1196, 408, sw=2.0)
box("class_output", "", 1200, 244, 194, 88, fill=WHITE, stroke=GRID, rounded=True, sw=1.1)
label("class_output_title", "情感类别输出", 1207, 251, 181, 34, fs=23, bold=True)
label("class_output_classes", "积极 · 中性 · 消极", 1207, 288, 181, 30, fs=18)
box("intensity_output", "", 1200, 362, 194, 88, fill=WHITE, stroke=GRID, rounded=True, sw=1.1)
label("intensity_output_title", "情感强度输出", 1207, 369, 181, 33, fs=23, bold=True)
line("intensity_axis", 1223, 423, 1372, 423, color=BORDER, sw=1.5)
box("intensity_marker", "", 1294, 414, 17, 17, fill=ORANGE, stroke=BORDER, rounded=True, sw=1.0)

# The lower branch uses four restrained vector miniatures; none represent measured sample values.
arrow("prediction_to_explanation", 975, 496, 975, 542, sw=2.1)
box("heaf_panel", "", 530, 546, 890, 316, fill=PAPER, stroke=GRID, rounded=True, sw=1.2)
label("heaf_heading", "HEAF 解释分析", 550, 556, 845, 39, fs=29, bold=True, align="left")
method_cards = [
    ("coal", 550, "模态联盟构建"),
    ("shap", 770, "沙普利贡献分析"),
    ("inter", 990, "模态交互分析"),
    ("time", 1210, "时间遮挡分析"),
]
for cid, x, title in method_cards:
    box(cid, "", x, 608, 190, 221, fill=WHITE, stroke=GRID, rounded=True, sw=1.0)
    label(cid+"_title", title, x+7, 615, 176, 42, fs=22, bold=True)
for x in (744, 964, 1184):
    arrow(f"method_flow_{x}", x, 721, x+23, 721, sw=1.8)

# Coalition miniature: three modality nodes feeding a small set of grouped chips.
for i, (fill, sym) in enumerate(((BLUE_SOFT, "文"), (YELLOW_SOFT, "音"), (PINK_SOFT, "视"))):
    box(f"coal_node_{i}", sym, 568+i*49, 674, 40, 40, fill=fill, stroke=BORDER, fs=20, bold=True, rounded=True, sw=1.0)
for i in range(4):
    box(f"coal_chip_{i}", "", 567+i*42, 729, 29, 18, fill=LIGHT_GRAY, stroke=GRID, rounded=True, sw=0.8)
label("coal_desc", "八种模态组合", 558, 773, 174, 34, fs=20)

# Contribution miniature: equal-length bars intentionally avoid implying results.
for i, (sym, fill) in enumerate((("文", BLUE_SOFT), ("音", YELLOW_SOFT), ("视", PINK_SOFT))):
    yy = 681+i*31
    label(f"shap_label_{i}", sym, 782, yy-2, 29, 27, fs=19)
    box(f"shap_bar_{i}", "", 812, yy, 111, 18, fill=fill, stroke=BORDER, sw=0.7)
label("shap_desc", "计算模态贡献", 778, 773, 174, 34, fs=20)

# Pairwise miniature: exactly the three modality pairs.
for i, (pair, fill) in enumerate((("文—音", BLUE_SOFT), ("文—视", PINK_SOFT), ("音—视", YELLOW_SOFT))):
    box(f"interaction_pair_{i}", pair, 1015, 674+i*31, 139, 26, fill=fill, stroke=BORDER, fs=18, rounded=True, sw=0.7)
label("inter_desc", "分析两两交互", 998, 773, 174, 34, fs=20)

# Temporal miniature: a slot axis, schematic curve, and a subtle highlighted interval.
box("time_window", "", 1260, 679, 47, 73, fill="#FCEBD3", stroke="none")
line("time_axis_x", 1231, 751, 1383, 751, color=BORDER, sw=1.1)
line("time_axis_y", 1231, 673, 1231, 751, color=BORDER, sw=1.1)
curve = [(1236, 731), (1255, 724), (1275, 707), (1294, 713), (1312, 682), (1334, 706), (1354, 700), (1378, 715)]
for i, ((x1, y1), (x2, y2)) in enumerate(zip(curve[:-1], curve[1:])):
    line(f"curve_{i}", x1, y1, x2, y2, color=BLUE, sw=2.3)
label("time_desc", "定位关键特征区间", 1217, 773, 176, 34, fs=19)


# Stage 3: three output types, with grounding states kept explicit.
box("result_modality", "", 1498, 185, 354, 193, fill=PAPER, stroke=GRID, rounded=True, sw=1.2)
label("modality_heading", "模态级解释", 1515, 196, 320, 41, fs=27, bold=True, align="left")
for i, (lab, fill, pale) in enumerate((("主导模态", BLUE_SOFT, BLUE_PALE), ("各模态贡献值", YELLOW_SOFT, YELLOW_PALE), ("主要交互对", PINK_SOFT, PINK_PALE))):
    yy = 246+i*40
    box(f"modality_result_{i}", lab, 1518, yy, 313, 32, fill=pale, stroke=GRID, fs=20, rounded=True, sw=0.8)
    box(f"modality_result_accent_{i}", "", 1521, yy+3, 6, 26, fill=fill, stroke="none")

box("result_ground", "", 1498, 400, 354, 266, fill=PAPER, stroke=GRID, rounded=True, sw=1.2)
label("ground_heading", "关键证据定位", 1515, 410, 320, 40, fs=27, bold=True, align="left")
ground_rows = [
    ("文本片段", "已验证", BLUE_SOFT, BLUE_PALE, GREEN_SOFT, GREEN, False),
    ("音频特征槽位", "未验证", YELLOW_SOFT, YELLOW_PALE, LIGHT_GRAY, GRAY, True),
    ("视觉特征槽位", "未验证", PINK_SOFT, PINK_PALE, LIGHT_GRAY, GRAY, True),
]
for i, (lab, status, fill, pale, badge_fill, badge_stroke, dashed) in enumerate(ground_rows):
    yy = 462+i*61
    box(f"ground_row_{i}", lab, 1517, yy, 213, 48, fill=pale, stroke=GRID, fs=20, rounded=True, sw=0.8)
    box(f"ground_row_accent_{i}", "", 1520, yy+3, 6, 42, fill=fill, stroke="none")
    box(f"ground_status_{i}", status, 1737, yy+5, 95, 38, fill=badge_fill, stroke=badge_stroke, fs=18, bold=True, rounded=True, dashed=dashed, sw=1.0)
label("ground_note", "音频原始秒数、视觉原始帧号不输出", 1508, 637, 335, 27, fs=15, color=GRAY)

box("result_faith", "", 1498, 688, 354, 174, fill=PAPER, stroke=GRID, rounded=True, sw=1.2)
label("faith_heading", "解释有效性验证", 1515, 699, 320, 41, fs=27, bold=True, align="left")
label("faith_label_top", "关键区间删除", 1512, 751, 134, 29, fs=19, align="left")
label("faith_label_random", "随机区间删除", 1512, 795, 134, 29, fs=19, align="left")
box("faith_bar_top", "", 1649, 756, 151, 19, fill=BLUE_SOFT, stroke=BORDER, sw=0.7)
box("faith_bar_random", "", 1649, 800, 91, 19, fill="#F6D8AF", stroke=BORDER, sw=0.7)
label("faith_note", "预测边际变化示意", 1511, 835, 325, 24, fs=16, color=GRAY)

# Cross-stage arrows have no case-specific or training meaning.
arrow("features_to_methods", 473, 757, 527, 757, color=BLUE, sw=2.6)
arrow("prediction_to_output", 1443, 334, 1495, 334, color=BLUE, sw=2.5)
arrow("explanation_to_output", 1443, 741, 1495, 741, color=BLUE, sw=2.5)
label("figure_footer", "示意图仅展示方法流程；音频与视觉证据保留在特征空间，不映射为未经验证的原始时间或帧。", 155, 886, 1590, 28, fs=20, color=GRAY)

ElementTree(mx).write(TARGET, encoding="utf-8", xml_declaration=True)


def render_vector_outputs():
    plt.rcParams.update({
        "font.family": "Microsoft YaHei",
        "svg.fonttype": "path",
        "svg.hashsalt": "figure7-q3-framework-zh",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for item in ART:
        if item[0] == "box":
            _, value, x, y, w, h, fill, stroke, fs, bold, align, rounded, dashed, color, sw, valign = item
            if fill != "none" or stroke != "none":
                kwargs = {
                    "facecolor": "none" if fill == "none" else fill,
                    "edgecolor": "none" if stroke == "none" else stroke,
                    "linewidth": sw,
                    "linestyle": (0, (5, 4)) if dashed else "solid",
                }
                radius = min(10, w/2, h/2)
                shape = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", **kwargs) if rounded else Rectangle((x, y), w, h, **kwargs)
                ax.add_patch(shape)
            if value:
                tx = x + (w/2 if align == "center" else 4 if align == "left" else w-4)
                ty = y + (h/2 if valign == "middle" else 3 if valign == "top" else h-3)
                math_labels = {"xₜ": r"$x_t$", "xₐ": r"$x_a$", "xᵥ": r"$x_v$", "fₜ": r"$f_t$", "fₐ": r"$f_a$", "fᵥ": r"$f_v$"}
                ax.text(tx, ty, math_labels.get(value, value), ha=align, va="center" if valign == "middle" else valign,
                        fontsize=fs*0.75, fontweight="bold" if bold else "normal", color=color,
                        linespacing=1.12, clip_on=False)
        else:
            _, x1, y1, x2, y2, color, sw, head, dashed = item
            if head:
                shape = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                        linewidth=sw, color=color, linestyle=(0, (5, 4)) if dashed else "solid",
                                        shrinkA=0, shrinkB=0)
                ax.add_patch(shape)
            else:
                ax.plot((x1, x2), (y1, y2), color=color, linewidth=sw,
                        linestyle=(0, (5, 4)) if dashed else "solid", solid_capstyle="butt")
    base = OUT / "figure7_q3_framework_zh"
    fig.savefig(base.with_suffix(".svg"), format="svg", facecolor=WHITE, metadata={"Date": "2026-09-24"})
    fig.savefig(base.with_suffix(".pdf"), format="pdf", facecolor=WHITE, metadata={"CreationDate": datetime(2026, 9, 24, tzinfo=timezone.utc)})
    fig.savefig(base.with_suffix(".png"), format="png", dpi=150, facecolor=WHITE)
    svg_path = base.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


render_vector_outputs()
print(TARGET)
