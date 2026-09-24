from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "outputs" / "final" / "q3" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DRAWIO = OUT / "figure7_heaf_overview_zh.drawio"

W, H = 3000, 1510
C = {
    "ink": "#30343B", "muted": "#68717A", "line": "#D5D9DE", "panel": "#FAFBFC",
    "text": "#4477AA", "text_fill": "#E8F0F7", "audio": "#C67A31", "audio_fill": "#F8EBDD",
    "vision": "#4E8063", "vision_fill": "#E8F0E9", "purple": "#81769A", "purple_fill": "#F0EDF5",
    "green": "#4E8063", "green_fill": "#E7F1E8", "gray_fill": "#F1F2F3", "orange": "#D58B45",
}

mx = Element("mxfile", {"host": "app.diagrams.net", "modified": "2026-09-24T00:00:00.000Z", "agent": "Codex", "version": "24.7.17", "type": "device"})
diagram = SubElement(mx, "diagram", {"id": "heaf-overview", "name": "HEAF 中文总览"})
model = SubElement(diagram, "mxGraphModel", {"dx": str(W), "dy": str(H), "grid": "1", "gridSize": "10", "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1", "pageScale": "1", "pageWidth": str(W), "pageHeight": str(H), "math": "0", "shadow": "0"})
root = SubElement(model, "root")
SubElement(root, "mxCell", {"id": "0"})
SubElement(root, "mxCell", {"id": "1", "parent": "0"})
cells = {}


def vertex(cid, label, x, y, w, h, fill="none", stroke="none", *, fs=22, color=None, bold=False, rounded=False, dashed=False, align="center", valign="middle", stroke_w=1.4, spacing=5, font="Microsoft YaHei"):
    style = ["html=1", "whiteSpace=wrap", "overflow=hidden", f"fontFamily={font}", f"fontSize={fs}", f"fontColor={color or C['ink']}", f"align={align}", f"verticalAlign={valign}", f"spacing={spacing}", f"strokeWidth={stroke_w}"]
    if fill != "none": style += [f"fillColor={fill}", "fillOpacity=100"]
    else: style += ["fillColor=none"]
    if stroke != "none": style += [f"strokeColor={stroke}"]
    else: style += ["strokeColor=none"]
    if rounded: style += ["rounded=1", "arcSize=12"]
    if dashed: style += ["dashed=1", "dashPattern=6 4"]
    if bold: style += ["fontStyle=1"]
    c = SubElement(root, "mxCell", {"id": cid, "value": escape(label).replace("\n", "<br>"), "style": ";".join(style) + ";", "vertex": "1", "parent": "1"})
    SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
    cells[cid] = c
    return cid


def edge(cid, source, target, color="#7B838A", *, dashed=False, width=1.8, arrow=True, label=None, label_color=None):
    style = ["edgeStyle=orthogonalEdgeStyle", "rounded=0", "orthogonalLoop=1", "jettySize=auto", "html=1", f"strokeColor={color}", f"strokeWidth={width}", "endArrow=block" if arrow else "endArrow=none", "endFill=1"]
    if dashed: style += ["dashed=1", "dashPattern=6 4"]
    attrs = {"id": cid, "value": escape(label or ""), "style": ";".join(style) + ";", "edge": "1", "parent": "1", "source": source, "target": target}
    c = SubElement(root, "mxCell", attrs)
    SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})
    return c


# Canvas title and four clearly separated stages.
vertex("title", "图7  HEAF 可解释预测框架总览", 70, 24, 2860, 58, fs=36, bold=True)
stage_x = [(55, 585), (680, 970), (1685, 600), (2310, 635)]
stage_names = ["阶段一｜输入与冻结预测器", "阶段二｜HEAF 层级解释", "阶段三｜原始证据映射", "阶段四｜解释卡输出"]
for i, ((x, w), name) in enumerate(zip(stage_x, stage_names)):
    vertex(f"stage{i+1}", name, x, 100, w, 52, C["panel"], C["line"], fs=25, bold=True, rounded=True, stroke_w=1.2)
for i, x in enumerate((655, 1665, 2290)):
    vertex(f"separator{i}", "", x, 170, 2, 1195, C["line"], C["line"], stroke_w=1)
for i, (x, w, fs) in enumerate(((642, 35, 24), (1650, 32, 22), (2282, 24, 17))):
    vertex(f"stage_arrow{i}", "→", x, 106, w, 38, fs=fs, bold=True, color=C["muted"])

# Stage 1: three feature streams, frozen predictor and dual outputs.
modalities = [
    ("text", "文本特征", "50 × 768", C["text"], C["text_fill"], 190),
    ("audio", "音频特征", "50 × 74", C["audio"], C["audio_fill"], 280),
    ("vision", "视觉特征", "50 × 35", C["vision"], C["vision_fill"], 370),
]
for key, name, dims, col, fill, y in modalities:
    vertex(f"{key}_icon", "", 72, y+8, 50, 44, fill, col, rounded=True, stroke_w=1.8)
    # editable mini feature rows indicate aligned temporal vectors
    for j in range(4):
        vertex(f"{key}_tile{j}", "", 80+j*10, y+19, 7, 20, col, "none", rounded=True)
    vertex(f"{key}_label", f"{name}\n{dims}", 132, y, 190, 62, fill, col, fs=20, bold=True, rounded=True)
    vertex(f"{key}_out", "", 346, y+7, 176, 48, fill, col, rounded=True)
    for j in range(8):
        vertex(f"{key}_outtile{j}", "", 356+j*19, y+17, 14, 26, col if j in (0, 7) else "#FFFFFF", col, rounded=True, stroke_w=0.8)
    edge(f"{key}_to_pred", f"{key}_label", "mask_note", col, width=1.5)
vertex("mask_note", "输入使用有效前缀掩码\nQ3 仅推理与解释，不训练模型", 65, 475, 510, 74, C["gray_fill"], C["line"], fs=18, color=C["muted"], rounded=True)
vertex("predictor", "冻结的 B5-P2 预测器\n（随机种子 42；参数保持不变）", 70, 580, 505, 62, C["purple_fill"], C["purple"], fs=23, bold=True, rounded=True, stroke_w=2)
edge("input_to_predictor", "mask_note", "predictor", C["purple"], width=1.8)
blocks = [
    ("masked_pool", "带掩码均值池化", 675),
    ("attention_pool", "注意力残差池化", 775),
    ("fusion_mlp", "融合多层感知机", 875),
    ("dual_head", "双头输出", 975),
]
for cid, label, y in blocks:
    vertex(cid, label, 112, y, 420, 64, "#FFFFFF", C["purple"], fs=21, bold=cid in ("attention_pool", "dual_head"), rounded=True, stroke_w=1.7)
for i in range(len(blocks)-1): edge(f"stage1_flow{i}", blocks[i][0], blocks[i+1][0], C["purple"], width=1.8)
edge("predict_to_pool", "predictor", "masked_pool", C["purple"], width=1.8)
vertex("class_output", "情感类别预测\n负向｜中性｜正向", 74, 1085, 240, 105, C["text_fill"], C["text"], fs=20, bold=True, rounded=True)
vertex("reg_output", "情感强度预测\n连续回归输出", 333, 1085, 240, 105, C["audio_fill"], C["audio"], fs=20, bold=True, rounded=True)
edge("head_to_class", "dual_head", "class_output", C["text"], width=1.6)
edge("head_to_reg", "dual_head", "reg_output", C["audio"], width=1.6)

# Stage 2: three hierarchical explanation levels.
vertex("heaf_outer", "", 695, 175, 940, 1125, "#FFFFFF", C["purple"], rounded=True, stroke_w=2)
vertex("shapley_panel", "", 715, 192, 900, 355, C["panel"], C["line"], rounded=True)
vertex("shapley_title", "一、模态级归因｜精确沙普利值（Shapley）", 735, 205, 855, 42, fs=22, bold=True, align="left")
coalitions = [("∅", "空集"), ("T", "文本"), ("A", "音频"), ("V", "视觉"), ("T+A", "文本+音频"), ("T+V", "文本+视觉"), ("A+V", "音频+视觉"), ("T+A+V", "三模态")]
for i, (code, name) in enumerate(coalitions):
    row, col = divmod(i, 4)
    x, y = 740+col*207, 265+row*66
    vertex(f"coalition{i}", f"{code}\n{name}", x, y, 188, 54, "#FFFFFF", C["line"], fs=17, bold=(i==7), rounded=True, stroke_w=1.2)
vertex("coalition_note", "对 8 种模态联盟逐一计算预测；分类与回归分别归因", 740, 405, 842, 34, fs=16, color=C["muted"], align="left")
for i, (lab, fill, stroke) in enumerate((("文本贡献", C["text_fill"], C["text"]), ("音频贡献", C["audio_fill"], C["audio"]), ("视觉贡献", C["vision_fill"], C["vision"]))):
    vertex(f"phi{i}", lab, 755+i*260, 460, 225, 58, fill, stroke, fs=20, bold=True, rounded=True)

vertex("interaction_panel", "", 715, 565, 900, 205, C["panel"], C["line"], rounded=True)
vertex("interaction_title", "二、交互级解释｜配对交互作用", 735, 578, 855, 38, fs=22, bold=True, align="left")
for i, (lab, c) in enumerate((("文本—音频", C["text"]), ("文本—视觉", C["vision"]), ("音频—视觉", C["audio"]))):
    vertex(f"interact{i}", lab, 750+i*275, 635, 245, 76, "#FFFFFF", c, fs=20, bold=True, rounded=True, stroke_w=1.7)
vertex("interaction_note", "依据联盟预测的二阶差分；负值称为负交互，不直接等同冗余", 740, 722, 850, 32, fs=16, color=C["muted"], align="left")

vertex("temporal_panel", "", 715, 788, 900, 370, C["panel"], C["line"], rounded=True)
vertex("temporal_title", "三、时间级解释｜连续窗口遮挡", 735, 800, 855, 40, fs=22, bold=True, align="left")
vertex("timeline_label", "有效序列槽位（50 槽示意）", 740, 850, 470, 28, fs=16, color=C["muted"], align="left")
for i in range(50):
    fill = "#F1F3F5" if i < 14 or i > 27 else "#F2C98D"
    stroke = "#C8CDD2" if i < 14 or i > 27 else C["orange"]
    vertex(f"slot{i}", "", 740+i*16.5, 890, 13, 34, fill, stroke, rounded=True, stroke_w=0.7)
vertex("window_note", "橙色仅示意滑动窗口；不表示固定区间", 740, 934, 850, 27, fs=15, color=C["muted"], align="left")
vertex("temporal_params", "ρ：0.30　｜　步长：1　｜　仅在有效前缀内滑动", 740, 970, 850, 34, C["purple_fill"], C["purple"], fs=17, bold=True, rounded=True)
vertex("temporal_compare", "比较遮挡前后的分类边际与回归输出", 740, 1018, 430, 50, "#FFFFFF", C["line"], fs=17, rounded=True)
vertex("key_interval", "关键特征区间\n起止槽位", 1200, 1018, 370, 50, C["audio_fill"], C["audio"], fs=18, bold=True, rounded=True)
vertex("heaf_summary", "解释汇总：分类主导模态　｜　回归主导模态　｜　关键交互　｜　关键特征区间", 730, 1180, 870, 75, C["purple_fill"], C["purple"], fs=18, bold=True, rounded=True)

# Stage 3: verified text branch and explicitly unverified raw A/V mappings.
branches = [
    ("ground_text", "文本证据", "文本特征行 → 词元位置\n→ 原文字符区间", "已验证", C["text"], C["text_fill"], False),
    ("ground_audio", "音频证据", "仅保留音频特征槽位区间\n原始秒级定位未验证", "未验证", C["audio"], C["audio_fill"], True),
    ("ground_vision", "视觉证据", "仅保留视觉特征槽位区间\n原始帧定位未验证", "未验证", C["vision"], C["vision_fill"], True),
]
for i, (cid, title, body, status, col, fill, dashed) in enumerate(branches):
    y = 195+i*205
    vertex(cid, "", 1710, y, 550, 170, "#FFFFFF" if dashed else fill, col if not dashed else "#9AA1A8", rounded=True, dashed=dashed, stroke_w=1.8)
    vertex(cid+"_title", title, 1730, y+12, 300, 38, fs=21, bold=True, align="left", color=col if not dashed else C["muted"])
    vertex(cid+"_status", status, 2070, y+10, 160, 38, C["green_fill"] if not dashed else C["gray_fill"], C["green"] if not dashed else "#9AA1A8", fs=17, bold=True, rounded=True, dashed=dashed)
    vertex(cid+"_body", body, 1732, y+62, 490, 75, "none", "none", fs=18, align="left", color=C["ink"])
vertex("ground_caveat", "文本证据可回溯；音频与视觉原始秒/帧映射未验证，\n因此不输出未经核验的时间戳或帧定位。", 1710, 825, 550, 105, C["gray_fill"], C["line"], fs=18, color=C["muted"], bold=True, rounded=True)
vertex("ground_interval", "音频 / 视觉结果保留为特征空间槽位区间", 1710, 960, 550, 65, C["purple_fill"], C["purple"], fs=18, rounded=True)

# Stage 4: concise structured explanation card.
vertex("card", "", 2330, 180, 595, 1025, "#FFFFFF", C["ink"], rounded=True, stroke_w=2)
vertex("card_title", "解释卡输出", 2350, 198, 555, 52, C["purple_fill"], C["purple"], fs=25, bold=True, rounded=True)
card_rows = [
    ("样本编号", "逐样本标识"), ("预测情感类别", "负向 / 中性 / 正向"),
    ("情感强度", "回归输出"), ("分类主导模态", "文本 / 音频 / 视觉"),
    ("回归主导模态", "文本 / 音频 / 视觉"), ("关键特征区间", "模态与槽位起止位置"),
    ("验证文本证据", "原文片段（若适用）"), ("音频 / 视觉证据", "未验证；原始秒 / 帧留空"),
]
for i, (lab, val) in enumerate(card_rows):
    y = 270+i*94
    fill = "#FFFFFF" if i%2 else C["panel"]
    vertex(f"cardrow{i}", "", 2350, y, 555, 76, fill, C["line"], rounded=True, stroke_w=0.9)
    vertex(f"cardlabel{i}", lab, 2365, y+8, 190, 58, "none", "none", fs=17, bold=True, align="left")
    vertex(f"cardvalue{i}", val, 2560, y+8, 330, 58, "none", "none", fs=16, color=C["muted"], align="left")
vertex("card_note", "按样本填写预测与归因结果；\n不以可回溯性改变主导模态。", 2350, 1035, 555, 98, C["gray_fill"], C["line"], fs=17, color=C["muted"], rounded=True)

vertex("footer", "最终解释卡用于未标注样本的推理说明；不代表预测正确性或现实因果关系。", 320, 1370, 2360, 50, fs=18, color=C["muted"])

ElementTree(mx).write(DRAWIO, encoding="utf-8", xml_declaration=True)
print(DRAWIO)
