"""Move Q3 method content into a copy of the Q2 slide's four-box layout.

Run from the repository root. The two Desktop PPTXs remain untouched; this
script replaces only q2_framework_q3_draft.pptx in its output directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import wave
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent
Q2_SOURCE = Path(r"C:\Users\17299\Desktop\q2 架构图.pptx")
Q3_SOURCE = Path(r"C:\Users\17299\Desktop\q3架构tu1.pptx")
Q3_COPY = HERE / "q3_information_reference_copy.pptx"
OUTPUT = HERE / "q2_framework_q3_draft.pptx"
FRAME = FIGURES / "data/figure7_sample09_context_frame.png"
WAV = FIGURES / "data/figure7_sample09_audio.wav"
MANIFEST = HERE / "transfer_manifest.json"

INK = "303942"
MUTED = "68737D"
FLOW = "2F6FAE"
TEXT = "A9D3E9"
AUDIO = "EFD98B"
VISION = "EFB7B2"
PALE_TEXT = "EDF7FB"
PALE_AUDIO = "FFF9E9"
PALE_VISION = "FDF0EF"
PALE_LILAC = "F8F4FA"
PALE_GREEN = "EDF7EF"
GREEN = "46765A"
GRID = "D4DDE4"
WHITE = "FFFFFF"


def rgb(code: str) -> RGBColor:
    return RGBColor.from_string(code)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rect(slide, x, y, w, h, *, fill=WHITE, edge=GRID, rounded=True,
         dash=False, width=1.0, transparent=False):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    box = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if transparent:
        box.fill.background()
    else:
        box.fill.solid()
        box.fill.fore_color.rgb = rgb(fill)
    box.line.color.rgb = rgb(edge)
    box.line.width = Pt(width)
    if dash:
        box.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if rounded:
        try:
            box.adjustments[0] = 0.09
        except (IndexError, AttributeError):
            pass
    return box


def label(slide, content, x, y, w, h, *, size=14, bold=False, color=INK,
          align="center", valign=MSO_ANCHOR.MIDDLE):
    item = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = item.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = valign
    for index, part in enumerate(content.split("\n")):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.text = part
        p.alignment = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT}[align]
        p.space_before = p.space_after = Pt(0)
        for run in p.runs:
            run.font.name = "Microsoft YaHei"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = rgb(color)
    return item


def boxed(slide, content, x, y, w, h, *, fill=WHITE, edge=GRID,
          size=13.0, bold=False, color=INK):
    rect(slide, x, y, w, h, fill=fill, edge=edge)
    label(slide, content, x+0.03, y+0.03, w-0.06, h-0.06,
          size=size, bold=bold, color=color)


def stroke(slide, x1, y1, x2, y2, *, color=FLOW, width=1.3):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                            Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = rgb(color)
    connector.line.width = Pt(width)
    return connector


def arrow(slide, x1, y1, x2, y2, *, color=FLOW, width=1.5):
    stroke(slide, x1, y1, x2, y2, color=color, width=width)
    tip = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,
                                 Inches(x2-0.066), Inches(y2-0.066),
                                 Inches(0.132), Inches(0.132))
    tip.rotation = 90 + math.degrees(math.atan2(y2-y1, x2-x1))
    tip.fill.solid()
    tip.fill.fore_color.rgb = rgb(color)
    tip.line.fill.background()


def keep_q2_structure(slide) -> None:
    group = next(s for s in slide.shapes if s.shape_type == MSO_SHAPE_TYPE.GROUP)
    original_children = list(group.shapes)
    if len(original_children) < 7:
        raise ValueError("Q2 diagram no longer has the four expected outer containers")
    kept = (0, 2, 4, 6)
    for index, item in enumerate(original_children):
        if index not in kept:
            group.shapes._spTree.remove(item._element)
    for item in list(slide.shapes):
        if item is not group:
            slide.shapes._spTree.remove(item._element)
    original_bounds = [(round(s.left/914400, 2), round(s.top/914400, 2),
                        round(s.width/914400, 2), round(s.height/914400, 2))
                       for s in group.shapes]
    expected = [(0.16,0.75,3.74,5.68),(3.98,0.68,5.93,5.68),
                (10.12,0.79,2.79,2.51),(10.23,3.43,2.76,2.90)]
    if original_bounds != expected:
        raise ValueError(f"Q2 panel geometry changed: {original_bounds}")
    for item in group.shapes:
        item.fill.background()
        item.line.fill.background()
    # Four overlays reuse the measured Q2 boxes because grouped source outlines
    # disappear after the group is stripped by python-pptx / PowerPoint.
    for (x,y,w,h), color in zip(expected, (TEXT,"D9C8E2",AUDIO,"BBDDD9")):
        rect(slide,x,y,w,h,edge=color,transparent=True,dash=True,width=1.3)


def feature_cloud(slide, name, dims, x, y, fill, edge):
    cloud = slide.shapes.add_shape(MSO_SHAPE.CLOUD, Inches(x), Inches(y),
                                   Inches(1.54), Inches(0.93))
    cloud.fill.solid()
    cloud.fill.fore_color.rgb = rgb(fill)
    cloud.line.color.rgb = rgb(edge)
    cloud.line.width = Pt(1.0)
    label(slide, f"{name}\n{dims}", x+0.17, y+0.19, 1.20, 0.53,
          size=12.6, bold=True)


def audio_waveform(slide):
    with wave.open(str(WAV), "rb") as source:
        assert source.getframerate() == 16000 and source.getnchannels() == 1
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(float)
    rms = np.array([np.sqrt(np.mean(chunk*chunk)) for chunk in np.array_split(samples,60)])
    peak = float(np.max(rms))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("The paired audio waveform is empty")
    stroke(slide,0.36,3.65,1.65,3.65,color="C9B470",width=0.4)
    for i,value in enumerate(rms):
        x=0.36+1.29*i/(len(rms)-1)
        half=0.02+0.29*value/peak
        stroke(slide,x,3.65-half,x,3.65+half,color="AA7A2E",width=0.9)


def input_panel(slide) -> None:
    boxed(slide,"① 多模态输入特征",1.11,0.52,1.74,0.39,
          edge=TEXT,size=14.0,bold=True)
    for title,y,color in (("文本",1.15,TEXT),("音频",2.91,AUDIO),("视觉",4.58,VISION)):
        rect(slide,0.47,y+0.03,0.10,0.24,fill=color,edge=color,rounded=False)
        label(slide,title,0.60,y,0.69,0.31,size=15.0,bold=True,align="left")
    boxed(slide,"“I did not like this\nmovie at all”",0.28,1.52,1.47,0.70,
          fill=PALE_TEXT,edge=TEXT,size=11.0)
    label(slide,"原文片段",0.34,2.25,1.30,0.21,size=10.5,color=MUTED)
    audio_waveform(slide)
    label(slide,"原视频音轨波形",0.32,4.03,1.40,0.21,size=10.1,color=MUTED)
    slide.shapes.add_picture(str(FRAME),Inches(0.34),Inches(5.00),width=Inches(1.40))
    label(slide,"原视频场景（非定位帧）",0.26,6.12,1.60,0.20,
          size=9.3,color=MUTED)
    for y,name,dims,fill,edge in ((1.53,"文本特征","50×768",PALE_TEXT,TEXT),
                                  (3.30,"音频特征","50×74",PALE_AUDIO,AUDIO),
                                  (5.00,"视觉特征","50×35",PALE_VISION,VISION)):
        arrow(slide,1.78,y+0.47,2.03,y+0.47,width=1.2)
        feature_cloud(slide,name,dims,2.07,y,fill,edge)
    for y in (2.73,4.47):
        stroke(slide,0.33,y,3.70,y,color="E3E9ED",width=0.7)
    # Keep the Q2 source's three parallel left-to-centre transitions.
    for y in (1.99,3.76,5.47):
        arrow(slide,3.80,y,4.13,y,color="E6A04B",width=1.8)


def predictor_panel(slide) -> None:
    boxed(slide,"② 情感预测与 HEAF",5.39,0.52,2.60,0.39,
          edge="D9C8E2",size=14.5,bold=True)
    label(slide,"双路池化预测",4.23,1.02,2.40,0.26,
          size=13.5,bold=True,color=FLOW,align="left")
    boxed(slide,"三模态\n特征",4.26,1.54,1.03,0.77,
          fill=WHITE,edge=GRID,size=12.6,bold=True)
    boxed(slide,"掩码均值池化",5.61,1.37,1.53,0.49,
          fill=PALE_TEXT,edge=TEXT,size=12.3,bold=True)
    boxed(slide,"注意力残差池化",5.61,2.03,1.53,0.49,
          fill=PALE_LILAC,edge="D9C8E2",size=11.6,bold=True)
    stroke(slide,5.31,1.91,5.48,1.91,color=FLOW,width=1.4)
    stroke(slide,5.48,1.61,5.48,2.28,color=FLOW,width=1.2)
    arrow(slide,5.48,1.61,5.58,1.61,width=1.2)
    arrow(slide,5.48,2.28,5.58,2.28,width=1.2)
    for y in (1.62,2.28):
        stroke(slide,7.16,y,7.49,y,color=FLOW,width=1.2)
    stroke(slide,7.49,1.62,7.49,2.28,color=FLOW,width=1.1)
    rect(slide,7.35,1.79,0.28,0.28,fill=PALE_LILAC,edge="BDAFCC")
    label(slide,"＋",7.36,1.78,0.26,0.28,size=17,bold=True)
    arrow(slide,7.65,1.95,7.89,1.95,width=1.3)
    boxed(slide,"融合表示",7.92,1.53,1.65,0.82,
          fill=WHITE,edge=GRID,size=13.7,bold=True)
    stroke(slide,4.23,2.78,9.69,2.78,color=GRID,width=0.75)
    arrow(slide,6.93,2.54,6.93,3.03,width=1.0)
    label(slide,"HEAF 层级解释",4.22,3.01,2.60,0.30,
          size=14.7,bold=True,color=FLOW,align="left")


def heaf_panel(slide) -> None:
    xs=(4.21,5.60,6.99,8.38)
    captions=("八种联盟","Shapley贡献","模态交互","时间遮挡")
    for x,name in zip(xs,captions):
        rect(slide,x,3.54,1.28,1.63,fill=WHITE,edge=GRID)
        label(slide,name,x+0.04,4.73,1.20,0.27,size=11.9,bold=True)
    for i in range(3):
        arrow(slide,xs[i]+1.30,4.29,xs[i+1]-0.03,4.29,width=1.1)
    for i in range(8):
        rect(slide,4.36+(i%4)*0.27,3.82+(i//4)*0.29,0.20,0.18,
             fill=PALE_TEXT,edge=TEXT,rounded=False,width=0.7)
    for i,(color,w) in enumerate(((TEXT,0.71),(AUDIO,0.60),(VISION,0.50))):
        rect(slide,5.80,3.84+i*0.27,w,0.12,fill=color,edge=color,rounded=False)
    for x,y,color in ((7.31,3.88,TEXT),(7.73,3.88,AUDIO),(7.52,4.28,VISION)):
        rect(slide,x,y,0.13,0.13,fill=color,edge=color)
    stroke(slide,7.37,3.95,7.79,3.95,color=MUTED,width=0.9)
    stroke(slide,7.39,4.00,7.58,4.34,color=MUTED,width=0.9)
    stroke(slide,7.79,4.00,7.58,4.34,color=MUTED,width=0.9)
    for i in range(10):
        c="FBE7CB" if 4<=i<=6 else "E6EEF2"
        rect(slide,8.47+i*0.109,4.03,0.098,0.34,fill=c,edge=WHITE,
             rounded=False,width=0.1)
    boxed(slide,"主导模态  ·  模态交互  ·  关键特征区间",4.43,5.44,5.04,0.51,
          fill=PALE_LILAC,edge="D9C8E2",size=13.2,bold=True)
    label(slide,"归因  →  局部遮挡  →  删除对照",4.52,6.02,4.88,0.20,
          size=10.3,color=MUTED)


def right_panels(slide) -> None:
    boxed(slide,"③ 预测与解释卡",10.57,0.59,1.99,0.40,
          edge=AUDIO,size=14.2,bold=True)
    for i,(tag,detail) in enumerate((("预测","类别 · 情感强度"),
                                     ("模态","贡献 · 主导模态"),
                                     ("局部","交互 · 特征区间"))):
        y=1.22+0.62*i
        boxed(slide,tag,10.37,y,0.59,0.42,edge=GRID,size=11.7,bold=True)
        boxed(slide,detail,11.06,y,1.59,0.42,edge=GRID,size=10.6)
    arrow(slide,9.79,1.96,10.21,1.96,width=1.7)
    arrow(slide,11.58,3.33,11.58,3.50,width=1.3)
    boxed(slide,"④ 证据回溯与有效性",10.34,3.38,2.54,0.41,
          edge="BBDDD9",size=13.4,bold=True)
    boxed(slide,"文本证据：原文可回溯",10.42,4.06,2.39,0.43,
          fill=PALE_GREEN,edge="B8D7C1",size=11.2,bold=True,color=GREEN)
    boxed(slide,"音 / 视：仅特征槽位",10.42,4.65,2.39,0.43,
          fill="F5F7F8",edge=GRID,size=11.1,color=MUTED)
    label(slide,"原始秒数与帧号未验证",10.42,5.11,2.39,0.24,
          size=10.1,color=MUTED)
    boxed(slide,"关键删除  ↔  随机删除",10.42,5.54,2.39,0.47,
          fill=PALE_TEXT,edge=TEXT,size=10.9)
    arrow(slide,9.80,5.73,10.25,5.73,width=1.6)


def main() -> None:
    for path in (Q2_SOURCE,Q3_SOURCE,Q3_COPY,FRAME,WAV):
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(Q3_COPY) != digest(Q3_SOURCE):
        raise ValueError("Q3 information copy differs from source")
    shutil.copy2(Q2_SOURCE,OUTPUT)
    deck=Presentation(str(OUTPUT))
    if len(deck.slides)!=1 or (deck.slide_width,deck.slide_height)!=(12192000,6858000):
        raise ValueError("Source Q2 deck is not the expected one-slide 16:9 architecture")
    slide=deck.slides[0]
    keep_q2_structure(slide)
    input_panel(slide)
    predictor_panel(slide)
    heaf_panel(slide)
    right_panels(slide)
    label(slide,"图7  情感预测与层级证据归因框架",0.50,0.05,12.33,0.36,
          size=19.6,bold=True)
    label(slide,"原视频画面和音频波形仅作输入示意，不表示关键时间或帧定位。",
          0.64,6.81,12.0,0.22,size=10.7,color=MUTED)
    deck.save(str(OUTPUT))
    check=Presentation(str(OUTPUT))
    if len(check.slides)!=1 or len(check.slides[0].shapes)<100:
        raise RuntimeError("Output PPTX could not be reopened with expected shape content")
    manifest={
        "q2_source":str(Q2_SOURCE),"q2_source_sha256":digest(Q2_SOURCE),
        "q3_source":str(Q3_SOURCE),"q3_source_sha256":digest(Q3_SOURCE),
        "q3_reference_copy":str(Q3_COPY),"q3_reference_copy_sha256":digest(Q3_COPY),
        "edited_q2_copy":str(OUTPUT),"edited_q2_copy_sha256":digest(OUTPUT),
        "q2_framework": "original four dashed regions, measured coordinates, three parallel modality lanes and left-to-centre arrows",
        "q3_content": ["sentiment prediction", "eight-coalition Shapley", "pairwise interaction",
                       "continuous temporal occlusion", "explanation card", "raw evidence and deletion comparison"],
        "omitted_q3_example_values": ["Negative", "0.56", "0.28", "0.16", "10S/20S/30S"],
        "reason_for_omission": "the supplied method diagram does not document these numbers as real model outputs; A/V slot-to-time grounding is unverified",
        "input_media_sample_id":"09",
        "input_text_fragment":"I did not like this movie at all",
        "input_frame":str(FRAME),"input_frame_sha256":digest(FRAME),
        "input_audio":str(WAV),"input_audio_sha256":digest(WAV),
        "media_role":"source-input illustration only; not HEAF-localized evidence",
    }
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("CREATED",OUTPUT,"shapes",len(check.slides[0].shapes))


if __name__=="__main__":
    main()
