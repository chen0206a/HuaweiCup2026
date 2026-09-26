"""Re-export existing Q3 matplotlib figures; only labels are changed."""
from pathlib import Path
import json, re, hashlib
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation

O=Path(__file__).resolve().parent
ROOT=O.parent.parent
E=ROOT/'E2026'
ORIG=E/'outputs/final/q3/figures'
changes=[]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(fig):
    return [{'lines':[{'x':np.asarray(l.get_xdata()).tolist(),'y':np.asarray(l.get_ydata()).tolist()} for l in ax.lines],
             'bars':[[float(p.get_x()),float(p.get_y()),float(p.get_width()),float(p.get_height())] for p in ax.patches if hasattr(p,'get_width')],
             'collections':[[path.vertices.tolist() for path in c.get_paths()] for c in ax.collections],
             'images':[{'shape':list(i.get_array().shape),'sha256':hashlib.sha256(np.asarray(i.get_array()).tobytes()).hexdigest()} for i in ax.images]} for ax in fig.axes]
def load_script(name, replacements):
    p=ORIG/'scripts'/name
    s=p.read_text(encoding='utf-8-sig')
    s=s.replace('ROOT = Path(__file__).resolve().parents[5]',f'ROOT = Path({str(E)!r})')
    for old,new in replacements.items():
        assert old in s,old
        s=s.replace(old,new)
    target='fig15_q3_faithfulness_validation' if name.startswith('figure8') else 'fig16_q3_case_explanations'
    s=re.sub(r'^BASE = .*$',lambda _:f'BASE = Path({str(O/"figures/q3"/target)!r})',s,flags=re.M)
    s=re.sub(r'^PREVIEW = .*$',lambda _:f'PREVIEW = Path({str(O/"qa/figure_preview")!r})',s,flags=re.M)
    s=s.replace('    archive_previous()\n','').replace('    archive_old_figure()\n','')
    (O/'figure_sources'/name).write_text(s,encoding='utf-8')
    ns={'__file__':str(O/'figure_sources'/name),'__name__':'figure_source'}
    exec(compile(s,str(p),'exec'),ns)
    changes.append({'source':str(p),'source_sha256':sha(p),'labels':replacements})
    return ns
def export(fig, stem):
    p=O/'figures/q3'/stem
    for ext in ('pdf','png','svg'): fig.savefig(p.with_suffix('.'+ext),dpi=300,facecolor='white',bbox_inches=None)
    # Record plotted values, before any display transformation; no inference.
    vals=snapshot(fig)
    (O/'qa'/f'{stem}_plot_values.json').write_text(json.dumps(vals,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)

ns=load_script('figure8_q3_faithfulness_zh.py',{
    '分类边际下降':'分类对数优势下降','边际下降差值':'对数优势下降差',
    '分类边际变化':'分类对数优势变化','高贡献窗口':'高影响位置','高贡献区间与随机区间对比':'高影响位置与随机位置对比',
    '音频':'语音','分类主模态':'分类主导模态','回归主模态':'回归主导模态','随机等长窗口':'随机位置'})
export(ns['draw'](ns['load_metrics']()),'fig15_q3_faithfulness_validation')
ns=load_script('figure9_q3_case_studies_v2.py',{
    '分类对数赔率贡献':'分类对数优势贡献','类别边际下降':'对数优势下降',
    '时间遮挡曲线':'局部遮挡曲线','已验证文本证据':'原文字符级',
    '视觉特征级证据案例':'视觉主导典型案例','负面':'消极','正面':'积极','音频':'语音',
    '分类主模态':'分类主导模态'})
cards=ns['load_cards']()
fragment=ns['validate_text_fragment'](cards['14'])
frames=ns['validate_frames'](cards)
export(ns['draw'](cards,frames,fragment),'fig16_q3_case_explanations')

# Export an unchanged-label plot for numerical geometry comparison.
for script,stem,kind in [('figure8_q3_faithfulness_zh.py','fig15_q3_faithfulness_validation','metrics'),('figure9_q3_case_studies_v2.py','fig16_q3_case_explanations','cases')]:
    s=(ORIG/'scripts'/script).read_text(encoding='utf-8-sig').replace('ROOT = Path(__file__).resolve().parents[5]',f'ROOT = Path({str(E)!r})')
    ns0={'__file__':str(ORIG/'scripts'/script),'__name__':'figure_original'}
    exec(compile(s,script,'exec'),ns0)
    old=ns0['draw'](ns0['load_metrics']()) if kind=='metrics' else ns0['draw'](cards,frames,fragment)
    new=ns['draw'](cards,frames,fragment) if kind=='cases' else None
    # Compare line/bar arrays to the snapshot from the changed-label rendering.
    arr=snapshot(old)
    saved=json.loads((O/'qa'/f'{stem}_plot_values.json').read_text(encoding='utf-8'))
    assert arr==saved,stem
    plt.close(old)
    if new is not None: plt.close(new)

# Source-only edit: Office preflight disallows COM in the current session.
src=ROOT/'论文整理/paper_revision_work/paper/figures/q3/fig14_q3_heaf_framework_source.pptx'
p=Presentation(src)
replace={'HEAF解释分析':'预测解释分析','证据核验':'输入来源回溯','文本证据':'原文字符级',
         '音频/视觉特征':'语音/视觉','未对齐特征行':'未对齐特征行级','已核验':'','未核验':'','音频特征\n50×74':'语音特征\n50×74'}
for sh in p.slides[0].shapes:
    if sh.has_text_frame and sh.text in replace:
        text={279:'原文片段',280:'特征行'}.get(sh.shape_id,replace[sh.text])
        lines=text.split('\n')
        for index,para in enumerate(sh.text_frame.paragraphs):
            if para.runs:
                para.runs[0].text=lines[index] if index<len(lines) else ''
                for run in para.runs[1:]:run.text=''
p.save(O/'figure_sources/fig14_q3_framework_refined_source.pptx')
changes.append({'source':str(src),'source_sha256':sha(src),'labels':replace,'additional_shape_labels':{'279':'原文片段','280':'特征行'},'pdf_export':'export_framework.ps1; WTS-backed Office preflight'})
(O/'qa/figure_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
print('Two figures exported; all plotted line/bar values unchanged. Framework source prepared.')
