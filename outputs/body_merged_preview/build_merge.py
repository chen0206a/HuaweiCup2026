from pathlib import Path
import re, shutil, json, hashlib, difflib

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {1: ROOT/'outputs/q1_final_refine', 2: ROOT/'outputs/q2_point_final', 3: ROOT/'outputs/q3_point_final'}
NAMES = {1:'q1_final_refine', 2:'q2_point_final', 3:'q3_point_final'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = []
for folder in ['sections','generated','figures','qa']:
    (OUT/folder).mkdir(exist_ok=True)

def record(p):
    manifest.append({'path':str(p.relative_to(ROOT)).replace('\\','/'), 'sha256':sha(p)})

def copy(p, target):
    record(p)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target)

texts={}
diffs=[]
for q,src in SOURCES.items():
    record(src/(NAMES[q]+'.tex')); record(src/(NAMES[q]+'.pdf'))
    for p in (src/'figures'/f'q{q}').glob('*.pdf'):
        copy(p,OUT/'figures'/f'q{q}'/p.name)
    for p in (src/'generated').glob('*.tex'):
        copy(p,OUT/'generated'/p.name)
    if q==1:
        p=src/'sections/05_q1.tex'; record(p)
        original=p.read_text(encoding='utf-8')
    else:
        full=(src/(NAMES[q]+'.tex')).read_text(encoding='utf-8')
        original=full.split('\\begin{document}',1)[1].split('\\end{document}',1)[0]
        if q==2:
            original=original.split('\\clearpage\n{\\small\n\\begin{thebibliography}',1)[0]
    body=re.sub(r'\\setcounter\{(?:section|figure|table|equation)\}\{[^}]+\}', '', original)
    body=re.sub(r'^\\songti\\zihao\{-4\}(?:\\pagestyle\{plain\})?\s*\n', '', body, flags=re.M)
    body=body.strip()+'\n'
    if q==2:
        body=body.replace('\\includegraphics[width=.94\\textwidth]{{q2 架构图}.pdf}', '\\includegraphics[width=.94\\textwidth]{figures/q2/q2 架构图.pdf}')
        for p in (src/'figures/q2').glob('*.pdf'):
            if p.name!='q2 架构图.pdf':
                body=body.replace('{'+p.name+'}', '{figures/q2/'+p.name+'}')
    if q==3:
        body=body.replace('% 保留图16在第二页的原有分页','% 保留原稿方法流程图前的分页')
    texts[q]=body
    (OUT/'sections'/f'q{q}.tex').write_text(body,encoding='utf-8')
    diffs.append(''.join(difflib.unified_diff(original.splitlines(True),body.splitlines(True),fromfile=f'q{q}_source_body',tofile=f'q{q}_merged_body')))

copy(SOURCES[1]/'q1_numbers.tex',OUT/'q1_numbers.tex')
# Only a counter-aware continuation header and common caption formatting change.
p=OUT/'generated/table_q1_all_samples.tex'
t=p.read_text(encoding='utf-8')
changed=t.replace('续表2（接上页）','续表~\\ref{tab:q1_all_samples}（接上页）').replace('font=small,labelfont=bf','font=small,labelfont=normalfont')
p.write_text(changed,encoding='utf-8')
diffs.append(''.join(difflib.unified_diff(t.splitlines(True),changed.splitlines(True),fromfile='table_q1_all_samples_source',tofile='table_q1_all_samples_merged')))

# Two-digit global bibliography numbers need a little more room in the method column.
p=OUT/'generated/table_q2_public_overview.tex'
t=p.read_text(encoding='utf-8')
changed=t.replace('p{.13\\textwidth}', 'p{.15\\textwidth}')
p.write_text(changed,encoding='utf-8')
diffs.append(''.join(difflib.unified_diff(t.splitlines(True),changed.splitlines(True),fromfile='table_q2_public_overview_source',tofile='table_q2_public_overview_merged')))

# Preserve already typeset Q1 bibliography metadata and Q2 final bibliography text.
bbl=SOURCES[1]/'build/q1_final_refine.bbl'; record(bbl)
bib1=bbl.read_text(encoding='utf-8')
bib2=(SOURCES[2]/'q2_point_final.tex').read_text(encoding='utf-8').split('\\begin{thebibliography}{99}',1)[1].split('\\end{thebibliography}',1)[0]
entries={}
for bib in [bib1,bib2]:
    chunks=re.split(r'(?=\\bibitem\{)',bib)
    for chunk in chunks:
        m=re.match(r'\\bibitem\{([^}]+)\}',chunk)
        if m:
            chunk=chunk.split('\\end{thebibliography}')[0].strip()
            assert m[1] not in entries
            entries[m[1]]=chunk

def expand_inputs(text):
    return re.sub(r'\\input\{([^}]+)\}',lambda m: expand_inputs((OUT/(m[1]+('' if m[1].endswith('.tex') else '.tex'))).read_text(encoding='utf-8')),text)
expanded='\n'.join(expand_inputs(texts[q]) for q in [1,2,3])
cites=[]
for match in re.finditer(r'\\cite\{([^}]+)\}',expanded):
    for key in match[1].split(','):
        if key not in cites: cites.append(key)
assert set(cites)==set(entries) and len(cites)==16
refs='{\\small\n\\begin{thebibliography}{99}\n\n'+'\n\n'.join(entries[k] for k in cites)+'\n\n\\end{thebibliography}\n}\n'
(OUT/'references_working.tex').write_text(refs,encoding='utf-8')

main=r'''\documentclass[UTF8,a4paper,12pt]{ctexart}
\usepackage[a4paper,top=2.4cm,bottom=2.2cm,left=2.25cm,right=2.25cm]{geometry}
\setmainfont{Times New Roman}
\usepackage{setspace,graphicx,booktabs,array,multirow,longtable,tabularx}
\usepackage{amsmath,amssymb,bm,mathtools,siunitx}
\usepackage{caption,subcaption,float,xcolor,enumitem,placeins,needspace}
\usepackage[hidelinks]{hyperref}
\usepackage[nameinlink,noabbrev]{cleveref}
\input{q1_numbers.tex}
\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\setstretch{1.12}
\newcommand{\QOneParagraph}{\par\noindent\hspace*{2em}}
\newcommand{\QTwoParagraph}{\par\noindent\hspace*{2em}}
\newcommand{\QThreeParagraph}{\par\noindent\hspace*{2em}}
\newcounter{q2algorithm}
\captionsetup{font=small,labelfont=normalfont,labelsep=quad,justification=centering}
\ctexset{
 section={format=\centering\heiti\zihao{4},beforeskip=1.5ex,afterskip=1.2ex},
 subsection={format=\heiti\zihao{-4},beforeskip=1.2ex,afterskip=.6ex},
 subsubsection={format=\heiti\zihao{-4},beforeskip=.8ex,afterskip=.4ex}}
\renewcommand{\refname}{参考文献}
\graphicspath{{figures/q1/}{figures/q2/}{figures/q3/}}
\begin{document}
\songti\zihao{-4}\pagestyle{plain}
\setcounter{section}{2}
\input{sections/q1.tex}
\FloatBarrier
\input{sections/q2.tex}
\FloatBarrier
\input{sections/q3.tex}
\FloatBarrier
\clearpage
\input{references_working.tex}
\end{document}
'''
(OUT/'body_merged_preview.tex').write_text(main,encoding='utf-8')
(OUT/'qa/input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'qa/merge_only.diff').write_text('\n'.join(diffs),encoding='utf-8')
(OUT/'qa/reference_order.json').write_text(json.dumps(cites,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'.gitignore').write_text('build/\nqa/page-*.png\nqa/sheet-*.png\nqa/pdf_text.txt\n',encoding='utf-8')
(OUT/'.gitattributes').write_text('*.pdf binary\n',encoding='utf-8')
print('Merged 3 frozen sources, 17 figures, 20 tables, 16 bibliography entries.')
