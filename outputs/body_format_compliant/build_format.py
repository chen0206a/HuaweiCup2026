from pathlib import Path
import re, shutil, json, hashlib

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SRC=ROOT/'outputs/body_merged_preview'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=[]
for d in ['sections','generated','figures','qa','build']:(OUT/d).mkdir(exist_ok=True)
for d in ['sections','generated','figures']:
    shutil.copytree(SRC/d,OUT/d,dirs_exist_ok=True)
for name in ['q1_numbers.tex','references_working.tex']:
    shutil.copy2(SRC/name,OUT/name)
for p in [SRC/'body_merged_preview.tex',SRC/'body_merged_preview.pdf',SRC/'references_working.tex',SRC/'q1_numbers.tex']+list((SRC/'sections').glob('*.tex'))+list((SRC/'generated').glob('*.tex'))+list((SRC/'figures').rglob('*.pdf')):
    manifest.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})

for p in list((OUT/'sections').glob('*.tex'))+list((OUT/'generated').glob('*.tex'))+[OUT/'references_working.tex']:
    t=p.read_text(encoding='utf-8')
    t=re.sub(r'\\(?:small|footnotesize|scriptsize|tiny)(?![A-Za-z])',r'\\zihao{-4}',t)
    t=t.replace('font=small','font=compliant').replace('\\zihao{5}','\\zihao{-4}').replace('\\heiti','\\songti')
    # A paragraph indentation macro inside a table cell is layout, not body text.
    def clean_tables(m):return re.sub(r'\\Q(?:One|Two|Three)Paragraph\s*','',m[0])
    t=re.sub(r'\\begin\{(?:tabular|tabularx)\}.*?\\end\{(?:tabular|tabularx)\}',clean_tables,t,flags=re.S)
    p.write_text(t,encoding='utf-8')

def patch(name,a,b):
    p=OUT/'generated'/name;t=p.read_text(encoding='utf-8');assert a in t,(name,a);p.write_text(t.replace(a,b),encoding='utf-8')

patch('table_q1_all_samples.tex',r'\begin{longtable}{@{}p{4.1cm} r c c c c p{2.4cm} r@{}}',r'\begin{longtable}{@{}>{\raggedright\arraybackslash}p{4.1cm}>{\centering\arraybackslash}p{1.25cm}>{\centering\arraybackslash}p{1.15cm}>{\centering\arraybackslash}p{2.25cm}>{\centering\arraybackslash}p{1.3cm}>{\centering\arraybackslash}p{2.3cm}>{\centering\arraybackslash}p{1.9cm}>{\centering\arraybackslash}p{.9cm}@{}}')
patch('table_q1_all_samples.tex',r'样本ID & 时长/s & 模态 & 维度(T/A/V) & 对齐粒度 & 有效窗(T/A/V) & 文本来源 & 回退',r'样本ID & 时长/s & 模态 & \shortstack{维度\\(T/A/V)} & \shortstack{对齐\\粒度} & \shortstack{有效窗\\(T/A/V)} & \shortstack{文本\\来源} & 回退')
patch('table_q1_feature.tex',r'\begin{tabularx}{\textwidth}{@{}lXrrcr@{}}',r'\begin{tabularx}{\textwidth}{@{}lXcccr@{}}')
patch('table_q1_feature.tex',r'原生特征数 & 特征维数',r'\shortstack{原生\\特征数} & \shortstack{特征\\维数}')
patch('table_q1_feature.tex',r'平均时间覆盖率',r'\shortstack{平均时间\\覆盖率}')
patch('table_q1_feature.tex',r'\centering',r'\centering\zihao{-4}')
patch('table_q1_case_mapping.tex',r'{2.8cm}',r'{2.85cm}')
patch('table_q1_case_mapping.tex',r'原生索引',r'\shortstack{原生\\索引}')
patch('table_q1_case_mapping.tex',r'p{0.8cm}',r'p{1.0cm}')
patch('table_q1_case_mapping.tex',r'交叠时长/s',r'\shortstack{交叠时长\\/s}')

# Mean and sample SD stay unchanged; stack only their typography.
for name in ['table_q1_alignment.tex','table_q1_visual.tex','table_q1_modality.tex']:
    p=OUT/'generated'/name;t=p.read_text(encoding='utf-8')
    t=re.sub(r'(\d+\.\d+) \$\\pm\$ (\d+\.\d+)',lambda m:r'\shortstack{'+m[1]+r'\\$\pm$ '+m[2]+'}',t)
    p.write_text(t,encoding='utf-8')

for name in ['table_q2_main.tex','table_q2_public_baselines.tex','table_q2_public_missing.tex']:
    p=OUT/'generated'/name;t=p.read_text(encoding='utf-8')
    t=re.sub(r'\$(\d+\.\d+)\\pm(\d+\.\d+)\$',lambda m:r'\shortstack{$'+m[1]+r'$\\$\pm'+m[2]+'$}',t)
    if name!='table_q2_main.tex':
        t=t.replace(r'\begin{tabular}{llrcccc}',r'\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.0cm}>{\centering\arraybackslash}p{2.5cm}>{\centering\arraybackslash}p{2.1cm}*{4}{>{\centering\arraybackslash}X}@{}}').replace(r'\end{tabular}',r'\end{tabularx}')
        t=t.replace('完整输入训练',r'\shortstack{完整输入\\训练}').replace('缺失增强训练',r'\shortstack{缺失增强\\训练}')
        t=t.replace(r'宏平均F1$\uparrow$',r'\shortstack{宏平均\\F1$\uparrow$}').replace(r'相关系数$\uparrow$',r'\shortstack{相关系数\\$\uparrow$}')
    else:
        t=t.replace(r'\begin{tabular}{llcccc}',r'\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.5cm}l*{4}{>{\centering\arraybackslash}X}@{}}').replace(r'\end{tabular}',r'\end{tabularx}')
    p.write_text(t,encoding='utf-8')

patch('table_q2_public_overview.tex',r'p{.15\textwidth}',r'p{.17\textwidth}')
patch('table_q2_public_overview.tex',r'p{.17\textwidth}>{\raggedright\arraybackslash}X',r'p{.15\textwidth}>{\raggedright\arraybackslash}X')

# Formal template: only section titles use Hei; all remaining Han text uses Song.
main=(SRC/'body_merged_preview.tex').read_text(encoding='utf-8')
main=main.replace(r'\setmainfont{Times New Roman}',r'''\setmainfont{Times New Roman}
\setsansfont{Times New Roman}
\setmonofont{Times New Roman}
\setCJKmainfont[BoldFont=SimSun,ItalicFont=SimSun]{SimSun}
\setCJKsansfont[BoldFont=SimSun,ItalicFont=SimSun]{SimSun}
\setCJKmonofont[BoldFont=SimSun,ItalicFont=SimSun]{SimSun}''')
main=main.replace(r'\setstretch{1.12}',r'\setstretch{1}').replace('font=small','font=compliant')
main=main.replace(r'\captionsetup{font=compliant',r'\DeclareCaptionFont{compliant}{\songti\zihao{-4}}'+'\n'+r'\captionsetup{font=compliant')
main=main.replace(r'subsection={format=\heiti',r'subsection={format=\songti').replace(r'subsubsection={format=\heiti',r'subsubsection={format=\songti')
(OUT/'body_format_compliant.tex').write_text(main,encoding='utf-8')
(OUT/'qa/input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'.gitignore').write_text('build/\nqa/page-*.png\nqa/sheet-*.png\nqa/pdf_text.txt\n',encoding='utf-8')
(OUT/'.gitattributes').write_text('*.pdf binary\n',encoding='utf-8')
print('Formatting copy prepared; original figures retained as requested.')
