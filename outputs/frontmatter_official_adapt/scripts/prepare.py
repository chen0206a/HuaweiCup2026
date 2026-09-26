from pathlib import Path
import shutil,urllib.request,json,hashlib
import fitz
root=Path('outputs/frontmatter_official_adapt'); base=Path('outputs/layout_refine_round1')
for n in ['sections','generated','figures','frontmatter']:
 shutil.copytree(base/n,root/n,dirs_exist_ok=True)
shutil.copy2(base/'q1_numbers.tex',root/'q1_numbers.tex')
ref=root/'template_reference';ref.mkdir(exist_ok=True)
for repo,file in [('Suwren/2026_23rdCPGMCM_LaTeX','settings/format.tex'),('Suwren/2026_23rdCPGMCM_LaTeX','gmcmthesis.cls'),('GuangchenJ/Huawei_Cup_2026','gmcmthesis.cls'),('GuangchenJ/Huawei_Cup_2026','gmcm.bst'),('rudykon/GMCM2026-LaTeX-Template','gmcm2026.cls')]:
 try:
  data=urllib.request.urlopen('https://raw.githubusercontent.com/'+repo+'/main/'+file).read(); (ref/(repo.split('/')[0]+'_'+Path(file).name)).write_bytes(data)
 except Exception as e: print(file,e)
shutil.copy2(ref/'GuangchenJ_gmcm.bst',root/'gmcm.bst')
shutil.copy2(Path('outputs/封面目录摘要/official_cover.pdf'),root/'frontmatter/official_cover.pdf')
d=fitz.open('outputs/封面目录摘要/official_reference/template.pdf');out=fitz.open();p=out.new_page(width=d[1].rect.width,height=d[1].rect.height);p.show_pdf_page(fitz.Rect(0,115,595.3,211),d,1,clip=fitz.Rect(0,115,595.3,211));out.save(root/'frontmatter/official_abstract_header.pdf')
(root/'references.bib').write_bytes(Path('outputs/q1_final_refine/bibliography/references.bib').read_bytes())
(root/'gmcm.bst').write_bytes(urllib.request.urlopen('https://raw.githubusercontent.com/Suwren/2026_23rdCPGMCM_LaTeX/main/settings/gmcm.bst').read())
s=(base/'paper_layout_refine_round1.tex').read_text(encoding='utf-8-sig')
s=s.replace('\\usepackage[hidelinks]{hyperref}','\\usepackage[numbers,sort&compress]{natbib}\n\\usepackage[hidelinks]{hyperref}')
s=s.replace('\\begin{document}',r'''\makeatletter
\renewcommand{\l@section}{\@dottedtocline{1}{0em}{3.6em}}
\renewcommand{\l@subsection}{\@dottedtocline{2}{1.5em}{2.8em}}
\renewcommand{\l@subsubsection}{\@dottedtocline{3}{3em}{3.6em}}
\renewcommand{\@pnumwidth}{2em}
\renewcommand{\@tocrmarg}{2.5em}
\makeatother
\setlength{\bibsep}{0pt}
\begin{document}''')
s=s.replace('\\phantomsection\\label{front:abstract}',r'''\newgeometry{top=3cm,bottom=1.75cm,left=2.25cm,right=2.25cm}
\AddToHookNext{shipout/background}{\put(0,-\paperheight){\includegraphics[width=\paperwidth,height=\paperheight]{frontmatter/official_abstract_header.pdf}}}
\vspace*{4.5cm}
\phantomsection\label{front:abstract}''')
s=s.replace('\\setcounter{section}{2}',r'''\restoregeometry
% 待撰写：\input{sections/problem_restatement.tex}
% 待撰写：\input{sections/assumptions_symbols.tex}
\setcounter{section}{2}''')
s=s.replace('\\input{references_working.tex}',r'''\bibliographystyle{gmcm}
\bibliography{references}''')
a=s.index('\\clearpage\n\\phantomsection\n\\addcontentsline{toc}{section}{附录A')
s=s[:a]+r'''
% 待撰写：\input{sections/evaluation_limits.tex}
% 待撰写：\input{sections/conclusion.tex}
\clearpage
\appendix
\setcounter{section}{0}
\ctexset{section={name={附录,},number=\Alph{section}}}
\section{附件1完整特征构建结果}
\setcounter{table}{0}
\renewcommand{\thetable}{\Alph{section}\arabic{table}}
\renewcommand{\theHtable}{appendix.\Alph{section}.\arabic{table}}
\input{generated/table_q1_all_samples_appendix.tex}
% 附录B待撰写：\section{关键算法与复现信息}
% \input{sections/appendix_reproducibility.tex}
\end{document}
'''
(root/'paper_frontmatter_official_adapt.tex').write_text(s,encoding='utf-8')
print('Prepared',root)
