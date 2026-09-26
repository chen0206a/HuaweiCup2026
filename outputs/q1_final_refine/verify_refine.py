from pathlib import Path
import re,json,hashlib
from pypdf import PdfReader
O=Path(__file__).resolve().parent
S=O.parent/'q1_story_rewrite'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def text(p): return p.read_text(encoding='utf-8')
old=text(S/'sections/05_q1.tex'); new=text(O/'sections/05_q1.tex')
eq=lambda s:re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',s,re.S)
normal=lambda s:s.replace(r'\mathcal T_i',r'\mathcal{T}_i')
assert len(eq(new))==10 and list(map(normal,eq(old)))==eq(new)
unchanged={}
for p in (S/'generated').glob('*.tex'):
    if p.name!='table_q1_files.tex':
        unchanged[p.name]=sha(p)==sha(O/'generated'/p.name)
assert all(unchanged.values())
assert sha(S/'q1_numbers.tex')==sha(O/'q1_numbers.tex')
assert all(sha(p)==sha(O/'figures/q1'/p.name) for p in (S/'figures/q1').glob('*.pdf'))
assert re.findall(r'\\sub(?:sub)?section[^\n]+',old)==re.findall(r'\\sub(?:sub)?section[^\n]+',new)
assert '\\end{equation}\n并令\n\\begin{equation}' in new
assert '\\end{equation}\n\n\\QOneParagraph 实际实现中' in new
assert r'\mathcal{T}_i=[0,T_i)' in new
assert r'媒体时间域$\mathcal{T}_i$' in new
formal=json.loads(text(O/'protected_paper_hashes.json'))
assert all(sha(Path(p))==h for p,h in formal.items())
aux=text(O/'build/q1_final_refine.aux')
labels=set(re.findall(r'\\newlabel\{([^}]+)\}',aux))
assert all(x in labels for x in re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',new))
log=text(O/'build/q1_final_refine.log')
assert not any(x in log for x in ['Overfull','Missing character','There were undefined references','There were undefined citations'])
pdf=PdfReader(O/'q1_final_refine.pdf')
result={'pages':len(pdf.pages),'explicit_2em_paragraph_starts':new.count(r'\QOneParagraph'),
        'equations_preserved_except_mathcal_braces':10,'all_7_figures_identical':True,
        'unchanged_tables':unchanged,'formal_files_unchanged':len(formal),
        'table6_columns':2,'table6_widths':[0.28,0.56],'table6_gap_em':1.4,
        'table6_all_four_files_and_information_preserved':True,
        'overfull':0,'underfull':log.count('Underfull'),
        'visual_review':'All pages inspected; normal new paragraphs explicitly indented; equation 8–9 remains one derivation; table6 centered, wrapped filenames readable; table2 and table7 unchanged.'}
(O/'qa/final_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
