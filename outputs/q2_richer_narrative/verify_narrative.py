from pathlib import Path
import re,json,hashlib,csv
from pypdf import PdfReader
O=Path(__file__).resolve().parent; S=O.parent/'q2_final_refine'; ROOT=O.parent.parent
t=(O/'q2_richer_narrative.tex').read_text(encoding='utf-8'); old=(S/'q2_final_refine.tex').read_text(encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def env(s,n):return re.findall(r'\\begin\{'+n+r'\}.*?\\end\{'+n+r'\}',s,re.S)
checks={}
checks['18_equations_byte_identical']=len(env(t,'equation'))==18 and env(t,'equation')==env(old,'equation')
checks['all_inline_tables_figures_algorithms_and_bibliography_identical']=all(env(t,n)==env(old,n) for n in ['table','figure','itemize','thebibliography'])
checks['all_subsection_titles_identical']=all(re.findall(r'\\'+n+r'\{[^}]*\}',t)==re.findall(r'\\'+n+r'\{[^}]*\}',old) for n in ['section','subsection','subsubsection'])
checks['all_six_table_files_and_seven_figures_byte_identical']=all(sha(p)==sha(S/p.relative_to(O)) for name in ['generated','figures'] for p in (O/name).rglob('*') if p.is_file())
hashes=json.loads((O/'qa/input_hashes.json').read_text(encoding='utf-8'))
checks['previous_version_unchanged']=all(sha(Path(p))==h for p,h in hashes.items())
protected=json.loads((S/'qa/protected_files.json').read_text(encoding='utf-8'))
checks['formal_paper_and_model_code_unchanged']=all(sha(Path(p))==h for p,h in protected.items())
# Validate only the 12 methods actually retained in the paper, not a larger historical comparison.
ranking={}
for variant in ['baselines','missing']:
    content=(O/'generated'/('table_q2_public_'+variant+'.tex')).read_text(encoding='utf-8')
    names=[l.split('&')[0].strip() for l in content.splitlines() if re.search(r'\$\d+\.\d+\\pm',l)]
    p=ROOT/'E2026/outputs/final/q2/public_baselines_expanded'/('q2_public_baselines_'+('clean' if variant=='baselines' else 'missing')+'_summary.csv')
    with p.open(encoding='utf-8-sig',newline='') as f: data={r['model']:r for r in csv.DictReader(f)}
    rows={n:data['P2' if n=='LTARP' else n] for n in names}
    ranking[variant]={m:sorted(names,key=lambda n:float(rows[n][m+'_mean']),reverse=m!='mae')[0] for m in ['accuracy','macro_f1','mae','pearson']}
checks['clean_four_best_mean_claim_supported']=all(n=='LTARP' for n in ranking['baselines'].values())
checks['missing_three_best_mean_claim_supported']=all(ranking['missing'][m]=='LTARP' for m in ['macro_f1','mae','pearson'])
intro=t.split(r'\label{subsec:q2_plan}')[1].split(r'\begin{figure}')[0]
checks['intro_three_paragraphs']=intro.count(r'\QTwoParagraph')==3
checks['ordinary_paragraph_2em_macro_preserved']=r'\newcommand{\QTwoParagraph}{\par\noindent\hspace*{2em}}' in t and not re.search(r'\n\n[\u4e00-\u9fff]',t)
aux=(O/'build/q2_richer_narrative.aux').read_text(encoding='utf-8')
figs=[int(x) for x in re.findall(r'\\newlabel\{fig:q2_[^@}]+\}\{\{(\d+)\}',aux)]
tables=[int(x) for x in re.findall(r'\\newlabel\{tab:q2_[^@}]+\}\{\{(\d+)\}',aux)]
checks['numbers_preserved']=figs==list(range(8,15)) and tables==list(range(8,14))
log=(O/'build/q2_richer_narrative.log').read_text(encoding='utf-8',errors='replace')
checks['no_overfull_undefined_missing_character']=not re.search('Overfull|undefined|Undefined|Missing character:|\n!',log)
pdf=PdfReader(O/'q2_richer_narrative.pdf')
(O/'qa/pdf_text.txt').write_text('\n\n'.join(p.extract_text() or '' for p in pdf.pages),encoding='utf-8')
r=dict(checks=checks,passed=all(checks.values()),pages=len(pdf.pages),ranking_among_table_methods=ranking,overfull=log.count('Overfull'),underfull=log.count('Underfull'),figure_numbers=figs,table_numbers=tables)
(O/'qa/verification.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False)); assert r['passed']
