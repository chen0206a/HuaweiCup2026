"""Check existing result preservation; never run models or read raw datasets."""
from pathlib import Path
import csv,hashlib,json,re
from pypdf import PdfReader
O=Path(__file__).resolve().parent
ROOT=O.parent.parent
S=O.parent/'q2_story_rewrite'
checks={}
def check(n,v): checks[n]=bool(v)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def env(t,n): return re.findall(r'\\begin\{'+n+r'\}.*?\\end\{'+n+r'\}',t,re.S)
tex=(O/'q2_final_refine.tex').read_text(encoding='utf-8')
old=(S/'q2_story_rewrite.tex').read_text(encoding='utf-8')
generated={p.stem:p.read_text(encoding='utf-8') for p in (O/'generated').glob('*.tex')}
expanded=tex
for n,c in generated.items(): expanded=expanded.replace(r'\input{generated/'+n+'.tex}',c)
protected=json.loads((O/'qa/protected_files.json').read_text(encoding='utf-8'))
check('source_formal_paper_and_model_code_unchanged',all(Path(p).is_file() and sha(Path(p))==h for p,h in protected.items()))
check('seven_figure_assets_identical',len(list((O/'figures/q2').glob('*.pdf')))==7 and all(sha(p)==sha(S/'figures/q2'/p.name) for p in (O/'figures/q2').glob('*.pdf')))
check('all_existing_numeric_tables_identical',all((O/'generated'/p.name).read_bytes()==p.read_bytes() for p in (S/'generated').glob('*.tex') if p.name!='table_q2_public_overview.tex'))
check('eleven_subsections_and_subsection_titles_unchanged',re.findall(r'\\subsection\{[^}]*\}',tex)==re.findall(r'\\subsection\{[^}]*\}',old) and len(re.findall(r'\\subsection\{',tex))==11)
check('subsubsection_titles_unchanged',re.findall(r'\\subsubsection\{[^}]*\}',tex)==re.findall(r'\\subsubsection\{[^}]*\}',old))
check('ordinary_paragraphs_explicit_2em_indent',r'\newcommand{\QTwoParagraph}{\par\noindent\hspace*{2em}}' in tex and not re.search(r'\n\n[\u4e00-\u9fff]',tex))
eqs=env(tex,'equation'); oeqs=env(old,'equation')
check('18_equations_after_removing_duplicate_mean',len(eqs)==18 and len(oeqs)==19)
check('other_16_equations_identical',all(eqs[i]==oeqs[i] for i in range(18) if i not in [7,10]))
check('wce_matches_existing_weighted_mean',r'\sum_{i=1}^N w_{c_i}' in eqs[7])
check('attention_matches_existing_linear_scorer',r'\mathbf w_m^\top\mathbf x_{it}^{(m)} + b_m' in eqs[10])
labels=re.findall(r'\\label\{([^}]+)\}',expanded)
refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',expanded)
check('all_labels_resolve_without_duplicates',set(refs)<=set(labels) and len(labels)==len(set(labels)))
check('11_citations_resolve',set(re.findall(r'\\cite\{([^}]+)\}',expanded))==set(re.findall(r'\\bibitem\{([^}]+)\}',tex)) and len(re.findall(r'\\bibitem\{',tex))==11)
aux=(O/'build/q2_final_refine.aux').read_text(encoding='utf-8')
figure_numbers=[int(x) for x in re.findall(r'\\newlabel\{fig:q2_[^@}]+\}\{\{(\d+)\}',aux)]
table_numbers=[int(x) for x in re.findall(r'\\newlabel\{tab:q2_[^@}]+\}\{\{(\d+)\}',aux)]
check('figure_numbers_8_to_14_unchanged',figure_numbers==list(range(8,15)))
check('table_numbers_8_to_13_unchanged',table_numbers==list(range(8,14)))
log=(O/'build/q2_final_refine.log').read_text(encoding='utf-8',errors='replace')
check('no_overfull_or_undefined_or_missing_characters','Overfull' not in log and not re.search(r'undefined|Undefined|Missing character:|\n!',log))
metrics=['accuracy','macro_f1','mae','pearson']
for variant in ['baselines','missing']:
    path=ROOT/'E2026/outputs/final/q2/public_baselines_expanded'/('q2_public_baselines_'+('clean' if variant=='baselines' else 'missing')+'_summary.csv')
    data={r['model']:r for r in readcsv(path)}
    for line in generated['table_q2_public_'+variant].splitlines():
        vals=re.findall(r'\$(\d+\.\d+)\\pm(\d+\.\d+)\$',line)
        if not vals:continue
        model=line.split('&')[0].strip(); row=data['P2' if model=='LTARP' else model]
        check('table_'+variant+'_'+model, len(vals)==4 and all(a==f"{float(row[m+'_mean']):.4f}" and b==f"{float(row[m+'_sample_sd']):.4f}" for m,(a,b) in zip(metrics,vals)))
fairpath=ROOT/'E2026/outputs/final/q2/checkpoint_selection_fairness/q2_cleanselect_fair_comparison_summary.csv'
fair={r['model']:r for r in readcsv(fairpath)}
for line in generated['table_q2_main'].splitlines():
    vals=re.findall(r'\$(\d+\.\d+)\\pm(\d+\.\d+)\$',line)
    if not vals:continue
    condition,model=[x.strip() for x in line.split('&')[:2]]
    prefix='clean_' if condition=='完整输入' else 'mean_missing_'; row=fair['B0' if model=='MMP' else 'P2']
    check('main_'+condition+'_'+model,len(vals)==4 and all(a==f"{float(row[prefix+m+'_mean']):.4f}" and b==f"{float(row[prefix+m+'_sample_sd']):.4f}" for m,(a,b) in zip(metrics,vals)))
pred=readcsv(ROOT/'E2026/outputs/final/q2/attachment3/attachment3_predictions.csv')
rows=re.findall(r'\\detokenize\{(附件3_\d+)\}\} & (消极|中性|积极) & ([+-]?\d+\.\d+) & (\d+\.\d+)',expanded)
byid={r['sample_id']:r for r in pred}; cn={'Negative':'消极','Neutral':'中性','Positive':'积极'}
audit_byid={r['sample_id']:r for r in readcsv(ROOT/'E2026/outputs/final/q2/attachment3/attachment3_predictions_audit.csv')}
check('all_30_predictions_unchanged',len(rows)==len({r[0] for r in rows})==30 and all(c==cn[byid[i]['predicted_class_name']] and abs(float(v)-float(byid[i]['predicted_intensity']))<=.00000051 and abs(float(conf)-float(audit_byid[i]['confidence']))<=.000051 for i,c,v,conf in rows))
manifest=json.loads((ROOT/'E2026/outputs/final/q2/q2_checkpoint_manifest.json').read_text(encoding='utf-8'))
lockrows=[]
for r in manifest['checkpoints']:
    cp=r['checkpoint']; p=ROOT/'E2026'/cp['relative_path']; ok=sha(p)==cp['sha256']
    check('locked_checkpoint_'+r['model']+'_'+str(r['seed']),ok)
    lockrows.append(dict(model=r['model'],seed=r['seed'],sha256=cp['sha256'],criterion=r['selection_criterion'],epoch=r['best_epoch']))
cleanrows=readcsv(ROOT/'E2026/outputs/final/q2/checkpoint_selection_fairness/q2_cleanselect_fair_comparison_seedwise.csv')
for r in cleanrows:
    check('cleanselect_checkpoint_'+r['model']+'_'+r['seed'],sha(ROOT/'E2026'/r['checkpoint_path'])==r['checkpoint_sha256'])
bundle=readcsv(ROOT/'E2026/outputs/final/q2/public_baselines_expanded/checkpoints/checkpoint_manifest.csv')
check('public_comparison_B0_P2_use_cleanselect_hashes',all(next(r for r in bundle if r['model']==x['model'] and r['seed']==x['seed'])['sha256']==x['checkpoint_sha256'] for x in cleanrows))
pdf=PdfReader(O/'q2_final_refine.pdf')
pages=[p.extract_text() or '' for p in pdf.pages]
(O/'qa/pdf_text.txt').write_text('\n\n'.join(pages),encoding='utf-8')
report=dict(checks=checks,passed=all(checks.values()),pages=len(pages),equations=len(eqs),figure_numbers=figure_numbers,table_numbers=table_numbers,overfull=log.count('Overfull'),underfull=log.count('Underfull'),locked_checkpoints=lockrows,cleanselect_checkpoints=cleanrows)
(O/'qa/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(pages=len(pages),checks=len(checks),failed=[n for n,v in checks.items() if not v],overfull=report['overfull']),ensure_ascii=False))
assert report['passed']
