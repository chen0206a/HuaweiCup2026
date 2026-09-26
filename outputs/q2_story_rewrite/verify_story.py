"""Read-only checks of source preservation and existing result tables."""
from pathlib import Path
import csv, hashlib, json, re
from pypdf import PdfReader

O = Path(__file__).resolve().parent
ROOT = O.parent.parent
SRC = Path('D:/java录屏/问题二_最全文字扩写版.tex')
source = SRC.read_text(encoding='utf-8')
tex = (O/'q2_story_rewrite.tex').read_text(encoding='utf-8')
generated = {p.stem: p.read_text(encoding='utf-8') for p in (O/'generated').glob('*.tex')}
expanded = tex
for name, content in generated.items():
    expanded = expanded.replace(r'\input{generated/'+name+'.tex}', content)
checks = {}
def check(name, passed):
    checks[name] = bool(passed)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def env(text, name):
    return re.findall(r'\\begin\{'+name+r'\}.*?\\end\{'+name+r'\}', text, re.S)
check('original_tex_and_pdf_unchanged', all(
    sha(p) == sha(O/'source'/p.name) for p in [SRC, SRC.with_suffix('.pdf')]))
check('core_equations_1_to_19_byte_identical',
      env(expanded,'equation') == env(source,'equation')[:19])
source_tables = {re.search(r'\\label\{([^}]+)\}',t)[1]: t
                 for t in env(source,'table') if re.search(r'\\label\{tab:q2_',t)}
rewritten_tables = {re.search(r'\\label\{([^}]+)\}',t)[1]: t
                    for t in env(expanded,'table') if re.search(r'\\label\{tab:q2_',t)}
check('original_result_table_cells_byte_identical',
      len(source_tables)==4 and all(env(t,'tabular')==env(rewritten_tables[k],'tabular')
                                  for k,t in source_tables.items()))
check('attachment3_longtable_byte_identical',
      env(expanded,'longtable') == env(source,'longtable'))
check('original_metric_formulas_preserved',
      env(source,'itemize')[0] in expanded)
check('bibliography_byte_identical',
      env(expanded,'thebibliography') == env(source,'thebibliography'))
check('seven_figure_assets_byte_identical', all(
    sha(O/e['local_file']) == e['sha256'] == sha(Path(e['source']))
    for e in json.loads((O/'qa/figure_manifest.json').read_text(encoding='utf-8'))))
protected = json.loads((O/'qa/protected_paper_hashes.json').read_text(encoding='utf-8'))
check('formal_paper_snapshot_unchanged', all(
    Path(p).is_file() and sha(Path(p)) == h for p,h in protected.items()))
labels = re.findall(r'\\label\{([^}]+)\}', expanded)
refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', expanded)
check('all_references_resolve', set(refs) <= set(labels))
check('no_duplicate_labels', len(labels) == len(set(labels)))
check('eleven_subsections', len(re.findall(r'\\subsection\{',tex)) == 11)
check('explicit_2em_paragraph_macro_defined',
      r'\newcommand{\QTwoParagraph}{\par\noindent\hspace*{2em}}' in tex)
check('no_hardcoded_figure_or_table_references',
      not re.search(r'[图表]\s*[0-9]',tex))
aux = (O/'build/q2_story_rewrite.aux').read_text(encoding='utf-8')
figure_numbers = [int(x) for x in re.findall(r'\\newlabel\{fig:q2_[^@}]+\}\{\{(\d+)\}',aux)]
table_numbers = [int(x) for x in re.findall(r'\\newlabel\{tab:q2_[^@}]+\}\{\{(\d+)\}',aux)]
check('figure_numbers_8_to_14', figure_numbers == list(range(8,15)))
check('table_numbers_8_to_13', table_numbers == list(range(8,14)))
log = (O/'build/q2_story_rewrite.log').read_text(encoding='utf-8',errors='replace')
check('no_overfull_boxes', 'Overfull' not in log)
check('no_undefined_references_or_citations', not re.search(r'undefined|Undefined',log))
check('no_missing_glyphs_or_latex_errors', 'Missing character:' not in log and '\n!' not in log)

def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
metric_names = ['accuracy','macro_f1','mae','pearson']
comparison=[]
for variant in ['baselines','missing']:
    summary_path=ROOT/'E2026/outputs/final/q2/public_baselines_expanded'/(
        'q2_public_baselines_'+('clean' if variant=='baselines' else 'missing')+'_summary.csv')
    data={r['model']:r for r in readcsv(summary_path)}
    for line in generated['table_q2_public_'+variant].splitlines():
        vals=re.findall(r'\$(\d+\.\d+)\\pm(\d+\.\d+)\$',line)
        if not vals: continue
        model=line.split('&')[0].strip()
        row=data['P2' if model=='LTARP' else model]
        ok=all(a==f"{float(row[m+'_mean']):.4f}" and b==f"{float(row[m+'_sample_sd']):.4f}"
               for m,(a,b) in zip(metric_names,vals))
        check('public_'+variant+'_'+model+'_four_metrics_match_csv',ok and len(vals)==4)
        comparison.append({'table':variant,'model':model,'source':str(summary_path),
                           'four_means_and_sample_sds_match':ok})
fair_path=ROOT/'E2026/outputs/final/q2/checkpoint_selection_fairness/q2_cleanselect_fair_comparison_summary.csv'
fair={r['model']:r for r in readcsv(fair_path)}
for line in generated['table_q2_main'].splitlines():
    vals=re.findall(r'\$(\d+\.\d+)\\pm(\d+\.\d+)\$',line)
    if not vals: continue
    cond,model=[x.strip() for x in line.split('&')[:2]]
    prefix='clean_' if cond=='完整输入' else 'mean_missing_'
    row=fair['B0' if model=='MMP' else 'P2']
    check('main_'+cond+'_'+model+'_matches_fair_summary',
          len(vals)==4 and all(a==f"{float(row[prefix+m+'_mean']):.4f}" and
                               b==f"{float(row[prefix+m+'_sample_sd']):.4f}"
                               for m,(a,b) in zip(metric_names,vals)))
prediction_path=ROOT/'E2026/outputs/final/q2/attachment3/attachment3_predictions.csv'
predictions=readcsv(prediction_path)
rows=re.findall(r'\\detokenize\{(附件3_\d+)\}\} & (消极|中性|积极) & ([+-]?\d+\.\d+) & (\d+\.\d+)',expanded)
check('attachment3_exactly_30_unique_prediction_rows',len(rows)==30 and len({r[0] for r in rows})==30)
byid={r['sample_id']:r for r in predictions}
class_names={'Negative':'消极','Neutral':'中性','Positive':'积极'}
check('attachment3_classes_and_intensities_match_csv',
      all(c==class_names[byid[i]['predicted_class_name']] and
          abs(float(v)-float(byid[i]['predicted_intensity']))<=0.00000051 for i,c,v,_ in rows))
pdf=PdfReader(O/'q2_story_rewrite.pdf')
report={'checks':checks,'all_preservation_checks_pass':all(checks.values()),
        'pages':len(pdf.pages),'core_equations':len(env(expanded,'equation')),
        'figure_numbers':figure_numbers,'table_numbers':table_numbers,
        'original_prediction_rows':len(rows),
        'overfull_count':log.count('Overfull'),'underfull_count':log.count('Underfull'),
        'public_table_comparison':comparison,
        'note':'Existing source/code/citation conflicts are separately recorded in q2_source_issues.md; passing preservation checks does not resolve them.'}
(O/'qa/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'qa/pdf_text.txt').write_text('\n\n'.join(p.extract_text() or '' for p in pdf.pages),encoding='utf-8')
print(json.dumps({'pages':report['pages'],'checks':len(checks),'passed':sum(checks.values()),
                  'failed':[n for n,p in checks.items() if not p],'underfull':report['underfull_count']},ensure_ascii=False))
assert all(checks.values()), 'See qa/verification.json'
