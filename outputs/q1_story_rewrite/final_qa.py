from pathlib import Path
import re, json, hashlib
from pypdf import PdfReader

O = Path(__file__).resolve().parent
OLD = O.parent / 'q1_rewrite_preview_v2'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
protected_counts = {}
for name in ['protected_paper_hashes.json', 'qa/v2_protected_hashes.json']:
    hashes = json.loads((O/name).read_text(encoding='utf-8'))
    assert all(Path(p).is_file() and sha(Path(p)) == h for p,h in hashes.items()), name
    protected_counts[name] = len(hashes)
body = (O/'sections/05_q1.tex').read_text(encoding='utf-8')
oldbody = (OLD/'sections/05_q1.tex').read_text(encoding='utf-8')
equations = lambda s: re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', s, re.S)
assert len(equations(body)) == 10 and equations(body) == equations(oldbody)
figs = {p.name: sha(p) == sha(OLD/'figures/q1'/p.name) for p in (O/'figures/q1').glob('*.pdf')}
assert len(figs) == 7 and all(figs.values())
graphics = lambda s: re.findall(r'\\includegraphics[^\n]*', s)
assert graphics(body) == graphics(oldbody)
assert sha(O/'q1_numbers.tex') == sha(OLD/'q1_numbers.tex')
tables = {}
for name in ['table_q1_feature.tex','table_q1_visual.tex','table_q1_modality.tex','table_q1_case_mapping.tex']:
    tables[name] = sha(O/'generated'/name) == sha(OLD/'generated'/name)
assert all(tables.values())
table = (O/'generated/table_q1_all_samples.tex').read_text(encoding='utf-8')
oldtable = (OLD/'generated/table_q1_all_samples.tex').read_text(encoding='utf-8')
rows = re.findall(r'^\\texttt.*$', table, re.M)
assert len(rows) == 100 and rows == re.findall(r'^\\texttt.*$', oldtable, re.M)
tabular = lambda s: re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', s, re.S)
assert tabular((O/'generated/table_q1_alignment.tex').read_text(encoding='utf-8')) == tabular((OLD/'generated/table_q1_alignment.tex').read_text(encoding='utf-8'))
aux = (O/'build/q1_story_rewrite.aux').read_text(encoding='utf-8')
labels = {k: {'number':n,'page':int(p)} for k,n,p in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}', aux) if not k.endswith('@cref')}
refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', body)
assert all(k in labels for k in refs)
assert all(k in refs for k in labels if k.startswith(('tab:','fig:')))
for term in ['不意味着','不能说明','不等价于','不应理解为','并非','不是赛题','不是为了','而非宣称','不用于证明','为回应赛题','本节回答','评委可据此','Shapley','Q3','审计','PASS','本轮']:
    assert term not in body, term
log = (O/'build/q1_story_rewrite.log').read_text(encoding='utf-8', errors='replace')
assert not any(t in log for t in ['Overfull','Missing character','There were undefined references','There were undefined citations'])
reader = PdfReader(O/'q1_story_rewrite.pdf')
text = '\n'.join(p.extract_text() or '' for p in reader.pages)
(O/'qa/pdf_text.txt').write_text(text, encoding='utf-8')
assert not any(t in text.replace(' ','') for t in ['图8','问题二：','问题三：','官方文本'])
result = {'pdf_pages':len(reader.pages), 'q1_content_pages':11, 'equations_exactly_preserved':10,
          'figures_identical':figs, 'figure_placement_options_identical':True,
          'table2_rows_unchanged':100, 'unchanged_tables':tables,
          'table3_result_body_unchanged':True,'numbers_file_unchanged':True,
          'protected_file_counts':protected_counts, 'figure_count':7,
          'table_count':sum(k.startswith('tab:') for k in labels),
          'references':len(re.findall(r'\\bibcite\{',aux)), 'labels':labels,
          'overfull_count':log.count('Overfull'),'underfull_count':log.count('Underfull'),
          'visual_review':'All 12 pages reviewed: legible figures and tables, repeated table2 headers on pages 7–8, no clipped content; one underfull warning in longtable note layout.',
          'no_new_experiments':True,'formal_paper_unchanged':True}
(O/'qa/final_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['labels','figures_identical']},ensure_ascii=False,indent=2))
