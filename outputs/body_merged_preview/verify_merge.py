from pathlib import Path
import re,json,hashlib
from pypdf import PdfReader

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,result):
    checks.append({'check':name,'pass':bool(result)})
    assert result,name
def text(p):return p.read_text(encoding='utf-8')
manifest=json.loads(text(OUT/'qa/input_manifest.json'))
check('所有冻结输入文件哈希未变化',all(sha(ROOT/x['path'])==x['sha256'] for x in manifest))
source={1:ROOT/'outputs/q1_final_refine',2:ROOT/'outputs/q2_point_final',3:ROOT/'outputs/q3_point_final'}
names={1:'q1_final_refine',2:'q2_point_final',3:'q3_point_final'}
def canonical(t):
    t=t.replace('{{q2 架构图}.pdf}','{q2 架构图.pdf}')
    t=re.sub(r'%[^\n]*','',t)
    t=re.sub(r'\\setcounter\{(?:section|figure|table|equation)\}\{[^}]+\}','',t)
    t=re.sub(r'^\\songti\\zihao\{-4\}(?:\\pagestyle\{plain\})?\s*\n','',t,flags=re.M)
    # Graphic paths and punctuation braces are integration-only changes.
    t=re.sub(r'\\includegraphics(\[[^\]]*\])\{(?:\{)?(?:figures/q[123]/)?([^{}]+?)(?:\})?\}',r'\\includegraphics\1{\2}',t)
    return re.sub(r'\s+','',t)
for q in [1,2,3]:
    s=source[q]
    if q==1: original=text(s/'sections/05_q1.tex')
    else:
        original=text(s/(names[q]+'.tex')).split('\\begin{document}',1)[1].split('\\end{document}',1)[0]
        if q==2:original=original.split('\\clearpage\n{\\small\n\\begin{thebibliography}',1)[0]
    merged=text(OUT/'sections'/f'q{q}.tex')
    check(f'Q{q}正文与数字不改写',canonical(original)==canonical(merged))
    eq=lambda t:re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',t,re.S)
    check(f'Q{q}所有公式逐字不变',eq(original)==eq(merged))
    check(f'Q{q}标题顺序不变',re.findall(r'\\(?:sub)*section\{[^}]+\}',original)==re.findall(r'\\(?:sub)*section\{[^}]+\}',merged))
    check(f'Q{q}全部图件哈希相同',all(sha(p)==sha(OUT/'figures'/f'q{q}'/p.name) for p in (s/'figures'/f'q{q}').glob('*.pdf')))
    for p in (s/'generated').glob('*.tex'):
        a=text(p);b=text(OUT/'generated'/p.name)
        if p.name=='table_q1_all_samples.tex':b=b.replace('续表~\\ref{tab:q1_all_samples}（接上页）','续表2（接上页）').replace('font=small,labelfont=normalfont','font=small,labelfont=bf')
        if p.name=='table_q2_public_overview.tex':b=b.replace('p{.15\\textwidth}','p{.13\\textwidth}')
        check(f'{p.name}内容不变',a==b)
    built=s/'build'/(names[q]+'.pdf')
    if built.exists():check(f'Q{q}输入PDF与该终稿编译目录PDF一致',sha(built)==sha(s/(names[q]+'.pdf')))

aux=text(OUT/'build/body_merged_preview.aux')
labels={m[1]:(m[2],int(m[3])) for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux) if '@cref' not in m[1]}
expected={1:([1,7],[1,7],[1,10]),2:([8,14],[8,13],[11,28]),3:([15,17],[14,20],[29,32])}
for q in [1,2,3]:
    check(f'Q{q}节号={q+2}',labels[f'sec:q{q}'][0]==str(q+2))
    for i,prefix in enumerate(['fig:','tab:','eq:']):
        nums=[int(v[0]) for k,v in labels.items() if k.startswith(prefix) and (f'q{q}_' in k or f'q{q}:' in k)]
        lo,hi=expected[q][i]
        if prefix=='eq:':
            count=len(re.findall(r'\\begin\{equation\}',text(OUT/'sections'/f'q{q}.tex')))
            check(f'Q{q}公式数量及自动编号范围{lo}–{hi}',count==hi-lo+1 and all(lo<=n<=hi for n in nums) and len(set(nums))==len(nums))
        else:
            check(f'Q{q} {prefix}编号连续{lo}–{hi}',sorted(nums)==list(range(lo,hi+1)))
check('图8为LTARP结构图',labels['fig:q2_arch'][0]=='8' and 'LTARP三模态池化与双任务预测流程' in text(OUT/'sections/q2.tex'))
check('图15为Q3首图',labels['fig:fig14_q3_heaf_framework'][0]=='15')
check('表14为Q3首表',labels['tab:q3_interaction'][0]=='14')
alltex='\n'.join(text(p) for p in list((OUT/'sections').glob('*.tex'))+list((OUT/'generated').glob('*.tex')))
check('分问题文件无编号重置',not re.search(r'\\setcounter\{(?:section|figure|table|equation)\}',alltex))
check('无硬编码图表及公式引用',not re.search(r'[图表]\s*\d+|式\s*[（(]\s*\d+',re.sub(r'%[^\n]*','',alltex)))
check('交叉引用全部有目标',all(k in labels for k in re.findall(r'\\(?:ref|eqref|cref|Cref)\{([^}]+)\}',alltex)))
log=text(OUT/'build/body_merged_preview.log')
check('无undefined/multiply-defined/overfull/缺字',not re.search(r'(?:Reference|Citation).*undefined|There were undefined|multiply.defined|Overfull|Missing character',log))
refs=text(OUT/'references_working.tex')
order=json.loads(text(OUT/'qa/reference_order.json'))
check('16条工作版文献按首次引用排序',re.findall(r'\\bibitem\{([^}]+)\}',refs)==order and len(order)==16)
check('正文无独立参考文献页',not re.search(r'\\(?:begin\{thebibliography\}|bibliography\{)',alltex))
reader=PdfReader(OUT/'body_merged_preview.pdf');pages=[p.extract_text() or '' for p in reader.pages]
check('无空白整页',all(len(p.strip())>30 for p in pages))
# All caption page ranges stay inside their corresponding question section.
starts=[labels[f'sec:q{q}'][1] for q in [1,2,3]]
for q in [1,2,3]:
    lo=starts[q-1]; hi=starts[q] if q<3 else len(pages)+1
    check(f'Q{q}图表不跨到下一问',all(lo<=v[1]<hi for k,v in labels.items() if k.startswith(('fig:','tab:')) and (f'q{q}_' in k or f'q{q}:' in k)))
protected=json.loads(text(ROOT/'outputs/q1_final_refine/protected_paper_hashes.json'))
main=[(Path(p),h) for p,h in protected.items() if p.endswith('main.tex')]
check('正式main.tex保持原哈希',all(sha(p)==h for p,h in main))
(OUT/'qa/pdf_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{p}' for i,p in enumerate(pages)),encoding='utf-8')
(OUT/'qa/checks.json').write_text(json.dumps({'pages':len(pages),'section_pages':starts,'labels':labels,'checks':checks},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'{len(checks)} checks passed; {len(pages)} pages; section starts {starts}')
