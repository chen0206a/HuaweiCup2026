from pathlib import Path
from collections import Counter
import re,json,hashlib,logging
from pypdf import PdfReader
from pypdf.generic import NameObject,DictionaryObject

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];SRC=ROOT/'outputs/body_merged_preview'
read=lambda p:p.read_text(encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,condition):
    checks.append({'check':name,'pass':bool(condition)})
    assert condition,name
def remove_command(t,cmd,n):
    pattern=re.compile(re.escape(cmd)+r'\s*')
    while True:
        m=pattern.search(t)
        if not m:break
        pos=m.end()
        for _ in range(n):
            while pos<len(t) and t[pos].isspace():pos+=1
            assert t[pos]=='{',(cmd,t[pos:pos+40])
            level=1;pos+=1
            while level:
                if t[pos]=='{':level+=1
                elif t[pos]=='}':level-=1
                pos+=1
        t=t[:m.start()]+t[pos:]
    return t
def canonical(t):
    t=re.sub(r'%[^\n]*','',t)
    for cmd,n in [(r'\setlength',2),(r'\renewcommand',2),(r'\captionsetup',1),(r'\zihao',1)]:t=remove_command(t,cmd,n)
    # Remove table layout arguments, preserving all caption/header/data payload.
    for cmd,n in [(r'\begin{tabular}',1),(r'\begin{tabularx}',2),(r'\begin{longtable}',1),(r'\multirow',2)]:t=remove_command(t,cmd,n)
    t=re.sub(r'\\end\{(?:tabular|tabularx|longtable)\}','',t)
    t=re.sub(r'\\(?:small|footnotesize|scriptsize|tiny|heiti|songti|shortstack|QOneParagraph|QTwoParagraph|QThreeParagraph)(?![A-Za-z])','',t)
    # Shortstack line breaks and grouping affect only presentation.
    return re.sub(r'[\s{}$]|\\\\','',t)

manifest=json.loads(read(OUT/'qa/input_manifest.json'))
check('合并稿输入文件未变化',all(sha(ROOT/x['path'])==x['sha256'] for x in manifest))
for folder in ['sections','generated']:
    for p in (SRC/folder).glob('*.tex'):
        check(f'{folder}/{p.name}文字数字不变',canonical(read(p))==canonical(read(OUT/folder/p.name)))
check('16条参考文献元数据不变',canonical(read(SRC/'references_working.tex'))==canonical(read(OUT/'references_working.tex')))
for q in [1,2,3]:
    s=read(SRC/'sections'/f'q{q}.tex');t=read(OUT/'sections'/f'q{q}.tex')
    eq=lambda x:re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',x,re.S)
    check(f'Q{q}公式内容不变',eq(s)==eq(t))
    check(f'Q{q}所有标题与顺序不变',re.findall(r'\\(?:sub)*section\{[^}]+\}',s)==re.findall(r'\\(?:sub)*section\{[^}]+\}',t))
check('17张图件按用户要求原样保留',all(sha(p)==sha(OUT/p.relative_to(SRC)) for p in (SRC/'figures').rglob('*.pdf')))
check('Q1全部100条记录不变',len(re.findall(r'\\detokenize\{',read(OUT/'generated/table_q1_all_samples.tex')))==100)
tex=read(OUT/'body_format_compliant.tex')
check('单倍行距',r'\setstretch{1}' in tex and r'\setstretch{1.12}' not in tex)
check('正文与图表注小四',r'\songti\zihao{-4}' in tex and r'\DeclareCaptionFont{compliant}{\songti\zihao{-4}}' in tex)
check('一级标题四号黑体居中',r'section={format=\centering\heiti\zihao{4}' in tex)
check('二三级标题小四宋体',r'subsection={format=\songti\zihao{-4}' in tex and r'subsubsection={format=\songti\zihao{-4}' in tex)
check('英文正文/无衬线/代码字体均新罗马',all('\\'+cmd+'{Times New Roman}' in tex for cmd in ['setmainfont','setsansfont','setmonofont']))
body='\n'.join(read(p) for p in list((OUT/'sections').glob('*.tex'))+list((OUT/'generated').glob('*.tex')))+read(OUT/'references_working.tex')
check('无小字号或黑体中文命令残留',not re.search(r'\\(?:small|footnotesize|scriptsize|tiny|heiti)(?![A-Za-z])|\\zihao\{5\}|font=small',body))
log=read(OUT/'build/body_format_compliant.log')
check('无overfull/缺字/未定义引用/重复标签',not re.search(r'Overfull|Missing character|(?:Reference|Citation).*undefined|There were undefined|multiply.defined',log))

aux=read(OUT/'build/body_format_compliant.aux')
labels={m[1]:(m[2],int(m[3])) for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux) if '@cref' not in m[1]}
for q,(figs,tabs) in enumerate([((1,7),(1,7)),((8,14),(8,13)),((15,17),(14,20))],1):
    check(f'Q{q}节号正确',labels[f'sec:q{q}'][0]==str(q+2))
    for prefix,rng in [('fig:',figs),('tab:',tabs)]:
        nums=[int(v[0]) for k,v in labels.items() if k.startswith(prefix) and (f'q{q}_' in k or f'q{q}:' in k)]
        check(f'Q{q}{prefix}编号连续',sorted(nums)==list(range(rng[0],rng[1]+1)))

# Audit only LaTeX text: user explicitly retains embedded figure typography.
logging.getLogger('pypdf').setLevel(logging.ERROR)
reader=PdfReader(OUT/'body_format_compliant.pdf');font_counts=Counter();bad=[];pages=[]
for n,p in enumerate(reader.pages,1):
    resources=p['/Resources'];old=resources.get('/XObject');resources[NameObject('/XObject')]=DictionaryObject()
    def visit(t,cm,tm,font,size):
        family=str(font.get('/BaseFont')) if font else ''
        if any('\u4e00'<=c<='\u9fff' for c in t):
            font_counts[(family,round(size,3))]+=len(t.strip())
            valid=('SimSun' in family and abs(size-12)<.01) or ('SimHei' in family and abs(size-14)<.01)
            if not valid:bad.append({'page':n,'text':t.strip(),'font':family,'size':size})
    pages.append(p.extract_text(visitor_text=visit))
    if old is not None:resources[NameObject('/XObject')]=old
check('PDF实测所有排版中文为12pt宋体或一级14pt黑体',not bad)
check('无空白整页',all(len(p.strip())>30 for p in pages))
check('图8仍为LTARP结构图',labels['fig:q2_arch'][0]=='8')
protected=json.loads(read(ROOT/'outputs/q1_final_refine/protected_paper_hashes.json'))
check('正式main.tex未修改',all(sha(Path(p))==h for p,h in protected.items() if p.endswith('main.tex')))
report={'pages':len(reader.pages),'checks':checks,'labels':labels,'latex_chinese_font_counts':[{'font':k[0],'size_bp':k[1],'characters':v} for k,v in font_counts.items()],'noncompliant_latex_chinese':bad,'figure_typography':'unchanged_per_user_request','math_symbols':'original_equations_and_math_font_preserved'}
(OUT/'qa/checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'qa/pdf_text.txt').write_text('\n\n'.join(f'PAGE {n+1}\n{p}' for n,p in enumerate(pages)),encoding='utf-8')
print(f'{len(checks)} checks passed; {len(reader.pages)} pages; PDF Chinese font audit passed.')
