"""Apply only the five requested Q3 context edits and compile a separate preview."""
from pathlib import Path
import re,shutil,subprocess,json,hashlib,difflib
import pymupdf
from PIL import Image,ImageDraw

O=Path(__file__).resolve().parent
S=O.parent/'q3_story_refine'
for name in ('build','qa','generated','figures/q3'):(O/name).mkdir(parents=True,exist_ok=True)
original=(S/'q3_story_refine.tex').read_text(encoding='utf-8')
text=original
start=text.index('\\QThreeParagraph 问题二已经得到')
end=text.index('\\begin{figure}',start)
overview=r'''\QThreeParagraph 问题二已经完成三模态情感类别与连续强度预测。针对问题三，本问不再调整预测模型，而是固定问题二得到的模型参数，进一步分析不同模态和局部输入对当前预测结果的作用。解释对象是同一条样本在不同输入条件下的输出变化。这些分析针对固定模型的响应，不作为现实情感形成的因果解释。

\QThreeParagraph 由于最终预测由文本、语音和视觉共同产生，仅有类别和连续强度还无法区分各模态的具体作用。本问需要回答：三种模态分别贡献多少，两种模态共同输入时是否产生额外作用，以及影响当前预测的信息位于哪些特征位置。因此，分析依次从模态贡献转向模态交互，再定位局部特征区间。

\QThreeParagraph 整体求解路线见图~\ref{fig:fig14_q3_heaf_framework}。输入为$50\times768$的文本特征、$50\times74$的语音特征和$50\times35$的视觉特征，先枚举8种模态组合计算Shapley贡献，再利用相同组合输出分析模态交互；随后对分类主导模态进行连续窗口遮挡，定位影响当前类别的特征区间。枚举全部组合可比较同一模态在不同输入条件下的作用，连续窗口遮挡则保留区间的连续性，便于随后验证响应并回查输入来源。所选区间通过随机等长窗口对照和删除实验检验，跨种子比较考察初始化变化的影响，最后沿来源记录回查原文片段或未对齐特征行。预测与解释始终使用问题二的双任务模型，分类和回归分别采用各自的解释目标，使两种输出的作用方向都有明确含义。

'''
text=text[:start]+overview+text[end:]
replacements={
 r'\QThreeParagraph 分类解释固定完整输入预测的类别':r'\QThreeParagraph 确定模态贡献的计算方式后，还需要为各模态组合规定一致的比较目标。由于分类与回归的输出形式不同，本文分别定义两类解释目标。分类解释固定完整输入预测的类别',
 r'\subsection{解释干预一致性与稳定性验证}\label{subsec:q3_faithfulness}'+'\n'+r'\subsubsection{实验设计}':r'\subsection{解释干预一致性与稳定性验证}\label{subsec:q3_faithfulness}'+'\n'+r'\QThreeParagraph 前述模态贡献和连续窗口遮挡给出了对当前预测起主要作用的模态及关键特征区间，但这些位置是否对应更明显的模型响应，还需要通过独立验证子集上的干预对照进行检验。'+'\n\n'+r'\subsubsection{实验设计}',
 r'\QThreeParagraph 解释实验使用问题二的固定seed 42模型。':r'\QThreeParagraph 为此，本文固定问题二的seed 42模型。',
 r'\QThreeParagraph 附件4的20条样本预测为消极':r'\QThreeParagraph 完成解释方法及独立验证后，本文将固定的预测与解释流程应用于附件4的20条无标签样本。附件4的20条样本预测为消极',
 r'\QThreeParagraph 图~\ref{fig:fig16_q3_case_explanations}选取文本主导的样本14和视觉主导的样本02，比较不同模态作用及其输入回溯结果。':r'\QThreeParagraph 附件4多数样本由文本主导，但也存在视觉主导的个例。为比较不同模态主导时的贡献和来源回溯结果，图~\ref{fig:fig16_q3_case_explanations}选取文本主导的样本14和视觉主导的样本02进行分析。'}
for old,new in replacements.items():
    assert text.count(old)==1,old
    text=text.replace(old,new,1)
(O/'q3_context_final.tex').write_text(text,encoding='utf-8')
for p in (S/'generated').glob('*.tex'):shutil.copy2(p,O/'generated'/p.name)
for p in (S/'figures/q3').glob('*.pdf'):shutil.copy2(p,O/'figures/q3'/p.name)

checks=[]
def check(name,value):
    checks.append({'check':name,'pass':bool(value)})
    assert value,name
def section(t,k):
    block=t[t.index('\\label{subsec:q3_'+k+'}')+len('\\label{subsec:q3_'+k+'}'):]
    return block.split('\\subsection{',1)[0]
for k in ('shapley','interaction','temporal','grounding','cases'):
    if k=='shapley':
        a=section(original,k).split('\\subsubsection{解释目标与主导模态}',1)[0]
        b=section(text,k).split('\\subsubsection{解释目标与主导模态}',1)[0]
    elif k=='cases':
        a=section(original,k).split('\\QThreeParagraph 样本14',1)[1]
        b=section(text,k).split('\\QThreeParagraph 样本14',1)[1]
    else:a,b=section(original,k),section(text,k)
    check('原样保留 '+k,a==b)
check('仅五处章节改动',all(section(original,k)==section(text,k) for k in ('interaction','temporal','grounding')))
check('3.1三个自然段',section(text,'overview').split('\\begin{figure}',1)[0].count('\\QThreeParagraph')==3)
for env in ('equation','table','figure'):
    check(env+'全部原样保留',re.findall(r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}',original,re.S)==re.findall(r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}',text,re.S))
check('版式设置原样保留',original.split('\\section{',1)[0]==text.split('\\section{',1)[0])
for sub in ('generated','figures/q3'):
    for p in (O/sub).glob('*'):
        check('资源字节不变 '+p.name,p.read_bytes()==(S/sub/p.name).read_bytes())
(O/'qa/context_diff.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='q3_story_refine.tex',tofile='q3_context_final.tex')),encoding='utf-8')
for i in range(3):
    r=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q3_context_final.tex'],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (O/'build'/f'compile_{i}.txt').write_bytes(r.stdout)
    if r.returncode:raise RuntimeError(r.stdout.decode('utf-8',errors='replace')[-4000:])
shutil.copy2(O/'build/q3_context_final.pdf',O/'q3_context_final.pdf')
log=(O/'build/q3_context_final.log').read_text(encoding='utf-8',errors='replace')
for error in ('Overfull','undefined','Missing character'):check('日志无 '+error,error not in log)
pdf=pymupdf.open(O/'q3_context_final.pdf')
check('页数12至13',12<=len(pdf)<=13)
(O/'qa/pdf_text.txt').write_text('\n'.join(p.get_text() for p in pdf),encoding='utf-8')
for i,page in enumerate(pdf):page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False).save(O/'qa'/f'page-{i+1:02}.png')
for start in range(0,len(pdf),3):
    canvas=Image.new('RGB',(710*min(3,len(pdf)-start),1010),'#ddd')
    for j,i in enumerate(range(start,min(start+3,len(pdf)))):
        img=Image.open(O/'qa'/f'page-{i+1:02}.png').convert('RGB');img.thumbnail((690,976))
        canvas.paste(img,(710*j+(710-img.width)//2,28))
        ImageDraw.Draw(canvas).text((710*j+12,8),f'Page {i+1}',fill='black')
    canvas.save(O/'qa'/f'sheet-{start//3+1}.png')
(O/'qa/checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pages':len(pdf),'checks':len(checks),'all_passed':all(x['pass'] for x in checks)},ensure_ascii=False))
