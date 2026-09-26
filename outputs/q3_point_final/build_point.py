from pathlib import Path
import re, shutil, subprocess, json, difflib
import pymupdf
from PIL import Image, ImageDraw
O=Path(__file__).resolve().parent
S=O.parent/'q3_context_final'
for d in ('build','qa','generated','figures/q3'):(O/d).mkdir(parents=True,exist_ok=True)
a=(S/'q3_context_final.tex').read_text(encoding='utf-8'); b=a
edits=[
('这些分析针对固定模型的响应，不作为现实情感形成的因果解释。',''),
('枚举全部组合可比较同一模态在不同输入条件下的作用，连续窗口遮挡则保留区间的连续性，便于随后验证响应并回查输入来源。所选区间通过随机等长窗口对照和删除实验检验，跨种子比较考察初始化变化的影响，最后沿来源记录回查原文片段或未对齐特征行。','所选区间进一步通过随机等长窗口对照和删除实验检验，并结合跨种子比较观察结果稳定性，最后沿来源记录回查原文片段或未对齐特征行。'),
('解释干预一致性与稳定性验证','关键区间的干预验证与稳定性分析'),
('所选窗口本身由较大遮挡效应确定，对照差异包含这一选择优势，因此该实验衡量输出敏感性及干预一致性。','由于所选窗口本身依据遮挡效应确定，该比较主要用于观察这些位置对固定模型输出的敏感程度。上述干预结果描述输入变化与固定模型输出之间的关系，不用于推断现实情感形成的因果机制。'),
('应用于附件4的20条无标签样本。附件4的20条样本预测为消极、中性和积极的数量分别为7、5、8，','应用于附件4的20条无标签样本，其中预测为消极、中性和积极的数量分别为7、5和8，'),
('两类主导模态在17条样本上一致，占85\\%，其余样本体现类别支持与强度变化的差异。','分类主导模态与回归主导模态在17条样本上一致，占85\\%，其余3条样本的两类主导模态不同。'),
('这段身份介绍由模型中的连续位置落实到可阅读的原文，说明中性预测的主要文本支持来自何处；','所选文本区间可进一步回查到上述原文片段，从而明确模型中主要文本支持对应的具体内容。'),
('该结果将视觉主导判断缩小到已有输入记录中的具体位置，便于复查所选特征；','因此，视觉主导样本的关键位置可以进一步对应到已有的未对齐视觉特征记录。')]
for old,new in edits:
    assert b.count(old)==1,old
    b=b.replace(old,new,1)
b=b.replace('\\begin{figure}[H]', '\\newpage % 保留图16在第二页的原有分页\n\\begin{figure}[H]', 1)
(O/'q3_point_final.tex').write_text(b,encoding='utf-8')
checks=[]
def check(k,v):
    assert v,k
    checks.append({'check':k,'pass':True})
for env in ('equation','table','figure'):
    pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    check(env+'不变',re.findall(pat,a,re.S)==re.findall(pat,b,re.S))
def section(t,k):return t.split('\\label{subsec:q3_'+k+'}',1)[1].split('\\subsection{',1)[0]
for k in ('shapley','interaction','temporal','grounding'):check(k+'全文不变',section(a,k)==section(b,k))
check('因果边界仅一次',b.count('因果')==1)
check('版式不变',a.split('\\section{',1)[0]==b.split('\\section{',1)[0])
for folder in ('generated','figures/q3'):
    for p in (S/folder).glob('*'):
        if p.suffix in ('.tex','.pdf'):
            shutil.copy2(p,O/folder/p.name)
            check(p.name+'字节不变',p.read_bytes()==(O/folder/p.name).read_bytes())
(O/'qa/point_diff.patch').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='q3_context_final.tex',tofile='q3_point_final.tex')),encoding='utf-8')
for i in range(3):
    r=subprocess.run([r'D:\Latex\miktex\bin\x64\xelatex.exe','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q3_point_final.tex'],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (O/'build'/f'compile_{i}.txt').write_bytes(r.stdout)
    assert r.returncode==0,r.stdout.decode('utf-8',errors='replace')[-3000:]
shutil.copy2(O/'build/q3_point_final.pdf',O/'q3_point_final.pdf')
log=(O/'build/q3_point_final.log').read_text(encoding='utf-8',errors='replace')
for term in ('Overfull','undefined','Missing character'):check('编译无'+term,term not in log)
pdf=pymupdf.open(O/'q3_point_final.pdf')
for start in range(0,len(pdf),3):
    sheet=Image.new('RGB',(710*min(3,len(pdf)-start),1010),'#ddd')
    for j,i in enumerate(range(start,min(start+3,len(pdf)))):
        p=pdf[i].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False)
        im=Image.frombytes('RGB',[p.width,p.height],p.samples);im.thumbnail((690,976))
        sheet.paste(im,(710*j+(710-im.width)//2,28));ImageDraw.Draw(sheet).text((710*j+12,8),f'Page {i+1}',fill='black')
    sheet.save(O/'qa'/f'sheet-{start//3+1}.png')
(O/'qa/checks.json').write_text(json.dumps({'pages':len(pdf),'checks':checks},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pages':len(pdf),'checks':len(checks)},ensure_ascii=False))
