"""Create a standalone Q2 refinement from the locked story preview."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess

O=Path(__file__).resolve().parent
ROOT=O.parent.parent
S=ROOT/'outputs/q2_story_rewrite'
for name in ['build','generated','figures/q2','qa']: (O/name).mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if not (O/'qa/protected_files.json').exists():
    paths=[p for p in S.rglob('*') if p.is_file() and 'build' not in p.relative_to(S).parts
           and not p.name.startswith('page-')]
    paper=ROOT/'论文润色工作区/E2026_论文整理包/01_LaTeX论文工程/paper'
    paths += [p for p in paper.rglob('*') if p.is_file() and
              not any(x.startswith('build') for x in p.relative_to(paper).parts)]
    paths += [ROOT/'E2026/src/models/baseline.py', ROOT/'E2026/src/models/pooling_residual.py',
              ROOT/'E2026/src/models/public_baselines.py', ROOT/'E2026/src/models/public_baselines_extended.py',
              ROOT/'E2026/scripts/run_public_baseline_extended.py', ROOT/'E2026/src/data/attachment3_inference.py',
              ROOT/'E2026/src/q3/text_feature_reconstruction.py', ROOT/'E2026/scripts/run_attachment3_final_inference.py']
    (O/'qa/protected_files.json').write_text(json.dumps({str(p):sha(p) for p in paths},ensure_ascii=False,indent=2),encoding='utf-8')
for p in (S/'figures/q2').glob('*.pdf'): shutil.copy2(p,O/'figures/q2'/p.name)
for p in (S/'generated').glob('*.tex'): shutil.copy2(p,O/'generated'/p.name)
t=(S/'q2_story_rewrite.tex').read_text(encoding='utf-8')
def replace(old,new):
    global t
    if t.count(old)!=1: raise ValueError('Expected exactly one occurrence: '+old[:60])
    t=t.replace(old,new)
def paragraph(prefix,new):
    global t
    pattern=r'\\QTwoParagraph '+re.escape(prefix)+r'[^\n]*'
    t,n=re.subn(pattern,lambda _:r'\QTwoParagraph '+new,t)
    if n!=1: raise ValueError('Paragraph not unique: '+prefix)

# Documentation corrections reproduce the existing locked implementation.
replace(r'\mathcal L_{\mathrm{WCE}} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=0}^2 w_k \cdot \mathbb I(c_i = k) \log \hat p_{ik}.',
        r'\mathcal L_{\mathrm{WCE}} = -\frac{\sum_{i=1}^N w_{c_i}\log\hat p_{i c_i}}{\sum_{i=1}^N w_{c_i}}.')
replace('回归采用SmoothL1损失：','其中分类损失按当前批次标签权重之和归一化，$N$为批次样本数。回归采用SmoothL1损失：')
replace(r'e_{it}^{(m)} = f_m(\mathbf x_{it}^{(m)}) = \mathbf w_{m,2}^\top \tanh(\mathbf W_{m,1} \mathbf x_{it}^{(m)} + \mathbf b_{m,1}) + b_{m,2}.',
        r'e_{it}^{(m)} = f_m(\mathbf x_{it}^{(m)}) = \mathbf w_m^\top\mathbf x_{it}^{(m)} + b_m.')
replace('模态$m$的打分记为','每个模态使用一个线性标量打分器，其打分记为')
replace('掩码均值池化（MMP）','掩码均值池化（Masked Mean Pooling，MMP）')
replace('轻量时间注意力残差池化（LTARP）','轻量时间注意力残差池化（Lightweight Temporal Attention Residual Pooling，LTARP）')
replace(r'位置$\mathcal P=\{\mathrm{front},\mathrm{middle},\mathrm{rear}\}$',
        r'位置$\mathcal P=\{\text{前段},\text{中段},\text{后段}\}$')
replace('目标比例$\\mathcal R=', '目标缺失比例$\\rho_{\\mathrm{target}}\\in\\mathcal R=')
replace('前、中、后段的起点','前段、中段、后段的起点')
replace('其余模态和填充掩码保持原样。每个场景', '该遮挡区间采用零起始槽位索引；其余模态和填充掩码保持原样。每个场景')

replace(r'完整输入对应$S_{\mathrm{clean}}$；跨场景得分$R$对完整输入及54场景平均等权组合：',
        r'''连续情感强度标签位于$[-3,3]$，跨度为6，因此以$\operatorname{MAE}/6$缩放误差量级，并通过$1-\operatorname{MAE}/6$统一为越大越好的方向。Pearson的取值范围为$[-1,1]$，$(\operatorname{Pearson}+1)/2$将其映射至$[0,1]$，故式~\eqref{eq:q2_score_s}中的最后一项等价于给该映射值赋予$1/4$权重。两项分类指标合计占$1/2$，两项回归指标合计占$1/2$，兼顾极性识别与连续强度预测。回归输出不作截断，6作为标签尺度使用，而非预测误差的上界。

\QTwoParagraph 本文将完整输入对应的得分记为$S_{\mathrm{clean}}$，再定义跨场景综合鲁棒得分$R$：''')
replace(r'\end{equation}'+'\n'+r'\QTwoParagraph 结构比较表',
        r'\end{equation}'+'\n'+r'完整输入得分占$1/2$，54种缺失场景的等权平均占$1/2$。这一组合同时考虑正常输入与受损输入下的表现，使配置选择兼顾完整输入能力和缺失后的保持程度。'+'\n\n'+r'\QTwoParagraph 结构比较表')
paragraph('结构比较表',r'''结构比较表~\ref{tab:q2_main}以及公开方法表~\ref{tab:q2_public_baselines}、\ref{tab:q2_public_missing}，均按完整输入验证得分$S_{\mathrm{clean}}$选择参数；选定后同时评价完整输入与全部54种缺失场景，保证两张公开方法结果表使用同一套参数。缺失敏感性与鲁棒诊断图~\ref{fig:q2_ratio}--\ref{fig:q2_error}则沿用按$R$选定的LTARP配置，MMP仍使用对应初始化的完整输入最优基线。前者比较统一完整输入标准下的结构表现，后者分析最终配置在固定缺失场景集合中的响应与波动；每次初始化均固定一套参数，不按缺失模态、比例或位置重新选择。''')
paragraph('在统一完整输入选模标准下',r'''在统一完整输入选模标准下，LTARP在宏平均F1、MAE和Pearson的三种子均值上均优于MMP，准确率变化较小。对相同模型参数施加54种缺失后，三项指标的平均改善仍保留，准确率则略有下降（表~\ref{tab:q2_main}）。MAE与Pearson的改善分别对应连续强度绝对误差的减少和预测趋势的一致性提高。

\QTwoParagraph 验证集中积极样本最多，准确率按样本汇总，受多数类别识别情况的影响较大；宏平均F1对三类等权，能够反映整体准确率之外的类别差异。当前改善更多体现在宏平均F1，说明评价两种池化结构时需要同时观察总体识别和类别均衡表现。中性类的混淆仍然明显，其具体错误分布在后续分析中结合混淆矩阵观察。''')
paragraph('缺失长度相同',r'''固定目标缺失比例$\rho_{\mathrm{target}}=0.3$后，缺失位置仍会改变模型表现。文本前段与后段遮挡对宏平均F1的影响大于中段；包含文本的双模态缺失也呈相近规律，语音、视觉及语音--视觉组合的影响较小。分类与回归的位置响应并不完全一致，例如文本中段遮挡降低宏平均F1，却使MAE略有下降（图~\ref{fig:q2_location}）。''')
paragraph('模态与位置的平均规律之外',r'''前述结果呈现了不同缺失场景的平均变化，随机初始化、类别混淆和特殊输入则反映这些变化在具体条件下的表现。三次初始化的配对收益较小，进一步结合完整输入分类错误与原生视觉全零样本，可以观察总体平均之外的差异（图~\ref{fig:q2_error}）。''')
paragraph('验证集另有15条',r'''验证集另有15条原生视觉特征整段全零样本。该子集中，部分初始化的LTARP分类表现低于MMP，呈现与总体平均趋势不同的结果；样本量较少，其表现结合三次初始化单独观察。''')
old=re.search(r'\\QTwoParagraph 均值--最大值残差池化保留均值分支.*?支持采用轻量内容加权补充的选择。',t,re.S)[0]
replace(old,r'''\QTwoParagraph 均值--最大值残差池化沿用式~\eqref{eq:q2_mean_pool}的掩码均值表示，并增加有效槽位上的逐特征维最大值，再经残差组合接入原有网络。其得分接近LTARP，但仅保留一次初始化结果，主要作为结构探索参考，不与三随机种子结果作精细统计比较。LTARP的三种子综合均值在本组最高，同时只增加883个参数，在本组候选中提供了较好的综合表现。''')
paragraph('常规融合路线',r'''这些方法分别通过张量交互、时序记忆、表征约束或辅助任务组织多模态信息，缺失增强方法还引入重构、表征对齐、元采样及潜在表示推断。比较同时考察完整输入预测能力、连续缺失后的表现和参数规模：前两项反映同一配置在不同输入条件下的结果，参数量补充表示模型复杂度。各方法接入统一特征与双任务接口，以下结果均限定在本文统一适配与评价设置下。''')
paragraph('参数规模进一步',r'''参数规模进一步体现性能与复杂度的折中。LTARP总参数量为164343，约16.4万，明显小于TFN、MulT和MISA等结构，与MMIM和MissModal处于相近量级。在表~\ref{tab:q2_public_baselines}--\ref{tab:q2_public_missing}所列模型中，其完整输入和缺失平均宏平均F1均保持较高水平。图~\ref{fig:q2_params}分别对照这两类输入条件；不同方法的参数规模跨越一个数量级以上，横轴采用对数尺度，使轻量结构和较大融合网络能够在同一坐标中比较。这组结果体现了LTARP在当前接口下的性能与规模平衡，结构复杂度本身不足以决定预测表现。''')
paragraph('完成模型选择后',r'''完成模型选择后，将LTARP用于附件3的30条无标签样本。对齐版提供\texttt{text\_bert}词元输入，以及\texttt{audio}和\texttt{vision}数值特征。单样本的\texttt{text\_bert}形状为$3\times50$，三行依次为词元编号、有效位掩码和词元类型编号，并非模型所需的$50\times768$连续文本表示。因此，使用已验证的\texttt{bert-base-uncased}接口编码有效词元前缀，取最后隐层逐词元表示，再在填充后缀补零至$50\times768$；语音、视觉特征及有效位掩码保持原样。''')
paragraph('接口检查沿用20条',r'''已有接口检查使用附件4中20条同时包含\texttt{text\_bert}和预计算\texttt{text}的无标签样本，逐行比较全部604个有效槽位；604/604个重构行的最大余弦相似度位置均为同索引，特征最大绝对差为$3.0041\times10^{-5}$、RMSE为$8.3977\times10^{-7}$。输入同一固定参数模型后，分类logits最大绝对差为$2.44\times10^{-6}$，回归为$1.12\times10^{-6}$。该数值检查支持词元编码与既有文本特征接口的衔接，附件3随后沿此固定路径完成推理。''')
replace('逐样本类别、强度和置信度见表', '类别与连续情感强度由两个输出头独立预测，推理阶段不使用回归值符号对分类结果进行二次修正。逐样本类别、连续情感强度和置信度见表')
paragraph('文本局部缺失的影响最明显',r'''54种缺失场景中，文本局部缺失的影响最明显，分类与回归对位置的响应有所不同。LTARP在宏平均F1、MAE和Pearson上总体优于MMP，但提升幅度较小，不同随机初始化下仍存在一定波动；公开方法比较中保持了较好的性能--规模折中，并完成附件3全部30条样本的预测。下一问固定该预测模型，进一步分析三模态及局部输入对预测结果的作用。''')
# Chinese terminology in prose/captions; math operator names and existing assets unchanged.
t=t.replace('宏平均F1','宏平均 F1')
t=t.replace('每次初始化均固定一套参数，不按缺失模态', '在每类实验内，每次初始化均固定一套参数，不按缺失模态')
t=t.replace('情感类别与强度','情感类别与连续情感强度').replace('连续强度','连续情感强度')
t=t.replace('双任务预测流程。','双任务预测流程。')

overview=r'''\begin{table}[H]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{公开方法的统一接口适配概览}\label{tab:q2_public_overview}
\begin{tabularx}{.98\textwidth}{@{}>{\raggedright\arraybackslash}p{.13\textwidth}>{\raggedright\arraybackslash}p{.17\textwidth}>{\raggedright\arraybackslash}X r@{}}
\toprule
方法 & 类型 & 本文适配中的核心机制 & 参数量\\
\midrule
TFN\cite{q2tfn} & 张量融合 & 三路表示增广后的外积融合 & 4,840,670\\
LMF\cite{q2lmf} & 低秩融合 & 模态专属低秩因子融合 & 323,276\\
MFN\cite{q2mfn} & 时序记忆融合 & 三路递归编码、记忆注意力与门控记忆 & 261,764\\
MulT\cite{q2mult} & 跨模态注意力 & 六路定向跨模态注意力与模态内记忆 & 624,094\\
MISA\cite{q2misa} & 表征解耦 & 共享/特有表示、重构与差异约束 & 1,118,852\\
Self-MM\cite{q2selfmm} & 多任务辅助学习 & 融合预测与单模态伪目标辅助监督 & 82,119\\
MMIM\cite{q2mmim} & 互信息约束 & 模态间及融合--单模态的InfoNCE约束 & 163,204\\
\addlinespace
TFR-Net\cite{q2tfr} & 缺失重构 & 连续遮挡、时序上下文与潜在表示重构 & 415,620\\
MissModal\cite{q2missmodal} & 缺失表征对齐 & 几何对比、分布与情感语义对齐 & 163,204\\
M3S\cite{q2m3s} & 缺失元采样 & 低秩骨干、完整支持/缺失查询的一步一阶元更新 & 323,276\\
MMIN\cite{q2mmin} & 缺失模态推断 & 潜在表示推断、残差细化与循环约束 & 292,164\\
\bottomrule
\end{tabularx}
\end{table}
'''
(O/'generated/table_q2_public_overview.tex').write_text(overview,encoding='utf-8')
references=json.loads((O/'qa/references_verified.json').read_text(encoding='utf-8'))
bib='\\clearpage\n{\\small\n\\begin{thebibliography}{99}\n'+'\n'.join(r['tex'] for r in references)+'\n\\end{thebibliography}\n}'
t=re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',lambda _:bib,t,flags=re.S)
(O/'q2_final_refine.tex').write_text(t,encoding='utf-8')
for i in range(3):
    r=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q2_final_refine.tex'],
                     cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (O/'build'/f'compile_{i}.txt').write_bytes(r.stdout)
    if r.returncode: raise RuntimeError(r.stdout.decode('utf-8',errors='replace')[-5000:])
shutil.copy2(O/'build/q2_final_refine.pdf',O/'q2_final_refine.pdf')
print('Compiled Q2 final refinement.')
