from pathlib import Path
import re,shutil,subprocess,json,hashlib

O=Path(__file__).resolve().parent
ROOT=O.parent.parent
SRC=Path('D:/java录屏/问题二_最全文字扩写版.tex')
PAPER=ROOT/'论文润色工作区/E2026_论文整理包/01_LaTeX论文工程/paper'
O.mkdir(exist_ok=True)
for name in ['source','generated','figures/q2','build','qa','evidence']:
    (O/name).mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
source=SRC.read_text(encoding='utf-8')
for p in [SRC,SRC.with_suffix('.pdf')]: shutil.copy2(p,O/'source'/p.name)
protected={str(p):sha(p) for p in PAPER.rglob('*') if p.is_file() and not any(x.startswith('build') for x in p.relative_to(PAPER).parts)}
if not (O/'qa/protected_paper_hashes.json').exists():
    (O/'qa/protected_paper_hashes.json').write_text(json.dumps(protected,ensure_ascii=False,indent=2),encoding='utf-8')
sections={re.search(r'\\label\{subsec:q2_([^}]+)\}',s)[1]:s for s in re.split(r'(?=\\subsection\{)',source)[1:]}
eqs={f'{k}_{i}':e for k,s in sections.items() for i,e in enumerate(re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',s,re.S))}
figures={re.search(r'\\label\{fig:q2_([^}]+)\}',f)[1]:f for f in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',source,re.S)}
captions={
 'arch':'LTARP三模态池化与双任务预测流程。',
 'data':'附件2的样本、情感标签与有效序列长度分布。',
 'ratio':'单模态缺失比例与验证集性能。阴影为三随机种子的样本标准差，虚线为完整输入均值。',
 'location':'不同模态与位置相对完整输入的指标变化。各指标使用独立色标。',
 'gain':'LTARP相对MMP的配对改善。MAE取MMP减LTARP，其余指标取LTARP减MMP；正值表示改善。',
 'error':'随机初始化稳定性与误差分析。（a）配对综合分数；（b）混淆矩阵；（c）场景分组；（d）原生视觉全零子集。',
 'params':'公开多模态模型的参数规模与宏平均F1比较。（a）完整输入验证集；（b）54种连续缺失场景平均。横轴为总参数量（对数尺度），误差线为三随机种子的样本标准差。'}
manifest=[]
for key,f in list(figures.items()):
    f=re.sub(r'\\caption\{[^\n]*\}',lambda _:r'\caption{'+captions[key]+'}',f)
    if key=='error': f=f.replace(r'\begin{figure}[htbp]',r'\begin{figure}[H]')
    figures[key]=f
    if key=='arch':
        p=PAPER/'figures/q2/fig09_q2_model_architecture.pdf'
        target='q2 架构图.pdf'
    else:
        target=re.search(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',f)[1]
        p=PAPER/'figures/q2'/target
    assert p.is_file(),p
    shutil.copy2(p,O/'figures/q2'/target)
    manifest.append({'label':key,'source':str(p),'local_file':'figures/q2/'+target,'sha256':sha(p)})
(O/'qa/figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
tables={}
for block in re.findall(r'\\begin\{table\}.*?\\end\{table\}',source,re.S):
    label=re.search(r'\\label\{tab:q2_([^}]+)\}',block)
    if label: tables[label[1]]=block
table_captions={
 'main':r'统一完整输入选模规则下的MMP与LTARP比较（三随机种子均值$\pm$样本标准差）',
 'ablation':'候选结构与训练策略消融。均值--最大值残差池化仅有一次初始化。',
 'public_baselines':r'公开方法与LTARP的完整输入验证集结果（三随机种子均值$\pm$样本标准差）',
 'public_missing':r'公开方法与LTARP的54种连续缺失场景平均结果（三随机种子均值$\pm$样本标准差）'}
for k,t in tables.items():
    t=re.sub(r'\\caption\{[^\n]*\}',lambda _:r'\caption{'+table_captions[k]+'}',t)
    t=t.replace(r'\begin{table}[htbp]',r'\begin{table}[H]')
    (O/'generated'/f'table_q2_{k}.tex').write_text(t,encoding='utf-8')
pred=re.search(r'\\begingroup\\small\\setlength\{\\tabcolsep\}\{7pt\}.*?\\endgroup',source,re.S)[0]
(O/'generated/table_q2_attachment3.tex').write_text(pred,encoding='utf-8')
metrics=re.search(r'基础性能采用四项标准指标评价：\n(\\begin\{itemize\}.*?\\end\{itemize\})',source,re.S)[1]
bib=re.search(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',source,re.S)[0]
# The source bibliography is preserved; detected provenance/reference issues are
# recorded separately rather than silently changing this writing-only task.
overview=r'''\begin{table}[H]
\centering\small
\caption{公开方法的统一接口适配概览}\label{tab:q2_public_overview}
\begin{tabularx}{.96\textwidth}{@{}l l X r@{}}
\toprule
方法 & 类型 & 本文适配中的核心机制 & 参数量\\
\midrule
TFN\cite{q2tfn} & 常规融合 & 张量外积融合 & 4,840,670\\
LMF\cite{q2lmf} & 常规融合 & 低秩张量分解 & 323,276\\
MFN\cite{q2mfn} & 时序融合 & 跨模态记忆更新 & 261,764\\
MulT\cite{q2mult} & 时序融合 & 定向跨模态注意力 & 624,094\\
MISA\cite{q2misa} & 表征解耦 & 共享与特有空间约束 & 1,118,852\\
Self-MM\cite{q2selfmm} & 多任务 & 单模态辅助监督 & 82,119\\
MMIM\cite{q2mmim} & 信息约束 & 互信息约束与联合融合 & 163,204\\
\addlinespace
TFR-Net\cite{q2tfr} & 缺失增强 & 连续掩码与表征重构 & 415,620\\
MissModal\cite{q2missmodal} & 缺失增强 & 完整与缺失表征对齐 & 163,204\\
M3S\cite{q2m3s} & 缺失增强 & 缺失采样与元更新 & 323,276\\
MMIN\cite{q2mmin} & 缺失增强 & 潜在缺失推断与循环约束 & 292,164\\
\bottomrule
\end{tabularx}
\end{table}'''
(O/'generated/table_q2_public_overview.tex').write_text(overview,encoding='utf-8')
body=r'''\section{问题二：缺失场景下的多模态情感预测模型}\label{sec:q2}

\subsection{问题分析与总体方案}\label{subsec:q2_plan}
\QTwoParagraph 附件2已经提供对齐到50个时间位置的文本、语音和视觉特征。局部连续缺失发生时，一种或多种模态会在一段时间内失去观测，其影响随模态、位置和持续长度而变化。本问据此研究情感极性分类与连续强度回归：完整输入能保留多少预测信息，连续缺失后性能如何变化，以及能否通过轻量的结构调整改善跨场景表现。

\QTwoParagraph 先以掩码均值池化（MMP）建立基线，分别聚合三模态有效槽位，再拼接融合并输出情感类别与强度。均值池化对所有有效位置赋予相同权重，难以区分剩余内容的信息量。本文在该路径上加入轻量时间注意力残差池化（LTARP），由内容打分形成补充表示，与均值表示共同参与预测，整体流程见图~\ref{fig:q2_arch}。

{{fig:arch}}

\subsection{数据表示与预测任务}\label{subsec:q2_data}
\QTwoParagraph 两种池化方式共用附件2的对齐输入与任务标签。对齐特征文件为\texttt{aligned-50.pkl}，训练、验证和测试划分分别含3395、728和727条样本。训练集用于学习参数，验证集用于结构选择与评价；测试划分和无标签附件3不参与选模。

\QTwoParagraph 统一最大长度为$T=50$，样本$i$的三模态输入为
{{eq:data_0}}
样本有效长度为$L_i$（$3\le L_i\le50$），由随附注意力掩码确定。定义有效槽位指示变量
{{eq:data_1}}
\QTwoParagraph 缺失实验仅作用于原本有效的时间位置。附件2中已有的零值输入保持原始状态，并与后续人为构造的连续缺失分别记录。填充后缀由掩码排除，原始有效范围在遮挡前后保持一致。

\QTwoParagraph 连续情感强度标签记为$y_i\in[-3.0,+3.0]$，三类极性由其符号确定：
{{eq:data_2}}
其中$y_i=0$对应中性类，分类与回归分别描述极性类别和连续强度。

\QTwoParagraph 训练集与验证集均以积极样本为最多，中性样本相对较少，类别比例呈现不均衡；连续强度集中于零附近，两侧分布覆盖不同程度的极性。有效序列长度也有明显差异，训练与验证的中位数分别为22和23.5，小于统一长度50。因此，类别权重和有效位掩码分别进入损失计算与特征聚合（图~\ref{fig:q2_data}）。

{{fig:data}}
\FloatBarrier

\subsection{掩码均值池化基线}\label{subsec:q2_base}
\QTwoParagraph 为单独观察连续缺失对预测的影响，先构造不含复杂时序模块的简单基线。MMP只在有效槽位上计算均值，再完成模态映射、融合和双任务预测。对模态$m\in\{T,A,V\}$，其均值为
{{eq:base_0}}
均值向量通过模态专属映射$P_m(\cdot)$得到128维表示：
{{eq:base_1}}
三路表示拼接为384维向量$\mathbf h_i^{\mathrm{concat}}=[\mathbf h_{i,0}^{(T)};\mathbf h_{i,0}^{(A)};\mathbf h_{i,0}^{(V)}]$，再经128维融合层得到
{{eq:base_2}}
联合表征分别连接三分类与强度回归输出：
{{eq:base_3}}
预测类别取$\hat c_i=\arg\max_{k\in\{0,1,2\}}\hat p_{ik}$。

\QTwoParagraph 训练集的消极、中性和积极样本数分别为967、758和1670。采用类别频数逆加权交叉熵，使三类在分类目标中获得相应权重：
{{eq:base_4}}
回归采用SmoothL1损失：
{{eq:base_5}}
两项任务以相同权重联合优化：
{{eq:base_6}}

\QTwoParagraph MMP共含163460个参数，直接使用附件2的原始数值特征，不额外标准化。采用AdamW，学习率$10^{-3}$、权重衰减$10^{-4}$、批量大小128、随机失活率0.1；最多训练80轮，提前停止耐心轮数为12轮。基线在完整输入上训练。

\subsection{轻量时间注意力残差池化模型}\label{subsec:q2_pool}
\QTwoParagraph MMP结构简单、参数较少，均值聚合为比较缺失影响提供了稳定起点。但连续缺失发生后，剩余位置所含情感信息并不均等。LTARP保留均值路径，增加逐槽位内容打分器，使模型能为剩余内容形成不同权重。模态$m$的打分记为
{{eq:pool_0}}
权重在原始有效槽位内归一化，并形成内容加权表示：
{{eq:pool_1}}
填充位的$\alpha_{it}^{(m)}$恒为0。

\QTwoParagraph 内容加权表示作为均值表示的补充，两路共享模态映射$P_m$，以可学习系数$\gamma_m$进行残差组合：
{{eq:pool_2}}
该表示继续进入原有融合层与双任务输出，新增分支只改变模态内的聚合方式。

\QTwoParagraph 训练首先完成MMP的模态映射、融合与双任务输出学习，再以其参数初始化LTARP。第二阶段保持这些已有参数不变，仅学习三个模态打分器及三个残差系数。令$\gamma_m$初值为0，初始模态输出为
{{eq:pool_3}}
与MMP保持一致。随后，内容加权分支通过残差系数逐步参与预测。新增可训练参数为883个，总参数量增至164343，从而以较小的参数增量补充时间位置的内容差异。

\subsection{连续缺失场景与评价协议}\label{subsec:q2_benchmark}
\QTwoParagraph 两种池化方式的差异需要在相同受损输入上比较。连续缺失主要由三个因素决定：缺失哪种模态、发生在哪个位置，以及持续多长。因此，评价在完整输入之外分别组合模态、比例和位置，形成固定的54种场景。

\begin{itemize}[leftmargin=2.5em,itemsep=0pt,topsep=2pt]
\item 单模态场景集$\mathcal C_1$：$m\in\{T,A,V\}$，目标比例$\mathcal R=\{0.1,0.2,0.3,0.4,0.5\}$，位置$\mathcal P=\{\mathrm{front},\mathrm{middle},\mathrm{rear}\}$，共$3\times5\times3=45$种。
\item 双模态场景集$\mathcal C_2$：模态对为$\{T,A\}$、$\{T,V\}$、$\{A,V\}$，固定$\rho_{\mathrm{target}}=0.3$，分别取三个位置，共$3\times1\times3=9$种。
\end{itemize}

\QTwoParagraph 对有效长度$L_i$，遮挡槽位数与实际比例为
{{eq:benchmark_0}}
槽位数必须取整数，短序列的实际比例会偏离目标比例；验证集中，目标比例0.1的实际范围为0.0714--0.3333，目标比例0.5为0.4000--0.6667，对应均值为0.1040和0.5009。图表横轴用目标比例标识场景，实际比例逐样本记录。

\QTwoParagraph 前、中、后段的起点$s_i$分别为$0$、$\lfloor(L_i-\ell_i)/2\rfloor$和$L_i-\ell_i$。遮挡将指定模态在$[s_i,s_i+\ell_i-1]$内的特征置零，其余模态和填充掩码保持原样。每个场景都在完整的728条验证样本上评价，各样本按自身有效长度计算区间。

\QTwoParagraph 分类评价同时采用准确率与宏平均F1，分别观察总体识别与类别均衡表现；回归采用MAE和皮尔逊相关系数，分别观察绝对误差与变化趋势。四项指标定义为：
{{metrics}}

\QTwoParagraph 将分类与回归表现组合为得分$S$：
{{eq:benchmark_1}}
完整输入对应$S_{\mathrm{clean}}$；跨场景得分$R$对完整输入及54场景平均等权组合：
{{eq:benchmark_2}}
\QTwoParagraph 结构比较表~\ref{tab:q2_main}和公开方法结果表~\ref{tab:q2_public_baselines}、\ref{tab:q2_public_missing}统一按$S_{\mathrm{clean}}$选择模型参数，再评价完整与缺失输入。缺失敏感性分析关注跨场景表现，图~\ref{fig:q2_ratio}--\ref{fig:q2_error}沿用按$R$选择的LTARP参数及对应MMP基线。两组评价各自共用一套已选参数，对所有场景执行相同流程。

\QTwoParagraph 场景统计先在种子内汇总，再计算种子间均值与样本标准差。设第$s$个种子在场景$j$的指标为$M_{s,j}$，其缺失场景均值为
{{eq:benchmark_3}}
三次初始化是统计重复单位，54个场景在各自初始化内等权平均。算法~\ref{alg:q2_benchmark}概括执行步骤。

\begin{table}[htbp]
\centering\small
\begin{tabular}{p{.94\textwidth}}
\toprule
\refstepcounter{q2algorithm}\label{alg:q2_benchmark}\textbf{算法\arabic{q2algorithm}\quad 连续缺失场景评价}\\
\midrule
\textbf{输入}：验证集三模态特征、有效掩码与已选模型。\\
\textbf{输出}：完整输入及54场景指标、$S_{\mathrm{clean}}$与$R$。\\
1. 预测完整输入，计算四项指标及$S_{\mathrm{clean}}$。\\
2. 遍历场景配置$(\mathcal M_{\mathrm{target}},\rho_{\mathrm{target}},\mathrm{pos})$。\\
3. 对各样本读取$L_i$，按式~\eqref{eq:q2_mask_len}计算$\ell_i$并确定$s_i$。\\
4. 在原输入副本中置零目标模态的连续区间，保持有效掩码。\\
5. 预测当前场景，记录四项指标与$S_j$。\\
6. 汇总54场景结果，按式~\eqref{eq:q2_robust_r}计算$R$。\\
\bottomrule
\end{tabular}
\end{table}
\FloatBarrier

\subsection{基线与LTARP的结构比较}\label{subsec:q2_fair}
\QTwoParagraph 在统一完整输入选模标准下，LTARP在宏平均F1、MAE和相关系数上均优于MMP均值，准确率变化较小。对相同模型参数施加54种缺失后，三项指标的平均改善仍保留，准确率则略有下降（表~\ref{tab:q2_main}）。内容加权分支的收益主要体现于类别均衡性和连续强度预测。

\input{generated/table_q2_main.tex}

\QTwoParagraph 以跨场景综合得分考察三次初始化，种子42和43呈正增益，种子44近乎持平；配对差$\Delta R=R_{\mathrm{LTARP}}-R_{\mathrm{MMP}}$分别为$+0.005600$、$+0.003791$和$-0.000090$，平均为$+0.003100\pm0.002907$。总体收益较小，同时存在初始化敏感性。
\FloatBarrier

\subsection{缺失类型、位置与比例分析}\label{subsec:q2_patterns}
\QTwoParagraph 总体平均保留了改善，但不同模态的缺失影响有所差异。图~\ref{fig:q2_ratio}中，文本随遮挡比例增大退化最明显，语音和视觉变化较平缓；分类与回归对输入损失的响应程度也不同。文本遮挡由0.1增至0.5时，宏平均F1从0.6163降至0.5933，显示语言内容在这组样本中的较强作用。

{{fig:ratio}}
\FloatBarrier

\QTwoParagraph 缺失长度相同，发生位置仍会改变预测。固定目标比例0.3后，文本前段与后段遮挡对宏平均F1的影响大于中段；包含文本的双模态缺失也呈相近规律，语音、视觉及语音--视觉组合的影响较小。分类与回归的位置响应并不完全一致，例如文本中段遮挡降低宏平均F1，却使MAE略有下降（图~\ref{fig:q2_location}）。

{{fig:location}}
\FloatBarrier

\QTwoParagraph 相同模态、位置和比例下的配对差进一步体现LTARP的作用（图~\ref{fig:q2_gain}）。宏平均F1和Pearson的改善更为一致，在语音、视觉及语音--视觉缺失条件下较明显；准确率随位置与模态组合出现正负波动。内容加权补充有助于保持部分剩余信息，其效果仍随受损输入而变化。

{{fig:gain}}
\FloatBarrier

\subsection{稳定性、错误分布与消融分析}\label{subsec:q2_ablation}
\subsubsection{随机初始化稳定性与验证集错误分布}
\QTwoParagraph 模态与位置的平均规律之外，初始化变化、类别混淆和特殊输入条件进一步反映模型的适用边界。三次训练的配对收益较小，因而需结合完整输入的分类误差及原生视觉全零样本，观察这些改善在不同条件下的保持程度（图~\ref{fig:q2_error}）。

{{fig:error}}
\FloatBarrier

\QTwoParagraph 三次初始化中有两次提高，一次近乎持平，场景分组后的平均差也较小。完整输入下，消极与积极类的召回相对较高，中性类最低，为43.48\%；中性样本更容易被分到两侧极性类别，弱极性与类别边界仍是主要混淆位置。

\QTwoParagraph 验证集另有15条原生视觉特征整段全零样本。该子集中，部分初始化的LTARP分类表现低于MMP，说明较小的总体平均收益可以与特殊输入上的退化同时出现。该子集样本量较少，其表现结合三次初始化单独观察。

\subsubsection{候选结构与训练策略消融}
\QTwoParagraph 为比较改善来自何种结构或训练调整，保留时间编码、训练期连续遮挡和均值--最大值残差池化三种候选，与MMP及LTARP并列（表~\ref{tab:q2_ablation}）。

\input{generated/table_q2_ablation.tex}

\QTwoParagraph 单模态时间编码在聚合前加入顺序建模，参数量明显增加，综合得分却低于MMP。训练期连续遮挡保持网络不变：每条样本以0.5概率保留完整输入，其余样本中75\%遮挡一个模态、25\%遮挡两个模态，比例从$\{0.1,0.2,0.3,0.4,0.5\}$选择，起点在有效范围内随机确定，掩码保持原值。这种缺失增强也未带来综合得分改善。

\QTwoParagraph 均值--最大值残差池化保留均值分支
{{eq:ablation_0}}
并以有效槽位内逐特征维的最大值提供补充，再经残差组合接入原有网络。其得分接近LTARP，但仅保留一次初始化结果，主要作为结构探索参考，不与三随机种子结果作精细统计比较。LTARP的三种子综合均值在本组最高，同时只增加883个参数，支持采用轻量内容加权补充的选择。
\FloatBarrier

\subsection{公开方法适配与对比}\label{subsec:q2_public}
\QTwoParagraph 在基线及候选结构之外，进一步观察LTARP与其他多模态建模路线的相对表现。选取11种公开方法，统一适配到附件2的aligned-50输入与分类、回归双任务接口；三模态维度、数据划分、指标和完整输入选模标准保持一致，各方法按既有适配配置训练。表~\ref{tab:q2_public_overview}概括适配中的机制与规模。

\input{generated/table_q2_public_overview.tex}

\QTwoParagraph 常规融合路线通过张量、时序注意力、表征解耦或辅助任务组织多模态信息；缺失增强路线进一步引入重构、表征对齐、元更新和缺失推断。它们保留相应机制并接入本问的特征与双任务接口，以下结果均限定在本文统一适配与评价设置下。

\QTwoParagraph 完整输入下，LTARP四项指标均处于较优水平（表~\ref{tab:q2_public_baselines}）。部分轻量方法及缺失增强方法的表现接近LTARP，而更复杂的融合结构未呈现同等的平均优势。Self-MM和MMIM借助辅助任务或信息约束，在较小规模下也保留了较强预测信息；TFR-Net的宏平均F1接近LTARP，但回归指标有所差异。因此，分类均衡性和回归结果需要结合观察。

\input{generated/table_q2_public_baselines.tex}

\QTwoParagraph 54种连续缺失等权平均后，LTARP在宏平均F1、MAE和Pearson上保持较好的综合表现，准确率与TFR-Net、MMIN接近（表~\ref{tab:q2_public_missing}）。完整输入表现与缺失后保持程度共同决定跨场景结果；缺失增强方法之间也存在差异。

\input{generated/table_q2_public_missing.tex}
\FloatBarrier

\QTwoParagraph 参数规模进一步体现性能与复杂度的折中。LTARP总参数量为164343，约16.4万，明显小于TFN、MulT和MISA等结构，与MMIM和MissModal处于相近量级。在表~\ref{tab:q2_public_baselines}--\ref{tab:q2_public_missing}所列模型中，其完整输入和缺失平均宏平均F1均保持较高水平（图~\ref{fig:q2_params}），较小规模仍保留了两类任务的有效表示。

{{fig:params}}
\FloatBarrier

\subsection{附件3预测}\label{subsec:q2_attachment3}
\subsubsection{文本特征重构与核验}
\QTwoParagraph 完成模型选择后，将LTARP用于附件3的30条无标签样本。附件3对齐版提供\texttt{text\_bert}、\texttt{audio}和\texttt{vision}，缺少预计算的768维\texttt{text}，因此先以\texttt{bert-base-uncased}编码给定词元输入，重构$50\times768$文本表示。

\QTwoParagraph 接口检查沿用20条同时具有词元输入与预计算文本表示的样本，对604个有效槽位逐行比较。重构行的最大余弦相似度位置均为同索引，特征最大绝对差为$3.0041\times10^{-5}$、RMSE为$8.3977\times10^{-7}$。输入同一固定参数模型后，分类输出最大绝对差为$2.44\times10^{-6}$，回归为$1.12\times10^{-6}$。该检查连接重构文本表示与既有预测接口，随后按固定路径完成附件3推理。

\subsubsection{预测结果汇总}
\QTwoParagraph 30条样本预测为消极、中性和积极的数量分别为7、11和12，对应23.33\%、36.67\%和40.00\%。连续强度均值为$+0.123114$、样本标准差为0.611792，范围为$[-1.642545,+1.539758]$；平均分类置信度为0.624599。逐样本类别、强度和置信度见表~\ref{tab:q2_attachment3_all}，与独立CSV保存的同批结果对应。附件3不提供真实标签，因此本节仅报告预测结果，不计算Accuracy、Macro-F1、MAE等监督评价指标。

\input{generated/table_q2_attachment3.tex}
\FloatBarrier

\subsection{本问小结}\label{subsec:q2_summary}
\QTwoParagraph 本问从附件2的三模态对齐特征出发，以MMP建立简单的双任务基线，再通过LTARP内容加权残差分支补充不同时间位置的信息差异。第二阶段仅新增883个参数，沿用原有融合与输出路径，在相同结构下处理完整输入和连续缺失输入。

\QTwoParagraph 文本局部缺失的影响最明显，分类与回归对位置的响应有所不同。LTARP相对基线在宏平均F1、MAE和Pearson上总体改善，同时保留小幅收益与初始化敏感性的边界；公开方法比较中也呈现较好的性能--规模折中，并完成附件3预测。下一问固定该预测模型，进一步分析不同模态及局部时间区域对模型输出的贡献。
'''
for k,v in eqs.items(): body=body.replace('{{eq:'+k+'}}',v)
for k,v in figures.items(): body=body.replace('{{fig:'+k+'}}',v)
body=body.replace('{{metrics}}',r'\Needspace{7\baselineskip}'+'\n'+metrics)
body=body.replace(r'\input{generated/table_q2_attachment3.tex}',r'\Needspace{36\baselineskip}'+'\n'+r'\input{generated/table_q2_attachment3.tex}')
body=body.replace('MMP只在有效槽位上计算均值，再完成模态映射、融合和双任务预测。', 'MMP只在有效槽位上计算均值，再完成模态映射、融合和双任务预测。均值表示将不同有效长度的序列组织为固定维度，使各样本可以进入同一融合网络；同时保留一条明确的等权聚合路径，便于后续观察内容加权的增量。')
body=body.replace('原始有效范围在遮挡前后保持一致。', '原始有效范围在遮挡前后保持一致。人工置零的位置仍位于有效前缀中，均值聚合的分母继续使用原有效长度，模型因而接收相同长度、局部内容受损的序列。')
body=body.replace('填充位的$\\alpha_{it}^{(m)}$恒为0。', '填充位的$\\alpha_{it}^{(m)}$恒为0。人工遮挡后的零值仍按原有效指示参与打分与聚合，权重变化由当前输入内容决定。')
body=body.replace('新增分支只改变模态内的聚合方式。', '新增分支只改变模态内的聚合方式。共享映射使两路表示具有相同维度，均值路径保留整体信息，内容加权路径提供随样本变化的补充；残差系数控制补充分支进入预测的程度。')
body=body.replace('评价在完整输入之外分别组合模态、比例和位置，形成固定的54种场景。', '评价在完整输入之外分别组合模态、比例和位置，形成固定的54种场景。单模态场景沿比例变化观察各输入的敏感程度，双模态场景在同一比例下观察信息同时受损的影响；三个位置区分起始、中央和末段缺失，连接受损范围与时间组织。')
body=body.replace('三次初始化是统计重复单位，54个场景在各自初始化内等权平均。', '三次初始化是统计重复单位，54个场景在各自初始化内等权平均。这样，场景平均表示模型在固定受损条件集合中的总体表现，种子间标准差则描述同一训练设置的初始化波动。')
body=body.replace('内容加权分支的收益主要体现于类别均衡性和连续强度预测。', '内容加权分支的收益主要体现于类别均衡性和连续强度预测。宏平均F1按类别等权汇总，能够呈现准确率之外的类别差异；MAE与相关系数的同步改善，则分别对应强度误差及预测趋势的变化。')
body=body.replace('内容加权补充有助于保持部分剩余信息，其效果仍随受损输入而变化。', '内容加权补充有助于保持部分剩余信息，其效果仍随受损输入而变化。包含文本的受损场景影响较大，而仅语音或视觉受损时仍有文本输入；配对差和相对完整输入的退化分别描述结构收益与信息损失，两者共同刻画缺失规律。')
body=body.replace('该子集样本量较少，其表现结合三次初始化单独观察。', '原生全零输入在有效前缀内保持原值，它所对应的观测条件与人工遮挡实验分别记录。该子集样本量较少，其表现结合三次初始化单独观察。')
assert not re.search(r'\{\{(?:eq|fig|metrics):?',body)
preamble=source.split(r'\begin{document}')[0]
preamble=preamble.replace('graphicx,booktabs,array,longtable','graphicx,booktabs,array,longtable,tabularx')
preamble=preamble.replace(r'\setlength{\parindent}{2em}',r'''\setlength{\parindent}{2em}
\newcommand{\QTwoParagraph}{\par\noindent\hspace*{2em}}
\newcounter{q2algorithm}''')
preamble=re.sub(r'\\graphicspath\{[^\n]*',lambda _:r'\graphicspath{{figures/q2/}}',preamble)
tex=preamble+r'''\begin{document}
\songti\zihao{-4}\pagestyle{plain}
\setcounter{section}{1}\setcounter{figure}{7}\setcounter{table}{7}
'''+body+'\n'+bib+'\n'+r'\end{document}'+'\n'
(O/'q2_story_rewrite.tex').write_text(tex,encoding='utf-8')
for i in range(3):
    args=['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q2_story_rewrite.tex']
    r=subprocess.run(args,cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (O/'build'/f'compile_{i}.txt').write_bytes(r.stdout)
    if r.returncode: raise RuntimeError(r.stdout.decode('utf-8',errors='replace')[-4000:])
shutil.copy2(O/'build/q2_story_rewrite.pdf',O/'q2_story_rewrite.pdf')
print('Compiled Q2 story rewrite.')
