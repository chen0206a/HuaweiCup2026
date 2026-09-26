from pathlib import Path
import re, shutil, json, hashlib, subprocess

O = Path(__file__).resolve().parent
ROOT = O.parent.parent
SRC = Path('D:/java录屏/07_q3.tex')
PAPER = ROOT/'论文润色工作区/E2026_论文整理包/01_LaTeX论文工程/paper'
for name in ('source','generated','figures/q3','build','qa','evidence','figure_sources'):
    (O/name).mkdir(parents=True, exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
source = SRC.read_text(encoding='utf-8-sig')
for p in (SRC, SRC.parent/'Q3_最终定稿版.pdf'):
    shutil.copy2(p,O/'source'/p.name)
protected = [SRC, SRC.parent/'Q3_最终定稿版.pdf', ROOT/'paper/main.tex', PAPER/'main.tex', PAPER/'sections/05_q1.tex', PAPER/'sections/06_q2.tex', PAPER/'sections/07_q3.tex']
protected += list((ROOT/'E2026/outputs/q3').glob('*.json'))
protected += list((ROOT/'E2026/outputs/q3/final').glob('*'))
protected = {str(p):sha(p) for p in protected if p.is_file()}
if not (O/'qa/protected_hashes.json').exists():
    (O/'qa/protected_hashes.json').write_text(json.dumps(protected,ensure_ascii=False,indent=2),encoding='utf-8')
eqs={re.search(r'\\label\{([^}]+)\}',e)[1]:e for e in re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',source,re.S)}
tables={re.search(r'\\label\{([^}]+)\}',t)[1]:t for t in re.findall(r'\\begin\{table\}.*?\\end\{table\}',source,re.S)}
for k,t in list(tables.items()):
    t=t.replace(r'\begin{table}[htbp]',r'\begin{table}[H]')
    t=t.replace('分类主导证据层级','分类关键位置来源').replace('类别或状态','类别或来源')
    if k=='tab:q3_seed_stability':
        t=t.replace('Shapley排序相关','Shapley秩相关')
        t=t.replace('分类排序相关仅统计两种子预测类别一致的样本；区间定位按分类目标定义，回归未单独定义关键区间。','秩相关为逐样本三模态Shapley贡献的Spearman系数均值。分类仅统计两种子预测类别一致的样本；IoU为主导模态相同时的区间交并比，模态不同时记为0；回归未单独定义关键区间。')
    tables[k]=t
for name in ('table_q3_attachment4_all.tex','table_q3_attachment4_explanations.tex'):
    t=(PAPER/'generated'/name).read_text(encoding='utf-8-sig')
    t=t.replace('分类贡献与证据层级','分类贡献与输入来源').replace('证据层级','输入来源')
    t=t.replace('& 特征行级','& 未对齐特征行级')
    (O/'generated'/name).write_text(t,encoding='utf-8')
figures={}
captions={
 'fig:fig14_q3_heaf_framework':'固定预测模型的解释分析流程。输入为文本、语音和视觉特征，输出为预测类别、连续强度、模态贡献与关键特征区间。',
 'fig:fig15_q3_faithfulness_validation':r'主导模态与删除干预结果。（A）728条验证样本的分类、回归主导模态计数；（B）396条独立验证样本的删除比例曲线；（C）高影响位置与随机位置的配对下降差。阴影为按视频分组的95\%自助采样区间。',
 'fig:fig16_q3_case_explanations':r'附件4的典型样本。（A）样本14的文本贡献、局部遮挡曲线与原文片段；（B）样本02的视觉贡献、局部遮挡曲线与未对齐特征行对应。浅色带标出所选特征区间，视频画面为上下文示意。'}
for f in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',source,re.S):
    k=re.search(r'\\label\{([^}]+)\}',f)[1]
    path=re.search(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',f)[1]
    if not (O/path).exists(): shutil.copy2(PAPER/path,O/path)
    f=f.replace(r'\begin{figure}[htbp]',r'\begin{figure}[H]')
    f=re.sub(r'\\caption\{[^\n]*\}',lambda _:r'\caption{'+captions[k]+'}',f)
    figures[k]=f

body=r'''\section{问题三：多模态情感预测的可解释分析与证据定位}\label{sec:q3}

\subsection{问题分析与总体方案}\label{subsec:q3_overview}
问题二已经得到每条样本的情感类别和连续强度。问题三沿用该预测模型，进一步分析这些结果主要受到哪些输入影响。模型参数保持固定，解释对象是同一条样本在不同输入条件下的输出变化。

文本内容、说话方式和画面特征可能提供不同的情感线索。最终类别将三路信息合并为一个判断，却没有显示各模态的作用大小，也没有指出重要信息位于何处。因此，本问关注三种模态各自贡献多少、两种模态共同输入时有何额外作用，以及影响当前预测的具体特征位置。

本文通过实际改变输入观察预测响应。先枚举模态组合，计算文本、语音和视觉的Shapley贡献；再利用同一批组合输出分析模态交互；随后在分类主导模态内逐段遮挡连续特征区间，定位对当前类别影响较大的位置。分类与连续强度采用各自的解释目标，使两种输出的作用方向都有明确含义。

所选区间与随机等长区间进行比较，并通过逐步删除高影响位置观察累计效应。跨种子比较考察解释对模型初始化的敏感程度；来源回溯则将特征索引对应到原文片段或未对齐特征记录。这些分析评价输入与固定模型输出的关系，结论限定于模型响应，不作为现实情感形成的因果解释。

图~\ref{fig:fig14_q3_heaf_framework}给出总体流程。三路输入分别为$50\times768$的文本特征、$50\times74$的语音特征和$50\times35$的视觉特征，预测与解释均使用问题二的双任务模型。

{{fig:fig14_q3_heaf_framework}}

\subsection{模态贡献分析}\label{subsec:q3_shapley}
\subsubsection{精确Shapley贡献}
只比较完整输入与去掉某一模态后的输出，得到的是该模态在另外两种模态都存在时的作用。一个模态的边际变化还可能取决于其他输入是否参与：单独有效的信息在联合输入中可能被替代，也可能在另一模态提供的条件下发挥更大作用。因此，需要把不同组合中的变化一起考虑。

Shapley贡献将一个模态加入不同组合时的边际变化加权平均。令$\mathcal M=\{T,A,V\}$分别表示文本、语音和视觉，$S\subseteq\mathcal M$为保留的模态集合，$v(S)$为相应组合的解释目标。三种模态只有$2^3=8$种组合，即空组合、三个单模态组合、三个双模态组合及完整组合，可以直接完整枚举，无需Monte Carlo采样近似。对模态$m$，其贡献为
{{eq:eq:q3_shapley}}
式中权重对应模态在不同加入顺序下遇到集合$S$的比例。三模态情况下，加入空组合或另外两模态均在场时的权重各为$1/3$，加入单模态组合时各为$1/6$。每个模态按相同规则计算，使单独作用与联合输入下的增量都参与贡献分配。

计算时，未保留模态只在有效位置清零，填充位置与原生零记录保持原样。空组合对应全部有效模态清零后的模型输出，作为本问的共同基准。精确计算满足$\sum_m\phi_m=v(\mathcal M)-v(\varnothing)$，用于检查贡献分配是否完整；该关系针对模型输出差，空组合的预测值本身由固定模型决定。

\subsubsection{解释目标与主导模态}
分类解释固定完整输入预测的类别$c^\star$，所有模态组合均衡量对这一类别的支持程度。若只记录遮挡后类别是否翻转，许多尚未越过分类边界的变化会被忽略；采用分类对数优势（log-odds），则能保留当前类别相对于其他类别的连续变化。记$z_k(S)$为组合$S$下类别$k$的logit，回归直接使用连续强度预测$\widehat y^{\mathrm r}(S)$，两种目标定义为
{{eq:eq:q3_value}}
其中，分类目标等于$\log[p_{c^\star}(S)/(1-p_{c^\star}(S))]$。即使某个组合使模型改判其他类别，$c^\star$也保持不变，前后差值仍指向同一个解释对象。回归目标保留模型输出的原始尺度，因此分类和回归分别计算贡献并确定主导模态。

分类正贡献表示该模态支持当前类别，负贡献表示其反向作用；回归正、负贡献分别对应连续强度向正、负方向的变化。贡献的绝对值表示变化幅度，符号保留方向。例如，某模态可以增强中性判断，同时使连续强度降低，两个结果分别依据各自目标解读。

分类主导模态优先取最大正Shapley贡献；若三种贡献均非正，则取绝对贡献最大者，并标明其非支持性质。回归主导模态取绝对贡献最大者。两类选择遇到并列时，均按文本、语音、视觉的固定顺序处理。后续连续窗口遮挡以分类主导模态为分析对象，回归输出用于观察同一区间对强度的影响。

主导模态确定局部分析对象，三路贡献仍共同保留。非主导模态可能提供较小的同向支持，也可能产生较大的反向作用；只报告最大值会掩盖这种差异。逐样本表因此列出完整的三模态贡献，便于结合大小和方向阅读当前预测。

\subsection{模态交互分析}\label{subsec:q3_interaction}
Shapley贡献将完整输入相对空输入的预测变化分配给各模态，其中已经包含联合输入的影响。模态交互另行回答：两种模态共同输入后，相对于各自作用的简单相加，多出了什么变化。贡献较大的模态与其他模态仍可能呈现正或负交互，二者描述的方面不同。

本文考察文本--语音、文本--视觉和语音--视觉三对关系。对模态$i,j$及剩余模态$k$，分别计算第三模态缺席和在场时的二阶差，再取平均：
{{eq:eq:q3_interaction}}
正交互表示联合输入产生的输出变化高于相应加性基准，负交互表示低于这一基准。符号针对式~\eqref{eq:q3_value}的解释目标，具体原因需结合样本输入判断；分类和回归分别计算。

附件2验证集728条样本的交互统计见表~\ref{tab:q3_interaction}。平均交互概括作用的平均大小，主要符号比例为该方向在样本中出现的频率。文本--视觉在两个目标下平均为正，其中分类均值为$+0.0781$；语音--视觉的平均方向则由分类负向转为回归正向。可见，共同输入对类别支持与连续强度的影响并不完全一致。

{{tab:tab:q3_interaction}}

总体方向也不能代替单样本结果。文本--视觉的分类交互有60.3\%为正，仍有相当一部分样本呈现其他方向。群体统计用于概括这批输入的共同特点，逐条解释则保留样本自身的交互量，与其模态贡献和局部位置一起阅读。

\subsection{连续窗口遮挡与关键区间定位}\label{subsec:q3_temporal}
模态贡献比较整路输入的作用，尚未指出重要信息位于序列何处。本文在分类主导模态内部遮挡连续特征区间，保持其他模态及窗口外输入不变，通过输出变化定位局部影响。连续窗口覆盖相邻特征，可以观察一段输入共同被去除后的响应，避免把片段作用仅归于单个槽位。

对样本$i$，有效长度为$L_i$，窗口宽度为$w_i=\max(1,\operatorname{round}(0.30L_i))$，滑动步长为1。窗口只在有效序列内生成，依次覆盖从起点到末端的所有可用连续区间。有效掩码保持不变，清零操作表示去除当前窗口的观测内容；原生零记录沿用输入原值，填充位置不进入候选窗口。记$g(\mathbf x)$为固定类别的分类对数优势，窗口$W$的遮挡效应为
{{eq:eq:q3_occlusion}}
其中$\mathbf x_i^{\setminus W}$仅将分类主导模态在窗口内的有效特征清零。$D_i(W)>0$表示删除后当前类别支持下降，窗口原本支持该预测；$D_i(W)<0$表示删除后支持上升，窗口原本对当前类别起反向作用。

若存在正效应窗口，选择$D_i(W)$最大的窗口；若所有窗口均无正效应，则选择绝对效应最大的窗口并标注方向。平局时取最早位置。结果使用从零开始的对齐索引$[a,b)$表示，包含$a$至$b-1$位置。对每个有效槽位，将覆盖该位置的窗口效应最大值记为局部曲线；这条曲线描述连续遮挡下的响应，窗口之间可以重叠。

窗口比例通过附件2验证集的设计子集确定。既有实验比较$\{0.10,0.20,0.30\}$，按所选窗口相对随机等长窗口的平均分类对数优势下降差选择比例；三者依次为0.268837、0.366684和0.440487，因此选定0.30。随后固定该比例，在独立验证子集报告结果，并用于附件4。该选择依据对应局部分类影响，实际区间宽度仍随每条样本的有效长度变化。

区间索引首先定位的是模型接收的特征，具体输入来源在第~\ref{subsec:q3_grounding}节回溯。对同一所选窗口，回归同时记录完整输出减遮挡输出的有符号变化及其绝对值，分别描述强度移动方向和幅度。

三模态Shapley只需8种组合，三对交互复用这些输出。连续窗口的候选数为$L_i-w_i+1$，随有效长度线性增加。在最长50位的序列上，单样本的贡献计算与局部定位均可直接遍历；模型参数、有效掩码、解释目标及窗口规则固定后，同一输入的结果可按上述步骤复算。

\subsection{解释干预一致性与稳定性验证}\label{subsec:q3_faithfulness}
\subsubsection{实验设计}
解释实验使用问题二的固定seed 42模型。其在附件2验证集上的基础性能见表~\ref{tab:q3_predictor}，其中中性类别F1和召回率低于另两类，是主要分类误差来源之一。基础性能提供预测质量背景，下述实验评价所选位置与该模型输出之间的对应关系。

{{tab:tab:q3_predictor}}

附件2验证集共728条片段，按video ID分为设计子集和独立验证子集。前者含332条片段、113个视频，后者含396条片段、126个视频，两组video ID无重叠。同一视频的片段可能共享说话人和场景，按视频分组可以避免这些共同条件跨越两组。设计子集用于确定窗口比例，独立验证子集用于报告最终干预结果；这里的独立性针对窗口比例选择与解释验证。

对每条独立验证样本，从同一主导模态的有效序列中随机抽取32次等长窗口，以其平均效应作为对照。等长条件控制被遮挡位置的数量，使比较集中于位置差异。另按局部曲线从高到低删除10\%、20\%、30\%、40\%的有效位置，与删除同样数量的随机位置比较，观察累计影响。随机种子为20260924，95\%区间采用按视频分组的1000次自助采样，以保留同视频片段的组内关联。

\subsubsection{核心干预结果}
表~\ref{tab:q3_faithfulness}中，三个分类指标方向一致：所选窗口删除后引起的输出变化大于随机等长窗口。分类对数优势平均下降差为0.400735，95\%区间为$[0.3635,0.4372]$；置信度和当前类别logit也呈同向差异。这说明固定模型对所选位置更敏感。所选窗口本身由较大遮挡效应确定，对照差异包含这一选择优势，因此该实验衡量输出敏感性及干预一致性。

{{tab:tab:q3_faithfulness}}

回归结果需要区分方向与幅度。有符号变化差为$-0.018095$，区间跨零，表明删除后连续强度没有统一的移动方向；不同样本的正、负变化可能在平均时抵消。绝对变化差为$+0.103545$，说明同一所选区间仍引起更大的强度数值变化。局部窗口依据分类目标选出，回归结果描述该窗口对另一输出的响应。

删除曲线进一步显示影响随去除范围的变化。删除比例由10\%增至40\%时，高影响位置与随机位置的平均分类对数优势下降差依次为0.108296、0.237758、0.418076和0.465465（图~\ref{fig:fig15_q3_faithfulness_validation}）。差异在四个比例下均为正并逐渐扩大，说明高影响位置的累计删除持续改变当前类别支持。曲线按局部效应排序后删除多个位置，与表中的单个连续窗口对照采用不同的删除集合。

{{fig:fig15_q3_faithfulness_validation}}

\subsubsection{跨种子稳定性}
使用既有seed 43、44模型与seed 42比较，保持独立验证样本和窗口比例相同。表~\ref{tab:q3_seed_stability}同时报告主导模态一致率、Shapley贡献的Spearman秩相关系数及关键区间交并比（IoU）。秩相关先在每条样本的三个贡献值之间计算，再对样本取均值；分类相关仅比较两个模型预测类别相同的样本，使贡献排序对应同一类别。

{{tab:tab:q3_seed_stability}}

分类主导模态一致率为0.869和0.831，区间IoU为0.579和0.527；回归主导模态一致率为0.778和0.783。模态选择相对稳定，具体窗口边界更敏感。主导模态只在三路输入之间比较，窗口却要从多个重叠候选中选出最大效应位置；若相邻窗口作用接近，初始化导致的小幅输出变化就可能改变边界。局部区间因此宜结合模态贡献和连续曲线阅读。

\subsection{关键位置的输入来源回溯}\label{subsec:q3_grounding}
连续窗口遮挡给出对齐特征中的索引，来源回溯将这些位置连接到可阅读的文本或已有特征记录。附件4的对应关系来自其自身的词元输入、原文和未对齐特征，按三种模态现有记录能够确认的精度分别报告。

文本采用“对齐特征槽位$\rightarrow$BERT词元$\rightarrow$原文字符区间”的映射。利用固定BERT编码路径重构文本特征后，对20条样本的604个有效槽位进行逐行比较，全部行的最大余弦相似度匹配位置均为同索引。平均同索引余弦为0.999999994，最大绝对差为$3.0041\times10^{-5}$，均方根误差为$8.3977\times10^{-7}$。这些数值支持对齐文本行与词元位置的对应关系。

词元到原文的字符跨度由分词器提供，因而可将所选窗口内的实际词元回到原文字符区间。特殊词元本身不产生原文片段，窗口内的文本以真实字符跨度读取。这一处理使结果既保留模型中的槽位索引，也给出可直接阅读的文字；两种坐标描述的是同一输入在不同表示中的位置。

语音和视觉通过逐行特征匹配回到未对齐数组。附件4的564/564个非零语音对齐行、534/534个非零视觉对齐行，都能在对应样本的未对齐特征中找到唯一且逐元素相等的行。这种对应依赖实际特征值，不要求把50个槽位均分到视频时长；原生零行则保留原记录，不赋予未经确认的来源索引。

据此，文本位置报告为“原文字符级”，语音和视觉位置报告为“未对齐特征行级”。现有音视频来源记录还缺少从未对齐行到音频秒数或视频帧时间的可靠映射，位置精度止于特征行。典型案例图保留的视频画面用于显示上下文，其选取独立于关键特征区间。

\subsection{附件4预测与解释结果}\label{subsec:q3_attachment4}
附件4的20条样本预测为消极、中性和积极的数量分别为7、5、8，分布见表~\ref{tab:q3_attachment4}。全部类别、连续强度和分类置信度列于表~\ref{tab:q3_attachment4_all}；分类Shapley贡献与关键特征区间列于表~\ref{tab:q3_attachment4_explanations}。附件4没有真实标签，因此本节报告预测及解释结果，不计算Accuracy、F1、MAE或Pearson等监督评价指标。

分类主导模态为文本、视觉、语音的样本数分别为19、1、0，回归分别为18、2、0。两类主导模态在17条样本上一致，占85\%，其余样本体现类别支持与强度变化的差异。表中$\phi_T,\phi_A,\phi_V$均对应分类对数优势，保留符号以显示同向支持和反向作用。

{{tab:tab:q3_attachment4}}

\input{generated/table_q3_attachment4_all.tex}

\input{generated/table_q3_attachment4_explanations.tex}

\Needspace{5\baselineskip}
同一类别内部的输出也有明显差异。样本02与样本09均被预测为消极，连续强度却分别为$-0.027301$和$-1.987819$，分类置信度分别为0.3898和0.9671。类别、强度和置信度描述不同方面，逐样本结果将它们分别保留，避免用一个类别标签替代连续输出。

\subsection{典型样本分析与本问小结}\label{subsec:q3_cases}
图~\ref{fig:fig16_q3_case_explanations}选取文本主导的样本14和视觉主导的样本02，比较不同模态作用及其输入回溯结果。

样本14预测为中性，连续强度为$+0.154869$。三模态分类贡献为$\phi_T=+1.536072$、$\phi_A=+0.054570$、$\phi_V=-1.023311$，文本是主要正向输入，视觉对当前中性类别起反向作用，语音贡献较小。文本关键特征区间$[1,12)$对应原文字符$[0,38)$，片段为“He is the co-founder of Rossen and Vet”。这段身份介绍由模型中的连续位置落实到可阅读的原文，说明中性预测的主要文本支持来自何处；贡献的正向指对中性类别的支持，与积极情感标签含义不同。

样本02预测为消极，连续强度为$-0.027301$。其贡献为$\phi_T=-0.197115$、$\phi_A=+0.227718$、$\phi_V=+0.829375$，视觉支持最大，语音同向但较小，文本产生反向作用。视觉关键特征区间$[1,7)$包含对齐槽位1--6，可逐行对应到未对齐视觉特征行36--41。该结果将视觉主导判断缩小到已有输入记录中的具体位置，便于复查所选特征；图中的场景画面提供观看上下文。

{{fig:fig16_q3_case_explanations}}

两个案例的回溯终点不同：样本14可以直接读取原文字符，样本02当前可查到未对齐视觉特征行。同一遮挡方法给出特征区间，而能够进一步阅读或展示何种输入，取决于该模态保存的来源记录。

本问固定问题二的预测模型，以模态组合输出计算三模态贡献与两两交互，再用连续窗口遮挡定位局部影响。随机等长窗口、删除曲线和跨种子比较分别考察局部响应、累计影响及初始化变化，使解释结果可以结合模型的实际输出评估。

附件4的20条样本均得到类别、连续强度、模态贡献和关键区间，并按已有来源记录连接到原文或未对齐特征行。逐样本的同向与反向贡献、不同主导模态及不同位置，共同呈现固定模型使用三路输入的差异，也为检查具体预测提供了可复查的记录。
'''
for k,v in eqs.items(): body=body.replace('{{eq:'+k+'}}',v)
for k,v in tables.items(): body=body.replace('{{tab:'+k+'}}',v)
for k,v in figures.items(): body=body.replace('{{'+k+'}}',v)
assert '{{' not in body
# Indent ordinary prose, including paragraphs immediately after headings/floats.
body=re.sub(r'(?m)^([\u4e00-\u9fff“].*)',lambda m:r'\QThreeParagraph '+m[1],body)
preamble=r'''\documentclass[UTF8,a4paper,12pt]{ctexart}
\usepackage[a4paper,top=2.4cm,bottom=2.2cm,left=2.35cm,right=2.35cm]{geometry}
\setmainfont{Times New Roman}
\usepackage{graphicx,booktabs,array,multirow,longtable,tabularx,amsmath,amssymb,bm,mathtools}
\usepackage{caption,float,placeins,needspace,setspace}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\setstretch{1.6}
\newcommand{\QThreeParagraph}{\par\noindent\hspace*{2em}}
\captionsetup{font=small,labelfont=normalfont,labelsep=quad,justification=centering}
\ctexset{
 section={format=\centering\heiti\zihao{4},beforeskip=1.5ex,afterskip=1.2ex},
 subsection={format=\heiti\zihao{-4},beforeskip=1.2ex,afterskip=.6ex},
 subsubsection={format=\heiti\zihao{-4},beforeskip=.8ex,afterskip=.4ex}}
\begin{document}
\songti\zihao{-4}\pagestyle{plain}
\setcounter{section}{2}\setcounter{figure}{15}\setcounter{table}{10}
'''
(O/'q3_story_refine.tex').write_text(preamble+body+'\n\\end{document}\n',encoding='utf-8')
for i in range(3):
    r=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q3_story_refine.tex'],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (O/'build'/f'compile_{i}.txt').write_bytes(r.stdout)
    if r.returncode: raise RuntimeError(r.stdout.decode('utf-8',errors='replace')[-4000:])
shutil.copy2(O/'build/q3_story_refine.pdf',O/'q3_story_refine.pdf')
print('Compiled Q3 narrative refinement.')
