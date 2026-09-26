"""Enhance section motivations and transitions without modifying results."""
from pathlib import Path
import re,shutil,subprocess,json,hashlib
O=Path(__file__).resolve().parent
S=O.parent/'q2_final_refine'
(O/'build').mkdir(parents=True,exist_ok=True)
(O/'qa').mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if not (O/'qa/input_hashes.json').exists():
    (O/'qa/input_hashes.json').write_text(json.dumps({str(p):sha(p) for p in S.rglob('*') if p.is_file() and 'build' not in p.relative_to(S).parts and 'qa' not in p.relative_to(S).parts},ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['figures','generated']:shutil.copytree(S/name,O/name,dirs_exist_ok=True)
t=(S/'q2_final_refine.tex').read_text(encoding='utf-8')
def para(prefix,new):
    global t
    pattern=r'\\QTwoParagraph '+re.escape(prefix)+r'[^\n]*'
    t,n=re.subn(pattern,lambda _:r'\QTwoParagraph '+new,t)
    if n!=1:raise ValueError('Expected one paragraph: '+prefix)
para('附件2已经提供',r'''针对问题二，附件2已经提供对齐到50个时间位置的文本、语音和视觉特征，本文因此直接从三模态序列的情感建模开始，不再进行原始特征提取与跨模态时序对齐。按照题目要求，本问同时完成情感极性分类和连续情感强度回归，并比较完整输入与局部连续缺失条件下的预测表现。缺失所涉及的模态、位置和持续长度会改变可利用的信息，因此需要在统一输入条件下建立预测模型，再系统考察不同缺失形式的影响。''')
para('先以掩码均值池化',r'''为建立结构简单、便于分析缺失影响的比较参照，本文首先采用掩码均值池化（Masked Mean Pooling，MMP），聚合各模态有效时间位置上的特征，再进行融合并输出分类与回归结果。这一路径能够处理不同有效长度的序列，但对同一模态内所有有效位置等权处理，局部缺失后仍难以区分剩余位置的重要程度。

\QTwoParagraph 针对这一限制，本文在均值表示之外引入内容驱动的时间加权残差分支，构成轻量时间注意力残差池化（Lightweight Temporal Attention Residual Pooling，LTARP）。均值分支保留整体统计信息，新增分支根据当前样本内容调整时间位置权重，为突出剩余序列中更有判别力的信息提供补充。后续在固定连续缺失场景中评价这一补充的实际收益，整体结构见图~\ref{fig:q2_arch}。''')
para('两种池化方式共用',r'''为使后续比较主要反映聚合方式与缺失场景带来的差异，本文在统一的数据划分和标签定义下使用附件2的对齐输入。对齐特征文件为\texttt{aligned-50.pkl}，训练、验证和测试划分分别含3395、728和727条样本。训练集用于学习参数，验证集用于结构选择与评价；测试划分和无标签附件3不参与选模。''')
para('为单独观察连续缺失',r'''在上述输入与任务定义基础上，本文首先建立掩码均值池化基线MMP，作为后续改进结构的比较参照。有效位掩码使填充后缀不参与聚合，各模态由此得到固定维度的均值向量，再接入共同的融合与双任务输出路径。对模态$m\in\{T,A,V\}$，其均值为''')
para('MMP结构简单、参数较少',r'''当局部连续缺失移除一段信息后，剩余有效位置所包含的情感信息并不均等。MMP仍采用等权聚合，难以突出更有判别力的位置。为使聚合权重能够随当前样本内容变化，LTARP在均值分支之外增加逐槽位内容打分，并将加权表示作为残差补充。每个模态使用一个线性标量打分器，其打分记为''')
para('两种池化方式的差异',r'''为系统比较不同连续缺失形式对预测的影响，并在相同受损条件下观察MMP与LTARP的表现，本文从缺失模态、缺失比例和缺失位置三个维度构造固定场景。单模态场景沿比例变化考察各输入的敏感程度，双模态场景在同一比例下考察信息同时受损的影响；前段、中段和后段遮挡则用于区分缺失发生位置。三个维度组合形成54种场景，使模型比较与缺失规律分析共享同一组受损输入。''')
para('将分类与回归表现',r'''分类与回归分别反映极性识别和连续情感强度预测，单项指标不能同时概括两项任务。为在参数选择时兼顾二者，本文将四项评价指标组合为得分$S$：''')
para('本文将完整输入对应',r'''完整输入得分只能描述正常观测下的表现，连续缺失后的保持程度还需结合受损场景评价。本文将完整输入对应的得分记为$S_{\mathrm{clean}}$，并以其与54种缺失场景的平均得分共同定义综合鲁棒得分$R$：''')
para('在统一完整输入选模标准下',r'''首先在统一完整输入选模标准下直接比较MMP与LTARP，以观察聚合结构改变后的分类与回归表现。LTARP在宏平均 F1、MAE和Pearson的三种子均值上均优于MMP，准确率变化较小。对相同模型参数施加54种缺失后，三项指标的平均改善仍保留，准确率则略有下降（表~\ref{tab:q2_main}）。MAE与Pearson的改善分别对应连续情感强度绝对误差的减少和预测趋势的一致性提高。''')
para('总体平均保留了改善',r'''表~\ref{tab:q2_main}给出了完整输入与54种缺失场景的总体平均，但平均值不能呈现不同缺失模态、比例和位置的差异。为进一步辨别这些差异，以下分解考察最终鲁棒配置的缺失响应。图~\ref{fig:q2_ratio}中，文本随遮挡比例增大退化最明显，语音和视觉变化较平缓；分类与回归对输入损失的响应程度也不同。文本遮挡由0.1增至0.5时，宏平均 F1从0.6163降至0.5933，显示语言内容在这组样本中的较强作用。''')
para('相同模态、位置和比例下',r'''在比例与位置规律之外，图~\ref{fig:q2_gain}进一步比较最终鲁棒配置与MMP基线在相同受损场景下的响应差异。此处LTARP采用按$R$选定的配置，分析目的在于观察最终配置的场景收益；统一完整输入选模标准下的直接结构比较仍以表~\ref{tab:q2_main}为准。宏平均 F1和Pearson的改善更为一致，在语音、视觉及语音--视觉缺失条件下较明显；准确率随位置与模态组合出现正负波动。内容加权补充有助于保持部分剩余信息，其效果仍随受损输入而变化。包含文本的受损场景影响较大，而仅语音或视觉受损时仍有文本输入；配对差和相对完整输入的退化分别描述结构收益与信息损失，两者共同刻画缺失规律。''')
para('前述结果呈现了',r'''前述分析给出了不同缺失场景的平均响应，下面进一步从随机初始化、类别混淆和特殊输入三个方面观察这些结果的保持程度。三次初始化的配对收益较小，结合完整输入分类错误与原生视觉全零样本，可以识别总体平均之外的差异（图~\ref{fig:q2_error}）。''')
para('为比较改善来自何种',r'''在考察最终配置的稳定性与错误分布后，进一步比较不同结构和训练调整能否带来类似收益。本文保留时间编码、训练期连续遮挡和均值--最大值残差池化三种候选，与MMP及LTARP并列（表~\ref{tab:q2_ablation}）。''')
para('在基线及候选结构之外',r'''完成MMP与LTARP的内部比较及候选结构分析后，为考察LTARP与其他多模态建模路线的相对表现，本文进一步选取11种公开方法进行统一适配比较。各方法使用附件2的aligned-50输入与分类、回归双任务接口，三模态维度、数据划分、指标和完整输入选模标准保持一致，并按既有适配配置训练。表~\ref{tab:q2_public_overview}概括适配中的机制与规模。''')
para('完整输入下，LTARP四项',r'''在本文统一适配与评价设置下，LTARP在完整输入的四项指标上均取得表~\ref{tab:q2_public_baselines}中最优的三随机种子均值。部分轻量方法及缺失增强方法的表现接近LTARP，而更复杂的融合结构未呈现同等的平均优势。Self-MM和MMIM借助辅助任务或信息约束，在较小规模下也保留了较强预测信息；TFR-Net的宏平均 F1接近LTARP，但回归指标有所差异。因此，分类均衡性和回归结果需要结合观察。''')
para('54种连续缺失等权平均后',r'''在本文统一适配与评价设置下，54种连续缺失等权平均后，LTARP的宏平均 F1和Pearson在表~\ref{tab:q2_public_missing}中最高，MAE最低，准确率与TFR-Net、MMIN接近。完整输入表现与缺失后保持程度共同决定跨场景结果；缺失增强方法之间也存在差异。''')
para('完成模型选择后',r'''完成上述验证与模型选择后，将LTARP用于附件3的30条无标签样本。对齐版提供\texttt{text\_bert}词元输入，以及\texttt{audio}和\texttt{vision}数值特征。单样本的\texttt{text\_bert}形状为$3\times50$，三行依次为词元编号、有效位掩码和词元类型编号，并非模型所需的$50\times768$连续文本表示。因此，使用已验证的\texttt{bert-base-uncased}接口编码有效词元前缀，取最后隐层逐词元表示，再在填充后缀补零至$50\times768$；语音、视觉特征及有效位掩码保持原样。''')
para('本问从附件2的三模态',r'''本问以附件2已对齐的三模态特征为起点，建立MMP双任务基线，并通过LTARP内容加权残差分支补充时间位置的信息差异。在共用融合与输出路径的基础上，第二阶段仅新增883个参数，形成完整输入和连续缺失输入下可直接评价的预测模型。''')
para('54种缺失场景中',r'''围绕54种连续缺失场景的比较表明，文本局部缺失的影响最明显，分类与回归对位置的响应有所不同。LTARP在宏平均 F1、MAE和Pearson上总体优于MMP，但提升幅度较小，不同随机初始化下仍存在一定波动。与公开方法的统一适配比较进一步体现了其性能与参数规模的折中。在此基础上，本问完成附件3全部30条无标签样本的预测，并为问题三的解释分析固定预测模型。''')
# Avoid a half-empty page followed by an isolated data figure after the longer opening.
t=t.replace(r'\label{fig:q2_data}\end{figure}'+'\n'+r'\FloatBarrier',
            r'\label{fig:q2_data}\end{figure}')
(O/'q2_richer_narrative.tex').write_text(t,encoding='utf-8')
for i in range(3):
    r=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q2_richer_narrative.tex'],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (O/'build'/f'compile_{i}.txt').write_bytes(r.stdout)
    if r.returncode:raise RuntimeError(r.stdout.decode('utf-8',errors='replace')[-3000:])
shutil.copy2(O/'build/q2_richer_narrative.pdf',O/'q2_richer_narrative.pdf')
print('Compiled narrative preview.')
