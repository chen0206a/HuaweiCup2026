from pathlib import Path
import re, shutil, subprocess, json, hashlib

O=Path(__file__).resolve().parent
V2=O.parent/'q1_rewrite_preview_v2'
old=(V2/'sections/05_q1.tex').read_text(encoding='utf-8')
for name in ['figures','generated','bibliography']:
    shutil.copytree(V2/name,O/name,dirs_exist_ok=True)
for name in ['q1_numbers.tex','protected_paper_hashes.json']:
    shutil.copy2(V2/name,O/name)
for name in ['sections','build','qa']:(O/name).mkdir(exist_ok=True)
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in V2.rglob('*') if p.is_file() and 'evidence' not in p.relative_to(V2).parts and 'qa' not in p.relative_to(V2).parts and 'build' not in p.relative_to(V2).parts}
(O/'qa/v2_protected_hashes.json').write_text(json.dumps(protected,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'q1_story_rewrite.tex').write_text((V2/'q1_rewrite_preview_v2.tex').read_text(encoding='utf-8'),encoding='utf-8')
equations=re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',old,re.S)
figures=re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',old,re.S)
assert len(equations)==10 and len(figures)==7
captions=[
    '三模态特征提取与时序对齐流程。',
    '附件1样本与三模态特征统计。',
    '文本模态特征提取流程。色块为表示结构示意。',
    '语音模态特征提取流程。色块为表示结构示意。',
    '视觉模态特征提取流程。色块为表示结构示意。',
    '三模态原生序列到公共时间窗的映射。矩阵色值为结构示意。',
    '对齐方法的辅助评价。点为均值，半透明区间为均值上下一个样本标准差。',
]
article=r'''\section{问题一：多模态特征提取与时序对齐}\label{sec:q1}

\subsection{问题分析与总体方案}\label{subsec:q1_framework}
附件1给出了100条包含文本、语音和画面信息的视频样本。三种模态经过各自编码后，文本以词语或片段形成表示，语音形成较密集的声学序列，视觉对应离散采样帧。要将这些长度和时间粒度不同的序列组织在一起，首先需要确定每个特征在原视频中对应的时间范围。

本文把独立编码、尚未进入公共时间轴的向量序列称为{\heiti 原生特征}，把每个向量对应的媒体时间区间称为它的{\heiti 时间支持}。文本的时间支持来自词语或识别片段，语音来自WAV样本范围，视觉来自采样帧附近的区间。这样，每个向量既保留内容表示，也带有能够连接其他模态的时间信息。

整体方案如图~\ref{fig:fig01_q1_overview}所示。先建立统一媒体时间基准，分别提取带时间支持的三模态原生特征，再将每条视频的有效时长划分为50个公共窗。各模态按最大时间重叠规则选取每个窗的来源，形成统一长度的序列，并同步保存有效掩码和原始来源记录。

@@FIG1@@

\subsection{原始数据预处理与统一时间基准}\label{subsec:q1_timeline}
三模态的时间联系从原始媒体开始建立。每条样本以视频标识和片段编号定位，将附件1文件夹名、MP4文件名与\texttt{label-100.xlsx}中的记录连接，取得对应的赛题转写与情感标注。媒体路径及文件哈希随处理记录保存，使三种特征始终连接到同一片段。

音频由FFmpeg提取MP4的第一音轨，转换为16 kHz单声道PCM WAV；视频帧的显示时间戳（PTS）由FFprobe读取。对全部100条媒体核验后，音视频流起点、首视频帧PTS和首解码音频帧PTS均为0 s，因而WAV样本零点与视频PTS零点共同构成相对时间起点。

以容器时长和最后一帧PTS加一个名义帧间隔中的较大者作为有效时长$T_i$，第$i$条样本的媒体时间域为
@@EQ1@@
文本时间、WAV样本位置换算的秒数及视频帧PTS都在这一时间域中记录。为得到统一长度的序列，将每条样本自身的有效时长等分为$K=50$个公共窗，并保存各窗的实际秒数边界。由于视频长度不同，同一窗号的时间范围随样本时长变化。

图~\ref{fig:fig05_q1_data_statistics}中，视频时长以较短片段为主，同时包含少量较长片段。文本有效窗数的样本间差异较大，语音和视觉则较集中；文本内容主要采用赛题转写，时间戳回退涉及少数样本。这些差异在共同时间基准下保留，成为逐样本特征记录的一部分。

@@FIG2@@

\subsection{三模态原生特征提取}\label{subsec:q1_native}
有了共同的媒体时间坐标，就可以分别为词语、声学位置和采样画面建立时间支持。三种模态各自编码，原生序列保留自身采样密度，每一行向量与对应时间区间及原始来源一起保存。

\subsubsection{文本特征提取}\label{subsubsec:q1_text}
文本的内容和时间信息分别来自赛题转写与同一视频的语音识别。采用Whisper large-v3-turbo获取词级和片段级时间\cite{radford2023whisper}，将能够对应到识别词序的赛题词语连接到这些区间；未匹配词保留在原转写中，编码时使用已建立时间支持的词语。对于文本对应条件不满足的片段，采用媒体ASR内容。最终\QOneOfficialTextCount 条使用赛题转写，\QOneMediaAsrCount 条使用媒体ASR。例如，样本\texttt{\detokenize{-NFrJFQijFE__2}}的赛题转写描述仪器准备，而识别内容描述鹰的听觉和视觉，该样本采用媒体ASR，同时保留原转写。

词级区间须为有限值，满足$0\leq s<e\leq T_i$，且起止边界相对于前一原始词区间保持时间顺序。无效或非单调的词级区间回退到所属识别片段的有效范围；若该片段也无有效时间，则记录为未解决。时间来源和回退标记随文本记录保存，区分词级和片段级支持。

带时间信息的词语列表送入RoBERTa-base编码\cite{liu2019roberta}。快速分词器按预分词列表处理，最大长度为510词元；取最后一层隐藏表示，并按词索引平均同一词的子词向量，得到768维原生表示。特殊词元由分词器索引排除，截断后没有对应词元的单元保留提示记录。模型使用预训练权重进行推理。第$j$个文本特征及其时间支持为
@@EQ2@@
编码流程见图~\ref{fig:fig02_q1_text_feature_extraction}。

@@FIG3@@
\FloatBarrier

\subsubsection{语音特征提取}\label{subsubsec:q1_audio}
文本建立了语言内容的时间位置，语音则需将连续波形编码后的声学位置连接回WAV。使用WavLM-base-plus处理16 kHz单声道输入\cite{chen2022wavlm}，取最后一层逐位置表示，得到768维声学序列。

根据模型卷积核与步长，计算累计步距$H$和感受范围$R$。当WAV含$N$个样本时，第$j$个声学位置对应采样区间$[jH,\min(N,jH+R))$；区间端点除以采样率后转换为秒，并将末端限制在$T_i$以内。输出位置数与模型长度计算函数核对，原生向量同时记录其WAV采样范围。由此得到
@@EQ3@@
声学位置较密集，同一个公共窗内可能覆盖多个原生区间，后续通过时间交叠确定该窗的语音来源。特征提取流程见图~\ref{fig:fig03_q1_audio_feature_extraction}。

@@FIG4@@
\FloatBarrier

\subsubsection{视觉特征提取}\label{subsubsec:q1_vision}
文本和语音已有媒体时间区间，视觉特征进一步由采样画面的真实显示时刻定位。以每秒4帧建立目标时间网格，在PTS序列中选择离目标时刻最近的帧；等距时取较早帧，相邻目标选中同一帧时保留一次。所选画面按源帧编号解码，经配套图像处理器缩放、归一化后，送入SigLIP2-base-patch16-224\cite{tschannen2025siglip2}，得到768维视觉向量。

以相邻已选帧PTS的中点划分支持区间，首边界为0，尾边界为$T_i$。每个向量同时保留源帧编号和PTS：前者定位原视频画面，后者记录其显示时刻，支持区间则参与公共窗映射。视觉原生表示为
@@EQ4@@
全部视频的顺序解码帧与FFprobe记录的显示时间经核验保持一致，源帧编号与PTS可以联合回查视觉来源。提取流程见图~\ref{fig:fig04_q1_video_feature_extraction}。

@@FIG5@@
\FloatBarrier

\subsection{基于最大时间重叠的跨模态时序对齐}\label{subsec:q1_alignment}
经过独立提取，三种模态都具有原生向量和时间支持。跨模态对齐据此将它们映射为固定长度序列，使同一公共位置汇集来自相同媒体时间范围的文本、语音和视觉信息。

一个公共窗可与同一模态的多个原生区间交叠。本文选取正交叠时长最长的一个，让该窗直接继承对应向量；这样，每个有效位置都连接到一个确定的原生来源。

将第$i$条样本的有效时长均分为$K=\QOneBins$个半开区间：
@@EQ5@@
记$b_{ik}=kT_i/K$，模态$m\in\{T,A,V\}$的原生时间支持为
@@EQ6@@
其与公共窗的交叠长度为
@@EQ7@@
当存在正交叠时，选择
@@EQ8@@
并令
@@EQ9@@
最大交叠并列时，优先选择区间中心更接近公共窗中心者，仍并列则取较早原生索引。所选索引、时间支持和实际交叠时长写入来源记录。无正交叠时，令$\mathbf z_{ik}^{(m)}=\mathbf0$、$M_{ik}^{(m)}=0$，保留空来源；有效性由掩码$M_{ik}^{(m)}$表示。单样本及全量结果分别组织为
@@EQ10@@
图~\ref{fig:fig06_q1_temporal_alignment}展示不同时间密度的原生序列依据实际媒体时间进入同一公共时间轴的过程。

@@FIG6@@
\FloatBarrier

\subsection{特征构建结果与完整性检验}\label{subsec:q1_validation}
对全部\QOneSamples 条样本完成映射后，文本、语音和视觉各形成$100\times50\times768$的单精度浮点数组，有效掩码各为$100\times50$的布尔数组。三种模态的原生向量总数及平均覆盖率见表~\ref{tab:q1_feature_scheme}。

\input{generated/table_q1_feature.tex}

文本、语音和视觉的有效位置分别为4121、4955和5000，平均覆盖率为\QOneTextCoverage、\QOneAudioCoverage 和\QOneVisionCoverage。文本以离散词语和片段出现，部分公共窗内没有对应文本特征，由文本掩码标记；连续声学序列更密集，语音仅在少数位置出现空窗；视觉支持区间覆盖全部50窗。覆盖率差异由各模态的时间组织体现出来。

完整性检查确认100个样本编号唯一，矩阵行序与样本清单一致，三模态数组形状及掩码尺寸匹配，特征数值均有限，无效位置均为零。来源表包含全部15000个“样本—公共窗—模态”组合，每个有效位置均能连接到被选原生索引及原始来源。

\subsection{全量样本特征构建结果}\label{subsec:q1_all_samples}
各样本的媒体时长和语言内容不同，对应的特征覆盖也有所变化。表~\ref{tab:q1_all_samples}逐条列出100条样本的时长、模态、维度和对齐粒度，并保留三模态有效窗、文本来源及时间戳回退记录。

\input{generated/table_q1_all_samples.tex}

文本有效窗数为13--49，语音为47--50，视觉均为50，文本时间支持呈现出最明显的样本间差异。75条采用赛题转写，25条采用媒体ASR；7条样本包含时间戳回退，累计21个实际回退位置。逐样本记录因而同时呈现覆盖范围和文本时间支持的粒度，与总体平均覆盖率共同反映特征构建结果。
\FloatBarrier

\subsection{方案比较与信息保持能力验证}\label{subsec:q1_analysis}
在确认特征能够完整读取并回溯后，进一步比较不同方案对情感信息的保持能力。使用固定的轻量预测模型作为特征探针，在同一训练与评价设置下分别改变对齐方法、视觉编码器和模态组合，以准确率、宏平均F1、MAE及皮尔逊相关系数观察差异。

探针将每模态768维输入映射为32维，在公共位置拼接三路表示，经小型感知机变换和联合有效位均值池化，输出情感类别与连续强度。采用40轮AdamW训练，学习率0.002、权重衰减0.001，目标为交叉熵加0.25倍平滑绝对误差；标准化参数由各训练折的有效位置估计。评价为5次重复分层五折，表中报告均值与样本标准差。由于按片段标签分层、未按原视频分组，该实验用于Q1内部方案比较。

\subsubsection{时序对齐方法比较}\label{subsubsec:q1_align_ablation}
首先比较最大时间重叠、最近中心和重叠加权平均，结果见表~\ref{tab:q1_alignment_ablation}及图~\ref{fig:fig07_q1_alignment_method_comparison}。三种方案的平均指标总体接近，最近中心在部分指标上略高；同时，其\QOneNearestZeroOverlap 个有效文本公共窗选择了与目标窗零交叠的来源。

\input{generated/table_q1_alignment.tex}

重叠加权平均将多个原生向量合成为一个公共位置，来源随之变为多行向量及权重的组合。最大时间重叠则保留正时间交叠与单一来源，使每个公共位置能够直接回查到原生行。综合辅助评价、时间对应和回查方式，最终采用最大时间重叠对齐。

@@FIG7@@
\FloatBarrier

\subsubsection{视觉特征提取方法比较}\label{subsubsec:q1_vision_ablation}
确定时间映射后，比较不同视觉表示对情感信息的保持效果。固定采样帧、PTS及对齐规则，仅将SigLIP2替换为DINOv3\cite{simeoni2025dinov3}，得到表~\ref{tab:q1_visual_backbone}的结果。

\input{generated/table_q1_visual.tex}

SigLIP2的准确率、宏平均F1和相关系数均值更高，MAE均值更低，单次提取耗时也较短，因此选为最终视觉编码器。两组使用相同视觉来源，差异体现这些画面经不同骨干形成的表示效果。
\FloatBarrier

\subsubsection{模态组合比较}\label{subsubsec:q1_modality_ablation}
在所选对齐方法与视觉编码器下，再比较三个单模态、三种双模态及三模态组合，结果见表~\ref{tab:q1_modality_ablation}。各组合沿用相同公共窗与有效位，通过增减输入观察任务信息的变化。

\input{generated/table_q1_modality.tex}

文本单模态保留了较强的情感信息，加入语音后分类指标进一步提高；文本与视觉组合在连续强度预测的MAE和相关系数上均值较优。分类与强度预测各有优势组合，表明语言内容、声音和画面对两类任务的作用有所差异。三模态简单拼接的结果仍有融合改进空间。
\FloatBarrier

\subsection{特征文件组织与复现说明}\label{subsec:q1_repro}
上述结果分别保存为三模态特征、有效掩码、样本行序和逐窗来源四部分，文件组织见表~\ref{tab:q1_output_files}。模型标识、版本与文件校验信息随特征清单保存。

\input{generated/table_q1_files.tex}

读取时以样本编号确定矩阵行，再按模态取得特征与有效掩码。需要回查媒体来源时，按样本编号、公共窗索引及模态连接来源表，得到原生行号、时间区间与文本、WAV或源帧记录；原生向量和同名说明文件随结果一并保存。

复现流程依次为媒体信息核对、音频转换及文本时间获取、三模态原生编码、时间支持建立、公共窗划分和最大时间重叠映射，最后保存四部分产物并检查其对应关系。核心环境为PyTorch 2.14.0+cu130与Transformers 4.57.3，编码器及采样设置沿用前述配置。
\FloatBarrier

\subsection{典型样本分析与本问小结}\label{subsec:q1_case_summary}
样本\texttt{\detokenize{-a55Q6RWvTA__3}}的有效时长为22.154 s，划分为50个公共窗。表~\ref{tab:q1_case_mapping}列出第24--26窗，即零起始索引23--25的九条三模态映射记录。

\input{generated/table_q1_case_mapping.tex}

第25窗的范围为$[10.634,11.077)$ s。文本索引34的支持区间为10.900--11.120 s，语音索引542为10.840--10.865 s，视觉索引43为10.617--10.867 s。三个区间宽度不同，但都与该窗发生实际交叠，由此汇入同一个公共位置。

三个连续窗内，文本原生索引从31变为34、36，语音从520变为542、564，视觉从42变为43、45。索引变化速度反映各模态的采样密度；公共窗按媒体时间组织这些来源，同时保留其原有序列位置。

每个公共位置还可沿原生索引和时间区间回查文本词语、WAV样本或视频帧。例如，第25窗的视觉特征来自帧322，PTS为10.733 s，显示时刻与该特征的支持区间分别记录。这样，从统一序列可以逐层返回实际媒体来源。

本文从媒体时间出发构建三模态原生特征，通过最大时间重叠形成统一的50位序列，最终得到100条样本的特征、有效掩码和来源记录。辅助比较支持所采用的对齐与视觉编码方案，真实样本呈现了统一时间位置与文本、语音和画面来源之间的对应关系。
\FloatBarrier
'''
for i,e in enumerate(equations,1):article=article.replace(f'@@EQ{i}@@',e)
for i,(f,c) in enumerate(zip(figures,captions),1):
    f=re.sub(r'\\caption\{[^}]*\}',lambda _:r'\caption{'+c+'}',f)
    article=article.replace(f'@@FIG{i}@@',f)
assert '@@' not in article
(O/'sections/05_q1.tex').write_text(article,encoding='utf-8')
# Only shorten table notes. All 100 data rows and table formatting are retained.
p=O/'generated/table_q1_all_samples.tex'
t=p.read_text(encoding='utf-8')
start=t.index(r'\par\vspace{0.2em}')
t=t[:start]+r'''\par\vspace{0.2em}\noindent\begin{minipage}{\textwidth}{\footnotesize\noindent 注：\par
\begin{enumerate}[label=(\arabic*),leftmargin=2.4em,labelsep=.4em,itemsep=0pt,topsep=0pt,parsep=0pt]
\item T、A、V分别表示文本、语音、视觉；“有效窗(T/A/V)”依次给出三模态的有效公共窗数。
\item “赛题转写”为附件1 \texttt{label-100.xlsx} 的text字段；“媒体ASR”为当前视频语音识别得到的文本。
\item “回退”为词级时间戳不可用时，实际采用所属识别片段时间区间的映射次数。
\end{enumerate}}\end{minipage}
'''
p.write_text(t,encoding='utf-8')
p=O/'generated/table_q1_alignment.tex'
t=p.read_text(encoding='utf-8')
t=re.sub(r'\\par\\smallskip\\noindent\\footnotesize 注：[^\n]+',r'\\par\\smallskip\\noindent\\footnotesize 注：数值为均值$\\pm$样本标准差。',t)
p.write_text(t,encoding='utf-8')
(O/'generated/table_q1_files.tex').write_text(r'''\begin{table}[H]
\centering
\caption{对齐特征与来源文件组织}\label{tab:q1_output_files}
\small
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{@{}p{6.0cm}X>{\raggedright\arraybackslash}p{3.8cm}@{}}
\toprule
文件 & 内容 & 组织形式 \\
\midrule
\texttt{aligned\_features\_fp32.npz} & \texttt{text/audio/vision} & 各$100\times50\times768$，FP32 \\
\texttt{masks.npz} & 三模态有效掩码 & 各$100\times50$，布尔值 \\
\texttt{sample\_ids.csv} & 行号与样本编号 & 100条对应记录 \\
\texttt{source\_mapping\_15000.csv} & 逐公共窗来源 & 15000条记录 \\
\bottomrule
\end{tabularx}
\end{table}
''',encoding='utf-8')
# In this preview the old configuration table is replaced by the file table;
# its model/input/alignment settings remain in the narrative.
for i,args in enumerate([
    ['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q1_story_rewrite.tex'],
    ['bibtex','build/q1_story_rewrite'],
    ['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q1_story_rewrite.tex'],
    ['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q1_story_rewrite.tex']
]):
    r=subprocess.run(args,cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (O/'build'/f'compile_{i}.txt').write_bytes(r.stdout)
    if r.returncode:
        print(r.stdout.decode('utf-8',errors='replace')[-5000:])
        raise SystemExit(r.returncode)
shutil.copy2(O/'build/q1_story_rewrite.pdf',O/'q1_story_rewrite.pdf')
print('Compiled Q1 story preview.')
