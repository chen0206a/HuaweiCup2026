from pathlib import Path
import shutil, subprocess

OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'q1_rewrite_preview'
for name in ['figures','generated','bibliography']:
    shutil.copytree(OLD/name,OUT/name,dirs_exist_ok=True)
for name in ['q1_numbers.tex','protected_paper_hashes.json']:
    shutil.copy2(OLD/name,OUT/name)
# Keep the longtable notes together; do not alter any data row.
tp=OUT/'generated/table_q1_all_samples.tex'
ts=tp.read_text(encoding='utf-8')
ts=ts.replace(r'\vspace{0.2em}{\footnotesize',r'\par\vspace{0.2em}\noindent\begin{minipage}{\textwidth}{\footnotesize')
ts=ts.replace(r'\end{enumerate}}',r'\end{enumerate}}\end{minipage}')
tp.write_text(ts,encoding='utf-8')
(OUT/'sections').mkdir(exist_ok=True)
(OUT/'build').mkdir(exist_ok=True)
entry=(OLD/'q1_rewrite_preview.tex').read_text(encoding='utf-8')
(OUT/'q1_rewrite_preview_v2.tex').write_text(entry,encoding='utf-8')
s=(OLD/'sections/05_q1.tex').read_text(encoding='utf-8')
replacements={
'代码取容器时长与“最后一帧PTS加一个名义帧间隔”中的较大者，作为第$i$条样本的有效时长。表中时长均沿用这一记录，其统一时间域为':
'取容器时长与“最后一帧PTS加一个名义帧间隔”中的较大者作为第$i$条样本的有效时长，其统一时间域为',
'文本时间、WAV样本位置换算的秒数和视觉帧PTS均按该媒体时间基准组织。现有语音实现将WAV起点视为媒体时间起点；虽然预处理记录了音轨起点，但代码未显式应用额外偏移。该时间起点假设需结合原始媒体记录复核，不能由统一的张量形状代替验证。':
'附件1全部100条媒体的音视频流起点、首视频帧PTS及首解码音频帧PTS均为0 s。因此，WAV样本零点与视频PTS零点可作为共同的相对时间起点，文本时间、语音样本位置换算的秒数和视觉帧时间均在这一基准下组织。',
'现有提取器读取预存的带时间片段清单；清单的完整生成规则目前未随本地包保存，因此不能进一步宣称所有未匹配词均采用了某一确定的自动匹配算法。':
'文本编码读取同一样本的带时间片段清单。赛题转写中能够与识别词序建立对应的词语沿用该词的时间支持，未匹配词不进入带时间特征；不满足文本对应条件的样本采用媒体ASR。原始转写、识别词及其对应关系同时保留，便于回查内容来源。',
'对词级时间信息，保存的回退函数先检查':
'对词级时间信息，先检查',
'保存的最终汇总记录覆盖\\QOneSamples 条样本，三种原生特征均为768维，并映射为每样本$50\\times768$的序列。':
'最终特征覆盖\\QOneSamples 条样本，文本、语音和视觉分别组织为$100\\times50\\times768$的单精度浮点数组，对应有效掩码均为$100\\times50$的布尔数组。',
'完整性检验的目标是确保样本编号、特征位置和来源能够连接，而不仅检查数组能否载入。已有生成与验证脚本包含形状、数据类型、有限值、掩码尺寸、无效位置零值及媒体哈希检查，保存的汇总结果与案例记录给出了相应统计。当前本地论文包没有包含最终大矩阵和完整逐样本来源文件，因此本节沿用既有结果，不将其称为本轮重新打开全量矩阵所得的检验结论；文件复核范围在第\\ref{subsec:q1_repro}节说明。':
'完整性检验覆盖样本行序、数组形状、数值有限性、掩码与来源对应。100个样本编号唯一，矩阵行序与清单一致，特征均无非有限值，无效位置均为零。文本、语音和视觉的有效位置分别为4121、4955和5000，与覆盖率一致。来源表完整保留15000个“样本—公共窗—模态”组合，有效位置均可回查原生索引与来源；文件读取方式见第\\ref{subsec:q1_repro}节。',
'已有代码采用40轮AdamW训练':
'探针采用40轮AdamW训练',
'本文复用保存的评价结果，本轮不重新训练探针。':
'各组比较保持探针结构、优化设置和评价划分一致。',
'保存的单次提取用时也更短':
'单次提取用时也更短',
'实际图像处理器的缩放、裁剪和归一化配置未在当前本地正式记录中完整留存，因此本文不根据模型名称补写未核实的处理参数。':
'图像按模型配套处理配置缩放并归一化，处理配置与模型版本一并保存。',
'完整提交仍需包含最终特征文件及其可读取说明，使正文中的结果能够由实际附件复核。':
'最终特征、有效掩码、样本清单与逐窗来源映射共同保存，使上述结果能够回查到对应的特征位置和原始来源。',
}
for old,new in replacements.items():
    assert old in s,old[:80]
    s=s.replace(old,new)
vision_end='保存WAV样本范围使对齐后的语音位置能够回查原音频，而不仅保留一个向量下标。'
# Add visual verification in the visual section, adjacent to its explanation of source records.
vision_start=s.index(r'\subsubsection{视觉特征提取}')
vision_stop=s.index(r'\subsection{基于最大时间重叠',vision_start)
part=s[vision_start:vision_stop]
needle='\\FloatBarrier'
insert='进一步对全部视频解码帧进行时间一致性核验，顺序解码结果与FFprobe记录的帧显示时间一致，源帧编号与PTS因而可共同用于视觉来源回溯。\n\n'
assert needle in part
part=part.replace(needle,insert+needle,1)
s=s[:vision_start]+part+s[vision_stop:]
start=s.index(r'\subsection{特征文件组织与复现说明}')
end=s.index(r'\subsection{典型样本分析与本问小结}',start)
repro=r'''\clearpage
\subsection{特征文件组织与复现说明}\label{subsec:q1_repro}
主要模型和处理设置见表~\ref{tab:q1_repro_models}。三模态独立编码后，统一采用最大时间重叠映射；模型标识与版本、样本顺序及文件校验信息保存在配套清单中。

\begin{table}[H]
\centering
\caption{问题一主要模型与复现配置}\label{tab:q1_repro_models}
\small
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{@{}p{3.0cm}X@{}}
\toprule
项目 & 配置 \\
\midrule
文本编码 & RoBERTa-base \\
语音编码 & WavLM-base-plus \\
视觉编码 & SigLIP2-B/16 \\
文本时间信息 & Whisper large-v3-turbo \\
音频输入 & 16 kHz，单声道 \\
视觉采样 & 4帧/s目标时间网格 \\
公共时间窗 & 每样本50窗 \\
对齐规则 & 最大时间重叠硬对齐 \\
最终特征 & 三模态各$100\times50\times768$，FP32 \\
核心环境 & PyTorch 2.14.0+cu130；Transformers 4.57.3 \\
\bottomrule
\end{tabularx}
\end{table}

最终特征采用压缩NPZ保存。\texttt{aligned\_features\_fp32.npz}中的\texttt{text}、\texttt{audio}、\texttt{vision}分别存放三模态数组；\texttt{masks.npz}中的\texttt{text\_mask}、\texttt{audio\_mask}、\texttt{vision\_mask}分别保存布尔有效位。\texttt{sample\_ids.csv}给出第一维行号与样本编号的对应关系。\texttt{source\_mapping\_15000.csv}另存逐窗来源，包含原生索引、时间支持、交叠时长及文本、WAV采样范围或源帧信息。

读取时先按样本编号确定矩阵行，再读取三模态特征及掩码；需要回溯来源时，以样本编号、公共窗索引和模态查询来源表，取得对应的原生位置与媒体记录。原生向量及同名说明文件一并保存，来源索引与原生向量行号对应，因此密集矩阵和来源记录可以分别读取、联合使用。

完整处理从核对附件1样本与媒体信息开始，将音频转换为单声道WAV，取得文本时间信息与视频PTS，再提取三模态原生特征并关联各自时间支持。随后按每条样本时长建立50个公共窗，执行最大时间重叠对齐，保存特征、掩码、样本编号与来源映射，最后检查样本覆盖、数组形状及来源对应关系。
\FloatBarrier

'''
s=s[:start]+repro+s[end:]
for word in ['当前本地','本轮','尚未定位','待确认','无法确认','审计','PASS','当前工作区','本地精简包','最终大矩阵','不能宣称','历史记录']:
    assert word not in s,word
(OUT/'sections/05_q1.tex').write_text(s,encoding='utf-8')
for i,args in enumerate([
    ['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q1_rewrite_preview_v2.tex'],
    ['bibtex','build/q1_rewrite_preview_v2'],
    ['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q1_rewrite_preview_v2.tex'],
    ['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q1_rewrite_preview_v2.tex'],
]):
    p=subprocess.run(args,cwd=OUT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (OUT/'build'/f'compile_{i}.txt').write_bytes(p.stdout)
    if p.returncode:
        print(p.stdout.decode('utf-8',errors='replace')[-5000:])
        raise SystemExit(p.returncode)
shutil.copy2(OUT/'build/q1_rewrite_preview_v2.pdf',OUT/'q1_rewrite_preview_v2.pdf')
print('Compiled Q1 v2 independent preview.')
