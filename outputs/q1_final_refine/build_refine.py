from pathlib import Path
import shutil,re,subprocess,json,hashlib
S=Path(__file__).resolve().parent.parent/'q1_story_rewrite'
O=S.parent/'q1_final_refine'
O.mkdir(exist_ok=True)
for folder in ['figures','generated','bibliography']:
    shutil.copytree(S/folder,O/folder,dirs_exist_ok=True)
for name in ['q1_numbers.tex','protected_paper_hashes.json']:
    shutil.copy2(S/name,O/name)
(O/'build').mkdir(exist_ok=True)
(O/'qa').mkdir(exist_ok=True)
body=(S/'sections/05_q1.tex').read_text(encoding='utf-8')
body=body.replace(r'\mathcal T_i=[0,T_i).',r'\mathcal{T}_i=[0,T_i).')
body=body.replace('文本时间、WAV样本位置换算的秒数及视频帧PTS都在这一时间域中记录。',r'文本时间、WAV样本位置换算的秒数及视频帧PTS都在媒体时间域$\mathcal{T}_i$中记录。')
body=body.replace('带时间信息的词语列表送入RoBERTa-base编码',r'带时间信息的词语列表按图~\ref{fig:fig02_q1_text_feature_extraction}的流程送入RoBERTa-base编码')
body=body.replace('编码流程见图~\\ref{fig:fig02_q1_text_feature_extraction}。\n','')
body=body.replace('原生向量同时记录其WAV采样范围。由此得到','原生向量同时记录其WAV采样范围。声学特征及时间支持记为')
body=body.replace('特征提取流程见图~\\ref{fig:fig03_q1_audio_feature_extraction}。',r'原始波形到声学表示的处理见图~\ref{fig:fig03_q1_audio_feature_extraction}。')
body=body.replace('提取流程见图~\\ref{fig:fig04_q1_video_feature_extraction}。',r'采样与编码步骤见图~\ref{fig:fig04_q1_video_feature_extraction}。')
body=body.replace('最大交叠并列时，优先选择','\n实际实现中，最大交叠并列时，优先选择')
body=body.replace('使同一公共位置汇集来自相同媒体时间范围的文本、语音和视觉信息。',r'使同一公共位置汇集来自相同媒体时间范围的文本、语音和视觉信息（图~\ref{fig:fig06_q1_temporal_alignment}）。')
body=body.replace('图~\\ref{fig:fig06_q1_temporal_alignment}展示不同时间密度的原生序列依据实际媒体时间进入同一公共时间轴的过程。\n','')
body=body.replace('需要回查媒体来源时，按样本编号、公共窗索引及模态连接来源表，', '媒体来源可按样本编号、公共窗索引及模态连接来源表，')
body=body.replace('复现流程依次为媒体信息核对、音频转换及文本时间获取、三模态原生编码、时间支持建立、公共窗划分和最大时间重叠映射，最后保存四部分产物并检查其对应关系。','复现时依次核对媒体信息，转换音频并获取文本时间，完成三模态原生编码、时间支持建立、公共窗划分和最大时间重叠映射，再保存并检查四部分产物。')
# Explicit paragraph starts are determined here, not by float or heading defaults.
continuations=('记$b_{ik}', '其与公共窗', '当存在正交叠', '并令')
lines=[]; in_equation=False; starts=[]
for line in body.splitlines():
    if line.startswith(r'\begin{equation}'):
        in_equation=True
    if line and not line.startswith('\\') and not in_equation:
        if not line.startswith(continuations):
            starts.append(line[:50])
            line=r'\QOneParagraph '+line
            if line.startswith(r'\QOneParagraph 复现时'):
                line=r'\par\needspace{4\baselineskip}'+'\n'+line
    lines.append(line)
    if line.startswith(r'\end{equation}'):
        in_equation=False
body='\n'.join(lines)+'\n'
assert '\\end{equation}\n并令\n\\begin{equation}' in body
(O/'sections').mkdir(exist_ok=True)
(O/'sections/05_q1.tex').write_text(body,encoding='utf-8')
entry=(S/'q1_story_rewrite.tex').read_text(encoding='utf-8')
entry=entry.replace(r'\setlength{\parindent}{2em}',r'''\setlength{\parindent}{2em}
% Semantic new paragraphs: explicit indentation after headings, floats and displays.
\newcommand{\QOneParagraph}{\par\noindent\hspace*{2em}}''')
(O/'q1_final_refine.tex').write_text(entry,encoding='utf-8')
(O/'generated/table_q1_files.tex').write_text(r'''\begin{table}[H]
\centering
\caption{对齐特征与来源文件组织}\label{tab:q1_output_files}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.28\textwidth}@{\hspace{1.4em}}>{\raggedright\arraybackslash}p{0.56\textwidth}@{}}
\toprule
文件 & 内容与组织形式 \\
\midrule
\texttt{aligned\_features\_}\newline\texttt{fp32.npz} & \texttt{text/audio/vision}；各$100\times50\times768$，FP32 \\
\path{masks.npz} & 三模态有效掩码；各$100\times50$，布尔值 \\
\path{sample_ids.csv} & 行号与样本编号；100条对应记录 \\
\texttt{source\_mapping\_}\newline\texttt{15000.csv} & 逐公共窗来源；15000条记录 \\
\bottomrule
\end{tabular}
\end{table}
''',encoding='utf-8')
for i,args in enumerate([
    ['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q1_final_refine.tex'],
    ['bibtex','build/q1_final_refine'],
    ['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q1_final_refine.tex'],
    ['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','q1_final_refine.tex']]):
    r=subprocess.run(args,cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (O/'build'/f'compile_{i}.txt').write_bytes(r.stdout)
    if r.returncode: raise RuntimeError(r.stdout.decode('utf-8',errors='replace')[-3000:])
shutil.copy2(O/'build/q1_final_refine.pdf',O/'q1_final_refine.pdf')
(O/'qa/semantic_paragraphs.json').write_text(json.dumps(starts,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Compiled {O}; explicitly indented paragraphs: {len(starts)}')
