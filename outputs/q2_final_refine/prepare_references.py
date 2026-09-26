"""Prepare checked bibliography from primary publication metadata."""
from pathlib import Path
import json,re
O=Path(__file__).resolve().parent
metadata=json.loads((O/'qa/reference_metadata.json').read_text(encoding='utf-8'))
old=(O.parent/'q2_story_rewrite/q2_story_rewrite.tex').read_text(encoding='utf-8')
configs=[
 ('TFN','q2tfn','Zadeh A, Chen M, Poria S, et al.','Proceedings of EMNLP','2017','1103--1114','增加DOI，原文献身份一致'),
 ('LMF','q2lmf','Liu Z, Shen Y, Lakshminarasimhan V B, et al.','Proceedings of ACL, Volume 1','2018','2247--2256','增加DOI，原文献身份一致'),
 ('MFN','q2mfn','Zadeh A, Liang P P, Mazumder N, et al.','Proceedings of the AAAI Conference on Artificial Intelligence','2018, 32(1)','5634--5641','补全卷期和DOI，页码由官方PDF首尾页确认'),
 ('MulT','q2mult','Tsai Y H H, Bai S, Liang P P, et al.','Proceedings of ACL','2019','6558--6569','增加DOI，原文献身份一致'),
 ('MISA','q2misa','Hazarika D, Zimmermann R, Poria S.','Proceedings of ACM Multimedia','2020','1122--1131','增加DOI，原文献身份一致'),
 ('Self-MM','q2selfmm','Yu W, Xu H, Yuan Z, et al.','Proceedings of the AAAI Conference on Artificial Intelligence','2021, 35(12)','10790--10797','原引用误用了MMIM论文；更正为Self-MM原论文'),
 ('MMIM','q2mmim','Han W, Chen H, Poria S.','Proceedings of EMNLP','2021','9180--9192','原题名和作者错配；更正为层级互信息最大化原论文'),
 ('TFR-Net','q2tfr','Yuan Z, Li W, Xu H, et al.','Proceedings of ACM Multimedia','2021','4400--4407','末页由4408更正为4407并补DOI'),
 ('MissModal','q2missmodal','Lin R, Hu H.','Transactions of the Association for Computational Linguistics','2023, 11','1686--1702','原引用为另一篇缺失情感识别论文；更正为MissModal原论文'),
 ('M3S','q2m3s','Chi H, Yang M, Zhu J, et al.','Proceedings of AACL-IJCNLP, Volume 1','2022','121--130','原题名、作者、会议和年份均错配；以本地适配引用的Missing Modality meets Meta Sampling为准'),
 ('MMIN','q2mmin','Zhao J, Li R, Jin Q.','Proceedings of ACL-IJCNLP, Volume 1','2021','2608--2618','作者Qin Jin的姓为Jin，原Qin J更正为Jin Q，并补DOI'),
]
mechanisms={
 'TFN':'增广三路表示的外积融合；本地采用预计算对齐特征，文本递归编码、语音视觉池化。',
 'LMF':'模态专属低秩因子；本地为rank=8低秩融合。',
 'MFN':'模态内递归记忆、跨记忆注意力及门控记忆；本地使用三路LSTMCell。',
 'MulT':'定向跨模态注意力；本地为六路跨注意力及三路模态内时序记忆。',
 'MISA':'共享与模态特有表示；本地保留重构、差异约束和CMD约束，编码接口适配。',
 'Self-MM':'多任务与单模态辅助目标；本地采用标签与融合预测各半的单模态伪目标，并非原动态权重机制的完整复刻。',
 'MMIM':'层级互信息约束；本地以模态间和融合—单模态InfoNCE作为约束，不等同于原文全部BA/CPC估计器。',
 'TFR-Net':'缺失条件下特征重构；本地为时序上下文与潜在表示SmoothL1重构，不宣称原仓库完整运行。',
 'MissModal':'完整/缺失表征的几何、分布及情感语义对齐；本地用对比项、均值/标准差代理和分类KL实现。',
 'M3S':'缺失模态元采样；本地以LMF为骨干，完整输入支持更新及缺失查询的一步一阶元更新，非另一篇multi-head/meta-mining论文。',
 'MMIN':'缺失模态潜在表示推断；本地含残差细化和循环约束，非原模型全部CRA实现。',
}
records=[]
for name,key,shortauthors,venue,year,pages,reason in configs:
    m=metadata[name]
    if 'citation_metadata' in m:
        c=m['citation_metadata']; title=c['citation_title'][0]; authors=c['citation_author']; doi=c['citation_doi'][0]
    else:
        title=m['title']+(': '+m['subtitle'][0] if m['subtitle'] else '')
        authors=m['authors']; doi=m['doi']
    url=m.get('url',m['retrieval_url'])
    if name in ['MISA','TFR-Net']: url='https://dl.acm.org/doi/'+doi
    typ='J' if name=='MissModal' else 'C'
    sep='//'+venue if typ=='C' else '. '+venue
    tex=r'\bibitem{'+key+'} '+shortauthors+' '+title+'['+typ+']'+sep+'. '+year+': '+pages+r'. DOI: \url{'+doi+'}.'
    current=re.search(r'\\bibitem\{'+key+r'\} ([^\n]*)',old)[1]
    records.append(dict(method=name,key=key,title=title,authors=authors,venue=venue,year=year,pages=pages.replace('--','–'),doi=doi,official_url=url,metadata_url=m['retrieval_url'],current=current,reason=reason,adaptation=mechanisms[name],tex=tex))
(O/'qa/references_verified.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# Q2 原始文献与接口适配核对','',
 '核对日期：2026-09-27。以ACL Anthology、AAAI出版页面/原始PDF及ACM出版者存入Crossref的DOI元数据为依据。ACM页面访问受限时，明确采用出版者登记元数据，不以二手综述替代。', '',
 '本稿保留11种公开方法。名称对应原论文；表10描述的是本文统一特征接口下的实际适配机制，不宣称原仓库模型或原论文结果的直接复现。参考文献编号仅为独立Q2稿的局部编号，全文合并后应按首次引用顺序统一生成。','']
for r in records:
    lines += ['## '+r['method'],'', '**当前引用：** '+r['current'],'', '**核对后：** '+r['title'],
              '', '- 作者：'+ '; '.join(r['authors']),'- 会议/期刊：'+r['venue'],'- 年份/卷期：'+r['year'],'- 页码：'+r['pages'],
              '- DOI：'+r['doi'],'- 官方来源：['+r['method']+'原论文]('+r['official_url']+')',
              '- 元数据入口：[出版元数据]('+r['metadata_url']+')','- 修订：'+r['reason'],'', '**与代码对应：** '+r['adaptation'],'']
lines += ['## 实现证据','',
 '`E2026/src/models/public_baselines.py`：TFN、MulT、MISA。',
 '`E2026/src/models/public_baselines_extended.py`：LMF、MFN、Self-MM、MMIM、TFR-Net、MissModal、MMIN、M3S。',
 '`E2026/scripts/run_public_baseline_extended.py`：M3S支持/查询一步一阶更新与其他辅助损失训练路径。',
 'M3S本地adaptation notes明确指向AACL 2022原论文；模型类继承低秩骨干，训练器执行元更新，文献身份与这条实现路线一致。',
 '', 'MFN页码补充依据：[AAAI官方PDF](https://ojs.aaai.org/index.php/AAAI/article/download/12021/11880)，首尾页为5634和5641。']
(O/'q2_reference_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Prepared 11 verified references.')
