from pathlib import Path
import zipfile,hashlib,json,csv,io,shutil,os,re,sys
import numpy as np
ROOT=Path('D:/华为杯');Q=ROOT/'Q1_方法流程与实验结果_交付包';E=ROOT/'E2026';DEST=ROOT/'outputs/competition_E2026_submission';ZIP=ROOT/'outputs/competition_E2026_submission.zip';QZIP=Path('D:/java录屏/q1_completion_20260926.zip')
if DEST.exists():
 assert DEST.resolve()==(ROOT/'outputs/competition_E2026_submission').resolve()
 shutil.rmtree(DEST)
DEST.mkdir(parents=True)
manifest=[]
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):x.update(c)
 return x.hexdigest()
def record(p,category,source):manifest.append({'file':p.relative_to(DEST).as_posix(),'category':category,'bytes':p.stat().st_size,'sha256':h(p),'source':source})
def cp(src,rel,category):
 src=Path(src);assert src.is_file(),src
 dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);record(dst,category,src.name);return dst
with zipfile.ZipFile(QZIP) as z:
 names=z.namelist()
 required=['final/aligned_features_fp32.npz','final/masks.npz','final/sample_ids.csv','final/feature_manifest.json','final/README.md','trace/source_mapping_15000.csv']
 trace=sorted(n for n in names if n.startswith('trace/aligned_json/') and n.endswith('.json'))
 assert len(trace)==100
 for n in required+trace:
  dst=DEST/'Q1'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read(n));record(dst,'Q1 feature/trace',n)
# Exact frozen feature contract and source mappings.
feat=np.load(DEST/'Q1/final/aligned_features_fp32.npz',allow_pickle=False);mask=np.load(DEST/'Q1/final/masks.npz',allow_pickle=False)
assert len(feat.files)==3 and all(feat[k].shape==(100,50,768) and feat[k].dtype==np.float32 and np.isfinite(feat[k]).all() for k in feat.files)
assert len(mask.files)==3 and all(mask[k].shape==(100,50) for k in mask.files)
with (DEST/'Q1/final/sample_ids.csv').open(encoding='utf-8-sig',newline='') as f:ids=list(csv.reader(f))
with (DEST/'Q1/trace/source_mapping_15000.csv').open(encoding='utf-8-sig',newline='') as f:source=list(csv.reader(f))
assert len(ids)-1==100 and len(source)-1==15000
for rel in ['src/q1/temporal.py','src/q1/timestamp_fallback.py',*('src/q1/extractors/'+n for n in ['__init__.py','base.py','registry.py','text.py','audio.py','vision.py']),'src/q1/__init__.py','scripts/q1_full_feature_run.py','scripts/prepare_q1_cpu.py','scripts/build_q1_text_final.py','scripts/build_q1_final_local.py','scripts/verify_q1_final_local.py','configs/q1_full.yaml','configs/q1_feature_interface.schema.json']:
 cp(Q/rel,'Q1/'+rel,'Q1 reproducibility')
code=['scripts/run_attachment3_final_inference.py','scripts/run_q3_attachment4_final.py','src/data/__init__.py','src/data/dataset.py','src/data/attachment3_inference.py','src/models/__init__.py','src/models/baseline.py','src/models/pooling_residual.py','src/evaluation/missing_benchmark.py','src/q3/__init__.py',*('src/q3/'+n for n in ['data_adapter.py','coalitions.py','evidence_grounding.py','faithfulness.py','frozen_predictor.py','text_grounding.py','text_feature_reconstruction.py','temporal_occlusion.py']),'configs/final/q2_b5_p2.yaml','configs/final/q3_heaf.yaml','outputs/final/q2/q2_model_lock.json','outputs/q3/q3_method_lock.json','outputs/q3/q3_explanation_schema.json','outputs/q3/q3_text_row_identity.json','outputs/q3/q3_av_grounding_audit.json','data/manifests/attachment3_sealed_inventory.json','data/manifests/q3/attachment4_inventory.json','outputs/final/q2/attachment3/attachment3_text_interface_audit.json']
for rel in code:cp(E/rel,'E2026/'+rel,'Q2/Q3 reproducibility')
# Both tasks share the same predictor parameters; check against the lock before packing.
ck=E/'outputs/checkpoints/b5_pooling_p2_best_robust_score.pt';assert h(ck)=='cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff'
cp(ck,'E2026/outputs/checkpoints/'+ck.name,'Q2/Q3 shared checkpoint')
q2=cp(E/'outputs/final/q2/attachment3/attachment3_predictions.csv','results/attachment3_predictions.csv','official prediction CSV')
q3=cp(E/'outputs/q3/final/attachment4_predictions_explanations.csv','results/attachment4_predictions_explanations.csv','official prediction/explanation CSV')
for p,n in [(q2,30),(q3,20)]:
 with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==n and len({r['sample_id'] for r in rows})==n
# Supplementary machine-readable output retains detail without altering official CSVs.
# Collect observed package versions; these describe the present local runtime, not historical training hardware.
mods=['python','numpy','torch','transformers','tokenizers','huggingface-hub','PyYAML','jsonschema','scikit-learn','pandas','librosa','opencv-python','pymupdf']
import importlib.metadata as im
def installed_version(name):
 if name=='python':return sys.version.split()[0]
 try:return im.version(name)
 except im.PackageNotFoundError:return 'not installed in packaging environment'
vers={m:installed_version(m) for m in mods}
(DEST/'environment.json').write_text(json.dumps({'packaging_runtime_observed':vers,'historical_training_runtime':'see configs/lock files; do not infer from packaging runtime','external_pretrained_weights':'not bundled under 50MB limit; retrieve pinned revisions in source/configs and verify hashes'},ensure_ascii=False,indent=2),encoding='utf-8')
record(DEST/'environment.json','environment','local metadata')
readme='''# E题竞赛附件（匿名整理版）

本包仅含竞赛附件材料，不含论文封面、参赛单位、队号或队员姓名。官方附件1–4原始视频/输入数据由赛题提供，不在本包重复提交。

## 文件
- `Q1/final/aligned_features_fp32.npz`：100样本、文本/语音/视觉各(100,50,768) float32；`masks.npz`是对应有效位；`sample_ids.csv`定行序，`feature_manifest.json`给元信息。
- `Q1/trace/source_mapping_15000.csv`：100×50×3来源映射；`trace/aligned_json/`为100个逐样本时间与来源记录。没有改写任何特征值、掩码或来源数据。
- `Q1/src`、`Q1/scripts`、`Q1/configs`：特征构建与最大重叠对齐核心代码；从`Q1`目录运行并以`src`加入PYTHONPATH。赛题原始视频、预训练编码器、词级时间支持等上游输入须按配置与原始实验记录提供；本包内成品特征可直接读取。
- `E2026/outputs/checkpoints/b5_pooling_p2_best_robust_score.pt`：Q2/Q3共用的最终预测器，SHA256见MANIFEST.json；Q3没有额外训练的模型参数。
- `E2026/src`、`scripts`、`configs`、`outputs/*lock*`：Q2预测、Q3解释的核心算法/固定协议和接口代码。运行命令从`E2026`目录执行：`python scripts/run_attachment3_final_inference.py` 或 `python scripts/run_q3_attachment4_final.py`。两脚本还要求原赛题附件3/4数据放在既定输入目录，且需要其内部审计文件、模型缓存/环境和固定清单；不要在缺文件时改模型或自行重算。
- `results/attachment3_predictions.csv`：附件3无标签预测，30条。
- `results/attachment4_predictions_explanations.csv`：附件4无标签预测与解释，20条。A/V原始媒体时间未验证，相关字段为NA；没有根据附件4真实标签评价。
- `environment.json`：当前打包机可观察的软件版本；不冒充历史训练环境。
- `MANIFEST.json`：逐文件SHA256、大小、来源。

## 复现边界
原始媒体、附件2/3/4官方输入以及大型公开预训练权重不重复装入附件。Q1预训练编码器由`Q1/configs/q1_full.yaml`指定。附件3文本重构采用源码中固定的`bert-base-uncased`修订版及权重SHA256；须自行按公开来源下载，并在运行前通过哈希门槛。历史模型训练硬件/库的完整锁文件未在本包中找到，不将打包机环境伪称训练环境。提交CSV为现有最终输出的逐字节副本，不需要重新推理才能使用。

## 大小和匿名
压缩包须≤50,000,000字节；校验结果见`PACKAGE_AUDIT.json`。身份词扫描仅覆盖文件内容与文件名的已知参赛单位、队号、队员姓名和本机用户名，不替代人工最终审核。
'''
(DEST/'README.md').write_text(readme,encoding='utf-8');record(DEST/'README.md','instructions','generated')
# Anonymity scan, including file names and binary contents of document-like files.
terms=['武汉轻工大学','26104960057','路晨','宋文杰','徐丽','17299','C:/Users/17299','C:\\Users\\17299']
findings=[]
for p in DEST.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(DEST).as_posix()
 if any(t in rel for t in terms):findings.append((rel,'filename'))
 if p.suffix.lower() in {'.py','.md','.yaml','.yml','.json','.jsonl','.csv','.txt'}:
  s=p.read_text(encoding='utf-8-sig',errors='replace')
  for t in terms:
   if t in s:findings.append((rel,t))
assert not findings,findings[:20]
# Build twice: audit is inside the archive; check all digests and CSVs again after extraction.
manifest.sort(key=lambda x:x['file'])
(DEST/'MANIFEST.json').write_text(json.dumps({'official_problem':'2026 E','files':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted(DEST.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(DEST).as_posix())
size=ZIP.stat().st_size
assert size<=50_000_000,('size exceeds official cap',size)
audit={'status':'PACKAGE_ASSEMBLED_QA_PASSED','archive_bytes':size,'archive_MB_decimal':round(size/1e6,3),'limit_bytes':50_000_000,'q1_samples':100,'q1_trace_json':len(trace),'q1_source_rows':15000,'attachment3_rows':30,'attachment4_rows':20,'checkpoint_sha256':h(ck),'identity_scan_matches':len(findings),'file_count':len(manifest),'known_reproducibility_limits':['official raw media and Attachment2/3/4 inputs excluded','large pretrained encoders excluded by size limit','historical training environment only partially documented']}
(DEST/'PACKAGE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(ZIP,'a',zipfile.ZIP_DEFLATED,compresslevel=9) as z:z.write(DEST/'PACKAGE_AUDIT.json','PACKAGE_AUDIT.json')
assert ZIP.stat().st_size<=50_000_000
with zipfile.ZipFile(ZIP) as z:
 assert z.testzip() is None
 for row in manifest:assert hashlib.sha256(z.read(row['file'])).hexdigest()==row['sha256'],row['file']
 assert z.read('results/attachment3_predictions.csv')==q2.read_bytes()
 assert z.read('results/attachment4_predictions_explanations.csv')==q3.read_bytes()
print(json.dumps(audit,ensure_ascii=False,indent=2))
