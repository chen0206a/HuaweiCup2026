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
code=['scripts/run_attachment3_final_inference.py','scripts/run_q3_attachment4_final.py','src/data/__init__.py','src/data/dataset.py','src/data/attachment3_inference.py','src/models/__init__.py','src/models/baseline.py','src/models/pooling_residual.py','src/evaluation/missing_benchmark.py','src/q3/__init__.py',*('src/q3/'+n for n in ['data_adapter.py','coalitions.py','evidence_grounding.py','faithfulness.py','frozen_predictor.py','text_grounding.py','text_feature_reconstruction.py','temporal_occlusion.py']),'configs/final/q2_b5_p2.yaml','configs/final/q2_b0_wce.yaml','configs/final/q3_heaf.yaml','configs/b5_pooling.yaml','configs/b5_p2_multiseed.yaml','outputs/metrics/b2_benchmark_definition.json','data/manifests/q2_missing_benchmark_manifest.json','outputs/final/q2/q2_checkpoint_manifest.json','src/data/preprocess.py','src/data/block_mask.py','src/utils/__init__.py','src/utils/metrics.py','src/models/temporal.py','src/training/__init__.py','src/training/evaluate.py','src/training/train.py','src/training/train_b1.py','src/training/run_b5_pooling.py','src/training/run_b5_p2_multiseed.py','outputs/final/q2/q2_model_lock.json','outputs/q3/q3_method_lock.json','outputs/q3/q3_explanation_schema.json','outputs/q3/q3_text_row_identity.json','outputs/q3/q3_av_grounding_audit.json','data/manifests/attachment3_sealed_inventory.json','data/manifests/q3/attachment4_inventory.json','outputs/final/q2/attachment3/attachment3_text_interface_audit.json']
for rel in code:cp(E/rel,'E2026/'+rel,'Q2/Q3 reproducibility')
# Both tasks share the same predictor parameters; check against the lock before packing.
ck=E/'outputs/checkpoints/b5_pooling_p2_best_robust_score.pt';assert h(ck)=='cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff'
cp(ck,'E2026/outputs/checkpoints/'+ck.name,'Q2/Q3 shared checkpoint')
# All three paired initialization checkpoints are part of the final Q2 evidence.
checkpoint_manifest=json.loads((E/'outputs/final/q2/q2_checkpoint_manifest.json').read_text(encoding='utf-8'))
for entry in checkpoint_manifest['checkpoints']:
 info=entry['checkpoint'];src=E/info['relative_path']
 assert h(src)==info['sha256'],src
 if src!=ck and not (DEST/'E2026'/info['relative_path']).exists():
  cp(src,'E2026/'+info['relative_path'],'Q2 paired-seed parameter file')
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
(DEST/'environment.json').write_text(json.dumps({'packaging_runtime_observed':vers,'historical_training_runtime':'see recorded configs and Q1 environment file','external_pretrained_weights':'not bundled under 50MB limit; retrieve pinned revisions in source/configs and verify hashes'},ensure_ascii=False,indent=2),encoding='utf-8')
record(DEST/'environment.json','environment','local metadata')
cp(ROOT/'outputs/appendix_handoff_pack/environment/q1_software_versions.txt','Q1/environment_recorded.txt','Q1 historical recorded environment')
readme='''# E题附件文件说明

## 文件目录
- `Q1/final/`：100条样本的三模态FP32特征、有效位掩码、样本ID和特征清单。
- `Q1/trace/`：逐样本时间与来源记录、15,000行来源映射。
- `Q1/src/`、`Q1/scripts/`、`Q1/configs/`：特征提取与时序对齐代码和配置。
- `E2026/src/`、`E2026/scripts/`、`E2026/configs/`：情感预测、缺失场景评价与解释算法的代码和配置。
- `E2026/outputs/checkpoints/`：三次初始化的基线与最终模型参数；Q2和Q3共用最终预测器。
- `results/attachment3_predictions.csv`：30条样本的预测结果。
- `results/attachment4_predictions_explanations.csv`：20条样本的预测与解释结果。
- `Q1/environment_recorded.txt`、`environment.json`：已记录的软件版本和打包环境信息。
- `RUNBOOK.md`：读取方法、目录配置与运行步骤。
- `MANIFEST.json`：文件大小与SHA256。

运行使用赛题提供的原始附件数据；公开预训练编码器按配置中指定的模型与修订版本获取。
'''
(DEST/'README.md').write_text(readme,encoding='utf-8');record(DEST/'README.md','instructions','generated')
runbook='''# 核心材料读取与运行说明

## Q1
`Q1/final/sample_ids.csv`按行列出100个ID；`aligned_features_fp32.npz`包含三个形状各为(100,50,768)的float32矩阵，`masks.npz`包含对应有效位。下面的代码只读成品，不重新抽取：

```python
import numpy as np
x = np.load('Q1/final/aligned_features_fp32.npz', allow_pickle=False)
m = np.load('Q1/final/masks.npz', allow_pickle=False)
print(x.files, m.files)
```

`trace/source_mapping_15000.csv`与`trace/aligned_json/`记录原生索引、时间支持和原始来源。重复生成这些成品需要赛题附件1原视频、配置指定的公开编码器权重与原实验的转写/时间支持记录；所需的原视频及时间支持由赛题素材和已有记录提供。Q1代码从`Q1`目录执行，设置`PYTHONPATH=src`，使用`configs/q1_full.yaml`。Q1历史已记录软件版本见`Q1/environment_recorded.txt`，未记录项不推断。

## Q2
将赛题附件2对应`aligned_50.pkl`放在`E2026/data/raw/`，最终模型参数位于`E2026/outputs/checkpoints/`。冻结的训练与评价协议见`E2026/configs/final/q2_b5_p2.yaml`、`E2026/outputs/metrics/b2_benchmark_definition.json`。三次初始化每个B0/P2参数对均已保留，SHA256见`q2_checkpoint_manifest.json`。源码包括训练、掩码构造、54场景评价、最终推理。历史配置中的服务器绝对路径是原运行记录；在新机器上只将数据、模型和输出路径指向新的实际位置，不改变其它参数或数据划分。

## Q3与专项预测
从`E2026`目录运行，先放置官方附件3/4 aligned数据与相关审计所需文件；运行入口分别为：

```
python scripts/run_attachment3_final_inference.py
python scripts/run_q3_attachment4_final.py
```

这两个脚本的锁定协议及预检清单位于`E2026/configs/final/`、`E2026/outputs/q3/`、`E2026/data/manifests/`。附件3文本特征重建使用源码中固定修订版的公开BERT权重与tokenizer；权重未随附件提交，需要在本地缓存并通过源码内SHA256核对。附件4音频/视觉原始时间映射未验证，CSV相关字段为NA。

## 环境与依赖
可从源码导入关系看到核心依赖：Python、NumPy、PyTorch、PyYAML、scikit-learn、transformers、tokenizers、huggingface-hub、jsonschema，以及Q1特征提取所需的音视频工具。软件版本记录见`environment.json`和`Q1/environment_recorded.txt`。原媒体提取与训练使用赛题原始输入及配置指定的公开预训练权重。
'''
(DEST/'RUNBOOK.md').write_text(runbook,encoding='utf-8');record(DEST/'RUNBOOK.md','reproduction instructions','generated')

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
audit={'status':'PACKAGE_ASSEMBLED_QA_PASSED','archive_bytes':size,'limit_bytes':50_000_000,'q1_samples':100,'q1_trace_json':len(trace),'q1_source_rows':15000,'attachment3_rows':30,'attachment4_rows':20,'checkpoint_sha256':h(ck),'identity_scan_matches':len(findings),'file_count':len(manifest),'known_reproducibility_limits':['official raw media and Attachment2/3/4 inputs excluded','large pretrained encoders excluded by size limit','historical training environment only partially documented']}
(ROOT/'outputs/competition_E2026_submission_build_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
assert ZIP.stat().st_size<=50_000_000
with zipfile.ZipFile(ZIP) as z:
 assert z.testzip() is None
 for row in manifest:assert hashlib.sha256(z.read(row['file'])).hexdigest()==row['sha256'],row['file']
 assert z.read('results/attachment3_predictions.csv')==q2.read_bytes()
 assert z.read('results/attachment4_predictions_explanations.csv')==q3.read_bytes()
print(json.dumps(audit,ensure_ascii=False,indent=2))
