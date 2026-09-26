from pathlib import Path
import csv, json, hashlib, zipfile
from collections import Counter
import numpy as np

OUT=Path(__file__).resolve().parent
B=OUT/'evidence/completion'
def readj(p): return json.loads(p.read_text(encoding='utf-8'))
def readcsv(p): return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
hash_rows=readcsv(B/'audit/bundle_sha256.csv')
for r in hash_rows:
    p=B/r['path']
    assert p.is_file() and p.stat().st_size==int(r['bytes']) and sha(p)==r['sha256'],r['path']
manifest=readj(B/'final/feature_manifest.json')
for name in ['aligned_features_fp32.npz','masks.npz','sample_ids.csv']:
    assert sha(B/'final'/name)==manifest['files'][name]['sha256'],name
idrows=readcsv(B/'final/sample_ids.csv')
ids=[r['sample_id'] for r in idrows]
assert len(ids)==len(set(ids))==100
assert [int(r['index']) for r in idrows]==list(range(100))
assert ids==manifest['sample_order']
with np.load(B/'final/aligned_features_fp32.npz',allow_pickle=False) as z:
    assert set(z.files)=={'text','audio','vision'}
    features={m:z[m] for m in z.files}
with np.load(B/'final/masks.npz',allow_pickle=False) as z:
    assert set(z.files)=={'text_mask','audio_mask','vision_mask'}
    masks={m:z[m+'_mask'] for m in features}
stats={}
for m,x in features.items():
    mask=masks[m]
    assert x.shape==(100,50,768) and x.dtype==np.float32 and np.isfinite(x).all()
    assert mask.shape==(100,50) and mask.dtype==np.bool_
    assert np.all(x[~mask]==0)
    stats[m]={'shape':list(x.shape),'dtype':str(x.dtype),'finite':True,'mask_shape':list(mask.shape),
              'mask_dtype':str(mask.dtype),'valid_positions':int(mask.sum()),'coverage':float(mask.mean()),
              'invalid_rows_zero':True,'valid_all_zero_rows':int(np.all(x[mask]==0,axis=-1).sum())}
assert [stats[m]['valid_positions'] for m in ['text','audio','vision']]==[4121,4955,5000]
rows=readcsv(B/'trace/source_mapping_15000.csv')
assert len(rows)==15000
lookup={(r['sample_id'],int(r['window_index']),r['modality']):r for r in rows}
assert len(lookup)==15000
summ=readcsv(OUT.parent/'q1_rewrite_preview/evidence/results/05_appendix/q1_sample_summary_100.csv')
assert [r['sample_id'] for r in summ]==ids
native_stats=Counter()
source_counts=Counter()
fallback_counts={}
for i,sid in enumerate(ids):
    trace=readj(B/'trace/aligned_json'/f'{sid}.json')
    timeline=readj(B/'timeline'/f'{sid}.json')
    assert trace['sample_id']==timeline['sample_id']==sid
    assert len(trace['bins'])==len(timeline['bins'])==50
    assert trace['source_mp4_sha256']==timeline['source_sha256']==summ[i]['source_mp4_hash']
    assert float(summ[i]['duration'])==timeline['duration']
    with np.load(B/'trace/alignment_source_npz'/f'{sid}.npz',allow_pickle=False) as z:
        for m in features:
            assert np.array_equal(z[m+'_values'],features[m][i])
            assert np.array_equal(z[m+'_valid'],masks[m][i])
    for m,x in features.items():
        native=np.load(B/'native'/m/f'{sid}.npy',allow_pickle=False)
        meta=readj(B/'native'/m/f'{sid}.json')
        assert native.ndim==2 and native.shape[1]==768 and native.dtype==np.float32 and np.isfinite(native).all()
        assert len(meta['records'])==len(native)
        if m=='text':
            fallback_counts[sid]=sum('segment' in r.get('timestamp_source','') for r in meta['records'])
            assert fallback_counts[sid]==int(summ[i]['timestamp_fallback_count'])
        native_stats[m]+=len(native)
        assert int(summ[i][m+'_valid_bins'])==int(masks[m][i].sum())
        for w in range(50):
            r=lookup[(sid,w,m)]
            valid=r['valid']=='True'
            assert int(r['row_index'])==i and valid==bool(masks[m][i,w])
            bb=trace['bins'][w]
            assert float(r['window_start_seconds'])==bb['start'] and float(r['window_end_seconds'])==bb['end']
            mapping=bb['modalities'][m]
            if not valid:
                assert not mapping and r['native_feature_index']=='' and float(r['actual_overlap_seconds'])==0
                continue
            assert len(mapping)==1
            a=mapping[0]; ni=int(r['native_feature_index']); nr=meta['records'][ni]
            assert ni==a['feature_index']==nr['feature_index']
            assert np.array_equal(x[i,w],native[ni])
            s,e=float(r['source_support_start_seconds']),float(r['source_support_end_seconds'])
            assert [s,e]==a['source_interval'] and s==nr['start'] and e==nr['end']
            overlap=max(0,min(bb['end'],e)-max(bb['start'],s))
            assert overlap>0 and abs(overlap-float(r['actual_overlap_seconds']))<1e-12
            assert r['source_mp4_sha256']==trace['source_mp4_sha256']
            if m=='audio':
                assert int(r['wav_sample_start'])==a['source']['wav_sample_start']
                assert int(r['wav_sample_end'])==a['source']['wav_sample_end']
            if m=='text':
                assert r['text']==nr['text']
                assert r['text_feature_source']==a['source']['text_feature_source']
                assert r['timestamp_source']==nr['timestamp_source']
                assert json.loads(r['transcript_span_json'])==a['source']['transcript_span']
                assert json.loads(r['source_words_json'])==a['source']['source_words']
            if m=='vision':
                assert json.loads(r['frame_indices_json'])==a['source']['frame_indices']
                assert json.loads(r['frame_pts_seconds_json'])==a['source']['frame_pts_seconds']
    source_counts[summ[i]['text_feature_source']]+=1
assert dict(native_stats)=={'text':1847,'audio':38776,'vision':3205}
time=readcsv(B/'audit/timebase_validation_100.csv')
assert len(time)==100 and {r['sample_id'] for r in time}==set(ids)
cols=['video_stream_start_seconds','audio_stream_start_seconds','first_video_frame_pts_seconds','first_decoded_audio_frame_pts_seconds']
assert all(float(r[k])==0 for r in time for k in cols)
assert all(r['ffprobe_frame_count']==r['opencv_decoded_frame_count'] for r in time)
frames=sum(int(r['ffprobe_frame_count']) for r in time)
maxerr=max(float(r['max_opencv_vs_ffprobe_pts_error_seconds']) for r in time)
assert frames==23241 and maxerr<0.001
cases=[lookup[('-a55Q6RWvTA__3',w,m)] for w in [23,24,25] for m in ['text','audio','vision']]
expect=[(31,'10.300','10.520','0.220'),(520,'10.400','10.425','0.025'),(42,'10.367','10.617','0.250'),
        (34,'10.900','11.120','0.177'),(542,'10.840','10.865','0.025'),(43,'10.617','10.867','0.233'),
        (36,'11.320','11.700','0.200'),(564,'11.280','11.305','0.025'),(45,'11.117','11.367','0.250')]
for r,(ni,s,e,o) in zip(cases,expect):
    assert int(r['native_feature_index'])==ni
    assert [f'{float(r[k]):.3f}' for k in ['source_support_start_seconds','source_support_end_seconds','actual_overlap_seconds']]==[s,e,o]
assert [r['text'].strip() for r in cases[::3]]==['best','life','free']
assert [json.loads(r['frame_indices_json']) for r in cases[2::3]]==[[315],[322],[337]]
assert [int(r['wav_sample_start']) for r in cases[1::3]]==[166400,173440,180480]
assert [int(r['wav_sample_end']) for r in cases[1::3]]==[166800,173840,180880]
assert [f"{json.loads(r['frame_pts_seconds_json'])[0]:.3f}" for r in cases[2::3]]==['10.500','10.733','11.233']
assert [(f"{float(r['window_start_seconds']):.3f}",f"{float(r['window_end_seconds']):.3f}") for r in cases[::3]]==[('10.191','10.634'),('10.634','11.077'),('11.077','11.520')]
result={'bundle_hash_files_checked':len(hash_rows),'bundle_file_count':sum(1 for p in B.rglob('*') if p.is_file()),
        'zip_sha256':sha(Path('D:/java录屏/q1_completion_20260926.zip')),
        'core_file_hashes':{n:sha(B/'final'/n) for n in ['aligned_features_fp32.npz','masks.npz','sample_ids.csv']},
        'sample_count':100,'id_order_equals_manifest_and_previous_summary':True,'arrays':stats,
        'source_mapping_rows':15000,'source_mapping_unique_keys':15000,'source_mapping_fields':list(rows[0]),
        'native_rows':dict(native_stats),'native_and_sample_aligned_values_match':True,
        'source_counts':dict(source_counts),'selected_text_fallback_samples':sum(v>0 for v in fallback_counts.values()),
        'selected_text_fallback_total':sum(fallback_counts.values()),'timebase':{'samples':100,'all_four_starts_zero':True,'decoded_frames':frames,
        'frame_counts_match':True,'max_pts_error_seconds':maxerr,'threshold_seconds':0.001,'above_threshold_samples':0,
        'source':'provided validation CSV/report; raw MP4/WAV not in bundle, media not redecoded in this task'},
        'case_rows_verified':cases,'environment_boundary':'GPU Python/OS not recorded by supplied package; no invented Ubuntu/Python3.12.3'}
(OUT/'qa').mkdir(exist_ok=True)
(OUT/'qa/completion_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['case_rows_verified','source_mapping_fields']},ensure_ascii=False,indent=2))
