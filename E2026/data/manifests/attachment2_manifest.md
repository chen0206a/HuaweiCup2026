
# Attachment 2 aligned-50 manifest

- Status: **VERIFIED**; train/valid/test = **3395/728/727**.
- Raw pickle: `data/raw/aligned_50.pkl`; SHA256 `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`.
- Label table: `data/raw/label.xlsx`; SHA256 `0d6351ace01a2edee8861b21b2893c82d916d2a3e8b69d3b4f832ca841349875`.
- Text/audio/vision: `(N,50,768)` float32, `(N,50,74)` float64, `(N,50,35)` float64 in the raw pickle. The model data interface casts audio/vision to float32.
- Class mapping: 0 Negative, 1 Neutral, 2 Positive. XLSX uses `video_id`, `clip_id`, `label`, `annotation`, `mode`.
- Padding source: `text_bert[:,1,:] == 1`; text embeddings can be nonzero outside this mask. Native all-zero valid timesteps remain data, not missing markers.
- IDs: unique within splits, disjoint across splits, one-to-one with XLSX; labels agree.
- Local path difference: files were copied to `data/raw/`, while historical server configs name `data/raw/attachment2/`. No raw file was moved.
- Attachment 2 test was checked for structure and label integrity only; it did not enter model selection.
