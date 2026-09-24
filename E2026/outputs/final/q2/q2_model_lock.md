
# Q2 final model lock

**Status: LOCKED.** Baseline: B0-WCE. Final: B5-P2 attention residual pooling.

The input is the existing aligned-50 T/A/V features (768/74/35 dimensions), with the audited `text_bert[:,1,:]` padding mask. B0 applies masked mean, modality projection, concat fusion, and 3-class plus regression heads. P2 adds a per-modality scalar attention pool and zero-initialized gamma residual. Seed-specific B0 weights are frozen and held in eval mode; only 883 P2 parameters train. No normalization or BlockMask training augmentation is used.

| Model | 3-seed validation robust score (mean ± sample SD) |
|---|---:|
| B0-WCE | 0.741677 ± 0.001458 |
| B5-P2 | 0.744777 ± 0.001492 |

Paired P2−B0 robust delta: seed42 +0.005600, seed43 +0.003791, seed44 -0.000090; mean +0.003100 ± 0.002907. Two seeds improved; seed44 was approximately tied. The result is seed-sensitive.

Benchmark SHA256: `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`. Source commit before this lock: `aa3f07cc8a43d2ffe46f81f3d150143b7fd485fe`. Attachment 3 remains **SEALED**. Attachment 2 test was not used for model selection in this stage.

The actual local raw path is `data/raw/`; historical server configs use `data/raw/attachment2/`. The manifest records both without moving the raw files. The six main B0/P2 checkpoints are present and hash-indexed. P2 seed43/44 best-clean checkpoint files are referenced by existing results but are not presently in this local checkpoint directory; no existing file was deleted.
