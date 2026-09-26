# Expanded Q2 baseline QA

- Published architecture adaptations: 12; project comparators: 2.
- Train/valid sizes: 3395/728; seeds: 42, 43, 44.
- Clean seedwise rows: 42; mean-missing seedwise rows: 42; summary rows per condition: 14.
- New model runs with preflight and measured checkpoint: 27/27.
- Reused earlier public-model runs: TFN/MulT/MISA, 9/9.
- Reused locked CleanSelect comparators: B0/P2, 6/6; grouped re-evaluation matched prior scores within 1e-5.
- Every new benchmark result has 728 validation samples in clean plus the 54 official scenario IDs, in frozen order.
- Sample SD computed with ddof=1 after averaging 54 scenarios within each seed.
- aligned_50.pkl SHA256: `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`.
- Benchmark definition SHA256: `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`.
- Adaptation model code SHA256: `e9d87b420de6cd9a79b9751ca7c04211b80112956f7e592ee134f14d4b0c5158`.
- Training runner SHA256: `bb47ce181cf5549a1aa048ca6c1e2b0d1f42b3773a44364eef90c0ddc4b7172a`.
- Dataset loader used include_test=False; no Attachment3/4 code path occurs in the training/evaluation runner.
- Checkpoint manifest contains 42 verified SHA256 values; bundle files are local hardlinks or byte-identical copies.
