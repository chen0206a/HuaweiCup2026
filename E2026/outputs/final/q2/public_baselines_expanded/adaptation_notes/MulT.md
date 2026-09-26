# MulT aligned-50 adaptation

Source: https://github.com/yaohungt/Multimodal-Transformer at `a670936824ee722c8494fd98d204977a1d663c7a`. License: MIT.

Six directed pairwise attention streams; three self-memory streams; sinusoidal positions; padding-aware attention; 3-class and regression heads.

Code is a new adaptation of the published mechanism. The input is Attachment2 precomputed 50-step features. The common task uses train-only balanced CE plus SmoothL1; MISA retains its original core auxiliary objectives with published default coefficients. Padding comes from `text_bert[:,1,:]`; native all-zero vectors remain data. Models are selected on clean validation selection score, never test.
