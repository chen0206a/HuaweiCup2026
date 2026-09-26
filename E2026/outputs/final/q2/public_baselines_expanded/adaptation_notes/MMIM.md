# MMIM aligned-50 adaptation

Original: [Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis](https://aclanthology.org/2021.emnlp-main.723/) (EMNLP 2021).

Official code: https://github.com/declare-lab/Multimodal-Infomax at `cd0774c5a712ca5f1a5497dbf27dde11cade7434`. License: Unverified; no source copied. No source copied.

Core retained: pairwise unimodal and fusion-to-unimodal information preservation.

Input adaptation and departures: InfoNCE surrogate rather than original BA/CPC estimates; masked-mean aligned-50 encoders; common two-task heads. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.

Heads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.

Auxiliary and architecture constants: Mean of six symmetric InfoNCE terms, temperature 0.2: three modality pairs and three fusion-to-modality terms.

Training regime: standard clean training. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.
