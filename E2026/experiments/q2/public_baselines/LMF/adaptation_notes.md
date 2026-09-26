# LMF aligned-50 adaptation

Original: [Efficient Low-rank Multimodal Fusion With Modality-Specific Factors](https://aclanthology.org/P18-1209/) (ACL 2018).

Official code: https://github.com/Justin1904/Low-rank-Multimodal-Fusion at `ea4755124d19bddf7941695215e98f16fbb1fff9`. License: No LICENSE file in inspected repository listing. No source copied.

Core retained: modality-specific low-rank tensor factors.

Input adaptation and departures: Aligned-50 text LSTM and pooled audio/vision; rank 8; common two-task heads. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.

Heads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.

Auxiliary and architecture constants: No auxiliary loss; factor rank 8.

Training regime: standard clean training. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.
