# MissModal aligned-50 adaptation

Original: [MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis](https://aclanthology.org/2023.tacl-1.94/) (TACL 2023).

Official code: https://github.com/RH-Lin/MissModal at `83713ff92d64084effd4e0f53334bfcc61e31f32`. License: No LICENSE file in inspected revision; code unavailable. No source copied.

Core retained: contrastive, distribution and sentiment-semantic alignment of clean/missing representations.

Input adaptation and departures: Aligned-50 block-masked student and clean stop-gradient teacher; mean/std distribution proxy; original code unavailable; common two-task heads. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.

Heads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.

Auxiliary and architecture constants: 0.1 times each geometric InfoNCE, mean/std distribution alignment, and sentiment KL alignment.

Training regime: missing-aware training. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.
