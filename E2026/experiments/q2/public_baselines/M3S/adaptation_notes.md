# M3S aligned-50 adaptation

Original: [Missing Modality meets Meta Sampling (M3S): An Efficient Universal Approach for Multimodal Sentiment Analysis with Missing Modality](https://aclanthology.org/2022.aacl-main.10/) (AACL 2022).

Official code: No verified official implementation located at `N/A`. License: N/A; reimplemented from paper description. No source copied.

Core retained: missing-modality meta sampling around an existing backbone.

Input adaptation and departures: LMF backbone; one-step first-order clean support/masked query meta update, not exact original MAML; common two-task heads. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.

Heads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.

Auxiliary and architecture constants: One inner SGD-style adaptation step of size 0.001 on clean support; outer loss is 0.5 clean + 0.5 masked query.

Training regime: missing-aware training. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.
