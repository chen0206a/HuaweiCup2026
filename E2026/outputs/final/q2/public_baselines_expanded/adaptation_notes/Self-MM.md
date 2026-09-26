# Self-MM aligned-50 adaptation

Original: [Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/17289) (AAAI 2021).

Official code: https://github.com/thuiar/Self-MM at `1786283c81eeb507f317fa1c70a3faf77e67cee0`. License: Unverified; no source copied. No source copied.

Core retained: fused and unimodal sentiment branches with pseudo supervision.

Input adaptation and departures: Masked-mean aligned-50 encoders; stop-gradient fused/regression pseudo targets; no original dynamic task-weighting; common two-task heads. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.

Heads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.

Auxiliary and architecture constants: Mean SmoothL1 of three unimodal branches to 0.5 true regression label + 0.5 detached fused regression prediction.

Training regime: standard clean training. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.
