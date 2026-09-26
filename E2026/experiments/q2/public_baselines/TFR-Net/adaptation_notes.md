# TFR-Net aligned-50 adaptation

Original: [Transformer-based Feature Reconstruction Network for Robust Multimodal Sentiment Analysis](https://doi.org/10.1145/3474085.3475585) (ACM MM 2021).

Official code: https://github.com/thuiar/TFR-Net at `00f06a68a44a6b6fc0043f01954336f30ec3b090`. License: Unverified; no source copied. No source copied.

Core retained: temporal Transformer context and latent feature reconstruction.

Input adaptation and departures: Single shared context layer on aligned-50 with block-masked training; compact latent reconstruction, not original full network; common two-task heads. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.

Heads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.

Auxiliary and architecture constants: Mean SmoothL1 latent reconstruction at zero-trigger valid positions; one 192-dimensional Transformer context layer.

Training regime: missing-aware training. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.
