# MAG-BERT aligned-50 adaptation

Original: [Integrating Multimodal Information in Large Pretrained Transformers](https://aclanthology.org/2020.acl-main.214/) (ACL 2020).

Official code: https://github.com/WasifurRahman/BERT_multimodal_transformer at `dc7876fc30f7ef362999200911e3d4d8a2bca107`. License: Unverified; no source copied. No source copied.

Core retained: audio/vision-conditioned MAG shift into language states.

Input adaptation and departures: Apply gate to provided BERT features and train a small 2-layer encoder; original pretrained BERT is not fine-tuned; common two-task heads. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.

Heads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.

Auxiliary and architecture constants: No auxiliary loss; MAG beta_shift 1.0 and two 128-dimensional Transformer encoder layers.

Training regime: standard clean training. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.
