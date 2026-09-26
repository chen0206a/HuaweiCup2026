# MMIN aligned-50 adaptation

Original: [Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities](https://aclanthology.org/2021.acl-long.203/) (ACL 2021).

Official code: https://github.com/AIM3-RUC/MMIN at `c1f39f84cb97f2a75d7a97b2ba7ce96b76566106`. License: MIT; inspected repository contains no implementation files. No source copied.

Core retained: missing-modality latent imagination, residual refinement and cycle consistency.

Input adaptation and departures: Aligned-50 modality pooling; whole-view sampled missing training; simplified two-stage imagination, not original full CRA; adapt emotion task to shared two-task sentiment heads. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.

Heads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.

Auxiliary and architecture constants: Latent imagination SmoothL1 plus 0.1 cycle SmoothL1; whole-modality missing sampling.

Training regime: missing-aware training. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.
