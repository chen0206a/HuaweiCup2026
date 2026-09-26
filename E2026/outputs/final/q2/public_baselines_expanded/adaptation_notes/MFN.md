# MFN aligned-50 adaptation

Original: [Memory Fusion Network for Multi-view Sequential Learning](https://ojs.aaai.org/index.php/AAAI/article/view/12021) (AAAI 2018).

Official code: https://github.com/pliang279/MFN at `b0453fb6e21b581c796246a0c32a6551f2028c1b`. License: MIT (repository). No source copied.

Core retained: three recurrent streams, delta-memory attention, gated memory.

Input adaptation and departures: Use aligned-50 timesteps and padding mask; smaller hidden/memory; common two-task heads. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.

Heads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.

Auxiliary and architecture constants: No auxiliary loss; hidden 32, memory 64.

Training regime: standard clean training. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.
