# Q2 B2.1 training-seed stability

Why: assess whether the B2 BlockMask direction and B0/B1 ranking depend on training seed. The four architectures, hyperparameters, checkpoint rules and fixed validation benchmark are unchanged.

Training seeds: 42 (existing frozen B2 result, reused), 43 and 44 (new runs). The benchmark seed remains 20260923; its definition file SHA-256 was verified unchanged. Only attachment2 train/valid was used; test and attachment3 were not constructed or evaluated.

Result: B2-B0 minus B0-WCE robust score is negative in all three paired seeds (-0.0009, -0.0011, -0.0074). B2-B1 minus B1-WCE is positive for 42/43 and negative for 44 (+0.0023, +0.0020, -0.0051). B2-B0 robust score exceeds B2-B1 in all three seeds (+0.0032, +0.0044, +0.0034). See metrics.json and outputs/metrics/b21_multiseed_runs.csv for full evidence.

Paper status: useful as a limited stability check; three seeds on one validation split do not justify significance claims. Recommended backbone for later work is B0 on the prespecified robust score. Stop at B2.1 and await further instructions before B3.
