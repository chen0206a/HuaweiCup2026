# Q2 experiment index

All reported robust scores come from existing result JSON; no experiment was run during locking.

| Experiment | Seeds | Robust result | Conclusion | Role | Source commit |
|---|---|---|---|---|---|
| B0-WCE | 42,43,44 | 0.741677 ± 0.001458 | KEEP | BASELINE | 3ae71251c8 |
| B1 Transformer | 42,43,44 | 0.735101 ± 0.001985 | STOP | — | 4370a1cb3e |
| B2 BlockMask | 42,43,44 | see JSON | STOP | — | 4370a1cb3e |
| B3 reconstruction | 42 | 0.732304 | STOP | — | 2a3c692030 |
| B3.1 frozen reconstruction | 42 | see JSON | DIAGNOSTIC | — | 12172c4df2 |
| B3.2 reconstruction stability | 42,43,44 | see JSON | STOP | — | 12172c4df2 |
| B4' dynamic gate | 42 | 0.743974 | DIAGNOSTIC | — | d420c0f483 |
| B5-P1 Mean+Max | 42 | 0.744320 | DIAGNOSTIC | — | 8270832920 |
| B5-P2 Attention | 42,43,44 | 0.744777 ± 0.001492 | KEEP | FINAL | 3ae71251c8 |
| B5-F low-rank fusion | 42 | 0.738850 | STOP | — | 2fb68ff6f8 |
| B5-L1 lambda balance | 42 | see JSON | STOP | — | 4e0434490f |
| B5-H0 head diagnostic | 42 | diagnostic only | DIAGNOSTIC | — | d04fdb3002 |
| B5-N1 text z-score | 42 | 0.734474 | STOP | — | aa3f07cc8a |

`B0-WCE` is the baseline; `B5-P2` is the final model. P2 improved in two seeds and approximately tied in seed 44.

See `q2_experiment_index.json` for source files, hypotheses, changed components and checkpoint availability.
