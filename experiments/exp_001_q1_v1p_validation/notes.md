# Q1 V1-P validation

Purpose: test the frozen structure-aware partition on 12 structurally selected
holdout cases. Selection uses graph and official V0 results only. The four cases
used to develop V1-lite are excluded.

The user requested a quick Q1 closure after the V0 screen. Screening had already
completed all 100 cases at that point. To bound the comparison time, selection
uses cases whose official V0 four-candidate screen plus single-core denominator
required at most 60 seconds. This excludes the most expensive large graphs and
must be stated as a limitation.

Run from the repository root, in order:

```text
python -m src.q1.validate_v1_partition screen
python -m src.q1.validate_v1_partition select
python -m src.q1.validate_v1_partition run
python -m src.q1.validate_v1_partition report
```

All 96 algorithm rows succeeded (48 paired case×K combinations). Mean and median
improvements in official makespan were 20.90% and 7.50%. V1-P improved 62.50%,
degraded 12.50%, and tied 25.00% of combinations; three combinations degraded by
more than 5%, concentrated in case_002 and case_078. The worst was case_002 K=2
at -30.58%. The official-evaluator calls made in the comparison totaled 260 and
380.9 seconds of evaluator time; cache hits were recorded separately.

Decision: freeze the current Q1 V1-P implementation and stop Q1 optimization.
The result is suitable as a limited, clearly labeled comparison in the paper,
not as a claim of uniform improvement or full-100-case performance. The fastest
screening subset excludes the most expensive large graphs. See
`q1/Q1_FREEZE_REPORT.md` and `q1/V1_VALIDATION_REPORT.md` for the full evidence.
