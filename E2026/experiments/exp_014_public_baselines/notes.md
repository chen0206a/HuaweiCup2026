# Q2 public architecture baselines

Ran TFN, MulT, and MISA aligned-50 adaptations with seeds 42/43/44 on Attachment2 train/valid. All nine runs completed and were evaluated on the frozen 54-scenario missing benchmark. No Attachment2 test, Attachment3, Attachment4, or external sentiment data was used.

Best clean and robust aggregate among these baselines was MulT. The locked attention residual Q2 model remains higher on the reported four metrics, but its historical checkpoint was selected by robust score whereas baseline checkpoints were selected on clean valid; missing ranking must be described with that caveat. MISA retains extra auxiliary objectives and has much larger regression MAE. This experiment can enter the paper as an adapted architecture comparison, with source and adaptation caveats.

See `outputs/final/q2/public_baselines/Baseline_Experiment_Report.md` and per-seed source manifests. Do not change the locked Q2 model based on this comparison.
