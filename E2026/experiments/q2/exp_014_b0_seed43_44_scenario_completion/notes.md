# Q2 Figure 4/5 B0 seed43/44 completion

Purpose: complete previously absent B0 seed43/44 per-scenario validation metrics for frozen Figure 4/5 data. Used locked local checkpoints and Attachment2 valid only. No training, tuning, checkpoint edits, benchmark edits, test, or Attachment3 access.

Result: B0 seed43 and seed44 each produced clean + 54 missing scenario rows. Their clean metrics matched the Q2 checkpoint manifest within 1e-6. Merged source coverage is 330 rows with 55 unique conditions for each model/seed. Complete Figure 4/5 per-seed and seed-level sample mean/SD tables are written in `outputs/final/q2/q2_plotting_handoff_v2/plot_data/`.

This is a validation evaluation completion, not a new model-selection experiment. No figures were generated.
