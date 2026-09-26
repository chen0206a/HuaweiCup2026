"""Write source provenance and adaptation contracts for the Q2 comparison."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final/q2/public_baselines_expanded"

# URLs and revisions were inspected for architectural guidance only. This project
# does not copy source files from the listed repositories.
ROWS = [
    dict(model="TFN", family="standard fusion", paper_title="Tensor Fusion Network for Multimodal Sentiment Analysis", venue_year="EMNLP 2017", paper_link="https://aclanthology.org/D17-1115/", code_url="https://github.com/Justin1904/TensorFusionNetworks", code_commit="ef0e78b5583159de9b74ef2cdef6031bd9f94b37", license="No LICENSE file in inspected revision", adaptation_file="src/models/public_baselines.py", core="outer-product tensor fusion", adaptation="Packed text LSTM and masked-mean audio/vision; common two-task heads", training_regime="standard clean training"),
    dict(model="LMF", family="standard fusion", paper_title="Efficient Low-rank Multimodal Fusion With Modality-Specific Factors", venue_year="ACL 2018", paper_link="https://aclanthology.org/P18-1209/", code_url="https://github.com/Justin1904/Low-rank-Multimodal-Fusion", code_commit="ea4755124d19bddf7941695215e98f16fbb1fff9", license="No LICENSE file in inspected repository listing", adaptation_file="src/models/public_baselines_extended.py", core="modality-specific low-rank tensor factors", adaptation="Aligned-50 text LSTM and pooled audio/vision; rank 8; common two-task heads", training_regime="standard clean training"),
    dict(model="MFN", family="standard fusion", paper_title="Memory Fusion Network for Multi-view Sequential Learning", venue_year="AAAI 2018", paper_link="https://ojs.aaai.org/index.php/AAAI/article/view/12021", code_url="https://github.com/pliang279/MFN", code_commit="b0453fb6e21b581c796246a0c32a6551f2028c1b", license="MIT (repository)", adaptation_file="src/models/public_baselines_extended.py", core="three recurrent streams, delta-memory attention, gated memory", adaptation="Use aligned-50 timesteps and padding mask; smaller hidden/memory; common two-task heads", training_regime="standard clean training"),
    dict(model="MulT", family="standard fusion", paper_title="Multimodal Transformer for Unaligned Multimodal Language Sequences", venue_year="ACL 2019", paper_link="https://aclanthology.org/P19-1656/", code_url="https://github.com/yaohungt/Multimodal-Transformer", code_commit="a670936824ee722c8494fd98d204977a1d663c7a", license="MIT", adaptation_file="src/models/public_baselines.py", core="directed pairwise cross-modal attention", adaptation="Apply six directed streams to aligned-50 input with padding-aware attention; common two-task heads", training_regime="standard clean training"),
    dict(model="MISA", family="standard fusion", paper_title="MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis", venue_year="ACM MM 2020", paper_link="https://doi.org/10.1145/3394171.3413678", code_url="https://github.com/declare-lab/MISA", code_commit="ec42faddde0d210cf7368aebf2118fe9570e7102", license="MIT", adaptation_file="src/models/public_baselines.py", core="shared/private decomposition with similarity, difference and reconstruction losses", adaptation="Use precomputed text features and pooled aligned-50 streams; common two-task heads; retain auxiliary losses", training_regime="standard clean training"),
    dict(model="Self-MM", family="standard fusion", paper_title="Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis", venue_year="AAAI 2021", paper_link="https://ojs.aaai.org/index.php/AAAI/article/view/17289", code_url="https://github.com/thuiar/Self-MM", code_commit="1786283c81eeb507f317fa1c70a3faf77e67cee0", license="Unverified; no source copied", adaptation_file="src/models/public_baselines_extended.py", core="fused and unimodal sentiment branches with pseudo supervision", adaptation="Masked-mean aligned-50 encoders; stop-gradient fused/regression pseudo targets; no original dynamic task-weighting; common two-task heads", training_regime="standard clean training"),
    dict(model="MMIM", family="standard fusion", paper_title="Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis", venue_year="EMNLP 2021", paper_link="https://aclanthology.org/2021.emnlp-main.723/", code_url="https://github.com/declare-lab/Multimodal-Infomax", code_commit="cd0774c5a712ca5f1a5497dbf27dde11cade7434", license="Unverified; no source copied", adaptation_file="src/models/public_baselines_extended.py", core="pairwise unimodal and fusion-to-unimodal information preservation", adaptation="InfoNCE surrogate rather than original BA/CPC estimates; masked-mean aligned-50 encoders; common two-task heads", training_regime="standard clean training"),
    dict(model="MAG-BERT", family="standard fusion", paper_title="Integrating Multimodal Information in Large Pretrained Transformers", venue_year="ACL 2020", paper_link="https://aclanthology.org/2020.acl-main.214/", code_url="https://github.com/WasifurRahman/BERT_multimodal_transformer", code_commit="dc7876fc30f7ef362999200911e3d4d8a2bca107", license="Unverified; no source copied", adaptation_file="src/models/public_baselines_extended.py", core="audio/vision-conditioned MAG shift into language states", adaptation="Apply gate to provided BERT features and train a small 2-layer encoder; original pretrained BERT is not fine-tuned; common two-task heads", training_regime="standard clean training"),
    dict(model="TFR-Net", family="missing-aware", paper_title="Transformer-based Feature Reconstruction Network for Robust Multimodal Sentiment Analysis", venue_year="ACM MM 2021", paper_link="https://doi.org/10.1145/3474085.3475585", code_url="https://github.com/thuiar/TFR-Net", code_commit="00f06a68a44a6b6fc0043f01954336f30ec3b090", license="Unverified; no source copied", adaptation_file="src/models/public_baselines_extended.py", core="temporal Transformer context and latent feature reconstruction", adaptation="Single shared context layer on aligned-50 with block-masked training; compact latent reconstruction, not original full network; common two-task heads", training_regime="missing-aware training"),
    dict(model="MissModal", family="missing-aware", paper_title="MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis", venue_year="TACL 2023", paper_link="https://aclanthology.org/2023.tacl-1.94/", code_url="https://github.com/RH-Lin/MissModal", code_commit="83713ff92d64084effd4e0f53334bfcc61e31f32", license="No LICENSE file in inspected revision; code unavailable", adaptation_file="src/models/public_baselines_extended.py", core="contrastive, distribution and sentiment-semantic alignment of clean/missing representations", adaptation="Aligned-50 block-masked student and clean stop-gradient teacher; mean/std distribution proxy; original code unavailable; common two-task heads", training_regime="missing-aware training"),
    dict(model="M3S", family="missing-aware", paper_title="Missing Modality meets Meta Sampling (M3S): An Efficient Universal Approach for Multimodal Sentiment Analysis with Missing Modality", venue_year="AACL 2022", paper_link="https://aclanthology.org/2022.aacl-main.10/", code_url="No verified official implementation located", code_commit="N/A", license="N/A; reimplemented from paper description", adaptation_file="src/models/public_baselines_extended.py + scripts/run_public_baseline_extended.py", core="missing-modality meta sampling around an existing backbone", adaptation="LMF backbone; one-step first-order clean support/masked query meta update, not exact original MAML; common two-task heads", training_regime="missing-aware training"),
    dict(model="MMIN", family="missing-aware", paper_title="Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities", venue_year="ACL 2021", paper_link="https://aclanthology.org/2021.acl-long.203/", code_url="https://github.com/AIM3-RUC/MMIN", code_commit="c1f39f84cb97f2a75d7a97b2ba7ce96b76566106", license="MIT; inspected repository contains no implementation files", adaptation_file="src/models/public_baselines_extended.py", core="missing-modality latent imagination, residual refinement and cycle consistency", adaptation="Aligned-50 modality pooling; whole-view sampled missing training; simplified two-stage imagination, not original full CRA; adapt emotion task to shared two-task sentiment heads", training_regime="missing-aware training"),
    dict(model="B0", family="project baseline", paper_title="Project B0 masked-mean baseline", venue_year="Project model", paper_link="N/A", code_url="N/A", code_commit="N/A", license="Project repository", adaptation_file="src/models/baseline.py", core="three masked-mean projections and concatenation fusion", adaptation="Original project checkpoint; weighted CE plus SmoothL1", training_regime="standard clean training"),
    dict(model="P2", family="project model", paper_title="Project P2 attention-residual pooling", venue_year="Project model", paper_link="N/A", code_url="N/A", code_commit="N/A", license="Project repository", adaptation_file="src/models/pooling_residual.py", core="frozen B0 plus learned attention pooling residual", adaptation="Original project checkpoints; clean-selected for fair comparison", training_regime="standard clean training"),
]

AUXILIARY_SPEC = {
    "LMF": "No auxiliary loss; factor rank 8.",
    "MFN": "No auxiliary loss; hidden 32, memory 64.",
    "Self-MM": "Mean SmoothL1 of three unimodal branches to 0.5 true regression label + 0.5 detached fused regression prediction.",
    "MMIM": "Mean of six symmetric InfoNCE terms, temperature 0.2: three modality pairs and three fusion-to-modality terms.",
    "MAG-BERT": "No auxiliary loss; MAG beta_shift 1.0 and two 128-dimensional Transformer encoder layers.",
    "TFR-Net": "Mean SmoothL1 latent reconstruction at zero-trigger valid positions; one 192-dimensional Transformer context layer.",
    "MissModal": "0.1 times each geometric InfoNCE, mean/std distribution alignment, and sentiment KL alignment.",
    "M3S": "One inner SGD-style adaptation step of size 0.001 on clean support; outer loss is 0.5 clean + 0.5 masked query.",
    "MMIN": "Latent imagination SmoothL1 plus 0.1 cycle SmoothL1; whole-modality missing sampling.",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(ROWS[0]) + ["additional_pretrained_weights", "external_sentiment_data"]
    csv_path = OUT / "q2_public_baseline_registry.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ROWS:
            writer.writerow({**row, "additional_pretrained_weights": "no", "external_sentiment_data": "no"})
    lines = ["# Q2 public baseline registry", "", "All results are aligned-50 architecture adaptations trained on Attachment2 only. They are not full reproductions of the original papers. No external sentiment training data or additional pretrained model weights are used.", "", "| Model | Paper | Venue | Code revision | License | Regime |", "|---|---|---|---|---|---|"]
    for row in ROWS:
        paper = f"[{row['paper_title']}]({row['paper_link']})" if row["paper_link"] != "N/A" else row["paper_title"]
        code = f"[{row['code_commit'][:8]}]({row['code_url']})" if row["code_commit"] != "N/A" else "N/A"
        lines.append(f"| {row['model']} | {paper} | {row['venue_year']} | {code} | {row['license']} | {row['training_regime']} |")
    (OUT / "baseline_registry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "q2_public_baseline_registry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for row in ROWS:
        if row["model"] in {"TFN", "MulT", "MISA", "B0", "P2"}:
            continue
        directory = ROOT / "experiments/q2/public_baselines" / row["model"]
        directory.mkdir(parents=True, exist_ok=True)
        note = f"# {row['model']} aligned-50 adaptation\n\nOriginal: [{row['paper_title']}]({row['paper_link']}) ({row['venue_year']}).\n\nOfficial code: {row['code_url']} at `{row['code_commit']}`. License: {row['license']}. No source copied.\n\nCore retained: {row['core']}.\n\nInput adaptation and departures: {row['adaptation']}. All inputs are Attachment2 precomputed 50-step text/audio/vision features, with padding masked according to the audited dataset. No new tokenizer or model weights are loaded.\n\nHeads: original task head is replaced/adapted with the shared 3-class classification and continuous regression heads. Main task objective is train-count balanced CE plus SmoothL1. No external sentiment dataset is used.\n\nAuxiliary and architecture constants: {AUXILIARY_SPEC[row['model']]}\n\nTraining regime: {row['training_regime']}. Validation checkpoint is selected on the same clean selection score used by every model; the frozen 54-scenario benchmark is evaluated afterward.\n"
        (directory / "adaptation_notes.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
