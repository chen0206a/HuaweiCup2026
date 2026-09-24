"""Evaluate locked B0 seed43/44 on the frozen Attachment2 valid benchmark.

This is inference only: it never constructs a train/test Dataset, builds an
optimizer, changes a checkpoint, or writes to the existing Q2 result files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import Aligned50Dataset  # noqa: E402
from src.evaluation.missing_benchmark import (  # noqa: E402
    BENCHMARK_SEED,
    evaluate_benchmark,
    scenarios,
)
from src.models.baseline import B0Baseline  # noqa: E402


MANIFEST = ROOT / "outputs/final/q2/q2_checkpoint_manifest.json"
BENCHMARK_FILE = ROOT / "outputs/metrics/b2_benchmark_definition.json"
SCENARIO_MANIFEST = ROOT / "data/manifests/q2_missing_benchmark_manifest.json"
ATTACHMENT2_MANIFEST = ROOT / "data/manifests/attachment2_manifest.json"
PKL_PATH = ROOT / "data/raw/aligned_50.pkl"
OLD_SCENARIO_CSV = ROOT / "outputs/final/q2/q2_plotting_handoff_v2/plot_data/scenario_details_available.csv"
PLOT_DATA = ROOT / "outputs/final/q2/q2_plotting_handoff_v2/plot_data"
EXP_DIR = ROOT / "experiments/q2/exp_014_b0_seed43_44_scenario_completion"
EXPECTED_BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
METRICS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")
DETAIL_FIELDS = (
    "model", "training_seed", "scenario_id", "modalities", "rho", "location",
    "sample_count", "accuracy", "macro_f1", "mae", "pearson", "selection_score",
    "vision_all_zero_count", "vision_all_zero_accuracy", "vision_all_zero_macro_f1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_lock_and_input() -> tuple[dict[int, dict], Aligned50Dataset, list]:
    checkpoint_manifest = read_json(MANIFEST)
    benchmark = read_json(BENCHMARK_FILE)
    scenario_manifest = read_json(SCENARIO_MANIFEST)
    attachment2_manifest = read_json(ATTACHMENT2_MANIFEST)

    definition_hash = sha256(BENCHMARK_FILE)
    require(definition_hash == EXPECTED_BENCHMARK_SHA256,
            f"benchmark SHA256 changed: {definition_hash}")
    require(definition_hash == scenario_manifest["definition_file"]["sha256"],
            "scenario manifest definition hash differs")
    expected_scenarios = [scenario.as_dict() for scenario in scenarios()]
    require(benchmark["seed"] == BENCHMARK_SEED == scenario_manifest["benchmark_seed"],
            "benchmark seed differs from frozen seed")
    require(benchmark["scenarios"] == expected_scenarios,
            "benchmark scenario definitions differ from evaluation implementation")
    require(scenario_manifest["scenarios"] == expected_scenarios,
            "scenario manifest differs from evaluation implementation")
    require(scenario_manifest["scenario_counts"] == {
        "clean": 1, "single_modality_missing": 45, "double_modality_missing": 9,
    }, "scenario counts differ from frozen definition")

    selected: dict[int, dict] = {}
    for row in checkpoint_manifest["checkpoints"]:
        if row["model"] == "B0-WCE" and row["seed"] in (43, 44):
            seed = int(row["seed"])
            checkpoint = ROOT / row["checkpoint"]["relative_path"]
            require(checkpoint.is_file(), f"seed{seed} checkpoint missing: {checkpoint}")
            digest = sha256(checkpoint)
            require(digest == row["checkpoint"]["sha256"],
                    f"seed{seed} checkpoint SHA256 mismatch: {digest}")
            require(checkpoint.stat().st_size == row["checkpoint"]["size_bytes"],
                    f"seed{seed} checkpoint size differs from manifest")
            selected[seed] = {"row": row, "path": checkpoint, "sha256": digest}
    require(set(selected) == {43, 44}, "locked B0 seed43/44 checkpoints not both found")

    valid_count = int(attachment2_manifest["splits"]["valid"]["sample_count"])
    require(valid_count == 728, f"locked valid sample count changed: {valid_count}")
    with PKL_PATH.open("rb") as stream:
        pickle_data = pickle.load(stream)
    require("valid" in pickle_data, "Attachment2 valid split missing")
    # Only the valid split is passed to the dataset/evaluation code. The test
    # split is not indexed, validated, instantiated, or evaluated.
    valid = Aligned50Dataset(pickle_data["valid"], "valid")
    require(len(valid) == valid_count, f"loaded valid count differs: {len(valid)}")
    require(valid.padding_mask.shape == (valid_count, 50), "valid padding mask shape changed")
    require(int(valid.padding_mask.sum()) == 18628, "valid effective length sum changed")
    require(not bool((valid.padding_mask[:, 1:] & ~valid.padding_mask[:, :-1]).any()),
            "padding mask is no longer a valid prefix")
    require(attachment2_manifest["padding_mask_source"] ==
            "text_bert[:, 1, :] == 1, verified binary valid-prefix mask",
            "Attachment2 padding source differs from lock")

    # Confirm the frozen intervention preserves padding/native-zero semantics
    # using only two valid examples. The existing implementation clones source
    # features and zeroes only requested valid rows; padding is not changed.
    from src.data.block_mask import predictor_inputs
    from src.data.dataset import MODALITIES
    from src.evaluation.missing_benchmark import mask_scenario

    examples = [valid[i] for i in range(2)]
    batch = {key: torch.stack([ex[key] for ex in examples])
             for key in (*MODALITIES, "padding_mask", "availability_mask", "native_zero_mask")}
    batch["id"] = [ex["id"] for ex in examples]
    scene = expected_scenarios[0]
    from src.evaluation.missing_benchmark import Scenario
    masked, metadata = mask_scenario(batch, Scenario(
        scene["scenario_id"], (scene["modalities"],), scene["rho"], scene["location"]
    ))
    require(torch.equal(masked["padding_mask"], batch["padding_mask"]),
            "block masking modified padding mask")
    require(torch.equal(batch["text"][0], examples[0]["text"]),
            "masking modified source feature tensor")
    require(torch.all(masked["text"][0, :metadata[0]["block_length"]] == 0).item(),
            "text block was not native-zeroed")
    require(torch.equal(masked["audio"], batch["audio"]) and
            torch.equal(masked["vision"], batch["vision"]),
            "masking altered an unrequested modality")
    require(set(predictor_inputs(masked)) == {*MODALITIES, "padding_mask"},
            "availability/native-zero metadata leaked into predictor inputs")
    require(len(expected_scenarios) + 1 == 55, "expected clean + 54 scenarios")
    return selected, valid, expected_scenarios


def evaluate_seed(seed: int, selected: dict, valid: Aligned50Dataset,
                  expected_scenarios: list[dict]) -> list[dict]:
    state = torch.load(selected[seed]["path"], map_location="cpu", weights_only=False)
    require(int(state["config"]["training"]["seed"]) == seed,
            f"seed{seed} checkpoint training seed mismatch")
    require(state["config"]["preprocessing"]["normalization"] == "none" and
            state["config"]["training"]["evaluate_test"] is False,
            f"seed{seed} checkpoint preprocessing/test policy differs from lock")
    model = B0Baseline(**state["config"]["model"]).to(torch.device("cpu"))
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.float().eval()
    loader = DataLoader(valid, batch_size=int(state["config"]["data"]["batch_size"]),
                        shuffle=False, num_workers=0)
    torch.set_grad_enabled(False)
    rows = evaluate_benchmark(model, loader, torch.device("cpu"), scenario_chunk_size=6)
    require(len(rows) == 55, f"seed{seed} returned {len(rows)} rows, expected 55")
    expected_ids = ["clean"] + [row["scenario_id"] for row in expected_scenarios]
    require([row["scenario_id"] for row in rows] == expected_ids,
            f"seed{seed} scenario order/IDs differ from lock")
    for row in rows:
        require(row["sample_count"] == len(valid),
                f"seed{seed}/{row['scenario_id']} valid sample count differs")
        for metric in METRICS:
            require(math.isfinite(float(row[metric])),
                    f"non-finite {metric}: seed{seed}/{row['scenario_id']}")
        require(row["modalities"] == ("none" if row["scenario_id"] == "clean" else
                                     next(s["modalities"] for s in expected_scenarios
                                          if s["scenario_id"] == row["scenario_id"])),
                f"seed{seed}/{row['scenario_id']} modality mapping differs")
    locked_clean = selected[seed]["row"]["validation_metrics"]["clean"]
    for metric in METRICS:
        delta = abs(float(rows[0][metric]) - float(locked_clean[metric]))
        require(delta <= 1e-6,
                f"seed{seed} clean {metric} differs from checkpoint lock by {delta:.3g}")
    return rows


def flatten(model: str, seed: int, rows: list[dict]) -> list[dict]:
    flat = []
    for row in rows:
        subset = row.get("vision_all_zero_metrics") or {}
        flat.append({
            "model": model, "training_seed": seed,
            "scenario_id": row["scenario_id"], "modalities": row["modalities"],
            "rho": row["rho"], "location": row["location"],
            "sample_count": row["sample_count"],
            **{metric: row[metric] for metric in METRICS},
            "vision_all_zero_count": row["vision_all_zero_count"],
            "vision_all_zero_accuracy": subset.get("accuracy"),
            "vision_all_zero_macro_f1": subset.get("macro_f1"),
        })
    return flat


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    names = fields or tuple(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_mean_sd(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group_key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        if len(group) != 3 or {int(row["training_seed"]) for row in group} != {42, 43, 44}:
            raise RuntimeError(f"mean/SD group does not contain exactly seeds 42/43/44: {group_key}")
        item = dict(zip(keys, group_key))
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_sample_sd"] = stdev(values)  # sample SD, ddof=1
        item["n_seeds"] = 3
        result.append(item)
    return result


def main() -> None:
    torch.set_num_threads(4)
    selected, valid, expected_scenarios = validate_lock_and_input()
    evaluated = {}
    for seed in (43, 44):
        print(f"Starting inference-only B0-WCE seed{seed}: valid={len(valid)}, conditions=55",
              flush=True)
        evaluated[seed] = evaluate_seed(seed, selected, valid, expected_scenarios)
        print(f"Finished B0-WCE seed{seed}: 55/55 conditions; clean metrics match lock",
              flush=True)

    old_rows = list(csv.DictReader(OLD_SCENARIO_CSV.open(encoding="utf-8-sig", newline="")))
    require(len(old_rows) == 220, f"existing locked detail rows changed: {len(old_rows)}")
    new_rows = []
    for seed in (43, 44):
        new_rows.extend(flatten("B0-WCE", seed, evaluated[seed]))
    combined = old_rows + new_rows
    require(len(combined) == 330, f"merged rows={len(combined)}, expected 330")
    group_counts: dict[tuple[str, int], int] = defaultdict(int)
    unique = set()
    for row in combined:
        key = (row["model"], int(row["training_seed"]))
        group_counts[key] += 1
        scenario_key = (*key, row["scenario_id"])
        require(scenario_key not in unique, f"duplicate condition: {scenario_key}")
        unique.add(scenario_key)
        require(int(row["sample_count"]) == 728, f"wrong sample count in {scenario_key}")
        for metric in METRICS:
            require(math.isfinite(float(row[metric])), f"non-finite metric in {scenario_key}")
    required_groups = {(model, seed) for model in ("B0-WCE", "B5-P2") for seed in (42, 43, 44)}
    require(set(group_counts) == required_groups and all(n == 55 for n in group_counts.values()),
            f"model-seed scenario coverage differs: {dict(group_counts)}")
    expected_ids = {"clean", *(row["scenario_id"] for row in expected_scenarios)}
    for model, seed in required_groups:
        ids = {row["scenario_id"] for row in combined
               if row["model"] == model and int(row["training_seed"]) == seed}
        require(ids == expected_ids, f"incomplete conditions: {model} seed{seed}")

    # Check the clean per-seed metrics against the locked checkpoint manifest.
    checkpoint_manifest = read_json(MANIFEST)
    for model, seed in required_groups:
        checkpoint_row = next(row for row in checkpoint_manifest["checkpoints"]
                              if row["model"] == model and row["seed"] == seed)
        clean = next(row for row in combined if row["model"] == model and
                     int(row["training_seed"]) == seed and row["scenario_id"] == "clean")
        for metric in METRICS:
            delta = abs(float(clean[metric]) -
                        float(checkpoint_row["validation_metrics"]["clean"][metric]))
            require(delta <= 1e-6,
                    f"historical clean metric mismatch: {model} seed{seed} {metric} delta={delta}")
        missing_rows = [row for row in combined if row["model"] == model and
                        int(row["training_seed"]) == seed and row["scenario_id"] != "clean"]
        require(len(missing_rows) == 54, f"missing-scenario count differs for {model} seed{seed}")
        for metric in METRICS:
            missing_mean = mean(float(row[metric]) for row in missing_rows)
            delta = abs(missing_mean -
                        float(checkpoint_row["validation_metrics"]["mean_missing"][metric]))
            require(delta <= 1e-6,
                    f"historical mean-missing {model} seed{seed} {metric} differs by {delta}")

    combined.sort(key=lambda row: (0 if row["model"] == "B0-WCE" else 1,
                                   int(row["training_seed"]),
                                   0 if row["scenario_id"] == "clean" else
                                   1 + next(i for i, s in enumerate(expected_scenarios)
                                            if s["scenario_id"] == row["scenario_id"])))

    # Per-seed Figure 4: average early/middle/late within a single modality/rho.
    fig4_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in combined:
        if row["modalities"] in ("text", "audio", "vision"):
            fig4_groups[(row["model"], int(row["training_seed"]), row["modalities"],
                         float(row["rho"]))].append(row)
    fig4_seed = []
    for key, group in sorted(fig4_groups.items(), key=lambda x: tuple(map(str, x[0]))):
        require(len(group) == 3 and {r["location"] for r in group} == {"early", "middle", "late"},
                f"Figure4 missing location rows: {key}")
        item = {"model": key[0], "training_seed": key[1], "modalities": key[2], "rho": key[3]}
        for metric in METRICS:
            item[metric] = mean(float(row[metric]) for row in group)
        item["scenario_count_averaged"] = 3
        fig4_seed.append(item)
    require(len(fig4_seed) == 90, f"Figure4 rows={len(fig4_seed)}, expected 90")
    fig4_mean = aggregate_mean_sd(fig4_seed, ("model", "modalities", "rho"))
    require(len(fig4_mean) == 30, f"Figure4 mean/SD rows={len(fig4_mean)}, expected 30")

    # Per-seed Figure 5: average rho within each single modality/location.
    fig5_groups: dict[tuple, list[dict]] = defaultdict(list)
    pair_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in combined:
        modalities = row["modalities"]
        if modalities in ("text", "audio", "vision"):
            fig5_groups[(row["model"], int(row["training_seed"]), modalities,
                         row["location"])].append(row)
        elif modalities in ("text+audio", "text+vision", "audio+vision"):
            pair_groups[(row["model"], int(row["training_seed"]), modalities,
                         row["location"])].append(row)

    def average_scenarios(groups: dict, expected_count: int, label: str) -> list[dict]:
        output = []
        for key, group in sorted(groups.items(), key=lambda x: tuple(map(str, x[0]))):
            require(len(group) == expected_count, f"{label} scenario count differs: {key}")
            item = {"model": key[0], "training_seed": key[1], "modalities": key[2],
                    "location": key[3]}
            for metric in METRICS:
                item[metric] = mean(float(row[metric]) for row in group)
            item["scenario_count_averaged"] = expected_count
            output.append(item)
        return output

    fig5_seed = average_scenarios(fig5_groups, 5, "Figure5 single modality")
    fig5_mean = aggregate_mean_sd(fig5_seed, ("model", "modalities", "location"))
    pair_seed = average_scenarios(pair_groups, 1, "Figure5 double modality")
    pair_codes = {"text+audio": "TA", "text+vision": "TV", "audio+vision": "AV"}
    for row in pair_seed:
        row["pair"] = pair_codes[row["modalities"]]
    pair_mean = aggregate_mean_sd(pair_seed, ("model", "modalities", "pair", "location"))
    require(len(fig5_seed) == 54 and len(fig5_mean) == 18,
            f"Figure5 single-modality counts differ: {len(fig5_seed)}/{len(fig5_mean)}")
    require(len(pair_seed) == 54 and len(pair_mean) == 18,
            f"Figure5 pair counts differ: {len(pair_seed)}/{len(pair_mean)}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    write_csv(PLOT_DATA / "scenario_details_complete.csv", combined, DETAIL_FIELDS)
    write_csv(PLOT_DATA / "fig4_modality_by_rho_per_seed_complete.csv", fig4_seed)
    write_csv(PLOT_DATA / "fig4_modality_by_rho_mean_sd.csv", fig4_mean)
    write_csv(PLOT_DATA / "fig5_modality_by_location_per_seed_complete.csv", fig5_seed)
    write_csv(PLOT_DATA / "fig5_modality_by_location_mean_sd.csv", fig5_mean)
    write_csv(PLOT_DATA / "fig5_double_modality_location_per_seed.csv", pair_seed)
    write_csv(PLOT_DATA / "fig5_double_modality_location_mean_sd.csv", pair_mean)
    for seed in (43, 44):
        write_csv(EXP_DIR / f"b0_seed{seed}_scenario_metrics.csv",
                  flatten("B0-WCE", seed, evaluated[seed]), DETAIL_FIELDS)

    by_seed = {}
    for seed in (43, 44):
        by_seed[str(seed)] = {
            "checkpoint_sha256": selected[seed]["sha256"],
            "row_count": len(evaluated[seed]),
            "clean_metrics": {metric: evaluated[seed][0][metric] for metric in METRICS},
            "scenario_ids": [row["scenario_id"] for row in evaluated[seed]],
            "finite_metrics": True,
            "clean_matches_locked_manifest_atol_1e-6": True,
        }
    metrics_doc = {
        "experiment": "Q2 B0 seed43/44 frozen missing-scenario completion",
        "evaluation_only": True,
        "used_splits": ["valid"],
        "test_used": False,
        "attachment3_used": False,
        "training_or_tuning": False,
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "benchmark_seed": BENCHMARK_SEED,
        "valid_count": len(valid),
        "per_seed": by_seed,
        "combined": {
            "scenario_rows": len(combined), "expected_scenario_rows": 330,
            "unique_model_seed_scenario_keys": len(unique),
            "model_seed_group_sizes": {f"{m}/seed{s}": n for (m, s), n in sorted(group_counts.items())},
            "fig4_modality_rho_per_seed_rows": len(fig4_seed),
            "fig4_modality_rho_mean_sd_rows": len(fig4_mean),
            "fig5_modality_location_per_seed_rows": len(fig5_seed),
            "fig5_modality_location_mean_sd_rows": len(fig5_mean),
            "fig5_double_modality_location_per_seed_rows": len(pair_seed),
            "fig5_double_modality_location_mean_sd_rows": len(pair_mean),
            "mean_sd_method": "mean and sample standard deviation across seed42/43/44; scenarios are not independent replicates",
        },
    }
    (EXP_DIR / "metrics.json").write_text(json.dumps(metrics_doc, ensure_ascii=False,
                                                    indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")

    check = f'''# Q2 Figure 4/5 per-scenario completion check

Status: **COMPLETE**
Run date: 2026-09-24
Evaluation scope: Attachment2 valid only; no training, tuning, test, or Attachment3.

## Preflight

- B0 seed43 checkpoint SHA256: `{selected[43]['sha256']}` — manifest match.
- B0 seed44 checkpoint SHA256: `{selected[44]['sha256']}` — manifest match.
- Attachment2 valid samples: {len(valid)}; valid timesteps: 18,628; feature dimensions: text 768 / audio 74 / vision 35.
- Benchmark definition SHA256: `{EXPECTED_BENCHMARK_SHA256}`; seed 20260923; 54 missing conditions (45 single-modality + 9 double-modality) plus clean.
- Evaluation reused `src/evaluation/missing_benchmark.py`; continuous-block semantics, valid-prefix mask, padding preservation and native-zero handling were unchanged.
- Each new seed produced clean + 54 conditions (55 rows). Clean metrics matched that seed's checkpoint manifest within absolute tolerance 1e-6.

## Completed data

| Output | Rows | Aggregation |
|---|---:|---|
| `scenario_details_complete.csv` | 330 | 2 models × 3 seeds × 55 conditions |
| `fig4_modality_by_rho_per_seed_complete.csv` | 90 | model × seed × modality × rho; averages three positions within each condition |
| `fig4_modality_by_rho_mean_sd.csv` | 30 | mean and sample SD across 3 seeds |
| `fig5_modality_by_location_per_seed_complete.csv` | 54 | model × seed × modality × position; averages five ratios |
| `fig5_modality_by_location_mean_sd.csv` | 18 | mean and sample SD across 3 seeds |
| `fig5_double_modality_location_per_seed.csv` | 54 | TA/TV/AV × model × seed × position; frozen rho=0.3 |
| `fig5_double_modality_location_mean_sd.csv` | 18 | TA/TV/AV mean and sample SD across 3 seeds |

Mean/SD treats seeds as the independent repeats (`ddof=1`); scenario conditions are not treated as independent replicates. Existing incomplete source CSVs were retained unchanged.

**FIG4_DATA_COMPLETE = YES**
**FIG5_DATA_COMPLETE = YES**
'''
    check = "\n".join(line.rstrip() for line in check.splitlines()) + "\n"
    (ROOT / "outputs/final/q2/q2_fig45_completion_check.md").write_text(check, encoding="utf-8")
    notes = '''# Q2 Figure 4/5 B0 seed43/44 completion

Purpose: complete previously absent B0 seed43/44 per-scenario validation metrics for frozen Figure 4/5 data. Used locked local checkpoints and Attachment2 valid only. No training, tuning, checkpoint edits, benchmark edits, test, or Attachment3 access.

Result: B0 seed43 and seed44 each produced clean + 54 missing scenario rows. Their clean metrics matched the Q2 checkpoint manifest within 1e-6. Merged source coverage is 330 rows with 55 unique conditions for each model/seed. Complete Figure 4/5 per-seed and seed-level sample mean/SD tables are written in `outputs/final/q2/q2_plotting_handoff_v2/plot_data/`.

This is a validation evaluation completion, not a new model-selection experiment. No figures were generated.
'''
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    config_doc = {
        "name": "q2_fig45_b0_seed43_44_scenario_completion",
        "date": "2026-09-24",
        "task": "inference_only",
        "models": ["B0-WCE"], "seeds": [43, 44],
        "checkpoint_manifest": "outputs/final/q2/q2_checkpoint_manifest.json",
        "input": "data/raw/aligned_50.pkl", "split": "valid", "valid_count": 728,
        "benchmark_definition": "outputs/metrics/b2_benchmark_definition.json",
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA256, "benchmark_seed": BENCHMARK_SEED,
        "conditions": ["clean", "54 frozen missing scenarios"],
        "evaluation_implementation": "src/evaluation/missing_benchmark.py",
        "training": False, "test_used": False, "attachment3_used": False,
        "mean_sd": "across three seeds; sample SD (ddof=1)",
    }
    (EXP_DIR / "config.yaml").write_text(
        "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in config_doc.items()) + "\n",
        encoding="utf-8")
    print("Q2 Figure 4/5 completion: PASS")
    print(json.dumps({"seed43_clean": by_seed["43"]["clean_metrics"],
                      "seed44_clean": by_seed["44"]["clean_metrics"],
                      "complete_rows": len(combined), "outputs": [
                          str(PLOT_DATA / name) for name in (
                              "scenario_details_complete.csv",
                              "fig4_modality_by_rho_per_seed_complete.csv",
                              "fig4_modality_by_rho_mean_sd.csv",
                              "fig5_modality_by_location_per_seed_complete.csv",
                              "fig5_modality_by_location_mean_sd.csv",
                              "fig5_double_modality_location_per_seed.csv",
                              "fig5_double_modality_location_mean_sd.csv",
                          )]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
