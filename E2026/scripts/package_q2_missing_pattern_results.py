# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a-d) modality-wise trend panels -> E2026/outputs/final/q3/figures/scripts/figure8_q3_faithfulness_zh.py -> param inherit
# (a-d) typography and multipanel layout -> E2026/outputs/final/q3/figures/scripts/figure9_q3_case_studies_v2.py -> param inherit
# RULE: Copy inherited styling only; load all plotted values from validated local result files.
"""Package Q2 modality-missing trends and descriptive public-baseline results.

This script performs no model loading, inference, training, or data-split access.
It validates frozen result files, writes the preflight report before plotting,
then exports publication assets and an auditable ZIP bundle.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": None,
    "savefig.dpi": 300,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "E2026"
Q2 = E / "outputs/final/q2"
PLOT_DATA = Q2 / "q2_plotting_handoff_v2/plot_data"
PUBLIC = Q2 / "public_baselines"
OUT = Q2 / "q2_missing_pattern_public_baseline_bundle"
ZIP_PATH = Q2 / "q2_missing_pattern_public_baseline_bundle.zip"
PREFLIGHT_ROOT = Q2 / "DATA_PREFLIGHT_REPORT.md"

P2_DETAIL = PLOT_DATA / "scenario_details_complete.csv"
P2_RHO_AGG = PLOT_DATA / "fig4_modality_by_rho_per_seed_complete.csv"
MODEL_LOCK = Q2 / "q2_model_lock.json"
Q2_CKPT_MANIFEST = Q2 / "q2_checkpoint_manifest.json"
BENCH_FILE = E / "outputs/metrics/b2_benchmark_definition.json"
BENCH_MANIFEST = E / "data/manifests/q2_missing_benchmark_manifest.json"
BASELINE_MANIFEST = PUBLIC / "baseline_checkpoint_manifest.json"
BASELINE_REPORT = PUBLIC / "Baseline_Experiment_Report.md"
BASELINE_INTEGRATION = PUBLIC / "baseline_paper_integration_check.md"
COMPLETION_CHECK = Q2 / "q2_fig45_completion_check.md"

MODELS = ("TFN", "MulT", "MISA")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "audio", "vision")
LOCATIONS = ("early", "middle", "late")
RHOS = (0.1, 0.2, 0.3, 0.4, 0.5)
METRICS = ("accuracy", "macro_f1", "mae", "pearson")
EXTRA_METRICS = ("selection_score",)
ALL_METRICS = METRICS + EXTRA_METRICS

# Modality fills follow manuscript Figures 11/12. Darker line colors preserve
# their semantic mapping while remaining legible at manuscript size.
PASTELS = {"text": "#BFDCE6", "audio": "#EEE7B0", "vision": "#E9C9CC"}
LINES = {"text": "#4F7F95", "audio": "#B28D32", "vision": "#B66F7A"}
DARK = "#333333"
GRID = "#D9D9D9"
ACCENT = "#F0B36D"
LABELS = {"text": "文本缺失", "audio": "音频缺失", "vision": "视觉缺失"}
METRIC_LABELS = {
    "accuracy": ("准确率（↑）", "准确率"),
    "macro_f1": ("宏平均 F1（↑）", "宏平均 F1"),
    "mae": ("平均绝对误差（MAE，↓）", "平均绝对误差"),
    "pearson": ("皮尔逊相关系数（↑）", "皮尔逊相关系数"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def validate_sources() -> dict:
    lock = load_json(MODEL_LOCK)
    q2_ckpt = load_json(Q2_CKPT_MANIFEST)
    bm = load_json(BENCH_MANIFEST)
    bdef = load_json(BENCH_FILE)
    base_manifest = load_json(BASELINE_MANIFEST)
    details = pd.read_csv(P2_DETAIL)
    expected_scenarios = [
        (r["scenario_id"], r["modalities"], float(r["rho"]), r["location"])
        for r in bm["scenarios"]
    ]
    definition_scenarios = [
        (r["scenario_id"], r["modalities"], float(r["rho"]), r["location"])
        for r in bdef["scenarios"]
    ]
    benchmark_sha = sha256(BENCH_FILE)
    lock_sha = lock["benchmark"]["sha256"]
    if bm["status"] != "FROZEN_VERIFIED" or expected_scenarios != definition_scenarios:
        raise RuntimeError("Frozen benchmark manifest and definition differ")
    if benchmark_sha != bm["definition_file"]["sha256"] or benchmark_sha != lock_sha:
        raise RuntimeError("Benchmark SHA256 differs from Q2 lock")
    if len(expected_scenarios) != 54 or bm["attachment2_test_used_for_selection"]:
        raise RuntimeError("Unexpected benchmark size or test-use flag")

    # Verify the frozen P2 seed checkpoints and the result file named by the lock.
    q2_p2_checkpoint_checks = []
    for row in q2_ckpt["checkpoints"]:
        if row["model"] != "B5-P2":
            continue
        cp = E / row["checkpoint"]["relative_path"]
        okay = (cp.is_file() and cp.stat().st_size == row["checkpoint"]["size_bytes"]
                and sha256(cp) == row["checkpoint"]["sha256"])
        q2_p2_checkpoint_checks.append({"seed": int(row["seed"]), "path": row["checkpoint"]["relative_path"],
                                        "sha256": row["checkpoint"]["sha256"], "pass": okay})
    if len(q2_p2_checkpoint_checks) != 3 or not all(r["pass"] for r in q2_p2_checkpoint_checks):
        raise RuntimeError("Locked P2 seed42/43/44 checkpoints do not match Q2 manifest")
    source_result = lock["source_result"]
    lock_result_path = E / source_result["relative_path"]
    if (not lock_result_path.is_file() or lock_result_path.stat().st_size != source_result["size_bytes"]
            or sha256(lock_result_path) != source_result["sha256"]):
        raise RuntimeError("Q2 locked multiseed source result does not match model lock")

    # Q2 P2 curve gate: exact seed × modality × rho × location coverage.
    p2 = details[(details.model == "B5-P2") & details.modalities.isin(MODALITIES)
                 & details.rho.isin(RHOS)].copy()
    p2_key = list(zip(p2.training_seed.astype(int), p2.modalities,
                      p2.rho.astype(float), p2.location))
    expected_curve = {
        (seed, modality, rho, location)
        for seed in SEEDS for modality in MODALITIES for rho in RHOS
        for location in LOCATIONS
    }
    curve_unique = len(set(p2_key)) == len(p2_key)
    curve_complete = set(p2_key) == expected_curve and len(p2) == 135
    curve_finite = finite_frame(p2, METRICS)
    curve_model_consistent = set(p2.model) == {"B5-P2"} and set(p2.sample_count) == {728}
    if not (curve_unique and curve_complete and curve_finite and curve_model_consistent):
        raise RuntimeError("P2 single-modality curve data failed completeness/finite checks")

    # Three clean-reference values per metric come from the locked Q2 result.
    clean_lock = lock["validation_summary"]["P2_clean"]
    for metric in METRICS:
        locked_values = np.asarray(clean_lock[metric]["values"], dtype=float)
        observed = details[(details.model == "B5-P2") & (details.scenario_id == "clean")]
        observed = observed.sort_values("training_seed")[metric].to_numpy(dtype=float)
        if len(locked_values) != 3 or not np.allclose(locked_values, observed, atol=1e-12, rtol=0):
            raise RuntimeError(f"P2 clean reference does not match model lock: {metric}")

    # The existing per-seed aggregate must be the equal-weight mean over the
    # three locations, not an independent scenario-level replicate.
    agg_existing = pd.read_csv(P2_RHO_AGG)
    agg_p2 = agg_existing[agg_existing.model == "B5-P2"]
    if len(agg_p2) != 45 or agg_p2.duplicated(["training_seed", "modalities", "rho"]).any():
        raise RuntimeError("Existing P2 per-seed rho aggregate has unexpected shape")
    calc = p2.groupby(["training_seed", "modalities", "rho"], as_index=False)[list(METRICS)].mean()
    check = calc.merge(agg_p2, on=["training_seed", "modalities", "rho"], suffixes=("_raw", "_agg"), validate="one_to_one")
    for metric in METRICS:
        if not np.allclose(check[f"{metric}_raw"], check[f"{metric}_agg"], atol=1e-12, rtol=0):
            raise RuntimeError(f"Location aggregation mismatch against locked handoff: {metric}")

    # Audit all existing public-baseline scenario JSON and checkpoint hashes.
    baseline_rows = []
    baseline_clean_rows = []
    baseline_checks = []
    expected_ids = {s[0] for s in expected_scenarios}
    manifest_ckpts = {(r["model"], int(r["seed"])): r for r in base_manifest["checkpoints"]}
    for model in MODELS:
        for seed in SEEDS:
            result_path = E / f"experiments/q2/public_baselines/{model}/metrics_seed{seed}.json"
            result = load_json(result_path)
            rows = result["missing"]["rows"]
            missing = [r for r in rows if r["scenario_id"] != "clean"]
            clean_rows = [r for r in rows if r["scenario_id"] == "clean"]
            ids = [r["scenario_id"] for r in missing]
            scenario_ok = len(missing) == 54 and len(set(ids)) == 54 and set(ids) == expected_ids
            finite_ok = all(all(math.isfinite(float(r[m])) for m in ALL_METRICS) for r in missing)
            hash_ok = result["benchmark_definition_sha256"] == benchmark_sha
            clean_ok = len(clean_rows) == 1 and all(
                m in clean_rows[0] and math.isfinite(float(clean_rows[0][m])) for m in METRICS
            )
            ckpt = manifest_ckpts[(model, seed)]
            ckpt_path = E / ckpt["path"]
            ckpt_ok = ckpt_path.is_file() and ckpt_path.stat().st_size == ckpt["size_bytes"] and sha256(ckpt_path) == ckpt["sha256"]
            pass_all = result["smoke"] is False and scenario_ok and finite_ok and hash_ok and clean_ok and ckpt_ok
            baseline_checks.append({"model": model, "seed": seed, "records": len(missing),
                                    "scenario_complete": scenario_ok, "finite": finite_ok,
                                    "benchmark_sha_match": hash_ok, "clean_present": clean_ok,
                                    "checkpoint_sha_match": ckpt_ok, "pass": pass_all,
                                    "result_path": str(result_path.relative_to(ROOT)),
                                    "checkpoint_path": ckpt["path"], "checkpoint_sha256": ckpt["sha256"]})
            if not pass_all:
                raise RuntimeError(f"Public baseline data integrity failed: {model} seed {seed}")
            for r in missing:
                baseline_rows.append({
                    "model": model, "seed": seed, "scenario_id": r["scenario_id"],
                    "modalities": r["modalities"], "rho": r["rho"], "location": r["location"],
                    "sample_count": r.get("sample_count"),
                    **{m: float(r[m]) for m in ALL_METRICS},
                })
            clean = clean_rows[0]
            baseline_clean_rows.append({"model": model, "seed": seed,
                                        **{m: float(clean[m]) for m in METRICS},
                                        "selection_score": float(clean["selection_score"])})

    base_df = pd.DataFrame(baseline_rows)
    if len(base_df) != 486 or base_df.duplicated(["model", "seed", "scenario_id"]).any():
        raise RuntimeError("Public baseline union does not contain exactly 486 unique scenarios")
    # Cross-check derived per-seed means against the already-published summary.
    old_per_seed = pd.read_csv(PUBLIC / "baseline_missing_per_seed.csv")
    per_seed_new = base_df.groupby(["model", "seed"], as_index=False)[list(ALL_METRICS)].mean()
    compare = per_seed_new.merge(old_per_seed, on=["model", "seed"], suffixes=("_raw", "_summary"), validate="one_to_one")
    for metric in ALL_METRICS:
        if not np.allclose(compare[f"{metric}_raw"], compare[f"{metric}_summary"], atol=1e-10, rtol=0):
            raise RuntimeError(f"Public baseline per-seed summary mismatch: {metric}")
    # Cross-check selected checkpoint provenance against the lock manifest.
    if len(manifest_ckpts) != 9 or not all(r["pass"] for r in baseline_checks):
        raise RuntimeError("Public baseline seed/checkpoint coverage is incomplete")

    return {
        "lock": lock, "q2_ckpt": q2_ckpt, "benchmark": bm, "benchmark_sha256": benchmark_sha,
        "q2_p2_checkpoint_checks": q2_p2_checkpoint_checks,
        "expected_scenarios": expected_scenarios, "details": details, "p2_curve": p2,
        "baseline_rows": base_df, "baseline_clean_rows": pd.DataFrame(baseline_clean_rows),
        "baseline_checks": baseline_checks, "curve_complete": curve_complete,
        "curve_unique": curve_unique, "curve_finite": curve_finite,
        "curve_model_consistent": curve_model_consistent,
    }


def write_preflight(data: dict) -> str:
    p2 = data["p2_curve"]
    baseline_checks = data["baseline_checks"]
    per_model = {m: sum(r["model"] == m for r in baseline_checks) * 54 for m in MODELS}
    text = f"""# E2026 Q2 缺失规律与公开 Baseline 数据预检

**本地数据完整性检查先于绘图与制表。** 本报告根据已有本地结果文件生成；未训练、未运行推理、未读取 test、未读取 Attachment3/4，也未修改正文。

## A. 分模态缺失率曲线

- 场景来源：`E2026/outputs/final/q2/q2_plotting_handoff_v2/plot_data/scenario_details_complete.csv`（330 行，2 模型 × 3 seed × 55 条件）。
- P2 单模态场景记录：**{len(p2)}/135**；唯一组合：**{len(p2.drop_duplicates(['training_seed','modalities','rho','location']))}/135**。
- Seed：42/43/44；模态：文本/音频/视觉；ρ：0.1–0.5；位置：early/middle/late，全部齐全。
- Accuracy、Macro-F1、MAE、Pearson 数值全部有限；每条记录 `sample_count=728`。
- 冻结 benchmark SHA256：`{data['benchmark_sha256']}`；Q2 模型锁、benchmark manifest 与 benchmark definition 一致。
- P2 seed42/43/44 checkpoint 与 Q2 checkpoint manifest 校验通过；模型锁引用的多 seed 汇总结果 SHA256 也匹配。
- 与已完成 Figure 4 逐 seed 聚合文件核对：每个 seed × 模态 × ρ 均为三个位置等权平均，45/45 项四项指标一致（绝对容差 1e-12）。
- 完整输入参考值来自 `q2_model_lock.json` 的 P2 clean validation 三 seed 原始值，且与场景表 clean 行一致。

**CURVE_DATA_COMPLETE = YES**

## B. TFN / MulT / MISA 公开 Baseline 缺失场景

- 数据来源：每个模型/seed 的既有 `E2026/experiments/q2/public_baselines/<model>/metrics_seed<seed>.json`，其中含 clean 与逐场景结果；另有锁定摘要及 checkpoint manifest。
- 每模型缺失记录：TFN **{per_model['TFN']}/162**；MulT **{per_model['MulT']}/162**；MISA **{per_model['MISA']}/162**。
- Seed 42/43/44 各有 54/54 场景；三个模型的 scenario ID 集合与冻结 benchmark 的 54 项完全一致。
- 九份结果的 benchmark SHA、clean 行、finite 指标和非 smoke 状态均通过；九个 checkpoint 文件大小与 SHA256 均匹配 manifest。
- 全量缺失明细共 **{len(data['baseline_rows'])}/486** 条唯一模型 × seed × 场景记录；Accuracy、Macro-F1、MAE、Pearson、selection score 均有限。
- 不纳入 Attachment2 test、Attachment3 或 Attachment4。已存在的实验报告将这些 public baseline 定义为 Attachment2 train/valid 的适配训练与 valid 评估结果。

**BASELINE_MISSING_DATA_COMPLETE = YES**

## C. 门槛结论

两项数据门槛均满足，可以继续绘制分模态退化曲线并生成描述性缺失场景对比表。Baseline checkpoint 按 clean validation 选择，锁定 P2 按 robust score 选择；比较表必须标为**描述性比较**，不能据此声称严格公平的鲁棒性排名或显著优越。
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DATA_PREFLIGHT_REPORT.md").write_text(text, encoding="utf-8")
    PREFLIGHT_ROOT.write_text(text, encoding="utf-8")
    return text


def aggregate_curve(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    p2 = data["p2_curve"]
    per_seed = p2.groupby(["training_seed", "modalities", "rho"], as_index=False)[list(METRICS)].mean()
    per_seed = per_seed.rename(columns={"training_seed": "seed", "modalities": "modality"})
    expected_n = len(SEEDS) * len(MODALITIES) * len(RHOS)
    if len(per_seed) != expected_n:
        raise RuntimeError(f"Expected {expected_n} per-seed curve cells, got {len(per_seed)}")
    rows = []
    for (modality, rho), group in per_seed.groupby(["modality", "rho"], sort=False):
        group = group.set_index("seed").reindex(SEEDS)
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            rows.append({"modality": modality, "rho": float(rho), "metric": metric,
                         "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                         **{f"seed{s}": float(group.loc[s, metric]) for s in SEEDS}})
    aggregate = pd.DataFrame(rows).sort_values(["modality", "metric", "rho"],
                                                key=lambda s: s.map({"text": 0, "audio": 1, "vision": 2, "accuracy": 0, "macro_f1": 1, "mae": 2, "pearson": 3}) if s.name in ("modality", "metric") else s)
    return per_seed, aggregate.reset_index(drop=True)


def clean_stats(lock: dict) -> dict[str, tuple[float, float]]:
    result = {}
    for metric in METRICS:
        vals = np.asarray(lock["validation_summary"]["P2_clean"][metric]["values"], dtype=float)
        result[metric] = (float(vals.mean()), float(vals.std(ddof=1)))
    return result


def make_figure(stats: pd.DataFrame, clean: dict[str, tuple[float, float]], out_base: Path) -> None:
    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 120 * mm), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.105, right=0.97, bottom=0.14, top=0.82,
                          wspace=0.34, hspace=0.58)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    fig.suptitle("不同模态缺失比例下的性能退化", y=0.975, fontsize=9,
                 fontweight="bold", color=DARK)
    x = np.asarray(RHOS, dtype=float)
    panel_order = ("accuracy", "macro_f1", "mae", "pearson")
    panel_ids = ("a", "b", "c", "d")
    for ax, metric, panel in zip(axes, panel_order, panel_ids):
        ax.set_facecolor("white")
        ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(DARK)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=DARK, length=2.5, width=0.55, pad=2)
        ax.set_axisbelow(True)
        for modality in MODALITIES:
            part = stats[(stats.modality == modality) & (stats.metric == metric)].sort_values("rho")
            if len(part) != 5:
                raise RuntimeError(f"Expected five ratios for {modality}/{metric}")
            mean = part["mean"].to_numpy(float)
            sd = part["sd"].to_numpy(float)
            line_color = LINES[modality]
            band_color = PASTELS[modality]
            ax.fill_between(x, mean - sd, mean + sd, color=band_color,
                            alpha=0.35, linewidth=0, zorder=1)
            ax.plot(x, mean, color=line_color, linewidth=2.4, marker="o", markersize=4.5,
                    markerfacecolor=line_color, markeredgecolor="white", markeredgewidth=0.45,
                    label=LABELS[modality], zorder=3)
        center, _sd = clean[metric]
        ax.axhline(center, color="#777777", linewidth=0.85, linestyle=(0, (3, 2)), zorder=2)
        ax.set_xticks(x, [f"{r:.1f}" for r in x])
        ax.set_xlabel("缺失比例 ρ", labelpad=3)
        ax.set_ylabel(METRIC_LABELS[metric][0], labelpad=3)
        ax.set_xlim(0.07, 0.53)
        all_vals = stats.loc[stats.metric == metric, ["mean", "sd"]]
        lows = np.r_[all_vals["mean"].to_numpy() - all_vals["sd"].to_numpy(), center]
        highs = np.r_[all_vals["mean"].to_numpy() + all_vals["sd"].to_numpy(), center]
        span = float(highs.max() - lows.min())
        pad = max(span * 0.12, 0.008)
        ax.set_ylim(float(lows.min() - pad), float(highs.max() + pad))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.set_title(f"({panel}) {METRIC_LABELS[metric][1]}", loc="left", fontsize=8.2,
                     fontweight="bold", color=DARK, pad=7)
    handles = [
        Line2D([0], [0], color=LINES[m], marker="o", linewidth=2.4, markersize=4.5,
               markerfacecolor=LINES[m], markeredgecolor="white", label=LABELS[m])
        for m in MODALITIES
    ]
    handles.append(Line2D([0], [0], color="#777777", linewidth=0.85,
                          linestyle=(0, (3, 2)), label="完整输入（三 seed 均值）"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.91), ncol=4,
               frameon=False, fontsize=7.1, handlelength=2.0, columnspacing=1.6)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(out_base.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def derive_comparison(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []
    for (model, seed), part in data["baseline_rows"].groupby(["model", "seed"], sort=False):
        means = part[list(ALL_METRICS)].mean()
        per_seed_rows.append({"model": model, "seed": int(seed), **means.to_dict()})
    # P2 comes from the same 54 frozen missing scenarios in the completed Q2 table.
    p2 = data["details"][(data["details"].model == "B5-P2") &
                         (data["details"].scenario_id != "clean")].copy()
    for (seed, _group) in p2.groupby("training_seed"):
        pass
    for seed, part in p2.groupby("training_seed", sort=True):
        if len(part) != 54 or part.scenario_id.nunique() != 54:
            raise RuntimeError(f"P2 seed {seed} does not have the same 54 scenarios")
        means = part[list(METRICS) + ["selection_score"]].mean()
        per_seed_rows.append({"model": "本文模型（P2）", "seed": int(seed), **means.to_dict()})
    per_seed = pd.DataFrame(per_seed_rows)
    # Compute each seed's robust score using the frozen clean + missing rule.
    baseline_summary = {}
    for model in MODELS:
        for seed in SEEDS:
            obj = load_json(E / f"experiments/q2/public_baselines/{model}/metrics_seed{seed}.json")
            baseline_summary[(model, seed)] = float(obj["missing"]["summary"]["robust_score"])
    # Derive each P2 robust score from the same frozen definition: mean of
    # clean selection score and the seed's average across all 54 scenarios.
    p2_clean = pd.read_csv(PLOT_DATA / "scenario_details_complete.csv")
    p2_clean = p2_clean[(p2_clean.model == "B5-P2") & (p2_clean.scenario_id == "clean")].set_index("training_seed")
    for seed in SEEDS:
        miss = per_seed[(per_seed.model == "本文模型（P2）") & (per_seed.seed == seed)].iloc[0]
        baseline_summary[("本文模型（P2）", seed)] = float(
            0.5 * (p2_clean.loc[seed, "selection_score"] + miss["selection_score"])
        )
    for (model, seed), value in baseline_summary.items():
        # The table's means use per-seed mean over 54; robust score remains auxiliary.
        per_seed.loc[(per_seed.model == model) & (per_seed.seed == seed), "robust_score"] = value
    table_rows = []
    model_order = ["TFN", "MulT", "MISA", "本文模型（P2）"]
    for model in model_order:
        part = per_seed[per_seed.model == model].sort_values("seed")
        if len(part) != 3:
            raise RuntimeError(f"Comparison lacks three seed rows for {model}")
        row = {"model": model, "n_seeds": 3, "n_scenarios_per_seed": 54}
        for metric in ALL_METRICS + ("robust_score",):
            values = part[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_mean_sd"] = f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
        table_rows.append(row)
    return per_seed, pd.DataFrame(table_rows), pd.DataFrame(baseline_clean_rows_from_sources(data))


def baseline_clean_rows_from_sources(data: dict) -> list[dict]:
    rows = data["baseline_clean_rows"].to_dict(orient="records")
    lock = data["lock"]["validation_summary"]["P2_clean"]
    for i, seed in enumerate(SEEDS):
        rows.append({"model": "本文模型（P2）", "seed": seed,
                     **{m: float(lock[m]["values"][i]) for m in METRICS}})
    return rows


def write_outputs(data: dict, preflight_text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DATA_PREFLIGHT_REPORT.md").write_text(preflight_text, encoding="utf-8")
    per_seed_curve, curve = aggregate_curve(data)
    per_seed_curve.to_csv(OUT / "q2_modality_missing_per_seed.csv", index=False, float_format="%.12g")
    curve.to_csv(OUT / "q2_modality_missing_degradation_data.csv", index=False, float_format="%.12g")
    raw_p2 = data["p2_curve"].rename(columns={"training_seed": "seed", "modalities": "modality"})
    raw_p2.to_csv(OUT / "q2_modality_missing_scenarios_p2.csv", index=False, float_format="%.12g")
    clean = clean_stats(data["lock"])
    figure_base = OUT / "fig_q2_modality_missing_degradation"
    make_figure(curve, clean, figure_base)

    per_seed_compare, comparison, clean_all = derive_comparison(data)
    data["baseline_rows"].to_csv(OUT / "q2_public_baseline_missing_scenarios_per_seed.csv", index=False, float_format="%.12g")
    per_seed_compare.to_csv(OUT / "q2_public_baseline_missing_per_seed.csv", index=False, float_format="%.12g")
    comparison.to_csv(OUT / "q2_public_baseline_missing_comparison.csv", index=False, float_format="%.12g")
    clean_all.to_csv(OUT / "q2_all_models_clean_per_seed.csv", index=False, float_format="%.12g")

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{固定 54 个缺失场景上的描述性性能比较（均值 $\pm$ 样本标准差）}",
        r"\label{tab:q2-public-baseline-missing}", r"\small",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"方法 & 准确率 $\uparrow$ & 宏平均 F1 $\uparrow$ & MAE $\downarrow$ & Pearson $\uparrow$ \\",
        r"\midrule",
    ]
    names = {"TFN": "TFN", "MulT": "MulT", "MISA": "MISA", "本文模型（P2）": "本文模型"}
    for row in comparison.to_dict(orient="records"):
        vals = [row[f"{m}_mean_sd"].replace(" ± ", r" $\pm$ ") for m in METRICS]
        lines.append(names[row["model"]] + " & " + " & ".join(vals) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\parbox{\linewidth}{\footnotesize 注：每个 seed 先对 54 个场景求均值，再对 3 个 seed 计算均值与样本标准差（$n=3$）。该表仅作描述性比较；公开模型按 clean validation 选 checkpoint，本文模型按冻结 robust score 选 checkpoint，选模准则不同，不据此宣称严格公平的鲁棒性排名。}",
        r"\end{table}",
    ])
    (OUT / "q2_public_baseline_missing_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    curve_audit = [
        "# 分模态缺失比例曲线审计", "",
        "## 来源与样本单位", "",
        f"- 场景级输入：`{P2_DETAIL.relative_to(ROOT).as_posix()}`；完整场景表 330 行。",
        f"- P2 单模态缺失行：{len(raw_p2)}（seed × modality × rho × location = 3 × 3 × 5 × 3）。每行对应一次 Attachment2 valid 条件评估，不是一次训练。",
        f"- Benchmark SHA256：`{data['benchmark_sha256']}`。位置列表为 early/middle/late。",
        "- 每个 seed × 模态 × rho 内先对三个位置作等权平均；每条曲线点是三 seed 均值，带状区域为 seed 间样本标准差（ddof=1，n=3）。不把位置或场景当独立重复。",
        "- 完整输入虚线为 Q2 锁定模型 clean valid 的三 seed 均值。", "",
        "## 输出数据", "",
        "- `q2_modality_missing_scenarios_p2.csv`：135 条位置级原始场景值。",
        "- `q2_modality_missing_per_seed.csv`：每个 seed × 模态 × rho 的位置平均值。",
        "- `q2_modality_missing_degradation_data.csv`：绘图均值、样本 SD 和 seed42/43/44 明细。",
        "- 四个 panel 分别为 Accuracy、Macro-F1、MAE、Pearson；MAE 越低越好，其余越高越好。", "",
        "## 端点变化（rho=0.5 减 rho=0.1）", "",
        "| 指标 | 文本缺失 | 音频缺失 | 视觉缺失 |", "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        vals = []
        for modality in MODALITIES:
            p = curve[(curve.metric == metric) & (curve.modality == modality)].set_index("rho")
            vals.append(float(p.loc[0.5, "mean"] - p.loc[0.1, "mean"]))
        curve_audit.append(f"| {metric} | " + " | ".join(f"{v:+.4f}" for v in vals) + " |")
    curve_audit.extend([
        "", "## 图注草稿", "",
        "图X 不同模态缺失比例下的性能退化。每个ρ下先对 early、middle、late 三个位置等权平均，再计算三次随机初始化的均值；阴影表示三 seed 间样本标准差。虚线表示完整输入验证集的三 seed 均值。结果为 Attachment2 validation 上的描述性比较，不应解释为现实因果效应。",
    ])
    (OUT / "q2_modality_missing_degradation_audit.md").write_text("\n".join(curve_audit) + "\n", encoding="utf-8")

    baseline_audit = [
        "# 公开 Baseline 缺失场景比较审计", "",
        "## 数据来源与完整性", "",
        "- TFN/MulT/MISA 逐场景结果来自 `E2026/experiments/q2/public_baselines/<model>/metrics_seed{42,43,44}.json`，已逐条核验，不从论文表格反推。",
        f"- 每模型 3 seed × 54 场景 = 162 条；总计 {len(data['baseline_rows'])} 条唯一缺失场景记录。所有场景 ID 与冻结 benchmark 完全一致。",
        f"- 每个结果文件的 benchmark SHA256 与冻结值一致：`{data['benchmark_sha256']}`；每个文件均标记 `smoke=false`，clean 行存在，指标数值有限。",
        "- 九个公开 baseline checkpoint 的文件大小及 SHA256 均与 `baseline_checkpoint_manifest.json` 一致。本文模型 P2 的 54 场景值来自已经锁定的 Q2 完整场景表。",
        "- 指标统计：每 seed 先对 54 场景作等权平均；之后按三个 seed 计算均值与样本 SD（ddof=1）。有效重复单位是 seed（n=3），不是 162 个 seed × scenario。",
        "- 数据边界：Attachment2 train/valid 已有结果；没有使用 Attachment2 test、Attachment3 或 Attachment4；不重新训练或推理。", "",
        "## 缺失场景比较（均值 ± 样本 SD）", "",
        "| 方法 | Accuracy↑ | Macro-F1↑ | MAE↓ | Pearson↑ | R（辅助） |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict(orient="records"):
        baseline_audit.append("| " + " | ".join([
            row["model"], *[row[f"{m}_mean_sd"] for m in METRICS], row["robust_score_mean_sd"]
        ]) + " |")
    baseline_audit.extend([
        "", "## 解释边界", "",
        "公开 TFN/MulT/MISA checkpoint 按 clean validation 选择；已锁定的 P2 checkpoint 按 robust score 选择。由于选模准则不一致，本表只能称为**描述性缺失场景比较**，不用于声称本文模型严格优于公开模型或显著超过全部 baseline。公开 baseline 是基于对齐特征的架构适配实现，不能称作原论文公开数字的复现。",
        "", "## 文件", "",
        "- `q2_public_baseline_missing_scenarios_per_seed.csv`：486 条公开 baseline 原始场景结果。",
        "- `q2_public_baseline_missing_per_seed.csv`：每个模型/seed 的 54 场景平均。",
        "- `q2_public_baseline_missing_comparison.csv`：三 seed 均值/SD 与展示字段。",
        "- `q2_public_baseline_missing_comparison.tex`：可人工审阅的 LaTeX 表格片段；未自动插入论文。",
        "- `q2_all_models_clean_per_seed.csv`：已有 clean validation 逐 seed 对照。",
    ])
    (OUT / "q2_public_baseline_missing_audit.md").write_text("\n".join(baseline_audit) + "\n", encoding="utf-8")

    # Preserve source metrics, manifests and reports for later independent review.
    sources = [MODEL_LOCK, Q2_CKPT_MANIFEST, BENCH_FILE, BENCH_MANIFEST, BASELINE_MANIFEST,
               BASELINE_REPORT, BASELINE_INTEGRATION, COMPLETION_CHECK, P2_DETAIL, P2_RHO_AGG,
               PUBLIC / "baseline_missing_per_seed.csv", PUBLIC / "baseline_missing_mean_sd.csv",
               PUBLIC / "baseline_clean_per_seed.csv"]
    copied = []
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = OUT / "source_files" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({"source": str(source.relative_to(ROOT)).replace("\\", "/"),
                       "bundle_path": str(destination.relative_to(OUT)).replace("\\", "/"),
                       "size_bytes": source.stat().st_size, "sha256": sha256(source)})
    for model in MODELS:
        for seed in SEEDS:
            source = E / f"experiments/q2/public_baselines/{model}/metrics_seed{seed}.json"
            destination = OUT / "source_metrics" / model / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append({"source": str(source.relative_to(ROOT)).replace("\\", "/"),
                           "bundle_path": str(destination.relative_to(OUT)).replace("\\", "/"),
                           "size_bytes": source.stat().st_size, "sha256": sha256(source)})
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    provenance = {
        "bundle_status": "LOCAL_RESULTS_ONLY_NO_TRAINING_NO_INFERENCE",
        "benchmark_sha256": data["benchmark_sha256"],
        "q2_model_lock": str(MODEL_LOCK.relative_to(ROOT)).replace("\\", "/"),
        "p2_curve_rows": 135,
        "public_baseline_missing_rows": 486,
        "seed_replicates": list(SEEDS),
        "scenario_ids_sha256": hashlib.sha256("\n".join(sorted(s[0] for s in data["expected_scenarios"])).encode()).hexdigest(),
        "source_files": copied,
        "p2_checkpoint_hashes_verified": data["q2_p2_checkpoint_checks"],
        "baseline_checkpoint_hashes_verified": True,
        "test_attachment3_attachment4_used": False,
        "paper_modified": False,
    }
    (OUT / "source_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Q2 缺失规律与公开 Baseline 结果包

此包根据 E2026 本地锁定 validation 结果生成。没有训练模型、加载 checkpoint 做推理、使用 test/Attachment3/4，或修改论文正文。

## 核心交付

- `fig_q2_modality_missing_degradation.png/.pdf/.svg`：P2 在文本/音频/视觉单模态缺失下的四指标退化曲线。
- `q2_modality_missing_degradation_data.csv`：60 个模态 × rho × 指标统计单元，含三 seed 数值及 mean/sample SD。
- `q2_modality_missing_scenarios_p2.csv`：对应的 135 条位置级记录。
- `q2_public_baseline_missing_comparison.csv/.tex`：TFN、MulT、MISA 与本文 P2 在固定 54 个缺失场景上的描述性比较。
- `q2_public_baseline_missing_scenarios_per_seed.csv`：486 条公开模型 × seed × 场景明细。
- 两份审计、完整预检报告、checkpoint/benchmark 来源清单及逐 seed baseline 结果 JSON。

## 统计规则

对每个 seed × 模态 × rho，先对 early/middle/late 等权平均；再以 seed 为独立重复计算 mean 和 sample SD（ddof=1，n=3）。Baseline 对每 seed 的 54 个场景求均值后，同样按 3 seed 统计。图中阴影表示 seed 间样本 SD。

## 论文表述边界

公开模型按 clean validation 选 checkpoint，P2 按冻结 robust score 选 checkpoint；baseline 比较仅为描述性结果，不支持严格公平的鲁棒性排名或“显著优于”的结论。完整输入 clean参考来自锁定 P2 clean validation。

## 复核

- `DATA_PREFLIGHT_REPORT.md`：两项完整性门槛。
- `q2_modality_missing_degradation_audit.md`：曲线来源、聚合、端点变化与图注。
- `q2_public_baseline_missing_audit.md`：baseline 计数、来源、统计单位与选模限制。
- `source_manifest.json`：源路径、文件大小、SHA256 与数据边界。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    # Machine-readable figures/row gates for the completed bundle.
    qa = {
        "curve_data_complete": True, "curve_scenario_rows": int(len(raw_p2)),
        "curve_summary_rows": int(len(curve)), "baseline_missing_complete": True,
        "baseline_scenario_rows": int(len(data["baseline_rows"])),
        "baseline_per_seed_rows": int(len(per_seed_compare)),
        "benchmark_sha256": data["benchmark_sha256"],
        "all_metric_outputs_finite": bool(np.isfinite(curve.select_dtypes(include=[np.number]).to_numpy()).all()
                                           and np.isfinite(comparison.select_dtypes(include=[np.number]).to_numpy()).all()),
        "clean_reference_source": "q2_model_lock.json validation_summary.P2_clean",
        "mean_sd_unit": "seed; sample SD ddof=1; n=3",
        "paper_modified": False,
    }
    (OUT / "bundle_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not qa["all_metric_outputs_finite"]:
        raise RuntimeError("Final bundle contains non-finite summary values")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT.parent))
    print(f"Bundle directory: {OUT}")
    print(f"ZIP: {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")
    print(f"Curve rows={len(raw_p2)}, aggregate rows={len(curve)}, baseline missing rows={len(data['baseline_rows'])}")
    print(comparison[["model", *[f"{m}_mean_sd" for m in METRICS], "robust_score_mean_sd"]].to_string(index=False))


def main() -> None:
    data = validate_sources()
    preflight = write_preflight(data)
    # Hard stop after the preflight report if a completeness gate is not met.
    if "CURVE_DATA_COMPLETE = YES" not in preflight or "BASELINE_MISSING_DATA_COMPLETE = YES" not in preflight:
        raise RuntimeError("Preflight gate did not pass; outputs beyond the report are blocked")
    write_outputs(data, preflight)


if __name__ == "__main__":
    main()
