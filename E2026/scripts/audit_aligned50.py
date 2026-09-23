#!/usr/bin/env python3
"""Read-only structural and label audit for aligned_50.pkl and label.xlsx."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def label_name(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "invalid"
    if not math.isfinite(x):
        return "invalid"
    if x < 0:
        return "Negative"
    if x == 0:
        return "Neutral"
    return "Positive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--xlsx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # The pickle is supplied project data. Load once, read-only; never re-save it.
    import pickle
    with args.pkl.open("rb") as f:
        data = pickle.load(f)

    anomalies: list[dict] = []
    report: dict = {
        "inputs": {"pkl": str(args.pkl), "xlsx": str(args.xlsx)},
        "pkl_top_level_type": type(data).__name__,
        "top_level_keys": list(data.keys()) if isinstance(data, dict) else None,
        "required_splits_present": {s: isinstance(data, dict) and s in data for s in ("train", "valid", "test")},
        "splits": {},
        "split_overlap": {},
        "labels": {},
        "xlsx_crosscheck": {},
        "padding": {},
        "duplicate_samples": {},
    }

    split_ids: dict[str, list[str]] = {}
    all_ids: dict[str, str] = {}
    field_names = sorted({k for v in data.values() if isinstance(v, dict) for k in v})
    modality_names = [m for m in ("text", "audio", "vision") if m in field_names]
    zero_masks: dict[str, np.ndarray] = {}
    global_feature_digest_owner: dict[str, tuple[str, str, int]] = {}
    global_duplicate_pairs = []

    for split_name, split in data.items():
        if not isinstance(split, dict):
            anomalies.append({"type": "split_not_mapping", "split": split_name, "detail": type(split).__name__})
            continue
        n_candidates = [len(v) for v in split.values() if hasattr(v, "__len__")]
        n = n_candidates[0] if n_candidates else 0
        summary = {"sample_count": n, "fields": {}, "id": {}}
        ids_raw = split.get("id")
        ids = [str(x) for x in ids_raw] if ids_raw is not None else []
        split_ids[split_name] = ids
        id_counts = Counter(ids)
        dup_ids = [x for x, c in id_counts.items() if c > 1]
        id_fmt = Counter("matches video$_$clip" if re.fullmatch(r".+\$_\$\d+", x) else "other" for x in ids)
        summary["id"] = {
            "present": ids_raw is not None,
            "count": len(ids),
            "unique_count": len(id_counts),
            "duplicate_count": sum(c - 1 for c in id_counts.values()),
            "duplicate_examples": dup_ids[:10],
            "format_counts": dict(id_fmt),
            "examples": ids[:3],
        }
        if len(ids) != n:
            anomalies.append({"type": "id_length_mismatch", "split": split_name, "detail": f"ids={len(ids)} samples={n}"})
        for sample_id in ids:
            if sample_id in all_ids:
                anomalies.append({"type": "duplicate_id_across_splits", "split": split_name, "sample_id": sample_id, "detail": all_ids[sample_id]})
            else:
                all_ids[sample_id] = split_name

        # Inspect observed nested metadata for a possible temporal validity mask.
        text_bert = np.asarray(split["text_bert"]) if "text_bert" in split else None
        valid_mask = None
        valid_mask_info = {"candidate_field": "text_bert" if text_bert is not None else None, "candidate_component_index": 1 if text_bert is not None and text_bert.ndim == 3 and text_bert.shape[1] > 1 else None, "validated_binary_prefix_suffix_pattern": False}
        if text_bert is not None and text_bert.ndim == 3 and text_bert.shape[0] == n and text_bert.shape[1] > 1:
            candidate = text_bert[:, 1, :]
            if np.isin(candidate, [0, 1]).all() and np.all(np.diff(candidate, axis=1) <= 0) and np.all(candidate.sum(axis=1) > 0):
                valid_mask = candidate.astype(bool)
                lengths = candidate.sum(axis=1)
                valid_mask_info.update({"validated_binary_prefix_suffix_pattern": True, "timesteps": int(candidate.shape[1]), "min_valid_length": int(lengths.min()), "median_valid_length": float(np.median(lengths)), "max_valid_length": int(lengths.max()), "zero_suffix_for_all_samples": True})
        summary["observed_validity_mask_candidate"] = valid_mask_info

        raw_text = split.get("raw_text")
        empty_text = []
        if raw_text is not None:
            empty_text = [i for i, x in enumerate(raw_text) if not str(x).strip()]
            for i in empty_text[:100]:
                anomalies.append({"type": "empty_raw_text", "split": split_name, "sample_id": ids[i] if i < len(ids) else "", "index": i, "detail": "empty or whitespace"})
        summary["id"]["empty_raw_text_count"] = len(empty_text)

        # Per-field shape, dtype and numeric quality statistics.
        for field, value in split.items():
            arr = np.asarray(value)
            fs = {"python_type": type(value).__name__, "shape": list(arr.shape), "dtype": str(arr.dtype)}
            if arr.ndim == 0 or len(arr) != n:
                anomalies.append({"type": "field_sample_count_mismatch", "split": split_name, "field": field, "detail": f"shape={arr.shape}, expected first dim {n}"})
            if np.issubdtype(arr.dtype, np.number):
                x = arr
                finite = np.isfinite(x)
                fs.update({
                    "min": float(np.min(x[finite])) if finite.any() else None,
                    "max": float(np.max(x[finite])) if finite.any() else None,
                    "mean": float(np.mean(x[finite])) if finite.any() else None,
                    "std_population": float(np.std(x[finite])) if finite.any() else None,
                    "nan_count": int(np.isnan(x).sum()),
                    "inf_count": int(np.isinf(x).sum()),
                    "finite_count": int(finite.sum()),
                    "zero_count": int((x == 0).sum()),
                    "zero_sample_count": int(np.all(x == 0, axis=tuple(range(1, x.ndim))).sum()) if x.ndim > 1 else None,
                })
                fs["max_abs_exact"] = max(abs(fs["min"]), abs(fs["max"])) if finite.any() else None
                fs["abs_quantiles_sampled"] = {}
                if finite.any():
                    flat = x.reshape(-1)
                    sample_size = min(flat.size, 2_000_000)
                    if flat.size <= sample_size:
                        sampled = flat
                    else:
                        sampled = flat[np.random.default_rng(20260923).integers(0, flat.size, size=sample_size)]
                    sampled = np.abs(sampled[np.isfinite(sampled)].astype(np.float64, copy=False))
                    fs["abs_quantiles_sampled"] = {q: float(v) for q, v in zip(("0.5", "0.99", "0.999", "0.9999"), np.quantile(sampled, [0.5, 0.99, 0.999, 0.9999]))}
                    fs["abs_quantiles_sample_size"] = int(sampled.size)
                    fs["abs_quantiles_sampling"] = "exact if <= 2,000,000 values; otherwise deterministic uniform sampling with replacement (seed 20260923), capped at 2,000,000 values"
                fs["count_abs_gt_1e6"] = int(np.count_nonzero(x > 1e6) + np.count_nonzero(x < -1e6))
                fs["count_abs_ge_499_9"] = int(np.count_nonzero(x >= 499.9) + np.count_nonzero(x <= -499.9))
                if field == "audio" and fs["count_abs_ge_499_9"]:
                    for i, t, q in np.argwhere((x >= 499.9) | (x <= -499.9))[:100]:
                        anomalies.append({"type": "audio_boundary_value_review", "split": split_name, "field": field, "sample_id": str(split.get("id", [""])[int(i)]), "index": int(i), "detail": f"timestep={int(t)}, feature_dim={int(q)}, value={float(x[i, t, q])}; boundary-valued observation, not classified as invalid"})
                if arr.ndim == 3 and arr.shape[0] == n and arr.shape[1] <= 8:
                    fs["component_axis1_profiles"] = []
                    for component in range(arr.shape[1]):
                        c = arr[:, component, :]
                        fs["component_axis1_profiles"].append({
                            "component_index": component,
                            "min": float(np.min(c)), "max": float(np.max(c)),
                            "zero_ratio": float(np.mean(c == 0)),
                            "unique_value_count_capped_at_10000": int(min(np.unique(c).size, 10000)),
                        })
                if fs["nan_count"] or fs["inf_count"] or fs["count_abs_gt_1e6"]:
                    anomalies.append({"type": "numeric_nonfinite_or_extreme", "split": split_name, "field": field, "detail": {k: fs[k] for k in ("nan_count", "inf_count", "count_abs_gt_1e6", "min", "max")}})
            elif arr.dtype.kind in "USO":
                fs["empty_count"] = int(sum(not str(x).strip() for x in arr.reshape(-1)))
            summary["fields"][field] = fs

        # Temporal diagnostics from actual rank-3 modality fields.
        for mod in modality_names:
            arr = np.asarray(split[mod])
            if arr.ndim != 3:
                anomalies.append({"type": "modality_not_rank3", "split": split_name, "field": mod, "detail": list(arr.shape)})
                continue
            zmask = np.all(arr == 0, axis=2)
            zero_masks[f"{split_name}:{mod}"] = zmask
            t = zmask.shape[1]
            leading = trailing = interior = allzero_rows = 0
            for row in zmask:
                nz = np.flatnonzero(~row)
                if not nz.size:
                    allzero_rows += 1
                    continue
                first, last = int(nz[0]), int(nz[-1])
                leading += int(row[:first].sum())
                trailing += int(row[last + 1 :].sum())
                interior += int(row[first : last + 1].sum())
            dim_var = np.nanvar(arr.astype(np.float64), axis=(0, 1))
            constant_dims = np.flatnonzero(dim_var == 0).tolist()
            summary["fields"][mod]["temporal"] = {
                "timesteps_per_sample": t,
                "all_zero_sample_count": int(np.all(zmask, axis=1).sum()),
                "all_zero_timestep_count": int(zmask.sum()),
                "all_zero_timestep_ratio": float(zmask.mean()),
                "leading_zero_timestep_count": leading,
                "trailing_zero_timestep_count": trailing,
                "interior_zero_timestep_count": interior,
                "samples_with_interior_zero_timestep": int(sum(bool(np.any(row[np.flatnonzero(~row)[0]:np.flatnonzero(~row)[-1]+1])) for row in zmask if (~row).any())),
                "constant_feature_dimensions_global": constant_dims,
                "constant_feature_dimension_count": len(constant_dims),
            }
            if valid_mask is not None and valid_mask.shape == zmask.shape:
                zvalid = zmask & valid_mask
                zpad = zmask & ~valid_mask
                nonzero_pad = ~zmask & ~valid_mask
                early = middle = late = 0
                for rowz, rowv in zip(zmask, valid_mask):
                    vi = np.flatnonzero(rowv)
                    if vi.size:
                        chunks = np.array_split(vi, 3)
                        early += int(rowz[chunks[0]].sum())
                        middle += int(rowz[chunks[1]].sum())
                        late += int(rowz[chunks[2]].sum())
                summary["fields"][mod]["temporal"]["against_observed_text_bert_mask"] = {
                    "all_zero_timestep_within_valid_positions": int(zvalid.sum()),
                    "valid_position_count": int(valid_mask.sum()),
                    "all_zero_rate_within_valid_positions": float(zvalid.sum() / valid_mask.sum()),
                    "all_zero_timestep_within_masked_suffix": int(zpad.sum()),
                    "masked_suffix_position_count": int((~valid_mask).sum()),
                    "nonzero_modality_timestep_inside_masked_suffix": int(nonzero_pad.sum()),
                    "valid_region_zero_counts_by_early_middle_late": {"early": early, "middle": middle, "late": late},
                    "samples_all_zero_across_valid_region": int(np.all(zmask | ~valid_mask, axis=1).sum()),
                }
            if np.any(zmask):
                for i in np.flatnonzero(np.all(arr == 0, axis=(1, 2)))[:100]:
                    anomalies.append({"type": "all_zero_modality_sample", "split": split_name, "field": mod, "sample_id": ids[int(i)] if int(i) < len(ids) else "", "index": int(i), "detail": "entire modality sequence is zero"})

        # Exact duplicate multimodal samples, based on raw feature bytes.
        if ids and modality_names:
            digests = global_feature_digest_owner
            duplicate_groups = []
            for i in range(n):
                h = hashlib.sha256()
                for mod in modality_names:
                    h.update(np.ascontiguousarray(split[mod][i]).view(np.uint8))
                key = h.hexdigest()
                if key in digests:
                    owner_split, owner_id, owner_i = digests[key]
                    duplicate_groups.append((owner_split, owner_id, owner_i, i))
                    global_duplicate_pairs.append({"split_a": owner_split, "id_a": owner_id, "split_b": split_name, "id_b": ids[i]})
                else:
                    digests[key] = (split_name, ids[i], i)
            summary["exact_duplicate_multimodal_feature_pairs"] = len(duplicate_groups)
            summary["exact_duplicate_multimodal_feature_examples"] = [
                {"split1": a, "id1": b, "split2": split_name, "id2": ids[j]} for a, b, _, j in duplicate_groups[:10]
            ]
            for a, b, _, j in duplicate_groups:
                anomalies.append({"type": "exact_duplicate_multimodal_features", "split": f"{a},{split_name}", "sample_id": ids[j], "detail": f"same modality feature bytes as {b}"})

        report["splits"][split_name] = summary

    # Cross-modal zero-timestep agreement, using shared IDs and 50-step arrays.
    padding = {}
    for split_name in split_ids:
        ms = [zero_masks.get(f"{split_name}:{m}") for m in modality_names]
        if len(ms) == 3 and all(m is not None and m.shape == ms[0].shape for m in ms):
            pairs = {}
            for i, a in enumerate(modality_names):
                for b in modality_names[i + 1 :]:
                    ma, mb = zero_masks[f"{split_name}:{a}"], zero_masks[f"{split_name}:{b}"]
                    pairs[f"{a}__{b}"] = {"exact_mask_agreement_ratio": float(np.mean(ma == mb)), "both_zero_positions": int((ma & mb).sum()), "only_first_zero": int((ma & ~mb).sum()), "only_second_zero": int((~ma & mb).sum())}
            padding[split_name] = pairs
    report["padding"]["cross_modality_zero_mask_comparison"] = padding
    report["padding"]["explicit_length_fields"] = [k for k in field_names if "length" in k.lower() or "mask" in k.lower()]
    report["padding"]["nested_mask_candidates"] = {s: v.get("observed_validity_mask_candidate") for s, v in report["splits"].items()}
    report["padding"]["interpretation_note"] = "All-zero timestep is an observed data property; the candidate binary prefix/suffix mask inside text_bert is checked against each modality. It is not assumed that all zero feature vectors mean padding or missing."
    report["duplicate_samples"]["exact_multimodal_feature_duplicate_pair_count_across_all_splits"] = len(global_duplicate_pairs)
    report["duplicate_samples"]["exact_multimodal_feature_duplicate_examples"] = global_duplicate_pairs[:20]

    # Split intersections and duplicate IDs.
    names = list(split_ids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = sorted(set(split_ids[a]) & set(split_ids[b]))
            report["split_overlap"][f"{a}__{b}"] = {"overlap_count": len(inter), "examples": inter[:20]}
            for sid in inter:
                anomalies.append({"type": "split_id_overlap", "split": f"{a},{b}", "sample_id": sid, "detail": "ID occurs in multiple splits"})

    # Labels: derive polarity from regression by the problem's strict sign rule.
    label_rows = []
    label_summary = {}
    class_order = ["Negative", "Neutral", "Positive"]
    for split_name, split in data.items():
        if not isinstance(split, dict):
            continue
        ids = [str(x) for x in split.get("id", [])]
        ann = np.asarray(split.get("annotations", [])) if "annotations" in split else None
        cls = np.asarray(split.get("classification_labels", []))
        reg = np.asarray(split.get("regression_labels", []))
        cls_values = Counter(str(x) for x in cls.tolist())
        ann_values = Counter(str(x) for x in ann.tolist()) if ann is not None else Counter()
        mismatch_cls = []
        for i, x in enumerate(cls):
            # Preserve actual values; only normalize recognizable strings for comparison.
            s = str(x).strip().lower()
            cls_norm = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}.get(s)
            if cls_norm is not None and i < len(reg) and cls_norm != label_name(reg[i]):
                mismatch_cls.append(i)
        label_summary[split_name] = {
            "classification_labels_present": "classification_labels" in split,
            "classification_raw_value_counts": dict(cls_values),
            "annotations_present": ann is not None,
            "annotation_raw_value_counts": dict(ann_values),
            "regression": {},
            "classification_vs_regression_mismatch_count": len(mismatch_cls),
            "classification_vs_regression_mismatch_examples": [{"id": ids[i] if i < len(ids) else "", "classification": str(cls[i]), "regression": float(reg[i])} for i in mismatch_cls[:20]],
        }
        if reg.size:
            finite = np.isfinite(reg.astype(float))
            rv = reg.astype(float)[finite]
            label_summary[split_name]["regression"] = {
                "count": int(reg.size), "nan_count": int(np.isnan(reg.astype(float)).sum()), "inf_count": int(np.isinf(reg.astype(float)).sum()),
                "min": float(rv.min()) if rv.size else None, "max": float(rv.max()) if rv.size else None,
                "mean": float(rv.mean()) if rv.size else None, "std_population": float(rv.std()) if rv.size else None,
                "quantiles": {str(q): float(np.quantile(rv, q)) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)} if rv.size else {},
                "sign_rule_class_counts": dict(Counter(label_name(x) for x in rv)),
            }
        for i in mismatch_cls:
            anomalies.append({"type": "classification_regression_polarity_mismatch", "split": split_name, "sample_id": ids[i] if i < len(ids) else "", "index": i, "detail": f"classification={cls[i]}, regression={reg[i]}"})
    report["labels"] = label_summary

    # Excel cross-check by observed columns and pkl ID convention.
    xls = pd.ExcelFile(args.xlsx)
    report["xlsx_crosscheck"]["sheet_names"] = xls.sheet_names
    xl_parts = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(args.xlsx, sheet_name=sheet)
        xl_parts.append(df.assign(_sheet=sheet))
    xdf = pd.concat(xl_parts, ignore_index=True) if xl_parts else pd.DataFrame()
    report["xlsx_crosscheck"]["row_count"] = int(len(xdf))
    report["xlsx_crosscheck"]["columns"] = list(xdf.columns)
    id_cols = [c for c in ("video_id", "clip_id") if c in xdf.columns]
    if len(id_cols) == 2:
        xdf["_id"] = xdf["video_id"].astype(str) + "$_$" + xdf["clip_id"].map(lambda x: str(int(x)) if pd.notna(x) and float(x).is_integer() else str(x))
    else:
        xdf["_id"] = ""
    pkl_id_set = set(all_ids)
    xlsx_id_counts = Counter(xdf["_id"].astype(str))
    xlsx_id_set = set(xlsx_id_counts)
    report["xlsx_crosscheck"].update({
        "constructed_id_rule": "video_id + '$_$' + clip_id" if len(id_cols) == 2 else None,
        "id_duplicate_count": sum(c - 1 for c in xlsx_id_counts.values()),
        "pkl_ids_missing_from_xlsx_count": len(pkl_id_set - xlsx_id_set),
        "pkl_ids_missing_from_xlsx_examples": sorted(pkl_id_set - xlsx_id_set)[:20],
        "xlsx_ids_missing_from_pkl_count": len(xlsx_id_set - pkl_id_set),
        "xlsx_ids_missing_from_pkl_examples": sorted(xlsx_id_set - pkl_id_set)[:20],
        "one_to_one_id_set_match": pkl_id_set == xlsx_id_set and len(xdf) == len(pkl_id_set),
        "mode_counts": dict(Counter(xdf["mode"].astype(str))) if "mode" in xdf.columns else {},
    })
    for s in data:
        pids = set(split_ids.get(s, []))
        if "mode" in xdf.columns:
            xids = set(xdf.loc[xdf["mode"].astype(str).str.lower() == str(s).lower(), "_id"].astype(str))
            report["xlsx_crosscheck"].setdefault("per_split", {})[s] = {"pkl_count": len(pids), "xlsx_count": len(xids), "id_set_match": pids == xids, "pkl_only_count": len(pids - xids), "xlsx_only_count": len(xids - pids)}

    # Compare label and annotation columns where present.
    label_matches = {"label_column": "label" in xdf.columns, "annotation_column": "annotation" in xdf.columns, "label_numeric_mismatch_count": None, "annotation_pkl_field_present": "annotations" in field_names, "annotation_pkl_mismatch_count": None}
    pkl_by_id = {}
    for sname, split in data.items():
        for i, sid in enumerate(split.get("id", [])):
            pkl_by_id[str(sid)] = {
                "reg": float(split["regression_labels"][i]) if "regression_labels" in split else None,
                "cls": str(split["classification_labels"][i]) if "classification_labels" in split else None,
                "ann": str(split["annotations"][i]) if "annotations" in split else None,
                "split": sname,
            }
    label_mis, ann_mis = [], []
    if "label" in xdf.columns:
        for _, row in xdf.iterrows():
            sid = str(row["_id"])
            if sid in pkl_by_id and pd.notna(row["label"]):
                a, b = pkl_by_id[sid]["reg"], float(row["label"])
                if a is not None and not np.isclose(a, b, rtol=0, atol=1e-7):
                    label_mis.append((sid, a, b))
    if "annotation" in xdf.columns:
        for _, row in xdf.iterrows():
            sid = str(row["_id"])
            if sid in pkl_by_id and pkl_by_id[sid]["ann"] is not None and pd.notna(row["annotation"]):
                if pkl_by_id[sid]["ann"].strip().lower() != str(row["annotation"]).strip().lower():
                    ann_mis.append((sid, pkl_by_id[sid]["ann"], str(row["annotation"])))
    label_matches["label_numeric_mismatch_count"] = len(label_mis)
    label_matches["label_numeric_mismatch_examples"] = [{"id": a, "pkl": b, "xlsx": c} for a, b, c in label_mis[:20]]
    label_matches["annotation_pkl_mismatch_count"] = len(ann_mis) if "annotations" in field_names else None
    label_matches["annotation_pkl_mismatch_examples"] = [{"id": a, "pkl": b, "xlsx": c} for a, b, c in ann_mis[:20]]
    report["xlsx_crosscheck"]["label_comparison"] = label_matches

    # Infer the stored classification encoding from observed Excel annotations,
    # and compare the regression sign rule against those same annotations.
    cls_ann = Counter()
    reg_ann = Counter()
    cls_mis = []
    reg_mis = []
    canonical_annotations = {"negative", "neutral", "positive"}
    for _, row in xdf.iterrows():
        sid = str(row["_id"])
        if sid not in pkl_by_id:
            continue
        truth = str(row["annotation"]).strip().lower() if "annotation" in xdf.columns and pd.notna(row.get("annotation")) else ""
        item = pkl_by_id[sid]
        if truth in canonical_annotations:
            cls_ann[(item["cls"], truth)] += 1
            reg_ann[(label_name(item["reg"]).lower(), truth)] += 1
            if label_name(item["reg"]).lower() != truth:
                reg_mis.append((sid, label_name(item["reg"]), truth))
    encoding_map = defaultdict(Counter)
    for (raw_cls, truth), count in cls_ann.items():
        encoding_map[raw_cls][truth] += count
    inferred_map = {raw: vals.most_common(1)[0][0] for raw, vals in encoding_map.items() if vals}
    encoding_is_unambiguous = all(len(vals) == 1 for vals in encoding_map.values()) and len(inferred_map) >= 2
    for _, row in xdf.iterrows():
        sid = str(row["_id"])
        truth = str(row["annotation"]).strip().lower() if "annotation" in xdf.columns and pd.notna(row.get("annotation")) else ""
        if sid in pkl_by_id and truth in canonical_annotations and inferred_map.get(pkl_by_id[sid]["cls"]) != truth:
            cls_mis.append((sid, pkl_by_id[sid]["cls"], truth))
    report["labels"]["classification_encoding_crosschecked_to_xlsx_annotation"] = {
        "contingency_raw_classification_value_by_annotation": {str(raw): dict(counts) for raw, counts in encoding_map.items()},
        "inferred_encoding": inferred_map,
        "mapping_unambiguous": encoding_is_unambiguous,
        "classification_annotation_mismatch_count": len(cls_mis),
        "classification_annotation_mismatch_examples": [{"id": a, "pkl_classification": b, "xlsx_annotation": c} for a, b, c in cls_mis[:20]],
        "regression_sign_vs_xlsx_annotation_mismatch_count": len(reg_mis),
        "regression_sign_vs_xlsx_annotation_mismatch_examples": [{"id": a, "regression_sign_class": b, "xlsx_annotation": c} for a, b, c in reg_mis[:20]],
    }
    label_rows = []
    for split_name, split in data.items():
        if not isinstance(split, dict):
            continue
        cls_vals = [str(v) for v in split.get("classification_labels", [])]
        ann_by_cls = {raw: ann for raw, ann in inferred_map.items()}
        mapped = [ann_by_cls.get(v, "unmapped") for v in cls_vals]
        for cls_name in class_order:
            count = mapped.count(cls_name.lower())
            label_rows.append({"source": "pkl_classification_mapped_by_xlsx_annotation", "split": split_name, "class": cls_name, "count": count, "proportion": count / len(cls_vals) if cls_vals else None})
        if "annotation" in xdf.columns and "mode" in xdf.columns:
            xsplit = xdf.loc[xdf["mode"].astype(str).str.lower() == split_name.lower()]
            for cls_name in class_order:
                count = int((xsplit["annotation"].astype(str).str.strip().str.capitalize() == cls_name).sum())
                label_rows.append({"source": "xlsx_annotation", "split": split_name, "class": cls_name, "count": count, "proportion": count / len(xsplit) if len(xsplit) else None})
    all_reg = np.concatenate([np.asarray(v["regression_labels"], dtype=float) for v in data.values() if isinstance(v, dict) and "regression_labels" in v])
    report["labels"]["overall_regression"] = {
        "count": int(all_reg.size), "min": float(np.min(all_reg)), "max": float(np.max(all_reg)),
        "mean": float(np.mean(all_reg)), "std_population": float(np.std(all_reg)),
        "quantiles": {str(q): float(np.quantile(all_reg, q)) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)},
        "sign_rule_class_counts": dict(Counter(label_name(x) for x in all_reg)),
    }
    all_cls = [str(v) for split in data.values() if isinstance(split, dict) for v in split.get("classification_labels", [])]
    all_mapped = [inferred_map.get(v, "unmapped") for v in all_cls]
    for cls_name in class_order:
        count = all_mapped.count(cls_name.lower())
        label_rows.append({"source": "pkl_classification_mapped_by_xlsx_annotation", "split": "all", "class": cls_name, "count": count, "proportion": count / len(all_cls) if all_cls else None})
        if "annotation" in xdf.columns:
            count_x = int((xdf["annotation"].astype(str).str.strip().str.capitalize() == cls_name).sum())
            label_rows.append({"source": "xlsx_annotation", "split": "all", "class": cls_name, "count": count_x, "proportion": count_x / len(xdf) if len(xdf) else None})
    for sid, a, b in cls_mis:
        anomalies.append({"type": "pkl_classification_xlsx_annotation_mismatch", "sample_id": sid, "detail": f"pkl={a}; xlsx={b}"})
    for sid, a, b in reg_mis:
        anomalies.append({"type": "regression_sign_xlsx_annotation_mismatch", "sample_id": sid, "detail": f"regression sign={a}; xlsx={b}"})
    for sid, a, b in label_mis:
        anomalies.append({"type": "xlsx_regression_label_mismatch", "sample_id": sid, "detail": f"pkl={a}; xlsx={b}"})
    for sid, a, b in ann_mis:
        anomalies.append({"type": "xlsx_annotation_mismatch", "sample_id": sid, "detail": f"pkl={a}; xlsx={b}"})

    # Compact CSVs.
    split_rows = []
    for sname, ss in report["splits"].items():
        row = {"split": sname, "sample_count": ss["sample_count"], "unique_id_count": ss["id"]["unique_count"], "duplicate_id_count": ss["id"]["duplicate_count"], "empty_raw_text_count": ss["id"]["empty_raw_text_count"]}
        for field, fs in ss["fields"].items():
            prefix = field + "__"
            for k in ("shape", "dtype", "min", "max", "mean", "std_population", "nan_count", "inf_count", "zero_sample_count", "count_abs_gt_1e6", "count_abs_ge_499_9"):
                if k in fs:
                    row[prefix + k] = json.dumps(fs[k]) if isinstance(fs[k], list) else fs[k]
            if "temporal" in fs:
                for k, v in fs["temporal"].items():
                    row[prefix + k] = json.dumps(v) if isinstance(v, list) else v
        split_rows.append(row)
    pd.DataFrame(split_rows).to_csv(args.out / "split_summary.csv", index=False)
    pd.DataFrame(label_rows).to_csv(args.out / "label_distribution.csv", index=False)
    if anomalies:
        pd.DataFrame(anomalies).to_csv(args.out / "anomalies.csv", index=False)
    elif (args.out / "anomalies.csv").exists():
        (args.out / "anomalies.csv").unlink()

    (args.out / "aligned50_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=jsonable) + "\n", encoding="utf-8")

    md = ["# aligned_50.pkl 与 label.xlsx 数据审计", "", f"- pkl：`{args.pkl}`", f"- Excel：`{args.xlsx}`", "- 方法：读取真实 pickle 与 Excel；不修改输入、不做标准化。连续特征的 min/max/mean/std、NaN/Inf、全零计数及极端值计数为全量统计；std 使用总体标准差（ddof=0）。绝对值分位数最多抽取 2,000,000 个值，样本较大时使用固定随机种子的均匀有放回抽样；抽样细节在 JSON 中记录。", "", "## 结构与字段", "", f"- 顶层 keys：`{report['top_level_keys']}`", f"- split 存在：`{report['required_splits_present']}`", ""]
    for sname, ss in report["splits"].items():
        md += [f"### {sname}: {ss['sample_count']} 条", "", f"ID：{ss['id']}", "", "| 字段 | shape | dtype | min | max | mean | std | NaN | Inf | 全零样本 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for field, fs in ss["fields"].items():
            md.append("| {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} |".format(field, fs["shape"], fs["dtype"], fs.get("min", "—"), fs.get("max", "—"), fs.get("mean", "—"), fs.get("std_population", "—"), fs.get("nan_count", "—"), fs.get("inf_count", "—"), fs.get("zero_sample_count", "—")))
        md.append("")
    md += ["## 标签", "", "```json", json.dumps(report["labels"], ensure_ascii=False, indent=2), "```", "", "## ID、split 与 Excel 交叉核对", "", "```json", json.dumps({"split_overlap": report["split_overlap"], "xlsx_crosscheck": report["xlsx_crosscheck"], "duplicates": {s: v.get("exact_duplicate_multimodal_feature_pairs", 0) for s, v in report["splits"].items()}}, ensure_ascii=False, indent=2), "```", "", "## 零 timestep / padding 观察", ""]
    for sname, ss in report["splits"].items():
        for mod in modality_names:
            t = ss["fields"].get(mod, {}).get("temporal")
            if t:
                md.append(f"- `{sname}.{mod}`：{json.dumps(t, ensure_ascii=False)}")
    md += ["", f"跨模态零位置比较：`{json.dumps(report['padding'].get('cross_modality_zero_mask_comparison', {}), ensure_ascii=False)}`", "", "### 是否可用 `all(feature == 0)` 判断 padding / missing？", "", "不能仅凭全零判断。实际 `text_bert` 的第 1 号分量在本数据中呈二值前缀有效、后缀为零模式；audio/vision 在该后缀均为零，而 text 特征在后缀仍非零，说明数值零无法统一代表 padding。audio/vision 也有位于该候选有效区间内的全零时间步，vision 还存在整段全零样本。建议 padding mask 使用经过核验的 `text_bert[:, 1, :]`，后续需结合特征文件说明确认该分量语义；availability mask 与 padding mask 分开维护，并单独记录人工 block mask 来源与位置。未确认前，不能将有效区间的全零样本自动判为缺失。", "", "## 异常", "", f"共记录 {len(anomalies)} 条异常/待核验项。详细记录见 `anomalies.csv`（若该文件存在）。"]
    (args.out / "aligned50_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
