# aligned_50.pkl 与 label.xlsx 数据审计

- pkl：`/root/workspace/E2026/data/raw/attachment2/aligned_50.pkl`
- Excel：`/root/workspace/E2026/data/raw/attachment2/label.xlsx`
- 方法：读取真实 pickle 与 Excel；不修改输入、不做标准化。连续特征的 min/max/mean/std、NaN/Inf、全零计数及极端值计数为全量统计；std 使用总体标准差（ddof=0）。绝对值分位数最多抽取 2,000,000 个值，样本较大时使用固定随机种子的均匀有放回抽样；抽样细节在 JSON 中记录。

## 结构与字段

- 顶层 keys：`['train', 'valid', 'test']`
- split 存在：`{'train': True, 'valid': True, 'test': True}`

### train: 3395 条

ID：{'present': True, 'count': 3395, 'unique_count': 3395, 'duplicate_count': 0, 'duplicate_examples': [], 'format_counts': {'matches video$_$clip': 3395}, 'examples': ['-3g5yACwYnA$_$4', '-HwX2H8Z4hY$_$3', '-HwX2H8Z4hY$_$9'], 'empty_raw_text_count': 0}

| 字段 | shape | dtype | min | max | mean | std | NaN | Inf | 全零样本 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| raw_text | `[3395]` | `<U1662` | — | — | — | — | — | — | — |
| audio | `[3395, 50, 74]` | `float64` | -47.73226547241211 | 500.0 | 1.0542140927720114 | 14.948231889686882 | 0 | 0 | 0 |
| vision | `[3395, 50, 35]` | `float64` | -37.746498107910156 | 29.028600692749023 | -0.383946401756436 | 1.7497746037506856 | 0 | 0 | 110 |
| id | `[3395]` | `<U16` | — | — | — | — | — | — | — |
| text_bert | `[3395, 3, 50]` | `int64` | 0.0 | 29824.0 | 536.660559646539 | 2126.5886562234555 | 0 | 0 | 0 |
| classification_labels | `[3395]` | `float64` | 0.0 | 2.0 | 1.2070692194403534 | 0.8566521022966087 | 0 | 0 | None |
| regression_labels | `[3395]` | `float64` | -3.0 | 3.0 | 0.16583210777639465 | 1.1149631461954514 | 0 | 0 | None |
| text | `[3395, 50, 768]` | `float32` | -9.978325843811035 | 4.965427398681641 | -0.0101705901324749 | 0.45943358540534973 | 0 | 0 | 0 |

### valid: 728 条

ID：{'present': True, 'count': 728, 'unique_count': 728, 'duplicate_count': 0, 'duplicate_examples': [], 'format_counts': {'matches video$_$clip': 728}, 'examples': ['-UacrmKiTn4$_$7', '-qDkUB0GgYY$_$5', '0utROiKxejc$_$1'], 'empty_raw_text_count': 0}

| 字段 | shape | dtype | min | max | mean | std | NaN | Inf | 全零样本 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| raw_text | `[728]` | `<U1570` | — | — | — | — | — | — | — |
| audio | `[728, 50, 74]` | `float64` | -44.918243408203125 | 499.0 | 1.0798619597803265 | 14.933064617639944 | 0 | 0 | 0 |
| vision | `[728, 50, 35]` | `float64` | -30.58169937133789 | 26.221500396728516 | -0.38082381484071454 | 1.7549254227062359 | 0 | 0 | 15 |
| id | `[728]` | `<U16` | — | — | — | — | — | — | — |
| text_bert | `[728, 3, 50]` | `int64` | 0.0 | 29589.0 | 558.924010989011 | 2155.7387683647567 | 0 | 0 | 0 |
| classification_labels | `[728]` | `float64` | 0.0 | 2.0 | 1.1813186813186813 | 0.8452078342381841 | 0 | 0 | None |
| regression_labels | `[728]` | `float64` | -3.0 | 3.0 | 0.16094322484191304 | 1.0428864926774235 | 0 | 0 | None |
| text | `[728, 50, 768]` | `float32` | -9.845565795898438 | 4.626781940460205 | -0.010211119428277016 | 0.46113914251327515 | 0 | 0 | 0 |

### test: 727 条

ID：{'present': True, 'count': 727, 'unique_count': 727, 'duplicate_count': 0, 'duplicate_examples': [], 'format_counts': {'matches video$_$clip': 727}, 'examples': ['-AUZQgSxyPQ$_$2', '-HeZS2-Prhc$_$2', '-MeTTeMJBNc$_$0'], 'empty_raw_text_count': 0}

| 字段 | shape | dtype | min | max | mean | std | NaN | Inf | 全零样本 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| raw_text | `[727]` | `<U1479` | — | — | — | — | — | — | — |
| audio | `[727, 50, 74]` | `float64` | -50.873077392578125 | 500.0 | 1.0803867092167778 | 15.170482412993083 | 0 | 0 | 0 |
| vision | `[727, 50, 35]` | `float64` | -29.1875 | 27.900800704956055 | -0.3753004025233552 | 1.7269197714373534 | 0 | 0 | 28 |
| id | `[727]` | `<U16` | — | — | — | — | — | — | — |
| text_bert | `[727, 3, 50]` | `int64` | 0.0 | 29476.0 | 544.1078771205869 | 2105.8678589344804 | 0 | 0 | 0 |
| classification_labels | `[727]` | `float64` | 0.0 | 2.0 | 1.2132049518569463 | 0.8586105922893306 | 0 | 0 | None |
| regression_labels | `[727]` | `float64` | -3.0 | 3.0 | 0.14121962741447774 | 1.1573125385208927 | 0 | 0 | None |
| text | `[727, 50, 768]` | `float32` | -9.85290241241455 | 5.553917407989502 | -0.010209925472736359 | 0.46076700091362 | 0 | 0 | 0 |

## 标签

```json
{
  "train": {
    "classification_labels_present": true,
    "classification_raw_value_counts": {
      "2.0": 1670,
      "0.0": 967,
      "1.0": 758
    },
    "annotations_present": false,
    "annotation_raw_value_counts": {},
    "regression": {
      "count": 3395,
      "nan_count": 0,
      "inf_count": 0,
      "min": -3.0,
      "max": 3.0,
      "mean": 0.16583210777639465,
      "std_population": 1.1149631461954514,
      "quantiles": {
        "0": -3.0,
        "0.01": -2.6666667461395264,
        "0.05": -2.0,
        "0.25": -0.3333333432674408,
        "0.5": 0.0,
        "0.75": 0.6666666865348816,
        "0.95": 2.0,
        "0.99": 2.6666667461395264,
        "1": 3.0
      },
      "sign_rule_class_counts": {
        "Positive": 1670,
        "Negative": 967,
        "Neutral": 758
      }
    },
    "classification_vs_regression_mismatch_count": 0,
    "classification_vs_regression_mismatch_examples": []
  },
  "valid": {
    "classification_labels_present": true,
    "classification_raw_value_counts": {
      "2.0": 338,
      "1.0": 184,
      "0.0": 206
    },
    "annotations_present": false,
    "annotation_raw_value_counts": {},
    "regression": {
      "count": 728,
      "nan_count": 0,
      "inf_count": 0,
      "min": -3.0,
      "max": 3.0,
      "mean": 0.16094322484191304,
      "std_population": 1.0428864926774235,
      "quantiles": {
        "0": -3.0,
        "0.01": -2.6666667461395264,
        "0.05": -1.6666666269302368,
        "0.25": -0.3333333432674408,
        "0.5": 0.0,
        "0.75": 0.6666666865348816,
        "0.95": 2.0,
        "0.99": 2.3333332538604736,
        "1": 3.0
      },
      "sign_rule_class_counts": {
        "Positive": 338,
        "Neutral": 184,
        "Negative": 206
      }
    },
    "classification_vs_regression_mismatch_count": 0,
    "classification_vs_regression_mismatch_examples": []
  },
  "test": {
    "classification_labels_present": true,
    "classification_raw_value_counts": {
      "2.0": 362,
      "0.0": 207,
      "1.0": 158
    },
    "annotations_present": false,
    "annotation_raw_value_counts": {},
    "regression": {
      "count": 727,
      "nan_count": 0,
      "inf_count": 0,
      "min": -3.0,
      "max": 3.0,
      "mean": 0.14121962741447774,
      "std_population": 1.1573125385208927,
      "quantiles": {
        "0": -3.0,
        "0.01": -2.6666667461395264,
        "0.05": -2.0,
        "0.25": -0.3333333432674408,
        "0.5": 0.0,
        "0.75": 1.0,
        "0.95": 2.0,
        "0.99": 2.3333332538604736,
        "1": 3.0
      },
      "sign_rule_class_counts": {
        "Positive": 362,
        "Negative": 207,
        "Neutral": 158
      }
    },
    "classification_vs_regression_mismatch_count": 0,
    "classification_vs_regression_mismatch_examples": []
  },
  "classification_encoding_crosschecked_to_xlsx_annotation": {
    "contingency_raw_classification_value_by_annotation": {
      "2.0": {
        "positive": 2370
      },
      "0.0": {
        "negative": 1380
      },
      "1.0": {
        "neutral": 1100
      }
    },
    "inferred_encoding": {
      "2.0": "positive",
      "0.0": "negative",
      "1.0": "neutral"
    },
    "mapping_unambiguous": true,
    "classification_annotation_mismatch_count": 0,
    "classification_annotation_mismatch_examples": [],
    "regression_sign_vs_xlsx_annotation_mismatch_count": 0,
    "regression_sign_vs_xlsx_annotation_mismatch_examples": []
  },
  "overall_regression": {
    "count": 4850,
    "min": -3.0,
    "max": 3.0,
    "mean": 0.1614089366424944,
    "std_population": 1.1109889858383002,
    "quantiles": {
      "0": -3.0,
      "0.01": -2.6666667461395264,
      "0.05": -2.0,
      "0.25": -0.3333333432674408,
      "0.5": 0.0,
      "0.75": 1.0,
      "0.95": 2.0,
      "0.99": 2.6666667461395264,
      "1": 3.0
    },
    "sign_rule_class_counts": {
      "Positive": 2370,
      "Negative": 1380,
      "Neutral": 1100
    }
  }
}
```

## ID、split 与 Excel 交叉核对

```json
{
  "split_overlap": {
    "train__valid": {
      "overlap_count": 0,
      "examples": []
    },
    "train__test": {
      "overlap_count": 0,
      "examples": []
    },
    "valid__test": {
      "overlap_count": 0,
      "examples": []
    }
  },
  "xlsx_crosscheck": {
    "sheet_names": [
      "label"
    ],
    "row_count": 4850,
    "columns": [
      "video_id",
      "clip_id",
      "text",
      "label",
      "annotation",
      "mode",
      "_sheet"
    ],
    "constructed_id_rule": "video_id + '$_$' + clip_id",
    "id_duplicate_count": 0,
    "pkl_ids_missing_from_xlsx_count": 0,
    "pkl_ids_missing_from_xlsx_examples": [],
    "xlsx_ids_missing_from_pkl_count": 0,
    "xlsx_ids_missing_from_pkl_examples": [],
    "one_to_one_id_set_match": true,
    "mode_counts": {
      "train": 3395,
      "valid": 728,
      "test": 727
    },
    "per_split": {
      "train": {
        "pkl_count": 3395,
        "xlsx_count": 3395,
        "id_set_match": true,
        "pkl_only_count": 0,
        "xlsx_only_count": 0
      },
      "valid": {
        "pkl_count": 728,
        "xlsx_count": 728,
        "id_set_match": true,
        "pkl_only_count": 0,
        "xlsx_only_count": 0
      },
      "test": {
        "pkl_count": 727,
        "xlsx_count": 727,
        "id_set_match": true,
        "pkl_only_count": 0,
        "xlsx_only_count": 0
      }
    },
    "label_comparison": {
      "label_column": true,
      "annotation_column": true,
      "label_numeric_mismatch_count": 0,
      "annotation_pkl_field_present": false,
      "annotation_pkl_mismatch_count": null,
      "label_numeric_mismatch_examples": [],
      "annotation_pkl_mismatch_examples": []
    }
  },
  "duplicates": {
    "train": 0,
    "valid": 0,
    "test": 0
  }
}
```

## 零 timestep / padding 观察

- `train.text`：{"timesteps_per_sample": 50, "all_zero_sample_count": 0, "all_zero_timestep_count": 0, "all_zero_timestep_ratio": 0.0, "leading_zero_timestep_count": 0, "trailing_zero_timestep_count": 0, "interior_zero_timestep_count": 0, "samples_with_interior_zero_timestep": 0, "constant_feature_dimensions_global": [], "constant_feature_dimension_count": 0, "against_observed_text_bert_mask": {"all_zero_timestep_within_valid_positions": 0, "valid_position_count": 83672, "all_zero_rate_within_valid_positions": 0.0, "all_zero_timestep_within_masked_suffix": 0, "masked_suffix_position_count": 86078, "nonzero_modality_timestep_inside_masked_suffix": 86078, "valid_region_zero_counts_by_early_middle_late": {"early": 0, "middle": 0, "late": 0}, "samples_all_zero_across_valid_region": 0}}
- `train.audio`：{"timesteps_per_sample": 50, "all_zero_sample_count": 0, "all_zero_timestep_count": 92933, "all_zero_timestep_ratio": 0.5474698085419735, "leading_zero_timestep_count": 3460, "trailing_zero_timestep_count": 89473, "interior_zero_timestep_count": 0, "samples_with_interior_zero_timestep": 0, "constant_feature_dimensions_global": [], "constant_feature_dimension_count": 0, "against_observed_text_bert_mask": {"all_zero_timestep_within_valid_positions": 6855, "valid_position_count": 83672, "all_zero_rate_within_valid_positions": 0.08192704847499761, "all_zero_timestep_within_masked_suffix": 86078, "masked_suffix_position_count": 86078, "nonzero_modality_timestep_inside_masked_suffix": 0, "valid_region_zero_counts_by_early_middle_late": {"early": 3438, "middle": 22, "late": 3395}, "samples_all_zero_across_valid_region": 0}}
- `train.vision`：{"timesteps_per_sample": 50, "all_zero_sample_count": 110, "all_zero_timestep_count": 97117, "all_zero_timestep_ratio": 0.5721178203240059, "leading_zero_timestep_count": 4263, "trailing_zero_timestep_count": 86611, "interior_zero_timestep_count": 743, "samples_with_interior_zero_timestep": 146, "constant_feature_dimensions_global": [], "constant_feature_dimension_count": 0, "against_observed_text_bert_mask": {"all_zero_timestep_within_valid_positions": 11039, "valid_position_count": 83672, "all_zero_rate_within_valid_positions": 0.131931829046754, "all_zero_timestep_within_masked_suffix": 86078, "masked_suffix_position_count": 86078, "nonzero_modality_timestep_inside_masked_suffix": 0, "valid_region_zero_counts_by_early_middle_late": {"early": 4734, "middle": 1561, "late": 4744}, "samples_all_zero_across_valid_region": 110}}
- `valid.text`：{"timesteps_per_sample": 50, "all_zero_sample_count": 0, "all_zero_timestep_count": 0, "all_zero_timestep_ratio": 0.0, "leading_zero_timestep_count": 0, "trailing_zero_timestep_count": 0, "interior_zero_timestep_count": 0, "samples_with_interior_zero_timestep": 0, "constant_feature_dimensions_global": [], "constant_feature_dimension_count": 0, "against_observed_text_bert_mask": {"all_zero_timestep_within_valid_positions": 0, "valid_position_count": 18628, "all_zero_rate_within_valid_positions": 0.0, "all_zero_timestep_within_masked_suffix": 0, "masked_suffix_position_count": 17772, "nonzero_modality_timestep_inside_masked_suffix": 17772, "valid_region_zero_counts_by_early_middle_late": {"early": 0, "middle": 0, "late": 0}, "samples_all_zero_across_valid_region": 0}}
- `valid.audio`：{"timesteps_per_sample": 50, "all_zero_sample_count": 0, "all_zero_timestep_count": 19274, "all_zero_timestep_ratio": 0.5295054945054946, "leading_zero_timestep_count": 774, "trailing_zero_timestep_count": 18500, "interior_zero_timestep_count": 0, "samples_with_interior_zero_timestep": 0, "constant_feature_dimensions_global": [], "constant_feature_dimension_count": 0, "against_observed_text_bert_mask": {"all_zero_timestep_within_valid_positions": 1502, "valid_position_count": 18628, "all_zero_rate_within_valid_positions": 0.08063130770882543, "all_zero_timestep_within_masked_suffix": 17772, "masked_suffix_position_count": 17772, "nonzero_modality_timestep_inside_masked_suffix": 0, "valid_region_zero_counts_by_early_middle_late": {"early": 744, "middle": 17, "late": 741}, "samples_all_zero_across_valid_region": 0}}
- `valid.vision`：{"timesteps_per_sample": 50, "all_zero_sample_count": 15, "all_zero_timestep_count": 20208, "all_zero_timestep_ratio": 0.5551648351648352, "leading_zero_timestep_count": 945, "trailing_zero_timestep_count": 18325, "interior_zero_timestep_count": 188, "samples_with_interior_zero_timestep": 43, "constant_feature_dimensions_global": [], "constant_feature_dimension_count": 0, "against_observed_text_bert_mask": {"all_zero_timestep_within_valid_positions": 2436, "valid_position_count": 18628, "all_zero_rate_within_valid_positions": 0.1307708825424093, "all_zero_timestep_within_masked_suffix": 17772, "masked_suffix_position_count": 17772, "nonzero_modality_timestep_inside_masked_suffix": 0, "valid_region_zero_counts_by_early_middle_late": {"early": 1025, "middle": 348, "late": 1063}, "samples_all_zero_across_valid_region": 15}}
- `test.text`：{"timesteps_per_sample": 50, "all_zero_sample_count": 0, "all_zero_timestep_count": 0, "all_zero_timestep_ratio": 0.0, "leading_zero_timestep_count": 0, "trailing_zero_timestep_count": 0, "interior_zero_timestep_count": 0, "samples_with_interior_zero_timestep": 0, "constant_feature_dimensions_global": [], "constant_feature_dimension_count": 0, "against_observed_text_bert_mask": {"all_zero_timestep_within_valid_positions": 0, "valid_position_count": 18309, "all_zero_rate_within_valid_positions": 0.0, "all_zero_timestep_within_masked_suffix": 0, "masked_suffix_position_count": 18041, "nonzero_modality_timestep_inside_masked_suffix": 18041, "valid_region_zero_counts_by_early_middle_late": {"early": 0, "middle": 0, "late": 0}, "samples_all_zero_across_valid_region": 0}}
- `test.audio`：{"timesteps_per_sample": 50, "all_zero_sample_count": 0, "all_zero_timestep_count": 19495, "all_zero_timestep_ratio": 0.5363136176066025, "leading_zero_timestep_count": 727, "trailing_zero_timestep_count": 18768, "interior_zero_timestep_count": 0, "samples_with_interior_zero_timestep": 0, "constant_feature_dimensions_global": [], "constant_feature_dimension_count": 0, "against_observed_text_bert_mask": {"all_zero_timestep_within_valid_positions": 1454, "valid_position_count": 18309, "all_zero_rate_within_valid_positions": 0.07941449560325523, "all_zero_timestep_within_masked_suffix": 18041, "masked_suffix_position_count": 18041, "nonzero_modality_timestep_inside_masked_suffix": 0, "valid_region_zero_counts_by_early_middle_late": {"early": 727, "middle": 0, "late": 727}, "samples_all_zero_across_valid_region": 0}}
- `test.vision`：{"timesteps_per_sample": 50, "all_zero_sample_count": 28, "all_zero_timestep_count": 20519, "all_zero_timestep_ratio": 0.564484181568088, "leading_zero_timestep_count": 863, "trailing_zero_timestep_count": 18055, "interior_zero_timestep_count": 201, "samples_with_interior_zero_timestep": 41, "constant_feature_dimensions_global": [], "constant_feature_dimension_count": 0, "against_observed_text_bert_mask": {"all_zero_timestep_within_valid_positions": 2478, "valid_position_count": 18309, "all_zero_rate_within_valid_positions": 0.1353432737997706, "all_zero_timestep_within_masked_suffix": 18041, "masked_suffix_position_count": 18041, "nonzero_modality_timestep_inside_masked_suffix": 0, "valid_region_zero_counts_by_early_middle_late": {"early": 1031, "middle": 344, "late": 1103}, "samples_all_zero_across_valid_region": 28}}

跨模态零位置比较：`{"train": {"text__audio": {"exact_mask_agreement_ratio": 0.45253019145802653, "both_zero_positions": 0, "only_first_zero": 0, "only_second_zero": 92933}, "text__vision": {"exact_mask_agreement_ratio": 0.4278821796759941, "both_zero_positions": 0, "only_first_zero": 0, "only_second_zero": 97117}, "audio__vision": {"exact_mask_agreement_ratio": 0.9753519882179676, "both_zero_positions": 92933, "only_first_zero": 0, "only_second_zero": 4184}}, "valid": {"text__audio": {"exact_mask_agreement_ratio": 0.4704945054945055, "both_zero_positions": 0, "only_first_zero": 0, "only_second_zero": 19274}, "text__vision": {"exact_mask_agreement_ratio": 0.44483516483516483, "both_zero_positions": 0, "only_first_zero": 0, "only_second_zero": 20208}, "audio__vision": {"exact_mask_agreement_ratio": 0.9743406593406594, "both_zero_positions": 19274, "only_first_zero": 0, "only_second_zero": 934}}, "test": {"text__audio": {"exact_mask_agreement_ratio": 0.46368638239339754, "both_zero_positions": 0, "only_first_zero": 0, "only_second_zero": 19495}, "text__vision": {"exact_mask_agreement_ratio": 0.43551581843191195, "both_zero_positions": 0, "only_first_zero": 0, "only_second_zero": 20519}, "audio__vision": {"exact_mask_agreement_ratio": 0.9718294360385145, "both_zero_positions": 19495, "only_first_zero": 0, "only_second_zero": 1024}}}`

### 是否可用 `all(feature == 0)` 判断 padding / missing？

不能仅凭全零判断。实际 `text_bert` 的第 1 号分量在本数据中呈二值前缀有效、后缀为零模式；audio/vision 在该后缀均为零，而 text 特征在后缀仍非零，说明数值零无法统一代表 padding。audio/vision 也有位于该候选有效区间内的全零时间步，vision 还存在整段全零样本。建议 padding mask 使用经过核验的 `text_bert[:, 1, :]`，后续需结合特征文件说明确认该分量语义；availability mask 与 padding mask 分开维护，并单独记录人工 block mask 来源与位置。未确认前，不能将有效区间的全零样本自动判为缺失。

## 异常

共记录 152 条异常/待核验项。详细记录见 `anomalies.csv`（若该文件存在）。
