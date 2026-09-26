# LTARP 核心模块消融

## 固定协议

- Attachment2 aligned_50；只构建 train/valid Dataset，test 不参与评价或选择。
- 每个 seed 加载同 seed CleanSelect MMP；冻结 modality projection、fusion、分类头和回归头，仅优化变体分支。
- AdamW，lr=0.001，weight decay=0.0001，batch size=128，最多80 epochs，patience=12。
- balanced train Weighted CE + SmoothL1，lambda_reg=1.0；不做标准化或训练缺失增强。
- 每 epoch 仅用完整输入 valid 的 S_val 选择 checkpoint；随后固定 checkpoint 评估 clean 与冻结54场景。Benchmark SHA256：`3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`。
- 附件2 aligned_50 SHA256：`66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`。
- 结果均值的标准差为 sample SD（ddof=1）；paired delta 正值表示变体改善，MAE 采用 MMP−variant。

## 对照复核

A0 MMP 与 A4 Full LTARP 复用现存 checkpoint；逐项验证 SHA256 与 CleanSelect 记录，并重新跑同一 validation benchmark。

| Model | Seed | Clean reference check |
|---|---:|---|
| A0 | 42 | PASS; max abs metric error 6.8e-09 |
| A4 | 42 | PASS; max abs metric error 5.8e-09 |
| A0 | 43 | PASS; max abs metric error 8.2e-09 |
| A4 | 43 | PASS; max abs metric error 6.9e-09 |
| A0 | 44 | PASS; max abs metric error 1.6e-09 |
| A4 | 44 | PASS; max abs metric error 3.8e-09 |

## 三 seed 主结果

| Variant | Clean Acc | Clean Macro-F1 | Clean MAE | Clean Pearson | Missing Acc | Missing Macro-F1 | Missing MAE | Missing Pearson | Robust score | Total params | Stage-2 trainable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 MMP | 0.6401 $\pm$ 0.0048 | 0.6139 $\pm$ 0.0091 | 0.6079 $\pm$ 0.0103 | 0.6409 $\pm$ 0.0043 | 0.6349 $\pm$ 0.0062 | 0.6092 $\pm$ 0.0075 | 0.6111 $\pm$ 0.0098 | 0.6359 $\pm$ 0.0038 | 0.7417 $\pm$ 0.0015 | 163460 | 0/0/0 |
| A1 Attention-only | 0.6401 $\pm$ 0.0036 | 0.6237 $\pm$ 0.0045 | 0.5942 $\pm$ 0.0058 | 0.6612 $\pm$ 0.0037 | 0.6323 $\pm$ 0.0048 | 0.6150 $\pm$ 0.0027 | 0.6005 $\pm$ 0.0021 | 0.6536 $\pm$ 0.0027 | 0.7462 $\pm$ 0.0002 | 164340 | 880/880/880 |
| A2 Mean-attention concat | 0.6273 $\pm$ 0.0083 | 0.6165 $\pm$ 0.0061 | 0.6101 $\pm$ 0.0148 | 0.6393 $\pm$ 0.0113 | 0.6203 $\pm$ 0.0106 | 0.6094 $\pm$ 0.0084 | 0.6145 $\pm$ 0.0158 | 0.6318 $\pm$ 0.0135 | 0.7381 $\pm$ 0.0044 | 263028 | 99568/99568/99568 |
| A3 Fixed residual | 0.6429 $\pm$ 0.0050 | 0.6230 $\pm$ 0.0082 | 0.6100 $\pm$ 0.0177 | 0.6488 $\pm$ 0.0069 | 0.6351 $\pm$ 0.0083 | 0.6149 $\pm$ 0.0051 | 0.6141 $\pm$ 0.0167 | 0.6426 $\pm$ 0.0062 | 0.7447 $\pm$ 0.0016 | 164340 | 880/880/880 |
| A4 Full LTARP | 0.6415 $\pm$ 0.0050 | 0.6243 $\pm$ 0.0026 | 0.6052 $\pm$ 0.0063 | 0.6475 $\pm$ 0.0024 | 0.6334 $\pm$ 0.0038 | 0.6163 $\pm$ 0.0013 | 0.6086 $\pm$ 0.0059 | 0.6417 $\pm$ 0.0024 | 0.7447 $\pm$ 0.0014 | 164343 | 883/883/883 |
| A5 Text-only attention | 0.6410 $\pm$ 0.0065 | 0.6242 $\pm$ 0.0031 | 0.6034 $\pm$ 0.0098 | 0.6496 $\pm$ 0.0043 | 0.6326 $\pm$ 0.0051 | 0.6158 $\pm$ 0.0025 | 0.6072 $\pm$ 0.0087 | 0.6434 $\pm$ 0.0041 | 0.7448 $\pm$ 0.0014 | 164230 | 770/770/770 |
| A6 Audio-only attention | 0.6433 $\pm$ 0.0080 | 0.6183 $\pm$ 0.0037 | 0.6059 $\pm$ 0.0081 | 0.6409 $\pm$ 0.0045 | 0.6356 $\pm$ 0.0072 | 0.6108 $\pm$ 0.0054 | 0.6091 $\pm$ 0.0078 | 0.6359 $\pm$ 0.0040 | 0.7430 $\pm$ 0.0006 | 163536 | 76/76/76 |
| A7 Vision-only attention | 0.6429 $\pm$ 0.0095 | 0.6204 $\pm$ 0.0039 | 0.6079 $\pm$ 0.0102 | 0.6408 $\pm$ 0.0044 | 0.6357 $\pm$ 0.0077 | 0.6129 $\pm$ 0.0037 | 0.6109 $\pm$ 0.0095 | 0.6358 $\pm$ 0.0040 | 0.7434 $\pm$ 0.0019 | 163497 | 37/37/37 |

## 相对 MMP 的 paired delta

| Variant | Condition | Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD | Positive seeds |
|---|---|---|---:|---:|---:|---:|---:|
| A1 | clean | accuracy | -0.00137 | +0.00000 | +0.00137 | +0.00000 ± 0.00137 | 1/3 |
| A1 | clean | macro_f1 | +0.01658 | +0.00570 | +0.00696 | +0.00975 ± 0.00595 | 3/3 |
| A1 | clean | mae | +0.01723 | +0.00864 | +0.01524 | +0.01371 ± 0.00449 | 3/3 |
| A1 | clean | pearson | +0.02268 | +0.02481 | +0.01323 | +0.02024 ± 0.00616 | 3/3 |
| A1 | mean_missing | accuracy | -0.00468 | +0.00038 | -0.00341 | -0.00257 ± 0.00263 | 1/3 |
| A1 | mean_missing | macro_f1 | +0.01094 | +0.00520 | +0.00124 | +0.00579 ± 0.00488 | 3/3 |
| A1 | mean_missing | mae | +0.01659 | +0.00199 | +0.01348 | +0.01068 ± 0.00769 | 3/3 |
| A1 | mean_missing | pearson | +0.02100 | +0.02058 | +0.01156 | +0.01771 ± 0.00533 | 3/3 |
| A2 | clean | accuracy | -0.02610 | -0.00137 | -0.01099 | -0.01282 ± 0.01246 | 0/3 |
| A2 | clean | macro_f1 | +0.01012 | +0.00743 | -0.00977 | +0.00259 ± 0.01079 | 2/3 |
| A2 | clean | mae | +0.02207 | -0.01758 | -0.01086 | -0.00212 ± 0.02122 | 1/3 |
| A2 | clean | pearson | +0.01432 | -0.00236 | -0.01681 | -0.00161 ± 0.01558 | 1/3 |
| A2 | mean_missing | accuracy | -0.02938 | +0.00120 | -0.01562 | -0.01460 ± 0.01531 | 1/3 |
| A2 | mean_missing | macro_f1 | +0.00636 | +0.00888 | -0.01487 | +0.00012 ± 0.01304 | 2/3 |
| A2 | mean_missing | mae | +0.02106 | -0.01794 | -0.01317 | -0.00335 ± 0.02127 | 1/3 |
| A2 | mean_missing | pearson | +0.01250 | -0.00283 | -0.02203 | -0.00412 ± 0.01731 | 1/3 |
| A3 | clean | accuracy | +0.00275 | +0.00412 | +0.00137 | +0.00275 ± 0.00137 | 3/3 |
| A3 | clean | macro_f1 | +0.00970 | +0.01014 | +0.00728 | +0.00904 ± 0.00154 | 3/3 |
| A3 | clean | mae | -0.01447 | -0.00141 | +0.00985 | -0.00201 ± 0.01217 | 1/3 |
| A3 | clean | pearson | +0.00415 | +0.01112 | +0.00840 | +0.00789 ± 0.00351 | 3/3 |
| A3 | mean_missing | accuracy | +0.00191 | +0.00239 | -0.00366 | +0.00021 ± 0.00336 | 2/3 |
| A3 | mean_missing | macro_f1 | +0.00742 | +0.00859 | +0.00097 | +0.00566 ± 0.00410 | 3/3 |
| A3 | mean_missing | mae | -0.01499 | -0.00274 | +0.00871 | -0.00301 ± 0.01185 | 1/3 |
| A3 | mean_missing | pearson | +0.00356 | +0.00903 | +0.00754 | +0.00671 ± 0.00282 | 3/3 |
| A4 | clean | accuracy | +0.00137 | +0.00275 | +0.00000 | +0.00137 ± 0.00137 | 2/3 |
| A4 | clean | macro_f1 | +0.02009 | +0.01102 | +0.00000 | +0.01037 ± 0.01006 | 2/3 |
| A4 | clean | mae | +0.00965 | -0.00218 | +0.00087 | +0.00278 ± 0.00615 | 2/3 |
| A4 | clean | pearson | +0.01033 | +0.00952 | -0.00002 | +0.00661 ± 0.00576 | 2/3 |
| A4 | mean_missing | accuracy | -0.00430 | +0.00036 | -0.00056 | -0.00150 ± 0.00247 | 1/3 |
| A4 | mean_missing | macro_f1 | +0.01347 | +0.00811 | -0.00042 | +0.00705 ± 0.00700 | 2/3 |
| A4 | mean_missing | mae | +0.00946 | -0.00255 | +0.00083 | +0.00258 ± 0.00619 | 2/3 |
| A4 | mean_missing | pearson | +0.00901 | +0.00825 | -0.00002 | +0.00575 ± 0.00501 | 2/3 |
| A5 | clean | accuracy | +0.00275 | +0.00137 | -0.00137 | +0.00092 ± 0.00210 | 2/3 |
| A5 | clean | macro_f1 | +0.01854 | +0.01178 | +0.00064 | +0.01032 ± 0.00904 | 3/3 |
| A5 | clean | mae | +0.00411 | +0.00371 | +0.00584 | +0.00455 ± 0.00113 | 3/3 |
| A5 | clean | pearson | +0.00987 | +0.01392 | +0.00214 | +0.00864 ± 0.00599 | 3/3 |
| A5 | mean_missing | accuracy | -0.00366 | -0.00117 | -0.00201 | -0.00228 ± 0.00127 | 0/3 |
| A5 | mean_missing | macro_f1 | +0.01141 | +0.00853 | -0.00039 | +0.00652 ± 0.00615 | 2/3 |
| A5 | mean_missing | mae | +0.00402 | +0.00251 | +0.00533 | +0.00395 ± 0.00141 | 3/3 |
| A5 | mean_missing | pearson | +0.00853 | +0.01215 | +0.00186 | +0.00751 ± 0.00522 | 3/3 |
| A6 | clean | accuracy | +0.00687 | +0.00275 | +0.00000 | +0.00321 ± 0.00346 | 2/3 |
| A6 | clean | macro_f1 | +0.01055 | +0.00266 | +0.00000 | +0.00441 ± 0.00549 | 2/3 |
| A6 | clean | mae | +0.00513 | -0.00025 | +0.00131 | +0.00206 ± 0.00277 | 2/3 |
| A6 | clean | pearson | -0.00015 | -0.00016 | +0.00025 | -0.00002 ± 0.00023 | 1/3 |
| A6 | mean_missing | accuracy | +0.00186 | +0.00025 | -0.00000 | +0.00070 ± 0.00101 | 2/3 |
| A6 | mean_missing | macro_f1 | +0.00480 | -0.00012 | +0.00011 | +0.00160 ± 0.00277 | 2/3 |
| A6 | mean_missing | mae | +0.00505 | -0.00019 | +0.00117 | +0.00201 ± 0.00272 | 2/3 |
| A6 | mean_missing | pearson | -0.00010 | -0.00014 | +0.00018 | -0.00002 ± 0.00017 | 1/3 |
| A7 | clean | accuracy | +0.00824 | +0.00000 | +0.00000 | +0.00275 ± 0.00476 | 1/3 |
| A7 | clean | macro_f1 | +0.01942 | +0.00000 | +0.00000 | +0.00647 ± 0.01121 | 1/3 |
| A7 | clean | mae | +0.00005 | -0.00004 | +0.00023 | +0.00008 ± 0.00014 | 2/3 |
| A7 | clean | pearson | -0.00027 | +0.00002 | -0.00008 | -0.00011 ± 0.00015 | 1/3 |
| A7 | mean_missing | accuracy | +0.00247 | -0.00005 | +0.00008 | +0.00083 ± 0.00142 | 2/3 |
| A7 | mean_missing | macro_f1 | +0.01082 | +0.00005 | +0.00010 | +0.00365 ± 0.00621 | 3/3 |
| A7 | mean_missing | mae | +0.00064 | -0.00002 | +0.00021 | +0.00027 ± 0.00034 | 2/3 |
| A7 | mean_missing | pearson | -0.00031 | +0.00002 | -0.00008 | -0.00012 ± 0.00017 | 1/3 |

### Robust score paired delta

| Variant | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD | Positive seeds |
|---|---:|---:|---:|---:|---:|
| A1 | +0.00612 | +0.00447 | +0.00292 | +0.00450 ± 0.00160 | 3/3 |
| A2 | -0.00230 | +0.00095 | -0.00933 | -0.00356 ± 0.00526 | 1/3 |
| A3 | +0.00259 | +0.00433 | +0.00213 | +0.00302 ± 0.00116 | 3/3 |
| A4 | +0.00544 | +0.00379 | -0.00009 | +0.00305 ± 0.00284 | 2/3 |
| A5 | +0.00495 | +0.00432 | +0.00009 | +0.00312 ± 0.00264 | 3/3 |
| A6 | +0.00321 | +0.00067 | +0.00009 | +0.00132 ± 0.00166 | 3/3 |
| A7 | +0.00510 | +0.00000 | +0.00002 | +0.00171 ± 0.00294 | 3/3 |

## 结果解读

- Attention-only 相对 MMP 的 clean Macro-F1 paired delta 为 +0.00975；mean-missing Macro-F1 delta 为 +0.00579；A1 paired robust-score 增益为 +0.00450。因此 attention-only 在 F1、MAE、Pearson 上优于均值池化，但 missing Accuracy 均值略低。
- Mean branch 加回后（A4 相对 A1），clean/missing Macro-F1 分别多 +0.00062 / +0.00126，但 robust score paired delta 为 -0.00146 ± 0.00134；其收益集中在分类指标，未提高整体鲁棒分数。
- Fixed gamma=1 与 learnable gamma 的 clean Macro-F1 paired delta 分别为 +0.00904、+0.01037；learnable gamma 相对 fixed gamma 的 clean/missing Macro-F1 再增加 +0.00133 / +0.00139，MAE improvement 再增加 +0.00479 / +0.00558；但 Accuracy/Pearson 略降，robust-score差仅 +0.00003 ± 0.00258，没有稳定整体优势。
- Concat 与 residual 的 clean Macro-F1 分别为 +0.00259、+0.01037（相对 MMP）；mean-missing Macro-F1 分别为 +0.00012、+0.00705。
- 单模态 attention 中，Text（A5）的 mean-missing Macro-F1 paired delta 最大 (+0.00652)，robust-score paired delta 也最大 (+0.00312）；Audio/Vision 对应 robust 增益为 +0.00132/+0.00171。
- concat（A2）robust score 为 0.73812，比 residual LTARP（A4）低 -0.00661；A2 的 seed 间 SD (0.00437) 也是各变体中最大，residual 更稳定。
- A4 的四项原始指标中，clean 三 seed 平均配对差均为正；mean-missing 有 75% 的指标均值为正，其中 Accuracy 均值略降。配对表同时列出每个 seed 的方向，综合分数不代表四项指标同步提升。
- Seed 稳定性以 paired delta 的方向和离散程度判断。A4 mean-missing Macro-F1 三 seed 方向：+0.01347, +0.00811, -0.00042.
- A2 每模态 concat projection 为 Linear(256,128)（含 bias）；相对 MMP 的新增总参数量为 99568。总参数与第二阶段可训练参数分列报告。

## 输出说明

逐 seed 场景指标、训练 checkpoint/hash、参数量以及新增变体的每 epoch clean validation 历史均保存在同一输出目录的 CSV/JSON、checkpoints 和 training_logs 中；A0/A4 复用权重及核验记录一并列出。
未使用 Attachment2 test、Attachment3 或 Attachment4。
