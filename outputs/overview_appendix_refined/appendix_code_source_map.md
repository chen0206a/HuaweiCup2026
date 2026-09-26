# 附录代码来源

展示片段使用压缩包中留存的真实源码。函数保持原判断和计算，仅删除函数说明字符串、注释；B2省略不相关的 mean_max 分支，保留原 mean_attention 路径。展示文件依赖原工程的数据类型、模型初始化和辅助函数，不作为独立实验入口。

|源文件（appendices/code/）|函数|原始行号|
|---|---|---|
|q1_temporal.py|align_to_bins_hard|63–96|
|q2_baseline.py|masked_mean_pool|10–16|
|q2_pooling_residual.py|masked_attention_pool|20–30|
|q2_pooling_residual.py|forward|60–78|
|q2_block_mask.py|block_length|19–22|
|q2_block_mask.py|block_start|25–35|
|q2_missing_benchmark.py|scenarios|29–37|
|q2_missing_benchmark.py|mask_scenario|40–46|
|q3_coalitions.py|exact_shapley|55–69|
|q3_coalitions.py|pair_interactions|72–84|
|q3_temporal_occlusion.py|window_length|18–21|
|q3_temporal_occlusion.py|evaluate_windows|71–106|

B4联盟常量来自 q3_coalitions.py 第12–16行。B3调用的 apply_blocks（q2_block_mask.py 第44–90行）及B5调用的 evaluate_position_sets（q3_temporal_occlusion.py 第44–68行）均保留在完整代码目录；不改写其清零和有效位语义。Bootstrap完整源码 q3_faithfulness.py 继续保留，但不再印入附录。

展示片段另做纯排版换行，以Python AST逐段比较确认换行前后语义树相同。B3比例/位置常量来自q2_block_mask.py第15–16行；B4的PAIR_NAMES来自q3_coalitions.py第17行。
