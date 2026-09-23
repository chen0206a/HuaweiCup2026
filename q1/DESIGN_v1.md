# Q1第一版实现规格：安全多层分块与预算约束自适应局部搜索

日期：2026-09-23。状态：候选实现规格，尚未实现正式求解器或获得性能结果。继承 `notes/a_plan_review_20260923.md` 的官方语义核查。本文件细化用户新提供的合并方案，不修改全局验证协议。

## 1. 范围与版本

- V0对照：等工作量连续分块、关键路径列表调度、固定概率三类邻域。
- V1主方法：三类拓扑初解、安全多层粗化/逐层细化、Tensor级通信计数、计入搬运的轻量代理、按真实收益与耗时更新算子概率。
- V2可选：少量非连续节点迁移；之后才考虑真正的区域破坏—修复式ALNS。
- GNN/强化学习不纳入V1依赖。

Shift、Merge/Split、Move加权选择属于自适应局部搜索；只有显式移除一批分配/边界、随后联合修复并验证，才称ALNS。实验未证明之前，不将多层或自适应模块写成必然更优。

## 2. 数据结构与接口

保持原始ID；如使用内部紧凑编号，必须保存双向映射。操作的pipe以原始字段为准。

|结构|必要字段|
|---|---|
|OpRecord|id、op_type、pipe、cycles、input_tids、output_tids、preds、succs|
|TensorRecord|id、size、pos、eligible_producers、eligible_consumers、original_copy_out标记|
|Block|sgid、成员集合、稳定内部顺序、各Pipe工作量、内部CP、边界读写、L1/UB风险、版本号|
|Plan|op_to_sgid、block_dag、core_of_block、core_orders、全局block_order、粗化树|
|Candidate|parent_hash、operator、proxy_features、构造/筛选耗时、合法性状态|
|Evaluation|plan_hash、scene、k、makespan、搬运/spill/时序、耗时、有效/失败/超时状态|

建议接口：

```python
load_problem(graph_path, config_path) -> Problem
generate_topological_orders(problem) -> list[list[int]]
build_initial_blocks(problem, order, granularity) -> Plan
boundary_traffic_a(problem, plan, touched_tensors=None) -> Traffic
coarsen_adjacent(problem, plan, limits) -> Hierarchy
list_schedule_a(problem, partition, proxy) -> Plan
refine_level(problem, plan, hierarchy, budget) -> list[Candidate]
validate_plan_a(problem, plan) -> Validation
proxy_score_a(problem, plan) -> ProxyResult
evaluate_official(problem, plan, budget) -> Evaluation
update_operator_weights(history, config) -> Weights
export_and_recheck(problem, best_valid, output_dir) -> Evaluation
```

每个结构上的缓存由成员、顺序、相关边界与版本号失效。原始graph/config、评估代码哈希参与完整评价缓存key；在未证明ID无关前不做激进重编号去重。

## 3. 场景A的边界字节精确计数

令P_t为张量t的非COPY生产者集合，C_t为非COPY消费者集合；P_tg、C_tg为它们与块g成员的交集。定义：

I_tg = 1[C_tg非空且P_tg为空]。

O_tg = 1[P_tg非空且（存在原始COPY_OUT，或C_t为空，或C_t在g外仍有消费者）]。

不计spill时的重建COPY总字节：

V_A = Σ_t size(t) Σ_g (I_tg + O_tg)。

这对应当前原始评估器的按Task边界插入行为，包括共享输入的重复读取。新增搬运为重建量减原图COPY量，再加官方spill量。分区变化仅需重算受影响tensor的触及块，不能逐消费者边累加。

原图存在多个生产者、特殊位置或额外算子时，不额外假定唯一生产者；必须与官方构造逐项核对。此公式不可直接挪用到Q2的source-core→target-core计数。

## 4. 代理：统一周期尺度，显式计入搬运

用户新公式max(W_M,W_V,CP)+λPmem改进了串行依赖估计，但计算-only的CP仍会漏掉边界COPY。V1对每个块构造简化局部图：保留原计算依赖，按上述边界规则加入COPY节点，以独占60 bytes/cycle时的传输耗时作为其乐观时长。

计算以下特征：

- W_p(g)：各Pipe的独占工作量，分别保留M、V、MTE2、MTE3。
- CP_ideal(g)：含边界COPY的简化局部依赖关键路径。
- V_g：边界COPY总字节。
- m_g：无量纲的L1/UB存活内存风险。

内存风险可用候选局部顺序逐步更新live集合，释放最后使用的张量，同时计入输入与输出共存需求；对缓存r，在n_g个步骤上计算mean_j[(live_r(j)/capacity_r-1)_+]，再对L1、UB求和。该指标不等于spill量，也不是实际时序上的内存积分。

代理Task耗时：

tau0(g) = max{CP_ideal(g), max_p W_p(g), V_g/60}。
tau_hat(g) = tau0(g) · [1 + λ_m m_g]。

lambda_m是待验证的无量纲参数。避免把字节、周期、负载差和内存面积未经转换直接相加；tau_hat不是严格下界或真实耗时。

列表调度将tau_hat用于块时长，按正确的100/1000等待规则得到Cmax_list。完整代理可定义为：

F_proxy = max{Cmax_list, V_A/60}。

V_A/60用于反映全局共享DDR工作量，采取max而非再次相加，避免把已经计入局部Task的搬运再收费一遍。该式仍忽略复杂争用，不能代替官方模拟。以V_A和内存风险作为附加排序信息；首轮同时保留低F、低搬运和不同结构候选，降低代理误筛风险。

## 5. 列表调度与核内顺序

从ready集合按向上关键路径优先取块。跨核代价在核心尚未确定时只作排序启发；不要把它既写入块耗时又写入实际EST。

对候选核心k，在全部依赖已安排后：

EST(g,k)=max{所有前驱的预计完成时刻，同核上一块完成+100，跨核前驱完成+1000}。

第一块忽略同核切换项。所有前驱项可在满足同核全局拓扑顺序时简化，但实现显式保留以便检查。EFT=EST+tau_hat，选择最小者，确定性并列规则可使用新增负载、核心编号。

得到一个合法的全局块拓扑序，各核心列表是它的子序列。该公共序保证场景A的块依赖与同核顺序不会形成环；导出前仍执行官方校验。Q2/Q3不能因此省去更细的执行图检查。

## 6. Multilevel的安全实现

### 6.1 禁止仅凭强通信或直接连边收缩

反例：u→v、u→w、w→v。原图无环，但合并u、v后得到uv→w→uv。直接连边不是安全收缩条件，本次用Kahn检查复核了该反例。

V1只合并同一合法拓扑序中相邻、互不重叠的块，因所有跨块边都沿序前进，收缩后仍无环。每轮合并后的新边界必须重新计数；不能用过期增益批量合并相互重叠的对。

### 6.2 粗化收益

通信收益使用完整重建字节差DeltaV，而非连接两块的原始边权之和。排序指标可用：

G(u,v) = DeltaV/60 - [tau_hat(u∪v)-max(tau_hat(u),tau_hat(v))]_+。

两项均为周期。第二项是保守的聚合损失近似，不证明原块可完全并行；该式仅用于候选排序。它不含真实spill收益，因此必须允许多粒度种子，且不因G负就永久排除更粗层。

对独立块还可补充顺序敏感的代理比较，避免仅按共享输入聚合而损失并行性。设置块工作量与内存风险的软上限；超出缓存代理容量不直接宣判官方不可行，因为评估器可spill。

### 6.3 层次与解粗化

从按工作量形成的微块出发，每轮选择互不重叠的相邻合并，保存合并树。记录多个中间层，不一次粗化到k块并丢弃其他层。候选目标尺度可覆盖约8k、4k、2k、k，具体尺度与起始微块数属于预算参数，需基线后验证。

在粗层做列表分核；解开一层时子块先继承父块核心，再对该层边界做Shift、Split与Move。必须真正保留并使用层次关系，否则只是贪心合并，不能声称实现了多层粗化—细化。

任何送入官方评价器的粗层解都要展开为每个原始非COPY操作的sgid映射。评估器始终读取原始图，不能用超节点的聚合cycles代替原始操作并作为正式成绩。

## 7. 三类邻域与可选非连续修正

1. Shift：在相邻块边界移动少量操作；两个块保持非空，增量更新相关张量与块内关键路径。
2. Merge/Split：合并相邻块；拆分时优先使用粗化树的既有边界，也可比较低边界搬运处的切点。
3. Move：把整块迁到另一个核心，按公共块拓扑序插入；重算核心等待与负载。

每轮限制候选数量，便宜增量指标先截断，只有入围候选才重算全局代理。不能对所有原始节点对做O(n²)合并搜索，也不能每评一个边界就重新对全图做完整分析。

V2若允许单节点迁入不相邻块，须先验证新的块商图无环，再重建公共拓扑序与各核列表。无法修复则回退。

重要性质：任何商图无环的划分，都存在一个使各块连续的原图拓扑序——先对块排序，再连接各块内部拓扑序即可。因此真正的限制是固定少数π和可用邻域，而不是“连续块编码”本身。非连续修正后可重建π，继续使用同一表示。

## 8. 自适应选择如何形成可靠反馈

第一版可更新Shift、Merge/Split、Move三个算子族的概率。不要用代理改善充当真实收益。

对已经真实评价的候选i，记录其生成父方案X_i，而非异步评价结束时的当前最好方案；定义：

gain_i=max{0,(T(X_i)-T(candidate_i))/T(X_i)}。
reward_i=gain_i/[1+cost_i/c_ref]。

cost_i含生成、筛选及实际评价开销；c_ref为固定参考或预先规定的评价耗时尺度。若使用批量候选，开销分摊口径预先固定。该定义将收益与开销兼顾，属于启发式收益评分。

按窗口对已评价候选平均reward更新权重：w_o←(1-rho)w_o+rho·mean_reward_o。随后用：

p_o=(1-epsilon)w_o/Σw + epsilon/m。

权重和为0则回退均匀选择。初始化正权重，保留探索底率。rho、epsilon均须配置并记录，不能事后按最好结果选择。

未通过代理筛选、未经真实评价的候选属于未观测，不记为零收益；实际失败/超时单列记录，不能把超时当成无穷Makespan参与论文均值。每个算子获得的真实试验过少时，维持固定/均匀概率，不声称已经学到可靠偏好。

DDR总字节多、某核结束晚、spill多、Cache命中低，都只是诊断线索：可能处于非关键路径，或由上游等待导致。先用来生成候选，再用真实改进验证，不把这些统计量当成瓶颈的因果证明。

## 9. 评价闭环伪代码

```text
read graph + fixed config; verify hashes and input
evaluate all-on-one-core fallback; retain best_valid
construct three legal topological orders
for each order within construction budget:
    build microblocks and safe adjacent coarsening hierarchy
    schedule selected coarse levels, lift to original operations
    uncoarsen selected levels and create refinements
screen candidates; retain structural diversity and exploration slots
evaluate a small set with the original evaluator; update best_valid
while enough time remains after final-check reserve:
    choose parent from evaluated incumbents
    select neighborhood family (fixed or adaptive)
    generate a bounded candidate batch
    validate; compute incremental then full proxy for shortlisted candidates
    evaluate shortlisted and occasional exploration candidates
    update reward from parent-to-child true improvement and actual cost
    keep verified improvements; preserve best_valid independently
export best_valid and re-run the original CLI
```

全图单核保底可能本身昂贵，应独立给予足够时间。若连保底尚未通过评价，明确返回“未取得有效结果”，不能输出未经验证方案并标成成功。

最终目标按(Makespan,新增搬运)字典序选择；搜索运行时间作为独立效率指标。原始评估器的真实Makespan优先于所有代理分数。

## 10. 验收与消融

- 数据覆盖、边界COPY计数：用共享输入、多消费者扇出、链式依赖、独立分支小例对照官方构造。
- 合法性：安全收缩反例、核心交错依赖、迁移后组合环、空核、0大小张量等边界情形。
- 性能验证：按代表规模测真实评价成本，按墙钟预算比较V0与V1；附评价次数便于解释。
- 消融：单层/多层、固定/自适应选择、无/有通信计数、无/有内存风险。最终统一评估100例和要求核数；具体调参与验证安排仍待全局确认。
- 如果多层初始化或自适应选择在相同预算下没有可靠收益，就不作为最终主方法卖点。

## 11. 文献核对及适用边界

以下来自本次可访问的出版社/作者记录与摘要，未完成全部论文实现复现。外部论文百分比不能作为本题预期收益。

1. [Papp等，SPAA 2024](https://arxiv.org/abs/2404.15246)：确有多层调度，核心采用扩展BSP/NUMA模型，可借鉴层次优化。题目共享DDR、固定核内调度和spill不等同于其模型；高通信成本场景下的倍数改善不是通用保证。
2. [EFT-GVNS，ESWA](https://www.sciencedirect.com/science/article/abs/pii/S0957417423018298)：可借鉴EFT构造与变邻域框架；其同机通信零成本等假设不能照搬场景A。附件中的全部百分比本次未逐项核实，不纳入方案性能依据。
3. [Johnn等，C&OR 2024](https://doi.org/10.1016/j.cor.2024.106791)：确有GRLOS及轻量LRW，但主要实证是五类路径规划问题；两者含预训练，LRW不等同于本方案在线权重更新。研究支持算子选择思路，不证明在本题DAG上GNN更强。
4. [CADE，JSA 2025](https://www.sciencedirect.com/science/article/pii/S138376212500044X)：确有亲和分配与延迟执行；其在线调度接口不同于本题静态方案输出。只能迁移思想，不直接引入题目接口无法表达的延迟/预取。
5. [Tan等，IJIS 2025](https://onlinelibrary.wiley.com/doi/10.1155/int/7562400)：包含GCN/Transformer、PPO、A2C及EFT类分配；涉及异构核心、节点复制和功耗目标。本题同构核心与操作唯一分配条件不同，不采用其复制机制。

建议论文主要引用与实际实现对应的多层DAG调度、列表调度和自适应选择来源，不因文献较新而扩大实现范围。
