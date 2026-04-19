# PR 文档草稿：BBSE 与 OWLS 稳定性修复

## 建议标题

Fix BBSE prior/stability issues and improve OWLS rescale correctness and diagnostics

## 摘要

这个 PR 主要做两类工作：

1. 修复闭集 label shift 基线 `BBSE` 中的先验约束和数值稳定性问题
2. 修复开放集 `OWLS / OSLS` 路径中的 rescale 分支错误，并补充关键诊断信息

这次改动的重点不是引入新算法，而是修正当前实现里会影响结果可信度的几个关键问题，让实验结果更符合方法设定，也让后续排查更有抓手。

从影响面上看，这个 PR 主要提升三件事：

- 正确性：修复会直接改变估计结果的实现错误
- 稳定性：降低求解器或数值扰动带来的异常输出
- 可观测性：让 `BBSE negative w` 和 `OWLS r=1` 这类问题不再只能靠猜

## 背景

最近在 Open Set Label Shift 评估过程中，持续暴露出两类问题。

第一类问题发生在 BBSE 路径。现有代码在运行时可能会出现 `Negative value exist in BBSE estimation of w` 的 warning。这个 warning 表示 BBSE 估计出的类别重加权系数 `w` 已经离开了概率意义上的可行域。虽然代码后面会对负值做裁剪，但这通常不是一个可以忽略的小波动，而是在提示当前估计已经不稳定。进一步静态检查后又发现，`cvxpy` 版本的 BBSE 在约束中使用的源域先验本身也构造错了，这会进一步放大问题。

第二类问题发生在 OWLS / OSLS 路径，尤其是在 `r=1` 的设置下更明显。静态排查发现，当前 `rescale` 的分支选择逻辑与实际调用的 rescale 方法原先是反着接的，这会直接影响 `sigma_1`、`sigma_0`、`rho_s`、`rho_t0`、`rho_t` 这一整条估计链路。与此同时，当前实现仍然把 postprocessor 的原始 OOD 分数通过 `1 + conf` 的方式送进 OWLS 当作 `h(x)`，这和论文中更接近概率型 ID score 的假设并不完全一致，因此也需要明确暴露出来，而不是继续作为隐含行为存在。

基于这两个问题，这个 PR 将 BBSE 和 OWLS 的修复统一整理为一次“label shift 可靠性修复”，但在实现上仍然保持两条链路各自独立、便于 review。

## 改动内容

### 1. 修复 `bbse_lsq.py` 中的源域先验构造

此前 `cvxpy` 版本 BBSE 在构造约束时，并不是根据 `train_labels` 的真实类别频率来得到源域先验，而是错误地用“出现过哪些类别 id”去拼约束项。这样得到的并不是 `P_s(Y=i)`，会导致 BBSE 的约束条件偏离方法本意。

这次改动将其修正为基于 `train_labels` 真实频率统计得到 `source_py`，从而让约束真正对应到：

`sum_i P_s(Y=i) * w_i = 1`

这个修改的重要性在于：

- 旧实现会静默地把优化问题推向错误的可行域
- 即使求解器正常返回，`w` 也可能因为约束错误而被系统性拉偏
- 在长尾场景下，这种偏差会更明显，因为各类先验质量本来就不均衡

### 2. 为 BBSE 增加显式的稳定化处理

这次在 BBSE 求解结束后新增了 `stabilize_w(...)`，用于把求解器输出重新投回 BBSE 的合理可行域。

新逻辑包括：

- 只在出现明显负值时发出 warning
- 将解裁剪到非负区间
- 按照修正后的 `source_py` 做重新归一化
- 如果裁剪后已经退化到无意义状态，则直接报错而不是继续吞掉问题

这个修改解决的是一个很实际的问题：仅仅把负值裁成 `0` 并不够，因为裁剪之后的结果仍然可能不满足 BBSE 需要的归一化约束。也就是说，旧逻辑只是“让输出看起来合法”，但没有保证它仍然在 BBSE 的数学设定里。现在这一步做完之后，`w` 至少会被拉回到一个更合理的约束集合中。

### 3. 为 `cvxpy` BBSE 增加求解器回退

在 `cvxpy` 路径中，现在优先使用 `OSQP`，如果失败则回退到 `SCS`。

这部分改动的目标比较直接：

- 降低求解器自身状态导致的评估失败
- 在不改变外部使用方式的前提下，提高 BBSE 路径的鲁棒性

### 4. 修复 `open_set_label_shift.py` 中的 rescale 分支反接问题

静态检查确认，当前 OWLS 路径里 `rescale` 模式的判定与后续实际调用的 rescale 方法原先是反着接的。这个 PR 将其修正为：

- `threshold`，或 `auto` 且 `cls_num <= 100` 时，使用 threshold rescale
- `logistic`，或 `auto` 且 `cls_num > 100` 时，使用 logistic rescale

这是本次 PR 中最核心的 OWLS 实现修复之一，因为它不是日志层面的“可见性问题”，而是会直接改变估计链路行为的逻辑错误。由于 `h(x)` rescale 后的统计量会继续传导到 `rho_s` 与 `rho_t` 修正，这个 bug 足以单独解释一部分 `r=1` 设定下的异常现象。

### 5. 显式暴露当前 OWLS 对 `1 + conf` 的依赖

当前实现仍然保留了原有行为：先取 postprocessor 的原始 OOD 分数 `conf`，再通过 `1 + conf` 的方式把它送进 OWLS，当作后续 rescale 的输入。

这次 PR 没有直接改写这条路径，而是先把它明确暴露出来，并在 evaluator 和 OWLS 内部增加 warning 与诊断信息，说明这是一种兼容当前实现的启发式做法，不应直接等同于论文中严格意义上的概率型 `h(x)`。

这样做的价值在于：

- 不打破现有实验行为，避免一次 PR 改动过大
- 让 reviewer 和后续实验使用者明确知道当前代码与论文假设之间仍有差距
- 一旦后续 `rho_t` 表现异常，可以先从 `h(x)` 的构造开始定位，而不是误以为都是 EM 或 correction 公式本身的问题

### 6. 为 OWLS 整条链路补充诊断信息

为了让 `r=1` 失效这类问题真正可排查，这次在源域和目标域两侧都补充了更细的统计输出。当前会记录：

- raw postprocessor score 的统计
- `1 + conf` 之后 `shifted_for_owls` 的统计
- rescaled 之后 ID / OOD 的均值、median、gap
- ID 各类别上的 classwise mean 统计
- `rho_s`、`rho_t0`、`rho_t`、`rho_t_gt`
- `sigma_1`、`sigma_0`、`sigma_0 / T`
- `rho_t_correction()` 中的原始值、gap、裁剪前后结果

这部分改动的目标不是“多打一些日志”，而是让我们能够区分几类完全不同的失败来源：

- 原始 OOD score 本身就不适合直接拿来做 `h(x)`
- score 在 rescale 后 ID/OOD 仍然分不开
- `mu_1 - mu_0` 太小，导致 correction 公式数值敏感
- `rho_t` 的异常主要来自 correction，而不是 EM 估计本身

### 7. 明确当前 `MAP` 路径的解释边界

在 `MAP` 路径里，这次补充了显式 warning，说明当前代码更应被理解为“启发式先验版 EM”，而不是对论文 OSLS-MAP 的严格一比一实现。同时默认 OOD prior 也调整成了更中性的对称形式。

这一点很重要，因为在没有明确提示的情况下，实验结果很容易被误读成“已经严格复现了 paper 里的 MAP 版本”。现在至少解释边界是清楚的：这个 PR 改善了实现，但没有过度宣称它已经完全对齐理论版本。

### 8. 补充与复现相关的辅助改动

除了上述核心修复，这次也顺手补了 `scripts/eval_ood.py` 的 `--data-root` 参数，避免评估脚本依赖硬编码数据路径。这个改动虽然不属于 BBSE 或 OWLS 算法本身，但对复现实验和对外使用会更友好。

## 预期收益

合入这个 PR 之后，预期会有以下改进：

- BBSE 的 `w` 估计会满足正确的源域先验约束
- BBSE 出现负值时，更容易区分是数值噪声还是明显失稳
- OWLS 会真正使用预期的 rescale 策略，而不是走错分支
- `r=1` 场景下的异常不再只能看最终 `rho_t`，而是能沿链路回溯到具体失真位置
- 日志会更清楚地区分“实现 bug”与“方法假设尚未满足”这两类问题

## 不在本次 PR 解决范围内的事项

这个 PR 解决的是明确的实现问题和可观测性问题，但它并不声称已经完全解决了论文与代码之间的所有差异。

当前仍然没有在这个 PR 中完成的事情包括：

- 用严格校准后的概率型 `h(x)` 替换当前 `1 + conf` 的启发式路径
- 保证所有数据集、所有 postprocessor 下的 `r=1` 问题都被彻底修复
- 完整重写并严格对齐论文里的 OSLS-MAP 推导和实现

换句话说，这个 PR 的定位是“先修确定有问题的实现，再把剩余问题暴露出来”，而不是一次性把所有理论差异都消灭掉。

## 建议的验证方式

这次改动最有价值的验证，不是只看单个最终指标，而是结合新增诊断一起看行为有没有变得更合理。

建议优先验证以下配置：

1. `cifar100 + LT100 + r=1.0`
2. 同配置下对照 `r=0.1`

建议重点观察：

- source / target 的 rescaled score gap 是否变得更合理
- target ID 上的 classwise mean std 是否异常大
- `rho_t correction stats` 是否仍然频繁发生 clipping
- BBSE 在长尾配置下是否仍然经常出现明显负值

如果修复后 `r=1.0` 仍然明显把 `rho_t` 推向 `1.0`，下一步更应该优先收紧 `h(x)` 的构造，而不是继续把问题归因到闭集 baseline。

## 主要涉及文件

- `openood/label_shift/closed_set_label_shift/bbse_lsq.py`
- `openood/label_shift/open_set_label_shift.py`
- `openood/evaluation_api/evaluator.py`
- `openood/label_shift/models.py`
- `openood/label_shift/em.py`
- `scripts/eval_ood.py`

## Review 关注点

建议 reviewer 重点看三件事：

1. BBSE 现在的约束与稳定化是否真正回到了方法要求的可行域
2. OWLS 的 rescale 分支修复是否与原设计意图一致
3. 新增 warning 和 diagnostics 是否足以把“实现错误”和“建模限制”区分开

如果需要拆开 review，这个 PR 也可以自然分成两部分来看：

- BBSE 先验修复与稳定化
- OWLS rescale 修复与诊断增强
