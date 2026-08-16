# 为什么不能直接用全局变量？（硬核数学推导篇）

在回答数学推导之前，我们必须先解答你的第一个致命疑惑：**既然全局方差会随着当前 Batch 的增大而增大，为什么没有压制效果？**

## 一、 破除错觉：值的更新 $\neq$ 梯度的传递

在 PyTorch 等深度学习框架中，存在一个核心概念叫**计算图 (Computational Graph)**。只有被记录在计算图里的操作，才能参与反向传播求导。

- **当前 Batch 的方差** $\sigma_B^2$**：** 它是直接通过公式 $\frac{1}{m}\sum(x_i - \mu)^2$ 从 $x_i$ 算出来的。这个计算过程**在计算图内**。所以反向求导时，$\frac{\partial \sigma_B^2}{\partial x_i} \neq 0$。
- **全局方差的 EMA 更新：** 确实，你说的对，全局方差的值会通过 $0.9 \times 老记录 + 0.1 \times \sigma_B^2$ 被更新。**但是！这个更新操作在框架底层是被 `detach()`（截断）的！** 它属于一个“后台偷偷记账”的动作，绝对不会被加入到反向传播的计算图里。
- **为什么必须截断？** 如果不截断，那么求导时，当前 Batch 的梯度就要顺着那个 $0.9$ 一直往回追溯到上一个 Batch，再追溯到上上个 Batch……这叫“沿时间反向传播”，会直接把显存撑爆！

**结论：** 尽管在数值上，全局变量被 $x_i$ 影响了；但在微积分（求导）的眼里，**全局变量就是一块没有感情的石头（常数）**，$\frac{\partial \sigma_{global}}{\partial x_i}$ **永远等于 0**。

## 二、 数学对决：为什么常数会导致网络崩溃？

现在，我们直接用微积分的链式法则，对比一下三种情况的梯度推导。我们的终极目标是把误差传递回 $x_i$，即求 $\mathbf{\frac{\partial L}{\partial x_i}}$。（因为正如你所敏锐察觉的，只有求出了 $x_i$ 的误差，才能继续往回传，最终精准惩罚制造 $x_i$ 的上一层权重 $w_i$）。

### 方案 A：直接使用全局变量（常数）—— 刹车失灵的数学证明

假设我们不听劝，直接用全局均值 $\mu_G$ 和全局标准差 $\sigma_G$（它们在求导时是常数）。 前向公式为：
$$
\hat{x}_i = \frac{x_i - \mu_G}{\sigma_G}
$$
根据最基础的微积分链式法则求导：
$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i}
$$
因为 $\mu_G$ 和 $\sigma_G$ 是常数，所以 $\frac{\partial \hat{x}_i}{\partial x_i} = \frac{1}{\sigma_G}$。 代入得到：
$$
\mathbf{ \frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sigma_G} }
$$
**数学解读（为什么崩溃）：** 你看这个公式，梯度完全是一个线性常数倍数！如果 $x_i$ 极其巨大，这个梯度公式里**没有任何一项**会跳出来阻止它。网络只要觉得增大 $x_i$ 能降低 Loss，就会毫无阻力地无限增大 $w_i$ 和 $x_i$，最终走向数值溢出（NaN）。

### 方案 B：正常 BN（使用当前批次变量）—— 压制效应的数学证明

现在，我们老老实实用当前批次的 $\mu_B$ 和 $\sigma_B$。 前向公式依然是：
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sigma_B}
$$
**但是！极其重要的一点来了！** $\mu_B$ 和 $\sigma_B$ 不再是常数，它们都是由 $x_i$ 计算出来的变量！ 所以在进行多元函数求导时，链式法则必须**兵分三路**（这就是我们在前面文档里说的“大锅饭的连带责任”）：
$$
\frac{\partial L}{\partial x_i} = \underbrace{ \frac{\partial L}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} }_{\text{直接影响}} + \underbrace{ \frac{\partial L}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} }_{\text{均值的连带影响}} + \underbrace{ \frac{\partial L}{\partial \sigma_B^2} \frac{\partial \sigma_B^2}{\partial x_i} }_{\text{方差的连带影响}}
$$
经过极其漫长和复杂的微积分展开，我们直接跳到化简后的**终极公式**：
$$
\frac{\partial L}{\partial x_i} = \frac{1}{m \sigma_B} \left( m \frac{\partial L}{\partial \hat{x}_i} - \sum_{k=1}^m \frac{\partial L}{\partial \hat{x}_k} \mathbf{- \hat{x}_i \sum_{k=1}^m \frac{\partial L}{\partial \hat{x}_k} \hat{x}_k} \right)
$$
**见证奇迹的时刻：** 第三项 $\mathbf{- \hat{x}_i \sum \dots}$ 自带一个**负号**和 $\mathbf{\hat{x}_i}$。当上一层权重疯涨导致 $x_i$ 极大时，这一项会产生一个极其巨大的**负梯度反向拉力**，强制在下一次更新权重 $w_i$ 时，把 $w_i$ 给狠狠地拽回来！这就是完美的数学压制。

### 方案 C：Batch Renormalization (BN 的改进) —— 偷梁换柱，鱼与熊掌兼得

你一定会好奇，Batch Renorm 在数学上等于“直接使用全局变量”，那它是怎么保住方案 B 里的那个“负梯度弹簧”的？

答案就在于代码底层巧妙的**梯度截断技巧**。

Batch Renorm 的前向公式（篡改了标准 BN 的第 3 步）是：
$$
x'_i = \hat{x}_i \cdot r + d = \left( \frac{x_i - \mu_B}{\sigma_B} \right) \cdot r + d
$$
*(其中* $r$ *和* $d$ *是根据全局变量算出来的修正因子)*

**【极其关键的绝妙设计】：** 在深度学习框架中进行反向传播求导时，Batch Renorm 算法会**强制规定** $r$ **和** $d$ **是绝对的常数（不参与求导）**！

现在，我们对这个新公式求梯度 $\frac{\partial L}{\partial x_i}$。我们先求对中间变量 $\hat{x}_i$ 的导数：
$$
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial x'_i} \cdot \frac{\partial x'_i}{\partial \hat{x}_i}
$$
因为 $d$ 是常数，$r$ 也是常数，所以 $\frac{\partial x'_i}{\partial \hat{x}_i}$ 直接就等于 $\mathbf{r}$。
$$
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial x'_i} \cdot \mathbf{r}
$$
你看！算到这里，我们已经把来自最终 Boss 的误差信号，乘上常数 $r$ 之后，**完美地交接给了标准的** $\hat{x}_i$！

接下来的事情，**完完全全和【方案 B（正常 BN）】一模一样！** 因为 $\hat{x}_i = \frac{x_i - \mu_B}{\sigma_B}$，这里面依然包含着实时的均值和方差，所以链式法则依然会“兵分三路”，依然会生成那个带有负号的“弹簧惩罚项”： $\mathbf{- \hat{x}_i \sum \dots}$

### 深入追问：既然 $r$ 和 $d$ 包含 $x_i$，为什么数学上不把它们当变量求导？

你的质疑极其硬核：$r = \frac{\sigma_\beta}{\sigma_G}$，明明 $\sigma_\beta$ 是从 $x_i$ 算出来的，凭什么求导时忽略它？

这正是深度学习中“纯粹的数学”向“工程设计”妥协的经典案例。这叫做 **Stop-Gradient（梯度截断）**。作者（Sergey Ioffe）在原论文中明确说明了这是故意为之，原因有二：

**原因 1：防止目标错位（网络会去“作弊”）** 我们训练网络，是为了让网络去优化特征 $x_i$，从而做对分类任务。 $r$ 和 $d$ 只是一个“测量工具”，用来衡量当前批次偏离全局分布有多远。 如果我们把 $r$ 也当成变量求导，网络就会发现一条“捷径”：**为了降低 Loss，我不去好好提取特征了，我只要故意改变** $x_i$**，把当前批次的方差** $\sigma_\beta$ **强行拉大，就能改变** $r$**，从而把误差给抵消掉！** 这就好比：老师（BN层）给全班考试加了 10 分的修正（$d$）。学生（$x_i$）为了考高分，不去好好复习，反而想办法去“操纵老师的加分标准”。把 $r$ 和 $d$ 设为常数，就是断了网络的作弊念头。

**原因 2：避免退化回普通 BN 的深渊** 如果你非要严谨，对 $x'_i = \hat{x}_i \cdot r + d$ 进行完全的链式求导。 因为 $r$ 本身就是 $\frac{\sigma_\beta}{\sigma_G}$，展开求导后，那些多出来的梯度项会互相抵消和干扰，**把你好不容易加上的** $r$ **和** $d$ **的效果在梯度层面完全抹平！** 最终传下去的梯度，又会变得和“小 Batch 下极度不稳定的普通 BN 梯度”一模一样。这样 Batch Renorm 就彻底失去了意义。

**总结：** 在工程上，我们让 $r$ 和 $d$ 只在**前向传播**时起作用（用来对齐分布，防止数据飘飞），但在**反向传播**时强制切断它们与 $x_i$ 的导数联系（当成常数）。这属于一种“算法 Hack”。你的直觉没有错，数学上确实被截断了，而这恰恰是这个算法能够成功运作的灵魂所在。