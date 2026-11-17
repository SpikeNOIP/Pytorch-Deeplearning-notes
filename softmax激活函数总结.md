# Softmax 激活函数总结

> 一份关于 Softmax 的原理、范围、用法与适用场景的笔记。
> 环境：PyTorch `torch.softmax` / `torch.nn.Softmax`。

---

## 一、原理

### 1.1 数学定义

Softmax 把一个**向量**（一组任意实数，常叫 logits/分数）转换成一个**概率分布**。对第 $i$ 个元素：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\displaystyle\sum_{j=1}^{K} e^{z_j}}
$$

其中 $K$ 是类别数，$z = [z_1, z_2, \dots, z_K]$ 是输出层 $K$ 个神经元产生的原始分数。

### 1.2 它和 Sigmoid/ReLU/Tanh 的本质区别

前面几个激活函数都是**逐元素（element-wise）**的——每个输入 $x$ 独立算，互不影响。Softmax 不同：

- **它是"向量级"函数**：算第 $i$ 个输出时，要**除以所有元素之和**，所以每个输出都依赖其它所有元素。
- 输出**彼此约束**：所有输出加起来恒等于 1。

所以 Softmax 的正确用法是"**给一组分数整体转成概率**"，而不是"对单个神经元做非线性变换"。

### 1.3 核心性质

1. **输出是概率分布**：每个 $\text{softmax}(z_i) \in (0,1)$，且 $\sum_i \text{softmax}(z_i) = 1$。
2. **保持大小顺序**：$z$ 越大的元素，softmax 后概率越大（单调递增），所以 argmax 的结果不变。
3. **平移不变性**：给所有分数同加一个常数，结果不变：

$$
\text{softmax}(z_i + c) = \text{softmax}(z_i)
$$

### 1.4 导数（是雅可比矩阵，不是标量）

因为 Softmax 是"向量 → 向量"，它的导数是一个**雅可比矩阵**。对第 $i$ 个输出关于第 $j$ 个输入求偏导：

$$
\frac{\partial p_i}{\partial z_j} = p_i\,(\delta_{ij} - p_j)
$$

其中 $p_i = \text{softmax}(z_i)$，$\delta_{ij}$ 是克罗内克符号（$i=j$ 时为 1，否则为 0）。展开成两种情况：

$$
\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i), & i = j \\[4pt] -p_i\,p_j, & i \neq j \end{cases}
$$

> 直观理解：某类别概率变高，别的类别概率必然被"挤"下来，所以有正有负、彼此耦合。这也是 Softmax 与逐元素激活函数（导数各自独立）最大的不同。

---

## 二、范围

| 项目 | 值 |
|------|-----|
| 输入 $z_i$ | $(-\infty, +\infty)$，任意实数 |
| 每个输出 $p_i$ | $(0, 1)$，开区间 |
| 所有输出之和 | **恒等于 1**（概率分布） |
| 用途 | 把"分数"变成"概率" |

**关键含义**：输出可直接解释为"属于各类别的概率"，天然适合**多分类**。

---

## 三、用法（PyTorch）

### 3.1 三种等价写法

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

z = torch.tensor([2.3, -1.5, 0.4])   # 3 个类别的分数（logits）

# 写法1：函数式（必须指定 dim）
p1 = torch.softmax(z, dim=0)
p1 = F.softmax(z, dim=0)

# 写法2：作为网络层
softmax_layer = nn.Softmax(dim=1)    # dim=1 表示对"类别"那一维做 softmax
p2 = softmax_layer(z.unsqueeze(0))

print(p1)   # tensor([0.835, 0.018, 0.147])，加起来 = 1
```

> **注意 `dim` 参数**：一定要指定对哪一维做 softmax。对 2D 输入 `(batch, 类别)` 通常用 `dim=1`（对每个样本的各类别概率归一化）。

### 3.2 多分类输出层的标准做法

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3),      # 输出 3 个分数（3 个类别）
)

logits = model(x)          # 原始分数
prob = torch.softmax(logits, dim=1)   # 转成概率
pred = torch.argmax(prob, dim=1)      # 取概率最大的类别
```

### 3.3 数值稳定的推荐写法（训练时）

训练多分类时，**推荐用 `nn.CrossEntropyLoss`**，它内部自动做了 `log_softmax + NLLLoss`，此时**模型最后一层不要自己加 Softmax**（否则会重复）：

```python
criterion = nn.CrossEntropyLoss()     # 内部已含 softmax
loss = criterion(logits, labels)      # logits 是未经过 softmax 的分数
```

**数值稳定性技巧**：手写 Softmax 时，先减掉最大值再求 exp，避免 $e^{z_i}$ 溢出：

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

（根据平移不变性，减不减最大值结果相同，但能防止溢出。）

---

## 四、适用场景

### ✅ 适合

| 场景 | 为什么 |
|------|--------|
| **多分类输出层**（3 类及以上） | 把 $K$ 个分数转成 $K$ 个概率，和为 1 |
| 需要输出"概率分布"的地方 | 输出可直接解释为置信度 |

### ❌ 不适合

| 场景 | 原因 |
|------|------|
| **二分类输出层** | 用 Sigmoid（1 个输出概率）更简单，Softmax 也可以但多此一举 |
| **隐藏层** | Softmax 会强制所有神经元输出和为 1，彼此耦合，不适合做隐藏层激活 |
| **多标签分类**（一个样本可属多个类） | 各类别不互斥，不能用"和为 1"的 Softmax，应用 Sigmoid |

---

## 五、Softmax 与 Sigmoid 对比

| 对比项 | Sigmoid | Softmax |
|--------|---------|---------|
| 作用对象 | 单个标量 | 整个向量 |
| 输出个数 | 1 个 | K 个（和输入一样多） |
| 输出关系 | 独立 | 相互耦合，和为 1 |
| 主要用途 | 二分类输出层 | 多分类输出层 |
| 隐藏层 | 曾经用，现已淘汰 | 不用 |

---

## 六、核心公式速记

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \qquad \sum_{i=1}^{K} \text{softmax}(z_i) = 1
$$

$$
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)
$$

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}} \quad (\text{数值稳定形式})
$$

---

## 七、一句话总结

Softmax 是把**一组分数（logits）转成概率分布**的"向量级"函数：$\text{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}$，输出都在 $(0,1)$ 且和为 1，因此**专门用于多分类输出层**；它不是逐元素函数（每个输出依赖所有输入），导数是一个雅可比矩阵，且不适合做隐藏层激活。
