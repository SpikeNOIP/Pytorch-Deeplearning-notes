# Sigmoid 激活函数总结

> 一份关于 Sigmoid 激活函数的原理、范围、用法与适用场景的笔记。
> 环境：PyTorch `torch.sigmoid` / `torch.nn.Sigmoid`。

---

## 一、原理

### 1.1 数学定义

Sigmoid 把任意实数压缩到 $(0, 1)$ 区间，公式为：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

其中 $e$ 是自然常数（约 $2.71828$），$x$ 是神经元的加权求和结果 $z = w \cdot x + b$。

### 1.2 函数图像（S 形曲线）

Sigmoid 是一条**光滑的 S 形（sigmoid = S 形）曲线**：

- 当 $x \to +\infty$：$\sigma(x) \to 1$
- 当 $x \to -\infty$：$\sigma(x) \to 0$
- 当 $x = 0$：$\sigma(0) = 0.5$

它**单调递增**，两头趋向 0 和 1，中间（$x=0$ 附近）接近一条直线。

### 1.3 导数（反向传播要用）

Sigmoid 的导数有一个非常漂亮的简洁形式：

$$
\sigma'(x) = \sigma(x) \cdot \bigl(1 - \sigma(x)\bigr)
$$

也就是说，导数值可以**直接用函数输出 $\sigma(x)$ 算出来**，不需要额外计算 $e^{-x}$。

**导数的取值范围**：

$$
0 < \sigma'(x) \le 0.25
$$

最大值出现在 $x = 0$ 处（此时 $\sigma(0)=0.5$，$\sigma'(0)=0.5 \times 0.5 = 0.25$）。两头（$|x|$ 很大时）导数趋近于 0——这正是"梯度消失"的根源（见第四节）。

---

## 二、范围

| 项目 | 值 |
|------|-----|
| 输入范围 $x$ | $(-\infty, +\infty)$，任意实数 |
| 输出范围 $\sigma(x)$ | $(0, 1)$，**开区间**，不包含 0 和 1 |
| 中心点 | $\sigma(0) = 0.5$ |
| 输出符号 | 恒为正（**非零中心**，这是缺点之一） |

**关键含义**：输出永远落在 $(0, 1)$ 之间，因此天然可以解释成**概率**——这就是它被大量用于二分类输出层的原因。

---

## 三、用法（PyTorch）

### 3.1 三种等价写法

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-2.0, 0.0, 2.0])

# 写法1：函数式
y1 = torch.sigmoid(x)
y1 = F.sigmoid(x)

# 写法2：作为网络层（放进 nn.Sequential）
sigmoid_layer = nn.Sigmoid()
y2 = sigmoid_layer(x)

# 结果
print(y1)   # tensor([0.1192, 0.5000, 0.8808])
```

### 3.2 二分类输出层的标准做法（重点）

二分类模型最后一层输出 1 个值，再接 Sigmoid 变成概率：

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1),    # 输出 1 个分数（logit）
    nn.Sigmoid(),        # 转成 (0,1) 的概率
)

prob = model(x)          # prob ∈ (0, 1)
pred = (prob > 0.5).long()   # 概率 > 0.5 判为类别1
```

### 3.3 数值稳定的推荐写法

训练二分类时，**推荐用 `nn.BCEWithLogitsLoss`**（它内部把 Sigmoid 和交叉熵合并，数值更稳定），此时**模型最后一层不要自己加 Sigmoid**：

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1),    # 只输出 logit，不加 Sigmoid
)

criterion = nn.BCEWithLogitsLoss()   # 内部自动做 sigmoid + BCE
loss = criterion(logits, labels)     # logits 是未经过 sigmoid 的分数
```

---

## 四、适用场景

### ✅ 适合

| 场景 | 为什么 |
|------|--------|
| **二分类输出层** | 输出 $(0,1)$，天然是"属于类别1的概率" |
| **多标签分类** | 每个输出神经元独立表示"是否含某标签"，各是一个 $(0,1)$ 概率 |
| **门控机制**（LSTM/GRU 的门） | 输出 $(0,1)$ 恰好能当"门的开合程度"（0 关 1 开） |
| 需要平滑概率输出的地方 | 输出连续、可解释为概率 |

### ❌ 不适合（作为隐藏层激活函数已基本被淘汰）

| 问题 | 原因 |
|------|------|
| **梯度消失** | 导数最大才 $0.25$，反向传播逐层相乘，越乘越小，深层网络学不动 |
| **非零中心** | 输出恒为正，导致反向传播时所有梯度同号（全是正或全是负），优化时"之"字形震荡，收敛变慢 |
| **指数运算** | 含 $e^{-x}$，计算比 ReLU 慢 |

> 现在隐藏层几乎都用 **ReLU**（$f(x)=\max(0,x)$）及其变体，Sigmoid 主要保留在**二分类输出层**和**门控结构**里。

---

## 五、核心公式速记

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \qquad
\sigma(x) \in (0, 1)
$$

$$
\sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr), \qquad
0 < \sigma'(x) \le 0.25
$$

$$
\lim_{x\to +\infty}\sigma(x)=1,\qquad
\lim_{x\to -\infty}\sigma(x)=0,\qquad
\sigma(0)=0.5
$$

---

## 六、一句话总结

Sigmoid 把任意实数压到 $(0,1)$，输出可解释为**概率**，因此最常用在**二分类/多标签的输出层**和**门控机制**里；但它导数最大只有 $0.25$、输出非零中心，会导致**梯度消失**和收敛变慢，所以**不适合做隐藏层激活函数**（已被 ReLU 取代）。