# 反向传播（Backpropagation, BP）算法总结

> 一份关于反向传播算法原理与规则的笔记。
> 前置知识：前向传播（加权求和 + 激活）、损失函数、链式法则、梯度下降。

---

## 一、BP 是什么

BP（反向传播）是训练神经网络的核心算法，它的目标是：

> **算出损失函数 $\mathcal{L}$ 对每一个参数（权重 $w$、偏置 $b$）的梯度，供梯度下降更新参数使用。**

一句话：前向传播算"答案"，反向传播算"每个参数该改多少"。

```
① 前向传播：输入 → ... → 输出 → 算 loss
② 反向传播：loss → ... → 算出每个 w、b 的梯度（BP 算法）
③ 更新参数：w = w - lr × 梯度（梯度下降）
```

BP 只负责第 **②** 步——**算梯度，不更新参数**。更新是第 ③ 步梯度下降做的。

---

## 二、核心原理：链式法则（Chain Rule）

### 2.1 链式法则是什么

如果 $y = f(u)$，$u = g(x)$，那么 $y$ 对 $x$ 的导数是"两段导数相乘"：

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

BP 的本质，就是**沿着前向传播的计算路径，把每一段的局部导数一路乘回去**。

### 2.2 前向传播是一条"链"

一个神经元的前向是两步：$z = w \cdot x + b$，然后 $a = f(z)$，再算 loss：

$$
x \xrightarrow{\ \times w + b\ } z \xrightarrow{\ f\ } a \xrightarrow{\ \text{loss}\ } \mathcal{L}
$$

反向传播就是**倒着**把这条链的导数乘回去：

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

其中三段分别是：

| 段 | 导数 | 含义 |
|----|------|------|
| $\dfrac{\partial \mathcal{L}}{\partial a}$ | 损失函数对输出求导 | 前面传来的梯度 |
| $\dfrac{\partial a}{\partial z} = f'(z)$ | **激活函数的导数** | 穿过激活函数 |
| $\dfrac{\partial z}{\partial w} = x$ | 加权求和对权重求导 | 穿过加权求和 |

**核心规则①**：反向传播穿过激活函数时，必须乘上它的导数 $f'(z)$。

---

## 三、算法的三个步骤

### 第 1 步：前向传播

从输入层开始，逐层算加权求和 + 激活，直到输出层，算出预测值和 loss。

### 第 2 步：反向传播（算梯度）

从输出层开始，**从后往前**逐层推进，用链式法则算出每个 $w$、$b$ 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial w^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(l)}} \cdot f'(z^{(l)}) \cdot a^{(l-1)}
$$

### 第 3 步：更新参数（梯度下降）

$$
w \leftarrow w - lr \cdot \frac{\partial \mathcal{L}}{\partial w}
$$

---

## 四、单个神经元的反向传播规则

一个神经元前向是 $z = w \cdot x + b$、$a = f(z)$，它的反向规则是：

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot f'(z) \cdot x
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial a} \cdot f'(z) \cdot 1
$$

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial a} \cdot f'(z) \cdot w \quad (\text{继续传给上一层的输入})
$$

**核心规则②**：对 $w$ 求导要乘输入 $x$，对 $b$ 求导乘 1，对输入 $x$ 求导乘 $w$（用来继续往前传）。

---

## 五、多层网络的反向传播规则

### 5.1 从后往前，逐层算

反向传播**从输出层开始**，按"输出层 → 隐藏层2 → 隐藏层1 → 输入层"的顺序，一层层往前，**每个神经元都要算一遍**自己的 $w$、$b$ 梯度。

### 5.2 梯度累加规则（关键！）

一个神经元如果被**多个下游神经元**连接，它收到的梯度是**这几路之和**：

$$
\frac{\partial \mathcal{L}}{\partial a_j} = \sum_{k} \frac{\partial \mathcal{L}}{\partial a_k} \cdot \frac{\partial a_k}{\partial a_j}
$$

（所有下游神经元传给它的梯度要**加起来**。）

---

## 六、完整手算例子（两层隐藏层）

网络：输入 1 个 → 隐藏层1（2神经元）→ 隐藏层2（2神经元）→ 输出 1 个。为了数字干净，先不加激活函数，损失用 MSE。

### 网络与权重

$$
h_1 = w_{11}x + b_1,\quad h_2 = w_{12}x + b_2
$$

$$
h_3 = w_{31}h_1 + w_{32}h_2 + b_3,\quad h_4 = w_{41}h_1 + w_{42}h_2 + b_4
$$

$$
o = w_{51}h_3 + w_{52}h_4 + b_5,\quad \mathcal{L} = (o - y)^2
$$

取值：$x=1$，$y=1$；$w_{11}=1, b_1=0$；$w_{12}=2, b_2=0$；$w_{31}=1, w_{32}=1, b_3=0$；$w_{41}=1, w_{42}=-1, b_4=0$；$w_{51}=1, w_{52}=1, b_5=0$。

### ① 前向传播

$$
h_1 = 1,\quad h_2 = 2,\quad h_3 = 3,\quad h_4 = -1,\quad o = 2,\quad \mathcal{L} = (2-1)^2 = 1
$$

### ② 反向传播（从后往前）

**输出层**：

$$
\frac{\partial \mathcal{L}}{\partial o} = 2(o-y) = 2
$$

$$
\frac{\partial \mathcal{L}}{\partial w_{51}} = 2 \cdot h_3 = 6,\quad
\frac{\partial \mathcal{L}}{\partial w_{52}} = 2 \cdot h_4 = -2
$$

$$
\frac{\partial \mathcal{L}}{\partial h_3} = 2 \cdot w_{51} = 2,\quad
\frac{\partial \mathcal{L}}{\partial h_4} = 2 \cdot w_{52} = 2
$$

**隐藏层2**（$h_3$、$h_4$ 各算各的，然后累加）：

$$
\frac{\partial \mathcal{L}}{\partial w_{31}} = 2 \cdot h_1 = 2,\quad
\frac{\partial \mathcal{L}}{\partial w_{32}} = 2 \cdot h_2 = 4
$$

$$
\frac{\partial \mathcal{L}}{\partial w_{41}} = 2 \cdot h_1 = 2,\quad
\frac{\partial \mathcal{L}}{\partial w_{42}} = 2 \cdot h_2 = 4
$$

**梯度累加**（$h_1$ 同时被 $h_3$、$h_4$ 连接）：

$$
\frac{\partial \mathcal{L}}{\partial h_1} = \underbrace{2}_{\text{来自 }h_3} + \underbrace{2}_{\text{来自 }h_4} = 4
$$

$$
\frac{\partial \mathcal{L}}{\partial h_2} = 2 + (-2) = 0
$$

**隐藏层1**：

$$
\frac{\partial \mathcal{L}}{\partial w_{11}} = 4 \cdot x = 4,\quad
\frac{\partial \mathcal{L}}{\partial w_{12}} = 0 \cdot x = 0
$$

### ③ 更新参数

$$
w_{11} \leftarrow 1 - lr \cdot 4, \quad \dots \quad \text{（每个参数都按 } w \leftarrow w - lr\cdot\nabla w \text{ 更新）}
$$

---

## 七、BP 核心规则速记

1. **从后往前**：从输出层开始，逐层向输入层推进。
2. **链式法则**：每一段的局部导数连乘。
3. **穿过激活函数乘 $f'(z)$**：这是激活函数影响反向传播的方式。
4. **对 w 乘输入、对 b 乘 1**：$\frac{\partial \mathcal{L}}{\partial w} = \delta \cdot x$，$\frac{\partial \mathcal{L}}{\partial b} = \delta$（$\delta$ 是传到这一层的梯度）。
5. **梯度累加**：一个神经元被多个下游连接时，梯度是几路之和。
6. **每个参数都算**：没有哪个神经元的 w、b 被跳过。
7. **BP 只算梯度不更新**：更新交给梯度下降（`optimizer.step()`）。

**统一公式**（记这一个就够）：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot f'(z) \cdot x}
$$

---

## 八、一句话总结

反向传播 = **用链式法则，从输出层开始、从后往前，把损失对每个参数的梯度一路乘出来**。规则就三条：穿过激活函数乘 $f'(z)$、对权重乘输入 $x$、被多个神经元连接时梯度累加。它只负责"算梯度"，真正的参数更新由梯度下降（$w \leftarrow w - lr \cdot \nabla w$）完成。
