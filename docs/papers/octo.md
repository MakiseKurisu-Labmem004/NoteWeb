---

title: "Octo: An Open-Source Generalist Robot Policy"
description: "Octo 的核心问题是如何用一个通用机器人策略兼容不同机器人、传感器、任务定义和动作空间。核心理解：将异构输入统一为 Token，通过 Block-wise Causal Transformer 得到 Readout 决策表示，再用轻量 Diffusion Head 生成连续 Action Chunk。"
date: "2026-09-05"
tags: ["待分类"]
status: "待读"
venue: "Robotics: Science and Systems (RSS) 2024"
authors: "Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, et al."
paper: ""
code: ""
--------

# Octo

## 1. 一句话理解

Octo 是一个 **Transformer-based generalist robot policy**：

$$
\boxed{
\text{异构 Observation / Task}
\rightarrow
\text{统一 Tokens}
\rightarrow
\text{Transformer}
\rightarrow
\text{Readout}
\rightarrow
\text{Diffusion Action Head}
\rightarrow
\text{Action Chunk}
}
$$

真正重要的不是“用了 Transformer + Diffusion”，而是：

> **用模块化 Token 接口统一不同机器人输入，用 Readout Token 把通用 Transformer 表示连接到可替换的 Action Head。**

因此到了新机器人上，可以增加新的 Sensor Adapter / Action Head，再用少量数据微调。

---

# 2. 整体架构

脑中记住这一条：

```text
Language / Goal Image
        │
Images / Proprio / Sensors
        │
        ▼
Tokenizer / Adapter
        │
        ▼
统一 Token Sequence
        │
        ▼
Block-wise Causal Transformer
        │
        ▼
Readout Token R_t
        │
        ▼
决策表示 e_R,t
        │
        ▼
Diffusion Action Head
        │
        ▼
[a_t, a_t+1, ..., a_t+H]
```

核心分工：

* **Tokenizer / Adapter**：解决“输入长得不一样”
* **Transformer**：融合任务、视觉和历史信息
* **Readout Token**：提取当前时刻的决策表示
* **Diffusion Head**：把决策表示变成连续动作序列

---

# 3. 输入为什么能兼容不同机器人

Octo 不要求所有机器人原始输入完全一样，而是要求：

$$
\boxed{\text{最终都变成同一 hidden dimension 的 Token}}
$$

例如：

```text
Image      → Image Tokenizer → visual tokens
Language   → Text Tokenizer  → task tokens
Proprio    → Adapter         → state tokens
Force      → New Adapter     → force tokens
```

不同数据集缺某种输入：

$$
\boxed{\text{Padding + Mask}}
$$

新增传感器：

$$
\boxed{\text{New Sensor} \rightarrow \text{New Adapter} \rightarrow Token}
$$

因此 Octo 的“通用”更多体现在 **统一 Token Interface**，而不是强迫所有机器人使用完全相同的硬件接口。

---

# 4. Token Sequence + Attention

序列大致：

$$
[
T_{task},
O_{t-1},
R_{t-1},
O_t,
R_t
]
$$

其中一个时间步的 Observation 又可以包含：

$$
O_t=
[
T_t^{primary},
T_t^{wrist},
T_t^{state}
]
$$

### Block-wise Causal Attention

不要把它理解成普通 Decoder 的下三角 causal mask。

核心规则：

$$
\boxed{
\text{同一 timestep 的不同 observation 可以互相 Attention}
}
$$

但：

$$
\boxed{
\text{过去不能看到未来 observation}
}
$$

所以它同时需要：

* 同一时刻多模态充分融合
* 保持时间因果关系

可以记成：

```text
t-1 observation  ←→  t-1 observation

        ↓

t observation    ←→  t observation

但 t-1 看不到 t
```

---

# 5. Readout Token：Octo 最关键的接口

每个时间步都有一个：

$$
R_t
$$

它可以读取：

$$
\boxed{
Task + O_{\le t}
}
$$

但 Observation / Task **不会反过来读取 Readout**。

所以经过 Transformer：

$$
R_t\rightarrow e_{R,t}
$$

其中：

$$
\boxed{
e_{R,t}
=
\text{截至 }t\text{ 时刻，对“现在该怎么行动”的决策表示}
}
$$

注意：

$$
\boxed{e_{R,t}\neq a_t}
$$

它只是 Action Head 的 condition。

真正关系是：

$$
e_{R,t}
\rightarrow
\text{Action Head}
\rightarrow
[a_t,a_{t+1},...]
$$

Readout 的意义就是把：

> **通用 Transformer 表示**

和

> **具体 Action Decoder**

解耦。

---

# 6. 为什么输出 Action Chunk

不是：

$$
o_t\rightarrow a_t
$$

而是：

$$
\boxed{
o_{\le t}
\rightarrow
[a_t,a_{t+1},a_{t+2},a_{t+3}]
}
$$

Action Chunk 的主要作用：

* 直接建模一段局部动作轨迹
* 提高动作时间连续性
* 避免每一步独立预测造成抖动
* 配合 receding horizon 实现闭环控制

Octo 默认配置中可以记：

$$
H_{action}=4
$$

---

# 7. Diffusion Action Head

Transformer 不直接预测 Action。

它产生：

$$
e_{R,t}
$$

Diffusion Head 接收：

$$
\boxed{
e_{R,t},\;x_k,\;k
}
$$

其中：

* \(e_{R,t}\)：Transformer 给出的条件
* \(x_k\)：当前 noisy action chunk
* \(k\)：diffusion timestep

结构可简化为：

```text
e_R,t ──────────────┐
                    │
x_k ────────────────┼→ MLPResNet → ε_hat
                    │
k → Fourier Embed ──┘
```

训练目标：

$$
x_k=
\sqrt{\bar\alpha_k}x_0+
\sqrt{1-\bar\alpha_k}\epsilon
$$

然后预测：

$$
\hat\epsilon
=
\epsilon_\theta(x_k,k,e_R)
$$

Loss：

$$
\boxed{
L=\|\epsilon-\hat\epsilon\|^2
}
$$

### 为什么不用普通 L2 Action Regression？

因为机器人动作可能是多模态的。

例如：

```text
绕左边   ✓
绕右边   ✓
```

L2 容易学习二者平均：

```text
直着走   ✗
```

即 **hedging**。

Diffusion 则能表示一个复杂的连续动作分布。

---

# 8. 训练数据流

记住：

```text
Raw Trajectory
      ↓
统一数据格式 / Action normalization
      ↓
Observation Window + Action Chunk
      ↓
Goal Relabel / Task
      ↓
Padding + Mask
      ↓
多数据集 Mixing + Shuffle
      ↓
Image Augmentation
      ↓
Batch
      ↓
Tokenizer
      ↓
Transformer
      ↓
Readout
      ↓
Diffusion Loss
```

最重要的数据形状：

$$
Observation:
[B,H_{obs},...]
$$

$$
Action:
[B,H_{obs},H_{action},D_{action}]
$$

因此 observation window 中 **每一个时间步都有自己的 action chunk target**。

例如：

```text
R_t-1 → [a_t-1, a_t,   a_t+1, a_t+2]

R_t   → [a_t,   a_t+1, a_t+2, a_t+3]
```

这也是为什么每个 timestep 都放一个 Readout Token。

---

# 9. Goal Image

Goal Image 不只是简单作为另一个独立 Token 放进 Transformer。

实际实现的重要设计是 **early-goal fusion**：

$$
I_t^{current}
+
I^{goal}
$$

沿 channel 拼接：

$$
[H,W,3]+[H,W,3]
\rightarrow
[H,W,6]
$$

然后一起经过：

$$
\text{CNN Stem}\rightarrow\text{Patchify}
$$

因此 current / goal 的比较从视觉编码阶段就开始了。

---

# 10. 推理流程

推理时没有真实 Action。

首先：

```text
当前 Observation History
        +
Task
        ↓
Tokenizer
        ↓
Transformer
        ↓
e_R,t
```

然后 Action Head 从：

$$
x_K\sim\mathcal N(0,I)
$$

开始：

$$
x_K
\rightarrow
x_{K-1}
\rightarrow
...
\rightarrow
x_0
$$

最终：

$$
x_0
=
[a_t,a_{t+1},a_{t+2},a_{t+3}]
$$

### 一个非常重要的效率点

不是每次 diffusion denoising 都重新运行 Transformer。

而是：

```text
Observation
     ↓
Transformer     ← 只计算一次
     ↓
e_R,t
     ↓
Diffusion Head  ← 重复多次
     ↓
Action Chunk
```

所以 20-step diffusion ≠ 20 次完整 Transformer forward。

---

# 11. 执行：Receding Horizon Control

假设预测：

$$
[a_t,a_{t+1},a_{t+2},a_{t+3}]
$$

不一定全部执行。

例如只执行：

$$
a_t
$$

然后：

```text
执行 a_t
   ↓
获得新观测 o_t+1
   ↓
重新运行 Policy
   ↓
预测新的 Action Chunk
```

形成：

$$
\boxed{
观察
\rightarrow
预测一段
\rightarrow
执行前几步
\rightarrow
重新观察
\rightarrow
重新规划
}
$$

这就是 **Receding Horizon Control**。

---

# 12. 新机器人如何适配

Octo 的模块化设计最终服务于这个目标。

### 新传感器

```text
New Sensor
    ↓
New Adapter
    ↓
Token
    ↓
Transformer
```

### 新 Action Space

```text
Transformer
    ↓
Readout
    ↓
New Action Head
```

然后：

$$
\boxed{\text{用少量目标机器人数据进行 Full-model Finetuning}}
$$

微调仍然使用相同的 Diffusion Objective。

---

# 13. 最后只记这 6 个关键词

如果几个月以后重新看 Octo，只需要看到下面几个词，就应该能把整篇架构重新想起来：

### ① Modular Token Interface

不同机器人 / 传感器 → Token。

### ② Block-wise Causal Attention

同时间步多模态双向融合，不能看未来。

### ③ Readout Token

只读不写，形成当前时刻决策表示 \(e_{R,t}\)。

### ④ Diffusion Action Head

$$
(e_R,x_k,k)\rightarrow\hat\epsilon
$$

### ⑤ Action Chunk

一次生成未来若干连续动作。

### ⑥ Receding Horizon

执行前几步 → 新观测 → 再规划。

---

# 14. 脑内最终模型

```text
异构机器人数据
      ↓
Tokenizer / Adapter
      ↓
统一 Token Interface
      ↓
Block-wise Causal Transformer
      ↓
Readout R_t
      ↓
决策表示 e_R,t
      ↓
Diffusion Action Head
      ↓
Action Chunk
      ↓
执行前几步
      ↓
重新观测
      └──────────────↺
```

> **Octo = 用 Token 化解决异构输入，用 Transformer 学跨机器人共享表征，用 Readout 提取决策状态，用 Diffusion 生成连续 Action Chunk，再通过 Receding Horizon 构成闭环控制。**
