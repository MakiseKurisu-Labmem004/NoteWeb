---

title: "RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation"
description: "RDT-1B 面向双臂操作中的动作多模态和多机器人数据异构问题。核心理解：以机器人状态和带噪 Action Chunk 为 Transformer 主序列，将语言和图像作为条件通过交替 Cross-Attention 注入，并直接预测干净动作块。"
date: "2026-09-05"
venue: "arXiv / 2024"
authors: "Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, Jun Zhu"
paper: ""
code: ""
--------

# RDT-1B

![](./images/RDT-1B.png)

## 1. 一句话理解

RDT-1B 本质上是一个：

$$
\boxed{
\text{以 Robot State + Noisy Action 为主序列的 Diffusion Transformer}
}
$$

语言和图像**不直接与动作拼成一条长序列**，而是作为条件：

$$
\boxed{
\text{Language / Image}
\xrightarrow{\text{Cross-Attention}}
\text{Action Generation}
}
$$

最终学习：

$$
p(A_{t:t+63}\mid l,o_t)
$$

一次生成未来：

$$
\boxed{64\text{ 步 Action Chunk}}
$$

最核心的结构可以记成：

```text
Language ── T5 ───── Adapter ────────┐
                                     │
Images ─── SigLIP ── Adapter ────────┤
                                     │ Cross-Attention
                                     ▼
                         RDT Transformer × 28
                                     ▲
                                     │
Diffusion Step ──────────────────────┤
Control Frequency ──────────────────┤ Main Sequence
Robot State ─────────────────────────┤
Noisy Action Chunk × 64 ─────────────┘
                                     │
                                     ▼
                                MLP Decoder
                                     │
                                     ▼
                          Predicted Clean Action
                              64 × 128
```

---

## 2. 为什么需要 RDT

### 2.1 Action Multi-Modality

对于完全相同的：

$$
(o_t,l)
$$

可能存在多种都正确的动作轨迹：

* 左手抓、右手辅助；
* 右手抓、左手辅助；
* 从左侧绕过去；
* 从右侧绕过去；
* 先移动左手；
* 先移动右手。

因此动作并不适合建模成唯一的确定映射：

$$
(o_t,l)\rightarrow a_t
$$

而应该学习：

$$
\boxed{
p(A\mid o_t,l)
}
$$

RDT 因此采用 Diffusion Model 来表示复杂、多峰的动作分布。

---

### 2.2 Multi-Robot Heterogeneity

不同机器人的数据存在：

$$
\boxed{
\text{不同 DoF}
+
\text{不同 State 定义}
+
\text{不同 Action 定义}
+
\text{不同 Control Frequency}
}
$$

例如不同机器人中第 $i$ 个 action dimension 可能代表完全不同的物理量。

直接：

```text
Robot A Action
Robot B Action
Robot C Action
      ↓
直接 Padding / Concat
      ↓
Transformer
```

会产生严重的语义错位。

所以 RDT 定义：

$$
\boxed{
128D\ \text{Physically Interpretable Unified State/Action Space}
}
$$

重点不是简单变成 $128$ 维，而是：

> **128 个位置分别具有固定物理语义。**

不同机器人根据其实际物理定义，将 state/action 填入对应位置。

不存在的维度：

$$
\boxed{\text{Mask}}
$$

而不是随便赋予新的物理意义。

---

## 3. RDT 的整体架构

整个模型应该分成三部分理解：

$$
\boxed{
\text{Condition Encoder}
+
\text{Robot Main Sequence}
+
\text{Diffusion Transformer}
}
$$

总体：

```text
                         Language l
                             │
                             ▼
                       Frozen T5-XXL
                             │
                         4096D tokens
                             │
                             ▼
                      Language Adapter
                             │
                         2048D tokens
                             │
                             ├──────────────┐
                             │              │
Images                       │              │
  │                          │              │
  ▼                          │              │
Frozen SigLIP                │              │
  │                          │              │
1152D Patch Tokens           │              │
  │                          │              │
  ▼                          │              │
Image Adapter                │              │
  │                          │              │
2048D Tokens                 │              │
  └──────────────────────────┤
                             │
                             │ Cross-Attention
                             ▼
                   ┌───────────────────────┐
                   │   RDT Transformer     │
                   │      × 28 Layers      │
                   └──────────▲────────────┘
                              │
                              │ Main Sequence
                              │
                   [k][freq][state][A_k ×64]
                              │
                              ▼
                          RMSNorm
                              │
                              ▼
                         MLP Decoder
                              │
                              ▼
                    Predicted Clean Action
                           Â_0 × 64
```

这里最重要的是区分：

$$
\boxed{\text{Main Sequence}}
$$

和：

$$
\boxed{\text{Condition Sequence}}
$$

---

## 4. 128D Unified Physical Space

设某机器人 $r$ 原始 proprioception 为：

$$
s_t^{(r)}
\in
\mathbb R^{d_s^{(r)}}
$$

原始 action 为：

$$
a_t^{(r)}
\in
\mathbb R^{d_a^{(r)}}
$$

不同机器人：

$$
d_s^{(r_1)}
\neq
d_s^{(r_2)}
$$

$$
d_a^{(r_1)}
\neq
d_a^{(r_2)}
$$

RDT 首先通过机器人对应的物理映射：

$$
U_r
$$

将它们放入统一空间：

$$
z_t
=
U_r(s_t^{(r)})
\in
\mathbb R^{128}
$$

$$
a_t
=
U_r(a_t^{(r)})
\in
\mathbb R^{128}
$$

同时产生 Valid Dimension Mask：

$$
m_z\in\{0,1\}^{128}
$$

$$
m_a\in\{0,1\}^{128}
$$

其中：

$$
m_i=1
$$

表示：

> 当前机器人确实拥有这个物理维度。

而：

$$
m_i=0
$$

表示：

> 当前机器人不存在这个维度。

因此不同机器人最终都变成：

$$
\boxed{
State\in\mathbb R^{128}
}
$$

$$
\boxed{
Action\in\mathbb R^{128}
}
$$

这是多机器人联合训练的基础。

---

## 5. RDT 的输入到底有哪些

一个训练样本可以抽象为：

$$
\boxed{
(l,\ I,\ z_t,\ c,\ A_0)
}
$$

其中：

$$
l
$$

是语言任务；

$$
I
$$

是多摄像头历史图像；

$$
z_t\in\mathbb R^{128}
$$

是机器人当前 proprioception；

$$
c
$$

是 control frequency；

$$
A_0
=
[a_t,\ldots,a_{t+63}]
\in
\mathbb R^{64\times128}
$$

是真实未来 Action Chunk。

Diffusion 训练以后还会得到：

$$
A_k
$$

以及 diffusion timestep：

$$
k
$$

所以实际进入网络的信息是：

$$
\boxed{
(l,I,z_t,c,A_k,k)
}
$$

---

## 6. Language Encoding

输入自然语言：

$$
l
$$

先 tokenize：

$$
l
\rightarrow
[w_1,\ldots,w_{N_L}]
$$

再进入冻结的 T5-XXL：

$$
[w_1,\ldots,w_{N_L}]
\xrightarrow{\text{T5-XXL}}
H_L
$$

得到：

$$
H_L
\in
\mathbb R^{N_L\times4096}
$$

然后通过 Language Adapter：

$$
\mathbb R^{4096}
\rightarrow
\mathbb R^{2048}
$$

得到：

$$
\boxed{
C_L
=
[L_1,\ldots,L_{N_L}]
\in
\mathbb R^{N_L\times2048}
}
$$

它不会 concat 到 Main Sequence。

而是保存为：

$$
\boxed{\text{Language Condition}}
$$

在对应 Transformer Layer 的 Cross-Attention 中作为：

$$
K,V
$$

使用。

---

## 7. Image Encoding

RDT 最多可以使用：

$$
2\text{ 个时间时刻}
\times
3\text{ 个摄像头}
$$

例如：

```text
ext(t-1)
right_wrist(t-1)
left_wrist(t-1)

ext(t)
right_wrist(t)
left_wrist(t)
```

设：

$$
T_o=2
$$

摄像头数量：

$$
N_c=3
$$

则共有：

$$
T_oN_c=6
$$

张图像。

设每张原始图像：

$$
I_{j}
\in
\mathbb R^{H_I\times W_I\times3}
$$

冻结的 SigLIP 将其编码为 patch tokens：

$$
I_j
\xrightarrow{\text{SigLIP}}
V_j
$$

设每张图产生 $N_P$ 个 patch token：

$$
V_j
\in
\mathbb R^{N_P\times1152}
$$

六张图组合：

$$
V
=
[V_1;V_2;\ldots;V_6]
$$

因此：

$$
V
\in
\mathbb R^{6N_P\times1152}
$$

然后 Image Adapter：

$$
\mathbb R^{1152}
\rightarrow
\mathbb R^{2048}
$$

得到：

$$
\boxed{
C_I
\in
\mathbb R^{6N_P\times2048}
}
$$

它同样不会进入 Main Sequence。

而保存为：

$$
\boxed{\text{Image Condition}}
$$

供 Cross-Attention 使用。

---

## 8. Low-Dimensional Inputs

RDT 中以下数据属于低维机器人数据：

$$
\boxed{
\text{State}
+
\text{Action}
+
\text{Control Frequency}
+
\text{Diffusion Timestep}
}
$$

### 8.1 Robot State

统一后的 state：

$$
z_t
\in
\mathbb R^{128}
$$

经过 State Adapter / MLP：

$$
z_t
\xrightarrow{\text{MLP}}
T_z
$$

得到：

$$
T_z
\in
\mathbb R^{2048}
$$

也就是：

$$
\boxed{\text{一个 State Token}}
$$

---

### 8.2 Action

每一个 noisy action：

$$
a_i^k
\in
\mathbb R^{128}
$$

经过 Action / State Adapter：

$$
a_i^k
\xrightarrow{\text{MLP}}
T_{a_i}
$$

得到：

$$
T_{a_i}
\in
\mathbb R^{2048}
$$

一共 $64$ 个 action：

$$
A_k
\in
\mathbb R^{64\times128}
$$

变成：

$$
T_A
\in
\mathbb R^{64\times2048}
$$

---

### 8.3 Diffusion Timestep

扩散时间步：

$$
k
$$

先做 sinusoidal / Fourier-style embedding：

$$
k
\xrightarrow{\text{SinCos}}
e_k
$$

再经过 MLP：

$$
e_k
\xrightarrow{\text{MLP}}
T_k
$$

最终：

$$
T_k\in\mathbb R^{2048}
$$

是：

$$
\boxed{\text{一个 Diffusion Timestep Token}}
$$

---

### 8.4 Control Frequency

控制频率：

$$
c
$$

例如：

$$
5Hz,\ 10Hz,\ 20Hz,\ 50Hz
$$

同样：

$$
c
\xrightarrow{\text{SinCos / Fourier}}
e_c
\xrightarrow{\text{MLP}}
T_c
$$

得到：

$$
T_c
\in
\mathbb R^{2048}
$$

即：

$$
\boxed{\text{一个 Frequency Token}}
$$

为什么必须告诉模型 frequency？

因为同样：

$$
\Delta x=0.01
$$

如果：

$$
c=5Hz
$$

和：

$$
c=50Hz
$$

其实际运动速度完全不同。

因此 frequency 本质上告诉模型：

> **如何按照时间尺度解释 Action。**

---

## 9. Main Sequence 是怎么构造的

完成上述编码之后：

$$
T_k\in\mathbb R^{1\times2048}
$$

$$
T_c\in\mathbb R^{1\times2048}
$$

$$
T_z\in\mathbb R^{1\times2048}
$$

$$
T_A\in\mathbb R^{64\times2048}
$$

按顺序 concat：

$$
X_0
=
[
T_k;
T_c;
T_z;
T_{a_1};
\ldots;
T_{a_{64}}
]
$$

得到：

$$
\boxed{
X_0
\in
\mathbb R^{67\times2048}
}
$$

因为：

$$
1+1+1+64=67
$$

再加入 multimodal position embedding：

$$
X_0
\leftarrow
X_0+P
$$

shape 不改变：

$$
67\times2048
$$

这就是整个 Transformer 真正进行 Self-Attention 的：

$$
\boxed{\text{Main Sequence}}
$$

因此 RDT **不是**：

```text
Language
Image
State
Action
  ↓
全部 concat
  ↓
Self-Attention
```

而是：

```text
[k][freq][state][action ×64]
           │
           ▼
      Self-Attention
           │
           │ Q
           ▼
      Cross-Attention
           ▲
           │ K,V
      Language / Image
```

---

## 10. 一个 RDT Transformer Block

RDT-1B 有：

$$
\boxed{28\text{ Layers}}
$$

hidden dimension：

$$
D=2048
$$

attention heads：

$$
h=32
$$

因此每个 head dimension：

$$
d_h
=
\frac{2048}{32}
=
64
$$

每一层结构：

$$
\boxed{
\text{RMSNorm}
\rightarrow
\text{Self-Attention}
\rightarrow+
\text{RMSNorm}
\rightarrow
\text{Cross-Attention}
\rightarrow+
\text{RMSNorm}
\rightarrow
\text{FFN}
\rightarrow+
}
$$

也就是：

```text
X_l
 │
 ▼
RMSNorm
 │
 ▼
Self-Attention
 │
 + Residual
 │
 ▼
RMSNorm
 │
 ▼
Cross-Attention
 │
 + Residual
 │
 ▼
RMSNorm
 │
 ▼
FFN
 │
 + Residual
 │
 ▼
X_{l+1}
```

---

## 11. Self-Attention 到底在做什么

输入：

$$
X
\in
\mathbb R^{67\times2048}
$$

先：

$$
\bar X
=
\operatorname{RMSNorm}(X)
$$

shape 不变：

$$
67\times2048
$$

然后生成：

$$
Q=\bar XW_Q
$$

$$
K=\bar XW_K
$$

$$
V=\bar XW_V
$$

所以：

$$
\boxed{
Q,K,V
\text{ 全部来自 Main Sequence}
}
$$

拆成 32 个 Attention Heads：

$$
Q,K,V
\in
\mathbb R^{32\times67\times64}
$$

RDT 还对：

$$
Q,K
$$

做 QK-Norm：

$$
Q
\leftarrow
\operatorname{RMSNorm}(Q)
$$

$$
K
\leftarrow
\operatorname{RMSNorm}(K)
$$

然后：

$$
H_{\text{self}}
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{64}}
\right)V
$$

重新合并 heads：

$$
H_{\text{self}}
\in
\mathbb R^{67\times2048}
$$

Residual：

$$
X'
=
X+H_{\text{self}}
$$

shape：

$$
\boxed{
67\times2048
}
$$

---

## 12. Action Token 在 Self-Attention 中看什么

Main Sequence：

$$
[k,c,z,a_1,\ldots,a_{64}]
$$

所以任意 action token：

$$
a_i
$$

可以 Attend：

$$
\boxed{
[k,c,z,a_1,\ldots,a_{64}]
}
$$

也就是说：

$$
a_i
\leftrightarrow
a_j
$$

并没有 causal mask 限制。

因此所有未来 action：

$$
a_1,\ldots,a_{64}
$$

可以：

$$
\boxed{\text{双向交换信息}}
$$

这意味着 RDT 不是：

$$
a_1
\rightarrow
a_2
\rightarrow
a_3
\rightarrow\cdots
$$

这种 autoregressive generation。

而是：

$$
\boxed{
\text{一次联合建模整段未来轨迹}
}
$$

比如：

$$
a_{20}
$$

可以影响：

$$
a_5
$$

从而让整个 Action Chunk 在空间和时间上保持协调。

---

## 13. Cross-Attention 到底在做什么

Self-Attention 之后：

$$
X'
\in
\mathbb R^{67\times2048}
$$

先：

$$
\bar X'
=
\operatorname{RMSNorm}(X')
$$

然后 Cross-Attention。

关键区别：

$$
\boxed{
Q=W_Q\bar X'
}
$$

即：

$$
Q=\text{Main Sequence}
$$

但是：

$$
\boxed{
K,V=W_{KV}C
}
$$

其中：

$$
C=C_L
$$

或者：

$$
C=C_I
$$

所以：

$$
\boxed{
Q=\text{Robot State / Action}
}
$$

$$
\boxed{
K,V=\text{Language / Image}
}
$$

例如对于 Language Layer：

$$
Q
\in
\mathbb R^{32\times67\times64}
$$

$$
K_L,V_L
\in
\mathbb R^{32\times N_L\times64}
$$

Attention：

$$
H_{\text{cross}}
=
\operatorname{softmax}
\left(
\frac{QK_L^\top}{\sqrt{64}}
\right)V_L
$$

最后：

$$
H_{\text{cross}}
\in
\mathbb R^{67\times2048}
$$

Residual：

$$
X''
=
X'
+
H_{\text{cross}}
$$

shape 不改变：

$$
\boxed{
67\times2048
}
$$

可以把 Cross-Attention 理解成：

> 当前 state/action token 主动去询问：
>
> **“生成我这个动作时，需要从语言或者图像中读取什么信息？”**

---

## 14. Language / Image 为什么交替 Cross-Attention

如果直接：

$$
C=[C_L;C_I]
$$

通常：

$$
N_{\text{image}}
\gg
N_{\text{language}}
$$

例如图像可能有几千 token：

$$
N_I\sim10^3
$$

而语言只有：

$$
N_L\sim10^1
$$

于是可能发生：

$$
\boxed{
\text{Vision Dominates Language}
}
$$

因此 RDT 不在同一个 Cross-Attention 中简单把两者拼起来，而采用交替注入：

```text
Layer 1  → Language Cross-Attention

Layer 2  → Image Cross-Attention

Layer 3  → Language Cross-Attention

Layer 4  → Image Cross-Attention

...

Layer 27 → Language Cross-Attention

Layer 28 → Image Cross-Attention
```

抽象：

$$
\boxed{
C^{(l)}
=
\begin{cases}
C_L,&l\text{ 为 Language Layer}\\
C_I,&l\text{ 为 Image Layer}
\end{cases}
}
$$

因此：

$$
Language
\rightarrow
Image
\rightarrow
Language
\rightarrow
Image
\rightarrow\cdots
$$

是 RDT 最有辨识度的架构设计之一。

---

## 15. FFN 做什么

Cross-Attention 后：

$$
X''
\in
\mathbb R^{67\times2048}
$$

先：

$$
\bar X''
=
\operatorname{RMSNorm}(X'')
$$

然后：

$$
H_{\text{FFN}}
=
\operatorname{FFN}(\bar X'')
$$

再 residual：

$$
X_{\text{out}}
=
X''
+
H_{\text{FFN}}
$$

所以 FFN 不负责：

> token 与 token 之间交换信息。

Attention 已经完成信息交互。

FFN 负责：

$$
\boxed{
\text{对每个 Token 的 Feature 做非线性变换}
}
$$

输入输出 shape 都仍然：

$$
67\times2048
$$

---

## 16. Independent Random Masking

多模态数据还有第二种 heterogeneity：

$$
\boxed{\text{Information Heterogeneity}}
$$

例如 exterior camera 通常信息量非常大。

如果所有训练样本都完整提供 exterior camera，模型可能学习 shortcut：

> 不需要认真理解 wrist camera、state 或 language，只看 exterior camera 就可以。

RDT 因此在训练时独立 Mask 不同条件。

例如：

```text
Language        ✓
Exterior Cam    ×
Right Wrist     ✓
Left Wrist      ✓
State           ✓
Frequency       ×
```

另一个样本可能：

```text
Language        ×
Exterior Cam    ✓
Right Wrist     ×
Left Wrist      ✓
State           ✓
Frequency       ✓
```

这样迫使模型学习：

$$
\boxed{
\text{Multiple Redundant Cues}
}
$$

核心作用：

$$
\boxed{
\text{Prevent Shortcut Learning}
}
$$

---

## 17. Diffusion：RDT 直接预测 Clean Action

真实动作块：

$$
A_0
=
[a_t,\ldots,a_{t+63}]
$$

其中：

$$
A_0
\in
\mathbb R^{64\times128}
$$

随机采样 diffusion timestep：

$$
k
\sim
\operatorname{Uniform}\{1,\ldots,K\}
$$

采样高斯噪声：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

其中：

$$
\epsilon
\in
\mathbb R^{64\times128}
$$

前向加噪：

$$
\boxed{
A_k
=
\sqrt{\bar\alpha_k}A_0
+
\sqrt{1-\bar\alpha_k}\epsilon
}
$$

因此：

$$
A_k
\in
\mathbb R^{64\times128}
$$

shape 没有改变。

只是：

$$
\boxed{
\text{Clean Action}
\rightarrow
\text{Noisy Action}
}
$$

RDT 接收：

$$
(l,I,z_t,c,A_k,k)
$$

预测：

$$
\boxed{
\hat A_0
=
f_\theta(l,I,z_t,c,A_k,k)
}
$$

而不是预测：

$$
\epsilon
$$

所以训练目标：

$$
\boxed{
\mathcal L
=
\|A_0-\hat A_0\|_2^2
}
$$

最值得和 Octo 区分：

$$
\boxed{
Octo:
A_k
\rightarrow
\hat\epsilon
}
$$

而：

$$
\boxed{
RDT:
A_k
\rightarrow
\hat A_0
}
$$

---

## 18. 为什么说 RDT Transformer 本身就是 Denoiser

Octo 可以抽象成：

```text
Observation
    │
    ▼
Transformer
    │
    ▼
Condition Representation
    │
    ▼
Small Diffusion Head
    │
    ▼
Action Denoising
```

也就是说 Transformer 主要：

$$
\boxed{\text{Encode Observation}}
$$

真正 iterative diffusion 主要发生在小 Action Head 中。

RDT 则是：

```text
Noisy Action A_k
      │
      ▼
RDT Transformer
      │
      ▼
Predicted Clean Action Â_0
```

因为：

$$
A_k
$$

本身就是 Transformer Main Sequence 的一部分。

因此：

$$
\boxed{
\text{RDT Transformer 本身就是 Diffusion Denoiser}
}
$$

---

## 19. 完整数据符号流程：训练

下面用一个 batch 的数据完整走一次。

设：

$$
B=\text{Batch Size}
$$

$$
H_A=64
$$

$$
D_R=128
$$

$$
D=2048
$$

$$
N_H=32
$$

---

### 19.1 Step 0：原始机器人轨迹

对于 batch 中第 $b$ 个样本，原始数据：

$$
\mathcal D^{(b)}
=
(
l^{(b)},
I^{(b)},
s_t^{(b)},
c^{(b)},
A_{\text{raw}}^{(b)}
)
$$

其中：

语言：

$$
l^{(b)}
=
\text{String}
$$

图像：

$$
I^{(b)}
=
\{
I_{t-1}^{ext},
I_{t-1}^{rw},
I_{t-1}^{lw},
I_t^{ext},
I_t^{rw},
I_t^{lw}
\}
$$

原始机器人状态：

$$
s_t^{(b)}
\in
\mathbb R^{d_s^{(r)}}
$$

原始未来动作：

$$
A_{\text{raw}}^{(b)}
\in
\mathbb R^{64\times d_a^{(r)}}
$$

控制频率：

$$
c^{(b)}
\in
\mathbb R
$$

这里不同机器人：

$$
d_s^{(r)}
$$

和：

$$
d_a^{(r)}
$$

可以不同。

---

### 19.2 Step 1：映射到统一 128D 物理空间

根据机器人类型 $r$ 的物理映射：

$$
U_r
$$

状态：

$$
s_t^{(r)}
\in
\mathbb R^{d_s^{(r)}}
$$

变成：

$$
\boxed{
z_t
=
U_r(s_t^{(r)})
\in
\mathbb R^{128}
}
$$

动作：

$$
A_{\text{raw}}
\in
\mathbb R^{64\times d_a^{(r)}}
$$

变成：

$$
\boxed{
A_0
=
U_r(A_{\text{raw}})
\in
\mathbb R^{64\times128}
}
$$

同时产生：

$$
M_z\in\{0,1\}^{128}
$$

以及：

$$
M_A
\in
\{0,1\}^{64\times128}
$$

表示哪些统一物理维度在当前机器人中有效。

batch 后：

$$
Z
\in
\mathbb R^{B\times128}
$$

$$
A_0
\in
\mathbb R^{B\times64\times128}
$$

---

### 19.3 Step 2：Independent Random Masking

训练时对：

$$
Language
$$

$$
Camera_i
$$

$$
State
$$

$$
Control\ Frequency
$$

等条件进行独立随机 Mask。

形式上可以写成：

$$
\tilde C_j
=
m_jC_j
$$

其中：

$$
m_j\in\{0,1\}
$$

因此输入从：

$$
(l,I,z_t,c)
$$

变成：

$$
(\tilde l,\tilde I,\tilde z_t,\tilde c)
$$

数据 shape 基本不改变，只改变：

$$
\boxed{\text{哪些条件有效}}
$$

---

### 19.4 Step 3：构造真实 Action Chunk

从 trajectory 中取未来 $64$ 步：

$$
A_0
=
[a_t,a_{t+1},\ldots,a_{t+63}]
$$

每个 action 已经在统一空间：

$$
a_i\in\mathbb R^{128}
$$

所以：

$$
\boxed{
A_0
\in
\mathbb R^{B\times64\times128}
}
$$

这是监督标签。

---

### 19.5 Step 4：采样 Diffusion Timestep

每个 sample 随机：

$$
k
\sim
U\{1,\ldots,K\}
$$

batch：

$$
k
\in
\mathbb R^{B}
$$

---

### 19.6 Step 5：采样噪声

生成：

$$
\epsilon
\sim
\mathcal N(0,I)
$$

shape：

$$
\boxed{
\epsilon
\in
\mathbb R^{B\times64\times128}
}
$$

和 $A_0$ 完全相同。

---

### 19.7 Step 6：Action 加噪

根据 $k$：

$$
A_k
=
\sqrt{\bar\alpha_k}A_0
+
\sqrt{1-\bar\alpha_k}\epsilon
$$

因此：

$$
A_0:
B\times64\times128
$$

经过 diffusion：

$$
\Downarrow
$$

$$
A_k:
B\times64\times128
$$

shape 不变。

改变的是：

$$
\boxed{
\text{Data Distribution}
}
$$

从干净专家动作变成带噪动作。

---

### 19.8 Step 7：编码 Diffusion Timestep

输入：

$$
k
\in
\mathbb R^B
$$

经过：

$$
k
\xrightarrow{\text{Sinusoidal Embedding}}
E_k
\xrightarrow{\text{MLP}}
T_k
$$

得到：

$$
\boxed{
T_k
\in
\mathbb R^{B\times1\times2048}
}
$$

---

### 19.9 Step 8：编码 Control Frequency

输入：

$$
c
\in
\mathbb R^B
$$

经过：

$$
c
\xrightarrow{\text{Sinusoidal/Fourier Embedding}}
E_c
\xrightarrow{\text{MLP}}
T_c
$$

得到：

$$
\boxed{
T_c
\in
\mathbb R^{B\times1\times2048}
}
$$

---

### 19.10 Step 9：编码 Robot State

输入：

$$
Z
\in
\mathbb R^{B\times128}
$$

经过 State Adapter：

$$
Z
\xrightarrow{\text{MLP}}
T_z
$$

得到：

$$
\boxed{
T_z
\in
\mathbb R^{B\times1\times2048}
}
$$

---

### 19.11 Step 10：编码 Noisy Action

输入：

$$
A_k
\in
\mathbb R^{B\times64\times128}
$$

对每个 future action：

$$
a_i^k
\in
\mathbb R^{128}
$$

应用 MLP：

$$
a_i^k
\xrightarrow{\text{MLP}}
T_{a_i}
\in
\mathbb R^{2048}
$$

于是：

$$
\boxed{
T_A
\in
\mathbb R^{B\times64\times2048}
}
$$

格式变化：

$$
B\times64\times128
$$

$$
\Downarrow
$$

$$
B\times64\times2048
$$

---

### 19.12 Step 11：构造 Main Sequence

将：

$$
T_k
$$

$$
T_c
$$

$$
T_z
$$

$$
T_A
$$

沿 sequence dimension concat：

$$
X_0
=
[
T_k;
T_c;
T_z;
T_A
]
$$

shape：

$$
B\times
(1+1+1+64)
\times2048
$$

即：

$$
\boxed{
X_0
\in
\mathbb R^{B\times67\times2048}
}
$$

然后：

$$
X_0
\leftarrow
X_0+P
$$

position embedding 后：

$$
\boxed{
B\times67\times2048
}
$$

---

### 19.13 Step 12：编码 Language Condition

语言 tokenizer 后：

$$
W
\in
\mathbb N^{B\times N_L}
$$

进入冻结 T5：

$$
W
\xrightarrow{\text{T5}}
H_L
$$

$$
H_L
\in
\mathbb R^{B\times N_L\times4096}
$$

Language Adapter：

$$
4096
\rightarrow
2048
$$

得到：

$$
\boxed{
C_L
\in
\mathbb R^{B\times N_L\times2048}
}
$$

---

### 19.14 Step 13：编码 Image Condition

假设：

$$
T_o=2
$$

$$
N_c=3
$$

总共：

$$
N_{img}=6
$$

张图。

原始输入可以抽象为：

$$
I
\in
\mathbb R^{
B\times6\times H_I\times W_I\times3
}
$$

每张图经过 SigLIP：

$$
I_j
\rightarrow
V_j
\in
\mathbb R^{N_P\times1152}
$$

全部图像展开：

$$
V
\in
\mathbb R^{
B\times6N_P\times1152
}
$$

然后 Image Adapter：

$$
1152
\rightarrow
2048
$$

得到：

$$
\boxed{
C_I
\in
\mathbb R^{B\times6N_P\times2048}
}
$$

---

### 19.15 Step 14：进入第 1 个 RDT Block

当前：

$$
X_0
\in
\mathbb R^{B\times67\times2048}
$$

首先：

$$
\bar X_0
=
\operatorname{RMSNorm}(X_0)
$$

shape：

$$
B\times67\times2048
$$

---

### 19.16 Step 15：Self-Attention

投影：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

拆成 $32$ heads：

$$
Q,K,V
\in
\mathbb R^{
B\times32\times67\times64
}
$$

QK-Norm：

$$
Q\leftarrow RMSNorm(Q)
$$

$$
K\leftarrow RMSNorm(K)
$$

计算：

$$
S
=
\frac{QK^\top}{\sqrt{64}}
$$

其中：

$$
S
\in
\mathbb R^{
B\times32\times67\times67
}
$$

Softmax：

$$
P
=
\operatorname{softmax}(S)
$$

再：

$$
H
=
PV
$$

得到：

$$
H
\in
\mathbb R^{
B\times32\times67\times64
}
$$

合并 heads：

$$
H_{\text{self}}
\in
\mathbb R^{B\times67\times2048}
$$

Residual：

$$
X'
=
X_0+H_{\text{self}}
$$

最终：

$$
\boxed{
X'
\in
\mathbb R^{B\times67\times2048}
}
$$

此时每个 Action Token 已经融合：

$$
[k,c,z,a_1,\ldots,a_{64}]
$$

的信息。

---

### 19.17 Step 16：Language Cross-Attention

如果当前 Block 使用 Language：

$$
C=C_L
$$

Main Sequence 产生：

$$
Q
\in
\mathbb R^{
B\times32\times67\times64
}
$$

Language Condition 产生：

$$
K_L,V_L
\in
\mathbb R^{
B\times32\times N_L\times64
}
$$

Attention Score：

$$
S_L
=
QK_L^\top
$$

shape：

$$
\boxed{
B\times32\times67\times N_L
}
$$

意味着：

> 67 个 Main Sequence token 分别查询语言中的 $N_L$ 个 token。

得到：

$$
H_{\text{cross}}
\in
\mathbb R^{B\times67\times2048}
$$

然后：

$$
X''
=
X'
+
H_{\text{cross}}
$$

---

### 19.18 Step 17：FFN

先：

$$
\bar X''
=
RMSNorm(X'')
$$

然后：

$$
H_{\text{FFN}}
=
FFN(\bar X'')
$$

最后：

$$
X_1
=
X''
+
H_{\text{FFN}}
$$

shape 始终：

$$
\boxed{
B\times67\times2048
}
$$

---

### 19.19 Step 18：继续 28 层

因此：

```text
X_0

↓ Block 1
Self-Attention
Language Cross-Attention
FFN

↓ Block 2
Self-Attention
Image Cross-Attention
FFN

↓ Block 3
Self-Attention
Language Cross-Attention
FFN

↓ Block 4
Self-Attention
Image Cross-Attention
FFN

...

↓ Block 28
Self-Attention
Image Cross-Attention
FFN
```

整个过程中：

$$
X_l
\in
\mathbb R^{B\times67\times2048}
$$

sequence length 和 hidden dimension 都不改变。

改变的是：

$$
\boxed{
\text{每个 Token 内部包含的信息}
}
$$

随着层数增加，Action Token 逐渐融合：

$$
\text{Diffusion Time}
+
\text{Frequency}
+
\text{State}
+
\text{Other Actions}
+
\text{Language}
+
\text{Vision}
$$

---

### 19.20 Step 19：最终 RMSNorm

28 层之后：

$$
X_{28}
\in
\mathbb R^{B\times67\times2048}
$$

经过：

$$
\tilde X
=
RMSNorm(X_{28})
$$

shape：

$$
B\times67\times2048
$$

---

### 19.21 Step 20：MLP Decoder

每一个 token feature：

$$
x_i
\in
\mathbb R^{2048}
$$

经过 Decoder：

$$
x_i
\xrightarrow{\text{MLP Decoder}}
y_i
$$

映射回统一 action dimension。

只需要最后 $64$ 个 Action Token 对应的输出：

$$
[
y_4,\ldots,y_{67}
]
$$

得到：

$$
\boxed{
\hat A_0
\in
\mathbb R^{B\times64\times128}
}
$$

所以：

$$
B\times67\times2048
$$

经过 Decoder 并选 Action Tokens：

$$
\Downarrow
$$

$$
\boxed{
B\times64\times128
}
$$

---

### 19.22 Step 21：计算 Loss

真实动作：

$$
A_0
\in
\mathbb R^{B\times64\times128}
$$

预测：

$$
\hat A_0
\in
\mathbb R^{B\times64\times128}
$$

计算：

$$
\boxed{
\mathcal L
=
\|A_0-\hat A_0\|_2^2
}
$$

对不同机器人不存在的 unified dimensions，通过有效维度 mask 保持其无效语义。

最后：

```text
Loss
 ↓
Backpropagation
 ↓
RDT Transformer
Adapters
State / Action Encoder
MLP Decoder
```

而：

```text
T5-XXL   Frozen
SigLIP   Frozen
```

---

## 20. 训练流程压缩成一条数据链

完整训练流程：

```text
Raw Multi-Robot Trajectory
(l, images, raw_state, raw_actions, frequency)

        │
        ▼

Robot-specific physical mapping

raw_state
    ↓
state ∈ R^128

raw_action
    ↓
action ∈ R^128

        │
        ▼

Future Action Chunk

A_0
shape:
[B, 64, 128]

        │
        ├──────────────────────────────┐
        │                              │
        ▼                              │
sample k                             Label
        │                              │
sample ε                              │
        │                              │
        ▼                              │

A_k = √ᾱ_k A_0 + √(1-ᾱ_k) ε          │
                                         │
shape: [B,64,128]                       │
        │                               │
        ▼                               │

Action MLP                              │
        │                               │
        ▼                               │
[B,64,2048]                            │
                                        │
State [B,128]                           │
  ↓ State MLP                           │
[B,1,2048]                             │
                                        │
k                                      │
 ↓ SinCos + MLP                         │
[B,1,2048]                             │
                                        │
frequency                              │
 ↓ SinCos + MLP                         │
[B,1,2048]                             │
                                        │
        └────────── concat ─────────────┘
                    │
                    ▼

Main Sequence
[k][freq][state][a_1]...[a_64]

shape:
[B,67,2048]

                    │
                    ▼

              RDT Block ×28
                    │
     ┌──────────────┴───────────────┐
     │                              │
     ▼                              ▼
Language                         Images
  ↓                               ↓
T5                              SigLIP
  ↓                               ↓
[B,N_L,4096]                 [B,6N_P,1152]
  ↓                               ↓
Adapter                          Adapter
  ↓                               ↓
[B,N_L,2048]                 [B,6N_P,2048]
     │                              │
     └──── alternating Cross-Attn ──┘

                    │
                    ▼

             [B,67,2048]

                    │
                    ▼

               RMSNorm

                    │
                    ▼

              MLP Decoder

                    │
                    ▼

take Action Tokens only

                    │
                    ▼

Predicted Clean Action

Â_0
shape:
[B,64,128]

                    │
                    ▼

L = ||A_0 - Â_0||²
```

---

## 21. 完整数据符号流程：推理

训练结束以后，真实未来动作：

$$
A_0
$$

当然不存在。

因此推理必须从随机噪声开始。

---

### 21.1 Step 1：获得当前 Observation

机器人当前时刻获得：

$$
o_t
=
(I_t,z_t)
$$

并拥有任务：

$$
l
$$

以及：

$$
c
$$

所以条件是：

$$
(l,I,z_t,c)
$$

---

### 21.2 Step 2：编码固定条件

Language：

$$
l
\rightarrow
T5
\rightarrow
Adapter
\rightarrow
C_L
$$

$$
C_L
\in
\mathbb R^{B\times N_L\times2048}
$$

Image：

$$
I
\rightarrow
SigLIP
\rightarrow
Adapter
\rightarrow
C_I
$$

$$
C_I
\in
\mathbb R^{B\times6N_P\times2048}
$$

State：

$$
z_t
\rightarrow
MLP
\rightarrow
T_z
$$

$$
T_z
\in
\mathbb R^{B\times1\times2048}
$$

Frequency：

$$
c
\rightarrow
Embedding+MLP
\rightarrow
T_c
$$

$$
T_c
\in
\mathbb R^{B\times1\times2048}
$$

这些条件描述当前机器人和任务。

---

### 21.3 Step 3：初始化纯高斯 Action

没有真实动作，所以直接：

$$
\boxed{
A_K
\sim
\mathcal N(0,I)
}
$$

shape：

$$
\boxed{
A_K
\in
\mathbb R^{B\times64\times128}
}
$$

此时 $A_K$ 基本没有有意义的动作结构。

---

### 21.4 Step 4：构造第一个 Main Sequence

当前 diffusion timestep：

$$
k=K
$$

编码：

$$
T_K
\in
\mathbb R^{B\times1\times2048}
$$

当前 noisy action：

$$
A_K
\in
\mathbb R^{B\times64\times128}
$$

经过 Action MLP：

$$
T_{A_K}
\in
\mathbb R^{B\times64\times2048}
$$

构造：

$$
X_K
=
[
T_K;
T_c;
T_z;
T_{A_K}
]
$$

得到：

$$
\boxed{
X_K
\in
\mathbb R^{B\times67\times2048}
}
$$

---

### 21.5 Step 5：完整运行 RDT × 28

经过：

$$
RDT_\theta
(
X_K,
C_L,
C_I
)
$$

得到：

$$
X'_K
\in
\mathbb R^{B\times67\times2048}
$$

Decoder：

$$
X'_K
\rightarrow
\hat A_0^{(K)}
$$

其中：

$$
\boxed{
\hat A_0^{(K)}
\in
\mathbb R^{B\times64\times128}
}
$$

这里模型的意思是：

> 根据当前噪声 $A_K$ 和所有条件，我认为最终干净动作应该接近 $\hat A_0^{(K)}$。

---

### 21.6 Step 6：Reverse Diffusion 更新

根据 DDPM posterior：

$$
\boxed{
A_{K-1}
=
C_1(K)\hat A_0^{(K)}
+
C_2(K)A_K
+
\sigma_Kz
}
$$

其中：

$$
z\sim\mathcal N(0,I)
$$

shape：

$$
A_K:
B\times64\times128
$$

$$
\hat A_0^{(K)}:
B\times64\times128
$$

得到：

$$
\boxed{
A_{K-1}:
B\times64\times128
}
$$

shape 不改变。

但：

$$
\boxed{
A_{K-1}
\text{ 比 }
A_K
\text{ 更接近有效动作}
}
$$

---

### 21.7 Step 7：重新运行整个 Transformer

现在：

$$
k=K-1
$$

$$
A_k=A_{K-1}
$$

重新编码：

$$
A_{K-1}
\rightarrow
T_{A_{K-1}}
$$

然后重新构造：

$$
X_{K-1}
=
[
T_{K-1};
T_c;
T_z;
T_{A_{K-1}}
]
$$

再次：

$$
X_{K-1}
\rightarrow
RDT\times28
\rightarrow
\hat A_0^{(K-1)}
$$

再：

$$
A_{K-2}
=
C_1\hat A_0^{(K-1)}
+
C_2A_{K-1}
+
\sigma z
$$

---

### 21.8 Step 8：不断迭代

于是：

$$
A_K
\xrightarrow{RDT}
\hat A_0^{(K)}
\rightarrow
A_{K-1}
$$

$$
A_{K-1}
\xrightarrow{RDT}
\hat A_0^{(K-1)}
\rightarrow
A_{K-2}
$$

$$
\vdots
$$

$$
A_1
\xrightarrow{RDT}
\hat A_0^{(1)}
\rightarrow
A_0
$$

整体：

$$
\boxed{
A_K
\rightarrow
A_{K-1}
\rightarrow
A_{K-2}
\rightarrow
\cdots
\rightarrow
A_0
}
$$

最终：

$$
A_0
=
[a_t,a_{t+1},\ldots,a_{t+63}]
$$

其中：

$$
\boxed{
A_0
\in
\mathbb R^{64\times128}
}
$$

即未来 64 步动作块。

---

## 22. 推理流程压缩成一条数据链

```text
Current Observation

Language l
Images I
Robot State z_t
Control Frequency c

        │
        ▼

Language:
l → T5 → Adapter → C_L

Image:
I → SigLIP → Adapter → C_I

State:
z_t → MLP → T_z

Frequency:
c → Embedding → MLP → T_c

        │
        ▼

Random Action Initialization

A_K ~ N(0,I)

shape:
[64,128]

        │
        ▼

encode A_K

[64,128]
   ↓
Action MLP
   ↓
[64,2048]

        │
        ▼

construct Main Sequence

[K][freq][state][A_K ×64]

shape:
[67,2048]

        │
        ▼

RDT Transformer ×28

Language / Image
alternating Cross-Attention

        │
        ▼

MLP Decoder

        │
        ▼

Â_0^(K)

shape:
[64,128]

        │
        ▼

Reverse Diffusion

A_{K-1}
=
C_1 Â_0^(K)
+
C_2 A_K
+
σ_K z

        │
        ▼

A_{K-1}

        │
        ▼

重新构造 Main Sequence

        │
        ▼

RDT Transformer ×28

        │
        ▼

Â_0^(K-1)

        │
        ▼

A_{K-2}

        │
       ...

        ▼

A_0

        │
        ▼

Final Action Chunk

[a_t,...,a_{t+63}]
```

---

## 23. 一个非常重要的效率区别

RDT 每一次 diffusion step 都改变：

$$
A_k
$$

而：

$$
A_k
$$

就在 Transformer Main Sequence 里面。

所以：

$$
A_k
\rightarrow
A_{k-1}
$$

之后必须重新执行：

$$
\boxed{
RDT\ Transformer\times28
}
$$

也就是：

```text
A_K
 ↓
RDT ×28
 ↓
Â_0
 ↓
Reverse Diffusion
 ↓
A_K-1
 ↓
RDT ×28
 ↓
Â_0
 ↓
Reverse Diffusion
 ↓
A_K-2
 ↓
...
```

这和 Octo 有本质区别。

Octo：

```text
Observation
    ↓
Transformer Backbone
    ↓
Condition Representation
    ↓
Small Diffusion Head
    ↓
多次去噪
```

所以可以理解为：

$$
\boxed{
Octo:
Transformer=\text{Condition Encoder}
}
$$

而：

$$
\boxed{
RDT:
Transformer=\text{Diffusion Model}
}
$$

这也是为什么 RDT 的 diffusion inference 成本更高。

---

## 24. RDT vs Octo

|                  | Octo                      | RDT-1B                              |
| ---------------- | ------------------------- | ----------------------------------- |
| Transformer 主序列  | Task + Observation        | State + Noisy Action                |
| Language / Image | Transformer 输入 Token      | Cross-Attention Condition           |
| Action Diffusion | 独立小 Action Head           | RDT Transformer 本身                  |
| 预测目标             | Noise $\epsilon$          | Clean Action $A_0$                  |
| Action Chunk     | 较短                        | 64                                  |
| Action Tokens    | Diffusion Head 内部处理       | Transformer 主序列                     |
| Condition 融合     | Block-wise Self-Attention | Language / Image 交替 Cross-Attention |
| 多机器人统一           | Padding / Adapter 等       | 128D Unified Physical Space         |

最核心区别：

$$
\boxed{
Octo:
\text{先理解 Observation，再由小 Head 生成动作}
}
$$

而：

$$
\boxed{
RDT:
\text{直接让大 Transformer 对整段动作进行去噪}
}
$$

---

## 25. 最后只记这 7 个点

### 25.1 Main Sequence

$$
\boxed{
[k,\ freq,\ state,\ noisy\ actions_{1:64}]
}
$$

长度：

$$
67
$$

hidden：

$$
2048
$$

---

### 25.2 Language / Image 是 Condition

不是：

$$
[Language;Image;State;Action]
$$

全部 concat。

而是：

$$
Q=\text{Main Sequence}
$$

$$
K,V=\text{Condition}
$$

---

### 25.3 Alternating Cross-Attention

$$
\boxed{
Language
\rightarrow
Image
\rightarrow
Language
\rightarrow
Image
}
$$

避免：

$$
Vision\ Dominates\ Language
$$

---

### 25.4 Action Tokens 双向 Self-Attention

$$
a_i
\leftrightarrow
a_j
$$

64 步未来动作整体联合建模：

$$
\boxed{
\text{Trajectory-Level Coordination}
}
$$

而不是 autoregressive action generation。

---

### 25.5 Predict Clean Action

训练时：

$$
A_0
\rightarrow
A_k
\rightarrow
RDT
\rightarrow
\hat A_0
$$

即：

$$
\boxed{
A_k\rightarrow\hat A_0
}
$$

---

### 25.6 128D Unified Physical Space

不同机器人：

$$
Robot_A,\ Robot_B,\ Robot_C
$$

先：

$$
\boxed{
\text{按物理语义映射到统一 128D 空间}
}
$$

再进行联合训练。

---

### 25.7 Transformer 本身就是 Denoiser

因为：

$$
A_k
$$

直接进入 Transformer：

$$
\boxed{
A_k
\rightarrow
RDT\times28
\rightarrow
\hat A_0
}
$$

所以每一次 diffusion iteration：

$$
\boxed{
\text{都要重新运行整个 RDT Transformer}
}
$$

---

## 26. 脑内最终模型

看到 RDT-1B，只需要在脑中形成下面这张图：

```text
                     Multi-Robot Data
                           │
                           ▼
                128D Unified Physical Space
                           │
              ┌────────────┴─────────────┐
              │                          │
          Robot State               Action Chunk
           [128]                     [64,128]
              │                          │
              │                     Add Noise
              │                          │
              │                         A_k
              │                          │
              ▼                          ▼
          State MLP                  Action MLP
              │                          │
           [1,2048]                 [64,2048]
              │                          │
              └────────────┬─────────────┘
                           │
            k ── Embedding ┤
         freq ── Embedding ┤
                           │
                           ▼

          Main Sequence = [k][freq][state][A_k×64]

                     [67,2048]

                           │
                           ▼

              ┌────────────────────────┐
              │     RDT ×28 Layers     │
              │                        │
              │ RMSNorm                │
              │    ↓                   │
              │ Self-Attention         │
              │    ↓                   │
              │ Cross-Attention ◄──────── Language
              │    ↓                   │      ↓
              │ FFN                    │     T5
              │                        │      ↓
              │ Cross-Attention ◄──────── Adapter
              │                        │
              │ Cross-Attention ◄──────── Images
              │                        │      ↓
              │                        │   SigLIP
              │                        │      ↓
              │                        │   Adapter
              └────────────┬───────────┘
                           │
                           ▼

                       RMSNorm
                           │
                           ▼
                      MLP Decoder
                           │
                           ▼

                   Predicted Clean Action

                       Â_0
                     [64,128]

                           │
             ┌─────────────┴──────────────┐
             │                            │
          Training                     Inference
             │                            │
             ▼                            ▼
       compare A_0               Reverse Diffusion
             │                            │
             ▼                            ▼
       MSE Loss                  A_k → A_{k-1}
                                          │
                                          └──────↺
```

最终一句话：

> **RDT-1B = 先用具有固定物理语义的 128D 空间统一不同机器人数据，再把 Diffusion Timestep、Control Frequency、Robot State 和 64 个 Noisy Action Token 组成 Transformer Main Sequence；Language 与 Image 分别经过冻结的 T5 / SigLIP 编码，并通过 28 层中交替出现的 Cross-Attention 注入动作序列；64 个 Action Token 在 Self-Attention 中双向联合建模，最后由 Transformer 本身完成 Diffusion Denoising，直接预测未来 64 步 Clean Action Chunk。**
