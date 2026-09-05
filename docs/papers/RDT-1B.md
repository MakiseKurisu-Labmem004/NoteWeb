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

一次生成未来 **64 步 Action Chunk**。

---

# 2. 为什么需要 RDT

两个核心问题。

### Action Multi-Modality

同一个状态可能有多种正确动作：

* 双臂角色可以交换
* 路径可以不同
* 动作时序可以不同

所以不适合只学一个确定映射：

$$
(o_t,l)\rightarrow a_t
$$

而是建模一个动作分布：

$$
\boxed{
p(A\mid o_t,l)
}
$$

Diffusion 用来解决这个问题。

---

### Multi-Robot Heterogeneity

不同机器人存在：

* 不同机械结构
* 不同自由度
* 不同动作定义
* 不同控制频率

直接混合训练容易产生 negative transfer。

RDT 因此还设计了：

$$
\boxed{
128D\ \text{Physically Interpretable Unified State/Action Space}
}
$$

核心不是简单 padding，而是：

> **128 个位置拥有固定物理含义，不同机器人按照物理语义填入对应位置。**

---

# 3. 整体架构

看到 RDT 时脑中先出现：

```text
Language
   ↓
Frozen T5-XXL
   ↓
Language Adapter
   ↓
Language Condition ─────────────┐
                                │
Images                          │
   ↓                            │
Frozen SigLIP                   │
   ↓                            │
Image Adapter                   │
   ↓                            │
Image Condition ────────────────┤
                                │ Cross-Attention
                                ▼
                     RDT Transformer ×28
                                ▲
                                │
Diffusion Step ──┐              │
Control Freq ────┤              │
Robot State ─────┼→ Main Sequence
Noisy Action ×64 ┘
                                │
                                ▼
                           MLP Decoder
                                │
                                ▼
                      Predicted Clean
                        Action ×64
```

最重要的是分清：

$$
\boxed{\text{Main Sequence}}
$$

和

$$
\boxed{\text{Condition Sequence}}
$$

---

# 4. Main Sequence

RDT 真正进行 Self-Attention 的序列：

$$
\boxed{
X=
[
T_k,
T_c,
T_z,
T_{a_1},
\dots,
T_{a_{64}}
]
}
$$

分别表示：

* \(T_k\)：Diffusion timestep
* \(T_c\)：Control frequency
* \(T_z\)：Robot state
* \(T_{a_i}\)：第 \(i\) 个 noisy action token

因此长度大约：

$$
1+1+1+64=67
$$

hidden size：

$$
D=2048
$$

这条主序列才是 RDT 的“主体”。

---

# 5. Language / Image 是条件，不是主序列

语言：

$$
l
\rightarrow
\text{T5-XXL}
\rightarrow
4096D
\rightarrow
\text{Adapter}
\rightarrow
2048D
$$

得到：

$$
C_L=[L_1,\dots,L_m]
$$

图像：

$$
I
\rightarrow
\text{SigLIP}
\rightarrow
1152D
\rightarrow
\text{Adapter}
\rightarrow
2048D
$$

得到：

$$
C_I=[I_1,\dots,I_n]
$$

所以不是：

```text
Language + Image + State + Action
                 ↓
              concat
```

而是：

```text
State + Noisy Action
        ↓
   Main Sequence
        │
        │ Query
        ▼
Cross-Attention
        ▲
        │ K,V
Language / Image
```

这是理解 RDT 最重要的架构点之一。

---

# 6. 一个 RDT Block

RDT-1B：

$$
\boxed{
28\text{ Layers}
}
$$

每层结构：

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
Main Sequence
     ↓
Self-Attention
     ↓
机器人状态 / 动作之间的信息交互
     ↓
Cross-Attention
     ↓
注入 Language / Image
     ↓
FFN
```

---

# 7. Self-Attention 在看什么

这里：

$$
Q,K,V
$$

全部来自 Main Sequence：

$$
X=[k,c,z,a_1,\dots,a_{64}]
$$

因此每个 Action Token 可以看：

$$
\boxed{
[k,c,z,a_1,\dots,a_{64}]
}
$$

64 个 Action Token **可以双向 Attention**。

这很重要。

RDT 不是：

$$
a_1\rightarrow a_2\rightarrow a_3
$$

这种自回归生成。

而是：

$$
\boxed{
\text{联合建模整个未来 64 步动作轨迹}
}
$$

所以未来 Action Token 之间可以互相协调。

---

# 8. Cross-Attention 在看什么

Cross-Attention：

$$
\boxed{
Q=\text{Main Sequence}
}
$$

$$
\boxed{
K,V=\text{Language / Image Condition}
}
$$

即：

> State / Action Token 主动从语言和视觉中读取生成动作所需要的信息。

可以理解成：

$$
\boxed{
\text{“我现在这个动作候选，需要从任务和视觉中知道什么？”}
}
$$

---

# 9. Language / Image 为什么交替注入

如果每层都直接使用：

$$
[C_L;C_I]
$$

由于：

$$
N_{image}\gg N_{language}
$$

图像 Token 数量会远大于语言 Token，容易让：

$$
\boxed{
Vision\ Dominates\ Language
}
$$

所以 RDT 采用：

```text
Layer 1  → Language
Layer 2  → Image
Layer 3  → Language
Layer 4  → Image
...
Layer 27 → Language
Layer 28 → Image
```

即：

$$
\boxed{
Language
\rightarrow
Image
\rightarrow
Language
\rightarrow
Image
\rightarrow\cdots
}
$$

这是 RDT 很有辨识度的设计。

---

# 10. 为什么需要 Control Frequency

不同机器人数据可能来自：

$$
5Hz,\ 10Hz,\ 20Hz,\ 50Hz
$$

同一个动作：

$$
\Delta x=0.01
$$

在不同控制频率下对应的真实运动速度不同。

所以：

$$
\boxed{
\text{Control Frequency 本身也是一个输入 Token}
}
$$

使模型知道：

> “这些 Action 数值应该按照什么时间尺度解释。”

---

# 11. 多摄像头输入

RDT 可以使用：

$$
2\text{ 个时间步}
\times
3\text{ 个摄像头}
$$

例如：

```text
ext(t-1)
right wrist(t-1)
left wrist(t-1)

ext(t)
right wrist(t)
left wrist(t)
```

每张图：

$$
Image
\rightarrow
SigLIP
\rightarrow
Patch Tokens
$$

然后全部组成：

$$
C_I
$$

作为 Image Condition。

---

# 12. Independent Random Masking

不同输入的信息量不同。

例如外部相机可能非常强，模型可能形成 shortcut：

> “只看 exterior camera 就够了。”

所以训练时随机 Mask：

```text
Language       ✓ / ×
Exterior Cam   ✓ / ×
Wrist Cam      ✓ / ×
State          ✓ / ×
Frequency      ✓ / ×
```

迫使模型利用多种信息源。

核心作用：

$$
\boxed{
\text{防止 Shortcut Learning}
}
$$

以及学习：

$$
\boxed{
\text{multiple redundant cues}
}
$$

---

# 13. Diffusion：RDT 直接预测 Clean Action

真实 Action Chunk：

$$
A_0
=
[a_t,\dots,a_{t+63}]
$$

随机选择 diffusion step \(k\)，加入噪声：

$$
A_k=
\sqrt{\bar\alpha_k}A_0
+
\sqrt{1-\bar\alpha_k}\epsilon
$$

RDT 输入：

$$
(l,o_t,A_k,k)
$$

输出：

$$
\boxed{
\hat A_0
=
f_\theta(l,o_t,A_k,k)
}
$$

直接预测 **干净动作块**。

Loss：

$$
\boxed{
L=\|A_0-\hat A_0\|^2
}
$$

这和 Octo 非常值得对比：

$$
\boxed{
Octo:\quad A_k\rightarrow\hat\epsilon
}
$$

$$
\boxed{
RDT:\quad A_k\rightarrow\hat A_0
}
$$

---

# 14. 为什么说 Transformer 本身就是 Denoiser

这是 RDT 和 Octo 最核心的区别。

### Octo

```text
Observation
    ↓
Transformer
    ↓
Condition e_R
    ↓
Small Diffusion Head
    ↓
Action
```

Diffusion 发生在 Transformer 外部。

---

### RDT

```text
Noisy Action
     ↓
Transformer Main Sequence
     ↓
Predicted Clean Action
```

Noisy Action 本身直接进入 Transformer。

因此：

$$
\boxed{
\text{RDT Transformer 本身就是 Diffusion Denoiser}
}
$$

---

# 15. 推理流程

从纯高斯噪声：

$$
A_K\sim\mathcal N(0,I)
$$

开始。

当前：

$$
A_k
$$

和：

* language
* image
* robot state
* frequency
* diffusion timestep

一起进入 RDT：

$$
A_k
\rightarrow
RDT
\rightarrow
\hat A_0
$$

然后根据 reverse diffusion：

$$
\boxed{
A_{k-1}
=
C_1\hat A_0
+
C_2A_k
+
\sigma_kz
}
$$

得到更干净的：

$$
A_{k-1}
$$

不断：

$$
A_K
\rightarrow
A_{K-1}
\rightarrow
\cdots
\rightarrow
A_0
$$

最终：

$$
\boxed{
[a_t,\dots,a_{t+63}]
}
$$

---

# 16. 一个非常重要的效率区别

因为 RDT 的：

$$
A_k
$$

就在 Transformer Main Sequence 中，

所以每个 diffusion step：

$$
A_k\rightarrow A_{k-1}
$$

都必须重新运行一次 **RDT Transformer**。

即：

```text
A_K → RDT → A_K-1
            ↓
A_K-1 → RDT → A_K-2
              ↓
             ...
```

而 Octo 是：

```text
Observation
    ↓
Transformer       ← 一次
    ↓
e_R
    ↓
Small Action Head ← 多次 diffusion
```

所以：

$$
\boxed{
\text{Octo：Transformer 是 Condition Encoder}
}
$$

$$
\boxed{
\text{RDT：Transformer 本身是 Diffusion Model}
}
$$

---

# 17. 训练流程

脑中记住：

```text
Multi-Robot Trajectories
        ↓
统一 128D State / Action Space
        ↓
Observation + Future Action Chunk
        ↓
Random Condition Mask
        ↓
真实 Action A_0
        ↓
随机 diffusion timestep k
        ↓
加噪 → A_k
        ↓
构造 Main Sequence
[k][freq][state][A_k ×64]
        ↓
RDT Transformer ×28
        ↕
Language / Image
交替 Cross-Attention
        ↓
MLP Decoder
        ↓
Predicted A_0
        ↓
MSE Loss
```

---

# 18. 128D Unified Physical Space

多机器人联合训练时：

$$
\boxed{
D_{state/action}=128
}
$$

重点不是 128 这个数字本身，而是：

$$
\boxed{
\text{每个位置都有固定物理意义}
}
$$

不同机器人的数据：

```text
Robot A ─┐
Robot B ─┼→ 统一物理空间 → RDT
Robot C ─┘
```

缺少的维度进行 Mask。

这个设计试图让跨机器人数据共享的不是：

> “第 17 个数值”

而是：

> “第 17 个物理量”。

---

# 19. RDT vs Octo

|                  | Octo                      | RDT-1B                            |
| ---------------- | ------------------------- | --------------------------------- |
| Transformer 主要处理 | Task + Observation        | State + Noisy Action              |
| Image / Language | 输入 Token                  | Cross-Attention Condition         |
| Action Decoder   | 独立 Diffusion Head         | Transformer 本身                    |
| Diffusion 预测目标   | Noise \(\epsilon\)        | Clean Action \(A_0\)              |
| Action Chunk     | 较短                        | 64                                |
| 多机器人统一           | Adapter + Padding / Mask  | 128D Unified Physical Space       |
| 条件融合             | Block-wise Self-Attention | Language/Image 交替 Cross-Attention |

最值得记：

$$
\boxed{
Octo：
先理解环境，再让小 Action Head 生成动作
}
$$

$$
\boxed{
RDT：
直接让大 Transformer 对整段动作做去噪
}
$$

---

# 20. 最后只记这 6 个点

## ① Main Sequence

$$
[k,\ freq,\ state,\ noisy\ actions_{1:64}]
$$

---

## ② Language / Image 是 Condition

不是全部 concat。

$$
Q=\text{Main Sequence}
$$

$$
K,V=\text{Condition}
$$

---

## ③ Alternating Cross-Attention

$$
Language\rightarrow Image\rightarrow Language\rightarrow Image
$$

防止视觉信息压制语言。

---

## ④ Action Tokens 双向 Self-Attention

64 步未来动作一起建模：

$$
\boxed{
\text{整段动作协调}
}
$$

而不是 autoregressive。

---

## ⑤ Predict Clean Action

$$
\boxed{
A_k\rightarrow\hat A_0
}
$$

直接预测干净 Action Chunk。

---

## ⑥ 128D Unified Physical Space

通过固定物理语义统一多机器人状态和动作。

---

# 21. 脑内最终模型

```text
                  Language
                     ↓
              T5 + Adapter
                     │
                     │
Images → SigLIP → Adapter
                     │
              Condition Tokens
                     │
                     │ Cross-Attention
                     ▼
        ┌────────────────────────┐
        │    RDT × 28 Layers     │
        │                        │
        │ Self-Attention         │
        │        ↓               │
        │ Cross-Attention        │
        │        ↓               │
        │ FFN                    │
        └──────────▲─────────────┘
                   │
                   │ Main Sequence
                   │
        [k][freq][state][A_k ×64]
                   │
                   ▼
              MLP Decoder
                   │
                   ▼
              Predicted A_0
                   │
                   ▼
           Reverse Diffusion
                   │
                   └──────────↺
```

> **RDT-1B = 用 128D 物理空间统一多机器人数据，以 State + Noisy Action 为 Transformer 主序列，通过 Language / Image 交替 Cross-Attention 提供条件，让 28 层 Transformer 本身完成 Action Diffusion 去噪，最终生成未来 64 步 Action Chunk。**
