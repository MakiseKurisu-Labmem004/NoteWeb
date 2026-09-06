---

title: "Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations"
description: "FISH 解决少量示范下机器人策略不够鲁棒、又难以人工设计奖励的问题。核心理解：先从约一分钟示范得到一个固定 Base Policy，再通过专家轨迹与机器人 Rollout 的 Optimal Transport 自动产生奖励，用 Off-policy Actor-Critic 只学习 Residual Correction。"
date: "2026-09-05"
venue: "Robotics: Science and Systems (RSS) 2023"
authors: "Siddhant Haldar, Jyothish Pari, Anant Rai, Lerrel Pinto"
paper: ""
code: ""
--------

# FISH

![](./images/FISH.png)

## 1. 一句话理解

FISH 的核心不是重新训练一个完整 Policy，而是：

$$
\boxed{
\text{固定一个还不错的 Base Policy}
+
\text{RL 学一个 Residual Correction}
}
$$

机器人最终动作：

$$
\boxed{
a_t=a_t^b+a_t^r
}
$$

其中：

* $a_t^b$：Base Policy 给出的基础动作；
* $a_t^r$：Residual Policy 学到的修正量。

Residual 的训练不需要人工 Reward，而是：

$$
\boxed{
\text{Robot Rollout}
\xleftrightarrow{\text{Optimal Transport}}
\text{Expert Demonstration}
\rightarrow
r_t^{OT}
}
$$

再使用 Off-policy Actor-Critic 学习如何修正 Base Policy。

因此可以把 FISH 记成：

$$
\boxed{
\text{Imitation}
+
\text{Residual RL}
+
\text{Optimal Transport Reward}
}
$$

---

## 2. FISH 的整个系统只记两个阶段

```text
             Expert Demonstrations
                      │
                      ▼
════════════════════════════════════
 Phase 1：Offline Imitation
════════════════════════════════════
                      │
             学习视觉 Representation
                      │
                      ▼
                BC Encoder
                      │
                      ▼
             Fixed Base Policy π_b

════════════════════════════════════
 Phase 2：Online Residual RL
════════════════════════════════════
                      │
 Current Observation → Base Action a_b
                      │
                      ▼
              Residual Policy π_r
                      │
                      ▼
                  a_b + a_r
                      │
                      ▼
                    Robot
                      │
                      ▼
                   Rollout
                      │
                      ▼
       与 Expert Trajectory 做 OT
                      │
                      ▼
                  OT Reward
                      │
                      ▼
               Replay Buffer
                      │
                      ▼
          Off-policy Actor-Critic
                      │
                      └──────────────↺
```

整个 FISH 最重要的是：

> **先用少量 Demonstration 告诉机器人“大致该怎么做”，再让 RL 只学习“哪里需要修”。**

---

## 3. Phase 1：先得到 Base Policy

FISH 只有很少的人类示范：

$$
\boxed{
\text{约 1 分钟 Demonstrations}
}
$$

一条专家示范可以记为：

$$
\tau^e
=
\left[
(o_1^e,a_1^e),
(o_2^e,a_2^e),
\dots,
(o_T^e,a_T^e)
\right]
$$

其中：

* $o_t^e$：专家在时刻 $t$ 的视觉 Observation；
* $a_t^e$：专家对应的机器人 Action。

如果有多条示范：

$$
\mathcal D_E
=
\{
\tau^{e,1},
\tau^{e,2},
\dots,
\tau^{e,N}
\}
$$

FISH 倾向使用：

$$
\boxed{
\text{Non-parametric Base Policy}
}
$$

它的核心不是重新训练一个很大的网络：

$$
o_t
\rightarrow
\text{Neural Network}
\rightarrow
a_t
$$

而更像：

$$
\boxed{
\text{当前视觉状态}
\rightarrow
\text{在 Expert Demonstrations 中匹配}
\rightarrow
\text{使用对应 Expert Action}
}
$$

---

## 4. BC 在 FISH 中主要负责什么

这是 FISH 中非常容易混淆的一点。

虽然 Base Policy 可以采用 Non-parametric Policy，但系统仍然先训练一个：

$$
\boxed{
\text{Behavior Cloning Policy}
}
$$

不过 FISH 后续真正重要的是其中的：

$$
\boxed{
\text{Visual Encoder}
}
$$

把高维视觉 Observation：

$$
o_t
$$

映射成低维视觉表示：

$$
\boxed{
o_t
\xrightarrow{f_\psi}
z_t
}
$$

其中：

$$
z_t\in\mathbb R^{d_z}
$$

是视觉 Representation。

后面的：

* Base Policy；
* Residual Policy；
* Optimal Transport Reward；

都建立在这种视觉 Representation 上。

所以最值得记的是：

> **BC Policy 本身不是 FISH 最终的核心 Policy，BC Encoder 才是后续系统共享的视觉表示基础。**

---

## 5. Base Policy：先给一个“基本正确”的动作

当前 Observation：

$$
o_t
$$

先通过固定 Visual Encoder：

$$
o_t
\xrightarrow{f_\psi}
z_t
$$

得到：

$$
z_t
$$

然后 Base Policy：

$$
\boxed{
a_t^b=\pi_b(z_t)
}
$$

Base Policy 要回答的问题可以理解成：

> “当前机器人状态最像 Expert Demonstration 中的哪个状态？专家当时是怎么做的？”

一种典型形式是 Closed-loop VINN：

```text
Current RGB Observation
        │
        ▼
Fixed BC Encoder
        │
        ▼
       z_t
        │
        ▼
在 Expert Feature Database 中做 kNN
        │
        ▼
找到最相似的 Expert States
        │
        ▼
读取对应 Expert Actions
        │
        ▼
通过 LWR 等方式组合
        │
        ▼
Base Action a_t^b
```

它不是 Open-loop：

```text
a_1^e → a_2^e → a_3^e → ...
```

而是每执行一步之后重新观察环境：

$$
o_t
\rightarrow
a_t^b
\rightarrow
o_{t+1}
\rightarrow
a_{t+1}^b
$$

因此 Base Policy 本身是闭环的。

---

## 6. 为什么不直接 RL 微调 Base Policy

假设已经有一个虽然不够鲁棒、但基本能够完成任务的：

$$
\pi_b
$$

最直接的想法是：

$$
\pi_b
\xrightarrow{\text{Online RL}}
\pi'
$$

但是 RL 训练初期：

* Critic 不准确；
* Online Data 很少；
* Exploration 存在噪声；
* Policy Update 可能很大。

于是：

$$
\boxed{
\text{Direct RL Fine-tuning}
\rightarrow
\text{可能破坏已有 Imitation Ability}
}
$$

所以 FISH 不修改 Base Policy：

$$
\boxed{
\pi_b\ \text{固定}
}
$$

而是额外训练：

$$
\boxed{
\pi_r
}
$$

也就是 Residual Policy。

---

## 7. Residual Policy

Residual Policy 不需要重新学习整个任务。

它的输入是：

$$
(z_t,a_t^b)
$$

输出：

$$
\boxed{
a_t^r
=
\pi_r(z_t,a_t^b)
}
$$

其中：

* $z_t$：当前视觉状态；
* $a_t^b$：Base Policy 已经给出的基础动作；
* $a_t^r$：Residual Correction。

最终执行动作：

$$
\boxed{
a_t
=
a_t^b+a_t^r
}
$$

所以 Residual Policy 不是回答：

> “这个任务到底应该怎么完成？”

而是在回答：

> “Base Policy 已经给了一个基本动作，现在还需要往哪个方向修一点？”

因此：

$$
\boxed{
\text{Base Policy}
=
\text{解决主要任务}
}
$$

$$
\boxed{
\text{Residual Policy}
=
\text{处理 Distribution Shift 和执行误差}
}
$$

---

## 8. FISH 没有人工 Reward

机器人执行动作：

$$
a_t
$$

以后产生：

$$
o_{t+1}
$$

连续执行得到一整条 Robot Rollout：

$$
\tau^b
=
\left[
(o_1^b,a_1),
(o_2^b,a_2),
\dots,
(o_T^b,a_T)
\right]
$$

但是 FISH 没有手工设计：

```text
抓取成功：+10
碰撞：-5
距离目标过远：-1
```

这样的 Task-specific Reward。

FISH 的核心假设是：

$$
\boxed{
\text{机器人行为越像 Expert Demonstration}
\Rightarrow
\text{Reward 越高}
}
$$

而“像不像”通过：

$$
\boxed{
\text{Optimal Transport}
}
$$

来计算。

---

## 9. OT Reward：FISH 的第二个核心

专家轨迹：

$$
\tau^e
=
(o_1^e,\dots,o_{T_e}^e)
$$

机器人 Rollout：

$$
\tau^b
=
(o_1^b,\dots,o_{T_b}^b)
$$

全部经过固定 Visual Encoder：

$$
o\rightarrow z
$$

得到专家 Feature Sequence：

$$
Z^e
=
(z_1^e,\dots,z_{T_e}^e)
$$

和机器人 Feature Sequence：

$$
Z^b
=
(z_1^b,\dots,z_{T_b}^b)
$$

写成矩阵：

$$
Z^e
\in
\mathbb R^{T_e\times d_z}
$$

$$
Z^b
\in
\mathbb R^{T_b\times d_z}
$$

然后计算 Robot 第 $t$ 帧与 Expert 第 $t'$ 帧的 Cosine Distance：

$$
\boxed{
C_{t,t'}
=
1-
\frac{
z_t^b\cdot z_{t'}^e
}{
\|z_t^b\|\|z_{t'}^e\|
}
}
$$

最终得到：

$$
\boxed{
C\in\mathbb R^{T_b\times T_e}
}
$$

即整条 Robot Trajectory 和 Expert Trajectory 的：

$$
\boxed{
\text{Trajectory Cost Matrix}
}
$$

---

## 10. 为什么不能直接逐帧比较

不能简单规定：

$$
o_1^b\leftrightarrow o_1^e
$$

$$
o_2^b\leftrightarrow o_2^e
$$

$$
\cdots
$$

因为 Expert 和 Robot 执行速度可能不同。

Expert 的阶段可能是：

```text
Reach → Reach → Grasp → Lift
```

Robot 可能是：

```text
Reach → Reach → Reach → Grasp → Lift
```

虽然最后完成的是同一个行为，但：

$$
t_{\text{robot}}
\neq
t_{\text{expert}}
$$

因此不能做固定时间对齐。

Optimal Transport 学习一个 Transport Plan：

$$
\boxed{
\mu^*
\in
\mathbb R^{T_b\times T_e}
}
$$

其中：

$$
\mu_{t,t'}^*
$$

表示：

> Robot 第 $t$ 个状态应该以多大权重匹配 Expert 第 $t'$ 个状态。

因此 OT 比较的是：

$$
\boxed{
\text{两条完整视觉轨迹是否经历了相似的行为过程}
}
$$

而不是要求 timestep 严格对齐。

---

## 11. OT 怎么变成 Reward

Optimal Transport 求：

$$
\boxed{
\mu^*
=
\arg\min_{\mu}
\langle C,\mu\rangle
}
$$

并满足对应的 Transport Constraints。

实际可以通过带熵正则的 OT，并使用 Sinkhorn Algorithm 高效求解。

得到：

$$
\mu^*
\in
\mathbb R^{T_b\times T_e}
$$

之后，Robot 第 $t$ 个 timestep 的 OT Reward：

$$
\boxed{
r_t^{OT}
=
-\sum_{t'=1}^{T_e}
C_{t,t'}\mu_{t,t'}^*
}
$$

因此：

$$
C\downarrow
\Rightarrow
r^{OT}\uparrow
$$

也就是：

$$
\boxed{
\text{与 Expert 越相似}
\Rightarrow
\text{Reward 越高}
}
$$

最重要的一句话：

$$
\boxed{
\text{OT 在 FISH 中就是自动 Reward Generator}
}
$$

---

## 12. Reward 什么时候得到

普通 RL 往往是：

$$
(o_t,a_t)
\rightarrow
r_t
$$

执行一步马上获得 Reward。

但是 FISH 的 OT Reward 通常需要完整 Robot Rollout。

```text
完整执行一条 Robot Rollout
          │
          ▼
得到整条 Robot Trajectory
          │
          ▼
与 Expert Trajectory 做 OT
          │
          ▼
得到 Transport Plan μ*
          │
          ▼
反过来计算每个 timestep 的 Reward
```

也就是：

$$
\tau^b
\rightarrow
OT(\tau^b,\tau^e)
\rightarrow
\mu^*
$$

然后：

$$
\mu^*
\rightarrow
[r_1^{OT},\dots,r_{T_b}^{OT}]
$$

最后重新构造 RL Transition：

$$
\boxed{
(o_t,a_t,r_t^{OT},o_{t+1})
}
$$

所以 OT 的作用发生在：

$$
\boxed{
\text{Episode / Rollout 完成之后}
}
$$

---

## 13. Replay Buffer + Off-policy RL

每次真实机器人 Rollout 都很昂贵。

因此得到：

$$
(o_t,a_t,r_t^{OT},o_{t+1})
$$

之后，会保存到 Replay Buffer：

$$
\boxed{
\mathcal B
=
\{
(o_t,a_t,r_t^{OT},o_{t+1})
\}
}
$$

后续即使 Policy 从：

$$
\pi_r^{(1)}
\rightarrow
\pi_r^{(2)}
\rightarrow
\pi_r^{(3)}
$$

不断更新，旧的数据仍然可以训练新的 Policy。

因此：

$$
\boxed{
\text{一次真实机器人交互}
\rightarrow
\text{多次 Gradient Update}
}
$$

这就是 Off-policy Learning 对 FISH 特别重要的原因：

$$
\boxed{
\text{最大化真实机器人数据的 Sample Efficiency}
}
$$

---

## 14. Actor-Critic 到底各学什么

### Actor

FISH 的 Actor 就是 Residual Policy：

$$
\boxed{
\pi_r(z_t,a_t^b)
\rightarrow
a_t^r
}
$$

最终用于环境的动作不是：

$$
a_t^r
$$

而是：

$$
\boxed{
a_t
=
a_t^b+a_t^r
}
$$

Actor 的目标就是找到：

$$
\boxed{
\text{能够让长期累计 OT Reward 更高的 Residual Action}
}
$$

---

### Critic

Critic 学习：

$$
\boxed{
Q(z_t,a_t)
}
$$

它回答：

> 在当前视觉状态 $z_t$ 下执行动作 $a_t$，之后继续按照当前策略执行，未来累计 OT Reward 大约是多少？

因此：

```text
OT Reward
    │
    ▼
Critic 学习 Action Value
    │
    ▼
Q(z,a)
    │
    ▼
指导 Residual Actor
    │
    ▼
Residual Correction 越来越好
```

---

## 15. FISH 使用 3-step Return

普通 one-step TD Target：

$$
y_t^{(1)}
=
r_t
+
\gamma
Q(z_{t+1},a_{t+1})
$$

只使用一个真实 Reward：

$$
r_t
$$

然后立刻用 Critic 估计后面的未来。

FISH 使用：

$$
\boxed{
3\text{-step return}
}
$$

也就是：

$$
\boxed{
y_t
=
r_t
+
\gamma r_{t+1}
+
\gamma^2r_{t+2}
+
\gamma^3
Q(z_{t+3},a_{t+3})
}
$$

Critic Loss：

$$
\boxed{
L_Q
=
\left(
Q(z_t,a_t)-y_t
\right)^2
}
$$

它的核心作用是：

$$
\boxed{
\text{一次使用连续 3 步真实 OT Reward}
}
$$

这样 Reward Signal 可以更快传播到较早的动作。

因此只需要记：

$$
\boxed{
\text{FISH}
=
\text{Off-policy Actor-Critic}
+
\text{3-step TD}
}
$$

---

## 16. 整个训练闭环

```text
                  Expert Demonstrations
                           │
                           ▼
                    Train BC Policy
                           │
                           ▼
                    Fixed BC Encoder
                           │
                           ▼
                Expert Feature Database
                           │
                           │
                           ▼
Current Observation ──► Encoder
                           │
                           ▼
                          z_t
                           │
                 ┌─────────┴──────────┐
                 │                    │
                 ▼                    │
           Base Policy π_b            │
                 │                    │
                 ▼                    │
               a_t^b                  │
                 │                    │
                 └──────────────┐     │
                                ▼     ▼
                         Residual Actor π_r
                                │
                                ▼
                              a_t^r
                                │
                                ▼
                         a_t=a_t^b+a_t^r
                                │
                                ▼
                              Robot
                                │
                                ▼
                         New Observation
                                │
                                ▼
                          Robot Rollout
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
          Expert Trajectory            Robot Trajectory
                  │                           │
                  └────────────► OT ◄─────────┘
                                │
                                ▼
                            OT Rewards
                                │
                                ▼
                           Replay Buffer
                                │
                                ▼
                     Off-policy Actor-Critic
                                │
                                ▼
                     Update Residual Policy
                                │
                                └──────────────↺
```

---

## 17. 用数据符号完整跑一遍 FISH

这一节只关注一件事：

> **数据从进入 FISH 开始，到训练完成，每一步究竟变成了什么。**

完整的数据流可以写成：

$$
\boxed{
\mathcal D_E
\rightarrow
\text{Visual Representation}
\rightarrow
\text{Base Policy}
\rightarrow
\text{Robot Rollout}
\rightarrow
OT
\rightarrow
r^{OT}
\rightarrow
\mathcal B
\rightarrow
\text{Actor-Critic}
\rightarrow
\pi_r
}
$$

---

### 17.1 定义符号

假设一共有 $N$ 条 Expert Demonstrations：

$$
\mathcal D_E
=
\{
\tau^{e,1},
\dots,
\tau^{e,N}
\}
$$

第 $n$ 条 Demonstration：

$$
\tau^{e,n}
=
\left[
(o_1^{e,n},a_1^{e,n}),
\dots,
(o_{T_n}^{e,n},a_{T_n}^{e,n})
\right]
$$

设视觉 Observation 记为：

$$
o_t
$$

如果使用连续 $k$ 帧作为 Observation，可以统一记成：

$$
o_t
=
[I_{t-k+1},\dots,I_t]
$$

符号上可以看作：

$$
o_t
\in
\mathbb R^{k\times H\times W\times 3}
$$

如果只使用单帧，则可以退化成：

$$
o_t\in\mathbb R^{H\times W\times3}
$$

动作记为：

$$
a_t\in\mathbb R^{d_a}
$$

例如它可能表示机器人连续控制量，但这里不关心每一维的具体物理意义。

因此最初的数据格式就是：

$$
\boxed{
(o_t^e,a_t^e)
}
$$

即：

```text
Raw RGB Observation
+
Continuous Robot Action
```

---

### 17.2 Step 1：把 Expert Demonstration 展开成监督学习样本

原始数据：

$$
\mathcal D_E
=
\{
\tau^{e,1},
\dots,
\tau^{e,N}
\}
$$

其中每条轨迹是一个时间序列。

首先把轨迹中的 timestep 展开：

$$
\tau^{e,n}
\rightarrow
\{
(o_t^{e,n},a_t^{e,n})
\}_{t=1}^{T_n}
$$

把所有 Demonstrations 合并：

$$
\boxed{
\mathcal D_{BC}
=
\{
(o_i^e,a_i^e)
\}_{i=1}^{M}
}
$$

其中：

$$
M
=
\sum_{n=1}^{N}T_n
$$

数据格式变化：

```text
原始格式：

N 条 trajectory
[
    [(o_1,a_1), ..., (o_T,a_T)],
    ...
]

            ↓ flatten timestep

BC Dataset：

(o_1,a_1)
(o_2,a_2)
...
(o_M,a_M)
```

也就是：

$$
\boxed{
\text{Trajectory Dataset}
\rightarrow
\text{Observation-Action Pairs}
}
$$

---

### 17.3 Step 2：训练 Behavior Cloning 网络

BC 网络可以概念化为：

$$
o_i^e
\xrightarrow{f_\psi}
z_i^e
\xrightarrow{g_\omega}
\hat a_i^e
$$

其中：

$$
f_\psi:
\mathbb R^{\text{image}}
\rightarrow
\mathbb R^{d_z}
$$

是 Visual Encoder。

所以单个样本的数据变化：

$$
\boxed{
o_i^e
\rightarrow
z_i^e
\rightarrow
\hat a_i^e
}
$$

格式变化：

$$
o_i^e
\in
\mathbb R^{k\times H\times W\times3}
$$

经过 Encoder：

$$
z_i^e
=
f_\psi(o_i^e)
\in
\mathbb R^{d_z}
$$

再经过 BC Action Head：

$$
\hat a_i^e
=
g_\omega(z_i^e)
\in
\mathbb R^{d_a}
$$

用 Expert Action：

$$
a_i^e
\in
\mathbb R^{d_a}
$$

监督：

$$
\hat a_i^e
\approx
a_i^e
$$

因此 BC 阶段本质上是：

$$
\boxed{
(o_i^e,a_i^e)
\rightarrow
(z_i^e,\hat a_i^e)
\rightarrow
L_{BC}
}
$$

---

### 17.4 Step 3：训练完 BC 后，只保留重要组件

BC 训练完成以后：

$$
f_\psi
$$

已经能够把：

$$
\text{RGB Robot Observation}
$$

映射成：

$$
\text{Robot-task-aware Visual Representation}
$$

FISH 后续重点使用固定的：

$$
\boxed{
f_\psi
}
$$

于是：

$$
\psi
\leftarrow
\text{freeze}
$$

之后不再依靠在线 RL 去大幅修改 Visual Representation。

---

### 17.5 Step 4：把 Expert Demonstrations 转换成 Feature Database

对所有 Expert Observation：

$$
o_i^e
$$

通过固定 Encoder：

$$
z_i^e
=
f_\psi(o_i^e)
$$

于是：

$$
(o_i^e,a_i^e)
$$

变成：

$$
(z_i^e,a_i^e)
$$

最终得到：

$$
\boxed{
\mathcal M_E
=
\{
(z_i^e,a_i^e)
\}_{i=1}^{M}
}
$$

可以把它理解成：

```text
Expert Memory

┌──────────────┬──────────────┐
│ Feature z_i  │ Action a_i   │
├──────────────┼──────────────┤
│ z_1^e        │ a_1^e        │
│ z_2^e        │ a_2^e        │
│ ...          │ ...          │
│ z_M^e        │ a_M^e        │
└──────────────┴──────────────┘
```

数据格式完成第一次重要变化：

$$
\boxed{
(o_i^e,a_i^e)
\rightarrow
(z_i^e,a_i^e)
}
$$

即：

$$
\boxed{
\text{Pixel Space}
\rightarrow
\text{Latent Feature Space}
}
$$

---

### 17.6 Step 5：Online Rollout 开始，机器人得到当前 Observation

机器人环境返回：

$$
o_t^b
$$

数据仍然是 Raw Visual Observation：

$$
o_t^b
\in
\mathbb R^{k\times H\times W\times3}
$$

通过同一个固定 Encoder：

$$
\boxed{
z_t^b
=
f_\psi(o_t^b)
}
$$

得到：

$$
z_t^b
\in
\mathbb R^{d_z}
$$

所以：

```text
Robot Raw RGB
     │
     ▼
Fixed BC Encoder
     │
     ▼
Robot Feature z_t^b
```

---

### 17.7 Step 6：Base Policy 查询 Expert Memory

当前机器人 Feature：

$$
z_t^b
$$

与 Expert Database：

$$
\{
z_1^e,\dots,z_M^e
\}
$$

比较距离。

得到：

$$
d(z_t^b,z_i^e)
$$

然后寻找：

$$
\mathcal N_k(z_t^b)
=
kNN
\left(
z_t^b,
\{z_i^e\}
\right)
$$

得到 $k$ 个最近的 Expert Feature：

$$
\{
z_{j_1}^e,
\dots,
z_{j_k}^e
\}
$$

同时取出对应动作：

$$
\{
a_{j_1}^e,
\dots,
a_{j_k}^e
\}
$$

经过 Base Policy 的局部组合：

$$
\boxed{
a_t^b
=
\pi_b(z_t^b)
}
$$

因此这一阶段的数据变化：

$$
z_t^b
$$

变成：

$$
\{
(z_{j_l}^e,a_{j_l}^e)
\}_{l=1}^{k}
$$

最后压缩为一个：

$$
a_t^b\in\mathbb R^{d_a}
$$

即：

$$
\boxed{
z_t^b
\rightarrow
kNN
\rightarrow
\text{Expert Neighbor Actions}
\rightarrow
a_t^b
}
$$

---

### 17.8 Step 7：Residual Actor 产生 Correction

现在已经有：

$$
z_t^b\in\mathbb R^{d_z}
$$

以及：

$$
a_t^b\in\mathbb R^{d_a}
$$

将两者作为 Residual Actor 的输入：

$$
\boxed{
[z_t^b;a_t^b]
}
$$

其维度可以记作：

$$
[z_t^b;a_t^b]
\in
\mathbb R^{d_z+d_a}
$$

Residual Actor：

$$
\boxed{
a_t^r
=
\pi_\phi(z_t^b,a_t^b)
}
$$

输出：

$$
a_t^r\in\mathbb R^{d_a}
$$

所以数据格式：

$$
\boxed{
\mathbb R^{d_z}
+
\mathbb R^{d_a}
\rightarrow
\mathbb R^{d_a}
}
$$

也就是：

```text
Visual Feature z_t
       +
Base Action a_t^b
       │
       ▼
Residual Actor
       │
       ▼
Correction a_t^r
```

---

### 17.9 Step 8：Base Action 与 Residual Action 相加

现在：

$$
a_t^b\in\mathbb R^{d_a}
$$

$$
a_t^r\in\mathbb R^{d_a}
$$

逐维相加：

$$
\boxed{
a_t
=
a_t^b+a_t^r
}
$$

所以：

$$
\mathbb R^{d_a}
+
\mathbb R^{d_a}
\rightarrow
\mathbb R^{d_a}
$$

得到真正发送给 Robot Controller 的 Action：

$$
a_t
$$

这里没有复杂的 Decoder：

```text
Base Action
    │
    ├────────┐
    │        │
    │    Residual Action
    │        │
    └─── + ──┘
         │
         ▼
   Final Action
```

---

### 17.10 Step 9：环境执行 Action，得到下一帧

机器人执行：

$$
a_t
$$

环境转移：

$$
(o_t^b,a_t)
\xrightarrow{P}
o_{t+1}^b
$$

于是单步交互数据暂时形成：

$$
\boxed{
(o_t^b,a_t,o_{t+1}^b)
}
$$

注意：

> 此时还没有 $r_t^{OT}$。

因为 FISH 的 Reward 要等整条 Rollout 完成之后才能计算。

继续执行：

$$
o_{t+1}^b
\rightarrow
z_{t+1}^b
\rightarrow
a_{t+1}^b
\rightarrow
a_{t+1}^r
\rightarrow
a_{t+1}
$$

直到：

$$
t=T_b
$$

得到完整 Robot Rollout：

$$
\boxed{
\tau^b
=
[
(o_1^b,a_1),
\dots,
(o_{T_b}^b,a_{T_b})
]
}
$$

---

### 17.11 Step 10：Robot Rollout 转换成 Feature Sequence

Robot Rollout 中的 Observation：

$$
[
o_1^b,\dots,o_{T_b}^b
]
$$

全部进入固定 Encoder：

$$
z_t^b=f_\psi(o_t^b)
$$

于是：

$$
\boxed{
Z^b
=
[
z_1^b,\dots,z_{T_b}^b
]
}
$$

矩阵形式：

$$
Z^b
\in
\mathbb R^{T_b\times d_z}
$$

Expert Trajectory 同样：

$$
\boxed{
Z^e
=
[
z_1^e,\dots,z_{T_e}^e
]
}
$$

$$
Z^e
\in
\mathbb R^{T_e\times d_z}
$$

于是：

```text
Robot Pixels
[T_b × Image]

        ↓ Encoder

Robot Features
[T_b × d_z]


Expert Pixels
[T_e × Image]

        ↓ Encoder

Expert Features
[T_e × d_z]
```

---

### 17.12 Step 11：两条 Feature Sequence 变成 Cost Matrix

现在有：

$$
Z^b
\in
\mathbb R^{T_b\times d_z}
$$

$$
Z^e
\in
\mathbb R^{T_e\times d_z}
$$

对每一个：

$$
(z_t^b,z_{t'}^e)
$$

计算：

$$
C_{t,t'}
=
1-
\frac{
z_t^b\cdot z_{t'}^e
}{
\|z_t^b\|\|z_{t'}^e\|
}
$$

最终得到：

$$
\boxed{
C
\in
\mathbb R^{T_b\times T_e}
}
$$

所以数据格式发生：

$$
\boxed{
(T_b\times d_z)
+
(T_e\times d_z)
\rightarrow
(T_b\times T_e)
}
$$

这个矩阵不再是 Feature。

它表示：

$$
\boxed{
\text{Robot 每个 timestep 与 Expert 每个 timestep 的不相似程度}
}
$$

---

### 17.13 Step 12：Cost Matrix 经过 Optimal Transport

输入：

$$
C
\in
\mathbb R^{T_b\times T_e}
$$

Optimal Transport 求：

$$
\mu^*
=
\arg\min_\mu
\langle C,\mu\rangle
$$

得到：

$$
\boxed{
\mu^*
\in
\mathbb R^{T_b\times T_e}
}
$$

所以 Shape 不变：

$$
T_b\times T_e
\rightarrow
T_b\times T_e
$$

但是数据意义完全改变。

输入：

$$
C_{t,t'}
=
\text{两帧有多不相似}
$$

输出：

$$
\mu_{t,t'}^*
=
\text{两帧应该匹配多少}
$$

因此：

$$
\boxed{
\text{Distance Matrix}
\rightarrow
\text{Temporal Matching Matrix}
}
$$

---

### 17.14 Step 13：Transport Matrix 压缩成 Reward Vector

对于 Robot 第 $t$ 个 timestep：

$$
r_t^{OT}
=
-\sum_{t'=1}^{T_e}
C_{t,t'}\mu_{t,t'}^*
$$

于是：

$$
C,\mu^*
\in
\mathbb R^{T_b\times T_e}
$$

经过沿 Expert 时间维度求和：

$$
\boxed{
r^{OT}
=
[
r_1^{OT},
\dots,
r_{T_b}^{OT}
]
}
$$

其中：

$$
r^{OT}
\in
\mathbb R^{T_b}
$$

所以 Shape 变化：

$$
\boxed{
T_b\times T_e
\rightarrow
T_b
}
$$

即：

```text
Robot × Expert Matching Matrix
             │
             ▼
     沿 Expert 时间维求和
             │
             ▼
Robot Per-step Reward Vector
```

---

### 17.15 Step 14：重新给 Robot Rollout 补上 Reward

刚才保存的 Rollout：

$$
\tau^b
=
[
(o_1^b,a_1),
\dots,
(o_{T_b}^b,a_{T_b})
]
$$

现在有：

$$
[
r_1^{OT},
\dots,
r_{T_b}^{OT}
]
$$

于是把 Reward 插回每一个 timestep：

$$
\boxed{
(o_t^b,a_t,r_t^{OT},o_{t+1}^b)
}
$$

整条轨迹从：

$$
\boxed{
(o_t,a_t,o_{t+1})
}
$$

变成 RL 可以使用的：

$$
\boxed{
(o_t,a_t,r_t,o_{t+1})
}
$$

这一步非常关键。

因为：

> **OT 的真正作用，就是把原本没有 Reward 的 Demonstration Matching 问题转换成标准 Off-policy RL 数据。**

---

### 17.16 Step 15：Transition 写入 Replay Buffer

将所有 timestep 放入：

$$
\boxed{
\mathcal B
}
$$

即：

$$
\mathcal B
\leftarrow
\mathcal B
\cup
\{
(o_t^b,a_t,r_t^{OT},o_{t+1}^b)
\}_{t=1}^{T_b-1}
$$

Replay Buffer 的逻辑格式：

```text
┌─────────┬────────┬──────────┬─────────┐
│   o_t   │  a_t   │   r_t    │ o_{t+1} │
├─────────┼────────┼──────────┼─────────┤
│   ...   │  ...   │   ...    │   ...   │
└─────────┴────────┴──────────┴─────────┘
```

注意这里存的：

$$
a_t
$$

是最终执行动作：

$$
a_t=a_t^b+a_t^r
$$

而不是只存 Residual：

$$
a_t^r
$$

---

### 17.17 Step 16：从 Replay Buffer 取连续 3-step 数据

FISH 使用 3-step Return。

因此训练 Critic 时，需要得到：

$$
(o_t,a_t,r_t)
$$

$$
(o_{t+1},a_{t+1},r_{t+1})
$$

$$
(o_{t+2},a_{t+2},r_{t+2})
$$

以及：

$$
o_{t+3}
$$

也就是：

$$
\boxed{
[
o_t,
a_t,
r_t,
r_{t+1},
r_{t+2},
o_{t+3}
]
}
$$

对于 Observation：

$$
o_t
$$

再次经过固定 Encoder：

$$
z_t=f_\psi(o_t)
$$

所以 Actor-Critic 真正看到的主要是：

$$
\boxed{
(z_t,a_t,r_t,z_{t+3})
}
$$

---

### 17.18 Step 17：计算 $t+3$ 时刻的新 Policy Action

对：

$$
o_{t+3}
$$

先：

$$
z_{t+3}
=
f_\psi(o_{t+3})
$$

Base Policy：

$$
a_{t+3}^b
=
\pi_b(z_{t+3})
$$

Residual Actor：

$$
a_{t+3}^r
=
\pi_r(z_{t+3},a_{t+3}^b)
$$

最终：

$$
\boxed{
a_{t+3}
=
a_{t+3}^b+a_{t+3}^r
}
$$

然后 Critic 计算：

$$
Q(z_{t+3},a_{t+3})
$$

---

### 17.19 Step 18：构造 3-step TD Target

真实 OT Reward：

$$
r_t^{OT}
$$

$$
r_{t+1}^{OT}
$$

$$
r_{t+2}^{OT}
$$

加上 Critic 对更远未来的估计：

$$
Q(z_{t+3},a_{t+3})
$$

构成：

$$
\boxed{
y_t
=
r_t^{OT}
+
\gamma r_{t+1}^{OT}
+
\gamma^2r_{t+2}^{OT}
+
\gamma^3Q(z_{t+3},a_{t+3})
}
$$

这里的数据变化是：

```text
3 个 scalar Reward
        +
1 个 scalar Q-value
        │
        ▼
1 个 scalar TD Target y_t
```

即：

$$
\boxed{
(r_t,r_{t+1},r_{t+2},Q_{t+3})
\rightarrow
y_t
}
$$

---

### 17.20 Step 19：更新 Critic

当前 Critic 输入：

$$
(z_t,a_t)
$$

输出：

$$
Q_\theta(z_t,a_t)
$$

Target：

$$
y_t
$$

Loss：

$$
\boxed{
L_Q
=
\left(
Q_\theta(z_t,a_t)-y_t
\right)^2
}
$$

然后：

$$
\theta
\leftarrow
\theta
-
\eta_Q\nabla_\theta L_Q
$$

因此 Critic 学到：

$$
\boxed{
(z,a)
\rightarrow
\text{长期累计 OT Reward}
}
$$

---

### 17.21 Step 20：更新 Residual Actor

对于 Replay Buffer 中的 Observation：

$$
o_t
$$

先：

$$
z_t=f_\psi(o_t)
$$

然后 Base Policy 给：

$$
a_t^b=\pi_b(z_t)
$$

Residual Actor：

$$
a_t^r=\pi_\phi(z_t,a_t^b)
$$

最终 Policy Action：

$$
\tilde a_t
=
a_t^b+a_t^r
$$

Critic 对其评价：

$$
Q_\theta(z_t,\tilde a_t)
$$

Actor 希望这个值越大越好：

$$
\boxed{
\max_\phi
Q_\theta
\left(
z_t,
a_t^b+
\pi_\phi(z_t,a_t^b)
\right)
}
$$

等价地可以写成 Loss：

$$
\boxed{
L_\pi
=
-
Q_\theta
\left(
z_t,
a_t^b+
\pi_\phi(z_t,a_t^b)
\right)
}
$$

更新：

$$
\phi
\leftarrow
\phi
-
\eta_\pi\nabla_\phi L_\pi
$$

注意：

$$
\boxed{
\pi_b\ \text{不更新}
}
$$

$$
\boxed{
f_\psi\ \text{保持固定}
}
$$

主要学习的是：

$$
\boxed{
\pi_r
}
$$

以及用于训练它的：

$$
\boxed{
Q
}
$$

---

### 17.22 Step 21：更新后的 Residual Policy 再去真实机器人 Rollout

现在：

$$
\pi_r^{(k)}
$$

经过 Actor-Critic 更新变成：

$$
\pi_r^{(k+1)}
$$

下一轮真实机器人执行：

$$
o_t
\rightarrow
z_t
$$

$$
z_t
\rightarrow
a_t^b
$$

$$
(z_t,a_t^b)
\rightarrow
a_t^{r,(k+1)}
$$

$$
a_t
=
a_t^b+a_t^{r,(k+1)}
$$

产生新的：

$$
\tau^{b,(k+1)}
$$

然后再次：

$$
\tau^{b,(k+1)}
\rightarrow
OT
\rightarrow
r^{OT}
\rightarrow
\mathcal B
\rightarrow
\text{Actor-Critic Update}
$$

因此整个 Online Training 是：

$$
\boxed{
\text{Rollout}
\rightarrow
OT
\rightarrow
Replay
\rightarrow
RL Update
\rightarrow
\text{New Rollout}
\rightarrow
\cdots
}
$$

---

### 17.23 完整数据格式变化总结

| 阶段                | 输入                            | 操作                            | 输出                                  |
| ----------------- | ----------------------------- | ----------------------------- | ----------------------------------- |
| Expert Data       | $\tau^e$                      | 展开 trajectory                 | $(o_i^e,a_i^e)$                     |
| BC Encoder        | $o_i^e$                       | Visual Encoding               | $z_i^e\in\mathbb R^{d_z}$           |
| Expert Memory     | $(z_i^e,a_i^e)$               | 保存                            | $\mathcal M_E$                      |
| Robot Observation | $o_t^b$                       | Encoder                       | $z_t^b$                             |
| Base Policy       | $z_t^b$                       | kNN / Non-parametric Matching | $a_t^b$                             |
| Residual Actor    | $(z_t^b,a_t^b)$               | Neural Policy                 | $a_t^r$                             |
| Final Action      | $(a_t^b,a_t^r)$               | Addition                      | $a_t$                               |
| Environment       | $(o_t^b,a_t)$                 | Robot Dynamics                | $o_{t+1}^b$                         |
| Robot Rollout     | ${o_t^b}$                     | Encoder                       | $Z^b\in\mathbb R^{T_b\times d_z}$   |
| Expert Trajectory | ${o_{t'}^e}$                  | Encoder                       | $Z^e\in\mathbb R^{T_e\times d_z}$   |
| OT Cost           | $(Z^b,Z^e)$                   | Pairwise Cosine Distance      | $C\in\mathbb R^{T_b\times T_e}$     |
| OT Matching       | $C$                           | Sinkhorn / OT                 | $\mu^*\in\mathbb R^{T_b\times T_e}$ |
| Reward            | $(C,\mu^*)$                   | 沿 Expert 时间维聚合                | $r^{OT}\in\mathbb R^{T_b}$          |
| RL Transition     | Rollout + Reward              | 对齐 timestep                   | $(o_t,a_t,r_t,o_{t+1})$             |
| Replay Buffer     | Transition                    | 存储                            | $\mathcal B$                        |
| 3-step Target     | $r_t,r_{t+1},r_{t+2},Q_{t+3}$ | TD Backup                     | $y_t$                               |
| Critic            | $(z_t,a_t,y_t)$               | TD Learning                   | 更新 $Q_\theta$                       |
| Actor             | $(z_t,a_t^b,Q)$               | Policy Gradient               | 更新 $\pi_r$                          |

把它压缩成一条数据链：

$$
\boxed{
\begin{aligned}
&(o^e,a^e)\\
&\downarrow f_\psi\\
&(z^e,a^e)\\
&\downarrow\text{Expert Memory}\\
&o_t^b
\rightarrow z_t^b
\rightarrow a_t^b\\
&\downarrow\pi_r\\
&a_t^r\\
&\downarrow\\
&a_t=a_t^b+a_t^r\\
&\downarrow\text{Robot}\\
&\tau^b\\
&\downarrow f_\psi\\
&Z^b\\
&\xleftrightarrow{\text{pairwise distance}}
Z^e\\
&\downarrow\\
&C\\
&\downarrow OT\\
&\mu^*\\
&\downarrow\\
&r^{OT}\\
&\downarrow\\
&(o_t,a_t,r_t,o_{t+1})\\
&\downarrow\mathcal B\\
&\text{3-step Actor-Critic}\\
&\downarrow\\
&\pi_r\text{ updated}
\end{aligned}
}
$$

---

## 18. 推理 / 部署时反而非常简单

训练完成以后：

$$
\boxed{
\text{不再需要 OT}
}
$$

因为 OT 的功能只是：

$$
\boxed{
\text{Training-time Reward Generator}
}
$$

部署时不需要：

* OT；
* Cost Matrix；
* Sinkhorn；
* Critic；
* Replay Buffer；
* RL Update。

真正执行的是：

```text
Current RGB Observation
        │
        ▼
Fixed Encoder
        │
        ▼
       z_t
        │
        ├──────────────► Base Policy
        │                    │
        │                    ▼
        │                   a_t^b
        │                    │
        └──────────────► Residual Policy
                             │
                             ▼
                            a_t^r
                             │
                             ▼
                    a_t = a_t^b + a_t^r
                             │
                             ▼
                           Robot
                             │
                             ▼
                      New Observation
                             │
                             └──────────────↺
```

符号形式：

$$
o_t
\xrightarrow{f_\psi}
z_t
$$

$$
a_t^b
=
\pi_b(z_t)
$$

$$
a_t^r
=
\pi_r(z_t,a_t^b)
$$

$$
\boxed{
a_t=a_t^b+a_t^r
}
$$

然后得到：

$$
o_{t+1}
$$

再次执行相同流程。

所以推理是一个标准的：

$$
\boxed{
\text{Closed-loop Visual Control}
}
$$

---

## 19. FISH 与直接 BC / RL 的区别

### Behavior Cloning

$$
o_t
\rightarrow
\pi_{BC}
\rightarrow
a_t
$$

问题：

> 少量 Demonstration 无法覆盖机器人可能偏离到的所有状态，一旦进入 Demonstration Distribution 之外的状态就容易失败。

---

### 直接 Online RL

$$
o_t
\rightarrow
\pi_{RL}
\rightarrow
a_t
$$

问题：

* 从零探索真实机器人代价高；
* Sample Efficiency 低；
* 需要人工 Reward；
* 初期策略表现很差。

---

### FISH

$$
\boxed{
\text{Imitation 提供“基本会做”}
+
\text{Residual RL 提供“会纠错”}
}
$$

同时使用：

$$
\boxed{
\text{OT 替代人工 Reward}
}
$$

因此 FISH 实际把：

$$
\boxed{
\text{Offline Imitation}
+
\text{Online RL}
+
\text{Trajectory Matching}
}
$$

连接到了一起。

---

## 20. FISH vs ROT

这个对比最适合理解为什么 FISH 要使用 Residual Learning。

### ROT

ROT 的基本思想是：

$$
\boxed{
\text{直接使用 RL 微调原来的 BC Policy}
}
$$

因此：

$$
\pi_{BC}
\rightarrow
\pi_{\text{ROT}}
$$

原来的 Policy 参数本身不断被修改。

风险：

$$
\boxed{
\text{RL Update}
\rightarrow
\text{可能破坏已有 Imitation Ability}
}
$$

---

### FISH

FISH 则是：

$$
\boxed{
\pi_b\text{ 固定}
+
\pi_r\text{ 学 Correction}
}
$$

Base Policy 始终是一个 Stable Anchor。

最终：

$$
a_t
=
\underbrace{a_t^b}_{\text{Base}}
+
\underbrace{a_t^r}_{\text{Correction}}
$$

因此最值得记的是：

> **不要让 RL 重新学习“怎么完成任务”，而只让 RL 学习“Base Policy 哪里需要修”。**

---

## 21. 最后只记这 5 个点

### 21.1 Fixed Base Policy

先用少量 Expert Demonstrations 得到一个基本能够工作的 Base Policy。

$$
\boxed{
\pi_b\text{ 不动}
}
$$

---

### 21.2 Residual Correction

$$
\boxed{
a_t=a_t^b+a_t^r
}
$$

RL 只学习：

$$
a_t^r
$$

不重新学习整个任务。

---

### 21.3 OT = Reward Function

$$
\boxed{
\text{Robot Trajectory}
\xleftrightarrow{OT}
\text{Expert Trajectory}
}
$$

产生：

$$
r_t^{OT}
$$

越像 Expert：

$$
C\downarrow
\Rightarrow
r^{OT}\uparrow
$$

---

### 21.4 Replay Buffer + Off-policy

旧 Robot Rollout 可以反复训练：

$$
\boxed{
\text{一次真实交互}
\rightarrow
\text{多次 Gradient Update}
}
$$

因此提高真实机器人数据利用率。

---

### 21.5 Actor-Critic

Critic：

$$
Q(z,a)
$$

学习：

> 哪个 Action 能得到更高的长期累计 OT Reward？

Actor：

$$
\pi_r
$$

学习：

> Base Action 还应该怎么修？

---

## 22. 脑内最终模型

```text
             ~1 min Expert Demonstrations
                         │
                         ▼
                  Offline Imitation
                         │
                         ▼
                   BC Visual Encoder
                         │
               ┌─────────┴──────────┐
               │                    │
               ▼                    ▼
        Expert Features        Fixed Encoder
               │                    │
               ▼                    │
      Expert Memory / VINN           │
               │                    │
               │             Current Observation
               │                    │
               │                    ▼
               │                   z_t
               │                    │
               └──────────────► Base Policy
                                    │
                                    ▼
                                   a_t^b
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         │                     ▼
                         │             Residual Actor π_r
                         │                     │
                         │                     ▼
                         │                    a_t^r
                         │                     │
                         └────────── + ─────────┘
                                    │
                                    ▼
                              a_t=a_t^b+a_t^r
                                    │
                                    ▼
                                  Robot
                                    │
                                    ▼
                              Robot Rollout
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
             Expert Features                   Robot Features
                   │                                 │
                   └──────────────► OT ◄─────────────┘
                                    │
                                    ▼
                                OT Reward
                                    │
                                    ▼
                               Replay Buffer
                                    │
                                    ▼
                         3-step Actor-Critic
                                    │
                                    ▼
                           Update Residual π_r
                                    │
                                    └──────────────↺
```

最终只需要记住这一条：

$$
\boxed{
o_t
\rightarrow
z_t
\rightarrow
\begin{cases}
\pi_b(z_t)\rightarrow a_t^b\\
\pi_r(z_t,a_t^b)\rightarrow a_t^r
\end{cases}
\rightarrow
a_t=a_t^b+a_t^r
}
$$

训练期间再多一个：

$$
\boxed{
\tau^b
\xleftrightarrow{OT}
\tau^e
\rightarrow
r^{OT}
\rightarrow
\text{Off-policy Actor-Critic}
\rightarrow
\pi_r
}
$$

因此 FISH 的完整核心可以概括为：

> **FISH = 用少量 Expert Demonstrations 建立固定的视觉表征和 Base Policy，用 Optimal Transport 把 Robot Rollout 与 Expert Trajectory 的相似程度转换成逐 timestep Reward，再利用 Replay Buffer 和 Off-policy Actor-Critic 只训练 Residual Policy，使机器人从“基本会模仿”提升为“发生偏差后也能够自主纠正”。**
