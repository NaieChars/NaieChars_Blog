---
title: Smoothed Particle Hydrodynamics （SPH流体模拟）理论学习笔记
published: 2026-07-31
pinned: true
description: 记录了学习 SPH 流体模拟的一些知识，包括理论定义、数学推导、算法实现等等
tags: [SPH流体模拟, 计算机图形学]
category: 技术
draft: false
---

```txt
Smoothed Particle Hydrodynamics
|
├── 1. Introduction
|      ├── Eulerian vs Lagrangian
|      ├── Mesh-based vs Meshless
|      └── SPH Overview
|
├── 2. Mathematical Foundation
|      ├── Continuum Mechanics
|      ├── Conservation Laws
|      ├── Navier-Stokes Equation
|      └── Lagrangian Fluid Description
|
├── 3. SPH Approximation Theory
|      ├── Kernel Approximation
|      ├── Particle Approximation
|      ├── Kernel Properties
|      └── Gradient/Laplacian Approximation
|
├── 4. Weakly Compressible SPH
|      ├── Density Estimation
|      ├── Equation of State
|      ├── Pressure Force
|      ├── Viscosity
|      └── Time Integration
|
├── 5. Incompressible SPH
|      ├── PCISPH
|      ├── IISPH
|      └── DFSPH
|
├── 6. Boundary Handling
|
├── 7. Neighbor Search
|
├── 8. GPU Implementation
|
└── 9. Advanced Topics
       ├── Rigid-Fluid Coupling
       ├── Multi-phase Flow
       └── Rendering
```


# 流体模拟概览
流体模拟的目标是求解流体运动规律。宏观尺度下，流体运动由：质量守恒、动量守恒、能量守恒来描述。而对于不可压缩液体诸如水、蜂蜜等等，通常只需要关注质量与动量守恒即可。

## 两种流体的描述
### 欧拉描述
**核心定义：固定空间位置，观察流过该处的流体状态随时间的变化**  
**速度场 $u(x, t)$**:这是一个场函数，x是空间坐标，t是时间，我们只关心在这个空间位置，任意时刻的流速是多少，而不关心是哪个粒子在该位置。因此不难得出，任意时刻的流体属性都可以写成位置和时间的函数

> [!NOTE]
> **什么是场**？所谓场就是**空间中的每一个位置，都对应一个物理量**

### 拉格朗日描述
**核心定义：跟踪流体微元自身的运动**  
我们需要给每一个流体微元一个身份标签，这个标签通常就是其在初始时刻所处位置 $\text{X}$，$\text{X}$ 被称为物质坐标或拉格朗日坐标。用 $\text{X}$ 可以描述粒子轨迹 $x(t)$ 的一族曲线：

$$
\mathbf{x} = \chi(\mathbf{X}, t)
$$
相应的对其求关于t的偏导即可得到速度与加速度随位置的变化。

## 基于欧拉网格的流体模拟
将空间划分为均匀固定的网格，每一个 cell 都可以监测并保存该时刻网格内流体的密度速度压力等等，然后求解 **Navier-Stokes 方程**。优点在于数学理论成熟，精度高，工程应用广泛。缺点在于：自由页面很难处理。

### 动量守恒 —— Navier-Stokes 方程
**核心思想：流体的动量变化率等于作用在它上面所有力的总和（牛顿第二定律）**。流体的所受力分为两类：
- 体积力：作用在流体微团整体上的远程力，如重力 $\rho g$
- 表面力：由周围流体或壁面通过接触施加的力，包括压力（正应力）和粘性应力（切应力），统一用应力张量 $\sigma$ 表示

N-S方程：

$$
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}
$$

* $\frac{\partial \mathbf{u}}{\partial t}$：当地加速度，固定点上速度随时间的变化
* $\mathbf{u} \cdot \nabla \mathbf{u}$：对流加速度，因流体运动而穿越速度梯度产生的加速度
* $-\frac{1}{\rho} \nabla p$：压力梯度项，由高压指向低压，是加速度。
* $\nu \nabla^2 \mathbf{u}$：粘性扩散项，动量从高速区向低速区扩散，使速度分布均匀化
* $\mathbf{f}$：体积力，如重力加速度 $\mathbf{g}$

> [!note]
> 详细的公式推导这里就不做展示了，这里涉及到专业的流体力学领域。

## 光滑粒子流体模拟
SPH 的**核心思想**可以概括为：**将连续介质表示为有限数量粒子，并利用核函数对粒子邻域进行加权插值，从而近似连续场，再将连续场离散化求和得到计算机能够计算的公式**。
SPH 三个关键词：Meshless、Lagrangian、Smoothed
### Meshless
直观体现拉格朗日描述，无网格，完美解决自由液面飞溅的场景。
### Lagrangian
追踪流体微元思想的数字化实现，每个粒子存储的信息包括位置、速度、质量、密度与压力，求解 N-S 方程进而得到粒子的连续运动
### Smoothed
连续流体通过粒子集合近似，真实流体 $f(x)$ 可近似为：

$$
f(x) \approx \sum_{j} f_j W(x - x_j, h)
$$

任何一点的物理量，都是由其周围邻居粒子的对应值加权平均得来。权重 $W(x - x_j, h)$ 就是**核函数（Kernel Function）**，权重曲线中间高，两边低（离中心粒子越近，权重越大）

# SPH 数学原理
这一节主要是讲为什么一堆离散粒子可以描述连续流体。  
这里给出一个数学工具：**Dirac Delta函数**，记作 $\delta(x)$  

$\delta(x)$ 不是传统意义上的函数，而是一个**广义函数（分布）**，直观上的物理意义是用来描述一个理想质点的密度分布。假设一个质量为 m = 1 的质点，体积为零地集中在原点 x=0 处。那么这团物质的空间密度分布就是$\delta(x)$，它具有两个重要性质：
- 在除原点处，$delta(x)=0$
- 积分 $\int_{-\infty}^{+\infty} \delta(x) dx = 1$


**核心功能：筛选性质**：

$$
\int_{-\infty}^{+\infty} f(x) \delta(x - a) dx = f(a)
$$

这相当于一个精确采样操作：它把函数 $f(x)$ 在 $x=a$ 这一点的值 $f(a)$  筛了出来，其他所有点对积分结果的贡献都为零。  
但是问题随之而来，Dirac Delta函数无限高无限窄，计算机无法表示，于是 SPH 不用 $\delta$，而用一个平滑函数 $W$ 去近似它，它便是之前提及的核函数

于是上一节公式：

$$
f(x) = \int f(x')\delta(x - x')dx'
$$

便可以**核近似（Kernel Approximation）** 为：

$$
f(x) \approx \int f(x')W(x - x', h)dx'
$$

# 核函数
SPH 的所有公式，本质上都是在问**怎样设计**一个好的 kernel  
上一节最后的公式由核近似得到，那么为何可以用 kernel 代替 $\delta$？  
答案是 kernel 不需要核 $\delta$ 完全一模一样，只需要满足 $h\rightarrow 0$ 时，$W(\text{x}, h)\rightarrow \delta(\text{x})$，因此 kernel 可以看作是 **Dirac δ 的数值近似（Numerical Approximation）**

物理意义：**粒子的影响力分布函数（Influence Function），本质就是一个 距离-权重 的映射**，因此其数学上可以表示为：

$$
W = W(r, h)
$$

其中：$r = \|\mathbf{x} - \mathbf{x}'\|$，$h$ 为**支撑半径**，即定义 kernel 有效范围。$h$值小，计算快，噪声大；值大，计算慢，表面容易过度平滑，因此，$h$ 决定了**精度、稳定性和性能之间的平衡**。

### Kernel 的四个基本性质
- Normalization（归一化）：$\int_{\Omega} W(\mathbf{x}, h)d\mathbf{x} = 1$，这是**零阶一致性（Zeroth-order Consistency）**，它保证SPH 至少能正确表示一个常数。
- Compact Support（紧支撑）：**必须满足当 $r>h$ 时，$W(r,h)=0$**，只计算中心粒子周围几十个粒子，复杂度近似 $O(N)$。
- Positivity（非负性）：权重必须为正很好理解
- Symmetry（对称性）：kernel 只与距离有关，又由于相互作用力大小相等，所以天然对称

# 粒子近似
这里和黎曼积分是完全一样的，都是离散化思想（在CG中涉及到积分计算基本到最后都是离散化求和）    
在 SPH 中，**粒子代表一个微小流体体积（Fluid Volume Element）**。我们将上上节的核近似公式离散化为求和得：

$$
A_i \approx \sum_{j} A_j \frac{m_j}{\rho_j} W(\mathbf{x} - \mathbf{x}_j, h)
$$

这是许多论文里的**SPH母公式**的标准写法，其中 $\frac{m_j}{\rho_j}$表示粒子所代表的微元体积

# Weakly Compressible SPH（弱可压缩 SPH）
SPH 流体有两种主流的实现方法，这里先讲更简单易懂的**Weakly Compressible SPH (WCSPH)**  
它的**核心思想是：我不强求密度绝对不变，而是允许它有一点点（比如 1%）的可压缩性，从而极大简化计算**

## 密度估算
为何首先计算密度？当我们将流体微元化后，只知道每个粒子的质量速度与位置，压力通常不知道，而是由密度来求得的，从而**密度决定压力，压力决定压力力，压力力决定粒子的运动**。

对于 SPH 母公式：

$$
A_i \approx \sum_{j} A_j \frac{m_j}{\rho_j} W_{ij}
$$

我们要求密度，于是令 $A=\rho$，可得：

$$
\rho_i \approx \sum_{j} m_j \frac{\rho_j}{\rho_j} W_{ij}=\sum_{j} m_j W_{ij}
$$

这就是 **SPH 最经典的密度公式**，可以理解为：**中心粒子的密度等于所有邻居粒子的质量，按照 Kernel 权重加权后的总和**

> [!question]
> 既然水几乎不可压缩，为什么还要每一帧重新计算密度？  
> 如果仍然强制认为密度恒定，就无法判断哪里发生了压缩、哪里发生了拉伸，也就无法计算恢复流体体积的压力。因此，即使目标是模拟不可压缩水，我们仍然需要估计当前密度，然后通过压力或约束把它拉回到**静止密度**附近。  
> 后面实现 SPH 的主流方法还有 PCISPH / IISPH / DFSPH 等，是通过迭代求解，让密度尽可能保持在目标值附近。核心目的都是如何让密度误差更小

# 状态方程
前面我们只在讨论一种表示流体的方法，这一节会说明为什么密度变化会产生压力

**什么是状态方程（EOS，Equation of State）？**   
其目的是建立压力与密度的关系：$p=f(\rho)$，像高中学的理想气体方程 $PV=nRT$ 便是状态方程

## Tait Equation
WCSPH 几乎都采用 Tait Equation，这是整个WCSPH最经典的公式之一：

$$
p = B \left[ \left( \frac{\rho}{\rho_0} \right)^\gamma - 1 \right]
$$

- $\rho$：当前密度，由上一节估算而得
- $p_0$：静止密度，水通常为 $1000kg/m^3$
- $\gamma$：一般取 $\gamma=7$，这是一个经验数值，来源于 Tait 方程
- $B$：体积模量常数，衡量液体难不难压缩，值越大，流体越难压缩

### B 的计算
B 过小会导致水持续压缩造成海绵一样的效果，B 过大会剧烈震荡发生数值爆炸，因此对 B 必须进行合理的选择  
在实际工程中，我们通常会设置一个**人工声速 $c_0$ 来反推 B**：

$$
B = \frac{\rho_0 c_0^2}{\gamma}
$$

为什么用人工声速？敲击一下水，压力以声速传播，水中的声速大概 1480 m/s，但是如果我们取 1480，时间步长必须极小，GPU跑不动，所以我们人为降低声速，在保证模拟稳定的同时还能极大加快计算效率。

# 压力

这里先澄清一个概念，真正推动粒子的是**压力变化而不是压力本身**。  
数学上，压力变化率写成：$\nabla p$，表示**压力增加最快的方向**，而粒子总是从高压流向低压，于是前面需要加一个符号 $-\nabla p$

至于如何去求梯度，可以对梯度进行插值，把梯度作用到 kernel 得到：

$$
\nabla A_i = \sum_{j} m_j \frac{A_j}{\rho_j} \nabla W_{ij}
$$

**为什么对 kernel 求梯度**，因为在 SPH 中唯一连续的且与位置相关的就只有 kernel 了，因此梯度只能作用于 kernel

> [!note]
> 这里可以从积分角度理解为什么梯度能作用到kernel  
> 因为积分变量是 $x'$，而我们要求导的是观察点 $x$：
>
> $$
A(x) = \int A(x')W(x - x', h) dx'
$$
> 对 $x$ 求梯度：
>
> $$
\nabla A(x) = \int A(x')\nabla W(x - x', h) dx'
$$
> 
> 注意：
> 
> * $A(x')$ 与 $x$ 无关，因此可以看作常数；
> * 真正依赖 $x$ 的只有 Kernel。
>

由于上面公式有一个问题，i能推动j，但是j不一定推动i，（核函数满足反对称性，用 $W_{ij} = -W_{ji}$，代入进得到相互作用力并不相等，矛盾。于是Monaghan提出了对称形式：

$$
\mathbf{a}_i = - \sum_{j} m_j \left( \frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right) \nabla W_{ij}
$$