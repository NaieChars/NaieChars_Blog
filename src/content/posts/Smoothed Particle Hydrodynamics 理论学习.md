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


## 流体模拟概览
流体模拟的目标是求解流体运动规律。宏观尺度下，流体运动由：质量守恒、动量守恒、能量守恒来描述。而对于不可压缩液体诸如水、蜂蜜等等，通常只需要关注质量与动量守恒即可。

### 两种流体的描述
#### 欧拉描述
**核心定义：固定空间位置，观察流过该处的流体状态随时间的变化**  
**速度场 $u(x, t)$**:这是一个场函数，x是空间坐标，t是时间，我们只关心在这个空间位置，任意时刻的流速是多少，而不关心是哪个粒子在该位置。因此不难得出，任意时刻的流体属性都可以写成位置和时间的函数

> [!NOTE]
> **什么是场**？所谓场就是**空间中的每一个位置，都对应一个物理量**

#### 拉格朗日描述
**核心定义：跟踪流体微元自身的运动**  
我们需要给每一个流体微元一个身份标签，这个标签通常就是其在初始时刻所处位置 $\text{X}$，$\text{X}$ 被称为物质坐标或拉格朗日坐标。用 $\text{X}$ 可以描述粒子轨迹 $x(t)$ 的一族曲线：

$$
\mathbf{x} = \chi(\mathbf{X}, t)
$$
相应的对其求关于t的偏导即可得到速度与加速度随位置的变化。

### 基于欧拉网格的流体模拟
将空间划分为均匀固定的网格，每一个 cell 都可以监测并保存该时刻网格内流体的密度速度压力等等，然后求解 **Navier-Stokes 方程**。优点在于数学理论成熟，精度高，工程应用广泛。缺点在于：自由页面很难处理。

#### 动量守恒 —— Navier-Stokes 方程
**核心思想：流体的动量变化率等于作用在它上面所有力的总和（牛顿第二定律）**。流体的所受力分为两类：
- 体积力：作用在流体微团整体上的远程力，如重力 $\rho g$
- 表面力：由周围流体或壁面通过接触施加的力，包括压力（正应力）和粘性应力（切应力），统一用应力张量 $\sigma$ 表示

N-S方程：

$$
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}
$$

* $\frac{\partial \mathbf{u}}{\partial t}$：当地加速度，固定点上速度随时间的变化
* $\mathbf{u} \cdot \nabla \mathbf{u}$：对流加速度，因流体运动而穿越速度梯度产生的加速度
* $-\frac{1}{\rho} \nabla p$：压力梯度项，由高压指向低压的驱动力
* $\nu \nabla^2 \mathbf{u}$：粘性扩散项，动量从高速区向低速区扩散，使速度分布均匀化
* $\mathbf{f}$：体积力，如重力加速度 $\mathbf{g}$

> [!note]
> 详细的公式推导这里就不做展示了，这里涉及到专业的流体力学领域。

### 光滑粒子流体模拟
SPH 的**核心思想**可以概括为：**将连续介质表示为有限数量粒子，并利用核函数对粒子邻域进行加权插值，从而近似连续场，再将连续场离散化求和得到计算机能够计算的公式**。
SPH 三个关键词：Meshless、Lagrangian、Smoothed
#### Meshless
直观体现拉格朗日描述，无网格，完美解决自由液面飞溅的场景。
#### Lagrangian
追踪流体微元思想的数字化实现，每个粒子存储的信息包括位置、速度、质量、密度与压力，求解 N-S 方程进而得到粒子的连续运动
#### Smoothed
连续流体通过粒子集合近似，真实流体 $f(x)$ 可近似为：

$$
f(x) \approx \sum_{j} f_j W(x - x_j, h)
$$

任何一点的物理量，都是由其周围邻居粒子的对应值加权平均得来。权重 $W(x - x_j, h)$ 就是**核函数（Kernel Function）**，权重曲线中间高，两边低（离中心粒子越近，权重越大）

## SPH 数学原理
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

## 核函数
SPH 的所有公式，本质上都是在问**怎样设计**一个好的 kernel  
上一节最后的公式由核近似得到，那么为何可以用 kernel 代替 $\delta$？  
答案是 kernel 不需要核 $\delta$ 完全一模一样，只需要满足 $h\rightarrow 0$ 时，$W(\text{x}, h)\rightarrow \delta(\text{x})$，因此 kernel 可以看作是 **Dirac δ 的数值近似（Numerical Approximation）**

物理意义：**粒子的影响力分布函数（Influence Function），本质就是一个 距离-权重 的映射**，因此其数学上可以表示为：

$$
W = W(r, h)
$$

其中：$r = \|\mathbf{x} - \mathbf{x}'\|$，$h$ 为**支撑半径**，即定义 kernel 有效范围。$h$值小，计算快，噪声大；值大，计算慢，表面容易过度平滑，因此，$h$ 决定了**精度、稳定性和性能之间的平衡**。

#### Kernel 的四个基本性质
- Normalization（归一化）：$\int_{\Omega} W(\mathbf{x}, h)d\mathbf{x} = 1$，这是**零阶一致性（Zeroth-order Consistency）**，它保证SPH 至少能正确表示一个常数。
- Compact Support（紧支撑）：**必须满足当 $r>h$ 时，$W(r,h)=0$**，只计算中心粒子周围几十个粒子，复杂度近似 $O(N)$。
- Positivity（非负性）：权重必须为正很好理解
- Symmetry（对称性）：kernel 只与距离有关，又由于相互作用力大小相等，所以天然对称

## 粒子近似