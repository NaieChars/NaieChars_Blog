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

在实际的工程中，密度求和**通常用 Ploy6 核**，数学形式如下（三维空间中）：

$$
W_{\text{poly6}}(r, h) = \frac{315}{64\pi h^9} \begin{cases} (h^2 - r^2)^3, & 0 \leq r \leq h \\ 0, & r > h \end{cases}
$$

前面的系数为归一化系数，保证空间积分值为1。

为什么叫 Ploy6 核，因为将多项式展开得到的 r 最高次为6


代码实现：

```cpp
__device__ __forceinline__ float poly6(float r, float h) 
{
    if (r < 0.0f || r > h) return 0.0f;
    float h2 = h * h;
    float r2 = r * r;
    float diff = h2 - r2;
    // 315 / (64 * pi * h^9)
    const float coef = 315.0f / (64.0f * 3.14159265358979f);
    float h9 = h2 * h2 * h2 * h2 * h;
    return (coef / h9) * diff * diff * diff;
}
```

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


这里可以从积分角度理解为什么梯度能作用到kernel  
因为积分变量是 $x'$，而我们要求导的是观察点 $x$：

$$
A(x) = \int A(x')W(x - x', h) dx'
$$
对 $x$ 求梯度：

$$
\nabla A(x) = \int A(x')\nabla W(x - x', h) dx'
$$

注意：

* $A(x')$ 与 $x$ 无关，因此可以看作常数；
* 真正依赖 $x$ 的只有 Kernel。

由于上面公式有一个问题，i能推动j，但是j不一定推动i，（核函数满足反对称性，用 $W_{ij} = -W_{ji}$，代入进得到相互作用力并不相等，矛盾。于是Monaghan提出了对称形式：

$$
\mathbf{f}_i^{pressure} = -\sum_{j} m_j \frac{p_i + p_j}{2\rho_j} \nabla W(\mathbf{r}_i - \mathbf{r}_j, h)
$$

**为什么梯度核不能用Poly6？** 因为在 $r\rightarrow 0$ s时，ploy6和梯度趋于0，这显然是不对的，因此**Spiky核**专门为解决这个问题设计：

$$
W_{spiky}(r, h) = \frac{15}{\pi h^6} (h - r)^3，\quad \nabla W_{spiky} = -\frac{45}{\pi h^6} (h - r)^2 \cdot \frac{\mathbf{r}}{|\mathbf{r}|}
$$

它的梯度在 $r \to 0$ 时趋于 $-\frac{45}{\pi h^6} h^2 \cdot \hat{\mathbf{r}}$（有限且是这个核函数梯度绝对值最大的地方），随 $r$ 增大单调衰减到 0，正好符合越近排斥力越强的物理直觉。

代码里需要注意的边界情况：$r \to 0$ 时方向向量 $\mathbf{r}/|\mathbf{r}|$ 会除零，必须加epsilon保护（可以用 `r < 1e-6f` 直接跳过，反正粒子重合的极端情况下这一项对应情况本来也没有稳定意义）

# 粘性
**粘性描述的是流体内部不同速度区域之间相互阻碍运动的能力（压力倾向于恢复密度，粘性倾向于恢复速度）**  

Navier-Stokes 方程写作：

$$
\frac{D\mathbf{v}}{Dt} = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{v} + \mathbf{g}
$$

我们现在只关注第二项：

$$
\boxed{\nu\nabla^2\mathbf{v}}
$$

其中 $\nu$ 被称为**Kinematic Viscosity（运动粘度）**，单位 $m^2/s$；$\nabla^2$ 是拉普拉斯算子（**某一点和周围平均值相比偏离了多少**）

粘性力在SPH里的标准形式为：

$$
\mathbf{f}_i^{viscosity} = \mu \sum_{j} m_j \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_j} \nabla^2 W_{viscosity}(r, h)
$$

关键点在于用的是Müller论文里专门构造的**粘度核**：

$$
\nabla^2 W_{viscosity}(r, h) = \frac{45}{\pi h^6} (h - r)
$$

> [!question]
> **为什么除以 $\rho_j$**：
> - 直觉上理解成"密度越大的邻居，代表它周围物质更'厚重'，对中心粒子的粘滞拖拽应该按它自身单位体积折算（加权），除以 $\rho_j$ 是把邻居粒子的质量贡献换算成'单位密度下'的速度影响"  
> 
> **$\mu$该取多大**：
> - 真实水的动态粘度系数只有约 $0.001 Pa·s$，如果直接拿这个物理真值代入，阻尼效果基本没有。因为真实分子粘度在这种粗粒度的离散化下完全不够用，图形学里常用的 $\mu$ 取值在0.1~5这个量级，本质上是**数值粘度补偿离散化损失**
> 

# 物理弹簧阻尼边界处理
**真实流体随着靠近墙壁，压力逐渐增大、逐渐减速**。  

这里先不讲刚体耦合（后面提到），先从简单易懂的弹簧阻尼模型开始：把边界当成一个连续的排斥力场，而不是离散事件。给每面墙定义一个边界层，厚度就取光滑核半径。数学化为：

$$
\mathbf{a}_{boundary} = k \cdot (h - d) \cdot \hat{\mathbf{n}} \quad (d < h)
$$

其中 $d$ 是粒子沿墙面法线方向到墙的距离，$\hat{\mathbf{n}}$ 是指向流体内部的法线，$k$ 是弹簧刚度。$d$ 越小（离墙越近/穿透越深），排斥力越大——这是个**标准的弹簧模型**。

只有弹簧还不够，纯弹簧会让粒子在墙边持续弹跳（能量守恒，永远荡不平）。所以**加一个阻尼项**，只在粒子朝墙运动时起作用（避免把已经在远离墙的粒子也粘住）：

$$
\mathbf{a}_{damping} = -c \cdot (\mathbf{v} \cdot \hat{\mathbf{n}}) \cdot \hat{\mathbf{n}} \quad (\mathbf{v} \cdot \hat{\mathbf{n}} < 0)
$$

这两项加起来本质上是一个**弹簧-阻尼系统（spring-damper）**，这和 RESTITUTION 最大的区别在于：它的弹性是 $k$ 和 $c$ 两个参数配合的结果，而不是一个孤立的乘法系数，我们用用阻尼比这个物理量描述：

$$
\zeta = \frac{c}{2\sqrt{k}}
$$

* $\zeta < 1$（欠阻尼）：粒子碰撞后还会有轻微回弹，类似 RESTITUTION>0 的效果
* $\zeta = 1$（临界阻尼）：最快趋于静止、不回弹，是水这种粘性流体撞击硬地面比较接近的观感
* $\zeta > 1$（过阻尼）：趋于静止的过程被拖慢，会显得“糊”

# 表面张力
## Color Field（颜色场）方法

Müller et al. 2003原始SPH流体论文里表面张力的做法比较巧妙，核心思路是：先构造一个标量场，**衡量这个位置周围有多像是流体内部**：

$$
c_i = \sum_{j} \frac{m_j}{\rho_j} W_{poly6}(r_{ij}, h)
$$

这个场在流体内部近似为常数，只有在表面附近才会因为一侧邻居缺失而产生梯度，所以 $\nabla c_i$ 天然指向缺少邻居粒子的方向，**也就是表面的法线方向**。

表面张力的驱动力来自于曲率（表面越弯曲，收缩趋势越强），曲率对应到这个颜色场上就是拉普拉斯算子：

$$
\mathbf{f}_i^{surface} = -\sigma \cdot \nabla^2 c_i \cdot \frac{\nabla c_i}{|\nabla c_i|}
$$

$\sigma$是表面张力系数，$\nabla c_i / |\nabla c_i|$是归一化后的表面法线方向，$\nabla^2 c_i$（标量）决定了这个力沿法线方向是“往外顶”还是“往内收”，大小对应曲率大小。

然后很关键的一步是加上**阈值判断**：流体内部的粒子 $\nabla c_i$ 理论上应该趋于**零向量**，但SPH是离散求和，实际算出来的值不会精确为0，而是在0附近有噪声波动。如果不做任何判断，直接对着一个模长接近0的向量做归一化（$\mathbf{n} / |\mathbf{n}|$），会导致内部粒子的**力方向完全由数值噪声决定**，这是表面张力实现里最典型的不稳定来源，具体表现是内部粒子莫名其妙地被推来推去、整个流体看起来在“抽搐”。

标准做法是设一个阈值 $\epsilon$，只有 $|\nabla c_i| > \epsilon$（说明这个粒子确实在表面附近，梯度是真实信号而不是噪声）才施加表面张力，否则直接跳过：

$$
\mathbf{f}_i^{surface} = \begin{cases} 
-\sigma \cdot \nabla^2 c_i \cdot \frac{\nabla c_i}{|\nabla c_i|} & |\nabla c_i| > \epsilon \\ 
0 & \text{otherwise} 
\end{cases}
$$

这个 $\epsilon$ 具体取多少，理论上算不准（取决于粒子分辨率、$h$、$mass$ 这些参数的组合）