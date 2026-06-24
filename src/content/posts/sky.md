---
title: 基于物理的大气渲染技术理论知识
published: 2026-06-18
pinned: true
description: 本文档是一篇有关大气渲染理论的文章
tags: [OpenGL, 计算机图形学]
category: 技术
draft: false
---

# 基于物理的大气渲染技术

<!--more-->
> [!NOTE]
这里仅提供相关的理论知识与公式，至于更加详细的渲染方案（比如公式在着色器中的近似计算，各项参数基于不同引擎与情景的设置，考虑臭氧的影响，优化GPU计算指令等等）建议您亲自阅览这方面相关的着色器文件。
> 

[这里是一份MC光影（Eclipse）的大气渲染着色器文件（有较为详细的注释）](https://github.com/NaieChars/NaieChars_Blog/blob/master/public/sky_render/phisical_sky.glsl)  
[Eclipse_Shader的github官方仓库](https://github.com/Merlin1809/Eclipse-Shader)


在此先叠甲：本人也是才开始学习，如有错误不当之处请多多包涵，欢迎评论区指出。


### 1. 瑞利散射（Rayleigh Scattering）
**条件**：散射粒子尺寸远小于光的波长
**特点**：
    - 散射强度与波长的4次方成反比：$I ∝ 1/λ^4$
    - 前向和后向散射几乎对称（偶极子辐射模式）

**散射截面积：** 物理上描述一个粒子散射光的能力。
在图形学中我们并不直接用绝对物理值，而是使用瑞利散射系数 β_R，它包含密度和截面积，并分解为 RGB 三个通道分别的系数。

通常在 RGB 渲染中，我们将瑞利散射系数定义为海平面处的值（单位：m⁻¹）：
```glsl
β_R = (5.8e-6, 13.5e-6, 33.1e-6)   // R, G, B
```

### 2. 米氏散射（Mie Scattering）
**条件**：粒子尺寸接近或大于波长
    - 散射强度几乎不依赖波长（所有颜色一样被散射）
    - 强烈前向散射：大部分光沿着原方向继续前进，形成太阳周围的光晕。

**散射系数：** 图形学中设为一个常数（或略带波长依赖），通常取：
```glsl
β_M = (2.0e-5, 2.0e-5, 2.0e-5)   // 海平面值
```

### 3. 衰减与光学深度（Optical Depth）
光在介质中传播时，不仅会被散射到视线中（内散射），还会因为散射和吸收而衰减（外散射消光）。
**消光系数（单位 m^-1）**:

$$
\beta_e = \beta_s + \beta_a
$$

这里我们通常忽略吸收，只考虑散射，所以：$\beta_e = \beta_s$

穿过距离 ds 后的强度变化：

$$
\mathrm{d}I = -\beta_e \, I \, \mathrm{d}s
$$

解此微分方程得到 **Beer-Lambert** 定律：

$$
I(s) = I_0 \exp\left(-\int_0^s \beta_e(s')\, \mathrm{d}s'\right)
$$

定义光学深度$\tau$:

$$
\tau = \int_0^s \beta_e(s')\, \mathrm{d}s'
$$

透射率T:入射光经过这段距离后剩下的比例

$$
T(s) = e^{-\tau}
$$

### 4. 大气密度模型 (Density Model)
散射系数与粒子数密度成正比，密度随海拔高度 h（单位 km）指数衰减。
（1）瑞利散射（空气分子）归一化密度：

$$
\rho_R(h) = \exp\!\left(-\frac{h}{H_R}\right)
$$

典型瑞利标高 $H_R ≈ 8 km$ 

（2）米氏散射（气溶胶）归一化密度：

$$
\rho_M(h) = \exp\!\left(-\frac{h}{H_M}\right)
$$

典型米氏标高 $H_M ≈ 1.2 km$

**任意高度 h 处的实际散射系数：**

$$
\beta_R(h) = \beta_{R0} \cdot \rho_R(h), \qquad \beta_M(h) = \beta_{M0} \cdot \rho_M(h)
$$

海平面系数典型值:

$$
\beta_{R0} = (5.8\times10^{-6},\ 13.5\times10^{-6},\ 33.1\times10^{-6}) \quad (\text{R,G,B})
$$

$$
\beta_{M0} = (2.0\times10^{-5},\ 2.0\times10^{-5},\ 2.0\times10^{-5})
$$

### 5. 单次散射模型 (Single Scattering)
**总辐射亮度积分：**

$$
L = \int_{O}^{P} \left[ I_{\text{sun}} \cdot T(\text{Sun} \to X) \cdot \beta_s(X) \cdot P(\theta) \cdot T(X \to O) \right] \mathrm{d}s
$$

其中：视线方向$V$，相机位置$O$，太阳辐照度$I_{sun}$，每个点X的散射贡献包括：太阳到$X$的衰减、&X&处散射、$X$到相机的衰减

$$
L_{\text{total}} = L_R + L_M
$$

$$
L_R = \int_{O}^{P} I_{\text{sun}} \cdot T_R(\text{Sun} \to X) \cdot \beta_R(X) \cdot P_R(\theta) \cdot T_R(X \to O)\, \mathrm{d}s
$$

$$
L_M = \int_{O}^{P} I_{\text{sun}} \cdot T_M(\text{Sun} \to X) \cdot \beta_M(X) \cdot P_M(\theta) \cdot T_M(X \to O)\, \mathrm{d}s
$$

透射率由光学深度给出：

$$
T_R(\text{path}) = \exp\!\left(-\int_{\text{path}} \beta_R(s)\, \mathrm{d}s\right)
$$

$$
T_M(\text{path}) = \exp\!\left(-\int_{\text{path}} \beta_M(s)\, \mathrm{d}s\right)
$$

瑞利相位函数：

$$
P_R(\theta) = \frac{3}{16\pi} (1 + \cos^2\theta)
$$

米氏相位函数（Cornette-Shanks 近似）：

$$
P_M(\theta, g) = \frac{3}{8\pi} \frac{(1 - g^2)(1 + \cos^2\theta)}{(2 + g^2)(1 + g^2 - 2g\cos\theta)^{3/2}}
$$

### 6. Ray Marching 数值积分
将视线分成$N$段（$N=$ 10 ~ 16）
总距离$L$，步长$d_s=L/N$，采样点：

$$
X_i = O + \left(i + \frac{1}{2}\right) \mathrm{d}s \cdot V, \quad i=0,\dots,N-1
$$

每个$X_i$步骤：
- 海拔高度 $h_i=X_i - R_{Earth}$
- 计算 $\beta_R(h_i), \beta_M(h_i)$
- 计算太阳到 $X_i$ 的光学深度 $\tau_{R}^{sun}, \tau_{M}^{sun}$
- 累加相机到 $X_i$的光学深度
- 
$$
\tau_R^{\text{cam}} \mathrel{+}= \beta_R(h_i)\,\mathrm{d}s,\quad \tau_M^{\text{cam}} \mathrel{+}= \beta_M(h_i)\,\mathrm{d}s
$$

散射角余弦 $cos\theta=V\cdot{D}$ （$D$为太阳方向单位向量），累加贡献：

$$
L_R \mathrel{+}= I_{\text{sun}} \cdot T_R^{\text{sun}} \cdot \beta_R \cdot P_R \cdot T_R^{\text{cam}} \cdot \mathrm{d}s
$$

$$
L_M \mathrel{+}= I_{\text{sun}} \cdot T_M^{\text{sun}} \cdot \beta_M \cdot P_M \cdot T_M^{\text{cam}} \cdot \mathrm{d}s
$$

最终颜色：

$$
\mathbf{C} = L_R \cdot \boldsymbol{\beta}_{R0}^{\text{rgb}} + L_M \cdot \boldsymbol{\beta}_{M0}^{\text{rgb}}
$$

### 7. 太阳透射率计算
太阳到散射点 X 的光学深度需沿太阳方向 D 从 X 积分到大气顶层。

简单方法：在主循环内再嵌套一个小的 Ray Marching 循环，步数 M 较小（如 5 步）。

伪公式（单通道）：
$$
\tau_{\text{sun}} = \sum_{j=0}^{M-1} \beta_0 \cdot \exp\!\left(-\frac{h(Y_j)}{H}\right) \cdot \mathrm{d}s_{\text{sun}}
$$

### 8. 地球-大气几何求交
交点公式：

$$
\|O\|^2 + 2 t (O\cdot V) + t^2 = R^2
$$

$$
\Delta = 4\left[ (O\cdot V)^2 - (\|O\|^2 - R^2) \right]
$$

判别式小于0不相交，否则：

$$
t = - (O\cdot V) \pm \frac{\sqrt{\Delta}}{2}
$$
