---
title: 体积渲染（Volume Rendering）与次表面散射（SSS）的随机游走理论
published: 2026-07-21
pinned: true
description: 本节详细介绍了体积渲染与次表面散射的随机游走，主要用于路径追踪管线，有别与传统光栅化管线的次表面散射实现，比如 Disney 的 Burley SSS
tags: [体积渲染, 次表面散射, 计算机图形学]
category: 技术
draft: false
---

## 体积介质（Participating Media）
当光线接触云，烟雾，玉石等介质后，并不会马上反射，而是进入这些介质内部，在内部不断散射，或被吸收，或从另一端射出，我们在外面看起来这个介质便是半透明的。这种光在介质内部传播的过程就叫做**Participating Media（参与介质）**

## 次表面散射（Subsurface Scattering，SSS）
### 随机游走基本概念
传统方法（如偶极子模型）会用扩散方程近似这个整体效果，而随机游走则是**直接模拟每个散射步骤：**

1. 光线以一定角度折射进入物体内部
2. 根据介质的散射系数和吸收系数，随机采样一个自由程距离，这便是光子沿当前方向的移动距离
3. 到达最终距离后，光子发生散射，方向按介质的**相位函数**随机改变。同时根据吸收概率，光子可能被吸收终止（通常用俄罗斯轮盘赌实现）。
4. 重复2，3步骤，直到光子被吸收或从另一个表面折射出去。

###  消光系数
我们将两个使辐射度变小的效应合在一起，用消光系数 (extinction coefficient) 来描述，通常记作 $\sigma_t$，它是吸收系数 $\sigma_a$ 和散射系数 $\sigma_s$ 的和：

$$
\sigma_t(\boldsymbol{x}) = \sigma_a(\boldsymbol{x}) + \sigma_s(\boldsymbol{x})
$$

表示：**单位长度内光损失的强弱。**

### 散射概率
散射概率 $\text{albedo}$ 通常用 $\alpha$ 表示: $\alpha = \frac{\sigma_s}{\sigma_t}$，表示**碰撞以后继续散射的概率。**

### 光子碰撞位置满足指数分布
在参与介质中，光线沿着方向 $ω$ 传播时，由于吸收和外散射，辐射亮度会不断衰减。这个衰减规律满足**比尔–朗伯定律**：

$$
L(t) = L(0) \cdot \exp \left( - \int_{0}^{t} \sigma_t (\mathbf{x} + s\boldsymbol{\omega}) \, ds \right)
$$

- $\mathbf{x} + s\boldsymbol{\omega}$：光线参数方程，表示进行到距离 $s$ 的位置。
- $\sigma_t$：该位置的消光系数
- $t$：从起点起总的传播距离
- $L(0)$：在起点处、沿方向 $ω$ 的入射辐射亮度
- &L(t)&：传播距离 $t$ 后，沿同一方向的出射辐射亮度

如果介质是均匀的，消光系数在空间中是**常数**：

$$
\sigma_t(\mathbf{x}) = \sigma_t \quad (\text{常数})
$$

则积分简化为如下形式

$$
\int_{0}^{t} \sigma_t \, ds = \sigma_t \cdot t
$$

于是**比尔-朗伯定律成为（指数衰减形式）：**

$$
L(t) = L(0) e^{-\sigma_t t}
$$

从物理上讲，一个光子从起点出发，在距离 \(t\) 处首次发生碰撞的概率密度函数 \(p(t)\)，正是把这个**衰减过程解释为概率分布**：

$$
p(t) = \sigma_t e^{-\sigma_t t}
$$

这就是指数分布。它的累积分布函数是 \(F(t) = 1 - e^{-\sigma_t t}\)。

用逆变换采样很容易生成符合这一分布的**随机自由程**，这便是我们代码里要求的 $t$：

$$
t = - \frac{\ln(1 - \xi)}{\sigma_t} \quad \text{或等价地} \quad t = - \frac{\ln \xi}{\sigma_t}
$$

其中 \(\xi\) 是 \([0,1)\) 上均匀分布的随机数。


### 蒙特卡洛方法
这里跟路径追踪里的蒙特卡洛采样一样，不计算所有的可能路径，我们只需要：
- 按照特定的概率分布（如指数分布采自由程，相位函数采散射方向）随机生成一条路径；
- 为这条路径计算一个权重（辐射亮度除以采样概率）；
- 重复生成多条路径，取平均值。

只要概率密度非零地覆盖所有有贡献的区域，估计就**是无偏的**。

### Henyey-Greenstein 相函数

在参与介质内部没有表面，所以我们用 **Phase Function（相函数）** 来决定特定方向来的光散射后前往哪个方向，它只描述概率，可以理解为 **给定入射方向，出射方向的概率密度**

Henyey-Greenstein (HG) 是渲染中最经典的相函数：

$$
p(\cos\theta) = \frac{1}{4\pi} \frac{1-g^2}{(1+g^2-2g\cos\theta)^{3/2}}
$$

其中 \(\cos\theta = \omega \cdot \omega'\)，\(\theta\) 是散射角。

**参数g的含义：**
- $g = 0$：各向同性，所有方向的概率完全相等
- $g > 0$：散射角小，光继续沿原方向附近前进，是**前向散射**。皮肤、牛奶、大部分生物组织、云层都是前向散射。

所以整个路径的构建就变成了：指数分布给步长 + 相函数给方向，周而复始。

> [!NOTE]
> 想要了解更加详细的数学推导以及更加精确的理论，可以查阅 Physically Based Rendering (PBRT) 中关于 BSSRDF 和随机游走章节，11章还有16.4节
>

次表面散射的随机游走模型在 OpenGL Compute Shader 里的工程实现，属于光线递进循环函数的分支（依据不同材质类型）

```glsl
else if (mat.type == 4)
        {
            // ---- Isotropic 体积介质 -------
            vec3 unitDir = normalize(rayDir);
            float cosTheta = min(dot(-unitDir, rec.normal), 1.0);
            float iorBoundary = max(mat.iorB, 1.01); // 边界折射率,防止<=1导致数值异常

            // Schlick近似,和你Day26给dielectric写的公式完全一样,这里复用同样的思路
            float r0 = (1.0 - iorBoundary) / (1.0 + iorBoundary);
            r0 = r0 * r0;
            float reflectance = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);

            if (randFloat(rngState) < reflectance)
            {
                // ---- 表面反射----
                vec3 reflected = reflect(unitDir, rec.normal);
                rayOrigin = rec.point + rec.normal * 0.001;
                rayDir = reflected;
            }
            else
            {
                // ---- 折射进入介质:先算出折射方向,再用这个方向开始内部随机游走 ----
                vec3 refractedDir = refract(unitDir, rec.normal, 1.0 / iorBoundary);

                Sphere mediumSphere = spheres[rec.sphereIndex];
                float sigma_t = max(mat.ir, 0.001);
                float scatterAlbedo = clamp(mat.fuzz, 0.0, 1.0);
                float g = clamp(mat.iorR, -0.99, 0.99);

                vec3 mediumPos = rec.point;
                vec3 mediumDir = refractedDir; //用折射方向而不是原始rayDir作为介质内传播方向

                float distToExit = sphereFarRoot(mediumSphere, mediumPos, mediumDir);
                if (distToExit < 0.0) distToExit = mediumSphere.radius * 2.0;

                bool exitedMedium = false;
                bool absorbed = false;
                int maxMediumBounces = 24;

                for (int mBounce = 0; mBounce < maxMediumBounces; mBounce++)
                {
                    float freeFlight = -log(1.0 - randFloat(rngState)) / sigma_t;
                    if (freeFlight >= distToExit)
                    {
                        mediumPos = mediumPos + mediumDir * distToExit;
                        exitedMedium = true;
                        break;
                    }

                    mediumPos = mediumPos + mediumDir * freeFlight;
                    distToExit -= freeFlight;
                    attenuation *= mat.albedo;

                    if (randFloat(rngState) > scatterAlbedo)
                    {
                        absorbed = true;
                        break;
                    }

                    float u1 = randFloat(rngState);
                    float u2 = randFloat(rngState);
                    float cosThetaP;
                    if (abs(g) < 1e-3)
                    {
                        cosThetaP = 1.0 - 2.0 * u1;
                    }
                    else
                    {
                        float sqrTerm = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1);
                        cosThetaP = (1.0 + g * g - sqrTerm * sqrTerm) / (2.0 * g);
                    }
                    float sinThetaP = sqrt(max(0.0, 1.0 - cosThetaP * cosThetaP));
                    float phi = 2.0 * PI * u2;

                    ONB scatterBasis = buildONB(mediumDir);
                    vec3 localDir = vec3(sinThetaP * cos(phi), sinThetaP * sin(phi), cosThetaP);
                    mediumDir = normalize(onbLocal(scatterBasis, localDir));

                    distToExit = sphereFarRoot(mediumSphere, mediumPos, mediumDir);
                    if (distToExit < 0.0) distToExit = mediumSphere.radius * 2.0;
                }

                if (absorbed || !exitedMedium)
                {
                    validBounce = false;
                }
                else
                {
                    rayOrigin = mediumPos + mediumDir * 0.001;
                    rayDir = mediumDir;
                }
            }
        }
```