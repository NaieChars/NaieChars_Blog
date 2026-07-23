---
title: 时间性抗锯齿 TAA（Temporal Anti-Aliasing）
published: 2026-07-23
pinned: true
description: Temporal Anti-Aliasing (TAA) 是当今实时渲染中最核心的抗锯齿技术之一。正如NVIDIA的一篇综述所述，它可被正式定义为 “时间摊销的超采样”（temporally-amortized supersampling）本文主要是以笔记的方式呈现TAA技术，供参考学习。
tags: [OpenGL, Compte Shader, C++, RayTracer]
category: 技术
draft: false
---

## 回顾 SSAA 与 MSAA 
**为什么要进行抗锯齿？**锯齿的来源是在于场景是在三维空间内定义的，它是可连续的状态，而输出的屏幕时是以一个离散的二维数组进行输出的，这便会引发一个问题：我们判断一个点是否被一个像素所覆盖变成了一个单纯“覆盖了”还是“没覆盖”的一个情况，连续的信息丢失，导致了锯齿出现。

### SSAA
超采样（SSAA）便是一个暴力抗锯齿的方法。拿4xSSAA为例，假设我们的输出屏幕大小 800x600，4xSSAA会先渲染到一个 1600x1200 的缓冲纹理上，再通过下采样到 800x600。也就是说我们将一个像素分成了四个小格子（**子像素采样，Sub-pixel**），对这四个小格子分别采样再做平均，便可以得到这个像素最终的输出颜色。这样做是数学上最完美的抗锯齿，只要采样率足够高，可以很好进行抗锯齿，但是缺点也很明显，因为一个像素相比原来的的计算量是4倍，占用的图像缓冲区（render target）也是四倍，在实时渲染中这样的代价高昂。由此我们可以进行 MSAA 采样。

### MSAA
多重采样（MSAA）的优化之处在于，它**只用对每个像素中心进行一次着色计算**，这个颜色结果会被该像素内所有被三角形覆盖的的子像素样本所共享（以 4xMSAA 举例，一个三角形的一条边把这个像素竖着分成左右两半，左边两个子像素被覆盖了，所以像素中心的着色计算结果给这两个被覆盖的子像素，而右边的两个子像素颜色为空，最后再将4个子像素颜色混合，得到该像素最终输出颜色）。这样一个像素只用计算一次，性能非常好。

## 从超采样到时间摊销
TAA 相较于 SSAA，其优雅之处便在于**将单帧内昂贵的多子像素采样成本，分摊到连续的多帧之中**，在每一帧里我们对每一个像素进行一个采样，但采样点在像素内的位置是随机**抖动（Jitter）**的，这样，经过 $N$ 帧的累计，我们便可以获得与单帧 NxSSAA 等效的样本集。

### 采样点抖动：Halton 低差异序列
每一帧的采样点需要在一个像素内均匀分布，才能保证累计效果的品质。在实践中，TAA通常会使用**Halton序列**等低差异序列来生成每一帧的抖动偏移量。  
Halton序列就是：把自然数分别写成2进制、3进制、5进制...，等质数进制，然后把每个数的数字顺序颠倒，当成小数点的后几位。这样算出来的点，能很均匀地铺满整个空间。

> [!NOTE]
> 基本的数学理论可以自行了解，这里简单提一下：Halton序列的本质是一维范德科尔普特序列在多维空间的直积扩展。其核心算子为激进反转，分布特性包括低差异性与渐进式完美分层。
>

在实时图形学中，TAA利用Halton序列生成每一帧的 **2D 抖动偏移量 $j_t$** ，由于TAA仅需二维子像素偏移，工程实现通常仅取前两个素数基数：

$$
\mathbf{j}_t = (\Phi_2(t), \Phi_3(t))
$$

严格的原生Halton序列在基数 $b=2$ 下存在明显的**相关条纹**（Correlated Streaks），因为其在 $x$ 维度上频繁地在0和0.5震荡，若直接用于TAA，可能会导致高频边缘在垂直方向产生“蠕虫”状闪烁。现代引擎往往采用乱序（Scrambled）Halton序列或递推克罗内克（Kronecker）序列来打破二进制的结构性对齐。

核心Halton序列GLSL实现：

```glsl
 // 基数2 激进反转
uint reverseBits32(uint x) 
{
     // 交换相邻位
     x = ((x & 0x55555555u) << 1u) | ((x & 0xAAAAAAAAu) >> 1u);
     // 交换相邻2位
     x = ((x & 0x33333333u) << 2u) | ((x & 0xCCCCCCCCu) >> 2u);
     // 交换相邻4位
     x = ((x & 0x0F0F0F0Fu) << 4u) | ((x & 0xF0F0F0F0u) >> 4u);
     // 交换相邻8位
     x = ((x & 0x00FF00FFu) << 8u) | ((x & 0xFF00FF00u) >> 8u);
     // 交换相邻16位
     x = (x << 16u) | (x >> 16u);
     return x;
 }

 float radicalInverse_VdC2(uint i) 
 {
     // 将反转后的整数映射到 [0, 1) 浮点数 (除以 2^32)
     return float(reverseBits32(i)) * 2.3283064365386963e-10f;
 }

// 2. 基数3 激进反转
 float radicalInverse_VdC3(uint i) 
 {
     const float invBase = 1.0f / 3.0f;
     float invBaseN = invBase;      // 当前位权值: 1/3, 1/9, 1/27 ...
     float res = 0.0f;
     
     while (i > 0u) 
     {
         uint digit = i % 3u;       // 提取最低位（三进制）
         res += float(digit) * invBaseN;
         invBaseN *= invBase;       // 下一位权值缩小3倍
         i /= 3u;                   // 右移一位三进制
     }
     return res;
 }

 vec2 getHaltonJitter(uint frameIndex) 
 {
     // 前两个素数基数: 2 和 3
     float u = radicalInverse_VdC2(frameIndex);
     float v = radicalInverse_VdC3(frameIndex);
     
     // 映射到子像素偏移量：使得采样点以像素中心为原点
     return vec2(u, v) - 0.5f;
 }
```

## 重投影
重投影的核心解决目标是找到**当前帧的某个像素在历史帧中位于何处**。为了找到这个位置，光栅化管线与路径追踪管线有不同的方法。

### 光栅化管线与路径追踪管线求运动向量的区别
#### 数据源头
- **光栅化（顶点驱动）**，运动来自**顶点**，我们只需要维护当前帧和上一帧的 MVP 矩阵，在顶点着色器中分别对同一个顶点进行这两套矩阵变换到 NDC 空间，两者的 NDC 坐标差值就是这个顶点的运动向量，随后硬件光栅化器自动对图元进行重心坐标差值，进而得到每个像素平滑过渡的运动向量。
- **路径追踪（射线驱动）**，路径追踪里的**点源自相机发出射线与图元相交**，我们需要实时知道这个点的世界坐标，随后分别乘上当前帧和上一帧的相机观察投影矩阵 VP 矩阵，计算出两者的屏幕 uv 作差才能得到该像素的运动向量。

#### 计算时域
- **光栅化（延迟计算）**：运动向量通常在**单独的G-Buffer Pass**或**片段着色器末尾**计算。因为顶点着色器已经算好了插值后的NDC，像素着色器只需要做一次简单的减法。
- **路径追踪（即时计算）**：路径追踪的路径是逐像素、逐弹射独立生成的。为了避免额外的显存开销和第二次遍历加速结构（BVH），专业的路径追踪器（如PBRT v4或实时RTX管线）给出建议：在路径追踪主循环击中表面、计算完辐射度（Radiance）后，立刻在同一个着色器内核中利用刚得到的命中点坐标计算运动向量，并将其写入一张独立的运动向量纹理（Motion Vector Texture）。如果是后续通过屏幕坐标反推世界坐标进行计算，会引入巨大的浮点误差。

#### 遮挡与几何不连续性处理
- **光栅化（由深度测试解决）**：当运动向量跨越三角形边界时，光栅化器拥有图元ID信息。如果当前帧的像素在上一帧被前景物体遮挡，光栅化可以通过比较上一帧对应位置的历史深度缓冲区（Depth Buffer）来剔除无效历史。
- **路径追踪（需额外的差异校验）**：路径追踪有几个严重问题：
  - 当前帧某个像素可能命中的是物体 N，但上一帧可能命中的是物体 M
  - 因为路径追踪本质上的采样是随机的，相邻两个像素的命中点可能落在两个完全不连续的几何体上。
这导致单纯依靠运动向量重投影会产生严重的“拖尾鬼影”。因此，路径追踪的TAA必须强制依赖“世界空间位置”和“法线”的差异校验

#### 半透明介质的处理
- **光栅化（只处理直接可见表面）**：运动向量仅代表屏幕表面（第一层可见表面）的移动。对于反射、折射或阴影，光栅化通常放弃计算，仅依赖屏幕空间反射（SSR）来近似，运动向量无法修正反射画面中的像素偏移。
- **路径追踪（妥协定义）**：路径追踪的最终像素颜色由多条光线路径的平均值决定，这便说不清这个像素的运动向量到底属于哪个物体。我们选择妥协方案，只计算“相机可见命中点”（即第一条射线击中的点）的运动向量。

### 历史缓冲与指数平滑
我们通常指维护一个历史颜色缓冲，它存储了上一帧经TAA处理后的最终颜色结果。  
当前帧的最终颜色 \(C_t\)，是由 **当前帧的原始颜色** \(C^{raw}_t\) 和**历史颜色** \(H = C_{t-1}(p')\) 进行加权混合得到的。
最常用的混合方法是**指数平滑滤波（Exponential Smoothing Filter）**：

$$
C_t = \alpha \cdot C^{raw}_t + (1 - \alpha) \cdot H
$$

这里的 \(\alpha\) 是一个在 \((0, 1)\) 范围内的**混合因子**（Blending Factor）。

这个递推公式展开后，可以看到它是一个无限脉冲响应（IIR）滤波器，其历史帧的权重呈指数级衰减：

$$
C_t = \alpha C^{raw}_t + (1 - \alpha) \left[ \alpha C^{raw}_{t-1} + (1 - \alpha) C^{raw}_{t-2} + \cdots \right]
$$

* 当 \(\alpha\) 较大时，当前帧的权重更高，画面响应更快，运动拖影（Ghosting）更少，但抗锯齿效果可能减弱。
* 当 \(\alpha\) 较小时，历史帧的权重更高，抗锯齿效果更好，但运动拖影和模糊会更明显。

### 对抗投影与细节丢失的具体操作
#### 光栅化做法
最常用的技术是**裁剪（Clipping） 或缩回（Clamping）**，
具体做法是，以当前帧像素 \(p\) 为中心，在其周围的一个小邻域（如3x3或5x5）内，计算当前帧原始颜色的均值 \(\mu\) 和方差 \(\sigma^2\)。然后，我们将历史颜色 \(H\) 裁剪到这个由 \([\mu - k\sigma, \mu + k\sigma]\) 定义的范围内（\(k\) 通常取1或2）：

$$
H' = \text{clamp}(H, \mu - k\sigma, \mu + k\sigma)
$$

最后，使用修正后的历史颜色 \(H'\) 来代替 \(H\) 进行混合：

$$
C_t = \alpha \cdot C_t^{raw} + (1 - \alpha) \cdot H'
$$

这个缩回操作能有效防止因历史样本与当前场景差异过大而产生的错误颜色污染画面。

#### 路径追踪做法
路径追踪产生的原始颜色（\(C_t^{raw}\)）服从**高方差、重尾分布**（Heavy-tailed distribution）。
在一个 3x3 邻域内，只要有 1 个像素恰好采样到了高亮光源（Firefly），**该邻域的 \(\sigma\)（标准差）会被拉得极大**，导致 \([\mu - k\sigma, \mu + k\sigma]\) 这个区间宽得离谱，历史裁剪直接失效（等于没剪）。

因此，工业界常采用**极值箱体（AABB / Min-Max）裁剪法**，并引入**颜色空间转换**。

我们做一个简单的事：找出这3x3邻域内（9个像素）像素的RGB三通道各自分别的最低值和最高值。将其围成一个**包围盒 aabb**。现在我们拿到上一帧中心像素的颜色，塞进这个 aabb 里面（即 clamp 做法，比如某个颜色通道范围是 10 ~ 200，我的这个像素该颜色通道值为100，直接放入，此时最大值为200，最小值变为100，如果不在这个范围，就就近取边界）。这里以亮度为例，如果这个中心像素是 **“火蝇”** （纯白坏点，亮度比如10000），我们可以**剔除邻域内的极端离群值（Outlier rejection）**，也就是把它去掉，用周围3x3的平均来填充它，再放入这个aabb。随后我们对当前颜色和历史帧颜色按照一定权重比例混合，比如历史帧取90%，当前帧取10%。

#### 转换到 YCoCg 空间裁剪
直接在RGB空间裁剪会导致**色相偏移（Hue Shift）**，于是我们换一套坐标系。**YCoCg** 就是 RGB 空间经过线性代数里的“基变换（矩阵旋转）”得到的新坐标系。
- Y（Luma，亮度）：管“明暗”，即这张图是亮还是暗（黑白灰）。
- Co（Orange-Blue，橙色-蓝色色度）：管“红蓝偏向”。
- Cg（Green-Magenta，绿色-品红色度）：管“绿紫偏向”。

**只对亮度和色度分别裁剪**可以有效避免色相偏移。

以下是基于 3x3 邻域 AABB + YCoCg 色彩空间 的历史验证函数：

```glsl
// RGB 转 YCoCg 色彩空间
vec3 rgbToYCoCg(vec3 rgb)
{
    return vec3(
        0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b,
        0.5 * rgb.r - 0.5 * rgb.b,
       -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b
    );
}

// YCoCg 转 RGB 逆变换
vec3 ycocgToRgb(vec3 ycocg)
{
    float tmp = ycocg.y - ycocg.z;
    return vec3(
        ycocg.x + ycocg.y,
        ycocg.x + ycocg.z,
        ycocg.x - tmp
    );
}

// 在 YCoCg 空间中提取 3x3 邻域的 AABB（带离群值容差）
void getNeighborhoodAABB(ivec2 p, out vec3 minVal, out vec3 maxVal)
{
    minVal = vec3(1e10);
    maxVal = vec3(-1e10);

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            ivec2 coord = p + ivec2(dx, dy);
            coord = clamp(coord, ivec2(0), ivec2(u_Resolution - 1.0));

            vec3 rawColor = imageLoad(u_CurrentRawColor, coord).rgb;
            vec3 ycocg = rgbToYCoCg(rawColor);

            minVal = min(minVal, ycocg);
            maxVal = max(maxVal, ycocg);
        }
    }

    // 轻微扩展包围盒范围，防止过度裁剪并保留细节
    vec3 ext = (maxVal - minVal) * 0.1;
    minVal -= ext;
    maxVal += ext;
}

// 历史颜色验证主函数：将历史颜色裁剪到局部 AABB 范围内
vec3 validateAndClampHistory(vec3 historyColor, ivec2 pixelCoord)
{
    vec3 minBox, maxBox;
    getNeighborhoodAABB(pixelCoord, minBox, maxBox);

    vec3 histYCoCg = rgbToYCoCg(historyColor);
    vec3 clampedYCoCg = clamp(histYCoCg, minBox, maxBox);

    return ycocgToRgb(clampedYCoCg);
}
```