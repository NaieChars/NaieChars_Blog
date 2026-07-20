---
title: 多分辨率分层 Bloom
published: 2026-07-20
pinned: true
description: 本节主要是针对分层 Bloom 展开，这是现代游戏引擎实现电影级光晕效果的核心架构
tags: [Bloom, 计算机图形学]
category: 技术
draft: false
---

## 单层高斯模糊缺陷
单层高斯模糊无论半径多大只有单一尺度，这会导致：
- 亮部边缘形成均匀厚度的“光圈”，缺乏真实镜头因色散和散射产生的层次过渡。
- 大半径模糊会吞噬高亮区域的纹理细节；小半径模糊则无法产生铺满屏幕的恢弘光晕，导致细节丢失

我们的**分层策略**便是：利用**图像金字塔（Image Pyramid）**，在低分辨率处理大尺度光晕（低频），在高分辨率处理小尺度光晕（高频），最后重组成连续的光晕分布。

## 图像金字塔
### 下采样链（Downsample Chain）
这里其实就是 Mipmap 做法，我们从**全分辨率（设为 Level 0）**开始，不断将分辨率减半（1/2， 1/4， 1/8 ... 直到 1/32 或 1/64），每一层都需要独立存储为一张纹理（或Mipmap切片）

> [!Warning]
> 下采样时**必须进行滤波**（通常用双线性或高斯滤波），而不能直接取整数点，否则会产生严重的摩尔纹和锯齿。
>

### 13-tap 高光提取降采样
常规的双线性下采样会对所有亮度一视同仁，容易让高光在降采样的第一步就被“平均”掉。因此，行业内更偏爱**带亮度倾向性的 13-tap 降采样核。**

该核是覆盖了 5x5 虚拟网格的 13 个特定点（9 个整数偏移点 + 4 个半像素偏移点），并分成了 5 个 2x2 小组，赋予了完全不同的组权重：

```txt
                 X = -1.0    X = -0.5    X = 0.0     X = 0.5     X = 1.0
                 
Y =  1.0         [ k ]       [   ]       [ l ]       [   ]       [ m ]
                                                   
Y =  0.5         [   ]       [ i ]       [   ]       [ j ]       [   ]
                                                   
Y =  0.0         [ f ]       [   ]       [ g ]       [   ]       [ h ]
                                                   
Y = -0.5         [   ]       [ d ]       [   ]       [ e ]       [   ]
                                                   
Y = -1.0         [ a ]       [   ]       [ b ]       [   ]       [ c ]
```

| 小组 | 组成点 | 组总权重 | 单点均摊权重 |
| :--- | :--- | :--- | :--- |
| **中心组** | d, e, i, j | 0.125 | 0.03125 |
| 外围组1 | a, b, g, f | 0.03125 | 0.0078125 |
| 外围组2 | b, c, h, g | 0.03125 | 0.0078125 |
| 外围组3 | f, g, l, k | 0.03125 | 0.0078125 |
| 外围组4 | g, h, m, l | 0.03125 | 0.0078125 |

**为什么这样能抗闪烁并保细节？**
- **传统 4-tap Box 滤波：** 高光点如果在 2x2 采样框的边界上，一旦画面晃动或摄像机微移，高光在下一级 Mip 里会时而出现、时而消失，导致严重的亮度闪烁
- **13-tap 算法：** 通过高权重中心点 g 锁死当前像素亮度，同时因为有了 4 个半偏移点和 4 个边中点，即使高光点有微小位移，它依然会被周围至少 3~4 个采样点捕捉到，

**为什么总权重只有 0.25？**
降采样不只是缩小，它同时模拟着**光线散射能量损失**，确保在上采样后画面能量守恒。


采样函数的具体实现

```glsl
vec3 downsampleBox13(vec2 uv) {
    vec2 t = srcTexelSize;
    vec3 a = texture(srcTex, uv + t * vec2(-1.0, -1.0)).rgb;
    vec3 b = texture(srcTex, uv + t * vec2( 0.0, -1.0)).rgb;
    vec3 c = texture(srcTex, uv + t * vec2( 1.0, -1.0)).rgb;
    vec3 d = texture(srcTex, uv + t * vec2(-0.5, -0.5)).rgb;
    vec3 e = texture(srcTex, uv + t * vec2( 0.5, -0.5)).rgb;
    vec3 f = texture(srcTex, uv + t * vec2(-1.0,  0.0)).rgb;
    vec3 g = texture(srcTex, uv).rgb;
    vec3 h = texture(srcTex, uv + t * vec2( 1.0,  0.0)).rgb;
    vec3 i = texture(srcTex, uv + t * vec2(-0.5,  0.5)).rgb;
    vec3 j = texture(srcTex, uv + t * vec2( 0.5,  0.5)).rgb;
    vec3 k = texture(srcTex, uv + t * vec2(-1.0,  1.0)).rgb;
    vec3 l = texture(srcTex, uv + t * vec2( 0.0,  1.0)).rgb;
    vec3 m = texture(srcTex, uv + t * vec2( 1.0,  1.0)).rgb;

    // 中心2x2权重最高(0.125*4),外围四个2x2区块权重较低(0.03125*4),
    // 这个权重分布是COD论文里验证过的,能有效避免降采样时的"高光闪烁"
    vec3 result = (d + e + i + j) * 0.125;
    result += (a + b + g + f) * 0.03125;
    result += (b + c + h + g) * 0.03125;
    result += (f + g + l + k) * 0.03125;
    result += (g + h + m + l) * 0.03125;
    return result;
}
```

### 上采样链（Upsample Chain）
模糊处理后，需要把小图还原回屏幕大小。通常采用双线性插值（硬件原生支持）或双三次插值（质量略高但开销大）。

### 3x3 tent filter升采样
下图是 `upsampleTent` 在屏幕/纹理空间中，9次采样最终落到的实际物理位置

```txt
                X 方 向 （左 <--------- 右）
                -------------------------------------------
                X偏移 = -1.0      X偏移 = 0.0      X偏移 = +1.0
                (o.z = -dx)      (o.w = 0)        (o.x = +dx)
                
   Y方向        [ 左上角 ]       [ 正上方 ]        [ 右上角 ]
  (上) Y+       uv + o.zy        uv + o.wy        uv + o.xy
  ↑            权重: 1           权重: 2           权重: 1
  
   Y方向        [ 正左方 ]       [ 正中心 ]        [ 正右方 ]
  (中) 0        uv + o.zw           uv            uv + o.xw
                权重: 2           权重: 4           权重: 2
  
   Y方向        [ 左下角 ]       [ 正下方 ]        [ 右下角 ]
  (下) Y-       uv - o.xy        uv - o.wy        uv - o.zy
                权重: 1           权重: 2           权重: 1
```

对低分辨率采样函数

```glsl
vec3 upsampleTent(vec2 uv) {
    vec4 o = srcTexelSize.xyxy * vec4(1.0, 1.0, -1.0, 0.0) * bloomRadius;

    vec3 result  = texture(srcTex, uv - o.xy).rgb;
    result += texture(srcTex, uv - o.wy).rgb * 2.0;
    result += texture(srcTex, uv - o.zy).rgb;
    result += texture(srcTex, uv + o.zw).rgb * 2.0;
    result += texture(srcTex, uv       ).rgb * 4.0;
    result += texture(srcTex, uv + o.xw).rgb * 2.0;
    result += texture(srcTex, uv + o.zy).rgb;
    result += texture(srcTex, uv + o.wy).rgb * 2.0;
    result += texture(srcTex, uv + o.xy).rgb;

    return result / 16.0;
}
```

`o` 是一个偏移向量:
- `o.x = dx * 1.0` = 向右 1 个像素

- `o.y = dy * 1.0` = 向上 1 个像素

- `o.z = dx * (-1.0)` = 向左 1 个像素

- `o.w = dy * 0.0` = 垂直方向 0

## 多频段模糊
### 固定核半径
在提取完亮部过后，我们要对每一层进行单独的高斯模糊，每一层都使用一个极小的固定核，一般取 3 * 3，5 * 5，因为降低了分辨率，使用极小的核就可以覆盖大范围，极大节约了性能，同时还能实现大光晕。
### 模糊迭代
对于最深的那一层，为了获得极致的柔和拖尾，通常会对该层重复执行 2~3 次可分离高斯模糊（即水平+垂直算一次，再做一次水平+垂直）。这能让光晕边缘呈现自然的“高斯钟形”衰减。

## 分层合成
所有层级独立上采样回全分辨率，存储为独立的临时纹理（或利用Mipmap逐级上采样写入）。最终合成公式为一个加权累加和：

$$
Bloom_{\mathit{Final}}(x, y) = \sum_{i=0}^{N} \left( \mathit{UpSample}(\mathit{Blurred}_i) \times W_i \right)
$$

其中 $W_i$ 是每一层独立的控制参数。

我们在CPU端可以这样实现，这里是基于opengl实现

```cpp
// ---------- 阶段3: 逐级降采样,构建mip链 ----------
        downsampleProgram.use();
        GLuint srcTex = brightTexture;
        int srcW = SCR_WIDTH, srcH = SCR_HEIGHT;

        for (int level = 1; level <= NUM_BLOOM_MIPS; level++) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, srcTex);
            downsampleProgram.setInt("srcTex", 0);
            downsampleProgram.setVec2("srcTexelSize", 1.0f / srcW, 1.0f / srcH);
            glBindImageTexture(0, bloomMips[level], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

            GLuint gx = (mipWidths[level] + 15) / 16;
            GLuint gy = (mipHeights[level] + 15) / 16;
            glDispatchCompute(gx, gy, 1);
            // 注意:这里barrier要同时包含TEXTURE_FETCH,因为下一轮要用sampler2D采样这次写入的结果,
            // 光有IMAGE_ACCESS_BARRIER只保证image2D读写顺序,不保证texture()采样能看到最新数据
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

            srcTex = bloomMips[level];
            srcW = mipWidths[level];
            srcH = mipHeights[level];
        }

        // ---------- 阶段4: 逐级升采样叠加,从最小一级往回叠加,最终叠回brightTexture本身 ----------
        upsampleProgram.use();
        for (int level = NUM_BLOOM_MIPS; level >= 1; level--) {
            GLuint dstTex = (level == 1) ? brightTexture : bloomMips[level - 1];
            int dstW = (level == 1) ? SCR_WIDTH  : mipWidths[level - 1];
            int dstH = (level == 1) ? SCR_HEIGHT : mipHeights[level - 1];

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, bloomMips[level]);
            upsampleProgram.setInt("srcTex", 0);
            upsampleProgram.setVec2("srcTexelSize", 1.0f / mipWidths[level], 1.0f / mipHeights[level]);
            upsampleProgram.setFloat("bloomRadius", 1.0f); // 想要更柔和的大范围光晕可以调到1.5~2.0试试

            glBindImageTexture(0, dstTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

            GLuint gx = (dstW + 15) / 16;
            GLuint gy = (dstH + 15) / 16;
            glDispatchCompute(gx, gy, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        }
```


## 合成中的参数把握
#### 强度衰减曲线
  - 在高斯模糊中的 `bloomIntensity` 可以根据分层自行调整，靠近高分辨率的层级强度设定较低（避免过度锐利刺眼），靠近低分辨率的层级强度设定较高（制造恢弘氛围）。
  - 也可以自行设计 S 型曲线
#### 色调偏移与色差
    - 真实镜头的折射率随波长变化（红光偏外，蓝光偏内），对最深层增加红色和绿色的权重，让光晕外围呈现暖黄色；对内层增加蓝色权重。
#### 尺寸缩放
  - 即使层级固定（如 1/16），合成时也可以乘以一个缩放系数。如果调大某一层的权重或拉长其曝光时间，等效于物理上的“过曝膨胀”。

## 更好的混合策略
#### 频域交叉淡入（Cross-fading）
在相邻层之间加入平滑的权重过渡。即并不是简单地分频，而是让高频层也带有一点低频成分，低频层也保留一点高频边缘。这通常通过对上采样后的图像进行重叠加权平均实现。

#### 高亮保留（Specular Preservation）
在合成最终颜色时，不要直接把模糊结果加到原图上。先进的做法是：只叠加模糊层的增量部分，且与源图像的亮度进行相对比较。这样做能避免暗部区域被Bloom染脏，保持画面通透。

