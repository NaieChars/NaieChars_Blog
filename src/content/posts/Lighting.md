---
title: OpenGL 学习笔记
published: 2026-06-10
pinned: true
description: 本文档为个人学习笔记，内容基于 LearnOpenGL CN 各章节整理归纳，并附加了一些工程渲染技巧
tags: [Markdown, Blogging, OpenGL]
category: 技术
image: ./cover_image/MC_forest.png
draft: false
---

# 基于OpenGL的光照描述与高级渲染

---
注：本文档为个人学习笔记，内容基于 LearnOpenGL CN 各章节整理归纳，并附上了我的一些个人实战经验以及常用引擎级渲染技巧。特此声明：部分图片为 LearnOpenGL CN 原站截图，所有文字及代码均为手工编写与总结，少量内容可能存在拼写错误，还请谅解。

**<span style="font-size: 24px;">目录</span>**
- [基于OpenGL的光照描述与高级渲染](#基于opengl的光照描述与高级渲染)
  - [1. Phong Lighting Model](#1-phong-lighting-model)
  - [2. Blinn-Phong Lighting Model](#2-blinn-phong-lighting-model)
  - [3. Gamma Correction](#3-gamma-correction)
  - [4. Shadow Mapping](#4-shadow-mapping)
  - [5. CSM](#5-csm)
  - [6. Omnidirectional Shadow Maps](#6-omnidirectional-shadow-maps)
  - [7. Normal Mapping](#7-normal-mapping)
  - [8. Parallax Mapping](#8-parallax-mapping)
  - [9. HDR](#9-hdr)
  - [10. Bloom](#10-bloom)
  - [11. Multi-scale Bloom, Karis Average And Soft Threshold](#11-multi-scale-bloom-karis-average-and-soft-threshold)
  - [12. Deferred Shading](#12-deferred-shading)
  - [13. SSAO](#13-ssao)
---

## 1. Phong Lighting Model
**冯氏光照模型**主要由三部分组成，Ambient(环境光)，diffuse(漫反射)，specular(镜面光照)
1. **Ambient**
由于我们目前不采用 global illumination 算法，冯氏模型下的环境光分量是由环境光强度因子决定的
```glsl
.fs
float ambientStrength = 0.05;
vec3 ambient = ambientStrength * lightColor;
```
2. **diffuse**
diffuse 的实现需要 Normal Vector & lightPos** & FragPos
- **Normal Vector** 随顶点传入着色器
- **lightPos** 通过 unform 传入着色器
- **FragPos** 首先在 .vs 中顶点数据乘 model 矩阵得到，从模型空间转换到世界空间进行光照计算
```glsl
.vs
FragPos = vec3(model * vec4(aPos, 1.0));
```

- >vec4(aPos, 1.0) 将三维坐标转换为齐次坐标，允许平移变换到世界空间
- >最后输出类型为 vec3，舍弃最后的 w 分量，（ w 分量一般用于延迟渲染），此处舍弃以节约带宽

下一步，标准化法线和入射光线向量，计算光照对片段的实际漫反射影响因子，以及最后的漫反射分量
```glsl
.fs
float diff = max(dot(normal, lightDir), 0.0);
vec3 diffuse = diff * lightColor;
```
1. **specular**
基于世界空间的处理：
我们需要观察者坐标得到 viewDir，用 reflect 函数得到基于 normal 向量的反射光线
```glsl
.fs
float specularStrength = 0.5;
vec3 viewDir = noramlize(viewPos - FragPos);
vec3 reflectDir = reflect(-lightDir, normal);

float spec = pow(max(dot(reflectDir, viewDir), 0.0), shininess);
vec3 specular = specularStrength * spec * lightColor;
```
- >确定镜面反射强度大小，不要过度高光
- >取点乘非负结果后取 shininess 次幂，shininess 即高光反光度
- >基于以上内容，我们可以将对应的 ambientStrength, specularStrength, diffuseStrength 作为向量写入光的结构体材质里面（不同光三种属性参数不同）

4. **点光源（Point Light）**
- 衰减公式（同样满足 Gamma 衰减）：

$$
F_{att} = \frac{1.0}{K_c + K_l\cdot d + K_q\cdot d^2}
$$

- 数据选择：[LearnintOpenGL CN官网数据](https://learnopengl-cn.github.io/02%20Lighting/05%20Light%20casters/)
- 代码实现
```glsl
.fs
struct Light {
    // 光源基本信息
    //...
    float constant;
    float linear;
    float quadratic;
};

float distance    = length(light.position - FragPos);
float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
// 最后乘上影响因子
ambient  *= attenuation; 
diffuse  *= attenuation;
specular *= attenuation;
```
- >对照公式即可理解

1. **多光源**
   
多光源处理本质即将不同光照的（ambient + diffuse + specular）进行叠加

---

## 2. Blinn-Phong Lighting Model
基于 phong 模型的局限性分析：对于 $Specular = (\mathbf{R}\cdot\mathbf{V})^{shi}$ 当 shininess 很小的时候，导致衰减很慢，高光过于明亮

- **Blinn-Phong**
  
脱离反射向量，采取标准化半程向量：

$$
\vec{H} = \frac{\vec{L} + \vec{V}}{\|\vec{L} + \vec{V}\|}
$$

```glsl
.fs
    vec3 halfwayDir = normalize(lightDir + viewDir);
```
- >后续镜面光分量的计算改变点只有对表面法线和半程向量的一次约束点乘

---

## 3. Gamma Correction
由于显示器物理特性，导致其具有显示器 Gamma，（通常为 Gamma2.2），这与人眼所察觉颜色亮度吻合。但这种显示器非线性映射不利于我们对颜色进行线性操作，因此我们引入 **Gamma 校正**
<p align="center">
  <img src="/markdown_picture/md_lighting/Gamma.png" width="300">
</p>

- >点线代表线性颜色/亮度值（Gamma 为 1），实线代表显示器显示颜色，虚线代表 Gamma 校正曲线。例如我们将颜色 (0.5, 0.0, 0.0) 翻倍至 (1.0, 0.0, 0.0)，在显示器上便是从 (0.218, 0.0, 0.0) 翻倍至 1，翻了4.5倍！

- **SRGB纹理**
所有创建的纹理（albedo / diffuse）都是源于 SRGB 空间的纹理，因此在处理这些颜色时，应该将其转换为线性空间（以避免两次 Gamma校正）。当创建一个纹理时，通常使用 GL_SRGB / GL_SRGB_ALPHA 内置纹理格式以自动将颜色转换到线性空间中
```cpp
.cpp
glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
// 当纹理要引入 alpha 元素时，用GL_SRGB_ALPHA
```

后面阶段即引入后期处理，在后处理的四边形上应用一次 Gamma Correction
```glsl
.fs
color = pow(color, vec3(1.0 / 2.2));
```
- >在所有后处理（HDR, Bloom 等等）最后加上 Gamma 校正
---

## 4. Shadow Mapping

**阴影映射（Shadow Mapping）**：通过变换矩阵P将视角转换到光源视角，通过对片段进行采样获得深度值并保存在 **深度贴图（depth map）** 或 **阴影贴图 （shadow map）** 之中

**<span style="font-size: 24px;">a. Rendering Shadow</span>**

- **（1）创建深度缓冲对象与深度纹理**
```cpp
.cpp
unsigned int depthMapFBO;
glGenFramebuffers(1, &depthMapFBO);

const unsigned int SHADOW_WIDTH = 4096, SHADOW_HEIGHT = 4096; // 阴影贴图分辨率
unsigned int depthMap;
glGenTextures(1, &depthMap);
glBindTexture(GL_TEXTURE_2D, depthMap);
glTexImage2D(
    GL_TEXTURE_2D, 
    0, 
    GL_DEPTH_COMPONENT,   // 我们只关心深度值，需要将纹理格式指定为GL_DEPTH_COMPONENT
    SHADOW_WIDTH, 
    SHADOW_HEIGHT, 
    0, 
    GL_DEPTH_COMPONENT, 
    GL_FLOAT, 
    NULL);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAEST);
·····
```
- >我们只关心深度值，需要将纹理格式指定为GL_DEPTH_COMPONENT
- >SHADOW_WIDTH = 4096, SHADOW_HEIGHT = 4096 为分辨率
- **（2）绑定深度纹理作为帧缓冲的深度缓冲**
```cpp
.cpp
glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO); 
glFramebufferTexture2D(
    GL_FRAMEBUFFER, 
    GL_DEPTH-ATTACHMENT, 
    GL_TEXTURE_2D, 
    depthMap, 
    0); 
glDrawBuffer(GL_NONE);   <=
glReadBuffer(GL_NONE);   <=
glBindFramebuffer(GL_FRAMEBUFFER, 0);
```
- >此处不需要颜色缓冲，然而不包含颜色缓冲的帧缓冲不完整，因此我们显式地告诉 OpenGL 不渲染任何颜色数据

- **（3）CPU 内渲染逻辑**
```cpp
.cpp
// 1. 首选渲染深度贴图
glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);      // 切换视口！
glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);     // 切换到离屏渲染（FBO）
glClear(GL_DEPTH_BUFFER_BIT);                       // 清除残留深度值
ConfigureShaderAndMatrices();                       // 变换视角到光源视角
RenderScene();
glBindFramebuffer(GL_FRAMEBUFFER, 0);               // 切换回默认缓冲（屏幕）
// 2. 像往常一样渲染场景，但这次使用深度贴图
glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);            // 切换回视口！
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
ConfigureShaderAndMatrices();
glBindTexture(GL_TEXTURE_2D, depthMap);  <=
RenderScene();
```
在 Shader里面，每个片段变换到光源空间，拿到当前深度值，比对 shadowMap 判断是否在阴影中。渲染两次 RenderScene() 目的：第一次记录深度，第二次使用深度

- **（4）光源空间的变换**
  
平行光，我们将对光源采取**正交投影矩阵**，透视不会进行改变
```cpp
.cpp
float near_plane = 1,0f, far_plane = 7.5f;  // 相机范围影响阴影精度，范围越大，精度越差
glm::mat4 lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);   // ortho 决定裁剪框的大小，并将画面压缩进NDC
glm::mat4 lightView = glm::lookAt(
    glm::vec3(-2.0f, 4.0f, -1.0f),  // 光源位置
    glm::vec3( 0.0f, 0.0f,  0.0f),  // 光源看向
    glm::vec3( 0.0f, 1.0f,  0.0f)   // 光源上方向
);

glm::mat4 lightSpaceMatrix = lightProjection * lightView;

shadowShader.use();
// 将矩阵传入 shader
glUniformMtrix4fv(
    lightSpaceMatrixLocation,           // 变量位置
    1,                                  // 传入矩阵个数
    GL_FALSE,                           // 是否转置
    glm::value_ptr(lightSpaceMatrix)    // 数据指针
);
```
- >矩阵的计算对GPU来说开销极大，通常在cpu里处理好再通过 uniform 传入 shader

- **（5）着色器内阴影渲染**
  
```glsl
.vs
vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(vs_out.FragPos, 1.0);
```
```glsl
.fs
// shadow 的值为1代表在阴影中
uniform ShadowMap;

main..
float shadow = ShadowCaculation(fs_in.FragPosLightSpace);
vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;
FragColor = vec4(lighting, 1.0);
```
判断一个片段是否在阴影中需要在裁切空间中进行，因此需要将光源空间的片段位置转换为裁切空间的标准化设备坐标。当我们输出一个裁剪空间顶点到 gl_Postion 时，OpenGL 自动进行透视除法，但 FragPosLightSpace 并不会通过 gl_Position 传到片段着色器，需要我们手动透视除法

```glsl
.fs
float ShadowCaculation(vec4 fragPosLightSpace)
{
    // 执行透视除法
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // 变换到[0, 1]范围
    projCoords = projCoords * 0.5 + 0.5;     
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float shadow = currentDepth > closestDepth ? 1.0 : 0.0;

    return shadow;
}
```
- >因为来自深度贴图的深度范围是[0, 1]，我们打算用 projCoords 从深度贴图采样，所以将NDC坐标变换到[0, 1]
- >深度贴图保存的是每个像素方向上，距离光源最近的那个片段深度。texture 从深度贴图里采样当前 uv 坐标（即该像素方向上）的最近表面深度，并存入R通道
- >projCoords.z 存的是当前片段到光源的距离

**<span style="font-size: 24px;">b. Optimize Shadow Mapping</span>**

- **（1）Shadow Acne**
  
由于阴影精度以及深度贴图分辨率等问题，当我们比较没在阴影中的片段时，就等于是将该片段与自己比较，于是由于细小误差导致出现摩尔纹，可以通过 **Shadow Bias** 解决
```glsl
.fs
float bias = max(0.05 * (1 - dot(normal, lightDir)), 0.005);
float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
```
- >动态 bias，光照与表面越斜，bias 越大，最小临界为 0.005，即垂直表面照射时仍保证有 bias

- **（2）Peter Panning**
  
由于存在 bias，导致贴地物体会出现**阴影悬浮**，此时可以通过 **front face culling 正面剔除** 来解决，应用后可以有效减小 bias 偏移量，从而减小 Peter Panning
由于我们的深度值表示 near -> far 之间的相对位置，尽管不渲染正面，深度值由于物体厚度相对于整个深度空间占比很小，所以数值上变化同样很小，恰巧满足我们的需求
我们在阴影贴图生成阶段进行正面剔除
```cpp
.cpp
glCullFace(GL_FRONT);
// ...渲染场景到 DepthMap
glCullFace(GL_BACK); // 设置回原来的面剔除
```

- **（3）过采样**
  
由于我们深度贴图的环绕方式默认为 GL_REPEAT，这将导致在光的视锥范围以外的区域会被判定为处于阴影中
```cpp
.cpp
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
```
- >纹理坐标越界过后，直接采取边界颜色（GL_CLAMP_BORDER）
- >最后两行设置边界颜色，（这里的颜色其实是深度），在超出范围后，shader 里面的 depth 变量直接返回 depth = 1.0 ，代表远处都在光照里，前方并没有阴影遮挡

同样，如果点在光源的 far plane 后方，即 projCoords.z > 1.0，会被误判成处于阴影中，此时我们需要解决深度超出光源范围
```glsl
.fs
float ShadowCalculation(vec4 fragPosLightSpace)
{
    [...]
    if(projCoords.z > 1.0)
        shadow = 0.0;

    return shadow;
}
```

基于上面两者的联系：
uv 越界，返回 depth = 1.0; z（深度）越界，强制 shadow = 0

- **（4）PCF（percentage-closer-filtering）**
  
**百分比渐进滤波**用来处理阴影的锯齿问题，通过多次采样深度贴图，每一次采样的纹理坐标稍有不同，再取平均得相对柔和的阴影
```glsl
.fs
float shadow = 0;
vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
for (int x = -1; x <= 1; ++x0)
{
    for (int y = -1; y <= 1; ++y)
    {
        float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
        shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
    }
}
shadow /= 9.0;
```
- >textureSize 即画面分辨率，texelSize 即一个像素大小
- >图示在 3 * 3 范围内采样 9 次，最后取平均

- **PCSS（Percentage-Closer Soft Shadows）**
  
**PCSS** 是一种基于 Shadow Mapping 的软阴影算法，通过估计遮挡物与受光点之间的相对距离，动态调整阴影滤波半径，从而模拟**半影（penumbra）效果**
```glsl
.fs
// 首先寻找遮光物 Blocker
for (PCF)
{
    for (PCF)
    {
        float sampleDepth = textrue(shadowMap, projCoords.xy + vec2(x,y)*texelSize).r;
        if (sampleDepth < currentDepth)
        {
            // 找到遮光片段
            avgBlockerDepth += sampleDepth;
            blockerCount++;
        }
    }
}
if (blockerCount == 0) return 0.0; // 没有片段在阴影中，shadow返回0
avgBlockerDepth /= blockerCount;   // 得到平均遮挡物相对光源的深度

// Penumbra 半影估计
float penumbra = (currentDepth - avgBlockerDepth) / avgBlokerDepth;

float radius = penumbra * 5.0; // (PCF可变半径)

for (PCF)
{
    for (PCF)
    {
        vec2 offset = vec2(x,y) * texelSize * radius;
        float pcfDepth = texture(shadowMap, projCoords.xy + offset).r;

        shadow += currentDepth > pcfDepth ? 1.0 : 0.0;
        samples++;
    }
}
shadow /= samples;
return shadow;
```
- > 在计算 Blocker 与当前片段的距离时，再除以一个 Blocker 平均深度使其变成相对距离而非绝对距离

---

## 5. CSM
**级联阴影贴图（Cascaded Shadow Maps）** 是一种将**摄像机视锥体（View Frustum）按深度划分为多个区间（Cascades）**，并为每个区间生成阴影贴图的技术，以此来提示在不同距离下的采样精度分布
- **（1）视锥切分（Split）**
  
我们常用**混合分割（Practical Split Scheme）**

$$
d_i = \lambda\cdot d_{i}^{log} + (1 - \lambda)\cdot d_{i}^{uniform}
$$

  - 均匀分割：
  
  $$
    d_{i}^{uniform} = n + (f - n)\frac{i}{k}
  $$

  - 对数分割：
  
  $$
    d_{i}^{log} = n \cdot (\frac{f}{n})^{\frac{i}{k}}
  $$

  - $\lambda$ 一般取 0.5 ~ 0.9

- **（2）计算每个 Cascade 的光源矩阵**
对于每个 cascade：
    - 找到这个视锥体的8个角点（world space）
        - 使用相机参数（FOV，aspect，near_i，far_i）计算顶点
    - 将角点转换到光源视空间（light view space）
        - 用光源的 View 矩阵 lightViewMatrix
    - 构建包围盒（AABB）
        - 找 (minX, maxX, minY, maxY, minZ, maxZ)
    - 基于该包围盒构建正交投影矩阵
        ```cpp
        .cpp
        ortho(
            minX, maxX,
            minY, maxY,
            minZ, maxZ
        )
        ```
- **（3）构建 shadowShader**
  
大致思路：在片段着色器中，根据片元在视空间中的深度值（即相机角度），与 cascade 分割平面比较，确定所属 cascade，并选择对应的 shadow map
```glsl
.fs
uniform float cascadeSplit[NUM_CASCADES];       // 分割距离
uniform mat4 lightMatrices[NUM_CASCADES];       // 多个阴影矩阵
uniform sampler2D shadowMaps[NUM_CASCADES];     // 多张阴影贴图

// 判断层级
float depth = abs(viewSpacePos.z);
int cascadeIndex = 0;
for (int i = 0; i < NUM_CASCADES; i++)
{
    if (depth < cascadeSplits[i])
    {
        cascadeIndex = i;
        break;
    }
}

// 之后阴影计算同前，只是根据 Index 用数组里的数据
```
- >abs函数表示取绝对值，因为在视空间里 z 是负值，我们只关心深度

- **（4）过渡混合（Cascade Blending）**
避免在分界处的不自然，我们假定一个 blend range 通过混合两张阴影贴图实现过渡混合
- 定义过渡区间
- 计算权重，即在过渡区间的距离，使变化速率呈现正态分布的感觉
```glsl
.fs
// 算当前层
    ...
float shadow0 = ...

if (i == NUM_CASCADES - 1) return shadow0; // 边界不参与迷糊

if (depth > split - range)
{
    float shadow1 = ...
    float weight = smoothstep(split - range, split + range, depth);
    float shadow = mix(shadow0, shadow1, weight);
}
return shadow0;
```
- >smoothstep内置函数，平滑的 0 -> 1 的过渡
- >range 一般取 （split * 0.05 ~ 0.15）

- **（5） Texel Snapping**
当相机移动时，阴影矩阵会变化，导致阴影在屏幕上抖动，本质原因在于阴影贴图的像素没有对齐世界，现在使 cascade 中心点强行对齐格子
- 计算世界单位对应一个texel多大
```cpp
.cpp
worldUnitsPerTexel = (maxX - minX) / shadowMapResolution   
center.x = floor(center.x / worldUnitsPerTexel) * worldUnitsPerTexel;
center.y = floor(center.y / worldUnitsPerTexel) * worldUnitsPerTexel;
```
- >阴影范围宽度除以阴影贴图分辨率
- >floor() 向下取整
- >center = 当前 cascade 盒子的中心点

---
## 6. Omnidirectional Shadow Maps
本小节暂且不写，目前不想写阴影了

---
## 7. Normal Mapping
**法线贴图（Normal Mapping）** 是一种将向量的xyz作为rgb存储的2D纹理。这将是一种偏蓝的纹理，因为所有的法线都偏向z轴（0， 0， 1），这是一种偏蓝的颜色
```glsl
.fs
uniform sampler2D normalMap;

void main()
{
    // 从法线贴图[0， 1]范围获取法线
    normal = texture(normalMap, fs_in.TexCoords).rgb;
    // 将法线向量重新映射到[-1， 1]
    normal = normalize(normal * 2.0 - 1.0);

    //....光照处理
}
```
- >注意应用法线贴图的时候一定要解压法线向量，即重新映射

- **（1）切线空间（tangent spcce）**
切线空间是位于三角形表面之上的空间，法线相对于单个三角形的局部坐标系。法线贴图中的法线向量定义在切线空间中，由此我们需要 **TBN 矩阵**把法线从切线空间变换到不同空间。

<p align="center">
  <img src="/markdown_picture/md_lighting/TBN_Caculation.png" width="400">
</p>

如图，我们要求**切线（tangent），副切线（Bitangent）**，本质是把纹理空间（UV）的方向映射到模型空间（3D）里。
$P_1$, $P_2$, $P_3$ 为三个点，边 $E_2$ 与纹理坐标的差$\Delta U_2$、$\Delta V_2$构成一个三角形，$\Delta U_2$与切线向量$T$方向相同，$\Delta V_2$与副切线向量$B$方向相同，，所以我们可以对$E$进行线性组合：

$$
\begin{aligned}
(E_{1x}, E_{1y}, E_{1z}) &= \Delta U_1 (T_x, T_y, T_z) + \Delta V_1 (B_x, B_y, B_z) \\
(E_{2x}, E_{2y}, E_{2z}) &= \Delta U_2 (T_x, T_y, T_z) + \Delta V_2 (B_x, B_y, B_z)
\end{aligned}
$$

即：

$$
\begin{bmatrix}
E_{1x} & E_{1y} & E_{1z} \\
E_{2x} & E_{2y} & E_{2z}
\end{bmatrix}=
\begin{bmatrix}
\Delta U_1 & \Delta V_1 \\
\Delta U_2 & \Delta V_2
\end{bmatrix}
\begin{bmatrix}
T_x & T_y & T_z \\
B_x & B_y & B_z
\end{bmatrix}
$$

由此我们可以求得切线和副切线的坐标：

$$
\begin{bmatrix}
T_x & T_y & T_z \\
B_x & B_y & B_z
\end{bmatrix}=
\frac{1}{\Delta U_1 \Delta V_2 - \Delta U_2 \Delta V_1}
\begin{bmatrix}
\Delta V_2 & -\Delta V_1 \\
-\Delta U_2 & \Delta U_1
\end{bmatrix}
\begin{bmatrix}
E_{1x} & E_{1y} & E_{1z} \\
E_{2x} & E_{2y} & E_{2z}
\end{bmatrix}
$$

现在我们可以手动计算切线与副切线：
```cpp
.cpp
// 先计算第一个三角形的边和deltaUV坐标
glm::vec3 edge1 = pos2 - pos1;
glm::vec3 edge2 = pos3 - pos1;
glm::vec2 deltaUV1 = uv2 - uv1;
glm::vec2 deltaUV2 = uv3 - uv1;

float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
tangent1.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
tangent1.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
tangent1.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
tangent1 = glm::normalize(tangent1);

bitangent1.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
bitangent1.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
bitangent1.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
bitangent1 = glm::normalize(bitangent1);  

[...] // 对平面的第二个三角形采用类似步骤计算切线和副切线
```
- >其实这里可以不用计算副切线，因为我们将法向量和切线传入着色器后，完全可以通过叉乘得到副切线，不过这里展示常规的做法只是让我们知道怎么求切线与副切线
- >最后还要对结果向量进行标准化！
- 算出切线后，我们可以加在顶点坐标后面，当做一个顶点着色器属性
```glsl
.vs
#version 330 core
...
layout (location = 3) in vec3 tangent;
```
在顶点着色器 main 函数里创建 TBN 矩阵
```glsl
.vs
void main()
{
    [...]
    vec3 T = normalize(vec3(model * vec4(tangent, 0.0)));
    vec3 N = normalize(vec3(model * vec4(normal, 0.0)));
    vec3 B = normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);
}
```
先将向量变换到世界空间，然后把相应向量放入mat3构造器即可创建TBN。这里有个细节，如果模型被旋转放缩，那就不能用 model 矩阵而改用法线矩阵:
代码实现如下，通过**法线矩阵 normalMatrix** 纠正旋转拉伸错误。然后，由于 T 可能不完全垂直于 N （数值误差 + 插值），我们还需要加上 **Gram-Schmidt正交化**：
```glsl
.vs
// 标准工程级写法
mat3 normalMatrix = transpose(inverse(mat3(model)));

vec3 T = normalize(normalMatrix * tangent);
vec3 N = normalize(normalMatrix * normal);
// 正交化（很关键！）
T = normalize(T - dot(T, N) * N);
vec3 B = cross(N, T);
```
- >dot(T, N) * N表示 T 在 N 上的分量

在我们有了 TBN 矩阵过后，我们将其传入片段着色器，将其左乘到对应法线得到转换到世界空间的正确法向量：
```glsl
.fs
normal = texture(normalMap, fs_in.TexCoords).rgb;
normal = normalize(normal * 2.0 - 1.0);   
normal = normalize(fs_in.TBN * normal);
```
- >最后所有计算都将在世界空间展开
---

## 8. Parallax Mapping
视差贴图通过 **高度图（Height Map）** 来模拟表面深度，高度图即灰度图，白色代表高，黑代表低。

- **Parallax Mapping**

由于纹理贴在物体表面，我们必须将 viewDir 乘上 viewDir 矩阵，以此转换到切线空间
- 核心思想，根据视线方向计算偏移 UV

$$
UV^{'} = UV + \frac{viewDir_{xy}}{viewDir_z}\cdot height\cdot scale
$$

$viewDir$：视线方向（在切线空间！）

$\frac{viewDir_{xy}}{viewDir_z}$：表示斜着看的程度

$height$：高度图采样值

$scale$：高度缩放（控制强度）

这里详细说说为什么 UV 偏移会导致看起来高度有变化：
由于偏移多少是由视角和高度决定的，高度越大，偏移越多（对照公式），高度越小，偏移越小，于是产生了相对错位，这种相对错位就是深度的来源，本质上只是一种视觉偏差。

<p align="center">
  <img src="/markdown_picture/md_lighting/Parallax_mapping.png" width="500">
</p>

具体代码实现，先看顶点着色器：
```glsl
.vs
[...] // TBN 矩阵
vs_out.TangentViewPos = TBN * viewPos;
vs_out.TangentFragPos = TBN * vs_out.FragPos;
```
- >首先要把片段位置，观察者位置变换到切线空间并传递给片段着色器
再看片段着色器：
```glsl
.fs
uniform float height_scale;
vec2 ParallaxMapping(vec2 texCoords, vec3 TangentViewDir);

void main()
{
    // 切线空间的 viewDir （只用于视差）
    vec3 TangentViewDir = normalize(fs_in.TangentViewPos - fs_in.TangentFragPos);
    // 计算偏移后的UV
    vec2 texCoords = parallaxMapping(fs_in.TexCoords, viewDir);

    // 通过新的纹理坐标进行所有贴图的采样
    vec3 diffuse = texture(diffuseMap, texCoords);
    vec3 normal = texture(normalMap, texCoords);
    normal = normalize(normal * 2.0 - 1.0);
}
```
- >这里有一个点需要格外注意，视差贴图改变了纹理的坐标，因此所有的贴图必须要用新的坐标来采样，才能做到点与点的统一。后面如果是在世界空间做计算，还需将 normal 转到世界空间s

下面是 ParallexMapping 函数的具体实现，传入 TangentViewDir 以及原纹理，并返回新的纹理坐标
```glsl
.fs
vec2 ParallaxMapping(vec2 texCoords, vec3 viewDir)
{ 
    float height =  texture(depthMap, texCoords).r;    
    vec2 p = viewDir.xy / viewDir.z * (height * height_scale);
    return texCoords - p;    
}
```
- >最后返回的是加还是减取决于 viewDir 的定义，如果从表面到眼镜（一般都是这样），则 UV -= offset;
  
由于在平面边缘上，纹理坐标超出了0到1的范围进行采样，根据纹理的环绕方式导致了不真实结果。解决办法是当它超出默认纹理坐标范围进行采样时就丢弃这个片段

```glsl
.fs
texCoords = ParallaxMapping(fs_in.TexCoords, viewDir);
if (texCoords.x > 1.0 || texCoords.y > 1.0 || texCoords.x < 0.0 || texCoords.y < 0.0)
    discard; // 不绘制该片段
```

- **（2）Steep Parallex Mapping**

**陡峭视差映射** 是视差映射的扩展，通过采样数的提高从而提高精确性
<p align="center">
  <img src="/markdown_picture/md_lighting/Steep_parallax_mapping.png" width="500">
</p>

我们从上到下遍历深度层，我们把每个深度层和储存在深度贴图中的它的深度值进行对比。通过比较当前深度值与在深度贴图此时 UV 的对应深度值 （此时 UV 根据遍历到多少层有不同的 UV 偏移量） 如果前者更大（深度值越大越深），说明该点已经在物体内，于是我们取当前层的 UV 偏移为最终 UV偏移量（离散点近似，采样越多，即层间距越小越精确）。

下面我们对 ParallaxMapping 函数进行修改
```glsl
.fs
vec2 Parallax(vec2 texCoords, vec3 TangentViewDir)
{
    const float numLayers = 10;
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    vec2 p = TangentViewDir.xy * height_scale;  // 总偏移量
    vec2 deltaTexCoords = p / numLayers;        // 每一层偏移量
}
```
- >我们先定义层的数量，计算每一层深度，最后计算纹理坐标偏移量，每一层我们都必须沿着P的方向移动

然后我们遍历所有层，直到找到小于这一层深度值的深度贴图
```glsl
.fs
vec2 currentTexCoords = texCoords;
float currentDepthMapValue = texture(depthMap, currentTexCoords).r;

while (currentLayerDepth < currentDepthMapValue)
{
    currentTexCoords -= deltaTexCoords;
    currentDepthMapValue = texture(depthMap, currentTexCoords).r;
    currentLayerDepth += layerDepth;
}

return cuurentTexCoords;
```

- **Parallax Occlusion Mapping**

**视差遮蔽映射**与陡峭视差映射原理相同，但在选取深度层作为偏移坐标时不是用触碰的第一个深度层的纹理坐标，而是在触碰前和后，在深度层之间进行线性插值。

```glsl
.fs
// 在找到 currentTexCoords 之后
vec2 prevTexCoords = currentTexCoords + deltaTexCoords; // 向之前方向移动
// 计算两个层级各自对真实深度距离多少
float afterDepth = currentDepthMapValue - currentLayerDepth;
float beforeDepth = texture(depthMap, prevTexCoords).r - currentLayerDepth + layerDepth;
// 计算权重，线性插值求交点
float weight = afterDepth / (afterDepth - beforeDepth);
vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);

return finalTexCoords;
```

---

## 9. HDR
**高动态范围（High Dynamic Range）** 允许我们将光能量设置值超过[0, 1]，从而获得大范围的黑暗与明亮的场景细节，最后通过 **色调映射（Tone Mapping）** 将HDR值转换到**LDR（Low Dynamic Range）** 以让显示器正常输出（显示器只输出[0, 1]的颜色）

- **（1）浮点帧缓冲**
默认情况下 opengl 给颜色缓冲的内部格式为 GL_RGBA8，只能保存0到1，我们可以将内部格式改为**GL_RGB16F, GL_RGBA16F, GL_RGB32F, GL_RGBA32F** 来存储超过0.0到1.0的颜色值

下面是从创建帧缓冲，到挂上允许存储 HDR 的颜色挂件的完整流程
```cpp
.cpp
unsigned int fbo;
glGenFrameBuffers(1, &fbo);
glBindFramebuffer(GL_FRAMEBUFFER, fbo);

unsigned int colorBuffer;
glGenTextures(1, &colorBuffer);
glBindTexture(GL_TEXTURE_2D, colorBuffer);

glTexImage2D(
    GL_TEXTURE_2D,
    0,                      // mipmap 层级
    GL_RGBA16F,             // 内部格式，GPU 存储图片的约定
    width,                  
    height,
    0,                      // 边框，永远为0，历史遗留问题
    GL_RGBA,                // 传入进的数据的格式
    GL_FLOAT,               // 数据类型. GL_UNSIGNED_BYTE（0 ~ 255 的普通贴图），GL_FLOAT（浮点数 HDR专用）
    NULL                    // 只申请显存，不给初始化数据
);

glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorBuffer, 0);
```
- **（2）色调映射（Tone Mapping）**
此过程我们将创建一个新的着色器来完成 HDR 到 LDR 的转变，下面先看在main 文件里的实际渲染思路
```cpp
.cpp
glBindFramebuffer(GL_FRAMEBUFFER, fbo);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
cube_shader.use();
RenderScene();

glBindFramebuffer(GL_FRAMEBUFFER, 0);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

hdrShader.use();
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D, fbo);
hdrShader.setInt("screenTexture", 0);
RenderQuad();
```

- **Reinhard 算法**

$$
L_{out} = \frac{L_{in}}{L_{in} + 1}
$$

```glsl
.fs
// Reinhard色调映射
vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
```

- **色调映射（曝光）**

$$
L_{out} = 1 - e^{-L_{in}\cdot exposure}
$$

```glsl
.fs
uniform sampler2D scrrenTexture;
uniform float exposure;

void main()
{
    vec3 hdrColor = texture(screenTexture, TexCoords).rgb;

    // 色调映射（曝光）
    vec3 mapped = vec3(1.0) - exp(-hdrColor * exposure);

    // Gamma 校正
    mapped = pow(mapped, vec3(1.0 / 2.2));

    FragColor = vec4(mapped, 1.0);
}
```
当然，还有很多优质算法，各有各的侧重点以及适用场景，此处仅详细介绍了较为基础的两种算法

---

## 10. Bloom

**泛光Bloom**很好实现，特别是在有了 HDR 后。通过 **MRT（Multiple Render Targets）** 技术，我们可以实现一个片段着色器可以同时输出到多个纹理。
使用这个技术的前提是 main 文件里面有多个颜色附件，通过```GL_COLOR_ATTAVHMENT0```、```GL_COLOR_ATTACHMENT1```得到有两个颜色缓冲的帧缓冲。
我们在片段着色器中需要指定一个布局 location 标识符，location = 0 表示写到颜色缓冲0，以此类推
```glsl
.fs
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;
```

对应到 FBO：
```cpp
.cpp
// 创建多个颜色附件
GLuint colorBuffers[2];
glGenTextures(2, colorBuffers);

for (unsigned int i = 0; i < 2; i++)
{
    glBindTexture(GL_TEXTURE_2D, colorBuffers[i]);
    glTexImage2D(GL_TEXTURE, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

    glFramebufferTexture2D(
        GL_FRAMEBUFFER, 
        GL_COLOR_ATTACHMENT0 + i,
        GL_TEXTURE_2D,
        colorBuffers[i], 0);
}
```
紧接着，我们需要显式地告诉 OpenGL 我们正在通过 glDrawBuffers 渲染多个颜色缓冲，可以通过传递多个颜色附件解决
```cpp
.cpp
GLuint attachments[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
glDrawBuffers(2, attachments);
```
现在我们可以直接在我们渲染的片段上提取亮度超过阈值的片段了：
```glsl
.fs
[...] // 正常处理光照，输出 FragColor
FragColor = vec4(lighting, 1.0f);

// 下面判断光亮是否超过阈值
float threshold = 1.0f         // 泛光阈值
float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
if (brightness > threshold)
    BrightColor = vec4(FragColor.rgb, 1.0);
else
    BrightColor = vec4(0.0);
```
- >人眼对 R, G, B 三种颜色的敏感性不同，因此需要一个权重，绿色最大，蓝色最小。我们对 FragColor 做权重的点乘，得到亮度值，并用亮度值与阈值比价
- >Bloom 的提取是在 top mapping 之前的，阈值是基于 HDR 值进行判断！这意味着上面的 brightness 的值完全可以大于1，所以阈值的设定很关键
- >这里是对阈值 threshold 的经验取值：
  正常 Bloom: 1.0 ~ 2.0
  很强 Bloom: 0.8 ~ 1.0
- >这里其实可以用Soft Threshold，在下面讲解

- **（1）高斯模糊（Gaussian blur）**

高斯模糊是一种基于高斯曲线的加权平均滤波方法，用于实现平滑和模糊效果

<p align="center">
  <img src="/markdown_picture/md_lighting/Gaussian_blur.png" width="400">
</p>

由于高斯函数的特性，可分离为两个一元函数相乘，这允许我们将原本 N * N 的采样次数降低为 2N 的采样次数，这极大地节约了性能。

首先先实现高斯模糊的片段着色器，（这里注意跟它对接的顶点着色器是一个专门用来后处理的顶点着色器，直接输出顶点位置和UV坐标）
```glsl
.fs
# version 330 core
out vec4 FragColor;
in vec2 TexColor;

uniform sampler2D image;
uniform bool horizontal;        // ture 往左右模糊，false 往上下模糊
uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);     // 权重数组


void main()
{
    vec2 tex_offset = 1.0 / textureSize(image, 0);      // 一个像素在 UV 空间里的大小
    vec3 result = texture(image, TexCoords).rgb * weight[0];    // 当前像素（中心像素）乘上最大权重
    // 横向模糊
    if (horizontal)
    {
        for (int i = 1; i < 5; ++i)
        {
            result += texture(image, TexCoords + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
            result += texture(image, Texture - vec2(tex_offset.x) * i, 0.0).rgb * weight[i];
        }
    }
    else    // 纵向采样
    {
        for (int i = 1; i < 5; ++i)
        {
            result += texture(image, TexCoords + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
            result += texture(image, TexCoords - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
        }
    }

    FragColor = vec4(result, 1.0);
}
```

接下来我们创建两个帧缓冲，每一个帧缓冲配一张颜色挂件
为什么这里用两个帧缓冲 **（双缓冲Ping-pong）** ？因为如果同时在一个纹理上读和写会产生读写冲突，所以我们用两个纹理分别进行读和写，在每一轮模糊结束之后，两者交换彼此身份，在上一次模糊的基础上继续模糊
```cpp
.cpp
GLuint pingpongFBO[2];
GLuint pingpongBuffer[2];
glGenFramebuffers(2, pingpongFBO);
glGenTextures(2, pingpongBuffer);
for (GLuint i = 0; i < 2; i++)
{
    glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[i]);
    glBindTexture(GL_TEXTURE_2D, pingpongBuffer[i]);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL
    );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pingpongBuffer[i], 0
    );
}
```
得到一个 HDR 纹理之后，我们对其进行10次模糊，5次横向，5次纵向
```cpp
.cpp
GLboolean horizontal = true, first_iteration = true;
GLuint amount = 10;
shaderBlur.use();
for (GLuint i = 0; i < amount; i++)
{
    glBindFramebuffer(GL_FRAMEBUFFER, pinpongFBO[horizontal]);
    glUniform1i(glGenUniformLocation(shaderBlur.Program, "horizontal"), horizontal);    
    // 这个函数意思就是把 CPU里 horizontal 的值（整型），传递给 Shader 里面一个叫 horizontal 的变量，从而控制是横向还是纵向模糊
    glBindTexture(
        GL_TEXTURE_2D,
        first_iteration ? colorBuffers[1] : pingpongBuffers[!horizontal]
    ); // 这一次模糊，我决定用哪一张图作为输入？如果 first_iteration 为真，就用 1 号，否则用 !horizontal
    RenderQuad();
    horizontal = !horizontal;
    if (first_iteration)
        first_iteration = false; 
    // 第一次我们用的是从 MRT 提取出来的亮色部分，后面才是用的模糊后的
}
```
最后是 blend 着色器，把两个纹理混合
```glsl
.fs
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D scene;
uniform sampler2D bloomBlur;
uniform float exposure;

void main()
{             
    const float gamma = 2.2;
    vec3 hdrColor = texture(scene, TexCoords).rgb;      
    vec3 bloomColor = texture(bloomBlur, TexCoords).rgb;
    hdrColor += bloomColor;  // 混合片段
    // tone mapping
    vec3 result = vec3(1.0) - exp(-hdrColor * exposure);
    // 最后进行 Gamma 校正    
    result = pow(result, vec3(1.0 / gamma));
    FragColor = vec4(result, 1.0f);
}
```

大体的渲染流程如下：
a. 首先绑定 hdrFBO，输出:
    colorBuffer[0] -> scene (HDR)
    colorBuffer[1] -> Bright (HDR)
b. 接下来对 Bright 进行高斯模糊
c. 将 scene 的片段和模糊的片段混合
d. 进行 tone mapping 与 Gamma 校正

---

## 11. Multi-scale Bloom, Karis Average And Soft Threshold
（原谅我这一章基本全是代码实现，因为真的偏硬核技术）
本小节是基于 Bloom 部分的引擎化升级，使用了 UE 风格等经典引擎的技术。
之前的 Bloom 渲染流程：
场景 → 提取亮色 → 高斯模糊（ping-pong）→ 合成
现在改进之后：
场景 → 提取亮色 → 多级 downsample → 多级 upsample → 合成

- **创建多级纹理**
首先需要存储每层尺寸
```cpp
struct BloomMip
{
    unsigned int texture;
    int width;
    int height;
};
```

```cpp
.cpp
const int bloomLevels = 5;

unsigned int mipFBO;
glGenFramebuffers(1, &mipFBO);

std::vector<BloomMip> mipChain;

// 接下来要创建第一张 mip ，已经是原图的一半了
int mipWidth = SCR_WIDTH / 2;
int mipHeight = SCR_HEIGHT / 2;

for (int i = 0; i < bloomLevels; i++)
{
    BloomMip mip;

    mip.width = mipWidth;
    mip.height = mipHeight;

    glGenTextures(1, &mip.texture);
    glBindTexture(GL_TEXTURE_2D, mip.texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
                 mipWidth, mipHeight,
                 0, GL_RGBA, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    mipChain.push_back(mip);   // vector 自带的成员函数

    mipWidth  /= 2;
    mipHeight /= 2;
}
```
- >我们创建的每一层分辨率都会减小一半，分辨率越小，模糊范围越大！
- **（1）Downsample Shader 下采样**
  
```glsl
.fs
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
uniform sampler2D srcTexture;

void main()
{
    // 像素采样时的必须操作，求一个像素位移对应多少 UV 坐标
    vec2 texelSize = 1.0 / textureSize(srcTexture, 0);

    vec3 result = texture(srcTexture, TexCoords).rgb * 0.25;                        // 本身
    result += textrue(srcTexture, TexCoords + vec2(texelSize.x, 0.0)).rgb * 0.25;   // 右边
    result += texture(srcTexture, TexCoords + vec2(0.0, texelSize.y)).rgb * 0.25;   // 上边
    result += texture(srcTexture, TexCoords + texelSize).rgb * 0.25;                //右上角

    FragColor = vec4(result, 1.0);
}
```
- >相当于对这个像素以及它右、上、右上四个像素进行平均采样，得到一个更小的模糊的像素

也可以中心对称采样
```
vec3 result = vec3(0.0);

result += texture(srcTexture, TexCoords + vec2(-texelSize.x, -texelSize.y)).rgb;
result += texture(srcTexture, TexCoords + vec2( texelSize.x, -texelSize.y)).rgb;
result += texture(srcTexture, TexCoords + vec2(-texelSize.x,  texelSize.y)).rgb;
result += texture(srcTexture, TexCoords + vec2( texelSize.x,  texelSize.y)).rgb;

result *= 0.25;
```

- **（1）执行 Downsample Chain**
```cpp
.cpp
glBindFramebuffer(GL_FRAMEBUFFER, mipFBO);

unsigned int currentSrc = BrightTexture; // 输入：亮度提取结果

for (int i = 0; i < bloomLevels; i++)
{
    BloomMip &mip = mipChain[i];

    // 设置当前层分辨率
    glViewport(0, 0, mip.width, mip.height);

    // 把输出目标绑定为当前 mip
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D,
                           mip.texture,
                           0);

    downsampleShader.use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, currentSrc);

    downsampleShader.setInt("srcTexture", 0);

    renderQuad();

    // 下一层输入 = 当前层
    currentSrc = mip.texture;
}
```

- **（3）Upsample**
把小分辨率的模糊光，一层层加回大图
下面是 upsample.fs
```glsl
.fs
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D srcTexture;   // 小图（更模糊）
uniform sampler2D dstTexture;   // 当前层（稍微清晰）
uniform float intensity;        // 叠加强度
void main()
{
    vec3 small = texture(srcTexture, TexCoords).rgb;
    vec3 large = texture(dstTexture, TexCoords).rgb;

    // 核心：叠加
    vec3 result = (small + large) * intensity;

    FragColor = vec4(result, 1.0);
}
```

Upsample 渲染流程
```cpp
.cpp
glBindFramebuffer(GL_FRAMEBUFFER, mipFBO);

for (int i = bloomLevels - 1; i > 0; i--)
{
    BloomMip &mip     = mipChain[i];
    BloomMip &prevMip = mipChain[i - 1];

    // 输出写到更大的那一层
    glViewport(0, 0, prevMip.width, prevMip.height);

    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D,
                           prevMip.texture,
                           0);

    upsampleShader.use();

    // 小图（更模糊）
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mip.texture);

    // 大图（当前层）
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, prevMip.texture);

    upsampleShader.setInt("srcTexture", 0);
    upsampleShader.setInt("dstTexture", 1);

    renderQuad();
}
```

- **（4）Karis Average**
防止某个特别亮像素过渡干扰周边像素，我们采取越亮的元素权重越小（之前是每个像素权重均为0.25）
```glsl
.fs
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D srcTexture;

float KarisWeight(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return 1.0 / (1.0 + luma);
}

void main()
{
    vec2 texelSize = 1.0 / textureSize(srcTexture, 0);

    vec3 c0 = texture(srcTexture, TexCoords).rgb;
    vec3 c1 = texture(srcTexture, TexCoords + vec2(texelSize.x, 0.0)).rgb;
    vec3 c2 = texture(srcTexture, TexCoords + vec2(0.0, texelSize.y)).rgb;
    vec3 c3 = texture(srcTexture, TexCoords + texelSize).rgb;

    float w0 = KarisWeight(c0);
    float w1 = KarisWeight(c1);
    float w2 = KarisWeight(c2);
    float w3 = KarisWeight(c3);

    vec3 result =
        c0 * w0 +
        c1 * w1 +
        c2 * w2 +
        c3 * w3;

    float totalWeight = w0 + w1 + w2 + w3;

    result /= totalWeight;

    FragColor = vec4(result, 1.0);
}
```
- >这部分代码还是比较好理解的，就根据光照改变了权重
  
- **（5）Soft Threshold**
我们在之前提取光亮部分有一个问题，当光照突然超过阈值的时候，会突然有泛光效果，这会导致边界很生硬。现在我们可以用 Soft Threshold 技术实现当光接近阈值的时候就会有微弱柔和的光出现，即把接近亮度阈值的区域，平滑地变成 bloom 输入

这里我们不用 MRT 技术了，我们重新创建一个新的返回明亮片段的着色器 **Bright_Pass_Shader**

```glsl
.fs
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D scene;

uniform float threshold; // 比如 1.0
uniform float knee;      // 比如 0.5

void main()
{
    vec3 color = texture(scene, TexCoords).rgb;

    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));    // 取亮度

    float soft = brightness - threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);

    float contribution = max(soft, brightness - threshold);

    vec3 result = color * contribution;

    FragColor = vec4(result, 1.0);
}
```
- >只允许 [threshold - knee, threshold + knee] 范围内参与平滑计算
- >使用平方函数平滑过渡
- >与硬阈值结果取 max

## 12. Deferred Shading
当存在很多光源时，**前向渲染**会逐一算每个片段和每个光源，性能直接挂掉。由此我们可以使用**延迟渲染**，将所有像素的信息存下来，再按照屏幕逐像素计算光照。

- **（1）Geometry Pass**
**G缓冲（G-buffer）**是对所有用来存储光照相关的数据，并在最后的光照处理中使用的所有纹理的总称：
一般包括 Position, Normal, Albedo, Specular

下面是渲染循环中的渲染流程，伪代码:
```cpp
.cpp
glBindFramebuffer(gBuffer);

for (每个物体)
{
    用 gbufferShader 渲染
}
```
在几何渲染处理阶段，我们首先初始化一个帧缓冲队对象，同时它包含多个颜色缓冲和一个单独的深度缓冲。位置和纹理，为你使用高精度的纹理，而对于颜色加镜面，使用默认纹理。
```cpp
.cpp
GLuint gBuffer;
glGenFramebuffers(1, &gBuffer);
glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
GLuint gPosition, gNormal, gColorSpec;

// - 位置颜色缓冲
glGenTextures(1, &gPosition);
glBindTexture(GL_TEXTURE_2D, gPosition);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0

// - 法线颜色缓冲
[...] 同上，GL_COLOR_ATTACHMENT1

// - 颜色 + 镜面颜色缓冲
[...] 同上，只不过内部格式为GL_RGBA, GL_COLOR_ATTACHMENT2

// - 告诉OpenGL我们将要使用(帧缓冲的)哪种颜色附件来进行渲染
GLuint attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
glDrawBuffers(3, attachments);

// 之后同样添加渲染缓冲对象(Render Buffer Object)为深度缓冲(Depth Buffer)，并检查完整性
[...]
```
然后我们用 MRT 技术将片段着色器的输出放到对应的纹理中：
```glsl
.fs
#version 330 core
layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec4 gAlbedoSpec;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;

void main()
{    
    // 存储第一个G缓冲纹理中的片段位置向量
    gPosition = FragPos;
    // 同样存储对每个逐片段法线到G缓冲中
    gNormal = normalize(Normal);
    // 和漫反射对每个逐片段颜色
    gAlbedoSpec.rgb = texture(texture_diffuse1, TexCoords).rgb;
    // 存储镜面强度到gAlbedoSpec的alpha分量
    gAlbedoSpec.a = texture(texture_specular1, TexCoords).r;
}  
```

- **（2）Lighting Pass**
在光照处理阶段我们会渲染一个 2D 全屏正方形，并在每个像素上运行光照片段着色器
```cpp
.cpp
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
shaderLightingPass.Use();
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_2D, gPosition);
glActiveTexture(GL_TEXTURE1);
glBindTexture(GL_TEXTURE_2D, gNormal);
glActiveTexture(GL_TEXTURE2);
glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
// 同样发送光照相关的uniform
SendAllLightUniformsToShader(shaderLightingPass);
glUniform3fv(glGetUniformLocation(shaderLightingPass.Program, "viewPos"), 1, &camera.Position[0]);
RenderQuad(); 
```
而光照片段着色器中，FragPos, Normal, Albedo 直接从传入的G缓冲中采样获取数据。

- **结合延迟渲染和前向渲染**
由于 blending 需要对多个片段进行操作。而延迟渲染是对从 G缓冲中提取的单一片段进行操作，因此我们可以将两者结合渲染。
- **更多优化方案**
**Clustered Shading = 在 Tile-based 基础上引入深度划分**，使光源筛选从2D升级为3D，大幅提升光照计算效率和精度。这也是 UE5 在多光源处理时的思路，现在先暂时不深究。
（此节后续仍有开发空间）

---

## 13. SSAO
**屏幕空间环境光遮蔽（Screen-Space Ambient Occlusion, SSAO）** 原理：对于铺屏四边形上的每一个片段，我们根据周边的深度值计算一个**遮蔽因子（Occlusion Factor）**，这个遮蔽因子之后会被用来减少或抵消片段的环境光照分量。我们通过采集片段周围的 **法向半球体（Normal-oriented Hemisphere）** 的多个深度样本，并和当前深度值比较得到，高于片段深度值的样本个数就是我们想要的遮蔽因子。

- **（1）法向半球体**
简单来说，法向半球体用来描述这给点朝外能看见哪些方向。由于我们只关心外面的空间，所以只采样法线朝上的那一半空间。我们将在切线空间内生成采样核心

<p align="center">
  <img src="/markdown_picture/md_lighting/SSAO1.png" width="500">
</p>

下面**生成一堆法向半球体里的随机采样点**

```cpp
.cpp
// 随机数生成器
std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0);  // 生成 0 ~ 1 之间的随机数
std::default_random_engine generator;

// 生成一个随机方向，得到法向半球
glm::vec3 sample(
    randomFloats(generator) * 2.0 - 1.0, 
    randomFloats(generator) * 2.0 - 1.0, 
    randomFloats(generator)
);

// 归一化：变成单位方向向量
sample = glm::normalize(sample);

// 乘上一个随机长度，得到的点将从球表面转换到整个半球体里面，实现分布的随机化
sample *= randomFloats(generator);

// scale, 控制采样的分布，前面的点更密集，后面的点更远，实现中心密集分布
float scale = float(i) / 64.0;
scale = 0.1f + 0.9f * scale * scale; // 关键！
sample *= scale;
ssaoKernel.push_back(sample);
```

于是我们的得到了一个**大部分样本靠近原点的核心分布**

<p align="center">
  <img src="/markdown_picture/md_lighting/SSAO2.png" width="500">
</p>

- **（2）随机核心转动**
创建一个小的随机旋转向量纹理平铺到屏幕上

```cpp
.cpp
// 我们创建一个 4 * 4 朝向切线空间平面法线的随机旋转向量数组：
std::vector<glm::vec3> ssaoNoise;
for (GLuint i = 0; i < 16; i++)
{
    glm::vec3 noise(
        randomFloats(generator) * 2.0 - 1.0, 
        randomFloats(generator) * 2.0 - 1.0, 
        0.0f); 
    ssaoNoise.push_back(noise);

// 创建纹理
GLuint noiseTexture; 
glGenTextures(1, &noiseTexture);
glBindTexture(GL_TEXTURE_2D, noiseTexture);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}
```

- **SSAO着色器**

我们需要存储SSAO阶段的结果，我们还要创建一个帧缓冲对象：
```cpp
.cpp
GLuint ssaoFBO;
glGenFramebuffers(1, &ssaoFBO);  
glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
GLuint ssaoColorBuffer;

glGenTextures(1, &ssaoColorBuffer);
glBindTexture(GL_TEXTURE_2D, ssaoColorBuffer);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoColorBuffer, 0);
```
由于环境遮蔽的结果是一个灰度值，我们将只需纹理的红色分量，我们将颜色缓冲的内部格式设为 GL_RED
完整渲染阶段
```cpp
.cpp
// 几何处理阶段: 渲染到G缓冲中
glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    [...]
glBindFramebuffer(GL_FRAMEBUFFER, 0);  

// 使用G缓冲渲染SSAO纹理
glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    shaderSSAO.Use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gPositionDepth);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gNormal);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, noiseTexture);
    SendKernelSamplesToShader();
    glUniformMatrix4fv(projLocation, 1, GL_FALSE, glm::value_ptr(projection));
    RenderQuad();
glBindFramebuffer(GL_FRAMEBUFFER, 0);

// 光照处理阶段: 渲染场景光照
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
shaderLightingPass.Use();
[...]
glActiveTexture(GL_TEXTURE3);
glBindTexture(GL_TEXTURE_2D, ssaoColorBuffer);
[...]
RenderQuad();
```
shaderSSAO这个着色器将对应G缓冲纹理(包括线性深度)，噪声纹理和法向半球核心样本作为输入参数：

```glsl
.fs
#version 330 core
out float FragColor;
in vec2 TexCoords;

uniform sampler2D gPositionDepth;
uniform sampler2D gNormal;
uniform sampler2D texNoise;

uniform vec3 samples[64];
uniform mat4 projection;

// 屏幕的平铺噪声纹理会根据屏幕分辨率除以噪声大小的值来决定
const vec2 noiseScale = vec2(800.0/4.0, 600.0/4.0); // 屏幕 = 800x600

void main()
{
    [...]
}

```