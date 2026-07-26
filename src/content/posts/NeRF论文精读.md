---
title: 论文精读 NeRF-Representing Scenes as Neural Radiance Fields for View Synthesis
published: 2026-07-26
pinned: true
description: 本节主要精读《NeRF-Representing Scenes as Neural Radiance Fields for View Synthesis》，理解体积渲染和 NeRF
tags: [DL, 神经网络, NeRF, 计算机视觉, Python, PyTorch] 
category: 技术
draft: false
---

[论文下载地址 NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis（Mildenhall et al., ECCV 2020）](https://arxiv.org/pdf/2003.08934)

## 基于辐射场的体积渲染

### 路径追踪器与NeRF的光线逻辑不同
#### 路径追踪光线
在路径追踪器里，光线在场景里的行为是**离散事件序列**——光线要么打中一个表面（材质分类、重要性采样、算attenuation），要么打中光源（自发光计入终止），要么啥都没打中（背景纯黑，直接break）。每次hit都是一个**二元判定（hit or miss）**，然后在hit点做完材质计算后，光线继续弹射（recursion/俄罗斯轮盘赌控制深度）。
#### NeRF光线
NeRF的光线完全不弹射，就是相机发出的一条**直线**，笔直穿过一段空间，不会因采样改变方向。但是作为交换，NeRF认为**光线路径上的每一个点都有一个连续的体积密度 $𝜎(x)$**，可以认为是**光线在位置 $x$ 处终止的微分概率**。

### 从物理模型推导渲染方程
我们假设光线沿方向前进时，辐射亮度 $L$ 的变化只受两件事影响（NeRF简化模型，不考虑光线被其他方向散射进来，只考虑吸收和自发光）：
1. **吸收**：走过一小段距离 $dt$，光线有一定概率消失，消失率正比于密度 $\sigma$
2. **自发光**：这一小段本身也会往外发光，贡献正比于密度 $\sigma$ 乘上这一点的颜色 $c$

写成微分方程：

$$
\frac{dL}{dt} = -\sigma(t)L(t) + \sigma(t)c(t)
$$

求解微方方程可得：

$$
L(t) = \int_{t_n}^{t} T(s) \sigma(s) c(s) \, ds
$$

其中：

$$
T(t) = \exp \left( - \int_{t_n}^{t} \sigma(s) \, ds \right)
$$

$T(t)$这一项，就是沿途累积的 attenuation 的连续版本，**物理意义：光线从起点走到t这个位置，一路上没被吸收的概率（或是剩余能量比例）**。  
而整个积分，便是把路径上每一点的自发光 $c(s)$ 按这一点本身多大权重发光 $\sigma(s)$ 和还有多少能到达相机 $T(s)$ 相乘，得到这条光线上累加每次的衰减得到最终颜色，只不过相较于路径追踪，这里从离散求和变成了连续积分。

### 积分离散化
#### 分层采样（stratified sampling）
**为什么不能像普通黎曼积分那样均匀切分区间**？因为论文作者发现如果每次训练用的采样点位置都固定不变，网络的分辨率会被这个固定的离散网格限制死，因为其学不会空间中任意点的值。

而分层采样便是先把 $[t_n, t_f]$均匀分成 $N$ 个小区间，但**在每个小区间内部随机取一个点**，公式为

$$
t_i \sim \mathcal{U} \left[ t_n + \frac{i-1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n) \right]
$$

**每次训练迭代，具体的采样位置都不一样**，这就迫使网络必须学会空间里任意连续位置的值，而不是只记住几个固定网格点。这根 **TAA** 里面对像素划分成 N 份，每个子像素采样会进行随机抖动几乎差不多。

### 离散求和公式
有了这 $N$ 个采样点 $\{t_1, ..., t_N\}$，积分被近似成：

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \left( 1 - \exp(-\sigma_i \delta_i) \right) c_i
$$

其中：

$$
T_i = \exp \left( -\sum_{j=1}^{i-1} \sigma_j \delta_j \right)
$$

$\delta_i = t_{i+1} - t_i$ 是相邻采样点的间距。其中每一项的物理含义：

- $T_i$：透过率的离散化，也就是把从起点到第 $i$ 个点之前，所有段的衰减连续相乘。注意这里是**累乘**（数学上**等价于把 $\exp$ 里的求和展开**），这和路径追踪器里每次 hit 更新一次 `attenuation *= ...`的写法在结构上几乎一模一样

- $(1 - \exp(-\sigma_i \delta_i))$：这一项**不是直接用 $\sigma_i$，而是第 $i$ 段的局部不透明度**。当 $\sigma_i \delta_i$ 很小时（密度低或者段很短），这一项约等于 $\sigma_i \delta_i$（泰勒展开一阶近似），退化回和积分公式几乎一致的形式；当 $\sigma_i$ 很大时（比如撞上了实心物体表面），这一项趋近于 1，表示这一段几乎完全不透明，光线走到这基本完全挡住了，这其实就是**比尔-朗伯定律的直接推论**

- $c_i$：这个点网络预测出的颜色，直接加权累加进最终颜色

整个公式可以这样理解：从相机出发，沿着这条光线走过 $N$ 个采样点，每个点对最终颜色的贡献 $=$ 光线到该点之前还剩多少能量（$T_i$） $\times$ 该点本身有多不透明（$1 - \exp(-\sigma_i \delta_i)$） $\times$ 该点颜色（$c_i$），把这 $N$ 个点的贡献加起来就是这条光线最终呈现的颜色。

### 与蒙特卡洛积分的关系
看整个离散化公式，**采样点的选取用了蒙特卡洛的分层采样策略，但一旦采样点确定了，公式内部的求和是解析、确定性的**，也就是说只有一层随机。但是路径追踪里发射光线随机，每次hit后也要产生一个新随机数来决定下一项，两者有明显区别。然而该离散公式仍然具有类似蒙特卡洛方法的无偏性，只要采样点足够多足够密，会收敛到真实积分。

### 代码实现
这是一个测试代码，包含离散求和公式的实现与验证

<details>
<summary> 点击展开代码</summary>

```python
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- 分层采样 -------------
def stratified_sample(t_n, t_f, N, device):
    # 返回：shape(N,)的采样点t值
    bin_edges = torch.linspace(t_n, t_f, N + 1, device=device)  # 等间距生成 N + 1 个点，总bin数为N，shape(N+1,)
    bin_starts = bin_edges[:-1] # 每个 bin 的左边界
    bin_ends = bin_edges[1:]    # 每个 bin 的有边界

    rand_offsets = torch.rand(N, device=device) # shape(N,)
    t_samples = bin_starts + rand_offsets * (bin_ends - bin_starts)
    return t_samples


# 验证一下：采样点应该都落在[t_n, t_f]内，且是递增的
t_n, t_f = 2.0, 6.0
N = 8
t_samples = stratified_sample(t_n, t_f, N, device)
print("采样点:", t_samples)
print("是否递增:", torch.all(t_samples[1:] > t_samples[:-1]).item())
print("是否都在范围内:", (t_samples.min() >= t_n) and (t_samples.max() <= t_f))

# --------- 假想场景：sigma(t)、color(t)，模拟一团位于t=4附近的雾 ----------
def toy_sigma(t):
    # 用高斯函数模拟密度：在t=4附近密度最高
    return 8.0 * torch.exp(-0.5 * ((t - 4.0) / 0.5) ** 2)

def toy_color(t):
    # 颜色暂且固定为红色
    N = t.shape[0]
    color = torch.zeros(N, 3, device=t.device)
    color[:, 0] = 1.0
    return color

# ------------ 体渲染积分 -------------
def volume_render(t_sample, sigma, color):
    """
    t_samples: (N,) 采样点位置，沿光线递增
    sigma: (N,) 每个采样点的密度
    color: (N, 3) 每个采样点的颜色
    返回: 最终颜色 (3,), 以及每个采样点的权重 T_i*alpha_i (N,)，方便调试可视化
    """
    N = t_sample.shape[0]

    # 计算相邻采样点间距 delta_i = t_{i+1} - t_i
    # 最后一个点没有下一个点，论文做法是给它一个很大的delta（近似当作趋于无穷远）
    delta = t_sample[1:] - t_sample[:-1]
    delta_last = torch.tensor([1e10], device=t_sample.device)   # 最后一点接近无穷
    delta = torch.cat([delta, delta_last], dim=0)

    # alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1.0 - torch.exp(-delta * sigma)

    # T_i = exp(-sum_{j<i} sigma_j * delta_j)
    # 注意T_1永远是1（还没经过任何衰减）
    sigma_delta = sigma * delta  # shape (N,)
    # torch.cumsum算累加和；我们要的是"严格小于i"的累加，所以要在前面手动插入一个0再去掉最后一项
    accumulated = torch.cumsum(sigma_delta, dim=0)
    accumulated_exclusive = torch.cat([
        torch.zeros(1, device=t_sample.device),
        accumulated[:-1]
    ])  # shape (N,)，第i项变成sum_{j<i}，第0项是0

    T = torch.exp(-accumulated_exclusive)

    weights = T * alpha  # shape (N,)，这就是每个采样点对最终颜色的"权重"

    # 最终颜色 = sum(weights_i * color_i)
    final_color = torch.sum(weights.unsqueeze(1) * color, dim=0)  # shape (3,)
    return final_color, weights

# -------- 运行验证结果 ------------
sigma_vals = toy_sigma(t_samples)      # (N,)
color_vals = toy_color(t_samples)      # (N, 3)

final_color, weights = volume_render(t_samples, sigma_vals, color_vals)
print("\n采样点t:", t_samples.cpu().numpy())
print("对应密度sigma:", sigma_vals.cpu().numpy())
print("每个点的权重(T_i*alpha_i):", weights.cpu().numpy())
print("权重之和(应该<=1):", weights.sum().item())
print("最终渲染颜色 (R,G,B):", final_color.cpu().numpy())

# ---------- 用更密集的采样点，画出sigma、T、weight随t变化的曲线，直观验证物理意义 ----------
N_dense = 200
t_dense = torch.linspace(t_n, t_f, N_dense, device=device)
sigma_dense = toy_sigma(t_dense)
color_dense = toy_color(t_dense)
_, weights_dense = volume_render(t_dense, sigma_dense, color_dense)

# 手动重新算一遍T用于画图（前面volume_render没有单独返回T，这里重新提取逻辑画图用）
delta_dense = t_dense[1:] - t_dense[:-1]
delta_dense = torch.cat([delta_dense, torch.tensor([1e10], device=device)])
sigma_delta_dense = sigma_dense * delta_dense
accumulated_dense = torch.cumsum(sigma_delta_dense, dim=0)
accumulated_excl_dense = torch.cat([torch.zeros(1, device=device), accumulated_dense[:-1]])
T_dense = torch.exp(-accumulated_excl_dense)

t_np = t_dense.cpu().numpy()
sigma_np = sigma_dense.cpu().numpy()
T_np = T_dense.cpu().numpy()
weights_np = weights_dense.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(t_np, sigma_np)
axes[0].set_title("sigma(t) — 密度分布（假想的雾）")
axes[0].set_xlabel("t")

axes[1].plot(t_np, T_np)
axes[1].set_title("T(t) — 透过率，应单调递减")
axes[1].set_xlabel("t")

axes[2].plot(t_np, weights_np)
axes[2].set_title("weight(t) = T*alpha — 每点对最终颜色的贡献")
axes[2].set_xlabel("t")

plt.tight_layout()
plt.savefig("volume_render_toy.png", dpi=120)
plt.show()
print("\n图已保存为 volume_render_toy.png")
```

</details>

在终端你应该会看见如下输出：

```powershell
样点: tensor([2.4819, 2.7023, 3.2912, 3.8842, 4.3187, 4.6949, 5.3758, 5.9204],
       device='cuda:0')
是否递增: True
是否都在范围内: tensor(True, device='cuda:0')

采样点t: [2.481948  2.702264  3.2912261 3.8841789 4.318664  4.694871  5.3757515
 5.920372 ]
对应密度sigma: [7.9697035e-02 2.7560255e-01 2.9291751e+00 7.7882209e+00 6.5296149e+00
 3.0457594e+00 1.8160109e-01 5.0105201e-03]
每个点的权重(T_i*alpha_i): [1.7405272e-02 1.4722256e-01 6.8828648e-01 1.4209706e-01 4.5608659e-03
 3.7393314e-04 5.0630360e-06 4.8701768e-05]
权重之和(应该<=1): 0.9999999403953552
最终渲染颜色 (R,G,B): [0.99999994 0.         0.        ]
```

同时得到如下实验结果图：

<p align="center">
  <img src="/markdown_picture/神经网络/volume_render_toy.png" width="600">
</p>
<p align="center">
  sigma、T、weight随t变化的曲线
</p>

> [!NOTE]
> 右图`weight(t)`应该也是一个类似钟形但**比sigma更尖、峰值在sigma峰值稍靠前一点**的曲线——这是因为权重是`T`(还没被完全吸收)和`alpha`(这一点本身吸收多少)的乘积，在雾团中心之前`T`还很高、`alpha`开始变大，两者相乘在峰值前达到最大，过了峰值之后虽然`alpha`还大，但`T`已经掉得很低了

## 网络结构
论文里提到：尽管神经网络理论上是万能函数近似器，但让网络直接处理xyzθφ这种原始坐标，渲染出来的高频细节（颜色和几何上的高频变化）表现很差，这是因为**深度网络天生偏向学习低频函数（谱偏差）**，如果**先用高频函数把输入映射到更高维空间**再喂给网络，能显著改善对高频数据的拟合能力。

## 分层体积采样（Hierarchical sampling）

## 损失函数和训练细节