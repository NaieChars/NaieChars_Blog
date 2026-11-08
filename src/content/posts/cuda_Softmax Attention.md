---
title: Softmax Attention CUDA 算子
published: 2026-11-04
pinned: true
description: 解析 CUDA 算子
tags: [CUDA]
category: 技术
draft: false
---

# Softmax 算子

## Softmax 稳定形式公式
对于一行输入：

$$
x=[x_0, x_1, ..., x_{L-1}]
$$
Softmax 的定义是：

$$
y_i=\frac{e^{x_i}}{\sum_{j=0}^{L-1}e^{x_j}}
$$
可以拆分成两步：
- 对每一个元素取指得到正数
- 除以这些正数的总和，使得输出之和为 1

对于计算机而言，我们通常会先求输入的 max，因为计算机能表示的数值范围有限，如果输入元素有 1000，`expf(100)` 会溢出，所以解决办法是：**所有元素先减去同一个最大值，再取指**

即 Softmax 的稳定形式为：

$$
\boxed{y_i=\frac{e^{x_i - m}}{\sum_{j=0}^{L-1}e^{x_j-m}}}
$$

## 分层归约思路
> 在 CUDA 中，我们通常采用分层归约进行 Softmax

先考虑一个常见方案：**一个 block 处理一整行输入**。

假设一行有 1024 个元素，一个 block 256 个线程，每一个线程处理 4 个元素，即

| 线程 | 负责的元素下标 |
| --- | --- |
| thread 0 | 0、256、512、768 |
| thread 1 | 1、257、513、769 |
| thread 255 | 255、511、767、1023 |

1. 第一步，每一个线程先求自己的局部最大值：

$$
m_t=\max_{i\in I_t}x_i
$$
其中，$I_t$ 表示线程 t 负责的下标集合，然后把所有吸纳成的局部最大值进行**最大值归约**，得到本行最大值 $m$：

$$
m=\max_tm_i
$$
2. 每一个线程都用同一个 m，计算自己的局部指数和，再归约所有的局部和：

$$
s_t=\sum_{i\in I_t}e^{x_i-m} \\

s=\sum_t s_t

$$
3. 最后一步，每一个线程计算自己负责的输出

> 如果一个把一行分成若干块，每一个块求出自己的局部最大值与局部和，最终需要归约为整体最大值与局部和

需要注意的是：不同块的局部和不能直接相加，需要先把他们换成同一个基准：

$$

e^{x_i-m}
=
e^{x_i-m_b}\cdot e^{m_b-m}
$$
因此，正确的整体指数和：

$$
\boxed{
s=\sum_b s_b\,e^{m_b-m}
}
$$

# Softmax Attention 算子

## 缩放点积注意力公式

$$
\boxed{O = \text{softmax}(\frac{QK^T}{\sqrt{d}})V} 
$$

这是 Transformer 中使用的一种注意力计算，下面分别讲解一下每一个变量的含义：

| 矩阵 | 形状 | 直观含义 |
| --- | --- | --- |
| $Q$：Query | $M\times d$ | 有 $M$ 个查询，每个查询用 $d$ 维向量表示 |
| $K$：Key | $N\times d$ | 有 $N$ 个用于匹配的键 |
| $V$：Value | $N\times d$ | 每个键对应的实际内容 |
| $O$：Output | $M\times d$ | 每个查询得到一个 $d$ 维结果 |

其中：**$K$ 的第 $j$ 行与 $V$  的第 $j$ 行是一对**，用 Key 判断匹配程度，用 Value 提供最终参与加权的内容。注意力的输出就是这些 Value 的加权和

## 1. 计算匹配分数：$S=QK^T/\sqrt{d}$

取第 $i$ 个 Query（Q 的第 $i$ 行），与第 $j$ 个 Key（$K^T$ 的第 $j$ 列） 做点积：

$$

S_{ij}
=
\frac{1}{\sqrt d}
\sum_{r=0}^{d-1}Q_{ir}K_{jr}

$$
$S_{ij}$ 越大，表示按照这里的点积评分方式，Query $i$ 与 key $j$ 越匹配。**一行 $S_{i,:}$ 就是一个 Query 对全部 $N$ 个 Key 的评分**

除以 $\sqrt d$ 是为了控制分数的尺度：在常见的统计假设下，点积的方差随 $d$ 增大，缩放可以避免 Softmax 过于尖锐。

> [!NOTE]
>为什么随着 $d$ 增大，方差变大？
> 对于点积公式，假设各分量相互独立，均值为 0，方差为 1，相互独立的随机变量相加，方差也相加，因此 $\text{Var}(s) = \Sigma\text{Var}(q_rk_r)=d$
> 所以方差随着 $d$ 增长，标准差随着 $\sqrt{d}$ 增长，标准差更直接地反映结果的典型波动幅度。
> 因此除以 $\sqrt{d}$ 能让不同维度下的评分保持相近的波动尺度，避免维度增大时评分差距变得过大，使 Softmax 权重过度集中。

## 2. 对每行做 Softmax：将评分变成权重

固定 Query $i$，对他的 $N$ 个评分进行稳定 Softmax：

$$
P_{ij} = \frac{\exp{(S_{ij} - m_i)}}{\Sigma_{t=0}^{N-1}\exp{(S_{it}-m_i)}}
$$

得到的**权重** $P$ 仍然为 $M \times N$，并且每一行：

$$
P_{ij} > 0, \quad \Sigma_{j=0}^{N-1} P_{ij} = 1
$$

这里要分清两个维度：
- **点积沿 $d$ 维求和：计算一对 Query、Key 的分数**。
- **Softmax 沿 $N$ 维归一化：比较同一个 Query 对所有 Key 的分数**。

## 3. 对 Value 加权求和：$O=PV$
对于 Query $i$，用刚才得到的权重，对所有 Value 向量进行加权:

$$
O_{i,:}
=
\sum_{j=0}^{N-1}P_{ij}V_{j,:}
$$
也就是说，一个权重 $P_{ij}$ 乘的是整个 Value 向量 $V_{j,:}$，其中所有的分量使用同一个权重。

# CUDA 代码实现

## 背景

- `Q` 矩阵大小：$M \times d$，`K` 与 `V` 矩阵大小：$N \times d$
- $1 \leq$ `M`，`N` $\leq 100,000$
- $1 \leq$ `d` $\leq 128$
- 这里先采用 `M` $= 512$，`N` $= 256$

第一个版本先约定：
```cpp
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;
```

## 分层归约
在 Softmax 中，我们需要对每一个 block 求出当前的局部和与局部 max，需要用到分层归约的技巧

首先通过 `__shfl_down_sync` 实现 warp 内线程的互相通信
```cpp
template <bool IS_MAX>
__device__ __forceinline__ float warp_reduce(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        const float other = __shfl_down_sync(0xffffffffu, value, offset);
        if (IS_MAX)
            value = fmaxf(value, other);
        else
            value += other;
    }
    return value;
}
```

`block_reduce` 调用 `warp_reduce`，实现一个 Block 归约的输出

```cpp
template <bool IS_MAX>
__device__ __forceinline__ float block_reduce(float value)
{
    __shared__ float shared[NUM_WARPS];
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    value = warp_reduce<IS_MAX>(value);
    if (lane == 0) 
        shared[warp_id] = value;

    __syncthreads();

    if (warp_id == 0)
    {
        const float identity = (IS_MAX) ? -CUDART_INF_F : 0.0f;
        value = (lane < NUM_WARPS) ? shared[lane] : identity;
        value = warp_reduce<IS_MAX>(value);
        
        if (lane == 0)
            shared[0] = value;
    }
    __syncthreads();

    value = shared[0];

    __syncthreads();
    return value;
}
```

## 核心 Kernel
### K 访存优化：一个 warp 负责一个 Key 评分
先计算评分，也就是 $P_{ij} = Q_i \times K_j^T$，即评分矩阵 i, j 的位置表示第 i 个输入向量对第 j 个 Key 打的分数（匹配程度）

很容易想到的一个计算代码，一个线程负责一个 Key 向量的评分：

```cpp
const int tid = threadIdx.x;
const int row = blockIdx.x;

if (row >= M) return;

// compute score
// 每一个线程计算一个 Q-K 评分，无效线程后续仍然参与归约
float score = -CUDART_INF_F;
if (tid < N)
{
    float dot = 0.0f;
    for (int e = 0; e < d; ++e)
    {
        dot += Q[row * d + e] * K[tid * d + e];
    }

    score = dot * rsqrtf(static_cast<float>(d));
}
```
上述代码出现了一个经典的性能优化问题——访存未合并，相邻两个线程同时在处理某个 `e` 时，从显存拿 K 的值并不是合并访存，中间相差了 d 个元素。当然这种情况下，GPU 可能会把相关的扇区放在 L1 L2 中，真实的性能损耗得看 L1 L2 命中率。

针对上面的情况，可以**考虑一个 warp 负责一个 Key 向量的评分**，`lane = 0, 1...31` 读取 $K$ 同一行的相邻分量（即第 $j$ 个 Key）。所以，固定 Query $i$ 和 Key $j$ 后，点积沿特征下标 $e$ 展开。$e$ 从 0 增到 d-1 时，`K[j * d + e]` 正好沿着原始 $K$ 的第 $j$ 行连续读取。

拿 $d=128$ 举例，一个 warp 的 32 个 lane 可以分四轮计算同一个点积：

| lane | 负责的特征下标 \\(e\\) |
| --- | --- |
| 0 | 0、32、64、96 |
| 1 | 1、33、65、97 |
| … | … |
| 31 | 31、63、95、127 |

第一轮，32 个 lane 同时读 `K[j*d+0]` 到 `K[j*d+31]`，各自乘上 Q 的对应分量。四轮后，每个 lane 手里有 4 项乘积的局部和。把这 32 个局部和传给写好的 `warp_reduce<false>`，lane 0 就会得到完整点积，并写下评分。同时，**一个 warp 可以负责多个 Key 的评分**，一共 N 个 Key，仍然是一个 block 负责一个 Query。

```cpp
for (int j = warp_id; j < N; j += NUM_WARPS)
{
    float partial = 0.0f;
    for (int e = 0; e < d; e += 32)
    {
        partial += Q[row * d + e] * K[j * d + e];
    }
    
    const float dot = warp_reduce<false>(partial);
    if (lane == 0)
        weights[j] = dot * rsqrtf(static_cast<float>(d));  // 这个阶段暂存评分
}

// Softmax 阶段重新分工，每个线程接手一个评分
const float score = (tid < N) ? weights[tid] : -CUDART_INF_F;
```

### Online Softmax 优化
普通稳定性 Softmax（Safe Softmax）很直观，先找全局最大值，再统一计算。通过减去全局最大值 m 来防止指数溢出。

**Online Softmax** 的精妙之处在于，它**不等待看到全局最大值**，而是维护一套“当前最优”的状态，并**在遇到新数据时进行一致性修正**。

固定一个 Query，设它对各个 Key 的评分是 $s_j$，输出向量可以写成:

$$
\mathbf o
=\sum_j \frac{e^{s_j-m}}{\sum_k e^{s_k-m}}\mathbf v_j
=\frac{\underbrace{\sum_j e^{s_j-m}\mathbf v_j}_{\text{加权和 }\mathbf a}}
       {\underbrace{\sum_j e^{s_j-m}}_{\text{指数和 }\ell}}.
$$

