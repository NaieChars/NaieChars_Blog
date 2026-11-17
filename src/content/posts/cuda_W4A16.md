---
title: Fused W4A16 GEMV CUDA 算子
published: 2026-11-15
pinned: true
description: 解析 CUDA 算子
tags: [CUDA]
category: 技术
draft: false
---

# 问题背景

- 对于一个输入向量 $X$，大小为 `1 x K`（FP16）
- 压缩的量化权重矩阵 $W_q$，大小为 `N x (K / 2)`（uint8）

**解压量化权重**：对于 $W_q$，Row-Major 存储，每一个 Byte 存 2 个 INT4
- $W_{\text{int4}}[n, 2k] = W_q[n,k] \quad \& \quad 0xF$
-  $W_{\text{int4}}[n, 2k + 1] = (W_q[n,k] >> 4) \quad \& \quad 0xF$

量化参数：
- $G = 128$，$S$ 大小 `N x (K/G)`，$Z$ 大小 `N x (K/G)`

量化公式：

$$
\begin{aligned}
\hat{W}[n, k] &= (W_{\text{int4}}[n, k] - Z[n, \lfloor k/G \rfloor]) \times S[n, \lfloor k/G \rfloor] \\
Y &= X\hat{W}^T \quad (Y \in \mathbb{R}^{1 \times N})
\end{aligned}
$$
## 本机环境
- RTX 5060 Laptop GPU
- CUDA 12.8

# Baseline

一个很直观的想法是：**一个线程处理一行**，总共需要 N 个线程，每一个线程负责输出一个 `y[n]`。需要注意的是数据类型的转换，我们用 `int` 来接解包后的权重，反量化与最后结果计算都是浮点运算，因此需要将输入向量与量化参数由 `FP16` 转换成 `FP32`，**计算需要高精度**

> 值得一提的是：我们选择用一个 `int` 来存放一个解包后权重，而没有选择更小位数的 `uint8_t`，本质原因是 RTX 5060 的通用寄存器以 32 bit 为单位，单独参与运算通常也是采用同样粒度的寄存器

```cpp
__global__ void w4a16_kernel(
    const __half* x,
    const uint8_t* w_q,
    const __half* scales,
    const __half* zeros,
    __half* y,
    int64_t N, int K, int G)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int wq_cols = K / 2;
    const int groups_per_row = K / G;
    const int bytes_per_group = G / 2;
    if (tid >= N) return;
    
    float acc = 0.0f;
    for (int b = 0; b < wq_cols; b++)
    {
        const int packed = w_q[tid * wq_cols + b];
        const int q0 = packed & 0xfu;
        const int q1 = (packed >> 4) & 0xfu;
        
        const int group_idx = b / bytes_per_group + tid * groups_per_row;
        
        const float scale = __half2float(scales[group_idx]);
        const float zero = __half2float(zeros[group_idx]);
        
        const float w0 = (q0 - zero) * scale;
        const float w1 = (q1 - zero) * scale;
        
        const float x0 = __half2float(x[2 * b]);
        const float x1 = __half2float(x[2 * b + 1]);
        acc += x0 * w0 + x1 * w1;
    }
    y[tid] = __float2half_rn(acc);
}
```
上面的代码很直观，但存在巨大的**访存问题**，当两个相邻线程在同一个 `b` 的循环里时，读取 `w_q` 元素相差 `wq_cols` 地址，我们可以先看 ncu 分析：

> 以下内容来自本机的 Nsight Compute 终端输出，用例规模 `N = 4096` `K =4096`，`gird x block = (512, 1, 1) x (256, 1, 1)`

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         5.97
    SM Frequency                    Mhz       773.32
    Elapsed Cycles                cycle    1,190,156
    Memory Throughput                 %        56.24
    DRAM Throughput                   %         8.54
    Duration                         ms         1.53
    L1/TEX Cache Throughput           %        98.30
    L2 Cache Throughput               %         1.64
    SM Active Cycles              cycle   677,132.92
    Compute (SM) Throughput           %         8.88
    ----------------------- ----------- ------------

    Section: Memory Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Memory Throughput        Gbyte/s        16.30
    Mem Busy                       %        56.24
    Max Bandwidth                  %         8.55
    L1/TEX Hit Rate                %        98.90
    L2 Compression Ratio                        0
    L2 Hit Rate                    %         1.70
    Mem Pipes Busy                 %         8.55
    -------------------- ----------- ------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit Barriers                  block           24
    Block Limit SM                        block           24
    Block Limit Registers                 block            6
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        16.72
    Achieved Active Warps Per SM           warp         8.03
    ------------------------------- ----------- ------------   
```

- 接回刚刚提到的固定循环 `b`，相邻线程读取 `w_q` 的地址相隔 2048 B，所以一条权重加载指令，32 个 lane 各读一个 B，却触及了 32 个不同的 sector，这点与 **L1/TEX Throughput 98.30%** 密切相关，**L1/TEX 处理压力大**。
- 同样注意到 **L1/TEX Hit Rate 98.90%**，说明请求到 L1 的大多数请求还是命中了，这是源于每一个线程在下一个 `b` 会继续在当前 sector 读取元素。
- 还有两个很关键的指标 **Achieved Occupancy 16.72%**，**Achieved Active Warps Per SM  8.03Warps**。由于 RTX 5060 Laptop GPU 是 SM_12，硬件规定每个 SM 最多同时驻留 48 个 warp，这里 warp 活跃度低是因为一共 4096 行权重，一共只发射了 16 个 block，导致一个 SM 里的 warp 太少，warp 调度器很难通过频繁发射 warp 来掩盖访存延迟。**Compute (SM) Throughput 8.88%** 远低于访存相关吞吐 **Memory Throughput 56.24%**，显然是**计算瓶颈**。
- 对于 SM Occupancy，其理论最大值为 **Theoretical Occupancy 100%，Theoretical Active Warps per SM 48Warps**

# Warp profiling

接下来可以一个 warp 处理一行压缩权重，这能很好地解决相邻线程访存地址跨度太大的问题。每一个 warp 计算一行结束后，进行一次 warp sum reduction

```cpp
__global__ void w4a16_kernel(
    const __half* x,
    const uint8_t* w_q,
    const __half* scales,
    const __half* zeros,
    __half* y,
    int64_t N, int K, int G)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int wq_cols = K / 2;
    const int groups_per_row = K / G;
    const int bytes_per_group = G / 2;
    const int lane = threadIdx.x & 31u;
    const int warp_idx = threadIdx.x >> 5;
    const int global_warp_idx = tid >> 5;
    if (global_warp_idx >= N) return;
    
    float acc = 0.0f;
    for (int b = lane; b < wq_cols; b += 32)
    {
        const int packed = w_q[global_warp_idx * wq_cols + b];
        const int q0 = packed & 0xfu;
        const int q1 = (packed >> 4) & 0xfu;
        
        const int global_group_idx = b / bytes_per_group + groups_per_row * global_warp_idx;
        const float scale = __half2float(scales[global_group_idx]);
        const float zero = __half2float(zeros[global_group_idx]);
        
        const float w0 = (q0 - zero) * scale;
        const float w1 = (q1 - zero) * scale;
        
        const float x0 = __half2float(x[2 * b]);
        const float x1 = __half2float(x[2 * b + 1]);
        
        acc += x0 * w0 + x1 * w1;
    }
    
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    
    if (lane == 0)
        y[global_warp_idx] = __float2half_rn(acc);
}
```

下面还是先看 ncu 分析：

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.99
    SM Frequency                    Ghz         1.25
    Elapsed Cycles                cycle      161,923
    Memory Throughput                 %        63.91
    DRAM Throughput                   %        23.00
    Duration                         us       128.61
    L1/TEX Cache Throughput           %        66.83
    L2 Cache Throughput               %         9.94
    SM Active Cycles              cycle   154,162.92
    Compute (SM) Throughput           %        66.79
    ----------------------- ----------- ------------                            

    Section: Memory Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Memory Throughput        Gbyte/s        80.87
    Mem Busy                       %        32.00
    Max Bandwidth                  %        63.91
    L1/TEX Hit Rate                %        90.08
    L2 Compression Ratio                        0
    L2 Hit Rate                    %         3.97
    Mem Pipes Busy                 %        63.91
    -------------------- ----------- ------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit Barriers                  block           24
    Block Limit SM                        block           24
    Block Limit Registers                 block            6
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        87.77
    Achieved Active Warps Per SM           warp        42.13
    ------------------------------- ----------- ------------
```

- 首先值得关注的是 **Achieved Occupancy 87.77%**，**Achieved Active Warps Per SM 42.13Warps**，相较于上一个版本，这个版本发射的总 block 数量为 512 个，一个 SM 内可以驻留更多的 warp，相应的 Compute (SM) Throughput 也从 8.88% 上升至 66.79%
- 其次是之前探讨过的话题，用 warp 为单位处理一行 `w_q` 能明显改善访存。**L1/TEX Cache Throughput 66.83%**，**访存压力明显有所改善**。
- 当前版本的 Compute (SM) Throughput 66.79% 和 Memory Throughput 63.91% 大致相当，计算和访存是平衡的

通过观察不难看出，每一次循环我们都进行了一次解包，但显然一个 group 横跨的范围是两个循环，因此下一个优化版本我们可以考虑按量化参数组循环，**复用量化参数**

# group profiling

为了能够**复用同一组的量化参数**，这里给出一个解法：在外层加上一个 group 循环

```cpp
__global__ void w4a16_kernel(
    const __half* x,
    const uint8_t* w_q,
    const __half* scales,
    const __half* zeros,
    __half* y,
    int64_t N, int K, int G)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int wq_cols = K / 2;
    const int groups_per_row = K / G;
    const int bytes_per_group = G / 2;
    const int lane = threadIdx.x & 31u;
    const int warp_idx = threadIdx.x >> 5;
    const int global_warp_idx = tid >> 5;
    if (global_warp_idx >= N) return;
    
    float acc = 0.0f;
    for (int g = 0; g < groups_per_row; g++)
    {
        const int global_group_idx = g + groups_per_row * global_warp_idx;
        const float scale = __half2float(scales[global_group_idx]);
        const float zero = __half2float(zeros[global_group_idx]);
        
        for (int b_in_group = lane; b_in_group < bytes_per_group; b_in_group += 32)
        {
            const int b = g * bytes_per_group + b_in_group;
            const int packed = w_q[global_warp_idx * wq_cols + b];
            const int q0 = packed & 0xfu;
            const int q1 = (packed >> 4) & 0xfu;
            
            const float w0 = (q0 - zero) * scale;
            const float w1 = (q1 - zero) * scale;
            
            const float x0 = __half2float(x[2 * b]);
            const float x1 = __half2float(x[2 * b + 1]);
            
            acc += x0 * w0 + x1 * w1;
        }
    }
    
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    
    if (lane == 0)
        y[global_warp_idx] = __float2half_rn(acc);
}
```

看具体的 ncu 性能指标：

> 本次性能测试新加了 Section: Compute Workload Analysis 与 Section: Instruction Statistics，方便与后续的版本分析对比

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.99
    SM Frequency                    Ghz         1.29
    Elapsed Cycles                cycle      141,435
    Memory Throughput                 %        58.94
    DRAM Throughput                   %        27.95
    Duration                         us       109.34
    L1/TEX Cache Throughput           %        62.66
    L2 Cache Throughput               %        11.97
    SM Active Cycles              cycle   132,239.88
    Compute (SM) Throughput           %        65.64
    ----------------------- ----------- ------------        

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         2.79
    Executed Ipc Elapsed  inst/cycle         2.63
    Issue Slots Busy               %        69.78
    Issued Ipc Active     inst/cycle         2.79
    SM Busy                        %        69.78
    -------------------- ----------- ------------  

    Section: Memory Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Memory Throughput        Gbyte/s        98.29
    Mem Busy                       %        29.52
    Max Bandwidth                  %        58.94
    L1/TEX Hit Rate                %        88.99
    L2 Compression Ratio                        0
    L2 Hit Rate                    %         4.91
    Mem Pipes Busy                 %        58.94
    -------------------- ----------- ------------

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst    92,278.15
    Executed Instructions                           inst    9,596,928
    Avg. Issued Instructions Per Scheduler          inst    92,278.15
    Issued Instructions                             inst    9,596,928
    ---------------------------------------- ----------- ------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit Barriers                  block           24
    Block Limit SM                        block           24
    Block Limit Registers                 block            6
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        85.70
    Achieved Active Warps Per SM           warp        41.13
    ------------------------------- ----------- ------------
```

