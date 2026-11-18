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

还可以进一步在外层循环中让第一个线程读取，其他线程**通过 warp 内广播**来获取相同参数

```cpp
float scale, zero;
if (lane == 0)
{
    scale = __half2float(scales[global_group_idx]);
    zero = __half2float(zeros[global_group_idx]);
}
zero = __shfl_down_sync(0xffffffffu, zero, 0);
scale = __shfl_down_sync(0xffffffffu, scale, 0);
```


先具体的 ncu 性能指标：

> 本次性能测试新加了 Section: Compute Workload Analysis 与 Section: Instruction Statistics，方便与后续的版本分析对比

> 第一项为无广播版本的指标

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
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
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        85.70
    Achieved Active Warps Per SM           warp        41.13
    ------------------------------- ----------- ------------
```

> 第二项为有广播版本的指标

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Elapsed Cycles                cycle      150,604
    Memory Throughput                 %        68.90
    DRAM Throughput                   %        21.69
    Duration                         us       117.09
    L1/TEX Cache Throughput           %        71.20
    L2 Cache Throughput               %         9.99
    SM Active Cycles              cycle   144,713.19
    Compute (SM) Throughput           %        68.90
    ----------------------- ----------- ------------
    
    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         2.79
    Executed Ipc Elapsed  inst/cycle         2.70
    Issue Slots Busy               %        69.86
    Issued Ipc Active     inst/cycle         2.79
    SM Busy                        %        69.86
    -------------------- ----------- ------------

    Section: Memory Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Memory Throughput        Gbyte/s        76.25
    Mem Busy                       %        34.50
    Max Bandwidth                  %        68.90
    L1/TEX Hit Rate                %        88.97
    L2 Compression Ratio                        0
    L2 Hit Rate                    %         4.96
    Mem Pipes Busy                 %        68.90
    -------------------- ----------- ------------

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst   101,100.31
    Executed Instructions                           inst   10,514,432
    Avg. Issued Instructions Per Scheduler          inst   101,100.31
    Issued Instructions                             inst   10,514,432
    ---------------------------------------- ----------- ------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        85.77
    Achieved Active Warps Per SM           warp        41.17
    ------------------------------- ----------- ------------
```

可以看到，有广播的版本反而比无广播版本的耗时更多。具体原因可以从指令看出：
- 前者 warp 调度器发射的 warp 级指令总数为 **Issued Instructions 9,596,928**；后者的该项值为 **Issued Instructions 10,514,432**

指令数量变多主要来自于几个方面，这里将从 SASS 分析具体每一条相关汇编指令执行了多少次：

> 可以用 ncu 测量每一条 SASS 实际执行了几次

- 广播版本新增了如下指令：（后者是动态 warp 执行次数）
  - `SHFL.IDX` 广播指令：262,144
  - `ISETP.NE.AND` 生成谓词分支：131,072
  - `BRA` 条件跳转：131,072
  - `BSSY.RECONVERGENT`：设定重汇合位置：131,072
  - `BSYNC.RECONVERGENT`：执行重汇合：262,144

上述新增指令累计 917,504，正好是多出来的指令数。

多出来的指令数抵消掉了 warp 内广播所带来的收益（查资料可得，`SHFL` 约 29 个周期，L1 命中加载约 44 个周期，两者的延迟不同），并且相邻线程命中的是同一个 sector。因此后续的版本继续是以非广播为基线优化

# 向量加载
为了进一步减少指令数以及访存压力，可以采取向量加载。本次版本选择 `W-q` 用 `uint32_t` 加载，`x` 用 `uint4` 加载。

在开始优化前，建议先查看上一个版本的 SASS 指令，因为很多情况下，编译器会自动发射宽加载指令。

```
LDG.E.U16 R29, desc[UR8][R8.64]
LDG.E.U16 R27, desc[UR8][R8.64+0x2]  
```

以上是编译器生成的加载连续两个相邻 `x` 的指令，可以看到这里编译器并没有优化为宽加载。

值得注意的是，向量加载通常要求严格的**地址对齐**，在本次例子中，`x` 需要 16 字节对齐，`w_q` 需要 4 字节对齐。否则要单独处理跨扇区访问。

优化代码如下：

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
    const int64_t global_warp_idx = tid >> 5;
    
    if (global_warp_idx >= N) return;
    
    const int64_t wq_row_begin = global_warp_idx * wq_cols;
    const int64_t group_row_begin = glabal_warp_idx + groups_per_row;
    const int lane_in_group = lane % 16;
    float acc = 0.0f;
    
    for (int g = lane >> 4; g < groups_per_rows; g += 20)
    {
        const int64_t group_id = group_row + g;
        const float scale = __half2float(scales[group_id]);
        const float zero = __half2float(zeros[group_id]);
        
        const int b = g * bytes_per_group + lane_in_group * 4; // b: 第一个打包权重字节的下标
        const uint32_t w_packed = *reinterpret_cast<const uint32_t*>(w_q + wq_row_begin + b);
        const uint4 x_packed = *reinterpret_cast<const uint4*>(x + 2 * b);
        const uint32_t w_word[4] = {x_packed.x, x_packed.y, x_packed.z, x_packed.w};
        
        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            const uint32_t packed = w_packed >> (i * 8);
            const int q0 = packed & 0xfu;
            const int q1 = (packed >> 4) & 0xFu;
            
            const float w0 = (q0 - zero) * scale;
            const float w1 = (q1 - zero) * scale;
            
            const uint32_t x_bits = x_words[i];
            const __half x0 = __ushort_as_half(static_cast<uint16_t>(x_bits));
            const __half x1 = __ushort_as_half(static_cast<uint16_t>(x_bits >> 16));
            
            acc += w0 * __half2float(x0);
            acc += w1 * __half2float(x1);
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

以下是当前版本的 ncu 指标：

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Elapsed Cycles                cycle       66,449
    Memory Throughput                 %        46.78
    DRAM Throughput                   %        46.78
    Duration                         us        54.46
    L1/TEX Cache Throughput           %        36.25
    L2 Cache Throughput               %        11.84
    SM Active Cycles              cycle    61,716.81
    Compute (SM) Throughput           %        75.26
    ----------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         3.23
    Executed Ipc Elapsed  inst/cycle         3.01
    Issue Slots Busy               %        80.79
    Issued Ipc Active     inst/cycle         3.23
    SM Busy                        %        80.79
    -------------------- ----------- ------------                                             

    Section: Memory Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Memory Throughput        Gbyte/s       164.38
    Mem Busy                       %        28.40
    Max Bandwidth                  %        46.78
    L1/TEX Hit Rate                %        80.24
    L2 Compression Ratio                        0
    L2 Hit Rate                    %         3.75
    Mem Pipes Busy                 %        33.77
    -------------------- ----------- ------------

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst    49,860.92
    Executed Instructions                           inst    5,185,536
    Avg. Issued Instructions Per Scheduler          inst    49,860.92
    Issued Instructions                             inst    5,185,536
    ---------------------------------------- ----------- ------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        85.95
    Achieved Active Warps Per SM           warp        41.26
    ------------------------------- ----------- ------------
```

- 首先关注 **Memory Throughput** 与 **Compute (SM) Throughput**，这次向量加载明显降低了访存的压力，导致该 kernel 的**瓶颈转移到了计算上**
- 向量加载最直接的影响是 **Issued Instructions 5,185,536**，比前版本少了将近一半的指令数，这部分指令数主要来自于访问 glabal memory
- 详细分析计算瓶颈：SM 吞吐量的总汇总值为 74.99%，展开相关子指标，整理如下：

```
--------------- ----------- ---------------
Metric Name     Metric Unit    Metric Value
--------------- ----------- ---------------
Issue Active              %           75.09
Inst Executed             %           75.09
ALUHeavy                  %           57.48
FMAHeavy                  %           50.69
LSU                       %           33.69
```

```
-----------------------------------------------------------------------  -----------  ------------
Metric Name                                                              Metric Unit  Metric Value
-----------------------------------------------------------------------  -----------  ------------
smsp__warps_active.avg.peak_sustained                                      warp                12
smsp__warps_active.avg.per_cycle_active                                    warp             10.14
smsp__warps_eligible.avg.per_cycle_active                                  warp              4.63
smsp__issue_active.avg.per_cycle_active                                   warp/cycle         0.80
smsp__issue_active.avg.pct_of_peak_sustained_active                         %                79.56
smsp__issue_inst0.avg.pct_of_peak_sustained_active                          %                20.44

smsp__warp_issue_stalled_not_selected_per_warp_active.pct                  %                38.04
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct               %                22.63
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct            %                15.87
smsp__warp_issue_stalled_wait_per_warp_active.pct                          %                 7.12
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct              %                 1.34
-----------------------------------------------------------------------  -----------  ------------
```

**Issue Active / Inst Executed 75.09%** 整个 **SM 的指令发射与执行是比较繁忙的**，平均每一个调度器有 10.14 个驻留 warp，4.63 个就绪 warp，每周期约发射 0.8 条指令。`smsp__issue_inst0.avg.pct_of_peak_sustained_active 20.44%` 表示约有 1/5 的周期没有就绪 warp，`smsp__warp_issue_stalled_not_selected_per_warp_active.pct 38.04%` 表示同时有多个 warp 就绪。这表明**就绪的 warp 在时间上分布不均匀**。

后续通过 ncu Source 可以定位指标对应的具体的 SASS 指令，warp 未准备就绪的指令出自于这里：**压缩权重加载到首次解包之间**

0xb40  LDG.E     R16, ...            // 加载压缩权重
0xc90  LOP3.LUT  R32, R16, 0xf, ...  // 等 R16 就绪，取低 4 位

所以总的来说，当前版本的主要问题在于：一方面要发射大量解包、转换和反量化指令，另一方面在部分周期等压缩权重就绪。优化策略是，可以提前进行加载指令的发射。

# inline PTX
进一步优化可以采取 PTX 虚拟指令：
- 将解包过程转换成 PTX
- 将小分支转换成 PTX

对于小分支，例如 `if (lane == 0)`，SASS 已经在最后一条线程间 shuffle 后用 `@p0 EXIT` 让其余 lane 退出。这表明编译器其实是可以自动用谓词优化了小分支，这里就不再对其写 PTX

可以将按位与取位改写成 PTX：

```cpp
int q1;
asm("bfe.u32 %0, %1, 4, 4;"
    : "=r"(q1)
    : "r"(packed));
```

上述 PTX 表示直接从第 4 位开始取 4 位放入 `p1`

实测 ncu 相关指标如下：

```
无 PTX
    Avg. Executed Instructions Per Scheduler        inst    45,489.23
    Executed Instructions                           inst    4,730,880
    Avg. Issued Instructions Per Scheduler          inst    45,489.23
    Issued Instructions                             inst    4,730,880

有 PTX
    Avg. Executed Instructions Per Scheduler        inst    46,119.38
    Executed Instructions                           inst    4,796,416
    Avg. Issued Instructions Per Scheduler          inst    46,119.38
    Issued Instructions                             inst    4,796,416
```

可看到，更改为 PTX 后**指令数量反而变多**，这里核对了 SASS 发现多出的指令来源于取最高 4 位：

源代码里，内层循环只有 4 次，使用 `#pragma unroll` 展开循环。在取最后高位时，编译器知道最后 `i = 3`，所以原版的第 4 次循环可以简化为 `w_packed >> 28`（右移后只剩下最高 4 位，**`& 0xfu` 可以省略**），省略的证据来自于下面的 SASS 指令：

```
C++ 版
0x1110  SHF.R.U32.HI R26, RZ, 0x1c, R26   // 右移 28 位
0x1150  I2FP.F32.U32 R26, R26

bfe.u32 版
0x10b0  SHF.R.U32.HI R26, RZ, 0x18, R26   // 右移 24 位，q0 也用这份结果
0x10f0  SHF.R.U32.HI R9, RZ, 0x4, R26     // 再右移 4 位
0x1140  SGXT.U32     R9, R9, 0x4          // 再取低 4 位并做零扩展
0x1180  I2FP.F32.S32 R9, R9
```

这里 0x10b0 的右移还供同一字节的 q0 使用，整个循环来看，两版的移位指令总数相同。净增来自掩码和位域提取指令从 14 条变成了 16 条。`N = K = 4096`、`G = 128` 时，一行一个 warp，所以有 4,096 个 warp。每个 warp 在源码里跑 32 / 2 = 16 轮 g 循环，编译器把两轮合进一个 SASS 循环体，所以实际跑 8 轮。每轮净增 2 条，刚好是 4096 × 8 × 2 = 65,536

当然还可以对其他代码写 PTX，这里实际测量个人感觉提升效果不大。所以暂且不采用 PTX