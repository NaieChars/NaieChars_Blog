---
title: CUDA 并行归约性能指标分析
published: 2026-10-05
pinned: true
tags: [CUDA]
category: 技术
draft: false
---

[实验源码与本地测量数据](https://github.com/NaieChars/CUDA-/tree/main/labs/reduction)

> **配置**：
> - RTX 5060 Laptop GPU，-26 SM，L2=32 MiB，
> - 驱动 592.00，
> - CUDA 12.8.61，
> - Nsight Compute 2025.2.0；
> - Windows，
> - Release、原配置 native 架构、-lineinfo

# 1. 前言
**归约**即对一个数组进行某些操作使得最后输出一个数。本实验的归约指将一个数组全部元素求和，输出最终求和结果。与[合并访存](cuda_合并访存.md)类似，本实验从**三个版本的核函数**进行性能分析：**shared 版**（每个线程读取一个数值 + 共享内存），**multi 版**（每个线程读取多个数值，串行累加再求和），**shuffle 版**（线程之间的通讯不经过共享内存，通过 `__shfl_down_sync` 实现）。

# 2. 实验结果

测试对象包含 5 个：shared I=1，multi_item I=1，shuffle I=1，multi_item I=4，shuffle I=4

| 配置 | 完整归约 μs：均值 ± 样本SD | 中位数 μs | 范围 μs | 输入有效带宽 GB/s：逐次值均值 |
| --- | --- | --- | --- | --- |
| shared I=1 | 470.09 ± 9.34 | 466.71 | 459.82～485.99 | 142.81 |
| multi_item I=4 | 201.90 ± 8.07 | 199.12 | 194.04～221.52 | 332.83 |
| shuffle I=4 | 203.20 ± 9.59 | 198.31 | 193.73～220.44 | 330.90 |
| multi_item I=1 | 516.65 ± 24.04 | 507.17 | 495.83～564.99 | 130.13 |
| shuffle I=1 | 365.79 ± 12.87 | 365.17 | 348.45～390.64 | 183.67 |

**主结果**：multi_item / shuffle 相对 shared 的平均时间加速分别为 2.328× / 2.313×。shuffle − multi_item 的轮次配对**时间差**均值为 1.30 μs，95% t 区间 [-5.17, 7.77] μs：

# 3. 分析

| 自然频率首轮；3 次均值 | shared I=1 | multi_item I=4 | shuffle I=4 |
| --- | --- | --- | --- |
| 首轮 μs | 462.37 | 200.07 | 199.17 |
| 实际 SM GHz | 2.69 | 2.69 | 2.49 |
| DRAM 吞吐 % | 45.91 | 97.67 | 97.47 |
| 动态 warp 指令（百万） | 64.03 | 24.43 | 13.53 |
| shared load warp 指令 | **1,638,400.00** | **409,600.00** | 16,384.00 |
| shared store warp 指令 | 1,310,720.00 | 327,680.00 | 131,072.00 |
| Occupancy % | 85.85 | 91.69 | 87.81 |
| Eligible warps / scheduler active cycle | 0.84 | 0.85 | 0.44 |
| Issue active %（elapsed） | 49.44 | 43.69 | 26.31 |
| Long scoreboard / issue-active 原始 ratio | 6.86 | 14.57 | 33.22 |

## 3.1 比较 shared 到 multi_item 指标
- **理想合并访存**：  
  - 三版首轮 global load 都是 524,288 个 request、2,097,152 个 sector，即 4 sector/request；每个 request 对应完整 warp 的 32 个连续 float，合并读取已达理想值

- **block 数量减少，multi_item 摊薄 block 内通信/控制成本**:  
  - 线程数量减少为原来的 1/4，相应的 block 数减为 1/4。
  - **shared load/store warp 指令**由 1,638,400/1,310,720 降为 409,600/327,680，均恰好减为 1/4：每一个 block 的共享内存仍然要存 265 （BlockDim.x = 256）个线程 sum，所以俩版本后面累加 sum 产生的指令是一样的，只是说由于 block 数量减少为 1/4，整个 Grid 的 shared memory 指令操作减少了。
  - 理论上**同步指令数量**也减少为 1/4，但不代表同步耗时减少 4 倍。
  - **总 warp 指令**由 64.029M 降为 24.429M（减少 61.8%）：没有减少 3/4 的原因是 multi 版本还增加了线程的循环指令，以及 multi 版本**增加了浮点加法指令**等等
  - 每输入元素对应的归一化 Warp 指令计数从 3.816 降为 1.456

> 根据：`smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`（浮点加法指令数），
> - shared I = 1： 16,711,680
> - multi I = 4： 20,955,136。
> - 可以通过计算得到 shared I=1 的浮点加法数恰好等于：255 x 65536 = 16,711,680（一个 block 内进行 255 次加法，乘上 block 数量，也就是每个 block 内的树形归约），multi_item I=4 的浮点加法数则恰好等于：N + 1/4 x 16,711,680 = 20,955,136。这里的第一项 N 对应局部累加，第二项对应每个 Block 内部的树形归约。

证据支持**摊薄 block 内通信/控制成本**，即每个 Block 处理的输入量增加了四倍，因而 shared memory 归约、同步和部分 Block 级控制操作，分摊到每个输入元素上的成本降低了。

- **进一步分析 LSU 指标**：  
  - shared 版 L1/TEX throughput=87.36%，与 l1tex__lsuin_requests=87.36% 一致；shared data-pipe wavefront 利用率只有 43.51%。说明高的是 LSU 请求入口压力，但不足以称为 shared 数据带宽打满。
  - 自然频率主采样中 barrier/short-scoreboard 累积 warp-cycle 从 189.47M/289.13M 降到 47.32M/62.56M。它们与结构变化方向一致

## 3.2 比较 multi_item 到 shuffle 指标
- **通信减少**
  -  I=4、grid 和读入数据保持相同；shared load/store warp 指令再降为 16,384/131,072（分别减少 96%/60%），总 warp 指令从 24.429M 降为 13.533M（减少 44.6%）。源码每 block 的屏障由 8 次降为 1 次。
- **代价**
  - shuffle 的有效线程 FADD 数是 39,862,272，反而高于 multi_item 的 20,955,136。所有 lane 都执行 shuffle 加法，较高 lane 的中间结果没有用作最终和。