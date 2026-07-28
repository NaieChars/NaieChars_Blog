---
title: GPU 微架构与 Nsight Compute 性能分析
published: 2026-07-28
pinned: true
description: 本文简单介绍了一些关于 GPU 微架构的知识，从存储层次、执行流水线、Wrap停顿以及Nsight Compute 对应指标这几个方面来展开
tags: [GPU微架构]
category: 技术
draft: false
---

## 存储层次（Memory Hierarchy）

### Cache（缓存）
**SM**从**Global Memory（显存）** 访问数据非常慢（几百个时钟周期），而且程序员必须显式地用代码（`__shared__`，显式调用显存数据）把数据从显存搬进共享内存。
**Cache**是GPU核心和显存之间的一个**临时存储器（极小极快全自动）**，设计目的便是为了隐藏 Global Memory 高延迟  
它有两个核心思想：
- **时间局部性**：刚访问过的数据，Cache觉得未来可能再次访问，会将其自动存放在其内部，下次访问直接从Cache里面取。
- **空间局部性**：当你访问一堆显存里的连续内存时，GPU 会自动把大片一次性放进Cache里，之后的访问就不用挨个从显存读取。

#### Cache Hit / Miss（缓存命中与未命中）
> 关于Cache Hit Rate（缓存命中率）更深入的话题，后面细说。

**Hit 表示数据已经在高速缓存中，Miss 表示需要继续向更慢的存储层查找**。Cache有两层（下面细说），例如线程需要 a[100]，GPU先查看 L1 Cache，如果有直接返回 L1 Hit，若没有，查看 L2 Cache。相应的若 L2 Chache 有，返回 L2 Hit，若没有才访问显存。

### Cache Hierarchy（分层缓存）
分层缓存每层有一个特点：**越往下（朝显存方向），能存储的数据越多，访问速度越慢**。Cache 有两层，第一层是**紧挨着 SM 的 L1 Cache**，每个 SM 都有自己的 L1 Cache，彼此不能通信。而**第二层 L2 Cache 在 GPU 上只有一个**（逻辑上的），可以让不同 SM 共享缓存的数据。  
一次完整的访问流程：

$$
\text{Thread} \quad \longrightarrow \quad \text{Register} \quad \longrightarrow \quad \text{L1 Cache} \quad \longrightarrow \quad \text{L2 Cache} \quad \longrightarrow \quad \text{Global Memory}
$$

> [!NOte]
> **L2 的命中率不是对全部访问计算，而是针对进入 L2 的访问计算**。例如100次访问 L1：80 次命中，所以 L1 Hit Rate = 80%。剩下的 20 次 Miss 继续访问 L2，若 20 次里面有 18 次命中，则 L2 Hit Rate = 18 / 20 = 90%
>

#### Shared Memory 与 L1 关系
在很多 NVIDIA GPU 架构中：**Shared Memory 和 L1 Cache 在硬件资源上有一定共享关系**（具体实现会因架构而不同），但从编程模型来看，它们是两种完全不同的存储：
- L1 Cache：硬件自动管理，程序员不能决定缓存哪些数据
- Shared Memory：程序员显式申请、读写和管理

### Cache Hit Rate（缓存命中率）
GPU对于 Cache 的第一个访问几乎都是 Miss，这叫做 **Cold Miss（冷启动未命中）**。  
**如何提高 Hit Rate?** 连续访问同一块内存，或者是访问一块连续的内存。

#### Cache Line（缓存行）
GPU读取内存是以**缓存行的形式来读取一整块内存**，而不是一次只读一个 float 或 int。这也说明了上面访问一块连续的内存命中率会很高。相应的随机访问的命中率就很低，这就是很多图算法、稀疏矩阵程序性能难优化的原因

#### 性能分析不只看命中率
即使 L1 命中率高达99%，但是如果是10000次访问，有将近199次Miss，Miss的时间加起来就可能带来和99% Hit 相当甚至更大的等待时间。所以，我们分析性能时一定要结合：Memory Throughput（显存是否已经跑满）、SM Throughput（计算单元在摸鱼没）、Warp Stall（Warp 为什么停下），这些后面都会提到

### Memory Throughput（内存吞吐率）
**这是判断 Memory Bound（访存瓶颈） 的第一指标！Memory Throughput 即 GPU 每秒能够搬运多少数据（GPU与显存之间，不包括L1、L2内部数据流动）**，在 Nsight 里通常是按百分比计量，表示已经使用了理论最大内存宽带的百分之多少。  

#### 访存瓶颈与计算瓶颈
如果一个核函数Memory Throughput几乎接近100%，SM Throughput 反而很低，那么这个程序存在访存瓶颈，反之则是计算瓶颈。如果两边都比较高，则说明优化已经很不错了。

---

## 执行系统（Execution Pipeline）

### Pipeline（流水线）
流水线的思想：不让硬件等前一条指令完全结束，而是让不同阶段同时处理不同指令。
**CPU Pipeline 与 GPU Pipeline 的区别**
CPU 的目的是让线程尽可能快，所以CPU有分支预测，乱序执行，大量复制控制逻辑；而 GPU 的目标是让大量线程并行运行，所以要让更多 Wrap 交替执行。  

#### Latency Hiding（延迟隐藏）
这是 GPU 优化最核心的思想。当调度器发射一个 Wrap 给执行单元时，在 Load 数据阶段可能会有较高的访问延迟，这时候 GPU 直接会切换到另一个 Wrap，这便是用计算隐藏延迟：**用大量 Warp 的切换，隐藏单个 Warp 的等待时间**   
显然，如果一个 SM 里只有一个 wrap，会导致性能下降，这一就是为什么一个SM可以计算大量wrap。

#### Pipeline Stall（流水线停顿）
**GPU 常见 Stall 原因**
- Memory Stall（访存等待）
- Dependency Stall（依赖等待）：下一个指令用到的数据是上一句得出的，必须等上一步算完
- Execution Stall（执行等待）：计算单元都没有空

### Scoreboard（记分牌机制）
**Scoreboard 是 GPU 内部记录指令依赖状态的一张表**，只负责记录哪些寄存器还在等结果，哪些操作没完成，哪些wrap可以继续执行，可以理解为一份任务清单。这也就是调度器能够灵活安排不同 Wrap 上线程的核心所在，像上面所说的访存等待的底层原因便是 Scoreboard 检测到仍有数据没有传回来。每个周期 Scheduler（调度器）都会询问每一个wrap，实则是在查询 Scoreboard。

### Stall LG Throttle
GPU 中，计算是 CUDA Core 的核心职责，它把读取数据交给Load/Store Unit（LSU）处理，LSU 接收调度器的指令，向 L1 查找相应数据。当 Wrap 疯狂发送多个 Load 请求使得 LSU 满了，于是只能等待。这便是 Throttle（即门口排队的指令太多，请求发不出去）。  
**优化 LG Throttle**：
- 提高数据复用性
- 改善成访问连续数据
- 减少读取的数据大小（AoS 与 SoA）

### Memory Coalescing（内存合并访问）
这个概念直接决定你的 Warp 一次 Global Memory 访问，到底需要几次显存事务，这也是很多 CUDA 程序性能差距几十倍的原因。  
GPU访问 Global Memory 时以 **Memory Transaction（内存事务）** 为单位（类似于 Cache Line），当所一个 Wrap 里 32 个线程连续访问时 `x = data[i];`，那么Wrap直接访问32个连续地址，一次性带回，刚好一次 Memory Transaction （128 Byte）

### Shared Memory Bank Conflict（共享内存银行冲突）
硬件内部将shared memory划分成了很多小块，每个小块叫**bank**，这说明其并不是一整块连续内存。假设有32个bank，恰巧可以服务32个线程，如果每一个bank只映射一个thread，那么称这种情况为**完美访问**。如果多个线程需要访问同一个bank，那么便会造成**Bank Conflict**  
一个**经典案例**：假设对一个矩阵按行连续读取，效果非常好（矩阵按行存储），但是一旦跨行访问，就变成访问显存，导致效率下降，于是我们想到用共享内存。但是如果共享内存布局不好，又会产生Bank Conflict，所以优化矩阵转置时经常加入padding，比如 `__shared__ float tile[32][33];`33是为了避免32倍数访问导致同一个bank。

### Occupancy（占用率）
**Occupancy 表示一个 SM 中，当前活跃 Warp 数量占 SM 最大支持 Warp 数量的比例**。提高 Occupancy 的本质便是**延迟隐藏**  
高Occupancy不一定好，其中有很多情况，需要结合很多参数来看


## 总结
以上便是一些常用到的 GPU 性能指标的简单介绍，当我们判断一个程序是否优化得很好时，不能只盯着一个指标从绝对的偏大偏小来看。因为每一个部分都与其他的紧密相连，我们在性能分析的时候也要全局判断。