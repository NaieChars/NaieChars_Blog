---
title: 分析 Nsight Compute 性能指标参数
published: 2026-10-07
pinned: true
description: cuda 性能分析常用指标汇总
tags: [CUDA, Nsight Compute]
category: 技术
draft: false
---

# Nsight Compute 常用性能指标
## 指标名称
### 常用前缀
| 前缀 | 对应硬件或范围 |
| :--- | :--- |
| `gpu__` | GPU 整体层面的汇总指标 |
| `sm__` | SM（流式多处理器）层面 |
| `smsp__` | SM 子分区 / Warp 调度器层面 |
| `l1tex__` | L1 / Texture 相关硬件 |
| `lts__` | L2 Cache 相关硬件 |
| `dram__` | GPU 外部显存 DRAM |
| `launch__` | Kernel 启动配置及资源限制 |

### 常用后缀
| 后缀 | 人话解释 |
| :--- | :--- |
| `.sum` | 将相关硬件实例的计数累加，常用于总量 |
| `.avg` | 对相关硬件实例求平均 |
| `.pct_of_peak_sustained_elapsed` | 相对于持续峰值的百分比，按整个测量时段计算 |
| `.pct_of_peak_sustained_active` | 相对于持续峰值的百分比，按单元活动周期计算 |
| `.ratio` | 比率或归一化后的指标，不一定是百分比 |
| `.per_cycle_active` | 按硬件单元活动周期计算的平均值 |

## Time 指标
`gpu__time_duration.sum    399.52 us`：NCU 测得的GPU执行时长，单位微秒  

## 吞吐率
1. `dram__bytes_op_read.sum    67.20 Mbyte`  
含义：测量期间，DRAM 执行的读取操作对应的数据传输总量。记录的是实际显存流量口径。

2. `dram__bytes_op_write.sum    58.41 Mbyte`
含义：测量期间，DRAM 执行的写入操作对应的数据传输总量。

> 以上两个指标统计的层级是 DRAM，因此**不能**把这两个指标直接等同于源代码里的**逻辑读写数据量**。

3. `dram__throughput.avg.pct_of_peak_sustained_elapsed    89.41%`
含义：DRAM 吞吐率相对于 NCU 所采用的持续峰值参考值的比例。
  - 数值越高，意味着这一测量时段内 DRAM 吞吐率越接近对应的持续峰值参考值。
  - 数值较低，并不能单独证明显存访问效率差。
  - 数值较高，也不一定能证明访存模式高效，因为大量冗余传输同样可能产生很高的流量。

4. `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed    89.41%`
含义：GPU Speed Of Light 中的整体计算 / 内存吞吐率指标。  
它是一个高层次的汇总指标，主要用于粗略判断计算单元或内存相关资源距离各自峰值有多远。这个吞吐率由一组底层指标共同构成，不是把所有硬件吞吐率简单相加。

5. `l1tex__throughput.avg.pct_of_peak_sustained_elapsed    36.47%`
含义：L1 / Texture 相关硬件单元的吞吐率相对于持续峰值的比例。

6. `lts__throughput.avg.pct_of_peak_sustained_elapsed    28.86%`
含义：L2 Cache 相关硬件的吞吐率相对于持续峰值的比例。

7. `sm__throughput.avg.pct_of_peak_sustained_elapsed`
含义：SM 相关计算吞吐率相对于持续峰值的比例。

## 全局内存访问
1. `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`
含义：全局内存读取操作在 L1/TEX 请求处理路径上产生的请求总数。
>注意：**一个 Request 不是一个线程的一次读操作**。 对全局内存而言，通常是一个 Warp 执行一次内存指令，向 L1/TEX 发出一个请求；后续会根据线程地址产生相应的内存分区访问。

2. `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum`
含义：全局内存写入操作在 L1/TEX 请求处理路径上产生的请求总数。

3. `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
含义：全局内存读取请求涉及的 L1/TEX Sectors 总数。

4. `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`

## 共享内存：Wavefront & Bank Conflict
1. `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
含义：共享内存读取操作发生的 Bank Conflict 计数。

2. `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`
含义：共享内存写入操作对应的 Bank Conflict 计数。

3. `1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum`
含义：为完成共享内存读取操作，L1/TEX 数据处理通路需要处理的 Wavefront 总数。

## Launch 与资源限制
**Occupancy** 是 SM 上活动 Warp 数量相对于最大可容纳 Warp 数量的比例。 更高的 Occupancy 不保证更快，但较低的 Occupancy 可能减少隐藏访存延迟的能力。

1. `launch__occupancy_limit_blocks    24 block`
含义：只考虑 SM 所能同时管理的 Block 数量限制时，理论上最多可以驻留多少个 Block。  
它反映的是硬件的 Block 槽位限制，不代表实际运行时一定有 24 个 Block 同时驻留。

2. `launch__occupancy_limit_registers    8 block`
含义：考虑寄存器资源限制时，理论上最多可以同时驻留多少个该配置的 Block。  
每个线程需要寄存器，而每个 Block 又包含许多线程。寄存器分配数量越大，每个 Block 消耗的寄存器资源通常越多，因此能够同时驻留的 Block 数可能受到限制。

3. `launch__occupancy_limit_shared_mem    12 block`
含义：考虑每个 Block 使用的共享内存大小时，理论上最多可同时驻留多少个 Block。
如果共享内存是限制因素，减少每个 Block 的共享内存需求可能增加驻留数量。

4. `launch__occupancy_limit_warps    6 block`
含义：考虑每个 Block 的线程 / Warp 数量限制时，理论上最多可同时驻留多少个 Block。

5. `launch__registers_per_thread`
含义：每个线程需要的寄存器数量。  
寄存器用于保存线程的局部状态、临时变量、中间结果等。这个数字会影响寄存器资源占用，从而影响每个 SM 能够同时驻留多少线程或 Block。

6. `launch__shared_mem_config_size`
含义：该 Kernel 启动所对应的共享内存配置大小。**它不是每个 Block 实际申请的共享内存大小。**

7. `launch__shared_mem_per_block_allocated`
含义：每个 Block 实际分配的共享内存量。
硬件资源按一定粒度分配，因此实际报告值不一定等于源代码中各个共享内存数组的字节数直接相加

8. `sm__warps_active.avg.pct_of_peak_sustained_active`
含义：活动 Warp 数量相对于最大可容纳 Warp 数量的比例，即常说的实际或 achieved Occupancy。  
它关注的是：有多少 Warp 处于活动状态。但一个活动中的 Warp 不一定能立刻执行下一条指令。例如，它可能正在等待全局内存加载结果，也可能正在等待同一个 Block 中其他 Warp 到达同步点。因此，Occupancy 高不等于指令发射利用率高。要知道活动 Warp 中有多少真正准备好执行指令，还需要 Scheduler 指标。

## Warp 调度：Eligible、Issue Active 与指令数

Warp 的基本状态：
- Active Warp：处于活动状态
- Eigible Warp：已经准备好可以进行下一条指令
- Issued Warp：半周期实际被调度器选中并发射指令

1. `smsp__warps_eligible.avg.per_cycle_active`
含义：每个 SM 子分区的 Warp 调度器，在其活动周期中，平均有多少个 Warp 已经满足发射条件，可以执行下一条指令。

2. `smsp__issue_active.avg.pct_of_peak_sustained_elapsed`
含义：Warp 调度器的指令发射活动，相对于持续峰值发射能力的比例，并按整个测量时段计算。  
它关心的是：**调度器有多频繁地实际发射指令**。

3. `smsp__inst_executed.sum`
含义：所有相关 SM 子分区累计执行的 Warp 级指令数。  
一般等于计算源代码语句产生的指令数，一条指令如果由多个 Warp 分别执行，Warp 级执行计数会对每个执行该指令的 Warp 进行统计

## Stall
1. `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`
含义：与 CTA 同步屏障有关的 Warp 等待情况。即有多少 Warp 的指令发射受到 Block 内同步条件的影响。

2. `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`
含义：Warp 正在等待由 L1/TEX 相关内存操作引起的依赖结果。

3. `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio`
含义：Warp 正在等待某类较短延迟的内存输入 / 输出依赖完成。  
Short Scoreboard 经常与共享内存访问有关，但不能简单地将它等同于 Shared Memory Stall。其他 MIO（Memory Input/Output）相关操作也可能产生这类依赖。