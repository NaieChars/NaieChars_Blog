---
title: Naive vs Shared Memory Tiling：针对CUDA经典优化案例的现代硬件重新验证
published: 2026-07-29
pinned: true
tags: [CUDA, N体模拟, GPU微架构]
category: 技术
draft: false
---


## 前言

Shared memory tiling 优化 N 体引力模拟，是 CUDA 教学里最经典的案例之一，出自 2007 年的《GPU Gems 3》第 31 章《Fast N-Body Simulation with CUDA》。书里写到一句"tiled 版本比 naive 版本快数倍甚至十几倍"。

这篇文章记录的是我在一块 **RTX 5060 Laptop GPU（Blackwell 架构，Compute Capability 12.0，sm_120）** 上重新做这组对比实验时发生的事——我的输出结果和教科书结论有明显出入，naive 版本反而比 tiling 版本还快。经过各方面的排查，最后用 Nsight Compute 的硬件计数器把整条因果链条 verify 了一遍。如果你也在现代 GPU 上跑这个经典案例，并且发现效果没有教程里说的那么夸张，这篇文章可能有用。

---

## 实验背景

### 物理模型

N 体引力模拟：每个粒子受到其余所有粒子的万有引力影响，是标准的 all-pairs（全对全）O(n²) 计算：

$$
a_i = \sum_{j \neq i} \frac{G \cdot m_j \cdot (\text{pos}_j - \text{pos}_i)}{\left( |\text{pos}_j - \text{pos}_i|^2 + \epsilon^2 \right)^{1.5}}
$$

其中 `ε`（softening，软化因子）用来防止两个粒子距离趋近于 0 时分母趋近于 0、加速度炸成无穷大。

积分用半隐式（辛）Euler，比完全显式 Euler 数值稳定性更好，几乎零额外成本：

```cpp
v_new = v_old + a * dt
x_new = x_old + v_new * dt   // 注意用的是刚更新出来的新速度，不是旧速度
```

### 初始条件
构造一个**有解析解**的场景（而不是直接随机）：给一个大质量中心天体，其余粒子按照圆周轨道所需的切向速度初始化：

```
v = sqrt(G * M_center / r)
```

如果力算对了，粒子应该绕中心稳定转圈；如果错了，粒子会很快螺旋坠入中心或直接飞散。于是这里便踩了第一个坑。

**踩坑记录**：最初，粒子全部缓慢坍塌向中心（反而还挺好看）。后来顿悟原来是环绕粒子自身的质量设得太大（起初每个粒子质量我设置的 1.0，2000 个粒子加起来占了中心天体质量的近 40%），但是轨道初速度公式只按中心天体质量算，没算上其余粒子之间的相互吸引，于是粒子的初速度减慢，向心力太大导致逐渐坠落。我把环绕粒子质量调小便可忽略其余粒子影响，进而恢复稳定圆周运动。

### Kernel 结构设计

N 体模拟里，粒子 i 的加速度依赖**其他所有粒子当前帧的位置**。。如果在同一个 kernel 里，一边让线程 A 更新粒子 0 的位置，一边让线程 B 计算粒子 1 的受力（需要读取粒子 0 的位置），那么根据线程调度顺序不同，线程 B 可能读到的是线程 A 已经写入的新位置，这便是**竞态问题**，解决办法：把计算和更新拆成两个kernel

所以固定用两阶段结构：

```
computeForces  kernel: 只读粒子位置，只写加速度数组
integrateEuler kernel: 读加速度数组 + 粒子当前状态，更新位置和速度
```

两个 kernel 之间天然有一个同步点（下一个 kernel 要等上一个 kernel 全部线程执行完才会开始），这解决了竞态问题。

---

## Naive 版本

每个线程负责一个粒子 i，暴力遍历全部 n 个粒子，累加对 i 的引力贡献：

```cpp
__global__ void computeForcesNaive(const Particle* particles, float* accX, float* accY,
                                    int n, float G, float softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float myPosX = particles[i].posX;
    float myPosY = particles[i].posY;
    float aX = 0.0f, aY = 0.0f;

    for (int j = 0; j < n; ++j) {
        if (j == i) continue;

        float dx = particles[j].posX - myPosX;
        float dy = particles[j].posY - myPosY;
        float distSqr = dx * dx + dy * dy + softening * softening;

        float invDist = rsqrtf(distSqr);   // 快速倒数平方根内建函数
        float invDist3 = invDist * invDist * invDist;

        aX += G * particles[j].mass * dx * invDist3;
        aY += G * particles[j].mass * dy * invDist3;
    }

    accX[i] = aX;
    accY[i] = aY;
}
```

这个版本朴素在于：每个线程都独立地把全部 n 个粒子的数据从 global memory 读了一遍，字面意义上的重复访存。

---

## Tiled 版本：Shared Memory 优化

Tiling 的思路：把 n 个粒子切成若干tile，大小等于 blockDim.x。block 内线程协作，把一个 tile 的数据先统一搬进 shared memory，再让全 block 复用这份数据算完这个 tile 的贡献，然后换下一个 tile。

```cpp
#define TILE_SIZE 256

struct TileData { float posX, posY, mass; };

__global__ void computeForcesTiled(const Particle* particles, float* accX, float* accY,
                                    int n, float G, float softening)
{
    __shared__ TileData tile[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 注意：这里不能提前 return！即使 i >= n，这个线程仍然要参与
    // 后面协作搬运数据到 shared memory 的工作，否则会导致本block内的
    // __syncthreads() 死锁——因为 __syncthreads() 要求同一 block 内
    // 所有线程都执行到这一行才能继续，提前退出的线程永远不会走到那一行，
    // 其余线程会永远卡在那里等
    float myPosX = (i < n) ? particles[i].posX : 0.0f;
    float myPosY = (i < n) ? particles[i].posY : 0.0f;

    float aX = 0.0f, aY = 0.0f;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int loadIdx = t * TILE_SIZE + threadIdx.x;

        // 协作加载：每个线程只搬1个粒子进shared memory，分摊带宽
        if (loadIdx < n) {
            tile[threadIdx.x] = { particles[loadIdx].posX,
                                   particles[loadIdx].posY,
                                   particles[loadIdx].mass };
        } else {
            tile[threadIdx.x] = { 0.0f, 0.0f, 0.0f }; // 质量为0，不产生引力贡献
        }

        __syncthreads(); // 同步点1：确保整个block都搬完数据，才能开始读

        for (int k = 0; k < TILE_SIZE; ++k) {
            int j = t * TILE_SIZE + k;
            if (j == i || j >= n) continue;

            float dx = tile[k].posX - myPosX;
            float dy = tile[k].posY - myPosY;
            float distSqr = dx * dx + dy * dy + softening * softening;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            aX += G * tile[k].mass * dx * invDist3;
            aY += G * tile[k].mass * dy * invDist3;
        }

        __syncthreads(); // 同步点2：确保整个block都用完数据，才能覆盖写下一tile
    }

    if (i < n) { accX[i] = aX; accY[i] = aY; }
}
```

**几个容易踩的坑**：
1. **同步点1漏掉**：会读到别的线程还没搬完的脏 shared memory 数据，结果错误且不稳定复现（取决于调度）。
2. **同步点2漏掉**：会有线程已经开始往 shared memory 写下一个 tile 的数据，但还有线程没读完当前 tile，数据被提前覆盖
3. **越界线程提前 return**：会导致同 block 内的 `__syncthreads()` 死锁，这个坑比数值错误更隐蔽，因为程序会直接卡死而不是给出错误结果。

---

## 第一轮实测结果：

用 `cudaEvent` 测总耗时，N=2000，Release 模式：

| 版本 | 总耗时 | 平均每帧 |
|---|---|---|
| naive | 156.26 ms | 0.3125 ms |
| tiled | 166.15 ms | 0.3323 ms |

tiled 反而**更慢**。把 N 放大到 20000，差距依然不明显；放大到 200000（Debug/Release都测过，数据量已经从 40KB 级别涨到 4MB 级别），才终于看到 **tiled 反超**：

| N | naive 平均每帧 | tiled 平均每帧 | tiled提升 |
|---|---|---|---|
| 200000 | 189.38 ms | 178.23 ms | 约 6% |

6% 的提升和教程里说的数倍加速差距很大。于是我用 Nsight Compute 把两版 kernel 的硬件行为实测出来，找真正的原因。

---

## 用 Nsight Compute 得到的两个可能原因

### Naive 版本的 profiling 结果

| 指标 | 数值 |
|---|---|
| Compute (SM) Throughput | 75.02% |
| Memory Throughput | 75.02% |
| L1/TEX Hit Rate | **99.75%** |
| L2 Hit Rate | 79.05% |
| Achieved Occupancy | 94.81% |
| Stall LG Throttle | **5.96%**（Warp State里最大的一项） |
| Stall Long Scoreboard | 2.55% |

### Tiled 版本的 profiling 结果

| 指标 | 数值 |
|---|---|
| Compute (SM) Throughput | **88.96%** |
| Memory Throughput | 79.54% |
| L1/TEX Hit Rate | 66.97% |
| L2 Hit Rate | 76.58% |
| Stall LG Throttle | **0%** |
| Stall Barrier | 0.25%（新出现） |

### 解释 1：Warp Broadcast 已经替 naive 版本做了大半优化

内层循环 `for (int j = 0; j < n; ++j)` 里，`j` 的值对**同一个 warp 内的全部32个线程来说完全相同**（SIMT 模型下同一 warp 的线程执行进度一致）。这意味着 warp 里32个线程在同一时刻请求的是**同一个** `particles[j]` 地址。

现代 GPU（Fermi架构之后）对这种情况有**硬件级优化**：同一 warp 内所有线程访问同一地址时，内存控制器把这次请求合并成**一次**物理事务，取回后**广播**给32个线程，而不是发起32次独立请求。

**99.75% 的 L1/TEX Hit Rate 直接证实了这一点**，也就是说几乎所有读取请求根本没有真正打到显存上。这也解释了 GPU Gems 3 原文能展现数倍加速的原因：2007 年的是 G80/Tesla 架构（compute capability 1.x），其没有 L1/L2 缓存与 warp broadcast 优化，naive 版本用老的显卡架构确实会把显存宽带跑爆。现在的硬件已经把重复读取同一地址这个问题自动优化了很大一部分，tiling 手动实现的提取数据到共享内存很大程度上是在优化一个已经被硬件优化过的问题。

### 解释 2：差异在于 LSU 发射槽位，而非显存带宽

如果 naive 版本 99.75% 都缓存命中，为什么 tiled 还能快 6%？这是因为 `Stall LG Throttle`。naive 版本里，即使数据早就在 L1 里，**内层循环每次迭代仍然要向 Load/Store Unit（LSU）发射一条 global load 指令**，由于发指令这个动作本身要占用 LSU 流水线的发射槽位。当这类指令发射得足够密集，LSU 本身会成为限制吞吐的一环，这在 naive 版本里贡献了 5.96% 的 stall。

Shared memory 有独立于 global load 的访问路径，不走 LSU 这条通道。Tiled 版本把内层循环改成读 shared memory 后，这部分 stall **完全消失（5.96% → 0%）**，代价是新增了 `__syncthreads()` 带来的 `Stall Barrier: 0.25%`。两者相减，净收益的量级恰好对应实测的 6% 提升。

同时可以看到 `Compute (SM) Throughput` 从 75.02% 涨到 88.96%，而 `Memory Throughput` 基本没变（75.02% → 79.54%）——这进一步印证了收益来源：**tiling 带来的提升不是数据传输变快了，而是给计算单元腾出了更连续的流水线空档**，减少了计算指令被 LSU 发射阻塞的次数。

Tiled 版本 L1/TEX Hit Rate 从 99.75% 掉到 66.97% 并不是坏事，因为协作加载阶段每个线程只搬一次自己负责的粒子，复用次数天然变少，但内层计算主体已经完全转移到不经过 L1 的 shared memory 通道上了。

---

## 结论

在现代 GPU（Ampere/Ada/Blackwell 这类有完善 L1/L2 缓存和 warp broadcast 机制的架构）上重新实测这个经典案例：

1. **N 比较小（几千到两万）时，naive 和 tiled 几乎没有差异**，甚至 tiled 可能因为额外的同步开销略慢，这时候数据集小到能被 L1/L2 缓存完全容纳，硬件已经把重复读取的问题解决了。
2. **N 足够大（数据量明显超过 L2 缓存容量）时，tiled 开始体现优势**，但幅度（约 6%）远小于教程里常说的"数倍提升"
3. Tiling 真正的收益机制变成在现代硬件上，**从节省显存带宽变成了节省 LSU 指令发射槽位、给计算腾出连续执行的空档"**，这是和 2007 年原始教程完全不同的性能模型
---

## 附：环境信息

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell architecture, Compute Capability 12.0 / sm_120)
- CUDA Toolkit: 13.3
- Nsight Compute: 2026.2.1
- 构建方式: CMake + MSVC, Release 模式（`CMAKE_CUDA_ARCHITECTURES 120`）
