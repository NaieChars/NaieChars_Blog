---
title: CUDA 学习笔记
published: 2026-07-27
pinned: true
description: 从 GPU 架构开始，依次记录了一些 CUDA 常见语法与一些常用工程技巧，属于一份偏个人向的笔记
tags: [CUDA]
category: 技术
draft: false
---


# 前置知识
这一部分**很重要**，能够帮助理解为什么一些常用的GPU工程技巧要这么设计，这里从**硬件架构**和**内存层次结构**两方面展开，做一次系统性总结

> [!NOTE]
> 关于 GPU 的一些微架构层面的性能分析，比如：Cache Hit Rate、L1/L2 Cache、Memory Throughput、Stall LG Throttle、Scoreboard等，会留到后面的一篇关于stall类型层级下的tiling性能对比文章中详细说明。如果只是浅浅应付 CUDA 编程，不去关心性能分析与优化，下面这些GPU基础知识就应该足够了。

## GPU 硬件架构

### 1.核心计算单元：流多处理器 (SM)
SM 是**GPU最基本的计算单元**，内部包含数十到上百个**CUDA核心**（负责执行整数和浮点运算）、**张量核心**（Tensor Cores，NeRF里会用到）等    
以笔者的显卡为例（RTX 5060 Laptop GPU），包含SM 26组，CUDA核心3328个，张量核心104个，光线追踪核心26个，纹理单元104个，光栅单元48个

### 2.执行模型：SIMT (单指令多线程)
- SIMT 执行模型即：同一个 Warp 里的 32 个线程执行同一个命令，但各自处理自己的数据
- **Warp（线程束）** 是 SM 创建、管理、调度和执行线程的**基本单位**。一个 Warp 包含 **32个** 线程，
- **硬件调度**：当一个 SM 接收到一个或多个线程块时，它将这些块划分为 Warp，每个 Warp 由 **Warp 调度器（Warp Scheduler）** 来调度执行。每个 Warp 划分方式确定，**包含连续递增线程 ID 的线程**，第一个 Warp 包含线程 0
- **线程束发散**：如果 Warp 内32个线程的执行路径不同（如有`if-else`分支），会导致**线程束发散（Warp Divergence）**。此时硬件会**串行化**这些分支路径，它会先执行一条路径，同时禁用不走这条路径的线程，导致有的线程存在闲置的真空期，性能下降

#### 2.1 独立线程调度（Volta 架构起）
**线程之间完全并发**，GPU 为每个线程维护执行状态，每个线程有自己的 PC，调用栈。线程可以在**子 Warp 粒度**上发散和重新聚合。调度优化器决定如何将活跃线程重新组合成 SIMT 单元，从而在保持 SIMT 高吞吐量的同时获得更大的灵活性

对于给定的 kernel，能在流多处理器上同时驻留并处理的 block 和 warp 数量，**取决于该 kernel 使用的寄存器和共享内存量**，以及流多处理器上可用的寄存器和共享内存量。此外，每个流多处理器对驻留的 block 数量和驻留的 warp 数量也都有上限。

> 如果每个多处理器上没有足够的寄存器或共享内存来处理**至少一个 block**，那么 **kernel 将启动失败。**


## 内存层级 Memory Hierarchy
CUDA 将内存分为多个层次，其核心思想是：**离计算单元越近的内存，速度越快，但容量越小**，下面这些都是很重要，也很好理解：  
- **寄存器 (Registers)**: 速度最快，位于SM内，每个线程私有。数量极其有限，是高性能的关键。
- **共享内存 (Shared Memory):** 位于SM内，速度仅次于寄存器，保证同一个线程块（block）内的**线程可以共享数据**
- **本地内存 (Local Memory): 逻辑上私有，物理上在全局内存中**，当寄存器不够用时，编译器会将部分变量溢出到这里，应尽量避免。
- **常量内存与纹理内存 (Constant/Texture Memory)**: 位于显存中，可供所有线程访问（只读）
- **全局内存 (Global Memory): 容量最大、延迟最高**，所有数据必须先拷贝到这里，GPU才能访问。

# 1. 编程接口


# 2. 编程模型

## 1.1 Kernels
### __global__函数修饰符

`__global__` 是一个**函数执行空间修饰符（Execution Space Specifier），放在函数签名前**。它告诉编译器**这个函数由 CPU 调用，但在 GPU 上由大量线程并行执行**。被它修饰的函数，我们称之为 **核函数（Kernel）**

- **必须返回`void`**，得到的数据通过指针（显存地址）传回 CPU
- 调用语法为 `<<<M, T>>>` ，指定线程组织方式，类似于OpenGL 里的 `glDispatchCompute(num_groups_x, num_groups_y, num_groups_z)`
- **调用异步性**：CPU执行到`kernel<<<...>>>();`时，会将任务扔给GPU，然后**马上返回**继续执行下一行 CPU 代码，不会等待GPU算完

## 1.2 线程层级 Kernel Hierarchy

### 三层线程组织
- **Thread（线程）**：最基本的执行单元，每个线程都有自己独立的执行路径和私有寄存器。
- **Thread Block（线程块）**：一组线程的集合，块内线程可以协作，通过共享内存交换数据，并用 `__syncthreads()` 进行同步。
- **Grid（网格）**：一组线程块的集合。不同块之间无法直接通信或同步，块与块是独立执行的。

> [!IMPORtant]
> **Warp 是CUDA执行代码的最小硬件调度单位， 是 32 个 连续线程的集合**。SM 里的调度器每次取一个 Warp（32 条线程）的指令，发给 CUDA 核心去执行。如果这 32 个线程执行的是同一个指令（没有分支），它们可以同时在一个时钟周期内完成。假如你启动了一个 Block，里面有 256 个线程，那么这个 Block 会被 SM 拆分为 8 个 Warp，它们**轮流占用** SM 里的 CUDA 核心进行计算（这里的轮流就很有意思，当第 1 个 Warp 算完并卡在等待下一个数据时（**访存延迟**），SM 立刻切换第 2 个 Warp 上核心计算。这便是**GPU用计算掩盖延迟**）

### 四个关键内置变量
CUDA 在核函数内部自动提供了三个特殊变量，不需要声明：

```cpp
// d_out：指向显存一个数组（int）的指针
// n：需要处理的数据总数
__global__ void writeGlobalIndex(int* d_out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 过滤多余线程
    if (idx >= n) return;
    d_out[idx] = idx;
}
```


三个特殊变量：
- **`threadIdx`**：当前线程在所属 **block 内**的局部索引（从 0 开始），这些 `Idx` 只是**逻辑地址**，并非硬件物理编号。
- **`blockIdx`**：当前 block 在整个网格（grid）中的索引（从 0 开始）
- **`blockDim`**：每个 block 里有多少个线程
- **`girdDim`**：网格内的块数

全局线程索引的通用公式：（一维网格，一维块）

```cpp
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
```

它们都是 dim3 类型（一个包含 x, y, z 的结构体），可以支持 1D、2D、3D 的组织方式。因为我们这个例子只用了一维，所以只取 `.x` 分量。

### 块内协作与限制
块内协作的方式：
- 通过 **Shared Memory** 共享数据
- 通过 `__syncthreads()` 设置同步点，块内所有线程必须在此等待，然后才能继续

> [!note]
> 一个线程块对应一个 SM，一个 SM 上可以跑多个线程块，一个 Kernel 可以被多个线层块执行
>

**线程数上限**：当前 GPU 上，一个线程块最多包含 1024 个线程

### 线程块集群 Thread Block Clusters
Thread Block Cluster 是**一组 Thread Block 的集合**，它们被硬件保证**同时调度在同一个 GPC（Graphics Processing Cluster）上**

核心机制：**Distributed Shared Memory（DSM**，分布式共享内存），Cluster 内每个 Block 的 shared memory，可以被 Cluster 内其他 Block 的线程直接访问，不再绕道 Global Memory

> 这是 Hopper（CC 9.0+）才有的能力

启动方式分为两种：
- 编译时确定： `__cluster_dims__(2, 1, 1)`
- 运行时确定：

## 1.3 硬件多线程 Hardware Multithreading
与 CPU 依靠指令级并行和分支预测来隐藏延迟不同，GPU 主要依赖线程级并行来最大化功能单元的**占用率**。当某个 Warp 因等待数据而停顿时，Warp 调度器可以立即切换到另一个就绪的 Warp 执行，从而**用计算隐藏延迟**。


#### 显存分配、初始化数据、核函数调用示例
我们通常进行如下的显存分配、初始化与核函数调用：

```cpp
const int N = 1000;
int* d_out = nullptr;
cudaMalloc(&d_out, N * sizeof(int));
cudaMemset(d_out, 0xFF, N * sizeof(int));

int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;

writeGlobalIndex<<<gridSize, blockSize>>>(d_out, N);
```
> cudaMalloc：在GPU 显存上分配一块内存，大小是 N * sizeof(int) 字节。

> cudaMemset：将刚分配的显存全部用字节 0xFF 填充。每个 int 占 4 字节，全部填 0xFF 后，4 字节合并就变成了 0xFFFFFFFF，作为有符号整数解读就是 -1。

> (N + blockSize - 1) / blockSize：向上取整计算 block 数量。

### cudaMemcpy
`cudaMemcpy()` 函数是 CUDA 里最基础的**数据传输函数**

```cpp
cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
```
参数说明：`目标地址`、`源地址`、`字节数`、`传输方向`

### cudaMalloc 与 cudaFree
等效于C中的 `malloc` 和 `free`，只不过操作对象变成了显存。养成显式释放的良好习惯

### __shared__变量修饰符
表示这个变量**存放在共享内存里**，而不是存在显存里，使得同一个block里面的线程都可以访问。典型应用：粒子模拟中的 tiling 思想，线程 `threadIdx.x` 负责把第 tile 块里第 `threadIdx.x` 个粒子的数据，从 global memory 搬到 shared memory 数组里。
> [!warning]
> 这里强调一下tiling思想的一个**坑**（具体的物理模拟粒子系统理论可以链接到我的另一篇文章）
> 在核函数里一般都会有 `if (i > n) return;` 这句，目的是过滤多余线程。但是一旦我们用了tiling思想（或者是需要线程同步执行的情况下`__syncthreads()`），就不能直接过滤掉多余线程。因为如果让多余线程提前退出，这些线程根本不会走到后面的 `__syncthreads()`，而其他线程会在那里死等——这会导致**kernel 挂起（deadlock）**
>
> 其实还有一个**坑**。2010年过后Fermi架构使得：当一个 warp 内所有线程请求同一个 global memory 地址时，内存控制器会把这次请求合并成一次物理内存事务，取回数据后**广播（broadcast）** 给这32个线程的寄存器，而不是发起32次独立的内存请求，实际下来tiling能捡到的优化油水非常少，尤其是对于更偏"计算密集"的kernel。（详情见粒子相关的文章）

### __device__函数修饰符

## 一些常用工程技巧

### 实用工具宏
这是一个很实用的错误抛出工具宏，很多项目都有它的身影，把以下内容放在文件开头

```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
```
之后每次关于cuda函数的调用，都可以这样表示：

```cpp
CUDA_CHECK(cudaMalloc(...));
CUDA_CHECK(cudaMemcpy(...));
```

### 将数据依次性打包进寄存器
在核函数里，通常现将要传给 GPU 的数据**赋值给一个局部变量**（CUDA里，简单的局部变量（非数组、不被取地址）通常会被编译器放进寄存器），请看下面两种核函数代码:

```cpp
if (idx >= n) return;

    paticles[idx].velY += gravity * dt;
    paticles[idx].posX += paticles[idx].velX * dt;
    paticles[idx].posY += paticles[idx].velY * dt;
```

每一行 particles[idx] 都代表一次对显存（global memory）的读写。全局内存的延迟非常高（几百个时钟周期）。

```cpp
if (idx >= n) return;
    // 先读到寄存器里改，最后一次性写回，减少global memory访问次数
    Particle p = particles[idx];

    p.velY += gravity * dt;
    p.posX += p.velX * dt;
    p.posY += p.velY * dt;
```

我们将 Particles[]赋值给了一个局部变量p，其存储在寄存器里，几乎零延迟。我们**先将整个结构从全局内存一次性打包到寄存器**，寄存器里发生所有的**中间计算**，最后再把**修改结果一次性写回全局内存**。

### 竞态问题
如果一个Kernel的**计算基于旧状态**，就不能边读边写同一个数据区域，应该分成两个kernel来完成。  
我们可以举一个引力模拟的例子：  
每个粒子的加速度，取决于其他所有粒子当前的位置。如果在同一个 kernel 里，一边让线程 A 更新粒子 0 的位置，一边让线程 B 计算粒子 1 的受力（需要读取粒子 0 的位置），那么根据线程调度顺序不同，线程 B 可能读到的是线程 A 已经写入的新位置，这便是**竞态问题**，解决办法：把计算和更新拆成两个kernel


# CUDA + Cmake (+ OpenCV) 项目的工程经验

> [!note]
> 以下内容来源笔者实际踩坑，调BUG调得之销魂，这里简单记录一些有价值的踩坑实录，希望能帮到一部分也在这条路上的人
>

## `__device__ 函数名` 跨 `.cu` 文件无法解析

比如下面一个结构：

```
util.cu
    └── __device__ solveAX0()

TriangulationCUDA.cu
    └── triangulateKernel()
            └── solveAX0()
```

> 我在 `Triangulation.cu` 里有 `#inlcude "util.cuh"`，但编译器告诉我仍找不到 `solveAX0` 的函数实现
>

编译器抛出**异常**如下：

```
ptxas fatal
Unresolved extern function
```

本质原因是：**`#include .cuh` 只会让当前文件知道函数声明，不会复制函数实现**。`__device__` 属于GPU代码，跨 `.cu` 使用时，还需要 CUDA 的 **devie linking**，**解决方法**如下：

```cmake
set_target_properties(cuda_sfm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
```

## `LNK2019`：Host 函数找不到

这个错误的具体原因是：**不同目录的同名 `.cu` 文件**生成的中间文件冲突，导致链接失败。

拿这个例子主要是说明在**遇到链接错误时**该如何解决问题，常见的做法是在 `build/Debug`（这里具体看`.obj` 文件在哪里）里面找**有没有相关源文件的 `.obj` 文件**。更进一步，可以用:

```PowerShell
dumpbin /symbols FeatureMatch.obj
```

查找中间文件里的具体内容，看有没有具体的函数。

这里得出的经验：
> CUDA / CMake 项目里，不要让同一个 target 下不同目录存在同名源文件。

这种结构虽然源码层面合法，但是 MSBuild 的中间文件命名容易踩坑