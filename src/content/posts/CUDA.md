---
title: CUDA 学习笔记
published: 2026-07-27
pinned: true
description: 从 GPU 架构开始，依次记录了一些 CUDA 常见语法与一些常用工程技巧，属于一份偏个人向的笔记
tags: [CUDA]
category: 技术
draft: false
---

**参考资料：**
- [CUDA C++ Programming Guide 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)
- [CUDA C++ Programming Guide 12.8 · Contents](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/contents.html)
- [CUDA C++ Best Practices Guide 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

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

---

# 1. 编程模型

## 1.1 Kernels
### 1.1.1 `__global__` 函数修饰符

`__global__` 是一个**函数执行空间修饰符（Execution Space Specifier），放在函数签名前**。它告诉编译器**这个函数由 CPU 调用，但在 GPU 上由大量线程并行执行**。被它修饰的函数，我们称之为 **核函数（Kernel）**

- **必须返回`void`**，得到的数据通过指针（显存地址）传回 CPU
- 调用语法为 `<<<<blocksPerGrid, threadsPerBlock>>>` ，指定线程组织方式，类似于OpenGL 里的 `glDispatchCompute(num_groups_x, num_groups_y, num_groups_z)`
- **调用异步性**：CPU 执行到`kernel<<<...>>>();`时，会将任务扔给GPU，然后**马上返回**继续执行下一行 CPU 代码，不会等待GPU算完

## 1.2 线程层级 Kernel Hierarchy

### 1.2.1 三层线程组织
- **Thread（线程）**：最基本的执行单元，每个线程都有自己独立的执行路径和私有寄存器。
- **Thread Block（线程块）**：一组线程的集合，块内线程可以协作，通过共享内存交换数据，并用 `__syncthreads()` 进行同步。
- **Grid（网格）**：一组线程块的集合。不同块之间无法直接通信或同步，块与块是独立执行的。

> [!IMPORtant]
> **Warp 是CUDA执行代码的最小硬件调度单位， 是 32 个 连续线程的集合**。SM 里的调度器每次取一个 Warp（32 条线程）的指令，发给 CUDA 核心去执行。如果这 32 个线程执行的是同一个指令（没有分支），它们可以同时在一个时钟周期内完成。假如你启动了一个 Block，里面有 256 个线程，那么这个 Block 会被 SM 拆分为 8 个 Warp，它们**轮流占用** SM 里的 CUDA 核心进行计算（这里的轮流就很有意思，当第 1 个 Warp 算完并卡在等待下一个数据时（**访存延迟**），SM 立刻切换第 2 个 Warp 上核心计算。这便是**GPU用计算掩盖延迟**）

### 1.2.2 四个关键内置变量
CUDA 在核函数内部自动提供了三个特殊变量，不需要声明：

```cpp
// A / B：指向显存一个数组（int）的指针
// n：需要处理的数据总数
__global__ void writeGlobalIndex(int* A, int* B, int* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 过滤多余线程，防止他们越界访问内存
    if (idx >= n) return;

    C[i] = A[i] + B[i];
}
```

三个特殊变量：
- **`threadIdx`**：当前线程在所属 **block 内**的局部索引（从 0 开始），这些 `Idx` 只是**逻辑地址**，并非硬件物理编号。
- **`blockIdx`**：当前 block 在整个网格（grid）中的索引（从 0 开始）
- **`blockDim`**：每个 block 里有多少个线程
- **`girdDim`**：网格内的块数

详细谈谈上面的例子：
这里传入了三个指针 `A B C`，`A B` 指向显存里的两个有数据的连续地址，`C` 指向待存放计算结果的地址，N 就是需要处理的数据总数。
`idx` 由 `int idx = blockIdx.x * blockDim.x + threadIdx.x;` 计算得出，如果这里有不止一个 block，每个 block 有 256 个线程，在 block 0 中，threadIdx.x 范围是从 0 到 255，在 block 1 中，threadIdx.x 范围也是从 0 到 255 ……，但是经过上面的计算，每个参与计算的线程都会有自己独一无二的线程 ID，随后每个线程根据自己的 `threadIdx` 到显存里取值，你可以将 `A[0]` 理解为 0 号线程取 A 指向内存的第1个元素，这里的 `idx` 就是数组索引。

全局线程索引的通用公式：（假设一维网格，一维块）

```cpp
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
```

这三个关键内置变量都是 dim3 类型（一个包含 x, y, z 的结构体），可以支持 1D、2D、3D 的组织方式。因为我们这个例子只用了一维，所以只取 `.x` 分量。

### 1.2.3 块内协作与限制
块内协作的方式：
- 通过 **Shared Memory** 共享数据
- 通过 `__syncthreads()` 设置同步点，块内所有线程必须在此等待，然后才能继续

> [!note]
> 一个线程块对应一个 SM，一个 SM 上可以跑多个线程块，一个 Kernel 可以被多个线层块执行
>

**线程数上限**：当前 GPU 上，一个线程块最多包含 1024 个线程

### 1.2.4 线程块集群 Thread Block Clusters
Thread Block Cluster 是**一组 Thread Block 的集合**，它们被硬件保证**同时调度在同一个 GPC（Graphics Processing Cluster）上**

核心机制：**Distributed Shared Memory（DSM**，分布式共享内存），Cluster 内每个 Block 的 shared memory，可以被 Cluster 内其他 Block 的线程直接访问，不再绕道 Global Memory

> 这是 Hopper（CC 9.0+）才有的能力

启动方式分为两种：
- 编译时确定： `__cluster_dims__(2, 1, 1)`
- 运行时确定：

## 1.3 硬件多线程 Hardware Multithreading
与 CPU 依靠指令级并行和分支预测来隐藏延迟不同，GPU 主要依赖线程级并行来最大化功能单元的**占用率**。当某个 Warp 因等待数据而停顿时，Warp 调度器可以立即切换到另一个就绪的 Warp 执行，从而**用计算隐藏延迟**。

---

# 2. 编程接口

## 2.1 nvcc 编译器
核函数可以用 **CUDA 指令集架构 PTX** 书写。具体的编译流程是：你的 CUDA / C++ 源码被 nvcc 编译成 PTX，再由 ptxas 编译成 cubin，即对应架构的 GPU 机器码 SASS

## 2.2 CUDA runtime
绝大多数 CUDA 程序用的都是 **CUDA Runtime API**，它**建立在 Driver API 之上**，提供隐式初始化、自动上下文管理、设备内存管理、流与事件、错误检查等功能，核心是**让你用更简洁的 `cuda` 前缀函数完成 GPU 编程，而不需要手动管理底层上下文和模块**。

> [!note]
> - **CUDA Runtime（运行时）**：可以看做是一个高级的、自动化的管家，核心任务就是让你用更少的代码来更轻松地指挥 GPU
> - **CUDA Context（上下文）**：可以理解为 GPU 上的独立工作区或进程，所有与 GPU 相关的资源，比如显存分配、加载模块、创建流等都归属于某个 Context。一个 GPU 可能被多个程序同时使用，Context 就像一道隔离墙，保证每个程序的资源互不干扰。当一个 Context 被销毁时，系统会自动清理它占用的所有资源
>
> **两者关系：CUDA Runtime 会在后台，自动为每个设备管理一个“主上下文（Primary Context）”**，在一个程序里，所有使用 Runtime API 的 Host 线程，默认都共享这同一个主上下文

### 2.2.1 初始化
**在 CUDA 12.0 之后**：

- **隐式初始化（默认行为）**：Runtime 会在你第一次调用需要它的函数时，自动为设备 0（默认 GPU）完成初始化和主上下文的创建
- **显式初始化（推荐）**：为了更好的控制，你可以主动调用 `cudaSetDevice()` 或 `cudaInitDevice()` 来指定设备并触发初始化

### 2.2.2 Device Memory
GPU 的计算流程通常是：  

> **在主机分配内存 → 拷贝数据到设备 → 在设备上执行 kernel → 把结果拷回主机**

设备内存通常可以分配成两种：**线性内存** 和 **CUDA 数组**。
- **CUDA 数组（CUDA arrays）**：是不透明（opaque）的，专门为纹理拾取（Texture fetching）优化。你不能直接用普通指针去操作，只能用特定 API 去读。
- **线性内存（Linear Memory）**：是在一个统一的地址空间里分配的，类似于 C 的线性内存

三个操作线性内存的**关键 API**：
- `cudaMalloc`：`cudaError_t cudaMalloc(void** devPtr, size_t size);`
    - `devPtr`：指向设备指针的指针
    - `size`：要分配的字节数
- `cudaFree`：`cudaError_t cudaFree(void* devPtr);`
- `cudaMemcpy`：`cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);`
  - `dst`：目标地址
  - `src`：源地址
  - `count`：拷贝的字节数
  - `kind`：拷贝方向

**合并访存：同一个 Warp 里的 32 个线程，请求的地址是否落在同一个 128 字节的内存块（Cache Line / 内存事务）里**

假设现在有一个二维数组：宽度为 10 个 float （40字节），在显存里它们是连续存储的，第 0 行，0 ~ 39 字节，第 1 行，40 ~ 79 字节。如果现在我们启动一个 Warp 去访问这块显存，**硬件读取内容是按照 128 字节来整体读取一块连续内存**（这里我假设一次读取 128 字节），所以当线程要读地址 40 ~ 79 时，这些地址全部落在第 1 个 128 字节块（0~127）里。这没问题，一次内存事务就搞定了（虽然只用了其中 40 个字节，但至少没跨块），但如果我们去访问第 4 行，120 ~ 127 这部分，在第 1 个 128 字节块里（0~127）；128 ~ 159 这部分，在第 2 个 128 字节块里（128~255）。于是一个 Warp 里的 32 个线程，虽然只连续地访问了 40 个字节，却横跨了两个 128 字节的内存块，硬件不得不发起 **2 次内存事务** 去搬这 40 个字，这就是所谓的**没对齐导致的非合并访存**。

对于矩阵、图像等多维数据，CUDA 提供了 `cudaMallocPitch` 和 `cudaMalloc3D` 来解决这类数据容易导致的非合并访存问题。

- `cudaMallocPitch`：`cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);`

`cudaMallocPitch` 会在每一行的末尾加上 **Padding（填充字节）**，把每行的**实际字节数（Pitch）** 补齐到 128 的倍数，比如 128 字节。于是第 0 行：地址 0 ~ 39（有效数据），40 ~ 127（空白填充）。**这样一来，无论访问哪一行的开头，它永远从 128 字节的整数倍地址开始。**

- `cudaMemcpy2D`：`cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);`
  
这里 `dpitch` 和 `spitch` 分别是目标、源的每行跨度

```cpp
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) 
    {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) 
        {
            float element = row[c];
        }
    }
}
```

---

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