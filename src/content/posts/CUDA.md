---
title: GPU 基本架构与 CUDA 编程
published: 2026-07-27
pinned: true
description: 从 GPU 架构开始，依次记录了一些 CUDA 常见语法与一些常用工程技巧，属于一份偏个人向的笔记
tags: [CUDA]
category: 技术
draft: false
---

> [!NOTE]
> 讲解一些GPU编程知识时，会涉及到一些OpenGL Compute Shader 的知识类比。  
> 建议读者学习 CUDA 时，尽量配套实现一个完整的工程项目，学习知识的同时不至于那么枯燥，同时还能了解许多 GPU 编程时的工程技巧。笔者学习 CUDA 时，顺带完成了一个物理模拟粒子系统与真实流体的项目。

## 前置知识
这一部分**很重要**，能够帮助理解为什么一些常用的GPU工程技巧要这么设计，这里从**硬件架构**和**内存层次结构**两方面展开，做一次系统性总结

> [!NOTE]
> 关于 GPU 的一些微架构比如：Cache Hit Rate、L1/L2 Cache、Memory Throughput、Stall LG Throttle、Scoreboard等，会留到后面的一篇关于stall类型层级下的tiling性能对比文章中详细说明。如果只是浅浅应付 CUDA 编程，下面这些GPU基础知识就应该足够了。

### GPU 硬件架构

#### 1.核心计算单元：流多处理器 (SM)
SM是**GPU最基本的计算单元**，内部包含数十到上百个**CUDA核心**（负责执行整数和浮点运算）、**张量核心**（Tensor Cores，NeRF里会用到）等    
以笔者的显卡为例（RTX 5060 Laptop GPU），包含SM 26组，CUDA核心3328个，张量核心104个，光线追踪核心26个，纹理单元104个，光栅单元48个

#### 2.执行模型：SIMT (单指令多线程)
- SIMT执行模型即：一个SM内的所有CUDA核心都听从同一个指令，但处理不同的数据。
- **Warp（线程束）** 是GPU调度和执行的基本单位。一个Warp包含 **32个** 线程，它们以锁步（lockstep）方式执行同一条指令。如果Warp内32个线程的执行路径不同（如有很多if-else分支），会导致**线程束发散（Warp Divergence）**，不同分支根据复杂度其执行速度不同，导致有的线程存在闲置的真空期，性能严重下降

### CUDA内存层次结构
CUDA将内存分为多个层次，其核心思想是：**离计算单元越近的内存，速度越快，但容量越小**，下面这些都是很重要，也很好理解：  
- **寄存器 (Registers)**: 速度最快，位于SM内，每个线程私有。数量极其有限，是高性能的关键。
- **共享内存 (Shared Memory): 可编程的高速缓存**，位于SM内，速度仅次于寄存器，保证同一个线程块（block）内的线程可以共享数据
- **本地内存 (Local Memory): 逻辑上私有，物理上在全局内存中**，当寄存器不够用时，编译器会将部分变量溢出到这里，应尽量避免。
- **常量内存与纹理内存 (Constant/Texture Memory)**: 位于显存中，可供所有线程访问（只读）
- **全局内存 (Global Memory): 容量最大、延迟最高**，所有数据必须先拷贝到这里，GPU才能访问。

> [!IMPORtant]
> **Block、Grid、Thread、Wrap的关系**  
> 硬件关系前面已经提及，GPU包含SM，SM包含CUDA核心。这里主要是区分软件关系:  
> 线程、线程块、网格均是软件逻辑，他们在你调用核函数 `kernel<<<block, thread>>>` 出现，也就是说你要在 CPU 端分配好执行这个核函数要多少 block，每个 block 多少 thread。**Thread** 是最底层的 执行个体，**Block** 是 Thread 的容器，**Grid** 是 Block 的容器。 
> 
> **硬件映射**： 
> - 一个 Block，只会被调度到 1 个 SM 上运行，绝对不会拆分到多个 SM。
> - 一个 SM 可以同时运行多个 Block。
> - CUDA 核心不是固定分配给某个线程的，而是分时复用的（一个核心可以管理多个活跃线程，**核心负责算，调度器负责等**）
> 
> **Wrap 是CUDA执行代码的最小硬件调度单位， 是 32 个 连续线程的集合**。SM 里的调度器每次取一个 Warp（32 条线程）的指令，发给 CUDA 核心去执行。如果这 32 个线程执行的是同一个指令（没有分支），它们可以同时在一个时钟周期内完成。假如你启动了一个 Block，里面有 256 个线程，那么这个 Block 会被 SM 拆分为 8 个 Warp，它们**轮流占用** SM 里的 CUDA 核心进行计算（这里的轮流就很有意思，当第 1 个 Warp 算完并卡在等待下一个数据时（**访存延迟**），SM 立刻切换第 2 个 Warp 上核心计算。这便是**GPU用计算掩盖延迟**）

## 一些CUDA基础知识
### nvcc编译驱动
nvcc 是 NVIDIA 官方的 CUDA 编译器驱动（NVIDIA CUDA Compiler Driver），最关键的核心工作机制是**分离编译（自动区分主机代码（CPU端的C++）与设备代码（GPU上运行的部分））**
- 预处理与分离：读取 `.cu` 文件，将 GPU 代码提取出来
- 编译 GPU 代码
- 将剩下的 CPU 代码直接转发给C++编译器
- 链接

### __global__函数修饰符
`__global__` 是一个**函数执行空间修饰符（Execution Space Specifier），放在函数签名前**。它告诉编译器**这个函数由 CPU 调用，但在 GPU 上由大量线程并行执行**。被它修饰的函数，我们称之为 **核函数（Kernel）**

- **必须返回`void`**，得到的数据通过指针（显存地址）传回 CPU
- 调用语法为 `<<<M, T>>>` ，指定线程组织方式，类似于OpenGL 里的 `glDispatchCompute(num_groups_x, num_groups_y, num_groups_z)`
- **调用异步性**：CPU执行到`kernel<<<...>>>();`时，会将任务扔给GPU，然后**马上返回**继续执行下一行 CPU 代码，不会等待GPU算完
- 不能包含静态全局变量，与上述GPU内存模型有关

#### 三个关键内置变量
CUDA 在核函数内部自动提供了三个特殊变量，不需要声明  
先来看一个典型核函数及其调用：

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
- threadIdx：当前线程在所属 block 内的局部索引（从 0 开始）
- blockIdx：当前 block 在整个网格（grid）中的索引（从 0 开始）
- blockDim：每个 block 里有多少个线程

它们都是 dim3 类型（一个包含 x, y, z 的结构体），可以支持 1D、2D、3D 的组织方式。因为我们这个例子只用了一维，所以只取 `.x` 分量。

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