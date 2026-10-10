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
CUDA 将内存分为多个层次，其核心思想是：**离计算单元越近的内存，速度越快，但容量越小**：

- **寄存器 (Registers)**: 速度最快，位于SM内，每个线程私有。数量极其有限，是高性能的关键。
- **共享内存 (Shared Memory):** 位于SM内，速度仅次于寄存器，保证**同一个线程块**（block）内的**线程可以共享数据**
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

> [!IMPORTANT]
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

> [!NOTE]
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


#### 合并访存
**合并访存：同一个 Warp 里的 32 个线程，请求的地址是否落在同一个 128 字节的内存块（Cache Line / 内存事务）里**

假设现在有一个二维数组：宽度为 10 个 float （40字节），在显存里它们是连续存储的，第 0 行，0 ~ 39 字节，第 1 行，40 ~ 79 字节。如果现在我们启动一个 Warp 去访问这块显存，**硬件读取内容是按照 128 字节来整体读取一块连续内存**（这里我假设一次读取 128 字节），所以当线程要读地址 40 ~ 79 时，这些地址全部落在第 1 个 128 字节块（0 ~ 127）里。这没问题，一次内存事务就搞定了（虽然只用了其中 40 个字节，但至少没跨块），但如果我们去访问第 4 行，120 ~ 127 这部分，在第 1 个 128 字节块里（0 ~ 127）；128 ~ 159 这部分，在第 2 个 128 字节块里（128~255）。于是一个 Warp 里的 32 个线程，虽然只连续地访问了 40 个字节，却横跨了两个 128 字节的内存块，硬件不得不发起 **2 次内存事务** 去搬这 40 个字，这就是所谓的**没对齐导致的非合并访存**。

对于矩阵、图像等多维数据，**CUDA 提供了 `cudaMallocPitch` 和 `cudaMalloc3D` 来解决这类数据容易导致的非合并访存问题**。

- `cudaMallocPitch`：`cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);`

`cudaMallocPitch` 会在每一行的末尾加上 **Padding（填充字节）**，把每行的**实际字节数（Pitch）** 补齐到 128 的倍数，比如 128 字节。于是第 0 行：地址 0 ~ 39（有效数据），40 ~ 127（空白填充）。**这样一来，无论访问哪一行的开头，它永远从 128 字节的整数倍地址开始。**

- `cudaMemcpy2D`：`cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);`
  
这里 `dpitch` 和 `spitch` 分别是目标、源的每行跨度

下面给一个代码例子：

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

    // 实际上上诉代码我们会借助 Idx，不会让每个线程跑几层 for 循环：
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Col
    Int r = blockIdx.y * blockDim.y + threadIdx.y; // Row

    if (c < width && r < height)
    {
        float* row = (float*)((char*)devPtr + r * pitch);
        row[c] = 100.0f;
    }
}
```

> 对于 `float* row = (float*)((char*)devPtr + r * pitch);`，其中 `devPtr` 的类型是 `float*`，那么 `devPtr + 1` 就意味着增加 4 个字节。然而 `pitch` 的大小是 1 字节，如果写成 `devPtr + r * pitch`，编译器会以为你要增加 `r * pitch * 4` 个字节，导致找错。所以将 `devPtr` 强制转换成 `*char`，这样 `(*char) + 1` 代表的就是增加一个字节。


那么**如何从主机（CPU）端访问和拷贝在设备（GPU）端声明的全局变量或常量变量**（即 `__device__` 和 `__constant__` 变量）？

通过在变量前加上 `__device__` 或 `__constant__` 可以将**变量声明在设备端**，这样他们会存在于 CUDA 上下文中，任何 Kernel 可以直接访问他们（即使没有通过函数参数传入他们）。但这存在一个问题，即 CPU 端并没有这些设备端变量的地址，于是 `cuadMemcpyToSymbol` 允许我们只传入这个变量的符号名，CUDA 底层去查表以找到他们在显存里真正的地址，随后进行拷贝等工作。

```cpp
__constant__ float constData[256];  // 设备端的常量内存
float data[256];                    // host 端指针
cudaMemcpyToSymbol(constData, data, sizeof(data));  // host -> device
cudaMemcpyFromSymbol(data, constData, sizeof(data));
```

> 上述示例中的 `constData` 并不是指向数组首元素的地址，也就是说这里并**没有发生退化**，**`constData` 只是一个符号名**

```cpp
__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

这个例子是一个很巧妙的设计：先在主机端创建一个指针 `ptr`，用 `cudaMalloc` 划分一块显存，并将这块显存的地址赋值给 `ptr`，随后再将 `ptr` 的值拷贝给 CUDA 全局变量 `devPointer`，这样所有的 Kernel 都可以通过 `devPointer` 访问这块显存

> 为什么要绕一圈 Host 端的指针，因为 `cudaMalloc` 是运行时分配显存，`__device__ float* devPointer` 此时并没有指向任何有效的动态显存，所以需要一个 Host 端的指针在运行时接住 `cudaMalloc` 在运行时返回的 GPU 显存地址。当然你也可以直接传参，在 kernel 内赋值也可以避免这个问题

### 2.2.3 Shared Memory
Shared Memory **通过 `__shared__` 声明**，Shared Memory 是 **thread block 级别**的内存，一个 block 里的所有线程都可以进行访问，速度远快于全局内存，生命周期与 block 相同。 

#### tiling 技术
**用 Shared Memory 做矩阵乘法分块**

如果一个矩阵太大，一个 block 的共享内存放不下，那么可以把这个矩阵拆成几个小块，每一个 block 负责一个小块的乘法，最后将各小块拼接起来得到乘法结果，看下面具体的计算例子：

$$
A_{4 \times 4} \times B_{4\times 4} = C_{4 \times 4} 
$$

设 `block_size = 2`，那么每一个 thread block 处理一个 2 x 2 的 C 子矩阵，我们将 C 从左至右，从上到下分成 4 个 2 x 2 子矩阵 $C_{00}, C_{01}, C_{10}, C_{11}$，同理 A 与 B 也这样分，那么有如下计算：

$$
C_{00} = A_{00} \times B_{00} + A_{01} \times B_{10} \\
C_{01} = A_{00} \times B_{01} + A_{01} \times B_{11} \\
C_{10} = A_{10} \times B_{00} + A_{11} \times B_{10} \\
C_{11} = A_{10} \times B_{01} + A_{11} \times B_{11}
$$
 
$A_{00}$ 同时被 block(0, 0) 和 block(0, 1) 读取，所以 $A_{00}$ 要从全局内存搬 2 次，也就是说：

> 每个 A 的子块从 Global Memory 被读取 `B.width / block_size` 次（即一个 A 子块需要服务多少个横向 C Block）；每个 B 的子块从 Global Memory 被读取 `A.height / block_size` 次

因此 Shared Memory 能**更加节省带宽**。

#### Bank Conflict
Shared Memory 通常**被划分为 32 个 bank**，每一个 bank **带宽**为 4 字节，可以近似理解为每个 bank 每个周期服务一个 32-bit word（4 字节）的访问，以此来大幅提高 Shared Memory 的带宽。**如果同一个 Warp** 的一次 Shared Memory 访问中，**多个线程访问了同一个 bank 的不同地址**，就需要拆分访问，也就是发生了所谓的 **Bank Conflict**

对于 `float data[i]`，可以用可以用如下公式计算 bank：

$$
\text{bank}_{id} = \text{idx} \% 32
$$

连续索引 0 ~ 31 会映射到 Bank 0 ~ 31；32 又映射到 Bank 0，33 映射到 Bank 1，以此类推

所以为了避免 bank conflict 我们需要着重关注访问的形式，如果是连续访问就没有冲突，如果访问是 `data[idx * 32]`，每个线程都会访问 bank 0，冲突严重。

#### braodcast
例如 `x = data[0]`，硬件发现每一个线程访问的都是 bank 0 中的同一个地址，那么硬件会采取 braodcast 行为，只取一次数据然后再分发给每一个线程。（只要有多个线程访问了同一个地址，硬件就可以自行对这部分线程做 braodcast）

### 2.2.4 L2 Cache

# 3. 基础性能优化
## 3.1 最大化指令吞吐量
**吞吐量**：每个 SM 在每一个时钟周期内执行的操作数

因为一个 warp 有 32 个线程，**一条指令对应 32 个操作**，所以：  
> **指令吞吐量 = 操作吞吐量 / 32**

> [!NOTE]
> 可以这样**理解指令吞吐量**：硬件每一个周期**最多能处理并发射**多少条这类指令。  
> 在 GPU 中，指令发射和指令执行是两个不同的概念。比如对于 FFMA，一个周期最多发射 2 条这类指令，经过 4 个周期的**延迟**才能得到计算结果。这里说的指令吞吐量是一个稳态下的速率，即稳态下，FP32 流水线每个周期能完成 2 条 warp FFMA，同时也在接收 2 条新的。

### 3.1.1 算术指令
不同算术指令的吞吐量差异巨大，选择正确的指令类型本身就是一种优化。

> 不同指令的吞吐量参考 CUDA 白皮书

#### 针对算术指令的优化
- 在精度要求不高的情况下，优先使用 **FP32**，这是硬件原生支持的，吞吐量最高
- 用**内建函数**替代标准库函数，他们的优点是速度快很多，代价是精度略低。常见内建函数 `__sinf()`，`__expf()`，`__logf()`等，常用的 `__fdividef(x, y)` 提供了比 `/` 更快的单精度除法
- 通过开启 nvcc 的相关编译选项来进行近似算法
- **用位运算替代整数除法/模数**：整数除法和取模运算开销很大，因为它们会编译成多达 20 条指令。如果 `n` 是 2 的幂，`(i / n)` 等价于 `(i >> log2(n))`，`(i % n)` 等价于 `(i & (n - 1))`；如果 `n` 是字面量，编译器会自动进行这些转换
- 对于常量，建议加上 `f` 显式表示为单精度，由于 C/C++ 标准，无后缀的常量是 double，对于 `float a = 3.14`，变量 `a` 是单精度，编译器会在此插入一条转换指令，于是便会出现额外开销。
- 半精度优化

#### 半精度计算

> [!NOTE]
> **半精度**的一些优化：
> 有两种半精度格式：
> - FP16（指数位5位，尾数位10位，具有较高精度，但由于只有5位指数导致范围很窄，最大只能到 65504，容易在训练时溢出，因此适合推理）
> - BF16（指数位8位，尾数位7位，具有与FP32相同的范围，但精度只有FP16的大约8/1，适合训练
>
> GPU 寄存器是32位，通常将两个半精度数一起放进一个32位寄存器里，让硬件的一条指令同时处理两个操作（SIMD)，这就要用到向量类型 `half2`（FP16） 和 `__nv_bfloat162`（BF16），以及对应的内建函数

半精度计算的具体代码实例：  

```cpp
#include <cuda_fp16.h>      // 用于 half 和 half2
#include <cuda_bf16.h>      // 用于 __nv_bfloat16 和 __nv_bfloat162

// 示例：使用 half2 进行 SIMD 运算
__global__ void half2_example(half2* a, half2* b, half2* c) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    half2 va = a[idx];
    half2 vb = b[idx];
    // __hfma2: 一条指令完成两个 FMA
    // SM 调度器发射一条 warp 指令，指令进入 FP16 ALU 后，
    // 硬件并排打开两个 16 位计算通道
    c[idx] = __hfma2(va, vb, va); 
}
```

### 3.1.2 控制流指令
控制流指令（`if, switch, do, for, while`）很容易导致一个 warp 里的不同线程走不同的路径，导致 **warp divergence**。当一个 warp 内发生线程发散时，硬件不得不串行化这些不同的路径，这使得 **warp 执行的指令总数增加**了，有效吞吐量下降

减少 warp 发散的**策略**：
- **让控制条件依赖 `threadIdx.x / 32`**：如果分支条件只取决于 warp 的 ID（即` threadIdx.x / 32`），那么同一个 warp 内的所有线程会走同一条路径，不会发散
- **编译器优化：分支预测（Predication）** ：仅针对于短小的分支
- **使用 `#pragma unroll` 循环展开**。（下面详细讲讲）

#### 循环展开 `#pragma unroll`
普通循环在每一个周期结束时都会判断循环条件是否满足，使用 `#pragma unroll` 将循环展开成指定次数，没有条件判断等操作了。具体提升性能的方面如下：
- 减少了指令发射的开销：如果不展开，循环体里的 `IADD`, `ISETP`, `BRA` 这些指令会占用宝贵的发射槽，展开后，省下的发射槽可以留给 `FFMA`, `LDG`等，指令吞吐量直接提升
- 增加指令级并行（**ILP**）：由于指令延迟的存在，如果每条 `FFMA` 指令都必须等待上一次计算的结果（一条依赖链），那么中间必然存在3个空闲周期。如果依赖链有4条（ABCD），轮流执行不同的指令，等到第4个周期开始（第三周执行完链D后），又可以接着执行链 A。这里体现的就是**用足够多的独立指令或活跃 Warp，隐藏单条指令的延迟。** 回到循环展开，在展开过后，暴露出非循环控制指令，比如地址变成常量偏移，多次迭代访存可以并发；编译器可以用多个累加器实现对依赖链的拆分，因此制造出 ILP

循环展开及实现 ILP 的代码示例：

```cpp
// 一般循环
for (int i = 0; i < 8; i++) 
{
    sum += a[i] * b[i];
}
// 编译器生成的大致指令序列为：
I1:  LDG  r1, [a + i*4]      // 读 a[i]
I2:  LDG  r2, [b + i*4]      // 读 b[i]
I3:  FFMA sum, r1, r2, sum   // sum = r1*r2 + sum
I4:  IADD i, i, 1            // i++
I5:  ISETP p, i, 8           // i < 8 ?
I6:  BRA  p, I1              // 跳回 I1


// 循环展开优化
// 手动将循环内的依赖链拆分成4条，使得 FFMA 吞吐量最大化
#pragma unroll
for (int i = 0; i < 8; i += 4)
{
  sum0 += a[i] * b[i];
  sum1 += a[i + 1] * b[i + 1];
  sum2 += a[i + 2] * b[i + 2];
  sum3 += a[i + 3] * b[i + 3];
}
// 编译器生成的大致指令序列
// ---------- 第一次迭代 (i = 0) ----------
LDG R1, [a+0]      // 读 a[0]
LDG R2, [b+0]      // 读 b[0]
LDG R3, [a+4]      // 读 a[1]
LDG R4, [b+4]      // 读 b[1]
LDG R5, [a+8]      // 读 a[2]
LDG R6, [b+8]      // 读 b[2]
LDG R7, [a+12]     // 读 a[3]
LDG R8, [b+12]     // 读 b[3]

FFMA sum0, R1, R2, sum0   // sum0 += a[0]*b[0]
FFMA sum1, R3, R4, sum1   // sum1 += a[1]*b[1]
FFMA sum2, R5, R6, sum2   // sum2 += a[2]*b[2]
FFMA sum3, R7, R8, sum3   // sum3 += a[3]*b[3]
```


> 在 CUDA 里，如果循环次数在编译期是已知的常量（比如 `#define N 4`），编译器默认会自动完全展开。只有当循环次数是动态的（比如 `n` 是变量），或者循环体太大时，才需要手动加 `#pragma unroll` 提示编译器

**展开的负面影响**：
- 寄存器压力爆炸：展开会让多个变量同时存活，一旦寄存器不够用，变量会溢出到本地内存，本地内存延迟极高，带宽极低，一旦溢出性能会大跌，反而得不偿失。
- 指令缓冲（I-Cache）未命中：GPU 指令缓冲有限。循环展开后，代码体积暴增，指令缓存装不下。SM 每次执行都要从 L2 甚至是显存里重新拉取指令，这导致严重的 Instruction Fetch Stall

### 3.1.3 同步指令
`__syncthreads()` 会占用 4 个时钟周期来为一个 warp 发出指令，因此尽量减少不必要的 `__syncthreads()` 调用，使用 `__restrict__` 指针可以帮助编译器优化掉一些不必要的同步

# 4. C++ 扩展上的 CUDA 语法
## 4.1 函数执行空间限定符
### 4.1.1 `__global__`
### 4.1.2 `__device__`
### 4.1.3 `__host__`
有连体标签：`__host__ __device__`。作用：让**同一份代码**，既能在 CPU 上跑，也能在 GPU 上跑。编译器会自动帮你生成两份机器码，以此提高代码复用率

### 4.1.4 Undefined Behavior
> [!WARNING]
不要跨界调用，在有 `__CUDA_ARCH__`（GPU代码）里调用只标明了 `__host__` 的代码是未定义的；同理在 CPU 代码里也一样。

### 4.1.5 内联微调
1. `__forceinline__`：**强制内联**。主要用于极其频繁调用、逻辑简单的小函数。如果**你确定**某个函数内联后能**大幅提升性能**，就用它
2. `__noinline__`：**拒绝内联**。主要用于**极其庞大**的函数，如果内联，代码体积暴增导致 I-Cache 暴跌，或寄存器溢出到本地内存，增加了访存开销
3. `__inline_hint__`：**建议内联**。

> [!WARNING]
> 1. **不能用在 `inline` 函数上**：CUDA 里的这些限定符和 C++ 原生的 `inline` 关键字不要混用。
> 2. **不能用在 `__global__` 核函数上**：这些只针对 `__device__` 函数，Kernel 是入口，它没法被谁内联。

## 4.2 空间限定符
### 4.2.1 不加任何标签
自动变量，如 `int a = 0`，通常存进寄存器里，如果寄存器不够会溢出到本地内存，性能暴跌

### 4.2.2 `__device__`
存进全局内存。  
机制：读取时走 L1/L2，如果 Warp 里的 32 个线程读**不同地址**，它支持得很好（**Coalescing合并访问**）。但如果有 32 个线程读**同一个地址**，全局内存会 **braodcast**，但延迟依然很高。

### 4.2.3 `__constant__`
存进常量内存，生命周期 = 程序。  
机制：背后有专用的**常量缓存（Constant Cache）**。当 Warp 里的 32 个线程**全部读取同一个地址**时，它会以极快的速度进行一次 **Broadcast**，完美命中。**但它的致命弱点在这里**：如果 Warp 里的 32 个线程读**不同地址**，就会发生 **“串行化（Serialization）”**，性能直接暴跌 32 倍。  
数据存放建议：存放**所有线程共享的、全程不变的参数**。比如：卷积核的权重、物理模拟里的重力加速度 `g`、矩阵的维度大小。

在 CPU 端对这类 GPU 变量进行赋值：
```cpp
__device__ int dev_var;
__constant__ int const_var;

// 主机端（CPU）操作：
cudaMemcpyToSymbol(const_var, &host_val, sizeof(int)); // 往常量内存里传值
cudaMemcpyFromSymbol(&host_val, dev_var, sizeof(int)); // 从全局内存里拿值
cudaGetSymbolAddress(&ptr, dev_var); // 获取它在 GPU 上的真实指针
```

>`cudaMemcpy` 传的是“地址”，`cudaMemcpyToSymbol` 传的是“变量名”。  
> 定义的时候**不能初始化**（`__constant__ int x = 5;` 在某些版本会报错）。必须先启动一次，在 CPU 端用 `cudaMemcpyToSymbol` 赋值，然后再启动 Kernel 使用。

### 4.2.4 `__shared__`
共享内存是每一个 block 私有的，没有全局唯一的固定地址。

#### 4.2.4.1 动态共享内存
允许在启动 kernel 之后，从外面传一个大小进去；与之相反的是静态共享内存，例如 `__shared__ float share[16][16]`，大小在编译期就写死了，不能修改。

动态共享内存使用示例：
```cpp
// 声明时，前面加 extern
__global__ void myKernel(float* data) 
{
    extern __shared__ float sdata[]; // 注意：这里没写大小！只是一个占位符
    // ...
}

// 传大小
int shared_mem_size = 1024 * sizeof(float); // 比如我要 4KB
myKernel<<<grid, block, shared_mem_size>>>(data); //  <<< >>> 中的第三个参数就是我们动态指定的共享内存大小
```

如果想搞三个数组：`short array0[128]`、`float array1[64]`、`int array2[256]`，它们紧挨着放。由于动态共享内存只给了你一个**起始地址**（`array`），它是一段连续的字节。对于多种不同类型的数据（比如 `short`、`float`、`int`），你必须自己规划怎么切。

```cpp
extern __shared__ float array[];
__device__ void func() // __device__ or __global__ function
{
  short* array0 = (short*)array;
  float* array1 = (float*)&array0[128];
  int* array2 = (int*)&array1[64]
}
```
`float` 与 `int` 类型要求 4 字节对齐（首地址必须是 4 的倍数），如果上述代码变成 `float* array1 = (float*)&array0[127];`报错。

### 4.2.5 `__grid_constant__`
这是一个性能补丁。对于计算能力大于等于 7.0 的架构，`__grid_constant__` 用于修饰一个 **`const` 限定的、非引用类型、只读的 `__global__` 函数参数**

**解决的问题**：如果在 CPU 端定义了一个很大的结构体`s`，里面可能是一个大矩阵，将后作为参数**按值传递**给 kernel，按值传递则意味着**每一个线程都认为自己拿到了一份独立的**`s`。当你在 kernel 里尝试获取 `s` 的地址，编译器会被迫在每一个线程的本地内存上，为 `s` 复制一份副本！因此用 `__grid_constant__` 告诉编译器，这个参数是只读，所有线程都共用一个地址（指针）。

在之前的矩阵分块例子中，可以将核函数声明修改成这样：
```cpp
__global__ void matMul(const __grid_constant__ Matrix A, const __grid_constant__ Matrix B,  Matrix C);
```

### 4.2.6 `__restrict__`
`__restrict__` 主要解决的是别名问题，例如下面代码

```cpp
void foo(const float* a,
         const float* b,
         float* c)
{
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] * b[0] * a[1];
    // ...
}
```
在 C 类语言中，指针 `a`、`b` 和 `c` 可能存在别名（即指向同一块内存），因此任何通过 `c` 进行的写入都可能修改 `a` 或 `b` 的元素。这意味着为了保证功能正确性，编译器**不能**将 `a[0]` 和 `b[0]` 加载到寄存器中，相乘，然后把结果同时存入 `c[0]` 和 `c[1]`  

通过将 `a`、`b`、`c` 设为受限指针，程序员向编译器断言：这些指针实际上没有别名，在这种情况下意味着通过 `c` 的写入永远不会覆盖 `a` 或 `b` 的元素

> [!NOTE]
> **所有指针参数都需要被设为受限的**，编译器优化器才能从中获得任何好处

如果在 Kernel 里输入输出是两块完全不同的显存，无脑加上 `__restrict__`：

```cpp
__global__ void kernel(const float* __restrict__ input, float* __restrict__ output)
```

# 5. 协作组
## 5.1 同步函数
`__syncthreads()` 阻塞当前线程，直到**同一个 Block** 内的所有线程都执行到了这个点。并且保证：在此之前，所有线程对全局内存和共享内存的写入，对 Block 内其他线程都可见了。（在GPU里写操作一般不会立刻写到全局内存或共享内存，而是先写进**写缓冲区**，或在 L1 里挂着，这里的可见性就是强制要求把所有线程的写操作全部刷到共享内存和全局内存，确保大家都看得见，这样之后比如 warp1 读取共享内存就一定可以读取到 warp0 计算得到的数据

> **致命陷阱**：  
> `__syncthreads()` 允许放在条件语句（如 `if`）里，但条件必须对整个 Block 的评估结果一致！如果条件依赖 `threadIdx.x`，导致一半线程进 if，一半不进，`__syncthreads()` 就会造成**死锁（挂起）**或未定义行为

```cpp
if (threadIdx.x < 16) 
{
    // 只有一半线程会执行到这里
    __syncthreads(); // 死锁！另一半线程永远到不了这里
}
```

其余三种同步函数：
```cpp
int __syncthreads_count(int predicate);    // 统计 predicate 非零的线程有多少个
int __syncthreads_and(int predicate);      // 所有线程都满足条件返回1，否则为0
int __syncthreads_or(int predicate);       // 只要有一个线程的条件判断为1，则函数返回1，否则为0
```

参数通常写成布尔条件，返回值是**同一个 Block 内所有线程投票后的结果**，而且这个结果会**返回给 Block 内的每一个线程**。主要用途是检查“是否存在某个事件”，比如“是否有线程发现了错误”“是否有线程已经完成了某个任务”。

**Warp 级同步：`__syncwarp(unsigned mask=0xffffffff)`**  
由于在 Volta 架构之后，实现了独立线程调度，所以需要在 warp 内设置同步。mask是一个 32 位无符号整数，`0xffffffff` 换算成3进制就是32个1，表示要等待32个线程
- 每个调用线程必须把自己的位 (bit) 设置在 mask 中
- 内存保证

---

### __shared__变量修饰符
表示这个变量**存放在共享内存里**，而不是存在显存里，使得同一个block里面的线程都可以访问。典型应用：粒子模拟中的 tiling 思想，线程 `threadIdx.x` 负责把第 tile 块里第 `threadIdx.x` 个粒子的数据，从 global memory 搬到 shared memory 数组里。

> [!WARNING]
> 这里强调一下tiling思想的一个**坑**（具体的物理模拟粒子系统理论可以链接到我的另一篇文章）
> 在核函数里一般都会有 `if (i > n) return;` 这句，目的是过滤多余线程。但是一旦我们用了tiling思想（或者是需要线程同步执行的情况下`__syncthreads()`），就不能直接过滤掉多余线程。因为如果让多余线程提前退出，这些线程根本不会走到后面的 `__syncthreads()`，而其他线程会在那里死等——这会导致**kernel 挂起（deadlock）**
>
> 其实还有一个**坑**。2010年过后Fermi架构使得：当一个 warp 内所有线程请求同一个 global memory 地址时，内存控制器会把这次请求合并成一次物理内存事务，取回数据后**广播（broadcast）** 给这32个线程的寄存器，而不是发起32次独立的内存请求，实际下来tiling能捡到的优化油水非常少，尤其是对于更偏"计算密集"的kernel。（详情见粒子相关的文章）



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

### 将数据一次性打包进寄存器
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