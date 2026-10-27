---
title: CUDA SSAS 分析GPU机器码
published: 2026-10-12
pinned: true
description: 通过 NVIDIA 反汇编工具 cuobjdump 分析设备端核函数指令构成，为性能分析提供证据
tags: [CUDA, C++, cuobjdump]
category: 技术
draft: false
---

> [!NOTE]
> 本文尚未完结

# 1. 编译与汇编基础
GPU 接收的是按其指令格式编码的机器码，**汇编**用可读的名称、寄存器、操作数描述这些底层操作，**反汇编**即把机器码转换成可读指令的过程。

## 整体过程

```
   CUDA C++（kernels.cu）
           ↓ 编译前端
PTX：面向虚拟指令集的中间表示；寄存器也是虚拟的
           ↓ ptxas / 后续机器码生成
目标 GPU 的机器码，存放于 cubin 等设备代码对象中
           ↓ cuobjdump / nvdisasm 反汇编
SASS：目标 GPU 机器指令的可读文本

另一条可能路径：程序携带 PTX → 驱动 JIT → 当前设备的机器码
```

> **总结：PTX 帮你看中间层的操作语义；实际 SASS 帮你检查某个目标版本最终生成的指令。**

注意：`.exe` 同时含主机程序和设备代码；`cubin` 是设备代码对象

# 2. 相关工具使用
## 2.1. 工具确认
- **cuobjdump**: 描述 `exe` 里有哪些设备指令和资源声明
- **nvdisasm**：描述独立 cubin 的反汇编、控制流是什么
- **ncu-ui**：在图形界面关联源码、SASS 和指标

> [!NOTE]
> 得到的反汇编内容因不同设备有差异，本机 GPU 为 RTX 5060 laptop，sm_120，cuda 12.8

先确定本地有 cuobjdump，执行下面的 powershaell 指令即可确认。理想输出为：显示工具路径，版本含 12.8，最后一行 True

```powershell
Set-Location 'D:\CG\Kernel_Optimization\cuda-labs'
Get-Command cuobjdump.exe, nvdisasm.exe, ncu.bat, ncu-ui.bat
cuobjdump.exe --version
Test-Path '.\build\labs\sgemm\Release\sgemm.exe'
```

# 3. 指令读取
**结构：操作名，写哪里，从哪里取值，是否有条件**

对下面 LDS / FFMA，**目的操作数在前；不要把这个位置规则机械套到 store 上**。

```
/*03b0*/ LDS     R28, [R27+0x400];
/*03c0*/ LDS.128 R8,  [R0];
/*0420*/ FFMA    R8,  R8, R28, R29;
/*0600*/ @!P0 BRA 0x2a0;
```

以上是简化版本，即不包括完整行中的 &wr、&req、?trans、?WAIT，省去了编码和调度标记（这些之后提及），**反汇编结果完整一行如下**：

```
/*0000*/    LDC   R1,   c[0x0][0x37c]    &wr=0x0   ?trans1;  /* 0x0000df00ff017b82 */
```

这里只给出计组课上没有提及的内容或者是跟 GPU 相关的内容

| 写法 | 含义 |
| --- | --- |
| P0,@P0, @!P0 | 谓词；条件真/条件假时执行 |
| LDS / STS | shared 到寄存器 /寄存器到 shared |
| GDS / GTS | 同上，换成 global |
| FFMA, D,A,B,C | D ← A×B+C，FP32融合乘加 |
| BRA | 分支跳转，针对程序计数器 |
| BAR.SYNC | 线程同步屏障 |
| MOV / IADD / ISETP | 搬值 / 整数加 / 比较并设置谓词 |
| RZ / RU | 零寄存器 /uniform 寄存器 |

具体的指令族可查看 [官方文档](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-binary-utilities/index.html#instruction-set-reference)

> [!WARNING]
> 地址从 0x400 到 0x440 偏移了多少字节？
>
> 这里的偏移单位是字节！不是比特！也就是说偏移了 64 B

**说明 `.128` 后缀**

后缀 `.128` 表示每个参与执行的线程读取 128 bit，也就是 16 字节。 而普通的 `LDS R8, [R0]`; 默认读取 32 bit，也就是 4 字节。

```
/*03c0*/ LDS.128 R8,  [R0];
```

表示每个线程从自己 R0 指定的 shared memory 地址开始，连续读取 16 字节，**放进 4 个连续的 32 位寄存器**：R8, R9, R10, R11

### bank 相关知识回顾

Shared memory 被组织成 **32** 个**可以并行访问的** bank。连续的 4 字节字依次分配给 bank 0、1、2……31，然后再次回到 bank 0。因此 bank 0 保存的不只是地址 0～3，还保存地址 128～131、256～259 等位置的数据。

> [!NOTE]
> 这里一个线程读取 128 bit 跟**扇区**有什么关系？
没有关系。。sector 是用来描述从显存读取数据时的存储单位，本机是 32 bit。因此分析 LDS 时，重点是 **bank conflict** 和 **wavefront**；分析普通 global memory 合并访存时，重点才是覆盖了多少 sector。
>

