---
title: CUDA 并行归约性能指标分析
published: 2026-10-05
pinned: true
tags: [CUDA]
category: 技术
draft: false
---

[实验源码与本地测量数据](https://github.com/NaieChars/CUDA-/tree/main/labs/reduction)

> 环境配置：RTX 5060 Laptop GPU，26 SM，L2=32 MiB，驱动 592.00，CUDA 12.8.61，Nsight Compute 2025.2.0；Windows，Release、原配置 native 架构、-lineinfo