---
title: Smoothed Particle Hydrodynamics （SPH流体模拟）理论学习笔记
published: 2026-07-31
pinned: true
description: 记录了学习 SPH 流体模拟的一些知识，包括理论定义、数学推导、算法实现等等
tags: [SPH流体模拟, 计算机图形学]
category: 技术
draft: false
---

```txt
Smoothed Particle Hydrodynamics
|
├── 1. Introduction
|      ├── Eulerian vs Lagrangian
|      ├── Mesh-based vs Meshless
|      └── SPH Overview
|
├── 2. Mathematical Foundation
|      ├── Continuum Mechanics
|      ├── Conservation Laws
|      ├── Navier-Stokes Equation
|      └── Lagrangian Fluid Description
|
├── 3. SPH Approximation Theory
|      ├── Kernel Approximation
|      ├── Particle Approximation
|      ├── Kernel Properties
|      └── Gradient/Laplacian Approximation
|
├── 4. Weakly Compressible SPH
|      ├── Density Estimation
|      ├── Equation of State
|      ├── Pressure Force
|      ├── Viscosity
|      └── Time Integration
|
├── 5. Incompressible SPH
|      ├── PCISPH
|      ├── IISPH
|      └── DFSPH
|
├── 6. Boundary Handling
|
├── 7. Neighbor Search
|
├── 8. GPU Implementation
|
└── 9. Advanced Topics
       ├── Rigid-Fluid Coupling
       ├── Multi-phase Flow
       └── Rendering
```

