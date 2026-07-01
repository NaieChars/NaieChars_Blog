---
title: 从C到C++：我的学习总结（CG方向）
published: 2026-07-01
pinned: true
description: 是时候好好系统复习/学习一下C++了，针对于CG方向，这是我的学习总结。
tags: [C++]
category: 技术
draft: false
---

# 从C到C++：我的学习总结（CG方向）

## 一、前言
- 为何要写这篇总结性笔记：我们学校大一的程序设计与算法课程是基于C语言进行的，在C语言基础扎实的情况下，加之学习CG不可避免会使用C++，于是我打算好好系统地学习一下，尤其是许多C++的现代特性，这在CG里有重要用途。这篇文章可以相当于是一份给有相同背景的人的参考路径吧，里面也记录了我踩过的坑，也包含了一些CG里C++的常见应用场景，我尽量不写成长篇大论的语法字典。
- 学习方式：我给自己制定了一个适合CG方向学习C++的路线（即目录所呈现内容，包含学习先后顺序）。

注：有不当之处欢迎在评论区指正！

---

## 二、C++与C的基础语法的不同点

### 引用
引用在CG里的用途主要是在函数传参，不用再拷贝一份，直接可进行修改，或者加上 `const` 变成只读。

```cpp
void draw(const Mesh& mesh);    // 不修改对象，且不复制对象
void setShader(const Shader& shader);
void update(Camera& camera);    // 修改
```

### namespace
在引擎里可以看到如下结构，很方便区分哪个函数是哪个模块
> [!WARNING]
不要在头文件里面写 `using namespace`，别人包含你头文件后会被迫卷入这一切！
```cpp
namespace Engine::Render
{
    void Draw();
}
//调用时
Engine::Render::Draw();
Engine::Math::Normalize();
```

### string
`std::string` 比C中的 `char[]` 字符串更加方便，本质已经是一个类，可以简洁地进行很多操作，比如拼接，比较，获取长度等（如果不清楚具体操作的话，可以自行搜索）  

下面是更重要的一点：
由于在 OpenGL 中 Shder/modelPath/texturePath 等的存储一般是字符串
```cpp
std::string Shader = "...";
```
但是很多函数（比如 OpenGL、GLFW、Assimp 等 C 风格接口时）需要的是 `const char*`，这就需要通过 `shader.c_str()`与C字符串互换，返回值就是 `const char*`。

### inline
内联一般用在数学库里面，许多函数比如 `dot()`、`cross()`、`normalize()` 等，建议编译器将函数直接展开，减少函数调用开销。

> [!IMPORTANT]
>- 类里面定义的成员函数，默认都是 inline
>- inline 允许函数定义到头文件，当有多个`.cpp`都包含该头文件时，不会发生函数重复定义
>

### 函数重载与默认参数
默认参数写在参数列表末尾，可以替换简单的函数重载，让图形学 API更加简洁。在 CG 中加载纹理 `LoadTexture(path)` 内部可能就是 `LoadTexture(path, true)`，这里的 `true`表示默认翻转图片。

### nullptr
为什么不用 NULL：  
因为很多编译器就是 `#define NULL 0`，本质上 NULL 是0，可能会导致函数重载有问题。

### enum class
`enum class` 是现代 C++ 推荐的枚举类型，必须通过 `类型名::成员名` 访问，不会污染命名空间。  
例如纹理过滤
```cpp
enum class FilterMode   // 纹理过滤方式
{
    Nearest,
    Linear
};
// 调用
texture.SetFilter(FilterMode::Linear);
```

---

## 面向对象

### struct
C++ 的 `struct` 可以包有成员函数、构造函数、析构函数、继承……几乎和 `class` 一样。两者的区别在于默认权限不同。  

> [!IMPORTANT]
> 现代 C++ 有个约定：`struct` 表示纯数据，`class` 表示有行为的对象。

### this 指针
