---
title: 从 C 到 C++：我的学习总结（CG方向）
published: 2026-07-01
pinned: true
description: 是时候好好系统复习/学习一下C++了，针对于CG方向，这是我的学习总结。
tags: [C++]
category: 技术
draft: false
---

# 从 C 到 C++：我的学习总结（CG方向）

## 一、前言
- 为何要写这篇总结性笔记：我们学校大一的程序设计与算法课程是基于C语言进行的，在 C 语言基础扎实的情况下，加之学习 CG 不可避免会使用C++，于是我打算好好系统地学习一下，尤其是许多C++的现代特性，这在 CG 里有重要用途。这篇文章可以相当于是一份给有相同背景的人的参考路径吧，里面也记录了我踩过的坑，也包含了一些 CG 里 C++的常见应用场景，我尽量不写成长篇大论的语法字典。
- 学习方式：我给自己制定了一个适合 CG 方向学习 C++ 的路线（即目录所呈现内容，包含学习先后顺序）。

注：有不当之处欢迎在评论区指正！

---
 
## 二、C++ 与 C 的基础语法的不同点

### 引用
引用在 CG 里的用途主要是在函数传参，不用再拷贝一份，直接可进行修改，或者加上 `const` 变成只读。

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
由于在 OpenGL 中 Shder / modelPath / texturePath 等的存储一般是字符串
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
为什么不用 `NULL`：  
因为很多编译器就是 `#define NULL 0`，本质上 `NULL` 是0，可能会导致函数重载有问题。

### enum class
`enum class` 是现代 C++ 推荐的枚举类型，必须通过 `类型名::成员名` 访问，**不会污染命名空间**。 
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

### struct 与 class
C++ 的 `struct` 可以包有成员函数、构造函数、析构函数、继承……几乎和 `class` 一样。两者的区别在于默认权限不同。  

> [!IMPORTANT]
> 现代 C++ 有个约定：`struct` 表示纯数据，`class` 表示有行为的对象。

### this 指针与 const 成员函数
根据 `this` 指针可以返回自己的特性，可以设计链式调用（Method Chaining）。
- C++ 风格

```cpp
class Camera
{
public:
    Camera& Move(/*参数*/)
    {
        position += ...;
        return *this;   // 函数返回 camera自己的引用（Camera&）
    }
};
// 调用
camera.Move(1.0f, 0.0f, 0.0f).Rotate(0.1f, 0.0f);
```

- C 风格相似写法

```c
Camera* Move(Camera* self); // 类似 C 风格的“面向对象”
```

<details>
<summary>const 成员函数背后真正的实现原理</summary>

**每一个非静态成员函数，都隐藏着一个 `this` 指针。** 所以成员函数内部之所以能直接访问 `position`，其实都是通过 `this->position` 做到的。  
如果是 const 成员函数，下面举个例子：

```cpp
class Camera {
public:
    int position;
    
    void Print() const {
        // 这句其实等价于 this->position = 10;
        position = 10;          // 编译错误！
        // 错误原因：this 的类型是 const Camera*，
        // 所以 this->position 是 const int，不能修改。
    }
};
```
`*this` 的类型是 `Camera* const` （顶层const，`this` 不能变，`*this` 可以变）
而成员函数末尾的 `const` 修饰的是 `*this`，因此 `this` 类型变成了 `const Camera* const`，**指向的对象从“可修改”变成了“只读”**。

</details>

### 构造函数与成员初始化列表
在 CG 里面，我们创建一个 `Shader` 对象，`Texture` 对象。

```cpp
Shader shader("lighting.vert", "lighting.frag");
Texture texture("wall.png");
```
在 Shader 构造函数里面就可能已经完成了读取 Shader 文件，编译 Shader，链接程序。在 Texture 构造函数里可能已经完成加载图片，上传 GPU，保存纹理 ID。

> [!IMPORTANt]
> <details>
> <summary>有三种成员变量必须用初始化列表[可展开详情]
> </summary>
> 
> 1. const 成员变量（常量，初始化后不能改）  
> 
> 2. 引用成员变量（引用必须绑定，不能空着再“赋值”）  
> 
> 3. 没有默认构造函数的类类型成员（它必须马上被构造，不能等到函数体再初始化）  
> </details>

### static 成员
非静态成员函数需要创建一个对象后才能调用，而静态成员函数可以通过 `类名::函数()` 直接调用。本质还是静态成员函数没有 `this` 指针，这也导致了 `static` 成员函数没法访问普通成员函数与普通成员变量。  

C++17 后，static 成员的声明与定义可以直接写在类里面，会用到 `inline` 关键字，目的是告诉编译器和链接器这个变量可能在多个编译单元（.cpp 文件）中被定义，要帮助我合并成同一个，避免报重定义错误。  
其实可以用更简单的 `constexpr` 修饰，`constexpr` 隐含 `inline`，那么它其实一直都可以在类内初始化。

```cpp
// Camera.h (C++17)
class Camera {
public:
    inline static int count = 0; // 声明 + 定义，可以直接赋值

    // C++17 后，constexpr 隐含 inline，所以对于 const 成员更推荐这样写
    constexpr static int maxFov = 120; 
};
```

### 对象生命周期
**记住这个准则：资源的生命周期一定要比使用它的对象长。**  

简单举个 OpenGL 里的一个例子，Shader 和源码字符串
```cpp
const char* vertexSrc = R"(...着色器代码...)";  // 源码字符串是“资源”
GLuint shader = glCreateShader(GL_VERTEX_SHADER);
glShaderSource(shader, 1, &vertexSrc, nullptr);  // 只是告诉 OpenGL 源码在哪，关键在于这里的 &vertexSrc
glCompileShader(shader);

// 危险：如果 vertexSrc 在这个点之前被释放（比如出了作用域），glCompileShader 可能读到垃圾
// 安全做法：保证 vertexSrc 在 glCompileShader 调用时依然有效
```

这样就诞生了 **RAII** 的自动化管理流程：**对象生命周期管理资源生命周期。**

