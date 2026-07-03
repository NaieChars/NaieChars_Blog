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
<summary>const 成员函数背后真正的实现原理 [点击展开]</summary>

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
> 请了解  
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

### 对象组合
现代 CG 和游戏引擎更偏向组合而不是复杂的继承，记住下面几点即可  
**成员对象选构造，再构造外层对象，析构相反。**  
成员对象几乎都通过初始化列表完成初始化。

### explicit
只要构造函数只有一个参数，优先考虑加上 `explicit` 来禁止隐式转换，防止编译器偷偷创建对象。

<details>
<summary>深入了解隐式转换 [点击展开]</summary>
假如我有一个 `Point` 类，可以用两个坐标构造

```cpp
class Point {
public:
    Point(float x, float y) {}
};
```
下面这种写法在不加 `explicit` 是合法的

```cpp
Point p = {1.0f, 2.0f};
Point p = Point(1.0f, 2.0f); 
// 编译器偷偷把代码变成了这样，实际上创建了一个临时对象，再用这个对象初始化 p。
// 不过，实际上编译器几乎一定会把这个过程优化掉，并不会真的生成了一个临时对象，这个优化叫做复制消除 
```
</details>

### 拷贝构造函数
标准写法：
```cpp
ClassName(const ClassName& other);  // 拷贝构造

// 调用时
Camera c1;
Camera c2 = c1; // 这里是初始化不是赋值！
Camera c2(c1);  // 上式等于这个
```
<details>
<summary>浅拷贝和深拷贝 [点击展开]</summary>
先来看一个危险的浅拷贝

```cpp
class Camera {
public:
    float fov;
    float* config;  // 注意这个指针！！指向一段动态内存（比如一堆额外参数）

    Camera(float f) : fov(f) {
        config = new float[100];  // 从堆上分配100个float
    }

    ~Camera() {
        delete[] config;  // 释放这块内存
    }
};
// 如果这样写
Camera c1(90.0f);
Camera c2 = c1;  // 调用编译器自动生成的拷贝构造函数
```
**编译器自动生成的拷贝构造函数做的是“浅拷贝”**，只会机械复制所有成员变量，这就导致 `c1.config` 和 `c2.config` 现在指向堆里的同一块内存，当函数结束时，`c1` 和 `c2` 都会被销毁，于是同一块内存被释放了两次。  

于是有了深拷贝 

```cpp
// 在类里自己写拷贝构造函数
    // 深拷贝的拷贝构造函数
    Camera(const Camera& other) : fov(other.fov) 
    {
        config = new float[100];   // 1. 先给自己分配一块全新的内存
        for (int i = 0; i < 100; ++i) 
        {
            config[i] = other.config[i]; // 2. 把对方内存里的数据逐字节复制过来
        }
    }
```

在现代 C++ 中，只要不用裸指针（`float*`），改用智能指针或直接存容器（如 `std::vector<float>`），标准库的默认拷贝构造函数就是深拷贝，不用写 `new/delete`，也不用写拷贝构造函数，这就是零原则。

</details>

在 CG 中，`Texture`、`Shader`、`Mesh` 等资源类通常都需要认真设计拷贝行为。因为它们管理的不只是几个变量，而是 GPU 资源或堆内存。很多现代引擎宁愿禁止拷贝（`= delete`），也不会让编译器默认进行浅拷贝，以避免资源重复释放或状态混乱。

### 拷贝赋值
**两个对象都必须已经存在**，当 `c1 = c2` 时，调用的是**拷贝赋值运算符**  
在 CG 中，资源类通常都会自己实现 `operator=`，或者直接禁用拷贝。

<details>
<summary>深入了解拷贝赋值运算符（深拷贝）[点击展开]</summary>

```cpp
class Camera {
    float* config; // 动态分配的内存
    size_t size;
public:
    // 正确的拷贝赋值
    Camera& operator=(const Camera& other) 
    {
        if (this == &other)   // 1. 自赋值检查
            return *this;
        
        delete[] config;      // 2. 释放旧资源

        size = other.size;
        config = new float[size];          // 3. 分配新资源
        for (size_t i = 0; i < size; ++i)  // 4. 深拷贝数据
            config[i] = other.config[i];

        return *this;
    }
};
```
在上面的例子中，`Camera& operator=(const Camera& other)` 叫拷贝赋值运算符，重载了 `=` 号。当把一个对象赋值给另一个已经存在的对象时调用。

返回 `Camera&` 的原因：为了支持链式赋值。  
这是一个深拷贝。

</details>

### 重载运算符

数学库（如 GLM）大量使用运算符重载，使代码更接近数学表达式。  
最经典的重载 `+` 以满足向量加法：

```cpp
class Vec3
{
public:
    float x, y, z;

    Vec3 operator+(const Vec3& other) const
    {
        return {x + other.x,
                y + other.y,
                z + other.z};
    }
};
```
现在 `Vec3 c = a + b;` 实际就是 `Vec3 c = a.operator+(b);`

### 继承
**组合优于继承**

### 虚函数与虚析构函数（override关键字）

<details>
<summary>虚函数基础知识点 [点击展开]</summary>

先看没有虚函数的类
```cpp
class Animal
{
public:
    void Speak() { std::cout << "Animal";}
};

class Dog : public Animal
{
public:
    void Speak() {std::cout << "Dog";}
};

// 创建 Dog 类并试图调用 Dog 的 Speak()
Animal* animal = new Dog(); 
// 这里是在堆上分配一块足够大的内存，用来存放一个 Dog 对象。
// 括号 () 就是告诉编译器：用默认构造函数来初始化这个对象。

animal->Speak();
```
最后其实 `animal->Speak();` 调用的是 `Animal` 类里面的 `Speak()`，而不是 `Dog` 里的 `Speak()`  。

这是因为：  
没有 `virtual` 时，编译器是根据“指针的类型”来决定调用哪个函数，而不是根据“指针指向的对象究竟是什么”。因此，无论把什么派生类对象塞给这个指针，调用的永远是 `Animal::Speak`。这便是 **“静态绑定”**，这也是 C++ 出于对性能的考量才这样设计，如果每个函数调用都要去猜对象的真实类型，程序运行就会慢很多。

于是我们对 `Animal` 类里面的 `Speak()` 前加上 `virtual`，来实现**动态绑定**，之后再调用 `animal->Speak();`指向的就是 `Dog` 里的 `Speak()` 了。

```cpp
class Animal 
{
public:
    virtual void Speak() { std::cout << "Animal"; }
};
```
</details>

<details>
<summary>从内存角度理解虚析构函数 [点击展开]</summary>

当我确定**这个类会被别人继承**时，那就要用虚析构函数。  
还是上面的例子，如果没有 `vietual`，当我想 `delete animal` 时，编译器只会 `~Animal()`，而没有 `~Dog()`，导致内存泄漏。

这里还要注意一下析构顺序，先派生类，再基类。因为如果先释放了基类，基类的指针的都没了，派生类也没法清理了

虚析构函数标准答案：

```cpp
virtual ~Shape() = default;
```

</details>

> [!tip]
在派生类中重写一个基类的虚函数时，建议在函数声明末尾加上**override**关键字。  
`void Speak() final override; `  
假如我重写时将函数名不小心拼写错误，编译器可以直接报错，提醒你你的函数名有问题，实际上并没有实现重写。
>

虚函数就是实现多态的核心机制。它让你可以用统一的基类接口（如 `Draw()`、`Update()`、`Render()`）操作不同类型的对象，而真正执行哪个函数，由对象的实际类型在运行时决定。这也是游戏引擎、渲染框架和 GUI 系统广泛采用的设计方式。

### 多态
**继承是基础，虚函数与基类指针（或引用）是工具，多态才是最终目的。**  
**多态 = 用统一的接口，操作不同类型的对象，而对象自己决定执行哪种行为。** 从而渲染器、场景管理器、游戏循环等系统只依赖统一的接口，而不用关心对象到底是谁，这正是现代游戏引擎和图形学框架能够不断扩展新对象类型，而核心代码几乎不用修改的关键。

### 纯虚函数与抽象类

<details>
<summary>纯虚函数是啥 [点击展开]</summary>

纯虚函数只有接口，没有实现，标志如下：
```cpp
virtual void Draw() = 0;
```

很简单，比如基类 `Shape` 有一个 `Draw()`函数:

```cpp
class Shape
{
public:
    virtual void Draw()
    {
        std::cout << "Draw Shape";
    }
};

// 调用
Shape shape;
shape.Draw();
```

这当然能运行，可是现实中 `Shape` 是什么并不知道，真正知道该具体画什么的的是 `Cube`（画个立方体）、`Plane`（画地板） 这些 `Shape` 的派生类，所以我将基类的 `Draw` 设置成纯虚函数，意思变成 **“我不知道怎么画，你们子类自己决定。”**

</details>

> [!WARNING]
> **抽象类**（含有一个纯虚函数的类）无法进行实例化对象。编译器直接禁止的。

抽象类的最大作用是**规定接口**，至于怎么实现，子类决定。

### Rule of Zero
现代 C++ 更推荐 Rule of Zero：尽量使用 `std::vector`、`std::string`、智能指针等 RAII 类型，让标准库帮你管理类。

---

