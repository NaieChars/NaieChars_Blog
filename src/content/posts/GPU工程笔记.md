---
title: GPU工程实践笔记-C++/CUDA/Powershell
published: 2026-10-03
pinned: true
description: 记录可复用的工程设计、资源管理、错误处理、命令行参数解析与自动化实践。
tags: [CUDA, C++, Powershell]
category: 技术
draft: false
---

[查看源码](https://github.com/NaieChars/CUDA-/blob/main/labs/sgemm/src/main.cu)

# C++ 部分

> [!NOTE]
> 在阅读本文前，你需要掌握 C++ 的基本语法，并了解一些 C++ 的常用工程技巧，包括但不限于：接口与实现分离，RAII，工厂模式，配置模式……
> 如果你忘记了一些常用语法，部分语法在文中也会略有提及。

本文以 `labs/sgemm/src/main.cu` 为例，整理了 CUDA 实验框架中可复用的 C++ 工程思路。主要讲述的是**各部分的分工设计**，以及**关键语法**如何支撑这些设计。以后的跑分实验均可沿用此模版，有很好的学习意义。

> 学习进度（2026-10-19，第二讲）：在第一讲基础上，补充非负 int 解析、浮点容差检查与矩阵大小的乘法溢出防护。后续章节随对话补充；当前内容基于源码阅读与标准资料核对，未新增运行验证。

# 1. 基本架构

## 1.1 main 控制流程

```cpp
try { return run(parse_options(argc, argv)); }
```

这里先执行 `parse_options`，成功返回配置后才调用 `run`。

**主控制流程**如下：

```text
命令行文本
    ↓ parse_options：转换类型，检查参数约束
Options 配置
    ↓ run：选择路径并组织实验
    ├─ help / list / plan：输出信息后返回
    └─ GPU 实验：准备数据 → 启动核函数 → 校验 → 测量或采集
```

`--help` 等分支位于 `run` 内，也会先经过 `parse_options`，即帮助参数不会跳过所有参数检查

| 模块 | 当前源码位置 | 主要职责 |
| --- | --- | --- |
| 配置及参数检查 | `main.cu` 的 `Options`、`parse_*`、`matrix_elements` | 将文本转为配置，检查数值与组合约束 |
| 正确性检查 | `main.cu` 的 `validate` | 比较 GPU 输出与 CPU double 参考值 |
| 实验编排 | `main.cu` 的 `run` | 查询设备、准备数据、选择检查/计时/采集路径 |
| 程序入口 | `main.cu` 的 `main` | 连接解析与执行，处理传播到这里的标准异常 |
| 算子接口与实现 | `kernels.cuh` / `kernels.cu` | 描述变体、计算启动配置、分派并实现核函数 |
| 公共基础设施 | `common/include/cuda_*.cuh` | 管理显存、CUDA 检错和事件计时 |

## 1.2 匿名命名空间

> **作用**：把实现细节留在本翻译单元

```cpp
namespace
{
constexpr int kInvalidArguments = 2;
constexpr int kNotImplemented = 3;
constexpr int kValidationFailed = 4;
// Options、解析函数、validate、run 等也在其中。
}
```

没有名字的 `namespace` 称为匿名命名空间。这里用它收纳**本翻译单元**（一个源文件经过预处理、展开头文件后交给编译器的内容）的内部实现，避免把通用名字（如 `run`、`Options`）作为跨文件接口暴露出去。

`constexpr int` 在这里声明可用于常量表达式的整数常量；`k` 前缀只是常量命名习惯。**用 `kValidationFailed` 表达返回原因**，比散落的 `return 4` 更容易读懂。

## 1.3 Options

> **作用**：把一次实验的配置组织成一个值，类似于<u>配置模式</u>。

```cpp
struct Options
{
    int mode = 0;
    size_t m = 512, n = 512, k = 512;
    int warmup = 10;
    int iterations = 100;
    double atol = 1e-4, rtol = 1e-4;
    bool help = false, list = false, plan_only = false;
    bool check_only = false, profile = false;
};
```

与 `class` 类似，执行 `Options options;` 时，这里的各成员会使用相应默认值，不需要额外写一个只负责赋值的构造函数。

实验配置组合成 `Options` 后，解析器负责构造配置，执行函数通过 `run(const Options& options)` 读取配置。相较于一长串独立参数，调用更清楚，新增配置项时也更容易维护。

注意：`Options` 本身没有强制所有约束。例如仍可手动写入 `options.iterations = 0`。当前程序依靠 `parse_options` 检查其负责的输入约束，`run` 再检查设备相关限制；“配置结构”不等于“永远不可能产生非法状态的类型”。

# 2. 针对命令行参数错误的处理

## 2.1 检测命令行字符串合规性

### 2.1.1 严格解析无符号十进制整数

#### 核心实现

```cpp
size_t parse_unsigned(const std::string& text, const char* name)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string(name) + " must be an unsigned integer");

    const auto value = std::stoull(text);
    if (value > std::numeric_limits<size_t>::max())
        throw std::invalid_argument(std::string(name) + " is too large");

    return static_cast<size_t>(value);
}
```

这个函数承诺：成功时返回一个能由 `size_t` 表示的非负整数；失败时抛出异常。

内部顺序是：**检查文本格式 → 转换为中间整数 → 检查目标类型范围 → 转为目标类型**。转换成功，只能说明标准库能够提取数值，不能自动证明它符合本程序的输入规则。

#### 参数：输入内容与错误上下文分开

```cpp
parse_unsigned("512", "M");
parse_unsigned("abc", "warmup"); // 报错会带上 warmup
```

- `const std::string& text`：**以只读引用接收字符串**。已有 `std::string` 传入时不必复制字符串；若传入 `argv[i]` 这样的 C 字符串，**仍需先隐式构造临时 `std::string`**，临时对象可绑定到这个引用。
- `const char* name`：借用一个以空字符结尾的字符序列，用于说明哪个参数出错。这里不取得其所有权；`const` 修饰所指字符，并不表示指针变量本身不可改指向。
- `size_t`：我在 Windows 上表示的是 64 位无符号整数，具体位宽跟系统有关。

#### empty、find_first_not_of 与 npos

`text.empty()` 检查是否为空。

`text.find_first_not_of("0123456789")` 从头寻找第一个不属于字符集合 `0`～`9` 的字符，**返回第一个非法字符位置**。

| 文本 | 查找结果 | 含义 |
| --- | --- | --- |
| `"512"` | **`std::string::npos`** | 没有找到非数字字符 |
| `"-1"` | `0` | 第一个字符就不属于集合 |
| `""` | **`std::string::npos`** | 没有字符，因此也找不到违规字符 |

`std::string::npos` 是字符串类型提供的静态成员常量，此处表示**没有找到**。因此 `!= npos` 表示确实找到了违规字符[标准草案：字符串查找](https://eel.is/c++draft/string.find)、[查找条件定义](https://eel.is/c++draft/string.view.find)

空字符串同样会返回 `npos`，所以需要单独用 `empty()` 拒绝它。

#### `stoull`

```cpp
const auto value = std::stoull(text);
// 此处相当于：
// const unsigned long long value = std::stoull(text);
```

`std::stoull` 是 `<string>` 提供的自由函数，返回 `unsigned long long`；`auto` 让编译器根据初始化表达式推导类型，并没有让变量变成动态类型。

`std::stoull` 可以只转换字符串开头的有效部分，例如直接调用 `std::stoull("12x")` 可返回 12。因此，本程序先要求全文都是数字。这里的 `unsigned` 也不能替代文本规则检查。

> [!NOTE]
若纯数字文本表示的值连 `unsigned long long` 都放不下，`std::stoull` 会**抛出** **`std::out_of_range`**，不会执行后面的 `size_t` 范围检查。**即不是所有错误都是本函数手动抛出的 `std::invalid_argument`**。[标准草案：数值转换](https://eel.is/c++draft/string.conversions)
>

#### numeric_limits 与 static_cast

```cpp
std::numeric_limits<size_t>::max()
```

`numeric_limits<size_t>` 指定查询 `size_t` 特征的类模板特化，`::max()` 调用它的静态成员函数。它返回该类型的最大可表示值，不需要先创建一个对象。[标准草案：数值类型特征](https://eel.is/c++draft/numeric.limits.members)

`static_cast<size_t>(value)` 显式表达转换意图，但它**本身不负责抛出越界异常**。

#### throw 与字符串拼接
> **`throw` 的核心思想：底层描述错误，上层决定处理方式**

```cpp
throw std::invalid_argument(std::string(name) + " is too large");
```

`std::string(name)` **先构造字符串对象**（很容易忘记，直接用 C 字符串指针相加），让后面执行字符串拼接

`throw` 抛出异常后，这条执行路径立即离开当前函数，沿调用链寻找匹配的 `catch`。当前文件的入口负责接住传播到这里的标准异常：

```cpp
// main()...
catch (const std::exception& error)
{
    std::fprintf(stderr, "Error: %s\nUse --help for usage.\n", error.what());
    return kInvalidArguments;
}
```

`std::invalid_argument` 和 `std::out_of_range` 都派生自 `std::exception`，可被这里的基类引用接住。`error.what()` 读取异常的描述文字。

这样，**解析函数负责发现并描述错误，入口负责面向用户输出和选择退出码**。这种职责划分**便于复用解析函数，也便于单独检查解析行为**。

不过，当前的 `CUDA_CHECK` 宏走的是直接打印并 `exit` 的路径，不经过此 `catch`。异常传播时的栈展开与直接 `exit` 的资源清理行为也不同，不能把两者视为同一种错误处理机制。

  > [!WARNING]
> 该函数只是检测格式是否有效，格式有效不代表业务有效。由于在我的 CUDA 性能实验中，矩阵的维度必须大于 0，而这项条件束缚是后续 `matrix_elements` 检查的业务约束。所以这里体现了一个关键的设计思路：**将格式、类型范围和业务约束分层检查，并让错误携带调用上下文**。

### 2.1.2 复用已有解析逻辑，收紧到 int 范围
#### 核心实现

```cpp
int parse_int(const std::string& text, const char* name)
{
    const size_t value = parse_unsigned(text, name);
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(name) + " is too large");
    return static_cast<int>(value);
}
```

这个函数在已有解析能力上增加一层限制：先复用 `parse_unsigned` 的文本与范围检查，再确认数值不超过 `int` 的最大值，最后转换。

`static_cast<size_t>(std::numeric_limits<int>::max())` 将比较上界显式转换到与 `value` 相同的类型。在本项目的平台上，`size_t` 可以表示这个正整数上界。比较通过后，再将 `value` 转成 `int`。

> [!NOTE]
> 名称里的 `int` 表示返回类型。由于复用了 `parse_unsigned`，它实际只接受非负整数，并不是一个接受正负号的通用 int 解析器。

为什么这些配置用 int？`mode` 是少量模式编号，`warmup`、`iterations` 则用于 int 循环计数，返回类型与使用处保持一致即可。矩阵大小与下标继续使用适合表达大小的 `size_t`。

**设计思维**：每层复用下层已经建立的保证，只增加自己负责的约束。例如 `warmup = 0` 可以有效，而 `iterations = 0` 会由后续 `parse_options` 拒绝；这类业务差异不必塞进通用解析函数。

### 2.1.3 浮点容差：完整转换、有限、非负

```cpp
double parse_tolerance(const std::string& text)
{
    size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value) || value < 0)
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    return value;
}
```

浮点文本可以包含小数点、指数等形式，例如 `0.001`、`1e-4`。这里让标准库负责数值语法，再根据本程序的需求检查转换结果。

#### 一个返回值，加一个输出参数

`std::stod` 返回转换得到的 double；第二个参数的类型是 `size_t*`，用于写回第一个未转换字符的下标。

```cpp
size_t used = 0;
double value = std::stod("0.01abc", &used);
// value 为 0.01，used 为 4。
```

`&used` 取得变量地址，让函数通过指针修改调用者的变量。这是**输出参数**：一个普通的局部变量即可接收结果，不涉及动态分配。

`used == text.size()` 表示没有剩余字符；`used != text.size()` 则拒绝只转换成功一部分的文本。注意 `used` 是停止位置，不是有效数字的个数。[标准草案：字符串数值转换](https://eel.is/c++draft/string.conversions)

如果文本根本无法转换，`stod` 会直接抛出 `invalid_argument`；如果转换报告超出范围，则抛出 `out_of_range`，不会继续执行后面的 if。

#### 三种检查各自保证什么

| 条件 | 拒绝的情况 | 例子 |
| --- | --- | --- |
| `used != text.size()` | 有未转换的尾部字符 | `"0.01abc"` |
| `!std::isfinite(value)` | NaN 或正负无穷 | `"nan"`、`"inf"` |
| `value < 0` | 负容差 | `"-0.01"` |

`std::isfinite` 检查数值是否有限；零属于有限值。仅判断 `< 0` 无法拒绝 NaN 和正无穷，因此需要独立检查。[Microsoft：cmath 分类函数](https://learn.microsoft.com/en-us/cpp/standard-library/cmath?view=msvc-170)

这里也有输入规则上的取舍：`stod` 可以跳过前导空白，因此 `" 0.01"` 可以通过当前函数，而 `"0.01 "` 留下尾部空白，会被完整性检查拒绝。前导加号、科学计数法也可接受；它没有沿用上一节“全文只能是数字”的规则。[Microsoft：strtod 的转换规则](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/strtod-strtod-l-wcstod-wcstod-l?view=msvc-170)

**设计思维**：标准库负责解释数值文本，程序负责规定结果应满足的条件。容差需要是有限非负数，才能用于后续数值比较。

## 2.2 矩阵大小：在乘法发生前防止溢出

```cpp
size_t matrix_elements(size_t rows, size_t columns, size_t element_bytes)
{
    const size_t limit = std::numeric_limits<size_t>::max();
    if (rows == 0 || columns == 0 || rows > limit / columns)
        throw std::invalid_argument("matrix dimensions must be positive and fit size_t");
    const size_t count = rows * columns;
    if (count > limit / element_bytes)
        throw std::invalid_argument("matrix byte size does not fit size_t");
    return count;
}
```

### 2.2.1 为什么不能先乘，再检查

```cpp
// 错误思路：
if (rows * columns > limit) { /* 报错 */ }
```

`rows`、`columns` 都是 `size_t`，乘积也按该无符号类型计算。超出范围时结果按模回绕，比较看到的已经不是数学上的真实乘积。

因此，无符号乘积不会通过 `> size_t最大值` 来暴露问题。无符号回绕是语言规定的行为，但会破坏“需要分配多少元素”的业务含义。[标准草案：无符号整数运算](https://eel.is/c++draft/basic.fundamental)

### 2.2.2 把乘法约束改写成除法约束

对正整数 `columns`，希望满足：

```text
rows × columns ≤ limit
等价于
rows ≤ floor(limit / columns)
```

因此，代码先检查 `rows > limit / columns`，通过后才计算乘积。整数除法自然向下取整，正好给出允许的最大行数。

假设一种类型最大值为 255：`rows = 20`、`columns = 13` 时，真实乘积为 260；先检查 `20 > 255 / 13`，也就是 `20 > 19`，就可以提前拒绝。这里是便于推导的假想范围，不是本机 size_t 的宽度。

### 2.2.3 短路求值保护除法

```cpp
rows == 0 || columns == 0 || rows > limit / columns
```

这里条件的排列具有安全意义：当 `columns == 0` 时，内建 `||` 不再计算后面的除法，从而避免除零。检查前置条件必须发生在使用这个条件的运算之前。[标准草案：逻辑或](https://eel.is/c++draft/expr.log.or)

### 2.2.4 元素数与字节数是两层约束

第一道检查保证 `rows * columns` 能表示；第二道检查保证 `count * element_bytes` 也能表示。

元素数装得下，不代表再乘 `sizeof(float)` 或 `sizeof(double)` 后的字节数仍装得下。即使函数最后返回的是元素个数，调用者分配和拷贝内存时也需要安全的字节数。

```cpp
matrix_elements(options.m, options.n, sizeof(double));
```

解析阶段对 C 的形状使用 `sizeof(double)`，对应 CPU double 参考结果的大小；GPU 的 float 输出需要的字节数更少。`sizeof(T)` 返回类型 T 占用的字节数，类型形式不需要创建对象。

> [!NOTE]
> 当前调用处都传入 `sizeof(float)` 或 `sizeof(double)`，保证 `element_bytes > 0`。函数自身没有检查第三个参数是否为零，因此不能假设它能安全接受任意 size_t 参数组合。

检查通过仅表示**大小的算术表达没有超出 size_t 范围**，并不保证主机内存、显存或容器容量足够，实际分配仍可能失败。

本节待自测：若某整数类型最大值为 255，矩阵形状为 10×10，每元素 4 字节，两道检查各自会怎样？解释时区分元素数与字节数。
