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

# 1. 基本架构

## 1.1 main 控制流程

```cpp
try { return run(parse_options(argc, argv)); }
```

先看 `main()`，这里先执行 `parse_options`，成功返回配置后才调用 `run`。

**大致的主控制流程**如下：

```text
命令行文本
    ↓ parse_options：转换类型，检查参数约束
Options 配置
    ↓ run：选择路径并组织实验
    ├─ help / list / plan：输出信息后返回
    └─ GPU 实验：准备数据 → 启动核函数 → 校验 → 测量或采集
```

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

没有名字的 `namespace` 即匿名命名空间。这里用它收纳**本翻译单元**（一个源文件经过预处理、展开头文件后交给编译器的内容）的内部实现，避免把通用名字（如 `run`、`Options`）作为跨文件接口暴露出去。

`constexpr int` 在这里声明编译期常量；**用 `kValidationFailed` 表达返回原因**，这样可以避免到处散落 `return 4` ，增加可读性。

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

实验配置组合成 `Options` 后，**解析器**负责构造配置，执行函数通过 `run(const Options& options)` 读取配置。

注意：`Options` 本身没有强制所有约束。例如仍可手动写入 `options.iterations = 0`。当前程序依靠 `parse_options` 检查其负责的输入约束，`run` 再检查设备相关限制。这里其实体现了本次设计的核心：**解析器负责构造配置，但是不保证所有的配置都是合法的，具体是否合法需要按照各自的业务在后续判断。**

# 2. 针对命令行参数错误的处理

## 2.1 检测命令行字符串合规性

### 2.1.1 严格解析无符号十进制整数

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

内部顺序是：**检查文本格式 → 转换为中间整数 → 检查目标类型范围 → 转为目标类型**。若转换成功，我们只能说明标准库能够提取数值，不能说明它符合本程序的输入规则。

#### （1）参数：输入内容与错误上下文分开

```cpp
parse_unsigned("512", "M");
parse_unsigned("abc", "warmup"); // 报错会带上 warmup
```

- `const char* name`：借用一个以空字符结尾的字符序列，**用于说明哪个参数出错**。他是一个常量指针，指针变量本身可改指向。
- `size_t`：在 Windows 上表示的是 64 位无符号整数，具体位宽跟系统有关。

> [!IMPORTANT]
> 如何选择 `size_t` 、 `unsigned int` 和 `int`？（windows）
>
> 这里其实是存储大小的区别，`size_t` 为 64 位，后者为 32 位，在核函数中，一般数据的索引，数据的格式大小用 `size_t`；一个 Block 内线程的索引用 `int`，因为由于运算可能涉及到负数，无符号整数遇到负号会会绕。`unsigned int` 用得比较少（至少目前来看是这样的）

#### empty、find_first_not_of 与 npos

- `text.find_first_not_of("0123456789")` 从头寻找第一个不属于字符集合 `0`～`9` 的字符，**返回第一个非法字符位置**。**如果没有字符，也会返回 `std::string::npos`**
- `std::string::npos` 表示**没有找到**。因此在此语境下 `!= npos` 表示找到了违规字符 [标准草案：字符串查找](https://eel.is/c++draft/string.find)、[查找条件定义`](https://eel.is/c++draft/string.view.find)

> 空字符串同样会返回 `npos`，所以需要单独用 `empty()` 拒绝它。

#### stoull

```cpp
const auto value = std::stoull(text);
// 此处相当于：
// const unsigned long long value = std::stoull(text);
```

- `std::stoull` 是 `<string>` 提供的自由函数，返回 `unsigned long long`

`std::stoull` 可以只转换字符串开头的有效部分，例如直接调用 `std::stoull("12x")` 可返回 12。因此，本程序先要求全文都是数字。**这里的 `unsigned` 不能替代文本规则检查**。

> [!NOTE]
若纯数字文本表示的值连 `unsigned long long` 都放不下，`std::stoull` 会**抛出** **`std::out_of_range`**，不会执行后面的 `size_t` 范围检查。**即不是所有错误都是本函数手动抛出的 `std::invalid_argument`**。[标准草案：数值转换](https://eel.is/c++draft/string.conversions)
>

#### numeric_limits 与 static_cast

```cpp
std::numeric_limits<size_t>::max()
```

- `numeric_limits<size_t>` 非常常用，指定查询 `size_t` 特征的类模板特化，`::max()` 调用它的静态成员函数。它返回该类型的最大可表示值，**不需要先创建一个对象**。[标准草案：数值类型特征](https://eel.is/c++draft/numeric.limits.members)

- `static_cast<size_t>(value)` 显式表达转换意图，但它**本身不负责抛出越界异常**。

#### throw 与字符串拼接

> **`throw` 的核心思想：底层描述错误，上层决定处理方式**

```cpp
throw std::invalid_argument(std::string(name) + " is too large");
```

`std::string(name)` **先构造字符串对象**（很容易忘记，导致有时候写成直接用 C 字符串指针相加），让后面执行字符串拼接

`throw` 抛出异常后，这条执行路径立即离开当前函数，**沿调用链寻找匹配的 `catch`**。当前文件的入口负责接住传播到这里的标准异常：

```cpp
// main()...
catch (const std::exception& error)
{
    std::fprintf(stderr, "Error: %s\nUse --help for usage.\n", error.what());
    return kInvalidArguments;
}
```

- `std::exception`  
- `error.what()`

`std::invalid_argument` 和 `std::out_of_range` 都派生自 `std::exception`，可被这里的基类引用接住。`error.what()` 读取异常的描述文字。

这样，**解析函数负责发现并描述错误，入口负责面向用户输出和选择退出码**。这种职责划分**便于复用解析函数，也便于单独检查解析行为**。

> [!NOTE]
注意到，当前的 `CUDA_CHECK` 宏走的是直接打印并 `exit` 的路径，不经过此 `catch`。异常传播时的栈展开与直接 `exit` 的资源清理行为也不同，不能把两者视为同一种错误处理机制。
> 

  > [!WARNING]
> 该函数只是检测格式是否有效，格式有效不代表业务有效。由于在我的 CUDA 性能实验中，矩阵的维度必须大于 0，而这项条件束缚是后续 `matrix_elements` 检查的业务约束。所以这里体现了一个关键的设计思路：**将格式、类型范围和业务约束分层检查，并让错误携带调用上下文**。

### 2.1.2 复用解析逻辑

> 将解析内容收紧到 `int`

```cpp
int parse_int(const std::string& text, const char* name)
{
    const size_t value = parse_unsigned(text, name);
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(name) + " is too large");
    return static_cast<int>(value);
}
```

这个函数在已有解析能力上增加一层限制：先复用 `parse_unsigned` 的文本与范围检查，再**确认数值不超过 `int` 的最大值**，最后转换。

> [!NOTE]
> 名称里的 `int` 表示返回类型。由于复用了 `parse_unsigned`，它实际只接受非负整数，并不是一个接受正负号的通用 int 解析器。

> [!NOTE]
**为什么这些配置用 int？** `mode` 是少量模式编号，`warmup`、`iterations` 则用于 int 循环计。矩阵大小与下标继续使用适合表达大小的 `size_t`。

**设计思维**：每层复用下层已经建立的保证，只增加自己负责的约束。例如 `warmup = 0` 可以有效，而 `iterations = 0` 会由后续 `parse_options` 拒绝；这类业务差异不必塞进通用解析函数。

### 2.1.3 浮点容差

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

**浮点文本**可以**包含小数点、指数**等形式，如 `0.001`、`1e-4`。这里先让标准库负责数值语法，再根据本程序的需求来检查转换结果。

#### 函数解析

- `std::stod` 将字符串转换为 double；他其实**有两个返回值**，第一个是返回 double 类型的 value，第二个返回通过第二个参数接住（即**输出参数**：第二个参数的类型是 `size_t*`，用于写回第一个未转换字符的下标）

```cpp
size_t used = 0;
double value = std::stod("0.01abc", &used);
// value 为 0.01，used 为 4。
```

`used != text.size()` 拒绝只转换成功一部分的文本。**注意** `used` 是停止位置，不是有效数字的个数。[标准草案：字符串数值转换](https://eel.is/c++draft/string.conversions)
- `std::isfinite(x)` 用于判断 x 是否是有限数，无穷或 NAN 会返回 false
- 小于 0 也需要手动单独检查

如果文本根本无法转换，`stod` 会直接抛出 `invalid_argument`；如果转换报告超出范围，则抛出 `out_of_range`，不会继续执行后面的 if。

**设计思维**：标准库负责解释数值文本，程序负责规定结果应满足的条件。**容差需要是有限非负数**，才能用于后续数值比较。

## 2.2 矩阵大小限制

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

> [!WARNING]
> 不能先乘再检查!
>
> `rows`、`columns` 都是 `size_t`，乘积也按该无符号类型计算。超出范围时结果按模回绕，比较看到的已经不是数学上的真实乘积。

```cpp
// 错误思路：
if (rows * columns > limit) { /* 报错 */ }
```


因此，无符号乘积不会通过 `> size_t最大值` 来暴露问题 [标准草案：无符号整数运算](https://eel.is/c++draft/basic.fundamental)

- **把乘法约束改写成除法约束**

对正整数 `columns`，希望满足：rows × columns ≤ limit，该**表达式等价**于 rows ≤ floor(limit / columns)

因此，代码先检查 `rows > limit / columns`，通过后才计算乘积。整数除法自然向下取整，正好给出允许的最大行数。

- **短路求值保护除法**

```cpp
rows == 0 || columns == 0 || rows > limit / columns
```

这里**条件的排列**具有安全意义：当 `columns == 0` 时，内建 `||` 不再计算后面的除法，从而避免除零。检查前置条件必须发生在使用这个条件的运算之前。

```cpp
matrix_elements(options.m, options.n, sizeof(double));
```

解析阶段对 C 的形状使用 `sizeof(double)`，对应 CPU double 参考结果的大小；GPU 的 float 输出需要的字节数更少。`sizeof(T)` 返回类型 T 占用的字节数，类型形式不需要创建对象。

> [!NOTE]
> 当前调用处都传入 `sizeof(float)` 或 `sizeof(double)`，保证 `element_bytes > 0`。函数自身没有检查第三个参数是否为零，因此**不能假设它能安全接受任意 size_t 参数组合**。

检查通过仅表示**大小的算术表达没有超出 size_t 范围**，并不保证主机内存、显存或容器容量足够，实际分配仍可能失败。

## 2.3 parse_options：从零散文本组装实验配置

前面的解析函数检查一个数值，这个函数则负责识别参数角色，并检查一次实验的整体配置。

### 2.3.1 先看两阶段结构

```text
Options 默认值 + argv 文本序列
    ↓ 第一阶段：扫描参数
    ├─ 无值开关：设置布尔成员
    ├─ 带值选项：消费下一个参数，转换后设置成员
    └─ 位置参数：按原顺序暂存到 positional
    ↓ 第二阶段：解释位置参数并检查整体约束
    mode / M / N / K → 检查模式、矩阵大小、迭代次数、选项互斥
    ↓
按值返回 Options
```

这种分工允许开关穿插在位置参数之间；位置参数彼此的相对顺序仍必须是 `mode M N K`。本节先解释 C++ 的解析实现，终端引号、调用与脚本语法留到后续统一整理。

### 2.3.2 argv 文本与 vector 的职责

```cpp
Options options;
std::vector<std::string> positional;
for (int i = 1; i < argc; ++i)
{
    const std::string arg = argv[i];
    // 按参数角色处理 arg。
}
```

常见调用中，`argv[0]` 是程序调用名称，`argv[1]` 起才是用户提供的参数；`argc` 包含程序名称这一项。`char** argv` 可以理解为指向一组 C 字符串指针，`argv[i]` 取出其中一个以空字符结尾的字符串。[标准草案：main 参数](https://eel.is/c++draft/basic.start.main)

`const std::string arg = argv[i]` 构造一个拥有自身内容的字符串，方便使用字符串成员函数和按内容比较的 `==`。它还保留当前参数的内容，后续即使修改 i，arg 也仍表示原来的选项名。

`std::vector<std::string>` 是字符串的动态数组。遇到位置参数时，`positional.push_back(arg)` 将字符串复制到末尾，保留出现顺序；循环内 arg 离开作用域不会让已保存的字符串悬空。`size()` 取数量，`empty()` 判断是否为空，`operator[]` 按下标访问；下标访问本身不做越界检查，因此后文先检查数量再访问。[标准草案：vector 插入](https://eel.is/c++draft/vector.modifiers)

### 2.3.3 带值选项：一次分支消费两个参数

```cpp
else if (arg == "--warmup" || arg == "--iterations" ||
         arg == "--atol" || arg == "--rtol")
{
    if (++i >= argc) throw std::invalid_argument(arg + " needs a value");
    if (arg == "--warmup") options.warmup = parse_int(argv[i], "warmup");
    else if (arg == "--iterations") options.iterations = parse_int(argv[i], "iterations");
    else if (arg == "--atol") options.atol = parse_tolerance(argv[i]);
    else options.rtol = parse_tolerance(argv[i]);
}
```

以这组收到的文本为例：

```text
下标：  0       1    2       3        4    5    6         7
参数：sgemm     1   128   --warmup    2   64   32   --check-only
```

| i 的位置 | 本轮动作 |
| --- | --- |
| 1、2 | 将 `"1"`、`"128"` 存入 positional |
| 3 | 保存 arg 为 `"--warmup"`；`++i` 移到 4，解析 `"2"` |
| 本轮结束 | for 自身的 `++i` 将 i 推进到 5，不再把 `"2"` 当成位置参数 |
| 5、6 | 将 `"64"`、`"32"` 存入 positional |
| 7 | 将 check_only 设为 true |

最终 `positional` 是 `{"1", "128", "64", "32"}`，对应 mode=1、M=128、N=64、K=32；warmup=2，check_only=true。

`if (++i >= argc)` 先推进索引，再判断有没有值可读。若选项位于末尾，抛异常发生在访问 `argv[i]` 之前。

最后的 `else options.rtol = ...` 依赖外层已将 arg 限定在四个名称中：排除前三个后，只可能剩下 `--rtol`。如果以后新增带值选项，必须一起维护这个分派逻辑。

**设计思维**：把 i 当作读取游标，分支负责消费本参数额外需要的值，外层循环负责推进到下一项。边界检查要发生在读取之前。

### 2.3.4 rfind 的特殊用法：只检查前缀

```cpp
else if (arg.rfind("--", 0) == 0)
    throw std::invalid_argument("unknown option: " + arg);
else positional.push_back(arg);
```

`rfind` 的第二个参数限制匹配起始位置的上界。传入 0 后，只有从下标 0 开始的匹配才可能成功，因此这里等价于判断字符串是否以 `--` 开头。[标准草案：rfind 查找条件](https://eel.is/c++draft/string.view.find)

| arg | `rfind("--", 0)` | 这个条件的结果 |
| --- | --- | --- |
| `"--typo"` | 0 | 识别为未知长选项 |
| `"abc--def"` | npos | 不满足前缀条件 |
| `"128"` | npos | 不满足前缀条件 |

已知选项的分支必须位于这道检查之前，否则合法的 `--warmup` 等也会被当成未知选项。

**设计思维**：明确拒绝拼错的长选项，让用户尽早得到错误信息。位置参数留给后续阶段解释。

### 2.3.5 位置参数与组合约束

```cpp
if (positional.size() > 4)
    throw std::invalid_argument("expected: mode M N K");
if (!positional.empty()) options.mode = parse_int(positional[0], "mode");
gemm_variant(options.mode);
if (positional.size() > 1) options.m = parse_unsigned(positional[1], "M");
if (positional.size() > 2) options.n = parse_unsigned(positional[2], "N");
if (positional.size() > 3) options.k = parse_unsigned(positional[3], "K");
```

这里只覆盖用户提供的前若干项，其余成员继续保留 Options 中的默认值。各处的数量判断同时保证 `positional[index]` 不越界。

`gemm_variant(options.mode);` 虽然没有保存返回值，却有检查作用：当前实现找到合法模式就返回描述，找不到就抛出异常。这里复用描述查询函数验证 mode，避免在解析器里重复维护一套合法编号。

合法编号与是否允许实际执行属于不同检查：解析器检查 mode 是否存在；run 再检查变体的 implemented 标记等执行条件。

后续约束可以分为：

| 检查 | 负责的条件 |
| --- | --- |
| 三次 matrix_elements 调用 | A、B、C 形状及相应字节数可表示 |
| `options.iterations < 1` | 计时迭代次数必须为正 |
| `options.profile && options.check_only` | 两种运行请求不能同时开启 |

两个开关单独都合法，但组合可能冲突；因此，单值解析与整体配置检查需要不同层次。

### 2.3.6 按值返回配置，不返回局部引用

```cpp
return options;
```

函数返回类型是 Options，调用者得到一个结果对象，其有效性不依赖函数内局部变量的生存期。不能为了避免复制，改成返回指向这个局部对象的引用或指针。

对于这里的 `return options`，编译器可以实施具名返回值优化（NRVO），直接在结果位置构造对象；该优化允许发生，但不保证一定发生。即使没有进行优化，按值返回当前配置也具有正确的对象语义。[标准草案：拷贝消除](https://eel.is/c++draft/class.copy.elision)

这也允许入口写成 `run(parse_options(argc, argv))`：返回的临时结果可绑定到 run 的 const 引用参数，在这次调用期间有效。

### 2.3.7 当前解析器的明确边界

| 输入情况 | 当前源码行为 |
| --- | --- |
| 多次提供 `--warmup` | 后面的值覆盖前面的值 |
| `--warmup=2` | 不支持等号形式，被当成未知长选项 |
| `--warmup --plan` | 将 `--plan` 当作 warmup 的值，随后因非数字而报错 |
| 未知短选项 `-x` | 进入位置参数处理，随后由数值解析拒绝 |
| 单独的 `--` | 没有“结束选项解析”的特殊含义，会被拒绝 |
| `--help` 与其他错误参数并存 | 仍要先通过 parse_options，帮助标志不会跳过检查 |

这些是当前小型解析器的取舍，不应将它描述为支持所有常见命令行约定的通用库。

## 2.4 print_help：相邻字符串字面量的拼接

```cpp
std::puts(
    "Usage: sgemm ...\n"
    "Options:\n"
    "  --help  Show help\n");
```

这里相邻的普通字符串字面量会在翻译阶段合并成一个字符串；没有逗号，因此 puts 仍只接收一个参数。这种写法方便把长帮助文本按行排版，不需要运行时通过 std::string 的 `+` 拼接。[标准草案：字符串字面量](https://eel.is/c++draft/lex.string)

源码换行本身不会成为输出换行，文本里的 `\n` 才负责换行；puts 还会在输出末尾额外写入一个换行。字面量相邻拼接只适用于这种语法，两个字符串变量不能直接并排书写来拼接。

本节待自测：若收到 `sgemm 2 --warmup 0 64 --iterations 5 32 16`，positional 最终有哪些字符串？哪些 Options 成员被覆盖？暂不要求运行 GPU，先按读取游标推导。