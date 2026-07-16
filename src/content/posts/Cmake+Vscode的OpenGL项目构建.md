---
title: 用 Cmake + VSCode 构建 OpenGL 项目
published: 2026-07-16
pinned: true
description: 我每次在构建 OpenGL 项目的时候，都会时不时地踩坑，这里详细记录了如何一步步在 Windows 中正确用 Cmake 构建一个 OpenGL 项目。
tags: [Cmake, OpenGL, GLFW, GLM, GLAD, C++]
category: 技术
draft: false
---

## 前言
本文主要记录如何使用 CMake + VSCode 在 Windows 环境下搭建图形学项目的开发环境，涵盖 GLFW、GLAD、GLM 等常用库的配置与集成。

文中内容源于我的实际踩坑经历，力求简洁实用，也欢迎读者批评指正。

### 你可能会用到的 powershell 命令
- 清空文件夹：`Remove-Item -Recurse -Force 文件夹名`
- 展现某个文件夹的下一级所有文件与文件夹：`Get-ChildItem 文件夹名`
- 列出文件夹中所有文件与文件夹：`ls -Recurse`
- 查看某文件中某一行的内容：`Get-Content 文件名 | Select-Object -Index 第几行` （注意这里的第几行应该比实际行少1，因为行索引从0开始，比如我要查看12行，我命令里写11）

### 你会用到的 VSCode 扩展
- CMake Tools
- Shader languages support for VS Code


## 第一步、在本地创建新项目

#### 1. 在本地创建一个项目

- **项目结构推荐：**
  ```txt
    项目名/
    |———— CMakeLists.txt
    |———— third_party/
    |            |──── glad/
    |            |       |──── src/
    |            |       └———— include/
    |            |———— glfw/
    |            |———— glm/
    |            |———— stb/
    |———— src/
    |      └─── main.cpp
    └─── .gitignore
  ```

- 在 OpenGL 项目中一般都会用到下面这五个常用库：
    - GLFW：创建操作系统窗口，并绑定 OpenGL 渲染上下文；处理键盘、鼠标等输入事件。
    - GLAD：这是 OpenGL 的函数加载器，在程序运行时，动态获取当前显卡驱动的 OpenGL 函数地址 **（必须项）**
    - GLM：提供图形学常用的数学运算，比如向量、矩阵、叉积等等
    - stb_image.h：放在 stb 文件夹内，负责图片读取，将图片解码为原始像素数组存储在 CPU 中，用于环境贴图和纹理贴图
    - stb_image_write.h：图片输出，将渲染结果保存为静态图片

#### 2. 拉取 GLAD

- 打开 <https://gen.glad.sh>
- 在页面中，Generator 默认 C/C++，APIs 里选择 gl，下拉列表里选择 Version 4.6（现在的最新版本）。将 gl 同行右侧的 Compatibility 更改为 Core（我现在的布局大致是这样，之后可能会稍加变动），其他可以先不用管，下拉到最后在右下角点击 GENETATE，然后在弹出的界面选择下载 glad.zip，最后解压。
- 解压过后你会得到一个 glad 文件，里面应该有 inlude 与 src 两个文件夹，将他们整个拷贝到项目的 third_party/glad/ 下（如上图结构）

#### 3. 拉取 GLFW
- 打开 <https://github.com/glfw/glfw>
- 克隆整个仓库到本地，你应该会得到一个叫 glfw-master.zip 的文件，解压它。
- 将 glfw-master 文件里的内容全部拷贝到 third_party/glfw/ 下（如上图结构）

#### 4. 拉取 GLM
- 打开 <https://github.com/g-truc/glm>
- 克隆整个仓库到本地，你应该会得到一个叫 glm-master.zip 的文件，解压它。
- 将 glm-master 文件里的内容全部拷贝到 third_party/glm/ 下（如上图结构）

#### 5. 获取 stb_image.h
- 打开 <https://github.com/nothings/stb/blob/master/stb_image.h>
- 将整个 stb_image.h 拷贝到 third/stb/ 下

#### 6. 获取 stb_image_write.h
- 打开 <https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h>
- 将整个 stb_image_write.h 拷贝到 third/stb/ 下

网址如果打不开建议刷新一下，并用梯子解决


#### 7. 在 .gitignore 文件内写上：

```
build/
.vscode/
```

---

## 第二步、写 CMakeLists.txt 文件

将下面的内容复制进 CMakeLists.txt，**注意把其中“你的项目名”改成你具体的项目名**


```txt
cmake_minimum_required(VERSION 3.15)
project(你的项目名)

# 强制 UTF-8
if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- GLFW ----------
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw)

# ---------- GLAD ----------
add_library(glad STATIC third_party/glad/src/glad.c)
target_include_directories(glad PUBLIC third_party/glad/include)

# ---------- GLM ----------
set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glm)

# ---------- stb ----------
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_SOURCE_DIR}/third_party/stb)

# ---------- 主程序 ----------
add_executable(你的项目名 src/main.cpp)
target_link_libraries(你的项目名 PRIVATE glfw glad glm::glm stb)

# windows 特有的
if(WIN32)   
    target_link_libraries(你的项目名 PRIVATE opengl32)
endif()
```


---

## 第三步、src/main.cpp （验证）

将一下内容复制进 main.cpp，稍后进行运行检查是否报错。

<details>
<summary> 点击展开 main.cpp 代码用于检验</summary>

```cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

int main() {
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Compute Shader Raytracer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n";
        return -1;
    }

    // 今天的验证重点：确认显卡支持compute shader所需的版本
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << "\n";
    std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

    int workGroupCount[3];
    for (int i = 0; i < 3; i++)
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i, &workGroupCount[i]);
    std::cout << "Max compute work groups: "
              << workGroupCount[0] << ", " << workGroupCount[1] << ", " << workGroupCount[2] << "\n";

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
```
</details>

**这里特别强调一下 main.cpp 顶部如何包含库：**

```cpp
// 1. OpenGL 加载器，必须最先
#include <glad/glad.h>

// 2. 窗口/输入库
#include <GLFW/glfw3.h>

// 3. 数学库
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// 4. 图像加载，仅在这一个文件里定义实现宏
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// 5. 标准库
#include <iostream>
#include <vector>
```

---

## 配置并编译

在你 VSCode 的终端中依次输入下面的命令 **（确保下面命令都是在根目录中进行，以及我这里用的是 powershell！）：**  
#### 首先输入：

```powershell
cmake -B build
```

第一步成功标志是：你应该会在最后看到终端给你如下这般信息：

```powershell
···
-- Configuring done (11.1s)
-- Generating done (0.1s)
-- Build files have been written to: D:/CG/raytracer_gpu/build   <- 注意这里：build 文件在根目录生成，这是我的一个项目示例目录
```

> [!tip]
> 如果你已经输入并执行了上述命令，但是发现自己忘保存文件等等情况（总之就是你想重来），你可以按 CTRL + C 终止。然后输入 `Remove-Item -Recurse -Force build` 清空 build 文件夹 （**切记！**）。

#### 接下来输入：

```powershell
cmake --build build
```

第二步成功标志是：你应该会在最后看到终端给你如下这般信息：

```powershell
raytracer_gpu.vcxproj -> D:\CG\raytracer_gpu\build\Debug\raytracer_gpu.exe
```

如果都没问题，那么恭喜你可以进入最后一步了，试着跑一下你刚生成的 exe
#### 最后输入：

```powershell
.\build\Debug\你的项目名.exe       
```

应该会看到一个深灰色小窗口弹出来，同时终端打印类似这样的内容：

```powershell
OpenGL version: 4.3.0 NVIDIA 592.00
GLSL version: 4.30 NVIDIA via Cg compiler
Max compute work groups: 2147483647, 65535, 65535
```

这里打印的是 OpenGL 和 GLSL 版本，以及你 GPU 支持的最大计算工作组数量（了解一下你的 GPU 上限）  
如果一切顺利，恭喜你完成了最难的环境配置部分！

## 以后项目该如何维护？
#### 需要修改 CMakeLists.txt 的情况
在 src/ 里新增 .cpp 源文件（例如 `Camera.cpp`、`Scene.cpp`）
这样改，增添 cpp 文件即可：

```txt
# ---------- 主程序 ----------
add_executable(你的项目名 
        src/main.cpp
        src/Camera.cpp
        src/Scene.cpp  
)
```

#### 不需要修改 CMakeLists.txt 的情况
修改 `main.cpp` 里的代码，在 shaders/ 里增加着色器文件（如 `.comp`、`.vert`、`.glsl`）



## push 到 github
既然有了一个专属于自己的 OpenGL 项目，怎么说都得 push 到你的 github 上吧。  
如果你下面这部分很熟，那么本篇文章的内容就到此为止了。

#### 1. 在你的 github 里创建一个新仓库。

> [!TIP]
> 建立新仓库时最好不要勾选一同创建 README.md，这会让后面的推送过程变得复杂，会多出从远程仓库把 README 拉下来和本地仓库合并的这一步。所以我一般会选择在本地创建 README.md
>

#### 2. 在本地初始化 git
在你的根目录终端内依次输入以下指令：

```powershell
git init
git add .
git commit -m "Initial commit"
```

至此，你的本地仓库已经建立，并完成了第一次提交。不出意外的话，你的根目录下会多出一个 .git 文件夹。

#### 3. 关联远程仓库

复制你刚才在 GitHub 上创建好的仓库地址（形如 https://github.com/用户名/仓库名.git），然后执行：

```powershell
git remote add origin https://github.com/用户名/仓库名.git
```

再最后一步前，先检查你本地分支的类型，因为你的 github 上新建仓库默认分支是 main，输入以下指令

```powershell
git branch
```

如果输出：`* master`，那么将本地分支重命名：

```powershell
git branch -m master main
```

直到看见输出：`* main`

最后输入：

```powershell
git push -u origin main
```

以后提交只用输入 `git push` 即可

如果一切顺利，恭喜你，项目已经成功托管到 GitHub 上了！