---
title: Compute Shader 重构 CPU 路径追踪器
published: 2026-07-18
pinned: true
description: 前段时间我按照 RTIOW 和 RTINW 搭建了一个 CPU 路径追踪器，实际渲染速度非常慢，于是我打算用 Compute Shader 重构一下。这其中会涉及到从 CPU 编程到 GPU 编程的思维转变，以及怎么写一个 Compute Shader。本文记录了我重构的完整过程以及一些经验技巧。
tags: [OpenGL, Compte Shader, C++, RayTracer]
category: 技术
draft: false
---

> [!NOTE]
阅读前，请确保你有一定的 OpenGL 基础，且已经跟着《Ray Tracing in Next Week》完成了一遍 CPU 路径追踪器搭建  
为了节省空间，本文的代码均折叠了起来，需要你手动展开。同时，每一个代码块都已经写好了详细的注释，方便逐行理解。   
由于我也是第一次接触 Comepute Shader，我会**从最开始写起**。为了回忆之前所学，我在一些基础的 OpenGL 内容后面也给了详细解释。  
至于为什么要如此详细地给出完整代码，是为了给读者节约点自己架构的时间（懒人模式）
>

---

# 开始
## 一张图厘清 Compute Shader 数据流

<p align="center">
  <img src="/markdown_picture/ComputeShader/ComputeShader.png" width="500">
</p>
<p align="center">
  Compute Shader 数据流
</p>


## 开始我们的第一个 "Hello.comp"

下面是一个简单的 `hello.comp` 文件写法：

<details>
<summary> hello.comp 文件内容</summary>

```glsl
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgb32f, binding = 0) uniform image2D outputImage;    // 声明一张可读写图像

void main()
{
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);  // gl_GlobalInvocation 是 GLSL Compute Shader 的内置输入变量，类型是 uvec3
    ivec2 imageSize = imageSize(outputImage);            // GLSL内置函数，直接问图像多大，不用CPU传参数

    // 检查像素坐标是否越界
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y)
        return;

    vec2 uv = vec2(pixelCoord) / vec2(imageSize);
    imageStore(outputImage, pixelCoord, vec4(uv.x, uv.y, 0.2, 1.0));    // 3个参数：在什么上面画，画到哪个坐标，颜色RGBA
}

// 流程：
// compute shader 把结果写进一张纹理（用image2D写入）
// 一个全屏三角形/quad，配合fragment shader把这张纹理采样出来显示

// 我们在 CPU 端计算需要的线程采用的是向上取整，那么难免会因为尺寸不是整倍数而导致线程有多余
// 这些多余的线程在检查到自己负责的坐标超出图像范围时，会立即 return，结束执行。
```

</details>

由于我的显示结果处理方式是全屏 quad + 纹理采样，所以下面用 `quad.vert`、`quad.frag` 画一个全屏三角形/quad：  

<details>
<summary> quad.vert 文件内容</summary>

```glsl
// quad.vert
#version 430 core

// 不需要顶点缓冲，用顶点ID技巧生成一个全屏三角形
out vec2 texCoord;

void main()
{
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);  // 生成屏幕坐标，三顶点依次为 (0,0), (2, 0), (0, 2), gl_VertexID 是 GLSL 的内置整型变量，表示当前顶点在绘制命令中的索引。
                                                                // 当你用 glDrawArrays(GL_TRIANGLES, 0, 3) 绘制 3 个顶点时，gl_VertexID 的值依次为 0、1、2。
    texCoord = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);  // 转换到裁剪坐标。这里坐标计算是很经典的将三角形放大处理，使其占满范围是 [-1, 1] 的裁剪空间。
}
// 流程：
// 每个顶点独立计算自己的位置，完全并行，结果直接输入到光栅化器。
```
</details>

> [!tip]
不用建**实际的**VAO/VBO传顶点数据，靠 `gl_VertexID`（0,1,2）直接在shader里算出一个覆盖全屏的大三角形。你只需要在CPU端调用 `glDrawArrays(GL_TRIANGLES, 0, 3)`，不用绑定任何顶点缓冲。
>

<details>
<summary> quad.frag 文件内容</summary>

```glsl
// quad.frag
#version 430 core

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D screenTexture;    // 声明一个 2D 纹理采样器 uniform，它是在应用程序中（CPU 端）传入的。

void main()
{
    fragColor = texture(screenTexture, texCoord);   // 在 screenTexture 上，用纹理坐标 texCoord 进行采样，将得到的颜色值直接赋给输出 fragColor。
}
```
</details>

接下来是 `main.cpp` 文件内容，里面包含了对着色器文件获取、编译、链接的自定义工具函数（后面再进行封装）。

<details>
<summary> main.cpp 文件内容</summary>

```cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

const int SCR_WIDTH = 800;
const int SCR_HEIGHT = 600;

// ---------一个检查错误的小工具函数，在关键调用后面加上可以查看 OpenGL 的静默失败-------------
void checkGLError(const char* label) 
{
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "[GL ERROR] " << label << ": 0x" << std::hex << err << std::dec << "\n";
    }
}


// ---- 工具函数：读取shader文件文本 ----
std::string readFile(const char* path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open shader file: "  << path << "\n";
        return "";
    }
    std::stringstream ss;
    ss << file.rdbuf();  // file.rdbuf() 返回文件流的底层缓冲区指针，ss << 把整个文件缓冲区的剩余内容一次性导入ss字符串流中
    return ss.str();
}

// ---- 工具函数：编译单个shader，返回着色器ID，带错误检查 ----
GLuint compileShader(GLenum type, const std::string& source)
{
    GLuint shader = glCreateShader(type);   // 创建一个空的着色器对象，并返回它的句柄
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr); // 给着色器对象内部拷贝一份源码副本
    glCompileShader(shader);    // 编译

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
        std::cerr << "Shader compile error:\n" << infoLog << "\n";
    }
    return shader;
}

// ---- 工具函数：链接program，返回GPU可执行程序，带错误检查 ----
GLuint linkProgram(std::initializer_list<GLuint> shaders)
{
    GLuint program = glCreateProgram();
    for (GLuint s : shaders) glAttachShader(program, s);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) 
    {
        char infoLog[1024];
        glGetProgramInfoLog(program, 1024, nullptr, infoLog);
        std::cerr << "Program link error:\n" << infoLog << "\n";
    }
    for (GLuint s : shaders) glDeleteShader(s);  // 链接完清空shaders
    return program;
}

int main()
{
    if (!glfwInit()) 
    {
        std::cerr << "GLFW init failed\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Compute Shader Raytracer", nullptr, nullptr);
    if (!window) 
    {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) 
    {
        std::cerr << "GLAD init failed\n";
        return -1;
    }


    // ---- 没有绑定VAO的 glDrawArrays不会执行，所以我们仍要绑定一个 VAO，哪怕根本不用顶点属性----
    GLuint dummyVAO;
    glGenVertexArrays(1, &dummyVAO);
    glBindVertexArray(dummyVAO);

    // ----------------- 创建输出图像（compute shader写入的目标）-------------------------
    GLuint outputTexture;
    glGenTextures(1, &outputTexture);                                      // 生成一个纹理对象ID，并保存到outputTexture
    glBindTexture(GL_TEXTURE_2D, outputTexture);                           // 绑定对象到 GL_TEXTURE_2D目标，因为 OpenGL 是状态机，所以后续对 GL_TEXTURE_2D的操作都会作用到这个纹理上
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT);   // glTexStorage2D 而不是 glTexImage2D —— 这是创建"不可变存储"的纹理，compute shader写入这种纹理效率更好，也是目前推荐用法。
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);                                       // 解绑 GL_TEXTURE_2D

    GLuint computeShader = compileShader(GL_COMPUTE_SHADER, readFile("src/shaders/hello.comp"));    // 相对路径，相对于运行exe时的工作目录，
    GLuint computeProgram = linkProgram({computeShader});

    GLuint vertShader = compileShader(GL_VERTEX_SHADER, readFile("src/shaders/quad.vert"));
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, readFile("src/shaders/quad.frag"));
    GLuint quadProgram = linkProgram({vertShader, fragShader});


    while (!glfwWindowShouldClose(window)) 
    {
        // 派发compute shader，把结果写进 outputTexture
        glUseProgram(computeProgram);
        glBindImageTexture(0, outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);   // glBindImageTexture(0... 这里的 0 要和.comp里 binding = 0 对应上, 这是CPU和GPU之间的接口约定

        GLuint groupsX = (SCR_WIDTH + 15) / 16;                 // 整数向上取整
        GLuint groupsY = (SCR_HEIGHT + 15) / 16;
        glDispatchCompute(groupsX, groupsY, 1);                 // 派发 Compute Shader 工作：告诉 GPU 启动 groupsX × groupsY × 1 个工作组。(z=1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);    // 确保屏障之前的所有纹理图像写入（imageStore）对屏障之后的同一图像单元访问可见。

        // 把结果画到屏幕
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(quadProgram);
        glActiveTexture(GL_TEXTURE0);                                           // 激活 0 号纹理
        glBindTexture(GL_TEXTURE_2D, outputTexture);
        glUniform1i(glGetUniformLocation(quadProgram, "screenTexture"), 0);     // 设置着色器程序 quadProgram 中的 uniform 采样器 screenTexture 的值。glGetUniformLocation 查询 "screenTexture" 的位置。值 0 表示它从纹理单元 0 采样
        glDrawArrays(GL_TRIANGLES, 0, 3);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;

}
```

</details>

> [!Warning]
> **这里有一个极其隐蔽的坑！** 在 OpenGL 3.2+ Core Profile 中，所有绘制命令必须在绑定了一个非零 VAO 的情况下才能执行，否则 `glDrawArrays` 会直接失败（产生 `GL_INVALID_OPERATION` 错误）。所以即使你的全屏三角形用的无顶点缓冲技巧，仍然需要绑定一个 VAO，哪怕里面什么都没有。

如果一切顺利，你将得到如下这个彩色窗口

<p align="center">
  <img src="/markdown_picture/ComputeShader/HelloComp.png" width="300">
</p>
<p align="center">
  第一个窗口！
</p>

## 封装 Shader 类
这里先不封装 Camera 类：CPU 路径追踪里Camera类存在的意义，是因为相机要在C++端生成射线（get_ray(u,v)这种方法调用）。但GPU版本不一样——射线生成这段逻辑要整个搬进GLSL里，C++端的Camera退化成只负责把几个vec3（origin、lower_left_corner、horizontal、vertical）通过uniform传给GPU，封装成类反而多一层不必要抽象。

Shader 类就很值得做：

<details>
<summary> Shader.h</summary>

```cpp
#ifndef SHADER_H
#define SHADER_H

#include "glad/glad.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <initializer_list>

class Shader
{
    public:
        GLuint program;

        // compute shader 专用
        static Shader computeShader(const std::string& path)
        {
            GLuint cs = compile(GL_COMPUTE_SHADER, readFile(path));
            return Shader(link({cs}));
        }

        // vertex+fragment组合
        static Shader graphicsShader(const std::string& vertPath, const std::string& fragPath) 
        {
            GLuint vs = compile(GL_VERTEX_SHADER, readFile(vertPath));
            GLuint fs = compile(GL_FRAGMENT_SHADER, readFile(fragPath));
            return Shader(link({vs, fs}));
        }

        // 激活着色器可执行程序
        void use() const { glUseProgram(program); }

        void setInt(const std::string& name, int v) const 
        {
            glUniform1i(glGetUniformLocation(program, name.c_str()), v);
        }

        void setFloat(const std::string& name, float v) const 
        {
            glUniform1f(glGetUniformLocation(program, name.c_str()), v);
        }

        void setVec3(const std::string& name, float x, float y, float z) const 
        {
            glUniform3f(glGetUniformLocation(program, name.c_str()), x, y, z);
        }

    private:
        Shader(GLuint prog) : program(prog) {}

        //-------------------- 读取、编译、链接三件套--------------------

        static std::string readFile(const std::string& path) 
        {
            std::ifstream file(path);
            if (!file.is_open()) { std::cerr << "Failed to open: " << path << "\n"; return ""; }
            std::stringstream ss; ss << file.rdbuf();
            return ss.str();
        }

        static GLuint compile(GLenum type, const std::string& src) 
        {
            GLuint shader = glCreateShader(type);
            const char* s = src.c_str();
            glShaderSource(shader, 1, &s, nullptr);
            glCompileShader(shader);
            GLint success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                char log[1024];
                glGetShaderInfoLog(shader, 1024, nullptr, log);
                std::cerr << "Shader compile error:\n" << log << "\n";
            }
            return shader;
        }

        static GLuint link(std::initializer_list<GLuint> shaders) 
        {
            GLuint prog = glCreateProgram();
            for (GLuint s : shaders) glAttachShader(prog, s);
            glLinkProgram(prog);
            GLint success;
            glGetProgramiv(prog, GL_LINK_STATUS, &success);
            if (!success) {
                char log[1024];
                glGetProgramInfoLog(prog, 1024, nullptr, log);
                std::cerr << "Program link error:\n" << log << "\n";
            }
            for (GLuint s : shaders) glDeleteShader(s);
            return prog;
        }
};

#endif
```

</details>

---

# 单球光追

新建 `src/shaders/raytrace.comp`，把RTIOW的相机射线生成 + 球体求交移植过来：

<details>
<summary> raytrace.comp文件 </summary>

```glsl
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32f, binding = 0) uniform image2D outputImage;

const float INF = uintBitsToFloat(0x7F800000u);

// 相机参数
uniform vec3 camOrigin;
uniform vec3 camLowerLeftCorner;
uniform vec3 camHorizontal;
uniform vec3 camVertical;

// 球体参数，暂时先写死一个球，后面再改成 SSBO 存放多个球
uniform vec3 sphereCenter;
uniform float sphereRadius;

// 光线与球体求交，返回命中距离，没命中返回-1.0
float hitSphere(vec3 center, float radius, vec3 rayOrigin, vec3 rayDir, float t_min, float t_max)
{
    vec3 oc = rayOrigin - center;
    float a = dot(rayDir, rayDir);
    float halfB = dot(oc, rayDir);
    float c = dot(oc, oc) - radius * radius;

    float discriminant = halfB * halfB - a * c;

    if (discriminant < 0.0) return -1.0;

    float root = (-halfB - sqrt(discriminant)) / a;
    if (!(root > t_min && root < t_max))
    {
        root = (-halfB + sqrt(discriminant)) / a;
        if (!(root > t_min && root < t_max))
            return -1.0;
    }
    return root;
}


vec3 rayColor(vec3 rayOrigin, vec3 rayDir)
{
    float t = hitSphere(sphereCenter, sphereRadius, rayOrigin, rayDir, 0.0001, INF);
    if (t > 0.0)
    {
        vec3 normal = normalize(rayOrigin + rayDir * t - sphereCenter);
        return 0.5 * (normal + vec3(1.0));
    }
    
    vec3 unitDir = normalize(rayDir);
    float a = 0.5 * (unitDir.y + 1.0);
    return (1.0 - a) * vec3(1.0) + a * vec3(0.5, 0.7, 1.0); 
}

void main()
{
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imageSize = imageSize(outputImage);
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y) return;

    float u = float(pixelCoord.x) / float(imageSize.x);
    float v = float(pixelCoord.y) / float(imageSize.y);

    vec3 rayDir = camLowerLeftCorner + u * camHorizontal + v * camVertical - camOrigin;
    vec3 color = rayColor(camOrigin, rayDir);

    imageStore(outputImage, pixelCoord, vec4(color, 1.0));
}
```

</details>

同样在`main.cpp`里面增添相机的参数设置，传入 uniform

<details>
<summary> main.cpp 更新</summary>

```cpp
raytraceProgram.use();
glBindImageTexture(0, outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

// 相机参数，和RTIOW里camera构造函数的计算逻辑一致
float aspectRatio = float(SCR_WIDTH) / float(SCR_HEIGHT);
float viewportHeight = 2.0f;
float viewportWidth = aspectRatio * viewportHeight;
float focalLength = 1.0f;

raytraceProgram.setVec3("camOrigin", 0, 0, 0);
raytraceProgram.setVec3("camHorizontal", viewportWidth, 0, 0);
raytraceProgram.setVec3("camVertical", 0, viewportHeight, 0);
raytraceProgram.setVec3("camLowerLeftCorner",
    -viewportWidth/2, -viewportHeight/2, -focalLength);

raytraceProgram.setVec3("sphereCenter", 0, 0, -1);
raytraceProgram.setFloat("sphereRadius", 0.5f);

GLuint groupsX = (SCR_WIDTH + 15) / 16;
GLuint groupsY = (SCR_HEIGHT + 15) / 16;
glDispatchCompute(groupsX, groupsY, 1);
glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
```
</details>

这里我们用经典的办法让法线表示颜色，输出一个球。如果一切顺利你将得到下面的输出：

<p align="center">
  <img src="/markdown_picture/ComputeShader/sphere.png" width="300">
</p>
<p align="center">
  第一彩色球！
</p>

---
# BVH
注：下面真的就是面向我自己的笔记了，主要是代码量大，架构复杂（要把 RTINW 的部分代码移植过来），导致很难写清楚每一步。
## 大致流程

我们都知道，在CPU上每个线程都有自己独立的、几MB大的调用栈，完全可以容纳递归，但是GPU就不同。  
在GPU中，一个workgroup里成千上万个线程是"锁步"执行的（同一时刻尽量跑同一条指令），每个线程分到的栈空间极小（几百字节到几KB级别），而且大部分GLSL编译器直接不支持递归函数调用，因为编译器需要在编译时确定每个线程占用多少寄存器/栈空间，递归的深度在编译期是不确定的。  
所以唯一的解决办法就是：把递归改成**用一个显式数组模拟栈的循环**，为什么是数组，因为GPU用不了指针。  

首先来顺一遍流程，从程序开始，**CPU和GPU做了什么**。（重点放在bvh构建上）  

#### 初始化 OpenGL 环境
创建GLFW窗口，加载函数指针，生成空VAO，创建一张2D纹理作为compute shader的输出目标
#### 构建场景（C++ 传统 OOP 方式）
用 `shared_ptr<hittable>` 构建一个场景世界。
#### CPU 端递归构建 BVH 树
调用 `bvh_node` 构造函数，递归分割、每个节点计算好自己的 AABB，形成一棵树
#### 拍平*
将递归树翻译成 GPU 可用的连续数组，拍平过后得到两个 `vector`：（这里的变量名不代表最后的变量名，先理解即可）
- `flatNodes`：`vector<GPUBVHNode>`，每个元素代表一个树节点（内部/叶子），通过整数索引引用。
- `flatSpheres`：`vector<GPUSphere>`，所有球体数据紧密排列，叶子节点的 `rightChild` 指向这里。
#### 上传数据到 GPU
创建两个 SSBO  
- BVH SSBO (binding = 1)：
  - `glBufferData` 将 `flatNodes` 二进制拷贝到显存。

- Sphere SSBO (binding = 2)：
    - `glBufferData` 将 `flatSpheres` 二进制拷贝到显存。
- 设置 Uniform，绑定输出纹理

#### GPU 渲染每一帧
派发线程，每个线程进行 BVH 求交，着色，写入纹理等等

## 对 CPU 端进行 BVH 重构
BVH 重构可谓是相当有挑战的一部分，下面我们的工作核心是：**将面向对象的递归场景树，彻底拍平为 GPU 可用的扁平数组，同时消除运行时多态。**

下面只给出了核心函数与文件（有较为详细注释），具体的C++类的改写还得自己来。这一步暂时不管材质和动态模糊（我真的不喜欢动态模糊，所以我把抽象类的接口以及球的构造函数改成了无动态模糊的类型），可以直接从 RTINW 迁移过来的Cpp文件有：aabb.h, bvh.h, hittable_list.h, hittable.h, rtweekend.h,  sphere.h, vec3.h  
- vec3.h 还是先保留，我们在最后的bvh_h里面拍平树的时候再用glm就是，不是很影响。
- ray.h为何没有迁移？因为光线都是在GPU里面算好的，C++端基本不需要了，最多只是填参数而已，这里我选择把ray相关的函数参数都改成 rayOrigin和rayDir.
- sphere.h基本上就退化为了创建场景时有用，创建一个球的类型
- rtweekend.h基本没用，因为随机函数后面用 GPU随机数生成器
- hittable_list.h里面hittable_list类的hit函数直接架空，因为我们会在GPU端实现。
- aabb.h计算包围盒与光线交点的hit函数直接架空，我们在GPU里实现。

<details>
<summary>bvh.h完整文件</summary>

```cpp
#ifndef BVH_H
#define BVH_H


#include "hittable_list.h"
#include "sphere.h"

#include <algorithm>
#include <glm/glm.hpp>

// bvh_node 类是RTINW自带的，基本没做任何修改
class bvh_node : public hittable
{
    public:
    bvh_node() {}

    bvh_node(std::vector<shared_ptr<hittable>>& objects, size_t start, size_t end)
    {
        int axis = random_int(0, 2);

        auto comparator = (axis == 0) ? box_x_compare
                    : (axis == 1) ? box_y_compare
                                  : box_z_compare;


        size_t object_span = end - start;   // start 和 end 是区间下标 [start, end)，object_span 即为这个节点应该处理的物体个数。

        if (object_span == 1)
        {
            left = right = objects[start];
        }
        else if (object_span == 2)
        {
            left = objects[start];
            right = objects[start + 1];
        }
        else
        {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);

            auto mid = start + object_span / 2;
            // 递归构建树
            left = make_shared<bvh_node>(objects, start, mid);
            right = make_shared<bvh_node>(objects, mid, end);
        }

        aabb box_left, box_right;

        if (  !left->bounding_box (box_left) || !right->bounding_box(box_right))
            std::cerr << "No bounding box in bvh_node constructor.\n";

        bbox = surrounding_box(box_left, box_right);
    }

    // 遍历左右子树求交取最近的交点————现在在 GPU 实现
    bool hit(const vec3& ray_origin, const vec3& ray_dir, double t_min, double t_max, hit_record& rec) const override 
    {
        /*
        if (!bbox.hit(ray_origin, ray_dir, t_min, t_max))
            return false;
        bool hit_left = left->hit(r, t_min, t_max, rec);
        bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);
        return hit_left || hit_right;
        */

        return false;
    }

    bool bounding_box(aabb& output_box) const override 
    { 
        output_box = bbox;
        return true; 
    }

    shared_ptr<hittable> left;
    shared_ptr<hittable> right;
    aabb bbox;

    static bool box_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis_index)
    {
        aabb box_a;
        aabb box_b;

        if (!a->bounding_box(box_a) || !b->bounding_box(box_b))
        std::cerr << "No bounding box in bvh_node constructor.\n";

        return box_a.min().e[axis_index] < box_b.min().e[axis_index];
    }

    static bool box_x_compare (const shared_ptr<hittable> a, const shared_ptr<hittable> b)
    {
        return box_compare(a, b, 0);
    }

    static bool box_y_compare (const shared_ptr<hittable> a, const shared_ptr<hittable> b) 
    {
        return box_compare(a, b, 1);
    }

    static bool box_z_compare (const shared_ptr<hittable> a, const shared_ptr<hittable> b) 
    {
        return box_compare(a, b, 2);
    }
};

//======================================================================

struct alignas(16) GPUBVHNode   // 强制整个结构体的对齐方式为 16 字节边界
{
    glm::vec3 aabbMin;          // 包围盒的最小顶点坐标
    int leftChild;              // 内部节点填左孩子下标，叶子填-1
                                // 这个int紧跟在vec3后面，会被塞进vec3的16字节对齐槽里，不会额外占空间

    glm::vec3 aabbMax;
    int rightChild;             // 内部节点填左孩子下标，叶子填图元下标

    int isLeaf;                 // 1 叶子，0 内部节点
    float pad0, pad1, pad2;     // 凑齐16字节，避免下一个node因为对齐产生偏移
};
// 所以最终一个 bvhnode占用48字节
// 为何在CPU端要这样设计节点？
// std430规则下vec3本身要按16字节对齐，正好剩一个4字节空当，编译器会自动把紧跟着的float塞进去，不会浪费也不会错位，这是个很常用的省内存技巧。

struct GPUSphere 
{
    glm::vec3 center;
    float radius;

    int materialId; // Day26要用，先占位填0
    float pad0, pad1, pad2;
};

class BVHFlattener
{
    public:
        std::vector<GPUBVHNode> flatNodes;
        std::vector<GPUSphere> flatSpheres;

        // 入口：传入根节点，返回根节点在flatNodes里的下标
        int flatten(shared_ptr<hittable> root)
        {
            return flattenNode(root);
        }

    private:
        int flattenNode(shared_ptr<hittable> node)
        {
            auto asBVH = std::dynamic_pointer_cast<bvh_node>(node);

            if (asBVH)
            {
                // 如果是内部节点，先递归拍平左右孩子，得到他们在数组中的索引
                int leftIdx = flattenNode(asBVH->left);
                int rightIdx = flattenNode(asBVH->right);

                aabb box = asBVH->bbox;
                GPUBVHNode gpuNode;
                gpuNode.aabbMin = glm::vec3(box.min().x(), box.min().y(), box.min().z());
                gpuNode.aabbMax = glm::vec3(box.max().x(), box.max().y(), box.max().z());
                gpuNode.leftChild = leftIdx;
                gpuNode.rightChild = rightIdx;
                gpuNode.isLeaf = 0;

                flatNodes.push_back(gpuNode);
                return (int)flatNodes.size() - 1;
            }
            else
            {
                // 是叶子：node直接是一个球
                auto s = std::dynamic_pointer_cast<sphere>(node);
                GPUSphere gpuSphere;
                gpuSphere.center = glm::vec3(s->center.x(), s->center.y(), s->center.z());
                gpuSphere.radius = s->radius;
                gpuSphere.materialId = 0;
                flatSpheres.push_back(gpuSphere);
                int sphereIdx = (int)flatSpheres.size() - 1;

                aabb box;
                node->bounding_box(box);
                GPUBVHNode gpuNode;
                gpuNode.aabbMin = glm::vec3(box.min().x(), box.min().y(), box.min().z());
                gpuNode.aabbMax = glm::vec3(box.max().x(), box.max().y(), box.max().z());
                gpuNode.leftChild = -1;
                gpuNode.rightChild = sphereIdx; // leaf节点复用rightChild字段存图元下标
                gpuNode.isLeaf = 1;

                flatNodes.push_back(gpuNode);
                return (int)flatNodes.size() - 1;
            }
        }
};

#endif
```

</details>

特别注意里面新类`BVHFlattener`，这个就是在 **CPU端拍平树的关键！** 还应注意的是C++里`GPUSphere`、`BVHFlattener` **数据结构的定义**

<details>
<summary>raytrace.comp完整文件</summary>

```glsl
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32f, binding = 0) uniform image2D outputImage;

const float INF = uintBitsToFloat(0x7F800000u);

// 相机参数
uniform vec3 camOrigin;
uniform vec3 camLowerLeftCorner;
uniform vec3 camHorizontal;
uniform vec3 camVertical;

// bvh 根节点索引
uniform int bvhRootIndex;

struct BVHNode 
{
    vec3 aabbMin;
    int leftChild;
    vec3 aabbMax;
    int rightChild;
    int isLeaf;
    float pad0, pad1, pad2;
};

struct Sphere 
{
    vec3 center;
    float radius;
    int materialId;
    float pad0, pad1, pad2;
};

// 声明着色器存储缓冲区SSBO
layout(std430, binding = 1) readonly buffer BVHBuffer {BVHNode nodes[];};
layout(std430, binding = 2) readonly buffer SphereBuffer {Sphere spheres[];};

// 光线与球体求交，返回命中距离，没命中返回-1.0
float hitSphere(vec3 center, float radius, vec3 rayOrigin, vec3 rayDir, float t_min, float t_max)
{
    vec3 oc = rayOrigin - center;
    float a = dot(rayDir, rayDir);
    float halfB = dot(oc, rayDir);
    float c = dot(oc, oc) - radius * radius;

    float discriminant = halfB * halfB - a * c;

    if (discriminant < 0.0) return -1.0;

    float root = (-halfB - sqrt(discriminant)) / a;
    if (!(root > t_min && root < t_max))
    {
        root = (-halfB + sqrt(discriminant)) / a;
        if (!(root > t_min && root < t_max))
            return -1.0;
    }
    return root;
}

// 包围盒求交，slab方法
bool aabbHit(vec3 boxMin, vec3 boxMax, vec3 rayOrigin, vec3 rayDir, float tMax)
{
    float tmin = 0.0;      // 光线起点内部视为 0
    float tmax = tMax;

    for (int i = 0; i < 3; ++i) 
    {
        float invD = 1.0 / rayDir[i];
        float t0 = (boxMin[i] - rayOrigin[i]) * invD;
        float t1 = (boxMax[i] - rayOrigin[i]) * invD;
        if (invD < 0.0) 
        {   // 确保 t0 是近面，t1 是远面
            float tmp = t0; t0 = t1; t1 = tmp;
        }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
        if (tmax <= tmin)
            return false;
    }
    return true;
}

// BVH遍历器，参数输出命中球的索引以及命中距离
bool traverseBVH(vec3 rayOrigin, vec3 rayDir, out int hitSphereIdx, out float hitT)
{
    int stack[32];
    int stackPtr = 0;
    stack[stackPtr++] = bvhRootIndex; // 根节点

    hitT = INF;
    bool hitAnything = false;

    while (stackPtr > 0)
    {
        int nodeIdx = stack[--stackPtr];
        BVHNode node = nodes[nodeIdx];

        if (!aabbHit(node.aabbMin, node.aabbMax, rayOrigin, rayDir, hitT))
            continue;
        
        if (node.isLeaf == 1)
        {
            int sIdx = node.rightChild; // leaf节点里rightChild存的是图元下标
            float t = hitSphere(spheres[sIdx].center, spheres[sIdx].radius, rayOrigin, rayDir, 0.0001, hitT);
            if (t != -1.0)
            {
                hitT = t;
                hitSphereIdx = sIdx;
                hitAnything = true;
            }
        }
        else
        {
            stack[stackPtr++] = node.leftChild;
            stack[stackPtr++] = node.rightChild;
        }
    }
    return hitAnything;
}


vec3 rayColor(vec3 rayOrigin, vec3 rayDir)
{
    int hitSphereIdx;
    float t;
    bool hit = traverseBVH(rayOrigin, rayDir, hitSphereIdx, t);

    if (hit)
    {
        Sphere s = spheres[hitSphereIdx];
        vec3 hitPoint = rayOrigin + rayDir * t;
        vec3 normal = normalize(hitPoint - s.center);

        return 0.5 * (normal + vec3(1.0));
    }
    
    vec3 unitDir = normalize(rayDir);
    float a = 0.5 * (unitDir.y + 1.0);
    return (1.0 - a) * vec3(1.0) + a * vec3(0.5, 0.7, 1.0); 
}



void main()
{
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imageSize = imageSize(outputImage);
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y) return;

    float u = float(pixelCoord.x) / float(imageSize.x);
    float v = float(pixelCoord.y) / float(imageSize.y);

    vec3 rayDir = normalize(camLowerLeftCorner + u * camHorizontal + v * camVertical - camOrigin);
    vec3 color = rayColor(camOrigin, rayDir);

    imageStore(outputImage, pixelCoord, vec4(color, 1.0));
}
```
</details>

我们在GPU里完成对bvh节点数组的遍历，对aabb包围盒的求交。

为了使main不臃肿，新建一个gpu_buffer.h，来创建 SSBO，然后再创建一个 scene.h 来构建场景。

<details>
<summary> gpu_buffer.h完整文件</summary>

```cpp
#ifndef GPU_BUFFER_H
#define GPU_BUFFER_H

#include <glad/glad.h>
#include "bvh.h"
#include <vector>

GLuint createBVHSSBO(const std::vector<GPUBVHNode>& nodes)
{
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, nodes.size() * sizeof(GPUBVHNode), nodes.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo);// 注意这里对应的binding序号
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return ssbo;
}

GLuint createShereSSBO(const std::vector<GPUSphere>& spheres)
{
    
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, spheres.size() * sizeof(GPUSphere), spheres.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return ssbo;
}
#endif
```

</details>


<details>
<summary>scene.h完整文件 </summary>

```cpp
#ifndef SCENE_H
#define SCENE_H

#include "bvh.h"

hittable_list createScene()
{
    hittable_list world;


    world.add(
        make_shared<sphere>(
            vec3(0,0,-1),
            0.5f
        )
    );


    world.add(
        make_shared<sphere>(
            vec3(1,0,-2),
            0.5f
        )
    );


    world.add(
        make_shared<sphere>(
            vec3(-1,0,-2),
            0.5f
        )
    );


    return world;
}

#endif
```

</details>


<details>
<summary>main.cpp</summary>

```cpp
// main函数里
    Shader raytraceProgram = Shader::computeShader("src/shaders/raytrace.comp");
    Shader quadProgram = Shader::graphicsShader("src/shaders/quad.vert", "src/shaders/quad.frag");

    //----------- 场景构建---------------
    hittable_list world = createScene();

    //----------- 构建BVH ---------------
    auto bvh = make_shared<bvh_node>(world.objects, 0, world.objects.size());
    //----------- flatten --------------
    BVHFlattener flattener;
    int rootIndex = flattener.flatten(bvh);
    //----------- 上传 GPU --------------
    static_assert(sizeof(GPUBVHNode) == 48, "GPUBVHNode must be 48 bytes");
    static_assert(sizeof(GPUSphere)  == 32, "GPUSphere must be 32 bytes");
    createBVHSSBO(flattener.flatNodes);
    createShereSSBO(flattener.flatSpheres);
```

</details>

正常渲染结果应该是屏幕中有三个球，都是紫色渐变。


> [!tip]
**在GPU内编码少用else-if分支：线程束发散**，GPU 以 warp（通常 32 个线程）为单位执行指令。当同一 warp 内的线程因 if-else 走入不同分支时，它们只能串行执行，这直接导致并行效率减半甚至更低，且分支内部计算越重，浪费越严重。

>

# GPU 的随机数生成
## GPU 随机数生成器：PCG哈希
C++的`std::mt19937`是一个有状态的对象，每次调用会修改内部状态、下次调用产生不同的数——这个模型在GPU上完全不适用，因为几千个线程如果共享一个"状态对象"会产生数据竞争，而每个线程各自维护一个独立的mt19937实例又太重（占用寄存器多，性能差）。
GPU上标准做法是用哈希函数模拟随机数：给定一个整数种子，哈希出另一个"看起来随机"的整数，没有全局状态，纯函数。PCG是业界最常用的一种，速度快、分布质量好。

在 `raytrace.comp` 里加：

```glsl
// PCG哈希，输入一个整数种子，输出另一个"随机"整数
uint pcgHash(uint input_)
{
    uint state = input_ * 747796405u + 2891336453u;                         // 线性同余生成器公式，新状态 = 旧状态 × 大奇数 + 另一个大奇数，两个奇数是PCG 作者精心挑选的“魔法数字”
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;    // 旋转+乘法，用数据本身的几个高位去决定怎样搅动剩下的位
    return (word >> 22u) ^ word;
}

// 每个线程调用后，rngState会被更新，下次调用产生不同的数
float randFloat(inout uint rngState)
{
    rngState = pcgHash(rngState);           // 更新状态
    return float(rngState) / 4294967296.0;  // 除以 2^32，映射到 [0,1)
}

// 给每个线程发独一无二的种子
uint initRNG(ivec2 pixelCoord, ivec2 imgSize, uint frameCount) 
{
    uint seed = uint(pixelCoord.x) + uint(pixelCoord.y) * uint(imgSize.x);  // 将二维像素坐标展开成一维索引seed
    seed = seed * 719393u + frameCount * 96923u;                            // 用一个大奇数放大，免相邻种子在经过 pcgHash 后产生的序列仍有微弱关联。引入帧数，让同一像素在不同帧的种子完全不同
    return pcgHash(seed);
}
```

**为什么要结合frameCount**：如果只用像素坐标做种子，同一个像素每一帧都会用完全一样的种子，产生一模一样的随机序列——这样多帧累积的时候，噪点不会被平均掉，因为每帧的"随机"抖动其实都相同。

再加两个基于`randFloat`的辅助函数，对应RTIOW里`random_in_unit_sphere()`和`random_unit_vector()`：

<details>
<summary>两个重要的随机采样函数</summary>

```glsl
// 对应RTIOW的random_unit_vector()，用解析公式直接生成单位球面上均匀分布的点
vec3 randomUnitVector(inout uint rngState) 
{
    float z = randFloat(rngState) * 2.0 - 1.0;
    float a = randFloat(rngState) * 2.0 * PI;
    float r = sqrt(max(0.0, 1.0 - z * z));
    return vec3(r * cos(a), r * sin(a), z);
}

// 对应RTIOW的random_in_unit_disk()，用于metal的fuzz模糊反射
vec2 randomInUnitDisk(inout uint rngState) 
{
    float a = randFloat(rngState) * 2.0 * PI;
    float r = sqrt(randFloat(rngState));
    return vec2(r * cos(a), r * sin(a));
}

```
</details>

## 材质系统的移植
### 扩展BVHFlattener，拍平材质
CPP的`material.h`包含`lambertian`/`metal`/`dielectric`三个类，都继承`material`基类，靠虚函数`scatter()`实现多态。GPU没有虚函数，处理方式只能是**打上类型标签，塞进一个统一的struct。**同时 `scatter()` 统一在 GPU 里完成

在`bvh.h`里添加（这里因为我的`GPUShere`和`GPUBVHNode`在这个文件，我也只好把材质的结构体也加在此处）

<details>
<summary>bvh.h</summary>

```cpp
// 和GLSL端严格对应，注意字段顺序影响std430对齐
struct GPUMaterial {
    glm::vec3 albedo;  // lambertian/metal用，颜色
    float fuzz;        // metal专用，紧跟在albedo后面，卡进vec3的对齐空当
    float ir;          // dielectric专用，折射率
    int type;           // 0=lambertian, 1=metal, 2=dielectric
    float pad0, pad1;  // 凑齐32字节(16的倍数)
};
static_assert(sizeof(GPUMaterial) == 32, "GPUMaterial size mismatch, check alignment");

...

class BVHFlattener {
public:
    std::vector<GPUBVHNode> flatNodes;
    std::vector<GPUSphere> flatSpheres;
    std::vector<GPUMaterial> flatMaterials; // 新增

    int flatten(shared_ptr<hittable> root) {
        return flattenNode(root);
    }

private:
    // 材质去重：同一个material对象可能被多个球共用，避免重复存储
    std::unordered_map<material*, int> materialCache;

    int flattenMaterial(shared_ptr<material> mat) {
        auto it = materialCache.find(mat.get());
        if (it != materialCache.end()) return it->second; // 已经拍平过，直接复用下标

        GPUMaterial gpuMat{};
        if (auto lamb = std::dynamic_pointer_cast<lambertian>(mat)) {
            gpuMat.type = 0;
            gpuMat.albedo = glm::vec3(lamb->get_albedo().x(), lamb->get_albedo().y(), lamb->get_albedo().z());
        } else if (auto met = std::dynamic_pointer_cast<metal>(mat)) {
            gpuMat.type = 1;
            gpuMat.albedo = glm::vec3(met->get_albedo().x(), met->get_albedo().y(), met->get_albedo().z());
            gpuMat.fuzz = (float)met->get_fuzz();
        } else if (auto diel = std::dynamic_pointer_cast<dielectric>(mat)) {
            gpuMat.type = 2;
            gpuMat.ir = (float)diel->get_ir();
        }

        flatMaterials.push_back(gpuMat);
        int idx = (int)flatMaterials.size() - 1;
        materialCache[mat.get()] = idx;
        return idx;
    }

    int flattenNode(shared_ptr<hittable> node) {
        auto asBVH = std::dynamic_pointer_cast<bvh_node>(node);

        if (asBVH) {
            int leftIdx  = flattenNode(asBVH->get_left());
            int rightIdx = flattenNode(asBVH->get_right());

            aabb box = asBVH->get_box();
            GPUBVHNode gpuNode{};
            gpuNode.aabbMin = glm::vec3(box.min().x(), box.min().y(), box.min().z());
            gpuNode.aabbMax = glm::vec3(box.max().x(), box.max().y(), box.max().z());
            gpuNode.leftChild = leftIdx;
            gpuNode.rightChild = rightIdx;
            gpuNode.isLeaf = 0;

            flatNodes.push_back(gpuNode);
            return (int)flatNodes.size() - 1;
        } else {
            auto s = std::dynamic_pointer_cast<sphere>(node);

            GPUSphere gpuSphere{};
            gpuSphere.center = glm::vec3(s->center.x(), s->center.y(), s->center.z());
            gpuSphere.radius = (float)s->radius;
            gpuSphere.materialId = flattenMaterial(s->get_material()); // 关键：把材质也拍平，记录下标

            flatSpheres.push_back(gpuSphere);
            int sphereIdx = (int)flatSpheres.size() - 1;

            aabb box;
            node->bounding_box(0, 0, box);
            GPUBVHNode gpuNode{};
            gpuNode.aabbMin = glm::vec3(box.min().x(), box.min().y(), box.min().z());
            gpuNode.aabbMax = glm::vec3(box.max().x(), box.max().y(), box.max().z());
            gpuNode.leftChild = -1;
            gpuNode.rightChild = sphereIdx;
            gpuNode.isLeaf = 1;

            flatNodes.push_back(gpuNode);
            return (int)flatNodes.size() - 1;
        }
    }
};
```
</details>

然后在 `gpu_buffer.h` 里传 SSBO 即可，这里注意是 `binding = 3`

### GLSL 端完善 hit record

之前的`hitSphere`只返回一个距离`t`，现在散射需要交点坐标、法线方向、材质id，得升级成完整的"hit record"，traverseBVH也得跟着更新

<details>
<summary>raytrace.comp</summary>

```glsl
struct HitRecord
{
    vec3 point;
    vec3 normal;
    float t;
    int materialId;
    bool frontFace; // 折射时用到，光线从内部还是外部打到球
};

struct Material
{
    glm::vec3 albedo;   // lambertian/metal 用颜色
    float fuzz;         // metal专用，紧跟在albedo后面，卡进vec3的对齐空当
    float ir;           // dielectric专用，折射率
    int type;           // 0=lambertian, 1=metal, 2=dielectric
    float pad0, pad1;
};

layout(std430, binding = 3) readonly buffer MaterialBuffer {Material materials[];};

// ------------------- 核心计算：光线求交----------------------
// 光线与球体求交
bool hitSphere(Sphere s, vec3 rayOrigin, vec3 rayDir, float t_min, float t_max, out HitRecord rec)
{
    vec3 oc = rayOrigin - s.center;
    float a = dot(rayDir, rayDir);
    float halfB = dot(oc, rayDir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = halfB * halfB - a * c;
    if (discriminant < 0.0) return false;

    float root = (-halfB - sqrt(discriminant)) / a;
    if (!(root > t_min && root < t_max))
    {
        root = (-halfB + sqrt(discriminant)) / a;
        if (!(root > t_min && root < t_max))
            return false;
    }

    rec.t = root;
    rec.point = rayOrigin + root * rayDir;
    vec3 outwardNormal = (rec.point - s.center) / s.radius;
    rec.frontFace = dot(rayDir, outwardNormal) < 0.0;
    rec.normal = rec.frontFace ? outwardNormal : -outwardNormal;
    rec.materialId = s.materialId;
    return true;
}

// BVH遍历器，参数输出命中球的索引以及命中距离
bool traverseBVH(vec3 rayOrigin, vec3 rayDir, float tMin, float tMax, out HitRecord rec)
{
    int stack[32];
    int stackPtr = 0;
    stack[stackPtr++] = bvhRootIndex; 

    HitRecord tempRec;
    float closestSoFar = tMax;
    bool hitAnything = false;

    while (stackPtr > 0)
    {
        int nodeIdx = stack[--stackPtr];
        BVHNode node = nodes[nodeIdx];

        if (!aabbHit(node.aabbMin, node.aabbMax, rayOrigin, rayDir, closestSoFar))
            continue;
        
        if (node.isLeaf == 1)
        {
            int sIdx = node.rightChild;
            if (hitSphere(spheres[sIdx], rayOrigin, rayDir, tMin, closestSoFar, tempRec))
            {
                closestSoFar = tempRec.t;
                rec = tempRec;
                hitAnything = true;
            }
        }
        else
        {
            stack[stackPtr++] = node.leftChild;
            stack[stackPtr++] = node.rightChild;
        }
    }
    return hitAnything;
}
```
</details>

### 材质散射函数

在 `raytrace.comp` 添加三种材质的散射函数

<details>
<summary>scatter函数</summary>

```glsl
// ------------------------- 散射 --------------------------------
// 返回值：是否发生散射(dielectric/lambertian/metal理论上总会散射，
// 但metal在极端角度下反射方向可能指向物体内部，这时候算作被吸收)
// scatteredDir: 散射后的新射线方向
// attenuation: 这次散射的反射率，为什么是albedo？还剩下的颜色可以理解为还剩的能量，满能量就是白光vec3(1.0)
bool scatter(HitRecord rec, vec3 rayDir, inout uint rngState, out vec3 scatteredDir, out vec3 attenuation)
{
    Material mat = materials[rec.materialId];

    if (mat.type == 0)
    {
        // ---- Lambertian ----
        vec3 scatterDir = rec.normal + randomUnitVector(rngState);
        // 退化情况处理：如果随机方向正好和法线反向抵消，方向会变成接近0的向量
        if (length(scatterDir) < 1e-4) scatterDir = rec.normal;
        scatteredDir = normalize(scatterDir);
        attenuation = mat.albedo;
        return true;
    }
    else if (mat.type == 1)
    {
        // ---- Metal ----
        vec3 reflected = reflect(normalize(rayDir), rec.normal);
        vec2 fuzzOffset = randomInUnitDisk(rngState) * mat.fuzz;
        scatteredDir = normalize(reflected + vec3(fuzzOffset, 0.0));
        attenuation = mat.albedo;
        return dot(scatteredDir, rec.normal) > 0.0; // 模糊后如果方向钻进物体内部,视为被吸收
    }
    else
    {
        // ---- Dielectric(玻璃) ----
        attenuation = vec3(1.0); // 玻璃不吸收颜色
        float refractionRatio = rec.frontFace ? (1.0 / mat.ir) : mat.ir;

        vec3 unitDir = normalize(rayDir);
        float cosTheta = min(dot(-unitDir, rec.normal), 1.0);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = refractionRatio * sinTheta > 1.0;

        // Schlick近似，对应RTIOW的reflectance()函数
        float r0 = (1.0 - refractionRatio) / (1.0 + refractionRatio);
        r0 = r0 * r0;
        float reflectance = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);

        if (cannotRefract || reflectance > randFloat(rngState)) 
        {
            scatteredDir = reflect(unitDir, rec.normal);
        } else 
        {
            scatteredDir = refract(unitDir, rec.normal, refractionRatio);
        }
        return true;
    }
}
```
</details>

### 主循环：多次反弹的迭代路径追踪

`ray_color()`改成固定次数的for循环，一旦没命中或者达到最大深度就提前跳出：

<details>
</summary>ray_color()与main()</summary>

```glsl
// ----------------------- Draw! ---------------------------
vec3 rayColor(vec3 rayOrigin, vec3 rayDir, inout uint rngState)
{
    vec3 attenuation = vec3(1.0);
    int maxDepth = 16;
    
    for (int depth = 0; depth < maxDepth; depth++)
    {
        HitRecord rec;
        if (traverseBVH(rayOrigin, rayDir, 0.0001, INF, rec))
        {
            vec3 scatteredDir;
            vec3 matAttenuation;
            if (scatter(rec, rayDir, rngState, scatteredDir, matAttenuation))
            {
                attenuation *= matAttenuation;
                rayOrigin = rec.point;
                rayDir = scatteredDir;
            }
            else
            {
                return vec3(0.0);
            }
        }
        else
        {
            vec3 unitDir = normalize(rayDir);
            float a = 0.5 * (unitDir.y + 1.0);
            vec3 skyColor = (1.0 - a) * vec3(1.0) + a * vec3(0.5, 0.7, 1.0); 
            return attenuation * skyColor;
        }
    }
    return vec3(0.0); // 超过最大深度还没跳出，视为完全吸收
}


//-------------------- main ----------------------
void main()
{
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imageSize = imageSize(outputImage);
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y) return;

    uint rngState = initRNG(pixelCoord, imageSize, frameCount);

    float u = (float(pixelCoord.x) + randFloat(rngState)) / float(imageSize.x);     // 随机偏移做抗锯齿
    float v = (float(pixelCoord.y) + randFloat(rngState)) / float(imageSize.y);

    vec3 rayDir = normalize(camLowerLeftCorner + u * camHorizontal + v * camVertical - camOrigin);
    vec3 color = rayColor(camOrigin, rayDir, rngState);

    imageStore(outputImage, pixelCoord, vec4(color, 1.0));
}
```

</details>

最后在CPU端`main.cpp`主循环里加一个frameCount计数器：

```
uint32_t frameCount = 0;
// ...主循环内，dispatch之前
raytraceProgram.setInt("frameCount", (int)frameCount);
frameCount++;
```