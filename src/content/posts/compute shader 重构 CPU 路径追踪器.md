---
title: Compute Shader 重构 CPU 路径追踪器
published: 2026-07-18
pinned: true
description: 前段时间我按照 RTIOW 和 RTINW 搭建了一个 CPU 路径追踪器，实际渲染速度非常慢，于是我打算用 Compute Shader 重构一下。这其中便会涉及到从 CPU 编程到 GPU 编程的思维转变。所以我打算以此写一篇偏个人向的笔记，记录一下重构过程中新的知识以及一些经验技巧。
tags: [OpenGL, Compte Shader, C++, RayTracer]
category: 技术
draft: false
---

> [!NOTE]
阅读前，请确保你有一定的 OpenGL 和路径追踪基础  
为了节省空间，本文的代码均折叠了起来，需要你手动展开。同时，每一个代码块都已经写好了详细的注释，方便逐行理解。   
由于我也是第一次接触 Comepute Shader，我会**从最开始写起**。为了回忆之前所学，我在一些基础的 OpenGL 内容后面也给了详细解释。
>

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
不用建VAO/VBO传顶点数据，靠 `gl_VertexID`（0,1,2）直接在shader里算出一个覆盖全屏的大三角形。你只需要在CPU端调用 `glDrawArrays(GL_TRIANGLES, 0, 3)`，不用绑定任何顶点缓冲。
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
  <img src="/markdown_picture/ComputeShader/HelloComp.png" width="500">
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

# 单球光追

新建 `src/shaders/raytrace.comp`，把RTIOW的相机射线生成 + 球体求交搬过来：

