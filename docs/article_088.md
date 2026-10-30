

# 第88章 测试体系全览 —— 电影特效工作室的质检防线

## 学习目标

-   理解 Storybook 测试夹具的设计原则：如何通过独立的 Markdown 内容文件模拟真实 LLM 输出场景
-   掌握 Vitest 与 Storybook 的集成机制：setProjectAnnotations 如何合并预览配置与 a11y 插件
-   学会构建类型安全的测试 Mock：通过 Partial\<T> 模式灵活覆盖 Store 状态而不破坏类型约束
-   解析 Agentic 消息结构的测试策略：从原始 DatabaseMessage 到结构化 Section 的转换验证
-   理解剪贴板数据的序列化与反序列化：JSON 结构化格式与纯文本模式的边界处理
-   掌握 LaTeX 与货币符号的区分算法：基于启发式规则的文本预处理与保护机制
-   学习模型 ID 解析与 URI 模板处理：从字符串解析到资源定位的完整链路
-   了解前端测试金字塔：单元测试、组件测试与 E2E 测试的分层协作

## 88.1 生活类比：电影特效工作室的"质检部门"

想象 llama.cpp WebUI 的测试体系是一家**电影特效工作室的质检部门**。在这个工作室里，每一帧画面都经过严密的质检才能上映。Markdown 夹具就像**特效镜头剧本库**，里面存放着 AI 教程、API 文档、博客文章等不同风格的测试剧本，每份剧本都为检验特定渲染效果而精心设计。Storybook 环境则是**摄影棚灯光调试间**，确保不同布景下演员肤色一致——也就是让单元测试环境与 Storybook 预览环境的配置保持统一。

Mock 工具函数如同**替身演员与绿幕**，它们让测试无需真实服务器即可模拟各种模态场景（纯视觉、纯音频、双模态、无模态），就像演员在绿幕前表演，后期再合成背景。Agentic 消息测试则是**剧本结构分析**，把对话解构成推理、工具调用、文本段落等标准化镜头，确保每场戏的结构清晰可控。剪贴板格式化功能宛如**海关安检**，确保数据进出时格式合规，无论是 JSON 结构化格式还是纯文本模式都能正确转换。LaTeX 保护机制则是**货币防伪检测**，需要区分 $5 美元符号与 $x$ 数学符号——看似相似却含义迥异，防止误伤正常文本。模型 ID 解析器像极了**身份证识别器**，从 `meta-llama/Llama-3.1-8B-Q4_K_M` 中精确提取组织、参数量、精度等关键信息。URI 模板处理则是**资源调度中心**，将 `{/id}` 这样的占位符展开为具体的 API 路径，并修复展开过程中产生的 `///` 异常路径。推理上下文过滤宛如**剧本审校员**，确保"内心独白"（推理过程）不会混入对外展示的台词，保护用户隐私。测试金字塔则构成了完整的**质检流水线**：单元测试是零件检查，组件测试是整车测试，E2E 测试是最终路测。

就像特效部门需要为每一场戏准备精确的剧本和替身，前端测试也需要为每一种 UI 状态准备真实的 Markdown 内容与 Mock 数据，确保最终渲染效果万无一失。

## 88.2 源码地图

```
tools/server/webui/tests/stories/fixtures/
├── ai-tutorial.ts
│   └── AI_TUTORIAL_MD               # SvelteKit+WebSocket 教程 Markdown，含代码块/表格/env（1-145行）
├── api-docs.ts
│   └── API_DOCS_MD                  # REST API 文档，含 HTTP 示例/错误格式/SDK 代码（1-139行）
├── blog-post.ts
│   └── BLOG_POST_MD                 # Rust 所有权系统博客，含行内代码/引用块/嵌套列表（1-119行）
├── data-analysis.ts
│   └── DATA_ANALYSIS_MD             # 业务分析报表，含 KPI/ASCII 对齐/表格分隔符（1-125行）
├── empty.ts
│   └── EMPTY_MD                     # 空字符串，测试空状态渲染（1-2行）
├── math-formulas.ts
│   └── MATH_FORMULAS_MD             # LaTeX 数学公式集，含行内/展示模式/mhchem（1-248行）
├── readme.ts
│   └── README_MD                    # 虚构 Web 框架 README，含徽章/链接/安装指南（1-130行）
└── storybook-mocks.ts
    ├── mockServerProps()            # 直接注入服务器属性到 Store，支持 Partial 覆盖（10-47行）
    ├── resetServerStore()           # 重置 Store 状态防止测试污染（52-62行）
    └── mockConfigs                  # 预定义模态配置（visionOnly/audioOnly/both/none）（67-79行）

tools/server/webui/.storybook/
└── vitest.setup.ts
    └── __main__                     # 合并 a11y 注解与项目预览配置，注册 beforeAll 钩子（1-14行）

tools/server/webui/tests/unit/
├── agentic-sections.test.ts
│   ├── makeAssistant()              # 构造 Assistant 消息工厂，支持 Partial 覆盖（6-18行）
│   ├── makeToolMsg()                # 构造 Tool 消息工厂，关联 toolCallId（20-32行）
│   └── __main__                     # 测试套件：deriveAgenticSections 与 hasAgenticContent（34-177行）
├── agentic-strip.test.ts
│   ├── stripLegacyContextMarkers()  # 剥离旧格式标记的正则处理（10-17行）
│   └── __main__                     # 测试套件：验证工具调用块/推理块/不完整标记的清理（37-90行）
├── clipboard.test.ts
│   └── __main__                     # 测试 formatMessageForClipboard/parseClipboardContent/hasClipboardAttachments（1-355行）
├── latex-protection.test.ts
│   └── __main__                     # 测试 maskInlineLaTeX 与 preprocessLaTeX（1-363行）
├── model-id-parser.test.ts
│   └── __main__                     # 测试 ModelsService.parseModelId 的组织/参数/量化解析（1-265行）
├── model-names.test.ts
│   └── __main__                     # 测试 normalizeModelName 与 isValidModelName（5-50行）
├── reasoning-context.test.ts
│   └── __main__                     # 测试 reasoningContent 字段的注入与排除逻辑（1-87行）
└── uri-template.test.ts
    └── __main__                     # 测试 extractTemplateVariables/expandTemplate/isTemplateComplete/normalizeResourceUri（9-137行）

tools/server/webui/tests/e2e/
└── demo.test.ts
    └── __main__                     # Playwright 验证首页 H1 可见性（1-6行）

tools/server/webui/tests/client/
└── page.svelte.test.ts
    └── __main__                     # 使用 vitest-browser-svelte 渲染 TestWrapper 冒烟测试（1-11行）
```

## 88.3 Markdown 内容夹具库 —— 模型输出的"剧本与道具组"

在测试 Markdown 渲染组件时，最大的挑战是如何获取真实的、多样化的测试数据。你不能在测试文件中硬编码超长文本——那会严重污染测试代码的可读性与可维护性。llama.cpp WebUI 团队采用了**独立夹具文件**的模式：将具有代表性的 Markdown 内容存放在 `tests/stories/fixtures/` 目录下，每个文件对应一种内容风格。

这些夹具的设计哲学是"**一个文件，一种风格，多个边界**"。每个夹具都针对特定类型的 Markdown 内容进行深度覆盖，让渲染器在真实场景下接受检验。其整体架构如下图所示：

```mermaid
graph TD
    A[Markdown 夹具库] --> B[ai-tutorial.ts]
    A --> C[api-docs.ts]
    A --> D[blog-post.ts]
    A --> E[data-analysis.ts]
    A --> F[math-formulas.ts]
    A --> G[empty.ts]
    A --> H[readme.ts]

    B --> B1[AI 教程: 标题层级/代码块/表格]
    C --> C1[API 文档: HTTP/JSON/多语言 SDK]
    D --> D1[技术博客: 行内代码/嵌套列表/链接]
    E --> E1[数据分析: KPI/ASCII 对齐]
    F --> F1[数学公式: LaTeX 行内/展示模式]
    G --> G1[空状态: 空字符串渲染]
    H --> H1[README: 徽章/安装指南]

    style A fill:#e1f5fe
    style B1 fill:#f0f4c3
    style C1 fill:#f0f4c3
    style D1 fill:#f0f4c3
    style E1 fill:#f0f4c3
    style F1 fill:#f0f4c3
    style G1 fill:#f0f4c3
    style H1 fill:#f0f4c3
```

### 88.3.1 `AI_TUTORIAL_MD`：技术教程类内容的测试夹具

源码路径：`tools/server/webui/tests/stories/fixtures/ai-tutorial.ts` - `AI_TUTORIAL_MD`（1-145行）

```typescript
// ① 定义演示AI助手回复教程的夹具：使用 String.raw 保证原始字符不被转义
// ② 第2行：文档主标题 H1，测试渲染器对顶级标题的支持
export const AI_TUTORIAL_MD = String.raw`
# Building a Modern Chat Application with SvelteKit

// ③ 第4行：加粗与内嵌链接，验证行内元素的解析
I'll help you create a **production-ready chat application** using SvelteKit, TypeScript, and WebSockets. This implementation includes real-time messaging, user authentication, and message persistence.

// ④ 第6行：Emoji 三级标题，测试特殊字符的渲染兼容性
## 🚀 Quick Start

// ⑤ 第11行：${'```'} 表达式安全插入代码块分隔符，避免与模板字符串语法冲突
${'```'}bash
npm create svelte@latest chat-app
cd chat-app
npm install
npm install socket.io socket.io-client
npm install @prisma/client prisma
npm run dev
${'```'}

// ⑥ 第21行：项目结构 ASCII 树，测试渲染器对纯文本结构图的保留
## 📁 Project Structure

${'```'}
chat-app/
├── src/
│   ├── routes/
│   │   ├── +layout.svelte
│   │   ├── +page.svelte
│   │   └── api/
│   │       └── socket/+server.ts
│   ├── lib/
│   │   ├── components/
│   │   │   ├── ChatMessage.svelte
│   │   │   └── ChatInput.svelte
│   │   └── stores/
│   │       └── chat.ts
│   └── app.html
├── prisma/
│   └── schema.prisma
└── package.json
${'```'}

// ⑦ 第40行：TypeScript 代码块，测试语法高亮
## 💻 Implementation

### WebSocket Server

${'```'}typescript
// src/lib/server/socket.ts
import { Server } from 'socket.io';
import type { ViteDevServer } from 'vite';

export function initializeSocketIO(server: ViteDevServer) {
    const io = new Server(server.httpServer || server, {
        cors: {
            origin: process.env.ORIGIN || 'http://localhost:5173',
            credentials: true
        }
    });

    io.on('connection', (socket) => {
        console.log('User connected:', socket.id);
        
        socket.on('message', async (data) => {
            // Broadcast to all clients
            io.emit('new-message', {
                id: crypto.randomUUID(),
                userId: socket.id,
                content: data.content,
                timestamp: new Date().toISOString()
            });
        });

        socket.on('disconnect', () => {
            console.log('User disconnected:', socket.id);
        });
    });

    return io;
}
${'```'}

// ⑧ 第96行：性能指标表格，测试 Markdown 表格渲染
## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Message Latency** | < 50ms |
| **Concurrent Users** | 10,000+ |
| **Messages/Second** | 5,000+ |
| **Uptime** | 99.9% |

// ⑨ 第110行：环境变量代码块与 Dockerfile，测试键值对渲染
## 🔧 Configuration

### Environment Variables

${'```'}env
DATABASE_URL="postgresql://user:password@localhost:5432/chat"
JWT_SECRET="your-secret-key"
REDIS_URL="redis://localhost:6379"
${'```'}

...
`;
```

**为什么用 `String.raw`？** 这是一个关键的设计决策。`String.raw` 模板标签不会处理转义字符，使得 Markdown 中的反引号、反斜杠等特殊字符能够原样保留。如果使用普通模板字符串，`\n` 会被转换为换行符，`\t` 会被转换为制表符，破坏 Markdown 的原始结构。同时，`${'```'}` 表达式也是安全插入代码块分隔符的标准手法——因为三个反引号直接写在模板字符串中会与模板语法冲突，而通过 `${'```'}` 可以优雅地绕过这一限制。

这段代码展示了夹具文件的核心设计模式：使用 `String.raw` 定义原始 Markdown 字符串，并通过 `${'```'}` 表达式安全地插入代码块分隔符，避免与模板字符串语法冲突。该夹具整体用于测试 AI 助手回复技术教程时的渲染效果，覆盖标题层级、行内格式、代码块、表格、Emoji 等边界情况，确保 SvelteKit+WebSocket 教程内容在 Markdown 渲染器中能正确呈现。

### 88.3.2 `API_DOCS_MD`：API 文档类内容的测试夹具

源码路径：`tools/server/webui/tests/stories/fixtures/api-docs.ts` - `API_DOCS_MD`（1-139行）

```typescript
// ① 定义 REST API 文档夹具：覆盖 Bearer 认证、端点参数表格、JSON 响应体
// ② 第2行：文档标题 H1，测试渲染器对顶级标题的支持
export const API_DOCS_MD = String.raw`
# REST API Documentation

// ③ 第3行：认证小节标题，含加粗与内嵌格式
## 🔐 Authentication
All API requests require authentication using **Bearer tokens**. Include your API key in the Authorization header:

// ④ 第5行：HTTP 代码块，测试请求示例的高亮
${'```'}http
GET /api/v1/users
Host: api.example.com
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
${'```'}

// ⑤ 第10行：端点小节，测试 GET 与 POST 请求的文档化
## 📍 Endpoints
### Users API
#### **GET** /api/v1/users
Retrieve a paginated list of users.

// ⑥ 第12-18行：查询参数表格，测试表头/对齐/默认值渲染
**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| limit | integer | 20 | Items per page |
| sort | string | "created_at" | Sort field |
| order | string | "desc" | Sort order |

// ⑦ 第20行：JSON 响应体代码块，测试结构化数据渲染
**Response:** 200 OK
${'```'}json
{
  "data": [
    {
      "id": "usr_1234567890",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "admin",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 156,
    "pages": 8
  }
}
${'```'}

// ⑧ 第42行：错误格式定义，测试错误对象渲染
### Error Responses
The API returns errors in a consistent format:
${'```'}json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "email",
        "message": "Email format is invalid"
      }
    ]
  }
}
${'```'}

// ⑨ 第56行：SDK 示例，测试多语言语法高亮
### SDK Examples
**JavaScript/TypeScript:**
${'```'}typescript
import { ApiClient } from '@example/api-sdk';
const client = new ApiClient({
  apiKey: process.env.API_KEY
});
const users = await client.users.list({
  page: 1,
  limit: 20
});
${'```'}
...
`;
```

API 文档夹具重点测试渲染器对**HTTP 请求示例**、**JSON 响应体**和**错误格式定义**的处理。它还包含 TypeScript 与 Python 双语言的 SDK 示例代码，检验渲染器对不同语言语法高亮的支持。该夹具整体用于测试 AI 生成 API 文档时的完整渲染效果，确保认证信息、端点表格、JSON 结构与错误码格式都能清晰呈现给开发者用户。

这段代码展示了如何在一个夹具中混合多种 Markdown 结构（标题、引用、代码块、表格），构建接近真实 API 文档的复杂场景。

### 88.3.3 `BLOG_POST_MD`：技术博客类内容的测试夹具

源码路径：`tools/server/webui/tests/stories/fixtures/blog-post.ts` - `BLOG_POST_MD`（1-119行）

```typescript
// ① 定义技术博客夹具：涵盖行内代码、引用块、列表与 Rust 代码示例
// ② 第2行：博客标题 H1，测试文章标题渲染
export const BLOG_POST_MD = String.raw`
# Understanding Rust's Ownership System

// ③ 第4行：元信息行，含斜体与发布日期
*Published on March 15, 2024 • 8 min read*

// ④ 第6行：引言段落，测试长文本自动换行
Rust's ownership system is one of its most distinctive features, enabling memory safety without garbage collection. In this post, we'll explore how ownership works and why it's revolutionary for systems programming.

// ⑤ 第10行：小节标题 H2，验证二级标题
## What is Ownership?

// ⑥ 第12-15行：有序列表，测试编号与加粗混排
### The Three Rules of Ownership

1. **Each value has a single owner**
2. **There can only be one owner at a time**
3. **When the owner goes out of scope, the value is dropped**

// ⑦ 第17-22行：无序列表与加粗对比
## Memory Management Without GC

Traditional approaches to memory management:

- **Manual management** (C/C++): Error-prone, leads to bugs
- **Garbage collection** (Java, Python): Runtime overhead
- **Ownership** (Rust): Compile-time safety, zero runtime cost

// ⑧ 第24行：Rust 代码块，测试语法高亮与错误标记
## Basic Examples

### Variable Scope

${'```'}rust
fn main() {
    let s = String::from("hello");  // s comes into scope
    
    // s is valid here
    println!("{}", s);
    
}  // s goes out of scope and is dropped
${'```'}
```

这段代码展示了博客夹具中最基础的一种代码块形式——**带语法高亮的 Rust 代码块**。该代码块展示了 Rust 中变量的作用域与所有权转移，配合其后跟随的"Move Semantics"示例，共同测试渲染器对 Rust 语法高亮与特殊注释符号（如 `//`）的兼容性。夹具通过这种"代码块+解释"的交替模式，模拟了一篇真实技术博客的排版结构，确保渲染器在遇到连续多个代码块时能保持正确的段落分隔。

接下来，夹具继续展示更复杂的场景——**含错误标记的代码块**，用以测试渲染器的容错能力：

```typescript
// ⑨ 第47行：含错误标记的代码块，测试特殊字符容错
### Move Semantics

${'```'}rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;  // s1 is moved to s2
    
    // println!("{}", s1);  // ❌ ERROR: s1 is no longer valid
    println!("{}", s2);     // ✅ OK: s2 owns the string
}
${'```'}

// ⑩ 第78行：嵌套小节与代码块，测试三级标题层级
## Common Pitfalls

### Dangling References

${'```'}rust
fn dangle() -> &String {  // ❌ ERROR: missing lifetime specifier
    let s = String::from("hello");
    &s  // s will be dropped, leaving a dangling reference
}
${'```'}

// ⑪ 第105行：结语与进一步阅读链接，测试外部链接渲染
## Conclusion

Rust's ownership system eliminates entire classes of bugs at compile time. While it has a learning curve, the benefits in safety and performance are worth it.

## Further Reading

- [The Rust Book - Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [Rust by Example - Ownership](https://doc.rust-lang.org/rust-by-example/scope/move.html)
- [Rustlings Exercises](https://github.com/rust-lang/rustlings)
...
`;
```

博客夹具的特点是**行内代码与代码块混排**、**引用块**（`>` 前缀）、**嵌套列表**和**外部链接**。它还包含 `#` 标题层级从 H1 到 H3 的嵌套结构，以及带"❌ ERROR"和"✅ OK"标记的 Rust 代码示例，测试渲染器对特殊字符的容错能力。该夹具整体用于测试 AI 生成技术博客文章时的渲染兼容性，确保读者在阅读长文时，代码示例、列表结构与链接跳转都能正常工作。

### 88.3.4 `DATA_ANALYSIS_MD`：数据分析报表类内容的测试夹具

源码路径：`tools/server/webui/tests/stories/fixtures/data-analysis.ts` - `DATA_ANALYSIS_MD`（1-125行）

```typescript
// ① 定义业务分析报表夹具：包含 KPI 列表、ASCII 对齐与表格分隔符
// ② 第2行：报表标题 H1，测试仪表盘标题
export const DATA_ANALYSIS_MD = String.raw`
# Q4 2024 Business Analytics Report

// ③ 第3行：副标题行，测试斜体元信息
*Executive Summary • Generated on January 15, 2025*

// ④ 第4行：KPI 小节，测试 ASCII 对齐渲染
## 📊 Key Performance Indicators

${'```'}
Daily Active Users (DAU):   1.2M (+65% YoY)
Monthly Active Users (MAU): 4.5M (+48% YoY)
User Retention (Day 30):    68% (+12pp YoY)
Average Session Duration:   24min (+35% YoY)
${'```'}

// ⑤ 第17行：更多指标与可视化，测试图表/表格混合
## 📈 Revenue Metrics
...
`;
```

该夹具涵盖 KPI 列表、ASCII 对齐与表格分隔符，用于测试基于 Markdown 的数据报表场景下的渲染效果。

### 88.3.5 `MATH_FORMULAS_MD`：数学公式类内容的测试夹具

源码路径：`tools/server/webui/tests/stories/fixtures/math-formulas.ts` - `MATH_FORMULAS_MD`（1-248行）

```typescript
// ① 定义数学公式夹具：包含 LaTeX 行内/展示模式、mhchem 化学式
// ② 第1行：ESLint 禁用不规则空白检查，保留数学公式的原始空白
/* eslint-disable no-irregular-whitespace */
// ③ 第2行：注释说明内容用途
// Math Formulas Content
export const MATH_FORMULAS_MD = String.raw`
...(此处省略具体的 markdown 公式内容)...
`;
```

该夹具覆盖了 LaTeX 行内公式（如 `\(...\)`）、展示模式（如 `\[...\]`）以及化学式标记（mhchem 的 `\ce{...}` 与 `\pu{...}`），用于测试渲染器在数学公式密集场景下的正确处理能力。由于公式中大量使用特殊空白字符（如 Unicode 窄空格、数学斜体空格），文件顶部用 `/* eslint-disable no-irregular-whitespace */` 禁用了 ESLint 对不规则空白的检查，以保证夹具内容的原样保留。

### 88.3.6 `EMPTY_MD`：空状态内容的测试夹具

源码路径：`tools/server/webui/tests/stories/fixtures/empty.ts` - `EMPTY_MD`（1-2行）

```typescript
// ① 空状态夹具：验证渲染组件在空字符串输入下的表现
// ② 第2行：空字符串常量，作为测试用例的最小输入
export const EMPTY_MD = '';
```

该夹具虽然只有一行，却是测试中不可或缺的边界样本——它验证渲染组件在**空字符串输入**下的表现，既不抛错也不显示垃圾内容。

而 `mockConfigs`（源码路径：`tools/server/webui/tests/stories/fixtures/storybook-mocks.ts` - `mockConfigs`，67-79行）则提供了四种预定义的多模态配置（`visionOnly`、`audioOnly`、`bothModalities`、`noModalities`），让测试无需手写模态配置即可模拟不同场景。

## 88.4 Storybook 测试环境搭建 —— 摄影棚灯光调试间的"联姻仪式"

当你在 Vitest 中运行测试时，是否希望 Storybook 的全局配置（装饰器、参数、全局样式）同样生效？答案是肯定的。storybook-mocks 只是数据层面，而环境配置需要 `vitest.setup.ts` 来完成这次"联姻"。

### 88.4.1 `__main__`：初始化并合并预览配置

源码路径：`tools/server/webui/.storybook/vitest.setup.ts` - `__main__`（1-14行）

```typescript
// ① 导入 Storybook 的无障碍检查插件（a11y）的预览注解，确保测试环境同样具备无障碍检查能力
import * as a11yAddonAnnotations from '@storybook/addon-a11y/preview';

// ② 从 SvelteKit 集成包中导入注解合并工具，用于合并多组注解到一个项目级上下文
import { setProjectAnnotations } from '@storybook/sveltekit';

// ③ 导入项目的自定义预览配置（含 SvelteKit 集成、全局装饰器等），保持与 Storybook 预览一致
import * as previewAnnotations from './preview';

// ④ 从 Vitest 导入 beforeAll 钩子，用于注册测试执行前的全局初始化逻辑
import { beforeAll } from 'vitest';

// ⑤ 将 a11y 插件注解与项目预览注解合并到项目级上下文，形成统一的测试环境配置
const project = setProjectAnnotations([a11yAddonAnnotations, previewAnnotations]);

// ⑥ 注册全局 beforeAll：在测试套件开始前执行注解中的初始化逻辑，确保环境就绪
beforeAll(async () => {
	if (project.beforeAll) {
		// ⑦ 如果项目注解定义了 beforeAll 钩子，则等待其执行完成
		await project.beforeAll();
	}
});
```

**为什么需要合并两组注解？** 注意 `setProjectAnnotations` 接收了一个数组 `[a11yAddonAnnotations, previewAnnotations]`。这意味着 Storybook 的无障碍检查插件（a11y）的注解与项目的预览配置（`./preview`，包含 SvelteKit 集成等）被合并到了一起。

这段代码定义了测试环境的"宪法"：将 a11y 插件与项目 preview 配置统一注册到测试环境，并在每个测试文件开始前执行全局初始化钩子。这里的 `setProjectAnnotations` 就像我们摄影棚灯光调试间的**调光台**，将两路光源（a11y 插件与 preview 配置）合并为统一的照明方案，确保演员（组件）在不同布景（测试环境与 Storybook 预览）下的肤色完全一致。

### 88.4.2 `mockServerProps()`：替身演员与绿幕的注入

源码路径：`tools/server/webui/tests/stories/fixtures/storybook-mocks.ts` - `mockServerProps()`（10-47行）

```typescript
// ① 定义 mockServerProps 函数：接收 Partial 类型的 props 覆盖参数，用于灵活模拟服务器属性
export function mockServerProps(props: Partial<ApiLlamaCppServerProps>): void {
	// ② 重置上一次测试遗留的 pointer-events 样式（下拉菜单清理），防止样式污染
	const body = document.querySelector('body');
	// ③ 如果 body 元素存在，则清空其 pointerEvents 内联样式
	if (body) body.style.pointerEvents = '';

	// ④ 直接设置 props 供测试使用：通过类型断言绕过 Store 的只读限制，注入测试数据
	(serverStore as unknown as { props: ApiLlamaCppServerProps }).props = {
		// ⑤ 设置模型路径，若未指定则使用默认值 'test-model'
		model_path: props.model_path || 'test-model',
		// ⑥ 设置模态配置：vision 与 audio，未指定时默认均为 false
		modalities: {
			vision: props.modalities?.vision ?? false,
			audio: props.modalities?.audio ?? false
		},
		// ⑦ 将所有传入的 props 字段合并覆盖到默认值之上
		...props
	} as ApiLlamaCppServerProps;

	// ⑧ 设置 router 模式 role 使 activeModelId 可被设置
	(serverStore as unknown as { props: ApiLlamaCppServerProps }).props.role = 'ROUTER';

	// ⑨ 同步 mock modelsStore 的多模态检查函数，使测试环境能正确判断模态支持
	const vision = props.modalities?.vision ?? false;
	// ⑩ 同样读取 audio 模态配置
	const audio = props.modalities?.audio ?? false;

	// ⑪ 使用 eslint-disable 抑制 any 类型的告警，因为测试 Mock 场景需要绕过类型检查
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	(modelsStore as any).modelSupportsVision = () => vision;
	// ⑫ 同理，mock modelSupportsAudio 方法
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	(modelsStore as any).modelSupportsAudio = () => audio;

	// ⑬ Mock 模型列表使 activeModelId 可被解析：提供一个测试模型供 Store 查询
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	(modelsStore as any).models = [
		{
			id: 'test-model',
			name: 'Test Model',
			model: 'test-model'
		}
	];

	// ⑭ Mock 选中的模型 ID，确保 UI 层可以获取当前选择的模型
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	(modelsStore as any).selectedModelId = 'test-model';
}
```

这里的关键设计是**类型安全的部分覆盖**：参数类型 `Partial<ApiLlamaCppServerProps>` 允许你只传入需要覆盖的字段，其余字段使用默认值。这避免了每次测试都要构造完整的 props 对象。

这段代码展示了如何通过类型断言直接修改 Store 内部状态，绕过真实 API 调用，并提供默认值保证字段完整性。这里就像替身演员的**绿幕注入**——无需真实服务器，只需在绿幕（Mock 数据）前完成表演（渲染），后期再合成真实场景。

### 88.4.3 `resetServerStore()` 与 `mockConfigs`：环境清理与预定义配置

源码路径：`tools/server/webui/tests/stories/fixtures/storybook-mocks.ts` - `resetServerStore()`（52-62行）

```typescript
// ① 定义 resetServerStore 函数：将 Store 恢复到干净的初始状态，防止测试间污染
export function resetServerStore(): void {
	// ② 重置 props 为默认值：空模型路径与双 false 模态
	(serverStore as unknown as { props: ApiLlamaCppServerProps }).props = {
		// ③ 模型路径置空，表示未选择任何模型
		model_path: '',
		// ④ 模态配置重置：vision 与 audio 均为 false
		modalities: {
			vision: false,
			audio: false
		}
	} as ApiLlamaCppServerProps;
	// ⑤ 清空错误信息字段，确保没有残留错误状态
	(serverStore as unknown as { error: string }).error = '';
	// ⑥ 重置加载状态为 false，确保没有残留的 loading 标志
	(serverStore as unknown as { loading: boolean }).loading = false;
}
```

`resetServerStore` 负责**状态隔离**——将 Store 恢复到"出厂设置"，防止测试之间的污染。同时，`mockConfigs` 提供了四种预定义配置（`visionOnly`、`audioOnly`、`bothModalities`、`noModalities`），让你不用每次手写模态配置。其源码路径为 `tools/server/webui/tests/stories/fixtures/storybook-mocks.ts` - `mockConfigs`（67-79行）：

```typescript
// ① 定义 mockConfigs 常量：预定义常用的多模态配置组合，减少测试代码重复
export const mockConfigs = {
	// ② visionOnly：仅视觉模态启用，适合测试纯视觉场景
	visionOnly: {
		modalities: { vision: true, audio: false }
	},
	// ③ audioOnly：仅音频模态启用，适合测试纯音频场景
	audioOnly: {
		modalities: { vision: false, audio: true }
	},
	// ④ bothModalities：双模态同时启用，适合测试多模态融合场景
	bothModalities: {
		modalities: { vision: true, audio: true }
	},
	// ⑤ noModalities：无模态启用，适合测试默认无多模态场景
	noModalities: {
		modalities: { vision: false, audio: false }
	}
} as const;
```

## 88.5 Agentic 消息结构测试 —— 剧本结构分析的"解剖课"

当模型不仅能生成文本，还能调用工具时，消息结构变得复杂了。`deriveAgenticSections` 是解析这类复杂消息的核心工具，它将原始消息数组解析为结构化段落。测试这类逻辑，我们需要遵循**消息构造器模式**。

### 88.5.1 `makeAssistant()` 与 `makeToolMsg()`：消息工厂函数

源码路径：`tools/server/webui/tests/unit/agentic-sections.test.ts` - `makeAssistant()`（6-18行）

```typescript
// ① 定义 makeAssistant 工厂函数：接收 Partial<DatabaseMessage> 覆盖参数，默认为空对象
function makeAssistant(overrides: Partial<DatabaseMessage> = {}): DatabaseMessage {
	return {
		// ② 默认 id 为 'ast-1'，可用 overrides.id 覆盖
		id: overrides.id ?? 'ast-1',
		// ③ 默认对话 ID 为 'conv-1'，用于消息分组
		convId: 'conv-1',
		// ④ 默认类型为 'text'，表示普通文本消息
		type: 'text',
		// ⑤ 默认时间戳为当前时间，模拟消息创建时间
		timestamp: Date.now(),
		// ⑥ 默认角色为 ASSISTANT，标识助手消息
		role: MessageRole.ASSISTANT,
		// ⑦ 默认内容为空字符串，可用 overrides.content 设置实际内容
		content: overrides.content ?? '',
		// ⑧ 父消息指针为 null，表示根消息
		parent: null,
		// ⑨ 子消息数组为空，表示无后续关联消息
		children: [],
		// ⑩ 将所有覆盖参数展开合并到默认值之上
		...overrides
	} as DatabaseMessage;
}
```

`makeAssistant` 是一个**工厂函数**：它返回一个结构完整的 `DatabaseMessage` 对象，并通过 `...overrides` 支持字段级别的局部覆盖。你不需要为每个测试用例手写完整的消息对象，只需覆盖你关心的字段（如 `content`、`reasoningContent`）。

源码路径：`tools/server/webui/tests/unit/agentic-sections.test.ts` - `makeToolMsg()`（20-32行）

```typescript
// ① 定义 makeToolMsg 工厂函数：专门构造 TOOL 角色的消息，携带 toolCallId
function makeToolMsg(overrides: Partial<DatabaseMessage> = {}): DatabaseMessage {
	return {
		// ② 默认 id 为 'tool-1'，可用 overrides.id 覆盖
		id: overrides.id ?? 'tool-1',
		// ③ 默认对话 ID 为 'conv-1'，与 makeAssistant 保持一致
		convId: 'conv-1',
		// ④ 默认类型为 'text'，表示普通文本消息
		type: 'text',
		// ⑤ 默认时间戳为当前时间
		timestamp: Date.now(),
		// ⑥ 默认角色为 TOOL，标识工具执行结果消息
		role: MessageRole.TOOL,
		// ⑦ 默认内容为 'tool result'，模拟工具返回的文本
		content: overrides.content ?? 'tool result',
		// ⑧ 父消息指针为 null，表示根消息
		parent: null,
		// ⑨ 子消息数组为空
		children: [],
		// ⑩ 默认 toolCallId 为 'call_1'，与 makeAssistant 中的默认值配对
		toolCallId: overrides.toolCallId ?? 'call_1',
		// ⑪ 将所有覆盖参数展开合并
		...overrides
	} as DatabaseMessage;
}
```

注意 `makeToolMsg` 的默认 `toolCallId` 是 `'call_1'`，这与 `makeAssistant` 中的默认值一致。这种"**默认值配对**"设计让多轮工具调用的构造变得简单——你无需担心关联缺失。

### 88.5.2 `__main__`：核心测试套件

源码路径：`tools/server/webui/tests/unit/agentic-sections.test.ts` - `__main__`（34-177行）

```typescript
// ① 定义 deriveAgenticSections 测试套件：覆盖从空消息到复杂多轮工具调用的场景
describe('deriveAgenticSections', () => {
	// ② 边界测试：空内容应返回空段落数组，验证不会产生多余段落
	it('returns empty array for assistant with no content', () => {
		// ③ 使用 makeAssistant 创建空内容消息
		const messages = [makeAssistant({ content: '' })];
		// ④ 调用核心函数解析消息为段落
		const result = deriveAgenticSections(messages);
		// ⑤ 断言结果为空数组
		expect(result).toEqual([]);
	});

	// ⑥ 基础测试：纯文本消息应转换为单个段落，验证基本转换逻辑
	it('returns text section for simple assistant message', () => {
		// ⑦ 创建包含文本内容的消息
		const messages = [makeAssistant({ content: 'Hello' })];
		// ⑧ 调用解析函数
		const result = deriveAgenticSections(messages);
		// ⑨ 断言段落类型与内容正确
		expect(result).toEqual([{ type: 'text', content: 'Hello' }]);
	});

	// ⑩ 推理内容测试：包含 reasoningContent 时应拆分为独立段落
	it('returns reasoning + text for message with reasoning', () => {
		// ⑪ 创建带推理内容与正文的消息
		const messages = [makeAssistant({
			content: 'Answer',
			reasoningContent: 'Let me think'
		})];
		// ⑫ 调用解析函数
		const result = deriveAgenticSections(messages);
		// ⑬ 断言推理与文本段落分离
		expect(result).toEqual([
			{ type: 'reasoning', content: 'Let me think' },
			{ type: 'text', content: 'Answer' }
		]);
	});

	// ⑭ 单轮工具调用测试：文本与工具结果交替输出
	it('single turn: assistant with tool calls and results', () => {
		// ⑮ 构造一次工具调用与结果的多消息序列
		const messages = [
			makeAssistant({ content: 'Let me search', toolCalls: JSON.stringify([{ id: 'call_1' }]) }),
			makeToolMsg({ toolCallId: 'call_1', content: 'Result' }),
			makeAssistant({ content: 'Final answer' })
		];
		// ⑯ 调用解析函数
		const result = deriveAgenticSections(messages);
		// ⑰ 断言段落结构：文本、工具调用、文本
		expect(result).toHaveLength(3);
	});

	// ⑱ 挂起状态测试：未完成的工具调用被正确标记
	it('single turn: pending tool call without result', () => { ... });

	// ⑲ 多轮分组测试：同会话的多次助手消息被聚合为一个会话
	it('multi-turn: two assistant turns grouped as one session', () => { ... });

	// ⑳ 复杂多轮测试：三个轮次包含工具调用与结果交替
	it('multi-turn: three turns with tool calls', () => { ... });

	// ㉑ 流式边界测试：最后一轮调用未结束的流式场景
	it('multi-turn: streaming tool calls on last turn', () => { ... });
});

// ㉒ 定义 hasAgenticContent 测试套件：判断消息是否包含 Agentic 特征
describe('hasAgenticContent', () => {
	// ㉓ 普通消息测试：无工具调用与推理内容时返回 false
	it('returns false for plain assistant', () => {
		const msg = makeAssistant({ content: 'Hello' });
		expect(hasAgenticContent(msg)).toBe(false);
	});

	// ㉔ 工具调用存在即判定为 true
	it('returns true when message has toolCalls', () => {
		const msg = makeAssistant({ toolCalls: JSON.stringify([{ id: 'call_1' }]) });
		expect(hasAgenticContent(msg)).toBe(true);
	});

	// ㉕ 关联的工具结果也视为 Agentic，即使当前消息无工具调用
	it('returns true when toolMessages are provided', () => { ... });

	// ㉖ 空 JSON 数组不触发属性判定，保持普通消息语义
	it('returns false for empty toolCalls JSON', () => {
		const msg = makeAssistant({ toolCalls: '[]' });
		expect(hasAgenticContent(msg)).toBe(false);
	});
});
```

这段代码定义了 `deriveAgenticSections` 与 `hasAgenticContent` 两个核心功能的完整测试矩阵。测试套件的覆盖面从基础的空内容、纯文本，到复杂的多轮工具调用会话，再到流式工具调用边界。这种"从简单到复杂"的设计，能精确定位 `deriveAgenticSections` 的每个逻辑分支。通过 `hasAgenticContent` 的四个用例，我们验证了判断逻辑的完整性：工具调用字段（`toolCalls`）与工具结果消息（`toolMessages`）都是判定 Agentic 内容的关键信号，而空 JSON 数组则不应触发误判。

### 88.5.3 `stripLegacyContextMarkers()`：旧格式兼容处理

源码路径：`tools/server/webui/tests/unit/agentic-strip.test.ts` - `stripLegacyContextMarkers()`（10-17行）

```typescript
// ① 定义 stripLegacyContextMarkers 函数：通过正则替换剥离旧格式的 Agentic 标记
function stripLegacyContextMarkers(content: string): string {
	return content
		// ② 全局替换完整的推理块标记（REASONING_BLOCK），移除所有出现的完整推理内容
		.replace(new RegExp(LEGACY_AGENTIC_REGEX.REASONING_BLOCK.source, 'g'), '')
		// ③ 移除推理开始标记（REASONING_OPEN），处理不完整或无结束标记的推理块
		.replace(LEGACY_AGENTIC_REGEX.REASONING_OPEN, '')
		// ④ 全局替换完整的工具调用块标记（AGENTIC_TOOL_CALL_BLOCK），清理完整工具块
		.replace(new RegExp(LEGACY_AGENTIC_REGEX.AGENTIC_TOOL_CALL_BLOCK.source, 'g'), '')
		// ⑤ 移除工具调用开始标记（AGENTIC_TOOL_CALL_OPEN），处理未闭合的工具调用块
		.replace(LEGACY_AGENTIC_REGEX.AGENTIC_TOOL_CALL_OPEN, '');
}
```

当模型更新后，**旧格式的推理标记**（如 `<<<reasoning_content_start>>>` 这样的标记）必须被清理。这个函数通过四步正则替换，依次剥离完整的推理块、推理开始标记、完整的工具调用块和工具调用开始标记。

### 88.5.4 `__main__`：旧格式标记清理的测试套件

源码路径：`tools/server/webui/tests/unit/agentic-strip.test.ts` - `__main__`（37-90行）

```typescript
// ① 定义旧格式标记清理测试套件：验证迁移期间各类标记的剥离效果
describe('legacy agentic marker stripping (for migration)', () => {
	// ② 完整工具调用块测试：确保完整块被移除，周围文本保留
	it('strips a complete tool call block, leaving surrounding text', () => {
		// ③ 拼接测试输入：前置文本 + 完整工具块 + 后置文本
		const input = 'Before.' + COMPLETE_BLOCK + 'After.';
		// ④ 调用清理函数处理输入
		const result = stripLegacyContextMarkers(input);
		// ⑤ 断言不包含任何 <<< 标记
		expect(result).not.toContain('<<<');
		// ⑥ 断言前置文本被保留
		expect(result).toContain('Before.');
		// ⑦ 断言后置文本被保留
		expect(result).toContain('After.');
	});

	// ⑧ 多块测试：多个完整工具块被全部移除
	it('strips multiple complete tool call blocks', () => { ... });

	// ⑨ 未闭合块测试：无 END 标记的块也被正确剥离
	it('strips an open/partial tool call block (no END marker)', () => { ... });

	// ⑩ 无标记测试：普通文本不受影响
	it('does not alter content with no markers', () => {
		const input = 'Just a normal assistant response.';
		expect(stripLegacyContextMarkers(input)).toBe(input);
	});

	// ⑪ 推理块测试：推理标记被独立剥离
	it('strips reasoning block independently', () => {
		const input = '<<<reasoning_content_start>>>think hard<<<reasoning_content_end>>>Answer.';
		expect(stripLegacyContextMarkers(input)).toBe('Answer.');
	});

	// ⑫ 组合测试：推理与工具块同时被清理
	it('strips both reasoning and agentic blocks together', () => { ... });

	// ⑬ 空字符串测试：空输入安全通过
	it('empty string survives', () => {
		expect(stripLegacyContextMarkers('')).toBe('');
	});

	// ⑭ 标记检测测试：HAS_LEGACY_MARKERS 正则能正确识别旧标记
	it('detects legacy markers', () => { ... });
});
```

这段代码验证了旧格式标记的完整清理逻辑：无论是完整块、未闭合块还是组合场景，所有旧标记都能被剥离，同时不破坏周围的正常文本。这是数据迁移期保证用户对话历史不被旧标记污染的关键防线。

## 88.6 剪贴板消息格式化 —— 海关安检的数据进出

当用户复制聊天消息到剪贴板时，是直接复制纯文本，还是复制结构化数据？这取决于消息是否包含附件。`formatMessageForClipboard` 与 `parseClipboardContent` 构成了一对"打包"与"拆包"函数。其完整的数据流如下图所示：

```mermaid
graph TD
    A[原始消息 + 附件列表] --> B{是否有文本附件？}
    B -- 否 --> C[返回纯文本原样]
    B -- 是 --> D{asPlainText 模式？}
    D -- 是 --> E[纯文本拼接: 消息 + 附件内容]
    D -- 否 --> F[JSON 序列化: 首行消息 + 附件数组]
    F --> G[过滤非文本附件: 图片/PDF 被丢弃]
    E --> H[剪贴板输出]
    G --> H
    H --> I[parseClipboardContent 解析]
    I --> J{特殊格式检测: 首行是否为引号?}
    J -- 否 --> K[作为普通文本返回]
    J -- 是 --> L[JSON 数组解析]
    L --> M{解析成功?}
    M -- 否 --> N[优雅降级为普通文本]
    M -- 是 --> O[恢复消息与文本附件]
    K --> P[最终解析结果]
    N --> P
    O --> P
```

### 88.6.1 `__main__`：剪贴板格式化与解析的测试套件

源码路径：`tools/server/webui/tests/unit/clipboard.test.ts` - `__main__`（1-355行）

```typescript
// ① 导入 Vitest 测试框架的 describe/it/expect 方法
import { describe, it, expect } from 'vitest';
// ② 导入附件类型枚举，用于区分文本/图片/PDF 等附件类型
import { AttachmentType } from '$lib/enums';
// ③ 导入剪贴板工具函数：格式化、解析与附件检测
import {
    formatMessageForClipboard,
    parseClipboardContent,
    hasClipboardAttachments
} from '$lib/utils/clipboard';

// ④ 定义 formatMessageForClipboard 测试套件：验证打包逻辑
describe('formatMessageForClipboard', () => {
    // ⑤ 基础场景：无附件时应返回纯文本原样
    it('returns plain content when no extras', () => {
        // ⑥ 调用打包函数，传入内容与 undefined 附件
        const result = formatMessageForClipboard('Hello world', undefined);
        // ⑦ 断言返回内容与输入一致
        expect(result).toBe('Hello world');
    });

    // ⑧ 空数组场景：空附件数组等效于无附件
    it('returns plain content when extras is empty array', () => { ... });

    // ⑨ 空字符串场景：空内容应安全返回空字符串
    it('handles empty string content', () => { ... });

    // ⑩ 纯非文本附件场景：仅图片附件时返回纯文本
    it('returns plain content when extras has only non-text attachments', () => { ... });

    // ⑪ 关键过滤测试：过滤非文本附件，仅保留文本附件
    it('filters non-text attachments and keeps only text ones', () => {
        // ⑫ 混合附件数组：图片、文本、PDF 三种类型
        const extras = [
            { type: AttachmentType.IMAGE as const, name: 'image.png', base64Url: 'data:image/png;base64,...' },
            { type: AttachmentType.TEXT as const, name: 'file.txt', content: 'Text content' },
            { type: AttachmentType.PDF as const, name: 'doc.pdf', base64Data: 'data:application/pdf;base64,...', content: 'PDF content', processedAsImages: false }
        ];
        // ⑬ 调用打包函数
        const result = formatMessageForClipboard('Hello', extras);
        // ⑭ 断言 JSON 中包含文本附件文件名
        expect(result).toContain('"file.txt"');
        // ⑮ 断言图片文件名被过滤
        expect(result).not.toContain('image.png');
        // ⑯ 断言 PDF 文件名被过滤
        expect(result).not.toContain('doc.pdf');
    });

    // ⑰ 特殊字符测试：引号、换行等需正确转义
    it('handles content with quotes and special characters', () => {
        const content = 'Hello "world" with\nnewline';
        const extras = [{ type: AttachmentType.TEXT as const, name: 'test.txt', content: 'Test content' }];
        const result = formatMessageForClipboard(content, extras);
        expect(result.startsWith('"')).toBe(true);
        const parsed = JSON.parse(result.split('\n')[0]);
        expect(parsed).toBe(content);
    });

    // ⑱ 纯文本模式测试：asPlainText 为 true 时使用纯文本拼接
    it('formats as plain text when asPlainText is true', () => {
        const extras = [
            { type: AttachmentType.TEXT as const, name: 'file1.txt', content: 'File 1 content' },
            { type: AttachmentType.TEXT as const, name: 'file2.txt', content: 'File 2 content' }
        ];
        const result = formatMessageForClipboard('Hello world', extras, true);
        expect(result).toBe('Hello world\n\nFile 1 content\n\nFile 2 content');
    });
});

// ⑲ 定义 parseClipboardContent 测试套件：验证拆包逻辑
describe('parseClipboardContent', () => {
    // ⑳ 普通文本场景：非特殊格式时按纯文本解析
    it('returns plain text as message when not in special format', () => {
        const result = parseClipboardContent('Hello world');
        expect(result.message).toBe('Hello world');
        expect(result.textAttachments).toHaveLength(0);
    });

    // ㉑ 截断 JSON 场景：格式损坏时优雅降级为纯文本
    it('returns original text when JSON array is malformed', () => { ... });

    // ㉒ 有效 JSON 场景：正确解析消息与附件
    it('parses message with text attachments', () => {
        const input = `"Hello world"
[
  {"type":"TEXT","name":"file1.txt","content":"File 1 content"},
  {"type":"TEXT","name":"file2.txt","content":"File 2 content"}
]`;
        const result = parseClipboardContent(input);
        expect(result.message).toBe('Hello world');
        expect(result.textAttachments).toHaveLength(2);
        expect(result.textAttachments[0].name).toBe('file1.txt');
    });

    // ㉓ 非法附件过滤：无效类型与缺失字段的附件被过滤
    it('filters out invalid attachment objects', () => { ... });

    // ㉔ 转义引号测试：JSON 中的转义字符被正确还原
    it('handles escaped quotes in message', () => { ... });

    // ㉕ 换行字符测试：JSON 中的 \n 被还原为实际换行
    it('handles newlines in message', () => { ... });
});

// ㉖ 定义 hasClipboardAttachments 测试套件：检测剪贴板是否含附件
describe('hasClipboardAttachments', () => {
    // ㉗ 普通文本：不含附件信息
    it('returns false for plain text', () => {
        expect(hasClipboardAttachments('Hello world')).toBe(false);
    });

    // ㉘ 有效格式：正确检测附件存在
    it('returns true for valid format with attachments', () => {
        const input = `"Hello"
[{"type":"TEXT","name":"file.txt","content":"test"}]`;
        expect(hasClipboardAttachments(input)).toBe(true);
    });

    // ㉙ 空附件数组：不视为含附件
    it('returns false for format with empty attachments array', () => { ... });

    // ㉚ 损坏 JSON：不视为含附件
    it('returns false for malformed JSON', () => { ... });
});

// ㉛ 定义 roundtrip 边界测试套件：验证打包与拆包的一致性
describe('roundtrip edge cases', () => {
    // ㉜ 空消息与附件：空内容消息与附件被正确往返保真
    it('preserves empty message with attachments', () => { ... });

    // ㉝ 空附件内容：附件 content 为空时往返保持空字符串
    it('preserves attachment with empty content', () => { ... });

    // ㉞ 多重反斜杠：Windows 路径等场景保持原样
    it('preserves multiple backslashes', () => {
        const content = 'Path: C:\\Users\\test\\file.txt';
        const extras = [{ type: AttachmentType.TEXT as const, name: 'path.txt', content: 'D:\\Data\\file' }];
        const formatted = formatMessageForClipboard(content, extras);
        const parsed = parseClipboardContent(formatted);
        expect(parsed.message).toBe(content);
        expect(parsed.textAttachments[0].content).toBe('D:\\Data\\file');
    });

    // ㉟ 制表符与空白：各种空白字符正确保真
    it('preserves tabs and various whitespace', () => { ... });
});
```

这段代码定义了剪贴板格式化的**核心规则**：无附件时返回纯文本；有文本附件时自动过滤非文本附件（图片、PDF）并序列化为 JSON；用户指定 Plain Text 模式时拼接内容。而 `parseClipboardContent` 则反向解析：识别纯文本与 JSON，优雅处理截断 JSON、非法附件类型等异常输入。通过 round-trip 测试（先格式化再解析），确保了**打包与拆包的一致性**：特殊字符（引号、反斜杠、换行）经过转义后能无损还原，Unicode 字符（中日韩文字与 Emoji）在各种模式下都能正确保真。

## 88.7 LaTeX 保护机制 —— 货币符号的"防伪检测"

在 Markdown 渲染中，`$` 符号既可能是货币符号，也可能是 LaTeX 数学公式的定界符。一个渲染器要区分 `$5`（货币）与 `$x$`（数学公式），依赖启发式规则。

### 88.7.1 `__main__`：LaTeX 保护函数的测试套件

源码路径：`tools/server/webui/tests/unit/latex-protection.test.ts` - `__main__`（1-363行）

```typescript
// ① ESLint 禁用不规则空白检查，保留 LaTeX 公式中的特殊空白
/* eslint-disable no-irregular-whitespace */
// ② 导入 Vitest 测试框架与 LaTeX 保护工具函数
import { describe, it, expect, test } from 'vitest';
import { maskInlineLaTeX, preprocessLaTeX } from '$lib/utils/latex-protection';

// ③ 定义 maskInlineLaTeX 测试套件：验证行内 LaTeX 保护逻辑
describe('maskInlineLaTeX', () => {
    // ④ 核心边界：数字后紧跟 $ 视为货币，不应被保护
    it('should protect LaTeX $x + y$ but not money $3.99', () => { ... });
    // ⑤ 货币测试：$5 与 $12.99 不应被当作 LaTeX 处理
    it('should ignore money like $5 and $12.99', () => { ... });
    // ⑥ 行内公式测试：$a^2 + b^2$ 在文本中应被识别
    it('should protect inline math $a^2 + b^2$ even after text', () => { ... });
    // ⑦ 单位边界：$ 后有字母（如单位符号）不应被保护
    it('should not protect math that has letter after closing $ (e.g. units)', () => { ... });
    // ⑧ 标点边界：$x$ 后跟标点时应被正确保护
    it('should allow $x$ followed by punctuation', () => { ... });
    // ⑨ 多行文本测试：跨行场景下的保护逻辑
    it('should work across multiple lines', () => { ... });
    // ⑩ 数字与矩阵：$100 是货币，$matrix$ 是数学
    it('should not protect $100 but protect $matrix$', () => { ... });
    // ⑪ 空文本边界：空字符串安全通过
    it('empty text', () => { ... });
});

// ⑫ 定义 preprocessLaTeX 测试套件：验证 LaTeX 定界符预处理
describe('preprocessLaTeX', () => {
    // ⑬ 转换测试：行内 \(...\) 转换为 $...$ 格式
    test('converts inline \\( ... \\) to $...$', () => { ... });
    // ⑭ 展示公式保留：\[...\] 应被保留为展示模式
    test('preserves display math \\[ ... \\] and protects adjacent text', () => { ... });
    // ⑮ 货币非 LaTeX：金额不被误判为数学公式
    test('does not interpret currency values as LaTeX', () => { ... });
    // ⑯ 转义测试：孤立 $ 后跟数字（$5 → \$5），有效数学不被转义
    test('escapes isolated $ before digits ($5 → \\$5), but not valid math', () => { ... });
    // ⑰ 代码块保护：backtick 代码块内的 \( 不应被转换
    test('preserves code blocks', () => { ... });
    // ⑱ 引用块测试：blockquotes 中的行内/展示公式
    test('LaTeX in blockquotes with inline math', () => { ... });
    // ⑲ mhchem 化学式：安全处理化学公式标记
    test('handles mhchem notation safely if present', () => { ... });
});
```

**启发式规则的边界情况**：

-   数字后紧跟 `$` 视为货币（`$5`）
-   字母后接 `$` 视为 LaTeX（`$x$`）
-   单元格内的 `$` 与空格有特定处理
-   `$$` 双美元符号视为展示公式
-   代码块（backtick）内的 `\(` 不应被转换

这段代码展示了如何通过"**双层防御**"保护 LaTeX：首先用 `preprocessLaTeX` 将 `\(...\)` 和 `\[...\]` 转换为标准的 `$...$` 与 `$$...$$` 格式，然后用 `maskInlineLaTeX` 在 Markdown 渲染前将特殊字符替换为占位符，防止被解析器误伤。这里的启发式规则就像货币防伪检测中的**水印识别**，通过一系列检查点（数字后 `$`、字母后 `$`、代码块标记）区分真币（数学公式）与假币（货币符号）。

整个处理流程如下：

```mermaid
graph TD
    A[原始 Markdown 文本] --> B{preprocessLaTeX}
    B -- 转换 --> C[标准 LaTeX 定界符（$...$/$$...$$）]
    C --> D{是否有货币符号冲突？}
    D -- 是 --> E[maskInlineLaTeX 将货币 $ 转义为 \\$]
    D -- 否 --> F[保留原样]
    E --> G[Mask 后的安全文本]
    F --> G
    G --> H[Markdown 渲染器处理]
    H --> I[渲染完成后还原 LaTeX 内容]
    I --> J[最终安全渲染结果]
```

## 88.8 模型 ID 解析与命名规范 —— 模型的"身份证识别"与资源调度

用户输入的模型路径可能是 HuggingFace 的 `org/model` 格式，也可能是本地文件路径 `C:\models\model.bin`。`normalizeModelName` 需要统一处理这些格式差异，而 `parseModelId` 则进一步结构化解析标识。

### 88.8.1 `__main__`：模型名称规范化的测试用例

源码路径：`tools/server/webui/tests/unit/model-names.test.ts` - `__main__`（5-50行）

```typescript
// ① 保留 HuggingFace 格式测试：单斜杠被视为 HF 格式并保留原样
it('preserves Hugging Face org/model format (single slash)', () => {
	// ② 单斜杠是 HF 格式，保留 org/model 结构
	// ③ 断言 HH 格式的模型路径被原样保留
	expect(normalizeModelName('meta-llama/Llama-3.1-8B')).toBe('meta-llama/Llama-3.1-8B');
	// ④ 断言另一个单斜杠路径也被保留
	expect(normalizeModelName('models/model-name-1')).toBe('models/model-name-1');
});

// ⑤ 多段路径提取测试：多个斜杠时提取文件名
it('extracts filename from multi-segment paths', () => {
	// ⑥ 多个斜杠 -> 仅提取文件名
	// ⑦ 断言三段路径仅返回最后的模型名
	expect(normalizeModelName('path/to/model/model-name-2')).toBe('model-name-2');
	// ⑧ 断言绝对路径也仅返回最后的目录名
	expect(normalizeModelName('/absolute/path/to/model')).toBe('model');
});

// ⑨ 反斜杠路径测试：Windows 风格路径也提取文件名
it('extracts filename from backslash paths', () => {
	// ⑩ Windows 路径使用反斜杠，同样提取最后一个路径段
	expect(normalizeModelName('C\Models\model-name-1')).toBe('model-name-1');
	// ⑪ 多段反斜杠路径同样提取末尾文件名
	expect(normalizeModelName('path\to\model\model-name-2')).toBe('model-name-2');
});

// ⑫ 混合路径分隔符测试：正反斜杠共存时提取文件名
it('handles mixed path separators', () => {
	// ⑬ 混合使用正反斜杠的路径应正常提取
	expect(normalizeModelName('path/to\model/model-name-2')).toBe('model-name-2');
});

// ⑭ 简单名称测试：无路径分隔符时原样返回
it('returns simple names as-is', () => {
	expect(normalizeModelName('simple-model')).toBe('simple-model');
	expect(normalizeModelName('model-name-2')).toBe('model-name-2');
});

// ⑮ 空白处理：首尾空白会被去除
it('trims whitespace', () => {
	expect(normalizeModelName('  model-name  ')).toBe('model-name');
});

// ⑯ 空输入测试：空字符串或纯空白返回空字符串
it('returns empty string for empty input', () => {
	expect(normalizeModelName('')).toBe('');
	expect(normalizeModelName('   ')).toBe('');
});

// ⑰ 有效名称校验测试：有效模型名应返回 true
it('returns true for valid names', () => {
	expect(isValidModelName('model')).toBe(true);
	expect(isValidModelName('path/to/model.bin')).toBe(true);
});

// ⑱ 空值校验测试：空字符串或纯空白返回 false
it('returns false for empty values', () => {
	expect(isValidModelName('')).toBe(false);
	expect(isValidModelName('   ')).toBe(false);
});
```

**关键设计决策**：单斜杠保留为 HuggingFace 格式，而多斜杠路径提取末级文件名。这解释了 `parseModelId` 在 `meta-llama/Llama-3.1-8B` 中如何拆分出组织名 `meta-llama` 和模型名 `Llama-3.1-8B`，并进一步提取参数规模 `8B` 和量化类型 `Q4_K_M`。

### 88.8.2 `__main__`：模型 ID 结构化解析的测试套件

源码路径：`tools/server/webui/tests/unit/model-id-parser.test.ts` - `__main__`（1-265行）

```typescript
// ① 导入 Vitest 测试框架与 ModelsService 服务
import { describe, expect, it } from 'vitest';
import { ModelsService } from '$lib/services/models.service';

// ② 从服务中解构 parseModelId 方法
const { parseModelId } = ModelsService;

// ③ 定义 parseModelId 测试套件
describe('parseModelId', () => {
	// ④ 标准 HF 格式：组织名/模型名-参数量-量化
	it('parses standard HF format', () => { ... });
	// ⑤ 组织名缺失或空字符串
	it('handles missing organization', () => { ... });
	// ⑥ 参数量大小写与分隔符变体（8b/8B/8-b/8_B）
	it('handles parameter size variants', () => { ... });
	// ⑦ 量化类型多种写法（Q4_K_M/q4km/Q8_0）
	it('parses quantization variants', () => { ... });
	// ⑧ 未知格式优雅降级：返回 unknown 并保留原始字符串
	it('degrades gracefully for unknown formats', () => { ... });
});
```

在 `model-id-parser.test.ts` 中，测试覆盖了**组织名与模型名拆分**、**参数规模识别**、**量化类型提取**以及各种分隔符变体（短横线、下划线、无分隔符）。对于未知格式，解析器优雅降级，保持原样或标记为 unknown。

### 88.8.3 `__main__`：URI 模板处理与推理上下文过滤的测试套件

在测试体系中，还有两个容易被忽略却至关重要的工具模块：**URI 模板处理**与**推理上下文过滤**。前者负责 RFC 6570 标准的模板展开，后者保证推理过程的"内心独白"不会污染对外展示的文本。

**URI 模板处理**（源码路径：`tools/server/webui/tests/unit/uri-template.test.ts` - `__main__`，9-137行）

`extractTemplateVariables` 从 `{/id}`、`{?query}` 等模板中提取变量名；`expandTemplate` 将变量值填入模板生成具体 URI；`isTemplateComplete` 校验所有必填变量是否已赋值；`normalizeResourceUri` 则处理路径式展开产生的 `///` 异常。

```typescript
// ① 定义 extractTemplateVariables 测试套件：验证模板变量提取
describe('extractTemplateVariables', () => {
	// ② 简单变量提取：{id} → ['id']
	it('extracts simple variables', () => {...});
	// ③ 多变量提取：{a}{b} → ['a', 'b']
	it('extracts multiple variables', () => {...});
	// ④ 带操作符变量：{?q} → ['q']
	it('extracts variables with operators', () => {...});
	// ⑤ 逗号分隔列表提取
	it('extracts comma-separated variable lists', () => {...});
	// ⑥ 去重测试：{x}{x} → ['x']
	it('deduplicates variable names', () => {...});
	// ⑦ 片段展开：#fragment 中的变量
	it('handles fragment expansion', () => {...});
	// ⑧ 路径段展开：{/path} 中的变量
	it('handles path segment expansion', () => {...});
	// ⑨ 无变量模板：返回空数组
	it('returns empty array for template without variables', () => {...});
	// ⑩ 爆炸修饰符去除：{list*} → ['list']
	it('strips explode modifier', () => {...});
	// ⑪ 前缀修饰符去除
	it('strips prefix modifier', () => {...});
});

// ⑫ 定义 expandTemplate 测试套件：验证模板展开
describe('expandTemplate', () => {
	// ⑬ 简单变量展开：用具体值替换占位符
	it('expands simple variable', () => {...});
	// ⑭ 保留变量展开：不进行 URL 编码
	it('expands reserved variable (no encoding)', () => {...});
	// ⑮ 多变量展开
	it('expands multiple variables', () => {...});
	// ⑯ 缺失变量处理：缺失时留空
	it('leaves empty for missing variables', () => {...});
	// ⑰ 片段展开：处理 #fragment
	it('expands fragment', () => {...});
	// ⑱ 路径段展开
	it('expands path segments', () => {...});
	// ⑲ 查询参数展开
	it('expands query parameters', () => {...});
	// ⑳ 静态部分不变：模板中非变量部分原样保留
	it('keeps static parts unchanged', () => {...});
});

// ㉑ 定义 isTemplateComplete 测试套件：验证模板完整性
describe('isTemplateComplete', () => {
	// ㉒ 所有变量已填：返回 true
	it('returns true when all variables are filled', () => {...});
	// ㉓ 缺失变量：返回 false
	it('returns false when a variable is missing', () => {...});
	// ㉔ 空值变量：返回 false
	it('returns false when a variable is empty', () => {...});
	// ㉕ 纯空白变量：返回 false
	it('returns false when a variable is whitespace only', () => {...});
	// ㉖ 无变量模板：返回 true
	it('returns true for template without variables', () => {...});
	// ㉗ 多变量全部填满：返回 true
	it('returns true when all multiple variables are filled', () => {...});
});

// ㉘ 定义 normalizeResourceUri 测试套件：验证 URI 规范化
describe('normalizeResourceUri', () => {
	// ㉙ 正常 URI 通过：不改变标准 URI
	it('passes through a normal URI unchanged', () => {...});
	// ㉚ 三重斜杠规范化：路径式模板展开产生的 /// 被修复为 /
	it('normalizes triple-slash URIs from path-style template expansion', () => {...});
	// ㉛ 四重斜杠规范化
	it('normalizes quadruple-slash URIs', () => {...});
	// ㉜ file:// URI 处理
	it('handles file:// URIs', () => {...});
	// ㉝ HTTP URI 不变
	it('handles http URIs unchanged', () => {...});
	// ㉞ 非 URI 字符串不变
	it('returns non-URI strings unchanged', () => {...});
});
```

这段代码验证了 URI 模板处理工具集的完整功能：从变量提取、展开、完整性校验到 URI 规范化，构成了一条完整的资源定位链路。这些工具就像我们的**资源调度中心**，将模板中的占位符精准展开为可用的 API 路径，并修复展开过程中可能产生的 `///` 异常。

**推理上下文过滤**（源码路径：`tools/server/webui/tests/unit/reasoning-context.test.ts` - `__main__`，1-87行）

在新架构中，推理内容存储在独立字段 `reasoningContent` 中，而非内嵌于消息正文。测试验证了：推理内容不会出现在 `content` 字段；API 转换时 `reasoning_content` 能正确注入或排除（当 `excludeReasoningFromContext` 为 true 时）；无推理内容时字段保持 undefined。

```typescript
// ① 导入 Vitest 测试框架与消息角色枚举
import { describe, it, expect } from 'vitest';
import { MessageRole } from '$lib/enums';

// ② 文档注释：说明新架构下推理内容的存储方式
/**
 * Tests for the new reasoning content handling.
 * In the new architecture, reasoning content is stored in a dedicated
 * `reasoningContent` field on DatabaseMessage, not embedded in content with tags.
 * The API sends it as `reasoning_content` on ApiChatMessageData.
 */

// ③ 定义推理内容测试套件：验证结构化存储与 API 转换逻辑
describe('reasoning content in new structured format', () => {
	// ④ 独立字段测试：推理内容不污染 content 字段
	it('reasoning is stored as separate field, not in content', () => {
		// ⑤ 模拟新 chat store 的消息结构
		const message = {
			content: 'The answer is 4.',
			reasoningContent: 'Let me think: 2+2=4, basic arithmetic.'
		};

		// ⑥ 内容应为纯净文本，不含内部标记
		expect(message.content).not.toContain('<<<');
		// ⑦ 断言 content 保持原样
		expect(message.content).toBe('The answer is 4.');

		// ⑧ 推理存在于专用字段
		expect(message.reasoningContent).toBe('Let me think: 2+2=4, basic arithmetic.');
	});

	// ⑨ API 转换测试：reasoning_content 被正确包含
	it('convertDbMessageToApiChatMessageData includes reasoning_content', () => {
		// ⑩ 模拟数据库消息结构
		const dbMessage = {
			role: MessageRole.ASSISTANT,
			content: 'The answer is 4.',
			reasoningContent: 'Let me think: 2+2=4, basic arithmetic.'
		};

		// ⑪ 模拟 API 消息转换逻辑
		const apiMessage: Record<string, unknown> = {
			role: dbMessage.role,
			content: dbMessage.content
		};
		// ⑫ 推理内容存在时注入 reasoning_content 字段
		if (dbMessage.reasoningContent) {
			apiMessage.reasoning_content = dbMessage.reasoningContent;
		}

		// ⑬ 断言内容字段正确
		expect(apiMessage.content).toBe('The answer is 4.');
		// ⑭ 断言推理字段被正确注入
		expect(apiMessage.reasoning_content).toBe('Let me think: 2+2=4, basic arithmetic.');
		// ⑮ 无内部标记泄漏
		expect(apiMessage.content).not.toContain('<<<');
		expect(apiMessage.reasoning_content).not.toContain('<<<');
	});

	// ⑯ 排除测试：excludeReasoningFromContext 为 true 时排除推理
	it('API message excludes reasoning when excludeReasoningFromContext is true', () => {
		const dbMessage = {
			role: MessageRole.ASSISTANT,
			content: 'The answer is 4.',
			reasoningContent: 'internal thinking'
		};

		// ⑰ 设置排除标志为 true
		const excludeReasoningFromContext = true;

		const apiMessage: Record<string, unknown> = {
			role: dbMessage.role,
			content: dbMessage.content
		};
		// ⑱ 排除标志为 true 时不注入推理字段
		if (!excludeReasoningFromContext && dbMessage.reasoningContent) {
			apiMessage.reasoning_content = dbMessage.reasoningContent;
		}

		// ⑲ 断言内容字段正确
		expect(apiMessage.content).toBe('The answer is 4.');
		// ⑳ 断言推理字段未被注入
		expect(apiMessage.reasoning_content).toBeUndefined();
	});

	// ㉑ 无推理测试：无推理内容时字段保持 undefined
	it('handles messages with no reasoning', () => {
		const dbMessage = {
			role: MessageRole.ASSISTANT,
			content: 'No reasoning here.',
			reasoningContent: undefined
		};

		const apiMessage: Record<string, unknown> = {
			role: dbMessage.role,
			content: dbMessage.content
		};
		// ㉒ 无推理内容时不注入字段
		if (dbMessage.reasoningContent) {
			apiMessage.reasoning_content = dbMessage.reasoningContent;
		}

		// ㉓ 断言内容字段正确
		expect(apiMessage.content).toBe('No reasoning here.');
		// ㉔ 断言推理字段保持 undefined
		expect(apiMessage.reasoning_content).toBeUndefined();
	});
});
```

这段代码验证了两个测试模块的配合：URI 模板确保 API 路径正确展开，推理上下文过滤确保敏感推理内容不会未经授权地泄露。后者就像我们的**剧本审校员**，确保"内心独白"（推理过程）不会混入对外展示的台词，保护用户隐私与数据安全。

## 88.9 设计中的取舍

在构建这套测试体系时，我们面临了多个关键的设计决策，每个决策都伴随着取舍。

第一个取舍是关于 Markdown 测试数据的组织方式。我们为什么不直接在手写测试文件中嵌入完整的 Markdown 内容，而要建立独立的夹具文件库？手写完整内容会带来三个问题：首先，测试文件会被超长文本污染，导致代码逻辑难以快速定位，开发者需要滚动很久才能找到断言语句。其次，相同的内容在多个测试文件中无法复用，破坏 DRY 原则，每次需要修改内容时都要同步更新多个位置。第三，当夹具内容需要调整时，例如模拟新的模型输出格式，开发者需要在多个测试文件中逐一更新，极易遗漏并引入难以追踪的错误。而独立夹具文件通过"一个文件一种风格"的模式解决了这些问题，让内容的增删改以极低的成本进行。这种设计也有其代价：读者需要在文件与测试之间进行"跳转"，增加了认知负担，但这一代价在可维护性收益面前是可以接受的。

第二个取舍是关于 Mock 实现方式的选型。我们为什么使用类型断言（`as unknown as`）而不是 `vi.mock()` 来修改 Store 内部状态？`vi.mock()` 适合模块级替换，但在需要逐测试修改 Store 内部状态的场景下不够灵活——你必须在每个测试文件中重新声明 mock 逻辑，无法精确控制某个字段的值。通过类型断言快速修改对象属性虽然牺牲了一点类型安全（需要用 `eslint-disable` 抑制警告），但获得了极大的灵活性：你可以精确控制某个字段的值，而不必重写整个模块。这种方式的代价是断言失败时类型系统无法提供保护，但这在测试场景中是可以接受的——测试的目的是验证运行行为，而非类型安全。

第三个取舍与 LaTeX 保护机制相关：为什么不直接屏蔽所有的 `$` 符号？如果屏蔽所有 `$`，虽然从根本上解决了渲染冲突，但会导致货币金额（$5）也被错误转换或转义，破坏用户期望的正常显示。因此我们采用启发式规则：数字后紧邻 `$` 视为货币，字母后视为数学公式。

这个设计还需要面对一个**模糊场景的取舍**：当金额后紧跟字母时（如 `$5x$`），当前启发式规则会将其识别为数学公式而非货币。从语义上看，`$5x` 可能是"5 倍的 x"（数学表达式），也可能是"5x 元"（货币加字母）。由于数学公式在技术文档中更为常见，且货币通常以独立数字或带小数位出现，因此规则倾向于将其归类为数学公式。这个决策牺牲了极少数边界情况（如国际代码如 `$5CAD` 表示 5 加元）的正确识别，但换来了绝大多数场景的准确渲染。这种取舍的核心是**最大化覆盖典型场景，同时优雅降级处理模糊情况**。

第四个取舍关乎剪贴板格式的双模式设计。为什么剪贴板既支持 JSON 也支持纯文本模式？JSON 结构化格式保留了附件的类型与内容元数据，适合 WebUI 与 WebUI 之间的数据迁移，确保复制粘贴不会丢失附件信息。但纯文本模式让用户能以最通用的格式复制消息到任何应用（如微信、邮件），提供最大的兼容性。两种模式的代价各不相同：JSON 模式的代价是特殊字符需要转义，增加了格式化与解析的复杂度；而纯文本模式的代价是丢失附件元数据，只能保留纯文本内容。这种双模式设计让用户根据使用场景灵活选择，实现了**功能性与通用性的平衡**。

## 88.10 动手练习

在此部分，我们通过五个练习掌握本章核心概念。每个练习包含具体步骤与需要回答的问题。建议按顺序完成，逐步构建对测试体系的全景认识。

**练习 1：创建 Markdown 夹具。** 阅读 `tools/server/webui/tests/stories/fixtures/` 目录下的 `ai-tutorial.ts` 与 `math-formulas.ts`。创建新夹具文件 `custom-code.ts`。该文件需包含三类内容：含 Python 装饰器语法的代码块；含中文与 Emoji 混排的段落；含未闭合代码块边界的异常样本。回答问题：为什么用 `String.raw` 而不是普通模板字符串定义 Markdown 内容？该夹具应测试哪些渲染边界？

**练习 2：理解 Store Mock 隔离。** 阅读 `storybook-mocks.ts` 中 `mockServerProps` 与 `resetServerStore`，以及 `vitest.setup.ts` 的环境初始化逻辑。设计测试场景：Test A 调用 `mockServerProps({ vision: true })`；Test B 不调用 reset 直接运行并断言默认 `vision: false`。观察失败原因。回答问题：`resetServerStore` 为何重置 `pointerEvents`？`mockServerProps` 为何需同时 mock `modelsStore.modelSupportsVision`？

**练习 3：分析 LaTeX 保护。** 阅读 `latex-protection.test.ts` 中处理 `$5` 与 `$x$` 的启发式规则测试。编写新测试用例验证以下边界：`'Price is $5 and $x$ is variable'` 只保护后者；`'$$e = mc^2$$'` 作为独立行展示公式；`` '`$not_math$`' `` 在代码块内不转换。回答问题：当前实现如何区分 `$100` 与 `$x$`？代码块保护机制如何实现？

**练习 4：验证剪贴板往返。** 阅读 `clipboard.test.ts` 中 round-trip 测试，理解 `formatMessageForClipboard` 与 `parseClipboardContent` 的配对逻辑。设计边界场景：消息含 JSON 特殊字符（引号、反斜杠、换行）；附件含空 `content` 字段；Plain Text 模式导出后重新导入。回答问题：为何需要 `hasClipboardAttachments` 函数？JSON 模式与 Plain Text 模式在数据保真度上有何差异？

**练习 5：构建 Agentic 对话。** 阅读 `agentic-sections.test.ts` 中 `makeAssistant` 与 `makeToolMsg` 工厂函数。构造两轮 Agentic 对话：第一轮 Assistant 发起工具调用（pending 状态）；第二轮 Tool 返回结果，Assistant 生成最终回复。运行测试验证 `deriveAgenticSections` 是否将两轮对话正确分组。回答问题：`hasAgenticContent` 判定中 `toolCalls` 与 `toolMessages` 的关系是什么？为何需处理空字符串 content 的边界？

## 88.11 本章小结

在前面的十个章节中，我们从多个维度深入剖析了 llama.cpp WebUI 前端测试体系的完整面貌。现在让我们回顾本章的核心脉络：我们从 Markdown 内容夹具库的设计原则出发，理解了如何通过 `String.raw` 与独立文件组织，为不同渲染场景准备精确的"剧本与道具"。接着我们剖析了 Storybook 与 Vitest 的集成机制，掌握了 `setProjectAnnotations` 与 `beforeAll` 钩子如何实现全局环境的一致化。之后我们拆解了 Agentic 消息测试，学会了用工厂函数模式构造复杂多轮对话，并用 `deriveAgenticSections` 将原始消息解析为结构化段落。随后我们分析了剪贴板数据进出的"海关检验"，理解了 JSON 与纯文本双模式的设计哲学与边界处理。我们还研究了 LaTeX 保护机制的启发式规则，掌握了区分货币与数学公式的"防弹衣"算法。最后我们学习了模型 ID 解析与命名规范，理解了从字符串到结构化元数据的完整链路，以及 URI 模板处理与推理上下文过滤如何共同守护资源定位的安全性与用户隐私。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| Markdown Fixtures | 独立的内容夹具文件，提供多样化真实文本样本测试 Markdown 渲染边界 |
| setProjectAnnotations | Storybook 测试环境配置合并机制，确保单元测试与 Storybook 行为一致 |
| mockServerProps / mockConfigs | Store 状态注入工具与预定义配置，通过 Partial\<T> 实现灵活的服务器属性模拟 |
| deriveAgenticSections | 消息结构解析器，将原始对话转换为文本、推理、工具调用等标准化段落 |
| formatMessageForClipboard | 智能剪贴板序列化，根据附件类型自动切换纯文本/JSON 结构化格式 |
| parseClipboardContent | 剪贴板反序列化，支持格式自动识别与非法输入的优雅降级 |
| maskInlineLaTeX / preprocessLaTeX | LaTeX 保护机制，区分货币符号与数学公式，防止 Markdown 渲染冲突 |
| parseModelId | 模型标识解析器，从字符串提取组织名、参数规模、量化类型等元数据 |
| normalizeModelName | 模型名称规范化，统一处理 HuggingFace 路径与本地文件路径 |
| URI Template Utils | RFC 6570 模板处理工具集，支持变量提取、展开、完整性校验与 URI 规范化 |
| reasoningContent | 推理内容独立字段，与消息主体分离以支持隐私过滤与结构化处理 |
| TestWrapper / vitest-browser-svelte | 组件级冒烟测试工具，验证 Svelte 组件在测试环境中的基础渲染能力 |

下一章中，我们将学习前端遗留与基准测试，解析支持旧版本聊天协议的 SimpleChat 前端，以及基于 k6 的服务端性能基准测试，理解兼容性与性能这两个"试金石"如何共同守护 llama-server 的质量。

