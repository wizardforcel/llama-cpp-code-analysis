

# 第107章 优化器、多模态与扩展组件测试 —— 模型能力的"长尾校验"

## 学习目标

- 理解 gallocr 内存分配器的虚拟后端测试架构，包括 dummy_backend 与 dummy_backend_context 的设计意图、接口实现与内存统计机制
- 掌握 GGML 内存分配器的关键行为：缓冲区分裂、碎片整理、张量复用、视图零拷贝、多缓冲区类型分配与重新分配
- 理解线程池屏障同步的压力测试方法，包括动态线程数切换、多图交替计算的竞态与隔离性验证
- 掌握跨后端数值一致性验证的完整流程：随机权重 GGUF 构造、NMSE 误差度量、保存-加载往返测试与 MoE 架构分类
- 理解多模型 × 多上下文笛卡尔积并发测试的设计模式与原子标志位错误聚合机制
- 掌握模型加载取消与 KV 缓存碎片状态恢复的工程细节，以及日志库并发压力测试方法
- 能够独立编写针对内存分配器、并发场景和模型架构的白盒测试用例

## 107.1 生活类比

想象 llama.cpp 的测试体系是一个综合质检实验室，每个测试文件都像一台专门的检测仪器，用来验证引擎在极端工况下的表现。就像汽车制造商需要在风洞、耐久赛道和碰撞实验室中反复测试新车型，llama.cpp 的测试套件覆盖了从内存分配器的精密调度到多后端数值一致性，从线程安全到模型加载取消的全链路验证。虚拟后端如同可编程的风洞，能够精确控制内存容量和对齐约束；gallocr 分配器则是智能仓库管理员，在有限的货架空间里通过缓冲区分裂、碎片整理和内存复用最大化利用率；线程池屏障测试模拟交响乐团指挥在极短时间内同步众多乐手的考验；而多模型并发测试就像多车道高速收费站，检验系统在同时处理众多独立车道时是否会出现拥堵或事故。这些测试共同构成了 llama.cpp 能力的"长尾校验"，确保引擎不仅在理想条件下运行良好，更能在复杂、压力和边界场景中保持稳定。

## 107.2 源码地图

tests/test-alloc.cpp —— GGML 内存分配器测试
├── 虚拟后端定义（第21-115行）
│   ├── dummy_backend_context                  # 虚拟后端上下文
│   │   └── allocated_total()                  # 统计已分配总内存（25-31行）
│   ├── dummy_backend                          # 虚拟后端实例（context + buffer_type）
│   ├── dummy_backend_buffer_type_get_name()   # 缓冲区类型名称（34-36行）
│   ├── dummy_backend_buffer_type_alloc_buffer() # 分配缓冲区（39-44行）
│   ├── dummy_backend_buffer_type_get_alignment() # 返回对齐要求（47-50行）
│   ├── dummy_backend_buffer_type_get_max_size()  # 返回最大缓冲区限制（53-56行）
│   ├── dummy_backend_buffer_type_is_host()       # 标记为主机内存（59-61行）
│   ├── dummy_backend_buffer_free_buffer()        # 从列表中移除并释放缓冲区（66-73行）
│   ├── dummy_backend_buffer_get_base()           # 返回固定基地址（76-78行）
│   ├── dummy_backend_buffer_init_tensor()        # 空操作初始化（81-83行）
│   ├── dummy_backend_buffer_memset_tensor()      # 空操作（85行）
│   ├── dummy_backend_buffer_set_tensor()         # 空操作（87行）
│   ├── dummy_backend_buffer_get_tensor()         # 空操作（89行）
│   ├── dummy_backend_buffer_clear()              # 空操作（91行）
│   └── dummy_backend_init(max_buffer_size, alignment) # 初始化虚拟后端（96-115行）
├── 测试辅助工具（第125-265行）
│   ├── make_context()                    # 创建 ggml 上下文与计算图（125-133行）
│   ├── make_input_1d()                   # 创建一维输入张量（136-140行）
│   ├── make_input_with_size()            # 按字节大小创建输入张量（143-146行）
│   ├── assign_names()                    # 为上下文所有张量统一命名（148-153行）
│   ├── get_leaf_id()                     # 按名称查找叶子节点索引（156-165行）
│   ├── get_node_id()                     # 按名称查找节点索引（167-176行）
│   ├── allocate_graph()                  # 分配图内存（gallocr 封装）（181-188行）
│   ├── check_all_allocated()             # 验证所有节点已分配（193-199行）
│   ├── check_max_size()                  # 验证张量不超出缓冲区上限（204-211行）
│   ├── can_reuse_memory()                # 判断其他张量是否仍被引用（213-231行）
│   ├── memory_overlap()                  # 检查两区间是否交叠（234-242行）
│   ├── get_view_source()                 # 回溯到最终视图源张量（244-249行）
│   └── check_no_overlap()                # 全图张量重叠检查（251-265行）
├── 分配器行为测试（第269-556行）
│   ├── test_max_size_too_many_tensors()  # 缓冲区容量耗尽→自动分裂（273-291行）
│   ├── test_max_size_tensor_too_large()  # 大张量跨越缓冲区剩余空间（294-309行）
│   ├── test_tensor_larger_than_max_size() # 单张量超上限→分配更大缓冲区（313-325行）
│   ├── test_not_enough_chunks()           # 超出最大块数→动态增长（329-349行）
│   ├── test_fill_leftover_space()         # 填充缓冲区剩余空间（352-366行）
│   ├── test_view_inplace()                # 视图+in-place 零额外内存（369-385行）
│   ├── test_reuse_and_free()              # 释放块的复用与再分配（388-406行）
│   ├── test_merge_free_block()            # 相邻空闲块合并（409-428行）
│   ├── test_prefer_already_allocated_memory() # 优先复用已分配内存（432-444行）
│   ├── test_multiple_buffer_types()       # 多缓冲区类型手动映射（448-503行）
│   ├── test_buffer_size_zero()            # 零尺寸缓冲区的安全处理（506-528行）
│   └── test_reallocation()                # gallocr 复用时的重新分配（531-556行）
├── run()                                 # 测试执行辅助函数（559-564行）
└── main()                                # 按序运行全部 13 个测试（566-583行）

tests/test-barrier.cpp —— 线程池同步压力测试
├── test_barrier(n_threads, n_rounds)     # 大量小算子图的屏障开销基准（12-86行）
│   ├── 构造 1000×2 个 mul_mat 链式图（Q4_0 量化）
│   └── 输出每节点纳秒耗时统计
├── test_active(n_threads, n_rounds)      # 动态线程数切换竞态检测（89-135行）
│   ├── (i%4)==0 → 1 线程，否则 → n_threads
│   └── 每轮重建 cplan 验证竞态安全
├── test_multi_graph(n_threads, n_rounds) # 多图交替执行测试（138-204行）
│   ├── gf0: 64维 2层链  vs  gf1: 256维 4层链
│   └── 交替计算验证 work_data 隔离
└── __main__                              # 解析线程数/轮数参数，调度三个测试（207-222行）

tests/test-llama-archs.cpp —— 跨后端架构一致性测试
├── 随机数据与工具
│   ├── nmse()                            # 归一化均方误差计算（30-46行）
│   ├── set_tensor_data()                 # 按名称哈希生成确定性随机权重（49-70行）
│   ├── usage()                           # 打印程序用法（73-75行）
│   ├── get_tokens()                      # 生成随机 token 序列（77-87行）
│   └── silent_model_load_progress()      # 静默进度回调（192-194行）
├── 模型构造与推理
│   ├── get_gguf_ctx()                    # 按架构生成随机权重 GGUF（89-190行）
│   ├── get_model_and_ctx()               # 从 GGUF 上下文或文件创建模型+上下文（196-224行）
│   ├── get_logits()                      # 编码/解码获取全部 logits（226-260行）
│   ├── moe_mandatory()                   # 判断架构是否强制 MoE（262-306行）
│   └── moe_implemented()                 # 判断架构是否支持 MoE 变体（308-324行）
├── 测试入口
│   ├── save_models()                     # 保存全部架构 GGUF 到目录模式（326-389行）
│   └── test_backends()                   # CPU vs 各后端 NMSE 对比+往返测试（391-491行）
└── main()                                # CLI 参数解析，二选一入口（493-547行）

tests/test-thread-safety.cpp —— 多线程并发推理测试
└── __main__                              # 主测试流程（16-137行）
    ├── 枚举 GPU 设备列表，加载 num_models = gpu_count + 2 个模型副本
    ├── 每模型派生 n_parallel 个上下文线程执行 decode 循环
    └── std::atomic<bool> failed 聚合错误

tests/test-autorelease.cpp —— 自动释放简单测试
└── main()                                # 在独立线程中执行完整 init/free 周期（11-26行）

tests/test-model-load-cancel.cpp —— 加载取消测试
└── main()                                # 加载进度超过 50% 时取消（5-26行）

tests/test-state-restore-fragmented.cpp —— KV 碎片状态恢复测试
└── __main__                              # 三序列 KV 碎片制造与恢复（14-119行）
    ├── 交错解码 3 序列创建共享 KV 缓存
    ├── llama_state_seq_get_data() 保存序列 1 状态
    ├── llama_memory_seq_rm() 删除序列 1 → 制造碎片
    └── llama_state_seq_set_data() 非连续槽位恢复

tests/test-log.cpp —— 日志系统压力测试
└── __main__                              # 8线程×1000条随机类型日志并发写入（5-32行）
    ├── 随机选择 INF/WRN/ERR/DBG 日志级别
    └── 随机切换 timestamps / prefixes 配置

## 107.3 虚拟后端与测试基础设施 —— 内存分配的"模拟沙盘与度量衡"

### 107.3.1 为什么需要 dummy_backend_context 与 dummy_backend？

在真实硬件上测试内存分配器就像在实际道路上测试新款制动系统：变量太多，结果不可控。为了获得可重复、可测量的结论，llama.cpp 的测试套件构建了一个`dummy_backend`——纯软件实现的虚拟后端，它不依赖任何真实 GPU 或 CPU，却能完整模拟后端分配缓冲区、报告已用内存、强制对齐和容量上限的行为。这个设计让我们能够像在受控实验室里那样，精确调节“风洞”参数（如最大缓冲区大小和内存对齐要求），专注观察分配器在各种极端工况下的决策逻辑。

核心在于`dummy_backend_context`：它持有两个关键配置——`max_buffer_size`（单个缓冲区的上限）和`alignment`（内存地址对齐要求），并维护一个已分配缓冲区列表。通过`allocated_total()`方法，我们可以实时监控已用内存总量，这成为验证分配器行为的核心断言依据。例如，当测试某个场景下理论上不应超过 32 字节时，我们只需断言`allocated_total() <= 32`即可。

结构如下：

```cpp
// 源码路径：tests/test-alloc.cpp - dummy_backend_context.allocated_total()（25-31行）
struct dummy_backend_context {
    std::vector<ggml_backend_buffer_t> buffers;
    size_t max_buffer_size;
    size_t alignment;

    size_t allocated_total() const {
        size_t n = 0;
        for (ggml_backend_buffer_t buf : buffers) {
            n += ggml_backend_buffer_get_size(buf);
        }
        return n;
    }
};
```

### 107.3.2 ggml_backend_buffer 接口的空操作艺术

为了让虚拟后端专注于测试分配策略而非数据传输，`dummy_backend` 实现了`ggml_backend_buffer` 接口的所有方法——但这些方法全是空操作（no-op）。`init_tensor`、`memset_tensor`、`set_tensor`、`get_tensor` 和 `clear` 函数体内什么也不做，直接返回成功或无操作。这种设计有一个巧妙的副产品：`get_base()` 返回固定地址 `alloc_base`（一个全局常量），使得所有张量的内存地址变得可预测。当每个张量都从同一个基地址偏移分配时，我们就能通过简单的地址运算计算出精确的内存布局，进而验证张量是否真的复用了同一块内存、是否发生了越界，或者两个张量的内存区域是否不当重叠。

这就像在仓库里给每件货物贴上带有精确坐标的条形码——不管货物实际存放在哪个货架，我们总能通过扫描条码知道它究竟占用了哪块地板空间。

具体实现包括：

```cpp
// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_type_alloc_buffer（39-44行）
static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend_context * ctx    = (dummy_backend_context *) buft->context;
    ggml_backend_buffer_t & buffer = ctx->buffers.emplace_back();
    buffer                         = ggml_backend_buffer_init(buft, ctx->buffer_interface, ctx, size);
    return buffer;
}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_type_get_alignment（47-50行）
static size_t dummy_backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->alignment;
}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_type_get_max_size（53-56行）
static size_t dummy_backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->max_buffer_size;
}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_type_is_host（59-61行）
static bool dummy_backend_buffer_type_is_host(ggml_backend_buffer_type_t) {
    return true;
}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_free_buffer（66-73行）
static void dummy_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;
    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    ctx->buffers.erase(i);
}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_get_base（76-78行）
static void * dummy_backend_buffer_get_base(ggml_backend_buffer_t) {
    return alloc_base;
}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_init_tensor（81-83行）
static ggml_status dummy_backend_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_memset_tensor（85行）
static void dummy_backend_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_set_tensor（87行）
static void dummy_backend_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) {}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_get_tensor（89行）
static void dummy_backend_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) {}

// 源码路径：tests/test-alloc.cpp - dummy_backend_buffer_clear（91行）
static void dummy_backend_buffer_clear(ggml_backend_buffer_t, uint8_t) {}
```

### 107.3.3 测试夹具的标准化构造

可靠的测试需要一致的起点。`make_context()` 函数封装了创建 ggml 上下文和计算图的标准流程：它预留了足够的元数据空间（48 倍张量开销加上图开销），设置 `no_alloc = true`（延迟分配），然后初始化上下文和空图。这一步确保了所有测试在相同的初始条件下开始。

为进一步简化，`make_input_1d()` 和 `make_input_with_size()` 两个函数能够快速生成标记为 `GGML_TENSOR_FLAG_INPUT` 的标准输入张量——前者按元素个数，后者按字节大小（需为 4 的倍数）。有了这些输入张量，我们就能用 `assign_names()` 为上下文中的所有张量分配格式化名称（如 `x0`、`x1`、`x2`），从而通过名称查找函数 `get_leaf_id()` 和 `get_node_id()` 建立张量名称与图节点索引的映射。这种命名约定在后续进行多缓冲区类型的手动映射时至关重要，因为它让我们能够明确指定哪些叶子张量或节点应该由哪个后端缓冲区类型负责分配。

实现如下：

```cpp
// 源码路径：tests/test-alloc.cpp - make_context（125-133行）
static test_context_with_graph make_context() {
    ggml_init_params params{};
    params.mem_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context *   ctx     = ggml_init(params);
    ggml_context_ptr ctx_ptr = ggml_context_ptr(ctx);
    ggml_cgraph *    graph   = ggml_new_graph(ctx);
    return { ctx, graph, std::move(ctx_ptr) };
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - make_input_1d（136-140行）
static ggml_tensor * make_input_1d(ggml_context * ctx, int64_t n_elements) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(t);
    return t;
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - make_input_with_size（143-146行）
static ggml_tensor * make_input_with_size(ggml_context * ctx, size_t size_bytes) {
    GGML_ASSERT(size_bytes % 4 == 0);
    return make_input_1d(ctx, size_bytes / 4);
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - assign_names（148-153行）
static void assign_names(ggml_context * ctx, const char * prefix = \"x\") {
    int i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        ggml_format_name(t, \"%s%d\", prefix, i++);
    }
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - get_leaf_id（156-165行）
static int get_leaf_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (strncmp(graph->leafs[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, \"leaf not found: %s\\n\", tensor_name);
    return -1;
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - get_node_id（167-176行）
static int get_node_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (strncmp(graph->nodes[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, \"node not found: %s\", tensor_name);
    return -1;
}
```

### 107.3.4 内存安全的三重校验机制

当分配器声称已经完成内存分配时，我们如何知道它没有撒谎？测试套件通过三重检验构建起信任：

第一道防线是 `check_all_allocated()`：它遍历计算图中的每一个节点，断言每个张量的 `buffer` 和 `data` 指针都不为空——这确保了没有张量漏分配。

第二道是 `check_max_size()`：它验证每个张量的实际内存范围（地址到地址+大小）是否完全落在其缓冲区类型允许的最大范围内。这里的关键是通过 `t->data - buffer_base` 计算出偏移量，然后检查 `offset + size ≤ max_size`。这相当于确认货物没有超出货架边界。

第三道更为精妙：`memory_overlap()`、`can_reuse_memory()` 和 `check_no_overlap()` 构成了图级内存安全网。前者检查两个张量是否在同一缓冲区且内存区域有重叠；后者判断在当前操作点之后，其他张量是否仍被后续节点引用（从而决定当前张量的内存是否可以安全复用）；最后一个则遍历所有张量对，仅在它们共享同一视图源时允许重叠，否则强烈断言必须满足复用条件。最关键的辅助函数是 `get_view_source()`：它沿着 `view_src` 链条一路向下，直到找到最底层的源张量——这确保了无论张量经过了多少次视图操作（如 reshape、permute、view），我们总能追溯到它真正占用的原始内存块。

通过这三重机制，测试套件不仅能验证“是否分配了内存”，更能确认“是否按照正确的方式分配了内存”。

函数实现如下：

```cpp
// 源码路径：tests/test-alloc.cpp - check_all_allocated（193-199行）
static void check_all_allocated(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - check_max_size（204-211行）
static void check_max_size(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        auto   buft     = ggml_backend_buffer_get_type(t->buffer);
        size_t max_size = ggml_backend_buft_get_max_size(buft);
        size_t offset   = (char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer);
        GGML_ASSERT(t->data >= ggml_backend_buffer_get_base(t->buffer));
        GGML_ASSERT((size_t) offset + ggml_nbytes(t) <= max_size);
    }
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - can_reuse_memory（213-231行）
static bool can_reuse_memory(ggml_cgraph * graph, int current_i, ggml_tensor * current, ggml_tensor * other) {
    if (other->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    // Check if `other` is still \"alive\", ie. an input to any node after the `current` op
    for (int i = current_i; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t == current && ggml_op_can_inplace(t->op)) {
                continue;
            }
            if (t->src[s] == other) {
                return false;
            }
            if (t->src[s] && t->src[s]->view_src == other) {
                return false;
            }
        }
    }
    return true;
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - memory_overlap（234-242行）
static bool memory_overlap(ggml_tensor * a, ggml_tensor * b) {
    if (a->buffer != b->buffer) {
        return false;
    }
    int64_t a0 = (int64_t) a->data;
    int64_t a1 = a0 + ggml_nbytes(a);
    int64_t b0 = (int64_t) b->data;
    int64_t b1 = b0 + ggml_nbytes(b);
    return a1 > b0 && b1 > a0;
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - get_view_source（244-249行）
static ggml_tensor * get_view_source(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - check_no_overlap（251-265行）
static void check_no_overlap(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        for (int j = 0; j < i; ++j) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            ggml_tensor * o = ggml_graph_node(graph, j);
            GGML_ASSERT(t != o);

            if (get_view_source(t) == get_view_source(o)) {
                continue;
            }
            if (memory_overlap(t, o)) {
                GGML_ASSERT(can_reuse_memory(graph, i, t, o));
            }
        }
    }
}
```

## 107.4 gallocr 分配器行为测试矩阵 —— 内存策略的"十三道极限考题"

### 107.4.1 缓冲区容量的极限挑战

内存分配器的首要职责是在有限资源下完成分配任务。当可用空间不足时，它必须有明确的扩容策略。`test_max_size_too_many_tensors()` 正是这一考验的开篇：我们创建了七个各占 8B 的张量（总计 56B），但将虚拟后端的单缓冲区上限设为仅 16B。理论上，前两个张量能塞进第一个缓冲区（16B），但第三个张量就已经超限——此时分配器应该自行其是，要么分配第二个 16B 缓冲区，要么直接申请一个更大的缓冲区来容纳所有内容。

测试的巧妙之处在于最后的断言：`allocated_total() <= 16 + 16`。它没有死磕具体分配了多少缓冲区（因为实现细节可能变化），而只是确保总分配量不会超过两个最大缓冲区之和——这为实现留出了灵活空间，同时仍然守住了资源不会无限制膨胀的底线。

类似地，`test_max_size_tensor_too_large()` 聚焦于另一个边界：即使总容量足够，单个大张量也可能卡在缓冲区的剩余碎片里。这里我们先分配一个 16B 张量（占满第一个缓冲区的前半段），再分配一个 8B 张量（占满后半段），随后试图通过 concat 操作生成一个 24B 张量。虽然总容量仍是 32B，但前一个缓冲区已经没有连续的 24B 空间可用——分配器必须再开辟新空间。断言 `allocated_total() <= 32 + 24` 同样允许它要么申请第二个缓冲区，要么直接升级到更大的单一缓冲区。

当单个张量本身就超过上限时，`test_tensor_larger_than_max_size()` 亮出了底牌：我们故意制造了一个 24B 张量，而后端上限仍是 16B。此时分配器别无选择，只能无视之前的上限设置，直接分配一个至少能容纳 24B 的缓冲区。精确的断言 `allocated_total() == 24` 验证了它没有多分配一 byte。

最后，`test_not_enough_chunks()` 考验的是分配器内部的碎片管理策略：当单个缓冲区被划分成固定数量的块（比如 16 个）时，会不会因为块数用尽而拒绝服务，即使总内存尚有剩余？测试中我们反复使用 add 操作累积张量，直到第 17 个输入张量需要分配时，才发现虽然总内存还有空间，但可用块已经用完。一个健康的分配器应该能够动态增加块数或通过合并空闲块来腾出空间——这正是测试所验证的。

函数如下：

```cpp
// 源码路径：tests/test-alloc.cpp - test_max_size_too_many_tensors（273-291行）
static void test_max_size_too_many_tensors() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[7];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_mul(ctx, x[0], x[1]);
    x[4] = ggml_add(ctx, x[1], x[2]);
    x[5] = ggml_add(ctx, x[3], x[0]);
    x[6] = ggml_add(ctx, x[4], x[5]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[6], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 16 + 16);
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - test_max_size_tensor_too_large（294-309行）
static void test_max_size_tensor_too_large() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);    // chunk 0, [0 , 16)
    x[1] = make_input_with_size(ctx, 8);     // chunk 0, [16, 24)
    x[2] = ggml_concat(ctx, x[0], x[1], 0);  // chunk 1, [0 , 24)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 24);
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - test_tensor_larger_than_max_size（313-325行）
static void test_tensor_larger_than_max_size() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[1], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() == 24);
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - test_not_enough_chunks（329-349行）
static void test_not_enough_chunks() {
    const int max_chunks = 16;
    const int max_size   = 8;

    dummy_backend backend      = dummy_backend_init(max_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[max_chunks + 1];
    for (int i = 0; i < max_chunks + 1; ++i) {
        x[i] = make_input_with_size(ctx, max_size);
    }
    ggml_tensor * acc = x[0];
    for (int i = 0; i < max_chunks; ++i) {
        acc = ggml_add(ctx, acc, x[i + 1]);
    }
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, acc, &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() > max_chunks * max_size);
}
```

### 107.4.2 内存复用与碎片整理

高效的分配器不仅要知道什么时候该分配新内存，更要知道什么时候能够安全复用已释放的空间。`test_view_inplace()` 展示了理想状态：通过一系列视图操作（reshape → permute → view）创建的张量链，所有张量其实共享同一块原始内存。当我们随后对最末端的视图执行原地加法（in-place add）时，理想情况下不应产生任何新分配——因为操作直接在源张量的内存上进行修改。测试通过断言 `allocated_total() <= 24` 验证了这一点（原始输入 16B + 第二个输入 8B = 24B，中间全程零额外开销）。

而 `test_reuse_and_free()` 则深入探讨了释放后内存块的命运。测试中我们先分配三个张量：x0（24B）、x1（8B）、x2（8B），随后：
- x3 = x1 + x2（复用 x1 和 x2 的空间，释放 x2）
- x4 = pad(x0, 2)（为 x0 添加 2B 边缘，需要新空间，释放 x0）
- x5 = scale(x4, 2.0)（从 x0 释放出的空间中分配给 x4 的扩展版）
- x6 = x4 + x5（复用 x4 和 x5 的空间，释放 x5）
- x7 = view(x6, 2, 8）（创建 x6 的视图）
- x8 = x3 + x7（复用 x3 和 x7 的空间）

整个过程如同仓库管理员不断腾出空间、重新上架货物。最终断言 `allocated_total() <= 40 + 32 + 32` 并非指向具体数字，而是确保即使在最坏情况下（所有操作都触发了新分配），总内存也不会失控。

碎片整理的智慧体现在 `test_merge_free_block()`：当相邻的空闲块被释放时，分配器是否能够察觉到并将它们合并成更大的可用块？测试中我们先分配三个 16B 张量，经过若干操作释放了中间的某些块，最终通过断言验证总分配量被有效控制在合理范围内——这证明空闲块确实被正确合并，避免了因小孔洞导致的“大张量分配失败”。

更深层次的优化在于 `test_prefer_already_allocated_memory()`：当既有新申请的空间，又有已释放但尚未被重新利用的块时，分配器应该优先使用哪一方？测试中我们制造了这样一个场景：x0 占用 24B，剩余 8B 空闲；经过 mean 操作后，x1 虽然在逻辑上“释放”了 x0，但实际内存布局变成了 [24B已释放][4B][4B未用]。当再次需要分配一个类似 x0 大小的张量（x2）时，理想的分配器会直接把 x2 放进那个刚刚释放的 24B 块，而不是去动用那 8B 的剩余碎片。断言 `allocated_total() <= 28` 验证了这一点——因为如果它真的复用了 24B 块，则总内存只需覆盖 x0（24B）和均值结果（4B）；若错误地去用了碎片空间，则可能需要额外分配。

函数如下：

```cpp
// 源码路径：tests/test-alloc.cpp - test_view_inplace（369-385行）
static void test_view_inplace() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[6];
    x[0] = make_input_1d(ctx, 4);                // chunk 0, [0, 16)
    x[1] = ggml_reshape_2d(ctx, x[0], 2, 2);     // view of x0
    x[2] = ggml_permute(ctx, x[1], 1, 0, 2, 3);  // view of x0
    x[3] = ggml_view_1d(ctx, x[2], 2, 4);        // view of x0
    x[4] = make_input_1d(ctx, 2);                // chunk 0, [16, 24)
    x[5] = ggml_add(ctx, x[3], x[4]);            // reuse (inplace add)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[5], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 24);
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - test_reuse_and_free（388-406行）
static void test_reuse_and_free() {
    dummy_backend backend      = dummy_backend_init(40);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_add(ctx, x[1], x[2]);        // reuse, free x2
    x[4] = ggml_pad(ctx, x[0], 2, 0, 0, 0);  // alloc new buffer, free x0
    x[5] = ggml_scale(ctx, x[4], 2.0f);      // alloc from free block
    x[6] = ggml_add(ctx, x[4], x[5]);        // reuse, free x5
    x[7] = ggml_view_1d(ctx, x[6], 2, 8);    // view
    x[8] = ggml_add(ctx, x[3], x[7]);        // reuse
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 40 + 32 + 32);
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - test_merge_free_block（409-428行）
static void test_merge_free_block(size_t max_buffer_size) {
    dummy_backend backend      = dummy_backend_init(max_buffer_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = make_input_with_size(ctx, 16);
    x[3] = ggml_mean(ctx, x[0]);
    x[4] = ggml_mean(ctx, x[1]);
    x[5] = ggml_pad(ctx, x[2], 2, 0, 0, 0);
    x[6] = ggml_add(ctx, x[3], x[4]);
    x[7] = ggml_pad(ctx, x[6], 5, 0, 0, 0);
    x[8] = ggml_add(ctx, x[5], x[7]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 32 + 24);
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - test_prefer_already_allocated_memory（432-444行）
static void test_prefer_already_allocated_memory() {
    dummy_backend backend      = dummy_backend_init(32, /*align*/ 4);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 24);  // [24b][8b unused]
    x[1] = ggml_mean(ctx, x[0]);           // [24b free][4b][4b unused]
    x[2] = ggml_mean(ctx, x[1]);           // should be allocated in the 24b block
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() <= 28);
}
```

### 107.4.3 多缓冲区类型与生命周期管理

真实系统 seldom 只有一种内存类型。为了验证分配器在异构环境中的表现，`test_multiple_buffer_types()` 人工构造了两个虚拟后端：后端 A 上限为 32B，后端 B 上限被设为 `SIZE_MAX`（实际代表“无限大”）。测试中我们让某些张量（如来自模型 A 的权重）强制绑定到后端 A，某些张量（如来自模型 B 的偏置）绑定到后端 B，而中间计算的张量则依然由 gallocr 负责分配。

这种设计迫使我们跳过默认的后端调度器，直接通过 `ggml_gallocr_reserve_n()` 手动指定：哪些叶子张量应该由哪个后端缓冲区类型分配，哪些计算节点又应使用哪种类型。通过这种显式映射，我们能够 izolated 地验证 gallocr 在多后端场景下的分发策略是否正确——例如，它是否确实把模型 A 的两个 16B 张量放进了后端 A 的缓冲区（总计不超过 32+32+24B），同时确认模型 B 的张量只走了后端 B 的路线（不超过 32+24B）。

为确保极端情况的健壮性，`test_buffer_size_zero()` 专门测试了当某个后端缓冲区类型的最大尺寸被设为零时的行为：此时即使有张量请求分配，该后端也不应产生任何实际分配（保持 0 字节占用），所有内存压力应由其他健康后端承担。测试中我们验证了后端 A（正常）确实分配了 16B，而后端 B（零尺寸）恰好保持在 0B——这证明了分配器能够敬畏零限制而不尝试越界分配。

最后，`test_reallocation()` 检验了分配器在生命周期中的自适应能力：当同一个 gallocr 实例被用于分配两个连续的计算图时，如果第二个图中某个块的需求大小发生了变化，它是否能够察觉到并相应地重新分配？测试中我们先跑了一个需要 40B 总空间的图，紧接着跑了一个理论上也只需 40B 的图——但内部结构变化导致实际需求保持不变。断言在两次分配后 `allocated_total() == 40` （未增加未减少）验证了分配器正确地复用了之前的分配器状态，并在必要时进行了真正的重新分配而非泄漏。

函数如下：

```cpp
// 源码路径：tests/test-alloc.cpp - test_multiple_buffer_types（448-503行）
static void test_multiple_buffer_types() {
    dummy_backend backend_a = dummy_backend_init(32);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX);

    auto [ctx_a, _a, ctx_a_ptr] = make_context();
    auto [ctx_b, _b, ctx_b_ptr] = make_context();
    auto [ctx, graph, ctx_ptr]  = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx_a, 16);
    a[1] = make_input_with_size(ctx_a, 16);
    assign_names(ctx_a, \"a\");

    ggml_tensor * b[2];
    b[0] = make_input_with_size(ctx_b, 24);
    b[1] = make_input_with_size(ctx_b, 4);
    assign_names(ctx_b, \"b\");

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_mul(ctx, x[0], a[0]);
    x[2] = ggml_pad(ctx, x[1], 2, 0, 0, 0);
    x[3] = ggml_mul(ctx, x[2], b[0]);
    x[4] = ggml_mean(ctx, x[3]);
    x[5] = ggml_add(ctx, x[4], b[1]);
    x[6] = ggml_pad(ctx, x[5], 3, 0, 0, 0);
    x[7] = ggml_add(ctx, x[6], a[1]);
    x[8] = ggml_scale(ctx, x[7], 2.0f);
    assign_names(ctx, \"x\");

    ggml_backend_buffer_ptr    buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, &backend_a.buffer_type));
    ggml_backend_buffer_ptr    buf_b(ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, &backend_b.buffer_type));
    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };

    // assign buffer types manually to avoid extra complexity from backend scheduler
    ggml_set_output(x[8]);
    ggml_build_forward_expand(graph, x[8]);

    GGML_ASSERT(graph->n_leafs == 5);
    int leaf_buffer_ids[5];
    leaf_buffer_ids[get_leaf_id(graph, \"a0\")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, \"a1\")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, \"b0\")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, \"b1\")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, \"x0\")] = 0;

    GGML_ASSERT(graph->n_nodes == 8);
    int node_buffer_ids[8];
    node_buffer_ids[get_node_id(graph, \"x1\")] = 0;
    node_buffer_ids[get_node_id(graph, \"x2\")] = 0;
    node_buffer_ids[get_node_id(graph, \"x3\")] = 1;
    node_buffer_ids[get_node_id(graph, \"x4\")] = 1;
    node_buffer_ids[get_node_id(graph, \"x5\")] = 1;
    node_buffer_ids[get_node_id(graph, \"x6\")] = 1;
    node_buffer_ids[get_node_id(graph, \"x7\")] = 0;
    node_buffer_ids[get_node_id(graph, \"x8\")] = 0;

    ggml_gallocr_ptr galloc(ggml_gallocr_new_n(bufts, 2));
    ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    ggml_gallocr_alloc_graph(galloc.get(), graph);

    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend_a.context->allocated_total() <= 32 + 32 + 24);
    GGML_ASSERT(backend_b.context->allocated_total() <= 32 + 24);
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - test_buffer_size_zero（506-528行）
static void test_buffer_size_zero() {
    dummy_backend backend_a    = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_b    = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);

    ggml_set_output(x[1]);
    ggml_build_forward_expand(graph, x[1]);

    int leaf_buffer_ids[1] = { 0 };
    int node_buffer_ids[1] = { 0 };

    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };
    ggml_gallocr_ptr           galloc   = ggml_gallocr_ptr(ggml_gallocr_new_n(bufts, 2));
    bool                       res1     = ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    bool                       res2     = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(res1 && res2);

    check_all_allocated(graph);
    GGML_ASSERT(backend_a.context->allocated_total() == 16);
    GGML_ASSERT(backend_b.context->allocated_total() == 0);
}
```

```cpp
// 源码路径：tests/test-alloc.cpp - test_reallocation（531-556行）
static void test_reallocation() {
    dummy_backend    backend = dummy_backend_init(32, /*align*/ 4);
    ggml_gallocr_ptr galloc;
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 24);
        x[1] = make_input_with_size(ctx, 16);
        x[2] = ggml_view_1d(ctx, x[0], 4, 0);
        x[3] = ggml_add(ctx, x[2], x[1]);
        assign_names(ctx);

        galloc = allocate_graph(graph, x[3], &backend.buffer_type);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 20);
        x[1] = make_input_with_size(ctx, 20);
        x[2] = ggml_add(ctx, x[0], x[1]);
        assign_names(ctx);
        ggml_set_output(x[2]);
        ggml_build_forward_expand(graph, x[2]);

        bool result = ggml_gallocr_alloc_graph(galloc.get(), graph);
        GGML_ASSERT(result);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
}
```

## 107.5 线程同步与图调度压力测试 —— 并发计算的"极限体能测试"

### 107.5.1 为什么专门构造“大量小算子”的计算图？

在 GPU 加速的世界里，大矩阵乘（GEMM）往往能够利用硬件流水线和缓存得到高度优化，其背后的调度开销反而在整体耗时中占比甚微。为了把放大镜对准线程同步本身的成本，我们需要一种“显影剂”——那就是大量小算子链式计算图。

`test_barrier()` 正是如此设计：它构建了一个包含 1000 层的链，每层由两次 Q4_0 量化矩阵乘（64×128 和 128×64）组成。虽然每次乘法的绝对耗时很小，但由于矩阵乘不是原子操作，其中间结果必须在线程之间同步——这意味着每经过一层，所有工作线程都必须在屏障点相遇。当层数达到 1000 时，这些微小的同步开销便被放大成了可测量的性能瓶颈，直接暴露了 `ggml_threadpool` 在高频同步场景下的延迟表现。

测试不只报告总耗时，更贴心地拆解了每节点、每轮的纳秒级开销——这相当于给调度器装上了显微镜，让我们能够看到哪怕是几十纳秒的抖动也无法遁形。

函数如下：

```cpp
// 源码路径：tests/test-barrier.cpp - test_barrier（12-86行）
static void test_barrier(int n_threads, int n_rounds) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Create graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Lots of small, parallel ops where barriers in between will dominate
    struct ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  64);
    for (int i = 0; i < 1000; i++) {
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 128);
        out = ggml_mul_mat(ctx, a, out);

        struct ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
        out = ggml_mul_mat(ctx, d, out);
    }

    ggml_build_forward_expand(gf, out);
    int n_nodes = ggml_graph_n_nodes(gf);

    // Create threadpool
    struct ggml_threadpool_params tpp  = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool* threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        fprintf(stderr, \"threadpool create failed : n_threads %d\\n\", n_threads);
        exit(1);
    }

    // The test runs with constant number of threads
    struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, threadpool);
    std::vector<uint8_t> work_data(cplan.work_size);
    cplan.work_data = work_data.data();

    std::cerr << \"graph-compute with\"
              << \"\\n n_threads: \" << n_threads
              << \"\\n   n_nodes: \" << n_nodes
              << \"\\n  n_rounds: \" << n_rounds
              << \"\\n\";

    // Warmup
    ggml_graph_compute(gf, &cplan);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i=0; i < n_rounds; i++) {
        ggml_graph_compute(gf, &cplan);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    std::cerr << \"graph-compute took \" << usec << \" usec \"
              << \"\\n \" << (float) usec / n_rounds << \" usec per-iter\"
              << \"\\n \" << (float) nsec / (n_rounds * n_nodes) << \" nsec per-node\"
              << \"\\n\";

    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
}
```

### 107.5.2 动态线程数切换的竞态检测

真实使用场景中，用户可能会在程序运行过程中动态调整线程数（比如根据当前负载或功耗策略）。`test_active()` 专门模拟了这种情况：它设定一个规律——每四次迭代（i % 4 == 0），就把线程数强制切换回 1；否则使用完整的 `n_threads`。

这种看似简单的切换实际上暗藏杀机：当线程池要从 1 线程扩展到 N 线程（或反过来收缩）时，内部的工作分配计划（cplan）必须被重新生成。如果在重建过程中有正在飞行的矩阵乘操作还在使用旧的工作数据区域（work_data），就可能发生数据竞争——一个线程还在写入，另一个线程却已经开始读取新内容了。

测试的智慧在于它不依赖于最终结果的正确性（因为即使出现竞态，最终数值在某些运气好的情况下也可能对），而是通过强制频繁切换（在这里是每四轮一次）来放大竞态窗口，使得任何不安全的实现几乎必然会在某次迭代中崩溃或产生明显错误。能够通过这个测试，就意味着线程池在调整规模时能够妥善地：停止所有正在进行的工作，安全地回收或重新分配工作数据，然后在新线程数下重建计划并恢复计算。

函数如下：

```cpp
// 源码路径：tests/test-barrier.cpp - test_active（89-135行）
static void test_active(int n_threads, int n_rounds) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Create graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Small graph with, parallel ops with barriers
    struct ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  64);
    for (int i = 0; i < 2; i++) {
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 128);
        out = ggml_mul_mat(ctx, a, out);

        struct ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
        out = ggml_mul_mat(ctx, d, out);
    }

    ggml_build_forward_expand(gf, out);
    int n_nodes = ggml_graph_n_nodes(gf);

    // Create threadpool
    struct ggml_threadpool_params tpp  = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool* threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        fprintf(stderr, \"threadpool create failed : n_threads %d\\n\", n_threads);
        exit(1);
    }

    std::cerr << \"graph-compute with\"
              << \"\\n n_threads: \" << n_threads
              << \"\\n   n_nodes: \" << n_nodes
              << \"\\n  n_rounds: \" << n_rounds
              << \"\\n\";

    // In this test we keep changing the number of threads every 4th iteration
    // to test for race conditions in that path

    for (int i=0; i < n_rounds; i++) {
        struct ggml_cplan cplan = ggml_graph_plan(gf, (i % 4) == 0 ? 1 : n_threads, threadpool);
        std::vector<uint8_t> work_data(cplan.work_size);
        cplan.work_data = work_data.data();

        ggml_graph_compute(gf, &cplan);
    }

    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
}
```

### 107.5.3 多图交替执行的缓存污染测试

现实中，一个线程池往往需要服务不止一种工作负载。`test_multi_graph()` 检验了在同一套线程池和工作数据缓冲区下，交替执行两个截然不同大小的计算图是否会导致“缓存污染”——即前一个图残留的工作数据错误地影响了后一个图的执行。

测试中我们准备了两个图：gf0 是一个较小的链（64 维，2 层），而 gf1 是一个更大的链（256 维，4 层），故意让后者的工作数据需求超过前者。随后在主循环中，我们让它们交替执行：偶数轮算 gf0，奇数轮算 gf1。如果工作数据缓冲区在切换时没有被正确清理或重新分配，那么当从 gf1 切回 gf0 时，后者可能会读取到其实不属于自己的、来自前一轮 gf1 的残余数据——这就像在厨房里用切生肉的砧板直接处理熟食，交叉污染在所难免。

测试通过在每次图计算后强制重建 cplan（并因此重新分配 work_data）来避免这种情况。能够通过此测试，就意味着调度器具备在图切换时彻底隔离工作状态的能力：无论上一个图多么庞大或复杂，它都能确保下一个图从一个干净的状态开始，不受历史遗留数据的干扰。

函数如下：

```cpp
// 源码路径：tests/test-barrier.cpp - test_multi_graph（138-204行）
static void test_multi_graph(int n_threads, int n_rounds) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Create graphs
    struct ggml_cgraph * gf0 = ggml_new_graph(ctx);
    {
        // Small graph with parallel ops with barriers
        struct ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  64);
        for (int i = 0; i < 2; i++) {
            struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 128);
            out = ggml_mul_mat(ctx, a, out);

            struct ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
            out = ggml_mul_mat(ctx, d, out);
        }

        ggml_build_forward_expand(gf0, out);
    }

    struct ggml_cgraph * gf1 = ggml_new_graph(ctx);
    {
        // Small graph with parallel ops with barriers
        // Use larger tensors to make sure work_data size is larger than gf0
        struct ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  256);
        for (int i = 0; i < 4; i++) {
            struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 128);
            out = ggml_mul_mat(ctx, a, out);

            struct ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 256);
            out = ggml_mul_mat(ctx, d, out);
        }

        ggml_build_forward_expand(gf1, out);
    }

    // Create threadpool
    struct ggml_threadpool_params tpp  = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool* threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        fprintf(stderr, \"threadpool create failed : n_threads %d\\n\", n_threads);
        exit(1);
    }

    std::cerr << \"graph-compute with\"
              << \"\\n gf0 n_nodes: \" << ggml_graph_n_nodes(gf0)
              << \"\\n gf1 n_nodes: \" << ggml_graph_n_nodes(gf1)
              << \"\\n   n_threads: \" << n_threads
              << \"\\n    n_rounds: \" << n_rounds
              << \"\\n\";

    // In this test we keep changing the number of threads every 4th iteration
    // and we compute two graphs back to back to test graph frequent graph switching

    for (int i=0; i < n_rounds; i++) {
        struct ggml_cplan cplan0 = ggml_graph_plan(gf0, (i % 4) == 0 ? 1 : n_threads, threadpool);
        std::vector<uint8_t> work_data0(cplan0.work_size);
        cplan0.work_data = work_data0.data();

        struct ggml_cplan cplan1 = ggml_graph_plan(gf1, (i % 4) == 0 ? 1 : n_threads, threadpool);
        std::vector<uint8_t> work_data1(cplan1.work_size);
        cplan1.work_data = work_data1.data();

        ggml_graph_compute(gf0, &cplan0);
        ggml_graph_compute(gf1, &cplan1);
    }

    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
}
```

### 107.5.4 性能基准与日志压力

除了核心算法验证，测试套件也不忘记底层基础设施的健康状况。`test_barrier()` 本身就是一个性能基准：通过报告每节点纳秒耗时，它为不同线程数下的同步开销建立了可比较的基线。而另一个文件 `test-log.cpp` 则专门压力测试了日志系统——在这个多线程并发写入的试炼场中，8 个线程各自狂飙 1000 条随机类型的日志（信息、警告、错误、调试），并且在写 Process 每隔几次就随机切换时间戳和前缀的启用状态。

这相当于让日志系统在一场密集的多人快速说话大会中，不断被要求切换报告格式（比如有时候带时间戳有时候不带，有时候带前缀有时候不带）。能够在这般嘈杂且频繁变配的环境下不丢字、不乱序、不崩溃，才算真正经得起考验的日志实现。

### 107.5.5 测试调度中心

所有这些压力测试由 `test-barrier.cpp` 中的 `__main__` 函数统一调度。它首先根据硬件并发能力（但不超过 4）设定默认线程数，并把轮数设为 100。用户可以通过命令行参数覆盖这两个值。随后它依次执行：
1. `test_barrier(n_threads, n_rounds)` —— 基础屏障开销测试
2. `test_active(n_threads, n_rounds * 100)` —— 动态线程切换（这里放大了轮数以增加切换频率）
3. `test_multi_graph(n_threads, n_rounds * 10)` —— 多图交替（同样放大轮数）

这种梯度设计确保了我们不仅测试了基础同步开销，还验证了在更苛刻的动态切换和频繁图交替场景下的表现。

函数如下：

```cpp
// 源码路径：tests/test-barrier.cpp - __main__（207-222行）
int main(int argc, char *argv[]) {
    int n_threads = std::max(1, std::min(4, (int) std::thread::hardware_concurrency()));
    int n_rounds  = 100;

    if (argc > 1) {
        n_threads = std::atoi(argv[1]);
    }

    if (argc > 2) {
        n_rounds  = std::atoi(argv[2]);
    }

    test_barrier(n_threads, n_rounds);
    test_active(n_threads,  n_rounds * 100);
    test_multi_graph(n_threads,  n_rounds * 10);

    return 0;
}
```

## 107.6 模型架构与后端交叉验证 —— 推理引擎的"跨平台一致性测试"

### 107.6.1 确定性随机权重的“空白模型”工厂

想要在不下载真实模型的情况下验证数值一致性，就需要一种能够按需生成“空白但可运行”模型的工厂——这就是 `get_gguf_ctx()` 的用武之地。它接受一个模型架构（如 LLM_ARCH_LLAMA）和一个布尔值（是否启用 MoE），然后返回一个装有随机权重的 GGUF 上下文。这些权重虽然是随机的，但却是确定性的：相同的张量名称和种子必然会产生相同的浮点值。

这种确定性来自 `set_tensor_data()`：它把张量的名称喂进哈希函数，再加上一个全局种子，得到一个可重现的随机数生成器种子。然后它根据张量的数据类型（F32 或 F16）从正态分布中采样数值，并写入后端。这意味着不管你在什么机器上、什么时候运行测试，只要架构和种子相同，生成的 GGUF 文件在字节层面就是完全相同的——这为后续的跨后端比较提供了不可或缺的可复现性基石。

有了这个 GGUF 上下文，`get_model_and_ctx()` 就能够封装出从 GGUF 创建 llama 模型和上下文的全流程，而 `get_logits()` 进一步包装了编码/解码和 logits 提取。这样一来，整个推理过程就像被装进了一个黑箱：输入是确定性的 token 序列（由 `get_tokens()` 生成），输出是一维 float 向量（所有时间步的 logits 拼接），而内部所有 llama_model 与 GGUF 的交互细节都被屏蔽了起来。

### 107.6.2 NMSE 作为数值精度的“测距仪”

当我们把同一个模型丢进 CPU 和 GPU 后端时，即使理论上应该得到完全相同的结果，实际运行中也常常会出现微小的数值偏差——这既可能源于浮点运算的非结合性，也可能来自不同后端的优化路径（比如 GPU 可能用了更激进的融合指令）。为了量化这种偏差是否还在可以接受的范围内，llama.cpp 采用了归一化均方误差（NMSE）作为度量标准。

NMSE 的计算直白易懂：先算出两个向量（比如 CPU logits 和 GPU logits）逐元素差的平方和（这就是均方误差 MSE），然后用第一个向量自身的平方和来归一化。结果是一个无量纲比例：值越小，表示两个向量越相似；值越大，表示偏差越严重。在测试中我们把阈值设定为 1e-4——也就是说，如果 CPU 和 GPU 之间的 NMSE 小于这个值，我们就认为它们在数值上是一致的；否则就判定为失败。

这个阈值并非随意设定：1e-4 相对应的是约 0.01% 的相对误差。考虑到 FP16 相比 FP32 在精度上的固有限制，以及某些后端优化（如 SDPA 实现）可能带来的轻微数值漂移，这个值在严格性和可实现性之间取得了平衡。如果把阈值调得更严（比如 1e-6），则很可能导致大量原本可以接受的后端测试失败——特别是那些严重依赖低精度格式或激进算子融合的 GPU 后端。

我们只在 T5 架构上开启编码路径（对应代码中的 `const bool encode = arch == LLM_ARCH_T5;`）这是因为在标准解码器架构中，模型通常只处理自回归生成：它接受过去的 token，预测下一个 token 的概率分布。而在 T5 这样的 encoder-decoder 架构中，推理过程被清晰地分成了两个阶段：首先是编码阶段，模型把完整的输入序列转换为一组上下文表示；随后是解码阶段，基于这些表示和过去生成的 token 来预测下一个 token。由于我们的测试流程是直接喂入 token 并读取输 logits，只有在编码器需要显式处理完整输入序列的架构（如 T5）上，我们才必须走编码路径；而在纯解码器架构上，等效的信息已经可以通过初始状态隐含地传递进去了。

最后一个细节体现了测试的严谨性：在做往返一致性（Roundtrip）检验时，我们不仅比较了设备 logits 与 CPU logits 的 NMSE，更加入了一个字节级的逐字节精确比对：`logits_roundtrip[i] != logits_dev[i]`。这一步看似苛刻，其实是为了捕获那些在数值上可能“接近”但事实上由于内存布局或序列化细微差异而导致的不一致。比如，如果在保存到 GGUF 时某个权重被错误地写进了偏移一个字节的位置，或者在加载时发生了错误的字节序转换，那么即使最终预测结果在统计上看起来没问题，这个逐字节比对也会立刻红旗——因为它直接检验了比特级的保真度。

### 107.6.3 MoE 架构的强制/可选开关

混合专家（MoE）架构增加了额外的复杂度：它不是简单地在每层用一个前馈网络，而是维护多个“专家”网络，并通过一个门控网络动态选择哪些专家应该被激活。这种设计意味着如果模型声称自己是 MoE 但底层实现却没有专家路由逻辑，那么在推理时就会发生未定义行为——比如尝试访问不存在的专家权重。

为了在测试中正确区分不同架构的需求，我们引入了两个判定函数：`moe_mandatory()` 和 `moe_implemented()`。前者用来回答“这个架构是不是必须使用 MoE 模式？——也就是说，如果我不启用专家路由，模型根本没法跑？”；后者则问“这个架构是不是至少支持 MoE 变体？——也就是说，无论是强制还是可选，它是否具备运行 MoE 的能力？”

具体来说，`moe_mandatory()` 列出了诸如 LLAMA4、GROK、QWEN2MOE/QWEN3MOE 等 30+ 架构：这些模型在设计时就假设了专家路由的存在，若强行以密集模式运行则会缺失关键组件。而 `moe_implemented()` 的范围更广：它不仅包括所有强制 MoE 的架构，还额外纳入了那些虽然可以以密集模式运行但同样具备 MoE 实现的架构，比如原始 LLaMA、REFACT、MINICPM 等——这使得测试矩阵能够智能地生成：对于强制 MoE 的架构，我们只测试 MoE 模式；对于可选但已实现的架构，我们则同时测试密集和 MoE 两种模式；而对于既不强制也不支持的架构，则完全跳过 MoE 相关测试。

### 107.6.4 保存-加载往返一致性（Roundtrip）

即使模型在推理过程中数值一致，也不能保证它的持久化机制是可靠的。也许在把权重写入 GGUF 文件时发生了截断，也许在读取时发生了错误的解码，又或者在分片处理时丢失了某些张量。为了从根本上验证序列化层的健康，测试套件专门设计了保存-加载往返一致性检验：先把模型保存到 GGUF 文件（或内存中的临时文件），然后重新加载它，最后用相同的输入再次推理并比较输出。

在 `test_backends()` 中，这一步是通过 `tmpfile()` 创建一个匿名临时文件来实现的（在 Windows 可能需要管理员权限，若失败则跳过该步骤）。我们先把模型和上下文打包进一个 `llama_model_saver`，调用 `add_kv_from_model()` 和 `add_tensors_from_model()` 将所有元数据和权重写入文件，然后 `save()` 到临时句柄。随后 `rewind()` 将文件指针拉回开头，以便通过 `get_model_and_ctx()` （这次传入文件指针而非 GGUF 上下文）重新构建模型和上下文对象。

重新加载后，我们再次喂入相同的 token 序列并提取新 logits。此时如果有两重断言能够通过，那就意味着持久化链是健康的：
- 首先是 NMSE 检验：重新加载后的 logits 与直接在设备上推理得到的 logits 在数值上必须一致（阈值仍为 1e-4）；
- 其次是逐字节比对：两段 logits 在内存中的每一个字节都必须完全相等，这能够捕获任何在序列化或反序列化过程中引起的比特级偏差，比如由于端转换不当导致的高位字节被错误地放到了低位。

能够通过这两重检验，就意味着从内存权重到 GGUF 文件、再从 GGUF 文件回到内存权重的完整往返过程，不仅没有改变模型的预测行为，更没有在比特级别上损失任何信息。

### 107.6.5 测试入口

所有这些验证由 `tests/test-llama-archs.cpp` 中的 `main()` 函数统一编排。程序接受两种互斥的运行模式：
- 如果用户通过 `-o/--out` 指定了输出目录，则进入“模型保存”模式：遍历所有支持的架构，为每个架构（密集/MoE 视情况而定）生成对应的 GGUF 文件并保存到指定目录；
- 否则进入默认的“后端一致性测试”模式：执行我们上面详述的跨后端 NMSE 对比和往返验证。

这种设计让同一个二进制既能用来生成测试用的模型制品，又能用来验证这些制品在不同硬件上的行为一致性——真正做到了“一模多用”。

## 107.7 多线程并发推理与资源管理 —— 高压环境下的“稳定性试炼”

### 107.7.1 多模型 × 多上下文的笛卡尔积压测

真实的服务部署从不只跑一个模型或一个上下文。为了验证 llama.cpp 在高并发场景下的稳定性，`test-thread-safety.cpp` 采取了笛卡尔积式的压测策略：它不是简单地拉满线程数去跑同一个模型，而是为每种可用的硬件后端（GPU 和 CPU）分别准备了模型的副本，然后让每个模型副本再派生出多个上下文去并行推理。

具体来说，程序首先枚举所有可用的后端设备，并挑出其中的 GPU 设备（忽略 CPU 和其他类型）。设想我们有 N 张 GPU 卡，那么就需要准备 N + 2 个模型副本——其中前 N 个对应每张 GPU 卡的独享使用（split_mode = LLAMA_SPLIT_MODE_NONE，并绑定到特定设备），第 N+1 个代表纯 CPU 运行（main_gpu = -1），而最后一个（第 N+2 个）则测试层级分割模式（split_mode = LLAMA_SPLIT_MODE_LAYER，即在支持的情况下把某些层卸载到 GPU）。

对于每个这样准备好的模型副本，我们再根据用户配置的 `n_parallel` （至少为 1）来决定需要派生多个上下文。每个上下文都在自己的独立线程中生命周期完整地执行一次推理任务：从 tokenize 输入提示词，到解码生成指定数量的新 token，最后输出结果。整个过程中，只要有任意一个线程因为资源分配失败、解码错误或其他非预期情况而提前退出，都会通过共享的原子标志位将失败状态向上传递。

### 107.7.2 原子标志位的失败聚合机制

在多线程测试中，最怕的就是某个线程出错却悄悄退出，而主线程仍在乐观地等待结果——这会导致测试误以为一切正常，其实已经悄悄丢失了关键错误信息。为了避免这种情况，`test-thread-safety.cpp` 并没有选择在第一个出错的线程上直接 `abort()` 或返回错误码，而是采用了一个更为温和且信息完整的策略：它声明了一个 `std::atomic<bool> failed` 变量（初始为 false），并让每个工作线程在发现任何问题时将其设为 true。

使用原子类型的好处在于它保证了对这个布尔值的读写在多线程环境下是安全的——没有锁的开销，也没有教条式的竞态风险。主线程在创建完所有工作线程后，会通过 `join()` 一个一个等待它们完成；只有当所有线程都平安结束后，它才会检查 `failed` 的最终值。如果此时仍为 false，则说明全程无人掉线；如果为 true，则表示至少有一次有线程遇到了问题。

这种设计的另一个优势是它允许所有线程完成它们本该做的工作——哪怕是出错的线程，也会在设置失败标志之后继续执行直到自然结束（除非是立即导致段错误的致命故障）。这意味着当测试结束时，我们不仅知道“有人出错了”，而且因为没有强行中断，日志、内存状态和其他副作用有机会被完整地表现出来，从而为后续排查提供了更丰富的现场信息。

值得一提的是，测试中真正可能产生共享状态的其实很少：每个线程内部都有自己独立的采样器（sampler）和批次对象（batch），它们是在线程启动时作为闭包捕获的模型指针的副本而被创建的。这意味即使多个线程共享同一个模型实例，它们的采样状态（比如随机种子的进度、Top-K 缓存等）也是互不干扰的——因为这些状态都封装在了线程局部的对象里。当然，模型本身的只读权重（以及可能的 KV 缓存如果被共享配置）才是真正的共享资源，但只要这些资源在访问时是线程安全的（就像 llama.cpp 的设计所保证的那样），就不需要额外的同步措施。

### 107.7.3 优雅的中断与清理

稳定性不仅体在于“正常情况下不挂掉”，更体在于“异常情况下能否安全退出”。`test-autorelease.cpp` 和 `test-model-load-cancel.cpp` 两个测试专门检验了这两方面。

前者测试了最基础的生命周期清理：在一个独立线程中，它执行了完整的后端初始化 → 模型加载 → 上下文创建 → 上下文释放 → 模型释放 → 后端释放这一闭环流程，然后立即让线程退出。能够通过此测试，就意味着即使在非主线程的环境下，llama.cpp 也没有留下全局状态残留（比如未注销的后端或悬挂的资源句柄），而且其析构函数在被异步调用时同样能够正确地释放所有占用的资源。

后者则聚焦于模型加载过程中的可取消性：想象一下用户正在下载一个几十吉布的大模型， soudain 意识到自己选错了——此时如果强行让用户眼睑睁看着进度条跑到 100% 再结束，显然是不友好的体验。 llama.cpp 的解决方案是在模型加载函数中预留了一个进度回调（progress_callback）接口：用户可以提供一个 lambda，在每次读取一定数据后被调用；如果该函数返回 true，则加载过程会被立即中断。

在 `test-model-load-cancel.cpp` 中，我们人为设定了这样一个回调：一旦加载进度超过 50% （比如已经读取了半个文件），就立即返回 true 以触发取消。测试的验证点有两个：首先是加载函数必须返回 nullptr（表示因取消而未成功创建模型对象）；其次是在这次取消操作之后，必须不存在内存泄漏——换句话说，已经分配的资源（如已读取的文件缓冲区或部分解析的数据结构）都应该被妥善地归还，而不是因为中断而变成了“孤儿”。

能够通过此测试，就说明 llama.cpp 在面对用户主动中断时，不但不会崩溃或挂住资源，而且能够做到“刹车稳”：就像一列高速列车在发现前方有障碍时，不仅能够依靠制动系统在安全距离内停下来，而且不会因而脱轨或留下散落的车厢。

### 107.7.4 特殊场景下的行为

值得一提的是，即使在极端情况下，测试套件也能够给出明确的结论。比如当只有一个 CPU 而没有任何 GPU 可用时，`test-thread-safety.cpp` 会如何表现？此时后端设备枚举结果中 GPU 列表为空，所以 `gpu_dev_count` 为 0，进而 `num_models = 0 + 1 + 1 = 2`——也就是说，我们会准备两个模型副本：一个对应纯 CPU 运行（因为 main_gpu = -1），另一个则测试层级分割模式（虽然在纯 CPU 环境下可能退化为等效的密集模式）。每个模型副本仍会派生出 `n_parallel` 个上下文去并行推理。虽然此时失去了“多GPU协同”的压测维度，但单机多线程并发推理的稳定性验证仍然是有效的——毕竟，即使是在 CPU 单核环境下，多个上下文如果不加以保护也一样会在共享资源上产生竞争。

同理，在 `test-barrier.cpp` 中的 `test_active` 函数里，如果我们把切换频率从 “每四轮一次” 调整为 “每两轮一次”（即把条件从 `(i % 4) == 0` 改为 `(i % 2) == 0`），测试会变得更加严格——不是因为每次切换本身更难，而是因为竞态窗口变得更频繁了。当线程数在更短的间隔内反复变动时，工作线程有更少的机会去完成实际计算，而更多的时间被浪费在不断地重建 cplan 和重新分配 work_data 上。这种情况下，任何线程池在调度切换时的不安全实现（比如在未同步的情况下修改共享状态）就更容易被放大和捕捉到。

## 107.8 KV 缓存碎片整理 —— 运行时状态的"外科手术"

### 107.8.1 人为制造 KV 缓存碎片

在长期运行的推理会话中，KV 缓存（用于存储历史 key 和 value 以避免重复计算）往往会呈现出碎片化的状态——这就像一个停车场，由于车辆一直在以不规律的顺序进出，最终导致虽然总空位数量充足，但却没有足够大的连续空间来停下一辆大货车。

`test-state-restore-fragmented.cpp` 正是通过人为制造这种碎片场景来测试系统的恢复能力。测试中我们启动了三个并行序列（n_parallel = 3），并让它们交错地喂入 token：序列 0 的第一个 token，然后是序列 1 的第一个 token，接着是序列 2 的第一个 token，再回到序列 0 的第二个 token，如此循环。这种交错解码的结果是，在 KV 缓存中，三个序列的状态被物理地交织在一起：比如位置 0 处可能存着序列 0 的状态，位置 1 处是序列 1 的状态，位置 2 处是序列 2 的状态，位置 3 处又回到了序列 0 的状态……以此类推。

随后我们故意删除掉了中间的序列 1：通过调用 `llama_memory_seq_rm()` 并指定序列 ID 为 1，我们把该序列在所有时间步上的状态都从缓存中抹除了。此时 KV 缓存的状态变成了：位置 0：序列 0，位置 1：（空），位置 2：序列 2，位置 3：序列 0，位置 4：（空），位置 5：序列 2……也就是我们所说的 “0.2.0.2.” 模式——总有效状态依然存在，但被有效地切成了小块，中间夹着无法使用的空洞。

### 107.8.2 非连续槽位的状态恢复

有了这个人造的碎片缓存，测试接下来尝试执行曾经被认为是“危险操作”的事情：把之前删掉的序列 1 的状态写回去。在修复之前，如果你尝试这样做，系统很可能会因为找不到足够大的连续空闲槽而失败——毕竟它在寻找时可能只看了能否“一口气”分配出所需大小的连续内存，而忽略了其实可以通过“拼凑”多个不连续的小块来达到同样效果的可能性。

修复之后，`llama_state_seq_set_data()` 函数获得了支持非连续槽位写入的能力。它不再强制要求目标序列的所有状态必须落在一块连续的内存区域上，而是允许它们被分散写入到多个 previamente 释放出的、现在可用的小空间里——就像把一辆被拆开运输的家具，虽然零件分散在不同的货车里，但只要在目的地都收齐了，就还是能够重新组装成原来的样子。

测试的验证过程是完整的：我们首先通过 `llama_state_seq_get_data()` 把序列 1 在删除前的完整状态读取到一个缓冲区里；然后删除序列 1 制造碎片；接着尝试用这个缓冲区的内容去恢复序列 1 的状态到如今碎片化的缓存中；最后为了确认恢复不仅是在纸上成功，而是真正可以使用的，我们喂入一个新的 token 并尝试解码——如果此时能够正常生成输出且没有报错，那就意味状态恢复不仅在逻辑上完整，而且在功能上可用。

能够通过这个测试，就说明 llama.cpp 的 KV 缓存管理不仅能够应对理想中的连续分配场景，而且在真实世界中常见的碎片化状态下，仍然具备将状态安全写回并后续使用的能力——这对于长期运行的会话（比如多轮对话或长文本生成）而言，是一项基本的健康保证。

### 107.8.3 测试调度中心

这个测试的入口同样集中在 `tests/test-state-restore-fragmented.cpp` 中的 `__main__` 函数里。它首先按照惯例解析命令行参数（虽然本测试未使用任何自定义参数，但保留了这一接口以保持与其他测试工具的一致性），然后执行标准的初始化流程：后端初始化 → 根据配置创建模型和上下文。

随后是我们上面所述的完整流程：tokenize 固定提示词（这里用的是 70 个全为 1 的 token ID） → 交错解码三个序列以制造共享 KV 缓存 → 保存序列 1 的状态 → 删除序列 1 制造碎片 → 尝试将状态写回碎片化缓存 → 验证解码功能正常。每一步都配有详细的日志输出和错误检查，确保一旦出现偏差，就能够精准定位是哪一个环节出了问题。

## 107.9 日志系统压力测试

### 107.9.1 设计意图

日志系统虽然不直接参与模型计算，但却是开发者和运维人员了解程序内部状态的重要窗口。在高并发场景下，如果日志库本身不能经受住压力，那么即使引擎其余部分运行正常，我们也可能因为日志丢失、格式错乱或甚至因日志写入导致的段错误而失去对系统行为的可见性。

为了验证其在压力下的表现，`test-log.cpp` 设计了一个简单而残酷的场景：启动 8 个线程（这个数量既能显著产生竞争，又不会过度压垮普通测试机器），每个线程独立地循环写入 1000 条日志消息。这些消息不是单一类型，而是在四个等级（信息 INF、警告 WRN、错误 ERR、调试 DBG）中随机选择——这样可以测试日志库在处理不同 severity 时的分支是否正确。

更具挑战性的是，在写日志的过程中，程序不会死死锁定一种输出格式。相反，在每写完若干条日志后，它会掷一下“硬币”：如果随机数落在 0-4 之间（概率 50%），就切换一次时间戳的启用状态；如果再次落在同样范围内，则切换前缀的启用状态。这相当于让日志系统在工作中不断被要求改变自己的“礼仪”——有时候要报告 précise 时间，有时候不要；有时候要加上模块前缀进行过滤，有时候不要。能够在这种频繁变配且高并发的环境下依然保持正确的输出顺序、不丢字节、不崩溃，才算是真正经得起考验的日志实现。

### 107.9.2 实现细节

从实现角度看，这个测试的美妙之处在于它的简洁。主函数 `__main__` 仅做了三件事：首先声明一个常量 `n_thread = 8` 来固定线程数；然后创建一个长度为 8 的 `std::thread` 数组；最后用一个循环为每个线程赋值一个 lambda 表达式——这个闭包里才是真正的工作逻辑。

在每个工作线程中，我们又看到一个熟悉的模式：固定消息数量 `n_msg = 1000`，然后进入一个从 0 到 999 的计数循坏。在每次迭代中，它通过 `std::rand() % 4` 随机选择一个日志等级（0 对应信息，1 对应警告，等等），随后用相应的宏（`LOG_INF`、`LOG_WRN`、`LOG_ERR`、`LOG_DBG`）打印出一条带有线程索引和当前计数的消息——比如 “Thread 3: 42” 表示第三个线程正在处理它的第 42 条消息。

紧跟着这个日志输出，是那两个可选的格式切换语句：如果 `rand() % 10 < 5`（又是 50% 的概率），那么就分别调用 `common_log_set_timestamps()` 和 `common_log_set_prefix()` 来切换时间戳和前缀的显示状态，参数同样是 0 或 1 的随机值（代表关闭或开启）。值得注意的是，这两个调度作用于全局日志上下文（通过 `common_log_main()` 获得），也就是说所有线程共享同一套日志配置——这正是导致需要同步保护的根源：如果一个线程正在写日志而另一个线程就在此时修改了输出格式，那么很可能出现一条日志其一半带时间戳、其一半不带的混乱情况。

能够在这个设置下通过测试，就说明 llama.cpp 的日志库内部已经做好了必要的同步保护——无论是对日志计数器的递增、对格式开关位的修改，还是对实际文件/控制台的写操作，都被恰当地保护了起来，以至于即使在多线程乱序调用且配置频繁变动的环境下，输出依然是井然有序、可读且完整的。

函数如下：

```cpp
// 源码路径：tests/test-log.cpp - __main__（5-32行）
int main() {
    const int n_thread = 8;

    std::thread threads[n_thread];
    for (int i = 0; i < n_thread; i++) {
        threads[i] = std::thread([i]() {
            const int n_msg = 1000;

            for (int j = 0; j < n_msg; j++) {
                const int log_type = std::rand() % 4;

                switch (log_type) {
                    case 0: LOG_INF(\"Thread %d: %d\\n\", i, j); break;
                    case 1: LOG_WRN(\"Thread %d: %d\\n\", i, j); break;
                    case 2: LOG_ERR(\"Thread %d: %d\\n\", i, j); break;
                    case 3: LOG_DBG(\"Thread %d: %d\\n\", i, j); break;
                    default:
                        break;
                }

                if (rand() % 10 < 5) {
                    common_log_set_timestamps(common_log_main(), rand() % 2);
                    common_log_set_prefix(common_log_main(), rand() % 2);
                }
            }
        });
    }

    for (int i = 0; i < n_thread; i++) {
        threads[i].join();
    }

    return 0;
}
```

## 107.10 小结

在这一章中，我们深入探讨了 llama.cpp 测试体系中那些不那么聚焦但同样关键的“长尾”组件——从内存分配器的精密调度到多后端数值一致性，从线程安全到模型加载取消，从 KV 缓存碎片恢复到日志系统的并发压力。这些测试就像是引擎底盘上的传感器和应变片，虽然不像发动机或变速箱那样显眼，却在保证整车能够在颠簸路面、极端天气和长途旅行中安全可靠地运行方面发挥着不可替代的作用。

首先我们理解了虚拟后端（dummy_backend）如何作为可编程的“内存沙盘”，让我们能够在不依赖真实硬件的情况下精确控制内存容量和对齐约束，专注观察 gallocr 分配器在各种极端工况下的决策逻辑。接着我们看到，这个分配器不仅知道什么时候该分配新内存，更明白什么时候能够安全复用已释放的空间——通过缓冲区分裂、碎片整理和优先重用已释放块的策略，它在有限资源下最大化了内存利用率。我们还验证了它在多缓冲区类型、生命周期重新分配以及零尺寸边界情况下的表现。

在并发维度上，线程池屏障测试通过大量小算子链式图放大了同步开销，直接暴露了调度器在高频同步场景下的延迟表现；动态线程数切换测试则模拟了用户在运行中调整线程数的真实场景，验证了线程池在规模变化时能够妥善地重建工作计划而不引发数据竞争；多图交替测试进一步检查了调度器在切换不同负载时是否能够彻底隔离工作状态，防止历史遗留数据的污染。

跨后端一致性测试则为我们提供了一套完整的工具链：从使用确定性随机权重生成“空白但可运行”的 GGUF 模型，到采用 NMSE 作为数值精度的测距仪（阈值 1e-4 在严格性和可实现性之间取得了平衡），再通过保存-加载往返验证确保序列化层不损比特，以及利用 MoE 强制/可选判定函数智能生成测试矩阵。值得一提的是，逐字节比对的加入使得我们能够捕获那些在数值上可能“接近”但事实上由于内存布局或序列化细微差异而导致的不一致。

多线程并发推理测试通过笛卡尔积式的压测策略（为每个 GPU/CPU 加载模型副本，再派生多个上下文并行推理）验证了系统在多实例、多设备、多上下文同时运行时的稳定性，而原子标志位的失败聚合机制则确保了即使有线程出错，也不会导致提前退出和信息丢失。优雅的中断与清理测试（如 autorelease 和模型加载取消）进一步确认了资源在异常路径下能否被正确释放，而 KV 缓存碎片恢复测试则证明了状态管理能够在真实世界常见的碎片化场景下依然正常工作。

最后，日志系统压力测试用 8 线程×1000 条随机日志及随机格式切换验证了其在高压环境下的线程安全性和健壮性——因为即使引擎其余部分运行得再好，如果连日志窗口都模糊不清，我们又怎能真正了解它内部发生了什么？

设计取舍分析：
测试套件在设计时做了以下关键取舍：
- **虚拟后端 vs 真实硬件**：选择虚拟后端以获得可重复、可控的测试环境，牺牲了一些硬件特定行为的覆盖，但获得了精确的内存分配验证能力。
- **确定性随机数据 vs 真实模型权重**：使用哈希生成的确定性随机权重确保测试可复现，但无法捕获真实模型中的结构性数值特性；这种取舍优先考虑了测试的稳定性和可调试性。
- ** NMSE 阈值设置 (1e-4)**: 在严格性（捕获真实数值偏差）和可实现性（考虑 FP16 限制和后端优化）之间取得平衡；更严格的阈值会增加误报，更宽松的阈值会漏掉真实问题。
- ** 逐字节比对的加入**: 尽管增加了测试的严格性，但这是必要的，以捕获序列化过程中的比特级错误，这些错误在纯数值比对中可能被掩盖。
- ** 笛卡尔积并发测试**: 为每个设备准备模型副本并派生多个上下文，虽然增加了测试复杂度，但这是验证多实例并发安全性的必要方法。
- ** 原子失败标志**: 选择使用 `std::atomic<bool>` 而不是立即退出，以收集完整的错误信息，这增加了测试的诊断价值，但略微延长了测试时间。

概念 | 解释
------|------
dummy_backend / dummy_backend_context | 虚拟后端沙盘，纯内存模拟硬件，可配置容量/对齐约束，专注测试分配策略逻辑
dummy_backend_buffer_type_* | 缓冲区类型接口家族：名称/对齔/大小/主机标识，定义虚拟后端的内存属性边界
dummy_backend_buffer_* | 缓冲区操作接口：分配/释放/基地址/初始化/读写清空，全空实现剥离硬件依赖
ggml_backend_buffer | GGML 缓冲区抽象，虚拟后端通过空接口实现记录分配行为，剥离真实硬件依赖
ggml_gallocr / gallocr | GGML 全局分配器，负责计算图内存的自动布局：缓冲区分裂、碎片整理、张量复用
make_context / make_input_1d / make_input_with_size | 测试夹具工厂：快速构建标准化 ggml 上下文与输入张量，确保测试一致性
assign_names / get_leaf_id / get_node_id | 图节点标识工具：统一命名与索引查找，支撑多缓冲区手动映射与调试
allocate_graph | gallocr 分配入口封装：设置输出、构建前向图、执行内存分配的一站式辅助
check_all_allocated / check_max_size | 分配完备性校验：确保所有张量已分配缓冲区且不超出后端容量限制
can_reuse_memory() | 引用生命周期分析：检查后续节点是否仍引用某张量，决定内存能否安全复用
memory_overlap() / check_no_overlap() | 内存安全警卫：严格校验张量区间是否交叠，仅允许声明 in-place 的算子复用
get_view_source() | 视图链溯源：沿 view_src 回溯到最终源张量，确保视图不占用新内存
缓冲区分裂（test_max_size_too_many_tensors） | 当缓冲区容量耗尽时，gallocr 自动分配新缓冲区或更大缓冲区
碎片整理（merge_free_block） | 相邻空闲块自动合并为更大可用块，减少碎片，提升大对象分配成功率
test_prefer_already_allocated_memory | 优先复用已释放内存块，减少真实分配开销，验证分配器的空间局部性优化
test_multiple_buffer_types | 多缓冲区类型手动映射测试，验证跨设备 leaf/node 分配策略的正确性
test_reallocation | gallocr 复用场景：不同图复用同一分配器时，自动检测块大小变化并重新分配
test_barrier（线程池压力测试） | 1000 层链式小算子图强制高频线程同步，暴露屏障开销的 ACM 复杂度
test_active（动态线程切换） | 每4轮在1线程和n线程间切换，验证 cplan 重建与数据竞争的线程安全性
test_multi_graph（多图交替） | 不同大小图交替计算，验证 work_data 缓冲区在图切换时的隔离正确性
set_tensor_data / get_tokens | 确定性测试数据生成器：基于名称哈希的随机权重与随机 token 序列，保证可复现性
get_gguf_ctx() | 架构-权重解耦的「空白模型」工厂，用随机权重+GGUF 元数据构造全部架构的可测试模型
get_model_and_ctx / get_logits | 模型-上下文创建与推理结果提取封装，屏蔽 llama_model 与 GGUF 的底层交互细节
NMSE（归一化均方误差） | 跨后端数值精度的度量衡：CPU vs GPU logits 对比，阈值 1e-4 判定一致性
moe_mandatory / moe_implemented | MoE 架构分类器：标记哪些架构必须启用专家路由、哪些可选，驱动测试矩阵生成
save_models / test_backends | 模型保存与后端验证双模式：支持批量生成 GGUF 或执行跨后端 NMSE 对比
往返一致性（Roundtrip） | 模型保存→回载→推理，验证序列化/反序列化无损，确保保存操作不破坏权重精度
std::atomic<bool> failed（错误聚合） | 多线程测试的「安全网」：不立即 abort，让所有线程跑完收集完整错误信息
进度回调取消（progress_callback） | 优雅的加载中断机制：返回 true 即触发取消，验证取消后无内存泄漏且返回 nullptr
KV 碎片状态恢复 | llama_state_seq_set_data 支持非连续槽位写入，解决长期运行导致的缓存碎片化问题
多模型×多上下文并发 | 笛卡尔积压测：每个 GPU/CPU 设备各加载模型副本，每副本再派生多上下文并行推理，验证全矩阵稳定性
自动释放（autorelease） | 在独立线程中执行完整 init/free 周期，验证全局状态在任意线程入口/出口处的清理正确性
日志并发压力（test-log） | 8线程并发写入与配置热切换，验证日志库在高压环境下的线程安全性

