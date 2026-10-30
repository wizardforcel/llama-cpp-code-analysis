
# 第17章 上下文与运行时 —— 推理引擎的"驾驶舱"

## 学习目标

- 理解llama_context作为推理引擎入口的整体职责和架构
- 掌握llama_cparams结构体如何统一管理推理上下文配置
- 深入了解sched_reserve机制如何预热图与检测Flash Attention/Gated Delta Net支持
- 熟悉encode/decode主循环的内部流程：批分配、图构建、结果提取
- 理解状态序列化/反序列化的实现原理与文件格式
- 了解适配器（LoRA/cvec）的运行时应用与调度器重预留机制
- 掌握线程池、中止回调、内存分布统计等运行时控制接口的实现
- 理解采样器链与graph_params/can_reuse机制的协同工作
- 能阅读并理解轻量级训练相关的opt_init/opt_epoch实现

## 17.1 生活类比

想象 `llama_context` 是一家餐厅的「中央厨房总监」：**llama_cparams** = 餐厅的运营手册（座位数、菜量、营业时段、是否开启外卖窗口）；**后端初始化** = 招募各路厨师（GPU主厨、CPU副厨、加速器小工），并登记他们的特长；**sched_reserve** = 餐厅开业前的「试菜大会」——预演最繁忙时段的菜谱，确定设备容量；**encode/decode** = 日常营业：接到订单（batch）→ 拆分给厨师（ubatch）→ 上菜（logits/embeddings）；**graph_reserve** = 提前准备好的「万能菜谱模板」，相同菜式直接复用模板；**process_ubatch** = 后厨工位：检查是否可复用旧菜谱（can_reuse），否则现做，并最终摆盘；**can_reuse / graph_reuse** = 经验丰富的厨师对相同菜式直接沿用上次做法（省时）；**attach_threadpool** = 给厨房分配帮工数量（提示处理 vs 单token生成帮工数不同）；**set_abort_callback** = 安装火警按钮，紧急时所有厨师立刻停手；**set_adapters_lora** = 餐厅推出新口味（LoRA），需要重新培训厨师（触发reserve）；**state_serialize** = 把当前所有菜品半成品（KV缓存）打包冷冻，恢复时解冻继续烹饪；**state_seq_save_file** = 把某一桌的半成品单独打包外卖，下次回来继续吃；**opt_init/opt_epoch** = 后厨研发新菜品（小规模训练/微调）；**memory_breakdown** = 月底盘点：模型原料、餐桌占用、临时工位各占多少储物空间；**output_reorder** = 出餐顺序按顾客订单调整（循环模型的特殊排序）；**set_sampler** = 给餐厅配备不同口味品鉴师（采样器），不同桌配不同口味。就像总监需要协调厨师、菜谱、储物与上菜节奏，`llama_context` 也在后端、调度器、内存与图执行之间精细调度，确保推理又快又稳。

## 17.2 源码地图

src/llama-cparams.h
├── struct llama_cparams        # 推理上下文参数（n_ctx/n_batch/flash_attn等）
├── LLAMA_MAX_SEQ               # 最大序列数限制
src/llama-cparams.cpp
└── llama_max_parallel_sequences() # 返回LLAMA_MAX_SEQ
src/llama-context.h
├── struct llama_context        # 上下文核心类（Pimpl模式）
│   ├── llama_context()         # 构造
│   ├── ~llama_context()        # 析构
│   ├── sched_reserve()         # 调度器与图预留
│   ├── synchronize()           # 同步后端与统计
│   ├── encode()/decode()       # 编码/解码入口
│   ├── process_ubatch()        # 单微批次图处理
│   ├── set_adapters_lora()     # 应用LoRA适配器
│   ├── adapters_lora_are_same() # LoRA集合差异比较
│   ├── set_adapter_cvec()      # 应用控制向量
│   ├── attach/detach_threadpool() # 线程池管理
│   ├── set_n_threads()         # 动态线程数
│   ├── set_abort_callback()    # 中止回调
│   ├── set_embeddings/causal_attn/warmup() # 运行时配置
│   ├── set_sampler()           # 后端采样器管理
│   ├── state_get_size/data()/set_data() # 状态序列化
│   ├── state_load_file()/state_save_file() # 文件IO
│   ├── state_seq_get/save/load_file()   # 单序列持久化
│   ├── state_write_data()/state_read_data() # 序列化抽象IO
│   ├── state_seq_write_data()/state_seq_read_data() # 单序列序列化
│   ├── opt_init()/opt_epoch()/opt_epoch_iter() # 训练相关API
│   ├── perf_get_data()/perf_reset() # 性能统计
│   ├── memory_breakdown()      # 内存分布统计
│   ├── output_reserve()/output_reorder() # 输出缓冲区
│   ├── output_resolve_row()    # 批次行到输出行的映射
│   ├── graph_max_nodes()       # 最大图节点数估算
│   ├── graph_reserve()         # 手动图预留
│   ├── graph_compute()         # 图执行入口
│   ├── graph_params()          # 打包llm_graph_params
│   ├── graph_get_cb()          # 图节点命名回调
│   ├── get_gf_res_reserve()    # 预留图结果获取
│   ├── get_logits/embeddings() # 输出数据访问
│   ├── get_sampled_tokens/probs/logits() # 后端采样结果
│   ├── get_cparams()/get_model()/get_sched()/get_memory() # 成员访问
│   ├── n_ctx()/n_ctx_seq()/n_batch()/n_ubatch()/n_seq_max() # 维度查询
│   ├── n_threads()/n_threads_batch() # 线程数查询
│   └── memory_update()         # 内存模块更新
├── struct llama_memory_breakdown_data # 内存分布数据
│   └── total()                 # 总和计算
├── struct llama_cross          # 跨注意力上下文
├── struct buffer_view<T>       # buffer视图模板
├── struct sampling_info        # 后端采样信息
src/llama-context.cpp
├── llama_context::llama_context() # 构造：参数合并、后端初始化、内存模块创建、采样器映射初始化
├── llama_context::~llama_context() # 析构：日志检查、释放opt_ctx
├── llama_context::sched_reserve()   # 调度器预留、FA/GDN自动检测、最坏情况图分配
├── llama_context::synchronize()     # 同步所有后端、更新性能统计
├── llama_context::memory_update()   # 触发内存模块更新与图重预留
├── llama_context::process_ubatch()  # 核心：图复用/重建、set_inputs、graph_compute
├── llama_context::encode()          # 编码器主循环：非因果注意力、提取logits/embeddings
├── llama_context::decode()          # 解码器主循环：KV缓存管理、backend采样、输出重排
├── llama_context::set_sampler()     # 后端采样器初始化/卸载
├── llama_context::set_embeddings()/set_causal_attn()/set_warmup() # 运行时配置
├── llama_context::output_reserve()  # 分配buf_output及sampling buffers
├── llama_context::output_reorder()  # 循环模型输出重排（swap应用）
├── llama_context::output_resolve_row() # 批量索引到输出行的转换
├── llama_context::state_write_data()/state_read_data() # 序列化抽象IO
├── llama_context::state_load_file()/state_save_file()   # 文件IO
├── llama_context::state_seq_*()     # 单序列状态序列化
├── llama_context::graph_max_nodes() # 根据架构估算最大图节点数
├── llama_context::graph_reserve()   # 构造dummy ubatch并预留图
├── llama_context::graph_params()    # 打包llm_graph_params
├── llama_context::graph_compute()   # 设置线程池/线程数，调用sched
├── llama_context::graph_get_cb()    # 图节点命名/后端选择回调
├── llama_context::set_adapters_lora() / adapters_lora_are_same() # LoRA管理
├── llama_context::set_adapter_cvec() # 控制向量管理
├── llama_context::attach_threadpool/detach_threadpool/set_n_threads() # 线程池管理
├── llama_context::set_abort_callback() # 中止回调注入所有后端
├── llama_context::opt_init()        # 初始化ggml_opt_context，标记可训练参数
├── llama_context::opt_epoch()       # 数据集分训练/评估循环
├── llama_context::opt_epoch_iter()  # 单batch训练迭代：构建opt图、设置labels、调用ggml_opt_eval
├── llama_context::perf_get_data()/perf_reset() # 性能统计接口
├── llama_context::memory_breakdown() # 统计模型/上下文/计算缓冲区内存
├── llama_context::get_*()/n_*() 各类简单访问器
├── static build_seq_to_output_row() # 构建seq_id到output_row映射
├── static copy_tensor_async_ints/floats/candidates() # 异步将后端采样结果拷回host
├── static needs_raw_logits()       # 判断是否需要原始logits
├── static llama_set_param()        # 训练时标记可训练张量
├── class llama_io_write_dummy/buffer/file # 序列化输出抽象实现
├── class llama_io_read_buffer/file        # 序列化输入抽象实现
├── 接口层函数 (llama_attach_threadpool/llama_set_adapters_lora/llama_decode/...) # C API实现
├── llama_context_default_params()  # 默认参数
├── llama_init_from_model()         # 模型到上下文的初始化入口
├── llama_perf_context_print/reset() # 性能日志/重置
├── llama_memory_breakdown_print()  # 内存使用表格化打印
└── llama_opt_param_filter_all() / llama_opt_init() / llama_opt_epoch() # 训练C API
src/llama-adapter.h
├── struct llama_adapter_cvec        # 控制向量：每层一个方向向量
│   ├── tensor_for()                 # 按层获取张量
│   ├── apply_to()                   # 在图节点中加到cur
│   ├── apply()                      # 数据加载与层范围设置
│   └── init()                       # 按buft分配ctx/buffer
├── struct llama_adapter_lora_weight # 单个LoRA权重对(a, b, get_scale)
│   ├── get_scale()                  # 按alpha/rank计算实际缩放
│   └── 构造函数                     # 初始化a/b张量
├── struct llama_adapter_lora        # 完整LoRA：ab_map/alpha/gguf_kv/alora_tokens
│   ├── get_weight()                 # 按基模型张量名查找
│   ├── get_n_nodes()                # 估算图节点开销
│   └── 构造/析构                    # 注册与注销到model.loras
├── llama_adapter_lora_init_impl()   # 加载GGUF格式LoRA文件
│   ├── 元数据校验（type/arch/alpha/alora_tokens）
│   ├── lora_a/b配对
│   ├── 按buft分配ctx/buffer
│   └── 数据回读并复制到目标buffer
└── llama_adapter_lora_init/free/meta_* # C API
src/llama-adapter.cpp
├── llama_adapter_cvec::init/apply/tensor_for/apply_to # 控制向量实现
├── llama_adapter_lora_init_impl    # LoRA加载完整流程
├── llama_adapter_meta_val_str/count/key_by_index/val_str_by_index # 元数据查询
├── llama_adapter_get_alora_n_invocation_tokens/invocation_tokens # alora调用
└── llama_adapter_lora_free         # 释放适配器

## 17.3 上下文参数结构 —— 推理引擎的"驾驶舱仪表盘"

`llama_cparams` 是推理上下文的"设计图纸"，集中管理了所有可配置参数。它不包含任何运行时状态，只是一组纯数据字段，供 `llama_context` 构造时合并模型超参数与用户传入的 `llama_context_params` 后使用。理解这些字段的含义，就是掌握了推理引擎的"驾驶舱仪表盘"。

### 17.3.1 `llama_cparams`：上下文参数完整定义

源码路径：`src/llama-cparams.h` - `llama_cparams`（第1-55行）

```cpp
// 类型即图纸，先理解数据结构
struct llama_cparams {
    // ① 上下文容量
    uint32_t n_ctx;              // 最大上下文长度（所有序列共享）
    uint32_t n_ctx_seq;          // 单序列上下文长度（映射场景下可能不同）
    uint32_t n_batch;            // 逻辑批处理大小
    uint32_t n_ubatch;           // 物理微批次大小
    uint32_t n_seq_max;          // 最大并行序列数（受LLAMA_MAX_SEQ限制）

    // ② 线程配置
    uint32_t n_threads;          // 生成阶段线程数
    uint32_t n_threads_batch;    // 提示处理阶段线程数

    // ③ RoPE 与 YaRN
    float rope_freq_base;        // RoPE 基础频率（0=用模型默认）
    float rope_freq_scale;       // RoPE 频率缩放
    enum llama_rope_scaling_type rope_scaling_type;  // YaRN等缩放类型
    float yarn_ext_factor;       // YaRN 外推因子
    float yarn_attn_factor;      // YaRN 注意力缩放因子
    float yarn_beta_fast;        // YaRN 高频修正
    float yarn_beta_slow;        // YaRN 低频修正
    uint32_t yarn_orig_ctx;      // YaRN 原始上下文长度

    // ④ 注意力与缓存策略
    enum llama_attention_type attention_type;     // 因果/非因果/等
    enum llama_flash_attn_type flash_attn_type;   // Flash Attention 开关
    bool flash_attn;                  // 运行时是否启用 FA（自动检测后）
    bool offload_kqv;                 // K/V 卸载到 GPU
    bool op_offload;                  // 算子卸载
    bool swa_full;                    // 滑动窗口全量
    bool kv_unified;                  // 统一KV缓存
    bool causal_attn;                 // 因果注意力开关
    float defrag_thold;               // 碎片整理阈值

    // ⑤ 输出模式
    bool embeddings;                  // 是否输出 embeddings
    enum llama_pooling_type pooling_type; // MEAN/CLS/LAST/RANK

    // ⑥ 量化与设备
    enum ggml_type type_k;            // K 缓存量化类型
    enum ggml_type type_v;            // V 缓存量化类型

    // ⑦ 回调与采样
    ggml_log_callback cb_eval;        // 图节点计算回调
    void * cb_eval_user_data;         // 回调用户数据
    llama_sampler * sampler;          // 初始采样器（已弃用，改用set_samper）
    size_t n_sampler;                 // 采样器数量（已弃用）

    // ⑧ 自动检测开关
    bool auto_fa;                     // 自动检测 Flash Attention
    bool auto_fgdn;                   // 自动检测融合 Gated Delta Net

    // ⑨ 中止与预热
    bool warmup;                      // 预热模式
    bool (*abort_callback)(void *);   // 中止回调
    void * abort_callback_data;       // 回调数据

    // ⑩ 内部流水线并行标志（构造时根据条件设置）
    bool pipeline_parallel;
};
```

这段代码定义了推理上下文的所有可配置参数，涵盖了上下文容量、线程配置、RoPE/YaRN、注意力策略、输出模式、量化类型、回调与自动检测等十大类字段，是 `llama_context` 构造时的唯一配置来源。

### 17.3.2 `LLAMA_MAX_SEQ` 与最大并行序列数

源码路径：`src/llama-cparams.h` - `LLAMA_MAX_SEQ`（第1-55行）；`src/llama-cparams.cpp` - `llama_max_parallel_sequences()`（第1-10行）

```cpp
// src/llama-cparams.h
#define LLAMA_MAX_SEQ 256  // 硬编码上限，防止资源耗尽

// src/llama-cparams.cpp
size_t llama_max_parallel_sequences(void) {
    return LLAMA_MAX_SEQ;
}
```

`LLAMA_MAX_SEQ` 是一个硬编码常量（256），限制了单个上下文能同时服务的最大序列数。`llama_max_parallel_sequences()` 仅返回该常量，供外部查询。构造函数会校验 `n_seq_max` 不超过此值。

## 17.4 生命周期管理 —— "驾驶舱"的建造与拆除

`llama_context` 的构造函数完成了从参数合并到后端初始化、内存模块创建、调度器预热的全流程；析构函数则负责清理后端缓冲区与优化器上下文。

### 17.4.1 `llama_context()`：构造函数全流程

源码路径：`src/llama-context.cpp` - `llama_context::llama_context()`（第30-380行）

```cpp
llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {

    // ① 参数合并：用户参数 + 模型超参数 + 默认值
    cparams = params;
    // 处理 n_ctx / n_batch / n_ubatch / n_seq_max 为 0 的回退逻辑
    // 校验 n_seq_max <= LLAMA_MAX_SEQ

    // ② YaRN 缩放参数计算（支持 DeepSeek V2 mscale_all_dims 特殊路径）
    if (cparams.yarn_attn_factor < 0 && cparams.rope_freq_scale > 1.0f) {
        // 计算 yarn_attn_factor = 0.1 * ln(scale) + 1.0 等
    }

    // ③ 后端初始化：按优先级创建 GPU/ACCEL/CPU 后端
    // 首先添加所有非 CPU 后端（GPU、NPU 等）
    for (auto & dev : model.devices) {
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            backends.emplace_back(ggml_backend_dev_init(dev, nullptr));
        }
    }
    // 单独添加 ACCEL 类型后端（Metal/CUDA 等加速器专用路径）
    for (auto & dev : model.devices) {
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            backends.emplace_back(ggml_backend_dev_init(dev, nullptr));
        }
    }
    // 最后确保有 CPU 后端兜底
    // 收集所有后端的 set_n_threads 函数指针到 set_n_threads_fns

    // ④ 初始化输出缓冲区 buf_output
    // ⑤ 创建内存模块 memory（传入 KV 类型、SWA 配置等）
    // ⑥ 初始化采样器映射：注入用户提供的后端 sampler chain
    // ⑦ 设置 pipeline_parallel 标志（多GPU+异步事件支持时启用）
    // ⑧ 初始化全词表 token ID 列表供后端采样使用

    // ⑨ 最后调用 sched_reserve() 预热调度器与图
    sched_reserve();
}
```

这段代码实现了上下文构造的完整流程：参数合并与校验 → YaRN 参数计算 → 后端初始化（GPU优先、ACCEL单独添加、CPU兜底） → 输出缓冲区预分配 → 内存模块创建 → 采样器映射初始化 → pipeline_parallel 判定 → 词表初始化 → 调度器预热。

### 17.4.2 `~llama_context()`：析构函数

源码路径：`src/llama-context.cpp` - `llama_context::~llama_context()`（第380-410行）

```cpp
llama_context::~llama_context() {
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            // 打印各后端计算缓冲区大小与预期是否匹配的日志
            LLAMA_LOG_DEBUG("%s: backend %zu compute buffer size: expected %zu, actual %zu\n",
                __func__, i, expected_sizes[i], actual_sizes[i]);
        }
    }
    ggml_opt_free(opt_ctx);  // 释放训练优化器上下文（如果存在）
    // 智能指针成员（cvec、loras、balloc、memory、sched等）随对象销毁自动释放
}
```

析构函数首先检查并记录各后端计算缓冲区的实际大小与预期是否一致（用于诊断内存泄漏或分配异常），然后释放 `ggml_opt_context`，其余资源由智能指针自动管理。

## 17.5 调度器与图预留 —— "发动机"的启动与预热

`sched_reserve()` 是推理引擎的"试菜大会"：它负责创建后端调度器、自动检测 Flash Attention 与 Gated Delta Net 支持、预留最坏情况下的计算图，确保后续推理无需重新分配内存。

### 17.5.1 `sched_reserve()`：调度器预留与自动检测

源码路径：`src/llama-context.cpp` - `llama_context::sched_reserve()`（第412-660行）

```cpp
void llama_context::sched_reserve() {
    if (!sched_need_reserve) return;  // ① 快速路径：无需重新预留

    // ② 同步调度器，创建两个图结果实例
    if (sched) ggml_backend_sched_synchronize(sched.get());
    gf_res_prev   = std::make_unique<llm_graph_result>();
    gf_res_reserve = std::make_unique<llm_graph_result>();

    // ③ 创建 ggml_backend_sched 调度器
    // pipeline_parallel=true 时启用多 GPU 流水线并行
    sched = std::make_unique<ggml_backend_sched>(
        backend_ptrs,                    // 后端设备数组
        nullptr,                         // 无图分割回调
        cparams.pipeline_parallel,       // 是否启用流水线并行
        cparams.op_offload,              // 是否启用算子卸载
        graph_max_nodes(cparams.n_ubatch) // 最大节点数估算
    );

    // ④ 初始化内存模块完整上下文，为图预留提供初始状态
    auto mctx_full = memory->init_full();
    if (!mctx_full) return;

    // ⑤ 自动检测 Flash Attention 支持（按层验证 FA 节点与 KV 设备一致性）
    if (cparams.auto_fa) {
        // 预留一个单 token 图，检查是否有 FLASH_ATTN_EXT 节点
        // 比较 FA 设备与 KV 设备是否一致，不一致则禁用 FA 并记录日志
        cparams.flash_attn = detect_flash_attn_support(mctx_full.get());
    }

    // ⑥ 自动检测融合 Gated Delta Net 支持（分别验证 AR 与 CH 模式）
    if (cparams.auto_fgdn) {
        // 分别预留自回归（1 token）和分块（16*n_seqs tokens）路径的图
        // 根据设备分配情况设置 fused_gdn_ar 和 fused_gdn_ch 开关
        detect_fused_gdn_support(mctx_full.get());
    }

    // ⑦ 预留最坏情况 pp 图（prompt processing，batch 模式）
    // ⑧ 预留最坏情况 tg 图（token generation，单 token 模式）
    // 第二次 pp 预留避免 ggml-alloc 在推理过程中触发 reallocation
    // 记录分片数和节点数日志，便于性能诊断

    // ⑨ pipeline_parallel 失败时自动回退到串行调度
    sched_need_reserve = false;
}
```

这段代码实现了调度器的完整预留流程：快速返回检查 → 同步旧调度器 → 创建新调度器（含 pipeline_parallel 配置） → 初始化内存上下文 → Flash Attention 自动检测 → Gated Delta Net 自动检测 → 最坏情况 pp/tg 图预留 → 失败回退 → 标记预留完成。

### 17.5.2 Flash Attention 自动检测细节

Flash Attention 检测的核心逻辑是：构造一个单 token 的解码器图，检查图中是否包含 `FLASH_ATTN_EXT` 节点，并验证该节点所在的设备是否与 K/V 缓存设备一致。若不一致，则禁用 FA 并记录日志，避免跨设备数据传输开销。

### 17.5.3 Gated Delta Net 自动检测细节

Gated Delta Net 检测分两条路径：
- **AR 模式**（自回归）：预留 1 token 图，验证是否有融合 GDN 节点
- **CH 模式**（分块）：预留 `16 * n_seqs` tokens 图，验证分块路径融合支持
根据检测结果分别设置 `fused_gdn_ar` 和 `fused_gdn_ch` 标志。

### 17.5.4 `synchronize()`：后端同步与统计更新

源码路径：`src/llama-context.cpp` - `llama_context::synchronize()`（第660-700行）

```cpp
void llama_context::synchronize() {
    if (!sched) return;
    ggml_backend_sched_synchronize(sched.get());  // 等待所有异步计算完成
    // 更新性能统计：t_eval_us、n_eval 等
}
```

同步所有后端的异步计算，并更新内部性能计数器。

### 17.5.5 `memory_update()`：内存模块更新与图重预留

源码路径：`src/llama-context.cpp` - `llama_context::memory_update()`（第820-870行）

```cpp
bool llama_context::memory_update(bool optimize) {
    if (!memory) return false;
    bool updated = memory->update(optimize);  // 应用待处理的移位/复制/碎片整理
    if (updated) {
        gf_res_prev->reset();  // 内存布局变了，之前的图不能复用
        sched_need_reserve = true;  // 触发下次 sched_reserve 重新预留
    }
    return updated;
}
```

当内存模块发生更新（如 KV 缓存移位、碎片整理）时，重置 `gf_res_prev` 并标记需要重新预留图。

### 17.5.6 `graph_max_nodes()`：最大图节点数估算

源码路径：`src/llama-context.cpp` - `llama_context::graph_max_nodes()`（第2780-2800行）

```cpp
uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    // Qwen3Next/Kimi Linear/Qwen35/Qwen35MoE 等模型有特殊估算
    if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    // 通用估算：max(1024, 8 * n_tensors) + sum(lora_n_nodes)
    uint32_t res = std::max<uint32_t>(1024u, 8u * model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}
```

针对特定架构（Qwen3Next、Kimi Linear、Qwen35 等）返回 `n_tokens * 40` 的较大估值，其他模型按 `max(1024, 8 * n_tensors)` 加上 LoRA 节点开销估算，避免调度器预留节点不足导致构建失败。

## 17.6 编码与解码 —— "交通指挥中心"的日常调度

`encode()` 用于编码器模型（BERT、T5 等），`decode()` 用于自回归解码器（LLaMA、Qwen 等）。两者都通过 `process_ubatch` 执行计算图，但 KV 缓存管理、注意力类型、输出提取方式截然不同。

### 17.6.1 `process_ubatch()`：微批次处理核心

源码路径：`src/llama-context.cpp` - `llama_context::process_ubatch()`（第940-1040行）

```cpp
llm_graph_result * llama_context::process_ubatch(
        const llama_ubatch & ubatch,
        llm_graph_type gtype,
        llama_memory_context_i * mctx,
        ggml_status & ret) {

    // ① 应用内存上下文（KV 缓存移位/复制等）
    if (mctx && !mctx->apply()) {
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // ② 获取上一次图结果与新图参数
    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // ③ 图复用判断：can_reuse 基于 gparams 完整判等
    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        // 复用图：pipeline_parallel 需先同步，避免覆盖输入张量
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }
        n_reused++;
    } else {
        // 重建图：reset → build_graph → alloc_graph
        res->reset();
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        gf = model.build_graph(gparams);
        if (!gf || !ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    // ④ 设置输入张量数据
    res->set_inputs(&ubatch);

    // ⑤ 执行图计算
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;
    return res;
}
```

这段代码是微批次处理的核心：**应用内存上下文 → 判断图复用（can_reuse） → 复用则同步并 set_inputs，否则重建图并分配 → 执行图计算**。`graph_reuse_disable` 环境变量可强制关闭复用。

### 17.6.2 `encode()`：编码器主循环

源码路径：`src/llama-context.cpp` - `llama_context::encode()`（第1050-1300行）

```cpp
int llama_context::encode(const llama_batch & batch_inp) {
    // ① 校验输入：token 与 embd 互斥
    // ② 初始化批次分配器，强制一次性处理整个 batch（split_simple）
    // ③ 临时关闭因果注意力：cparams.causal_attn = false
    // ④ 调用 process_ubatch 执行 ENCODER 图
    // ⑤ 恢复 causal_attn
    // ⑥ 提取 logits（async copy 到 host buffer）
    // ⑦ 提取 embeddings：按 pooling_type 处理 NONE/MEAN/CLS/LAST/RANK
    // ⑧ T5 特殊处理：将 encoder 输出暂存到 cross 结构供后续 decoder 交叉注意力
    return 0;
}
```

编码器模式的关键点：**非因果注意力**、**无 KV 缓存**、**一次性处理全 batch**、**支持多种 pooling**、**T5 交叉注意力暂存**。

### 17.6.3 `decode()`：解码器主循环

源码路径：`src/llama-context.cpp` - `llama_context::decode()`（第1300-1750行）

```cpp
int llama_context::decode(const llama_batch & batch_inp) {
    // ① 校验后端采样器约束：每序列最多 1 个输出 token
    // ② memory_update(false) 应用待处理移位/复制
    // ③ 循环 init_batch → 获取 mctx；失败时回退 memory_update(true) 缓存优化
    // ④ reserve output buffer
    // ⑤ do-while 循环处理每个 ubatch：
    //     - 计算 n_outputs
    //     - process_ubatch(DECODER, mctx)
    //     - 失败时回滚该 ubatch 的 KV 缓存条目
    //     - 提取 logits/embeddings（async copy）
    //     - 后端采样输出异步拷回（copy_tensor_async_*）
    //     - 累计 n_outputs_prev
    // ⑥ output_reorder 懒重排（循环模型）
    // ⑦ 设置 output_ids 映射
    return 0;
}
```

解码器主循环的关键点：**KV 缓存管理（init_batch/mctx）** → **分 ubatch 循环** → **后端采样支持** → **输出重排（output_swaps）** → **失败回滚 KV**。

### 17.6.4 `build_seq_to_output_row()`：序列到输出行映射

源码路径：`src/llama-context.cpp` - `build_seq_to_output_row()`（第1305-1335行）

```cpp
static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(
        const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (!ubatch.output[i]) continue;  // 只统计输出 token
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}
```

构建 `seq_id` 到 `output_row` 的映射，配合 `copy_tensor_async_*` 实现后端采样结果的正确归档。

### 17.6.5 `needs_raw_logits()` 与 `copy_tensor_async_*`

源码路径：`src/llama-context.cpp` - `needs_raw_logits()`（第1440-1460行）、`copy_tensor_async_ints/floats/candidates()`（第1335-1440行）

```cpp
static bool needs_raw_logits(const llama_ubatch & ubatch,
                             const std::map<llama_seq_id, llama_sampler *> & samplers) {
    // 只要有输出 token 的序列没有配置后端采样器，就需要原始 logits
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) continue;
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) return true;
        }
    }
    return false;  // 所有序列都用后端采样
}
```

```cpp
// 异步将后端采样结果拷回 host buffer
static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) { ... }

static void copy_tensor_async_floats(...) { ... }  // logits/probs
static void copy_tensor_async_candidates(...) { ... }  // candidates
```

`needs_raw_logits` 判断是否有序列未配置后端采样器，需拷回原始 logits；`copy_tensor_async_*` 系列函数异步将后端采样结果（token/logits/probs/candidates）拷回 host buffer。

## 17.7 线程池与中止回调 —— "方向盘"与"刹车系统"

### 17.7.1 `attach_threadpool()` / `detach_threadpool()` / `set_n_threads()`

源码路径：`src/llama-context.cpp` - 对应方法（第2200-2240行）

```cpp
void llama_context::attach_threadpool(
        ggml_threadpool_t threadpool,
        ggml_threadpool_t threadpool_batch) {
    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;  // batch 缺省退化为前者
}

void llama_context::detach_threadpool() {
    this->threadpool = this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads_batch;
    // 立即下发到所有后端
    for (const auto & [backend, fn] : set_n_threads_fns) {
        fn(backend, n_threads);
    }
}
```

线程池区分提示处理（`threadpool_batch`）与生成阶段（`threadpool`），缺省时共用。`set_n_threads` 修改配置后立即通过函数指针数组下发到所有已注册后端。

### 17.7.2 `set_abort_callback()`：中止回调注入

源码路径：`src/llama-context.cpp` - `llama_context::set_abort_callback()`（第2240-2270行）

```cpp
void llama_context::set_abort_callback(
        bool (*abort_callback)(void * data), void * abort_callback_data) {
    this->abort_callback = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), abort_callback, abort_callback_data);
        }
    }
}
```

通过动态库符号查找 `ggml_backend_set_abort_callback` 并注入每个后端，后端在计算过程中可检查中止信号并提前退出。

### 17.7.3 `set_warmup()`：预热模式

源码路径：`src/llama-context.cpp` - `llama_context::set_warmup()`（第2270-2280行）

```cpp
void llama_context::set_warmup(bool value) {
    if (cparams.warmup == value) return;
    cparams.warmup = value;
    // warmups 通常使用小 batch，无需 reserve
    // sched_need_reserve = true;  // 故意不触发
}
```

预热模式通常用小 batch 运行，不触发图重预留。

## 17.8 适配器配置 —— "个性化座椅"的安装与调节

### 17.8.1 `set_adapters_lora()`：LoRA 运行时管理

源码路径：`src/llama-context.cpp` - `llama_context::set_adapters_lora()` / `adapters_lora_are_same()`（第2300-2380行）

```cpp
void llama_context::set_adapters_lora(
        llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {

    if (adapters_lora_are_same(adapters, n_adapters, scales)) return;  // 无变化直接返回

    loras.reset(new llama_adapter_loras());  // 重建映射
    for (size_t i = 0; i < n_adapters; i++) {
        if (scales[i] != 0.0f) {  // 忽略零缩放
            loras->insert({adapters[i], scales[i]});
        }
    }
    sched_need_reserve = true;  // 触发调度器重预留
}
```

```cpp
bool llama_context::adapters_lora_are_same(
        llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    size_t n_non_zero = 0;
    for (size_t i = 0; i < n_adapters; i++) {
        if (scales[i] == 0.0f) continue;  // 零缩放忽略
        n_non_zero++;
        auto it = loras->find(adapters[i]);
        if (it == loras->end() || it->second != scales[i]) return false;
    }
    return n_non_zero == loras->size();
}
```

通过 `adapters_lora_are_same` 快速比对新旧适配器集合（忽略 scale=0 的项），相同则直接返回；否则重建映射并触发 `sched_need_reserve`。

### 17.8.2 `set_adapter_cvec()`：控制向量应用

源码路径：`src/llama-context.cpp` - `llama_context::set_adapter_cvec()`（第2380-2440行）

```cpp
bool llama_context::set_adapter_cvec(
        const float * data, size_t len, int32_t n_embd,
        int32_t il_start, int32_t il_end) {
    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);
    sched_need_reserve = true;
    return res;
}
```

委托给 `llama_adapter_cvec::apply` 校验 `n_embd` 并按层写入数据，同样触发调度器重预留。

### 17.8.3 `set_sampler()`：后端采样器挂载

源码路径：`src/llama-context.cpp` - `llama_context::set_sampler()`（第2380-2440行）

```cpp
bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) return true;

    const bool can_offload = sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;  // 仅支持 chain 类型

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());
        sampler->iface->backend_init(sampler, buft);  // 初始化后端采样器
        sampling.samplers[seq_id] = sampler;
        sched_need_reserve = true;
        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("sampler '%s' cannot be offloaded to backend", llama_sampler_name(sampler));
        if (sampling.samplers.count(seq_id) > 0) sched_need_reserve = true;
        sampling.samplers.erase(seq_id);
        return false;
    }

    sampling.samplers.erase(seq_id);
    sched_need_reserve = true;
    return true;
}
```

仅支持 `llama_sampler_chain` 类型且实现了 `backend_init`/`backend_apply` 的采样器才能后端化；否则退化为 CPU 采样并触发重预留。

### 17.8.4 `set_embeddings()` / `set_causal_attn()`

源码路径：`src/llama-context.cpp` - 对应方法（第2280-2300行）

```cpp
void llama_context::set_embeddings(bool value) {
    cparams.embeddings = value;
    // TODO: 暂不触发 reserve
}

void llama_context::set_causal_attn(bool value) {
    if (cparams.causal_attn == value) return;
    cparams.causal_attn = value;
    sched_need_reserve = true;  // 注意力类型切换必须重预留
}
```

切换因果/非因果注意力会触发图重预留，embeddings 模式切换暂不触发。

## 17.9 状态序列化 —— "驾驶记忆"的存档与读档

状态序列化通过 `llama_io_write_i` / `llama_io_read_i` 抽象屏蔽底层 buffer/file 差异，支持全局状态与单序列状态的保存/加载。

### 17.9.1 核心序列化接口

源码路径：`src/llama-context.cpp` - `state_get_size/data/set_data`、`state_load/save_file`、`state_seq_*`（第1750-2050行）

```cpp
size_t llama_context::state_get_size() {
    llama_io_write_dummy dummy;
    return state_write_data(dummy);  // 只统计字节数
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer writer(dst, size);
    return state_write_data(writer);
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer reader(src, size);
    return state_read_data(reader);
}
```

`state_get_size` 使用 `llama_io_write_dummy` 只计算所需字节数；`state_get_data`/`state_set_data` 分别委托给 buffer 实现。

### 17.9.2 文件格式：`state_save_file` / `state_load_file`

源码路径：`src/llama-context.cpp` - `state_load_file` / `state_save_file`（第1750-1950行）

```cpp
bool llama_context::state_save_file(
        const char * filepath,
        const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");
    llama_io_write_file writer(&file);

    // 文件格式：魔数 → 版本 → token数量 → token数据 → state数据
    writer.write(LLAMA_SESSION_MAGIC);  // 魔数
    writer.write(LLAMA_SESSION_VERSION); // 版本
    writer.write((uint32_t)n_token_count);
    for (size_t i = 0; i < n_token_count; i++) writer.write(tokens[i]);
    state_write_data(writer);  // 委托给 memory 模块序列化 KV 数据
    return true;
}

bool llama_context::state_load_file(
        const char * filepath,
        llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");
    llama_io_read_file reader(&file);

    // 校验魔数和版本
    uint32_t magic, version;
    reader.read_to(&magic, sizeof(magic));
    reader.read_to(&version, sizeof(version));
    if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) return false;

    // 读取 token 数量与数据
    uint32_t n_token_count;
    reader.read_to(&n_token_count, sizeof(n_token_count));
    // ... 读取 tokens ...
    state_read_data(reader);  // 恢复 KV 缓存
    return true;
}
```

文件格式：**魔数 `LLAMA_SESSION_MAGIC` → 版本 → token 数量 → token 数组 → KV 缓存数据**。加载时先校验魔数和版本，再恢复上下文状态。

### 17.9.3 单序列状态：`state_seq_save_file` / `state_seq_load_file`

单序列状态使用 `LLAMA_STATE_SEQ_MAGIC` 区分，`flags` 参数控制包含的字段，适用于单独保存/恢复某一对话轮次。

### 17.9.4 抽象 IO 类层次

| 类名 | 用途 | 关键实现 |
|------|------|----------|
| `llama_io_write_dummy` | 只统计字节数，用于 `state_get_size` | `write`/`write_tensor` 仅累加 `size_written` |
| `llama_io_write_buffer` | 写入用户提供的内存 buffer | `memcpy` 到指针，校验缓冲区溢出 |
| `llama_io_write_file` | 写入文件 | `write_tensor` 用临时缓冲避免边读边写冲突 |
| `llama_io_read_buffer` | 从内存 buffer 读取 | 指针移动读取，校验溢出 |
| `llama_io_read_file` | 从文件读取 | `read_to` 直接读入目标，`read` 用临时缓冲 |

## 17.10 性能与内存 —— "车辆仪表"的监控与诊断

### 17.10.1 `perf_get_data()` / `perf_reset()`

源码路径：`src/llama-context.cpp` - 对应方法（第2440-2530行）

```cpp
llama_perf_context_data llama_context::perf_get_data() const {
    // 返回加载时间、prompt eval 时间、单 token eval 时间、token数量、评估次数、图复用次数
}

void llama_context::perf_reset() {
    t_eval_us = t_p_eval_us = 0;
    n_eval = n_p_eval = n_reused = 0;
    t_start_us = ggml_time_us();
}
```

收集并重置性能计数器。

### 17.10.2 `memory_breakdown()` 与 `llama_memory_breakdown_print()`

源码路径：`src/llama-context.cpp` - `memory_breakdown()`（第2440-2530行）、`llama_memory_breakdown_print()`（第2750-2870行）

```cpp
std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>
llama_context::memory_breakdown() const {
    // 聚合 model.memory_breakdown() + memory->memory_breakdown() + sched compute buffers
    // 按 buft 分类统计 model/context/compute 三类内存
}
```

```cpp
void llama_memory_breakdown_print(const struct llama_context * ctx) {
    // 表格化输出每个设备的 total/free/self/unaccounted
    // Host 内存单独汇总，GPU 设备逐个打印
}
```

按后端缓冲区类型统计内存使用，表格化输出每个设备的 total/free/self/unaccounted，便于诊断内存占用和优化硬件卸载。

## 17.11 训练优化 —— "驾驶技术"的训练模式

### 17.11.1 `opt_init()`：优化器初始化

源码路径：`src/llama-context.cpp` - `llama_context::opt_init()` / `llama_set_param()`（第2530-2750行）

```cpp
void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    // 校验 n_ctx_train 能被 n_batch 整除、n_batch 能被 n_ubatch 整除
    // 创建 ggml_opt_context，配置优化器参数（如交叉熵损失）
    // 设置参数过滤器，通过 llama_set_param 标记需要优化的张量
}
```

```cpp
static void llama_set_param(ggml_tensor * cur, const char * name, void * user_data) {
    auto * ctx = (llama_context *)user_data;
    // 过滤 F32 张量，跳过 token_embd 与 rope_freqs（FIXME 已知问题）
    if (cur->type != GGML_TYPE_F32) return;
    if (strcmp(name, "token_embd.weight") == 0) return;
    if (strcmp(name, "rope_freqs.weight") == 0) return;
    ggml_set_param(ctx->opt_ctx, cur);
}
```

`opt_init` 创建 `ggml_opt_context`，`llama_set_param` 过滤 F32 张量并跳过 `token_embd` 与 `rope_freqs`，其余标记为可训练。

### 17.11.2 `opt_epoch()` / `opt_epoch_iter()`：训练循环

```cpp
void llama_context::opt_epoch(...) {
    // 数据集按 idata_split 拆分为训练集（前段）与评估集（后段）
    // 分别设置 idata_in_loop 与回调
    // 调用 opt_epoch_iter 处理每个 batch
}

void llama_context::opt_epoch_iter(...) {
    memory->clear(true);  // 清空 KV 缓存
    // 按 n_batch/n_ubatch 切片构建 batch
    // 遍历 n_ctx/n_batch 个窗口，每窗口内按 n_ubatch 微批处理
    // 为每个微批次构建 opt 图，准备 labels，调用 ggml_opt_eval
}
```

`opt_epoch` 将数据集划分训练/评估，`opt_epoch_iter` 在 `n_ctx` 窗口内按 `n_batch`/`n_ubatch` 构建 opt 图并执行训练/评估迭代。

## 17.12 图执行与控制向量 —— "引擎室"的精密齿轮

### 17.12.1 `graph_reserve()`：手动图预留

源码路径：`src/llama-context.cpp` - `llama_context::graph_reserve()`（第2900-3100行）

```cpp
ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs,
        const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    // 校验 n_tokens 为 n_seqs 倍数，自动 round up
    // 构造 dummy ubatch：每序列设置一个 output token 激活所有后端采样器
    // 调用 model.build_graph 构建图并通过 ggml_backend_sched_reserve 分配缓冲区
    // 支持 split_only 模式（仅切分图）与 no_alloc 模式（dry-run 估算大小）
}
```

手动预留图用于基准测试或预热，构造 dummy ubatch 激活所有后端采样器，支持 `split_only` 与 `no_alloc` 模式。

### 17.12.2 `graph_compute()`：图执行入口

源码路径：`src/llama-context.cpp` - `llama_context::graph_compute()`（第2900-3100行）

```cpp
ggml_status llama_context::graph_compute(ggml_cgraph * gf, bool batched) {
    int n_threads = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch : threadpool;

    // 设置 CPU 后端线程池
    if (backend_cpu) { /* set_threadpool */ }
    // 下发线程数到所有后端
    for (const auto & [backend, fn] : set_n_threads_fns) fn(backend, n_threads);

    return ggml_backend_sched_graph_compute_async(sched.get(), gf);
}
```

根据 `batched` 参数选择线程池与线程数，下发到所有后端，调用调度器异步执行图。

### 17.12.3 `graph_get_cb()`：图节点命名与后端亲和性

源码路径：`src/llama-context.cpp` - `llama_context::graph_get_cb()`（第2900-3100行）

```cpp
llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        // 格式化节点名称：带 il 后缀
        if (il >= 0) ggml_format_name(cur, "%s-%d", name, il);
        else ggml_set_name(cur, name);

        // norm 层后端亲和性调度：仅小 batch 或全 offload 时
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                // 尝试将 norm 分配到该层所在设备，减少跨后端传输
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == model.dev_layer(il)) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}
```

对 norm 层做后端亲和性调度（仅小 batch 或全 offload 时），将 norm 分配到该层所在设备，减少跨后端数据传输开销。

### 17.12.4 `graph_params()`：打包图构建参数

源码路径：`src/llama-context.cpp` - `llama_context::graph_params()`（第2900-3100行）

```cpp
llm_graph_params llama_context::graph_params(
        llm_graph_result * res,
        const llama_ubatch & ubatch,
        const llama_memory_context_i * mctx,
        llm_graph_type gtype) const {
    return {
        .arch        = model.arch,
        .hparams     = model.hparams,
        .cparams     = cparams,
        .ubatch      = ubatch,
        .gtype       = gtype,
        .sched       = sched.get(),
        .backend_cpu = backend_cpu,
        .cvec        = cvec.get(),
        .loras       = loras.get(),
        .mctx        = mctx,
        .cross       = &cross,
        .samplers    = sampling.samplers,
        .n_outputs   = n_outputs,
        .cb          = graph_get_cb(),
        .res         = res,
    };
}
```

将所有图构建所需的上下文打包为 `llm_graph_params`，作为 `model.build_graph` 的唯一输入，确保图结构由该参数集唯一确定。

### 17.12.5 `output_reserve()` 与 `output_reorder()`

源码路径：`src/llama-context.cpp` - `output_reserve()` / `output_reorder()` / `output_resolve_row()`（第1820-2050行）

```cpp
uint32_t llama_context::output_reserve(int32_t n_outputs) {
    // 分配 buf_output 并布局 logits/embd/sampling 子 buffer
    // 计算所需大小：logits + embd + backend_float + backend_token
    // 仅在需要更大容量时重新分配，复用现有 buffer
    // 返回 n_outputs_max
}

void llama_context::output_reorder() {
    // 按 output_swaps 懒重排 logits/embd/sampling 数据
    // selection sort 最小化交换次数
    // 应用交换后清空 output_swaps
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    // 将 batch token 索引通过 output_ids 转换为输出 buffer 行号
    // 支持负数索引（从末尾倒数）
}
```

`output_reserve` 统一分配输出缓冲区；`output_reorder` 在读取输出时按 `output_swaps` 懒重排，避免立即拷贝；`output_resolve_row` 将用户索引映射到物理行号。

### 17.12.6 控制向量适配器 `llama_adapter_cvec`

源码路径：`src/llama-adapter.cpp` - `llama_adapter_cvec::init/apply/apply_to/tensor_for`（第30-130行）

```cpp
bool llama_adapter_cvec::init(const llama_model & model) {
    // 为每个 buft 创建独立 ggml_context
    // 第 0 层无张量（占位 nullptr）
    // 第 1..n_layer-1 层创建 1D F32 张量 (n_embd)
    // 按 buft 分配 buffer 并清零
}

bool llama_adapter_cvec::apply(const llama_model & model,
                               const float * data, size_t len,
                               int32_t n_embd, int32_t il_start, int32_t il_end) {
    // 校验 n_embd，设置 layer_start/end
    // 按层切片写入：offset = n_embd * (il - 1)
}

ggml_tensor * llama_adapter_cvec::apply_to(ggml_context * ctx,
                                           ggml_tensor * cur, int il) const {
    // 图构建时调用：cur = ggml_add(ctx, cur, layer_dir)
    ggml_tensor * layer_dir = tensor_for(il);
    if (layer_dir) cur = ggml_add(ctx, cur, layer_dir);
    return cur;
}
```

控制向量按层存储 1D F32 方向张量，`init` 为每个 buft 分配独立 ctx/buffer，`apply` 校验并写入数据，`apply_to` 在图构建时直接 `ggml_add` 到激活。

## 17.13 设计中的取舍

**为什么用 `llama_io_write_dummy` 计算序列化大小而不是预先计算？**

GGUF 状态包含动态的 KV 缓存数据，其大小取决于当前内存模块的实际占用（序列数、位置、碎片程度）。通过统一的 `llama_io_write_i` 抽象，`dummy` 实现复用了完整的写入逻辑（包括魔数、版本、token数组、memory 序列化），避免了手工计算公式与实际写入逻辑不一致的风险。代价是多一次完整的序列化遍历，但序列化本身是低频操作，开销可接受。

**为什么 `process_ubatch` 中 `pipeline_parallel` 模式复用图前必须 `synchronize`？**

流水线并行下，上一轮 `graph_compute_async` 可能仍在 GPU 上运行。若直接 `set_inputs` 覆盖输入张量，会导致读写冲突。同步确保前一轮计算完全结束，输入张量不再被读取后再写入新数据。这是正确性前置条件，而非性能优化。

**为什么 `output_reorder` 采用懒重排（selection sort + 记录 swaps）而不是立即重排？**

循环模型（RWKV、Mamba 等）的输出顺序与用户提交的 batch 顺序可能不一致。立即重排需要同步后端数据到 host，破坏异步流水线。懒重排将交换操作记录下来，仅在用户真正调用 `get_logits_ith`/`get_embeddings_ith` 时在 host 内存中完成，保持异步计算流水线不阻塞。

**为什么 `set_sampler` 仅支持 `llama_sampler_chain` 且要求 `backend_init`/`backend_apply`？**

后端采样需要采样器实现特定的后端接口（初始化、应用），且链式结构便于统一管理多个采样阶段。单一采样器或不支持后端接口的采样器无法安全卸载，强制退化到 CPU 采样保证正确性。

## 17.14 动手练习

### 17.14.1 练习1：阅读llama_context构造与析构
阅读 `src/llama-context.cpp` 第30-380行的 `llama_context::llama_context()` 实现：
1. 构造函数如何处理 `n_seq_max` 超出 `LLAMA_MAX_SEQ` 的情况？
2. YaRN 缩放参数 (yarn_attn_factor) 是如何根据 `rope_yarn_log_mul` 计算的？
3. `kv_unified` 与 `n_ctx_seq` 的关系是什么？
4. 后端初始化顺序是什么？为什么把 ACCEL 类型后端单独添加？
5. 阅读析构函数：它如何诊断计算缓冲区大小与预期不符的情况？

回答问题：
- 构造失败时哪些异常会被抛出？调用方如何处理？
- 为何 `sched_reserve()` 在构造函数末尾而非更早调用？

### 17.14.2 练习2：解析sched_reserve中的自动检测逻辑
阅读 `src/llama-context.cpp` 中 `llama_context::sched_reserve()` 第412-660行：
1. `cparams.auto_fa` 检测 Flash Attention 支持的完整流程是什么？
2. 如何判断 `fa_device_mismatch`？它对 `cparams.flash_attn` 有什么影响？
3. `cparams.auto_fgdn` 检测分块 GDN 时为何要构造 `n_tokens_ch = 16*n_seqs`？
4. 预留 pp 与 tg 两种最坏情况图的目的是什么？
5. 当 pipeline_parallel 分配失败时的回退策略是什么？

回答问题：
- 为何先做 FA 检测再做 GDN 检测？顺序有依赖吗？
- `gf_res_prev` 与 `gf_res_reserve` 两个图结果实例的职责分别是什么？

### 17.14.3 练习3：理解decode主循环的KV缓存交互
阅读 `src/llama-context.cpp` 中 `llama_context::decode()` 第1100-1500行：
1. `memory_update(false)` 与 `init_batch()` 的关系是什么？
2. `mctx->get_status()` 返回 `FAILED_PREPARE` 时如何尝试缓存优化（`memory_update(true)`）？
3. 何时需要 `output_swaps`？循环模型如何被重排？
4. `copy_tensor_async_*` 系列函数如何把后端采样结果异步拷回host？
5. 当 `process_ubatch` 失败时如何回滚已写入的KV缓存条目？

回答问题：
- 为何 `decode()` 内层循环用 `do { } while (mctx->next())` 而非 `while`？
- 后端采样（`sampling.samplers`）存在时对 `n_outputs` 有什么约束？

### 17.14.4 练习4：状态序列化文件格式解读
阅读 `src/llama-context.cpp` 第1500-1900行的状态序列化代码：
1. 写出 `state_save_file` 写入文件的字节布局（按顺序）。
2. `llama_io_write_dummy` 的作用是什么？它如何计算所需大小？
3. `state_read_data` 中为何要先校验 `arch_str`？
4. 单序列状态文件（`state_seq_save_file`）与全局状态文件（`state_save_file`）在格式上有什么区别？
5. `llama_io_write_file::write_tensor` 为什么用临时缓冲而不是直接 `ggml_backend_tensor_get`？

回答问题：
- 魔数 `LLAMA_SESSION_MAGIC` 与 `LLAMA_STATE_SEQ_MAGIC` 的区分意义？
- 如果KV缓存类型发生变化，旧的session文件能否正确恢复？

### 17.14.5 练习5：LoRA适配器加载与运行时应用
阅读 `src/llama-adapter.cpp` 中 `llama_adapter_lora_init_impl` 全文：
1. 加载GGUF时如何校验 `general.type == adapter` 与 `adapter.type == lora`？
2. 如何将 `*.lora_a` 与 `*.lora_b` 张量配对成 `ab_map`？
3. 为什么对 `token_embd.weight` 的形状校验与其他张量不同？
4. `buft_extra` 列表的作用是什么？什么情况下 LoRA 会被强制回退到 CPU？
5. 在 `llama_context::set_adapters_lora()` 中为何要比较新旧权重并忽略 scale=0 的项？

回答问题：
- LoRA 应用后 `sched_need_reserve = true` 会带来什么开销？
- 如果两个 LoRA 适配同一层，GGML 计算图如何同时叠加它们的缩放？

### 17.14.6 练习6：图复用与can_reuse的边界
阅读 `src/llama-context.cpp` 中 `process_ubatch` 与 `graph_reserve` 实现：
1. `res->can_reuse(gparams)` 的判等条件是什么？哪些参数变化会强制重建图？
2. `graph_reuse_disable` 环境变量在何处被读取？其作用范围是什么？
3. `set_inputs` 调用为何在图复用与图重建后都需要执行？
4. `pipeline_parallel` 模式下为何需要在 set_inputs 前先做 `ggml_backend_sched_synchronize`？
5. `output_resolve_row` 如何与 `output_swaps` 配合保证输出顺序？

回答问题：
- 为何 `gf_res_prev` 在 `memory_update` 中要被 reset？
- `graph_max_nodes` 为 Qwen3Next/Kimi Linear/Qwen35 等模型返回的特殊值是什么含义？

### 17.14.7 练习7：训练接口opt_init与opt_epoch
阅读 `src/llama-context.cpp` 中 `opt_init`、`opt_epoch`、`opt_epoch_iter` 实现：
1. `llama_set_param` 过滤张量的规则是什么？哪些张量会被显式跳过？
2. `opt_params.opt_period = n_batch / n_ubatch` 的意义是什么？
3. `opt_epoch_iter` 中为何要 `memory->clear(true)`？
4. `labels_sparse` 如何被转换为稠密的 `ggml_tensor labels`？
5. 训练和评估回调 `callback_train` 与 `callback_eval` 在 `opt_epoch` 中何时被触发？

回答问题：
- `idata_split` 如何划分训练集与评估集？
- 训练结束后 `opt_ctx` 何时被释放？

### 17.14.8 练习8：运行时控制接口与采样器挂载
阅读 `src/llama-context.cpp` 中 `attach_threadpool`、`detach_threadpool`、`set_n_threads`、`set_abort_callback`、`set_sampler` 实现：
1. 线程池的 `threadpool_batch` 为何在缺省时退化为 `threadpool`？
2. `set_n_threads` 修改 cparams 后如何通知所有后端？
3. `set_abort_callback` 通过动态库符号注入回调的实现机制是什么？
4. `set_sampler` 中 `can_offload` 的三个条件各是什么？为什么缺一不可？

回答问题：
- 当采样器无法后端化时会发生什么？为什么需要 `sched_need_reserve`？

## 17.15 本章小结

这一章中我们学习了 llama.cpp 核心推理上下文 `llama_context` 的完整实现。首先剖析了 `llama_cparams` 这一"驾驶舱仪表盘"结构体，理解其如何统管上下文容量、线程、RoPE/YaRN、注意力策略、输出模式、量化类型、自动检测开关等十大类配置。其次深入构造与析构流程，看参数合并、后端初始化（GPU优先、ACCEL单独添加、CPU兜底）、内存模块创建、采样器映射、调度器预热的全链路。接着详细解读 `sched_reserve` 的"试菜大会"：Flash Attention 与 Gated Delta Net 自动检测、最坏情况 pp/tg 图预留、pipeline_parallel 回退机制。然后对比 `encode` 与 `decode` 两大主循环：编码器走非因果注意力、无KV缓存、一次性处理；解码器管理KV缓存、分ubatch循环、支持后端采样、输出懒重排。随后讲解 `process_ubatch` 的图复用核心（`can_reuse` 判等、`graph_reuse_disable` 强制关闭）、线程池与中止回调的运行时控制、LoRA/cvec 适配器的动态挂载与触发重预留、状态序列化的抽象 IO 体系与文件格式（魔数、版本、token数组、KV数据）、性能统计与内存分布诊断、以及轻量级训练接口 `opt_init/opt_epoch` 的实现细节。最后梳理了图执行入口 `graph_compute`、参数打包 `graph_params`、后端亲和性回调 `graph_get_cb`、输出缓冲区管理与控制向量适配器的实现。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| `llama_cparams` | 推理上下文参数集合，统管ctx/batch/线程/RoPE/Flash Attention等开关 |
| `llama_context` 构造 | 合并参数→初始化后端→创建内存模块→预热调度器→初始化采样器映射 |
| `sched_reserve` | FA/GDN自动检测、预留最坏情况pp/tg图，避免推理时再分配 |
| `graph_reserve` | 用dummy ubatch构造真实图，触发gallocr分配计算缓冲区 |
| `graph_reuse_disable` | 通过环境变量LLAMA_GRAPH_REUSE_DISABLE强制关闭can_reuse |
| `pipeline_parallel` | 多GPU层分割+offload_kqv开启+异步事件支持时启用，否则回退 |
| `process_ubatch` | 图复用核心：can_reuse则同步后直接set_inputs，否则reset并build_graph |
| `can_reuse` | 根据gparams判等决定是否复用上次图，避免每次重建 |
| `graph_get_cb` | 对norm层做后端亲和性调度，减少跨后端数据传输 |
| `encode()` | 编码器模式：强制非因果注意力，无KV缓存，处理整个batch |
| `decode()` | 解码器主循环：KV缓存管理、backend采样、输出重排 |
| `memory_update` | 应用内存模块更新并触发最坏情况图重预留 |
| `output_reserve` | 分配host buffer并布局logits/embd/采样数组 |
| `output_reorder` | 对循环模型按用户原始batch顺序懒重排logits/embeddings |
| `output_resolve_row` | 将batch token索引通过output_ids转换为输出buffer行号 |
| `set_adapters_lora` | 比对已有loras→若无变化直接返回；否则重建映射并触发reserve |
| `set_adapter_cvec` | 按层写入控制向量数据并触发reserve |
| `attach_threadpool` | 绑定外部线程池，prompt处理与token生成可分别指定 |
| `set_n_threads` | 运行时调整n_threads/n_threads_batch并立即下发所有后端 |
| `set_abort_callback` | 通过动态库符号给所有后端安装中止回调 |
| `set_sampler` | 按seq_id注入后端采样器，仅支持chain类型，否则退化为CPU采样 |
| `state_serialize` | 通过llama_io_write_i/read_i抽象序列化KV缓存到buffer或文件 |
| `state_load_file` | 校验魔数LLAMA_SESSION_MAGIC、版本、token数量再恢复KV |
| `state_seq_save_file` | 单序列状态保存，使用LLAMA_STATE_SEQ_MAGIC |
| `llama_io_write_dummy` | 只统计字节数的占位IO，用于state_get_size |
| `llama_io_write_buffer/file` | 具体序列化实现，分别写入内存与文件 |
| `llama_io_read_buffer/file` | 具体反序列化实现，分别读取内存与文件 |
| `needs_raw_logits` | 判断是否有序列未配置后端采样器，需拷回原始logits |
| `copy_tensor_async_*` | 异步将后端采样结果（token/logits/probs/candidates）拷回host buffer |
| `build_seq_to_output_row` | 将每个seq_id映射到其output_row以支持正确采样 |
| `opt_init` | 创建ggml_opt_context，逐层调用llama_set_param标记可训练张量 |
| `llama_set_param` | 过滤F32张量，跳过token_embd与rope_freqs，调ggml_set_param |
| `opt_epoch` | 数据集划分为train/eval循环，每个batch构建opt图并设置labels |
| `opt_epoch_iter` | 在n_ctx窗口内按n_batch/n_ubatch构建opt图，调用ggml_opt_eval |
| `memory_breakdown` | 按buft聚合模型/上下文/计算缓冲区内存用量 |
| `llama_memory_breakdown_print` | 表格化输出每个设备的total/free/self/unaccounted |
| `llama_adapter_cvec` | 控制向量按层存1D张量，apply时直接ggml_add到激活 |
| `llama_adapter_lora` | GGUF格式加载、a/b配对、按buft分配ctx/buffer并回读数据 |
| `llama_adapter_lora_weight::get_scale` | 按alpha/rank计算实际LoRA缩放系数 |
| `llama_adapter_meta_*` | 查询LoRA GGUF元数据KV对 |
| `llama_adapter_get_alora_*` | 获取alora调用token序列 |

下一章中，我们将学习内存与缓存系统 —— 推理状态的"记忆宫殿"，深入理解 llama.cpp 如何管理推理过程中的状态缓存：标准 KV Cache、滑动窗口 KV Cache（ISWA）、循环式记忆（RNN/SSM）以及混合内存架构。
