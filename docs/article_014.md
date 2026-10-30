
# 第14章 模型对象与计算图构造 —— 推理引擎的"心脏"

## 学习目标

- 理解 llama_model 类的整体架构、生命周期（构造、析构、Pimpl 模式）以及与 llama_hparams、llama_vocab、llama_layer 的关系
- 掌握模型元数据的四阶段加载流水线（load_arch → load_hparams → load_vocab → load_tensors）的协作机制
- 理解张量按设备分配策略（make_cpu_buft_list / make_gpu_buft_list / select_buft）的实现原理
- 熟悉 llama_layer 与各子结构体（posnet、convnext、shortconv、nextn）如何承载 80+ 架构的张量差异
- 掌握 build_graph 如何根据 arch 路由到对应 llm_build_* 类的实现，以及 create_memory 的多种缓存分配
- 了解 RoPE 双源频率切换（hparams vs cparams）与 LongRoPe 长短因子自动选择机制
- 理解模型信息查询接口（arch_name/type_name/desc/print_info 等）的输出格式与调试用途
- 理解 llm_graph_input_i 家族如何抽象不同模型架构的计算图输入并支持图复用
- 掌握 llm_graph_result 如何承载图构建产物与 can_reuse 机制复用已有计算图
- 了解 llm_graph_context 提供的 build_norm/build_ffn/build_moe_ffn/build_attn_mha 等公共子图构造函数
- 熟悉 graph 类型枚举（DEFAULT/ENCODER/DECODER）与 T5 等 encoder-decoder 架构的分发逻辑
- 理解图构建中的归一化、注意力、FFN、循环状态、池化、采样等模块化协作方式

## 14.1 生活类比

想象 llama_model 是一座**大型图书馆**的核心管理机构：`arch / type` 是图书馆的牌匾与分类号（LLaMA 类、Gemma 类、MoE 类）；`hparams` 是馆藏目录（层数、卷册数、开本尺寸等基础编目信息）；`vocab` 是索引卡柜（所有 Token 与词典信息）；`layers` 是楼层办公室的**集装箱**（每层存放归一化、注意力、FFN 各种零件）；`pimpl` 是地下仓库（存放 mmap、mlock、buffer、device 等重资源，对外只暴露层）。`load_arch / load_hparams / load_vocab / load_tensors` 就是**四阶段审阅流程**：先验牌匾、再核目录、再查索引、最后清点藏书。`make_cpu_buft_list / make_gpu_buft_list` 是**分配货架**（按 ACCEL→Host→Extra→CPU 优先级摆放）；`select_buft` 是**临时试放货物**，看哪条货架能装下这个大家伙；`build_graph` 是依据图纸书把藏书重新装订成**一本书**（计算图）；`create_memory` 是**阅览室**：标准阅览室、滑动窗口阅览室（ISWA）、循环阅览室（RNN）、混合阅览室；`get_rope_factors` 是**翻书节奏**：短诗用短节奏、长篇用长节奏。想象 llama_graph 系统是一套**印刷厂的装订流水线**：`llm_graph_input_i 家族` 是不同类型的**纸张输入口**（正文纸、插图纸、索引纸），各司其职；`llm_graph_context` 是**装订工坊**，持有所有工具与半成品；`build_norm / build_ffn / build_attn_mha / build_moe_ffn` 是流水线上的**通用机器**（裁切机、装订机、压膜机）；`llm_graph_result` 是最终装订好的**书册**（含 logits、embeddings 等成品章节）；`can_reuse` 是如果新订单内容与旧订单相同，**复用旧书**（跳过装订）；`build_pooling / build_sampling / build_dense_out` 是最后的**封面包装**与**书签插入**工序。就像图书馆馆长需要按楼层、书架、读者类型精心分配每本书的位置，llama_model 也要精确规划每个张量在哪个设备、哪个 buffer 上安家；而装订工坊则需要把不同类型的纸张与零件按图组装成最终可印刷的成品书。

## 14.2 源码地图

src/llama-model.h
├── enum llm_type                            # 模型规模枚举（14M/7B/671B/8x7B 等 100+ 规模标签）
├── std::string llama_rope_scaling_type_name(...) # RoPE 缩放类型名称
├── struct llama_layer_posnet                # WavTokenizer PosNet 子结构
│   ├── norm1 / norm1_b / conv1 / conv1_b    # resnet 块
│   ├── attn_q / attn_k / attn_v / attn_o    # 注意力投影
│   └── norm / norm_b                        # 输出归一化
├── struct llama_layer_convnext              # ConvNeXt 子结构
│   ├── dw / dw_b / norm / norm_b            # 深度卷积 + LN
│   ├── pw1 / pw1_b / pw2 / pw2_b / gamma    # 点卷积 + 缩放
├── struct llama_layer_shortconv             # LFM2 短卷积子结构
│   └── in_proj / conv / out_proj
├── struct llama_layer_nextn                 # NextN/MTP 预测层
│   └── eh_proj / embed_tokens / enorm / en_hnorm / shared_head_*
├── struct llama_layer                       # 单层张量容器（最大结构）
│   ├── attn_norm / attn_q_norm / attn_k_norm # 归一化家族
│   ├── wq / wk / wv / wo / wqkv / wq_a / wq_b # 注意力权重
│   ├── ffn_gate / ffn_up / ffn_down         # 稠密 FFN
│   ├── ffn_gate_exps / ffn_up_exps / ffn_down_exps # MoE 专家
│   ├── ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp # 共享专家
│   ├── ssm_in / ssm_x / ssm_dt / ssm_a / ssm_d # Mamba/SSM 状态
│   ├── time_mix_* (RWKV6/7) / channel_mix_* # RWKV 时间/通道混合
│   ├── rope_long / rope_short / rope_freqs  # RoPE 因子
│   ├── per_layer_inp_gate / per_layer_proj / altup_* # Gemma3n AltUp
│   ├── attn_sinks (OpenAI-MoE) / indexer_* (DSA) # 特殊架构
│   └── posnet / convnext / shortconv / nextn # 内嵌子结构
├── struct llama_model                       # 模型主对象
│   ├── llm_type type / llm_arch arch / std::string name
│   ├── llama_hparams hparams / llama_vocab vocab
│   ├── tok_embd / type_embd / pos_embd / tok_norm # 嵌入家族
│   ├── output_norm / output / output_b      # 输出投影
│   ├── cls / cls_b / cls_out / cls_out_b / cls_norm # 分类头
│   ├── tok_embd_per_layer / altup_proj / per_layer_model_proj # Gemma3n/4
│   ├── std::vector<llama_layer> layers      # 层数组
│   ├── dense_2_out_layers / dense_3_out_layers # SentenceTransformers
│   ├── gguf_kv / devices / tensors_by_name / loras
│   ├── llama_model() / ~llama_model()      # 构造/析构
│   ├── load_stats/load_arch/load_hparams/load_vocab/load_tensors # 加载流水线
│   ├── arch_name / type_name / desc / size / n_tensors / n_devices
│   ├── n_gpu_layers / split_mode / memory_breakdown
│   ├── dev_layer / dev_output / select_buft / get_tensor
│   ├── get_rope_freq_base / get_rope_freq_scale / get_rope_factors
│   ├── create_memory / build_graph          # 内存与图构造
│   └── struct impl / std::unique_ptr<impl> pimpl # Pimpl 隐藏
src/llama-model.cpp
├── const char * llm_type_name(llm_type)    # 100+ 规模字符串映射
├── static llama_expert_gating_func_name(...) # 门控函数名
├── static llama_rope_scaling_type_from_string(...) # RoPE 字符串→枚举
├── static make_cpu_buft_list(...)          # CPU 缓冲类型优先级装配
├── static make_gpu_buft_list(...)          # GPU 缓冲类型装配（含 split）
├── struct llama_model::impl                # 私有实现
│   ├── n_elements / n_bytes / desc_str
│   ├── mappings / mlock_bufs / mlock_mmaps
│   ├── ctxs_bufs / cpu_buft_list / gpu_buft_list
│   └── dev_input / dev_layer / dev_output
├── llama_model::llama_model()              # 构造（pimpl 初始化）
├── llama_model::~llama_model()             # 析构（释放 LoRA）
├── llama_model::load_stats()               # 写入元素统计
├── llama_model::load_arch()                # 读取 arch（未知则抛）
├── llama_model::load_hparams()             # 解析 GGUF KV（100+ 架构 switch）
├── llama_model::load_vocab()               # 委托 llama_vocab.load
├── llama_model::load_tensors()             # 创建设备映射 + 张量创建（80+ 架构 switch）
├── llama_model::arch_name/type_name/desc()
├── llama_model::size/n_tensors/n_devices()
├── llama_model::n_gpu_layers() / split_mode()
├── llama_model::memory_breakdown()         # 按 buft 聚合缓冲
├── llama_model::n_elements() / print_info() # 打印详细模型信息
├── llama_model::dev_layer/dev_output()
├── buft_supported<F>() / select_buft<F>()  # 能力探测选 buft
├── llama_model::select_buft() / has_tensor_overrides() / get_tensor()
├── llama_model::get_rope_freq_base/scale() # SWA 双源切换
├── llama_model::get_rope_factors()         # LongRoPe 长短因子选择
├── llama_model::create_memory()            # 五种缓存架构分支
│   ├── return nullptr（编码器/扩散/无状态）
│   ├── llama_memory_recurrent              # Mamba/RWKV
│   ├── llama_memory_hybrid / _iswa        # Jamba/Falcon-H1/Nemotron-H
│   ├── llama_kv_cache_iswa                # Gemma3/Gemma4（带 SWA）
│   └── llama_kv_cache                      # 标准 Transformer
├── llama_model::build_graph()              # 80+ 架构 → llm_build_* 路由
│   ├── switch (arch) 模板类分发（<false>/<true>）
│   ├── llm->build_pooling()                # 分类头
│   ├── llm->build_sampling()               # 后端采样
│   ├── llm->build_dense_out()              # dense 投影
│   └── llm->res->set_outputs() / get_gf()
├── llama_model_default_params()            # 默认参数（设备、张量分割等）
├── llama_model_get_vocab/free/n_*         # C API 实现
├── llama_model_rope_type()                  # arch → RoPE 类型
├── llama_model_meta_val_str/count/key_*/val_*_by_index() # 元数据查询
├── llama_model_desc/size/chat_template()
├── llama_model_n_params/has_encoder/has_decoder/decoder_start_token()
├── llama_model_is_recurrent/is_hybrid/is_diffusion()
└── llama_internal_get_tensor_map()          # 内部测试用
src/llama-graph.h
├── enum llm_graph_type / llm_ffn_op_type / llm_ffn_gate_type / llm_norm_type
├── struct llama_cross                       # 跨注意力临时容器
├── class llm_graph_input_i                 # 输入抽象基类
│   ├── set_input / can_reuse / LLAMA_GRAPH_INPUT_DEBUG
├── class llm_graph_input_embd / llm_graph_input_pos
├── class llm_graph_input_attn_temp           # 温度缩放
├── class llm_graph_input_pos_bucket / _kv    # T5 相对位置桶
├── class llm_graph_input_out_ids / mean / cls
├── class llm_graph_input_rs                  # 循环状态
├── class llm_graph_input_cross_embd
├── class llm_graph_input_attn_no_cache / _kv / _k / _kv_iswa / _cross
├── class llm_graph_input_mem_hybrid / _k / _iswa / _sampling
├── class llm_graph_result                   # 图构建产物
│   ├── t_inp_tokens / t_inp_embd / t_logits / t_embd / t_embd_pooled
│   ├── t_sampled_logits / t_candidates / t_sampled / t_sampled_probs
│   ├── inputs / ctx_compute / gf / max_nodes
│   ├── reset / set_inputs / set_outputs / can_reuse
│   └── add_input / set_params / get_*
├── struct llm_graph_params                  # 图构建参数 + allow_reuse
│   ├── arch / hparams / cparams / ubatch / gtype
│   ├── sched / backend_cpu / cvec / loras / mctx / cross
│   ├── samplers / n_outputs / cb / res
│   ├── samplers_equal()
│   └── allow_reuse()                        # 图拓扑复用判定
└── struct llm_graph_context                 # 图构建上下文 + 各种 build_*
    ├── build_lora_mm / build_lora_mm_id
    ├── build_norm / build_ffn / build_moe_ffn (双版本)
    ├── build_inp_embd / pos / attn_scale / out_ids / mean / cls
    ├── build_inp_cross_embd / pos_bucket_enc / pos_bucket_dec
    ├── build_pos_bias / build_attn_mha
    ├── build_attn_inp_no_cache / kv / k / kv_iswa / cross
    ├── build_rs / build_rs_inp / build_rwkv_token_shift_load/store
    ├── build_inp_mem_hybrid / _k / _iswa
    ├── build_pooling / build_sampling / build_dense_out
    └── cb / ctx0 / gf / res

## 14.3 llama_model 类骨架与生命周期 —— 模型对象的"主入口"与"幕后管家"

**`llama_model` 是 llama.cpp 中所有模型状态的"中央枢纽"**。它持有 `llm_type type`（如 `LLM_TYPE_7B`）与 `llm_arch arch`（如 `LLM_ARCH_LLAMA`）两大身份标识，聚合超参数 `llama_hparams hparams`、词表 `llama_vocab vocab` 与层数组 `std::vector<llama_layer> layers`。嵌入/输出张量（`tok_embd`、`output_norm`、`output`）、分类头（`cls`、`cls_out`）、Dense 输出层（`dense_2_out_layers` 等）都直接挂在模型对象上。`tensors_by_name` 是张量名 → 张量指针的映射，供量化、调试、量化统计等模块按名查找。`loras` 集合跟踪已关联的 LoRA 适配器，确保析构时统一释放。

### 14.3.1 llama_model：模型主对象与 Pimpl 实现

**源码路径：`src/llama-model.h` - `llama_model`（第 540-589 行）**

```cpp
// 模型主对象：持有所有推理期状态的中央枢纽
struct llama_model {
    // ─── 身份标识 ───
    llm_type type;                    // 模型规模标签，如 LLM_TYPE_7B
    llm_arch arch;                    // 模型架构枚举，如 LLM_ARCH_LLAMA
    std::string name;                 // 从 GGUF 读取的 general.name

    // ─── 核心子系统 ───
    llama_hparams hparams;            // 超参数容器（层数、头数、RoPE 等）
    llama_vocab vocab;                // 词表与分词器状态

    // ─── 嵌入/输出/分类头 ───
    ggml_tensor * tok_embd = nullptr;          // token 词嵌入
    ggml_tensor * type_embd = nullptr;         // token type 嵌入（BERT 等）
    ggml_tensor * pos_embd = nullptr;          // 位置嵌入（绝对位置编码模型）
    ggml_tensor * tok_norm = nullptr;          // 输入归一化（部分架构）

    ggml_tensor * output_norm = nullptr;       // 输出归一化
    ggml_tensor * output = nullptr;            // 输出投影（LM Head）
    ggml_tensor * output_b = nullptr;          // 输出偏置

    ggml_tensor * cls = nullptr;               // 分类头权重
    ggml_tensor * cls_b = nullptr;             // 分类头偏置
    ggml_tensor * cls_out = nullptr;           // 分类输出投影
    ggml_tensor * cls_out_b = nullptr;         // 分类输出偏置
    ggml_tensor * cls_norm = nullptr;          // 分类归一化

    // ─── Gemma3n/4 特有：逐层投影 ───
    std::vector<ggml_tensor *> tok_embd_per_layer;
    std::vector<ggml_tensor *> altup_proj;
    std::vector<ggml_tensor *> per_layer_model_proj;

    // ─── 层数组：每层一个"集装箱" ───
    std::vector<llama_layer> layers;           // 核心层张量容器

    // ─── SentenceTransformers 等密集输出层 ───
    std::vector<ggml_tensor *> dense_2_out_layers;
    std::vector<ggml_tensor *> dense_2_out_layers_b;
    std::vector<ggml_tensor *> dense_3_out_layers;

    // ─── 元数据与资源 ───
    std::map<std::string, std::string> gguf_kv;      // 原始 GGUF KV（字符串化）
    std::vector<ggml_backend_dev_t> devices;         // 可用后端设备列表
    std::vector<std::pair<std::string, ggml_tensor *>> tensors_by_name; // 名称→张量
    std::vector<llama_adapter_lora *> loras;         // 已关联的 LoRA 适配器

    // ─── 构造/析构 ───
    explicit llama_model(const struct llama_model_params & params);
    ~llama_model();

    // ─── 四阶段加载流水线 ───
    void load_stats  (llama_model_loader & ml);
    void load_arch   (llama_model_loader & ml);
    void load_hparams(llama_model_loader & ml);
    void load_vocab  (llama_model_loader & ml);
    bool load_tensors(llama_model_loader & ml); // 返回 false 表示被进度回调取消

    // ─── 模型信息查询 ───
    std::string arch_name() const;
    std::string type_name() const;
    std::string desc() const;
    size_t size() const;              // 文件大小
    size_t n_tensors() const;         // 张量总数
    size_t n_devices() const;         // 设备数
    uint32_t n_gpu_layers() const;    // GPU 卸载层数
    llama_split_mode split_mode() const;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;
    uint64_t n_elements() const;      // 参数总量
    void print_info() const;          // 完整体检报告

    // ─── 设备与缓冲查询 ───
    ggml_backend_dev_t dev_layer(int il) const;
    ggml_backend_dev_t dev_output() const;
    ggml_backend_buffer_type_t select_buft(int il) const;
    bool has_tensor_overrides() const;
    const struct ggml_tensor * get_tensor(const char * name) const;

    // ─── RoPE 参数获取 ───
    float get_rope_freq_base (const llama_cparams & cparams, int il) const;
    float get_rope_freq_scale(const llama_cparams & cparams, int il) const;
    ggml_tensor * get_rope_factors(const llama_cparams & cparams, int il) const;

    // ─── 内存与图构造 ───
    llama_memory_i * create_memory(const llama_memory_params & params, const llama_cparams & cparams) const;
    ggml_cgraph * build_graph(const llm_graph_params & params) const;

private:
    // ─── Pimpl：隐藏重资源实现细节 ───
    struct impl;
    std::unique_ptr<impl> pimpl;

    // ─── 运行时参数（只读） ───
    const llama_model_params params;
};
```

这段代码定义了 `llama_model` 结构体，它是整个推理引擎的核心数据容器。除了上述字段，私有嵌套结构体 `impl` 封装了 `mappings`（内存映射）、`mlock_bufs/mlock_mmaps`（内存锁定）、`ctxs_bufs`（后端上下文与缓冲区）、`cpu_buft_list/gpu_buft_list`（缓冲类型候选列表）、`dev_input/dev_layer/dev_output`（设备分配表）等**重资源**，实现了经典的 **Pimpl（Pointer to Implementation）模式**：头文件只暴露轻量接口，编译防火墙隔离了 GGML 后端细节，加速增量编译。

### 14.3.2 构造与析构：pimpl 初始化与 LoRA 回收

**源码路径：`src/llama-model.cpp` - `llama_model::llama_model()` 与 `~llama_model()`（第 271-280 行）**

```cpp
// 构造函数：记录参数并初始化 pimpl
llama_model::llama_model(const llama_model_params & params)
    : params(params), pimpl(std::make_unique<impl>()) {
    // tensor_buft_overrides 是否提供了自定义缓冲区覆盖规则
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

// 析构函数：遍历 loras 向量，逐个 delete 释放适配器内存
llama_model::~llama_model() {
    for (auto * lora : loras) {
        delete lora;
    }
}
```

构造函数仅做两件事：保存传入的 `params`、初始化 `pimpl` 智能指针，并根据 `params.tensor_buft_overrides` 设置 `has_tensor_overrides` 标志。析构函数体现了 **RAII** 原则：`pimpl` 自动释放内部所有 GGML 上下文、缓冲区、内存映射；`loras` 向量中每个 `llama_adapter_lora*` 显式 `delete`，确保 LoRA 适配器的 A/B 矩阵、缩放因子等资源被正确回收。

### 14.3.3 impl 内部结构：地下仓库的货架清单

**源码路径：`src/llama-model.cpp` - `llama_model::impl`（第 200-210 行）**

```cpp
// 私有实现：藏在地下仓库的重资源
struct llama_model::impl {
    impl() = default;
    ~impl() = default;

    // 统计信息
    uint64_t n_elements = 0;          // 总参数量
    size_t n_bytes = 0;               // 文件字节数
    std::string desc_str;             // 缓存的 desc() 字符串

    // 内存映射与锁定
    std::vector<llama_mmap> mappings;      // mmap 映射对象
    std::vector<ggml_backend_buffer_ptr> mlock_bufs; // mlock 缓冲区
    std::vector<llama_mlock> mlock_mmaps;  // mlock 内存锁定

    // 后端上下文与缓冲区：每个设备对应一组
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    // 缓冲类型候选列表
    buft_list_t cpu_buft_list;                              // CPU 侧候选
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list; // GPU 侧候选

    // 设备分配表
    struct { ggml_backend_dev_t dev; buft_list_t buft_list; } dev_input;
    std::vector<struct { ggml_backend_dev_t dev; buft_list_t buft_list; }> dev_layer;
    struct { ggml_backend_dev_t dev; buft_list_t buft_list; } dev_output;

    // 张量覆盖标志
    bool has_tensor_overrides = false;
};
```

`impl` 结构体是**编译防火墙**的核心：所有与 GGML 后端、内存管理、设备相关的类型（`ggml_context_ptr`、`ggml_backend_buffer_ptr`、`ggml_backend_dev_t`、`buft_list_t` 等）都只在这里出现，头文件 `llama-model.h` 完全不需要包含这些重头文件。`ctxs_bufs` 以向量形式存储每个设备的 `(ggml_context, buffer_list)` 对，支撑多设备并行推理。`dev_layer` 向量长度等于层数，每个元素记录该层归属的设备及其可用的 `buft_list`，供 `select_buft` 做能力探测时使用。

---

## 14.4 模型元数据加载流水线 —— 四阶段"档案审阅"的接力

四阶段流水线按顺序执行：`load_arch → load_hparams → load_vocab → load_tensors`。每阶段只做一件事，产出供下一阶段使用的数据结构，形成清晰的**流水线契约**。

### 14.4.1 load_arch：读取架构枚举，未知即报错

**源码路径：`src/llama-model.cpp` - `llama_model::load_arch()`（第 287-293 行）**

```cpp
void llama_model::load_arch(llama_model_loader & ml) {
    arch = ml.get_arch();                          // 从 GGUF 读取 general.architecture
    if (arch == LLM_ARCH_UNKNOWN) {                // 未知架构直接抛异常
        throw std::runtime_error(
            "unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}
```

这一步极其轻量：调用 `ml.get_arch()` 解析 GGUF 中的 `general.architecture` 字段，映射到 `llm_arch` 枚举。若无法识别（如新架构未在 `llama-arch.cpp` 注册），直接抛出异常终止加载，避免后续阶段在错误假设上浪费资源。

### 14.4.2 load_hparams：双轨制解析（字符串归档 + 结构化派生）

**源码路径：`src/llama-model.cpp` - `llama_model::load_hparams()`（第 295-694 行）**

```cpp
void llama_model::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;

    // ─── 第一轨：把所有 KV 字符串化归档到 gguf_kv ───
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) continue;     // 数组稍后按需解析
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);              // 供 llama_model_meta_* 查询
    }

    // ─── 通用键：模型名（非必需）───
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // ─── 纯词表模型（vocab_only）或 CLIP：只需张量，无需 hparams ───
    if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) {
        return;
    }

    // ─── 第二轨：结构化读取核心超参数 ───
    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,       hparams.n_expert_used,   false);
    // ... 省略若干通用 KV 读取 ...

    // ─── 巨型 switch：80+ 架构的特有 KV 解析 ───
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLAMA_EMBED: {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
            // 根据 n_layer 与 n_vocab 推断 type（1B/3B/7B/8B/13B/34B/70B...）
            // ...
        } break;
        case LLM_ARCH_LLAMA4: {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
            ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
            ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,   hparams.n_moe_layer_step);

            // ISWA 滑动窗口配置
            const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
            if (found_swa && hparams.n_swa == 0) {
                hparams.swa_type = LLAMA_SWA_TYPE_NONE;
                hparams.n_no_rope_layer_step = hparams.n_layer; // 全层 RoPE
            } else {
                hparams.swa_type = LLAMA_SWA_TYPE_CHUNKED;
                hparams.n_swa = 8192;
                // SWA 层专用 RoPE 频率
                hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
                hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
            }
            // ...
        } break;
        // ... 其余 70+ case 省略 ...
    }
}
```

`load_hparams` 采用**双轨制**：
1. **归档轨**：全量遍历 GGUF KV，非数组项全部 `gguf_kv_to_str` 存入 `gguf_kv`（`std::map<string,string>`），保留原始元数据供 `llama_model_meta_*` 运行时查询、日志打印、调试。
2. **派生轨**：通过 `ml.get_key` / `ml.get_key_or_arr` 结构化读取 `n_ctx_train`、`n_embd`、`n_layer`、`n_head_arr`、`n_ff_arr` 等数值字段，填充 `hparams`。

**关键细节**：
- `n_head_arr` 与 `n_head_kv_arr` 支持**逐层变化**（如 GQA、DeepSeek MLA），默认 `n_head_kv_arr = n_head_arr` 再尝试读取 `LLM_KV_ATTENTION_HEAD_COUNT_KV` 覆盖。
- `LLM_ARCH_CLIP` 与 `vocab_only` 模型直接 `return`，跳过后续超参数解析，因为它们不需要构建语言模型计算图。
- `rope_yarn_log_mul` 读取后**除以 0.1f**（`hparams.rope_yarn_log_mul /= 0.1f`），这是针对 DeepSeek-V2 早期导出 Bug 的兼容性修正（参考 `[TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]`）。
- `set_swa_pattern(swa_period)` 将滑动窗口周期（如 4 = 3 chunked + 1 full）编码进 `hparams.swa_layers` 位图，供 `hparams.is_swa(il)` 判断使用。

### 14.4.3 load_vocab：委托给 llama_vocab

**源码路径：`src/llama-model.cpp` - `llama_model::load_vocab()`（第 2405-2408 行）**

```cpp
void llama_model::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);           // 架构专用 KV 前缀
    vocab.load(ml, kv);                     // 委托词表加载
}
```

极简委托：`LLM_KV(arch)` 生成带架构前缀的键名（如 `tokenizer.ggml.tokens`），`vocab.load` 内部根据 `vocab_type` 分发到 SPM/BPE/WPM/UGM/RWKV/PLaMo2 等具体加载器，完成 tokens、scores、merges、special tokens 等字段填充。

### 14.4.4 load_tensors：模型灵魂的"组装车间"

**源码路径：`src/llama-model.cpp` - `llama_model::load_tensors()`（第 2410-3000+ 行）**

```cpp
bool llama_model::load_tensors(llama_model_loader & ml) {
    // ─── 1. 解析设备分配参数 ───
    const auto & split_mode   = params.split_mode;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;
    const int n_layer      = hparams.n_layer;
    const int n_gpu_layers = this->n_gpu_layers();

    // ─── 2. 构建 CPU/GPU 缓冲类型候选列表 ───
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
    for (auto * dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev, split_mode, tensor_split);
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end()); // CPU fallback
        pimpl->gpu_buft_list.emplace(dev, std::move(buft_list));
    }

    // ─── 3. 计算每层设备归属 dev_layer、dev_input、dev_output ───
    // （见下节详细解析）

    // ─── 4. 巨型 switch (arch)：逐层创建张量 ───
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_LLAMA4: {
            tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
            output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
            output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
            if (!output) output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

            for (int i = 0; i < n_layer; ++i) {
                auto & layer = layers[i];
                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k_gqa}, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_v_gqa}, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
                // ... FFN、MoE、RoPE、KV 归一化等张量 ...
            }
        } break;
        // ... 80+ 其他架构 case ...
    }

    // ─── 5. 通用 pass：加载 .scale 与 .input_scale（NVFP4 等量化场景）───
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        // 统一加载各张量的 scale
        if (layer.wq)       layer.wq_s      = create_tensor(tn(LLM_TENSOR_ATTN_Q, "scale", i), {1}, TENSOR_NOT_REQUIRED);
        if (layer.wk)       layer.wk_s      = create_tensor(tn(LLM_TENSOR_ATTN_K, "scale", i), {1}, TENSOR_NOT_REQUIRED);
        // ... 省略 ...
    }
    return true;
}
```

`load_tensors` 是**最重、最长**的函数（2000+ 行），核心逻辑分五步：
1. **参数解析**：读取 `split_mode`、`tensor_split`、`n_gpu_layers` 等。
2. **缓冲候选列表构建**：调用 `make_cpu_buft_list` / `make_gpu_buft_list`（详见下节）。
3. **设备分配表计算**：生成 `dev_input`、`dev_layer[]`、`dev_output`（详见下节）。
4. **架构分发创建张量**：一个超长 `switch (arch)`，每个 case 按该架构的张量命名约定，调用 `create_tensor` / `create_tensor_as_view` 填充 `layer.*` 字段。`TENSOR_NOT_REQUIRED` 标记可选张量（如 bias、output），`TENSOR_DUPLICATED` 让 `output` 复用 `tok_embd`（权重共享）。
5. **通用量化 scale pass**：遍历所有层，统一加载 `.scale` 与 `.input_scale` 张量，支持 NVFP4、MXFP4 等高级量化格式。

---

## 14.5 张量按设备分配策略 —— 把"楼层"分给"租户"

设备分配的核心目标：根据用户指定的 `n_gpu_layers` 与 `tensor_split`，将每一层的张量放到 CPU 或特定 GPU 上，生成 `dev_layer[i]`、`dev_input`、`dev_output` 三张表，供后续 `create_tensor` 选择 `buft` 使用。

### 14.5.1 make_cpu_buft_list：CPU 侧缓冲类型优先级装配

**源码路径：`src/llama-model.cpp` - `make_cpu_buft_list()`（第 177-235 行）**

```cpp
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices,
                                      bool use_extra_bufts, bool no_host) {
    buft_list_t buft_list;

    // ① ACCEL 设备（NPU 等加速器）的 host buffer 优先
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            if (buft != ggml_backend_cpu_buffer_type()) { // 排除纯 CPU buffer
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // ② Host buffer：用于大 batch 卸载到 GPU 时减少数据传输
    if (!no_host) {
        for (auto * dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
            if (buft) {
                buft_list.emplace_back(dev, buft);
                break; // 只取第一个
            }
        }
    }

    // ③ Extra buffer types：CPU 后端注册的额外 buffer（如 pinned memory）
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // ④ 最后兜底：标准 CPU buffer
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }
    return buft_list;
}
```

**优先级顺序**：ACCEL（NPU/DSP 等加速器的 host buffer）→ Host buffer（GPU 直连内存）→ Extra（pinned memory 等）→ 标准 CPU buffer。这种分层设计让**大 batch 推理时 embedding、output 等小张量留在 host buffer**，避免频繁 PCIe 传输；计算密集层则尽量落到 GPU/ACCEL buffer。

### 14.5.2 make_gpu_buft_list：GPU 侧缓冲类型装配（含 split 模式）

**源码路径：`src/llama-model.cpp` - `make_gpu_buft_list()`（第 237-268 行）**

```cpp
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev,
                                      llama_split_mode split_mode,
                                      const float * tensor_split) {
    buft_list_t buft_list;

    // ① Split buffer：按行切分张量跨多 GPU（需后端支持 ggml_backend_split_buffer_type）
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto split_buft_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (split_buft_fn) {
            size_t dev_index = ...; // 查找设备在注册表中的索引
            auto * buft = split_buft_fn(dev_index, tensor_split);
            if (buft) buft_list.emplace_back(dev, buft);
        }
    }

    // ② 设备默认 buffer
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // ③ Extra buffer types（如 GPU pinned memory）
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra_bufts_fn) {
        ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts_fn(dev);
        while (extra_bufts && *extra_bufts) {
            buft_list.emplace_back(dev, *extra_bufts);
            ++extra_bufts;
        }
    }
    return buft_list;
}
```

GPU 侧列表：**Split buffer（跨卡行切分）→ 默认 buffer → Extra buffer**。构建完成后，`load_tensors` 会把 `cpu_buft_list` 追加到每个 GPU 的列表末尾，形成 **GPU 优先、CPU 兜底** 的完整候选链。

### 14.5.3 设备分配表计算：dev_layer / dev_input / dev_output

**源码路径：`src/llama-model.cpp` - `llama_model::load_tensors()` 设备分配片段（第 2440-2480 行）**

```cpp
// tensor_split 为空或全零 → 按可用内存自动分配
std::vector<float> splits(n_devices());
if (tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float v){ return v == 0.0f; })) {
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] = ggml_backend_dev_free_memory(devices[i]); // 可用显存
    }
} else {
    splits.assign(tensor_split, tensor_split + n_devices());
}
// 归一化累积分布
float sum = std::accumulate(splits.begin(), splits.end(), 0.0f);
for (size_t i = 0; i < n_devices(); ++i) {
    splits[i] /= sum;
    if (i > 0) splits[i] += splits[i - 1];
}

// 计算实际卸载到 GPU 的层数
int i_gpu_start = std::max(n_layer + 1 - n_gpu_layers, 0);   // 从第几层开始上 GPU
int act_gpu_layers = std::min(n_gpu_layers, n_layer + 1);    // 实际上 GPU 的层数（含 output）

// dev_input 永远在 CPU
pimpl->dev_input.dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
pimpl->dev_input.buft_list = pimpl->cpu_buft_list;

// dev_layer：upper_bound 在 splits 累积数组中查表
auto get_layer_buft_list = [&](int il) -> buft_list_t & {
    if (il < i_gpu_start) return pimpl->cpu_buft_list; // 前几层留 CPU
    float pos = float(il - i_gpu_start) / act_gpu_layers; // 归一化位置 [0,1]
    size_t idx = std::upper_bound(splits.begin(), splits.begin() + n_devices(), pos) - splits.begin();
    return pimpl->gpu_buft_list.at(devices[idx]);        // 对应 GPU 的 buft_list
};
for (int il = 0; il < n_layer; ++il) {
    pimpl->dev_layer.emplace_back();
    pimpl->dev_layer.back().dev = get_layer_buft_list(il).front().first; // 取该 GPU 设备
    pimpl->dev_layer.back().buft_list = get_layer_buft_list(il);
}

// dev_output 跟随最后一层设备
pimpl->dev_output = pimpl->dev_layer.empty() ? pimpl->dev_input : pimpl->dev_layer.back();
```

**关键点**：
- `tensor_split` 为空 → **按可用显存比例自动分配**；非空 → **用户指定比例**（如 `1.0,0.5` 表示 GPU0 分 2/3、GPU1 分 1/3）。
- `i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0)`：`n_gpu_layers` 计入 output 层，故 `n_layer + 1`。若 `n_gpu_layers=20`、`n_layer=32`，则 `i_gpu_start=13`，第 0-12 层留 CPU，第 13-32 层+output 上 GPU。
- `act_gpu_layers = min(n_gpu_layers, n_layer + 1)`：防止用户指定过大导致除零。
- `upper_bound(splits, pos)`：`pos` 在 `[0,1]` 上均匀分布，落在哪个累积区间就分给哪张 GPU。这是**加权轮询**的标准算法。
- `dev_input` 永远 CPU：embedding 层极小且访问频繁，留在 CPU 减少 PCIe 往返。
- `dev_output = dev_layer.back()`：output 与最后一层同设备，避免最后一层 hidden state 跨设备拷贝再做 LM Head 投影。

---

## 14.6 层结构体与张量映射 —— "千面模型"的"零件仓库"

`llama_layer` 是**单层张量指针的集装箱**，设计容纳 80+ 架构的所有可能张量。通过 `switch (arch)` 在 `load_tensors` 中按需填充，未用字段保持 `nullptr`。

### 14.6.1 llama_layer：主容器（节选核心字段）

**源码路径：`src/llama-model.h` - `struct llama_layer`（第 1-180 行）**

```cpp
struct llama_layer {
    // ─── 归一化家族 ───
    ggml_tensor * attn_norm = nullptr;       // 注意力前归一化
    ggml_tensor * attn_norm_b = nullptr;     // 偏置（部分架构）
    ggml_tensor * attn_q_norm = nullptr;     // Q 归一化（Qwen、Gemma 等）
    ggml_tensor * attn_q_norm_b = nullptr;
    ggml_tensor * attn_k_norm = nullptr;     // K 归一化
    ggml_tensor * attn_k_norm_b = nullptr;
    ggml_tensor * attn_post_norm = nullptr;  // 注意力后归一化（DeepSeek、OpenAI-MoE）
    ggml_tensor * attn_post_norm_b = nullptr;

    // ─── 注意力权重 ───
    ggml_tensor * wq = nullptr;              // Q 投影
    ggml_tensor * wk = nullptr;              // K 投影
    ggml_tensor * wv = nullptr;              // V 投影
    ggml_tensor * wo = nullptr;              // O 投影
    ggml_tensor * wqkv = nullptr;            // 融合 QKV（ChatGLM、GLM 等）
    ggml_tensor * bq = nullptr;              // Q 偏置
    ggml_tensor * bk = nullptr;
    ggml_tensor * bv = nullptr;
    ggml_tensor * bo = nullptr;

    // MLA 低秩投影（DeepSeek-V2/3）
    ggml_tensor * wq_a = nullptr;            // Q 下投影
    ggml_tensor * wq_b = nullptr;            // Q 上投影
    ggml_tensor * wk_a = nullptr;            // KV 共享下投影
    ggml_tensor * wk_b = nullptr;            // K 上投影
    ggml_tensor * wv_b = nullptr;            // V 上投影
    ggml_tensor * wkv_a_mqa = nullptr;       // MQA 共享下投影
    ggml_tensor * wkv_b = nullptr;           // KV 上投影

    // ─── 稠密 FFN ───
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_norm_b = nullptr;
    ggml_tensor * ffn_gate = nullptr;        // Gate 投影（SwiGLU 等）
    ggml_tensor * ffn_gate_b = nullptr;
    ggml_tensor * ffn_up = nullptr;          // Up 投影
    ggml_tensor * ffn_up_b = nullptr;
    ggml_tensor * ffn_down = nullptr;        // Down 投影
    ggml_tensor * ffn_down_b = nullptr;

    // ─── MoE 专家网络（3D: n_embd × n_ff_exp × n_expert）───
    ggml_tensor * ffn_gate_inp = nullptr;    // 门控输入投影 (n_embd, n_expert)
    ggml_tensor * ffn_gate_inp_b = nullptr;
    ggml_tensor * ffn_gate_exps = nullptr;   // 专家 Gate 权重
    ggml_tensor * ffn_up_exps = nullptr;     // 专家 Up 权重
    ggml_tensor * ffn_down_exps = nullptr;   // 专家 Down 权重
    ggml_tensor * ffn_exp_probs_b = nullptr; // 专家偏置

    // ─── 共享专家（稀疏 MoE 补充）───
    ggml_tensor * ffn_gate_shexp = nullptr;
    ggml_tensor * ffn_up_shexp = nullptr;
    ggml_tensor * ffn_down_shexp = nullptr;

    // ─── SSM / Mamba 状态投影 ───
    ggml_tensor * ssm_in = nullptr;
    ggml_tensor * ssm_x = nullptr;
    ggml_tensor * ssm_dt = nullptr;
    ggml_tensor * ssm_a = nullptr;
    ggml_tensor * ssm_d = nullptr;
    ggml_tensor * ssm_conv1d = nullptr;
    ggml_tensor * ssm_dt_b = nullptr;
    ggml_tensor * ssm_A = nullptr;           // A_log 或离散化 A

    // ─── RWKV 时间/通道混合 ───
    ggml_tensor * time_mix_first = nullptr;
    ggml_tensor * time_mix_decay = nullptr;
    ggml_tensor * time_mix_lerp_k = nullptr; // RWKV6Qwen2 插值系数
    ggml_tensor * time_mix_lerp_v = nullptr;
    ggml_tensor * time_mix_lerp_g = nullptr;
    ggml_tensor * time_mix_lerp_w = nullptr;
    ggml_tensor * time_mix_lerp_k2 = nullptr; // RWKV7 扩展
    ggml_tensor * time_mix_lerp_v2 = nullptr;
    ggml_tensor * time_mix_lerp_a2 = nullptr;
    ggml_tensor * time_mix_lerp_g2 = nullptr;
    ggml_tensor * time_mix_lerp_k_a = nullptr;
    ggml_tensor * time_mix_lerp_v_a = nullptr;
    ggml_tensor * channel_mix_k = nullptr;
    ggml_tensor * channel_mix_r = nullptr;

    // ─── RoPE 频率因子 ───
    ggml_tensor * rope_long = nullptr;       // LongRoPe 长因子
    ggml_tensor * rope_short = nullptr;      // LongRoPe 短因子
    ggml_tensor * rope_freqs = nullptr;      // 标准 RoPE 预计算频率

    // ─── Gemma3n AltUp / Laurel ───
    ggml_tensor * per_layer_inp_gate = nullptr;
    ggml_tensor * per_layer_proj = nullptr;
    ggml_tensor * per_layer_post_norm = nullptr;
    ggml_tensor * altup_coef = nullptr;
    ggml_tensor * altup_coef_vec = nullptr;
    ggml_tensor * laurel_rank = nullptr;

    // ─── 特殊架构 ───
    ggml_tensor * attn_sinks = nullptr;      // OpenAI-MoE 注意力池
    ggml_tensor * indexer_attn_q_b = nullptr; // GLM-DSA 索引器
    ggml_tensor * indexer_attn_kv_b = nullptr;

    // ─── 内嵌子结构体 ───
    llama_layer_posnet   posnet;   // WavTokenizer
    llama_layer_convnext convnext; // ConvNeXt (SigLIP 等视觉塔)
    llama_layer_shortconv shortconv; // LFM2 短卷积
    llama_layer_nextn    nextn;    // NextN/MTP 预测层
};
```

**设计哲学**：**稀疏填充、统一容器**。不管是标准 Transformer、MoE、Mamba、RWKV、还是混合架构，所有张量指针都预先声明在 `llama_layer` 中。`load_tensors` 的 `switch (arch)` 只填充该架构需要的字段，其余保持 `nullptr`。这种**结构体联合**风格避免了虚函数派发开销，也方便 `print_info`、`memory_breakdown` 统一遍历。

### 14.6.2 子结构体：专项零件打包管理

**源码路径：`src/llama-model.h` - 子结构体定义（第 1-60 行）**

```cpp
// WavTokenizer PosNet：4 类 ResNet 块 + 注意力 + 输出归一化
struct llama_layer_posnet {
    ggml_tensor * norm1 = nullptr; ggml_tensor * norm1_b = nullptr;
    ggml_tensor * conv1 = nullptr; ggml_tensor * conv1_b = nullptr;
    ggml_tensor * norm2 = nullptr; ggml_tensor * norm2_b = nullptr;
    ggml_tensor * conv2 = nullptr; ggml_tensor * conv2_b = nullptr;
    ggml_tensor * norm3 = nullptr; ggml_tensor * norm3_b = nullptr;
    ggml_tensor * conv3 = nullptr; ggml_tensor * conv3_b = nullptr;
    ggml_tensor * norm4 = nullptr; ggml_tensor * norm4_b = nullptr;
    ggml_tensor * conv4 = nullptr; ggml_tensor * conv4_b = nullptr;
    ggml_tensor * attn_q = nullptr; ggml_tensor * attn_k = nullptr;
    ggml_tensor * attn_v = nullptr; ggml_tensor * attn_o = nullptr;
    ggml_tensor * norm = nullptr;   ggml_tensor * norm_b = nullptr;
};

// ConvNeXt 块：深度卷积 + LayerNorm + 点卷积 + gamma 缩放
struct llama_layer_convnext {
    ggml_tensor * dw = nullptr; ggml_tensor * dw_b = nullptr;
    ggml_tensor * norm = nullptr; ggml_tensor * norm_b = nullptr;
    ggml_tensor * pw1 = nullptr; ggml_tensor * pw1_b = nullptr;
    ggml_tensor * pw2 = nullptr; ggml_tensor * pw2_b = nullptr;
    ggml_tensor * gamma = nullptr;
};

// LFM2 短卷积：输入投影 → 卷积 → 输出投影
struct llama_layer_shortconv {
    ggml_tensor * in_proj = nullptr;
    ggml_tensor * conv = nullptr;
    ggml_tensor * out_proj = nullptr;
};

// NextN/MTP 预测层：eh_proj + embed_tokens + 归一化 + 共享头
struct llama_layer_nextn {
    ggml_tensor * eh_proj = nullptr;
    ggml_tensor * embed_tokens = nullptr;
    ggml_tensor * enorm = nullptr; ggml_tensor * enorm_b = nullptr;
    ggml_tensor * en_hnorm = nullptr; ggml_tensor * en_hnorm_b = nullptr;
    ggml_tensor * shared_head = nullptr; ggml_tensor * shared_head_b = nullptr;
};
```

每个子结构体对应一个**独立的子模块**（PosNet 音频位置编码、ConvNeXt 视觉塔、ShortConv 短卷积混合、NextN 多 Token 预测），内嵌在 `llama_layer` 中随层实例化。这种**组合优于继承**的设计，让主层结构体保持扁平，避免多态开销。

---

## 14.7 构建计算图：build_graph 的"调度总指挥"

`build_graph` 是 **"模型 → GGML 计算图"** 的总装车间。它根据 `arch` 实例化对应的 `llm_build_*` 模板类，调用其 `build()` 方法构建主干图，再统一接上 pooling/sampling/dense_out 尾部装配线，最后返回 `ggml_cgraph*` 给调度器执行。

### 14.7.1 build_graph：80+ 架构路由与尾部装配

**源码路径：`src/llama-model.cpp` - `llama_model::build_graph()`（第 8558-9171 行）**

```cpp
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm; // 智能指针管理图构建上下文

    // ─── 巨型 switch：架构 → 具体 builder 类 ───
    switch (arch) {
        case LLM_ARCH_LLAMA:
            llm = std::make_unique<llm_build_llama<false>>(*this, params); // 标准 LLaMA
            break;
        case LLM_ARCH_LLAMA4: {
            // Gemma3/4 同源：根据 swa_type 选择 ISWA 还是标准实现
            if (hparams.swa_type == LLAMA_SWA_TYPE_NONE) {
                llm = std::make_unique<llm_build_llama<false>>(*this, params);
            } else {
                llm = std::make_unique<llm_build_llama_iswa>(*this, params);
            }
        } break;
        case LLM_ARCH_LLAMA_EMBED:
            llm = std::make_unique<llm_build_llama<true>>(*this, params); // embedding 模式
            break;
        case LLM_ARCH_PHI3: {
            // Phi-3 模板特化：SWA 与非 SWA 双版本
            if (hparams.swa_type == LLAMA_SWA_TYPE_NONE) {
                llm = std::make_unique<llm_build_phi3<false>>(*this, params);
            } else {
                llm = std::make_unique<llm_build_phi3<true>>(*this, params);
            }
        } break;
        case LLM_ARCH_T5: {
            // Encoder-Decoder 分发：由 params.gtype 决定
            if (params.gtype == LLM_GRAPH_TYPE_ENCODER) {
                llm = std::make_unique<llm_build_t5_enc>(*this, params);
            } else {
                llm = std::make_unique<llm_build_t5_dec>(*this, params);
            }
        } break;
        case LLM_ARCH_GEMMA3: {
            // Gemma3 同理：ISWA 模板参数
            if (hparams.swa_type == LLAMA_SWA_TYPE_NONE) {
                llm = std::make_unique<llm_build_gemma3<false>>(*this, params);
            } else {
                llm = std::make_unique<llm_build_gemma3<true>>(*this, params);
            }
        } break;
        // ... 80+ 其他 case ...
        default:
            GGML_ABORT("fatal error: unsupported architecture in build_graph");
    }

    // ─── 统一的"尾部装配线" ───
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm); // 分类/池化头
    llm->build_sampling();                                        // 后端采样器子图
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers); // SentenceTransformers
    llm->res->set_outputs();                                      // 标记图输出张量
    return llm->res->get_gf();                                    // 返回 ggml_cgraph*
}
```

**核心机制**：
- **模板特化分发**：同一架构家族（LLaMA、Phi-3、Gemma3）通过 `llm_build_xxx<iswa>` 模板参数区分 SWA 与非 SWA 变体，避免代码重复。
- **Encoder/Decoder 分流**：T5 等架构通过 `params.gtype`（`LLM_GRAPH_TYPE_ENCODER/DECODER`）决定实例化 `llm_build_t5_enc` 还是 `llm_build_t5_dec`。
- **统一尾部**：所有 builder 共享 `llm_graph_context` 基类提供的 `build_pooling`、`build_sampling`、`build_dense_out`，最后 `set_outputs()` 标记 `t_logits`、`t_embd` 等为图输出，`get_gf()` 取出 `ggml_cgraph*` 交给 `ggml_backend_sched` 调度。

### 14.7.2 create_memory：五大缓存架构分支

**源码路径：`src/llama-model.cpp` - `llama_model::create_memory()`（第 8535-8652 行）**

```cpp
llama_memory_i * llama_model::create_memory(const llama_memory_params & params,
                                            const llama_cparams & cparams) const {
    llama_memory_i * res = nullptr;
    switch (arch) {
        // ① 无状态架构：Encoder、扩散、WavTokenizer 等直接返回 nullptr
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        // ... 省略 ...
        case LLM_ARCH_LLADA:
        case LLM_ARCH_RND1:
            res = nullptr;
            break;

        default:
            // ② 纯循环架构：Mamba/RWKV/DeltaNet
            if (llm_arch_is_recurrent(arch)) {
                res = new llama_memory_recurrent(*this, GGML_TYPE_F32, GGML_TYPE_F32,
                                                cparams.offload_kqv,
                                                std::max((uint32_t)1, cparams.n_seq_max),
                                                cparams.n_seq_max, nullptr);
            }
            // ③ 混合架构：Transformer + RNN（Jamba、Falcon-H1、Nemotron-H）
            else if (llm_arch_is_hybrid(arch)) {
                // Falcon-H1：全层既有注意力又有 SSM
                // Nemotron-H：按层交替（filter_attn/filter_recr 闭包区分）
                if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                    res = new llama_memory_hybrid_iswa(...); // 带 ISWA 的混合内存
                } else {
                    res = new llama_memory_hybrid(...);      // 标准混合内存
                }
            }
            // ④ 标准/ISWA Transformer
            else {
                llama_memory_i::layer_reuse_cb reuse = nullptr;
                // Gemma3n/Gemma4：KV cache 复用优化（后几层共享前层 KV）
                if (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4) {
                    reuse = [&](int32_t il) {
                        if (il >= (int32_t)hparams.n_layer_kv_from_start) {
                            return (int32_t)hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
                        }
                        return -1;
                    };
                }
                if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                    res = new llama_kv_cache_iswa(*this, params.type_k, params.type_v,
                                                  !cparams.flash_attn, cparams.offload_kqv,
                                                  params.swa_full, cparams.n_ctx_seq,
                                                  cparams.n_seq_max, cparams.n_ubatch, 1,
                                                  nullptr, reuse);
                } else {
                    res = new llama_kv_cache(*this, params.type_k, params.type_v,
                                             !cparams.flash_attn, cparams.offload_kqv,
                                             cparams.kv_unified, cparams.n_ctx_seq,
                                             cparams.n_seq_max, 1,
                                             hparams.n_swa, hparams.swa_type,
                                             nullptr, nullptr);
                }
            }
    }
    return res;
}
```

**五大分支对应表**：

| 架构类别 | 代表架构 | Memory 实现 | 关键特性 |
|----------|----------|-------------|----------|
| 无状态 | BERT、CLIP、LLaDA、RND1 | `nullptr` | 无 KV cache，前向即得 |
| 纯循环 | Mamba、Mamba2、RWKV6/7 | `llama_memory_recurrent` | R/S 状态按序列维护 |
| 混合 | Jamba、Falcon-H1、Nemotron-H | `llama_memory_hybrid` / `_iswa` | 双缓存：attn KV + recurrent state |
| ISWA Transformer | Gemma3、Gemma4、Mistral3 | `llama_kv_cache_iswa` | 基础 KV + SWA KV 双缓存 |
| 标准 Transformer | LLaMA、Qwen、Mistral、Falcon | `llama_kv_cache` | 单 KV cache，可选滑动窗口 |

`reuse` 闭包实现 **KV cache 层间共享**：Gemma3n/4 后几层（`n_layer_kv_from_start` 之后）复用前层的 KV buffer，节省显存。`is_swa(il)` 判断该层是否为 SWA 层，决定偏移 1 还是 2 层。

---

## 14.8 RoPE 频率与因子获取：位置编码的"调音台"

RoPE 参数采用**双轨制**：训练期固定值（`hparams.rope_freq_base_train` 等）与推理期可调值（`cparams.rope_freq_base` 等）并存，SWA 层另有独立频率。

### 14.8.1 get_rope_freq_base / get_rope_freq_scale：双源切换

**源码路径：`src/llama-model.cpp` - `get_rope_freq_base/scale`（第 8515-8521 行）**

```cpp
float llama_model::get_rope_freq_base(const llama_cparams & cparams, int il) const {
    // SWA 层使用训练时固定的 SWA 频率，普通层使用推理时可调频率
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}
```

`hparams.is_swa(il)` 内部查 `hparams.swa_layers[il/32] & (1u << (il%32))` 位图。SWA 层（如 Gemma3/4 的滑动窗口层）**锁定训练时频率**，防止推理时动态调整破坏窗口内相对位置关系；普通层允许用户通过 `--rope-freq-base`、`--rope-freq-scale` 运行时调整（YaRN 扩展上下文常用）。

### 14.8.2 get_rope_factors：LongRoPe 短长因子自动切换

**源码路径：`src/llama-model.cpp` - `get_rope_factors()`（第 8523-8533 行）**

```cpp
ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    const uint32_t n_ctx_seq = cparams.n_ctx_seq; // 当前序列上下文长度
    // 优先级 1：rope_freqs 张量存在（预计算的完整因子表，如 Phi-3、Gemma3n）
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }
    // 优先级 2：超过原始训练长度 → 用长因子
    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }
    // 优先级 3：否则用短因子
    return layers[il].rope_short;
}
```

**三级回退机制**：
1. `rope_freqs`：预计算的完整频率表（`n_rot/2` 维），用于 Phi-3、Gemma3n 等**逐层频率各异**的架构。
2. `rope_long`：LongRoPe 长因子（`n_ctx_seq > n_ctx_orig_yarn` 时），对应扩展上下文后的高频压缩因子。
3. `rope_short`：LongRoPe 短因子（基础上下文内），保持原始频率比例。

`load_tensors` 中仅当 GGUF 包含 `rope_factors_long/short` 或 `rope_freqs` 张量时才填充对应字段，否则均为 `nullptr` 走默认计算路径。

---

## 14.9 模型信息查询与打印："产品说明书"与"体检报告"

### 14.9.1 基础查询接口

**源码路径：`src/llama-model.cpp` - 基础查询方法（第 8202-8513 行）**

```cpp
std::string llama_model::arch_name() const { return llm_arch_name(arch); }
std::string llama_model::type_name() const { return llm_type_name(type); }
std::string llama_model::desc() const { return pimpl->desc_str; }
size_t llama_model::size() const { return pimpl->n_bytes; }
size_t llama_model::n_tensors() const { return tensors_by_name.size(); }
size_t llama_model::n_devices() const { return devices.size(); }
uint32_t llama_model::n_gpu_layers() const {
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer + 1; // -1 表示全卸载
}
llama_split_mode llama_model::split_mode() const { return params.split_mode; }

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (hparams.no_alloc) {
            // 无分配模式：查询虚拟大小
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}
uint64_t llama_model::n_elements() const { return pimpl->n_elements; }
ggml_backend_dev_t llama_model::dev_layer(int il) const { return pimpl->dev_layer.at(il).dev; }
ggml_backend_dev_t llama_model::dev_output() const { return pimpl->dev_output.dev; }
```

这些接口为**外部工具**（`llama-cli`、`llama-server`、`llama-bench`、Python 绑定）提供标准化的模型元信息访问。`memory_breakdown()` 按 `buft` 聚合，便于排查显存占用（如 `CUDA buffer: 12.3 GiB`, `CPU buffer: 4.1 GiB`）。

### 14.9.2 print_info：完整的"体检报告"

**源码路径：`src/llama-model.cpp` - `llama_model::print_info()`（第 8256-8461 行）**

```cpp
void llama_model::print_info() const {
    // 打印辅助 lambda：检测数组是否逐层变化
    auto print_f = [](const std::function<uint32_t(uint32_t)> & f, uint32_t n) { ... };

    LLAMA_LOG_INFO("%s: arch                  = %s\n", __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only            = %d\n", __func__, hparams.vocab_only);
    LLAMA_LOG_INFO("%s: no_alloc              = %d\n", __func__, hparams.no_alloc);
    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train           = %u\n", __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd                = %u\n", __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_layer               = %u\n", __func__, hparams.n_layer);
        // 逐层变化的字段用 print_f 打印数组
        LLAMA_LOG_INFO("%s: n_head                = %s\n", __func__,
                       print_f([&](uint32_t il){ return hparams.n_head(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv             = %s\n", __func__,
                       print_f([&](uint32_t il){ return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_rot                 = %u\n", __func__, hparams.n_rot_full);
        LLAMA_LOG_INFO("%s: n_swa                 = %u\n", __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n", __func__, hparams.f_norm_rms_eps);
        // ... 省略几十行超参数打印 ...

        // 架构专用扩展打印
        if (arch == LLM_ARCH_DEEPSEEK) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n", __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_lora_q              = %d\n", __func__, hparams.n_lora_q);
            LLAMA_LOG_INFO("%s: n_lora_kv             = %d\n", __func__, hparams.n_lora_kv);
            LLAMA_LOG_INFO("%s: n_embd_head_k_mla     = %d\n", __func__, hparams.n_embd_head_k_mla());
        }
        // ... Qwen MoE、Granite、MiniCPM、BailingMoE、GroveMoE 等专用字段 ...
    }
    // 总参数量格式化
    if (pimpl->n_elements >= 1e12) LLAMA_LOG_INFO("%s: model params = %.2f T\n", ...);
    else if (...) ...
    LLAMA_LOG_INFO("%s: general.name          = %s\n", __func__, name.c_str());
    vocab.print_info(); // 最后打印词表信息
}
```

`print_info` 是**调试首选入口**：它把 `hparams` 中 100+ 字段按分类（架构、维度、注意力、FFN、MoE、RoPE、SSM、架构专用）逐行打印。`print_f` lambda 自动检测数组是否逐层变化：若全层相同只打印单值，否则打印 `[v0, v1, v2, ...]` 完整数组，极大压缩日志体积同时保留细节。

---

## 14.10 计算图输入抽象：llm_graph_input_i 家族的"纸张入口"

`llm_graph_input_i` 及其派生类将**计算图所需的外部输入**（token ids、位置、KV cache 索引、掩码、循环状态等）抽象为统一的"纸张输入口"，每种注意力机制、内存模型对应一种具体输入类。

### 14.10.1 基类与核心虚函数

**源码路径：`src/llama-graph.h` - `llm_graph_input_i`（第 100-117 行）**

```cpp
class llm_graph_input_i {
public:
    llm_graph_input_i() {
        const char * LLAMA_GRAPH_INPUT_DEBUG = getenv("LLAMA_GRAPH_INPUT_DEBUG");
        debug = LLAMA_GRAPH_INPUT_DEBUG ? atoi(LLAMA_GRAPH_INPUT_DEBUG) : 0;
    }
    virtual ~llm_graph_input_i() = default;

    // 每次 forward 前调用：把当前 ubatch 数据写入张量
    virtual void set_input(const llama_ubatch * ubatch) = 0;

    // 图复用判定：当前输入张量是否兼容新参数
    virtual bool can_reuse(const llm_graph_params & params) { return false; }

    bool debug = false;
};
```

- `set_input(ubatch)`：推理循环每步调用，将 `ubatch.token`、`pos`、`seq_id` 等写入对应 `ggml_tensor`。
- `can_reuse(params)`：`llm_graph_result::can_reuse` 调用，判断旧图输入是否兼容新 batch（如 token 数、序列数、seq_id 组合是否相同）。

### 14.10.2 注意力输入的五种"纸张"

| 输入类 | 对应模型/场景 | 核心张量 |
|--------|---------------|----------|
| `llm_graph_input_attn_no_cache` | BERT 等编码器、无缓存注意力 | `kq_mask`、`kq_mask_swa` |
| `llm_graph_input_attn_kv` | 标准 Transformer 解码器 | `k_idxs`、`v_idxs`、`kq_mask` |
| `llm_graph_input_attn_k` | V-less 优化（仅缓存 K） | `k_idxs`、`kq_mask` |
| `llm_graph_input_attn_kv_iswa` | Gemma3/4、Mistral3 ISWA 双缓存 | `k_idxs`/`v_idxs` + `k_idxs_swa`/`v_idxs_swa` + 双掩码 |
| `llm_graph_input_attn_cross` | T5 Decoder 交叉注意力 | `cross_kq_mask` |

**源码片段（`llm_graph_input_attn_kv` 节选）**：
```cpp
class llm_graph_input_attn_kv : public llm_graph_input_i {
    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_kv_cache_context * mctx; // KV cache 上下文
    ggml_tensor * self_k_idxs = nullptr;
    ggml_tensor * self_v_idxs = nullptr;
    ggml_tensor * self_kq_mask = nullptr;
    ggml_tensor * self_kq_mask_cnv = nullptr; // 转换后的掩码

public:
    llm_graph_input_attn_kv(const llama_hparams & hparams,
                            const llama_cparams & cparams,
                            const llama_kv_cache_context * mctx)
        : hparams(hparams), cparams(cparams), mctx(mctx) {}
    void set_input(const llama_ubatch * ubatch) override; // 填充 k/v idxs、因果掩码
    bool can_reuse(const llm_graph_params & params) override; // 检查 mctx、ubatch 兼容性
    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }
    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }
};
```

### 14.10.3 混合内存输入：把 attn 与 rs 打包

**源码路径：`src/llama-graph.h` - `llm_graph_input_mem_hybrid`（第 449-469 行）**

```cpp
class llm_graph_input_mem_hybrid : public llm_graph_input_i {
    std::unique_ptr<llm_graph_input_attn_kv> inp_attn; // 注意力侧输入
    std::unique_ptr<llm_graph_input_rs> inp_rs;        // 循环状态侧输入
    const llama_cparams & cparams;
    const llama_memory_hybrid_context * mctx;

public:
    llm_graph_input_mem_hybrid(const llama_cparams & cparams,
                               std::unique_ptr<llm_graph_input_attn_kv> inp_attn,
                               std::unique_ptr<llm_graph_input_rs> inp_rs,
                               const llama_memory_hybrid_context * mctx)
        : inp_attn(std::move(inp_attn)), inp_rs(std::move(inp_rs)),
          cparams(cparams), mctx(mctx) {}

    void set_input(const llama_ubatch * ubatch) override {
        inp_attn->set_input(ubatch); // 委托给两个子输入
        inp_rs->set_input(ubatch);
    }
    bool can_reuse(const llm_graph_params & params) override {
        return inp_attn->can_reuse(params) && inp_rs->can_reuse(params); // 双方都可复用才行
    }
    llm_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs * get_recr() const { return inp_rs.get(); }
};
```

**组合模式**：`mem_hybrid` 将 `attn_kv` 与 `rs` 两个输入对象**组合**而非继承，`set_input`/`can_reuse` 直接委托。`_k` 版本用于 V-less 优化，`_iswa` 版本用于 Nemotron-H 等带 ISWA 的混合架构。

### 14.10.4 循环状态输入：RNN 的读写口

**源码路径：`src/llama-graph.h` - `llm_graph_input_rs`（第 249-254 行）**

```cpp
class llm_graph_input_rs : public llm_graph_input_i {
    const llama_memory_recurrent_context * mctx;
    // s_copy_main / s_copy_extra：用于 seq_id_unq 去重时的状态拷贝
    ggml_tensor * s_copy_main = nullptr;
    ggml_tensor * s_copy_extra = nullptr;

public:
    llm_graph_input_rs(const llama_memory_recurrent_context * mctx) : mctx(mctx) {}
    void set_input(const llama_ubatch * ubatch) override; // 写入当前 token 的 R/S 状态
    bool can_reuse(const llm_graph_params & params) override; // 检查 mctx、seq_id 兼容
};
```

`s_copy_main` 与 `s_copy_extra` 对应 **`equal_seqs` 模式**下的状态去重：当多序列共享同一 KV cache 流时，仅为唯一 `seq_id` 维护一份 R/S 状态，推理时通过 `ggml_get_rows` 复制到对应位置。

---

## 14.11 计算图产物与复用：llm_graph_result 的"成品书"

### 14.11.1 核心字段：图构建的所有产出

**源码路径：`src/llama-graph.h` - `llm_graph_result`（第 704-732 行）**

```cpp
class llm_graph_result {
    // ─── 关键输出张量 ───
    ggml_tensor * t_inp_tokens = nullptr;    // 输入 token ids [n_tokens]
    ggml_tensor * t_inp_embd = nullptr;      // 输入嵌入 [n_embd, n_tokens]
    ggml_tensor * t_logits = nullptr;        // 语言模型 logits [n_vocab, n_tokens]
    ggml_tensor * t_embd = nullptr;          // 句向量/pooling 输出 [n_embd, n_seq]
    ggml_tensor * t_embd_pooled = nullptr;   // 池化后嵌入

    // ─── 采样相关张量表（按 seq_id 索引）───
    std::map<llama_seq_id, ggml_tensor *> t_sampled_logits; // 采样后 logits
    std::map<llama_seq_id, ggml_tensor *> t_candidates;     // 候选 token
    std::map<llama_seq_id, ggml_tensor *> t_sampled;        // 采样结果 token
    std::map<llama_seq_id, ggml_tensor *> t_sampled_probs;  // 采样概率

    // ─── 图元数据 ───
    std::vector<llm_graph_input_ptr> inputs; // 所有输入对象
    ggml_context_ptr ctx_compute;            // 计算用 ggml_context
    ggml_cgraph * gf = nullptr;              // 计算图指针
    int64_t max_nodes = 0;                   // 最大节点数

    // ─── 复用与生命周期 ───
    bool can_reuse(const llm_graph_params & params); // 综合所有 inputs->can_reuse
    void set_inputs(const llama_ubatch * ubatch);    // 批量调用 inputs[i]->set_input
    void set_outputs();                              // 标记 t_logits/t_embd 等为图输出
    void reset();                                    // 释放图与上下文
    llm_graph_input_i * add_input(llm_graph_input_ptr input);
    void set_params(const llm_graph_params & params);
};
```

`t_sampled_logits` 等 `map<seq_id, tensor*>` 支持**多序列并发采样**：每个序列独立维护采样状态，`llama_context::decode` 循环中按 `seq_id` 取对应张量。

### 14.11.2 双层复用判定：allow_reuse 与 can_reuse

**源码路径：`src/llama-graph.h` - `llm_graph_params::allow_reuse()`（第 642-692 行）**

```cpp
bool llm_graph_params::allow_reuse(const llm_graph_params & other) const {
    // 1. ubatch 拓扑一致性
    bool can_reuse_ubatch = ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
                            ubatch.n_tokens == other.ubatch.n_tokens &&
                            ubatch.n_seqs == other.ubatch.n_seqs &&
                            ubatch.n_seqs_unq == other.ubatch.n_seqs_unq &&
                            (!ubatch.token && !other.ubatch.token || !ubatch.embd && !other.ubatch.embd);

    // 2. equal_seqs 模式下：seq_id_unq 必须完全相同
    if (can_reuse_ubatch && ubatch.equal_seqs()) {
        if (!ubatch.data) can_reuse_ubatch = false; // 旧数据已释放不可复用
        else {
            for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
            }
        }
    }
    if (!can_reuse_ubatch) return false;

    // 3. 输出数量、采样器、cparams、arch、gtype、cvec、loras、cross 全等
    return n_outputs == other.n_outputs &&
           samplers_equal(samplers, other.samplers) &&
           cparams.embeddings == other.cparams.embeddings &&
           cparams.causal_attn == other.cparams.causal_attn &&
           arch == other.arch && gtype == other.gtype &&
           cvec == other.cvec && loras == other.loras && cross == other.cross;
}
```

**两层校验**：
- **`llm_graph_params::allow_reuse`**：**拓扑级**判定。比较 `ubatch` 形状、序列分布、采样器配置、上下文参数、架构类型等**静态拓扑要素**。只有拓扑完全一致，图结构才可复用。
- **`llm_graph_input_i::can_reuse`**：**输入级**判定。各输入子类重写，检查 `n_tokens`、`mctx` 指针、`samplers` map 等**运行时状态**是否兼容。

**源码路径：`src/llama-graph.h` - `llm_graph_result::can_reuse()`（第 728 行）**

```cpp
bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    // 先检查 params 级拓扑复用
    if (!this->params.allow_reuse(params)) return false;
    // 再检查每个输入对象级复用
    for (auto & input : inputs) {
        if (!input->can_reuse(params)) return false;
    }
    return true; // 双层通过才真正复用
}
```

只有**两层都返回 true**，`llama_context::decode` 才会跳过 `build_graph` 直接复用旧 `gf`，节省构图开销（典型场景：多轮对话同 batch size、同参数）。

---

## 14.12 计算图上下文与公共子图：llm_graph_context 的"装订工坊"

`llm_graph_context` 是**图构建的工具箱**，持有 `hparams`、`cparams`、`ubatch`、`sched`、`backend_cpu`、`cvec`、`loras`、`mctx`、`samplers` 等上下文，提供 30+ 个 `build_*` 公共子图构造函数，供具体 `llm_build_*` 类调用组装完整图。

### 14.12.1 归一化与 FFN：通用机器

**源码路径：`src/llama-graph.h` - `build_norm` / `build_ffn`（第 876-899 行）**

```cpp
// 归一化：type 切换 LN / RMSNorm / GroupNorm
ggml_tensor * build_norm(ggml_tensor * cur,
                         ggml_tensor * mw, ggml_tensor * mb,
                         llm_norm_type type, int il) const;

// FFN：type_op 切换 SILU/GELU/RELU/RELU_SQR/SWIGLU/GEGLU/REGLU
// type_gate 切换 SEQ（串行 gate→up）或 PAR（gate 与 up 并行）
ggml_tensor * build_ffn(ggml_tensor * cur,
                        ggml_tensor * up, ggml_tensor * up_b, ggml_tensor * up_s,
                        ggml_tensor * gate, ggml_tensor * gate_b, ggml_tensor * gate_s,
                        ggml_tensor * down, ggml_tensor * down_b, ggml_tensor * down_s,
                        ggml_tensor * act_scales,
                        llm_ffn_op_type type_op, llm_ffn_gate_type type_gate, int il) const;
```

- `build_norm`：`type=LLM_NORM` (LayerNorm)、`LLM_NORM_RMS` (RMSNorm)、`LLM_NORM_GROUP` (GroupNorm)。`mw`/`mb` 为 weight/bias，可能为 `nullptr`（RMSNorm 无 bias）。
- `build_ffn`：`type_op` 覆盖主流激活；`type_gate=LLM_FFN_SEQ` 先 gate 再 up（标准 SwiGLU），`LLM_FFN_PAR` gate 与 up 并行（OpenAI-MoE 等）。

### 14.12.2 MoE FFN：双版本统一子图

**源码路径：`src/llama-graph.h` - `build_moe_ffn` 两个重载（第 903-944 行）**

```cpp
// 版本 1：不带 bias（Qwen2MoE、Qwen3MoE 等）
ggml_tensor * build_moe_ffn(ggml_tensor * cur,
                            ggml_tensor * gate_inp,
                            ggml_tensor * up_exps, ggml_tensor * gate_exps,
                            ggml_tensor * down_exps, ggml_tensor * exp_probs_b,
                            int64_t n_expert, int64_t n_expert_used,
                            llm_ffn_op_type type_op, bool norm_w, float w_scale,
                            llama_expert_gating_func_type gating_op, int il,
                            ggml_tensor * probs_in = nullptr, ... ) const;

// 版本 2：带 bias（OpenAI-MoE、GLM4MoE 等）
ggml_tensor * build_moe_ffn(ggml_tensor * cur,
                            ggml_tensor * gate_inp, ggml_tensor * gate_inp_b,
                            ggml_tensor * up_exps, ggml_tensor * up_exps_b,
                            ggml_tensor * gate_exps, ggml_tensor * gate_exps_b,
                            ggml_tensor * down_exps, ggml_tensor * down_exps_b,
                            ggml_tensor * exp_probs_b, ... ) const;
```

**调用点选择**：`llm_build_qwen2moe` 调用版本 1（无 bias）；`llm_build_openai_moe` 调用版本 2（带 `bq/bk/bv/bo` 与 `ffn_gate_inp_b` 等 bias）。内部统一实现：**gate_inp → softmax/sigmoid → top-k → 专家加权求和 → down**，`norm_w` 控制是否对专家权重做 RMSNorm，`w_scale` 用于量化反量化缩放。

### 14.12.3 多头注意力：build_attn_mha 统一入口

**源码路径：`src/llama-graph.h` - `build_attn_mha`（第 963-973 行）**

```cpp
ggml_tensor * build_attn_mha(ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
                             ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
                             ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens]
                             ggml_tensor * kq_b,    // QK 偏置（ALiBi、相对位置）
                             ggml_tensor * kq_mask, // 因果/滑动窗口掩码
                             ggml_tensor * sinks,   // Attention sinks (OpenAI-MoE)
                             ggml_tensor * v_mla,   // MLA V 吸收矩阵 [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                             float kq_scale, int il) const;
```

**扩展参数**：
- `kq_b`：ALiBi bias、T5 相对位置桶 bias、或 DeepSeek MLA 的 `kq_scale` 融合。
- `sinks`：`[n_head_q]` 向量，实现 Attention Sinks（首 token 保持注意力）。
- `v_mla`：MLA（Multi-head Latent Attention）的 V 吸收投影，将低秩 KV 恢复到完整头维。
- `kq_scale`：推理时动态缩放（YaRN、LongRoPE）。

内部根据 `params.cparams.flash_attn`、`hparams.causal_attn`、`hparams.swa_type` 自动选择 **Flash Attention**、**标准 SDPA**、**ISWA 双流** 等后端实现。

### 14.12.4 循环状态构建：build_rs 与视图优化

**源码路径：`src/llama-graph.h` - `build_rs`（第 1070-1090 行）**

```cpp
ggml_tensor * build_rs(ggml_tensor * s,                    // 当前状态 [state_size, n_seqs]
                       ggml_tensor * state_copy_main,      // 主序列拷贝索引
                       ggml_tensor * state_copy_extra,     // 额外序列拷贝索引
                       int32_t state_size, int32_t n_seqs,
                       uint32_t n_rs, uint32_t rs_head, uint32_t rs_size,
                       int32_t rs_zero,
                       const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;
```

- `get_state_rows` 回调默认 `ggml_get_rows`，可注入自定义实现（如 `ggml_get_rows_cpy` 避免拷贝）。
- `state_copy_main` / `state_copy_extra`：**视图而非拷贝**。当 `equal_seqs` 模式下多序列共享状态时，仅为唯一 `seq_id` 分配物理存储，推理时通过 `get_rows` 将同一状态"广播"到多个序列槽位，避免 `ggml_cpy` 开销。

### 14.12.5 输入构建工厂：按需生产"纸张"

**源码路径：`src/llama-graph.h` - `build_inp_*` / `build_attn_inp_*`（第 948-1060 行）**

```cpp
// 基础输入
ggml_tensor * build_inp_embd(ggml_tensor * tok_embd) const;
ggml_tensor * build_inp_pos() const;
ggml_tensor * build_inp_attn_scale() const;     // Llama4 temperature scaling
ggml_tensor * build_inp_out_ids() const;        // 标记需采样的 token
ggml_tensor * build_inp_mean() const;           // embedding mean pooling
ggml_tensor * build_inp_cls() const;            // 分类头 token 选择

// T5 专用
ggml_tensor * build_inp_cross_embd() const;
ggml_tensor * build_inp_pos_bucket_enc() const;
ggml_tensor * build_inp_pos_bucket_dec() const;

// Attention 输入工厂：根据 mctx 类型自动分发
llm_graph_input_attn_no_cache * build_attn_inp_no_cache() const;
llm_graph_input_attn_kv       * build_attn_inp_kv() const;
llm_graph_input_attn_k        * build_attn_inp_k() const;
llm_graph_input_attn_kv_iswa  * build_attn_inp_kv_iswa() const;
llm_graph_input_attn_cross    * build_attn_inp_cross() const;

// 混合内存输入工厂
llm_graph_input_mem_hybrid      * build_inp_mem_hybrid() const;
llm_graph_input_mem_hybrid_k    * build_inp_mem_hybrid_k() const;
llm_graph_input_mem_hybrid_iswa * build_inp_mem_hybrid_iswa() const;

// 循环状态输入
llm_graph_input_rs * build_rs_inp() const;
```

这些工厂方法**封装了输入对象的构造细节**（如 `llm_graph_input_attn_kv` 需要 `hparams`、`cparams`、`mctx`），具体 builder 类只需调用 `llm->build_attn_inp_kv()` 即可获得配置好的输入对象，再传给 `build_attn`。

### 14.12.6 尾部装配：Pooling、Sampling、Dense Out

**源码路径：`src/llama-graph.h` - `build_pooling` / `build_sampling` / `build_dense_out`（第 1113-1130 行）**

```cpp
// 分类/池化头：cls token 或 mean pooling
void build_pooling(ggml_tensor * cls, ggml_tensor * cls_b,
                   ggml_tensor * cls_out, ggml_tensor * cls_out_b,
                   ggml_tensor * cls_norm) const;

// 后端采样器：将 sampler 链接入图，实现 GPU 端采样
void build_sampling() const;

// SentenceTransformers 等密集输出投影
void build_dense_out(ggml_tensor * dense_2, ggml_tensor * dense_2_b,
                     ggml_tensor * dense_3) const;
```

- `build_pooling`：BERT、嵌入模型使用，将 `t_embd` 池化为 `t_embd_pooled`。
- `build_sampling`：启用 `cparams.use_backend_sampling` 时，把 `llama_sampler_chain` 编译进图，实现**全 GPU 端采样**，省去 logits 回传 CPU 开销。
- `build_dense_out`：双塔/三塔嵌入模型的最终投影层。

### 14.12.7 LoRA 与控制向量注入

**源码路径：`src/llama-graph.h` - `build_lora_mm` / `build_cvec`（第 864-873 行）**

```cpp
// 矩阵乘 + 可选 LoRA + 可选 per-tensor scale
ggml_tensor * build_lora_mm(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * w_s = nullptr) const;

// 带索引的矩阵乘（专家路由场景）：w 形状 [n_embd, n_ff_exp, n_expert]，ids 选专家
ggml_tensor * build_lora_mm_id(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * ids) const;

// 控制向量：将 cvec 加到指定层隐藏状态
ggml_tensor * build_cvec(ggml_tensor * cur, int il) const;
```

`build_lora_mm` 实现 **`cur @ w^T + cur @ lora_A @ lora_B * scale`**，`w_s` 用于 NVFP4 等量化权重的逐张量反量化缩放。`build_lora_mm_id` 处理 MoE 专家权重的索引化矩阵乘。

---

## 14.13 设计中的取舍

**为什么不用虚函数多态来处理不同架构的 `load_tensors` 与 `build_graph`？**

llama.cpp 采用**巨型 `switch (arch)` + 模板特化**而非虚函数派发：
- **性能**：虚函数调用阻碍内联，`switch` 编译期可完全展开，构图热路径零开销。
- **编译期分发**：模板参数 `<iswa>`、`<embed>` 让编译器为每个变体生成专用代码，避免运行时分支预测失败。
- **代码复用**：`llm_build_llama<false>` 与 `<true>` 共享 90% 逻辑，仅在注意力输入、KV cache 类型处插入差异分支。

**Pimpl 模式的 trade-off 是什么？**

- **优点**：头文件极简（不含 GGML 类型），编译防火墙加速增量构建；二进制兼容性强，内部字段增减不影响 ABI。
- **缺点**：每次访问 `pimpl->xxx` 多一次指针间接寻址；调试时需在 debugger 中展开 `pimpl` 才能看到真实字段。

**为什么 `dev_input` 永远绑定 CPU？**

Embedding 查表（`ggml_get_rows`）是**内存带宽受限**算子，张量极小（`[n_embd, n_tokens]`），PCIe 传输开销远大于计算收益。实测将 embedding 留在 CPU、仅 hidden states 上 GPU，**端到端延迟更低**。

**`can_reuse` 双层校验的必要性？**

- `allow_reuse` 检查**静态拓扑**（batch 形状、采样器配置、架构），防止图结构不匹配导致崩溃。
- `can_reuse` 检查**动态状态**（KV cache 指针、序列 ID 分布、sampler 内部状态），防止复用旧图时读取脏数据。
- 两层缺一不可：仅拓扑同不够（如 KV cache 已满需扩容），仅状态同不够（如 batch size 变了图节点数不同）。

---

## 14.14 动手练习

### 14.14.1 练习 1：追踪 llama_model 的生命周期
阅读 `src/llama-model.h` 与 `src/llama-model.cpp` 中构造函数与析构函数的实现，回答问题：
1. 构造时 llama_model 如何利用 Pimpl 模式隔离哪些重资源？
2. 析构时如何保证所有已关联的 LoRA 适配器被正确回收？
3. `pimpl->has_tensor_overrides` 的初始化逻辑与 `params.tensor_buft_overrides` 是什么关系？
4. impl 默认析构是否能释放 `ctxs_bufs` 中的 `ggml_backend_buffer`？

### 14.14.2 练习 2：解剖 load_hparams 的 GGUF KV 解析
阅读 `src/llama-model.cpp` 中 `llama_model::load_hparams()` 完整实现，回答问题：
1. 第一段循环 `for (int i = 0; i < gguf_get_n_kv(ctx); i++)` 把所有 KV 字符串化存入 `gguf_kv` 的目的是什么？
2. 读取 `LLM_KV_GENERAL_NAME` 时为何 `required=false`？
3. 当模型是 CLIP（`LLM_ARCH_CLIP`）时为何只加载 tensors、不加载 hparams？
4. 找到为 `LLM_ARCH_LLAMA4` 解析 ISWA 滑动窗口 KV 的代码，描述 `n_no_rope_layer_step` 在什么情况下被设为 `n_layer`？
5. 解释 `rope_yarn_log_mul` 字段的 0.1f 后处理的目的（参考 `[TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]`）。

### 14.14.3 练习 3：理解张量按设备分配
阅读 `src/llama-model.cpp` 中 `llama_model::load_tensors()` 中关于设备分配的部分，回答问题：
1. `tensor_split == nullptr` 或全零时如何按可用内存自动分配？
2. `i_gpu_start = std::max(int(hparams.n_layer) + 1 - n_gpu_layers, 0)` 的含义是什么？
3. `act_gpu_layers` 为什么是 `std::min(n_gpu_layers, int(n_layer) + 1)`？
4. 在 `get_layer_buft_list` 闭包中，`std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers)` 的累计分布查找如何把层分配到 GPU？
5. 为什么 `dev_input` 永远指向 CPU 设备？
6. `dev_output = get_layer_buft_list(n_layer)` 意味着输出层与最后一层设备相同，这是出于什么考量？

### 14.14.4 练习 4：对比 build_graph 中的架构路由
阅读 `src/llama-model.cpp` 中 `llama_model::build_graph()` 的 switch 语句（80+ case），回答问题：
1. `LLM_ARCH_LLAMA4` 为何根据 `swa_type` 选择 `llm_build_llama<false>` 或 `llm_build_llama_iswa`？
2. `LLM_ARCH_PHI3` 如何用模板特化处理 SWA 与非 SWA 变体？
3. `LLM_ARCH_T5` 通过什么参数在 encoder / decoder 之间分发？
4. 找到 `LLM_ARCH_GEMMA3` 的 case，描述 `llm_build_gemma3<true>` 与 `llm_build_gemma3<false>` 的选择条件。
5. `build_graph` 末尾依次调用 `build_pooling` / `build_sampling` / `build_dense_out` / `set_outputs`，各自的目的是什么？

### 14.14.5 练习 5：跟踪 RoPE 参数的双源切换
阅读 `src/llama-model.cpp` 中 `get_rope_freq_base` / `get_rope_freq_scale` / `get_rope_factors` 的实现，回答问题：
1. SWA 层与普通层为何使用使用不同的频率源？
2. `hparams.is_swa(il)` 的判定逻辑在 `llama_hparams` 中如何实现？
3. `get_rope_factors` 中三种优先级：`rope_freqs` > `rope_long` > `rope_short` 的判断条件分别是什么？
4. 在何种情况下 `rope_freqs` 不为空（由 `load_tensors` 何时填充）？

### 14.14.6 练习 6：理解 create_memory 的五大分支
阅读 `src/llama-model.cpp` 中 `llama_model::create_memory()` 的完整实现，回答问题：
1. 哪些架构直接返回 `nullptr`（无状态）？为何这样设计？
2. `llm_arch_is_recurrent(arch)` 与 `llm_arch_is_hybrid(arch)` 如何判定，分别落到哪种 memory？
3. Falcon-H1 与 Nemotron-H 作为 hybrid 架构，`layer_filter` 的差异在哪里？
4. Gemma3n / Gemma4 通过 `reuse` lambda 实现什么优化？`hparams.n_layer_kv_from_start` 的含义？
5. 当 `swa_type != LLAMA_SWA_TYPE_NONE` 但不是 hybrid 时，为何选择 `llama_kv_cache_iswa` 而非 `llama_kv_cache`？

### 14.14.7 练习 7：摸清 llama_layer 的张量全景
阅读 `src/llama-model.h` 中 `struct llama_layer` 完整字段，回答问题：
1. 归一化家族包含哪些成员？分别对应 attn/ffn/moe/post/output 哪几个位置？
2. MoE 专家张量（`ffn_gate_exps` / `ffn_down_exps` / `ffn_up_exps`）与共享专家张量（`ffn_*_shexp`）的形状区别是什么？
3. `ffn_latent_down` / `ffn_latent_up` 出现在哪些架构？推测其用途。
4. `per_layer_inp_gate` / `per_layer_proj` / `per_layer_post_norm` / `altup_*` / `laurel_*` 这些字段对应哪个特殊架构？
5. RWKV7 相比 RWKV6 多出哪些 `time_mix_*` 字段？
6. RWKV6Qwen2 的 `time_mix_lerp_*` 与 RWKV6 有什么本质差异？

### 14.14.8 练习 8：理解图输入家族的设计
阅读 `src/llama-graph.h` 中 `llm_graph_input_*` 类族，回答问题：
1. 基类 `llm_graph_input_i` 的 `set_input` 与 `can_reuse` 抽象方法分别在什么阶段被调用？
2. `llm_graph_input_attn_no_cache` / `_kv` / `_k` / `_kv_iswa` / `_cross` 五种 attn 输入类分别对应哪类模型？
3. `llm_graph_input_mem_hybrid` / `_k` / `_iswa` 的成员 `inp_attn` 与 `inp_rs` 的作用是什么？
4. `llm_graph_input_rs` 如何支持 RNN 状态读写？`s_copy_main` 与 `s_copy_extra` 的用途差异是什么？
5. `llm_graph_input_attn_temp` 是为哪个架构的温度缩放机制服务的？

### 14.14.9 练习 9：追踪 llm_graph_result 的复用机制
阅读 `src/llama-graph.h` 中 `llm_graph_result` 与 `llm_graph_params::allow_reuse` 的实现，回答问题：
1. `llm_graph_result::can_reuse` 与 `llm_graph_params::allow_reuse` 的区别是什么？两者的判定时机有何不同？
2. `t_inp_tokens` / `t_inp_embd` / `t_logits` / `t_embd` / `t_embd_pooled` 分别保存哪类计算产物？
3. `t_sampled_logits` / `t_candidates` / `t_sampled` / `t_sampled_probs` 这四个 map 在何时被填充？
4. `allow_reuse` 中如何通过 `seq_id_unq` 数组保证 equal_seqs 模式下复用安全？

### 14.14.10 练习 10：理解公共子图构造函数
阅读 `src/llama-graph.h` 中 `llm_graph_context` 的公共方法，回答问题：
1. `build_norm` 如何通过 `llm_norm_type` 切换 `LLM_NORM`/`LLM_NORM_RMS`/`LLM_NORM_GROUP`？
2. `build_ffn` 接受 `type_op`（SILU/GELU/RELU/RELU_SQR/SWIGLU/GEGLU/REGLU）和 `type_gate`（SEQ/PAR）参数的意义是什么？
3. 两个 `build_moe_ffn` 重载分别用于什么场景？带 bias 的版本与不带 bias 的版本如何在调用点选择？
4. `build_attn_mha` 的 `sinks` 与 `v_mla` 参数分别服务于哪些模型？
5. `build_rs` 通过 `state_copy_main` 与 `state_copy_extra` 如何避免不必要的数据拷贝？

---

## 14.15 本章小结

这一章中我们学习/了解/讨论了 llama_model 与计算图构建系统的核心机制。首先，我们剖析了 `llama_model` 作为模型状态中央枢纽的类骨架与 Pimpl 生命周期管理；其次，我们详细追踪了四阶段加载流水线（`load_arch → load_hparams → load_vocab → load_tensors`）如何将 GGUF 文件转化为内存中的结构化模型对象；接着，我们深入理解了张量按设备分配策略（`make_cpu_buft_list`/`make_gpu_buft_list`/`select_buft`）如何根据用户配置自动规划每层张量的物理放置；然后，我们全景式浏览了 `llama_layer` 及其子结构体如何以"零件仓库"形式容纳 80+ 架构的张量差异；随后，我们解读了 `build_graph` 的架构路由分发与统一尾部装配线，以及 `create_memory` 的五大缓存架构分支；接着，我们掌握了 RoPE 双源频率切换与 LongRoPe 长短因子自动选择的调音台机制；之后，我们熟悉了模型信息查询接口与 `print_info` 体检报告的输出格式；最后，我们系统梳理了 `llm_graph_input_i` 家族、`llm_graph_result` 复用机制、`llm_graph_context` 公共子图工厂三大核心组件，理解计算图从输入抽象、产物承载到模块化构造的完整流水线。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| `llama_model` | 模型状态中央枢纽，持有 hparams/vocab/layers/pimpl 等所有推理期状态 |
| `Pimpl 模式` | 将 mappings/mlock/ctxs_bufs/dev_layer 等重资源隐藏在 impl 中，头文件保持轻量 |
| `load_arch` | 读取模型架构枚举，未知架构直接抛异常 |
| `load_hparams` | 把 GGUF KV 先全量字符串化到 gguf_kv，再用结构化读取派生 n_layer/n_head/n_embd 等数值字段 |
| `load_tensors` | 按架构 switch（80+）逐层填充 llama_layer 张量，最后统一加载 .scale 与 .input_scale 通用 pass |
| `make_cpu_buft_list` | 按 ACCEL→Host→Extra→CPU 优先级装配 CPU 缓冲类型候选 |
| `make_gpu_buft_list` | 装配 GPU 缓冲类型，支持 split 模式并追加 CPU fallback |
| `dev_layer` | 通过 upper_bound 在累积 split 数组中查表，把每一层分配到合适的设备 |
| `select_buft` | 通过临时 ggml_add 算子测试设备能力，返回第一个支持当前 op 的 buft |
| `buft_supported` | 构造临时算子与缓冲，调用 ggml_backend_dev_supports_op 判断设备能力 |
| `llama_layer` | 容纳 attn/ffn/moe/ssm/rwkv/rope/altup 等几十上百个张量指针的容器 |
| `llama_layer_posnet` | WavTokenizer PosNet 的 4 类 block 子结构（resnet×4 + attn + norm） |
| `llama_layer_convnext` | ConvNeXt 的 dw/norm/pw1/pw2/gamma 子结构 |
| `llama_layer_shortconv` | LFM2 短卷积的 in_proj/conv/out_proj 三件套 |
| `build_graph` | 依据 arch 路由到对应 llm_build_* 类实例化，追加 pooling/sampling/dense_out，返回 ggml_cgraph |
| `create_memory` | 依据 arch 分发到 nullptr/recurrent/hybrid/iswa/kv_cache 五种内存实现 |
| `get_rope_freq_base/scale` | SWA 层使用训练期频率，否则使用 cparams 当前频率 |
| `get_rope_factors` | LongRoPe 短长因子按 n_ctx_seq 自动切换 |
| `print_info` | 输出架构/层数/头数/专家配置/SSM 参数等完整诊断报告 |
| `llm_graph_input_*` | 图输入抽象类家族，按注意力类型分为 no_cache/kv/k/kv_iswa/cross 等 |
| `llm_graph_result` | 图构建产物，承载 logits/embd/embd_pooled 与 can_reuse 复用机制 |
| `llm_graph_params::allow_reuse` | 通过比较 ubatch/samplers/cparams/arch 判断图拓扑是否可复用 |
| `llm_graph_input_i::can_reuse` | 子类可重写以检查 hparams/cparams/ubatch 等参数是否兼容复用 |
| `llm_graph_input_mem_hybrid_*` | 混合内存输入，组合 attn 与 rs 两种输入以支持 Transformer+RNN 混合架构 |
| `llm_graph_context::build_moe_ffn` | MoE FFN 的统一子图构造函数，支持带/不带 bias 与归一化变体 |
| `llm_graph_context::build_attn_mha` | 多头注意力的统一子图构造函数，支持 ALiBi、sinks、MLA 等扩展 |
| `llm_graph_context::build_rs` | 循环状态（R/S）的读写子图，支持视图与副本操作以避免不必要拷贝 |
| `llm_graph_type` | 图类型枚举，区分 DEFAULT/ENCODER/DECODER 以支持 T5 等 encoder-decoder 架构 |
| `llama_model_rope_type` | 依据 arch 返回 NORM/NEOX/MROPE/IMROPE/NONE 等 RoPE 类型 |
| `llama_model_meta_*` | 对外元数据查询接口（val_str/count/key/val_by_index）便于调试与日志输出 |
| `llama_internal_get_tensor_map` | 返回内部张量名映射，供测试与调试使用 |

下一章中，我们将学习词汇表与分词系统 —— 文本与 Token 的"双向桥梁"，理解 llama.cpp 如何加载模型词汇表、解析不同分词器（SPM、BPE、WPM、UGM、RWKV、PLaMo2），并执行文本分词和反分词。

