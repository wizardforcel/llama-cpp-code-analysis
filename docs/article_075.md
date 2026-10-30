



# 第75章 MTMD 多模态处理核心 —— 国际货运中心的"总指挥调度室"

## 学习目标

- 掌握 MTMD 模块的公共 C API 与 C++ RAII 封装设计
- 理解 `mtmd_context` 如何统一管理视觉（CLIP）与音频编码器的初始化与生命周期
- 深入多模态输入的 Tokenization 机制：文本切割、媒体标记识别与 Chunk 生成
- 掌握 MTMD Helper 层如何实现 M-RoPE 多维位置编码与跨模态批处理解码
- 理解图像预处理（缩放、切片、归一化）与音频预处理（梅尔频谱、分块）的流水线
- 学会使用 `mtmd_bitmap` 等抽象处理图像与音频原始数据，并理解其内存管理策略
- 掌握智能指针包装类（deleter 模式）与 C API 资源管理的最佳实践

## 75.1 生活类比

想象 MTMD 是一个**国际货运中心**（多模态处理中心）：
- **`mtmd_context`** = 货运中心的**总指挥调度室**，同时管理「图像集装箱码头」（视觉编码器）和「音频货运站」（音频编码器），确保不同模态的货物（数据）能统一换算成标准集装箱（嵌入向量）；
- **`mtmd::bitmap / RAII`** = **智能集装箱管理系统**，通过自动门禁（deleter）确保集装箱离开园区时自动归还，避免遗失（内存泄漏）；
- **`mtmd_tokenizer`** = **智能分拣流水线**，识别文本中的 `<__media__>` 标记（集装箱编号），将文本包裹与图像/音频集装箱交替排列，生成最终装船清单；
- **图像预处理家族** = **多种规格整形车间**，从固定尺寸（fixed_size）到动态规划（dyn_size），再到高精度切片（LLaVA-UHD/LFM2），满足不同货船的装载要求；
- **M-RoPE 位置编码** = **集装箱的立体坐标系统**（x 轴、y 轴、时间轴），确保二维图像空间或一维音频时间信息在运输过程中不丢失；
- **MTMD Helper** = **起重机操作员**，将编码后的集装箱（嵌入向量）按批次精准吊装到指定位置（KV Cache），并处理特殊集装箱（非因果注意力模式）；
- **图像/音频解码器（Helper 层）** = **海关检验与分类中心**，自动检测货物类型（通过 magic bytes 识别 JPEG/PNG/WAV/MP3），并调用对应的拆箱工具。

---

## 75.2 源码地图

```text
tools/mtmd/
├── mtmd.h
│   ├── mtmd_bitmap / mtmd_input_chunk / mtmd_input_chunks  # 原始数据与输入块核心结构（C API）
│   ├── mtmd_context_params / mtmd_default_marker()         # 上下文参数与默认标记
│   ├── mtmd::bitmap / mtmd::bitmaps / mtmd::input_chunks   # C++ RAII 包装
│   │   ├── bitmap.*  # 构造函数、析构、访问器
│   │   ├── bitmaps.c_ptr()  # 转换为 C API 所需的原始指针数组
│   │   └── input_chunks.*  # 构造、析构、size、operator[]
│   ├── deleter classes  # mtmd_context_deleter 等
│   ├── mtmd_init_from_file() / mtmd_free()                 # 上下文生命周期
│   ├── mtmd_tokenize() / mtmd_encode_chunk() / mtmd_encode() # 编码入口
│   ├── mtmd_decode_use_mrope() / mtmd_decode_use_non_causal() # 解码模式查询
│   └── mtmd_bitmap_* / mtmd_input_chunk_* / mtmd_image_tokens_get_* # 数据访问接口
├── mtmd.cpp
│   ├── mtmd_context::mtmd_context() / init_vision() / init_audio() # 视觉/音频预处理器分派与初始化
│   ├── mtmd_context::get_clip_ctx() / lookup_token() / token_to_piece() / proj_type_*() # 上下文工具
│   ├── mtmd_tokenizer::mtmd_tokenizer() / tokenize() / add_text() / add_media() / add_image_media() / add_audio_media() / split_batch_to_chunk() # Tokenization 主逻辑
│   ├── mtmd_encode() / mtmd_get_output_embd() / mtmd_decode_use_*() / mtmd_tokenize() # 编码与解码模式查询
│   ├── mtmd_bitmap_init() / mtmd_input_chunk_copy() / mtmd_input_chunk_get_*() # 数据操作核心实现
│   ├── mtmd_debug_encode_*() / mtmd_debug_preprocess_*() / mtmd_test_create_input_chunks() # 调试与测试接口
│   └── mtmd_image_tokens::clone() / mtmd_audio_tokens::clone() # 内部 Token 结构深拷贝
├── mtmd-helper.h / mtmd-helper.cpp
│   ├── mtmd_helper_log_set() / mtmd_helper_get_n_tokens() / mtmd_helper_get_n_pos() # 日志与统计入口
│   ├── mtmd_helper_bitmap_init_from_buf() / mtmd_helper_bitmap_init_from_file() # 资源加载与类型检测
│   ├── mtmd_helper_logger::log() / log_v() # 内部日志封装
│   ├── decode_embd_batch::decode_embd_batch() / set_position_normal() / set_position_mrope_*() / get_view() # M-RoPE 核心
│   └── mtmd_helper_eval_chunk_single() / mtmd_helper_eval_chunks() / mtmd_helper_decode_image_chunk() # 批量解码与评估流水线
├── mtmd-image.h / mtmd-image.cpp
│   ├── mtmd_image_preprocessor::mtmd_image_preprocessor() / ~mtmd_image_preprocessor() / preprocess() / img_u8_to_f32() # 基类接口
│   ├── img_tool::resize() / crop() / composite() / fill() / clip() / lerp() # 基础图像操作
│   ├── img_tool::resize_bilinear / resize_bicubic / resize_bicubic_pillow / calc_size_preserved_ratio(overload 1/2) # 插值算法与尺寸计算
│   ├── mtmd_image_preprocessor_llava_uhd::* / mtmd_image_preprocessor_lfm2::* / mtmd_image_preprocessor_internvl::* / mtmd_image_preprocessor_idefics3::* # 模型定制预处理
│   └── mtmd_image_preprocessor_fixed_size / dyn_size / longest_edge / deepseekocr / youtuvl # 各模型专用预处理
└── mtmd-audio.h / mtmd-audio.cpp
    ├── mtmd_audio_cache::fill_sin_cos_table() / fill_hann_window() / fill_mel_filterbank_matrix() # 音频缓存构建
    ├── mtmd_audio_preprocessor::initialize() / preprocess()  # 抽象基类接口
    ├── mtmd_audio_preprocessor_whisper::initialize() / preprocess() # Whisper 梅尔频谱提取
    ├── mtmd_audio_preprocessor_conformer::initialize() / preprocess() # Conformer 梅尔频谱提取
    └── mtmd_audio_streaming_istft::* # 流式 ISTFT 波形重建
```

---

## 75.3 MTMD 公共 C API 与 RAII 封装 —— 多模态世界的"智能集装箱系统"

`mtmd.h` 是整个多模态模块的“对外大门”，它定义了三个核心数据结构：`mtmd_bitmap`（原始数据容器）、`mtmd_input_chunk`（单个输入块）、`mtmd_input_chunks`（输入块集合）。为了让 C++ 用户免受手动内存管理之苦，头文件还提供了基于 `std::unique_ptr` + 自定义 `deleter` 的 RAII 包装类。

### 75.3.1 核心数据结构三剑客

我们先来看 C API 层面的三个基础结构体，它们是多模态数据流转的“身份证”。

**源码路径：`tools/mtmd/mtmd.h` - `mtmd_bitmap` / `mtmd_input_chunk` / `mtmd_input_chunks`（第 1-229 行）**

```c
// ① 原始图像/音频数据的"集装箱"
typedef struct mtmd_bitmap {
    uint32_t nx, ny;                 // 宽/高（音频时 nx=采样点数, ny=1）
    std::vector<unsigned char> data; // RGB 字节流 或 float PCM 流
    bool is_audio = false;           // 区分图像/音频
    std::string id;                  // 可选标识，用于追踪
} mtmd_bitmap;

// ② 单个输入块：文本 token 序列、图像 tokens 或音频 tokens 的"三通阀门"
typedef struct mtmd_input_chunk {
    enum mtmd_input_chunk_type type; // TEXT / IMAGE / AUDIO
    std::vector<llama_token> tokens_text;     // 文本模式下的 token
    std::unique_ptr<mtmd_image_tokens> tokens_image; // 图像模式
    std::unique_ptr<mtmd_audio_tokens> tokens_audio; // 音频模式
} mtmd_input_chunk;

// ③ 整个输入的"货运清单"：有序的 chunk 集合
typedef struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
} mtmd_input_chunks;
```

> **代码解析**：这段代码定义了多模态数据流转的最小单元。**`mtmd_bitmap`** 负责承载原始字节（图像为 RGB，音频为 float PCM），**`mtmd_input_chunk`** 通过 `type` 字段区分模态，内部用互斥的智能指针持有对应模态的 token 数据，**`mtmd_input_chunks`** 则是一个有序容器，记录了完整对话轮次中文本与媒体交替出现的顺序。这种“类型标签 + 互斥字段”设计避免了联合体的复杂性，同时保证内存紧凑。

### 75.3.2 C++ RAII 智能指针：内存安全的"自动门"

C API 要求调用者手动 `free`，极易泄漏。`mtmd.h` 为此提供了四个 `deleter` 仿函数与三个 RAII 包装类。

**源码路径：`tools/mtmd/mtmd.h` - `mtmd_context_deleter` 等（第 232-249 行）**

```cpp
// 四大 deleter：统一入口，析构时自动调用对应的 C API free 函数
struct mtmd_context_deleter {
    void operator()(mtmd_context *val) { mtmd_free(val); }
};
struct mtmd_bitmap_deleter {
    void operator()(mtmd_bitmap *val) { mtmd_bitmap_free(val); }
};
struct mtmd_input_chunks_deleter {
    void operator()(mtmd_input_chunks *val) { mtmd_input_chunks_free(val); }
};
struct mtmd_input_chunk_deleter {
    void operator()(mtmd_input_chunk *val) { mtmd_input_chunk_free(val); }
};
```

**源码路径：`tools/mtmd/mtmd.h` - `mtmd::bitmap` 类（第 254-266 行）**

```cpp
// mtmd::bitmap：对 mtmd_bitmap* 的零开销封装
class bitmap {
    std::unique_ptr<mtmd_bitmap, mtmd_bitmap_deleter> ptr; // 核心：unique_ptr + 自定义 deleter
public:
    bitmap() : ptr(nullptr) {}                                      // ① 默认构造：空指针
    bitmap(mtmd_bitmap *bmp) : ptr(bmp) {}                          // ② 从裸指针接管所有权
    bitmap(bitmap &&other) noexcept : ptr(std::move(other.ptr)) {}  // ③ 移动构造：所有权转移
    bitmap(uint32_t nx, uint32_t ny, const unsigned char *data) {   // ④ 数据构造：直接创建
        ptr.reset(mtmd_bitmap_init(nx, ny, data));
    }
    ~bitmap() = default; // 由 unique_ptr 自动调用 deleter

    // 访问器：透传到底层 C API
    uint32_t nx() const { return mtmd_bitmap_get_nx(ptr.get()); }
    uint32_t ny() const { return mtmd_bitmap_get_ny(ptr.get()); }
    const unsigned char *data() const { return mtmd_bitmap_get_data(ptr.get()); }
    size_t n_bytes() const { return mtmd_bitmap_get_n_bytes(ptr.get()); }
    std::string id() const { return mtmd_bitmap_get_id(ptr.get()); }
    void set_id(const char *id) const { mtmd_bitmap_set_id(ptr.get(), id); }
};
```

**源码路径：`tools/mtmd/mtmd.h` - `mtmd::bitmaps` 与 `mtmd::input_chunks`（第 271-288 行）**

```cpp
// 批量 bitmap 容器：提供 c_ptr() 转换为 C API 所需的 const mtmd_bitmap** 数组
class bitmaps {
    std::vector<bitmap> entries;
public:
    ~bitmaps() = default;
    std::vector<const mtmd_bitmap *> c_ptr() {
        std::vector<const mtmd_bitmap *> res(entries.size());
        for (size_t i = 0; i < entries.size(); i++) res[i] = entries[i].ptr.get();
        return res;
    }
};

// 输入块集合包装：size() 与 operator[] 便捷访问
class input_chunks {
    std::unique_ptr<mtmd_input_chunks, mtmd_input_chunks_deleter> ptr;
public:
    input_chunks() = default;
    input_chunks(mtmd_input_chunks *chunks) : ptr(chunks) {}
    ~input_chunks() = default;
    size_t size() const { return mtmd_input_chunks_size(ptr.get()); }
    const mtmd_input_chunk *operator[](size_t idx) const {
        return mtmd_input_chunks_get(ptr.get(), idx);
    }
};
```

> **代码解析**：这段代码实现了**类型即图纸**的思想：通过 `unique_ptr` + `deleter` 模式，将 C 风格的“手动管理”升级为 C++ 风格的“栈上自动释放”。`bitmap` 类的四个构造函数覆盖了“空对象、接管裸指针、移动语义、从数据创建”四大场景；`bitmaps::c_ptr()` 解决了 C++ 容器到 C 数组指针的桥接问题；`input_chunks` 则提供了类 STL 的访问接口。所有析构函数均为 `= default`，依赖 `unique_ptr` 的析构器链式调用 `deleter`，实现零开销抽象。

### 75.3.3 核心 C API 与全局函数

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_init_from_file` / `mtmd_free`（第 463-476 行）**

```cpp
// 从 .mmproj 文件创建多模态上下文（内部 new mtmd_context，失败返回 nullptr）
mtmd_context *mtmd_init_from_file(const char *mmproj_fname,
                                  const struct llama_model *text_model,
                                  const struct mtmd_context_params ctx_params) {
    try {
        return new mtmd_context(mmproj_fname, text_model, ctx_params);
    } catch (const std::exception &e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return nullptr;
    }
}

// 销毁上下文（配合 deleter 使用）
void mtmd_free(mtmd_context *ctx) { delete ctx; }
```

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_tokenize`（第 979-984 行）**

```cpp
// 公共 Tokenization 入口：构建 tokenizer 并执行切分
int32_t mtmd_tokenize(mtmd_context *ctx,
                      mtmd_input_chunks *output,
                      const mtmd_input_text *text,
                      const mtmd_bitmap **bitmaps,
                      size_t n_bitmaps) {
    mtmd_tokenizer tokenizer(ctx, text, bitmaps, n_bitmaps);
    return tokenizer.tokenize(output);
}
```

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_bitmap_init` 等数据操作（第 1075-1125 行）**

```cpp
// 从 RGB 数据创建 bitmap（内部 new + memcpy）
mtmd_bitmap *mtmd_bitmap_init(uint32_t nx, uint32_t ny, const unsigned char *data) {
    mtmd_bitmap *bitmap = new mtmd_bitmap;
    bitmap->nx = nx; bitmap->ny = ny;
    size_t data_size = (size_t)nx * ny * 3;
    bitmap->data.resize(data_size);
    std::memcpy(bitmap->data.data(), data, data_size);
    return bitmap;
}

// 音频专用创建：is_audio=true，ny=1，data 为 float[]
mtmd_bitmap *mtmd_bitmap_init_from_audio(size_t n_samples, const float *data) {
    mtmd_bitmap *bitmap = new mtmd_bitmap;
    bitmap->nx = n_samples; bitmap->ny = 1; bitmap->is_audio = true;
    bitmap->data.resize(n_samples * sizeof(float));
    std::memcpy(bitmap->data.data(), data, n_samples * sizeof(float));
    return bitmap;
}

// 释放 bitmap（配合 deleter）
void mtmd_bitmap_free(mtmd_bitmap *bitmap) { if (bitmap) delete bitmap; }
```

> **代码解析**：这段代码构建了**资源全生命周期管理**闭环：`mtmd_init_from_file` 通过异常安全创建上下文，`mtmd_tokenize` 封装了分词器的构造与调用，`mtmd_bitmap_init*` 系列负责原始数据到内部容器的拷贝与格式标记。所有 `*_free` 函数均做空指针检查，配合 RAII 实现“零泄漏”。

### 75.3.4 Chunk 访问与深拷贝接口

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_input_chunk_get_n_tokens` / `mtmd_input_chunk_get_n_pos`（第 1159-1180 行）**

```cpp
// 获取 chunk 的 token 总数（按类型分派）
size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk *chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_tokens(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else { GGML_ABORT("invalid chunk type"); }
}

// 获取 chunk 的位置总数（M-RoPE 下 n_pos != n_tokens）
llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk *chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_pos(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else { GGML_ABORT("invalid chunk type"); }
}
```

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_input_chunk_copy` 深拷贝实现（第 1189-1206 行）**

```cpp
mtmd_input_chunk *mtmd_input_chunk_copy(const mtmd_input_chunk *chunk) {
    mtmd_input_chunk *copy = new mtmd_input_chunk{
        chunk->type,
        chunk->tokens_text,   // vector 自动深拷贝
        nullptr,
        nullptr,
    };
    if (chunk->tokens_image) {
        // 图像 tokens 必须调用 clone() 深拷贝内部 batch_f32
        copy->tokens_image = mtmd_image_tokens_ptr(new mtmd_image_tokens());
        *copy->tokens_image = chunk->tokens_image->clone();
    }
    if (chunk->tokens_audio) {
        copy->tokens_audio = mtmd_audio_tokens_ptr(new mtmd_audio_tokens());
        *copy->tokens_audio = chunk->tokens_audio->clone();
    }
    return copy;
}
```

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_image_tokens::clone` / `mtmd_audio_tokens::clone`（第 50-74 行）**

```cpp
// 图像 tokens 深拷贝：复制 nx/ny/mrope 标志 + batch_f32（含智能指针数组）
mtmd_image_tokens clone() {
    return mtmd_image_tokens{
        nx, ny, use_mrope_pos, batch_f32.clone(), id
    };
}

// 音频 tokens 深拷贝：复制 n_tokens + batch_f32 + id
mtmd_audio_tokens clone() {
    return mtmd_audio_tokens{
        n_tokens, batch_f32.clone(), id
    };
}
```

> **代码解析**：这段代码揭示了**多模态位置编码的核心差异**：文本/音频 `n_pos == n_tokens`，而图像在 M-RoPE 模式下 `n_pos = max(nx, ny)`（见 `mtmd_image_tokens_get_n_pos` 第 1224-1231 行）。深拷贝通过 `clone()` 显式实现，避免了智能指针数组的浅拷贝陷阱，保证了 `mtmd_input_chunk_copy` 语义的正确性。

---

## 75.4 MTMD 核心上下文 —— 多模态推理的"交通指挥中心"

`mtmd_context` 是多模态模块的“大脑”，它在构造时同时加载视觉与音频 CLIP 模型，校验嵌入维度兼容性，并根据 `projector_type` 动态分派预处理器。

### 75.4.1 构造函数：双引擎装配与维度校验

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_context::mtmd_context()`（第 147-222 行）**

```cpp
mtmd_context::mtmd_context(const char *mmproj_fname,
                           const llama_model *text_model,
                           const mtmd_context_params &ctx_params) :
    text_model(text_model),
    print_timings(ctx_params.print_timings),
    n_threads(ctx_params.n_threads),
    media_marker(ctx_params.media_marker),
    n_embd_text(llama_model_n_embd_inp(text_model))
{
    // 1. 兼容性检查：不再支持旧版 image_marker
    if (std::string(ctx_params.image_marker) != MTMD_DEFAULT_IMAGE_MARKER) {
        throw std::runtime_error("custom image_marker is not supported anymore, use media_marker instead");
    }
    if (media_marker.empty()) throw std::runtime_error("media_marker must not be empty");

    // 2. 构建 CLIP 初始化参数（透传 GPU/FlashAttn/预热等配置）
    clip_context_params ctx_clip_params { /* ... */ };
    auto res = clip_init(mmproj_fname, ctx_clip_params); // 一次性加载视觉/音频 CLIP
    ctx_v = res.ctx_v; ctx_a = res.ctx_a;
    if (!ctx_v && !ctx_a) throw std::runtime_error("no vision/audio encoder found");

    // 3. 双模态嵌入维度一致性校验（关键！）
    if (ctx_v && ctx_a) {
        int n_embd_v = clip_n_mmproj_embd(ctx_v);
        int n_embd_a = clip_n_mmproj_embd(ctx_a);
        if (n_embd_v != n_embd_a) throw std::runtime_error("vision/audio embed dim mismatch");
    }
    int n_embd_clip = clip_n_mmproj_embd(ctx_v ? ctx_v : ctx_a);
    if (n_embd_text != n_embd_clip)
        throw std::runtime_error("text model embed dim != mmproj embed dim");

    // 4. 分派初始化
    if (ctx_v) init_vision();
    if (ctx_a) init_audio();
}
```

> **代码解析**：这段代码完成了**“双引擎装配”**：一次性调用 `clip_init` 加载视觉/音频编码器，随后进行**三层维度校验**——视觉≠音频、文本≠视觉/音频，任何不匹配直接抛异常。这是多模态模型能否正常工作的前提：LLM 的输入嵌入维度必须与投影器输出维度完全一致。

### 75.4.2 init_vision()：视觉预处理器的"选美大赛"

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_context::init_vision()`（第 224-354 行）**

```cpp
void mtmd_context::init_vision() {
    GGML_ASSERT(ctx_v != nullptr);
    image_preproc.reset(); // 先释放旧实例

    projector_type proj = clip_get_projector_type(ctx_v);

    switch (proj) {
        // MLP/LDP/GLM-Edge/Janus-Pro 等：固定尺寸或 LLaVA-UHD（取决于是否有 pinpoints）
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_COGVLM:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_GLM_EDGE: {
            bool has_pinpoints = !clip_get_hparams(ctx_v)->image_res_candidates.empty();
            if (has_pinpoints)
                image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
            else
                image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
        } break;

        // MiniCPM-V 系列：版本化模板 + LLaVA-UHD 切片
        case PROJECTOR_TYPE_MINICPMV: {
            int minicpmv_version = clip_is_minicpmv(ctx_v);
            if (minicpmv_version == 2) { /* 2.5 模板 */ }
            else if (minicpmv_version >= 3) { /* 2.6 模板 */ }
            image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
        } break;

        // Qwen2-VL/Qwen3-VL：动态尺寸 + 特殊标记
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL: {
            img_beg = "<|vision_start|>"; img_end = "<|vision_end|>";
            image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
        } break;

        // Gemma3/3NV：固定尺寸 + <start/end_of_image>
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA3NV: {
            img_beg = "<start_of_image>"; img_end = "<end_of_image>";
            image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
        } break;

        // ... 其他 20+ 分支省略 ...
        // LFM2: MTMD_SLICE_TMPL_LFM2 + 自定义行尾模板
        // InternVL: <img>/</img> + internvl 预处理器
        // DeepSeekOCR: 仅 img_end="\n" + deepseekocr 预处理器
        // YouTuVL: 动态尺寸 + </vision_start/end/>
        default:
            throw std::runtime_error("unexpected vision projector type");
    }
    GGML_ASSERT(image_preproc != nullptr);
}
```

> **代码解析**：这段代码展示了**策略模式**的教科书级应用：`projector_type` 枚举驱动 `switch` 分派，每个分支实例化对应的预处理器子类，并配置该模型特有的 `img_beg/img_end/slice_tmpl` 等特殊标记模板。注意 MiniCPM-V 根据版本号选择不同切片模板（2.5 vs 2.6），Qwen 系列统一使用动态尺寸预处理器但标记不同。

### 75.4.3 init_audio()：音频预处理器的"调音师"

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_context::init_audio()`（第 356-400 行）**

```cpp
void mtmd_context::init_audio() {
    GGML_ASSERT(ctx_a != nullptr);
    audio_preproc.reset();
    projector_type proj = clip_get_projector_type(ctx_a);

    LOG_WRN("audio input is in experimental stage..."); // 实验性警告

    switch (proj) {
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_QWEN25O:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_GLMA: {
            // Whisper 族：统一使用 Whisper 预处理器
            aud_beg = (proj == PROJECTOR_TYPE_VOXTRAL) ? "[BEGIN_AUDIO]" : "<|audio_bos|>";
            aud_end = (proj == PROJECTOR_TYPE_VOXTRAL) ? "" : "<|audio_eos|>";
            audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
        } break;

        case PROJECTOR_TYPE_LFM2A: {
            // Conformer 族：LFM2 专用
            audio_preproc = std::make_unique<mtmd_audio_preprocessor_conformer>(ctx_a);
        } break;

        default:
            throw std::runtime_error("unexpected audio projector type");
    }
    GGML_ASSERT(audio_preproc != nullptr);
    audio_preproc->initialize(); // 显式初始化缓存（sin/cos表、窗函数、Mel滤波器组）
}
```

> **代码解析**：音频侧相对简单：Whisper 族（Qwen2-Audio、Voxtral、Ultravox 等）共用 `mtmd_audio_preprocessor_whisper`，仅标记 token 不同；LFM2-Audio 独享 `mtmd_audio_preprocessor_conformer`。调用 `initialize()` 预计算 FFT 所需的三大缓存表，为后续流式处理铺路。

---

## 75.5 Tokenization 与 Chunk 切分 —— 文本与媒体的"分词手术刀"

`mtmd_tokenizer` 是多模态输入的“分词手术刀”，它按 `media_marker`（默认 `<__media__>`）切割文本，交替生成文本 chunk 与媒体 chunk。

### 75.5.1 构造与主流程 tokenize()

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_tokenizer::mtmd_tokenizer()` / `tokenize()`（第 508-577 行）**

```cpp
// 构造：保存上下文、文本、位图数组、词汇表；兼容旧版 image_marker
mtmd_tokenizer::mtmd_tokenizer(mtmd_context *ctx,
                               const mtmd_input_text *text,
                               const mtmd_bitmap **bitmaps,
                               size_t n_bitmaps) : ctx(ctx), bitmaps(bitmaps, bitmaps + n_bitmaps) {
    add_special   = text->add_special;
    parse_special = text->parse_special;
    input_text    = text->text;
    vocab         = llama_model_get_vocab(ctx->text_model);
    // 兼容层：自动将旧标记替换为统一 media_marker
    string_replace_all(input_text, MTMD_DEFAULT_IMAGE_MARKER, ctx->media_marker);
}

// 主流程：按 media_marker 切分，交替处理文本与媒体
int32_t mtmd_tokenizer::tokenize(mtmd_input_chunks *output) {
    cur.entries.clear();
    std::vector<std::string> parts = split_text(input_text, ctx->media_marker);
    size_t i_bm = 0;
    for (auto &part : parts) {
        if (part == ctx->media_marker) {
            // 媒体标记：消费下一个 bitmap
            if (i_bm >= bitmaps.size()) return 1; // 数量不匹配报错
            const mtmd_bitmap *bitmap = bitmaps[i_bm++];
            if (int32_t res = add_media(bitmap); res != 0) return res;
        } else {
            // 文本片段：直接加入
            add_text(part, parse_special);
        }
    }
    // BOS/EOS 处理（插入到首个/末尾文本 chunk）
    if (add_special && llama_vocab_get_add_bos(vocab)) { /* ... */ }
    if (add_special && llama_vocab_get_add_eos(vocab)) add_text({llama_vocab_eos(vocab)});
    if (i_bm != bitmaps.size()) return 1; // bitmap 多余报错
    *output = std::move(cur);
    return 0;
}
```

> **代码解析**：这段代码实现了**“分拣流水线”**：`split_text` 保留分隔符，遍历片段时遇到标记消费 bitmap、遇到文本调用 `add_text`。`bitmaps` 向量与标记数量必须严格对应，否则报错。BOS/EOS 自动插入到首尾文本 chunk，保证模型看到完整序列。

### 75.5.2 文本合并与媒体双通道

**源码路径：`tools/mtmd/mtmd.cpp` - `add_text` / `add_media`（第 579-794 行）**

```cpp
// 文本合并：连续文本 chunk 自动融合，避免碎片化
void mtmd_tokenizer::add_text(const std::vector<llama_token> &tokens) {
    if (tokens.empty()) return;
    if (!cur.entries.empty() && cur.entries.back().type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        cur.entries.back().tokens_text.insert(cur.entries.back().tokens_text.end(), tokens.begin(), tokens.end());
    } else {
        cur.entries.emplace_back(MTMD_INPUT_CHUNK_TYPE_TEXT, tokens, nullptr, nullptr);
    }
}

// 媒体统一入口：根据 is_audio 分发
int32_t mtmd_tokenizer::add_media(const mtmd_bitmap *bitmap) {
    if (!bitmap->is_audio) return add_image_media(bitmap);
    else return add_audio_media(bitmap);
}
```

**源码路径：`tools/mtmd/mtmd.cpp` - `add_image_media` 核心逻辑节选（第 859-871 行）**

```cpp
// 图像处理：预处理 -> 判断是否切片 -> 生成 chunk 序列
int32_t mtmd_tokenizer::add_image_media(const mtmd_bitmap *bitmap) {
    // 1. 前置标记
    if (!ctx->img_beg.empty()) add_text(ctx->img_beg, true);

    // 2. bitmap -> clip_image_u8 -> 预处理器 -> clip_image_f32_batch
    clip_image_u8_ptr img_u8(clip_image_u8_init());
    img_u8->nx = bitmap->nx; img_u8->ny = bitmap->ny;
    img_u8->buf.resize(bitmap->data.size());
    std::memcpy(img_u8->buf.data(), bitmap->data.data(), img_u8->nx * img_u8->ny * 3);

    clip_image_f32_batch batch_f32;
    if (!ctx->image_preproc->preprocess(*img_u8, batch_f32)) return 2;

    // 3. 切片模板分支（MiniCPM-V/LLaVA-UHD/LFM2/Idefics3/Llama4）
    if (has_tiling_template) {
        auto chunks = split_batch_to_chunk(std::move(batch_f32), bitmap->id);
        // 按模板插入 ov_img_start/slices_start/row_end/slices_end 等标记
        // 交替插入概览图 chunk 与切片 chunk
    } else {
        // 非切片：聚合为单个 image_tokens chunk
        mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
        image_tokens->batch_f32 = std::move(batch_f32);
        image_tokens->id = bitmap->id;
        // M-RoPE 模式下设置 nx/ny
        cur.entries.emplace_back(MTMD_INPUT_CHUNK_TYPE_IMAGE, {}, std::move(image_tokens), nullptr);
    }

    // 4. 后置标记
    if (!ctx->img_end.empty()) add_text(ctx->img_end, true);
    return 0;
}
```

**源码路径：`tools/mtmd/mtmd.cpp` - `add_audio_media` 核心逻辑（第 872-921 行）**

```cpp
// 音频处理：预处理 -> 每个 mel_spec_chunk 封装为独立 audio chunk
int32_t mtmd_tokenizer::add_audio_media(const mtmd_bitmap *bitmap) {
    if (!ctx->aud_beg.empty()) add_text(ctx->aud_beg, true);

    std::vector<mtmd_audio_mel> mel_spec_chunks;
    const float *samples = (const float *)bitmap->data.data();
    size_t n_samples = bitmap->data.size() / sizeof(float);
    if (!ctx->audio_preproc->preprocess(samples, n_samples, mel_spec_chunks)) return 2;

    for (auto &mel_spec : mel_spec_chunks) {
        clip_image_f32_ptr mel_f32(clip_image_f32_init());
        mel_f32->nx = mel_spec.n_len; mel_f32->ny = mel_spec.n_mel;
        mel_f32->buf = std::move(mel_spec.data);

        size_t n_tokens = clip_n_output_tokens(ctx->ctx_a, mel_f32.get());
        clip_image_f32_batch batch_f32;
        batch_f32.is_audio = true;
        batch_f32.entries.push_back(std::move(mel_f32));

        mtmd_audio_tokens_ptr audio_tokens(new mtmd_audio_tokens);
        audio_tokens->n_tokens = n_tokens;
        audio_tokens->batch_f32 = std::move(batch_f32);
        audio_tokens->id = bitmap->id;

        cur.entries.emplace_back(MTMD_INPUT_CHUNK_TYPE_AUDIO, {}, nullptr, std::move(audio_tokens));
    }
    if (!ctx->aud_end.empty()) add_text(ctx->aud_end, true);
    return 0;
}
```

> **关键设计点**：
> - **文本合并**：`add_text` 检测末尾 chunk 类型，同类合并，减少 chunk 数量。
> - **图像双模式**：`has_tiling_grid` 判断是否切片。切片模式下 `split_batch_to_chunk` 将批次拆为单图 chunk，并按模板插入大量特殊标记（`ov_img_start`、`slices_start`、`row_end` 等）；非切片模式聚合为单 chunk。
> - **音频分块**：Whisper/Conformer 预处理器输出多个 `mtmd_audio_mel`（每个 3000 帧），每个封装为独立 audio chunk，中间插入 `audio_cfg` 标记。

---

## 75.6 多模态资源编码与图像预处理基座 —— 数据整形的"美容工坊"

### 75.6.1 编码与查询接口

**源码路径：`tools/mtmd/mtmd.cpp` - `mtmd_encode_chunk` / `mtmd_encode` / `mtmd_decode_use_*`（第 985-1062 行）**

```cpp
// 统一编码入口：文本无操作，图像走 mtmd_encode，音频直接 clip_image_batch_encode
int32_t mtmd_encode_chunk(mtmd_context *ctx, const mtmd_input_chunk *chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) return 0;
    else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        if (!ctx->ctx_v) return 1;
        return mtmd_encode(ctx, chunk->tokens_image.get());
    } else { // AUDIO
        if (!ctx->ctx_a) return 1;
        int n_mmproj_embd = ctx->n_embd_text;
        ctx->image_embd_v.resize(chunk->tokens_audio->n_tokens * n_mmproj_embd);
        bool ok = clip_image_batch_encode(ctx->ctx_a, ctx->n_threads,
                                          &chunk->tokens_audio->batch_f32,
                                          ctx->image_embd_v.data());
        return ok ? 0 : 1;
    }
}

// 图像编码分派：LLaVA/MiniCPM/GLM/InternVL 逐张编码，其余批量编码
int32_t mtmd_encode(mtmd_context *ctx, const mtmd_image_tokens *image_tokens) {
    auto proj_type = clip_get_projector_type(ctx->ctx_v);
    int n_mmproj_embd = clip_n_mmproj_embd(ctx->ctx_v);
    ctx->image_embd_v.resize(image_tokens->n_tokens() * n_mmproj_embd);
    bool ok = false;
    if (clip_is_llava(ctx->ctx_v) || clip_is_minicpmv(ctx->ctx_v)
        || clip_is_glm(ctx->ctx_v) || proj_type == PROJECTOR_TYPE_INTERNVL) {
        // 逐张编码（历史遗留限制）
        for (size_t i = 0; i < image_tokens->batch_f32.entries.size(); i++) {
            int n_tok = clip_n_output_tokens(ctx->ctx_v, image_tokens->batch_f32.entries[i].get());
            ok = clip_image_encode(ctx->ctx_v, ctx->n_threads,
                                   image_tokens->batch_f32.entries[i].get(),
                                   ctx->image_embd_v.data() + i * n_mmproj_embd * n_tok);
        }
    } else {
        ok = clip_image_batch_encode(ctx->ctx_v, ctx->n_threads,
                                     &image_tokens->batch_f32, ctx->image_embd_v.data());
    }
    return ok ? 0 : 1;
}

// 查询接口：M-RoPE 与非因果注意力
bool mtmd_decode_use_mrope(mtmd_context *ctx) {
    switch (ctx->proj_type_v()) {
        case PROJECTOR_TYPE_QWEN2VL: case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL: case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_PADDLEOCR: return true;
        default: return false;
    }
}
bool mtmd_decode_use_non_causal(mtmd_context *ctx) {
    switch (ctx->proj_type_v()) {
        case PROJECTOR_TYPE_GEMMA3: case PROJECTOR_TYPE_GEMMA4V: return true;
        default: return false;
    }
}
```

> **代码解析**：编码层面的**策略分派**很关键：LLaVA/MiniCPM/GLM/InternVL 因历史实现限制只能逐张编码，其余模型支持批量编码。解码模式查询函数直接映射 `projector_type`，供上层决定位置编码与注意力掩码。

### 75.6.2 图像预处理基类与工具集

**源码路径：`tools/mtmd/mtmd-image.h` - `mtmd_image_preprocessor` 基类（第 17-23 行）**

```cpp
struct mtmd_image_preprocessor {
    const clip_hparams &hparams;
    mtmd_image_preprocessor(const clip_ctx *ctx) : hparams(*clip_get_hparams(ctx)) {}
    virtual ~mtmd_image_preprocessor() = default;
    virtual bool preprocess(const clip_image_u8 &img, clip_image_f32_batch &output) = 0;

    // 归一化转换：uint8 -> float32，支持 mean/std 或无参数版本
    void img_u8_to_f32(const clip_image_u8 &src, clip_image_f32 &dst,
                       const float mean[3], const float std[3]);
    void img_u8_to_f32(const clip_image_u8 &src, clip_image_f32 &dst);
};
```

**源码路径：`tools/mtmd/mtmd-image.cpp` - `img_tool` 核心工具（第 1-420 行）**

```cpp
// 智能缩放：支持保持比例+填充（letterbox）或强制拉伸
static void resize(const clip_image_u8 &src, clip_image_u8 &dst,
                   const clip_image_size &target, resize_algo algo,
                   bool add_padding, std::array<uint8_t,3> pad_color) {
    dst.nx = target.width; dst.ny = target.height;
    dst.buf.resize(3 * dst.nx * dst.ny);
    if (!add_padding) { /* 直接插值 */ }
    else {
        // 等比缩放至 target 内接矩形，居中粘贴到 pad_color 背景上
        float scale = std::min(float(target.width)/src.nx, float(target.height)/src.ny);
        int nw = std::min(int(std::ceil(src.nx*scale)), target.width);
        int nh = std::min(int(std::ceil(src.ny*scale)), target.height);
        clip_image_u8 resized; resize_algo(src, resized, nw, nh);
        fill(dst, pad_color);
        composite(dst, resized, (target.width-nw)/2, (target.height-nh)/2);
    }
}

// 三大插值算法
static void resize_bilinear(...);   // 双线性：速度快、质量中
static void resize_bicubic(...);    // 双三次：质量高、边缘锐利
static void resize_bicubic_pillow(...); // Pillow 兼容：可分离卷积、定点优化

// 尺寸计算策略（重载1：最长边对齐；重载2：min/max_pixels 智能缩放）
static clip_image_size calc_size_preserved_ratio(const clip_image_size &inp, int align, int longest_edge);
static clip_image_size calc_size_preserved_ratio(const clip_image_size &inp, int align, int min_pix, int max_pix);
```

> **基类与工具集总结**：`mtmd_image_preprocessor` 定义了“输入 uint8 图像 → 输出 float32 批次”的契约，`img_u8_to_f32` 统一处理归一化。`img_tool` 提供了**缩放、裁剪、合成、填充**四大原子操作，三种插值算法满足不同质量/速度需求，两个 `calc_size_preserved_ratio` 重载分别服务“最长边约束”与“像素预算约束”两类模型。

---

## 75.7 专用图像预处理器家族 —— 各显神通的"整形外科医生"

不同模型对图像分辨率、切片策略要求迥异，MTMD 通过继承体系实现了 9 种预处理器。

### 75.7.1 LLaVA-UHD：高分辨率的"切片大师"

**源码路径：`tools/mtmd/mtmd-image.cpp` - `mtmd_image_preprocessor_llava_uhd::preprocess` / `get_slice_instructions`（第 386-697 行）**

```cpp
// 主流程：获取切片指令 -> 执行切片 -> 每片归一化 -> 填充 batch_f32
bool mtmd_image_preprocessor_llava_uhd::preprocess(const clip_image_u8 &img,
                                                   clip_image_f32_batch &output) {
    auto inst = get_slice_instructions({img.nx, img.ny});
    auto imgs = slice_image(img, inst);
    for (auto &u8img : imgs) {
        clip_image_f32_ptr res(clip_image_f32_init());
        img_u8_to_f32(*u8img, *res, hparams.image_mean, hparams.image_std);
        output.entries.push_back(std::move(res));
    }
    output.grid_x = inst.grid_size.width;
    output.grid_y = inst.grid_size.height;
    return true;
}

// 核心策略：pinpoints 候选分辨率 OR 自动网格搜索
slice_instructions mtmd_image_preprocessor_llava_uhd::get_slice_instructions(const clip_image_size &orig) {
    slice_instructions res;
    int patch = hparams.patch_size, slice = hparams.image_size;
    bool has_slices = orig.width > slice || orig.height > slice;
    bool has_pinpoints = !hparams.image_res_candidates.empty();

    if (!has_slices) { // 无需切片：直接概览图
        res.overview_size = {slice, slice};
        return res;
    }

    if (has_pinpoints) { // 有候选集：选最佳分辨率 -> 等分网格切片
        auto refine = select_best_resolution(orig, hparams.image_res_candidates);
        res.overview_size = {slice, slice};
        res.refined_size = refine;
        for (int y=0; y<refine.height; y+=slice)
            for (int x=0; x<refine.width; x+=slice)
                res.slices.push_back({x,y,{std::min(slice,refine.width-x), std::min(slice,refine.height-y)}});
        res.grid_size = {refine.width/slice, refine.height/slice};
        return res;
    }

    // 无候选集：自动计算最佳网格（get_best_grid）-> 细化尺寸（get_refine_size）-> 切片
    auto best_overview = get_best_resize(orig, slice, patch, true);
    res.overview_size = best_overview;
    int max_slices = 9;
    float log_ratio = log(float(orig.width)/orig.height);
    float ratio = float(orig.width*orig.height)/(slice*slice);
    int multiple = std::min(int(ceil(ratio)), max_slices);
    auto best_grid = get_best_grid(max_slices, multiple, log_ratio);
    auto refine = get_refine_size(orig, best_grid, slice, patch, true);
    res.grid_size = best_grid; res.refined_size = refine;
    // 生成切片坐标...
    return res;
}
```

> **LLaVA-UHD 算法精髓**：
> 1. **概览图**：始终生成 `slice_size x slice_size` 的缩略图（global feature）。
> 2. **切片策略**：
>    - 有 `pinpoints`（预设分辨率候选集）→ `select_best_resolution` 选最接近原图比例且浪费最小的分辨率，等分切片。
>    - 无 `pinpoints` → `get_best_grid` 搜索最接近原图长宽比的网格（如 2x2, 3x1 等），`get_refine_size` 计算能被网格整除的细化尺寸，再切片。
> 3. **网格信息**：`grid_x/grid_y` 记录切片布局，供 M-RoPE 位置编码使用。

### 75.7.2 其他预处理器速览

| 预处理器 | 核心策略 | 典型模型 |
|----------|----------|----------|
| `fixed_size` | 强制缩放为 `image_size x image_size` 正方形 | Gemma3, Nemotron-V2-VL |
| `dyn_size` | `min/max_pixels` 约束下保持比例缩放，尺寸对齐 `patch_size * n_merge` | Qwen2-VL, Qwen3-VL, Pixtral, Phi4, GLM4V, InternVL, Kimi-VL, Gemma4V, DeepSeekOCR |
| `longest_edge` | 限制最长边 ≤ `image_longest_edge`，等比缩放 | LightOnOCR |
| `llava_uhd` | 概览图 + 自适应网格切片 | LLaVA, MiniCPM-V, Llama4, LFM2, Idefics3 |
| `idefics3` | 先最长边约束缩放，再按 `image_size` 网格切分 | Idefics3 |
| `internvl` | 复用 LLaVA-UHD 切片逻辑，`overview_first=false` | InternVL |
| `deepseekocr` | 面积匹配最近的原生分辨率 {1024, 1280}，无切片 | DeepSeek-OCR |
| `youtuvl` | 线性搜索缩放比例，使 patch 数 ≤ `max_num_patches` | YouTuVL |

> **表格引导**：上表总结了 9 种预处理器的核心策略与适用模型，展示了 MTMD 如何通过多态复用基础图像操作（缩放、切片、归一化）满足不同架构的输入需求。

---

## 75.8 音频预处理与特征提取 —— 声音世界的"频谱分析师"

音频侧通过 **梅尔频谱图** 将波形压缩为二维时频表示，分为 Whisper 与 Conformer 两大流派。

### 75.8.1 缓存构建：FFT 的"备忘录"

**源码路径：`tools/mtmd/mtmd-audio.cpp` - `mtmd_audio_cache` 三大填充函数（第 16-104 行）**

```cpp
// 正弦/余弦查找表：用于 Cooley-Tukey FFT
void mtmd_audio_cache::fill_sin_cos_table(uint32_t n) {
    sin_vals.resize(n); cos_vals.resize(n);
    for (uint32_t i=0; i<n; i++) {
        double theta = (2*M_PI*i)/n;
        sin_vals[i] = sinf(theta); cos_vals[i] = cosf(theta);
    }
}

// 汉宁窗：periodic=true 用于 STFT 分析，false 用于合成
void mtmd_audio_cache::fill_hann_window(uint32_t length, bool periodic) {
    hann_window.resize(length);
    int offset = periodic ? 0 : -1;
    for (uint32_t i=0; i<length; i++)
        hann_window[i] = 0.5f * (1.0f - cosf(2*M_PI*i/(length+offset)));
}

// 梅尔滤波器组矩阵：Slaney 刻度 + _area 归一化
void mtmd_audio_cache::fill_mel_filterbank_matrix(int n_mel, int n_fft, int sample_rate,
                                                  float fmin, float fmax,
                                                  bool slaney_area_norm, float scale) {
    // 1. Hz -> Mel (Slaney)
    // 2. Mel 均分 -> Hz 边界 -> FFT bin 索引
    // 3. 三角滤波器填充 matrix[mel][bin]
    // 4. Slaney 面积归一化：除以 (f_max-f_min)/2 * (mel_scale * scale)
}
```

> **代码解析**：缓存层将**昂贵的三角函数与滤波器组构建前置**，`initialize()` 阶段一次性计算，后续每帧 FFT/IFFT 直接查表，实现了“空间换时间”。

### 75.8.2 Whisper 预处理器：标准梅尔频谱流水线

**源码路径：`tools/mtmd/mtmd-audio.cpp` - `mtmd_audio_preprocessor_whisper::preprocess`（第 330-401 行）**

```cpp
bool mtmd_audio_preprocessor_whisper::preprocess(const float *samples, size_t n_samples,
                                                 std::vector<mtmd_audio_mel> &output) {
    // 1. 参数：n_fft=400, hop=160, n_mel=80/128, window=400
    int n_fft = hparams.audio_n_fft; // 400
    int hop = hparams.audio_hop_length; // 160
    int n_mel = hparams.n_mel_bins;     // 80 or 128

    // 2. 填充：反射填充 (n_fft-hop)/2 = 120 两侧
    int pad = (n_fft - hop)/2;
    std::vector<float> padded(n_samples + 2*pad);
    // reflect padding 实现...

    // 3. 滑窗 STFT：每帧 400 点，步长 160
    int n_frames = (padded.size() - n_fft)/hop + 1;
    std::vector<std::vector<float>> mel_spec(n_mel, std::vector<float>(n_frames));

    // 并行逐帧：加窗 -> FFT -> 幅度谱 -> Mel 滤波 -> log(1e-10 + x)
    #pragma omp parallel for
    for (int t=0; t<n_frames; t++) {
        // 加窗、FFT、功率谱、Mel 投影、对数压缩
    }

    // 4. 分块：每 3000 帧一个 chunk（30秒），最后一块可能不足
    for (int chunk_start=0; chunk_start<n_frames; chunk_start+=3000) {
        int chunk_len = std::min(3000, n_frames - chunk_start);
        mtmd_audio_mel mel; mel.n_len = chunk_len; mel.n_mel = n_mel;
        mel.data.resize(n_mel * chunk_len);
        // 转置存储：mel-major [n_mel][n_len] 以便后续批处理
        for (int m=0; m<n_mel; m++)
            for (int t=0; t<chunk_len; t++)
                mel.data[m*chunk_len + t] = mel_spec[m][chunk_start+t];
        output.push_back(std::move(mel));
    }
    return true;
}
```

> **Whisper 流水线关键点**：
> - **反射填充** 保证边界平滑。
> - **对数梅尔频谱**：`log(max(1e-10, mel))` 动态范围压缩。
> - **分块策略**：固定 3000 帧（约 30 秒），便于固定显存分配。
> - **存储布局**：`mel-major`（频带优先），与 CLIP 视觉 `channel-first` 习惯一致。

### 75.8.3 Conformer 预处理器：中心填充 + 预强调 + 自然对数

**源码路径：`tools/mtmd/mtmd-audio.cpp` - `mtmd_audio_preprocessor_conformer::preprocess`（第 415-453 行）**

```cpp
bool mtmd_audio_preprocessor_conformer::preprocess(const float *samples, size_t n_samples,
                                                   std::vector<mtmd_audio_mel> &output) {
    // 差异点：
    // 1. 中心填充：pad = n_fft/2 (200)，而非反射填充
    // 2. 预强调：y[t] = x[t] - 0.97 * x[t-1]
    // 3. 自然对数：log(max(1e-10, x))，而非 log10
    // 4. 无分块：直接输出完整 mel_spec 为单个 chunk
    // 5. 存储布局同 Whisper：mel-major
}
```

> **代码解析**：Conformer 更贴近语音识别学术标准：预强调增强高频，中心填充避免边界伪影，自然对数符合数学惯例。

### 75.8.4 流式 ISTFT：从频谱重建波形

**源码路径：`tools/mtmd/mtmd-audio.cpp` - `mtmd_audio_streaming_istft`（第 459-551 行）**

```cpp
// 构造：预计算 IFFT 所需缓存、窗函数、重叠缓冲区
mtmd_audio_streaming_istft::mtmd_audio_streaming_istft(int n_fft, int hop_length) :
    n_fft(n_fft), hop_length(hop_length),
    n_fft_bins(n_fft/2 + 1),
    overlap_buffer(n_fft, 0), window_sum_buffer(n_fft, 0),
    padding_to_remove((n_fft-hop_length)/2) { /* ... */ }

// 处理单帧：频谱 -> IFFT -> 重叠相加 -> 提取 hop_length 样本
std::vector<float> mtmd_audio_streaming_istft::process_frame(const float *frame_spectrum) {
    // 1. frame_spectrum (实/虚交错) -> 补全共轭对称 -> IFFT -> 实部
    // 2. 加窗 (hann) -> 叠加到 overlap_buffer
    // 3. 窗口和归一化：output = overlap[pad:pad+hop] / window_sum[pad:pad+hop]
    // 4. 环形缓冲：memmove overlap_buffer 左移 hop_length
    return output; // 长度 = hop_length
}

// 结束流：排空剩余缓冲区
std::vector<float> mtmd_audio_streaming_istft::flush() {
    // 处理 overlap_buffer 中剩余有效样本，直到窗口和接近零
    return output;
}
```

> **流式 ISTFT 核心**：**重叠相加法 (OLA)** + **窗口和归一化**。`overlap_buffer` 维护最近 `n_fft` 个时域样本的累加，`window_sum_buffer` 记录窗函数平方和，输出时逐点相除消除窗函数增益。`padding_to_remove` 跳过首帧的反射填充残留。

---

## 75.9 MTMD Helper 高层评估与资源加载 —— 解码"模型输入"的最后一公里

`mtmd-helper` 是上层应用（如 `llama-mtmd-cli`）的直接调用层，封装了“加载资源 → Tokenize → Encode → Decode”的完整流水线。

### 75.9.1 decode_embd_batch：M-RoPE 位置的"编排师"

**源码路径：`tools/mtmd/mtmd-helper.cpp` - `decode_embd_batch` 类（第 155-246 行）**

```cpp
// 构造：预分配 pos/seq_id/logits 数组，n_pos_per_embd=1(普通)或4(M-RoPE)
decode_embd_batch::decode_embd_batch(float *embd, int32_t n_tokens,
                                     int n_pos_per_embd, int n_mmproj_embd)
    : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
    pos.resize(n_tokens * n_pos_per_embd);
    n_seq_id.resize(n_tokens); seq_ids.resize(n_tokens+1);
    logits.resize(n_tokens); seq_id_0.resize(1);
    seq_ids[n_tokens] = nullptr;
    batch = {n_tokens, nullptr, embd, pos.data(), n_seq_id.data(), seq_ids.data(), logits.data()};
}

// 普通线性位置：pos = pos_0 + i
void set_position_normal(llama_pos pos_0, llama_seq_id seq_id) {
    for (int i=0; i<batch.n_tokens; i++) {
        batch.pos[i] = pos_0 + i;
        batch.n_seq_id[i]=1; batch.seq_id[i]=seq_id_0.data(); batch.logits[i]=false;
    }
}

// M-RoPE 2D（图像）：四维位置 (t=0, y, x, 0)，按行优先展开
void set_position_mrope_2d(llama_pos pos_0, int nx, int ny, llama_seq_id seq_id) {
    GGML_ASSERT(n_pos_per_embd==4 && nx>0 && ny>0 && nx*ny==batch.n_tokens);
    for (int y=0; y<ny; y++) for (int x=0; x<nx; x++) {
        int i = y*nx + x;
        pos[i]                     = pos_0;       // t=0
        pos[i + batch.n_tokens]    = pos_0 + y;   // y
        pos[i + batch.n_tokens*2]  = pos_0 + x;   // x
        pos[i + batch.n_tokens*3]  = 0;           // 保留维度
    }
    // seq_id/logits 同 normal
}

// M-RoPE 1D（音频）：四维位置 (t, t, t, 0)，时间轴三维共用
void set_position_mrope_1d(llama_pos pos_0, llama_seq_id seq_id) {
    for (int i=0; i<batch.n_tokens; i++) {
        pos[i] = pos[i+batch.n_tokens] = pos[i+batch.n_tokens*2] = pos_0 + i;
        pos[i+batch.n_tokens*3] = 0;
    }
}

// 微批次视图：M-RoPE 时需重组 pos 为连续块（列转置）
llama_batch get_view(int offset, int n_tokens) {
    if (n_pos_per_embd > 1) { // M-RoPE
        pos_view.clear(); pos_view.reserve(n_tokens * n_pos_per_embd);
        for (int i=0; i<n_pos_per_embd; i++) {
            size_t src = i * batch.n_tokens + offset;
            pos_view.insert(pos_view.end(), pos.data()+src, pos.data()+src+n_tokens);
        }
        return {n_tokens, nullptr, embd+offset*n_mmproj_embd, pos_view.data(),
                n_seq_id+offset, seq_ids+offset, logits+offset};
    } else { // 普通
        return {n_tokens, nullptr, embd+offset*n_mmproj_embd, pos.data()+offset,
                n_seq_id+offset, seq_ids+offset, logits+offset};
    }
}
```

> **M-RoPE 布局解密**：
> - 内存中 `pos` 采用**通道优先**存储：`[t0...tn, y0...yn, x0...xn, 0...0]`（长度 4*n_tokens）。
> - `get_view` 为微批次切片时，必须将这 4 个通道对应的片段**拼接成通道交织**格式：`[t_off, y_off, x_off, 0, t_off+1, y_off+1, x_off+1, 0, ...]`，否则 llama.cpp 内核会读错位置。这就是 `pos_view` 重组的由来。

### 75.9.2 资源加载：格式无关的"海关检验"

**源码路径：`tools/mtmd/mtmd-helper.cpp` - `mtmd_helper_bitmap_init_from_buf`（第 462-486 行）**

```cpp
mtmd_bitmap *mtmd_helper_bitmap_init_from_buf(mtmd_context *ctx,
                                              const unsigned char *buf, size_t len) {
    // 1. 先嗅探音频 magic bytes（WAV/MP3/FLAC/OGG/OPUS）
    if (audio_helpers::is_audio_file((const char*)buf, len)) {
        int sample_rate = mtmd_get_audio_sample_rate(ctx);
        if (sample_rate < 0) { LOG_ERR("model not support audio"); return nullptr; }
        std::vector<float> pcmf32;
        if (!audio_helpers::decode_audio_from_buf(buf, len, sample_rate, pcmf32))
            return nullptr;
        return mtmd_bitmap_init_from_audio(pcmf32.size(), pcmf32.data());
    }
    // 2. 非音频视为图像：stb_image 解码为 RGB
    int nx, ny, nc;
    auto *data = stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3); // 强制 3 通道
    if (!data) { LOG_ERR("failed to decode image"); return nullptr; }
    mtmd_bitmap *result = mtmd_bitmap_init(nx, ny, data);
    stbi_image_free(data);
    return result;
}
```

> **设计亮点**：**“先音频后图像”**的检测顺序避免了误判（图像文件极不可能以音频 magic bytes 开头）。音频走 `miniaudio` 解码重采样到模型要求采样率，图像走 `stb_image` 统一输出 RGB 三通道，屏蔽了上层对格式的感知。

### 75.9.3 评估流水线：Chunk 级分派与批量解码

**源码路径：`tools/mtmd/mtmd-helper.cpp` - `mtmd_helper_eval_chunk_single` / `mtmd_helper_eval_chunks`（第 336-430 行）**

```cpp
// 单 chunk 评估：文本直接 decode，媒体先 encode 再 decode
int32_t mtmd_helper_eval_chunk_single(mtmd_context *ctx, llama_context *lctx,
                                      const mtmd_input_chunk *chunk,
                                      llama_pos n_past, llama_seq_id seq_id,
                                      int32_t n_batch, bool logits_last,
                                      llama_pos *new_n_past) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        // 文本：分批 llama_decode，最后一个 token 可选输出 logits
        llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
        for (size_t i=0; i<n_tokens; ) { /* 填 batch -> decode -> 更新 n_past */ }
    } else { // IMAGE / AUDIO
        const char *name = (chunk->type==MTMD_INPUT_CHUNK_TYPE_IMAGE)?"image":"audio";
        // 1. 编码
        if (mtmd_encode_chunk(ctx, chunk) != 0) return -1;
        // 2. 解码嵌入
        float *embd = mtmd_get_output_embd(ctx);
        return mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd,
                                              n_past, seq_id, n_batch, new_n_past);
    }
    return 0;
}

// 批量评估：遍历所有 chunk，末尾 chunk 可选 logits_last
int32_t mtmd_helper_eval_chunks(mtmd_context *ctx, llama_context *lctx,
                                const mtmd_input_chunks *chunks,
                                llama_pos n_past, llama_seq_id seq_id,
                                int32_t n_batch, bool logits_last,
                                llama_pos *new_n_past) {
    for (size_t i=0; i<mtmd_input_chunks_size(chunks); i++) {
        bool last = (i == n_chunks-1) && logits_last;
        if (mtmd_helper_eval_chunk_single(ctx, lctx,
            mtmd_input_chunks_get(chunks, i), n_past, seq_id, n_batch, last, &n_past))
            return -1;
        *new_n_past = n_past;
    }
    return 0;
}
```

**源码路径：`tools/mtmd/mtmd-helper.cpp` - `mtmd_helper_decode_image_chunk`（第 263-331 行）**

```cpp
int32_t mtmd_helper_decode_image_chunk(mtmd_context *ctx, llama_context *lctx,
                                       const mtmd_input_chunk *chunk,
                                       float *encoded_embd, llama_pos n_past,
                                       llama_seq_id seq_id, int32_t n_batch,
                                       llama_pos *new_n_past) {
    int n_mmproj_embd = llama_model_n_embd_inp(llama_get_model(lctx));
    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;
    int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);

    decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

    // 位置编码分派
    if (mtmd_decode_use_mrope(ctx)) {
        if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            auto img_tok = mtmd_input_chunk_get_tokens_image(chunk);
            batch_embd.set_position_mrope_2d(n_past,
                mtmd_image_tokens_get_nx(img_tok),
                mtmd_image_tokens_get_ny(img_tok), seq_id);
        } else { // AUDIO
            batch_embd.set_position_mrope_1d(n_past, seq_id);
        }
    } else {
        batch_embd.set_position_normal(n_past, seq_id);
    }

    // 非因果注意力（Gemma3/4V）
    if (mtmd_decode_use_non_causal(ctx)) llama_set_causal_attn(lctx, false);

    // 微批次循环
    for (int i=0; i<n_img_batches; i++) {
        llama_batch view = batch_embd.get_view(i*n_batch, n_tokens_batch);
        if (llama_decode(lctx, view) != 0) { /* 恢复 causal -> 返回错误 */ }
    }
    n_past += mtmd_input_chunk_get_n_pos(chunk);
    *new_n_past = n_past;
    if (mtmd_decode_use_non_causal(ctx)) llama_set_causal_attn(lctx, true);
    return 0;
}
```

> **流水线总结**：
> 1. **文本 chunk** → 直接 `llama_decode`（自带位置）。
> 2. **媒体 chunk** → `mtmd_encode_chunk` 生成嵌入 → `decode_embd_batch` 构建批次 → 设置位置（Normal/M-RoPE 2D/1D） → 可选关闭因果注意力 → 微批次 `llama_decode` → 恢复注意力模式。
> 3. `n_past` 按 `n_pos` 推进（M-RoPE 下 `n_pos = max(nx,ny)`），保证后续文本位置连续。

---

## 75.10 多模态调试与测试 —— 模型行为的"显微镜"

**源码路径：`tools/mtmd/mtmd.cpp` - 调试接口簇（第 1267-1371 行）**

```cpp
// 内部调试编码：开启 clip_debug_output_embeddings，打印嵌入向量
static void mtmd_debug_encode_impl(mtmd_context *ctx, clip_ctx *ctx_clip, clip_image_f32 &img) {
    clip_set_debug_output_embeddings(ctx_clip, true);
    int n_mmproj = clip_n_mmproj_embd(ctx_clip);
    int n_tok = clip_n_output_tokens(ctx_clip, &img);
    std::vector<float> out(n_tok * n_mmproj);
    clip_image_encode(ctx_clip, ctx->n_threads, &img, out.data());
}

// 对外暴露：图像调试
void mtmd_debug_encode_image(mtmd_context *ctx, const std::vector<std::vector<float>> &image) {
    clip_image_f32 inp; inp.nx=image.size(); inp.ny=inp.nx;
    inp.buf.reserve(inp.nx*inp.ny);
    for (auto &row: image) inp.buf.insert(inp.buf.end(), row.begin(), row.end());
    mtmd_debug_encode_impl(ctx, ctx->ctx_v, inp);
}

// 对外暴露：音频调试
void mtmd_debug_encode_audio(mtmd_context *ctx, const std::vector<float> &input) {
    int n_mel = clip_get_hparams(ctx->ctx_a)->n_mel_bins;
    clip_image_f32 inp; inp.nx=input.size(); inp.ny=n_mel;
    inp.buf.resize(input.size()*n_mel);
    for (size_t i=0; i<input.size(); i++)
        for (int j=0; j<n_mel; j++) inp.buf[j*inp.nx+i] = input[i];
    mtmd_debug_encode_impl(ctx, ctx->ctx_a, inp);
}

// 预处理验证：输出切片/梅尔块元信息
void mtmd_debug_preprocess_image(mtmd_context *ctx, const std::vector<uint8_t> &rgb, int nx, int ny) {
    clip_image_u8 img{nx,ny,rgb};
    clip_image_f32_batch batch;
    ctx->image_preproc->preprocess(img, batch);
    LOG_INF("preprocessed to %d entries", batch.entries.size());
    for (size_t i=0; i<batch.entries.size(); i++)
        LOG_INF("entry %zu: nx=%d ny=%d", i, batch.entries[i]->nx, batch.entries[i]->ny);
}

void mtmd_debug_preprocess_audio(mtmd_context *ctx, const std::vector<float> &samples) {
    std::vector<mtmd_audio_mel> mels;
    ctx->audio_preproc->preprocess(samples.data(), samples.size(), mels);
    LOG_INF("preprocessed to %zu mel chunks", mels.size());
    for (size_t i=0; i<mels.size(); i++)
        for (int m=0; m<mels[i].n_mel; m++)
            for (int t=0; t<mels[i].n_len; t++)
                LOG_INF("mel[%zu][%d][%d]=%f", i, m, t, mels[i].data[m*mels[i].n_len+t]);
}

// 单元测试辅助：构造含文本+图像的标准 chunks
mtmd_input_chunks *mtmd_test_create_input_chunks() {
    auto chunks = mtmd_input_chunks_init();
    // text chunk: tokens 1..5
    chunks->entries.emplace_back(MTMD_INPUT_CHUNK_TYPE_TEXT, std::vector{1,2,3,4,5}, nullptr, nullptr);
    // image chunk: 4x4 grid, 16 entries
    auto img_tok = std::make_unique<mtmd_image_tokens>();
    img_tok->nx=4; img_tok->ny=4; img_tok->batch_f32.entries.resize(16); img_tok->id="image_1";
    chunks->entries.emplace_back(MTMD_INPUT_CHUNK_TYPE_IMAGE, {}, std::move(img_tok), nullptr);
    return chunks;
}
```

> **调试接口价值**：`debug_encode_*` 可直接查看 CLIP 输出嵌入；`debug_preprocess_*` 打印预处理器输出的切片数量、尺寸、梅尔频谱数值，便于对比 Python 侧参考实现；`test_create_input_chunks` 为单测提供标准输入。

---

## 75.11 设计中的取舍

| 设计决策 | 背景与权衡 | 选择理由 |
|----------|------------|----------|
| **互斥字段而非联合体/变体** | `mtmd_input_chunk` 需同时支持文本/图像/音频，且各模态元数据差异大（图像需 nx/ny/grid，音频仅需 n_tokens） | 互斥 `unique_ptr` 语义清晰、内存紧凑，避免 `std::variant` 的访问开销与 `union` 的非平凡析构复杂性。 |
| **RAII 仅在 C++ 层，C API 保留裸指针** | 需兼容纯 C 调用者（如 Python bindings、Go CGO） | C 层零开销、零依赖；C++ 层通过 `deleter` 实现零成本抽象，两全其美。 |
| **M-RoPE 内存布局：写时通道优先、读时通道交织** | `set_position_mrope_2d` 需批量写入单通道（缓存友好）；llama.cpp 内核期望 `llama_batch.pos` 为 token-major | `decode_embd_batch::get_view` 现场重组，以微小的拷贝成本换取写入性能与内核兼容性。 |
| **音频预处理器统一输出 mel-major** | 视觉 CLIP 习惯 channel-first；音频梅尔频谱的“通道”是频带 | 复用 `clip_image_f32_batch` 与 `clip_image_batch_encode`，避免维度转置或新增数据结构。 |
| **图像切片策略：概览图 + 网格切片** | 高分辨率图像无法直接送入固定尺寸编码器，且全图高分辨率计算量爆炸 | 概览图提供全局语义，切片保留局部细节，网格信息供 M-RoPE 还原空间关系，兼顾效果与算力。 |

---

## 75.12 动手练习

### 75.12.1 练习 1：追踪多模态输入的完整生命周期
阅读 `tools/mtmd/mtmd.cpp` 中 `mtmd_tokenizer` 的构造与核心方法（第508-978行），以及 `mtmd.h` 中的 RAII 包装类（第254-288行），回答以下问题：
1. 当输入文本包含 `<__media__>` 标记时，`tokenize()` 如何交替生成文本 chunk 与媒体 chunk？`add_image_media()` 与 `add_audio_media()` 分别承担什么角色？
2. `mtmd::bitmap` 的四个构造函数分别适用于什么场景？移动构造函数如何确保资源所有权的正确转移？
3. `mtmd_input_chunk_get_n_pos()` 在处理 M-RoPE 图像时与普通文本有何不同？为什么要区分 n_tokens 和 n_pos？
4. 如果模型同时启用了视觉与音频支持，`mtmd_context` 如何确保两种模态的嵌入维度兼容性？

### 75.12.2 练习 2：分析 M-RoPE 位置编码与资源加载机制
阅读 `tools/mtmd/mtmd-helper.cpp` 中 `decode_embd_batch` 的位置设置方法与资源加载函数，完成以下分析：
1. 对比 `set_position_normal()` 与 `set_position_mrope_2d()`：对于一张 2x2 的图像 patch（共4个token），普通模式和 M-RoPE 模式下的 `pos[]` 数组内容分别是什么？
2. 解释为什么 `mtmd_helper_bitmap_init_from_buf()` 需要先检测音频 magic bytes？如果误判会导致什么后果？
3. `get_view()` 方法在处理 M-RoPE 时为什么要重新组织 `pos_view`？如果不重新组织会发生什么错误？

### 75.12.3 练习 3：设计一个新的图像预处理策略
假设你要为一个新模型 "MyVLM" 实现图像预处理，要求：输入图像必须调整为 448x448，且宽高比保持为原图（不足部分填充黑色），并支持可选的网格切片（2x2）。
参考 `tools/mtmd/mtmd-image.h` 和 `mtmd-image.cpp` 中的 `mtmd_image_preprocessor_fixed_size`、`mtmd_image_preprocessor_dyn_size` 与 `mtmd_image_preprocessor_llava_uhd`：
1. 你会选择继承哪个基类？为什么？需要重写哪些方法？
2. 描述预处理流程：该如何组合 `img_tool::resize()`（带 padding）、`slice_image()` 与 `img_u8_to_f32()`？
3. 在 `mtmd_context::init_vision()` 中，如何根据 `projector_type` 注册你的新预处理器？
4. 如果该模型需要 M-RoPE 位置编码，`batch_f32` 中的 `grid_x` 和 `grid_y` 应该如何设置以配合 `decode_embd_batch`？

### 75.12.4 练习 4：调试资源内存管理与深拷贝
阅读 `tools/mtmd/mtmd.h` 第232-288行的 deleter 与 RAII 包装类，以及 `mtmd.cpp` 中的 `mtmd_input_chunk_copy` 与 `clone()` 方法：
1. `mtmd::bitmap` 和 `mtmd::input_chunks` 使用了哪种智能指针模式？自定义 deleter 与默认 deleter 有何不同？
2. `mtmd_input_chunk_copy()` 中对 `tokens_image` 和 `tokens_audio` 的深拷贝是如何实现的？为什么需要调用 `clone()` 而不是简单的指针复制？
3. 在什么情况下调用 `mtmd_bitmap_free()` 会导致双重释放错误？RAII 如何避免这类问题？如果上层应用混用 C API 与 C++ 包装类会发生什么？

---

## 75.13 本章小结

首先，我们剖析了公共 C API 与 C++ RAII 封装，理解 `mtmd_bitmap`/`mtmd_input_chunk`/`mtmd_input_chunks` 三大核心数据结构如何通过 `unique_ptr+deleter` 实现零泄漏资源管理。其次，我们深入 `mtmd_context` 构造流程，看它如何双引擎加载视觉/音频 CLIP、三层校验嵌入维度、并按 `projector_type` 分派 20+ 种预处理器。接着，我们跟随 `mtmd_tokenizer` 走完了“按标记切分→文本合并→图像双模式（切片/非切片）→音频分块”的完整 Tokenization 链路。随后，我们探访了图像预处理工坊：从 `img_tool` 通用操作到 LLaVA-UHD 自适应切片算法，再到 9 种模型专用预处理器的策略差异。音频侧则揭示了 Whisper/Conformer 两大梅尔频谱流水线的异同，以及流式 ISTFT 的重叠相加重建原理。最后，我们见证了 `mtmd-helper` 如何将 M-RoPE 位置编码、非因果注意力、微批次解码、格式无关资源加载串联成上层应用可直接调用的“评估流水线”。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| mtmd_context | 多模态处理核心上下文，统一管理视觉编码器（CLIP-V）与音频编码器（CLIP-A），校验嵌入维度兼容性，并按 projector_type 初始化模态专用预处理器 |
| mtmd::bitmap / mtmd_bitmap | 原始多模态数据容器（图像RGB字节流/音频PCM浮点），C++层通过 unique_ptr + 自定义 deleter 实现 RAII 自动管理，支持ID标记追踪与零拷贝访问 |
| mtmd::input_chunks | 输入块集合的C++包装，提供size()和operator[]访问，通过deleter自动释放资源，避免手动管理C API指针 |
| mtmd_tokenizer | 多模态输入分词器，按 media_marker 切割文本，交替生成文本 chunk 与媒体 chunk，支持图像（image_media）与音频（audio_media）的独立处理路径，处理 BOS/EOS 特殊标记插入 |
| 图像预处理策略家族 | 包含固定尺寸、动态尺寸、最长边、LLaVA-UHD切片、LFM2、Idefics3、InternVL、DeepSeekOCR、YouTuVL等多种策略，通过继承基类实现多态处理 |
| 音频预处理流水线 | PCM采样 → FFT（Cooley-Tukey）→ Mel滤波器组 → 对数压缩 → 分块（Whisper）或中心填充+预强调（Conformer），支持流式ISTFT重建 |
| M-RoPE 位置编码 | 多维旋转位置编码：图像使用 (t=0, y, x, 0) 四维表示空间位置；音频使用 (t, t, t, 0) 表示时间序列，通过 decode_embd_batch 的 set_position_mrope_2d/1d 管理 |
| mtmd_helper_eval_chunks | 高层评估接口：文本 chunk 直接 decode，媒体 chunk 先 encode 为嵌入再 decode，自动处理 micro-batch 分割与 logits 标记，支持多模态混合输入 |
| mtmd_helper_bitmap_init_from_buf/file | 智能资源加载入口：自动检测音频magic bytes，图像走stb_image解码，音频走miniaudio转PCM，统一输出mtmd_bitmap，实现格式无关的多模态输入 |

感谢你读到了这里！本章节已完整覆盖 MTMD 多模态处理核心模块的架构设计与源码实现。后续章节将继续深入 llama.cpp 的其他核心子系统。



