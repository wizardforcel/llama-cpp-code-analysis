
# 第28章 Gemma 家族与多通道创新 —— Google 的"实验场"

## 学习目标

- 理解 Gemma 家族从初代到 Gemma4 的演进脉络及各代架构的核心差异
- 掌握 RMSNorm + RoPE + Self-Attention + SwiGLU/GeLU FFN 的标准 Transformer 构建模式
- 理解 ISWA（Intermittent Sliding Window Attention）双层缓存与按层 RoPE 频率的实现
- 理解 attn_q_norm / attn_k_norm / attn_post_norm / ffn_post_norm 等"夹心"归一化的工程意义
- 掌握 AltUp 多通道并行表示的 predict / correct 机制及其与单通道的形状转换
- 理解 Gemma4 中共享 KV 层（has_kv == false）与 V 投影省略的极致稀疏化设计
- 理解 per-layer embedding 在每层动态注入额外信息的实现原理
- 掌握 f_final_logit_softcapping 通过 tanh 平滑 logits 的实现手法
- 掌握 laurel 低秩残差块与 gaussian_topk 稀疏门控的实现
- 能阅读并修改自定义模型构建器，理解 llm_graph_context 的构图 API

## 28.1 生活类比

想象 Gemma 家族是一组 Google 出品的智能翻译工厂，每一代都在解决不同的工业难题：Gemma 初代是标准流水线工厂，RMSNorm → RoPE → Attention → FFN，每一步都是教科书级别的标准作业；Gemma Embedding 是双轨装卸码头，既能处理文本 token（标准集装箱），也能直接接收视觉嵌入（整箱到货），通过 ubatch.token 判断走哪种卸货通道；Gemma2-ISWA 是双层仓库，一部分层只看最近的小窗口（短期仓库），另一部分层看完整历史（长期仓库），大幅节省存储；Gemma3 模板类是工厂设计院，通过 iswa 模板参数，一个设计图纸同时画两套工厂（带/不带滑动窗），编译时再决定具体建哪种；Gemma3n-ISWA 是并行假设实验中心，同时维护 n_altup 个"平行宇宙"，每层只更新其中一条（active），最后用路由器投票决定哪个宇宙最靠谱。其中，laurel 是实验中心的草稿本，在正式实验前先用低秩近似快速演算一遍（L(R(x))），再把草稿笔记加入结果；gaussian_topk 是实验素材的"优选筛"，把低于均值的素材统统扔掉（ReLU 截断），只保留高于均值 f_sparsity_std_mul 倍标准差的优质原料；calc_magnitude 是实验数据的能量表，计算每个平行宇宙的"信号强度" sqrt(sum(x²))，用来归一化宇宙间的相对影响。Gemma4-ISWA 则是极致共享仓库：共享 KV Cache 让多层共用同一批历史货物（不再每层都备货），V 投影省略把 K 当 V 用（偷懒但还能用），每层嵌入注入让每层都额外塞一份"定制小纸条"（per-layer embedding），MoE 自定义路由让多个专家分担不同货物处理，主路由+辅助专家分工合作。f_final_logit_softcapping 是末端正负能量限制器，通过 tanh 把 logits 限制在 ±c 范围内，防止极端值爆炸。就像 Google 不断在工程化上做减法：让同样的工厂能处理更多品种、占用更少仓库、产出更快更稳。

## 28.2 源码地图

```
src/models/gemma.cpp
├── llm_build_gemma::llm_build_gemma()           # Gemma 初代经典解码器（1-130行）
│   ├── build_inp_embd()                          # 嵌入输入
│   ├── ggml_scale(sqrtf(n_embd))                 # Embedding 缩放
│   ├── build_attn_inp_kv()                       # 标准 KV 输入
│   ├── build_norm(LLM_NORM_RMS)                  # pre-attention RMSNorm
│   ├── build_lora_mm(wq/wk/wv) + ggml_rope_ext   # Q/K/V + RoPE
│   ├── ggml_scale(1/sqrtf(n_embd_head))          # 显式注意力缩放
│   ├── build_attn(...)                           # 标准自注意力
│   ├── build_norm(LLM_NORM_RMS)                  # pre-FFN RMSNorm
│   ├── build_ffn(LLM_FFN_GELU, PAR)              # GELU 激活前馈
│   └── res->t_logits                             # 输出 logits

src/models/gemma-embedding.cpp
├── llm_build_gemma_embedding::llm_build_gemma_embedding()  # Gemma 嵌入模型（1-150行）
│   ├── ubatch.token ? sqrtf(n_embd) : 1.0f       # 视觉嵌入直通
│   ├── build_attn_inp_no_cache()                 # 非自回归输入
│   ├── build_norm(attn_q_norm/attn_k_norm)       # Q/K 前置归一化
│   ├── ggml_scale(hparams.f_attention_scale)     # 可学习注意力缩放
│   ├── build_norm(attn_post_norm)                # 注意力后归一化
│   ├── build_ffn(LLM_FFN_GELU, PAR)
│   ├── build_norm(ffn_post_norm)                 # FFN 后归一化
│   └── res->t_embd                               # 只输出 embedding

src/models/gemma2-iswa.cpp
├── llm_build_gemma2_iswa::llm_build_gemma2_iswa()  # Gemma2 滑动窗口版（1-160行）
│   ├── build_attn_inp_kv_iswa()                  # 双层 KV 缓存输入
│   ├── model.get_rope_freq_base/scale(il)        # 按层 RoPE 频率
│   ├── build_norm(attn_post_norm)                # 夹心归一化
│   ├── build_norm(ffn_post_norm)
│   ├── ggml_scale/tanh/scale                     # 最终 logits 软帽
│   └── res->t_logits

src/models/gemma3.cpp
├── template struct llm_build_gemma3<iswa>        # Gemma3 模板类（1-180行）
│   ├── using inp_attn_type = std::conditional_t  # 编译期分叉内存输入
│   ├── if constexpr (iswa)                       # ISWA 分支
│   │   ├── build_attn_inp_kv_iswa()
│   │   └── model.get_rope_freq_base/scale(il)
│   ├── if constexpr (!iswa)
│   │   ├── build_attn_inp_kv()
│   │   └── freq_base / freq_scale 全局配置
│   ├── build_norm(attn_q_norm/attn_k_norm)
│   ├── build_norm(attn_post_norm)
│   ├── build_norm(ffn_post_norm)
│   ├── if (hparams.f_final_logit_softcapping)    # 可选软帽
│   └── template struct llm_build_gemma3<false>  # 显式实例化
│       template struct llm_build_gemma3<true>

src/models/gemma3n-iswa.cpp
├── llm_build_gemma3n_iswa::llm_build_gemma3n_iswa()  # AltUp 多通道架构（1-130行）
│   ├── project_per_layer_inputs()                # 每层输入投影
│   ├── calc_magnitude()                          # 计算向量幅值
│   ├── ggml_concat(inpL, altup_added, 2)         # 拼接多通道
│   ├── for il = 0 .. n_layer:
│   │   ├── altup_predict()                       # 多通道预测
│   │   ├── view_2d_slice(i_altup_act)            # 取 active 通道
│   │   ├── laurel()                              # 低秩残差块
│   │   ├── if (hparams.has_kv(il))               # 是否本层计算 KV
│   │   ├── gaussian_topk()                       # 稀疏化门控
│   │   ├── altup_correct()                       # 多通道修正
│   │   └── project_per_layer_inputs (each layer) # 每层输入再注入
│   └── altup_unembd_proj / view_2d_slice / mean  # 多通道融合回单通道
├── llm_build_gemma3n_iswa::calc_magnitude(ggml_tensor* x)                # 辅助：sqrt(sum(x²))
├── llm_build_gemma3n_iswa::view_2d_slice(ggml_tensor* x, int idx)        # 辅助：3D → 2D 切片
├── llm_build_gemma3n_iswa::get_per_layer_inputs()                        # 辅助：n_embd_altup × n_layer
├── llm_build_gemma3n_iswa::project_per_layer_inputs()                    # 辅助：投影+归一+缩放
├── llm_build_gemma3n_iswa::laurel()                                     # 辅助：L(R(x)) + x 低秩块
├── llm_build_gemma3n_iswa::gaussian_topk()                              # 辅助：基于均值+标准差阈值
├── llm_build_gemma3n_iswa::altup_compute_router_modalities()             # AltUp 路由器
├── llm_build_gemma3n_iswa::altup_predict()                              # AltUp 预测
└── llm_build_gemma3n_iswa::altup_correct()                              # AltUp 修正

src/models/gemma4-iswa.cpp
├── llm_build_gemma4_iswa::llm_build_gemma4_iswa()    # Gemma4 极致稀疏化（1-180行）
│   ├── project_per_layer_inputs()                # 每层输入投影
│   ├── model.get_rope_freq_base/scale(il)        # 按层 RoPE
│   ├── if (!hparams.is_swa(il))                  # 全局层用 rope_freqs
│   │   └── freq_factors = model.layers[il].rope_freqs
│   ├── build_norm(attn_q_norm)
│   ├── if (hparams.has_kv(il))                   # 仅部分层算 KV
│   │   ├── build_norm(attn_k_norm)
│   │   ├── Vcur = model.layers[il].wv ? wv·x : Kcur  # V 可省略
│   │   └── build_attn()
│   ├── build_norm(attn_post_norm)
│   ├── if (model.layers[il].ffn_gate_inp)        # MoE 分支
│   │   ├── build_ffn (shared MLP)
│   │   ├── ggml_rms_norm + mul(ffn_gate_inp_s)   # 自定义 MoE logits
│   │   └── build_moe_ffn()                       # 专家路由
│   ├── else: build_ffn(LLM_FFN_GELU, PAR)
│   ├── build_norm(ffn_post_norm)
│   ├── per_layer_inp_gate → GELU → mul → per_layer_proj → post_norm
│   ├── if (model.layers[il].out_scale)           # 层残差缩放
│   └── if (hparams.f_final_logit_softcapping)    # 可选软帽
├── llm_build_gemma4_iswa::view_2d_slice()                              # 辅助：3D → 2D
├── llm_build_gemma4_iswa::get_per_layer_inputs()                      # n_embd_per_layer × n_layer
└── llm_build_gemma4_iswa::project_per_layer_inputs()                  # 投影+缩放+permute
```

## 28.3 Gemma 初代基础构建 —— 经典解码器的"通用骨架"

Gemma 初代的实现之所以能成为后续家族的参考模板，是因为它建立了最标准的 Transformer 解码器流水线：标准的 RMSNorm + RoPE + Self-Attention + FFN 流水线，唯一的特色是用 sqrt(n_embd) 对 embedding 做缩放（与 LLaMA 一致），注意力缩放通过 1.0f / sqrtf(float(n_embd_head)) 显式计算，没有 attn_q_norm/attn_k_norm 这种前置归一化，也没有 attn_post_norm 和 ffn_post_norm 的"夹心"结构。它是理解后续 gemma2 / gemma3 / gemma3n / gemma4 进化的起点。

### 28.3.1 llm_build_gemma::llm_build_gemma()：Gemma 初代主构建流程

源码路径：`src/models/gemma.cpp` - `llm_build_gemma`（第1-80行）

```cpp
llm_build_gemma::llm_build_gemma(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();                              // ① 获取每头维度（Value头）
    ggml_tensor * cur;                                                                // ② 当前层张量指针
    ggml_tensor * inpL;                                                               // ③ 残差流主干指针
    inpL = build_inp_embd(model.tok_embd);                                            // ④ Token Embedding 查表
    inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));                                     // ⑤ 缩放 Embedding：×sqrt(d_model)
    cb(inpL, "inp_scaled", -1);                                                       // ⑥ 记录调试节点
    ggml_tensor * inp_pos = build_inp_pos();                                          // ⑦ 位置编码输入
    auto * inp_attn = build_attn_inp_kv();                                            // ⑧ 标准 KV Cache 输入
    ggml_tensor * inp_out_ids = build_inp_out_ids();                                  // ⑨ 输出索引（用于最后一层取行）
    for (int il = 0; il < n_layer; ++il) {                                            // ⑩ 逐层构建
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);   // ⑪ Pre-Attention RMSNorm
        cb(cur, "attn_norm", il);
        {                                                                             // ⑫ Attention 子图
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);              // ⑬ Q 投影 + LoRA
            cb(Qcur, "Qcur", il);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);              // ⑭ K 投影 + LoRA
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);              // ⑮ V 投影 + LoRA
            cb(Vcur, "Vcur", il);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);         // ⑯ 重塑为 [head_dim, n_head, n_tokens]
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);      // ⑰
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);      // ⑱
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow); // ⑲ RoPE
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow); // ⑳
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));           // ㉑ 显式注意力缩放 1/sqrt(d_head)
            cb(Qcur, "Qcur_scaled", il);
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il); // ㉒ 标准自注意力
        }
        if (il == n_layer - 1 && inp_out_ids) {                                       // ㉓ 最后一层按输出索引取行
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);                              // ㉔ Attention 残差相加
        cb(sa_out, "sa_out", il);
        cur = build_norm(sa_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);   // ㉕ Pre-FFN RMSNorm
        cb(cur, "ffn_norm", il);
        {                                                                             // ㉖ FFN 子图
            cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_GELU, LLM_FFN_PAR, il); // GELU 激活
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, sa_out);                                             // ㉗ FFN 残差相加
        cur = build_cvec(cur, il);                                                     // ㉘ Control Vector 注入（可选）
        cb(cur, "l_out", il);
        inpL = cur;                                                                    // ㉙ 更新残差流主干
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);                  // ㉚ 最终 RMSNorm
    cb(cur, "result_norm", -1);
    res->t_embd = cur;                                                                 // ㉛ 保存 embedding 输出
    cur = build_lora_mm(model.output, cur);                                            // ㉜ LM Head 投影 + LoRA
    cb(cur, "result_output", -1);
    res->t_logits = cur;                                                               // ㉝ 保存 logits 输出
    ggml_build_forward_expand(gf, cur);                                                // ㉞ 图构建完成
}
```

这段代码定义了 Gemma 初代经典解码器的完整构建流程：从 Token Embedding 开始，经过 sqrt(n_embd) 缩放，进入标准的 Pre-Norm Transformer 块循环，每层包含 RMSNorm → QKV 投影 → RoPE → 显式注意力缩放 → 自注意力 → 残差 → RMSNorm → GELU FFN → 残差，最后经过输出归一化和 LM Head 产出 logits。

## 28.4 Gemma Embedding 双模态嵌入 —— 文本与图像的"双轨入口"

Embedding 模型需要支持原始嵌入输入，因为多模态场景下，视觉编码器输出已经是向量，直接传入跳过 tokenize。通过 `ubatch.token` 三元判断决定是否乘以 sqrt(n_embd)；引入 attn_q_norm 与 attn_k_norm 在 RoPE 之前做归一化；引入 attn_post_norm 和 ffn_post_norm 形成"pre-norm + post-norm"夹心结构；使用 `build_attn_inp_no_cache()` 因为 Embedding 是非自回归的；用 `hparams.f_attention_scale` 替代硬编码的 sqrt 倒数缩放；`res->t_logits` 未设置，因为只输出 embedding 不输出 logits。

### 28.4.1 llm_build_gemma_embedding::llm_build_gemma_embedding()：嵌入模型主构建流程

源码路径：`src/models/gemma-embedding.cpp` - `llm_build_gemma_embedding`（第3-110行）

```cpp
llm_build_gemma_embedding::llm_build_gemma_embedding(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();                                          // ① 每头维度
    ggml_tensor * cur;                                                                            // ② 当前张量
    ggml_tensor * inpL;                                                                           // ③ 残差主干
    inpL = build_inp_embd(model.tok_embd);                                                        // ④ Token Embedding
    // important: do not normalize weights for raw embeddings input (i.e. encoded image embeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);                           // ⑤ 三元判断：文本缩放，视觉直通
    cb(inpL, "inp_scaled", -1);
    ggml_tensor * inp_pos = build_inp_pos();                                                      // ⑥ 位置编码
    auto * inp_attn = build_attn_inp_no_cache();                                                  // ⑦ 非自回归注意力输入（无 KV Cache）
    ggml_tensor * inp_out_ids = build_inp_out_ids();                                              // ⑧ 输出索引
    for (int il = 0; il < n_layer; ++il) {                                                        // ⑨ 逐层
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);                         // ⑩ 按层 RoPE 基频
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);                        // ⑪ 按层 RoPE 缩放
        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);               // ⑫ Pre-Attention Norm
        cb(cur, "attn_norm", il);
        // self-attention
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);                         // ⑬ Q 投影
            cb(Qcur, "Qcur", il);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);                         // ⑭ K 投影
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);                         // ⑮ V 投影
            cb(Vcur, "Vcur", il);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);                    // ⑯ 重塑
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);                 // ⑰
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);                 // ⑱
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);        // ⑲ Q 前置归一化（关键差异）
            cb(Qcur, "Qcur_normed", il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow); // ⑳ Q RoPE
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);        // ㉑ K 前置归一化（关键差异）
            cb(Kcur, "Kcur_normed", il);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow); // ㉒ K RoPE
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
            Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);                             // ㉓ 可学习注意力缩放
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il); // ㉔ 注意力计算
        }
        if (il == n_layer - 1 && inp_out_ids) {                                                   // ㉕ 最后一层取行
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);           // ㉖ Attention Post-Norm（夹心）
        cb(cur, "attn_post_norm", il);
        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);                                         // ㉗ Attention 残差
        cb(sa_out, "sa_out", il);
        cur = build_norm(sa_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);              // ㉘ Pre-FFN Norm
        cb(cur, "ffn_norm", il);
        // feed-forward network
        {
            cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_GELU, LLM_FFN_PAR, il); // ㉙ GELU FFN
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, -1);            // ㉚ FFN Post-Norm（夹心）
        cb(cur, "ffn_post_norm", -1);
        cur = ggml_add(ctx0, cur, sa_out);                                                        // ㉛ FFN 残差
        cur = build_cvec(cur, il);                                                                // ㉜ Control Vector
        cb(cur, "l_out", il);
        inpL = cur;                                                                               // ㉝ 更新主干
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);                             // ㉞ 最终 Norm
    cb(cur, "result_norm", -1);
    res->t_embd = cur;                                                                            // ㉟ 只输出 embedding
    ggml_build_forward_expand(gf, cur);                                                           // ㊱ 图完成
}
```

这段代码实现了 Gemma Embedding 模型的完整构建流程：核心差异在于第⑤行的三元判断实现文本/视觉双轨入口，第⑦行使用无缓存注意力输入，第⑲⑳㉑㉒行引入 Q/K 前置归一化，第㉖㉚行形成双重 Post-Norm 夹心结构，第㉟行只输出 embedding 不产生 logits。

## 28.5 Gemma2-ISWA 滑动窗口 —— 长上下文的"记忆瘦身术"

ISWA（Intermittent Sliding Window Attention）的引入是为了减少 KV Cache 内存占用，只对部分层启用全局注意力。通过 `build_attn_inp_kv_iswa()` 构建双层缓存输入；引入 RoPE 频率按层缩放（`get_rope_freq_base/scale`）；加入 `attn_post_norm` 与 `ffn_post_norm` 形成对称残差；加入 `f_final_logit_softcapping` 通过 tanh 平滑 logits，`ggml_scale → ggml_tanh → ggml_scale` 三步实现软帽。

### 28.5.1 llm_build_gemma2_iswa::llm_build_gemma2_iswa()：ISWA 版主构建流程

源码路径：`src/models/gemma2-iswa.cpp` - `llm_build_gemma2_iswa`（第26-188行）

```cpp
llm_build_gemma2_iswa::llm_build_gemma2_iswa(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();                                          // ① 每头维度
    ggml_tensor * cur;                                                                            // ② 当前张量
    ggml_tensor * inpL;                                                                           // ③ 残差主干
    inpL = build_inp_embd(model.tok_embd);                                                        // ④ Token Embedding
    inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));                                                 // ⑤ Embedding 缩放
    cb(inpL, "inp_scaled", -1);
    ggml_tensor * inp_pos = build_inp_pos();                                                      // ⑥ 位置编码
    auto * inp_attn = build_attn_inp_kv_iswa();                                                   // ⑦ ISWA 双层 KV Cache 输入（关键差异）
    ggml_tensor * inp_out_ids = build_inp_out_ids();                                              // ⑧ 输出索引
    for (int il = 0; il < n_layer; ++il) {                                                        // ⑨ 逐层
        const float freq_base_l  = model.get_rope_freq_base (cparams, il);                        // ⑩ 按层 RoPE 基频
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);                        // ⑪ 按层 RoPE 缩放
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);               // ⑫ Pre-Attention Norm
        cb(cur, "attn_norm", il);
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);                         // ⑬ Q 投影
            cb(Qcur, "Qcur", il);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);                         // ⑭ K 投影
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);                         // ⑮ V 投影
            cb(Vcur, "Vcur", il);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);                 // ⑯ 重塑
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);                 // ⑰
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);                 // ⑱
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow); // ⑲ Q RoPE
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow); // ⑳ K RoPE
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);                             // ㉑ 可学习注意力缩放
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il); // ㉒ 注意力计算
        }
        if (il == n_layer - 1 && inp_out_ids) {                                                   // ㉓ 最后一层取行
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);           // ㉔ Attention Post-Norm
        cb(cur, "attn_post_norm", il);
        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);                                         // ㉕ Attention 残差
        cb(sa_out, "sa_out", il);
        cur = build_norm(sa_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);              // ㉖ Pre-FFN Norm
        cb(cur, "ffn_norm", il);
        {
            cur = build_ffn(cur, model.layers[il].ffn_up,   NULL, NULL, model.layers[il].ffn_gate, NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_GELU, LLM_FFN_PAR, il); // ㉗ GELU FFN
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, -1);            // ㉘ FFN Post-Norm
        cb(cur, "ffn_post_norm", -1);
        cur = ggml_add(ctx0, cur, sa_out);                                                        // ㉙ FFN 残差
        cur = build_cvec(cur, il);                                                                // ㉚ Control Vector
        cb(cur, "l_out", il);
        inpL = cur;                                                                               // ㉛ 更新主干
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);                             // ㉜ 最终 Norm
    cb(cur, "result_norm", -1);
    res->t_embd = cur;                                                                            // ㉝ 保存 embedding
    cur = build_lora_mm(model.output, cur);                                                       // ㉞ LM Head
    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);                        // ㉟ Logit Softcapping 第1步：缩放
    cur = ggml_tanh(ctx0, cur);                                                                   // ㊱ 第2步：tanh 非线性压缩
    cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);                               // ㊲ 第3步：反缩放
    cb(cur, "result_output", -1);
    res->t_logits = cur;                                                                          // ㊳ 保存 logits
    ggml_build_forward_expand(gf, cur);                                                           // ㊴ 图完成
}
```

这段代码实现了 Gemma2-ISWA 的完整构建流程：第⑦行引入 ISWA 双层 KV Cache 输入，第⑩⑪行实现按层 RoPE 频率，第㉔㉘行形成对称的 Post-Norm 夹心结构，第㉟㊱㊲行三步实现 Logit Softcapping（scale → tanh → scale）。

## 28.6 Gemma3 模板类 —— 一个类，两种内存策略

Gemma3 使用模板参数控制 ISWA，通过 `template<bool iswa>` 在编译期分叉内存输入类型，使用 `std::conditional_t` 优雅切换 `attn_kv` 与 `attn_kv_iswa`；ISWA 模式下启用按层 RoPE 频率缩放，非 ISWA 用全局配置；保留 attn_q_norm / attn_k_norm / attn_post_norm / ffn_post_norm；`f_final_logit_softcapping` 可选启用（`if (hparams.f_final_logit_softcapping)`）；文件末尾显式实例化 `template struct llm_build_gemma3<false/true>`，这是代码复用的经典 C++ 技巧。

### 28.6.1 llm_build_gemma3::llm_build_gemma3<iswa>()：模板化构建流程

源码路径：`src/models/gemma3.cpp` - `llm_build_gemma3`（第11-103行）

```cpp
template <bool iswa>
llm_build_gemma3<iswa>::llm_build_gemma3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();                                          // ① 每头维度
    ggml_tensor * cur;                                                                            // ② 当前张量
    ggml_tensor * inpL;                                                                           // ③ 残差主干
    inpL = build_inp_embd(model.tok_embd);                                                        // ④ Token Embedding
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);                           // ⑤ 三元判断：文本缩放/视觉直通
    cb(inpL, "inp_scaled", -1);
    ggml_tensor * inp_pos = build_inp_pos();                                                      // ⑥ 位置编码
    using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>; // ⑦ 编译期类型切换
    inp_attn_type * inp_attn = nullptr;
    if constexpr (iswa) {                                                                         // ⑧ ISWA 分支
        inp_attn = build_attn_inp_kv_iswa();
    } else {                                                                                      // ⑨ 标准分支
        inp_attn = build_attn_inp_kv();
    }
    ggml_tensor * inp_out_ids = build_inp_out_ids();                                              // ⑩ 输出索引
    for (int il = 0; il < n_layer; ++il) {                                                        // ⑪ 逐层
        float freq_base_l = 0.0f;
        float freq_scale_l = 0.0f;
        if constexpr (iswa) {                                                                     // ⑫ ISWA 按层 RoPE
            freq_base_l = model.get_rope_freq_base(cparams, il);
            freq_scale_l = model.get_rope_freq_scale(cparams, il);
        } else {                                                                                  // ⑬ 标准全局 RoPE
            freq_base_l = freq_base;
            freq_scale_l = freq_scale;
        }
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);               // ⑭ Pre-Attention Norm
        cb(cur, "attn_norm", il);
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);                         // ⑮ Q 投影
            cb(Qcur, "Qcur", il);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);                         // ⑯ K 投影
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);                         // ⑰ V 投影
            cb(Vcur, "Vcur", il);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);                    // ⑱ 重塑
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);                 // ⑲
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);                 // ⑳
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);        // ㉑ Q 前置归一化
            cb(Qcur, "Qcur_normed", il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow); // ㉒ Q RoPE
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);        // ㉓ K 前置归一化
            cb(Kcur, "Kcur_normed", il);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow); // ㉔ K RoPE
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);                             // ㉕ 注意力缩放
            cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il); // ㉖ 注意力计算
        }
        if (il == n_layer - 1 && inp_out_ids) {                                                   // ㉗ 最后一层取行
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);           // ㉘ Attention Post-Norm
        cb(cur, "attn_post_norm", il);
        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);                                         // ㉙ Attention 残差
        cb(sa_out, "sa_out", il);
        cur = build_norm(sa_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);              // ㉚ Pre-FFN Norm
        cb(cur, "ffn_norm", il);
        {
            cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_GELU, LLM_FFN_PAR, il); // ㉛ GELU FFN
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, -1);            // ㉜ FFN Post-Norm
        cb(cur, "ffn_post_norm", il);
        cur = ggml_add(ctx0, cur, sa_out);                                                        // ㉝ FFN 残差
        cur = build_cvec(cur, il);                                                                // ㉞ Control Vector
        cb(cur, "l_out", il);
        inpL = cur;                                                                               // ㉟ 更新主干
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);                             // ㊱ 最终 Norm
    cb(cur, "result_norm", -1);
    res->t_embd = cur;                                                                            // ㊲ 保存 embedding
    cur = build_lora_mm(model.output, cur);                                                       // ㊳ LM Head
    if (hparams.f_final_logit_softcapping) {                                                      // ㊴ 可选 Softcapping
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;                                                                          // ㊵ 保存 logits
    ggml_build_forward_expand(gf, cur);                                                           // ㊶ 图完成
}
```

这段代码展示了 Gemma3 模板类的核心设计：第⑦行使用 `std::conditional_t` 在编译期选择注意力输入类型，第⑧⑨⑫⑬行通过 `if constexpr` 实现 ISWA/标准两种模式的零开销分叉，第㊴行可选的 Softcapping 由运行时参数控制。文件末尾显式实例化两个模板特化：

```cpp
template struct llm_build_gemma3<false>;
template struct llm_build_gemma3<true>;
```

这行代码强制编译器生成两个具体的类实例，分别对应标准注意力和 ISWA 注意力版本，避免了在使用时的隐式实例化开销。

## 28.7 Gemma3n 多通道 AltUp —— "并行假设空间"的图构建

AltUp（Alternating Updates）的核心思想是同时维护 n_altup 个并行表示通道，每层只更新其中一条（active），通过 `altup_predict` 用 router 计算所有通道的预测值，通过 `altup_correct` 用 router 计算修正后的多通道。入参嵌入（inp_per_layer）单独投影到 n_embd_altup 维度，多通道拼接：`ggml_concat(inpL, altup_added, 2)`，输出时通过 `altup_unembd_proj` 投影回单一通道。共享 KV 层（`hparams.has_kv(il)` 为 false）复用之前的 KV Cache，V 投影在 shared KV 层被省略（`if (hparams.has_kv(il))` 分支）。Gaussian TopK 激活稀疏化（`f_sparsity_std_mul` 控制阈值），Laurel 残差块：`tmp = L(R(x)) + x` 的低秩增量，calc_magnitude 计算 sqrt(sum(x²)) 用于幅值归一化，view_2d_slice 从 3D 张量切出 active 通道的 2D 视图。

### 28.7.1 llm_build_gemma3n_iswa::llm_build_gemma3n_iswa()：AltUp 主构建流程

源码路径：`src/models/gemma3n-iswa.cpp` - `llm_build_gemma3n_iswa`（第8-142行）

```cpp
llm_build_gemma3n_iswa::llm_build_gemma3n_iswa(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params),
    model(model),
    n_embd_head(model.hparams.n_embd_head_k()),
    n_embd_altup(model.hparams.n_embd_altup),
    n_altup(model.hparams.n_altup),
    i_altup_act(model.hparams.i_altup_act) {
    ggml_tensor * cur;
    ggml_tensor * inpL;
    
    inpL = build_inp_embd(model.tok_embd);                                                          // ① Token Embedding
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);                            // ② 三元判断缩放
    cb(inpL, "inp_scaled", -1);
    ggml_tensor * inp_pos = build_inp_pos();                                                        // ③ 位置编码
    auto * inp_attn = build_attn_inp_kv_iswa();                                                     // ④ ISWA 注意力输入
    ggml_tensor * inp_per_layer = project_per_layer_inputs(inpL, get_per_layer_inputs());           // ⑤ 每层输入投影
    
    {                                                                                               // ⑥ 多通道初始化：堆叠 n_altup 个通道
        ggml_tensor * target_magnitude = calc_magnitude(inpL);                                      // ⑦ 计算主干幅值
        ggml_tensor * inp_repeated     = ggml_repeat_4d(ctx0, inpL, n_embd, n_tokens, n_altup - 1, 1); // ⑧ 重复主干到其他通道
        ggml_tensor * altup_added = ggml_mul_mat(ctx0, model.altup_proj, inp_repeated);             // ⑨ 投影生成额外通道
        ggml_tensor * new_magnitude = calc_magnitude(altup_added);                                  // ⑩ 计算新通道幅值
        altup_added                 = ggml_div(ctx0, ggml_mul(ctx0, altup_added, target_magnitude), new_magnitude); // ⑪ 幅值归一化对齐
        inpL                        = ggml_concat(ctx0, inpL, altup_added, 2);                      // ⑫ 拼接：[n_embd, n_tokens, n_altup]
        cb(inpL, "inp_stacked", -1);
    }
    
    for (int il = 0; il < n_layer; ++il) {                                                          // ⑬ 逐层处理
        ...
        ggml_tensor * predictions = altup_predict(cur, il);                                         // ⑭ AltUp 预测阶段
        ...
        ggml_tensor * corrected = altup_correct(predictions, attn_ffw_laurel_gated, il);            // ⑮ AltUp 修正阶段
        ...
        inpL = cur;                                                                                 // ⑯ 更新主干
    }
    cur = inpL;
    
    {                                                                                               // ⑰ 多通道融合回单通道
        ...
        for (int i = 0; i < n_altup - 1; ++i) {
            cur = ggml_add(ctx0, cur, view_2d_slice(altup_unembd, i));                              // ⑱ 累加所有额外通道
        }
        cur = ggml_scale(ctx0, cur, 1.0f / float(n_altup));                                         // ⑲ 平均融合
        cb(cur, "unembd_merged", -1);
    }
    
    ggml_tensor * inp_out_ids = build_inp_out_ids();                                                // ⑳ 取输出行
    cur                       = ggml_get_rows(ctx0, cur, inp_out_ids);
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);                               // ㉑ 最终 Norm
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);                                                         // ㉒ LM Head
    {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);                      // ㉓ Softcapping
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);                                                             // ㉔ 图完成
}
```

这段代码实现了 Gemma3n 的核心 AltUp 机制：第⑥⑫块完成多通道初始堆叠（主干 + 投影生成的 n_altup-1 个通道），第⑭⑮行在每层执行 predict-correct 两阶段更新，第⑰⑲块通过平均融合将多通道压缩回单通道。

### 28.7.2 llm_build_gemma3n_iswa::calc_magnitude()：向量幅值计算

源码路径：`src/models/gemma3n-iswa.cpp` - `calc_magnitude`（第144-146行）

```cpp
ggml_tensor * llm_build_gemma3n_iswa::calc_magnitude(ggml_tensor * x) {
    return ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, x)));  // sqrt(sum(x²))：L2 范数
}
```

这段代码实现了向量幅值计算：先逐元素平方（`ggml_sqr`），再按行求和（`ggml_sum_rows`），最后开方（`ggml_sqrt`），得到每个 token 向量的 L2 范数，用于多通道幅值对齐归一化。

### 28.7.3 llm_build_gemma3n_iswa::view_2d_slice()：3D 张量切片

源码路径：`src/models/gemma3n-iswa.cpp` - `view_2d_slice`（第149-153行）

```cpp
ggml_tensor * llm_build_gemma3n_iswa::view_2d_slice(ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);  // 断言：索引不越界第3维
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * ggml_element_size(x));  // 零拷贝视图
}
```

这段代码从 3D 张量 `[n_embd, n_tokens, n_altup]` 的第3维切出索引为 `idx` 的 2D 视图 `[n_embd, n_tokens]`，实现零拷贝访问特定通道，是 AltUp 机制中频繁使用的核心辅助函数。

### 28.7.4 llm_build_gemma3n_iswa::get_per_layer_inputs()：获取每层额外输入

源码路径：`src/models/gemma3n-iswa.cpp` - `get_per_layer_inputs`（第156-173行）

```cpp
ggml_tensor * llm_build_gemma3n_iswa::get_per_layer_inputs() {
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);                                    // ① 创建嵌入输入节点
    ggml_tensor * inp_per_layer;
    if (ubatch.token) {                                                                           // ② 文本路径：正常 token 查表
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;
        inp_per_layer = ggml_get_rows(ctx0, model.tok_embd_per_layer, inp->tokens);               // ③ 查表得到 [n_embd_altup * n_layer, n_tokens]
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_altup, n_layer, n_tokens);   // ④ 重塑为 3D
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, sqrtf((float) n_embd_altup));             // ⑤ 缩放
        cb(inp_per_layer, "inp_per_layer_selected", -1);
        res->add_input(std::move(inp));
    } else {                                                                                      // ⑥ 视觉路径：用 padding token 填充
        const int64_t embd_size = model.tok_embd_per_layer->ne[0];
        ggml_tensor * padding = ggml_view_1d(ctx0, model.tok_embd_per_layer, embd_size, 0);       // ⑦ 取第0行
        inp_per_layer = ggml_cast(ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_altup, n_layer, 1);           // ⑧ 重塑 [n_embd_altup, n_layer, 1]
        cb(inp_per_layer, "inp_per_layer_vision", -1);
    }
    return inp_per_layer;
}
```

这段代码实现了每层额外输入的双路径获取：文本模式下通过 token ID 查表并重塑为 `[n_embd_altup, n_layer, n_tokens]`，视觉模式下直接取嵌入矩阵第0行（padding token）广播到所有层。

### 28.7.5 llm_build_gemma3n_iswa::project_per_layer_inputs()：投影每层输入

源码路径：`src/models/gemma3n-iswa.cpp` - `project_per_layer_inputs`（第176-190行）

```cpp
ggml_tensor * llm_build_gemma3n_iswa::project_per_layer_inputs(ggml_tensor * inputs_embeds, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);                        // ① 投影缩放 1/sqrt(d_model)
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);                                  // ② 残差缩放 1/sqrt(2)
    
    ggml_tensor * per_layer_proj = ggml_mul_mat(ctx0, model.per_layer_model_proj, inputs_embeds); // ③ 主干投影
    per_layer_proj               = ggml_scale(ctx0, per_layer_proj, per_layer_projection_scale);  // ④ 缩放
    per_layer_proj               = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_altup, n_layer, n_tokens); // ⑤ 重塑
    per_layer_proj               = build_norm(per_layer_proj, model.per_layer_proj_norm, NULL, LLM_NORM_RMS, -1); // ⑥ 归一化
    cb(per_layer_proj, "per_layer_proj", -1);
    
    inp_per_layer = ggml_add(ctx0, per_layer_proj, inp_per_layer);                                // ⑦ 与原始每层输入相加
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);                       // ⑧ 缩平
    cb(inp_per_layer, "inp_per_layer", -1);
    
    inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));               // ⑨ 维度置换：[n_embd_altup, n_tokens, n_layer]
    return inp_per_layer;
}
```

这段代码将主干 embedding 投影到每层输入空间，与原始每层输入相加后进行维度置换，使层维度成为最后一维，便于后续逐层 `view_2d_slice` 切片。

### 28.7.6 llm_build_gemma3n_iswa::laurel()：Laurel 低秩残差块

源码路径：`src/models/gemma3n-iswa.cpp` - `laurel`（第193-201行）

```cpp
ggml_tensor * llm_build_gemma3n_iswa::laurel(ggml_tensor * cur, int il) {
    ggml_tensor * tmp = cur;
    tmp               = build_lora_mm(model.layers[il].laurel_l, tmp);                            // ① 低秩投影 L：n_embd → n_embd_laurel
    tmp               = build_lora_mm(model.layers[il].laurel_r, tmp);                            // ② 低秩投影 R：n_embd_laurel → n_embd
    tmp               = build_norm(tmp, model.layers[il].laurel_post_norm, NULL, LLM_NORM_RMS, il); // ③ 后置归一化
    tmp               = ggml_add(ctx0, tmp, cur);                                                 // ④ 残差连接：L(R(x)) + x
    cb(tmp, "laurel_out", il);
    return tmp;
}
```

这段代码实现了 Laurel 低秩残差块：两个小矩阵乘法实现低秩近似 `L(R(x))`，后接 RMSNorm 和残差相加，相当于在主干上叠加一个轻量级的"修正草稿"。

### 28.7.7 llm_build_gemma3n_iswa::gaussian_topk()：高斯 TopK 稀疏门控

源码路径：`src/models/gemma3n-iswa.cpp` - `gaussian_topk`（第204-210行）

```cpp
ggml_tensor * llm_build_gemma3n_iswa::gaussian_topk(ggml_tensor * x) {
    ggml_tensor * mean = ggml_mean(ctx0, x);                                                      // ① 计算均值
    ggml_tensor * std  = ggml_sqrt(ctx0, ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, ggml_sub(ctx0, x, mean))), 1.0f / (float) (x->ne[0] - 1))); // ② 计算标准差
    ggml_tensor * cutoff_x = ggml_add(ctx0, mean, ggml_scale(ctx0, std, f_sparsity_std_mul));     // ③ 截断阈值：mean + k * std
    return ggml_relu(ctx0, ggml_sub(ctx0, x, cutoff_x));                                          // ④ ReLU( x - cutoff )：只保留显著激活
}
```

这段代码实现了基于统计量的稀疏门控：计算均值和标准差，设定阈值为 `mean + f_sparsity_std_mul * std`，只保留高于阈值的激活值（ReLU 截断负值），实现"优选筛"效果。

### 28.7.8 llm_build_gemma3n_iswa::altup_compute_router_modalities()：路由器模态计算

源码路径：`src/models/gemma3n-iswa.cpp` - `altup_compute_router_modalities`（第215-223行）

```cpp
ggml_tensor * llm_build_gemma3n_iswa::altup_compute_router_modalities(ggml_tensor * x, int il) {
    ggml_tensor * router_inputs = build_norm(x, model.layers[il].altup_router_norm, NULL, LLM_NORM_RMS, il); // ① Router 专用 Norm
    router_inputs = ggml_scale(ctx0, router_inputs, 1.0f / (float) n_embd);                                  // ② 1/n_embd 缩放
    ggml_tensor * output = ggml_mul_mat(ctx0, model.layers[il].altup_router, router_inputs);                 // ③ 线性投影
    return ggml_tanh(ctx0, output);                                                                          // ④ tanh 激活 → [-1, 1] 权重
}
```

这段代码计算每个通道的路由权重：输入先经专用 RMSNorm，再按嵌入维度缩放，通过线性层投影到 `n_altup` 维，最后 tanh 压缩到 [-1, 1] 区间作为混合系数。

### 28.7.9 llm_build_gemma3n_iswa::altup_predict()：AltUp 预测阶段

源码路径：`src/models/gemma3n-iswa.cpp` - `altup_predict`（第227-241行）

```cpp
ggml_tensor * llm_build_gemma3n_iswa::altup_predict(ggml_tensor * cur, int il) {
    ggml_tensor * activated  = view_2d_slice(cur, i_altup_act);                                    // ① 取出 active 通道 [n_embd, n_tokens]
    ggml_tensor * modalities = altup_compute_router_modalities(activated, il);                     // ② 计算路由权重
    cb(modalities, "modalities", il);
    
    ggml_tensor * all_coefs = build_lora_mm(model.layers[il].altup_predict_coef, modalities);      // ③ 预测系数矩阵 [n_altup, n_altup]
    cb(all_coefs, "all_coefs", il);
    all_coefs = ggml_reshape_3d(ctx0, all_coefs, n_altup, n_altup, n_tokens);                    // ④ 重塑为 3D
    
    ggml_tensor * cur_permuted = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));            // ⑤ 置换：[n_tokens, n_altup, n_embd]
    ggml_tensor * predictions  = ggml_mul_mat(ctx0, cur_permuted, all_coefs);                      // ⑥ 批量矩阵乘：所有通道线性组合
    
    predictions = ggml_cont(ctx0, ggml_permute(ctx0, predictions, 0, 2, 1, 3));                   // ⑦ 置换回：[n_embd, n_altup, n_tokens]
    predictions = ggml_add(ctx0, predictions, cur);                                               // ⑧ 残差连接：预测值 + 原值
    cb(predictions, "predictions", il);
    
    return predictions;
}
```

这段代码实现了 AltUp 预测阶段：取 active 通道计算路由权重，生成 `n_altup × n_altup` 的系数矩阵，将所有通道作为向量进行线性组合预测下一状态，最后加上残差。

### 28.7.10 llm_build_gemma3n_iswa::altup_correct()：AltUp 修正阶段

源码路径：`src/models/gemma3n-iswa.cpp` - `altup_correct`（第246-263行）

```cpp
ggml_tensor * llm_build_gemma3n_iswa::altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il) {
    ggml_tensor * modalities = altup_compute_router_modalities(activated, il);                     // ① 重新计算路由权重
    cb(modalities, "modalities", il);
    
    ggml_tensor * active_prediction = view_2d_slice(predictions, i_altup_act);                     // ② 取出 active 通道的预测值
    ggml_tensor * innovation        = ggml_sub(ctx0, activated, active_prediction);                // ③ 创新项 = 真实 - 预测
    cb(innovation, "innovation", il);
    
    ggml_tensor * all_coefs = build_lora_mm(model.layers[il].altup_correct_coef, modalities);      // ④ 修正系数矩阵
    all_coefs               = ggml_scale_bias(ctx0, all_coefs, 1.0f, 1.0f);                        // ⑤ +1.0 偏移：确保对角线接近 1
    cb(all_coefs, "all_coefs", il);
    all_coefs = ggml_transpose(ctx0, all_coefs);                                                   // ⑥ 转置
    all_coefs = ggml_cont_3d(ctx0, all_coefs, 1, n_tokens, n_altup);                               // ⑦ 重塑
    
    innovation              = ggml_repeat_4d(ctx0, innovation, n_embd, n_tokens, n_altup, 1);      // ⑧ 广播创新项到所有通道
    ggml_tensor * corrected = ggml_mul(ctx0, innovation, all_coefs);                               // ⑨ 按系数加权创新
    corrected               = ggml_add(ctx0, corrected, predictions);                              // ⑩ 修正 = 预测 + 加权创新
    cb(corrected, "corrected", il);
    
    return corrected;
}
```

这段代码实现了 AltUp 修正阶段：计算 active 通道的预测误差（创新项），通过路由器生成的修正系数矩阵将创新项广播并加权分配到所有通道，最后加到预测值上得到修正后的多通道表示。

## 28.8 Gemma4 共享 KV 与每层嵌入 —— "极致稀疏化"的工业实践

Gemma4 更激进地共享 KV：大部分层不计算 K/V，直接复用更早层的 KV Cache；通过 `hparams.has_kv(il)` 区分是否为本层的 KV 计算层；全局层（non-SWA）使用 `rope_freqs` 频率因子实现比例 RoPE；V 投影可选省略：若 wv 不存在则直接复用 K 作为 V；引入 MoE 支持：`ffn_gate_inp` 存在时走专家路由；路由 logits 由 attn_out（而非 norm 后的 cur）计算；MoE 场景支持单独的 `ffn_gate_up_exps`（合并的 up/gate 矩阵）；缩放专家：`ffn_gate_up_exps_s`、`ffn_down_exps_s` 提供 per-channel scale；每层嵌入投影：`per_layer_inp_gate → GELU → mul → per_layer_proj → post_norm`；`out_scale` 层缩放因子实现类似 DeepSeek 的残差缩放；V 投影用 `ggml_rms_norm` 而非 build_norm（细节差异）。

### 28.8.1 llm_build_gemma4_iswa::llm_build_gemma4_iswa()：Gemma4 主构建流程

源码路径：`src/models/gemma4-iswa.cpp` - `llm_build_gemma4_iswa`（第1-221行）

```cpp
llm_build_gemma4_iswa::llm_build_gemma4_iswa(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);                                                          // ① Token Embedding
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);                            // ② 三元判断缩放
    cb(inpL, "inp_scaled", -1);

    ggml_tensor * inp_pos = build_inp_pos();                                                        // ③ 位置编码
    auto * inp_attn = build_attn_inp_kv_iswa();                                                     // ④ ISWA 注意力输入

    ggml_tensor * inp_per_layer = nullptr;
    if (model.tok_embd_per_layer) {                                                                 // ⑤ 每层嵌入投影准备
        inp_per_layer = project_per_layer_inputs(inpL, get_per_layer_inputs());
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();                                                // ⑥ 输出索引

    for (int il = 0; il < n_layer; ++il) {                                                          // ⑦ 逐层
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);                           // ⑧ 按层 RoPE 基频
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);                          // ⑨ 按层 RoPE 缩放
        const int   n_rot_l      = hparams.n_rot(il);                                               // ⑩ 按层旋转维度

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);              // ⑪ Pre-Attention Norm
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {                                                                  // ⑫ 全局层使用比例 RoPE
            freq_factors = model.layers[il].rope_freqs;                                             // ⑬ rope_freqs 频率因子
        }

        // Q projection (shared for both non-KV and KV layers)
        ggml_tensor * Qcur;
        {
            Qcur = build_lora_mm(model.layers[il].wq, cur);                                         // ⑭ Q 投影
            cb(Qcur, "Qcur", il);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);                     // ⑮ 重塑
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);      // ⑯ Q 前置归一化
            cb(Qcur, "Qcur_normed", il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);                   // ⑰ Q RoPE（支持频率因子）
            cb(Qcur, "Qcur_pos", il);
        }

        // self-attention
        if (hparams.has_kv(il)) {                                                                   // ⑱ 仅 KV 层计算 K/V
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);                           // ⑲ K 投影
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = model.layers[il].wv                                               // ⑳ V 投影可省略
                                    ? build_lora_mm(model.layers[il].wv, cur)
                                    : Kcur; // if v_proj is not present, use Kcur as Vcur
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);                 // ㉑ 重塑
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);      // ㉒ K 前置归一化
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);                              // ㉓ V 用直接 rms_norm（非 build_norm）
            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);                   // ㉔ K RoPE
            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn, model.layers[il].wo, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);                                                // ㉕ 完整注意力
        } else {
            // reuse KV cache of earlier layers
            cur = build_attn(inp_attn, model.layers[il].wo, nullptr, Qcur, nullptr, nullptr, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);                                                // ㉖ 共享 KV：只用 Q，复用缓存
        }

        if (il == n_layer - 1 && inp_out_ids) {                                                    // ㉗ 最后一层取行
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);         // ㉘ Attention Post-Norm
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);                                        // ㉙ Attention 残差
        cb(attn_out, "attn_out", il);

        // feed-forward network
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;                        // ㉚ 判断是否 MoE 层
        if (is_moe_layer) {
            // MLP (shared exp)
            ggml_tensor * cur_mlp = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il); // ㉛ 共享专家 Pre-Norm
            cb(cur_mlp, "ffn_norm_1", il);

            cur_mlp = build_ffn(cur_mlp, model.layers[il].ffn_up, nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);                                       // ㉜ 共享专家 FFN
            cur_mlp = build_norm(cur_mlp, model.layers[il].ffn_post_norm_1, nullptr, LLM_NORM_RMS, il); // ㉝ 共享专家 Post-Norm
            cb(cur_mlp, "ffn_mlp", il);

            // Expert FFN
            ggml_tensor * cur_moe = build_norm(attn_out, model.layers[il].ffn_pre_norm_2, nullptr, LLM_NORM_RMS, il); // ㉞ 专家 Pre-Norm（基于 attn_out）
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);             // ㉟ 直接 RMSNorm
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));                             // ㊱ 1/sqrt(d) 缩放
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);                            // ㊲ per-channel scale
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp);              // ㊳ 自定义路由 logits
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe, nullptr, nullptr, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps, nullptr, nullptr,
                    model.layers[il].ffn_down_exps_s);                                             // ㊴ 专家 FFN（合并 gate/up）
            cur_moe = build_norm(cur_moe, model.layers[il].ffn_post_norm_2, nullptr, LLM_NORM_RMS, il); // ㊵ 专家 Post-Norm
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);                                               // ㊶ 共享+专家合并
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);      // ㊷ 标准 FFN Pre-Norm
            cb(cur, "ffn_norm", il);
            cur = build_ffn(cur, model.layers[il].ffn_up, nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);                                       // ㊸ 标准 GELU FFN
            cb(cur, "ffn_out", il);
        }

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);          // ㊹ 统一 FFN Post-Norm
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);                                                       // ㊺ FFN 残差

        // per-layer embedding
        if (inp_per_layer) {                                                                       // ㊻ 每层嵌入注入
            ggml_tensor * pe_in = cur;
            cb(cur, "pe_in", il);

            cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur);                         // ㊼ Gate 投影
            cur = ggml_gelu(ctx0, cur);                                                            // ㊽ GELU 激活
            ggml_tensor * inp_this_layer = view_2d_slice(inp_per_layer, il);                       // ㊾ 取当前层嵌入

            if (il == n_layer - 1 && inp_out_ids) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);                                             // ㊿ 逐元素相乘（门控）
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur);                             // ㊱ 投影回 n_embd
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, il); // ㊲ 后置归一化
            cb(cur, "per_layer_embd_out", il);

            cur = ggml_add(ctx0, pe_in, cur);                                                      // ㊳ 残差相加
        }

        if (model.layers[il].out_scale) {                                                          // ㊴ 层残差缩放
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);                                                                 // ㊵ Control Vector
        cb(cur, "l_out", il);
        inpL = cur;
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);                           // ㊶ 最终 Norm
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);                                                        // ㊷ LM Head

    if (hparams.f_final_logit_softcapping) {                                                       // ㊸ 可选 Softcapping
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);                                                            // ㊹ 图完成
}
```

这段代码实现了 Gemma4 的极致稀疏化设计：第⑱行通过 `has_kv(il)` 判断是否为 KV 计算层，第⑳行 V 投影可省略（用 K 代替），第㉖行非 KV 层直接复用缓存；第㉚行引入 MoE 分支，第㊳行自定义路由 logits 计算（基于 attn_out 而非 norm 后的 cur），第㊴行使用合并的 `ffn_gate_up_exps` 矩阵；第㊻块实现每层嵌入注入的五步流程；第㊴行层残差缩放类似 DeepSeek 机制。

### 28.8.2 llm_build_gemma4_iswa::view_2d_slice()：3D 切片辅助函数

源码路径：`src/models/gemma4-iswa.cpp` - `view_2d_slice`（第222-226行）

```cpp
ggml_tensor * llm_build_gemma4_iswa::view_2d_slice(ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}
```

与 Gemma3n 相同，从 `[n_embd_per_layer, n_tokens, n_layer]` 切出第 `il` 层的 `[n_embd_per_layer, n_tokens]` 视图。

### 28.8.3 llm_build_gemma4_iswa::get_per_layer_inputs()：获取每层嵌入输入

源码路径：`src/models/gemma4-iswa.cpp` - `get_per_layer_inputs`（第227-250行）

```cpp
ggml_tensor * llm_build_gemma4_iswa::get_per_layer_inputs() {
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    ggml_tensor * inp_per_layer;
    if (ubatch.token) {                                                                            // ① 文本路径
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;
        inp_per_layer = ggml_get_rows(ctx0, model.tok_embd_per_layer, inp->tokens);                // ② 查表
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, n_tokens); // ③ 重塑
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, sqrtf((float) n_embd_per_layer));         // ④ 缩放
        cb(inp_per_layer, "inp_per_layer_selected", -1);
        res->add_input(std::move(inp));
    } else {                                                                                       // ⑤ 视觉路径：用 padding token(0) 填充
        const int64_t embd_size = model.tok_embd_per_layer->ne[0];
        ggml_tensor * padding = ggml_view_1d(ctx0, model.tok_embd_per_layer, embd_size, 0);       // ⑥ 取第0行
        inp_per_layer = ggml_cast(ctx0, padding, GGML_TYPE_F32);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, 1);       // ⑦ 重塑 [n_embd_per_layer, n_layer, 1]
        cb(inp_per_layer, "inp_per_layer_vision", -1);
    }
    return inp_per_layer;
}
```

双路径获取每层嵌入：文本模式按 token 查表重塑为 3D，视觉模式取 padding token 嵌入广播到所有层。

### 28.8.4 llm_build_gemma4_iswa::project_per_layer_inputs()：投影每层输入

源码路径：`src/models/gemma4-iswa.cpp` - `project_per_layer_inputs`（第251-266行）

```cpp
ggml_tensor * llm_build_gemma4_iswa::project_per_layer_inputs(ggml_tensor * inputs_embeds, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);                        // ① 投影缩放
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);                                  // ② 残差缩放

    ggml_tensor * per_layer_proj = ggml_mul_mat(ctx0, model.per_layer_model_proj, inputs_embeds); // ③ 主干投影
    per_layer_proj               = ggml_scale(ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj               = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens); // ④ 重塑
    per_layer_proj               = build_norm(per_layer_proj, model.per_layer_proj_norm, nullptr, LLM_NORM_RMS, -1); // ⑤ 归一化
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = ggml_add(ctx0, per_layer_proj, inp_per_layer);                                // ⑥ 相加
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);                       // ⑦ 缩平
    cb(inp_per_layer, "inp_per_layer", -1);

    inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));               // ⑧ 置换：层维度移到最后
    return inp_per_layer;
}
```

将主干 embedding 投影到每层空间，与原始每层输入相加归一化后置换维度，便于逐层 `view_2d_slice` 切片。

## 28.9 设计中的取舍

**为什么 Gemma 要引入这么多 Post-Norm（attn_post_norm, ffn_post_norm）？**

标准 Transformer 只有 Pre-Norm，Gemma2 起引入 Post-Norm 形成"夹心"结构（Pre-Norm → 子层 → Post-Norm → Residual）。这样做的 trade-off 是：增加了计算量（多两次 RMSNorm/层），但显著提升了深层网络的训练稳定性，防止残差流方差爆炸。对于 Gemma 这类追求高质量、大深度的模型，稳定性收益远大于额外 Norm 的开销。

**为什么 Gemma3 用模板类而不是运行时分支？**

`template<bool iswa>` + `if constexpr` 在编译期完全消除了分支，生成两个独立的类实例，零运行时开销。运行时分支虽然灵活，但会在热路径上引入分支预测压力和代码膨胀。对于推理引擎这种性能敏感场景，编译期特化是更优选择。

**AltUp 多通道机制的 trade-off 是什么？**

AltUp 用 n_altup 倍的显存/计算（仅 active 通道计算 Attention/FFN，但存储所有通道），换取类似 Ensemble 的鲁棒性和表达能力。predict-correct 两阶段设计借鉴了卡尔曼滤波/扩散模型的预测-修正思想。代价是实现复杂度大幅上升（形状变换、切片、融合），工程维护成本高。

**Gemma4 共享 KV Cache（has_kv=false）的风险在哪里？**

多层共享 KV Cache 意味着这些层看到的"历史"完全相同，失去了层级化的信息提取能力。适用于：浅层特征提取足够、长上下文场景下内存受限。不适用于：需要深层推理、复杂依赖建模的任务。Google 通过实验确定了最佳的 KV 共享层间隔模式。

**为什么 Gemma4 的 MoE 路由 logits 基于 attn_out 而非 norm 后的 cur？**

attn_out = Attention(x) + x 包含了原始残差信息，未经过 FFN Pre-Norm 的压缩，信息量更大。直接对 attn_out 做 RMSNorm + 缩放 + 线性投影得到 logits，避免了信息瓶颈。这是一个"早路由"的设计选择，trade-off 是路由决策与专家计算的特征分布可能不完全对齐。

## 28.10 动手练习

- 对比 Gemma 各代的 RMSNorm 数量与位置
  - 阅读 `src/models/gemma.cpp`、`src/models/gemma2-iswa.cpp`、`src/models/gemma3.cpp`、`src/models/gemma3n-iswa.cpp`、`src/models/gemma4-iswa.cpp`，对比五份源码中每一层调用 `build_norm()` 的次数与对象。
  - 回答问题：
    1. Gemma 初代每层有几次 RMSNorm？分别是什么？
    2. Gemma2-ISWA 在 Gemma 基础上新增了哪些 post-norm？
    3. Gemma3n 中的 RMSNorm 次数最多，请列出至少 8 处使用 build_norm 的位置（含 altup_router_norm、laurel_post_norm、per_layer_post_norm 等）
    4. Gemma4 中 V 投影后的归一化用的是什么算子？为什么不用 build_norm？
    5. 这五个模型中，attn_q_norm 和 attn_k_norm 何时被引入？

- 追踪 Gemma3n 中 AltUp 多通道的形状变化
  - 阅读 `src/models/gemma3n-iswa.cpp` 第 1-130 行（构造函数），重点关注 `ggml_concat`、`ggml_view_3d`、`view_2d_slice` 等形状操作。
  - 回答问题：
    1. 输入 `inpL` 进入主循环前的最终形状是什么？（三个维度分别代表什么）
    2. `inp_per_layer` 的形状 `[n_embd_altup, n_tokens, n_layer]` 是如何通过 `ggml_permute` 从 `[n_embd_altup, n_layer, n_tokens]` 变换而来的？
    3. `altup_predict` 返回的 predictions 形状是什么？它如何保持与输入 cur 形状一致？
    4. 每层结束时，AltUp 如何把 `[n_embd, n_tokens, n_altup]` 重新融合回 `[n_embd, n_tokens]`？
    5. `view_2d_slice(corrected, 0)` 和 `view_2d_slice(corrected, i_altup_act)` 在末尾的融合中各起什么作用？

- 理解 Gemma4 的 MoE 路由自定义实现
  - 阅读 `src/models/gemma4-iswa.cpp` 第 110-140 行（`is_moe_layer` 分支），重点关注自定义 MoE logits 计算部分。
  - 回答问题：
    1. Gemma4 的 MoE 路由 logits 与常规 LLaMA/Mixtral 有什么不同？计算 logits 时用了哪些张量？
    2. `model.layers[il].ffn_gate_inp_s` 起什么作用？它在 logits 计算的哪一步使用？
    3. 为什么把 `ffn_gate_inp`（路由矩阵）作为参数传入 `build_moe_ffn` 时传 `nullptr`？
    4. `ffn_gate_up_exps`、`ffn_down_exps`、`ffn_down_exps_s` 分别是什么？s 后缀代表什么？
    5. `cur_mlp`（共享专家）和 `cur_moe`（路由专家）是如何被合并的？合并后是否还经过 ffn_post_norm？

- 解析 f_final_logit_softcapping 的数学含义
  - 在 `gemma2-iswa.cpp`、`gemma3.cpp`、`gemma3n-iswa.cpp`、`gemma4-iswa.cpp` 中均有如下三步实现：
    ```
    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
    cur = ggml_tanh(ctx0, cur);
    cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    ```
  - 回答问题：
    1. 这三步操作的复合函数表达式 f(x) 是什么？它的值域是多少？
    2. 与直接 `ggml_clamp(cur, -c, c)` 相比，使用 tanh 软帽有什么优势？
    3. 如果 `f_final_logit_softcapping` 为 30.0，原始 logits 为 ±100，最终会被压缩到什么范围？
    4. Gemma3 中为什么用 `if (hparams.f_final_logit_softcapping)` 判断？什么情况下会跳过软帽？
    5. 在 GBNF 语法约束采样时，logits 软帽会影响哪些采样策略？

- 深入 Gemma3n 的 AltUp 辅助函数
  - 阅读 `src/models/gemma3n-iswa.cpp` 第 200 行以后的辅助函数实现：`calc_magnitude`、`laurel`、`gaussian_topk`、`altup_predict`、`altup_correct`、`altup_compute_router_modalities`。
  - 回答问题：
    1. `calc_magnitude(x)` 的计算流程是什么？它返回的张量形状是什么（输入是 `[n_embd, n_tokens]` 时）？
    2. `laurel(cur, il)` 为什么需要 `laurel_post_norm`？为什么不在 laurel_r 后立即加？
    3. `gaussian_topk(x)` 的阈值 `cutoff_x = mean + f_sparsity_std_mul * std` 是如何计算的？最终激活函数为何选用 ReLU 而不是 GeLU？
    4. `altup_compute_router_modalities` 为什么先经过 `altup_router_norm` 再做 `1/n_embd` 缩放？这与主分支的 RMSNorm 有何不同？
    5. `altup_correct` 中 `ggml_scale_bias(all_coefs, 1.0f, 1.0f)` 的作用是什么？为什么要 +1.0？
    6. `altup_predict` 中 `cur_permuted` 经过 `ggml_permute(ctx0, cur, 1, 2, 0, 3)` 后变成什么形状？为什么需要 ggml_cont？

- 对比 Gemma3n 与 Gemma4 的"每层输入"机制
  - 阅读 `gemma3n-iswa.cpp` 中的 `get_per_layer_inputs()` 和 `project_per_layer_inputs()` 方法，对比 `gemma4-iswa.cpp` 中同名方法的实现差异。
  - 回答问题：
    1. Gemma3n 与 Gemma4 的 `get_per_layer_inputs()` 在视觉路径（`!ubatch.token`）分支有什么共同点和差异？
    2. `per_layer_projection_scale = 1/sqrt(n_embd)` 和 `per_layer_input_scale = 1/sqrt(2)` 这两个缩放因子分别用于什么场景？
    3. Gemma3n 中 `inp_per_layer` 在每层如何与 `corrected` 张量交互？关键算子是哪个？
    4. Gemma4 中 `inp_per_layer` 经过 `per_layer_inp_gate → GELU → mul → per_layer_proj → per_layer_post_norm` 五步，最终如何与残差流相加？
    5. 为什么 Gemma3 模板类没有 per-layer input 机制？

## 28.11 本章小结

这一章中我们学习了 Gemma 家族从初代到 Gemma4 的完整演进脉络。首先我们分析了 Gemma 初代作为标准 Transformer 解码器的基础骨架，确立了 RMSNorm + RoPE + Attention + GELU FFN 的核心范式；其次我们探讨了 Gemma Embedding 如何通过三元判断支持文本/视觉双轨入口，并引入了 Q/K 前置归一化与双重 Post-Norm 夹心结构；接着我们深入了 Gemma2-ISWA 的滑动窗口机制，理解了双层 KV Cache 与按层 RoPE 频率如何实现长上下文的记忆瘦身；然后我们剖析了 Gemma3 模板类通过编译期特化实现 ISWA/标准双模式零开销切换的工程技巧；之后我们详细拆解了 Gemma3n 的 AltUp 多通道架构，包括多通道堆叠、predict-correct 两阶段更新、Laurel 低秩残差、Gaussian TopK 稀疏门控以及多通道融合回单通道的完整流程；最后我们解读了 Gemma4 的极致稀疏化设计：共享 KV Cache、V 投影省略、每层嵌入动态注入、自定义 MoE 路由以及层残差缩放等前沿技术的组合应用。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| llm_build_gemma | Gemma 初代经典解码器，标准 pre-norm Transformer + sqrt(n_embd) 嵌入缩放 |
| llm_build_gemma_embedding | Gemma 嵌入模型，ubatch.token 三元判断支持原始向量输入，引入 Q/K post_norm |
| llm_build_gemma2_iswa | Gemma2 滑动窗口版，build_attn_inp_kv_iswa() + 按层 RoPE + 最终 logit 软帽 |
| llm_build_gemma3<iswa> | Gemma3 模板类，std::conditional_t 编译期分叉 ISWA/标准内存输入 |
| llm_build_gemma3n_iswa | Gemma3n AltUp 多通道，n_altup 个并行通道每层只更新 active 一条，路由投票融合 |
| altup_predict / altup_correct | 多通道先预测后修正的两阶段机制，类似 MoE 的稀疏激活思路 |
| altup_compute_router_modalities | 通过 tanh(Linear(RMSNorm(x))) 计算每个通道的路由权重 |
| laurel | 低秩增量块 tmp = L(R(x)) + x，补充主干的细粒度特征 |
| gaussian_topk | 基于均值+标准差阈值的稀疏门控，f_sparsity_std_mul 控制稀疏度 |
| calc_magnitude | 计算向量幅值 sqrt(sum(x²))，用于多通道幅值归一化 |
| view_2d_slice | 从 3D 张量按第 3 维切片取 2D 视图，多通道架构的常用辅助方法 |
| llm_build_gemma4_iswa | Gemma4 极致稀疏化，共享 KV 层 + V 投影省略 + 每层嵌入注入 + MoE 支持 |
| has_kv(il) | 判断当前层是否为本层计算 KV 的层，复用更早层的 KV Cache |
| rope_freqs | 比例 RoPE 频率因子，全局层（非 SWA）使用以实现更平滑的频率调度 |
| ffn_gate_inp / ffn_gate_up_exps | Gemma4 的 MoE 路由与合并 gate_up 矩阵实现 |
| per_layer_inp_gate → GELU → mul → per_layer_proj | 每层嵌入注入流程：门控 → 激活 → 乘 per-layer 输入 → 投影 → 后归一化 |
| out_scale | 层残差缩放因子，类似 DeepSeek 的残差缩放机制 |
| f_final_logit_softcapping | logits 软帽：scale → tanh → scale 三步平滑输出分布 |
| build_attn_inp_kv_iswa | ISWA 双层 KV 缓存输入接口，将 SWA 与非 SWA 层分开管理 |
| build_attn_inp_no_cache | 非自回归输入（嵌入模型专用），不维护 KV Cache |
| ubatch.token 三元判断 | 决定缩放与归一化策略：原始向量直通 vs 文本 token 标准缩放 |
| ffn_post_norm_1 / ffn_pre_norm_2 / ffn_post_norm_2 | Gemma4 MoE 层中共享专家与路由专家的归一化分工 |
| per_layer_proj_norm / per_layer_post_norm | Gemma3n/4 per-layer input 通路中的投影归一化与输出后归一化 |
| ffn_gate_inp_s | Gemma4 MoE 路由 logits 计算中的 per-channel 缩放矩阵 |
| altup_correct_scale / altup_unembd_proj | AltUp 修正阶段的 active 通道缩放因子与多通道融合投影矩阵 |
| n_embd_altup / n_altup / i_altup_act | AltUp 三超参数：单通道嵌入维度、并行通道数、每层更新的 active 通道索引 |

下一章中，我们将学习 Mamba 与状态空间模型 —— Transformer 的"挑战者"，理解 SSM 如何在 llama.cpp 中通过专用基类与混合架构落地。
