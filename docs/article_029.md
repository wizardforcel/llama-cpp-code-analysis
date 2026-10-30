


# 第29章 Mamba 与混合架构 —— SSM 与 Attention 的协同进化

## 学习目标

- 理解 Mamba/Mamba2 SSM 循环层的两段式构建流水线（Conv1D + SSM 扫描 + 门控融合）
- 掌握 build_mamba_layer 与 build_mamba2_layer 在状态管理、维度布局和分组 RMS Norm 上的差异
- 理解混合架构（Hybrid）中三种 SSM+Attention 组合策略：按层切换（Jamba）、并行融合（Falcon-H1）、三分支（Nemotron-H）
- 掌握混合输入管理：build_rs_inp 与 build_inp_mem_hybrid 如何分别管理纯循环与混合状态
- 理解模板类显式实例化（如 llm_build_plamo3<iswa>）的 C++ 工程实践与 ISWA 复用机制
- 了解 PLaMo 系列预归一化（attn_q_norm/attn_k_norm）以及后归一化（attn_post_norm/ffn_post_norm）的设计哲学
- 掌握 Granite-Hybrid 的残差缩放（f_residual_scale）和 logits 缩放（f_logit_scale）等架构特性

## 29.1 生活类比

想象 SSM 与 Attention 的混合架构是一场**“特种部队的协同作战”**：**Mamba（SSM 循环层）** = 高速公路巡逻队，擅长长距离快速扫描，对历史信息有“记忆缓存”；**Attention（注意力层）** = 精准狙击手，能锁定任意位置的关键信息，但每次行动成本较高；**Jamba（按层切换）** = 班长调度员，按地形决定派哪支队伍上场（SSM 层 / Attention 层交替）；**Falcon-H1（并行融合）** = 双警组同时出动，一队巡逻一队狙击，情报汇总后联合决策；**Nemotron-H（三分支）** = 三军种联合作战，可全 SSM、全 Attention 或纯 FFN 任意搭配；**Granite-Hybrid** = 强调“作战日志标准化”的部队，所有行动都按统一的残差缩放比例记录；**PLaMo2/PLaMo3** = 强调“前后检查站”的精锐部队，每个动作前后都有独立的安全检查（post-norm）；**状态缓存（conv state / ssm state）** = 巡逻队的“交接班记录本”，下一班直接继承上一班的工作；**get_ssm_rows lambda** = “按需调阅”机制——只调取未修改部分，避免重复劳动；**ISWA（Intermittent Sliding Window Attention）** = 滑动窗口哨兵，重要路段加强巡逻，普通路段常规处理。

## 29.2 源码地图

```
src/models/mamba-base.cpp
├── llm_build_mamba_base::llm_build_mamba_base()       # 基类构造（空实现，仅传递参数）
├── build_mamba_layer()                    # Mamba1 三段式构建（Conv1D + SSM + 门控）
│   ├── ssm_dt_b_c_rms 条件分支           # 部分变体（FalconMamba）需要对 B/C/dt 做 RMS Norm
│   ├── build_rs 获取 conv state 缓存     # Conv1D 状态管理
│   ├── ggml_ssm_conv 完成 1D 卷积        # 卷积阶段
│   ├── ggml_ssm_scan 并行关联扫描        # SSM 核心算子
│   └── ggml_swiglu_split 门控融合        # z 与 y 融合
├── build_mamba2_layer()                   # Mamba2 多头分组实现
│   ├── n_head = ssm_dt_rank              # Mamba2 多头设计
│   ├── 一次性投出 z/xBC/dt              # 三路张量
│   ├── 分组 RMS Norm (n_group)           # 多组 SSM 稳定性
│   └── ssm_norm 归一化                   # 组级归一化权重

src/models/mamba.cpp
├── llm_build_mamba::llm_build_mamba()
│   ├── build_inp_embd 获取嵌入输入
│   ├── build_rs_inp 获取循环输入上下文
│   ├── model.arch 分发（MAMBA2 vs MAMBA）
│   └── build_norm + build_lora_mm 输出层

src/models/jamba.cpp
├── llm_build_jamba::llm_build_jamba()
│   ├── build_inp_mem_hybrid 同时管理 KV 与循环状态
│   ├── n_head_kv == 0 判断 SSM/Attention 层
│   ├── build_attn 时 NULL RoPE（位置无关）
│   ├── ffn_gate_inp 区分 FFN/MoE 分支
│   └── LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX 门控

src/models/falcon-h1.cpp
├── llm_build_falcon_h1::llm_build_falcon_h1()
│   ├── build_attn 与 build_mamba2_layer 并行调用
│   ├── ggml_rope_ext 完整 RoPE 配置（n_rot/freq_base/ext_factor）
│   ├── kq_scale 双重来源（f_attention_scale）
│   └── Aggregation：ggml_add(attn_out, ssm_out)

src/models/nemotron-h.cpp
├── llm_build_nemotron_h::llm_build_nemotron_h()
│   ├── is_recurrent/n_ff 三分支判断
│   └── GGML_ASSERT(n_embd_head == n_embd_head_k)
├── build_attention_layer()                 # 标准 Attention + 可选 bias
│   ├── Q/K/V 可选加 bq/bk/bv
│   └── kq_scale 自定义缩放
└── build_ffn_layer()                       # MoE + 共享专家
    ├── LLM_FFN_RELU_SQR 激活（Nemotron 特色）
    ├── ffn_latent_down/up 潜空间压缩
    ├── LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID 门控
    └── ffn_shexp 共享专家分支

src/models/granite-hybrid.cpp
├── llm_build_granite_hybrid::llm_build_granite_hybrid()
│   ├── rope_finetuned 条件启用 RoPE
│   └── f_logit_scale 输出 logits 缩放
├── build_attention_layer()                 # 条件 RoPE
│   ├── get_rope_factors 取得 YaRN 缩放因子
│   └── ggml_rope_ext 注入 rope_factors
└── build_layer_ffn()                       # Granite 残差缩放 + MoE + 共享专家
    ├── f_residual_scale 残差缩放
    ├── MoE + shexp 共享专家（n_ff_shexp）
    └── 输出前再次缩放残差

src/models/plamo2.cpp
├── llm_build_plamo2::llm_build_plamo2()
│   ├── pre/post-mixer-norm 双归一化结构
│   ├── build_plamo2_attn_layer 调用
│   └── build_plamo2_mamba_layer 调用
├── build_plamo2_attn_layer()               # QKV 拆分 + Q/K 独立 RMS Norm
│   ├── ggml_view_3d 拆分 qkv 联合张量
│   ├── attn_q_norm/attn_k_norm 预归一化
│   └── norm → RoPE → build_attn 顺序
└── build_plamo2_mamba_layer()              # 定制化 Mamba2 变体
    ├── zx permute+cont_4d 重新布局
    ├── dt_dim = max(64, n_embd/16)        # 动态 dt 维度
    ├── B/C/dt 三者都做 RMS Norm
    ├── get_ssm_rows 复用 ssm_scan 逻辑
    └── y + D*x + swiglu_split(z, y) 门控融合

src/models/plamo3.cpp
├── template <bool iswa>
├── llm_build_plamo3<iswa>::llm_build_plamo3()
│   ├── std::conditional_t 类型分支
│   ├── build_attn_inp_kv_iswa() vs build_attn_inp_kv()
│   ├── 每层差异化 RoPE 频率（get_rope_freq_base）
│   ├── attn_q_norm/attn_k_norm 预归一化
│   └── 双残差 + 后归一化结构
└── template struct llm_build_plamo3<false>;
  template struct llm_build_plamo3<true>;  # 显式实例化
```

## 29.3 Mamba 基础构建器 —— SSM 循环层的"万能工厂"

**`llm_build_mamba_base` 的角色：Mamba 系列的"共同基因"**

作为 Mamba/Mamba2/Jamba/Falcon-H1 等模型的共同基类，它提供了 SSM 层的核心构建方法。通过 `build_mamba_layer` 和 `build_mamba2_layer` 两个关键方法，支持两代 SSM 架构的差异。内部通过 `hparams`（模型超参数）、`ctx0`（ggml 计算图）、`gf`（前向图）、`inp`（输入）来完成构图。

### 29.3.1 llm_build_mamba_base::llm_build_mamba_base()：基类构造与参数透传

**源码路径**：`src/models/mamba-base.cpp` - `llm_build_mamba_base`（第1-5行）

这段代码定义了 `llm_build_mamba_base` 的构造函数，它仅将传入的 `llm_graph_params` 参数透传给父类 `llm_graph_context` 的构造函数。没有额外的成员变量初始化，体现了“基类只做骨架”的设计哲学：所有超参数、张量指针、后端缓冲等上下文数据均由父类统一持有，子类通过继承直接访问。

```cpp
// 类型即图纸，先理解数据结构
llm_build_mamba_base::llm_build_mamba_base(const llm_graph_params & params) : llm_graph_context(params) {}
```

### 29.3.2 build_mamba_layer()：Mamba1 三段式流水线（Conv1D + SSM 扫描 + 门控融合）

**源码路径**：`src/models/mamba-base.cpp` - `llm_build_mamba_base::build_mamba_layer()`（第7-106行）

这段代码实现了 Mamba1 核心层的三段式计算流水线：**输入投影 → Conv1D 卷积 → SSM 并行扫描 → 门控融合 → 输出投影**。关键设计点包括：
1. **状态管理**：通过 `build_rs` 配合 `get_ssm_rows` lambda 实现“按需读取、精确写回”状态缓存，`kv_head` 定位当前序列槽位，避免全量拷贝。
2. **条件归一化**：`ssm_dt_b_c_rms` 标志或显式存在 `ssm_dt_norm/b_norm/c_norm` 权重时，对 dt/B/C 做 RMS Norm，兼容 FalconMamba/Jamba 等变体。
3. **门控融合**：`ggml_swiglu_split(z, y)` 是 Mamba 系列标志性输出方式，将投影分支 z 作为门控，与 SSM 输出 y 融合。

```cpp
// 逐行注释解释
ggml_tensor * llm_build_mamba_base::build_mamba_layer(llm_graph_input_rs * inp,
                                                      ggml_tensor *        cur,
                                                      const llama_model &  model,
                                                      const llama_ubatch & ubatch,
                                                      int                  il) {
    const auto * mctx_cur = inp->mctx;                                      // 获取当前层的内存上下文
    const auto kv_head = mctx_cur->get_head();                              // 获取当前序列在状态缓存中的头位置（用于写回状态）
    const auto & layer = model.layers[il];                                  // 取出当前层的权重结构体
    const int64_t d_conv         = hparams.ssm_d_conv;                      // 卷积核宽度
    const int64_t d_inner        = hparams.ssm_d_inner;                     // 内部投影维度（扩展维度）
    const int64_t d_state        = hparams.ssm_d_state;                     // 状态空间维度
    const int64_t dt_rank        = hparams.ssm_dt_rank;                     // 时间步秩
    const int64_t n_head         = d_inner;                                 // Mamba1：头数等于 d_inner，即每个通道一个头
    const int64_t head_dim       = 1;                                       // 头维度为 1（标量 SSM）
    const int64_t n_seqs         = ubatch.n_seqs;                           // 当前批次的序列数
    const bool    ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;                  // 是否对 dt/B/C 做 RMS Norm（FalconMamba/Jamba 需要）
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;                       // 每个序列的 token 数
    GGML_ASSERT(n_seqs != 0);                                               // 断言：必须有序列
    GGML_ASSERT(ubatch.equal_seqs());                                       // 断言：所有序列长度相等（批处理要求）
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);                  // 断言：总 token 数一致
    GGML_ASSERT(d_inner % n_head == 0);                                     // 断言：d_inner 能被 n_head 整除
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);                  // 获取卷积状态缓存 [n_layer, ...]
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);                  // 获取 SSM 状态缓存 [n_layer, ...]
    ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs); // 从缓存读取/初始化卷积状态
    conv               = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);   // 重塑为 (d_conv-1, d_inner, n_seqs)
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);   // 输入重塑为 (n_embd, n_seq_tokens, n_seqs)
    ggml_tensor * xz = build_lora_mm(layer.ssm_in, cur, layer.ssm_in_s);  // 输入投影：得到 x 和 z 两路 (2*d_inner, ...)
    ggml_tensor * x  = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0); // 切片取前 d_inner 为 x
    ggml_tensor * z = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner * ggml_element_size(xz)); // 切片取后 d_inner 为 z
    { // Conv1D 阶段
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0); // 拼接历史状态与当前输入 (d_conv-1 + n_seq_tokens, d_inner, n_seqs)
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2], n_seq_tokens * (conv_x->nb[0])); // 取最后 d_conv-1 步作为新状态
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv, ggml_view_1d(ctx0, conv_states_all, (d_conv - 1) * (d_inner) * (n_seqs), kv_head * (d_conv - 1) * (d_inner) * ggml_element_size(conv_states_all)))); // 写回卷积状态缓存（按 kv_head 定位序列槽位）
        x = ggml_ssm_conv(ctx0, conv_x, layer.ssm_conv1d);                // 执行 1D 卷积
        x = ggml_add(ctx0, x, layer.ssm_conv1d_b);                        // 加卷积偏置
        x = ggml_silu(ctx0, x);                                           // SiLU 激活
    }
    { // SSM 阶段
        ggml_tensor * x_db = build_lora_mm(layer.ssm_x, x);               // x 投影得到 dt、B、C 拼接张量 (dt_rank + 2*d_state, ...)
        ggml_tensor * dt   = ggml_view_3d(ctx0, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0); // 切片取 dt
        ggml_tensor * B = ggml_view_4d(ctx0, x_db, d_state, 1, n_seq_tokens, n_seqs, d_state * x_db->nb[0], x_db->nb[1], x_db->nb[2], ggml_element_size(x_db) * dt_rank); // 切片取 B (d_state, 1, n_seq_tokens, n_seqs)
        ggml_tensor * C = ggml_view_4d(ctx0, x_db, d_state, 1, n_seq_tokens, n_seqs, d_state * x_db->nb[0], x_db->nb[1], x_db->nb[2], ggml_element_size(x_db) * (dt_rank + d_state)); // 切片取 C
        if (ssm_dt_b_c_rms || (layer.ssm_dt_norm && layer.ssm_b_norm && layer.ssm_c_norm)) { // 条件归一化：部分变体要求对 dt/B/C 做 RMS Norm
            dt = build_norm(dt, layer.ssm_dt_norm, NULL, LLM_NORM_RMS, il);
            B  = build_norm(B, layer.ssm_b_norm, NULL, LLM_NORM_RMS, il);
            C  = build_norm(C, layer.ssm_c_norm, NULL, LLM_NORM_RMS, il);
        }
        dt = build_lora_mm(layer.ssm_dt, dt);                             // dt 投影：(dt_rank, ...) -> (d_inner, ...)
        dt = ggml_add(ctx0, dt, layer.ssm_dt_b);                          // 加 dt 偏置
        cur = x;                                                          // 保存原始 x 用于后续跳跃连接
        x   = ggml_reshape_4d(ctx0, x, head_dim, n_head, n_seq_tokens, n_seqs); // 重塑为 4D (1, d_inner, n_seq_tokens, n_seqs)
        ggml_tensor * A = layer.ssm_a;                                    // 取出 A 矩阵 (log 空间的状态衰减参数)
        auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) { // Lambda：按需获取 SSM 状态行
            ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());
            return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);          // 调用核心 SSM 并行扫描算子
        };
        ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows); // 执行 SSM 扫描并读取/写入状态
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_seqs, x->nb[3] * x->ne[3]), ggml_view_1d(ctx0, ssm_states_all, d_state * d_inner * n_seqs, kv_head * d_state * d_inner * ggml_element_size(ssm_states_all)))); // 写回 SSM 状态缓存
        ggml_tensor * y = ggml_view_3d(ctx0, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[2], x->nb[3], 0); // 取出扫描输出 y
        y = ggml_add(ctx0, y, ggml_mul(ctx0, cur, layer.ssm_d));          // 跳跃连接：y += x * D
        y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);               // 门控融合：z (gate) 与 y 融合
        cur = build_lora_mm(layer.ssm_out, y, layer.ssm_out_s);           // 输出投影回 n_embd
    }
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);  // 恢复为 (n_embd, n_tokens)
    return cur;
}
```

### 29.3.3 build_mamba2_layer()：Mamba2 多头分组实现

**源码路径**：`src/models/mamba-base.cpp` - `llm_build_mamba_base::build_mamba2_layer()`（第108-218行）

这段代码实现了 Mamba2 的核心层构建，与 Mamba1 的核心差异在于：
1. **多头设计**：`n_head = ssm_dt_rank`，`head_dim = d_inner / n_head`，将 SSM 扩展为多头并行。
2. **一次性投影**：`zxBCdt` 一次性产出 z、xBC、dt 三路张量，dt 直接作为头维度，无需再经过 `ssm_dt` 投影矩阵。
3. **分组 RMS Norm**：引入 `n_group` 分组，对 `y` 重塑为 `(d_inner/n_group, n_group, ...)` 后做归一化，稳定多组 SSM 训练。
4. **状态形状变化**：卷积状态包含 `xBC`（维度 `d_inner + 2*n_group*d_state`），SSM 状态按头分组管理。

```cpp
// 逐行注释解释
ggml_tensor * llm_build_mamba_base::build_mamba2_layer(llm_graph_input_rs * inp,
                                                        ggml_tensor *        cur,
                                                        const llama_model &  model,
                                                        const llama_ubatch & ubatch,
                                                        int                  il) const {
    const auto * mctx_cur = inp->mctx;                                    // 获取内存上下文
    const auto kv_head = mctx_cur->get_head();                            // 当前序列头位置
    const int64_t d_conv   = hparams.ssm_d_conv;                          // 卷积核宽度
    const int64_t d_inner  = hparams.ssm_d_inner;                         // 内部维度
    const int64_t d_state  = hparams.ssm_d_state;                         // 状态维度
    const int64_t n_head   = hparams.ssm_dt_rank;                         // Mamba2 关键差异：头数 = dt_rank（多头设计）
    const int64_t head_dim = d_inner / n_head;                            // 头维度 = d_inner / n_head
    const int64_t n_group  = hparams.ssm_n_group;                         // 分组数（用于分组 RMS Norm）
    const int64_t n_seqs   = ubatch.n_seqs;                               // 序列数
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;                     // 每序列 token 数
    GGML_ASSERT(n_seqs != 0);                                             // 基础断言
    GGML_ASSERT(ubatch.equal_seqs());                                     // 
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);                // 
    GGML_ASSERT(d_inner % n_head  == 0);                                  // d_inner 能被头数整除
    GGML_ASSERT(d_inner % d_state == 0);                                  // d_inner 能被状态维度整除
    GGML_ASSERT(d_inner % n_group == 0);                                  // d_inner 能被分组数整除
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);                // 卷积状态缓存
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);                // SSM 状态缓存
    ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs); // 读取卷积状态
    conv               = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs); // 卷积状态包含 xBC
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);  // 输入重塑
    ggml_tensor * zxBCdt = build_lora_mm(model.layers[il].ssm_in, cur, model.layers[il].ssm_in_s); // 一次性投影：z + xBC + dt
    ggml_tensor * z   = ggml_view_4d(ctx0, zxBCdt, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * zxBCdt->nb[0], zxBCdt->nb[1], zxBCdt->nb[2], 0); // 切片 z
    ggml_tensor * xBC = ggml_view_3d(ctx0, zxBCdt, d_inner + 2 * n_group * d_state, n_seq_tokens, n_seqs, zxBCdt->nb[1], zxBCdt->nb[2], d_inner * ggml_element_size(zxBCdt)); // 切片 xBC
    ggml_tensor * dt  = ggml_view_3d(ctx0, zxBCdt, n_head, n_seq_tokens, n_seqs, zxBCdt->nb[1], zxBCdt->nb[2], (2 * d_inner + 2 * n_group * d_state) * ggml_element_size(zxBCdt)); // 切片 dt
    { // Conv1D 阶段
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0); // 拼接历史与当前
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs, conv_x->nb[1], conv_x->nb[2], n_seq_tokens * (conv_x->nb[0])); // 取新状态
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv, ggml_view_1d(ctx0, conv_states_all, (d_conv - 1) * (d_inner + 2 * n_group * d_state) * (n_seqs), kv_head * (d_conv - 1) * (d_inner + 2 * n_group * d_state) * ggml_element_size(conv_states_all)))); // 写回卷积状态
        xBC = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);   // 1D 卷积
        xBC = ggml_add(ctx0, xBC, model.layers[il].ssm_conv1d_b);         // 加偏置
        xBC = ggml_silu(ctx0, xBC);                                       // SiLU
    }
    { // SSM 阶段
        ggml_tensor * x = ggml_view_4d(ctx0, xBC, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * xBC->nb[0], xBC->nb[1], xBC->nb[2], 0); // 取出 x (head_dim, n_head, ...)
        ggml_tensor * B = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0], xBC->nb[1], xBC->nb[2], d_inner * ggml_element_size(xBC)); // B: (d_state, n_group, ...)
        ggml_tensor * C = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0], xBC->nb[1], xBC->nb[2], (d_inner + n_group * d_state) * ggml_element_size(xBC)); // C: (d_state, n_group, ...)
        dt = ggml_add(ctx0, ggml_cont(ctx0, dt), model.layers[il].ssm_dt_b); // dt 加偏置（无需再投影，dt_rank == n_head）
        ggml_tensor * A = model.layers[il].ssm_a;                         // A 矩阵
        auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) { // 复用 SSM 扫描 lambda
            ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());
            return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
        };
        ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows); // 执行扫描
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_seqs, ggml_nelements(x) * x->nb[0]), ggml_view_1d(ctx0, ssm_states_all, d_state * d_inner * n_seqs, kv_head * d_state * d_inner * ggml_element_size(ssm_states_all)))); // 写回状态
        ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs, x->nb[1], n_head * x->nb[1], n_seq_tokens * n_head * x->nb[1], 0); // 取出 y
        y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d)); // 跳跃连接 y += x * D
        cb(y, "mamba2_y_add_d", il);
        y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);               // 门控融合（与 Mamba1 一致）
        if (model.layers[il].ssm_norm) {                                  // Mamba2 独有：分组 RMS Norm
            y = ggml_reshape_4d(ctx0, y, d_inner / n_group, n_group, n_seq_tokens, n_seqs);
            y = build_norm(y, model.layers[il].ssm_norm, NULL, LLM_NORM_RMS, il);
        }
        y = ggml_reshape_3d(ctx0, y, d_inner, n_seq_tokens, n_seqs);      // 恢复形状
        cur = build_lora_mm(model.layers[il].ssm_out, y, model.layers[il].ssm_out_s); // 输出投影
    }
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);  // 恢复 2D
    cb(cur, "mamba_out", il);
    return cur;
}
```

## 29.4 Mamba 与 Mamba2 模型 —— 纯 SSM 的"独行者"

**`llm_build_mamba` 的极简结构：循环层的"标准化样板"**

与 Transformer 最大的不同：循环层没有 QKV、RoPE 和 Attention 计算。每一层调用一次 `build_mamba_layer` 或 `build_mamba2_layer`，由 `model.arch` 判断属于哪种 SSM。通过 `rs_inp = build_rs_inp()` 获取循环输入上下文，再交给基类的 layer 构建器。

### 29.4.1 llm_build_mamba::llm_build_mamba()：纯 SSM 模型主构建流程

**源码路径**：`src/models/mamba.cpp` - `llm_build_mamba::llm_build_mamba()`（第1-45行）

这段代码展示了纯 SSM 模型的标准化构建流程：**嵌入 → 循环层堆叠（含 Norm、SSM、残差） → 最终 Norm → LM Head**。关键点：
- 使用 `build_rs_inp()` 而非 `build_inp_mem_hybrid()`，因为纯 SSM 不需要 KV Cache。
- `model.arch` 运行时分发到 `build_mamba_layer` 或 `build_mamba2_layer`，体现“基类提供两种实现，子类按需选择”的多态思想。
- `inp_out_ids` 优化：最后一层只提取参与采样的 token 行，节省 logits 计算量。

```cpp
// 逐行注释解释
llm_build_mamba::llm_build_mamba(const llama_model & model, const llm_graph_params & params) : llm_build_mamba_base(params) {
    ggml_tensor * cur;                                                  // 当前层张量指针
    ggml_tensor * inpL;                                                 // 上一层输出（残差基线）
    inpL = build_inp_embd(model.tok_embd);                              // Token 嵌入查表
    auto * rs_inp = build_rs_inp();                                     // 获取纯循环输入上下文（仅 SSM 状态）
    ggml_tensor * inp_out_ids = build_inp_out_ids();                    // 最后 token 采样用的行索引
    for (int il = 0; il < n_layer; ++il) {                              // 遍历所有层
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il); // RMS Norm
        cb(cur, "attn_norm", il);
        if (model.arch == LLM_ARCH_MAMBA2) {                            // 架构分发：Mamba2 vs Mamba1
            cur = build_mamba2_layer(rs_inp, cur, model, ubatch, il);
        } else {
            cur = build_mamba_layer(rs_inp, cur, model, ubatch, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {                         // 最后一层：只取需要的输出行
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = ggml_add(ctx0, cur, inpL);                                // 残差连接
        cur = build_cvec(cur, il);                                      // 控制向量注入（可选）
        cb(cur, "l_out", il);
        inpL = cur;                                                     // 更新残差基线
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);   // 最终归一化
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);                             // LM Head 投影
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);                                 // 标记图输出
}
```

## 29.5 Jamba 混合架构 —— Mamba 与 Attention 的"交叉跑道"

**`llm_build_jamba` 的核心设计：按层类型切换路径**

通过 `hparams.n_head_kv(il) == 0` 判断该层是否是 Mamba 层。`n_head_kv == 0` 表示该层没有 KV 头，即为纯 SSM 层。否则为标准 Self-Attention 层，执行 QKV 投影 + `build_attn`。

### 29.5.1 llm_build_jamba::llm_build_jamba()：按层切换的混合构建

**源码路径**：`src/models/jamba.cpp` - `llm_build_jamba::llm_build_jamba()`（第5-95行）

这段代码实现了 Jamba 的**按层切换**混合策略：
1. **混合输入管理**：`build_inp_mem_hybrid()` 同时提供 `get_recr()`（SSM 状态）和 `get_attn()`（KV Cache）。
2. **层类型判断**：`n_head_kv == 0` 作为 Mamba 层的标志，无需额外配置字段。
3. **无 RoPE Attention**：`build_attn` 所有位置编码参数传 `NULL`，体现 Jamba “位置无关”注意力设计。
4. **双残差结构**：先 `attn/ssm` 残差，再 `ffn` 残差，符合标准 Transformer 块结构。

```cpp
// 逐行注释解释
llm_build_jamba::llm_build_jamba(const llama_model & model, const llm_graph_params & params) : llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();                // V 头维度
    ggml_tensor * cur;                                                  // 当前张量
    ggml_tensor * inpL;                                                 // 残差基线
    inpL = build_inp_embd(model.tok_embd);                              // 嵌入层
    auto * inp_hybrid = build_inp_mem_hybrid();                         // 混合输入：同时管理 KV Cache 与 SSM 状态
    ggml_tensor * inp_out_ids = build_inp_out_ids();                    // 采样行索引
    for (int il = 0; il < n_layer; ++il) {                              // 遍历层
        const int64_t n_head_kv = hparams.n_head_kv(il);                // 判断层类型：0 = Mamba，>0 = Attention
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        if (n_head_kv == 0) {                                           // Mamba 分支
            cur = build_mamba_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);
        } else {                                                        // Attention 分支
            struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Qcur, "Qcur", il); cb(Kcur, "Kcur", il); cb(Vcur, "Vcur", il);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            cb(Qcur, "Qcur", il); cb(Kcur, "Kcur", il); cb(Vcur, "Vcur", il);
            // No RoPE :)                                                  // Jamba Attention 无 RoPE
            cur = build_attn(inp_hybrid->get_attn(),                    // 标准注意力计算
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, NULL, NULL, NULL, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {                         // 最后一层行采样
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, inpL, cur);       // Attention/SSM 残差
        cb(cur, "ffn_inp", il);
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il); // FFN Norm
        cb(cur, "ffn_norm", il);
        if (model.layers[il].ffn_gate_inp == nullptr) {                 // FFN vs MoE 分支
            cur = build_ffn(cur,                                        // 标准 FFN (SwiGLU + PAR)
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {                                                        // MoE 分支
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, false,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);         // Softmax 门控
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, ffn_inp, cur);                             // FFN 残差
        cur = build_cvec(cur, il);                                      // 控制向量
        cb(cur, "l_out", il);
        inpL = cur;                                                     // 更新基线
    }
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);  // 最终 Norm
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);                             // LM Head
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
```

## 29.6 Falcon-H1 混合架构 —— Mamba2 + Attention 的"分支汇流"

**`llm_build_falcon_h1` 的核心设计：每层并行 Attention 与 Mamba2**

与 Jamba 按层切换不同，Falcon-H1 每层同时计算 Attention 和 Mamba2。两者结果通过 `ggml_add` 加和，再经过 FFN 输出。这种“并行混合”在保持 SSM 线性复杂度的同时，提供了 Attention 的精确位置建模能力。

### 29.6.1 llm_build_falcon_h1::llm_build_falcon_h1()：并行融合的层构建

**源码路径**：`src/models/falcon-h1.cpp` - `llm_build_falcon_h1::llm_build_falcon_h1()`（第1-107行）

以下是核心逻辑的逐行解析（节选关键段落）：

```cpp
// 关键段落：并行分支计算与聚合
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * inpSA = inpL;
    cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "attn_norm", il);

    // self-attention 分支
    ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    // ... reshape + RoPE (完整配置) ...
    ggml_tensor * attn_out = build_attn(inp->get_attn(),
                                model.layers[il].wo, NULL,
                                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

    // Mamba2 分支
    cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il); // 复用同一 Norm
    cb(cur, "ssm_in", il);
    ggml_tensor * ssm_out = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);

    // Aggregation：两路输出相加
    cur   = ggml_add(ctx0, attn_out, ssm_out);
    inpSA = ggml_add(ctx0, cur, inpSA);  // 残差连接

    // FFN 分支（带 bias）
    cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
    cur = build_ffn(cur,
            model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
            model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
            model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cur = ggml_add(ctx0, cur, inpSA);
}
```

**设计要点对比**：

| 特性 | Jamba (按层切换) | Falcon-H1 (并行融合) |
|------|------------------|----------------------|
| **层内结构** | 非此即彼：要么 SSM 要么 Attn | 双分支并行：Attn + Mamba2 同时算 |
| **RoPE** | Attention 无 RoPE | Attention 完整 RoPE 配置 |
| **聚合方式** | 无（路径互斥） | `ggml_add(attn_out, ssm_out)` 逐元素相加 |
| **Norm 复用** | 每分支独立 Norm | Attention 与 Mamba2 共享同一 `attn_norm` |
| **FFN Bias** | 无 | 有 (`ffn_up_b`, `ffn_gate_b`, `ffn_down_b`) |

## 29.7 Nemotron-H 混合架构 —— 「按层类型」的三路分支

**`llm_build_nemotron_h` 的三分支设计**

通过 `hparams.is_recurrent(il)` 判断该层是否是循环层（Mamba2）。否则 `hparams.n_ff(il) == 0` 时为纯 Attention 层。否则为 FFN 层（包含普通 FFN 和 MoE FFN）。这种设计允许模型在同一架构中混用三种层类型。

### 29.7.1 llm_build_nemotron_h::llm_build_nemotron_h()：三分支主循环

**源码路径**：`src/models/nemotron-h.cpp` - `llm_build_nemotron_h::llm_build_nemotron_h()`（第1-49行）

```cpp
// 逐行注释解释
llm_build_nemotron_h::llm_build_nemotron_h(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());  // 硬约束：Q/K 头维度必须相等
    ggml_tensor * cur;
    ggml_tensor * inpL;
    inpL = build_inp_embd(model.tok_embd);
    ggml_build_forward_expand(gf, inpL);
    auto * inp = build_inp_mem_hybrid();                  // 混合输入
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        if (hparams.is_recurrent(il)) {                   // 分支 1：SSM 层
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        } else if (hparams.n_ff(il) == 0) {               // 分支 2：纯 Attention 层
            cur = build_attention_layer(cur, inp->get_attn(), model, n_embd_head, il);
        } else {                                          // 分支 3：FFN 层（含 MoE）
            cur = build_ffn_layer(cur, model, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        cur = ggml_add(ctx0, cur, inpSA);                 // 统一残差
        cb(cur, "nemotron_h_block_out", il);
        inpL = cur;
    }
    // ... 最终 Norm + LM Head ...
}
```

### 29.7.2 build_attention_layer()：标准 Attention + 可选 Bias

**源码路径**：`src/models/nemotron-h.cpp` - `llm_build_nemotron_h::build_attention_layer()`（第50-93行）

```cpp
// 逐行注释解释
ggml_tensor * llm_build_nemotron_h::build_attention_layer(ggml_tensor *             cur,
                                                          llm_graph_input_attn_kv * inp_attn,
                                                          const llama_model &       model,
                                                                int64_t             n_embd_head,
                                                                int                 il) {
    ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);       // Q 投影
    if (model.layers[il].bq) { Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); } // 可选 Q bias
    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);       // K 投影
    if (model.layers[il].bk) { Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk); } // 可选 K bias
    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);       // V 投影
    if (model.layers[il].bv) { Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv); } // 可选 V bias
    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il), n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale; // 自定义缩放
    cur = build_attn(inp_attn, model.layers[il].wo, model.layers[il].bo, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il); // 输出投影 wo + 可选 bo
    cb(cur, "attn_out", il);
    return cur;
}
```

### 29.7.3 build_ffn_layer()：MoE + 共享专家的 Nemotron 特色

**源码路径**：`src/models/nemotron-h.cpp` - `llm_build_nemotron_h::build_ffn_layer()`（第94-144行）

```cpp
// 逐行注释解释
ggml_tensor * llm_build_nemotron_h::build_ffn_layer(ggml_tensor * cur, const llama_model & model, int il) {
    if (model.layers[il].ffn_gate_inp == nullptr) {                     // 普通 FFN 分支
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                NULL,                      NULL,                        NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);                     // Nemotron 特色：ReLU 平方激活
        cb(cur, "ffn_out", il);
    } else {                                                            // MoE 分支
        ggml_tensor * inp_emb    = cur;
        ggml_tensor * inp_latent = cur;
        if (model.layers[il].ffn_latent_down) {                         // 潜空间压缩
            inp_latent = ggml_mul_mat(ctx0, model.layers[il].ffn_latent_down, cur);
        }
        ggml_tensor * router_logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur); // 路由 logits
        cb(router_logits, "ffn_moe_logits", il);
        ggml_tensor * moe_out = build_moe_ffn(inp_latent,               // MoE 计算
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                nullptr, // no gate
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,                       // 专家概率偏置
                n_expert, n_expert_used,
                LLM_FFN_RELU_SQR, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,                  // Sigmoid 门控（区别于 Softmax）
                il,
                router_logits, nullptr,
                model.layers[il].ffn_up_exps_s,
                nullptr,
                model.layers[il].ffn_down_exps_s);
        cb(moe_out, "ffn_moe_out", il);
        if (model.layers[il].ffn_latent_up) {                           // 潜空间还原
            moe_out = ggml_mul_mat(ctx0, model.layers[il].ffn_latent_up, moe_out);
        }
        ggml_tensor * ffn_shexp = build_ffn(inp_emb,                    // 共享专家分支
                    model.layers[il].ffn_up_shexp,   NULL, model.layers[il].ffn_up_shexp_s,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);
        cur = ggml_add(ctx0, moe_out, ffn_shexp);                       // MoE + 共享专家融合
        cb(cur, "ffn_out", il);
    }
    cur = build_cvec(cur, il);
    cb(cur, "l_out", il);
    return cur;
}
```

**Nemotron-H 关键创新点**：

| 特性 | 说明 |
|------|------|
| **三分支层类型** | `is_recurrent` / `n_ff==0` / `else` 实现 SSM / Attention / FFN 三种层任意排布 |
| **ReLU 平方激活** | `LLM_FFN_RELU_SQR` 替代标准 SwiGLU，论文证明有助于训练稳定性 |
| **Sigmoid 门控** | `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` 使专家选择更稀疏、可解释 |
| **共享专家** | `ffn_shexp` 独立于路由专家，保证基础表达能力 |
| **潜空间压缩** | `ffn_latent_down/up` 在路由前压缩维度，降低 MoE 计算量 |

## 29.8 Granite-Hybrid 混合架构 —— IBM 的"Mamba2 + 标准注意力 + MoE"

**`llm_build_granite_hybrid` 的混合模式**

与 Nemotron-H 类似的二分结构：`is_recurrent(il)` 判断是 Mamba2 还是 Attention。但 MoE 在 FFN 中处理，不与 SSM 层耦合。这种“线性层（SSM/Attn） + FFN（含 MoE）”的二段设计更接近标准 Transformer 风格。

### 29.8.1 llm_build_granite_hybrid::llm_build_granite_hybrid()：主构建流程

**源码路径**：`src/models/granite-hybrid.cpp` - `llm_build_granite_hybrid::llm_build_granite_hybrid()`（第3-38行）

```cpp
// 关键逻辑节选
for (int il = 0; il < n_layer; ++il) {
    struct ggml_tensor * inpSA = inpL;
    cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
    if (hparams.is_recurrent(il)) {
        cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);  // SSM 分支
    } else {
        cur = build_attention_layer(cur, inp_pos, inp->get_attn(), model, n_embd_head, il); // Attention 分支
    }
    // ... 行采样 ...
    cur = build_layer_ffn(cur, inpSA, model, il);  // 统一 FFN 入口（含 MoE+共享专家）
    inpL = cur;
}
if (hparams.f_logit_scale) {  // Logits 缩放
    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
}
```

### 29.8.2 build_attention_layer()：条件 RoPE 与 YaRN 支持

**源码路径**：`src/models/granite-hybrid.cpp` - `llm_build_granite_hybrid::build_attention_layer()`（第40-103行）

```cpp
// 关键逻辑节选
const bool use_rope = hparams.rope_finetuned;  // 条件 RoPE 标志
if (use_rope) {
    ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);  // 获取 YaRN 缩放因子
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, ...);     // 注入 factors
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, ...);
}
```

### 29.8.3 build_layer_ffn()：残差缩放 + MoE + 共享专家

**源码路径**：`src/models/granite-hybrid.cpp` - `llm_build_granite_hybrid::build_layer_ffn()`（第105-160行）

```cpp
// 关键逻辑节选
if (hparams.f_residual_scale) {  // 残差缩放（Granite 特色）
    cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
}
ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);  // 残差相加作为 FFN 输入

if (model.layers[il].ffn_gate_inp == nullptr) {  // 普通 FFN
    cur = build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR, il);
} else {  // MoE FFN
    cur = build_norm(ffn_inp, ..., LLM_NORM_RMS, il);
    ggml_tensor * moe_out = build_moe_ffn(..., LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il); // Softmax 门控
    if (hparams.n_ff_shexp > 0) {  // 共享专家
        ggml_tensor * ffn_shexp = build_ffn(cur, ..., LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_add(ctx0, moe_out, ffn_shexp);
    } else {
        cur = moe_out;
    }
}
if (hparams.f_residual_scale) {  // 输出前再次缩放残差
    cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
}
cur = ggml_add(ctx0, cur, ffn_inp);  // FFN 残差
```

**Granite-Hybrid 独特设计**：
1. **双重残差缩放**：`f_residual_scale` 作用于 Attention/SSM 输出后、FFN 输入前，以及 FFN 输出后、最终残差加法前。
2. **条件 RoPE**：`rope_finetuned` 标志控制是否启用 RoPE，配合 `get_rope_factors` 支持 YaRN 等缩放策略。
3. **Logits 缩放**：`f_logit_scale` 在 LM Head 后缩放 logits，控制输出分布方差。
4. **MoE + 共享专家**：复用标准 `build_moe_ffn` + `build_ffn(shexp)` 模式，门控用 Softmax。

## 29.9 PLaMo2 混合架构 —— 日本模型的"Mamba + 自定义注意力"

**`llm_build_plamo2` 的独特结构：双残差 + 后归一化**

与 Falcon-H1/Jamba 不同，PLaMo2 实现了“pre-mixer-norm + post-mixer-norm”。每个 block 由两个子层组成：mixer（SSM 或 Attn）和 FFN，每个子层都前后归一化。这种结构类似于 LLaMA 的 RMSNorm 设计，但 mixer 也有独立的 post_norm。

### 29.9.1 llm_build_plamo2::llm_build_plamo2()：双归一化主循环

**源码路径**：`src/models/plamo2.cpp` - `llm_build_plamo2::llm_build_plamo2()`（第1-203行）

```cpp
// 关键结构节选
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * residual = inpL;
    // pre_mixer_norm
    cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
    const bool is_mamba_layer = hparams.is_recurrent(il);
    if (is_mamba_layer) {
        cur = build_plamo2_mamba_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);
    } else {
        cur = build_plamo2_attn_layer(inp_hybrid->get_attn(), inp_pos, cur, model, il);
    }
    // post_mixer_norm  ← 关键差异：mixer 后再归一化
    cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
    cur = ggml_add(ctx0, cur, residual);  // mixer 残差
    residual = cur;
    // pre-ffn norm
    cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
    cur = build_ffn(..., LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);  // SEQ 串行模式
    // post ffn norm  ← FFN 后也归一化
    cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
    cur = ggml_add(ctx0, cur, residual);  // FFN 残差
    inpL = cur;
}
```

### 29.9.2 build_plamo2_attn_layer()：QKV 联合投影 + Q/K 预归一化

**源码路径**：`src/models/plamo2.cpp` - `llm_build_plamo2::build_plamo2_attn_layer()`（第47-108行）

```cpp
// 关键逻辑节选
ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, cur);  // 单一 wqkv 张量
// ggml_view_3d 拆分 Q/K/V
ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, ..., q_offset * ...);
ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, ..., k_offset * ...);
ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, ..., v_offset * ...);

Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);  // Q 预归一化
Qcur = ggml_rope_ext(...);  // RoPE 在 Norm 之后
Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);  // K 预归一化
Kcur = ggml_rope_ext(...);
cur = build_attn(inp, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, ..., 1.0f/sqrtf(float(n_embd_head_v)), il);
```

### 29.9.3 build_plamo2_mamba_layer()：定制化 Mamba2 变体

**源码路径**：`src/models/plamo2.cpp` - `llm_build_plamo2::build_plamo2_mamba_layer()`（第110-260行）

```cpp
// 关键差异点
ggml_tensor * zx = build_lora_mm(model.layers[il].ssm_in, cur);  // 投影出 zx (2*d_inner)
zx = ggml_permute(ctx0, zx, 0, 2, 1, 3);                         // 置换维度
zx = ggml_cont_4d(ctx0, zx, head_dim * 2, n_heads, n_seq_tokens, n_seqs); // 连续化为 4D (head_dim*2, n_heads, ...)
ggml_tensor * x = ggml_view_4d(ctx0, zx, head_dim, n_heads, ..., head_dim * ...); // 切片 x
ggml_tensor * z = ggml_view_4d(ctx0, zx, head_dim, n_heads, ..., 0);              // 切片 z

// SSM 内部：dt_dim 动态计算
const int64_t dt_dim = std::max(64, int(hparams.n_embd / 16));  // 动态 dt 维度
ggml_tensor * B  = ggml_view_3d(ctx0, x_bcdt, d_state, ...);   // B
ggml_tensor * C  = ggml_view_3d(ctx0, x_bcdt, d_state, ...);   // C
ggml_tensor * dt = ggml_view_3d(ctx0, x_bcdt, dt_dim, ...);    // dt

// B/C/dt 三者都做 RMS Norm（PLaMo2 强制）
B  = build_norm(B, model.layers[il].ssm_b_norm, ...);
C  = build_norm(C, model.layers[il].ssm_c_norm, ...);
dt = build_norm(dt, model.layers[il].ssm_dt_norm, ...);

// ... ssmm_scan + 状态保存 ...
ggml_tensor * D = ggml_reshape_2d(ctx0, model.layers[il].ssm_d, 1, n_heads);  // D 参数广播
y = ggml_add(ctx0, y, ggml_mul(ctx0, x, D));  // y += x * D
y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);  // 门控融合
```

**PLaMo2 设计亮点**：
- **Q/K 预归一化**：在 RoPE 之前对 Q/K 做 RMS Norm，提升位置编码稳定性。
- **动态 `dt_dim`**：`max(64, n_embd/16)` 自适应时间步秩。
- **强制三元组归一化**：B/C/dt 无条件 RMS Norm，区别于基类的条件归一化。
- **SEQ 模式 FFN**：`LLM_FFN_SEQ` 串行执行（gate→up→down），而非并行 PAR 模式。

## 29.10 PLaMo3 模板化架构 —— 显式实例化的"ISWA 复用术"

**`llm_build_plamo3<iswa>` 的模板化设计**

通过 `template <bool iswa>` 模板参数，让同一份代码既能生成标准版也能生成 ISWA 版。文件末尾 `template struct llm_build_plamo3<false>;` 和 `template struct llm_build_plamo3<true>;` 显式实例化。这种方式让 C++ 编译器自动为不同模板参数生成两份独立的类，零运行时开销。

### 29.10.1 llm_build_plamo3<iswa>::llm_build_plamo3()：模板化构建入口

**源码路径**：`src/models/plamo3.cpp` - `llm_build_plamo3<iswa>::llm_build_plamo3()`（第1-108行）

```cpp
// 逐行注释解释
llm_build_plamo3<iswa>::llm_build_plamo3(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {  // 直接继承 llm_graph_context（非 mamba_base）
    const int64_t head_dim_q = hparams.n_embd_head_k();
    const int64_t head_dim_v = hparams.n_embd_head_v();
    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>; // 编译期类型分支
    inp_attn_type * inp_attn = nullptr;
    if constexpr (iswa) {  // 编译期条件：选择 ISWA 或标准 KV 输入构建
        inp_attn = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;
        float freq_base_l  = 0.0f;
        float freq_scale_l = 0.0f;
        if constexpr (iswa) {  // ISWA 模式下每层 RoPE 频率可配置
            freq_base_l  = model.get_rope_freq_base (cparams, il);
            freq_scale_l = model.get_rope_freq_scale(cparams, il);
        } else {
            freq_base_l  = freq_base;    // 标准模式用全局频率
            freq_scale_l = freq_scale;
        }
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, cur);  // 联合 QKV
        // ... 拆分 Q/K/V (同 PLaMo2) ...
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);  // Q 预归一化
        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);  // K 预归一化
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l, ...); // 注入每层频率
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l, ...);
        const float attn_scale = 1.0f / sqrtf(float(head_dim_q));
        cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, attn_scale, il);
        // ... 行采样 ...
        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);  // Attention post-norm
        cur = ggml_add(ctx0, cur, residual);  // Attention 残差
        residual = cur;
        cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cur = build_ffn(..., LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);  // SEQ 模式 FFN
        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);  // FFN post-norm
        cur = ggml_add(ctx0, cur, residual);  // FFN 残差
        cur = build_cvec(cur, il);
        inpL = cur;
    }
    // ... 最终 Norm + LM Head ...
}
```

### 29.10.2 显式模板实例化

**源码路径**：`src/models/plamo3.cpp` - 全局实例化（第109-111行）

```cpp
// 类型即图纸：显式实例化避免跨编译单元链接问题
template struct llm_build_plamo3<false>;  // 标准 KV Cache 版本
template struct llm_build_plamo3<true>;   // ISWA 版本
```

**模板化设计精髓对比**：

| 技术点 | 作用 | 优势 |
|--------|------|------|
| `template <bool iswa>` | 编译期参数化 | 零运行时开销，类型安全 |
| `std::conditional_t<iswa, A, B>` | 类型别名分支 | `inp_attn` 类型编译期确定，调用正确的 `build_attn` 重载 |
| `if constexpr (iswa)` | 编译期代码分支 | 死代码消除，两个版本二进制互不包含对方逻辑 |
| `model.get_rope_freq_base(cparams, il)` | 每层差异化 RoPE | ISWA 模式支持层级频率，标准模式用全局频率 |
| `template struct ...` | 显式实例化 | 保证两个版本的符号在编译期生成，链接器可直接解析 |

## 29.11 设计中的取舍

为什么 Mamba 基类要同时提供 `build_mamba_layer` 和 `build_mamba2_layer`？

Mamba1 与 Mamba2 在张量布局、头数定义、投影方式上差异巨大（单头标量 vs 多头分组），强行统一会导致参数爆炸和大量 `if` 分支。基类提供两个正交实现，子类按 `model.arch` 选择，既保持代码复用（状态管理、Conv1D、SSM Scan、门控融合逻辑共享），又避免了运行时分支开销。

为什么 Jamba 用 `n_head_kv == 0` 判断层类型，而 Nemotron-H 用 `is_recurrent(il)`？

Jamba 的层类型由模型配置隐式决定（无 KV 头即为 SSM），配置简单但不够显式。Nemotron-H 显式存储 `is_recurrent` 数组，配合 `n_ff` 三分支，表达能力更强，支持 SSM/Attn/FFN 任意排列，适合更复杂的混合架构搜索。

Falcon-H1 并行融合与 Jamba 按层切换的本质区别是什么？

Falcon-H1 每层**同时**拥有 Attention 与 SSM 计算能力，计算图呈“分叉-汇合”菱形结构，参数量约为单分支的 2 倍，但捕捉局部精确依赖与全局压缩记忆的能力互补。Jamba 每层**单一**功能，参数量较省，但需通过层间传递融合信息，属于“时空分工”而非“空间并行”。

为什么 PLaMo 系列坚持“预归一化 + 后归一化”双层 Norm？

标准 Transformer 只有 pre-norm（或 post-norm）。PLaMo 在每个 mixer（Attn/SSM）与 FFN 子层**前后**各加一层 Norm，形成 `pre → mixer → post → residual → pre → ffn → post → residual` 结构。这在深层网络中提供了更强的梯度流控制和数值稳定性，代价是额外 4 次 Norm 计算/层。

C++ 类模板显式实例化（`template struct ...`）为何必须写在 `.cpp` 末尾？

类模板成员函数定义在头文件会导致代码膨胀；定义在 `.cpp` 又会导致隐式实例化失败（其他编译单元看不到定义）。显式实例化在 `.cpp` 末尾强制编译器生成 `iswa=false/true` 两个版本的完整类代码，既保持接口在头文件整洁，又保证链接期符号可见，是大型 C++ 项目模板复用的标准工程实践。

## 29.12 动手练习

1. **对比 Mamba1 与 Mamba2 的 SSM 层构建差异**
   - 阅读 `src/models/mamba-base.cpp` 第 15-186 行（`build_mamba_layer`）和第 188-358 行（`build_mamba2_layer`）。
   - 回答问题：
     1. 两种实现中 `n_head` 与 `head_dim` 的计算方式有何不同？
     2. 为什么 Mamba2 一次性投出 z/xBC/dt 三个张量，而 Mamba1 是分开投出 xz 与 x_db？
     3. Mamba2 引入的「分组 RMS Norm」（`ssm_norm`）作用在哪个张量上？分组数 `n_group` 如何影响？
     4. 在 SSM 扫描后，两种实现对状态保存（`ggml_cpy` 到 `ssm_states_all`）的字节偏移计算有何差异？
     5. `ggml_swiglu_split` 在两种实现中的调用方式是否完全一致？

2. **追踪 Mamba 混合架构的状态管理**
   - 阅读 `src/models/jamba.cpp`、`src/models/falcon-h1.cpp`、`src/models/nemotron-h.cpp` 的构造函数。
   - 回答问题：
     1. 三种架构各自通过什么方法获取混合输入对象？它们分别返回什么类型的指针？
     2. Jamba 通过 `hparams.n_head_kv(il) == 0` 区分层类型，Falcon-H1 和 Nemotron-H 分别用什么方式判断？
     3. Falcon-H1 的「并行融合」在计算图中对应什么 ggml 操作？与 Jamba 的「按层切换」有何本质区别？
     4. Nemotron-H 的 `build_ffn_layer` 中，MoE 分支与共享专家（`ffn_shexp`）是如何融合的？
     5. Nemotron-H 中 `LLM_FFN_RELU_SQR` 激活与标准 SwiGLU 在数学上有何不同？

3. **剖析 PLaMo 系列的预归一化与模板化设计**
   - 阅读 `src/models/plamo2.cpp` 第 102-160 行（`build_plamo2_attn_layer`）和 `src/models/plamo3.cpp` 整个文件。
   - 回答问题：
     1. PLaMo2 中 Q/K 的 RMS Norm 发生在 RoPE 之前还是之后？这种「norm-then-rope」设计有什么好处？
     2. PLaMo3 中 `std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>` 的作用是什么？编译后会生成几种类型？
     3. 解释 `if constexpr (iswa)` 与普通 `if` 的区别，并说明 RoPE 参数 `freq_base_l/freq_scale_l` 如何根据模板参数变化。
     4. PLaMo3 文件末尾的两行 `template struct llm_build_plamo3<false>;` 和 `<true>;` 为什么是必须的？
     5. 对比 PLaMo2 与 PLaMo3 在 Attention 子层结构上的异同：两者都有 `attn_q_norm/attn_k_norm` 吗？FFN 激活方式有何不同？

4. **理解 Granite-Hybrid 的缩放策略与 MoE 设计**
   - 阅读 `src/models/granite-hybrid.cpp` 整个文件。
   - 回答问题：
     1. `hparams.rope_finetuned` 标志的作用是什么？什么情况下 Granite 模型不需要 RoPE？
     2. `hparams.f_residual_scale` 在哪些位置被使用？它影响的是哪部分计算图的输出？
     3. `hparams.f_logit_scale` 与 `f_residual_scale` 的数学作用有何不同？为什么 logit 缩放需要除以（`1.0f / f_logit_scale`）？
     4. 当 `model.layers[il].ffn_gate_inp == nullptr` 和 `hparams.n_ff_shexp > 0` 同时成立时，FFN 实际如何构建？
     5. `ggml_rope_ext` 中传入的 `rope_factors`（通过 `get_rope_factors` 获取）通常用于支持哪些 RoPE 缩放策略？

5. **实战：实现一个简化版 Mamba 状态保存**
   - 基于 `src/models/mamba-base.cpp` 中 `build_mamba_layer` 的状态管理逻辑，回答问题：
     1. 在 SSM 扫描后，状态保存的 `ggml_view_1d` 是如何定位到当前序列（`kv_head`）对应的状态槽位的？
     2. 为什么需要先 `ggml_cpy` 回状态缓存，而不是直接原地修改（inplace）？
     3. `build_rs` 中的 lambda `get_ssm_rows` 接受哪些参数？它在状态复用时如何跳过未修改部分的拷贝？
     4. 如果你需要在状态保存后立即读取（例如做残差流验证），应该在计算图的哪个位置插入读取节点？
     5. 思考：Mamba 状态管理为何比 Transformer 的 KV Cache 复杂？提示：从「按 token 增量更新」与「按序列整体重算」两个角度分析。

## 29.13 本章小结

这一章中我们学习了 Mamba/SSM 与 Attention 混合架构在 llama.cpp 中的图构建实现。首先我们深入 `llm_build_mamba_base` 基类，理解了 Mamba1 三段式流水线（Conv1D + SSM 扫描 + 门控融合）与 Mamba2 多头分组设计（一次性投影、分组 RMS Norm）的核心差异。其次我们剖析了四种主流混合策略：Jamba 的按层切换（`n_head_kv==0` 判断）、Falcon-H1 的并行融合（`ggml_add` 聚合双分支）、Nemotron-H 的三分支任意排布（`is_recurrent`/`n_ff` 判断）以及 Granite-Hybrid 的标准化二段式（线性层+FFN 含 MoE）。接着我们探讨了 PLaMo2 的双归一化结构与定制化 Mamba2 变体，最后掌握了 PLaMo3 的模板化显式实例化技巧（`std::conditional_t` + `if constexpr` + 显式实例化）实现 ISWA 与标准版零开销复用。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| `llm_build_mamba_base` | SSM 循环层构建的共同基类，提供 build_mamba_layer 与 build_mamba2_layer 两个核心方法 |
| `llm_build_mamba_base::llm_build_mamba_base()` | 基类构造函数，仅将 llm_graph_params 透传给 llm_graph_context，无额外初始化逻辑 |
| `build_mamba_layer` | Mamba1 三段式流水线（Conv1D + SSM 扫描 + 门控融合），支持 ssm_dt_b_c_rms 条件归一化 |
| `build_mamba2_layer` | Mamba2 多头分组设计，n_head = ssm_dt_rank，引入分组 RMS Norm |
| `build_rs` | 获取循环状态缓存（conv state / ssm state），通过 lambda 实现按需复用 |
| `ggml_ssm_conv` | SSM 模型的 1D 卷积算子，与历史状态拼接形成 d_conv-1+n_seq_tokens 长度 |
| `ggml_ssm_scan` | SSM 核心并行关联扫描算子，实现 Mamba 论文 Annex D 的优化 |
| `ggml_swiglu_split` | 将 z 分支与 y 分支做门控融合，是 Mamba 系列的标志性输出方式 |
| `build_inp_mem_hybrid` | 混合输入管理，同时支持 KV Cache 与循环状态（SSM） |
| `llm_build_mamba` | Mamba/Mamba2 主模型构造函数，由 model.arch 分发到 build_mamba_layer 或 build_mamba2_layer |
| `llm_build_jamba` | AI21 Jamba 按层切换 Mamba/Attention，无 RoPE 的 Attention 变体 |
| `llm_build_falcon_h1` | Falcon-H1 并行融合 Attention 与 Mamba2，ggml_add 聚合两路输出 |
| `llm_build_nemotron_h` | Nemotron-H 三分支（SSM/Attention/FFN）混合架构，支持 MoE + 共享专家 |
| `LLM_FFN_RELU_SQR` | Nemotron 特色的 ReLU 平方激活函数 |
| `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` | Nemotron 使用的 Sigmoid 门控（区别于 Softmax） |
| `ffn_shexp` | Nemotron/Granite 的共享专家分支，独立于 MoE 路由专家 |
| `llm_build_granite_hybrid` | IBM Granite-Hybrid，f_residual_scale 残差缩放 + f_logit_scale logits 缩放 |
| `rope_finetuned` | Granite 条件启用 RoPE 的标志，配合 get_rope_factors 支持 YaRN |
| `llm_build_plamo2` | PLaMo2 双归一化结构（pre/post-mixer-norm），Q/K 独立 RMS Norm 后再 RoPE |
| `build_plamo2_mamba_layer` | PLaMo2 定制化 Mamba2，dt_dim 动态调整，B/C/dt 三者都做 RMS Norm |
| `llm_build_plamo3<iswa>` | 模板化 PLaMo3，通过显式实例化复用 ISWA 与标准 KV Cache |
| `std::conditional_t<iswa, ...>` | C++ 类型分支技巧，根据 iswa 决定 attn 输入类型 |
| `if constexpr (iswa)` | 编译期条件分支，为模板参数生成不同二进制代码 |
| `template struct ...` | 显式模板实例化，避免隐式实例化跨编译单元链接问题 |
| `attn_q_norm / attn_k_norm` | Q/K 预归一化权重，PLaMo 系列在 RoPE 之前归一化的特殊设计 |
| `attn_post_norm / ffn_post_norm` | PLaMo 风格的后归一化，每个 mixer 子层后独立归一化 |
| `build_attn_inp_kv_iswa` | 构建 ISWA 注意力输入对象，支持 SWA 与非 SWA 双 KV Cache |
| `kq_scale` | 注意力 QK 缩放因子，可由 hparams.f_attention_scale 自定义 |

下一章中，我们将学习 RWKV 与线性注意力架构，理解这些 RNN-like 模型如何通过专属基类实现时间混合与通道混合，并探索 DeltaNet、Qwen3Next 等新一代线性注意力模型的图构建实现。

感谢你读到了这里，恭喜你完成了第30章的学习！你已经掌握了 llama.cpp 中 SSM 与混合架构的核心图构建逻辑，为理解后续的 RWKV、DeltaNet 等线性注意力模型打下了坚实基础。


