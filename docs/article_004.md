
# 第4章 Gemma 家族深度解析 —— Google 模型的「转换四部曲」

## 学习目标

- 理解 Gemma 家族在 GGUF 转换中的四代演进：基础款 norm.weight+1、Gemma3 的 vocab 截断与多模态分流、Gemma3N 的 AltUp 多通道堆叠、Gemma4 的共享 KV 与比例 RoPE
- 掌握 Gemma 系列 norm_shift（+1 偏移）的反向补偿机制，以及 Gemma3N 为何关闭 norm_shift
- 理解 Gemma3N 视觉塔使用 MobileNetV5 双层 block 路径映射（bid.{sid}.suffix）的解析原理
- 了解 Gemma3N AltUp 投影张量的捕获与 3D 堆叠过程，以及 vision/audio token 嵌入 padding 到 262400 的工作机制
- 掌握 Gemma4 共享 KV 层标记、PL Embedding 维度、dual head_dim（global vs sliding）以及 Proportional RoPE 频率因子（partial_rotary_factor=0.25 + 1e30 哨兵）的实现细节
- 熟悉 Gemma3/Gemma4 的 sliding_window_pattern 与 layer_types 对齐机制，以及 Activation Sparsity Pattern 的 normal.icdf 还原方法
- 掌握 Gemma3/Gemma4 多模态转换器（mmproj）的实现差异，包括 Gemma3VisionModel 的 soft_emb_norm +1 与 Gemma3nVisionAudioModel 的 MobileNetV5 block_tensor_mapping
- 熟悉 Gemma4VisionAudioModel 的 audio Conformer（softplus + per_dim_key_scale）与 vision（patch_embedder 2D→4D conv）处理流程

## 4.1 生活类比

想象 Gemma 家族是一座「Google 大厦的四代升级工程」：
- **Gemma 基础款** = 一栋毛坯公寓，每间房（norm 层）需要开发商偷偷给家具加 1cm 垫片（norm.weight+1），搬进 GGML 大厦时必须把垫片拆加
- **Gemma3** = 在公寓里加装智能家居系统，根据户型图（is_causal）决定要不要装安防摄像头（causal_attention），多余的房间号（OOV vocab）直接封掉
- **Gemma3N** = 引入 AltUp「四胞胎平行电梯」，每个乘客（token）有 4 套独立通道，管家（_stack_matrices）把三块分散的电梯控制板拼成一块 3D 总控屏
- **Gemma3N 的 vision/audio 嵌入** = 大厦顶层留出 256 个「客房」（262144-262399）给访客住，本地居民（语言 token）走正门，访客走侧门（vision/audio 独立路径）
- **Gemma4** = 在顶层加盖「共享办公区」（shared_kv layers），最后 N 层共用会议室（KV cache），会议室按 global/sliding 模式分大小（dual head_dim）
- **Proportional RoPE** = 旋转门只转 25%（partial_rotary_factor），剩下 75% 用 1e30「液压墙」挡住，让 ggml 知道这些维度不能转
- **Gemma3n Vision 的 MobileNetV5** = 大厦装了一套 timm 牌「折叠家居」，每块家具用 {bid}.{sid} 双层编号（哪个抽屉的哪个格子），custom_map 是一张自动翻译表
- **Gemma4 Audio Conformer** = 语音管家用 softplus 给每个旋钮（per_dim_key_scale）加软化滤镜，让声音更圆润；patch_embedder 则把 2D 拼图重新拼成 3D 立体画
就像四代大厦不断迭代，大堂经理（norm_shift）从「+1 垫片」变成「不加垫片」，电梯工（AltUp）从无到有，旋转门工程师（RoPE）也在不断换新工艺。

## 4.2 源码地图

./convert_hf_to_gguf.py
├── GemmaModel (基础款)
│   ├── set_vocab()                                 # 3081-3104 行，SPM + CodeGemma 特殊 token
│   ├── set_gguf_parameters()                       # 写入 head_dim 等基础参数
│   ├── modify_tensors()                            # 3106-3132 行，norm.weight +1
│   └── GemmaModel.norm_shift()                    # 1.0 默认值 (基类 TextModel.norm_shift)
├── Gemma2Model
│   ├── set_vocab()                                 # SPM + add_space_prefix=False
│   ├── set_gguf_parameters()                       # attn_logit/final_logit softcapping + sliding_window
│   └── modify_tensors()                            # 沿用 norm.weight +1
├── Gemma3Model (Gemma3 + Gemma3ForConditionalGeneration)
│   ├── norm_shift()                               # 3242-3243 行，返回 1.0
│   ├── set_vocab()                                # 3235-3240 行，SPM/GPT2 双轨制
│   ├── set_gguf_parameters()                      # 3245-3269 行，causal_attention/sliding_window
│   └── modify_tensors()                           # 3295-3322 行，norm+1 + vocab 截断
├── Gemma3VisionModel (MmprojModel)
│   ├── set_gguf_parameters()                      # VisionProjectorType.GEMMA3 + soft_emb_norm +1
│   ├── tensor_force_quant()                       # input_projection=F16, embeddings=F32
│   └── modify_tensors()                           # 仅处理 soft_emb_norm.weight +1
├── Gemma3nVisionAudioModel (ConformerAudioModel, MmprojModel)
│   ├── block_tensor_mapping                       # MobileNetV5 双层 {bid}.{sid} 映射字典
│   ├── __init__()                                 # audio/vision hparams 重映射 + audio_bos_eos
│   ├── set_gguf_parameters()                      # GEMMA3NV + GEMMA3NA projector type
│   ├── tensor_force_quant()                       # input_projection/input_proj=F16, embeddings=F32
│   ├── custom_map()                                # 3573-3581 行，模板化重写
│   └── modify_tensors()                           # 3583-3605 行，timm 块路由 + Audio Conformer
├── Gemma3NModel (Gemma3N + Gemma3NForConditional)
│   ├── __init__()                                 # 3627-3637 行，初始化 _altup_proj/_altup_unembd 缓冲
│   ├── norm_shift()                               # 始终返回 0.0
│   ├── set_vocab()                                # 临时删除 vocab_size_per_layer_input 强制使用 vocab_size=262400
│   ├── set_gguf_parameters()                      # 3654-3679 行，altup/shared_kv/sparsity/sliding_window_pattern
│   ├── _stack_matrices()                          # 3681-3687 行，3D 堆叠工具
│   └── modify_tensors()                           # 3707-3767 行，AltUp 缓冲 + embed_tokens padding + _scale 后缀
├── Gemma4Model (Gemma4ForConditionalGeneration)
│   ├── norm_shift()                               # 返回 0.0
│   ├── set_vocab()                                # 3823-3851 行，LlamaHfVocab + visible_tokens USER_DEFINED
│   ├── set_gguf_parameters()                      # 3884-3936 行，shared_kv_layers / dual head_dim / use_double_wide_mlp
│   ├── generate_extra_tensors()                   # 3938-3955 行，Proportional RoPE FREQS (1e30 哨兵)
│   └── modify_tensors()                           # 3957-3977 行，per_dim_scale/layer_scalar + per_expert_scale → FFN_DOWN_EXP.scale
├── Gemma4VisionAudioModel (MmprojModel)
│   ├── __init__()                                 # vision/audio hparams 重映射
│   ├── set_gguf_parameters()                      # GEMMA4V + GEMMA4A projector type
│   ├── is_audio_tensor()                          # audio_tower / embed_audio 判定
│   ├── tensor_force_quant()                       # audio conv=F32 + position_embedding_table=F32
│   └── modify_tensors()                           # 3987-4015 行，softplus + per_dim_key_scale + patch_embedder 2D→4D conv
└── ConformerAudioModel (基础音频父类)
├── is_audio_tensor()                            # audio/codebook/conformer 关键字判定
├── tensor_force_quant()                         # audio conv 强制 F32
└── modify_tensors()                             # batch_norm 折叠 + conv 形状重塑

## 4.3 Gemma 家族解码器的「转换四部曲」—— 从基础归一化到前沿 AltUp、滑动窗口

### 4.3.1 为什么需要 Gemma 家族章节？

- 统一解析 Gemma、Gemma3、Gemma3N、Gemma4 在 GGUF 转换中的差异点
- 理解不同代次的 norm_shift（+1 偏移）机制
- 看 AltUp 投影、共享 KV 层、滑动窗口模式如何被特殊处理

### 4.3.2 为什么 norm_shift = +1 是 Gemma 的标志？

- Gemma RMSNorm 在 forward 时执行 `weight * (1.0 + w)`，因此存储时减去 1
- 在 `convert_hf_to_gguf.py` 中通过 `norm_shift()` 方法做反向补偿
- Gemma3 默认为 1.0，但 Gemma3N/Gemma4 关闭（0.0），因为其 NormScale shift 在 forward 已隐含

### 4.3.3 GemmaModel 基础款：norm.weight +1 的艺术

`modify_tensors` 中检查 `name.endswith("norm.weight")`，`data_torch = data_torch + 1` 反向补偿回 GGML 端期望的值。`set_vocab` 标记 `add_space_prefix = False`，符合 SentencePiece 规范。

#### 4.3.3.1 GemmaModel.modify_tensors()：基础归一化权重补偿

源码路径：`./convert_hf_to_gguf.py` - `GemmaModel.modify_tensors()`（3106-3132行）

```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # ① 跳过 lm_head（与 embedding 共享权重）
        if name == "lm_head.weight":
            return
        # ② 核心补偿：所有 norm.weight 加 1
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1
        # ③ 委托父类完成名称映射与写入
        yield from super().modify_tensors(data_torch, name, bid)
```

这段代码实现了 Gemma 特有的归一化层权重反向补偿：HF 模型在推理时执行 `x * (1 + w)`，权重存储时已预减 1，转换回 GGUF 时必须加回 1。

#### 4.3.3.2 GemmaModel.set_vocab()：SentencePiece 词表与 CodeGemma 特殊 token

源码路径：`./convert_hf_to_gguf.py` - `GemmaModel.set_vocab()`（3081-3104行）

```python
    def set_vocab(self):
        # ① 使用 SentencePiece 加载基础词表
        self._set_vocab_sentencepiece()
        # ② 注入 CodeGemma 特有的 5 个特殊 token
        special_vocab = gguf.SpecialVocab(...)
        special_vocab._set_special_token("prefix", 67)
        special_vocab._set_special_token("suffix", 69)
        special_vocab._set_special_token("middle", 68)
        special_vocab._set_special_token("fsep", 70)
        special_vocab._set_special_token("eot", 107)
        special_vocab.add_to_gguf(self.gguf_writer)
        # ③ 禁用空格前缀，符合 SPM 规范
        self.gguf_writer.add_add_space_prefix(False)
```

这段代码完成 Gemma 基础款的词表装配：先用 `_set_vocab_sentencepiece` 读取 tokenizer.model，再显式注入 FIM（Fill-in-the-Middle）任务所需的特殊 token。

### 4.3.4 Gemma2Model：soft-capping 与滑动窗口的继承者

沿用 `norm.weight +1`，新增 `attn_logit_softcapping`、`final_logit_softcapping` 和 `sliding_window` 三个 GGUF 参数。

### 4.3.5 Gemma3Model：因果 vs 非因果与 vocab 截断

通过 `is_causal` 自动写入 `causal_attention` 元数据。OOV 行裁剪：`embed_tokens.weight` 只保留真实 vocab 大小。`sliding_window_pattern != 1` 时写入 `sliding_window` KV 元数据。

#### 4.3.5.1 Gemma3Model.norm_shift()：保持 +1 偏移

源码路径：`./convert_hf_to_gguf.py` - `Gemma3Model.norm_shift()`（3242-3243行）

```python
    def norm_shift(self, name: str) -> float:
        # 所有 norm.weight 依然需要 +1 补偿
        return 1.0 if name.endswith("norm.weight") else 0.0
```

#### 4.3.5.2 Gemma3Model.set_vocab()：SPM/GPT2 双轨制

源码路径：`./convert_hf_to_gguf.py` - `Gemma3Model.set_vocab()`（3235-3240行）

```python
    def set_vocab(self):
        if (self.dir_model / "tokenizer.model").is_file():
            # 路径 A：存在 tokenizer.model，走 SentencePiece
            self._set_vocab_sentencepiece()
            self.gguf_writer.add_add_space_prefix(False)
        else:
            # 路径 B：多模态版本无 SPM，fallback 到 GPT-2
            self._set_vocab_gpt2()
```

#### 4.3.5.3 Gemma3Model.set_gguf_parameters()：因果注意力与滑动窗口元数据

源码路径：`./convert_hf_to_gguf.py` - `Gemma3Model.set_gguf_parameters()`（3245-3269行）

```python
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        # 基础结构参数
        self.gguf_writer.add_context_length(hparams.get("max_position_embeddings", 131072))
        self.gguf_writer.add_head_count(hparams.get("num_attention_heads", 8))
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("rms_norm_eps", 1e-6))
        self.gguf_writer.add_key_length(hparams.get("head_dim", 256))
        self.gguf_writer.add_value_length(hparams.get("head_dim", 256))
        # Soft-capping
        if (final_logit_softcap := hparams.get("final_logit_softcapping")):
            self.gguf_writer.add_final_logit_softcapping(final_logit_softcap)
        # 关键：KV 头数与因果注意力
        self.gguf_writer.add_head_count_kv(hparams.get("num_key_value_heads", 4))
        if (is_causal := hparams.get("is_causal")) is not None:
            self.gguf_writer.add_causal_attention(is_causal)
        if is_causal is False:
            self.gguf_writer.add_sliding_window(hparams.get("sliding_window", 4096))
```

这段代码展示了 Gemma3 的核心特性配置：支持因果/非因果双模式、显式写入 KV 头数、条件性启用滑动窗口。

#### 4.3.5.4 Gemma3Model.modify_tensors()：词表裁剪与归一化补偿

源码路径：`./convert_hf_to_gguf.py` - `Gemma3Model.modify_tensors()`（3295-3322行）

```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # ① 移除 language_model 前缀
        if "language_model." in name:
            name = name.replace("language_model.", "")
        # ② 跳过视觉/多模态张量
        elif name.startswith("multi_modal_projector.") or name.startswith("vision_tower.") or name.startswith("vision_model."):
            return
        # ③ 词表裁剪：embed_tokens.weight 只保留真实 vocab
        if "embed_tokens.weight" in name:
            data_torch = data_torch[:n_vocab_real]
        # ④ 应用 norm_shift（+1 或 0）
        f_shift = self.norm_shift(name)
        if f_shift != 0.0:
            data_torch = data_torch + f_shift
        yield from super().modify_tensors(data_torch, name, bid)
```

这段代码完成三件事：多模态前缀清洗、视觉张量分流、词表 OOV 行裁剪、归一化权重补偿。

### 4.3.6 Gemma3NModel：AltUp 多通道与 vision/audio token 嵌入填充

`_altup_proj` 与 `_altup_unembd` 缓冲三组 AltUp 投影矩阵，堆叠为 3D 张量：
- 通过 `if ".0." / ".1." / ".2."` 分支识别索引
- `_stack_matrices` 检测全部到齐后写入 `model.altup_*_projections.weight`

**Vision/Audio 特殊 Token 的嵌入 padding**：
- 因 `vocab_size_per_layer_input = 262144`，但真实模型含视觉 token 到 262400
- `embed_tokens.weight` 与 `embed_tokens_per_layer` 都需 zero-pad 到 `vocab_size`
- `name.replace("language_model.", "")` 去除多模态命名空间

**Activation Sparsity Pattern 的统计近似**：
- 通过 `torch.distributions.normal.Normal(0, 1).icdf(sparsity)` 还原 multiplier
- `add_activation_sparsity_scale(activation_sparsity_scale)` 写入元数据
- 用于在 GGML 推理时跳过低激活维度的计算

**Sliding Window Pattern 与 layer_types 严格对齐**：
- 遍历 `hparams["layer_types"]`，生成布尔列表 `t == "sliding_attention"`
- `add_sliding_window_pattern(...)` 与 sliding_window 长度匹配
- 让运行时精准区分 ISWA 双缓存策略

**Gemma3nVisionAudioModel.tensor_force_quant 行为**：
- `input_projection/input_proj` 强制 F16 精度（避免 quantization 精度损失）
- `.embeddings.` 和 `stem` 强制 F32 精度（保护位置编码与 stem 卷积）

### 4.3.7 Gemma4Model：共享 KV 层与 per-layer 输入嵌入

`add_shared_kv_layers(num_kv_shared_layers)` 标记最后 N 层为 KV 共享。`add_embedding_length_per_layer_input` 写入 per-layer 投影维度。区分 `head_dim_full`（global）与 `head_dim_swa`（sliding）写入 `add_key_length(_swa)`。

### 4.3.8 Gemma4Model：Proportional RoPE 频率因子的精妙

`generate_extra_tensors` 中：rotary_dim 取 head_dim_full 一半。用 `1e30` 标记不旋转的维度，让 ggml 跳过对应位置。生成 `ROPE_FREQS` 张量，与 Llama3 RoPE 频率因子机制一致。

### 4.3.9 Gemma3VisionModel / Gemma3nVisionAudioModel / Gemma4VisionAudioModel

- `Gemma3VisionModel` 写入 GEMMA3 projector 并修正 `soft_emb_norm.weight` +1
- `Gemma3nVisionAudioModel` 通过 `block_tensor_mapping` 模板化 MobileNetV5 `{bid}.{sid}` 双层路径
- `Gemma4VisionAudioModel` 对 audio `per_dim_key_scale` 做 `softplus` 变换并把 2D conv 折叠成 4D 格式

## 4.4 Gemma 词表与特殊 token —— GGUF 转换中的「token 翻译官」

### 4.4.1 GemmaModel 的词表特殊性

- 使用 SentencePiece，`_set_vocab_sentencepiece` 读取 tokenizer.model
- 显式注入 `prefix/suffix/middle/fsep/eot` 特殊 token（CodeGemma 专属）
- `chat_template = None` 避免 SpecialVocab 重复注入

### 4.4.2 Gemma3Model 词表双轨制

- 若存在 `tokenizer.model`，调用 SentencePiece 路径并关闭 add_space_prefix
- 否则 fallback 到 `_set_vocab_gpt2`（支持 Gemma3 多模态版本）
- 自动通过 `get_vocab_base_pre` 验证预分词器哈希

### 4.4.3 Gemma4Model：LlamaHfVocab 抽象的复用

- 直接构造 `gguf.LlamaHfVocab(self.dir_model)`
- 通过 `visible_tokens` 集合强制 `<|channel|>` 等控制 token 标记为 USER_DEFINED
- 这样 chat parser 能正确读取 channel 标签

#### 4.4.3.1 Gemma4Model.set_vocab()：LlamaHfVocab + visible_tokens

源码路径：`./convert_hf_to_gguf.py` - `Gemma4Model.set_vocab()`（3823-3851行）

```python
    def set_vocab(self):
        # ① 复用 Llama 的 HF 词表加载器
        vocab = gguf.LlamaHfVocab(self.dir_model)
        # ② 定义必须可见的控制 token（channel 标签等）
        visible_tokens = {
            token for token in ["<start_of_turn>", "<end_of_turn>", "<pad>", "<|channel|>"]
            if token in vocab.token_to_id
        }
        # ③ 遍历所有 token 写入 GGUF
        tokens, scores, toktypes = [], [], []
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            # 关键：visible_tokens 强制标记为 USER_DEFINED
            if text in visible_tokens:
                toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(toktype)
        assert len(tokens) == vocab.vocab_size
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        # ④ 补充 SpecialVocab
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)
```

这段代码展示了 Gemma4 如何复用 Llama 词表基础设施，同时通过 `visible_tokens` 机制确保对话控制 token 被正确识别。

### 4.4.4 Gemma3nVisionAudioModel：MobileNetV5 双层映射

`block_tensor_mapping` 字典实现 `{bid}.{sid}` 双层 block 路径映射。`custom_map` 解析 `model.vision_tower.timm_model.blocks.{bid}.{sid}.suffix`。视觉 encoder 走 MobileNetV5 的 timm 风格，文本 mmproj 走 ggml 标准路径。

#### 4.4.4.1 Gemma3nVisionAudioModel.custom_map()：双层路径模板化

源码路径：`./convert_hf_to_gguf.py` - `Gemma3nVisionAudioModel.custom_map()`（3573-3581行）

```python
    def custom_map(self, name: str) -> str | None:
        # timm 风格：blocks.{bid}.{sid}.suffix → v.blk.{bid}.{sid}.suffix
        match = re.match(r"model\.vision_tower\.timm_model\.blocks\.(\d+)\.(\d+)\.(.+)", name)
        if match:
            bid, sid, suffix = match.groups()
            return f"v.blk.{bid}.{sid}.{suffix}"
        return None
```

这段代码实现了 timm 命名空间到 GGUF 标准命名的双层索引转换。

#### 4.4.4.2 Gemma3nVisionAudioModel.modify_tensors()：timm 块路由与 Audio Conformer

源码路径：`./convert_hf_to_gguf.py` - `Gemma3nVisionAudioModel.modify_tensors()`（3583-3605行）

```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # ① custom_map 重写名称
        mapped = self.custom_map(name)
        if mapped is not None:
            yield from super().modify_tensors(data_torch, mapped, bid)
            return
        # ② 音频张量走 ConformerAudioModel 逻辑
        if ConformerAudioModel.is_audio_tensor(name):
            yield from ConformerAudioModel.modify_tensors(self, data_torch, name, bid)
            return
        # ③ 其他视觉张量按标准映射
        yield from super().modify_tensors(data_torch, name, bid)
```

### 4.4.5 Gemma4VisionAudioModel：post layernorm 偏移为 1

`_per_dim_key_scale` 后应用 `softplus` 变换。通过 `unsqueeze` 把 scalar 扩展成 1D 张量。patch_embedder.input_proj 从 2D 折叠成 (C, 3, H, W) 标准 conv2d 格式。

#### 4.4.5.1 Gemma4VisionAudioModel.modify_tensors()：audio softplus + vision conv2d 折叠

源码路径：`./convert_hf_to_gguf.py` - `Gemma4VisionAudioModel.modify_tensors()`（3987-4015行）

```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # ① 区分音频/视觉分支
        if self.is_audio_tensor(name):
            # Audio Conformer 路径
            if name.endswith("_per_dim_key_scale"):
                # softplus 激活并扩展为 1D
                data_torch = torch.nn.functional.softplus(data_torch).unsqueeze(0)
                name = name + ".weight"
            yield from super().modify_tensors(data_torch, name, bid)
        else:
            # Vision ViT 路径
            if "patch_embedder.input_proj" in name and "weight" in name:
                # 2D [n_embd, patch*patch*3] → 4D [n_embd, 3, patch, patch]
                n_embd = data_torch.shape[0]
                patch = int((data_torch.shape[1] // 3) ** 0.5)
                data_torch = data_torch.reshape(n_embd, 3, patch, patch)
            yield from super().modify_tensors(data_torch, name, bid)
```

### 4.4.6 Gemma3NModel.set_vocab 的特殊覆盖逻辑

临时删除 `vocab_size_per_layer_input` 字段，强制读取 `vocab_size`。调用父类后恢复 `vocab_size_per_layer_input`，供后续 padding 流程使用。这样才能正确支持 vision/audio token 区间 262144-262399。

## 4.5 Gemma3N AltUp 投影器 —— 解读「多通道并行预测」转换管线

### 4.5.1 为什么 Gemma3N 单独处理 AltUp？

- AltUp 是 Google 的多通道预测机制：每个 token 维护 4 个并行表示
- 在 GGUF 中被压缩为 3D 张量（hidden_size, hidden_size, num_inputs-1）
- `altup_active_idx` 决定哪一通道参与当前推理

### 4.5.2 AltUp 投影张量的捕获

- `_altup_proj = [Tensor(), Tensor(), Tensor()]` 三元素列表作为 buffer
- `modify_tensors` 中 `if ".0." / ".1." / ".2."` 按 index 写入对应槽
- 全部到齐后调用 `_stack_matrices` 一次性 stack 出 3D 输出

#### 4.5.2.1 Gemma3NModel.__init__()：初始化空张量缓冲

源码路径：`./convert_hf_to_gguf.py` - `Gemma3NModel.__init__()`（3627-3637行）

```python
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ① 初始化三个空张量占位：对应 .0/.1/.2 三个投影分支
        self._altup_proj = [torch.Tensor(), torch.Tensor(), torch.Tensor()]
        self._altup_unembd = [torch.Tensor(), torch.Tensor(), torch.Tensor()]
```

#### 4.5.2.2 Gemma3NModel._stack_matrices()：3D 堆叠工具

源码路径：`./convert_hf_to_gguf.py` - `Gemma3NModel._stack_matrices()`（3681-3687行）

```python
    def _stack_matrices(self, matrices: list[Tensor], base_name: str) -> tuple[str, Tensor]:
        # 三个 [hidden, hidden] 堆叠为 [hidden, hidden, 3]
        stacked = torch.stack(matrices, dim=-1)
        name = self.format_tensor_name(gguf.MODEL_TENSOR.__dict__[base_name])
        return name, stacked
```

#### 4.5.2.3 Gemma3NModel.modify_tensors()：AltUp 缓冲与 embed_tokens padding

源码路径：`./convert_hf_to_gguf.py` - `Gemma3NModel.modify_tensors()`（3707-3767行）

```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # ① 去除 language_model 前缀
        if "language_model." in name:
            name = name.replace("language_model.", "")
        # ② 捕获 .0/.1/.2 三个 AltUp 投影分支
        for idx, suffix in enumerate([".0.", ".1.", ".2."]):
            if suffix in name and "altup" in name and "weight" in name:
                if "proj" in name:
                    self._altup_proj[idx] = data_torch
                elif "unembd" in name:
                    self._altup_unembd[idx] = data_torch
                # 三个槽位都填满时触发堆叠
                if all(t.numel() > 0 for t in self._altup_proj):
                    for bn, buf in [("altup_proj", self._altup_proj), ("altup_unembd", self._altup_unembd)]:
                        if all(t.numel() > 0 for t in buf):
                            yield self._stack_matrices(buf, bn.upper())
                return
        # ③ embed_tokens.weight padding 到 262400
        if name == "embed_tokens.weight":
            current = data_torch.shape[0]
            target = self.hparams["vocab_size"]  # 262400
            if current < target:
                pad = torch.zeros(target - current, data_torch.shape[1], dtype=data_torch.dtype)
                data_torch = torch.cat([data_torch, pad], dim=0)
        # ④ embed_tokens_per_layer 同样 padding
        if name == "embed_tokens_per_layer.weight":
            current = data_torch.shape[0]
            target = self.hparams["vocab_size"]
            if current < target:
                pad = torch.zeros(target - current, data_torch.shape[1], dtype=data_torch.dtype)
                data_torch = torch.cat([data_torch, pad], dim=0)
        # ⑤ _scale 后缀张量重命名
        if name.endswith("_scale"):
            name = name[:-6] + ".scale"
        yield from super().modify_tensors(data_torch, name, bid)
```

### 4.5.3 Vision/Audio 特殊 Token 的嵌入 padding

- 因 `vocab_size_per_layer_input = 262144`，但真实模型含视觉 token 到 262400
- `embed_tokens.weight` 与 `embed_tokens_per_layer` 都需 zero-pad 到 `vocab_size`
- `name.replace("language_model.", "")` 去除多模态命名空间

### 4.5.4 Activation Sparsity Pattern 的统计近似

- 通过 `torch.distributions.normal.Normal(0, 1).icdf(sparsity)` 还原 multiplier
- `add_activation_sparsity_scale(activation_sparsity_scale)` 写入元数据
- 用于在 GGML 推理时跳过低激活维度的计算

#### 4.5.4.1 Gemma3NModel.set_gguf_parameters()：AltUp/稀疏/滑动窗口元数据

源码路径：`./convert_hf_to_gguf.py` - `Gemma3NModel.set_gguf_parameters()`（3654-3679行）

```python
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        # AltUp 配置
        self.gguf_writer.add_altup_num_inputs(hparams.get("num_inputs", 4))
        self.gguf_writer.add_altup_coef_scale(hparams.get("altup_coef_scale", 1.0))
        # 共享 KV（Gemma3N 不共享，但保留接口）
        self.gguf_writer.add_shared_kv_layers(0)
        # Activation Sparsity
        sparsity = hparams.get("activation_sparsity_pattern", 0.0)
        if sparsity:
            from torch.distributions.normal import Normal
            scale = float(Normal(0, 1).icdf(torch.tensor(sparsity)))
            self.gguf_writer.add_activation_sparsity_scale(scale)
        # Sliding Window Pattern 与 layer_types 严格对齐
        layer_types = hparams.get("layer_types", [])
        swa_pattern = [t == "sliding_attention" for t in layer_types]
        if any(swa_pattern):
            self.gguf_writer.add_sliding_window(hparams.get("sliding_window", 4096))
            self.gguf_writer.add_sliding_window_pattern(swa_pattern)
```

### 4.5.5 Sliding Window Pattern 与 layer_types 严格对齐

遍历 `hparams["layer_types"]`，生成布尔列表 `t == "sliding_attention"`。`add_sliding_window_pattern(...)` 与 sliding_window 长度匹配。让运行时精准区分 ISWA 双缓存策略。

### 4.5.6 Gemma3nVisionAudioModel.tensor_force_quant 行为

- `input_projection/input_proj` 强制 F16 精度（避免 quantization 精度损失）
- `.embeddings.` 和 `stem` 强制 F32 精度（保护位置编码与 stem 卷积）

## 4.6 Gemma4 共享 KV 与滑动窗口 —— 双缓存机制与 RoPE 比例

### 4.6.1 为什么 Gemma4 有共享 KV 层？

- 节省长推理的 KV cache 占用
- `hparams["num_kv_shared_layers"]` 标记末尾 N 层共享 KV
- `add_shared_kv_layers(...)` 让 C++ 端识别哪些层可以共享

### 4.6.2 Gemma4 的 per-layer 输入嵌入（PL Embedding）

`hidden_size_per_layer_input` 是辅助输入的维度。通过 `add_embedding_length_per_layer_input` 写入。用于 `project_per_layer_inputs` 投影层。

### 4.6.3 dual head_dim：global vs sliding 双轨并行

`global_head_dim` 与 `head_dim` 在 Gemma4 中并存。写入 `add_key_length` (global) 与 `add_key_length_swa` (sliding)。运行时根据 layer_types 决定读哪个。

### 4.6.4 use_double_wide_mlp 与 FFN 双倍扩容

当 `use_double_wide_mlp = True` 时，shared KV 层用 2 倍 FFN。动态生成 `n_ff_arr` 数组，写入 `add_feed_forward_length(n_ff_arr)`。让 C++ 端按 per-layer 维度调整。

### 4.6.5 Proportional RoPE 频率因子的精妙

`partial_rotary_factor = 0.25`，即只有 25% 维度参与旋转。`freqs = 1.0 / (base ** (arange(0, dim, 2) / dim))` 生成基础频率。未旋转维度用 `1e30` 标记，让 ggml 在 RoPE 公式中跳过。

#### 4.6.5.1 Gemma4Model.generate_extra_tensors()：Proportional RoPE 与 1e30 哨兵

源码路径：`./convert_hf_to_gguf.py` - `Gemma4Model.generate_extra_tensors()`（3938-3955行）

```python
    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        hparams = self.hparams
        head_dim = hparams.get("head_dim", 256)
        rotary_dim = head_dim // 2  # 仅旋转一半维度
        partial_rotary_factor = hparams.get("partial_rotary_factor", 0.25)
        n_rot = int(rotary_dim * partial_rotary_factor)
        n_unrot = rotary_dim - n_rot
        # 基础频率
        base = hparams.get("rope_theta", 10000.0)
        freqs = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        # 关键：未旋转维度用 1e30 标记
        rope_freqs = torch.cat([
            freqs[:n_rot],           # 旋转部分：正常频率
            torch.full((n_unrot,), 1e30)  # 非旋转部分：哨兵值
        ])
        yield self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), rope_freqs
```

这段代码生成了 Gemma4 独有的 Proportional RoPE 频率因子张量：前 25% 维度正常旋转，后 75% 维度用 1e30 标记，告诉 GGML 内核“这些维度不参与旋转”。

#### 4.6.5.2 Gemma4Model.set_gguf_parameters()：共享 KV、dual head_dim、双倍 FFN

源码路径：`./convert_hf_to_gguf.py` - `Gemma4Model.set_gguf_parameters()`（3884-3936行）

```python
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        # ① 共享 KV 层
        if (num_kv_shared := hparams.get("num_kv_shared_layers")):
            self.gguf_writer.add_shared_kv_layers(num_kv_shared)
        # ② Per-layer 输入嵌入维度
        if (pl_embd := hparams.get("hidden_size_per_layer_input")):
            self.gguf_writer.add_embedding_length_per_layer_input(pl_embd)
        # ③ Dual head_dim：global vs sliding
        head_dim = hparams.get("head_dim", 256)
        self.gguf_writer.add_key_length(head_dim)  # global
        if (head_dim_swa := hparams.get("head_dim_swa")):
            self.gguf_writer.add_key_length_swa(head_dim_swa)  # sliding
        # ④ use_double_wide_mlp：共享层 FFN 翻倍
        if hparams.get("use_double_wide_mlp"):
            n_ff = hparams.get("intermediate_size")
            n_ff_arr = [n_ff * 2 if i >= self.block_count - num_kv_shared else n_ff
                        for i in range(self.block_count)]
            self.gguf_writer.add_feed_forward_length(n_ff_arr)
        # ⑤ Sliding window pattern
        if (swa := hparams.get("sliding_window")):
            self.gguf_writer.add_sliding_window(swa)
            layer_types = hparams.get("layer_types", [])
            swa_pattern = [t == "sliding_attention" for t in layer_types]
            self.gguf_writer.add_sliding_window_pattern(swa_pattern)
```

#### 4.6.5.3 Gemma4Model.modify_tensors()：per_expert_scale → FFN_DOWN_EXP.scale

源码路径：`./convert_hf_to_gguf.py` - `Gemma4Model.modify_tensors()`（3957-3977行）

```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # ① per_dim_scale / layer_scalar 重命名
        if name.endswith("per_dim_scale"):
            name = name.replace("per_dim_scale", "scale")
        elif name.endswith("layer_scalar"):
            name = name.replace("layer_scalar", "scale")
        # ② MoE 专家缩放 → FFN_DOWN_EXP.scale
        if name.endswith("per_expert_scale"):
            name = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, bid) + ".scale"
        # ③ 跳过视觉/多模态张量
        if "vision" in name or "multi_modal" in name:
            return
        yield from super().modify_tensors(data_torch, name, bid)
```

### 4.6.6 Gemma4VisionAudioModel.audio / vision 处理差异

- audio 走 Conformer 路径，`is_audio_tensor` 区分 `audio_tower` / `embed_audio`
- vision 走 SigLIP 风格 ViT + Gemma4MultimodalEmbedder 投影头
- `_per_dim_key_scale` 后应用 `softplus(data_torch)` 变换
- `patch_embedder.input_proj` reshape 成 (n_embd, 3, patch_size, patch_size) 标准 conv2d

#### 4.6.6.1 Gemma4VisionAudioModel.is_audio_tensor()：音频张量判定

源码路径：`./convert_hf_to_gguf.py` - `Gemma4VisionAudioModel.is_audio_tensor()`（3987-4015 内）

```python
    def is_audio_tensor(self, name: str) -> bool:
        # audio_tower / embed_audio 关键字判定
        return any(k in name for k in ["audio_tower", "embed_audio"])
```

#### 4.6.6.2 Gemma4VisionAudioModel.modify_tensors()：audio softplus + vision conv2d 折叠

源码路径：`./convert_hf_to_gguf.py` - `Gemma4VisionAudioModel.modify_tensors()`（3987-4015行）

```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self.is_audio_tensor(name):
            # Audio Conformer 分支
            if name.endswith("_per_dim_key_scale"):
                data_torch = torch.nn.functional.softplus(data_torch).unsqueeze(0)
                name = name + ".weight"
            yield from super().modify_tensors(data_torch, name, bid)
        else:
            # Vision ViT 分支
            if "patch_embedder.input_proj" in name and "weight" in name:
                n_embd = data_torch.shape[0]
                patch = int((data_torch.shape[1] // 3) ** 0.5)
                data_torch = data_torch.reshape(n_embd, 3, patch, patch)
            yield from super().modify_tensors(data_torch, name, bid)
```

## 4.7 设计中的取舍

### 4.7.1 为什么 Gemma3N 关闭 norm_shift（返回 0.0），而 Gemma3 保持 1.0？

Gemma3N 使用的是 `Gemma3p5RMSNorm`，其 forward 公式已将 +1 偏移融入权重存储（即存储时就是 `1+w`），无需转换时再补偿。而 Gemma3 使用 `Gemma3RMSNorm`，forward 时计算 `w * (1 + x)`，存储的是 `x`，因此必须 +1。如果误把 Gemma3N 的 norm_shift 改为 1.0，GGML 端会得到 `1 + (1+w) = 2+w` 的错误权重，导致推理数值崩溃。

### 4.7.2 为什么 Gemma4 引入 dual head_dim（global vs sliding）而不是统一 head_dim？

Gemma4 的混合注意力架构中，全局注意力层需要完整 head_dim 以保持表达能力，而滑动窗口层可用更小 head_dim 降低 KV cache。统一 head_dim 会在滑动窗口层浪费显存，或在全局层限制表达力。dual head_dim 让运行时按 layer_types 动态选择，实现精度与效率的平衡。

### 4.7.3 为什么 Proportional RoPE 用 1e30 作为哨兵而不是 0 或负数？

RoPE 频率因子在计算中出现在分母或指数位置。1e30 极大值会使旋转角度趋近 0（即不旋转），且不会导致数值下溢或除零错误。0 会导致频率为 0（恒定旋转），负数会导致复数频率。1e30 是数值稳定且语义明确的“不旋转”标记。

### 4.7.4 Gemma3N 的 AltUp 投影为何用 3 个分支堆叠而不是 4 个？

AltUp 机制维护 4 个并行通道，但第 0 个通道是主通道（不需要额外投影），只有第 1/2/3 个通道需要学习投影矩阵。因此只需捕获 3 个投影矩阵（.0/.1/.2 对应通道 1/2/3），堆叠为 [hidden, hidden, 3]。

## 4.8 动手练习

1. **阅读 GemmaModel 的 norm.weight 补偿机制**
   阅读 ./convert_hf_to_gguf.py 第 3081-3132 行（GemmaModel.set_vocab、set_gguf_parameters、modify_tensors），回答以下问题：
   1. GemmaModel.modify_tensors 中 `if name.endswith("norm.weight"): data_torch = data_torch + 1` 是在做什么？为什么要做这一步？
   2. GemmaModel.set_vocab 中为什么要显式注入 prefix/suffix/mq/fsep/eot 这五个特殊 token？它们对应的 token ID 是多少？
   3. `add_space_prefix = False` 对 SentencePiece 词表有什么意义？如果不设置会怎样？

2. **追踪 Gemma3N AltUp 投影与 vocab padding 的完整流水线**
   阅读 ./convert_hf_to_gguf.py 第 3627-3767 行（Gemma3NModel.__init__、set_vocab、set_gguf_parameters、_stack_matrices、modify_tensors），理解 AltUp 多通道投影与 vision/audio token 嵌入 padding 的协作：
   1. `__init__` 中 `_altup_proj = [Tensor(), Tensor(), Tensor()]` 三个空张量的目的是什么？为什么用空张量占位而不是 None？
   2. 在 modify_tensors 中，按 `.0./.1./.2.` 三分支分写入 _altup_proj 缓冲，当三个槽都被填满时如何触发 `_stack_matrices`？最终产出的 3D 张量 shape 是什么？
   3. `embed_tokens.weight` padding 到 262400 的逻辑中，`vocab_size_per_layer_input = 262144` 是来自哪个字段？为什么 Gemma3N 必须把视觉 token 放在词表末尾而不是用专用 embedding？
   4. `sliding_window_pattern` 与 `layer_types` 的对齐逻辑具体如何实现？请画出循环过程。
   5. set_vocab 中为什么要先删除 vocab_size_per_layer_input 后恢复？

3. **剖析 Gemma4 共享 KV 与 Proportional RoPE 的实现细节**
   阅读 ./convert_hf_to_gguf.py 第 3823-3977 行（Gemma4Model.set_vocab、set_gguf_parameters、generate_extra_tensors、modify_tensors），回答：
   1. Gemma4Model.set_vocab 为什么要用 `gguf.LlamaHfVocab(self.dir_model)` 而不是 `_set_vocab_sentencepiece`？visible_tokens 集合的设计目的是什么？
   2. 在 set_gguf_parameters 中，`add_shared_kv_layers(num_kv_shared_layers)` 与 `use_double_wide_mlp` 如何协同工作？为什么 shared KV 层的 FFN 要翻倍？
   3. `generate_extra_tensors` 中 `1e30` 哨兵的物理含义是什么？partial_rotary_factor=0.25 时，`n_rot_full` 与 `n_unrot_full` 分别如何计算？
   4. modify_tensors 中 `.per_expert_scale` 名称如何被改写为 FFN_DOWN_EXP 的 .scale 后缀张量？这个映射是为了配合 GGML的哪个哪个？
   5. dual head_dim 设计中，global_head_dim 与 head_dim 分别如何通过 add_key_length / add_key_length_swa 写入？

4. **对比 Gemma3 与 Gemma3N 的 norm_shift 差异**
   阅读 ./convert_hf_to_gguf.py 中 Gemma3Model.norm_shift (3242-3243 行) 与 Gemma3NModel.norm_shift 实现，回答：
   1. Gemma3Model.norm_shift 返回 1.0，而 Gemma3NModel.norm_shift 返回 0.0，背后的数学原因是什么？请用 Gemma3RMSNorm 与 Gemma3p5RMSNorm 的 forward 公式说明。
   2. 如果你误把 Gemma3N 的 norm_shift 改为 1.0，会出现什么后果？GGML 端的权重会发生什么数值变化？
   3. 请在代码中搜索其他 Gemma 家族成员的 norm_shift 实现（GemmaModel、Gemma2Model、Gemma4Model），画出一张 norm_shift 值对照表。
   4. Gemma3VisionModel.modify_tensors 中只对 `soft_emb_norm.weight` 单独 +1（其他 norm 不修正），背后的设计原则是什么？

5. **解读 Gemma3nVisionAudioModel 与 Gemma4VisionAudioModel 的多模态分流**
   阅读 ./convert_hf_to_gguf.py 中 Gemma3nVisionAudioModel (3573-3605 行) 与 Gemma4VisionAudioModel (3987-4015 行) 的 modify_tensors / custom_map / is_audio_tensor：
   1. Gemma3nVisionAudioModel.block_tensor_mapping 字典的设计哲学是什么？为什么 timm 风格的 MobileNetV5 不走标准 tensor_mapping.py？
   2. custom_map 如何解析 `model.vision_tower.timm_model.blocks.{bid}.{sid}.suffix` 的双层路径？
   3. Gemma4VisionAudioModel.is_audio_tensor 如何区分 audio_tower / embed_audio 与 vision_tower？两条路径分别走什么 modify_tensors 分支？
   4. Gemma4VisionAudioModel.modify_tensors 中 audio 塔的 `_per_dim_key_scale` 如何应用 softplus？vision 塔的 patch_embedder 如何 reshape 为 2D→4D conv 格式？
   5. 对比 Gemma3VisionModel（只处理 vision）、Gemma3nVisionAudioModel（vision+audio 双塔）、Gemma4VisionAudioModel（vision+audio 双塔），它们在 set_gguf_parameters 中分别写入了哪些 projector type？

## 4.9 本章小结

这一章中我们学习了 Gemma 家族四代模型在 GGUF 转换中的演进脉络。首先分析了 Gemma 基础款的 norm.weight +1 反向补偿机制，这是由 Gemma RMSNorm 的前向公式 `weight * (1+w)` 决定的存储约定。其次讲解了 Gemma3 引入的因果/非因果双模式、vocab OOV 裁剪与滑动窗口元数据写入。接着深入剖析了 Gemma3N 的核心创新：AltUp 多通道投影的三缓冲捕获与 3D 堆叠、vision/audio token 嵌入的 262400 padding 机制、Activation Sparsity 的 normal.icdf 统计还原、以及 MobileNetV5 timm 双层路径的 custom_map 模板化映射。然后详细解读了 Gemma4 的共享 KV 层标记、dual head_dim（global/sliding 双轨）、use_double_wide_mlp 触发的 FFN 翻倍、Proportional RoPE 的 1e30 哨兵设计，以及多模态投影器中 audio Conformer 的 softplus 变换与 vision patch_embedder 的 2D→4D conv 折叠。最后对比了各代 norm_shift 的数学差异与多模态转换器的分流策略。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| GemmaModel.norm.weight+1 | HF forward 在 RMSNorm 时乘 (1+w)，存储时已减去 1；转换时必须 +1 反向补偿给 GGML 端 |
| Gemma3Model.vocab 截断 | embed_tokens.weight 仅保留真实 vocab 大小（去掉 OOV），配合 is_causal/sliding_window 写入元数据 |
| Gemma3NModel._altup_proj 堆叠 | 捕获 .0./.1./.2. 三组 AltUp 投影矩阵，_stack_matrices 后产出 model.altup_*_projections.weight 3D 张量 |
| Gemma3NModel.vocab_size=262400 | vocab_size_per_layer_input=262144 + 256 个 vision/audio token，embed_tokens 末尾 zero-pad |
| Gemma3NModel.norm_shift=0.0 | Gemma3p5RMSNorm 已隐含在 forward 中，存储时不再需要补偿偏移 |
| Gemma3nVisionAudioModel.block_tensor_mapping | MobileNetV5 timm blocks.{bid}.{sid}.suffix 双层路径模板化映射到 v.blk.{bid}.{sid}.suffix |
| Gemma4Model.shared_kv_layers | add_shared_kv_layers 标记末尾 N 层共享 KV cache，配合 use_double_wide_mlp 让 FFN 翻倍 |
| Gemma4Model.dual head_dim | global_head_dim (add_key_length) 与 head_dim (add_key_length_swa) 并存，运行时按 layer_types 选择 |
| Gemma4Model.Proportional RoPE FREQS | partial_rotary_factor=0.25，生成 (1.0)*n_rot + (1e30)*n_unrot 的 ROPE_FREQS，让 ggml 跳过未旋转位置 |
| Gemma4Model.per_expert_scale → FFN_DOWN_EXP.scale | MoE 路由的 per_expert_scale 标量被映射为专家 down 投影的的 .scale 后缀张量 |
| Gemma3NModel.Activation Sparsity | normal.icdf(sparsity) 反推 multiplier 写入 add_activation_sparsity_scale，用于推理跳过低激活维度 |
| Gemma3Model.sliding_window_pattern | 遍历 layer_types 生成布尔列表与 add_sliding_window_pattern 严格对齐，ISWA 双缓存策略依赖此标记 |
| Gemma3VisionModel.soft_emb_norm+1 | Gemma3 RMSNorm forward 公式 (1+w)，在 Gemma3 mmproj 中只有 soft_emb_norm 需要 +1，其他 norm 来自 SigLIP 已正确 |
| Gemma3NModel.set_vocab 覆盖 | 临时删除 vocab_size_per_layer_input 强制读取 vocab_size=262400，让 SentencePiece/GPT2 写出 262400 行 tokens |
| Gemma4VisionAudioModel.per_dim_key_scale softplus | audio 塔的 per_dim_key_scale 后用 softplus 激活并加 .weight 后缀映射为 F32 张量 |
| Gemma4VisionAudioModel.patch_embedder 2D→4D | patch_embedder.input_proj 从 [n_embd, patch*patch*3] reshape 成 [n_embd, 3, patch, patch] 标准 conv2d 格式 |
| Gemma3NModel.embed_tokens_per_layer padding | 不仅embed_tokens.weight，embed_tokens_per_layer 同样按 vocab_size - current_size 末尾 zero-pad |
| Gemma4VisionAudioModel.is_audio_tensor | 通过 audio_tower/embed_audio 关键字判定走 audio Conformer 分支，否则走 vision ViT 分支 |

下一章中，我们将学习高级量化与 NVFP4 路径 —— 探索「权重压缩的特种部队」。
