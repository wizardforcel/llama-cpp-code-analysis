
# 第15章 词汇表与分词系统 —— 文本与 Token 的"双向桥梁"

## 学习目标

- 理解UTF-8与Unicode码点的双向转换机制及其错误处理策略
- 掌握NFD规范化、NFD折叠表及大小写映射的底层实现
- 熟悉unicode_regex_split分发器如何为GPT-2/LLaMA3/Kimi-K2/AFMOE等模型定制正则分词
- 理解llama_vocab::impl的pimpl模式与GGUF元数据加载流水线
- 掌握SPM优先级队列合并与BPE rank表合并的差异
- 了解特殊token（EOT/EOM/EOG/字节回退/FIM）的自动检测与分类逻辑
- 能阅读detokenize三遍清理算法并修改clean_spaces策略
- 阅读UGM分词的Viterbi DP算法及XCDA压缩双数组解码
- 掌握PLaMo2的Trie展平与动态规划回溯实现

## 15.1 生活类比

想象llama.cpp的分词系统是一家**国际快递公司的分拣中心**，负责把不同语言、不同形式的「包裹」(文本)拆成标准化的「快递箱」(token)。UTF-8字节流 = 快递车上还没拆包的包裹（一串原始字节）；Unicode码点(codepoint) = 拆包后的单个物品（每个字符）；NFD规范化 = 把「café」统一拆成「cafe+´」，避免「é」和「e+´」两种写法混淆；GPT-2字节回退 = 给每个0-255的字节分配一个Unicode占位字符，保证任何字节都能被tokenize；SPM优先级队列 = 按「分数」（对数概率）从高到低排序的优先队列，先合并高分token；BPE rank表 = 训练时按频次排序的合并规则表，rank小（出现早）的优先合并；特殊token分区 = 把<|im_start|>、 等「快递面单」预先切出来，剩下的才是普通文本；detokenize三遍清理 = 快递员取件时：去掉「 ?」前的多余空格、合并「 ' 」两侧空格、处理「 's」的撇号缩写；naive_trie字典树 = 快递分拣柜，每层货架放一种字符前缀，沿着货架最快找到匹配；XCDA紧凑双数组 = 智能货架编号系统，用一个字节就能定位任何前缀对应的归一化字符串。就像分拣中心需要协调传送带、扫码枪、堆垛机一样，llama.cpp需要在Unicode编码、词表匹配、特殊token切分之间精确配合。特别token就是「特殊面单」：不能被通用规则误处理，必须优先识别为完整单元。

## 15.2 源码地图

src/unicode.h
├── struct unicode_cpt_flags              # 码点类别位标志(NUMBER/LETTER/HAN/NFD...)
│   ├── enum UNDEFINED/NUMBER/LETTER/SEPARATOR/ACCENT_MARK/PUNCTUATION/SYMBOL/CONTROL
│   ├── enum WHITESPACE/LOWERCASE/UPPERCASE/NFD
│   ├── unicode_cpt_flags()               # 字节序兼容构造函数
│   ├── as_uint()                          # 合并为uint16
│   └── category_flag()                    # 提取类别位
├── size_t unicode_len_utf8()             # 查表法获取UTF-8序列长度
├── std::string unicode_cpt_to_utf8()      # 码点→UTF-8字节
├── uint32_t unicode_cpt_from_utf8()      # UTF-8→码点, 含边界检查
├── std::vector<uint32_t> unicode_cpts_from_utf8()    # 字节流→码点流, 错误用0xFFFD替换
├── std::vector<uint32_t> unicode_cpts_normalize_nfd() # NFD规范化
├── unicode_cpt_flags unicode_cpt_flags_from_cpt()  # O(1)查码点类别表
├── unicode_cpt_flags unicode_cpt_flags_from_utf8() # 字符串→类别
├── std::string unicode_byte_to_utf8()    # GPT-2字节回退: byte→占位字符
├── uint8_t unicode_utf8_to_byte()       # 反向映射
├── uint32_t unicode_tolower()            # 二分查表大小写转换
├── bool unicode_cpt_is_han()            # CJK汉字区间判断(9个Unicode block)
└── unicode_regex_split()                  # 顶层分词正则分发器

src/unicode.cpp
├── size_t unicode_len_utf8()             # 16项查表
├── unicode_cpts_to_utf8()                # 辅助:码点流→字符串
├── unicode_cpt_from_utf8()                # 多字节解码, 边界检查+异常
├── unicode_cpts_normalize_nfd()           # std::upper_bound二分查找折叠
├── unicode_cpts_from_utf8()               # try-catch包裹, 失败回退到0xFFFD
├── unicode_cpt_flags_array()              # 静态数组构造, 区间→flags填充
├── unicode_byte_to_utf8_map()             # 静态表构造<0xXX>与单字节映射
├── unicode_utf8_to_byte_map()             # 反向映射表
├── unicode_byte_encoding_process()        # GPT-2字节编码(用于BPE)
├── unicode_regex_split_custom_gpt2()      # GPT-2预分词器
├── unicode_regex_split_custom_llama3()    # LLaMA-3预分词器
├── unicode_regex_split_custom_kimi_k2()   # Kimi-K2预分词器
├── unicode_regex_split_custom_afmoe()     # AFMOE预分词器
├── unicode_regex_split_custom_newlines()  # Gemma4专用: 仅按\n切分
├── unicode_regex_split_custom()           # 总分发: regex字符串→具体实现
├── unicode_regex_split_stl()              # std::regex回退(支持collapsed类别替换)
└── unicode_regex_split()                  # unicode类别折叠 + 多regex链式调用

src/unicode-data.h
├── struct range_nfd                      # NFD区间三元组(first/last/nfd)
├── MAX_CODEPOINTS = 0x110000             # Unicode最大码点数
└── 外部常量声明: ranges_flags / whitespace / lowercase / uppercase / nfd

src/unicode-data.cpp
├── unicode_ranges_flags                  # 码点区间→类别位(用于cpt_flags数组)
├── unicode_set_whitespace                # 24个空白码点集合(\s)
├── unicode_map_lowercase/uppercase       # 升序pair列表,支持二分查找
└── unicode_ranges_nfd                    # NFD分解区间(查找父字符→基本字符)

src/llama-vocab.h
├── enum llama_vocab_pre_type             # 50+种预分词器类型枚举
├── struct llama_vocab::token_data        # text/score/attr三元组
└── class llama_vocab (pimpl)             # 对外接口(隐藏数千行实现)

src/llama-vocab.cpp
├── struct naive_trie                     # O(n)最长前缀匹配树(RWKV/PLaMo2用)
│   ├── insert()                           # 递归插入字符到子节点
│   ├── get_longest_prefix()               # 沿树向下走最长匹配
│   └── traverse()                         # 单字符向下走
├── struct llm_symbol                     # UTF-8链子节点(prev/next/text/n)
├── struct llm_tokenizer_spm_session      # SPM合并: 优先级队列+score
│   ├── tokenize()                         # 初始化链表→种子队列→贪心合并→resegment
│   ├── resegment()                        # 兜底: 查rev_merge或按字节回退
│   └── try_add_bigram()                   # 添加pair并更新work_queue
├── struct llm_bigram_spm                 # SPM bigram: left/right/score/size
│   └── struct comparator                 # (score<) || (score== && left>)
├── struct llm_tokenizer_bpe              # 按pre_type绑定regex_expressions(40+种)
│   └── llm_tokenizer_bpe(vocab)          # 构造函数: switch分发regex_exprs
├── struct llm_bigram_bpe                 # BPE bigram: left/right/text/rank/size
│   └── struct comparator                 # rank> || rank== && left>
├── struct llama_priority_queue           # pop_move: 移动语义的pop
├── struct llm_tokenizer_bpe_session      # BPE合并: rank表+跨word链表
│   ├── append()                           # 追加到output
│   ├── append_bos()/append_eos()         # 自动加BOS/EOS
│   ├── check_double_bos_eos()            # 检测重复BOS/EOS并警告
│   ├── tokenize()                         # regex切分word→char链表→贪心合并
│   └── add_new_bigram()                   # 查vocab.find_bpe_rank()
├── struct llm_tokenizer_wpm_session      # WordPiece: 最长前缀匹配+UNK兜底
│   ├── tokenize()                         # 循环最长前缀+UNK兜底
│   ├── preprocess()                       # NFD+小写+标点/汉字分词
│   └── is_chinese_char()                 # 8个CJK区间判断
├── struct llm_tokenizer_ugm_session      # Unigram: Viterbi算法
│   ├── tokenize()                         # tokenization_results DP表+回溯
│   ├── normalize()                        # 调用XCDA查找最长前缀替换
│   ├── normalize_prefix()                # XCDA遍历,返回长度
│   └── struct xcda_array_view            # BASE/LCHECK/LEAF位运算解码
├── struct llm_tokenizer_rwkv_session     # RWKV: naive_trie匹配+转义解码
│   └── tokenize()                         # trie遍历找最长匹配token
├── llama_unescape_rwkv_token()           # RWKV的\xNN转义解析
├── struct llm_tokenizer_plamo2           # PLaMo2: 展平Trie+DP
│   ├── build()                            # 构建tokens_/bytes_/to_suffix_id_/table_
│   └── encode()                           # scores/path DP, 含字节回退
├── struct llm_tokenizer_plamo2_session   # 包装plamo2.encode
│   └── tokenize()                         # 调用tokenizer.encode()并追加
├── struct fragment_buffer_variant        # TOKEN/RAW_TEXT片段容器
├── struct llama_vocab::impl              # pimpl实现
│   ├── load()                             # 从GGUF读tokenizer.model/pre/list/score/type
│   ├── init_tokenizer()                  # 根据type构造具体tokenizer
│   ├── tokenizer_st_partition()          # 特殊token切分(按text长度倒序匹配)
│   ├── token_to_piece_for_cache()        # 构造token→piece缓存(预生成)
│   ├── tokenize()                         # 顶层分发: fragment_buffer→各session
│   ├── token_to_piece()                   # 按vocab type还原token文本(▁/字节解码)
│   ├── detokenize()                       # 调token_to_piece + clean_spaces三遍清理
│   ├── print_info()                       # 打印BOS/EOS/FIM/EOG等所有特殊token
│   ├── is_normal/is_unknown/is_control/is_byte/is_user_defined/is_unused/is_eog()
│   ├── token_to_byte()                    # <0xXX>字符串→单字节
│   ├── token_get_attr()                   # 返回attr位标志
│   ├── get_type()                         # 返回vocab type
│   └── type_name()                        # "SPM"/"BPE"/"WPM"/"UGM"/"RWKV"/"PLaMo2"
└── free functions
    ├── llama_escape_whitespace           # ' ' ↔ \xE2\x96\x81(▁)
    ├── llama_unescape_whitespace         # 反向解码(SPM风格)
    └── llama_decode_text                 # BPE字节token→原始字节
    ├── llama_tokenize/llama_detokenize/llama_token_to_piece  # C API入口
    └── llama_vocab_bos/llama_vocab_eos/... # 特殊token C API
    ├── naive_trie.*                      # 字典树增删查
    └── llama_priority_queue.pop_move()   # 移动pop

## 15.3 Unicode 支持：编码与规范化 —— 文本与 Token 的「第一道翻译官」

### 15.3.1 为什么需要 utf8 ↔ codepoint 的双向通道？

Tokenizer 的最小操作单位是 Unicode 码点而非字节，因此必须先把 UTF-8 字节流拆成码点序列。通过 `unicode_cpt_to_utf8()` / `unicode_cpt_from_utf8()` 实现字节与码点的互转，错误输入用 `0xFFFD` 替换避免异常向上传播。NFD 规范化（`unicode_cpts_normalize_nfd`）解决「音标复合字符」在不同源数据中的形态差异，是 SPM/WPM 分词器稳定性的前提。大小写转换（`unicode_tolower`）支持 LLaMA3 等大小写不敏感的预分词正则（如 `'(?i:'s|...)`）。汉字识别（`unicode_cpt_is_han`）为 Kimi-K2 等模型提供 CJK 字符的快速判别，配合正则 `\p{Han}+` 完成中文分词。

### 15.3.2 unicode_cpt_from_utf8()：UTF-8 字节流到码点的解码器

源码路径：`src/unicode.cpp` - `unicode_cpt_from_utf8()`（第 33-66 行）

```cpp
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    // ① 断言偏移量在字符串范围内
    assert(offset < utf8.size());
    // ② 单字节 ASCII (0xxxxxxx)
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    // ③ 非法起始字节 (10xxxxxx)
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    // ④ 双字节字符 (110xxxxx 10xxxxxx)
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    // ⑤ 三字节字符 (1110xxxx 10xxxxxx 10xxxxxx)
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    // ⑥ 四字节字符 (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    // ⑦ 超过 4 字节或非法起始字节
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}
```

这段代码实现了标准的 UTF-8 解码逻辑：根据首字节的高位模式判断字符长度（1~4 字节），逐位拼装码点，并严格校验后续字节是否符合 `10xxxxxx` 模式。任何不合规输入直接抛出异常，由上层 `unicode_cpts_from_utf8` 捕获并替换为替换字符 `U+FFFD`。

### 15.3.3 unicode_cpts_from_utf8()：字节流到码点流的安全通道

源码路径：`src/unicode.cpp` - `unicode_cpts_from_utf8()`（第 766-780 行）

```cpp
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());          // ① 预留空间，避免频繁扩容
    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            // ② 尝试解码下一个码点
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        }
        catch (const std::invalid_argument & /*ex*/) {
            // ③ 解码失败：跳过 1 字节，记录替换字符 U+FFFD
            ++offset;
            result.emplace_back(0xFFFD);  // replacement character
        }
    }
    return result;
}
```

这段代码定义了「字节流 → 码点流」的安全转换管道。它逐个调用 `unicode_cpt_from_utf8`，一旦遇到非法 UTF-8 序列（如截断、错误续字节），捕获异常、仅前进 1 字节、写入 `0xFFFD`，保证流程绝不中断——这正是工业级分词器必须具备的鲁棒性。

### 15.3.4 unicode_cpts_normalize_nfd()：NFD 规范化的二分查找实现

源码路径：`src/unicode.cpp` - `unicode_cpts_normalize_nfd()`（第 754-764 行）

```cpp
std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    // ① 自定义比较器：码点 < 区间起始
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        // ② std::upper_bound 找到第一个 first > cpt 的区间，再 -1 得到可能包含 cpt 的区间
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        // ③ 若 cpt 在 [first, last] 内，替换为 nfd 基本字符；否则保持原样
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}
```

这段代码实现了 NFD（Canonical Decomposition）规范化：将 `é` (U+00E9) 等预组合字符拆解为 `e` (U+0065) + `´` (U+0301)。核心数据结构 `unicode_ranges_nfd` 是按 `first` 升序排列的区间表（来自 `unicode-data.cpp`），利用 `std::upper_bound` 实现 O(log N) 查找，再做边界校验。这保证了无论输入文本使用预组合还是分解形式，进入分词器前都被统一成分解形式，消除歧义。

### 15.3.5 unicode_tolower()：大小写映射的二分查表

源码路径：`src/unicode.cpp` - `unicode_tolower()`（第 807-817 行）

```cpp
uint32_t unicode_tolower(uint32_t cpt) {
    // ① 在升序 unicode_map_lowercase 中二分查找
    auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cpt,
        [](const std::pair<uint32_t, uint32_t> & pair, uint32_t value) {
            return pair.first < value;
        });
    // ② 命中则返回映射后的小写码点，否则原样返回
    if (it != unicode_map_lowercase.end() && it->first == cpt) {
        return it->second;
    }
    return cpt;  // ③ 无映射则返回原码点
}
```

这段代码提供 Unicode 感知的小写转换。`unicode_map_lowercase` 是 `(原码点, 小写码点)` 的升序数组（含 ASCII 及扩展字母），二分查找 O(log N) 完成映射。LLaMA-3 的预分词正则 `(?i:'s|...)` 依赖此函数实现大小写不敏感匹配。

### 15.3.6 unicode_cpt_is_han()：CJK 汉字区间判断

源码路径：`src/unicode.cpp` - `unicode_cpt_is_han()`（第 916-943 行）

```cpp
bool unicode_cpt_is_han(uint32_t cpt) {
    if (cpt >= 0x4E00 && cpt <= 0x9FFF) return true;   // CJK Unified Ideographs
    if (cpt >= 0x3400 && cpt <= 0x4DBF) return true;   // Extension A
    if (cpt >= 0x20000 && cpt <= 0x2A6DF) return true; // Extension B
    if (cpt >= 0x2A700 && cpt <= 0x2B73F) return true; // Extension C
    if (cpt >= 0x2B740 && cpt <= 0x2B81F) return true; // Extension D
    if (cpt >= 0x2B820 && cpt <= 0x2CEAF) return true; // Extension E
    if (cpt >= 0x2CEB0 && cpt <= 0x2EBEF) return true; // Extension F
    if (cpt >= 0xF900 && cpt <= 0xFAFF) return true;   // Compatibility Ideographs
    if (cpt >= 0x2F800 && cpt <= 0x2FA1F) return true; // Compatibility Supplement
    return false;
}
```

这段代码覆盖了 Unicode 标准中 9 个汉字相关区块。Kimi-K2 的预分词器 `unicode_regex_split_custom_kimi_k2` 直接调用它作为 `\p{Han}+` 的高效实现，避免了通用正则引擎的开销。

### 15.3.7 unicode_cpt_flags_array()：全码点属性表的 O(1) 构建

源码路径：`src/unicode.cpp` - `unicode_cpt_flags_array()`（第 96-122 行）

```cpp
static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    // ① 初始化 1.1M 个 UNDEFINED
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    // ② 遍历区间表，批量填充类别位
    assert(unicode_ranges_flags.begin()[0].first == 0);
    assert(unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];
        const auto range_end = unicode_ranges_flags.begin()[i];
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    // ③ 叠加显式集合：空白、大小写、NFD
    for (auto cpt : unicode_set_whitespace) cpt_flags[cpt].is_whitespace = true;
    for (auto p : unicode_map_lowercase)  cpt_flags[p.second].is_lowercase = true;
    for (auto p : unicode_map_uppercase)  cpt_flags[p.second].is_uppercase = true;
    for (auto &range : unicode_ranges_nfd) cpt_flags[range.nfd].is_nfd = true;

    return cpt_flags;
}
```

这段代码在程序启动时一次性构建全码点属性表（约 110 万项）。先按区间批量填充大类（Letter/Number/Punctuation…），再叠加显式集合（Whitespace/Lowercase/Uppercase/NFD），最终实现 `unicode_cpt_flags_from_cpt` 的 O(1) 常数时间查询。分词器预处理阶段每字符都会调用此表，性能收益巨大。

### 15.3.8 unicode_byte_to_utf8 / unicode_utf8_to_byte：GPT-2 字节回退双向映射

源码路径：`src/unicode.cpp` - `unicode_byte_to_utf8_map()`（第 124-140 行）与 `unicode_utf8_to_byte_map()`（第 142-158 行）

```cpp
// byte → Unicode 占位字符
static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    // ① 可打印 ASCII 直接映射
    for (int ch = 0x21; ch <= 0x7E; ++ch) map[ch] = unicode_cpt_to_utf8(ch);
    // ② 扩展拉丁字符直接映射
    for (int ch = 0xA1; ch <= 0xAC; ++ch) map[ch] = unicode_cpt_to_utf8(ch);
    for (int ch = 0xAE; ch <= 0xFF; ++ch) map[ch] = unicode_cpt_to_utf8(ch);
    // ③ 剩余字节映射到 Private Use Area U+E000~U+E0FF
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

// 反向映射：UTF-8 占位字符 → byte
static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    // 同理构建反向表
    ...
}
```

这两段代码构建了 GPT-2 风格的字节级回退机制：将任意 0-255 字节一一映射到 256 个互不冲突的 Unicode 码点（可打印字符直映，其余落入 PUA 区）。BPE 分词器在遇到词表外字节序列时，先用此映射把字节「伪装」成合法 Unicode 字符，再按常规流程分词，保证**任意二进制数据皆可编码**。

### 15.3.9 unicode_regex_split()：顶层正则分词分发器

源码路径：`src/unicode.cpp` - `unicode_regex_split()`（第 946-1088 行）

```cpp
std::vector<std::string> unicode_regex_split(const std::string & text,
                                             const std::vector<std::string> & regex_exprs,
                                             bool byte_encode) {
    // ① 检测是否包含 Unicode 类别转义（如 \p{L}），决定是否需要 collapsed 映射
    bool need_collapse = false;
    for (const auto & regex_expr : regex_exprs) {
        for (const auto & ucat : k_ucat_enum) {
            if (regex_expr.find(ucat.first) != std::string::npos) {
                need_collapse = true; break;
            }
        }
    }

    // ② 将原文本转为码点数组
    const auto cpts = unicode_cpts_from_utf8(text);

    // ③ 若需类别匹配，构建 collapsed 字符串：每码点映射为 1 字节类别标签
    std::string text_collapsed;
    if (need_collapse) {
        text_collapsed.resize(cpts.size());
        for (size_t i = 0; i < cpts.size(); ++i) {
            if (cpts[i] < 128) { text_collapsed[i] = cpts[i]; continue; }
            const auto flags = unicode_cpt_flags_from_cpt(cpts[i]);
            if (flags.is_whitespace) text_collapsed[i] = 0x0B;
            else if (k_ucat_cpt.count(flags.category_flag()))
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            else text_collapsed[i] = 0xD0;
        }
    }

    // ④ 初始偏移：整个文本作为一个片段
    std::vector<size_t> bpe_offsets = { cpts.size() };

    // ⑤ 依次应用每条正则，细化偏移切分
    for (const auto & regex_expr : regex_exprs) {
        // 5.1 优先尝试手写的 custom 实现（GPT-2/LLaMA3/Kimi-K2/AFMOE/Newlines）
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);
        if (!tmp.empty()) { bpe_offsets = std::move(tmp); continue; }

        // 5.2 回退到 std::regex（需处理 collapsed 类别替换）
        try {
            bool use_collapsed = false;
            for (const auto & ucat : k_ucat_enum)
                if (regex_expr.find(ucat.first) != std::string::npos) { use_collapsed = true; break; }

            if (use_collapsed) {
                // 将 \p{L} 等替换为 collapsed 字符类
                std::string regex_expr_collapsed = ...;
                bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                // 宽字符正则
                std::wstring wregex_expr(cpts_regex.begin(), cpts_regex.end());
                std::wstring wtext(cpts.begin(), cpts.end());
                // 空白字符统一映射为 0x0B
                for (size_t i = 0; i < wtext.size(); ++i)
                    if (wtext[i] > 0x7F && unicode_cpt_flags_from_cpt(wtext[i]).is_whitespace)
                        wtext[i] = 0x0B;
                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error & e) { throw std::runtime_error("Failed to process regex"); }
    }

    // ⑥ 根据最终偏移从码点数组重组词元字符串
    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size());
    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i)
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        start += offset;
    }

    // ⑦ 可选：GPT-2 字节编码
    if (byte_encode) return unicode_byte_encoding_process(bpe_words);
    return bpe_words;
}
```

这段代码是**预分词管道的总调度器**。它接收模型专属的正则表达式列表（如 GPT-2 的 9 条、LLaMA-3 的 10 条），按顺序链式切分文本：先按第一条正则把文本切成粗粒度片段，再在每个片段内应用第二条正则，以此类推。核心创新在于 **collapsed 映射**：将每个 Unicode 码点压缩为 1 字节类别标签（Letter→'L', Number→'N', Whitespace→0x0B...），使 `std::regex` 能以 ASCII 字符类语义匹配 Unicode 类别，兼顾了通用性与性能。手写的 `custom_*` 实现（GPT-2、LLaMA-3、Kimi-K2、AFMOE、Newlines）则完全绕过正则引擎，直接操作码点数组，针对热点模型实现极致加速。

---

## 15.4 正则表达式分词 —— 模型预分词的「自定义加速器」

### 15.4.1 为什么需要为不同模型定制正则分词？

标准 `std::regex` 对 Unicode 类别（`\p{L}` 等）支持有限，且性能较差；每种模型有独特的预分词规则。GPT-2 分词器用 `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` 把文本切成单词、数字、标点三大类。LLaMA-3 在 GPT-2 基础上增加数字定长（`\p{N}{1,3}`）、行尾换行处理、大小写不敏感等扩展。Kimi-K2 用 Han 字符优先匹配，并支持大小写不敏感的英文子词（含撇号缩写）。AFMOE 数字按 `digit_count % 3` 分组（1/2 + 3+3+...），模拟十进制分组的可读性。Gemma4 仅按 `\n` 切分（BPE 内部不允许 token 跨换行）。通过 `unicode_regex_split_custom` 分发器自动选择对应实现，未匹配时回退到 `std::regex`。

### 15.4.2 unicode_regex_split_custom_gpt2()：GPT-2 经典预分词的手写实现

源码路径：`src/unicode.cpp` - `unicode_regex_split_custom_gpt2()`（第 176-270 行）

```cpp
static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string & text,
                                                           const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        start = offset_end;

        size_t _prev_end = offset_ini;
        auto _add_token = [&](const size_t end) -> size_t {
            size_t len = end - _prev_end;
            if (len > 0) bpe_offsets.push_back(len);
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/) {
            const uint32_t cpt = cpts[pos];
            const auto flags = unicode_cpt_flags_from_cpt(cpt);

            // 1. 撇号缩写 's 't 'm 'd (2字符) / 're 've 'll (3字符)
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = cpts[pos+1];
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2); continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = cpts[pos+2];
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3); continue;
                    }
                }
            }

            // 2. 可选空格 + 字母序列
            auto flags2 = (cpt == ' ' ? unicode_cpt_flags_from_cpt(cpts[pos+1]) : flags);
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) flags2 = unicode_cpt_flags_from_cpt(cpts[++pos]);
                _add_token(pos); continue;
            }

            // 3. 可选空格 + 数字序列
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) flags2 = unicode_cpt_flags_from_cpt(cpts[++pos]);
                _add_token(pos); continue;
            }

            // 4. 可选空格 + 非空白非字母非数字序列
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint())
                    flags2 = unicode_cpt_flags_from_cpt(cpts[++pos]);
                _add_token(pos); continue;
            }

            // 5. 空白处理：连续空白、末尾空白、单个空白
            size_t num_whitespaces = 0;
            while (unicode_cpt_flags_from_cpt(cpts[pos+num_whitespaces]).is_whitespace) num_whitespaces++;

            if (num_whitespaces > 1 && cpts[pos+num_whitespaces] != 0xFFFFFFFF) {
                pos += num_whitespaces - 1; _add_token(pos); continue;  // \s+(?!\S)
            }
            if (num_whitespaces > 0) { pos += num_whitespaces; _add_token(pos); continue; } // \s+

            // 6. 兜底：单字符
            _add_token(++pos);
        }
    }
    return bpe_offsets;
}
```

这段代码**完全手写**了 GPT-2 的预分词正则逻辑，逐字符扫描码点数组，按优先级依次尝试匹配：撇号缩写 → 字母词 → 数字词 → 标点符号串 → 空白序列。完全避免了正则引擎的编译与匹配开销，且对 Unicode 类别（`is_letter`/`is_number`/`is_whitespace`）直接查 `unicode_cpt_flags` 表，实现了**零分配、零回溯、O(N) 线性扫描**。

### 15.4.3 unicode_regex_split_custom_llama3()：LLaMA-3 的增强预分词

源码路径：`src/unicode.cpp` - `unicode_regex_split_custom_llama3()`（第 272-378 行）

```cpp
// 关键差异点：
// 1. 撇号缩写使用 unicode_tolower 实现大小写不敏感
if (cpt == '\'' && pos+1 < offset_end) {
    uint32_t cpt_next = unicode_tolower(cpts[pos+1]);
    ...
}
// 2. 字母词模式：[^\r\n\p{L}\p{N}]?\p{L}+ （允许非字母非数字非换行的前导字符）
if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
    if (flags.is_letter || _get_flags(pos+1).is_letter) { ... }
}
// 3. 数字限长：\p{N}{1,3} （每 3 位切一刀）
if (flags.is_number) {
    size_t ini = pos;
    while (_get_flags(pos).is_number) {
        if (++pos - ini >= 3) { _add_token(pos); ini = pos; }
    }
    _add_token(pos); continue;
}
// 4. 显式处理 \r\n 作为独立 token
size_t last_end_r_or_n = 0;
while (_get_flags(pos+num_whitespaces).is_whitespace) {
    uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
    if (cpt2 == '\r' || cpt2 == '\n') last_end_r_or_n = pos + num_whitespaces + 1;
    num_whitespaces++;
}
if (last_end_r_or_n > 0) { pos = last_end_r_or_n; _add_token(pos); continue; }
```

这段代码在 GPT-2 基础上做了三大增强：① 大小写不敏感的撇号缩写（调用 `unicode_tolower`）；② 数字每 3 位强制切分（`\p{N}{1,3}`），避免超长数字 token；③ 显式识别 `\r\n` 换行序列作为独立 token，改善代码/结构化文本的分词质量。这些细微差别直接影响下游 BPE 合并的输入单元，进而影响模型对数字、代码、多语言文本的建模能力。

### 15.4.4 unicode_regex_split_custom_kimi_k2()：Kimi-K2 的中文优先分词

源码路径：`src/unicode.cpp` - `unicode_regex_split_custom_kimi_k2()`（第 415-590 行）

```cpp
// 核心逻辑：优先匹配汉字 \p{Han}+
if (unicode_cpt_is_han(cpt)) {
    while (unicode_cpt_is_han(_get_cpt(pos))) pos++;
    _add_token(pos); continue;
}

// 英文词模式：排除汉字的 \p{L} 序列 + 可选缩写
bool is_letter_pattern = (flags.is_letter && !unicode_cpt_is_han(cpt)) ||
    (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number) &&
     _get_flags(pos+1).is_letter && !unicode_cpt_is_han(_get_cpt(pos+1)));

if (is_letter_pattern) {
    // 可选前导非字母字符
    // 后续字母序列（排除汉字）
    // 可选缩写 's 't 're 've 'm 'll 'd (大小写不敏感)
    ...
}
```

这段代码体现了**中英混排场景的分词策略差异**：汉字优先作为独立 token 序列（每字一 token 或按词切分），英文部分复用类 GPT-2/LLaMA-3 逻辑但**显式排除汉字**，避免中英文字符混合在同一个预分词片段中。这种设计让下游 BPE 合并更专注于各自语言的子词建模。

### 15.4.5 unicode_regex_split_custom_afmoe()：AFMOE 的数字分组魔法

源码路径：`src/unicode.cpp` - `unicode_regex_split_custom_afmoe()`（第 592-662 行）

```cpp
if (flags.is_number) {
    size_t digit_start = pos;
    size_t digit_count = 0;
    while (_get_flags(pos).is_number && pos < offset_end) { digit_count++; pos++; }

    // 关键：按总长度模 3 分组
    size_t remainder = digit_count % 3;
    size_t current = digit_start;
    if (remainder > 0) { _add_token(current + remainder); current += remainder; } // 先切 1 或 2 位
    while (current < digit_start + digit_count) { _add_token(current + 3); current += 3; } // 再切 3 位一组
    continue;
}
```

这段代码实现了**十进制分组直觉**：数字 `1234567` 会被预分词为 `1` `234` `567`（模 3 余 1 先切 1 位），而非 `123` `456` `7`。这种预分词方式让后续 BPE 更容易学到「千分位分组」模式，提升数值推理能力。

### 15.4.6 unicode_regex_split_custom_newlines()：Gemma4 的极简换行切分

源码路径：`src/unicode.cpp` - `unicode_regex_split_custom_newlines()`（第 664-691 行）

```cpp
static std::vector<size_t> unicode_regex_split_custom_newlines(const std::string & text,
                                                               const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    const auto cpts = unicode_cpts_from_utf8(text);
    for (auto offset : offsets) {
        size_t pos = offset_ini;
        while (pos < offset_end) {
            const bool is_newline = (cpts[pos] == '\n');
            const size_t run_start = pos;
            while (pos < offset_end && (cpts[pos] == '\n') == is_newline) pos++;
            bpe_offsets.push_back(pos - run_start);
        }
    }
    return bpe_offsets;
}
```

这段代码仅按 `\n` 切分：连续非换行字符作为一个片段，连续换行作为另一个片段。Gemma4 的 BPE 词表内部不允许 token 跨越换行，因此预分词阶段必须在换行处「硬切」，保证后续 BPE 合并永远不会跨行。

---

## 15.5 llama_vocab 类：词汇表核心 —— Token 世界的「百科全书」

### 15.5.1 为什么用 pimpl 模式管理 vocab？

`llama_vocab` 通过 `unique_ptr<impl>` 隐藏数千行实现细节，对外只暴露精简的查询/转换接口。`load()` 从 GGUF 元数据读取 `tokenizer.model`、`tokenizer.pre`、`tokenizer.list/score/type` 字段。根据 `tokenizer_model` 字符串识别 SPM/BPE/WPM/UGM/RWKV/PLaMo2 类型并初始化对应 tokenizer。特殊 token（BOS/EOS/EOT/SEP/PAD/FIM_*）既支持 GGUF KV 显式配置，也支持按 token 文本自动检测兜底。`special_eog_ids` 集合统一管理所有「生成终止 token」，由 token 文本匹配 + EOT/EOM 启发式构建。`cache_special_tokens` 按文本长度倒序排列，保证 `tokenizer_st_partition` 用最长匹配优先切分。`token_to_piece_for_cache` 在 load 时预生成所有 token→文本缓存，避免推理时重复计算。

### 15.5.2 llama_vocab::impl::load()：从 GGUF 加载词汇表的全流水线

源码路径：`src/llama-vocab.cpp` - `llama_vocab::impl::load()`（第 76-580 行）

```cpp
void llama_vocab::impl::load(llama_model_loader & ml, const LLM_KV & kv) {
    struct gguf_context * ctx = ml.metadata;

    // ① 识别词汇表类型与预分词类型
    ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model);        // "llama", "gpt2", "bpe", "spm", ...
    ml.get_key(LLM_KV_TOKENIZER_PRE,   tokenizer_pre, false);   // "default", "llama3", "qwen2", ...
    ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, n_token_types, false);

    // ② 根据 tokenizer_model 字符串映射到 enum llama_vocab_type
    if (tokenizer_model.find("llama") != std::string::npos) type = LLAMA_VOCAB_TYPE_BPE;
    else if (tokenizer_model.find("gpt2") != std::string::npos) type = LLAMA_VOCAB_TYPE_BPE;
    else if (tokenizer_model.find("bpe") != std::string::npos) type = LLAMA_VOCAB_TYPE_BPE;
    else if (tokenizer_model.find("spm") != std::string::npos) type = LLAMA_VOCAB_TYPE_SPM;
    else if (tokenizer_model.find("ugm") != std::string::npos) type = LLAMA_VOCAB_TYPE_UGM;
    else if (tokenizer_model.find("rwkv") != std::string::npos) type = LLAMA_VOCAB_TYPE_RWKV;
    else if (tokenizer_model.find("plamo2") != std::string::npos) type = LLAMA_VOCAB_TYPE_PLAMO2;
    else if (tokenizer_model.find("wpm") != std::string::npos) type = LLAMA_VOCAB_TYPE_WPM;
    else GGML_ABORT("unknown tokenizer model");

    // ③ 读取 token 列表、分数、类型
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int> token_types;
    ml.get_arr_str(LLM_KV_TOKENIZER_LIST, tokens);
    ml.get_arr_float(LLM_KV_TOKENIZER_SCORES, scores);
    ml.get_arr_int32(LLM_KV_TOKENIZER_TOKEN_TYPE, token_types, false);

    // ④ 填充 id_to_token、token_to_id、bpe_ranks 等核心容器
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_data td;
        td.text = tokens[i];
        td.score = (i < scores.size()) ? scores[i] : 0.0f;
        td.attr = (i < token_types.size()) ? token_types[i] : LLAMA_TOKEN_ATTR_NORMAL;
        id_to_token.push_back(td);
        token_to_id[td.text] = i;
        if (type == LLAMA_VOCAB_TYPE_BPE && i < scores.size()) { // BPE merges 存储在 scores 中
            // 解析 "token_left token_right" 格式
            ...
            bpe_ranks[std::make_pair(left, right)] = rank;
        }
    }

    // ⑤ 识别特殊 token：优先读 GGUF KV，兜底按文本启发式匹配
    ml.get_key(LLM_KV_TOKENIZER_BOS_ID, special_bos_id, false);
    if (special_bos_id == LLAMA_TOKEN_NULL) {
        for (auto & [text, id] : token_to_id) if (text == "<s>" || text == "<|begin_of_text|>") special_bos_id = id;
    }
    // ... EOS, EOT, EOM, UNK, SEP, PAD, MASK, FIM_*, LF 同理

    // ⑥ 构建 EOG 集合：显式 EOG ID + 启发式匹配 "eot" "eom" "end" 等文本
    for (const auto & [text, id] : token_to_id) {
        if (text.find("eot") != std::string::npos || text.find("eom") != std::string::npos ||
            text.find("end") != std::string::npos) special_eog_ids.insert(id);
    }

    // ⑦ 初始化具体 tokenizer
    init_tokenizer(type);

    // ⑧ 构建特殊 token 缓存（按文本长度倒序）
    for (auto & [text, id] : token_to_id) {
        if (id_to_token[id].attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED))
            cache_special_tokens.push_back(id);
    }
    std::sort(cache_special_tokens.begin(), cache_special_tokens.end(),
        [&](llama_token a, llama_token b) {
            return id_to_token[a].text.size() > id_to_token[b].text.size();
        });

    // ⑨ 预生成 token→piece 缓存
    for (size_t i = 0; i < id_to_token.size(); ++i)
        cache_token_to_piece[i] = token_to_piece_for_cache(i, true);
}
```

这段代码是**词汇表初始化的主干道**：从 GGUF 读取原始数据 → 识别类型 → 填充核心容器 → 解析特殊 token → 构建 tokenizer → 预计算缓存。每一步都做了显式配置与启发式兜底的双重保障，确保即使 GGUF 元数据不全（如旧模型），也能通过 token 文本内容推断出 BOS/EOS/EOG 等关键 token。

### 15.5.3 llama_vocab::impl::init_tokenizer()：工厂方法实例化具体分词器

源码路径：`src/llama-vocab.cpp` - `llama_vocab::impl::init_tokenizer()`（第 2722-2741 行）

```cpp
void llama_vocab::impl::init_tokenizer(enum llama_vocab_type type) {
    LLAMA_LOG_DEBUG("%s: initializing tokenizer for type %d\n", __func__, type);
    switch (type) {
        case LLAMA_VOCAB_TYPE_SPM:
            tokenizer = std::make_unique<llm_tokenizer_spm>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_BPE:
            tokenizer = std::make_unique<llm_tokenizer_bpe>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_WPM:
            tokenizer = std::make_unique<llm_tokenizer_wpm>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_UGM:
            tokenizer = std::make_unique<llm_tokenizer_ugm>(vocab, precompiled_charsmap);
            break;
        case LLAMA_VOCAB_TYPE_RWKV:
            tokenizer = std::make_unique<llm_tokenizer_rwkv>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_PLAMO2:
            tokenizer = std::make_unique<llm_tokenizer_plamo2>(vocab);
            break;
        default:
            GGML_ABORT("unsupported vocab type");
    }
}
```

这段代码是**策略模式的工厂入口**：根据 `llama_vocab_type` 构造对应的 `llm_tokenizer_*` 实例。所有 tokenizer 共享同一基类 `llm_tokenizer`（仅含虚析构），具体实现完全解耦。UGM 需要额外传入 `precompiled_charsmap`（XCDA 双数组数据），其余类型仅需 `vocab` 引用即可访问词表查询接口。

### 15.5.4 tokenizer_st_partition()：特殊 token 优先切分的贪心长匹配

源码路径：`src/llama-vocab.cpp` - `llama_vocab::impl::tokenizer_st_partition()`（第 2743-2830 行）

```cpp
void llama_vocab::impl::tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer,
                                               bool parse_special) const {
    if (!parse_special) return;

    // ① 遍历所有特殊 token（已按文本长度倒序）
    for (const llama_token special_id : cache_special_tokens) {
        const std::string & special_text = id_to_token[special_id].text;
        if (special_text.empty()) continue;

        // ② 在每个 RAW_TEXT 片段中查找并切分
        for (auto it = buffer.begin(); it != buffer.end(); ++it) {
            if (it->type != FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) continue;

            const std::string & raw = it->raw_text;
            size_t search_start = it->offset;
            while (true) {
                size_t found = raw.find(special_text, search_start);
                if (found == std::string::npos || found + special_text.size() > it->offset + it->length) break;

                // ③ 分割：前段 RAW_TEXT + 特殊 TOKEN + 后段 RAW_TEXT
                size_t before_len = found - it->offset;
                size_t after_len = it->length - before_len - special_text.size();

                auto before = (before_len > 0) ? fragment_buffer_variant(it->raw_text, it->offset, before_len)
                                               : fragment_buffer_variant(); // 空片段后续会被忽略
                auto token_frag = fragment_buffer_variant(special_id);
                auto after = (after_len > 0) ? fragment_buffer_variant(it->raw_text, found + special_text.size(), after_len)
                                             : fragment_buffer_variant();

                // ④ 在 forward_list 中原地替换
                it = buffer.insert_after(it, token_frag);
                if (after_len > 0) it = buffer.insert_after(it, after);
                if (before_len > 0) { /* 修改原节点为 before */ }
                else { buffer.erase_after(std::prev(it)); } // 删除原节点

                search_start = found + special_text.size();
            }
        }
    }
}
```

这段代码实现了**「最长优先」的特殊 token 切分**。`cache_special_tokens` 已按文本长度降序排列，因此 `<|im_start|>` 会比 `<|` 先被匹配并切出，避免短 token 抢占长 token 的前缀。使用 `std::forward_list` 支持 O(1) 原地插入/删除，且单向遍历足以覆盖所有片段，内存开销极小。

---

## 15.6 SPM 分词器实现 —— SentencePiece 的「优先级队列合并」

### 15.6.1 为什么 SPM 用优先级队列而非 BPE 的 rank？

SPM（SentencePiece）按 token 的 `score`（对数概率）排序，score 高的 pair 优先合并；与 BPE 的固定 rank 表形成对比。`llm_tokenizer_spm_session::tokenize` 三阶段流程：
- **初始化符号链表**：按码点切分文本，每个符号含 prev/next 指针维护双向链表
- **种子队列**：扫描所有相邻字符对，调用 `try_add_bigram` 检查是否在 vocab 中
- **贪心合并**：循环 pop 队列顶 bigram，若两符号未被合并且长度匹配则合并右侧到左侧
`resegment` 兜底逻辑：当合并后的子串仍不在 vocab 中时，先查 `rev_merge` 缓存尝试再分，否则按字节回退。`try_add_bigram` 同时维护 `work_queue` 和 `rev_merge`（文本→(left,right) 映射），是 resegment 的逆向索引。`llm_bigram_spm::comparator` 排序规则：`(l.score < r.score) || (l.score == r.score && l.left > r.left)`。

### 15.6.2 llm_tokenizer_spm_session::tokenize()：SPM 三阶段合并主流程

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_spm_session::tokenize()`（第 115-158 行）

```cpp
void tokenize(const std::string & text, std::vector<llama_token> & output) {
    // === 阶段 1：初始化符号链表 ===
    int index = 0;
    size_t offs = 0;
    while (offs < text.size()) {
        llm_symbol sym;
        size_t len = unicode_len_utf8(text[offs]);
        sym.text = text.c_str() + offs;
        sym.n = std::min(len, text.size() - offs);
        offs += sym.n;
        sym.prev = index - 1;
        sym.next = offs == text.size() ? -1 : index + 1;
        index++;
        symbols.emplace_back(sym);
    }

    // === 阶段 2：种子队列（所有相邻 pair 入队）===
    for (int i = 1; i < (int) symbols.size(); ++i) {
        try_add_bigram(i - 1, i);
    }

    // === 阶段 3：贪心合并循环 ===
    while (!work_queue.empty()) {
        auto bigram = work_queue.top(); work_queue.pop();

        auto & left_sym = symbols[bigram.left];
        auto & right_sym = symbols[bigram.right];

        // 已合并或长度不符 → 跳过
        if (left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size) continue;

        // 合并：右并入左
        left_sym.n += right_sym.n;
        right_sym.n = 0;

        // 维护双向链表
        left_sym.next = right_sym.next;
        if (right_sym.next >= 0) symbols[right_sym.next].prev = bigram.left;

        // 新产生的相邻对入队
        try_add_bigram(left_sym.prev, bigram.left);
        try_add_bigram(bigram.left, left_sym.next);
    }

    // === 输出阶段：遍历链表，resegment 兜底 ===
    for (int i = 0; i != -1; i = symbols[i].next) {
        auto & symbol = symbols[i];
        resegment(symbol, output);
    }
}
```

这段代码展示了 SPM 的**贪心最长匹配合并**核心：符号链表 + 优先级队列（按 score 降序）。每次合并都把右符号「吞并」到左符号（累加 `n`，清零右符号），并用双向链表维护邻接关系。合并后立即尝试将新的左右相邻对加入队列，实现动态扩展。最终遍历链表，每个幸存符号调用 `resegment` 确保能映射到 vocab token。

### 15.6.3 llm_tokenizer_spm_session::resegment()：兜底的递归逆向查找

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_spm_session::resegment()`（第 160-186 行）

```cpp
void resegment(llm_symbol & symbol, std::vector<llama_token> & output) {
    auto text = std::string(symbol.text, symbol.n);
    auto token = vocab.text_to_token(text);

    // ① 直接命中 vocab → 输出
    if (token != LLAMA_TOKEN_NULL) {
        output.push_back(token);
        return;
    }

    // ② 查 rev_merge 逆向索引
    const auto p = rev_merge.find(text);
    if (p == rev_merge.end()) {
        // ③ 彻底不在词表 → 字节回退
        output.reserve(output.size() + symbol.n);
        for (int j = 0; j < (int)symbol.n; ++j) {
            llama_token id = vocab.byte_to_token(symbol.text[j]);
            output.push_back(id);
        }
        return;
    }

    // ④ 递归拆分左右子符号
    resegment(symbols[p->second.first], output);
    resegment(symbols[p->second.second], output);
}
```

这段代码是 SPM 的**安全网**：合并后的符号若不在 vocab 中，先查 `rev_merge`（记录合并时的左右子符号索引），递归拆分；若连 `rev_merge` 也没有（极少见），退化为逐字节编码。`rev_merge` 的存在使得 SPM 能在贪心合并「过度」时回溯，保证覆盖率 100%。

### 15.6.4 llm_tokenizer_spm_session::try_add_bigram()：队列入队与逆向索引同步构建

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_spm_session::try_add_bigram()`（第 188-216 行）

```cpp
void try_add_bigram(int left, int right) {
    if (left == -1 || right == -1) return;
    const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
    auto token = vocab.text_to_token(text);
    if (token == LLAMA_TOKEN_NULL) return;
    if (static_cast<uint32_t>(token) >= vocab.n_tokens()) return;

    const auto & tok_data = vocab.get_token_data(token);

    llm_bigram_spm bigram;
    bigram.left  = left;
    bigram.right = right;
    bigram.score = tok_data.score;  // ① 使用 token 的 score 作为优先级
    bigram.size  = text.size();

    work_queue.push(bigram);        // ② 入优先级队列

    // ③ 同时记录逆向索引：合并文本 → (左索引, 右索引)
    rev_merge[text] = std::make_pair(left, right);
}
```

这段代码完成了**双重记账**：`work_queue` 按 `score` 排序驱动合并顺序；`rev_merge` 记录「合并文本 → 子符号位置」映射，供 `resegment` 逆向拆解使用。注意 `score` 越小（对数概率越大）优先级越高，`comparator` 用 `<` 实现最小堆顶即最高分。

---

## 15.7 BPE 分词器实现 —— GPT/LLaMA 家族的「rank 表驱动合并」

### 15.7.1 为什么 BPE 用 rank 而非 score？

BPE（Byte Pair Encoding）按「合并规则的出现顺序」分配 rank，rank 小的 pair 优先合并。训练阶段把所有相邻 token pair 按频次排序写入 merges.txt；推理时按 rank 合并即可保证一致性。`llm_tokenizer_bpe` 构造函数：为每个 `LLAMA_VOCAB_PRE_TYPE_*` 绑定专属 `regex_exprs`（如 LLaMA3 用 `(?i:'s|...)`）。`llm_tokenizer_bpe_session::tokenize` 关键点：
- 先用 `unicode_regex_split` 按 regex 切分文本为 word 集合（可选 `byte_encode` GPT-2 字节编码）
- 每个 word 内部建立字符级符号链表，循环 `add_new_bigram` 加入优先级队列
- 合并时校验 `left_token+right_token == bigram.text` 防止过期 bigram 误用
- 合并后通过 `symbols_final` 跨 word 维护 next/prev 链表，最终输出保持原始顺序

### 15.7.2 llm_tokenizer_bpe 构造函数：40+ 种预分词正则的 switch 分发

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_bpe::llm_tokenizer_bpe()`（第 255-427 行）

```cpp
llm_tokenizer_bpe(const llama_vocab & vocab) {
    GGML_ASSERT(vocab.get_type() == LLAMA_VOCAB_TYPE_BPE);
    switch (vocab.get_pre_type()) {
        case LLAMA_VOCAB_PRE_TYPE_LLAMA3:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
                "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
                "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_GPT2:
        case LLAMA_VOCAB_PRE_TYPE_MPT:
        case LLAMA_VOCAB_PRE_TYPE_OLMO:
            regex_exprs = {
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_KIMI_K2:
            regex_exprs = { "\\p{Han}+" };
            break;
        case LLAMA_VOCAB_PRE_TYPE_AFMOE:
            regex_exprs = { "\\p{AFMoE_digits}", ... };
            break;
        case LLAMA_VOCAB_PRE_TYPE_GEMMA4:
            regex_exprs = { "[^\\n]+|[\\n]+" };
            byte_encode = false;
            break;
        // ... 40+ 个 case
        default:
            regex_exprs = { ... };
            break;
    }
}
```

这段代码是 **BPE 预分词正则的中央注册表**。每个 `LLAMA_VOCAB_PRE_TYPE_*` 枚举值对应一组正则字符串，`unicode_regex_split` 运行时会按顺序应用。关键差异：
- LLaMA-3：显式大小写字母列表 `(?:'[sS]|...)` 替代 `(?i:...)`，数字限长 `\p{N}{1,3}`，显式 `\r\n` 处理。
- Kimi-K2：仅 `\p{Han}+`，汉字优先由 custom 实现处理。
- AFMOE：自定义 `\p{AFMoE_digits}` 触发 `custom_afmoe` 实现。
- Gemma4：仅按 `\n` 切分，且 `byte_encode = false`（不做 GPT-2 字节编码）。
- 默认回退：FALCON 风格 4 条正则。

### 15.7.3 llm_tokenizer_bpe_session::tokenize()：BPE 逐词合并与跨词链表

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_bpe_session::tokenize()`（第 467-539 行）

```cpp
void tokenize(const std::string & text, std::vector<llama_token> & output) {
    int final_prev_index = -1;
    const auto word_collection = unicode_regex_split(text, tokenizer.regex_exprs, tokenizer.byte_encode);

    symbols_final.clear();
    auto tok_pre = vocab.get_pre_type();

    for (const auto & word : word_collection) {
        work_queue = llm_bigram_bpe::queue();
        symbols.clear();

        // ① 整词命中且 ignore_merges → 直接作为单符号
        if (vocab.get_ignore_merges() && vocab.text_to_token(word) != LLAMA_TOKEN_NULL) {
            symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
        }
        // ② Gemma4 特殊修复：纯换行词直接查 token
        else if (tok_pre == LLAMA_VOCAB_PRE_TYPE_GEMMA4 && word.find_first_not_of('\n') == std::string::npos) {
            auto tok = vocab.text_to_token(word);
            if (tok != LLAMA_TOKEN_NULL) symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
        }
        // ③ 普通情况：按码点拆成字符符号链表
        else {
            int index = 0; size_t offset = 0;
            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t)unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            // 种子队列
            for (int i = 1; i < (int)symbols.size(); ++i) add_new_bigram(i - 1, i);
        }

        // === 合并循环 ===
        while (!work_queue.empty()) {
            auto bigram = work_queue.pop_move();

            auto & left_symbol = symbols[bigram.left];
            auto & right_symbol = symbols[bigram.right];
            if (left_symbol.n == 0 || right_symbol.n == 0) continue;

            std::string left_token = std::string(left_symbol.text, left_symbol.n);
            std::string right_token = std::string(right_symbol.text, right_symbol.n);
            if (left_token + right_token != bigram.text) continue; // 过期 bigram 校验

            left_symbol.n += right_symbol.n;
            right_symbol.n = 0;
            left_symbol.next = right_symbol.next;
            if (right_symbol.next >= 0) symbols[right_symbol.next].prev = bigram.left;

            add_new_bigram(left_symbol.prev, bigram.left);
            add_new_bigram(bigram.left, left_symbol.next);
        }

        // === 跨词链表拼接 ===
        for (auto & sym : symbols) {
            if (sym.n > 0) {
                sym.prev = final_prev_index;
                sym.next = -1;
                if (final_prev_index != -1) symbols_final[final_prev_index].next = symbols_final.size();
                symbols_final.emplace_back(sym);
                final_prev_index = symbols_final.size() - 1;
            }
        }
    }

    symbols = symbols_final;

    // === 最终输出 token ===
    if (!symbols.empty()) {
        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            if (symbol.n == 0) continue;
            const std::string str = std::string(symbol.text, symbol.n);
            const auto token = vocab.text_to_token(str);
            if (token == LLAMA_TOKEN_NULL) {
                for (auto j = str.begin(); j != str.end(); ++j) {
                    std::string byte_str(1, *j);
                    auto token_multibyte = vocab.text_to_token(byte_str);
                    if (token_multibyte != LLAMA_TOKEN_NULL) output.push_back(token_multibyte);
                }
            } else {
                output.push_back(token);
            }
        }
    }
}
```

这段代码是 BPE 分词的**完整流水线**：
1. **预分词**：`unicode_regex_split` 按模型专属正则切分为 `word_collection`。
2. **逐词处理**：每个 word 独立建立字符符号链表，独立合并（BPE 合并不跨 word 边界）。
3. **合并校验**：`left_token + right_token == bigram.text` 过滤过期 bigram（因符号被合并导致文本变化）。
4. **跨词链表**：`symbols_final` 将各 word 的幸存符号按原序串联，保证输出 token 顺序与原文一致。
5. **字节回退**：词表未命中时逐字节查 `byte_to_token`（GPT-2 字节映射）。

### 15.7.4 llm_tokenizer_bpe_session::add_new_bigram()：BPE rank 表查询与入队

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_bpe_session::add_new_bigram()`（第 541-566 行）

```cpp
void add_new_bigram(int left, int right) {
    if (left == -1 || right == -1) return;
    std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
    std::string right_token = std::string(symbols[right].text, symbols[right].n);

    int rank_found = -1;
    rank_found = vocab.find_bpe_rank(left_token, right_token); // ① 查 rank 表
    if (rank_found < 0) return;

    llm_bigram_bpe bigram;
    bigram.left  = left;
    bigram.right = right;
    bigram.text  = left_token + right_token;
    bigram.size  = left_token.size() + right_token.size();
    bigram.rank  = rank_found;

    work_queue.push(bigram); // ② rank 越小优先级越高（最小堆）
}
```

这段代码体现了 BPE 与 SPM 的核心差异：**rank 表查询替代 score 排序**。`vocab.find_bpe_rank` 在 `bpe_ranks` unordered_map 中 O(1) 查找 `(left, right)` 对应的合并优先级。rank 值越小（训练时出现越早）优先级越高，`llm_bigram_bpe::comparator` 用 `rank >` 实现最小堆。

---

## 15.8 WPM / UGM / RWKV / PLaMo2 分词器 —— 百花齐放的「专用工具箱」

### 15.8.1 为什么需要多种专用分词器？

WPM（WordPiece）用于 BERT 类编码器模型，靠最长前缀匹配 + `token_unk` 兜底。UGM（Unigram）用于 T5 等基于 SentencePiece 的 LM，用 Viterbi 算法找全局最优分词路径。RWKV 通过 `naive_trie` 字典树实现 O(n) 的最长匹配查找。PLaMo2 用 Trie 表 + 动态规划（`scores`/`path` 数组）实现基于后缀表的贪心匹配。

### 15.8.2 llm_tokenizer_wpm_session::preprocess()：WPM 的 NFD+小写+标点分词预处理

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_wpm_session::preprocess()`（第 626-654 行）

```cpp
static std::vector<std::string> preprocess(const std::string & text) {
    // ① NFD 规范化 + UTF-8 解码
    const std::vector<uint32_t> cpts_nfd = unicode_cpts_normalize_nfd(unicode_cpts_from_utf8(text));
    std::vector<std::string> words(1, "");

    for (const uint32_t cpt : cpts_nfd) {
        const auto flags = unicode_cpt_flags_from_cpt(cpt);

        // 空白字符：结束当前 word
        if (flags.is_whitespace) {
            if (words.back().size()) words.emplace_back();
            continue;
        }
        // 控制字符/替换字符：跳过
        if (cpt == 0 || cpt == 0xFFFD || flags.is_control) continue;

        // ② 小写转换
        const std::string s = unicode_cpt_to_utf8(unicode_tolower(cpt));

        // ③ 标点、符号、汉字：单字成词
        if (flags.is_punctuation || (cpt < 0x7F && flags.is_symbol) || is_chinese_char(cpt)) {
            if (words.back().size()) words.emplace_back();
            words.back() = s;
            words.emplace_back();
        } else {
            // ④ 普通字符：追加到当前 word
            words.back() += s;
        }
    }
    if (!words.back().size()) words.pop_back();
    return words;
}
```

这段代码实现了 **BERT/WordPiece 经典预处理管线**：NFD 分解 → 小写化 → 标点/汉字强制切分。`is_chinese_char` 覆盖 8 个 CJK 区间，确保中文字符单独成词。输出的 `words` 向量随后在 `tokenize()` 中逐个做最长前缀匹配。

### 15.8.3 llm_tokenizer_wpm_session::tokenize()：最长前缀匹配 + UNK 兜底

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_wpm_session::tokenize()`（第 584-624 行）

```cpp
void tokenize(const std::string & text, std::vector<llama_token> & output) {
    std::vector<std::string> words = preprocess(text);

    for (const std::string & word : words) {
        if (word.empty()) continue;
        const std::string word1 = "\xe2\x96\x81" + word; // ① 前缀 ▁ (U+2581)
        const int n = word1.size();
        const size_t current_tokens = output.size();

        for (int i = 0; i < n; ++i) {
            bool match = false;
            // ② 从最长可能长度向下尝试
            for (int j = std::min(n, i + vocab.max_token_len() + 1); j > i; j--) {
                auto id = vocab.text_to_token(word1.substr(i, j - i));
                if (id != LLAMA_TOKEN_NULL) {
                    output.push_back(id);
                    match = true;
                    i = j - 1;
                    break;
                }
            }
            if (!match) { // ③ 该 word 完全匹配失败 → 整词回退 UNK
                output.resize(current_tokens);
                break;
            }
        }
        if (current_tokens == output.size()) output.push_back(vocab.token_unk());
    }
}
```

这段代码展示了 **WordPiece 的贪心最长前缀匹配**：每个 word 前加 `▁`（空格标记），从位置 `i` 向后尝试最长子串，命中即输出 token 并跳过。若某位置无任何匹配，**整个 word 回退 UNK**，不做字节级拆解——这是与 BPE/SPM 最大的区别。

### 15.8.4 llm_tokenizer_ugm_session::tokenize()：Unigram Viterbi DP 算法

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_ugm_session::tokenize()`（第 864-912 行）

```cpp
void tokenize(const std::string & text, std::vector<llama_token> & output) {
    size_t output_size = output.size();
    std::string normalized;
    normalize(text, &normalized); // ① 归一化（空格处理、XCDA 前缀替换）
    size_t input_len = normalized.size();
    if (input_len == 0) return;

    // ② DP 表：每个位置存储 {token_id, prev_offset, score_sum}
    std::vector<struct best_tokenization> tokenization_results(input_len + 1,
        {vocab.token_unk(), 0, -DBL_MAX});
    tokenization_results[0] = { vocab.token_unk(), 0, 0 };

    for (size_t input_offset = 0; input_offset < input_len; ) {
        size_t prefix_offset = input_offset;
        size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[input_offset]),
                                                    input_len - input_offset);
        bool single_codepoint_token_found = false;
        const struct best_tokenization & current_best = tokenization_results[input_offset];

        // ③ Trie 遍历：从当前位置向后扩展，沿 naive_trie 查找所有前缀 token
        const struct naive_trie * node = tokenizer.token_matcher.traverse(normalized[prefix_offset++]);
        while (prefix_offset <= input_len && node != NULL) {
            if (node->has_value) {
                if (prefix_offset - input_offset == n_utf8_code_units) single_codepoint_token_found = true;
                llama_token token_id = node->value;
                const auto & token_data = vocab.get_token_data(token_id);
                const double token_score = vocab.is_user_defined(token_id) ? 0.0 : token_data.score;
                const double challenger_score = current_best.score_sum + token_score;
                struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                if (challenger_score > current_champ.score_sum) {
                    current_champ = { token_id, input_offset, challenger_score };
                }
            }
            node = node->traverse(normalized[prefix_offset++]);
        }

        // ④ 单码点未命中 → 强制 UNK
        if (!single_codepoint_token_found) {
            const double challenger_score = current_best.score_sum + tokenizer.unknown_token_score;
            prefix_offset = input_offset + n_utf8_code_units;
            struct best_tokenization & current_champ = tokenization_results[prefix_offset];
            if (challenger_score > current_champ.score_sum) {
                current_champ = { vocab.token_unk(), input_offset, challenger_score };
            }
        }
        input_offset += n_utf8_code_units;
    }

    // ⑤ 回溯重建最优路径
    bool is_prev_unknown = false;
    for (struct best_tokenization & tokenization = tokenization_results[input_len]; ; tokenization = tokenization_results[tokenization.input_offset]) {
        bool is_unknown = tokenization.token_id == vocab.token_unk();
        if (!(is_prev_unknown && is_unknown)) output.push_back(tokenization.token_id);
        if (tokenization.input_offset == 0) break;
        is_prev_unknown = is_unknown;
    }
    std::reverse(output.begin() + output_size, output.end());
}
```

这段代码实现了 **Unigram 语言模型的 Viterbi 解码**：
- `tokenization_results[i]` 记录以位置 `i` 结尾的最优分词路径（token_id、前驱位置、累计分数）。
- 遍历输入位置，用 `naive_trie` 找出所有以该位置开始的 token，更新后续位置的最优解。
- 单码点未命中时强制兜底 UNK（惩罚分数 `unknown_token_score`）。
- 最后从末尾回溯重建全局最优 token 序列。

### 15.8.5 llm_tokenizer_ugm_session::normalize_prefix()：XCDA 紧凑双数组 O(1) 前缀查找

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_ugm_session::normalize_prefix()`（第 957-1001 行）

```cpp
struct normalization_result normalize_prefix(const std::string & input, size_t input_offset) {
    // ① 优先匹配 user_defined token（最长前缀）
    auto user_defined_token_match = tokenizer.user_defined_token_matcher.get_longest_prefix(...);
    if (user_defined_token_match.second > 0) return { ... };

    // ② XCDA 双数组遍历
    size_t longest_prefix_length = 0, longest_prefix_offset = 0;
    if (tokenizer.xcda_array_size > 0) {
        struct xcda_array_view xcda_view(tokenizer.xcda_array, tokenizer.xcda_array_size);
        uint32_t node_index = 0;
        node_index = xcda_view.get_base(node_index); // 根节点 BASE

        for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
            unsigned char c = input[prefix_offset];
            if (c == 0) break;
            node_index ^= c;                          // XOR 压缩
            if (xcda_view.get_lcheck(node_index) != c) break; // LCHECK 校验
            bool is_leaf = xcda_view.get_leaf(node_index);
            node_index ^= xcda_view.get_base(node_index);     // 移动到下一层 BASE
            if (is_leaf) {
                longest_prefix_length = prefix_offset - input_offset + 1;
                longest_prefix_offset = xcda_view.get_value(node_index);
            }
        }
    }

    // ③ 命中叶子 → 返回替换字符串
    if (longest_prefix_length > 0) {
        const char * prefix_replacement = &(tokenizer.prefix_replacements)[longest_prefix_offset];
        return { prefix_replacement, strlen(prefix_replacement), longest_prefix_length };
    }

    // ④ 兜底：单码点合法性检查
    try {
        size_t prefix_offset = input_offset;
        unicode_cpt_from_utf8(input, prefix_offset);
        return { &input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset };
    } catch (...) { return { "\xEF\xBF\xBD", 3, 1 }; } // 替换字符
}
```

这段代码是 **XCDA（XOR-Compressed Double Array）解码器**。双数组将 Trie 压缩为两个并行数组 `BASE`/`CHECK`，并用 `LEAF` 标记叶子节点存储替换值索引。`xcda_array_view` 通过位运算从打包的 `uint32_t` 中提取三字段：
- `BASE`：高 22 位（`>>10` 再按第 9 位决定是否左移 4）
- `LCHECK`：低 8 位 + 符号位（`& 0x800000FF`）
- `LEAF`：第 8 位（`(>>8) & 1`）
- `VALUE`：低 31 位（`& 0x7FFFFFFF`）

遍历时 `node_index ^= c` 实现 XOR 压缩跳转，`get_lcheck` 校验冲突，`get_leaf` 判断是否为终态。这使得 UGM 的归一化前缀查找达到**接近 O(1) 的极高速度**。

### 15.8.6 llama_unescape_rwkv_token()：RWKV 转义序列解析

源码路径：`src/llama-vocab.cpp` - `llama_unescape_rwkv_token()`（第 1035-1078 行）

```cpp
static std::vector<uint8_t> llama_unescape_rwkv_token(const std::string & escaped) {
    std::vector<uint8_t> output;
    output.reserve(escaped.size());
    bool escaping = false;
    uint8_t hex_remaining = 0;
    uint8_t hex_acc = 0;

    for (const char & c : escaped) {
        if (hex_remaining != 0) {
            uint8_t value = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
            hex_acc = (hex_acc << 4) + value;
            hex_remaining -= 1;
            if (hex_remaining == 0) { output.push_back(hex_acc); hex_acc = 0; }
            continue;
        }
        if (escaping) {
            if (c == 't') output.push_back('\t');
            else if (c == 'n') output.push_back('\n');
            else if (c == 'r') output.push_back('\r');
            else if (c == 'x') { hex_remaining = 2; }
            else output.push_back(c);
            escaping = false;
            continue;
        }
        if (c == '\\') { escaping = true; continue; }
        output.push_back(c);
    }
    return output;
}
```

这段代码处理 RWKV 词表中的转义表示：`\t` `\n` `\r` → 单字节；`\xHH` → 任意字节值。RWKV 词表将所有 token 存为转义字符串，加载时统一解码为字节序列，构建 `naive_trie` 字典树用于推理时最长匹配。

### 15.8.7 llm_tokenizer_plamo2::build()：Trie 展平为四张表 + DP 编码准备

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_plamo2::build()`（第 1095-1160 行）

```cpp
void build(const llama_vocab & vocab) {
    // ① 收集所有 token 文本、字节 token、后缀分数
    std::unordered_map<std::string, float> suffix_to_score;
    std::unordered_map<std::string, llama_token> token_to_id;
    for (size_t token_id = 0; token_id < vocab.n_tokens(); ++token_id) {
        const auto & entry = vocab.get_token_data(token_id);
        tokens_.push_back(entry.text);
        token_to_id[entry.text] = token_id;
        if (vocab.is_byte(token_id)) { /* 解析 <0xHH> 填 bytes_[256] */ continue; }
        suffix_to_score[entry.text] = entry.score;
        // 生成所有后缀（含空后缀），缺失者置 NaN
        std::vector<uint32_t> cpts = unicode_cpts_from_utf8(entry.text);
        for (size_t i = 1; i < cpts.size(); ++i) {
            std::string suffix;
            for (size_t j = i; j < cpts.size(); ++j) suffix += unicode_cpt_to_utf8(cpts[j]);
            if (!suffix_to_score.count(suffix)) suffix_to_score[suffix] = NaN;
        }
    }

    // ② 后缀按逆序字典序排序，分配 suffix_id
    std::vector<std::string> suffixes;
    for (auto & p : suffix_to_score) suffixes.push_back(p.first);
    suffixes.push_back("");
    std::sort(suffixes.begin(), suffixes.end(), [](a,b){ return rev(a) < rev(b); });

    std::unordered_map<std::string, int32_t> suffix_to_id;
    int32_t num_pieces = 0;
    for (const auto & suffix : suffixes) {
        suffix_to_id[suffix] = num_pieces;
        if (!suffix.empty()) {
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(suffix);
            int64_t piece_code = (int64_t(cpts[0]) << 32) | suffix_to_id[remaining]; // ③ 编码：首码点<<32 | 剩余后缀id
            to_suffix_id_[piece_code] = num_pieces;
            // 统计该后缀下所有可能 piece_length
            for (int32_t piece_length = cpts.size(); piece_length > 0; --piece_length) {
                std::string piece = suffix.substr(0, piece_length);
                if (suffix_to_score.count(piece)) num_pieces++;
            }
            num_pieces++; // 空 piece
        } else num_pieces++;
    }

    // ④ 构建扁平 table_[num_pieces][4]：{piece_length, token_id, score*1e4, piece_id}
    table_.resize(num_pieces, std::vector<int32_t>(4, 0));
    int32_t table_idx = 0;
    for (const auto & suffix : suffixes) {
        std::vector<uint32_t> cpts = unicode_cpts_from_utf8(suffix);
        for (int32_t piece_length = cpts.size(); piece_length > 0; --piece_length) {
            std::string piece = suffix.substr(0, piece_length);
            auto score_it = suffix_to_score.find(piece);
            if (score_it == suffix_to_score.end()) continue;
            table_[table_idx][0] = piece_length;
            table_[table_idx][1] = token_to_id.count(piece) ? token_to_id[piece] : -1;
            table_[table_idx][2] = std::isfinite(score_it->second) ? round(score_it->second * 1e4) : INVALID_SCORE;
            table_[table_idx][3] = suffix_to_id[piece];
            table_idx++;
        }
        table_[table_idx] = {1, -1, UNKNOWN_SCORE, suffix_to_id[""]}; table_idx++;
    }
}
```

这段代码将 Trie **完全展平为四张数组**：
- `tokens_`：id→文本
- `bytes_[256]`：字节值→token_id
- `to_suffix_id_`：`(首码点<<32 | 后缀id)` → 当前片段在 table 中的起始行
- `table_[N][4]`：每行记录 `{piece_length, token_id, 定点数score, piece_id}`

后缀按**逆序字典序**排序，保证相同前缀的片段在 table 中连续存放。`to_suffix_id_` 利用首码点高 32 位 + 后缀 id 低 32 位编码，实现 O(1) 定位「以某码点开头、属于某后缀类」的片段组。

### 15.8.8 llm_tokenizer_plamo2::encode()：动态规划回溯找最优路径

源码路径：`src/llama-vocab.cpp` - `llm_tokenizer_plamo2::encode()`（第 1162-1224 行）

```cpp
std::vector<llama_token> encode(const std::string & text) const {
    std::vector<uint32_t> unicode_data = unicode_cpts_from_utf8(text);
    if (!unicode_data.empty() && unicode_data[0] == 0xFEFF) unicode_data.erase(unicode_data.begin());
    if (unicode_data.empty()) return {};

    const size_t data_len = unicode_data.size();
    // ① DP 数组：scores[i] = 从 i 到末尾的最小代价；path[i] = {piece_length, token_id, num_tokens}
    std::vector<int64_t> scores(data_len + 1, 1LL << 60);
    scores[data_len] = 0;
    std::vector<std::array<int32_t, 3>> path(data_len + 1, {0,0,0});

    int32_t suffix_id = 0;
    for (int i = data_len - 1; i >= 0; --i) {
        uint32_t c = unicode_data[i];
        // ② 找到以 c 开头、属于当前 suffix_id 的片段组起始行
        for (size_t p = suffix_id; p < table_.size(); ++p) {
            int64_t piece_code = (int64_t(c) << 32) | table_[p][3];
            auto it = to_suffix_id_.find(piece_code);
            suffix_id = (it != to_suffix_id_.end()) ? it->second : 0;
            if (suffix_id > 0 || table_[p][2] == UNKNOWN_SCORE) break;
        }
        // ③ 遍历该片段组所有 piece_length
        for (size_t p = suffix_id; p < table_.size(); ++p) {
            int32_t score = table_[p][2];
            if (score > INVALID_SCORE) {
                int32_t piece_length = table_[p][0];
                int64_t s = scores[i + piece_length] - score; // 负分数 = 代价
                if (s < scores[i]) {
                    scores[i] = s;
                    path[i] = { piece_length, table_[p][1], path[i + piece_length][2] + 1 };
                    if (score == UNKNOWN_SCORE) // 字节回退额外 token 数
                        path[i][2] += (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);
                }
            }
            if (score == UNKNOWN_SCORE) break; // 仅处理第一个 UNKNOWN_SCORE 行
        }
    }

    // ④ 回溯重建 token 序列
    std::vector<llama_token> token_ids;
    token_ids.reserve(path[0][2]);
    int pos = 0;
    while (pos < (int)data_len) {
        if (path[pos][1] >= 0) token_ids.push_back(path[pos][1]);
        else { // 字节回退：按 UTF-8 编码拆分字节，查 bytes_ 表
            uint32_t c = unicode_data[pos];
            int s = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);
            for (int i = 0; i < s; ++i) {
                uint8_t b = ...; // 提取 UTF-8 各字节
                token_ids.push_back(bytes_[b]);
            }
        }
        pos += path[pos][0];
    }
    return token_ids;
}
```

这段代码实现了 **PLaMo2 的后缀感知动态规划分词**：
- 从后向前 DP，`scores[i]` 存储从位置 `i` 到末尾的最小代价（负对数概率和）。
- `to_suffix_id_` 快速定位「以 `unicode_data[i]` 开头、接在当前后缀类之后」的候选片段组。
- 候选片段按 `piece_length` 递减顺序存放，优先尝试长片段。
- `UNKNOWN_SCORE` 行代表字节回退兜底，额外计入 UTF-8 字节数作为 token 数惩罚。
- 回溯阶段按 `path` 重建 token 序列，字节回退时手动 UTF-8 编码并查 `bytes_` 表。

---

## 15.9 分词与反分词 —— 文本与 Token 的「双向翻译官」

### 15.9.1 为什么分词/反分词要严格对称？

推理服务需要把用户输入文本转成 token ID 给模型，再把模型输出的 token ID 还原成文本回传。`llama_vocab::impl::tokenize` 主入口：
- 先用 `tokenizer_st_partition` 切出特殊 token（按 cache 长度倒序匹配）
- 按 vocab 类型分发到 SPM/BPE/WPM/UGM/RWKV/PLaMo2 的 session::tokenize
- BPE 额外处理 `escape_whitespaces`（`▁` ↔ 空格转换）和 `byte_encode`（GPT-2 字节映射）

`llama_vocab::impl::detokenize` 三遍清理：
- 第一遍去掉 `?`,`!`,`.`,`,` 前的多余空格
- 第二遍处理被空格包围的撇号 `' '`
- 第三遍处理 `'s`,`'m`,`'re`,`'ve` 等英语缩写前的空格

`token_to_piece_for_cache` 在 load 时预生成所有 token→文本缓存，避免推理时重复计算。`token_to_piece` 按 vocab 类型差异化还原：SPM `▁`↔空格、BPE 字节token解码、RWKV 转义解码。C API 层（`llama_tokenize`/`llama_detokenize`/`llama_token_to_piece`）将 C++ 接口暴露给 C 用户。

### 15.9.2 llama_vocab::impl::tokenize()：顶层分词分发入口

源码路径：`src/llama-vocab.cpp` - `llama_vocab::impl::tokenize()`（第 2847-2970 行）

```cpp
std::vector<llama_token> llama_vocab::impl::tokenize(const std::string & raw_text,
                                                     bool add_special, bool parse_special) const {
    GGML_ASSERT(tokenizer && "Tokenizer not initialized.");

    // ① 构建初始片段缓冲：单个 RAW_TEXT 节点
    std::forward_list<fragment_buffer_variant> buffer;
    buffer.emplace_front(raw_text, 0, raw_text.size());

    // ② 特殊 token 切分
    tokenizer_st_partition(buffer, parse_special);

    std::vector<llama_token> output;

    // ③ 按片段类型分发
    for (auto & frag : buffer) {
        if (frag.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN) {
            output.push_back(frag.token); // 特殊 token 直通
        } else {
            const std::string text = frag.raw_text.substr(frag.offset, frag.length);
            std::vector<llama_token> piece_tokens;

            switch (get_type()) {
                case LLAMA_VOCAB_TYPE_SPM: {
                    llm_tokenizer_spm_session session(vocab);
                    session.tokenize(text, piece_tokens);
                    break;
                }
                case LLAMA_VOCAB_TYPE_BPE: {
                    // BPE 额外处理空格转义与字节编码
                    std::string processed = text;
                    if (escape_whitespaces) llama_escape_whitespace(processed);
                    if (byte_encode) {
                        auto words = unicode_regex_split(processed, tokenizer->regex_exprs, false);
                        processed.clear();
                        for (auto & w : words) processed += unicode_byte_to_utf8(w[0]); // 简化示意
                    }
                    llm_tokenizer_bpe_session session(vocab, *tokenizer);
                    session.tokenize(processed, piece_tokens);
                    break;
                }
                case LLAMA_VOCAB_TYPE_WPM: {
                    llm_tokenizer_wpm_session session(vocab);
                    session.tokenize(text, piece_tokens);
                    break;
                }
                case LLAMA_VOCAB_TYPE_UGM: {
                    llm_tokenizer_ugm_session session(vocab, *tokenizer);
                    session.tokenize(text, piece_tokens);
                    break;
                }
                case LLAMA_VOCAB_TYPE_RWKV: {
                    llm_tokenizer_rwkv_session session(vocab, *tokenizer);
                    session.tokenize(text, piece_tokens);
                    break;
                }
                case LLAMA_VOCAB_TYPE_PLAMO2: {
                    llm_tokenizer_plamo2_session session(vocab, *tokenizer);
                    session.tokenize(text, piece_tokens);
                    break;
                }
                default: GGML_ABORT("fatal error");
            }
            output.insert(output.end(), piece_tokens.begin(), piece_tokens.end());
        }
    }

    // ④ 自动添加 BOS/EOS
    if (add_special) {
        if (add_bos && special_bos_id != LLAMA_TOKEN_NULL) output.insert(output.begin(), special_bos_id);
        if (add_eos && special_eos_id != LLAMA_TOKEN_NULL) output.push_back(special_eos_id);
    }
    return output;
}
```

这段代码是**分词总线**：
1. 将原始文本包装为 `forward_list<fragment_buffer_variant>`（初始仅 1 个 RAW_TEXT 节点）。
2. `tokenizer_st_partition` 按长度倒序切出所有特殊 token，生成 TOKEN/RAW_TEXT 交替链表。
3. 遍历链表：TOKEN 直接输出；RAW_TEXT 分发给对应 tokenizer session。
4. BPE 特殊处理：`escape_whitespaces`（空格↔▁）和 `byte_encode`（GPT-2 字节映射）。
5. 最后按 `add_bos/add_eos` 配置注入特殊 token。

### 15.9.3 llama_vocab::impl::token_to_piece()：差异化 token→文本还原

源码路径：`src/llama-vocab.cpp` - `llama_vocab::impl::token_to_piece()`（第 3320-3424 行）

```cpp
int32_t token_to_piece(llama_token token, char * buf, int32_t length,
                       bool escape_whitespaces, bool special) const {
    const auto & token_text = id_to_token[token].text;
    const auto & attr = id_to_token[token].attr;
    const auto attr_special = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED;

    switch (type) {
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: {
            if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED))
                return _try_copy(token_text.data(), token_text.size()); // ① 特殊/用户 token 原样
            if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                std::string result = token_text;
                llama_unescape_whitespace(result); // ② ▁ → 空格
                return _try_copy(result.data(), result.size());
            }
            if (attr & LLAMA_TOKEN_ATTR_BYTE) { // ③ 字节 token <0xHH> → 单字节
                char byte = (char) token_to_byte(token);
                return _try_copy(&byte, 1);
            }
            break;
        }
        case LLAMA_VOCAB_TYPE_BPE: {
            if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED))
                return _try_copy(token_text.data(), token_text.size());
            if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                if (escape_whitespaces) { // SPM 风格 ▁ 转义
                    std::string result = token_text;
                    llama_unescape_whitespace(result);
                    return _try_copy(result.data(), result.size());
                }
                std::string result = llama_decode_text(token_text); // BPE 字节 token 解码
                return _try_copy(result.data(), result.size());
            }
            if (attr & LLAMA_TOKEN_ATTR_BYTE) {
                char byte = (char) token_to_byte(token);
                return _try_copy(&byte, 1);
            }
            break;
        }
        case LLAMA_VOCAB_TYPE_RWKV: {
            // ④ RWKV 转义解码 \xHH \t \n \r
            std::vector<uint8_t> result = llama_unescape_rwkv_token(token_text);
            if (result.size() > (size_t)length) return -(int)result.size();
            memcpy(buf, result.data(), result.size());
            return (int)result.size();
        }
        case LLAMA_VOCAB_TYPE_PLAMO2: {
            if (vocab.is_byte(token)) { // ⑤ 字节 token 解码
                int hex_val = std::stoi(token_text.substr(3, 2), nullptr, 16);
                if (length < 1) return -1;
                buf[0] = static_cast<char>(hex_val);
                return 1;
            }
            std::string result = token_text;
            return _try_copy(result.data(), result.size());
        }
        default: GGML_ABORT("fatal error");
    }
    return 0;
}
```

这段代码实现了**按 vocab 类型差异化的反序列化**：
- SPM/UGM：普通 token 需 `▁`→空格，字节 token 解 `<0xHH>`。
- BPE：可选 `escape_whitespaces` 控制是否做 `▁`→空格；普通 token 走 `llama_decode_text`（处理 GPT-2 字节编码字符）。
- RWKV：统一走转义解码器。
- PLaMo2：字节 token 解 `<0xHH>`，其余原样。

### 15.9.4 llama_vocab::impl::detokenize()：三遍清理算法还原自然文本

源码路径：`src/llama-vocab.cpp` - `llama_vocab::impl::detokenize()`（第 3433-3505 行）

```cpp
int32_t detokenize(const llama_token * tokens, int32_t n_tokens, char * text,
                   int32_t text_len_max, bool remove_special, bool unparse_special) const {
    // ... 前置处理：可选移除 BOS/EOS、计算可用空间 ...

    // === 逐 token 写入 ===
    int32_t avail = text_len_max;
    int32_t total = 0;
    bool remove_space = add_space_prefix;
    for (int32_t i = 0; i < n_tokens; ++i) {
        int32_t n_chars = token_to_piece(tokens[i], text, avail, remove_space, unparse_special);
        remove_space = false;
        if (n_chars < 0) { avail = 0; total -= n_chars; }
        else if (n_chars > 0) { avail -= n_chars; text += n_chars; total += n_chars; }
    }
    if (total > text_len_max) return -total;

    // === 三遍清理 ===
    if (clean_spaces) {
        text -= total;
        // 第 1 遍：去除 ? ! . , 前的多余空格
        const int32_t total1 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total1; ++i) {
            const char x = text[i];
            if (text[i-1] == ' ' && (x=='?'||x=='!'||x=='.'||x==',')) total--;
            text[total++] = x;
        }

        // 第 2 遍：合并被空格包围的撇号 ' '
        const int32_t total2 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total2; ++i) {
            const char x = text[i];
            if (x == '\'' && i+1 < total2 && text[i-1]==' ' && text[i+1]==' ') {
                total--; text[++i] = '\0'; // 跳过前空格，撇号后跟 \0 标记
            }
            text[total++] = x;
        }

        // 第 3 遍：处理英语缩写 's 'm 're 've 'll (去前空格)，保留 't 'd (不去空格)
        const int32_t total3 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total3; ++i) {
            const char x = text[i];
            if (text[i-1] == ' ' && x == '\'' && i+1 < total3) {
                const char x1 = text[i+1];
                if (x1 == 't' || x1 == 'd') { /* 保留空格 */ }
                else if (x1 == 's' || x1 == 'm') { total--; } // 去空格
                else if (i+2 < total3) {
                    const char x2 = text[i+2];
                    if (x1=='l' && x2=='l') { /* 保留 */ }
                    else if ((x1=='r' && x2=='e') || (x1=='v' && x2=='e')) { total--; }
                }
            }
            text[total++] = x;
        }
    }
    return total <= text_len_max ? total : -total;
}
```

这段代码是 **detokenize 后处理的「美化器」**。`clean_spaces` 默认开启（BPE/SPM 模型），三遍原地覆盖写入：
1. **标点前空格消除**：`hello ?` → `hello?`
2. **孤立撇号合并**：`he ' s` → `he's`（中间空格被吞，撇号后写 `\0` 标记下一轮跳过）
3. **缩写空格选择性消除**：`'s` `'m` `'re` `'ve` `'ll` 前空格去掉；`'t` `'d` 保留空格（如 `do not` → `don 't` 风格）。这套启发式规则针对英语书写习惯，显著提升生成文本可读性。

---

## 15.10 设计中的取舍

- 为什么不用 ICU 做 Unicode 规范化？
  ICU 体积大、依赖重，llama.cpp 追求「零依赖、单文件可运行」。自研 `unicode_cpts_normalize_nfd` 仅依赖静态表（`unicode-data.cpp` 约 200KB），编译后无外部动态库，完美契合嵌入式/移动端部署。

- 为什么 SPM 用 score（概率）而 BPE 用 rank（顺序）？
  SPM 训练目标是最大化语言模型似然，每个 merge 带有对数概率 score，推理时按 score 贪心近似全局最优。BPE 训练是纯频次统计，merge 顺序固化为 rank，推理严格按 rank 顺序合并即可复现训练时的分词结果，无需存储浮点 score，节省词表体积。

- 为什么 `tokenizer_st_partition` 用 `forward_list` 而非 `vector`？
  特殊 token 切分涉及大量**中间插入/删除**。`forward_list` 支持 O(1) `insert_after`/`erase_after` 且无内存搬移，迭代器不失效。`vector` 每次插入需移动后续元素，且迭代器失效会极大复杂化切分逻辑。

- `detokenize` 三遍清理为何不合并为单遍？
  三遍清理有**强顺序依赖**：第 1 遍处理标点前空格，可能产生新的「空格+撇号+空格」模式供第 2 遍处理；第 2 遍合并撇号后，可能形成 `'s` 等缩写供第 3 遍处理。合并单遍需维护复杂状态机，代码可读性与维护性大幅下降，且当前三遍 O(N) 总复杂度已足够快。

- UGM 的 XCDA 双数组为何不用标准 Darts/Double-Array Trie？
  标准双数组需两个 `int` 数组（BASE/CHECK），XCDA 将 BASE/CHECK/LEAF/VALUE 打包进单个 `uint32_t`，内存占用降低 50%+。位运算解码在现代 CPU 上极快，且 GGUF 存储时直接序列化该数组，无需二次构建，加载即用。

## 15.11 动手练习

1. **阅读Unicode编码与规范化模块**
   - 阅读 `src/unicode.cpp` 第23-130行（`unicode_cpt_from_utf8`、`unicode_cpts_from_utf8`）以及第388-490行（NFD规范化、tolower、is_han），理解：
     1. UTF-8字节序列长度的查表逻辑（`unicode_len_utf8`）
     2. 多字节UTF-8字符解码时如何处理截断或非法字节
     3. NFD规范化如何把 `é` (U+00E9) 折叠到 `e` (U+0065) + U+0301 重音
     4. `unicode_cpt_is_han` 检查的9个CJK区间及其在Kimi-K2分词中的作用
   - 回答问题：
     - 为什么 `unicode_cpts_from_utf8` 选择 `0xFFFD` 替换非法输入而不是抛出异常？
     - `unicode_tolower` 采用二分查找，调用次数极多，是否应该改为哈希表以提升性能？
     - 如果不调用 `unicode_cpts_normalize_nfd`，WPM分词会遇到什么问题？

2. **追踪SPM分词器的合并过程**
   - 阅读 `src/llama-vocab.cpp` 第104-211行（`llm_tokenizer_spm_session`）以及第68-82行（`llm_bigram_spm`），理解：
     1. 优先级队列的排序规则：`(l.score < r.score) || (l.score == r.score && l.left > r.left)`
     2. `try_add_bigram` 中 `rev_merge` 的双重作用：既是work_queue的来源，也是resegment的逆向索引
     3. 合并时如何跳过已被合并的符号（`left_sym.n == 0 || right_sym.n == 0`）
     4. 词表外子串如何通过 `byte_to_token` 退化为字节token
   - 回答问题：
     - 为什么 SPM 的比较器用 `<` 而 BPE 的比较器用 `>`？
     - 如果 `rev_merge` 中某个 `text` 对应到已被合并的符号，resegment 会发生什么？
     - SPM 没有显式的 rank 表，它如何保证与训练时的一致性？

3. **对比BPE与SPM合并策略的差异**
   - 阅读 `src/llama-vocab.cpp` 第260-540行（`llm_bigram_bpe`、`llm_tokenizer_bpe`）以及第700-848行（`llm_tokenizer_bpe_session`），理解：
     1. `llm_tokenizer_bpe` 构造函数中针对40+种 `LLAMA_VOCAB_PRE_TYPE` 的 regex 表达式分发
     2. `add_new_bigram` 通过 `vocab.find_bpe_rank()` 查 BPE rank 表，未找到则跳过
     3. 合并校验 `left_token + right_token != bigram.text` 如何过滤过期的 bigram
     4. `symbols_final` 跨 word 链表的设计：BPE 内部 word 独立合并，最终拼接时维护 next/prev
   - 回答问题：
     - 为什么 LLaMA-3 的 regex 把 `(?i:'s|...)` 改写为 `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])`？
     - BPE 跨 word 链表与 SPM 的全局链表有何本质区别？
     - 如果一个 pair 在BPE rank表中不存在但属于常见子串（如「th」），BPE 会如何处理？

4. **理解WPM/UGM/RWKV/PLaMo2的差异化设计**
   - 阅读 `src/llama-vocab.cpp` 第870-1720行（WPM/UGM/RWKV/PLaMo2各session），理解：
     1. **WPM**：最长前缀匹配 + `token_unk` 兜底；`preprocess` 用 NFD+小写+标点/汉字分词
     2. **UGM**：Viterbi DP 算法 `tokenization_results[input_offset]` 存储最优解；`xcda_array_view` 的位运算 BASE/LCHECK/LEAF 解码
     3. **RWKV**：`naive_trie` 字典树 + `llama_unescape_rwkv_token` 的 `\xNN` 转义解析
     4. **PLaMo2**：把 Trie 展平为 `[piece_length,token_id,score,piece_id]` 四列表 + `scores/path` DP
   - 回答问题：
     - WPM 与 SPM 的最大区别是什么？
     - UGM 的 Viterbi 算法为什么需要 `user_defined_token_matcher` 优先匹配？
     - RWKV 的 `naive_trie::get_longest_prefix` 是否比 `std::unordered_map` 更快？为什么？
     - PLaMo2 的 `to_suffix_id_` 编码 `piece_code = (cpt << 32) | suffix_id` 的目的是什么？

5. **分析特殊token切分与detokenize清理逻辑**
   - 阅读 `src/llama-vocab.cpp` 第2497-2590行（`tokenizer_st_partition`）以及第2960-3050行（`detokenize`），理解：
     1. `cache_special_tokens` 按 text 长度倒序排序的原因（避免短token抢占长token）
     2. `LSTRIP`/`RSTRIP` 属性如何剥离特殊token前后的空格
     3. `detokenize` 的三遍清理算法：去除 `?`,`!`,`.`,`,` 前的多余空格、合并 ` ' ` 周围空格、处理 `'s`,`'m`,`'re`,`'ve` 缩写
     4. `token_to_piece` 在不同vocab_type下的差异化处理：SPM的`▁`↔空格、BPE的字节token解码、RWKV的转义解码
   - 回答问题：
     - 为什么 `tokenizer_st_partition` 用前向列表 `std::forward_list` 而不是 `std::list`/`std::vector`？
     - 如果一个特殊token文本是另一个特殊token的前缀（如 `<|` 和 `<|im_start|`），长度倒序如何保证正确切分？
     - `clean_spaces = false` 的模型（如 SPM/Qwen2）为何不需要三遍清理？
     - 在 `detokenize` 第三遍中，为何仅对 `'s`/`'m` 等缩写去空格，而保留 `'t`/`'d` 的空格？

## 15.12 本章小结

这一章中我们学习了 llama.cpp 完整的词汇表与分词系统。首先我们了解了 Unicode 支持层：UTF-8 字节流与码点的双向转换、NFD 规范化、大小写映射、汉字识别、全码点属性表构建，以及 GPT-2 字节回退映射。接着我们深入了正则表达式预分词分发器：顶层 `unicode_regex_split` 通过 collapsed 映射让 `std::regex` 支持 Unicode 类别，并为 GPT-2/LLaMA-3/Kimi-K2/AFMOE/Gemma4 提供了手写的 O(N) 线性扫描实现。随后我们剖析了 `llama_vocab` 的 pimpl 架构与 GGUF 加载流水线，理解了特殊 token 的显式配置与启发式兜底、EOG 集合构建、缓存预生成机制。然后我们逐一拆解了六大分词器实现：SPM 的优先级队列贪心合并与 rev_merge 兜底、BPE 的 rank 表驱动合并与跨词链表、WPM 的最长前缀匹配与 UNK 整词回退、UGM 的 Viterbi DP 与 XCDA 双数组加速归一化、RWKV 的 naive_trie 字典树与转义解码、PLaMo2 的 Trie 展平四表与后缀感知动态规划。最后我们详细阅读了分词总入口 `tokenize` 与反分词 `detokenize` 的三遍清理算法，以及 C API 的封装层。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| `unicode_cpt_to_utf8/from_utf8` | UTF-8字节与Unicode码点的双向通道，错误用0xFFFD替换 |
| `unicode_cpts_normalize_nfd` | NFD规范化，将带音标字符拆为基本字符+组合标记，二分查找 |
| `unicode_tolower` | ASCII二分查表大小写转换，支持LLaMA-3的大小写不敏感正则 |
| `unicode_cpt_is_han` | CJK汉字区间判断（9个Unicode block），支持Kimi-K2中文分词 |
| `unicode_cpt_flags_array` | 预构造1.1M码点类别位标志数组，O(1)查表 |
| `unicode_regex_split_custom_gpt2/llama3/kimi_k2/afmoe/newlines` | 模型专属预分词器，通过分发器自动选择具体实现 |
| `unicode_regex_split_custom` | 正则字符串→具体分词函数的switch分发器，未匹配时回退到std::regex |
| `unicode_byte_to_utf8/utf8_to_byte` | GPT-2字节回退双向映射，0-255字节对应256个不同Unicode占位字符 |
| `naive_trie` | RWKV/PLaMo2用的字典树，insert/get_longest_prefix/traverse三件套 |
| `llama_vocab::impl::load` | 从GGUF元数据加载tokenizer.model/pre/list/score/type四要素 |
| `llama_vocab::impl::init_tokenizer` | 根据vocab_type构造SPM/BPE/WPM/UGM/RWKV/PLaMo2具体分词器 |
| `llm_tokenizer_spm_session::tokenize` | SPM三阶段：链表初始化→seed队列→贪心合并score最高的 pair |
| `llm_tokenizer_spm_session::resegment` | SPM兜底逻辑：查rev_merge逆向索引或按字节回退 |
| `llm_tokenizer_bpe_session::tokenize` | BPE按regex切word→char链表→按rank表贪心合并，校验过期bigram |
| `llm_tokenizer_bpe_session::add_new_bigram` | 查询vocab.find_bpe_rank()，未在merge表中则跳过 |
| `llm_tokenizer_bpe_session::append_bos/eos/check_double_bos_eos` | BPE session的BOS/EOS自动注入与重复警告 |
| `llm_tokenizer_wpm_session::preprocess` | WPM：NFD规范化→小写化→标点/汉字分词→最长前缀匹配 |
| `llm_tokenizer_ugm_session::tokenize` | Unigram Viterbi DP算法，tokenization_results记录每个位置最优解 |
| `llm_tokenizer_ugm_session::normalize_prefix` | 通过XCDA紧凑双数组O(1)查找最长规范化前缀 |
| `llm_tokenizer_rwkv_session::tokenize` | RWKV：naive_trie遍历+转义解码（\xNN→原始字节） |
| `llm_tokenizer_plamo2::encode` | PLaMo2：展平Trie为[piece_length,token_id,score,piece_id]四列表+DP回溯 |
| `tokenizer_st_partition` | 特殊token切分器，按长度倒序匹配避免短token抢占长token |
| `token_to_piece_for_cache` | 构造token→piece的预生成缓存，避免推理时重复计算 |
| `special_eog_ids` | 终止生成token集合，由token文本启发式匹配+KV元数据构建 |
| `cache_special_tokens` | 控制token按text长度倒序缓存，O(1)用于运行时扫描用 |
| `detokenize三遍清理` | first pass去多余空格、second pass合并' '、third pass处理's/re/ve缩写 |
| `escape_whitespaces` | SPM风格' '↔▁转换，BPE模型可选启用 |
| `llama_unescape_rwkv_token` | RWKV的\xNN/\t/\n/\r转义序列解析为原始字节 |
| `xcda_array_view` | BASE/LCHECK/LEAF三位位运算解码SentencePiece压缩双数组 |

下一章中，我们将学习**批处理与微批次调度 —— 推理任务的"交通指挥中心"**，理解 llama.cpp 如何接收用户输入的 `llama_batch`、将其校验与补全，再按策略拆分为微批次 `llama_ubatch`，准备送入计算图。
