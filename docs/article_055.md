
# 第55章 工具函数与辅助能力 —— common 模块的「瑞士军刀」

## 学习目标

- 理解 Base64 双字母表设计（标准 vs URL 安全）及自动检测机制的实现原理
- 掌握 UTF-8 变长编码的位级解析技巧，包括首字节模式识别与多字节拼接
- 解析反向部分匹配正则的转换算法，理解流式前缀匹配的实现策略
- 掌握 GGML 张量调试工具的设计，包括类型安全访问、跨设备数据拷贝与结构化打印
- 理解正则过滤器在多模态调试中的应用，掌握回调数据结构的零拷贝或低开销数据获取策略
- 能运用这些底层工具处理二进制数据、Unicode 文本、流式解析及模型调试场景

## 55.1 生活类比

想象这些工具是**国际机场的地面支持系统**：Base64 就像行李与货物标签系统，标准标签用于普通通道，URL 安全标签用于特殊货物通道（避免 `+/` 在传送带造成混乱），自动检测如同扫码枪智能识别标签类型；Unicode 处理如同多语言海关查验，UTF-8 序列长度判断是快速查看护照类型；码点解析是将签证页（多字节）翻译成统一语言；完整性检查像是确认旅客是否拿齐了所有登机牌碎片；正则部分匹配则是行李传送带监控系统，不仅识别完全匹配标签的行李，还能识别正在传送带尽头（字符串末尾）部分可见、即将完全符合标签规则的行李；张量调试则是货物 X 光检测仪，`common_debug_cb_eval` 是固定在流水线旁的扫描仪，当货物（张量）经过时自动拍照（打印）；`base_callback_data` 是安检员的 filtering 清单，只关注特定类型的货物。就像机场需要这些系统协同确保行李正确流转，llama.cpp 需要这些工具确保数据在预处理、解析和调试阶段准确无误。

## 55.2 源码地图

common/base64.hpp
├── base64_error                      # 异常类（继承 std::runtime_error）
│   └── base64_error()                # 构造函数（21-28行）
├── base64                            # 编解码核心类
│   ├── encode()                      # 编码（迭代器/string/数组重载，64-159行）
│   ├── decode()                      # 解码（迭代器/string/数组重载，160-252行）
│   ├── decode_inplace()              # 就地解码节省内存（253-282行）
│   ├── max_decode_size()             # 计算解码缓冲区大小（283-297行）
│   ├── required_encode_size()        # 计算编码缓冲区大小（298-312行）
│   └── _base64_value()               # 字符映射与字母表自动检测（313-352行）
└── __main__                          # 头文件保护宏与标准库包含（1-47行）

common/unicode.h
├── struct utf8_parse_result          # 解析结果（码点/字节数/状态）
│   └── utf8_parse_result()           # 构造函数（8-9行）
├── common_utf8_sequence_length()     # 序列长度查询接口（20-22行）
├── common_utf8_is_complete()         # 末尾完整性检查接口（23-24行）
├── common_parse_utf8_codepoint()     # 码点解析接口（25行）
├── common_unicode_cpts_to_utf8()     # 多码点编码接口（26行）
├── common_unicode_cpt_to_utf8()      # 单码点编码接口（27行）
└── __main__                          # 头文件保护宏与类型定义（1-27行）

common/unicode.cpp
├── common_utf8_sequence_length()     # 首字节→序列长度查找表（10-14行）
├── common_parse_utf8_codepoint()     # 流式码点解析（16-61行）
├── common_utf8_is_complete()         # 末尾完整性检查（63-77行）
├── common_unicode_cpts_to_utf8()     # 多码点编码（79-85行）
└── common_unicode_cpt_to_utf8()      # 单码点UTF-8编码（87-114行）

common/regex-partial.h
├── common_string_range               # 字符串区间（起止位置）
│   ├── common_string_range(size_t, size_t)  # 构造函数带校验（13-17行）
│   ├── empty()                       # 空范围判断（20-22行）
│   └── operator==()                  # 相等比较（23-25行）
├── common_regex_match                # 匹配结果（FULL/PARTIAL类型+分组）
│   ├── operator==()                  # 相等比较（37-39行）
│   └── operator!=()                  # 不等比较（40-42行）
├── common_regex                      # 正则封装类
│   ├── common_regex()                # 构造函数（正向+反向正则，50行）
│   ├── search()                      # 统一搜索接口（52行）
│   └── str()                         # 获取原始模式（54行）
└── regex_to_reversed_partial_regex() # 转换函数声明（58行）

common/regex-partial.cpp
├── common_regex::common_regex()      # 构造函数实现（8-10行）
├── common_regex::search()            # 双重匹配逻辑（12-44行）
└── regex_to_reversed_partial_regex() # 核心转换算法（46-145行）
    ├── process lambda                # 递归解析：序列/选择/分组/量词
    ├── 字符类处理                    # [...] 原样保留
    ├── 量词展开                      # {n,m} → n必选 + (m-n)可选
    └── 序列反转                      # abcd → ^(?:(?:(?:d)?c)?b)?a

common/debug.h
├── common_debug_print_tensor<>       # 模板打印函数声明（13行）
├── common_debug_cb_eval<>            # 图执行回调声明（21行）
└── base_callback_data                # 回调数据与过滤器
    ├── tensor_filters                # 正则过滤器列表
    ├── data                          # 设备数据缓冲区
    ├── base_callback_data()          # 默认构造（27行）
    └── base_callback_data(params, patterns) # 带参构造（29-41行）

common/debug.cpp
├── common_ggml_ne_string()           # 维度格式化（7-17行）
├── common_ggml_get_float_value()     # 类型安全元素访问（19-46行）
├── common_debug_print_tensor<>()     # 智能截断打印（50-97行）
├── common_debug_cb_eval<>()          # 图执行调试回调（110-153行）
└── __main__                          # 显式模板实例化 <true>/<false>（156-159行）

## 55.3 Base64 编码解码 —— 二进制与文本的「翻译官」

Base64 是将二进制数据编码为可打印 ASCII 字符的通用方案。llama.cpp 实现了一个头文件级的 `base64` 类，支持标准字母表（`A-Z a-z 0-9 + /`）与 URL 安全字母表（`A-Z a-z 0-9 - _`），并提供自动检测、三级错误处理、就地解码等工程级特性。

### 55.3.1 base64_error：异常安全的「红灯」

```cpp
// common/base64.hpp (第21-28行)
class base64_error : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
```

这段代码定义了一个轻量级异常类，直接继承 `std::runtime_error` 的构造函数，用于在解码遇到非法字符或格式错误时抛出。所有编解码失败最终都汇聚到此异常，便于上层统一捕获。

### 55.3.2 编码核心：三字节到四字符的位魔术

```cpp
// common/base64.hpp (第64-127行)
template<typename Input_iterator, typename Output_iterator>
static Output_iterator encode(Input_iterator in_begin, Input_iterator in_end, Output_iterator out,
                              alphabet alphabet = alphabet::standard)
{
    constexpr auto pad = '=';                                              // 填充字符
    const char* alpha  = alphabet == alphabet::url_filename_safe
                            ? "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
                            : "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; // 选择字母表

    while (in_begin != in_end) {                                           // 逐 3 字节处理
        std::uint8_t i0 = 0, i1 = 0, i2 = 0;
        i0 = static_cast<std::uint8_t>(*in_begin);                         // 读取第 1 字节
        ++in_begin;
        *out = alpha[i0 >> 2 & 0x3f];                                      // 高 6 位
        ++out;

        if (in_begin != in_end) {                                          // 第 2 字节存在
            i1 = static_cast<std::uint8_t>(*in_begin);
            ++in_begin;
            *out = alpha[((i0 & 0x3) << 4) | (i1 >> 4 & 0x0f)];           // 低 2 位 + 高 4 位
            ++out;
        } else {                                                           // 仅 1 字节：补两个 =
            *out = alpha[(i0 & 0x3) << 4];
            ++out;
            *out = pad; ++out;
            *out = pad; ++out;
            break;
        }

        if (in_begin != in_end) {                                          // 第 3 字节存在
            i2 = static_cast<std::uint8_t>(*in_begin);
            ++in_begin;
            *out = alpha[((i1 & 0xf) << 2) | (i2 >> 6 & 0x03)];           // 低 4 位 + 高 2 位
            ++out;
        } else {                                                           // 仅 2 字节：补一个 =
            *out = alpha[(i1 & 0xf) << 2];
            ++out;
            *out = pad; ++out;
            break;
        }

        *out = alpha[i2 & 0x3f];                                           // 低 6 位
        ++out;
    }
    return out;
}
```

这段代码实现了标准的 Base64 编码算法：每 3 字节（24 位）拆成 4 个 6 位组，通过 `& 0x3f` 取值并查字母表。模板化迭代器接口同时支持 `std::string`、原始指针等任意输入输出范围。字母表选择在编译期分支完成，零运行时开销。

### 55.3.3 字符串重载与缓冲区预估

```cpp
// common/base64.hpp (第128-159行)
static std::string encode(const std::string& str, alphabet alphabet = alphabet::standard)
{
    std::string result;
    result.reserve(required_encode_size(str.length()) + 1);  // 预分配避免扩容
    encode(str.begin(), str.end(), std::back_inserter(result), alphabet);
    return result;
}

static std::string encode(const char* buffer, std::size_t size, alphabet alphabet = alphabet::standard)
{
    std::string result;
    result.reserve(required_encode_size(size) + 1);
    encode(buffer, buffer + size, std::back_inserter(result), alphabet);
    return result;
}
```

这两个重载提供面向 `std::string` 和 C 风格数组的便捷入口。`required_encode_size` 精确计算所需空间（含填充），配合 `reserve` 实现一次分配、零拷贝追加。

### 55.3.4 解码反向工程：6 位到 8 位的重组

```cpp
// common/base64.hpp (第160-220行)
template<typename Input_iterator, typename Output_iterator>
static Output_iterator decode(Input_iterator in_begin, Input_iterator in_end, Output_iterator out,
                              alphabet alphabet          = alphabet::auto_,
                              decoding_behavior behavior = decoding_behavior::moderate)
{
    std::uint8_t last  = 0;    // 上一个 6 位值
    auto bits          = 0;    // 当前累积的有效位数

    while (in_begin != in_end) {
        auto c = *in_begin;
        ++in_begin;
        if (c == '=') {                    // 遇到填充提前结束
            break;
        }
        auto part = _base64_value(alphabet, c);  // 字符 → 6 位值（含自动检测）
        if (bits + 6 >= 8) {               // 可输出一个完整字节
            *out = (last << (8 - bits)) | (part >> (bits - 2));
            ++out;
            bits -= 2;                     // 剩余 2 位留给下一轮
        } else {
            bits += 6;                     // 累积 6 位
        }
        last = part;
    }

    // 尾部填充字符校验
    if (behavior != decoding_behavior::loose) {
        while (in_begin != in_end) {
            auto c = *in_begin;
            ++in_begin;
            if (c != '=') {
                throw base64_error("invalid base64 character.");
            }
        }
    }
    return out;
}
```

解码采用流式状态机：`last` 保存上一个 6 位值，`bits` 记录当前缓冲的有效位数。每次读入新的 6 位 `part`，若 `bits+6 >= 8` 则组合出一个字节输出，剩余 2 位留待下一轮。这种“跨边界位拼接”避免了中间缓冲区，极其节省内存。

### 55.3.5 _base64_value：字符映射与字母表自动检测

```cpp
// common/base64.hpp (第313-352行)
static std::uint8_t _base64_value(alphabet& alphabet, char c)
{
    if (c >= 'A' && c <= 'Z') { return c - 'A'; }
    else if (c >= 'a' && c <= 'z') { return c - 'a' + 26; }
    else if (c >= '0' && c <= '9') { return c - '0' + 52; }

    if (alphabet == alphabet::standard) {                 // 显式标准表
        if (c == '+') return 62;
        else if (c == '/') return 63;
    } else if (alphabet == alphabet::url_filename_safe) { // 显式 URL 安全表
        if (c == '-') return 62;
        else if (c == '_') return 63;
    } else {                                               // 自动检测模式
        if (c == '+') { alphabet = alphabet::standard; return 62; }
        else if (c == '/') { alphabet = alphabet::standard; return 63; }
        else if (c == '-') { alphabet = alphabet::url_filename_safe; return 62; }
        else if (c == '_') { alphabet = alphabet::url_filename_safe; return 63; }
    }
    throw base64_error("invalid base64 character.");
}
```

这是自动检测的核心：当用户传入 `alphabet::auto_` 时，首次遇到 `+/` 或 `_-` 即锁定字母表类型，后续字符按该表严格校验。这种“延迟绑定”设计兼顾了灵活性与安全性。

### 55.3.6 就地解码与缓冲区计算

```cpp
// common/base64.hpp (第253-282行)
static void decode_inplace(std::string& str, alphabet alphabet = alphabet::auto_,
                           decoding_behavior behavior = decoding_behavior::moderate)
{
    str.resize(decode(str.begin(), str.end(), str.begin(), alphabet, behavior) - str.begin());
}

static char* decode_inplace(char* str, std::size_t size, alphabet alphabet = alphabet::auto_,
                            decoding_behavior behavior = decoding_behavior::moderate)
{
    return decode(str, str + size, str, alphabet, behavior);
}
```

`decode_inplace` 直接在原字符串/数组上操作，输出迭代器即为输入迭代器。由于解码后数据更短（3/4），不会发生覆盖冲突。返回新长度/指针，调用者 `resize` 即可截断多余尾部。

```cpp
// common/base64.hpp (第283-312行)
static std::size_t max_decode_size(std::size_t size) noexcept
{
    return (size / 4 + (size % 4 ? 1 : 0)) * 3;  // 向上取整到 4 的倍数，每 4 字符产出 3 字节
}

static std::size_t required_encode_size(std::size_t size) noexcept
{
    return (size / 3 + (size % 3 ? 1 : 0)) * 4;  // 向上取整到 3 的倍数，每 3 字节产出 4 字符
}
```

这两个工具函数用于预分配缓冲区：`max_decode_size` 给出解码可能的最大字节数（用于 `reserve`），`required_encode_size` 给出编码所需的确切字符数（含填充）。

---

## 55.4 Unicode 处理 —— 字符编码的「通关文牒」

UTF-8 是变长编码（1~4 字节），llama.cpp 提供了一套流式感知、边界安全的解析与编码工具，覆盖序列长度判定、码点解析、完整性检查、反向编码全流程。

### 55.4.1 utf8_parse_result：解析结果的「三态护照」

```cpp
// common/unicode.h (第8-9行)
struct utf8_parse_result {
    uint32_t codepoint;
    size_t bytes_consumed;
    enum status { SUCCESS, INCOMPLETE, INVALID } status;

    utf8_parse_result(enum status s, uint32_t cp = 0, size_t bytes = 0)
        : codepoint(cp), bytes_consumed(bytes), status(s) {}
};
```

解析结果封装了三种状态：`SUCCESS`（完整码点）、`INCOMPLETE`（输入截断，需更多数据）、`INVALID`（非法字节序列）。这种三态设计完美适配网络流、文件流等增量输入场景。

### 55.4.2 序列长度快速判定：首字节的四位掩码

```cpp
// common/unicode.cpp (第10-14行)
size_t common_utf8_sequence_length(unsigned char first_byte)
{
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(first_byte) >> 4;
    return lookup[highbits];
}
```

利用首字节高 4 位作为索引，O(1) 查表返回序列长度：`0xxx`→1、`110x`→2、`1110`→3、`11110`→4。连续字节 `10xxxxxx` 高 4 位为 `8/9/A/B`，查表返回 0，隐式标记为非法起始字节。无分支、零内存分配，极致高效。

### 55.4.3 流式感知的码点解析

```cpp
// common/unicode.cpp (第16-61行)
utf8_parse_result common_parse_utf8_codepoint(std::string_view input, size_t offset)
{
    if (offset >= input.size()) {
        return utf8_parse_result(utf8_parse_result::INCOMPLETE);  // 越界即不完整
    }
    if (!(input[offset] & 0x80)) {  // 0xxxxxxx：单字节 ASCII
        return utf8_parse_result(utf8_parse_result::SUCCESS, input[offset], 1);
    }
    if (!(input[offset] & 0x40)) {  // 10xxxxxx：非法起始字节
        return utf8_parse_result(utf8_parse_result::INVALID);
    }
    if (!(input[offset] & 0x20)) {  // 110xxxxx：两字节序列
        if (offset + 1 >= input.size()) return utf8_parse_result(utf8_parse_result::INCOMPLETE);
        if ((input[offset + 1] & 0xc0) != 0x80) return utf8_parse_result(utf8_parse_result::INVALID);
        auto result = ((input[offset] & 0x1f) << 6) | (input[offset + 1] & 0x3f);
        return utf8_parse_result(utf8_parse_result::SUCCESS, result, 2);
    }
    if (!(input[offset] & 0x10)) {  // 1110xxxx：三字节序列
        if (offset + 2 >= input.size()) return utf8_parse_result(utf8_parse_result::INCOMPLETE);
        if ((input[offset + 1] & 0xc0) != 0x80 || (input[offset + 2] & 0xc0) != 0x80)
            return utf8_parse_result(utf8_parse_result::INVALID);
        auto result = ((input[offset] & 0x0f) << 12) | ((input[offset + 1] & 0x3f) << 6) | (input[offset + 2] & 0x3f);
        return utf8_parse_result(utf8_parse_result::SUCCESS, result, 3);
    }
    if (!(input[offset] & 0x08)) {  // 11110xxx：四字节序列
        if (offset + 3 >= input.size()) return utf8_parse_result(utf8_parse_result::INCOMPLETE);
        if ((input[offset + 1] & 0xc0) != 0x80 || (input[offset + 2] & 0xc0) != 0x80 || (input[offset + 3] & 0xc0) != 0x80)
            return utf8_parse_result(utf8_parse_result::INVALID);
        auto result = ((input[offset] & 0x07) << 18) | ((input[offset + 1] & 0x3f) << 12) |
                      ((input[offset + 2] & 0x3f) << 6) | (input[offset + 3] & 0x3f);
        return utf8_parse_result(utf8_parse_result::SUCCESS, result, 4);
    }
    return utf8_parse_result(utf8_parse_result::INVALID);  // 11111xxx：非法
}
```

解析器按位模式层层分支，优先处理 ASCII 快速路径。每种多字节序列先检查剩余长度（`INCOMPLETE`），再验证后续字节均为 `10xxxxxx` 格式（否则 `INVALID`），最后通过位掩码与移位拼接码点。这种“长度优先、格式次之”的顺序最小化了分支预测失败。

### 55.4.4 完整性检查的边界处理

```cpp
// common/unicode.cpp (第63-77行)
bool common_utf8_is_complete(const std::string& s)
{
    if (s.empty()) return true;
    for (int i = 1; i <= std::min(4, (int)s.size()); i++) {
        unsigned char c = s[s.size() - i];
        if ((c & 0xC0) != 0x80) {  // 找到非续字节（即起始字节）
            int expected = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
            return i >= expected;  // 已读取字节数是否 ≥ 预期长度
        }
    }
    return false;  // 全是续字节，极其异常
}
```

从字符串末尾向前扫描最多 4 字节，找到第一个非 `10xxxxxx` 字节（即序列起始字节），根据其高位确定预期长度，对比已扫描字节数。这完美解决了“流式输入末尾是否被截断”的问题。

### 55.4.5 码点到 UTF-8 的编码转换

```cpp
// common/unicode.cpp (第87-114行)
std::string common_unicode_cpt_to_utf8(uint32_t cpt)
{
    std::string result;
    if (cpt <= 0x7f) {                       // 1 字节
        result.push_back(cpt);
        return result;
    }
    if (0x80 <= cpt && cpt <= 0x7ff) {       // 2 字节
        result.push_back(0xc0 | ((cpt >> 6) & 0x1f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x800 <= cpt && cpt <= 0xffff) {     // 3 字节
        result.push_back(0xe0 | ((cpt >> 12) & 0x0f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x10000 <= cpt && cpt <= 0x10ffff) { // 4 字节
        result.push_back(0xf0 | ((cpt >> 18) & 0x07));
        result.push_back(0x80 | ((cpt >> 12) & 0x3f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    throw std::invalid_argument("invalid codepoint");  // 超出 Unicode 范围
}
```

反向编码严格遵循 UTF-8 位模板：根据码点范围选择字节数，用移位和掩码填充各字节模板位。非法码点（>0x10FFFF 或代理对区间）直接抛异常，保证输出合法性。

---

## 55.5 正则部分匹配 —— 反向字符串的「镜中世界」

流式场景（如逐 token 生成）需要判断“当前文本是否为某模式的前缀”。llama.cpp 通过将正则表达式转换为作用于**反转字符串**的特殊正则，巧妙实现了部分匹配检测。

### 55.5.1 核心数据结构

```cpp
// common/regex-partial.h (第13-25行)
struct common_string_range {
    size_t begin, end;
    common_string_range(size_t begin, size_t end) : begin(begin), end(end) {
        if (begin > end) throw std::runtime_error("Invalid range");
    }
    common_string_range() = delete;  // 强制显式构造
    bool empty() const { return begin == end; }
    bool operator==(const common_string_range& other) const {
        return begin == other.begin && end == other.end;
    }
};
```

轻量级区间表示，构造时强制校验 `begin ≤ end`，防止逻辑错误。

```cpp
// common/regex-partial.h (第27-42行)
struct common_regex_match {
    enum type { COMMON_REGEX_MATCH_TYPE_FULL, COMMON_REGEX_MATCH_TYPE_PARTIAL } type;
    std::vector<common_string_range> groups;

    bool operator==(const common_regex_match& other) const {
        return type == other.type && groups == other.groups;
    }
    bool operator!=(const common_regex_match& other) const { return !(*this == other); }
};
```

匹配结果区分 `FULL`（完整匹配）与 `PARTIAL`（前缀匹配），并记录所有捕获组的位置区间。

### 55.5.2 核心转换算法：从左到右 → 从右到左

```cpp
// common/regex-partial.cpp (第46-145行)
std::string regex_to_reversed_partial_regex(const std::string& pattern)
{
    auto it = pattern.begin();
    const auto end = pattern.end();

    std::function<std::string()> process = [&]() {
        std::vector<std::vector<std::string>> alternatives(1);
        std::vector<std::string>* sequence = &alternatives.back();

        while (it != end) {
            if (*it == '[') {  // 字符类 [...] 原样保留
                auto start = it; ++it;
                while (it != end) {
                    if ((*it == '\\') && (++it != end)) ++it;
                    else if ((it != end) && (*it == ']')) break;
                    else ++it;
                }
                if (it == end) throw std::runtime_error("Unmatched '[' in pattern");
                ++it;
                sequence->push_back(std::string(start, it));
            }
            else if (*it == '*' || *it == '?' || *it == '+') {  // 量词附加到前一元素
                if (sequence->empty()) throw std::runtime_error("Quantifier without preceding element");
                sequence->back() += *it;
                auto is_star = *it == '*'; ++it;
                if (is_star && it != end && *it == '?') ++it;  // *? 非贪婪标记
            }
            else if (*it == '{') {  // {n,m} 量词展开
                if (sequence->empty()) throw std::runtime_error("Repetition without preceding element");
                ++it; auto start = it;
                while (it != end && *it != '}') ++it;
                if (it == end) throw std::runtime_error("Unmatched '{' in pattern");
                auto parts = string_split(std::string(start, it), ",");
                ++it;
                // 解析 min/max...
                auto part = sequence->back(); sequence->pop_back();
                for (int i = 0; i < *min; i++) sequence->push_back(part);      // n 个必选
                if (max) for (int i = *min; i < *max; i++) sequence->push_back(part + "?"); // 可选
                else sequence->push_back(part + "*");                            // 无上界
            }
            else if (*it == '(') {  // 分组递归
                ++it;
                if (it != end && *it == '?' && (it+1 != end) && *(it+1) == ':') it += 2;  // 跳过 (?: 
                auto sub = process();  // 递归处理子表达式
                if (*it != ')') throw std::runtime_error("Unmatched '(' in pattern");
                ++it;
                auto& part = sequence->emplace_back("(?:");
                part += sub; part += ")";
            }
            else if (*it == ')') break;  // 返回上层
            else if (*it == '|') { ++it; alternatives.emplace_back(); sequence = &alternatives.back(); }  // 选择分支
            else if (*it == '\\' && (++it != end)) { sequence->push_back("\\" + std::string(1, *it)); ++it; }  // 转义字符
            else if (it != end) { sequence->push_back(std::string(1, *it)); ++it; }  // 普通字符
        }

        // 构建反向部分匹配模式
        std::vector<std::string> res_alts;
        for (const auto& parts : alternatives) {
            auto& res = res_alts.emplace_back();
            for (size_t i = 0; i < parts.size() - 1; i++) res += "(?:";  // 开 n-1 个非捕获组
            for (auto it = parts.rbegin(); it != parts.rend(); ++it) {    // 反序遍历
                res += *it;
                if (it != parts.rend() - 1) res += ")?";
            }
        }
        return string_join(res_alts, "|");
    };

    auto res = process();
    if (it != end) throw std::runtime_error("Unmatched '(' in pattern");
    return "^(" + res + ")";  // 外层捕获组 + 锚定开头
}
```

**算法解读**：
1. **递归下降解析**：`process` lambda 将模式解析为 `alternatives`（选择分支）的列表，每个分支是 `sequence`（序列元素）的列表。
2. **量词展开**：`{n,m}` 展开为 `n` 个必选副本 + `m-n` 个可选副本（`?`），无上界变 `*`。
3. **序列反转与可选化**：`abcd` → 反序 `dcba` → 包裹成 `^(?:(?:(?:d)?c)?b)?a`。外层 `^` 锚定反转字符串开头（即原字符串末尾），内层嵌套 ` (?:...)? ` 使每个后缀均可匹配。
4. **字符类透传**：`[...]` 内部不解析，整体作为原子单元保留。

最终生成的正则如 `^(?:(?:(?:d)?c)?b)?a`，在反转字符串上 `regex_search` 即可匹配 `a`、`ab`、`abc`、`abcd` 的反转形式。

### 55.5.3 双重匹配搜索接口

```cpp
// common/regex-partial.cpp (第12-44行)
common_regex_match common_regex::search(const std::string& input, size_t pos, bool as_match) const
{
    std::smatch match;
    if (pos > input.size()) throw std::runtime_error("Position out of bounds");
    auto start = input.begin() + pos;

    // 1. 尝试完整匹配
    auto found = as_match
        ? std::regex_match(start, input.end(), match, rx)
        : std::regex_search(start, input.end(), match, rx);
    if (found) {
        common_regex_match res; res.type = COMMON_REGEX_MATCH_TYPE_FULL;
        for (size_t i = 0; i < match.size(); ++i) {
            auto begin = pos + match.position(i);
            res.groups.emplace_back(begin, begin + match.length(i));
        }
        return res;
    }

    // 2. 尝试部分匹配：在反转字符串上搜索
    std::match_results<std::string::const_reverse_iterator> srmatch;
    if (std::regex_search(input.rbegin(), input.rend() - pos, srmatch, rx_reversed_partial,
                          std::regex_constants::match_continuous)) {
        auto group = srmatch[1].str();
        if (group.length() != 0) {
            auto it = srmatch[1].second.base();  // 反向迭代器转正向
            if ((!as_match) || it == input.begin()) {
                common_regex_match res; res.type = COMMON_REGEX_MATCH_TYPE_PARTIAL;
                const size_t begin = std::distance(input.begin(), it);
                const size_t end = input.size();
                if (begin == std::string::npos || end == std::string::npos || begin > end)
                    throw std::runtime_error("Invalid range");
                res.groups.push_back({begin, end});
                return res;
            }
        }
    }
    return {};  // 无匹配
}
```

搜索采用“先完整后部分”的策略：`as_match=true` 时要求从 `pos` 处完全匹配（用于精确校验）；否则先 `regex_search` 找完整匹配，失败则在**反转字符串**上用 `rx_reversed_partial` 做 `match_continuous`（从开头连续匹配），对应原字符串的**后缀前缀匹配**。

---

## 55.6 张量调试工具 —— 计算图的「X光透视仪」

调试 GGML 计算图需要在图执行期间检查张量数值、形状、NaN 等。llama.cpp 提供了 `common_debug_print_tensor`、`common_debug_cb_eval` 及配套数据结构，支持跨设备内存访问、正则过滤、智能截断打印。

### 55.6.1 维度格式化与类型安全访问

```cpp
// common/debug.cpp (第7-17行)
static std::string common_ggml_ne_string(const ggml_tensor* t)
{
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) str += ", ";
    }
    return str;
}
```

将 4 维张量规格格式化为 `ne[0], ne[1], ne[2], ne[3]` 字符串，用于日志输出。

```cpp
// common/debug.cpp (第19-46行)
static float common_ggml_get_float_value(const uint8_t* data, ggml_type type,
                                         const size_t* nb, size_t i0, size_t i1, size_t i2, size_t i3)
{
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];  // 步长计算偏移
    float v;
    if (type == GGML_TYPE_F16)        v = ggml_fp16_to_fp32(*(const ggml_fp16_t*)&data[i]);
    else if (type == GGML_TYPE_F32)   v = *(const float*)&data[i];
    else if (type == GGML_TYPE_I64)   v = (float)*(const int64_t*)&data[i];
    else if (type == GGML_TYPE_I32)   v = (float)*(const int32_t*)&data[i];
    else if (type == GGML_TYPE_I16)   v = (float)*(const int16_t*)&data[i];
    else if (type == GGML_TYPE_I8)    v = (float)*(const int8_t*)&data[i];
    else if (type == GGML_TYPE_BF16)  v = ggml_bf16_to_fp32(*(const ggml_bf16_t*)&data[i]);
    else GGML_ABORT("fatal error");
    return v;
}
```

统一入口按类型分发，计算多维索引偏移（`nb` 为各维步长），统一转为 `float` 便于打印比较。量化类型不在此处理，由上层判断跳过。

### 55.6.2 智能截断打印

```cpp
// common/debug.cpp (第50-97行)
template <bool abort>
void common_debug_print_tensor(uint8_t* data, ggml_type type, const int64_t* ne, const size_t* nb, int64_t n)
{
    GGML_ASSERT(n > 0);
    float sum = 0;
    // 首先计算总和（用于 NaN 检测）
    for (int64_t i3 = 0; i3 < ne[3]; i3++)
        for (int64_t i2 = 0; i2 < ne[2]; i2++)
            for (int64_t i1 = 0; i1 < ne[1]; i1++)
                for (int64_t i0 = 0; i0 < ne[0]; i0++)
                    sum += common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);

    // 再打印，维度 > 2n 时截断显示首尾各 n
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG(INDENT "[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) { LOG(INDENT INDENT "..., \n"); i2 = ne[2] - n; }
            LOG(INDENT INDENT "[\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) { LOG(INDENT INDENT INDENT "..., \n"); i1 = ne[1] - n; }
                LOG(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) { LOG("   ..., "); i0 = ne[0] - n; }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) LOG(", ");
                }
                LOG("  ],\n");
            }
            LOG(INDENT INDENT "],\n");
        }
        LOG(INDENT "]\n");
        LOG(INDENT "sum = %f\n", sum);
    }
    if constexpr (abort) {
        if (std::isnan(sum)) { LOG("encountered NaN - aborting\n"); exit(0); }
    }
}
```

模板参数 `abort` 控制是否在检测到 NaN 总和时终止进程。打印采用四层嵌套循环对应 4 维张量，每层维度超过 `2*n` 时跳转到尾部 `n` 个元素，中间用 `...,` 省略。先求和后打印，保证 NaN 检测覆盖全张量。

### 55.6.3 图执行调试回调

```cpp
// common/debug.cpp (第110-153行)
template <bool abort_on_nan>
bool common_debug_cb_eval(struct ggml_tensor* t, bool ask, void* user_data)
{
    auto* cb_data = (base_callback_data*)user_data;
    const struct ggml_tensor* src0 = t->src[0];
    const struct ggml_tensor* src1 = t->src[1];

    if (ask) return true;  // 询问阶段：声明支持该节点

    // 正则过滤：名称匹配任一模式即通过
    bool matches_filter = cb_data->tensor_filters.empty();
    if (!matches_filter) {
        for (const auto& filter : cb_data->tensor_filters) {
            if (std::regex_search(t->name, filter)) { matches_filter = true; break; }
        }
    }

    // 打印元信息：op 类型、输入输出张量名及形状
    char src1_str[128] = { 0 };
    if (src1) snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, common_ggml_ne_string(src1).c_str());
    if (matches_filter) {
        LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__, t->name, ggml_type_name(t->type),
            ggml_op_desc(t), src0->name, common_ggml_ne_string(src0).c_str(), src1 ? src1_str : "",
            common_ggml_ne_string(t).c_str());
    }

    // 非主机内存：拷贝到 host 缓冲区
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    // 非量化张量且通过过滤：打印数值
    if (!ggml_is_quantized(t->type) && matches_filter) {
        uint8_t* data = is_host ? (uint8_t*)t->data : cb_data->data.data();
        common_debug_print_tensor<abort_on_nan>(data, t->type, t->ne, t->nb, 3);
    }
    return true;
}
```

回调遵循 `ggml_backend_sched_eval_callback` 协议：
- `ask=true`：询问是否处理该节点，直接返回 `true` 表示全处理。
- `ask=false`：实际执行阶段。先按正则过滤张量名，通过则打印元信息（op、输入输出形状）。
- 若张量在设备内存（GPU/NPU），通过 `ggml_backend_tensor_get` 拷贝到 `cb_data->data` 缓冲区（自动扩容）。
- 最后调用 `common_debug_print_tensor` 打印数值，`abort_on_nan` 控制是否因 NaN 退出。

### 55.6.4 回调数据结构与注册

```cpp
// common/debug.h (第27-41行)
struct base_callback_data {
    std::vector<std::regex> tensor_filters;  // 编译后的正则过滤器
    std::vector<uint8_t> data;               // 设备内存拷贝缓冲区

    base_callback_data() = default;

    base_callback_data(common_params& params, const std::vector<std::string>& filter_patterns) {
        for (const auto& pattern : filter_patterns) {
            try {
                std::string anchored_pattern = "^" + pattern;  // 锚定开头，精确匹配
                tensor_filters.emplace_back(anchored_pattern, std::regex::optimize);
            } catch (const std::regex_error& e) {
                throw std::runtime_error("Invalid regex pattern '" + pattern + "': " + e.what());
            }
        }
        params.cb_eval = common_debug_cb_eval<false>;   // 注册回调函数
        params.cb_eval_user_data = this;                // 传递 this 指针
    }
};
```

带参构造将字符串模式编译为优化后的 `std::regex`（锚定 `^` 避免子串误匹配），并将回调函数指针与 `user_data` 注册到 `common_params`，实现“构造即生效”的即插即用。

```cpp
// common/debug.cpp (第156-159行)
template bool common_debug_cb_eval<false>(ggml_tensor*, bool, void*);
template bool common_debug_cb_eval<true>(ggml_tensor*, bool, void*);
template void common_debug_print_tensor<false>(uint8_t*, ggml_type, const int64_t*, const size_t*, int64_t);
template void common_debug_print_tensor<true>(uint8_t*, ggml_type, const int64_t*, const size_t*, int64_t);
```

显式实例化两个布尔模板参数的版本，避免头文件模板膨胀，控制编译产物大小。

---

## 55.7 设计中的取舍

**为什么 Base64 解码要用 `last` 和 `bits` 两个状态变量？**
因为 Base64 是 6 位一组，字节是 8 位。解码时需要跨边界拼装：上一组剩余的 2~6 位与下一组的高位合成一个字节。`last` 保存上一组的 6 位值，`bits` 记录当前缓冲了多少有效位。这比“先拼成大整数再拆字节”更流式、更省内存。

**`decode_inplace` 为什么安全？**
Base64 编码膨胀系数是 4/3 ≈ 1.33，解码压缩系数是 3/4 = 0.75。输出数据比输入短，写指针永远不会追上读指针，原地覆盖无冲突。

**为什么 `common_utf8_is_complete` 要从后往前扫描最多 4 字节？**
UTF-8 最大 4 字节。从末尾找第一个非续字节（即首字节），根据其高位知预期长度，对比已扫描字节数即可判断是否完整。正向扫描需维护状态机，反向单次遍历更简单。

**反向部分匹配正则为什么要 `^(?:(?:...` 这种嵌套非捕获组？**
目的是在反转字符串上从开头（原字符串末尾）开始匹配，且允许每一步“可选”匹配下一个字符。`(?:x)?` 表示 x 可选，嵌套形成 `a`、`ab`、`abc`、`abcd` 的前缀匹配链。外层 `^` 锚定开头，保证只匹配后缀。

**调试回调为什么分 `ask=true` 和 `ask=false` 两阶段？**
GGML 调度器在图执行前先用 `ask=true` 询问回调是否支持该节点（可用于动态开关），`ask=false` 时才真正执行副作用（打印、拷贝数据）。这种设计允许回调根据运行时上下文决定是否介入，且不阻塞调度器主流程。

---

## 55.8 动手练习

### 55.8.1 练习 1：Base64 边界条件测试
阅读 `common/base64.hpp` 的 `encode()` 和 `decode()` 实现，分析以下边界情况：
1. 输入长度为 0、1、2、3、4、5 时的编码结果差异
2. 使用 `alphabet::url_filename_safe` 编码包含 `+/` 的数据，再用 `alphabet::standard` 解码会发生什么
3. `decoding_behavior::strict` 与 `loose` 在处理尾部 `=` 填充后跟随垃圾字符时的行为差异
4. 对比 `decode()` 与 `decode_inplace()` 的内存使用差异，理解就地操作的优势
回答问题：
- 为什么 `decode` 需要 `last` 和 `bits` 两个状态变量？
- 如何实现一个 `encode_inplace` 函数（提示：需要原地扩展空间）？

### 55.8.2 练习 2：UTF-8 流式解析器实现
基于 `common/unicode.cpp` 中的 `common_parse_utf8_codepoint`：
1. 构造测试用例：0x41（A）、0xC3 0xA9（é）、0xE4 0xB8 0xAD（中）、0xF0 0x9F 0x98 0x80（😀）
2. 故意截断上述多字节序列的最后 1-2 字节，观察解析器返回的 INCOMPLETE 状态
3. 注入非法后续字节（如 0xC0 0x41），观察 INVALID 处理
回答问题：
- `common_utf8_is_complete` 为什么要从后往前扫描最多 4 字节？
- 如果要在网络流中实现「边收边解」，如何利用这些函数的返回值设计状态机？

### 55.8.3 练习 3：正则部分匹配转换验证
阅读 `common/regex-partial.cpp` 的 `regex_to_reversed_partial_regex`：
1. 手动推导模式 `abc`、`a.c`、`[abc]{2,3}`、`(ab|cd)+` 的反向部分匹配形式
2. 使用验证函数测试（构造包含前缀的字符串如 `a`、`ab`、`abc`、`abcd`）
3. 观察 `x{n,m}` 展开为必选+可选的展开逻辑
4. 分析 `common_string_range` 如何记录匹配位置的边界
回答问题：
- 为什么最终反向正则需要 `^(?:(?:...` 这样的嵌套非捕获组结构？
- 如果原始模式包含 `^` 或 `$` 锚点，转换算法需要如何调整？

### 55.8.4 练习 4：张量调试工具实践
基于 `common/debug.cpp` 的调试工具：
1. 创建一个包含 NaN 值的 F32 张量，使用 `common_debug_print_tensor<true>` 观察程序终止行为
2. 构造一个 100x100 的 F16 张量，观察 `n=3` 时的截断打印格式
3. 在 `common_debug_cb_eval` 中设置正则过滤器，只打印名称包含 "attn" 的张量
4. 在 CUDA 后端上运行，观察非主机内存张量的数据拷贝过程（`ggml_backend_tensor_get`）
5. 使用 `common_ggml_ne_string` 打印不同形状张量的维度信息
回答问题：
- `common_ggml_get_float_value` 如何处理 BF16 与 F16 的位级差异？
- 为什么调试回调需要分 `ask=true` 和 `ask=false` 两个阶段？

---

## 55.9 本章小结

这一章中我们学习了 llama.cpp common 模块中的四大核心工具集。首先，Base64 编解码器实现了双字母表设计与自动检测，通过位操作实现高效的三字节到四字符转换，并提供就地解码与三级错误处理策略。其次，Unicode 处理工具基于查表法实现 O(1) 序列长度判定，流式感知解析器区分 SUCCESS/INCOMPLETE/INVALID 三态，完整性检查通过反向扫描解决截断问题，编码函数严格遵循 UTF-8 位模板。接着，正则部分匹配通过将正向正则转换为作用于反转字符串的嵌套可选组正则，巧妙实现了流式前缀匹配检测，搜索接口采用双重模式区分完整与部分匹配。最后，张量调试工具提供类型安全的元素访问、智能截断打印、图执行回调钩子与跨设备内存拷贝，配合正则过滤器实现选择性调试，显式模板实例化控制代码膨胀。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| `base64::encode/decode` | 双字母表Base64编解码，支持自动检测与严格/宽松错误处理 |
| `base64::decode_inplace` | 就地解码技术，直接在原字符串上操作以节省内存分配 |
| `_base64_value` | 6-bit值映射核心，自动识别标准/URL安全字母表 |
| `common_utf8_sequence_length` | O(1)查表法，通过高4位确定UTF-8序列长度 |
| `common_parse_utf8_codepoint` | 流式感知解析，区分成功/不完整/非法三种状态 |
| `common_utf8_is_complete` | 反向扫描验证末尾UTF-8序列完整性，防止截断乱码 |
| `common_unicode_cpt_to_utf8` | 码点到位模式的逆向编码，支持1-4字节UTF-8输出 |
| `regex_to_reversed_partial_regex` | 将正向正则转换为反向部分匹配，支持流式前缀检测 |
| `common_regex::search` | 双重模式搜索，返回FULL或PARTIAL匹配状态 |
| `common_string_range` | 轻量级字符串区间表示，支持边界校验与比较操作 |
| `common_ggml_ne_string` | 张量维度信息格式化为可读字符串 |
| `common_ggml_get_float_value` | 类型安全的统一元素访问，处理F16/F32/BF16/Int等多种类型 |
| `common_debug_print_tensor` | 智能截断打印，大维度张量只显示首尾部分 |
| `common_debug_cb_eval` | GGML图执行回调钩子，支持跨设备张量数据获取与NaN检测 |
| `base_callback_data` | 调试回调的过滤与数据载体，注册回调到参数对象并管理正则过滤器 |

下一章中，我们将学习 llama.cpp 的投机解码模块，理解如何通过草稿模型加速大模型推理的核心机制与实现细节。
