

# 第9章 结构化输出与语法约束 —— 让 LLM「听话」的工具集

## 学习目标

- 理解 SchemaConverter 如何将 JSON Schema 递归转换为 GBNF 文法
- 掌握正则表达式到 GBNF 的翻译机制（_visit_pattern 状态机）
- 熟悉 Pydantic 模型到 GBNF 的类型映射规则（map_pydantic_type_to_gbnf）
- 了解动态模型工厂如何从 Python 函数/字典生成 Pydantic 类
- 掌握结构化输出示例中 schema 注入与 JSON 校验的完整链路
- 理解嵌入服务并发测试如何验证 /embedding 端点的稳定性

## 9.1 生活类比

想象 GBNF 是一套语法警察的执法手册，而 JSON Schema 是嫌疑人的画像：SchemaConverter = 画像分析师，把模糊的画像（JSON Schema）翻译成清晰的执法手册条目（GBNF 规则）；_visit_pattern = 指纹比对师，把 ECMA-262 指纹（正则）转译为执法手册中的特征描述（GBNF 字符类与量词）；_build_object_rule = 档案整理员，处理必填项+可选项+额外项的复杂排列组合；PRIMITIVE_RULES = 执法手册的基础法条（true/false/null/string/number/object/array）；SPACE_RULE = 防止嫌疑人无限申诉的字数限制器，约束空格最多 1-2 个换行+20 个制表符；Pydantic 转换器 = 模型锻造厂，把 Python 装饰器（BaseModel）铸造成 GBNF 形状的标准化产品；动态模型工厂 = 变形金刚，把普通 Python 函数和 OpenAI 字典变形为统一形态的 Pydantic 类；文档生成器 = 翻译官，把 GBNF 技术语法翻译成 LLM 看得懂的 Markdown 说明书；server_embd = 质检员，反复按门铃（并发请求）测试大门（/embedding）是否每次都给出相同的回应（相似度≈1）。

## 9.2 源码地图

```
examples/json_schema_to_grammar.py
├── 工具函数
│   ├── _build_repetition()      # 生成{min,max}重复规则
│   ├── _generate_min_max_int()  # 按位生成整数区间规则
│   └── BuiltinRule              # 内置规则容器
├── 内置规则常量
│   ├── SPACE_RULE               # 空格约束规则
│   ├── PRIMITIVE_RULES          # 6种JSON基础类型
│   ├── STRING_FORMAT_RULES      # uuid/date/time等
│   ├── DOTALL / DOT             # 正则.匹配字符集
│   └── RESERVED_NAMES           # 保留名称集合
├── SchemaConverter              # 核心转换器类
│   ├── __init__()               # 初始化规则表与参数
│   ├── _format_literal()        # 字面量转义
│   ├── not_literal()            # 否定字面量生成
│   ├── _not_strings()           # Trie树驱动的非匹配串生成
│   ├── _add_rule()              # 规则去重与命名
│   ├── resolve_refs()           # $ref解析与远程加载
│   ├── _generate_union_rule()   # oneOf/anyOf联合类型
│   ├── _visit_pattern()         # 正则到GBNF翻译（核心）
│   ├── _resolve_ref()           # 引用规则解析
│   ├── _generate_constant_rule()# const/enum字面量
│   ├── _add_primitive()         # 注入基础规则
│   ├── visit()                  # 主入口分发
│   ├── _build_object_rule()     # 对象规则构建
│   └── format_grammar()         # 输出最终文法
└── main()                       # 命令行入口

examples/regex_to_grammar.py
└── __main__                     # 子进程调用json_schema_to_grammar.py

examples/pydantic_models_to_grammar.py
├── 枚举与命名
│   ├── PydanticDataType         # 支持的数据类型枚举
│   ├── map_pydantic_type_to_gbnf() # Python类型→GBNF名映射
│   └── format_model_and_field_name() # 驼峰转kebab-case
├── 数字与字符串规则生成
│   ├── generate_list_rule()     # 列表规则模板
│   ├── get_members_structure()  # 普通类结构提取
│   ├── regex_to_gbnf()          # 简单正则翻译
│   ├── generate_gbnf_integer_rules() # 整数位数约束
│   └── generate_gbnf_float_rules()   # 浮点精度约束
├── 字段规则生成
│   └── generate_gbnf_rule_for_type() # 递归生成字段规则
├── 模型规则生成
│   ├── generate_gbnf_grammar()  # 单模型规则生成
│   └── generate_gbnf_grammar_from_pydantic_models() # 多模型顶层规则
├── 依赖规则补全
│   └── get_primitive_grammar()  # 追加基础依赖规则
├── 文档生成
│   ├── generate_markdown_documentation() # Markdown文档
│   ├── generate_field_markdown()        # 字段详情
│   ├── generate_text_documentation()    # 纯文本文档
│   ├── generate_field_text()            # 纯文本字段详情
│   └── format_multiline_description()   # 多行描述缩进
├── 保存与一键生成
│   ├── save_gbnf_grammar_and_documentation()
│   ├── generate_and_save_gbnf_grammar_and_documentation()
│   ├── generate_gbnf_grammar_and_documentation()
│   ├── generate_gbnf_grammar_and_documentation_from_dictionaries()
│   └── remove_empty_lines()              # 去除空行
├── 辅助工具
│   ├── map_grammar_names_to_pydantic_model_class() # GBNF名到模型映射
│   ├── json_schema_to_python_types()     # JSON类型→Python类型
│   ├── list_to_enum()                    # 列表生成Enum类
│   └── format_json_example()             # JSON示例格式化
└── 动态模型工厂
    ├── create_dynamic_model_from_function() # 函数→Pydantic
    ├── convert_dictionary_to_pydantic_model() # 字典→Pydantic
    ├── create_dynamic_models_from_dictionaries()
    ├── add_run_method_to_dynamic_model() # 模型注入run方法
    └── __main__                     # 模块顶层执行入口（示例调用）

examples/json_schema_pydantic_example.py
├── create_completion()           # 通用HTTP客户端 + schema注入
├── Instructor备选分支           # 注释中的替代实现
├── QAPair模型                   # 数值范围约束示例
├── PyramidalSummary模型         # 嵌套递归 + 集合长度约束
└── __main__                     # 调用入口

examples/pydantic_models_to_grammar_examples.py
├── create_completion()          # /completion端点客户端
├── SendMessageToUser            # RCE演示模型
│   └── run()                    # 消息打印方法
├── Calculator                   # 计算器模型（Enum+Union）
│   └── run()                    # 数学运算方法
├── Book                         # 结构化输出模型
├── Category                     # 书籍分类分类
├── get_current_datetime         # 函数式工具定义
├── get_current_weather          # OpenAI格式函数定义
├── example_rce()                # 单函数调用示例
├── example_calculator()         # Enum工具调用示例
├── example_struct()             # 非函数调用结构化输出
├── example_concurrent()         # 多工具并发调用示例
├── main()                       # 示例调度
└── __main__                     # 入口

examples/server_embd.py
├── requests_post_async()        # requests异步包装
├── main()                       # 并发请求 /embedding
└── __main__                     # 余弦相似度计算
```

## 9.3 JSON Schema 转换器的"中央调度台" —— SchemaConverter 类

`SchemaConverter` 是整个 JSON Schema 到 GBNF 转换流程的中央调度台。它通过递归遍历 Schema 字典，构建出能够约束 LLM 输出的 GBNF 文法规则。其核心设计在于将内置规则注册表 `_rules` 与原始规则集合分离，避免重名冲突。

### 9.3.1 SchemaConverter.__init__()：初始化转换器上下文

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter.__init__()`（第166-173行）

```python
def __init__(self, *, prop_order, allow_fetch, dotall, raw_pattern):
    self._prop_order = prop_order          # 对象属性的优先级排序依据
    self._allow_fetch = allow_fetch        # 是否允许远程加载引用的 Schema
    self._dotall = dotall                  # 正则中 . 是否匹配换行符
    self._raw_pattern = raw_pattern        # 是否将字符串模式视为无引号原始正则
    self._rules = {                        # 内部规则表，预置 space 规则
        'space': SPACE_RULE,
    }
    self._refs = {}                        # 解析后的 $ref 缓存
    self._refs_being_resolved = set()      # 循环引用检测集合
```

这段代码定义了转换器的构造函数，接收四个配置参数并初始化内部状态。`_rules` 字典存储最终生成的 GBNF 规则，预置了 `space` 规则用于处理空白字符。`_refs` 用于缓存解析后的引用 Schema，`_refs_being_resolved` 用于检测循环引用。

### 9.3.2 PRIMITIVE_RULES 与 STRING_FORMAT_RULES：基础法条与特殊格式

源码路径：`examples/json_schema_to_grammar.py` - `PRIMITIVE_RULES`（第138-148行）、`STRING_FORMAT_RULES`（第151-156行）

```python
# 6种 JSON 基础类型的预定义 GBNF 规则
PRIMITIVE_RULES = {
    'boolean': BuiltinRule('"true" | "false" space', ['space']),
    'number': BuiltinRule('("-"? ([0] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space', ['space']),
    'string': BuiltinRule('"\"" ( [^"\\x7F\\x00-\\x1F] | "\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" space', ['space', 'char']),
    'null': BuiltinRule('"null" space', ['space']),
    'object': BuiltinRule('"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space', ['space', 'string', 'value']),
    'array': BuiltinRule('"[" space ( value ("," space value)* )? "]" space', ['space', 'value']),
    'char': BuiltinRule('[^"\\x7F\\x00-\\x1F] | "\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])', []),
    'value': BuiltinRule('object | array | string | number | boolean | null', []),
}

# uuid/date/time/date-time 等特殊字符串格式规则
STRING_FORMAT_RULES = {
    'uuid-string': BuiltinRule(r'"\"" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] "-" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] "-" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] "-" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] "-" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] "\"" space', ['space']),
    # date-string, time-string, date-time-string 类似...
}
```

这段代码定义了 JSON 六种基本类型及特殊字符串格式的预定义 GBNF 规则。`BuiltinRule` 封装了规则内容字符串与依赖列表，`PRIMITIVE_RULES` 覆盖了 boolean/number/string/null/object/array 及辅助的 char/value 规则；`STRING_FORMAT_RULES` 则为 uuid、date、time、date-time 提供了具体的字符级匹配模式。

### 9.3.3 _add_rule() 与 _add_primitive()：规则注册与去重机制

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter._add_rule()`（第230-237行）、`SchemaConverter._add_primitive()`（第470-475行）

```python
def _add_rule(self, name, rule):
    esc_name = INVALID_RULE_CHARS_RE.sub('-', name)  # 非法字符替换为连字符
    if esc_name not in self._rules or self._rules[esc_name] == rule:
        key = esc_name
    else:
        i = 0
        while f'{esc_name}{i}' in self._rules and self._rules[f'{esc_name}{i}'] != rule:
            i += 1
        key = f'{esc_name}{i}'
    self._rules[key] = rule
    return key

def _add_primitive(self, name: str, rule: BuiltinRule):
    n = self._add_rule(name, rule.content)
    for dep in rule.deps:                              # 递归注入依赖规则
        dep_rule = PRIMITIVE_RULES.get(dep) or STRING_FORMAT_RULES.get(dep)
        assert dep_rule, f'Rule {dep} not known'
        if dep not in self._rules:
            self._add_primitive(dep, dep_rule)
    return n
```

`_add_rule` 负责规则的去重与命名：将非法字符替换为连字符，若规则名冲突且内容不同则自动追加数字后缀。`_add_primitive` 则在注册基础规则时自动递归注入其依赖规则，确保引用的 `space`、`char` 等基础规则都已加载。

### 9.3.4 format_grammar()：最终文法输出

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter.format_grammar()`（第522-525行）

```python
def format_grammar(self):
    return '\n'.join(
        f'{name} ::= {rule}'
        for name, rule in sorted(self._rules.items(), key=lambda kv: kv[0])
    )
```

转换完成后，`format_grammar` 按规则名排序输出最终的 GBNF 文法文本，保证生成结果的确定性与可读性。

## 9.4 基础类型与字符串处理 —— 数字、字符串与空格规则

转换器内置了处理数字范围、字符串字符集与空格约束的核心工具函数。

### 9.4.1 _generate_min_max_int()：按位生成整数区间规则

源码路径：`examples/json_schema_to_grammar.py` - `_generate_min_max_int()`（第29-119行）

```python
def _generate_min_max_int(min_value: Optional[int], max_value: Optional[int], out: list, decimals_left: int = 16, top_level: bool = True):
    # 三个嵌套辅助函数
    def digit_range(from_char: str, to_char: str): ...      # 生成 [from-to] 字符集
    def more_digits(min_digits: int, max_digits: int): ...  # 生成 [0-9]{min,max} 量词
    def uniform_range(from_str: str, to_str: str): ...      # 逐位对齐生成区间规则

    # 核心逻辑：处理负数、分段、位数差异
    if min_value is not None and max_value is not None:
        if min_value < 0 and max_value < 0:
            out.append("\"-\" (")
            _generate_min_max_int(-max_value, -min_value, out, decimals_left, top_level=True)
            out.append(")")
            return
        if min_value < 0:
            out.append("\"-\" (")
            _generate_min_max_int(0, -min_value, out, decimals_left, top_level=True)
            out.append(") | ")
            min_value = 0
        # 正数区间：按位数分段生成 uniform_range
        ...
```

这个函数通过三个嵌套函数协作，将整数区间转换为 GBNF 字符类与量词的组合。它支持负数（通过取反递归）、变长位数（分段生成）、以及逐位对齐的 `uniform_range` 逻辑，确保生成的规则精确覆盖 `[min, max]` 区间。

### 9.4.2 _build_repetition()：生成 {min,max} 重复规则

源码路径：`examples/json_schema_to_grammar.py` - `_build_repetition()`（第15-27行）

```python
def _build_repetition(item_rule, min_items, max_items, separator_rule=None):
    if max_items == 0:
        return ""
    if min_items == 0 and max_items == 1:
        return f'{item_rule}?'
    if not separator_rule:
        if min_items == 1 and max_items is None:
            return f'{item_rule}+'
        elif min_items == 0 and max_items is None:
            return f'{item_rule}*'
        else:
            return f'{item_rule}{{{min_items},{max_items if max_items is not None else ""}}}'
    # 含分隔符时构造 (item sep item)* 形式
    result = item_rule + ' ' + _build_repetition(f'({separator_rule} {item_rule})', min_items - 1 if min_items > 0 else 0, max_items - 1 if max_items is not None else None)
    return f'({result})?' if min_items == 0 else result
```

该函数处理 GBNF 中的重复量词生成。无分隔符时直接映射 `?`、`+`、`*` 或 `{m,n}`；有分隔符时递归构造 `(item sep item)*` 结构，正确处理最小/最大出现次数。

### 9.4.3 SPACE_RULE 与 char 规则：空白约束与字符集

源码路径：`examples/json_schema_to_grammar.py` - 常量定义（第130-137行、第138-148行）

```python
SPACE_RULE = BuiltinRule(r'( [ \t\n] space)?', ['space'])  # 允许 0-1 个空白字符后递归

# PRIMITIVE_RULES 中的 char 规则
'char': BuiltinRule('[^"\\x7F\\x00-\\x1F] | "\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])', []),
```

`SPACE_RULE` 采用右递归 `( [ \t\n] space)?` 形式，限制连续空白字符；`char` 规则定义了 JSON 字符串中合法的字符：可打印 ASCII、转义序列（`\b`、`\f`、`\n`、`\r`、`\t`、`"`、`\`）及 Unicode 转义 `\uXXXX`。

## 9.5 Schema 引用解析器 —— 递归解决 '$ref' 与远程加载

JSON Schema 通过 `$ref` 复用定义，转换器需在遍历前将所有引用展开。

### 9.5.1 resolve_refs()：深度优先解析与远程获取

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter.resolve_refs()`（第239-274行）

```python
def resolve_refs(self, schema: dict, url: str):
    def visit(n: dict):
        if isinstance(n, list):
            return [visit(x) for x in n]
        elif isinstance(n, dict):
            ref = n.get('$ref')
            if ref is not None and ref not in self._refs:
                if ref.startswith('https://'):
                    assert self._allow_fetch, 'Fetching remote schemas is not allowed (use --allow-fetch for force)'
                    import requests
                    frag_split = ref.split('#')
                    base_url = frag_split[0]
                    target = self._refs.get(base_url)
                    if target is None:
                        target = self.resolve_refs(requests.get(ref).json(), base_url)
                        self._refs[base_url] = target
                    if len(frag_split) == 1 or frag_split[-1] == '':
                        return target
                elif ref.startswith('#/'):
                    target = schema
                    ref = f'{url}{ref}'
                    n['$ref'] = ref
                else:
                    raise ValueError(f'Unsupported ref {ref}')
                # 逐层解析 fragment 路径
                for sel in ref.split('#')[-1].split('/')[1:]:
                    ...
                self._refs[ref] = target
            else:
                for v in n.values():
                    visit(v)
        return n
    return visit(schema)
```

`resolve_refs` 采用深度优先遍历 Schema 树。遇到 `$ref` 时：HTTPS 引用需开启 `allow_fetch` 才允许远程获取，获取后递归解析远程 Schema；`#/...` 内部引用则转为绝对 URL 并按 `/` 分段逐层定位目标子 Schema。解析结果缓存至 `self._refs`，替换原 Schema 中的 `$ref` 为绝对引用，便于后续 `visit` 阶段直接展开。

### 9.5.2 _resolve_ref()：单引用的规则名映射与循环检测

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter._resolve_ref()`（第381-389行）

```python
def _resolve_ref(self, ref):
    ref_fragment = ref.split('#')[-1]
    ref_name = 'ref' + re.sub(r'[^a-zA-Z0-9-]+', '-', ref_fragment)
    if ref_name not in self._rules and ref not in self._refs_being_resolved:
        self._refs_being_resolved.add(ref)
        resolved = self._refs[ref]
        ref_name = self.visit(resolved, ref_name)
        self._refs_being_resolved.remove(ref)
    return ref_name
```

在 `visit` 阶段遇到 `$ref` 时调用。将 fragment 转为合法规则名（非法字符替换为 `-`），通过 `_refs_being_resolved` 集合检测循环引用，递归调用 `visit` 生成被引用 Schema 的规则，最后返回规则名供上层引用。

## 9.6 对象规则构建器 —— 必选/可选属性的"排列组合"

对象规则是 GBNF 生成中最复杂的部分，需处理必选/可选/任意键的所有合法组合。

### 9.6.1 _build_object_rule()：核心构建流程

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter._build_object_rule()`（第477-520行）

```python
def _build_object_rule(self, properties: List[Tuple[str, Any]], required: Set[str], name: str, additional_properties: Optional[Union[bool, Any]]):
    prop_order = self._prop_order
    # 1. 按 prop_order 排序，未列出的保持原顺序
    sorted_props = [kv[0] for _, kv in sorted(enumerate(properties), key=lambda ikv: (prop_order.get(ikv[1][0], len(prop_order)), ikv[0]))]

    # 2. 为每个属性生成 "key" : value 的 KV 规则
    prop_kv_rule_names = {}
    for prop_name, prop_schema in properties:
        prop_rule_name = self.visit(prop_schema, f'{name}{"-" if name else ""}{prop_name}')
        prop_kv_rule_names[prop_name] = self._add_rule(
            f'{name}{"-" if name else ""}{prop_name}-kv',
            fr'{self._format_literal(json.dumps(prop_name))} space ":" space {prop_rule_name}'
        )
    required_props = [k for k in sorted_props if k in required]
    optional_props = [k for k in sorted_props if k not in required]

    # 3. 处理 additionalProperties
    if additional_properties is not None and additional_properties != False:
        sub_name = f'{name}{"-" if name else ""}additional'
        value_rule = self.visit(additional_properties, f'{sub_name}-value') if isinstance(additional_properties, dict) else \
            self._add_primitive('value', PRIMITIVE_RULES['value'])
        key_rule = self._add_primitive('string', PRIMITIVE_RULES['string']) if not sorted_props \
            else self._add_rule(f'{sub_name}-k', self._not_strings(sorted_props))
        prop_kv_rule_names["*"] = self._add_rule(
            f'{sub_name}-kv',
            f'{key_rule} ":" space {value_rule}'
        )
        optional_props.append("*")

    # 4. 组装最终规则：必选串联 + 可选排列组合
    rule = '"{" space '
    rule += ' "," space '.join(prop_kv_rule_names[k] for k in required_props)

    if optional_props:
        rule += ' ('
        if required_props:
            rule += ' "," space ( '

        def get_recursive_refs(ks, first_is_optional):
            [k, *rest] = ks
            kv_rule_name = prop_kv_rule_names[k]
            comma_ref = f'( "," space {kv_rule_name} )'
            if first_is_optional:
                res = comma_ref + ('*' if k == '*' else '?')
            else:
                res = kv_rule_name + (' ' + comma_ref + "*" if k == '*' else '')
            if len(rest) > 0:
                res += ' ' + self._add_rule(
                    f'{name}{"-" if name else ""}{k}-rest',
                    get_recursive_refs(rest, first_is_optional=True)
                )
            return res

        rule += ' | '.join(
            get_recursive_refs(optional_props[i:], first_is_optional=False)
            for i in range(len(optional_props))
        )
        if required_props:
            rule += ' )'
        rule += ' )?'

    rule += ' "}" space'
    return rule
```

构建分四步：① 按 `prop_order` 排序属性；② 为每属性生成 `key: value` 的 KV 子规则；③ 处理 `additionalProperties`：若为字典则递归生成 value 规则，key 规则用 `_not_strings` 排除已知属性名；④ 组装：必选属性用 `,` 串联，可选属性通过 `get_recursive_refs` 递归生成所有合法子集组合（含 `*` 任意键），外层用 `( ... )?` 标记整体可选。

### 9.6.2 _not_strings() 与 not_literal()：Trie 树批量否定

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter._not_strings()`（第202-228行）、`SchemaConverter.not_literal()`（第180-200行）

```python
def not_literal(self, literal: str, dotall: bool = True, maybe_escaped_underscores = False) -> str:
    # 生成单字面量的否定规则：not_literal('abc') -> '([^a] | "a" ([^b] | "b" ([^c])?)?)?'
    def recurse(i: int):
        c = literal[i]
        yield f'[^{c}]'
        if i < len(literal) - 1:
            yield ' | '
            yield self._format_literal(c)
            yield ' ('
            yield from recurse(i + 1)
            yield ')?'
    return ''.join(('(', *recurse(0), ')'))

def _not_strings(self, strings):
    # 构建 Trie 树批量生成多字符串否定规则
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_string = False
        def insert(self, string):
            node = self
            for c in string:
                node = node.children.setdefault(c, TrieNode())
            node.is_end_of_string = True
    trie = TrieNode()
    for s in strings:
        trie.insert(s)
    char_rule = self._add_primitive('char', PRIMITIVE_RULES['char'])
    out = ['["] ( ']
    def visit(node):
        rejects = []
        first = True
        for c in sorted(node.children.keys()):
            child = node.children[c]
            rejects.append(c)
            if first: first = False
            else: out.append(' | ')
            out.append(f'[{c}]')
            if child.children:
                out.append(f' (')
                visit(child)
                out.append(')')
            elif child.is_end_of_string:
                out.append(f' {char_rule}+')
        if node.children:
            if not first: out.append(' | ')
            out.append(f'[^"{"".join(rejects)}"] {char_rule}*')
    visit(trie)
    out.append(f' ){"" if trie.is_end_of_string else "?"} ["] space')
    return ''.join(out)
```

`not_literal` 采用字符级递归生成否定规则；`_not_strings` 构建 Trie 树，对已知属性名批量生成高效的否定规则，用于 `additionalProperties` 的 key 约束——确保额外键不与已定义属性重名。

## 9.7 正则模式到 GBNF 的翻译器 —— _visit_pattern 深度解析

JSON Schema 的 `string.pattern` 使用 ECMA-262 正则，需翻译为 GBNF 文法。

### 9.7.1 _visit_pattern()：状态机核心逻辑

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter._visit_pattern()`（第281-379行）

```python
def _visit_pattern(self, pattern, name):
    '''
        Transforms a regular expression pattern into a GBNF rule.

        Input: https://json-schema.org/understanding-json-schema/reference/regular_expressions
        Output: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md

        Unsupported features: negative/positive lookaheads, greedy/non-greedy modifiers.

        Mostly a 1:1 translation, except for {x} / {x,} / {x,y} quantifiers for which
        we define sub-rules to keep the output lean.
    '''

    assert pattern.startswith('^') and pattern.endswith('$'), 'Pattern must start with "^" and end with "$"'
    pattern = pattern[1:-1]
    sub_rule_ids = {}

    i = 0
    length = len(pattern)

    def to_rule(s: tuple[str, bool]) -> str:
        (txt, is_literal) = s
        return "\"" + txt + "\"" if is_literal else txt

    def transform() -> tuple[str, bool]:
        '''
            Parse a unit at index i (advancing it), and return its string representation + whether it's a literal.
        '''
        nonlocal i
        nonlocal pattern
        nonlocal sub_rule_ids

        start = i
        # For each component of this sequence, store its string representation and whether it's a literal.
        # We only need a flat structure here to apply repetition operators to the last item, and
        # to merge literals at the and (we're parsing grouped ( sequences ) recursively and don't treat '|' specially
        # (GBNF's syntax is luckily very close to regular expressions!)
        seq: list[tuple[str, bool]] = []

        def get_dot():
            if self._dotall:
                rule = DOTALL
            else:
                # Accept any character... except \n and \r line break chars (\x0A and \xOD)
                rule = DOT
            return self._add_rule(f'dot', rule)

        def join_seq():
            nonlocal seq
            ret = []
            for is_literal, g in itertools.groupby(seq, lambda x: x[1]):
                if is_literal:
                    ret.append((''.join(x[0] for x in g), True))
                else:
                    ret.extend(g)
            if len(ret) == 1:
                return ret[0]
            return (' '.join(to_rule(x) for x in seq), False)

        while i < length:
            c = pattern[i]
            if c == '.':
                seq.append((get_dot(), False))
                i += 1
            elif c == '(':
                i += 1
                if i < length:
                    assert pattern[i] != '?', f'Unsupported pattern syntax "{pattern[i]}" at index {i} of /{pattern}/'
                seq.append((f'({to_rule(transform())})', False))
            elif c == ')':
                i += 1
                assert start > 0 and pattern[start-1] == '(', f'Unbalanced parentheses; start = {start}, i = {i}, pattern = {pattern}'
                return join_seq()
            elif c == '[':
                square_brackets = c
                i += 1
                while i < length and pattern[i] != ']':
                    if pattern[i] == '\\':
                        square_brackets += pattern[i:i+2]
                        i += 2
                    else:
                        square_brackets += pattern[i]
                        i += 1
                assert i < length, f'Unbalanced square brackets; start = {start}, i = {i}, pattern = {pattern}'
                square_brackets += ']'
                i += 1
                seq.append((square_brackets, False))
            elif c == '|':
                seq.append(('|', False))
                i += 1
            elif c in ('*', '+', '?'):
                seq[-1] = (to_rule(seq[-1]) + c, False)
                i += 1
            elif c == '{':
                curly_brackets = c
                i += 1
                while i < length and pattern[i] != '}':
                    curly_brackets += pattern[i]
                    i += 1
                assert i < length, f'Unbalanced curly brackets; start = {start}, i = {i}, pattern = {pattern}'
                curly_brackets += '}'
                i += 1
                nums = [s.strip() for s in curly_brackets[1:-1].split(',')]
                min_times = 0
                max_times = None
                try:
                    if len(nums) == 1:
                        min_times = int(nums[0])
                        max_times = min_times
                    else:
                        assert len(nums) == 2
                        min_times = int(nums[0]) if nums[0] else 0
                        max_times = int(nums[1]) if nums[1] else None
                except ValueError:
                    raise ValueError(f'Invalid quantifier {curly_brackets} in /{pattern}/')

                (sub, sub_is_literal) = seq[-1]

                if not sub_is_literal:
                    id = sub_rule_ids.get(sub)
                    if id is None:
                        id = self._add_rule(f'{name}-{len(sub_rule_ids) + 1}', sub)
                        sub_rule_ids[sub] = id
                    sub = id

                seq[-1] = (_build_repetition(f'"{sub}"' if sub_is_literal else sub, min_times, max_times), False)
            else:
                literal = ''
                while i < length:
                    if pattern[i] == '\\' and i < length - 1:
                        next = pattern[i + 1]
                        if next in ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS:
                            i += 1
                            literal += pattern[i]
                            i += 1
                        else:
                            literal += pattern[i:i+2]
                            i += 2
                    elif pattern[i] == '"' and not self._raw_pattern:
                        literal += '\\"'
                        i += 1
                    elif pattern[i] not in NON_LITERAL_SET and \
                            (i == length - 1 or literal == '' or pattern[i+1] == '.' or pattern[i+1] not in NON_LITERAL_SET):
                        literal += pattern[i]
                        i += 1
                    else:
                        break
                if literal:
                    seq.append((literal, True))

        return join_seq()

    return self._add_rule(
        name,
        to_rule(transform()) if self._raw_pattern \
            else "\"\\\"\"" + "(" + to_rule(transform()) + ")" + "\"\\\"\" space"
    )
```

`_visit_pattern` 实现了一个递归下降解析器：
- **状态机循环**：`while i < length` 逐字符解析，维护 `seq` 列表存储当前序列的各组件（字面量/非字面量）。
- **核心分支**：`.` → `DOT/DOTALL`；`()` → 递归 `transform()` 处理子组；`[]` → 直接复制字符集（含转义）；`|` → 交替；`*+?` → 直接追加量词；`{m,n}` → 调用 `_build_repetition` 生成子规则并通过 `sub_rule_ids` 缓存复用。
- **字面量合并优化**：`join_seq` 使用 `itertools.groupby` 将连续字面量合并为 `"..."`，减少规则长度。
- **最终包装**：非 `raw_pattern` 模式下，整体用 `"\\"" ... "\\"" space` 包裹，符合 JSON 字符串字面量语法。

### 9.7.2 _format_literal()：字面量转义

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter._format_literal()`（第175-178行）

```python
def _format_literal(self, literal):
    escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
        lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)) or m.group(0), literal
    )
    return f'"{escaped}"'
```

对 `\r`、`\n`、`"`、`\` 等特殊字符做转义，生成符合 GBNF 语法的 `"..."` 字符串字面量。

## 9.8 主入口与统一调度 —— visit 方法与命令行入口

### 9.8.1 visit()：Schema 类型分发总线

源码路径：`examples/json_schema_to_grammar.py` - `SchemaConverter.visit()`（第394-468行）

```python
def visit(self, schema, name):
    schema_type = schema.get('type')
    schema_format = schema.get('format')
    rule_name = name + '-' if name in RESERVED_NAMES else name or 'root'

    if (ref := schema.get('$ref')) is not None:
        return self._add_rule(rule_name, self._resolve_ref(ref))
    elif 'oneOf' in schema or 'anyOf' in schema:
        return self._add_rule(rule_name, self._generate_union_rule(name, schema.get('oneOf') or schema['anyOf']))
    elif isinstance(schema_type, list):
        return self._add_rule(rule_name, self._generate_union_rule(name, [{**schema, 'type': t} for t in schema_type]))
    elif 'const' in schema:
        return self._add_rule(rule_name, self._generate_constant_rule(schema['const']) + ' space')
    elif 'enum' in schema:
        rule = '(' + ' | '.join((self._generate_constant_rule(v) for v in schema['enum'])) + ') space'
        return self._add_rule(rule_name, rule)
    elif schema_type in (None, 'object') and \
         ('properties' in schema or \
          ('additionalProperties' in schema and schema['additionalProperties'] is not True)):
        required = set(schema.get('required', []))
        properties = list(schema.get('properties', {}).items())
        return self._add_rule(rule_name, self._build_object_rule(properties, required, name, schema.get('additionalProperties')))
    elif schema_type in (None, 'object', 'string') and 'allOf' in schema:
        # 合并多组件属性 + 枚举交集
        ...
    elif schema_type in (None, 'array') and ('items' in schema or 'prefixItems' in schema):
        # 元组或统一项数组
        ...
    elif schema_type in (None, 'string') and 'pattern' in schema:
        return self._visit_pattern(schema['pattern'], rule_name)
    elif schema_type in (None, 'string') and re.match(r'^uuid[1-5]?$', schema_format or ''):
        return self._add_primitive('root' if rule_name == 'root' else schema_format, PRIMITIVE_RULES['uuid'])
    elif schema_type in (None, 'string') and f'{schema_format}-string' in STRING_FORMAT_RULES:
        prim_name = f'{schema_format}-string'
        return self._add_rule(rule_name, self._add_primitive(prim_name, STRING_FORMAT_RULES[prim_name]))
    elif schema_type == 'string' and ('minLength' in schema or 'maxLength' in schema):
        char_rule = self._add_primitive('char', PRIMITIVE_RULES['char'])
        min_len = schema.get('minLength', 0)
        max_len = schema.get('maxLength')
        return self._add_rule(rule_name, r'"\"" ' + _build_repetition(char_rule, min_len, max_len) + r' "\"" space')
    elif schema_type in (None, 'integer') and \
            ('minimum' in schema or 'exclusiveMinimum' in schema or 'maximum' in schema or 'exclusiveMaximum' in schema):
        # 调用 _generate_min_max_int
        ...
    elif (schema_type == 'object') or (len(schema) == 0):
        return self._add_rule(rule_name, self._add_primitive('object', PRIMITIVE_RULES['object']))
    elif schema_type is None and isinstance(schema, dict):
        return self._add_rule(rule_name, self._add_primitive('value', PRIMITIVE_RULES['value']))
    else:
        assert schema_type in PRIMITIVE_RULES, f'Unrecognized schema: {schema}'
        return self._add_primitive('root' if rule_name == 'root' else schema_type, PRIMITIVE_RULES[schema_type])
```

`visit` 是转换器的单一入口，按 Schema 关键字分发：
- `$ref` → `_resolve_ref`
- `oneOf/anyOf` → `_generate_union_rule`
- `type` 为数组 → 展开为联合类型
- `const/enum` → `_generate_constant_rule` 字面量匹配
- `properties/additionalProperties` → `_build_object_rule`
- `allOf` → 合并属性、计算枚举交集
- `items/prefixItems` → 数组/元组规则
- `pattern` → `_visit_pattern`
- `format` → 特殊字符串格式规则
- `minLength/maxLength` → 字符重复规则
- `minimum/maximum` → `_generate_min_max_int`
- 空对象/无类型约束 → 退化为 `object`/`value` 基础规则

### 9.8.2 _generate_union_rule 与 _generate_constant_rule

源码路径：`examples/json_schema_to_grammar.py` - `_generate_union_rule()`（第276-279行）、`_generate_constant_rule()`（第391-392行）

```python
def _generate_union_rule(self, name, alt_schemas):
    return ' | '.join((
        self.visit(alt_schema, f'{name}{"-" if name else "alternative-"}{i}')
        for i, alt_schema in enumerate(alt_schemas)
    ))

def _generate_constant_rule(self, value):
    return self._format_literal(json.dumps(value))
```

联合规则为每个分支生成 `name-alternative-i` 子规则并用 `|` 串联；常量规则直接 `json.dumps` 序列化后转义包裹。

### 9.8.3 main()：命令行入口与参数透传

源码路径：`examples/json_schema_to_grammar.py` - `main()`（第527-570行）

```python
def main(args_in = None):
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--prop-order', default=[], type=lambda s: s.split(','), help='属性优先级顺序')
    parser.add_argument('--allow-fetch', action='store_true', default=False, help='允许 HTTPS 远程引用')
    parser.add_argument('--dotall', action='store_true', default=False, help='正则 . 匹配换行')
    parser.add_argument('--raw-pattern', action='store_true', default=False, help='原始正则模式')
    parser.add_argument('schema', help='Schema 文件路径 ("-" 表示 stdin)')
    args = parser.parse_args(args_in)

    # 三种来源加载 Schema
    if args.schema.startswith('https://'):
        url = args.schema; import requests; schema = requests.get(url).json()
    elif args.schema == '-':
        url = 'stdin'; schema = json.load(sys.stdin)
    else:
        url = f'file://{args.schema}'; with open(args.schema) as f: schema = json.load(f)

    converter = SchemaConverter(
        prop_order={name: idx for idx, name in enumerate(args.prop_order)},
        allow_fetch=args.allow_fetch, dotall=args.dotall, raw_pattern=args.raw_pattern)
    schema = converter.resolve_refs(schema, url)
    converter.visit(schema, '')
    print(converter.format_grammar())
```

命令行工具支持从文件、stdin、HTTPS URL 加载 Schema，透传四个核心参数给转换器，完成引用解析与遍历后输出排序后的 GBNF 文法。

## 9.9 正则到 GBNF 的轻量包装 —— regex_to_grammar.py 的"管道工"

### 9.9.1 __main__：子进程复用主转换器

源码路径：`examples/regex_to_grammar.py` - `__main__`（第1-20行）

```python
import json, subprocess, sys, os
assert len(sys.argv) >= 2
[_, pattern, *rest] = sys.argv

print(subprocess.check_output(
    [
        "python",
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "json_schema_to_grammar.py"),
        *rest, "-", "--raw-pattern",
    ],
    text=True,
    input=json.dumps({"type": "string", "pattern": pattern}, indent=2)))
```

该脚本不重复实现正则翻译，而是构造最小 JSON Schema `{"type": "string", "pattern": "<REGEX>"}`，通过子进程调用 `json_schema_to_grammar.py` 并透传额外参数（如 `--allow-fetch`），利用 `--raw-pattern` 让正则直接作为 GBNF 规则使用。`os.path.realpath(__file__)` 保证跨平台正确定位依赖脚本。

## 9.10 Pydantic 到 GBNF 的"炼金术" —— 类型映射与规则生成

`pydantic_models_to_grammar.py` 将 Python 类型注解转换为 GBNF 规则，并同步生成 Markdown/Text 文档。

### 9.10.1 PydanticDataType：数据类型枚举

源码路径：`examples/pydantic_models_to_grammar.py` - `PydanticDataType`（第54-74行）

```python
class PydanticDataType(Enum):
    STRING = "string"
    TRIPLE_QUOTED_STRING = "triple_quoted_string"
    MARKDOWN_CODE_BLOCK = "markdown_code_block"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    OBJECT = "object"
    ARRAY = "array"
    ENUM = "enum"
    ANY = "any"
    NULL = "null"
    CUSTOM_CLASS = "custom-class"
    CUSTOM_DICT = "custom-dict"
    SET = "set"
```

枚举定义了转换器支持的所有目标 GBNF 类型名，包含字符串家族扩展（三引号、Markdown 代码块）及复合类型标识。

### 9.10.2 map_pydantic_type_to_gbnf()：Python 类型到 GBNF 名的递归映射

源码路径：`examples/pydantic_models_to_grammar.py` - `map_pydantic_type_to_gbnf()`（第76-115行）

```python
def map_pydantic_type_to_gbnf(pydantic_type: type[Any]) -> str:
    origin_type = get_origin(pydantic_type)
    origin_type = pydantic_type if origin_type is None else origin_type

    if isclass(origin_type) and issubclass(origin_type, str):
        return PydanticDataType.STRING.value
    elif isclass(origin_type) and issubclass(origin_type, bool):
        return PydanticDataType.BOOLEAN.value
    elif isclass(origin_type) and issubclass(origin_type, int):
        return PydanticDataType.INTEGER.value
    elif isclass(origin_type) and issubclass(origin_type, float):
        return PydanticDataType.FLOAT.value
    elif isclass(origin_type) and issubclass(origin_type, Enum):
        return PydanticDataType.ENUM.value
    elif isclass(origin_type) and issubclass(origin_type, BaseModel):
        return format_model_and_field_name(origin_type.__name__)
    elif origin_type is list:
        element_type = get_args(pydantic_type)[0]
        return f"{map_pydantic_type_to_gbnf(element_type)}-list"
    elif origin_type is set:
        element_type = get_args(pydantic_type)[0]
        return f"{map_pydantic_type_to_gbnf(element_type)}-set"
    elif origin_type is Union:
        union_types = get_args(pydantic_type)
        union_rules = [map_pydantic_type_to_gbnf(ut) for ut in union_types]
        return f"union-{'-or-'.join(union_rules)}"
    elif origin_type is Optional:
        element_type = get_args(pydantic_type)[0]
        return f"optional-{map_pydantic_type_to_gbnf(element_type)}"
    elif isclass(origin_type):
        return f"{PydanticDataType.CUSTOM_CLASS.value}-{format_model_and_field_name(origin_type.__name__)}"
    elif origin_type is dict:
        key_type, value_type = get_args(pydantic_type)
        return f"custom-dict-key-type-{format_model_and_field_name(map_pydantic_type_to_gbnf(key_type))}-value-type-{format_model_and_field_name(map_pydantic_type_to_gbnf(value_type))}"
    else:
        return "unknown"
```

函数通过 `get_origin`/`get_args` 解构泛型类型，递归映射：
- 基础类型 → 对应枚举值
- `BaseModel` 子类 → kebab-case 模型名
- `list[T]`/`set[T]` → `<元素类型>-list/-set`
- `Union[...]` → `union-<类型1>-or-<类型2>...`
- `Optional[T]` → `optional-<T>`
- 普通类 → `custom-class-<Name>`
- `dict[K,V]` → `custom-dict-key-type-<K>-value-type-<V>`

### 9.10.3 format_model_and_field_name()：驼峰转 kebab-case

源码路径：`examples/pydantic_models_to_grammar.py` - `format_model_and_field_name()`（第117-126行）

```python
def format_model_and_field_name(model_name: str) -> str:
    parts = re.findall("[A-Z][^A-Z]*", model_name)
    if not parts:
        return model_name.lower().replace("_", "-")
    return "-".join(part.lower().replace("_", "-") for part in parts)
```

正则 `[A-Z][^A-Z]*` 拆分驼峰命名（如 `MyModel` → `["My", "Model"]`），转小写并用连字符连接，生成符合 GBNF 规则命名规范的标识符。

### 9.10.4 regex_to_gbnf()：简单正则翻译

源码路径：`examples/pydantic_models_to_grammar.py` - `regex_to_gbnf()`（第179-196行）

```python
def regex_to_gbnf(regex_pattern: str) -> str:
    gbnf_rule = regex_pattern
    gbnf_rule = gbnf_rule.replace("\\d", "[0-9]")
    gbnf_rule = gbnf_rule.replace("\\s", "[ \\t\\n]")
    return gbnf_rule
```

仅处理 `\d` → `[0-9]`、`\s` → `[ \t\n]` 两种常见转义，其余正则语法假定与 GBNF 兼容。

## 9.11 嵌套模型与特殊字符串 —— Pydantic 规则的"复杂组装"

### 9.11.1 generate_gbnf_rule_for_type()：字段类型调度器

源码路径：`examples/pydantic_models_to_grammar.py` - `generate_gbnf_rule_for_type()`（第296-415行）

```python
def generate_gbnf_rule_for_type(
    model_name, field_name, field_type, is_optional, processed_models, created_rules, field_info=None
) -> tuple[str, list[str]]:
    rules = []
    field_name = format_model_and_field_name(field_name)
    gbnf_type = map_pydantic_type_to_gbnf(field_type)
    origin_type = get_origin(field_type)
    origin_type = field_type if origin_type is None else origin_type

    if isclass(origin_type) and issubclass(origin_type, BaseModel):
        nested_model_name = format_model_and_field_name(field_type.__name__)
        nested_model_rules, _ = generate_gbnf_grammar(field_type, processed_models, created_rules)
        rules.extend(nested_model_rules)
        gbnf_type, rules = nested_model_name, rules
    elif isclass(origin_type) and issubclass(origin_type, Enum):
        enum_values = [f'"\\"{e.value}\\""' for e in field_type]
        enum_rule = f"{model_name}-{field_name} ::= {' | '.join(enum_values)}"
        rules.append(enum_rule)
        gbnf_type, rules = model_name + "-" + field_name, rules
    elif origin_type is list:
        element_type = get_args(field_type)[0]
        element_rule_name, additional_rules = generate_gbnf_rule_for_type(
            model_name, f"{field_name}-element", element_type, is_optional, processed_models, created_rules)
        rules.extend(additional_rules)
        array_rule = f"""{model_name}-{field_name} ::= "[" ws {element_rule_name} ("," ws {element_rule_name})*  "]""""
        rules.append(array_rule)
        gbnf_type, rules = model_name + "-" + field_name, rules
    elif origin_type is set:
        # 同 list 处理
        ...
    elif gbnf_type.startswith("custom-class-"):
        rules.append(get_members_structure(field_type, gbnf_type))
    elif gbnf_type.startswith("custom-dict-"):
        key_type, value_type = get_args(field_type)
        additional_key_type, additional_key_rules = generate_gbnf_rule_for_type(...)
        additional_value_type, additional_value_rules = generate_gbnf_rule_for_type(...)
        gbnf_type = rf'{gbnf_type} ::= "{"  ( {additional_key_type} ": "  {additional_value_type} ("," "\n" ws {additional_key_type} ": "  {additional_value_type})*  )? "}" '
        rules.extend(additional_key_rules); rules.extend(additional_value_rules)
    elif gbnf_type.startswith("union-"):
        union_types = get_args(field_type)
        union_rules = []
        for union_type in union_types:
            if not issubclass(union_type, type(None)):
                union_gbnf_type, union_rules_list = generate_gbnf_rule_for_type(...)
                union_rules.append(union_gbnf_type)
                rules.extend(union_rules_list)
        if len(union_rules) == 1:
            union_grammar_rule = f"{model_name}-{field_name}-optional ::= {' | '.join(union_rules)} | null"
        else:
            union_grammar_rule = f"{model_name}-{field_name}-union ::= {' | '.join(union_rules)}"
        rules.append(union_grammar_rule)
        gbnf_type = f"{model_name}-{field_name}-optional" if len(union_rules) == 1 else f"{model_name}-{field_name}-union"
    elif isclass(origin_type) and issubclass(origin_type, str):
        if field_info and field_info.json_schema_extra:
            triple_quoted_string = field_info.json_schema_extra.get("triple_quoted_string", False)
            markdown_string = field_info.json_schema_extra.get("markdown_code_block", False)
            gbnf_type = PydanticDataType.TRIPLE_QUOTED_STRING.value if triple_quoted_string else PydanticDataType.STRING.value
            gbnf_type = PydanticDataType.MARKDOWN_CODE_BLOCK.value if markdown_string else gbnf_type
        elif field_info and hasattr(field_info, "pattern"):
            regex_pattern = field_info.regex.pattern
            gbnf_type = f"pattern-{field_name} ::= {regex_to_gbnf(regex_pattern)}"
        else:
            gbnf_type = PydanticDataType.STRING.value
    elif issubclass(origin_type, float) and field_info and field_info.json_schema_extra:
        # 浮点精度约束 → generate_gbnf_float_rules
        ...
    elif issubclass(origin_type, int) and field_info and field_info.json_schema_extra:
        # 整数位数约束 → generate_gbnf_integer_rules
        ...
    return gbnf_type, rules
```

核心调度器，处理所有复杂类型组合：
- **嵌套 BaseModel**：递归生成子模型规则，主规则指向子模型名
- **Enum**：生成枚举值的 `|` 联合规则
- **list/set**：生成 `[ ws <elem> ("," ws <elem>)* ]` 结构
- **Union/Optional**：为每类型生成子规则，组合为 `union-...` 或 `optional-...`
- **custom-dict**：递归生成 key/value 子规则并组装
- **特殊字符串**：通过 `json_schema_extra` 检测 `triple_quoted_string`/`markdown_code_block` 标记
- **正则 pattern**：调用 `regex_to_gbnf` 简单翻译
- **数字精度/位数**：读取 `json_schema_extra` 中的 `max_precision`/`min_precision`/`max_digit`/`min_digit` 生成约束规则

### 9.11.2 generate_gbnf_integer_rules() 与 generate_gbnf_float_rules()：数字精度约束

源码路径：`examples/pydantic_models_to_grammar.py` - `generate_gbnf_integer_rules()`（第198-239行）、`generate_gbnf_float_rules()`（第241-294行）

```python
def generate_gbnf_integer_rules(max_digit=None, min_digit=None):
    additional_rules = []
    integer_rule = "integer-part" + (f"-max{max_digit}" if max_digit else "") + (f"-min{min_digit}" if min_digit else "")
    if max_digit is not None or min_digit is not None:
        integer_rule_part = ""
        if min_digit is not None:
            integer_rule_part += "[0-9] " * min_digit
        if max_digit is not None:
            optional_digits = max_digit - (min_digit if min_digit is not None else 0)
            integer_rule_part += "".join(["[0-9]? " for _ in range(optional_digits)])
        integer_rule_part = integer_rule_part.strip()
        if integer_rule_part:
            additional_rules.append(f"{integer_rule} ::= {integer_rule_part}")
    return integer_rule, additional_rules

def generate_gbnf_float_rules(max_digit=None, min_digit=None, max_precision=None, min_precision=None):
    additional_rules = []
    integer_part_rule = "integer-part" + (f"-max{max_digit}" if max_digit else "") + (f"-min{min_digit}" if min_digit else "")
    fractional_part_rule = "fractional-part" + (f"-max{max_precision}" if max_precision else "") + (f"-min{min_precision}" if min_precision else "")
    fractional_rule_part = "[0-9]" * (min_precision if min_precision is not None else 1)
    fractional_rule_part += "".join([" [0-9]?"] * ((max_precision - (min_precision if min_precision is not None else 1)) if max_precision is not None else 0))
    additional_rules.append(f"{fractional_part_rule} ::= {fractional_rule_part}")
    float_rule = f"float-{max_digit or 'X'}-{min_digit or 'X'}-{max_precision or 'X'}-{min_precision or 'X'}"
    additional_rules.append(f'{float_rule} ::= {integer_part_rule} "." {fractional_part_rule}')
    if max_digit is not None or min_digit is not None:
        integer_rule_part = "[0-9]" + (" [0-9]" * (min_digit - 1) if min_digit and min_digit > 1 else "") + "".join([" [0-9]?"] * (max_digit - (min_digit if min_digit is not None else 1)) if max_digit else [])
        additional_rules.append(f"{integer_part_rule} ::= {integer_rule_part.strip()}")
    return float_rule, additional_rules
```

整数规则生成 `integer-part-maxN-minM ::= [0-9] [0-9]? ...` 形式；浮点规则拆分为 `integer-part` 与 `fractional-part` 两个子规则，再组合为 `float-... ::= integer-part "." fractional-part`，支持独立控制整数位数与小数精度。

## 9.12 Pydantic 模型的"全家福" —— generate_gbnf_grammar 与主入口

### 9.12.1 generate_gbnf_grammar()：单模型规则生成

源码路径：`examples/pydantic_models_to_grammar.py` - `generate_gbnf_grammar()`（第417-488行）

```python
def generate_gbnf_grammar(model: type[BaseModel], processed_models: set[type[BaseModel]], created_rules: dict[str, list[str]]) -> tuple[list[str], bool]:
    if model in processed_models:
        return [], False
    processed_models.add(model)
    model_name = format_model_and_field_name(model.__name__)

    # 获取字段信息：Pydantic 用 model_fields，普通类用 __annotations__/__init__
    if not issubclass(model, BaseModel):
        if hasattr(model, "__annotations__") and model.__annotations__:
            model_fields = {name: (typ, ...) for name, typ in get_type_hints(model).items()}
        else:
            init_signature = inspect.signature(model.__init__)
            parameters = init_signature.parameters
            model_fields = {name: (param.annotation, param.default) for name, param in parameters.items() if name != "self"}
    else:
        model_fields = get_type_hints(model)

    model_rule_parts = []
    nested_rules = []
    has_markdown_code_block = False
    has_triple_quoted_string = False
    for field_name, field_info in model_fields.items():
        if not issubclass(model, BaseModel):
            field_type, default_value = field_info
            is_optional = (default_value is not inspect.Parameter.empty) and (default_value is not Ellipsis)
        else:
            field_type = field_info
            field_info = model.model_fields[field_name]
            is_optional = field_info.is_required is False and get_origin(field_type) is Optional

        rule_name, additional_rules = generate_gbnf_rule_for_type(
            model_name, format_model_and_field_name(field_name), field_type, is_optional, processed_models, created_rules, field_info)

        # 检测特殊字符串标记
        look_for_markdown_code_block = True if rule_name == "markdown_code_block" else False
        look_for_triple_quoted_string = True if rule_name == "triple_quoted_string" else False
        if not look_for_markdown_code_block and not look_for_triple_quoted_string:
            if rule_name not in created_rules:
                created_rules[rule_name] = additional_rules
            model_rule_parts.append(f' ws "\\"{field_name}\\"" ":" ws {rule_name}')
            nested_rules.extend(additional_rules)
        else:
            has_triple_quoted_string = look_for_triple_quoted_string
            has_markdown_code_block = look_for_markdown_code_block

    fields_joined = r' "," "\n" '.join(model_rule_parts)
    model_rule = rf'{model_name} ::= "{" "\n" {fields_joined} "\n" ws "}"'

    has_special_string = False
    if has_triple_quoted_string:
        model_rule += '"\\n" ws "}"'
        model_rule += '"\\n" triple-quoted-string'
        has_special_string = True
    if has_markdown_code_block:
        model_rule += '"\\n" ws "}"'
        model_rule += '"\\n" markdown-code-block'
        has_special_string = True
    all_rules = [model_rule] + nested_rules
    return all_rules, has_special_string
```

单模型生成流程：
1. 防循环：`processed_models` 集合
2. 统一获取字段：Pydantic 用 `model_fields` + `get_type_hints`，普通类回退 `__annotations__` 或 `__init__` 签名
3. 遍历字段调用 `generate_gbnf_rule_for_type`，收集嵌套规则
4. 检测 `markdown_code_block`/`triple_quoted_string` 特殊标记
5. 组装模型规则：`model ::= "{" "\n" field1 "," "\n" field2 ... "\n" ws "}"`
6. 特殊字符串时追加换行后的特殊规则引用

### 9.12.2 generate_gbnf_grammar_from_pydantic_models()：多模型顶层组装

源码路径：`examples/pydantic_models_to_grammar.py` - `generate_gbnf_grammar_from_pydantic_models()`（第490-574行）

```python
def generate_gbnf_grammar_from_pydantic_models(
    models: list[type[BaseModel]], outer_object_name: str | None = None, outer_object_content: str | None = None,
    list_of_outputs: bool = False) -> str:
    processed_models = set()
    all_rules = []
    created_rules = {}

    if outer_object_name is None:
        # 直接并联模式：root ::= ModelA | ModelB
        for model in models:
            model_rules, _ = generate_gbnf_grammar(model, processed_models, created_rules)
            all_rules.extend(model_rules)
        if list_of_outputs:
            root_rule = r'root ::= (" "| "\n") "[" ws grammar-models ("," ws grammar-models)* ws "]"' + "\n"
        else:
            root_rule = r'root ::= (" "| "\n") grammar-models' + "\n"
        root_rule += "grammar-models ::= " + " | ".join([format_model_and_field_name(model.__name__) for model in models])
        all_rules.insert(0, root_rule)
        return "\n".join(all_rules)

    elif outer_object_name is not None:
        # 函数调用模式：外层包 function 对象
        if list_of_outputs:
            root_rule = rf'root ::= (" "| "\n") "[" ws {format_model_and_field_name(outer_object_name)} ("," ws {format_model_and_field_name(outer_object_name)})* ws "]"' + "\n"
        else:
            root_rule = f"root ::= {format_model_and_field_name(outer_object_name)}\n"

        model_rule = rf'{format_model_and_field_name(outer_object_name)} ::= (" "| "\n") "{{" ws "\"{outer_object_name}\""  ":" ws grammar-models'
        fields_joined = " | ".join([rf"{format_model_and_field_name(model.__name__)}-grammar-model" for model in models])
        grammar_model_rules = f"\ngrammar-models ::= {fields_joined}"
        mod_rules = []
        for model in models:
            mod_rule = rf"{format_model_and_field_name(model.__name__)}-grammar-model ::= "
            mod_rule += rf'"\{model.__name__}\" "," ws "\{outer_object_content}\" ":" ws {format_model_and_field_name(model.__name__)}' + "\n"
            mod_rules.append(mod_rule)
        grammar_model_rules += "\n" + "\n".join(mod_rules)

        for model in models:
            model_rules, has_special_string = generate_gbnf_grammar(model, processed_models, created_rules)
            if not has_special_string:
                model_rules[0] += r'"\n" ws "}"'
            all_rules.extend(model_rules)

        all_rules.insert(0, root_rule + model_rule + grammar_model_rules)
        return "\n".join(all_rules)
```

三种顶层模式：
- **无外层对象**：`root ::= ModelA | ModelB`，支持 `list_of_outputs=True` 输出数组
- **单函数调用**：`root ::= function` → `function ::= {"function": "...", "params": ModelX}`
- **并发函数调用**：`list_of_outputs=True` 时 `root ::= [ function, function, ... ]`

### 9.12.3 get_primitive_grammar()：自动补全基础依赖

源码路径：`examples/pydantic_models_to_grammar.py` - `get_primitive_grammar()`（第576-626行）

```python
def get_primitive_grammar(grammar):
    type_list = []
    if "string-list" in grammar: type_list.append(str)
    if "boolean-list" in grammar: type_list.append(bool)
    if "integer-list" in grammar: type_list.append(int)
    if "float-list" in grammar: type_list.append(float)
    additional_grammar = [generate_list_rule(t) for t in type_list]

    primitive_grammar = r"""
boolean ::= "true" | "false"
null ::= "null"
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\"" ws
ws ::= ([ \t\n] ws)?
float ::= ("-"? ([0] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
integer ::= [0-9]+"""
    
    any_block = ""
    if "custom-class-any" in grammar:
        any_block = """value ::= object | array | string | number | boolean | null
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
number ::= integer | float"""

    markdown_code_block_grammar = ""
    if "markdown-code-block" in grammar:
        markdown_code_block_grammar = r'''markdown-code-block ::= opening-triple-ticks markdown-code-block-content closing-triple-ticks
markdown-code-block-content ::= ( [^`] | "`" [^`] | "`" "`" [^`] )*
opening-triple-ticks ::= "```" "python" "\n" | "```" "c" "\n" | ... 
closing-triple-ticks ::= "```" "\n"'''

    if "triple-quoted-string" in grammar:
        markdown_code_block_grammar = r"""triple-quoted-string ::= triple-quotes triple-quoted-string-content triple-quotes
triple-quoted-string-content ::= ( [^'] | "'" [^'] | "'" "'" [^'] )*
triple-quotes ::= "'''\""""

    return "\n" + "\n".join(additional_grammar) + any_block + primitive_grammar + markdown_code_block_grammar
```

扫描生成的 grammar 字符串，按需追加：
- 列表规则（string-list 等）
- 基础类型规则
- `value/object/array/number` 通用规则（当出现 `custom-class-any`）
- Markdown 代码块 / 三引号字符串专用规则

## 9.13 动态模型工厂 —— 从函数到 Pydantic 的"变形术"

### 9.13.1 create_dynamic_model_from_function()：函数签名+docstring → Pydantic 类

源码路径：`examples/pydantic_models_to_grammar.py` - `create_dynamic_model_from_function()`（第1147-1204行）

```python
def create_dynamic_model_from_function(func: Callable[..., Any]):
    sig = inspect.signature(func)
    assert func.__doc__ is not None
    docstring = parse(func.__doc__)  # docstring_parser 支持 Google/NumPy/Sphinx 风格
    dynamic_fields = {}
    param_docs = []
    for param in sig.parameters.values():
        if param.name == "self": continue
        if param.annotation == inspect.Parameter.empty:
            raise TypeError(...)
        param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)
        if not param_doc or not param_doc.description:
            raise ValueError(...)
        param_docs.append((param.name, param_doc))
        default_value = ... if param.default == inspect.Parameter.empty else param.default
        dynamic_fields[param.name] = (param.annotation, default_value)
    dynamic_model = create_model(f"{getattr(func, '__name__')}", **dynamic_fields)
    for name, param_doc in param_docs:
        dynamic_model.model_fields[name].description = param_doc.description
    dynamic_model.__doc__ = docstring.short_description
    def run_method_wrapper(self):
        func_args = {name: getattr(self, name) for name, _ in dynamic_fields.items()}
        return func(**func_args)
    setattr(dynamic_model, "run", run_method_wrapper)
    return dynamic_model
```

利用 `inspect.signature` 提取参数类型注解，`docstring_parser.parse` 解析文档字符串（需每参数有描述），动态创建 Pydantic 模型并注入 `run()` 方法调用原函数，实现从函数到可执行模型的转换。

### 9.13.2 convert_dictionary_to_pydantic_model()：JSON Schema/OpenAI 字典 → Pydantic 类

源码路径：`examples/pydantic_models_to_grammar.py` - `convert_dictionary_to_pydantic_model()`（第1277-1349行）

```python
def convert_dictionary_to_pydantic_model(dictionary: dict[str, Any], model_name: str = "CustomModel") -> type[Any]:
    fields = {}
    if "properties" in dictionary:
        for field_name, field_data in dictionary.get("properties", {}).items():
            if field_data == "object":
                submodel = convert_dictionary_to_pydantic_model(dictionary, f"{model_name}_{field_name}")
                fields[field_name] = (submodel, ...)
            else:
                field_type = field_data.get("type", "str")
                if field_data.get("enum", []):
                    fields[field_name] = (list_to_enum(field_name, field_data.get("enum", [])), ...)
                elif field_type == "array":
                    items = field_data.get("items", {})
                    if items != {}:
                        array = {"properties": items}
                        array_type = convert_dictionary_to_pydantic_model(array, f"{model_name}_{field_name}_items")
                        fields[field_name] = (list[array_type], ...)
                    else:
                        fields[field_name] = (list, ...)
                elif field_type == "object":
                    submodel = convert_dictionary_to_pydantic_model(field_data, f"{model_name}_{field_name}")
                    fields[field_name] = (submodel, ...)
                elif field_type == "required":
                    required = field_data.get("enum", [])
                    for key, field in fields.items():
                        if key not in required:
                            optional_type = fields[key][0]
                            fields[key] = (Optional[optional_type], ...)
                else:
                    field_type = json_schema_to_python_types(field_type)
                    fields[field_name] = (field_type, ...)
    # 支持 OpenAI function calling 格式：{"function": {"name":..., "parameters":...}}
    if "function" in dictionary: ...
    if "parameters" in dictionary: ...
    if "required" in dictionary: ...
    custom_model = create_model(model_name, **fields)
    return custom_model
```

递归解析字典结构：支持 `properties`、`enum`、`array`、`object`、`required` 等关键字，兼容 OpenAI Function Calling 格式（`function.name/description/parameters`），自动处理嵌套对象与数组项类型。

### 9.13.3 批量工厂与 run 方法注入

源码路径：`examples/pydantic_models_to_grammar.py` - `create_dynamic_models_from_dictionaries()`（第1234-1247行）、`add_run_method_to_dynamic_model()`（第1207-1231行）

```python
def create_dynamic_models_from_dictionaries(dictionaries: list[dict[str, Any]]):
    dynamic_models = []
    for func in dictionaries:
        model_name = format_model_and_field_name(func.get("name", ""))
        dyn_model = convert_dictionary_to_pydantic_model(func, model_name)
        dynamic_models.append(dyn_model)
    return dynamic_models

def add_run_method_to_dynamic_model(model: type[BaseModel], func: Callable[..., Any]):
    def run_method_wrapper(self):
        func_args = {name: getattr(self, name) for name in model.model_fields}
        return func(**func_args)
    setattr(model, "run", run_method_wrapper)
    return model
```

批量转换字典列表，以及为已有模型后向注入 `run` 方法，配合动态模型工厂形成完整工具链。

## 9.14 文档生成器 —— 让 LLM "读懂" Pydantic 模型的"说明书"

### 9.14.1 generate_markdown_documentation()：Markdown 格式文档

源码路径：`examples/pydantic_models_to_grammar.py` - `generate_markdown_documentation()`（第628-683行）

```python
def generate_markdown_documentation(
    pydantic_models: list[type[BaseModel]], model_prefix="Model", fields_prefix="Fields",
    documentation_with_field_description=True) -> str:
    documentation = ""
    pyd_models = [(model, True) for model in pydantic_models]
    for model, add_prefix in pyd_models:
        if add_prefix:
            documentation += f"{model_prefix}: {model.__name__}\n"
        else:
            documentation += f"Model: {model.__name__}\n"
        class_doc = getdoc(model)
        base_class_doc = getdoc(BaseModel)
        class_description = class_doc if class_doc and class_doc != base_class_doc else ""
        if class_description != "":
            documentation += "  Description: "
            documentation += format_multiline_description(class_description, 0) + "\n"
        if add_prefix:
            documentation += f"  {fields_prefix}:\n"
        else:
            documentation += f"  Fields:\n"
        if isclass(model) and issubclass(model, BaseModel):
            for name, field_type in get_type_hints(model).items():
                if get_origin(field_type) == list:
                    element_type = get_args(field_type)[0]
                    if isclass(element_type) and issubclass(element_type, BaseModel):
                        pyd_models.append((element_type, False))
                if get_origin(field_type) == Union:
                    for element_type in get_args(field_type):
                        if isclass(element_type) and issubclass(element_type, BaseModel):
                            pyd_models.append((element_type, False))
                documentation += generate_field_markdown(
                    name, field_type, model, documentation_with_field_description=documentation_with_field_description)
            documentation += "\n"
        if hasattr(model, "Config") and hasattr(model.Config, "json_schema_extra") and "example" in model.Config.json_schema_extra:
            documentation += f"  Expected Example Output for {format_model_and_field_name(model.__name__)}:\n"
            json_example = json.dumps(model.Config.json_schema_extra["example"])
            documentation += format_multiline_description(json_example, 2) + "\n"
    return documentation
```

生成结构化 Markdown：模型名、描述、字段列表（递归发现嵌套 BaseModel）、示例输出。缩进用 4 空格表达层次。

### 9.14.2 generate_field_markdown() 与 generate_text_documentation()

源码路径：`examples/pydantic_models_to_grammar.py` - `generate_field_markdown()`（第685-741行）、`generate_text_documentation()`（第841-891行）

```python
def generate_field_markdown(field_name, field_type, model, depth=1, documentation_with_field_description=True):
    indent = "    " * depth
    field_info = model.model_fields.get(field_name)
    field_description = field_info.description if field_info and field_info.description else ""
    origin_type = get_origin(field_type)
    origin_type = field_type if origin_type is None else origin_type

    if origin_type == list:
        element_type = get_args(field_type)[0]
        field_text = f"{indent}{field_name} ({format_model_and_field_name(field_type.__name__)} of {format_model_and_field_name(element_type.__name__)})"
    elif origin_type == Union:
        element_types = get_args(field_type)
        types = [format_model_and_field_name(t.__name__) for t in element_types]
        field_text = f"{indent}{field_name} ({' or '.join(types)})"
    else:
        field_text = f"{indent}{field_name} ({format_model_and_field_name(field_type.__name__)})"
    if field_description: field_text += f":\n        Description: {field_description}\n"
    else: field_text += "\n"
    if documentation_with_field_description and field_description:
        field_text += f"        Description: {field_description}\n"
    # 示例与嵌套详情
    ...
    return field_text
```

字段文档区分 `list[T]`、`Union[...]`、普通类型三种渲染方式，递归处理嵌套 BaseModel。`generate_text_documentation` 结构对称但缩进风格更适合 LLM 上下文。

## 9.15 Pydantic 结构化输出示例 —— json_schema_pydantic_example.py

### 9.15.1 create_completion()：通用 HTTP 客户端 + Schema 注入

源码路径：`examples/json_schema_pydantic_example.py` - `create_completion()`（第13-37行）

```python
def create_completion(*, response_model=None, endpoint="http://localhost:8080/v1/chat/completions", messages, **kwargs):
    response_format = None
    type_adapter = None
    if response_model:
        type_adapter = TypeAdapter(response_model)
        schema = type_adapter.json_schema()
        messages = [{
            "role": "system",
            "content": f"You respond in JSON format with the following schema: {json.dumps(schema, indent=2)}"
        }] + messages
        response_format = {"type": "json_object", "schema": schema}

    data = requests.post(endpoint, headers={"Content-Type": "application/json"},
                         json=dict(messages=messages, response_format=response_format, **kwargs)).json()
    if 'error' in data:
        raise Exception(data['error']['message'])
    content = data["choices"][0]["message"]["content"]
    return type_adapter.validate_json(content) if type_adapter else content
```

核心流程：
1. `TypeAdapter(response_model).json_schema()` 获取 Pydantic 模型的 JSON Schema
2. 构造系统提示词注入 Schema，引导 LLM 输出符合格式的 JSON
3. `response_format={"type": "json_object", "schema": schema}` 强制 JSON 输出
4. 调用本地 llama-server `/v1/chat/completions` 端点
5. `type_adapter.validate_json(content)` 反序列化并校验

### 9.15.2 QAPair 与 PyramidalSummary：约束示例模型

源码路径：`examples/json_schema_pydantic_example.py` - 类定义（第47-70行）

```python
class QAPair(BaseModel):
    class Config:
        extra = 'forbid'  # triggers additionalProperties: false in the JSON schema
    question: str
    concise_answer: str
    justification: str
    stars: Annotated[int, Field(ge=1, le=5)]

class PyramidalSummary(BaseModel):
    class Config:
        extra = 'forbid'
    title: str
    summary: str
    question_answers: Annotated[List[QAPair], MinLen(2)]
    sub_sections: Optional[Annotated[List['PyramidalSummary'], MinLen(2)]]
```

- `extra='forbid'` → JSON Schema `additionalProperties: false`，禁止额外字段
- `Annotated[int, Field(ge=1, le=5)]` → 数值范围约束
- `Annotated[List[QAPair], MinLen(2)]` → 集合最小长度约束
- `Optional[Annotated[List['PyramidalSummary'], MinLen(2)]]` → 递归自引用 + 可选 + 长度约束

## 9.16 函数调用示例库 —— pydantic_models_to_grammar_examples.py

### 9.16.1 create_completion()：/completion 端点客户端

源码路径：`examples/pydantic_models_to_grammar_examples.py` - `create_completion()`（第27-37行）

```python
def create_completion(host, prompt, gbnf_grammar):
    print(f"  Request:\n    Grammar:\n{textwrap.indent(gbnf_grammar, '      ')}\n    Prompt:\n{textwrap.indent(prompt.rstrip(), '      ')}")
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "grammar": gbnf_grammar}
    result = requests.post(f"http://{host}/completion", headers=headers, json=data).json()
    assert data.get("error") is None, data
    logging.info("Result: %s", result)
    content = result["content"]
    print(f"  Model: {result['model']}")
    print(f"  Result:\n{textwrap.indent(json.dumps(json.loads(content), indent=2), '    ')}")
    return content
```

直接调用 llama-server `/completion` 端点（非 OpenAI 兼容格式），传入 `prompt` 与 `grammar`，打印请求/响应便于调试。

### 9.16.2 SendMessageToUser 与 Calculator：工具模型定义

源码路径：`examples/pydantic_models_to_grammar_examples.py` - `SendMessageToUser`（第42-43行）、`Calculator`（第69-83行）

```python
class SendMessageToUser(BaseModel):
    chain_of_thought: str
    message: str
    def run(self):
        print(f"SendMessageToUser: {self.message}")

class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

class Calculator(BaseModel):
    number_one: Union[int, float]
    operation: MathOperation
    number_two: Union[int, float]
    def run(self):
        if self.operation == MathOperation.ADD:
            return self.number_one + self.number_two
        elif self.operation == MathOperation.SUBTRACT:
            return self.number_one - self.number_two
        elif self.operation == MathOperation.MULTIPLY:
            return self.number_one * self.number_two
        elif self.operation == MathOperation.DIVIDE:
            return self.number_one / self.number_two
        else:
            raise ValueError("Unknown operation.")
```

- `SendMessageToUser`：最简工具模型，演示 RCE（任意代码执行）风险
- `Calculator`：Enum 参数 `MathOperation` + `Union[int, float]` 多类型参数，GBNF 生成联合规则

### 9.16.3 example_rce() 与 example_calculator()：单/多工具调用演示

源码路径：`examples/pydantic_models_to_grammar_examples.py` - `example_rce()`（第47-62行）、`example_calculator()`（第89-112行）

```python
def example_rce(host):
    tools = [SendMessageToUser]
    gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
        pydantic_model_list=tools, outer_object_name="function", outer_object_content="function_parameters",
        model_prefix="Function", fields_prefix="Parameters")
    system_message = "You are an advanced AI... " + documentation
    user_message = "What is 42 * 42?"
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
    text = create_completion(host, prompt, gbnf_grammar)
    json_data = json.loads(text)
    tools_map = {tool.__name__: tool for tool in tools}
    tool = tools_map.get(json_data["function"])
    tool(**json_data["function_parameters"]).run()
```

```python
def example_calculator(host):
    tools = [SendMessageToUser, Calculator]
    gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(...)
    # system_message 构造同 example_rce
    user_message1 = "What is 42 * 42?"
    text = create_completion(host, prompt, gbnf_grammar)
    json_data = json.loads(text)
    expected = {"function": "Calculator", "function_parameters": {"number_one": 42, "operation": "multiply", "number_two": 42}}
    if json_data != expected: print("  Result is not as expected!")
    tool = tools_map.get(json_data["function"])
    result = tool(**json_data["function_parameters"]).run()
    print(f"  Call {json_data['function']} gave result {result}")
```

`example_rce` 演示最小函数调用链路；`example_calculator` 验证 LLM 输出与预期字典一致，并执行计算。

### 9.16.4 example_struct()：非函数调用结构化输出

源码路径：`examples/pydantic_models_to_grammar_examples.py` - `example_struct()`（第130-143行）

```python
def example_struct(host):
    tools = [Book]
    gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(pydantic_model_list=tools)
    system_message = "You are an advanced AI, tasked to create a dataset entry in JSON for a Book... " + documentation
    text = "The Feynman Lectures on Physics is a physics textbook..."
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant"
    text = create_completion(host, prompt, gbnf_grammar)
    json_data = json.loads(text)
    keys = sorted(["title", "author", "published_year", "keywords", "category", "summary"])
    if keys != sorted(json_data.keys()): print(f"Unexpected result: {sorted(json_data.keys())}")
    book = Book(**json_data)
    print(f"  As a Book object: %s" % book)
```

不使用 `outer_object_name`，直接以 `Book` 模型作为根规则，从非结构化文本提取结构化字段。

### 9.16.5 example_concurrent()：并发函数调用

源码路径：`examples/pydantic_models_to_grammar_examples.py` - `example_concurrent()`（第164-213行）

```python
def example_concurrent(host):
    current_weather_tool = {"type": "function", "function": {"name": "get_current_weather", ...}}
    current_weather_tool_model = convert_dictionary_to_pydantic_model(current_weather_tool)
    current_weather_tool_model = add_run_method_to_dynamic_model(current_weather_tool_model, get_current_weather)
    current_datetime_model = create_dynamic_model_from_function(get_current_datetime)
    tools = [SendMessageToUser, Calculator, current_datetime_model, current_weather_tool_model]
    gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
        pydantic_model_list=tools, outer_object_name="function", outer_object_content="params",
        model_prefix="Function", fields_prefix="Parameters", list_of_outputs=True)
    # prompt 构造...
    text = create_completion(host, prompt, gbnf_grammar)
    json_data = json.loads(text)
    expected = [
        {"function": "get_current_datetime", "params": {"output_format": "%Y-%m-%d %H:%M:%S"}},
        {"function": "get_current_weather", "params": {"location": "London", "unit": "celsius"}},
        {"function": "Calculator", "params": {"number_one": 42, "operation": "multiply", "number_two": 42}},
    ]
    if json_data != expected: print("  Result is not as expected!")
    for call in json_data:
        tool = tools_map.get(call["function"])
        result = tool(**call["params"]).run()
        print(f"  Call {call['function']} returned {result}")
```

演示三种工具来源融合：
- Python 函数 `get_current_datetime` → `create_dynamic_model_from_function`
- Pydantic 模型 `Calculator` → 直接使用
- OpenAI 字典 `get_current_weather` → `convert_dictionary_to_pydantic_model` + `add_run_method_to_dynamic_model`

`list_of_outputs=True` 生成数组根规则，支持单次推理输出多个函数调用。

## 9.17 嵌入服务并发测试 —— server_embd.py 的"压力小工"

### 9.17.1 requests_post_async()：requests 异步包装

源码路径：`examples/server_embd.py` - `requests_post_async()`（第13-14行）

```python
async def requests_post_async(*args, **kwargs):
    return await asyncio.threads.to_thread(requests.post, *args, **kwargs)
```

使用 `asyncio.threads.to_thread` 将阻塞的 `requests.post` 放入线程池，实现异步并发。

### 9.17.2 main()：并发请求 /embedding 端点

源码路径：`examples/server_embd.py` - `main()`（第16-25行）

```python
async def main():
    model_url = "http://127.0.0.1:6900"
    responses: list[requests.Response] = await asyncio.gather(*[requests_post_async(
        url=f"{model_url}/embedding",
        json={"content": "a "*1022}
    ) for i in range(n)])
    for response in responses:
        embedding = response.json()["embedding"]
        print(embedding[-8:])
        result.append(embedding)
```

并发发起 `n=8` 个 POST 请求，所有请求发送相同文本 `"a "*1022`，收集各响应的 `embedding` 字段。

### 9.17.3 __main__：余弦相似度计算

源码路径：`examples/server_embd.py` - `__main__`（第8-28行）

```python
n = 8
result = []
asyncio.run(main())
for i in range(n-1):
    for j in range(i+1, n):
        embedding1 = np.array(result[i])
        embedding2 = np.array(result[j])
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        print(f"Similarity between {i} and {j}: {similarity:.2f}")
```

计算两两向量的余弦相似度。理想情况下相似度应接近 1.0，验证嵌入模型在并发场景下的确定性与数值稳定性。

## 9.18 设计中的取舍

**为什么不直接用 JSON Schema 做约束？**

JSON Schema 表达能力强但验证复杂，LLM 推理时需实时约束 token 选择。GBNF 是上下文无关文法，可直接编译为有限状态自动机，在采样时逐 token 过滤非法续写，零开销集成到推理流程。SchemaConverter 将 JSON Schema 离线编译为 GBNF，把复杂度前置到转换阶段。

**_visit_pattern 为什么不支持反向引用？**

GBNF 文法是上下文无关的，不支持正则的反向引用（`\1` 等上下文相关特性）。JSON Schema pattern 虽允许，但转换器明确拒绝：`assert pattern[i] != '?'` 阻断 `(?=...)` 等预查语法，`{m,n}` 量词通过子规则展开而非原生支持。这是为了保证生成的 GBNF 可被 llama.cpp 采样器高效执行。

**Pydantic 转换器为什么区分 `optional-` 与 `union-` 前缀？**

`Optional[T]` 语义等价于 `Union[T, None]`，但 GBNF 生成策略不同：`optional-T ::= T-rule | null` 单分支加 null；`union-... ::= T1 | T2 | ...` 多分支不含 null。区分前缀避免规则名冲突，也便于 `get_primitive_grammar` 精确识别依赖。

**为什么需要动态模型工厂？**

实际应用中工具定义常来自 Python 函数（本地开发）或 OpenAI 字典（远程注册/配置文件）。统一转为 Pydantic 模型后，复用同一套 GBNF 生成、文档生成、run 执行流水线，避免为每种来源维护独立逻辑。

## 9.19 动手练习

### 9.19.1 练习1：追踪 JSON Schema 到 GBNF 的转换全过程

**内容**：
阅读 examples/json_schema_to_grammar.py 第449-530行与第232-244行，追踪以下 Schema 转换的完整调用链：
1. `{"type": "object", "properties": {"name": {"type": "string", "minLength": 1}}, "required": ["name"]}`
2. `{"oneOf": [{"type": "string"}, {"type": "number"}]}`

**回答问题**：
- 上述 Schema 会调用 visit() 中的哪些分支？
- _build_object_rule 如何处理 required 属性？
- 最终生成的 GBNF 规则文本大概是什么样子？

### 9.19.2 练习2：深入正则翻译器

**内容**：
阅读 examples/json_schema_to_grammar.py 第281-379行的 _visit_pattern 方法，尝试理解以下正则的转换：
1. `^[A-Z][a-z]+$`（大写开头的单词）
2. `^\d{3}-\d{4}$`（电话号码格式）
3. `^(.+)\1$`（带反向引用的复杂模式）

**回答问题**：
- join_seq() 函数如何合并连续字面量？为什么需要优化？
- sub_rule_ids 缓存机制在 {m,n} 量词中如何复用规则？
- 第三个模式会被正确转换吗？为什么不支持反向引用？

### 9.19.3 练习3：Pydantic 到 GBNF 的类型映射实验

**内容**：
阅读 examples/pydantic_models_to_grammar.py 第76-115行的类型映射函数：
1. 编写一个包含 str、int、float、Enum、list[str]、Optional[int]、Union[int, str] 的 Pydantic 模型
2. 调用 generate_gbnf_grammar_and_documentation 生成 GBNF
3. 调用 get_primitive_grammar 追加依赖规则

**回答问题**：
- format_model_and_field_name 如何处理 MyEnum 这种缩写？
- list[str] 和 set[str] 生成的规则有何区别？
- 为什么 Optional[T] 和 Union[T, None] 会生成不同的规则名前缀？

### 9.19.4 练习4：动态模型工厂的工作流

**内容**：
阅读 examples/pydantic_models_to_grammar.py 第1147-1204行的 create_dynamic_model_from_function：
1. 编写一个带类型注解和 Google 风格 docstring 的 Python 函数
2. 用 create_dynamic_model_from_function 转换为 Pydantic 类
3. 用 add_run_method_to_dynamic_model 验证 run() 方法可用

**回答问题**：
- 缺少类型注解或 docstring 时会抛出什么错误？
- docstring_parser.parse 支持哪些 docstring 风格？
- 转换后的模型如何与 generate_gbnf_grammar_and_documentation 衔接？

### 9.19.5 练习5：结构化输出的端到端实验

**内容**：
阅读 examples/json_schema_pydantic_example.py 全文与 examples/pydantic_models_to_grammar_examples.py 第218-273行的 example_concurrent：
1. 启动本地 llama-server
2. 运行 example_concurrent，观察输出的多个函数调用
3. 尝试修改 expected 字典中的某个参数值，再次运行观察报错信息

**回答问题**：
- create_completion 如何把 Pydantic 模型转换为 JSON Schema？
- TypeAdapter 与直接 model.schema() 的区别是什么？
- 如果 LLM 输出不符合预期，客户端会得到什么反馈？

## 9.20 本章小结

这一章中我们学习了 llama.cpp 生态中结构化输出与语法约束的完整工具链。首先深入剖析了 `json_schema_to_grammar.py` 中的 `SchemaConverter` 核心转换器，理解其如何递归遍历 JSON Schema，通过 `_add_rule` 去重注册、`resolve_refs` 解析引用、`_visit_pattern` 状态机翻译正则、`_build_object_rule` 组装对象排列组合，最终输出确定性的 GBNF 文法。其次探索了 `regex_to_grammar.py` 如何通过子进程复用主转换器实现正则到 GBNF 的轻量管道。接着系统讲解了 `pydantic_models_to_grammar.py` 的类型映射体系：`PydanticDataType` 枚举、`map_pydantic_type_to_gbnf` 递归映射、`generate_gbnf_rule_for_type` 处理嵌套模型/Enum/列表/联合/字典/特殊字符串/数字约束，以及 `generate_gbnf_grammar` 与 `generate_gbnf_grammar_from_pydantic_models` 的单/多模型组装策略。随后揭秘了动态模型工厂 `create_dynamic_model_from_function` 与 `convert_dictionary_to_pydantic_model` 如何将函数签名、OpenAI 字典统一变形为 Pydantic 类，配合 `run` 方法注入实现可执行工具模型。文档生成器 `generate_markdown_documentation` 与 `generate_text_documentation` 则为 LLM 系统提示词提供可读说明书。最后通过三个示例程序验证端到端链路：`json_schema_pydantic_example.py` 演示 OpenAI 兼容接口下的 Schema 注入与校验；`pydantic_models_to_grammar_examples.py` 覆盖 RCE 最小调用、Enum/Union 参数计算器、非函数调用结构化抽取、三源融合并发调用；`server_embd.py` 以并发请求验证嵌入服务的数值确定性。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| `SchemaConverter` | JSON Schema 到 GBNF 的中央转换器，递归遍历 Schema 构建文法规则 |
| `BuiltinRule` | 内置规则容器，封装字符串规则内容与依赖列表 |
| `PRIMITIVE_RULES` | 6 种 JSON 基础类型预定义规则（string/number/boolean/null/object/array） |
| `STRING_FORMAT_RULES` | uuid/date/time/date-time 等特殊字符串格式规则 |
| `SPACE_RULE` | 空格约束规则，限制模型生成过多空白 |
| `DOTALL/DOT` | 正则 . 元字符匹配字符集（DOTALL 含换行，DOT 不含 \n\r） |
| `RESERVED_NAMES` | 保留名称集合，防止与内置规则重名 |
| `_build_repetition` | 生成 {min,max} 重复规则，支持可选分隔符 |
| `_generate_min_max_int` | 按位生成整数区间规则，支持负数与小数位数控制 |
| `_format_literal` | 对字面量进行转义，生成符合 GBNF 的 "..." 字符串 |
| `not_literal` | 生成否定字面量规则，如 [^a] 或递归的非匹配结构 |
| `_not_strings` | 基于 Trie 树的非匹配字符串生成，用于 additionalProperties 的 key 约束 |
| `_add_rule` | 规则去重与命名，自动添加数字后缀避免冲突 |
| `resolve_refs` | 解析 $ref 字段，支持内部 # 指针与 HTTPS 远程加载 |
| `_generate_union_rule` | 生成 oneOf/anyOf 联合类型规则，使用 alternative- 前缀 |
| `_visit_pattern` | 正则到 GBNF 翻译器，使用状态机+join_seq 字面量合并+sub_rule_ids 缓存 |
| `_resolve_ref` | 引用规则解析，将 #fragment 转为 ref-name 规则 |
| `_generate_constant_rule` | const/enum 字面量生成，使用 json.dumps 处理嵌套结构 |
| `_add_primitive` | 注入基础规则到规则表，自动递归添加依赖 |
| `visit` | Schema 入口分发，按 type/oneOf/allOf/pattern 等字段路由到不同处理函数 |
| `_build_object_rule` | 对象规则构建，处理必选/可选/额外属性的所有合法组合 |
| `format_grammar` | 按规则名排序输出最终的 GBNF 规则文本 |
| `regex_to_grammar.py` | 正则到 GBNF 的轻量子进程包装，构造最小结构调用主转换器 |
| `PydanticDataType` | Pydantic 支持的数据类型枚举（含 triple_quoted_string 等扩展类型） |
| `map_pydantic_type_to_gbnf` | Python 类型到 GBNF 规则名的递归映射函数 |
| `format_model_and_field_name` | 驼峰名转 kebab-case 格式 |
| `generate_list_rule` | 生成 "," 分隔的列表规则模板 |
| `get_members_structure` | 从普通类（Enum/含 __annotations__/含 __init__）提取结构生成规则 |
| `regex_to_gbnf` | 简单正则翻译（\d→[0-9], \s→[ \t\n]） |
| `generate_gbnf_integer_rules` | 按 max_digit/min_digit 生成固定位数的整数规则 |
| `generate_gbnf_float_rules` | 按精度约束生成 float-*-*-*-* 格式规则 |
| `generate_gbnf_rule_for_type` | 字段类型到 GBNF 规则的转换调度器（含 BaseModel/Enum/list/Union/dict 等） |
| `generate_gbnf_grammar` | 从单个 Pydantic 模型生成完整 GBNF 规则列表（含 markdown_code_block/triple_quoted_string 检测） |
| `generate_gbnf_grammar_from_pydantic_models` | 多模型顶层规则组装，支持 outer_object_name 和 list_of_outputs |
| `get_primitive_grammar` | 根据 grammar 内容自动追加基础依赖规则（boolean/list/float 等） |
| `generate_markdown_documentation` | 为 LLM 系统提示词生成 Markdown 格式的模型说明书 |
| `generate_field_markdown` | 递归生成字段的 Markdown 文档（含 list/Union/BaseModel 分支） |
| `generate_text_documentation` | 生成纯文本格式文档 |
| `generate_field_text` | 递归生成字段的纯文本文档 |
| `format_multiline_description` | 多行描述按缩进级别格式化 |
| `save_gbnf_grammar_and_documentation` | 将 GBNF 与 Markdown 写入指定文件 |
| `remove_empty_lines` | 去除字符串中的空行 |
| `generate_and_save_gbnf_grammar_and_documentation` | 一键生成 GBNF 与 Markdown 并保存 |
| `generate_gbnf_grammar_and_documentation` | 返回 (grammar, documentation) 元组，不写文件 |
| `generate_gbnf_grammar_and_documentation_from_dictionaries` | 从字典列表生成 GBNF 与文档 |
| `create_dynamic_model_from_function` | 从 Python 函数签名+docstring 动态创建 Pydantic 类 |
| `convert_dictionary_to_pydantic_model` | 从 JSON Schema 字典/OpenAI 函数定义生成 Pydantic 类 |
| `create_dynamic_models_from_dictionaries` | 批量从字典列表创建 Pydantic 模型 |
| `add_run_method_to_dynamic_model` | 为已有 Pydantic 模型注入 run() 方法 |
| `map_grammar_names_to_pydantic_model_class` | 建立 GBNF 规则名到 Pydantic 模型类的反向索引 |
| `json_schema_to_python_types` | JSON Schema 类型字符串到 Python 类型的映射 |
| `list_to_enum` | 从值列表动态创建 Enum 子类 |
| `format_json_example` | 将 JSON 示例字典格式化为可读字符串 |
| `create_completion` | 结构化输出客户端，注入 schema 到系统提示词并强制 JSON 输出 |
| `QAPair` | 数值范围约束示例（Annotated[int, Field(ge=1, le=5)]） |
| `PyramidalSummary` | 嵌套递归 + 集合长度约束示例 |
| `example_rce` | RCE 演示：任意代码执行风险示例 |
| `example_calculator` | Enum 参数+Union[int,float]参数的 GBNF 生成示例 |
| `example_struct` | 非函数调用结构化输出：从非结构化文本提取书籍信息 |
| `example_concurrent` | Python 函数+Pydantic+OpenAI 字典三种工具的并发函数调用 |
| `server_embd.py` | 并发测试 llama-server /embedding 端点的稳定性与确定性 |

下一章中，我们将学习 llama.cpp 的采样系统——决定下一个 Token 的"指挥官"，理解从贪心、Top-K、Top-P 到 Mirostat、Grammar、DRY 等高级采样算法的实现原理与采样器链组合机制。

