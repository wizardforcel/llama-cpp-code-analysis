



# 第82章 server_models 调度中枢 —— 多模型实例的「列车调度室」

## 学习目标

- 理解 server_models 作为多模型「列车调度室」的核心职责与线程安全设计
- 掌握模型生命周期管理：加载（spawn）、监控（双保险线程）、卸载（优雅退出/强制杀死）
- 理解 LRU 淘汰策略与 models_max 并发上限的协同机制
- 掌握 HTTP 代理转发（server_http_proxy）与流式管道（pipe_t）的实现原理
- 熟悉跨平台兼容层：路径获取、环境变量、端口分配等系统调用的封装技巧
- 能阅读并修改模型路由 REST 端点（/models/load、/models/unload）
- 理解子进程与父进程间的命令通信协议（READY/SLEEP/EXIT）
- 掌握请求参数解析与模型自动加载（autoload）机制
- 掌握线程同步原语（mutex、condition_variable、atomic）在多进程管理中的应用
- 理解预设合并优先级与启动参数渲染机制

## 82.1 生活类比

想象 llama-server 的模型管理系统是一家**大型智能火车站的调度中心**：**server_models** = 列车调度总台（统一指挥所有车次的进出站）；**server_model_meta** = 每趟列车的「车牌档案」（记录着车型号、当前状态、停靠站台、最后发车时间）；**LRU 淘汰机制** = 站台容量管理（站台满了？让最久没发车的列车先回库休息）；**spawn 子进程** = 列车出站（调度台启动新车次，分配专属轨道 port，并派监控员跟车）；**双监控线程** = 随车安全员（一人看日志汇总车况，一人盯紧急制动信号，超时未归就强制拖回）；**pipe_t 管道** = 列车与调度中心的「专用对讲机」（流式传输实况，一喊停就断线）；**HTTP 代理转发** = 中转售票窗口（乘客在总服务台买票，后台自动转给具体车次窗口处理）；**状态机 UNLOADED/LOADING/LOADED/SLEEPING** = 列车状态（车库待命/正在出库/运行中/临时休眠）；**命令协议（READY/SLEEP/EXIT）** = 列车与调度台的「暗号」（标准应答信号，确保双方状态同步）。就像调度中心需要精确控制每趟列车的进出时间、防止站台拥堵，server_models 通过互斥锁和条件变量确保多线程环境下模型加载、卸载、状态迁移互不干扰，实现高并发下的稳定调度。

---

## 82.2 源码地图

```text
tools/server/server-models.h
├── 数据结构与枚举
│   ├── enum server_model_status              # 模型状态枚举「交通灯」
│   │   ├── SERVER_MODEL_STATUS_UNLOADED
│   │   ├── SERVER_MODEL_STATUS_LOADING
│   │   ├── SERVER_MODEL_STATUS_LOADED
│   │   └── SERVER_MODEL_STATUS_SLEEPING
│   │
│   ├── struct server_model_meta            # 模型元数据「身份证」
│   │   ├── preset / name / aliases / tags  # 预设配置与标识
│   │   ├── port / status / last_used       # 运行状态与时间戳
│   │   ├── update_args()                   # 渲染启动参数（注入host/port/alias）
│   │   ├── is_ready() / is_running() / is_failed()  # 状态查询
│   │   └── 构造函数与成员初始化
│   │
│   ├── class server_models                 # 多模型调度中枢（线程安全）
│   │   ├── 构造与初始化
│   │   │   ├── server_models()               # 构造函数：初始化预设、环境、路径
│   │   │   └── load_models()                 # 加载全部模型配置（缓存/本地/INI）
│   │   ├── 实例管理
│   │   │   ├── add_model()                   # 注册新模型到映射表
│   │   │   ├── load()                        # 加载模型：spawn子进程+双线程监控
│   │   │   ├── unload()                      # 卸载指定模型
│   │   │   ├── unload_all()                  # 停止所有模型
│   │   │   └── unload_lru()                  # LRU淘汰：释放最久未用实例
│   │   ├── 状态查询
│   │   │   ├── has_model() / get_meta() / get_all_meta()
│   │   │   ├── update_meta() / update_status()
│   │   │   ├── wait_until_loading_finished()
│   │   │   └── ensure_model_ready()          # 阻塞等待模型就绪
│   │   └── 子进程通信
│   │       ├── is_child_server()             # 检测是否子进程模式
│   │       ├── setup_child_server()          # 子进程初始化（发送READY）
│   │       └── notify_router_sleeping_state() # 通知父进程睡眠状态
│   │
│   ├── struct server_models_routes           # HTTP路由注册器
│   │   ├── server_models_routes()            # 构造函数：初始化并解析WebUI配置
│   │   └── init_routes()                     # 注册GET/POST处理逻辑
│   │
│   ├── struct server_http_proxy              # HTTP代理转发器
│   │   ├── server_http_proxy()               # 构造：建立连接、启动转发线程
│   │   └── ~server_http_proxy()              # 析构：清理回调
│   │
│   └── 工具函数与常量
│       ├── server_model_status_from_string() # 字符串转状态枚举
│       ├── server_model_status_to_string()   # 状态枚举转字符串
│       └── 常量定义（DEFAULT_STOP_TIMEOUT, CMD_*, CHILD_ADDR）
tools/server/server-models.cpp
├── 平台兼容层与工具函数
│   ├── get_server_exec_path()            # 获取可执行文件路径（Win/macOS/Linux）
│   ├── get_environment()                 # 获取环境变量（宽字符转UTF-8）
│   ├── wide_to_utf8()                    # Windows专用：宽字符转UTF-8
│   ├── get_free_port()                   # 获取空闲端口（bind(0)+getsockname）
│   ├── to_char_ptr_array()               # vector<string> → char*[]
│   ├── unset_reserved_args()             # 清理保留参数（SSL/API key等）
│   ├── to_lower_copy()                   # 字符串转小写
│   └── should_strip_proxy_header()       # 过滤代理响应头（CORS/Transfer-Encoding等）
│
├── 核心类方法实现
│   └── server_models
│       ├── server_models()               # 构造：初始化预设上下文、base_env、bin_path
│       ├── load_models()                 # 三源合并：缓存→本地→INI→base_preset
│       ├── add_model()                   # 解析alias/tags，校验冲突
│       ├── load()                        # 三步走：LRU→get_free_port→subprocess_create
│       │   └── 监控线程组
│       │       ├── log_thread            # 日志转发+READY/SLEEP信号识别
│       │       └── stopping_thread       # 监听退出/崩溃，超时force kill
│       ├── unload() / unload_all()       # 停止信号发送与线程join
│       ├── unload_lru()                  # 遍历找最久未用，阻塞等待退出
│       ├── ensure_model_ready()          # 加载+wait_until_loading_finished
│       ├── proxy_request()               # 创建server_http_proxy转发请求
│       └── update_status() / update_meta() # 线程安全的状态更新
│
├── struct pipe_t                         # 流式管道（共享队列+条件变量）
│   ├── read()                            # 阻塞读取，支持取消（close_read）
│   ├── write()                           # 写入数据，检测broken pipe
│   └── close_write() / close_read()      # 关闭端点并通知对端
│
├── server_http_proxy                     # HTTP代理实现
│   └── server_http_proxy()               # 构造：过滤头、异步发送、pipe流式读取
│
├── server_models_routes                  # 路由处理
│   ├── init_routes()                     # 注册lambda处理函数
│   ├── get_router_props()                # 返回/router属性（role:router）
│   ├── get_router_models()               # 返回模型列表（含状态、args、preset快照）
│   ├── post_router_models_load()         # POST /models/load
│   ├── proxy_get() / proxy_post()        # 转发请求到具体模型
│   └── router_validate_model()           # 验证模型存在性与autoload逻辑
│
├── res_ok() / res_err()                  # 统一响应格式化
├── is_autoload()                         # 解析autoload参数
└── 常量定义与全局代码
    ├── DEFAULT_STOP_TIMEOUT              # 默认停止超时时间（10秒）
    ├── CMD_ROUTER_TO_CHILD_EXIT          # 父进程到子进程：退出命令
    ├── CMD_CHILD_TO_ROUTER_READY         # 子进程到父进程：就绪信号
    ├── CMD_CHILD_TO_ROUTER_SLEEP         # 子进程到父进程：睡眠信号
    ├── CHILD_ADDR                        # 子进程监听地址（127.0.0.1）
```

---

## 82.3 核心数据结构：server_model_status 与 server_model_meta —— 模型状态的「交通灯」与「身份证」

### 82.3.1 server_model_status - 模型状态枚举「交通灯」

模型状态机只有四种状态，类比列车的四种运行态，所有调度决策均基于此状态流转。源码路径：`tools/server/server-models.h` 第15-21行。

```cpp
enum server_model_status {
    SERVER_MODEL_STATUS_UNLOADED,   // 未加载（在车库待命）
    SERVER_MODEL_STATUS_LOADING,    // 加载中（正在出库）
    SERVER_MODEL_STATUS_LOADED,     // 已加载就绪（运行中）
    SERVER_MODEL_STATUS_SLEEPING    // 休眠中（临时靠边停车）
};
```

这一段代码定义了模型完整生命周期的四态枚举。`UNLOADED` 是初始态与终止态；`LOADING` 是过渡态，等待子进程 `READY` 信号；`LOADED` 是服务态，可接收推理请求；`SLEEPING` 是节能态，子进程仍存活但暂不服务，新请求可唤醒。这四个状态构成了调度中心判断车辆能否发车、是否需要回库的核心依据。

---

### 82.3.2 server_model_meta - 模型元数据「身份证」

每个模型实例的完整档案，包含静态配置与动态运行时信息。源码路径：`tools/server/server-models.h` 第47-67行。

```cpp
struct server_model_meta {
    common_preset preset;                    // 预设配置（车辆配置单）
    std::string name;                        // 模型名称（车牌号）
    std::set<std::string> aliases;           // 别名集合（备用车牌）
    std::set<std::string> tags;              // 标签（车型分类）
    int port = 0;                            // 监听端口（专用站台号）
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED; // 当前状态
    int64_t last_used = 0;                   // 最后使用时间（最后发车时间戳）
    std::vector<std::string> args;           // 渲染后的启动参数（发车指令单）
    int exit_code = 0;                       // 退出码（事故报告码）
    int stop_timeout = DEFAULT_STOP_TIMEOUT; // 优雅停止超时（等待回库时限）

    // 状态查询方法
    bool is_ready() const {
        return status == SERVER_MODEL_STATUS_LOADED;
    }

    bool is_running() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING || status == SERVER_MODEL_STATUS_SLEEPING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }

    // 根据预设渲染最终启动参数
    void update_args(common_preset_context & ctx_preset, std::string bin_path);
};
```

这一段代码封装了模型运行所需的全部信息：从静态配置（preset、aliases、tags）到动态状态（status、port、last_used、exit_code），并提供了 `is_ready()`、`is_running()`、`is_failed()` 三个便捷查询方法，供调度逻辑快速判断模型可用性。每个模型实例在调度中心都有一份这样完整的档案，记录着它的配置单、当前状态、停靠站台、最后发车时间以及出站指令单。

---

### 82.3.3 server_model_meta::update_args - 渲染启动参数「发车指令单」

将预设模板实例化为最终命令行参数，注入运行时动态值（host、port、alias）。源码路径：`tools/server/server-models.cpp` 第127-134行。

```cpp
void server_model_meta::update_args(common_preset_context & ctx_preset, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);                    // 清理保留参数
    preset.set_option(ctx_preset, "LLAMA_ARG_HOST",  CHILD_ADDR);      // 注入 127.0.0.1
    preset.set_option(ctx_preset, "LLAMA_ARG_PORT",  std::to_string(port)); // 注入动态端口
    preset.set_option(ctx_preset, "LLAMA_ARG_ALIAS", name);              // 注入模型名称
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);                         // 生成最终启动参数
}
```

这一段代码实现了从预设模板到最终启动参数的渲染过程。首先清理预设中可能残留的路由器级参数（SSL、models_max 等），避免污染子进程；然后强制绑定 `127.0.0.1`，防止子进程监听 `0.0.0.0` 导致回环问题；接着注入 `get_free_port()` 分配的唯一端口号；再注入模型规范名称作为 alias，便于子进程自我标识；最后调用预设引擎渲染最终 `argv` 向量，存入 `args` 供 `subprocess_create_ex` 使用。这一过程如同调度员根据车辆配置单，填入具体的站台号、车牌号，生成一份最终的发车指令单。

---

### 82.3.4 server_model_status_to_string / server_model_status_from_string - 状态序列化「信号灯翻译官」

提供枚举与字符串的双向转换，用于 HTTP API 响应与配置解析。源码路径：`tools/server/server-models.h` 第23-45行。

```cpp
static server_model_status server_model_status_from_string(const std::string & status_str) {
    if (status_str == "unloaded") { return SERVER_MODEL_STATUS_UNLOADED; }
    if (status_str == "loading")  { return SERVER_MODEL_STATUS_LOADING;  }
    if (status_str == "loaded")   { return SERVER_MODEL_STATUS_LOADED;   }
    if (status_str == "sleeping") { return SERVER_MODEL_STATUS_SLEEPING; }
    throw std::runtime_error("invalid server model status");
}

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_UNLOADED: return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:  return "loading";
        case SERVER_MODEL_STATUS_LOADED:   return "loaded";
        case SERVER_MODEL_STATUS_SLEEPING: return "sleeping";
        default:                           return "unknown";
    }
}
```

这一段代码实现了标准的枚举-字符串互转工具。`from_string` 遇到非法值抛异常，`to_string` 兜底返回 `"unknown"`，保证 API 兼容性。这就像调度中心的信号灯翻译官，将内部的四种状态翻译成外部可读的字符串，或将外部传来的字符串解析为内部状态。

---

## 82.4 server_models 调度中枢：构造与预设加载 —— 调度台的「开站准备」与「建档入库」

### 82.4.1 server_models::server_models - 构造函数「开站仪式」

初始化预设上下文、收集环境变量、定位可执行文件路径，启动模型加载流程。源码路径：`tools/server/server-models.cpp` 第140-159行。

```cpp
server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params),
              base_env(get_environment()),
              base_preset(ctx_preset.load_from_args(argc, argv)) {
    // clean up base preset
    unset_reserved_args(base_preset, true);           // 清理模型级参数
    // set binary path
    try {
        bin_path = get_server_exec_path().string();   // 跨平台获取可执行路径
    } catch (const std::exception & e) {
        bin_path = argv[0];                           // 兜底用 argv[0]
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using original argv[0] as fallback: %s\n", argv[0]);
    }
    load_models();                                    // 触发三源合并加载
}
```

这一段代码完成了调度中心的开站准备工作。首先初始化预设上下文 `ctx_preset`，指定服务器场景常量 `LLAMA_EXAMPLE_SERVER`；保存基础启动参数 `base_params`（含 models_dir、models_max 等）；调用 `get_environment()` 获取当前进程环境变量副本（Windows 宽字符转 UTF-8）；从命令行参数解析基础预设 `base_preset`（最高优先级）；清理 `base_preset` 中的模型级参数（`--model`、`--mmproj` 等），仅保留全局参数；跨平台获取可执行文件绝对路径，失败回退到 `argv[0]`；最后调用 `load_models()` 执行三源合并与模型注册。这就像调度中心启动时，先确立场景规则、收集环境信息、定位调度指挥车位置，再开始建立车辆档案库。

---

### 82.4.2 server_models::load_models - 三源合并策略「三源合并建档」

从缓存、本地目录、INI 文件三源加载预设，层层叠加，最后注入 `base_preset` 生成最终配置。源码路径：`tools/server/server-models.cpp` 第222-343行。

```cpp
void server_models::load_models() {
    // 1. 缓存预设（优先级最低）
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_INF("Loaded %zu cached model presets\n", cached_models.size());

    // 2. 本地目录预设（--models-dir，覆盖缓存）
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_INF("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }

    // 3. 自定义 INI 预设（--models-preset，合并或新增）
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        SRV_INF("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }

    // 级联：global 作为基础层应用到三个源
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // 合并：本地覆盖缓存（同名模型本地优先）
    common_presets final_presets;
    for (const auto & [name, preset] : cached_models) {
        final_presets[name] = preset;
    }
    for (const auto & [name, preset] : local_models) {
        final_presets[name] = preset;  // 覆盖
    }

    // 处理 INI 自定义预设
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            final_presets[name].merge(custom);  // 存在则深度合并
        } else {
            final_presets[name] = custom;       // 不存在则直接加入
        }
    }

    // 最后注入 base_preset（命令行参数，最高优先级）
    for (auto & [name, preset] : final_presets) {
        preset.merge(base_preset);
    }

    // 转为 server_model_meta 并注册
    for (const auto & preset : final_presets) {
        server_model_meta meta{ /* ... 初始化所有字段 ... */ };
        add_model(std::move(meta));
    }

    // 处理 stop-timeout 与 autoload 启动（省略部分日志代码）
    // ...
}
```

这一段代码实现了**四层优先级合并**：缓存 < 本地目录 < INI 文件 < 命令行。`cascade` 先应用 global 基础配置，`merge` 实现深度覆盖。最终遍历合并后的预设构造 `server_model_meta`，调用 `add_model` 注册到 `mapping`。随后处理每个模型的 `stop-timeout` 配置和 `load_on_startup` 自动加载。这就像调度中心从三个渠道（历史缓存、本地车库、自定义调度单）收集车辆配置，先打上全局底色，再按优先级层层覆盖，最后盖上命令行的最高权限印章，生成最终档案入库。

---

### 82.4.3 server_models::add_model - 注册新模型「核发车牌」

解析预设中的 alias/tags，三层冲突检测，渲染参数，构造实例放入映射表。源码路径：`tools/server/server-models.cpp` 第161-221行。

```cpp
void server_models::add_model(server_model_meta && meta) {
    // 1. 模型名唯一性检查
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // 2. 模型名不得与现有别名冲突
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // 3. 解析预设中的 --alias 选项（逗号分隔）
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // 4. 解析预设中的 --tags 选项（逗号分隔）
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // 5. 新别名不得与现有模型名或别名冲突
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    // 6. 渲染最终启动参数并构造实例
    meta.update_args(ctx_preset, bin_path);
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<subprocess_s>(),
        /* th      */ std::thread(),
        /* meta    */ std::move(meta)
    };
}
```

这一段代码实现了三层冲突检测保证全局命名空间无歧义：(1) 名称唯一；(2) 名称不冲突已有别名；(3) 新别名不冲突现有名称或别名。解析预设中的 `LLAMA_ARG_ALIAS` 和 `LLAMA_ARG_TAGS`（逗号分隔、去空格），校验通过后渲染参数并构造 `instance_t` 入表。这就像车管所核发车牌：先查重车牌号、再查重备用车牌、解析备用车牌标签、确认无冲突后发放正式车牌并录入档案库。

---

## 82.5 线程安全的状态查询与更新 —— 调度大厅的「查询柜台」与「广播系统」

所有对 `mapping` 的读写均受 `std::mutex` 保护，状态变更后通过 `condition_variable` 广播通知等待者。这一组方法构成了调度中心对外提供查询服务和内部状态同步的基础设施。

### 82.5.1 server_models::has_model - 存在性检查（支持别名）「查车牌」

源码路径：`tools/server/server-models.cpp` 第356-366行。

```cpp
bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {           // 直接命中规范名
        return true;
    }
    for (const auto & [key, inst] : mapping) {           // 遍历检查别名
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}
```

这一段代码在持锁下查询模型是否存在。先直接在映射表中查找规范名称，命中即返回真；否则遍历所有模型检查别名集合，支持别名透明解析。这就像查询柜台先查正式车牌，查不到再查备用车牌登记簿。

---

### 82.5.2 server_models::get_meta - 获取元数据副本（支持别名）「取档案副本」

源码路径：`tools/server/server-models.cpp` 第368-381行。

```cpp
std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;                          // 返回副本
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;                            // 别名解析到规范名
        }
    }
    return std::nullopt;
}
```

这一段代码在持锁下获取模型元数据副本。直接命中返回副本，否则遍历别名解析到规范名再返回副本，找不到返回空。返回副本而非引用避免了锁外访问共享数据的竞争风险，体现了线程安全设计的防御性原则。

---

### 82.5.3 server_models::get_all_meta - 获取所有模型元数据快照「全量档案快照」

源码路径：`tools/server/server-models.cpp` 第418-427行。

```cpp
std::vector<server_model_meta> server_models::get_all_meta() {
    std::lock_guard<std::mutex> lk(mutex);
    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);                     // 拷贝所有副本
    }
    return result;
}
```

这一段代码在持锁下遍历映射表，拷贝所有模型元数据副本到向量返回。预留容量避免扩容开销，全量快照用于 `/models` 列表接口响应。

---

### 82.5.4 server_models::update_meta - 线程安全更新元数据并通知「更新档案并广播」

源码路径：`tools/server/server-models.cpp` 第346-354行。

```cpp
void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all();  // 唤醒 wait_until_loading_finished 等待者
}
```

这一段代码在持锁下更新指定模型的元数据，修改后调用 `cv.notify_all()` 广播通知。这是条件变量同步机制的关键一环：所有等待状态变更的线程（如 `wait_until_loading_finished`）都在 `cv` 上等待，状态更新后广播唤醒它们重新检查条件。这就像档案更新后立即通过广播系统通知所有关注该车辆状态的调度员。

---

## 82.6 子进程生命周期管理 —— 列车的「出站、监控、回库、清场」

### 82.6.1 server_models::load - 三步走加载「列车出站」

**核心流程**：LRU预检 → 双重检查并发上限 → 分配端口 → Spawn子进程 → 启动双监控线程。源码路径：`tools/server/server-models.cpp` 第463-597行。

```cpp
void server_models::load(const std::string & name) {
    if (!has_model(name)) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    unload_lru();  // ① 先尝试腾位置（锁外）

    std::lock_guard<std::mutex> lk(mutex);  // ② 持锁二次检查

    auto meta = mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // ③ 二次检查并发上限（防止 unload_lru 释锁到此获锁窗口期并发超限）
    if (base_params.models_max > 0) {
        size_t count_active = 0;
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                count_active++;
            }
        }
        if (count_active >= (size_t)base_params.models_max) {
            throw std::runtime_error("model limit reached, try again later");
        }
    }

    // ④ 准备实例信息
    instance_t inst;
    inst.meta           = meta;
    inst.meta.port      = get_free_port();           // 分配空闲端口
    inst.meta.status    = SERVER_MODEL_STATUS_LOADING;
    inst.meta.last_used = ggml_time_ms();

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<subprocess_s>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        inst.meta.update_args(ctx_preset, bin_path); // 渲染参数

        std::vector<std::string> child_args = inst.meta.args;
        std::vector<std::string> child_env  = base_env;
        child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port)); // 注入父进程端口

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args;

        std::vector<char *> argv = to_char_ptr_array(child_args);
        std::vector<char *> envp = to_char_ptr_array(child_env);

        int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        int result = subprocess_create_ex(argv.data(), options, envp.data(), inst.subproc.get());
        if (result != 0) {
            throw std::runtime_error("failed to spawn server instance");
        }

        inst.stdin_file = subprocess_stdin(inst.subproc.get());
    }

    // ⑤ 启动管理线程（捕获子进程句柄、端口、超时配置）
    inst.th = std::thread([this, name, child_proc = inst.subproc, port = inst.meta.port, stop_timeout = inst.meta.stop_timeout]() {
        FILE * stdin_file = subprocess_stdin(child_proc.get());
        FILE * stdout_file = subprocess_stdout(child_proc.get()); // 合并 stdout/stderr

        // --- 日志转发线程 ---
        std::thread log_thread([&]() {
            if (stdout_file) {
                char buffer[4096];
                while (fgets(buffer, sizeof(buffer), stdout_file) != nullptr) {
                    LOG("[%5d] %s", port, buffer);          // 转发日志
                    std::string str(buffer);
                    if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_READY)) {
                        this->update_status(name, SERVER_MODEL_STATUS_LOADED, 0);  // 识别 READY
                    } else if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_SLEEP)) {
                        this->update_status(name, SERVER_MODEL_STATUS_SLEEPING, 0); // 识别 SLEEP
                    }
                }
            } else {
                SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
            }
        });

        // --- 停止监控线程 ---
        std::thread stopping_thread([&]() {
            auto is_stopping = [this, &name]() {
                return this->stopping_models.find(name) != this->stopping_models.end();
            };
            auto should_wake = [&]() {
                return is_stopping() || !subprocess_alive(child_proc.get());
            };
            {
                std::unique_lock<std::mutex> lk(this->mutex);
                this->cv_stop.wait(lk, should_wake);        // 等待停止信号或子进程死亡
            }
            // 子进程可能已崩溃退出
            if (!subprocess_alive(child_proc.get())) {
                return;
            }
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            // 发送 EXIT 命令
            fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
            fflush(stdin_file);
            // 优雅等待或超时强杀
            int64_t start_time = ggml_time_ms();
            while (true) {
                std::unique_lock<std::mutex> lk(this->mutex);
                if (!is_stopping()) {
                    return; // 已被其他线程处理
                }
                int64_t elapsed = ggml_time_ms() - start_time;
                if (elapsed >= stop_timeout * 1000) {
                    SRV_WRN("force-killing model instance name=%s after %d seconds timeout\n", name.c_str(), stop_timeout);
                    subprocess_terminate(child_proc.get()); // 强制杀死
                    return;
                }
                this->cv_stop.wait_for(lk, std::chrono::seconds(1)); // 这里用 wait_for 实现定时轮询检查超时
            }
        });

        // 等待子进程退出（log_thread 读取 stdout 直到 EOF）
        if (log_thread.joinable()) {
            log_thread.join();
        }

        // 清理停止监控线程
        {
            std::lock_guard<std::mutex> lk(this->mutex);
            stopping_models.erase(name);
            cv_stop.notify_all();
        }
        if (stopping_thread.joinable()) {
            stopping_thread.join();
        }

        // 获取退出码并清理
        int exit_code = 0;
        subprocess_join(child_proc.get(), &exit_code);
        subprocess_destroy(child_proc.get());

        this->update_status(name, SERVER_MODEL_STATUS_UNLOADED, exit_code);
        SRV_INF("instance name=%s exited with status %d\n", name.c_str(), exit_code);
    });

    // ⑥ 清理旧实例（防止残留）
    {
        auto & old_instance = mapping[name];
        if (subprocess_alive(old_instance.subproc.get())) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            subprocess_terminate(old_instance.subproc.get());
        }
        if (old_instance.th.joinable()) {
            old_instance.th.join();
        }
    }

    mapping[name] = std::move(inst);
    cv.notify_all();
}
```

这一段代码实现了模型加载的完整流程。采用**双重检查锁定**防止并发超限：锁外调用 `unload_lru()` 尝试腾位，锁内二次计数确认。分配端口后构造子进程参数，注入 `LLAMA_SERVER_ROUTER_PORT` 供子进程识别父进程。启动子进程后，管理线程内部派生两个监控线程：`log_thread` 持续读取子进程合并后的 stdout/stderr，转发日志并识别 `READY`/`SLEEP` 信号调用 `update_status` 更新状态，如同随车安全员一边记录行车日志一边盯着仪表盘确认车辆就绪或休眠；`stopping_thread` 在 `cv_stop` 上等待停止信号（`stopping_models` 集合）或子进程意外退出（`subprocess_alive` 为假），被唤醒后发送 `EXIT` 命令，每秒检查一次是否超时，超时则 `subprocess_terminate` 强杀，如同另一名安全员专盯紧急制动信号，超时未停车就强制拖回车库。管理线程最后回收资源，更新状态为 `UNLOADED` 并记录退出码。

---

### 82.6.2 server_models::unload - 优雅停止信号「发车回库」

不直接杀进程，而是发送停止信号，由管理线程负责后续流程。源码路径：`tools/server/server-models.cpp` 第599-612行。

```cpp
void server_models::unload(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            stopping_models.insert(name);      // 标记为待停止
            cv_stop.notify_all();              // 唤醒 stopping_thread
            // 状态变更由管理线程处理
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}
```

这一段代码实现了优雅停止的起始动作。在持锁下检查模型是否运行中，若是则插入 `stopping_models` 集合并 `cv_stop.notify_all()` 唤醒管理线程中的 `stopping_thread`。实际的停止流程（发送 EXIT、等待/强杀、清理）由管理线程异步执行，避免阻塞调用方。这就像调度员在回库登记簿上打钩并按下呼叫按钮，随车安全员收到信号后执行具体的回库操作。

---

### 82.6.3 server_models::unload_all - 全量停止「全站清场」

遍历所有运行中模型发送停止信号，将管理线程移出映射表后再 `join`，避免死锁。源码路径：`tools/server/server-models.cpp` 第614-632行。

```cpp
void server_models::unload_all() {
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lk(mutex);
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) {
                SRV_INF("stopping model instance name=%s\n", name.c_str());
                stopping_models.insert(name);
                cv_stop.notify_all();
            }
            // 移出线程对象，避免在持锁时 join 死锁
            to_join.push_back(std::move(inst.th));
        }
    }
    for (auto & th : to_join) {
        if (th.joinable()) {
            th.join();
        }
    }
}
```

这一段代码实现了全站清场。在持锁下遍历所有运行中模型发送停止信号，同时将管理线程对象 `move` 到局部向量 `to_join`，释放锁后再逐个 `join`。若在持锁时 `join`，管理线程退出时需获取同一互斥锁（如 `update_status`），会导致死锁。这种“持锁收集、释锁等待”的模式是多线程资源清理的标准范式。

---

### 82.6.4 server_models::unload_lru - LRU淘汰「最久未用先走」

当活跃模型数达到 `models_max` 上限，自动淘汰最久未使用的实例。源码路径：`tools/server/server-models.cpp` 第429-462行。

```cpp
void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // 无限制
    }
    std::string lru_model_name = "";
    int64_t lru_last_used = ggml_time_ms();  // 单调时钟，防 NTP 跳变
    size_t count_active = 0;
    {
        std::unique_lock<std::mutex> lk(mutex);
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                count_active++;
                if (m.second.meta.last_used < lru_last_used) {
                    lru_model_name = m.first;
                    lru_last_used = m.second.meta.last_used;
                }
            }
        }
    }
    if (!lru_model_name.empty() && count_active >= (size_t)base_params.models_max) {
        SRV_INF("models_max limit reached, removing LRU name=%s\n", lru_model_name.c_str());
        unload(lru_model_name);
        // 阻塞等待卸载完成
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this, &lru_model_name]() {
                return mapping[lru_model_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
    }
}
```

这一段代码实现了 LRU 淘汰策略。在持锁遍历下找到 `last_used` 最小的运行中模型，调用 `unload` 发送停止信号，随后在锁内等待条件变量直到该模型状态变为 `UNLOADED`。使用 `ggml_time_ms()` 单调时钟保证 LRU 判定不受系统时间调整影响。这就像站台满员时，调度员查阅发车记录，让最久没发车的那趟列车先回库腾出站台。

---

### 82.6.5 server_models::wait_until_loading_finished - 阻塞等待加载完成「等发车确认」

源码路径：`tools/server/server-models.cpp` 第646-656行。

```cpp
void server_models::wait_until_loading_finished(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [this, &name]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return it->second.meta.status != SERVER_MODEL_STATUS_LOADING;
        }
        return false;
    });
}
```

这一段代码使用条件变量等待模型状态离开 `LOADING`。Lambda 捕获 `this` 和 `name`，每次被唤醒时检查模型是否存在且状态不再为 `LOADING`。这就像调度员在出站信号灯前等待，直到绿灯亮起（状态变为 LOADED 或其他非 LOADING 态）。

---

### 82.6.6 server_models::ensure_model_ready - 确保模型就绪（自动加载+阻塞等待）「保障发车」

路由层调用此方法保证模型可服务，内部结合自动加载与阻塞等待。源码路径：`tools/server/server-models.cpp` 第658-690行。

```cpp
bool server_models::ensure_model_ready(const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->is_ready()) {
        return false; // 已就绪
    }
    if (meta->status == SERVER_MODEL_STATUS_SLEEPING) {
        return false; // 睡眠态视为可用，新请求会唤醒
    }
    if (meta->status == SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model name=%s is not loaded, loading...\n", name.c_str());
        load(name);
    }

    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    wait_until_loading_finished(name);

    meta = get_meta(name);
    if (!meta.has_value() || meta->is_failed()) {
        throw std::runtime_error("model name=" + name + " failed to load");
    }

    return true;
}
```

这一段代码实现了“检查-加载-等待-校验”的完整就绪保障流程。先检查缓存：已就绪直接返回；休眠态视为可用（请求会唤醒）；未加载则触发 `load()` 并阻塞等待 `wait_until_loading_finished`，最终校验是否加载失败（`is_failed()`）。返回 `true` 表示本次调用触发了加载，`false` 表示模型原本就可用。这是路由层转发请求前的“闸口”检查。

---

### 82.6.7 server_models::update_status - 统一状态更新入口「状态变更广播台」

源码路径：`tools/server/server-models.cpp` 第634-644行。

```cpp
void server_models::update_status(const std::string & name, server_model_status status, int exit_code) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        meta.status    = status;
        meta.exit_code = exit_code;
    }
    cv.notify_all();  // 广播唤醒所有等待者
}
```

这一段代码是所有状态变更的统一入口。`log_thread` 识别信号、管理线程退出清理均通过此方法持锁修改状态并广播通知，保证 `wait_until_loading_finished` 和 `ensure_model_ready` 等待者能即时感知。这就像调度中心的中央广播系统，任何状态变更都第一时间广播全站。

---

### 82.6.8 server_models::proxy_request - HTTP代理转发入口「中转售票」

创建 `server_http_proxy` 对象转发请求到子进程，POST 请求更新 `last_used` 触发 LRU 刷新。源码路径：`tools/server/server-models.cpp` 第692-718行。

```cpp
server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (!meta->is_running()) {
        throw std::invalid_argument("model name=" + name + " is not running");
    }
    if (update_last_used) {
        std::unique_lock<std::mutex> lk(mutex);
        mapping[name].meta.last_used = ggml_time_ms();  // 更新 LRU 时间戳
    }
    SRV_INF("proxying request to model %s on port %d\n", name.c_str(), meta->port);
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += '?' + req.query_string;
    }
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            "http",
            CHILD_ADDR,           // 127.0.0.1
            meta->port,
            proxy_path,
            req.headers,
            req.body,
            req.should_stop,
            base_params.timeout_read,
            base_params.timeout_write
            );
    return proxy;
}
```

这一段代码实现了代理转发的入口逻辑。校验模型存在且运行中，POST 请求更新 `last_used` 刷新 LRU 时间戳，构造目标路径（含查询字符串），创建 `server_http_proxy` 执行实际转发。这就像总服务台收到乘客请求，核实车次运行中，更新发车记录，然后打电话转给具体车次窗口办理。

---

## 82.7 子进程自我感知与通信协议 —— 列车与调度台的「暗号联络」

父子进程通过**标准输出/标准输入**传递文本命令，零依赖、跨平台、易调试。

### 82.7.1 server_models::is_child_server - 子进程身份识别「查员工证」

源码路径：`tools/server/server-models.cpp` 第720-723行。

```cpp
bool server_models::is_child_server() {
    const char * router_port = std::getenv("LLAMA_SERVER_ROUTER_PORT");
    return router_port != nullptr;
}
```

这一段代码通过检测环境变量 `LLAMA_SERVER_ROUTER_PORT` 判断自己是否运行在路由器管理模式下。父进程启动子进程时注入该变量，子进程借此识别身份。这就像列车员上车先查员工证（环境变量），确认自己是受调度中心管理的列车而非独立运行。

---

### 82.7.2 server_models::setup_child_server - 子进程初始化「上车报到」

启动时发送 `READY` 信号，启动监控线程阻塞读取 `stdin` 等待 `EXIT` 命令或 EOF。源码路径：`tools/server/server-models.cpp` 第725-753行。

```cpp
std::thread server_models::setup_child_server(const std::function<void(int)> & shutdown_handler) {
    // 发送 READY 通知父进程
    common_log_pause(common_log_main());
    fflush(stdout);
    fprintf(stdout, "%s\n", CMD_CHILD_TO_ROUTER_READY);
    fflush(stdout);
    common_log_resume(common_log_main());

    // 启动监控线程读取 stdin
    return std::thread([shutdown_handler]() {
        SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
        bool eof = false;
        while (true) {
            std::string line;
            if (!std::getline(std::cin, line)) {  // 阻塞读取
                eof = true;                       // EOF = 父进程意外退出
                break;
            }
            if (line.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
                SRV_INF("%s", "exit command received, exiting...\n");
                shutdown_handler(0);              // 优雅关闭
                break;
            }
        }
        if (eof) {
            SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
            exit(1);                              // 强制退出
        }
    });
}
```

这一段代码实现了子进程的“报到”与“监听”流程。启动时暂停日志系统，向 stdout 发送 `CMD_CHILD_TO_ROUTER_READY`（`cmd_child_to_router:ready`）通知父进程已就绪，恢复日志系统。随后返回一个监控线程，阻塞读取 `stdin`：收到 `EXIT` 命令则调用关闭处理器优雅退出；检测到 EOF（父进程意外退出/管道破裂）则强制 `exit(1)`。这就像列车出站前向调度台报“已就绪”，然后派专人盯着对讲机，听到“回库”指令按程序回库，若对讲机突然没声了（EOF）则判定调度中心失联，启动应急制动。

---

### 82.7.3 server_models::notify_router_sleeping_state - 睡眠/唤醒通知「报备休眠」

源码路径：`tools/server/server-models.cpp` 第755-762行。

```cpp
void server_models::notify_router_sleeping_state(bool is_sleeping) {
    common_log_pause(common_log_main());
    fflush(stdout);
    fprintf(stdout, "%s\n", is_sleeping ? CMD_CHILD_TO_ROUTER_SLEEP : CMD_CHILD_TO_ROUTER_READY);
    fflush(stdout);
    common_log_resume(common_log_main());
}
```

这一段代码实现了子进程向父进程报告睡眠/唤醒状态。暂停日志避免日志穿插干扰命令解析，发送 `CMD_CHILD_TO_ROUTER_SLEEP`（`cmd_child_to_router:sleep`）或 `CMD_CHILD_TO_ROUTER_READY`，恢复日志。父进程 `log_thread` 识别这两条命令分别更新状态为 `SLEEPING` 或 `LOADED`。这就像列车临时靠边停车时向调度台报备“休眠”，准备发车时再报“就绪”。

---

## 82.8 HTTP 代理转发与流式管道 —— 中转窗口与专用对讲机

### 82.8.1 pipe_t - 线程安全流式管道「专用对讲机」

生产者-消费者模式，支持 EOF 检测、主动取消、broken pipe 语义。源码路径：`tools/server/server-models.cpp` 第1058-1091行。

```cpp
template <typename T>
struct pipe_t {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};

    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // 干净 EOF
            }
            if (should_stop()) {           // 外部取消信号（如客户端断开）
                close_read();              // 通知写端管道破裂
                return false;              // 取消/读端不再存活
            }
            cv.wait_for(lk, poll_interval); // 500ms 轮询平衡响应性与 CPU
        }
    }

    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};
```

这一段代码实现了核心数据通道 `pipe_t`。`writer_closed`/`reader_closed` 原子标志标记端点关闭状态；`read` 接受 `should_stop` 回调实现取消：若回调返回真，主动 `close_read()` 通知写端并返回假；`write` 检测 `reader_closed` 实现 broken pipe 语义，防止向已关闭的管道写入；500ms 轮询间隔折中响应性与 CPU 占用。这就像列车与调度中心的专用对讲机：一边说话（write）、一边听（read），任意一方挂断（close）对方都能感知，且支持“强行挂断”取消操作。

---

### 82.8.2 server_http_proxy - 异步代理转发器「中转售票窗口」

构造时即发起异步请求，通过 `pipe_t` 回传响应头和流式 body。源码路径：`tools/server/server-models.cpp` 第1108-1182行。

```cpp
server_http_proxy(const std::string & method, const std::string & scheme,
        const std::string & host, int port,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        const std::function<bool()> should_stop,
        int32_t timeout_read, int32_t timeout_write) {
    auto cli = std::make_shared<httplib::ClientImpl>(host, port);
    auto pipe = std::make_shared<pipe_t<msg_t>>();
    if (scheme == "https") {
        cli.reset(new httplib::SSLClient(host, port));
    }
    this->status = 500;
    // next 回调：供外部流式拉取响应体
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next;
    };
    // 响应头处理：过滤跳转/连接控制头 + CORS 头
    httplib::ResponseHandler response_handler = [pipe, cli](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) continue;  // 过滤
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return pipe->write(std::move(msg));
    };
    // 流式 body 接收
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };
    // 构造请求
    httplib::Request req;
    req.method = method;
    req.path = path;
    for (const auto & [key, value] : headers) {
        if (key == "Accept-Encoding" || key == "Transfer-Encoding") continue; // 跳过编码头
        if (key == "Host" || key == "host") {
            bool is_default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
            req.set_header(key, is_default_port ? host : host + ":" + std::to_string(port));
        } else {
            req.set_header(key, value);
        }
    }
    req.body = body;
    req.response_handler = response_handler;
    req.content_receiver = content_receiver;
    // 启动分离线程发送请求
    this->thread = std::thread([cli, pipe, req]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            pipe->write({{}, 500, "", ""});
            pipe->write({{}, 0, "proxy error: " + err_str, ""});
        }
        pipe->close_write(); // 标记写端关闭
    });
    this->thread.detach();
    // 主线程阻塞读取首个 msg_t（含状态码和头部）
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            this->status = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        }
    }
}
```

这一段代码实现了 `server_http_proxy` 的构造函数。创建 HTTP 客户端和 `pipe_t`，注册 `response_handler`（处理响应头，过滤 `Server`、`Transfer-Encoding`、`Content-Length`、`Keep-Alive` 及所有 `Access-Control-*` 头）和 `content_receiver`（流式写入 body 片段）。启动分离线程发送请求，主线程阻塞读取首个 `msg_t`（含状态码和头部），后续调用者通过 `next()` 回调流式拉取 body。错误时写入错误消息并关闭管道。这就像中转售票窗口：买票请求（HTTP请求）转给后台车次窗口（子进程），车次窗口出票过程（流式响应）通过对讲机（pipe）实时传回售票窗口，售票窗口再交给乘客。

---

### 82.8.3 请求头过滤与大小写归一化「过滤干扰信号」

防止头部回环和 CORS 污染。源码路径：`tools/server/server-models.cpp` 第1093-1106行。

```cpp
static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" ||
        header_name == "keep-alive") {
        return true;
    }
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }
    return false;
}
```

这一段代码实现了头部过滤工具。`to_lower_copy` 保证头名大小写不敏感比较。`should_strip_proxy_header` 过滤跳转/连接控制头和所有 CORS 相关头，避免代理链中头部重复或冲突。这就像对讲机通话前过滤掉无关的背景噪音和干扰信号，只传递有效内容。

---

## 82.9 路由 REST 端点 —— 对外的「调度总台」

### 82.9.1 server_models_routes::init_routes - 路由注册分发「柜台分流」

以 Lambda 注册五个核心端点。源码路径：`tools/server/server-models.cpp` 第765-867行。

```cpp
void server_models_routes::init_routes() {
    // GET /props - 查询路由器属性或代理到模型
    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // 返回路由器自身属性
            auto res = std::make_unique<server_http_res>();
            res_ok(res, {
                {"role", "router"},
                {"max_instances", 4},
                {"model_alias", "llama-server"},
                {"model_path", "none"},
                {"default_generation_settings", {{"params", json{}}, {"n_ctx", 0}}},
                {"webui_settings", webui_settings},
            });
            return res;
        }
        return proxy_get(req); // 有 model 参数则代理
    };

    // GET 代理转发
    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, false); // GET 不更新 last_used
    };

    // POST 代理转发
    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, true); // POST 更新 last_used
    };

    // POST /models/load - 手动加载模型
    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (meta->is_running()) {
            res_err(res, format_error_response("model is already running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(meta->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    // GET /models - 列出所有模型状态
    this->get_router_models = [this](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        for (const auto & meta : all_models) {
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                preset_copy.unset_option("LLAMA_ARG_HOST");
                preset_copy.unset_option("LLAMA_ARG_PORT");
                preset_copy.unset_option("LLAMA_ARG_ALIAS");
                preset_copy.unset_option("LLAMA_ARG_TAGS");
                status["preset"] = preset_copy.to_ini(); // 导出预设快照
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"] = true;
            }
            models_json.push_back(json {
                {"id", meta.name},
                {"aliases", meta.aliases},
                {"tags", meta.tags},
                {"object", "model"},
                {"owned_by", "llamacpp"},
                {"created", t},
                {"status", status},
            });
        }
        res_ok(res, {{"data", models_json}, {"object", "list"}});
        return res;
    };
}
```

这一段代码注册了五个核心端点：`get_router_props` 查询路由器自身或代理到模型；`proxy_get`/`proxy_post` 透传请求，POST 更新 `last_used`；`post_router_models_load` 手动加载，校验存在性与运行状态；`get_router_models` 返回所有模型详细快照（含 preset 导出、失败标记）。这就像调度总台设立不同柜台：查询台、中转窗口、调度指令窗口、档案查询窗口，各司其职。

---

### 82.9.2 router_validate_model - 模型验证与自动加载决策「安检闸机」

前置检查：别名解析、存在性、运行状态、autoload 决策。源码路径：`tools/server/server-models.cpp` 第780-800行。

```cpp
static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    // 解析别名为规范模型名
    name = meta->name;
    if (models_autoload) {
        models.ensure_model_ready(name);  // 自动加载并等待
    } else {
        if (!meta->is_running()) {
            res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return false;
        }
    }
    return true;
}
```

这一段代码实现了路由层的前置验证逻辑。模型名为空报错；不存在报错；解析别名为规范名；若 `autoload=true` 调用 `ensure_model_ready` 自动加载并等待；否则要求模型已运行，否则报错。这就像安检闸机：查验车票（模型名）、核实身份（别名解析）、检查车次是否运行（状态），自动购票模式下（autoload）还能现场补票（加载）。

---

### 82.9.3 is_autoload - 解析自动加载参数「读取自动购票设置」

优先读取请求参数 `autoload`，缺省回退到服务器全局配置。源码路径：`tools/server/server-models.cpp` 第802-810行。

```cpp
static bool is_autoload(const common_params & params, const server_http_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}
```

这一段代码实现了自动加载策略的解析：请求显式传 `autoload=true/1` 则开启，否则回退到服务器全局配置 `params.models_autoload`。这就像买票时乘客可指定“自动补票”，不指定则按车站默认规则。

---

### 82.9.4 res_ok / res_err - 统一响应格式化「标准回单」

源码路径：`tools/server/server-models.cpp` 第768-778行。

```cpp
static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}
```

这一段代码统一了成功与错误响应的格式。成功固定 200 状态码，错误从 error_data 提取 code 字段（默认 500），body 统一包裹在 `error` 对象中。这就像所有柜台统一使用标准回单格式，方便客户端解析。

---

## 82.10 平台兼容与工具函数 —— 跨平台的「幕后保障组」

### 82.10.1 get_server_exec_path - 跨平台获取可执行路径「定位调度指挥车」

Windows 用 `GetModuleFileNameW`，macOS 用 `_NSGetExecutablePath` 两次调用，Linux 读 `/proc/self/exe`。源码路径：`tools/server/server-models.cpp` 第51-86行。

```cpp
static std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);
    if (_NSGetExecutablePath(small_path, &size) == 0) {
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}
```

这一段代码实现了跨平台获取可执行文件绝对路径。Windows 直接大缓冲区调用 API；macOS 需两次调用 `_NSGetExecutablePath`（首次获取所需缓冲区大小，二次分配足够空间再取路径），并尝试 `canonical` 解析符号链接；Linux 读取 `/proc/self/exe` 符号链接。所有平台最终返回 `std::filesystem::path` 统一类型。这就像在不同城市的火车站（操作系统）用不同方式找到调度指挥车的停放位置，最终都得到一个标准地址。

---

### 82.10.2 get_environment / wide_to_utf8 - 环境变量与编码转换「收集环境情报」

Windows 环境变量为宽字符块，需逐项转 UTF-8。源码路径：`tools/server/server-models.cpp` 第103-145行。

```cpp
#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) return {};
    const int len = static_cast<int>(std::wcslen(ws));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, ws, len, nullptr, 0, nullptr, nullptr);
    if (bytes == 0) return {};
    std::string utf8(bytes, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, len, utf8.data(), bytes, nullptr, nullptr);
    return utf8;
}
#endif

static std::vector<std::string> get_environment() {
    std::vector<std::string> env;
#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) return env;
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wide_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) return env;
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif
    return env;
}
```

这一段代码实现了跨平台获取环境变量。Windows 调用 `GetEnvironmentStringsW` 获取宽字符块，逐项 `wide_to_utf8` 转换后存入向量，最后释放块内存；POSIX 直接遍历 `environ` 指针数组。`wide_to_utf8` 两次调用 `WideCharToMultiByte`：首次计算所需字节数，二次实转换。这就像情报员在不同编码规范的档案室抄录情报，统一转译成 UTF-8 标准格式存档。

---

### 82.10.3 get_free_port - 空闲端口分配「动态分配站台号」

`bind(0)` 让内核自动分配，`getsockname` 取回端口，立即关闭释放。源码路径：`tools/server/server-models.cpp` 第383-417行。

```cpp
static int get_free_port() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return -1;
    typedef SOCKET native_socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define CLOSE_SOCKET(s) closesocket(s)
#else
    typedef int native_socket_t;
#define INVALID_SOCKET_VAL -1
#define CLOSE_SOCKET(s) close(s)
#endif

    native_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VAL) {
#ifdef _WIN32 WSACleanup(); #endif
        return -1;
    }

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(0);  // 端口 0 = 内核自动分配

    if (bind(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32 WSACleanup(); #endif
        return -1;
    }

#ifdef _WIN32
    int namelen = sizeof(serv_addr);
#else
    socklen_t namelen = sizeof(serv_addr);
#endif
    if (getsockname(sock, (struct sockaddr*)&serv_addr, &namelen) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32 WSACleanup(); #endif
        return -1;
    }

    int port = ntohs(serv_addr.sin_port);
    CLOSE_SOCKET(sock);
#ifdef _WIN32 WSACleanup(); #endif
    return port;
}
```

这一段代码通过 `bind(port=0)` 让内核自动分配空闲端口，`getsockname` 取回分配到的端口号，立即关闭套接字释放端口。宏屏蔽了 Windows/Linux Socket API 差异（`SOCKET` vs `int`、`closesocket` vs `close`、`socklen_t` vs `int`），Windows 还需配对 `WSAStartup`/`WSACleanup`。注意：高并发下端口可能被其他进程抢占（TIME_WAIT 复用窗口），但概率极低。这就像调度员向车站系统申请一个临时站台号，系统自动分配一个空闲的，调度员记下后立即释放申请，实际使用时由列车占用。

---

### 82.10.4 to_char_ptr_array - vector<string> 转 argv 数组「转换指令单格式」

源码路径：`tools/server/server-models.cpp` 第427-435行。

```cpp
static std::vector<char *> to_char_ptr_array(const std::vector<std::string> & vec) {
    std::vector<char *> result;
    result.reserve(vec.size() + 1);
    for (const auto & s : vec) {
        result.push_back(const_cast<char*>(s.c_str()));
    }
    result.push_back(nullptr);  // 以 nullptr 结尾
    return result;
}
```

这一段代码将 `vector<string>` 转为 `argv` 兼容的 `char*[]` 数组（以 nullptr 结尾）。预留容量避免扩容导致指针失效，`const_cast` 去除 const 资格满足 C API 要求，原字符串生命周期由调用方保证。这就像将整理好的发车指令单（字符串向量）转印成调度系统要求的标准格式（C 风格字符串数组）。

---

### 82.10.5 unset_reserved_args - 清理保留参数「防配置污染」

移除路由器级参数（SSL、API Key、模型目录、并发限制、预设路径、autoload 开关），`unset_model_args=true` 时额外清理模型级参数。源码路径：`tools/server/server-models.cpp` 第88-101行。

```cpp
static void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_HF_REPO");
    }
}
```

这一段代码清理预设中不应传递给子进程的参数。路由器级参数（SSL、API Key、模型目录、并发限制、预设路径、autoload 开关）仅在路由器层生效，传给子进程会导致配置污染或冲突；`unset_model_args=true` 时额外清理模型级参数（model、mmproj、hf_repo），用于构建基础预设时剥离模型特定配置。这就像发车前收回调度员专用的调度指令单，只把车辆自己的配置单交给车长。

---

### 82.10.6 to_lower_copy - 字符串转小写「统一大小写」

源码路径：`tools/server/server-models.cpp` 第1093-1097行。

```cpp
static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}
```

这一段代码实现了字符串转小写副本，用于 HTTP 头名大小写不敏感比较。预分配结果字符串避免扩容，使用 `unsigned char` 避免 `tolower` 未定义行为。这就像将所有信号灯指令统一转为小写，避免大小写不一致导致识别错误。

---

### 82.10.7 should_strip_proxy_header - 代理响应头过滤判断「过滤干扰头」

源码路径：`tools/server/server-models.cpp` 第1099-1106行。

```cpp
static bool should_strip_proxy_header(const std::string & header_name) {
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" ||
        header_name == "keep-alive") {
        return true;
    }
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }
    return false;
}
```

这一段代码判断是否应过滤掉代理响应中的特定 HTTP 头。过滤 `Server`、`Transfer-Encoding`、`Content-Length`、`Keep-Alive` 等跳转/连接控制头，以及所有 `Access-Control-*` CORS 相关头，防止代理链中头部重复或冲突。这就像对讲机转发时自动屏蔽调度中心内部的管理指令，只转发业务内容。

---

## 82.11 设计中的取舍 —— 为什么这样设计？

**为什么父子进程通信用 stdio 而非共享内存/gRPC？**
控制面消息极少（仅 READY/SLEEP/EXIT），stdio 完全够用，符合“零依赖”理念。共享内存需要平台特定 API 增加复杂度，gRPC 需引入 Protobuf/grpc 库增大体积，而 stdio 天然跨平台、调试时直接看日志可见命令流、实现极简。

**为什么 LRU 用 `ggml_time_ms()` 而非 `system_clock`？**
`ggml_time_ms()` 基于单调时钟，不受 NTP 时间跳变、用户修改系统时间影响。系统时钟若回拨会导致“最久未用”判断失效，单调时钟保证 `last_used` 单调递增，LRU 判定永远正确。

**为什么 `pipe_t::read` 用 500ms 轮询而非无限等待？**
无限等待无法响应 `should_stop` 取消信号（客户端断开、服务器关闭）。500ms 折中：不频繁唤醒消耗 CPU，又能在半秒内感知取消。理想方案是 eventfd/Windows 事件对象实现真事件驱动，但会增加平台相关代码。

**预设合并优先级为何设计成“缓存 < 本地 < INI < CLI”？**
符合“显式优于隐式”、“后指定优先”直觉：用户手写 INI 覆盖自动缓存，命令行参数拥有最高话语权。`base_preset` 最后合并确保全局参数（线程数、上下文长度）统一生效，模型特定参数可被模型级预设覆盖。

**线程安全设计：细粒度锁还是大锁？**
采用**单一大锁 `mutex`** 保护整个 `mapping`，简化正确性证明。读操作（`get_meta`、`has_model`、`get_all_meta`）虽频繁但极快（仅拷贝元数据），大锁开销可接受。写操作（`load`、`unload`、`update_meta`）持锁时间短，仅做映射表修改，耗时操作（spawn、等待）在锁外完成。

**子进程隔离的优势是什么？**
故障隔离：子进程崩溃不影响路由器及其他模型；资源隔离：各模型独立内存空间、GPU 显存、线程池；版本隔离：不同模型可用不同编译参数/依赖版本（理论上）；安全隔离：子进程可降权运行，减小攻击面。

**流式管道 vs 缓冲区的考量？**
`pipe_t` 实现**生产者-消费者流式传输**，无需缓存完整响应体。支持 SSE/流式推理场景：子进程产生一个 token，代理立即转发一个 chunk。内存占用恒定（仅队列缓冲），适合长流式响应。

**跨平台宏屏蔽策略如何工作？**
Socket API 差异（`SOCKET` vs `int`、`closesocket` vs `close`、`socklen_t` vs `int`）用宏 `#define` 屏蔽。Windows 专用初始化/清理（`WSAStartup`/`WSACleanup`）配对调用。宽字符/UTF-8 转换封装在 `wide_to_utf8`，业务代码只处理 UTF-8。

---

## 82.12 动手练习

### 82.12.1 练习 1：LRU 淘汰策略深度分析
阅读 `unload_lru()` 与 `load()` 代码，回答：
1. `unload_lru()` 如何找到「最久未使用」模型？为何用 `ggml_time_ms()`？
2. `load()` 为何在锁内二次检查 `count_active >= models_max`？防止什么并发问题？
3. 两线程同时 `load()` 不同模型且已达 `models_max`，描述时序与结果。

### 82.12.2 练习 2：子进程生命周期与双线程监控
阅读 `load()` 中 `log_thread` 与 `stopping_thread`，回答：
1. `log_thread` 如何识别 READY/SLEEP 信号？通过什么通道发送？
2. `stopping_thread` 退出条件有哪些？为何用 `wait_for` 循环而非一次性等待？
3. 子进程崩溃时，`stopping_thread` 如何检测并处理？`subprocess_alive` 作用？

### 82.12.3 练习 3：HTTP代理与流式管道实现
阅读 `server_http_proxy` 与 `pipe_t`，回答：
1. `pipe_t::read()` 如何实现取消机制？`should_stop` 何时被调用？
2. `should_strip_proxy_header` 过滤哪些头？为何过滤 CORS 头？
3. 500ms `poll_interval` 对慢响应有何影响？如何优化为事件驱动？

### 82.12.4 练习 4：预设合并与优先级
阅读 `load_models()` 与 `common/preset.h` 的 `cascade`/`merge`，回答：
1. 三源（cached、local、custom INI）合并优先级？同名模型 cached 与 local 谁生效？
2. `base_preset` 何时注入？优先级相对模型特定预设高还是低？
3. `unset_reserved_args` 清理哪些参数？为何不能传给子进程？

### 82.12.5 练习 5：跨平台兼容层实践
阅读 `get_server_exec_path`、`get_environment`、`get_free_port`，回答：
1. Windows/Linux 获取可执行路径本质不同？macOS 为何两次调用 `_NSGetExecutablePath`？
2. Windows 环境变量为何宽字符？`wide_to_utf8` 失败返回什么？
3. `get_free_port` 端口何时真正释放？高并发是否可能冲突？

### 82.12.6 练习 6：模型路由与自动加载
阅读 `router_validate_model` 与 `is_autoload`，回答：
1. 如何处理模型别名？为何解析为规范名？
2. `is_autoload` 默认行为？请求参数如何覆盖全局配置？
3. `autoload=false` 且模型未加载时，返回什么错误？HTTP 状态码？

### 82.12.7 练习 7：线程同步与条件变量实战
阅读所有 `mutex`/`condition_variable` 使用位置，回答：
1. `update_meta` 为何 `cv.notify_all()`？无通知会怎样？
2. `wait_until_loading_finished` 等待条件？为何 lambda 捕获 `this` 和 `name`？
3. `ensure_model_ready` 加载失败若不抛异常而返回，对上层有何影响？

---

## 82.13 本章小结

本章深入剖析了 llama-server 的模型调度核心——`server_models` 类及其协作组件。首先我们理解了模型元数据 `server_model_meta` 与四态状态机 `server_model_status` 的设计，它们是调度决策的基础。其次我们详细追踪了 `load_models()` 的三源预设合并策略（缓存→本地→INI→CLI），以及 `add_model()` 的多层冲突检测机制。接着我们重点分析了 `load()` 的“三步走”加载流程：LRU 预检、端口分配、子进程 Spawn，以及双监控线程（日志转发+状态识别、停止监控+超时强杀）的精妙协作。然后我们探讨了卸载流程的优雅退出信号机制、LRU 淘汰的完整实现、就绪等待与自动加载的阻塞同步原语。随后我们剖析了父子进程通信协议（READY/SLEEP/EXIT）、HTTP 代理转发器 `server_http_proxy` 的异步流式设计、核心数据通道 `pipe_t` 的生产者-消费者实现与取消机制。最后我们梳理了路由 REST 端点的注册与分发逻辑、模型验证与自动加载决策、跨平台兼容层（可执行路径、环境变量、端口分配）的封装细节，以及预设合并优先级、保留参数清理等工程实用技巧。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| `server_models` | 多模型调度中枢，管理模型实例的加载、卸载与生命周期 |
| `server_model_meta` | 模型元数据档案，记录状态、端口、预设、最近使用时间 |
| `server_model_status` | 状态枚举四态：UNLOADED/LOADING/LOADED/SLEEPING |
| `server_model_status_to_string()` | 状态枚举与字符串的双向转换工具 |
| `load_models()` | 三源合并策略：缓存预设 → 本地目录 → 自定义INI，最后注入base_preset |
| `load()` / `unload()` | 模型生命周期：子进程spawn + 双线程监控（日志/停止） |
| `unload_lru()` | LRU淘汰算法，超出models_max时自动清理最久未用实例 |
| `ensure_model_ready()` | 阻塞式等待模型就绪，自动触发加载（autoload） |
| `pipe_t` | 线程安全流式管道，支持读写分离与取消机制 |
| `server_http_proxy` | HTTP代理转发器，过滤请求头并流式转发到子进程 |
| `router_validate_model()` | 路由前置验证，解析模型名、处理alias、检查autoload |
| `get_free_port()` | 跨平台端口分配（bind(0)+getsockname） |
| `is_child_server()` | 通过环境变量LLAMA_SERVER_ROUTER_PORT识别子进程身份 |
| `setup_child_server()` | 子进程初始化：发送READY信号，监控stdin退出命令 |
| `notify_router_sleeping_state()` | 子进程通知父进程进入/退出睡眠状态 |
| `update_args()` | 预设渲染：注入host/port/alias，生成最终启动参数 |
| `stopping_models` | 停止信号集合，协调多线程安全退出 |
| `is_autoload()` | 解析请求参数，判断是否自动加载未运行模型 |
| `CMD_*` 命令常量 | 父子进程间标准通信协议（READY/SLEEP/EXIT） |

下一章中，我们将学习任务参数与结果系统——推理作业的“设计图纸与成品”，理解一次推理请求在服务端内部的完整数据生命周期，包括任务参数解析、结果多格式序列化与提示缓存机制。



