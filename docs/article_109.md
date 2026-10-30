

# 第109章 性能时光机：基于SQLite/Git的跨提交性能对比机制

想象你是一位数据中心的质量控制总监，想象你有一台可以穿越代码历史的时光机，能够对比不同历史时期的生产效率报表。scripts/compare-llama-bench.py就是这台时光机的核心控制面板：它通过抽象基类统一基准数据处理接口，支持llama-bench与test-backend-ops双模式，同时集成GitPython实现基于Git历史的自动父提交查找与精细哈希管理。内置的_builds_init构建初始化与_check_keys数据完整性校验机制，强制要求核心字段如build_commit、test_time、avg_ts等存在。通过抽象方法builds_timestamp与get_rows定义数据访问契约，子类需实现具体存储逻辑。辅助方法get_commit_name与get_all_parent_hexsha8s支持人类可读名称与完整提交链查询，为跨提交性能对比提供了强大的历史追溯能力。

就像质量控制总监可以查看某个生产线在特定日期的产量报告，这个基类让我们能够追溯到任何代码提交对应的性能数据。

---

### 109.0.1 LlamaBenchData.__init__()：基类初始化与Git仓库绑定

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchData`（291-300行）

```python
def __init__(self, tool: str = "llama-bench"):
    self.tool = tool
    try:
        self.repo = git.Repo(".", search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        self.repo = None

    # Set schema-specific properties based on tool
    if self.tool == "llama-bench":
        self.check_keys = set(LLAMA_BENCH_KEY_PROPERTIES + ["build_commit", "test_time", "avg_ts"])
    elif self.tool == "test-backend-ops":
        self.check_keys = set(TEST_BACKEND_OPS_KEY_PROPERTIES + ["build_commit", "test_time"])
    else:
        assert False
```

**解释**：这段代码定义了性能数据处理的基类，首先绑定Git仓库以启用历史追溯功能，然后根据工具类型（llama-bench或test-backend-ops）设置必需的数据字段。通过self.repo属性，后续方法可以访问Git历史；通过self.check_keys集合，确保数据完整性。这种设计使得性能对比不仅能比较数值，还能定位到具体的代码提交，实现真正的“性能时光机”功能。就像质量控制总监可以查看某个生产线在特定日期的产量报告，这个基类让我们能够追溯到任何代码提交对应的性能数据。

---

### 109.0.2 LlamaBenchData._builds_init()：初始化构建长度参数

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchData`（302-303行）

```python
def _builds_init(self):
    self.build_len = self.build_len_min
```

**解释**：该方法初始化构建长度参数，将实例变量self.build_len设置为self.build_len_min。在子类中（如LlamaBenchDataSQLite3），此值会被数据库中的最小提交哈希长度覆盖，以确保哈希截断的一致性。就像质量控制总监的时光机需要先校准时间精度，这个方法为后续的哈希比对设定了基准长度。

---

### 109.0.3 LlamaBenchData._check_keys()：数据完整性校验

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchData`（305-308行）

```python
def _check_keys(self, keys: set) -> Optional[set]:
    """Private helper method that checks against required data keys and returns missing ones."""
    if not keys >= self.check_keys:
        return self.check_keys - keys
    return None
```

**解释**：此私有方法检查数据是否包含所需的关键字段，返回缺失字段集合。如果提供的keys集合是self.check_keys的超集（即包含所有必需字段），则返回None表示验证通过；否则返回缺失的字段集合。就像质量控制总监的检查清单，这个方法确保性能数据包含所有必需字段才能进行有效对比。

---

### 109.0.4 LlamaBenchData.find_parent_in_data()：Git历史父提交追溯

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchData.find_parent_in_data()`（310-322行）

```python
def find_parent_in_data(self, commit: git.Commit) -> Optional[str]:
    """Helper method to find the most recent parent measured in number of commits for which there is data."""
    heap: list[tuple[int, git.Commit]] = [(0, commit)]
    seen_hexsha8 = set()
    while heap:
        depth, current_commit = heapq.heappop(heap)
        current_hexsha8 = commit.hexsha[:self.build_len]
        if current_hexsha8 in self.builds:
            return current_hexsha8
        for parent in commit.parents:
            parent_hexsha8 = parent.hexsha[:self.build_len]
            if parent_hexsha8 not in seen_hexsha8:
                seen_hexsha8.add(parent_hexsha8)
                heapq.heappush(heap, (depth + 1, parent))
    return None
```

**解释**：此方法通过广度优先搜索（使用最小堆）在Git提交历史中查找最近的有数据记录的父提交。它从当前提交开始，逐级向父提交搜索，使用截断的8位哈希（hexsha8）作为比较键。堆确保我们总是先探索更近的提交（较小的depth），而seen_hexsha8集合防止重复访问同一提交。这种算法高效地解决了“在历史记录中找到最近的有效性能基线”问题，避免了线性扫描的低效率，尤其在有数百个提交的仓库中表现尤为重要。就像质量控制总监的时光机能够智能地跳过没有生产数据的历史时期，直接找到最近的有效基线进行对比。

**设计取舍分析**：
- 为什么不用逐行处理方案？逐行处理需要在应用层完成分组和聚合，对于大型基准数据集会消耗大量内存和CPU时间。
- 这种设计的trade-off是什么？SQL查询虽然高效，但增加了对特定数据库功能的依赖，降低了与非SQLite数据源的完全统一性。然而考虑到SQLite是该项目的标准依赖，这种trade-off是值得的。

---

### 109.0.5 LlamaBenchData.get_all_parent_hexsha8s()：递归获取提交的所有父提交哈希值

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchData.get_all_parent_hexsha8s()`（324-337行）

```python
def get_all_parent_hexsha8s(self, commit: git.Commit) -> Sequence[str]:
    """Helper method to recursively get hexsha8 values for all parents of a commit."""
    unvisited = [commit]
    visited   = []
    while unvisited:
        current_commit = unvisited.pop(0)
        visited.append(current_commit.hexsha[:self.build_len])
        for parent in current_commit.parents:
            if parent.hexsha[:self.build_len] not in visited:
                unvisited.append(parent)
    return visited
```

**解释**：此方法递归获取提交的所有父提交的哈希值。它使用广度优先搜索遍历Git提交树，收集所有祖先提交的截断哈希（hexsha8）。与find_parent_in_data不同，此方法返回完整的父提交链而不仅是最近的一个。就像质量控制总监的时光机不仅能找到最近的基线，还能完整回溯整条生产线的历史。

---

### 109.0.6 LlamaBenchData.get_commit_name()：根据提交哈希查找可读的提交名称

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchData.get_commit_name()`（339-347行）

```python
def get_commit_name(self, hexsha8: str) -> str:
    """Helper method to find a human-readable name for a commit if possible."""
    if self.repo is None:
        return hexsha8
    for h in self.repo.heads:
        if h.commit.hexsha[:self.build_len] == hexsha8:
            return h.name
    for t in self.repo.tags:
        if t.commit.hexsha[:self.build_len] == hexsha8:
            return t.name
    return hexsha8
```

**解释**：此方法根据提交哈希查找可读的提交名称。如果Git仓库可用，它会遍历所有分支和标签，寻找匹配的提交；若找到则返回对应的分支名或标签名，否则返回原始哈希。就像质量控制总监的时光机不仅显示日期编号，还能对应到具体的生产批次名称。

---

### 109.0.7 LlamaBenchData.get_commit_hexsha8()：根据可读名称查找提交哈希

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchData.get_commit_hexsha8()`（349-363行）

```python
def get_commit_hexsha8(self, name: str) -> Optional[str]:
    """Helper method to search for a commit given a human-readable name."""
    if self.repo is None:
        return None
    for h in self.repo.heads:
        if h.name == name:
            return h.commit.hexsha[:self.build_len]
    for t in self.repo.tags:
        if t.name == name:
            return t.commit.hexsha[:self.build_len]
    for remote in self.repo.remotes:
        for ref in remote.refs:
            if ref.name == name or ref.remote_head == name:
                return ref.commit.hexsha[:self.build_len]
    for c in self.repo.iter_commits("--all"):
        if c.hexsha[:self.build_len] == name[:self.build_len]:
            return c.hexsha[:self.build_len]
    return None
```

**解释**：此方法根据可读名称查找提交哈希。它搜索本地分支、标签、远程引用以及所有提交，以找到匹配名称的提交并返回其截断哈希。就像质量控制总监的时光机支持通过生产批次名称快速定位到具体的历史时间点。

---

### 109.0.8 LlamaBenchData.builds_timestamp()：抽象方法，按时间戳排序获取构建信息

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchData.builds_timestamp()`（365-367行）

```python
def builds_timestamp(self, reverse: bool = False) -> Union[Iterator[tuple], Sequence[tuple]]:
    """Helper method that gets rows of (build_commit, test_time) sorted by the latter."""
    return []
```

**解释**：此抽象方法定义了按时间戳排序获取构建信息的接口，子类需实现具体存储逻辑。它返回(build_commit, test_time)元组的序列，按test_time排序。就像质量控制总监的时光机需要统一的时间查询接口，无论是从数据库还是日志文件中提取时间戳。

---

### 109.0.9 LlamaBenchData.get_rows()：数据查询抽象接口

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchData.get_rows()`（369-378行）

```python
def get_rows(self, properties: list[str], hexsha8_baseline: str, hexsha8_compare: str) -> Sequence[tuple]:
    """
    Helper method that gets table rows for some list of properties.
    Rows are created by combining those where all provided properties are equal.
    The resulting rows are then grouped by the provided properties and the t/s values are averaged.
    The returned rows are unique in terms of property combinations.
    """
    return []
```

**解释**：此方法定义了数据访问契约，子类需实现具体存储逻辑。它描述了数据查询的核心逻辑：根据提供的属性列表过滤数据，确保只比较相同配置的运行（如相同的模型、提示长度等），然后将匹配的基准提交和目标提交的数据关联起来，按属性分组，最后计算平均tokens/s值。这种抽象设计让不同的数据源实现（SQLite3、JSONL等）都能遵循相同的查询模式，而主要逻辑保持不变。就像质量控制总监的时光机有统一的操作界面，无论是查询SQL数据库还是检查手工记录本，都能得到标准化的性能对比结果。

**设计取舍分析**：
- 为什么不用逐行处理方案？逐行处理需要在应用层完成分组和聚合，对于大型基准数据集会消耗大量内存和CPU时间。
- 这种设计的trade-off是什么？SQL查询虽然高效，但增加了对特定数据库功能的依赖，降低了与非SQLite数据源的完全统一性。然而考虑到SQLite是该项目的标准依赖，这种trade-off是值得的。

---

### 109.0.10 LlamaBenchDataSQLite3.__init__()：创建 SQLite 内存数据库连接并创建对应工具的表结构

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataSQLite3.__init__()`（401-417行）

```python
def __init__(self, tool: str = "llama-bench"):
    super().__init__(tool)
    if self.connection is None:
        self.connection = sqlite3.connect(":memory:")
        self.cursor = self.connection.cursor()

        # Set table name and schema based on tool
        if self.tool == "llama-bench":
            self.table_name = "llama_bench"
            db_fields = LLAMA_BENCH_DB_FIELDS
            db_types = LLAMA_BENCH_DB_TYPES
        elif self.tool == "test-backend-ops":
            self.table_name = "test_backend_ops"
            db_fields = TEST_BACKEND_OPS_DB_FIELDS
            db_types = TEST_BACKEND_OPS_DB_TYPES
        else:
            assert False

        self.cursor.execute(f"CREATE TABLE {self.table_name}({', '.join(' '.join(x) for x in zip(db_fields, db_types))});")
```

**解释**：此方法创建 SQLite 内存数据库连接并根据工具类型创建对应的表结构。它首先调用父类初始化，然后在内存中创建SQLite数据库连接，根据工具类型设置表名和字段结构，最后执行建表语句。就像质量控制总监的时光机内置了可重写的数据缓存，能够快速加载和查询历史性能数据。

---

### 109.0.11 LlamaBenchDataSQLite3._builds_init()：从数据库读取构建列表，统一提交哈希长度

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataSQLite3._builds_init()`（419-430行）

```python
def _builds_init(self):
    if self.connection:
        self.build_len_min = self.cursor.execute(f"SELECT MIN(LENGTH(build_commit)) from {self.table_name};").fetchone()[0]
        self.build_len_max = self.cursor.execute(f"SELECT MAX(LENGTH(build_commit)) from {self.table_name};").fetchone()[0]

        if self.build_len_min != self.build_len_max:
            logger.warning("Data contains commit hashes of differing lengths. It's possible that the wrong commits will be compared. "
                           "Try purging the the database of old commits.")
            self.cursor.execute(f"UPDATE {self.table_name} SET build_commit = SUBSTRING(build_commit, 1, {self.build_len_min});")

        builds = self.cursor.execute(f"SELECT DISTINCT build_commit FROM {self.table_name};").fetchall()
        self.builds = list(map(lambda b: b[0], builds))  # list[tuple[str]] -> list[str]
    super()._builds_init()
```

**解释**：此方法从数据库读取构建列表，统一提交哈希长度。它首先查询数据库中build_commit字段的最小和最大长度，若不一致则警告并将所有哈希截断到最小长度；然后获取所有唯一的构建提交并初始化self.builds列表；最后调用父类方法完成初始化。就像质量控制总监的时光机在加载历史数据前会先统一时间格式，确保所有记录使用相同的精度进行比较。

---

### 109.0.12 LlamaBenchDataSQLite3.builds_timestamp()：从数据库获取按时间戳排序的构建列表

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataSQLite3.builds_timestamp()`（432-435行）

```python
def builds_timestamp(self, reverse: bool = False) -> Union[Iterator[tuple], Sequence[tuple]]:
    data = self.cursor.execute(
        f"SELECT build_commit, test_time FROM {self.table_name} ORDER BY test_time;").fetchall()
    return reversed(data) if reverse else data
```

**解释**：此方法从数据库获取按时间戳排序的构建列表。它查询表中的build_commit和test_time字段，按test_time排序后返回结果；如果reverse为True，则反转顺序。就像质量控制总监的时光机能够正序或倒序浏览历史生产报告，便于分析性能趋势。

---

### 109.0.13 LlamaBenchDataSQLite3.get_rows()：根据工具类型分派到对应的行数据查询方法

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataSQLite3.get_rows()`（437-443行）

```python
def get_rows(self, properties: list[str], hexsha8_baseline: str, hexsha8_compare: str) -> Sequence[tuple]:
    if self.tool == "llama-bench":
        return self._get_rows_llama_bench(properties, hexsha8_baseline, hexsha8_compare)
    elif self.tool == "test-backend-ops":
        return self._get_rows_test_backend_ops(properties, hexsha8_baseline, hexsha8_compare)
    else:
        assert False
```

**解释**：此方法根据工具类型分派到对应的行数据查询方法。它像一个路由器，根据self.tool的值将查询请求转发给llama-bench或test-backend-ops专用的查询实现。就像质量控制总监的时光机有不同的查询面板针对不同的生产线类型，但操作逻辑保持一致。

---

### 109.0.14 LlamaBenchDataSQLite3._get_rows_llama_bench()：llama-bench SQL聚合查询

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataSQLite3._get_rows_llama_bench()`（445-452行）

```python
def _get_rows_llama_bench(self, properties: list[str], hexsha8_baseline: str, hexsha8_compare: str) -> Sequence[tuple]:
    select_string = ", ".join(
        [f"tb.{p}" for p in properties] + ["tb.n_prompt", "tb.n_gen", "tb.n_depth", "AVG(tb.avg_ts)", "AVG(tc.avg_ts)"]
    )
    equal_string = " AND ".join(
        [f"tb.{p} = tc.{p}" for p in LLAMA_BENCH_KEY_PROPERTIES] + [
            f"tb.build_commit = '{hexsha8_baseline}'", f"tc.build_commit = '{hexsha8_compare}'"
        ]
    )
    group_order_string = ", ".join([f"tb.{p}" for p in properties] + ["tb.n_gen", "tb.n_prompt", "tb.n_depth"])
    query = (f"SELECT {select_string} FROM {self.table_name} tb JOIN {self.table_name} tc ON {equal_string} "
             f"GROUP BY {group_order_string} ORDER BY {group_order_string};")
    return self.cursor.execute(query).fetchall()
```

**解释**：此方法实现了llama-bench性能数据的核心查询逻辑。它通过自我连接（JOIN）将基准提交和目标提交的数据关联起来，根据关键属性进行分组，然后计算平均tokens/s值。查询精心设计了equal_string来确保只比较相同配置的运行（如相同的模型、提示长度等），而group_order_string则保证结果有序。最终返回的元组包含了所有对比属性、测试名称以及基准和目标的性能值，为后续的speedup计算提供了原始数据。这种SQL聚合方法相比逐行处理，能够高效处理大规模基准数据集。就像质量控制总监的时光机内置了强大的数据库查询引擎，能够快速从海量历史数据中提取出可比较的性能指标。

**设计取舍分析**：
- 为什么不用逐行处理方案？逐行处理需要在应用层完成分组和聚合，对于大型基准数据集会消耗大量内存和CPU时间。
- 这种设计的trade-off是什么？SQL查询虽然高效，但增加了对特定数据库功能的依赖，降低了与非SQLite数据源的完全统一性。然而考虑到SQLite是该项目的标准依赖，这种trade-off是值得的。

---

### 109.0.15 LlamaBenchDataSQLite3._get_rows_test_backend_ops()：test-backend-ops SQL聚合查询

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataSQLite3._get_rows_test_backend_ops()`（454-464行）

```python
def _get_rows_test_backend_ops(self, properties: list[str], hexsha8_baseline: str, hexsha8_compare: str) -> Sequence[tuple]:
    # For test-backend-ops, we compare FLOPS and bandwidth metrics (prioritizing FLOPS over bandwidth)
    select_string = ", ".join(
        [f"tb.{p}" for p in properties] + [
            "AVG(tb.flops)", "AVG(tc.flops)",
            "AVG(tb.bandwidth_gb_s)", "AVG(tc.bandwidth_gb_s)"
        ]
    )
    equal_string = " AND ".join(
        [f"tb.{p} = tc.{p}" for p in TEST_BACKEND_OPS_KEY_PROPERTIES] + [
            f"tb.build_commit = '{hexsha8_baseline}'", f"tc.build_commit = '{hexsha8_compare}'",
            "tb.supported = 1", "tc.supported = 1", "tb.passed = 1", "tc.passed = 1"
        ]  # Only compare successful tests
    )
    group_order_string = ", ".join([f"tb.{p}" for p in properties])
    query = (f"SELECT {select_string} FROM {self.table_name} tb JOIN {self.table_name} tc ON {equal_string} "
             f"GROUP BY {group_order_string} ORDER BY {group_order_string};")
    return self.cursor.execute(query).fetchall()
```

**解释**：此方法实现了test-backend-ops性能数据的核心查询逻辑。它通过自我连接（JOIN）将基准提交和目标提交的数据关联起来，根据关键属性进行分组，然后计算平均FLOPS和带宽值。查询特别添加了supported和passed字段的过滤条件，确保只比较成功的测试。最终返回的元组包含了所有对比属性以及基准和目标的FLOPS和带宽性能值。就像质量控制总监的时光机为不同的生产线（如组装线vs测试线）提供了专门的性能指标查询，同时确保只比较正常运行的数据。

---

### 109.0.16 LlamaBenchDataSQLite3File.__init__()：连接 SQLite 文件数据库，检测表结构并选择正确的工具类型

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataSQLite3File.__init__()`（470-499行）

```python
def __init__(self, data_file: str, tool: Any):
    self.connection = sqlite3.connect(data_file)
    self.cursor = self.connection.cursor()

    # Check which table exists in the database
    tables = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    table_names = [table[0] for table in tables]

    # Tool selection logic
    if tool is None:
        if "llama_bench" in table_names:
            self.table_name = "llama_bench"
            tool = "llama-bench"
        elif "test_backend_ops" in table_names:
            self.table_name = "test_backend_ops"
            tool = "test-backend-ops"
        else:
            raise RuntimeError(f"No suitable table found in database. Available tables: {table_names}")
    elif tool == "llama-bench":
        if "llama_bench" in table_names:
            self.table_name = "llama_bench"
            tool = "llama-bench"
        else:
            raise RuntimeError(f"Table 'test' not found for tool 'llama-bench'. Available tables: {table_names}")
    elif tool == "test-backend-ops":
        if "test_backend_ops" in table_names:
            self.table_name = "test_backend_ops"
            tool = "test-backend-ops"
        else:
            raise RuntimeError(f"Table 'test_backend_ops' not found for tool 'test-backend-ops'. Available tables: {table_names}")
    else:
        raise RuntimeError(f"Unknown tool: {tool}")

    super().__init__(tool)
    self._builds_init()
```

**解释**：此方法连接 SQLite 文件数据库，检测表结构并选择正确的工具类型。它首先连接到指定的SQLite文件，检查数据库中存在哪些表，然后根据表名或用户提供的tool参数确定使用llama-bench还是test-backend-ops模式。就像质量控制总监的时光机能够自动识别插入的历史数据卡片属于哪种生产线类型，并加载相应的查询模板。

---

### 109.0.17 LlamaBenchDataSQLite3File.valid_format()：检查 SQLite 文件是否有效

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataSQLite3File.valid_format()`（501-513行）

```python
@staticmethod
def valid_format(data_file: str) -> bool:
    connection = sqlite3.connect(data_file)
    cursor = connection.cursor()

    try:
        if cursor.execute("PRAGMA schema_version;").fetchone()[0] == 0:
            raise sqlite3.DatabaseError("The provided input file does not exist or is empty.")
    except sqlite3.DatabaseError as e:
        logger.debug(f'"{data_file}" is not a valid SQLite3 file.', exc_info=e)
        cursor = None

    connection.close()
    return True if cursor else False
```

**解释**：此静态方法检查 SQLite 文件是否有效。它尝试连接数据库并检查schema_version；如果版本为0则认为文件无效或空；捕获异常后记录调试信息并返回False。就像质量控制总监在使用时光机前会先检查历史数据卡片是否完整且可读，防止因损坏数据导致的误判。

---

### 109.0.18 LlamaBenchDataJSONL.__init__()：读取 JSONL 文件，解析每一行 JSON 数据并存入数据库

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataJSONL.__init__()`（519-535行）

```python
def __init__(self, data_file: str, tool: str = "llama-bench"):
    super().__init__(tool)

    # Get the appropriate field list based on tool
    db_fields = LLAMA_BENCH_DB_FIELDS if tool == "llama-bench" else TEST_BACKEND_OPS_DB_FIELDS

    with open(data_file, "r", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            parsed = json.loads(line)

            for k in parsed.keys() - set(db_fields):
                del parsed[k]

            if (missing_keys := self._check_keys(parsed.keys())):
                raise RuntimeError(f"Missing required data key(s) at line {i + 1}: {', '.join(missing_keys)}")

            self.cursor.execute(f"INSERT INTO {self.table_name}({', '.join(parsed.keys())}) VALUES({', '.join('?' * len(parsed))});", tuple(parsed.values()))

    self._builds_init()
```

**解释**：此方法读取 JSONL 文件，解析每一行 JSON 数据并存入数据库。它逐行读取文件，解析JSON对象，移除不在字段列表中的键，验证必需字段存在后插入数据库。就像质量控制总监的时光机能够将手写的生产日志逐条录入系统，自动过滤无效字段并确保数据完整性。

---

### 109.0.19 LlamaBenchDataJSONL.valid_format()：检查 JSONL 文件格式是否有效

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataJSONL.valid_format()`（537-547行）

```python
@staticmethod
def valid_format(data_file: str) -> bool:
    try:
        with open(data_file, "r", encoding="utf-8") as fp:
            for line in fp:
                json.loads(line)
                break
    except Exception as e:
        logger.debug(f'"{data_file}" is not a valid JSONL file.', exc_info=e)
        return False

    return True
```

**解释**：此静态方法检查 JSONL 文件格式是否有效。它尝试读取文件的第一行并解析为JSON；如果成功则返回True，否则捕获异常并返回False。就像质量控制总监在使用时光机前会快速检查历史数据卡片的第一条记录是否为可识别格式，避免浪费时间处理损坏的文件。

---

### 109.0.20 LlamaBenchDataJSON.__init__()：读取多个 JSON 文件，解析每个条目并存入数据库

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataJSON.__init__()`（553-571行）

```python
def __init__(self, data_files: list[str], tool: str = "llama-bench"):
    super().__init__(tool)

    # Get the appropriate field list based on tool
    db_fields = LLAMA_BENCH_DB_FIELDS if tool == "llama-bench" else TEST_BACKEND_OPS_DB_FIELDS

    for data_file in data_files:
        with open(data_file, "r", encoding="utf-8") as fp:
            parsed = json.load(fp)

            for i, entry in enumerate(parsed):
                for k in entry.keys() - set(db_fields):
                    del entry[k]

                if (missing_keys := self._check_keys(entry.keys())):
                    raise RuntimeError(f"Missing required data key(s) at entry {i + 1}: {', '.join(missing_keys)}")

                self.cursor.execute(f"INSERT INTO {self.table_name}({', '.join(entry.keys())}) VALUES({', '.join('?' * len(entry))});", tuple(entry.values()))

    self._builds_init()
```

**解释**：此方法读取多个 JSON 文件，解析每个条目并存入数据库。它遍历所有输入文件，加载JSON数组，逐条处理条目时移除多余字段并验证必需字段后插入数据库。就像质量控制总监的时光机能够批量导入多个月份的生产报告，自动标准化格式并检查数据完整性。

---

### 109.0.21 LlamaBenchDataJSON.valid_format()：检查 JSON 文件格式是否有效

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataJSON.valid_format()`（573-586行）

```python
@staticmethod
def valid_format(data_files: list[str]) -> bool:
    if not data_files:
        return False

    for data_file in data_files:
        try:
            with open(data_file, "r", encoding="utf-8") as fp:
                json.load(fp)
        except Exception as e:
            logger.debug(f'"{data_file}" is not a valid JSON file.', exc_info=e)
            return False

    return True
```

**解释**：此静态方法检查 JSON 文件格式是否有效。它遍历所有输入文件，尝试将每个文件解析为JSON；如果任何文件解析失败则返回False。就像质量控制总监在使用时光机前会逐份检查历史数据卡片的可读性，确保所有数据源都能被正确解析。

---

### 109.0.22 LlamaBenchDataCSV.__init__()：读取多个 CSV 文件，解析每行数据并存入数据库

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataCSV.__init__()`（592-610行）

```python
def __init__(self, data_files: list[str], tool: str = "llama-bench"):
    super().__init__(tool)

    # Get the appropriate field list based on tool
    db_fields = LLAMA_BENCH_DB_FIELDS if tool == "llama-bench" else TEST_BACKEND_OPS_DB_FIELDS

    for data_file in data_files:
        with open(data_file, "r", encoding="utf-8") as fp:
            for i, parsed in enumerate(csv.DictReader(fp)):
                keys = set(parsed.keys())

                for k in keys - set(db_fields):
                    del parsed[k]

                if (missing_keys := self._check_keys(keys)):
                    raise RuntimeError(f"Missing required data key(s) at line {i + 1}: {', '.join(missing_keys)}")

                self.cursor.execute(f"INSERT INTO {self.table_name}({', '.join(parsed.keys())}) VALUES({', '.join('?' * len(parsed))});", tuple(parsed.values()))

    self._builds_init()
```

**解释**：此方法读取多个 CSV 文件，解析每行数据并存入数据库。它遍历所有输入文件，使用CSV DictReader逐行解析，移除不在字段列表中的键，验证必需字段后插入数据库。就像质量控制总监的时光机能够处理多个不同格式的生产表格，自动映射列名并确保只保留有效数据字段。

---

### 109.0.23 LlamaBenchDataCSV.valid_format()：检查 CSV 文件格式是否有效

**源码路径**：`scripts/compare-llama-bench.py` - `LlamaBenchDataCSV.valid_format()`（612-626行）

```python
@staticmethod
def valid_format(data_files: list[str]) -> bool:
    if not data_files:
        return False

    for data_file in data_files:
        try:
            with open(data_file, "r", encoding="utf-8") as fp:
                for parsed in csv.DictReader(fp):
                    break
        except Exception as e:
            logger.debug(f'"{data_file}" is not a valid CSV file.', exc_info=e)
            return False

    return True
```

**解释**：此静态方法检查 CSV 文件格式是否有效。它遍历所有输入文件，尝试读取每个文件的第一行作为标题并解析为字典；如果成功则继续，否则捕获异常返回False。就像质量控制总监在使用时光机前会快速检查历史数据卡片是否包含可识别的表头，确保后续数据能被正确映射。

