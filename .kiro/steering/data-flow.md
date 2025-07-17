# 股票分析系统数据流向

## 数据流向概述

系统数据流向遵循**自上而下的分层处理**原则，确保数据在各层之间有序流动：

```
外部数据源 → 数据爬取 → 数据存储 → 数据访问 → 业务处理 → 结果输出
```

## 详细数据流向

### 1. 数据获取流程

```
外部API/网站
    ↓
[crawler/](mdc:crawler/) - 数据爬取模块
    ↓
[db/clickhouse_db.py](mdc:db/clickhouse_db.py) - 数据库连接
    ↓
ClickHouse数据库 - 原始数据存储
```

**关键文件**:
- [crawler/integration/](mdc:crawler/integration/) - 数据集成模块
- [api/](mdc:api/) - 外部API接口实现

### 2. 数据访问流程

```
业务请求
    ↓
[db/query_executor.py](mdc:db/query_executor.py) - 统一查询执行器
    ↓
[db/sql_manager.py](mdc:db/sql_manager.py) - SQL查询管理
    ↓
[db/clickhouse_db.py](mdc:db/clickhouse_db.py) - 数据库连接
    ↓
ClickHouse数据库 - 数据查询
    ↓
DataFrame/结果集 - 返回数据
```

**核心原则**:
- **统一入口**: 所有数据访问必须通过查询执行器
- **查询管理**: 使用预定义的查询模板
- **连接池**: 统一的数据库连接池管理

### 3. 业务处理流程

```
原始数据
    ↓
[indicators/](mdc:indicators/) - 技术指标计算
    ↓
[formula/](mdc:formula/) - 公式计算
    ↓
[analysis/](mdc:analysis/) - 分析处理
    ↓
[strategy/](mdc:strategy/) - 策略执行
    ↓
业务结果
```

**数据处理链**:
1. **指标计算**: 基础技术指标计算
2. **公式运算**: 复杂公式和算法
3. **分析处理**: 市场分析和买点分析
4. **策略执行**: 选股策略和决策

### 4. 结果输出流程

```
业务结果
    ↓
[data/result/](mdc:data/result/) - 结果数据存储
    ↓
[bin/](mdc:bin/) - 主程序输出
    ↓
用户界面/文件输出
```

## 数据流向规范

### 1. 数据获取规范

**必须遵循**:
```python
# 正确的数据获取方式
from db.query_executor import get_query_executor
from db.sql_manager import QueryType

query_executor = get_query_executor()
data = query_executor.execute_query(
    QueryType.STOCK_DATA,
    {'code': '000001', 'start_date': '2024-01-01'}
)
```

**禁止使用**:
```python
# 错误的直接数据库访问
conn.query_dataframe("SELECT * FROM stock_info WHERE code = '000001'")
```

### 2. 数据处理规范

**分层处理**:
- **L3层**: 只负责数据获取和基础查询
- **L4层**: 负责数据转换和指标计算
- **L5层**: 负责业务逻辑和策略处理
- **L6层**: 负责结果输出和用户交互

**数据传递**:
- 使用标准的DataFrame格式
- 保持数据结构的一致性
- 添加必要的数据验证

### 3. 错误处理规范

**数据流中的错误处理**:
```python
try:
    # 数据获取
    data = query_executor.execute_query(QueryType.STOCK_DATA, params)
    
    # 数据验证
    if data.empty:
        logger.warning("未获取到数据")
        return None
    
    # 数据处理
    result = process_data(data)
    
except Exception as e:
    logger.error(f"数据处理失败: {e}")
    # 降级处理
    return fallback_process(params)
```

## 数据缓存策略

### 1. 查询缓存
- **位置**: [db/query_executor.py](mdc:db/query_executor.py)
- **策略**: 基于查询参数的智能缓存
- **TTL**: 根据数据更新频率设置

### 2. 计算缓存
- **位置**: [indicators/](mdc:indicators/) 和 [formula/](mdc:formula/)
- **策略**: 基于输入参数的结果缓存
- **失效**: 基于数据版本的缓存失效

### 3. 结果缓存
- **位置**: [data/result/](mdc:data/result/)
- **策略**: 持久化缓存重要计算结果
- **管理**: 定期清理过期缓存

## 数据质量保证

### 1. 数据验证
- **输入验证**: 验证查询参数的合法性
- **输出验证**: 验证返回数据的完整性
- **类型检查**: 确保数据类型的正确性

### 2. 数据监控
- **查询监控**: 监控查询执行时间和成功率
- **数据监控**: 监控数据质量和完整性
- **异常监控**: 及时发现和处理数据异常

### 3. 数据一致性
- **事务处理**: 确保数据操作的原子性
- **并发控制**: 处理并发访问的数据一致性
- **数据同步**: 保证多数据源的数据一致性

## 性能优化原则

### 1. 查询优化
- **查询合并**: 合并相似的查询请求
- **索引优化**: 确保查询使用合适的索引
- **分页查询**: 大数据量查询使用分页

### 2. 数据传输优化
- **数据压缩**: 压缩大数据量的传输
- **批量处理**: 批量处理多个数据请求
- **异步处理**: 使用异步方式处理耗时操作

### 3. 内存管理
- **数据释放**: 及时释放不再使用的数据
- **内存监控**: 监控内存使用情况
- **垃圾回收**: 合理利用Python的垃圾回收机制
description:
globs:
alwaysApply: true
---
