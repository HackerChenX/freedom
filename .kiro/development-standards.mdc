---
alwaysApply: true
---

# 股票分析系统开发规范

## 编码规范

### 1. 命名规范

**类名**: 使用大驼峰命名法
```python
class BaseStrategy:
    pass

class MomentumStrategy(BaseStrategy):
    pass
```

**函数/方法名**: 使用小写字母加下划线
```python
def get_logger(name: str) -> logging.Logger:
    pass

def parse_date(date_str: str) -> datetime:
    pass
```

**变量名**: 使用小写字母加下划线
```python
stock_code = "000001"
output_dir = "data/result"
```

**常量名**: 使用大写字母加下划线
```python
GLOBAL_DATE = "2024-01-01"
DEFAULT_TIMEOUT = 30
```

**文件名**: 使用小写字母加下划线
```python
# 正确
file_utils.py
date_utils.py
query_executor.py

# 错误
FileUtils.py
dateUtils.py
QueryExecutor.py
```

### 2. 导入规范

**导入顺序**: 标准库 > 第三方库 > 项目内模块
```python
# 标准库
import os
import sys
from typing import Dict, List, Optional

# 第三方库
import pandas as pd
import numpy as np
from clickhouse_driver import Client

# 项目内模块
from db.query_executor import get_query_executor
from db.sql_manager import QueryType
from utils.logger import get_logger
```

**禁止通配符导入**:
```python
# 错误
from utils import *

# 正确
from utils.logger import get_logger
from utils.date_utils import parse_date
```

**使用绝对导入**:
```python
# 推荐
from db.clickhouse_db import get_clickhouse_db

# 避免
from .clickhouse_db import get_clickhouse_db
```

### 3. 类型提示规范

**函数参数和返回值**:
```python
def get_stock_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取股票数据"""
    pass

def calculate_indicator(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """计算技术指标"""
    pass
```

**类属性**:
```python
class DatabaseConfig:
    host: str
    port: int
    database: str
    timeout: Optional[int] = None
```

**复杂类型**:
```python
from typing import Dict, List, Optional, Union, Tuple

def process_data(
    data: Dict[str, pd.DataFrame],
    params: Optional[Dict[str, Union[str, int]]] = None
) -> Tuple[bool, List[str]]:
    pass
```

### 4. 文档字符串规范

**使用Google风格**:
```python
def calculate_moving_average(data: pd.Series, period: int) -> pd.Series:
    """计算移动平均线
    
    Args:
        data: 价格数据序列
        period: 移动平均周期
        
    Returns:
        移动平均线数据序列
        
    Raises:
        ValueError: 当period小于1时抛出
        
    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> ma = calculate_moving_average(data, 3)
        >>> print(ma)
    """
    if period < 1:
        raise ValueError("period must be greater than 0")
    
    return data.rolling(window=period).mean()
```

**模块级文档**:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
技术指标计算模块

提供各种技术指标的计算功能，包括：
- 移动平均线（MA）
- 相对强弱指数（RSI）
- 布林带（BOLL）
- MACD指标

Usage:
    from indicators.technical import calculate_moving_average
    
    ma = calculate_moving_average(data, period=20)
"""
```

## 数据库访问规范

### 1. 统一查询接口

**必须使用查询执行器**:
```python
# 正确方式
from db.query_executor import get_query_executor
from db.sql_manager import QueryType

query_executor = get_query_executor()
data = query_executor.execute_query(
    QueryType.STOCK_DATA,
    {
        'code': '000001',
        'start_date': '2024-01-01',
        'end_date': '2024-12-31'
    }
)
```

**禁止直接数据库访问**:
```python
# 错误方式 - 禁止使用
conn = get_clickhouse_db()
data = conn.query_dataframe(
    "SELECT * FROM stock_info WHERE code = '000001'"
)
```

### 2. 查询参数化

**使用参数化查询**:
```python
# 正确
params = {
    'code': stock_code,
    'start_date': start_date,
    'end_date': end_date
}
data = query_executor.execute_query(QueryType.STOCK_DATA, params)

# 错误 - 字符串拼接
query = f"SELECT * FROM stock_info WHERE code = '{stock_code}'"
```

### 3. 错误处理

**数据库操作错误处理**:
```python
try:
    data = query_executor.execute_query(QueryType.STOCK_DATA, params)
    if data.empty:
        logger.warning(f"未获取到股票数据: {params}")
        return None
    return data
except Exception as e:
    logger.error(f"查询股票数据失败: {e}")
    # 降级处理
    return fallback_query(params)
```

## 配置管理规范

### 1. 配置文件使用

**配置获取**:
```python
from config import get_config

# 获取配置
config = get_config()
db_config = config.get('database', {})
host = db_config.get('host', 'localhost')
port = db_config.get('port', 9000)
```

**禁止硬编码**:
```python
# 错误 - 硬编码
HOST = "localhost"
PORT = 9000

# 正确 - 配置化
HOST = get_config().get('database.host', 'localhost')
PORT = get_config().get('database.port', 9000)
```

### 2. 环境变量支持

**环境变量优先**:
```python
import os
from config import get_config

# 环境变量优先
host = os.environ.get('CLICKHOUSE_HOST') or get_config().get('database.host', 'localhost')
port = int(os.environ.get('CLICKHOUSE_PORT', get_config().get('database.port', 9000)))
```

## 异常处理规范

### 1. 异常处理模式

**使用具体异常类型**:
```python
try:
    data = query_executor.execute_query(QueryType.STOCK_DATA, params)
except ConnectionError as e:
    logger.error(f"数据库连接失败: {e}")
    raise
except ValueError as e:
    logger.error(f"参数错误: {e}")
    return None
except Exception as e:
    logger.error(f"未知错误: {e}")
    raise
```

### 2. 日志记录

**统一日志记录**:
```python
from utils.logger import get_logger

logger = get_logger(__name__)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"开始处理数据，数据量: {len(data)}")
    
    try:
        result = complex_calculation(data)
        logger.info(f"数据处理完成，结果数量: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        raise
```

## 代码重用规范

### 功能重复检查
在实现任何新功能之前，**必须**先检查系统中是否已存在相似功能：

1. **检查现有工具模块**
   - 工具函数：`utils/` 目录下的各种工具类
   - 技术指标：`indicators/` 目录下的指标实现
   - 分析引擎：`analysis/engines/` 目录下的分析组件
   - 数据处理：`db/` 目录下的数据访问组件

2. **常见现有功能模块**
   ```python
   # 日志记录 - 使用现有的 utils/logger.py
   from utils.logger import getLogger
   logger = getLogger(__name__)
   
   # 缓存功能 - 使用现有的 utils/cache.py
   from utils.cache import LRUCache, MemoryCache
   
   # 装饰器 - 使用现有的 utils/decorators.py
   from utils.decorators import performance_monitor, exception_handler
   
   # 文件操作 - 使用现有的 utils/file_utils.py
   from utils.file_utils import ensure_dir, save_json, load_json
   
   # 日期管理 - 使用现有的 analysis/engines/date_manager.py
   from analysis.engines.date_manager import DateManager
   
   # 技术指标计算 - 使用现有的 utils/technical_utils.py
   from utils.technical_utils import calculate_ma, calculate_ema, calculate_macd
   ```

3. **功能扩展原则**
   - 优先在现有模块基础上扩展功能
   - 避免创建功能重复的新模块
   - 如需重构，应先讨论架构变更
   - 删除重复功能时需要全面测试

4. **检查方法**
   ```bash
   # 搜索相似功能
   grep -r "function_name" --include="*.py" .
   
   # 检查类似的类定义
   find . -name "*.py" -exec grep -l "class.*Similar" {} \;
   
   # 使用 codebase_search 工具查找语义相似的代码
   ```

### 重构指导原则
- 合并相似功能到统一模块
- 保持向后兼容性
- 提供迁移指南
- 更新相关文档

## 数据库访问约束

### ClickHouse表结构约束
系统使用ClickHouse数据库，必须严格遵循现有表结构：

#### 主要数据表结构

1. **stock_info表**（主要股票数据表）
   ```sql
   CREATE TABLE stock.stock_info (
       code String,           -- 股票代码
       name String,           -- 股票名称  
       date Date,             -- 交易日期
       level String,          -- K线周期
       open Float64,          -- 开盘价
       close Float64,         -- 收盘价
       high Float64,          -- 最高价
       low Float64,           -- 最低价
       volume Float64,        -- 成交量
       turnover_rate Float64, -- 换手率
       price_change Float64,  -- 价格变动
       price_range Float64,   -- 价格区间
       industry String,       -- 行业（默认空字符串）
       datetime DateTime,     -- 日期时间（默认当前时间）
       seq UInt32            -- 序号（默认0）
   ) ENGINE = ReplacingMergeTree()
   PRIMARY KEY (code, level, date, datetime, seq)
   ORDER BY (code, level, date, datetime, seq);
   ```

2. **crawler_articles表**（爬虫文章数据表）
   ```sql
   CREATE TABLE crawler_articles (
       id String,
       source String,
       title String,
       content String,
       author String,
       publish_time DateTime,
       url String,
       crawl_time DateTime DEFAULT now(),
       article_type String,
       view_count UInt32 DEFAULT 0,
       like_count UInt32 DEFAULT 0,
       comment_count UInt32 DEFAULT 0,
       stock_codes Array(String),
       concepts Array(String),
       sentiment_score Float32 DEFAULT 0.0
   ) ENGINE = MergeTree()
   ORDER BY (source, publish_time)
   ```

### SQL查询规范

1. **字段名称严格对齐**
   ```python
   # ✅ 正确：使用实际存在的字段
   query = """
   SELECT code, name, date, level, open, close, high, low, volume, 
          turnover_rate, price_change, price_range, industry
   FROM stock_info 
   WHERE code = '000001' AND date >= '2023-01-01'
   """
   
   # ❌ 错误：使用不存在的字段
   query = """
   SELECT code, name, date, level, open, close, high, low, volume,
          turnover, change_pct, market_cap  -- 这些字段不存在
   FROM stock_info 
   WHERE code = '000001'
   """
   ```

2. **数据类型约束**
   ```python
   # ✅ 正确：遵循字段类型
   params = {
       'code': '000001',           # String
       'date': '2023-01-01',       # Date
       'level': 'daily',           # String
       'volume': 1000000.0,        # Float64
       'seq': 1                    # UInt32
   }
   
   # ❌ 错误：类型不匹配
   params = {
       'code': 1,                  # 应为String
       'date': 20230101,           # 应为Date格式
       'volume': '1000000',        # 应为Float64
       'seq': 1.5                  # 应为UInt32
   }
   ```

3. **查询验证机制**
   ```python
   # 在查询前验证字段存在性
   def validate_query_fields(query: str, table_name: str) -> bool:
       """验证查询中的字段是否存在于表中"""
       # 获取表结构
       structure_query = f"DESCRIBE TABLE {table_name}"
       # 验证字段存在性
       # 返回验证结果
       pass
   
   # 使用统一的查询执行器
   from db.query_executor import get_query_executor
   executor = get_query_executor()
   result = executor.execute_query(query, QueryType.SELECT)
   ```

### 数据库连接配置
```python
# 使用配置文件管理数据库连接
clickhouse_config = {
    "host": "localhost",
    "port": 9000,
    "database": "stock",
    "user": "default",
    "password": "123456",
    "timeout": 30,
    "compression": True
}
```

### 表结构变更流程
1. 新增字段需要在`sql/ddl/`目录下创建DDL文件
2. 更新`models/stock_info.py`中的数据模型
3. 更新相关的查询语句和参数验证
4. 执行数据迁移脚本
5. 更新相关文档

## 测试规范

### 1. 单元测试

**测试文件结构**:
```
tests/
├── unit/              # 单元测试
│   ├── test_indicators.py
│   ├── test_query_executor.py
│   └── test_utils.py
├── integration/       # 集成测试
│   ├── test_database.py
│   └── test_api.py
└── fixtures/          # 测试数据
    ├── sample_data.csv
    └── test_config.json
```

**测试用例编写**:
```python
import unittest
from unittest.mock import Mock, patch
import pandas as pd

from indicators.technical import calculate_moving_average

class TestTechnicalIndicators(unittest.TestCase):
    
    def setUp(self):
        """测试前准备"""
        self.sample_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    def test_calculate_moving_average_normal(self):
        """测试正常情况下的移动平均线计算"""
        result = calculate_moving_average(self.sample_data, period=3)
        expected = pd.Series([None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_calculate_moving_average_invalid_period(self):
        """测试无效周期参数"""
        with self.assertRaises(ValueError):
            calculate_moving_average(self.sample_data, period=0)
```

### 2. 模拟测试

**数据库模拟**:
```python
@patch('db.query_executor.get_query_executor')
def test_get_stock_data(self, mock_executor):
    """测试股票数据获取"""
    # 模拟返回数据
    mock_data = pd.DataFrame({
        'code': ['000001'],
        'date': ['2024-01-01'],
        'close': [10.0]
    })
    mock_executor.return_value.execute_query.return_value = mock_data
    
    # 执行测试
    result = get_stock_data('000001', '2024-01-01', '2024-01-01')
    
    # 验证结果
    self.assertEqual(len(result), 1)
    self.assertEqual(result.iloc[0]['code'], '000001')
```

## 性能优化规范

### 1. 数据处理优化

**使用向量化操作**:
```python
# 推荐 - 向量化
data['ma'] = data['close'].rolling(window=20).mean()

# 避免 - 循环
ma_values = []
for i in range(len(data)):
    if i >= 19:
        ma_values.append(data['close'].iloc[i-19:i+1].mean())
    else:
        ma_values.append(None)
```

### 2. 内存管理

**及时释放资源**:
```python
def process_large_dataset(file_path: str) -> pd.DataFrame:
    """处理大数据集"""
    try:
        # 分块读取
        chunks = pd.read_csv(file_path, chunksize=10000)
        results = []
        
        for chunk in chunks:
            processed = process_chunk(chunk)
            results.append(processed)
            # 显式删除chunk，释放内存
            del chunk
        
        return pd.concat(results, ignore_index=True)
    finally:
        # 清理临时变量
        if 'chunks' in locals():
            del chunks
        if 'results' in locals():
            del results
```

## 代码审查检查清单

### 1. 基本检查
- [ ] 代码符合PEP8规范
- [ ] 函数和类有适当的文档字符串
- [ ] 使用了类型提示
- [ ] 导入语句按规范排序

### 2. 架构检查
- [ ] 遵循分层架构原则
- [ ] 使用统一的数据访问接口
- [ ] 配置项没有硬编码
- [ ] 没有使用全局单例模式

### 3. 质量检查
- [ ] 有适当的异常处理
- [ ] 有必要的日志记录
- [ ] 有单元测试覆盖
- [ ] 性能考虑合理

### 4. 安全检查
- [ ] 敏感信息没有硬编码
- [ ] 输入参数有验证
- [ ] SQL查询使用参数化
- [ ] 错误信息不泄露敏感信息
# 股票分析系统开发规范

## 编码规范

### 1. 命名规范

**类名**: 使用大驼峰命名法
```python
class BaseStrategy:
    pass

class MomentumStrategy(BaseStrategy):
    pass
```

**函数/方法名**: 使用小写字母加下划线
```python
def get_logger(name: str) -> logging.Logger:
    pass

def parse_date(date_str: str) -> datetime:
    pass
```

**变量名**: 使用小写字母加下划线
```python
stock_code = "000001"
output_dir = "data/result"
```

**常量名**: 使用大写字母加下划线
```python
GLOBAL_DATE = "2024-01-01"
DEFAULT_TIMEOUT = 30
```

**文件名**: 使用小写字母加下划线
```python
# 正确
file_utils.py
date_utils.py
query_executor.py

# 错误
FileUtils.py
dateUtils.py
QueryExecutor.py
```

### 2. 导入规范

**导入顺序**: 标准库 > 第三方库 > 项目内模块
```python
# 标准库
import os
import sys
from typing import Dict, List, Optional

# 第三方库
import pandas as pd
import numpy as np
from clickhouse_driver import Client

# 项目内模块
from db.query_executor import get_query_executor
from db.sql_manager import QueryType
from utils.logger import get_logger
```

**禁止通配符导入**:
```python
# 错误
from utils import *

# 正确
from utils.logger import get_logger
from utils.date_utils import parse_date
```

**使用绝对导入**:
```python
# 推荐
from db.clickhouse_db import get_clickhouse_db

# 避免
from .clickhouse_db import get_clickhouse_db
```

### 3. 类型提示规范

**函数参数和返回值**:
```python
def get_stock_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取股票数据"""
    pass

def calculate_indicator(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """计算技术指标"""
    pass
```

**类属性**:
```python
class DatabaseConfig:
    host: str
    port: int
    database: str
    timeout: Optional[int] = None
```

**复杂类型**:
```python
from typing import Dict, List, Optional, Union, Tuple

def process_data(
    data: Dict[str, pd.DataFrame],
    params: Optional[Dict[str, Union[str, int]]] = None
) -> Tuple[bool, List[str]]:
    pass
```

### 4. 文档字符串规范

**使用Google风格**:
```python
def calculate_moving_average(data: pd.Series, period: int) -> pd.Series:
    """计算移动平均线
    
    Args:
        data: 价格数据序列
        period: 移动平均周期
        
    Returns:
        移动平均线数据序列
        
    Raises:
        ValueError: 当period小于1时抛出
        
    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> ma = calculate_moving_average(data, 3)
        >>> print(ma)
    """
    if period < 1:
        raise ValueError("period must be greater than 0")
    
    return data.rolling(window=period).mean()
```

**模块级文档**:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
技术指标计算模块

提供各种技术指标的计算功能，包括：
- 移动平均线（MA）
- 相对强弱指数（RSI）
- 布林带（BOLL）
- MACD指标

Usage:
    from indicators.technical import calculate_moving_average
    
    ma = calculate_moving_average(data, period=20)
"""
```

## 数据库访问规范

### 1. 统一查询接口

**必须使用查询执行器**:
```python
# 正确方式
from db.query_executor import get_query_executor
from db.sql_manager import QueryType

query_executor = get_query_executor()
data = query_executor.execute_query(
    QueryType.STOCK_DATA,
    {
        'code': '000001',
        'start_date': '2024-01-01',
        'end_date': '2024-12-31'
    }
)
```

**禁止直接数据库访问**:
```python
# 错误方式 - 禁止使用
conn = get_clickhouse_db()
data = conn.query_dataframe(
    "SELECT * FROM stock_info WHERE code = '000001'"
)
```

### 2. 查询参数化

**使用参数化查询**:
```python
# 正确
params = {
    'code': stock_code,
    'start_date': start_date,
    'end_date': end_date
}
data = query_executor.execute_query(QueryType.STOCK_DATA, params)

# 错误 - 字符串拼接
query = f"SELECT * FROM stock_info WHERE code = '{stock_code}'"
```

### 3. 错误处理

**数据库操作错误处理**:
```python
try:
    data = query_executor.execute_query(QueryType.STOCK_DATA, params)
    if data.empty:
        logger.warning(f"未获取到股票数据: {params}")
        return None
    return data
except Exception as e:
    logger.error(f"查询股票数据失败: {e}")
    # 降级处理
    return fallback_query(params)
```

## 配置管理规范

### 1. 配置文件使用

**配置获取**:
```python
from config import get_config

# 获取配置
config = get_config()
db_config = config.get('database', {})
host = db_config.get('host', 'localhost')
port = db_config.get('port', 9000)
```

**禁止硬编码**:
```python
# 错误 - 硬编码
HOST = "localhost"
PORT = 9000

# 正确 - 配置化
HOST = get_config().get('database.host', 'localhost')
PORT = get_config().get('database.port', 9000)
```

### 2. 环境变量支持

**环境变量优先**:
```python
import os
from config import get_config

# 环境变量优先
host = os.environ.get('CLICKHOUSE_HOST') or get_config().get('database.host', 'localhost')
port = int(os.environ.get('CLICKHOUSE_PORT', get_config().get('database.port', 9000)))
```

## 异常处理规范

### 1. 异常处理模式

**使用具体异常类型**:
```python
try:
    data = query_executor.execute_query(QueryType.STOCK_DATA, params)
except ConnectionError as e:
    logger.error(f"数据库连接失败: {e}")
    raise
except ValueError as e:
    logger.error(f"参数错误: {e}")
    return None
except Exception as e:
    logger.error(f"未知错误: {e}")
    raise
```

### 2. 日志记录

**统一日志记录**:
```python
from utils.logger import get_logger

logger = get_logger(__name__)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"开始处理数据，数据量: {len(data)}")
    
    try:
        result = complex_calculation(data)
        logger.info(f"数据处理完成，结果数量: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        raise
```

## 代码重用规范

### 功能重复检查
在实现任何新功能之前，**必须**先检查系统中是否已存在相似功能：

1. **检查现有工具模块**
   - 工具函数：`utils/` 目录下的各种工具类
   - 技术指标：`indicators/` 目录下的指标实现
   - 分析引擎：`analysis/engines/` 目录下的分析组件
   - 数据处理：`db/` 目录下的数据访问组件

2. **常见现有功能模块**
   ```python
   # 日志记录 - 使用现有的 utils/logger.py
   from utils.logger import getLogger
   logger = getLogger(__name__)
   
   # 缓存功能 - 使用现有的 utils/cache.py
   from utils.cache import LRUCache, MemoryCache
   
   # 装饰器 - 使用现有的 utils/decorators.py
   from utils.decorators import performance_monitor, exception_handler
   
   # 文件操作 - 使用现有的 utils/file_utils.py
   from utils.file_utils import ensure_dir, save_json, load_json
   
   # 日期管理 - 使用现有的 analysis/engines/date_manager.py
   from analysis.engines.date_manager import DateManager
   
   # 技术指标计算 - 使用现有的 utils/technical_utils.py
   from utils.technical_utils import calculate_ma, calculate_ema, calculate_macd
   ```

3. **功能扩展原则**
   - 优先在现有模块基础上扩展功能
   - 避免创建功能重复的新模块
   - 如需重构，应先讨论架构变更
   - 删除重复功能时需要全面测试

4. **检查方法**
   ```bash
   # 搜索相似功能
   grep -r "function_name" --include="*.py" .
   
   # 检查类似的类定义
   find . -name "*.py" -exec grep -l "class.*Similar" {} \;
   
   # 使用 codebase_search 工具查找语义相似的代码
   ```

### 重构指导原则
- 合并相似功能到统一模块
- 保持向后兼容性
- 提供迁移指南
- 更新相关文档

## 数据库访问约束

### ClickHouse表结构约束
系统使用ClickHouse数据库，必须严格遵循现有表结构：

#### 主要数据表结构

1. **stock_info表**（主要股票数据表）
   ```sql
   CREATE TABLE stock.stock_info (
       code String,           -- 股票代码
       name String,           -- 股票名称  
       date Date,             -- 交易日期
       level String,          -- K线周期
       open Float64,          -- 开盘价
       close Float64,         -- 收盘价
       high Float64,          -- 最高价
       low Float64,           -- 最低价
       volume Float64,        -- 成交量
       turnover_rate Float64, -- 换手率
       price_change Float64,  -- 价格变动
       price_range Float64,   -- 价格区间
       industry String,       -- 行业（默认空字符串）
       datetime DateTime,     -- 日期时间（默认当前时间）
       seq UInt32            -- 序号（默认0）
   ) ENGINE = ReplacingMergeTree()
   PRIMARY KEY (code, level, date, datetime, seq)
   ORDER BY (code, level, date, datetime, seq);
   ```

2. **crawler_articles表**（爬虫文章数据表）
   ```sql
   CREATE TABLE crawler_articles (
       id String,
       source String,
       title String,
       content String,
       author String,
       publish_time DateTime,
       url String,
       crawl_time DateTime DEFAULT now(),
       article_type String,
       view_count UInt32 DEFAULT 0,
       like_count UInt32 DEFAULT 0,
       comment_count UInt32 DEFAULT 0,
       stock_codes Array(String),
       concepts Array(String),
       sentiment_score Float32 DEFAULT 0.0
   ) ENGINE = MergeTree()
   ORDER BY (source, publish_time)
   ```

### SQL查询规范

1. **字段名称严格对齐**
   ```python
   # ✅ 正确：使用实际存在的字段
   query = """
   SELECT code, name, date, level, open, close, high, low, volume, 
          turnover_rate, price_change, price_range, industry
   FROM stock_info 
   WHERE code = '000001' AND date >= '2023-01-01'
   """
   
   # ❌ 错误：使用不存在的字段
   query = """
   SELECT code, name, date, level, open, close, high, low, volume,
          turnover, change_pct, market_cap  -- 这些字段不存在
   FROM stock_info 
   WHERE code = '000001'
   """
   ```

2. **数据类型约束**
   ```python
   # ✅ 正确：遵循字段类型
   params = {
       'code': '000001',           # String
       'date': '2023-01-01',       # Date
       'level': 'daily',           # String
       'volume': 1000000.0,        # Float64
       'seq': 1                    # UInt32
   }
   
   # ❌ 错误：类型不匹配
   params = {
       'code': 1,                  # 应为String
       'date': 20230101,           # 应为Date格式
       'volume': '1000000',        # 应为Float64
       'seq': 1.5                  # 应为UInt32
   }
   ```

3. **查询验证机制**
   ```python
   # 在查询前验证字段存在性
   def validate_query_fields(query: str, table_name: str) -> bool:
       """验证查询中的字段是否存在于表中"""
       # 获取表结构
       structure_query = f"DESCRIBE TABLE {table_name}"
       # 验证字段存在性
       # 返回验证结果
       pass
   
   # 使用统一的查询执行器
   from db.query_executor import get_query_executor
   executor = get_query_executor()
   result = executor.execute_query(query, QueryType.SELECT)
   ```

### 数据库连接配置
```python
# 使用配置文件管理数据库连接
clickhouse_config = {
    "host": "localhost",
    "port": 9000,
    "database": "stock",
    "user": "default",
    "password": "123456",
    "timeout": 30,
    "compression": True
}
```

### 表结构变更流程
1. 新增字段需要在`sql/ddl/`目录下创建DDL文件
2. 更新`models/stock_info.py`中的数据模型
3. 更新相关的查询语句和参数验证
4. 执行数据迁移脚本
5. 更新相关文档

## 测试规范

### 1. 单元测试

**测试文件结构**:
```
tests/
├── unit/              # 单元测试
│   ├── test_indicators.py
│   ├── test_query_executor.py
│   └── test_utils.py
├── integration/       # 集成测试
│   ├── test_database.py
│   └── test_api.py
└── fixtures/          # 测试数据
    ├── sample_data.csv
    └── test_config.json
```

**测试用例编写**:
```python
import unittest
from unittest.mock import Mock, patch
import pandas as pd

from indicators.technical import calculate_moving_average

class TestTechnicalIndicators(unittest.TestCase):
    
    def setUp(self):
        """测试前准备"""
        self.sample_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    def test_calculate_moving_average_normal(self):
        """测试正常情况下的移动平均线计算"""
        result = calculate_moving_average(self.sample_data, period=3)
        expected = pd.Series([None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_calculate_moving_average_invalid_period(self):
        """测试无效周期参数"""
        with self.assertRaises(ValueError):
            calculate_moving_average(self.sample_data, period=0)
```

### 2. 模拟测试

**数据库模拟**:
```python
@patch('db.query_executor.get_query_executor')
def test_get_stock_data(self, mock_executor):
    """测试股票数据获取"""
    # 模拟返回数据
    mock_data = pd.DataFrame({
        'code': ['000001'],
        'date': ['2024-01-01'],
        'close': [10.0]
    })
    mock_executor.return_value.execute_query.return_value = mock_data
    
    # 执行测试
    result = get_stock_data('000001', '2024-01-01', '2024-01-01')
    
    # 验证结果
    self.assertEqual(len(result), 1)
    self.assertEqual(result.iloc[0]['code'], '000001')
```

## 性能优化规范

### 1. 数据处理优化

**使用向量化操作**:
```python
# 推荐 - 向量化
data['ma'] = data['close'].rolling(window=20).mean()

# 避免 - 循环
ma_values = []
for i in range(len(data)):
    if i >= 19:
        ma_values.append(data['close'].iloc[i-19:i+1].mean())
    else:
        ma_values.append(None)
```

### 2. 内存管理

**及时释放资源**:
```python
def process_large_dataset(file_path: str) -> pd.DataFrame:
    """处理大数据集"""
    try:
        # 分块读取
        chunks = pd.read_csv(file_path, chunksize=10000)
        results = []
        
        for chunk in chunks:
            processed = process_chunk(chunk)
            results.append(processed)
            # 显式删除chunk，释放内存
            del chunk
        
        return pd.concat(results, ignore_index=True)
    finally:
        # 清理临时变量
        if 'chunks' in locals():
            del chunks
        if 'results' in locals():
            del results
```

## 代码审查检查清单

### 1. 基本检查
- [ ] 代码符合PEP8规范
- [ ] 函数和类有适当的文档字符串
- [ ] 使用了类型提示
- [ ] 导入语句按规范排序

### 2. 架构检查
- [ ] 遵循分层架构原则
- [ ] 使用统一的数据访问接口
- [ ] 配置项没有硬编码
- [ ] 没有使用全局单例模式

### 3. 质量检查
- [ ] 有适当的异常处理
- [ ] 有必要的日志记录
- [ ] 有单元测试覆盖
- [ ] 性能考虑合理

### 4. 安全检查
- [ ] 敏感信息没有硬编码
- [ ] 输入参数有验证
- [ ] SQL查询使用参数化
- [ ] 错误信息不泄露敏感信息
