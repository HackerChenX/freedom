---
type: "always_apply"
description: "Example description"
---

# 股票分析系统架构规则（强制执行）

## 🏗️ 六层架构分层规则（严格遵守）

```
L6: 用户接口层 (bin/, api/) → 只能调用 L5
L5: 业务应用层 (strategy/, analysis/) → 只能调用 L4  
L4: 核心服务层 (indicators/, formula/) → 只能调用 L3
L3: 数据服务层 (db/interfaces/, db/managers/) → 只能调用 L2
L2: 存储访问层 (db/clickhouse_db.py, db/enhanced_connection_pool.py) → 只能调用 L1
L1: 基础设施层 (utils/, config/, enums/)
```

### ❌ 绝对禁止的跨层调用模式
```python
# 禁止：L6直接调用L3
from db.managers.data_access_manager import DataAccessManager  # 在bin/中

# 禁止：L5直接调用L2  
from db.enhanced_connection_pool import ClickHouseConnectionPool  # 在strategy/中

# 禁止：L4直接调用L1的具体实现
from db.clickhouse_db import get_clickhouse_db  # 在indicators/中
```

### ✅ 正确的分层调用模式
```python
# L6调用L5
from strategy.strategy_manager import StrategyManager  # 在bin/中

# L5调用L4
from indicators.complete_indicator_registry import get_indicator  # 在strategy/中

# L4调用L3
from db.interfaces.data_access_interface import DataAccessInterface  # 在indicators/中
```

## 🗄️ 数据库访问规则（严格执行）

### 数据库表结构标准
```sql
stock_info: (code, name, date, level, open, close, high, low, volume, turnover_rate, price_change, price_range, industry, datetime, seq)
```

### 强制查询要求
```python
# ✅ 标准查询模板 - 必须遵循
def get_stock_data(code: str, start_date: str, end_date: str, level: str = '日线'):
    return f"""
    SELECT code, name, date, open, high, low, close, volume, turnover_rate
    FROM stock_info 
    WHERE code = '{code}'
    AND level = '{level}'
    AND date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date ASC
    """

# ❌ 禁止的查询模式
# SELECT * FROM stock_info  # 禁止SELECT *
# SELECT ... FROM stock_info WHERE date > '2024-01-01'  # 缺少code条件
# SELECT ... FROM stock_info WHERE code = '000001'  # 缺少date范围
```

### 数据库连接池规范
```python
# ✅ 正确使用连接池
from db.enhanced_connection_pool import ClickHouseConnectionPool

class DataService:
    def __init__(self):
        self.pool = ClickHouseConnectionPool()
    
    def query_data(self, sql: str) -> pd.DataFrame:
        return self.pool.query_dataframe(sql)

# ❌ 禁止直接数据库连接
from db.clickhouse_db import get_clickhouse_db  # 禁止
```

## 🔧 依赖注入规范（强制执行）

### 服务注册和解析
```python
# ✅ 正确的依赖注入模式
from utils.container import container

class BusinessService:
    def __init__(self):
        self.data_access = container.resolve("DataAccessInterface")
        self.logger = container.resolve("Logger")

# ❌ 禁止直接依赖
class BadService:
    def __init__(self):
        self.db = get_clickhouse_db()  # 违反依赖注入原则
```

### 服务容器配置
所有服务必须在相应的 `__init__.py` 中注册：
```python
# db/__init__.py 中的服务注册示例
from utils.container import container
from .managers.data_access_manager import DataAccessManager

# 自动注册数据访问服务
container.register("DataAccessInterface", DataAccessManager)
```

## 📝 代码规范（强制执行）

### 命名规范
- **类名**: 大驼峰命名 (`StockAnalyzer`, `IndicatorRegistry`)
- **方法名**: 小写+下划线 (`get_stock_data`, `calculate_indicator`)
- **变量名**: 小写+下划线 (`stock_code`, `indicator_result`)
- **常量名**: 大写+下划线 (`MAX_RETRY_COUNT`, `DEFAULT_PERIOD`)
- **文件名**: 小写+下划线 (`stock_analyzer.py`, `indicator_registry.py`)

### 导入规范
```python
# 1. 标准库导入
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union

# 2. 第三方库导入  
import pandas as pd
import numpy as np

# 3. 项目内模块导入（绝对导入）
from config.database_config_manager import DatabaseConfigManager
from utils.logger import get_logger
from enums.indicator_types import IndicatorType

# ❌ 禁止的导入模式
# from db.clickhouse_db import *  # 通配符导入
# from ..relative_module import something  # 相对导入
```

### 异常处理规范
```python
from functools import wraps
from utils.logger import get_logger

logger = get_logger(__name__)

def exception_handler(reraise: bool = True, default_return=None):
    """标准异常处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"方法 {func.__name__} 执行失败: {e}")
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator

# 使用示例
class IndicatorService:
    @exception_handler(reraise=True)
    def calculate_indicator(self, code: str, indicator_type: str) -> Dict:
        # 业务逻辑
        pass
```

### 性能监控规范
```python
import time
from functools import wraps

def performance_monitor(threshold_seconds: float = 1.0):
    """性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > threshold_seconds:
                logger.warning(f"方法 {func.__name__} 执行时间过长: {execution_time:.2f}秒")
            
            return result
        return wrapper
    return decorator
```

## 🚀 指标系统规范

### 指标注册要求
所有指标必须在 [complete_indicator_registry.py](mdc:indicators/complete_indicator_registry.py) 中注册：

```python
# 指标注册示例
CORE_INDICATORS = {
    'MA': 'indicators.ma.SimpleMovingAverage',
    'EMA': 'indicators.ema.ExponentialMovingAverage',
    'MACD': 'indicators.macd.MACD',
    # ... 更多指标
}

def register_indicator(name: str, implementation_path: str):
    """注册指标实现"""
    try:
        module_path, class_name = implementation_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        indicator_class = getattr(module, class_name)
        INDICATOR_REGISTRY[name] = indicator_class
        logger.info(f"指标 {name} 注册成功")
        return True
    except Exception as e:
        logger.warning(f"指标 {name} 注册失败，使用Mock实现: {e}")
        INDICATOR_REGISTRY[name] = create_mock_indicator(name)
        return False
```

### 指标实现标准
```python
from abc import ABC, abstractmethod

class BaseIndicator(ABC):
    """指标基类"""
    
    def __init__(self, name: str, period: int = 20):
        self.name = name
        self.period = period
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指标值"""
        pass
    
    @abstractmethod
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取交易信号"""
        pass
```

## 🛡️ 质量保证要求

### 开发前检查清单
在开始任何新功能开发前，必须：

1. **检查现有实现**
   ```python
   # 搜索相似功能
   search_patterns = [
       f"def.*{feature_name}",
       f"class.*{feature_name.title()}",
       f"# {feature_name}"
   ]
   ```

2. **架构合规检查**
   - [ ] 遵循六层架构分层
   - [ ] 使用依赖注入
   - [ ] 包含异常处理装饰器
   - [ ] 添加性能监控装饰器

3. **代码质量检查**
   - [ ] 命名符合规范
   - [ ] 包含完整文档字符串
   - [ ] 通过静态分析检查

### 禁止重复建设
- 新功能前必须检查现有实现
- 扩展现有功能，不创建重复代码
- 相似逻辑抽象为公共方法
- 禁止复制粘贴超过10行代码

## 📊 性能要求

### 数据库性能
- 使用连接池管理连接（5-20并发连接）
- 实现查询结果缓存
- 使用批量查询减少网络开销
- 实现数据分页避免大结果集

### 内存管理
- 大数据集使用生成器
- 及时释放DataFrame
- 使用适当数据类型
- 控制并发任务数量

### 缓存策略
```python
from functools import lru_cache
from cachetools import TTLCache

class HighPerformanceIndicatorService:
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=300)
    
    @lru_cache(maxsize=128)
    def get_indicator_data(self, code: str, indicator: str, period: int):
        cache_key = f"{code}_{indicator}_{period}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self._calculate_indicator(code, indicator, period)
        self.cache[cache_key] = result
        return result
```

## 🚨 强制执行检查

### 代码生成后必须验证
- [ ] 遵循六层架构分层
- [ ] 使用依赖注入容器
- [ ] 扩展现有功能而非重复实现
- [ ] 包含完整异常处理
- [ ] 添加性能监控装饰器
- [ ] 符合命名规范
- [ ] 包含完整文档字符串
- [ ] 通过架构合规检查

### 违规处理
违反任何规则的代码都需要立即重构！系统必须保持：
1. 架构清晰，高内聚低耦合
2. 无重复建设，复用现有功能
3. 遵循数据库结构，统一查询标准
4. 注重性能，实现高效逻辑
5. 检查现有实现，优化而非重写

这些规则确保我们的105指标股票分析系统保持生产级别的代码质量和架构完整性。
description:
globs:
alwaysApply: true
---
