---
type: "always_apply"
---

# 股票分析系统 Cursor AI 代码生成规则

## 核心架构规则（强制遵守）

### 1. 六层架构分层规则
```
L6: 用户接口层 (bin/, api/) → 只能调用 L5
L5: 业务应用层 (strategy/, analysis/) → 只能调用 L4  
L4: 核心服务层 (indicators/, formula/) → 只能调用 L3
L3: 数据服务层 (db/interfaces/, db/managers/) → 只能调用 L2
L2: 存储访问层 (db/clickhouse_db.py) → 只能调用 L1
L1: 基础设施层 (utils/, config/, enums/)
```

**严格禁止：**
- 跨层调用（如 L6 直接调用 L3）
- 下层依赖上层
- 业务层直接访问数据库

### 2. 禁止重复建设
- 新功能前必须检查现有实现
- 扩展现有功能，不创建重复代码
- 相似逻辑抽象为公共方法
- 禁止复制粘贴超过10行代码

### 3. 数据库访问规则
**数据库表结构：**
```sql
stock_info: (code, name, date, level, open, close, high, low, volume, turnover_rate, price_change, price_range, industry, datetime, seq)
```

**强制要求：**
- 查询必须包含 code 条件
- 查询必须包含 date 范围  
- 查询必须指定 level
- 禁止 SELECT *
- 必须使用 ORDER BY

**标准查询模板：**
```python
def get_stock_data(code: str, start_date: str, end_date: str, level: str = '日线'):
    return f"""
    SELECT code, name, date, open, high, low, close, volume, turnover_rate
    FROM stock_info 
    WHERE code = '{code}'
    AND level = '{level}'
    AND date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date ASC
    """
```

## 代码生成规范

### 1. 命名规范（强制执行）
- 类名：大驼峰命名 (`StockAnalyzer`)
- 方法名：小写+下划线 (`get_stock_data`)
- 变量名：小写+下划线 (`stock_code`)
- 常量名：大写+下划线 (`MAX_RETRY_COUNT`)
- 文件名：小写+下划线 (`stock_analyzer.py`)

### 2. 导入规范
```python
# 1. 标准库导入
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# 2. 第三方库导入  
import pandas as pd
import numpy as np

# 3. 项目内模块导入（绝对导入）
from config.config import get_config
from utils.logger import get_logger
from enums.kline_period import KlinePeriod

# 禁止：
# from db.clickhouse_db import *  # 通配符导入
# from ..relative_module import something  # 相对导入
# from db.clickhouse_db import get_clickhouse_db  # 直接数据库依赖
```

### 3. 类设计模板
```python
class StandardClassTemplate:
    """
    类的标准文档字符串
    
    Attributes:
        attribute_name (type): 属性描述
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        """
        Args:
            param1: 参数1描述
            param2: 参数2描述，可选
        """
        self.param1 = param1
        self.param2 = param2
        self._private_attr = None
    
    def public_method(self, arg1: str) -> Dict[str, Any]:
        """
        Args:
            arg1: 参数描述
            
        Returns:
            Dict[str, Any]: 返回值描述
            
        Raises:
            ValueError: 异常描述
        """
        try:
            result = self._private_method(arg1)
            return result
        except Exception as e:
            logger.error(f"方法执行失败: {e}")
            raise
```

### 4. 依赖注入规范
```python
# ✅ 正确做法
class BusinessService:
    def __init__(self):
        self.data_access = container.resolve(IDataAccess)
        self.logger = container.resolve(ILogger)

# ❌ 禁止做法  
class BadService:
    def __init__(self):
        self.db = get_clickhouse_db()  # 禁止直接依赖
```

### 5. 异常处理规范
```python
from functools import wraps

def exception_handler(reraise: bool = True, default_return=None):
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
class DataService:
    @exception_handler(reraise=True)
    def get_stock_data(self, code: str) -> pd.DataFrame:
        # 业务逻辑
        pass
```

### 6. 性能监控规范
```python
def performance_monitor(threshold_seconds: float = 1.0):
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

## 高性能要求

### 1. 数据库连接
- 使用连接池管理连接
- 实现查询结果缓存
- 使用批量查询减少网络开销
- 实现数据分页避免大结果集

### 2. 内存管理
- 大数据集使用生成器
- 及时释放DataFrame
- 使用适当数据类型
- 控制并发任务数量

### 3. 缓存策略
```python
from functools import lru_cache
from cachetools import TTLCache

class HighPerformanceDataAccess:
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=300)
    
    @lru_cache(maxsize=128)
    def get_stock_data(self, code: str, start_date: str, end_date: str):
        cache_key = f"{code}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self._query_database(code, start_date, end_date)
        self.cache[cache_key] = result
        return result
```

## 开发前检查清单

生成任何代码前必须检查：

1. **现有功能检查**
   ```python
   # 搜索相似实现
   search_patterns = [
       f"def.*{feature_name}",
       f"class.*{feature_name.title()}",
       f"# {feature_name}"
   ]
   ```

2. **架构合规检查**
   - 是否遵循分层架构
   - 是否使用依赖注入
   - 是否包含异常处理
   - 是否添加性能监控

3. **代码质量检查**
   - 命名是否符合规范
   - 文档是否完整
   - 是否包含测试

## 禁止行为

### 绝对禁止的代码模式：
```python
# ❌ 直接数据库依赖
from db.clickhouse_db import get_clickhouse_db

# ❌ 跨层调用
from analysis.market.analyzer import MarketAnalyzer  # 在 bin/ 中

# ❌ 通配符导入
from utils import *

# ❌ 相对导入
from ..parent_module import something

# ❌ 重复实现
class DuplicateIndicatorCalculator:  # 已存在相似类
    pass

# ❌ 硬编码数据库查询
result = db.query("SELECT * FROM stock_info")

# ❌ 无异常处理
def risky_method():
    # 可能出错的操作，但没有异常处理
    pass
```

### 必须遵循的模式：
```python
# ✅ 依赖注入
self.data_access = container.resolve(IDataAccess)

# ✅ 分层调用
from indicators.macd import MACDIndicator  # 在 strategy/ 中

# ✅ 异常处理
@exception_handler(reraise=True)
def safe_method(self):
    pass

# ✅ 性能监控
@performance_monitor(threshold_seconds=2.0)
def monitored_method(self):
    pass

# ✅ 标准查询
query = StandardQueries.get_stock_data(code, start_date, end_date)
```

## 代码生成后检查

生成代码后必须验证：

- [ ] 遵循六层架构分层
- [ ] 使用依赖注入
- [ ] 扩展现有功能而非重复实现
- [ ] 包含完整异常处理
- [ ] 添加性能监控装饰器
- [ ] 符合命名规范
- [ ] 包含完整文档字符串
- [ ] 通过架构合规检查

## 总结

严格遵循以上规则，确保生成的代码：
1. 架构清晰，高内聚低耦合
2. 无重复建设，复用现有功能
3. 遵循数据库结构，统一查询标准
4. 注重性能，实现高效逻辑
5. 检查现有实现，优化而非重写

违反任何规则的代码都需要立即重构！ # 股票分析系统 Cursor AI 代码生成规则

## 核心架构规则（强制遵守）

### 1. 六层架构分层规则
```
L6: 用户接口层 (bin/, api/) → 只能调用 L5
L5: 业务应用层 (strategy/, analysis/) → 只能调用 L4  
L4: 核心服务层 (indicators/, formula/) → 只能调用 L3
L3: 数据服务层 (db/interfaces/, db/managers/) → 只能调用 L2
L2: 存储访问层 (db/clickhouse_db.py) → 只能调用 L1
L1: 基础设施层 (utils/, config/, enums/)
```

**严格禁止：**
- 跨层调用（如 L6 直接调用 L3）
- 下层依赖上层
- 业务层直接访问数据库

### 2. 禁止重复建设
- 新功能前必须检查现有实现
- 扩展现有功能，不创建重复代码
- 相似逻辑抽象为公共方法
- 禁止复制粘贴超过10行代码

### 3. 数据库访问规则
**数据库表结构：**
```sql
stock_info: (code, name, date, level, open, close, high, low, volume, turnover_rate, price_change, price_range, industry, datetime, seq)
```

**强制要求：**
- 查询必须包含 code 条件
- 查询必须包含 date 范围  
- 查询必须指定 level
- 禁止 SELECT *
- 必须使用 ORDER BY

**标准查询模板：**
```python
def get_stock_data(code: str, start_date: str, end_date: str, level: str = '日线'):
    return f"""
    SELECT code, name, date, open, high, low, close, volume, turnover_rate
    FROM stock_info 
    WHERE code = '{code}'
    AND level = '{level}'
    AND date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date ASC
    """
```

## 代码生成规范

### 1. 命名规范（强制执行）
- 类名：大驼峰命名 (`StockAnalyzer`)
- 方法名：小写+下划线 (`get_stock_data`)
- 变量名：小写+下划线 (`stock_code`)
- 常量名：大写+下划线 (`MAX_RETRY_COUNT`)
- 文件名：小写+下划线 (`stock_analyzer.py`)

### 2. 导入规范
```python
# 1. 标准库导入
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# 2. 第三方库导入  
import pandas as pd
import numpy as np

# 3. 项目内模块导入（绝对导入）
from config.config import get_config
from utils.logger import get_logger
from enums.kline_period import KlinePeriod

# 禁止：
# from db.clickhouse_db import *  # 通配符导入
# from ..relative_module import something  # 相对导入
# from db.clickhouse_db import get_clickhouse_db  # 直接数据库依赖
```

### 3. 类设计模板
```python
class StandardClassTemplate:
    """
    类的标准文档字符串
    
    Attributes:
        attribute_name (type): 属性描述
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        """
        Args:
            param1: 参数1描述
            param2: 参数2描述，可选
        """
        self.param1 = param1
        self.param2 = param2
        self._private_attr = None
    
    def public_method(self, arg1: str) -> Dict[str, Any]:
        """
        Args:
            arg1: 参数描述
            
        Returns:
            Dict[str, Any]: 返回值描述
            
        Raises:
            ValueError: 异常描述
        """
        try:
            result = self._private_method(arg1)
            return result
        except Exception as e:
            logger.error(f"方法执行失败: {e}")
            raise
```

### 4. 依赖注入规范
```python
# ✅ 正确做法
class BusinessService:
    def __init__(self):
        self.data_access = container.resolve(IDataAccess)
        self.logger = container.resolve(ILogger)

# ❌ 禁止做法  
class BadService:
    def __init__(self):
        self.db = get_clickhouse_db()  # 禁止直接依赖
```

### 5. 异常处理规范
```python
from functools import wraps

def exception_handler(reraise: bool = True, default_return=None):
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
class DataService:
    @exception_handler(reraise=True)
    def get_stock_data(self, code: str) -> pd.DataFrame:
        # 业务逻辑
        pass
```

### 6. 性能监控规范
```python
def performance_monitor(threshold_seconds: float = 1.0):
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

## 高性能要求

### 1. 数据库连接
- 使用连接池管理连接
- 实现查询结果缓存
- 使用批量查询减少网络开销
- 实现数据分页避免大结果集

### 2. 内存管理
- 大数据集使用生成器
- 及时释放DataFrame
- 使用适当数据类型
- 控制并发任务数量

### 3. 缓存策略
```python
from functools import lru_cache
from cachetools import TTLCache

class HighPerformanceDataAccess:
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=300)
    
    @lru_cache(maxsize=128)
    def get_stock_data(self, code: str, start_date: str, end_date: str):
        cache_key = f"{code}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self._query_database(code, start_date, end_date)
        self.cache[cache_key] = result
        return result
```

## 开发前检查清单

生成任何代码前必须检查：

1. **现有功能检查**
   ```python
   # 搜索相似实现
   search_patterns = [
       f"def.*{feature_name}",
       f"class.*{feature_name.title()}",
       f"# {feature_name}"
   ]
   ```

2. **架构合规检查**
   - 是否遵循分层架构
   - 是否使用依赖注入
   - 是否包含异常处理
   - 是否添加性能监控

3. **代码质量检查**
   - 命名是否符合规范
   - 文档是否完整
   - 是否包含测试

## 禁止行为

### 绝对禁止的代码模式：
```python
# ❌ 直接数据库依赖
from db.clickhouse_db import get_clickhouse_db

# ❌ 跨层调用
from analysis.market.analyzer import MarketAnalyzer  # 在 bin/ 中

# ❌ 通配符导入
from utils import *

# ❌ 相对导入
from ..parent_module import something

# ❌ 重复实现
class DuplicateIndicatorCalculator:  # 已存在相似类
    pass

# ❌ 硬编码数据库查询
result = db.query("SELECT * FROM stock_info")

# ❌ 无异常处理
def risky_method():
    # 可能出错的操作，但没有异常处理
    pass
```

### 必须遵循的模式：
```python
# ✅ 依赖注入
self.data_access = container.resolve(IDataAccess)

# ✅ 分层调用
from indicators.macd import MACDIndicator  # 在 strategy/ 中

# ✅ 异常处理
@exception_handler(reraise=True)
def safe_method(self):
    pass

# ✅ 性能监控
@performance_monitor(threshold_seconds=2.0)
def monitored_method(self):
    pass

# ✅ 标准查询
query = StandardQueries.get_stock_data(code, start_date, end_date)
```

## 代码生成后检查

生成代码后必须验证：

- [ ] 遵循六层架构分层
- [ ] 使用依赖注入
- [ ] 扩展现有功能而非重复实现
- [ ] 包含完整异常处理
- [ ] 添加性能监控装饰器
- [ ] 符合命名规范
- [ ] 包含完整文档字符串
- [ ] 通过架构合规检查

## 总结

严格遵循以上规则，确保生成的代码：
1. 架构清晰，高内聚低耦合
2. 无重复建设，复用现有功能
3. 遵循数据库结构，统一查询标准
4. 注重性能，实现高效逻辑
5. 检查现有实现，优化而非重写

违反任何规则的代码都需要立即重构！ 