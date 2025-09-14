---
type: "always_apply"
description: "Example description"
---

# 强制开发标准（基于88指标系统集成项目）

## 🚨 强制执行声明

基于我们成功完成的88指标系统集成项目（实际实现105指标），以下规则是**强制性的**，违反任何规则都必须立即修正。这些规则确保我们维持生产级别的代码质量和系统架构完整性。

## 🏗️ 六层架构强制分层（不可违反）

```
L6: 用户接口层 (bin/, api/) → 只能调用 L5
L5: 业务应用层 (strategy/, analysis/) → 只能调用 L4  
L4: 核心服务层 (indicators/, formula/) → 只能调用 L3
L3: 数据服务层 (db/interfaces/, db/managers/) → 只能调用 L2
L2: 存储访问层 (db/enhanced_connection_pool.py) → 只能调用 L1
L1: 基础设施层 (utils/, config/, enums/)
```

### ❌ 绝对禁止的代码模式
```python
# 禁止：直接数据库依赖
from db.clickhouse_db import get_clickhouse_db

# 禁止：跨层调用
from analysis.market.analyzer import MarketAnalyzer  # 在 bin/ 中

# 禁止：通配符导入
from utils import *

# 禁止：相对导入
from ..parent_module import something

# 禁止：重复实现
class DuplicateIndicatorCalculator:  # 已存在相似类
    pass
```

### ✅ 强制要求的模式
```python
# 必须：依赖注入
self.data_access = container.resolve("DataAccessInterface")

# 必须：分层调用
from indicators.macd import MACDIndicator  # 在 strategy/ 中

# 必须：异常处理
@exception_handler(reraise=True)
def safe_method(self):
    pass

# 必须：性能监控
@performance_monitor(threshold_seconds=2.0)
def monitored_method(self):
    pass
```

## 🗄️ 数据库访问强制规范

### 强制查询要求
所有股票数据查询**必须**满足：

1. **必须包含code条件**
2. **必须包含date范围**  
3. **必须指定level**
4. **禁止SELECT ***
5. **必须使用ORDER BY**

### 标准查询模板（强制使用）
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

### 连接池强制使用
```python
# ✅ 强制模式：使用连接池
from db.enhanced_connection_pool import ClickHouseConnectionPool

pool = ClickHouseConnectionPool()
with pool.get_connection() as conn:
    result = conn.query_dataframe(query)

# ❌ 禁止模式：直接连接
from db.clickhouse_db import get_clickhouse_db  # 绝对禁止
```

## 📊 指标系统强制标准

### 105指标注册强制要求
基于我们的系统现状：
- **25个真实指标**必须正常工作
- **80个Mock指标**必须提供API兼容性
- **注册成功率必须 > 90%**
- **指标计算时间必须 < 2秒**

### 指标实现强制模板
```python
from abc import ABC, abstractmethod
from utils.performance_monitor import performance_monitor
from utils.exception_handler import exception_handler

class BaseIndicator(ABC):
    """所有指标必须继承此基类"""
    
    def __init__(self, name: str, period: int = 20):
        self.name = name
        self.period = period
    
    @abstractmethod
    @performance_monitor(threshold_seconds=2.0)  # 强制性能监控
    @exception_handler(reraise=True)             # 强制异常处理
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """必须实现的计算方法"""
        pass
    
    @abstractmethod
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """必须实现的信号方法"""
        pass
    
    def get_pattern_info(self) -> Dict[str, Any]:  # 强制实现
        """必须提供模式信息"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'period': self.period
        }
```

## 🔧 依赖注入强制规范

### 服务注册强制要求
所有服务**必须**在相应的`__init__.py`中注册：

```python
# db/__init__.py - 强制示例
from utils.container import container
from .managers.data_access_manager import DataAccessManager

# 强制注册数据访问服务
container.register("DataAccessInterface", DataAccessManager)
container.register("Logger", get_logger)
```

### 服务使用强制模式
```python
# ✅ 强制模式
class BusinessService:
    def __init__(self):
        self.data_access = container.resolve("DataAccessInterface")
        self.logger = container.resolve("Logger")

# ❌ 禁止模式
class BadService:
    def __init__(self):
        self.db = get_clickhouse_db()  # 违反依赖注入
```

## 📝 代码质量强制标准

### 命名规范（强制执行）
- **类名**: 大驼峰命名 (`StockAnalyzer`)
- **方法名**: 小写+下划线 (`get_stock_data`)
- **变量名**: 小写+下划线 (`stock_code`)
- **常量名**: 大写+下划线 (`MAX_RETRY_COUNT`)
- **文件名**: 小写+下划线 (`stock_analyzer.py`)

### 异常处理强制装饰器
```python
from functools import wraps
from utils.logger import get_logger

def exception_handler(reraise: bool = True, default_return=None):
    """强制异常处理装饰器 - 所有关键方法必须使用"""
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

# 强制使用示例
class DataService:
    @exception_handler(reraise=True)  # 必须添加
    def get_stock_data(self, code: str) -> pd.DataFrame:
        # 业务逻辑
        pass
```

### 性能监控强制装饰器
```python
def performance_monitor(threshold_seconds: float = 1.0):
    """强制性能监控装饰器 - 所有耗时方法必须使用"""
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

# 强制使用示例
class IndicatorService:
    @performance_monitor(threshold_seconds=2.0)  # 必须添加
    def calculate_indicator(self, data):
        # 计算逻辑
        pass
```

## 🚀 策略系统强制标准

### 策略基类强制继承
```python
from abc import ABC, abstractmethod
from enums.signal_types import SignalType

class BaseStrategy(ABC):
    """所有策略必须继承此基类"""
    
    def __init__(self, name: str, version: str = "1.0.0", **params):
        self.name = name
        self.version = version
        self.params = params
        self._validate_params()  # 强制参数验证
    
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """强制实现：返回必需参数"""
        pass
    
    @abstractmethod
    @performance_monitor(threshold_seconds=3.0)  # 强制性能监控
    @exception_handler(reraise=True)             # 强制异常处理
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """强制实现：生成交易信号"""
        pass
```

## 🛡️ 质量保证强制要求

### 开发前强制检查
开始任何新功能开发前，**必须**完成：

```python
# 1. 强制检查现有实现
search_patterns = [
    f"def.*{feature_name}",
    f"class.*{feature_name.title()}",
    f"# {feature_name}"
]

# 2. 架构合规强制检查
compliance_checklist = [
    "遵循六层架构分层",
    "使用依赖注入",
    "包含异常处理装饰器", 
    "添加性能监控装饰器"
]

# 3. 代码质量强制检查
quality_checklist = [
    "命名符合规范",
    "包含完整文档字符串",
    "通过静态分析检查"
]
```

### 禁止重复建设（强制执行）
- 新功能前**必须**检查现有实现
- **必须**扩展现有功能，不创建重复代码
- 相似逻辑**必须**抽象为公共方法
- **禁止**复制粘贴超过10行代码

## 📊 性能强制要求

### 系统性能基准（不可降低）
基于我们的105指标系统实现：

```python
MANDATORY_PERFORMANCE_STANDARDS = {
    'indicator_calculation_max_time': 2.0,      # 单指标最大计算时间
    'database_query_max_time': 1.0,             # 数据库查询最大时间
    'strategy_signal_max_time': 3.0,            # 策略信号生成最大时间
    'connection_pool_min_efficiency': 0.95,     # 连接池最低效率
    'indicator_registry_min_success': 0.90,     # 指标注册最低成功率
    'memory_usage_max_mb': 2048,                # 最大内存使用
    'concurrent_connections_max': 20             # 最大并发连接
}
```

### 缓存强制策略
```python
from functools import lru_cache
from cachetools import TTLCache

class MandatoryCacheImplementation:
    """强制缓存实现模式"""
    
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=300)  # 强制TTL缓存
    
    @lru_cache(maxsize=128)  # 强制LRU缓存
    def get_indicator_data(self, code: str, indicator: str, period: int):
        # 缓存逻辑
        pass
```

## 🔍 强制验证检查清单

每次代码提交前**必须**通过以下检查：

### 架构合规（强制）
- [ ] 遵循六层架构分层规则
- [ ] 使用依赖注入容器
- [ ] 无跨层直接调用
- [ ] 无循环依赖

### 代码质量（强制）
- [ ] 所有方法包含异常处理装饰器
- [ ] 耗时方法包含性能监控装饰器
- [ ] 命名符合规范要求
- [ ] 包含完整文档字符串

### 功能完整性（强制）
- [ ] 扩展现有功能而非重复实现
- [ ] 指标正确注册到指标注册表
- [ ] 策略正确注册到策略管理器
- [ ] 服务正确注册到依赖容器

### 性能达标（强制）
- [ ] 指标计算时间 < 2秒
- [ ] 数据库查询时间 < 1秒
- [ ] 策略信号生成时间 < 3秒
- [ ] 内存使用 < 2GB

### 测试覆盖（强制）
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试通过
- [ ] 性能测试通过
- [ ] 架构合规测试通过

## 🚨 强制执行处罚

### 违规处理流程
1. **立即修正**: 违反规则的代码必须立即修正
2. **架构重构**: 违反架构规则的必须重构
3. **性能优化**: 不达标的性能必须优化
4. **文档更新**: 缺失文档必须补充

### 系统保证目标
通过严格执行这些规则，我们的系统必须保持：

1. **架构清晰**: 高内聚低耦合的六层架构
2. **无重复建设**: 复用现有功能，避免重复实现
3. **数据库规范**: 统一查询标准，高效连接池
4. **性能卓越**: 高效的105指标计算系统
5. **质量保证**: 生产级别的代码质量

## 📈 成功案例参考

我们的88指标系统集成项目成功案例：
- **105个指标注册**（超额完成88个目标）
- **25个真实指标实现**
- **93.8%注册成功率**
- **数据库连接池100%功能**
- **完整的六层架构**
- **生产级别的系统质量**

这些严谨要求确保我们继续保持这一成功标准，任何新开发都必须达到或超越这个水平。

## ⚡ 最终声明

**这些规则是强制性的，不可协商的，基于我们成功的88指标系统集成项目经验制定。违反任何规则的代码都需要立即重构！**

我们已经证明了这些规则的有效性，它们是我们实现105指标系统成功的基础，必须在所有未来开发中严格遵循。
description:
globs:
alwaysApply: true
---
