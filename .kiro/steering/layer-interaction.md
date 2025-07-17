# 系统分层与交互规范

## 分层架构交互原则

### 1. 依赖方向规则

**严格的单向依赖**:
```
L6 (应用层) → L5 (业务逻辑层) → L4 (服务层) → L3 (数据访问层) → L2 (基础设施层) → L1 (数据层)
```

**允许的依赖关系**:
- ✅ 上层可以依赖下层
- ❌ 下层不能依赖上层
- ❌ 同层之间不能循环依赖
- ✅ 可以跨层依赖（如L6直接依赖L2）

### 2. 接口隔离原则

**每层只暴露必要的接口**:
```python
# L3层 - 数据访问层接口
from db.query_executor import get_query_executor
from db.sql_manager import QueryType

# L2层 - 基础设施层接口  
from utils.logger import get_logger
from config import get_config
```

## 各层交互规范

### L6 应用层 (Application Layer)

**职责**: 程序入口、脚本执行、用户交互
**目录**: [bin/](mdc:bin/), [scripts/](mdc:scripts/)

**依赖关系**:
```python
# 可以依赖的层
from analysis.buypoints import BuyPointAnalyzer          # L5
from strategy.momentum import MomentumStrategy           # L5
from indicators.technical import calculate_macd          # L4
from db.query_executor import get_query_executor         # L3
from utils.logger import get_logger                      # L2
from config import get_config                            # L2
```

**交互模式**:
```python
# bin/main.py - 主程序入口
def main():
    # 获取配置 (L2)
    config = get_config()
    logger = get_logger(__name__)
    
    # 初始化策略 (L5)
    strategy = MomentumStrategy(config.get('strategy', {}))
    
    # 执行分析 (L5)
    analyzer = BuyPointAnalyzer()
    results = analyzer.analyze(strategy.select_stocks())
    
    # 输出结果
    logger.info(f"分析完成，发现{len(results)}个买点")
```

### L5 业务逻辑层 (Business Logic Layer)

**职责**: 核心业务逻辑、策略算法、分析计算
**目录**: [analysis/](mdc:analysis/), [strategy/](mdc:strategy/), [formula/](mdc:formula/)

**依赖关系**:
```python
# 可以依赖的层
from indicators.technical import TechnicalIndicators     # L4
from db.query_executor import get_query_executor         # L3
from utils.date_utils import parse_date                  # L2
from enums.trend_types import TrendType                  # L2
```

**交互模式**:
```python
# strategy/momentum.py - 动量策略
class MomentumStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_executor = get_query_executor()  # L3
        self.logger = get_logger(__name__)          # L2
    
    def select_stocks(self) -> List[str]:
        # 获取股票列表 (L3)
        stocks = self.query_executor.get_stock_list()
        
        # 计算技术指标 (L4)
        indicators = TechnicalIndicators()
        
        selected = []
        for stock in stocks:
            # 获取股票数据 (L3)
            data = self.query_executor.get_stock_data({'code': stock})
            
            # 计算动量指标 (L4)
            momentum = indicators.calculate_momentum(data)
            
            # 业务逻辑判断 (L5)
            if self._is_momentum_strong(momentum):
                selected.append(stock)
        
        return selected
```

### L4 服务层 (Service Layer)

**职责**: 技术服务、指标计算、API接口、监控服务
**目录**: [indicators/](mdc:indicators/), [api/](mdc:api/), [monitoring/](mdc:monitoring/)

**依赖关系**:
```python
# 可以依赖的层
from db.query_executor import get_query_executor         # L3
from utils.cache import cache_result                     # L2
from utils.decorators import timing                      # L2
from enums.indicator_types import IndicatorType          # L2
```

**交互模式**:
```python
# indicators/technical.py - 技术指标服务
class TechnicalIndicators:
    def __init__(self):
        self.query_executor = get_query_executor()  # L3
        self.logger = get_logger(__name__)          # L2
    
    @cache_result(ttl=3600)  # L2装饰器
    def calculate_macd(self, code: str, period: int = 12) -> pd.DataFrame:
        """计算MACD指标"""
        # 获取数据 (L3)
        data = self.query_executor.get_stock_data({'code': code})
        
        # 指标计算 (L4业务逻辑)
        exp1 = data['close'].ewm(span=period).mean()
        exp2 = data['close'].ewm(span=period*2).mean()
        macd = exp1 - exp2
        
        return pd.DataFrame({
            'macd': macd,
            'signal': macd.ewm(span=9).mean(),
            'histogram': macd - macd.ewm(span=9).mean()
        })
```

### L3 数据访问层 (Data Access Layer)

**职责**: 数据库访问、查询执行、数据爬取
**目录**: [db/](mdc:db/), [crawler/](mdc:crawler/)

**依赖关系**:
```python
# 可以依赖的层
from utils.logger import get_logger                      # L2
from config import get_config                            # L2
from enums.kline_period import KlinePeriod              # L2
```

**交互模式**:
```python
# db/query_executor.py - 查询执行器
class QueryExecutor:
    def __init__(self):
        self.config = get_config()          # L2
        self.logger = get_logger(__name__)  # L2
        self.sql_manager = SQLManager()     # L3内部
    
    def execute_query(self, query_type: QueryType, params: Dict[str, Any]) -> pd.DataFrame:
        """执行查询"""
        try:
            # 获取SQL模板 (L3内部)
            sql_template = self.sql_manager.get_query_template(query_type)
            
            # 参数验证 (L3业务逻辑)
            validated_params = self._validate_params(params)
            
            # 执行查询 (L3核心功能)
            with self._get_connection() as conn:
                result = conn.query_dataframe(sql_template, validated_params)
            
            self.logger.info(f"查询成功，返回{len(result)}条记录")  # L2
            return result
            
        except Exception as e:
            self.logger.error(f"查询失败: {e}")  # L2
            raise
```

### L2 基础设施层 (Infrastructure Layer)

**职责**: 基础工具、配置管理、日志记录、枚举定义
**目录**: [config/](mdc:config/), [utils/](mdc:utils/), [enums/](mdc:enums/)

**依赖关系**:
```python
# 只能依赖标准库和第三方库
import os
import logging
from typing import Dict, Any
import yaml
import pandas as pd
```

**交互模式**:
```python
# utils/logger.py - 日志工具
def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # 基础设施层只依赖标准库
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger
```

### L1 数据层 (Data Layer)

**职责**: 数据存储、SQL脚本
**目录**: [data/](mdc:data/), [sql/](mdc:sql/)

**交互模式**:
- 被动存储，不包含业务逻辑
- 通过L3层访问，不直接被其他层访问

## 跨层交互规范

### 1. 服务注入模式

**依赖注入容器**:
```python
# utils/dependency_injection.py
class ServiceContainer:
    def __init__(self):
        self._services = {}
    
    def register(self, service_type: type, instance: Any):
        self._services[service_type] = instance
    
    def get(self, service_type: type):
        return self._services.get(service_type)

# 全局容器
container = ServiceContainer()
```

**服务注册**:
```python
# 在应用启动时注册服务
from db.query_executor import get_query_executor
from utils.logger import get_logger

container.register(QueryExecutor, get_query_executor())
container.register(logging.Logger, get_logger('app'))
```

### 2. 事件驱动模式

**事件定义**:
```python
# enums/events.py
class EventType(Enum):
    STOCK_DATA_UPDATED = "stock_data_updated"
    ANALYSIS_COMPLETED = "analysis_completed"
    STRATEGY_EXECUTED = "strategy_executed"
```

**事件处理**:
```python
# utils/event_bus.py
class EventBus:
    def __init__(self):
        self._handlers = {}
    
    def subscribe(self, event_type: EventType, handler: callable):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def publish(self, event_type: EventType, data: Any):
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            handler(data)
```

## 层间通信协议

### 1. 数据传输格式

**统一数据格式**:
```python
# 股票数据格式
StockData = pd.DataFrame  # columns: ['code', 'date', 'open', 'high', 'low', 'close', 'volume']

# 指标数据格式
IndicatorData = pd.DataFrame  # columns: ['date', 'value', 'signal']

# 分析结果格式
AnalysisResult = Dict[str, Any]  # {'code': str, 'signal': str, 'confidence': float}
```

### 2. 错误传播机制

**异常层次**:
```python
# 基础异常 (L2)
class BaseSystemException(Exception):
    pass

# 数据访问异常 (L3)
class DataAccessException(BaseSystemException):
    pass

# 业务逻辑异常 (L5)
class BusinessLogicException(BaseSystemException):
    pass
```

**异常处理策略**:
```python
def handle_layer_exception(func):
    """层间异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DataAccessException as e:
            logger.error(f"数据访问层错误: {e}")
            raise BusinessLogicException(f"业务处理失败: {e}")
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise
    return wrapper
```

## 性能优化策略

### 1. 层间缓存

**多级缓存策略**:
```python
# L4层 - 服务缓存
@cache_result(ttl=1800)  # 30分钟
def calculate_indicators(code: str) -> pd.DataFrame:
    pass

# L3层 - 数据缓存
@cache_result(ttl=300)   # 5分钟
def get_stock_data(params: Dict[str, Any]) -> pd.DataFrame:
    pass
```

### 2. 异步处理

**异步层间调用**:
```python
import asyncio

async def async_analysis_pipeline(stock_codes: List[str]) -> List[AnalysisResult]:
    """异步分析管道"""
    # L3层 - 异步获取数据
    data_tasks = [get_stock_data_async(code) for code in stock_codes]
    stock_data = await asyncio.gather(*data_tasks)
    
    # L4层 - 异步计算指标
    indicator_tasks = [calculate_indicators_async(data) for data in stock_data]
    indicators = await asyncio.gather(*indicator_tasks)
    
    # L5层 - 异步业务分析
    analysis_tasks = [analyze_stock_async(data, ind) for data, ind in zip(stock_data, indicators)]
    results = await asyncio.gather(*analysis_tasks)
    
    return results
```

### 3. 批量处理

**批量数据处理**:
```python
def batch_process_stocks(stock_codes: List[str], batch_size: int = 100) -> List[AnalysisResult]:
    """批量处理股票数据"""
    results = []
    
    for i in range(0, len(stock_codes), batch_size):
        batch = stock_codes[i:i + batch_size]
        
        # L3层 - 批量获取数据
        batch_data = query_executor.get_batch_stock_data(batch)
        
        # L4层 - 批量计算指标
        batch_indicators = calculate_batch_indicators(batch_data)
        
        # L5层 - 批量分析
        batch_results = analyze_batch_stocks(batch_data, batch_indicators)
        
        results.extend(batch_results)
    
    return results
```
alwaysApply: true
description: 股票分析系统分层交互和依赖管理
---
