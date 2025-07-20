# Freedom项目命名规范

## 📋 命名规范标准

### 1. 类名 (Class Names)
- **规范**: 使用 PascalCase (首字母大写的驼峰命名)
- **示例**: `StrategyExecutor`, `DataAccessInterface`, `IndicatorRegistry`
- **错误示例**: `Strategy_executor`, `data_access_interface`, `indicator_registry`

### 2. 函数名和方法名 (Function/Method Names)
- **规范**: 使用 snake_case (下划线分隔的小写命名)
- **示例**: `execute_strategy`, `get_stock_data`, `calculate_indicator`
- **错误示例**: `executeStrategy`, `getStockData`, `calculateIndicator`

### 3. 变量名 (Variable Names)
- **规范**: 使用 snake_case
- **示例**: `stock_code`, `indicator_value`, `strategy_config`
- **错误示例**: `stockCode`, `indicatorValue`, `strategyConfig`

### 4. 常量名 (Constants)
- **规范**: 使用 UPPER_CASE (全大写下划线分隔)
- **示例**: `MAX_WORKERS`, `DEFAULT_TIMEOUT`, `CACHE_SIZE`
- **错误示例**: `maxWorkers`, `defaultTimeout`, `cacheSize`

### 5. 文件名 (File Names)
- **规范**: 使用 snake_case
- **示例**: `strategy_executor.py`, `data_access_interface.py`
- **错误示例**: `strategyExecutor.py`, `DataAccessInterface.py`

### 6. 包和模块名 (Package/Module Names)
- **规范**: 使用 snake_case，简短且描述性
- **示例**: `strategy`, `indicators`, `db`
- **错误示例**: `Strategy`, `Indicators`, `DB`

### 7. 异常类名 (Exception Names)
- **规范**: 使用 PascalCase，以 Error 结尾
- **示例**: `StrategyExecutionError`, `DataAccessError`, `IndicatorNotFoundError`
- **错误示例**: `Strategy_execution_error`, `DataAccess_Error`, `indicator_not_found_error`

### 8. 接口名 (Interface Names)
- **规范**: 使用 PascalCase，通常以 Interface 结尾
- **示例**: `DataAccessInterface`, `IndicatorInterface`, `CacheInterface`
- **错误示例**: `Data_access_interface`, `indicator_interface`

### 9. 枚举名 (Enum Names)
- **规范**: 类名使用 PascalCase，成员使用 UPPER_CASE
- **示例**: 
  ```python
  class QueryType(Enum):
      SELECT = "select"
      INSERT = "insert"
      UPDATE = "update"
  ```

### 10. 装饰器名 (Decorator Names)
- **规范**: 使用 snake_case
- **示例**: `@performance_monitor`, `@exception_handler`, `@cache_result`
- **错误示例**: `@performanceMonitor`, `@exceptionHandler`

## 🔧 需要修复的常见问题

### 1. 类名不一致
```python
# 错误
class Strategy_executor:
class data_access_interface:

# 正确
class StrategyExecutor:
class DataAccessInterface:
```

### 2. 异常类名不一致
```python
# 错误
class Strategy_execution_error(Exception):
class Indicator_not_found_error(Exception):

# 正确
class StrategyExecutionError(Exception):
class IndicatorNotFoundError(Exception):
```

### 3. 函数名不一致
```python
# 错误
def executeStrategy():
def getStockData():

# 正确
def execute_strategy():
def get_stock_data():
```

### 4. 导入语句不一致
```python
# 错误
from strategy.strategy_executor import Strategy_executor
from utils.exceptions import Strategy_execution_error

# 正确
from strategy.strategy_executor import StrategyExecutor
from utils.exceptions import StrategyExecutionError
```

## 📊 待修复的文件清单

### 高优先级文件 (核心功能)
1. `strategy/strategy_executor.py`
2. `strategy/strategy_manager.py`
3. `strategy/strategy_parser.py`
4. `db/interfaces/data_access_interface.py`
5. `utils/exceptions.py`
6. `indicators/complete_indicator_registry.py`

### 中优先级文件 (支持功能)
1. `strategy/signal_watcher.py`
2. `strategy/result_filter.py`
3. `db/unified_data_manager.py`
4. `utils/decorators.py`
5. `utils/logger.py`

### 低优先级文件 (辅助功能)
1. `analysis/` 目录下的文件
2. `scripts/` 目录下的文件
3. `tests/` 目录下的文件

## 🎯 修复计划

### 阶段1: 核心类和异常
1. 修复异常类名和导入
2. 修复核心类名和导入
3. 更新所有引用

### 阶段2: 函数和方法
1. 标准化函数命名
2. 更新函数调用
3. 修复装饰器使用

### 阶段3: 变量和常量
1. 标准化变量命名
2. 定义项目常量
3. 清理临时变量

### 阶段4: 验证和测试
1. 运行基本导入测试
2. 验证核心功能
3. 修复发现的问题

## 📝 检查清单

- [ ] 所有类名使用 PascalCase
- [ ] 所有函数名使用 snake_case
- [ ] 所有变量名使用 snake_case
- [ ] 所有常量使用 UPPER_CASE
- [ ] 所有异常类以 Error 结尾
- [ ] 所有导入语句正确
- [ ] 无循环导入
- [ ] 无缺失的类定义
- [ ] 所有装饰器使用正确参数
- [ ] 基本功能测试通过