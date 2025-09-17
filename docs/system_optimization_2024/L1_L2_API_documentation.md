# L1基础设施层和L2存储访问层API文档

## 📋 文档概览

本文档提供L1基础设施层和L2存储访问层的完整API参考，包括使用示例、最佳实践和故障排除指南。

**目标读者**: L3/L4/L5/L6层开发者  
**更新日期**: 2024-09-16  
**版本**: 1.0.0

---

## 🏗️ L1基础设施层API

### 1.1 依赖注入容器

#### 核心类: `UnifiedServiceContainer`

```python
from utils.unified_container import get_container, UnifiedServiceContainer, ServiceLifecycle

# 获取全局容器实例
container = get_container()

# 注册服务
container.register(
    service_type=IDataService,
    implementation_type=DataService,
    lifecycle=ServiceLifecycle.SINGLETON
)

# 注册工厂方法
container.register(
    service_type=IConnectionPool,
    factory=lambda: create_connection_pool(),
    lifecycle=ServiceLifecycle.SINGLETON
)

# 解析服务
data_service = container.resolve(IDataService)

# 检查服务是否已注册
if container.is_registered(IDataService):
    service = container.resolve(IDataService)
```

#### 最佳实践
```python
# ✅ 推荐：使用接口类型注册
from abc import ABC, abstractmethod

class IDataService(ABC):
    @abstractmethod
    def get_data(self) -> dict:
        pass

class DataService(IDataService):
    def get_data(self) -> dict:
        return {"data": "example"}

# 注册时使用接口类型
container.register_singleton(IDataService, DataService)

# ❌ 避免：直接使用具体类型
# container.register_singleton(DataService, DataService)
```

### 1.2 配置管理系统

#### 核心函数: `get_config()`, `get_config_manager()`

```python
from config.unified_config_manager import get_config, get_config_manager, set_config

# 获取配置值
database_host = get_config('database.host', 'localhost')
database_config = get_config('database')  # 获取整个database配置块
all_config = get_config()  # 获取所有配置

# 设置配置值
set_config('database.timeout', 30, persist=True)

# 获取配置管理器实例
config_manager = get_config_manager()
config_manager.reload_config()  # 重新加载配置
```

#### 配置文件结构
```yaml
# config/database.yaml
database:
  host: localhost
  port: 9000
  user: default
  password: ""
  database: stock_data
  connection_pool:
    min_connections: 5
    max_connections: 20
    timeout: 30

# config/application.yaml
system:
  name: stock-analysis-system
  version: 1.0.0
  environment: development

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### 最佳实践
```python
# ✅ 推荐：使用点号分隔的键名
host = get_config('database.host')
pool_size = get_config('database.connection_pool.max_connections', 20)

# ✅ 推荐：提供合理的默认值
timeout = get_config('database.timeout', 30)

# ❌ 避免：硬编码配置值
# host = "localhost"  # 应该从配置文件读取
```

### 1.3 日志系统

#### 核心函数: `get_logger()`, `init_logging()`

```python
from utils.logger import get_logger, init_logging

# 初始化日志系统（通常在应用启动时调用一次）
init_logging(level='INFO', to_console=True, to_file=True)

# 获取日志器
logger = get_logger(__name__)

# 使用日志器
logger.info("应用启动")
logger.warning("这是一个警告")
logger.error("发生错误", exc_info=True)

# 在类中使用
class DataService:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def process_data(self):
        self.logger.info("开始处理数据")
        try:
            # 处理逻辑
            pass
        except Exception as e:
            self.logger.error(f"数据处理失败: {e}", exc_info=True)
```

#### 最佳实践
```python
# ✅ 推荐：使用模块名作为日志器名称
logger = get_logger(__name__)

# ✅ 推荐：在异常处理中包含堆栈信息
try:
    risky_operation()
except Exception as e:
    logger.error(f"操作失败: {e}", exc_info=True)

# ✅ 推荐：使用适当的日志级别
logger.debug("调试信息")      # 详细的调试信息
logger.info("正常操作")       # 一般信息
logger.warning("警告信息")    # 警告但不影响运行
logger.error("错误信息")      # 错误信息
```

---

## 🗄️ L2存储访问层API

### 2.1 数据库连接池

#### 核心函数: `get_connection_pool()`

```python
from db.enhanced_connection_pool import get_connection_pool

# 获取连接池实例
pool = get_connection_pool()

# 使用连接执行查询
with pool.get_connection() as conn:
    result = conn.execute("SELECT COUNT(*) FROM stock_info")
    print(f"查询结果: {result}")

# 执行DataFrame查询
df = pool.query_dataframe("SELECT * FROM stock_info LIMIT 10")

# 获取连接池状态
status = pool.get_pool_status()
print(f"活跃连接: {status['active_connections']}")
print(f"总连接数: {status['total_connections']}")

# 并发查询（高性能）
result = pool.execute_concurrent_query(
    "SELECT * FROM stock_info WHERE code = %(code)s",
    {"code": "000001"}
)
```

#### 连接池配置
```python
# 连接池会自动从配置文件读取设置
# config/database.yaml
database:
  connection_pool:
    min_connections: 5      # 最小连接数
    max_connections: 20     # 最大连接数
    timeout: 30            # 连接超时时间
    health_check_interval: 60  # 健康检查间隔
```

#### 最佳实践
```python
# ✅ 推荐：使用上下文管理器
with pool.get_connection() as conn:
    result = conn.execute(query, params)
    # 连接会自动归还到池中

# ✅ 推荐：使用参数化查询
query = "SELECT * FROM stock_info WHERE code = %(code)s AND date = %(date)s"
params = {"code": "000001", "date": "2024-01-01"}
result = conn.execute(query, params)

# ❌ 避免：SQL注入风险
# query = f"SELECT * FROM stock_info WHERE code = '{code}'"  # 危险！
```

### 2.2 SQL查询管理

#### 核心类: `SQLManager`, `QueryType`

```python
from db.sql_manager import SQLManager, QueryType

# 创建SQL管理器实例
sql_manager = SQLManager()

# 获取预定义查询模板
stock_data_query = sql_manager.get_query(QueryType.STOCK_DATA)
stock_list_query = sql_manager.get_query(QueryType.STOCK_LIST)
stock_count_query = sql_manager.get_query(QueryType.STOCK_COUNT)

# 验证查询参数
params = {
    "code": "000001",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "level": "日线"
}

try:
    sql_manager.validate_params_sql_manager(QueryType.STOCK_DATA, params)
    print("参数验证通过")
except ValueError as e:
    print(f"参数验证失败: {e}")
```

#### 可用的查询类型
```python
# 主要查询类型
QueryType.STOCK_DATA          # 股票数据查询
QueryType.BATCH_STOCK_DATA    # 批量股票数据查询
QueryType.STOCK_LIST          # 股票列表查询
QueryType.STOCK_COUNT         # 股票数量查询
QueryType.DATE_RANGE          # 日期范围查询
QueryType.LATEST_DATA         # 最新数据查询
QueryType.PERFORMANCE_DATA    # 性能数据查询
QueryType.VALIDATION_DATA     # 验证数据查询

# 配置相关查询
QueryType.STRATEGY_CONFIG     # 策略配置查询
QueryType.SAVE_STRATEGY_CONFIG # 保存策略配置
```

#### 完整使用示例
```python
from db.enhanced_connection_pool import get_connection_pool
from db.sql_manager import SQLManager, QueryType

# 获取服务实例
pool = get_connection_pool()
sql_manager = SQLManager()

# 准备查询参数
params = {
    "code": "000001",
    "start_date": "2024-01-01", 
    "end_date": "2024-01-31",
    "level": "日线"
}

# 验证参数
sql_manager.validate_params_sql_manager(QueryType.STOCK_DATA, params)

# 获取查询模板
query_template = sql_manager.get_query(QueryType.STOCK_DATA)

# 执行查询
with pool.get_connection() as conn:
    # 方式1: 使用模板和参数
    result = conn.execute(query_template, params)
    
    # 方式2: 直接使用DataFrame查询
    df = conn.query_dataframe(query_template, params)
    
print(f"查询结果: {len(df)} 条记录")
```

---

## 🔗 3. 集成模式和最佳实践

### 3.1 标准集成模式

#### 服务初始化模式
```python
from utils.unified_container import get_container
from utils.logger import get_logger, init_logging
from config.unified_config_manager import get_config
from db.enhanced_connection_pool import get_connection_pool

class L3DataService:
    """L3数据服务层示例"""
    
    def __init__(self):
        # 初始化日志
        self.logger = get_logger(self.__class__.__name__)
        
        # 获取配置
        self.config = get_config('data_service', {})
        
        # 获取L2层服务
        self.connection_pool = get_connection_pool()
        self.sql_manager = SQLManager()
        
        # 注册到容器
        container = get_container()
        container.register_singleton(L3DataService, instance=self)
        
        self.logger.info("L3数据服务初始化完成")
    
    def get_stock_data(self, code: str, start_date: str, end_date: str):
        """获取股票数据"""
        try:
            # 准备参数
            params = {
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
                "level": "日线"
            }
            
            # 验证参数
            self.sql_manager.validate_params_sql_manager(QueryType.STOCK_DATA, params)
            
            # 获取查询
            query = self.sql_manager.get_query(QueryType.STOCK_DATA)
            
            # 执行查询
            with self.connection_pool.get_connection() as conn:
                df = conn.query_dataframe(query, params)
            
            self.logger.info(f"成功获取股票 {code} 的数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败: {e}", exc_info=True)
            raise
```

### 3.2 错误处理模式

#### 使用装饰器进行统一错误处理
```python
from utils.enhanced_exception_handler import exception_handler
from utils.enhanced_performance_monitor import performance_monitor

class DataProcessor:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def process_data(self, data_params: dict):
        """处理数据的标准模式"""
        self.logger.info("开始处理数据")
        
        # 业务逻辑
        result = self._do_processing(data_params)
        
        self.logger.info("数据处理完成")
        return result
    
    def _do_processing(self, params):
        # 具体的处理逻辑
        pass
```

### 3.3 配置驱动模式

#### 使用配置文件驱动服务行为
```python
class ConfigurableService:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        # 从配置文件读取服务配置
        self.config = get_config('services.data_processor', {
            'batch_size': 1000,
            'timeout': 30,
            'retry_count': 3,
            'cache_enabled': True
        })
        
        self.batch_size = self.config.get('batch_size', 1000)
        self.timeout = self.config.get('timeout', 30)
        self.retry_count = self.config.get('retry_count', 3)
        
        self.logger.info(f"服务配置: batch_size={self.batch_size}, timeout={self.timeout}")
```

---

## 🚨 4. 故障排除指南

### 4.1 常见问题和解决方案

#### 问题1: 容器服务未注册
```python
# 错误信息: ValueError: 服务 IDataService 未注册
# 解决方案:
container = get_container()
if not container.is_registered(IDataService):
    container.register_singleton(IDataService, DataService)
```

#### 问题2: 配置文件未找到
```python
# 错误信息: FileNotFoundError: config/database.yaml
# 解决方案: 确保配置文件存在并且路径正确
import os
config_path = "config/database.yaml"
if not os.path.exists(config_path):
    print(f"配置文件不存在: {config_path}")
    # 创建默认配置或检查路径
```

#### 问题3: 数据库连接失败
```python
# 错误信息: Connection refused
# 解决方案: 检查数据库配置和连接状态
pool = get_connection_pool()
status = pool.get_pool_status()
print(f"连接池状态: {status}")

# 检查配置
db_config = get_config('database')
print(f"数据库配置: {db_config}")
```

### 4.2 调试技巧

#### 启用详细日志
```python
from utils.logger import init_logging

# 启用DEBUG级别日志
init_logging(level='DEBUG', to_console=True, to_file=True)

# 查看容器状态
container = get_container()
print(f"已注册服务数量: {len(container._services)}")
for service_type in container._services:
    print(f"  - {service_type.__name__}")
```

#### 性能监控
```python
# 查看连接池性能
pool = get_connection_pool()
stats = pool.get_statistics()
print(f"连接池统计: {stats}")

# 查看缓存统计
cache_stats = pool.get_cache_stats()
print(f"缓存统计: {cache_stats}")
```

---

## 📚 5. 参考资料

### 5.1 相关文档
- [系统优化总体规划](./README.md)
- [架构修复指导原则](./architecture_repair_guidelines.md)
- [分层修复执行计划](./layered_repair_execution_plan.md)
- [L1/L2架构合规性审查报告](./L1_L2_architecture_compliance_audit.md)

### 5.2 代码示例
- [L1层测试示例](../../test_l1_comprehensive.py)
- [L2层测试示例](../../test_l2_comprehensive_validation.py)
- [集成测试示例](../../tests/integration/)

### 5.3 配置文件模板
- [数据库配置模板](../../config/database.yaml)
- [应用配置模板](../../config/application.yaml)

---

**文档维护**: 请确保在修改L1/L2层API时同步更新此文档
**反馈渠道**: 如有问题或建议，请在项目中提出issue

---

## 📋 API变更日志

### v1.0.0 (2024-09-16)
- 初始版本发布
- 完整的L1基础设施层API文档
- 完整的L2存储访问层API文档
- 集成模式和最佳实践指南
- 故障排除指南
