# L1基础设施层和L2存储访问层集成指南

## 📋 指南概览

本指南提供L1基础设施层和L2存储访问层的详细集成指南，帮助L3/L4/L5/L6层开发者快速、正确地集成底层服务。

**适用场景**: 新服务开发、现有服务重构、跨层集成  
**目标读者**: L3/L4/L5/L6层开发者  
**更新日期**: 2024-09-16

---

## 🚀 快速开始

### 1分钟快速集成

```python
# 1. 导入必要的模块
from utils.unified_container import get_container
from utils.logger import get_logger, init_logging
from config.unified_config_manager import get_config
from db.enhanced_connection_pool import get_connection_pool
from db.sql_manager import SQLManager, QueryType

# 2. 初始化服务
def initialize_service():
    # 初始化日志系统
    init_logging(level='INFO')
    logger = get_logger(__name__)
    
    # 获取配置
    config = get_config()
    
    # 获取数据库服务
    pool = get_connection_pool()
    sql_manager = SQLManager()
    
    logger.info("服务初始化完成")
    return pool, sql_manager, logger

# 3. 使用服务
pool, sql_manager, logger = initialize_service()

# 查询股票数据
query = sql_manager.get_query(QueryType.STOCK_DATA)
with pool.get_connection() as conn:
    df = conn.query_dataframe(query, {
        "code": "000001",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31", 
        "level": "日线"
    })

logger.info(f"获取到 {len(df)} 条股票数据")
```

---

## 🏗️ 详细集成步骤

### 步骤1: 项目结构准备

#### 1.1 确保配置文件存在
```bash
# 检查必要的配置文件
config/
├── database.yaml          # 数据库配置
├── application.yaml        # 应用配置
└── logging.yaml           # 日志配置（可选）
```

#### 1.2 配置文件示例
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
  name: my-service
  version: 1.0.0
  environment: development

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 步骤2: 服务类设计模式

#### 2.1 标准服务类模板
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd

from utils.unified_container import get_container
from utils.logger import get_logger
from config.unified_config_manager import get_config
from db.enhanced_connection_pool import get_connection_pool
from db.sql_manager import SQLManager, QueryType
from utils.enhanced_exception_handler import exception_handler
from utils.enhanced_performance_monitor import performance_monitor

class BaseDataService(ABC):
    """数据服务基类 - 标准集成模式"""
    
    def __init__(self, service_name: str):
        # 1. 初始化日志
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # 2. 加载配置
        self.service_name = service_name
        self.config = get_config(f'services.{service_name}', {})
        
        # 3. 获取L1/L2层服务
        self.container = get_container()
        self.connection_pool = get_connection_pool()
        self.sql_manager = SQLManager()
        
        # 4. 初始化服务特定配置
        self._initialize_service_config()
        
        # 5. 注册到容器
        self._register_to_container()
        
        self.logger.info(f"{service_name} 服务初始化完成")
    
    def _initialize_service_config(self):
        """初始化服务特定配置"""
        self.batch_size = self.config.get('batch_size', 1000)
        self.timeout = self.config.get('timeout', 30)
        self.cache_enabled = self.config.get('cache_enabled', True)
    
    def _register_to_container(self):
        """注册服务到容器"""
        if not self.container.is_registered(self.__class__):
            self.container.register_singleton(self.__class__, instance=self)
    
    @abstractmethod
    def process_data(self, params: Dict[str, Any]) -> Any:
        """抽象方法：处理数据"""
        pass
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def execute_query(self, query_type: QueryType, params: Dict[str, Any]) -> pd.DataFrame:
        """执行数据库查询的标准方法"""
        try:
            # 1. 验证参数
            self.sql_manager.validate_params_sql_manager(query_type, params)
            
            # 2. 获取查询模板
            query = self.sql_manager.get_query(query_type)
            
            # 3. 执行查询
            with self.connection_pool.get_connection() as conn:
                df = conn.query_dataframe(query, params)
            
            self.logger.info(f"查询执行成功，返回 {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.logger.error(f"查询执行失败: {e}", exc_info=True)
            raise
```

#### 2.2 具体服务实现示例
```python
class StockDataService(BaseDataService):
    """股票数据服务 - 具体实现示例"""
    
    def __init__(self):
        super().__init__('stock_data')
        
        # 服务特定的初始化
        self.supported_levels = ['日线', '周线', '月线', '15分钟', '30分钟', '60分钟']
    
    def process_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """处理股票数据"""
        return self.get_stock_data(
            code=params['code'],
            start_date=params['start_date'],
            end_date=params['end_date'],
            level=params.get('level', '日线')
        )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def get_stock_data(self, code: str, start_date: str, end_date: str, level: str = '日线') -> pd.DataFrame:
        """获取股票数据"""
        if level not in self.supported_levels:
            raise ValueError(f"不支持的数据级别: {level}")
        
        params = {
            'code': code,
            'start_date': start_date,
            'end_date': end_date,
            'level': level
        }
        
        return self.execute_query(QueryType.STOCK_DATA, params)
    
    @exception_handler(reraise=True)
    def get_stock_list(self, level: str = '日线', limit: int = 100) -> List[str]:
        """获取股票列表"""
        params = {'level': level, 'limit': limit}
        df = self.execute_query(QueryType.STOCK_LIST, params)
        return df['code'].tolist()
    
    @exception_handler(reraise=True)
    def get_stock_count(self, level: str = '日线') -> int:
        """获取股票数量"""
        params = {'level': level}
        df = self.execute_query(QueryType.STOCK_COUNT, params)
        return df.iloc[0]['total_stocks']
```

### 步骤3: 应用启动模式

#### 3.1 应用启动器
```python
from utils.logger import init_logging
from utils.unified_container import get_container

class ApplicationBootstrap:
    """应用启动器 - 标准启动模式"""
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.container = None
        self.logger = None
    
    def initialize(self):
        """初始化应用"""
        # 1. 初始化日志系统
        init_logging(level='INFO', to_console=True, to_file=True)
        self.logger = get_logger(self.app_name)
        
        # 2. 获取容器
        self.container = get_container()
        
        # 3. 注册核心服务
        self._register_core_services()
        
        # 4. 注册业务服务
        self._register_business_services()
        
        self.logger.info(f"{self.app_name} 应用初始化完成")
    
    def _register_core_services(self):
        """注册核心服务"""
        # L1/L2层服务会自动注册，这里可以注册额外的核心服务
        pass
    
    def _register_business_services(self):
        """注册业务服务"""
        # 注册具体的业务服务
        stock_service = StockDataService()
        # 其他业务服务...
    
    def get_service(self, service_type):
        """获取服务实例"""
        return self.container.resolve(service_type)
    
    def shutdown(self):
        """关闭应用"""
        self.logger.info(f"{self.app_name} 应用正在关闭...")
        # 清理资源
        if self.container:
            # 执行清理逻辑
            pass
        self.logger.info(f"{self.app_name} 应用已关闭")

# 使用示例
if __name__ == "__main__":
    app = ApplicationBootstrap("stock-analysis-app")
    
    try:
        # 初始化应用
        app.initialize()
        
        # 获取服务并使用
        stock_service = app.get_service(StockDataService)
        data = stock_service.get_stock_data("000001", "2024-01-01", "2024-01-31")
        print(f"获取到 {len(data)} 条数据")
        
    except Exception as e:
        print(f"应用运行失败: {e}")
    finally:
        app.shutdown()
```

---

## 🔧 高级集成模式

### 模式1: 依赖注入模式

```python
from typing import Protocol

# 1. 定义接口
class IStockDataProvider(Protocol):
    def get_stock_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        ...

class IConfigProvider(Protocol):
    def get_config(self, key: str, default: Any = None) -> Any:
        ...

# 2. 实现服务
class AdvancedStockService:
    def __init__(self, 
                 data_provider: IStockDataProvider,
                 config_provider: IConfigProvider):
        self.data_provider = data_provider
        self.config_provider = config_provider
        self.logger = get_logger(self.__class__.__name__)
    
    def analyze_stock(self, code: str):
        # 从配置获取分析参数
        analysis_period = self.config_provider.get_config('analysis.period', 30)
        
        # 获取数据
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=analysis_period)).strftime('%Y-%m-%d')
        
        data = self.data_provider.get_stock_data(code, start_date, end_date)
        
        # 执行分析
        return self._perform_analysis(data)

# 3. 注册和使用
container = get_container()
container.register_singleton(IStockDataProvider, StockDataService)
container.register_singleton(IConfigProvider, lambda: get_config)

# 自动解析依赖
advanced_service = container.resolve(AdvancedStockService)
```

### 模式2: 工厂模式

```python
class ServiceFactory:
    """服务工厂 - 统一创建服务实例"""
    
    @staticmethod
    def create_stock_service(service_type: str = 'default') -> BaseDataService:
        """创建股票服务"""
        if service_type == 'default':
            return StockDataService()
        elif service_type == 'cached':
            return CachedStockDataService()
        elif service_type == 'batch':
            return BatchStockDataService()
        else:
            raise ValueError(f"不支持的服务类型: {service_type}")
    
    @staticmethod
    def create_analysis_service(config: Dict[str, Any]) -> 'AnalysisService':
        """创建分析服务"""
        service_type = config.get('type', 'basic')
        
        if service_type == 'basic':
            return BasicAnalysisService(config)
        elif service_type == 'advanced':
            return AdvancedAnalysisService(config)
        else:
            raise ValueError(f"不支持的分析服务类型: {service_type}")

# 使用工厂
stock_service = ServiceFactory.create_stock_service('cached')
analysis_config = get_config('analysis', {})
analysis_service = ServiceFactory.create_analysis_service(analysis_config)
```

---

## 🚨 常见问题和解决方案

### 问题1: 服务初始化顺序问题

**问题描述**: 服务A依赖服务B，但服务B还未初始化

**解决方案**:
```python
class ServiceInitializer:
    """服务初始化器 - 管理初始化顺序"""
    
    def __init__(self):
        self.container = get_container()
        self.initialized_services = set()
    
    def initialize_services(self):
        """按正确顺序初始化服务"""
        # 1. 初始化基础服务
        self._init_base_services()
        
        # 2. 初始化数据服务
        self._init_data_services()
        
        # 3. 初始化业务服务
        self._init_business_services()
    
    def _init_base_services(self):
        """初始化基础服务"""
        if 'base' not in self.initialized_services:
            # 基础服务初始化
            init_logging()
            get_connection_pool()  # 确保连接池初始化
            self.initialized_services.add('base')
    
    def _init_data_services(self):
        """初始化数据服务"""
        if 'data' not in self.initialized_services:
            self._init_base_services()  # 确保基础服务已初始化
            StockDataService()
            self.initialized_services.add('data')
    
    def _init_business_services(self):
        """初始化业务服务"""
        if 'business' not in self.initialized_services:
            self._init_data_services()  # 确保数据服务已初始化
            # 初始化业务服务
            self.initialized_services.add('business')
```

### 问题2: 配置文件找不到

**问题描述**: 应用启动时找不到配置文件

**解决方案**:
```python
import os
from pathlib import Path

def ensure_config_files():
    """确保配置文件存在"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # 检查必要的配置文件
    required_configs = {
        'database.yaml': {
            'database': {
                'host': 'localhost',
                'port': 9000,
                'user': 'default',
                'password': '',
                'database': 'stock_data'
            }
        },
        'application.yaml': {
            'system': {
                'name': 'stock-analysis',
                'version': '1.0.0'
            }
        }
    }
    
    for config_file, default_content in required_configs.items():
        config_path = config_dir / config_file
        if not config_path.exists():
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_content, f, default_flow_style=False, allow_unicode=True)
            print(f"创建默认配置文件: {config_path}")

# 在应用启动前调用
ensure_config_files()
```

### 问题3: 内存泄漏问题

**问题描述**: 长时间运行后内存使用量持续增长

**解决方案**:
```python
import gc
import weakref
from contextlib import contextmanager

class ResourceManager:
    """资源管理器 - 防止内存泄漏"""
    
    def __init__(self):
        self.active_connections = weakref.WeakSet()
        self.active_services = weakref.WeakSet()
    
    @contextmanager
    def managed_connection(self):
        """管理数据库连接"""
        pool = get_connection_pool()
        conn = pool.get_connection()
        self.active_connections.add(conn)
        
        try:
            yield conn
        finally:
            # 确保连接被正确释放
            if hasattr(conn, 'close'):
                conn.close()
            # 从活跃连接中移除
            self.active_connections.discard(conn)
    
    def cleanup_resources(self):
        """清理资源"""
        # 强制垃圾回收
        gc.collect()
        
        # 检查活跃连接
        active_count = len(self.active_connections)
        if active_count > 0:
            print(f"警告: 仍有 {active_count} 个活跃连接")
    
    def get_resource_stats(self):
        """获取资源统计"""
        return {
            'active_connections': len(self.active_connections),
            'active_services': len(self.active_services)
        }

# 使用资源管理器
resource_manager = ResourceManager()

with resource_manager.managed_connection() as conn:
    # 使用连接
    result = conn.execute("SELECT COUNT(*) FROM stock_info")
```

---

## 📚 完整示例项目

### 项目结构
```
my_stock_service/
├── config/
│   ├── database.yaml
│   └── application.yaml
├── services/
│   ├── __init__.py
│   ├── base_service.py
│   ├── stock_service.py
│   └── analysis_service.py
├── main.py
└── requirements.txt
```

### 完整的main.py示例
```python
#!/usr/bin/env python3
"""
股票分析服务主程序
演示L1/L2层集成的完整示例
"""

import sys
import signal
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import init_logging, get_logger
from utils.unified_container import get_container
from config.unified_config_manager import get_config
from services.stock_service import StockDataService
from services.analysis_service import StockAnalysisService

class StockAnalysisApplication:
    """股票分析应用主类"""
    
    def __init__(self):
        self.logger = None
        self.container = None
        self.services = {}
        self.running = False
    
    def initialize(self):
        """初始化应用"""
        try:
            # 1. 初始化日志系统
            init_logging(level='INFO', to_console=True, to_file=True)
            self.logger = get_logger('StockAnalysisApp')
            self.logger.info("开始初始化股票分析应用...")
            
            # 2. 加载配置
            config = get_config()
            self.logger.info(f"配置加载完成: {config.get('system', {}).get('name', 'Unknown')}")
            
            # 3. 获取容器
            self.container = get_container()
            
            # 4. 初始化服务
            self._initialize_services()
            
            # 5. 设置信号处理
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info("股票分析应用初始化完成")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"应用初始化失败: {e}", exc_info=True)
            else:
                print(f"应用初始化失败: {e}")
            raise
    
    def _initialize_services(self):
        """初始化服务"""
        # 初始化股票数据服务
        self.services['stock_data'] = StockDataService()
        
        # 初始化分析服务
        self.services['analysis'] = StockAnalysisService()
        
        self.logger.info(f"已初始化 {len(self.services)} 个服务")
    
    def run(self):
        """运行应用"""
        self.running = True
        self.logger.info("股票分析应用开始运行...")
        
        try:
            # 示例：执行一些分析任务
            self._run_analysis_tasks()
            
        except Exception as e:
            self.logger.error(f"应用运行出错: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def _run_analysis_tasks(self):
        """运行分析任务"""
        stock_service = self.services['stock_data']
        analysis_service = self.services['analysis']
        
        # 获取股票列表
        stock_codes = stock_service.get_stock_list(limit=10)
        self.logger.info(f"获取到 {len(stock_codes)} 只股票")
        
        # 分析每只股票
        for code in stock_codes:
            if not self.running:
                break
                
            try:
                # 获取股票数据
                data = stock_service.get_stock_data(
                    code=code,
                    start_date="2024-01-01",
                    end_date="2024-01-31"
                )
                
                # 执行分析
                result = analysis_service.analyze_stock(code, data)
                self.logger.info(f"股票 {code} 分析完成: {result}")
                
            except Exception as e:
                self.logger.error(f"分析股票 {code} 失败: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"接收到信号 {signum}，准备关闭应用...")
        self.running = False
    
    def shutdown(self):
        """关闭应用"""
        self.logger.info("正在关闭股票分析应用...")
        
        # 清理服务
        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'shutdown'):
                    service.shutdown()
                self.logger.info(f"服务 {service_name} 已关闭")
            except Exception as e:
                self.logger.error(f"关闭服务 {service_name} 失败: {e}")
        
        self.logger.info("股票分析应用已关闭")

def main():
    """主函数"""
    app = StockAnalysisApplication()
    
    try:
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        print("\n用户中断，正在关闭...")
    except Exception as e:
        print(f"应用运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## 📋 集成检查清单

### 开发前检查
- [ ] 配置文件已准备 (database.yaml, application.yaml)
- [ ] 项目依赖已安装
- [ ] 数据库连接已测试
- [ ] 日志目录权限正确

### 开发中检查
- [ ] 服务类继承自BaseDataService
- [ ] 使用了@exception_handler装饰器
- [ ] 使用了@performance_monitor装饰器
- [ ] 正确使用了依赖注入容器
- [ ] 配置通过get_config()获取

### 部署前检查
- [ ] 所有服务正确注册到容器
- [ ] 日志输出正常
- [ ] 性能指标符合要求
- [ ] 错误处理覆盖完整
- [ ] 资源清理机制正常

---

**维护说明**: 本指南会随着L1/L2层API的更新而同步更新  
**技术支持**: 如遇到集成问题，请参考故障排除部分或提交issue
