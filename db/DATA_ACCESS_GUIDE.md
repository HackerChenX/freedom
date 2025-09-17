# L3数据服务层 - 数据访问接口说明

## 🎯 标准入口 (强制使用)

### 数据访问接口: `db.interfaces.data_access_interface`
```python
from db.interfaces.data_access_interface import DataAccessInterface

# 通过依赖注入获取实现
from utils.unified_container import get_container
container = get_container()
data_access = container.resolve(DataAccessInterface)
```

### 数据访问管理器: `db.managers.data_access_manager`
```python
from db.managers.data_access_manager import DataAccessManager

# 直接实例化
data_manager = DataAccessManager()

# 获取股票数据
stock_data = data_manager.get_stock_data_data_access_interface(
    code="000001",
    start_date="2024-01-01", 
    end_date="2024-01-31"
)
```

## ❌ 已废弃的入口 (不要使用)

- ~~`db.data_access_manager`~~ - 已移除
- ~~`db.data_manager`~~ - 已移除  
- ~~`db.enhanced_data_manager`~~ - 已移除
- ~~`db.unified_data_manager`~~ - 已移除
- ~~`db.optimized_data_access_manager`~~ - 已移除
- ~~`db.interfaces.optimized_data_access_interface`~~ - 已移除
- ~~`db.interfaces.mock_data_access`~~ - 已移除

## 📋 使用建议

1. **新代码**: 统一使用 `db.interfaces.data_access_interface` 和 `db.managers.data_access_manager`
2. **依赖注入**: 推荐通过容器获取数据访问接口实例
3. **旧代码**: 逐步迁移到标准入口

## 🔍 标准API方法

### 核心数据访问方法
- `get_stock_data_data_access_interface()` - 获取单只股票数据
- `get_stocks_data_batch_data_access_interface()` - 批量获取多只股票数据
- `get_stock_list_data_access_interface()` - 获取股票列表
- `check_data_exists_data_access_interface()` - 检查数据是否存在
- `get_latest_data_data_access_interface()` - 获取最新数据

---
更新时间: 2024-09-16
维护者: AI Assistant
