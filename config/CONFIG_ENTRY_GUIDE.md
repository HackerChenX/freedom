# 配置管理入口说明

## 🎯 标准入口 (推荐使用)

### 主要入口: `config.unified_config_manager`
```python
from config.unified_config_manager import get_config, get_config_manager

# 获取配置值
database_host = get_config('database.host', 'localhost')
app_config = get_config('application')

# 获取配置管理器实例
config_manager = get_config_manager()
config_manager.reload_config()
```

## 🔧 兼容入口 (特定用途)

### 数据库配置: `config.database_config_manager`
```python
from config.database_config_manager import DatabaseConfigManager

# 仅用于数据库配置访问
db_manager = DatabaseConfigManager()
db_config = db_manager.get_config()
```

### 统一数据库配置: `config.unified_database_config`
```python
from config.unified_database_config import get_unified_database_config

# 获取统一数据库配置
db_config = get_unified_database_config()
```

## ❌ 已废弃的入口 (不要使用)

- ~~`config.__init__`~~ - 已移除
- ~~`config.config`~~ - 已移除

## 📋 使用建议

1. **新代码**: 统一使用 `config.unified_config_manager`
2. **数据库配置**: 使用 `config.database_config_manager` 或 `config.unified_database_config`
3. **旧代码**: 逐步迁移到标准入口

## 🔍 配置文件位置

- 主配置: `config/application.yaml`
- 数据库配置: `config/database.yaml`
- 其他配置: `config/` 目录下的相应文件

---
更新时间: 2024-09-16
维护者: AI Assistant
