# L3数据服务层 - 缓存层使用指南

## 🎯 标准缓存入口 (强制使用)

### 缓存接口: `db.interfaces.cache_interface`
```python
from db.interfaces.cache_interface import ICacheService

# 通过依赖注入获取缓存服务
from utils.unified_container import get_container
container = get_container()
cache_service = container.resolve(ICacheService)
```

### 缓存服务: `db.services.cache_service`
```python
from db.services.cache_service import CacheService

# 直接实例化缓存服务
cache_service = CacheService()

# 基本缓存操作
cache_service.set("key", "value", ttl=300)
value = cache_service.get("key")
cache_service.delete("key")
cache_service.clear()
```

## 🔧 高级缓存功能

### 多层缓存支持
```python
from db.services.cache_service import CacheService, CacheLevel

cache_service = CacheService()

# 设置多层缓存
cache_service.set_multilevel(
    key="stock_data_000001",
    value=stock_data,
    ttl=300,
    levels=[CacheLevel.MEMORY, CacheLevel.DISK]
)

# 从多层缓存获取
data = cache_service.get_multilevel("stock_data_000001")
```

### 缓存装饰器
```python
from db.services.cache_service import cache_result

@cache_result(ttl=300, key_prefix="stock_data")
def get_stock_data(code: str, start_date: str, end_date: str):
    # 数据获取逻辑
    return data
```

## ❌ 已废弃的缓存入口 (不要使用)

- ~~`db.cache_layer`~~ - 已移除
- ~~`db.multi_layer_cache`~~ - 已移除
- ~~`db.query_cache`~~ - 已移除
- ~~`db.managers.cache_manager`~~ - 已移除

## 📋 缓存最佳实践

1. **统一入口**: 使用 `db.services.cache_service.CacheService`
2. **依赖注入**: 推荐通过容器获取缓存服务实例
3. **TTL设置**: 根据数据特性设置合适的过期时间
4. **多层缓存**: 对于重要数据使用内存+磁盘双层缓存
5. **缓存键**: 使用有意义的键名和前缀

## 🔍 标准缓存API

### 基本操作
- `set(key, value, ttl=None)` - 设置缓存
- `get(key, default=None)` - 获取缓存
- `delete(key)` - 删除缓存
- `clear()` - 清空缓存
- `exists(key)` - 检查键是否存在

### 批量操作
- `set_many(mapping, ttl=None)` - 批量设置
- `get_many(keys)` - 批量获取
- `delete_many(keys)` - 批量删除

### 统计信息
- `get_stats()` - 获取缓存统计
- `get_hit_rate()` - 获取命中率
- `get_size()` - 获取缓存大小

---
更新时间: 2024-09-16
维护者: AI Assistant
