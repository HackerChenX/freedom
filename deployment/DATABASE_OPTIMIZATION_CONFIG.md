# 数据库优化功能配置参数说明

## 📋 配置概述

本文档详细说明数据库优化功能的所有配置参数，包括推荐值、调优指南和环境适配建议。

---

## 🔧 连接池配置参数

### EnhancedConnectionPool 配置

#### 基础连接参数
```python
# 数据库连接配置
HOST = 'localhost'                    # ClickHouse服务器地址
PORT = 9000                          # ClickHouse端口 (TCP协议)
DATABASE = 'stock'                   # 数据库名称
USER = 'default'                     # 用户名
PASSWORD = ''                        # 密码 (如有)
```

#### 连接池大小配置
```python
MAX_CONNECTIONS = 20                 # 最大连接数
MIN_CONNECTIONS = 5                  # 最小连接数
```

**调优建议**:
- **小型环境** (< 10并发用户): MAX=10, MIN=3
- **中型环境** (10-50并发用户): MAX=20, MIN=5 (推荐)
- **大型环境** (> 50并发用户): MAX=50, MIN=10
- **超大型环境** (> 100并发用户): MAX=100, MIN=20

#### 连接管理配置
```python
MAX_IDLE_TIME = 300                  # 连接最大空闲时间 (秒)
HEALTH_CHECK_INTERVAL = 60           # 健康检查间隔 (秒)
CONNECTION_TIMEOUT = 30              # 连接超时时间 (秒)
QUERY_TIMEOUT = 300                  # 查询超时时间 (秒)
```

**调优建议**:
- **高频访问环境**: MAX_IDLE_TIME=600, HEALTH_CHECK_INTERVAL=30
- **低频访问环境**: MAX_IDLE_TIME=180, HEALTH_CHECK_INTERVAL=120
- **网络不稳定环境**: CONNECTION_TIMEOUT=60, QUERY_TIMEOUT=600

---

## 💾 查询缓存配置参数

### QueryCache 配置

#### 内存缓存配置
```python
MAX_MEMORY_SIZE = 1000               # 内存缓存最大条目数
CACHE_ENABLED = True                 # 是否启用缓存
DEFAULT_TTL = 1800                   # 默认缓存有效期 (秒)
```

**调优建议**:
- **内存充足环境**: MAX_MEMORY_SIZE=2000, DEFAULT_TTL=3600
- **内存受限环境**: MAX_MEMORY_SIZE=500, DEFAULT_TTL=900
- **实时性要求高**: DEFAULT_TTL=300
- **实时性要求低**: DEFAULT_TTL=7200

#### 磁盘缓存配置
```python
ENABLE_DISK_CACHE = True             # 是否启用磁盘缓存
MAX_DISK_SIZE = 5000                 # 磁盘缓存最大条目数
CACHE_DIR = 'cache'                  # 缓存目录路径
```

**调优建议**:
- **SSD存储**: MAX_DISK_SIZE=10000, 启用磁盘缓存
- **HDD存储**: MAX_DISK_SIZE=3000, 谨慎启用磁盘缓存
- **存储受限**: ENABLE_DISK_CACHE=False

#### 缓存策略配置
```python
LRU_ENABLED = True                   # 启用LRU策略
PRELOAD_ENABLED = False              # 是否启用预加载
COMPRESSION_ENABLED = False          # 是否启用压缩
```

---

## 📊 性能监控配置参数

### PerformanceMonitor 配置

#### 数据收集配置
```python
COLLECTION_INTERVAL = 10             # 数据收集间隔 (秒)
RETENTION_HOURS = 24                 # 数据保留时间 (小时)
ENABLE_ALERTS = True                 # 是否启用告警
```

**调优建议**:
- **详细监控**: COLLECTION_INTERVAL=5, RETENTION_HOURS=72
- **轻量监控**: COLLECTION_INTERVAL=30, RETENTION_HOURS=12
- **生产环境**: COLLECTION_INTERVAL=10, RETENTION_HOURS=48

#### 告警阈值配置
```python
# CPU使用率告警
CPU_WARNING_THRESHOLD = 80.0         # CPU警告阈值 (%)
CPU_CRITICAL_THRESHOLD = 95.0        # CPU严重告警阈值 (%)

# 内存使用率告警
MEMORY_WARNING_THRESHOLD = 85.0      # 内存警告阈值 (%)
MEMORY_CRITICAL_THRESHOLD = 95.0     # 内存严重告警阈值 (%)

# 查询性能告警
QUERY_TIME_WARNING = 5.0             # 查询时间警告阈值 (秒)
QUERY_TIME_CRITICAL = 10.0           # 查询时间严重告警阈值 (秒)

# 连接池使用率告警
POOL_USAGE_WARNING = 90.0            # 连接池使用率警告阈值 (%)

# 缓存命中率告警
CACHE_HIT_RATE_WARNING = 0.5         # 缓存命中率警告阈值 (50%)
```

**调优建议**:
- **高性能要求**: 降低所有阈值10-20%
- **稳定性优先**: 保持默认阈值
- **资源受限环境**: 提高阈值10-20%

---

## 🛡️ 稳定性增强配置参数

### StabilityEnhancer 配置

#### 重试机制配置
```python
MAX_RETRY_ATTEMPTS = 3               # 最大重试次数
RETRY_DELAY = 1.0                    # 初始重试延迟 (秒)
RETRY_STRATEGY = 'EXPONENTIAL'       # 重试策略
BACKOFF_FACTOR = 2.0                 # 退避因子
```

**重试策略选项**:
- `FIXED`: 固定间隔重试
- `LINEAR`: 线性增长重试间隔
- `EXPONENTIAL`: 指数退避重试 (推荐)

#### 熔断器配置
```python
FAILURE_THRESHOLD = 5                # 失败阈值
RECOVERY_TIMEOUT = 60                # 恢复超时时间 (秒)
HALF_OPEN_MAX_CALLS = 3              # 半开状态最大调用次数
```

**调优建议**:
- **网络稳定环境**: FAILURE_THRESHOLD=3, RECOVERY_TIMEOUT=30
- **网络不稳定环境**: FAILURE_THRESHOLD=10, RECOVERY_TIMEOUT=120
- **高可用要求**: FAILURE_THRESHOLD=2, RECOVERY_TIMEOUT=15

#### 降级策略配置
```python
ENABLE_GRACEFUL_DEGRADATION = True   # 启用优雅降级
DEGRADATION_TIMEOUT = 5.0            # 降级触发超时 (秒)
FALLBACK_CACHE_TTL = 300             # 降级缓存有效期 (秒)
```

---

## 🌍 环境特定配置

### 开发环境配置
```python
# 连接池 - 开发环境
MAX_CONNECTIONS = 5
MIN_CONNECTIONS = 2
HEALTH_CHECK_INTERVAL = 120

# 缓存 - 开发环境
MAX_MEMORY_SIZE = 200
DEFAULT_TTL = 600
ENABLE_DISK_CACHE = False

# 监控 - 开发环境
COLLECTION_INTERVAL = 30
RETENTION_HOURS = 6
ENABLE_ALERTS = False
```

### 测试环境配置
```python
# 连接池 - 测试环境
MAX_CONNECTIONS = 10
MIN_CONNECTIONS = 3
HEALTH_CHECK_INTERVAL = 60

# 缓存 - 测试环境
MAX_MEMORY_SIZE = 500
DEFAULT_TTL = 900
ENABLE_DISK_CACHE = True

# 监控 - 测试环境
COLLECTION_INTERVAL = 15
RETENTION_HOURS = 12
ENABLE_ALERTS = True
```

### 生产环境配置 (推荐)
```python
# 连接池 - 生产环境
MAX_CONNECTIONS = 20
MIN_CONNECTIONS = 5
HEALTH_CHECK_INTERVAL = 60

# 缓存 - 生产环境
MAX_MEMORY_SIZE = 2000
DEFAULT_TTL = 1800
ENABLE_DISK_CACHE = True

# 监控 - 生产环境
COLLECTION_INTERVAL = 10
RETENTION_HOURS = 48
ENABLE_ALERTS = True
```

---

## 📝 配置文件示例

### 完整配置文件 (config/db_optimization.conf)
```ini
[connection_pool]
host = localhost
port = 9000
database = stock
user = default
password = 
max_connections = 20
min_connections = 5
max_idle_time = 300
health_check_interval = 60
connection_timeout = 30
query_timeout = 300

[query_cache]
cache_enabled = true
max_memory_size = 2000
default_ttl = 1800
enable_disk_cache = true
max_disk_size = 5000
cache_dir = cache
lru_enabled = true
preload_enabled = false
compression_enabled = false

[performance_monitor]
collection_interval = 10
retention_hours = 48
enable_alerts = true
cpu_warning_threshold = 80.0
cpu_critical_threshold = 95.0
memory_warning_threshold = 85.0
memory_critical_threshold = 95.0
query_time_warning = 5.0
query_time_critical = 10.0
pool_usage_warning = 90.0
cache_hit_rate_warning = 0.5

[stability_enhancer]
max_retry_attempts = 3
retry_delay = 1.0
retry_strategy = EXPONENTIAL
backoff_factor = 2.0
failure_threshold = 5
recovery_timeout = 60
half_open_max_calls = 3
enable_graceful_degradation = true
degradation_timeout = 5.0
fallback_cache_ttl = 300
```

### Python配置加载示例
```python
import configparser

def load_optimization_config(config_file='config/db_optimization.conf'):
    """加载数据库优化配置"""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    return {
        'connection_pool': {
            'host': config.get('connection_pool', 'host'),
            'port': config.getint('connection_pool', 'port'),
            'database': config.get('connection_pool', 'database'),
            'max_connections': config.getint('connection_pool', 'max_connections'),
            'min_connections': config.getint('connection_pool', 'min_connections'),
            # ... 其他参数
        },
        'query_cache': {
            'cache_enabled': config.getboolean('query_cache', 'cache_enabled'),
            'max_memory_size': config.getint('query_cache', 'max_memory_size'),
            'default_ttl': config.getint('query_cache', 'default_ttl'),
            # ... 其他参数
        },
        # ... 其他配置节
    }
```

---

## 🔍 性能调优指南

### 1. 连接池调优
- **监控连接使用率**: 保持在70-80%
- **调整连接数**: 根据并发用户数和查询复杂度
- **优化健康检查**: 平衡检查频率和性能开销

### 2. 缓存调优
- **监控缓存命中率**: 目标 > 60%
- **调整TTL**: 根据数据更新频率
- **优化缓存大小**: 根据可用内存和查询模式

### 3. 监控调优
- **调整收集间隔**: 平衡监控精度和性能开销
- **优化告警阈值**: 减少误报，确保及时发现问题
- **配置告警通知**: 设置合适的通知渠道和频率

### 4. 稳定性调优
- **调整重试策略**: 根据网络环境和服务稳定性
- **优化熔断阈值**: 平衡服务保护和可用性
- **配置降级策略**: 确保关键功能的可用性

---

**配置文档版本**: v1.0  
**最后更新**: 2025-06-22  
**适用版本**: 数据库优化系统 v1.0
