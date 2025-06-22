# 数据库优化系统运维手册

## 📋 手册概述

本手册为系统管理员和运维人员提供数据库优化系统的日常运维指南，包含监控、故障诊断、性能调优和常见问题解决方案。

### 🎯 适用人员
- 系统管理员
- 数据库管理员
- 运维工程师
- 技术支持人员

---

## 🔍 系统监控

### 1. 性能监控仪表板

#### 关键指标监控
```bash
# 查看实时性能指标
python3 -c "
from monitoring.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
metrics = monitor.get_current_metrics()
for name, metric in metrics.items():
    print(f'{name}: {metric.value}')
"
```

#### 核心监控指标
| 指标名称 | 正常范围 | 警告阈值 | 严重阈值 | 说明 |
|---------|---------|---------|---------|------|
| CPU使用率 | < 70% | 75% | 90% | 系统CPU负载 |
| 内存使用率 | < 75% | 80% | 90% | 系统内存占用 |
| 查询响应时间 | < 1秒 | 2秒 | 5秒 | 数据库查询性能 |
| 连接池使用率 | < 80% | 85% | 95% | 数据库连接池状态 |
| 缓存命中率 | > 70% | < 60% | < 40% | 查询缓存效果 |

### 2. 告警管理

#### 查看活跃告警
```bash
# 查看当前活跃告警
python3 -c "
from monitoring.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
alerts = monitor.get_active_alerts()
if alerts:
    for alert in alerts:
        print(f'[{alert[\"severity\"]}] {alert[\"rule_name\"]}: {alert[\"message\"]}')
else:
    print('当前无活跃告警')
"
```

#### 清除告警
```bash
# 清除特定告警
python3 -c "
from monitoring.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
monitor.clear_alert('告警规则名称')
print('告警已清除')
"
```

### 3. 健康检查

#### 系统健康检查
```bash
# 运行完整健康检查
python3 -c "
from monitoring.performance_monitor import get_health_checker
checker = get_health_checker()
results = checker.run_all_checks()
print(f'整体状态: {results[\"overall_status\"]}')
for name, result in results['checks'].items():
    print(f'{name}: {result[\"status\"]} - {result[\"message\"]}')
"
```

#### 数据库连接检查
```bash
# 检查数据库连接状态
python3 -c "
from db.enhanced_connection_pool import get_connection_pool
pool = get_connection_pool()
stats = pool.get_stats()
print(f'连接池状态: {stats}')
print(f'活跃连接: {stats.get(\"current_active\", 0)}/{stats.get(\"total_connections\", 0)}')
"
```

---

## 🛠️ 故障诊断

### 1. 常见问题诊断

#### 问题1: 查询响应时间过长

**症状**: 查询响应时间超过2秒，用户反馈系统慢

**诊断步骤**:
```bash
# 1. 检查数据库连接池状态
python3 -c "
from db.enhanced_connection_pool import get_connection_pool
pool = get_connection_pool()
stats = pool.get_stats()
print('连接池统计:', stats)
"

# 2. 检查缓存命中率
python3 -c "
from db.enhanced_data_manager import get_enhanced_data_manager
dm = get_enhanced_data_manager()
stats = dm.get_stats()
print(f'缓存命中率: {stats.get(\"cache_hit_rate\", 0):.2%}')
"

# 3. 检查系统资源使用
top -n 1 | head -20
free -h
```

**解决方案**:
1. **连接池不足**: 增加最大连接数配置
2. **缓存命中率低**: 检查缓存配置，增加缓存大小
3. **系统资源不足**: 检查CPU/内存使用，考虑扩容

#### 问题2: 并发查询失败

**症状**: 高并发时出现查询失败，错误率增加

**诊断步骤**:
```bash
# 1. 检查连接池使用率
python3 -c "
from monitoring.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
metrics = monitor.get_current_metrics()
pool_usage = metrics.get('connection_pool_usage')
if pool_usage:
    print(f'连接池使用率: {pool_usage.value}%')
"

# 2. 检查错误日志
tail -100 /var/log/stock_system.log | grep -i error

# 3. 检查熔断器状态
python3 -c "
from utils.stability_enhancer import get_stability_manager
sm = get_stability_manager()
stats = sm.get_stats()
print('稳定性统计:', stats)
"
```

**解决方案**:
1. **连接池耗尽**: 增加连接池大小或优化查询
2. **熔断器触发**: 检查下游服务状态，重置熔断器
3. **网络问题**: 检查网络连接和防火墙设置

#### 问题3: 内存使用率过高

**症状**: 系统内存使用率超过80%，可能出现OOM

**诊断步骤**:
```bash
# 1. 检查内存使用详情
ps aux --sort=-%mem | head -20

# 2. 检查缓存使用情况
python3 -c "
from db.query_cache import get_query_cache
cache = get_query_cache()
stats = cache.get_stats()
print('缓存统计:', stats)
"

# 3. 检查是否有内存泄漏
python3 -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'当前进程内存: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**解决方案**:
1. **缓存过大**: 减少缓存大小配置
2. **内存泄漏**: 重启相关服务，检查代码
3. **系统内存不足**: 增加系统内存或优化内存使用

### 2. 日志分析

#### 重要日志文件
```bash
# 系统主日志
tail -f /var/log/stock_system.log

# 性能监控日志
tail -f /var/log/performance_monitor.log

# ClickHouse日志
tail -f /var/log/clickhouse-server/clickhouse-server.log

# 错误日志过滤
grep -i "error\|exception\|failed" /var/log/stock_system.log | tail -20
```

#### 日志级别说明
- **INFO**: 正常操作信息
- **WARNING**: 警告信息，需要关注
- **ERROR**: 错误信息，需要处理
- **CRITICAL**: 严重错误，需要立即处理

---

## ⚙️ 性能调优

### 1. 连接池优化

#### 调整连接池参数
```python
# 编辑配置文件 config/db_optimization.conf
[connection_pool]
max_connections = 30        # 根据并发需求调整
min_connections = 8         # 保持足够的最小连接
health_check_interval = 45  # 健康检查间隔
```

#### 监控连接池效果
```bash
# 监控连接池使用情况
watch -n 5 'python3 -c "
from db.enhanced_connection_pool import get_connection_pool
pool = get_connection_pool()
stats = pool.get_stats()
print(f\"活跃连接: {stats.get(\"current_active\", 0)}/{stats.get(\"total_connections\", 0)}\")
print(f\"平均响应时间: {stats.get(\"avg_response_time\", 0):.3f}秒\")
"'
```

### 2. 缓存优化

#### 调整缓存参数
```python
# 编辑配置文件
[query_cache]
max_memory_size = 3000      # 增加内存缓存大小
default_ttl = 2400          # 调整缓存有效期
enable_disk_cache = true    # 启用磁盘缓存
```

#### 监控缓存效果
```bash
# 监控缓存命中率
python3 -c "
from db.enhanced_data_manager import get_enhanced_data_manager
dm = get_enhanced_data_manager()
stats = dm.get_stats()
print(f'缓存命中率: {stats.get(\"cache_hit_rate\", 0):.2%}')
print(f'缓存大小: {stats.get(\"cache_size\", 0)} 条目')
"
```

### 3. 监控优化

#### 调整监控参数
```python
# 编辑配置文件
[performance_monitor]
collection_interval = 15    # 降低收集频率以减少开销
retention_hours = 72        # 增加数据保留时间
```

---

## 🔧 维护操作

### 1. 定期维护任务

#### 每日维护
```bash
#!/bin/bash
# daily_maintenance.sh

echo "开始每日维护任务..."

# 1. 清理过期缓存
python3 -c "
from db.query_cache import get_query_cache
cache = get_query_cache()
cache.cleanup_expired()
print('缓存清理完成')
"

# 2. 检查系统健康
python3 -c "
from monitoring.performance_monitor import get_health_checker
checker = get_health_checker()
results = checker.run_all_checks()
if results['overall_status'] != 'healthy':
    print(f'警告: 系统状态异常 - {results[\"overall_status\"]}')
else:
    print('系统健康检查通过')
"

# 3. 备份监控数据
timestamp=$(date +%Y%m%d)
python3 -c "
from monitoring.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
monitor.export_metrics('backup/metrics_${timestamp}.json', hours=24)
print('监控数据备份完成')
"

echo "每日维护任务完成"
```

#### 每周维护
```bash
#!/bin/bash
# weekly_maintenance.sh

echo "开始每周维护任务..."

# 1. 重启监控服务
python3 -c "
from monitoring.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
monitor.stop_monitoring()
import time
time.sleep(5)
monitor.start_monitoring()
print('监控服务重启完成')
"

# 2. 清理日志文件
find /var/log -name "*.log" -mtime +7 -exec gzip {} \;
find /var/log -name "*.log.gz" -mtime +30 -delete
echo "日志清理完成"

# 3. 系统性能报告
python3 -c "
from monitoring.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
stats = monitor.get_stats()
print('=== 周性能报告 ===')
print(f'监控运行时间: {stats.get(\"monitor_start_time\", \"未知\")}')
print(f'收集指标总数: {stats.get(\"total_metrics_collected\", 0)}')
print(f'触发告警总数: {stats.get(\"total_alerts_triggered\", 0)}')
"

echo "每周维护任务完成"
```

### 2. 紧急操作

#### 紧急重启
```bash
#!/bin/bash
# emergency_restart.sh

echo "执行紧急重启..."

# 1. 停止所有服务
systemctl stop stock_selection_service
systemctl stop buypoint_analysis_service
pkill -f "performance_monitor"

# 2. 清理资源
python3 -c "
from db.enhanced_connection_pool import get_connection_pool
pool = get_connection_pool()
pool.close_all_connections()
print('连接池已清理')
"

# 3. 重启服务
systemctl start stock_selection_service
systemctl start buypoint_analysis_service

# 4. 启动监控
python3 -c "
from monitoring.performance_monitor import start_monitoring
start_monitoring()
print('监控服务已启动')
"

echo "紧急重启完成"
```

#### 降级运行
```bash
#!/bin/bash
# degraded_mode.sh

echo "启用降级模式..."

# 1. 禁用缓存
python3 -c "
from db.query_cache import get_query_cache
cache = get_query_cache()
cache.disable()
print('缓存已禁用')
"

# 2. 减少连接池大小
python3 -c "
from db.enhanced_connection_pool import get_connection_pool
pool = get_connection_pool()
pool.resize(max_connections=10, min_connections=2)
print('连接池已缩减')
"

# 3. 降低监控频率
python3 -c "
from monitoring.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
monitor.collection_interval = 60  # 降低到1分钟
print('监控频率已降低')
"

echo "降级模式已启用"
```

---

## 📞 紧急联系

### 技术支持
- **系统管理员**: [电话] [邮箱]
- **数据库管理员**: [电话] [邮箱]
- **开发团队**: [电话] [邮箱]

### 升级路径
1. **L1支持**: 运维工程师 (日常问题)
2. **L2支持**: 系统管理员 (复杂问题)
3. **L3支持**: 开发团队 (代码问题)

### 紧急响应流程
1. **发现问题** → 检查监控告警
2. **初步诊断** → 运行健康检查
3. **问题分类** → 确定严重程度
4. **执行解决方案** → 按手册操作
5. **验证修复** → 确认问题解决
6. **记录总结** → 更新知识库

---

**运维手册版本**: v1.0  
**最后更新**: 2025-06-22  
**适用系统**: 数据库优化系统 v1.0
