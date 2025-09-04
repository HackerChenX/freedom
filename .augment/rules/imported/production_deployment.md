---
type: "agent_requested"
description: "Example description"
---

# 生产环境部署规范

## 🚀 部署前检查清单

### 系统完整性验证
基于我们的88指标系统集成项目，部署前必须完成：

- [ ] **105指标系统验证**
  - [ ] 25个真实指标正常工作
  - [ ] 80个Mock指标API兼容
  - [ ] 93.8%以上注册成功率
  - [ ] 指标计算性能<2秒

- [ ] **六层架构合规检查**
  - [ ] L6用户接口层(bin/, api/)正常
  - [ ] L5业务应用层(strategy/, analysis/)功能完整
  - [ ] L4核心服务层(indicators/, formula/)稳定
  - [ ] L3数据服务层(db/interfaces/, db/managers/)可靠
  - [ ] L2存储访问层(db/enhanced_connection_pool.py)高效
  - [ ] L1基础设施层(utils/, config/, enums/)健壮

- [ ] **数据库连接池验证**
  - [ ] ClickHouse连接池5-20并发连接正常
  - [ ] 连接池故障恢复机制测试通过
  - [ ] 查询性能监控正常
  - [ ] 连接泄漏检测无异常

- [ ] **依赖注入系统检查**
  - [ ] 服务容器注册完整
  - [ ] 循环依赖检测通过
  - [ ] 服务解析性能达标
  - [ ] 生命周期管理正确

## 🏗️ 生产环境架构要求

### 基础设施配置
```yaml
# deployment/production.yaml
production:
  environment: "production"
  
  database:
    clickhouse:
      cluster_mode: true
      replica_count: 3
      max_connections: 50
      connection_timeout: 30
      query_timeout: 300
      
  application:
    instances: 3
    max_memory: "8GB"
    max_cpu: "4cores"
    
  monitoring:
    metrics_enabled: true
    logging_level: "INFO"
    performance_tracking: true
    alert_thresholds:
      response_time: 5000ms
      error_rate: 1%
      memory_usage: 80%
```

### 服务部署配置
```python
# config/production_config.py
from config.database_config_manager import DatabaseConfigManager
from utils.logger import get_logger

class ProductionConfig:
    """生产环境配置"""
    
    # 数据库配置
    DATABASE_CONFIG = {
        'clickhouse': {
            'host': 'clickhouse-cluster.internal',
            'port': 9000,
            'user': 'stock_analysis',
            'password': '${CLICKHOUSE_PASSWORD}',  # 环境变量
            'database': 'stock_data',
            'pool_size': 20,
            'max_retries': 3
        }
    }
    
    # 指标系统配置
    INDICATOR_CONFIG = {
        'registry_cache_ttl': 3600,  # 1小时
        'calculation_timeout': 10,   # 10秒
        'batch_size': 1000,
        'parallel_workers': 8
    }
    
    # 策略系统配置
    STRATEGY_CONFIG = {
        'max_concurrent_strategies': 50,
        'signal_cache_ttl': 300,  # 5分钟
        'backtest_timeout': 1800  # 30分钟
    }
    
    # 监控配置
    MONITORING_CONFIG = {
        'metrics_port': 9090,
        'health_check_interval': 30,
        'log_level': 'INFO',
        'enable_tracing': True
    }
```

## 📊 性能要求和监控

### 性能基准
```python
# monitoring/performance_benchmarks.py
PERFORMANCE_BENCHMARKS = {
    # 指标计算性能
    'indicator_calculation': {
        'single_indicator_max_time': 2.0,      # 单指标最大计算时间
        'batch_indicators_max_time': 30.0,     # 批量指标最大时间
        'memory_usage_limit': '2GB',           # 内存使用限制
        'cpu_usage_limit': 80                  # CPU使用率限制
    },
    
    # 数据库性能
    'database_performance': {
        'connection_pool_efficiency': 0.95,    # 连接池效率
        'query_response_time_p95': 1000,       # 95%查询响应时间(ms)
        'concurrent_connections_max': 20,      # 最大并发连接
        'connection_leak_tolerance': 0         # 连接泄漏容忍度
    },
    
    # 策略系统性能
    'strategy_performance': {
        'signal_generation_max_time': 3.0,     # 信号生成最大时间
        'backtest_max_time': 1800,             # 回测最大时间
        'strategy_memory_limit': '1GB',        # 策略内存限制
        'concurrent_strategies_max': 50        # 最大并发策略数
    },
    
    # 系统整体性能
    'system_performance': {
        'api_response_time_p95': 2000,         # API响应时间
        'system_availability': 0.999,          # 系统可用性
        'error_rate_max': 0.01,                # 最大错误率
        'throughput_min': 1000                 # 最小吞吐量(req/min)
    }
}
```

### 监控指标收集
```python
# monitoring/metrics_collector.py
import time
import psutil
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """生产环境指标收集器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统级指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'timestamp': time.time(),
                'system': {
                    'cpu_usage_percent': cpu_percent,
                    'memory_usage_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_usage_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3)
                },
                'application': {
                    'uptime_seconds': time.time() - self.start_time,
                    'request_count': self.request_count,
                    'error_count': self.error_count,
                    'error_rate': self.error_count / max(self.request_count, 1),
                    'avg_response_time': sum(self.response_times[-100:]) / len(self.response_times[-100:]) if self.response_times else 0
                }
            }
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            return {}
    
    def record_request(self, response_time: float, error: bool = False):
        """记录请求指标"""
        self.request_count += 1
        self.response_times.append(response_time)
        if error:
            self.error_count += 1
        
        # 保持最近1000个响应时间记录
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
```

## 🔧 健康检查系统

### 应用健康检查
```python
# monitoring/health_checker.py
from enum import Enum
from typing import Dict, Any, List
from db.enhanced_connection_pool import ClickHouseConnectionPool
from indicators.complete_indicator_registry import get_indicator
from utils.container import container

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthChecker:
    """生产环境健康检查器"""
    
    def __init__(self):
        self.checks = {
            'database': self._check_database,
            'indicators': self._check_indicators,
            'container': self._check_container,
            'memory': self._check_memory,
            'disk': self._check_disk
        }
    
    def run_health_check(self) -> Dict[str, Any]:
        """运行完整健康检查"""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                results[check_name] = result
                
                if result['status'] == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result['status'] == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                results[check_name] = {
                    'status': HealthStatus.UNHEALTHY,
                    'message': f"健康检查失败: {str(e)}",
                    'timestamp': time.time()
                }
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            'overall_status': overall_status.value,
            'checks': results,
            'timestamp': time.time()
        }
    
    def _check_database(self) -> Dict[str, Any]:
        """检查数据库连接"""
        try:
            pool = ClickHouseConnectionPool()
            with pool.get_connection() as conn:
                result = conn.execute("SELECT 1")
                if result == [(1,)]:
                    return {
                        'status': HealthStatus.HEALTHY,
                        'message': 'Database connection successful',
                        'response_time_ms': 0  # 实际应该测量
                    }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Database connection failed: {str(e)}'
            }
    
    def _check_indicators(self) -> Dict[str, Any]:
        """检查指标系统"""
        try:
            # 测试核心指标
            core_indicators = ['MA', 'EMA', 'MACD', 'RSI']
            failed_indicators = []
            
            for indicator_name in core_indicators:
                try:
                    indicator = get_indicator(indicator_name)
                    if not indicator:
                        failed_indicators.append(indicator_name)
                except Exception:
                    failed_indicators.append(indicator_name)
            
            if not failed_indicators:
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': 'All core indicators available'
                }
            elif len(failed_indicators) < len(core_indicators) / 2:
                return {
                    'status': HealthStatus.DEGRADED,
                    'message': f'Some indicators unavailable: {failed_indicators}'
                }
            else:
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'message': f'Many indicators unavailable: {failed_indicators}'
                }
                
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Indicator system check failed: {str(e)}'
            }
    
    def _check_container(self) -> Dict[str, Any]:
        """检查依赖注入容器"""
        try:
            # 测试关键服务解析
            critical_services = ["DataAccessInterface", "Logger"]
            failed_services = []
            
            for service in critical_services:
                try:
                    resolved = container.resolve(service)
                    if not resolved:
                        failed_services.append(service)
                except Exception:
                    failed_services.append(service)
            
            if not failed_services:
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': 'All critical services available'
                }
            else:
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'message': f'Critical services unavailable: {failed_services}'
                }
                
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Container check failed: {str(e)}'
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """检查内存使用"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent < 80:
                status = HealthStatus.HEALTHY
            elif usage_percent < 90:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            
            return {
                'status': status,
                'message': f'Memory usage: {usage_percent:.1f}%',
                'usage_percent': usage_percent,
                'available_gb': memory.available / (1024**3)
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Memory check failed: {str(e)}'
            }
    
    def _check_disk(self) -> Dict[str, Any]:
        """检查磁盘空间"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = disk.percent
            
            if usage_percent < 80:
                status = HealthStatus.HEALTHY
            elif usage_percent < 90:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            
            return {
                'status': status,
                'message': f'Disk usage: {usage_percent:.1f}%',
                'usage_percent': usage_percent,
                'free_gb': disk.free / (1024**3)
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Disk check failed: {str(e)}'
            }
```

## 🚨 告警系统

### 告警规则配置
```python
# monitoring/alert_rules.py
ALERT_RULES = {
    'critical': {
        'database_connection_failure': {
            'condition': 'database_status == "unhealthy"',
            'threshold': 1,
            'duration': '1m',
            'message': 'Database connection failed - immediate attention required'
        },
        'high_error_rate': {
            'condition': 'error_rate > 0.05',
            'threshold': 0.05,
            'duration': '5m',
            'message': 'Error rate exceeds 5% - system degraded'
        },
        'memory_exhaustion': {
            'condition': 'memory_usage > 0.95',
            'threshold': 95,
            'duration': '2m',
            'message': 'Memory usage critical - potential OOM'
        }
    },
    
    'warning': {
        'slow_indicator_calculation': {
            'condition': 'indicator_calc_time > 5.0',
            'threshold': 5.0,
            'duration': '10m',
            'message': 'Indicator calculation time degraded'
        },
        'high_memory_usage': {
            'condition': 'memory_usage > 0.80',
            'threshold': 80,
            'duration': '15m',
            'message': 'Memory usage high - monitor closely'
        },
        'connection_pool_pressure': {
            'condition': 'connection_pool_usage > 0.90',
            'threshold': 90,
            'duration': '5m',
            'message': 'Connection pool under pressure'
        }
    }
}
```

## 📋 运维操作手册

### 常见问题处理

#### 1. 指标计算性能下降
```bash
# 检查指标注册状态
python -c "from indicators.complete_indicator_registry import get_registry_status; print(get_registry_status())"

# 重启指标注册
python -c "from indicators.complete_indicator_registry import register_all_indicators; register_all_indicators()"

# 清理指标缓存
redis-cli FLUSHDB  # 如果使用Redis缓存
```

#### 2. 数据库连接问题
```bash
# 检查连接池状态
python -c "from db.enhanced_connection_pool import ClickHouseConnectionPool; pool = ClickHouseConnectionPool(); print(f'Active: {pool._active_connections}, Pool: {len(pool._pool)}')"

# 重置连接池
systemctl restart stock-analysis-service

# 检查ClickHouse状态
clickhouse-client --query "SELECT 1"
```

#### 3. 内存泄漏检查
```bash
# 监控内存使用
watch -n 5 'ps aux | grep python | grep stock'

# 生成内存分析报告
python -m memory_profiler bin/freedom_select.py

# 强制垃圾回收
python -c "import gc; gc.collect(); print(f'Collected: {gc.get_count()}')"
```

### 部署更新流程
```bash
#!/bin/bash
# deployment/update_production.sh

set -e  # 遇到错误立即退出

echo "=== 生产环境更新流程 ==="

# 1. 预检查
echo "1. 运行预检查..."
python tools/deployment_checker.py --environment=production

# 2. 数据库备份
echo "2. 备份数据库..."
clickhouse-client --query "BACKUP DATABASE stock_data TO '/backup/$(date +%Y%m%d_%H%M%S)'"

# 3. 停止服务
echo "3. 停止服务..."
systemctl stop stock-analysis-service

# 4. 更新代码
echo "4. 更新代码..."
git pull origin main
pip install -r requirements.txt

# 5. 运行迁移
echo "5. 运行数据库迁移..."
python scripts/run_migrations.py

# 6. 重新注册指标
echo "6. 重新注册指标..."
python -c "from indicators.complete_indicator_registry import register_all_indicators; register_all_indicators()"

# 7. 启动服务
echo "7. 启动服务..."
systemctl start stock-analysis-service

# 8. 健康检查
echo "8. 运行健康检查..."
sleep 30  # 等待服务启动
python -c "from monitoring.health_checker import HealthChecker; hc = HealthChecker(); result = hc.run_health_check(); print(result['overall_status'])"

echo "=== 更新完成 ==="
```

## ✅ 生产部署检查清单

部署到生产环境前必须完成：

**系统验证**
- [ ] 105指标系统注册成功率 > 90%
- [ ] 数据库连接池功能正常
- [ ] 依赖注入容器工作正常
- [ ] 六层架构合规检查通过

**性能验证**
- [ ] 指标计算性能测试通过
- [ ] 数据库查询性能达标
- [ ] 内存使用在安全范围内
- [ ] 并发压力测试通过

**安全检查**
- [ ] 敏感信息环境变量化
- [ ] 数据库访问权限最小化
- [ ] API接口安全验证
- [ ] 日志敏感信息脱敏

**监控和告警**
- [ ] 健康检查端点正常
- [ ] 性能指标收集工作
- [ ] 告警规则配置完成
- [ ] 日志聚合配置完成

**备份和恢复**
- [ ] 数据库备份策略配置
- [ ] 配置文件备份
- [ ] 恢复流程验证
- [ ] 灾难恢复计划制定

**文档和培训**
- [ ] 运维手册完整
- [ ] 故障排查指南
- [ ] 团队培训完成
- [ ] 联系人信息更新

这些规范确保我们的105指标股票分析系统能够安全、稳定地在生产环境运行。
description:
globs:
alwaysApply: true
---
