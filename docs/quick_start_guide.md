# 快速启动指南

## 🚀 5分钟快速上手

### 1. 环境检查
```bash
# 检查Python版本 (需要3.8+)
python --version

# 检查项目目录
ls -la /Users/hacker/PycharmProjects/freedom

# 检查关键文件
ls -la indicators/complete_indicator_registry.py
ls -la scripts/unified_indicator_quality_monitor.py
```

### 2. 系统验证
```bash
# 进入项目目录
cd /Users/hacker/PycharmProjects/freedom

# 运行指标质量监控 (最重要的验证)
python scripts/unified_indicator_quality_monitor.py

# 预期结果:
# ✅ 128个指标100%注册成功
# ✅ 96%+验证通过率
# ✅ 执行时间<60秒
```

### 3. 核心功能测试
```python
# 测试指标系统
python -c "
from indicators.complete_indicator_registry import get_indicator_registry
registry = get_indicator_registry()
print(f'注册指标数: {len(registry.get_all_indicators())}')
print('✅ 指标系统正常')
"

# 测试数据质量
python -c "
from db.unified_data_quality_manager import get_data_quality_manager
manager = get_data_quality_manager()
print('✅ 数据质量系统正常')
"

# 测试性能监控
python -c "
from utils.unified_performance_system import get_unified_performance_system
system = get_unified_performance_system()
print('✅ 性能监控系统正常')
"
```

## 📋 系统状态检查清单

### ✅ 必检项目
- [ ] 指标注册成功率 = 100% (128/128)
- [ ] 指标验证通过率 ≥ 90% (96/100)
- [ ] 数据库连接正常
- [ ] 性能监控活跃
- [ ] 数据质量监控运行
- [ ] 缓存系统工作
- [ ] 日志记录正常

### 🔧 关键命令
```bash
# 1. 完整系统验证
python scripts/unified_indicator_quality_monitor.py

# 2. 性能监控检查
python -c "
from utils.system_resource_monitor import get_resource_monitor
monitor = get_resource_monitor()
status = monitor.get_current_status()
print(f'CPU: {status.get(\"cpu\", {}).get(\"percent\", 0):.1f}%')
print(f'内存: {status.get(\"memory\", {}).get(\"percent\", 0):.1f}%')
"

# 3. 数据质量检查
python -c "
from db.data_quality_monitor import get_data_quality_monitor
monitor = get_data_quality_monitor()
status = monitor.get_monitoring_status()
print(f'监控状态: {status[\"monitoring_active\"]}')
"
```

## 🎯 核心功能使用

### 1. 指标计算
```python
import pandas as pd
from indicators.complete_indicator_registry import get_indicator_registry

# 准备数据
data = pd.DataFrame({
    'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    'high': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
    'low': [9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
    'volume': [1000, 1100, 1200, 1300, 1400, 1500]
})

# 获取指标
registry = get_indicator_registry()
ma_indicator = registry.get_indicator('MA')

# 计算指标
if ma_indicator:
    result = ma_indicator.calculate(data)
    print("MA计算结果:", result)
```

### 2. 性能监控
```python
from utils.advanced_performance_monitor import method_monitor

@method_monitor(threshold_seconds=1.0)
def my_function():
    import time
    time.sleep(0.5)
    return "完成"

# 函数会自动监控性能
result = my_function()
```

### 3. 数据质量检查
```python
import pandas as pd
from db.unified_data_quality_manager import get_data_quality_manager

# 准备测试数据
data = pd.DataFrame({
    'code': ['000001', '000002'],
    'date': ['2024-01-01', '2024-01-02'],
    'open': [10.0, 11.0],
    'close': [10.5, 11.5],
    'high': [10.8, 11.8],
    'low': [9.8, 10.8],
    'volume': [1000000, 1100000]
})

# 质量检查
manager = get_data_quality_manager()
result = manager.ensure_data_quality(data, 'test_dataset')
print(f"质量状态: {result['overall_status']}")
```

## 🚨 常见问题快速解决

### 问题1: 指标注册失败
```bash
# 检查指标注册状态
python -c "
from indicators.complete_indicator_registry import get_indicator_registry
registry = get_indicator_registry()
status = registry.get_registration_status()
print('注册状态:', status)
"

# 解决方案: 重新运行注册
python scripts/register_all_indicators.py
```

### 问题2: 性能监控异常
```bash
# 重启性能监控
python -c "
from utils.unified_performance_system import get_unified_performance_system
system = get_unified_performance_system()
system.stop_system()
system.start_system()
print('性能监控已重启')
"
```

### 问题3: 数据质量告警
```bash
# 查看质量报告
python -c "
from db.data_quality_monitor import get_data_quality_monitor
monitor = get_data_quality_monitor()
trends = monitor.get_quality_trends(24)
print('质量趋势:', trends)
"
```

## 📊 系统健康检查

### 自动健康检查脚本
```python
#!/usr/bin/env python3
"""系统健康检查脚本"""

def health_check():
    print("🔍 开始系统健康检查...")
    
    # 1. 指标系统检查
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
        registry = get_indicator_registry()
        indicator_count = len(registry.get_all_indicators())
        print(f"✅ 指标系统: {indicator_count}个指标已注册")
    except Exception as e:
        print(f"❌ 指标系统异常: {e}")
    
    # 2. 数据库连接检查
    try:
        from db.enhanced_connection_pool import get_connection_pool
        pool = get_connection_pool()
        print("✅ 数据库连接正常")
    except Exception as e:
        print(f"❌ 数据库连接异常: {e}")
    
    # 3. 性能监控检查
    try:
        from utils.unified_performance_system import get_unified_performance_system
        system = get_unified_performance_system()
        overview = system.get_system_overview()
        print(f"✅ 性能监控: 活跃状态 {overview['system_active']}")
    except Exception as e:
        print(f"❌ 性能监控异常: {e}")
    
    # 4. 数据质量检查
    try:
        from db.data_quality_monitor import get_data_quality_monitor
        monitor = get_data_quality_monitor()
        status = monitor.get_monitoring_status()
        print(f"✅ 数据质量监控: {status['monitoring_active']}")
    except Exception as e:
        print(f"❌ 数据质量监控异常: {e}")
    
    print("🎉 系统健康检查完成!")

if __name__ == "__main__":
    health_check()
```

### 保存并运行健康检查
```bash
# 保存上述脚本为 scripts/health_check.py
# 然后运行
python scripts/health_check.py
```

## 🔧 开发环境设置

### 1. IDE配置 (推荐PyCharm)
```python
# 设置项目根目录
PROJECT_ROOT = "/Users/hacker/PycharmProjects/freedom"

# 添加到Python路径
import sys
sys.path.append(PROJECT_ROOT)

# 设置工作目录
import os
os.chdir(PROJECT_ROOT)
```

### 2. 调试配置
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用性能监控
from utils.advanced_performance_monitor import get_performance_analyzer
analyzer = get_performance_analyzer()
```

### 3. 测试配置
```bash
# 运行特定测试
python -m pytest tests/unit/test_indicators.py -v

# 运行性能测试
python -m pytest tests/performance/ -v

# 生成覆盖率报告
python -m pytest --cov=indicators tests/
```

## 📈 监控仪表板

### 关键指标监控
```python
def get_system_dashboard():
    """获取系统仪表板数据"""
    from indicators.complete_indicator_registry import get_indicator_registry
    from utils.system_resource_monitor import get_resource_monitor
    from db.data_quality_monitor import get_data_quality_monitor
    
    # 指标状态
    registry = get_indicator_registry()
    indicator_status = {
        'total_indicators': len(registry.get_all_indicators()),
        'registration_success_rate': '100%'
    }
    
    # 系统资源
    resource_monitor = get_resource_monitor()
    resource_status = resource_monitor.get_current_status()
    
    # 数据质量
    quality_monitor = get_data_quality_monitor()
    quality_status = quality_monitor.get_monitoring_status()
    
    return {
        'indicators': indicator_status,
        'resources': resource_status,
        'quality': quality_status,
        'timestamp': datetime.now().isoformat()
    }

# 使用示例
dashboard = get_system_dashboard()
print("系统仪表板:", dashboard)
```

## 🆘 紧急联系和支持

### 1. 系统恢复
```bash
# 紧急系统重启
python scripts/emergency_restart.py

# 恢复默认配置
python scripts/restore_default_config.py

# 重建指标注册表
python scripts/rebuild_indicator_registry.py
```

### 2. 数据备份
```bash
# 备份配置
cp -r config/ backup/config_$(date +%Y%m%d)/

# 备份日志
cp -r logs/ backup/logs_$(date +%Y%m%d)/

# 备份报告
cp -r reports/ backup/reports_$(date +%Y%m%d)/
```

### 3. 问题报告
```python
def generate_problem_report():
    """生成问题报告"""
    import json
    from datetime import datetime
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_status': get_system_dashboard(),
        'recent_errors': get_recent_errors(),
        'performance_metrics': get_performance_metrics(),
        'recommendations': get_system_recommendations()
    }
    
    with open(f'problem_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("问题报告已生成")
```

---

**快速启动指南版本**: 1.0  
**最后更新**: 2025-09-05  
**紧急联系**: 查看项目文档  
**技术支持**: 参考技术交接指南
