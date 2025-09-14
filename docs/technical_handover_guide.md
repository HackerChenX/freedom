# 技术交接指南

## 🎯 项目概述

本项目是一个基于128个技术指标的股票分析系统，采用六层架构设计，具备完整的性能监控和数据质量保证体系。

## 🏗️ 系统架构

### 六层架构设计
```
L6: 用户接口层 (bin/, api/) → 只能调用 L5
L5: 业务应用层 (strategy/, analysis/) → 只能调用 L4  
L4: 核心服务层 (indicators/, formula/) → 只能调用 L3
L3: 数据服务层 (db/interfaces/, db/managers/) → 只能调用 L2
L2: 存储访问层 (db/enhanced_connection_pool.py) → 只能调用 L1
L1: 基础设施层 (utils/, config/, enums/)
```

### 核心组件

#### 1. 指标系统 (`indicators/`)
- **complete_indicator_registry.py**: 128个指标注册中心
- **base_indicator.py**: 指标基类
- **各类指标实现**: MA, MACD, RSI, BOLL, KDJ等

#### 2. 数据层 (`db/`)
- **enhanced_connection_pool.py**: 增强连接池（任务5优化版本）
- **query_optimizer.py**: 查询优化器
- **unified_data_quality_manager.py**: 数据质量管理
- **data_quality_monitor.py**: 质量监控

#### 3. 性能监控 (`utils/`)
- **advanced_performance_monitor.py**: 高级性能监控
- **system_resource_monitor.py**: 系统资源监控
- **unified_performance_system.py**: 统一性能系统

#### 4. 策略系统 (`strategy/`)
- **strategy_manager.py**: 策略管理器
- **base_strategy.py**: 策略基类

## 🔧 关键技术实现

### 1. 指标注册机制
```python
# 指标注册示例
from indicators.complete_indicator_registry import get_indicator_registry

registry = get_indicator_registry()
indicator = registry.get_indicator('MACD')
result = indicator.calculate(data)
```

### 2. 数据库访问规范
```python
# 标准查询模板
def get_stock_data(code: str, start_date: str, end_date: str, level: str = '日线'):
    return f"""
    SELECT code, name, date, open, high, low, close, volume, turnover_rate
    FROM stock_info 
    WHERE code = '{code}'
    AND level = '{level}'
    AND date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date ASC
    """
```

### 3. 性能监控装饰器
```python
from utils.advanced_performance_monitor import method_monitor

@method_monitor(threshold_seconds=2.0)
def calculate_indicator(self, data):
    # 指标计算逻辑
    pass
```

### 4. 异常处理标准
```python
from utils.enhanced_exception_handler import exception_handler

@exception_handler(reraise=True)
def safe_method(self):
    # 业务逻辑
    pass
```

## 📊 系统状态监控

### 1. 指标质量监控
```bash
# 运行完整的指标质量检查
python scripts/unified_indicator_quality_monitor.py

# 预期结果
# - 128个指标100%注册成功
# - 96%+验证通过率
# - 执行时间<60秒
```

### 2. 性能监控
```python
from utils.unified_performance_system import get_unified_performance_system

system = get_unified_performance_system()
overview = system.get_system_overview()
analysis = system.run_performance_analysis(24)  # 24小时分析
```

### 3. 数据质量检查
```python
from db.unified_data_quality_manager import get_data_quality_manager

manager = get_data_quality_manager()
result = manager.ensure_data_quality(data, 'dataset_id')
```

## 🚨 重要规则和约束

### 1. 强制架构规则
- **禁止跨层调用**: L6不能直接调用L3
- **依赖注入**: 使用容器管理依赖
- **异常处理**: 所有关键方法必须有装饰器
- **性能监控**: 耗时方法必须监控

### 2. 数据库访问规则
- **必须包含**: code条件、date范围、level指定
- **禁止**: SELECT *
- **必须**: ORDER BY子句
- **使用**: 连接池访问

### 3. 代码质量要求
- **命名规范**: 类名大驼峰，方法名小写下划线
- **文档字符串**: 所有公共方法必须有文档
- **类型注解**: 推荐使用类型提示
- **日志记录**: 关键操作必须记录日志

## 🔄 开发流程

### 1. 新功能开发
1. **检查现有实现**: 避免重复开发
2. **遵循架构规范**: 确保分层正确
3. **添加监控装饰器**: 性能和异常处理
4. **编写测试**: 单元测试和集成测试
5. **运行质量检查**: 确保指标稳定性

### 2. 指标开发
1. **继承BaseIndicator**: 使用标准基类
2. **实现必需方法**: calculate, get_signal, get_patterns
3. **注册到注册表**: 在complete_indicator_registry.py中注册
4. **编写验证脚本**: 创建对应的validate_*.py
5. **运行质量监控**: 确保通过验证

### 3. 性能优化
1. **使用性能监控**: 识别瓶颈
2. **应用缓存策略**: 合理使用缓存
3. **优化数据库查询**: 使用查询优化器
4. **监控资源使用**: 关注CPU、内存使用

## 🧪 测试和验证

### 1. 单元测试
```bash
# 运行单元测试
python -m pytest tests/unit/

# 运行特定模块测试
python -m pytest tests/unit/test_indicators.py
```

### 2. 集成测试
```bash
# 运行集成测试
python -m pytest tests/integration/

# 运行性能测试
python -m pytest tests/performance/
```

### 3. 系统验证
```bash
# 完整系统验证
python scripts/system_validation.py

# 架构合规检查
python scripts/architecture_compliance_check.py
```

## 📈 性能基准

### 1. 系统性能指标
- **指标计算时间**: <2秒/指标
- **数据库查询时间**: <1秒/查询
- **缓存命中率**: >70%
- **系统响应时间**: <3秒

### 2. 质量指标
- **指标注册成功率**: 100%
- **指标验证通过率**: >90%
- **代码覆盖率**: >80%
- **错误率**: <1%

### 3. 资源使用
- **内存使用**: <2GB
- **CPU使用**: <80%
- **磁盘使用**: <85%
- **网络延迟**: <100ms

## 🔧 故障排除

### 1. 常见问题

#### 指标注册失败
```python
# 检查指标实现
from indicators.complete_indicator_registry import get_indicator_registry
registry = get_indicator_registry()
print(registry.get_registration_status())
```

#### 性能问题
```python
# 查看性能报告
from utils.unified_performance_system import get_unified_performance_system
system = get_unified_performance_system()
report = system.run_performance_analysis(1)  # 最近1小时
```

#### 数据质量问题
```python
# 运行数据质量检查
from db.data_quality_monitor import get_data_quality_monitor
monitor = get_data_quality_monitor()
status = monitor.get_monitoring_status()
```

### 2. 日志查看
```bash
# 查看系统日志
tail -f logs/system.log

# 查看错误日志
tail -f logs/error.log

# 查看性能日志
tail -f logs/performance.log
```

### 3. 系统重启
```bash
# 重启性能监控
python scripts/restart_performance_monitoring.py

# 重启数据质量监控
python scripts/restart_quality_monitoring.py

# 完整系统重启
python scripts/system_restart.py
```

## 📚 参考文档

### 1. 核心文档
- **架构规范**: `.augment/rules/imported/stock_analysis_architecture.md`
- **开发标准**: `.augment/rules/imported/mandatory_development_standards.md`
- **数据库规范**: `.augment/rules/imported/database_access.md`

### 2. API文档
- **指标API**: `docs/api/indicators.md`
- **数据API**: `docs/api/database.md`
- **性能API**: `docs/api/performance.md`

### 3. 示例代码
- **指标开发示例**: `examples/indicator_development.py`
- **策略开发示例**: `examples/strategy_development.py`
- **性能优化示例**: `examples/performance_optimization.py`

## 🆘 技术支持

### 1. 问题分类
- **架构问题**: 查看架构规范文档
- **性能问题**: 使用性能监控工具
- **数据问题**: 运行数据质量检查
- **指标问题**: 使用指标验证脚本

### 2. 联系方式
- **技术文档**: 项目docs目录
- **代码示例**: 项目examples目录
- **测试用例**: 项目tests目录

### 3. 最佳实践
- **定期运行质量监控**: 每次修改后
- **关注性能指标**: 每日检查
- **备份重要数据**: 定期备份
- **更新文档**: 及时更新变更

---

**文档版本**: 1.0  
**最后更新**: 2025-09-05  
**适用版本**: 当前系统版本  
**维护人员**: 开发团队
