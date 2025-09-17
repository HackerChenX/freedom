# 指标快速开发指南

## 概述

本指南提供了基于L4核心服务层BaseIndicator的标准化指标开发流程，确保新指标能够快速、正确地集成到系统中。

## 开发流程

### 1. 准备阶段

1. 确认指标需求和计算逻辑
2. 选择合适的基础周期参数
3. 确定输入数据要求

### 2. 创建指标类

```python
from indicators.base_indicator import BaseIndicator
from config.indicator_config import indicator_config

class MyNewIndicator(BaseIndicator):
    def __init__(self, period: int = None, **kwargs):
        # 使用配置管理避免硬编码
        period = period or indicator_config.get_period('medium')
        super().__init__(name="MyNewIndicator", period=period, **kwargs)
```

### 3. 实现必要方法

必须实现的抽象方法：
- `calculate(self, data: pd.DataFrame) -> pd.DataFrame`
- `get_signal(self, data: pd.DataFrame) -> Dict[str, Any]`

### 4. 测试指标

```python
from indicators.testing.indicator_test_framework import IndicatorTestFramework

framework = IndicatorTestFramework()
result = framework.test_indicator(MyNewIndicator, period=20)
```

### 5. 注册指标

在 `indicators/complete_indicator_registry.py` 中注册新指标：

```python
CORE_INDICATORS['MY_NEW'] = 'indicators.my_new_indicator.MyNewIndicator'
```

## 最佳实践

1. **使用配置管理**: 避免硬编码数字和阈值
2. **完善文档**: 提供详细的docstring和类型注解
3. **异常处理**: 使用@exception_handler装饰器
4. **性能监控**: 使用@performance_monitor装饰器
5. **数据验证**: 重写validate_data方法进行输入验证

## 示例

参考 `indicators/templates/indicator_template.py` 获取完整的实现示例。
