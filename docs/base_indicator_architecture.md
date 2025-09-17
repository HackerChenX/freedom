# BaseIndicator架构设计文档

## 概述

BaseIndicator是L4核心服务层的技术指标基类，为所有技术指标提供统一的基础架构。

## 核心设计原则

### 1. 抽象方法定义

BaseIndicator定义了两个核心抽象方法，所有子类必须实现：

```python
@abc.abstractmethod
@performance_monitor(threshold_seconds=2.0)
@exception_handler(reraise=True)
def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """计算指标值"""
    pass

@abc.abstractmethod
def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
    """获取交易信号"""
    pass
```

### 2. 扩展点方法

BaseIndicator提供了多个扩展点方法，子类可以根据需要重写：

- `validate_data()`: 数据验证
- `preprocess_data()`: 数据预处理
- `postprocess_result()`: 结果后处理
- `register_patterns()`: 注册指标形态

### 3. 工具方法

BaseIndicator提供了丰富的工具方法：

- `format_output()`: 格式化输出
- `get_metadata()`: 获取元数据
- `add_pattern()`: 添加形态信息
- `clear_result()`: 清除结果

## 继承指南

### 标准继承模式

```python
from indicators.base_indicator import BaseIndicator
import pandas as pd
from typing import Dict, Any

class MyIndicator(BaseIndicator):
    def __init__(self, period: int = 20, **kwargs):
        super().__init__(name="MyIndicator", period=period, **kwargs)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        # 实现计算逻辑
        result = data.copy()
        # ... 计算逻辑
        self._result = result
        return result
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        # 实现信号生成逻辑
        return {
            'signal': 'hold',
            'strength': 0.0,
            'timestamp': data.index[-1] if not data.empty else None
        }
```

### 最佳实践

1. **正确调用super().__init__()**
2. **实现所有抽象方法**
3. **使用装饰器进行性能监控和异常处理**
4. **遵循统一的输入输出格式**
5. **提供完整的文档字符串**

## 多态性支持

BaseIndicator支持完整的多态性调用：

```python
# 通过基类引用调用子类方法
indicator: BaseIndicator = MyIndicator(period=20)
result = indicator.calculate(data)
signal = indicator.get_signal(result)
patterns = indicator.get_patterns(result)
```

## 质量标准

### A+级标准要求

1. **继承合规性**: 95%+的指标正确继承BaseIndicator
2. **抽象方法实现**: 100%完整实现所有抽象方法
3. **多态性支持**: 90%+的多态性测试通过
4. **接口一致性**: 100%的接口调用成功

### 持续监控

使用`indicators/monitoring/inheritance_compliance_monitor.py`进行持续的合规性监控。
