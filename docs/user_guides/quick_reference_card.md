# 技术指标标准化快速参考卡

## 🚀 快速开始

### 1. 创建新指标类模板
```python
from indicators.base_indicator import BaseIndicator
from typing import Dict, Any
import pandas as pd

class YOUR_INDICATOR(BaseIndicator):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "YOUR_INDICATOR"
        self._default_parameters = self._get_default_parameters()
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        return {"period": 14}
    
    def set_parameters(self, **kwargs):
        # 参数验证逻辑
        self.period = kwargs.get('period', 14)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = self._calculate(data, **kwargs)
        self._result = result
        return result
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # 实现计算逻辑
        return data.copy()
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # 实现形态识别
        return pd.DataFrame(index=data.index)
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # 实现评分逻辑
        return pd.Series(50.0, index=data.index)
    
    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        return 0.7
```

### 2. Schema定义模板
```yaml
YOUR_INDICATOR:
  description: "指标中文描述"
  parameters:
    period:
      type: integer
      default: 14
      minimum: 1
      maximum: 200
  signals:
    YOUR_INDICATOR_BULLISH:
      description: "看涨信号"
      type: boolean
  patterns:
    YOUR_INDICATOR_上升:
      description: "上升趋势"
      type: boolean
  validation:
    required_columns: ["close"]
    min_data_points: 14
```

## ✅ 必须实现的方法

| 方法 | 用途 | 返回类型 |
|------|------|----------|
| `__init__(**kwargs)` | 构造函数 | None |
| `_get_default_parameters()` | 默认参数 | Dict[str, Any] |
| `set_parameters(**kwargs)` | 参数设置 | None |
| `calculate(data, **kwargs)` | 主计算方法 | pd.DataFrame |
| `_calculate(data, **kwargs)` | 内部计算 | pd.DataFrame |
| `get_patterns(data, **kwargs)` | 形态识别 | pd.DataFrame |
| `calculate_raw_score(data, **kwargs)` | 原始评分 | pd.Series |
| `calculate_confidence(score, patterns, signals)` | 置信度 | float |

## 📋 开发检查清单

### 代码实现
- [ ] 继承BaseIndicator
- [ ] 使用**kwargs构造函数
- [ ] 实现所有8个必需方法
- [ ] 添加参数验证逻辑
- [ ] 使用中文技术术语命名形态

### Schema配置
- [ ] 在indicator_parameter_schemas.yaml中添加定义
- [ ] 定义所有参数的类型和范围
- [ ] 使用清晰的中文形态名称
- [ ] 添加validation规则

### 质量验证
- [ ] 语法检查：`python -m py_compile indicators/your_indicator.py`
- [ ] 质量验证：`python3 final_quality_validator.py`
- [ ] 终极验证：`python3 ultimate_perfect_validator.py`
- [ ] 集成测试：`python3 system_integration_enhancer.py`

## 🚫 常见错误

### ❌ 错误做法
```python
# 不要这样做
class BadIndicator:  # 没有继承BaseIndicator
    def __init__(self, period=14):  # 不使用**kwargs
        self.period = period
```

### ✅ 正确做法
```python
# 应该这样做
class GoodIndicator(BaseIndicator):  # 继承BaseIndicator
    def __init__(self, **kwargs):  # 使用**kwargs
        super().__init__()
        self.name = "GOOD_INDICATOR"
        self._default_parameters = self._get_default_parameters()
        self.set_parameters(**kwargs)
```

## 🎯 命名规范

### 指标名称
- ✅ `MACD`, `RSI`, `BOLLINGER_BANDS`
- ❌ `macd`, `Rsi`, `bollinger-bands`

### 形态名称
- ✅ `MACD_金叉`, `RSI_超买`, `BOLL_上轨突破`
- ❌ `技术形态`, `未知形态`, `一般信号`

### 禁用词汇
- ❌ "技术形态"
- ❌ "未知形态"
- ❌ "中等股票"
- ❌ "一般信号"

### 推荐词汇
- ✅ "金叉"、"死叉"
- ✅ "突破"、"跌破"
- ✅ "超买"、"超卖"
- ✅ "上升趋势"、"下降趋势"

## 🔧 验证工具

### 质量验证
```bash
# 检查指标质量（目标：100%）
python3 final_quality_validator.py
```

### 终极验证
```bash
# 检查系统完整性（目标：5个100%）
python3 ultimate_perfect_validator.py
```

### 系统集成
```bash
# 检查集成状态
python3 system_integration_enhancer.py
```

### Schema验证
```bash
# 检查Schema定义
python3 schema_validator_fixer.py
```

## 📊 质量标准

### 100%完美状态要求
1. **100%高质量指标率** - 所有指标必须通过质量检查
2. **100%Schema验证成功率** - 所有Schema定义必须有效
3. **100%系统集成得分** - 所有组件必须正常工作
4. **100%零警告运行** - 系统运行无任何警告
5. **100%生产就绪状态** - 所有组件生产就绪

### 指标质量评分标准
- **类定义正确性**: 继承BaseIndicator，实现所有抽象方法
- **参数接口标准化**: 使用**kwargs，实现标准参数管理
- **Schema定义完整性**: 完整的参数、信号、形态定义
- **代码质量**: 无语法错误，良好的文档字符串

## 🔍 调试技巧

### 导入测试
```python
try:
    from indicators.your_indicator import YOUR_INDICATOR
    indicator = YOUR_INDICATOR()
    print("✓ 指标导入成功")
except Exception as e:
    print(f"✗ 指标导入失败: {e}")
```

### 参数验证测试
```python
from utils.indicator_parameter_validator import IndicatorParameterValidator

validator = IndicatorParameterValidator()
is_valid, errors = validator.validate_indicator_parameters('YOUR_INDICATOR', {
    'period': 20
})

if is_valid:
    print("✓ 参数验证通过")
else:
    print(f"✗ 参数验证失败: {errors}")
```

### 计算测试
```python
import pandas as pd
import numpy as np

# 创建测试数据
data = pd.DataFrame({
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 102,
    'low': np.random.randn(100).cumsum() + 98,
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
})

# 测试指标计算
indicator = YOUR_INDICATOR(period=14)
result = indicator.calculate(data)
print(f"✓ 计算成功，结果列: {result.columns.tolist()}")
```

## 📚 参考资源

- **完整文档**: `docs/indicator_standardization_guide.md`
- **示例代码**: 查看现有指标如`indicators/ma.py`
- **Schema文件**: `config/indicator_parameter_schemas.yaml`
- **验证工具**: `final_quality_validator.py`, `ultimate_perfect_validator.py`

## 🎯 目标

维持技术指标系统的**100%完美状态**：
- 82个标准化指标
- 100%高质量代码
- 100%Schema验证通过
- 零警告运行
- 生产环境就绪

遵循本快速参考卡，确保新增指标符合系统标准！
