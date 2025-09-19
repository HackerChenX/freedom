# 指标开发者实践指南

## 🎯 **指南目标**

本指南为技术指标开发者提供标准的实现模板和最佳实践，确保所有指标都能正确实现BaseIndicator抽象方法，并与L4层架构完美集成。

## 📋 **开发前检查清单**

### **1. 环境准备**
- [ ] 确认已安装所有依赖包（pandas, numpy等）
- [ ] 理解BaseIndicator抽象基类的设计
- [ ] 熟悉项目的六层架构规范
- [ ] 了解指标注册机制

### **2. 设计规划**
- [ ] 明确指标的数学计算逻辑
- [ ] 定义指标的输入参数和默认值
- [ ] 设计信号生成规则
- [ ] 确定输出数据格式

## 🏗️ **标准实现模板**

### **完整指标类模板**
```python
from indicators.base_indicator import BaseIndicator
from utils.performance_monitor import performance_monitor
from utils.exception_handler import exception_handler
import pandas as pd
import numpy as np
from typing import Dict, Any

class YourIndicator(BaseIndicator):
    """
    您的指标类文档字符串
    
    Args:
        period (int): 计算周期，默认值
        param2 (float): 其他参数，默认值
    
    Example:
        >>> indicator = YourIndicator(period=14)
        >>> values = indicator.calculate(stock_data)
        >>> signal = indicator.get_signal(values)
    """
    
    def __init__(self, period: int = 14, **kwargs):
        super().__init__("YOUR_INDICATOR_NAME")
        self.period = period
        # 其他参数初始化
        
        # 参数验证
        if period <= 0:
            raise ValueError("period必须大于0")
    
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标数值
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含指标计算结果的DataFrame
        """
        # 1. 数据验证
        self._validate_input_data(data)
        
        # 2. 指标计算
        result = self._perform_calculation(data)
        
        # 3. 结果验证
        self._validate_output_data(result)
        
        return result
    
    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=True)
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            data: 指标计算结果或包含指标数据的DataFrame
            
        Returns:
            标准化的交易信号字典
        """
        # 1. 数据验证
        if data.empty or len(data) < 2:
            return self._get_default_signal("数据不足")
        
        # 2. 信号生成逻辑
        signal = self._generate_signal_logic(data)
        
        # 3. 信号验证
        return self._validate_signal(signal)
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("输入数据必须是pandas.DataFrame")
        
        required_columns = ['close']  # 根据指标需求调整
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必需列: {missing_columns}")
        
        if len(data) < self.period:
            raise ValueError(f"数据量不足，需要至少{self.period}个数据点")
    
    def _perform_calculation(self, data: pd.DataFrame) -> pd.DataFrame:
        """执行具体的指标计算逻辑"""
        # 示例：简单移动平均
        close = data['close']
        ma_value = close.rolling(window=self.period).mean()
        
        result = pd.DataFrame(index=data.index)
        result['your_indicator_value'] = ma_value
        
        return result
    
    def _generate_signal_logic(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成信号的具体逻辑"""
        # 示例：基于移动平均的简单信号
        if 'your_indicator_value' not in data.columns:
            return self._get_default_signal("缺少指标数据")
        
        current_value = data['your_indicator_value'].iloc[-1]
        prev_value = data['your_indicator_value'].iloc[-2]
        current_price = data['close'].iloc[-1] if 'close' in data.columns else None
        
        # 信号生成逻辑
        if current_value > prev_value:
            return {
                'signal_type': 'buy',
                'strength': 0.7,
                'confidence': 0.8,
                'reason': '指标上升',
                'price': current_price,
                'metadata': {
                    'current_value': current_value,
                    'prev_value': prev_value
                }
            }
        elif current_value < prev_value:
            return {
                'signal_type': 'sell',
                'strength': 0.7,
                'confidence': 0.8,
                'reason': '指标下降',
                'price': current_price,
                'metadata': {
                    'current_value': current_value,
                    'prev_value': prev_value
                }
            }
        else:
            return self._get_default_signal("无明显信号")
    
    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """获取默认的持有信号"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'reason': reason,
            'metadata': {}
        }
    
    def _validate_output_data(self, result: pd.DataFrame) -> None:
        """验证输出数据格式"""
        if result.empty:
            raise ValueError("计算结果为空")
        
        # 检查是否包含NaN值（前几行可能为NaN）
        if result.iloc[-1:].isnull().all().any():
            raise ValueError("计算结果包含无效值")
    
    def _validate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """验证信号格式"""
        required_fields = ['signal_type', 'strength', 'confidence']
        
        for field in required_fields:
            if field not in signal:
                signal[field] = 0.0 if field != 'signal_type' else 'hold'
        
        # 验证信号类型
        if signal['signal_type'] not in ['buy', 'sell', 'hold']:
            signal['signal_type'] = 'hold'
        
        # 验证数值范围
        signal['strength'] = max(0.0, min(1.0, signal.get('strength', 0.0)))
        signal['confidence'] = max(0.0, min(1.0, signal.get('confidence', 0.0)))
        
        return signal
```

## 🔧 **具体指标实现示例**

### **RSI指标完整实现**
```python
class RSIIndicator(BaseIndicator):
    """相对强弱指数(RSI)指标"""
    
    def __init__(self, period: int = 14):
        super().__init__("RSI")
        self.period = period
        
        if period <= 0:
            raise ValueError("RSI周期必须大于0")
    
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算RSI指标"""
        self._validate_input_data(data)
        
        close = data['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        result = pd.DataFrame(index=data.index)
        result['rsi_value'] = rsi
        
        return result
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成RSI交易信号"""
        if len(data) < 2:
            return self._get_default_signal("数据不足")
        
        latest_rsi = data['rsi_value'].iloc[-1]
        prev_rsi = data['rsi_value'].iloc[-2]
        
        # 超卖反弹信号
        if latest_rsi < 30 and prev_rsi >= 30:
            strength = min((30 - latest_rsi) / 10, 1.0)
            return {
                'signal_type': 'buy',
                'strength': strength,
                'confidence': 0.8,
                'reason': 'RSI进入超卖区域',
                'metadata': {'rsi_value': latest_rsi}
            }
        
        # 超买回调信号
        elif latest_rsi > 70 and prev_rsi <= 70:
            strength = min((latest_rsi - 70) / 10, 1.0)
            return {
                'signal_type': 'sell',
                'strength': strength,
                'confidence': 0.8,
                'reason': 'RSI进入超买区域',
                'metadata': {'rsi_value': latest_rsi}
            }
        
        # 持有信号
        else:
            return {
                'signal_type': 'hold',
                'strength': 0.0,
                'confidence': 0.5,
                'reason': 'RSI处于正常区间',
                'metadata': {'rsi_value': latest_rsi}
            }
```

## ⚠️ **常见错误与解决方案**

### **1. 数据验证错误**
```python
# ❌ 错误做法：不验证输入数据
def calculate(self, data):
    return data['close'].rolling(14).mean()

# ✅ 正确做法：完整的数据验证
def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    if 'close' not in data.columns:
        raise ValueError("缺少close列")
    if len(data) < self.period:
        raise ValueError("数据量不足")
    # 计算逻辑...
```

### **2. 信号格式错误**
```python
# ❌ 错误做法：返回不标准的信号格式
def get_signal(self, data):
    return "buy"  # 字符串格式

# ✅ 正确做法：返回标准字典格式
def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
    return {
        'signal_type': 'buy',
        'strength': 0.8,
        'confidence': 0.9,
        'reason': '具体原因'
    }
```

### **3. 异常处理缺失**
```python
# ❌ 错误做法：不处理异常
def calculate(self, data):
    return data['close'].rolling(self.period).mean()

# ✅ 正确做法：使用装饰器处理异常
@exception_handler(reraise=True)
def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    # 计算逻辑...
```

## 📊 **测试与验证**

### **单元测试模板**
```python
import unittest
import pandas as pd
import numpy as np

class TestYourIndicator(unittest.TestCase):
    def setUp(self):
        self.indicator = YourIndicator(period=14)
        self.sample_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_calculate_basic(self):
        """测试基本计算功能"""
        result = self.indicator.calculate(self.sample_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
    
    def test_get_signal_basic(self):
        """测试基本信号生成"""
        values = self.indicator.calculate(self.sample_data)
        signal = self.indicator.get_signal(values)
        
        self.assertIn('signal_type', signal)
        self.assertIn(signal['signal_type'], ['buy', 'sell', 'hold'])
        self.assertIsInstance(signal['strength'], (int, float))
        self.assertIsInstance(signal['confidence'], (int, float))
    
    def test_data_validation(self):
        """测试数据验证"""
        with self.assertRaises(ValueError):
            self.indicator.calculate(pd.DataFrame())  # 空数据
        
        with self.assertRaises(ValueError):
            self.indicator.calculate(pd.DataFrame({'volume': [1, 2, 3]}))  # 缺少close列
```

## 🎯 **最佳实践总结**

1. **严格遵循模板**：使用标准模板确保一致性
2. **完整的数据验证**：验证输入数据的类型、格式和完整性
3. **标准化输出格式**：确保输出符合项目规范
4. **异常处理**：使用装饰器处理异常，提供有意义的错误信息
5. **性能监控**：使用性能监控装饰器跟踪执行时间
6. **完整的文档**：提供清晰的类和方法文档字符串
7. **单元测试**：为每个指标编写完整的单元测试
8. **代码复用**：将通用逻辑抽象为私有方法

通过遵循这些实践，您的指标将能够完美集成到L4层架构中，为上游L5业务层提供可靠的服务。
