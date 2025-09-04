---
type: "agent_requested"
description: "Example description"
---

# 指标开发规范（105指标系统）

## 🎯 指标系统概览

当前系统支持105个指标，分为9大类别：
- **Core Indicators (6个)**: MA, EMA, MACD, RSI, BOLL, PSY
- **Trend Indicators (10个)**: DMA, DMI, ADX, AROON, SAR, TRIX, CCI, 等
- **Oscillator Indicators (9个)**: KDJ, WR, CMO, STOCHRSI, 等
- **Volume Indicators (9个)**: OBV, AD, EMV, VOL, VR, 等
- **Volatility Indicators (4个)**: ATR, KC, VIX, STDDEV
- **ZXM System Indicators (35个)**: 专业交易系统指标
- **Pattern Recognition (21个)**: 蜡烛图形态识别
- **Enhanced Indicators (3个)**: 增强版指标
- **Professional Indicators (15个)**: 高级技术分析工具

## 📋 指标开发标准模板

### 基础指标类结构
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.performance_monitor import performance_monitor
from utils.exception_handler import exception_handler

logger = get_logger(__name__)

class BaseIndicator(ABC):
    """
    指标基类 - 所有指标必须继承此类
    
    Attributes:
        name (str): 指标名称
        period (int): 计算周期
        params (Dict): 指标参数
    """
    
    def __init__(self, name: str, period: int = 20, **kwargs):
        """
        初始化指标
        
        Args:
            name: 指标名称
            period: 计算周期
            **kwargs: 其他参数
        """
        self.name = name
        self.period = period
        self.params = kwargs
        self._validate_params()
    
    def _validate_params(self):
        """验证参数有效性"""
        if self.period <= 0:
            raise ValueError(f"指标 {self.name} 的周期必须大于0")
    
    @abstractmethod
    @performance_monitor(threshold_seconds=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 股票数据，必须包含 [open, high, low, close, volume] 列
            
        Returns:
            pd.DataFrame: 包含指标计算结果的数据框
            
        Raises:
            ValueError: 数据格式不正确
            InsufficientDataError: 数据量不足
        """
        pass
    
    @abstractmethod
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取交易信号
        
        Args:
            data: 包含指标值的数据框
            
        Returns:
            Dict[str, Any]: 交易信号信息
            {
                'signal': 'BUY'|'SELL'|'HOLD',
                'strength': float,  # 信号强度 0-1
                'confidence': float,  # 置信度 0-1
                'details': Dict  # 详细信息
            }
        """
        pass
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """
        获取指标模式信息
        
        Returns:
            Dict[str, Any]: 模式信息
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'period': self.period,
            'params': self.params,
            'category': self._get_category()
        }
    
    @abstractmethod
    def _get_category(self) -> str:
        """返回指标类别"""
        pass
```

### 具体指标实现示例
```python
class MovingAverage(BaseIndicator):
    """
    移动平均线指标
    
    计算公式：MA(n) = (C1 + C2 + ... + Cn) / n
    """
    
    def __init__(self, period: int = 20):
        super().__init__("MA", period)
    
    @performance_monitor(threshold_seconds=1.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线"""
        if len(data) < self.period:
            raise InsufficientDataError(f"数据量不足，需要至少{self.period}条数据")
        
        result = data.copy()
        result[f'MA_{self.period}'] = data['close'].rolling(
            window=self.period, min_periods=self.period
        ).mean()
        
        logger.info(f"MA_{self.period} 计算完成，数据行数: {len(result)}")
        return result
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取MA交易信号"""
        if f'MA_{self.period}' not in data.columns:
            raise ValueError("数据中缺少MA指标值")
        
        current_price = data['close'].iloc[-1]
        current_ma = data[f'MA_{self.period}'].iloc[-1]
        
        if pd.isna(current_ma):
            return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0}
        
        # 价格突破MA线
        if current_price > current_ma:
            strength = min((current_price - current_ma) / current_ma, 1.0)
            return {
                'signal': 'BUY',
                'strength': strength,
                'confidence': 0.7,
                'details': {
                    'current_price': current_price,
                    'ma_value': current_ma,
                    'breakout': True
                }
            }
        elif current_price < current_ma:
            strength = min((current_ma - current_price) / current_ma, 1.0)
            return {
                'signal': 'SELL',
                'strength': strength,
                'confidence': 0.7,
                'details': {
                    'current_price': current_price,
                    'ma_value': current_ma,
                    'breakdown': True
                }
            }
        
        return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.5}
    
    def _get_category(self) -> str:
        return 'CORE'
```

## 🏗️ 指标注册系统

### 注册机制
所有指标必须在 [complete_indicator_registry.py](mdc:indicators/complete_indicator_registry.py) 中注册：

```python
# 核心指标注册
CORE_INDICATORS = {
    'MA': 'indicators.ma.MovingAverage',
    'EMA': 'indicators.ema.ExponentialMovingAverage',
    'MACD': 'indicators.macd.MACD',
    'RSI': 'indicators.rsi.RelativeStrengthIndex',
    'BOLL': 'indicators.boll.BollingerBands',
    'PSY': 'indicators.psy.PsychologicalLine'
}

# 趋势指标注册
TREND_INDICATORS = {
    'DMA': 'indicators.trend.dma.DifferenceMovingAverage',
    'DMI': 'indicators.trend.enhanced_dmi.EnhancedDMI',
    'ADX': 'indicators.trend.adx.AverageDirectionalIndex',
    # ... 更多指标
}

def register_all_indicators():
    """注册所有105个指标"""
    total_registered = 0
    successful_registrations = 0
    
    # 批量注册各类指标
    for category, indicators in [
        ('CORE', CORE_INDICATORS),
        ('TREND', TREND_INDICATORS),
        ('OSCILLATOR', OSCILLATOR_INDICATORS),
        # ... 其他类别
    ]:
        logger.info(f"正在注册 {category} 类指标...")
        for name, implementation_path in indicators.items():
            if register_indicator(name, implementation_path):
                successful_registrations += 1
            total_registered += 1
    
    success_rate = (successful_registrations / total_registered) * 100
    logger.info(f"指标注册完成: {successful_registrations}/{total_registered} ({success_rate:.1f}%)")
    return success_rate >= 90.0  # 要求至少90%成功率
```

### Mock指标实现
```python
def create_mock_indicator(name: str) -> type:
    """为缺失的指标创建Mock实现"""
    
    class MockIndicator(BaseIndicator):
        def __init__(self, period: int = 20):
            super().__init__(name, period)
        
        def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
            result = data.copy()
            # 生成合理的模拟数据
            result[f'{name}_VALUE'] = np.random.normal(0, 1, len(data))
            return result
        
        def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'details': {'mock': True, 'indicator': name}
            }
        
        def _get_category(self) -> str:
            return 'MOCK'
    
    MockIndicator.__name__ = f'Mock{name}Indicator'
    return MockIndicator
```

## 🔧 指标开发工具

### 指标验证器
```python
class IndicatorValidator:
    """指标验证工具"""
    
    @staticmethod
    def validate_indicator(indicator: BaseIndicator, test_data: pd.DataFrame) -> Dict[str, Any]:
        """验证指标实现"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'performance': {}
        }
        
        try:
            # 性能测试
            start_time = time.time()
            result = indicator.calculate(test_data)
            calculation_time = time.time() - start_time
            
            validation_results['performance']['calculation_time'] = calculation_time
            
            # 数据完整性检查
            if result.isnull().sum().sum() > len(result) * 0.1:
                validation_results['warnings'].append("超过10%的数据为空值")
            
            # 信号生成测试
            signal = indicator.get_signal(result)
            if not all(key in signal for key in ['signal', 'strength', 'confidence']):
                validation_results['errors'].append("信号格式不完整")
                validation_results['valid'] = False
            
        except Exception as e:
            validation_results['errors'].append(f"指标计算失败: {str(e)}")
            validation_results['valid'] = False
        
        return validation_results
```

### 批量测试工具
```python
def batch_test_indicators(indicator_names: List[str], test_data: pd.DataFrame) -> Dict[str, Any]:
    """批量测试指标"""
    test_results = {}
    
    for name in indicator_names:
        try:
            indicator = get_indicator(name)
            if indicator:
                validator = IndicatorValidator()
                result = validator.validate_indicator(indicator, test_data)
                test_results[name] = result
                logger.info(f"指标 {name} 测试完成")
            else:
                test_results[name] = {'valid': False, 'errors': ['指标未找到']}
        except Exception as e:
            test_results[name] = {'valid': False, 'errors': [str(e)]}
    
    return test_results
```

## 🎨 ZXM系统指标规范

### ZXM指标基类
```python
class BaseZXMIndicator(BaseIndicator):
    """ZXM系统指标基类"""
    
    def __init__(self, name: str, period: int = 20, **kwargs):
        super().__init__(f"ZXM_{name}", period, **kwargs)
    
    def _get_category(self) -> str:
        return 'ZXM'
    
    @abstractmethod
    def get_buypoint_score(self, data: pd.DataFrame) -> float:
        """计算买点分数"""
        pass
```

### ZXM指标实现示例
```python
class ZXMVolumeIndicator(BaseZXMIndicator):
    """ZXM成交量指标"""
    
    def __init__(self, volume_threshold: float = 1.5):
        super().__init__("VOLUME", volume_threshold=volume_threshold)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        
        # 计算成交量相对强度
        vol_ma = data['volume'].rolling(window=20).mean()
        result['ZXM_VOLUME_STRENGTH'] = data['volume'] / vol_ma
        
        return result
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        current_strength = data['ZXM_VOLUME_STRENGTH'].iloc[-1]
        threshold = self.params['volume_threshold']
        
        if current_strength > threshold:
            return {
                'signal': 'BUY',
                'strength': min(current_strength / threshold, 1.0),
                'confidence': 0.8,
                'details': {'volume_strength': current_strength}
            }
        
        return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.5}
    
    def get_buypoint_score(self, data: pd.DataFrame) -> float:
        signal = self.get_signal(data)
        return signal['strength'] * signal['confidence']
```

## 🚀 性能优化要求

### 计算性能
- 单个指标计算时间不超过2秒
- 批量指标计算使用向量化操作
- 大数据集使用分块处理

### 内存使用
- 避免不必要的数据复制
- 使用适当的数据类型
- 及时释放临时变量

### 缓存策略
```python
from functools import lru_cache

class CachedIndicator(BaseIndicator):
    @lru_cache(maxsize=128)
    def _cached_calculate(self, data_hash: str, data: pd.DataFrame) -> pd.DataFrame:
        return self._raw_calculate(data)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        data_hash = pd.util.hash_pandas_object(data).sum()
        return self._cached_calculate(str(data_hash), data)
```

## ✅ 质量检查清单

新指标开发必须通过以下检查：

- [ ] 继承正确的基类（BaseIndicator或BaseZXMIndicator）
- [ ] 实现所有抽象方法
- [ ] 包含完整的文档字符串
- [ ] 添加性能监控装饰器
- [ ] 包含异常处理
- [ ] 通过单元测试
- [ ] 注册到指标注册表
- [ ] 性能测试通过（<2秒）
- [ ] 信号生成测试通过
- [ ] 代码覆盖率>80%

这些规范确保我们的105指标系统保持高质量和一致性。
description:
globs:
alwaysApply: true
---
