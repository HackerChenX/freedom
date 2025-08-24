#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小周期数据要求混入类

为所有技术指标定义最少需要的数据周期数，确保：
1. 每个指标明确定义最小数据要求
2. 统一的数据验证机制
3. 避免因数据不足导致的计算错误
4. 为双向验证提供准确的数据窗口参考
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class MinimumPeriodsMixin(ABC):
    """最小周期数据要求混入类"""
    
    @property
    @abstractmethod
    def minimum_periods(self) -> int:
        """
        返回指标计算所需的最少数据周期数
        
        Returns:
            int: 最少需要的数据周期数
            
        Note:
            - 这是指标能够产生有效计算结果的最小数据量
            - 包括预热期（warm-up period）的数据需求
            - 用于数据验证和双向验证的数据窗口确定
        """
        pass
    
    @property
    def recommended_periods(self) -> int:
        """
        返回推荐的数据周期数（通常是最小周期的2-3倍）
        
        Returns:
            int: 推荐的数据周期数
        """
        return max(self.minimum_periods * 2, 60)  # 至少60个周期
    
    @property
    def stable_periods(self) -> int:
        """
        返回指标稳定计算所需的数据周期数
        
        Returns:
            int: 稳定计算所需的数据周期数
        """
        return max(self.minimum_periods * 3, 120)  # 至少120个周期
    
    def validate_data_length(self, data: pd.DataFrame, strict: bool = False) -> Dict[str, Any]:
        """
        验证数据长度是否满足指标计算要求
        
        Args:
            data: 输入数据
            strict: 是否使用严格模式（推荐周期数）
            
        Returns:
            Dict: 验证结果
        """
        result = {
            'valid': False,
            'data_length': len(data),
            'minimum_required': self.minimum_periods,
            'recommended': self.recommended_periods,
            'stable': self.stable_periods,
            'validation_level': 'INSUFFICIENT',
            'message': ''
        }
        
        data_length = len(data)
        required_periods = self.recommended_periods if strict else self.minimum_periods
        
        if data_length >= self.stable_periods:
            result.update({
                'valid': True,
                'validation_level': 'STABLE',
                'message': f'数据充足，可进行稳定计算（{data_length}>={self.stable_periods}）'
            })
        elif data_length >= self.recommended_periods:
            result.update({
                'valid': True,
                'validation_level': 'RECOMMENDED',
                'message': f'数据满足推荐要求（{data_length}>={self.recommended_periods}）'
            })
        elif data_length >= self.minimum_periods:
            result.update({
                'valid': not strict,
                'validation_level': 'MINIMUM',
                'message': f'数据满足最低要求（{data_length}>={self.minimum_periods}）'
            })
        else:
            result.update({
                'valid': False,
                'validation_level': 'INSUFFICIENT',
                'message': f'数据不足，需要至少{self.minimum_periods}个周期，当前仅{data_length}个'
            })
        
        return result
    
    def get_period_requirements(self) -> Dict[str, int]:
        """
        获取所有周期要求
        
        Returns:
            Dict: 包含各种周期要求的字典
        """
        return {
            'minimum_periods': self.minimum_periods,
            'recommended_periods': self.recommended_periods,
            'stable_periods': self.stable_periods,
            'indicator_name': self.__class__.__name__
        }

class IndicatorPeriodRegistry:
    """指标周期要求注册表"""
    
    _registry = {}
    
    @classmethod
    def register_indicator_periods(cls, indicator_name: str, periods_info: Dict[str, int]):
        """注册指标的周期要求"""
        cls._registry[indicator_name] = periods_info
    
    @classmethod
    def get_indicator_periods(cls, indicator_name: str) -> Optional[Dict[str, int]]:
        """获取指标的周期要求"""
        return cls._registry.get(indicator_name)
    
    @classmethod
    def get_all_periods(cls) -> Dict[str, Dict[str, int]]:
        """获取所有指标的周期要求"""
        return cls._registry.copy()
    
    @classmethod
    def get_max_periods_by_type(cls) -> Dict[str, int]:
        """获取各类型的最大周期要求"""
        if not cls._registry:
            return {}
        
        max_periods = {
            'minimum': max(info['minimum_periods'] for info in cls._registry.values()),
            'recommended': max(info['recommended_periods'] for info in cls._registry.values()),
            'stable': max(info['stable_periods'] for info in cls._registry.values())
        }
        
        return max_periods

def calculate_minimum_periods_for_indicator(indicator_params: Dict[str, Any]) -> int:
    """
    根据指标参数计算最小周期数的通用函数
    
    Args:
        indicator_params: 指标参数字典
        
    Returns:
        int: 计算得出的最小周期数
    """
    # 提取常见的周期参数
    periods = []
    
    # 常见的周期参数名
    period_keys = [
        'period', 'periods', 'window', 'length', 'span',
        'fast_period', 'slow_period', 'signal_period',
        'short_period', 'long_period', 'ma_period',
        'ema_period', 'sma_period', 'lookback'
    ]
    
    for key in period_keys:
        if key in indicator_params:
            value = indicator_params[key]
            if isinstance(value, (int, float)) and value > 0:
                periods.append(int(value))
    
    if periods:
        # 返回最大周期数加上一些缓冲
        max_period = max(periods)
        return max_period + max(10, max_period // 2)  # 添加50%缓冲或至少10个周期
    else:
        # 默认最小周期
        return 30

def get_indicator_minimum_periods_mapping() -> Dict[str, int]:
    """
    获取所有已知指标的最小周期映射
    
    Returns:
        Dict: 指标名称到最小周期数的映射
    """
    return {
        # 核心指标 (P0)
        'MACD': 35,      # 慢线26 + 信号线9 = 35
        'RSI': 20,       # 周期14 + 缓冲6 = 20
        'KDJ': 18,       # K周期9 + D周期3 + 缓冲6 = 18
        'BOLL': 25,      # 周期20 + 缓冲5 = 25
        'MA': 25,        # 周期20 + 缓冲5 = 25
        'EMA': 25,       # 周期20 + 缓冲5 = 25
        
        # 重要指标 (P1)
        'ATR': 20,       # 周期14 + 缓冲6 = 20
        'CCI': 25,       # 周期20 + 缓冲5 = 25
        'MFI': 20,       # 周期14 + 缓冲6 = 20
        'OBV': 10,       # 累积指标，最小10个周期
        'STOCHRSI': 25,  # RSI14 + Stoch14 = 28，取25
        'AROON': 30,     # 周期25 + 缓冲5 = 30
        'ICHIMOKU': 55,  # 基准线26 + 转换线9 + 先行线52 = 55
        
        # 常用指标 (P2)
        'SAR': 15,       # 抛物线转向，最小15个周期
        'ADX': 20,       # 周期14 + 缓冲6 = 20
        'WMA': 25,       # 周期20 + 缓冲5 = 25
        'VORTEX': 20,    # 周期14 + 缓冲6 = 20
        'EMV': 20,       # 周期14 + 缓冲6 = 20
        'TRIX': 35,      # 三重指数平滑，需要更多数据
        'CMO': 20,       # 周期14 + 缓冲6 = 20
        'ROC': 15,       # 周期12 + 缓冲3 = 15
        
        # 专业指标 (P3)
        'KC': 25,        # 基于ATR，周期20 + 缓冲5 = 25
        'VIX': 30,       # 波动率指标，需要更多数据
        'VOLUME_RATIO': 15,  # 成交量比率，最小15个周期
        'ENHANCED_CCI': 30,  # 增强CCI，需要更多数据
        'ENHANCED_DMI': 25,  # 增强DMI，周期14 + 缓冲11 = 25
        
        # ZXM系列指标 (P4) - 通常需要更多数据
        'ZXM_WEEKLY_MACD': 50,    # 周线MACD，需要更多数据
        'ZXM_MONTHLY_MACD': 100,  # 月线MACD，需要更多数据
        'ZXM_TREND_SCORE': 60,    # 趋势评分，综合指标
        'ZXM_MARKET_SENTIMENT': 40,  # 市场情绪，需要足够样本
        'ZXM_FUND_FLOW': 30,      # 资金流向分析
        
        # 系统分析指标 (P5)
        'COMPOSITE_SCORE': 60,    # 综合评分，需要多个指标
        'PATTERN_RECOGNITION': 40, # 形态识别，需要足够样本
        'RISK_ASSESSMENT': 50,    # 风险评估，需要历史数据
    }

def validate_all_indicators_periods() -> Dict[str, Dict[str, Any]]:
    """
    验证所有指标的周期要求定义
    
    Returns:
        Dict: 验证结果
    """
    mapping = get_indicator_minimum_periods_mapping()
    validation_results = {}
    
    for indicator_name, min_periods in mapping.items():
        validation_results[indicator_name] = {
            'minimum_periods': min_periods,
            'recommended_periods': max(min_periods * 2, 60),
            'stable_periods': max(min_periods * 3, 120),
            'validation_status': 'DEFINED' if min_periods > 0 else 'UNDEFINED'
        }
    
    return validation_results

def get_unified_data_window_for_validation(indicator_names: list) -> int:
    """
    为多个指标确定统一的数据窗口大小
    
    Args:
        indicator_names: 指标名称列表
        
    Returns:
        int: 统一的数据窗口大小
    """
    mapping = get_indicator_minimum_periods_mapping()
    
    max_stable_periods = 0
    for indicator_name in indicator_names:
        if indicator_name in mapping:
            min_periods = mapping[indicator_name]
            stable_periods = max(min_periods * 3, 120)
            max_stable_periods = max(max_stable_periods, stable_periods)
    
    # 确保至少120个周期，最多300个周期
    return min(max(max_stable_periods, 120), 300)

def main():
    """演示最小周期要求系统"""
    
    print("📊 技术指标最小周期要求系统")
    print("=" * 60)
    
    # 显示所有指标的周期要求
    mapping = get_indicator_minimum_periods_mapping()
    validation_results = validate_all_indicators_periods()
    
    print(f"\n📋 已定义周期要求的指标数量: {len(mapping)}")
    
    # 按优先级分组显示
    priority_groups = {
        'P0核心指标': ['MACD', 'RSI', 'KDJ', 'BOLL', 'MA', 'EMA'],
        'P1重要指标': ['ATR', 'CCI', 'MFI', 'OBV', 'STOCHRSI', 'AROON', 'ICHIMOKU'],
        'P2常用指标': ['SAR', 'ADX', 'WMA', 'VORTEX', 'EMV', 'TRIX', 'CMO', 'ROC'],
        'P3专业指标': ['KC', 'VIX', 'VOLUME_RATIO', 'ENHANCED_CCI', 'ENHANCED_DMI']
    }
    
    for group_name, indicators in priority_groups.items():
        print(f"\n🔥 {group_name}:")
        for indicator in indicators:
            if indicator in validation_results:
                result = validation_results[indicator]
                print(f"  {indicator:20} 最小:{result['minimum_periods']:3d} 推荐:{result['recommended_periods']:3d} 稳定:{result['stable_periods']:3d}")
    
    # 显示统一数据窗口建议
    test_indicators = ['MACD', 'RSI', 'KDJ', 'BOLL']
    unified_window = get_unified_data_window_for_validation(test_indicators)
    print(f"\n🎯 多指标验证统一数据窗口建议: {unified_window}个周期")
    
    print(f"\n💡 使用建议:")
    print(f"  - 单指标验证: 使用各指标的稳定周期数")
    print(f"  - 多指标验证: 使用统一数据窗口")
    print(f"  - 双向验证: 确保使用相同的数据窗口大小")

if __name__ == "__main__":
    main()
