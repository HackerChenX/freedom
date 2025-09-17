"""
指标配置管理
集中管理所有指标的配置参数，消除硬编码
"""

from typing import Dict, Any


class IndicatorConfig:
    """指标配置类"""
    
    # 默认周期配置
    DEFAULT_PERIODS = {
        'short': 5,
        'medium': 20,
        'long': 60,
        'extra_long': 120
    }
    
    # 阈值配置
    THRESHOLDS = {
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'macd_signal_threshold': 0.1,
        'volume_spike_ratio': 2.0,
        'price_change_threshold': 0.05
    }
    
    # 性能配置
    PERFORMANCE = {
        'max_calculation_time': 2.0,
        'cache_ttl': 300,
        'batch_size': 1000
    }
    
    # 数据验证配置
    VALIDATION = {
        'min_data_points': 10,
        'required_columns': ['close', 'open', 'high', 'low', 'volume'],
        'max_missing_ratio': 0.1
    }
    
    @classmethod
    def get_period(cls, period_type: str) -> int:
        """获取周期配置"""
        return cls.DEFAULT_PERIODS.get(period_type, 20)
    
    @classmethod
    def get_threshold(cls, threshold_name: str) -> float:
        """获取阈值配置"""
        return cls.THRESHOLDS.get(threshold_name, 0.0)
    
    @classmethod
    def get_performance_setting(cls, setting_name: str) -> Any:
        """获取性能配置"""
        return cls.PERFORMANCE.get(setting_name)
    
    @classmethod
    def get_validation_setting(cls, setting_name: str) -> Any:
        """获取验证配置"""
        return cls.VALIDATION.get(setting_name)


# 全局配置实例
indicator_config = IndicatorConfig()
