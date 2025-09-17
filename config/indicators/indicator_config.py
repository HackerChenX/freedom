"""
指标集中化配置管理
提供统一的指标参数配置和管理机制
"""

from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorConfigManager:
    """指标配置管理器"""

    def __init__(self):
        self._configs = self._load_default_configs()

    def _load_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """加载默认配置"""
        return {
            'MA': {
                'period': 20,
                'price_field': 'close',
                'min_periods': 1
            },
            'EMA': {
                'period': 12,
                'alpha': None,
                'price_field': 'close'
            },
            'MACD': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'price_field': 'close'
            },
            'RSI': {
                'period': 14,
                'price_field': 'close',
                'overbought': 70,
                'oversold': 30
            },
            'BOLLINGER': {
                'period': 20,
                'std_dev': 2,
                'price_field': 'close'
            }
        }

    def get_config(self, indicator_name: str) -> Optional[Dict[str, Any]]:
        """获取指标配置"""
        return self._configs.get(indicator_name.upper())

    def set_config(self, indicator_name: str, config: Dict[str, Any]):
        """设置指标配置"""
        self._configs[indicator_name.upper()] = config
        logger.info(f"指标 {indicator_name} 配置已更新")

    def update_config(self, indicator_name: str, updates: Dict[str, Any]):
        """更新指标配置"""
        if indicator_name.upper() in self._configs:
            self._configs[indicator_name.upper()].update(updates)
            logger.info(f"指标 {indicator_name} 配置已更新")
        else:
            logger.warning(f"指标 {indicator_name} 配置不存在")

    def get_parameter(self, indicator_name: str, parameter_name: str, default_value: Any = None) -> Any:
        """获取指标参数"""
        config = self.get_config(indicator_name)
        if config:
            return config.get(parameter_name, default_value)
        return default_value

    def list_indicators(self) -> list:
        """列出所有配置的指标"""
        return list(self._configs.keys())


# 全局配置管理器实例
indicator_config_manager = IndicatorConfigManager()
