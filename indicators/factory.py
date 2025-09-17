from utils.container import container

#!/usr/bin/env python3
from utils.logger import get_logger

"""
指标工厂模块

提供指标创建和管理功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorFactory(BaseIndicator):
    """
    指标工厂类

    完全基于Complete_indicator_registry实现的统一指标工厂
    """

    def __init__(self):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        from indicators.base_indicator import BaseIndicator

        self._registry = complete_registry

    def register_indicator_Factory(self, name: str, indicator_class):
        """注册指标类"""
        return self._registry.register_indicator_safe(indicator_class, name)

    def create_indicator_Factory_Factory_Factory_factory(self, name: str, **kwargs):
        """创建指标实例"""
        return self._registry.create_indicator_Factory_Factory_Factory_factory(name, **kwargs)

    def get_available_indicators_Factory(self) -> List[str]:
        """获取可用指标列表"""
        return list(self._registry.get_all_indicators().keys())

    def is_indicator_available(self, name: str) -> bool:
        """检查指标是否可用"""
        return name in self._registry.get_all_indicators()

    def get_all_registered_indicators(self) -> List[str]:
        """获取所有已注册的指标名称"""
        return list(self._registry.get_all_indicators().keys())

    @classmethod
    def get_supported_indicators(cls) -> List[str]:
        """获取支持的指标列表"""
        from indicators.base_indicator import BaseIndicator

        return list(complete_registry.get_all_indicators().keys())

    @classmethod
    def create_indicator_Factory_Factory_Factory_factory_duplicate(cls, name: str, **kwargs):
        """类方法创建指标实例"""
        from indicators.base_indicator import BaseIndicator

        return complete_registry.create_indicator_Factory_Factory_Factory_factory(name, **kwargs)


class IndicatorFactory:
    """
    FACTORY 指标

    自动生成的最小化实现，支持参数标准化
    """

    def _get_default_parameters_factory(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中

    def set_parameters_Factory_Factory(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            from db.sql_manager import SQLManager, QueryType

            validator = IndicatorParameterValidator()

            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)

            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters("FACTORY", params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass

            # 设置参数
            self.period = params.get("period", 14)  # TODO: 将魔法数字提取到配置中

        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            self.period = 14  # TODO: 将魔法数字提取到配置中

    def calculate_Factory_Factory(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算FACTORY指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了FACTORY指标的Data_frame
        """
        result = self._calculate_factory(data, **kwargs)
        self._result = result
        return result

    def _calculate_factory(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算FACTORY指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了FACTORY指标的Data_frame
        """
        df = data.copy()

        # 最小化实现：返回原数据加上一个简单的计算列
        df[f"FACTORY_VALUE"] = df["close"].rolling(window=self.period).mean()

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df

    def calculate_raw_score_Factory_Factory(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Factory_Factory(data, **kwargs)
        return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

    def calculate_confidence_Factory_Factory(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5  # TODO: 将魔法数字提取到配置中

    def get_patterns_Factory_Factory(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)

    @property
    def minimum_periods(self) -> int:
        """
        Factory指标所需的最少数据周期数

        计算逻辑：使用默认值

        Returns:
            int: 最少需要的数据周期数
        """
        return 30  # TODO: 将魔法数字提取到配置中

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值

        Args:
            data: 输入数据，包含OHLCV等字段

        Returns:
            pd.DataFrame: 包含指标计算结果的数据框
        """
        if not self.validate_data(data):
            raise ValueError("输入数据不符合要求")

        # 预处理数据
        processed_data = self.preprocess_data(data)

        # TODO: 实现具体的指标计算逻辑
        result = processed_data.copy()
        result[f"{self.name}_value"] = processed_data["close"].rolling(window=self.period).mean()

        # 后处理结果
        result = self.postprocess_result(result)

        # 保存结果
        self._result = result

        return result

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取交易信号

        Args:
            data: 包含指标计算结果的数据

        Returns:
            Dict[str, Any]: 交易信号信息
        """
        if data.empty:
            return {"signal": "hold", "strength": 0.0, "timestamp": None}

        # TODO: 实现具体的信号生成逻辑
        latest_close = data["close"].iloc[-1] if "close" in data.columns else 0

        return {
            "signal": "hold",
            "strength": 0.0,
            "timestamp": data.index[-1] if not data.empty else None,
            "price": latest_close,
            "indicator": self.name,
        }
