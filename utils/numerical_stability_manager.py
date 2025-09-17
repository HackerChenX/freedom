#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数值稳定性管理器

专业金融量化交易系统的数值稳定性保障模块
用于处理技术指标计算中的数值精度、边界情况和异常值检测
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from decimal import Decimal, getcontext
from utils.logger import get_logger

logger = get_logger(__name__)

class NumericalStabilityManager:
    """
    数值稳定性管理器 - 金融级精度控制

    专业量化交易系统要求：
    - 6位小数精度保障
    - 边界情况处理
    - 数值溢出检测
    - 精度损失预防
    """

    def __init__(self, precision: int = 6, decimal_context_precision: int = 28):
        """
        初始化数值稳定性管理器

        Args:
            precision: 输出精度（小数位数）
            decimal_context_precision: Decimal计算精度
        """
        self.precision = precision
        self.decimal_context_precision = decimal_context_precision

        # 设置高精度计算上下文
        getcontext().prec = decimal_context_precision

        # 数值边界定义
        self.EXTREME_VALUE_THRESHOLD = 1e10
        self.MINIMAL_VALUE_THRESHOLD = 1e-10
        self.ZERO_THRESHOLD = 1e-12

    def ensure_series_precision(self, series: pd.Series, round_precision: Optional[int] = None) -> pd.Series:
        """
        确保Series的数值精度

        Args:
            series: 输入Series
            round_precision: 四舍五入精度（如果不指定，使用默认精度）

        Returns:
            pd.Series: 精度修正后的Series
        """
        if round_precision is None:
            round_precision = self.precision

        return series.round(round_precision)

    def check_and_fix_extreme_values(self, series: pd.Series,
                                   name: str = "Series",
                                   extreme_threshold: Optional[float] = None) -> pd.Series:
        """
        检查并修正极值

        Args:
            series: 输入Series
            name: 数据名称（用于日志）
            extreme_threshold: 极值阈值

        Returns:
            pd.Series: 修正后的Series
        """
        if extreme_threshold is None:
            extreme_threshold = self.EXTREME_VALUE_THRESHOLD

        # 检查无穷大值
        inf_mask = np.isinf(series)
        if inf_mask.any():
            logger.warning(f"{name} 存在{inf_mask.sum()}个无穷大值，已修正为NaN")
            series = series.copy()
            series[inf_mask] = np.nan

        # 检查过大值（可能的数值溢出）
        extreme_mask = (np.abs(series) > extreme_threshold) & ~pd.isna(series)
        if extreme_mask.any():
            logger.warning(f"{name} 存在{extreme_mask.sum()}个极值（>{extreme_threshold}），已修正为NaN")
            series = series.copy()
            series[extreme_mask] = np.nan

        return series

    def safe_division(self, numerator: Union[float, Decimal, pd.Series],
                     denominator: Union[float, Decimal, pd.Series],
                     default_value: float = np.nan) -> Union[float, pd.Series]:
        """
        安全除法运算

        Args:
            numerator: 分子
            denominator: 分母
            default_value: 分母为0时的默认值

        Returns:
            除法结果
        """
        if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
            # Series运算
            if not isinstance(numerator, pd.Series):
                numerator = pd.Series(numerator, index=denominator.index)
            if not isinstance(denominator, pd.Series):
                denominator = pd.Series(denominator, index=numerator.index)

            # 处理分母为0或接近0的情况
            zero_mask = (np.abs(denominator) < self.ZERO_THRESHOLD) | pd.isna(denominator)
            result = numerator / denominator
            result[zero_mask] = default_value

            return self.ensure_series_precision(result)

        else:
            # 标量运算
            if abs(denominator) < self.ZERO_THRESHOLD or pd.isna(denominator):
                return default_value

            result = float(Decimal(str(numerator)) / Decimal(str(denominator)))
            return round(result, self.precision)

    def safe_sqrt(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        安全开方运算

        Args:
            value: 输入值

        Returns:
            开方结果
        """
        if isinstance(value, pd.Series):
            # 确保非负值
            negative_mask = value < 0
            if negative_mask.any():
                logger.warning(f"开方运算存在{negative_mask.sum()}个负值，已修正为NaN")
                value = value.copy()
                value[negative_mask] = np.nan

            result = np.sqrt(value)
            return self.ensure_series_precision(result)

        else:
            if value < 0:
                logger.warning(f"开方运算存在负值{value}，返回NaN")
                return np.nan

            result = float(Decimal(str(value)).sqrt())
            return round(result, self.precision)

    def safe_log(self, value: Union[float, pd.Series],
                base: Optional[float] = None) -> Union[float, pd.Series]:
        """
        安全对数运算

        Args:
            value: 输入值
            base: 对数底数（默认自然对数）

        Returns:
            对数结果
        """
        if isinstance(value, pd.Series):
            # 确保正值
            non_positive_mask = value <= 0
            if non_positive_mask.any():
                logger.warning(f"对数运算存在{non_positive_mask.sum()}个非正值，已修正为NaN")
                value = value.copy()
                value[non_positive_mask] = np.nan

            if base is None:
                result = np.log(value)
            else:
                result = np.log(value) / np.log(base)

            return self.ensure_series_precision(result)

        else:
            if value <= 0:
                logger.warning(f"对数运算存在非正值{value}，返回NaN")
                return np.nan

            if base is None:
                result = float(Decimal(str(value)).ln())
            else:
                result = float(Decimal(str(value)).ln() / Decimal(str(base)).ln())

            return round(result, self.precision)

    def validate_indicator_range(self, series: pd.Series,
                               name: str,
                               min_value: Optional[float] = None,
                               max_value: Optional[float] = None) -> pd.Series:
        """
        验证指标值范围

        Args:
            series: 输入Series
            name: 指标名称
            min_value: 最小有效值
            max_value: 最大有效值

        Returns:
            验证后的Series
        """
        result = series.copy()

        if min_value is not None:
            below_min = (series < min_value) & ~pd.isna(series)
            if below_min.any():
                logger.warning(f"{name} 存在{below_min.sum()}个值低于最小值{min_value}")
                result[below_min] = min_value

        if max_value is not None:
            above_max = (series > max_value) & ~pd.isna(series)
            if above_max.any():
                logger.warning(f"{name} 存在{above_max.sum()}个值高于最大值{max_value}")
                result[above_max] = max_value

        return self.ensure_series_precision(result)

    def validate_rsi_range(self, rsi_series: pd.Series) -> pd.Series:
        """RSI专用范围验证（0-100）"""
        return self.validate_indicator_range(rsi_series, "RSI", 0.0, 100.0)

    def validate_percentage_range(self, series: pd.Series, name: str) -> pd.Series:
        """百分比指标范围验证（0-100）"""
        return self.validate_indicator_range(series, name, 0.0, 100.0)

    def detect_calculation_anomalies(self, series: pd.Series,
                                   name: str) -> Dict[str, int]:
        """
        检测计算异常

        Args:
            series: 输入Series
            name: 数据名称

        Returns:
            异常统计字典
        """
        anomalies = {}

        # 检测NaN值
        nan_count = series.isna().sum()
        anomalies['nan_count'] = nan_count

        # 检测无穷大值
        inf_count = np.isinf(series).sum()
        anomalies['inf_count'] = inf_count

        # 检测极值
        extreme_count = (np.abs(series) > self.EXTREME_VALUE_THRESHOLD).sum()
        anomalies['extreme_count'] = extreme_count

        # 检测零值
        zero_count = (np.abs(series) < self.ZERO_THRESHOLD).sum()
        anomalies['zero_count'] = zero_count

        # 记录异常日志
        if any(anomalies.values()):
            logger.info(f"{name} 计算异常统计: {anomalies}")

        return anomalies

    def precision_controlled_operation(self, operation_func, *args, **kwargs):
        """
        精度控制的运算操作

        Args:
            operation_func: 运算函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            精度控制后的结果
        """
        try:
            # 执行运算
            result = operation_func(*args, **kwargs)

            # 确保精度
            if isinstance(result, pd.Series):
                return self.ensure_series_precision(result)
            elif isinstance(result, (int, float)):
                return round(float(result), self.precision)
            else:
                return result

        except Exception as e:
            logger.error(f"精度控制运算失败: {e}")
            # 返回安全的默认值
            if len(args) > 0 and isinstance(args[0], pd.Series):
                return pd.Series(index=args[0].index, dtype='float64').fillna(np.nan)
            else:
                return np.nan

# 创建全局实例
_stability_manager = NumericalStabilityManager()

def get_stability_manager() -> NumericalStabilityManager:
    """获取数值稳定性管理器实例"""
    return _stability_manager

# 便捷函数
def ensure_precision(series: pd.Series, precision: int = 6) -> pd.Series:
    """确保Series精度"""
    return _stability_manager.ensure_series_precision(series, precision)

def safe_divide(numerator, denominator, default=np.nan):
    """安全除法"""
    return _stability_manager.safe_division(numerator, denominator, default)

def validate_rsi(rsi_series: pd.Series) -> pd.Series:
    """验证RSI范围"""
    return _stability_manager.validate_rsi_range(rsi_series)

def check_anomalies(series: pd.Series, name: str) -> Dict[str, int]:
    """检测异常"""
    return _stability_manager.detect_calculation_anomalies(series, name)