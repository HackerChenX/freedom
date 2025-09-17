from indicators.base_indicator import BaseIndicator

#!/usr/bin/env python3
"""
优化的ZXM指标计算器

使用向量化计算和NumPy优化，大幅提升ZXM指标计算性能

作者：AI Assistant
创建时间：2025-01-13  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import numba
from numba import jit, prange


class OptimizedZXMCalculator(BaseIndicator):
    """优化的ZXM计算器"""

    @staticmethod
    @jit(nopython=True, cache=True)
    def _fast_sma(values: np.ndarray, period: int, weight: int) -> np.ndarray:
        """
        快速SMA计算（JIT编译）

        Args:
            values: 输入数值数组
            period: 周期
            weight: 权重

        Returns:
            np.ndarray: SMA结果
        """
        result = np.zeros_like(values)
        result[0] = values[0]

        for i in range(1, len(values)):
            result[i] = (weight * values[i] + (period - weight) * result[i - 1]) / period

        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def _fast_rolling_min_max(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        快速滚动最小值和最大值计算

        Args:
            values: 输入数值数组
            window: 滚动窗口大小

        Returns:
            Tuple[np.ndarray, np.ndarray]: (最小值数组, 最大值数组)
        """
        n = len(values)
        min_vals = np.full(n, np.nan)
        max_vals = np.full(n, np.nan)

        for i in range(window - 1, n):
            start_idx = i - window + 1
            min_vals[i] = np.min(values[start_idx : i + 1])
            max_vals[i] = np.max(values[start_idx : i + 1])

        return min_vals, max_vals

    @staticmethod
    @jit(nopython=True, cache=True)
    def _fast_filter_condition(condition: np.ndarray, window: int) -> np.ndarray:
        """
        快速FILTER条件计算

        Args:
            condition: 条件数组
            window: 过滤窗口

        Returns:
            np.ndarray: 过滤结果
        """
        n = len(condition)
        result = np.zeros(n, dtype=np.bool_)

        for i in range(window, n):
            # 检查过去window周期内是否有满足条件的
            has_condition = False
            for j in range(i - window + 1, i + 1):
                if condition[j]:
                    has_condition = True
                    break
            result[i] = has_condition

        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def _fast_count_condition(condition: np.ndarray, window: int) -> np.ndarray:
        """
        快速COUNT条件计算

        Args:
            condition: 条件数组
            window: 计数窗口

        Returns:
            np.ndarray: 计数结果
        """
        n = len(condition)
        result = np.zeros(n, dtype=np.int32)

        for i in range(window, n):
            count = 0
            for j in range(i - window + 1, i + 1):
                if condition[j]:
                    count += 1
            result[i] = count

        return result

    def calculate_optimized_zxm(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        优化的ZXM指标计算

        Args:
            data: 输入数据，包含OHLC列

        Returns:
            pd.DataFrame: 计算结果
        """
        if len(data) < 70:  # 最少需要70行数据  # TODO: 将魔法数字提取到配置中
            return pd.DataFrame()

        result = data.copy()

        # 转换为NumPy数组以提升性能
        close_vals = data["close"].values
        high_vals = data["high"].values
        low_vals = data["low"].values

        # 1. 计算LLV和HHV（向量化）
        llv_55, hhv_55 = self._fast_rolling_min_max(low_vals, 55)  # TODO: 将魔法数字提取到配置中

        # 2. 计算RSV变种（向量化）
        divisor = hhv_55 - llv_55
        valid_mask = divisor > 0
        rsv_55 = np.full_like(close_vals, 0.0)
        rsv_55[valid_mask] = (close_vals[valid_mask] - llv_55[valid_mask]) / divisor[valid_mask] * 100

        # 3. 计算V11（优化的SMA）  # TODO: 将魔法数字提取到配置中
        sma_rsv_5 = self._fast_sma(rsv_55, 5, 1)  # TODO: 将魔法数字提取到配置中
        sma_sma_3 = self._fast_sma(sma_rsv_5, 3, 1)  # TODO: 将魔法数字提取到配置中
        v11 = 3 * sma_rsv_5 - 2 * sma_sma_3  # TODO: 将魔法数字提取到配置中

        # 4. 计算V11的EMA（使用pandas的向量化EMA）  # TODO: 将魔法数字提取到配置中
        v11_series = pd.Series(v11, index=data.index)
        ema_v11_3 = v11_series.ewm(span=3, adjust=False).mean().values  # TODO: 将魔法数字提取到配置中

        # 5. 计算V12（向量化）  # TODO: 将魔法数字提取到配置中
        v12 = np.full_like(close_vals, 0.0)
        valid_prev = ema_v11_3[:-1] != 0
        if len(valid_prev) > 0:
            v12[1:][valid_prev] = (ema_v11_3[1:] - ema_v11_3[:-1])[valid_prev] / ema_v11_3[:-1][valid_prev] * 100

        # 6. 计算AA和BB条件（优化的布尔运算）  # TODO: 将魔法数字提取到配置中
        aa_base = ema_v11_3 <= 13  # TODO: 将魔法数字提取到配置中
        aa_filter = self._fast_filter_condition(aa_base, 15)  # TODO: 将魔法数字提取到配置中
        aa = aa_base & aa_filter

        bb_base = (ema_v11_3 <= 13) & (v12 > 13)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        bb_filter = self._fast_filter_condition(bb_base, 10)
        bb = bb_base & bb_filter

        # 7. 计算XG（优化的计数）  # TODO: 将魔法数字提取到配置中
        combined_condition = aa | bb
        xg = self._fast_count_condition(combined_condition, 6)  # TODO: 将魔法数字提取到配置中

        # 8. 添加结果到DataFrame  # TODO: 将魔法数字提取到配置中
        result["V11"] = v11
        result["EMA_V11_3"] = ema_v11_3
        result["V12"] = v12
        result["AA"] = aa
        result["BB"] = bb
        result["XG"] = xg

        # 9. 生成信号（向量化）  # TODO: 将魔法数字提取到配置中
        result["buy_signal"] = xg > 0
        result["sell_signal"] = xg == 0
        result["hold_signal"] = xg == 0

        return result

    def batch_calculate_zxm(self, stock_data: dict) -> dict:
        """
        批量计算多只股票的ZXM指标

        Args:
            stock_data: {股票代码: DataFrame} 映射

        Returns:
            dict: {股票代码: 计算结果} 映射
        """
        results = {}

        for code, data in stock_data.items():
            try:
                if len(data) >= 70:  # TODO: 将魔法数字提取到配置中
                    result = self.calculate_optimized_zxm(data)
                    if not result.empty:
                        results[code] = result
            except Exception:
                # 静默跳过错误股票
                continue

        return results


# 全局优化计算器实例
_calculator_instance = None


def get_optimized_zxm_calculator() -> OptimizedZXMCalculator:
    """获取优化ZXM计算器单例"""
    global _calculator_instance

    if _calculator_instance is None:
        _calculator_instance = OptimizedZXMCalculator()

    return _calculator_instance
