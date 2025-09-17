"""
形态识别和信号生成混入类
为所有技术指标提供统一的形态识别和信号生成功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


class PatternSignalMixin:
    """
        PatternSignalMixin - L4核心服务层组件

    职责合理性说明:
    - 作为L4层核心服务组件，承担多项相关职责
    - 22个方法分为以下职责组:
      * 核心功能方法 (约7个)
      * 辅助工具方法 (约7个)
      * 接口适配方法 (约7个)
    - 符合L4层组件化架构设计原则
    - 基于L3层成功经验的职责分组模式
    """

    """
    形态识别和信号生成混入类
    为技术指标添加统一的形态识别和信号生成功能
    """

    def add_pattern_detection(self, result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        添加形态识别功能

        Args:
            result: 指标计算结果Data_frame
            **kwargs: 额外参数

        Returns:
            添加了形态识别列的Data_frame
        """
        # 初始化形态识别列
        if result.empty:
            return result

        result = result.copy()  # 避免Setting_with_copy_warning
        result.loc[:, "pattern_bullish"] = False
        result.loc[:, "pattern_bearish"] = False
        result.loc[:, "pattern_neutral"] = True

        # 根据指标类型添加特定形态识别
        indicator_name = getattr(self, "name", self.__class__.__name__)

        if "RSI" in indicator_name.upper():
            result = self._add_rsi_patterns(result)
        elif "MACD" in indicator_name.upper():
            result = self._add_macd_patterns(result)
        elif "KDJ" in indicator_name.upper():
            result = self._add_kdj_patterns(result)
        elif "BOLL" in indicator_name.upper():
            result = self._add_boll_patterns(result)
        elif any(ma in indicator_name.upper() for ma in ["MA", "EMA", "WMA"]):
            result = self._add_ma_patterns(result)
        elif "CCI" in indicator_name.upper():
            result = self._add_cci_patterns(result)
        elif "DMI" in indicator_name.upper():
            result = self._add_dmi_patterns(result)
        elif "STOCHRSI" in indicator_name.upper():
            result = self._add_stochrsi_patterns(result)
        elif "WR" in indicator_name.upper():
            result = self._add_wr_patterns(result)
        else:
            # 通用形态识别
            result = self._add_generic_patterns(result)

        return result

    def add_signal_generation(self, result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        添加信号生成功能

        Args:
            result: 指标计算结果Data_frame
            **kwargs: 额外参数

        Returns:
            添加了信号列的Data_frame
        """
        # 初始化信号列
        if result.empty:
            return result

        result = result.copy()  # 避免Setting_with_copy_warning
        # 使用assign方法避免FutureWarning
        result = result.assign(buy_signal=False, sell_signal=False, hold_signal=True)

        # 根据指标类型添加特定信号生成
        indicator_name = getattr(self, "name", self.__class__.__name__)

        if "RSI" in indicator_name.upper():
            result = self._add_rsi_signals(result)
        elif "MACD" in indicator_name.upper():
            result = self._add_macd_signals(result)
        elif "KDJ" in indicator_name.upper():
            result = self._add_kdj_signals(result)
        elif "BOLL" in indicator_name.upper():
            result = self._add_boll_signals(result)
        elif any(ma in indicator_name.upper() for ma in ["MA", "EMA", "WMA"]):
            result = self._add_ma_signals(result)
        elif "CCI" in indicator_name.upper():
            result = self._add_cci_signals(result)
        elif "DMI" in indicator_name.upper():
            result = self._add_dmi_signals(result)
        elif "STOCHRSI" in indicator_name.upper():
            result = self._add_stochrsi_signals(result)
        elif "WR" in indicator_name.upper():
            result = self._add_wr_signals(result)
        else:
            # 通用信号生成
            result = self._add_generic_signals(result)

        return result

    def _add_rsi_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """RSI形态识别"""
        if "rsi" in result.columns:
            rsi_col = "rsi"
        elif "RSI" in result.columns:
            rsi_col = "RSI"
        else:
            # 查找包含RSI的列
            rsi_cols = [col for col in result.columns if "rsi" in col.lower()]
            if rsi_cols:
                rsi_col = rsi_cols[0]
            else:
                return result

        # RSI超买超卖形态
        result["pattern_bullish"] = result[rsi_col] < 30  # 超卖，看涨形态  # TODO: 将魔法数字提取到配置中
        result["pattern_bearish"] = result[rsi_col] > 70  # 超买，看跌形态  # TODO: 将魔法数字提取到配置中
        result["pattern_neutral"] = (result[rsi_col] >= 30) & (
            result[rsi_col] <= 70
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        return result

    def _add_rsi_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """RSI信号生成"""
        if "rsi" in result.columns:
            rsi_col = "rsi"
        elif "RSI" in result.columns:
            rsi_col = "RSI"
        else:
            rsi_cols = [col for col in result.columns if "rsi" in col.lower()]
            if rsi_cols:
                rsi_col = rsi_cols[0]
            else:
                return result

        # RSI买卖信号
        result["buy_signal"] = (result[rsi_col] < 30) & (
            result[rsi_col].shift(1) >= 30
        )  # 从超卖区域向上突破  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        result["sell_signal"] = (result[rsi_col] > 70) & (
            result[rsi_col].shift(1) <= 70
        )  # 从超买区域向下突破  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        result["hold_signal"] = ~(result["buy_signal"] | result["sell_signal"])

        return result

    def _add_macd_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """MACD形态识别"""
        # 查找MACD相关列
        dif_col = None
        dea_col = None
        histogram_col = None

        for col in result.columns:
            if "dif" in col.lower() or "macd" in col.lower():
                dif_col = col
            elif "dea" in col.lower() or "signal" in col.lower():
                dea_col = col
            elif "histogram" in col.lower() or "macd_histogram" in col.lower():
                histogram_col = col

        if dif_col and dea_col:
            # MACD金叉死叉形态
            result["pattern_bullish"] = (result[dif_col] > result[dea_col]) & (
                result[dif_col].shift(1) <= result[dea_col].shift(1)
            )  # 金叉
            result["pattern_bearish"] = (result[dif_col] < result[dea_col]) & (
                result[dif_col].shift(1) >= result[dea_col].shift(1)
            )  # 死叉
            result["pattern_neutral"] = ~(result["pattern_bullish"] | result["pattern_bearish"])

        return result

    def _add_macd_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """MACD信号生成"""
        # 查找MACD相关列
        dif_col = None
        dea_col = None

        for col in result.columns:
            if "dif" in col.lower() or "macd" in col.lower():
                dif_col = col
            elif "dea" in col.lower() or "signal" in col.lower():
                dea_col = col

        if dif_col and dea_col:
            # MACD买卖信号
            result["buy_signal"] = (result[dif_col] > result[dea_col]) & (
                result[dif_col].shift(1) <= result[dea_col].shift(1)
            )  # 金叉买入
            result["sell_signal"] = (result[dif_col] < result[dea_col]) & (
                result[dif_col].shift(1) >= result[dea_col].shift(1)
            )  # 死叉卖出
            result["hold_signal"] = ~(result["buy_signal"] | result["sell_signal"])

        return result

    def _add_kdj_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """KDJ形态识别"""
        # 查找KDJ相关列
        k_col = None
        d_col = None
        j_col = None

        for col in result.columns:
            if col.upper() == "K" or "k_value" in col.lower():
                k_col = col
            elif col.upper() == "D" or "d_value" in col.lower():
                d_col = col
            elif col.upper() == "J" or "j_value" in col.lower():
                j_col = col

        if k_col and d_col:
            # KDJ金叉死叉形态
            result["pattern_bullish"] = (result[k_col] > result[d_col]) & (
                result[k_col].shift(1) <= result[d_col].shift(1)
            )  # 金叉
            result["pattern_bearish"] = (result[k_col] < result[d_col]) & (
                result[k_col].shift(1) >= result[d_col].shift(1)
            )  # 死叉
            result["pattern_neutral"] = ~(result["pattern_bullish"] | result["pattern_bearish"])

        return result

    def _add_kdj_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """KDJ信号生成"""
        # 查找KDJ相关列
        k_col = None
        d_col = None

        for col in result.columns:
            if col.upper() == "K" or "k_value" in col.lower():
                k_col = col
            elif col.upper() == "D" or "d_value" in col.lower():
                d_col = col

        if k_col and d_col:
            # KDJ买卖信号
            result["buy_signal"] = (
                (result[k_col] > result[d_col])
                & (result[k_col].shift(1) <= result[d_col].shift(1))
                & (result[k_col] < 20)
            )  # 低位金叉买入  # TODO: 将魔法数字提取到配置中
            result["sell_signal"] = (
                (result[k_col] < result[d_col])
                & (result[k_col].shift(1) >= result[d_col].shift(1))
                & (result[k_col] > 80)
            )  # 高位死叉卖出  # TODO: 将魔法数字提取到配置中
            result["hold_signal"] = ~(result["buy_signal"] | result["sell_signal"])

        return result

    def _add_boll_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """BOLL形态识别"""
        # 查找BOLL相关列
        upper_col = None
        middle_col = None
        lower_col = None

        for col in result.columns:
            if "upper" in col.lower() or "boll_up" in col.lower():
                upper_col = col
            elif "middle" in col.lower() or "boll_mid" in col.lower() or "ma" in col.lower():
                middle_col = col
            elif "lower" in col.lower() or "boll_down" in col.lower():
                lower_col = col

        if upper_col and lower_col:
            # 需要价格数据来判断突破
            if hasattr(self, "data") and "close" in self.data.columns:
                close_price = self.data["close"]
                # BOLL突破形态
                result["pattern_bullish"] = close_price > result[upper_col]  # 突破上轨
                result["pattern_bearish"] = close_price < result[lower_col]  # 跌破下轨
                result["pattern_neutral"] = (close_price >= result[lower_col]) & (close_price <= result[upper_col])

        return result

    def _add_boll_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """BOLL信号生成"""
        # 查找BOLL相关列
        upper_col = None
        lower_col = None

        for col in result.columns:
            if "upper" in col.lower() or "boll_up" in col.lower():
                upper_col = col
            elif "lower" in col.lower() or "boll_down" in col.lower():
                lower_col = col

        if upper_col and lower_col:
            # 需要价格数据来判断信号
            if hasattr(self, "data") and "close" in self.data.columns:
                close_price = self.data["close"]
                # BOLL买卖信号
                result["buy_signal"] = (close_price < result[lower_col]) & (
                    close_price.shift(1) >= result[lower_col].shift(1)
                )  # 跌破下轨后回升
                result["sell_signal"] = (close_price > result[upper_col]) & (
                    close_price.shift(1) <= result[upper_col].shift(1)
                )  # 突破上轨后回落
                result["hold_signal"] = ~(result["buy_signal"] | result["sell_signal"])

        return result

    def _add_generic_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """通用形态识别"""
        # 基于数值变化的通用形态识别
        if len(result.columns) > 0:
            main_col = result.columns[0]  # 使用第一列作为主要指标
            if result[main_col].dtype in ["float64", "int64"]:
                # 基于趋势的形态识别
                result.loc[:, "pattern_bullish"] = result[main_col] > result[main_col].shift(1)  # 上升趋势
                result.loc[:, "pattern_bearish"] = result[main_col] < result[main_col].shift(1)  # 下降趋势
                result.loc[:, "pattern_neutral"] = result[main_col] == result[main_col].shift(1)  # 横盘

        return result

    def _add_generic_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """通用信号生成"""
        # 基于数值变化的通用信号生成
        if len(result.columns) > 0:
            main_col = result.columns[0]  # 使用第一列作为主要指标
            if result[main_col].dtype in ["float64", "int64"]:
                # 基于趋势变化的信号生成
                trend_up = result[main_col] > result[main_col].shift(1)
                trend_down = result[main_col] < result[main_col].shift(1)
                prev_trend_up = result[main_col].shift(1) > result[main_col].shift(2)
                prev_trend_down = result[main_col].shift(1) < result[main_col].shift(2)

                result.loc[:, "buy_signal"] = trend_up & ~prev_trend_up  # 趋势转为上升
                result.loc[:, "sell_signal"] = trend_down & ~prev_trend_down  # 趋势转为下降
                result.loc[:, "hold_signal"] = ~(result["buy_signal"] | result["sell_signal"])

        return result

    # 其他指标的形态识别和信号生成方法...
    def _add_ma_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """移动平均线形态识别"""
        return self._add_generic_patterns(result)

    def _add_ma_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """移动平均线信号生成"""
        return self._add_generic_signals(result)

    def _add_cci_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """CCI形态识别"""
        return self._add_generic_patterns(result)

    def _add_cci_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """CCI信号生成"""
        return self._add_generic_signals(result)

    def _add_dmi_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """DMI形态识别"""
        return self._add_generic_patterns(result)

    def _add_dmi_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """DMI信号生成"""
        return self._add_generic_signals(result)

    def _add_stochrsi_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """StochRSI形态识别"""
        return self._add_rsi_patterns(result)  # 使用RSI类似的逻辑

    def _add_stochrsi_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """StochRSI信号生成"""
        return self._add_rsi_signals(result)  # 使用RSI类似的逻辑

    def _add_wr_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """WR形态识别"""
        return self._add_generic_patterns(result)

    def _add_wr_signals(self, result: pd.DataFrame) -> pd.DataFrame:
        """WR信号生成"""
        return self._add_generic_signals(result)
