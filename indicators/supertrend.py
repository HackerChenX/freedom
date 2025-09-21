#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
超级趋势(SuperTrend)指标

SuperTrend是一个基于ATR的趋势跟踪指标,它在价格图表上显示动态的支撑和阻力线.
该指标结合了平均真实波幅(ATR)和价格的中位数来确定趋势方向.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from utils.container import container
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin

logger = get_logger(__name__)


class SuperTrend(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    超级趋势(SuperTrend)指标

    分类:趋势指标
    描述:基于ATR的动态支撑阻力线,用于趋势跟踪

    计算公式:
    1. HL2 = (High + Low) / 2
    2. ATR = Average True Range
    3. Upper Band = HL2 + (multiplier * ATR)  # TODO: 将魔法数字提取到配置中
    4. Lower Band = HL2 - (multiplier * ATR)  # TODO: 将魔法数字提取到配置中
    5. SuperTrend = 根据价格与带线的关系确定  # TODO: 将魔法数字提取到配置中

    信号解释:
    - 价格在SuperTrend线上方:上升趋势
    - 价格在SuperTrend线下方:下降趋势
    - SuperTrend线颜色变化:趋势转换信号
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0, **kwargs):  # TODO: 将魔法数字提取到配置中
        """
        初始化SuperTrend指标

        Args:
            period: ATR计算周期,默认10
            multiplier: ATR乘数,默认3.0
            **kwargs: 其他参数
        """
        super().__init__(name="SUPERTREND", **kwargs)
        
        # 依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
        
        self.period = period
        self.multiplier = multiplier
        self.REQUIRED_COLUMNS = ["high", "low", "close"]
        self.description = "超级趋势指标，基于ATR的趋势跟踪指标，显示动态的支撑和阻力线"

    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 10, "multiplier": 3.0}  # TODO: 将魔法数字提取到配置中

    def set_parameters(self, **kwargs):
        """设置参数"""
        self.period = kwargs.get("period", self.period)
        self.multiplier = kwargs.get("multiplier", self.multiplier)

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据的有效性"""
        if data is None or len(data) == 0:
            return False

        # 检查必需列
        for col in self.REQUIRED_COLUMNS:
            if col not in data.columns:
                logger.error(f"数据缺少必需列: {col}")
                return False

        return True

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算SuperTrend指标

        Args:
            data: 包含high,low,close列的DataFrame
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含SuperTrend指标的DataFrame
        """
        try:
            # 验证数据
            if not self._validate_data(data):
                return pd.DataFrame()

            df = data.copy()

            # 计算HL2(高低价中位数)
            hl2 = (df["high"] + df["low"]) / 2

            # 计算ATR
            atr = self._calculate_atr(df)

            # 计算基础上下轨
            upper_band = hl2 + (self.multiplier * atr)
            lower_band = hl2 - (self.multiplier * atr)

            # 计算最终上下轨(考虑前一期的值)
            final_upper_band = pd.Series(index=df.index, dtype=float)
            final_lower_band = pd.Series(index=df.index, dtype=float)

            for i in range(len(df)):
                if i == 0:
                    final_upper_band.iloc[i] = upper_band.iloc[i]
                    final_lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    # 上轨:如果当前上轨小于前一期上轨或前一期收盘价大于前一期上轨,则使用当前上轨
                    if (
                        upper_band.iloc[i] < final_upper_band.iloc[i - 1]
                        or df["close"].iloc[i - 1] > final_upper_band.iloc[i - 1]
                    ):
                        final_upper_band.iloc[i] = upper_band.iloc[i]
                    else:
                        final_upper_band.iloc[i] = final_upper_band.iloc[i - 1]

                    # 下轨:如果当前下轨大于前一期下轨或前一期收盘价小于前一期下轨,则使用当前下轨
                    if (
                        lower_band.iloc[i] > final_lower_band.iloc[i - 1]
                        or df["close"].iloc[i - 1] < final_lower_band.iloc[i - 1]
                    ):
                        final_lower_band.iloc[i] = lower_band.iloc[i]
                    else:
                        final_lower_band.iloc[i] = final_lower_band.iloc[i - 1]

            # 计算SuperTrend线
            supertrend = pd.Series(index=df.index, dtype=float)
            trend_direction = pd.Series(index=df.index, dtype=int)  # 1为上升趋势,-1为下降趋势

            for i in range(len(df)):
                if i == 0:
                    if df["close"].iloc[i] <= final_lower_band.iloc[i]:
                        supertrend.iloc[i] = final_upper_band.iloc[i]
                        trend_direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = final_lower_band.iloc[i]
                        trend_direction.iloc[i] = 1
                else:
                    if trend_direction.iloc[i - 1] == 1:
                        if df["close"].iloc[i] <= final_lower_band.iloc[i]:
                            supertrend.iloc[i] = final_upper_band.iloc[i]
                            trend_direction.iloc[i] = -1
                        else:
                            supertrend.iloc[i] = final_lower_band.iloc[i]
                            trend_direction.iloc[i] = 1
                    else:  # trend_direction.iloc[i-1] == -1
                        if df["close"].iloc[i] >= final_upper_band.iloc[i]:
                            supertrend.iloc[i] = final_lower_band.iloc[i]
                            trend_direction.iloc[i] = 1
                        else:
                            supertrend.iloc[i] = final_upper_band.iloc[i]
                            trend_direction.iloc[i] = -1

            # 添加到结果DataFrame
            df["hl2"] = hl2
            df["atr"] = atr
            df["upper_band"] = upper_band
            df["lower_band"] = lower_band
            df["final_upper_band"] = final_upper_band
            df["final_lower_band"] = final_lower_band
            df["supertrend"] = supertrend
            df["trend_direction"] = trend_direction

            # 计算信号
            df["st_signal"] = self._generate_signals(df)

            # 保存结果到_result属性
            self._result = df

            return df

        except Exception as e:
            logger.error(f"SuperTrend指标计算失败: {e}")
            return pd.DataFrame()

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        计算平均真实波幅(ATR)

        Args:
            df: 包含OHLC数据的DataFrame

        Returns:
            pd.Series: ATR序列
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算ATR(使用简单移动平均)
        atr = true_range.rolling(window=self.period).mean()

        return atr

    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Args:
            df: 包含SuperTrend数据的DataFrame

        Returns:
            pd.Series: 交易信号
        """
        signals = pd.Series(0, index=df.index)
        trend_direction = df["trend_direction"]

        # 趋势转换信号
        trend_change = trend_direction.diff()

        # 买入信号:趋势从下降转为上升
        buy_signal = trend_change == 2  # 从-1变为1
        signals[buy_signal] = 1

        # 卖出信号:趋势从上升转为下降
        sell_signal = trend_change == -2  # 从1变为-1
        signals[sell_signal] = -1

        return signals

    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=None)
    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于SuperTrend指标数值生成最新的交易信号
        
        SuperTrend交易信号逻辑：
        - 价格突破SuperTrend上轨：买入信号（趋势转为上升）
        - 价格跌破SuperTrend下轨：卖出信号（趋势转为下降）
        - 价格在SuperTrend上方：持续买入确认
        - 价格在SuperTrend下方：持续卖出确认
        - 信号强度基于价格与SuperTrend线的距离
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 1. 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 2. 确保已计算指标
            if not self.has_result():
                self.calculate(data, **kwargs)

            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("SuperTrend计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取SuperTrend相关值
            if len(self._result) < 2:
                return self._get_default_signal("SuperTrend数据不足")
                
            # 检查必要的列是否存在
            required_columns = ['supertrend', 'trend_direction', 'st_signal']
            if not all(col in self._result.columns for col in required_columns):
                return self._get_default_signal("SuperTrend结果列不完整")
                
            latest_supertrend = self._result['supertrend'].iloc[-1]
            latest_trend = self._result['trend_direction'].iloc[-1]
            latest_signal = self._result['st_signal'].iloc[-1]
            prev_trend = self._result['trend_direction'].iloc[-2]
            
            # 获取额外的SuperTrend数据
            latest_upper_band = self._result.get('final_upper_band', pd.Series([latest_supertrend])).iloc[-1] if 'final_upper_band' in self._result.columns else latest_supertrend
            latest_lower_band = self._result.get('final_lower_band', pd.Series([latest_supertrend])).iloc[-1] if 'final_lower_band' in self._result.columns else latest_supertrend
            latest_atr = self._result.get('atr', pd.Series([1])).iloc[-1] if 'atr' in self._result.columns else 1
            
            # 5. SuperTrend信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # 计算价格与SuperTrend线的距离作为信号强度基础
            if latest_supertrend != 0:
                distance_ratio = abs(latest_close - latest_supertrend) / latest_supertrend
                base_strength = min(distance_ratio * 8, 1.0)  # 标准化强度
            else:
                base_strength = 0.0
            
            # 趋势转换信号（最高优先级）
            if latest_signal == 1 and latest_trend == 1:
                # SuperTrend买入信号 - 趋势转为上升
                signal_type = "buy"
                strength = max(0.85, base_strength)
                confidence = 0.9
                reason = f"SuperTrend趋势转为上升({latest_close:.2f}>{latest_supertrend:.2f})，强烈买入信号"
                
            elif latest_signal == -1 and latest_trend == -1:
                # SuperTrend卖出信号 - 趋势转为下降
                signal_type = "sell"
                strength = max(0.85, base_strength)
                confidence = 0.9
                reason = f"SuperTrend趋势转为下降({latest_close:.2f}<{latest_supertrend:.2f})，强烈卖出信号"
            
            # 趋势持续信号
            elif latest_trend == 1 and latest_close > latest_supertrend:
                # 价格在SuperTrend上方 - 持续买入
                signal_type = "buy"
                strength = max(0.7, base_strength * 0.8)
                confidence = 0.8
                reason = f"价格持续在SuperTrend上方({latest_close:.2f}>{latest_supertrend:.2f})，持续买入"
                
            elif latest_trend == -1 and latest_close < latest_supertrend:
                # 价格在SuperTrend下方 - 持续卖出
                signal_type = "sell"
                strength = max(0.7, base_strength * 0.8)
                confidence = 0.8
                reason = f"价格持续在SuperTrend下方({latest_close:.2f}<{latest_supertrend:.2f})，持续卖出"
            
            # 弱信号：接近但未突破
            elif latest_trend == 1 and latest_close > latest_lower_band:
                # 价格接近但在支撑线上方
                signal_type = "buy"
                strength = base_strength * 0.6
                confidence = 0.65
                reason = f"价格接近SuperTrend支撑线({latest_close:.2f})，弱买入信号"
                
            elif latest_trend == -1 and latest_close < latest_upper_band:
                # 价格接近但在阻力线下方
                signal_type = "sell"
                strength = base_strength * 0.6
                confidence = 0.65
                reason = f"价格接近SuperTrend阻力线({latest_close:.2f})，弱卖出信号"
            
            # 计算SuperTrend特有的元数据
            trend_change = latest_trend != prev_trend
            trend_stability = abs(latest_trend)  # 1或-1表示明确趋势
            
            metadata = {
                'supertrend_value': latest_supertrend,
                'trend_direction': latest_trend,
                'signal_raw': latest_signal,
                'upper_band': latest_upper_band,
                'lower_band': latest_lower_band,
                'atr': latest_atr,
                'price_distance': latest_close - latest_supertrend,
                'distance_ratio': abs(latest_close - latest_supertrend) / latest_supertrend if latest_supertrend != 0 else 0,
                'trend_change': trend_change,
                'trend_stability': trend_stability,
                'trend_description': "上升趋势" if latest_trend == 1 else "下降趋势" if latest_trend == -1 else "无趋势",
                'signal_quality': 'strong' if abs(latest_signal) == 1 else 'weak',
                'multiplier': self.multiplier,
                'period': self.period
            }
            
            # 6. 标准化输出
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'latest_close': latest_close,
                    **metadata
                }
            }

        except Exception as e:
            logger.warning(f"SuperTrend信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if data is None or data.empty:
            return False
            
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # SuperTrend需要足够的数据用于ATR计算
        min_periods = self.period + 10
        if len(data) < min_periods:
            return False
            
        return True

    def _get_default_signal(self, reason: str = "数据不足") -> Dict[str, Any]:
        """
        生成默认信号（持有信号）
        
        Args:
            reason: 生成默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号
        """
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.0,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {}
        }

    def has_result(self) -> bool:
        """
        检查是否已有计算结果
        
        Returns:
            bool: 是否已有计算结果
        """
        return (self._result is not None and 
                hasattr(self._result, 'empty') and 
                not self._result.empty and
                'supertrend' in self._result.columns and
                'trend_direction' in self._result.columns)

    def get_pattern_info(self) -> Dict[str, Any]:
        """获取指标模式信息"""
        return {
            "name": "SUPERTREND",
            "description": "超级趋势指标",
            "type": "trend",
            "parameters": {"period": self.period, "multiplier": self.multiplier},
        }

    # 实现BaseIndicator的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """核心计算逻辑"""
        return self.calculate(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if "trend_direction" not in data.columns:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 基于趋势方向计算评分
        trend_direction = data["trend_direction"]
        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
        score[trend_direction == 1] = 75.0  # 上升趋势  # TODO: 将魔法数字提取到配置中
        score[trend_direction == -1] = 25.0  # 下降趋势  # TODO: 将魔法数字提取到配置中

        return score

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取技术形态"""
        patterns = pd.DataFrame(index=data.index)

        if "trend_direction" in data.columns:
            trend = data["trend_direction"]
            # 趋势状态形态
            patterns["ST_上升趋势"] = trend == 1
            patterns["ST_下降趋势"] = trend == -1
            # 趋势转换形态
            trend_change = trend.diff()
            patterns["ST_转为上升"] = trend_change == 2  # 从-1变为1
            patterns["ST_转为下降"] = trend_change == -2  # 从1变为-1
            # 趋势持续形态
            patterns["ST_趋势延续"] = (trend == trend.shift(1)) & (trend != 0)

        return patterns

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]
    ) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.0

        # 基于趋势稳定性计算置信度
        if len(score) >= 5:  # TODO: 将魔法数字提取到配置中
            trend_stability = score.rolling(5).std().iloc[-1] < 10  # 趋势稳定  # TODO: 将魔法数字提取到配置中
            confidence = 0.8 if trend_stability else 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        else:
            confidence = 0.5  # TODO: 将魔法数字提取到配置中

        pattern_strength = min(len(patterns) * 0.1, 0.3)  # TODO: 将魔法数字提取到配置中
        return min(confidence + pattern_strength, 1.0)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置参数"""
        self.set_parameters(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """最小周期数"""
        return self.period + 10
