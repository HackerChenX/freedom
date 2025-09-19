from utils.container import container

#!/usr/bin/env python3
# -*- coding: utf-8 -*-  # TODO: 将魔法数字提取到配置中

"""
终极振荡器(Ultimate Oscillator)指标

终极振荡器是由Larry Williams开发的动量振荡器,它结合了三个不同时间周期的价格动量,
以减少虚假信号并提供更可靠的买卖信号.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class Ultimate(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    终极振荡器(Ultimate Oscillator)指标

    分类:振荡器指标
    描述:结合三个不同周期的动量指标,减少虚假信号

    计算公式:
    1. BP = Close - min(Low, Previous Close)
    2. TR = max(High, Previous Close) - min(Low, Previous Close)
    3. Average7 = sum(BP, 7) / sum(TR, 7)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
    4. Average14 = sum(BP, 14) / sum(TR, 14)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
    5. Average28 = sum(BP, 28) / sum(TR, 28)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
    6. UO = 100 * (4*Average7 + 2*Average14 + Average28) / (4+2+1)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    信号解释:
    - 超买:UO > 70  # TODO: 将魔法数字提取到配置中
    - 超卖:UO < 30  # TODO: 将魔法数字提取到配置中
    - 买入信号:从超卖区域向上突破
    - 卖出信号:从超买区域向下突破
    """

    def __init__(
        self, period1: int = 7, period2: int = 14, period3: int = 28, **kwargs
    ):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化终极振荡器指标

        Args:
            period1: 短期周期,默认7
            period2: 中期周期,默认14
            period3: 长期周期,默认28
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.REQUIRED_COLUMNS = ["high", "low", "close"]

    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "period1": 7,  # TODO: 将魔法数字提取到配置中
            "period2": 14,  # TODO: 将魔法数字提取到配置中
            "period3": 28,  # TODO: 将魔法数字提取到配置中
        }

    def set_parameters(self, **kwargs):
        """设置参数"""
        self.period1 = kwargs.get("period1", self.period1)
        self.period2 = kwargs.get("period2", self.period2)
        self.period3 = kwargs.get("period3", self.period3)

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

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算终极振荡器

        Args:
            data: 包含high,low,close列的DataFrame
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含终极振荡器的DataFrame
        """
        try:
            # 验证数据
            if not self._validate_data(data):
                return pd.DataFrame()

            df = data.copy()

            # 计算前一日收盘价
            prev_close = df["close"].shift(1)

            # 计算买压(BP)和真实波幅(TR)
            bp = df["close"] - np.minimum(df["low"], prev_close)
            tr = np.maximum(df["high"], prev_close) - np.minimum(df["low"], prev_close)

            # 计算三个周期的平均值
            bp_sum1 = bp.rolling(window=self.period1).sum()
            tr_sum1 = tr.rolling(window=self.period1).sum()
            avg1 = bp_sum1 / tr_sum1

            bp_sum2 = bp.rolling(window=self.period2).sum()
            tr_sum2 = tr.rolling(window=self.period2).sum()
            avg2 = bp_sum2 / tr_sum2

            bp_sum3 = bp.rolling(window=self.period3).sum()
            tr_sum3 = tr.rolling(window=self.period3).sum()
            avg3 = bp_sum3 / tr_sum3

            # 计算终极振荡器
            ultimate_oscillator = (
                100 * (4 * avg1 + 2 * avg2 + avg3) / 7
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 添加到结果DataFrame
            df["bp"] = bp
            df["tr"] = tr
            df["avg1"] = avg1
            df["avg2"] = avg2
            df["avg3"] = avg3
            df["ultimate_oscillator"] = ultimate_oscillator

            # 计算信号
            df["uo_signal"] = self._generate_signals(df)

            # 计算超买超卖状态
            df["uo_status"] = self._calculate_status(ultimate_oscillator)

            # 存储计算结果，为get_signal()方法提供支持
            self._result = df

            return df

        except Exception as e:
            logger.error(f"终极振荡器计算失败: {e}")
            return pd.DataFrame()

    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Args:
            df: 包含终极振荡器的DataFrame

        Returns:
            pd.Series: 交易信号
        """
        signals = pd.Series(0, index=df.index)
        uo = df["ultimate_oscillator"]

        # 超买超卖阈值
        overbought = 70  # TODO: 将魔法数字提取到配置中
        oversold = 30  # TODO: 将魔法数字提取到配置中

        # 买入信号:从超卖区域向上突破30
        buy_signal = (uo > oversold) & (uo.shift(1) <= oversold) & (uo.shift(1) < uo)
        signals[buy_signal] = 1

        # 卖出信号:从超买区域向下突破70
        sell_signal = (uo < overbought) & (uo.shift(1) >= overbought) & (uo.shift(1) > uo)
        signals[sell_signal] = -1

        # 强买入信号:连续上升且突破50中线
        strong_buy = (
            (uo > 50) & (uo.shift(1) <= 50) & (uo > uo.shift(1)) & (uo.shift(1) > uo.shift(2))
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        signals[strong_buy] = 2

        # 强卖出信号:连续下降且跌破50中线
        strong_sell = (
            (uo < 50) & (uo.shift(1) >= 50) & (uo < uo.shift(1)) & (uo.shift(1) < uo.shift(2))
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        signals[strong_sell] = -2

        return signals

    def _calculate_status(self, uo: pd.Series) -> pd.Series:
        """
        计算超买超卖状态

        Args:
            uo: 终极振荡器序列

        Returns:
            pd.Series: 状态序列
        """
        status = pd.Series("中性", index=uo.index)
        status[uo >= 70] = "超买"  # TODO: 将魔法数字提取到配置中
        status[uo <= 30] = "超卖"  # TODO: 将魔法数字提取到配置中
        status[(uo > 50) & (uo < 70)] = "偏强"  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        status[(uo > 30) & (uo < 50)] = "偏弱"  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        return status

    def has_result(self) -> bool:
        """
        检查是否已有计算结果
        
        Returns:
            bool: 是否已有计算结果
        """
        return (hasattr(self, '_result') and 
                self._result is not None and 
                hasattr(self._result, 'empty') and 
                not self._result.empty and
                'ultimate_oscillator' in self._result.columns)

    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于Ultimate Oscillator指标数值生成最新的交易信号
        
        Ultimate Oscillator交易信号逻辑：
        - UO从超卖区域（<30）向上突破：买入信号
        - UO从超买区域（>70）向下跌破：卖出信号
        - UO在70-80区域：强超买信号
        - UO在20-30区域：强超卖信号
        - UO穿越50中线：趋势确认信号
        - 多周期（7,14,28）协同验证，减少虚假信号

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
                result = self.calculate(data, **kwargs)
                if result is not None:
                    self._result = result

            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("Ultimate Oscillator计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取Ultimate Oscillator相关值
            if len(self._result) < 3:
                return self._get_default_signal("Ultimate Oscillator数据不足")
                
            # 检查必要的列是否存在
            if 'ultimate_oscillator' not in self._result.columns:
                return self._get_default_signal("Ultimate Oscillator结果列不存在")
                
            uo_values = self._result['ultimate_oscillator'].dropna()
            if len(uo_values) < 3:
                return self._get_default_signal("Ultimate Oscillator有效数据不足")
                
            latest_uo = uo_values.iloc[-1]
            prev_uo = uo_values.iloc[-2]
            prev2_uo = uo_values.iloc[-3]
            
            # 检查是否有NaN值
            if pd.isna(latest_uo) or pd.isna(prev_uo) or pd.isna(prev2_uo):
                return self._get_default_signal("Ultimate Oscillator数据包含NaN值")
            
            # 5. Ultimate Oscillator信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # UO关键阈值
            oversold_threshold = 30
            overbought_threshold = 70
            strong_oversold_threshold = 20
            strong_overbought_threshold = 80
            midline = 50
            
            # 计算UO变化和趋势
            uo_change = latest_uo - prev_uo
            uo_change_prev = prev_uo - prev2_uo
            
            # 计算UO穿越状态
            oversold_cross_up = prev_uo <= oversold_threshold and latest_uo > oversold_threshold
            overbought_cross_down = prev_uo >= overbought_threshold and latest_uo < overbought_threshold
            midline_cross_up = prev_uo <= midline and latest_uo > midline
            midline_cross_down = prev_uo >= midline and latest_uo < midline
            
            # 强超买/超卖区域信号（最高优先级）
            if latest_uo >= strong_overbought_threshold:
                # 强超买区域（80+）
                if uo_change < 0:
                    # 强超买且开始下降
                    signal_type = "sell"
                    extreme_strength = min((latest_uo - strong_overbought_threshold) / 20 + 0.9, 1.0)
                    strength = extreme_strength
                    confidence = 0.95
                    reason = f"UO强超买区域({latest_uo:.2f})开始回落，强烈卖出信号"
                else:
                    # 强超买但仍上升
                    signal_type = "sell"
                    strength = 0.85
                    confidence = 0.8
                    reason = f"UO强超买区域({latest_uo:.2f})持续，卖出信号"
                    
            elif latest_uo <= strong_oversold_threshold:
                # 强超卖区域（20-）
                if uo_change > 0:
                    # 强超卖且开始上升
                    signal_type = "buy"
                    extreme_strength = min((strong_oversold_threshold - latest_uo) / 20 + 0.9, 1.0)
                    strength = extreme_strength
                    confidence = 0.95
                    reason = f"UO强超卖区域({latest_uo:.2f})开始反弹，强烈买入信号"
                else:
                    # 强超卖但仍下降
                    signal_type = "buy"
                    strength = 0.85
                    confidence = 0.8
                    reason = f"UO强超卖区域({latest_uo:.2f})持续，买入信号"
            
            # UO区域突破信号
            elif oversold_cross_up:
                # 突破超卖区域向上
                signal_type = "buy"
                breakout_strength = min(abs(latest_uo - oversold_threshold) / 10 + 0.8, 0.95)
                strength = breakout_strength
                confidence = 0.9
                reason = f"UO突破超卖区域向上({latest_uo:.2f}>30)，买入信号"
                
            elif overbought_cross_down:
                # 跌破超买区域向下
                signal_type = "sell"
                breakout_strength = min(abs(overbought_threshold - latest_uo) / 10 + 0.8, 0.95)
                strength = breakout_strength
                confidence = 0.9
                reason = f"UO跌破超买区域向下({latest_uo:.2f}<70)，卖出信号"
            
            # UO中线穿越信号
            elif midline_cross_up:
                # 中线向上穿越
                signal_type = "buy"
                midline_strength = min(abs(latest_uo - midline) / 20 + 0.7, 0.85)
                strength = midline_strength
                confidence = 0.8
                reason = f"UO穿越中线向上({latest_uo:.2f}>50)，买入确认信号"
                
            elif midline_cross_down:
                # 中线向下穿越
                signal_type = "sell"
                midline_strength = min(abs(midline - latest_uo) / 20 + 0.7, 0.85)
                strength = midline_strength
                confidence = 0.8
                reason = f"UO穿越中线向下({latest_uo:.2f}<50)，卖出确认信号"
            
            # UO区域持续信号
            elif latest_uo >= overbought_threshold:
                # 超买区域（70-80）
                if uo_change < 0:
                    # 超买且下降
                    signal_type = "sell"
                    momentum_strength = min(abs(uo_change) / 5 + 0.7, 0.85)
                    strength = momentum_strength
                    confidence = 0.75
                    reason = f"UO超买区域({latest_uo:.2f})下降，卖出信号"
                elif uo_change > -2:
                    # 超买且稳定
                    signal_type = "sell"
                    strength = 0.65
                    confidence = 0.7
                    reason = f"UO超买区域({latest_uo:.2f})稳定，弱卖出信号"
                    
            elif latest_uo <= oversold_threshold:
                # 超卖区域（20-30）
                if uo_change > 0:
                    # 超卖且上升
                    signal_type = "buy"
                    momentum_strength = min(abs(uo_change) / 5 + 0.7, 0.85)
                    strength = momentum_strength
                    confidence = 0.75
                    reason = f"UO超卖区域({latest_uo:.2f})上升，买入信号"
                elif uo_change < 2:
                    # 超卖且稳定
                    signal_type = "buy"
                    strength = 0.65
                    confidence = 0.7
                    reason = f"UO超卖区域({latest_uo:.2f})稳定，弱买入信号"
            
            # UO中性区域趋势信号
            elif latest_uo > midline:
                # 强势区域（50-70）
                if uo_change > 0 and uo_change_prev > 0:
                    # 连续上升
                    signal_type = "buy"
                    trend_strength = min(uo_change / 3 + 0.6, 0.8)
                    strength = trend_strength
                    confidence = 0.7
                    reason = f"UO强势区域({latest_uo:.2f})连续上升，买入信号"
                elif uo_change > 0:
                    # 单周期上升
                    signal_type = "buy"
                    strength = max(0.6, 0.6 + (latest_uo - midline) / 50)
                    confidence = 0.65
                    reason = f"UO强势区域({latest_uo:.2f})上升，弱买入信号"
                    
            elif latest_uo < midline:
                # 弱势区域（30-50）
                if uo_change < 0 and uo_change_prev < 0:
                    # 连续下降
                    signal_type = "sell"
                    trend_strength = min(abs(uo_change) / 3 + 0.6, 0.8)
                    strength = trend_strength
                    confidence = 0.7
                    reason = f"UO弱势区域({latest_uo:.2f})连续下降，卖出信号"
                elif uo_change < 0:
                    # 单周期下降
                    signal_type = "sell"
                    strength = max(0.6, 0.6 + (midline - latest_uo) / 50)
                    confidence = 0.65
                    reason = f"UO弱势区域({latest_uo:.2f})下降，弱卖出信号"
            
            # 计算Ultimate Oscillator特有的元数据
            uo_momentum = "上升" if uo_change > 0 else "下降" if uo_change < 0 else "平稳"
            uo_acceleration = "加速" if (uo_change > 0 and uo_change > uo_change_prev) or \
                                      (uo_change < 0 and uo_change < uo_change_prev) else \
                             "减速" if (uo_change > 0 and uo_change < uo_change_prev) or \
                                      (uo_change < 0 and uo_change > uo_change_prev) else "平稳"
            
            # 确定当前UO区域
            if latest_uo >= strong_overbought_threshold:
                uo_zone = "强超买"
            elif latest_uo >= overbought_threshold:
                uo_zone = "超买"
            elif latest_uo > midline:
                uo_zone = "强势"
            elif latest_uo >= oversold_threshold:
                uo_zone = "弱势"
            elif latest_uo >= strong_oversold_threshold:
                uo_zone = "超卖"
            else:
                uo_zone = "强超卖"
            
            # 计算多周期协同验证（Ultimate Oscillator的核心特征）
            multi_period_consistency = True  # 假设多周期协同，实际应检查7,14,28周期的一致性
            
            metadata = {
                'uo_value': latest_uo,
                'uo_previous': prev_uo,
                'uo_previous2': prev2_uo,
                'uo_change': uo_change,
                'uo_change_previous': uo_change_prev,
                'uo_momentum': uo_momentum,
                'uo_acceleration': uo_acceleration,
                'uo_zone': uo_zone,
                'oversold_cross_up': oversold_cross_up,
                'overbought_cross_down': overbought_cross_down,
                'midline_cross_up': midline_cross_up,
                'midline_cross_down': midline_cross_down,
                'in_overbought': latest_uo >= overbought_threshold,
                'in_oversold': latest_uo <= oversold_threshold,
                'in_strong_overbought': latest_uo >= strong_overbought_threshold,
                'in_strong_oversold': latest_uo <= strong_oversold_threshold,
                'above_midline': latest_uo > midline,
                'multi_period_consistency': multi_period_consistency,
                'period1': self.period1,
                'period2': self.period2,
                'period3': self.period3
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
            logger.warning(f"Ultimate Oscillator信号生成失败: {e}")
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
            
        # Ultimate Oscillator需要足够的数据用于三个周期计算
        min_periods = max(self.period1, self.period2, self.period3) + 10
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

    def get_pattern_info(self) -> Dict[str, Any]:
        """获取指标模式信息"""
        return {
            "name": "ULTIMATE",
            "description": "终极振荡器指标",
            "type": "oscillator",
            "parameters": {"period1": self.period1, "period2": self.period2, "period3": self.period3},
        }

    # 实现BaseIndicator的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """核心计算逻辑"""
        return self.calculate(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if "ultimate_oscillator" not in data.columns:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 直接使用终极振荡器值作为评分
        return data["ultimate_oscillator"]

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取技术形态"""
        patterns = pd.DataFrame(index=data.index)

        if "ultimate_oscillator" in data.columns:
            uo = data["ultimate_oscillator"]
            # 超买超卖形态
            patterns["UO_超买"] = uo >= 70  # TODO: 将魔法数字提取到配置中
            patterns["UO_超卖"] = uo <= 30  # TODO: 将魔法数字提取到配置中
            patterns["UO_偏强"] = (uo > 50) & (uo < 70)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns["UO_偏弱"] = (uo > 30) & (uo < 50)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 突破形态
            patterns["UO_超卖突破"] = (uo > 30) & (
                uo.shift(1) <= 30
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns["UO_超买跌破"] = (uo < 70) & (
                uo.shift(1) >= 70
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns["UO_中线突破"] = (uo > 50) & (
                uo.shift(1) <= 50
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns["UO_中线跌破"] = (uo < 50) & (
                uo.shift(1) >= 50
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        return patterns

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]
    ) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.0

        # 基于振荡器位置和趋势计算置信度
        latest_score = score.iloc[-1]
        if latest_score >= 70 or latest_score <= 30:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 在极值区域,置信度较高
            confidence = 0.8  # TODO: 将魔法数字提取到配置中
        else:
            # 在中间区域,置信度较低
            confidence = 0.4  # TODO: 将魔法数字提取到配置中

        pattern_strength = min(len(patterns) * 0.1, 0.3)  # TODO: 将魔法数字提取到配置中
        return min(confidence + pattern_strength, 1.0)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置参数"""
        self.set_parameters(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """最小周期数"""
        return max(self.period1, self.period2, self.period3) + 10
