from utils.container import container

#!/usr/bin/env python
# -*- coding: utf-8 -*-  # TODO: 将魔法数字提取到配置中

import logging
from typing import Dict, Any
from typing import Dict, List

import numpy as np
import pandas as pd

from enums.indicator_types import Trend_type, Cross_type
from enums.indicator_enum import Indicator_enum
from indicators.common import crossover, crossunder
from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
import logging

logger = logging.getLogger(__name__)


class KeltnerChannel(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    肯特纳通道指标 (Keltner Channel)

    肯特纳通道是一种波动通道指标,由中轨(通常为EMA)加减一定倍数的ATR形成上下轨.
    相比于布林带使用标准差,肯特纳通道使用ATR衡量波动性,对价格突破和异常波动的反应更平滑.

    参数:
        period: 中轨移动平均周期,默认为20
        atr_period: ATR计算周期,默认为10
        multiplier: ATR乘数,用于计算通道宽度,默认为2.0
    """

    def __init__(
        self,
        period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,  # TODO: 将魔法数字提取到配置中
        name: str = "KC",
        description: str = "肯特纳通道指标",
        **kwargs
    ):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化KC指标"""
        # 正确调用父类初始化
        super().__init__(**kwargs)
        self.name = name
        self.description = description
        self.indicator_type = Indicator_enum.KC.name
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self._result = None
        self.REQUIRED_COLUMNS = ["high", "low", "close"]

    def set_parameters_Kc_Kc_Kc_kc(self, period: int = None, atr_period: int = None, multiplier: float = None):
        """
        设置指标参数

        Args:
            period: 中轨移动平均周期
            atr_period: ATR计算周期
            multiplier: ATR乘数
        """
        if period is not None:
            self.period = period
        if atr_period is not None:
            self.atr_period = atr_period
        if multiplier is not None:
            self.multiplier = multiplier

    # Ultra Think标准方法实现
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """通用计算接口"""
        return self._calculate_kc(data)

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """基类抽象方法实现"""
        return self._calculate_kc(data)
    
    def has_result(self) -> bool:
        """检查是否已计算结果"""
        return self._result is not None and not self._result.empty
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成KC指标的标准化交易信号
        
        KC (Keltner Channel) 特有信号逻辑:
        1. 通道突破: 价格突破上轨为买入信号，跌破下轨为卖出信号
        2. 通道回归: 价格从极端位置回归中轨
        3. 通道宽度: 宽度变化反映波动性变化
        4. 价格位置: 价格在通道中的相对位置
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化信号格式
            {
                'signal_type': 'buy'/'sell'/'hold',
                'strength': 0.0-1.0,
                'confidence': 0.0-1.0, 
                'timestamp': datetime,
                'price': float,
                'reason': str,
                'metadata': dict
            }
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 确保已计算KC指标
            if not self.has_result():
                self.calculate(data)
                
            if not self.has_result():
                return self._get_default_signal("KC计算结果为空")
                
            # 获取KC相关数据
            kc_result = self._result
            if 'kc_upper' not in kc_result.columns or 'kc_lower' not in kc_result.columns or 'kc_middle' not in kc_result.columns:
                return self._get_default_signal("KC通道数据不完整")
            
            kc_upper = kc_result['kc_upper']
            kc_lower = kc_result['kc_lower'] 
            kc_middle = kc_result['kc_middle']
            
            # 获取最新的有效数据点
            latest_idx = -1
            while latest_idx >= -len(kc_upper) and (pd.isna(kc_upper.iloc[latest_idx]) or pd.isna(kc_lower.iloc[latest_idx])):
                latest_idx -= 1
                
            if latest_idx < -len(kc_upper) or latest_idx < -1:
                return self._get_default_signal("KC数据不足")
                
            latest_upper = kc_upper.iloc[latest_idx]
            latest_lower = kc_lower.iloc[latest_idx]
            latest_middle = kc_middle.iloc[latest_idx]
            prev_upper = kc_upper.iloc[latest_idx - 1] if latest_idx - 1 >= -len(kc_upper) else latest_upper
            prev_lower = kc_lower.iloc[latest_idx - 1] if latest_idx - 1 >= -len(kc_lower) else latest_lower
            
            # 获取当前价格
            current_price = data['close'].iloc[-1] if 'close' in data.columns else 0.0
            prev_price = data['close'].iloc[-2] if len(data) >= 2 and 'close' in data.columns else current_price
            
            # 信号强度和置信度初始化
            base_strength = 0.0
            base_confidence = 0.5
            signal_type = 'hold'
            reason_parts = []
            
            # 1. 通道突破信号分析 (最高优先级)
            if current_price > latest_upper and prev_price <= prev_upper:  # 向上突破上轨
                signal_type = 'buy'
                base_strength = 0.9
                base_confidence = 0.9
                breakthrough_ratio = (current_price - latest_upper) / latest_upper
                reason_parts.append(f"向上突破KC上轨(突破幅度{breakthrough_ratio:.2%}，强烈看涨)")
                
            elif current_price < latest_lower and prev_price >= prev_lower:  # 向下跌破下轨
                signal_type = 'sell'
                base_strength = 0.9
                base_confidence = 0.9
                breakdown_ratio = (latest_lower - current_price) / latest_lower
                reason_parts.append(f"向下跌破KC下轨(跌破幅度{breakdown_ratio:.2%}，强烈看跌)")
                
            # 2. 价格在通道中的位置分析
            elif current_price > latest_upper:  # 价格在上轨之上
                signal_type = 'buy'
                base_strength = 0.7
                base_confidence = 0.8
                distance_ratio = (current_price - latest_upper) / (latest_upper - latest_middle)
                reason_parts.append(f"价格处于KC上轨上方(距离{distance_ratio:.1f}倍通道宽度)")
                
            elif current_price < latest_lower:  # 价格在下轨之下
                signal_type = 'sell'
                base_strength = 0.7
                base_confidence = 0.8
                distance_ratio = (latest_lower - current_price) / (latest_middle - latest_lower)
                reason_parts.append(f"价格处于KC下轨下方(距离{distance_ratio:.1f}倍通道宽度)")
                
            # 3. 通道回归信号
            elif current_price > latest_middle:  # 价格在中轨上方
                # 计算相对位置 (0.5-1.0)
                position_ratio = (current_price - latest_middle) / (latest_upper - latest_middle)
                
                if position_ratio > 0.8:  # 接近上轨
                    signal_type = 'sell'
                    base_strength = 0.6
                    base_confidence = 0.7
                    reason_parts.append(f"价格接近KC上轨({position_ratio:.1%}位置，可能回调)")
                elif position_ratio > 0.5:  # 中上区域
                    signal_type = 'buy'
                    base_strength = 0.4
                    base_confidence = 0.6
                    reason_parts.append(f"价格位于KC中上区域({position_ratio:.1%}位置)")
                    
            elif current_price < latest_middle:  # 价格在中轨下方
                # 计算相对位置 (0.0-0.5)
                position_ratio = (current_price - latest_lower) / (latest_middle - latest_lower)
                
                if position_ratio < 0.2:  # 接近下轨
                    signal_type = 'buy'
                    base_strength = 0.6
                    base_confidence = 0.7
                    reason_parts.append(f"价格接近KC下轨({position_ratio:.1%}位置，可能反弹)")
                elif position_ratio < 0.5:  # 中下区域
                    signal_type = 'sell'
                    base_strength = 0.4
                    base_confidence = 0.6
                    reason_parts.append(f"价格位于KC中下区域({position_ratio:.1%}位置)")
                    
            # 4. 通道宽度分析
            channel_width = latest_upper - latest_lower
            prev_channel_width = prev_upper - prev_lower if not pd.isna(prev_upper) and not pd.isna(prev_lower) else channel_width
            width_change_ratio = (channel_width - prev_channel_width) / prev_channel_width if prev_channel_width > 0 else 0
            
            # 5. 信号强度调整
            strength_multiplier = 1.0
            confidence_adjustment = 0.0
            
            # 通道宽度变化调整
            if width_change_ratio > 0.1:  # 通道快速扩张
                strength_multiplier *= 1.3
                confidence_adjustment += 0.15
                reason_parts.append(f"KC通道快速扩张({width_change_ratio:.1%}，波动性增加)")
            elif width_change_ratio < -0.1:  # 通道快速收缩
                strength_multiplier *= 0.8
                confidence_adjustment -= 0.1
                reason_parts.append(f"KC通道快速收缩({abs(width_change_ratio):.1%}，波动性减少)")
            elif abs(width_change_ratio) < 0.02:  # 通道宽度稳定
                confidence_adjustment += 0.05
                reason_parts.append("KC通道宽度稳定")
            
            # 价格相对通道宽度的调整
            price_channel_ratio = (current_price - latest_lower) / channel_width if channel_width > 0 else 0.5
            
            if signal_type in ['buy', 'sell']:
                # 极端位置增强信号
                if price_channel_ratio > 0.9 or price_channel_ratio < 0.1:
                    strength_multiplier *= 1.2
                    confidence_adjustment += 0.1
                    reason_parts.append("价格处于KC通道极端位置")
                    
                # 价格动量检查
                if len(data) >= 3:
                    price_momentum = (current_price - data['close'].iloc[-3]) / data['close'].iloc[-3]
                    if (signal_type == 'buy' and price_momentum > 0.02) or \
                       (signal_type == 'sell' and price_momentum < -0.02):
                        confidence_adjustment += 0.1
                        reason_parts.append("价格动量与信号一致")
            
            # 应用调整因子
            final_strength = min(1.0, base_strength * strength_multiplier)
            final_confidence = min(1.0, max(0.0, base_confidence + confidence_adjustment))
            
            # 如果没有明确信号，保持持有状态
            if not reason_parts:
                signal_type = 'hold'
                final_strength = 0.0
                final_confidence = 0.5
                reason_parts.append(f"价格位于KC通道中部({price_channel_ratio:.1%}位置)")
            
            # 构建元数据
            metadata = {
                'kc_upper': float(latest_upper),
                'kc_middle': float(latest_middle),
                'kc_lower': float(latest_lower),
                'channel_width': float(channel_width),
                'channel_width_change': float(width_change_ratio),
                'price_position_ratio': float(price_channel_ratio),
                'signal_source': 'KC_indicator',
                'calculation_method': 'keltner_channel_analysis',
                'data_points_used': len(kc_upper.dropna()),
                'period': self.period,
                'atr_period': self.atr_period,
                'multiplier': self.multiplier
            }
            
            # 添加通道位置分析
            if price_channel_ratio >= 0.8:
                metadata['channel_position'] = 'upper_extreme'
            elif price_channel_ratio >= 0.6:
                metadata['channel_position'] = 'upper_area'
            elif price_channel_ratio >= 0.4:
                metadata['channel_position'] = 'middle_area'
            elif price_channel_ratio >= 0.2:
                metadata['channel_position'] = 'lower_area'
            else:
                metadata['channel_position'] = 'lower_extreme'
            
            # 添加价格相关信息到元数据
            if 'close' in data.columns:
                metadata['current_price'] = float(current_price)
                metadata['price_vs_upper'] = float((current_price - latest_upper) / latest_upper) if latest_upper > 0 else 0.0
                metadata['price_vs_lower'] = float((current_price - latest_lower) / latest_lower) if latest_lower > 0 else 0.0
            
            return {
                'signal_type': signal_type,
                'strength': round(final_strength, 3),
                'confidence': round(final_confidence, 3),
                'timestamp': pd.Timestamp.now(),
                'price': float(current_price),
                'reason': '; '.join(reason_parts),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"KC信号生成失败: {e}")
            return self._get_default_signal(f"信号生成异常: {str(e)}")
    
    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        try:
            if data is None or data.empty:
                return False
                
            # 检查必需的列
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"KC信号生成缺少必需列: {col}")
                    return False
                    
            # 检查数据量
            min_periods = max(self.period, self.atr_period)
            if len(data) < min_periods:
                logger.warning(f"KC信号生成数据量不足: {len(data)} < {min_periods}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"KC数据验证失败: {e}")
            return False
    
    def _get_default_signal(self, reason: str = "无明确信号") -> Dict[str, Any]:
        """
        获取默认的持有信号
        
        Args:
            reason: 信号原因
            
        Returns:
            Dict[str, Any]: 默认信号
        """
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'price': 0.0,
            'reason': reason,
            'metadata': {
                'signal_source': 'KC_indicator',
                'default_signal': True,
                'indicator_name': 'KC'
            }
        }

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """生成交易信号"""
        result = self._calculate_kc(data)
        signals_df = pd.DataFrame(index=data.index)

        if "kc_upper" in result.columns and "kc_lower" in result.columns:
            close = data["close"]
            upper = result["kc_upper"]
            lower = result["kc_lower"]

            # KC信号:突破上轨=买入,跌破下轨=卖出
            signals_df["buy_signal"] = (close > upper).astype(int)
            signals_df["sell_signal"] = (close < lower).astype(int)

            # 信号强度基于突破程度
            middle = result["kc_middle"] if "kc_middle" in result.columns else (upper + lower) / 2
            signals_df["signal_strength"] = np.where(
                (close > upper) | (close < lower),
                "strong",
                np.where(abs(close - middle) / middle > 0.02, "moderate", "weak"),
            )
        else:
            signals_df["buy_signal"] = 0
            signals_df["sell_signal"] = 0
            signals_df["signal_strength"] = "weak"

        return signals_df

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """通用形态识别接口"""
        result = self._calculate_kc(data)
        patterns_df = pd.DataFrame(index=data.index)

        if "kc_upper" in result.columns and "kc_lower" in result.columns:
            close = data["close"]
            upper = result["kc_upper"]
            lower = result["kc_lower"]
            middle = result["kc_middle"] if "kc_middle" in result.columns else (upper + lower) / 2

            # KC形态识别
            patterns_df["KC_UPPER_BREAKOUT"] = close > upper
            patterns_df["KC_LOWER_BREAKOUT"] = close < lower
            patterns_df["KC_SQUEEZE"] = (
                (result["kc_width"] < result["kc_width"].rolling(20).mean() * 0.8)
                if "kc_width" in result.columns
                else False
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns_df["KC_EXPANSION"] = (
                (result["kc_width"] > result["kc_width"].rolling(20).mean() * 1.2)
                if "kc_width" in result.columns
                else False
            )  # TODO: 将魔法数字提取到配置中
            patterns_df["KC_MIDDLE_CROSS"] = (close > middle) & (close.shift(1) <= middle.shift(1))

        return patterns_df

    def calculate_confidence_Indicator_Base_Indicator(
        self, raw_score: pd.Series, patterns: pd.DataFrame, signals: Dict
    ) -> float:
        """计算置信度"""
        if raw_score.empty:
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        return min(
            0.9, max(0.1, abs(raw_score.iloc[-1] - 50) / 50)
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        result = self._calculate_kc(data)
        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        if "kc_upper" in result.columns and "kc_lower" in result.columns:
            close = data["close"]
            upper = result["kc_upper"]
            lower = result["kc_lower"]
            middle = result["kc_middle"] if "kc_middle" in result.columns else (upper + lower) / 2

            # 基于价格相对通道位置评分
            position = (close - lower) / (upper - lower)
            score = position * 100
            score = score.fillna(50.0)  # TODO: 将魔法数字提取到配置中

        return score

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """通用形态识别接口"""
        return self.get_patterns(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """通用参数设置接口"""
        self.set_parameters_Kc_Kc_Kc_kc(**kwargs)

    def _calculate_kc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算KC指标

        Args:
            df: 包含OHLCV数据的Data_frame

        Returns:
            包含kc_upper, kc_middle, kc_lower列的Data_frame
        """
        if self._result is not None:
            return self._result

        result = df.copy()

        # 计算中轨(EMA)
        result["kc_middle"] = result["close"].ewm(span=self.period, adjust=False).mean()

        # 计算真实波幅(TR)
        result["TR"] = np.maximum(
            result["high"] - result["low"],
            np.maximum(
                np.abs(result["high"] - result["close"].shift(1)), np.abs(result["low"] - result["close"].shift(1))
            ),
        )

        # 填充NaN值
        result["TR"] = result["TR"].fillna(result["high"] - result["low"])

        # 计算ATR
        result["ATR"] = result["TR"].rolling(window=self.atr_period).mean()

        # 计算上下轨
        result["kc_upper"] = result["kc_middle"] + self.multiplier * result["ATR"]
        result["kc_lower"] = result["kc_middle"] - self.multiplier * result["ATR"]

        # 计算通道宽度百分比(相对于中轨价格)
        result["kc_width"] = (result["kc_upper"] - result["kc_lower"]) / result["kc_middle"] * 100

        # 计算价格相对于通道的位置(0-100%),0表示在下轨,100表示在上轨
        channel_range = result["kc_upper"] - result["kc_lower"]
        # 避免除以零的情况
        result["kc_position"] = np.where(
            channel_range > 0,
            (result["close"] - result["kc_lower"]) / channel_range * 100,
            50,  # 默认为中间位置  # TODO: 将魔法数字提取到配置中
        )

        # 计算通道宽度变化率
        result["kc_width_chg"] = (
            result["kc_width"].pct_change(periods=5, fill_method=None) * 100
        )  # TODO: 将魔法数字提取到配置中

        # 删除临时计算列
        result = result.drop(["TR", "ATR"], axis=1)

        # 添加形态识别和信号生成
        result = self.add_pattern_detection(result)
        result = self.add_signal_generation(result)

        self._result = result
        return result

    def generate_signals_Kc(self, df: pd.DataFrame) -> List[Dict]:
        """
        生成标准化的交易信号

        Args:
            df: 包含OHLCV数据的Data_frame

        Returns:
            包含交易信号的字典列表
        """
        signals = []
        result = self.calculate(df)

        # 确保有足够的数据
        if len(result) < self.period + 5:  # TODO: 将魔法数字提取到配置中
            return signals

        # 获取最新数据
        latest = result.iloc[-1]
        prev = result.iloc[-2]

        # 当前价格
        current_price = latest["close"]

        # KC指标状态
        kc_middle = latest["kc_middle"]
        kc_upper = latest["kc_upper"]
        kc_lower = latest["kc_lower"]
        kc_position = latest["kc_position"]
        kc_width = latest["kc_width"]
        kc_width_chg = latest["kc_width_chg"]

        # 判断趋势方向
        if current_price > kc_middle:
            trend = Trend_type.UP
            trend_strength = 50  # TODO: 将魔法数字提取到配置中 + kc_position * 0.5  # 50-100  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        elif current_price < kc_middle:
            trend = Trend_type.DOWN
            trend_strength = 50  # TODO: 将魔法数字提取到配置中 - (100 - kc_position) * 0.5  # 0-50  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        else:
            trend = Trend_type.FLAT
            trend_strength = 50  # TODO: 将魔法数字提取到配置中

        # 基础信号评分(0-100)
        score = 50  # TODO: 将魔法数字提取到配置中  # 中性分值  # TODO: 将魔法数字提取到配置中

        # 判断价格与通道的关系
        if crossover(result["close"], result["kc_upper"]).any():  # 价格上穿上轨
            signal_type = "上穿上轨"
            signal_desc = "价格上穿肯特纳通道上轨,显示强势突破"
            cross_type = "GOLDEN_CROSS"
            score = 80  # TODO: 将魔法数字提取到配置中
        elif crossunder(result["close"], result["kc_lower"]).any():  # 价格下穿下轨
            signal_type = "下穿下轨"
            signal_desc = "价格下穿肯特纳通道下轨,显示弱势突破"
            cross_type = "DEATH_CROSS"
            score = 20  # TODO: 将魔法数字提取到配置中
        elif current_price > kc_upper:  # 价格在上轨之上
            signal_type = "上轨之上"
            signal_desc = "价格位于肯特纳通道上轨之上,显示超买状态"
            cross_type = Cross_type.NO_CROSS
            score = (
                70 + (current_price - kc_upper) / kc_upper * 100
            )  # 根据超出程度增加评分  # TODO: 将魔法数字提取到配置中
        elif current_price < kc_lower:  # 价格在下轨之下
            signal_type = "下轨之下"
            signal_desc = "价格位于肯特纳通道下轨之下,显示超卖状态"
            cross_type = "NO_CROSS"
            score = (
                30 - (kc_lower - current_price) / kc_lower * 100
            )  # 根据超出程度减少评分  # TODO: 将魔法数字提取到配置中
        elif crossover(result["close"], result["kc_middle"]).any():  # 价格上穿中轨
            signal_type = "上穿中轨"
            signal_desc = "价格上穿肯特纳通道中轨,显示由弱转强"
            cross_type = "GOLDEN_CROSS"
            score = 60  # TODO: 将魔法数字提取到配置中
        elif crossunder(result["close"], result["kc_middle"]).any():  # 价格下穿中轨
            signal_type = "下穿中轨"
            signal_desc = "价格下穿肯特纳通道中轨,显示由强转弱"
            cross_type = "DEATH_CROSS"
            score = 40  # TODO: 将魔法数字提取到配置中
        elif current_price > kc_middle:  # 价格在中轨和上轨之间
            signal_type = "中上区域"
            signal_desc = "价格位于肯特纳通道中轨和上轨之间,显示温和强势"
            cross_type = "NO_CROSS"
            score = (
                55 + kc_position * 0.15
            )  # 根据位置线性调整55-70  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        elif current_price < kc_middle:  # 价格在中轨和下轨之间
            signal_type = "中下区域"
            signal_desc = "价格位于肯特纳通道中轨和下轨之间,显示温和弱势"
            cross_type = "NO_CROSS"
            score = (
                45 - (100 - kc_position) * 0.15
            )  # 根据位置线性调整30-45  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        else:  # 价格在中轨上
            signal_type = "中轨位置"
            signal_desc = "价格位于肯特纳通道中轨,显示中性"
            cross_type = "NO_CROSS"
            score = 50  # TODO: 将魔法数字提取到配置中

        # 考虑通道宽度变化
        if kc_width_chg > 10:
            if current_price > kc_middle:
                score += 5  # TODO: 将魔法数字提取到配置中
                signal_desc += f",通道宽度扩大({kc_width_chg:.2f}%),上升波动加剧"
            else:
                score -= 5  # TODO: 将魔法数字提取到配置中
                signal_desc += f",通道宽度扩大({kc_width_chg:.2f}%),下降波动加剧"
        elif kc_width_chg < -10:
            signal_desc += f",通道宽度收窄({kc_width_chg:.2f}%),波动减弱,可能酝酿大行情"

        # 考虑通道宽度绝对水平
        if kc_width > 10:
            signal_desc += f",当前通道宽度较大({kc_width:.2f}%),市场波动性高"
        elif kc_width < 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            signal_desc += f",当前通道宽度较小({kc_width:.2f}%),市场波动性低,可能即将爆发"

        # 计算建议仓位(0-100%)
        if score >= 70:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            position_pct = min(100, score)
        elif score <= 30:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            position_pct = 0
        else:
            position_pct = (score - 30) * 100 / 40  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 生成买卖信号
        if score >= 70:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            buy_signal = True
            sell_signal = False
        elif score <= 30:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            buy_signal = False
            sell_signal = True
        else:
            buy_signal = False
            sell_signal = False

        # 计算置信度(0-100%)
        if cross_type in ["GOLDEN_CROSS", "DEATH_CROSS"]:
            if current_price > kc_upper or current_price < kc_lower:
                confidence = 85  # 突破外轨的交叉信号  # TODO: 将魔法数字提取到配置中
            else:
                confidence = 75  # 内部的交叉信号  # TODO: 将魔法数字提取到配置中
        elif current_price > kc_upper or current_price < kc_lower:
            confidence = 80  # 持续在外轨  # TODO: 将魔法数字提取到配置中
        else:
            confidence = (
                60 + abs(kc_position - 50) * 0.4
            )  # 根据位置调整  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 风险等级(1-5)  # TODO: 将魔法数字提取到配置中
        risk_level = 3  # TODO: 将魔法数字提取到配置中
        if kc_width > 8:  # TODO: 将魔法数字提取到配置中
            risk_level = 4  # 通道宽度大,波动性高  # TODO: 将魔法数字提取到配置中
        elif kc_width < 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            risk_level = 2  # 通道宽度小,波动性低

        # 止损计算
        if buy_signal:
            # 止损设为通道下轨或最近5天最低价,取较高者
            stop_loss = max(kc_lower, df["low"].iloc[-5:].min())  # TODO: 将魔法数字提取到配置中
        elif sell_signal:
            # 止损设为通道上轨或最近5天最高价,取较低者
            stop_loss = min(kc_upper, df["high"].iloc[-5:].max())  # TODO: 将魔法数字提取到配置中
        else:
            stop_loss = None

        # 创建信号字典
        signal = {
            "indicator": "KC",
            "timestamp": df.index[-1],
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "score": score,
            "trend": trend.value,
            "trend_strength": trend_strength,
            "signal_type": signal_type,
            "signal_desc": signal_desc,
            "cross_type": cross_type,
            "confidence": confidence,
            "risk_level": risk_level,
            "position_pct": position_pct,
            "stop_loss": stop_loss,
            "additional_info": {
                "kc_middle": kc_middle,
                "kc_upper": kc_upper,
                "kc_lower": kc_lower,
                "kc_position": kc_position,
                "kc_width": kc_width,
                "kc_width_chg": kc_width_chg,
            },
        }

        signals.append(signal)
        return signals

    def calculate_raw_score_Kc(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分(0-100分)

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            包含评分的Series,范围0-100
        """
        # 确保已计算指标
        if not isinstance(data, pd.DataFrame) or "kc_middle" not in data.columns:
            data = self.calculate(data)

        # 获取KC指标值
        close = data["close"]
        middle = data["kc_middle"]
        upper = data["kc_upper"]
        lower = data["kc_lower"]
        position = data["kc_position"]

        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中  # 默认中性评分

        # 价格位置评分
        # 1. 价格在上轨之上
        above_upper_mask = close > upper
        score[above_upper_mask] = (
            70 + (close[above_upper_mask] - upper[above_upper_mask]) / upper[above_upper_mask] * 100
        )  # TODO: 将魔法数字提取到配置中

        # 2. 价格在下轨之下
        below_lower_mask = close < lower
        score[below_lower_mask] = (
            30 - (lower[below_lower_mask] - close[below_lower_mask]) / lower[below_lower_mask] * 100
        )  # TODO: 将魔法数字提取到配置中

        # 3. 价格在中轨和上轨之间  # TODO: 将魔法数字提取到配置中
        between_mid_upper_mask = (close >= middle) & (close <= upper)
        score[between_mid_upper_mask] = (
            55 + position[between_mid_upper_mask] * 0.15
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 4. 价格在中轨和下轨之间  # TODO: 将魔法数字提取到配置中
        between_mid_lower_mask = (close <= middle) & (close >= lower)
        score[between_mid_lower_mask] = (
            45 - (100 - position[between_mid_lower_mask]) * 0.15
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 考虑交叉情况
        if len(data) >= 2:
            # 价格上穿上轨
            cross_up_upper_mask = (data["close"].shift(1) <= data["kc_upper"].shift(1)) & (
                data["close"] > data["kc_upper"]
            )
            score[cross_up_upper_mask] = 80  # TODO: 将魔法数字提取到配置中

            # 价格下穿下轨
            cross_down_lower_mask = (data["close"].shift(1) >= data["kc_lower"].shift(1)) & (
                data["close"] < data["kc_lower"]
            )
            score[cross_down_lower_mask] = 20  # TODO: 将魔法数字提取到配置中

            # 价格上穿中轨
            cross_up_middle_mask = (data["close"].shift(1) <= data["kc_middle"].shift(1)) & (
                data["close"] > data["kc_middle"]
            )
            score[cross_up_middle_mask] = 60  # TODO: 将魔法数字提取到配置中

            # 价格下穿中轨
            cross_down_middle_mask = (data["close"].shift(1) >= data["kc_middle"].shift(1)) & (
                data["close"] < data["kc_middle"]
            )
            score[cross_down_middle_mask] = 40  # TODO: 将魔法数字提取到配置中

        # 考虑通道宽度变化
        width_chg = data["kc_width_chg"]
        # 通道扩大,上升波动
        up_vol_mask = (width_chg > 10) & (close > middle)
        score[up_vol_mask] += 5  # TODO: 将魔法数字提取到配置中

        # 通道扩大,下降波动
        down_vol_mask = (width_chg > 10) & (close < middle)
        score[down_vol_mask] -= 5  # TODO: 将魔法数字提取到配置中

        # 确保分数在0-100范围内
        score = score.clip(0, 100)

        return score

    def calculate_confidence_Kc(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算KC指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态Data_frame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 基础置信度
        confidence = 0.5  # TODO: 将魔法数字提取到配置中

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.25  # TODO: 将魔法数字提取到配置中
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.1
        else:
            confidence += 0.15  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 2. 基于形态的置信度
        if isinstance(patterns, pd.DataFrame) and not patterns.empty:
            try:
                # 统计最近几个周期的形态数量
                numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    recent_data = (
                        patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                    )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    recent_patterns = recent_data.sum().sum()
                    if recent_patterns > 0:
                        confidence += min(recent_patterns * 0.05, 0.2)  # TODO: 将魔法数字提取到配置中
            except:
                pass

        # 3. 基于KC通道宽度的置信度  # TODO: 将魔法数字提取到配置中
        if hasattr(self, "_result") and self._result is not None and "kc_width" in self._result.columns:
            try:
                width_values = self._result["kc_width"].dropna()
                if len(width_values) > 0:
                    last_width = width_values.iloc[-1]
                    # 通道宽度适中时置信度较高
                    if 3 <= last_width <= 10:  # TODO: 将魔法数字提取到配置中
                        confidence += 0.15  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    elif last_width > 15:  # 通道过宽,波动性过高  # TODO: 将魔法数字提取到配置中
                        confidence -= 0.1
                    elif last_width < 1:  # 通道过窄,可能即将突破
                        confidence += 0.1
            except:
                pass

        # 4. 基于评分稳定性的置信度  # TODO: 将魔法数字提取到配置中
        if len(score) >= 5:  # TODO: 将魔法数字提取到配置中
            recent_scores = score.iloc[-5:]  # TODO: 将魔法数字提取到配置中
            score_stability = 1.0 - (recent_scores.std() / 50.0)  # TODO: 将魔法数字提取到配置中
            confidence += score_stability * 0.1

        return min(confidence, 1.0)

    def identify_patterns_Kc(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别KC指标形态

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            形态描述列表
        """
        # 确保已计算指标
        if not isinstance(data, pd.DataFrame) or "kc_middle" not in data.columns:
            data = self.calculate(data)

        # 获取KC数据
        close = data["close"]
        middle = data["kc_middle"]
        upper = data["kc_upper"]
        lower = data["kc_lower"]
        width = data["kc_width"]
        width_chg = data["kc_width_chg"]

        patterns = []

        # 检查价格位置
        if close.iloc[-1] > upper.iloc[-1]:
            patterns.append("KC超买区域")
        elif close.iloc[-1] < lower.iloc[-1]:
            patterns.append("KC超卖区域")
        elif close.iloc[-1] > middle.iloc[-1]:
            patterns.append("KC上行区域")
        elif close.iloc[-1] < middle.iloc[-1]:
            patterns.append("KC下行区域")

        # 检查交叉
        if len(data) >= 2:
            if close.iloc[-2] <= upper.iloc[-2] and close.iloc[-1] > upper.iloc[-1]:
                patterns.append("KC上穿上轨")
            elif close.iloc[-2] >= lower.iloc[-2] and close.iloc[-1] < lower.iloc[-1]:
                patterns.append("KC下穿下轨")
            elif close.iloc[-2] <= middle.iloc[-2] and close.iloc[-1] > middle.iloc[-1]:
                patterns.append("KC上穿中轨")
            elif close.iloc[-2] >= middle.iloc[-2] and close.iloc[-1] < middle.iloc[-1]:
                patterns.append("KC下穿中轨")

        # 检查通道宽度
        if width.iloc[-1] > 10:
            patterns.append("KC通道宽度大")
        elif width.iloc[-1] < 3:  # TODO: 将魔法数字提取到配置中
            patterns.append("KC通道宽度小")

        if width_chg.iloc[-1] > 10:
            patterns.append("KC通道扩张")
        elif width_chg.iloc[-1] < -10:
            patterns.append("KC通道收缩")

        # 检查价格在通道内的行为模式
        # 通道内震荡
        if (close.iloc[-5:] < upper.iloc[-5:]).all() and (
            close.iloc[-5:] > lower.iloc[-5:]
        ).all():  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            crossing_middle = False
            for i in range(1, 5):  # TODO: 将魔法数字提取到配置中
                if ((close.iloc[-i - 1] < middle.iloc[-i - 1]) and (close.iloc[-i] > middle.iloc[-i])) or (
                    (close.iloc[-i - 1] > middle.iloc[-i - 1]) and (close.iloc[-i] < middle.iloc[-i])
                ):
                    crossing_middle = True
                    break

            if crossing_middle:
                patterns.append("KC通道内震荡")

        # 连续触及上轨但未突破
        if (
            (close.iloc[-3:] <= upper.iloc[-3:]) & (close.iloc[-3:] >= upper.iloc[-3:] * 0.99)
        ).any():  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns.append("KC顶部测试")

        # 连续触及下轨但未突破
        if (
            (close.iloc[-3:] >= lower.iloc[-3:]) & (close.iloc[-3:] <= lower.iloc[-3:] * 1.01)
        ).any():  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns.append("KC底部测试")

        return patterns

    def get_patterns_Kc(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取KC指标的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的Data_frame
        """
        # 确保已计算KC
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        close = self._result["close"]
        middle = self._result["kc_middle"]
        upper = self._result["kc_upper"]
        lower = self._result["kc_lower"]
        width = self._result["kc_width"]

        patterns_df = pd.DataFrame(index=data.index)

        # 1. 价格位置形态
        patterns_df["KC_ABOVE_UPPER"] = close > upper
        patterns_df["KC_BELOW_LOWER"] = close < lower
        patterns_df["KC_ABOVE_MIDDLE"] = (close > middle) & (close <= upper)
        patterns_df["KC_BELOW_MIDDLE"] = (close < middle) & (close >= lower)
        patterns_df["KC_AT_MIDDLE"] = abs(close - middle) / middle < 0.01

        # 2. 突破形态
        patterns_df["KC_BREAK_UPPER"] = crossover(close, upper)
        patterns_df["KC_BREAK_LOWER"] = crossunder(close, lower)
        patterns_df["KC_BREAK_MIDDLE_UP"] = crossover(close, middle)
        patterns_df["KC_BREAK_MIDDLE_DOWN"] = crossunder(close, middle)

        # 3. 通道宽度形态  # TODO: 将魔法数字提取到配置中
        if len(width) >= 20:  # TODO: 将魔法数字提取到配置中
            width_ma = width.rolling(20).mean()  # TODO: 将魔法数字提取到配置中
            patterns_df["KC_WIDE_CHANNEL"] = width > width_ma * 1.5  # TODO: 将魔法数字提取到配置中
            patterns_df["KC_NARROW_CHANNEL"] = width < width_ma * 0.5  # TODO: 将魔法数字提取到配置中
            patterns_df["KC_EXPANDING"] = width > width.shift(1)
            patterns_df["KC_CONTRACTING"] = width < width.shift(1)

        # 4. 极值形态  # TODO: 将魔法数字提取到配置中
        patterns_df["KC_EXTREME_OVERBOUGHT"] = close > upper * 1.02
        patterns_df["KC_EXTREME_OVERSOLD"] = close < lower * 0.98  # TODO: 将魔法数字提取到配置中

        # 5. 回归形态  # TODO: 将魔法数字提取到配置中
        patterns_df["KC_RETURN_TO_MIDDLE"] = (close.shift(1) > upper.shift(1)) & (close <= upper) | (
            close.shift(1) < lower.shift(1)
        ) & (close >= lower)

        # 6. 震荡形态  # TODO: 将魔法数字提取到配置中
        if len(close) >= 10:
            # 检查是否在通道内震荡
            recent_close = close.iloc[-10:]
            recent_upper = upper.iloc[-10:]
            recent_lower = lower.iloc[-10:]
            recent_middle = middle.iloc[-10:]

            in_channel = (recent_close < recent_upper) & (recent_close > recent_lower)
            cross_middle = (
                ((recent_close.shift(1) < recent_middle.shift(1)) & (recent_close > recent_middle))
                | ((recent_close.shift(1) > recent_middle.shift(1)) & (recent_close < recent_middle))
            ).any()

            patterns_df["KC_OSCILLATING"] = in_channel.all() & cross_middle

        return patterns_df

    def register_patterns_Kc(self):
        """
        注册KC指标的技术形态
        """
        # 注册价格突破形态
        self.register_pattern_to_registry(
            pattern_id="KC_BREAK_UPPER",
            display_name="KC上轨突破",
            description="价格突破肯特纳通道上轨,强势信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="KC_BREAK_LOWER",
            display_name="KC下轨突破",
            description="价格跌破肯特纳通道下轨,弱势信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册中轨突破形态
        self.register_pattern_to_registry(
            pattern_id="KC_BREAK_MIDDLE_UP",
            display_name="KC中轨向上突破",
            description="价格向上突破肯特纳通道中轨,由弱转强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="KC_BREAK_MIDDLE_DOWN",
            display_name="KC中轨向下突破",
            description="价格向下突破肯特纳通道中轨,由强转弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册极值形态
        self.register_pattern_to_registry(
            pattern_id="KC_EXTREME_OVERBOUGHT",
            display_name="KC极度超买",
            description="价格远超肯特纳通道上轨,极度超买",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="KC_EXTREME_OVERSOLD",
            display_name="KC极度超卖",
            description="价格远低于肯特纳通道下轨,极度超卖",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册通道形态
        self.register_pattern_to_registry(
            pattern_id="KC_WIDE_CHANNEL",
            display_name="KC通道扩张",
            description="肯特纳通道宽度扩张,波动性增加但方向不确定",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

        self.register_pattern_to_registry(
            pattern_id="KC_NARROW_CHANNEL",
            display_name="KC通道收缩",
            description="肯特纳通道宽度收缩,可能酝酿突破但方向不确定",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

        # 注册KC状态形态(从centralized mapping迁移)
        self.register_pattern_to_registry(
            pattern_id="KC_ABOVE_MIDDLE",
            display_name="KC中轨上方",
            description="价格位于肯特纳通道中轨上方",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="KC_AT_MIDDLE",
            display_name="KC中轨附近",
            description="价格位于肯特纳通道中轨附近",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

        self.register_pattern_to_registry(
            pattern_id="KC_CONTRACTING",
            display_name="KC通道收缩",
            description="肯特纳通道收缩,波动率降低",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

        self.register_pattern_to_registry(
            pattern_id="KC_EXPANDING",
            display_name="KC通道扩张",
            description="肯特纳通道扩张,波动率增加",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL",
        )

    def get_pattern_info_Kc(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0,
        }

        # KC指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域,可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0,
            },
            "超卖区域": {
                "id": "超卖区域",
                "name": "超卖区域",
                "description": "指标进入超卖区域,可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0,
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域",
                "description": "指标处于中性区域,趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0,
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势,看涨信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 15.0,  # TODO: 将魔法数字提取到配置中
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势,看跌信号",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -15.0,  # TODO: 将魔法数字提取到配置中
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号,建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0,  # TODO: 将魔法数字提取到配置中
            },
            "卖出信号": {
                "id": "卖出信号",
                "name": "卖出信号",
                "description": "指标产生卖出信号,建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0,  # TODO: 将魔法数字提取到配置中
            },
        }

        return pattern_info_map.get(pattern_id, default_pattern)

    def _get_default_parameters_kc(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 20, "atr_period": 10, "multiplier": 2.0}  # TODO: 将魔法数字提取到配置中

    def set_parameters_Kc_Kc_Kc_kc_duplicate(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator

            validator = IndicatorParameterValidator()

            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)

            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters("KC", params)
            if not is_valid:
                from utils.logger import get_logger
                from db.sql_manager import SQLManager, QueryType

                logger = get_logger(__name__)
                logger.warning(f"KC参数验证失败: {'; '.join(errors)}")
                # 使用默认参数
                params = self._default_parameters.copy()

            # 设置参数(保持向后兼容)
            for key, value in params.items():
                setattr(self, key, value)

        except Exception:
            # 如果验证失败,静默处理
            pass

    @property
    def minimum_periods(self) -> int:
        """
        KeltnerChannel指标所需的最少数据周期数

        计算逻辑:基于参数 period(20), atr_period(10) 计算  # TODO: 将魔法数字提取到配置中

        Returns:
            int: 最少需要的数据周期数
        """
        period = self._parameters.get("period", 20)  # TODO: 将魔法数字提取到配置中
        atr_period = self._parameters.get("atr_period", 10)
        return max(period, atr_period) + 10


# 添加类别名供注册系统使用
KC = KeltnerChannel
KeltnerChannels = KeltnerChannel
