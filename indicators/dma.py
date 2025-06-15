#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from enums.indicator_types import TrendType, CrossType
from enums.indicator_enum import IndicatorEnum
from indicators.common import crossover, crossunder
from .base_indicator import BaseIndicator

logger = logging.getLogger(__name__)

class DMA(BaseIndicator):
    """
    轨道线指标 (Different of Moving Average)
    
    DMA指标由两条均线的差值组成，通过快速均线与慢速均线之差以及这个差值的移动平均线来判断中长期的买卖点。
    该指标适合中长期趋势判断，是一种典型的趋势跟踪指标。
    
    参数:
        fast_period: 短期均线周期，默认为10
        slow_period: 长期均线周期，默认为50
        ama_period: 差值平均线周期，默认为10
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 50, ama_period: int = 10,
                 name: str = "DMA", description: str = "轨道线指标"):
        """初始化DMA指标"""
        super().__init__(name, description)
        self.indicator_type = IndicatorEnum.DMA.name
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ama_period = ama_period
        self._result = None
        self.REQUIRED_COLUMNS = ['close']

    def set_parameters(self, fast_period: int = None, slow_period: int = None, ama_period: int = None):
        """
        设置指标参数

        Args:
            fast_period: 快速均线周期
            slow_period: 慢速均线周期
            ama_period: 差值平均线周期
        """
        if fast_period is not None:
            self.fast_period = fast_period
        if slow_period is not None:
            self.slow_period = slow_period
        if ama_period is not None:
            self.ama_period = ama_period
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算DMA指标
        
        Args:
            df: 包含close列的DataFrame
            
        Returns:
            包含DMA, AMA列的DataFrame
        """
        if self._result is not None:
            return self._result
            
        result = df.copy()
        
        # 计算快速均线和慢速均线
        result['FAST_MA'] = result['close'].rolling(window=self.fast_period).mean()
        result['SLOW_MA'] = result['close'].rolling(window=self.slow_period).mean()
        
        # 计算DMA值（两条均线之差）
        result['DMA'] = result['FAST_MA'] - result['SLOW_MA']
        
        # 计算DMA的移动平均线(AMA)
        result['AMA'] = result['DMA'].rolling(window=self.ama_period).mean()
        
        # 计算FASTMA与SLOWMA的百分比差值
        # 避免除以零
        result['DMA_PCT'] = np.where(
            result['SLOW_MA'] > 0,
            (result['FAST_MA'] / result['SLOW_MA'] - 1) * 100,
            0
        )
        
        # 计算FASTMA的变化率
        result['FAST_MA_CHG'] = result['FAST_MA'].pct_change(periods=5, fill_method=None) * 100
        
        # 删除不需要的临时列
        result = result.drop(['FAST_MA', 'SLOW_MA'], axis=1)
        
        self._result = result
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        生成标准化的交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含交易信号的字典列表
        """
        signals = []
        result = self.calculate(df)
        
        # 确保有足够的数据
        if len(result) < self.slow_period + 5:
            return signals
            
        # 获取最新数据
        latest = result.iloc[-1]
        prev = result.iloc[-2]
        
        # 当前价格
        current_price = latest['close']
        
        # DMA指标状态
        dma = latest['DMA']
        ama = latest['AMA']
        dma_pct = latest['DMA_PCT']
        fast_ma_chg = latest['FAST_MA_CHG']
        
        # 判断趋势方向
        if dma > 0 and dma > ama:
            trend = TrendType.UP
            trend_strength = min(100, 50 + dma_pct * 2)
        elif dma < 0 and dma < ama:
            trend = TrendType.DOWN
            trend_strength = min(100, 50 - dma_pct * 2)
        else:
            trend = TrendType.FLAT
            trend_strength = 50
        
        # 基础信号评分(0-100)
        score = 50  # 中性分值
        
        # 判断DMA和AMA的关系及DMA的绝对水平
        if crossover(result['DMA'], result['AMA']).any():  # DMA上穿AMA
            signal_type = "DMA上穿AMA"
            signal_desc = "DMA上穿AMA，显示由空头转为多头趋势"
            cross_type = "GOLDEN_CROSS"
            score = 70
        elif crossunder(result['DMA'], result['AMA']).any():  # DMA下穿AMA
            signal_type = "DMA下穿AMA"
            signal_desc = "DMA下穿AMA，显示由多头转为空头趋势"
            cross_type = "DEATH_CROSS"
            score = 30
        elif dma > ama and dma_pct > 0:  # DMA在AMA上方且为正
            signal_type = "多头趋势增强"
            signal_desc = f"DMA位于AMA上方，百分比差值为{dma_pct:.2f}%，多头趋势增强"
            cross_type = "NO_CROSS"
            score = 60 + min(30, dma_pct * 1.5)
        elif dma < ama and dma_pct < 0:  # DMA在AMA下方且为负
            signal_type = "空头趋势增强"
            signal_desc = f"DMA位于AMA下方，百分比差值为{dma_pct:.2f}%，空头趋势增强"
            cross_type = "NO_CROSS"
            score = 40 - min(30, abs(dma_pct * 1.5))
        elif dma > 0 and ama > 0:  # DMA和AMA都为正
            signal_type = "弱势多头"
            signal_desc = "DMA和AMA均为正值，处于弱势多头"
            cross_type = "NO_CROSS"
            score = 55
        elif dma < 0 and ama < 0:  # DMA和AMA都为负
            signal_type = "弱势空头"
            signal_desc = "DMA和AMA均为负值，处于弱势空头"
            cross_type = "NO_CROSS"
            score = 45
        else:  # 其他情况
            signal_type = "震荡整理"
            signal_desc = "DMA指标处于震荡状态，无明确方向"
            cross_type = "NO_CROSS"
            score = 50
            
        # 考虑FASTMA变化率调整评分
        if fast_ma_chg > 2:
            score += 5
            if signal_type.startswith("多头"):
                signal_desc += f"，短期均线加速上涨({fast_ma_chg:.2f}%)"
        elif fast_ma_chg < -2:
            score -= 5
            if signal_type.startswith("空头"):
                signal_desc += f"，短期均线加速下跌({fast_ma_chg:.2f}%)"
                
        # 计算建议仓位(0-100%)
        if score >= 70:
            position_pct = min(100, score)
        elif score <= 30:
            position_pct = 0
        else:
            position_pct = (score - 30) * 100 / 40
            
        # 生成买卖信号
        if score >= 70:
            buy_signal = True
            sell_signal = False
        elif score <= 30:
            buy_signal = False
            sell_signal = True
        else:
            buy_signal = False
            sell_signal = False
            
        # 计算置信度(0-100%)
        if cross_type in ["GOLDEN_CROSS", "DEATH_CROSS"]:
            confidence = 80
        elif abs(dma_pct) > 5:
            confidence = 75
        else:
            confidence = 60
            
        # 调整置信度根据趋势一致性
        if (dma > 0 and ama > 0) or (dma < 0 and ama < 0):
            confidence += 10
            
        # 风险等级(1-5)
        risk_level = 3
        if abs(dma_pct) > 10:
            risk_level = 4
            
        # 止损计算
        if buy_signal:
            # 止损设为当前价格的95%或最近5天最低价，取较高者
            stop_loss = max(current_price * 0.95, df['low'].iloc[-5:].min())
        elif sell_signal:
            # 止损设为当前价格的105%或最近5天最高价，取较低者
            stop_loss = min(current_price * 1.05, df['high'].iloc[-5:].max())
        else:
            stop_loss = None
            
        # 创建信号字典
        signal = {
            "indicator": "DMA",
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
                "dma": dma,
                "ama": ama,
                "dma_pct": dma_pct,
                "fast_ma_chg": fast_ma_chg
            },
            "market_environment": self.detect_market_environment(df).value if hasattr(self, 'detect_market_environment') else None,
            "volume_confirmation": self.check_volume_confirmation(df) if hasattr(self, 'check_volume_confirmation') else None
        }
        
        signals.append(signal)
        return signals
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分(0-100分)
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含评分的Series，范围0-100
        """
        # 确保已计算指标
        if not isinstance(data, pd.DataFrame) or 'DMA' not in data.columns:
            data = self.calculate(data)
        
        # 获取DMA指标值
        dma = data['DMA']
        ama = data['AMA']
        dma_pct = data['DMA_PCT']
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 默认中性评分
        
        # 计算趋势强度
        # 1. 上升趋势 (DMA > 0 且 DMA > AMA)
        uptrend_mask = (dma > 0) & (dma > ama)
        score[uptrend_mask] = 60 + np.minimum(30, dma_pct[uptrend_mask] * 1.5)
        
        # 2. 下降趋势 (DMA < 0 且 DMA < AMA)
        downtrend_mask = (dma < 0) & (dma < ama)
        score[downtrend_mask] = 40 - np.minimum(30, np.abs(dma_pct[downtrend_mask] * 1.5))
        
        # 3. 弱势多头 (DMA > 0 且 AMA > 0)
        weak_up_mask = (dma > 0) & (ama > 0) & ~uptrend_mask
        score[weak_up_mask] = 55
        
        # 4. 弱势空头 (DMA < 0 且 AMA < 0)
        weak_down_mask = (dma < 0) & (ama < 0) & ~downtrend_mask
        score[weak_down_mask] = 45
        
        # 考虑交叉情况
        if len(data) >= 2:
            # DMA上穿AMA
            cross_up_mask = (data['DMA'].shift(1) <= data['AMA'].shift(1)) & (data['DMA'] > data['AMA'])
            score[cross_up_mask] = 70
            
            # DMA下穿AMA
            cross_down_mask = (data['DMA'].shift(1) >= data['AMA'].shift(1)) & (data['DMA'] < data['AMA'])
            score[cross_down_mask] = 30
        
        # 考虑快速均线变化率
        fast_ma_chg = data['FAST_MA_CHG']
        # 快速上涨
        score[fast_ma_chg > 2] += 5
        # 快速下跌
        score[fast_ma_chg < -2] -= 5
        
        # 确保分数在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算DMA指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 75 or last_score < 25:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if isinstance(patterns, pd.DataFrame) and not patterns.empty:
            try:
                # 统计最近几个周期的形态数量
                numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    recent_data = patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                    recent_patterns = recent_data.sum().sum()
                    if recent_patterns > 0:
                        confidence += min(recent_patterns * 0.05, 0.2)
            except:
                pass

        # 3. 基于评分稳定性的置信度
        if len(score) >= 5:
            recent_scores = score.iloc[-5:]
            score_stability = 1.0 - (recent_scores.std() / 50.0)
            confidence += score_stability * 0.1

        return min(confidence, 1.0)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别DMA指标形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            形态描述列表
        """
        # 确保已计算指标
        if not isinstance(data, pd.DataFrame) or 'DMA' not in data.columns:
            data = self.calculate(data)
            
        # 获取DMA数据
        dma = data['DMA']
        ama = data['AMA']
        
        patterns = []
        
        # 检查趋势状态
        if dma.iloc[-1] > 0 and dma.iloc[-1] > ama.iloc[-1]:
            patterns.append("DMA多头趋势")
        elif dma.iloc[-1] < 0 and dma.iloc[-1] < ama.iloc[-1]:
            patterns.append("DMA空头趋势")
        elif dma.iloc[-1] > 0 and ama.iloc[-1] > 0:
            patterns.append("DMA弱势多头")
        elif dma.iloc[-1] < 0 and ama.iloc[-1] < 0:
            patterns.append("DMA弱势空头")
        else:
            patterns.append("DMA震荡整理")
            
        # 检查交叉
        if len(data) >= 2:
            if dma.iloc[-2] <= ama.iloc[-2] and dma.iloc[-1] > ama.iloc[-1]:
                patterns.append("DMA金叉")
            elif dma.iloc[-2] >= ama.iloc[-2] and dma.iloc[-1] < ama.iloc[-1]:
                patterns.append("DMA死叉")
                
        # 检查零轴交叉
        if len(data) >= 2:
            if dma.iloc[-2] <= 0 and dma.iloc[-1] > 0:
                patterns.append("DMA上穿零轴")
            elif dma.iloc[-2] >= 0 and dma.iloc[-1] < 0:
                patterns.append("DMA下穿零轴")
                
        # 检查DMA与AMA距离
        dma_ama_diff = abs(dma.iloc[-1] - ama.iloc[-1])
        avg_close = data['close'].mean()
        diff_pct = dma_ama_diff / avg_close * 100
        
        if diff_pct > 5:
            if dma.iloc[-1] > ama.iloc[-1]:
                patterns.append("DMA与AMA大幅偏离(看涨)")
            else:
                patterns.append("DMA与AMA大幅偏离(看跌)")
                
        # 检查DMA走势
        if len(data) >= 10:
            dma_trend = dma.iloc[-10:].diff().mean()
            
            if dma_trend > 0.1:
                patterns.append("DMA上升加速")
            elif dma_trend < -0.1:
                patterns.append("DMA下降加速")

        return patterns

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取DMA指标的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算DMA
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        dma = self._result['DMA']
        ama = self._result['AMA']
        dma_pct = self._result['DMA_PCT']

        patterns_df = pd.DataFrame(index=data.index)

        # 1. 趋势形态
        patterns_df['DMA_UPTREND'] = (dma > 0) & (dma > ama)
        patterns_df['DMA_DOWNTREND'] = (dma < 0) & (dma < ama)
        patterns_df['DMA_WEAK_UPTREND'] = (dma > 0) & (ama > 0) & ~patterns_df['DMA_UPTREND']
        patterns_df['DMA_WEAK_DOWNTREND'] = (dma < 0) & (ama < 0) & ~patterns_df['DMA_DOWNTREND']

        # 2. 交叉形态
        patterns_df['DMA_GOLDEN_CROSS'] = crossover(dma, ama)
        patterns_df['DMA_DEATH_CROSS'] = crossunder(dma, ama)

        # 3. 零轴穿越形态
        patterns_df['DMA_CROSS_UP_ZERO'] = crossover(dma, 0)
        patterns_df['DMA_CROSS_DOWN_ZERO'] = crossunder(dma, 0)
        patterns_df['DMA_ABOVE_ZERO'] = dma > 0
        patterns_df['DMA_BELOW_ZERO'] = dma < 0

        # 4. 强度形态
        patterns_df['DMA_STRONG_UPTREND'] = dma_pct > 5
        patterns_df['DMA_STRONG_DOWNTREND'] = dma_pct < -5

        # 5. 偏离形态
        if len(dma) >= 2:
            dma_ama_diff = abs(dma - ama)
            avg_close = data['close'].rolling(20).mean()
            diff_pct = (dma_ama_diff / avg_close * 100).fillna(0)

            patterns_df['DMA_LARGE_DIVERGENCE_UP'] = (diff_pct > 5) & (dma > ama)
            patterns_df['DMA_LARGE_DIVERGENCE_DOWN'] = (diff_pct > 5) & (dma < ama)

        # 6. 加速形态
        if len(dma) >= 10:
            dma_acceleration = dma.diff(5)
            patterns_df['DMA_ACCELERATION_UP'] = dma_acceleration > 0.1
            patterns_df['DMA_ACCELERATION_DOWN'] = dma_acceleration < -0.1

        return patterns_df

    def register_patterns(self):
        """
        注册DMA指标的技术形态
        """
        # 注册DMA趋势形态
        self.register_pattern_to_registry(
            pattern_id="DMA_UPTREND",
            display_name="DMA上升趋势",
            description="DMA大于0且DMA大于AMA，表示强势上升趋势",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

        self.register_pattern_to_registry(
            pattern_id="DMA_DOWNTREND",
            display_name="DMA下降趋势",
            description="DMA小于0且DMA小于AMA，表示强势下降趋势",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        # 注册DMA交叉形态
        self.register_pattern_to_registry(
            pattern_id="DMA_GOLDEN_CROSS",
            display_name="DMA金叉",
            description="DMA上穿AMA，显示由空头转为多头趋势",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="DMA_DEATH_CROSS",
            display_name="DMA死叉",
            description="DMA下穿AMA，显示由多头转为空头趋势",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0
        )

        # 注册DMA零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="DMA_CROSS_UP_ZERO",
            display_name="DMA上穿零轴",
            description="DMA从负值区域穿越零轴",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0
        )

        self.register_pattern_to_registry(
            pattern_id="DMA_CROSS_DOWN_ZERO",
            display_name="DMA下穿零轴",
            description="DMA从正值区域穿越零轴",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0
        )

        # 注册DMA强度形态
        self.register_pattern_to_registry(
            pattern_id="DMA_STRONG_UPTREND",
            display_name="DMA强势上涨",
            description="DMA百分比差值大于5%，表示强势上涨",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0
        )

        self.register_pattern_to_registry(
            pattern_id="DMA_STRONG_DOWNTREND",
            display_name="DMA强势下跌",
            description="DMA百分比差值小于-5%，表示强势下跌",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0
        )

    def get_pattern_info(self, pattern_id: str = None) -> Dict[str, Any]:
        """
        获取DMA指标的形态信息

        Args:
            pattern_id: 形态ID，如果为None则返回所有形态信息

        Returns:
            Dict[str, Any]: 形态信息字典
        """
        all_patterns = {
            'DMA_UPTREND': {
                'name': 'DMA上升趋势',
                'description': 'DMA大于0且DMA大于AMA，表示强势上升趋势',
                'type': 'trend',
                'strength': 'strong'
            },
            'DMA_DOWNTREND': {
                'name': 'DMA下降趋势',
                'description': 'DMA小于0且DMA小于AMA，表示强势下降趋势',
                'type': 'trend',
                'strength': 'strong'
            },
            'DMA_GOLDEN_CROSS': {
                'name': 'DMA金叉',
                'description': 'DMA上穿AMA，显示由空头转为多头趋势',
                'type': 'reversal',
                'strength': 'strong'
            },
            'DMA_DEATH_CROSS': {
                'name': 'DMA死叉',
                'description': 'DMA下穿AMA，显示由多头转为空头趋势',
                'type': 'reversal',
                'strength': 'strong'
            },
            'DMA_CROSS_UP_ZERO': {
                'name': 'DMA上穿零轴',
                'description': 'DMA从负值区域穿越零轴，趋势转正',
                'type': 'trend',
                'strength': 'medium'
            },
            'DMA_CROSS_DOWN_ZERO': {
                'name': 'DMA下穿零轴',
                'description': 'DMA从正值区域穿越零轴，趋势转负',
                'type': 'trend',
                'strength': 'medium'
            },
            'DMA_STRONG_UPTREND': {
                'name': 'DMA强势上涨',
                'description': 'DMA百分比差值大于5%，表示强势上涨',
                'type': 'trend',
                'strength': 'medium'
            },
            'DMA_STRONG_DOWNTREND': {
                'name': 'DMA强势下跌',
                'description': 'DMA百分比差值小于-5%，表示强势下跌',
                'type': 'trend',
                'strength': 'medium'
            }
        }

        if pattern_id is None:
            return all_patterns
        else:
            return all_patterns.get(pattern_id, {
                'name': pattern_id,
                'description': f'未知形态: {pattern_id}',
                'type': 'unknown',
                'strength': 'low'
            })