#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
import logging
from .base_indicator import BaseIndicator
from utils.signal_utils import crossover, crossunder
from enums.signal_strength import SignalStrength

logger = logging.getLogger(__name__)

class AD(BaseIndicator):
    """
    累积/派发线指标 (Accumulation/Distribution Line)
    
    AD指标将每日的成交量按照收盘价与最高最低价的关系进行加权，
    以反映成交量与价格的关系，评估资金流入流出情况。
    
    该指标常用于判断价格趋势的强弱，特别是通过量价背离来预测价格可能的反转。
    """
    
    def __init__(self, name: str = "AD", description: str = "累积/派发线指标"):
        """初始化AD指标"""
        super().__init__(name, description)
        self.indicator_type = "AD"
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self._result = None
        
    def set_parameters(self, **kwargs):
        """设置指标参数"""
        pass

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算AD指标的置信度。
        """
        return 0.5
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算AD指标
        
        Args:
            df: 包含high, low, close, volume列的DataFrame
            
        Returns:
            包含AD和AD_MA列的DataFrame
        """
        # 检查必要列是否存在
        required_columns = ['high', 'low', 'close', 'volume']
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"输入数据缺少必要的列: {column}")
        
        # 创建副本以避免修改原始数据
        df_copy = df.copy()
        
        # 计算价格位置
        price_position = ((df_copy['close'] - df_copy['low']) - (df_copy['high'] - df_copy['close'])) / (df_copy['high'] - df_copy['low'])
        
        # 处理分母为0的情况
        price_position = price_position.replace([np.inf, -np.inf], 0)
        
        # 计算资金流乘数
        money_flow_multiplier = price_position * df_copy['volume']
        
        # 计算AD指标
        df_copy['AD'] = money_flow_multiplier.cumsum()
        
        # 计算AD的移动平均
        df_copy['AD_MA'] = df_copy['AD'].rolling(window=14).mean()
        
        # 保存结果
        self._result = df_copy
        return df_copy

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取AD指标的所有形态信息
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        if not self.has_result():
            self.calculate(data)
            
        result = []
        
        # 如果没有计算结果，先计算
        if not self.has_result():
            self.calculate(data)
            
        # 获取AD指标数据
        ad_data = self.result['AD']
        ad_ma_data = self.result['AD_MA']
        
        # 检测金叉形态
        if len(ad_data) >= 2:
            cross_over = crossover(ad_data.iloc[-2:], ad_ma_data.iloc[-2:])
            if cross_over.any():
                pattern_data = {
                    'pattern_id': "AD_GOLDEN_CROSS",
                    'display_name': "AD金叉",
                    'indicator_id': self.name,
                    'strength': SignalStrength.STRONG.value,
                    'duration': 1,
                    'details': {
                        'ad_value': float(ad_data.iloc[-1]),
                        'ad_ma_value': float(ad_ma_data.iloc[-1])
                    }
                }
                result.append(pattern_data)
        
        # 检测死叉形态
        if len(ad_data) >= 2:
            cross_under = crossunder(ad_data.iloc[-2:], ad_ma_data.iloc[-2:])
            if cross_under.any():
                pattern_data = {
                    'pattern_id': "AD_DEATH_CROSS",
                    'display_name': "AD死叉",
                    'indicator_id': self.name,
                    'strength': SignalStrength.STRONG_NEGATIVE.value,
                    'duration': 1,
                    'details': {
                        'ad_value': float(ad_data.iloc[-1]),
                        'ad_ma_value': float(ad_ma_data.iloc[-1])
                    }
                }
                result.append(pattern_data)
        
        # 检测量价背离
        if len(data) >= 30 and 'close' in data.columns:
            price_trend = data['close'].pct_change(5).iloc[-1]
            ad_trend = ad_data.pct_change(5).iloc[-1]
            
            # 价格上涨但AD下降（顶背离）
            if price_trend > 0.02 and ad_trend < -0.02:
                pattern_data = {
                    'pattern_id': "AD_PRICE_DIVERGENCE_TOP",
                    'display_name': "AD与价格顶背离",
                    'indicator_id': self.name,
                    'strength': SignalStrength.VERY_STRONG_NEGATIVE.value,
                    'duration': 3,
                    'details': {
                        'price_trend': float(price_trend),
                        'ad_trend': float(ad_trend)
                    }
                }
                result.append(pattern_data)
            # 价格下跌但AD上升（底背离）
            elif price_trend < -0.02 and ad_trend > 0.02:
                pattern_data = {
                    'pattern_id': "AD_PRICE_DIVERGENCE_BOTTOM",
                    'display_name': "AD与价格底背离",
                    'indicator_id': self.name,
                    'strength': SignalStrength.VERY_STRONG.value,
                    'duration': 3,
                    'details': {
                        'price_trend': float(price_trend),
                        'ad_trend': float(ad_trend)
                    }
                }
                result.append(pattern_data)
        
        # 检测量能变化
        if len(ad_data) >= 2:
            ad_change = ad_data.pct_change().iloc[-1]
            
            # AD快速上涨
            if ad_change > 0.05:
                pattern_data = {
                    'pattern_id': "AD_RAPID_INCREASE",
                    'display_name': "AD快速上涨",
                    'indicator_id': self.name,
                    'strength': SignalStrength.MODERATE.value,
                    'duration': 2,
                    'details': {
                        'change_rate': float(ad_change)
                    }
                }
                result.append(pattern_data)
            # AD快速下跌
            elif ad_change < -0.05:
                pattern_data = {
                    'pattern_id': "AD_RAPID_DECREASE",
                    'display_name': "AD快速下跌",
                    'indicator_id': self.name,
                    'strength': SignalStrength.SELL.value,
                    'duration': 2,
                    'details': {
                        'change_rate': float(ad_change)
                    }
                }
                result.append(pattern_data)
        
        return pd.DataFrame(result)

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算AD指标评分（0-100分制）

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            float: 综合评分（0-100）
        """
        raw_score = self.calculate_raw_score(data, **kwargs)

        if raw_score.empty:
            return 50.0  # 默认中性评分

        last_score = raw_score.iloc[-1]

        # 应用市场环境调整
        market_env = kwargs.get('market_env', self._market_environment)
        adjusted_score = self._apply_market_environment_adjustment(market_env, last_score)

        # 计算置信度
        patterns = self.get_patterns(data)
        confidence = self.calculate_confidence(pd.Series([adjusted_score]), patterns, {})

        # 返回最终评分
        return float(np.clip(adjusted_score * confidence, 0, 100))

    def _apply_market_environment_adjustment(self, market_env, score: float) -> float:
        """
        根据市场环境调整评分

        Args:
            market_env: 市场环境
            score: 原始评分

        Returns:
            float: 调整后的评分
        """
        from indicators.base_indicator import MarketEnvironment

        if market_env == MarketEnvironment.BULL_MARKET:
            # 牛市中增强多头信号，弱化空头信号
            if score > 50:
                return score + (score - 50) * 0.2  # 多头信号增强
            else:
                return score + (score - 50) * 0.1  # 空头信号减弱
        elif market_env == MarketEnvironment.BEAR_MARKET:
            # 熊市中增强空头信号，弱化多头信号
            if score < 50:
                return score - (50 - score) * 0.2  # 空头信号增强
            else:
                return score - (score - 50) * 0.1  # 多头信号减弱
        elif market_env == MarketEnvironment.VOLATILE_MARKET:
            # 高波动市场需要更强的信号
            if score > 60 or score < 40:
                return score + (score - 50) * 0.15  # 极端信号更极端
            else:
                return 50 + (score - 50) * 0.8  # 中性信号更中性
        else:
            # 震荡市场，保持原评分
            return score
    
    def has_result(self) -> bool:
        """检查是否已计算结果"""
        return self._result is not None and not self._result.empty
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成AD指标的标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算AD指标
        if not self.has_result():
            self.calculate(data)
        
        # 获取AD相关值
        ad = self._result['AD']
        ad_ma = self._result['AD_MA']
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = '中性'
        signals['volume_confirmation'] = False
        
        # 计算ATR用于止损设置
        try:
            from indicators.atr import ATR
            atr_indicator = ATR()
            atr_data = atr_indicator.calculate(data)
            atr_values = atr_data['atr']
        except Exception as e:
            logger.warning(f"计算ATR失败: {e}")
            atr_values = pd.Series(0, index=data.index)
        
        # 1. AD上穿其均线，买入信号
        ad_crossover_ma = crossover(ad, ad_ma)
        signals.loc[ad_crossover_ma, 'buy_signal'] = True
        signals.loc[ad_crossover_ma, 'neutral_signal'] = False
        signals.loc[ad_crossover_ma, 'trend'] = 1
        signals.loc[ad_crossover_ma, 'signal_type'] = 'AD金叉'
        signals.loc[ad_crossover_ma, 'signal_desc'] = 'AD上穿其均线，表明买盘资金增加'
        signals.loc[ad_crossover_ma, 'confidence'] = 65.0
        signals.loc[ad_crossover_ma, 'position_size'] = 0.3
        signals.loc[ad_crossover_ma, 'risk_level'] = '中'
        
        # 2. AD下穿其均线，卖出信号
        ad_crossunder_ma = crossunder(ad, ad_ma)
        signals.loc[ad_crossunder_ma, 'sell_signal'] = True
        signals.loc[ad_crossunder_ma, 'neutral_signal'] = False
        signals.loc[ad_crossunder_ma, 'trend'] = -1
        signals.loc[ad_crossunder_ma, 'signal_type'] = 'AD死叉'
        signals.loc[ad_crossunder_ma, 'signal_desc'] = 'AD下穿其均线，表明卖盘资金增加'
        signals.loc[ad_crossunder_ma, 'confidence'] = 65.0
        signals.loc[ad_crossunder_ma, 'position_size'] = 0.3
        signals.loc[ad_crossunder_ma, 'risk_level'] = '中'
        
        # 3. 正背离信号（价格创新低，但AD未创新低）
        if 'close' in data.columns:
            close = data['close']
            
            # 获取价格和AD的局部低点
            lows_close = pd.Series(np.nan, index=close.index)
            lows_ad = pd.Series(np.nan, index=ad.index)
            
            # 简单的局部低点检测：如果一个点比前后N个点都低，则为局部低点
            window = 5
            for i in range(window, len(close) - window):
                if close.iloc[i] == close.iloc[i-window:i+window+1].min():
                    lows_close.iloc[i] = close.iloc[i]
                if ad.iloc[i] == ad.iloc[i-window:i+window+1].min():
                    lows_ad.iloc[i] = ad.iloc[i]
            
            # 检测正背离：价格创新低但AD未创新低
            for i in range(window*2, len(close)):
                if pd.notna(lows_close.iloc[i]) and pd.notna(lows_close.iloc[i-window*2:i-window].dropna().min()):
                    # 价格创新低
                    if lows_close.iloc[i] < lows_close.iloc[i-window*2:i-window].dropna().min():
                        # 查找相应时期的AD低点
                        recent_low_ad = lows_ad.iloc[i-window//2:i+window//2].dropna().min() if not lows_ad.iloc[i-window//2:i+window//2].dropna().empty else np.nan
                        prev_low_ad = lows_ad.iloc[i-window*2-window//2:i-window+window//2].dropna().min() if not lows_ad.iloc[i-window*2-window//2:i-window+window//2].dropna().empty else np.nan
                        
                        # AD未创新低（正背离）
                        if pd.notna(recent_low_ad) and pd.notna(prev_low_ad) and recent_low_ad > prev_low_ad:
                            # 只有在没有其他信号时才设置背离信号
                            if not signals.iloc[i]['buy_signal'] and not signals.iloc[i]['sell_signal']:
                                signals.iloc[i, signals.columns.get_loc('buy_signal')] = True
                                signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                                signals.iloc[i, signals.columns.get_loc('trend')] = 1
                                signals.iloc[i, signals.columns.get_loc('signal_type')] = 'AD正背离'
                                signals.iloc[i, signals.columns.get_loc('signal_desc')] = '价格创新低但AD未创新低，表明下跌动能减弱'
                                signals.iloc[i, signals.columns.get_loc('confidence')] = 75.0
                                signals.iloc[i, signals.columns.get_loc('position_size')] = 0.4
                                signals.iloc[i, signals.columns.get_loc('risk_level')] = '低'
        
            # 4. 负背离信号（价格创新高，但AD未创新高）
            highs_close = pd.Series(np.nan, index=close.index)
            highs_ad = pd.Series(np.nan, index=ad.index)
            
            # 简单的局部高点检测
            for i in range(window, len(close) - window):
                if close.iloc[i] == close.iloc[i-window:i+window+1].max():
                    highs_close.iloc[i] = close.iloc[i]
                if ad.iloc[i] == ad.iloc[i-window:i+window+1].max():
                    highs_ad.iloc[i] = ad.iloc[i]
            
            # 检测负背离：价格创新高但AD未创新高
            for i in range(window*2, len(close)):
                if pd.notna(highs_close.iloc[i]) and pd.notna(highs_close.iloc[i-window*2:i-window].dropna().max()):
                    # 价格创新高
                    if highs_close.iloc[i] > highs_close.iloc[i-window*2:i-window].dropna().max():
                        # 查找相应时期的AD高点
                        recent_high_ad = highs_ad.iloc[i-window//2:i+window//2].dropna().max() if not highs_ad.iloc[i-window//2:i+window//2].dropna().empty else np.nan
                        prev_high_ad = highs_ad.iloc[i-window*2-window//2:i-window+window//2].dropna().max() if not highs_ad.iloc[i-window*2-window//2:i-window+window//2].dropna().empty else np.nan
                        
                        # AD未创新高（负背离）
                        if pd.notna(recent_high_ad) and pd.notna(prev_high_ad) and recent_high_ad < prev_high_ad:
                            # 只有在没有其他信号时才设置背离信号
                            if not signals.iloc[i]['buy_signal'] and not signals.iloc[i]['sell_signal']:
                                signals.iloc[i, signals.columns.get_loc('sell_signal')] = True
                                signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                                signals.iloc[i, signals.columns.get_loc('trend')] = -1
                                signals.iloc[i, signals.columns.get_loc('signal_type')] = 'AD负背离'
                                signals.iloc[i, signals.columns.get_loc('signal_desc')] = '价格创新高但AD未创新高，表明上涨动能减弱'
                                signals.iloc[i, signals.columns.get_loc('confidence')] = 75.0
                                signals.iloc[i, signals.columns.get_loc('position_size')] = 0.4
                                signals.iloc[i, signals.columns.get_loc('risk_level')] = '低'
        
        # 5. AD趋势
        ad_trend = pd.Series(np.nan, index=ad.index)
        for i in range(20, len(ad)):
            # 计算20日趋势
            ad_slope = (ad.iloc[i] - ad.iloc[i-20]) / 20
            
            if ad_slope > 0:
                # AD上升趋势
                signals.iloc[i, signals.columns.get_loc('trend')] = 1
                if not signals.iloc[i]['buy_signal'] and not signals.iloc[i]['sell_signal']:
                    signals.iloc[i, signals.columns.get_loc('buy_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'AD上升趋势'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = 'AD持续上升，表明买盘持续涌入'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 60.0
                    signals.iloc[i, signals.columns.get_loc('position_size')] = 0.2
                    signals.iloc[i, signals.columns.get_loc('risk_level')] = '中'
            elif ad_slope < 0:
                # AD下降趋势
                signals.iloc[i, signals.columns.get_loc('trend')] = -1
                if not signals.iloc[i]['buy_signal'] and not signals.iloc[i]['sell_signal']:
                    signals.iloc[i, signals.columns.get_loc('sell_signal')] = True
                    signals.iloc[i, signals.columns.get_loc('neutral_signal')] = False
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'AD下降趋势'
                    signals.iloc[i, signals.columns.get_loc('signal_desc')] = 'AD持续下降，表明卖盘持续涌出'
                    signals.iloc[i, signals.columns.get_loc('confidence')] = 60.0
                    signals.iloc[i, signals.columns.get_loc('position_size')] = 0.2
                    signals.iloc[i, signals.columns.get_loc('risk_level')] = '中'
            
        # 6. 计算评分
        for i in range(len(signals)):
            if i > 0:  # 跳过第一个数据点
                # 基础分数是50
                score = 50.0
                
                # 根据AD趋势评分
                if i < len(ad) and i >= 20:
                    ad_slope = (ad.iloc[i] - ad.iloc[i-20]) / 20
                    
                    # 根据AD斜率调整分数
                    slope_score = ad_slope * 1000  # 缩放斜率以获得合适的分数调整
                    score += np.clip(slope_score, -25, 25)  # 限制斜率对分数的影响
                
                # AD与其均线的关系
                if i < len(ad) and i < len(ad_ma):
                    if ad.iloc[i] > ad_ma.iloc[i]:
                        score += 10  # AD在均线上方加10分
                    else:
                        score -= 10  # AD在均线下方减10分
                
                # 根据信号类型额外调整分数
                if signals.iloc[i]['signal_type'] == 'AD正背离':
                    score += 15
                elif signals.iloc[i]['signal_type'] == 'AD负背离':
                    score -= 15
                
                # 限制评分范围在0-100之间
                signals.iloc[i, signals.columns.get_loc('score')] = max(0, min(100, score))
        
        # 设置止损价格
        if 'low' in data.columns and 'high' in data.columns:
            # 买入信号的止损设为最近的低点
            buy_indices = signals[signals['buy_signal']].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx in data.index and idx > data.index[10]:  # 确保有足够的历史数据
                        pos = data.index.get_loc(idx)
                        if pos >= 10:
                            lookback = 5
                            recent_low = data.iloc[pos-lookback:pos]['low'].min()
                            atr_val = atr_values.iloc[pos] if pos < len(atr_values) else 0
                            signals.loc[idx, 'stop_loss'] = recent_low - atr_val
        
            # 卖出信号的止损设为最近的高点
            sell_indices = signals[signals['sell_signal']].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx in data.index and idx > data.index[10]:  # 确保有足够的历史数据
                        pos = data.index.get_loc(idx)
                        if pos >= 10:
                            lookback = 5
                            recent_high = data.iloc[pos-lookback:pos]['high'].max()
                            atr_val = atr_values.iloc[pos] if pos < len(atr_values) else 0
                            signals.loc[idx, 'stop_loss'] = recent_high + atr_val
        
        # 根据AD趋势判断市场环境
        signals['market_env'] = '中性'  # 默认中性市场
        
        # 计算20日AD趋势
        for i in range(20, len(ad)):
            ad_slope = (ad.iloc[i] - ad.iloc[i-20]) / 20
            
            if ad_slope > 0:
                # 上升趋势强度判断
                if ad_slope > ad.iloc[i-20:i].diff().std() * 2:
                    signals.iloc[i, signals.columns.get_loc('market_env')] = '强势'
                else:
                    signals.iloc[i, signals.columns.get_loc('market_env')] = '中性偏强'
            elif ad_slope < 0:
                # 下降趋势强度判断
                if abs(ad_slope) > ad.iloc[i-20:i].diff().std() * 2:
                    signals.iloc[i, signals.columns.get_loc('market_env')] = '弱势'
                else:
                    signals.iloc[i, signals.columns.get_loc('market_env')] = '中性偏弱'
            else:
                signals.iloc[i, signals.columns.get_loc('market_env')] = '震荡'
        
        # 设置成交量确认
        if 'volume' in data.columns:
            # 如果有成交量数据，检查成交量是否支持当前信号
            vol = data['volume']
            vol_avg = vol.rolling(window=20).mean()
            
            # 成交量大于20日均量1.5倍为放量
            vol_increase = vol > vol_avg * 1.5
            
            # 买入信号且成交量放大，确认信号
            signals.loc[signals['buy_signal'] & vol_increase, 'volume_confirmation'] = True
            
            # 卖出信号且成交量放大，确认信号
            signals.loc[signals['sell_signal'] & vol_increase, 'volume_confirmation'] = True
        
        return signals
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算AD原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算AD指标
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取AD相关值
        ad = self._result['AD']
        ad_ma = self._result['AD_MA']
        
        # 初始基础分50分
        score = pd.Series(50.0, index=data.index)
        
        # 基于AD趋势的评分
        for i in range(20, len(score)):
            if i >= len(ad):
                continue
                
            # 计算20日AD趋势
            ad_slope = (ad.iloc[i] - ad.iloc[i-20]) / 20
            
            # 根据AD斜率调整分数
            slope_score = ad_slope * 1000  # 缩放斜率以获得合适的分数调整
            score.iloc[i] += np.clip(slope_score, -25, 25)  # 限制斜率对分数的影响
            
            # AD与其均线的关系
            if i < len(ad_ma):
                if ad.iloc[i] > ad_ma.iloc[i]:
                    score.iloc[i] += 10  # AD在均线上方加10分
                else:
                    score.iloc[i] -= 10  # AD在均线下方减10分
        
        # 限制评分范围在0-100之间
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别AD技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算AD指标
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None:
            return patterns
        
        # 获取AD相关值
        ad = self._result['AD']
        ad_ma = self._result['AD_MA']
        
        # 最后一个有效的索引位置
        last_valid_idx = -1
        while last_valid_idx >= -len(ad) and pd.isna(ad.iloc[last_valid_idx]):
            last_valid_idx -= 1
        
        if last_valid_idx < -len(ad):
            return patterns
        
        # 1. AD与均线交叉形态
        if last_valid_idx-1 >= -len(ad) and ad.iloc[last_valid_idx-1] < ad_ma.iloc[last_valid_idx-1] and ad.iloc[last_valid_idx] > ad_ma.iloc[last_valid_idx]:
            patterns.append("AD金叉（AD上穿均线）")
        
        if last_valid_idx-1 >= -len(ad) and ad.iloc[last_valid_idx-1] > ad_ma.iloc[last_valid_idx-1] and ad.iloc[last_valid_idx] < ad_ma.iloc[last_valid_idx]:
            patterns.append("AD死叉（AD下穿均线）")
        
        # 2. AD趋势形态
        if last_valid_idx >= 20:
            ad_slope = (ad.iloc[last_valid_idx] - ad.iloc[last_valid_idx-20]) / 20
            
            if ad_slope > 0:
                patterns.append("AD上升趋势（买盘持续涌入）")
                
                # 上升趋势强度判断
                if ad_slope > ad.iloc[last_valid_idx-20:last_valid_idx].diff().std() * 2:
                    patterns.append("AD强势上升（买盘强烈涌入）")
            else:
                patterns.append("AD下降趋势（卖盘持续涌出）")
                
                # 下降趋势强度判断
                if abs(ad_slope) > ad.iloc[last_valid_idx-20:last_valid_idx].diff().std() * 2:
                    patterns.append("AD强势下降（卖盘强烈涌出）")
        
        # 3. AD背离形态
        if 'close' in data.columns and last_valid_idx >= 40:
            close = data['close']
            
            # 检查最近的两个价格低点
            last_20_min_idx = close.iloc[last_valid_idx-20:last_valid_idx].idxmin()
            prev_20_min_idx = close.iloc[last_valid_idx-40:last_valid_idx-20].idxmin()
            
            if last_20_min_idx is not None and prev_20_min_idx is not None:
                # 获取对应的AD值
                ad_at_last_low = ad.loc[last_20_min_idx]
                ad_at_prev_low = ad.loc[prev_20_min_idx]
                
                # 价格创新低但AD未创新低（正背离）
                if close.loc[last_20_min_idx] < close.loc[prev_20_min_idx] and ad_at_last_low > ad_at_prev_low:
                    patterns.append("AD正背离（价格创新低但AD未创新低）")
            
            # 检查最近的两个价格高点
            last_20_max_idx = close.iloc[last_valid_idx-20:last_valid_idx].idxmax()
            prev_20_max_idx = close.iloc[last_valid_idx-40:last_valid_idx-20].idxmax()
            
            if last_20_max_idx is not None and prev_20_max_idx is not None:
                # 获取对应的AD值
                ad_at_last_high = ad.loc[last_20_max_idx]
                ad_at_prev_high = ad.loc[prev_20_max_idx]
                
                # 价格创新高但AD未创新高（负背离）
                if close.loc[last_20_max_idx] > close.loc[prev_20_max_idx] and ad_at_last_high < ad_at_prev_high:
                    patterns.append("AD负背离（价格创新高但AD未创新高）")
        
        return patterns 