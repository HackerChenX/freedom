#!/usr/bin/env python3
"""
OBV (On-Balance Volume) 能量潮指标

OBV指标通过累计成交量来反映资金流向。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class OBV(BaseIndicator, PatternSignalMixin):
    """
    OBV (On-Balance Volume) 能量潮指标
    
    OBV指标通过累计成交量变化来判断资金流向。
    """
    
    def __init__(self, **kwargs):
        """
        初始化OBV指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "OBV"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters()
        
        # 应用用户参数
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"signal_period": 10}
    
    def set_parameters(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('OBV', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.signal_period = params.get('signal_period', 10)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算OBV指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了OBV指标的DataFrame
        """
        result = self._calculate(data, **kwargs)
        self._result = result
        return result
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算OBV指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了OBV指标的DataFrame
        """
        df = data.copy()
        
        # 确保数据有足够的长度
        if len(df) < 2:
            logger.warning(f"数据长度({len(df)})不足，返回原始数据")
            df['OBV'] = np.nan
            df['obv_ma'] = np.nan
            df['obv_signal'] = np.nan
            return df

        # 计算价格变化
        df['price_change'] = df['close'].diff()
        
        # 计算OBV (On-Balance Volume)
        obv = [0]  # 初始值为0
        for i in range(1, len(df)):
            if df['price_change'].iloc[i] > 0:
                # 价格上涨，加上成交量
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['price_change'].iloc[i] < 0:
                # 价格下跌，减去成交量
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                # 价格不变，OBV保持不变
                obv.append(obv[-1])
        
        df['OBV'] = obv
        df['obv'] = obv  # 为了一致性
        
        # 计算OBV移动平均线（信号线）
        df['obv_ma'] = df['OBV'].rolling(window=self.signal_period).mean()
        df[f'OBV_MA{self.signal_period}'] = df['obv_ma']  # 为了向后兼容
        
        # 计算OBV信号线（更短周期的移动平均）
        df['obv_signal'] = df['OBV'].rolling(window=5).mean()
        
        # 计算OBV变化率
        df['obv_change'] = df['OBV'].pct_change() * 100
        
        # 计算OBV波动率
        df['obv_volatility'] = df['obv_change'].rolling(window=10).std()
        
        # 计算OBV相对强度（与历史均值的关系）
        df['obv_strength'] = (df['OBV'] - df['OBV'].rolling(window=20).mean()) / (df['OBV'].rolling(window=20).std() + 1e-8)
        
        # 清理中间计算列
        df.drop(['price_change'], axis=1, inplace=True)
            
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（OBV指标特定逻辑）
        df = self._apply_obv_signal_logic(df)

        return df

    def _apply_obv_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用OBV指标特定的信号生成逻辑
        基于OBV值的变化和量价关系生成信号
        """
        try:
            # 获取OBV值
            if 'OBV' not in df.columns:
                # 如果没有OBV值，使用默认信号
                return df

            obv_value = df['OBV']
            obv_ma = df['obv_ma']
            obv_signal = df['obv_signal']
            close_price = df['close']

            # OBV信号生成逻辑：
            # BUY: OBV上升且价格上升（量价齐升），或OBV突破移动平均线
            # SELL: OBV下降且价格下降（量价齐跌），或OBV跌破移动平均线
            # HOLD: 量价背离或无明显趋势

            # 计算OBV和价格的变化
            obv_rising = obv_value > obv_value.shift(1)
            obv_falling = obv_value < obv_value.shift(1)
            price_rising = close_price > close_price.shift(1)
            price_falling = close_price < close_price.shift(1)
            
            # OBV与移动平均线的关系
            obv_above_ma = obv_value > obv_ma
            obv_below_ma = obv_value < obv_ma
            
            # OBV突破移动平均线
            obv_breakout_up = obv_above_ma & (obv_value.shift(1) <= obv_ma.shift(1))
            obv_breakdown = obv_below_ma & (obv_value.shift(1) >= obv_ma.shift(1))
            
            # 强势OBV信号
            strong_obv_up = obv_rising & (obv_value > obv_signal)
            strong_obv_down = obv_falling & (obv_value < obv_signal)

            # 生成信号
            df.loc[:, 'buy_signal'] = (obv_rising & price_rising) | obv_breakout_up | strong_obv_up
            df.loc[:, 'sell_signal'] = (obv_falling & price_falling) | obv_breakdown | strong_obv_down
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"OBV信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算OBV原始评分
        
        基于OBV指标的技术分析特点进行评分：
        1. OBV趋势评分 (35%)
        2. 量价关系评分 (30%)
        3. OBV突破评分 (25%)
        4. OBV强度评分 (10%)
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取OBV数据
        obv = self._result['OBV']
        obv_ma = self._result['obv_ma']
        obv_change = self._result['obv_change']
        obv_strength = self._result['obv_strength']
        close_price = data['close'] if 'close' in data.columns else self._result.get('close', pd.Series(index=data.index))
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 1. OBV趋势评分 (35%)
        # 基于OBV的变化趋势
        trend_score = pd.Series(0.0, index=data.index)
        
        obv_change_1 = obv - obv.shift(1)
        obv_change_2 = obv.shift(1) - obv.shift(2)
        
        # 连续上升
        trend_score = np.where((obv_change_1 > 0) & (obv_change_2 > 0), 15, trend_score)
        # 连续下降
        trend_score = np.where((obv_change_1 < 0) & (obv_change_2 < 0), -15, trend_score)
        # 单次上升
        trend_score = np.where((obv_change_1 > 0) & (obv_change_2 <= 0), 8, trend_score)
        # 单次下降
        trend_score = np.where((obv_change_1 < 0) & (obv_change_2 >= 0), -8, trend_score)
        
        # OBV与移动平均线的关系
        if len(obv_ma.dropna()) > 0:
            trend_score += np.where(obv > obv_ma, 3, -3)
        
        scores += trend_score * 0.35
        
        # 2. 量价关系评分 (30%)
        # 基于OBV与价格变化的一致性
        volume_price_score = pd.Series(0.0, index=data.index)
        
        if len(close_price.dropna()) > 0:
            price_change = close_price - close_price.shift(1)
            
            # 量价齐升（理想情况）
            volume_price_up = (obv_change_1 > 0) & (price_change > 0)
            volume_price_score = np.where(volume_price_up, 12, volume_price_score)
            
            # 量价齐跌（理想情况）
            volume_price_down = (obv_change_1 < 0) & (price_change < 0)
            volume_price_score = np.where(volume_price_down, -12, volume_price_score)
            
            # 量涨价跌（可能的底部信号）
            volume_up_price_down = (obv_change_1 > 0) & (price_change < 0)
            volume_price_score = np.where(volume_up_price_down, 6, volume_price_score)
            
            # 量跌价涨（可能的顶部信号）
            volume_down_price_up = (obv_change_1 < 0) & (price_change > 0)
            volume_price_score = np.where(volume_down_price_up, -6, volume_price_score)
        
        scores += volume_price_score * 0.3
        
        # 3. OBV突破评分 (25%)
        # 基于OBV突破重要阻力或支撑
        breakout_score = pd.Series(0.0, index=data.index)
        
        if len(obv_ma.dropna()) > 0:
            # 突破移动平均线
            obv_breakout_up = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
            breakout_score = np.where(obv_breakout_up, 12, breakout_score)
            
            # 跌破移动平均线
            obv_breakdown = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))
            breakout_score = np.where(obv_breakdown, -12, breakout_score)
            
            # 在移动平均线之上持续
            obv_above_sustained = (obv > obv_ma) & (obv.shift(1) > obv_ma.shift(1))
            breakout_score = np.where(obv_above_sustained, 5, breakout_score)
            
            # 在移动平均线之下持续
            obv_below_sustained = (obv < obv_ma) & (obv.shift(1) < obv_ma.shift(1))
            breakout_score = np.where(obv_below_sustained, -5, breakout_score)
        
        # OBV创新高/新低
        obv_high = obv.rolling(window=20).max()
        obv_low = obv.rolling(window=20).min()
        
        obv_new_high = obv >= obv_high
        obv_new_low = obv <= obv_low
        
        breakout_score += np.where(obv_new_high, 8, 0)
        breakout_score += np.where(obv_new_low, -8, 0)
        
        scores += breakout_score * 0.25
        
        # 4. OBV强度评分 (10%)
        # 基于OBV的相对强度
        strength_score = pd.Series(0.0, index=data.index)
        
        if len(obv_strength.dropna()) > 0:
            # 强势OBV
            strength_score = np.where(obv_strength > 1.5, 8, strength_score)
            strength_score = np.where((obv_strength > 0.5) & (obv_strength <= 1.5), 4, strength_score)
            strength_score = np.where((obv_strength >= -0.5) & (obv_strength <= 0.5), 0, strength_score)
            strength_score = np.where((obv_strength >= -1.5) & (obv_strength < -0.5), -4, strength_score)
            strength_score = np.where(obv_strength < -1.5, -8, strength_score)
        
        scores += strength_score * 0.1
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if self._result is None:
            return 0.5
            
        # 基于OBV指标的明确性计算置信度
        obv = self._result['OBV'].dropna()
        obv_volatility = self._result['obv_volatility'].dropna()
        
        if len(obv) == 0:
            return 0.5
        
        # 计算最近的OBV值
        recent_obv = obv.iloc[-1] if len(obv) > 0 else 0
        recent_volatility = obv_volatility.iloc[-1] if len(obv_volatility) > 0 else 5
        
        # OBV趋势一致性
        trend_consistency = 0
        if len(obv) >= 5:
            recent_trend = obv.iloc[-5:].diff().dropna()
            if len(recent_trend) > 0:
                # 如果趋势方向一致，提高置信度
                positive_changes = len(recent_trend[recent_trend > 0])
                negative_changes = len(recent_trend[recent_trend < 0])
                if positive_changes >= 3 or negative_changes >= 3:
                    trend_consistency = 0.25
                elif positive_changes >= 2 or negative_changes >= 2:
                    trend_consistency = 0.15
        
        # OBV变化幅度
        obv_magnitude = 0
        if len(obv) >= 2:
            recent_change = abs(obv.iloc[-1] - obv.iloc[-2])
            avg_change = abs(obv.diff()).mean()
            if recent_change > avg_change * 1.5:
                obv_magnitude = 0.2
            elif recent_change > avg_change:
                obv_magnitude = 0.1
        
        # 波动性适中性
        volatility_appropriateness = 0
        if 2 <= recent_volatility <= 8:
            volatility_appropriateness = 0.15
        elif 1 <= recent_volatility <= 12:
            volatility_appropriateness = 0.1
        
        base_confidence = 0.3 + trend_consistency + obv_magnitude + volatility_appropriateness
        return min(max(base_confidence, 0.2), 0.9)
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取OBV相关形态"""
        if not self.has_result():
            self.calculate(data, **kwargs)
            
        if self._result is None:
            return pd.DataFrame(index=data.index)
            
        patterns = pd.DataFrame(index=data.index)
        
        obv = self._result['OBV']
        obv_ma = self._result['obv_ma']
        obv_strength = self._result['obv_strength']
        
        # 基本形态
        obv_change = obv - obv.shift(1)
        patterns['OBV_RISING'] = obv_change > 0
        patterns['OBV_FALLING'] = obv_change < 0
        patterns['OBV_STABLE'] = abs(obv_change) < (obv.std() * 0.1)
        
        # 强度形态
        patterns['OBV_STRONG_RISING'] = obv_change > obv_change.std()
        patterns['OBV_STRONG_FALLING'] = obv_change < -obv_change.std()
        patterns['OBV_ACCELERATING'] = (obv_change > 0) & (obv_change > obv_change.shift(1))
        patterns['OBV_DECELERATING'] = (obv_change < 0) & (obv_change < obv_change.shift(1))
        
        # 与移动平均线的关系
        if len(obv_ma.dropna()) > 0:
            patterns['OBV_ABOVE_MA'] = obv > obv_ma
            patterns['OBV_BELOW_MA'] = obv < obv_ma
            patterns['OBV_BREAKOUT_UP'] = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
            patterns['OBV_BREAKDOWN'] = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))
        
        # 相对强度形态
        if len(obv_strength.dropna()) > 0:
            patterns['OBV_VERY_STRONG'] = obv_strength > 2
            patterns['OBV_STRONG'] = (obv_strength > 1) & (obv_strength <= 2)
            patterns['OBV_NEUTRAL'] = (obv_strength >= -1) & (obv_strength <= 1)
            patterns['OBV_WEAK'] = (obv_strength >= -2) & (obv_strength < -1)
            patterns['OBV_VERY_WEAK'] = obv_strength < -2
        
        # 极值形态
        obv_high = obv.rolling(window=20).max()
        obv_low = obv.rolling(window=20).min()
        patterns['OBV_NEW_HIGH'] = obv >= obv_high
        patterns['OBV_NEW_LOW'] = obv <= obv_low
        
        # 背离形态（需要价格数据）
        if 'close' in data.columns:
            close_price = data['close']
            price_high = close_price.rolling(window=10).max()
            price_low = close_price.rolling(window=10).min()
            
            # 顶背离：价格创新高，OBV未创新高
            patterns['OBV_TOP_DIVERGENCE'] = (close_price >= price_high) & (obv < obv.rolling(window=10).max())
            # 底背离：价格创新低，OBV未创新低
            patterns['OBV_BOTTOM_DIVERGENCE'] = (close_price <= price_low) & (obv > obv.rolling(window=10).min())
        
        return patterns
