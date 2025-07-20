#!/usr/bin/env python3
from utils.dependency_injection import get_logger
"""
MFI (Money Flow Index) 资金流量指标

MFI指标结合价格和成交量来衡量买卖压力。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class Mfi(BaseIndicator, PatternSignalMixin):
    """
    MFI (Money Flow Index) 资金流量指标
    
    MFI指标通过结合价格和成交量来识别超买超卖状态。
    """
    
    def __init__(self, **kwargs):
        """
        初始化MFI指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "MFI"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_mfi()
        
        # 应用用户参数
        self.set_parameters_Mfi(**kwargs)
    
    def _get_default_parameters_mfi(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14, "overbought": 80.0, "oversold": 20.0}
    
    def set_parameters_Mfi(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('MFI', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = params.get('period', 14)
        self.overbought = params.get('overbought', 80.0)
        self.oversold = params.get('oversold', 20.0)
    
    def calculate_Mfi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MFI指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了MFI指标的Data_frame
        """
        result = self._calculate_mfi(data, **kwargs)
        self._result = result
        return result

    def _calculate_mfi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算MFI指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了MFI指标的Data_frame
        """
        df = data.copy()

        # 确保数据有足够的长度
        if len(df) < self.period + 1:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({self.period + 1})，返回原始数据")
            df[f'MFI{self.period}'] = np.nan
            df['mfi'] = np.nan
            df['mfi_signal'] = np.nan
            return df

        # 计算典型价格 (Typical Price)
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3

        # 计算资金流量 (Money Flow)
        df['MF'] = df['TP'] * df['volume']

        # 计算价格变化
        df['TP_change'] = df['TP'].diff()

        # 分离正负资金流量
        df['PMF'] = np.where(df['TP_change'] > 0, df['MF'], 0)
        df['NMF'] = np.where(df['TP_change'] < 0, df['MF'], 0)

        # 计算资金流量比率
        pmf_sum = df['PMF'].rolling(window=self.period).sum()
        nmf_sum = df['NMF'].rolling(window=self.period).sum()

        # 计算MFI
        # 避免除零错误
        mfi_ratio = pmf_sum / (nmf_sum + 1e-10)  # 添加小数避免除零
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        df[f'MFI{self.period}'] = df['mfi']  # 为了向后兼容

        # 计算MFI信号线（移动平均）
        df['mfi_signal'] = df['mfi'].rolling(window=5).mean()
        
        # 计算MFI波动率
        df['mfi_volatility'] = df['mfi'].rolling(window=10).std()

        # 清理中间计算列
        df.drop(['TP', 'MF', 'TP_change', 'PMF', 'NMF'], axis=1, inplace=True)
            
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（MFI指标特定逻辑）
        df = self._apply_mfi_signal_logic(df)

        return df

    def _apply_mfi_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用MFI指标特定的信号生成逻辑
        基于MFI值的超买超卖区间生成信号
        """
        try:
            # 获取MFI值
            if 'mfi' not in df.columns:
                # 如果没有MFI值，使用默认信号
                return df

            mfi_value = df['mfi']
            mfi_signal = df['mfi_signal']

            # MFI信号生成逻辑：
            # BUY: MFI从超卖区间(< 20)向上突破，或MFI上穿信号线
            # SELL: MFI从超买区间(> 80)向下突破，或MFI下穿信号线
            # HOLD: MFI在正常区间(20-80)且无明显突破

            # 定义超买超卖区间
            oversold = mfi_value < self.oversold
            overbought = mfi_value > self.overbought
            normal = (mfi_value >= self.oversold) & (mfi_value <= self.overbought)

            # 检测突破和交叉
            mfi_rising = mfi_value > mfi_value.shift(1)
            mfi_falling = mfi_value < mfi_value.shift(1)
            
            # MFI与信号线交叉
            mfi_above_signal = mfi_value > mfi_signal
            mfi_below_signal = mfi_value < mfi_signal
            
            # 金叉死叉（当前上穿/下穿且前一期下穿/上穿）
            golden_cross = mfi_above_signal & (mfi_value.shift(1) <= mfi_signal.shift(1))
            death_cross = mfi_below_signal & (mfi_value.shift(1) >= mfi_signal.shift(1))

            # 生成信号
            df.loc[:, 'buy_signal'] = (oversold & mfi_rising) | (golden_cross & (mfi_value < 50))
            df.loc[:, 'sell_signal'] = (overbought & mfi_falling) | (death_cross & (mfi_value > 50))
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"MFI信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

    def calculate_raw_score_Mfi(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MFI原始评分
        
        基于MFI指标的技术分析特点进行评分：
        1. MFI位置评分 (40%)
        2. MFI趋势评分 (30%)
        3. 超买超卖评分 (20%)
        4. 背离信号评分 (10%)
        """
        if not self.has_result():
            self.calculate_Mfi(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取MFI数据
        mfi = self._result['mfi']
        mfi_signal = self._result['mfi_signal']
        mfi_volatility = self._result['mfi_volatility']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 1. MFI位置评分 (40%)
        # 基于MFI在0-100范围内的位置
        position_score = pd.Series(0.0, index=data.index)
        
        # 中性区间(30-70)评分较低，极端区间评分较高
        position_score = np.where(mfi < 20, 15, position_score)  # 超卖区间
        position_score = np.where((mfi >= 20) & (mfi < 30), 10, position_score)  # 接近超卖
        position_score = np.where((mfi >= 30) & (mfi < 40), 5, position_score)  # 偏弱
        position_score = np.where((mfi >= 40) & (mfi < 60), 0, position_score)  # 中性
        position_score = np.where((mfi >= 60) & (mfi < 70), 5, position_score)  # 偏强
        position_score = np.where((mfi >= 70) & (mfi < 80), 10, position_score)  # 接近超买
        position_score = np.where(mfi >= 80, 15, position_score)  # 超买区间
        
        scores += position_score * 0.4
        
        # 2. MFI趋势评分 (30%)
        # 基于MFI的变化趋势
        mfi_change = mfi - mfi.shift(1)
        mfi_change_2 = mfi.shift(1) - mfi.shift(2)
        
        trend_score = pd.Series(0.0, index=data.index)
        
        # 连续上升
        trend_score = np.where((mfi_change > 0) & (mfi_change_2 > 0), 12, trend_score)
        # 连续下降
        trend_score = np.where((mfi_change < 0) & (mfi_change_2 < 0), -12, trend_score)
        # 单次上升
        trend_score = np.where((mfi_change > 0) & (mfi_change_2 <= 0), 6, trend_score)
        # 单次下降
        trend_score = np.where((mfi_change < 0) & (mfi_change_2 >= 0), -6, trend_score)
        
        # 与信号线的关系
        if len(mfi_signal.dropna()) > 0:
            trend_score += np.where(mfi > mfi_signal, 3, -3)
        
        scores += trend_score * 0.3
        
        # 3. 超买超卖评分 (20%)
        # 基于MFI的超买超卖状态
        overbought_oversold_score = pd.Series(0.0, index=data.index)
        
        # 从超卖区间向上突破
        oversold_breakout = (mfi >= 20) & (mfi.shift(1) < 20)
        overbought_oversold_score = np.where(oversold_breakout, 15, overbought_oversold_score)
        
        # 从超买区间向下突破
        overbought_breakdown = (mfi <= 80) & (mfi.shift(1) > 80)
        overbought_oversold_score = np.where(overbought_breakdown, -15, overbought_oversold_score)
        
        # 在超买区间持续
        overbought_sustained = (mfi > 80) & (mfi.shift(1) > 80)
        overbought_oversold_score = np.where(overbought_sustained, -8, overbought_oversold_score)
        
        # 在超卖区间持续
        oversold_sustained = (mfi < 20) & (mfi.shift(1) < 20)
        overbought_oversold_score = np.where(oversold_sustained, 8, overbought_oversold_score)
        
        scores += overbought_oversold_score * 0.2
        
        # 4. 背离信号评分 (10%)
        # 基于MFI与价格的背离
        close_price = data['close'] if 'close' in data.columns else self._result.get('close', pd.Series(index=data.index))
        divergence_score = pd.Series(0.0, index=data.index)
        
        if len(close_price.dropna()) > 0:
            # 价格创新高但MFI未创新高（顶背离）
            price_high = close_price.rolling(window=5).max()
            mfi_high = mfi.rolling(window=5).max()
            
            price_new_high = close_price >= price_high
            mfi_not_new_high = mfi < mfi_high
            
            top_divergence = price_new_high & mfi_not_new_high
            divergence_score = np.where(top_divergence, -8, divergence_score)
            
            # 价格创新低但MFI未创新低（底背离）
            price_low = close_price.rolling(window=5).min()
            mfi_low = mfi.rolling(window=5).min()
            
            price_new_low = close_price <= price_low
            mfi_not_new_low = mfi > mfi_low
            
            bottom_divergence = price_new_low & mfi_not_new_low
            divergence_score = np.where(bottom_divergence, 8, divergence_score)
        
        scores += divergence_score * 0.1
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores

    def calculate_confidence_Mfi(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if self._result is None:
            return 0.5
            
        # 基于MFI指标的明确性计算置信度
        mfi = self._result['mfi'].dropna()
        mfi_volatility = self._result['mfi_volatility'].dropna()
        
        if len(mfi) == 0:
            return 0.5
        
        # 计算最近的MFI值
        recent_mfi = mfi.iloc[-1] if len(mfi) > 0 else 50
        recent_volatility = mfi_volatility.iloc[-1] if len(mfi_volatility) > 0 else 10
        
        # MFI位置明确性（极端位置置信度高）
        position_clarity = 0
        if recent_mfi < 20 or recent_mfi > 80:
            position_clarity = 0.3
        elif recent_mfi < 30 or recent_mfi > 70:
            position_clarity = 0.15
        
        # 趋势一致性
        trend_consistency = 0
        if len(mfi) >= 3:
            recent_trend = mfi.iloc[-3:].diff().dropna()
            if len(recent_trend) > 0:
                # 如果趋势方向一致，提高置信度
                if all(recent_trend > 0) or all(recent_trend < 0):
                    trend_consistency = 0.2
        
        # 波动性适中性（波动性太高或太低都降低置信度）
        volatility_appropriateness = 0
        if 5 <= recent_volatility <= 15:
            volatility_appropriateness = 0.15
        elif 3 <= recent_volatility <= 20:
            volatility_appropriateness = 0.1
        
        base_confidence = 0.35 + position_clarity + trend_consistency + volatility_appropriateness
        return min(max(base_confidence, 0.2), 0.9)

    def get_patterns_Mfi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取MFI相关形态"""
        if not self.has_result():
            self.calculate_Mfi(data, **kwargs)
            
        if self._result is None:
            return pd.DataFrame(index=data.index)
            
        patterns = pd.DataFrame(index=data.index)
        
        mfi = self._result['mfi']
        mfi_signal = self._result['mfi_signal']
        
        # 基本形态
        patterns['MFI_OVERSOLD'] = mfi < self.oversold
        patterns['MFI_OVERBOUGHT'] = mfi > self.overbought
        patterns['MFI_NORMAL'] = (mfi >= self.oversold) & (mfi <= self.overbought)
        
        # 强度形态
        patterns['MFI_STRONG_OVERSOLD'] = mfi < 10
        patterns['MFI_STRONG_OVERBOUGHT'] = mfi > 90
        patterns['MFI_WEAK'] = (mfi >= 20) & (mfi < 40)
        patterns['MFI_STRONG'] = (mfi > 60) & (mfi <= 80)
        
        # 趋势形态
        mfi_change = mfi - mfi.shift(1)
        patterns['MFI_RISING'] = mfi_change > 0
        patterns['MFI_FALLING'] = mfi_change < 0
        patterns['MFI_ACCELERATING'] = (mfi_change > 0) & (mfi_change > mfi_change.shift(1))
        patterns['MFI_DECELERATING'] = (mfi_change < 0) & (mfi_change < mfi_change.shift(1))
        
        # 交叉形态
        if len(mfi_signal.dropna()) > 0:
            patterns['MFI_ABOVE_SIGNAL'] = mfi > mfi_signal
            patterns['MFI_BELOW_SIGNAL'] = mfi < mfi_signal
            patterns['MFI_GOLDEN_CROSS'] = (mfi > mfi_signal) & (mfi.shift(1) <= mfi_signal.shift(1))
            patterns['MFI_DEATH_CROSS'] = (mfi < mfi_signal) & (mfi.shift(1) >= mfi_signal.shift(1))
        
        # 突破形态
        patterns['MFI_OVERSOLD_BREAKOUT'] = (mfi >= self.oversold) & (mfi.shift(1) < self.oversold)
        patterns['MFI_OVERBOUGHT_BREAKDOWN'] = (mfi <= self.overbought) & (mfi.shift(1) > self.overbought)
        
        return patterns
