#!/usr/bin/env python3
from utils.dependency_injection import get_logger
"""
VORTEX (Vortex Indicator) 涡流指标

涡流指标用于识别趋势的开始和结束，通过比较正向和负向价格运动来衡量趋势强度。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class Vortex(BaseIndicator, PatternSignalMixin):
    """
    VORTEX (Vortex Indicator) 涡流指标
    
    涡流指标通过计算正向和负向价格运动的比率来识别趋势。
    VI+ > VI- 表示上升趋势
    VI- > VI+ 表示下降趋势
    """
    
    def __init__(self, **kwargs):
        """
        初始化VORTEX指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "VORTEX"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_vortex()
        
        # 应用用户参数
        self.set_parameters_Vortex(**kwargs)
    
    def _get_default_parameters_vortex(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """实现抽象方法"""
        self.set_parameters_Vortex(**kwargs)
    
    def set_parameters_Vortex(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('VORTEX', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = params.get('period', 14)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算VORTEX指标 - 公共接口
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了VORTEX指标的Data_frame
        """
        result = self._calculate_baseindicator(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        核心计算逻辑，实现抽象方法
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了VORTEX指标的Data_frame
        """
        return self._calculate_vortex(data, **kwargs)
    
    def calculate_Vortex(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算VORTEX指标 - 向后兼容方法
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了VORTEX指标的Data_frame
        """
        return self.calculate(data, **kwargs)
    
    def _calculate_vortex(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算VORTEX指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了VORTEX指标的Data_frame
        """
        df = data.copy()
        
        # 确保数据有足够的长度
        if len(df) < self.period + 1:
            logger.warning(f"数据长度({len(df)})不足，需要至少{self.period + 1}条数据")
            df['VI_PLUS'] = np.nan
            df['VI_MINUS'] = np.nan
            df['VORTEX_DIFF'] = np.nan
            df['VORTEX_RATIO'] = np.nan
            return df

        # 计算真实范围 (True Range)
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算正向和负向涡流运动
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        
        # 正向涡流运动 (Positive Vortex Movement)
        df['vm_plus'] = abs(df['high'] - df['prev_low'])
        
        # 负向涡流运动 (Negative Vortex Movement)
        df['vm_minus'] = abs(df['low'] - df['prev_high'])
        
        # 计算指定周期内的累计值
        df['sum_vm_plus'] = df['vm_plus'].rolling(window=self.period).sum()
        df['sum_vm_minus'] = df['vm_minus'].rolling(window=self.period).sum()
        df['sum_tr'] = df['true_range'].rolling(window=self.period).sum()
        
        # 计算涡流指标
        df['VI_PLUS'] = df['sum_vm_plus'] / df['sum_tr']
        df['VI_MINUS'] = df['sum_vm_minus'] / df['sum_tr']
        
        # 计算涡流指标的差值和比率
        df['VORTEX_DIFF'] = df['VI_PLUS'] - df['VI_MINUS']
        df['VORTEX_RATIO'] = df['VI_PLUS'] / (df['VI_MINUS'] + 1e-8)  # 避免除零
        
        # 计算涡流指标的强度
        df['VORTEX_STRENGTH'] = abs(df['VORTEX_DIFF'])
        
        # 计算涡流指标的趋势
        df['VORTEX_TREND'] = np.where(df['VI_PLUS'] > df['VI_MINUS'], 1, 
                                     np.where(df['VI_PLUS'] < df['VI_MINUS'], -1, 0))
        
        # 计算涡流指标的变化率
        df['VI_PLUS_CHANGE'] = df['VI_PLUS'].pct_change() * 100
        df['VI_MINUS_CHANGE'] = df['VI_MINUS'].pct_change() * 100
        
        # 计算涡流指标的波动率
        df['VORTEX_VOLATILITY'] = df['VORTEX_DIFF'].rolling(window=10).std()
        
        # 清理中间计算列
        df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'prev_high', 'prev_low', 
                'vm_plus', 'vm_minus', 'sum_vm_plus', 'sum_vm_minus', 'sum_tr'], 
                axis=1, inplace=True)
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（VORTEX指标特定逻辑）
        df = self._apply_vortex_signal_logic(df)

        return df

    def _apply_vortex_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用VORTEX指标特定的信号生成逻辑
        基于VI+和VI-的交叉以及强度生成信号
        """
        try:
            # 获取VORTEX值
            if 'VI_PLUS' not in df.columns or 'VI_MINUS' not in df.columns:
                # 如果没有VORTEX值，使用默认信号
                return df

            vi_plus = df['VI_PLUS']
            vi_minus = df['VI_MINUS']
            vortex_strength = df['VORTEX_STRENGTH']
            vortex_trend = df['VORTEX_TREND']

            # VORTEX信号生成逻辑：
            # BUY: VI+ 上穿 VI- 且强度足够
            # SELL: VI- 上穿 VI+ 且强度足够
            # HOLD: 交叉信号不明确或强度不足

            # 计算交叉信号
            vi_plus_cross_up = (vi_plus > vi_minus) & (vi_plus.shift(1) <= vi_minus.shift(1))
            vi_minus_cross_up = (vi_minus > vi_plus) & (vi_minus.shift(1) <= vi_plus.shift(1))
            
            # 强度过滤 (避免弱信号)
            strong_signal = vortex_strength > vortex_strength.rolling(window=10).mean()
            
            # 趋势确认
            uptrend_confirmed = (vortex_trend == 1) & (vortex_trend.shift(1) != 1)
            downtrend_confirmed = (vortex_trend == -1) & (vortex_trend.shift(1) != -1)
            
            # 持续趋势信号
            sustained_uptrend = (vortex_trend == 1) & (vortex_trend.shift(1) == 1) & strong_signal
            sustained_downtrend = (vortex_trend == -1) & (vortex_trend.shift(1) == -1) & strong_signal

            # 生成信号
            df.loc[:, 'buy_signal'] = (vi_plus_cross_up & strong_signal) | uptrend_confirmed | sustained_uptrend
            df.loc[:, 'sell_signal'] = (vi_minus_cross_up & strong_signal) | downtrend_confirmed | sustained_downtrend
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"VORTEX信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df
    
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """实现抽象方法"""
        return self.calculate_raw_score_Vortex(data, **kwargs)
    
    def calculate_raw_score_Vortex(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算VORTEX原始评分
        
        基于VORTEX指标的技术分析特点进行评分：
        1. 趋势方向评分 (40%)
        2. 交叉信号评分 (30%)
        3. 强度评分 (20%)
        4. 持续性评分 (10%)
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取VORTEX数据
        vi_plus = self._result['VI_PLUS']
        vi_minus = self._result['VI_MINUS']
        vortex_diff = self._result['VORTEX_DIFF']
        vortex_strength = self._result['VORTEX_STRENGTH']
        vortex_trend = self._result['VORTEX_TREND']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 1. 趋势方向评分 (40%)
        # 基于VI+和VI-的相对位置
        trend_score = pd.Series(0.0, index=data.index)
        
        # 强烈上升趋势
        strong_uptrend = (vi_plus > vi_minus) & (vortex_diff > 0.1)
        trend_score = np.where(strong_uptrend, 15, trend_score)
        
        # 中等上升趋势
        moderate_uptrend = (vi_plus > vi_minus) & (vortex_diff > 0.05) & (vortex_diff <= 0.1)
        trend_score = np.where(moderate_uptrend, 10, trend_score)
        
        # 弱上升趋势
        weak_uptrend = (vi_plus > vi_minus) & (vortex_diff > 0) & (vortex_diff <= 0.05)
        trend_score = np.where(weak_uptrend, 5, trend_score)
        
        # 强烈下降趋势
        strong_downtrend = (vi_minus > vi_plus) & (vortex_diff < -0.1)
        trend_score = np.where(strong_downtrend, -15, trend_score)
        
        # 中等下降趋势
        moderate_downtrend = (vi_minus > vi_plus) & (vortex_diff < -0.05) & (vortex_diff >= -0.1)
        trend_score = np.where(moderate_downtrend, -10, trend_score)
        
        # 弱下降趋势
        weak_downtrend = (vi_minus > vi_plus) & (vortex_diff < 0) & (vortex_diff >= -0.05)
        trend_score = np.where(weak_downtrend, -5, trend_score)
        
        scores += trend_score * 0.4
        
        # 2. 交叉信号评分 (30%)
        # 基于VI+和VI-的交叉
        cross_score = pd.Series(0.0, index=data.index)
        
        # 金叉信号 (VI+ 上穿 VI-)
        golden_cross = (vi_plus > vi_minus) & (vi_plus.shift(1) <= vi_minus.shift(1))
        cross_score = np.where(golden_cross, 12, cross_score)
        
        # 死叉信号 (VI- 上穿 VI+)
        death_cross = (vi_minus > vi_plus) & (vi_minus.shift(1) <= vi_plus.shift(1))
        cross_score = np.where(death_cross, -12, cross_score)
        
        # 趋势持续信号
        if len(vortex_trend.dropna()) > 0:
            trend_continuation = (vortex_trend == vortex_trend.shift(1)) & (abs(vortex_trend) == 1)
            cross_score += np.where(trend_continuation, vortex_trend * 3, 0)
        
        scores += cross_score * 0.3
        
        # 3. 强度评分 (20%)
        # 基于涡流指标的强度
        strength_score = pd.Series(0.0, index=data.index)
        
        if len(vortex_strength.dropna()) > 0:
            # 计算强度的相对位置
            strength_mean = vortex_strength.rolling(window=20).mean()
            strength_std = vortex_strength.rolling(window=20).std()
            
            # 标准化强度
            strength_normalized = (vortex_strength - strength_mean) / (strength_std + 1e-8)
            
            # 很强
            strength_score = np.where(strength_normalized > 2, 10, strength_score)
            # 强
            strength_score = np.where((strength_normalized > 1) & (strength_normalized <= 2), 6, strength_score)
            # 中等
            strength_score = np.where((strength_normalized > 0.5) & (strength_normalized <= 1), 3, strength_score)
            # 弱
            strength_score = np.where((strength_normalized >= -0.5) & (strength_normalized <= 0.5), 0, strength_score)
            # 很弱
            strength_score = np.where(strength_normalized < -0.5, -3, strength_score)
        
        scores += strength_score * 0.2
        
        # 4. 持续性评分 (10%)
        # 基于趋势的持续性
        persistence_score = pd.Series(0.0, index=data.index)
        
        if len(vortex_trend.dropna()) > 0:
            # 计算趋势持续天数
            trend_changes = vortex_trend != vortex_trend.shift(1)
            trend_groups = trend_changes.cumsum()
            trend_duration = vortex_trend.groupby(trend_groups).cumcount() + 1
            
            # 长期趋势加分
            persistence_score = np.where(trend_duration >= 5, 5, persistence_score)
            persistence_score = np.where((trend_duration >= 3) & (trend_duration < 5), 3, persistence_score)
            persistence_score = np.where((trend_duration >= 2) & (trend_duration < 3), 1, persistence_score)
        
        scores += persistence_score * 0.1
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """实现抽象方法"""
        # 转换patterns参数适配原方法
        patterns_df = pd.DataFrame() if isinstance(patterns, list) else patterns
        return self.calculate_confidence_Vortex(score, patterns_df, signals)
    
    def calculate_confidence_Vortex(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if self._result is None:
            return 0.5
            
        # 基于VORTEX指标的明确性计算置信度
        vi_plus = self._result['VI_PLUS'].dropna()
        vi_minus = self._result['VI_MINUS'].dropna()
        vortex_strength = self._result['VORTEX_STRENGTH'].dropna()
        
        if len(vi_plus) == 0 or len(vi_minus) == 0:
            return 0.5
        
        # 计算最近的VORTEX值
        recent_diff = abs(vi_plus.iloc[-1] - vi_minus.iloc[-1]) if len(vi_plus) > 0 else 0
        recent_strength = vortex_strength.iloc[-1] if len(vortex_strength) > 0 else 0
        
        # VI+和VI-的分离度
        separation = 0
        if recent_diff > 0.15:
            separation = 0.3
        elif recent_diff > 0.1:
            separation = 0.2
        elif recent_diff > 0.05:
            separation = 0.15
        elif recent_diff > 0.02:
            separation = 0.1
        
        # 强度一致性
        strength_consistency = 0
        if len(vortex_strength) >= 5:
            recent_strength_trend = vortex_strength.iloc[-5:].diff().dropna()
            if len(recent_strength_trend) > 0:
                # 如果强度趋势一致，提高置信度
                positive_changes = len(recent_strength_trend[recent_strength_trend > 0])
                negative_changes = len(recent_strength_trend[recent_strength_trend < 0])
                if positive_changes >= 3 or negative_changes >= 3:
                    strength_consistency = 0.2
                elif positive_changes >= 2 or negative_changes >= 2:
                    strength_consistency = 0.1
        
        # 指标稳定性
        stability = 0
        if len(vi_plus) >= 10:
            vi_plus_volatility = vi_plus.iloc[-10:].std()
            vi_minus_volatility = vi_minus.iloc[-10:].std()
            avg_volatility = (vi_plus_volatility + vi_minus_volatility) / 2
            
            if avg_volatility < 0.05:
                stability = 0.15
            elif avg_volatility < 0.1:
                stability = 0.1
            elif avg_volatility < 0.2:
                stability = 0.05
        
        base_confidence = 0.25 + separation + strength_consistency + stability
        return min(max(base_confidence, 0.2), 0.9)
    
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """实现抽象方法"""
        return self.get_patterns_Vortex(data, **kwargs)
    
    def get_patterns_Vortex(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取VORTEX相关形态"""
        if not self.has_result():
            self.calculate(data, **kwargs)
            
        if self._result is None:
            return pd.DataFrame(index=data.index)
            
        patterns = pd.DataFrame(index=data.index)
        
        vi_plus = self._result['VI_PLUS']
        vi_minus = self._result['VI_MINUS']
        vortex_diff = self._result['VORTEX_DIFF']
        vortex_strength = self._result['VORTEX_STRENGTH']
        vortex_trend = self._result['VORTEX_TREND']
        
        # 基本形态
        patterns['VI_PLUS_ABOVE'] = vi_plus > vi_minus
        patterns['VI_MINUS_ABOVE'] = vi_minus > vi_plus
        patterns['VI_EQUAL'] = abs(vi_plus - vi_minus) < 0.01
        
        # 交叉形态
        patterns['VI_GOLDEN_CROSS'] = (vi_plus > vi_minus) & (vi_plus.shift(1) <= vi_minus.shift(1))
        patterns['VI_DEATH_CROSS'] = (vi_minus > vi_plus) & (vi_minus.shift(1) <= vi_plus.shift(1))
        
        # 强度形态
        if len(vortex_strength.dropna()) > 0:
            strength_mean = vortex_strength.rolling(window=20).mean()
            patterns['VORTEX_STRONG'] = vortex_strength > strength_mean * 1.5
            patterns['VORTEX_WEAK'] = vortex_strength < strength_mean * 0.5
            patterns['VORTEX_NORMAL'] = ~(patterns['VORTEX_STRONG'] | patterns['VORTEX_WEAK'])
        
        # 趋势形态
        if len(vortex_trend.dropna()) > 0:
            patterns['VORTEX_UPTREND'] = vortex_trend == 1
            patterns['VORTEX_DOWNTREND'] = vortex_trend == -1
            patterns['VORTEX_SIDEWAYS'] = vortex_trend == 0
            
            # 趋势持续性
            trend_changes = vortex_trend != vortex_trend.shift(1)
            trend_groups = trend_changes.cumsum()
            trend_duration = vortex_trend.groupby(trend_groups).cumcount() + 1
            
            patterns['VORTEX_TREND_STARTING'] = trend_duration == 1
            patterns['VORTEX_TREND_CONTINUING'] = trend_duration > 1
            patterns['VORTEX_LONG_TREND'] = trend_duration >= 5
        
        # 差值形态
        patterns['VORTEX_STRONG_BULL'] = vortex_diff > 0.1
        patterns['VORTEX_MODERATE_BULL'] = (vortex_diff > 0.05) & (vortex_diff <= 0.1)
        patterns['VORTEX_WEAK_BULL'] = (vortex_diff > 0) & (vortex_diff <= 0.05)
        patterns['VORTEX_STRONG_BEAR'] = vortex_diff < -0.1
        patterns['VORTEX_MODERATE_BEAR'] = (vortex_diff < -0.05) & (vortex_diff >= -0.1)
        patterns['VORTEX_WEAK_BEAR'] = (vortex_diff < 0) & (vortex_diff >= -0.05)
        patterns['VORTEX_NEUTRAL'] = abs(vortex_diff) <= 0.02
        
        # 极值形态
        vi_plus_high = vi_plus.rolling(window=20).max()
        vi_plus_low = vi_plus.rolling(window=20).min()
        vi_minus_high = vi_minus.rolling(window=20).max()
        vi_minus_low = vi_minus.rolling(window=20).min()
        
        patterns['VI_PLUS_NEW_HIGH'] = vi_plus >= vi_plus_high
        patterns['VI_PLUS_NEW_LOW'] = vi_plus <= vi_plus_low
        patterns['VI_MINUS_NEW_HIGH'] = vi_minus >= vi_minus_high
        patterns['VI_MINUS_NEW_LOW'] = vi_minus <= vi_minus_low
        
        return patterns
