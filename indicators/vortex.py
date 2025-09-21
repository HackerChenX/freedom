#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VORTEX (Vortex Indicator) 涡流指标

涡流指标用于识别趋势的开始和结束,通过比较正向和负向价格运动来衡量趋势强度.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

from utils.container import container
from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from db.sql_manager import SQLManager, QueryType
from utils.indicator_parameter_validator import IndicatorParameterValidator

logger = get_logger(__name__)


class Vortex(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    VORTEX (Vortex Indicator) 涡流指标
    
    涡流指标通过计算正向和负向价格运动的比率来识别趋势.
    VI+ > VI- 表示上升趋势
    VI- > VI+ 表示下降趋势
    """
    
    def __init__(self, **kwargs):
        """
        初始化VORTEX指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__(name="VORTEX", **kwargs)
        
        # 依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
        
        self.description = "涡流指标，用于识别趋势的开始和结束，通过比较正向和负向价格运动来衡量趋势强度"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_vortex()
        
        # 应用用户参数
        self.set_parameters_Vortex(**kwargs)
    
    def _get_default_parameters_vortex(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中
    
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
        except Exception as e:
            logger.error(f"错误: {e}")
            return pd.DataFrame()
    
    def validate_parameters(self, **kwargs):
        """验证参数"""
        try:
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('VORTEX', params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass
                
        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            pass
        
        # 设置参数
        self.period = params.get('period', 14)  # TODO: 将魔法数字提取到配置中
    
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算VORTEX指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了VORTEX指标的Data_frame
        """
        result = self._calculate_baseindicator(data, **kwargs)
        self._result = result
        return result
    
    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=None)
    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        获取VORTEX指标信号

        Args:
            data: 包含价格数据的DataFrame
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 包含signal, score, confidence的字典
        """
        try:
            # 计算VORTEX指标
            result = self.calculate(data, **kwargs)
            
            if result.empty or len(result) == 0:
                return {'signal': 'HOLD', 'score': 50.0, 'confidence': 0.5}
            
            # 获取最新的VORTEX值
            latest = result.iloc[-1]
            vi_plus = latest.get('vortex_vi_plus', 1.0)
            vi_minus = latest.get('vortex_vi_minus', 1.0)
            vortex_diff = latest.get('vortex_diff', 0.0)
            vortex_strength = latest.get('vortex_strength', 50.0)
            
            # 初始化信号
            signal = "HOLD"
            score = 50.0
            confidence = 0.5
            
            # VORTEX信号逻辑
            if vi_plus > vi_minus:  # 上升趋势
                if vortex_diff > 0.1:  # 强烈上升趋势
                    signal = "BUY"
                    score = min(85.0, 50.0 + vortex_diff * 200)
                    confidence = min(0.9, 0.6 + vortex_diff * 2)
                elif vortex_diff > 0.05:  # 中等上升趋势
                    signal = "BUY"
                    score = min(75.0, 50.0 + vortex_diff * 150)
                    confidence = min(0.8, 0.5 + vortex_diff * 3)
                else:  # 弱上升趋势
                    signal = "HOLD"
                    score = 60.0
                    confidence = 0.6
            elif vi_minus > vi_plus:  # 下降趋势
                if vortex_diff < -0.1:  # 强烈下降趋势
                    signal = "SELL"
                    score = max(15.0, 50.0 + vortex_diff * 200)
                    confidence = min(0.9, 0.6 + abs(vortex_diff) * 2)
                elif vortex_diff < -0.05:  # 中等下降趋势
                    signal = "SELL"
                    score = max(25.0, 50.0 + vortex_diff * 150)
                    confidence = min(0.8, 0.5 + abs(vortex_diff) * 3)
                else:  # 弱下降趋势
                    signal = "HOLD"
                    score = 40.0
                    confidence = 0.6
            else:  # 趋势不明确
                signal = "HOLD"
                score = 50.0
                confidence = 0.5
            
            # 结合强度调整置信度
            if vortex_strength > 70:  # 高强度
                confidence = min(0.9, confidence + 0.1)
            elif vortex_strength < 30:  # 低强度
                confidence = max(0.3, confidence - 0.1)
            
            return {
                'signal': signal,
                'score': float(score),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.warning(f"VORTEX信号获取失败: {e}")
            return {'signal': 'HOLD', 'score': 50.0, 'confidence': 0.5}
    
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        核心计算逻辑,实现抽象方法
        
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
            logger.warning(f"数据长度({len(df)})不足,需要至少{self.period + 1}条数据")
            df['vortex_vi_plus'] = np.nan
            df['vortex_vi_minus'] = np.nan
            df['vortex_diff'] = np.nan
            df['vortex_ratio'] = np.nan
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
        df['vortex_vi_plus'] = df['sum_vm_plus'] / df['sum_tr']
        df['vortex_vi_minus'] = df['sum_vm_minus'] / df['sum_tr']
        
        # 计算涡流指标的差值和比率
        df['vortex_diff'] = df['vortex_vi_plus'] - df['vortex_vi_minus']
        df['vortex_ratio'] = df['vortex_vi_plus'] / (df['vortex_vi_minus'] + 1e-8)  # 避免除零
        
        # 计算涡流指标的强度
        df['vortex_strength'] = abs(df['vortex_diff'])
        
        # 计算涡流指标的趋势
        df['vortex_trend'] = np.where(df['vortex_vi_plus'] > df['vortex_vi_minus'], 1, 
                                     np.where(df['vortex_vi_plus'] < df['vortex_vi_minus'], -1, 0))
        
        # 计算涡流指标的变化率
        df['vortex_vi_plus_change'] = df['vortex_vi_plus'].pct_change() * 100
        df['vortex_vi_minus_change'] = df['vortex_vi_minus'].pct_change() * 100
        
        # 计算涡流指标的波动率
        df['vortex_volatility'] = df['vortex_diff'].rolling(window=10).std()
        
        # 清理中间计算列
        df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'prev_high', 'prev_low', 
                'vm_plus', 'vm_minus', 'sum_vm_plus', 'sum_vm_minus', 'sum_tr'], 
                axis=1, inplace=True)
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑(VORTEX指标特定逻辑)
        df = self._apply_vortex_signal_logic(df)

        return df

    def _apply_vortex_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用VORTEX指标特定的信号生成逻辑
        基于VI+和VI-的交叉以及强度生成信号
        """
        try:
            # 获取VORTEX值
            if 'vortex_vi_plus' not in df.columns or 'vortex_vi_minus' not in df.columns:
                # 如果没有VORTEX值,使用默认信号
                return df

            vi_plus = df['vortex_vi_plus']
            vi_minus = df['vortex_vi_minus']
            vortex_strength = df['vortex_strength']
            vortex_trend = df['vortex_trend']

            # VORTEX信号生成逻辑:
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
            # 如果出错,使用默认信号
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
        
        基于VORTEX指标的技术分析特点进行评分:
        1. 趋势方向评分 (40%)  # TODO: 将魔法数字提取到配置中
        2. 交叉信号评分 (30%)  # TODO: 将魔法数字提取到配置中
        3. 强度评分 (20%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        4. 持续性评分 (10%)  # TODO: 将魔法数字提取到配置中
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
        
        # 获取VORTEX数据
        vi_plus = self._result['vortex_vi_plus']
        vi_minus = self._result['vortex_vi_minus']
        vortex_diff = self._result['vortex_diff']
        vortex_strength = self._result['vortex_strength']
        vortex_trend = self._result['vortex_trend']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
        
        # 1. 趋势方向评分 (40%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 基于VI+和VI-的相对位置
        trend_score = pd.Series(0.0, index=data.index)
        
        # 强烈上升趋势
        strong_uptrend = (vi_plus > vi_minus) & (vortex_diff > 0.1)
        trend_score = np.where(strong_uptrend, 15, trend_score)  # TODO: 将魔法数字提取到配置中
        
        # 中等上升趋势
        moderate_uptrend = (vi_plus > vi_minus) & (vortex_diff > 0.05) & (vortex_diff <= 0.1)  # TODO: 将魔法数字提取到配置中
        trend_score = np.where(moderate_uptrend, 10, trend_score)
        
        # 弱上升趋势
        weak_uptrend = (vi_plus > vi_minus) & (vortex_diff > 0) & (vortex_diff <= 0.05)  # TODO: 将魔法数字提取到配置中
        trend_score = np.where(weak_uptrend, 5, trend_score)  # TODO: 将魔法数字提取到配置中
        
        # 强烈下降趋势
        strong_downtrend = (vi_minus > vi_plus) & (vortex_diff < -0.1)
        trend_score = np.where(strong_downtrend, -15, trend_score)  # TODO: 将魔法数字提取到配置中
        
        # 中等下降趋势
        moderate_downtrend = (vi_minus > vi_plus) & (vortex_diff < -0.05) & (vortex_diff >= -0.1)  # TODO: 将魔法数字提取到配置中
        trend_score = np.where(moderate_downtrend, -10, trend_score)
        
        # 弱下降趋势
        weak_downtrend = (vi_minus > vi_plus) & (vortex_diff < 0) & (vortex_diff >= -0.05)  # TODO: 将魔法数字提取到配置中
        trend_score = np.where(weak_downtrend, -5, trend_score)  # TODO: 将魔法数字提取到配置中
        
        scores += trend_score * 0.4  # TODO: 将魔法数字提取到配置中
        
        # 2. 交叉信号评分 (30%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 基于VI+和VI-的交叉
        cross_score = pd.Series(0.0, index=data.index)
        
        # 金叉信号 (VI+ 上穿 VI-)
        golden_cross = (vi_plus > vi_minus) & (vi_plus.shift(1) <= vi_minus.shift(1))
        cross_score = np.where(golden_cross, 12, cross_score)  # TODO: 将魔法数字提取到配置中
        
        # 死叉信号 (VI- 上穿 VI+)
        death_cross = (vi_minus > vi_plus) & (vi_minus.shift(1) <= vi_plus.shift(1))
        cross_score = np.where(death_cross, -12, cross_score)  # TODO: 将魔法数字提取到配置中
        
        # 趋势持续信号
        if len(vortex_trend.dropna()) > 0:
            trend_continuation = (vortex_trend == vortex_trend.shift(1)) & (abs(vortex_trend) == 1)
            cross_score += np.where(trend_continuation, vortex_trend * 3, 0)  # TODO: 将魔法数字提取到配置中
        
        scores += cross_score * 0.3  # TODO: 将魔法数字提取到配置中
        
        # 3. 强度评分 (20%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 基于涡流指标的强度
        strength_score = pd.Series(0.0, index=data.index)
        
        if len(vortex_strength.dropna()) > 0:
            # 计算强度的相对位置
            strength_mean = vortex_strength.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            strength_std = vortex_strength.rolling(window=20).std()  # TODO: 将魔法数字提取到配置中
            
            # 标准化强度
            strength_normalized = (vortex_strength - strength_mean) / (strength_std + 1e-8)  # TODO: 将魔法数字提取到配置中
            
            # 很强
            strength_score = np.where(strength_normalized > 2, 10, strength_score)
            # 强
            strength_score = np.where((strength_normalized > 1) & (strength_normalized <= 2), 6, strength_score)  # TODO: 将魔法数字提取到配置中
            # 中等
            strength_score = np.where((strength_normalized > 0.5) & (strength_normalized <= 1), 3, strength_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 弱
            strength_score = np.where((strength_normalized >= -0.5) & (strength_normalized <= 0.5), 0, strength_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 很弱
            strength_score = np.where(strength_normalized < -0.5, -3, strength_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        scores += strength_score * 0.2
        
        # 4. 持续性评分 (10%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 基于趋势的持续性
        persistence_score = pd.Series(0.0, index=data.index)
        
        if len(vortex_trend.dropna()) > 0:
            # 计算趋势持续天数
            trend_changes = vortex_trend != vortex_trend.shift(1)
            trend_groups = trend_changes.cumsum()
            trend_duration = vortex_trend.groupby(trend_groups).cumcount() + 1
            
            # 长期趋势加分
            persistence_score = np.where(trend_duration >= 5, 5, persistence_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            persistence_score = np.where((trend_duration >= 3) & (trend_duration < 5), 3, persistence_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            persistence_score = np.where((trend_duration >= 2) & (trend_duration < 3), 1, persistence_score)  # TODO: 将魔法数字提取到配置中
        
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
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            
        # 基于VORTEX指标的明确性计算置信度
        vi_plus = self._result['VI_PLUS'].dropna()
        vi_minus = self._result['VI_MINUS'].dropna()
        vortex_strength = self._result['VORTEX_STRENGTH'].dropna()
        
        if len(vi_plus) == 0 or len(vi_minus) == 0:
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 计算最近的VORTEX值
        recent_diff = abs(vi_plus.iloc[-1] - vi_minus.iloc[-1]) if len(vi_plus) > 0 else 0
        recent_strength = vortex_strength.iloc[-1] if len(vortex_strength) > 0 else 0
        
        # VI+和VI-的分离度
        separation = 0
        if recent_diff > 0.15:  # TODO: 将魔法数字提取到配置中
            separation = 0.3  # TODO: 将魔法数字提取到配置中
        elif recent_diff > 0.1:
            separation = 0.2
        elif recent_diff > 0.05:  # TODO: 将魔法数字提取到配置中
            separation = 0.15  # TODO: 将魔法数字提取到配置中
        elif recent_diff > 0.02:
            separation = 0.1
        
        # 强度一致性
        strength_consistency = 0
        if len(vortex_strength) >= 5:  # TODO: 将魔法数字提取到配置中
            recent_strength_trend = vortex_strength.iloc[-5:].diff().dropna()  # TODO: 将魔法数字提取到配置中
            if len(recent_strength_trend) > 0:
                # 如果强度趋势一致,提高置信度
                positive_changes = len(recent_strength_trend[recent_strength_trend > 0])
                negative_changes = len(recent_strength_trend[recent_strength_trend < 0])
                if positive_changes >= 3 or negative_changes >= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    strength_consistency = 0.2
                elif positive_changes >= 2 or negative_changes >= 2:
                    strength_consistency = 0.1
        
        # 指标稳定性
        stability = 0
        if len(vi_plus) >= 10:
            vi_plus_volatility = vi_plus.iloc[-10:].std()
            vi_minus_volatility = vi_minus.iloc[-10:].std()
            avg_volatility = (vi_plus_volatility + vi_minus_volatility) / 2
            
            if avg_volatility < 0.05:  # TODO: 将魔法数字提取到配置中
                stability = 0.15  # TODO: 将魔法数字提取到配置中
            elif avg_volatility < 0.1:
                stability = 0.1
            elif avg_volatility < 0.2:
                stability = 0.05  # TODO: 将魔法数字提取到配置中
        
        base_confidence = 0.25 + separation + strength_consistency + stability  # TODO: 将魔法数字提取到配置中
        return min(max(base_confidence, 0.2), 0.9)  # TODO: 将魔法数字提取到配置中
    
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
            strength_mean = vortex_strength.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns['VORTEX_STRONG'] = vortex_strength > strength_mean * 1.5  # TODO: 将魔法数字提取到配置中
            patterns['VORTEX_WEAK'] = vortex_strength < strength_mean * 0.5  # TODO: 将魔法数字提取到配置中
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
            patterns['VORTEX_LONG_TREND'] = trend_duration >= 5  # TODO: 将魔法数字提取到配置中
        
        # 差值形态
        patterns['VORTEX_STRONG_BULL'] = vortex_diff > 0.1
        patterns['VORTEX_MODERATE_BULL'] = (vortex_diff > 0.05) & (vortex_diff <= 0.1)  # TODO: 将魔法数字提取到配置中
        patterns['VORTEX_WEAK_BULL'] = (vortex_diff > 0) & (vortex_diff <= 0.05)  # TODO: 将魔法数字提取到配置中
        patterns['VORTEX_STRONG_BEAR'] = vortex_diff < -0.1
        patterns['VORTEX_MODERATE_BEAR'] = (vortex_diff < -0.05) & (vortex_diff >= -0.1)  # TODO: 将魔法数字提取到配置中
        patterns['VORTEX_WEAK_BEAR'] = (vortex_diff < 0) & (vortex_diff >= -0.05)  # TODO: 将魔法数字提取到配置中
        patterns['VORTEX_NEUTRAL'] = abs(vortex_diff) <= 0.02
        
        # 极值形态
        vi_plus_high = vi_plus.rolling(window=20).max()  # TODO: 将魔法数字提取到配置中
        vi_plus_low = vi_plus.rolling(window=20).min()  # TODO: 将魔法数字提取到配置中
        vi_minus_high = vi_minus.rolling(window=20).max()  # TODO: 将魔法数字提取到配置中
        vi_minus_low = vi_minus.rolling(window=20).min()  # TODO: 将魔法数字提取到配置中
        
        patterns['VI_PLUS_NEW_HIGH'] = vi_plus >= vi_plus_high
        patterns['VI_PLUS_NEW_LOW'] = vi_plus <= vi_plus_low
        patterns['VI_MINUS_NEW_HIGH'] = vi_minus >= vi_minus_high
        patterns['VI_MINUS_NEW_LOW'] = vi_minus <= vi_minus_low
        
        return patterns

    @property
    def minimum_periods(self) -> int:
        """
        Vortex指标所需的最少数据周期数
        
        计算逻辑:使用默认值
        
        Returns:
            int: 最少需要的数据周期数
        """
        return 20  # TODO: 将魔法数字提取到配置中