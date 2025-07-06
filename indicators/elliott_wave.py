#!/usr/bin/env python3
"""
ELLIOTT_WAVE 指标

艾略特波浪指标 - 基于艾略特波浪理论的波浪形态分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import getLogger

logger = getLogger(__name__)


class ElliottWave(BaseIndicator, PatternSignalMixin):
    """
    ELLIOTT_WAVE 指标
    
    艾略特波浪指标，基于艾略特波浪理论的5-3波浪形态分析
    主要识别推动浪和调整浪的形态特征
    """
    
    def __init__(self, **kwargs):
        """
        初始化ELLIOTT_WAVE指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "ELLIOTT_WAVE"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_elliottwave()
        
        # 应用用户参数
        self.set_parameters_Wave(**kwargs)
    
    def _get_default_parameters_elliottwave(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "period": 20,  # 计算周期
            "min_wave_length": 5,  # 最小波浪长度
            "fibonacci_ratios": [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618],  # 斐波那契比例
            "wave_tolerance": 0.1  # 波浪识别容差
        }
    
    def set_parameters_Wave(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('ELLIOTT_WAVE', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 20)
            self.min_wave_length = params.get('min_wave_length', 5)
            self.fibonacci_ratios = params.get('fibonacci_ratios', [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618])
            self.wave_tolerance = params.get('wave_tolerance', 0.1)
                    
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            self.period = 20
            self.min_wave_length = 5
            self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
            self.wave_tolerance = 0.1
    
    def calculate_Wave(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ELLIOTT_WAVE指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了ELLIOTT_WAVE指标的Data_frame
        """
        result = self._calculate_elliottwave(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_elliottwave(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算ELLIOTT_WAVE指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了ELLIOTT_WAVE指标的Data_frame
        """
        df = data.copy()
        
        # 识别波浪转折点
        df['pivot_high'], df['pivot_low'] = self._identify_pivots(df)
        
        # 计算波浪特征
        df['wave_direction'] = self._calculate_wave_direction(df)
        df['wave_strength'] = self._calculate_wave_strength(df)
        
        # 识别推动浪形态
        df['impulse_wave'] = self._identify_impulse_waves(df)
        
        # 识别调整浪形态
        df['corrective_wave'] = self._identify_corrective_waves(df)
        
        # 计算斐波那契回撤和扩展
        df['fib_retracement'] = self._calculate_fib_retracement(df)
        df['fib_extension'] = self._calculate_fib_extension(df)
        
        # 计算波浪计数
        df['wave_count'] = self._calculate_wave_count(df)
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
    
    def _identify_pivots(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """识别波浪转折点"""
        high = df['high']
        low = df['low']
        
        # 使用滚动窗口识别局部高低点
        pivot_high = pd.Series(False, index=df.index)
        pivot_low = pd.Series(False, index=df.index)
        
        for i in range(self.min_wave_length, len(df) - self.min_wave_length):
            # 检查是否为局部高点
            if high.iloc[i] == high.iloc[i-self.min_wave_length:i+self.min_wave_length+1].max():
                pivot_high.iloc[i] = True
            
            # 检查是否为局部低点
            if low.iloc[i] == low.iloc[i-self.min_wave_length:i+self.min_wave_length+1].min():
                pivot_low.iloc[i] = True
        
        return pivot_high, pivot_low
    
    def _calculate_wave_direction(self, df: pd.DataFrame) -> pd.Series:
        """计算波浪方向"""
        close = df['close']
        direction = pd.Series(0.0, index=df.index)
        
        # 计算短期和长期趋势
        short_ma = close.rolling(window=self.min_wave_length).mean()
        long_ma = close.rolling(window=self.period).mean()
        
        # 上升波浪
        up_wave = (close > short_ma) & (short_ma > long_ma)
        direction[up_wave] = 1.0
        
        # 下降波浪
        down_wave = (close < short_ma) & (short_ma < long_ma)
        direction[down_wave] = -1.0
        
        return direction
    
    def _calculate_wave_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算波浪强度"""
        close = df['close']
        volume = df.get('volume', pd.Series(1.0, index=df.index))
        
        # 计算价格变化率
        price_change = close.pct_change().abs()
        
        # 计算成交量变化率
        volume_change = volume.pct_change().abs()
        
        # 综合强度 = 价格变化 * 成交量变化
        strength = price_change * volume_change
        
        # 标准化强度
        strength_normalized = strength.rolling(window=self.period).rank(pct=True)
        
        return strength_normalized.fillna(0.5)
    
    def _identify_impulse_waves(self, df: pd.DataFrame) -> pd.Series:
        """识别推动浪（5浪结构）"""
        impulse_signal = pd.Series(0.0, index=df.index)
        
        pivot_high = df['pivot_high']
        pivot_low = df['pivot_low']
        wave_direction = df['wave_direction']
        wave_strength = df['wave_strength']
        
        # 寻找5浪结构
        for i in range(len(df) - self.period):
            window_end = i + self.period
            
            # 获取窗口内的转折点
            window_highs = pivot_high.iloc[i:window_end]
            window_lows = pivot_low.iloc[i:window_end]
            
            # 统计转折点数量
            high_count = window_highs.sum()
            low_count = window_lows.sum()
            
            # 推动浪特征：5个主要转折点，强势方向
            if high_count >= 2 and low_count >= 2:
                avg_direction = wave_direction.iloc[i:window_end].mean()
                avg_strength = wave_strength.iloc[i:window_end].mean()
                
                # 强势上升推动浪
                if avg_direction > 0.3 and avg_strength > 0.6:
                    impulse_signal.iloc[window_end-1] = avg_strength * 20
                
                # 强势下降推动浪
                elif avg_direction < -0.3 and avg_strength > 0.6:
                    impulse_signal.iloc[window_end-1] = avg_strength * 15
        
        return impulse_signal
    
    def _identify_corrective_waves(self, df: pd.DataFrame) -> pd.Series:
        """识别调整浪（3浪结构）"""
        corrective_signal = pd.Series(0.0, index=df.index)
        
        wave_direction = df['wave_direction']
        wave_strength = df['wave_strength']
        close = df['close']
        
        # 寻找3浪调整结构
        for i in range(len(df) - self.min_wave_length * 3):
            window_end = i + self.min_wave_length * 3
            
            # 获取窗口内的数据
            window_direction = wave_direction.iloc[i:window_end]
            window_strength = wave_strength.iloc[i:window_end]
            window_close = close.iloc[i:window_end]
            
            # 调整浪特征：方向变化，强度适中，价格回撤
            direction_changes = (window_direction.diff().abs() > 0.5).sum()
            avg_strength = window_strength.mean()
            
            # 价格回撤比例
            price_start = window_close.iloc[0]
            price_end = window_close.iloc[-1]
            retracement = abs(price_end - price_start) / price_start
            
            # 调整浪条件
            if direction_changes >= 2 and 0.3 < avg_strength < 0.7 and retracement < 0.1:
                corrective_signal.iloc[window_end-1] = avg_strength * 10
        
        return corrective_signal
    
    def _calculate_fib_retracement(self, df: pd.DataFrame) -> pd.Series:
        """计算斐波那契回撤"""
        fib_signal = pd.Series(0.0, index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # 计算近期高低点
        recent_high = high.rolling(window=self.period).max()
        recent_low = low.rolling(window=self.period).min()
        
        # 计算当前价格在高低点之间的位置
        price_range = recent_high - recent_low
        price_position = (close - recent_low) / price_range
        
        # 检查是否接近斐波那契回撤位
        for ratio in self.fibonacci_ratios:
            if ratio <= 1.0:  # 只考虑回撤比例
                near_fib = np.abs(price_position - ratio) < self.wave_tolerance
                fib_signal[near_fib] += 10 * ratio
        
        return fib_signal
    
    def _calculate_fib_extension(self, df: pd.DataFrame) -> pd.Series:
        """计算斐波那契扩展"""
        fib_ext_signal = pd.Series(0.0, index=df.index)
        
        close = df['close']
        
        # 计算波浪长度
        wave_length = close.rolling(window=self.min_wave_length).apply(
            lambda x: x.iloc[-1] - x.iloc[0]
        )
        
        # 计算前一波浪长度
        prev_wave_length = wave_length.shift(self.min_wave_length)
        
        # 计算当前波浪与前一波浪的比例
        wave_ratio = wave_length / prev_wave_length
        
        # 检查是否接近斐波那契扩展比例
        for ratio in self.fibonacci_ratios:
            if ratio >= 1.0:  # 只考虑扩展比例
                near_fib_ext = np.abs(wave_ratio - ratio) < self.wave_tolerance
                fib_ext_signal[near_fib_ext] += 8 * (ratio - 1.0)
        
        return fib_ext_signal.fillna(0.0)
    
    def _calculate_wave_count(self, df: pd.DataFrame) -> pd.Series:
        """计算波浪计数"""
        wave_count = pd.Series(0.0, index=df.index)
        
        pivot_high = df['pivot_high']
        pivot_low = df['pivot_low']
        
        # 计算累计转折点数量
        total_pivots = (pivot_high | pivot_low).cumsum()
        
        # 计算波浪完成度
        wave_position = total_pivots % 8  # 8浪循环（5推动+3调整）
        
        # 在波浪循环的关键位置给予信号
        key_positions = [3, 5, 8]  # 第3浪、第5浪、调整浪结束
        
        for pos in key_positions:
            at_key_position = wave_position == pos
            wave_count[at_key_position] += 10
        
        return wave_count
    
    def calculate_raw_score_Wave(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分
        
        基于艾略特波浪分析的综合评分：
        - 推动浪信号（30%权重）
        - 调整浪信号（20%权重）
        - 斐波那契回撤（25%权重）
        - 斐波那契扩展（15%权重）
        - 波浪计数（10%权重）
        """
        if not self.has_result():
            self.calculate_Wave(data, **kwargs)
        
        result = self._result
        score = pd.Series(50.0, index=data.index)
        
        # 1. 推动浪信号评分（30%权重）
        impulse_wave = result.get('impulse_wave', pd.Series(0.0, index=data.index))
        impulse_score = np.clip(impulse_wave / 2, 0, 25)
        score += impulse_score * 0.3
        
        # 2. 调整浪信号评分（20%权重）
        corrective_wave = result.get('corrective_wave', pd.Series(0.0, index=data.index))
        corrective_score = np.clip(corrective_wave / 1.5, 0, 15)
        score += corrective_score * 0.2
        
        # 3. 斐波那契回撤评分（25%权重）
        fib_retracement = result.get('fib_retracement', pd.Series(0.0, index=data.index))
        fib_ret_score = np.clip(fib_retracement / 2, 0, 20)
        score += fib_ret_score * 0.25
        
        # 4. 斐波那契扩展评分（15%权重）
        fib_extension = result.get('fib_extension', pd.Series(0.0, index=data.index))
        fib_ext_score = np.clip(fib_extension / 2, 0, 15)
        score += fib_ext_score * 0.15
        
        # 5. 波浪计数评分（10%权重）
        wave_count = result.get('wave_count', pd.Series(0.0, index=data.index))
        count_score = np.clip(wave_count / 2, 0, 10)
        score += count_score * 0.1
        
        # 波浪方向加成
        wave_direction = result.get('wave_direction', pd.Series(0.0, index=data.index))
        wave_strength = result.get('wave_strength', pd.Series(0.5, index=data.index))
        
        # 强势上升波浪加分
        strong_up_wave = (wave_direction > 0.5) & (wave_strength > 0.7)
        score[strong_up_wave] += 8
        
        # 强势下降波浪适度加分（因为也是交易机会）
        strong_down_wave = (wave_direction < -0.5) & (wave_strength > 0.7)
        score[strong_down_wave] += 3
        
        # 确保评分在0-100范围内
        score = np.clip(score, 0, 100)
        
        return score
    
    def calculate_confidence_Wave(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.5
        
        # 基于评分分布和波浪理论的完整性计算置信度
        avg_score = score.mean()
        score_std = score.std()
        
        # 评分越高，置信度越高
        score_confidence = min(avg_score / 100, 1.0)
        
        # 评分稳定性越高，置信度越高
        stability_confidence = max(0.3, 1.0 - score_std / 50)
        
        # 波浪理论强调形态完整性
        pattern_confidence = 0.7  # 基础形态置信度
        
        # 综合置信度
        confidence = (score_confidence * 0.4 + stability_confidence * 0.3 + pattern_confidence * 0.3)
        
        return max(0.3, min(0.95, confidence))
    
    def get_patterns_Wave(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        if not self.has_result():
            self.calculate_Wave(data, **kwargs)
        
        result = self._result
        patterns = []
        
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 推动浪形态
            impulse_wave = last_row.get('impulse_wave', 0)
            if impulse_wave > 15:
                patterns.append("艾略特强势推动浪")
            elif impulse_wave > 8:
                patterns.append("艾略特推动浪")
            
            # 调整浪形态
            corrective_wave = last_row.get('corrective_wave', 0)
            if corrective_wave > 8:
                patterns.append("艾略特调整浪")
            elif corrective_wave > 5:
                patterns.append("艾略特弱调整浪")
            
            # 斐波那契形态
            fib_retracement = last_row.get('fib_retracement', 0)
            if fib_retracement > 15:
                patterns.append("艾略特斐波那契强回撤位")
            elif fib_retracement > 8:
                patterns.append("艾略特斐波那契回撤位")
            
            fib_extension = last_row.get('fib_extension', 0)
            if fib_extension > 10:
                patterns.append("艾略特斐波那契扩展位")
            elif fib_extension > 5:
                patterns.append("艾略特斐波那契扩展信号")
            
            # 波浪计数形态
            wave_count = last_row.get('wave_count', 0)
            if wave_count > 8:
                patterns.append("艾略特波浪关键转折点")
            elif wave_count > 5:
                patterns.append("艾略特波浪计数信号")
            
            # 波浪方向形态
            wave_direction = last_row.get('wave_direction', 0)
            wave_strength = last_row.get('wave_strength', 0.5)
            
            if wave_direction > 0.5 and wave_strength > 0.7:
                patterns.append("艾略特强势上升波浪")
            elif wave_direction > 0.3 and wave_strength > 0.5:
                patterns.append("艾略特上升波浪")
            elif wave_direction < -0.5 and wave_strength > 0.7:
                patterns.append("艾略特强势下降波浪")
            elif wave_direction < -0.3 and wave_strength > 0.5:
                patterns.append("艾略特下降波浪")
            
            # 转折点形态
            pivot_high = last_row.get('pivot_high', False)
            pivot_low = last_row.get('pivot_low', False)
            
            if pivot_high:
                patterns.append("艾略特波浪高点")
            if pivot_low:
                patterns.append("艾略特波浪低点")
        
        return pd.DataFrame({'patterns': [patterns]}, index=[data.index[-1]] if len(data) > 0 else [])


# 为了向后兼容，创建别名
elliott_wave = ELLIOTT_WAVE