#!/usr/bin/env python3
"""
AROON 指标

阿隆指标 - 趋势强度和方向识别指标
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class AROON(BaseIndicator, PatternSignalMixin):
    """
    AROON 指标
    
    阿隆指标用于识别趋势的强度和方向
    """
    
    def __init__(self, **kwargs):
        """
        初始化AROON指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "AROON"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters()
        
        # 应用用户参数
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
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
            is_valid, errors = validator.validate_indicator_parameters('AROON', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)
                    
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            self.period = 14
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算AROON指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了AROON指标的DataFrame
        """
        result = self._calculate(data, **kwargs)
        self._result = result
        return result
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算AROON指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了AROON指标的DataFrame
        """
        df = data.copy()
        
        # 获取高价和低价
        high = df['high']
        low = df['low']
        
        # 计算Aroon Up: (period - 最高价距今天数) / period * 100
        aroon_up = high.rolling(window=self.period).apply(
            lambda x: (self.period - x.argmax()) / self.period * 100
        )
        
        # 计算Aroon Down: (period - 最低价距今天数) / period * 100
        aroon_down = low.rolling(window=self.period).apply(
            lambda x: (self.period - x.argmin()) / self.period * 100
        )
        
        # 计算Aroon震荡器
        aroon_oscillator = aroon_up - aroon_down
        
        # 保存计算结果
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        df['aroon_oscillator'] = aroon_oscillator
        
        # 为了向后兼容，也保留AROON_VALUE列
        df['AROON_VALUE'] = aroon_oscillator
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（AROON指标特定逻辑）
        df = self._apply_aroon_signal_logic(df)

        return df

    def _apply_aroon_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用AROON指标特定的信号生成逻辑
        基于AROON UP/DOWN交叉和强度生成信号
        """
        try:
            # 获取AROON值
            if 'aroon_up' not in df.columns or 'aroon_down' not in df.columns:
                # 如果没有AROON值，使用默认信号
                return df

            aroon_up = df['aroon_up']
            aroon_down = df['aroon_down']
            aroon_osc = df['aroon_oscillator']

            # AROON信号生成逻辑：
            # BUY: Aroon Up > 70 且 Aroon Up > Aroon Down 且 Aroon Up 上升
            # SELL: Aroon Down > 70 且 Aroon Down > Aroon Up 且 Aroon Down 上升
            # HOLD: 其他情况

            # 强趋势条件
            aroon_up_strong = aroon_up > 70
            aroon_down_strong = aroon_down > 70
            
            # 交叉条件
            aroon_up_dominant = aroon_up > aroon_down
            aroon_down_dominant = aroon_down > aroon_up
            
            # 趋势条件
            aroon_up_rising = aroon_up > aroon_up.shift(1)
            aroon_down_rising = aroon_down > aroon_down.shift(1)

            # 生成信号
            df.loc[:, 'buy_signal'] = aroon_up_strong & aroon_up_dominant & aroon_up_rising
            df.loc[:, 'sell_signal'] = aroon_down_strong & aroon_down_dominant & aroon_down_rising
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"AROON信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算AROON原始评分
        
        基于AROON指标的技术分析特点进行评分：
        1. AROON UP强度评分 (40%)
        2. AROON DOWN分析 (30%)
        3. AROON震荡器分析 (20%)
        4. 趋势确认 (10%)
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取AROON数据
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        aroon_osc = self._result['aroon_oscillator']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 1. AROON UP强度评分 (40%)
        # AROON UP > 70: 强上升趋势 (+20分)
        # AROON UP 50-70: 中等上升趋势 (+10分)
        # AROON UP < 30: 弱势 (-10分)
        aroon_up_score = pd.Series(0.0, index=data.index)
        aroon_up_score = np.where(aroon_up > 70, 20, aroon_up_score)
        aroon_up_score = np.where((aroon_up >= 50) & (aroon_up <= 70), 10, aroon_up_score)
        aroon_up_score = np.where(aroon_up < 30, -10, aroon_up_score)
        scores += aroon_up_score * 0.4
        
        # 2. AROON DOWN分析 (30%)
        # AROON DOWN > 70: 强下降趋势 (-15分)
        # AROON DOWN < 30: 上升趋势确认 (+15分)
        aroon_down_score = pd.Series(0.0, index=data.index)
        aroon_down_score = np.where(aroon_down > 70, -15, aroon_down_score)
        aroon_down_score = np.where(aroon_down < 30, 15, aroon_down_score)
        scores += aroon_down_score * 0.3
        
        # 3. AROON震荡器分析 (20%)
        # 震荡器 > 50: 强上升趋势 (+10分)
        # 震荡器 > 0: 上升趋势 (+5分)
        # 震荡器 < -50: 强下降趋势 (-10分)
        # 震荡器变化趋势
        osc_change = aroon_osc - aroon_osc.shift(1)
        osc_score = pd.Series(0.0, index=data.index)
        osc_score = np.where(aroon_osc > 50, 10, osc_score)
        osc_score = np.where((aroon_osc > 0) & (aroon_osc <= 50), 5, osc_score)
        osc_score = np.where(aroon_osc < -50, -10, osc_score)
        # 震荡器上升趋势加分
        osc_score = np.where(osc_change > 0, osc_score + 3, osc_score)
        scores += osc_score * 0.2
        
        # 4. 趋势确认 (10%)
        # AROON UP和DOWN的差距越大，趋势越明确
        trend_strength = abs(aroon_up - aroon_down)
        trend_score = pd.Series(0.0, index=data.index)
        trend_score = np.where(trend_strength > 50, 8, trend_score)
        trend_score = np.where((trend_strength > 30) & (trend_strength <= 50), 5, trend_score)
        # 如果AROON UP占优势，加分
        trend_score = np.where((aroon_up > aroon_down) & (trend_strength > 30), 
                              trend_score + 2, trend_score)
        scores += trend_score * 0.1
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if self._result is None:
            return 0.5
            
        # 基于AROON指标的明确性计算置信度
        aroon_up = self._result['aroon_up'].dropna()
        aroon_down = self._result['aroon_down'].dropna()
        
        if len(aroon_up) == 0 or len(aroon_down) == 0:
            return 0.5
        
        # 计算最近的AROON值
        recent_up = aroon_up.iloc[-1] if len(aroon_up) > 0 else 50
        recent_down = aroon_down.iloc[-1] if len(aroon_down) > 0 else 50
        
        # 趋势越明确，置信度越高
        trend_clarity = abs(recent_up - recent_down) / 100
        
        # 极端值提高置信度
        extreme_confidence = 0
        if recent_up > 80 or recent_down > 80:
            extreme_confidence = 0.2
        
        base_confidence = 0.4 + trend_clarity * 0.4 + extreme_confidence
        return min(max(base_confidence, 0.3), 0.9)
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取AROON相关形态"""
        if not self.has_result():
            self.calculate(data, **kwargs)
            
        if self._result is None:
            return pd.DataFrame(index=data.index)
            
        patterns = pd.DataFrame(index=data.index)
        
        aroon_up = self._result['aroon_up']
        aroon_down = self._result['aroon_down']
        aroon_osc = self._result['aroon_oscillator']
        
        # 基本形态
        patterns['AROON_UPTREND'] = aroon_up > 70
        patterns['AROON_DOWNTREND'] = aroon_down > 70
        patterns['AROON_CONSOLIDATION'] = (aroon_up < 50) & (aroon_down < 50)
        
        # 交叉形态
        patterns['AROON_BULLISH_CROSS'] = (aroon_up > aroon_down) & (aroon_up.shift(1) <= aroon_down.shift(1))
        patterns['AROON_BEARISH_CROSS'] = (aroon_down > aroon_up) & (aroon_down.shift(1) <= aroon_up.shift(1))
        
        # 极端形态
        patterns['AROON_STRONG_UP'] = aroon_up > 80
        patterns['AROON_STRONG_DOWN'] = aroon_down > 80
        patterns['AROON_WEAK_TREND'] = (aroon_up < 30) & (aroon_down < 30)
        
        return patterns
