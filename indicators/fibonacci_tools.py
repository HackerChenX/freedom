from utils.container import container
#!/usr/bin/env python3
from utils.logger import get_logger
"""
FIBONACCI_TOOLS 指标

斐波那契工具指标 - 基于斐波那契回撤和扩展的技术分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class FibonacciTools(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    FIBONACCI_TOOLS 指标
    
    斐波那契工具指标,基于斐波那契数列的回撤和扩展分析
    主要分析价格在斐波那契关键位的支撑阻力效应
    """
    
    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化FIBONACCI_TOOLS指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "FIBONACCI_TOOLS"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_fibonaccitools()

        # 🔧 Ultra Think修复:设置内部minimum_periods值
        self._minimum_periods = 25  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 应用用户参数
        self.set_parameters_Tools(**kwargs)
    
    def _get_default_parameters_fibonaccitools(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "period": 20,  # 计算周期  # TODO: 将魔法数字提取到配置中
            "swing_period": 10,  # 波动周期
            "fib_levels": [0.236, 0.382, 0.5, 0.618,  # TODO: 将魔法数字提取到配置中 0.786]  # 斐波那契回撤位  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        }
    
    def set_parameters_Tools(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 🔧 Ultra Think修复:简化参数设置,确保参数修改功能正常
        try:
            # 直接设置参数,不依赖验证器
            self.period = kwargs.get('period', 20)  # TODO: 将魔法数字提取到配置中
            self.swing_period = kwargs.get('swing_period', 10)
            self.fib_levels = kwargs.get('fib_levels', [0.236, 0.382, 0.5, 0.618,  # TODO: 将魔法数字提取到配置中 0.786])  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 同步更新minimum_periods
            self._minimum_periods = max(self.period, self.swing_period) + 5  # TODO: 将魔法数字提取到配置中

        except Exception:
            # 如果设置失败,使用默认值
            self.period = 20  # TODO: 将魔法数字提取到配置中
            self.swing_period = 10
            self.fib_levels = [0.236, 0.382, 0.5, 0.618,  # TODO: 将魔法数字提取到配置中 0.786]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            self._minimum_periods = 25  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
    
    def calculate_Tools(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算FIBONACCI_TOOLS指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了FIBONACCI_TOOLS指标的Data_frame
        """
        result = self._calculate_fibonaccitools(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_fibonaccitools(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算FIBONACCI_TOOLS指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了FIBONACCI_TOOLS指标的Data_frame
        """
        df = data.copy()
        
        # 计算摆动高低点
        swing_high = df['high'].rolling(window=self.swing_period).max()
        swing_low = df['low'].rolling(window=self.swing_period).min()
        
        # 计算斐波那契回撤位
        fib_range = swing_high - swing_low
        
        # 计算各个斐波那契位
        for level in self.fib_levels:
            df[f'fib_{level:.3f}'] = swing_low + fib_range * level
        
        # 计算当前价格相对于斐波那契位的位置
        df['fib_position'] = self._calculate_fib_position(df)
        
        # 计算支撑阻力强度
        df['support_strength'] = self._calculate_support_strength(df)
        df['resistance_strength'] = self._calculate_resistance_strength(df)
        
        # 计算突破信号
        df['fib_breakout'] = self._calculate_breakout_signal(df)
        
        # 计算回撤确认信号
        df['fib_retracement'] = self._calculate_retracement_signal(df)
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
    
    def _calculate_fib_position(self, df: pd.DataFrame) -> pd.Series:
        """计算当前价格在斐波那契位中的位置"""
        position = pd.Series(0.0, index=df.index)
        
        close = df['close']
        
        for i, level in enumerate(self.fib_levels):
            fib_level = df[f'fib_{level:.3f}']
            
            # 价格在该斐波那契位之上
            above_level = close > fib_level
            position[above_level] = level
        
        return position
    
    def _calculate_support_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算支撑强度"""
        support = pd.Series(0.0, index=df.index)
        
        close = df['close']
        low = df['low']
        
        for level in self.fib_levels:
            fib_level = df[f'fib_{level:.3f}']
            
            # 价格接近斐波那契位的支撑
            near_support = (close <= fib_level * 1.02) & (close >= fib_level * 0.98)  # TODO: 将魔法数字提取到配置中
            
            # 最低价触及但未跌破斐波那契位
            touch_support = (low <= fib_level * 1.01) & (close > fib_level)
            
            support[near_support] += level * 10
            support[touch_support] += level * 15  # TODO: 将魔法数字提取到配置中
        
        return support
    
    def _calculate_resistance_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算阻力强度"""
        resistance = pd.Series(0.0, index=df.index)
        
        close = df['close']
        high = df['high']
        
        for level in self.fib_levels:
            fib_level = df[f'fib_{level:.3f}']
            
            # 价格接近斐波那契位的阻力
            near_resistance = (close >= fib_level * 0.98) & (close <= fib_level * 1.02)  # TODO: 将魔法数字提取到配置中
            
            # 最高价触及但未突破斐波那契位
            touch_resistance = (high >= fib_level * 0.99) & (close < fib_level)  # TODO: 将魔法数字提取到配置中
            
            resistance[near_resistance] += level * 10
            resistance[touch_resistance] += level * 15  # TODO: 将魔法数字提取到配置中
        
        return resistance
    
    def _calculate_breakout_signal(self, df: pd.DataFrame) -> pd.Series:
        """计算突破信号"""
        breakout = pd.Series(0.0, index=df.index)
        
        close = df['close']
        volume = df.get('volume', pd.Series(1.0, index=df.index))
        avg_volume = volume.rolling(window=self.period).mean()
        
        for level in self.fib_levels:
            fib_level = df[f'fib_{level:.3f}']
            
            # 向上突破斐波那契位
            upward_break = (close > fib_level) & (close.shift(1) <= fib_level)
            
            # 成交量放大确认突破
            volume_confirm = volume > avg_volume * 1.2
            
            breakout[upward_break & volume_confirm] += level * 20  # TODO: 将魔法数字提取到配置中
        
        return breakout
    
    def _calculate_retracement_signal(self, df: pd.DataFrame) -> pd.Series:
        """计算回撤信号"""
        retracement = pd.Series(0.0, index=df.index)
        
        close = df['close']
        
        # 计算价格趋势
        trend = close.rolling(window=self.period).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        for level in self.fib_levels:
            fib_level = df[f'fib_{level:.3f}']
            
            # 上升趋势中的回撤到斐波那契位
            uptrend_retracement = (trend > 0) & (close <= fib_level * 1.01) & (close >= fib_level * 0.99)  # TODO: 将魔法数字提取到配置中
            
            # 下降趋势中的反弹到斐波那契位
            downtrend_retracement = (trend < 0) & (close >= fib_level * 0.99) & (close <= fib_level * 1.01)  # TODO: 将魔法数字提取到配置中
            
            retracement[uptrend_retracement] += level * 15  # TODO: 将魔法数字提取到配置中
            retracement[downtrend_retracement] -= level * 10
        
        return retracement
    
    def calculate_raw_score_Tools(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分
        
        基于斐波那契分析的综合评分:
        - 支撑阻力强度(40%权重)  # TODO: 将魔法数字提取到配置中
        - 突破信号(30%权重)  # TODO: 将魔法数字提取到配置中
        - 回撤信号(20%权重)  # TODO: 将魔法数字提取到配置中
        - 位置评分(10%权重)
        """
        # 🔧 Ultra Think修复:移除has_result检查,直接计算
        # if not self.has_result():
        #     self.calculate_Tools(data, **kwargs)
        
        result = self._result
        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
        
        # 1. 支撑阻力评分(40%权重)  # TODO: 将魔法数字提取到配置中
        support_strength = result.get('support_strength', pd.Series(0.0, index=data.index))
        resistance_strength = result.get('resistance_strength', pd.Series(0.0, index=data.index))
        
        # 支撑强度加分,阻力强度减分
        support_score = np.clip(support_strength / 5, 0, 20)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        resistance_score = np.clip(resistance_strength / 5, 0, 20)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        score += support_score * 0.4  # TODO: 将魔法数字提取到配置中
        score -= resistance_score * 0.4  # TODO: 将魔法数字提取到配置中
        
        # 2. 突破信号评分(30%权重)  # TODO: 将魔法数字提取到配置中
        breakout_signal = result.get('fib_breakout', pd.Series(0.0, index=data.index))
        breakout_score = np.clip(breakout_signal / 10, 0, 30)  # TODO: 将魔法数字提取到配置中
        score += breakout_score * 0.3  # TODO: 将魔法数字提取到配置中
        
        # 3. 回撤信号评分(20%权重)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        retracement_signal = result.get('fib_retracement', pd.Series(0.0, index=data.index))
        retracement_score = np.clip(retracement_signal / 8,  # TODO: 将魔法数字提取到配置中 -20, 20)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        score += retracement_score * 0.2
        
        # 4. 位置评分(10%权重)  # TODO: 将魔法数字提取到配置中
        fib_position = result.get('fib_position', pd.Series(0.5, index=data.index))  # TODO: 将魔法数字提取到配置中
        
        # 在黄金分割位(0.618)附近加分  # TODO: 将魔法数字提取到配置中
        golden_ratio_bonus = np.where(
            np.abs(fib_position - 0.618) < 0.05,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            10,
            0
        )
        score += golden_ratio_bonus * 0.1
        
        # 在关键回撤位(0.382, 0.5)附近加分  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        key_level_bonus = np.where(
            (np.abs(fib_position - 0.382) < 0.03) | (np.abs(fib_position - 0.5) < 0.03),  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            8,  # TODO: 将魔法数字提取到配置中
            0
        )
        score += key_level_bonus * 0.1
        
        # 确保评分在0-100范围内
        score = np.clip(score, 0, 100)
        
        return score
    
    def calculate_confidence_Tools(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.5  # TODO: 将魔法数字提取到配置中
        
        # 基于评分分布和信号强度计算置信度
        avg_score = score.mean()
        score_std = score.std()
        
        # 评分越高,置信度越高
        score_confidence = min(avg_score / 100, 1.0)
        
        # 评分稳定性越高,置信度越高
        stability_confidence = max(0.3, 1.0 - score_std / 50)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 综合置信度
        confidence = (score_confidence * 0.7 + stability_confidence * 0.3)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        return max(0.3, min(0.95, confidence))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
    
    def get_patterns_Tools(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        # 🔧 Ultra Think修复:移除has_result检查,直接计算
        # if not self.has_result():
        #     self.calculate_Tools(data, **kwargs)
        
        result = self._result
        patterns = []
        
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 斐波那契位置形态
            fib_position = last_row.get('fib_position', 0.5)  # TODO: 将魔法数字提取到配置中
            
            if fib_position >= 0.618:  # TODO: 将魔法数字提取到配置中
                patterns.append("价格位于黄金分割位之上")
            elif fib_position >= 0.5:  # TODO: 将魔法数字提取到配置中
                patterns.append("价格位于50%回撤位之上")
            elif fib_position >= 0.382:  # TODO: 将魔法数字提取到配置中
                patterns.append("价格位于38.2%回撤位之上")
            elif fib_position >= 0.236:  # TODO: 将魔法数字提取到配置中
                patterns.append("价格位于23.6%回撤位之上")  # TODO: 将魔法数字提取到配置中
            else:
                patterns.append("价格位于主要斐波那契位之下")
            
            # 支撑阻力形态
            support_strength = last_row.get('support_strength', 0)
            resistance_strength = last_row.get('resistance_strength', 0)
            
            if support_strength > 5:  # TODO: 将魔法数字提取到配置中
                patterns.append("斐波那契强支撑位")
            elif support_strength > 2:
                patterns.append("斐波那契支撑位")
            
            if resistance_strength > 5:  # TODO: 将魔法数字提取到配置中
                patterns.append("斐波那契强阻力位")
            elif resistance_strength > 2:
                patterns.append("斐波那契阻力位")
            
            # 突破形态
            breakout_signal = last_row.get('fib_breakout', 0)
            if breakout_signal > 10:
                patterns.append("斐波那契位向上突破")
            elif breakout_signal > 5:  # TODO: 将魔法数字提取到配置中
                patterns.append("斐波那契位突破尝试")
            
            # 回撤形态
            retracement_signal = last_row.get('fib_retracement', 0)
            if retracement_signal > 8:  # TODO: 将魔法数字提取到配置中
                patterns.append("上升趋势斐波那契回撤")
            elif retracement_signal < -5:  # TODO: 将魔法数字提取到配置中
                patterns.append("下降趋势斐波那契反弹")
        
        return pd.DataFrame({'patterns': [patterns]}, index=[data.index[-1]] if len(data) > 0 else [])

    # 🔧 Ultra Think修复:实现BaseIndicator要求的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """实现BaseIndicator要求的_calculate_baseindicator方法"""
        return self._calculate_fibonaccitools(data, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """实现BaseIndicator要求的置信度计算方法"""
        return self.calculate_confidence_Tools(score, patterns, signals)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """实现BaseIndicator要求的原始评分计算方法"""
        return self.calculate_raw_score_Tools(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """实现BaseIndicator要求的形态获取方法"""
        return self.get_patterns_Tools(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """实现BaseIndicator要求的参数设置方法"""
        return self.set_parameters_Tools(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """实现MinimumPeriodsMixin要求的minimum_periods属性"""
        return getattr(self, '_minimum_periods', 25)  # TODO: 将魔法数字提取到配置中


# 为了向后兼容,创建别名
FIBONACCI_TOOLS = FibonacciTools