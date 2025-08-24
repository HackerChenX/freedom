#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PIVOT_POINTS指标 - 金融级标准实现
枢轴点（Pivot Points）是技术分析中用于确定潜在支撑和阻力位的重要工具

金融级核心特点:
1. 真实数学计算：严格按照经典枢轴点公式计算，绝不使用模拟逻辑
2. 完整功能架构：计算+评分+形态识别+信号生成+架构兼容
3. 架构完美兼容：遵循六层架构分层+核心原则+依赖注入
4. 性能优化考虑：缓存+异常处理+边界条件+监控+企业级
5. 国际金融标准：算法精度+数值稳定性+边界处理+容错机制

经典枢轴点算法:
PP (Pivot Point) = (High + Low + Close) / 3
R1 (Resistance 1) = 2*PP - Low
S1 (Support 1) = 2*PP - High  
R2 (Resistance 2) = PP + (High - Low)
S2 (Support 2) = PP - (High - Low)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from functools import wraps
import time

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def financial_grade_exception_handler(reraise: bool = True, default_return=None):
    """金融级异常处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"PIVOT_POINTS方法 {func.__name__} 执行失败: {e}")
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def financial_grade_performance_monitor(threshold_seconds: float = 2.0):
    """金融级性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > threshold_seconds:
                logger.warning(f"PIVOT_POINTS方法 {func.__name__} 执行时间过长: {execution_time:.2f}秒")
            
            return result
        return wrapper
    return decorator


class PivotPoints(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    PIVOT_POINTS (枢轴点) 指标 - 金融级标准实现
    
    金融级核心特点:
    1. 真实数学计算：PP = (H + L + C) / 3，严格经典公式
    2. 完整支撑阻力体系：R1/R2阻力位，S1/S2支撑位
    3. 架构完美兼容：遵循六层架构分层+核心原则  
    4. 性能优化考虑：缓存+异常处理+边界条件+监控
    5. 企业级质量：代码规范+文档完整+可维护性+扩展性
    
    技术指标含义:
    - PP (Pivot Point): 枢轴点，价格的中心平衡点
    - R1/R2: 第一、第二阻力位，价格上涨的潜在阻力
    - S1/S2: 第一、第二支撑位，价格下跌的潜在支撑
    - 枢轴点用于识别关键价格水平和交易机会
    
    核心算法: 经典枢轴点计算公式
    参数: 无（基于前一日OHLC数据计算）
    """
    
    def __init__(self, **kwargs):
        """
        初始化PIVOT_POINTS指标 - 金融级标准
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "PIVOT_POINTS"
        self.description = "枢轴点指标，金融级标准实现"
        self.indicator_type = "PIVOT_POINTS"
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
        self._result = None
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_pivot_points()
        
        # 应用用户参数
        self.set_parameters_Indicator_Base_Indicator(**kwargs)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算PIVOT_POINTS指标 - 公共接口
        
        Args:
            data: 包含OHLC数据的DataFrame
            
        Returns:
            添加了PIVOT_POINTS指标的DataFrame
        """
        result = self._calculate_baseindicator(data, **kwargs)
        self._result = result
        return result
    
    @financial_grade_performance_monitor(threshold_seconds=2.0)
    @financial_grade_exception_handler(reraise=True)
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        核心计算逻辑，实现抽象方法
        
        Args:
            data: 包含OHLC数据的DataFrame
            
        Returns:
            添加了PIVOT_POINTS指标的DataFrame
        """
        return self._calculate_pivot_points_financial_grade(data, **kwargs)
    
    def _calculate_pivot_points_financial_grade(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        金融级PIVOT_POINTS指标计算
        
        实现真实的枢轴点算法：
        PP = (High + Low + Close) / 3
        R1 = 2*PP - Low, S1 = 2*PP - High
        R2 = PP + (High - Low), S2 = PP - (High - Low)
        """
        df = data.copy()
        
        # 确保数据有足够长度
        if len(df) < 2:
            logger.warning("数据长度不足，无法计算PIVOT_POINTS指标，需要至少2行数据")
            self._add_default_pivot_columns(df)
            return df
        
        # 验证必需列
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                logger.error(f"PIVOT_POINTS: 缺少必需列 {col}")
                self._add_default_pivot_columns(df)
                return df
        
        # 获取OHLC数据
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 验证OHLC数据
        if high.isna().all() or low.isna().all() or close.isna().all():
            logger.warning("PIVOT_POINTS: OHLC数据包含大量空值")
            self._add_default_pivot_columns(df)
            return df
        
        # 金融级真实算法实现
        try:
            # 枢轴点计算基于前一日的高低收盘价
            # PP = (High[t-1] + Low[t-1] + Close[t-1]) / 3
            prev_high = high.shift(1)
            prev_low = low.shift(1)
            prev_close = close.shift(1)
            
            # 计算枢轴点 (Pivot Point)
            pivot_point = (prev_high + prev_low + prev_close) / 3
            df['PP'] = pivot_point
            
            # 计算第一阻力位和支撑位
            # R1 = 2*PP - Low[t-1]
            # S1 = 2*PP - High[t-1]
            resistance_1 = 2 * pivot_point - prev_low
            support_1 = 2 * pivot_point - prev_high
            
            df['R1'] = resistance_1
            df['S1'] = support_1
            
            # 计算第二阻力位和支撑位
            # R2 = PP + (High[t-1] - Low[t-1])
            # S2 = PP - (High[t-1] - Low[t-1])
            daily_range = prev_high - prev_low
            resistance_2 = pivot_point + daily_range
            support_2 = pivot_point - daily_range
            
            df['R2'] = resistance_2
            df['S2'] = support_2
            
            # 计算枢轴点中点值（用于额外分析）
            df['PP_MID_R1'] = (pivot_point + resistance_1) / 2  # PP到R1的中点
            df['PP_MID_S1'] = (pivot_point + support_1) / 2     # PP到S1的中点
            
            # 计算价格相对于枢轴点的位置
            df['PRICE_TO_PP'] = (close - pivot_point) / pivot_point * 100  # 价格相对PP的百分比
            
            # 计算枢轴点强度指标
            df['PP_STRENGTH'] = abs(daily_range) / pivot_point * 100  # 波动强度
            
            # 计算枢轴点趋势
            df['PP_TREND'] = np.where(close > pivot_point, 1,
                                     np.where(close < pivot_point, -1, 0))
            
            logger.debug("PIVOT_POINTS: 金融级计算完成")
            
        except Exception as e:
            logger.error(f"PIVOT_POINTS: 金融级计算失败: {e}")
            self._add_default_pivot_columns(df)
            return df
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)
        
        # 应用PIVOT_POINTS特定的信号生成逻辑
        df = self._apply_pivot_points_signal_logic_financial_grade(df)
        
        return df
    
    def _add_default_pivot_columns(self, df: pd.DataFrame):
        """添加默认的枢轴点列"""
        df['PP'] = np.nan
        df['R1'] = np.nan
        df['S1'] = np.nan
        df['R2'] = np.nan
        df['S2'] = np.nan
        df['PP_MID_R1'] = np.nan
        df['PP_MID_S1'] = np.nan
        df['PRICE_TO_PP'] = np.nan
        df['PP_STRENGTH'] = np.nan
        df['PP_TREND'] = 0
    
    def _apply_pivot_points_signal_logic_financial_grade(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用PIVOT_POINTS指标特定的信号生成逻辑 - 金融级标准
        基于价格与枢轴点、支撑阻力位的关系生成买卖信号
        """
        try:
            # 获取PIVOT_POINTS值
            required_cols = ['PP', 'R1', 'S1', 'R2', 'S2', 'close']
            if not all(col in df.columns for col in required_cols):
                return df
            
            pp = df['PP']
            r1 = df['R1']
            s1 = df['S1']
            r2 = df['R2']
            s2 = df['S2']
            close = df['close']
            pp_trend = df['PP_TREND']
            
            # PIVOT_POINTS信号生成逻辑：
            # BUY: 1) 价格从支撑位反弹 2) 突破阻力位继续上涨 3) 价格回调到PP获得支撑
            # SELL: 1) 价格从阻力位回落 2) 跌破支撑位继续下跌 3) 价格反弹到PP遇到阻力
            
            # 计算价格突破和反弹信号
            price_above_r1 = close > r1
            price_below_s1 = close < s1
            price_above_pp = close > pp
            price_below_pp = close < pp
            
            # 计算前一日价格位置（用于判断突破）
            prev_price_below_r1 = close.shift(1) <= r1
            prev_price_above_s1 = close.shift(1) >= s1
            prev_price_below_pp = close.shift(1) <= pp
            prev_price_above_pp = close.shift(1) >= pp
            
            # 买入信号条件
            # 1. 突破R1阻力位
            breakout_r1 = price_above_r1 & prev_price_below_r1
            
            # 2. 从S1支撑位反弹
            bounce_from_s1 = (close > s1) & (close.shift(1) <= s1) & (close.shift(2) <= s1)
            
            # 3. 价格回调到PP获得支撑（上升趋势中）
            pp_support = price_above_pp & prev_price_below_pp & (pp_trend == 1)
            
            # 4. 突破PP向上（下降趋势转为上升）
            breakout_pp_up = price_above_pp & prev_price_below_pp
            
            # 卖出信号条件
            # 1. 跌破S1支撑位
            breakdown_s1 = price_below_s1 & prev_price_above_s1
            
            # 2. 从R1阻力位回落
            reject_at_r1 = (close < r1) & (close.shift(1) >= r1) & (close.shift(2) >= r1)
            
            # 3. 价格反弹到PP遇到阻力（下降趋势中）
            pp_resistance = price_below_pp & prev_price_above_pp & (pp_trend == -1)
            
            # 4. 跌破PP向下（上升趋势转为下降）
            breakdown_pp_down = price_below_pp & prev_price_above_pp
            
            # 生成最终信号
            df.loc[:, 'buy_signal'] = breakout_r1 | bounce_from_s1 | pp_support | breakout_pp_up
            df.loc[:, 'sell_signal'] = breakdown_s1 | reject_at_r1 | pp_resistance | breakdown_pp_down
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])
            
            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)
            
        except Exception as e:
            logger.warning(f"PIVOT_POINTS信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True
        
        return df
    
    @financial_grade_exception_handler(reraise=True)
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """实现抽象方法"""
        return self.calculate_raw_score_PIVOT_POINTS_financial_grade(data, **kwargs)
    
    def calculate_raw_score_PIVOT_POINTS_financial_grade(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        金融级PIVOT_POINTS原始评分计算
        
        基于PIVOT_POINTS指标的技术分析特点进行评分：
        1. 枢轴点准确性评分 (35%)
        2. 支撑阻力有效性评分 (30%)
        3. 价格位置评分 (20%)
        4. 趋势一致性评分 (15%)
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取PIVOT_POINTS数据
        pp = self._result['PP']
        r1 = self._result['R1']
        s1 = self._result['S1']
        r2 = self._result['R2']
        s2 = self._result['S2']
        price_to_pp = self._result['PRICE_TO_PP']
        pp_strength = self._result['PP_STRENGTH']
        pp_trend = self._result['PP_TREND']
        close = data['close']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 1. 枢轴点准确性评分 (35%)
        accuracy_score = pd.Series(0.0, index=data.index)
        
        # 价格在合理范围内的枢轴点获得高分
        reasonable_pp = (pp > 0) & (pp.notna())
        accuracy_score = np.where(reasonable_pp, 15, accuracy_score)
        
        # 支撑阻力位逻辑正确性
        logical_levels = (r2 > r1) & (r1 > pp) & (pp > s1) & (s1 > s2)
        accuracy_score = np.where(logical_levels, accuracy_score + 20, accuracy_score + 5)
        
        scores += accuracy_score * 0.35
        
        # 2. 支撑阻力有效性评分 (30%)
        effectiveness_score = pd.Series(0.0, index=data.index)
        
        # 价格在支撑阻力位附近的反应
        near_r1 = abs(close - r1) / r1 < 0.02  # 价格接近R1
        near_s1 = abs(close - s1) / s1 < 0.02  # 价格接近S1
        near_pp = abs(close - pp) / pp < 0.015  # 价格接近PP
        
        effectiveness_score = np.where(near_r1 | near_s1 | near_pp, 20, effectiveness_score)
        
        # 价格突破后的持续性
        if len(close) >= 3:
            price_momentum = close.pct_change().rolling(window=3).mean()
            strong_momentum = abs(price_momentum) > 0.01
            effectiveness_score = np.where(strong_momentum, effectiveness_score + 10, effectiveness_score + 5)
        
        scores += effectiveness_score * 0.30
        
        # 3. 价格位置评分 (20%)
        position_score = pd.Series(0.0, index=data.index)
        
        if price_to_pp.notna().any():
            # 价格偏离枢轴点的程度（适度偏离更好）
            moderate_deviation = (abs(price_to_pp) >= 1) & (abs(price_to_pp) <= 5)
            position_score = np.where(moderate_deviation, 15, position_score)
            
            # 极端偏离减分
            extreme_deviation = abs(price_to_pp) > 10
            position_score = np.where(extreme_deviation, -5, position_score)
            
            # 价格在关键位之间移动
            between_levels = ((close > s1) & (close < r1)) | ((close > pp) & (close < r1))
            position_score = np.where(between_levels, position_score + 10, position_score + 5)
        
        scores += position_score * 0.20
        
        # 4. 趋势一致性评分 (15%)
        trend_score = pd.Series(0.0, index=data.index)
        
        # 趋势方向的持续性
        if len(pp_trend) >= 5:
            trend_consistency = abs(pp_trend.rolling(window=5).mean())
            strong_trend = trend_consistency >= 0.6
            trend_score = np.where(strong_trend, 15, trend_score)
            
            # 趋势变化的及时性
            trend_changes = (pp_trend != pp_trend.shift(1)) & (pp_trend != 0)
            timely_change = trend_changes & (abs(price_to_pp) < 3)
            trend_score = np.where(timely_change, trend_score + 10, trend_score + 5)
        
        scores += trend_score * 0.15
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    @financial_grade_exception_handler(reraise=False, default_return=0.7)
    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """实现抽象方法"""
        if self._result is None:
            return 0.7
        
        # 基于PIVOT_POINTS指标的可靠性计算置信度
        pp = self._result['PP'].dropna()
        pp_trend = self._result['PP_TREND'].dropna()
        pp_strength = self._result['PP_STRENGTH'].dropna()
        
        if len(pp) == 0:
            return 0.7
        
        # 计算枢轴点的稳定性
        recent_pp = pp.iloc[-10:] if len(pp) >= 10 else pp
        
        # 枢轴点变化的稳定性
        pp_stability = 0
        if len(recent_pp) >= 5:
            pp_volatility = recent_pp.std() / recent_pp.mean()
            if pp_volatility <= 0.05:  # 枢轴点相对稳定
                pp_stability = 0.3
            elif pp_volatility <= 0.10:
                pp_stability = 0.2
            else:
                pp_stability = 0.1
        
        # 趋势一致性
        trend_consistency = 0
        if len(pp_trend) >= 5:
            recent_trend = pp_trend.iloc[-5:]
            trend_changes = len(recent_trend[recent_trend != recent_trend.shift(1)].dropna())
            if trend_changes <= 1:  # 趋势稳定
                trend_consistency = 0.25
            elif trend_changes <= 2:
                trend_consistency = 0.15
            else:
                trend_consistency = 0.05
        
        # 强度指标的合理性
        strength_reasonableness = 0
        if len(pp_strength) >= 5:
            recent_strength = pp_strength.iloc[-5:]
            avg_strength = recent_strength.mean()
            if 1 <= avg_strength <= 5:  # 合理的波动强度
                strength_reasonableness = 0.2
            elif 0.5 <= avg_strength <= 8:
                strength_reasonableness = 0.15
            else:
                strength_reasonableness = 0.05
        
        base_confidence = 0.3 + pp_stability + trend_consistency + strength_reasonableness
        return min(max(base_confidence, 0.4), 0.95)
    
    @financial_grade_exception_handler(reraise=True)
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """实现抽象方法"""
        return self.get_patterns_PIVOT_POINTS_financial_grade(data, **kwargs)
    
    def get_patterns_PIVOT_POINTS_financial_grade(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """金融级PIVOT_POINTS形态识别"""
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.DataFrame(index=data.index)
        
        patterns = pd.DataFrame(index=data.index)
        
        pp = self._result['PP']
        r1 = self._result['R1']
        s1 = self._result['S1']
        r2 = self._result['R2']
        s2 = self._result['S2']
        close = data['close']
        pp_trend = self._result['PP_TREND']
        
        # 基本价格位置形态
        patterns['PP_ABOVE_PIVOT'] = close > pp
        patterns['PP_BELOW_PIVOT'] = close < pp
        patterns['PP_AT_PIVOT'] = abs(close - pp) / pp < 0.01
        
        # 支撑阻力测试形态
        patterns['PP_TEST_R1'] = (close >= r1 * 0.98) & (close <= r1 * 1.02)
        patterns['PP_TEST_S1'] = (close >= s1 * 0.98) & (close <= s1 * 1.02)
        patterns['PP_TEST_R2'] = (close >= r2 * 0.98) & (close <= r2 * 1.02)
        patterns['PP_TEST_S2'] = (close >= s2 * 0.98) & (close <= s2 * 1.02)
        
        # 突破形态
        patterns['PP_BREAKOUT_R1'] = (close > r1) & (close.shift(1) <= r1)
        patterns['PP_BREAKDOWN_S1'] = (close < s1) & (close.shift(1) >= s1)
        patterns['PP_BREAKOUT_R2'] = (close > r2) & (close.shift(1) <= r2)
        patterns['PP_BREAKDOWN_S2'] = (close < s2) & (close.shift(1) >= s2)
        
        # 反弹形态
        patterns['PP_BOUNCE_S1'] = (close > s1) & (close.shift(1) <= s1) & (close.shift(2) <= s1)
        patterns['PP_REJECT_R1'] = (close < r1) & (close.shift(1) >= r1) & (close.shift(2) >= r1)
        
        # 趋势形态
        patterns['PP_STRONG_UPTREND'] = (pp_trend == 1) & (close > r1)
        patterns['PP_STRONG_DOWNTREND'] = (pp_trend == -1) & (close < s1)
        patterns['PP_RANGE_TRADING'] = (close > s1) & (close < r1) & (pp_trend == 0)
        
        # 极值形态
        if len(close) >= 20:
            close_high = close.rolling(window=20).max()
            close_low = close.rolling(window=20).min()
            
            patterns['PP_NEW_HIGH'] = close >= close_high
            patterns['PP_NEW_LOW'] = close <= close_low
            
            # 多重支撑阻力形态
            patterns['PP_MULTIPLE_SUPPORT'] = (close <= s1 * 1.02) & (close >= s2 * 0.98)
            patterns['PP_MULTIPLE_RESISTANCE'] = (close >= r1 * 0.98) & (close <= r2 * 1.02)
        
        return patterns
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """实现抽象方法"""
        # PIVOT_POINTS指标通常不需要参数，基于前一日OHLC计算
        pass
    
    def _get_default_parameters_pivot_points(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {}  # 枢轴点指标不需要参数
    
    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return self._result is not None and not self._result.empty

    @property
    def minimum_periods(self) -> int:
        """
        PivotPoints指标所需的最少数据周期数
        
        计算逻辑：使用默认值
        
        Returns:
            int: 最少需要的数据周期数
        """
        return 30