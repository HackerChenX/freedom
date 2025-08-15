#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WILLIAMS_R指标 - 国际金融级标准实现
威廉指标（Williams %R）是技术分析中用于确定超买和超卖水平的重要动量振荡器

国际金融级核心特点:
1. 真实数学计算：严格按照经典威廉指标公式计算，绝不使用模拟逻辑
2. 完整功能架构：计算+评分+形态识别+信号生成+架构兼容
3. 架构完美兼容：遵循六层架构分层+核心原则+依赖注入
4. 性能优化考虑：缓存+异常处理+边界条件+监控+企业级
5. 华尔街交易标准：算法精度+数值稳定性+边界处理+微秒级计算速度

经典威廉指标算法:
%R = (Highest High - Close) / (Highest High - Lowest Low) * -100
其中：
- Highest High: N期内最高价
- Lowest Low: N期内最低价  
- Close: 当前收盘价
- 结果范围：-100 到 0
- 超卖水平：通常 <= -80
- 超买水平：通常 >= -20
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from functools import wraps
import time

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def international_financial_exception_handler(reraise: bool = True, default_return=None):
    """国际金融级异常处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"WILLIAMS_R方法 {func.__name__} 执行失败: {e}")
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def international_financial_performance_monitor(threshold_seconds: float = 0.001):  # 微秒级监控
    """国际金融级性能监控装饰器 - 微秒级精度"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > threshold_seconds:
                logger.warning(f"WILLIAMS_R方法 {func.__name__} 执行时间: {execution_time*1000:.2f}毫秒")
            
            return result
        return wrapper
    return decorator


class WilliamsR(BaseIndicator, PatternSignalMixin):
    """
    WILLIAMS_R (威廉指标) 指标 - 国际金融级标准实现
    
    国际金融级核心特点:
    1. 真实数学计算：%R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    2. 完整超买超卖体系：-20超买，-80超卖，精确信号生成
    3. 架构完美兼容：遵循六层架构分层+核心原则  
    4. 性能优化考虑：缓存+异常处理+边界条件+微秒级监控
    5. 华尔街交易级质量：代码规范+文档完整+可维护性+扩展性
    
    技术指标含义:
    - %R: 威廉指标值，-100到0之间的振荡器
    - 超买区域: %R >= -20，价格可能面临回调压力
    - 超卖区域: %R <= -80，价格可能面临反弹机会
    - 中性区域: -80 < %R < -20，价格处于正常波动范围
    - 威廉指标用于识别买卖时机和超买超卖状态
    
    核心算法: 经典威廉指标计算公式
    参数: period (默认14) - 计算周期
    """
    
    def __init__(self, period: int = 14, **kwargs):
        """
        初始化WILLIAMS_R指标 - 国际金融级标准
        
        Args:
            period: 计算周期，默认14
            **kwargs: 其他指标参数
        """
        super().__init__()
        self.name = "WILLIAMS_R"
        self.description = "威廉指标，国际金融级标准实现"
        self.indicator_type = "WILLIAMS_R"
        self.REQUIRED_COLUMNS = ['high', 'low', 'close']
        self._result = None
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_williams_r()
        self._default_parameters.update({
            'period': period,
            'overbought': -20,  # 超买水平
            'oversold': -80,    # 超卖水平
        })
        
        # 应用用户参数
        self.set_parameters_Indicator_Base_Indicator(**kwargs)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算WILLIAMS_R指标 - 公共接口
        
        Args:
            data: 包含HLC数据的DataFrame
            
        Returns:
            添加了WILLIAMS_R指标的DataFrame
        """
        result = self._calculate_baseindicator(data, **kwargs)
        self._result = result
        return result
    
    @international_financial_performance_monitor(threshold_seconds=0.001)
    @international_financial_exception_handler(reraise=True)
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        核心计算逻辑，实现抽象方法
        
        Args:
            data: 包含HLC数据的DataFrame
            
        Returns:
            添加了WILLIAMS_R指标的DataFrame
        """
        return self._calculate_williams_r_international_financial_grade(data, **kwargs)
    
    def _calculate_williams_r_international_financial_grade(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        国际金融级WILLIAMS_R指标计算
        
        实现真实的威廉指标算法：
        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        """
        df = data.copy()
        
        # 获取参数
        period = kwargs.get('period', self._default_parameters.get('period', 14))
        overbought = kwargs.get('overbought', self._default_parameters.get('overbought', -20))
        oversold = kwargs.get('oversold', self._default_parameters.get('oversold', -80))
        
        # 确保数据有足够长度
        if len(df) < period:
            logger.warning(f"数据长度不足，无法计算WILLIAMS_R指标，需要至少{period}行数据")
            self._add_default_williams_r_columns(df)
            return df
        
        # 验证必需列
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                logger.error(f"WILLIAMS_R: 缺少必需列 {col}")
                self._add_default_williams_r_columns(df)
                return df
        
        # 获取HLC数据
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 验证HLC数据
        if high.isna().all() or low.isna().all() or close.isna().all():
            logger.warning("WILLIAMS_R: HLC数据包含大量空值")
            self._add_default_williams_r_columns(df)
            return df
        
        # 国际金融级真实算法实现
        try:
            # 计算滚动窗口内的最高价和最低价
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            # 计算威廉指标 %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
            wr_denominator = highest_high - lowest_low
            
            # 处理除零情况（当最高价等于最低价时）
            williams_r = np.where(
                wr_denominator != 0,
                (highest_high - close) / wr_denominator * -100,
                -50  # 默认中性值
            )
            
            df['WILLIAMS_R'] = williams_r
            
            # 计算威廉指标移动平均线（用于信号平滑）
            df['WR_MA_3'] = pd.Series(williams_r).rolling(window=3).mean()
            df['WR_MA_6'] = pd.Series(williams_r).rolling(window=6).mean()
            
            # 计算威廉指标变化率
            df['WR_CHANGE'] = pd.Series(williams_r).diff()
            df['WR_CHANGE_RATE'] = pd.Series(williams_r).pct_change() * 100
            
            # 计算威廉指标动量指标
            df['WR_MOMENTUM'] = pd.Series(williams_r) - pd.Series(williams_r).shift(3)
            
            # 计算威廉指标振幅（用于衡量波动性）
            df['WR_VOLATILITY'] = pd.Series(williams_r).rolling(window=period).std()
            
            # 计算威廉指标位置（相对于历史区间的位置）
            wr_series = pd.Series(williams_r)
            wr_rolling_min = wr_series.rolling(window=period*2).min()
            wr_rolling_max = wr_series.rolling(window=period*2).max()
            
            df['WR_POSITION'] = np.where(
                (wr_rolling_max - wr_rolling_min) != 0,
                (wr_series - wr_rolling_min) / (wr_rolling_max - wr_rolling_min) * 100,
                50  # 默认中性位置
            )
            
            # 计算威廉指标趋势
            df['WR_TREND'] = np.where(williams_r > williams_r,  # 这里需要修正
                                     np.where(df['WR_MA_3'] > df['WR_MA_6'], 1,
                                             np.where(df['WR_MA_3'] < df['WR_MA_6'], -1, 0)), 0)
            
            # 正确的趋势计算
            df['WR_TREND'] = np.where(df['WR_MA_3'] > df['WR_MA_6'], 1,
                                     np.where(df['WR_MA_3'] < df['WR_MA_6'], -1, 0))
            
            # 计算威廉指标强度
            df['WR_STRENGTH'] = np.abs(williams_r + 50) / 50 * 100  # 相对于中性值-50的强度
            
            logger.debug(f"WILLIAMS_R: 国际金融级计算完成，周期={period}")
            
        except Exception as e:
            logger.error(f"WILLIAMS_R: 国际金融级计算失败: {e}")
            self._add_default_williams_r_columns(df)
            return df
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)
        
        # 应用WILLIAMS_R特定的信号生成逻辑
        df = self._apply_williams_r_signal_logic_international_financial_grade(df, overbought, oversold)
        
        return df
    
    def _add_default_williams_r_columns(self, df: pd.DataFrame):
        """添加默认的威廉指标列"""
        df['WILLIAMS_R'] = np.nan
        df['WR_MA_3'] = np.nan
        df['WR_MA_6'] = np.nan
        df['WR_CHANGE'] = np.nan
        df['WR_CHANGE_RATE'] = np.nan
        df['WR_MOMENTUM'] = np.nan
        df['WR_VOLATILITY'] = np.nan
        df['WR_POSITION'] = np.nan
        df['WR_TREND'] = 0
        df['WR_STRENGTH'] = np.nan
    
    def _apply_williams_r_signal_logic_international_financial_grade(self, df: pd.DataFrame, overbought: float, oversold: float) -> pd.DataFrame:
        """
        应用WILLIAMS_R指标特定的信号生成逻辑 - 国际金融级标准
        基于威廉指标的超买超卖水平和趋势变化生成买卖信号
        """
        try:
            # 获取WILLIAMS_R值
            if 'WILLIAMS_R' not in df.columns:
                return df
            
            wr = df['WILLIAMS_R']
            wr_ma_3 = df.get('WR_MA_3', wr)
            wr_change = df.get('WR_CHANGE', 0)
            wr_trend = df.get('WR_TREND', 0)
            
            # WILLIAMS_R信号生成逻辑：
            # BUY: 1) 从超卖区域反弹 2) 威廉指标向上突破 3) 威廉指标金叉
            # SELL: 1) 从超买区域回落 2) 威廉指标向下突破 3) 威廉指标死叉
            
            # 识别超买超卖区域
            in_oversold = wr <= oversold
            in_overbought = wr >= overbought
            
            # 识别从超卖超买区域的离开
            was_oversold = wr.shift(1) <= oversold
            was_overbought = wr.shift(1) >= overbought
            
            leaving_oversold = was_oversold & (wr > oversold)
            leaving_overbought = was_overbought & (wr < overbought)
            
            # 买入信号条件
            # 1. 从超卖区域反弹
            oversold_reversal = leaving_oversold & (wr_change > 0)
            
            # 2. 威廉指标向上突破-50中轴
            upward_breakout = (wr > -50) & (wr.shift(1) <= -50) & (wr_change > 0)
            
            # 3. 威廉指标快速线上穿慢速线（金叉）
            wr_ma_3_prev = wr_ma_3.shift(1)
            wr_ma_6_curr = df.get('WR_MA_6', wr)
            wr_ma_6_prev = wr_ma_6_curr.shift(1)
            
            golden_cross = (wr_ma_3 > wr_ma_6_curr) & (wr_ma_3_prev <= wr_ma_6_prev)
            
            # 4. 威廉指标强势上升
            strong_upward = (wr_change > 5) & (wr > -70) & (wr_trend == 1)
            
            # 卖出信号条件
            # 1. 从超买区域回落
            overbought_reversal = leaving_overbought & (wr_change < 0)
            
            # 2. 威廉指标向下突破-50中轴
            downward_breakdown = (wr < -50) & (wr.shift(1) >= -50) & (wr_change < 0)
            
            # 3. 威廉指标快速线下穿慢速线（死叉）
            death_cross = (wr_ma_3 < wr_ma_6_curr) & (wr_ma_3_prev >= wr_ma_6_prev)
            
            # 4. 威廉指标强势下降
            strong_downward = (wr_change < -5) & (wr < -30) & (wr_trend == -1)
            
            # 生成最终信号
            df.loc[:, 'buy_signal'] = (oversold_reversal | upward_breakout | 
                                      golden_cross | strong_upward)
            df.loc[:, 'sell_signal'] = (overbought_reversal | downward_breakdown | 
                                       death_cross | strong_downward)
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])
            
            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)
            
        except Exception as e:
            logger.warning(f"WILLIAMS_R信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True
        
        return df
    
    @international_financial_exception_handler(reraise=True)
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """实现抽象方法"""
        return self.calculate_raw_score_WILLIAMS_R_international_financial_grade(data, **kwargs)
    
    def calculate_raw_score_WILLIAMS_R_international_financial_grade(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        国际金融级WILLIAMS_R原始评分计算
        
        基于WILLIAMS_R指标的技术分析特点进行评分：
        1. 超买超卖准确性评分 (35%)
        2. 趋势识别有效性评分 (30%)
        3. 动量变化评分 (20%)
        4. 信号质量评分 (15%)
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取WILLIAMS_R数据
        wr = self._result['WILLIAMS_R']
        wr_ma_3 = self._result.get('WR_MA_3', wr)
        wr_change = self._result.get('WR_CHANGE', 0)
        wr_momentum = self._result.get('WR_MOMENTUM', 0)
        wr_volatility = self._result.get('WR_VOLATILITY', 0)
        wr_trend = self._result.get('WR_TREND', 0)
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 1. 超买超卖准确性评分 (35%)
        overbought_oversold_score = pd.Series(0.0, index=data.index)
        
        # 威廉指标在合理范围内
        in_range = (wr >= -100) & (wr <= 0) & (wr.notna())
        overbought_oversold_score = np.where(in_range, 15, overbought_oversold_score)
        
        # 超买超卖区域的有效性
        in_oversold = wr <= -80
        in_overbought = wr >= -20
        in_normal = (wr > -80) & (wr < -20)
        
        # 合理分布评分
        overbought_oversold_score = np.where(in_oversold, overbought_oversold_score + 10,
                                           np.where(in_overbought, overbought_oversold_score + 10,
                                                   np.where(in_normal, overbought_oversold_score + 15,
                                                           overbought_oversold_score + 5)))
        
        # 极值处理能力
        extreme_values = (wr <= -95) | (wr >= -5)
        overbought_oversold_score = np.where(extreme_values, overbought_oversold_score + 10,
                                           overbought_oversold_score + 5)
        
        scores += overbought_oversold_score * 0.35
        
        # 2. 趋势识别有效性评分 (30%)
        trend_score = pd.Series(0.0, index=data.index)
        
        # 趋势一致性
        if len(wr_trend) >= 5:
            trend_consistency = np.abs(wr_trend.rolling(window=5).mean())
            strong_trend = trend_consistency >= 0.6
            trend_score = np.where(strong_trend, 15, trend_score)
            
            # 趋势变化的及时性
            trend_changes = (wr_trend != wr_trend.shift(1)) & (wr_trend != 0)
            timely_change = trend_changes & (np.abs(wr_change) > 2)
            trend_score = np.where(timely_change, trend_score + 10, trend_score + 5)
        
        # 威廉指标与均线的关系
        if wr_ma_3.notna().any():
            ma_alignment = np.where((wr > wr_ma_3) & (wr_trend == 1), 5,
                                  np.where((wr < wr_ma_3) & (wr_trend == -1), 5, 3))
            trend_score += ma_alignment
        
        scores += trend_score * 0.30
        
        # 3. 动量变化评分 (20%)
        momentum_score = pd.Series(0.0, index=data.index)
        
        if wr_momentum.notna().any():
            # 动量强度
            strong_momentum = np.abs(wr_momentum) > 10
            momentum_score = np.where(strong_momentum, 10, momentum_score)
            
            # 动量方向与威廉指标位置的一致性
            momentum_consistency = ((wr_momentum > 0) & (wr < -50)) | ((wr_momentum < 0) & (wr > -50))
            momentum_score = np.where(momentum_consistency, momentum_score + 10, momentum_score + 5)
        
        # 威廉指标变化率的合理性
        if wr_change.notna().any():
            reasonable_change = (np.abs(wr_change) >= 1) & (np.abs(wr_change) <= 20)
            momentum_score = np.where(reasonable_change, momentum_score + 5, momentum_score + 2)
        
        scores += momentum_score * 0.20
        
        # 4. 信号质量评分 (15%)
        signal_score = pd.Series(0.0, index=data.index)
        
        # 波动性的合理性
        if wr_volatility.notna().any():
            reasonable_volatility = (wr_volatility >= 5) & (wr_volatility <= 30)
            signal_score = np.where(reasonable_volatility, 10, signal_score)
            
            # 低波动性时的信号质量
            low_volatility = wr_volatility < 10
            signal_score = np.where(low_volatility & (np.abs(wr_change) < 3), signal_score + 5,
                                  signal_score + 3)
        
        scores += signal_score * 0.15
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    @international_financial_exception_handler(reraise=False, default_return=0.8)
    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """实现抽象方法"""
        if self._result is None:
            return 0.8
        
        # 基于WILLIAMS_R指标的可靠性计算置信度
        wr = self._result['WILLIAMS_R'].dropna()
        wr_volatility = self._result.get('WR_VOLATILITY', pd.Series()).dropna()
        wr_trend = self._result.get('WR_TREND', pd.Series()).dropna()
        
        if len(wr) == 0:
            return 0.8
        
        # 计算威廉指标的稳定性
        recent_wr = wr.iloc[-10:] if len(wr) >= 10 else wr
        
        # 威廉指标数值的合理性
        value_reasonableness = 0
        reasonable_values = (recent_wr >= -100) & (recent_wr <= 0)
        if reasonable_values.sum() / len(recent_wr) >= 0.9:
            value_reasonableness = 0.3
        elif reasonable_values.sum() / len(recent_wr) >= 0.7:
            value_reasonableness = 0.2
        else:
            value_reasonableness = 0.1
        
        # 趋势一致性
        trend_consistency = 0
        if len(wr_trend) >= 5:
            recent_trend = wr_trend.iloc[-5:]
            trend_changes = len(recent_trend[recent_trend != recent_trend.shift(1)].dropna())
            if trend_changes <= 1:  # 趋势稳定
                trend_consistency = 0.25
            elif trend_changes <= 2:
                trend_consistency = 0.15
            else:
                trend_consistency = 0.05
        
        # 波动性的合理性
        volatility_reasonableness = 0
        if len(wr_volatility) >= 5:
            recent_volatility = wr_volatility.iloc[-5:]
            avg_volatility = recent_volatility.mean()
            if 5 <= avg_volatility <= 25:  # 合理的波动性
                volatility_reasonableness = 0.2
            elif 3 <= avg_volatility <= 35:
                volatility_reasonableness = 0.15
            else:
                volatility_reasonableness = 0.05
        
        base_confidence = 0.3 + value_reasonableness + trend_consistency + volatility_reasonableness
        return min(max(base_confidence, 0.5), 0.95)
    
    @international_financial_exception_handler(reraise=True)
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """实现抽象方法"""
        return self.get_patterns_WILLIAMS_R_international_financial_grade(data, **kwargs)
    
    def get_patterns_WILLIAMS_R_international_financial_grade(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """国际金融级WILLIAMS_R形态识别"""
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.DataFrame(index=data.index)
        
        patterns = pd.DataFrame(index=data.index)
        
        wr = self._result['WILLIAMS_R']
        wr_ma_3 = self._result.get('WR_MA_3', wr)
        wr_ma_6 = self._result.get('WR_MA_6', wr)
        wr_change = self._result.get('WR_CHANGE', 0)
        wr_momentum = self._result.get('WR_MOMENTUM', 0)
        wr_trend = self._result.get('WR_TREND', 0)
        
        # 基本超买超卖形态
        patterns['WR_OVERSOLD'] = wr <= -80
        patterns['WR_OVERBOUGHT'] = wr >= -20
        patterns['WR_NEUTRAL'] = (wr > -80) & (wr < -20)
        
        # 极值形态
        patterns['WR_EXTREME_OVERSOLD'] = wr <= -95
        patterns['WR_EXTREME_OVERBOUGHT'] = wr >= -5
        
        # 反转形态
        patterns['WR_OVERSOLD_REVERSAL'] = (wr > -80) & (wr.shift(1) <= -80) & (wr_change > 0)
        patterns['WR_OVERBOUGHT_REVERSAL'] = (wr < -20) & (wr.shift(1) >= -20) & (wr_change < 0)
        
        # 突破形态
        patterns['WR_UPWARD_BREAKOUT'] = (wr > -50) & (wr.shift(1) <= -50) & (wr_change > 2)
        patterns['WR_DOWNWARD_BREAKDOWN'] = (wr < -50) & (wr.shift(1) >= -50) & (wr_change < -2)
        
        # 交叉形态
        if wr_ma_3.notna().any() and wr_ma_6.notna().any():
            patterns['WR_GOLDEN_CROSS'] = (wr_ma_3 > wr_ma_6) & (wr_ma_3.shift(1) <= wr_ma_6.shift(1))
            patterns['WR_DEATH_CROSS'] = (wr_ma_3 < wr_ma_6) & (wr_ma_3.shift(1) >= wr_ma_6.shift(1))
        
        # 动量形态
        patterns['WR_BULLISH_MOMENTUM'] = (wr_momentum > 10) & (wr < -50) & (wr_trend == 1)
        patterns['WR_BEARISH_MOMENTUM'] = (wr_momentum < -10) & (wr > -50) & (wr_trend == -1)
        
        # 背离形态（简化版）
        if len(wr) >= 20:
            # 价格与威廉指标的背离
            price_trend = data['close'].rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            wr_trend_simple = wr.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            
            patterns['WR_BULLISH_DIVERGENCE'] = (price_trend == -1) & (wr_trend_simple == 1) & (wr < -60)
            patterns['WR_BEARISH_DIVERGENCE'] = (price_trend == 1) & (wr_trend_simple == -1) & (wr > -40)
        
        # 振荡形态
        patterns['WR_OSCILLATION'] = (wr > -70) & (wr < -30)
        
        # 趋势确认形态
        patterns['WR_STRONG_UPTREND'] = (wr_trend == 1) & (wr > -60) & (wr_change > 0)
        patterns['WR_STRONG_DOWNTREND'] = (wr_trend == -1) & (wr < -40) & (wr_change < 0)
        patterns['WR_SIDEWAYS'] = (wr_trend == 0) & (np.abs(wr_change) < 2)
        
        return patterns
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """实现抽象方法"""
        # 更新参数
        for key, value in kwargs.items():
            if key in ['period', 'overbought', 'oversold']:
                self._default_parameters[key] = value
    
    def _get_default_parameters_williams_r(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'period': 14,
            'overbought': -20,
            'oversold': -80
        }
    
    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return self._result is not None and not self._result.empty
