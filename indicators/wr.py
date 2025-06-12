#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
威廉指标(WR)

与KDJ配合使用，确认超买超卖
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple
import talib

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength

logger = get_logger(__name__)


class WR(BaseIndicator):
    """
    威廉指标(WR) (WR)
    
    分类：震荡类指标
    描述：与KDJ配合使用，确认超买超卖
    """
    
    def __init__(self, period: int = 14):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化威廉指标(WR)指标
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__(name="WR", description="威廉指标，确认超买超卖")
        self.period = period
        
    def set_parameters(self, period: int = None):
        """
        设置指标参数
        """
        if period is not None:
            self.period = period
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算WR指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含WR指标的DataFrame
        """
        return self.calculate(df)
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算威廉指标(WR)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了WR指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 实现威廉指标(WR)计算逻辑
        # WR = -100 * (HIGH(n) - CLOSE) / (HIGH(n) - LOW(n))
        # 其中HIGH(n)和LOW(n)分别为n周期内的最高价和最低价
        highest_high = df_copy['high'].rolling(window=self.period).max()
        lowest_low = df_copy['low'].rolling(window=self.period).min()
        
        # 计算WR值
        df_copy['wr'] = -100 * (highest_high - df_copy['close']) / (highest_high - lowest_low)
        
        # 保存结果
        self._result = df_copy
        
        return df_copy
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算WR原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算WR
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        wr = self._result['wr']
        
        # 1. 超买超卖区域评分
        # WR < -80（超卖）+20分
        oversold_condition = wr < -80
        score += oversold_condition * 20
        
        # WR > -20（超买）-20分
        overbought_condition = wr > -20
        score -= overbought_condition * 20
        
        # 2. WR穿越关键位置评分
        # WR从超卖区上穿-80+25分
        wr_cross_up_oversold = self.crossover(wr, -80)
        score += wr_cross_up_oversold * 25
        
        # WR从超买区下穿-20-25分
        wr_cross_down_overbought = self.crossunder(wr, -20)
        score -= wr_cross_down_overbought * 25
        
        # 3. 中线穿越评分
        # WR上穿-50+15分
        wr_cross_up_middle = self.crossover(wr, -50)
        score += wr_cross_up_middle * 15
        
        # WR下穿-50-15分
        wr_cross_down_middle = self.crossunder(wr, -50)
        score -= wr_cross_down_middle * 15
        
        # 4. WR背离评分
        if len(data) >= 20:
            divergence_score = self._calculate_wr_divergence(data['close'], wr)
            score += divergence_score
        
        # 5. WR极端值评分
        # WR < -90（极度超卖）+30分
        extreme_oversold = wr < -90
        score += extreme_oversold * 30
        
        # WR > -10（极度超买）-30分
        extreme_overbought = wr > -10
        score -= extreme_overbought * 30
        
        # 6. WR趋势评分
        wr_trend_score = self._calculate_wr_trend_score(wr)
        score += wr_trend_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别WR技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算WR
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        wr = self._result['wr']
        
        # 检查最近的信号
        recent_periods = min(10, len(wr))
        if recent_periods == 0:
            return patterns
        
        recent_wr = wr.tail(recent_periods)
        current_wr = recent_wr.iloc[-1]
        
        # 1. 超买超卖形态
        if current_wr <= -90:
            patterns.append("WR极度超卖")
        elif current_wr <= -80:
            patterns.append("WR超卖")
        elif current_wr >= -10:
            patterns.append("WR极度超买")
        elif current_wr >= -20:
            patterns.append("WR超买")
        
        # 2. 穿越形态
        if self.crossover(recent_wr, -80).any():
            patterns.append("WR上穿超卖线")
        if self.crossunder(recent_wr, -20).any():
            patterns.append("WR下穿超买线")
        if self.crossover(recent_wr, -50).any():
            patterns.append("WR上穿中线")
        if self.crossunder(recent_wr, -50).any():
            patterns.append("WR下穿中线")
        
        # 3. 背离形态
        if len(data) >= 20:
            divergence_type = self._detect_wr_divergence_pattern(data['close'], wr)
            if divergence_type:
                patterns.append(f"WR{divergence_type}")
        
        # 4. 钝化形态
        if self._detect_wr_stagnation(recent_wr, threshold=-80, periods=5, direction='low'):
            patterns.append("WR低位钝化")
        if self._detect_wr_stagnation(recent_wr, threshold=-20, periods=5, direction='high'):
            patterns.append("WR高位钝化")
        
        # 5. 反转形态
        if self._detect_wr_reversal_pattern(recent_wr):
            patterns.append("WR反转形态")
        
        return patterns
    
    def _calculate_wr_divergence(self, price: pd.Series, wr: pd.Series) -> pd.Series:
        """
        计算WR背离评分
        
        Args:
            price: 价格序列
            wr: WR序列
            
        Returns:
            pd.Series: 背离评分序列
        """
        divergence_score = pd.Series(0.0, index=price.index)
        
        if len(price) < 20:
            return divergence_score
        
        # 寻找价格和WR的峰值谷值
        window = 5
        for i in range(window, len(price) - window):
            price_window = price.iloc[i-window:i+window+1]
            wr_window = wr.iloc[i-window:i+window+1]
            
            if price.iloc[i] == price_window.max():  # 价格峰值
                if wr.iloc[i] != wr_window.max():  # WR未创新高
                    divergence_score.iloc[i:i+10] -= 25  # 负背离
            elif price.iloc[i] == price_window.min():  # 价格谷值
                if wr.iloc[i] != wr_window.min():  # WR未创新低
                    divergence_score.iloc[i:i+10] += 25  # 正背离
        
        return divergence_score
    
    def _calculate_wr_trend_score(self, wr: pd.Series) -> pd.Series:
        """
        计算WR趋势评分
        
        Args:
            wr: WR序列
            
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=wr.index)
        
        if len(wr) < 5:
            return trend_score
        
        # 计算WR斜率
        wr_slope = wr.diff(3)
        
        # 趋势评分
        trend_score += np.where(wr_slope > 5, 10, 0)   # 强烈上升+10分
        trend_score += np.where(wr_slope > 2, 5, 0)    # 温和上升+5分
        trend_score -= np.where(wr_slope < -5, 10, 0)  # 强烈下降-10分
        trend_score -= np.where(wr_slope < -2, 5, 0)   # 温和下降-5分
        
        return trend_score
    
    def _detect_wr_divergence_pattern(self, price: pd.Series, wr: pd.Series) -> Optional[str]:
        """
        检测WR背离形态
        
        Args:
            price: 价格序列
            wr: WR序列
            
        Returns:
            Optional[str]: 背离类型或None
        """
        if len(price) < 20:
            return None
        
        # 寻找最近的峰值和谷值
        recent_price = price.tail(20)
        recent_wr = wr.tail(20)
        
        price_extremes = []
        wr_extremes = []
        
        # 简化的极值检测
        for i in range(2, len(recent_price) - 2):
            if (recent_price.iloc[i] > recent_price.iloc[i-1] and 
                recent_price.iloc[i] > recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                wr_extremes.append(recent_wr.iloc[i])
            elif (recent_price.iloc[i] < recent_price.iloc[i-1] and 
                  recent_price.iloc[i] < recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                wr_extremes.append(recent_wr.iloc[i])
        
        if len(price_extremes) >= 2:
            price_trend = price_extremes[-1] - price_extremes[-2]
            wr_trend = wr_extremes[-1] - wr_extremes[-2]
            
            # 正背离：价格创新低但WR未创新低
            if price_trend < -0.01 and wr_trend > 2:
                return "正背离"
            # 负背离：价格创新高但WR未创新高
            elif price_trend > 0.01 and wr_trend < -2:
                return "负背离"
        
        return None
    
    def _detect_wr_stagnation(self, wr: pd.Series, threshold: float, 
                             periods: int, direction: str) -> bool:
        """
        检测WR钝化
        
        Args:
            wr: WR序列
            threshold: 阈值
            periods: 检测周期数
            direction: 方向 ('low' 或 'high')
            
        Returns:
            bool: 是否钝化
        """
        if len(wr) < periods:
            return False
        
        recent_wr = wr.tail(periods)
        
        if direction == 'low':
            return (recent_wr < threshold).all()
        elif direction == 'high':
            return (recent_wr > threshold).all()
        
        return False
    
    def _detect_wr_reversal_pattern(self, wr: pd.Series) -> bool:
        """
        检测WR反转形态
        
        Args:
            wr: WR序列
            
        Returns:
            bool: 是否为反转形态
        """
        if len(wr) < 5:
            return False
        
        # 检测V型反转：从极端位置快速反转
        recent_wr = wr.tail(5)
        
        # 从超卖区快速反转
        if (recent_wr.iloc[0] < -80 and recent_wr.iloc[-1] > -50 and
            (recent_wr.iloc[-1] - recent_wr.iloc[0]) > 20):
            return True
        
        # 从超买区快速反转
        if (recent_wr.iloc[0] > -20 and recent_wr.iloc[-1] < -50 and
            (recent_wr.iloc[0] - recent_wr.iloc[-1]) > 20):
            return True
        
        return False
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成威廉指标(WR)指标交易信号
        
        Args:
            df: 包含价格数据和WR指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - wr_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['wr']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        overbought = kwargs.get('overbought', -20)  # 超买阈值
        oversold = kwargs.get('oversold', -80)  # 超卖阈值
        
        # 实现信号生成逻辑
        df_copy['wr_signal'] = 0
        
        # 超卖区域上穿信号线为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['wr'].iloc[i-1] < oversold and df_copy['wr'].iloc[i] > oversold:
                df_copy.iloc[i, df_copy.columns.get_loc('wr_signal')] = 1
            
            # 超买区域下穿信号线为卖出信号
            elif df_copy['wr'].iloc[i-1] > overbought and df_copy['wr'].iloc[i] < overbought:
                df_copy.iloc[i, df_copy.columns.get_loc('wr_signal')] = -1
        
        return df_copy
        
    def _register_wr_patterns(self):
        """
        注册WR指标相关形态
        """
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册WR超买超卖形态
        registry.register(
            pattern_id="WR_OVERBOUGHT",
            display_name="WR超买",
            description="WR值高于-20，表明市场可能超买，存在回调风险",
            indicator_id="WR",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        registry.register(
            pattern_id="WR_OVERSOLD",
            display_name="WR超卖",
            description="WR值低于-80，表明市场可能超卖，存在反弹机会",
            indicator_id="WR",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        # 注册WR趋势形态
        registry.register(
            pattern_id="WR_UPTREND",
            display_name="WR上升趋势",
            description="WR值连续上升，表明价格相对高点接近",
            indicator_id="WR",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="WR_DOWNTREND",
            display_name="WR下降趋势",
            description="WR值连续下降，表明价格相对低点接近",
            indicator_id="WR",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0
        )
        
        # 注册WR零轴穿越形态
        registry.register(
            pattern_id="WR_CROSS_ABOVE_MID",
            display_name="WR上穿中轴",
            description="WR从下方穿越-50中轴线，表明买盘力量增强",
            indicator_id="WR",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="WR_CROSS_BELOW_MID",
            display_name="WR下穿中轴",
            description="WR从上方穿越-50中轴线，表明卖盘力量增强",
            indicator_id="WR",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0
        )
        
        # 注册WR背离形态
        registry.register(
            pattern_id="WR_BULLISH_DIVERGENCE",
            display_name="WR底背离",
            description="价格创新低，但WR未创新低，表明下跌动能减弱",
            indicator_id="WR",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="WR_BEARISH_DIVERGENCE",
            display_name="WR顶背离",
            description="价格创新高，但WR未创新高，表明上涨动能减弱",
            indicator_id="WR",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )
        
        # 注册WR反转形态
        registry.register(
            pattern_id="WR_BULLISH_REVERSAL",
            display_name="WR超卖反转",
            description="WR在超卖区见底回升，表明可能形成底部",
            indicator_id="WR",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=18.0
        )
        
        registry.register(
            pattern_id="WR_BEARISH_REVERSAL",
            display_name="WR超买反转",
            description="WR在超买区触顶回落，表明可能形成顶部",
            indicator_id="WR",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-18.0
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 初始化信号
        signals = {}
        
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
        
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
        
        return signals

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> list:
        """
        获取WR指标的技术形态
        """
        return self.identify_patterns(data, **kwargs)
