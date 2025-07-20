#!/usr/bin/env python
from utils.dependency_injection import get_logger
# -*- coding: utf-8 -*-

"""
威廉指标(WR_Wr)

与KDJ配合使用，确认超买超卖
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Dict, Optional, Tuple, Any
# import talib  # 移除talib依赖

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.indicator_utils import crossover, crossunder
from utils.dependency_injection import get_logger
from indicators.pattern_registry import PatternRegistry, PatternTypePatternRegistry, PatternStrengthPatternRegistry, PatternPolarity

logger = get_logger(__name__)


class WrWr(BaseIndicator, PatternSignalMixin):
    """
    威廉指标(WR_Wr) (WR_Wr)
    
    分类：震荡类指标
    描述：与KDJ配合使用，确认超买超卖
    """
    
    def __init__(self, **kwargs):
        """
        初始化WR指标

        Args:
            **kwargs: 指标参数，支持period、overbought、oversold等
        """
        super().__init__()
        self.REQUIRED_COLUMNS = ['high', 'low', 'close']
        self.name = "WR"
        self.description = "威廉指标"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_wr()

        # 应用用户参数
        self.set_parameters_Wr_Wr(**kwargs)
    
    def _get_default_parameters_wr(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14, "overbought": -20.0, "oversold": -80.0}

    def calculate_Wr_Wr(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算WR指标

        Args:
            data: 包含OHLCV数据的Data_frame
            **kwargs: 其他参数

        Returns:
            包含WR指标的Data_frame
        """
        return self._calculate_wr(data)
        
    def set_parameters_Wr_Wr(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典，支持以下参数：
                - period: 计算周期
                - overbought: 超买阈值
                - oversold: 超卖阈值
        """
        # 验证参数
        from utils.indicator_parameter_validator import IndicatorParameterValidator
        validator = IndicatorParameterValidator()
        
        # 合并默认参数和用户参数
        params = self._default_parameters.copy()
        params.update(kwargs)        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('WR_Wr', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = params.get('period', 14)
        self.overbought = params.get('overbought', -20.0)
        self.oversold = params.get('oversold', -80.0)
    def _validate_dataframe_wr(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证Data_frame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def compute_Wr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算WR指标
        
        Args:
            df: 包含OHLCV数据的Data_frame
                
        Returns:
            包含WR指标的Data_frame
        """
        return self.calculate_Wr_Wr(df)
        
    def _calculate_wr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算威廉指标(WR_Wr)指标
        
        Args:
            df: 包含OHLCV数据的Data_frame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了WR指标列的Data_frame
        """
        if df.empty:
            return pd.DataFrame()

        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe_wr(df, required_columns)
        
        df_copy = df.copy()
        
        # 实现威廉指标(WR_Wr)计算逻辑
        # WR_Wr = -100 * (HIGH(n) - CLOSE) / (HIGH(n) - LOW(n))
        # 其中HIGH(n)和LOW(n)分别为n周期内的最高价和最低价
        highest_high = df_copy['high'].rolling(window=self.period).max()
        lowest_low = df_copy['low'].rolling(window=self.period).min()
        
        # 计算WR值
        df_copy['wr'] = -100 * (highest_high - df_copy['close']) / (highest_high - lowest_low)
        
        # 添加形态识别和信号生成
        df_copy = self.add_pattern_detection(df_copy)
        df_copy = self.add_signal_generation(df_copy)

        # 保存结果
        self._result = df_copy

        return df_copy

    def calculate_raw_score_Wr_Wr(self, data: pd.DataFrame, **kwargs) -> pd.Series:
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
            self.calculate_Wr_Wr(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        wr = self._result['wr']
        
        # 1. 超买超卖区域评分
        # WR_Wr < -80（超卖）+20分
        oversold_condition = wr < -80
        score += oversold_condition * 20
        
        # WR_Wr > -20（超买）-20分
        overbought_condition = wr > -20
        score -= overbought_condition * 20
        
        # 2. WR穿越关键位置评分
        # WR从超卖区上穿-80+25分
        wr_cross_up_oversold = crossover(wr, -80)
        score += wr_cross_up_oversold * 25

        # WR从超买区下穿-20-25分
        wr_cross_down_overbought = crossunder(wr, -20)
        score -= wr_cross_down_overbought * 25

        # 3. 中线穿越评分
        # WR上穿-50+15分
        wr_cross_up_middle = crossover(wr, -50)
        score += wr_cross_up_middle * 15

        # WR下穿-50-15分
        wr_cross_down_middle = crossunder(wr, -50)
        score -= wr_cross_down_middle * 15
        
        # 4. WR背离评分
        if len(data) >= 20:
            divergence_score = self._calculate_wr_divergence_Wr(data['close'], wr)
            score += divergence_score
        
        # 5. WR极端值评分
        # WR_Wr < -90（极度超卖）+30分
        extreme_oversold = wr < -90
        score += extreme_oversold * 30
        
        # WR_Wr > -10（极度超买）-30分
        extreme_overbought = wr > -10
        score -= extreme_overbought * 30
        
        # 6. WR趋势评分
        wr_trend_score = self._calculate_wr_trend_score(wr)
        score += wr_trend_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns_Wr(self, data: pd.DataFrame, **kwargs) -> List[str]:
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
            self.calculate_Wr_Wr(data, **kwargs)
        
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
        if crossover(recent_wr, -80).any():
            patterns.append("WR上穿超卖线")
        if crossunder(recent_wr, -20).any():
            patterns.append("WR下穿超买线")
        if crossover(recent_wr, -50).any():
            patterns.append("WR上穿中线")
        if crossunder(recent_wr, -50).any():
            patterns.append("WR下穿中线")
        
        # 3. 背离形态
        if len(data) >= 20:
            divergence_type = self._detect_wr_divergence_pattern(data['close'], wr)
            if divergence_type:
                patterns.append(f"WR_Wr{divergence_type}")
        
        # 4. 钝化形态
        if self._detect_wr_stagnation(recent_wr, threshold=-80, periods=5, direction='low'):
            patterns.append("WR低位钝化")
        if self._detect_wr_stagnation(recent_wr, threshold=-20, periods=5, direction='high'):
            patterns.append("WR高位钝化")
        
        # 5. 反转形态
        if self._detect_wr_reversal_pattern(recent_wr):
            patterns.append("WR反转形态")
        
        
        # 添加形态识别和信号生成
        patterns = self.add_pattern_detection(patterns)
        patterns = self.add_signal_generation(patterns)

        return patterns
    
    def _calculate_wr_divergence_Wr(self, price: pd.Series, wr: pd.Series) -> pd.Series:
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
        
    def get_signals_Wr(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成威廉指标(WR_Wr)指标交易信号
        
        Args:
            df: 包含价格数据和WR指标的Data_frame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的Data_frame:
            - wr_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['wr']
        self._validate_dataframe_wr(df, required_columns)
        
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
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BEARISH,
            default_strength=Pattern_strength.MEDIUM,
            score_impact=-15.0,
            polarity=Pattern_polarity.NEGATIVE
        )

        registry.register(
            pattern_id="WR_OVERSOLD",
            display_name="WR超卖",
            description="WR值低于-80，表明市场可能超卖，存在反弹机会",
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BULLISH,
            default_strength=Pattern_strength.MEDIUM,
            score_impact=15.0,
            polarity=Pattern_polarity.POSITIVE
        )

        # 注册WR趋势形态
        registry.register(
            pattern_id="WR_UPTREND",
            display_name="WR上升趋势",
            description="WR值连续上升，表明价格相对高点接近",
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BULLISH,
            default_strength=Pattern_strength.MEDIUM,
            score_impact=12.0,
            polarity=Pattern_polarity.POSITIVE
        )

        registry.register(
            pattern_id="WR_DOWNTREND",
            display_name="WR下降趋势",
            description="WR值连续下降，表明价格相对低点接近",
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BEARISH,
            default_strength=Pattern_strength.MEDIUM,
            score_impact=-12.0,
            polarity=Pattern_polarity.NEGATIVE
        )
        
        # 注册WR零轴穿越形态
        registry.register(
            pattern_id="WR_CROSS_ABOVE_MID",
            display_name="WR上穿中轴",
            description="WR从下方穿越-50中轴线，表明买盘力量增强",
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BULLISH,
            default_strength=Pattern_strength.MEDIUM,
            score_impact=10.0,
            polarity=Pattern_polarity.POSITIVE
        )

        registry.register(
            pattern_id="WR_CROSS_BELOW_MID",
            display_name="WR下穿中轴",
            description="WR从上方穿越-50中轴线，表明卖盘力量增强",
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BEARISH,
            default_strength=Pattern_strength.MEDIUM,
            score_impact=-10.0,
            polarity=Pattern_polarity.NEGATIVE
        )

        # 注册WR背离形态
        registry.register(
            pattern_id="WR_BULLISH_DIVERGENCE",
            display_name="WR底背离",
            description="价格创新低，但WR未创新低，表明下跌动能减弱",
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BULLISH,
            default_strength=Pattern_strength.STRONG,
            score_impact=20.0,
            polarity=Pattern_polarity.POSITIVE
        )

        registry.register(
            pattern_id="WR_BEARISH_DIVERGENCE",
            display_name="WR顶背离",
            description="价格创新高，但WR未创新高，表明上涨动能减弱",
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BEARISH,
            default_strength=Pattern_strength.STRONG,
            score_impact=-20.0,
            polarity=Pattern_polarity.NEGATIVE
        )

        # 注册WR反转形态
        registry.register(
            pattern_id="WR_BULLISH_REVERSAL",
            display_name="WR超卖反转",
            description="WR在超卖区见底回升，表明可能形成底部",
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BULLISH,
            default_strength=Pattern_strength.STRONG,
            score_impact=18.0,
            polarity=Pattern_polarity.POSITIVE
        )

        registry.register(
            pattern_id="WR_BEARISH_REVERSAL",
            display_name="WR超买反转",
            description="WR在超买区触顶回落，表明可能形成顶部",
            indicator_id="WR_Wr",
            pattern_type=Pattern_type.BEARISH,
            default_strength=Pattern_strength.STRONG,
            score_impact=-18.0,
            polarity=Pattern_polarity.NEGATIVE
        )

    def generate_trading_signals_Wr(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
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
            self.calculate_Wr_Wr(data, **kwargs)
        
        # 初始化信号
        signals = {}
        
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
        
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
        
        return signals

    def get_patterns_Wr_Wr(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取WR相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的Data_frame
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate_Wr_Wr(data)

        if self._result is None or 'wr' not in self._result.columns:
            return pd.DataFrame(index=data.index)

        # 获取WR数据
        wr = self._result['wr']

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 1. WR超买超卖形态
        patterns_df['WR_EXTREME_OVERSOLD'] = wr < -90
        patterns_df['WR_OVERSOLD'] = (wr >= -90) & (wr < -80)
        patterns_df['WR_NORMAL'] = (wr >= -80) & (wr <= -20)
        patterns_df['WR_OVERBOUGHT'] = (wr > -20) & (wr <= -10)
        patterns_df['WR_EXTREME_OVERBOUGHT'] = wr > -10

        # 2. WR穿越形态
        patterns_df['WR_CROSS_ABOVE_OVERSOLD'] = crossover(wr, -80)
        patterns_df['WR_CROSS_BELOW_OVERBOUGHT'] = crossunder(wr, -20)
        patterns_df['WR_CROSS_ABOVE_MID'] = crossover(wr, -50)
        patterns_df['WR_CROSS_BELOW_MID'] = crossunder(wr, -50)

        # 3. WR趋势形态
        patterns_df['WR_RISING'] = wr > wr.shift(1)
        patterns_df['WR_FALLING'] = wr < wr.shift(1)
        patterns_df['WR_UPTREND'] = (
            (wr > wr.shift(1)) &
            (wr.shift(1) > wr.shift(2)) &
            (wr.shift(2) > wr.shift(3))
        )
        patterns_df['WR_DOWNTREND'] = (
            (wr < wr.shift(1)) &
            (wr.shift(1) < wr.shift(2)) &
            (wr.shift(2) < wr.shift(3))
        )

        # 4. WR钝化形态
        patterns_df['WR_LOW_STAGNATION'] = wr.rolling(5).apply(lambda x: (x < -80).all(), raw=False)
        patterns_df['WR_HIGH_STAGNATION'] = wr.rolling(5).apply(lambda x: (x > -20).all(), raw=False)

        # 5. WR反转形态
        # 从超卖区快速反转
        wr_change_5 = wr - wr.shift(4)
        patterns_df['WR_BULLISH_REVERSAL'] = (wr.shift(4) < -80) & (wr > -50) & (wr_change_5 > 20)
        patterns_df['WR_BEARISH_REVERSAL'] = (wr.shift(4) > -20) & (wr < -50) & (wr_change_5 < -20)

        # 确保所有列都是布尔类型，填充NaN为False
        for col in patterns_df.columns:
            patterns_df[col] = patterns_df[col].fillna(False).astype(bool)

        return patterns_df

    def calculate_confidence_Wr_Wr(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算WR指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态Data_frame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if not patterns.empty:
            # 检查WR形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.2)

        # 3. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.1, 0.15)

        # 4. 基于评分趋势的置信度
        if len(score) >= 3:
            recent_scores = score.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def calculate_score_Wr(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算最终评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 包含评分和置信度的字典
        """
        try:
            # 1. 计算原始评分序列
            raw_scores = self.calculate_raw_score_Wr_Wr(data, **kwargs)

            # 如果数据不足，返回中性评分
            if len(raw_scores) < 3:
                return {'score': 50.0, 'confidence': 0.5}

            # 取最近的评分作为最终评分，但考虑近期趋势
            recent_scores = raw_scores.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 最终评分 = 最新评分 + 趋势调整
            final_score = recent_scores.iloc[-1] + trend / 2

            # 确保评分在0-100范围内
            final_score = max(0, min(100, final_score))

            # 2. 获取形态和信号
            patterns = self.get_patterns_Wr_Wr(data, **kwargs)

            # 3. 计算置信度
            confidence = self.calculate_confidence_Wr_Wr(raw_scores, patterns, {})

            return {
                'score': final_score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"为指标 {self.name} 计算评分时出错: {e}")
            return {'score': 50.0, 'confidence': 0.0}

    def register_patterns_Wr(self):
        """
        注册WR指标的形态到全局形态注册表
        """
        # 注册WR超买超卖形态
        self.register_pattern_to_registry(
            pattern_id="WR_EXTREME_OVERSOLD",
            display_name="WR极度超卖",
            description="WR值低于-90，表明市场极度超卖，存在强烈反弹机会",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="WR_OVERSOLD",
            display_name="WR超卖",
            description="WR值在-90到-80之间，表明市场超卖",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="WR_OVERBOUGHT",
            display_name="WR超买",
            description="WR值在-20到-10之间，表明市场超买",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="WR_EXTREME_OVERBOUGHT",
            display_name="WR极度超买",
            description="WR值高于-10，表明市场极度超买，存在强烈回调风险",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册WR状态形态（从centralized mapping迁移）
        self.register_pattern_to_registry(
            pattern_id="WR_RISING",
            display_name="WR上升",
            description="威廉指标上升，超卖状态缓解",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="WR_NORMAL",
            display_name="WR正常",
            description="威廉指标处于正常范围",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="WR_LOW_STAGNATION",
            display_name="WR低位停滞",
            description="威廉指标在低位停滞",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=5.0,
            polarity="NEUTRAL"
        )

        # 注册WR穿越形态
        self.register_pattern_to_registry(
            pattern_id="WR_CROSS_ABOVE_OVERSOLD",
            display_name="WR上穿超卖线",
            description="WR从超卖区域向上突破-80线，看涨信号",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="WR_CROSS_BELOW_OVERBOUGHT",
            display_name="WR下穿超买线",
            description="WR从超买区域向下突破-20线，看跌信号",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册WR反转形态
        self.register_pattern_to_registry(
            pattern_id="WR_BULLISH_REVERSAL",
            display_name="WR超卖反转",
            description="WR在超卖区见底回升，表明可能形成底部",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="WR_BEARISH_REVERSAL",
            display_name="WR超买反转",
            description="WR在超买区触顶回落，表明可能形成顶部",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-18.0,
            polarity="NEGATIVE"
        )
    def get_pattern_info_Wr(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # WR指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)

    # ==================== 抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        return self.calculate_Wr_Wr(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        return self.calculate_raw_score_Wr(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        return self.get_patterns_Wr(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        return self.set_parameters_Wr_Wr(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """抽象基类要求的置信度计算方法"""
        return self.calculate_confidence_Wr(score, patterns, signals)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """统一的计算接口"""
        return self.calculate_Wr_Wr(data, **kwargs)

    # ==================== 兼容性方法 - 真实实现 ====================

    def get_patterns(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """真实实现：获取WR形态"""
        if data is None or data.empty:
            return pd.DataFrame()

        # 首先计算WR指标
        wr_data = self.calculate_Wr_Wr(data)

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 获取WR数据
        if isinstance(wr_data, dict):
            wr_values = wr_data.get('wr', pd.Series(index=data.index))
        else:
            wr_values = wr_data.get('wr', pd.Series(index=data.index))

        # 1. 超买形态 (WR > -20)
        patterns_df['WR_OVERBOUGHT'] = wr_values > -20

        # 2. 超卖形态 (WR < -80)
        patterns_df['WR_OVERSOLD'] = wr_values < -80

        # 3. 极端超买形态 (WR > -10)
        patterns_df['WR_EXTREME_OVERBOUGHT'] = wr_values > -10

        # 4. 极端超卖形态 (WR < -90)
        patterns_df['WR_EXTREME_OVERSOLD'] = wr_values < -90

        # 5. 从超卖区域反弹形态
        patterns_df['WR_OVERSOLD_BOUNCE'] = (wr_values > -80) & (wr_values.shift(1) <= -80)

        # 6. 从超买区域回落形态
        patterns_df['WR_OVERBOUGHT_FALL'] = (wr_values < -20) & (wr_values.shift(1) >= -20)

        # 7. 中性区域突破形态
        patterns_df['WR_NEUTRAL_BREAK_UP'] = (wr_values > -50) & (wr_values.shift(1) <= -50)
        patterns_df['WR_NEUTRAL_BREAK_DOWN'] = (wr_values < -50) & (wr_values.shift(1) >= -50)

        # 8. 背离形态检测
        if len(data) >= 20:
            # 简化的背离检测：价格创新高但WR未创新高
            price_high = data['high'].rolling(10).max()
            wr_high = wr_values.rolling(10).max()
            patterns_df['WR_BEARISH_DIVERGENCE'] = (
                (data['high'] >= price_high.shift(1)) &
                (wr_values < wr_high.shift(1)) &
                (wr_values > -50)
            )

            # 底背离形态：价格创新低但WR未创新低
            price_low = data['low'].rolling(10).min()
            wr_low = wr_values.rolling(10).min()
            patterns_df['WR_BULLISH_DIVERGENCE'] = (
                (data['low'] <= price_low.shift(1)) &
                (wr_values > wr_low.shift(1)) &
                (wr_values < -50)
            )

        return patterns_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """真实实现：计算WR原始评分"""
        if data.empty:
            return pd.Series(dtype=float)

        # 计算WR指标
        wr_data = self.calculate_Wr_Wr(data)

        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分

        # 获取WR数据
        if isinstance(wr_data, dict):
            wr_values = wr_data.get('wr', pd.Series(index=data.index))
        else:
            wr_values = wr_data.get('wr', pd.Series(index=data.index))

        # 1. 基于WR位置的评分
        # 超卖区域加分 (WR < -80)
        oversold_condition = wr_values < -80
        score += oversold_condition * 20

        # 极端超卖加分 (WR < -90)
        extreme_oversold_condition = wr_values < -90
        score += extreme_oversold_condition * 15

        # 超买区域减分 (WR > -20)
        overbought_condition = wr_values > -20
        score -= overbought_condition * 20

        # 极端超买减分 (WR > -10)
        extreme_overbought_condition = wr_values > -10
        score -= extreme_overbought_condition * 15

        # 2. 基于WR反弹和回落的评分
        oversold_bounce = (wr_values > -80) & (wr_values.shift(1) <= -80)
        overbought_fall = (wr_values < -20) & (wr_values.shift(1) >= -20)

        # 超卖反弹加分
        score += oversold_bounce * 15

        # 超买回落减分
        score -= overbought_fall * 15

        # 3. 基于WR趋势的评分
        # WR上升趋势加分
        wr_rising = wr_values > wr_values.shift(1)
        score += wr_rising * 5

        # WR下降趋势减分
        wr_falling = wr_values < wr_values.shift(1)
        score -= wr_falling * 5

        # 4. 基于WR距离中性位置的评分
        # WR越接近-50（中性），评分越接近50
        distance_from_neutral = np.abs(wr_values + 50)
        distance_bonus = np.maximum(0, 10 - distance_from_neutral / 5)
        score += distance_bonus

        # 限制评分在0-100之间
        return score.clip(0, 100)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：生成WR交易信号"""
        if data.empty:
            return pd.DataFrame()

        # 计算WR指标
        wr_data = self.calculate_Wr_Wr(data)
        result_df = data.copy()

        # 合并WR数据
        if isinstance(wr_data, dict):
            for col, values in wr_data.items():
                result_df[col] = values
        else:
            for col in wr_data.columns:
                result_df[col] = wr_data[col]

        # 初始化信号列
        result_df['wr_signal'] = 0
        result_df['wr_strength'] = 0.0
        result_df['wr_confidence'] = 0.0

        # 获取WR数据
        if isinstance(wr_data, dict):
            wr_values = wr_data.get('wr', pd.Series(index=data.index))
        else:
            wr_values = wr_data.get('wr', pd.Series(index=data.index))

        # 1. 超卖反弹买入信号
        oversold_bounce = (wr_values > -80) & (wr_values.shift(1) <= -80)
        result_df.loc[oversold_bounce, 'wr_signal'] = 1
        result_df.loc[oversold_bounce, 'wr_strength'] = 0.8
        result_df.loc[oversold_bounce, 'wr_confidence'] = 0.9

        # 2. 超买回落卖出信号
        overbought_fall = (wr_values < -20) & (wr_values.shift(1) >= -20)
        result_df.loc[overbought_fall, 'wr_signal'] = -1
        result_df.loc[overbought_fall, 'wr_strength'] = 0.8
        result_df.loc[overbought_fall, 'wr_confidence'] = 0.9

        # 3. 中性区域突破信号
        neutral_break_up = (wr_values > -50) & (wr_values.shift(1) <= -50)
        result_df.loc[neutral_break_up, 'wr_signal'] = 1
        result_df.loc[neutral_break_up, 'wr_strength'] = 0.6
        result_df.loc[neutral_break_up, 'wr_confidence'] = 0.7

        neutral_break_down = (wr_values < -50) & (wr_values.shift(1) >= -50)
        result_df.loc[neutral_break_down, 'wr_signal'] = -1
        result_df.loc[neutral_break_down, 'wr_strength'] = 0.6
        result_df.loc[neutral_break_down, 'wr_confidence'] = 0.7

        # 4. 极端超卖/超买信号
        extreme_oversold = wr_values < -90
        result_df.loc[extreme_oversold, 'wr_signal'] = 1
        result_df.loc[extreme_oversold, 'wr_strength'] = 0.9
        result_df.loc[extreme_oversold, 'wr_confidence'] = 0.8

        extreme_overbought = wr_values > -10
        result_df.loc[extreme_overbought, 'wr_signal'] = -1
        result_df.loc[extreme_overbought, 'wr_strength'] = 0.9
        result_df.loc[extreme_overbought, 'wr_confidence'] = 0.8

        return result_df

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> dict:
        """真实实现：计算WR综合评分"""
        if data.empty:
            return {'score': 50.0, 'confidence': 0.0, 'signals': {}}

        # 计算原始评分
        raw_score = self.calculate_raw_score(data, **kwargs)

        # 获取形态
        patterns = self.get_patterns(data, **kwargs)

        # 计算最终评分
        final_score = raw_score.iloc[-1] if not raw_score.empty else 50.0

        # 基于形态调整评分
        if not patterns.empty:
            latest_patterns = patterns.iloc[-1]

            # 正面形态加分
            if latest_patterns.get('WR_OVERSOLD', False):
                final_score += 15
            if latest_patterns.get('WR_EXTREME_OVERSOLD', False):
                final_score += 20
            if latest_patterns.get('WR_OVERSOLD_BOUNCE', False):
                final_score += 12
            if latest_patterns.get('WR_BULLISH_DIVERGENCE', False):
                final_score += 15

            # 负面形态减分
            if latest_patterns.get('WR_OVERBOUGHT', False):
                final_score -= 15
            if latest_patterns.get('WR_EXTREME_OVERBOUGHT', False):
                final_score -= 20
            if latest_patterns.get('WR_OVERBOUGHT_FALL', False):
                final_score -= 12
            if latest_patterns.get('WR_BEARISH_DIVERGENCE', False):
                final_score -= 15

        # 计算置信度
        wr_data = self.calculate_Wr_Wr(data)

        if isinstance(wr_data, dict):
            wr_value = wr_data.get('wr', pd.Series([0])).iloc[-1] if len(wr_data.get('wr', pd.Series([0]))) > 0 else 0
        else:
            wr_value = wr_data.get('wr', pd.Series([0])).iloc[-1] if len(wr_data.get('wr', pd.Series([0]))) > 0 else 0

        # 基于WR位置计算置信度
        if wr_value < -80 or wr_value > -20:
            confidence = 0.9  # 极端位置置信度高
        elif wr_value < -70 or wr_value > -30:
            confidence = 0.7
        elif wr_value < -60 or wr_value > -40:
            confidence = 0.5
        else:
            confidence = 0.3  # 中性区域置信度低

        # 限制评分范围
        final_score = max(0, min(100, final_score))

        return {
            'score': final_score,
            'confidence': confidence,
            'signals': {
                'wr_value': wr_value,
                'trend': 'up' if final_score > 60 else 'down' if final_score < 40 else 'neutral'
            }
        }

    def set_parameters(self, **kwargs):
        """真实实现：设置WR参数"""
        # 验证并设置period参数
        if 'period' in kwargs:
            period = kwargs['period']
            if isinstance(period, int) and 5 <= period <= 100:
                self.period = period
            else:
                logger.warning(f"无效的period参数: {period}, 保持原值")

        # 验证并设置overbought参数
        if 'overbought' in kwargs:
            overbought = kwargs['overbought']
            if isinstance(overbought, (int, float)) and -50 <= overbought <= 0:
                self.overbought = overbought
            else:
                logger.warning(f"无效的overbought参数: {overbought}, 保持原值")

        # 验证并设置oversold参数
        if 'oversold' in kwargs:
            oversold = kwargs['oversold']
            if isinstance(oversold, (int, float)) and -100 <= oversold <= -50:
                self.oversold = oversold
            else:
                logger.warning(f"无效的oversold参数: {oversold}, 保持原值")

        # 记录参数变更
        logger.info(f"WR参数已更新")

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：生成WR交易信号"""
        return self.get_signals(data, **kwargs)

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：计算WR指标"""
        return self.calculate_Wr_Wr(data, **kwargs)

    def calculate_confidence_Wr(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """真实实现：计算WR置信度"""
        if score.empty:
            return 0.3

        # 基础置信度
        confidence = 0.5

        # 基于评分的置信度调整
        latest_score = score.iloc[-1] if not score.empty else 50.0

        # 极端评分提高置信度
        if latest_score > 80 or latest_score < 20:
            confidence += 0.3
        elif latest_score > 70 or latest_score < 30:
            confidence += 0.2
        elif latest_score > 60 or latest_score < 40:
            confidence += 0.1

        # 基于形态的置信度调整
        if not patterns.empty:
            latest_patterns = patterns.iloc[-1]

            # 强势形态提高置信度
            if latest_patterns.get('WR_EXTREME_OVERSOLD', False):
                confidence += 0.2
            if latest_patterns.get('WR_EXTREME_OVERBOUGHT', False):
                confidence += 0.2
            if latest_patterns.get('WR_OVERSOLD_BOUNCE', False):
                confidence += 0.15
            if latest_patterns.get('WR_OVERBOUGHT_FALL', False):
                confidence += 0.15

            # 背离形态提高置信度
            if latest_patterns.get('WR_BULLISH_DIVERGENCE', False):
                confidence += 0.1
            if latest_patterns.get('WR_BEARISH_DIVERGENCE', False):
                confidence += 0.1

        # 基于信号的置信度调整
        if signals:
            signal_strength = signals.get('strength', 0)
            confidence += signal_strength * 0.1

        # 限制置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """兼容性方法：计算置信度"""
        return self.calculate_confidence_Wr(score, patterns, signals)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：识别形态"""
        return self.get_patterns(data, **kwargs)

    def calculate_raw_score_wr(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法：计算原始评分"""
        return self.calculate_raw_score(data, **kwargs)

    def register_patterns(self):
        """真实实现：注册WR形态到全局注册表"""
        try:
            registry = PatternRegistry()

            # 注册超卖形态
            registry.register_pattern_registry(
                pattern_id="WR_OVERSOLD",
                display_name="WR超卖",
                indicator_id="WR",
                pattern_type=PatternTypePatternRegistry.BULLISH,
                default_strength=PatternStrengthPatternRegistry.STRONG
            )

            # 注册超买形态
            registry.register_pattern_registry(
                pattern_id="WR_OVERBOUGHT",
                display_name="WR超买",
                indicator_id="WR",
                pattern_type=PatternTypePatternRegistry.BEARISH,
                default_strength=PatternStrengthPatternRegistry.STRONG
            )

            # 注册超卖反弹形态
            registry.register_pattern_registry(
                pattern_id="WR_OVERSOLD_BOUNCE",
                display_name="WR超卖反弹",
                indicator_id="WR",
                pattern_type=PatternTypePatternRegistry.BULLISH,
                default_strength=PatternStrengthPatternRegistry.MEDIUM
            )

            logger.info("WR形态注册完成")
            return True

        except Exception as e:
            logger.error(f"WR形态注册失败: {e}")
            return False


# 为了兼容指标注册表，创建别名
WR = WrWr
