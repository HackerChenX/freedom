#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
顺势指标(CCI)

判断价格偏离度，寻找短线机会
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class CCI(BaseIndicator):
    """
    顺势指标(CCI) (CCI)
    
    分类：震荡类指标
    描述：判断价格偏离度，寻找短线机会
    """
    
    def __init__(self, period: int = 14):
        self.REQUIRED_COLUMNS = ['high', 'low', 'close']
        """
        初始化顺势指标(CCI)指标
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__(name="CCI", description="顺势指标，判断价格偏离度")
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
            df: 要验证的DataFrame
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算顺势指标(CCI)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了CCI指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算典型价格TP (Typical Price)
        df_copy['TP'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
        
        # 计算简单移动平均MA
        df_copy['MA'] = df_copy['TP'].rolling(window=self.period).mean()
        
        # 计算偏差绝对值
        df_copy['MD'] = df_copy['TP'].rolling(window=self.period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        # 计算CCI指标
        df_copy['CCI'] = (df_copy['TP'] - df_copy['MA']) / (0.015 * df_copy['MD'])
        
        # 只保留CCI指标列
        result_df = df_copy[['CCI']].copy()

        # 保存结果
        self._result = result_df

        return result_df
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算CCI原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算CCI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        cci = self._result['CCI']
        
        # 1. 超买超卖区域评分
        # CCI < -100（超卖）+20分
        oversold_condition = cci < -100
        score += oversold_condition * 20
        
        # CCI > +100（超买）-20分
        overbought_condition = cci > 100
        score -= overbought_condition * 20
        
        # 2. CCI穿越关键位置评分
        # CCI从超卖区上穿-100+25分
        cci_cross_up_oversold = crossover(cci, -100)
        score += cci_cross_up_oversold * 25

        # CCI从超买区下穿+100-25分
        cci_cross_down_overbought = crossunder(cci, 100)
        score -= cci_cross_down_overbought * 25

        # 3. 零轴穿越评分
        # CCI上穿零轴+15分
        cci_cross_up_zero = crossover(cci, 0)
        score += cci_cross_up_zero * 15

        # CCI下穿零轴-15分
        cci_cross_down_zero = crossunder(cci, 0)
        score -= cci_cross_down_zero * 15
        
        # 4. CCI背离评分
        if len(data) >= 20:
            divergence_score = self._calculate_cci_divergence(data['close'], cci)
            score += divergence_score
        
        # 5. CCI极端值评分
        # CCI < -200（极度超卖）+30分
        extreme_oversold = cci < -200
        score += extreme_oversold * 30
        
        # CCI > +200（极度超买）-30分
        extreme_overbought = cci > 200
        score -= extreme_overbought * 30
        
        # 6. CCI趋势评分
        cci_trend_score = self._calculate_cci_trend_score(cci)
        score += cci_trend_score
        
        # 7. CCI强度评分
        cci_strength_score = self._calculate_cci_strength_score(cci)
        score += cci_strength_score
        
        return np.clip(score, 0, 100)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算CCI指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
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

        # 极端评分（超买/超卖）置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if isinstance(patterns, pd.DataFrame) and not patterns.empty:
            try:
                # 统计最近几个周期的形态数量
                numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    recent_data = patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                    recent_patterns = recent_data.sum().sum()
                    if recent_patterns > 0:
                        confidence += min(recent_patterns * 0.05, 0.2)
            except:
                pass

        # 3. 基于评分稳定性的置信度
        if len(score) >= 5:
            recent_scores = score.iloc[-5:]
            score_stability = 1.0 - (recent_scores.std() / 50.0)  # 标准差越小，稳定性越高
            confidence += score_stability * 0.1

        return min(confidence, 1.0)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别CCI技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算CCI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        cci = self._result['CCI']
        
        # 检查最近的信号
        recent_periods = min(10, len(cci))
        if recent_periods == 0:
            return patterns
        
        recent_cci = cci.tail(recent_periods)
        current_cci = recent_cci.iloc[-1]
        
        # 1. 超买超卖形态
        if current_cci <= -200:
            patterns.append("CCI极度超卖")
        elif current_cci <= -100:
            patterns.append("CCI超卖")
        elif current_cci >= 200:
            patterns.append("CCI极度超买")
        elif current_cci >= 100:
            patterns.append("CCI超买")
        
        # 2. 穿越形态
        if crossover(recent_cci, -100).any():
            patterns.append("CCI上穿超卖线")
        if crossunder(recent_cci, 100).any():
            patterns.append("CCI下穿超买线")
        if crossover(recent_cci, 0).any():
            patterns.append("CCI上穿零轴")
        if crossunder(recent_cci, 0).any():
            patterns.append("CCI下穿零轴")
        
        # 3. 背离形态
        if len(data) >= 20:
            divergence_type = self._detect_cci_divergence_pattern(data['close'], cci)
            if divergence_type:
                patterns.append(f"CCI{divergence_type}")
        
        # 4. 趋势形态
        if self._detect_cci_strong_trend(recent_cci):
            if current_cci > 0:
                patterns.append("CCI强势上升趋势")
            else:
                patterns.append("CCI强势下降趋势")
        
        # 5. 反转形态
        reversal_type = self._detect_cci_reversal_pattern(recent_cci)
        if reversal_type:
            patterns.append(f"CCI{reversal_type}")
        
        # 6. 钝化形态
        if self._detect_cci_stagnation(recent_cci, threshold=-100, periods=5, direction='low'):
            patterns.append("CCI低位钝化")
        if self._detect_cci_stagnation(recent_cci, threshold=100, periods=5, direction='high'):
            patterns.append("CCI高位钝化")
        
        return patterns
    
    def _calculate_cci_divergence(self, price: pd.Series, cci: pd.Series) -> pd.Series:
        """
        计算CCI背离评分
        
        Args:
            price: 价格序列
            cci: CCI序列
            
        Returns:
            pd.Series: 背离评分序列
        """
        divergence_score = pd.Series(0.0, index=price.index)
        
        if len(price) < 20:
            return divergence_score
        
        # 寻找价格和CCI的峰值谷值
        window = 5
        for i in range(window, len(price) - window):
            price_window = price.iloc[i-window:i+window+1]
            cci_window = cci.iloc[i-window:i+window+1]
            
            if price.iloc[i] == price_window.max():  # 价格峰值
                if cci.iloc[i] != cci_window.max():  # CCI未创新高
                    divergence_score.iloc[i:i+10] -= 25  # 负背离
            elif price.iloc[i] == price_window.min():  # 价格谷值
                if cci.iloc[i] != cci_window.min():  # CCI未创新低
                    divergence_score.iloc[i:i+10] += 25  # 正背离
        
        return divergence_score
    
    def _calculate_cci_trend_score(self, cci: pd.Series) -> pd.Series:
        """
        计算CCI趋势评分
        
        Args:
            cci: CCI序列
            
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=cci.index)
        
        if len(cci) < 5:
            return trend_score
        
        # 计算CCI斜率
        cci_slope = cci.diff(3)
        
        # 趋势评分
        trend_score += np.where(cci_slope > 20, 10, 0)   # 强烈上升+10分
        trend_score += np.where(cci_slope > 10, 5, 0)    # 温和上升+5分
        trend_score -= np.where(cci_slope < -20, 10, 0)  # 强烈下降-10分
        trend_score -= np.where(cci_slope < -10, 5, 0)   # 温和下降-5分
        
        return trend_score
    
    def _calculate_cci_strength_score(self, cci: pd.Series) -> pd.Series:
        """
        计算CCI强度评分
        
        Args:
            cci: CCI序列
            
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=cci.index)
        
        if len(cci) < 10:
            return strength_score
        
        # 计算CCI的波动强度
        cci_volatility = cci.rolling(window=10).std()
        cci_mean_volatility = cci_volatility.rolling(window=20).mean()
        
        # 高波动性+5分，低波动性-5分
        high_volatility = cci_volatility > cci_mean_volatility * 1.5
        low_volatility = cci_volatility < cci_mean_volatility * 0.5
        
        strength_score += high_volatility * 5
        strength_score -= low_volatility * 5
        
        return strength_score
    
    def _detect_cci_divergence_pattern(self, price: pd.Series, cci: pd.Series) -> Optional[str]:
        """
        检测CCI背离形态
        
        Args:
            price: 价格序列
            cci: CCI序列
            
        Returns:
            Optional[str]: 背离类型或None
        """
        if len(price) < 20:
            return None
        
        # 寻找最近的峰值和谷值
        recent_price = price.tail(20)
        recent_cci = cci.tail(20)
        
        price_extremes = []
        cci_extremes = []
        
        # 简化的极值检测
        for i in range(2, len(recent_price) - 2):
            if (recent_price.iloc[i] > recent_price.iloc[i-1] and 
                recent_price.iloc[i] > recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                cci_extremes.append(recent_cci.iloc[i])
            elif (recent_price.iloc[i] < recent_price.iloc[i-1] and 
                  recent_price.iloc[i] < recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                cci_extremes.append(recent_cci.iloc[i])
        
        if len(price_extremes) >= 2:
            price_trend = price_extremes[-1] - price_extremes[-2]
            cci_trend = cci_extremes[-1] - cci_extremes[-2]
            
            # 正背离：价格创新低但CCI未创新低
            if price_trend < -0.01 and cci_trend > 10:
                return "正背离"
            # 负背离：价格创新高但CCI未创新高
            elif price_trend > 0.01 and cci_trend < -10:
                return "负背离"
        
        return None
    
    def _detect_cci_strong_trend(self, cci: pd.Series) -> bool:
        """
        检测CCI强势趋势
        
        Args:
            cci: CCI序列
            
        Returns:
            bool: 是否为强势趋势
        """
        if len(cci) < 5:
            return False
        
        # 检测连续趋势
        recent_cci = cci.tail(5)
        
        # 连续上升或下降
        ascending = all(recent_cci.iloc[i] > recent_cci.iloc[i-1] for i in range(1, len(recent_cci)))
        descending = all(recent_cci.iloc[i] < recent_cci.iloc[i-1] for i in range(1, len(recent_cci)))
        
        # 且幅度较大
        amplitude = abs(recent_cci.iloc[-1] - recent_cci.iloc[0])
        
        return (ascending or descending) and amplitude > 50
    
    def _detect_cci_reversal_pattern(self, cci: pd.Series) -> Optional[str]:
        """
        检测CCI反转形态
        
        Args:
            cci: CCI序列
            
        Returns:
            Optional[str]: 反转类型或None
        """
        if len(cci) < 7:
            return None
        
        recent_cci = cci.tail(7)
        
        # V型反转：从极端位置快速反转
        if (recent_cci.iloc[0] < -150 and recent_cci.iloc[-1] > -50 and
            (recent_cci.iloc[-1] - recent_cci.iloc[0]) > 80):
            return "V型底部反转"
        
        # 倒V型反转：从极端位置快速反转
        if (recent_cci.iloc[0] > 150 and recent_cci.iloc[-1] < 50 and
            (recent_cci.iloc[0] - recent_cci.iloc[-1]) > 80):
            return "倒V型顶部反转"
        
        # 双底形态
        if self._detect_cci_double_bottom(recent_cci):
            return "双底形态"
        
        # 双顶形态
        if self._detect_cci_double_top(recent_cci):
            return "双顶形态"
        
        return None
    
    def _detect_cci_stagnation(self, cci: pd.Series, threshold: float, 
                              periods: int, direction: str) -> bool:
        """
        检测CCI钝化
        
        Args:
            cci: CCI序列
            threshold: 阈值
            periods: 检测周期数
            direction: 方向 ('low' 或 'high')
            
        Returns:
            bool: 是否钝化
        """
        if len(cci) < periods:
            return False
        
        recent_cci = cci.tail(periods)
        
        if direction == 'low':
            return (recent_cci < threshold).all()
        elif direction == 'high':
            return (recent_cci > threshold).all()
        
        return False
    
    def _detect_cci_double_bottom(self, cci: pd.Series) -> bool:
        """
        检测CCI双底形态
        
        Args:
            cci: CCI序列
            
        Returns:
            bool: 是否为双底形态
        """
        if len(cci) < 7:
            return False
        
        # 寻找两个相近的低点
        min_indices = []
        for i in range(1, len(cci) - 1):
            if cci.iloc[i] < cci.iloc[i-1] and cci.iloc[i] < cci.iloc[i+1]:
                min_indices.append(i)
        
        if len(min_indices) >= 2:
            # 检查最后两个低点
            last_two_mins = min_indices[-2:]
            min1_val = cci.iloc[last_two_mins[0]]
            min2_val = cci.iloc[last_two_mins[1]]
            
            # 两个低点相近且都在超卖区
            if abs(min1_val - min2_val) < 30 and min1_val < -100 and min2_val < -100:
                return True
        
        return False
    
    def _detect_cci_double_top(self, cci: pd.Series) -> bool:
        """
        检测CCI双顶形态
        
        Args:
            cci: CCI序列
            
        Returns:
            bool: 是否为双顶形态
        """
        if len(cci) < 7:
            return False
        
        # 寻找两个相近的高点
        max_indices = []
        for i in range(1, len(cci) - 1):
            if cci.iloc[i] > cci.iloc[i-1] and cci.iloc[i] > cci.iloc[i+1]:
                max_indices.append(i)
        
        if len(max_indices) >= 2:
            # 检查最后两个高点
            last_two_maxs = max_indices[-2:]
            max1_val = cci.iloc[last_two_maxs[0]]
            max2_val = cci.iloc[last_two_maxs[1]]
            
            # 两个高点相近且都在超买区
            if abs(max1_val - max2_val) < 30 and max1_val > 100 and max2_val > 100:
                return True
        
        return False
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成顺势指标(CCI)指标交易信号
        
        Args:
            df: 包含价格数据和CCI指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - cci_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['CCI']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        overbought = kwargs.get('overbought', 100)  # 超买阈值
        oversold = kwargs.get('oversold', -100)  # 超卖阈值
        
        # 初始化信号列
        df_copy['cci_signal'] = 0
        
        # CCI由超卖区上穿-100为买入信号
        df_copy.loc[crossover(df_copy['CCI'], oversold), 'cci_signal'] = 1
        
        # CCI由超买区下穿+100为卖出信号
        df_copy.loc[crossunder(df_copy['CCI'], overbought), 'cci_signal'] = -1
        
        return df_copy
        
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标并返回结果
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含计算结果的DataFrame
        """
        return self.calculate(df)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取CCI指标的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算CCI - 使用_calculate避免包含原始数据列
        calculated_data = self._calculate(data)

        if calculated_data is None or calculated_data.empty:
            return pd.DataFrame(index=data.index)

        if 'CCI' not in calculated_data.columns:
            return pd.DataFrame(index=data.index)

        cci = calculated_data['CCI']
        # 只创建包含形态的DataFrame，不包含原始数据列
        patterns_df = pd.DataFrame(index=calculated_data.index)

        # 1. 超买超卖形态
        patterns_df['CCI_OVERSOLD'] = cci <= -100
        patterns_df['CCI_EXTREME_OVERSOLD'] = cci <= -200
        patterns_df['CCI_OVERBOUGHT'] = cci >= 100
        patterns_df['CCI_EXTREME_OVERBOUGHT'] = cci >= 200

        # 2. 穿越形态（使用简化的穿越逻辑）
        if len(cci) >= 2:
            cci_prev = cci.shift(1)
            patterns_df['CCI_CROSS_UP_OVERSOLD'] = (cci_prev <= -100) & (cci > -100)
            patterns_df['CCI_CROSS_DOWN_OVERBOUGHT'] = (cci_prev >= 100) & (cci < 100)
            patterns_df['CCI_CROSS_UP_ZERO'] = (cci_prev <= 0) & (cci > 0)
            patterns_df['CCI_CROSS_DOWN_ZERO'] = (cci_prev >= 0) & (cci < 0)
        else:
            patterns_df['CCI_CROSS_UP_OVERSOLD'] = False
            patterns_df['CCI_CROSS_DOWN_OVERBOUGHT'] = False
            patterns_df['CCI_CROSS_UP_ZERO'] = False
            patterns_df['CCI_CROSS_DOWN_ZERO'] = False

        # 3. 趋势形态
        if len(cci) >= 5:
            cci_slope = cci.diff(3)
            patterns_df['CCI_STRONG_UPTREND'] = cci_slope > 20
            patterns_df['CCI_STRONG_DOWNTREND'] = cci_slope < -20

        # 4. 钝化形态
        if len(cci) >= 5:
            patterns_df['CCI_LOW_STAGNATION'] = cci.rolling(5).apply(lambda x: (x < -100).all(), raw=False)
            patterns_df['CCI_HIGH_STAGNATION'] = cci.rolling(5).apply(lambda x: (x > 100).all(), raw=False)

        # 确保所有列都是布尔类型，填充NaN为False
        for col in patterns_df.columns:
            patterns_df[col] = patterns_df[col].fillna(False).astype(bool)

        return patterns_df

    def register_patterns(self):
        """
        注册CCI指标的技术形态
        """
        # 注册CCI超买超卖形态
        self.register_pattern_to_registry(
            pattern_id="CCI_EXTREME_OVERSOLD",
            display_name="CCI极度超卖",
            description="CCI值低于-200，表示严重超卖",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_OVERSOLD",
            display_name="CCI超卖",
            description="CCI值低于-100，表示超卖",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_EXTREME_OVERBOUGHT",
            display_name="CCI极度超买",
            description="CCI值高于+200，表示严重超买",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_OVERBOUGHT",
            display_name="CCI超买",
            description="CCI值高于+100，表示超买",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册CCI穿越形态
        self.register_pattern_to_registry(
            pattern_id="CCI_CROSS_UP_OVERSOLD",
            display_name="CCI上穿超卖线",
            description="CCI从超卖区上穿-100线",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_CROSS_DOWN_OVERBOUGHT",
            display_name="CCI下穿超买线",
            description="CCI从超买区下穿+100线",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_CROSS_UP_ZERO",
            display_name="CCI上穿零轴",
            description="CCI上穿零轴线",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_CROSS_DOWN_ZERO",
            display_name="CCI下穿零轴",
            description="CCI下穿零轴线",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册CCI状态形态（从centralized mapping迁移）
        self.register_pattern_to_registry(
            pattern_id="CCI_OVERBOUGHT",
            display_name="CCI超买",
            description="CCI值高于+100，表示超买",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_OVERSOLD",
            display_name="CCI超卖",
            description="CCI值低于-100，表示超卖",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CCI_STRONG_UPTREND",
            display_name="CCI强势上升",
            description="CCI持续上升，表示强势上涨",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典
        """
        pattern_info_map = {
            'CCI_EXTREME_OVERSOLD': {
                'name': 'CCI极度超卖',
                'description': 'CCI值低于-200，表示严重超卖',
                'strength': 'strong',
                'type': 'bullish'
            },
            'CCI_OVERSOLD': {
                'name': 'CCI超卖',
                'description': 'CCI值低于-100，表示超卖',
                'strength': 'medium',
                'type': 'bullish'
            },
            'CCI_EXTREME_OVERBOUGHT': {
                'name': 'CCI极度超买',
                'description': 'CCI值高于+200，表示严重超买',
                'strength': 'strong',
                'type': 'bearish'
            },
            'CCI_OVERBOUGHT': {
                'name': 'CCI超买',
                'description': 'CCI值高于+100，表示超买',
                'strength': 'medium',
                'type': 'bearish'
            },
            'CCI_CROSS_UP_OVERSOLD': {
                'name': 'CCI上穿超卖线',
                'description': 'CCI从超卖区上穿-100线',
                'strength': 'strong',
                'type': 'bullish'
            },
            'CCI_CROSS_DOWN_OVERBOUGHT': {
                'name': 'CCI下穿超买线',
                'description': 'CCI从超买区下穿+100线',
                'strength': 'strong',
                'type': 'bearish'
            },
            'CCI_CROSS_UP_ZERO': {
                'name': 'CCI上穿零轴',
                'description': 'CCI上穿零轴线',
                'strength': 'medium',
                'type': 'bullish'
            },
            'CCI_CROSS_DOWN_ZERO': {
                'name': 'CCI下穿零轴',
                'description': 'CCI下穿零轴线',
                'strength': 'medium',
                'type': 'bearish'
            },
            'CCI_STRONG_UPTREND': {
                'name': 'CCI强势上升趋势',
                'description': 'CCI持续上升，表示强势上涨',
                'strength': 'strong',
                'type': 'bullish'
            },
            'CCI_STRONG_DOWNTREND': {
                'name': 'CCI强势下降趋势',
                'description': 'CCI持续下降，表示强势下跌',
                'strength': 'strong',
                'type': 'bearish'
            }
        }

        return pattern_info_map.get(pattern_id, {
            'name': pattern_id,
            'description': f'CCI形态: {pattern_id}',
            'strength': 'medium',
            'type': 'neutral'
        })
