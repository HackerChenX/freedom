#!/usr/bin/env python
from utils.dependency_injection import get_logger
# -*- coding: utf-8 -*-

"""
KDJ指标

随机指标KDJ，用于分析价格是否处于超买或超卖状态
"""

import numpy as np
from typing import Dict, Any
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from enum import Enum

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from indicators.common import crossover, crossunder
from indicators.pattern_registry import PatternRegistry, PatternTypePatternRegistry, PatternStrengthPatternRegistry, PatternInfo
from utils.dependency_injection import get_logger
from utils.decorators import log_calls, error_handling
from utils.technical_utils import calculate_kdj

logger = get_logger(__name__)


class KdjKdj(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    KDJ随机指标
    
    KDJ指标是RSI和随机指标的结合体，是一种超买超卖指标，用于判断股价走势的超买超卖状态。
    """
    
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        """
        初始化KDJ指标

        Args:
            n: RSV周期，默认为9
            m1: K值平滑因子，默认为3
            m2: D值平滑因子，默认为3
        """
        super().__init__()
        self.REQUIRED_COLUMNS = ['high', 'low', 'close']
        self.name = "KDJ"
        self.description = "随机指标"
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.is_available = True

    @property
    def minimum_periods(self) -> int:
        """返回计算KDJ指标所需的最小周期数"""
        return max(self.n, self.m1, self.m2) + 1

    def _register_patterns(self):
        """统一注册所有KDJ形态"""
        self.register_pattern_to_registry(
            pattern_id="KDJ_GOLDEN_CROSS",
            display_name="KDJ金叉",
            description="K线从下方突破D线，买入信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )
        self.register_pattern_to_registry(
            pattern_id="KDJ_DEATH_CROSS",
            display_name="KDJ死叉",
            description="K线从上方跌破D线，卖出信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )
        self.register_pattern_to_registry(
            pattern_id="KDJ_OVERBOUGHT",
            display_name="KDJ超买",
            description="K值高于80，超买信号",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )
        self.register_pattern_to_registry(
            pattern_id="KDJ_OVERSOLD",
            display_name="KDJ超卖",
            description="K值低于20，超卖信号",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )
        self.register_pattern_to_registry(
            pattern_id="KDJ_BULLISH_DIVERGENCE",
            display_name="KDJ看涨背离",
            description="价格创新低而KDJ未创新低，底部反转信号",
            pattern_type="REVERSAL",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )
        self.register_pattern_to_registry(
            pattern_id="KDJ_BEARISH_DIVERGENCE",
            display_name="KDJ看跌背离",
            description="价格创新高而KDJ未创新高，顶部反转信号",
            pattern_type="REVERSAL",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )
    
    def set_parameters_Kdj_Kdj_Kdj_kdj(self, **kwargs):
        """设置指标参数，可设置 'n', 'm1', 'm2'"""
        if 'n' in kwargs:
            self.n = int(kwargs['n'])
        if 'm1' in kwargs:
            self.m1 = int(kwargs['m1'])
        if 'm2' in kwargs:
            self.m2 = int(kwargs['m2'])
    
    def get_patterns_Kdj(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取KDJ指标的技术形态
        
        Args:
            data: 输入数据，通常是K线数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含形态信号的Data_frame
        """
        if 'K' not in data.columns or 'D' not in data.columns:
            calculated_data = self._calculate_kdj(data)
        else:
            calculated_data = data

        patterns_df = pd.DataFrame(index=calculated_data.index)
        
        # 交叉形态
        patterns_df['KDJ_GOLDEN_CROSS'] = self._detect_golden_cross(calculated_data['K'], calculated_data['D'])
        patterns_df['KDJ_DEATH_CROSS'] = self._detect_death_cross(calculated_data['K'], calculated_data['D'])
        
        # 超买超卖形态
        patterns_df['KDJ_OVERBOUGHT'] = self._detect_overbought(calculated_data)
        patterns_df['KDJ_OVERSOLD'] = self._detect_oversold(calculated_data)
        
        # 背离形态检测 - 这些方法返回布尔值而不是Series，需要逐行检测
        # 由于计算成本较高，只在最后一行检测背离
        if len(data) > 0:
            last_idx = data.index[-1]
            patterns_df.loc[last_idx, 'KDJ_BULLISH_DIVERGENCE'] = self._detect_bullish_divergence(calculated_data)
            patterns_df.loc[last_idx, 'KDJ_BEARISH_DIVERGENCE'] = self._detect_bearish_divergence(calculated_data)
            
            # 填充其他行为False，使用最新pandas方法避免FutureWarning
            # 先转换为bool类型，再填充False值
            patterns_df['KDJ_BULLISH_DIVERGENCE'] = patterns_df['KDJ_BULLISH_DIVERGENCE'].astype('boolean').fillna(False)
            patterns_df['KDJ_BEARISH_DIVERGENCE'] = patterns_df['KDJ_BEARISH_DIVERGENCE'].astype('boolean').fillna(False)
        
        return patterns_df

    def calculate_confidence_Kdj(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算KDJ指标的置信度。

        置信度基于以下因素：
        1.  KDJ值的位置：处于超买/超卖区的信号更可信。
        2.  J值的方向和极端性：J值是领先指标，其极端值和快速转向可以增强信心。
        3.  KDJ三线的排列形态：三线同向发散表明趋势强劲，收敛则表明犹豫。
        4.  K、D线间距：距离越大，趋势越明确。

        Args:
            score: 原始评分序列 (当前未使用)
            patterns: 形态Data_frame (当前未使用)
            signals: 信号字典 (当前未使用)

        Returns:
            float: 置信度 (0.0 - 1.0)
        """
        if not self.has_result() or len(self.result) < 5:
            return 0.5  # 数据不足，返回中性置信度

        # 获取最新的KDJ值
        latest_kdj = self.result.iloc[-1]
        k, d, j = latest_kdj['K'], latest_kdj['D'], latest_kdj['J']

        confidence = 0.5  # 基础置信度

        # 1. 位置因素: K值越极端，置信度越高
        k_extremity = abs(k - 50) / 50.0  # (0 for k=50, 1 for k=0 or 100)
        confidence += k_extremity * 0.2  # 最大贡献+0.2

        # 2. J值因素: J值是领先指标，绝对值越大，信号越明确
        j_factor = min(abs(j - 50) / 100.0, 1.0) # (0 for j=50, 1 for j>=150 or j<=-50)
        confidence += j_factor * 0.2 # 最大贡献+0.2

        # 3. 三线排列因素: K,D,J同向发散则置信度高
        if len(self.result) >= 3:
            prev_kdj = self.result.iloc[-2]
            is_uptrend = k > d and d > j and prev_kdj['K'] < k
            is_downtrend = k < d and d < j and prev_kdj['K'] > k
            if is_uptrend or is_downtrend:
                confidence += 0.15

        # 4. K、D线间距: 距离越大，趋势越明确
        kd_spread = abs(k - d)
        spread_factor = min(kd_spread / 20.0, 1.0) # 假设20是比较大的间距
        confidence += spread_factor * 0.1 # 最大贡献+0.1

        # 确保置信度在0到1之间
        return max(0.0, min(1.0, confidence))

    def calculate_score_Kdj(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        计算KDJ指标评分（0-100分制）

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            dict: 包含评分和置信度的字典
        """
        raw_score = self.calculate_raw_score_Kdj(data, **kwargs)
        patterns = self.get_patterns_Kdj(data, **kwargs)

        if raw_score.empty:
            return {
                'score': 50.0,
                'confidence': 0.5,
                'patterns': patterns,
                'raw_score': raw_score
            }

        last_score = raw_score.iloc[-1]

        # 计算置信度
        confidence = self.calculate_confidence_Kdj(raw_score, patterns, {})

        # 计算最终评分
        final_score = float(np.clip(last_score, 0, 100))

        return {
            'score': final_score,
            'confidence': confidence,
            'patterns': patterns,
            'raw_score': raw_score
        }

    def _detect_overbought(self, data: pd.DataFrame) -> pd.Series:
        """检测KDJ超买形态"""
        if 'K' not in data.columns:
            return pd.Series([False] * len(data), index=data.index)
        return data['K'] > 80.0

    def _detect_oversold(self, data: pd.DataFrame) -> pd.Series:
        """检测KDJ超卖形态"""
        if 'K' not in data.columns:
            return pd.Series([False] * len(data), index=data.index)
        return data['K'] < 20.0
    
    def _detect_robust_crossover(self, series1: pd.Series, series2: pd.Series, window: int = 3, cross_type: str = 'above') -> pd.Series:
        """
        更稳健的交叉检测，考虑交叉后的持续性
        
        Args:
            series1: 第一个序列 (例如, K线)
            series2: 第二个序列 (例如, D线)
            window: 确认交叉的窗口期
            cross_type: 'above' (金叉) 或 'below' (死叉)
            
        Returns:
            pd.Series: 交叉信号
        """
        # 初始化结果
        result = pd.Series(False, index=series1.index)
        
        # 数据不足以计算交叉
        if len(series1) < 3:
            return result
            
        # 计算前后位置关系
        if cross_type == 'above':
            # 查找从下向上穿越的点（金叉）
            # 前一点，series1低于或等于series2
            condition_before = series1.shift(1) <= series2.shift(1)
            # 当前点，series1高于series2
            condition_after = series1 > series2
            # 结合两个条件找到交叉点
            cross_points = condition_before & condition_after
        else:
            # 查找从上向下穿越的点（死叉）
            # 前一点，series1高于或等于series2
            condition_before = series1.shift(1) >= series2.shift(1)
            # 当前点，series1低于series2
            condition_after = series1 < series2
            # 结合两个条件找到交叉点
            cross_points = condition_before & condition_after
            
        # 找到所有潜在交叉点的索引
        cross_indices = np.where(cross_points)[0]
        
        # 没有发现交叉点，返回全False序列
        if len(cross_indices) == 0:
            return result
            
        # 对每个交叉点应用更严格的确认
        for idx in cross_indices:
            # 跳过开始的点，确保有前置数据
            if idx < 2:
                continue
                
            # 跳过结尾的点，确保有后续数据用于确认
            if idx >= len(series1) - window:
                continue
                
            # 确认交叉点前后的趋势方向
            if cross_type == 'above':
                # 金叉：确保交叉前series1一直低于series2，交叉后一直高于
                before_cross = series1.iloc[idx-window:idx] < series2.iloc[idx-window:idx]
                after_cross = series1.iloc[idx:idx+window] > series2.iloc[idx:idx+window]
                
                # 至少有window/2个点满足条件
                if before_cross.sum() >= window/2 and after_cross.sum() >= window/2:
                    result.iloc[idx] = True
            else:
                # 死叉：确保交叉前series1一直高于series2，交叉后一直低于
                before_cross = series1.iloc[idx-window:idx] > series2.iloc[idx-window:idx]
                after_cross = series1.iloc[idx:idx+window] < series2.iloc[idx:idx+window]
                
                # 至少有window/2个点满足条件
                if before_cross.sum() >= window/2 and after_cross.sum() >= window/2:
                    result.iloc[idx] = True
                    
        return result
    
    def _detect_golden_cross(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """检测金叉"""
        return self._detect_robust_crossover(series1, series2, window=3, cross_type='above')
    
    def _detect_death_cross(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """检测死叉"""
        return self._detect_robust_crossover(series1, series2, window=3, cross_type='below')
    
    def _detect_bullish_divergence(self, data: pd.DataFrame) -> bool:
        """检测KDJ看涨背离形态"""
        if not self.has_result() or 'K' not in data.columns or 'close' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 20:
            return False
        
        # 底背离：价格创新低，但KDJ指标未创新低
        try:
            # 获取最近20个周期的数据
            close = data['close'].iloc[-20:].values
            k_values = data['K'].iloc[-20:].values
            
            # 查找局部最小值的位置
            close_lows = []
            k_lows = []
            
            for i in range(1, len(close) - 1):
                if close[i] < close[i-1] and close[i] < close[i+1]:
                    close_lows.append((i, close[i]))
                
                if k_values[i] < k_values[i-1] and k_values[i] < k_values[i+1]:
                    k_lows.append((i, k_values[i]))
            
            # 至少需要2个低点才能形成背离
            if len(close_lows) < 2 or len(k_lows) < 2:
                return False
            
            # 排序找出最低的两个点
            close_lows.sort(key=lambda x: x[1])
            
            # 获取价格的两个最低点位置
            idx1, val1 = close_lows[0]
            idx2, val2 = close_lows[1]
            
            # 确保两个低点之间的距离足够
            if abs(idx1 - idx2) < 3:
                return False
            
            # 获取这两个时间点对应的K值
            k_val1 = k_values[idx1]
            k_val2 = k_values[idx2]
            
            # 如果价格第二个低点低于第一个低点，但K第二个低点高于第一个低点，则形成底背离
            return val2 < val1 and k_val2 > k_val1
        except Exception as e:
            logger.error(f"检测KDJ底背离形态出错: {e}")
            return False
    
    def _detect_bearish_divergence(self, data: pd.DataFrame) -> bool:
        """检测KDJ顶背离形态"""
        if not self.has_result() or 'K' not in data.columns or 'close' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 20:
            return False
        
        # 顶背离：价格创新高，但KDJ指标未创新高
        try:
            # 获取最近20个周期的数据
            close = data['close'].iloc[-20:].values
            k_values = data['K'].iloc[-20:].values
            
            # 查找局部最大值的位置
            close_highs = []
            k_highs = []
            
            for i in range(1, len(close) - 1):
                if close[i] > close[i-1] and close[i] > close[i+1]:
                    close_highs.append((i, close[i]))
                
                if k_values[i] > k_values[i-1] and k_values[i] > k_values[i+1]:
                    k_highs.append((i, k_values[i]))
            
            # 至少需要2个高点才能形成背离
            if len(close_highs) < 2 or len(k_highs) < 2:
                return False
            
            # 排序找出最高的两个点
            close_highs.sort(key=lambda x: x[1], reverse=True)
            
            # 获取价格的两个最高点位置
            idx1, val1 = close_highs[0]
            idx2, val2 = close_highs[1]
            
            # 确保两个高点之间的距离足够
            if abs(idx1 - idx2) < 3:
                return False
            
            # 获取这两个时间点对应的K值
            k_val1 = k_values[idx1]
            k_val2 = k_values[idx2]
            
            # 如果价格第二个高点高于第一个高点，但K第二个高点低于第一个高点，则形成顶背离
            return val2 > val1 and k_val2 < k_val1
        except Exception as e:
            logger.error(f"检测KDJ顶背离形态出错: {e}")
            return False
    
    def _detect_high_cross(self, data: pd.DataFrame) -> bool:
        """检测KDJ高位交叉形态"""
        if not self.has_result() or 'K' not in data.columns or 'D' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 2:
            return False
        
        # 获取最近两个周期的KD值
        k = data['K'].iloc[-2:].values
        d = data['D'].iloc[-2:].values
        
        # 高位交叉条件：K和D都大于75，且当前K在D上方，且前一周期K在D下方
        high_cross = (k[-1] > 75 and d[-1] > 75 and 
                      k[-1] > d[-1] and k[-2] <= d[-2])
        
        return high_cross
    
    def _detect_low_cross(self, data: pd.DataFrame) -> bool:
        """检测KDJ低位交叉形态"""
        if not self.has_result() or 'K' not in data.columns or 'D' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 2:
            return False
        
        # 获取最近两个周期的KD值
        k = data['K'].iloc[-2:].values
        d = data['D'].iloc[-2:].values
        
        # 低位交叉条件：K和D都小于25，且当前K在D上方，且前一周期K在D下方
        low_cross = (k[-1] < 25 and d[-1] < 25 and 
                    k[-1] > d[-1] and k[-2] <= d[-2])
        
        return low_cross
    
    def _detect_triple_cross(self, data: pd.DataFrame) -> bool:
        """检测KDJ三线交叉形态"""
        if not self.has_result() or 'K' not in data.columns or 'D' not in data.columns or 'J' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 2:
            return False
        
        # 获取最新的KDJ值
        k = data['K'].iloc[-1]
        d = data['D'].iloc[-1]
        j = data['J'].iloc[-1]
        
        # 三线交叉条件：K、D、J的值非常接近
        diff_kd = abs(k - d)
        diff_kj = abs(k - j)
        diff_dj = abs(d - j)
        
        # 如果所有差值都小于2，则认为是三线交叉
        return diff_kd < 2 and diff_kj < 2 and diff_dj < 2
    
    def _detect_j_breakthrough(self, data: pd.DataFrame) -> bool:
        """检测J值突破形态"""
        if not self.has_result() or 'J' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 5:
            return False
        
        # 获取最近5个周期的J值
        j_values = data['J'].iloc[-5:].values
        
        # J值突破条件：从负值快速上升到高于70
        if j_values[0] < 0 and j_values[-1] > 70:
            # 检查是否是持续上升
            is_rising = True
            for i in range(1, len(j_values)):
                if j_values[i] <= j_values[i-1]:
                    is_rising = False
                    break
            
            return is_rising
        
        return False
    
    def _detect_j_breakdown(self, data: pd.DataFrame) -> bool:
        """检测J线跌破关键位形态"""
        if 'J' not in data.columns or len(data) < 2:
            return False
        
        j = data['J']
        
        # J线从上方跌破0轴
        if j.iloc[-2] > 0 and j.iloc[-1] < 0:
            return True
            
        return False
    
    def calculate_Kdj(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算KDJ指标
        
        Args:
            data: 输入数据，必须包含 'high', 'low', 'close' 列
            
        Returns:
            pd.DataFrame: 包含K, D, J列的Data_frame
        """
        try:
            # 调用核心计算逻辑
            result_df = self._calculate_kdj(data)

            # 检查结果是否为DataFrame
            if not isinstance(result_df, pd.DataFrame):
                raise TypeError(f"指标 {self.name} 的_calculate方法必须返回pandas.DataFrame")

            # 保留原始基础数据列
            self._result = self._preserve_base_columns(data, result_df)
            self._error = None
            self.is_available = True

            return self._result

        except Exception as e:
            self._error = str(e)
            self.is_available = False
            return pd.DataFrame(index=data.index)

    def _calculate_kdj(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        核心计算逻辑
        """
        # 空数据处理
        if data is None or data.empty:
            logger.warning("KDJ计算: 输入数据为空")
            return pd.DataFrame(columns=['K', 'D', 'J'])

        # 检查数据长度是否足够
        min_periods = max(self.n, self.m1, self.m2) + 1
        if len(data) < min_periods:
            logger.warning(f"KDJ计算: 数据长度不足，需要至少{min_periods}个数据点，实际{len(data)}个")
            # 返回与输入数据长度相同的空结果
            result = pd.DataFrame(index=data.index)
            result['K'] = np.nan
            result['D'] = np.nan
            result['J'] = np.nan
            return result

        # 检查必需列
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"KDJ计算: 缺少必需列 {missing_cols}")
            result = pd.DataFrame(index=data.index)
            result['K'] = np.nan
            result['D'] = np.nan
            result['J'] = np.nan
            return result

        df = data.copy()

        # 计算RSV（改进边界条件处理）
        low_n = df['low'].rolling(window=self.n, min_periods=1).min()
        high_n = df['high'].rolling(window=self.n, min_periods=1).max()

        # 避免除零错误：当high_n == low_n时，RSV设为50
        denominator = high_n - low_n
        rsv = np.where(
            denominator != 0,
            (df['close'] - low_n) / denominator * 100,
            50.0  # 当分母为0时，RSV设为50（中性值）
        )
        rsv = pd.Series(rsv, index=df.index)
        rsv = rsv.fillna(50.0)  # 将NaN值也设为50

        # 使用标准SMA方法计算K, D, J（符合标准KDJ公式）
        # K值: RSV的简单移动平均
        df['K'] = rsv.rolling(window=self.m1, min_periods=1).mean()

        # D值: K值的简单移动平均
        df['D'] = df['K'].rolling(window=self.m2, min_periods=1).mean()

        # J值
        df['J'] = 3 * df['K'] - 2 * df['D']

        # 标准KDJ初始值处理：K和D的初始值通常设为50
        df['K'] = df['K'].fillna(50.0)
        df['D'] = df['D'].fillna(50.0)
        df['J'] = df['J'].fillna(50.0)

        # 确保K、D值在合理范围内（0-100）
        df['K'] = df['K'].clip(0, 100)
        df['D'] = df['D'].clip(0, 100)
        # J值可以超出0-100范围，这是正常的

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df

    def add_signals(self, data: pd.DataFrame, k_col: str = 'K', 
                   d_col: str = 'D', j_col: str = 'J') -> pd.DataFrame:
        """
        添加KDJ交易信号
        
        Args:
            data: 包含KDJ指标的Data_frame
            k_col: K值列名
            d_col: D值列名
            j_col: J值列名
            
        Returns:
            pd.DataFrame: 添加了信号的Data_frame
        """
        result = data.copy()
        
        # 计算超买超卖信号
        result['kdj_overbought'] = (result[k_col] > 80) & (result[d_col] > 80)
        result['kdj_oversold'] = (result[k_col] < 20) & (result[d_col] < 20)
        
        # 计算金叉和死叉信号
        result['kdj_buy_signal'] = self.get_buy_signal(result, k_col, d_col)
        result['kdj_sell_signal'] = self.get_sell_signal(result, k_col, d_col)
        
        # 计算J值超买超卖
        result['kdj_j_overbought'] = result[j_col] > 100
        result['kdj_j_oversold'] = result[j_col] < 0
        
        # 计算KDJ三线同向（顺势信号）
        result['kdj_uptrend'] = (result[j_col] > result[k_col]) & (result[k_col] > result[d_col])
        result['kdj_downtrend'] = (result[j_col] < result[k_col]) & (result[k_col] < result[d_col])
        
        return result
    
    def get_buy_signal(self, data: pd.DataFrame, k_col: str = 'K', d_col: str = 'D') -> pd.Series:
        """
        获取KDJ买入信号
        
        Args:
            data: 包含KDJ指标的Data_frame
            k_col: K值列名
            d_col: D值列名
            
        Returns:
            pd.Series: 买入信号序列（布尔值）
        """
        # KDJ金叉：K线从下方穿过D线，且处于低位（<30）
        golden_cross = (data[k_col] > data[d_col]) & (data[k_col].shift(1) <= data[d_col].shift(1))
        low_position = data[k_col] < 30
        
        return golden_cross & low_position
    
    def get_sell_signal(self, data: pd.DataFrame, k_col: str = 'K', d_col: str = 'D') -> pd.Series:
        """
        获取KDJ卖出信号
        
        Args:
            data: 包含KDJ指标的Data_frame
            k_col: K值列名
            d_col: D值列名
            
        Returns:
            pd.Series: 卖出信号序列（布尔值）
        """
        # KDJ死叉：K线从上方穿过D线，且处于高位（>70）
        death_cross = (data[k_col] < data[d_col]) & (data[k_col].shift(1) >= data[d_col].shift(1))
        high_position = data[k_col] > 70
        
        return death_cross & high_position
    
    def get_column_name(self, suffix: str = "") -> str:
        """
        获取指标列名
        
        Args:
            suffix: 列名后缀
            
        Returns:
            str: 指标列名
        """
        if suffix:
            return f"KDJ_{suffix}"
        return "KDJ_Kdj"
    
    def to_dict_kdj(self) -> Dict[str, Any]:
        """
        将指标转换为字典表示
        
        Returns:
            Dict[str, Any]: 指标的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
                'n': self.n,
                'm1': self.m1,
            'm2': self.m2,
            'market_environment': self._market_environment,
            'has_result': self.has_result(),
            'has_error': self.has_error()
        }
    
    def get_indicator_type(self) -> str:
        """获取指标类型"""
        return "KDJ_Kdj"
    
    def generate_trading_signals_Kdj(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据，包含价格和指标数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典
        """
        if not self.has_result():
            self.calculate_Kdj(data)

        result = self._result

        if result is None or result.empty:
            return {
                'buy_signal': False,
                'sell_signal': False,
                'overbought': False,
                'oversold': False
            }

        signals = {}
        
        # 基本信号
        signals['buy_signal'] = result['kdj_buy_signal'] if 'kdj_buy_signal' in result.columns else pd.Series(False, index=result.index)
        signals['sell_signal'] = result['kdj_sell_signal'] if 'kdj_sell_signal' in result.columns else pd.Series(False, index=result.index)
        
        # 超买超卖信号
        signals['overbought'] = result['kdj_overbought'] if 'kdj_overbought' in result.columns else pd.Series(False, index=result.index)
        signals['oversold'] = result['kdj_oversold'] if 'kdj_oversold' in result.columns else pd.Series(False, index=result.index)
        
        # 趋势信号
        signals['bull_trend'] = result['kdj_uptrend'] if 'kdj_uptrend' in result.columns else pd.Series(False, index=result.index)
        signals['bear_trend'] = result['kdj_downtrend'] if 'kdj_downtrend' in result.columns else pd.Series(False, index=result.index)
        
        # J值超买超卖信号
        signals['j_overbought'] = result['kdj_j_overbought'] if 'kdj_j_overbought' in result.columns else pd.Series(False, index=result.index)
        signals['j_oversold'] = result['kdj_j_oversold'] if 'kdj_j_oversold' in result.columns else pd.Series(False, index=result.index)
        
        # 信号强度
        signals['buy_strength'] = self._calculate_signal_strength(result, is_buy=True)
        signals['sell_strength'] = self._calculate_signal_strength(result, is_buy=False)
        
        return signals
    
    def _calculate_signal_strength(self, data: pd.DataFrame, is_buy: bool = True) -> pd.Series:
        """
        计算信号强度
        
        Args:
            data: 输入数据
            is_buy: 是否为买入信号
            
        Returns:
            pd.Series: 信号强度序列，值范围1-5
        """
        k = data['K']
        d = data['D']
        j = data['J']
        
        # 初始强度为0
        strength = pd.Series(0, index=data.index)
        
        if is_buy:
            # 买入信号强度
            # 1. 超卖区间加分
            strength = np.where(k < 20, strength + 1, strength)
            strength = np.where(k < 10, strength + 1, strength)
            
            # 2. KDJ三线同向上升加分
            three_line_up = (j > k) & (k > d)
            strength = np.where(three_line_up, strength + 1, strength)
            
            # 3. 金叉信号加分
            golden_cross = (k > d) & (k.shift(1) <= d.shift(1))
            strength = np.where(golden_cross, strength + 1, strength)
            
            # 4. J值极低加分
            strength = np.where(j < 0, strength + 1, strength)
            
        else:
            # 卖出信号强度
            # 1. 超买区间加分
            strength = np.where(k > 80, strength + 1, strength)
            strength = np.where(k > 90, strength + 1, strength)
            
            # 2. KDJ三线同向下降加分
            three_line_down = (j < k) & (k < d)
            strength = np.where(three_line_down, strength + 1, strength)
            
            # 3. 死叉信号加分
            death_cross = (k < d) & (k.shift(1) >= d.shift(1))
            strength = np.where(death_cross, strength + 1, strength)
            
            # 4. J值极高加分
            strength = np.where(j > 100, strength + 1, strength)
        
        # 确保强度在1-5范围内
        return pd.Series(np.clip(strength, 1, 5), index=data.index)
    
    def identify_patterns_Kdj(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        识别KDJ指标的所有技术形态
        
        Args:
            data: 包含价格和KDJ指标的Data_frame
            
        Returns:
            pd.DataFrame: 包含所有已识别形态的Data_frame，每列代表一种形态
        """
        # 确保KDJ值已计算
        if not all(col in data.columns for col in ['K', 'D', 'J']):
             data = self.calculate_Kdj(data)

        patterns = pd.DataFrame(index=data.index)
        
        # 使用已注册的检测函数
        from indicators.pattern_registry import get_pattern_registry
        registry = get_pattern_registry()
        kdj_patterns = registry.get_patterns_by_indicator('KDJ_Kdj')
        
        for pattern_id in kdj_patterns:
            pattern_info = registry.get_pattern(pattern_id)
            if pattern_info and 'detection_function' in pattern_info and callable(pattern_info['detection_function']):
                try:
                    # 调用检测函数
                    detection_result = pattern_info['detection_function'](data)
                    
                    # 将结果添加到DataFrame
                    if isinstance(detection_result, pd.Series):
                        patterns[pattern_info['display_name']] = detection_result
                    elif isinstance(detection_result, bool):
                        # 如果是布尔值，则在最后一行标记
                        if detection_result:
                            patterns.loc[patterns.index[-1], pattern_info['display_name']] = True
                            
                except Exception as e:
                    logger.warning(f"为形态 '{pattern_info['display_name']}' 调用检测函数失败: {e}")

        return patterns
    
    def calculate_raw_score_Kdj(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算KDJ指标的原始评分（0-100分制）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列，取值范围0-100
        """
        if not self.has_result():
            self.calculate_Kdj(data, **kwargs)
        
        # 获取KDJ指标值
        if self._result is None or not all(col in self._result.columns for col in ['K', 'D', 'J']):
            return pd.Series(50.0, index=data.index)

        k = self._result['K']
        d = self._result['D']
        j = self._result['J']
        
        # 基础评分计算
        # 1. 位置分：基于K值的位置（0-100），贡献40分权重
        position_score = k.copy()
        # 调整曲线，使得中间值(50)得分为50分，两端得分递减
        position_score = 50 - 40 * np.abs(position_score - 50) / 50
        # 在超买超卖区域有所调整
        position_score = np.where(k <= 20, 40 + (20 - k) * 1.5, position_score)  # 超卖区加分
        position_score = np.where(k >= 80, 40 - (k - 80) * 1.5, position_score)  # 超买区减分
        
        # 2. 趋势分：基于K和D的变化趋势，贡献30分权重
        k_trend = k - k.shift(3)
        d_trend = d - d.shift(3)
        trend_score = 50 + (k_trend + d_trend) * 3  # 上升加分，下降减分
        # 限制在0-100范围内
        trend_score = np.clip(trend_score, 0, 100)
        
        # 3. 金叉死叉分：检测金叉死叉情况，贡献20分权重
        golden_cross = (k > d) & (k.shift(1) <= d.shift(1))
        death_cross = (k < d) & (k.shift(1) >= d.shift(1))
        # 初始化交叉得分为50分（中性）
        cross_score = pd.Series(50.0, index=data.index)
        
        # 金叉加分，最近越近影响越大
        for i in range(5):
            mask = golden_cross.shift(i).fillna(False).astype(bool)
            score_boost = 30 * (0.8 ** i)  # 随距离衰减
            cross_score = np.where(mask, 50 + score_boost, cross_score)

        # 死叉减分，最近越近影响越大
        for i in range(5):
            mask = death_cross.shift(i).fillna(False).astype(bool)
            score_drop = 30 * (0.8 ** i)  # 随距离衰减
            cross_score = np.where(mask, 50 - score_drop, cross_score)
        
        # 4. J值影响分：J值对评分的调整，贡献10分权重
        j_score = 50 + (j - 50) * 0.2  # J值越高分数越高
        j_score = np.clip(j_score, 0, 100)
        
        # 合并各部分得分，按权重加权平均
        raw_score = (
            position_score * 0.4 +  # 位置分权重40%
            trend_score * 0.3 +     # 趋势分权重30%
            cross_score * 0.2 +     # 金叉死叉分权重20%
            j_score * 0.1           # J值影响分权重10%
        )
        
        # 最后，检测形态对评分的额外影响
        patterns = self.get_patterns_Kdj(data, **kwargs)
        
        # 形态影响分数：最多调整15分
        pattern_adjustment = pd.Series(0.0, index=data.index)
        
        # 使用PatternRegistry获取模式信息
        from indicators.pattern_registry import get_pattern_registry
        registry = get_pattern_registry()
        
        # 遍历所有检测到的形态
        for pattern_col in patterns.columns:
            # 获取模式信息
            pattern_info = registry.get_pattern(pattern_col)
            if pattern_info and 'score_impact' in pattern_info:
                score_impact = pattern_info['score_impact']
                # 对于每个时间点，如果形态存在，则应用调整
                for idx in patterns.index:
                    if patterns.at[idx, pattern_col]:
                        pattern_adjustment.at[idx] += np.clip(score_impact, -15, 15)
        
        # 应用形态调整（最多±15分）
        pattern_adjustment = np.clip(pattern_adjustment, -15, 15)
        raw_score += pattern_adjustment
        
        # 确保最终分数在0-100范围内
        final_score = np.clip(raw_score, 0, 100)
        
        return pd.Series(final_score, index=data.index)
    
    def _detect_stagnation(self, k: pd.Series, d: pd.Series, j: pd.Series, 
                          low_threshold: float = None, high_threshold: float = None, 
                          periods: int = 5) -> pd.Series:
        """
        检测KDJ指标的钝化现象（高位或低位的KDJ三线趋于收敛）
        
        Args:
            k: K值序列
            d: D值序列
            j: J值序列
            low_threshold: 低位阈值，如果指定则检测低位钝化
            high_threshold: 高位阈值，如果指定则检测高位钝化
            periods: 检测周期
            
        Returns:
            pd.Series: 钝化信号序列
        """
        # 计算三线的标准差，标准差降低表示线间距离缩小，即趋于收敛
        stds = pd.DataFrame({'K': k, 'D': d, 'J': j}).std(axis=1)
        
        # 计算标准差的变化率，负值表示标准差下降，即收敛
        std_change = stds.pct_change(periods)
        
        # 初始化钝化信号
        stagnation = pd.Series(False, index=k.index)
        
        # 检测高位钝化
        if high_threshold is not None:
            high_position = (k > high_threshold) | (d > high_threshold) | (j > high_threshold*1.2)
            high_converge = std_change < -0.2  # 标准差下降超过20%
            stagnation = stagnation | (high_position & high_converge)
            
        # 检测低位钝化
        if low_threshold is not None:
            low_position = (k < low_threshold) | (d < low_threshold) | (j < low_threshold*0.8)
            low_converge = std_change < -0.2  # 标准差下降超过20%
            stagnation = stagnation | (low_position & low_converge)
        
        return stagnation
    
    def _detect_divergence(self, price: pd.Series, k: pd.Series) -> Optional[str]:
        """
        检测KDJ与价格的背离
        
        Args:
            price: 价格序列
            k: K值序列
            
        Returns:
            Optional[str]: 背离类型，"bullish"表示底背离，"bearish"表示顶背离，None表示无背离
        """
        # 至少需要20个数据点才能进行背离分析
        if len(price) < 20 or len(k) < 20:
            return None
        
        # 找出最近的两个低点
        min_periods = 5
        
        # 使用移动窗口找出局部最低点和最高点
        price_min = price.rolling(window=min_periods*2+1, center=True).min()
        price_max = price.rolling(window=min_periods*2+1, center=True).max()
        k_min = k.rolling(window=min_periods*2+1, center=True).min()
        k_max = k.rolling(window=min_periods*2+1, center=True).max()
        
        # 定义价格和指标的低点/高点
        price_lows = (price == price_min)
        price_highs = (price == price_max)
        k_lows = (k == k_min)
        k_highs = (k == k_max)
        
        # 找出最近30个周期内的低点和高点
        recent = slice(-30, None)
        recent_price_lows = np.where(price_lows.iloc[recent])[0]
        recent_price_highs = np.where(price_highs.iloc[recent])[0]
        recent_k_lows = np.where(k_lows.iloc[recent])[0]
        recent_k_highs = np.where(k_highs.iloc[recent])[0]
        
        # 检查是否有足够的低点/高点
        if len(recent_price_lows) >= 2 and len(recent_k_lows) >= 2:
            # 取最近的两个低点
            last_price_low, prev_price_low = recent_price_lows[-1], recent_price_lows[-2]
            last_k_low, prev_k_low = recent_k_lows[-1], recent_k_lows[-2]
            
            # 检查底背离：价格创新低但KDJ不创新低
            if (price.iloc[recent][last_price_low] < price.iloc[recent][prev_price_low] and 
                k.iloc[recent][last_k_low] > k.iloc[recent][prev_k_low]):
                return "bullish"
                
        if len(recent_price_highs) >= 2 and len(recent_k_highs) >= 2:
            # 取最近的两个高点
            last_price_high, prev_price_high = recent_price_highs[-1], recent_price_highs[-2]
            last_k_high, prev_k_high = recent_k_highs[-1], recent_k_highs[-2]
            
            # 检查顶背离：价格创新高但KDJ不创新高
            if (price.iloc[recent][last_price_high] > price.iloc[recent][prev_price_high] and 
                k.iloc[recent][last_k_high] < k.iloc[recent][prev_k_high]):
                return "bearish"
        
        return None
    
    def _calculate_divergence_strength(self, price: pd.Series, indicator: pd.Series, 
                                     is_bullish: bool = True, lookback: int = 20) -> float:
        """
        计算背离强度
        
        Args:
            price: 价格序列
            indicator: 指标序列(通常是K线)
            is_bullish: 是否为底背离
            lookback: 回溯周期数
            
        Returns:
            float: 背离强度(0-100)
        """
        if len(price) < lookback or len(indicator) < lookback:
            return 50.0
        
        # 截取回溯窗口的数据
        price_window = price.iloc[-lookback:]
        indicator_window = indicator.iloc[-lookback:]
        
        if is_bullish:
            # 底背离: 寻找价格和指标的低点
            price_min_idx = price_window.idxmin()
            indicator_min_idx = indicator_window.idxmin()
            
            # 如果最低点不一致，说明存在背离
            if price_min_idx != indicator_min_idx:
                # 计算价格新低的程度
                price_latest_min = price_window.iloc[-5:].min()
                price_prev_min = price_window.iloc[:-5].min()
                price_decline = (price_prev_min - price_latest_min) / price_prev_min if price_prev_min > 0 else 0
                
                # 计算指标背离的程度
                indicator_latest_min = indicator_window.iloc[-5:].min()
                indicator_prev_min = indicator_window.iloc[:-5].min()
                indicator_improve = max(0, indicator_latest_min - indicator_prev_min)
                
                # 底背离在KDJ超卖区效果更好
                weight = 1.5 if indicator_latest_min < 20 else 1.0
                
                # 综合评分: 价格下跌越多，指标改善越明显，评分越高
                return min(100, 50 + price_decline * 100 * weight + indicator_improve * 10)
        else:
            # 顶背离: 寻找价格和指标的高点
            price_max_idx = price_window.idxmax()
            indicator_max_idx = indicator_window.idxmax()
            
            # 如果最高点不一致，说明存在背离
            if price_max_idx != indicator_max_idx:
                # 计算价格新高的程度
                price_latest_max = price_window.iloc[-5:].max()
                price_prev_max = price_window.iloc[:-5].max()
                price_rise = (price_latest_max - price_prev_max) / price_prev_max if price_prev_max > 0 else 0
                
                # 计算指标背离的程度
                indicator_latest_max = indicator_window.iloc[-5:].max()
                indicator_prev_max = indicator_window.iloc[:-5].max()
                indicator_weaken = max(0, indicator_prev_max - indicator_latest_max)
                
                # 顶背离在KDJ超买区效果更好
                weight = 1.5 if indicator_latest_max > 80 else 1.0
                
                # 综合评分: 价格上涨越多，指标减弱越明显，评分越高
                return min(100, 50 + price_rise * 100 * weight + indicator_weaken * 10)
        
        return 50.0

    def get_pattern_info_Kdj(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典，包含name, description, strength等
        """
        pattern_info_map = {
            'KDJ_GOLDEN_CROSS': {
                'name': 'KDJ金叉',
                'description': 'K线上穿D线，表示买入信号',
                'strength': 'medium',
                'type': 'bullish'
            },
            'KDJ_DEATH_CROSS': {
                'name': 'KDJ死叉',
                'description': 'K线下穿D线，表示卖出信号',
                'strength': 'medium',
                'type': 'bearish'
            },
            'KDJ_OVERSOLD': {
                'name': 'KDJ超卖',
                'description': 'KDJ值低于20，表示超卖状态',
                'strength': 'strong',
                'type': 'bullish'
            },
            'KDJ_OVERBOUGHT': {
                'name': 'KDJ超买',
                'description': 'KDJ值高于80，表示超买状态',
                'strength': 'strong',
                'type': 'bearish'
            },
            'KDJ_BULLISH_DIVERGENCE': {
                'name': 'KDJ牛市背离',
                'description': '价格创新低而KDJ不创新低，表示看涨信号',
                'strength': 'strong',
                'type': 'bullish'
            },
            'KDJ_BEARISH_DIVERGENCE': {
                'name': 'KDJ熊市背离',
                'description': '价格创新高而KDJ不创新高，表示看跌信号',
                'strength': 'strong',
                'type': 'bearish'
            }
        }

        return pattern_info_map.get(pattern_id, {
            'name': pattern_id,
            'description': f'KDJ形态: {pattern_id}',
            'strength': 'medium',
            'type': 'neutral'
        })

    def register_patterns_Kdj(self):
        """
        注册KDJ指标的形态到全局形态注册表
        """
        # 注册KDJ金叉形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_GOLDEN_CROSS",
            display_name="KDJ金叉",
            description="K线上穿D线形成金叉，表明短期动量转强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        # 注册KDJ死叉形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_DEATH_CROSS",
            display_name="KDJ死叉",
            description="K线下穿D线形成死叉，表明短期动量转弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册KDJ超买形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_OVERBOUGHT",
            display_name="KDJ超买",
            description="KDJ值超过80，进入超买区域，需警惕回调风险",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册KDJ超卖形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_OVERSOLD",
            display_name="KDJ超卖",
            description="KDJ值低于20，进入超卖区域，存在反弹机会",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        # 注册KDJ底背离形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_BULLISH_DIVERGENCE",
            display_name="KDJ底背离",
            description="价格创新低而KDJ未创新低，形成底背离",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        # 注册KDJ顶背离形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_BEARISH_DIVERGENCE",
            display_name="KDJ顶背离",
            description="价格创新高而KDJ未创新高，形成顶背离",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'n': 9,
            'm1': 3,
            'm2': 3,
            'k_period': 9,
            'k_slowing': 3,
            'd_period': 3
        }
    


    # ==================== 抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        return self.calculate_Kdj(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        return self.calculate_raw_score_Kdj(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        return self.get_patterns_Kdj(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        # 只更新_parameters字典，避免直接设置只读属性
        for key, value in kwargs.items():
            if key in self._parameters:
                self._parameters[key] = value
            elif key in ['n', 'm1', 'm2', 'k_period', 'k_slowing', 'd_period']:
                # 对于核心参数，即使不在_parameters中也要添加
                self._parameters[key] = value

        # 如果设置了核心参数，需要重新初始化内部状态
        if any(key in kwargs for key in ['n', 'm1', 'm2']):
            # 重新设置内部参数
            self._n = self._parameters.get('n', 9)
            self._m1 = self._parameters.get('m1', 3)
            self._m2 = self._parameters.get('m2', 3)

    def set_parameters(self, **kwargs):
        """标准参数设置方法"""
        return self.set_parameters_Indicator_Base_Indicator(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """抽象基类要求的置信度计算方法"""
        return self.calculate_confidence_Kdj(score, patterns, signals)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """统一的计算接口"""
        return self.calculate_Kdj(data, **kwargs)

    # ==================== 兼容性方法 - 真实实现 ====================

    def get_patterns(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """真实实现：获取KDJ形态"""
        if data is None or data.empty:
            return pd.DataFrame()

        # 首先计算KDJ指标
        kdj_data = self.calculate_Kdj(data)

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 获取KDJ数据
        k_values = kdj_data['K']
        d_values = kdj_data['D']
        j_values = kdj_data['J']

        # 1. 金叉形态 (K线上穿D线)
        patterns_df['KDJ_GOLDEN_CROSS'] = crossover(k_values, d_values)

        # 2. 死叉形态 (K线下穿D线)
        patterns_df['KDJ_DEATH_CROSS'] = crossunder(k_values, d_values)

        # 3. 超买形态 (K>80, D>80)
        patterns_df['KDJ_OVERBOUGHT'] = (k_values > 80) & (d_values > 80)

        # 4. 超卖形态 (K<20, D<20)
        patterns_df['KDJ_OVERSOLD'] = (k_values < 20) & (d_values < 20)

        # 5. J值极端超买 (J>100)
        patterns_df['KDJ_J_EXTREME_OVERBOUGHT'] = j_values > 100

        # 6. J值极端超卖 (J<0)
        patterns_df['KDJ_J_EXTREME_OVERSOLD'] = j_values < 0

        # 7. 顶背离形态
        if len(data) >= 20:
            # 简化的背离检测：价格创新高但KDJ未创新高
            price_high = data['high'].rolling(10).max()
            kdj_high = k_values.rolling(10).max()
            patterns_df['KDJ_BEARISH_DIVERGENCE'] = (
                (data['high'] >= price_high.shift(1)) &
                (k_values < kdj_high.shift(1)) &
                (k_values > 70)
            )

            # 底背离形态：价格创新低但KDJ未创新低
            price_low = data['low'].rolling(10).min()
            kdj_low = k_values.rolling(10).min()
            patterns_df['KDJ_BULLISH_DIVERGENCE'] = (
                (data['low'] <= price_low.shift(1)) &
                (k_values > kdj_low.shift(1)) &
                (k_values < 30)
            )

        return patterns_df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """真实实现：计算KDJ原始评分"""
        if data.empty:
            return pd.Series(dtype=float)

        # 计算KDJ指标
        kdj_data = self.calculate_Kdj(data)

        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分

        # 获取KDJ数据
        k_values = kdj_data['K']
        d_values = kdj_data['D']
        j_values = kdj_data['J']

        # 1. 基于KDJ位置的评分
        # 超卖区域加分 (K<30, D<30)
        oversold_condition = (k_values < 30) & (d_values < 30)
        score += oversold_condition * 15

        # 超买区域减分 (K>70, D>70)
        overbought_condition = (k_values > 70) & (d_values > 70)
        score -= overbought_condition * 15

        # 2. 基于交叉信号的评分
        golden_cross = crossover(k_values, d_values)
        death_cross = crossunder(k_values, d_values)

        # 金叉加分，特别是在超卖区域
        score += golden_cross * 20
        score += (golden_cross & (k_values < 50)) * 10  # 低位金叉额外加分

        # 死叉减分，特别是在超买区域
        score -= death_cross * 20
        score -= (death_cross & (k_values > 50)) * 10  # 高位死叉额外减分

        # 3. 基于J值的评分
        # J值极端超卖时加分 (J<0)
        score += (j_values < 0) * 12

        # J值极端超买时减分 (J>100)
        score -= (j_values > 100) * 12

        # 4. 基于趋势的评分
        # K值上升趋势加分
        k_rising = k_values > k_values.shift(1)
        score += k_rising * 5

        # D值上升趋势加分
        d_rising = d_values > d_values.shift(1)
        score += d_rising * 3

        # 限制评分在0-100之间
        return score.clip(0, 100)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：生成KDJ交易信号"""
        if data.empty:
            return pd.DataFrame()

        # 计算KDJ指标
        kdj_data = self.calculate_Kdj(data)
        result_df = data.copy()

        # 合并KDJ数据
        for col in kdj_data.columns:
            result_df[col] = kdj_data[col]

        # 初始化信号列
        result_df['kdj_signal'] = 0
        result_df['kdj_strength'] = 0.0
        result_df['kdj_confidence'] = 0.0

        # 获取KDJ数据
        k_values = kdj_data['K']
        d_values = kdj_data['D']
        j_values = kdj_data['J']

        # 1. 金叉买入信号
        golden_cross = crossover(k_values, d_values)
        result_df.loc[golden_cross, 'kdj_signal'] = 1

        # 根据位置调整信号强度
        low_position_golden = golden_cross & (k_values < 50)
        result_df.loc[low_position_golden, 'kdj_strength'] = 0.8
        result_df.loc[low_position_golden, 'kdj_confidence'] = 0.9

        high_position_golden = golden_cross & (k_values >= 50)
        result_df.loc[high_position_golden, 'kdj_strength'] = 0.5
        result_df.loc[high_position_golden, 'kdj_confidence'] = 0.6

        # 2. 死叉卖出信号
        death_cross = crossunder(k_values, d_values)
        result_df.loc[death_cross, 'kdj_signal'] = -1

        # 根据位置调整信号强度
        high_position_death = death_cross & (k_values > 50)
        result_df.loc[high_position_death, 'kdj_strength'] = 0.8
        result_df.loc[high_position_death, 'kdj_confidence'] = 0.9

        low_position_death = death_cross & (k_values <= 50)
        result_df.loc[low_position_death, 'kdj_strength'] = 0.5
        result_df.loc[low_position_death, 'kdj_confidence'] = 0.6

        # 3. 极端超买超卖信号
        extreme_oversold = j_values < 0
        result_df.loc[extreme_oversold, 'kdj_signal'] = 1
        result_df.loc[extreme_oversold, 'kdj_strength'] = 0.9
        result_df.loc[extreme_oversold, 'kdj_confidence'] = 0.8

        extreme_overbought = j_values > 100
        result_df.loc[extreme_overbought, 'kdj_signal'] = -1
        result_df.loc[extreme_overbought, 'kdj_strength'] = 0.9
        result_df.loc[extreme_overbought, 'kdj_confidence'] = 0.8

        return result_df

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> dict:
        """真实实现：计算KDJ综合评分"""
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
            if latest_patterns.get('KDJ_GOLDEN_CROSS', False):
                final_score += 15
            if latest_patterns.get('KDJ_OVERSOLD', False):
                final_score += 10
            if latest_patterns.get('KDJ_BULLISH_DIVERGENCE', False):
                final_score += 12

            # 负面形态减分
            if latest_patterns.get('KDJ_DEATH_CROSS', False):
                final_score -= 15
            if latest_patterns.get('KDJ_OVERBOUGHT', False):
                final_score -= 10
            if latest_patterns.get('KDJ_BEARISH_DIVERGENCE', False):
                final_score -= 12

        # 计算置信度
        kdj_data = self.calculate_Kdj(data)
        k_value = kdj_data['K'].iloc[-1] if not kdj_data.empty else 50.0
        d_value = kdj_data['D'].iloc[-1] if not kdj_data.empty else 50.0

        # 基于KDJ位置计算置信度
        if abs(k_value - d_value) > 10:  # K和D差距较大
            confidence = 0.8
        elif abs(k_value - d_value) > 5:
            confidence = 0.6
        else:
            confidence = 0.4

        # 极端位置提高置信度
        if k_value < 20 or k_value > 80:
            confidence = min(1.0, confidence + 0.2)

        # 限制评分范围
        final_score = max(0, min(100, final_score))

        return {
            'score': final_score,
            'confidence': confidence,
            'signals': {
                'k_value': k_value,
                'd_value': d_value,
                'trend': 'up' if final_score > 60 else 'down' if final_score < 40 else 'neutral'
            }
        }

    def set_parameters(self, **kwargs):
        """真实实现：设置KDJ参数"""
        # 验证并设置n参数
        if 'n' in kwargs:
            n = kwargs['n']
            if isinstance(n, int) and 1 <= n <= 50:
                self.n = n
            else:
                logger.warning(f"无效的n参数: {n}, 保持原值: {self.n}")

        # 验证并设置m1参数
        if 'm1' in kwargs:
            m1 = kwargs['m1']
            if isinstance(m1, int) and 1 <= m1 <= 10:
                self.m1 = m1
            else:
                logger.warning(f"无效的m1参数: {m1}, 保持原值: {self.m1}")

        # 验证并设置m2参数
        if 'm2' in kwargs:
            m2 = kwargs['m2']
            if isinstance(m2, int) and 1 <= m2 <= 10:
                self.m2 = m2
            else:
                logger.warning(f"无效的m2参数: {m2}, 保持原值: {self.m2}")

        # 记录参数变更
        logger.info(f"KDJ参数已更新: n={self.n}, m1={self.m1}, m2={self.m2}")

    def register_patterns(self):
        """真实实现：注册KDJ形态到全局注册表"""
        try:
            registry = PatternRegistry()

            # 注册金叉形态
            registry.register_pattern_registry(
                pattern_id="KDJ_GOLDEN_CROSS",
                display_name="KDJ金叉",
                indicator_id="KDJ",
                pattern_type=PatternTypePatternRegistry.BULLISH,
                default_strength=PatternStrengthPatternRegistry.STRONG
            )

            # 注册死叉形态
            registry.register_pattern_registry(
                pattern_id="KDJ_DEATH_CROSS",
                display_name="KDJ死叉",
                indicator_id="KDJ",
                pattern_type=PatternTypePatternRegistry.BEARISH,
                default_strength=PatternStrengthPatternRegistry.STRONG
            )

            # 注册超买形态
            registry.register_pattern_registry(
                pattern_id="KDJ_OVERBOUGHT",
                display_name="KDJ超买",
                indicator_id="KDJ",
                pattern_type=PatternTypePatternRegistry.BEARISH,
                default_strength=PatternStrengthPatternRegistry.MEDIUM
            )

            # 注册超卖形态
            registry.register_pattern_registry(
                pattern_id="KDJ_OVERSOLD",
                display_name="KDJ超卖",
                indicator_id="KDJ",
                pattern_type=PatternTypePatternRegistry.BULLISH,
                default_strength=PatternStrengthPatternRegistry.MEDIUM
            )

            logger.info("KDJ形态注册完成")
            return True

        except Exception as e:
            logger.error(f"KDJ形态注册失败: {e}")
            return False

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：生成KDJ交易信号"""
        return self.get_signals(data, **kwargs)

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """真实实现：计算KDJ指标"""
        return self.calculate_Kdj(data, **kwargs)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """兼容性方法：计算置信度"""
        return self.calculate_confidence_Kdj(score, patterns, signals)

    def _detect_robust_crossover(self, series1: pd.Series, series2: pd.Series, window: int = 2, cross_type: str = 'above') -> pd.Series:
        """真实实现：鲁棒交叉检测"""
        if len(series1) < window + 1 or len(series2) < window + 1:
            return pd.Series(False, index=series1.index)

        result = pd.Series(False, index=series1.index)

        for i in range(window, len(series1)):
            if cross_type == 'above':
                # 检查是否从下方穿越到上方，并且保持一定时间
                before_cross = all(series1.iloc[i-window:i] <= series2.iloc[i-window:i])
                after_cross = all(series1.iloc[i:i+1] > series2.iloc[i:i+1])

                if before_cross and after_cross:
                    result.iloc[i] = True

            elif cross_type == 'below':
                # 检查是否从上方穿越到下方，并且保持一定时间
                before_cross = all(series1.iloc[i-window:i] >= series2.iloc[i-window:i])
                after_cross = all(series1.iloc[i:i+1] < series2.iloc[i:i+1])

                if before_cross and after_cross:
                    result.iloc[i] = True

        return result


# 为了兼容指标注册表，创建别名
KDJ = KdjKdj
