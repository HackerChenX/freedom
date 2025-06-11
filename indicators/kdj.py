#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KDJ指标

随机指标KDJ，用于分析价格是否处于超买或超卖状态
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from enum import Enum

from indicators.base_indicator import BaseIndicator, MarketEnvironment
from indicators.common import crossover, crossunder
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength, PatternInfo
from utils.logger import get_logger
from utils.decorators import log_calls, error_handling
from utils.technical_utils import calculate_kdj

logger = get_logger(__name__)


class KDJ(BaseIndicator):
    """
    KDJ随机指标
    
    KDJ指标是RSI和随机指标的结合体，是一种超买超卖指标，用于判断股价走势的超买超卖状态。
    """
    
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化KDJ指标
        
        Args:
            n: RSV周期，默认为9
            m1: K值平滑因子，默认为3
            m2: D值平滑因子，默认为3
        """
        super().__init__(name="KDJ", description="随机指标")
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self._market_environment = MarketEnvironment.SIDEWAYS_MARKET  # 默认市场环境
        self._registered_patterns = False
    
    def _register_kdj_patterns(self):
        """将KDJ形态注册到全局形态注册表"""
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册金叉形态
        registry.register(
            pattern_id="KDJ_GOLDEN_CROSS",
            display_name="KDJ金叉",
            description="K线从下方突破D线，买入信号",
            indicator_id="KDJ",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=15.0,
            detection_function=self._detect_golden_cross
        )
        
        # 注册死叉形态
        registry.register(
            pattern_id="KDJ_DEATH_CROSS",
            display_name="KDJ死叉",
            description="K线从上方跌破D线，卖出信号",
            indicator_id="KDJ",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-15.0,
            detection_function=self._detect_death_cross
        )
        
        # 注册超买形态
        registry.register(
            pattern_id="KDJ_OVERBOUGHT",
            display_name="KDJ超买",
            description="K值高于80，超买信号",
            indicator_id="KDJ",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0,
            detection_function=self._detect_overbought
        )
        
        # 注册超卖形态
        registry.register(
            pattern_id="KDJ_OVERSOLD",
            display_name="KDJ超卖",
            description="K值低于20，超卖信号",
            indicator_id="KDJ",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0,
            detection_function=self._detect_oversold
        )
        
        # 注册看涨背离形态
        registry.register(
            pattern_id="KDJ_BULLISH_DIVERGENCE",
            display_name="KDJ看涨背离",
            description="价格创新低而KDJ未创新低，底部反转信号",
            indicator_id="KDJ",
            pattern_type=PatternType.REVERSAL,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0,
            detection_function=self._detect_bullish_divergence
        )
        
        # 注册看跌背离形态
        registry.register(
            pattern_id="KDJ_BEARISH_DIVERGENCE",
            display_name="KDJ看跌背离",
            description="价格创新高而KDJ未创新高，顶部反转信号",
            indicator_id="KDJ",
            pattern_type=PatternType.REVERSAL,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0,
            detection_function=self._detect_bearish_divergence
        )
    
    # 为了向后兼容，保留此方法
    def register_patterns_to_registry(self):
        """将KDJ形态注册到全局形态注册表（已弃用，保留此方法仅用于兼容性）"""
        pass

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取KDJ指标的技术形态
        
        Args:
            data: 输入数据，通常是K线数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含形态信号的DataFrame
        """
        if 'K' not in data.columns or 'D' not in data.columns:
            calculated_data = self._calculate_kdj(data)
        else:
            calculated_data = data

        patterns_df = pd.DataFrame(index=calculated_data.index)
        patterns_df['KDJ_GOLDEN_CROSS'] = self._detect_golden_cross(calculated_data)
        patterns_df['KDJ_DEATH_CROSS'] = self._detect_death_cross(calculated_data)
        
        return patterns_df

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算KDJ指标评分（0-100分制）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            float: 综合评分（0-100）
        """
        raw_score = self.calculate_raw_score(data, **kwargs)
        
        if raw_score.empty:
            return 50.0  # 默认中性评分
        
        last_score = raw_score.iloc[-1]
        
        # 应用市场环境调整
        market_env = kwargs.get('market_env', self._market_environment)
        adjusted_score = self.apply_market_environment_adjustment(market_env, last_score)
        
        # 计算置信度
        confidence = self.calculate_confidence(adjusted_score, self.get_patterns(data), {})
        
        # 返回最终评分
        return float(np.clip(adjusted_score * confidence, 0, 100))
    
    def _detect_overbought(self, data: pd.DataFrame) -> bool:
        """检测KDJ超买形态"""
        if not self.has_result() or 'K' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 2:
            return False
        
        # 获取最新的K值
        k_value = data['K'].iloc[-1]
        
        # 超买条件：K值大于80
        return k_value > 80.0
    
    def _detect_oversold(self, data: pd.DataFrame) -> bool:
        """检测KDJ超卖形态"""
        if not self.has_result() or 'K' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 2:
            return False
        
        # 获取最新的K值
        k_value = data['K'].iloc[-1]
        
        # 超卖条件：K值小于20
        return k_value < 20.0
    
    def _detect_golden_cross(self, data: pd.DataFrame) -> pd.Series:
        """
        检测KDJ金叉形态
        
        Args:
            data: 含有KDJ指标的DataFrame
            
        Returns:
            pd.Series: 标记金叉发生位置的布尔序列
        """
        required_columns = ['K', 'D']
        if not all(col in data.columns for col in required_columns):
            logger.warning(f"检测KDJ金叉形态时缺少必要的列: {required_columns}")
            return pd.Series([False] * len(data), index=data.index)
        
        k = data['K']
        d = data['D']
        return crossover(k, d)
    
    def _detect_death_cross(self, data: pd.DataFrame) -> pd.Series:
        """
        检测KDJ死叉形态
        
        Args:
            data: 含有KDJ指标的DataFrame
            
        Returns:
            pd.Series: 标记死叉发生位置的布尔序列
        """
        required_columns = ['K', 'D']
        if not all(col in data.columns for col in required_columns):
            logger.warning(f"检测KDJ死叉形态时缺少必要的列: {required_columns}")
            return pd.Series([False] * len(data), index=data.index)

        k = data['K']
        d = data['D']
        return crossunder(k, d)
    
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
        """检测J值击穿形态"""
        if not self.has_result() or 'J' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 5:
            return False
        
        # 获取最近5个周期的J值
        j_values = data['J'].iloc[-5:].values
        
        # J值击穿条件：从正值快速下降到低于30
        if j_values[0] > 100 and j_values[-1] < 30:
            # 检查是否是持续下降
            is_falling = True
            for i in range(1, len(j_values)):
                if j_values[i] >= j_values[i-1]:
                    is_falling = False
                    break
            
            return is_falling
        
        return False
    
    def set_market_environment(self, environment: Union[str, MarketEnvironment]) -> None:
        """
        设置市场环境
        
        Args:
            environment: 市场环境，可以是MarketEnvironment枚举或字符串
        """
        if isinstance(environment, MarketEnvironment):
            self._market_environment = environment
            return
            
        # 兼容旧版接口
        env_mapping = {
            "bull_market": MarketEnvironment.BULL_MARKET,
            "bear_market": MarketEnvironment.BEAR_MARKET,
            "sideways_market": MarketEnvironment.SIDEWAYS_MARKET,
            "volatile_market": MarketEnvironment.VOLATILE_MARKET,
            "normal": MarketEnvironment.SIDEWAYS_MARKET,
        }
        
        if environment not in env_mapping:
            valid_values = list(env_mapping.keys())
            raise ValueError(f"无效的市场环境，有效值为: {', '.join(valid_values)}")
            
        self._market_environment = env_mapping[environment]
    
    def get_market_environment(self) -> MarketEnvironment:
        """
        获取当前市场环境
        
        Returns:
            MarketEnvironment: 当前市场环境
        """
        return self._market_environment
    
    def detect_market_environment(self, data: pd.DataFrame) -> MarketEnvironment:
        """
        根据KDJ数值判断当前市场环境
        
        Args:
            data: 包含KDJ指标的DataFrame
            
        Returns:
            MarketEnvironment: 检测到的市场环境
        """
        try:
            if 'K' not in data.columns or 'D' not in data.columns or 'J' not in data.columns:
                return MarketEnvironment.SIDEWAYS_MARKET
            
            # 获取最新的KDJ值
            k_value = data['K'].iloc[-1]
            d_value = data['D'].iloc[-1]
            j_value = data['J'].iloc[-1]
            
            # 根据KDJ值范围判断市场环境
            if k_value > 80 and d_value > 80:
                self._market_environment = MarketEnvironment.UPTREND_MARKET
                return MarketEnvironment.UPTREND_MARKET
            elif k_value < 20 and d_value < 20:
                self._market_environment = MarketEnvironment.DOWNTREND_MARKET
                return MarketEnvironment.DOWNTREND_MARKET
            else:
                self._market_environment = MarketEnvironment.SIDEWAYS_MARKET
                return MarketEnvironment.SIDEWAYS_MARKET
        except Exception as e:
            logger.warning(f"判断市场环境出错: {e}")
            return MarketEnvironment.SIDEWAYS_MARKET
    
    def _calculate(self, *args, **kwargs) -> pd.DataFrame:
        """
        计算KDJ指标

        Args:
            *args: 如果第一个参数是DataFrame，则使用它
            **kwargs: 应该包含一个名为`data`的DataFrame
        
        Returns:
            pd.DataFrame: 包含K, D, J线的DataFrame
        """
        if args and isinstance(args[0], pd.DataFrame):
            data = args[0]
        elif 'data' in kwargs and isinstance(kwargs['data'], pd.DataFrame):
            data = kwargs['data']
        else:
            raise ValueError("calculate()需要一个DataFrame作为输入，通过'data'关键字参数或作为第一个位置参数提供")
        
        return self._calculate_kdj(data, **kwargs)

    def _calculate_kdj(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算KDJ指标
        
        Args:
            data: 包含OHLC数据的DataFrame
            **kwargs: 其他参数，可包含n、m1、m2
            
        Returns:
            pd.DataFrame: 包含K、D、J列的DataFrame
        """
        try:
            # 检查必要的列
            required_cols = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in data.columns]
                logger.error(f"计算KDJ指标需要 {', '.join(required_cols)} 列，缺少: {', '.join(missing)}")
                # 创建空的K, D, J列，避免后续错误
                result = data.copy()
                result['K'] = np.nan
                result['D'] = np.nan
                result['J'] = np.nan
                return result
            
            # 获取参数
            n = kwargs.get('n', self.n)
            m1 = kwargs.get('m1', self.m1)
            m2 = kwargs.get('m2', self.m2)
            
            # 确保足够的数据长度
            if len(data) < n:
                logger.warning(f"数据长度({len(data)})小于所需的回溯周期({n})，返回原始数据")
                result = data.copy()
                result['K'] = np.nan
                result['D'] = np.nan
                result['J'] = np.nan
                return result
            
            # 创建结果DataFrame
            result = data.copy()
            
            # 计算RSV
            high_n = result['high'].rolling(window=n).max()
            low_n = result['low'].rolling(window=n).min()
            close = result['close']
            
            # 防止除以零
            rsv_denominator = high_n - low_n
            rsv = np.where(rsv_denominator != 0, 
                           100 * (close - low_n) / rsv_denominator, 
                           0)
            
            # 计算K值 - 初始值为50
            k_values = np.zeros(len(data))
            k_values[0] = 50
            
            # 计算D值 - 初始值为50
            d_values = np.zeros(len(data))
            d_values[0] = 50
            
            # 计算K和D值
            for i in range(1, len(data)):
                if np.isnan(rsv[i]):
                    k_values[i] = k_values[i-1]
                    d_values[i] = d_values[i-1]
                else:
                    k_values[i] = (m1 * k_values[i-1] + (9 - m1) * rsv[i]) / 9
                    d_values[i] = (m2 * d_values[i-1] + (9 - m2) * k_values[i]) / 9
            
            # 计算J值
            j_values = 3 * k_values - 2 * d_values
            
            # 添加结果
            result['K'] = k_values
            result['D'] = d_values
            result['J'] = j_values
            
            return result
        except Exception as e:
            logger.error(f"计算KDJ指标时出错: {e}")
            # 返回原始数据，但添加空的KDJ列
            result = data.copy()
            result['K'] = np.nan
            result['D'] = np.nan
            result['J'] = np.nan
            return result
    
    def add_signals(self, data: pd.DataFrame, k_col: str = 'K', 
                   d_col: str = 'D', j_col: str = 'J') -> pd.DataFrame:
        """
        添加KDJ交易信号
        
        Args:
            data: 包含KDJ指标的DataFrame
            k_col: K值列名
            d_col: D值列名
            j_col: J值列名
            
        Returns:
            pd.DataFrame: 添加了信号的DataFrame
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
            data: 包含KDJ指标的DataFrame
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
            data: 包含KDJ指标的DataFrame
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
        return "KDJ"
    
    def to_dict(self) -> Dict[str, Any]:
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
            'market_environment': self._market_environment.value,
            'has_result': self.has_result(),
            'has_error': self.has_error()
        }
    
    def get_indicator_type(self) -> str:
        """获取指标类型"""
        return "KDJ"
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据，包含价格和指标数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典
        """
        if not self.has_result():
            self.calculate(data)
            
        result = self._result
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
    
    def identify_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        识别KDJ指标的所有技术形态
        
        Args:
            data: 包含价格和KDJ指标的DataFrame
            
        Returns:
            pd.DataFrame: 包含所有已识别形态的DataFrame，每列代表一种形态
        """
        # 确保KDJ值已计算
        if not all(col in data.columns for col in ['K', 'D', 'J']):
             data = self.calculate(data)

        patterns = pd.DataFrame(index=data.index)
        
        # 使用已注册的检测函数
        registry = PatternRegistry()
        kdj_patterns = registry.get_patterns_by_indicator('KDJ')
        
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
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算KDJ指标的原始评分（0-100分制）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列，取值范围0-100
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 获取KDJ指标值
        if 'K' not in data.columns or 'D' not in data.columns or 'J' not in data.columns:
            raise ValueError("数据中缺少KDJ指标列")
        
        k = data['K']
        d = data['D']
        j = data['J']
        
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
        cross_score = pd.Series(50, index=data.index)
        
        # 金叉加分，最近越近影响越大
        for i in range(5):
            mask = golden_cross.shift(i).fillna(False)
            score_boost = 30 * (0.8 ** i)  # 随距离衰减
            cross_score = np.where(mask, 50 + score_boost, cross_score)
        
        # 死叉减分，最近越近影响越大
        for i in range(5):
            mask = death_cross.shift(i).fillna(False)
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
        patterns = self.get_patterns(data, **kwargs)
        
        # 形态影响分数：最多调整15分
        pattern_adjustment = pd.Series(0, index=data.index)
        for pattern in patterns:
            # 找到对应的注册形态信息
            if pattern.pattern_id in self._registered_patterns:
                score_impact = self._registered_patterns[pattern.pattern_id]['score_impact']
                # 最多调整±15分
                pattern_adjustment += np.clip(score_impact, -15, 15)
        
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

# =====================
# 简化版KDJ形态分析器，供PatternAnalyzer等直接调用
# =====================

class KDJIndicator:
    """KDJ指标分析器（简化版，仅用于形态识别）"""
    def __init__(self, k_period: int = 9, d_period: int = 3, j_period: int = 3):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.k_period = k_period
        self.d_period = d_period
        self.j_period = j_period
        self.patterns = {
            'kdj_golden_cross': {
                'name': 'KDJ金叉',
                'description': 'K线上穿D线',
                'analyzer': self._analyze_golden_cross
            },
            'kdj_death_cross': {
                'name': 'KDJ死叉',
                'description': 'K线下穿D线',
                'analyzer': self._analyze_death_cross
            }
        }

    def _calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        return calculate_kdj(high, low, close, self.k_period, self.d_period, self.j_period)

    def analyze_pattern(self, pattern_id: str, data: pd.DataFrame) -> List[Dict]:
        if pattern_id not in self.patterns:
            raise ValueError(f"不支持的KDJ形态: {pattern_id}")
        return self.patterns[pattern_id]['analyzer'](data)

    def _analyze_golden_cross(self, data: pd.DataFrame) -> List[Dict]:
        k, d, j = self.calculate(data['high'], data['low'], data['close'])
        results = []
        for i in range(1, len(data)):
            if k.iloc[i-1] < d.iloc[i-1] and k.iloc[i] > d.iloc[i]:
                strength = min(1.0, abs(k.iloc[i] - d.iloc[i]) / (abs(k.iloc[i-1] - d.iloc[i-1]) + 1e-6))
                results.append({
                    'date': data.index[i],
                    'pattern': 'kdj_golden_cross',
                    'strength': strength,
                    'price': data['close'].iloc[i],
                    'k': k.iloc[i],
                    'd': d.iloc[i],
                    'j': j.iloc[i]
                })
        return results

    def _analyze_death_cross(self, data: pd.DataFrame) -> List[Dict]:
        k, d, j = self.calculate(data['high'], data['low'], data['close'])
        results = []
        for i in range(1, len(data)):
            if k.iloc[i-1] > d.iloc[i-1] and k.iloc[i] < d.iloc[i]:
                strength = min(1.0, abs(k.iloc[i] - d.iloc[i]) / (abs(k.iloc[i-1] - d.iloc[i-1]) + 1e-6))
                results.append({
                    'date': data.index[i],
                    'pattern': 'kdj_death_cross',
                    'strength': strength,
                    'price': data['close'].iloc[i],
                    'k': k.iloc[i],
                    'd': d.iloc[i],
                    'j': j.iloc[i]
                })
        return results 