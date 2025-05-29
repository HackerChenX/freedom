"""
移动平均线指标模块

实现各种移动平均线(MA)指标计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import ma, ema, wma, sma
from utils.logger import get_logger

logger = get_logger(__name__)


class MA(BaseIndicator):
    """
    移动平均线指标类
    
    计算各种类型的移动平均线
    """
    
    # MA类型枚举
    MA_TYPE_SIMPLE = 'simple'  # 简单移动平均
    MA_TYPE_EMA = 'ema'        # 指数移动平均
    MA_TYPE_WMA = 'wma'        # 加权移动平均
    MA_TYPE_SMA = 'sma'        # 平滑移动平均
    
    def __init__(self, name: str = "MA", description: str = "移动平均线指标", periods: List[int] = None):
        """
        初始化移动平均线指标
        
        Args:
            name: 指标名称
            description: 指标描述
            periods: 周期列表，默认为[5, 10, 20, 30, 60]
        """
        super().__init__(name, description)
        
        # 设置默认参数
        self._parameters = {
            'periods': periods or [5, 10, 20, 30, 60],  # 周期列表
            'ma_type': self.MA_TYPE_SIMPLE,  # MA类型
            'weight': 1                      # SMA的权重参数
        }
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """获取参数"""
        return self._parameters.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置参数
        
        Args:
            params: 参数字典
        """
        for key, value in params.items():
            if key in self._parameters:
                self._parameters[key] = value
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算移动平均线指标
        
        Args:
            data: 输入数据，必须包含'close'列
            args: 位置参数
            kwargs: 关键字参数，可包含periods、ma_type和weight
            
        Returns:
            pd.DataFrame: 包含各周期MA的DataFrame
        """
        # 确保数据包含close列
        self.ensure_columns(data, ['close'])
        
        # 获取参数
        periods = kwargs.get('periods', self._parameters['periods'])
        ma_type = kwargs.get('ma_type', self._parameters['ma_type'])
        weight = kwargs.get('weight', self._parameters['weight'])
        
        # 如果periods是单个值，转换为列表
        if not isinstance(periods, list):
            periods = [periods]
        
        # 根据MA类型选择对应的计算函数
        if ma_type == self.MA_TYPE_EMA:
            calc_func = ema
        elif ma_type == self.MA_TYPE_WMA:
            calc_func = wma
        elif ma_type == self.MA_TYPE_SMA:
            def calc_func(series, period):
                return sma(series, period, weight)
        else:  # 默认使用简单移动平均
            calc_func = ma
        
        # 计算各周期的移动平均线
        result = pd.DataFrame(index=data.index)
        for period in periods:
            ma_values = calc_func(data['close'], period)
            result[f'MA{period}'] = ma_values
        
        return result
    
    def is_uptrend(self, period: int) -> pd.Series:
        """
        判断指定周期的均线是否上升趋势
        
        Args:
            period: MA周期
            
        Returns:
            pd.Series: 趋势信号，True表示上升趋势
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        ma_col = f'MA{period}'
        if ma_col not in self._result.columns:
            raise ValueError(f"结果中不存在{ma_col}列")
            
        return self._result[ma_col] > self._result[ma_col].shift(1)
    
    def is_golden_cross(self, short_period: int, long_period: int) -> pd.Series:
        """
        判断是否形成金叉（短期均线上穿长期均线）
        
        Args:
            short_period: 短期均线周期
            long_period: 长期均线周期
            
        Returns:
            pd.Series: 金叉信号，True表示形成金叉
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        short_ma = f'MA{short_period}'
        long_ma = f'MA{long_period}'
        
        if short_ma not in self._result.columns or long_ma not in self._result.columns:
            raise ValueError(f"结果中不存在{short_ma}或{long_ma}列")
            
        return self.crossover(self._result[short_ma], self._result[long_ma])
    
    def is_death_cross(self, short_period: int, long_period: int) -> pd.Series:
        """
        判断是否形成死叉（短期均线下穿长期均线）
        
        Args:
            short_period: 短期均线周期
            long_period: 长期均线周期
            
        Returns:
            pd.Series: 死叉信号，True表示形成死叉
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        short_ma = f'MA{short_period}'
        long_ma = f'MA{long_period}'
        
        if short_ma not in self._result.columns or long_ma not in self._result.columns:
            raise ValueError(f"结果中不存在{short_ma}或{long_ma}列")
            
        return self.crossunder(self._result[short_ma], self._result[long_ma])
    
    def is_multi_uptrend(self, periods: List[int] = None) -> pd.Series:
        """
        判断是否多条均线多头排列（短期均线在上，长期均线在下）
        
        Args:
            periods: 均线周期列表，按从短到长排序
            
        Returns:
            pd.Series: 多头排列信号，True表示形成多头排列
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        if periods is None:
            # 使用默认周期列表，但要确保是有序的
            periods = sorted(self._parameters['periods'])
            
        # 确保所有需要的列都存在
        for period in periods:
            if f'MA{period}' not in self._result.columns:
                raise ValueError(f"结果中不存在MA{period}列")
        
        # 检查是否形成多头排列
        result = pd.Series(True, index=self._result.index)
        for i in range(len(periods) - 1):
            short_ma = f'MA{periods[i]}'
            long_ma = f'MA{periods[i+1]}'
            result &= (self._result[short_ma] > self._result[long_ma])
            
        return result
    
    def price_to_ma_ratio(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        计算价格与均线的比值
        
        Args:
            data: 包含close列的DataFrame
            period: 均线周期
            
        Returns:
            pd.Series: 价格与均线的比值
        """
        if not self.has_result():
            self.compute(data)
            
        ma_col = f'MA{period}'
        if ma_col not in self._result.columns:
            raise ValueError(f"结果中不存在{ma_col}列")
            
        return data['close'] / self._result[ma_col]
    
    def is_touching_ma(self, data: pd.DataFrame, period: int, threshold: float = 0.02) -> pd.Series:
        """
        判断价格是否接触移动平均线
        
        Args:
            data: 包含价格数据的DataFrame
            period: MA周期
            threshold: 接触阈值，价格与MA的相对差异小于该阈值即认为接触
            
        Returns:
            pd.Series: 接触信号，True表示接触
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        ma_col = f'MA{period}'
        if ma_col not in self._result.columns:
            raise ValueError(f"结果中不存在{ma_col}列")
            
        # 计算价格与MA的相对差异
        rel_diff = (data['close'] - self._result[ma_col]) / self._result[ma_col]
        
        # 判断是否接触
        return rel_diff.abs() <= threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成MA信号
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含信号的DataFrame
        """
        # 计算MA指标
        if not self.has_result():
            self.compute(data)
            
        if not self.has_result():
            return pd.DataFrame()
        
        # 获取MA参数
        periods = self._parameters['periods']
        if not isinstance(periods, list):
            periods = [periods]
            
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 按从短到长排序周期
        sorted_periods = sorted(periods)
        
        # 添加价格突破信号
        price_col = 'close'
        price = data[price_col]
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                ma_values = self._result[ma_col]
                # 价格上穿MA
                signals[f'price_cross_above_ma{period}'] = self.crossover(price, ma_values)
                # 价格下穿MA
                signals[f'price_cross_below_ma{period}'] = self.crossunder(price, ma_values)
                # 价格在MA上方
                signals[f'price_above_ma{period}'] = price > ma_values
                # 价格在MA下方
                signals[f'price_below_ma{period}'] = price < ma_values
        
        # 添加买入信号
        signals['buy_signal'] = False
        
        # 价格突破均线作为买入信号
        for period in periods:
            price_cross_col = f'price_cross_above_ma{period}'
            if price_cross_col in signals.columns:
                signals['buy_signal'] |= signals[price_cross_col]
        
        # 添加卖出信号
        signals['sell_signal'] = False
        
        # 价格跌破均线作为卖出信号
        for period in periods:
            price_cross_col = f'price_cross_below_ma{period}'
            if price_cross_col in signals.columns:
                signals['sell_signal'] |= signals[price_cross_col]
        
        # 计算信号强度
        strength = 50.0  # 默认中性
        
        # 价格在均线上方，信号强度增加
        price_above_count = 0
        for period in periods:
            price_above_col = f'price_above_ma{period}'
            if price_above_col in signals.columns and signals[price_above_col].iloc[-1]:
                price_above_count += 1
        
        strength += price_above_count * (20.0 / len(periods))
        
        # 价格在均线下方，信号强度减少
        price_below_count = 0
        for period in periods:
            price_below_col = f'price_below_ma{period}'
            if price_below_col in signals.columns and signals[price_below_col].iloc[-1]:
                price_below_count += 1
        
        strength -= price_below_count * (20.0 / len(periods))
        
        # 确保强度在0-100范围内
        strength = max(0.0, min(100.0, strength))
        
        # 添加信号强度
        signals['signal_strength'] = 0.0
        signals.loc[signals.index[-1], 'signal_strength'] = strength
        
        return signals
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MA原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算MA
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 获取价格数据
        close_price = data['close']
        
        # 获取参数
        periods = kwargs.get('periods', self._parameters['periods'])
        if not isinstance(periods, list):
            periods = [periods]
        
        # 1. 价格与均线关系评分
        price_ma_score = self._calculate_price_ma_score(close_price, periods)
        score += price_ma_score
        
        # 2. 均线交叉评分
        cross_score = self._calculate_ma_cross_score(periods)
        score += cross_score
        
        # 3. 均线趋势评分
        trend_score = self._calculate_ma_trend_score(periods)
        score += trend_score
        
        # 4. 均线排列评分
        arrangement_score = self._calculate_ma_arrangement_score(periods)
        score += arrangement_score
        
        # 5. 价格穿越评分
        penetration_score = self._calculate_price_penetration_score(close_price, periods)
        score += penetration_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别MA技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算MA
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 获取参数
        periods = kwargs.get('periods', self._parameters['periods'])
        if not isinstance(periods, list):
            periods = [periods]
        
        close_price = data['close']
        
        # 1. 检测均线交叉形态
        cross_patterns = self._detect_ma_cross_patterns(periods)
        patterns.extend(cross_patterns)
        
        # 2. 检测均线排列形态
        arrangement_patterns = self._detect_ma_arrangement_patterns(periods)
        patterns.extend(arrangement_patterns)
        
        # 3. 检测价格与均线关系形态
        price_patterns = self._detect_price_ma_patterns(close_price, periods)
        patterns.extend(price_patterns)
        
        # 4. 检测均线趋势形态
        trend_patterns = self._detect_ma_trend_patterns(periods)
        patterns.extend(trend_patterns)
        
        # 5. 检测均线支撑阻力形态
        support_resistance_patterns = self._detect_support_resistance_patterns(close_price, periods)
        patterns.extend(support_resistance_patterns)
        
        return patterns
    
    def _calculate_price_ma_score(self, close_price: pd.Series, periods: List[int]) -> pd.Series:
        """
        计算价格与均线关系评分
        
        Args:
            close_price: 收盘价序列
            periods: 周期列表
            
        Returns:
            pd.Series: 价格关系评分
        """
        price_score = pd.Series(0.0, index=close_price.index)
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                ma_values = self._result[ma_col]
                
                # 价格在均线上方+5分
                above_ma = close_price > ma_values
                price_score += above_ma * 5
                
                # 价格在均线下方-5分
                below_ma = close_price < ma_values
                price_score -= below_ma * 5
                
                # 价格距离均线的相对位置评分
                price_distance = (close_price - ma_values) / ma_values * 100
                
                # 距离适中（1-3%）额外加分
                moderate_distance = (abs(price_distance) >= 1) & (abs(price_distance) <= 3)
                price_score += moderate_distance * 3
        
        return price_score / len(periods)  # 平均化
    
    def _calculate_ma_cross_score(self, periods: List[int]) -> pd.Series:
        """
        计算均线交叉评分
        
        Args:
            periods: 周期列表
            
        Returns:
            pd.Series: 交叉评分
        """
        cross_score = pd.Series(0.0, index=self._result.index)
        
        # 需要至少两个周期才能计算交叉
        if len(periods) < 2:
            return cross_score
        
        sorted_periods = sorted(periods)
        
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            
            short_ma = f'MA{short_period}'
            long_ma = f'MA{long_period}'
            
            if short_ma in self._result.columns and long_ma in self._result.columns:
                # 金叉（短期均线上穿长期均线）+20分
                golden_cross = self.crossover(self._result[short_ma], self._result[long_ma])
                cross_score += golden_cross * 20
                
                # 死叉（短期均线下穿长期均线）-20分
                death_cross = self.crossunder(self._result[short_ma], self._result[long_ma])
                cross_score -= death_cross * 20
        
        return cross_score
    
    def _calculate_ma_trend_score(self, periods: List[int]) -> pd.Series:
        """
        计算均线趋势评分
        
        Args:
            periods: 周期列表
            
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                ma_values = self._result[ma_col]
                
                # 均线上升趋势+8分
                ma_rising = ma_values > ma_values.shift(1)
                trend_score += ma_rising * 8
                
                # 均线下降趋势-8分
                ma_falling = ma_values < ma_values.shift(1)
                trend_score -= ma_falling * 8
                
                # 均线加速上升+12分
                if len(ma_values) >= 3:
                    ma_accelerating = (ma_values.diff() > ma_values.shift(1).diff())
                    trend_score += ma_accelerating * 12
                
                # 均线加速下降-12分
                if len(ma_values) >= 3:
                    ma_decelerating = (ma_values.diff() < ma_values.shift(1).diff())
                    trend_score -= ma_decelerating * 12
        
        return trend_score / len(periods)  # 平均化
    
    def _calculate_ma_arrangement_score(self, periods: List[int]) -> pd.Series:
        """
        计算均线排列评分
        
        Args:
            periods: 周期列表
            
        Returns:
            pd.Series: 排列评分
        """
        arrangement_score = pd.Series(0.0, index=self._result.index)
        
        if len(periods) < 3:
            return arrangement_score
        
        sorted_periods = sorted(periods)
        
        # 检查多头排列（短期均线在上，长期均线在下）
        bullish_arrangement = pd.Series(True, index=self._result.index)
        bearish_arrangement = pd.Series(True, index=self._result.index)
        
        for i in range(len(sorted_periods) - 1):
            short_ma = f'MA{sorted_periods[i]}'
            long_ma = f'MA{sorted_periods[i + 1]}'
            
            if short_ma in self._result.columns and long_ma in self._result.columns:
                # 多头排列：短期均线 > 长期均线
                bullish_arrangement &= (self._result[short_ma] > self._result[long_ma])
                
                # 空头排列：短期均线 < 长期均线
                bearish_arrangement &= (self._result[short_ma] < self._result[long_ma])
        
        # 多头排列+25分
        arrangement_score += bullish_arrangement * 25
        
        # 空头排列-25分
        arrangement_score -= bearish_arrangement * 25
        
        return arrangement_score
    
    def _calculate_price_penetration_score(self, close_price: pd.Series, periods: List[int]) -> pd.Series:
        """
        计算价格穿越评分
        
        Args:
            close_price: 收盘价序列
            periods: 周期列表
            
        Returns:
            pd.Series: 穿越评分
        """
        penetration_score = pd.Series(0.0, index=close_price.index)
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                ma_values = self._result[ma_col]
                
                # 价格上穿均线+15分
                price_cross_up = self.crossover(close_price, ma_values)
                penetration_score += price_cross_up * 15
                
                # 价格下穿均线-15分
                price_cross_down = self.crossunder(close_price, ma_values)
                penetration_score -= price_cross_down * 15
        
        return penetration_score / len(periods)  # 平均化
    
    def _detect_ma_cross_patterns(self, periods: List[int]) -> List[str]:
        """
        检测均线交叉形态
        
        Args:
            periods: 周期列表
            
        Returns:
            List[str]: 交叉形态列表
        """
        patterns = []
        
        if len(periods) < 2:
            return patterns
        
        sorted_periods = sorted(periods)
        
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            
            short_ma = f'MA{short_period}'
            long_ma = f'MA{long_period}'
            
            if short_ma in self._result.columns and long_ma in self._result.columns:
                # 检查最近的交叉
                recent_periods = min(5, len(self._result))
                recent_short = self._result[short_ma].tail(recent_periods)
                recent_long = self._result[long_ma].tail(recent_periods)
                
                if self.crossover(recent_short, recent_long).any():
                    patterns.append(f"MA{short_period}上穿MA{long_period}")
                
                if self.crossunder(recent_short, recent_long).any():
                    patterns.append(f"MA{short_period}下穿MA{long_period}")
        
        return patterns
    
    def _detect_ma_arrangement_patterns(self, periods: List[int]) -> List[str]:
        """
        检测均线排列形态
        
        Args:
            periods: 周期列表
            
        Returns:
            List[str]: 排列形态列表
        """
        patterns = []
        
        if len(periods) < 3:
            return patterns
        
        sorted_periods = sorted(periods)
        
        # 检查当前排列状态
        if len(self._result) > 0:
            current_bullish = True
            current_bearish = True
            
            for i in range(len(sorted_periods) - 1):
                short_ma = f'MA{sorted_periods[i]}'
                long_ma = f'MA{sorted_periods[i + 1]}'
                
                if short_ma in self._result.columns and long_ma in self._result.columns:
                    current_short = self._result[short_ma].iloc[-1]
                    current_long = self._result[long_ma].iloc[-1]
                    
                    if pd.isna(current_short) or pd.isna(current_long):
                        continue
                    
                    if current_short <= current_long:
                        current_bullish = False
                    if current_short >= current_long:
                        current_bearish = False
            
            if current_bullish:
                patterns.append("均线多头排列")
            elif current_bearish:
                patterns.append("均线空头排列")
            else:
                patterns.append("均线交织状态")
        
        return patterns
    
    def _detect_price_ma_patterns(self, close_price: pd.Series, periods: List[int]) -> List[str]:
        """
        检测价格与均线关系形态
        
        Args:
            close_price: 收盘价序列
            periods: 周期列表
            
        Returns:
            List[str]: 价格关系形态列表
        """
        patterns = []
        
        if len(close_price) == 0:
            return patterns
        
        current_price = close_price.iloc[-1]
        above_count = 0
        below_count = 0
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                current_ma = self._result[ma_col].iloc[-1]
                
                if pd.isna(current_ma):
                    continue
                
                if current_price > current_ma:
                    above_count += 1
                elif current_price < current_ma:
                    below_count += 1
        
        total_ma = above_count + below_count
        if total_ma > 0:
            above_ratio = above_count / total_ma
            
            if above_ratio >= 0.8:
                patterns.append("价格强势上行")
            elif above_ratio >= 0.6:
                patterns.append("价格温和上行")
            elif above_ratio <= 0.2:
                patterns.append("价格强势下行")
            elif above_ratio <= 0.4:
                patterns.append("价格温和下行")
            else:
                patterns.append("价格均线附近震荡")
        
        # 检查价格穿越
        recent_periods = min(5, len(close_price))
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                recent_price = close_price.tail(recent_periods)
                recent_ma = self._result[ma_col].tail(recent_periods)
                
                if self.crossover(recent_price, recent_ma).any():
                    patterns.append(f"价格上穿MA{period}")
                
                if self.crossunder(recent_price, recent_ma).any():
                    patterns.append(f"价格下穿MA{period}")
        
        return patterns
    
    def _detect_ma_trend_patterns(self, periods: List[int]) -> List[str]:
        """
        检测均线趋势形态
        
        Args:
            periods: 周期列表
            
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        rising_count = 0
        falling_count = 0
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns and len(self._result) >= 2:
                ma_values = self._result[ma_col]
                current_ma = ma_values.iloc[-1]
                prev_ma = ma_values.iloc[-2]
                
                if pd.isna(current_ma) or pd.isna(prev_ma):
                    continue
                
                if current_ma > prev_ma:
                    rising_count += 1
                elif current_ma < prev_ma:
                    falling_count += 1
        
        total_ma = rising_count + falling_count
        if total_ma > 0:
            rising_ratio = rising_count / total_ma
            
            if rising_ratio >= 0.8:
                patterns.append("均线全面上升")
            elif rising_ratio >= 0.6:
                patterns.append("均线多数上升")
            elif rising_ratio <= 0.2:
                patterns.append("均线全面下降")
            elif rising_ratio <= 0.4:
                patterns.append("均线多数下降")
            else:
                patterns.append("均线方向分化")
        
        return patterns
    
    def _detect_support_resistance_patterns(self, close_price: pd.Series, periods: List[int]) -> List[str]:
        """
        检测均线支撑阻力形态
        
        Args:
            close_price: 收盘价序列
            periods: 周期列表
            
        Returns:
            List[str]: 支撑阻力形态列表
        """
        patterns = []
        
        if len(close_price) < 5:
            return patterns
        
        recent_periods = min(10, len(close_price))
        recent_price = close_price.tail(recent_periods)
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                recent_ma = self._result[ma_col].tail(recent_periods)
                
                # 检查支撑：价格多次接近均线但未跌破
                support_touches = 0
                resistance_touches = 0
                
                for i in range(1, len(recent_price)):
                    price_diff = abs(recent_price.iloc[i] - recent_ma.iloc[i]) / recent_ma.iloc[i]
                    
                    if price_diff < 0.02:  # 2%以内认为是接触
                        if recent_price.iloc[i] >= recent_ma.iloc[i]:
                            if recent_price.iloc[i-1] < recent_ma.iloc[i-1]:
                                support_touches += 1
                        else:
                            if recent_price.iloc[i-1] > recent_ma.iloc[i-1]:
                                resistance_touches += 1
                
                if support_touches >= 2:
                    patterns.append(f"MA{period}形成支撑")
                
                if resistance_touches >= 2:
                    patterns.append(f"MA{period}形成阻力")
        
        return patterns 