"""
布林带指标模块

实现布林带(BOLL)指标计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import boll as calc_boll
from utils.logger import get_logger

logger = get_logger(__name__)


class BOLL(BaseIndicator):
    """
    布林带指标类
    
    计算布林带上轨、中轨和下轨
    """
    
    def __init__(self, periods: int = 20, std_dev: float = 2.0, name: str = "BOLL", description: str = "布林带指标"):
        """
        初始化布林带指标
        
        Args:
            periods: 周期，默认20
            std_dev: 标准差倍数，默认2.0
            name: 指标名称
            description: 指标描述
        """
        super().__init__(name, description)
        
        # 设置默认参数
        self.periods = periods
        self.std_dev = std_dev
        self._parameters = {
            'periods': periods,      # 周期
            'std_dev': std_dev,     # 标准差倍数
            'bw_periods': 20    # 带宽变化率计算周期
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
        计算布林带指标
        
        Args:
            data: 输入数据，必须包含'close'列
            args: 位置参数
            kwargs: 关键字参数，可包含periods和std_dev
            
        Returns:
            pd.DataFrame: 包含upper、middle、lower列的DataFrame
        """
        # 确保数据包含close列
        self.ensure_columns(data, ['close'])
        
        # 获取参数
        periods = kwargs.get('periods', self._parameters['periods'])
        std_dev = kwargs.get('std_dev', self._parameters['std_dev'])
        
        # 计算布林带
        upper, middle, lower = calc_boll(data['close'], periods, std_dev)
        
        # 构建结果DataFrame
        result = data.copy()
        result['upper'] = upper
        result['middle'] = middle
        result['lower'] = lower
        
        # 保存结果
        self._result = result
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算BOLL原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算BOLL
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        close = self._result['close']
        upper = self._result['upper']
        middle = self._result['middle']
        lower = self._result['lower']
        
        # 1. 价格位置评分
        # 价格触及下轨（超卖）+20分
        price_at_lower = close <= lower
        score += price_at_lower * 20
        
        # 价格触及上轨（超买）-20分
        price_at_upper = close >= upper
        score -= price_at_upper * 20
        
        # 2. 价格突破评分
        # 价格突破下轨（强烈超卖）+25分
        price_break_lower = close < lower
        score += price_break_lower * 25
        
        # 价格突破上轨（强烈超买）-25分
        price_break_upper = close > upper
        score -= price_break_upper * 25
        
        # 3. 价格运行方向评分
        # 价格由下轨向中轨运行+15分
        price_from_lower_to_middle = self._detect_price_movement(close, lower, middle, direction='up')
        score += price_from_lower_to_middle * 15
        
        # 价格由上轨向中轨运行-15分
        price_from_upper_to_middle = self._detect_price_movement(close, upper, middle, direction='down')
        score -= price_from_upper_to_middle * 15
        
        # 4. 带宽变化评分 - 优化点：带宽变化率评估
        bandwidth = (upper - lower) / middle
        
        # 计算带宽变化率
        if len(bandwidth) >= 20:
            # 计算带宽20日变化率
            bw_periods = self._parameters['bw_periods']
            bandwidth_change_rate = (bandwidth - bandwidth.shift(bw_periods)) / bandwidth.shift(bw_periods)
            
            # 带宽快速收缩（可能孕育爆发行情）
            bandwidth_fast_contracting = bandwidth_change_rate < -0.2
            score += bandwidth_fast_contracting * 20
            
            # 带宽快速扩张（趋势确认）
            bandwidth_fast_expanding = bandwidth_change_rate > 0.2
            # 根据价格位置决定加分还是减分
            expanding_score = np.where(close > middle, 15, -15)  # 价格在中轨上方加分，下方减分
            score += bandwidth_fast_expanding * expanding_score
        else:
            # 如果历史数据不足，使用原有的简单判断
            bandwidth_expanding = bandwidth > bandwidth.shift(1)
            bandwidth_contracting = bandwidth < bandwidth.shift(1)
            score += bandwidth_contracting * 15
        
        # 带宽极低（即将突破）+20分
        bandwidth_percentile = bandwidth.rolling(window=60).rank(pct=True)
        extremely_low_bandwidth = bandwidth_percentile < 0.1
        score += extremely_low_bandwidth * 20
        
        # 5. 价格弹性系数 - 优化点：价格与轨道的弹性关系
        if len(close) >= 5:
            # 计算价格与最近触及的边界距离变化率
            price_position = (close - lower) / (upper - lower)  # 价格在带宽中的相对位置(0-1)
            
            # 识别从边界回归中轨的情况
            from_edge_to_middle = (
                # 从下轨回归
                ((price_position.shift(5) < 0.2) & (price_position > 0.2) & (price_position < 0.5)) |
                # 从上轨回归
                ((price_position.shift(5) > 0.8) & (price_position < 0.8) & (price_position > 0.5))
            )
            
            # 计算弹性系数：移动更快得分更高
            elasticity = abs(price_position - price_position.shift(5)) * 20
            elasticity_score = np.clip(elasticity, 0, 15)
            
            # 从下轨弹回加分，从上轨回落减分
            elasticity_direction = np.where(price_position.shift(5) < 0.5, 1, -1)
            score += from_edge_to_middle * elasticity_score * elasticity_direction
        
        # 6. 布林带斜率配合 - 优化点：价格与布林带方向一致性
        if len(middle) >= 10:
            # 计算中轨斜率
            middle_slope = (middle - middle.shift(5)) / middle.shift(5) * 100
            
            # 计算价格短期斜率
            price_slope = (close - close.shift(5)) / close.shift(5) * 100
            
            # 判断方向一致性
            direction_match = np.sign(middle_slope) == np.sign(price_slope)
            
            # 方向一致系数：方向一致时信号更强
            direction_coef = np.where(direction_match, 1.2, 0.8)
            
            # 应用方向一致系数调整评分
            # 注意：这里仅调整已有的评分，而不是添加新分数
            score_adjustment = (score - 50) * (direction_coef - 1.0)
            score += score_adjustment
        
        # 7. 中轨穿越评分
        # 价格上穿中轨+10分
        price_cross_up_middle = self.crossover(close, middle)
        score += price_cross_up_middle * 10
        
        # 价格下穿中轨-10分
        price_cross_down_middle = self.crossunder(close, middle)
        score -= price_cross_down_middle * 10
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别BOLL技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算BOLL
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        close = self._result['close']
        upper = self._result['upper']
        middle = self._result['middle']
        lower = self._result['lower']
        
        # 检查最近的信号
        recent_periods = min(10, len(close))  # 增加至10周期以识别更多形态
        if recent_periods == 0:
            return patterns
        
        recent_close = close.tail(recent_periods)
        recent_upper = upper.tail(recent_periods)
        recent_middle = middle.tail(recent_periods)
        recent_lower = lower.tail(recent_periods)
        
        current_close = recent_close.iloc[-1]
        current_upper = recent_upper.iloc[-1]
        current_middle = recent_middle.iloc[-1]
        current_lower = recent_lower.iloc[-1]
        
        # 1. 基本位置形态
        if current_close >= current_upper:
            patterns.append("布林带超买区")
        elif current_close <= current_lower:
            patterns.append("布林带超卖区")
        
        # 2. 突破形态
        if self.crossover(recent_close, recent_upper).any():
            patterns.append("突破布林带上轨")
        if self.crossunder(recent_close, recent_lower).any():
            patterns.append("突破布林带下轨")
        
        # 3. 带宽形态 - 优化：更精细的带宽形态识别
        bandwidth = (upper - lower) / middle
        recent_bandwidth = bandwidth.tail(recent_periods)
        
        # 计算带宽历史百分位
        if len(bandwidth) >= 60:
            bandwidth_percentile = bandwidth.rolling(window=60).rank(pct=True)
            recent_bw_percentile = bandwidth_percentile.tail(recent_periods)
            
            if recent_bw_percentile.iloc[-1] < 0.1:
                patterns.append("布林带极度收窄")
            elif recent_bw_percentile.iloc[-1] < 0.2:
                patterns.append("布林带收窄")
            elif recent_bw_percentile.iloc[-1] > 0.9:
                patterns.append("布林带极度扩张")
            elif recent_bw_percentile.iloc[-1] > 0.8:
                patterns.append("布林带扩张")
            
            # 带宽变化趋势
            if recent_bandwidth.iloc[-1] < recent_bandwidth.iloc[-3] * 0.9:
                patterns.append("带宽快速收窄")
            elif recent_bandwidth.iloc[-1] > recent_bandwidth.iloc[-3] * 1.1:
                patterns.append("带宽快速扩张")
        
        # 4. 弹性反转形态 - 优化：识别从边界弹回的形态
        price_position = (close - lower) / (upper - lower)  # 位置百分比(0-1)
        recent_position = price_position.tail(recent_periods)
        
        # 从下轨弹回中轨
        if (recent_position.iloc[-5] < 0.2 and recent_position.iloc[-1] > 0.4 and recent_position.iloc[-1] < 0.6):
            patterns.append("下轨弹回中轨")
        
        # 从上轨回落中轨
        if (recent_position.iloc[-5] > 0.8 and recent_position.iloc[-1] < 0.6 and recent_position.iloc[-1] > 0.4):
            patterns.append("上轨回落中轨")
        
        # 5. 方向一致性形态 - 优化：识别价格与布林带方向一致的形态
        if len(middle) >= 10:
            # 计算中轨斜率和价格斜率
            middle_slope = (middle.iloc[-1] - middle.iloc[-10]) / middle.iloc[-10]
            price_slope = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            
            if middle_slope > 0.01 and price_slope > 0.01:
                patterns.append("价格与布林带同步上涨")
            elif middle_slope < -0.01 and price_slope < -0.01:
                patterns.append("价格与布林带同步下跌")
            elif middle_slope > 0.01 and price_slope < 0:
                patterns.append("价格背离布林带上涨")
            elif middle_slope < -0.01 and price_slope > 0:
                patterns.append("价格背离布林带下跌")
        
        # 6. 走势形态
        if self._detect_w_bottom_pattern(recent_close, recent_lower):
            patterns.append("布林带W底形态")
        if self._detect_m_top_pattern(recent_close, recent_upper):
            patterns.append("布林带M顶形态")
        
        return patterns
    
    def _detect_price_movement(self, close: pd.Series, from_line: pd.Series, 
                              to_line: pd.Series, direction: str) -> pd.Series:
        """
        检测价格运行方向
        
        Args:
            close: 收盘价
            from_line: 起始线
            to_line: 目标线
            direction: 方向 ('up' 或 'down')
            
        Returns:
            pd.Series: 运行信号
        """
        movement = pd.Series(False, index=close.index)
        
        if direction == 'up':
            # 价格从下方向上方运行
            near_from = abs(close - from_line) / from_line < 0.02  # 接近起始线
            moving_to = close > close.shift(1)  # 价格上升
            approaching_to = (close - to_line).abs() < (close.shift(1) - to_line).abs()  # 接近目标线
            movement = near_from & moving_to & approaching_to
        elif direction == 'down':
            # 价格从上方向下方运行
            near_from = abs(close - from_line) / from_line < 0.02  # 接近起始线
            moving_to = close < close.shift(1)  # 价格下降
            approaching_to = (close - to_line).abs() < (close.shift(1) - to_line).abs()  # 接近目标线
            movement = near_from & moving_to & approaching_to
        
        return movement
    
    def _detect_bandwidth_squeeze(self, bandwidth: pd.Series) -> bool:
        """
        检测带宽收缩
        
        Args:
            bandwidth: 带宽序列
            
        Returns:
            bool: 是否收缩
        """
        if len(bandwidth) < 3:
            return False
        
        # 连续收缩
        return (bandwidth.iloc[-1] < bandwidth.iloc[-2] < bandwidth.iloc[-3])
    
    def _detect_bandwidth_expansion(self, bandwidth: pd.Series) -> bool:
        """
        检测带宽扩张
        
        Args:
            bandwidth: 带宽序列
            
        Returns:
            bool: 是否扩张
        """
        if len(bandwidth) < 3:
            return False
        
        # 连续扩张
        return (bandwidth.iloc[-1] > bandwidth.iloc[-2] > bandwidth.iloc[-3])
    
    def _detect_bollinger_squeeze(self, close: pd.Series, upper: pd.Series, lower: pd.Series) -> bool:
        """
        检测布林带收口形态
        
        Args:
            close, upper, lower: 价格和布林带序列
            
        Returns:
            bool: 是否为收口形态
        """
        if len(close) < 10:
            return False
        
        # 带宽持续收缩且价格在中轨附近震荡
        bandwidth = (upper - lower) / close
        bandwidth_trend = bandwidth.tail(5)
        
        # 带宽连续收缩
        bandwidth_contracting = (bandwidth_trend.diff() < 0).sum() >= 3
        
        # 价格在中轨附近
        middle = (upper + lower) / 2
        price_near_middle = abs(close.iloc[-1] - middle.iloc[-1]) / middle.iloc[-1] < 0.02
        
        return bandwidth_contracting and price_near_middle
    
    def _detect_bollinger_breakout(self, close: pd.Series, upper: pd.Series, lower: pd.Series) -> bool:
        """
        检测布林带突破形态
        
        Args:
            close, upper, lower: 价格和布林带序列
            
        Returns:
            bool: 是否为突破形态
        """
        if len(close) < 5:
            return False
        
        # 价格突破上轨或下轨，且伴随成交量放大
        recent_close = close.tail(3)
        recent_upper = upper.tail(3)
        recent_lower = lower.tail(3)
        
        # 突破上轨
        upper_breakout = (recent_close.iloc[-1] > recent_upper.iloc[-1] and 
                         recent_close.iloc[-2] <= recent_upper.iloc[-2])
        
        # 突破下轨
        lower_breakout = (recent_close.iloc[-1] < recent_lower.iloc[-1] and 
                         recent_close.iloc[-2] >= recent_lower.iloc[-2])
        
        return upper_breakout or lower_breakout

    def is_upper_breakout(self, data: pd.DataFrame) -> pd.Series:
        """
        判断是否突破上轨
        
        Args:
            data: 包含close列和计算结果的DataFrame
            
        Returns:
            pd.Series: 突破信号，True表示突破上轨
        """
        if not self.has_result():
            self.compute(data)
            
        return data['close'] > self._result['upper']
    
    def is_lower_breakout(self, data: pd.DataFrame) -> pd.Series:
        """
        判断是否突破下轨
        
        Args:
            data: 包含close列和计算结果的DataFrame
            
        Returns:
            pd.Series: 突破信号，True表示突破下轨
        """
        if not self.has_result():
            self.compute(data)
            
        return data['close'] < self._result['lower']
    
    def is_middle_crossover(self, data: pd.DataFrame) -> pd.Series:
        """
        判断是否上穿中轨
        
        Args:
            data: 包含close列和计算结果的DataFrame
            
        Returns:
            pd.Series: 上穿信号，True表示上穿中轨
        """
        if not self.has_result():
            self.compute(data)
            
        return self.crossover(data['close'], self._result['middle'])
    
    def is_middle_crossunder(self, data: pd.DataFrame) -> pd.Series:
        """
        判断是否下穿中轨
        
        Args:
            data: 包含close列和计算结果的DataFrame
            
        Returns:
            pd.Series: 下穿信号，True表示下穿中轨
        """
        if not self.has_result():
            self.compute(data)
            
        return self.crossunder(data['close'], self._result['middle'])
    
    def get_bandwidth(self) -> pd.Series:
        """
        获取带宽
        
        Returns:
            pd.Series: 带宽 (upper - lower) / middle
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        return (self._result['upper'] - self._result['lower']) / self._result['middle']
    
    def get_bandwidth_rate(self, periods: Optional[int] = None) -> pd.Series:
        """
        获取带宽变化率，用于判断市场即将突破的时机
        
        Args:
            periods: 变化率计算周期，默认使用参数中的bw_periods
            
        Returns:
            pd.Series: 带宽变化率，(当前带宽 - N周期前带宽) / N周期前带宽 * 100
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
        
        periods = periods or self._parameters['bw_periods']
        bandwidth = self.get_bandwidth()
        prev_bandwidth = bandwidth.shift(periods)
        
        return (bandwidth - prev_bandwidth) / prev_bandwidth * 100
    
    def is_bandwidth_squeeze(self, threshold: float = 20.0) -> pd.Series:
        """
        判断是否处于带宽收缩状态，带宽持续减小可能预示着行情即将爆发
        
        Args:
            threshold: 带宽收缩阈值，负值表示收缩幅度
            
        Returns:
            pd.Series: 收缩信号，True表示带宽收缩超过阈值
        """
        bw_rate = self.get_bandwidth_rate()
        return bw_rate < -threshold
    
    def is_bandwidth_expansion(self, threshold: float = 20.0) -> pd.Series:
        """
        判断是否处于带宽扩张状态，带宽快速增加通常意味着突破已经开始
        
        Args:
            threshold: 带宽扩张阈值，正值表示扩张幅度
            
        Returns:
            pd.Series: 扩张信号，True表示带宽扩张超过阈值
        """
        bw_rate = self.get_bandwidth_rate()
        return bw_rate > threshold
    
    def get_position(self, data: pd.DataFrame) -> pd.Series:
        """
        获取价格在布林带中的相对位置
        
        Args:
            data: 包含close列的DataFrame
            
        Returns:
            pd.Series: 相对位置，0表示在下轨，1表示在上轨，0.5表示在中轨
        """
        if not self.has_result():
            self.compute(data)
            
        close = data['close']
        upper = self._result['upper']
        lower = self._result['lower']
        
        # 计算相对位置 %B = (close - lower) / (upper - lower)
        position = (close - lower) / (upper - lower)
        
        return position.fillna(0.5)  # 如果计算失败，默认为中位

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成布林带标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算BOLL指标
        if not self.has_result():
            self.calculate(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 计算评分
        score = self.calculate_raw_score(data, **kwargs)
        signals['score'] = score
        
        # 检测形态
        patterns = self.identify_patterns(data, **kwargs)
        
        # 获取价格和布林带数据
        close_price = data['close']
        upper = self._result['upper']
        middle = self._result['middle']
        lower = self._result['lower']
        
        # 计算带宽和带宽百分比
        bandwidth = (upper - lower) / middle
        bandwidth_percentile = bandwidth.rolling(window=60).rank(pct=True)
        
        # 布林带突破信号
        # 1. 价格上穿上轨 - 卖出信号（价格过热）
        upper_breakout = (close_price > upper) & (close_price.shift(1) <= upper.shift(1))
        signals.loc[upper_breakout, 'sell_signal'] = True
        signals.loc[upper_breakout, 'neutral_signal'] = False
        signals.loc[upper_breakout, 'trend'] = -1
        signals.loc[upper_breakout, 'signal_type'] = '上轨突破'
        signals.loc[upper_breakout, 'signal_desc'] = '价格突破布林带上轨，可能出现短期回调'
        signals.loc[upper_breakout, 'confidence'] = 70.0
        signals.loc[upper_breakout, 'position_size'] = 0.3
        signals.loc[upper_breakout, 'risk_level'] = '高'
        
        # 2. 价格下穿下轨 - 买入信号（价格超跌）
        lower_breakout = (close_price < lower) & (close_price.shift(1) >= lower.shift(1))
        signals.loc[lower_breakout, 'buy_signal'] = True
        signals.loc[lower_breakout, 'neutral_signal'] = False
        signals.loc[lower_breakout, 'trend'] = 1
        signals.loc[lower_breakout, 'signal_type'] = '下轨突破'
        signals.loc[lower_breakout, 'signal_desc'] = '价格突破布林带下轨，可能出现反弹'
        signals.loc[lower_breakout, 'confidence'] = 70.0
        signals.loc[lower_breakout, 'position_size'] = 0.3
        signals.loc[lower_breakout, 'risk_level'] = '高'
        
        # 3. 价格由下向上穿过中轨 - 买入信号（趋势转向上）
        middle_crossover = (close_price > middle) & (close_price.shift(1) <= middle.shift(1))
        signals.loc[middle_crossover, 'buy_signal'] = True
        signals.loc[middle_crossover, 'neutral_signal'] = False
        signals.loc[middle_crossover, 'trend'] = 1
        signals.loc[middle_crossover, 'signal_type'] = '中轨上穿'
        signals.loc[middle_crossover, 'signal_desc'] = '价格上穿布林带中轨，趋势可能转为上升'
        signals.loc[middle_crossover, 'confidence'] = 65.0
        signals.loc[middle_crossover, 'position_size'] = 0.3
        signals.loc[middle_crossover, 'risk_level'] = '中'
        
        # 4. 价格由上向下穿过中轨 - 卖出信号（趋势转向下）
        middle_crossunder = (close_price < middle) & (close_price.shift(1) >= middle.shift(1))
        signals.loc[middle_crossunder, 'sell_signal'] = True
        signals.loc[middle_crossunder, 'neutral_signal'] = False
        signals.loc[middle_crossunder, 'trend'] = -1
        signals.loc[middle_crossunder, 'signal_type'] = '中轨下穿'
        signals.loc[middle_crossunder, 'signal_desc'] = '价格下穿布林带中轨，趋势可能转为下降'
        signals.loc[middle_crossunder, 'confidence'] = 65.0
        signals.loc[middle_crossunder, 'position_size'] = 0.3
        signals.loc[middle_crossunder, 'risk_level'] = '中'
        
        # 5. 带宽收缩 - 潜在的突破信号
        bandwidth_squeeze = bandwidth_percentile < 0.1  # 带宽处于近期10%的低位
        for i in range(len(bandwidth_squeeze)):
            if i > 0 and bandwidth_squeeze.iloc[i] and not bandwidth_squeeze.iloc[i-1]:
                # 带宽收缩开始的位置
                idx = bandwidth_squeeze.index[i]
                signals.loc[idx, 'neutral_signal'] = False
                signals.loc[idx, 'signal_type'] = '带宽收缩'
                signals.loc[idx, 'signal_desc'] = '布林带带宽显著收缩，准备可能的突破行情'
                signals.loc[idx, 'confidence'] = 75.0
                signals.loc[idx, 'risk_level'] = '低'
                # 根据当前趋势判断是买入还是卖出信号
                if close_price.iloc[i] > middle.iloc[i]:
                    signals.loc[idx, 'buy_signal'] = True
                    signals.loc[idx, 'trend'] = 1
                    signals.loc[idx, 'position_size'] = 0.4
                else:
                    signals.loc[idx, 'sell_signal'] = True
                    signals.loc[idx, 'trend'] = -1
                    signals.loc[idx, 'position_size'] = 0.4
        
        # 6. 带宽扩张 - 趋势加速信号
        bandwidth_expansion = bandwidth_percentile > 0.9  # 带宽处于近期90%的高位
        for i in range(len(bandwidth_expansion)):
            if i > 0 and bandwidth_expansion.iloc[i] and not bandwidth_expansion.iloc[i-1]:
                # 带宽扩张开始的位置
                idx = bandwidth_expansion.index[i]
                signals.loc[idx, 'neutral_signal'] = False
                signals.loc[idx, 'signal_type'] = '带宽扩张'
                signals.loc[idx, 'signal_desc'] = '布林带带宽显著扩张，趋势可能加速'
                signals.loc[idx, 'confidence'] = 70.0
                signals.loc[idx, 'risk_level'] = '中'
                # 根据当前趋势判断是买入还是卖出信号
                if close_price.iloc[i] > middle.iloc[i]:
                    signals.loc[idx, 'buy_signal'] = True
                    signals.loc[idx, 'trend'] = 1
                    signals.loc[idx, 'position_size'] = 0.5
                else:
                    signals.loc[idx, 'sell_signal'] = True
                    signals.loc[idx, 'trend'] = -1
                    signals.loc[idx, 'position_size'] = 0.5
        
        # 7. 根据形态设置更多信号
        for pattern in patterns:
            pattern_idx = signals.index[-5:]  # 假设形态影响最近5个周期
            
            if '上涨突破' in pattern or '支撑' in pattern:
                signals.loc[pattern_idx, 'buy_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = 1
                signals.loc[pattern_idx, 'signal_type'] = '布林突破'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 75.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '中'
            
            elif '下跌突破' in pattern or '阻力' in pattern:
                signals.loc[pattern_idx, 'sell_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = -1
                signals.loc[pattern_idx, 'signal_type'] = '布林回落'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 75.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '中'
            
            elif '带宽收缩' in pattern:
                signals.loc[pattern_idx, 'signal_type'] = '布林收缩'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 80.0
                signals.loc[pattern_idx, 'risk_level'] = '低'
        
        # 设置止损价格
        if 'low' in data.columns and 'high' in data.columns:
            # 买入信号的止损设为最近的低点或者下轨线
            buy_indices = signals[signals['buy_signal']].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近低点和下轨值的较小值作为止损
                        recent_low = data.loc[idx-lookback:idx, 'low'].min()
                        boll_stop = lower.loc[idx]
                        signals.loc[idx, 'stop_loss'] = min(recent_low, boll_stop)
            
            # 卖出信号的止损设为最近的高点或者上轨线
            sell_indices = signals[signals['sell_signal']].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近高点和上轨值的较大值作为止损
                        recent_high = data.loc[idx-lookback:idx, 'high'].max()
                        boll_stop = upper.loc[idx]
                        signals.loc[idx, 'stop_loss'] = max(recent_high, boll_stop)
        
        # 根据布林带判断市场环境
        # 带宽大 - 趋势市场，带宽小 - 震荡市场
        bandwidth_mean = bandwidth.rolling(window=20).mean()
        signals['market_env'] = 'sideways_market'  # 默认震荡市场
        
        # 带宽大于均值，且价格位于上方 - 上升趋势市场
        trend_up_idx = (bandwidth > bandwidth_mean) & (close_price > middle)
        signals.loc[trend_up_idx, 'market_env'] = 'uptrend_market'
        
        # 带宽大于均值，且价格位于下方 - 下降趋势市场
        trend_down_idx = (bandwidth > bandwidth_mean) & (close_price < middle)
        signals.loc[trend_down_idx, 'market_env'] = 'downtrend_market'
        
        # 带宽极小 - 强震荡市场
        tight_range_idx = bandwidth_percentile < 0.2
        signals.loc[tight_range_idx, 'market_env'] = 'strong_sideways_market'
        
        return signals 

    def _detect_w_bottom_pattern(self, price: pd.Series, lower: pd.Series) -> bool:
        """
        检测W底形态
        
        Args:
            price: 价格序列
            lower: 下轨序列
            
        Returns:
            bool: 是否为W底形态
        """
        if len(price) < 10:
            return False
        
        # 寻找接近或突破下轨的两个低点
        touch_lower = (price - lower).abs() / lower < 0.02  # 接近下轨的点
        touch_indices = np.where(touch_lower)[0]
        
        if len(touch_indices) >= 2:
            # 查找最近的两个接触点
            last_two = touch_indices[-2:]
            
            # 确保两点之间有反弹（中间点高于两端点）
            if len(last_two) == 2 and last_two[1] - last_two[0] >= 3:  # 至少间隔3个点
                middle_idx = (last_two[0] + last_two[1]) // 2
                if price.iloc[middle_idx] > price.iloc[last_two[0]] * 1.02 and price.iloc[middle_idx] > price.iloc[last_two[1]] * 1.02:
                    # 第二个低点后有向上突破
                    if len(price) > last_two[1] + 2 and price.iloc[-1] > price.iloc[last_two[1]] * 1.03:
                        return True
        
        return False

    def _detect_m_top_pattern(self, price: pd.Series, upper: pd.Series) -> bool:
        """
        检测M顶形态
        
        Args:
            price: 价格序列
            upper: 上轨序列
            
        Returns:
            bool: 是否为M顶形态
        """
        if len(price) < 10:
            return False
        
        # 寻找接近或突破上轨的两个高点
        touch_upper = (price - upper).abs() / upper < 0.02  # 接近上轨的点
        touch_indices = np.where(touch_upper)[0]
        
        if len(touch_indices) >= 2:
            # 查找最近的两个接触点
            last_two = touch_indices[-2:]
            
            # 确保两点之间有回调（中间点低于两端点）
            if len(last_two) == 2 and last_two[1] - last_two[0] >= 3:  # 至少间隔3个点
                middle_idx = (last_two[0] + last_two[1]) // 2
                if price.iloc[middle_idx] < price.iloc[last_two[0]] * 0.98 and price.iloc[middle_idx] < price.iloc[last_two[1]] * 0.98:
                    # 第二个高点后有向下突破
                    if len(price) > last_two[1] + 2 and price.iloc[-1] < price.iloc[last_two[1]] * 0.97:
                        return True
        
        return False 