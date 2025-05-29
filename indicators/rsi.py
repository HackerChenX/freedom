"""
RSI指标模块

实现RSI指标的计算和相关功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import rsi as calc_rsi
from utils.decorators import exception_handler, validate_dataframe, log_calls
from utils.logger import get_logger

logger = get_logger(__name__)


class RSI(BaseIndicator):
    """
    RSI(Relative Strength Index)指标
    
    RSI是一种动量震荡指标，通过计算价格变动的相对强度来衡量市场超买或超卖情况。
    """
    
    def __init__(self, period: int = 14, periods: Optional[Union[int, List[int]]] = None, 
                overbought: float = 70.0, oversold: float = 30.0):
        """
        初始化RSI指标
        
        Args:
            period: 计算周期，默认为14
            periods: period的别名，为了兼容性；如果是列表，则用于计算多周期RSI
            overbought: 超买阈值，默认为70
            oversold: 超卖阈值，默认为30
        """
        super().__init__(name="RSI", description="相对强弱指标")
        
        # 处理periods参数
        if periods is not None:
            if isinstance(periods, list) and len(periods) > 0:
                # 如果是列表，存储为多周期
                self.periods = periods
                self.period = periods[0]  # 兼容旧代码
            else:
                # 否则直接使用
                self.period = periods
                self.periods = [periods]
        else:
            # 如果periods为None，则使用period参数
            self.period = period
            self.periods = [period]
            
        self.overbought = overbought
        self.oversold = oversold
    
    @validate_dataframe(required_columns=['close'], min_rows=15)
    @log_calls(level='debug')
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, price_col: str = 'close', 
                  add_prefix: bool = False, **kwargs) -> pd.DataFrame:
        """
        计算RSI指标
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            price_col: 价格列名，默认为'close'
            add_prefix: 是否在输出列名前添加指标名称前缀
            kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含RSI指标的DataFrame
            
        Raises:
            ValueError: 如果输入数据不包含价格列
        """
        # 复制输入数据
        result = data.copy()
        
        # 处理周期列表
        periods = kwargs.get('periods', self.periods)
        if not isinstance(periods, list):
            periods = [periods]
        
        # 对每个周期计算RSI
        for period in periods:
            # 使用公共函数计算RSI
            rsi_values = calc_rsi(result[price_col].values, period)
            
            # 设置列名
            if add_prefix:
                rsi_col = self.get_column_name(str(period))
            else:
                rsi_col = f'RSI{period}'
            
            # 添加RSI列
            result[rsi_col] = rsi_values
            
            # 添加信号
            result = self.add_signals(result, rsi_col, period)
        
        # 如果是双周期RSI策略（6和14），添加协同判断信号
        if len(periods) >= 2 and 6 in periods and 14 in periods:
            result = self.add_dual_period_signals(result)
            
        # 保存结果
        self._result = result
        
        return result
    
    def get_buy_signal(self, data: pd.DataFrame, rsi_col: str = 'rsi') -> pd.Series:
        """
        获取RSI买入信号
        
        Args:
            data: 包含RSI指标的DataFrame
            rsi_col: RSI列名
            
        Returns:
            pd.Series: 买入信号序列，True表示买入信号
        """
        # 确保数据包含RSI列
        self.ensure_columns(data, [rsi_col])
        
        # 从超卖区域上穿超卖阈值
        cross_oversold = (data[rsi_col].shift(1) < self.oversold) & (data[rsi_col] >= self.oversold)
        
        # RSI从低位上升
        rising = data[rsi_col] > data[rsi_col].shift(1)
        
        # 综合信号：从超卖区域上穿 或 RSI在低位且上升
        buy_signal = cross_oversold | ((data[rsi_col] < 40) & rising)
        
        return buy_signal
    
    def get_sell_signal(self, data: pd.DataFrame, rsi_col: str = 'rsi') -> pd.Series:
        """
        获取RSI卖出信号
        
        Args:
            data: 包含RSI指标的DataFrame
            rsi_col: RSI列名
            
        Returns:
            pd.Series: 卖出信号序列，True表示卖出信号
        """
        # 确保数据包含RSI列
        self.ensure_columns(data, [rsi_col])
        
        # 从超买区域下穿超买阈值
        cross_overbought = (data[rsi_col].shift(1) > self.overbought) & (data[rsi_col] <= self.overbought)
        
        # RSI从高位下降
        falling = data[rsi_col] < data[rsi_col].shift(1)
        
        # 综合信号：从超买区域下穿 或 RSI在高位且下降
        sell_signal = cross_overbought | ((data[rsi_col] > 60) & falling)
        
        return sell_signal
    
    def add_signals(self, data: pd.DataFrame, rsi_col: str = 'rsi', period: int = None) -> pd.DataFrame:
        """
        添加RSI信号到数据
        
        Args:
            data: 包含RSI指标的DataFrame
            rsi_col: RSI列名
            period: RSI周期
            
        Returns:
            pd.DataFrame: 添加了RSI信号的DataFrame
        """
        # 确保数据包含RSI列
        self.ensure_columns(data, [rsi_col])
        
        # 复制输入数据
        result = data.copy()
        
        # 为多周期RSI添加后缀
        suffix = f"_{period}" if period is not None else ""
        
        # 添加信号列
        result[f'rsi_buy_signal{suffix}'] = self.get_buy_signal(result, rsi_col)
        result[f'rsi_sell_signal{suffix}'] = self.get_sell_signal(result, rsi_col)
        
        # 添加超买超卖区域标记
        result[f'rsi_overbought{suffix}'] = result[rsi_col] >= self.overbought
        result[f'rsi_oversold{suffix}'] = result[rsi_col] <= self.oversold
        
        # 添加趋势信息
        result[f'rsi_rising{suffix}'] = result[rsi_col] > result[rsi_col].shift(1)
        result[f'rsi_falling{suffix}'] = result[rsi_col] < result[rsi_col].shift(1)
        
        # 添加强度标签
        conditions = [
            (result[rsi_col] >= 80),
            (result[rsi_col] >= 60) & (result[rsi_col] < 80),
            (result[rsi_col] >= 40) & (result[rsi_col] < 60),
            (result[rsi_col] >= 20) & (result[rsi_col] < 40),
            (result[rsi_col] < 20)
        ]
        choices = ['极强', '强', '中性', '弱', '极弱']
        result[f'rsi_strength{suffix}'] = np.select(conditions, choices, default='未知')
        
        return result
    
    def add_dual_period_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加双周期RSI协同判断信号
        
        使用RSI6和RSI14两个周期的RSI值进行协同判断
        
        Args:
            data: 包含RSI6和RSI14的DataFrame
            
        Returns:
            pd.DataFrame: 添加了双周期RSI信号的DataFrame
        """
        # 确保数据包含RSI6和RSI14
        self.ensure_columns(data, ['RSI6', 'RSI14'])
        
        # 复制输入数据
        result = data.copy()
        
        # 双周期RSI买入信号：
        # 1. RSI6和RSI14都从超卖区上升
        # 2. RSI6上穿RSI14
        result['dual_rsi_buy_signal'] = (
            # 两个RSI都处于上升状态
            (result['rsi_rising_6'] & result['rsi_rising_14']) &
            (
                # 两个RSI都处于或曾处于超卖区
                ((result['RSI6'] <= 35) | (result['RSI6'].shift(1) <= 30)) &
                ((result['RSI14'] <= 40) | (result['RSI14'].shift(1) <= 35))
                |
                # 或者RSI6上穿RSI14（短期动能增强）
                ((result['RSI6'].shift(1) < result['RSI14'].shift(1)) &
                 (result['RSI6'] > result['RSI14']))
            )
        )
        
        # 双周期RSI卖出信号：
        # 1. RSI6和RSI14都从超买区下降
        # 2. RSI6下穿RSI14
        result['dual_rsi_sell_signal'] = (
            # 两个RSI都处于下降状态
            (result['rsi_falling_6'] & result['rsi_falling_14']) &
            (
                # 两个RSI都处于或曾处于超买区
                ((result['RSI6'] >= 65) | (result['RSI6'].shift(1) >= 70)) &
                ((result['RSI14'] >= 60) | (result['RSI14'].shift(1) >= 65))
                |
                # 或者RSI6下穿RSI14（短期动能减弱）
                ((result['RSI6'].shift(1) > result['RSI14'].shift(1)) &
                 (result['RSI6'] < result['RSI14']))
            )
        )
        
        # 多头趋势确认：RSI6>RSI14且两者都大于50
        result['dual_rsi_bullish'] = (
            (result['RSI6'] > result['RSI14']) &
            (result['RSI6'] > 50) & (result['RSI14'] > 50)
        )
        
        # 空头趋势确认：RSI6<RSI14且两者都小于50
        result['dual_rsi_bearish'] = (
            (result['RSI6'] < result['RSI14']) &
            (result['RSI6'] < 50) & (result['RSI14'] < 50)
        )
        
        # 顶背离信号：价格创新高但RSI未创新高
        if 'close' in data.columns:
            # 在前20个周期内寻找局部高点
            rolling_max_close = data['close'].rolling(window=20).max()
            rolling_max_rsi6 = data['RSI6'].rolling(window=20).max()
            rolling_max_rsi14 = data['RSI14'].rolling(window=20).max()
            
            # 价格创新高但RSI6和RSI14都未创新高
            result['dual_rsi_top_divergence'] = (
                (data['close'] >= rolling_max_close) &
                (data['RSI6'] < rolling_max_rsi6) &
                (data['RSI14'] < rolling_max_rsi14)
            )
            
            # 底背离信号：价格创新低但RSI未创新低
            rolling_min_close = data['close'].rolling(window=20).min()
            rolling_min_rsi6 = data['RSI6'].rolling(window=20).min()
            rolling_min_rsi14 = data['RSI14'].rolling(window=20).min()
            
            # 价格创新低但RSI6和RSI14都未创新低
            result['dual_rsi_bottom_divergence'] = (
                (data['close'] <= rolling_min_close) &
                (data['RSI6'] > rolling_min_rsi6) &
                (data['RSI14'] > rolling_min_rsi14)
            )
        
        return result
    
    def ensure_columns(self, data: pd.DataFrame, columns: List[str]) -> None:
        """
        确保DataFrame包含所需的列
        
        Args:
            data: 输入数据
            columns: 所需的列名列表
            
        Raises:
            ValueError: 如果数据不包含所需的列
        """
        missing_columns = [col for col in columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少所需的列: {', '.join(missing_columns)}")
    
    def get_column_name(self, suffix: str = "") -> str:
        """
        获取指标列名
        
        Args:
            suffix: 列名后缀
            
        Returns:
            str: 指标列名
        """
        if suffix:
            return f"{self.name.lower()}_{suffix}"
        return self.name.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将指标转换为字典表示
        
        Returns:
            Dict[str, Any]: 指标的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'periods': self.periods,
                'overbought': self.overbought,
                'oversold': self.oversold
            },
            'has_result': self.has_result(),
            'has_error': self.has_error(),
            'error': str(self._error) if self._error else None
        } 
    
    def is_dual_period_buy_signal(self, data: pd.DataFrame, index: int = -1) -> bool:
        """
        判断当前位置是否为双周期RSI买入信号
        
        Args:
            data: 包含双周期RSI信号的DataFrame
            index: 判断位置的索引，默认为最后一个
            
        Returns:
            bool: 是否为双周期RSI买入信号
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法")
            
        if 'dual_rsi_buy_signal' not in data.columns:
            raise ValueError("数据中缺少双周期RSI买入信号")
            
        if index == -1:
            index = len(data) - 1
            
        return bool(data['dual_rsi_buy_signal'].iloc[index])
    
    def is_dual_period_sell_signal(self, data: pd.DataFrame, index: int = -1) -> bool:
        """
        判断当前位置是否为双周期RSI卖出信号
        
        Args:
            data: 包含双周期RSI信号的DataFrame
            index: 判断位置的索引，默认为最后一个
            
        Returns:
            bool: 是否为双周期RSI卖出信号
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法")
            
        if 'dual_rsi_sell_signal' not in data.columns:
            raise ValueError("数据中缺少双周期RSI卖出信号")
            
        if index == -1:
            index = len(data) - 1
            
        return bool(data['dual_rsi_sell_signal'].iloc[index])
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算RSI原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算RSI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        rsi = self._result['RSI14']  # 使用正确的列名
        
        # 1. 超买超卖区域评分
        oversold_condition = rsi <= 30
        overbought_condition = rsi >= 70
        
        # 超卖区评分：RSI越低，加分越多
        oversold_score = np.where(oversold_condition, (30 - rsi) * 1.0, 0)  # 最多+30分
        score += oversold_score
        
        # 超买区评分：RSI越高，减分越多
        overbought_score = np.where(overbought_condition, (rsi - 70) * 1.0, 0)  # 最多-30分
        score -= overbought_score
        
        # 2. RSI穿越关键位置评分
        rsi_cross_up_30 = self.crossover(rsi, 30)
        rsi_cross_down_70 = self.crossunder(rsi, 70)
        score += rsi_cross_up_30 * 20    # RSI上穿30+20分
        score -= rsi_cross_down_70 * 20  # RSI下穿70-20分
        
        # 3. 中线穿越评分
        rsi_cross_up_50 = self.crossover(rsi, 50)
        rsi_cross_down_50 = self.crossunder(rsi, 50)
        score += rsi_cross_up_50 * 15    # RSI上穿50+15分
        score -= rsi_cross_down_50 * 15  # RSI下穿50-15分
        
        # 4. RSI背离评分
        if len(data) >= 20:
            divergence_score = self._calculate_rsi_divergence(data['close'], rsi)
            score += divergence_score
        
        # 5. RSI形态评分
        pattern_score = self._calculate_rsi_pattern_score(rsi)
        score += pattern_score
        
        # 6. RSI斜率评分
        slope_score = self._calculate_rsi_slope_score(rsi)
        score += slope_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别RSI技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算RSI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        rsi = self._result['RSI14']
        
        # 检查最近的信号
        recent_periods = min(10, len(rsi))
        if recent_periods == 0:
            return patterns
        
        recent_rsi = rsi.tail(recent_periods)
        current_rsi = recent_rsi.iloc[-1]
        
        # 1. 超买超卖形态
        if current_rsi <= 20:
            patterns.append("RSI极度超卖")
        elif current_rsi <= 30:
            patterns.append("RSI超卖")
        elif current_rsi >= 80:
            patterns.append("RSI极度超买")
        elif current_rsi >= 70:
            patterns.append("RSI超买")
        
        # 2. 穿越形态
        if self.crossover(recent_rsi, 30).any():
            patterns.append("RSI上穿30")
        if self.crossunder(recent_rsi, 70).any():
            patterns.append("RSI下穿70")
        if self.crossover(recent_rsi, 50).any():
            patterns.append("RSI上穿中线")
        if self.crossunder(recent_rsi, 50).any():
            patterns.append("RSI下穿中线")
        
        # 3. W底和M顶形态
        if self._detect_w_bottom_pattern(recent_rsi):
            patterns.append("RSI W底形态")
        if self._detect_m_top_pattern(recent_rsi):
            patterns.append("RSI M顶形态")
        
        # 4. 背离形态
        if len(data) >= 20:
            divergence_type = self._detect_rsi_divergence_pattern(data['close'], rsi)
            if divergence_type:
                patterns.append(f"RSI{divergence_type}")
        
        # 5. 钝化形态
        if self._detect_rsi_stagnation(recent_rsi, threshold=20, periods=5, direction='low'):
            patterns.append("RSI低位钝化")
        if self._detect_rsi_stagnation(recent_rsi, threshold=80, periods=5, direction='high'):
            patterns.append("RSI高位钝化")
        
        return patterns
    
    def _calculate_rsi_divergence(self, price: pd.Series, rsi: pd.Series) -> pd.Series:
        """
        计算RSI背离评分
        
        Args:
            price: 价格序列
            rsi: RSI序列
            
        Returns:
            pd.Series: 背离评分序列
        """
        divergence_score = pd.Series(0.0, index=price.index)
        
        if len(price) < 20:
            return divergence_score
        
        # 寻找价格和RSI的峰值谷值
        window = 5
        for i in range(window, len(price) - window):
            price_window = price.iloc[i-window:i+window+1]
            rsi_window = rsi.iloc[i-window:i+window+1]
            
            if price.iloc[i] == price_window.max():  # 价格峰值
                if rsi.iloc[i] != rsi_window.max():  # RSI未创新高
                    divergence_score.iloc[i:i+10] -= 25  # 负背离
            elif price.iloc[i] == price_window.min():  # 价格谷值
                if rsi.iloc[i] != rsi_window.min():  # RSI未创新低
                    divergence_score.iloc[i:i+10] += 25  # 正背离
        
        return divergence_score
    
    def _calculate_rsi_pattern_score(self, rsi: pd.Series) -> pd.Series:
        """
        计算RSI形态评分
        
        Args:
            rsi: RSI序列
            
        Returns:
            pd.Series: 形态评分序列
        """
        pattern_score = pd.Series(0.0, index=rsi.index)
        
        if len(rsi) < 20:
            return pattern_score
        
        # W底形态检测
        for i in range(10, len(rsi) - 10):
            window_rsi = rsi.iloc[i-10:i+10]
            if self._detect_w_bottom_pattern(window_rsi):
                pattern_score.iloc[i:i+5] += 25
        
        # M顶形态检测
        for i in range(10, len(rsi) - 10):
            window_rsi = rsi.iloc[i-10:i+10]
            if self._detect_m_top_pattern(window_rsi):
                pattern_score.iloc[i:i+5] -= 25
        
        return pattern_score
    
    def _calculate_rsi_slope_score(self, rsi: pd.Series) -> pd.Series:
        """
        计算RSI斜率评分
        
        Args:
            rsi: RSI序列
            
        Returns:
            pd.Series: 斜率评分序列
        """
        slope_score = pd.Series(0.0, index=rsi.index)
        
        if len(rsi) < 5:
            return slope_score
        
        # 计算5周期RSI斜率
        rsi_slope = rsi.diff(5)
        
        # 斜率评分
        slope_score += np.where(rsi_slope > 5, 10, 0)   # 强烈上升+10分
        slope_score += np.where(rsi_slope > 2, 5, 0)    # 温和上升+5分
        slope_score -= np.where(rsi_slope < -5, 10, 0)  # 强烈下降-10分
        slope_score -= np.where(rsi_slope < -2, 5, 0)   # 温和下降-5分
        
        return slope_score
    
    def _detect_w_bottom_pattern(self, rsi: pd.Series) -> bool:
        """
        检测W底形态
        
        Args:
            rsi: RSI序列
            
        Returns:
            bool: 是否存在W底形态
        """
        if len(rsi) < 10:
            return False
        
        # 简化的W底检测：寻找两个相近的低点
        min_indices = []
        for i in range(2, len(rsi) - 2):
            if (rsi.iloc[i] < rsi.iloc[i-1] and rsi.iloc[i] < rsi.iloc[i+1] and
                rsi.iloc[i] < rsi.iloc[i-2] and rsi.iloc[i] < rsi.iloc[i+2]):
                min_indices.append(i)
        
        if len(min_indices) >= 2:
            # 检查两个低点是否相近且都在超卖区
            last_two_mins = min_indices[-2:]
            min1_val = rsi.iloc[last_two_mins[0]]
            min2_val = rsi.iloc[last_two_mins[1]]
            
            if (abs(min1_val - min2_val) < 5 and 
                min1_val < 35 and min2_val < 35):
                return True
        
        return False
    
    def _detect_m_top_pattern(self, rsi: pd.Series) -> bool:
        """
        检测M顶形态
        
        Args:
            rsi: RSI序列
            
        Returns:
            bool: 是否存在M顶形态
        """
        if len(rsi) < 10:
            return False
        
        # 简化的M顶检测：寻找两个相近的高点
        max_indices = []
        for i in range(2, len(rsi) - 2):
            if (rsi.iloc[i] > rsi.iloc[i-1] and rsi.iloc[i] > rsi.iloc[i+1] and
                rsi.iloc[i] > rsi.iloc[i-2] and rsi.iloc[i] > rsi.iloc[i+2]):
                max_indices.append(i)
        
        if len(max_indices) >= 2:
            # 检查两个高点是否相近且都在超买区
            last_two_maxs = max_indices[-2:]
            max1_val = rsi.iloc[last_two_maxs[0]]
            max2_val = rsi.iloc[last_two_maxs[1]]
            
            if (abs(max1_val - max2_val) < 5 and 
                max1_val > 65 and max2_val > 65):
                return True
        
        return False
    
    def _detect_rsi_divergence_pattern(self, price: pd.Series, rsi: pd.Series) -> Optional[str]:
        """
        检测RSI背离形态
        
        Args:
            price: 价格序列
            rsi: RSI序列
            
        Returns:
            Optional[str]: 背离类型或None
        """
        if len(price) < 20:
            return None
        
        # 寻找最近的峰值和谷值
        recent_price = price.tail(20)
        recent_rsi = rsi.tail(20)
        
        price_extremes = []
        rsi_extremes = []
        
        # 寻找极值点
        for i in range(2, len(recent_price) - 2):
            if (recent_price.iloc[i] > recent_price.iloc[i-1] and 
                recent_price.iloc[i] > recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                rsi_extremes.append(recent_rsi.iloc[i])
            elif (recent_price.iloc[i] < recent_price.iloc[i-1] and 
                  recent_price.iloc[i] < recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                rsi_extremes.append(recent_rsi.iloc[i])
        
        if len(price_extremes) >= 2:
            price_trend = price_extremes[-1] - price_extremes[-2]
            rsi_trend = rsi_extremes[-1] - rsi_extremes[-2]
            
            # 正背离：价格创新低但RSI未创新低
            if price_trend < -0.01 and rsi_trend > 2:
                return "正背离"
            # 负背离：价格创新高但RSI未创新高
            elif price_trend > 0.01 and rsi_trend < -2:
                return "负背离"
        
        return None
    
    def _detect_rsi_stagnation(self, rsi: pd.Series, threshold: float, 
                              periods: int, direction: str) -> bool:
        """
        检测RSI钝化
        
        Args:
            rsi: RSI序列
            threshold: 阈值
            periods: 检测周期数
            direction: 方向 ('low' 或 'high')
            
        Returns:
            bool: 是否钝化
        """
        if len(rsi) < periods:
            return False
        
        recent_rsi = rsi.tail(periods)
        
        if direction == 'low':
            return (recent_rsi < threshold).all()
        elif direction == 'high':
            return (recent_rsi > threshold).all()
        
        return False 