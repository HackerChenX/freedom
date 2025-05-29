"""
MACD指标模块

实现MACD指标的计算和相关功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import macd as calc_macd, cross
from enums.indicator_types import CrossType


class MACD(BaseIndicator):
    """
    MACD(Moving Average Convergence Divergence)指标
    
    MACD是一种趋势跟踪的动量指标，通过计算两条不同周期的指数移动平均线之差，
    以及该差值的移动平均线来判断市场趋势和动量。
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        初始化MACD指标
        
        Args:
            fast_period: 快线周期，默认为12
            slow_period: 慢线周期，默认为26
            signal_period: 信号线周期，默认为9
        """
        super().__init__(name="MACD", description="移动平均线收敛散度指标")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.DataFrame, price_col: str = 'close', 
                  add_prefix: bool = False, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            price_col: 价格列名，默认为'close'
            add_prefix: 是否在输出列名前添加指标名称前缀
            kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含MACD指标的DataFrame
            
        Raises:
            ValueError: 如果输入数据不包含价格列
        """
        # 确保数据包含价格列
        self.ensure_columns(data, [price_col])
        
        # 复制输入数据
        result = data.copy()
        
        # 使用统一的公共函数计算MACD
        dif, dea, macd_hist = calc_macd(
            result[price_col].values,
            self.fast_period,
            self.slow_period,
            self.signal_period
        )
        
        # 确保前N个值为NaN，其中N = max(fast_period, slow_period) - 1
        min_periods = max(self.fast_period, self.slow_period) - 1
        dif[:min_periods] = np.nan
        dea[:min_periods + self.signal_period - 1] = np.nan
        macd_hist[:min_periods + self.signal_period - 1] = np.nan
        
        # 设置列名
        if add_prefix:
            macd_col = self.get_column_name('DIF')
            signal_col = self.get_column_name('DEA')
            hist_col = self.get_column_name('MACD')
        else:
            macd_col = 'DIF'
            signal_col = 'DEA'
            hist_col = 'MACD'
        
        # 添加结果列
        result[macd_col] = dif
        result[signal_col] = dea
        result[hist_col] = macd_hist
        
        # 添加信号
        result = self.add_signals(result, macd_col, signal_col, hist_col)
        
        return result
    
    def add_signals(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                   signal_col: str = 'DEA', hist_col: str = 'MACD') -> pd.DataFrame:
        """
        添加MACD交易信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            hist_col: 柱状图列名(MACD)
            
        Returns:
            pd.DataFrame: 添加了信号的DataFrame
        """
        result = data.copy()
        
        # 计算金叉和死叉信号
        result['macd_buy_signal'] = self.get_buy_signal(result, macd_col, signal_col)
        result['macd_sell_signal'] = self.get_sell_signal(result, macd_col, signal_col)
        
        # 计算零轴穿越信号
        result['macd_zero_cross_up'] = (result[macd_col] > 0) & (result[macd_col].shift(1) <= 0)
        result['macd_zero_cross_down'] = (result[macd_col] < 0) & (result[macd_col].shift(1) >= 0)
        
        # 计算柱状图趋势
        result['macd_hist_increasing'] = result[hist_col] > result[hist_col].shift(1)
        result['macd_hist_decreasing'] = result[hist_col] < result[hist_col].shift(1)
        
        # 计算背离指标
        result = self._add_divergence_signals(result, macd_col, price_col='close')
        
        return result
    
    def get_buy_signal(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                      signal_col: str = 'DEA') -> pd.Series:
        """
        获取MACD买入信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            
        Returns:
            pd.Series: 买入信号序列（布尔值）
        """
        # 使用公共cross函数检测金叉
        return pd.Series(
            cross(data[macd_col].values, data[signal_col].values),
            index=data.index
        )
    
    def get_sell_signal(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                       signal_col: str = 'DEA') -> pd.Series:
        """
        获取MACD卖出信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            
        Returns:
            pd.Series: 卖出信号序列（布尔值）
        """
        # 使用公共cross函数检测死叉
        return pd.Series(
            cross(data[signal_col].values, data[macd_col].values),
            index=data.index
        )
    
    def _add_divergence_signals(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                              price_col: str = 'close', window: int = 20) -> pd.DataFrame:
        """
        添加MACD背离信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            price_col: 价格列名
            window: 寻找背离的窗口大小
            
        Returns:
            pd.DataFrame: 添加了背离信号的DataFrame
        """
        result = data.copy()
        
        # 初始化背离信号列
        result['macd_bullish_divergence'] = False
        result['macd_bearish_divergence'] = False
        
        # 循环检测背离
        for i in range(window, len(result)):
            # 只检查窗口内的数据
            window_data = result.iloc[i-window:i+1]
            
            # 计算价格的局部最低点
            price_lows = window_data[window_data[price_col] == window_data[price_col].min()]
            
            # 计算价格的局部最高点
            price_highs = window_data[window_data[price_col] == window_data[price_col].max()]
            
            # 计算MACD的局部最低点
            macd_lows = window_data[window_data[macd_col] == window_data[macd_col].min()]
            
            # 计算MACD的局部最高点
            macd_highs = window_data[window_data[macd_col] == window_data[macd_col].max()]
            
            # 检查是否有足够的点来比较
            if len(price_lows) > 1 and len(macd_lows) > 1:
                # 检查看涨背离：价格创新低但MACD没有创新低
                if (price_lows.iloc[-1][price_col] < price_lows.iloc[0][price_col] and 
                    macd_lows.iloc[-1][macd_col] > macd_lows.iloc[0][macd_col]):
                    result.loc[result.index[i], 'macd_bullish_divergence'] = True
            
            # 检查是否有足够的点来比较
            if len(price_highs) > 1 and len(macd_highs) > 1:
                # 检查看跌背离：价格创新高但MACD没有创新高
                if (price_highs.iloc[-1][price_col] > price_highs.iloc[0][price_col] and 
                    macd_highs.iloc[-1][macd_col] < macd_highs.iloc[0][macd_col]):
                    result.loc[result.index[i], 'macd_bearish_divergence'] = True
        
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
    
    def get_cross_points(self, data: pd.DataFrame, cross_type: CrossType = CrossType.GOLDEN_CROSS,
                        macd_col: str = 'DIF', signal_col: str = 'DEA') -> pd.DataFrame:
        """
        获取MACD交叉点
        
        Args:
            data: 包含MACD指标的DataFrame
            cross_type: 交叉类型，金叉或死叉
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            
        Returns:
            pd.DataFrame: 交叉点DataFrame
        """
        if cross_type == CrossType.GOLDEN_CROSS:
            # 金叉：MACD从下方穿过信号线
            cross_points = data[self.get_buy_signal(data, macd_col, signal_col)]
        else:
            # 死叉：MACD从上方穿过信号线
            cross_points = data[self.get_sell_signal(data, macd_col, signal_col)]
        
        return cross_points

    def to_dict(self) -> Dict:
        """
        将指标转换为字典表示
        
        Returns:
            Dict: 指标的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period
            },
            'has_result': self.has_result(),
            'has_error': self.has_error(),
            'error': str(self._error) if self._error else None
        }
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成MACD信号
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含信号的DataFrame
        """
        # 计算MACD指标
        if not self.has_result():
            self.compute(data)
            
        if not self.has_result():
            return pd.DataFrame()
            
        # 获取MACD值
        dif = self._result['DIF']
        dea = self._result['DEA']
        macd_hist = self._result['MACD']
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 添加买入信号 - MACD金叉（DIF上穿DEA）
        signals['buy_signal'] = self.get_buy_signal(self._result)
        
        # 添加卖出信号 - MACD死叉（DIF下穿DEA）
        signals['sell_signal'] = self.get_sell_signal(self._result)
        
        # 添加零轴穿越信号
        signals['zero_cross_up'] = (dif > 0) & (dif.shift(1) <= 0)
        signals['zero_cross_down'] = (dif < 0) & (dif.shift(1) >= 0)
        
        # 柱状图趋势
        signals['hist_increasing'] = macd_hist > macd_hist.shift(1)
        signals['hist_decreasing'] = macd_hist < macd_hist.shift(1)
        
        # 计算信号强度
        # 范围是0-100，0表示最弱，100表示最强
        strength = 50.0  # 默认中性
        
        # 如果出现金叉，信号强度增加
        if signals['buy_signal'].iloc[-1]:
            strength += 25.0
            
        # 如果DIF穿越零轴向上，信号强度增加
        if signals['zero_cross_up'].iloc[-1]:
            strength += 15.0
            
        # 如果柱状图增加，信号强度增加
        if signals['hist_increasing'].iloc[-1]:
            strength += 10.0
            
        # 如果DIF和DEA都大于0，信号强度增加
        if dif.iloc[-1] > 0 and dea.iloc[-1] > 0:
            strength += 10.0
            
        # 如果出现死叉，信号强度减少
        if signals['sell_signal'].iloc[-1]:
            strength -= 25.0
            
        # 如果DIF穿越零轴向下，信号强度减少
        if signals['zero_cross_down'].iloc[-1]:
            strength -= 15.0
            
        # 如果柱状图减少，信号强度减少
        if signals['hist_decreasing'].iloc[-1]:
            strength -= 10.0
            
        # 确保强度在0-100范围内
        strength = max(0.0, min(100.0, strength))
        
        # 添加信号强度
        signals['signal_strength'] = 0.0
        signals.loc[signals.index[-1], 'signal_strength'] = strength
        
        return signals

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算MACD
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        dif = self._result['DIF']
        dea = self._result['DEA']
        macd_hist = self._result['MACD']
        
        # 1. 金叉死叉评分
        golden_cross = self.crossover(dif, dea)
        death_cross = self.crossunder(dif, dea)
        score += golden_cross * 20  # 金叉+20分
        score -= death_cross * 20   # 死叉-20分
        
        # 2. 零轴位置评分
        above_zero = (dif > 0) & (dea > 0)
        below_zero = (dif < 0) & (dea < 0)
        score += above_zero * 10    # 零轴上方+10分
        score -= below_zero * 10    # 零轴下方-10分
        
        # 3. 零轴穿越评分
        dif_cross_up = self.crossover(dif, 0)
        dif_cross_down = self.crossunder(dif, 0)
        score += dif_cross_up * 15    # DIF上穿零轴+15分
        score -= dif_cross_down * 15  # DIF下穿零轴-15分
        
        # 4. MACD柱状图评分
        hist_turn_positive = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
        hist_turn_negative = (macd_hist < 0) & (macd_hist.shift(1) >= 0)
        score += hist_turn_positive * 12  # 柱状图由负转正+12分
        score -= hist_turn_negative * 12  # 柱状图由正转负-12分
        
        # 5. 背离评分（简化版）
        if len(data) >= 20:
            # 检测价格和MACD的背离
            price_peaks = self._find_peaks(data['close'], 10)
            macd_peaks = self._find_peaks(dif, 10)
            
            if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
                # 简单的背离检测
                price_trend = price_peaks[-1] - price_peaks[-2]
                macd_trend = macd_peaks[-1] - macd_peaks[-2]
                
                # 正背离：价格创新低但MACD未创新低
                if price_trend < 0 and macd_trend > 0:
                    score.iloc[-10:] += 25  # 最近10个周期+25分
                # 负背离：价格创新高但MACD未创新高
                elif price_trend > 0 and macd_trend < 0:
                    score.iloc[-10:] -= 25  # 最近10个周期-25分
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别MACD技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算MACD
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        dif = self._result['DIF']
        dea = self._result['DEA']
        macd_hist = self._result['MACD']
        
        # 检查最近的信号
        recent_periods = min(5, len(dif))
        if recent_periods == 0:
            return patterns
        
        recent_dif = dif.tail(recent_periods)
        recent_dea = dea.tail(recent_periods)
        recent_hist = macd_hist.tail(recent_periods)
        
        # 1. 金叉死叉形态
        if self.crossover(recent_dif, recent_dea).any():
            if recent_dif.iloc[-1] > 0 and recent_dea.iloc[-1] > 0:
                patterns.append("MACD零轴上方金叉")
            else:
                patterns.append("MACD零轴下方金叉")
        
        if self.crossunder(recent_dif, recent_dea).any():
            if recent_dif.iloc[-1] > 0 and recent_dea.iloc[-1] > 0:
                patterns.append("MACD零轴上方死叉")
            else:
                patterns.append("MACD零轴下方死叉")
        
        # 2. 零轴穿越形态
        if self.crossover(recent_dif, 0).any():
            patterns.append("MACD DIF上穿零轴")
        if self.crossunder(recent_dif, 0).any():
            patterns.append("MACD DIF下穿零轴")
        
        # 3. 柱状图形态
        if (recent_hist.iloc[-1] > 0 and recent_hist.iloc[-2] <= 0):
            patterns.append("MACD柱状图由负转正")
        if (recent_hist.iloc[-1] < 0 and recent_hist.iloc[-2] >= 0):
            patterns.append("MACD柱状图由正转负")
        
        # 4. 背离形态检测
        if len(data) >= 20:
            divergence_type = self._detect_divergence(data['close'], dif)
            if divergence_type:
                patterns.append(f"MACD{divergence_type}")
        
        return patterns
    
    def _find_peaks(self, series: pd.Series, window: int) -> List[float]:
        """
        寻找序列中的峰值
        
        Args:
            series: 输入序列
            window: 窗口大小
            
        Returns:
            List[float]: 峰值列表
        """
        peaks = []
        if len(series) < window * 2:
            return peaks
        
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                peaks.append(series.iloc[i])
        
        return peaks
    
    def _detect_divergence(self, price: pd.Series, indicator: pd.Series) -> Optional[str]:
        """
        检测背离形态
        
        Args:
            price: 价格序列
            indicator: 指标序列
            
        Returns:
            Optional[str]: 背离类型或None
        """
        if len(price) < 20:
            return None
        
        # 寻找最近的峰值和谷值
        price_peaks = self._find_peaks(price, 5)
        indicator_peaks = self._find_peaks(indicator, 5)
        
        if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
            price_trend = price_peaks[-1] - price_peaks[-2]
            indicator_trend = indicator_peaks[-1] - indicator_peaks[-2]
            
            # 正背离：价格创新低但指标未创新低
            if price_trend < -0.01 and indicator_trend > 0.001:
                return "正背离"
            # 负背离：价格创新高但指标未创新高
            elif price_trend > 0.01 and indicator_trend < -0.001:
                return "负背离"
        
        return None 