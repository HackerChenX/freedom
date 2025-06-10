"""
统一移动平均线指标模块

整合多种移动平均线计算方法，包括:
- 简单移动平均线(SMA)
- 指数移动平均线(EMA)
- 加权移动平均线(WMA)
- 自适应移动平均线(AMA)
- Hull移动平均线(HMA)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Literal

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedMA(BaseIndicator):
    """
    统一移动平均线指标
    
    提供多种移动平均线计算方法，并支持自适应参数调整
    """
    
    # MA类型枚举
    MA_TYPE_SIMPLE = 'simple'    # 简单移动平均
    MA_TYPE_EMA = 'ema'          # 指数移动平均
    MA_TYPE_WMA = 'wma'          # 加权移动平均
    MA_TYPE_AMA = 'ama'          # 自适应移动平均
    MA_TYPE_HMA = 'hma'          # Hull移动平均
    
    # 支持的MA类型列表
    SUPPORTED_TYPES = [MA_TYPE_SIMPLE, MA_TYPE_EMA, MA_TYPE_WMA, MA_TYPE_AMA, MA_TYPE_HMA]
    
    def __init__(self, 
                 name: str = "UnifiedMA", 
                 description: str = "统一移动平均线指标",
                 periods: List[int] = None,
                 ma_type: str = MA_TYPE_SIMPLE):
        """
        初始化统一移动平均线指标
        
        Args:
            name: 指标名称
            description: 指标描述
            periods: 周期列表，默认为[5, 10, 20, 30, 60]
            ma_type: 移动平均线类型，可选值为simple, ema, wma, ama, hma
        """
        super().__init__(name, description)
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
        # 设置默认参数
        self._parameters = {
            'periods': periods or [5, 10, 20, 30, 60],  # 周期列表
            'ma_type': ma_type,                         # MA类型
            'weight': 1,                               # SMA的权重参数
            'fast_period': 2,                          # AMA快速周期
            'slow_period': 30,                         # AMA慢速周期
            'efficiency_period': 10,                   # AMA效率系数计算周期
            'price_col': 'close'                       # 价格列名
        }
        
        # 验证MA类型是否支持
        if ma_type not in self.SUPPORTED_TYPES:
            logger.warning(f"不支持的MA类型: {ma_type}，将使用默认类型: {self.MA_TYPE_SIMPLE}")
            self._parameters['ma_type'] = self.MA_TYPE_SIMPLE
    
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
                
        # 验证MA类型是否支持
        if 'ma_type' in params and params['ma_type'] not in self.SUPPORTED_TYPES:
            logger.warning(f"不支持的MA类型: {params['ma_type']}，将使用默认类型: {self.MA_TYPE_SIMPLE}")
            self._parameters['ma_type'] = self.MA_TYPE_SIMPLE
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算移动平均线指标
        
        Args:
            data: 输入数据，必须包含价格列
            args: 位置参数
            kwargs: 关键字参数，可包含periods、ma_type、weight等
            
        Returns:
            pd.DataFrame: 包含各周期MA的DataFrame
        """
        # 处理输入参数
        params = self._parameters.copy()
        params.update(kwargs)
        
        price_col = params.get('price_col', 'close')
        periods = params.get('periods', self._parameters['periods'])
        ma_type = params.get('ma_type', self._parameters['ma_type'])
        
        # 确保数据包含价格列
        self.ensure_columns(data, [price_col])
        
        # 如果periods是单个值，转换为列表
        if not isinstance(periods, list):
            periods = [periods]
        
        # 计算各周期的移动平均线
        result = data.copy()
        
        for period in periods:
            # 根据MA类型计算
            if ma_type == self.MA_TYPE_EMA:
                result[f'MA{period}'] = self._calculate_ema(data[price_col], period)
            elif ma_type == self.MA_TYPE_WMA:
                result[f'MA{period}'] = self._calculate_wma(data[price_col], period)
            elif ma_type == self.MA_TYPE_AMA:
                result[f'MA{period}'] = self._calculate_ama(
                    data[price_col], 
                    period,
                    params.get('efficiency_period', self._parameters['efficiency_period']),
                    params.get('fast_period', self._parameters['fast_period']),
                    params.get('slow_period', self._parameters['slow_period'])
                )
            elif ma_type == self.MA_TYPE_HMA:
                result[f'MA{period}'] = self._calculate_hma(data[price_col], period)
            else:  # 默认使用简单移动平均
                result[f'MA{period}'] = self._calculate_sma(data[price_col], period)
        
        return result
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """
        计算简单移动平均线
        
        Args:
            series: 输入序列
            period: 周期
            
        Returns:
            pd.Series: 简单移动平均线
        """
        return series.rolling(window=period).mean()
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        计算指数移动平均线
        
        Args:
            series: 输入序列
            period: 周期
            
        Returns:
            pd.Series: 指数移动平均线
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_wma(self, series: pd.Series, period: int) -> pd.Series:
        """
        计算加权移动平均线
        
        Args:
            series: 输入序列
            period: 周期
            
        Returns:
            pd.Series: 加权移动平均线
        """
        weights = np.arange(1, period + 1)
        result = series.rolling(window=period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )
        return result
    
    def _calculate_ama(self, series: pd.Series, period: int, 
                       efficiency_period: int = 10, 
                       fast_period: int = 2, 
                       slow_period: int = 30) -> pd.Series:
        """
        计算自适应移动平均线 (Kaufman's Adaptive Moving Average)
        
        Args:
            series: 输入序列
            period: 周期
            efficiency_period: 效率系数计算周期
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            
        Returns:
            pd.Series: 自适应移动平均线
        """
        # 计算方向变化
        direction = (series - series.shift(efficiency_period)).abs()
        
        # 计算波动
        volatility = series.diff().abs().rolling(window=efficiency_period).sum()
        
        # 计算效率系数
        efficiency_ratio = pd.Series(np.where(volatility != 0, direction / volatility, 0), index=series.index)
        
        # 计算平滑系数
        fast_sc = 2.0 / (fast_period + 1.0)
        slow_sc = 2.0 / (slow_period + 1.0)
        
        # 计算自适应系数
        adaptive_factor = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # 初始化AMA
        ama = pd.Series(index=series.index, dtype=float)
        ama.iloc[0] = series.iloc[0] if not np.isnan(series.iloc[0]) else 0
        
        # 计算AMA
        for i in range(1, len(series)):
            if np.isnan(series.iloc[i]) or np.isnan(adaptive_factor.iloc[i]):
                ama.iloc[i] = ama.iloc[i-1]
            else:
                ama.iloc[i] = adaptive_factor.iloc[i] * (series.iloc[i] - ama.iloc[i-1]) + ama.iloc[i-1]
        
        return ama
    
    def _calculate_hma(self, series: pd.Series, period: int) -> pd.Series:
        """
        计算Hull移动平均线 (Hull Moving Average)
        
        Args:
            series: 输入序列
            period: 周期
            
        Returns:
            pd.Series: Hull移动平均线
        """
        # 计算WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        # 计算半周期WMA
        wma_half = self._calculate_wma(series, half_period)
        
        # 计算全周期WMA
        wma_full = self._calculate_wma(series, period)
        
        # 计算2*WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full
        
        # 计算HMA
        hma = self._calculate_wma(raw_hma, sqrt_period)
        
        return hma
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成均线交叉信号
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            pd.DataFrame: 包含信号的DataFrame
        """
        # 计算均线
        if not self.has_result():
            self.compute(data)
            
        if not self.has_result():
            return pd.DataFrame()
        
        # 获取结果
        ma_data = self._result
        
        # 获取参数
        price_col = self._parameters.get('price_col', 'close')
        periods = sorted(self._parameters.get('periods', [5, 10, 20, 30, 60]))
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 计算价格与均线交叉信号
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in ma_data.columns:
                # 价格上穿均线
                signals[f'price_above_{ma_col}'] = data[price_col] > ma_data[ma_col]
                signals[f'price_cross_above_{ma_col}'] = self.crossover(data[price_col], ma_data[ma_col])
                signals[f'price_cross_below_{ma_col}'] = self.crossunder(data[price_col], ma_data[ma_col])
        
        # 计算均线交叉信号
        if len(periods) >= 2:
            for i in range(len(periods)-1):
                short_period = periods[i]
                for j in range(i+1, len(periods)):
                    long_period = periods[j]
                    short_ma = f'MA{short_period}'
                    long_ma = f'MA{long_period}'
                    if short_ma in ma_data.columns and long_ma in ma_data.columns:
                        # 短期均线上穿长期均线
                        signals[f'golden_cross_{short_ma}_{long_ma}'] = self.crossover(
                            ma_data[short_ma], ma_data[long_ma]
                        )
                        # 短期均线下穿长期均线
                        signals[f'death_cross_{short_ma}_{long_ma}'] = self.crossunder(
                            ma_data[short_ma], ma_data[long_ma]
                        )
        
        # 计算多头排列和空头排列
        if len(periods) >= 3:
            # 检查是否多头排列（短期均线在上，依次降低）
            bull_trend = pd.Series(True, index=data.index)
            for i in range(len(periods)-1):
                short_ma = f'MA{periods[i]}'
                long_ma = f'MA{periods[i+1]}'
                if short_ma in ma_data.columns and long_ma in ma_data.columns:
                    bull_trend &= (ma_data[short_ma] > ma_data[long_ma])
            
            # 检查是否空头排列（短期均线在下，依次升高）
            bear_trend = pd.Series(True, index=data.index)
            for i in range(len(periods)-1):
                short_ma = f'MA{periods[i]}'
                long_ma = f'MA{periods[i+1]}'
                if short_ma in ma_data.columns and long_ma in ma_data.columns:
                    bear_trend &= (ma_data[short_ma] < ma_data[long_ma])
            
            signals['bull_trend'] = bull_trend
            signals['bear_trend'] = bear_trend
        
        # 添加买入卖出信号
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        
        # 简单策略：短期均线上穿长期均线，且价格在中期均线上方
        if len(periods) >= 3:
            short_ma = f'MA{periods[0]}'
            mid_ma = f'MA{periods[1]}'
            long_ma = f'MA{periods[2]}'
            
            if all(col in ma_data.columns for col in [short_ma, mid_ma, long_ma]):
                # 买入信号：短期均线上穿中期均线，且价格在长期均线上方
                signals['buy_signal'] = (
                    self.crossover(ma_data[short_ma], ma_data[mid_ma]) & 
                    (data[price_col] > ma_data[long_ma])
                )
                
                # 卖出信号：短期均线下穿中期均线，且价格在长期均线下方
                signals['sell_signal'] = (
                    self.crossunder(ma_data[short_ma], ma_data[mid_ma]) & 
                    (data[price_col] < ma_data[long_ma])
                )
        
        return signals
    
    def get_ma_trend(self, period: int = None) -> pd.Series:
        """
        获取指定周期的MA趋势
        
        Args:
            period: MA周期，若为None则使用最短周期
            
        Returns:
            pd.Series: 趋势序列，1表示上升，0表示平，-1表示下降
        """
        if not self.has_result():
            logger.warning("尚未计算MA，无法获取趋势")
            return pd.Series()
            
        # 获取周期
        if period is None:
            periods = self._parameters.get('periods', [5, 10, 20, 30, 60])
            period = min(periods)
            
        # 获取对应的MA列
        ma_col = f'MA{period}'
        if ma_col not in self._result.columns:
            logger.warning(f"结果中不存在{ma_col}列，无法获取趋势")
            return pd.Series()
            
        # 计算MA的变化率
        ma_values = self._result[ma_col]
        ma_change = ma_values.diff()
        
        # 根据变化率判断趋势
        trend = pd.Series(0, index=ma_values.index)
        trend[ma_change > 0] = 1
        trend[ma_change < 0] = -1
        
        return trend
    
    def is_consolidation(self, period: int = None, threshold: float = 0.01) -> pd.Series:
        """
        检测是否处于盘整状态（MA走平）
        
        Args:
            period: MA周期，若为None则使用最短周期
            threshold: 变化率阈值，低于此值视为盘整
            
        Returns:
            pd.Series: 盘整状态序列，True表示盘整，False表示趋势
        """
        if not self.has_result():
            logger.warning("尚未计算MA，无法检测盘整")
            return pd.Series()
            
        # 获取周期
        if period is None:
            periods = self._parameters.get('periods', [5, 10, 20, 30, 60])
            period = min(periods)
            
        # 获取对应的MA列
        ma_col = f'MA{period}'
        if ma_col not in self._result.columns:
            logger.warning(f"结果中不存在{ma_col}列，无法检测盘整")
            return pd.Series()
            
        # 计算MA的变化率绝对值
        ma_values = self._result[ma_col]
        ma_change_pct = ma_values.pct_change().abs()
        
        # 判断是否盘整
        consolidation = ma_change_pct < threshold
        
        return consolidation
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算统一移动平均线指标的原始评分
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列，0-100
        """
        # 计算指标
        if not self.has_result():
            result = self.calculate(data, **kwargs)
        else:
            result = self._result.copy()
            
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 获取参数
        params = self._parameters.copy()
        params.update(kwargs)
        price_col = params.get('price_col', 'close')
        periods = params.get('periods', [5, 10, 20, 30, 60])
        
        # 确保数据包含价格列
        if price_col not in data.columns:
            return score
            
        # 选择短期、中期和长期MA进行评分
        periods = sorted(periods)
        if len(periods) >= 3:
            short_period = periods[0]           # 短期
            mid_period = periods[len(periods)//2]  # 中期
            long_period = periods[-1]           # 长期
        elif len(periods) == 2:
            short_period = periods[0]
            mid_period = periods[1]
            long_period = periods[1]
        else:
            short_period = mid_period = long_period = periods[0]
            
        # 获取对应的MA列
        short_ma_col = f'MA{short_period}'
        mid_ma_col = f'MA{mid_period}'
        long_ma_col = f'MA{long_period}'
        
        if short_ma_col not in result.columns or mid_ma_col not in result.columns or long_ma_col not in result.columns:
            return score
            
        # 获取价格与均线的关系
        price = data[price_col]
        short_ma = result[short_ma_col]
        mid_ma = result[mid_ma_col]
        long_ma = result[long_ma_col]
        
        # 价格相对于短期MA的位置
        price_vs_short = price / short_ma - 1
        # 价格相对于中期MA的位置
        price_vs_mid = price / mid_ma - 1
        # 价格相对于长期MA的位置
        price_vs_long = price / long_ma - 1
        
        # 短期MA相对于中期MA的位置
        short_vs_mid = short_ma / mid_ma - 1
        # 短期MA相对于长期MA的位置
        short_vs_long = short_ma / long_ma - 1
        # 中期MA相对于长期MA的位置
        mid_vs_long = mid_ma / long_ma - 1
        
        # 1. 价格位置评分：价格在均线上方加分，下方减分
        score += 10 * np.sign(price_vs_short)
        score += 8 * np.sign(price_vs_mid)
        score += 5 * np.sign(price_vs_long)
        
        # 2. 均线排列评分：多头排列加分，空头排列减分
        # 多头排列：短期 > 中期 > 长期
        bullish_alignment = (short_vs_mid > 0) & (mid_vs_long > 0) & (short_vs_long > 0)
        score[bullish_alignment] += 15
        
        # 空头排列：短期 < 中期 < 长期
        bearish_alignment = (short_vs_mid < 0) & (mid_vs_long < 0) & (short_vs_long < 0)
        score[bearish_alignment] -= 15
        
        # 3. 均线趋势评分
        short_trend = short_ma.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        mid_trend = mid_ma.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        long_trend = long_ma.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # 所有均线向上
        all_up = (short_trend > 0) & (mid_trend > 0) & (long_trend > 0)
        score[all_up] += 10
        
        # 所有均线向下
        all_down = (short_trend < 0) & (mid_trend < 0) & (long_trend < 0)
        score[all_down] -= 10
        
        # 4. 价格突破评分
        # 价格突破长期MA
        breakout_up = (price_vs_long.shift(1) <= 0) & (price_vs_long > 0)
        score[breakout_up] += 15
        
        # 价格跌破长期MA
        breakout_down = (price_vs_long.shift(1) >= 0) & (price_vs_long < 0)
        score[breakout_down] -= 15
        
        # 5. 盘整突破评分
        # 检测盘整
        consolidation = self.is_consolidation(period=long_period)
        
        # 盘整后向上突破
        consolidation_up = consolidation.shift(1) & (~consolidation) & (short_trend > 0)
        score[consolidation_up] += 10
        
        # 盘整后向下突破
        consolidation_down = consolidation.shift(1) & (~consolidation) & (short_trend < 0)
        score[consolidation_down] -= 10
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score 