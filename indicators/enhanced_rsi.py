"""
增强版RSI指标模块

扩展标准RSI指标功能，增加多周期协同判断、中枢识别和背离分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple, Literal

from indicators.base_indicator import BaseIndicator
from indicators.common import rsi as calc_rsi
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedRSI(BaseIndicator):
    """
    增强版RSI指标
    
    在标准RSI基础上增加多周期协同判断、中枢识别和背离分析
    """
    
    # RSI超买超卖阈值
    OVERBOUGHT_THRESHOLD = 70
    OVERSOLD_THRESHOLD = 30
    
    # RSI中枢区域
    CENTER_LOW = 40
    CENTER_HIGH = 60
    
    def __init__(self, name: str = "EnhancedRSI", description: str = "增强版RSI指标",
                 periods: List[int] = None, price_col: str = 'close'):
        """
        初始化增强版RSI指标
        
        Args:
            name: 指标名称
            description: 指标描述
            periods: RSI周期列表，默认为[6, 14, 21]
            price_col: 价格列名
        """
        super().__init__(name, description)
        
        # 设置参数
        self._parameters = {
            'periods': periods or [6, 14, 21],
            'price_col': price_col,
            'overbought_threshold': self.OVERBOUGHT_THRESHOLD,
            'oversold_threshold': self.OVERSOLD_THRESHOLD,
            'center_low': self.CENTER_LOW,
            'center_high': self.CENTER_HIGH,
            'smooth_period': 3,
            'divergence_window': 20,
            'divergence_threshold': 0.1,
            'use_multi_period': True
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
        计算增强版RSI指标
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 包含增强版RSI指标的DataFrame
        """
        # 处理输入参数
        params = self._parameters.copy()
        params.update(kwargs)
        
        price_col = params.get('price_col', 'close')
        periods = params.get('periods', [6, 14, 21])
        smooth_period = params.get('smooth_period', 3)
        
        # 确保数据包含价格列
        self.ensure_columns(data, [price_col])
        
        # 如果periods是单个值，转换为列表
        if not isinstance(periods, list):
            periods = [periods]
        
        # 复制输入数据
        result = pd.DataFrame(index=data.index)
        
        # 计算各周期的RSI
        for period in periods:
            # 计算单一周期的RSI
            rsi_values = calc_rsi(data[price_col].values, period)
            
            # 确保前N个值为NaN，其中N = period
            rsi_values[:period] = np.nan
            
            # 添加RSI结果列
            result[f'RSI{period}'] = rsi_values
            
            # 计算平滑后的RSI
            if smooth_period > 0:
                result[f'RSI{period}_smooth'] = pd.Series(rsi_values).rolling(window=smooth_period).mean().values
        
        # 如果使用多周期协同
        if params.get('use_multi_period', True) and len(periods) >= 2:
            # 计算主周期RSI（取中间值）
            main_period_idx = len(periods) // 2
            main_period = periods[main_period_idx]
            result['RSI_main'] = result[f'RSI{main_period}']
            
            # 计算周期间的协同指标
            result = self._calculate_multi_period_indicators(result, periods)
        
        # 计算RSI的状态和区域
        result = self._calculate_rsi_states(result, params)
        
        # 检测背离
        result = self._detect_divergence(result, data, params)
        
        return result
    
    def _calculate_multi_period_indicators(self, result: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        计算多周期RSI协同指标
        
        Args:
            result: RSI计算结果
            periods: 周期列表
            
        Returns:
            pd.DataFrame: 添加了多周期协同指标的结果
        """
        if len(periods) < 2:
            return result
        
        # 获取短期、中期和长期RSI（如果有）
        short_period = periods[0]
        mid_period = periods[len(periods) // 2] if len(periods) > 2 else periods[1]
        long_period = periods[-1]
        
        # 添加短中长期RSI列
        result['RSI_short'] = result[f'RSI{short_period}']
        result['RSI_mid'] = result[f'RSI{mid_period}']
        result['RSI_long'] = result[f'RSI{long_period}']
        
        # 计算RSI动量（短期RSI与长期RSI的差值）
        result['RSI_momentum'] = result['RSI_short'] - result['RSI_long']
        
        # 计算RSI趋势一致性（短中长期RSI是否同向变化）
        result['RSI_short_change'] = result['RSI_short'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        result['RSI_mid_change'] = result['RSI_mid'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        result['RSI_long_change'] = result['RSI_long'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # 一致性指标：1表示一致上升，-1表示一致下降，0表示不一致
        result['RSI_consensus'] = 0
        result.loc[(result['RSI_short_change'] > 0) & (result['RSI_mid_change'] > 0) & (result['RSI_long_change'] > 0), 'RSI_consensus'] = 1
        result.loc[(result['RSI_short_change'] < 0) & (result['RSI_mid_change'] < 0) & (result['RSI_long_change'] < 0), 'RSI_consensus'] = -1
        
        # 计算多周期RSI强度（加权平均）
        weights = np.array([0.5, 0.3, 0.2])  # 短中长期权重
        result['RSI_strength'] = 0
        
        for i, period in enumerate(periods[:3]):  # 最多使用前3个周期
            if i < len(weights):
                result['RSI_strength'] += weights[i] * result[f'RSI{period}']
        
        return result
    
    def _calculate_rsi_states(self, result: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        计算RSI状态
        
        Args:
            result: RSI计算结果
            params: 参数字典
            
        Returns:
            pd.DataFrame: 添加了RSI状态的结果
        """
        # 获取主要RSI列
        rsi_cols = [col for col in result.columns if col.startswith('RSI') and not col.endswith('_smooth')]
        
        # 获取超买超卖阈值
        overbought = params.get('overbought_threshold', self.OVERBOUGHT_THRESHOLD)
        oversold = params.get('oversold_threshold', self.OVERSOLD_THRESHOLD)
        center_low = params.get('center_low', self.CENTER_LOW)
        center_high = params.get('center_high', self.CENTER_HIGH)
        
        # 对每个RSI列计算状态
        for col in rsi_cols:
            # 计算超买超卖状态
            result[f'{col}_overbought'] = result[col] > overbought
            result[f'{col}_oversold'] = result[col] < oversold
            
            # 计算RSI所在区域：1=超买区，-1=超卖区，0=中枢区
            result[f'{col}_zone'] = 0
            result.loc[result[col] > center_high, f'{col}_zone'] = 1
            result.loc[result[col] < center_low, f'{col}_zone'] = -1
            
            # 计算RSI突破信号
            result[f'{col}_cross_up_{oversold}'] = self.crossover(result[col], pd.Series(oversold, index=result.index))
            result[f'{col}_cross_down_{overbought}'] = self.crossunder(result[col], pd.Series(overbought, index=result.index))
            
            # 计算RSI突破中枢信号
            result[f'{col}_cross_up_center'] = self.crossover(result[col], pd.Series(center_high, index=result.index))
            result[f'{col}_cross_down_center'] = self.crossunder(result[col], pd.Series(center_low, index=result.index))
        
        # 如果有多个RSI周期，计算它们的共同状态
        if len(rsi_cols) > 1:
            # 计算所有RSI是否同时超买
            result['all_overbought'] = True
            for col in rsi_cols:
                result['all_overbought'] &= result[f'{col}_overbought']
            
            # 计算所有RSI是否同时超卖
            result['all_oversold'] = True
            for col in rsi_cols:
                result['all_oversold'] &= result[f'{col}_oversold']
        
        return result
    
    def _detect_divergence(self, result: pd.DataFrame, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        检测RSI与价格的背离
        
        Args:
            result: RSI计算结果
            data: 输入数据
            params: 参数字典
            
        Returns:
            pd.DataFrame: 添加了背离指标的结果
        """
        # 检查必要的列是否存在
        price_col = params.get('price_col', 'close')
        divergence_window = params.get('divergence_window', 20)
        divergence_threshold = params.get('divergence_threshold', 0.1)
        
        if price_col not in data.columns:
            return result
        
        # 选择主要RSI指标
        if 'RSI_main' in result.columns:
            rsi_col = 'RSI_main'
        else:
            # 选择第一个RSI列
            rsi_cols = [col for col in result.columns if col.startswith('RSI') and not col.endswith('_smooth')]
            if not rsi_cols:
                return result
            rsi_col = rsi_cols[0]
        
        # 初始化背离列
        result['bullish_divergence'] = False
        result['bearish_divergence'] = False
        
        # 至少需要divergence_window个数据点
        if len(data) < divergence_window:
            return result
        
        # 遍历时间序列
        for i in range(divergence_window, len(data)):
            # 检查价格是否创新低而RSI未创新低 -> 看涨背离
            price_window = data[price_col].iloc[i-divergence_window:i+1]
            rsi_window = result[rsi_col].iloc[i-divergence_window:i+1]
            
            if not pd.isna(price_window.iloc[-1]) and not pd.isna(rsi_window.iloc[-1]):
                # 获取最近的最低点
                min_price_idx = price_window.idxmin()
                min_rsi_idx = rsi_window.idxmin()
                
                # 如果最低价是最新的，但RSI不是
                if min_price_idx == price_window.index[-1] and min_rsi_idx != rsi_window.index[-1]:
                    # 计算RSI低点之间的差异
                    rsi_current = rsi_window.iloc[-1]
                    rsi_prev_min = rsi_window.min()
                    
                    # 如果当前RSI高于先前的最低点一定比例，确认看涨背离
                    if rsi_current > rsi_prev_min * (1 + divergence_threshold):
                        result.loc[price_window.index[-1], 'bullish_divergence'] = True
                
                # 获取最近的最高点
                max_price_idx = price_window.idxmax()
                max_rsi_idx = rsi_window.idxmax()
                
                # 如果最高价是最新的，但RSI不是
                if max_price_idx == price_window.index[-1] and max_rsi_idx != rsi_window.index[-1]:
                    # 计算RSI高点之间的差异
                    rsi_current = rsi_window.iloc[-1]
                    rsi_prev_max = rsi_window.max()
                    
                    # 如果当前RSI低于先前的最高点一定比例，确认看跌背离
                    if rsi_current < rsi_prev_max * (1 - divergence_threshold):
                        result.loc[price_window.index[-1], 'bearish_divergence'] = True
        
        return result

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强版RSI指标的原始评分
        
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
        
        # 获取RSI参数
        params = self.parameters
        overbought = params.get('overbought_threshold', self.OVERBOUGHT_THRESHOLD)
        oversold = params.get('oversold_threshold', self.OVERSOLD_THRESHOLD)
        
        # 确定使用哪个RSI列进行评分
        if 'RSI_main' in result.columns:
            rsi_col = 'RSI_main'
        else:
            # 选择周期最接近14的RSI列（标准RSI周期）
            rsi_cols = [col for col in result.columns if col.startswith('RSI') and not '_' in col]
            if not rsi_cols:
                return score
                
            # 找到最接近14的周期
            periods = [int(col.replace('RSI', '')) for col in rsi_cols]
            closest_period = min(periods, key=lambda x: abs(x - 14))
            rsi_col = f'RSI{closest_period}'
        
        # 获取RSI值
        rsi_values = result[rsi_col]
        
        # 基于RSI值计算评分
        # RSI接近30以下，看涨评分
        score = 50 + (50 - rsi_values)  # RSI=0 -> 评分=100, RSI=50 -> 评分=50, RSI=100 -> 评分=0
        
        # 额外的信号加成
        
        # 背离加分
        if 'bullish_divergence' in result.columns:
            score[result['bullish_divergence']] += 15
        
        if 'bearish_divergence' in result.columns:
            score[result['bearish_divergence']] -= 15
        
        # 多周期协同加分
        if 'RSI_consensus' in result.columns:
            score[result['RSI_consensus'] == 1] += 10
            score[result['RSI_consensus'] == -1] -= 10
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成RSI信号
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            pd.DataFrame: 包含信号的DataFrame
        """
        # 计算RSI
        if not self.has_result():
            self.compute(data)
            
        if not self.has_result():
            return pd.DataFrame()
        
        # 获取RSI计算结果
        rsi_data = self._result
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 添加基本信号
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['trend'] = 0  # -1表示下降趋势，0表示中性，1表示上升趋势
        
        # 主要RSI列
        if 'RSI_main' in rsi_data.columns:
            main_rsi_col = 'RSI_main'
        else:
            # 使用第一个周期的RSI
            period = self._parameters['periods'][0]
            main_rsi_col = f'RSI{period}'
        
        # 超买超卖阈值
        oversold = self._parameters.get('oversold_threshold', self.OVERSOLD_THRESHOLD)
        overbought = self._parameters.get('overbought_threshold', self.OVERBOUGHT_THRESHOLD)
        
        # 1. 超卖反弹信号
        oversold_rebound = rsi_data[f'{main_rsi_col}_cross_up_{oversold}']
        
        # 2. 超买回落信号
        overbought_fallback = rsi_data[f'{main_rsi_col}_cross_down_{overbought}']
        
        # 3. 背离信号
        bullish_divergence = rsi_data['bullish_divergence']
        bearish_divergence = rsi_data['bearish_divergence']
        
        # 4. 多周期协同信号（如果可用）
        multi_period_buy = pd.Series(False, index=data.index)
        multi_period_sell = pd.Series(False, index=data.index)
        
        if 'RSI_consensus' in rsi_data.columns:
            # 买入：多周期RSI一致上升 + 处于超卖区域恢复
            multi_period_buy = (rsi_data['RSI_consensus'] > 0) & (rsi_data[f'{main_rsi_col}_zone'] == -1)
            
            # 卖出：多周期RSI一致下降 + 处于超买区域回落
            multi_period_sell = (rsi_data['RSI_consensus'] < 0) & (rsi_data[f'{main_rsi_col}_zone'] == 1)
        
        # 组合买入信号
        signals['buy_signal'] = (
            oversold_rebound | 
            bullish_divergence | 
            multi_period_buy
        )
        
        # 组合卖出信号
        signals['sell_signal'] = (
            overbought_fallback | 
            bearish_divergence | 
            multi_period_sell
        )
        
        # 判断趋势
        signals['trend'] = 0  # 默认中性
        
        # 使用RSI区域判断趋势
        signals.loc[rsi_data[f'{main_rsi_col}_zone'] > 0, 'trend'] = 1  # 上升趋势
        signals.loc[rsi_data[f'{main_rsi_col}_zone'] < 0, 'trend'] = -1  # 下降趋势
        
        # 如果有多周期一致性指标，使用它来加强趋势判断
        if 'RSI_consensus' in rsi_data.columns:
            signals.loc[rsi_data['RSI_consensus'] > 0, 'trend'] = 1  # 强化上升趋势
            signals.loc[rsi_data['RSI_consensus'] < 0, 'trend'] = -1  # 强化下降趋势
        
        # 信号强度评分（0-100）
        signals['score'] = 50  # 默认中性
        
        # 趋势加分/减分
        signals.loc[signals['trend'] > 0, 'score'] += 10
        signals.loc[signals['trend'] < 0, 'score'] -= 10
        
        # 背离加分/减分
        signals.loc[rsi_data['bullish_divergence'], 'score'] += 15
        signals.loc[rsi_data['bearish_divergence'], 'score'] -= 15
        
        # RSI位置加分/减分
        signals.loc[rsi_data[f'{main_rsi_col}'] < oversold, 'score'] += 20
        signals.loc[rsi_data[f'{main_rsi_col}'] > overbought, 'score'] -= 20
        
        # 买入/卖出信号加分/减分
        signals.loc[signals['buy_signal'], 'score'] += 15
        signals.loc[signals['sell_signal'], 'score'] -= 15
        
        # 确保分数在0-100范围内
        signals['score'] = signals['score'].clip(0, 100)
        
        return signals
    
    def plot_rsi(self, data: pd.DataFrame, ax=None, show_divergence=True):
        """
        绘制RSI图表
        
        Args:
            data: 输入数据
            ax: matplotlib轴对象
            show_divergence: 是否显示背离标记
            
        Returns:
            matplotlib轴对象
        """
        try:
            import matplotlib.pyplot as plt
            
            # 计算RSI
            if not self.has_result():
                self.compute(data)
                
            if not self.has_result():
                logger.error("没有RSI计算结果可供绘图")
                return None
            
            # 创建图表
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            
            # 获取所有RSI列
            rsi_cols = [col for col in self._result.columns if col.startswith('RSI') and 
                       not col.endswith(('_overbought', '_oversold', '_zone', '_cross_up', '_cross_down', '_change'))]
            
            # 绘制各周期RSI线
            for col in rsi_cols:
                if col.startswith('RSI_') and col != 'RSI_main':
                    continue  # 跳过辅助列
                ax.plot(self._result.index, self._result[col], label=col)
            
            # 绘制超买超卖阈值
            overbought = self._parameters.get('overbought_threshold', self.OVERBOUGHT_THRESHOLD)
            oversold = self._parameters.get('oversold_threshold', self.OVERSOLD_THRESHOLD)
            ax.axhline(y=overbought, color='r', linestyle='--', alpha=0.5, label=f'超买({overbought})')
            ax.axhline(y=oversold, color='g', linestyle='--', alpha=0.5, label=f'超卖({oversold})')
            
            # 绘制中枢区域
            center_low = self._parameters.get('center_low', self.CENTER_LOW)
            center_high = self._parameters.get('center_high', self.CENTER_HIGH)
            ax.axhspan(center_low, center_high, alpha=0.1, color='gray', label='中枢区域')
            
            # 标记背离
            if show_divergence:
                if 'bullish_divergence' in self._result.columns:
                    bull_div_idx = self._result[self._result['bullish_divergence']].index
                    if len(bull_div_idx) > 0 and 'RSI_main' in self._result.columns:
                        ax.scatter(bull_div_idx, self._result.loc[bull_div_idx, 'RSI_main'], 
                                marker='*', color='lime', s=100, label='看涨背离')
                    elif len(bull_div_idx) > 0:
                        # 使用第一个周期的RSI
                        period = self._parameters['periods'][0]
                        ax.scatter(bull_div_idx, self._result.loc[bull_div_idx, f'RSI{period}'], 
                                marker='*', color='lime', s=100, label='看涨背离')
                
                if 'bearish_divergence' in self._result.columns:
                    bear_div_idx = self._result[self._result['bearish_divergence']].index
                    if len(bear_div_idx) > 0 and 'RSI_main' in self._result.columns:
                        ax.scatter(bear_div_idx, self._result.loc[bear_div_idx, 'RSI_main'], 
                                marker='*', color='orangered', s=100, label='看跌背离')
                    elif len(bear_div_idx) > 0:
                        # 使用第一个周期的RSI
                        period = self._parameters['periods'][0]
                        ax.scatter(bear_div_idx, self._result.loc[bear_div_idx, f'RSI{period}'], 
                                marker='*', color='orangered', s=100, label='看跌背离')
            
            # 添加图例和标题
            ax.legend(loc='best')
            ax.set_title('Enhanced RSI')
            ax.set_ylabel('RSI')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            
            return ax
            
        except ImportError:
            logger.warning("缺少matplotlib库，无法绘图")
            return None 