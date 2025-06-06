"""
增强版MACD指标模块

对标准MACD指标进行增强，提供更精确的背离检测和双MACD交叉验证功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple, Literal

from indicators.base_indicator import BaseIndicator
from indicators.common import macd as calc_macd
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedMACD(BaseIndicator):
    """
    增强版MACD指标
    
    在标准MACD基础上增强了背离检测、双MACD交叉验证等功能
    """
    
    def __init__(self, name: str = "EnhancedMACD", description: str = "增强版MACD指标",
                 fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 secondary_fast: int = 5, secondary_slow: int = 35, secondary_signal: int = 5):
        """
        初始化增强版MACD指标
        
        Args:
            name: 指标名称
            description: 指标描述
            fast_period: 快线周期，默认为12
            slow_period: 慢线周期，默认为26
            signal_period: 信号线周期，默认为9
            secondary_fast: 第二组MACD的快线周期
            secondary_slow: 第二组MACD的慢线周期
            secondary_signal: 第二组MACD的信号线周期
        """
        super().__init__(name, description)
        
        # 设置主MACD参数
        self._parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'price_col': 'close',
            'use_secondary_macd': False,
            'secondary_fast': secondary_fast,
            'secondary_slow': secondary_slow,
            'secondary_signal': secondary_signal,
            'divergence_window': 20,
            'divergence_threshold': 0.01,
            'smoothing_period': 3,
            'histogram_color_change': True
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
        计算增强版MACD指标
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 包含增强版MACD指标的DataFrame
        """
        # 处理输入参数
        params = self._parameters.copy()
        params.update(kwargs)
        
        price_col = params.get('price_col', 'close')
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        use_secondary_macd = params.get('use_secondary_macd', False)
        secondary_fast = params.get('secondary_fast', 5)
        secondary_slow = params.get('secondary_slow', 35)
        secondary_signal = params.get('secondary_signal', 5)
        
        # 确保数据包含价格列
        self.ensure_columns(data, [price_col])
        
        # 复制输入数据
        result = pd.DataFrame(index=data.index)
        
        # 计算主MACD指标
        dif, dea, macd_hist = calc_macd(
            data[price_col].values,
            fast_period,
            slow_period,
            signal_period
        )
        
        # 确保前N个值为NaN，其中N = max(fast_period, slow_period) - 1
        min_periods = max(fast_period, slow_period) - 1
        dif[:min_periods] = np.nan
        dea[:min_periods + signal_period - 1] = np.nan
        macd_hist[:min_periods + signal_period - 1] = np.nan
        
        # 添加主MACD结果列
        result['DIF'] = dif
        result['DEA'] = dea
        result['MACD'] = macd_hist
        
        # 计算平滑后的MACD
        if params.get('smoothing_period', 0) > 0:
            smoothing_period = params['smoothing_period']
            result['DIF_smooth'] = pd.Series(dif).rolling(window=smoothing_period).mean().values
            result['DEA_smooth'] = pd.Series(dea).rolling(window=smoothing_period).mean().values
            result['MACD_smooth'] = pd.Series(macd_hist).rolling(window=smoothing_period).mean().values
        
        # 计算第二组MACD（如果启用）
        if use_secondary_macd:
            sec_dif, sec_dea, sec_macd_hist = calc_macd(
                data[price_col].values,
                secondary_fast,
                secondary_slow,
                secondary_signal
            )
            
            # 确保前N个值为NaN
            sec_min_periods = max(secondary_fast, secondary_slow) - 1
            sec_dif[:sec_min_periods] = np.nan
            sec_dea[:sec_min_periods + secondary_signal - 1] = np.nan
            sec_macd_hist[:sec_min_periods + secondary_signal - 1] = np.nan
            
            # 添加第二组MACD结果列
            result['DIF_2'] = sec_dif
            result['DEA_2'] = sec_dea
            result['MACD_2'] = sec_macd_hist
        
        # 添加背离检测和其他增强功能
        result = self._add_enhanced_features(result, data)
        
        return result
    
    def _add_enhanced_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加增强功能
        
        Args:
            result: MACD计算结果
            data: 原始数据
            
        Returns:
            pd.DataFrame: 添加了增强功能的结果
        """
        price_col = self._parameters.get('price_col', 'close')
        
        # 计算MACD柱状图变化率
        result['MACD_change'] = result['MACD'].diff()
        
        # 计算柱状图颜色变化信号（连续两个柱状同向变化）
        if self._parameters.get('histogram_color_change', True):
            # 记录MACD柱状图由负变正和由正变负的变化点
            result['MACD_positive'] = result['MACD'] > 0
            result['MACD_positive_change'] = result['MACD_positive'].diff()
            result['MACD_turn_positive'] = result['MACD_positive_change'] == 1
            result['MACD_turn_negative'] = result['MACD_positive_change'] == -1
            
            # 连续两个柱状向上或向下变化的确认
            result['MACD_up_confirm'] = (result['MACD_change'] > 0) & (result['MACD_change'].shift(1) > 0)
            result['MACD_down_confirm'] = (result['MACD_change'] < 0) & (result['MACD_change'].shift(1) < 0)
        
        # 计算零轴穿越信号
        result['DIF_cross_zero_up'] = (result['DIF'] > 0) & (result['DIF'].shift(1) <= 0)
        result['DIF_cross_zero_down'] = (result['DIF'] < 0) & (result['DIF'].shift(1) >= 0)
        
        # 计算DIF和DEA的交叉信号
        result['golden_cross'] = self.crossover(result['DIF'], result['DEA'])
        result['death_cross'] = self.crossunder(result['DIF'], result['DEA'])
        
        # 计算背离信号
        result = self._detect_divergence(result, data)
        
        # 计算双MACD协同信号（如果启用）
        if self._parameters.get('use_secondary_macd', False) and 'DIF_2' in result.columns:
            result['dual_macd_agree_bullish'] = (
                result['golden_cross'] & 
                self.crossover(result['DIF_2'], result['DEA_2'])
            )
            result['dual_macd_agree_bearish'] = (
                result['death_cross'] & 
                self.crossunder(result['DIF_2'], result['DEA_2'])
            )
            
            # 主次MACD同时在零轴以上/以下
            result['dual_macd_both_positive'] = (result['DIF'] > 0) & (result['DIF_2'] > 0)
            result['dual_macd_both_negative'] = (result['DIF'] < 0) & (result['DIF_2'] < 0)
        
        return result
    
    def _detect_divergence(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        检测背离信号
        
        Args:
            result: MACD计算结果
            data: 原始数据
            
        Returns:
            pd.DataFrame: 添加了背离信号的结果
        """
        price_col = self._parameters.get('price_col', 'close')
        window = self._parameters.get('divergence_window', 20)
        threshold = self._parameters.get('divergence_threshold', 0.01)
        
        # 初始化背离信号列
        result['bullish_divergence'] = False  # 价格创新低，但MACD未创新低（看涨）
        result['bearish_divergence'] = False  # 价格创新高，但MACD未创新高（看跌）
        result['hidden_bullish_divergence'] = False  # 价格未创新低，但MACD创新低（隐藏看涨）
        result['hidden_bearish_divergence'] = False  # 价格未创新高，但MACD创新高（隐藏看跌）
        result['divergence_strength'] = 0.0  # 背离强度
        
        # 计算局部最高点和最低点
        price = data[price_col].values
        dif = result['DIF'].values
        
        # 初始化高点和低点数组
        price_highs = np.zeros_like(price, dtype=bool)
        price_lows = np.zeros_like(price, dtype=bool)
        dif_highs = np.zeros_like(dif, dtype=bool)
        dif_lows = np.zeros_like(dif, dtype=bool)
        
        # 识别局部高低点
        for i in range(1, len(price) - 1):
            # 局部高点：比前后都高
            if price[i] > price[i-1] and price[i] > price[i+1]:
                price_highs[i] = True
            # 局部低点：比前后都低
            if price[i] < price[i-1] and price[i] < price[i+1]:
                price_lows[i] = True
                
            # DIF局部高点
            if dif[i] > dif[i-1] and dif[i] > dif[i+1]:
                dif_highs[i] = True
            # DIF局部低点
            if dif[i] < dif[i-1] and dif[i] < dif[i+1]:
                dif_lows[i] = True
        
        # 在滑动窗口中查找背离
        for i in range(window, len(price)):
            # 获取窗口数据
            window_start = max(0, i - window + 1)
            window_price = price[window_start:i+1]
            window_dif = dif[window_start:i+1]
            window_price_highs = price_highs[window_start:i+1]
            window_price_lows = price_lows[window_start:i+1]
            window_dif_highs = dif_highs[window_start:i+1]
            window_dif_lows = dif_lows[window_start:i+1]
            
            # 查找窗口内最近的两个价格高点
            price_high_indices = np.where(window_price_highs)[0]
            if len(price_high_indices) >= 2:
                last_high_idx = price_high_indices[-1]
                prev_high_idx = price_high_indices[-2]
                
                # 比较最近的两个价格高点和对应的DIF值
                if window_price[last_high_idx] > window_price[prev_high_idx]:
                    # 价格创新高
                    last_high_dif_idx = min(len(window_dif) - 1, last_high_idx)
                    prev_high_dif_idx = min(len(window_dif) - 1, prev_high_idx)
                    
                    if window_dif[last_high_dif_idx] < window_dif[prev_high_dif_idx]:
                        # 价格创新高但DIF未创新高 -> 看跌背离
                        result.loc[result.index[i], 'bearish_divergence'] = True
                        
                        # 计算背离强度
                        price_change = (window_price[last_high_idx] - window_price[prev_high_idx]) / window_price[prev_high_idx]
                        dif_change = (window_dif[last_high_dif_idx] - window_dif[prev_high_dif_idx]) / (abs(window_dif[prev_high_dif_idx]) + 1e-10)
                        
                        # 背离强度：价格变化与DIF变化的差异
                        strength = abs(price_change - dif_change)
                        result.loc[result.index[i], 'divergence_strength'] = strength
                elif window_price[last_high_idx] < window_price[prev_high_idx]:
                    # 价格未创新高
                    last_high_dif_idx = min(len(window_dif) - 1, last_high_idx)
                    prev_high_dif_idx = min(len(window_dif) - 1, prev_high_idx)
                    
                    if window_dif[last_high_dif_idx] > window_dif[prev_high_dif_idx]:
                        # 价格未创新高但DIF创新高 -> 隐藏看涨背离
                        result.loc[result.index[i], 'hidden_bullish_divergence'] = True
            
            # 查找窗口内最近的两个价格低点
            price_low_indices = np.where(window_price_lows)[0]
            if len(price_low_indices) >= 2:
                last_low_idx = price_low_indices[-1]
                prev_low_idx = price_low_indices[-2]
                
                # 比较最近的两个价格低点和对应的DIF值
                if window_price[last_low_idx] < window_price[prev_low_idx]:
                    # 价格创新低
                    last_low_dif_idx = min(len(window_dif) - 1, last_low_idx)
                    prev_low_dif_idx = min(len(window_dif) - 1, prev_low_idx)
                    
                    if window_dif[last_low_dif_idx] > window_dif[prev_low_dif_idx]:
                        # 价格创新低但DIF未创新低 -> 看涨背离
                        result.loc[result.index[i], 'bullish_divergence'] = True
                        
                        # 计算背离强度
                        price_change = (window_price[last_low_idx] - window_price[prev_low_idx]) / window_price[prev_low_idx]
                        dif_change = (window_dif[last_low_dif_idx] - window_dif[prev_low_dif_idx]) / (abs(window_dif[prev_low_dif_idx]) + 1e-10)
                        
                        # 背离强度：价格变化与DIF变化的差异
                        strength = abs(price_change - dif_change)
                        result.loc[result.index[i], 'divergence_strength'] = strength
                elif window_price[last_low_idx] > window_price[prev_low_idx]:
                    # 价格未创新低
                    last_low_dif_idx = min(len(window_dif) - 1, last_low_idx)
                    prev_low_dif_idx = min(len(window_dif) - 1, prev_low_idx)
                    
                    if window_dif[last_low_dif_idx] < window_dif[prev_low_dif_idx]:
                        # 价格未创新低但DIF创新低 -> 隐藏看跌背离
                        result.loc[result.index[i], 'hidden_bearish_divergence'] = True
        
        # 过滤掉弱背离信号
        for col in ['bullish_divergence', 'bearish_divergence', 'hidden_bullish_divergence', 'hidden_bearish_divergence']:
            mask = result[col] & (result['divergence_strength'] < threshold)
            result.loc[mask, col] = False
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含信号的DataFrame
        """
        if not self.has_result():
            self.calculate(data)
        
        # 复制输入数据
        signals = pd.DataFrame(index=data.index)
        signals['buy'] = False
        signals['sell'] = False
        signals['neutral'] = True
        
        # 检查结果是否有效
        if self._result is None or len(self._result) == 0:
            return signals
        
        # 从计算结果中获取各种信号
        result = self._result
        
        # 生成买入信号
        # 1. 金叉信号（主要买入信号）
        signals.loc[result['golden_cross'], 'buy'] = True
        
        # 2. DIF上穿零轴且为正值
        zero_cross_buy = result['DIF_cross_zero_up'] & (result['DEA'] > -0.5 * result['DEA'].rolling(window=10).std())
        signals.loc[zero_cross_buy, 'buy'] = True
        
        # 3. 双MACD共振买入信号
        if 'dual_macd_agree_bullish' in result.columns:
            signals.loc[result['dual_macd_agree_bullish'], 'buy'] = True
        
        # 4. 柱状图加速放大买入信号
        if 'MACD_expanding_strong' in result.columns:
            macd_expand_buy = result['MACD_expanding_strong'] & (result['DIF'] > result['DEA'])
            signals.loc[macd_expand_buy, 'buy'] = True
        
        # 5. 看涨背离买入信号
        if 'bullish_divergence' in result.columns:
            signals.loc[result['bullish_divergence'], 'buy'] = True
        
        # 生成卖出信号
        # 1. 死叉信号（主要卖出信号）
        signals.loc[result['death_cross'], 'sell'] = True
        
        # 2. DIF下穿零轴且为负值
        zero_cross_sell = result['DIF_cross_zero_down'] & (result['DEA'] < 0.5 * result['DEA'].rolling(window=10).std())
        signals.loc[zero_cross_sell, 'sell'] = True
        
        # 3. 双MACD共振卖出信号
        if 'dual_macd_agree_bearish' in result.columns:
            signals.loc[result['dual_macd_agree_bearish'], 'sell'] = True
        
        # 4. 柱状图加速缩小卖出信号
        if 'MACD_contracting_strong' in result.columns:
            macd_contract_sell = result['MACD_contracting_strong'] & (result['DIF'] < result['DEA'])
            signals.loc[macd_contract_sell, 'sell'] = True
        
        # 5. 看跌背离卖出信号
        if 'bearish_divergence' in result.columns:
            signals.loc[result['bearish_divergence'], 'sell'] = True
        
        # 设置中性信号（既不是买入也不是卖出）
        signals['neutral'] = ~(signals['buy'] | signals['sell'])
        
        return signals
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强版MACD指标的原始评分
        
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
        
        # 基于DIF和DEA的位置关系计算基础评分
        # DIF > DEA：看涨，DIF < DEA：看跌
        score += 20 * np.where(result['DIF'] > result['DEA'], 1, -1)
        
        # 调整基于MACD柱状图值和变化率
        # 柱状图为正值加分，为负值减分
        macd_factor = result['MACD'] / (result['MACD'].abs().rolling(window=20).max() + 1e-6)
        score += 10 * macd_factor
        
        # 基于MACD柱状图变化率的额外调整
        if 'MACD_change' in result.columns:
            # 柱状图增加速度加分/减分
            change_factor = result['MACD_change'] / (result['MACD_change'].abs().rolling(window=20).max() + 1e-6)
            score += 5 * change_factor
        
        # 特殊信号的得分调整
        
        # 金叉信号大幅加分
        if 'golden_cross' in result.columns:
            score[result['golden_cross']] += 20
            
        # 死叉信号大幅减分
        if 'death_cross' in result.columns:
            score[result['death_cross']] -= 20
        
        # DIF穿越零轴信号
        if 'DIF_cross_zero_up' in result.columns:
            score[result['DIF_cross_zero_up']] += 15
            
        if 'DIF_cross_zero_down' in result.columns:
            score[result['DIF_cross_zero_down']] -= 15
        
        # 双MACD共振信号（如果启用）
        if 'dual_macd_agree_bullish' in result.columns:
            score[result['dual_macd_agree_bullish']] += 15
            
        if 'dual_macd_agree_bearish' in result.columns:
            score[result['dual_macd_agree_bearish']] -= 15
        
        # 背离信号
        if 'bullish_divergence' in result.columns:
            score[result['bullish_divergence']] += 15
            
        if 'bearish_divergence' in result.columns:
            score[result['bearish_divergence']] -= 15
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def plot_macd(self, data: pd.DataFrame, ax=None, show_divergence=True):
        """
        绘制MACD图表
        
        Args:
            data: 输入数据
            ax: matplotlib轴对象
            show_divergence: 是否显示背离标记
            
        Returns:
            matplotlib轴对象
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            # 计算MACD
            if not self.has_result():
                self.compute(data)
                
            if not self.has_result():
                logger.error("没有MACD计算结果可供绘图")
                return None
            
            # 创建图表
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制DIF和DEA线
            ax.plot(self._result.index, self._result['DIF'], 'b-', label='DIF')
            ax.plot(self._result.index, self._result['DEA'], 'r-', label='DEA')
            
            # 绘制柱状图
            for i in range(len(self._result)):
                if i > 0:  # 跳过第一个点，因为无法计算变化
                    if self._result['MACD'].iloc[i] >= 0:
                        color = 'r'  # 正值为红色
                    else:
                        color = 'g'  # 负值为绿色
                    
                    ax.add_patch(Rectangle(
                        (i, 0),
                        0.6,
                        self._result['MACD'].iloc[i],
                        color=color,
                        alpha=0.5
                    ))
            
            # 绘制零轴
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 标记金叉和死叉
            if 'golden_cross' in self._result.columns:
                golden_cross_idx = self._result[self._result['golden_cross']].index
                death_cross_idx = self._result[self._result['death_cross']].index
                
                ax.scatter(golden_cross_idx, self._result.loc[golden_cross_idx, 'DIF'], 
                          marker='^', color='r', s=50, label='金叉')
                ax.scatter(death_cross_idx, self._result.loc[death_cross_idx, 'DIF'], 
                          marker='v', color='g', s=50, label='死叉')
            
            # 标记背离
            if show_divergence:
                if 'bullish_divergence' in self._result.columns:
                    bull_div_idx = self._result[self._result['bullish_divergence']].index
                    ax.scatter(bull_div_idx, self._result.loc[bull_div_idx, 'DIF'], 
                              marker='*', color='lime', s=100, label='看涨背离')
                
                if 'bearish_divergence' in self._result.columns:
                    bear_div_idx = self._result[self._result['bearish_divergence']].index
                    ax.scatter(bear_div_idx, self._result.loc[bear_div_idx, 'DIF'], 
                              marker='*', color='orangered', s=100, label='看跌背离')
            
            # 添加图例和标题
            ax.legend(loc='best')
            ax.set_title('Enhanced MACD')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            return ax
            
        except ImportError:
            logger.warning("缺少matplotlib库，无法绘图")
            return None 