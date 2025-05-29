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
        生成MACD信号
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            pd.DataFrame: 包含信号的DataFrame
        """
        # 计算MACD
        if not self.has_result():
            self.compute(data)
            
        if not self.has_result():
            return pd.DataFrame()
        
        # 获取MACD计算结果
        macd_data = self._result
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 添加基本信号
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['trend'] = 0  # -1表示下降趋势，0表示中性，1表示上升趋势
        
        # 根据MACD信号判断买入信号
        # 1. 金叉 + DIF在零轴以下 + 未出现背离
        basic_buy = (
            macd_data['golden_cross'] & 
            (macd_data['DIF'] < 0) & 
            ~macd_data['bearish_divergence']
        )
        
        # 2. 正背离确认（价格创新低但MACD未创新低）+ DIF上穿DEA
        divergence_buy = (
            macd_data['bullish_divergence'] & 
            macd_data['golden_cross']
        )
        
        # 3. 隐藏正背离确认
        hidden_buy = macd_data['hidden_bullish_divergence']
        
        # 4. 双MACD确认信号（如果启用）
        dual_buy = pd.Series(False, index=data.index)
        if self._parameters.get('use_secondary_macd', False) and 'dual_macd_agree_bullish' in macd_data.columns:
            dual_buy = macd_data['dual_macd_agree_bullish']
        
        # 5. 柱状图颜色变化确认（连续两个柱状向上）
        color_buy = False
        if 'MACD_up_confirm' in macd_data.columns:
            color_buy = macd_data['MACD_up_confirm'] & macd_data['MACD_turn_positive']
        
        # 组合买入信号
        signals['buy_signal'] = basic_buy | divergence_buy | hidden_buy | dual_buy | color_buy
        
        # 根据MACD信号判断卖出信号
        # 1. 死叉 + DIF在零轴以上 + 未出现正背离
        basic_sell = (
            macd_data['death_cross'] & 
            (macd_data['DIF'] > 0) & 
            ~macd_data['bullish_divergence']
        )
        
        # 2. 负背离确认（价格创新高但MACD未创新高）+ DIF下穿DEA
        divergence_sell = (
            macd_data['bearish_divergence'] & 
            macd_data['death_cross']
        )
        
        # 3. 隐藏负背离确认
        hidden_sell = macd_data['hidden_bearish_divergence']
        
        # 4. 双MACD确认信号（如果启用）
        dual_sell = pd.Series(False, index=data.index)
        if self._parameters.get('use_secondary_macd', False) and 'dual_macd_agree_bearish' in macd_data.columns:
            dual_sell = macd_data['dual_macd_agree_bearish']
        
        # 5. 柱状图颜色变化确认（连续两个柱状向下）
        color_sell = False
        if 'MACD_down_confirm' in macd_data.columns:
            color_sell = macd_data['MACD_down_confirm'] & macd_data['MACD_turn_negative']
        
        # 组合卖出信号
        signals['sell_signal'] = basic_sell | divergence_sell | hidden_sell | dual_sell | color_sell
        
        # 判断趋势
        signals['trend'] = 0  # 默认中性
        signals.loc[macd_data['DIF'] > macd_data['DEA'], 'trend'] = 1  # 上升趋势
        signals.loc[macd_data['DIF'] < macd_data['DEA'], 'trend'] = -1  # 下降趋势
        
        # 双MACD确认趋势（如果启用）
        if self._parameters.get('use_secondary_macd', False):
            if 'dual_macd_both_positive' in macd_data.columns:
                signals.loc[macd_data['dual_macd_both_positive'], 'trend'] = 1  # 强烈上升趋势
            if 'dual_macd_both_negative' in macd_data.columns:
                signals.loc[macd_data['dual_macd_both_negative'], 'trend'] = -1  # 强烈下降趋势
        
        # 信号强度评分（0-100）
        signals['score'] = 50  # 默认中性
        
        # 上升趋势加分
        signals.loc[signals['trend'] > 0, 'score'] += 10
        
        # 背离加分/减分
        signals.loc[macd_data['bullish_divergence'], 'score'] += 15
        signals.loc[macd_data['bearish_divergence'], 'score'] -= 15
        
        # 买入/卖出信号加分/减分
        signals.loc[signals['buy_signal'], 'score'] += 25
        signals.loc[signals['sell_signal'], 'score'] -= 25
        
        # 确保分数在0-100范围内
        signals['score'] = signals['score'].clip(0, 100)
        
        return signals
    
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