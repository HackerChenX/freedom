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
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.indicator_type = name.upper()
        
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
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取MACD指标的形态信号

        Args:
            data: 输入数据，通常是已经调用 _calculate 方法计算过的DataFrame
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含各种形态信号的DataFrame
        """
        # 定义所有可能的形态信号列
        pattern_cols = [
            'golden_cross', 'death_cross',
            'bullish_divergence', 'bearish_divergence',
            'hidden_bullish_divergence', 'hidden_bearish_divergence',
            'MACD_turn_positive', 'MACD_turn_negative',
            'DIF_cross_zero_up', 'DIF_cross_zero_down'
        ]
        
        # 如果启用了双MACD，添加相关信号
        if self._parameters.get('use_secondary_macd', False):
            pattern_cols.extend([
                'dual_macd_agree_bullish', 'dual_macd_agree_bearish',
                'dual_macd_both_positive', 'dual_macd_both_negative'
            ])

        # 筛选出data中存在的形态列
        existing_patterns = [col for col in pattern_cols if col in data.columns]
        
        if not existing_patterns:
            # 如果没有任何形态列，可能需要先运行计算
            # 但为避免复杂调用，这里返回空DataFrame
            logger.warning("未在输入数据中找到任何形态信号，请先运行 a.calculate(df)")
            return pd.DataFrame(index=data.index)
            
        return data[existing_patterns].copy()

    def register_patterns(self):
        """
        注册EnhancedMACD指标的形态到全局形态注册表
        """
        # 注册MACD金叉死叉形态
        self.register_pattern_to_registry(
            pattern_id="golden_cross",
            display_name="MACD金叉",
            description="DIF上穿DEA，表明上升动能增强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="death_cross",
            display_name="MACD死叉",
            description="DIF下穿DEA，表明下降动能增强",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册MACD背离形态
        self.register_pattern_to_registry(
            pattern_id="bullish_divergence",
            display_name="MACD看涨背离",
            description="价格创新低但MACD未创新低，表明下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="bearish_divergence",
            display_name="MACD看跌背离",
            description="价格创新高但MACD未创新高，表明上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册MACD隐藏背离形态
        self.register_pattern_to_registry(
            pattern_id="hidden_bullish_divergence",
            display_name="MACD隐藏看涨背离",
            description="价格未创新低但MACD创新低，表明上升趋势中的调整",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="hidden_bearish_divergence",
            display_name="MACD隐藏看跌背离",
            description="价格未创新高但MACD创新高，表明下降趋势中的反弹",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-18.0,
            polarity="NEGATIVE"
        )

        # 注册MACD零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="DIF_cross_zero_up",
            display_name="DIF上穿零轴",
            description="DIF上穿零轴，表明多头力量占优",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="DIF_cross_zero_down",
            display_name="DIF下穿零轴",
            description="DIF下穿零轴，表明空头力量占优",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册MACD柱状图变化形态
        self.register_pattern_to_registry(
            pattern_id="MACD_turn_positive",
            display_name="MACD柱状图转正",
            description="MACD柱状图由负转正，表明动能转强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MACD_turn_negative",
            display_name="MACD柱状图转负",
            description="MACD柱状图由正转负，表明动能转弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-12.0,
            polarity="NEGATIVE"
        )

        # 注册双MACD协同形态（如果启用）
        self.register_pattern_to_registry(
            pattern_id="dual_macd_agree_bullish",
            display_name="双MACD看涨共振",
            description="主次MACD同时发出看涨信号，信号更可靠",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=28.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="dual_macd_agree_bearish",
            display_name="双MACD看跌共振",
            description="主次MACD同时发出看跌信号，信号更可靠",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-28.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="dual_macd_both_positive",
            display_name="双MACD同时为正",
            description="主次MACD的DIF都在零轴以上，表明强势多头市场",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="dual_macd_both_negative",
            display_name="双MACD同时为负",
            description="主次MACD的DIF都在零轴以下，表明强势空头市场",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
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
        result = data.copy()
        
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
        生成买入和卖出信号
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含买卖信号的DataFrame
        """
        # 首先，计算所有指标和形态
        if 'DIF' not in data.columns:
            calc_data = self.calculate(data)
        else:
            calc_data = data

        # 获取所有形态
        patterns = self.get_patterns(calc_data)
        
        # 定义买入和卖出信号的构成
        buy_conditions = [
            'golden_cross',
            'bullish_divergence',
            'hidden_bullish_divergence',
            'DIF_cross_zero_up'
        ]
        
        sell_conditions = [
            'death_cross',
            'bearish_divergence',
            'hidden_bearish_divergence',
            'DIF_cross_zero_down'
        ]
        
        # 如果启用双MACD，添加协同信号
        if self._parameters.get('use_secondary_macd', False):
            buy_conditions.append('dual_macd_agree_bullish')
            sell_conditions.append('dual_macd_agree_bearish')

        # 初始化信号Series
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)

        # 聚合信号
        for col in buy_conditions:
            if col in patterns.columns:
                buy_signal |= patterns[col]
        
        for col in sell_conditions:
            if col in patterns.columns:
                sell_signal |= patterns[col]
        
        # 创建最终的信号DataFrame
        signals_df = pd.DataFrame({
            'buy_signal': buy_signal,
            'sell_signal': sell_signal
        }, index=data.index)

        return signals_df

    def get_signal_strength(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        计算信号强度
        
        Args:
            data: 输入数据
            
        Returns:
            Tuple[pd.Series, pd.Series]: 包含买入信号强度和卖出信号强度的元组
        """
        # 首先，计算所有指标和形态
        if 'DIF' not in data.columns:
            calc_data = self.calculate(data)
        else:
            calc_data = data

        # 获取所有形态
        patterns = self.get_patterns(calc_data)
        
        # 定义买入和卖出信号的构成
        buy_conditions = [
            'golden_cross',
            'bullish_divergence',
            'hidden_bullish_divergence',
            'DIF_cross_zero_up'
        ]
        
        sell_conditions = [
            'death_cross',
            'bearish_divergence',
            'hidden_bearish_divergence',
            'DIF_cross_zero_down'
        ]
        
        # 如果启用双MACD，添加协同信号
        if self._parameters.get('use_secondary_macd', False):
            buy_conditions.append('dual_macd_agree_bullish')
            sell_conditions.append('dual_macd_agree_bearish')

        # 初始化信号Series
        buy_signal = pd.Series(0, index=data.index)
        sell_signal = pd.Series(0, index=data.index)

        # 聚合信号
        for col in buy_conditions:
            if col in patterns.columns:
                buy_signal += patterns[col]
        
        for col in sell_conditions:
            if col in patterns.columns:
                sell_signal += patterns[col]
        
        # 计算信号强度
        buy_strength = buy_signal / len(buy_conditions)
        sell_strength = sell_signal / len(sell_conditions)

        return buy_strength, sell_strength
    
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
    
    def set_market_environment(self, market_env: str):
        self.market_env = market_env

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典，包含name, description, strength等
        """
        pattern_info_map = {
            'golden_cross': {
                'name': 'MACD金叉',
                'description': 'DIF线上穿DEA线，表示买入信号',
                'strength': 'strong',
                'type': 'bullish'
            },
            'death_cross': {
                'name': 'MACD死叉',
                'description': 'DIF线下穿DEA线，表示卖出信号',
                'strength': 'strong',
                'type': 'bearish'
            },
            'bullish_divergence': {
                'name': 'MACD牛市背离',
                'description': '价格创新低但MACD未创新低，表示看涨信号',
                'strength': 'very_strong',
                'type': 'bullish'
            },
            'bearish_divergence': {
                'name': 'MACD熊市背离',
                'description': '价格创新高但MACD未创新高，表示看跌信号',
                'strength': 'very_strong',
                'type': 'bearish'
            },
            'hidden_bullish_divergence': {
                'name': 'MACD隐藏牛市背离',
                'description': '价格未创新低但MACD创新低，表示隐藏看涨信号',
                'strength': 'strong',
                'type': 'bullish'
            },
            'hidden_bearish_divergence': {
                'name': 'MACD隐藏熊市背离',
                'description': '价格未创新高但MACD创新高，表示隐藏看跌信号',
                'strength': 'strong',
                'type': 'bearish'
            },
            'MACD_turn_positive': {
                'name': 'MACD转正',
                'description': 'MACD柱状图由负转正，表示上升动能',
                'strength': 'medium',
                'type': 'bullish'
            },
            'MACD_turn_negative': {
                'name': 'MACD转负',
                'description': 'MACD柱状图由正转负，表示下降动能',
                'strength': 'medium',
                'type': 'bearish'
            },
            'DIF_cross_zero_up': {
                'name': 'DIF零轴向上穿越',
                'description': 'DIF线从下方穿越零轴，表明由空头转为多头',
                'strength': 'medium',
                'type': 'bullish'
            },
            'DIF_cross_zero_down': {
                'name': 'DIF零轴向下穿越',
                'description': 'DIF线从上方穿越零轴，表明由多头转为空头',
                'strength': 'medium',
                'type': 'bearish'
            },
            'dual_macd_agree_bullish': {
                'name': '双MACD看涨共振',
                'description': '主次MACD同时出现金叉，表示强烈看涨信号',
                'strength': 'very_strong',
                'type': 'bullish'
            },
            'dual_macd_agree_bearish': {
                'name': '双MACD看跌共振',
                'description': '主次MACD同时出现死叉，表示强烈看跌信号',
                'strength': 'very_strong',
                'type': 'bearish'
            },
            'dual_macd_both_positive': {
                'name': '双MACD同时为正',
                'description': '主次MACD的DIF线同时在零轴以上，表示强势上涨',
                'strength': 'strong',
                'type': 'bullish'
            },
            'dual_macd_both_negative': {
                'name': '双MACD同时为负',
                'description': '主次MACD的DIF线同时在零轴以下，表示强势下跌',
                'strength': 'strong',
                'type': 'bearish'
            }
        }

        return pattern_info_map.get(pattern_id, {
            'name': pattern_id,
            'description': f'EnhancedMACD形态: {pattern_id}',
            'strength': 'medium',
            'type': 'neutral'
        })

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算EnhancedMACD指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if not patterns.empty:
            # 检查EnhancedMACD形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.2)

        # 3. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.1, 0.15)

        # 4. 基于评分趋势的置信度
        if len(score) >= 3:
            recent_scores = score.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))