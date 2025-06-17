"""
MACD指标分析模块

提供MACD指标的计算和分析功能
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.technical_utils import calculate_macd, crossover, crossunder
from indicators.base_indicator import BaseIndicator
from indicators.pattern_registry import PatternRegistry

logger = get_logger(__name__)

class MACD(BaseIndicator):
    """MACD指标"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 histogram_threshold: float = 0.0, divergence_window: int = 20, 
                 divergence_threshold: float = 0.05, zero_line_sensitivity: float = 0.001):
        """
        初始化MACD指标
        
        Args:
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            histogram_threshold: 柱状图阈值，用于过滤微小变化
            divergence_window: 背离检测窗口
            divergence_threshold: 背离检测阈值
            zero_line_sensitivity: 零轴敏感度，用于判断零轴附近的值
        """
        super().__init__()
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
        self.name = "MACD"
        
        # 设置MACD参数
        self._parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'price_col': 'close',
            'histogram_threshold': histogram_threshold,
            'divergence_window': divergence_window,
            'divergence_threshold': divergence_threshold,
            'zero_line_sensitivity': zero_line_sensitivity,
            'smoothing_enabled': False,
            'smoothing_period': 3
        }
        
        # 定义MACD形态
        self.patterns = {
            'macd_golden_cross': {
                'name': 'MACD金叉',
                'description': 'DIF从下向上穿越DEA',
                'analyzer': self._analyze_golden_cross
            },
            'macd_death_cross': {
                'name': 'MACD死叉',
                'description': 'DIF从上向下穿越DEA',
                'analyzer': self._analyze_death_cross
            },
            'macd_divergence': {
                'name': 'MACD背离',
                'description': '价格创新高/新低，但MACD未创新高/新低',
                'analyzer': self._analyze_divergence
            },
            'macd_double_bottom': {
                'name': 'MACD双底',
                'description': 'MACD形成双底形态，看涨信号',
                'analyzer': self._analyze_double_patterns
            },
            'macd_double_top': {
                'name': 'MACD双顶',
                'description': 'MACD形成双顶形态，看跌信号',
                'analyzer': self._analyze_double_patterns
            }
        }
        
        # 记录已注册的形态，防止重复注册
        self._registered_patterns = False
        
        # 设置形态注册表允许覆盖，避免警告
        PatternRegistry.set_allow_override(True)
        
        # 初始化基类（会自动调用register_patterns方法）
        super().__init__()
        
        # 重置形态注册表为不允许覆盖
        PatternRegistry.set_allow_override(False)
        
        self.is_available = True
    
    @property
    def fast_period(self) -> int:
        """获取快线周期参数"""
        return self._parameters['fast_period']
    
    @property
    def slow_period(self) -> int:
        """获取慢线周期参数"""
        return self._parameters['slow_period']
        
    @property
    def signal_period(self) -> int:
        """获取信号线周期参数"""
        return self._parameters['signal_period']
    
    def _register_macd_patterns(self):
        """
        注册MACD形态
        """
        # 注册MACD金叉形态
        self.register_pattern_to_registry(
            pattern_id="MACD_GOLDEN_CROSS",
            display_name="MACD金叉",
            description="MACD快线从下向上穿越慢线，看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )
        
        # 注册MACD死叉形态
        self.register_pattern_to_registry(
            pattern_id="MACD_DEATH_CROSS",
            display_name="MACD死叉",
            description="MACD快线从上向下穿越慢线，看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )
        
        # 注册MACD零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="MACD_ZERO_CROSS_ABOVE",
            display_name="MACD零轴向上穿越",
            description="MACD线从下方穿越零轴，表明由空头转为多头",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )
        
        self.register_pattern_to_registry(
            pattern_id="MACD_ZERO_CROSS_BELOW",
            display_name="MACD零轴向下穿越",
            description="MACD线从上方穿越零轴，表明由多头转为空头",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )
        
        # 注册MACD背离形态
        self.register_pattern_to_registry(
            pattern_id="MACD_BULLISH_DIVERGENCE",
            display_name="MACD底背离",
            description="价格创新低，但MACD未创新低，潜在看涨信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MACD_BEARISH_DIVERGENCE",
            display_name="MACD顶背离",
            description="价格创新高，但MACD未创新高，潜在看跌信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )
        
        # 注册MACD柱状图形态
        self.register_pattern_to_registry(
            pattern_id="MACD_HISTOGRAM_EXPANDING",
            display_name="MACD柱状图扩张",
            description="MACD柱状图连续增大，表明趋势加强",
            pattern_type="MOMENTUM",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MACD_HISTOGRAM_CONTRACTING",
            display_name="MACD柱状图收缩",
            description="MACD柱状图连续减小，表明趋势减弱",
            pattern_type="EXHAUSTION",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )
        
        # 注册MACD趋势形态
        self.register_pattern_to_registry(
            pattern_id="MACD_STRONG_BULLISH",
            display_name="MACD强势多头",
            description="MACD值处于高位且继续上升，表明强劲上涨趋势",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MACD_STRONG_BEARISH",
            display_name="MACD强势空头",
            description="MACD值处于低位且继续下降，表明强劲下跌趋势",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-18.0,
            polarity="NEGATIVE"
        )
        
        # 注册新增的MACD双顶双底形态
        self.register_pattern_to_registry(
            pattern_id="MACD_DOUBLE_BOTTOM",
            display_name="MACD双底",
            description="MACD形成双底形态，看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=22.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MACD_DOUBLE_TOP",
            display_name="MACD双顶",
            description="MACD形成双顶形态，看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-22.0,
            polarity="NEGATIVE"
        )
        
        # 注册新增的MACD形态
        self.register_pattern_to_registry(
            pattern_id="MACD_TRIPLE_CROSS",
            display_name="MACD三重穿越",
            description="MACD短期内多次穿越信号线，表明市场不稳定",
            pattern_type="VOLATILITY",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="MACD_ZERO_LINE_HESITATION",
            display_name="MACD零轴徘徊",
            description="MACD在零轴附近徘徊，表明市场处于犹豫状态",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )
    
    def parameters(self) -> Dict[str, Any]:
        """获取参数"""
        return self._parameters.copy()

    def set_parameters(self, **kwargs):
        """设置指标参数"""
        for key, value in kwargs.items():
            if key in self._parameters:
                self._parameters[key] = value

    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data: 输入数据，必须包含 'close' 列
            
        Returns:
            pd.DataFrame: 包含MACD线、信号线和柱状图的DataFrame
        """
        price_col = self._parameters['price_col']
        
        # 计算MACD
        macd_line, macd_signal, macd_histogram = calculate_macd(
            data[price_col],
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period
        )

        # 确保返回的是Series而不是数组，并处理可能的多维数组
        if not isinstance(macd_line, pd.Series):
            # 处理多维数组的情况
            if hasattr(macd_line, 'shape') and len(macd_line.shape) > 1:
                # 如果是多维数组，转换为numpy数组并取第一列
                macd_line = np.array(macd_line)
                if macd_line.shape[1] > 0:
                    macd_line = macd_line[:, 0]
                else:
                    macd_line = macd_line.flatten()
            elif hasattr(macd_line, 'flatten'):
                macd_line = macd_line.flatten()
            # 确保长度匹配
            if len(macd_line) != len(data.index):
                # 如果长度不匹配，用NaN填充或截断
                if len(macd_line) < len(data.index):
                    macd_line = np.concatenate([np.full(len(data.index) - len(macd_line), np.nan), macd_line])
                else:
                    macd_line = macd_line[:len(data.index)]
            macd_line = pd.Series(macd_line, index=data.index)

        if not isinstance(macd_signal, pd.Series):
            # 处理多维数组的情况
            if hasattr(macd_signal, 'shape') and len(macd_signal.shape) > 1:
                # 如果是多维数组，转换为numpy数组并取第一列
                macd_signal = np.array(macd_signal)
                if macd_signal.shape[1] > 0:
                    macd_signal = macd_signal[:, 0]
                else:
                    macd_signal = macd_signal.flatten()
            elif hasattr(macd_signal, 'flatten'):
                macd_signal = macd_signal.flatten()
            # 确保长度匹配
            if len(macd_signal) != len(data.index):
                # 如果长度不匹配，用NaN填充或截断
                if len(macd_signal) < len(data.index):
                    macd_signal = np.concatenate([np.full(len(data.index) - len(macd_signal), np.nan), macd_signal])
                else:
                    macd_signal = macd_signal[:len(data.index)]
            macd_signal = pd.Series(macd_signal, index=data.index)

        if not isinstance(macd_histogram, pd.Series):
            # 处理多维数组的情况
            if hasattr(macd_histogram, 'shape') and len(macd_histogram.shape) > 1:
                # 如果是多维数组，转换为numpy数组并取第一列
                macd_histogram = np.array(macd_histogram)
                if macd_histogram.shape[1] > 0:
                    macd_histogram = macd_histogram[:, 0]
                else:
                    macd_histogram = macd_histogram.flatten()
            elif hasattr(macd_histogram, 'flatten'):
                macd_histogram = macd_histogram.flatten()
            # 确保长度匹配
            if len(macd_histogram) != len(data.index):
                # 如果长度不匹配，用NaN填充或截断
                if len(macd_histogram) < len(data.index):
                    macd_histogram = np.concatenate([np.full(len(data.index) - len(macd_histogram), np.nan), macd_histogram])
                else:
                    macd_histogram = macd_histogram[:len(data.index)]
            macd_histogram = pd.Series(macd_histogram, index=data.index)

        # 统一列名
        result_df = pd.DataFrame({
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }, index=data.index)

        return result_df

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标的各种形态

        Args:
            data: 输入数据，至少包含'close'列
            **kwargs: 其他参数，用于覆盖默认参数

        Returns:
            一个包含各种形态布尔值的DataFrame
        """
        # 覆盖默认参数
        self._parameters.update(kwargs)

        # 核心计算
        macd_df = self._calculate(data, **self._parameters)
        dif = macd_df['macd_line']
        dea = macd_df['macd_signal']
        hist = macd_df['macd_histogram']

        # 初始化一个空的DataFrame来存储形态结果
        patterns_df = pd.DataFrame(index=data.index)

        # 1. 金叉和死叉
        patterns_df['MACD_GOLDEN_CROSS'] = self._detect_robust_crossover(dif, dea, window=3, cross_type='above')
        patterns_df['MACD_DEATH_CROSS'] = self._detect_robust_crossover(dif, dea, window=3, cross_type='below')

        # 2. 零轴穿越
        patterns_df['MACD_ZERO_CROSS_ABOVE'] = self._detect_robust_crossover(dif, 0, window=1, cross_type='above')
        patterns_df['MACD_ZERO_CROSS_BELOW'] = self._detect_robust_crossover(dif, 0, window=1, cross_type='below')

        # 3. 背离检测
        bullish_divergence, bearish_divergence = self._detect_divergence(
            data['close'],
            macd_df['macd_line'],
            window=self._parameters['divergence_window']
        )
        
        # 柱状图扩张和收缩
        histogram = macd_df['macd_histogram']
        histogram_expanding = pd.Series(False, index=data.index)
        histogram_contracting = pd.Series(False, index=data.index)
        
        # 检测连续3个柱状图扩张或收缩
        for i in range(3, len(histogram)):
            # 扩张: 连续3个柱状图变大
            if (histogram.iloc[i] > histogram.iloc[i-1] > histogram.iloc[i-2] > histogram.iloc[i-3]) and histogram.iloc[i] > 0:
                histogram_expanding.iloc[i] = True
            # 收缩: 连续3个柱状图变小
            elif (histogram.iloc[i] < histogram.iloc[i-1] < histogram.iloc[i-2] < histogram.iloc[i-3]) and histogram.iloc[i] < 0:
                histogram_contracting.iloc[i] = True
        
        # 双顶和双底形态
        double_top = pd.Series(False, index=data.index)
        double_bottom = pd.Series(False, index=data.index)
        
        # 使用简化的双顶双底检测逻辑
        from scipy.signal import find_peaks
        
        # 双顶
        macd_peaks, _ = find_peaks(dif.values, distance=10, prominence=0.1)
        if len(macd_peaks) >= 2:
            for i in range(1, len(macd_peaks)):
                idx = macd_peaks[i]
                if idx < len(double_top):
                    double_top.iloc[idx] = True
        
        # 双底
        macd_bottoms, _ = find_peaks(-dif.values, distance=10, prominence=0.1)
        if len(macd_bottoms) >= 2:
            for i in range(1, len(macd_bottoms)):
                idx = macd_bottoms[i]
                if idx < len(double_bottom):
                    double_bottom.iloc[idx] = True
        
        # 将所有形态信号合并到一个DataFrame
        patterns_df = pd.DataFrame({
            'MACD_GOLDEN_CROSS': patterns_df['MACD_GOLDEN_CROSS'],
            'MACD_DEATH_CROSS': patterns_df['MACD_DEATH_CROSS'],
            'MACD_BULLISH_DIVERGENCE': bullish_divergence,
            'MACD_BEARISH_DIVERGENCE': bearish_divergence,
            'MACD_ZERO_CROSS_ABOVE': patterns_df['MACD_ZERO_CROSS_ABOVE'],
            'MACD_ZERO_CROSS_BELOW': patterns_df['MACD_ZERO_CROSS_BELOW'],
            'MACD_HISTOGRAM_EXPANDING': histogram_expanding,
            'MACD_HISTOGRAM_CONTRACTING': histogram_contracting,
            'MACD_DOUBLE_TOP': double_top,
            'MACD_DOUBLE_BOTTOM': double_bottom
        }, index=data.index)
        
        # 合并基础计算结果和形态结果
        final_df = pd.concat([macd_df, patterns_df], axis=1)
        
        return final_df

    def get_indicator_type(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型字符串
        """
        return "MACD"

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD的原始得分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: MACD得分
        """
        # 首先获取所有形态
        patterns_df = self.get_patterns(data)
        
        # 初始化得分
        total_score = pd.Series(0.0, index=data.index)
        
        # 定义各形态的得分权重
        pattern_scores = {
            'MACD_GOLDEN_CROSS': 20.0,
            'MACD_DEATH_CROSS': -20.0,
            'MACD_BULLISH_DIVERGENCE': 25.0,
            'MACD_BEARISH_DIVERGENCE': -25.0,
            'MACD_ZERO_CROSS_ABOVE': 15.0,
            'MACD_ZERO_CROSS_BELOW': -15.0,
            'MACD_HISTOGRAM_EXPANDING': 10.0,
            'MACD_HISTOGRAM_CONTRACTING': -10.0,
            'MACD_DOUBLE_BOTTOM': 22.0,
            'MACD_DOUBLE_TOP': -22.0
        }
        
        # 计算各形态得分并累加
        for pattern, score in pattern_scores.items():
            if pattern in patterns_df.columns:
                # 将布尔列转换为0/1
                pattern_signal = patterns_df[pattern].astype(int)
                # 计算并累加得分
                total_score += pattern_signal * score
        
        # 基于MACD线和信号线的相对位置添加额外分数
        if 'macd_line' in patterns_df.columns and 'macd_signal' in patterns_df.columns:
            # 当MACD线在信号线上方，给予正分
            macd_above_signal = (patterns_df['macd_line'] > patterns_df['macd_signal']).astype(int) * 5.0
            # 当MACD线在信号线下方，给予负分
            macd_below_signal = (patterns_df['macd_line'] < patterns_df['macd_signal']).astype(int) * -5.0
            # 累加得分
            total_score += macd_above_signal + macd_below_signal
        
        # 基于MACD线相对于零轴的位置添加额外分数
        if 'macd_line' in patterns_df.columns:
            # 当MACD线在零轴上方，给予正分
            macd_above_zero = (patterns_df['macd_line'] > 0).astype(int) * 5.0
            # 当MACD线在零轴下方，给予负分
            macd_below_zero = (patterns_df['macd_line'] < 0).astype(int) * -5.0
            # 累加得分
            total_score += macd_above_zero + macd_below_zero
        
        return total_score

    def _detect_robust_crossover(self, series1: pd.Series, series2: Union[pd.Series, float, int], window: int = 3, cross_type: str = 'above') -> pd.Series:
        """
        更稳健的交叉检测，考虑交叉后的持续性
        
        Args:
            series1: 第一个序列 (例如, DIF)
            series2: 第二个序列 (例如, DEA 或一个常数)
            window: 确认交叉的窗口期
            cross_type: 'above' (金叉) 或 'below' (死叉)
            
        Returns:
            pd.Series: 交叉信号
        """
        # 初始化结果
        result = pd.Series(False, index=series1.index)
        
        # 数据不足以计算交叉
        if len(series1) < 3:
            return result
            
        # 如果series2是标量，将其转换为Series便于处理
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)
            
        # 计算前后位置关系
        if cross_type == 'above':
            # 查找从下向上穿越的点（金叉）
            # 前一点，series1低于或等于series2
            condition_before = series1.shift(1) <= series2.shift(1)
            # 当前点，series1高于series2
            condition_after = series1 > series2
            # 结合两个条件找到交叉点
            cross_points = condition_before & condition_after
        else:
            # 查找从上向下穿越的点（死叉）
            # 前一点，series1高于或等于series2
            condition_before = series1.shift(1) >= series2.shift(1)
            # 当前点，series1低于series2
            condition_after = series1 < series2
            # 结合两个条件找到交叉点
            cross_points = condition_before & condition_after
            
        # 找到所有潜在交叉点的索引
        cross_indices = np.where(cross_points)[0]
        
        # 没有发现交叉点，返回全False序列
        if len(cross_indices) == 0:
            return result
            
        # 对每个交叉点应用更严格的确认
        for idx in cross_indices:
            # 跳过开始的点，确保有前置数据
            if idx < 2:
                continue
                
            # 跳过结尾的点，确保有后续数据用于确认
            if idx >= len(series1) - window:
                continue
                
            # 确认交叉点前后的趋势方向
            if cross_type == 'above':
                # 金叉：确保交叉前series1一直低于series2，交叉后一直高于
                before_cross = series1.iloc[idx-window:idx] < series2.iloc[idx-window:idx]
                after_cross = series1.iloc[idx:idx+window] > series2.iloc[idx:idx+window]
                
                # 至少有window/2个点满足条件
                if before_cross.sum() >= window/2 and after_cross.sum() >= window/2:
                    result.iloc[idx] = True
            else:
                # 死叉：确保交叉前series1一直高于series2，交叉后一直低于
                before_cross = series1.iloc[idx-window:idx] > series2.iloc[idx-window:idx]
                after_cross = series1.iloc[idx:idx+window] < series2.iloc[idx:idx+window]
                
                # 至少有window/2个点满足条件
                if before_cross.sum() >= window/2 and after_cross.sum() >= window/2:
                    result.iloc[idx] = True
                    
        return result

    def _detect_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        检测价格与指标之间的背离
        
        Args:
            price: 价格序列 (例如, 'close')
            indicator: 指标序列 (例如, 'macd')
            window: 检测窗口
            
        Returns:
            Tuple[pd.Series, pd.Series]: (底背离信号, 顶背离信号)
        """
        # 初始化信号序列
        bullish_divergence = pd.Series(False, index=price.index)
        bearish_divergence = pd.Series(False, index=price.index)
        
        # 如果数据太短，直接返回
        if len(price) < window * 2:
            return bullish_divergence, bearish_divergence
        
        # 寻找价格和指标的局部高点和低点
        from scipy.signal import find_peaks
        
        # 使用更敏感的参数来找到足够的峰和谷
        price_peaks, _ = find_peaks(price.values, distance=window//2, prominence=0.01)
        price_troughs, _ = find_peaks(-price.values, distance=window//2, prominence=0.01)
        
        indicator_peaks, _ = find_peaks(indicator.values, distance=window//2, prominence=0.01)
        indicator_troughs, _ = find_peaks(-indicator.values, distance=window//2, prominence=0.01)
        
        # 确保我们找到了足够的峰和谷
        if len(price_peaks) < 2 or len(indicator_peaks) < 2:
            # 使用更宽松的参数再试一次
            price_peaks, _ = find_peaks(price.values, distance=window//3, prominence=0.005)
            indicator_peaks, _ = find_peaks(indicator.values, distance=window//3, prominence=0.005)
            
        if len(price_troughs) < 2 or len(indicator_troughs) < 2:
            # 使用更宽松的参数再试一次
            price_troughs, _ = find_peaks(-price.values, distance=window//3, prominence=0.005)
            indicator_troughs, _ = find_peaks(-indicator.values, distance=window//3, prominence=0.005)
        
        # 检测顶背离 (价格新高，指标未新高)
        if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
            for i in range(1, len(price_peaks)):
                p_peak1_idx, p_peak2_idx = price_peaks[i-1], price_peaks[i]
                
                # 找到与价格高点对应的指标高点
                # 选择靠近价格高点的指标高点
                ind_peaks_between = [idx for idx in indicator_peaks if 
                                    max(0, p_peak1_idx - window//2) <= idx <= min(len(indicator)-1, p_peak2_idx + window//2)]
                
                if len(ind_peaks_between) >= 2:
                    ind_peak1_idx, ind_peak2_idx = ind_peaks_between[0], ind_peaks_between[-1]
                    
                    # 判断顶背离条件：价格创新高，但指标未创新高
                    if (price.iloc[p_peak2_idx] > price.iloc[p_peak1_idx] and 
                        indicator.iloc[ind_peak2_idx] < indicator.iloc[ind_peak1_idx]):
                        bearish_divergence.iloc[p_peak2_idx] = True

        # 检测底背离 (价格新低，指标未新低)
        if len(price_troughs) >= 2 and len(indicator_troughs) >= 2:
            for i in range(1, len(price_troughs)):
                p_trough1_idx, p_trough2_idx = price_troughs[i-1], price_troughs[i]
                
                # 找到与价格低点对应的指标低点
                # 选择靠近价格低点的指标低点
                ind_troughs_between = [idx for idx in indicator_troughs if 
                                     max(0, p_trough1_idx - window//2) <= idx <= min(len(indicator)-1, p_trough2_idx + window//2)]
                
                if len(ind_troughs_between) >= 2:
                    ind_trough1_idx, ind_trough2_idx = ind_troughs_between[0], ind_troughs_between[-1]
                    
                    # 判断底背离条件：价格创新低，但指标未创新低
                    if (price.iloc[p_trough2_idx] < price.iloc[p_trough1_idx] and 
                        indicator.iloc[ind_trough2_idx] > indicator.iloc[ind_trough1_idx]):
                        bullish_divergence.iloc[p_trough2_idx] = True
                    
        return bullish_divergence, bearish_divergence

    def get_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成买入/卖出信号
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 包含买卖信号和其他分析结果的字典
        """
        # 首先，获取包含所有形态的DataFrame
        patterns_df = self.get_patterns(data)
        
        # 初始化信号字典
        signals = {
            'buy_signal': pd.Series(False, index=data.index),
            'sell_signal': pd.Series(False, index=data.index),
            'signals_details': []
        }
        
        # 定义买入和卖出形态
        buy_patterns = ["MACD_GOLDEN_CROSS", "MACD_BULLISH_DIVERGENCE", "MACD_ZERO_CROSS_ABOVE", "MACD_DOUBLE_BOTTOM"]
        sell_patterns = ["MACD_DEATH_CROSS", "MACD_BEARISH_DIVERGENCE", "MACD_ZERO_CROSS_BELOW", "MACD_DOUBLE_TOP"]
        
        # 聚合买入信号
        for pattern in buy_patterns:
            if pattern in patterns_df.columns:
                signals['buy_signal'] |= patterns_df[pattern]
        
        # 聚合卖出信号
        for pattern in sell_patterns:
            if pattern in patterns_df.columns:
                signals['sell_signal'] |= patterns_df[pattern]
        
        # 可以在这里添加更复杂的信号逻辑，例如组合多个形态
        
        # 记录详细信号
        for idx in data.index:
            triggered_patterns = []
            for col in patterns_df.columns:
                # 检查列是否存在且值为True
                if col in patterns_df and patterns_df.at[idx, col]:
                    triggered_patterns.append(col)
            
            if triggered_patterns:
                signals['signals_details'].append({
                    'date': idx,
                    'patterns': triggered_patterns
                })
                
        return signals

    def get_score(self, data: pd.DataFrame) -> float:
        """
        计算MACD指标在最后一个时间点的综合得分
        
        Args:
            data: 输入数据
            
        Returns:
            float: 最后一个时间点的综合得分
        """
        raw_score_series = self.calculate_raw_score(data)
        
        if raw_score_series.empty:
            return 0.0
            
        # 返回最后一个非空值
        last_score = raw_score_series.replace(0, pd.NA).ffill().iloc[-1]
        
        return last_score if pd.notna(last_score) else 0.0

    def analyze_pattern(self, pattern_id: str, data: pd.DataFrame) -> List[Dict]:
        """
        对指定的形态进行深入分析
        
        Args:
            pattern_id: 形态ID (例如, "MACD_GOLDEN_CROSS")
            data: 输入数据
            
        Returns:
            List[Dict]: 包含该形态发生详情的列表
        """
        # 获取所有形态
        patterns_df = self.get_patterns(data)
        
        # 检查形态ID是否存在
        if pattern_id not in patterns_df.columns:
            logger.warning(f"形态 '{pattern_id}' 未在结果中找到。")
            return []
            
        # 查找形态发生的时间点
        event_dates = patterns_df[patterns_df[pattern_id]].index
        
        analysis_results = []
        for date in event_dates:
            result = {
                'date': date,
                'pattern_id': pattern_id,
                'message': f"在 {date.strftime('%Y-%m-%d')} 检测到形态 '{pattern_id}'。",
                'context': {
                    'macd_line': patterns_df.at[date, 'macd_line'],
                    'macd_signal': patterns_df.at[date, 'macd_signal'],
                    'macd_histogram': patterns_df.at[date, 'macd_histogram'],
                    'close_price': data.at[date, 'close']
                }
            }
            analysis_results.append(result)
            
        return analysis_results

    def _analyze_golden_cross(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析金叉形态的具体情况
        """
        # ... 实现金叉的详细分析 ...
        # 此处可以返回更丰富的上下文信息
        return self.analyze_pattern("MACD_GOLDEN_CROSS", data)

    def _analyze_death_cross(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析死叉形态的具体情况
        """
        # ... 实现死叉的详细分析 ...
        return self.analyze_pattern("MACD_DEATH_CROSS", data)

    def _analyze_divergence(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析背离形态的具体情况
        """
        # ... 实现背离的详细分析 ...
        bullish_results = self.analyze_pattern("MACD_BULLISH_DIVERGENCE", data)
        bearish_results = self.analyze_pattern("MACD_BEARISH_DIVERGENCE", data)
        return bullish_results + bearish_results

    def _analyze_double_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析双顶/双底形态的具体情况
        """
        # ... 实现双顶/双底的详细分析 ...
        top_results = self.analyze_pattern("MACD_DOUBLE_TOP", data)
        bottom_results = self.analyze_pattern("MACD_DOUBLE_BOTTOM", data)
        return top_results + bottom_results

    def calculate_confidence(self, score: pd.Series, patterns: list, signals: dict) -> float:
        """
        计算指标的置信度
        
        Args:
            score: 得分序列
            patterns: 检测到的形态列表
            signals: 生成的信号字典
            
        Returns:
            float: 置信度分数 (0-1)
        """
        # 一个简单的置信度计算示例
        # 1. 得分的绝对值越大，置信度越高
        last_score = abs(score.iloc[-1])
        
        # 2. 检测到的形态越多，置信度越高
        num_patterns = len(patterns)
        
        # 3. 信号越强（例如，金叉后价格确实上涨），置信度越高
        
        # 归一化得分
        normalized_score = min(last_score / 50, 1.0) # 假设50是高分
        
        # 归一化形态数量
        normalized_patterns = min(num_patterns / 5, 1.0) # 假设5个形态是很多了
        
        # 综合置信度
        confidence = (normalized_score * 0.6) + (normalized_patterns * 0.4)
        
        return min(confidence, 1.0)

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典，包含name, description, strength等
        """
        pattern_info_map = {
            'MACD_GOLDEN_CROSS': {
                'name': 'MACD金叉',
                'description': 'MACD快线从下向上穿越慢线，看涨信号',
                'strength': 'strong',
                'type': 'bullish'
            },
            'MACD_DEATH_CROSS': {
                'name': 'MACD死叉',
                'description': 'MACD快线从上向下穿越慢线，看跌信号',
                'strength': 'strong',
                'type': 'bearish'
            },
            'MACD_ZERO_CROSS_ABOVE': {
                'name': 'MACD零轴向上穿越',
                'description': 'MACD线从下方穿越零轴，表明由空头转为多头',
                'strength': 'medium',
                'type': 'bullish'
            },
            'MACD_ZERO_CROSS_BELOW': {
                'name': 'MACD零轴向下穿越',
                'description': 'MACD线从上方穿越零轴，表明由多头转为空头',
                'strength': 'medium',
                'type': 'bearish'
            },
            'MACD_BULLISH_DIVERGENCE': {
                'name': 'MACD底背离',
                'description': '价格创新低，但MACD未创新低，潜在看涨信号',
                'strength': 'very_strong',
                'type': 'bullish'
            },
            'MACD_BEARISH_DIVERGENCE': {
                'name': 'MACD顶背离',
                'description': '价格创新高，但MACD未创新高，潜在看跌信号',
                'strength': 'very_strong',
                'type': 'bearish'
            },
            'MACD_HISTOGRAM_EXPANDING': {
                'name': 'MACD柱状图扩张',
                'description': 'MACD柱状图连续增大，表明趋势加强',
                'strength': 'medium',
                'type': 'momentum'
            },
            'MACD_HISTOGRAM_CONTRACTING': {
                'name': 'MACD柱状图收缩',
                'description': 'MACD柱状图连续减小，表明趋势减弱',
                'strength': 'medium',
                'type': 'exhaustion'
            },
            'MACD_DOUBLE_BOTTOM': {
                'name': 'MACD双底',
                'description': 'MACD形成双底形态，看涨信号',
                'strength': 'strong',
                'type': 'bullish'
            },
            'MACD_DOUBLE_TOP': {
                'name': 'MACD双顶',
                'description': 'MACD形成双顶形态，看跌信号',
                'strength': 'strong',
                'type': 'bearish'
            }
        }

        return pattern_info_map.get(pattern_id, {
            'name': pattern_id,
            'description': f'MACD形态: {pattern_id}',
            'strength': 'medium',
            'type': 'neutral'
        })