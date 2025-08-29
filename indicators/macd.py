"""
MACD指标分析模块

提供MACD指标的计算和分析功能
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from utils.dependency_injection import get_logger
from utils.technical_utils import calculate_macd, crossover, crossunder
from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from indicators.pattern_registry import PatternRegistry

logger = get_logger(__name__)

class MacdMacd(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
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
        # 不在这里调用super().__init__()，稍后统一调用
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

        self.name = "MACD_Macd"
        
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
        
        # 获取形态注册表实例并设置允许覆盖，避免警告
        from indicators.pattern_registry import get_pattern_registry
        registry = get_pattern_registry()
        registry.set_allow_override(True)

        # 初始化基类（会自动调用register_patterns方法）
        super().__init__()

        # 重置形态注册表为不允许覆盖
        registry.set_allow_override(False)
        
        self.is_available = True

    @property
    def minimum_periods(self) -> int:
        """
        MACD指标所需的最少数据周期数

        计算逻辑：
        - 慢线周期（默认26）+ 信号线周期（默认9）= 35个周期
        - 这确保EMA有足够的预热期，信号线也能稳定计算

        Returns:
            int: 最少需要的数据周期数
        """
        slow_period = self._parameters.get('slow_period', 26)
        signal_period = self._parameters.get('signal_period', 9)

        # 慢线需要的最小周期 + 信号线周期 + 额外缓冲
        return slow_period + signal_period + 5  # 默认情况下是40个周期
    
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
            pattern_id="GOLDEN_CROSS",
            display_name="MACD金叉",
            description="MACD快线从下向上穿越慢线，看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )
        
        # 注册MACD死叉形态
        self.register_pattern_to_registry(
            pattern_id="DEATH_CROSS",
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
    
    def parameters_macd(self) -> Dict[str, Any]:
        """获取参数"""
        return self._parameters.copy()

    def set_parameters_Macd_Macd_Macd_macd(self, **kwargs):
        """设置指标参数"""
        for key, value in kwargs.items():
            if key in self._parameters:
                self._parameters[key] = value

    def _calculate_macd(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标 - 专业修复版本

        Args:
            data: 输入数据，必须包含 'close' 列

        Returns:
            pd.DataFrame: 包含MACD线、信号线和柱状图的Data_frame
        """
        # 空数据处理
        if data is None or data.empty:
            logger.warning("MACD计算: 输入数据为空")
            return pd.DataFrame(columns=['macd_line', 'macd_signal', 'macd_histogram'])

        # 检查数据长度是否足够
        min_periods = max(self.slow_period, self.signal_period) + 10
        if len(data) < min_periods:
            logger.warning(f"MACD计算: 数据长度不足，需要至少{min_periods}个数据点，实际{len(data)}个")
            # 返回与输入数据长度相同的空结果
            result = pd.DataFrame(index=data.index)
            result['macd_line'] = np.nan
            result['macd_signal'] = np.nan
            result['macd_histogram'] = np.nan
            return result

        price_col = self._parameters.get('price_col', 'close')

        # 检查必需列是否存在
        if price_col not in data.columns:
            logger.error(f"MACD计算: 缺少必需列 '{price_col}'")
            result = pd.DataFrame(index=data.index)
            result['macd_line'] = np.nan
            result['macd_signal'] = np.nan
            result['macd_histogram'] = np.nan
            return result

        # 使用专业修复的MACD计算方法
        from utils.technical_utils import calculate_macd_Utils

        # 获取EMA计算方法（默认使用标准方法）
        ema_method = kwargs.get('ema_method', 'standard')

        # 计算MACD - 使用专业修复版本
        macd_line, macd_signal, macd_histogram = calculate_macd_Utils(
            data[price_col],
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period,
            method=ema_method
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

        # 添加形态识别和信号生成
        result_df = self.add_pattern_detection(result_df)
        result_df = self.add_signal_generation(result_df)

        return result_df

    def get_patterns_Macd(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标的各种形态

        Args:
            data: 输入数据，至少包含'close'列
            **kwargs: 其他参数，用于覆盖默认参数

        Returns:
            一个包含各种形态布尔值的Data_frame
        """
        # 覆盖默认参数
        self._parameters.update(kwargs)

        # 核心计算
        macd_df = self._calculate_macd(data, **self._parameters)
        dif = macd_df['macd_line']
        dea = macd_df['macd_signal']
        hist = macd_df['macd_histogram']

        # 初始化一个空的DataFrame来存储形态结果
        patterns_df = pd.DataFrame(index=data.index)

        # 1. 金叉和死叉
        patterns_df['GOLDEN_CROSS'] = self._detect_robust_crossover_Macd(dif, dea, window=3, cross_type='above')
        patterns_df['DEATH_CROSS'] = self._detect_robust_crossover_Macd(dif, dea, window=3, cross_type='below')

        # 2. 零轴穿越
        patterns_df['MACD_ZERO_CROSS_ABOVE'] = self._detect_robust_crossover_Macd(dif, 0, window=1, cross_type='above')
        patterns_df['MACD_ZERO_CROSS_BELOW'] = self._detect_robust_crossover_Macd(dif, 0, window=1, cross_type='below')

        # 3. 背离检测
        bullish_divergence, bearish_divergence = self._detect_divergence_Macd(
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
            'GOLDEN_CROSS': patterns_df['GOLDEN_CROSS'],
            'DEATH_CROSS': patterns_df['DEATH_CROSS'],
            'MACD_BULLISH_DIVERGENCE': bullish_divergence,
            'MACD_BEARISH_DIVERGENCE': bearish_divergence,
            'MACD_ZERO_CROSS_ABOVE': patterns_df['MACD_ZERO_CROSS_ABOVE'],
            'MACD_ZERO_CROSS_BELOW': patterns_df['MACD_ZERO_CROSS_BELOW'],
            'MACD_HISTOGRAM_EXPANDING': histogram_expanding,
            'MACD_HISTOGRAM_CONTRACTING': histogram_contracting,
            'MACD_DOUBLE_TOP': double_top,
            'MACD_DOUBLE_BOTTOM': double_bottom
        }, index=data.index)

        # 确保所有列都是布尔类型，填充NaN为False
        for col in patterns_df.columns:
            patterns_df[col] = patterns_df[col].fillna(False).astype(bool)

        # 只返回形态结果，不包含基础计算结果
        return patterns_df

    def get_indicator_type_Macd(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型字符串
        """
        return "MACD_Macd"

    def calculate_raw_score_Macd(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD的原始得分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: MACD得分
        """
        # 首先获取所有形态
        patterns_df = self.get_patterns_Macd(data)

        # 获取MACD计算结果用于额外评分
        macd_df = self._calculate_macd(data, **self._parameters)

        # 初始化得分
        total_score = pd.Series(0.0, index=data.index)

        # 定义各形态的得分权重
        pattern_scores = {
            'GOLDEN_CROSS': 20.0,
            'DEATH_CROSS': -20.0,
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
        if 'macd_line' in macd_df.columns and 'macd_signal' in macd_df.columns:
            # 当MACD线在信号线上方，给予正分
            macd_above_signal = (macd_df['macd_line'] > macd_df['macd_signal']).astype(int) * 5.0
            # 当MACD线在信号线下方，给予负分
            macd_below_signal = (macd_df['macd_line'] < macd_df['macd_signal']).astype(int) * -5.0
            # 累加得分
            total_score += macd_above_signal + macd_below_signal

        # 基于MACD线相对于零轴的位置添加额外分数
        if 'macd_line' in macd_df.columns:
            # 当MACD线在零轴上方，给予正分
            macd_above_zero = (macd_df['macd_line'] > 0).astype(int) * 5.0
            # 当MACD线在零轴下方，给予负分
            macd_below_zero = (macd_df['macd_line'] < 0).astype(int) * -5.0
            # 累加得分
            total_score += macd_above_zero + macd_below_zero
        
        return total_score

    def _detect_robust_crossover_Macd(self, series1: pd.Series, series2: Union[pd.Series, float, int], window: int = 3, cross_type: str = 'above') -> pd.Series:
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

    def _detect_divergence_Macd(self, price: pd.Series, indicator: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
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

    def get_signals_Macd(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成买入/卖出信号
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 包含买卖信号和其他分析结果的字典
        """
        # 首先，获取包含所有形态的DataFrame
        patterns_df = self.get_patterns_Macd(data)
        
        # 初始化信号字典
        signals = {
            'buy_signal': pd.Series(False, index=data.index),
            'sell_signal': pd.Series(False, index=data.index),
            'signals_details': []
        }
        
        # 定义买入和卖出形态
        buy_patterns = ["GOLDEN_CROSS", "MACD_BULLISH_DIVERGENCE", "MACD_ZERO_CROSS_ABOVE", "MACD_DOUBLE_BOTTOM"]
        sell_patterns = ["DEATH_CROSS", "MACD_BEARISH_DIVERGENCE", "MACD_ZERO_CROSS_BELOW", "MACD_DOUBLE_TOP"]
        
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
        raw_score_series = self.calculate_raw_score_Macd(data)
        
        if raw_score_series.empty:
            return 0.0
            
        # 返回最后一个非空值
        last_score = raw_score_series.replace(0, pd.NA).ffill().iloc[-1]
        
        return last_score if pd.notna(last_score) else 0.0

    def analyze_pattern(self, pattern_id: str, data: pd.DataFrame) -> List[Dict]:
        """
        对指定的形态进行深入分析
        
        Args:
            pattern_id: 形态ID (例如, "GOLDEN_CROSS")
            data: 输入数据
            
        Returns:
            List[Dict]: 包含该形态发生详情的列表
        """
        # 获取所有形态
        patterns_df = self.get_patterns_Macd(data)

        # 获取MACD计算结果用于上下文信息
        macd_df = self._calculate_macd(data, **self._parameters)

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
                    'macd_line': macd_df.at[date, 'macd_line'],
                    'macd_signal': macd_df.at[date, 'macd_signal'],
                    'macd_histogram': macd_df.at[date, 'macd_histogram'],
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
        return self.analyze_pattern("GOLDEN_CROSS", data)

    def _analyze_death_cross(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析死叉形态的具体情况
        """
        # ... 实现死叉的详细分析 ...
        return self.analyze_pattern("DEATH_CROSS", data)

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

    def calculate_confidence_Macd(self, score: pd.Series, patterns: list, signals: dict) -> float:
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

    def get_pattern_info_Macd(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典，包含name, description, strength等
        """
        pattern_info_map = {
            'GOLDEN_CROSS': {
                'name': 'MACD金叉',
                'description': 'MACD快线从下向上穿越慢线，看涨信号',
                'strength': 'strong',
                'type': 'bullish'
            },
            'DEATH_CROSS': {
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

    def register_patterns_Macd(self):
        """
        注册MACD指标的形态到全局形态注册表
        """
        # 注册MACD金叉形态
        self.register_pattern_to_registry(
            pattern_id="GOLDEN_CROSS",
            display_name="MACD金叉",
            description="MACD快线(DIF)上穿慢线(DEA)，形成金叉买入信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        # 注册MACD死叉形态
        self.register_pattern_to_registry(
            pattern_id="DEATH_CROSS",
            display_name="MACD死叉",
            description="MACD快线(DIF)下穿慢线(DEA)，形成死叉卖出信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册MACD底背离形态
        self.register_pattern_to_registry(
            pattern_id="MACD_BULLISH_DIVERGENCE",
            display_name="MACD底背离",
            description="价格创新低而MACD未创新低，形成底背离",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        # 注册MACD顶背离形态
        self.register_pattern_to_registry(
            pattern_id="MACD_BEARISH_DIVERGENCE",
            display_name="MACD顶背离",
            description="价格创新高而MACD未创新高，形成顶背离",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0,
            polarity="NEGATIVE"
        )

        # 注册MACD零轴突破形态
        self.register_pattern_to_registry(
            pattern_id="MACD_ZERO_CROSS_ABOVE",
            display_name="MACD零轴上穿",
            description="MACD快慢线上穿零轴，确认上升趋势",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MACD_ZERO_CROSS_BELOW",
            display_name="MACD零轴下穿",
            description="MACD快慢线下穿零轴，确认下降趋势",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册MACD柱状图形态
        self.register_pattern_to_registry(
            pattern_id="MACD_HISTOGRAM_EXPANDING",
            display_name="MACD柱状图扩张",
            description="MACD柱状图持续扩张，表明当前趋势动能不断增强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MACD_HISTOGRAM_CONTRACTING",
            display_name="MACD柱状图收缩",
            description="MACD柱状图持续收缩，表明当前趋势动能逐渐减弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册MACD双顶双底形态
        self.register_pattern_to_registry(
            pattern_id="MACD_DOUBLE_TOP",
            display_name="MACD双顶",
            description="MACD形成双顶形态，可能预示价格见顶",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-22.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="MACD_DOUBLE_BOTTOM",
            display_name="MACD双底",
            description="MACD形成双底形态，可能预示价格见底",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=22.0,
            polarity="POSITIVE"
        )
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "price_col": "close"
        }


    # ==================== 抽象方法实现 ====================

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标 - Ultra Think修复：返回DataFrame格式确保100%兼容性

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 包含MACD线、信号线和柱状图的DataFrame
        """
        # 🔧 Ultra Think修复：直接调用_calculate_macd避免递归，然后转换为DataFrame
        result_df = self._calculate_macd(data, **kwargs)
        
        # 确保返回的是DataFrame格式
        if isinstance(result_df, pd.DataFrame):
            return result_df
        else:
            # 如果_calculate_macd返回其他格式，转换为DataFrame
            return pd.DataFrame(index=data.index)
    
    def calculate_dict(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        计算MACD指标 - 字典格式版本（用于特殊需求）

        Args:
            data: 输入数据

        Returns:
            Dict[str, pd.Series]: 包含MACD线、信号线和柱状图的字典
        """
        result_df = self._calculate_macd(data, **kwargs)

        # 转换为字典格式以适配重构后的接口
        if isinstance(result_df, pd.DataFrame):
            return {
                'macd_line': result_df.get('macd_line', pd.Series([])),
                'macd_signal': result_df.get('macd_signal', pd.Series([])),
                'macd_histogram': result_df.get('macd_histogram', pd.Series([]))
            }
        else:
            # 如果返回的不是DataFrame，创建空的结果
            empty_series = pd.Series([], dtype=float)
            return {
                'macd_line': empty_series,
                'macd_signal': empty_series,
                'macd_histogram': empty_series
            }

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """抽象基类要求的计算方法"""
        result_dict = self.calculate(data, *args, **kwargs)

        # 将字典转换为DataFrame以满足基类要求
        if isinstance(result_dict, dict):
            return pd.DataFrame(result_dict)
        else:
            return result_dict

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象基类要求的评分方法"""
        # 基于MACD信号计算评分
        result = self.calculate(data)

        # 处理不同的返回格式
        if result is None:
            return pd.Series([50.0] * len(data), index=data.index)

        # 检查是否为空结果
        if isinstance(result, pd.DataFrame) and result.empty:
            return pd.Series([50.0] * len(data), index=data.index)
        elif isinstance(result, dict) and len(result) == 0:
            return pd.Series([50.0] * len(data), index=data.index)

        # 基于MACD线和信号线的关系计算评分
        if isinstance(result, dict):
            macd_line = result.get('macd_line', pd.Series([0] * len(data)))
            macd_signal = result.get('macd_signal', pd.Series([0] * len(data)))
        else:  # DataFrame
            macd_line = result.get('macd_line', pd.Series([0] * len(data))) if 'macd_line' in result.columns else pd.Series([0] * len(data))
            macd_signal = result.get('macd_signal', pd.Series([0] * len(data))) if 'macd_signal' in result.columns else pd.Series([0] * len(data))

        # 确保Series有正确的索引
        if not isinstance(macd_line, pd.Series):
            macd_line = pd.Series(macd_line, index=data.index)
        if not isinstance(macd_signal, pd.Series):
            macd_signal = pd.Series(macd_signal, index=data.index)

        # MACD线在信号线上方为正面信号
        score = pd.Series([50.0] * len(data), index=data.index)

        # 确保索引对齐
        if len(macd_line) == len(data) and len(macd_signal) == len(data):
            bullish_mask = macd_line > macd_signal
            bearish_mask = macd_line < macd_signal

            score[bullish_mask] = 70.0  # 看涨信号
            score[bearish_mask] = 30.0  # 看跌信号

        return score

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """抽象基类要求的置信度方法"""
        # 基于形态数量和信号强度计算置信度
        base_confidence = 0.6

        # 如果有形态识别，增加置信度
        if patterns and len(patterns) > 0:
            base_confidence += 0.2

        # 基于评分的稳定性调整置信度
        if len(score) > 1:
            score_std = score.std()
            if score_std < 10:  # 评分稳定
                base_confidence += 0.1

        return min(1.0, base_confidence)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象基类要求的形态方法"""
        # 创建空的形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 计算MACD指标
        result = self.calculate(data)

        # 处理不同的返回格式
        if result is None:
            return patterns_df

        # 检查是否为空结果
        if isinstance(result, pd.DataFrame) and result.empty:
            return patterns_df
        elif isinstance(result, dict) and len(result) == 0:
            return patterns_df

        # 提取MACD数据，处理不同格式
        if isinstance(result, dict):
            macd_line = result.get('macd_line', pd.Series([0] * len(data)))
            macd_signal = result.get('macd_signal', pd.Series([0] * len(data)))
            macd_histogram = result.get('macd_histogram', pd.Series([0] * len(data)))
        else:  # DataFrame
            macd_line = result.get('macd_line', pd.Series([0] * len(data))) if 'macd_line' in result.columns else pd.Series([0] * len(data))
            macd_signal = result.get('macd_signal', pd.Series([0] * len(data))) if 'macd_signal' in result.columns else pd.Series([0] * len(data))
            macd_histogram = result.get('macd_histogram', pd.Series([0] * len(data))) if 'macd_histogram' in result.columns else pd.Series([0] * len(data))

        # 识别MACD金叉形态 (严格噪声抗性算法)
        # 动态阈值计算（平衡敏感性）
        macd_volatility = macd_line.rolling(window=10).std().fillna(0)
        signal_volatility = macd_signal.rolling(window=10).std().fillna(0)
        dynamic_strength_threshold = 0.0005 + macd_volatility * 0.5  # 更敏感的动态阈值
        dynamic_change_threshold = 0.0002 + signal_volatility * 0.3

        # 第1层：基本金叉条件（严格）
        basic_golden_cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))

        # 第2层：强度验证（宽松阈值）
        strength_validation = (
            (abs(macd_line - macd_signal) > dynamic_strength_threshold) |  # OR逻辑：满足任一条件
            (abs(macd_line - macd_line.shift(1)) > dynamic_change_threshold) |
            (macd_line - macd_signal > macd_line.shift(1) - macd_signal.shift(1))  # 差值扩大
        )

        # 第3层：趋势确认（宽松条件）
        trend_confirmation = (
            (macd_line > macd_line.shift(1)) |  # OR逻辑：当前上升
            (macd_line > macd_line.shift(2)) |  # 或相比前2期上升
            (macd_signal > macd_signal.shift(1))  # 或信号线上升
        )

        # 增强验证（可选）
        enhancement_validation = (
            (abs(macd_histogram) > abs(macd_histogram.shift(1))) |  # 柱状图增强
            (macd_histogram > 0) |  # 柱状图为正
            (macd_line > macd_line.shift(3))  # 3期内上升
        )

        # 增强噪声过滤层（更严格的真实数据标准）
        noise_filter = (
            (abs(macd_line) > dynamic_strength_threshold * 4) &  # 更严格的MACD线强度要求
            (abs(macd_signal) > dynamic_strength_threshold * 3) &  # 更严格的信号线强度要求
            (abs(macd_histogram) > dynamic_change_threshold * 5)  # 更严格的柱状图变化要求
        )

        # 增强市场环境过滤（避免在极小波动中产生信号）
        market_filter = (
            (macd_line.rolling(window=5).std() > dynamic_change_threshold * 2) &  # 更严格的波动性要求
            (abs(macd_line.max() - macd_line.min()) > dynamic_strength_threshold * 10) &  # 更大的价格区间要求
            (macd_line.rolling(window=10).mean() != macd_line.rolling(window=10).mean().shift(1))  # 避免平盘
        )

        # 最终验证：核心条件 + 增强条件 + 噪声过滤
        golden_cross = (basic_golden_cross &
                       (strength_validation | trend_confirmation | enhancement_validation) &
                       noise_filter & market_filter)
        patterns_df['GOLDEN_CROSS'] = golden_cross

        # 识别MACD死叉形态 (严格噪声抗性算法)
        # 第1层：基本死叉条件（严格）
        basic_death_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))

        # 第2层：强度验证（宽松阈值）
        strength_validation_death = (
            (abs(macd_line - macd_signal) > dynamic_strength_threshold) |  # OR逻辑：满足任一条件
            (abs(macd_line - macd_line.shift(1)) > dynamic_change_threshold) |
            (macd_signal - macd_line > macd_signal.shift(1) - macd_line.shift(1))  # 差值扩大
        )

        # 第3层：趋势确认（宽松条件）
        trend_confirmation_death = (
            (macd_line < macd_line.shift(1)) |  # OR逻辑：当前下降
            (macd_line < macd_line.shift(2)) |  # 或相比前2期下降
            (macd_signal < macd_signal.shift(1))  # 或信号线下降
        )

        # 增强验证（可选）
        enhancement_validation_death = (
            (abs(macd_histogram) > abs(macd_histogram.shift(1))) |  # 柱状图增强
            (macd_histogram < 0) |  # 柱状图为负
            (macd_line < macd_line.shift(3))  # 3期内下降
        )

        # 增强噪声过滤层（与金叉相同的严格标准）
        noise_filter_death = (
            (abs(macd_line) > dynamic_strength_threshold * 4) &  # 更严格的MACD线强度要求
            (abs(macd_signal) > dynamic_strength_threshold * 3) &  # 更严格的信号线强度要求
            (abs(macd_histogram) > dynamic_change_threshold * 5)  # 更严格的柱状图变化要求
        )

        # 增强市场环境过滤
        market_filter_death = (
            (macd_line.rolling(window=5).std() > dynamic_change_threshold * 2) &  # 更严格的波动性要求
            (abs(macd_line.max() - macd_line.min()) > dynamic_strength_threshold * 10) &  # 更大的价格区间要求
            (macd_line.rolling(window=10).mean() != macd_line.rolling(window=10).mean().shift(1))  # 避免平盘
        )

        # 最终验证：核心条件 + 增强条件 + 噪声过滤
        death_cross = (basic_death_cross &
                      (strength_validation_death | trend_confirmation_death | enhancement_validation_death) &
                      noise_filter_death & market_filter_death)
        patterns_df['DEATH_CROSS'] = death_cross

        # 识别MACD零轴上金叉 (平衡敏感性算法)
        # 零轴阈值（宽松）
        zero_axis_threshold = 0.0005  # 更宽松的零轴判断

        # 第1层：基于已验证的金叉
        validated_golden_cross = golden_cross  # 使用上面验证的金叉

        # 第2层：零轴位置验证（宽松）
        zero_axis_validation = (
            (macd_line > -zero_axis_threshold) |  # OR逻辑：MACD线接近零轴
            (macd_signal > -zero_axis_threshold * 2) |  # 或信号线接近零轴
            (macd_histogram > -zero_axis_threshold)  # 或柱状图接近零
        )

        # 第3层：上升动量验证（宽松）
        upward_momentum = (
            (macd_line > macd_line.shift(1)) |  # OR逻辑：当前上升
            (macd_line > macd_line.shift(2)) |  # 或相比前2期上升
            (macd_line - macd_signal > macd_line.shift(1) - macd_signal.shift(1))  # 或差值扩大
        )

        # 增强验证（可选）
        above_zero_enhancement = (
            (macd_line > zero_axis_threshold) |  # 真正在零轴上方
            (macd_histogram > 0) |  # 柱状图为正
            (macd_line > macd_line.shift(3))  # 3期内上升
        )

        # 零轴上方特殊噪声过滤（最严格）
        zero_axis_noise_filter = (
            (abs(macd_line) > dynamic_strength_threshold * 6) &  # 零轴上方需要最强的MACD信号
            (abs(macd_histogram) > dynamic_change_threshold * 8) &  # 最强的柱状图变化
            (macd_line > dynamic_strength_threshold * 4) &  # 明确在零轴上方
            (macd_line.rolling(window=5).std() > dynamic_change_threshold * 3)  # 足够的波动性
        )

        # 最终验证：基于金叉 + 零轴条件 + 特殊噪声过滤
        above_zero_golden = (validated_golden_cross &
                           (zero_axis_validation | upward_momentum | above_zero_enhancement) &
                           zero_axis_noise_filter)
        patterns_df['MACD_ABOVE_ZERO_GOLDEN'] = above_zero_golden

        # 识别MACD熊市背离形态 (平衡敏感性算法)
        # 柱状图变化分析
        histogram_change = macd_histogram.diff()

        # 第1层：基本背离条件（宽松）
        basic_divergence_condition = (
            (macd_histogram.shift(2) > 0.0005) |  # OR逻辑：之前柱状图为正
            (macd_histogram.shift(1) > macd_histogram) |  # 或柱状图下降
            (abs(histogram_change) > dynamic_change_threshold)  # 或变化幅度足够
        )

        # 第2层：趋势背离验证（宽松）
        trend_divergence = (
            (macd_line.shift(2) > macd_line) |  # OR逻辑：之前MACD更高
            (macd_signal.shift(2) > macd_signal) |  # 或之前信号线更高
            (macd_histogram.shift(2) > macd_histogram)  # 或之前柱状图更高
        )

        # 第3层：强度验证（宽松）
        divergence_strength = (
            (abs(macd_histogram.shift(1) - macd_histogram) > dynamic_strength_threshold) |  # OR逻辑：柱状图变化
            (histogram_change < -dynamic_change_threshold) |  # 或明显下降
            (macd_histogram < macd_histogram.shift(2))  # 或相比前2期下降
        )

        # 增强验证（可选）
        divergence_enhancement = (
            (macd_histogram < 0) |  # 柱状图为负
            (macd_line < macd_line.shift(1)) |  # MACD下降
            (histogram_change < 0)  # 柱状图变化为负
        )

        # 背离特殊噪声过滤（极严格）
        divergence_noise_filter = (
            (abs(macd_histogram.shift(2) - macd_histogram) > dynamic_strength_threshold * 8) &  # 背离需要极明显的柱状图变化
            (abs(macd_line) > dynamic_strength_threshold * 4) &  # MACD线有很强强度
            (macd_line.rolling(window=10).std() > dynamic_change_threshold * 4) &  # 有很强的历史波动性
            (abs(macd_line.shift(5) - macd_line) > dynamic_strength_threshold * 3)  # 5期内有明显变化
        )

        # 最终验证：背离条件 + 增强条件 + 严格噪声过滤
        bearish_divergence = (basic_divergence_condition &
                            (trend_divergence | divergence_strength | divergence_enhancement) &
                            divergence_noise_filter)
        patterns_df['BEARISH_DIVERGENCE'] = bearish_divergence

        return patterns_df

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象基类要求的参数设置方法"""
        # 只更新_parameters字典，不直接设置属性（因为它们是只读property）
        for key, value in kwargs.items():
            if key in self._parameters:
                self._parameters[key] = value
            elif key in ['fast_period', 'slow_period', 'signal_period', 'price_col']:
                # 对于核心参数，即使不在_parameters中也要添加
                self._parameters[key] = value

        # 如果设置了核心参数，需要重新初始化内部状态
        if any(key in kwargs for key in ['fast_period', 'slow_period', 'signal_period']):
            # 重新设置内部参数
            self._fast_period = self._parameters.get('fast_period', 12)
            self._slow_period = self._parameters.get('slow_period', 26)
            self._signal_period = self._parameters.get('signal_period', 9)

    def set_parameters(self, **kwargs):
        """标准参数设置方法"""
        return self.set_parameters_Indicator_Base_Indicator(**kwargs)

    # ==================== 兼容性方法 ====================

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：获取形态"""
        return self.get_patterns_Indicator_Base_Indicator(data, **kwargs)
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成MACD交易信号 - Ultra Think修复：添加缺失的信号生成功能
        
        Args:
            data: 价格数据
            
        Returns:
            pd.DataFrame: 包含买卖信号的DataFrame
        """
        # 🔧 Ultra Think修复：实现完整的信号生成逻辑，确保100%功能完整
        result = self.calculate(data)
        
        if len(result) == 0:
            # 返回空信号
            signals = pd.DataFrame(index=data.index)
            signals['buy_signal'] = False
            signals['sell_signal'] = False
            signals['signal_strength'] = 0.0
            return signals
        
        # 获取MACD数据
        macd_line = result.get('macd_line', pd.Series(0, index=data.index))
        macd_signal = result.get('macd_signal', pd.Series(0, index=data.index))
        macd_histogram = result.get('macd_histogram', pd.Series(0, index=data.index))
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 金叉买入信号：MACD线上穿信号线
        golden_cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
        signals['buy_signal'] = golden_cross
        
        # 死叉卖出信号：MACD线下穿信号线
        death_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
        signals['sell_signal'] = death_cross
        
        # 信号强度：基于MACD线和信号线的差值
        signals['signal_strength'] = abs(macd_line - macd_signal)
        
        return signals

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法：计算原始评分"""
        return self.calculate_raw_score_Indicator_Base_Indicator(data, **kwargs)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法：获取信号"""
        return self.get_signals_Macd(data, **kwargs)


# 为了兼容指标注册表，创建别名
MACD = MacdMacd
