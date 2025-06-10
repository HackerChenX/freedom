"""
MACD指标分析模块

提供MACD指标的计算和分析功能
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from utils.technical_utils import calculate_macd
from indicators.base_indicator import BaseIndicator
from utils.signal_utils import crossover, crossunder

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
        from indicators.pattern_registry import PatternRegistry
        PatternRegistry.set_allow_override(True)
        
        # 初始化基类（会自动调用register_patterns方法）
        super().__init__()
        
        # 重置形态注册表为不允许覆盖
        PatternRegistry.set_allow_override(False)
    
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
            score_impact=20.0
        )
        
        # 注册MACD死叉形态
        self.register_pattern_to_registry(
            pattern_id="MACD_DEATH_CROSS",
            display_name="MACD死叉",
            description="MACD快线从上向下穿越慢线，看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0
        )
        
        # 注册MACD零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="MACD_ZERO_CROSS_ABOVE",
            display_name="MACD零轴向上穿越",
            description="MACD线从下方穿越零轴，表明由空头转为多头",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0
        )
        
        self.register_pattern_to_registry(
            pattern_id="MACD_ZERO_CROSS_BELOW",
            display_name="MACD零轴向下穿越",
            description="MACD线从上方穿越零轴，表明由多头转为空头",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0
        )
        
        # 注册MACD背离形态
        self.register_pattern_to_registry(
            pattern_id="MACD_BULLISH_DIVERGENCE",
            display_name="MACD底背离",
            description="价格创新低，但MACD未创新低，潜在看涨信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0
        )
        
        self.register_pattern_to_registry(
            pattern_id="MACD_BEARISH_DIVERGENCE",
            display_name="MACD顶背离",
            description="价格创新高，但MACD未创新高，潜在看跌信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0
        )
        
        # 注册MACD柱状图形态
        self.register_pattern_to_registry(
            pattern_id="MACD_HISTOGRAM_EXPANDING",
            display_name="MACD柱状图扩张",
            description="MACD柱状图连续增大，表明趋势加强",
            pattern_type="MOMENTUM",
            default_strength="MEDIUM",
            score_impact=10.0
        )
        
        self.register_pattern_to_registry(
            pattern_id="MACD_HISTOGRAM_CONTRACTING",
            display_name="MACD柱状图收缩",
            description="MACD柱状图连续减小，表明趋势减弱",
            pattern_type="EXHAUSTION",
            default_strength="MEDIUM",
            score_impact=-10.0
        )
        
        # 注册MACD趋势形态
        self.register_pattern_to_registry(
            pattern_id="MACD_STRONG_BULLISH",
            display_name="MACD强势多头",
            description="MACD值处于高位且继续上升，表明强劲上涨趋势",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0
        )
        
        self.register_pattern_to_registry(
            pattern_id="MACD_STRONG_BEARISH",
            display_name="MACD强势空头",
            description="MACD值处于低位且继续下降，表明强劲下跌趋势",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-18.0
        )
        
        # 注册新增的MACD双顶双底形态
        self.register_pattern_to_registry(
            pattern_id="MACD_DOUBLE_BOTTOM",
            display_name="MACD双底",
            description="MACD形成双底形态，看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=22.0
        )
        
        self.register_pattern_to_registry(
            pattern_id="MACD_DOUBLE_TOP",
            display_name="MACD双顶",
            description="MACD形成双顶形态，看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-22.0
        )
        
        # 注册新增的MACD形态
        self.register_pattern_to_registry(
            pattern_id="MACD_TRIPLE_CROSS",
            display_name="MACD三重穿越",
            description="MACD短期内多次穿越信号线，表明市场不稳定",
            pattern_type="VOLATILITY",
            default_strength="WEAK",
            score_impact=0.0
        )
        
        self.register_pattern_to_registry(
            pattern_id="MACD_ZERO_LINE_HESITATION",
            display_name="MACD零轴徘徊",
            description="MACD在零轴附近徘徊，表明市场处于犹豫状态",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0
        )
    
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
                
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data: 包含价格数据的DataFrame
            **kwargs: 额外参数，可以包含:
                - price_column: 用于计算的价格列，默认为'close'
                
        Returns:
            pd.DataFrame: 添加了MACD指标的DataFrame
        """
        if data.empty:
            return data
            
        # 获取参数
        price_column = kwargs.get('price_column', 'close')
        
        # 验证输入
        if price_column not in data.columns:
            raise ValueError(f"数据中不存在'{price_column}'列")
        
        df = data.copy()
        
        # 计算快速EMA
        df['ema_fast'] = df[price_column].ewm(span=self.fast_period, adjust=False).mean()
        
        # 计算慢速EMA
        df['ema_slow'] = df[price_column].ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算DIF (MACD线)
        df['dif'] = df['ema_fast'] - df['ema_slow']
        
        # 计算DEA (MACD信号线)
        df['dea'] = df['dif'].ewm(span=self.signal_period, adjust=False).mean()
        
        # 计算MACD柱状体
        df['macd'] = (df['dif'] - df['dea']) * 2
        
        # 保存结果
        self._result = df
        
        # 确保基础数据列被保留
        df = self._preserve_base_columns(data, df)
        
        return df

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取MACD形态
        
        Args:
            data: 输入数据
        
        Returns:
            pd.DataFrame: 包含MACD形态的DataFrame
        """
        macd_columns = ['macd_line', 'macd_signal', 'macd_histogram']
        # 检查是否已计算MACD，如果未计算，则进行计算
        if not all(col in data.columns for col in macd_columns):
            indicator_df = self.calculate(data, **kwargs)
        else:
            indicator_df = data

        # 检测金叉和死叉
        golden_cross = crossover(indicator_df['macd_line'], indicator_df['macd_signal'])
        death_cross = crossunder(indicator_df['macd_line'], indicator_df['macd_signal'])

        # 创建结果DataFrame
        patterns_df = pd.DataFrame(index=indicator_df.index)
        patterns_df['MACD_GOLDEN_CROSS'] = golden_cross
        patterns_df['MACD_DEATH_CROSS'] = death_cross

        # 检测零轴穿越
        patterns_df['MACD_ZERO_CROSS_ABOVE'] = self.crossover(indicator_df['macd_line'], 0)
        patterns_df['MACD_ZERO_CROSS_BELOW'] = self.crossunder(indicator_df['macd_line'], 0)

        # 检测柱状图扩张/收缩
        patterns_df['MACD_HISTOGRAM_EXPANDING'] = ((indicator_df['macd_histogram'] > indicator_df['macd_histogram'].shift(1)) & (indicator_df['macd_histogram'] > 0)) | ((indicator_df['macd_histogram'] < indicator_df['macd_histogram'].shift(1)) & (indicator_df['macd_histogram'] < 0))
        patterns_df['MACD_HISTOGRAM_CONTRACTING'] = ((indicator_df['macd_histogram'] < indicator_df['macd_histogram'].shift(1)) & (indicator_df['macd_histogram'] > 0)) | ((indicator_df['macd_histogram'] > indicator_df['macd_histogram'].shift(1)) & (indicator_df['macd_histogram'] < 0))

        # 检测背离
        bullish_div, bearish_div = self._detect_divergence(data['close'], indicator_df['macd_line'])
        patterns_df['MACD_BULLISH_DIVERGENCE'] = bullish_div
        patterns_df['MACD_BEARISH_DIVERGENCE'] = bearish_div

        # 合并形态列到指标数据中并返回
        return pd.concat([indicator_df, patterns_df], axis=1)

    def get_indicator_type(self) -> str:
        """获取指标类型"""
        return "MACD"

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD指标的原始评分（0-100分制）

        评分逻辑:
        - 金叉/死叉: 对评分有显著影响
        - 零轴位置: DIF在零轴上方为多头市场，下方为空头市场
        - 柱状图: 柱状图的值和变化趋势反映了动能
        - 背离: 强烈的反转信号
        """
        # 获取形态
        patterns = self.get_patterns(data, **kwargs)
        
        # 获取指标值
        indicator_df = self.result if self.has_result() else self.calculate(data, **kwargs)
        dif = indicator_df['macd_line']
        hist = indicator_df['macd_histogram']

        # 基础分
        score = pd.Series(50.0, index=data.index)

        # 1. 零轴位置影响 (基础趋势判断)
        score[dif > 0] += 10
        score[dif < 0] -= 10

        # 2. 柱状图影响 (动能判断)
        # 将柱状图归一化到-20到20的范围
        # 使用最近252个交易日（约一年）的数据进行归一化，避免受极值影响过大
        hist_norm = (hist / (hist.rolling(252).std().replace(0, 1))) * 5
        score += hist_norm.clip(-20, 20)
        
        # 3. 形态影响 (事件驱动)
        score[patterns['MACD_GOLDEN_CROSS']] += 15
        score[patterns['MACD_DEATH_CROSS']] -= 15
        score[patterns['MACD_BULLISH_DIVERGENCE']] += 25
        score[patterns['MACD_BEARISH_DIVERGENCE']] -= 25
        
        # 确保评分在0-100范围内
        return score.clip(0, 100)

    def _detect_robust_crossover(self, series1: pd.Series, series2: pd.Series, window: int = 3, cross_type: str = 'above') -> pd.Series:
        """
        检测鲁棒的交叉信号，过滤掉短暂的缠绕和抖动。

        Args:
            series1: 第一个序列 (e.g., DIF)
            series2: 第二个序列 (e.g., DEA)
            window: 交叉后状态需要维持的窗口期
            cross_type: 'above' (上穿/金叉) 或 'below' (下穿/死叉)

        Returns:
            pd.Series: 布尔型序列，标记有效交叉点
        """
        if cross_type == 'above':
            initial_cross = self.crossover(series1, series2)
            state_maintained = (series1 > series2).rolling(window=window).sum() == window
        elif cross_type == 'below':
            initial_cross = self.crossunder(series1, series2)
            state_maintained = (series1 < series2).rolling(window=window).sum() == window
        else:
            raise ValueError("cross_type must be 'above' or 'below'")

        # 信号必须是初始交叉点，并且在该点之后的状态得以维持
        # 我们需要将 state_maintained 的结果向后移动一个位置，因为我们要检查交叉 *之后* 的窗口
        robust_cross = initial_cross & state_maintained.shift(-(window-1))
        
        return robust_cross.fillna(False)

    def _detect_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        使用改进的向量化方法检测背离
        """
        # 计算滚动最低/最高价
        price_low = price.rolling(window=window).min()
        indicator_low = indicator.rolling(window=window).min()
        price_high = price.rolling(window=window).max()
        indicator_high = indicator.rolling(window=window).max()

        # 底背离: 价格创新低，但指标未创新低
        bullish_div = (price == price_low) & (price < price.shift(1)) & \
                      (indicator > indicator_low.shift(1))

        # 顶背离: 价格创新高，但指标未创新高
        bearish_div = (price == price_high) & (price > price.shift(1)) & \
                      (indicator < indicator_high.shift(1))

        return bullish_div.fillna(False), bearish_div.fillna(False)
    
    def get_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取MACD信号
        
        Args:
            data: 包含MACD指标的DataFrame
            
        Returns:
            包含信号的字典
        """
        # 确保已计算MACD指标和增强特征
        if not self.has_result():
            self.calculate(data)
            result = self._result
        else:
            result = data if 'macd_line' in data.columns else self._result
        
        signals = {
            'golden_cross': False,         # 金叉
            'death_cross': False,          # 死叉
            'hist_positive': False,        # 柱状图为正
            'hist_negative': False,        # 柱状图为负
            'hist_increasing': False,      # 柱状图增加
            'hist_decreasing': False,      # 柱状图减少
            'zero_cross_up': False,        # 零轴向上穿越
            'zero_cross_down': False,      # 零轴向下穿越
            'bullish_divergence': False,   # 底背离
            'bearish_divergence': False,   # 顶背离
            'double_bottom': False,        # 双底
            'double_top': False,           # 双顶
            'triple_cross': False,         # 三重穿越
            'zero_hesitation': False       # 零轴徘徊
        }
        
        if len(result) < 2:
            return signals
        
        # 获取最后一行数据
        last_row = result.iloc[-1]
        
        # 判断金叉和死叉
        signals['golden_cross'] = last_row.get('MACD_GOLDEN_CROSS', False)
        signals['death_cross'] = last_row.get('MACD_DEATH_CROSS', False)
        
        # 判断柱状图状态
        signals['hist_positive'] = last_row['macd_histogram'] > 0
        signals['hist_negative'] = last_row['macd_histogram'] < 0
        
        if len(result) >= 3:
            signals['hist_increasing'] = (result['macd_histogram'].iloc[-1] > result['macd_histogram'].iloc[-2] and 
                                         result['macd_histogram'].iloc[-2] > result['macd_histogram'].iloc[-3])
            signals['hist_decreasing'] = (result['macd_histogram'].iloc[-1] < result['macd_histogram'].iloc[-2] and 
                                         result['macd_histogram'].iloc[-2] < result['macd_histogram'].iloc[-3])
        
        # 判断零轴穿越
        signals['zero_cross_up'] = last_row.get('MACD_ZERO_CROSS_ABOVE', False)
        signals['zero_cross_down'] = last_row.get('MACD_ZERO_CROSS_BELOW', False)
        
        # 判断背离
        signals['bullish_divergence'] = last_row.get('MACD_BULLISH_DIVERGENCE', False)
        signals['bearish_divergence'] = last_row.get('MACD_BEARISH_DIVERGENCE', False)
        
        # 判断双顶双底
        signals['double_bottom'] = last_row.get('MACD_DOUBLE_BOTTOM', False)
        signals['double_top'] = last_row.get('MACD_DOUBLE_TOP', False)
        
        # 判断三重穿越
        signals['triple_cross'] = last_row.get('MACD_TRIPLE_CROSS', False)
        
        # 判断零轴徘徊
        signals['zero_hesitation'] = last_row.get('MACD_ZERO_LINE_HESITATION', False)
        
        return signals
        
    def get_score(self, data: pd.DataFrame) -> float:
        """
        计算MACD得分
        
        Args:
            data: 包含MACD指标的DataFrame
            
        Returns:
            MACD得分 (0-1)
        """
        if len(data) < 2:
            return 0.0
            
        score = 0.0
        signals = self.get_signals(data)
        
        # 根据信号计算得分
        if signals['golden_cross']:
            score += 0.4
        elif signals['death_cross']:
            score -= 0.4
            
        if signals['hist_positive']:
            score += 0.2
        elif signals['hist_negative']:
            score -= 0.2
            
        if signals['hist_increasing']:
            score += 0.2
        elif signals['hist_decreasing']:
            score -= 0.2
            
        # 归一化得分到0-1范围
        return max(0.0, min(1.0, (score + 1.0) / 2.0))
    
    def analyze_pattern(self, pattern_id: str, data: pd.DataFrame) -> List[Dict]:
        """
        分析指定形态
        
        Args:
            pattern_id: 形态ID
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[Dict]: 形态识别结果列表
        """
        if pattern_id not in self.patterns:
            raise ValueError(f"不支持的MACD形态: {pattern_id}")
            
        return self.patterns[pattern_id]['analyzer'](data)
    
    def _analyze_golden_cross(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析MACD金叉形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[Dict]: 金叉形态识别结果列表
        """
        dif, dea, macd = self.calculate(data['close'])
        results = []
        
        # 寻找金叉
        for i in range(1, len(data)):
            if dif.iloc[i-1] < dea.iloc[i-1] and dif.iloc[i] > dea.iloc[i]:
                # 计算形态强度
                strength = min(1.0, abs(dif.iloc[i] - dea.iloc[i]) / abs(dif.iloc[i-1] - dea.iloc[i-1]))
                
                results.append({
                    'date': data.index[i],
                    'pattern': 'macd_golden_cross',
                    'strength': strength,
                    'price': data['close'].iloc[i],
                    'dif': dif.iloc[i],
                    'dea': dea.iloc[i],
                    'macd': macd.iloc[i]
                })
        
        return results
    
    def _analyze_death_cross(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析MACD死叉形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[Dict]: 死叉形态识别结果列表
        """
        dif, dea, macd = self.calculate(data['close'])
        results = []
        
        # 寻找死叉
        for i in range(1, len(data)):
            if dif.iloc[i-1] > dea.iloc[i-1] and dif.iloc[i] < dea.iloc[i]:
                # 计算形态强度
                strength = min(1.0, abs(dif.iloc[i] - dea.iloc[i]) / abs(dif.iloc[i-1] - dea.iloc[i-1]))
                
                results.append({
                    'date': data.index[i],
                    'pattern': 'macd_death_cross',
                    'strength': strength,
                    'price': data['close'].iloc[i],
                    'dif': dif.iloc[i],
                    'dea': dea.iloc[i],
                    'macd': macd.iloc[i]
                })
        
        return results
    
    def _analyze_divergence(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析MACD背离形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[Dict]: 背离形态识别结果列表
        """
        dif, dea, macd = self.calculate(data['close'])
        results = []
        
        # 寻找顶背离
        for i in range(2, len(data)):
            # 价格创新高但MACD未创新高
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i-1] > data['high'].iloc[i-2] and
                macd.iloc[i] < macd.iloc[i-1] and 
                macd.iloc[i-1] < macd.iloc[i-2]):
                
                # 计算形态强度
                price_change = (data['high'].iloc[i] - data['high'].iloc[i-2]) / data['high'].iloc[i-2]
                macd_change = (macd.iloc[i] - macd.iloc[i-2]) / abs(macd.iloc[i-2])
                strength = min(1.0, abs(price_change * macd_change))
                
                results.append({
                    'date': data.index[i],
                    'pattern': 'macd_divergence',
                    'type': 'top',
                    'strength': strength,
                    'price': data['high'].iloc[i],
                    'macd': macd.iloc[i]
                })
            
            # 价格创新低但MACD未创新低
            elif (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                  data['low'].iloc[i-1] < data['low'].iloc[i-2] and
                  macd.iloc[i] > macd.iloc[i-1] and 
                  macd.iloc[i-1] > macd.iloc[i-2]):
                
                # 计算形态强度
                price_change = (data['low'].iloc[i] - data['low'].iloc[i-2]) / data['low'].iloc[i-2]
                macd_change = (macd.iloc[i] - macd.iloc[i-2]) / abs(macd.iloc[i-2])
                strength = min(1.0, abs(price_change * macd_change))
                
                results.append({
                    'date': data.index[i],
                    'pattern': 'macd_divergence',
                    'type': 'bottom',
                    'strength': strength,
                    'price': data['low'].iloc[i],
                    'macd': macd.iloc[i]
                })
        
        return results
    
    def _analyze_double_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析MACD双顶双底形态
        
        Args:
            data: 包含MACD指标的DataFrame
            
        Returns:
            List[Dict]: 双顶双底形态识别结果列表
        """
        # 确保已计算MACD
        if not self.has_result():
            self.calculate(data)
            
        result = self._result if self._result is not None else data
        results = []
        
        # 检查双顶
        double_tops = result[result['MACD_DOUBLE_TOP']].index
        for idx in double_tops:
            i = result.index.get_loc(idx)
            if i >= 2:
                results.append({
                    'date': idx,
                    'pattern': 'macd_double_top',
                    'strength': 0.8,  # 固定强度
                    'price': data['close'].iloc[i],
                    'macd': result['macd_line'].iloc[i],
                    'description': 'MACD双顶形态，可能的看跌信号'
                })
        
        # 检查双底
        double_bottoms = result[result['MACD_DOUBLE_BOTTOM']].index
        for idx in double_bottoms:
            i = result.index.get_loc(idx)
            if i >= 2:
                results.append({
                    'date': idx,
                    'pattern': 'macd_double_bottom',
                    'strength': 0.8,  # 固定强度
                    'price': data['close'].iloc[i],
                    'macd': result['macd_line'].iloc[i],
                    'description': 'MACD双底形态，可能的看涨信号'
                })
                
        return results