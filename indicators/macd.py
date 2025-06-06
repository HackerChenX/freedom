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
    
    def _register_macd_patterns(self):
        """
        注册MACD形态
        """
        # 如果已经注册过形态，直接返回，避免重复注册
        if self._registered_patterns:
            return
            
        # 标记为已注册
        self._registered_patterns = True
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
            default_strength="MEDIUM",
            score_impact=-5.0
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
                
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            包含MACD指标的DataFrame
        """
        # 处理输入参数
        params = self._parameters.copy()
        params.update(kwargs)
        
        price_col = params.get('price_col', 'close')
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        
        # 确保数据包含价格列
        self.ensure_columns(data, [price_col])
        
        # 复制输入数据
        df = data.copy()
        
        # 计算EMA
        df['ema_fast'] = df[price_col].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # 计算MACD线
        df['macd'] = df['ema_fast'] - df['ema_slow']
        
        # 计算信号线
        df['signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        df['hist'] = df['macd'] - df['signal']
        
        # 计算平滑后的MACD（如果启用）
        if params.get('smoothing_enabled', False):
            smoothing_period = params.get('smoothing_period', 3)
            df['macd_smooth'] = df['macd'].rolling(window=smoothing_period).mean()
            df['signal_smooth'] = df['signal'].rolling(window=smoothing_period).mean()
            df['hist_smooth'] = df['hist'].rolling(window=smoothing_period).mean()
        
        # 添加增强功能
        df = self._add_enhanced_features(df, data)
        
        # 保存结果
        self._result = df
        
        return df
    
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
            result = data if 'macd' in data.columns else self._result
        
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
        signals['golden_cross'] = last_row.get('golden_cross', False)
        signals['death_cross'] = last_row.get('death_cross', False)
        
        # 判断柱状图状态
        signals['hist_positive'] = last_row['hist'] > 0
        signals['hist_negative'] = last_row['hist'] < 0
        
        if len(result) >= 3:
            signals['hist_increasing'] = (result['hist'].iloc[-1] > result['hist'].iloc[-2] and 
                                         result['hist'].iloc[-2] > result['hist'].iloc[-3])
            signals['hist_decreasing'] = (result['hist'].iloc[-1] < result['hist'].iloc[-2] and 
                                         result['hist'].iloc[-2] < result['hist'].iloc[-3])
        
        # 判断零轴穿越
        signals['zero_cross_up'] = last_row.get('macd_cross_zero_up', False)
        signals['zero_cross_down'] = last_row.get('macd_cross_zero_down', False)
        
        # 判断背离
        signals['bullish_divergence'] = last_row.get('bullish_divergence', False)
        signals['bearish_divergence'] = last_row.get('bearish_divergence', False)
        
        # 判断双顶双底
        signals['double_bottom'] = last_row.get('double_bottom', False)
        signals['double_top'] = last_row.get('double_top', False)
        
        # 判断三重穿越
        signals['triple_cross'] = last_row.get('triple_cross', False)
        
        # 判断零轴徘徊
        signals['zero_hesitation'] = last_row.get('macd_zero_hesitation', False)
        
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
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型
        """
        return self.name.upper()
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算MACD和增强特征
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 获取计算结果
        result = self._result if self._result is not None else data
        
        # 初始化评分为50（中性）
        score = pd.Series(50.0, index=result.index)
        
        # 获取MACD相关数据
        macd = result['macd']
        signal = result['signal']
        hist = result['hist']
        
        # MACD金叉评分
        golden_cross = result.get('golden_cross', self.crossover(macd, signal))
        score += golden_cross * 20  # 金叉加20分
        
        # MACD死叉评分
        death_cross = result.get('death_cross', self.crossunder(macd, signal))
        score -= death_cross * 20  # 死叉减20分
        
        # MACD零轴上方评分
        score += (macd > 0) * 10  # MACD在零轴上方加10分
        score -= (macd < 0) * 10  # MACD在零轴下方减10分
        
        # 柱状图评分
        score += (hist > 0) * 5  # 柱状图为正加5分
        score -= (hist < 0) * 5  # 柱状图为负减5分
        
        # 柱状图变化趋势评分
        hist_increasing = hist > hist.shift(1)
        score += hist_increasing * 5  # 柱状图增加加5分
        score -= (~hist_increasing) * 5  # 柱状图减少减5分
        
        # 零轴穿越评分
        zero_cross_up = result.get('macd_cross_zero_up', (macd > 0) & (macd.shift(1) <= 0))
        zero_cross_down = result.get('macd_cross_zero_down', (macd < 0) & (macd.shift(1) >= 0))
        score += zero_cross_up * 15  # 零轴向上穿越加15分
        score -= zero_cross_down * 15  # 零轴向下穿越减15分
        
        # 背离评分
        bullish_divergence = result.get('bullish_divergence', False)
        bearish_divergence = result.get('bearish_divergence', False)
        score += bullish_divergence * 25  # 底背离加25分
        score -= bearish_divergence * 25  # 顶背离减25分
        
        # 双顶双底评分
        double_bottom = result.get('double_bottom', False)
        double_top = result.get('double_top', False)
        score += double_bottom * 22  # 双底加22分
        score -= double_top * 22  # 双顶减22分
        
        # 三重穿越评分（表示不稳定）
        triple_cross = result.get('triple_cross', False)
        score -= triple_cross * 5  # 三重穿越减5分
        
        # 增加背离强度的影响
        if 'divergence_strength' in result.columns:
            score += result['divergence_strength'] * 10  # 背离强度影响
        
        # 增加MACD趋势和强度的影响
        if 'macd_trend' in result.columns and 'macd_strength' in result.columns:
            score += result['macd_trend'] * result['macd_strength'] * 5  # 趋势强度影响
        
        # 限制评分范围在0-100之间
        score = score.clip(0, 100)
        
        return score
    
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
        histogram_threshold = self._parameters.get('histogram_threshold', 0.0)
        zero_line_sensitivity = self._parameters.get('zero_line_sensitivity', 0.001)
        
        # 计算MACD柱状图变化率
        result['hist_change'] = result['hist'].diff()
        
        # 记录MACD柱状图由负变正和由正变负的变化点
        result['hist_positive'] = result['hist'] > 0
        result['hist_positive_change'] = result['hist_positive'].diff()
        result['hist_turn_positive'] = result['hist_positive_change'] == 1
        result['hist_turn_negative'] = result['hist_positive_change'] == -1
        
        # 计算有效的MACD柱状图变化（过滤微小变化）
        result['hist_change_valid'] = result['hist_change'].abs() > histogram_threshold
        result['hist_up_valid'] = (result['hist_change'] > 0) & result['hist_change_valid']
        result['hist_down_valid'] = (result['hist_change'] < 0) & result['hist_change_valid']
        
        # 连续两个柱状向上或向下变化的确认
        result['hist_up_confirm'] = (result['hist_up_valid']) & (result['hist_up_valid'].shift(1))
        result['hist_down_confirm'] = (result['hist_down_valid']) & (result['hist_down_valid'].shift(1))
        
        # 计算零轴穿越信号
        result['macd_cross_zero_up'] = (result['macd'] > 0) & (result['macd'].shift(1) <= 0)
        result['macd_cross_zero_down'] = (result['macd'] < 0) & (result['macd'].shift(1) >= 0)
        
        # 计算零轴徘徊信号
        result['macd_zero_hesitation'] = result['macd'].abs() <= zero_line_sensitivity
        
        # 计算DIF和DEA的交叉信号
        result['golden_cross'] = self.crossover(result['macd'], result['signal'])
        result['death_cross'] = self.crossunder(result['macd'], result['signal'])
        
        # 计算三重穿越信号（短期内多次交叉）
        cross_window = 10
        result['triple_cross'] = (
            result.rolling(cross_window)['golden_cross'].sum() + 
            result.rolling(cross_window)['death_cross'].sum() >= 3
        )
        
        # 计算背离信号
        result = self._detect_divergence(result, data)
        
        # 计算双顶双底形态
        result = self._detect_double_patterns(result)
        
        # 计算MACD趋势方向和强度
        result['macd_trend'] = np.sign(result['macd'] - result['macd'].shift(5))
        result['macd_strength'] = (result['macd'] - result['signal']).abs() / data[price_col].rolling(20).std()
        
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
        threshold = self._parameters.get('divergence_threshold', 0.05)
        
        # 初始化背离信号列
        result['bullish_divergence'] = False  # 价格创新低，但MACD未创新低（看涨）
        result['bearish_divergence'] = False  # 价格创新高，但MACD未创新高（看跌）
        result['divergence_strength'] = 0.0  # 背离强度
        
        # 计算局部最高点和最低点
        price = data[price_col].values
        macd = result['macd'].values
        
        # 至少需要2*window+1个数据点才能进行背离检测
        if len(price) < 2 * window + 1:
            return result
        
        # 遍历每个窗口检测背离
        for i in range(window, len(price) - window):
            # 获取当前窗口和前一个窗口
            current_window_price = price[i-window:i+window+1]
            current_window_macd = macd[i-window:i+window+1]
            
            # 寻找当前窗口的价格最高点和最低点
            price_high_idx = np.nanargmax(current_window_price)
            price_low_idx = np.nanargmin(current_window_price)
            
            # 寻找当前窗口的MACD最高点和最低点
            macd_high_idx = np.nanargmax(current_window_macd)
            macd_low_idx = np.nanargmin(current_window_macd)
            
            # 检测顶背离：价格创新高，但MACD未创新高
            if price_high_idx == window and macd_high_idx != window:
                # 确认是否为有效背离（价格变化幅度超过阈值）
                price_change = (current_window_price[price_high_idx] - current_window_price[0]) / current_window_price[0]
                if price_change > threshold:
                    result.iloc[i, result.columns.get_loc('bearish_divergence')] = True
                    
                    # 计算背离强度
                    divergence_strength = price_change * (1 - macd[i] / macd[i-window])
                    result.iloc[i, result.columns.get_loc('divergence_strength')] = -divergence_strength
            
            # 检测底背离：价格创新低，但MACD未创新低
            if price_low_idx == window and macd_low_idx != window:
                # 确认是否为有效背离（价格变化幅度超过阈值）
                price_change = (current_window_price[0] - current_window_price[price_low_idx]) / current_window_price[0]
                if price_change > threshold:
                    result.iloc[i, result.columns.get_loc('bullish_divergence')] = True
                    
                    # 计算背离强度
                    divergence_strength = price_change * (1 - macd[i-window] / macd[i])
                    result.iloc[i, result.columns.get_loc('divergence_strength')] = divergence_strength
        
        return result
    
    def _detect_double_patterns(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        检测双顶双底形态
        
        Args:
            result: MACD计算结果
            
        Returns:
            pd.DataFrame: 添加了双顶双底信号的结果
        """
        # 初始化双顶双底信号列
        result['double_top'] = False
        result['double_bottom'] = False
        
        # 获取MACD数据
        macd = result['macd'].values
        
        # 至少需要30个数据点才能进行双顶双底检测
        if len(macd) < 30:
            return result
        
        # 计算局部极值
        for i in range(15, len(macd) - 5):
            # 获取前30个点
            window = macd[i-15:i+5]
            
            # 寻找窗口内的局部极值
            peaks = []
            troughs = []
            
            for j in range(1, len(window) - 1):
                # 局部最大值
                if window[j] > window[j-1] and window[j] > window[j+1]:
                    peaks.append((j, window[j]))
                # 局部最小值
                elif window[j] < window[j-1] and window[j] < window[j+1]:
                    troughs.append((j, window[j]))
            
            # 检测双顶：两个相近的高点之间有一个低点
            if len(peaks) >= 2:
                # 按位置排序
                peaks.sort(key=lambda x: x[0])
                
                # 检查相邻的高点
                for k in range(len(peaks) - 1):
                    peak1_idx, peak1_val = peaks[k]
                    peak2_idx, peak2_val = peaks[k+1]
                    
                    # 高点之间的距离适中（不太近也不太远）
                    if 3 <= peak2_idx - peak1_idx <= 10:
                        # 两个高点的值相近
                        if abs(peak1_val - peak2_val) / max(abs(peak1_val), abs(peak2_val)) < 0.2:
                            # 高点之间有一个低点
                            has_trough_between = False
                            for trough_idx, _ in troughs:
                                if peak1_idx < trough_idx < peak2_idx:
                                    has_trough_between = True
                                    break
                            
                            if has_trough_between:
                                result.iloc[i, result.columns.get_loc('double_top')] = True
                                break
            
            # 检测双底：两个相近的低点之间有一个高点
            if len(troughs) >= 2:
                # 按位置排序
                troughs.sort(key=lambda x: x[0])
                
                # 检查相邻的低点
                for k in range(len(troughs) - 1):
                    trough1_idx, trough1_val = troughs[k]
                    trough2_idx, trough2_val = troughs[k+1]
                    
                    # 低点之间的距离适中（不太近也不太远）
                    if 3 <= trough2_idx - trough1_idx <= 10:
                        # 两个低点的值相近
                        if abs(trough1_val - trough2_val) / max(abs(trough1_val), abs(trough2_val)) < 0.2:
                            # 低点之间有一个高点
                            has_peak_between = False
                            for peak_idx, _ in peaks:
                                if trough1_idx < peak_idx < trough2_idx:
                                    has_peak_between = True
                                    break
                            
                            if has_peak_between:
                                result.iloc[i, result.columns.get_loc('double_bottom')] = True
                                break
        
        return result
        
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
        double_tops = result[result['double_top']].index
        for idx in double_tops:
            i = result.index.get_loc(idx)
            if i >= 2:
                results.append({
                    'date': idx,
                    'pattern': 'macd_double_top',
                    'strength': 0.8,  # 固定强度
                    'price': data['close'].iloc[i],
                    'macd': result['macd'].iloc[i],
                    'description': 'MACD双顶形态，可能的看跌信号'
                })
        
        # 检查双底
        double_bottoms = result[result['double_bottom']].index
        for idx in double_bottoms:
            i = result.index.get_loc(idx)
            if i >= 2:
                results.append({
                    'date': idx,
                    'pattern': 'macd_double_bottom',
                    'strength': 0.8,  # 固定强度
                    'price': data['close'].iloc[i],
                    'macd': result['macd'].iloc[i],
                    'description': 'MACD双底形态，可能的看涨信号'
                })
                
        return results