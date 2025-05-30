import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List

from indicators.base_indicator import BaseIndicator
from utils.technical_utils import find_peaks_and_troughs


class EnhancedCCI(BaseIndicator):
    """
    增强型CCI(商品通道指数)指标
    
    具有以下增强特性:
    1. 自适应周期设计: 根据市场波动率自动调整周期
    2. 区间穿梭动态评估: 评估CCI在不同区间穿梭的质量和可靠性
    3. 周期协同分析: 结合多周期CCI指标增强信号可靠性
    4. 形态识别: 识别CCI形成的关键形态
    5. 市场环境自适应: 根据市场环境动态调整评分标准
    """

    # CCI标准区间定义
    EXTREME_OVERBOUGHT = 200  # 极度超买
    OVERBOUGHT = 100  # 超买
    NEUTRAL_HIGH = 0  # 中性偏多
    NEUTRAL_LOW = 0  # 中性偏空
    OVERSOLD = -100  # 超卖
    EXTREME_OVERSOLD = -200  # 极度超卖

    def __init__(self, period: int = 20, factor: float = 0.015, 
                 secondary_period: int = 40, adaptive: bool = True):
        """
        初始化增强型CCI指标
        
        Args:
            period (int): CCI计算周期
            factor (float): CCI计算因子，标准为0.015
            secondary_period (int): 二级CCI周期，用于周期协同分析
            adaptive (bool): 是否启用自适应周期
        """
        super().__init__()
        self.base_period = period
        self.factor = factor
        self.secondary_period = secondary_period
        self.adaptive = adaptive
        self.market_environment = "normal"
        
        # 实际使用周期（可能会根据自适应算法调整）
        self.current_period = period
        
        # 内部数据
        self._result = None
        self._price_data = None
        self._multi_period_result = None

    def set_market_environment(self, environment: str) -> None:
        """
        设置市场环境
        
        Args:
            environment (str): 市场环境类型 ('bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal')
        """
        valid_environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境类型: {environment}。有效类型: {valid_environments}")
        
        self.market_environment = environment
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算CCI指标
        
        Args:
            data (pd.DataFrame): 包含价格数据的DataFrame (必须包含'high', 'low', 'close'列)
            
        Returns:
            pd.DataFrame: 包含CCI指标结果的DataFrame
        """
        # 检查数据是否有效
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据必须包含'{col}'列")
        
        # 保存价格数据供后续使用
        self._price_data = data['close'].copy()
        
        # 如果是自适应模式，调整周期
        if self.adaptive:
            self.current_period = self._adapt_period(data)
        else:
            self.current_period = self.base_period
        
        # 计算主要CCI
        cci = self._calculate_cci(data, self.current_period)
        
        # 计算二级CCI（用于周期协同分析）
        cci_secondary = self._calculate_cci(data, self.secondary_period)
        
        # 计算CCI的均线
        cci_ma5 = cci.rolling(window=5).mean()
        cci_ma10 = cci.rolling(window=10).mean()
        cci_ma20 = cci.rolling(window=20).mean()
        
        # 计算CCI斜率
        cci_slope = (cci - cci.shift(3)) / 3
        
        # 计算CCI波动性
        cci_volatility = cci.rolling(window=10).std()
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            'cci': cci,
            'cci_secondary': cci_secondary,
            'cci_ma5': cci_ma5,
            'cci_ma10': cci_ma10,
            'cci_ma20': cci_ma20,
            'cci_slope': cci_slope,
            'cci_volatility': cci_volatility
        }, index=data.index)
        
        # 添加CCI状态分类
        result['state'] = self._classify_cci_state(cci)
        
        self._result = result
        
        # 计算多周期CCI结果
        self._calculate_multi_period_cci(data)
        
        return result
    
    def _calculate_cci(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        使用特定周期计算CCI
        
        Args:
            data (pd.DataFrame): 价格数据
            period (int): 计算周期
            
        Returns:
            pd.Series: CCI值
        """
        # 计算典型价格 (TP)
        tp = (data['high'] + data['low'] + data['close']) / 3
        
        # 计算移动平均
        tp_ma = tp.rolling(window=period).mean()
        
        # 计算平均偏差
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        # 计算CCI
        cci = (tp - tp_ma) / (self.factor * mad)
        
        return cci
    
    def _adapt_period(self, data: pd.DataFrame) -> int:
        """
        根据市场波动性自适应调整CCI周期
        
        Args:
            data (pd.DataFrame): 价格数据
            
        Returns:
            int: 自适应调整后的周期
        """
        # 使用ATR计算市场波动性
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        # 计算真实波幅 (TR)
        tr1 = high - low
        tr2 = np.abs(high - close)
        tr3 = np.abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=20).mean()
        
        # 计算近期ATR与长期ATR的比值
        recent_atr = atr.iloc[-10:].mean() if len(atr) >= 10 else atr.mean()
        long_term_atr = atr.iloc[-60:].mean() if len(atr) >= 60 else atr.mean()
        
        if pd.isna(recent_atr) or pd.isna(long_term_atr) or long_term_atr == 0:
            return self.base_period
        
        volatility_ratio = recent_atr / long_term_atr
        
        # 根据波动性调整周期
        if volatility_ratio > 1.5:  # 高波动性
            # 高波动市场使用较短周期
            return max(int(self.base_period * 0.7), 10)
        elif volatility_ratio < 0.7:  # 低波动性
            # 低波动市场使用较长周期
            return min(int(self.base_period * 1.3), 30)
        else:  # 正常波动性
            return self.base_period
    
    def _classify_cci_state(self, cci: pd.Series) -> pd.Series:
        """
        根据CCI值分类状态
        
        Args:
            cci (pd.Series): CCI值
            
        Returns:
            pd.Series: CCI状态分类
        """
        conditions = [
            cci > self.EXTREME_OVERBOUGHT,
            (cci > self.OVERBOUGHT) & (cci <= self.EXTREME_OVERBOUGHT),
            (cci > self.NEUTRAL_HIGH) & (cci <= self.OVERBOUGHT),
            (cci > self.NEUTRAL_LOW) & (cci <= self.NEUTRAL_HIGH),
            (cci > self.EXTREME_OVERSOLD) & (cci <= self.OVERSOLD),
            cci <= self.EXTREME_OVERSOLD
        ]
        
        choices = [
            'extreme_overbought',
            'overbought',
            'neutral_high',
            'neutral_low',
            'oversold',
            'extreme_oversold'
        ]
        
        return pd.Series(np.select(conditions, choices, default='neutral'), index=cci.index)
    
    def _calculate_multi_period_cci(self, data: pd.DataFrame) -> None:
        """
        计算多周期CCI用于周期协同分析
        
        Args:
            data (pd.DataFrame): 价格数据
        """
        # 计算不同周期的CCI
        periods = [
            max(int(self.current_period / 2), 5),  # 短周期
            self.current_period,                   # 中周期
            self.current_period * 2                # 长周期
        ]
        
        multi_period = {}
        for period in periods:
            multi_period[f'cci_{period}'] = self._calculate_cci(data, period)
        
        self._multi_period_result = pd.DataFrame(multi_period, index=data.index)
    
    def analyze_crossovers(self) -> pd.DataFrame:
        """
        分析CCI的关键交叉点
        
        Returns:
            pd.DataFrame: 包含交叉分析结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
        
        cci = self._result['cci']
        
        # 创建结果DataFrame
        crossovers = pd.DataFrame(index=cci.index)
        
        # 零轴交叉
        crossovers['zero_cross_up'] = (cci > 0) & (cci.shift(1) <= 0)
        crossovers['zero_cross_down'] = (cci < 0) & (cci.shift(1) >= 0)
        
        # 超买/超卖区域交叉
        crossovers['overbought_enter'] = (cci > self.OVERBOUGHT) & (cci.shift(1) <= self.OVERBOUGHT)
        crossovers['overbought_exit'] = (cci < self.OVERBOUGHT) & (cci.shift(1) >= self.OVERBOUGHT)
        crossovers['oversold_enter'] = (cci < self.OVERSOLD) & (cci.shift(1) >= self.OVERSOLD)
        crossovers['oversold_exit'] = (cci > self.OVERSOLD) & (cci.shift(1) <= self.OVERSOLD)
        
        # 极端区域交叉
        crossovers['extreme_overbought_enter'] = (cci > self.EXTREME_OVERBOUGHT) & (cci.shift(1) <= self.EXTREME_OVERBOUGHT)
        crossovers['extreme_overbought_exit'] = (cci < self.EXTREME_OVERBOUGHT) & (cci.shift(1) >= self.EXTREME_OVERBOUGHT)
        crossovers['extreme_oversold_enter'] = (cci < self.EXTREME_OVERSOLD) & (cci.shift(1) >= self.EXTREME_OVERSOLD)
        crossovers['extreme_oversold_exit'] = (cci > self.EXTREME_OVERSOLD) & (cci.shift(1) <= self.EXTREME_OVERSOLD)
        
        # 均线交叉
        crossovers['ma5_cross_up'] = (cci > self._result['cci_ma5']) & (cci.shift(1) <= self._result['cci_ma5'].shift(1))
        crossovers['ma5_cross_down'] = (cci < self._result['cci_ma5']) & (cci.shift(1) >= self._result['cci_ma5'].shift(1))
        
        # 交叉质量评估
        crossovers['crossover_strength'] = self._calculate_crossover_strength(cci)
        
        return crossovers
    
    def _calculate_crossover_strength(self, cci: pd.Series) -> pd.Series:
        """
        计算交叉强度
        
        Args:
            cci (pd.Series): CCI值
            
        Returns:
            pd.Series: 交叉强度评分 (0-100)
        """
        # 创建结果序列
        strength = pd.Series(0, index=cci.index)
        
        # 计算零轴交叉强度
        zero_cross_up = (cci > 0) & (cci.shift(1) <= 0)
        zero_cross_down = (cci < 0) & (cci.shift(1) >= 0)
        
        # 计算零轴交叉角度（斜率）
        cross_slope = cci - cci.shift(1)
        
        # 零轴交叉强度评分
        for i in range(1, len(cci)):
            if zero_cross_up.iloc[i]:
                # 向上交叉强度基于斜率和前期状态
                # 1. 从超卖区域反弹强度更大
                # 2. 斜率越大强度越大
                slope = cross_slope.iloc[i]
                prev_min = cci.iloc[max(0, i-5):i].min()
                
                # 基础分30，最高100
                base_score = 30
                
                # 斜率评分（最高30分）
                slope_score = min(30, abs(slope) * 15)
                
                # 前期状态评分（最高40分）
                prev_state_score = 0
                if prev_min < self.EXTREME_OVERSOLD:
                    prev_state_score = 40  # 从极度超卖区域反弹
                elif prev_min < self.OVERSOLD:
                    prev_state_score = 30  # 从超卖区域反弹
                elif prev_min < 0:
                    prev_state_score = 20  # 从负值区域反弹
                
                total_score = base_score + slope_score + prev_state_score
                strength.iloc[i] = total_score
                
            elif zero_cross_down.iloc[i]:
                # 向下交叉强度基于斜率和前期状态
                slope = abs(cross_slope.iloc[i])
                prev_max = cci.iloc[max(0, i-5):i].max()
                
                # 基础分30，最高100
                base_score = 30
                
                # 斜率评分（最高30分）
                slope_score = min(30, slope * 15)
                
                # 前期状态评分（最高40分）
                prev_state_score = 0
                if prev_max > self.EXTREME_OVERBOUGHT:
                    prev_state_score = 40  # 从极度超买区域下跌
                elif prev_max > self.OVERBOUGHT:
                    prev_state_score = 30  # 从超买区域下跌
                elif prev_max > 0:
                    prev_state_score = 20  # 从正值区域下跌
                
                total_score = base_score + slope_score + prev_state_score
                strength.iloc[i] = total_score
        
        return strength
    
    def analyze_multi_period_synergy(self) -> pd.DataFrame:
        """
        分析多周期CCI协同性
        
        Returns:
            pd.DataFrame: 包含多周期协同分析结果的DataFrame
        """
        if self._multi_period_result is None:
            return pd.DataFrame()
        
        multi_cci = self._multi_period_result
        
        # 获取各周期CCI列名
        columns = multi_cci.columns
        
        # 创建结果DataFrame
        synergy = pd.DataFrame(index=multi_cci.index)
        
        # 方向一致性分析
        directions = {}
        for col in columns:
            directions[f'{col}_up'] = multi_cci[col] > 0
            directions[f'{col}_down'] = multi_cci[col] < 0
        
        directions_df = pd.DataFrame(directions, index=multi_cci.index)
        
        # 计算多周期方向一致性
        up_agreement = directions_df[[c for c in directions_df.columns if c.endswith('_up')]].all(axis=1)
        down_agreement = directions_df[[c for c in directions_df.columns if c.endswith('_down')]].all(axis=1)
        
        synergy['bullish_agreement'] = up_agreement
        synergy['bearish_agreement'] = down_agreement
        
        # 计算多周期斜率一致性
        slopes = {}
        for col in columns:
            slopes[f'{col}_slope'] = multi_cci[col] - multi_cci[col].shift(3)
        
        slopes_df = pd.DataFrame(slopes, index=multi_cci.index)
        
        # 计算斜率符号一致性
        slope_cols = slopes_df.columns
        rising_slopes = (slopes_df > 0).all(axis=1)
        falling_slopes = (slopes_df < 0).all(axis=1)
        
        synergy['rising_momentum'] = rising_slopes
        synergy['falling_momentum'] = falling_slopes
        
        # 计算周期排列
        # 理想的多周期排列：短周期 > 中周期 > 长周期（上升趋势）
        # 或 短周期 < 中周期 < 长周期（下降趋势）
        if len(columns) >= 3:
            short_period = columns[0]
            mid_period = columns[1]
            long_period = columns[2]
            
            synergy['bullish_alignment'] = (
                (multi_cci[short_period] > multi_cci[mid_period]) & 
                (multi_cci[mid_period] > multi_cci[long_period])
            )
            
            synergy['bearish_alignment'] = (
                (multi_cci[short_period] < multi_cci[mid_period]) & 
                (multi_cci[mid_period] < multi_cci[long_period])
            )
        
        # 计算综合协同度评分 (0-100)
        synergy['synergy_score'] = self._calculate_synergy_score(synergy)
        
        return synergy
    
    def _calculate_synergy_score(self, synergy: pd.DataFrame) -> pd.Series:
        """
        计算多周期CCI协同度评分
        
        Args:
            synergy (pd.DataFrame): 包含协同分析的DataFrame
            
        Returns:
            pd.Series: 协同度评分 (0-100)
        """
        # 基础分50分（中性）
        score = pd.Series(50, index=synergy.index)
        
        # 方向一致性评分 (±20分)
        score += np.where(synergy['bullish_agreement'], 20, 0)
        score -= np.where(synergy['bearish_agreement'], 20, 0)
        
        # 动量一致性评分 (±15分)
        score += np.where(synergy['rising_momentum'], 15, 0)
        score -= np.where(synergy['falling_momentum'], 15, 0)
        
        # 周期排列评分 (±15分)
        if 'bullish_alignment' in synergy.columns:
            score += np.where(synergy['bullish_alignment'], 15, 0)
            score -= np.where(synergy['bearish_alignment'], 15, 0)
        
        # 限制分数范围在0-100之间
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self) -> pd.DataFrame:
        """
        识别CCI形态
        
        Returns:
            pd.DataFrame: 包含形态识别结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
        
        cci = self._result['cci']
        
        # 创建形态DataFrame
        patterns = pd.DataFrame(index=cci.index)
        
        # 1. 钩子形态（Hook）：CCI在超买/超卖区域形成拐点
        # 超买区域钩子（看跌）
        overbought_hook = (
            (cci.shift(2) > self.OVERBOUGHT) &
            (cci.shift(1) > self.OVERBOUGHT) &
            (cci > self.OVERBOUGHT) &
            (cci.shift(1) > cci.shift(2)) &  # 上升
            (cci < cci.shift(1))             # 然后下降
        )
        
        # 超卖区域钩子（看涨）
        oversold_hook = (
            (cci.shift(2) < self.OVERSOLD) &
            (cci.shift(1) < self.OVERSOLD) &
            (cci < self.OVERSOLD) &
            (cci.shift(1) < cci.shift(2)) &  # 下降
            (cci > cci.shift(1))             # 然后上升
        )
        
        patterns['overbought_hook'] = overbought_hook
        patterns['oversold_hook'] = oversold_hook
        
        # 2. 零轴穿越失败（Zero-line Rejection）：尝试穿越零轴但失败
        zero_rejection_bull = (
            (cci.shift(3) < -50) &
            (cci.shift(2) < -20) &
            (cci.shift(1) > -10) & (cci.shift(1) < 10) &  # 接近零轴
            (cci < cci.shift(1)) &                       # 未能突破
            (cci > -30)                                  # 但没有深度回落
        )
        
        zero_rejection_bear = (
            (cci.shift(3) > 50) &
            (cci.shift(2) > 20) &
            (cci.shift(1) < 10) & (cci.shift(1) > -10) &  # 接近零轴
            (cci > cci.shift(1)) &                       # 未能突破
            (cci < 30)                                   # 但没有深度回落
        )
        
        patterns['zero_rejection_bull'] = zero_rejection_bull
        patterns['zero_rejection_bear'] = zero_rejection_bear
        
        # 3. 头肩顶/底形态
        # 使用find_peaks_and_troughs函数查找峰谷
        peaks = find_peaks_and_troughs(cci, window=5, peak_type='peak')
        troughs = find_peaks_and_troughs(cci, window=5, peak_type='trough')
        
        # 初始化头肩顶/底
        head_shoulders_top = pd.Series(False, index=cci.index)
        head_shoulders_bottom = pd.Series(False, index=cci.index)
        
        # 检测头肩顶
        for i in range(len(peaks)-2):
            if i+2 >= len(peaks):
                continue
                
            left_shoulder_idx = peaks[i]
            head_idx = peaks[i+1]
            right_shoulder_idx = peaks[i+2]
            
            # 确保存在两个谷（颈线）
            neck_indices = [j for j in troughs if left_shoulder_idx < j < head_idx]
            if not neck_indices:
                continue
            left_neck_idx = neck_indices[-1]
            
            neck_indices = [j for j in troughs if head_idx < j < right_shoulder_idx]
            if not neck_indices:
                continue
            right_neck_idx = neck_indices[0]
            
            # 验证形态
            if (left_shoulder_idx < head_idx < right_shoulder_idx and
                cci.iloc[head_idx] > cci.iloc[left_shoulder_idx] and
                cci.iloc[head_idx] > cci.iloc[right_shoulder_idx] and
                abs(cci.iloc[left_shoulder_idx] - cci.iloc[right_shoulder_idx]) < 0.2 * cci.iloc[head_idx] and
                abs(cci.iloc[left_neck_idx] - cci.iloc[right_neck_idx]) < 0.1 * cci.iloc[head_idx]):
                
                head_shoulders_top.iloc[right_shoulder_idx] = True
        
        # 检测头肩底
        for i in range(len(troughs)-2):
            if i+2 >= len(troughs):
                continue
                
            left_shoulder_idx = troughs[i]
            head_idx = troughs[i+1]
            right_shoulder_idx = troughs[i+2]
            
            # 确保存在两个峰（颈线）
            neck_indices = [j for j in peaks if left_shoulder_idx < j < head_idx]
            if not neck_indices:
                continue
            left_neck_idx = neck_indices[-1]
            
            neck_indices = [j for j in peaks if head_idx < j < right_shoulder_idx]
            if not neck_indices:
                continue
            right_neck_idx = neck_indices[0]
            
            # 验证形态
            if (left_shoulder_idx < head_idx < right_shoulder_idx and
                cci.iloc[head_idx] < cci.iloc[left_shoulder_idx] and
                cci.iloc[head_idx] < cci.iloc[right_shoulder_idx] and
                abs(cci.iloc[left_shoulder_idx] - cci.iloc[right_shoulder_idx]) < 0.2 * abs(cci.iloc[head_idx]) and
                abs(cci.iloc[left_neck_idx] - cci.iloc[right_neck_idx]) < 0.1 * abs(cci.iloc[head_idx])):
                
                head_shoulders_bottom.iloc[right_shoulder_idx] = True
        
        patterns['head_shoulders_top'] = head_shoulders_top
        patterns['head_shoulders_bottom'] = head_shoulders_bottom
        
        # 4. 平台突破
        # 寻找CCI在较窄范围内波动然后突破的情况
        for i in range(20, len(cci)):
            # 检查前10个周期的波动范围
            range_window = cci.iloc[i-10:i]
            range_high = range_window.max()
            range_low = range_window.min()
            range_width = range_high - range_low
            
            # 判断是否为窄幅震荡
            is_narrow_range = range_width < self._result['cci_volatility'].iloc[i] * 0.5
            
            if is_narrow_range:
                # 向上突破
                if cci.iloc[i] > range_high + range_width * 0.3:
                    patterns.loc[cci.index[i], 'platform_breakout_up'] = True
                # 向下突破
                elif cci.iloc[i] < range_low - range_width * 0.3:
                    patterns.loc[cci.index[i], 'platform_breakout_down'] = True
                else:
                    patterns.loc[cci.index[i], 'platform_breakout_up'] = False
                    patterns.loc[cci.index[i], 'platform_breakout_down'] = False
            else:
                patterns.loc[cci.index[i], 'platform_breakout_up'] = False
                patterns.loc[cci.index[i], 'platform_breakout_down'] = False
        
        # 5. 背离
        patterns['bullish_divergence'] = self._detect_divergence(bullish=True)
        patterns['bearish_divergence'] = self._detect_divergence(bullish=False)
        
        return patterns
    
    def _detect_divergence(self, bullish: bool = True) -> pd.Series:
        """
        检测CCI与价格之间的背离
        
        Args:
            bullish (bool): 是否检测看涨背离
            
        Returns:
            pd.Series: 背离检测结果
        """
        if self._result is None or self._price_data is None:
            return pd.Series()
        
        cci = self._result['cci']
        price = self._price_data
        
        # 寻找CCI和价格的高点和低点
        if bullish:
            # 看涨背离：价格创新低但CCI未创新低
            price_lows = find_peaks_and_troughs(price, window=10, peak_type='trough')
            cci_lows = find_peaks_and_troughs(cci, window=10, peak_type='trough')
        else:
            # 看跌背离：价格创新高但CCI未创新高
            price_highs = find_peaks_and_troughs(price, window=10, peak_type='peak')
            cci_highs = find_peaks_and_troughs(cci, window=10, peak_type='peak')
        
        divergence = pd.Series(False, index=cci.index)
        
        if bullish:
            for i in range(1, len(price_lows)):
                if i >= len(price) or i >= len(price_lows):
                    continue
                    
                current_idx = price_lows[i]
                prev_idx = price_lows[i-1]
                
                # 检查价格是否创新低
                if price.iloc[current_idx] < price.iloc[prev_idx]:
                    # 寻找对应的CCI低点
                    # 在价格低点附近查找CCI低点
                    window_start = max(0, current_idx - 5)
                    window_end = min(len(cci), current_idx + 5)
                    window_indices = list(range(window_start, window_end))
                    
                    cci_low_indices = [idx for idx in cci_lows if idx in window_indices]
                    
                    if cci_low_indices:
                        cci_current_idx = cci_low_indices[-1]
                        
                        # 寻找前一个对应的CCI低点
                        prev_window_start = max(0, prev_idx - 5)
                        prev_window_end = min(len(cci), prev_idx + 5)
                        prev_window_indices = list(range(prev_window_start, prev_window_end))
                        
                        cci_prev_indices = [idx for idx in cci_lows if idx in prev_window_indices]
                        
                        if cci_prev_indices:
                            cci_prev_idx = cci_prev_indices[-1]
                            
                            # 检查CCI是否未创新低
                            if cci.iloc[cci_current_idx] > cci.iloc[cci_prev_idx]:
                                divergence.iloc[current_idx] = True
        else:
            for i in range(1, len(price_highs)):
                if i >= len(price) or i >= len(price_highs):
                    continue
                    
                current_idx = price_highs[i]
                prev_idx = price_highs[i-1]
                
                # 检查价格是否创新高
                if price.iloc[current_idx] > price.iloc[prev_idx]:
                    # 寻找对应的CCI高点
                    # 在价格高点附近查找CCI高点
                    window_start = max(0, current_idx - 5)
                    window_end = min(len(cci), current_idx + 5)
                    window_indices = list(range(window_start, window_end))
                    
                    cci_high_indices = [idx for idx in cci_highs if idx in window_indices]
                    
                    if cci_high_indices:
                        cci_current_idx = cci_high_indices[-1]
                        
                        # 寻找前一个对应的CCI高点
                        prev_window_start = max(0, prev_idx - 5)
                        prev_window_end = min(len(cci), prev_idx + 5)
                        prev_window_indices = list(range(prev_window_start, prev_window_end))
                        
                        cci_prev_indices = [idx for idx in cci_highs if idx in prev_window_indices]
                        
                        if cci_prev_indices:
                            cci_prev_idx = cci_prev_indices[-1]
                            
                            # 检查CCI是否未创新高
                            if cci.iloc[cci_current_idx] < cci.iloc[cci_prev_idx]:
                                divergence.iloc[current_idx] = True
        
        return divergence
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data (pd.DataFrame): 价格数据
            
        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        if self._result is None:
            self.calculate(data)
            
        if self._result is None:
            return pd.DataFrame()
        
        # 获取CCI数据
        cci = self._result['cci']
        
        # 分析交叉
        crossovers = self.analyze_crossovers()
        
        # 识别形态
        patterns = self.identify_patterns()
        
        # 分析多周期协同
        synergy = self.analyze_multi_period_synergy()
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 基本信号
        signals['zero_cross_up'] = crossovers['zero_cross_up']
        signals['zero_cross_down'] = crossovers['zero_cross_down']
        signals['oversold_exit'] = crossovers['oversold_exit']
        signals['overbought_exit'] = crossovers['overbought_exit']
        
        # 形态信号
        if 'oversold_hook' in patterns.columns:
            signals['oversold_hook'] = patterns['oversold_hook']
            signals['overbought_hook'] = patterns['overbought_hook']
            signals['bullish_divergence'] = patterns['bullish_divergence']
            signals['bearish_divergence'] = patterns['bearish_divergence']
            signals['head_shoulders_bottom'] = patterns['head_shoulders_bottom']
            signals['head_shoulders_top'] = patterns['head_shoulders_top']
            signals['platform_breakout_up'] = patterns.get('platform_breakout_up', pd.Series(False, index=data.index))
            signals['platform_breakout_down'] = patterns.get('platform_breakout_down', pd.Series(False, index=data.index))
        
        # 多周期协同信号
        if 'bullish_agreement' in synergy.columns:
            signals['bullish_agreement'] = synergy['bullish_agreement']
            signals['bearish_agreement'] = synergy['bearish_agreement']
            signals['bullish_alignment'] = synergy.get('bullish_alignment', pd.Series(False, index=data.index))
            signals['bearish_alignment'] = synergy.get('bearish_alignment', pd.Series(False, index=data.index))
        
        # 生成买卖信号
        signals['buy_signal'] = (
            signals['zero_cross_up'] | 
            signals['oversold_exit'] | 
            signals.get('oversold_hook', pd.Series(False, index=data.index)) | 
            signals.get('bullish_divergence', pd.Series(False, index=data.index)) | 
            signals.get('head_shoulders_bottom', pd.Series(False, index=data.index)) |
            signals.get('platform_breakout_up', pd.Series(False, index=data.index)) |
            (signals.get('bullish_agreement', pd.Series(False, index=data.index)) & 
             signals.get('bullish_alignment', pd.Series(False, index=data.index)))
        )
        
        signals['sell_signal'] = (
            signals['zero_cross_down'] | 
            signals['overbought_exit'] | 
            signals.get('overbought_hook', pd.Series(False, index=data.index)) | 
            signals.get('bearish_divergence', pd.Series(False, index=data.index)) | 
            signals.get('head_shoulders_top', pd.Series(False, index=data.index)) |
            signals.get('platform_breakout_down', pd.Series(False, index=data.index)) |
            (signals.get('bearish_agreement', pd.Series(False, index=data.index)) & 
             signals.get('bearish_alignment', pd.Series(False, index=data.index)))
        )
        
        # 计算信号分数
        signals['score'] = self.calculate_score(data)
        
        # 生成信号描述
        signals['signal_type'] = pd.Series('', index=signals.index)
        signals['signal_desc'] = pd.Series('', index=signals.index)
        signals['confidence'] = pd.Series(50, index=signals.index)
        
        # 设置信号类型和描述
        signals.loc[signals['zero_cross_up'], 'signal_type'] = 'CCI零轴上穿'
        signals.loc[signals['zero_cross_up'], 'signal_desc'] = 'CCI上穿零轴，看涨信号'
        signals.loc[signals['zero_cross_up'], 'confidence'] = 60
        
        signals.loc[signals['zero_cross_down'], 'signal_type'] = 'CCI零轴下穿'
        signals.loc[signals['zero_cross_down'], 'signal_desc'] = 'CCI下穿零轴，看跌信号'
        signals.loc[signals['zero_cross_down'], 'confidence'] = 60
        
        signals.loc[signals['oversold_exit'], 'signal_type'] = 'CCI超卖区域回升'
        signals.loc[signals['oversold_exit'], 'signal_desc'] = 'CCI从超卖区域回升，看涨信号'
        signals.loc[signals['oversold_exit'], 'confidence'] = 65
        
        signals.loc[signals['overbought_exit'], 'signal_type'] = 'CCI超买区域回落'
        signals.loc[signals['overbought_exit'], 'signal_desc'] = 'CCI从超买区域回落，看跌信号'
        signals.loc[signals['overbought_exit'], 'confidence'] = 65
        
        if 'oversold_hook' in signals.columns:
            signals.loc[signals['oversold_hook'], 'signal_type'] = 'CCI超卖区域钩子'
            signals.loc[signals['oversold_hook'], 'signal_desc'] = 'CCI在超卖区域形成拐点，看涨信号'
            signals.loc[signals['oversold_hook'], 'confidence'] = 70
            
            signals.loc[signals['overbought_hook'], 'signal_type'] = 'CCI超买区域钩子'
            signals.loc[signals['overbought_hook'], 'signal_desc'] = 'CCI在超买区域形成拐点，看跌信号'
            signals.loc[signals['overbought_hook'], 'confidence'] = 70
            
            signals.loc[signals['bullish_divergence'], 'signal_type'] = 'CCI正背离'
            signals.loc[signals['bullish_divergence'], 'signal_desc'] = '价格创新低但CCI未创新低，看涨信号'
            signals.loc[signals['bullish_divergence'], 'confidence'] = 75
            
            signals.loc[signals['bearish_divergence'], 'signal_type'] = 'CCI负背离'
            signals.loc[signals['bearish_divergence'], 'signal_desc'] = '价格创新高但CCI未创新高，看跌信号'
            signals.loc[signals['bearish_divergence'], 'confidence'] = 75
            
            signals.loc[signals['head_shoulders_bottom'], 'signal_type'] = 'CCI头肩底'
            signals.loc[signals['head_shoulders_bottom'], 'signal_desc'] = 'CCI形成头肩底形态，看涨信号'
            signals.loc[signals['head_shoulders_bottom'], 'confidence'] = 80
            
            signals.loc[signals['head_shoulders_top'], 'signal_type'] = 'CCI头肩顶'
            signals.loc[signals['head_shoulders_top'], 'signal_desc'] = 'CCI形成头肩顶形态，看跌信号'
            signals.loc[signals['head_shoulders_top'], 'confidence'] = 80
            
            signals.loc[signals['platform_breakout_up'], 'signal_type'] = 'CCI平台向上突破'
            signals.loc[signals['platform_breakout_up'], 'signal_desc'] = 'CCI从窄幅区间向上突破，看涨信号'
            signals.loc[signals['platform_breakout_up'], 'confidence'] = 70
            
            signals.loc[signals['platform_breakout_down'], 'signal_type'] = 'CCI平台向下突破'
            signals.loc[signals['platform_breakout_down'], 'signal_desc'] = 'CCI从窄幅区间向下突破，看跌信号'
            signals.loc[signals['platform_breakout_down'], 'confidence'] = 70
        
        if 'bullish_agreement' in signals.columns:
            signals.loc[signals['bullish_agreement'] & signals.get('bullish_alignment', pd.Series(False, index=data.index)), 'signal_type'] = 'CCI多周期看涨一致'
            signals.loc[signals['bullish_agreement'] & signals.get('bullish_alignment', pd.Series(False, index=data.index)), 'signal_desc'] = '多周期CCI均为看涨且排列良好，强烈看涨信号'
            signals.loc[signals['bullish_agreement'] & signals.get('bullish_alignment', pd.Series(False, index=data.index)), 'confidence'] = 85
            
            signals.loc[signals['bearish_agreement'] & signals.get('bearish_alignment', pd.Series(False, index=data.index)), 'signal_type'] = 'CCI多周期看跌一致'
            signals.loc[signals['bearish_agreement'] & signals.get('bearish_alignment', pd.Series(False, index=data.index)), 'signal_desc'] = '多周期CCI均为看跌且排列良好，强烈看跌信号'
            signals.loc[signals['bearish_agreement'] & signals.get('bearish_alignment', pd.Series(False, index=data.index)), 'confidence'] = 85
        
        # 添加标准格式字段
        signals['trend'] = np.where(signals['score'] > 50, 1, np.where(signals['score'] < 50, -1, 0))
        signals['risk_level'] = np.where(signals['confidence'] > 70, '低', np.where(signals['confidence'] > 60, '中', '高'))
        signals['position_size'] = signals['confidence'] / 100 * 0.1  # 最大仓位10%
        
        return signals
    
    def calculate_score(self, data: pd.DataFrame = None) -> pd.Series:
        """
        计算CCI综合评分 (0-100)
        
        Args:
            data (pd.DataFrame, optional): 价格数据，如果未提供则使用上次计算结果
            
        Returns:
            pd.Series: 评分 (0-100，50为中性)
        """
        if self._result is None and data is not None:
            self.calculate(data)
            
        if self._result is None:
            return pd.Series()
        
        # 获取CCI数据
        cci = self._result['cci']
        cci_slope = self._result['cci_slope']
        
        # 获取交叉分析
        crossovers = self.analyze_crossovers()
        
        # 获取形态识别
        patterns = self.identify_patterns()
        
        # 获取多周期协同分析
        synergy = self.analyze_multi_period_synergy()
        
        # 基础分数为50（中性）
        score = pd.Series(50, index=self._result.index)
        
        # 1. CCI位置评分 (±20分)
        # CCI > 0 看涨，CCI < 0 看跌
        score += np.where(cci > 0, (cci / 200).clip(0, 1) * 20, (cci / 200).clip(-1, 0) * 20)
        
        # 2. CCI斜率评分 (±15分)
        # 斜率为正看涨，斜率为负看跌
        normalized_slope = cci_slope / cci_slope.rolling(window=20).std().replace(0, 1)
        score += np.where(normalized_slope > 0, 
                          np.minimum(normalized_slope, 3) * 5, 
                          np.maximum(normalized_slope, -3) * 5)
        
        # 3. 交叉评分 (±10分)
        if not crossovers.empty:
            # 零轴向上交叉
            score.loc[crossovers['zero_cross_up']] += 10
            # 零轴向下交叉
            score.loc[crossovers['zero_cross_down']] -= 10
            # 从超卖区域回升
            score.loc[crossovers['oversold_exit']] += 10
            # 从超买区域回落
            score.loc[crossovers['overbought_exit']] -= 10
            
            # 根据交叉强度调整
            crossover_strength = crossovers['crossover_strength']
            score += np.where(
                (crossovers['zero_cross_up'] | crossovers['oversold_exit']), 
                (crossover_strength - 50) / 10,
                np.where(
                    (crossovers['zero_cross_down'] | crossovers['overbought_exit']),
                    (50 - crossover_strength) / 10,
                    0
                )
            )
        
        # 4. 形态评分 (±25分)
        if not patterns.empty:
            # 看涨形态
            if 'oversold_hook' in patterns.columns:
                score.loc[patterns['oversold_hook']] += 15
            if 'bullish_divergence' in patterns.columns:
                score.loc[patterns['bullish_divergence']] += 25
            if 'head_shoulders_bottom' in patterns.columns:
                score.loc[patterns['head_shoulders_bottom']] += 25
            if 'platform_breakout_up' in patterns.columns:
                score.loc[patterns['platform_breakout_up']] += 20
            
            # 看跌形态
            if 'overbought_hook' in patterns.columns:
                score.loc[patterns['overbought_hook']] -= 15
            if 'bearish_divergence' in patterns.columns:
                score.loc[patterns['bearish_divergence']] -= 25
            if 'head_shoulders_top' in patterns.columns:
                score.loc[patterns['head_shoulders_top']] -= 25
            if 'platform_breakout_down' in patterns.columns:
                score.loc[patterns['platform_breakout_down']] -= 20
        
        # 5. 多周期协同评分 (±20分)
        if not synergy.empty and 'synergy_score' in synergy.columns:
            # 协同评分转换为相对于中性50分的偏差
            synergy_adjustment = (synergy['synergy_score'] - 50) / 2.5
            score += synergy_adjustment
        
        # 6. 市场环境调整
        if self.market_environment == "bull_market":
            # 牛市中增强多头信号，弱化空头信号
            bull_adjustment = np.where(score > 50, (score - 50) * 0.2, (score - 50) * 0.1)
            score += bull_adjustment
        elif self.market_environment == "bear_market":
            # 熊市中增强空头信号，弱化多头信号
            bear_adjustment = np.where(score < 50, (50 - score) * 0.2, (50 - score) * 0.1)
            score -= bear_adjustment
        elif self.market_environment == "volatile_market":
            # 高波动市场需要更强的信号
            vol_adjustment = (score - 50).abs() * 0.3
            score = np.where(score > 50, 50 + vol_adjustment, 50 - vol_adjustment)
        
        # 限制分数范围在0-100之间
        score = score.clip(0, 100)
        
        return score 