"""
布林带指标模块

实现布林带(BOLL)指标计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.common import boll as calc_boll
from utils.logger import get_logger

logger = get_logger(__name__)


class BOLL(BaseIndicator):
    """
    布林带指标类
    
    计算布林带上轨、中轨和下轨，支持自适应带宽和动态评估系统
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, moving_average_type: str = 'sma'):
        """
        初始化布林带指标
        
        Args:
            period: 周期，默认为20
            std_dev: 标准差倍数，默认为2.0
            moving_average_type: 移动平均类型，'sma'或'ema'，默认为'sma'
        """
        super().__init__(name="BOLL", description="布林带")
        self.period = period
        self.std_dev = std_dev
        self.moving_average_type = moving_average_type
        self._market_environment = MarketEnvironment.SIDEWAYS_MARKET
        
        # 注册布林带指标形态
        self._register_boll_patterns()
    
    def _register_boll_patterns(self):
        """注册布林带指标的各种形态"""
        # 价格触及上轨形态
        self.register_pattern(
            pattern_id="BOLL_PRICE_TOUCH_UPPER",
            display_name="布林带价格触及上轨",
            detection_func=self._detect_price_touch_upper,
            score_impact=-10.0
        )
        
        # 价格触及下轨形态
        self.register_pattern(
            pattern_id="BOLL_PRICE_TOUCH_LOWER",
            display_name="布林带价格触及下轨",
            detection_func=self._detect_price_touch_lower,
            score_impact=10.0
        )
        
        # 价格突破上轨形态
        self.register_pattern(
            pattern_id="BOLL_PRICE_BREAK_UPPER",
            display_name="布林带价格突破上轨",
            detection_func=self._detect_price_break_upper,
            score_impact=-15.0
        )
        
        # 价格突破下轨形态
        self.register_pattern(
            pattern_id="BOLL_PRICE_BREAK_LOWER",
            display_name="布林带价格突破下轨",
            detection_func=self._detect_price_break_lower,
            score_impact=15.0
        )
        
        # 带宽扩大形态
        self.register_pattern(
            pattern_id="BOLL_BANDWIDTH_EXPANDING",
            display_name="布林带带宽扩大",
            detection_func=self._detect_bandwidth_expanding,
            score_impact=5.0
        )
        
        # 带宽收缩形态
        self.register_pattern(
            pattern_id="BOLL_BANDWIDTH_CONTRACTING",
            display_name="布林带带宽收缩",
            detection_func=self._detect_bandwidth_contracting,
            score_impact=-5.0
        )
        
        # 价格向上突破中轨形态
        self.register_pattern(
            pattern_id="BOLL_PRICE_CROSS_UP_MIDDLE",
            display_name="价格向上突破中轨",
            detection_func=self._detect_price_cross_up_middle,
            score_impact=12.0
        )
        
        # 价格向下突破中轨形态
        self.register_pattern(
            pattern_id="BOLL_PRICE_CROSS_DOWN_MIDDLE",
            display_name="价格向下突破中轨",
            detection_func=self._detect_price_cross_down_middle,
            score_impact=-12.0
        )
        
        # 上下轨平行形态
        self.register_pattern(
            pattern_id="BOLL_PARALLEL_BANDS",
            display_name="布林带上下轨平行",
            detection_func=self._detect_parallel_bands,
            score_impact=0.0
        )
        
        # W底形态
        self.register_pattern(
            pattern_id="BOLL_W_BOTTOM",
            display_name="布林带W底",
            detection_func=self._detect_w_bottom,
            score_impact=20.0
        )
        
        # M顶形态
        self.register_pattern(
            pattern_id="BOLL_M_TOP",
            display_name="布林带M顶",
            detection_func=self._detect_m_top,
            score_impact=-20.0
        )
    
    def _detect_price_touch_upper(self, data: pd.DataFrame) -> bool:
        """检测价格触及上轨形态"""
        if 'upper' not in data.columns or 'close' not in data.columns:
            return False
        
        # 价格触及上轨：收盘价接近上轨但不突破
        upper = data['upper']
        close = data['close']
        tolerance = 0.003  # 0.3%的容差
        
        # 计算价格与上轨的相对距离
        distance = (upper - close) / close
        touch_upper = (distance >= 0) & (distance <= tolerance)
        
        # 检查最近5个周期是否触及上轨
        return touch_upper.iloc[-5:].any()
    
    def _detect_price_touch_lower(self, data: pd.DataFrame) -> bool:
        """检测价格触及下轨形态"""
        if 'lower' not in data.columns or 'close' not in data.columns:
            return False
        
        # 价格触及下轨：收盘价接近下轨但不突破
        lower = data['lower']
        close = data['close']
        tolerance = 0.003  # 0.3%的容差
        
        # 计算价格与下轨的相对距离
        distance = (close - lower) / close
        touch_lower = (distance >= 0) & (distance <= tolerance)
        
        # 检查最近5个周期是否触及下轨
        return touch_lower.iloc[-5:].any()
    
    def _detect_price_break_upper(self, data: pd.DataFrame) -> bool:
        """检测价格突破上轨形态"""
        if 'upper' not in data.columns or 'close' not in data.columns:
            return False
        
        # 价格突破上轨：收盘价高于上轨
        upper = data['upper']
        close = data['close']
        break_upper = close > upper
        
        # 检查最近5个周期是否突破上轨
        return break_upper.iloc[-5:].any()
    
    def _detect_price_break_lower(self, data: pd.DataFrame) -> bool:
        """检测价格突破下轨形态"""
        if 'lower' not in data.columns or 'close' not in data.columns:
            return False
        
        # 价格突破下轨：收盘价低于下轨
        lower = data['lower']
        close = data['close']
        break_lower = close < lower
        
        # 检查最近5个周期是否突破下轨
        return break_lower.iloc[-5:].any()
    
    def _detect_bandwidth_expanding(self, data: pd.DataFrame) -> bool:
        """检测带宽扩大形态"""
        if 'bandwidth' not in data.columns:
            # 如果没有带宽列，计算带宽
            if 'upper' not in data.columns or 'lower' not in data.columns or 'middle' not in data.columns:
                return False
            
            upper = data['upper']
            lower = data['lower']
            middle = data['middle']
            bandwidth = (upper - lower) / middle
        else:
            bandwidth = data['bandwidth']
        
        # 带宽扩大：当前带宽大于过去3个周期的带宽
        expanding = (bandwidth > bandwidth.shift(1)) & (bandwidth.shift(1) > bandwidth.shift(2)) & (bandwidth.shift(2) > bandwidth.shift(3))
        
        # 检查最近5个周期是否存在带宽扩大
        return expanding.iloc[-5:].any()
    
    def _detect_bandwidth_contracting(self, data: pd.DataFrame) -> bool:
        """检测带宽收缩形态"""
        if 'bandwidth' not in data.columns:
            # 如果没有带宽列，计算带宽
            if 'upper' not in data.columns or 'lower' not in data.columns or 'middle' not in data.columns:
                return False
            
            upper = data['upper']
            lower = data['lower']
            middle = data['middle']
            bandwidth = (upper - lower) / middle
        else:
            bandwidth = data['bandwidth']
        
        # 带宽收缩：当前带宽小于过去3个周期的带宽
        contracting = (bandwidth < bandwidth.shift(1)) & (bandwidth.shift(1) < bandwidth.shift(2)) & (bandwidth.shift(2) < bandwidth.shift(3))
        
        # 检查最近5个周期是否存在带宽收缩
        return contracting.iloc[-5:].any()
    
    def _detect_price_cross_up_middle(self, data: pd.DataFrame) -> bool:
        """检测价格向上突破中轨形态"""
        if 'middle' not in data.columns or 'close' not in data.columns:
            return False
        
        # 价格向上突破中轨：价格从下方穿过中轨
        middle = data['middle']
        close = data['close']
        cross_up = (close > middle) & (close.shift(1) <= middle.shift(1))
        
        # 检查最近5个周期是否向上突破中轨
        return cross_up.iloc[-5:].any()
    
    def _detect_price_cross_down_middle(self, data: pd.DataFrame) -> bool:
        """检测价格向下突破中轨形态"""
        if 'middle' not in data.columns or 'close' not in data.columns:
            return False
        
        # 价格向下突破中轨：价格从上方穿过中轨
        middle = data['middle']
        close = data['close']
        cross_down = (close < middle) & (close.shift(1) >= middle.shift(1))
        
        # 检查最近5个周期是否向下突破中轨
        return cross_down.iloc[-5:].any()
    
    def _detect_parallel_bands(self, data: pd.DataFrame) -> bool:
        """检测上下轨平行形态"""
        if 'upper' not in data.columns or 'lower' not in data.columns:
            return False
        
        # 计算上下轨的斜率
        upper_slope = data['upper'].diff(5).iloc[-1]
        lower_slope = data['lower'].diff(5).iloc[-1]
        
        # 上下轨平行：斜率差异小于阈值
        slope_diff = abs(upper_slope - lower_slope)
        parallel = slope_diff < 0.01 * data['middle'].iloc[-1]  # 斜率差异小于中轨的1%
        
        return parallel
    
    def _detect_w_bottom(self, data: pd.DataFrame) -> bool:
        """检测W底形态"""
        if 'close' not in data.columns or 'lower' not in data.columns:
            return False
        
        # W底：价格两次接近或触及下轨，且中间有反弹
        close = data['close']
        lower = data['lower']
        
        # 使用窗口检测
        window_size = 20
        if len(close) < window_size:
            return False
        
        # 获取最近的窗口数据
        recent_close = close.iloc[-window_size:]
        recent_lower = lower.iloc[-window_size:]
        
        # 计算价格与下轨的距离
        distance = (recent_close - recent_lower) / recent_close
        
        # 寻找两次接近下轨的点，且中间有反弹
        touches = distance < 0.01  # 接近下轨的阈值
        touch_indices = np.where(touches)[0]
        
        if len(touch_indices) >= 2:
            # 检查中间是否有反弹
            for i in range(len(touch_indices) - 1):
                idx1 = touch_indices[i]
                idx2 = touch_indices[i + 1]
                
                # 至少间隔5个周期
                if idx2 - idx1 >= 5:
                    # 中间最高点比两端高
                    middle_high = recent_close.iloc[idx1:idx2].max()
                    if middle_high > recent_close.iloc[idx1] * 1.03 and middle_high > recent_close.iloc[idx2] * 1.03:
                        return True
        
        return False
    
    def _detect_m_top(self, data: pd.DataFrame) -> bool:
        """检测M顶形态"""
        if 'close' not in data.columns or 'upper' not in data.columns:
            return False
        
        # M顶：价格两次接近或触及上轨，且中间有回落
        close = data['close']
        upper = data['upper']
        
        # 使用窗口检测
        window_size = 20
        if len(close) < window_size:
            return False
        
        # 获取最近的窗口数据
        recent_close = close.iloc[-window_size:]
        recent_upper = upper.iloc[-window_size:]
        
        # 计算价格与上轨的距离
        distance = (recent_upper - recent_close) / recent_close
        
        # 寻找两次接近上轨的点，且中间有回落
        touches = distance < 0.01  # 接近上轨的阈值
        touch_indices = np.where(touches)[0]
        
        if len(touch_indices) >= 2:
            # 检查中间是否有回落
            for i in range(len(touch_indices) - 1):
                idx1 = touch_indices[i]
                idx2 = touch_indices[i + 1]
                
                # 至少间隔5个周期
                if idx2 - idx1 >= 5:
                    # 中间最低点比两端低
                    middle_low = recent_close.iloc[idx1:idx2].min()
                    if middle_low < recent_close.iloc[idx1] * 0.97 and middle_low < recent_close.iloc[idx2] * 0.97:
                        return True
        
        return False
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算布林带指标的原始评分（0-100分制）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列，取值范围0-100
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 获取布林带指标值
        if 'upper' not in data.columns or 'middle' not in data.columns or 'lower' not in data.columns or 'close' not in data.columns:
            raise ValueError("数据中缺少布林带指标列")
        
        close = data['close']
        upper = data['upper']
        middle = data['middle']
        lower = data['lower']
        
        # 计算带宽
        bandwidth = (upper - lower) / middle
        
        # 计算价格相对位置
        # 将价格位置映射到0-100：0表示在下轨以下，50表示在中轨，100表示在上轨以上
        price_position = (close - lower) / (upper - lower) * 100
        
        # 基础评分计算
        # 1. 位置分：基于价格相对布林带的位置，贡献50分权重
        # 映射规则：0->80, 25->65, 50->50, 75->35, 100->20
        # 这种映射使得价格越接近下轨，评分越高，反之越低
        position_score = np.where(
            price_position <= 50,
            80 - (price_position / 50) * 30,  # 0-50映射到80-50
            50 - ((price_position - 50) / 50) * 30  # 50-100映射到50-20
        )
        
        # 2. 趋势分：基于价格与中轨的关系，贡献25分权重
        # 计算价格与中轨的相对关系
        price_middle_ratio = close / middle - 1
        trend_score = 50 + price_middle_ratio * 200
        # 限制在30-70范围内
        trend_score = np.clip(trend_score, 30, 70)
        
        # 3. 带宽分：基于带宽的变化，贡献25分权重
        # 计算带宽变化率
        bandwidth_change = bandwidth / bandwidth.rolling(window=10).mean()
        # 带宽扩大时得分高，收缩时得分低
        bandwidth_score = np.where(
            bandwidth_change > 1,
            50 + (bandwidth_change - 1) * 50,  # 带宽扩大，加分
            50 - (1 - bandwidth_change) * 50   # 带宽收缩，减分
        )
        # 限制在30-70范围内
        bandwidth_score = np.clip(bandwidth_score, 30, 70)
        
        # 合并各部分得分，按权重加权平均
        raw_score = (
            position_score * 0.5 +  # 位置分权重50%
            trend_score * 0.25 +     # 趋势分权重25%
            bandwidth_score * 0.25   # 带宽分权重25%
        )
        
        # 检测形态对评分的影响
        patterns = self.get_patterns(data, **kwargs)
        
        # 形态影响分数：最多调整±20分
        pattern_adjustment = pd.Series(0, index=data.index)
        for pattern in patterns:
            # 找到对应的注册形态信息
            if pattern.pattern_id in self._registered_patterns:
                score_impact = self._registered_patterns[pattern.pattern_id]['score_impact']
                # 应用形态影响
                pattern_adjustment += score_impact
        
        # 限制形态调整范围
        pattern_adjustment = np.clip(pattern_adjustment, -20, 20)
        raw_score += pattern_adjustment
        
        # 确保最终分数在0-100范围内
        final_score = np.clip(raw_score, 0, 100)
        
        return pd.Series(final_score, index=data.index)
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算布林带指标
        
        Args:
            data: 输入数据，必须包含'close'列
            args: 位置参数
            kwargs: 关键字参数，可包含periods和std_dev
            
        Returns:
            pd.DataFrame: 包含upper、middle、lower列的DataFrame
        """
        # 确保数据包含close列
        self.ensure_columns(data, ['close'])
        
        # 获取参数
        periods = kwargs.get('periods', self.period)
        std_dev = kwargs.get('std_dev', self.std_dev)
        adaptive = kwargs.get('adaptive', True)
        
        # 如果启用自适应带宽，动态调整标准差倍数
        if adaptive:
            adaptive_std_dev = self.adjust_std_multiplier(data['close'])
            upper, middle, lower = calc_boll(data['close'], periods, adaptive_std_dev)
        else:
            upper, middle, lower = calc_boll(data['close'], periods, std_dev)
        
        # 构建结果DataFrame
        result = data.copy()
        result['upper'] = upper
        result['middle'] = middle
        result['lower'] = lower
        
        # 计算带宽
        result['bandwidth'] = (upper - lower) / middle
        
        # 分析带宽动态
        if len(result) >= 20:
            bandwidth_analysis = self.analyze_bandwidth_dynamics(result)
            result['bandwidth_expansion'] = bandwidth_analysis['expansion']
            result['bandwidth_contraction'] = bandwidth_analysis['contraction']
            result['bandwidth_extreme'] = bandwidth_analysis['extreme']
            result['bandwidth_trend_prediction'] = bandwidth_analysis['trend_prediction']
        
        # 分析中轨支撑阻力
        if len(result) >= 20:
            sr_analysis = self.analyze_middle_band_support_resistance(result)
            result['middle_support'] = sr_analysis['support']
            result['middle_resistance'] = sr_analysis['resistance']
            result['middle_strength'] = sr_analysis['strength']
            result['middle_reversal_probability'] = sr_analysis['reversal_probability']
        
        # 保存结果
        self._result = result
        
        return result
    
    def adjust_std_multiplier(self, close: pd.Series) -> pd.Series:
        """
        根据历史波动率动态调整布林带标准差倍数
        
        高波动时期减小倍数，低波动时期增大倍数，使得布林带更好地适应市场环境
        
        Args:
            close: 收盘价序列
            
        Returns:
            pd.Series: 自适应的标准差倍数序列
        """
        # 获取参数
        volatility_window = 60
        min_std_dev = 1.0
        max_std_dev = 3.0
        base_std_dev = self.std_dev
        
        # 初始化为基础标准差倍数
        adaptive_std_dev = pd.Series(base_std_dev, index=close.index)
        
        # 检查数据长度是否足够
        if len(close) < volatility_window:
            logger.warning(f"数据长度不足以计算自适应标准差，使用基础标准差 {base_std_dev}")
            return adaptive_std_dev
        
        # 计算滚动波动率（使用对数收益率的标准差）
        returns = np.log(close / close.shift(1)).dropna()
        rolling_volatility = returns.rolling(window=volatility_window).std()
        
        # 检查是否有足够的有效波动率数据
        valid_volatility = rolling_volatility.dropna()
        if len(valid_volatility) > 0:
            # 计算波动率的历史分位数
            rank_window = min(volatility_window*2, len(valid_volatility))
            volatility_rank = rolling_volatility.rolling(window=rank_window).rank(pct=True)
            
            # 使用布尔索引而不是位置索引，避免索引越界问题
            valid_ranks = ~volatility_rank.isna()
            
            # 低波动（分位数低）= 更高的倍数
            low_volatility = (volatility_rank < 0.2) & valid_ranks
            adaptive_std_dev[low_volatility] = base_std_dev * 1.2  # 增大倍数20%
            
            # 高波动（分位数高）= 更低的倍数
            high_volatility = (volatility_rank > 0.8) & valid_ranks
            adaptive_std_dev[high_volatility] = base_std_dev * 0.8  # 减小倍数20%
            
            # 正常波动 - 线性映射波动率分位数到标准差倍数
            normal_volatility = (~low_volatility & ~high_volatility) & valid_ranks
            if normal_volatility.any():
                # 0.2-0.8的分位数线性映射到0.8-1.2的调整因子
                adjust_factor = 1.2 - volatility_rank[normal_volatility]
                adaptive_std_dev[normal_volatility] = base_std_dev * adjust_factor
        
        # 确保标准差倍数在合理范围内
        adaptive_std_dev = adaptive_std_dev.clip(min_std_dev, max_std_dev)
        
        return adaptive_std_dev
    
    def analyze_bandwidth_dynamics(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        分析带宽变化趋势，识别压缩和扩张形态
        
        Args:
            data: 包含bandwidth列的DataFrame
            
        Returns:
            Dict[str, pd.Series]: 包含带宽分析结果的字典
        """
        bw_periods = 20
        threshold = 0.05
        
        # 初始化结果
        expansion = pd.Series(False, index=data.index)
        contraction = pd.Series(False, index=data.index)
        extreme = pd.Series(0, index=data.index)  # 0=正常, 1=极低带宽, 2=极高带宽
        trend_prediction = pd.Series(0, index=data.index)  # 0=无明确趋势, 1=即将扩张, -1=即将收缩
        
        bandwidth = data['bandwidth']
        
        # 计算带宽变化率
        bw_change_rate = bandwidth.pct_change(periods=bw_periods)
        
        # 识别带宽快速扩张
        expansion = bw_change_rate > 0.2  # 带宽20%以上的扩张
        
        # 识别带宽快速收缩
        contraction = bw_change_rate < -0.2  # 带宽20%以上的收缩
        
        # 计算带宽历史分位数，识别极值
        if len(bandwidth.dropna()) >= 50:
            bw_rank = bandwidth.rolling(window=100).rank(pct=True)
            
            # 极低带宽（压缩状态，可能蓄势待发）
            extreme_low = bw_rank < threshold
            # 极高带宽（扩张状态，可能过度波动）
            extreme_high = bw_rank > (1 - threshold)
            
            extreme = pd.Series(0, index=data.index)
            extreme[extreme_low] = 1
            extreme[extreme_high] = 2
            
            # 带宽趋势预测
            # 识别极低带宽后的扩张前兆
            for i in range(5, len(bandwidth)):
                if extreme.iloc[i] == 1:  # 极低带宽
                    # 检查是否有微弱扩张迹象
                    recent_bandwidth = bandwidth.iloc[i-5:i+1]
                    if recent_bandwidth.iloc[-1] > recent_bandwidth.iloc[-2] > recent_bandwidth.iloc[-3]:
                        trend_prediction.iloc[i] = 1  # 预测即将扩张
                elif extreme.iloc[i] == 2:  # 极高带宽
                    # 检查是否有微弱收缩迹象
                    recent_bandwidth = bandwidth.iloc[i-5:i+1]
                    if recent_bandwidth.iloc[-1] < recent_bandwidth.iloc[-2] < recent_bandwidth.iloc[-3]:
                        trend_prediction.iloc[i] = -1  # 预测即将收缩
        
        return {
            'expansion': expansion,
            'contraction': contraction,
            'extreme': extreme,
            'trend_prediction': trend_prediction
        }
    
    def analyze_middle_band_support_resistance(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        评估中轨作为支撑/阻力的有效性
        
        Args:
            data: 包含close, middle等列的DataFrame
            
        Returns:
            Dict[str, pd.Series]: 包含中轨支撑阻力分析结果的字典
        """
        lookback = 20
        
        # 初始化结果
        support = pd.Series(False, index=data.index)
        resistance = pd.Series(False, index=data.index)
        strength = pd.Series(0.0, index=data.index)  # 支撑/阻力强度，0-100
        reversal_probability = pd.Series(0.0, index=data.index)  # 反转概率，0-100
        
        close = data['close']
        middle = data['middle']
        
        # 初始化强度累计器
        cumulative_strength = 0
        
        for i in range(lookback, len(data)):
            # 获取当前窗口内的数据
            window_close = close.iloc[i-lookback:i]
            window_middle = middle.iloc[i-lookback:i]
            
            # 计算价格与中轨交叉次数
            crosses = ((window_close > window_middle) != (window_close.shift(1) > window_middle.shift(1))).sum()
            
            # 判断中轨作为支撑或阻力的有效性
            if close.iloc[i] > middle.iloc[i]:
                # 价格在中轨上方，评估中轨作为支撑的能力
                touches = ((window_close - window_middle).abs() < (window_close * 0.002)).sum()
                bounces = ((window_close < window_middle) & (window_close.shift(1) < window_middle.shift(1)) & 
                          (window_close.shift(-1) > window_middle.shift(-1))).sum()
                
                # 计算支撑强度
                if touches > 0:
                    support_ratio = bounces / touches
                    support_strength = min(100, support_ratio * 100)
                    support.iloc[i] = True
                    strength.iloc[i] = support_strength
                    
                    # 计算反转概率（从支撑转为阻力）
                    # 如果价格接近中轨且支撑强度较低，反转概率高
                    distance_to_middle = (close.iloc[i] - middle.iloc[i]) / middle.iloc[i]
                    if distance_to_middle < 0.01:  # 价格非常接近中轨
                        reversal_probability.iloc[i] = max(0, 100 - support_strength)
                    else:
                        reversal_probability.iloc[i] = max(0, (80 - support_strength) * (0.02 / distance_to_middle))
            else:
                # 价格在中轨下方，评估中轨作为阻力的能力
                touches = ((window_close - window_middle).abs() < (window_close * 0.002)).sum()
                rejections = ((window_close > window_middle) & (window_close.shift(1) > window_middle.shift(1)) & 
                             (window_close.shift(-1) < window_middle.shift(-1))).sum()
                
                # 计算阻力强度
                if touches > 0:
                    resistance_ratio = rejections / touches
                    resistance_strength = min(100, resistance_ratio * 100)
                    resistance.iloc[i] = True
                    strength.iloc[i] = resistance_strength
                    
                    # 计算反转概率（从阻力转为支撑）
                    # 如果价格接近中轨且阻力强度较低，反转概率高
                    distance_to_middle = (middle.iloc[i] - close.iloc[i]) / middle.iloc[i]
                    if distance_to_middle < 0.01:  # 价格非常接近中轨
                        reversal_probability.iloc[i] = max(0, 100 - resistance_strength)
                    else:
                        reversal_probability.iloc[i] = max(0, (80 - resistance_strength) * (0.02 / distance_to_middle))
            
            # 判断中轨是否为持续的支撑或阻力
            # 如果价格持续在中轨上方或下方，增加强度
            price_above = (window_close > window_middle).sum()
            price_below = (window_close < window_middle).sum()
            
            if price_above > lookback * 0.8:  # 80%的时间价格在中轨上方
                # 中轨成为可靠支撑的可能性更高
                cumulative_strength += 5
            elif price_below > lookback * 0.8:  # 80%的时间价格在中轨下方
                # 中轨成为可靠阻力的可能性更高
                cumulative_strength += 5
            else:
                # 价格频繁穿越中轨，支撑阻力不明显
                cumulative_strength = max(0, cumulative_strength - 2)
            
            # 累积强度衰减
            cumulative_strength = max(0, cumulative_strength - 1)
            
            # 添加累积强度到当前强度
            strength.iloc[i] += min(30, cumulative_strength)  # 最多额外增加30分
            
            # 确保强度在0-100范围内
            strength.iloc[i] = min(100, max(0, strength.iloc[i]))
            
            # 分析最近的中轨穿越模式，调整反转概率
            if i >= lookback + 3:
                recent_crosses = ((close.iloc[i-3:i] > middle.iloc[i-3:i]) != 
                                 (close.iloc[i-3:i].shift(1) > middle.iloc[i-3:i].shift(1))).sum()
                
                # 如果最近频繁穿越中轨，增加反转概率
                if recent_crosses >= 2:
                    reversal_probability.iloc[i] = min(100, reversal_probability.iloc[i] + 20)
        
        return {
            'support': support,
            'resistance': resistance,
            'strength': strength,
            'reversal_probability': reversal_probability
        }
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别BOLL技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算BOLL
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        close = self._result['close']
        upper = self._result['upper']
        middle = self._result['middle']
        lower = self._result['lower']
        
        # 检查最近的信号
        recent_periods = min(10, len(close))  # 增加至10周期以识别更多形态
        if recent_periods == 0:
            return patterns
        
        recent_close = close.tail(recent_periods)
        recent_upper = upper.tail(recent_periods)
        recent_middle = middle.tail(recent_periods)
        recent_lower = lower.tail(recent_periods)
        
        current_close = recent_close.iloc[-1]
        current_upper = recent_upper.iloc[-1]
        current_middle = recent_middle.iloc[-1]
        current_lower = recent_lower.iloc[-1]
        
        # 1. 基本位置形态
        if current_close >= current_upper:
            patterns.append("布林带超买区")
        elif current_close <= current_lower:
            patterns.append("布林带超卖区")
        
        # 2. 突破形态
        if self.crossover(recent_close, recent_upper).any():
            patterns.append("突破布林带上轨")
        if self.crossunder(recent_close, recent_lower).any():
            patterns.append("突破布林带下轨")
        
        # 3. 带宽形态 - 优化：更精细的带宽形态识别
        bandwidth = (upper - lower) / middle
        recent_bandwidth = bandwidth.tail(recent_periods)
        
        # 计算带宽历史百分位
        if len(bandwidth) >= 60:
            bandwidth_percentile = bandwidth.rolling(window=60).rank(pct=True)
            recent_bw_percentile = bandwidth_percentile.tail(recent_periods)
            
            if recent_bw_percentile.iloc[-1] < 0.1:
                patterns.append("布林带极度收窄")
            elif recent_bw_percentile.iloc[-1] < 0.2:
                patterns.append("布林带收窄")
            elif recent_bw_percentile.iloc[-1] > 0.9:
                patterns.append("布林带极度扩张")
            elif recent_bw_percentile.iloc[-1] > 0.8:
                patterns.append("布林带扩张")
            
            # 带宽变化趋势
            if recent_bandwidth.iloc[-1] < recent_bandwidth.iloc[-3] * 0.9:
                patterns.append("带宽快速收窄")
            elif recent_bandwidth.iloc[-1] > recent_bandwidth.iloc[-3] * 1.1:
                patterns.append("带宽快速扩张")
        
        # 4. 弹性反转形态 - 优化：识别从边界弹回的形态
        price_position = (close - lower) / (upper - lower)  # 位置百分比(0-1)
        recent_position = price_position.tail(recent_periods)
        
        # 从下轨弹回中轨
        if (recent_position.iloc[-5] < 0.2 and recent_position.iloc[-1] > 0.4 and recent_position.iloc[-1] < 0.6):
            patterns.append("下轨弹回中轨")
        
        # 从上轨回落中轨
        if (recent_position.iloc[-5] > 0.8 and recent_position.iloc[-1] < 0.6 and recent_position.iloc[-1] > 0.4):
            patterns.append("上轨回落中轨")
        
        # 5. 方向一致性形态 - 优化：识别价格与布林带方向一致的形态
        if len(middle) >= 10:
            # 计算中轨斜率和价格斜率
            middle_slope = (middle.iloc[-1] - middle.iloc[-10]) / middle.iloc[-10]
            price_slope = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            
            if middle_slope > 0.01 and price_slope > 0.01:
                patterns.append("价格与布林带同步上涨")
            elif middle_slope < -0.01 and price_slope < -0.01:
                patterns.append("价格与布林带同步下跌")
            elif middle_slope > 0.01 and price_slope < 0:
                patterns.append("价格背离布林带上涨")
            elif middle_slope < -0.01 and price_slope > 0:
                patterns.append("价格背离布林带下跌")
        
        # 6. 走势形态
        if self._detect_w_bottom_pattern(recent_close, recent_lower):
            patterns.append("布林带W底形态")
        if self._detect_m_top_pattern(recent_close, recent_upper):
            patterns.append("布林带M顶形态")
        
        # 增加新的带宽相关形态识别
        if 'bandwidth_extreme' in self._result.columns:
            extreme = self._result['bandwidth_extreme'].iloc[-1]
            if extreme == 1:
                patterns.append("带宽极度收缩")
            elif extreme == 2:
                patterns.append("带宽极度扩张")
        
        # 增加带宽趋势预测形态
        if 'bandwidth_trend_prediction' in self._result.columns:
            trend_prediction = self._result['bandwidth_trend_prediction'].iloc[-1]
            if trend_prediction == 1:
                patterns.append("带宽即将扩张")
            elif trend_prediction == -1:
                patterns.append("带宽即将收缩")
        
        # 增加中轨支撑阻力形态识别
        if 'middle_support' in self._result.columns and 'middle_strength' in self._result.columns:
            is_support = self._result['middle_support'].iloc[-1]
            is_resistance = self._result['middle_resistance'].iloc[-1]
            strength = self._result['middle_strength'].iloc[-1]
            
            if is_support and strength > 70:
                patterns.append("中轨强支撑")
            elif is_support and strength > 50:
                patterns.append("中轨中等支撑")
            elif is_resistance and strength > 70:
                patterns.append("中轨强阻力")
            elif is_resistance and strength > 50:
                patterns.append("中轨中等阻力")
            
            # 增加中轨反转概率形态
            if 'middle_reversal_probability' in self._result.columns:
                reversal_prob = self._result['middle_reversal_probability'].iloc[-1]
                
                if is_support and reversal_prob > 70:
                    patterns.append("中轨支撑即将失效")
                elif is_support and reversal_prob > 50:
                    patterns.append("中轨支撑可能失效")
                elif is_resistance and reversal_prob > 70:
                    patterns.append("中轨阻力即将失效")
                elif is_resistance and reversal_prob > 50:
                    patterns.append("中轨阻力可能失效")
        
        return patterns
    
    def _detect_price_movement(self, close: pd.Series, from_line: pd.Series, 
                              to_line: pd.Series, direction: str) -> pd.Series:
        """
        检测价格从一条线向另一条线移动的情况
        
        Args:
            close: 收盘价序列
            from_line: 起始线
            to_line: 目标线
            direction: 方向，'up'或'down'
            
        Returns:
            pd.Series: 布尔序列，表示是否检测到此类移动
        """
        movement = pd.Series(False, index=close.index)
        
        if direction == 'up':
            # 价格从下方向上方运行
            near_from = abs(close - from_line) / from_line < 0.02  # 接近起始线
            moving_to = close > close.shift(1)  # 价格上升
            approaching_to = (close - to_line).abs() < (close.shift(1) - to_line).abs()  # 接近目标线
            movement = near_from & moving_to & approaching_to
        elif direction == 'down':
            # 价格从上方向下方运行
            near_from = abs(close - from_line) / from_line < 0.02  # 接近起始线
            moving_to = close < close.shift(1)  # 价格下降
            approaching_to = (close - to_line).abs() < (close.shift(1) - to_line).abs()  # 接近目标线
            movement = near_from & moving_to & approaching_to
        
        return movement
    
    def _detect_w_bottom_pattern(self, price: pd.Series, lower: pd.Series) -> bool:
        """
        检测W底形态
        
        Args:
            price: 价格序列
            lower: 下轨序列
            
        Returns:
            bool: 是否为W底形态
        """
        if len(price) < 10:
            return False
        
        # 寻找接近或突破下轨的两个低点
        touch_lower = (price - lower).abs() / lower < 0.02  # 接近下轨的点
        touch_indices = np.where(touch_lower)[0]
        
        if len(touch_indices) >= 2:
            # 查找最近的两个接触点
            last_two = touch_indices[-2:]
            
            # 确保两点之间有反弹（中间点高于两端点）
            if len(last_two) == 2 and last_two[1] - last_two[0] >= 3:  # 至少间隔3个点
                middle_idx = (last_two[0] + last_two[1]) // 2
                if price.iloc[middle_idx] > price.iloc[last_two[0]] * 1.02 and price.iloc[middle_idx] > price.iloc[last_two[1]] * 1.02:
                    # 第二个低点后有向上突破
                    if len(price) > last_two[1] + 2 and price.iloc[-1] > price.iloc[last_two[1]] * 1.03:
                        return True
        
        return False

    def _detect_m_top_pattern(self, price: pd.Series, upper: pd.Series) -> bool:
        """
        检测M顶形态
        
        Args:
            price: 价格序列
            upper: 上轨序列
            
        Returns:
            bool: 是否为M顶形态
        """
        if len(price) < 10:
            return False
        
        # 寻找接近或突破上轨的两个高点
        touch_upper = (price - upper).abs() / upper < 0.02  # 接近上轨的点
        touch_indices = np.where(touch_upper)[0]
        
        if len(touch_indices) >= 2:
            # 查找最近的两个接触点
            last_two = touch_indices[-2:]
            
            # 确保两点之间有回调（中间点低于两端点）
            if len(last_two) == 2 and last_two[1] - last_two[0] >= 3:  # 至少间隔3个点
                middle_idx = (last_two[0] + last_two[1]) // 2
                if price.iloc[middle_idx] < price.iloc[last_two[0]] * 0.98 and price.iloc[middle_idx] < price.iloc[last_two[1]] * 0.98:
                    # 第二个高点后有向下突破
                    if len(price) > last_two[1] + 2 and price.iloc[-1] < price.iloc[last_two[1]] * 0.97:
                        return True
        
        return False

    def set_market_environment(self, environment: Union[str, MarketEnvironment]) -> None:
        """
        设置市场环境
        
        Args:
            environment: 市场环境，可以是MarketEnvironment枚举或字符串
        """
        if isinstance(environment, MarketEnvironment):
            self._market_environment = environment
            return
            
        # 兼容旧版接口
        env_mapping = {
            "bull_market": MarketEnvironment.BULL_MARKET,
            "bear_market": MarketEnvironment.BEAR_MARKET,
            "sideways_market": MarketEnvironment.SIDEWAYS_MARKET,
            "volatile_market": MarketEnvironment.VOLATILE_MARKET,
            "normal": MarketEnvironment.SIDEWAYS_MARKET,
        }
        
        if environment not in env_mapping:
            valid_values = list(env_mapping.keys())
            raise ValueError(f"无效的市场环境，有效值为: {', '.join(valid_values)}")
            
        self._market_environment = env_mapping[environment]
    
    def get_market_environment(self) -> MarketEnvironment:
        """
        获取当前市场环境
        
        Returns:
            MarketEnvironment: 当前市场环境
        """
        return self._market_environment
    
    def detect_market_environment(self, data: pd.DataFrame) -> MarketEnvironment:
        """
        根据价格数据和布林带状态检测市场环境
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            MarketEnvironment: 检测到的市场环境
        """
        if 'close' not in data.columns:
            raise ValueError("输入数据必须包含'close'列")
            
        # 确保已计算布林带
        if not self.has_result():
            self.calculate(data)
            
        result = self._result
        price = data['close']
        
        # 检查是否有足够的数据
        if len(price) < 60:
            return MarketEnvironment.SIDEWAYS_MARKET
            
        # 计算短期和长期趋势
        ma20 = price.rolling(window=20).mean()
        ma60 = price.rolling(window=60).mean()
        
        # 获取布林带带宽和带宽变化
        if 'bandwidth' in result.columns:
            bandwidth = result['bandwidth']
            bandwidth_change = bandwidth.pct_change(periods=20)
            avg_bandwidth = bandwidth.iloc[-60:].mean()
            latest_bandwidth = bandwidth.iloc[-1]
            
            # 使用布林带带宽来检测市场环境
            if latest_bandwidth > avg_bandwidth * 1.5:
                if price.iloc[-1] > result['upper'].iloc[-1]:
                    # 价格突破上轨且带宽扩大 - 强势牛市
                    return MarketEnvironment.BULL_MARKET
                elif price.iloc[-1] < result['lower'].iloc[-1]:
                    # 价格突破下轨且带宽扩大 - 强势熊市
                    return MarketEnvironment.BEAR_MARKET
                else:
                    # 带宽大但价格在轨道内 - 高波动市场
                    return MarketEnvironment.VOLATILE_MARKET
            elif latest_bandwidth < avg_bandwidth * 0.5:
                # 带宽极度压缩 - 可能处于震荡市场
                return MarketEnvironment.SIDEWAYS_MARKET
        
        # 使用传统方法检测市场环境
        # 计算波动率
        returns = price.pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)  # 年化波动率
            
        # 获取最新值
        latest_price = price.iloc[-1]
        latest_ma20 = ma20.iloc[-1]
        latest_ma60 = ma60.iloc[-1]
        latest_volatility = volatility.iloc[-1]
        
        # 计算长期波动率均值
        long_term_volatility = volatility.iloc[-60:].mean() if len(volatility) >= 60 else volatility.mean()
        
        # 判断市场环境
        if latest_volatility > long_term_volatility * 1.5:
            # 高波动率市场
            return MarketEnvironment.VOLATILE_MARKET
        elif latest_price > latest_ma20 and latest_ma20 > latest_ma60 and latest_price > price.iloc[-20:].min() * 1.1:
            # 牛市条件: 价格高于20日均线，20日均线高于60日均线，且价格比近期最低点高10%以上
            return MarketEnvironment.BULL_MARKET
        elif latest_price < latest_ma20 and latest_ma20 < latest_ma60 and latest_price < price.iloc[-20:].max() * 0.9:
            # 熊市条件: 价格低于20日均线，20日均线低于60日均线，且价格比近期最高点低10%以上
            return MarketEnvironment.BEAR_MARKET
        elif abs((latest_price / latest_ma60) - 1) < 0.05:
            # 盘整市场: 价格在长期均线附近波动不超过5%
            return MarketEnvironment.SIDEWAYS_MARKET
        else:
            # 默认为盘整市场
            return MarketEnvironment.SIDEWAYS_MARKET
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成布林带交易信号
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 包含各类信号的字典
        """
        # 确保已计算布林带
        if not self.has_result():
            self.calculate(data)
            
        result = self._result
        price = data['close']
        
        # 初始化信号字典
        signals = {}
        
        # 获取主要布林带数据
        upper = result['upper']
        middle = result['middle']
        lower = result['lower']
        
        # 计算价格与各轨道的关系
        price_above_upper = price > upper
        price_below_lower = price < lower
        price_above_middle = price > middle
        price_below_middle = price < middle
        price_crossing_upper = (price > upper) & (price.shift(1) <= upper)
        price_crossing_lower = (price < lower) & (price.shift(1) >= lower)
        price_crossing_middle_up = (price > middle) & (price.shift(1) <= middle)
        price_crossing_middle_down = (price < middle) & (price.shift(1) >= middle)
        
        # 计算带宽变化信号
        if 'bandwidth' in result.columns:
            bandwidth = result['bandwidth']
            bandwidth_expanding = bandwidth > bandwidth.shift(1)
            bandwidth_contracting = bandwidth < bandwidth.shift(1)
            signals['bandwidth_expanding'] = bandwidth_expanding
            signals['bandwidth_contracting'] = bandwidth_contracting
            
            # 带宽极值信号
            if 'bandwidth_extreme' in result.columns:
                bandwidth_extreme_low = result['bandwidth_extreme'] == 1
                bandwidth_extreme_high = result['bandwidth_extreme'] == 2
                signals['bandwidth_extreme_low'] = bandwidth_extreme_low
                signals['bandwidth_extreme_high'] = bandwidth_extreme_high
        
        # 生成基础信号
        # 1. 突破上轨 - 超买信号
        overbought_signal = price_above_upper
        # 2. 突破下轨 - 超卖信号
        oversold_signal = price_below_lower
        # 3. 价格在中轨上方 - 多头趋势
        bullish_trend = price_above_middle
        # 4. 价格在中轨下方 - 空头趋势
        bearish_trend = price_below_middle
        
        # 生成交叉信号
        # 5. 价格向上穿越中轨 - 买入信号
        buy_signal = price_crossing_middle_up
        # 6. 价格向下穿越中轨 - 卖出信号
        sell_signal = price_crossing_middle_down
        # 7. 价格从下方反弹接近下轨 - 超卖反弹买入信号
        bounce_buy_signal = (price_below_lower.shift(1)) & (price > price.shift(1)) & (price < lower)
        # 8. 价格从上方回落接近上轨 - 超买回落卖出信号
        bounce_sell_signal = (price_above_upper.shift(1)) & (price < price.shift(1)) & (price > upper)
        
        # 生成带宽相关信号
        if 'bandwidth_trend_prediction' in result.columns:
            # 9. 带宽预测信号
            expansion_prediction = result['bandwidth_trend_prediction'] == 1
            contraction_prediction = result['bandwidth_trend_prediction'] == -1
            signals['expansion_prediction'] = expansion_prediction
            signals['contraction_prediction'] = contraction_prediction
        
        # 计算支撑阻力信号
        if 'middle_support' in result.columns and 'middle_resistance' in result.columns:
            middle_support = result['middle_support']
            middle_resistance = result['middle_resistance']
            middle_support_active = middle_support == 1
            middle_resistance_active = middle_resistance == 1
            signals['middle_support_active'] = middle_support_active
            signals['middle_resistance_active'] = middle_resistance_active
            
            # 支撑位买入信号
            support_buy_signal = middle_support_active & (price > price.shift(1)) & price_below_middle
            # 阻力位卖出信号
            resistance_sell_signal = middle_resistance_active & (price < price.shift(1)) & price_above_middle
            signals['support_buy_signal'] = support_buy_signal
            signals['resistance_sell_signal'] = resistance_sell_signal
        
        # 计算W底和M头形态信号
        w_bottom = pd.Series(False, index=data.index)
        m_top = pd.Series(False, index=data.index)
        
        # 检测形态（每20个点检测一次，提高性能）
        for i in range(40, len(price), 5):
            window_size = 40
            start_idx = max(0, i - window_size)
            
            # 检测W底
            w_bottom.iloc[i] = self._detect_w_bottom_pattern(
                price.iloc[start_idx:i+1], 
                lower.iloc[start_idx:i+1]
            )
            
            # 检测M头
            m_top.iloc[i] = self._detect_m_top_pattern(
                price.iloc[start_idx:i+1], 
                upper.iloc[start_idx:i+1]
            )
        
        # 计算信号强度
        buy_strength = pd.Series(0, index=data.index)
        sell_strength = pd.Series(0, index=data.index)
        
        # 设置基础信号强度
        for i in range(len(price)):
            # 买入信号强度评估
            strength = 0
            if buy_signal.iloc[i]:
                strength += 3  # 中轨上穿基础强度
            if bounce_buy_signal.iloc[i]:
                strength += 4  # 下轨反弹强度更高
            if w_bottom.iloc[i]:
                strength += 5  # W底形态最强
            if 'support_buy_signal' in signals and signals['support_buy_signal'].iloc[i]:
                strength += 2  # 支撑位买入
                
            buy_strength.iloc[i] = min(5, strength)  # 限制最大强度为5
            
            # 卖出信号强度评估
            strength = 0
            if sell_signal.iloc[i]:
                strength += 3  # 中轨下穿基础强度
            if bounce_sell_signal.iloc[i]:
                strength += 4  # 上轨回落强度更高
            if m_top.iloc[i]:
                strength += 5  # M头形态最强
            if 'resistance_sell_signal' in signals and signals['resistance_sell_signal'].iloc[i]:
                strength += 2  # 阻力位卖出
                
            sell_strength.iloc[i] = min(5, strength)  # 限制最大强度为5
        
        # 添加所有基础信号到字典
        signals['buy_signal'] = buy_signal | bounce_buy_signal | w_bottom
        signals['sell_signal'] = sell_signal | bounce_sell_signal | m_top
        signals['buy_strength'] = buy_strength
        signals['sell_strength'] = sell_strength
        signals['overbought'] = overbought_signal
        signals['oversold'] = oversold_signal
        signals['bullish_trend'] = bullish_trend
        signals['bearish_trend'] = bearish_trend
        signals['price_crossing_upper'] = price_crossing_upper
        signals['price_crossing_lower'] = price_crossing_lower
        signals['w_bottom'] = w_bottom
        signals['m_top'] = m_top
        
        # 计算并添加评分
        score = self.calculate_raw_score(data)
        signals['score'] = score
        
        return signals
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取布林带指标的所有形态信息
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 包含形态信息的字典列表
        """
        if not self.has_result():
            self.calculate(data)
            
        result = []
        
        # 检查是否有足够的数据
        if len(self.result) < 2:
            return result
        
        return result

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算指标评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 评分结果，包含：
                - raw_score: 原始评分序列
                - final_score: 最终评分序列
                - market_environment: 市场环境
                - patterns: 识别的形态
                - signals: 生成的信号
                - confidence: 置信度
        """
        try:
            # 确保已计算布林带
            if not self.has_result():
                self.calculate(data)
                
            # 检测市场环境
            market_env = self.detect_market_environment(data)
            self.set_market_environment(market_env)
            
            # 计算原始评分
            raw_score = self.calculate_raw_score(data, **kwargs)
            
            # 应用市场环境调整
            final_score = self.apply_market_environment_adjustment(raw_score, market_env)
            
            # 识别形态
            patterns = self.identify_patterns(data, **kwargs)
            
            # 生成信号
            signals = self.generate_trading_signals(data, **kwargs)
            
            # 计算置信度
            confidence = self.calculate_confidence(raw_score, patterns, signals)
            
            # 构建返回结果
            result = {
                'raw_score': raw_score,
                'final_score': final_score,
                'market_environment': market_env,
                'patterns': patterns,
                'signals': signals,
                'confidence': confidence
            }
            
            return result
            
        except Exception as e:
            self._error = e
            import traceback
            traceback.print_exc()
            logger.error(f"计算布林带评分时发生错误: {str(e)}")
            return {
                'raw_score': None,
                'final_score': None,
                'market_environment': None,
                'patterns': [],
                'signals': {},
                'confidence': 0.0,
                'error': str(e)
            }
            
    def apply_market_environment_adjustment(self, score: pd.Series, market_env: MarketEnvironment) -> pd.Series:
        """
        根据市场环境调整评分
        
        Args:
            score: 原始评分序列
            market_env: 市场环境
            
        Returns:
            pd.Series: 调整后的评分序列
        """
        adjusted_score = score.copy()
        
        if market_env == MarketEnvironment.BULL_MARKET:
            # 牛市环境下，上涨信号得分提高，下跌信号得分降低
            adjusted_score = np.where(score > 50, 
                                    score + (score - 50) * 0.2,  # 多头信号增强
                                    score + (score - 50) * 0.1)  # 空头信号减弱
        elif market_env == MarketEnvironment.BEAR_MARKET:
            # 熊市环境下，下跌信号得分提高，上涨信号得分降低
            adjusted_score = np.where(score < 50, 
                                    score - (50 - score) * 0.2,  # 空头信号增强
                                    score - (score - 50) * 0.1)  # 多头信号减弱
        elif market_env == MarketEnvironment.VOLATILE_MARKET:
            # 高波动市场，极端信号得分更加极端，中性信号得分更加中性
            adjusted_score = np.where(
                (score > 60) | (score < 40),  # 极端信号
                score + (score - 50) * 0.15,  # 极端更极端
                50 + (score - 50) * 0.8  # 中性更中性
            )
            
        # 限制评分范围在0-100之间
        return pd.Series(np.clip(adjusted_score, 0, 100), index=score.index)
        
    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算指标结果的置信度
        
        Args:
            score: 评分序列
            patterns: 识别的形态
            signals: 生成的信号
            
        Returns:
            float: 置信度（0-1）
        """
        # 基础置信度：0.5
        confidence = 0.5
        
        # 根据形态数量调整置信度
        if len(patterns) >= 3:
            confidence += 0.1  # 多种形态同时出现，置信度提高
        
        # 根据评分极端程度调整置信度
        latest_score = score.iloc[-1]
        if latest_score > 80 or latest_score < 20:
            confidence += 0.15  # 评分极端，置信度提高
        elif 40 <= latest_score <= 60:
            confidence -= 0.1  # 评分中性，置信度降低
        
        # 根据信号强度调整置信度
        if 'buy_strength' in signals and signals['buy_strength'].iloc[-1] >= 4:
            confidence += 0.1  # 强买入信号，置信度提高
        if 'sell_strength' in signals and signals['sell_strength'].iloc[-1] >= 4:
            confidence += 0.1  # 强卖出信号，置信度提高
        
        # 根据高质量形态调整置信度
        high_quality_patterns = ["布林带挤压后突破", "W底形态", "M顶形态", "持续上轨压制", "持续下轨支撑"]
        for pattern in high_quality_patterns:
            if pattern in patterns:
                confidence += 0.05  # 高质量形态，置信度提高
                
        # 限制置信度范围在0-1之间
        return max(0.0, min(1.0, confidence))
        
    def has_result(self) -> bool:
        """
        检查是否已计算过指标
        
        Returns:
            bool: 是否已计算过指标
        """
        return hasattr(self, '_result') and self._result is not None 