"""
增强型能量潮(OBV)指标模块

实现改进版的能量潮指标，优化计算方法和信号质量，增加多周期适应能力和市场环境感知
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.obv import OBV
from utils.logger import get_logger
from utils.technical_utils import find_peaks_and_troughs
from utils.indicator_utils import crossover, crossunder

logger = get_logger(__name__)


class EnhancedOBV(OBV):
    """
    增强型能量潮(On Balance Volume)指标
    
    在标准OBV基础上增加了多周期适应能力、噪声过滤、信号质量评估和市场环境感知
    """
    
    def __init__(self, 
                ma_period: int = 30, 
                sensitivity: float = 1.0,
                noise_filter: float = 0.005,
                multi_periods: List[int] = None,
                smooth_period: int = 5,
                adaptive: bool = True,
                use_smoothed_obv: bool = True,
                smoothing_period: int = 5):
        """
        初始化增强型OBV指标
        
        Args:
            ma_period: OBV均线周期，默认为30日
            sensitivity: 灵敏度参数，控制对价格变化的响应程度，默认为1.0
            noise_filter: 噪声过滤阈值，价格变动小于该比例时视为无变动，默认为0.5%
            multi_periods: 多周期分析参数，默认为[5, 10, 20, 60]
            smooth_period: 平滑周期
            adaptive: 是否启用自适应模式
            use_smoothed_obv: 是否使用平滑后的OBV
            smoothing_period: 平滑周期
        """
        self.indicator_type = "ENHANCEDOBV"
        super().__init__()
        self.name = "EnhancedOBV"
        self.description = "增强型OBV指标"
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.ma_period = ma_period
        self.sensitivity = sensitivity
        self.noise_filter = noise_filter
        self.multi_periods = multi_periods or [5, 10, 20, 60]
        self.smooth_period = smooth_period
        self.adaptive = adaptive
        self.market_environment = "normal"
        self.use_smoothed_obv = use_smoothed_obv
        self.smoothing_period = smoothing_period
        
        # 内部参数
        self._result = None
        self._price_data = None
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型
        """
        return self.indicator_type
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算增强型OBV指标
        
        Args:
            data: 输入数据，包含价格和成交量数据
            
        Returns:
            pd.DataFrame: 计算结果，包含OBV及其均线和多周期指标
        """
        # 确保数据包含必需的列
        required_columns = ["close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据必须包含'{col}'列")
        
        # 复制输入数据
        result = data.copy()
        self._price_data = data['close']
        
        # 计算增强型OBV - 使用修正的价格变动方向，增加灵敏度和噪声过滤
        obv = self._calculate_enhanced_obv(result)
        result["obv"] = obv

        # 如果启用，计算平滑后的OBV
        if self.use_smoothed_obv:
            result['obv_smooth'] = obv.rolling(window=self.smoothing_period).mean()
        else:
            result['obv_smooth'] = obv
        
        # 计算OBV均线
        result["obv_ma"] = pd.Series(result['obv_smooth']).rolling(window=self.ma_period).mean()
        
        # 计算多周期OBV均线
        for period in self.multi_periods:
            if period != self.ma_period:  # 避免重复计算
                result[f"obv_ma{period}"] = pd.Series(result['obv_smooth']).rolling(window=period).mean()
        
        # 计算OBV动量 - 衡量OBV的变化速率
        result["obv_momentum"] = self._calculate_obv_momentum(result['obv_smooth'])
        
        # 计算OBV变化率 - 相对变化百分比
        result["obv_rate"] = result['obv_smooth'].pct_change(periods=5).fillna(0) * 100
        
        # 计算量价相关性
        result["volume_price_corr"] = self._calculate_volume_price_correlation(data, window=20)
        
        # 保存结果
        self._result = result
        
        return result
    
    def _calculate_enhanced_obv(self, data: pd.DataFrame) -> pd.Series:
        """
        计算增强型OBV
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 增强型OBV序列
        """
        # 复制输入数据
        close = data["close"].values
        volume = data["volume"].values
        
        # 初始化OBV
        obv = np.zeros(len(data))
        
        # 计算增强型OBV
        for i in range(1, len(data)):
            # 计算价格变动百分比
            price_change_pct = (close[i] / close[i-1] - 1)
            
            # 应用噪声过滤 - 小于阈值的价格变动忽略
            if abs(price_change_pct) < self.noise_filter:
                obv[i] = obv[i-1]  # 保持不变
            else:
                # 应用灵敏度调整 - 根据价格变动幅度调整成交量权重
                volume_weight = 1.0
                if abs(price_change_pct) > 0.02:  # 大幅变动时增加权重
                    volume_weight = 1.0 + (abs(price_change_pct) - 0.02) * 10 * self.sensitivity
                
                # 计算加权成交量
                weighted_volume = volume[i] * volume_weight
                
                if price_change_pct > 0:  # 价格上涨
                    obv[i] = obv[i-1] + weighted_volume
                else:  # 价格下跌
                    obv[i] = obv[i-1] - weighted_volume
        
        return pd.Series(obv, index=data.index)
    
    def _calculate_obv_momentum(self, obv: pd.Series, period: int = 5) -> pd.Series:
        """
        计算OBV动量
        
        Args:
            obv: OBV序列
            period: 计算周期
            
        Returns:
            pd.Series: OBV动量序列
        """
        # 计算OBV变化率
        obv_change = obv.diff(period)
        
        # 标准化动量（除以OBV绝对值的移动平均，避免除以零）
        obv_abs_ma = obv.abs().rolling(window=period).mean()
        obv_abs_ma = obv_abs_ma.replace(0, 1)  # 避免除以零
        
        obv_momentum = obv_change / obv_abs_ma
        
        return obv_momentum.fillna(0)
    
    def _calculate_volume_price_correlation(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        计算量价相关性
        
        Args:
            data: 输入数据
            window: 滚动窗口大小
            
        Returns:
            pd.Series: 量价相关性序列
        """
        # 计算价格和成交量的变化率
        price_change = data['close'].pct_change()
        volume_change = data['volume'].pct_change()
        
        # 使用滚动窗口计算相关性
        corr = price_change.rolling(window=window).corr(volume_change).fillna(0)
        
        return corr
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强型OBV原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算OBV
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        obv_smooth = self._result['obv_smooth']
        obv_ma = self._result['obv_ma']
        close = self._result['close']
        obv_momentum = self._result['obv_momentum']
        volume_price_corr = self._result['volume_price_corr']
        
        # 1. OBV趋势与价格趋势一致性评分 (权重提高)
        obv_trend = obv_smooth > obv_smooth.shift(5)  # OBV上升趋势
        price_trend = close > close.shift(5)  # 价格上升趋势
        
        # 趋势一致+20分 (原来是+15分)
        trend_consistency = (obv_trend & price_trend) | (~obv_trend & ~price_trend)
        score += trend_consistency * 20
        
        # 2. 优化后的OBV背离评分
        divergence_score = self._calculate_enhanced_divergence(close, obv_smooth)
        score += divergence_score
        
        # 3. 多周期OBV趋势一致性评分 (新增)
        multi_period_score = self._calculate_multi_period_consistency(data)
        score += multi_period_score
        
        # 4. OBV动量评分 (新增)
        # OBV动量为正+10分
        score += (obv_momentum > 0.01) * 10
        # OBV动量为负-10分
        score -= (obv_momentum < -0.01) * 10
        
        # 5. 量价相关性评分 (新增)
        # 量价正相关+10分
        score += (volume_price_corr > 0.5) * 10
        # 量价负相关-5分
        score -= (volume_price_corr < -0.5) * 5
        
        # 6. OBV均线交叉评分 (根据市场环境调整)
        # OBV上穿均线信号
        obv_cross_up_ma = crossover(obv_smooth, obv_ma)
        # OBV下穿均线信号
        obv_cross_down_ma = crossunder(obv_smooth, obv_ma)
        
        # 7. 市场环境感知评分调整
        market_env = self.market_environment
        
        if market_env == 'bull_market':
            # 牛市中，上穿信号更重要
            score += obv_cross_up_ma * 20
            score -= obv_cross_down_ma * 10
        elif market_env == 'bear_market':
            # 熊市中，下穿信号更重要
            score += obv_cross_up_ma * 10
            score -= obv_cross_down_ma * 20
        else:
            # 其他市场，信号权重平衡
            score += obv_cross_up_ma * 15
            score -= obv_cross_down_ma * 15
        
        # 限制得分范围
        return np.clip(score, 0, 100)
    
    def _calculate_enhanced_divergence(self, price: pd.Series, obv: pd.Series) -> pd.Series:
        """
        计算增强型OBV背离评分
        
        Args:
            price: 价格序列
            obv: OBV序列
            
        Returns:
            pd.Series: 背离评分序列
        """
        divergence_score = pd.Series(0.0, index=price.index)
        
        if len(price) < 20:
            return divergence_score
            
        price = price.dropna()
        obv = obv.dropna()
        
        if len(price) < 20 or len(obv) < 20:
            return divergence_score

        # 使用局部极值点寻找更准确的背离
        price_peaks, price_troughs = find_peaks_and_troughs(price.values, window=10)
        obv_peaks, obv_troughs = find_peaks_and_troughs(obv.values, window=10)

        # 检查顶背离（价格创新高，OBV未创新高）
        for i in range(1, len(price_peaks)):
            if i >= len(price_peaks) or price_peaks[i] >= len(price): continue
            
            current_price_peak_idx = price_peaks[i]
            prev_price_peak_idx = price_peaks[i-1]
            
            if price.iloc[current_price_peak_idx] > price.iloc[prev_price_peak_idx]:
                # 寻找对应的OBV高点
                current_obv_peak_idx = -1
                prev_obv_peak_idx = -1
                for p in obv_peaks:
                    if abs(p - current_price_peak_idx) <= 5: current_obv_peak_idx = p
                    if abs(p - prev_price_peak_idx) <= 5: prev_obv_peak_idx = p
                
                if current_obv_peak_idx != -1 and prev_obv_peak_idx != -1:
                    if obv.iloc[current_obv_peak_idx] < obv.iloc[prev_obv_peak_idx]:
                        divergence_score.iloc[current_price_peak_idx:] -= 30
        
        # 检查底背离（价格创新低，OBV未创新低）
        for i in range(1, len(price_troughs)):
            if i >= len(price_troughs) or price_troughs[i] >= len(price): continue

            current_price_trough_idx = price_troughs[i]
            prev_price_trough_idx = price_troughs[i-1]

            if price.iloc[current_price_trough_idx] < price.iloc[prev_price_trough_idx]:
                current_obv_trough_idx = -1
                prev_obv_trough_idx = -1
                for t in obv_troughs:
                    if abs(t - current_price_trough_idx) <= 5: current_obv_trough_idx = t
                    if abs(t - prev_price_trough_idx) <= 5: prev_obv_trough_idx = t

                if current_obv_trough_idx != -1 and prev_obv_trough_idx != -1:
                    if obv.iloc[current_obv_trough_idx] > obv.iloc[prev_obv_trough_idx]:
                        divergence_score.iloc[current_price_trough_idx:] += 30

        return divergence_score.clip(-30, 30)
    
    def _calculate_multi_period_consistency(self, data: pd.DataFrame) -> pd.Series:
        """
        计算多周期OBV趋势一致性评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 多周期一致性评分
        """
        if not self.has_result():
            return pd.Series(0.0, index=data.index)
        
        score = pd.Series(0.0, index=data.index)
        
        obv_smooth = self._result['obv_smooth']
        
        # 不同周期的OBV趋势
        trends = {}
        
        # 计算各周期趋势
        for period in self.multi_periods:
            trends[period] = obv_smooth > obv_smooth.shift(period)
        
        # 检查趋势一致性
        if len(trends) >= 2:
            # 计算趋势一致的比例
            trend_agreement = pd.Series(0, index=data.index, dtype=float)
            
            for i in range(len(data)):
                up_trends = sum(1 for period in self.multi_periods if i >= period and trends[period].iloc[i])
                down_trends = sum(1 for period in self.multi_periods if i >= period and not trends[period].iloc[i])
                
                total_trends = up_trends + down_trends
                if total_trends > 0:
                    # 计算一致性得分 (-1到1)
                    agreement = abs(up_trends - down_trends) / total_trends
                    # 将一致性得分转换为评分
                    trend_agreement.iloc[i] = agreement
            
            # 趋势一致性高加分，最多+15分
            score += trend_agreement * 15
        
        return score
    
    def evaluate_signal_quality(self, signal: pd.Series, data: pd.DataFrame) -> float:
        """
        评估信号质量
        
        Args:
            signal: 信号序列
            data: 输入数据
            
        Returns:
            float: 信号质量得分 (0-1)
        """
        if not self.has_result() or signal.sum() == 0:
            return 0.5
        
        # 获取信号点
        signal_points = signal[signal].index
        if len(signal_points) == 0:
            return 0.5
        
        # 计算OBV多周期一致性
        multi_period_score = self._calculate_multi_period_consistency(data)
        
        # 计算量价相关性
        volume_price_corr = self._result['volume_price_corr']
        
        # 计算OBV动量
        obv_momentum = self._result['obv_momentum']
        
        # 计算信号点的各项指标平均值
        avg_multi_period = multi_period_score.loc[signal_points].mean() / 15.0  # 归一化到0-1
        avg_volume_price_corr = (volume_price_corr.loc[signal_points].mean() + 1) / 2  # 归一化到0-1
        avg_obv_momentum = (obv_momentum.loc[signal_points].clip(-0.1, 0.1) + 0.1) / 0.2  # 归一化到0-1
        
        # 加权平均计算最终质量得分
        quality_score = avg_multi_period * 0.4 + avg_volume_price_corr * 0.3 + avg_obv_momentum * 0.3
        
        return quality_score
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成OBV交易信号
        
        Args:
            data: 输入数据
            *args, **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 信号DataFrame
        """
        # 确保已计算OBV
        if not self.has_result():
            self.calculate(data, *args, **kwargs)
            
        result = self._result.copy()
        obv_smooth = result['obv_smooth']
        obv_ma = result['obv_ma']

        # 创建信号DataFrame
        signals = pd.DataFrame(index=result.index)
        signals['obv'] = result['obv']
        signals['obv_ma'] = result['obv_ma']
        
        # 计算OBV评分
        obv_score = self.calculate_raw_score(data)
        signals['score'] = obv_score
        
        # 生成买入信号
        buy_signal = (
            (signals['score'] > 70) |  # 评分高于70
            (crossover(obv_smooth, obv_ma))  # OBV上穿均线
        )

        # 生成卖出信号
        sell_signal = (
            (signals['score'] < 30) |  # 评分低于30
            (crossunder(obv_smooth, obv_ma))  # OBV下穿均线
        )
        
        # 应用市场环境调整
        market_env = self.market_environment
        if market_env == 'bull_market':
            # 牛市中降低买入门槛，提高卖出门槛
            buy_signal = buy_signal | (signals['score'] > 65)
            sell_signal = sell_signal & (signals['score'] < 25)
        elif market_env == 'bear_market':
            # 熊市中提高买入门槛，降低卖出门槛
            buy_signal = buy_signal & (signals['score'] > 75)
            sell_signal = sell_signal | (signals['score'] < 35)
        
        signals['buy_signal'] = buy_signal
        signals['sell_signal'] = sell_signal
        
        # 计算指标多空趋势
        signals['bull_trend'] = signals['score'] > 60
        signals['bear_trend'] = signals['score'] < 40
        
        # 添加信号强度
        signals['signal_strength'] = self._calculate_signal_strength(result, signals)
        
        return signals
    
    def _calculate_signal_strength(self, result: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """
        计算信号强度
        
        Args:
            result: 指标计算结果
            signals: 信号DataFrame
            
        Returns:
            pd.Series: 信号强度序列
        """
        strength = pd.Series(0.0, index=result.index)  # 中性强度为0
        
        # 买入信号强度
        for i in range(len(result)):
            if signals['buy_signal'].iloc[i]:
                # 根据评分、OBV动量和多周期一致性确定信号强度
                score = signals['score'].iloc[i]
                momentum = result['obv_momentum'].iloc[i] if 'obv_momentum' in result else 0
                
                if score > 85 and momentum > 0.05:
                    strength.iloc[i] = 1.0  # 非常强
                elif score > 75:
                    strength.iloc[i] = 0.8  # 强
                elif score > 65:
                    strength.iloc[i] = 0.6  # 中等
                else:
                    strength.iloc[i] = 0.4  # 弱
            
            elif signals['sell_signal'].iloc[i]:
                # 根据评分、OBV动量和多周期一致性确定信号强度
                score = signals['score'].iloc[i]
                momentum = result['obv_momentum'].iloc[i] if 'obv_momentum' in result else 0
                
                if score < 15 and momentum < -0.05:
                    strength.iloc[i] = -1.0  # 非常强（负向）
                elif score < 25:
                    strength.iloc[i] = -0.8  # 强（负向）
                elif score < 35:
                    strength.iloc[i] = -0.6  # 中等（负向）
                else:
                    strength.iloc[i] = -0.4  # 弱（负向）
        
        return strength
    
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
    
    def calculate_flow_gradient(self) -> pd.DataFrame:
        """
        计算资金流向梯度
        
        Returns:
            pd.DataFrame: 包含资金流向梯度分析的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
        
        obv = self._result['obv']
        obv_smooth = self._result.get('obv_smooth', obv) # 兼容未平滑的情况
        
        # 计算梯度 (一阶导数)
        gradient = obv_smooth.diff()
        
        # 计算加速度 (二阶导数)
        acceleration = gradient.diff()
        
        # 归一化处理
        gradient_std = gradient.rolling(20).std().replace(0, 1)
        normalized_gradient = (gradient / gradient_std).fillna(0)
        
        acceleration_std = acceleration.rolling(20).std().replace(0, 1)
        normalized_acceleration = (acceleration / acceleration_std).fillna(0)
        
        # 创建结果DataFrame
        flow_gradient = pd.DataFrame({
            'gradient': gradient,
            'acceleration': acceleration,
            'normalized_gradient': normalized_gradient,
            'normalized_acceleration': normalized_acceleration
        }, index=obv.index)
        
        # 标记资金流向状态
        flow_gradient['strong_inflow'] = (normalized_gradient > 1.5) & (normalized_acceleration > 0)
        flow_gradient['weak_inflow'] = (normalized_gradient > 0) & (normalized_gradient <= 1.5)
        flow_gradient['strong_outflow'] = (normalized_gradient < -1.5) & (normalized_acceleration < 0)
        flow_gradient['weak_outflow'] = (normalized_gradient < 0) & (normalized_gradient >= -1.5)
        flow_gradient['neutral'] = (normalized_gradient.abs() < 0.5)
        
        return flow_gradient
    
    def detect_divergence(self) -> pd.DataFrame:
        """
        检测OBV与价格之间的背离
        
        Returns:
            pd.DataFrame: 包含背离分析结果的DataFrame
        """
        if self._result is None or self._price_data is None:
            return pd.DataFrame()
        
        obv = self._result.get('obv_smooth', self._result['obv']).ffill()
        price = self._price_data
        
        if len(price) < 20 or len(obv) < 20:
             return pd.DataFrame(columns=[
                'bullish_divergence',
                'bearish_divergence',
                'hidden_bullish_divergence',
                'hidden_bearish_divergence',
                'divergence_strength'
             ])

        # 查找高点和低点
        price_peaks, price_troughs = find_peaks_and_troughs(price.values, window=10)
        obv_peaks, obv_troughs = find_peaks_and_troughs(obv.values, window=10)
        
        # 创建结果DataFrame
        divergence = pd.DataFrame(0, index=price.index, columns=[
            'bullish_divergence',  # 价格创新低但OBV未创新低（看涨）
            'bearish_divergence',  # 价格创新高但OBV未创新高（看跌）
            'hidden_bullish_divergence',  # 价格低点上升但OBV低点下降（看涨）
            'hidden_bearish_divergence',  # 价格高点下降但OBV高点上升（看跌）
            'divergence_strength'  # 背离强度
        ])
        
        # 检测常规背离
        for i in range(1, len(price_peaks)):
            if i >= len(price_peaks) or price_peaks[i] >= len(price): continue
            current_price_peak_idx = price_peaks[i]
            prev_price_peak_idx = price_peaks[i-1]
            
            if price.iloc[current_price_peak_idx] > price.iloc[prev_price_peak_idx]:
                current_obv_peak_idx = -1
                prev_obv_peak_idx = -1
                for p in obv_peaks:
                    if abs(p - current_price_peak_idx) <= 5: current_obv_peak_idx = p
                    if abs(p - prev_price_peak_idx) <= 5: prev_obv_peak_idx = p
                
                if current_obv_peak_idx != -1 and prev_obv_peak_idx != -1 and prev_obv_peak_idx < current_obv_peak_idx:
                    if obv.iloc[current_obv_peak_idx] < obv.iloc[prev_obv_peak_idx]:
                        # 计算背离强度
                        price_change = (price.iloc[current_price_peak_idx] / price.iloc[prev_price_peak_idx]) - 1
                        with np.errstate(divide='ignore', invalid='ignore'):
                            obv_change = (obv.iloc[current_obv_peak_idx] / obv.iloc[prev_obv_peak_idx]) - 1
                        strength = abs(price_change - obv_change) / max(abs(price_change), abs(obv_change), 1e-9)
                        
                        divergence.loc[price.index[current_price_peak_idx], 'bearish_divergence'] = 1
                        divergence.loc[price.index[current_price_peak_idx], 'divergence_strength'] = strength

        for i in range(1, len(price_troughs)):
            if i >= len(price_troughs) or price_troughs[i] >= len(price): continue
            current_price_trough_idx = price_troughs[i]
            prev_price_trough_idx = price_troughs[i-1]

            if price.iloc[current_price_trough_idx] < price.iloc[prev_price_trough_idx]:
                current_obv_trough_idx = -1
                prev_obv_trough_idx = -1
                for t in obv_troughs:
                    if abs(t - current_price_trough_idx) <= 5: current_obv_trough_idx = t
                    if abs(t - prev_price_trough_idx) <= 5: prev_obv_trough_idx = t

                if current_obv_trough_idx != -1 and prev_obv_trough_idx != -1 and prev_obv_trough_idx < current_obv_trough_idx:
                    if obv.iloc[current_obv_trough_idx] > obv.iloc[prev_obv_trough_idx]:
                        price_change = (price.iloc[current_price_trough_idx] / price.iloc[prev_price_trough_idx]) - 1
                        with np.errstate(divide='ignore', invalid='ignore'):
                            obv_change = (obv.iloc[current_obv_trough_idx] / obv.iloc[prev_obv_trough_idx]) - 1
                        strength = abs(price_change - obv_change) / max(abs(price_change), abs(obv_change), 1e-9)
                        
                        divergence.loc[price.index[current_price_trough_idx], 'bullish_divergence'] = 1
                        divergence.loc[price.index[current_price_trough_idx], 'divergence_strength'] = strength

        # 检测隐藏背离
        for i in range(1, len(price_peaks)):
            if i >= len(price_peaks) or price_peaks[i] >= len(price): continue
            current_price_peak_idx = price_peaks[i]
            prev_price_peak_idx = price_peaks[i-1]

            if price.iloc[current_price_peak_idx] < price.iloc[prev_price_peak_idx]:
                current_obv_peak_idx = -1
                prev_obv_peak_idx = -1
                for p in obv_peaks:
                    if abs(p - current_price_peak_idx) <= 5: current_obv_peak_idx = p
                    if abs(p - prev_price_peak_idx) <= 5: prev_obv_peak_idx = p

                if current_obv_peak_idx != -1 and prev_obv_peak_idx != -1 and prev_obv_peak_idx < current_obv_peak_idx:
                    if obv.iloc[current_obv_peak_idx] > obv.iloc[prev_obv_peak_idx]:
                        divergence.loc[price.index[current_price_peak_idx], 'hidden_bearish_divergence'] = 1

        for i in range(1, len(price_troughs)):
            if i >= len(price_troughs) or price_troughs[i] >= len(price): continue
            current_price_trough_idx = price_troughs[i]
            prev_price_trough_idx = price_troughs[i-1]

            if price.iloc[current_price_trough_idx] > price.iloc[prev_price_trough_idx]:
                current_obv_trough_idx = -1
                prev_obv_trough_idx = -1
                for t in obv_troughs:
                    if abs(t - current_price_trough_idx) <= 5: current_obv_trough_idx = t
                    if abs(t - prev_price_trough_idx) <= 5: prev_obv_trough_idx = t

                if current_obv_trough_idx != -1 and prev_obv_trough_idx != -1 and prev_obv_trough_idx < current_obv_trough_idx:
                    if obv.iloc[current_obv_trough_idx] < obv.iloc[prev_obv_trough_idx]:
                        divergence.loc[price.index[current_price_trough_idx], 'hidden_bullish_divergence'] = 1
        
        return divergence
    
    def calculate_price_volume_synergy(self) -> pd.DataFrame:
        """
        计算价格与成交量的协同性
        
        Returns:
            pd.DataFrame: 包含量价协同分析的DataFrame
        """
        if self._result is None or self._price_data is None:
            return pd.DataFrame()
        
        obv = self._result['obv_smooth']
        price = self._price_data
        
        # 计算价格变化率
        price_change = price.pct_change()
        
        # 计算OBV变化率
        obv_change = obv.pct_change()
        
        # 创建结果DataFrame
        synergy = pd.DataFrame(index=price.index)
        
        # 方向一致性：价格和OBV变化方向相同
        synergy['direction_synergy'] = (price_change * obv_change) > 0
        
        # 强度匹配：价格和OBV变化幅度接近
        synergy['magnitude_diff'] = abs(price_change.abs() - obv_change.abs())
        
        # 理想配合：价格上涨时OBV增加更多，价格下跌时OBV减少更少
        synergy['ideal_up'] = (price_change > 0) & (obv_change > price_change)
        synergy['ideal_down'] = (price_change < 0) & (obv_change > price_change)
        
        # 不良配合：价格上涨时OBV增加较少，价格下跌时OBV减少更多
        synergy['poor_up'] = (price_change > 0) & (obv_change < price_change)
        synergy['poor_down'] = (price_change < 0) & (obv_change < price_change)
        
        # 综合协同度评分 (0-100)
        # 基础分50分
        base_score = 50
        
        # 方向一致性加分
        direction_score = np.where(synergy['direction_synergy'], 10, -10)
        
        # 强度匹配评分 (差异越小评分越高)
        magnitude_score = -synergy['magnitude_diff'] * 100
        magnitude_score = magnitude_score.clip(-10, 10)
        
        # 理想/不良配合评分
        ideal_score = np.where(synergy['ideal_up'] | synergy['ideal_down'], 15, 0)
        poor_score = np.where(synergy['poor_up'] | synergy['poor_down'], -15, 0)
        
        # 计算总分
        synergy['synergy_score'] = base_score + direction_score + magnitude_score + ideal_score + poor_score
        
        # 限制分数范围在0-100之间
        synergy['synergy_score'] = synergy['synergy_score'].clip(0, 100)
        
        return synergy
    
    def identify_patterns(self) -> pd.DataFrame:
        """
        识别OBV形态
        
        Returns:
            pd.DataFrame: 包含形态识别结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
        
        obv = self._result.get('obv_smooth', self._result['obv'])
        
        # 获取资金流向数据
        flow = self.calculate_flow_gradient()
        
        # 获取背离数据
        divergence = self.detect_divergence()
        
        # 创建形态DataFrame
        patterns = pd.DataFrame(index=self._result.index)
        
        # 1. OBV突破形态：OBV突破前期高点
        obv_high = obv.rolling(window=20).max().shift(1)
        patterns['obv_breakout'] = obv > obv_high
        
        # 2. OBV跌破形态：OBV跌破前期低点
        obv_low = obv.rolling(window=20).min().shift(1)
        patterns['obv_breakdown'] = obv < obv_low
        
        # 3. OBV加速上涨形态：OBV斜率加速上升
        acceleration = flow['normalized_acceleration']
        patterns['obv_acceleration_up'] = (acceleration > 1) & (acceleration.shift(1) > 0) & (acceleration > acceleration.shift(1))
        
        # 4. OBV加速下跌形态：OBV斜率加速下降
        patterns['obv_acceleration_down'] = (acceleration < -1) & (acceleration.shift(1) < 0) & (acceleration < acceleration.shift(1))
        
        # 5. OBV双底形态：OBV在低位形成W底
        obv_diff = obv.diff()
        w_bottom = (
            (obv_diff.shift(10) < 0) &  # 前期下降
            (obv_diff.shift(5) > 0) &   # 中期上升
            (obv_diff.shift(3) < 0) &   # 近期下降
            (obv_diff > 0)              # 当前上升
        )
        patterns['obv_w_bottom'] = w_bottom
        
        # 6. OBV双顶形态：OBV在高位形成M顶
        m_top = (
            (obv_diff.shift(10) > 0) &  # 前期上升
            (obv_diff.shift(5) < 0) &   # 中期下降
            (obv_diff.shift(3) > 0) &   # 近期上升
            (obv_diff < 0)              # 当前下降
        )
        patterns['obv_m_top'] = m_top
        
        # 7. 指标背离
        patterns['bullish_divergence'] = divergence['bullish_divergence'] == 1
        patterns['bearish_divergence'] = divergence['bearish_divergence'] == 1
        patterns['hidden_bullish_divergence'] = divergence['hidden_bullish_divergence'] == 1
        patterns['hidden_bearish_divergence'] = divergence['hidden_bearish_divergence'] == 1
        
        # 8. OBV横盘整理：OBV波动率低
        obv_volatility = obv.diff().rolling(10).std()
        obv_avg_volatility = obv_volatility.rolling(30).mean()
        patterns['obv_consolidation'] = obv_volatility < (obv_avg_volatility * 0.5)
        
        return patterns

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        if 'ma_period' in kwargs:
            self.ma_period = kwargs['ma_period']
        if 'sensitivity' in kwargs:
            self.sensitivity = kwargs['sensitivity']
        if 'noise_filter' in kwargs:
            self.noise_filter = kwargs['noise_filter']
        if 'multi_periods' in kwargs:
            self.multi_periods = kwargs['multi_periods']
        if 'smooth_period' in kwargs:
            self.smooth_period = kwargs['smooth_period']
        if 'adaptive' in kwargs:
            self.adaptive = kwargs['adaptive']
        if 'use_smoothed_obv' in kwargs:
            self.use_smoothed_obv = kwargs['use_smoothed_obv']
        if 'smoothing_period' in kwargs:
            self.smoothing_period = kwargs['smoothing_period']

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算EnhancedOBV指标的置信度

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
            # 检查EnhancedOBV形态
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

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取EnhancedOBV相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        patterns = pd.DataFrame(index=data.index)

        # 获取OBV数据
        obv_smooth = self._result['obv_smooth']
        obv_ma = self._result['obv_ma']

        # 基本形态
        patterns['OBV_ABOVE_MA'] = obv_smooth > obv_ma
        patterns['OBV_BELOW_MA'] = obv_smooth < obv_ma
        patterns['OBV_RISING'] = obv_smooth > obv_smooth.shift(1)
        patterns['OBV_FALLING'] = obv_smooth < obv_smooth.shift(1)

        # 交叉形态
        patterns['OBV_CROSS_MA_UP'] = crossover(obv_smooth, obv_ma)
        patterns['OBV_CROSS_MA_DOWN'] = crossunder(obv_smooth, obv_ma)

        # 突破形态
        obv_high = obv_smooth.rolling(window=20).max().shift(1)
        obv_low = obv_smooth.rolling(window=20).min().shift(1)
        patterns['OBV_BREAKOUT'] = obv_smooth > obv_high
        patterns['OBV_BREAKDOWN'] = obv_smooth < obv_low

        # 背离形态（简化版）
        if 'obv_momentum' in self._result.columns:
            obv_momentum = self._result['obv_momentum']
            patterns['OBV_BULLISH_MOMENTUM'] = obv_momentum > 0.01
            patterns['OBV_BEARISH_MOMENTUM'] = obv_momentum < -0.01

        return patterns

    def register_patterns(self):
        """
        注册EnhancedOBV指标的形态到全局形态注册表
        """
        # 注册OBV交叉形态
        self.register_pattern_to_registry(
            pattern_id="OBV_CROSS_MA_UP",
            display_name="OBV上穿均线",
            description="OBV从下方穿越移动平均线，表明资金流入增强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_CROSS_MA_DOWN",
            display_name="OBV下穿均线",
            description="OBV从上方穿越移动平均线，表明资金流出增强",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0
        )

        # 注册OBV突破形态
        self.register_pattern_to_registry(
            pattern_id="OBV_BREAKOUT",
            display_name="OBV向上突破",
            description="OBV突破前期高点，表明强势资金流入",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_BREAKDOWN",
            display_name="OBV向下突破",
            description="OBV跌破前期低点，表明强势资金流出",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0
        )

        # 注册OBV动量形态
        self.register_pattern_to_registry(
            pattern_id="OBV_BULLISH_MOMENTUM",
            display_name="OBV看涨动量",
            description="OBV动量为正，表明资金流入加速",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_BEARISH_MOMENTUM",
            display_name="OBV看跌动量",
            description="OBV动量为负，表明资金流出加速",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0
        )

        # 注册OBV背离形态
        self.register_pattern_to_registry(
            pattern_id="OBV_BULLISH_DIVERGENCE",
            display_name="OBV看涨背离",
            description="价格创新低但OBV未创新低，表明下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_BEARISH_DIVERGENCE",
            display_name="OBV看跌背离",
            description="价格创新高但OBV未创新高，表明上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成EnhancedOBV交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, pd.Series]: 包含买卖信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data)

        if self._result is None:
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0.0, index=data.index)
            }

        obv_smooth = self._result['obv_smooth']
        obv_ma = self._result['obv_ma']

        # 生成信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 1. OBV均线交叉信号
        cross_up = crossover(obv_smooth, obv_ma)
        cross_down = crossunder(obv_smooth, obv_ma)

        buy_signal |= cross_up
        sell_signal |= cross_down
        signal_strength += cross_up * 0.7
        signal_strength += cross_down * 0.7

        # 2. OBV突破信号
        obv_high = obv_smooth.rolling(window=20).max().shift(1)
        obv_low = obv_smooth.rolling(window=20).min().shift(1)

        breakout = obv_smooth > obv_high
        breakdown = obv_smooth < obv_low

        buy_signal |= breakout
        sell_signal |= breakdown
        signal_strength += breakout * 0.8
        signal_strength += breakdown * 0.8

        # 3. OBV动量信号
        if 'obv_momentum' in self._result.columns:
            obv_momentum = self._result['obv_momentum']
            strong_momentum_up = obv_momentum > 0.05
            strong_momentum_down = obv_momentum < -0.05

            buy_signal |= strong_momentum_up
            sell_signal |= strong_momentum_down
            signal_strength += strong_momentum_up * 0.6
            signal_strength += strong_momentum_down * 0.6

        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength
        }

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
