"""
增强型能量潮(OBV)指标模块

实现改进版的能量潮指标，优化计算方法和信号质量，增加多周期适应能力和市场环境感知
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.obv import OBV
from utils.logger import get_logger
from utils.technical_utils import find_peaks_and_troughs

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
                adaptive: bool = True):
        """
        初始化增强型OBV指标
        
        Args:
            ma_period: OBV均线周期，默认为30日
            sensitivity: 灵敏度参数，控制对价格变化的响应程度，默认为1.0
            noise_filter: 噪声过滤阈值，价格变动小于该比例时视为无变动，默认为0.5%
            multi_periods: 多周期分析参数，默认为[5, 10, 20, 60]
            smooth_period: 平滑周期
            adaptive: 是否启用自适应模式
        """
        super().__init__(ma_period=ma_period)
        self.name = "EnhancedOBV"
        self.description = "增强型能量潮指标，优化计算方法和信号质量，增加多周期适应和市场环境感知"
        self.sensitivity = sensitivity
        self.noise_filter = noise_filter
        self.multi_periods = multi_periods or [5, 10, 20, 60]
        self.indicator_type = "volume"  # 指标类型：量价类
        self.smooth_period = smooth_period
        self.adaptive = adaptive
        self.market_environment = "normal"
        
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
        self.ensure_columns(data, ["close", "volume"])
        
        # 复制输入数据
        result = data.copy()
        
        # 计算增强型OBV - 使用修正的价格变动方向，增加灵敏度和噪声过滤
        obv = self._calculate_enhanced_obv(result)
        result["obv"] = obv
        
        # 计算OBV均线
        result["obv_ma"] = pd.Series(obv).rolling(window=self.ma_period).mean().values
        
        # 计算多周期OBV均线
        for period in self.multi_periods:
            if period != self.ma_period:  # 避免重复计算
                result[f"obv_ma{period}"] = pd.Series(obv).rolling(window=period).mean().values
        
        # 计算OBV动量 - 衡量OBV的变化速率
        result["obv_momentum"] = self._calculate_obv_momentum(obv)
        
        # 计算OBV变化率 - 相对变化百分比
        result["obv_rate"] = obv.pct_change(periods=5).fillna(0) * 100
        
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
        obv_change = obv - obv.shift(period)
        
        # 标准化动量（除以OBV绝对值的移动平均，避免除以零）
        obv_abs_ma = obv.abs().rolling(window=period).mean()
        obv_abs_ma = obv_abs_ma.replace(0, 1)  # 避免除以零
        
        obv_momentum = obv_change / obv_abs_ma
        
        return obv_momentum
    
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
        corr = pd.Series(index=data.index)
        
        for i in range(window, len(data)):
            if i >= window:
                window_price_change = price_change.iloc[i-window+1:i+1]
                window_volume_change = volume_change.iloc[i-window+1:i+1]
                corr.iloc[i] = window_price_change.corr(window_volume_change)
        
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
        
        obv = self._result['obv']
        obv_ma = self._result['obv_ma']
        close = self._result['close']
        obv_momentum = self._result['obv_momentum']
        volume_price_corr = self._result['volume_price_corr']
        
        # 1. OBV趋势与价格趋势一致性评分 (权重提高)
        obv_trend = obv > obv.shift(5)  # OBV上升趋势
        price_trend = close > close.shift(5)  # 价格上升趋势
        
        # 趋势一致+20分 (原来是+15分)
        trend_consistency = (obv_trend & price_trend) | (~obv_trend & ~price_trend)
        score += trend_consistency * 20
        
        # 2. 优化后的OBV背离评分
        divergence_score = self._calculate_enhanced_divergence(close, obv)
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
        obv_cross_up_ma = self.crossover(obv, obv_ma)
        # OBV下穿均线信号
        obv_cross_down_ma = self.crossunder(obv, obv_ma)
        
        # 7. 市场环境感知评分调整
        market_env = self.get_market_environment()
        
        if market_env == MarketEnvironment.BULL_MARKET:
            # 牛市中，上穿信号更重要
            score += obv_cross_up_ma * 20
            score -= obv_cross_down_ma * 10
        elif market_env == MarketEnvironment.BEAR_MARKET:
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
        
        # 使用局部极值点寻找更准确的背离
        price_highs = []
        price_lows = []
        obv_highs = []
        obv_lows = []
        
        # 寻找局部极值点
        window = 5
        for i in range(window, len(price) - window):
            # 价格高点
            if price.iloc[i] > price.iloc[i-1] and price.iloc[i] > price.iloc[i+1] and \
               price.iloc[i] == price.iloc[i-window:i+window+1].max():
                price_highs.append(i)
                
            # 价格低点
            if price.iloc[i] < price.iloc[i-1] and price.iloc[i] < price.iloc[i+1] and \
               price.iloc[i] == price.iloc[i-window:i+window+1].min():
                price_lows.append(i)
                
            # OBV高点
            if obv.iloc[i] > obv.iloc[i-1] and obv.iloc[i] > obv.iloc[i+1] and \
               obv.iloc[i] == obv.iloc[i-window:i+window+1].max():
                obv_highs.append(i)
                
            # OBV低点
            if obv.iloc[i] < obv.iloc[i-1] and obv.iloc[i] < obv.iloc[i+1] and \
               obv.iloc[i] == obv.iloc[i-window:i+window+1].min():
                obv_lows.append(i)
        
        # 检查顶背离（价格创新高，OBV未创新高）
        for i in price_highs:
            if i > window * 2:
                # 获取之前的高点
                prev_highs = [ph for ph in price_highs if ph < i and i - ph <= 20]
                if prev_highs:
                    prev_high = prev_highs[-1]
                    # 价格创新高
                    if price.iloc[i] > price.iloc[prev_high]:
                        # 检查OBV是否也创新高
                        obv_new_high = False
                        for oh in obv_highs:
                            if abs(oh - i) <= window:  # OBV高点与价格高点接近
                                obv_new_high = True
                                # 检查OBV是否相对之前高点更高
                                prev_obv_highs = [povh for povh in obv_highs if povh < oh and oh - povh <= 20]
                                if prev_obv_highs and obv.iloc[oh] <= obv.iloc[prev_obv_highs[-1]]:
                                    # OBV未创新高，顶背离
                                    divergence_score.iloc[i:i+15] -= 30
                                break
        
        # 检查底背离（价格创新低，OBV未创新低）
        for i in price_lows:
            if i > window * 2:
                # 获取之前的低点
                prev_lows = [pl for pl in price_lows if pl < i and i - pl <= 20]
                if prev_lows:
                    prev_low = prev_lows[-1]
                    # 价格创新低
                    if price.iloc[i] < price.iloc[prev_low]:
                        # 检查OBV是否也创新低
                        obv_new_low = False
                        for ol in obv_lows:
                            if abs(ol - i) <= window:  # OBV低点与价格低点接近
                                obv_new_low = True
                                # 检查OBV是否相对之前低点更低
                                prev_obv_lows = [povl for povl in obv_lows if povl < ol and ol - povl <= 20]
                                if prev_obv_lows and obv.iloc[ol] >= obv.iloc[prev_obv_lows[-1]]:
                                    # OBV未创新低，底背离
                                    divergence_score.iloc[i:i+15] += 30
                                break
        
        return divergence_score
    
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
        
        obv = self._result['obv']
        
        # 不同周期的OBV趋势
        trends = {}
        
        # 计算各周期趋势
        for period in self.multi_periods:
            trends[period] = obv > obv.shift(period)
        
        # 检查趋势一致性
        if len(trends) >= 2:
            # 计算趋势一致的比例
            trend_agreement = pd.Series(0, index=data.index)
            
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
            (self.crossover(result['obv'], result['obv_ma']))  # OBV上穿均线
        )
        
        # 生成卖出信号
        sell_signal = (
            (signals['score'] < 30) |  # 评分低于30
            (self.crossunder(result['obv'], result['obv_ma']))  # OBV下穿均线
        )
        
        # 应用市场环境调整
        market_env = self.get_market_environment()
        if market_env == MarketEnvironment.BULL_MARKET:
            # 牛市中降低买入门槛，提高卖出门槛
            buy_signal = buy_signal | (signals['score'] > 65)
            sell_signal = sell_signal & (signals['score'] < 25)
        elif market_env == MarketEnvironment.BEAR_MARKET:
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
        strength = pd.Series(SignalStrength.NEUTRAL.value, index=result.index)
        
        # 买入信号强度
        for i in range(len(result)):
            if signals['buy_signal'].iloc[i]:
                # 根据评分、OBV动量和多周期一致性确定信号强度
                score = signals['score'].iloc[i]
                momentum = result['obv_momentum'].iloc[i] if 'obv_momentum' in result else 0
                
                if score > 85 and momentum > 0.05:
                    strength.iloc[i] = SignalStrength.VERY_STRONG.value
                elif score > 75:
                    strength.iloc[i] = SignalStrength.STRONG.value
                elif score > 65:
                    strength.iloc[i] = SignalStrength.MODERATE.value
                else:
                    strength.iloc[i] = SignalStrength.WEAK.value
            
            elif signals['sell_signal'].iloc[i]:
                # 根据评分、OBV动量和多周期一致性确定信号强度
                score = signals['score'].iloc[i]
                momentum = result['obv_momentum'].iloc[i] if 'obv_momentum' in result else 0
                
                if score < 15 and momentum < -0.05:
                    strength.iloc[i] = SignalStrength.VERY_STRONG_NEGATIVE.value
                elif score < 25:
                    strength.iloc[i] = SignalStrength.STRONG_NEGATIVE.value
                elif score < 35:
                    strength.iloc[i] = SignalStrength.MODERATE_NEGATIVE.value
                else:
                    strength.iloc[i] = SignalStrength.WEAK_NEGATIVE.value
        
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
        obv_smooth = self._result['obv_smooth']
        
        # 计算梯度 (一阶导数)
        gradient = obv_smooth.diff()
        
        # 计算加速度 (二阶导数)
        acceleration = gradient.diff()
        
        # 归一化处理
        gradient_std = gradient.rolling(20).std()
        normalized_gradient = gradient / gradient_std
        normalized_gradient = normalized_gradient.fillna(0)
        
        acceleration_std = acceleration.rolling(20).std()
        normalized_acceleration = acceleration / acceleration_std
        normalized_acceleration = normalized_acceleration.fillna(0)
        
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
        
        obv = self._result['obv_smooth'].fillna(method='ffill')
        price = self._price_data
        
        # 查找高点和低点
        price_peaks = find_peaks_and_troughs(price, window=10, peak_type='peak')
        price_troughs = find_peaks_and_troughs(price, window=10, peak_type='trough')
        obv_peaks = find_peaks_and_troughs(obv, window=10, peak_type='peak')
        obv_troughs = find_peaks_and_troughs(obv, window=10, peak_type='trough')
        
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
            if i >= len(price) or i >= len(obv_peaks):
                continue
                
            # 看跌背离：价格创新高但OBV未创新高
            if (price.iloc[price_peaks[i]] > price.iloc[price_peaks[i-1]] and 
                obv.iloc[obv_peaks[i]] < obv.iloc[obv_peaks[i-1]]):
                
                # 计算背离强度
                price_change = (price.iloc[price_peaks[i]] / price.iloc[price_peaks[i-1]]) - 1
                obv_change = (obv.iloc[obv_peaks[i]] / obv.iloc[obv_peaks[i-1]]) - 1
                strength = abs(price_change - obv_change) / max(abs(price_change), abs(obv_change))
                
                divergence.loc[price.index[price_peaks[i]], 'bearish_divergence'] = 1
                divergence.loc[price.index[price_peaks[i]], 'divergence_strength'] = strength
        
        for i in range(1, len(price_troughs)):
            if i >= len(price) or i >= len(obv_troughs):
                continue
                
            # 看涨背离：价格创新低但OBV未创新低
            if (price.iloc[price_troughs[i]] < price.iloc[price_troughs[i-1]] and 
                obv.iloc[obv_troughs[i]] > obv.iloc[obv_troughs[i-1]]):
                
                # 计算背离强度
                price_change = (price.iloc[price_troughs[i]] / price.iloc[price_troughs[i-1]]) - 1
                obv_change = (obv.iloc[obv_troughs[i]] / obv.iloc[obv_troughs[i-1]]) - 1
                strength = abs(price_change - obv_change) / max(abs(price_change), abs(obv_change))
                
                divergence.loc[price.index[price_troughs[i]], 'bullish_divergence'] = 1
                divergence.loc[price.index[price_troughs[i]], 'divergence_strength'] = strength
        
        # 检测隐藏背离
        for i in range(1, len(price_peaks)):
            if i >= len(price) or i >= len(obv_peaks):
                continue
                
            # 隐藏看跌背离：价格高点下降但OBV高点上升
            if (price.iloc[price_peaks[i]] < price.iloc[price_peaks[i-1]] and 
                obv.iloc[obv_peaks[i]] > obv.iloc[obv_peaks[i-1]]):
                
                divergence.loc[price.index[price_peaks[i]], 'hidden_bearish_divergence'] = 1
        
        for i in range(1, len(price_troughs)):
            if i >= len(price) or i >= len(obv_troughs):
                continue
                
            # 隐藏看涨背离：价格低点上升但OBV低点下降
            if (price.iloc[price_troughs[i]] > price.iloc[price_troughs[i-1]] and 
                obv.iloc[obv_troughs[i]] < obv.iloc[obv_troughs[i-1]]):
                
                divergence.loc[price.index[price_troughs[i]], 'hidden_bullish_divergence'] = 1
        
        return divergence
    
    def calculate_price_volume_synergy(self) -> pd.DataFrame:
        """
        计算价格与成交量的协同性
        
        Returns:
            pd.DataFrame: 包含量价协同分析的DataFrame
        """
        if self._result is None or self._price_data is None:
            return pd.DataFrame()
        
        obv = self._result['obv']
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
        
        obv = self._result['obv_smooth']
        obv_ma10 = self._result['obv_ma10']
        obv_ma20 = self._result['obv_ma20']
        
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