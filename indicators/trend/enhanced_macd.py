"""
增强型MACD指标模块

实现改进版的MACD指标，优化计算方法和信号质量，增加多周期适应能力和市场环境感知
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.macd import MACD
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedMACD(MACD):
    """
    增强型MACD指标
    
    在标准MACD基础上增加了动态周期调整、多周期分析、柱状体变化率分析、趋势强度评估等功能
    """
    
    def __init__(self, 
                fast_period: int = 12, 
                slow_period: int = 26, 
                signal_period: int = 9,
                sensitivity: float = 1.0,
                multi_periods: List[Tuple[int, int, int]] = None,
                volume_weighted: bool = False,
                adapt_to_volatility: bool = True):
        """
        初始化增强型MACD指标
        
        Args:
            fast_period: 快速EMA周期，默认为12
            slow_period: 慢速EMA周期，默认为26
            signal_period: 信号线周期，默认为9
            sensitivity: 灵敏度参数，控制对价格变化的响应程度，默认为1.0
            multi_periods: 多周期分析参数，默认为[(8, 17, 9), (12, 26, 9), (24, 52, 18)]
            volume_weighted: 是否使用成交量加权，默认为False
            adapt_to_volatility: 是否根据波动率自适应调整参数，默认为True
        """
        self.indicator_type = "trend"  # 指标类型：趋势类
        super().__init__(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
        self.name = "EnhancedMACD"
        self.description = "增强型MACD指标，优化计算方法和信号质量，增加多周期适应和市场环境感知"
        self.sensitivity = sensitivity
        self.multi_periods = multi_periods or [(8, 17, 9), (12, 26, 9), (24, 52, 18)]
        self.volume_weighted = volume_weighted
        self.adapt_to_volatility = adapt_to_volatility
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型
        """
        return self.indicator_type
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算增强型MACD指标
        
        Args:
            data: 输入数据，包含OHLC和成交量数据
            
        Returns:
            pd.DataFrame: 计算结果，包含MACD及其多周期指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 复制输入数据
        result = data.copy()
        
        # 如果需要自适应波动率，调整参数
        fast_period, slow_period, signal_period = self.fast_period, self.slow_period, self.signal_period
        
        if self.adapt_to_volatility:
            # 计算价格波动率
            volatility = self._calculate_volatility(data["close"])
            # 调整参数
            fast_period, slow_period, signal_period = self._adapt_parameters_to_volatility(
                volatility, self.fast_period, self.slow_period, self.signal_period
            )
        
        # 计算标准MACD（可能使用调整后的参数）
        if self.volume_weighted and "volume" in data.columns:
            self._calculate_volume_weighted_macd(result, fast_period, slow_period, signal_period)
        else:
            self._calculate_macd(result, fast_period, slow_period, signal_period)
        
        # 计算多周期MACD
        for fast, slow, signal in self.multi_periods:
            # 避免重复计算
            if fast == fast_period and slow == slow_period and signal == signal_period:
                continue
                
            if self.volume_weighted and "volume" in data.columns:
                self._calculate_multi_period_volume_weighted_macd(result, fast, slow, signal)
            else:
                self._calculate_multi_period_macd(result, fast, slow, signal)
        
        # 计算MACD柱状体变化率
        result["hist_change_rate"] = self._calculate_histogram_change_rate(result["macd_hist"])
        
        # 计算MACD趋势强度
        result["trend_strength"] = self._calculate_trend_strength(result["macd_hist"])
        
        # 计算MACD零线交叉角度
        result["zero_cross_angle"] = self._calculate_zero_cross_angle(result["macd"])
        
        # 计算MACD与信号线交叉角度
        result["signal_cross_angle"] = self._calculate_signal_cross_angle(result["macd"], result["macd_signal"])
        
        # 计算MACD偏离度
        result["macd_deviation"] = self._calculate_macd_deviation(result["macd"], result["macd_signal"])
        
        # 保存结果
        self._result = result
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强型MACD指标原始评分 (0-100分)
        
        Args:
            data: 输入数据，包含OHLC和成交量数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 确保已计算指标
        if not self.has_result():
            result = self.calculate(data)
        else:
            result = self._result
        
        # 初始化评分，默认为50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 检查结果是否有效
        if result.empty or 'macd' not in result.columns:
            return score
        
        # 1. 基础MACD指标评分 (0-30分)
        macd_score = self._calculate_basic_macd_score(result)
        
        # 2. 多周期MACD一致性评分 (0-20分)
        multi_period_score = self._calculate_multi_period_consistency_score(result)
        
        # 3. 趋势强度评分 (0-20分)
        trend_strength_score = self._calculate_trend_strength_score(result)
        
        # 4. 柱状体变化率评分 (0-15分)
        hist_change_score = self._calculate_hist_change_score(result)
        
        # 5. 交叉角度和偏离度评分 (0-15分)
        cross_angle_score = self._calculate_cross_angle_score(result)
        
        # 综合评分
        final_score = macd_score + multi_period_score + trend_strength_score + hist_change_score + cross_angle_score
        
        # 确保评分在0-100范围内
        final_score = final_score.clip(0, 100)
        
        return final_score
    
    def _calculate_basic_macd_score(self, result: pd.DataFrame) -> pd.Series:
        """
        计算基础MACD指标评分
        
        Args:
            result: 包含MACD指标的DataFrame
            
        Returns:
            pd.Series: 评分序列，取值范围0-30
        """
        # 初始化评分
        score = pd.Series(15.0, index=result.index)  # 默认中性分数
        
        # MACD值与信号线的相对位置
        macd = result['macd']
        signal = result['macd_signal']
        hist = result['macd_hist']
        
        # MACD > 信号线，看涨 (15-30分)
        # MACD < 信号线，看跌 (0-15分)
        for i in range(len(score)):
            if hist.iloc[i] > 0:  # MACD > 信号线
                # MACD和信号线都 > 0，强烈看涨
                if macd.iloc[i] > 0 and signal.iloc[i] > 0:
                    score.iloc[i] = 25 + min(5, hist.iloc[i] * 50)  # 最高30分
                # MACD和信号线都 < 0，但MACD > 信号线，弱看涨
                elif macd.iloc[i] < 0 and signal.iloc[i] < 0:
                    score.iloc[i] = 15 + min(5, hist.iloc[i] * 50)  # 最高20分
                # 其他情况，中等看涨
                else:
                    score.iloc[i] = 20 + min(5, hist.iloc[i] * 50)  # 最高25分
            else:  # MACD < 信号线
                # MACD和信号线都 < 0，强烈看跌
                if macd.iloc[i] < 0 and signal.iloc[i] < 0:
                    score.iloc[i] = 5 - min(5, -hist.iloc[i] * 50)  # 最低0分
                # MACD和信号线都 > 0，但MACD < 信号线，弱看跌
                elif macd.iloc[i] > 0 and signal.iloc[i] > 0:
                    score.iloc[i] = 15 - min(5, -hist.iloc[i] * 50)  # 最低10分
                # 其他情况，中等看跌
                else:
                    score.iloc[i] = 10 - min(5, -hist.iloc[i] * 50)  # 最低5分
        
        # 检测金叉和死叉
        for i in range(1, len(score)):
            # 金叉：MACD从下方穿过信号线
            if macd.iloc[i-1] < signal.iloc[i-1] and macd.iloc[i] >= signal.iloc[i]:
                score.iloc[i] = min(30, score.iloc[i] + 10)  # 加分，但不超过30分
            # 死叉：MACD从上方穿过信号线
            elif macd.iloc[i-1] > signal.iloc[i-1] and macd.iloc[i] <= signal.iloc[i]:
                score.iloc[i] = max(0, score.iloc[i] - 10)  # 减分，但不低于0分
        
        return score
    
    def _calculate_multi_period_consistency_score(self, result: pd.DataFrame) -> pd.Series:
        """
        计算多周期MACD一致性评分
        
        Args:
            result: 包含多周期MACD指标的DataFrame
            
        Returns:
            pd.Series: 评分序列，取值范围0-20
        """
        # 初始化评分
        score = pd.Series(10.0, index=result.index)  # 默认中性分数
        
        # 获取所有周期的MACD柱状图列
        hist_columns = [col for col in result.columns if col.startswith('macd_hist_') or col == 'macd_hist']
        
        if len(hist_columns) <= 1:
            return score  # 如果没有多周期数据，返回默认分数
        
        # 计算各周期MACD柱状图的一致性
        for i in range(len(score)):
            positive_count = 0
            negative_count = 0
            
            # 统计各周期MACD柱状图的正负值
            for col in hist_columns:
                if result[col].iloc[i] > 0:
                    positive_count += 1
                elif result[col].iloc[i] < 0:
                    negative_count += 1
            
            total_periods = len(hist_columns)
            
            # 计算一致性比例
            if positive_count > negative_count:
                consistency_ratio = positive_count / total_periods
                # 看涨一致性评分 (10-20分)
                score.iloc[i] = 10 + consistency_ratio * 10
            elif negative_count > positive_count:
                consistency_ratio = negative_count / total_periods
                # 看跌一致性评分 (0-10分)
                score.iloc[i] = 10 - consistency_ratio * 10
            # 如果正负相等，保持中性评分10分
        
        return score
    
    def _calculate_trend_strength_score(self, result: pd.DataFrame) -> pd.Series:
        """
        计算趋势强度评分
        
        Args:
            result: 包含趋势强度指标的DataFrame
            
        Returns:
            pd.Series: 评分序列，取值范围0-20
        """
        # 初始化评分
        score = pd.Series(10.0, index=result.index)  # 默认中性分数
        
        if 'trend_strength' not in result.columns or 'macd_hist' not in result.columns:
            return score
        
        trend_strength = result['trend_strength']
        hist = result['macd_hist']
        
        # 基于趋势强度和柱状图方向评分
        for i in range(len(score)):
            # 归一化趋势强度到0-10
            normalized_strength = min(1.0, trend_strength.iloc[i] / 0.5) * 10
            
            if hist.iloc[i] > 0:  # 柱状图为正，上升趋势
                # 趋势强度评分 (10-20分)
                score.iloc[i] = 10 + normalized_strength
            else:  # 柱状图为负，下降趋势
                # 趋势强度评分 (0-10分)
                score.iloc[i] = 10 - normalized_strength
        
        return score
    
    def _calculate_hist_change_score(self, result: pd.DataFrame) -> pd.Series:
        """
        计算柱状体变化率评分
        
        Args:
            result: 包含柱状体变化率的DataFrame
            
        Returns:
            pd.Series: 评分序列，取值范围0-15
        """
        # 初始化评分
        score = pd.Series(7.5, index=result.index)  # 默认中性分数
        
        if 'hist_change_rate' not in result.columns or 'macd_hist' not in result.columns:
            return score
        
        change_rate = result['hist_change_rate']
        hist = result['macd_hist']
        
        # 柱状体变化率评分
        for i in range(len(score)):
            # 归一化变化率到0-7.5
            normalized_change = min(1.0, abs(change_rate.iloc[i]) / 0.2) * 7.5
            
            if hist.iloc[i] > 0 and change_rate.iloc[i] > 0:  # 正柱状体且增长
                # 强烈看涨 (7.5-15分)
                score.iloc[i] = 7.5 + normalized_change
            elif hist.iloc[i] < 0 and change_rate.iloc[i] < 0:  # 负柱状体且减小
                # 强烈看跌 (0-7.5分)
                score.iloc[i] = 7.5 - normalized_change
            elif hist.iloc[i] > 0 and change_rate.iloc[i] < 0:  # 正柱状体但减小
                # 弱看涨 (7.5-11.25分)
                score.iloc[i] = 7.5 + normalized_change / 2
            elif hist.iloc[i] < 0 and change_rate.iloc[i] > 0:  # 负柱状体但增长
                # 弱看跌 (3.75-7.5分)
                score.iloc[i] = 7.5 - normalized_change / 2
        
        return score
    
    def _calculate_cross_angle_score(self, result: pd.DataFrame) -> pd.Series:
        """
        计算交叉角度和偏离度评分
        
        Args:
            result: 包含交叉角度和偏离度的DataFrame
            
        Returns:
            pd.Series: 评分序列，取值范围0-15
        """
        # 初始化评分
        score = pd.Series(7.5, index=result.index)  # 默认中性分数
        
        required_columns = ['zero_cross_angle', 'signal_cross_angle', 'macd_deviation', 'macd_hist']
        if not all(col in result.columns for col in required_columns):
            return score
        
        zero_cross_angle = result['zero_cross_angle']
        signal_cross_angle = result['signal_cross_angle']
        macd_deviation = result['macd_deviation']
        hist = result['macd_hist']
        
        # 交叉角度和偏离度评分
        for i in range(len(score)):
            # 角度分数 (0-7.5分)
            normalized_zero_angle = min(1.0, abs(zero_cross_angle.iloc[i]) / 45) * 3.75
            normalized_signal_angle = min(1.0, abs(signal_cross_angle.iloc[i]) / 45) * 3.75
            
            # 偏离度分数 (0-7.5分)
            normalized_deviation = min(1.0, abs(macd_deviation.iloc[i]) / 0.5) * 7.5
            
            # 根据柱状体方向和各指标方向计算分数
            if hist.iloc[i] > 0:  # 柱状体为正，看涨
                # 计算加权平均分数 (7.5-15分)
                angle_score = 7.5
                if zero_cross_angle.iloc[i] > 0:
                    angle_score += normalized_zero_angle
                if signal_cross_angle.iloc[i] > 0:
                    angle_score += normalized_signal_angle
                
                deviation_score = 7.5
                if macd_deviation.iloc[i] > 0:
                    deviation_score += normalized_deviation
                else:
                    deviation_score -= normalized_deviation
                
                # 综合分数
                score.iloc[i] = (angle_score + deviation_score) / 2
            else:  # 柱状体为负，看跌
                # 计算加权平均分数 (0-7.5分)
                angle_score = 7.5
                if zero_cross_angle.iloc[i] < 0:
                    angle_score -= normalized_zero_angle
                if signal_cross_angle.iloc[i] < 0:
                    angle_score -= normalized_signal_angle
                
                deviation_score = 7.5
                if macd_deviation.iloc[i] < 0:
                    deviation_score -= normalized_deviation
                else:
                    deviation_score += normalized_deviation
                
                # 综合分数
                score.iloc[i] = (angle_score + deviation_score) / 2
        
        return score
    
    def _calculate_macd(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> None:
        """
        计算MACD指标
        
        Args:
            data: 输入数据
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
        """
        # 计算快速和慢速EMA
        fast_ema = data["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data["close"].ewm(span=slow_period, adjust=False).mean()
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            # 调整快速EMA的响应度
            center = slow_ema
            fast_ema = center + (fast_ema - center) * self.sensitivity
        
        # 计算MACD线 = 快速EMA - 慢速EMA
        macd = fast_ema - slow_ema
        
        # 计算信号线 = MACD的EMA
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图 = MACD线 - 信号线
        hist = macd - signal
        
        # 保存结果
        data["fast_ema"] = fast_ema
        data["slow_ema"] = slow_ema
        data["macd"] = macd
        data["macd_signal"] = signal
        data["macd_hist"] = hist
    
    def _calculate_volume_weighted_macd(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> None:
        """
        计算成交量加权MACD指标
        
        Args:
            data: 输入数据
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
        """
        # 确保数据包含成交量列
        if "volume" not in data.columns:
            logger.warning("数据中不包含成交量列，使用标准MACD计算")
            self._calculate_macd(data, fast_period, slow_period, signal_period)
            return
        
        # 计算成交量归一化（确保成交量变化不会过大影响价格）
        volume_normalized = data["volume"] / data["volume"].rolling(window=20).mean()
        volume_normalized = volume_normalized.clip(0.5, 2.0)  # 限制范围，避免极端值
        
        # 价格乘以归一化成交量
        price_volume = data["close"] * volume_normalized
        
        # 计算快速和慢速EMA（使用成交量加权价格）
        fast_ema = price_volume.ewm(span=fast_period, adjust=False).mean()
        slow_ema = price_volume.ewm(span=slow_period, adjust=False).mean()
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            center = slow_ema
            fast_ema = center + (fast_ema - center) * self.sensitivity
        
        # 计算MACD线
        macd = fast_ema - slow_ema
        
        # 计算信号线
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        hist = macd - signal
        
        # 保存结果
        data["volume_fast_ema"] = fast_ema
        data["volume_slow_ema"] = slow_ema
        data["macd"] = macd
        data["macd_signal"] = signal
        data["macd_hist"] = hist
    
    def _calculate_multi_period_macd(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> None:
        """
        计算多周期MACD指标
        
        Args:
            data: 输入数据
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
        """
        # 计算快速和慢速EMA
        fast_ema = data["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data["close"].ewm(span=slow_period, adjust=False).mean()
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            center = slow_ema
            fast_ema = center + (fast_ema - center) * self.sensitivity
        
        # 计算MACD线
        macd = fast_ema - slow_ema
        
        # 计算信号线
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        hist = macd - signal
        
        # 保存结果 - 使用周期作为后缀
        period_suffix = f"_{fast_period}_{slow_period}_{signal_period}"
        data[f"fast_ema{period_suffix}"] = fast_ema
        data[f"slow_ema{period_suffix}"] = slow_ema
        data[f"macd{period_suffix}"] = macd
        data[f"macd_signal{period_suffix}"] = signal
        data[f"macd_hist{period_suffix}"] = hist
    
    def _calculate_multi_period_volume_weighted_macd(self, data: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> None:
        """
        计算多周期成交量加权MACD指标
        
        Args:
            data: 输入数据
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
        """
        # 确保数据包含成交量列
        if "volume" not in data.columns:
            logger.warning("数据中不包含成交量列，使用标准MACD计算")
            self._calculate_multi_period_macd(data, fast_period, slow_period, signal_period)
            return
        
        # 计算成交量归一化
        volume_normalized = data["volume"] / data["volume"].rolling(window=20).mean()
        volume_normalized = volume_normalized.clip(0.5, 2.0)
        
        # 价格乘以归一化成交量
        price_volume = data["close"] * volume_normalized
        
        # 计算快速和慢速EMA
        fast_ema = price_volume.ewm(span=fast_period, adjust=False).mean()
        slow_ema = price_volume.ewm(span=slow_period, adjust=False).mean()
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            center = slow_ema
            fast_ema = center + (fast_ema - center) * self.sensitivity
        
        # 计算MACD线
        macd = fast_ema - slow_ema
        
        # 计算信号线
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        hist = macd - signal
        
        # 保存结果 - 使用周期作为后缀
        period_suffix = f"_{fast_period}_{slow_period}_{signal_period}"
        data[f"volume_fast_ema{period_suffix}"] = fast_ema
        data[f"volume_slow_ema{period_suffix}"] = slow_ema
        data[f"macd{period_suffix}"] = macd
        data[f"macd_signal{period_suffix}"] = signal
        data[f"macd_hist{period_suffix}"] = hist
    
    def _calculate_volatility(self, price_series: pd.Series, window: int = 20) -> float:
        """
        计算价格波动率
        
        Args:
            price_series: 价格序列
            window: 计算窗口
            
        Returns:
            float: 波动率
        """
        # 计算价格百分比变化
        returns = price_series.pct_change()
        
        # 计算波动率（标准差）
        volatility = returns.rolling(window=window).std().iloc[-1]
        
        return volatility
    
    def _adapt_parameters_to_volatility(self, volatility: float, 
                                      fast_period: int, 
                                      slow_period: int, 
                                      signal_period: int) -> Tuple[int, int, int]:
        """
        根据波动率调整MACD参数
        
        Args:
            volatility: 波动率
            fast_period: 原快速EMA周期
            slow_period: 原慢速EMA周期
            signal_period: 原信号线周期
            
        Returns:
            Tuple[int, int, int]: 调整后的参数 (fast_period, slow_period, signal_period)
        """
        # 定义波动率阈值
        low_volatility = 0.01
        high_volatility = 0.03
        
        # 针对不同波动率环境调整参数
        if volatility < low_volatility:
            # 低波动率环境 - 使用较短周期，增加灵敏度
            adjusted_fast = max(6, int(fast_period * 0.7))
            adjusted_slow = max(13, int(slow_period * 0.8))
            adjusted_signal = max(5, int(signal_period * 0.8))
        elif volatility > high_volatility:
            # 高波动率环境 - 使用较长周期，减少噪音
            adjusted_fast = int(fast_period * 1.3)
            adjusted_slow = int(slow_period * 1.2)
            adjusted_signal = int(signal_period * 1.2)
        else:
            # 中等波动率 - 使用原始参数
            adjusted_fast = fast_period
            adjusted_slow = slow_period
            adjusted_signal = signal_period
        
        logger.debug(f"根据波动率({volatility:.4f})调整MACD参数: {fast_period}->{adjusted_fast}, {slow_period}->{adjusted_slow}, {signal_period}->{adjusted_signal}")
        
        return adjusted_fast, adjusted_slow, adjusted_signal
    
    def _calculate_histogram_change_rate(self, hist: pd.Series, window: int = 3) -> pd.Series:
        """
        计算MACD柱状体变化率
        
        Args:
            hist: MACD柱状体序列
            window: 计算窗口
            
        Returns:
            pd.Series: 柱状体变化率
        """
        # 计算柱状体一阶差分
        hist_diff = hist.diff(periods=1)
        
        # 计算变化率（相对于柱状体绝对值）
        hist_abs = hist.abs()
        change_rate = hist_diff / (hist_abs + 1e-10)  # 避免除以0
        
        # 使用移动平均平滑变化率
        change_rate_smooth = change_rate.rolling(window=window).mean()
        
        return change_rate_smooth
    
    def _calculate_trend_strength(self, hist: pd.Series, window: int = 14) -> pd.Series:
        """
        计算MACD趋势强度
        
        Args:
            hist: MACD柱状体序列
            window: 计算窗口
            
        Returns:
            pd.Series: 趋势强度
        """
        # 计算柱状体在窗口内的一致性
        # 正值表示上升趋势，负值表示下降趋势，绝对值表示强度
        trend_strength = pd.Series(0.0, index=hist.index)
        
        for i in range(window, len(hist)):
            window_hist = hist.iloc[i-window:i]
            
            # 计算正柱状体和负柱状体的比例
            positive_ratio = (window_hist > 0).mean()
            negative_ratio = (window_hist < 0).mean()
            
            # 计算趋势强度
            if positive_ratio > 0.5:  # 上升趋势
                strength = positive_ratio * 2 - 1  # 映射到 0~1 范围
                trend_strength.iloc[i] = strength
            elif negative_ratio > 0.5:  # 下降趋势
                strength = negative_ratio * 2 - 1  # 映射到 0~1 范围
                trend_strength.iloc[i] = -strength
            else:  # 无明显趋势
                trend_strength.iloc[i] = 0
        
        return trend_strength
    
    def _calculate_zero_cross_angle(self, macd: pd.Series) -> pd.Series:
        """
        计算MACD零线交叉角度
        
        Args:
            macd: MACD序列
            
        Returns:
            pd.Series: 零线交叉角度
        """
        # 计算MACD斜率
        macd_slope = macd.diff(periods=1)
        
        # 计算角度（近似值，用斜率表示）
        angle = macd_slope.copy()
        
        # 只关注零线交叉点附近
        zero_cross = (macd > 0) != (macd.shift(1) > 0)
        non_cross_indices = ~zero_cross
        angle.loc[non_cross_indices] = 0
        
        return angle
    
    def _calculate_signal_cross_angle(self, macd: pd.Series, signal: pd.Series) -> pd.Series:
        """
        计算MACD与信号线交叉角度
        
        Args:
            macd: MACD序列
            signal: 信号线序列
            
        Returns:
            pd.Series: 信号线交叉角度
        """
        # 计算MACD和信号线的斜率
        macd_slope = macd.diff(periods=1)
        signal_slope = signal.diff(periods=1)
        
        # 计算交叉角度（用斜率差表示）
        angle = macd_slope - signal_slope
        
        return angle
    
    def _calculate_macd_deviation(self, macd: pd.Series, signal: pd.Series, window: int = 20) -> pd.Series:
        """
        计算MACD偏离度（MACD与信号线的偏离程度）
        
        Args:
            macd: MACD序列
            signal: 信号线序列
            window: 计算窗口
            
        Returns:
            pd.Series: MACD偏离度
        """
        # 计算MACD与信号线的差距
        diff = macd - signal
        
        # 计算历史标准差
        std = diff.rolling(window=window).std()
        
        # 计算偏离度（标准化）
        deviation = diff / (std + 1e-10)
        
        return deviation

    def identify_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        识别MACD相关形态
        
        当前版本暂未实现详细的形态识别，返回一个空的DataFrame。
        后续可在此基础上扩展，如识别金叉、死叉、背离等。
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 形态识别结果
        """
        raise ValueError("This is a test to see if this method is called.")
        if self._result is not None:
            return pd.DataFrame(index=self._result.index)
        return pd.DataFrame(index=data.index) 