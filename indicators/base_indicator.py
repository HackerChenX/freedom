"""
技术指标基类模块

提供技术指标计算的通用接口和功能
"""

import abc
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class MarketEnvironment(Enum):
    """市场环境枚举"""
    BULL_MARKET = "牛市"
    BEAR_MARKET = "熊市"
    SIDEWAYS_MARKET = "震荡市"
    VOLATILE_MARKET = "高波动市"
    BREAKOUT_MARKET = "突破市场"  # 添加突破市场类型


class SignalStrength(Enum):
    """信号强度枚举"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NEUTRAL = 0
    VERY_WEAK_NEGATIVE = -1
    WEAK_NEGATIVE = -2
    MODERATE_NEGATIVE = -3
    STRONG_NEGATIVE = -4
    VERY_STRONG_NEGATIVE = -5


class BaseIndicator(abc.ABC):
    """
    技术指标基类
    
    所有技术指标类应继承此类，并实现必要的抽象方法
    """
    
    def __init__(self, name: str = "", description: str = "", weight: float = 1.0):
        """
        初始化技术指标
        
        Args:
            name: 指标名称，可选参数，如果未提供则使用子类的name属性
            description: 指标描述
            weight: 指标权重，用于综合评分
        """
        # 如果未提供name，则尝试使用子类中定义的name属性
        if not name and hasattr(self, 'name'):
            pass  # 已经有name属性，不需要重新赋值
        else:
            self.name = name
            
        self.description = description
        self.weight = weight
        self._result = None
        self._error = None
        self._score_cache = {}
        self._market_environment = MarketEnvironment.SIDEWAYS_MARKET  # 默认市场环境
    
    @property
    def result(self) -> Optional[pd.DataFrame]:
        """获取计算结果"""
        return self._result
    
    @property
    def error(self) -> Optional[Exception]:
        """获取错误信息"""
        return self._error
    
    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return self._result is not None
    
    def has_error(self) -> bool:
        """检查是否有错误"""
        return self._error is not None
    
    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: 输入数据，通常是K线数据
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 计算结果
        """
        pass
    
    def set_market_environment(self, market_env: MarketEnvironment) -> None:
        """
        设置当前市场环境
        
        Args:
            market_env: 市场环境
        """
        self._market_environment = market_env
        logger.debug(f"设置 {self.name} 的市场环境为: {market_env.value}")
    
    def get_market_environment(self) -> MarketEnvironment:
        """
        获取当前市场环境
        
        Returns:
            MarketEnvironment: 当前市场环境
        """
        return self._market_environment
    
    def evaluate_signal_quality(self, signal: pd.Series, data: pd.DataFrame) -> float:
        """
        评估信号质量
        
        Args:
            signal: 信号序列（True/False）
            data: 原始数据
            
        Returns:
            float: 信号质量得分（0-100）
        """
        if signal.sum() == 0:
            return 50.0  # 无信号，返回中性分数
        
        # 基础信号质量评估因素
        quality_factors = {}
        
        # 1. 信号一致性 - 信号是否频繁变化
        signal_changes = signal.diff().fillna(0).abs().sum()
        total_signals = signal.sum()
        consistency = max(0, 100 - (signal_changes / total_signals * 100))
        quality_factors['consistency'] = consistency
        
        # 2. 信号与价格趋势的一致性
        if 'close' in data.columns:
            price_trend = data['close'].pct_change(5).iloc[-1] * 100  # 5日价格变化率
            # 买入信号时价格上涨或卖出信号时价格下跌，信号质量更高
            trend_consistency = 50
            if signal.name == 'buy_signal' and price_trend > 0:
                trend_consistency = 50 + min(50, price_trend * 5)
            elif signal.name == 'sell_signal' and price_trend < 0:
                trend_consistency = 50 + min(50, abs(price_trend) * 5)
            quality_factors['trend_consistency'] = trend_consistency
        
        # 3. 信号时机 - 是否在价格极值点附近
        if 'high' in data.columns and 'low' in data.columns:
            high_series = data['high']
            low_series = data['low']
            
            # 计算最近N日的极值
            lookback = 20
            recent_high = high_series.rolling(window=lookback).max().iloc[-1]
            recent_low = low_series.rolling(window=lookback).min().iloc[-1]
            
            current_price = data['close'].iloc[-1]
            price_range = recent_high - recent_low
            
            if price_range > 0:
                relative_position = (current_price - recent_low) / price_range
                
                # 买入信号在低位，卖出信号在高位，质量更高
                timing_score = 50
                if signal.name == 'buy_signal':
                    timing_score = 100 - relative_position * 100  # 越低越好
                elif signal.name == 'sell_signal':
                    timing_score = relative_position * 100  # 越高越好
                
                quality_factors['timing'] = timing_score
        
        # 4. 市场环境适应性
        market_env = self.get_market_environment()
        env_adaptation = 50  # 默认中性
        
        if signal.name == 'buy_signal':
            if market_env == MarketEnvironment.BULL_MARKET:
                env_adaptation = 80  # 牛市买入信号质量高
            elif market_env == MarketEnvironment.BEAR_MARKET:
                env_adaptation = 20  # 熊市买入信号质量低
        elif signal.name == 'sell_signal':
            if market_env == MarketEnvironment.BULL_MARKET:
                env_adaptation = 30  # 牛市卖出信号质量低
            elif market_env == MarketEnvironment.BEAR_MARKET:
                env_adaptation = 70  # 熊市卖出信号质量高
        
        quality_factors['market_adaptation'] = env_adaptation
        
        # 计算加权平均质量得分
        weights = {
            'consistency': 0.25,
            'trend_consistency': 0.25,
            'timing': 0.25,
            'market_adaptation': 0.25
        }
        
        weighted_score = 0
        total_weight = 0
        
        for factor, score in quality_factors.items():
            if factor in weights:
                weighted_score += score * weights[factor]
                total_weight += weights[factor]
        
        final_quality = weighted_score / total_weight if total_weight > 0 else 50
        return max(0, min(100, final_quality))
    
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
            # 计算原始评分
            raw_score = self.calculate_raw_score(data, **kwargs)
            
            # 检测市场环境
            market_env = self.detect_market_environment(data)
            self.set_market_environment(market_env)
            
            # 应用市场环境权重调整
            adjusted_score = self.apply_market_environment_adjustment(raw_score, market_env)
            
            # 识别形态
            patterns = self.identify_patterns(data, **kwargs)
            
            # 生成信号
            signals = self.generate_trading_signals(data, **kwargs)
            
            # 计算置信度
            confidence = self.calculate_confidence(adjusted_score, patterns, signals)
            
            return {
                'raw_score': raw_score,
                'final_score': adjusted_score,
                'market_environment': market_env,
                'patterns': patterns,
                'signals': signals,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"计算 {self.name} 评分时出错: {e}")
            # 返回默认值
            return {
                'raw_score': pd.Series(50.0, index=data.index),
                'final_score': pd.Series(50.0, index=data.index),
                'market_environment': MarketEnvironment.SIDEWAYS_MARKET,
                'patterns': [],
                'signals': {},
                'confidence': 50.0
            }
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分（子类可重写此方法实现具体的评分逻辑）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 默认实现：基于信号生成简单评分
        signals = self.generate_signals(data, **kwargs)
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 缓存键，用于避免重复计算
        cache_key = f"{data.index[0]}_{data.index[-1]}_{len(data)}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
            
        if 'buy_signal' in signals.columns:
            # 评估买入信号质量
            signal_quality = self.evaluate_signal_quality(signals['buy_signal'], data)
            # 买入信号基础加分
            score += signals['buy_signal'] * 20
            # 根据信号质量进行额外调整
            quality_adjustment = (signal_quality - 50) / 50 * 10  # 最多额外加减10分
            score += signals['buy_signal'] * quality_adjustment
            
        if 'sell_signal' in signals.columns:
            # 评估卖出信号质量
            signal_quality = self.evaluate_signal_quality(signals['sell_signal'], data)
            # 卖出信号基础减分
            score -= signals['sell_signal'] * 20
            # 根据信号质量进行额外调整
            quality_adjustment = (signal_quality - 50) / 50 * 10  # 最多额外加减10分
            score -= signals['sell_signal'] * quality_adjustment
        
        # 趋势强度评分
        if 'bull_trend' in signals.columns:
            trend_strength = signals['bull_trend'].astype(float)
            score += trend_strength * 10  # 上升趋势加分
            
        if 'bear_trend' in signals.columns:
            trend_strength = signals['bear_trend'].astype(float)
            score -= trend_strength * 10  # 下降趋势减分
            
        # 指标位置评分
        if hasattr(self, 'evaluate_position_score'):
            position_score = self.evaluate_position_score(data, **kwargs)
            score += position_score
            
        # 形态识别评分
        if hasattr(self, 'evaluate_pattern_score'):
            pattern_score = self.evaluate_pattern_score(data, **kwargs)
            score += pattern_score
            
        # 背离评分
        if hasattr(self, 'evaluate_divergence_score'):
            divergence_score = self.evaluate_divergence_score(data, **kwargs)
            score += divergence_score
            
        # 确保分数在0-100范围内
        score = np.clip(score, 0, 100)
        
        # 缓存结果
        self._score_cache[cache_key] = score
            
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别技术形态（子类可重写此方法实现具体的形态识别）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 基础形态识别：金叉死叉
        signals = self.generate_signals(data, **kwargs)
        
        if 'buy_signal' in signals.columns and signals['buy_signal'].any():
            patterns.append("买入信号")
        if 'sell_signal' in signals.columns and signals['sell_signal'].any():
            patterns.append("卖出信号")
            
        return patterns
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号（子类可重写此方法实现具体的信号生成）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典
        """
        # 调用原有的generate_signals方法
        signals_df = self.generate_signals(data, **kwargs)
        
        # 转换为字典格式
        signals = {}
        for col in signals_df.columns:
            signals[col] = signals_df[col]
            
        return signals
    
    def detect_market_environment(self, data: pd.DataFrame, lookback_period: int = 60) -> MarketEnvironment:
        """
        检测市场环境
        
        Args:
            data: 输入数据
            lookback_period: 回看周期
            
        Returns:
            MarketEnvironment: 市场环境
        """
        if len(data) < lookback_period:
            return MarketEnvironment.SIDEWAYS_MARKET
        
        recent_data = data.tail(lookback_period)
        
        # 1. 计算趋势强度
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # 2. 计算波动率
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        # 3. 计算动量
        momentum = self.calculate_momentum(recent_data)
        
        # 4. 计算成交量趋势
        volume_trend = self.calculate_volume_trend(recent_data) if 'volume' in recent_data.columns else 0
        
        # 5. 计算突破特征
        breakout_score = self.calculate_breakout_score(recent_data)
        
        # 综合判断市场环境
        if price_change > 0.2 and momentum > 0.6:
            return MarketEnvironment.BULL_MARKET
        elif price_change < -0.2 and momentum < -0.6:
            return MarketEnvironment.BEAR_MARKET
        elif volatility > 0.3 and abs(price_change) < 0.1:
            return MarketEnvironment.VOLATILE_MARKET
        elif breakout_score > 0.7:
            return MarketEnvironment.BREAKOUT_MARKET
        else:
            return MarketEnvironment.SIDEWAYS_MARKET
        
    def calculate_momentum(self, data: pd.DataFrame) -> float:
        """
        计算市场动量
        
        Args:
            data: 输入数据
            
        Returns:
            float: 动量得分，范围[-1, 1]，正值表示上升动量，负值表示下降动量
        """
        if len(data) < 10:
            return 0
            
        # 使用多个时间窗口计算价格变化率
        windows = [5, 10, 20]
        momentum_scores = []
        
        for window in windows:
            if len(data) < window:
                continue
                
            # 计算窗口的价格变化率
            change_rate = (data['close'].iloc[-1] - data['close'].iloc[-window]) / data['close'].iloc[-window]
            # 标准化变化率为[-1, 1]区间
            norm_change = np.tanh(change_rate * 5)  # 使用tanh函数将变化率映射到[-1, 1]
            momentum_scores.append(norm_change)
            
        # 返回平均动量得分
        return np.mean(momentum_scores) if momentum_scores else 0
        
    def calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """
        计算成交量趋势
        
        Args:
            data: 输入数据
            
        Returns:
            float: 成交量趋势得分，范围[-1, 1]，正值表示放量，负值表示缩量
        """
        if 'volume' not in data.columns or len(data) < 10:
            return 0
            
        # 计算近期和远期的平均成交量
        recent_vol = data['volume'].tail(5).mean()
        earlier_vol = data['volume'].iloc[-10:-5].mean() if len(data) >= 10 else recent_vol
        
        # 计算成交量变化率
        vol_change = (recent_vol - earlier_vol) / earlier_vol if earlier_vol > 0 else 0
        
        # 标准化为[-1, 1]区间
        return np.tanh(vol_change * 3)
        
    def calculate_breakout_score(self, data: pd.DataFrame) -> float:
        """
        计算突破特征得分
        
        Args:
            data: 输入数据
            
        Returns:
            float: 突破特征得分，范围[0, 1]，越高表示突破可能性越大
        """
        if len(data) < 20:
            return 0
            
        # 计算近期价格区间
        recent_high = data['high'].tail(5).max()
        recent_low = data['low'].tail(5).min()
        
        # 计算前期价格区间
        previous_high = data['high'].iloc[-20:-5].max()
        previous_low = data['low'].iloc[-20:-5].min()
        
        # 无突破时得分为0
        breakout_score = 0
        
        # 向上突破
        if recent_high > previous_high:
            # 计算突破幅度
            breakout_magnitude = (recent_high - previous_high) / previous_high
            # 突破得分，最高为1
            breakout_score = min(breakout_magnitude * 10, 1)
            
        # 向下突破
        elif recent_low < previous_low:
            # 计算突破幅度
            breakout_magnitude = (previous_low - recent_low) / previous_low
            # 突破得分，最高为1
            breakout_score = min(breakout_magnitude * 10, 1)
            
        return breakout_score
    
    def apply_market_environment_adjustment(self, score: pd.Series, market_env: MarketEnvironment) -> pd.Series:
        """
        应用市场环境权重调整
        
        Args:
            score: 原始评分
            market_env: 市场环境
            
        Returns:
            pd.Series: 调整后的评分
        """
        # 不同指标类型在不同市场环境下的权重调整
        env_adjustments = {
            MarketEnvironment.BULL_MARKET: {
                'trend': 1.2,      # 牛市中趋势指标权重提高
                'oscillator': 0.8,  # 牛市中震荡指标权重降低
                'volume': 1.1,      # 牛市中成交量指标权重适度提高
                'volatility': 0.9   # 牛市中波动率指标权重适度降低
            },
            MarketEnvironment.BEAR_MARKET: {
                'trend': 0.8,      # 熊市中趋势指标权重降低
                'oscillator': 1.2,  # 熊市中震荡指标权重提高
                'volume': 1.0,      # 熊市中成交量指标权重保持不变
                'volatility': 1.1   # 熊市中波动率指标权重提高
            },
            MarketEnvironment.SIDEWAYS_MARKET: {
                'trend': 0.7,      # 震荡市场中趋势指标权重显著降低
                'oscillator': 1.3,  # 震荡市场中震荡指标权重显著提高
                'volume': 1.0,      # 震荡市场中成交量指标权重保持不变
                'volatility': 1.0   # 震荡市场中波动率指标权重保持不变
            },
            MarketEnvironment.VOLATILE_MARKET: {
                'trend': 0.8,      # 高波动市场中趋势指标权重降低
                'oscillator': 0.9,  # 高波动市场中震荡指标权重适度降低
                'volume': 1.1,      # 高波动市场中成交量指标权重适度提高
                'volatility': 1.2   # 高波动市场中波动率指标权重提高
            },
            MarketEnvironment.BREAKOUT_MARKET: {
                'trend': 1.1,      # 突破市场中趋势指标权重适度提高
                'oscillator': 0.8,  # 突破市场中震荡指标权重降低
                'volume': 1.3,      # 突破市场中成交量指标权重显著提高
                'volatility': 1.2   # 突破市场中波动率指标权重提高
            }
        }
        
        # 获取当前指标类型
        indicator_type = self.get_indicator_type()
        
        # 获取适用的调整系数
        adjustment = env_adjustments.get(market_env, {}).get(indicator_type, 1.0)
        
        # 计算中性点偏移量
        neutral_point = 50
        deviation = score - neutral_point
        
        # 应用调整：保持中性点不变，放大或缩小偏离中性点的距离
        adjusted_score = neutral_point + deviation * adjustment
        
        # 确保分数在0-100范围内
        return np.clip(adjusted_score, 0, 100)
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型，可能的值包括：'trend', 'oscillator', 'volume', 'volatility'
        """
        # 默认实现：根据类名判断指标类型
        class_name = self.__class__.__name__.lower()
        
        trend_keywords = ['ma', 'ema', 'dma', 'macd', 'boll', 'sar', 'dmi', 'adx', 'trix', 'trend']
        oscillator_keywords = ['rsi', 'kdj', 'cci', 'stoch', 'wr', 'bias', 'oscillator']
        volume_keywords = ['vol', 'obv', 'mfi', 'vr', 'vosc', 'volume']
        volatility_keywords = ['atr', 'volatility', 'keltner', 'kc']
        
        for keyword in trend_keywords:
            if keyword in class_name:
                return 'trend'
                
        for keyword in oscillator_keywords:
            if keyword in class_name:
                return 'oscillator'
                
        for keyword in volume_keywords:
            if keyword in class_name:
                return 'volume'
                
        for keyword in volatility_keywords:
            if keyword in class_name:
                return 'volatility'
                
        return 'trend'  # 默认为趋势指标
    
    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算置信度
        
        Args:
            score: 评分序列
            patterns: 识别的形态
            signals: 生成的信号
            
        Returns:
            float: 置信度（0-100）
        """
        base_confidence = 50.0
        
        # 1. 基于评分的置信度
        latest_score = score.iloc[-1] if len(score) > 0 else 50.0
        score_confidence = abs(latest_score - 50) * 1.5  # 距离中性分越远，置信度越高
        
        # 2. 基于形态数量和质量的置信度
        pattern_confidence = 0
        # 形态权重字典，不同形态有不同的重要性
        pattern_weights = {
            "背离": 20,          # 背离形态最重要
            "双重背离": 25,      # 双重背离更重要
            "黄金交叉": 15,      # 黄金交叉较重要
            "死亡交叉": 15,      # 死亡交叉较重要
            "突破": 15,          # 突破较重要
            "支撑": 10,          # 支撑一般重要
            "阻力": 10,          # 阻力一般重要
            "超买": 10,          # 超买一般重要
            "超卖": 10,          # 超卖一般重要
            "买入信号": 10,      # 买入信号一般重要
            "卖出信号": 10,      # 卖出信号一般重要
            "W底": 20,           # W底较重要
            "M顶": 20,           # M顶较重要
            "三重底": 25,        # 三重底很重要
            "三重顶": 25,        # 三重顶很重要
            "头肩顶": 20,        # 头肩顶较重要
            "头肩底": 20,        # 头肩底较重要
            "钓鱼线": 15,        # 钓鱼线较重要
            "吞没": 15,          # 吞没较重要
            "孕线": 12           # 孕线一般重要
        }
        
        # 计算形态置信度
        for pattern in patterns:
            for key, weight in pattern_weights.items():
                if key in pattern:
                    pattern_confidence += weight
                    break
            else:
                # 默认权重
                pattern_confidence += 5
                
        # 限制形态置信度上限
        pattern_confidence = min(pattern_confidence, 30)
        
        # 3. 基于信号强度和一致性的置信度
        signal_confidence = 0
        valid_signals = []
        
        for signal_name, signal_series in signals.items():
            if signal_series.any():
                # 只统计最近的信号
                recent_signal = signal_series.iloc[-5:].any()
                if recent_signal:
                    valid_signals.append(signal_name)
                    
                    # 评估信号质量
                    if hasattr(self, 'evaluate_signal_quality'):
                        signal_quality = self.evaluate_signal_quality(signal_series, score.to_frame())
                        signal_confidence += signal_quality * 0.1  # 最多贡献10分
        
        # 信号一致性评估
        if 'buy_signal' in valid_signals and 'sell_signal' in valid_signals:
            # 信号矛盾，降低置信度
            signal_confidence -= 10
        elif 'buy_signal' in valid_signals and 'bull_trend' in valid_signals:
            # 信号一致，提高置信度
            signal_confidence += 5
        elif 'sell_signal' in valid_signals and 'bear_trend' in valid_signals:
            # 信号一致，提高置信度
            signal_confidence += 5
            
        # 限制信号置信度上限
        signal_confidence = min(max(0, signal_confidence), 20)
        
        # 4. 市场环境匹配度
        market_env = self.get_market_environment()
        market_confidence = 0
        
        # 判断指标类型与市场环境的匹配度
        indicator_type = self.get_indicator_type()
        
        # 市场环境与指标类型匹配度
        env_type_match = {
            (MarketEnvironment.BULL_MARKET, 'trend'): 10,
            (MarketEnvironment.BULL_MARKET, 'oscillator'): 0,
            (MarketEnvironment.BULL_MARKET, 'volume'): 5,
            (MarketEnvironment.BULL_MARKET, 'volatility'): 0,
            
            (MarketEnvironment.BEAR_MARKET, 'trend'): 0,
            (MarketEnvironment.BEAR_MARKET, 'oscillator'): 10,
            (MarketEnvironment.BEAR_MARKET, 'volume'): 5,
            (MarketEnvironment.BEAR_MARKET, 'volatility'): 5,
            
            (MarketEnvironment.SIDEWAYS_MARKET, 'trend'): -5,
            (MarketEnvironment.SIDEWAYS_MARKET, 'oscillator'): 10,
            (MarketEnvironment.SIDEWAYS_MARKET, 'volume'): 0,
            (MarketEnvironment.SIDEWAYS_MARKET, 'volatility'): 0,
            
            (MarketEnvironment.VOLATILE_MARKET, 'trend'): -5,
            (MarketEnvironment.VOLATILE_MARKET, 'oscillator'): 0,
            (MarketEnvironment.VOLATILE_MARKET, 'volume'): 5,
            (MarketEnvironment.VOLATILE_MARKET, 'volatility'): 10,
            
            (MarketEnvironment.BREAKOUT_MARKET, 'trend'): 5,
            (MarketEnvironment.BREAKOUT_MARKET, 'oscillator'): -5,
            (MarketEnvironment.BREAKOUT_MARKET, 'volume'): 10,
            (MarketEnvironment.BREAKOUT_MARKET, 'volatility'): 5,
        }
        
        market_confidence = env_type_match.get((market_env, indicator_type), 0)
        
        # 计算总置信度
        total_confidence = base_confidence + score_confidence + pattern_confidence + signal_confidence + market_confidence
        
        # 确保置信度在0-100范围内
        return max(0, min(100, total_confidence))
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成指标信号
        
        Args:
            data: 输入数据，通常是K线数据
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果，例如：
                - golden_cross: 金叉信号
                - dead_cross: 死叉信号
                - buy_signal: 买入信号
                - sell_signal: 卖出信号
                - bull_trend: 多头趋势
                - bear_trend: 空头趋势
        """
        # 默认实现：计算指标并返回一个空的信号DataFrame
        if not self.has_result():
            self.calculate(data, *args, **kwargs)
            
        # 创建空的信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        
        return signals
    
    def compute(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算技术指标并处理异常
        
        Args:
            data: 输入数据，通常是K线数据
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 计算结果
        
        Raises:
            Exception: 计算过程中出现异常
        """
        try:
            self._result = self.calculate(data, *args, **kwargs)
            self._error = None
            return self._result
        except Exception as e:
            logger.error(f"计算指标 {self.name} 时出错: {e}")
            self._error = e
            self._result = None
            raise
    
    def safe_compute(self, data: pd.DataFrame, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        安全计算技术指标，不抛出异常
        
        Args:
            data: 输入数据，通常是K线数据
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            Optional[pd.DataFrame]: 计算结果，如果出错则返回None
        """
        try:
            return self.compute(data, *args, **kwargs)
        except Exception as e:
            logger.error(f"安全计算指标 {self.name} 时出错: {e}")
            return None
    
    def get_column_name(self, suffix: str = "") -> str:
        """
        获取指标列名
        
        Args:
            suffix: 列名后缀
            
        Returns:
            str: 指标列名
        """
        if suffix:
            return f"{self.name}_{suffix}"
        return self.name
    
    @staticmethod
    def ensure_columns(data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        确保数据包含必需的列
        
        Args:
            data: 输入数据
            required_columns: 必需的列名列表
            
        Returns:
            bool: 是否包含所有必需的列
            
        Raises:
            ValueError: 如果缺少必需的列
        """
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必需的列: {', '.join(missing_columns)}")
        return True
    
    @staticmethod
    def crossover(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
        """
        计算两个序列的上穿信号
        
        Args:
            series1: 第一个序列
            series2: 第二个序列或标量值
            
        Returns:
            pd.Series: 上穿信号序列，上穿为True，否则为False
        """
        series1 = pd.Series(series1)
        
        if isinstance(series2, (int, float)):
            # 如果是标量，创建同样长度的序列
            series2 = pd.Series([series2] * len(series1), index=series1.index)
        else:
            series2 = pd.Series(series2)
            
        return (series1.shift(1) < series2.shift(1)) & (series1 > series2)
    
    @staticmethod
    def crossunder(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
        """
        计算两个序列的下穿信号
        
        Args:
            series1: 第一个序列
            series2: 第二个序列或标量值
            
        Returns:
            pd.Series: 下穿信号序列，下穿为True，否则为False
        """
        series1 = pd.Series(series1)
        
        if isinstance(series2, (int, float)):
            # 如果是标量，创建同样长度的序列
            series2 = pd.Series([series2] * len(series1), index=series1.index)
        else:
            series2 = pd.Series(series2)
            
        return (series1.shift(1) > series2.shift(1)) & (series1 < series2)
    
    @staticmethod
    def sma(series: pd.Series, periods: int) -> pd.Series:
        """
        计算简单移动平均线
        
        Args:
            series: 输入序列
            periods: 周期
            
        Returns:
            pd.Series: 简单移动平均线
        """
        return series.rolling(window=periods).mean()
    
    @staticmethod
    def ema(series: pd.Series, periods: int) -> pd.Series:
        """
        计算指数移动平均线
        
        Args:
            series: 输入序列
            periods: 周期
            
        Returns:
            pd.Series: 指数移动平均线
        """
        return series.ewm(span=periods, adjust=False).mean()
    
    @staticmethod
    def highest(series: pd.Series, periods: int) -> pd.Series:
        """
        计算周期内最高值
        
        Args:
            series: 输入序列
            periods: 周期
            
        Returns:
            pd.Series: 周期内最高值
        """
        return series.rolling(window=periods).max()
    
    @staticmethod
    def lowest(series: pd.Series, periods: int) -> pd.Series:
        """
        计算周期内最低值
        
        Args:
            series: 输入序列
            periods: 周期
            
        Returns:
            pd.Series: 周期内最低值
        """
        return series.rolling(window=periods).min()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, periods: int) -> pd.Series:
        """
        计算平均真实范围(ATR)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            periods: 周期
            
        Returns:
            pd.Series: ATR值
        """
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=periods).mean()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将指标转换为字典表示
        
        Returns:
            Dict[str, Any]: 指标的字典表示
        """
        return {
            "name": self.name,
            "description": self.description,
            "has_result": self.has_result(),
            "has_error": self.has_error(),
            "error": str(self.error) if self.has_error() else None
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"<{self.__class__.__name__} name='{self.name}' has_result={self.has_result()} has_error={self.has_error()}>" 