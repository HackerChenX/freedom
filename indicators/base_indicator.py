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
        
        if 'buy_signal' in signals.columns:
            score += signals['buy_signal'] * 20  # 买入信号+20分
        if 'sell_signal' in signals.columns:
            score -= signals['sell_signal'] * 20  # 卖出信号-20分
            
        return np.clip(score, 0, 100)
    
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
        
        # 计算趋势强度
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # 计算波动率
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        # 判断市场环境
        if price_change > 0.2:  # 上涨超过20%
            return MarketEnvironment.BULL_MARKET
        elif price_change < -0.2:  # 下跌超过20%
            return MarketEnvironment.BEAR_MARKET
        elif volatility > 0.3:  # 高波动率
            return MarketEnvironment.VOLATILE_MARKET
        else:
            return MarketEnvironment.SIDEWAYS_MARKET
    
    def apply_market_environment_adjustment(self, score: pd.Series, market_env: MarketEnvironment) -> pd.Series:
        """
        应用市场环境权重调整
        
        Args:
            score: 原始评分
            market_env: 市场环境
            
        Returns:
            pd.Series: 调整后的评分
        """
        # 根据市场环境调整评分
        if market_env == MarketEnvironment.BULL_MARKET:
            # 牛市中，适度提高买入信号的权重
            adjustment = 1.1
        elif market_env == MarketEnvironment.BEAR_MARKET:
            # 熊市中，适度降低买入信号的权重
            adjustment = 0.9
        elif market_env == MarketEnvironment.VOLATILE_MARKET:
            # 高波动市场中，降低信号权重
            adjustment = 0.8
        else:
            # 震荡市场，保持原有权重
            adjustment = 1.0
        
        adjusted_score = score * adjustment
        return np.clip(adjusted_score, 0, 100)
    
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
        
        # 基于评分的置信度
        latest_score = score.iloc[-1] if len(score) > 0 else 50.0
        score_confidence = abs(latest_score - 50) * 2  # 距离中性分越远，置信度越高
        
        # 基于形态数量的置信度
        pattern_confidence = min(len(patterns) * 10, 30)  # 每个形态增加10分，最多30分
        
        # 基于信号强度的置信度
        signal_confidence = 0
        for signal_name, signal_series in signals.items():
            if signal_series.any():
                signal_confidence += 10
        signal_confidence = min(signal_confidence, 20)  # 最多20分
        
        total_confidence = base_confidence + score_confidence + pattern_confidence + signal_confidence
        return min(total_confidence, 100.0)
    
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