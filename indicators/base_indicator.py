"""
技术指标基类模块

提供技术指标计算的通用接口和功能
"""

import abc
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from enum import Enum
import inspect

from utils.logger import get_logger

logger = get_logger(__name__)

# 定义基础数据列
BASE_COLUMNS = ['open', 'high', 'low', 'close', 'volume']


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


class PatternResult:
    """形态识别结果类"""
    
    def __init__(self, 
                pattern_id: str, 
                display_name: str,
                strength: float,
                duration: int = 1,
                details: Dict[str, Any] = None):
        """
        初始化形态识别结果
        
        Args:
            pattern_id: 形态唯一标识符
            display_name: 形态显示名称
            strength: 形态强度，通常为0-100的值
            duration: 形态持续的天数
            details: 形态详细信息，如形态的具体参数
        """
        self.pattern_id = pattern_id
        self.display_name = display_name
        self.strength = strength
        self.duration = duration
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典表示
        
        Returns:
            Dict[str, Any]: 形态识别结果的字典表示
        """
        return {
            'pattern_id': self.pattern_id,
            'display_name': self.display_name,
            'strength': self.strength,
            'duration': self.duration,
            'details': self.details
        }


class BaseIndicator(abc.ABC):
    """
    技术指标基类
    
    所有技术指标类应继承此类，并实现必要的抽象方法
    """
    
    def __init__(self, name: str = "", description: str = "", weight: float = 1.0):
        """
        初始化基础技术指标
        
        Args:
            name: 指标名称，如果不提供则尝试使用类属性
            description: 指标描述
            weight: 指标权重
        """
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
        
        # 自动注册形态
        self.register_patterns()
    
    def register_patterns(self):
        """
        注册指标形态
        
        自动查找并调用形态注册方法，标准化形态注册流程
        """
        # 获取指标类型（大写）
        indicator_type = self.get_indicator_type()
        
        # 查找所有可能的形态注册方法
        possible_methods = [
            f"_register_{indicator_type.lower()}_patterns",  # 如：_register_macd_patterns
            "_register_patterns",                          # 通用方法
            f"register_{indicator_type.lower()}_patterns"    # 旧格式
        ]
        
        # 尝试调用形态注册方法
        for method_name in possible_methods:
            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                logger.debug(f"调用 {self.__class__.__name__} 的形态注册方法: {method_name}")
                try:
                    getattr(self, method_name)()
                    # 找到并成功调用一个方法后退出
                    return
                except Exception as e:
                    logger.error(f"调用 {method_name} 注册形态时出错: {e}")
    
    def register_pattern_to_registry(self, pattern_id: str, display_name: str, 
                                   description: str, pattern_type: str,
                                   default_strength: str = "MEDIUM", 
                                   score_impact: float = 0.0,
                                   detection_function=None):
        """
        将形态注册到全局形态注册表
        
        Args:
            pattern_id: 形态ID
            display_name: 显示名称
            description: 形态描述
            pattern_type: 形态类型（如 BULLISH, BEARISH, NEUTRAL 等）
            default_strength: 默认强度（如 VERY_STRONG, STRONG, MEDIUM, WEAK, VERY_WEAK）
            score_impact: 对评分的影响（-100到100）
            detection_function: 形态检测函数
        """
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
        
        # 获取指标类型
        indicator_id = self.get_indicator_type()
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 获取PatternType枚举值
        try:
            if isinstance(pattern_type, PatternType):
                pattern_type_enum = pattern_type
            else:
                pattern_type_enum = getattr(PatternType, pattern_type.upper())
        except (AttributeError, KeyError):
            logger.warning(f"未知的形态类型: {pattern_type}，使用默认NEUTRAL")
            pattern_type_enum = PatternType.NEUTRAL
        
        # 获取PatternStrength枚举值
        try:
            if isinstance(default_strength, PatternStrength):
                strength_enum = default_strength
            else:
                strength_enum = getattr(PatternStrength, default_strength.upper())
        except (AttributeError, KeyError):
            logger.warning(f"未知的强度类型: {default_strength}，使用默认MEDIUM")
            strength_enum = PatternStrength.MEDIUM
        
        # 注册形态
        registry.register(
            pattern_id=pattern_id,
            display_name=display_name,
            description=description,
            indicator_id=indicator_id,
            pattern_type=pattern_type_enum,
            default_strength=strength_enum,
            score_impact=score_impact,
            detection_function=detection_function
        )
    
    @property
    def result(self) -> Optional[pd.DataFrame]:
        """
        获取指标计算结果
        
        Returns:
            Optional[pd.DataFrame]: 指标计算结果，如果尚未计算则返回None
        """
        return self._result
    
    @property
    def error(self) -> Optional[Exception]:
        """获取指标计算错误"""
        return self._error
    
    def has_result(self) -> bool:
        """是否已经计算出结果"""
        return self._result is not None
    
    def has_error(self) -> bool:
        """是否有计算错误"""
        return self._error is not None
    
    def _preserve_base_columns(self, source_df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        在计算结果中保留基础数据列，确保链式计算能够正常进行
        
        Args:
            source_df: 源数据DataFrame
            result_df: 计算结果DataFrame
            
        Returns:
            pd.DataFrame: 包含原始基础列的结果DataFrame
        """
        # 检查源数据中有哪些基础列
        available_columns = [col for col in BASE_COLUMNS if col in source_df.columns]
        
        # 如果结果中缺少这些列，则从源数据中复制
        for col in available_columns:
            if col not in result_df.columns:
                result_df[col] = source_df[col]
        
        return result_df
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算指标的模板方法，包含列保护和错误处理
        
        Args:
            data: 输入数据
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            pd.DataFrame: 包含指标和原始基础列的结果
        """
        try:
            # 执行核心计算
            result_df = self._calculate(data, *args, **kwargs)
            
            # 保护基础列
            final_df = self._preserve_base_columns(data, result_df)
            
            # 缓存结果
            self._result = final_df
            self._error = None
            
            return final_df
            
        except Exception as e:
            logger.error(f"计算指标 {self.name} 时出错: {e}")
            self._error = e
            # 在出错时返回一个包含错误信息的空DataFrame
            error_df = pd.DataFrame(index=data.index)
            error_df['error'] = str(e)
            return self._preserve_base_columns(data, error_df)

    @abc.abstractmethod
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算指标的核心逻辑
        
        Args:
            data: 输入数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 指标计算结果
        """
        raise NotImplementedError("子类必须实现 _calculate 方法")
    
    def get_registered_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有已注册的形态
        
        Returns:
            Dict[str, Dict[str, Any]]: 已注册形态字典
        """
        from indicators.pattern_registry import PatternRegistry
        
        # 获取指标类型
        indicator_id = self.get_indicator_type()
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 从PatternRegistry获取该指标的所有形态
        pattern_ids = registry.get_patterns_by_indicator(indicator_id)
        
        # 转换形态信息格式
        patterns = {}
        for pattern_id in pattern_ids:
            pattern_info = registry.get_pattern(pattern_id)
            if pattern_info:
                # 转换为兼容格式
                patterns[pattern_id] = {
                    'display_name': pattern_info.get('display_name', ''),
                    'detection_func': pattern_info.get('detection_function'),
                    'score_impact': pattern_info.get('score_impact', 0.0)
                }
        
        return patterns
    
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
            Dict[str, Any]: 包含评分、形态、信号等信息的字典
        """
        if not self.has_result():
            if 'price_col' in kwargs:
                price_col = kwargs.pop('price_col')
                self.calculate(data, price_col=price_col, **kwargs)
            else:
                self.calculate(data, **kwargs)

        # 获取原始评分（0-100分制）
        raw_score = self.calculate_raw_score(data, **kwargs)

        if raw_score.empty:
            return {
                'score': 50.0,
                'raw_score': 50.0,
                'patterns': [],
                'signals': {},
                'market_environment': MarketEnvironment.SIDEWAYS_MARKET,
                'confidence': 0.5,
                'indicator': self.name,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        # 检测市场环境
        market_env = kwargs.get('market_env', None)
        if market_env is None:
            market_env = self.detect_market_environment(data)
            self.set_market_environment(market_env)

        # 应用市场环境调整
        adjusted_score = self.apply_market_environment_adjustment(raw_score, market_env)

        # 识别当前形态
        patterns = self.get_patterns(data, **kwargs)

        # 生成交易信号
        signals = self.generate_trading_signals(data, **kwargs)

        # 计算信号置信度
        confidence = self.calculate_confidence(adjusted_score, [p['pattern_id'] for p in patterns], signals)

        # 确保最终评分在0-100范围内
        final_score = pd.Series(np.clip(adjusted_score, 0, 100), index=raw_score.index)

        return {
            'score': final_score,
            'raw_score': raw_score,
            'patterns': patterns,
            'signals': signals,
            'market_environment': market_env,
            'confidence': confidence,
            'indicator': self.name,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    @abc.abstractmethod
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算指标的原始评分
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        pass
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        获取指标的所有形态。

        此方法作为形态识别的主要入口。它首先尝试调用子类中可能实现的
        `identify_patterns` 方法。如果 `identify_patterns` 未实现或未返回结果，
        则会使用在 `PatternRegistry` 中注册的 `detection_function` 来自动检测形态。

        [!!] 标准返回格式 (推荐):
        pd.DataFrame: 索引为日期，列为形态ID（如 'GOLDEN_CROSS'），值为布尔值。
                      这种格式性能更优，且与上层分析器兼容性最好。

        [!!] 已弃用的返回格式 (不推荐):
        List[Dict[str, Any]]: 包含历史上所有形态的字典列表。
                               此格式仅为向后兼容保留，存在性能问题，
                               并可能在未来的版本中被移除。

        Args:
            data (pd.DataFrame): 包含指标计算值的DataFrame。
            **kwargs: 其他可能的参数。

        Returns:
            Union[pd.DataFrame, List[Dict[str, Any]]]: 识别出的形态。
                                                      优先返回DataFrame。
        """
        # 优先尝试调用子类实现的 identify_patterns 方法
        if hasattr(self, 'identify_patterns') and callable(getattr(self, 'identify_patterns')):
            try:
                patterns = self.identify_patterns(data, **kwargs)
                if patterns is not None and (isinstance(patterns, pd.DataFrame) or patterns):
                    return patterns
            except Exception as e:
                logger.error(f"调用 {self.name} 的 identify_patterns 方法时出错: {e}")
        
        # 如果 identify_patterns 不可用或无返回，则使用注册的检测函数
        return self._detect_patterns_from_registry(data)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> Optional[Union[pd.DataFrame, List[str], List[Dict]]]:
        """
        识别指标的特定形态（由子类实现）。

        子类应重写此方法以提供自定义的、复杂的形态识别逻辑。

        [!!] 标准返回格式 (推荐):
        pd.DataFrame: 索引为日期，列为形态ID，值为布尔值。

        [!!] 已弃用的返回格式 (不推荐):
        List[str] 或 List[Dict]: 包含形态ID的列表或形态详情的字典列表。
                                  此格式仅为向后兼容保留。

        Args:
            data (pd.DataFrame): 包含指标计算值的DataFrame。
            **kwargs: 其他可能的参数。

        Returns:
            Optional[Union[pd.DataFrame, List[str], List[Dict]]]: 
                识别出的形态。如果未实现，应返回None。
        """
        return None
    
    def _detect_patterns_from_registry(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        使用在 PatternRegistry 中注册的检测函数来识别形态。

        Args:
            data (pd.DataFrame): 包含指标计算值的DataFrame。

        Returns:
            Optional[pd.DataFrame]: 识别出的形态。如果未实现，应返回None。
        """
        return None
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据，必须包含价格数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 交易信号，包含buy和sell系列
        """
        logger.warning(f"指标 {self.name} 使用默认的交易信号生成方法")
        
        # 创建空信号
        signals = {}
        signals["buy"] = pd.Series(False, index=data.index)
        signals["sell"] = pd.Series(False, index=data.index)
        
        # 尝试使用已计算的结果生成简单信号
        if self._result is not None and not self._result.empty:
            try:
                # 基于已识别的形态生成信号
                patterns = self.identify_patterns(self._result)
                
                # 从PatternRegistry获取形态信息
                from indicators.pattern_registry import PatternRegistry
                registry = PatternRegistry()
                
                # 遍历形态生成信号
                for pattern_id in patterns:
                    pattern_info = registry.get_pattern(pattern_id)
                    if pattern_info:
                        score_impact = pattern_info.get('score_impact', 0)
                        
                        # 基于得分影响生成信号
                        if score_impact > 0:  # 正面影响，生成买入信号
                            signals["buy"].iloc[-1] = True
                        elif score_impact < 0:  # 负面影响，生成卖出信号
                            signals["sell"].iloc[-1] = True
            except Exception as e:
                logger.error(f"生成交易信号时出错: {e}")
        
        return signals
    
    def detect_market_environment(self, data: pd.DataFrame, lookback_period: int = 60) -> MarketEnvironment:
        """
        检测当前市场环境
        
        通过分析价格、成交量和动量来确定当前市场环境
        
        Args:
            data: 输入数据
            lookback_period: 回溯周期，默认60个交易日
            
        Returns:
            MarketEnvironment: 检测到的市场环境
        """
        if 'close' not in data.columns:
            logger.warning("无法检测市场环境: 缺少收盘价数据")
            return MarketEnvironment.SIDEWAYS_MARKET
            
        # 确保有足够的数据
        if len(data) < lookback_period:
            logger.warning(f"数据长度不足 {lookback_period} 个周期，无法准确检测市场环境")
            return MarketEnvironment.SIDEWAYS_MARKET
        
        # 计算各种市场特征
        momentum_score = self.calculate_momentum(data)
        volume_score = self.calculate_volume_trend(data) if 'volume' in data.columns else 50
        breakout_score = self.calculate_breakout_score(data)
        
        # 根据特征得分确定市场环境
        if momentum_score > 70 and volume_score > 60:
            return MarketEnvironment.BULL_MARKET
        elif momentum_score < 30 and volume_score > 60:
            return MarketEnvironment.BEAR_MARKET
        elif breakout_score > 70:
            return MarketEnvironment.BREAKOUT_MARKET
        elif abs(momentum_score - 50) < 15 and abs(volume_score - 50) < 15:
            return MarketEnvironment.SIDEWAYS_MARKET
        else:
            return MarketEnvironment.VOLATILE_MARKET
        
    def calculate_momentum(self, data: pd.DataFrame) -> float:
        """
        计算价格动量得分
        
        Args:
            data: 输入数据
            
        Returns:
            float: 动量得分（0-100）
        """
        if 'close' not in data.columns:
            return 50.0
            
        # 只使用最近的数据
        lookback = min(60, len(data))
        recent_data = data.tail(lookback)
        
        close = recent_data['close']
        
        # 计算不同周期的动量
        momentum_5 = close.pct_change(5).iloc[-1] * 100  # 5日动量
        momentum_10 = close.pct_change(10).iloc[-1] * 100  # 10日动量
        momentum_20 = close.pct_change(20).iloc[-1] * 100  # 20日动量
        
        # 计算短期均线趋势
        if len(close) >= 20:
            ma5 = close.rolling(5).mean()
            ma10 = close.rolling(10).mean()
            ma20 = close.rolling(20).mean()
            
            # 均线多头排列（MA5 > MA10 > MA20）
            bullish_ma_alignment = (ma5.iloc[-1] > ma10.iloc[-1]) and (ma10.iloc[-1] > ma20.iloc[-1])
            # 均线空头排列（MA5 < MA10 < MA20）
            bearish_ma_alignment = (ma5.iloc[-1] < ma10.iloc[-1]) and (ma10.iloc[-1] < ma20.iloc[-1])
        else:
            bullish_ma_alignment = False
            bearish_ma_alignment = False
        
        # 计算动量综合得分（0-100分）
        momentum_score = 50  # 基础分
        
        # 短期动量贡献
        momentum_score += momentum_5 * 2  # 每1%贡献2分
        
        # 中期动量贡献
        momentum_score += momentum_10 * 1.5  # 每1%贡献1.5分
        
        # 长期动量贡献
        momentum_score += momentum_20 * 1  # 每1%贡献1分
        
        # 均线排列贡献
        if bullish_ma_alignment:
            momentum_score += 10  # 多头排列加10分
        elif bearish_ma_alignment:
            momentum_score -= 10  # 空头排列减10分
        
        # 限制得分范围在0-100之间
        return max(0, min(100, momentum_score))
        
    def calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """
        计算成交量趋势得分
        
        Args:
            data: 输入数据
            
        Returns:
            float: 成交量趋势得分（0-100）
        """
        if 'volume' not in data.columns or 'close' not in data.columns:
            return 50.0
            
        # 只使用最近的数据
        lookback = min(60, len(data))
        recent_data = data.tail(lookback)
        
        volume = recent_data['volume']
        close = recent_data['close']
        
        # 计算成交量趋势
        avg_volume_5 = volume.rolling(5).mean()
        avg_volume_20 = volume.rolling(20).mean()
        
        # 计算最近5日成交量与20日成交量的比值
        if avg_volume_20.iloc[-1] > 0:
            volume_ratio = avg_volume_5.iloc[-1] / avg_volume_20.iloc[-1]
        else:
            volume_ratio = 1.0
        
        # 计算价格与成交量的配合度
        price_up = close.pct_change() > 0
        high_volume = volume > avg_volume_20
        
        # 放量上涨和缩量下跌是积极信号
        positive_volume_price = (price_up & high_volume) | (~price_up & ~high_volume)
        positive_ratio = positive_volume_price.sum() / len(positive_volume_price)
        
        # 计算成交量趋势得分（0-100分）
        volume_score = 50  # 基础分
        
        # 成交量比值贡献
        if volume_ratio > 1:
            volume_score += (volume_ratio - 1) * 20  # 放量
        else:
            volume_score -= (1 - volume_ratio) * 20  # 缩量
        
        # 价量配合度贡献
        volume_score += (positive_ratio - 0.5) * 40  # 价量配合度贡献
        
        # 限制得分范围在0-100之间
        return max(0, min(100, volume_score))
        
    def calculate_breakout_score(self, data: pd.DataFrame) -> float:
        """
        计算价格突破得分
        
        Args:
            data: 输入数据
            
        Returns:
            float: 突破得分（0-100）
        """
        if 'close' not in data.columns or 'high' not in data.columns or 'low' not in data.columns:
            return 50.0
            
        # 只使用最近的数据
        lookback = min(60, len(data))
        recent_data = data.tail(lookback)
        
        close = recent_data['close']
        high = recent_data['high']
        low = recent_data['low']
        
        # 计算不同周期的压力/支撑位
        resistance_20 = high.rolling(20).max()
        support_20 = low.rolling(20).min()
        
        # 检查是否有效突破
        current_close = close.iloc[-1]
        
        # 向上突破
        if len(resistance_20) >= 5:
            prev_resistance = resistance_20.iloc[-5]
            up_breakout = current_close > prev_resistance
            up_breakout_strength = (current_close / prev_resistance - 1) * 100 if up_breakout else 0
        else:
            up_breakout = False
            up_breakout_strength = 0
            
        # 向下突破
        if len(support_20) >= 5:
            prev_support = support_20.iloc[-5]
            down_breakout = current_close < prev_support
            down_breakout_strength = (1 - current_close / prev_support) * 100 if down_breakout else 0
        else:
            down_breakout = False
            down_breakout_strength = 0
        
        # 计算突破后的量能变化
        if 'volume' in recent_data.columns:
            avg_volume_5 = recent_data['volume'].rolling(5).mean()
            avg_volume_20 = recent_data['volume'].rolling(20).mean()
            
            if len(avg_volume_5) > 0 and len(avg_volume_20) > 0 and avg_volume_20.iloc[-1] > 0:
                volume_change = avg_volume_5.iloc[-1] / avg_volume_20.iloc[-1] - 1
            else:
                volume_change = 0
        else:
            volume_change = 0
        
        # 计算突破得分（0-100分）
        breakout_score = 50  # 基础分
        
        # 向上突破贡献
        if up_breakout:
            breakout_score += min(up_breakout_strength * 2, 30)  # 最多加30分
            breakout_score += volume_change * 20  # 放量突破加分
        
        # 向下突破贡献
        if down_breakout:
            breakout_score -= min(down_breakout_strength * 2, 30)  # 最多减30分
            breakout_score -= volume_change * 20  # 放量突破减分
        
        # 限制得分范围在0-100之间
        return max(0, min(100, breakout_score))
    
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
            # 牛市环境下，看涨信号更可信，看跌信号需谨慎
            adjusted_score = np.where(
                score > 50,  # 看涨信号
                score + (score - 50) * 0.2,  # 提高看涨信号的评分
                score + (score - 50) * 0.1   # 降低看跌信号的评分
            )
        elif market_env == MarketEnvironment.BEAR_MARKET:
            # 熊市环境下，看跌信号更可信，看涨信号需谨慎
            adjusted_score = np.where(
                score < 50,  # 看跌信号
                score - (50 - score) * 0.2,  # 降低看跌信号的评分
                score - (score - 50) * 0.1   # 提高看涨信号的评分
            )
        elif market_env == MarketEnvironment.VOLATILE_MARKET:
            # 高波动环境下，极端信号更有价值，中性信号更加中性
            adjusted_score = np.where(
                (score > 70) | (score < 30),  # 极端信号
                score + (score - 50) * 0.2,   # 增强极端信号
                50 + (score - 50) * 0.8       # 弱化中性信号
            )
        elif market_env == MarketEnvironment.BREAKOUT_MARKET:
            # 突破市场中，趋势信号更为重要
            momentum_factor = 0.3
            adjusted_score = 50 + (score - 50) * (1 + momentum_factor)
        
        # 转换为 pd.Series 并限制在 0-100 范围内
        if isinstance(adjusted_score, np.ndarray):
            adjusted_score = pd.Series(adjusted_score, index=score.index)
            
        return adjusted_score.clip(0, 100)
    
    def get_indicator_type(self) -> str:
        """获取指标类型（大写）"""
        # 尝试从类名获取
        class_name = self.__class__.__name__
        if class_name:
            return class_name.upper()
        
        # 尝试从实例名称获取
        if hasattr(self, 'name'):
            return self.name.upper()
        
        # 默认返回BASE_INDICATOR
        return "BASE_INDICATOR"
    
    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算指标结果的置信度
        
        Args:
            score: 评分序列
            patterns: 识别的形态列表
            signals: 生成的信号字典
            
        Returns:
            float: 置信度，范围0-1
        """
        # 初始置信度为0.5（中等）
        confidence = 0.5
        
        # 1. 根据形态数量调整置信度
        pattern_count = len(patterns)
        if pattern_count > 3:
            confidence += 0.1  # 多个形态同时出现，提高置信度
        elif pattern_count == 0:
            confidence -= 0.1  # 没有形态，降低置信度
        
        # 2. 根据得分极端程度调整置信度
        latest_score = score.iloc[-1] if len(score) > 0 else 50
        if latest_score > 80 or latest_score < 20:
            confidence += 0.15  # 极端评分，提高置信度
        elif 40 <= latest_score <= 60:
            confidence -= 0.05  # 中性评分，降低置信度
        
        # 3. 根据信号强度调整置信度
        for signal_name, signal_series in signals.items():
            if 'buy_strength' in signal_name or 'sell_strength' in signal_name:
                if len(signal_series) > 0 and signal_series.iloc[-1] >= 4:
                    confidence += 0.05  # 强信号，提高置信度
        
        # 4. 根据信号一致性调整置信度
        buy_signal = signals.get('buy_signal', pd.Series(False, index=score.index))
        sell_signal = signals.get('sell_signal', pd.Series(False, index=score.index))
        
        if len(buy_signal) > 0 and len(sell_signal) > 0:
            latest_buy = buy_signal.iloc[-1]
            latest_sell = sell_signal.iloc[-1]
            
            if latest_buy and latest_sell:
                confidence -= 0.2  # 买卖信号同时出现，显著降低置信度
            elif not latest_buy and not latest_sell:
                confidence -= 0.05  # 无买卖信号，略微降低置信度
        
        # 5. 根据市场环境适应性调整置信度
        # 这里已经在评分中考虑了市场环境，可以不再额外调整置信度
        
        # 6. 根据特定形态提高置信度
        high_confidence_patterns = [
            "金叉", "死叉", "底背离", "顶背离", "突破", "支撑", "阻力",
            "超买", "超卖", "高位钝化", "低位钝化", "突破上轨", "跌破下轨"
        ]
        
        for pattern in patterns:
            for high_conf_pattern in high_confidence_patterns:
                if high_conf_pattern in pattern:
                    confidence += 0.05
                    break
        
        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成信号（为了兼容性保留的方法）
        
        Args:
            data: 输入数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 包含信号的DataFrame
        """
        logger.warning("generate_signals方法已废弃，请使用generate_trading_signals方法")
        
        # 调用新的方法获取信号字典
        signals_dict = self.generate_trading_signals(data, *args, **kwargs)
            
        # 转换为DataFrame
        signals_df = pd.DataFrame(index=data.index)
        for name, series in signals_dict.items():
            signals_df[name] = series
        
        return signals_df
    
    def compute(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算指标（为了兼容性保留的方法）
        
        相当于calculate方法的别名
        
        Args:
            data: 输入数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 计算结果
        """
        return self.calculate(data, *args, **kwargs)
    
    def safe_compute(self, data: pd.DataFrame, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        安全计算指标，捕获异常
        
        Args:
            data: 输入数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Optional[pd.DataFrame]: 计算结果，如果发生异常则返回None
        """
        try:
            return self.calculate(data, *args, **kwargs)
        except Exception as e:
            self._error = e
            logger.error(f"计算 {self.name} 指标时出错: {str(e)}", exc_info=True)
            return None
    
    def get_column_name(self, suffix: str = "") -> str:
        """
        获取指标列名
        
        Args:
            suffix: 列名后缀
            
        Returns:
            str: 指标列名
        """
        return f"{self.name.lower()}{'_' + suffix if suffix else ''}"
    
    @staticmethod
    def ensure_columns(data: pd.DataFrame, required_columns: List[str]) -> None:
        """
        确保DataFrame中包含所有必需的列

        Args:
            data: 输入的DataFrame
            required_columns: 必需的列名列表

        Raises:
            ValueError: 如果缺少任何必需的列
        """
        if not required_columns:
            return

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            error_message = f"输入数据缺少所需的列: {', '.join(missing_columns)}"
            logger.error(error_message)
            raise ValueError(error_message)
    
    @staticmethod
    def crossover(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
        """
        检测series1是否从下方穿过series2
        
        Args:
            series1: 第一个序列
            series2: 第二个序列或固定值
            
        Returns:
            pd.Series: 布尔序列，True表示发生穿越
        """
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)
            
        # 确保两个序列长度相同
        min_len = min(len(series1), len(series2))
        series1 = series1.iloc[-min_len:]
        series2 = series2.iloc[-min_len:]
        
        # 检测当前值大于等于series2，前一个值小于series2
        return (series1 >= series2) & (series1.shift(1) < series2.shift(1))
    
    @staticmethod
    def crossunder(series1: pd.Series, series2: Union[pd.Series, float, int]) -> pd.Series:
        """
        检测series1是否从上方穿过series2
        
        Args:
            series1: 第一个序列
            series2: 第二个序列或固定值
            
        Returns:
            pd.Series: 布尔序列，True表示发生穿越
        """
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)
            
        # 确保两个序列长度相同
        min_len = min(len(series1), len(series2))
        series1 = series1.iloc[-min_len:]
        series2 = series2.iloc[-min_len:]
        
        # 检测当前值小于等于series2，前一个值大于series2
        return (series1 <= series2) & (series1.shift(1) > series2.shift(1))
    
    @staticmethod
    def sma(series: pd.Series, periods: int) -> pd.Series:
        """
        计算简单移动平均线
        
        Args:
            series: 输入序列
            periods: 周期
            
        Returns:
            pd.Series: SMA序列
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
            pd.Series: EMA序列
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
            pd.Series: 周期内最高值序列
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
            pd.Series: 周期内最低值序列
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
            pd.Series: ATR序列
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=periods).mean()
        
        return atr
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将指标转换为字典表示
        
        Returns:
            Dict[str, Any]: 指标的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'weight': self.weight,
            'has_result': self.has_result(),
            'has_error': self.has_error(),
            'error': str(self._error) if self._error else None,
            'market_environment': self._market_environment.value if self._market_environment else None,
            'indicator_type': self.get_indicator_type()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}({self.description})"
    
    def __repr__(self) -> str:
        """表示对象的字符串"""
        return self.__str__()