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
from indicators.pattern_registry import PatternRegistry

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
        self.is_available = False # 默认指标为不可用
        self._result = None
        self._error = None
        self._score_cache = {}
        self._market_environment = MarketEnvironment.SIDEWAYS_MARKET  # 默认市场环境
        
    def initialize(self):
        """
        在所有子类参数都设置完毕后执行初始化
        """
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
                                   detection_function=None,
                                   polarity: str = None):
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
            polarity: 模式极性（POSITIVE, NEGATIVE, NEUTRAL），用于买点分析过滤
        """
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength, PatternPolarity

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
        
        # 转换极性字符串为枚举
        polarity_enum = None
        if polarity:
            polarity_upper = polarity.upper()
            if polarity_upper == "POSITIVE":
                polarity_enum = PatternPolarity.POSITIVE
            elif polarity_upper == "NEGATIVE":
                polarity_enum = PatternPolarity.NEGATIVE
            elif polarity_upper == "NEUTRAL":
                polarity_enum = PatternPolarity.NEUTRAL

        # 注册形态
        registry.register(
            pattern_id=pattern_id,
            display_name=display_name,
            description=description,
            indicator_id=indicator_id,
            pattern_type=pattern_type_enum,
            default_strength=strength_enum,
            score_impact=score_impact,
            detection_function=detection_function,
            polarity=polarity_enum
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
        """检查指标是否已经计算过"""
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

        # 记录调试信息
        logger.debug(f"指标 {self.name}: 源数据列 {list(source_df.columns)}, 结果列 {list(result_df.columns)}, 基础列 {available_columns}")

        # 如果结果中缺少这些列，则从源数据中复制
        for col in available_columns:
            if col not in result_df.columns:
                try:
                    # 确保源数据的列是Series而不是DataFrame
                    if isinstance(source_df[col], pd.Series):
                        result_df[col] = source_df[col]
                        logger.debug(f"指标 {self.name}: 成功复制基础列 {col}")
                    elif isinstance(source_df[col], pd.DataFrame):
                        # 如果是DataFrame，取第一列
                        result_df[col] = source_df[col].iloc[:, 0]
                        logger.debug(f"指标 {self.name}: 从DataFrame复制基础列 {col}")
                    else:
                        # 其他情况，尝试直接赋值
                        result_df[col] = source_df[col]
                        logger.debug(f"指标 {self.name}: 直接复制基础列 {col}")
                except Exception as e:
                    logger.error(f"指标 {self.name}: 无法复制列 {col}: {e}")
            else:
                logger.debug(f"指标 {self.name}: 基础列 {col} 已存在于结果中")

        # 验证结果
        missing_base_cols = [col for col in available_columns if col not in result_df.columns]
        if missing_base_cols:
            logger.error(f"指标 {self.name}: 保留基础列后仍缺少: {missing_base_cols}")

        return result_df
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算指标，并处理异常和结果保存
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含指标计算结果的DataFrame
        """
        try:
            # 确保必需的列存在
            if hasattr(self, 'REQUIRED_COLUMNS'):
                missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"指标 {self.name} 计算缺少必需列: {missing_cols}")

            # 调用核心计算逻辑
            result_df = self._calculate(data, *args, **kwargs)
            
            # 检查结果是否为DataFrame
            if not isinstance(result_df, pd.DataFrame):
                raise TypeError(f"指标 {self.name} 的_calculate方法必须返回pandas.DataFrame")

            # 保留原始基础数据列
            self._result = self._preserve_base_columns(data, result_df)
            self._error = None
            self.is_available = True
            
            return self._result

        except Exception as e:
            logger.error(f"计算指标 '{self.name}' 时出错: {e}", exc_info=True)
            self._error = e
            self._result = None
            self.is_available = False
            # 在出错时返回一个空的DataFrame，结构与输入数据保持一致
            return pd.DataFrame(index=data.index)

    @abc.abstractmethod
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        核心计算逻辑，必须由子类实现
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含指标计算结果的DataFrame
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
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型名称，默认为类名的大写形式
        
        Returns:
            str: 指标类型名称
        """
        return self.__class__.__name__.upper()

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算指标的综合评分
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 包含评分和置信度的字典
        """
        try:
            # 1. 计算原始得分
            raw_score = self.calculate_raw_score(data, **kwargs)
            
            # 如果原始得分计算失败，返回中性分
            if raw_score.empty:
                return {'score': 50.0, 'confidence': 0.5}
            
            last_raw_score = raw_score.iloc[-1]
            
            # 2. 获取形态和信号
            patterns = self.get_patterns(data, **kwargs)
            
            # 3. 计算置信度
            confidence = self.calculate_confidence(raw_score, patterns, {})
            
            # 4. 最终得分
            final_score = last_raw_score * confidence
            
            # 确保分数在0-100之间
            final_score = max(0, min(100, final_score))
            
            return {
                'score': final_score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"为指标 {self.name} 计算评分时出错: {e}")
            return {'score': 50.0, 'confidence': 0.0}

    @abc.abstractmethod
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算指标的原始评分，范围0-100
        必须由子类实现
        """
        raise NotImplementedError("子类必须实现 calculate_raw_score 方法")

    @abc.abstractmethod
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        获取指标的所有技术形态
        必须由子类实现
        """
        raise NotImplementedError("子类必须实现 get_patterns 方法")

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
        # 首先确保指标已计算
        if not self.has_result():
            self.calculate(data, **kwargs)

        # 如果计算后仍无结果，返回None
        if not self.has_result():
            return None
            
        # 调用子类实现的get_patterns方法
        return self.get_patterns(self.result, **kwargs)
    
    def _detect_patterns_from_registry(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用在注册表中定义的检测函数来识别形态
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含形态信号的DataFrame
        """
        registry = PatternRegistry()
        indicator_patterns = registry.get_patterns_by_indicator(self.get_indicator_type())
        
        patterns_df = pd.DataFrame(index=data.index)
        
        for pattern_id in indicator_patterns:
            pattern_info = registry.get_pattern(pattern_id)
            
            if not pattern_info:
                continue
                
            detection_func = pattern_info.get('detection_function')
            display_name = pattern_info.get('display_name')

            if callable(detection_func) and display_name:
                try:
                    # 调用检测函数
                    detection_result = detection_func(data)
                    
                    # 将结果添加到DataFrame
                    if isinstance(detection_result, pd.Series):
                        patterns_df[display_name] = detection_result
                    elif isinstance(detection_result, bool):
                        if detection_result:
                            patterns_df.loc[patterns_df.index[-1], display_name] = True
                            
                except Exception as e:
                    logger.warning(f"为形态 '{display_name}' 调用检测函数失败: {e}")
            
        return patterns_df
    
    @abc.abstractmethod
    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算当前信号或评分的置信度
        
        必须由子类实现
        """
        raise NotImplementedError("子类必须实现 calculate_confidence 方法")

    def get_column_name(self, suffix: str = "") -> str:
        """
        获取指标列名
        
        Args:
            suffix: 列名后缀
            
        Returns:
            str: 指标列名
        """
        base_name = self.get_indicator_type()
        if suffix:
            return f"{base_name}_{suffix}"
        return base_name

    def to_dict(self) -> Dict[str, Any]:
        """
        将指标实例的主要属性序列化为字典
        
        Returns:
            Dict[str, Any]: 指标的字典表示
        """
        params = {}
        if hasattr(self, '_parameters'):
            params = self._parameters
        elif hasattr(self, '__dict__'):
            # 提取公共属性作为参数
            params = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and isinstance(v, (int, float, str, bool))}

        return {
            'name': self.name,
            'description': self.description,
            'parameters': params,
            'has_result': self.has_result(),
            'has_error': self.has_error()
        }

    @abc.abstractmethod
    def set_parameters(self, **kwargs):
        """
        设置指标参数
        必须由子类实现
        """
        raise NotImplementedError("子类必须实现 set_parameters 方法")
    
    def __str__(self) -> str:
        return f"{self.name} Indicator"
        
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"

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
            'name': '技术指标分析',
            'description': f'基于技术指标的分析: {pattern_id}',
            'type': 'NEUTRAL'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
