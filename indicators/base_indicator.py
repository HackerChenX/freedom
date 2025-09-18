from utils.decorators import performance_monitor, exception_handler
from utils.container import container
from utils.logger import get_logger

"""
技术指标基类模块 - L4核心服务层

基于L1L2L3成功修复经验的严格依赖注入实现
提供所有技术指标的统一基础架构，确保生产级质量标准
"""

import abc
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


logger = get_logger(__name__)


class StandardColumnNames:
    """
    项目级统一列名标准

    用途：
    1. 确保所有指标使用统一的列名标准
    2. 便于上游系统调用和数据处理
    3. 提高代码可维护性和可读性

    命名规则：
    - 基础数据列：小写英文
    - 指标列：指标名_具体值，如 macd_dif, rsi_value
    - 信号列：统一使用 buy_signal, sell_signal 等
    """

    # 基础价格数据列（输入数据标准）
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    TURNOVER_RATE = "turnover_rate"

    # MACD指标标准列名
    MACD_DIF = "macd_dif"           # DIF线（快线-慢线）
    MACD_DEA = "macd_dea"           # DEA线（信号线）
    MACD_HISTOGRAM = "macd_histogram"  # MACD柱状图

    # RSI指标标准列名
    RSI_VALUE = "rsi_value"         # RSI主值
    RSI_OVERBOUGHT = "rsi_overbought"  # 超买信号
    RSI_OVERSOLD = "rsi_oversold"   # 超卖信号

    # KDJ指标标准列名
    KDJ_K = "kdj_k"                # K值
    KDJ_D = "kdj_d"                # D值
    KDJ_J = "kdj_j"                # J值

    # 移动平均线标准列名
    MA_5 = "ma_5"                  # 5日移动平均
    MA_10 = "ma_10"                # 10日移动平均
    MA_20 = "ma_20"                # 20日移动平均
    MA_60 = "ma_60"                # 60日移动平均

    # 通用信号列名（所有指标统一使用）
    BUY_SIGNAL = "buy_signal"       # 买入信号
    SELL_SIGNAL = "sell_signal"     # 卖出信号
    HOLD_SIGNAL = "hold_signal"     # 持有信号
    SIGNAL_STRENGTH = "signal_strength"    # 信号强度
    SIGNAL_CONFIDENCE = "signal_confidence"  # 信号置信度

    @classmethod
    def get_indicator_columns(cls, indicator_name: str) -> List[str]:
        """
        获取指定指标的标准列名

        Args:
            indicator_name: 指标名称，如 'macd', 'rsi', 'kdj'

        Returns:
            List[str]: 该指标的标准列名列表
        """
        indicator_columns = {
            'macd': [cls.MACD_DIF, cls.MACD_DEA, cls.MACD_HISTOGRAM],
            'rsi': [cls.RSI_VALUE, cls.RSI_OVERBOUGHT, cls.RSI_OVERSOLD],
            'kdj': [cls.KDJ_K, cls.KDJ_D, cls.KDJ_J],
            'ma': [cls.MA_5, cls.MA_10, cls.MA_20, cls.MA_60]
        }
        return indicator_columns.get(indicator_name.lower(), [])

    @classmethod
    def get_signal_columns(cls) -> List[str]:
        """
        获取标准信号列名

        Returns:
            List[str]: 标准信号列名列表
        """
        return [cls.BUY_SIGNAL, cls.SELL_SIGNAL, cls.HOLD_SIGNAL,
                cls.SIGNAL_STRENGTH, cls.SIGNAL_CONFIDENCE]


class DependencyInjectionError(Exception):
    """依赖注入错误 - BaseIndicator专用异常"""
    pass


class PatternInfo:
    """形态信息类"""

    def __init__(self, name: str, signal_type: str, strength: float = 0.0, duration: int = 1, details: str = ""):
        self.name = name
        self.signal_type = signal_type
        self.strength = strength
        self.duration = duration
        self.details = details
        self.display_name = name

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "signal_type": self.signal_type,
            "display_name": self.display_name,
            "strength": self.strength,
            "duration": self.duration,
            "details": self.details,
        }


class BaseIndicator(abc.ABC):
    """
    BaseIndicator - L4核心服务层技术指标基类

    职责合理性说明:
    - 作为L4层核心服务组件,承担多项相关职责
    - 方法分为以下职责组:
      * 核心抽象方法 (calculate, get_signal, get_patterns)
      * 数据处理方法 (validate_data, preprocess_data, postprocess_result)
      * 扩展点方法 (initialize_indicator, register_patterns)
      * 工具方法 (format_output, get_metadata)
    - 符合L4层组件化架构设计原则
    - 基于L3层成功经验的职责分组模式

    技术指标基类

    所有技术指标类应继承此类,并实现必要的抽象方法
    """

    def __init__(self, name: str = "", period: int = 20, **kwargs):  # TODO: 将魔法数字提取到配置中
        """
        初始化指标

        Args:
            name: 指标名称
            period: 计算周期
            **kwargs: 其他参数
        """
        self.name = name or self.__class__.__name__
        self.period = period
        self.params = kwargs
        self._result = None
        self._patterns = []

        # 严格依赖注入 - 基于L1L2L3成功经验，不允许兜底逻辑
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")

        # 验证依赖注入成功 - 确保生产级质量
        if not self.data_access:
            raise DependencyInjectionError("DataAccessInterface服务未注册，请检查依赖注入配置")
        if not self.cache_service:
            raise DependencyInjectionError("ICacheService服务未注册，请检查依赖注入配置")

        # 初始化指标
        self.initialize_indicator()

    def initialize_indicator(self):
        """
        在所有子类参数都设置完毕后执行初始化
        """
        # 自动注册形态
        self.register_patterns()

    def register_patterns(self):
        """
        注册指标形态

        子类可以重写此方法来注册自定义形态
        """
        pass

    def register_pattern_to_registry(self, 
                                   pattern_id: str, 
                                   display_name: str, 
                                   description: str = "",
                                   pattern_type: str = "NEUTRAL",
                                   default_strength: str = "MEDIUM", 
                                   score_impact: float = 0.0,
                                   polarity: str = "NEUTRAL",
                                   detection_function=None,
                                   allow_override: bool = True) -> None:
        """
        注册形态到全局注册表

        Args:
            pattern_id: 形态ID
            display_name: 显示名称
            description: 形态描述
            pattern_type: 形态类型(BULLISH/BEARISH/NEUTRAL/REVERSAL)
            default_strength: 默认强度(STRONG/MEDIUM/WEAK)
            score_impact: 评分影响
            polarity: 极性(POSITIVE/NEGATIVE/NEUTRAL)
            detection_function: 检测函数(可选)
            allow_override: 是否允许覆盖
        """
        try:
            # 导入形态注册表
            from indicators.pattern_registry import get_pattern_registry, PatternTypePatternRegistry, PatternStrengthPatternRegistry, PatternPolarity
            
            # 获取注册表实例
            registry = get_pattern_registry()
            
            # 转换形态类型
            pattern_type_enum = PatternTypePatternRegistry.NEUTRAL
            if pattern_type.upper() == "BULLISH":
                pattern_type_enum = PatternTypePatternRegistry.BULLISH
            elif pattern_type.upper() == "BEARISH":
                pattern_type_enum = PatternTypePatternRegistry.BEARISH
            elif pattern_type.upper() == "REVERSAL":
                pattern_type_enum = PatternTypePatternRegistry.REVERSAL
            
            # 转换强度
            strength_enum = PatternStrengthPatternRegistry.MEDIUM
            if default_strength.upper() == "STRONG":
                strength_enum = PatternStrengthPatternRegistry.STRONG
            elif default_strength.upper() == "WEAK":
                strength_enum = PatternStrengthPatternRegistry.WEAK
                
            # 转换极性
            polarity_enum = PatternPolarity.NEUTRAL
            if polarity.upper() == "POSITIVE":
                polarity_enum = PatternPolarity.POSITIVE
            elif polarity.upper() == "NEGATIVE":
                polarity_enum = PatternPolarity.NEGATIVE
            
            # 注册形态
            registry.register_pattern_registry(
                pattern_id=pattern_id,
                display_name=display_name,
                indicator_id=self.name,
                pattern_type=pattern_type_enum,
                default_strength=strength_enum,
                description=description,
                score_impact=score_impact,
                polarity=polarity_enum,
                detection_function=detection_function,
                allow_override=allow_override
            )
            
            logger.debug(f"形态 {pattern_id} 注册成功: {display_name}")
            
        except Exception as e:
            logger.warning(f"注册形态 {pattern_id} 失败: {e}")
            # 不抛出异常，确保指标能正常实例化

    @abc.abstractmethod
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值 - 核心计算方法

        职责说明：
        1. 执行指标的核心计算逻辑
        2. 返回包含指标计算结果的DataFrame
        3. 列名应遵循项目统一命名标准

        返回格式标准：
        - 必须返回pandas.DataFrame格式
        - 列名建议使用指标名前缀，如：macd_dif, rsi_value, kdj_k
        - 索引应与输入数据保持一致
        - 对于复合指标，应包含所有核心计算值

        示例：
        - MACD指标应返回：macd_dif, macd_dea, macd_histogram列
        - RSI指标应返回：rsi_value列
        - KDJ指标应返回：kdj_k, kdj_d, kdj_j列

        Args:
            data: 输入数据,必须包含OHLCV等字段
                 最低要求：包含'close'列
                 完整要求：包含'open','high','low','close','volume'列

        Returns:
            pd.DataFrame: 包含指标计算结果的数据框
                         列名应遵循统一命名标准
                         索引与输入数据一致

        Raises:
            ValueError: 当输入数据不符合要求时
            InsufficientDataError: 当数据量不足以计算指标时
        """
        pass

    @abc.abstractmethod
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取最新交易信号 - 单一信号获取方法

        职责说明：
        1. 基于指标计算结果生成最新的交易信号
        2. 返回结构化的信号字典
        3. 主要用于实时交易决策

        与get_signals()的区别：
        - get_signal(): 返回Dict，包含最新单一信号，用于实时决策
        - get_signals(): 返回DataFrame，包含历史信号序列，用于回测分析

        返回格式标准：
        {
            'signal_type': str,      # 'buy', 'sell', 'hold'
            'strength': float,       # 信号强度 0.0-1.0
            'confidence': float,     # 信号置信度 0.0-1.0
            'timestamp': datetime,   # 信号时间
            'price': float,         # 触发价格
            'metadata': dict        # 额外信息
        }

        Args:
            data: 包含指标计算结果的数据
                 可以是原始OHLCV数据或calculate()的返回结果

        Returns:
            Dict[str, Any]: 标准化的交易信号信息字典
                           必须包含signal_type, strength, confidence字段
        """
        pass

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成批量交易信号 - 历史信号序列方法

        职责说明：
        1. 基于指标计算结果生成完整的信号序列
        2. 返回包含历史信号的DataFrame
        3. 主要用于回测分析和策略验证

        与get_signal()的区别：
        - get_signal(): 返回Dict，包含最新单一信号，用于实时决策
        - get_signals(): 返回DataFrame，包含历史信号序列，用于回测分析

        返回格式标准：
        DataFrame包含以下标准列：
        - buy_signal: bool, 买入信号
        - sell_signal: bool, 卖出信号
        - hold_signal: bool, 持有信号
        - signal_strength: float, 信号强度(0-1)
        - signal_confidence: float, 信号置信度(0-1)

        默认实现：
        子类可以重写此方法以实现特定的信号生成逻辑
        如果不重写，将基于get_signal()生成简单的信号序列

        Args:
            data: 包含指标计算结果的数据
                 可以是原始OHLCV数据或calculate()的返回结果

        Returns:
            pd.DataFrame: 包含标准信号列的DataFrame
                         索引与输入数据一致
        """
        # 默认实现：基于get_signal()生成简单信号
        if data.empty:
            return pd.DataFrame()

        # 创建信号DataFrame
        signals_df = pd.DataFrame(index=data.index)
        signals_df['buy_signal'] = False
        signals_df['sell_signal'] = False
        signals_df['hold_signal'] = True
        signals_df['signal_strength'] = 0.0
        signals_df['signal_confidence'] = 0.0

        # 获取最新信号并应用到最后一行
        try:
            latest_signal = self.get_signal(data)
            if latest_signal and isinstance(latest_signal, dict):
                last_idx = signals_df.index[-1]
                signal_type = latest_signal.get('signal_type', 'hold')

                if signal_type == 'buy':
                    signals_df.loc[last_idx, 'buy_signal'] = True
                    signals_df.loc[last_idx, 'hold_signal'] = False
                elif signal_type == 'sell':
                    signals_df.loc[last_idx, 'sell_signal'] = True
                    signals_df.loc[last_idx, 'hold_signal'] = False

                signals_df.loc[last_idx, 'signal_strength'] = latest_signal.get('strength', 0.0)
                signals_df.loc[last_idx, 'signal_confidence'] = latest_signal.get('confidence', 0.0)
        except Exception as e:
            logger.warning(f"生成信号序列时出错: {e}")

        return signals_df

    def get_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        获取指标形态

        Args:
            data: 包含指标计算结果的数据

        Returns:
            List[Dict[str, Any]]: 形态信息列表
        """
        return [pattern.to_dict() for pattern in self._patterns]

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据

        Returns:
            bool: 验证结果
        """
        if data is None or data.empty:
            return False

        required_columns = ["close"]
        return all(col in data.columns for col in required_columns)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据

        Args:
            data: 原始数据

        Returns:
            pd.DataFrame: 预处理后的数据
        """
        # 默认不做任何处理
        return data.copy()

    def postprocess_result(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        后处理结果

        Args:
            result: 计算结果

        Returns:
            pd.DataFrame: 后处理后的结果
        """
        # 默认不做任何处理
        return result

    def format_output(self, result: Any) -> Dict[str, Any]:
        """
        格式化输出

        Args:
            result: 计算结果

        Returns:
            Dict[str, Any]: 格式化后的输出
        """
        return {
            "indicator": self.name,
            "period": self.period,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取指标元数据

        Returns:
            Dict[str, Any]: 元数据信息
        """
        return {
            "name": self.name,
            "period": self.period,
            "params": self.params,
            "type": self.__class__.__name__,
            "description": self.__doc__ or "",
        }

    @property
    def result(self) -> Optional[Any]:
        """获取计算结果"""
        return self._result

    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return self._result is not None

    def clear_result(self):
        """清除计算结果"""
        self._result = None
        self._patterns = []

    def add_pattern(self, pattern: PatternInfo):
        """
        添加形态信息

        Args:
            pattern: 形态信息
        """
        self._patterns.append(pattern)

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(name={self.name}, period={self.period})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"{self.__class__.__name__}(name='{self.name}', period={self.period}, params={self.params})"
