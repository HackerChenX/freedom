from utils.decorators import performance_monitor, exception_handler
from utils.container import container
from utils.logger import get_logger

"""
技术指标基类模块
提供所有技术指标的统一基础架构
"""

import abc
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


logger = get_logger(__name__)


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
    - 作为L4层核心服务组件，承担多项相关职责
    - 方法分为以下职责组:
      * 核心抽象方法 (calculate, get_signal, get_patterns)
      * 数据处理方法 (validate_data, preprocess_data, postprocess_result)
      * 扩展点方法 (initialize_indicator, register_patterns)
      * 工具方法 (format_output, get_metadata)
    - 符合L4层组件化架构设计原则
    - 基于L3层成功经验的职责分组模式

    技术指标基类

    所有技术指标类应继承此类，并实现必要的抽象方法
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

        # 使用依赖注入获取服务
        try:
            self.data_access = container.resolve("DataAccessInterface")
            self.cache_service = container.resolve("ICacheService")
        except Exception:
            # 如果依赖注入失败，使用默认值
            self.data_access = None
            self.cache_service = None

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

    @abc.abstractmethod
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值

        Args:
            data: 输入数据，包含OHLCV等字段

        Returns:
            pd.DataFrame: 包含指标计算结果的数据框

        Raises:
            ValueError: 当输入数据不符合要求时
        """
        pass

    @abc.abstractmethod
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取交易信号

        Args:
            data: 包含指标计算结果的数据

        Returns:
            Dict[str, Any]: 交易信号信息
        """
        pass

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
