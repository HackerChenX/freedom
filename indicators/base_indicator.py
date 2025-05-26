"""
技术指标基类模块

提供技术指标计算的通用接口和功能
"""

import abc
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Callable

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseIndicator(abc.ABC):
    """
    技术指标基类
    
    所有技术指标类应继承此类，并实现必要的抽象方法
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化技术指标
        
        Args:
            name: 指标名称，可选参数，如果未提供则使用子类的name属性
            description: 指标描述
        """
        # 如果未提供name，则尝试使用子类中定义的name属性
        if not name and hasattr(self, 'name'):
            pass  # 已经有name属性，不需要重新赋值
        else:
            self.name = name
            
        self.description = description
        self._result = None
        self._error = None
    
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
    def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        计算两个序列的上穿信号
        
        Args:
            series1: 第一个序列
            series2: 第二个序列
            
        Returns:
            pd.Series: 上穿信号序列，上穿为True，否则为False
        """
        series1, series2 = pd.Series(series1), pd.Series(series2)
        return (series1.shift(1) < series2.shift(1)) & (series1 > series2)
    
    @staticmethod
    def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        计算两个序列的下穿信号
        
        Args:
            series1: 第一个序列
            series2: 第二个序列
            
        Returns:
            pd.Series: 下穿信号序列，下穿为True，否则为False
        """
        series1, series2 = pd.Series(series1), pd.Series(series2)
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
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(window=periods).mean()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将指标转换为字典表示
        
        Returns:
            Dict[str, Any]: 指标的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'has_result': self.has_result(),
            'has_error': self.has_error(),
            'error': str(self._error) if self._error else None
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        """开发者字符串表示"""
        return self.__str__() 