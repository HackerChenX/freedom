"""
指标计算器接口定义

定义技术指标计算的标准接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np

from enums.indicator_types import IndicatorType


class IIndicatorCalculator(ABC):
    """
    指标计算器接口
    
    定义所有技术指标计算的标准接口
    """
    
    @abstractmethod
    def calculate(self, 
                  data: pd.DataFrame, 
                  params: Optional[Dict[str, Any]] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        计算指标
        
        Args:
            data: 输入数据
            params: 计算参数
            
        Returns:
            Union[pd.Series, pd.DataFrame]: 计算结果
        """
        pass
    
    @abstractmethod
    def get_indicator_type(self) -> IndicatorType:
        """
        获取指标类型
        
        Returns:
            IndicatorType: 指标类型枚举
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        获取计算所需的数据列
        
        Returns:
            List[str]: 必需的数据列名列表
        """
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        获取默认参数
        
        Returns:
            Dict[str, Any]: 默认参数字典
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 验证结果
        """
        pass


class ITrendIndicator(IIndicatorCalculator):
    """
    趋势指标接口
    """
    
    @abstractmethod
    def get_trend_direction(self, data: pd.DataFrame) -> pd.Series:
        """
        获取趋势方向
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 趋势方向序列
        """
        pass


class IVolatilityIndicator(IIndicatorCalculator):
    """
    波动率指标接口
    """
    
    @abstractmethod
    def get_volatility_level(self, data: pd.DataFrame) -> pd.Series:
        """
        获取波动率水平
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 波动率水平序列
        """
        pass


class IMomentumIndicator(IIndicatorCalculator):
    """
    动量指标接口
    """
    
    @abstractmethod
    def get_momentum_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        获取动量信号
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 动量信号数据
        """
        pass


class IVolumeIndicator(IIndicatorCalculator):
    """
    成交量指标接口
    """
    
    @abstractmethod
    def get_volume_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        获取成交量分析
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 成交量分析数据
        """
        pass


class IPatternIndicator(IIndicatorCalculator):
    """
    形态指标接口
    """
    
    @abstractmethod
    def detect_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        检测形态模式
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 形态检测结果
        """
        pass


class ICompositeIndicator(IIndicatorCalculator):
    """
    复合指标接口
    """
    
    @abstractmethod
    def get_component_indicators(self) -> List[IIndicatorCalculator]:
        """
        获取组成指标列表
        
        Returns:
            List[IIndicatorCalculator]: 组成指标列表
        """
        pass
    
    @abstractmethod
    def combine_signals(self, signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        组合信号
        
        Args:
            signals: 各组成指标的信号
            
        Returns:
            pd.DataFrame: 组合信号结果
        """
        pass


class IIndicatorFactory(ABC):
    """
    指标工厂接口
    """
    
    @abstractmethod
    def create_indicator(self, 
                        indicator_type: Union[str, IndicatorType], 
                        params: Optional[Dict[str, Any]] = None) -> IIndicatorCalculator:
        """
        创建指标实例
        
        Args:
            indicator_type: 指标类型
            params: 初始化参数
            
        Returns:
            IIndicatorCalculator: 指标计算器实例
        """
        pass
    
    @abstractmethod
    def get_available_indicators(self) -> List[str]:
        """
        获取可用指标列表
        
        Returns:
            List[str]: 可用指标名称列表
        """
        pass
    
    @abstractmethod
    def register_indicator(self, 
                          name: str, 
                          indicator_class: type, 
                          category: IndicatorType) -> None:
        """
        注册新指标
        
        Args:
            name: 指标名称
            indicator_class: 指标类
            category: 指标类别
        """
        pass 