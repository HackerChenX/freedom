"""
IAnalysisEngine - L4层标准接口
基于L3层成功经验设计的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd


class IAnalysisEngine(ABC):
    """
    IAnalysisEngine - L4层标准接口
    
    基于L3层成功经验设计，提供统一的接口规范
    """
    
    @abstractmethod
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        执行核心功能
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 验证结果
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取元数据信息
        
        Returns:
            Dict[str, Any]: 元数据
        """
        pass
