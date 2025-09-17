"""
分析器基础类
基于L3层成功经验设计的统一分析器基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from utils.decorators import performance_monitor, exception_handler
from utils.container import container
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseAnalyzer(ABC):
    """
    分析器基础类 - 所有分析器必须继承此类
    
    基于L3层成功经验设计，提供统一的分析器接口和功能
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        初始化分析器
        
        Args:
            name: 分析器名称
            description: 分析器描述
        """
        self.name = name
        self.description = description
        self._result = None
        self._error = None
        
        # 使用依赖注入获取服务
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
    
    @abstractmethod
    @performance_monitor(threshold_seconds=3.0)
    @exception_handler(reraise=True)
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        执行分析
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        pass
    
    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """获取分析结果"""
        return self._result
    
    @property
    def error(self) -> Optional[Exception]:
        """获取分析错误"""
        return self._error
    
    def has_result(self) -> bool:
        """检查是否有分析结果"""
        return self._result is not None
    
    def has_error(self) -> bool:
        """检查是否有分析错误"""
        return self._error is not None
