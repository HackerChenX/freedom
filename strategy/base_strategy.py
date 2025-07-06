"""
选股策略基类模块

提供选股策略的通用接口和功能
"""

import abc
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd

from utils.logger import getLogger

logger = getLogger(__name__)


class BaseStrategy(abc.ABC):
    """
    选股策略基类
    
    所有选股策略类应继承此类，并实现必要的抽象方法
    """
    
    def __init___69(self, name: str, description: str = ""):
        """
        初始化选股策略
        
        Args:
            name: 策略名称
            description: 策略描述
        """
        # 使用依赖注入获取服务
        from utils.dependency_injection import get_service
        from db.interfaces.data_access_interface import Data_access_interface
        from db.interfaces.indicator_calculator_interface import IIndicator_calculator
        
        self.data_access = get_service(Data_access_interface)
        self.indicator_calculator = get_service(IIndicator_calculator)
        
        self.name = name
        self.description = description
        self._result = None
        self._error = None
        self._parameters = {}
    
    @property
    def result_Strategy(self) -> Optional[pd.DataFrame]:
        """获取选股结果"""
        return self._result
    
    @property
    def error_Strategy(self) -> Optional[Exception]:
        """获取错误信息"""
        return self._error
    
    @property
    def parameters_Strategy(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self._parameters.copy()
    
    def set_parameter_Base_Strategy(self, key: str, value: Any) -> None:
        """
        设置策略参数
        
        Args:
            key: 参数名
            value: 参数值
        """
        self._parameters[key] = value
    
    def set_parameters_Strategy(self, params: Dict[str, Any]) -> None:
        """
        批量设置策略参数
        
        Args:
            params: 参数字典
        """
        self._parameters.update(params)
    
    def has_result_Strategy(self) -> bool:
        """检查是否有选股结果"""
        return self._result is not None
    
    def has_error_Strategy(self) -> bool:
        """检查是否有错误"""
        return self._error is not None
    
    @abc.abstractmethod
    def select_Strategy_Base_Strategy(self, universe: List[str], *args, **kwargs) -> pd.DataFrame:
        """
        执行选股策略
        
        Args:
            universe: 股票代码列表，表示选股范围
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 选股结果，包含股票代码、名称等信息
        """
        pass
    
    def run_Base_Strategy(self, universe: List[str], *args, **kwargs) -> pd.DataFrame:
        """
        运行选股策略并处理异常
        
        Args:
            universe: 股票代码列表，表示选股范围
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 选股结果
        
        Raises:
            Exception: 选股过程中出现异常
        """
        try:
            self._result = self.select_Strategy_Base_Strategy(universe, *args, **kwargs)
            self._error = None
            return self._result
        except Exception as e:
            logger.error_Strategy(f"执行选股策略 {self.name} 时出错: {e}")
            self._error = e
            self._result = None
            raise
    
    def safe_run_Strategy(self, universe: List[str], *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        安全运行选股策略，不抛出异常
        
        Args:
            universe: 股票代码列表，表示选股范围
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            Optional[pd.DataFrame]: 选股结果，如果出错则返回None
        """
        try:
            return self.run_Base_Strategy(universe, *args, **kwargs)
        except Exception as e:
            logger.error_Strategy(f"安全运行选股策略 {self.name} 时出错: {e}")
            return None
    
    def to_dict_Strategy(self) -> Dict[str, Any]:
        """
        将策略转换为字典表示
        
        Returns:
            Dict[str, Any]: 策略的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self._parameters,
            'has_result': self.has_result_Strategy(),
            'has_error': self.has_error_Strategy(),
            'error': str(self._error) if self._error else None
        }
    
    def __str___Base_Strategy(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr___Base_Strategy(self) -> str:
        """开发者字符串表示"""
        return self.__str___Base_Strategy() 