"""
统一选股策略基类模块

整合BaseStrategy和EnhancedBaseStrategy的功能，
提供完整的选股策略基础设施，支持：
1. 依赖注入和接口分离
2. 多周期数据支持
3. 智能时间管理
4. 标准化的策略模式
"""

import abc
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

from utils.logger import getLogger
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from db.interfaces.indicator_calculator_interface import IindicatorCalculator
from enums.kline_period import KlinePeriod
from utils.decorators import exception_handler, performance_monitor

logger = getLogger(__name__)


@dataclass
class PeriodConfig:
    """周期配置类"""
    period: str  # 周期：'15m', '30m', '1h', '1d', '1w', '1M'
    period_name: str  # 周期名称：'15分钟', '30分钟', '1小时', '日线', '周线', '月线'
    kline_period: KlinePeriod  # K线周期枚举
    
    @classmethod
    def get_all_periods(cls) -> List['PeriodConfig']:
        """获取所有支持的周期配置"""
        return [
            cls('15m', '15分钟', KlinePeriod.MIN_15),
            cls('30m', '30分钟', KlinePeriod.MIN_30), 
            cls('1h', '1小时', KlinePeriod.MIN_60),
            cls('1d', '日线', KlinePeriod.DAILY),
            cls('1w', '周线', KlinePeriod.WEEKLY),
            cls('1M', '月线', KlinePeriod.MONTHLY)
        ]
    
    @classmethod
    def get_period_by_name(cls, period: str) -> Optional['PeriodConfig']:
        """根据周期名称获取配置"""
        for config in cls.get_all_periods():
            if config.period == period:
                return config
        return None


class UnifiedBaseStrategy(abc.ABC):
    """
    统一选股策略基类
    
    整合了原BaseStrategy和EnhancedBaseStrategy的功能，
    提供完整的策略基础设施支持
    """
    
    def __init__(self, name: str, description: str = "", 
                 default_period: str = '1d', enable_multi_period: bool = False):
        """
        初始化统一策略基类
        
        Args:
            name: 策略名称
            description: 策略描述
            default_period: 默认周期
            enable_multi_period: 是否启用多周期支持
        """
        # 使用依赖注入获取服务
        self.data_access = get_service(DataAccessInterface)
        self.indicator_calculator = get_service(IindicatorCalculator)
        
        # 基本属性
        self.name = name
        self.description = description
        self.default_period = default_period
        self.enable_multi_period = enable_multi_period
        
        # 状态管理
        self._result = None
        self._error = None
        self._parameters = {}
        self._last_execution_time = None
        
        # 周期配置
        self.supported_periods = PeriodConfig.get_all_periods()
        self.current_period_config = PeriodConfig.get_period_by_name(default_period)
        
        logger.info(f"初始化策略: {name}, 默认周期: {default_period}")
    
    @property
    def result(self) -> Optional[pd.DataFrame]:
        """获取选股结果"""
        return self._result
    
    @property
    def error(self) -> Optional[str]:
        """获取错误信息"""
        return self._error
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self._parameters.copy()
    
    def set_parameters(self, **kwargs) -> None:
        """
        设置策略参数
        
        Args:
            **kwargs: 参数键值对
        """
        self._parameters.update(kwargs)
        logger.debug(f"策略 {self.name} 参数已更新: {kwargs}")
    
    def get_parameter(self, key: str, default_value: Any = None) -> Any:
        """
        获取策略参数
        
        Args:
            key: 参数键
            default_value: 默认值
            
        Returns:
            参数值
        """
        return self._parameters.get(key, default_value)
    
    def set_period(self, period: str) -> bool:
        """
        设置当前周期
        
        Args:
            period: 周期名称
            
        Returns:
            bool: 设置是否成功
        """
        config = PeriodConfig.get_period_by_name(period)
        if config:
            self.current_period_config = config
            logger.info(f"策略 {self.name} 周期已设置为: {period}")
            return True
        else:
            logger.warning(f"不支持的周期: {period}")
            return False
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def execute(self, universe: List[str], 
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                **kwargs) -> pd.DataFrame:
        """
        执行策略的统一入口
        
        Args:
            universe: 股票池
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 选股结果
        """
        try:
            # 清除之前的结果和错误
            self._result = None
            self._error = None
            
            # 设置执行时间
            self._last_execution_time = datetime.now()
            
            # 参数验证
            if not universe:
                raise ValueError("股票池不能为空")
            
            # 智能时间管理
            if not end_date:
                end_date = self._get_latest_trading_date()
            
            if not start_date:
                start_date = self._calculate_start_date(end_date)
            
            logger.info(f"执行策略 {self.name}, 股票池大小: {len(universe)}, "
                       f"时间范围: {start_date} - {end_date}")
            
            # 调用具体策略实现
            result = self.select_stocks(universe, start_date, end_date, **kwargs)
            
            # 结果验证和处理
            if result is not None:
                result = self._process_result(result)
                self._result = result
                logger.info(f"策略 {self.name} 执行完成，选出 {len(result)} 只股票")
            else:
                logger.warning(f"策略 {self.name} 未返回结果")
                self._result = pd.DataFrame()
            
            return self._result
            
        except Exception as e:
            error_msg = f"策略 {self.name} 执行失败: {str(e)}"
            logger.error(error_msg)
            self._error = error_msg
            raise
    
    @abc.abstractmethod
    def select_stocks_unified_base_strategy(self, universe: List[str], 
                     start_date: str, end_date: str, 
                     **kwargs) -> pd.DataFrame:
        """
        具体的选股逻辑实现
        
        Args:
            universe: 股票池
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 选股结果，必须包含code列
        """
        pass
    
    def get_stock_data(self, stock_code: str, 
                      start_date: str, end_date: str,
                      period: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票数据的统一接口
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期，不指定则使用当前周期
            
        Returns:
            pd.DataFrame: 股票数据
        """
        if not period:
            period_config = self.current_period_config
        else:
            period_config = PeriodConfig.get_period_by_name(period)
            
        if not period_config:
            raise ValueError(f"不支持的周期: {period}")
        
        return self.data_access.get_stock_data(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            period=period_config.kline_period
        )
    
    def calculate_indicator(self, stock_code: str, 
                          indicator_name: str,
                          start_date: str, end_date: str,
                          **params) -> pd.DataFrame:
        """
        计算技术指标的统一接口
        
        Args:
            stock_code: 股票代码
            indicator_name: 指标名称
            start_date: 开始日期
            end_date: 结束日期
            **params: 指标参数
            
        Returns:
            pd.DataFrame: 包含指标数据的DataFrame
        """
        # 获取基础数据
        data = self.get_stock_data(stock_code, start_date, end_date)
        
        if data.empty:
            return pd.DataFrame()
        
        # 计算指标
        return self.indicator_calculator.calculate(
            indicator_name=indicator_name,
            data=data,
            **params
        )
    
    def _get_latest_trading_date(self) -> str:
        """获取最新交易日期"""
        try:
            return self.data_access.get_latest_trading_date()
        except Exception as e:
            logger.warning(f"获取最新交易日期失败: {e}")
            return datetime.now().strftime('%Y-%m-%d')
    
    def _calculate_start_date(self, end_date: str, 
                            lookback_days: int = 120) -> str:
        """
        智能计算开始日期
        
        Args:
            end_date: 结束日期
            lookback_days: 回看天数
            
        Returns:
            str: 开始日期
        """
        try:
            # 使用数据访问接口计算
            return self.data_access.get_previous_trading_date(
                end_date, lookback_days
            )
        except Exception as e:
            logger.warning(f"计算开始日期失败: {e}")
            # 回退到简单计算
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=lookback_days * 1.5)  # 考虑非交易日
            return start_dt.strftime('%Y-%m-%d')
    
    def _process_result(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        处理和标准化选股结果
        
        Args:
            result: 原始结果
            
        Returns:
            pd.DataFrame: 处理后的结果
        """
        if result.empty:
            return pd.DataFrame(columns=['code', 'name', 'score', 'reason'])
        
        # 确保必需列存在
        if 'code' not in result.columns:
            raise ValueError("选股结果必须包含code列")
        
        # 添加策略信息
        result = result.copy()
        result['strategy'] = self.name
        result['execution_time'] = self._last_execution_time
        
        # 如果没有score列，设置默认分数
        if 'score' not in result.columns:
            result['score'] = 100.0
        
        # 排序
        if 'score' in result.columns:
            result = result.sort_values('score', ascending=False)
        
        return result
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """
        验证策略参数
        
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 基本验证
        if not self.name:
            errors.append("策略名称不能为空")
        
        if not self.current_period_config:
            errors.append("周期配置无效")
        
        # 子类可以重写此方法添加具体验证
        custom_errors = self._validate_custom_parameters()
        errors.extend(custom_errors)
        
        return len(errors) == 0, errors
    
    def _validate_custom_parameters(self) -> List[str]:
        """
        子类重写此方法实现自定义参数验证
        
        Returns:
            List[str]: 错误信息列表
        """
        return []
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            Dict[str, Any]: 策略信息
        """
        return {
            'name': self.name,
            'description': self.description,
            'default_period': self.default_period,
            'enable_multi_period': self.enable_multi_period,
            'supported_periods': [p.period for p in self.supported_periods],
            'current_period': self.current_period_config.period if self.current_period_config else None,
            'parameters': self.parameters,
            'last_execution_time': self._last_execution_time,
            'has_result': self._result is not None,
            'has_error': self._error is not None
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Strategy({self.name}, period={self.default_period})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"UnifiedBaseStrategy(name='{self.name}', "
                f"description='{self.description}', "
                f"default_period='{self.default_period}')")


# 为了向后兼容，创建别名
BASE_STRATEGY = UnifiedBaseStrategy
ENHANCED_BASE_STRATEGY = UnifiedBaseStrategy