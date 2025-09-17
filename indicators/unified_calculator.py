from typing import Dict, Any
from utils.container import container
from indicators.base_indicator import BaseIndicator
"""
统一指标计算基类模块

提供统一的指标计算基础设施,包括:
1. 标准化的指标计算接口
2. 通用的数据验证和处理
3. 统一的指标注册和管理  # TODO: 将魔法数字提取到配置中
4. 高效的批量计算支持  # TODO: 将魔法数字提取到配置中
"""

from utils.logger import get_logger

import abc
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.common_utils import DataProcessor, ValidationUtils, CacheUtils
from utils.decorators import exception_handler, performance_monitor
from enums.indicator_types import Indicatortype_indicator_types
from db.sql_manager import SQLManager, QueryType

# 创建兼容的枚举类
class IndicatorType(BaseIndicator):
    TREND = "trend"
    MOMENTUM = "momentum" 
    VOLUME = "volume"
    VOLATILITY = "volatility"
    COMPOSITE = "composite"

logger = get_logger(__name__)


class CalculationMode(BaseIndicator,Enum):
    """计算模式"""
    SINGLE = "single"      # 单次计算
    BATCH = "batch"        # 批量计算
    STREAMING = "streaming"  # 流式计算


@dataclass
class IndicatorResult(BaseIndicator):
    """指标计算结果"""
    name: str
    data: Union[pd.Series, pd.DataFrame]
    metadata: Dict[str, Any]
    calculation_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict_unified_calculator(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'data_shape': self.data.shape if hasattr(self.data, 'shape') else None,
            'metadata': self.metadata,
            'calculation_time': self.calculation_time,
            'success': self.success,
            'error_message': self.error_message
        }


class UnifiedIndicatorCalculator(BaseIndicator,abc.ABC):
    """
    统一指标计算基类
    
    所有技术指标都应该继承此类,确保接口一致性
    """
    
    def __init__(self, name: str, indicator_type: IndicatorType):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.name = name
        self.indicator_type = indicator_type
        self.cache_enabled = True
        self.cache_ttl = 300  # 5分钟缓存  # TODO: 将魔法数字提取到配置中
        
        # 性能统计
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        self.last_calculation_time = None
        
        logger.debug(f"初始化指标计算器: {name}")
    
    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame, **params) -> Union[pd.Series, pd.DataFrame]:
        """
        计算指标的核心方法
        
        Args:
            data: 输入数据
            **params: 计算参数
            
        Returns:
            Union[pd.Series, pd.DataFrame]: 计算结果
        """
        pass
    
    @abc.abstractmethod
    def get_required_columns_unified_calculator(self) -> List[str]:
        """
        获取计算所需的数据列
        
        Returns:
            List[str]: 必需的列名
        """
        pass
    
    @abc.abstractmethod
    def get_default_params_unified_calculator(self) -> Dict[str, Any]:
        """
        获取默认参数
        
        Returns:
            Dict[str, Any]: 默认参数字典
        """
        pass
    
    @abc.abstractmethod
    def validate_params_unified_calculator(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证参数有效性
        
        Args:
            params: 参数字典
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        pass
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=1.0)
    def calculate_with_validation(self, data: pd.DataFrame, **params) -> IndicatorResult:
        """
        带验证的计算方法
        
        Args:
            data: 输入数据
            **params: 计算参数
            
        Returns:
            IndicatorResult: 计算结果
        """
        start_time = datetime.now()
        
        try:
            # 1. 数据验证
            if not self._validate_input_data(data):
                raise ValueError(f"输入数据不符合要求: {self.name}")
            
            # 2. 参数验证
            merged_params = self.get_default_params()
            merged_params.update(params)
            
            is_valid, errors = self.validate_params(merged_params)
            if not is_valid:
                raise ValueError(f"参数验证失败: {errors}")
            
            # 3. 缓存检查  # TODO: 将魔法数字提取到配置中
            if self.cache_enabled:
                cache_key = self._generate_cache_key(data, merged_params)
                cached_result = CacheUtils.get(cache_key)
                if cached_result:
                    logger.debug(f"使用缓存结果: {self.name}")
                    return cached_result
            
            # 4. 执行计算  # TODO: 将魔法数字提取到配置中
            result_data = self.calculate(data, **merged_params)
            
            # 5. 结果验证  # TODO: 将魔法数字提取到配置中
            if not self._validate_result(result_data):
                raise ValueError(f"计算结果验证失败: {self.name}")
            
            # 6. 创建结果对象  # TODO: 将魔法数字提取到配置中
            calculation_time = (datetime.now() - start_time).total_seconds()
            result = IndicatorResult(
                name=self.name,
                data=result_data,
                metadata={
                    'params': merged_params,
                    'input_shape': data.shape,
                    'indicator_type': self.indicator_type.value if hasattr(self.indicator_type, 'value') else str(self.indicator_type)
                },
                calculation_time=calculation_time,
                success=True
            )
            
            # 7. 缓存结果  # TODO: 将魔法数字提取到配置中
            if self.cache_enabled:
                CacheUtils.set(cache_key, result, self.cache_ttl)
            
            # 8. 更新统计信息  # TODO: 将魔法数字提取到配置中
            self._update_stats(calculation_time)
            
            return result
            
        except Exception as e:
            calculation_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"指标计算失败: {self.name} - {str(e)}"
            logger.error(error_msg)
            
            return IndicatorResult(
                name=self.name,
                data=pd.DataFrame(),
                metadata={},
                calculation_time=calculation_time,
                success=False,
                error_message=error_msg
            )
    
    def calculate_batch(self, data_list: List[pd.DataFrame], **params) -> List[IndicatorResult]:
        """
        批量计算指标
        
        Args:
            data_list: 数据列表
            **params: 计算参数
            
        Returns:
            List[IndicatorResult]: 计算结果列表
        """
        results = []
        
        for i, data in enumerate(data_list):
            try:
                result = self.calculate_with_validation(data, **params)
                results.append(result)
            except Exception as e:
                logger.error(f"批量计算第{i}项失败: {e}")
                results.append(IndicatorResult(
                    name=self.name,
                    data=pd.DataFrame(),
                    metadata={},
                    calculation_time=0.0,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if data.empty:
            return False
        
        required_columns = self.get_required_columns()
        is_valid, missing_columns = ValidationUtils.validate_dataframe_schema(data, required_columns)
        
        if not is_valid:
            logger.warning(f"缺少必需列: {missing_columns}")
            return False
        
        return True
    
    def _validate_result(self, result: Union[pd.Series, pd.DataFrame]) -> bool:
        """验证计算结果"""
        if result is None:
            return False
        
        if isinstance(result, (pd.Series, pd.DataFrame)) and result.empty:
            return False
        
        return True
    
    def _generate_cache_key(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        data_hash = hash(str(data.values.tobytes()) + str(data.index.tolist()))
        params_hash = hash(str(sorted(params.items())))
        return f"{self.name}_{data_hash}_{params_hash}"
    
    def _update_stats(self, calculation_time: float) -> None:
        """更新统计信息"""
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        self.last_calculation_time = datetime.now()
    
    def get_stats_unified_calculator(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_time = self.total_calculation_time / self.calculation_count if self.calculation_count > 0 else 0
        
        return {
            'name': self.name,
            'calculation_count': self.calculation_count,
            'total_calculation_time': self.total_calculation_time,
            'average_calculation_time': avg_time,
            'last_calculation_time': self.last_calculation_time
        }
    
    def clear_cache_unified_calculator(self) -> None:
        """清除缓存"""
        CacheUtils.clear()
    
    def set_cache_config(self, enabled: bool = True, ttl: int = 300) -> None:  # TODO: 将魔法数字提取到配置中
        """设置缓存配置"""
        self.cache_enabled = enabled
        self.cache_ttl = ttl


class TrendIndicatorBase(BaseIndicator,UnifiedIndicatorCalculator):
    """趋势指标基类"""
    
    def __init__(self, name: str):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(name, IndicatorType.TREND)
    
    def get_trend_direction(self, data: pd.DataFrame) -> pd.Series:
        """
        获取趋势方向
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 趋势方向 (1: 上升, -1: 下降, 0: 震荡)
        """
        result = self.calculate(data)
        
        if isinstance(result, pd.Series):
            return np.sign(result.diff())
        elif isinstance(result, pd.DataFrame):
            # 如果返回多列,使用第一列
            return np.sign(result.iloc[:, 0].diff())
        else:
            return pd.Series(dtype=float)


class MomentumIndicatorBase(BaseIndicator,UnifiedIndicatorCalculator):
    """动量指标基类"""
    
    def __init__(self, name: str):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(name, IndicatorType.MOMENTUM)
    
    def get_momentum_signals(self, data: pd.DataFrame, 
                           overbought_threshold: float = 80,  # TODO: 将魔法数字提取到配置中
                           oversold_threshold: float = 20) -> pd.DataFrame:  # TODO: 将魔法数字提取到配置中
        """
        获取动量信号
        
        Args:
            data: 输入数据
            overbought_threshold: 超买阈值
            oversold_threshold: 超卖阈值
            
        Returns:
            pd.DataFrame: 动量信号
        """
        result = self.calculate(data)
        
        if isinstance(result, pd.Series):
            signals = pd.DataFrame(index=result.index)
            signals['value'] = result
            signals['overbought'] = result > overbought_threshold
            signals['oversold'] = result < oversold_threshold
            signals['signal'] = 0
            signals.loc[signals['overbought'], 'signal'] = -1  # 卖出信号
            signals.loc[signals['oversold'], 'signal'] = 1    # 买入信号
        else:
            signals = pd.DataFrame()
        
        return signals


class VolumeIndicatorBase(BaseIndicator,UnifiedIndicatorCalculator):
    """成交量指标基类"""
    
    def __init__(self, name: str):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(name, IndicatorType.VOLUME)
    
    def get_required_columns_unified_calculator(self) -> List[str]:
        """成交量指标通常需要volume列"""
        return ['volume']
    
    def get_volume_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        获取成交量分布
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 成交量分布
        """
        result = self.calculate(data)
        
        if isinstance(result, pd.Series):
            profile = pd.DataFrame(index=result.index)
            profile['volume_indicator'] = result
            profile['volume_raw'] = data['volume']
            profile['volume_ratio'] = result / data['volume'].rolling(20).mean()  # TODO: 将魔法数字提取到配置中
        else:
            profile = pd.DataFrame()
        
        return profile


class VolatilityIndicatorBase(BaseIndicator,UnifiedIndicatorCalculator):
    """波动率指标基类"""
    
    def __init__(self, name: str):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(name, IndicatorType.VOLATILITY)
    
    def get_volatility_level(self, data: pd.DataFrame) -> pd.Series:
        """
        获取波动率水平
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 波动率水平
        """
        result = self.calculate(data)
        
        if isinstance(result, pd.Series):
            return result
        elif isinstance(result, pd.DataFrame):
            # 如果返回多列,使用第一列
            return result.iloc[:, 0]
        else:
            return pd.Series(dtype=float)


class CompositeIndicatorBase(BaseIndicator,UnifiedIndicatorCalculator):
    """复合指标基类"""
    
    def __init__(self, name: str, component_indicators: List[UnifiedIndicatorCalculator]):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__(name, IndicatorType.COMPOSITE)
        self.component_indicators = component_indicators
    
    def calculate_components(self, data: pd.DataFrame, **params) -> Dict[str, IndicatorResult]:
        """
        计算所有组成指标
        
        Args:
            data: 输入数据
            **params: 计算参数
            
        Returns:
            Dict[str, IndicatorResult]: 组成指标结果
        """
        results = {}
        
        for indicator in self.component_indicators:
            try:
                result = indicator.calculate_with_validation(data, **params)
                results[indicator.name] = result
            except Exception as e:
                logger.error(f"组成指标 {indicator.name} 计算失败: {e}")
                results[indicator.name] = IndicatorResult(
                    name=indicator.name,
                    data=pd.DataFrame(),
                    metadata={},
                    calculation_time=0.0,
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def get_required_columns_unified_calculator(self) -> List[str]:
        """获取所有组成指标需要的列"""
        all_columns = []
        for indicator in self.component_indicators:
            all_columns.extend(indicator.get_required_columns())
        
        return list(set(all_columns))  # 去重


class IndicatorCalculatorFactory(BaseIndicator):
    """指标计算器工厂"""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, calculator_class: type) -> None:
        """注册指标计算器"""
        cls._registry[name] = calculator_class
        logger.debug(f"注册指标计算器: {name}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> UnifiedIndicatorCalculator:
        """创建指标计算器实例"""
        if name not in cls._registry:
            raise ValueError(f"未知的指标: {name}")
        
        calculator_class = cls._registry[name]
        return calculator_class(**kwargs)
    
    @classmethod
    def get_available_indicators(cls) -> List[str]:
        """获取可用指标列表"""
        return list(cls._registry.keys())
    
    @classmethod
    def get_indicators_by_type(cls, indicator_type: IndicatorType) -> List[str]:
        """按类型获取指标"""
        result = []
        for name, calculator_class in cls._registry.items():
            try:
                # 创建临时实例检查类型
                temp_instance = calculator_class()
                if temp_instance.indicator_type == indicator_type:
                    result.append(name)
            except:
                continue
        
        return result


# 示例:简单移动平均线指标
class SimpleMovingAverageCalculator(BaseIndicator,TrendIndicatorBase):
    """简单移动平均线"""
    
    def __init__(self):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__("SMA")
    
    def calculate(self, data: pd.DataFrame, **params) -> pd.Series:
        period = params.get('period', 20)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        return data['close'].rolling(window=period).mean()
    
    def get_required_columns_unified_calculator(self) -> List[str]:
        return ['close']
    
    def get_default_params_unified_calculator(self) -> Dict[str, Any]:
        return {'period': 20}  # TODO: 将魔法数字提取到配置中
    
    def validate_params_unified_calculator(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        period = params.get('period', 20)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if not isinstance(period, int) or period <= 0:
            errors.append("period必须是正整数")
        
        return len(errors) == 0, errors


# 注册示例指标
IndicatorCalculatorFactory.register('SMA', SimpleMovingAverageCalculator)
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取交易信号
        
        Args:
            data: 包含指标计算结果的数据
            
        Returns:
            Dict[str, Any]: 交易信号信息
        """
        if data.empty:
            return {'signal': 'hold', 'strength': 0.0, 'timestamp': None}
        
        # TODO: 实现具体的信号生成逻辑
        latest_close = data['close'].iloc[-1] if 'close' in data.columns else 0
        
        return {
            'signal': 'hold',
            'strength': 0.0,
            'timestamp': data.index[-1] if not data.empty else None,
            'price': latest_close,
            'indicator': self.name
        }
