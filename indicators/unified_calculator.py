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
class IndicatorType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum" 
    VOLUME = "volume"
    VOLATILITY = "volatility"
    COMPOSITE = "composite"

logger = get_logger(__name__)


class CalculationMode(Enum):
    """计算模式"""
    SINGLE = "single"      # 单次计算
    BATCH = "batch"        # 批量计算
    STREAMING = "streaming"  # 流式计算


@dataclass
class IndicatorResult:
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


class UnifiedIndicatorCalculator(abc.ABC):
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


class TrendIndicatorBase(BaseIndicator, UnifiedIndicatorCalculator):
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


class MomentumIndicatorBase(BaseIndicator, UnifiedIndicatorCalculator):
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


class VolumeIndicatorBase(BaseIndicator, UnifiedIndicatorCalculator):
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


class VolatilityIndicatorBase(BaseIndicator, UnifiedIndicatorCalculator):
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


class CompositeIndicatorBase(BaseIndicator, UnifiedIndicatorCalculator):
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


class IndicatorCalculatorFactory:
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
class SimpleMovingAverageCalculator(TrendIndicatorBase):
    """简单移动平均线"""
    
    def __init__(self, period: int = 20):
        # 先保存period，避免被super().__init__覆盖
        self.period = period
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        super().__init__("SMA")
        # 确保period不被覆盖
        self.period = period
    
    def calculate(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        period = params.get('period', self.period)  # 优先使用实例的period
        sma_values = data['close'].rolling(window=period).mean()
        
        # 返回标准DataFrame格式
        result_df = data.copy()
        result_df['sma'] = sma_values
        result_df['sma_signal'] = 0
        
        # 添加信号标记
        if len(sma_values) > 1:
            price_above_sma = data['close'] > sma_values
            price_above_sma_prev = data['close'].shift(1) > sma_values.shift(1)
            
            # 上穿信号
            result_df.loc[price_above_sma & ~price_above_sma_prev, 'sma_signal'] = 1
            # 下穿信号  
            result_df.loc[~price_above_sma & price_above_sma_prev, 'sma_signal'] = -1
        
        # 存储结果以供get_signal使用
        self._result = result_df
        return result_df
    
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

    def has_result(self) -> bool:
        """检查指标是否有计算结果"""
        return (hasattr(self, '_result') and 
                self._result is not None and 
                not self._result.empty and
                'sma' in self._result.columns)

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于SMA指标数值生成最新的交易信号
        
        SMA交易信号逻辑：
        - 价格上穿SMA：买入信号
        - 价格下穿SMA：卖出信号  
        - SMA趋势向上：支持买入
        - SMA趋势向下：支持卖出
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 1. 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 2. 获取period并计算或使用存储的结果
            period = getattr(self, 'period', 20)
            if self.has_result():
                result_df = self._result
                sma_values = result_df['sma']
            else:
                sma_values = data['close'].rolling(window=period).mean()
            
            if len(sma_values) < 2 or sma_values.isna().iloc[-1]:
                return self._get_default_signal("SMA数据不足")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            prev_close = data['close'].iloc[-2] if len(data) > 1 else latest_close
            latest_sma = sma_values.iloc[-1]
            prev_sma = sma_values.iloc[-2] if len(sma_values) > 1 else latest_sma
            
            # 4. SMA信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # 检查价格与SMA的关系
            price_above_sma_now = latest_close > latest_sma
            price_above_sma_prev = prev_close > prev_sma
            sma_rising = latest_sma > prev_sma
            sma_falling = latest_sma < prev_sma
            
            # 价格上穿SMA - 买入信号
            if price_above_sma_now and not price_above_sma_prev:
                signal_type = "buy"
                strength = 0.75
                confidence = 0.8
                reason = "价格上穿简单移动平均线，买入信号"
                if sma_rising:
                    strength = min(0.9, strength + 0.15)
                    confidence = min(0.9, confidence + 0.1)
                    reason = "价格上穿上升趋势SMA，强烈买入信号"
                    
            # 价格下穿SMA - 卖出信号
            elif not price_above_sma_now and price_above_sma_prev:
                signal_type = "sell"
                strength = 0.75
                confidence = 0.8
                reason = "价格下穿简单移动平均线，卖出信号"
                if sma_falling:
                    strength = min(0.9, strength + 0.15)
                    confidence = min(0.9, confidence + 0.1)
                    reason = "价格下穿下降趋势SMA，强烈卖出信号"
                    
            # 价格持续在SMA上方且SMA上升 - 持续买入
            elif price_above_sma_now and sma_rising:
                signal_type = "buy"
                strength = 0.6
                confidence = 0.7
                reason = "价格持续在上升SMA上方，持续买入信号"
                
            # 价格持续在SMA下方且SMA下降 - 持续卖出
            elif not price_above_sma_now and sma_falling:
                signal_type = "sell"
                strength = 0.6
                confidence = 0.7
                reason = "价格持续在下降SMA下方，持续卖出信号"
            
            # 计算价格相对SMA的偏离度
            if latest_sma > 0:
                price_deviation = abs(latest_close - latest_sma) / latest_sma
                metadata['price_deviation'] = price_deviation
                metadata['sma_value'] = latest_sma
                metadata['sma_trend'] = 'rising' if sma_rising else 'falling' if sma_falling else 'flat'
                metadata['sma_period'] = period
                
                # 基于偏离度调整信号强度
                if price_deviation > 0.03:  # 偏离超过3%
                    if signal_type in ['buy', 'sell']:
                        strength = min(1.0, strength + price_deviation * 1.5)
            
            # 5. 标准化输出
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'latest_close': latest_close,
                    **metadata
                }
            }

        except Exception as e:
            logger.warning(f"SMA信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """标准数据验证"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        if 'close' not in data.columns:
            return False
        
        # 检查数据量
        period = getattr(self, 'period', 20)
        # 确保period是整数
        if not isinstance(period, int):
            period = 20
        if len(data) < period:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "默认持有") -> Dict[str, Any]:
        """默认信号格式"""
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {}
        }


# 注册示例指标
IndicatorCalculatorFactory.register('SMA', SimpleMovingAverageCalculator)
