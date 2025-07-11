"""
策略执行模板模块

提供标准化的策略执行模板和流程，包括：
1. 策略执行生命周期管理
2. 标准化的策略执行流程
3. 统一的结果处理和验证
4. 策略组合和链式执行支持
5. 执行监控和性能统计
"""

import abc
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from utils.logger import getLogger
from utils.common_utils import DataProcessor, ValidationUtils, CacheUtils, merge_dicts
from utils.decorators import exception_handler, performance_monitor
from strategy.unified_base_strategy import UnifiedBaseStrategy

logger = getLogger(__name__)


class ExecutionPhase(Enum):
    """执行阶段"""
    INITIALIZATION = "initialization"
    DATA_PREPARATION = "data_preparation"
    STRATEGY_EXECUTION = "strategy_execution"
    RESULT_PROCESSING = "result_processing"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """执行上下文"""
    strategy: UnifiedBaseStrategy
    universe: List[str]
    start_date: str
    end_date: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    execution_id: str
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy.name,
            'universe_size': len(self.universe),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'execution_id': self.execution_id,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ExecutionResult:
    """执行结果"""
    execution_id: str
    strategy_name: str
    status: ExecutionStatus
    selected_stocks: pd.DataFrame
    execution_time: float
    phase_times: Dict[str, float]
    statistics: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    completed_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'strategy_name': self.strategy_name,
            'status': self.status.value,
            'selected_count': len(self.selected_stocks),
            'execution_time': self.execution_time,
            'phase_times': self.phase_times,
            'statistics': self.statistics,
            'warnings': self.warnings,
            'errors': self.errors,
            'completed_at': self.completed_at.isoformat()
        }


class StrategyExecutionTemplate(abc.ABC):
    """
    策略执行模板基类
    
    定义标准化的策略执行流程和生命周期管理
    """
    
    def __init__(self, name: str):
        self.name = name
        self.hooks = {}  # 执行钩子
        self.validators = []  # 验证器
        self.interceptors = []  # 拦截器
        
        # 执行统计
        self.execution_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
        
        logger.info(f"初始化策略执行模板: {name}")
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """
        执行策略的主要入口点
        
        Args:
            context: 执行上下文
            
        Returns:
            ExecutionResult: 执行结果
        """
        start_time = datetime.now()
        phase_times = {}
        warnings = []
        errors = []
        status = ExecutionStatus.PENDING
        
        try:
            logger.info(f"开始执行策略: {context.strategy.name}, ID: {context.execution_id}")
            
            status = ExecutionStatus.RUNNING
            
            # 1. 初始化阶段
            phase_start = datetime.now()
            self._run_phase_with_hooks(ExecutionPhase.INITIALIZATION, context)
            self._execute_initialization(context)
            phase_times[ExecutionPhase.INITIALIZATION.value] = (datetime.now() - phase_start).total_seconds()
            
            # 2. 数据准备阶段
            phase_start = datetime.now()
            self._run_phase_with_hooks(ExecutionPhase.DATA_PREPARATION, context)
            prepared_data = self._execute_data_preparation(context)
            phase_times[ExecutionPhase.DATA_PREPARATION.value] = (datetime.now() - phase_start).total_seconds()
            
            # 3. 策略执行阶段
            phase_start = datetime.now()
            self._run_phase_with_hooks(ExecutionPhase.STRATEGY_EXECUTION, context)
            raw_result = self._execute_strategy(context, prepared_data)
            phase_times[ExecutionPhase.STRATEGY_EXECUTION.value] = (datetime.now() - phase_start).total_seconds()
            
            # 4. 结果处理阶段
            phase_start = datetime.now()
            self._run_phase_with_hooks(ExecutionPhase.RESULT_PROCESSING, context)
            processed_result = self._execute_result_processing(context, raw_result)
            phase_times[ExecutionPhase.RESULT_PROCESSING.value] = (datetime.now() - phase_start).total_seconds()
            
            # 5. 验证阶段
            phase_start = datetime.now()
            self._run_phase_with_hooks(ExecutionPhase.VALIDATION, context)
            validation_warnings = self._execute_validation(context, processed_result)
            warnings.extend(validation_warnings)
            phase_times[ExecutionPhase.VALIDATION.value] = (datetime.now() - phase_start).total_seconds()
            
            # 6. 完成阶段
            phase_start = datetime.now()
            self._run_phase_with_hooks(ExecutionPhase.FINALIZATION, context)
            statistics = self._execute_finalization(context, processed_result)
            phase_times[ExecutionPhase.FINALIZATION.value] = (datetime.now() - phase_start).total_seconds()
            
            # 确定最终状态
            status = ExecutionStatus.WARNING if warnings else ExecutionStatus.SUCCESS
            self.success_count += 1
            
        except Exception as e:
            error_msg = f"策略执行失败: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            status = ExecutionStatus.ERROR
            processed_result = pd.DataFrame()
            statistics = {}
            self.error_count += 1
        
        finally:
            # 更新执行统计
            self.execution_count += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            self.total_execution_time += execution_time
            
            # 创建执行结果
            result = ExecutionResult(
                execution_id=context.execution_id,
                strategy_name=context.strategy.name,
                status=status,
                selected_stocks=processed_result,
                execution_time=execution_time,
                phase_times=phase_times,
                statistics=statistics,
                warnings=warnings,
                errors=errors,
                completed_at=datetime.now()
            )
            
            logger.info(f"策略执行完成: {context.strategy.name}, 状态: {status.value}, 耗时: {execution_time:.2f}s")
            
            return result
    
    @abc.abstractmethod
    def _execute_initialization(self, context: ExecutionContext) -> None:
        """
        执行初始化逻辑
        
        Args:
            context: 执行上下文
        """
        pass
    
    @abc.abstractmethod
    def _execute_data_preparation(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        执行数据准备逻辑
        
        Args:
            context: 执行上下文
            
        Returns:
            Dict[str, Any]: 准备好的数据
        """
        pass
    
    @abc.abstractmethod
    def _execute_strategy(self, context: ExecutionContext, prepared_data: Dict[str, Any]) -> pd.DataFrame:
        """
        执行策略逻辑
        
        Args:
            context: 执行上下文
            prepared_data: 准备好的数据
            
        Returns:
            pd.DataFrame: 策略执行结果
        """
        pass
    
    @abc.abstractmethod
    def _execute_result_processing(self, context: ExecutionContext, raw_result: pd.DataFrame) -> pd.DataFrame:
        """
        执行结果处理逻辑
        
        Args:
            context: 执行上下文
            raw_result: 原始结果
            
        Returns:
            pd.DataFrame: 处理后的结果
        """
        pass
    
    @abc.abstractmethod
    def _execute_validation(self, context: ExecutionContext, processed_result: pd.DataFrame) -> List[str]:
        """
        执行验证逻辑
        
        Args:
            context: 执行上下文
            processed_result: 处理后的结果
            
        Returns:
            List[str]: 警告信息列表
        """
        pass
    
    @abc.abstractmethod
    def _execute_finalization(self, context: ExecutionContext, processed_result: pd.DataFrame) -> Dict[str, Any]:
        """
        执行完成逻辑
        
        Args:
            context: 执行上下文
            processed_result: 处理后的结果
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        pass
    
    def add_hook(self, phase: ExecutionPhase, hook: Callable[[ExecutionContext], None]) -> None:
        """添加执行钩子"""
        if phase not in self.hooks:
            self.hooks[phase] = []
        self.hooks[phase].append(hook)
    
    def add_validator(self, validator: Callable[[ExecutionContext, pd.DataFrame], List[str]]) -> None:
        """添加验证器"""
        self.validators.append(validator)
    
    def add_interceptor(self, interceptor: Callable[[ExecutionContext], bool]) -> None:
        """添加拦截器"""
        self.interceptors.append(interceptor)
    
    def _run_phase_with_hooks(self, phase: ExecutionPhase, context: ExecutionContext) -> None:
        """运行阶段钩子"""
        if phase in self.hooks:
            for hook in self.hooks[phase]:
                try:
                    hook(context)
                except Exception as e:
                    logger.warning(f"钩子执行失败: {phase.value} - {e}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        avg_execution_time = self.total_execution_time / self.execution_count if self.execution_count > 0 else 0
        success_rate = self.success_count / self.execution_count if self.execution_count > 0 else 0
        
        return {
            'template_name': self.name,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_execution_time
        }


class StandardStrategyExecutionTemplate(StrategyExecutionTemplate):
    """
    标准策略执行模板
    
    提供通用的策略执行流程实现
    """
    
    def __init__(self):
        super().__init__("StandardTemplate")
    
    def _execute_initialization(self, context: ExecutionContext) -> None:
        """初始化执行环境"""
        logger.debug(f"初始化策略执行环境: {context.strategy.name}")
        
        # 验证参数
        is_valid, errors = context.strategy.validate_parameters()
        if not is_valid:
            raise ValueError(f"策略参数验证失败: {errors}")
        
        # 设置策略参数
        context.strategy.set_parameters(**context.parameters)
    
    def _execute_data_preparation(self, context: ExecutionContext) -> Dict[str, Any]:
        """准备执行数据"""
        logger.debug(f"准备数据: 股票池大小 {len(context.universe)}")
        
        # 验证股票代码
        valid_codes = [code for code in context.universe if ValidationUtils.validate_stock_code(code)]
        invalid_codes = set(context.universe) - set(valid_codes)
        
        if invalid_codes:
            logger.warning(f"发现无效股票代码: {invalid_codes}")
        
        # 验证日期
        if not ValidationUtils.validate_date_format(context.start_date):
            raise ValueError(f"无效的开始日期: {context.start_date}")
        
        if not ValidationUtils.validate_date_format(context.end_date):
            raise ValueError(f"无效的结束日期: {context.end_date}")
        
        return {
            'valid_universe': valid_codes,
            'invalid_codes': list(invalid_codes),
            'data_range': {
                'start': context.start_date,
                'end': context.end_date
            }
        }
    
    def _execute_strategy(self, context: ExecutionContext, prepared_data: Dict[str, Any]) -> pd.DataFrame:
        """执行策略"""
        logger.debug(f"执行策略: {context.strategy.name}")
        
        valid_universe = prepared_data['valid_universe']
        
        # 执行策略
        result = context.strategy.execute(
            universe=valid_universe,
            start_date=context.start_date,
            end_date=context.end_date,
            **context.parameters
        )
        
        return result
    
    def _execute_result_processing(self, context: ExecutionContext, raw_result: pd.DataFrame) -> pd.DataFrame:
        """处理策略结果"""
        logger.debug(f"处理策略结果: {len(raw_result)} 条记录")
        
        if raw_result.empty:
            return raw_result
        
        # 标准化结果格式
        result = raw_result.copy()
        
        # 确保必要的列存在
        if 'code' not in result.columns:
            raise ValueError("策略结果必须包含code列")
        
        # 添加标准字段
        if 'timestamp' not in result.columns:
            result['timestamp'] = datetime.now()
        
        if 'strategy_name' not in result.columns:
            result['strategy_name'] = context.strategy.name
        
        if 'execution_id' not in result.columns:
            result['execution_id'] = context.execution_id
        
        # 清理和验证数据
        result = DataProcessor.clean_dataframe(result, drop_na=True, deduplicate=True)
        
        # 排序
        if 'score' in result.columns:
            result = result.sort_values('score', ascending=False)
        
        return result
    
    def _execute_validation(self, context: ExecutionContext, processed_result: pd.DataFrame) -> List[str]:
        """验证执行结果"""
        warnings = []
        
        # 基本验证
        if processed_result.empty:
            warnings.append("策略未选出任何股票")
        
        # 股票池覆盖率验证
        if not processed_result.empty:
            coverage_rate = len(processed_result) / len(context.universe)
            if coverage_rate > 0.5:
                warnings.append(f"选股覆盖率较高: {coverage_rate:.2%}")
            elif coverage_rate < 0.01:
                warnings.append(f"选股覆盖率较低: {coverage_rate:.2%}")
        
        # 分数验证
        if 'score' in processed_result.columns:
            scores = processed_result['score']
            if scores.isna().any():
                warnings.append("发现空分数值")
            
            if (scores < 0).any():
                warnings.append("发现负分数值")
        
        # 运行自定义验证器
        for validator in self.validators:
            try:
                validator_warnings = validator(context, processed_result)
                warnings.extend(validator_warnings)
            except Exception as e:
                warnings.append(f"验证器执行失败: {e}")
        
        return warnings
    
    def _execute_finalization(self, context: ExecutionContext, processed_result: pd.DataFrame) -> Dict[str, Any]:
        """完成策略执行"""
        logger.debug(f"完成策略执行: {context.strategy.name}")
        
        # 计算统计信息
        stats = {
            'total_universe_size': len(context.universe),
            'selected_count': len(processed_result),
            'selection_rate': len(processed_result) / len(context.universe) if context.universe else 0
        }
        
        # 分数统计
        if not processed_result.empty and 'score' in processed_result.columns:
            scores = processed_result['score']
            stats.update({
                'score_statistics': {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'median': float(scores.median())
                }
            })
        
        # 更新策略内部状态
        if hasattr(context.strategy, '_last_execution_stats'):
            context.strategy._last_execution_stats = stats
        
        return stats


class BatchStrategyExecutionTemplate(StrategyExecutionTemplate):
    """
    批量策略执行模板
    
    支持批量执行多个策略或多个股票池
    """
    
    def __init__(self, batch_size: int = 100):
        super().__init__("BatchTemplate")
        self.batch_size = batch_size
    
    def execute_batch(self, contexts: List[ExecutionContext]) -> List[ExecutionResult]:
        """
        批量执行策略
        
        Args:
            contexts: 执行上下文列表
            
        Returns:
            List[ExecutionResult]: 执行结果列表
        """
        results = []
        
        for i in range(0, len(contexts), self.batch_size):
            batch_contexts = contexts[i:i + self.batch_size]
            
            logger.info(f"执行批次 {i//self.batch_size + 1}/{len(contexts)//self.batch_size + 1}")
            
            batch_results = []
            for context in batch_contexts:
                try:
                    result = self.execute(context)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"批量执行失败: {context.execution_id} - {e}")
                    batch_results.append(ExecutionResult(
                        execution_id=context.execution_id,
                        strategy_name=context.strategy.name,
                        status=ExecutionStatus.ERROR,
                        selected_stocks=pd.DataFrame(),
                        execution_time=0.0,
                        phase_times={},
                        statistics={},
                        warnings=[],
                        errors=[str(e)],
                        completed_at=datetime.now()
                    ))
            
            results.extend(batch_results)
        
        return results
    
    def _execute_initialization(self, context: ExecutionContext) -> None:
        """批量模式初始化"""
        # 使用标准模板的初始化逻辑
        standard_template = StandardStrategyExecutionTemplate()
        standard_template._execute_initialization(context)
    
    def _execute_data_preparation(self, context: ExecutionContext) -> Dict[str, Any]:
        """批量模式数据准备"""
        standard_template = StandardStrategyExecutionTemplate()
        return standard_template._execute_data_preparation(context)
    
    def _execute_strategy(self, context: ExecutionContext, prepared_data: Dict[str, Any]) -> pd.DataFrame:
        """批量模式策略执行"""
        standard_template = StandardStrategyExecutionTemplate()
        return standard_template._execute_strategy(context, prepared_data)
    
    def _execute_result_processing(self, context: ExecutionContext, raw_result: pd.DataFrame) -> pd.DataFrame:
        """批量模式结果处理"""
        standard_template = StandardStrategyExecutionTemplate()
        return standard_template._execute_result_processing(context, raw_result)
    
    def _execute_validation(self, context: ExecutionContext, processed_result: pd.DataFrame) -> List[str]:
        """批量模式验证"""
        standard_template = StandardStrategyExecutionTemplate()
        return standard_template._execute_validation(context, processed_result)
    
    def _execute_finalization(self, context: ExecutionContext, processed_result: pd.DataFrame) -> Dict[str, Any]:
        """批量模式完成"""
        standard_template = StandardStrategyExecutionTemplate()
        return standard_template._execute_finalization(context, processed_result)


# 工厂方法
def create_execution_context(strategy: UnifiedBaseStrategy,
                           universe: List[str],
                           start_date: str,
                           end_date: str,
                           parameters: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> ExecutionContext:
    """创建执行上下文"""
    import uuid
    
    return ExecutionContext(
        strategy=strategy,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters or {},
        metadata=metadata or {},
        execution_id=str(uuid.uuid4()),
        created_at=datetime.now()
    )


def create_execution_template(template_type: str = "standard", **kwargs) -> StrategyExecutionTemplate:
    """创建执行模板"""
    if template_type == "standard":
        return StandardStrategyExecutionTemplate()
    elif template_type == "batch":
        batch_size = kwargs.get('batch_size', 100)
        return BatchStrategyExecutionTemplate(batch_size)
    else:
        raise ValueError(f"未知的模板类型: {template_type}")