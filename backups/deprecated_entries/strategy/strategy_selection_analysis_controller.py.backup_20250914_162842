"""
策略选股分析模块主控制器

整合策略配置、选股执行、策略评估等核心功能
提供统一的策略选股分析服务接口
遵循六层架构规范，实现高效的策略选股分析
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from strategy.enhanced_strategy_config_engine import (
    EnhancedStrategyConfigEngine, StrategyConfig
)
from strategy.enhanced_stock_selection_engine import EnhancedStockSelectionEngine
from strategy.enhanced_strategy_evaluation_system import EnhancedStrategyEvaluationSystem

logger = get_logger(__name__)


class StrategySelectionAnalysisController:
    """
    策略选股分析模块主控制器
    
    整合策略配置、选股执行、策略评估等核心功能
    """
    
    def __init__(self):
        """初始化控制器"""
        self.logger = logger
        
        # 初始化核心组件
        self.config_engine = EnhancedStrategyConfigEngine()
        self.selection_engine = EnhancedStockSelectionEngine()
        self.evaluation_system = EnhancedStrategyEvaluationSystem()
        
        # 注册到依赖注入容器
        container = get_container()
        container.register(type(self.config_engine), instance=self.config_engine)
        container.register(type(self.selection_engine), instance=self.selection_engine)
        container.register(type(self.evaluation_system), instance=self.evaluation_system)
        
        # 性能统计
        self.performance_stats = {
            'total_operations': 0,
            'strategy_creations': 0,
            'stock_selections': 0,
            'strategy_evaluations': 0,
            'total_execution_time': 0.0,
            'average_operation_time': 0.0
        }
        
        self.logger.info("策略选股分析控制器初始化完成")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def create_strategy_from_formula(self, formula: str, strategy_name: str,
                                   description: str = "", **kwargs) -> Dict[str, Any]:
        """
        从通达信风格公式创建策略
        
        Args:
            formula: 通达信风格公式
            strategy_name: 策略名称
            description: 策略描述
            **kwargs: 其他配置参数
            
        Returns:
            Dict[str, Any]: 创建结果
        """
        start_time = time.time()
        self.logger.info(f"开始从公式创建策略: {strategy_name}")
        
        try:
            # 解析公式
            conditions = self.config_engine.parse_formula(formula)
            
            if not conditions:
                return {
                    'success': False,
                    'error': '公式解析失败，未能提取有效条件',
                    'strategy_id': None
                }
            
            # 构建策略配置数据
            config_data = {
                'name': strategy_name,
                'description': description or f"基于公式创建的策略: {formula}",
                'version': kwargs.get('version', '1.0.0'),
                'rules': [{
                    'name': f"{strategy_name}_主规则",
                    'description': f"基于公式 {formula} 的主要选股规则",
                    'formula': formula,
                    'min_score': kwargs.get('min_score', 60.0),
                    'max_results': kwargs.get('max_results', 50)
                }],
                'global_settings': {
                    'min_score': kwargs.get('min_score', 60.0),
                    'max_results': kwargs.get('max_results', 50),
                    'risk_level': kwargs.get('risk_level', 'medium')
                }
            }
            
            # 创建策略配置
            strategy_config = self.config_engine.create_strategy_config(config_data)
            
            # 验证策略配置
            validation_result = self.config_engine.validate_strategy_config(strategy_config)
            
            # 更新统计
            self.performance_stats['strategy_creations'] += 1
            self.performance_stats['total_operations'] += 1
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            
            result = {
                'success': True,
                'strategy_id': strategy_config.strategy_id,
                'strategy_name': strategy_config.name,
                'conditions_parsed': len(conditions),
                'validation_result': validation_result,
                'execution_time': execution_time,
                'strategy_config': asdict(strategy_config)
            }
            
            self.logger.info(f"策略创建成功: {strategy_name}, ID: {strategy_config.strategy_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"创建策略时出错: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy_id': None
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=15.0)
    def execute_stock_selection(self, strategy_config: StrategyConfig,
                               stock_pool: Optional[List[str]] = None,
                               selection_date: Optional[str] = None) -> Dict[str, Any]:
        """
        执行股票选股
        
        Args:
            strategy_config: 策略配置
            stock_pool: 股票池
            selection_date: 选股日期
            
        Returns:
            Dict[str, Any]: 选股结果
        """
        start_time = time.time()
        self.logger.info(f"开始执行选股: {strategy_config.name}")
        
        try:
            # 执行选股
            selection_result = self.selection_engine.execute_selection(
                strategy_config=strategy_config,
                stock_pool=stock_pool,
                selection_date=selection_date
            )
            
            # 更新统计
            self.performance_stats['stock_selections'] += 1
            self.performance_stats['total_operations'] += 1
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            
            # 添加执行统计
            selection_result['controller_stats'] = {
                'execution_time': execution_time,
                'operation_id': f"selection_{int(time.time())}",
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"选股执行完成: {len(selection_result.get('selected_stocks', []))} 只股票被选中")
            return selection_result
            
        except Exception as e:
            self.logger.error(f"执行选股时出错: {e}")
            return {
                'success': False,
                'error': str(e),
                'selected_stocks': [],
                'metrics': None
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def evaluate_strategy_performance(self, strategy_config: StrategyConfig,
                                    evaluation_period: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        评估策略性能
        
        Args:
            strategy_config: 策略配置
            evaluation_period: 评估期间
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        start_time = time.time()
        self.logger.info(f"开始评估策略: {strategy_config.name}")
        
        try:
            # 执行策略评估
            evaluation_result = self.evaluation_system.evaluate_strategy(
                strategy_config=strategy_config,
                evaluation_period=evaluation_period
            )
            
            # 更新统计
            self.performance_stats['strategy_evaluations'] += 1
            self.performance_stats['total_operations'] += 1
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            
            result = {
                'success': True,
                'evaluation_result': asdict(evaluation_result),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"策略评估完成: 综合评分 {evaluation_result.overall_score:.2f}, 评级 {evaluation_result.grade}")
            return result
            
        except Exception as e:
            self.logger.error(f"评估策略时出错: {e}")
            return {
                'success': False,
                'error': str(e),
                'evaluation_result': None
            }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=20.0)
    def run_complete_analysis(self, formula: str, strategy_name: str,
                            stock_pool: Optional[List[str]] = None,
                            evaluation_period: Optional[Tuple[str, str]] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        运行完整的策略分析流程
        
        Args:
            formula: 通达信风格公式
            strategy_name: 策略名称
            stock_pool: 股票池
            evaluation_period: 评估期间
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 完整分析结果
        """
        start_time = time.time()
        self.logger.info(f"开始完整策略分析: {strategy_name}")
        
        try:
            # 1. 创建策略
            strategy_creation_result = self.create_strategy_from_formula(
                formula=formula,
                strategy_name=strategy_name,
                **kwargs
            )
            
            if not strategy_creation_result['success']:
                return {
                    'success': False,
                    'error': f"策略创建失败: {strategy_creation_result['error']}",
                    'stage': 'strategy_creation'
                }
            
            # 重建策略配置对象
            strategy_config_data = strategy_creation_result['strategy_config']
            strategy_config = self.config_engine.create_strategy_config(strategy_config_data)
            
            # 2. 执行选股
            selection_result = self.execute_stock_selection(
                strategy_config=strategy_config,
                stock_pool=stock_pool
            )
            
            # 3. 评估策略
            evaluation_result = self.evaluate_strategy_performance(
                strategy_config=strategy_config,
                evaluation_period=evaluation_period
            )
            
            # 4. 汇总结果
            total_execution_time = time.time() - start_time
            
            complete_result = {
                'success': True,
                'strategy_creation': strategy_creation_result,
                'stock_selection': selection_result,
                'strategy_evaluation': evaluation_result,
                'summary': {
                    'strategy_id': strategy_config.strategy_id,
                    'strategy_name': strategy_name,
                    'selected_stocks_count': len(selection_result.get('selected_stocks', [])),
                    'overall_score': evaluation_result.get('evaluation_result', {}).get('overall_score', 0.0),
                    'grade': evaluation_result.get('evaluation_result', {}).get('grade', 'N/A'),
                    'total_execution_time': total_execution_time,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"完整策略分析完成: {strategy_name}, 耗时 {total_execution_time:.2f}秒")
            return complete_result
            
        except Exception as e:
            self.logger.error(f"完整策略分析时出错: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'complete_analysis'
            }
    
    @exception_handler(reraise=False, default_return={})
    @performance_monitor(threshold=1.0)
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        # 更新平均操作时间
        if self.performance_stats['total_operations'] > 0:
            self.performance_stats['average_operation_time'] = (
                self.performance_stats['total_execution_time'] / 
                self.performance_stats['total_operations']
            )
        
        return {
            'system_status': 'running',
            'components': {
                'config_engine': 'active',
                'selection_engine': 'active',
                'evaluation_system': 'active'
            },
            'performance_stats': self.performance_stats.copy(),
            'supported_features': {
                'formula_parsing': True,
                'stock_selection': True,
                'strategy_evaluation': True,
                'complete_analysis': True
            },
            'system_info': {
                'max_workers': self.selection_engine.max_workers,
                'cache_enabled': self.selection_engine.cache_enabled,
                'indicators_available': getattr(self.selection_engine.indicator_registry, '__len__', lambda: 0)() if hasattr(self.selection_engine.indicator_registry, '__len__') else len(getattr(self.selection_engine.indicator_registry, 'registry', {}))
            }
        }
    
    @exception_handler(reraise=False, default_return="")
    @performance_monitor(threshold=2.0)
    def export_analysis_report(self, analysis_result: Dict[str, Any], 
                             format_type: str = 'json') -> str:
        """导出分析报告"""
        try:
            if format_type.lower() == 'json':
                return json.dumps(analysis_result, indent=2, ensure_ascii=False, default=str)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")
                
        except Exception as e:
            self.logger.error(f"导出分析报告时出错: {e}")
            return ""
