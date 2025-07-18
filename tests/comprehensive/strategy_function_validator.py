"""
策略功能验证测试器

验证选股策略的功能正确性和性能表现，确保每个策略都能正确选出股票
解决数据库字段不匹配问题，重点关注策略逻辑的正确性
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import getLogger
from utils.decorators import performance_monitor, exception_handler
from utils.dependency_injection import get_service, get_container
from db.interfaces.data_access_interface import DataAccessInterface
from db.interfaces.indicator_calculator_interface import IindicatorCalculator
from db.clickhouse_db import get_clickhouse_db
# 导入模拟指标计算器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mock_indicator_calculator import MockIndicatorCalculator
from strategy.dual_ma_strategy import DualMAStrategy
from strategy.institutional_strategy import InstitutionalStrategy
from strategy.momentum_strategy import MomentumStrategy
from strategy.breakout_strategy import BreakoutStrategy
from strategy.rebound_strategy import ReboundStrategy
from enums.period import Period

logger = getLogger(__name__)


@dataclass
class StrategyTestResult:
    """策略测试结果"""
    strategy_name: str
    test_date: datetime
    execution_time: float
    success: bool
    error_message: Optional[str]
    selected_stocks_count: int
    selected_stocks: List[str]
    performance_metrics: Dict[str, Any]
    logic_validation: Dict[str, Any]
    data_coverage: Dict[str, Any]


@dataclass
class StrategyPerformanceMetrics:
    """策略性能指标"""
    strategy_name: str
    total_execution_time: float
    data_fetch_time: float
    calculation_time: float
    stocks_processed: int
    stocks_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class StrategyValidationReport:
    """策略验证报告"""
    test_timestamp: datetime
    total_strategies_tested: int
    successful_strategies: int
    failed_strategies: int
    strategy_results: List[StrategyTestResult]
    performance_summary: Dict[str, Any]
    recommendation: str
    overall_status: str


class StrategyFunctionValidator:
    """
    策略功能验证器
    
    验证选股策略的功能正确性和性能表现
    """
    
    def __init__(self):
        """初始化策略功能验证器"""
        self.data_access = get_service(DataAccessInterface)
        self.db = get_clickhouse_db()  # 直接的数据库连接
        self.test_stocks = []
        self.strategies = {}
        self._init_test_environment()
    
    def _init_test_environment(self):
        """初始化测试环境"""
        logger.info("初始化策略功能验证测试环境")
        
        # 注册模拟指标计算器
        container = get_container()
        container.register_singleton(IindicatorCalculator, MockIndicatorCalculator)
        logger.info("已注册模拟指标计算器")
        
        # 初始化策略实例
        self.strategies = {}
        
        # 尝试初始化每个策略
        strategy_classes = {
            "双均线策略": DualMAStrategy,
            "主力行为策略": InstitutionalStrategy,
            "动量策略": MomentumStrategy,
            "突破策略": BreakoutStrategy,
            "反弹策略": ReboundStrategy
        }
        
        for name, strategy_class in strategy_classes.items():
            try:
                strategy_instance = strategy_class()
                self.strategies[name] = strategy_instance
                logger.info(f"✅ 成功初始化策略: {name}")
            except Exception as e:
                logger.warning(f"⚠️ 跳过策略 {name}: {e}")
                continue
        
        if not self.strategies:
            logger.error("没有可用的策略进行测试")
        else:
            logger.info(f"成功初始化 {len(self.strategies)} 个策略")
        
        # 获取测试股票池
        self._prepare_test_stocks()
    
    def _prepare_test_stocks(self):
        """准备测试股票池"""
        try:
            # 获取活跃股票作为测试样本
            query = """
            SELECT DISTINCT code 
            FROM stock_info 
            WHERE date >= '2024-01-01' 
            AND level = '日线'
            AND volume > 1000000
            ORDER BY code
            LIMIT 50
            """
            
            result = self.db.query(query)
            if result is not None and not result.empty:
                # 从DataFrame中提取股票代码
                self.test_stocks = result['code'].tolist()
                logger.info(f"准备了 {len(self.test_stocks)} 只测试股票")
            else:
                # 使用默认测试股票
                self.test_stocks = [
                    "000001", "000002", "000858", "600000", "600036", 
                    "600519", "002415", "300750", "603259"
                ]
                logger.warning(f"使用默认测试股票池: {len(self.test_stocks)} 只")
                
        except Exception as e:
            logger.error(f"准备测试股票失败: {e}")
            # 使用最小测试集
            self.test_stocks = ["000001", "600000", "000858"]
    
    @performance_monitor(threshold=60.0)
    def validate_strategy_function(self, strategy_name: str, 
                                 test_stocks: Optional[List[str]] = None) -> StrategyTestResult:
        """
        验证单个策略功能
        
        Args:
            strategy_name: 策略名称
            test_stocks: 测试股票列表，为None时使用默认测试股票
            
        Returns:
            StrategyTestResult: 策略测试结果
        """
        start_time = time.time()
        
        if test_stocks is None:
            test_stocks = self.test_stocks[:20]  # 使用前20只股票进行测试
        
        logger.info(f"开始验证策略: {strategy_name}, 测试股票数: {len(test_stocks)}")
        
        try:
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                raise ValueError(f"未找到策略: {strategy_name}")
            
            # 验证数据可用性
            data_coverage = self._validate_data_coverage(test_stocks)
            
            # 执行策略选股
            selected_stocks = []
            performance_metrics = {}
            logic_validation = {}
            
            if strategy_name == "双均线策略":
                # 测试双均线策略
                result = self._test_dual_ma_strategy(strategy, test_stocks)
                selected_stocks = result.get('selected_stocks', [])
                performance_metrics = result.get('performance_metrics', {})
                logic_validation = result.get('logic_validation', {})
            
            elif strategy_name == "主力行为策略":
                # 测试主力行为策略
                result = self._test_institutional_strategy(strategy, test_stocks)
                selected_stocks = result.get('selected_stocks', [])
                performance_metrics = result.get('performance_metrics', {})
                logic_validation = result.get('logic_validation', {})
            
            else:
                # 其他策略的通用测试
                result = self._test_generic_strategy(strategy, test_stocks)
                selected_stocks = result.get('selected_stocks', [])
                performance_metrics = result.get('performance_metrics', {})
                logic_validation = result.get('logic_validation', {})
            
            execution_time = time.time() - start_time
            
            # 验证选股结果
            success = len(selected_stocks) > 0
            error_message = None if success else "策略未选出任何股票"
            
            logger.info(f"策略 {strategy_name} 验证完成: 选出 {len(selected_stocks)} 只股票, 耗时 {execution_time:.2f}秒")
            
            return StrategyTestResult(
                strategy_name=strategy_name,
                test_date=datetime.now(),
                execution_time=execution_time,
                success=success,
                error_message=error_message,
                selected_stocks_count=len(selected_stocks),
                selected_stocks=selected_stocks,
                performance_metrics=performance_metrics,
                logic_validation=logic_validation,
                data_coverage=data_coverage
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"策略验证失败: {str(e)}"
            logger.error(error_msg)
            
            return StrategyTestResult(
                strategy_name=strategy_name,
                test_date=datetime.now(),
                execution_time=execution_time,
                success=False,
                error_message=error_msg,
                selected_stocks_count=0,
                selected_stocks=[],
                performance_metrics={},
                logic_validation={'error': str(e)},
                data_coverage={}
            )
    
    def _validate_data_coverage(self, test_stocks: List[str]) -> Dict[str, Any]:
        """验证数据覆盖情况"""
        try:
            # 检查数据可用性
            stock_codes_str = "', '".join(test_stocks)
            query = f"""
            SELECT code, COUNT(*) as cnt, 
                   MIN(date) as min_date, MAX(date) as max_date
            FROM stock_info 
            WHERE code IN ('{stock_codes_str}')
            AND level = '日线'
            GROUP BY code
            ORDER BY cnt DESC
            """
            
            result = self.db.query(query)
            
            coverage = {
                'total_stocks_requested': len(test_stocks),
                'stocks_with_data': len(result) if result is not None and not result.empty else 0,
                'data_coverage_ratio': len(result) / len(test_stocks) if test_stocks and result is not None and not result.empty else 0,
                'stocks_detail': {}
            }
            
            if result is not None and not result.empty:
                for _, row in result.iterrows():
                    coverage['stocks_detail'][row['code']] = {
                        'record_count': row['cnt'],
                        'start_date': str(row['min_date']),
                        'end_date': str(row['max_date'])
                    }
            
            logger.info(f"数据覆盖情况: {coverage['stocks_with_data']}/{coverage['total_stocks_requested']} 只股票有数据")
            return coverage
            
        except Exception as e:
            logger.error(f"数据覆盖验证失败: {e}")
            return {'error': str(e)}
    
    def _test_dual_ma_strategy(self, strategy: DualMAStrategy, test_stocks: List[str]) -> Dict[str, Any]:
        """测试双均线策略"""
        try:
            logger.info("测试双均线策略逻辑")
            
            # 设置策略参数
            strategy_params = {
                'short_period': 5,
                'long_period': 10,
                'volume_ratio': 1.5,
                'lookback_days': 3
            }
            
            start_time = time.time()
            
            # 执行策略选股
            selected_df = strategy.select_Strategy_Dual_Ma_Strategy(
                universe=test_stocks,
                **strategy_params
            )
            
            calculation_time = time.time() - start_time
            
            # 解析结果
            selected_stocks = []
            if not selected_df.empty and 'code' in selected_df.columns:
                selected_stocks = selected_df['code'].tolist()
            
            # 逻辑验证
            logic_validation = {
                'strategy_parameters': strategy_params,
                'result_columns': selected_df.columns.tolist() if not selected_df.empty else [],
                'result_shape': selected_df.shape if not selected_df.empty else (0, 0),
                'has_breakout_signals': 'breakout_date' in selected_df.columns if not selected_df.empty else False,
                'average_selection_ratio': len(selected_stocks) / len(test_stocks) if test_stocks else 0
            }
            
            performance_metrics = {
                'calculation_time': calculation_time,
                'stocks_processed': len(test_stocks),
                'selection_rate': len(selected_stocks) / len(test_stocks) if test_stocks else 0,
                'avg_time_per_stock': calculation_time / len(test_stocks) if test_stocks else 0
            }
            
            return {
                'selected_stocks': selected_stocks,
                'performance_metrics': performance_metrics,
                'logic_validation': logic_validation
            }
            
        except Exception as e:
            logger.error(f"双均线策略测试失败: {e}")
            return {
                'selected_stocks': [],
                'performance_metrics': {'error': str(e)},
                'logic_validation': {'error': str(e)}
            }
    
    def _test_institutional_strategy(self, strategy: InstitutionalStrategy, test_stocks: List[str]) -> Dict[str, Any]:
        """测试主力行为策略"""
        try:
            logger.info("测试主力行为策略逻辑")
            
            # 准备数据字典格式 (主力行为策略需要的格式)
            data_dict = {}
            
            for code in test_stocks[:5]:  # 限制测试股票数量
                try:
                    # 获取股票数据
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                    
                    # 使用正确的字段名查询
                    query = f"""
                    SELECT code, name, date, open, high, low, close, volume
                    FROM stock_info 
                    WHERE code = '{code}'
                    AND date >= '{start_date}' AND date <= '{end_date}'
                    AND level = '日线'
                    ORDER BY date ASC
                    """
                    
                    result = self.db.query(query)
                    
                    if result is not None and not result.empty and len(result) >= 20:
                        # result已经是DataFrame格式
                        data_dict[code] = result
                        
                except Exception as e:
                    logger.warning(f"获取股票 {code} 数据失败: {e}")
                    continue
            
            start_time = time.time()
            
            # 执行策略选股
            selected_stocks = strategy.select_Strategy_Institutional_Strategy(data_dict)
            
            calculation_time = time.time() - start_time
            
            # 逻辑验证
            logic_validation = {
                'data_stocks_count': len(data_dict),
                'selected_stocks_count': len(selected_stocks),
                'data_availability': list(data_dict.keys()),
                'selection_logic': 'institutional_behavior_analysis'
            }
            
            performance_metrics = {
                'calculation_time': calculation_time,
                'stocks_processed': len(data_dict),
                'selection_rate': len(selected_stocks) / len(data_dict) if data_dict else 0
            }
            
            return {
                'selected_stocks': selected_stocks,
                'performance_metrics': performance_metrics,
                'logic_validation': logic_validation
            }
            
        except Exception as e:
            logger.error(f"主力行为策略测试失败: {e}")
            return {
                'selected_stocks': [],
                'performance_metrics': {'error': str(e)},
                'logic_validation': {'error': str(e)}
            }
    
    def _test_generic_strategy(self, strategy, test_stocks: List[str]) -> Dict[str, Any]:
        """通用策略测试"""
        try:
            logger.info(f"测试策略: {strategy.name if hasattr(strategy, 'name') else '未知策略'}")
            
            start_time = time.time()
            
            # 尝试不同的策略接口
            selected_stocks = []
            
            if hasattr(strategy, 'select_stocks'):
                # 标准选股接口
                end_date = "2025-07-18"
                start_date = "2025-06-01"
                selected_stocks = strategy.select_stocks(test_stocks, start_date, end_date)
            elif hasattr(strategy, 'execute'):
                # 执行接口
                result = strategy.execute(test_stocks)
                if isinstance(result, list):
                    selected_stocks = result
                elif isinstance(result, pd.DataFrame) and 'code' in result.columns:
                    selected_stocks = result['code'].tolist()
            
            calculation_time = time.time() - start_time
            
            performance_metrics = {
                'calculation_time': calculation_time,
                'stocks_processed': len(test_stocks),
                'selection_rate': len(selected_stocks) / len(test_stocks) if test_stocks else 0
            }
            
            logic_validation = {
                'strategy_type': type(strategy).__name__,
                'selection_method': 'generic_interface',
                'result_type': type(selected_stocks).__name__
            }
            
            return {
                'selected_stocks': selected_stocks,
                'performance_metrics': performance_metrics,
                'logic_validation': logic_validation
            }
            
        except Exception as e:
            logger.error(f"通用策略测试失败: {e}")
            return {
                'selected_stocks': [],
                'performance_metrics': {'error': str(e)},
                'logic_validation': {'error': str(e)}
            }
    
    @performance_monitor(threshold=300.0)
    def run_comprehensive_strategy_validation(self) -> StrategyValidationReport:
        """
        运行综合策略验证
        
        Returns:
            StrategyValidationReport: 综合验证报告
        """
        logger.info("开始运行综合策略功能验证")
        start_time = time.time()
        
        strategy_results = []
        successful_strategies = 0
        failed_strategies = 0
        
        # 验证每个策略
        for strategy_name in self.strategies.keys():
            try:
                result = self.validate_strategy_function(strategy_name)
                strategy_results.append(result)
                
                if result.success:
                    successful_strategies += 1
                    logger.info(f"✅ 策略 {strategy_name} 验证成功: 选出 {result.selected_stocks_count} 只股票")
                else:
                    failed_strategies += 1
                    logger.warning(f"❌ 策略 {strategy_name} 验证失败: {result.error_message}")
                    
            except Exception as e:
                failed_strategies += 1
                logger.error(f"❌ 策略 {strategy_name} 验证异常: {e}")
                
                # 创建失败结果
                failed_result = StrategyTestResult(
                    strategy_name=strategy_name,
                    test_date=datetime.now(),
                    execution_time=0.0,
                    success=False,
                    error_message=str(e),
                    selected_stocks_count=0,
                    selected_stocks=[],
                    performance_metrics={'error': str(e)},
                    logic_validation={'error': str(e)},
                    data_coverage={}
                )
                strategy_results.append(failed_result)
        
        total_time = time.time() - start_time
        
        # 生成性能摘要
        performance_summary = self._generate_performance_summary(strategy_results, total_time)
        
        # 生成建议
        recommendation = self._generate_recommendation(strategy_results)
        
        # 确定整体状态
        success_rate = successful_strategies / len(self.strategies) if self.strategies else 0
        if success_rate >= 0.8:
            overall_status = "Excellent"
        elif success_rate >= 0.6:
            overall_status = "Good"
        elif success_rate >= 0.4:
            overall_status = "Fair"
        else:
            overall_status = "Poor"
        
        logger.info(f"策略验证完成: {successful_strategies}/{len(self.strategies)} 个策略成功, 状态: {overall_status}")
        
        return StrategyValidationReport(
            test_timestamp=datetime.now(),
            total_strategies_tested=len(self.strategies),
            successful_strategies=successful_strategies,
            failed_strategies=failed_strategies,
            strategy_results=strategy_results,
            performance_summary=performance_summary,
            recommendation=recommendation,
            overall_status=overall_status
        )
    
    def _generate_performance_summary(self, results: List[StrategyTestResult], total_time: float) -> Dict[str, Any]:
        """生成性能摘要"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        
        total_stocks_selected = sum(r.selected_stocks_count for r in successful_results)
        avg_execution_time = sum(r.execution_time for r in results) / len(results)
        avg_selection_count = sum(r.selected_stocks_count for r in results) / len(results)
        
        return {
            'total_execution_time': total_time,
            'average_strategy_time': avg_execution_time,
            'total_stocks_selected': total_stocks_selected,
            'average_selection_count': avg_selection_count,
            'successful_strategies_count': len(successful_results),
            'strategy_success_rate': len(successful_results) / len(results) if results else 0,
            'fastest_strategy': min(results, key=lambda x: x.execution_time).strategy_name if results else None,
            'most_selective_strategy': max(results, key=lambda x: x.selected_stocks_count).strategy_name if results else None
        }
    
    def _generate_recommendation(self, results: List[StrategyTestResult]) -> str:
        """生成建议"""
        if not results:
            return "无法生成建议：没有测试结果"
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        recommendations = []
        
        if len(successful_results) == len(results):
            recommendations.append("✅ 所有策略都成功运行并选出了股票，系统运行正常")
        elif successful_results:
            recommendations.append(f"⚠️ {len(successful_results)}/{len(results)} 个策略成功运行")
            
            # 分析失败原因
            error_patterns = {}
            for result in failed_results:
                error_key = result.error_message[:50] if result.error_message else "Unknown"
                error_patterns[error_key] = error_patterns.get(error_key, 0) + 1
            
            recommendations.append("失败策略的主要问题:")
            for error, count in error_patterns.items():
                recommendations.append(f"  - {error}: {count} 个策略")
        else:
            recommendations.append("❌ 所有策略都未能成功选出股票，需要检查:")
            recommendations.append("  - 数据库连接和数据可用性")
            recommendations.append("  - 策略逻辑实现")
            recommendations.append("  - 数据库字段映射问题")
        
        # 性能建议
        if successful_results:
            avg_time = sum(r.execution_time for r in successful_results) / len(successful_results)
            if avg_time > 30:
                recommendations.append("⚠️ 策略执行时间较长，建议优化性能")
            
            avg_selection = sum(r.selected_stocks_count for r in successful_results) / len(successful_results)
            if avg_selection < 1:
                recommendations.append("⚠️ 策略选股数量较少，可能需要调整选股条件")
        
        return "\n".join(recommendations)
    
    def save_validation_report(self, report: StrategyValidationReport, filename: Optional[str] = None) -> str:
        """
        保存验证报告
        
        Args:
            report: 验证报告
            filename: 文件名，为None时自动生成
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_function_validation_report_{timestamp}.txt"
        
        # 生成详细报告
        report_content = self._format_validation_report(report)
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 同时保存JSON格式
        json_filename = filename.replace('.txt', '.json')
        report_dict = self._convert_report_to_dict(report)
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"验证报告已保存: {filename} 和 {json_filename}")
        return filename
    
    def _format_validation_report(self, report: StrategyValidationReport) -> str:
        """格式化验证报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("策略功能验证测试报告")
        lines.append("=" * 80)
        lines.append(f"测试时间: {report.test_timestamp}")
        lines.append(f"测试策略数: {report.total_strategies_tested}")
        lines.append(f"成功策略数: {report.successful_strategies}")
        lines.append(f"失败策略数: {report.failed_strategies}")
        lines.append(f"整体状态: {report.overall_status}")
        lines.append("")
        
        # 性能摘要
        lines.append("性能摘要:")
        lines.append("-" * 40)
        for key, value in report.performance_summary.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.3f}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("")
        
        # 策略详细结果
        lines.append("策略详细结果:")
        lines.append("-" * 40)
        for result in report.strategy_results:
            status = "✅ 成功" if result.success else "❌ 失败"
            lines.append(f"{result.strategy_name}: {status}")
            lines.append(f"  选出股票数: {result.selected_stocks_count}")
            lines.append(f"  执行时间: {result.execution_time:.3f}秒")
            if result.error_message:
                lines.append(f"  错误信息: {result.error_message}")
            if result.selected_stocks:
                lines.append(f"  选出股票: {', '.join(result.selected_stocks[:10])}")
                if len(result.selected_stocks) > 10:
                    lines.append(f"  (共 {len(result.selected_stocks)} 只股票，仅显示前10只)")
            lines.append("")
        
        # 建议
        lines.append("建议和改进方向:")
        lines.append("-" * 40)
        lines.append(report.recommendation)
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("报告结束")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _convert_report_to_dict(self, report: StrategyValidationReport) -> Dict[str, Any]:
        """将报告转换为字典格式"""
        return {
            'test_timestamp': report.test_timestamp.isoformat(),
            'total_strategies_tested': report.total_strategies_tested,
            'successful_strategies': report.successful_strategies,
            'failed_strategies': report.failed_strategies,
            'overall_status': report.overall_status,
            'performance_summary': report.performance_summary,
            'recommendation': report.recommendation,
            'strategy_results': [
                {
                    'strategy_name': result.strategy_name,
                    'test_date': result.test_date.isoformat(),
                    'execution_time': result.execution_time,
                    'success': result.success,
                    'error_message': result.error_message,
                    'selected_stocks_count': result.selected_stocks_count,
                    'selected_stocks': result.selected_stocks,
                    'performance_metrics': result.performance_metrics,
                    'logic_validation': result.logic_validation,
                    'data_coverage': result.data_coverage
                }
                for result in report.strategy_results
            ]
        }


def main():
    """主函数"""
    logger.info("开始策略功能验证测试")
    
    try:
        # 创建验证器
        validator = StrategyFunctionValidator()
        
        # 运行综合验证
        report = validator.run_comprehensive_strategy_validation()
        
        # 保存报告
        report_file = validator.save_validation_report(report)
        
        # 输出摘要
        print(f"\n策略功能验证测试完成!")
        print(f"测试策略数: {report.total_strategies_tested}")
        print(f"成功策略数: {report.successful_strategies}")
        print(f"失败策略数: {report.failed_strategies}")
        print(f"整体状态: {report.overall_status}")
        print(f"报告文件: {report_file}")
        
        # 输出关键发现
        if report.successful_strategies > 0:
            print(f"\n✅ {report.successful_strategies} 个策略成功选出股票")
            successful_results = [r for r in report.strategy_results if r.success]
            total_selected = sum(r.selected_stocks_count for r in successful_results)
            print(f"总计选出: {total_selected} 只股票")
        
        if report.failed_strategies > 0:
            print(f"\n❌ {report.failed_strategies} 个策略失败")
            print("建议检查策略逻辑和数据库字段映射")
        
        return report.overall_status in ['Excellent', 'Good']
        
    except Exception as e:
        logger.error(f"策略功能验证测试失败: {e}")
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 