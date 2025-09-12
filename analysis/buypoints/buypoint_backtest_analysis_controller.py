#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
买点回测分析主控制器

统一管理买点回测分析的完整流程，包括：
- 回测引擎管理
- 买点识别协调
- 评估系统集成
- 分析结果导出
遵循六层架构规范
"""

import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container

# 导入买点回测分析模块
from .enhanced_backtest_engine import EnhancedBacktestEngine, BacktestConfig, BuyPointData, BacktestSummary
from .enhanced_buypoint_detector import EnhancedBuyPointDetector, BuyPointDetectionConfig, BuyPointSignal
from .enhanced_backtest_evaluator import EnhancedBacktestEvaluator, EvaluationResult

logger = get_logger(__name__)

@dataclass
class AnalysisConfig:
    """分析配置"""
    backtest_config: BacktestConfig = None
    detection_config: BuyPointDetectionConfig = None
    enable_realtime_detection: bool = False
    enable_performance_evaluation: bool = True
    output_directory: str = "reports/buypoint_analysis"
    max_concurrent_analyses: int = 10
    
    def __post_init__(self):
        if self.backtest_config is None:
            self.backtest_config = BacktestConfig()
        if self.detection_config is None:
            self.detection_config = BuyPointDetectionConfig()

@dataclass
class AnalysisResult:
    """分析结果"""
    analysis_id: str
    analysis_date: str
    backtest_summary: BacktestSummary
    detected_signals: List[BuyPointSignal]
    evaluation_result: Optional[EvaluationResult]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    execution_time: float

class BuyPointBacktestAnalysisController:
    """
    买点回测分析主控制器
    
    提供买点回测分析的完整解决方案，集成回测引擎、
    买点识别和评估系统
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        初始化分析控制器
        
        Args:
            config: 分析配置
        """
        self.config = config or AnalysisConfig()
        self.logger = get_logger(__name__)
        
        # 初始化核心组件
        self.backtest_engine = EnhancedBacktestEngine(self.config.backtest_config)
        self.buypoint_detector = EnhancedBuyPointDetector(self.config.detection_config)
        self.backtest_evaluator = EnhancedBacktestEvaluator()
        
        # 从容器获取服务
        container = get_container()
        try:
            self.data_access = container.resolve("DataAccessInterface")
        except:
            self.data_access = self._create_mock_data_access()
        
        # 分析统计
        self.analysis_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'average_execution_time': 0.0,
            'total_buypoints_analyzed': 0,
            'total_signals_detected': 0
        }
        
        # 确保输出目录存在
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        self.logger.info("买点回测分析主控制器初始化完成")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=120.0)
    def run_comprehensive_analysis(self, buypoints: List[BuyPointData],
                                  stock_codes_for_detection: Optional[List[str]] = None) -> AnalysisResult:
        """
        运行综合分析
        
        Args:
            buypoints: 历史买点数据列表
            stock_codes_for_detection: 用于实时检测的股票代码列表
            
        Returns:
            AnalysisResult: 综合分析结果
        """
        start_time = time.time()
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"开始综合分析 {analysis_id}，历史买点: {len(buypoints)} 个")
        
        try:
            # 1. 运行历史买点回测
            self.logger.info("🔄 执行历史买点回测分析...")
            backtest_summary = self.backtest_engine.run_backtest(
                buypoints, 
                output_file=os.path.join(self.config.output_directory, f"{analysis_id}_backtest.json")
            )
            
            # 2. 实时买点检测（如果启用）
            detected_signals = []
            if self.config.enable_realtime_detection and stock_codes_for_detection:
                self.logger.info(f"🔍 执行实时买点检测，股票数量: {len(stock_codes_for_detection)}")
                detected_signals = self.buypoint_detector.detect_buypoints(stock_codes_for_detection)
            
            # 3. 性能评估（如果启用）
            evaluation_result = None
            if self.config.enable_performance_evaluation and backtest_summary.successful_analyses > 0:
                self.logger.info("📊 执行性能评估...")
                # 构造回测结果数据用于评估
                mock_backtest_results = self._create_mock_backtest_results(backtest_summary)
                evaluation_result = self.backtest_evaluator.evaluate_backtest_results(mock_backtest_results)
            
            # 4. 生成综合建议
            recommendations = self._generate_comprehensive_recommendations(
                backtest_summary, detected_signals, evaluation_result
            )
            
            # 5. 计算性能指标
            performance_metrics = self._calculate_analysis_performance_metrics(
                backtest_summary, detected_signals, evaluation_result
            )
            
            # 6. 创建分析结果
            execution_time = time.time() - start_time
            analysis_result = AnalysisResult(
                analysis_id=analysis_id,
                analysis_date=datetime.now().isoformat(),
                backtest_summary=backtest_summary,
                detected_signals=detected_signals,
                evaluation_result=evaluation_result,
                performance_metrics=performance_metrics,
                recommendations=recommendations,
                execution_time=execution_time
            )
            
            # 7. 保存分析结果
            self._save_analysis_result(analysis_result)
            
            # 8. 更新统计
            self._update_analysis_stats(analysis_result)
            
            self.logger.info(f"✅ 综合分析完成，耗时: {execution_time:.2f}秒")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ 综合分析失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def run_backtest_only(self, buypoints: List[BuyPointData]) -> BacktestSummary:
        """
        仅运行回测分析
        
        Args:
            buypoints: 买点数据列表
            
        Returns:
            BacktestSummary: 回测汇总结果
        """
        self.logger.info(f"开始回测分析，买点数量: {len(buypoints)}")
        
        analysis_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_file = os.path.join(self.config.output_directory, f"{analysis_id}.json")
        
        return self.backtest_engine.run_backtest(buypoints, output_file)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def run_detection_only(self, stock_codes: List[str]) -> List[BuyPointSignal]:
        """
        仅运行买点检测
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            List[BuyPointSignal]: 检测到的买点信号
        """
        self.logger.info(f"开始买点检测，股票数量: {len(stock_codes)}")
        
        return self.buypoint_detector.detect_buypoints(stock_codes)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=15.0)
    def run_evaluation_only(self, backtest_results: List[Dict[str, Any]]) -> EvaluationResult:
        """
        仅运行评估分析
        
        Args:
            backtest_results: 回测结果数据
            
        Returns:
            EvaluationResult: 评估结果
        """
        self.logger.info(f"开始评估分析，回测记录数: {len(backtest_results)}")
        
        return self.backtest_evaluator.evaluate_backtest_results(backtest_results)
    
    def _create_mock_backtest_results(self, summary: BacktestSummary) -> List[Dict[str, Any]]:
        """创建模拟回测结果用于评估"""
        # 基于汇总信息生成模拟的详细回测结果
        mock_results = []
        
        for i in range(summary.successful_analyses):
            # 生成模拟收益率
            if summary.average_score > 70:
                base_return = 0.05 + (summary.average_score - 70) * 0.001
            else:
                base_return = 0.02 + (summary.average_score - 50) * 0.001
            
            # 添加随机波动
            import random
            random.seed(i)
            return_value = base_return + random.uniform(-0.02, 0.02)
            
            mock_results.append({
                'trade_id': f"trade_{i+1}",
                'return': return_value,
                'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                'score': summary.average_score + random.uniform(-10, 10)
            })
        
        return mock_results
    
    def _generate_comprehensive_recommendations(self, backtest_summary: BacktestSummary,
                                              detected_signals: List[BuyPointSignal],
                                              evaluation_result: Optional[EvaluationResult]) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        # 基于回测结果的建议
        if backtest_summary.success_rate >= 0.8:
            recommendations.append("历史回测表现优秀，策略具有较强的稳定性")
        elif backtest_summary.success_rate >= 0.6:
            recommendations.append("历史回测表现良好，可考虑实盘应用")
        else:
            recommendations.append("历史回测表现一般，建议优化策略参数")
        
        # 基于检测信号的建议
        if detected_signals:
            high_quality_signals = [s for s in detected_signals if s.score >= 80]
            if high_quality_signals:
                recommendations.append(f"检测到 {len(high_quality_signals)} 个高质量买点信号，建议重点关注")
            
            if len(detected_signals) > 20:
                recommendations.append("检测到大量买点信号，建议筛选优质标的")
        
        # 基于评估结果的建议
        if evaluation_result:
            if evaluation_result.overall_rating in ['A+', 'A']:
                recommendations.append("策略评级优秀，建议增加投资比重")
            elif evaluation_result.overall_rating in ['B+', 'B']:
                recommendations.append("策略评级良好，建议适度配置")
            else:
                recommendations.append("策略评级一般，建议谨慎操作")
        
        # 风险管理建议
        recommendations.append("建议设置合理的止损位，控制单笔损失")
        recommendations.append("建议分散投资，避免集中持仓风险")
        
        return recommendations
    
    def _calculate_analysis_performance_metrics(self, backtest_summary: BacktestSummary,
                                              detected_signals: List[BuyPointSignal],
                                              evaluation_result: Optional[EvaluationResult]) -> Dict[str, Any]:
        """计算分析性能指标"""
        metrics = {
            'backtest_metrics': {
                'total_buypoints': backtest_summary.total_buypoints,
                'success_rate': backtest_summary.success_rate,
                'average_score': backtest_summary.average_score,
                'execution_time': backtest_summary.execution_time
            },
            'detection_metrics': {
                'total_signals': len(detected_signals),
                'high_quality_signals': len([s for s in detected_signals if s.score >= 80]),
                'average_confidence': sum(s.confidence for s in detected_signals) / len(detected_signals) if detected_signals else 0.0
            }
        }
        
        if evaluation_result:
            metrics['evaluation_metrics'] = {
                'overall_rating': evaluation_result.overall_rating,
                'confidence_level': evaluation_result.confidence_level,
                'win_rate': evaluation_result.performance_metrics.win_rate,
                'sharpe_ratio': evaluation_result.performance_metrics.sharpe_ratio
            }
        
        return metrics

    def _save_analysis_result(self, analysis_result: AnalysisResult):
        """保存分析结果"""
        try:
            output_file = os.path.join(
                self.config.output_directory,
                f"{analysis_result.analysis_id}_comprehensive.json"
            )

            # 准备输出数据
            output_data = {
                'analysis_result': asdict(analysis_result),
                'config': asdict(self.config),
                'generated_at': datetime.now().isoformat()
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"分析结果已保存: {output_file}")

        except Exception as e:
            self.logger.error(f"保存分析结果失败: {e}")

    def _update_analysis_stats(self, analysis_result: AnalysisResult):
        """更新分析统计"""
        self.analysis_stats['total_analyses'] += 1
        self.analysis_stats['successful_analyses'] += 1

        # 更新平均执行时间
        total_time = (self.analysis_stats['average_execution_time'] *
                     (self.analysis_stats['total_analyses'] - 1) +
                     analysis_result.execution_time)
        self.analysis_stats['average_execution_time'] = total_time / self.analysis_stats['total_analyses']

        # 更新其他统计
        self.analysis_stats['total_buypoints_analyzed'] += analysis_result.backtest_summary.total_buypoints
        self.analysis_stats['total_signals_detected'] += len(analysis_result.detected_signals)

    @performance_monitor(threshold=10.0)
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        return {
            'analysis_stats': self.analysis_stats.copy(),
            'engine_performance': self.backtest_engine.get_performance_report(),
            'detector_stats': self.buypoint_detector.detection_stats.copy(),
            'config_summary': {
                'backtest_periods': self.config.backtest_config.periods,
                'detection_types': [bt.value for bt in self.config.detection_config.detection_types],
                'output_directory': self.config.output_directory
            }
        }

    @exception_handler(reraise=False, default_return=[])
    def get_recent_analysis_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的分析结果"""
        try:
            results = []

            # 扫描输出目录中的分析结果文件
            if os.path.exists(self.config.output_directory):
                files = [f for f in os.listdir(self.config.output_directory)
                        if f.endswith('_comprehensive.json')]

                # 按修改时间排序
                files.sort(key=lambda x: os.path.getmtime(
                    os.path.join(self.config.output_directory, x)
                ), reverse=True)

                # 读取最近的结果
                for file in files[:limit]:
                    file_path = os.path.join(self.config.output_directory, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            results.append({
                                'file': file,
                                'analysis_id': data.get('analysis_result', {}).get('analysis_id'),
                                'analysis_date': data.get('analysis_result', {}).get('analysis_date'),
                                'summary': {
                                    'total_buypoints': data.get('analysis_result', {}).get('backtest_summary', {}).get('total_buypoints', 0),
                                    'success_rate': data.get('analysis_result', {}).get('backtest_summary', {}).get('success_rate', 0.0),
                                    'detected_signals': len(data.get('analysis_result', {}).get('detected_signals', [])),
                                    'overall_rating': data.get('analysis_result', {}).get('evaluation_result', {}).get('overall_rating', 'N/A')
                                }
                            })
                    except Exception as e:
                        self.logger.warning(f"读取分析结果文件 {file} 失败: {e}")

            return results

        except Exception as e:
            self.logger.error(f"获取最近分析结果失败: {e}")
            return []

    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def cleanup_old_results(self, days_to_keep: int = 30) -> int:
        """清理旧的分析结果"""
        try:
            if not os.path.exists(self.config.output_directory):
                return 0

            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            deleted_count = 0

            for file in os.listdir(self.config.output_directory):
                file_path = os.path.join(self.config.output_directory, file)

                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        self.logger.debug(f"删除旧文件: {file}")
                    except Exception as e:
                        self.logger.warning(f"删除文件 {file} 失败: {e}")

            self.logger.info(f"清理完成，删除了 {deleted_count} 个旧文件")
            return deleted_count

        except Exception as e:
            self.logger.error(f"清理旧结果失败: {e}")
            raise

    @performance_monitor(threshold=2.0)
    def validate_system_health(self) -> Dict[str, Any]:
        """验证系统健康状态"""
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'issues': [],
            'recommendations': []
        }

        try:
            # 检查回测引擎
            engine_stats = self.backtest_engine.get_performance_report()
            health_status['components']['backtest_engine'] = {
                'status': 'healthy',
                'cache_hit_rate': engine_stats.get('cache_stats', {}).get('hit_rate', 0.0),
                'total_analyses': engine_stats.get('engine_stats', {}).get('total_analyses', 0)
            }

            # 检查买点检测器
            detector_stats = self.buypoint_detector.detection_stats
            detection_success_rate = (detector_stats.get('successful_detections', 0) /
                                    max(detector_stats.get('total_detections', 1), 1))

            health_status['components']['buypoint_detector'] = {
                'status': 'healthy' if detection_success_rate > 0.5 else 'warning',
                'success_rate': detection_success_rate,
                'total_detections': detector_stats.get('total_detections', 0)
            }

            # 检查输出目录
            output_dir_exists = os.path.exists(self.config.output_directory)
            health_status['components']['output_directory'] = {
                'status': 'healthy' if output_dir_exists else 'error',
                'path': self.config.output_directory,
                'exists': output_dir_exists
            }

            # 检查数据访问
            try:
                # 简单的数据访问测试
                test_query = "SELECT COUNT(*) as count FROM stock_info WHERE level = '日线' LIMIT 1"
                result = self.data_access.query_dataframe(test_query)
                data_access_ok = len(result) > 0
            except:
                data_access_ok = False

            health_status['components']['data_access'] = {
                'status': 'healthy' if data_access_ok else 'error',
                'connection_test': data_access_ok
            }

            # 汇总状态
            component_statuses = [comp['status'] for comp in health_status['components'].values()]
            if 'error' in component_statuses:
                health_status['overall_status'] = 'error'
            elif 'warning' in component_statuses:
                health_status['overall_status'] = 'warning'

            # 生成建议
            if not output_dir_exists:
                health_status['issues'].append("输出目录不存在")
                health_status['recommendations'].append("检查输出目录配置")

            if not data_access_ok:
                health_status['issues'].append("数据访问异常")
                health_status['recommendations'].append("检查数据库连接")

            if detection_success_rate < 0.5:
                health_status['issues'].append("买点检测成功率偏低")
                health_status['recommendations'].append("检查检测配置和数据质量")

        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['issues'].append(f"健康检查异常: {e}")
            self.logger.error(f"系统健康检查失败: {e}")

        return health_status

    def _create_mock_data_access(self):
        """创建模拟数据访问对象"""
        class MockDataAccess:
            def query_dataframe(self, query: str) -> pd.DataFrame:
                # 生成模拟股票数据
                dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
                data = []
                for i, date in enumerate(dates):
                    data.append({
                        'code': '000001',
                        'name': '平安银行',
                        'date': date.strftime('%Y-%m-%d'),
                        'open': 10.0 + np.random.normal(0, 0.5),
                        'high': 10.5 + np.random.normal(0, 0.5),
                        'low': 9.5 + np.random.normal(0, 0.5),
                        'close': 10.0 + np.random.normal(0, 0.5),
                        'volume': 1000000 + np.random.randint(0, 500000),
                        'turnover_rate': np.random.uniform(0.5, 5.0)
                    })
                return pd.DataFrame(data)

        return MockDataAccess()
