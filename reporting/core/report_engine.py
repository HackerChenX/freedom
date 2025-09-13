#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高性能回测报告生成引擎

核心功能：
- 多格式报告生成（HTML、PDF、Excel、JSON）
- 专业级可视化图表生成
- 报告模板管理系统
- 自动化分发机制
- 高性能优化（生成时间<60秒）

设计目标：
- 支持1000页+大型报告
- 图表渲染精度>300DPI
- 模板加载时间<5秒
- 集成ClickHouse真实数据
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_logger, get_service
from utils.decorators import performance_monitor, exception_handler
from db.interfaces.data_access_interface import DataAccessInterface
from analysis.integrated_performance_framework import PerformanceEvaluationFramework

logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """报告生成配置"""
    # 基础配置
    title: str = "量化策略回测报告"
    subtitle: str = ""
    author: str = "量化系统"
    company: str = "量化投资团队"
    logo_path: Optional[str] = None

    # 输出配置
    output_formats: List[str] = None
    output_dir: str = "./reports"
    filename_prefix: str = "backtest_report"

    # 内容配置
    include_executive_summary: bool = True
    include_performance_metrics: bool = True
    include_risk_analysis: bool = True
    include_benchmark_comparison: bool = True
    include_time_series_analysis: bool = True
    include_stress_testing: bool = True
    include_factor_analysis: bool = True
    include_position_analysis: bool = True

    # 可视化配置
    chart_theme: str = "professional"  # professional, dark, light
    chart_dpi: int = 300
    chart_format: str = "png"  # png, svg, pdf
    chart_style: str = "seaborn-v0_8"
    color_palette: str = "viridis"

    # 性能配置
    parallel_chart_generation: bool = True
    max_workers: int = 8
    cache_charts: bool = True
    compress_images: bool = True

    # 模板配置
    template_name: str = "default"
    custom_css: Optional[str] = None
    brand_colors: Dict[str, str] = None

    # 分发配置
    auto_email: bool = False
    email_recipients: List[str] = None
    auto_archive: bool = True
    archive_retention_days: int = 90

    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['html', 'pdf']
        if self.brand_colors is None:
            self.brand_colors = {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'success': '#F18F01',
                'warning': '#C73E1D',
                'accent': '#6A994E'
            }
        if self.email_recipients is None:
            self.email_recipients = []


@dataclass
class ReportGenerationRequest:
    """报告生成请求"""
    strategy_name: str
    evaluation_results: Dict[str, Any]
    config: ReportConfig
    generation_timestamp: datetime = None
    request_id: str = None

    def __post_init__(self):
        if self.generation_timestamp is None:
            self.generation_timestamp = datetime.now()
        if self.request_id is None:
            self.request_id = hashlib.md5(
                f"{self.strategy_name}_{self.generation_timestamp.isoformat()}".encode()
            ).hexdigest()


class BacktestReportEngine:
    """
    高性能回测报告生成引擎

    主要功能：
    1. 统一的报告生成接口
    2. 多格式输出支持
    3. 专业级图表生成
    4. 模板管理
    5. 性能优化
    6. 自动化分发
    """

    def __init__(self,
                 base_config: Optional[ReportConfig] = None,
                 cache_dir: str = "./cache/reports",
                 template_dir: str = "./templates/reports"):
        """
        初始化报告引擎

        Args:
            base_config: 基础配置
            cache_dir: 缓存目录
            template_dir: 模板目录
        """
        self.base_config = base_config or ReportConfig()
        self.cache_dir = Path(cache_dir)
        self.template_dir = Path(template_dir)

        # 创建目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        Path(self.base_config.output_dir).mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self._initialize_components()

        # 性能统计
        self.performance_stats = {
            'total_reports': 0,
            'total_generation_time': 0.0,
            'avg_generation_time': 0.0,
            'chart_cache_hits': 0,
            'chart_cache_misses': 0,
            'template_cache_hits': 0,
            'template_cache_misses': 0
        }

        logger.info(f"回测报告引擎初始化完成")
        logger.info(f"缓存目录: {self.cache_dir}")
        logger.info(f"模板目录: {self.template_dir}")
        logger.info(f"输出目录: {self.base_config.output_dir}")

    def _initialize_components(self):
        """初始化核心组件"""
        # 延迟导入可视化组件以优化启动时间
        self.visualization_engine = None
        self.template_manager = None
        self.distribution_manager = None

        # 获取数据访问服务
        try:
            self.data_access = get_service(DataAccessInterface)
        except Exception as e:
            logger.warning(f"无法获取数据访问服务: {e}")
            self.data_access = None

        # 初始化性能评估框架
        self.performance_framework = PerformanceEvaluationFramework()

        # 缓存管理
        self.chart_cache = {}
        self.template_cache = {}

    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def generate_report(self, request: ReportGenerationRequest) -> Dict[str, Any]:
        """
        生成回测报告

        Args:
            request: 报告生成请求

        Returns:
            Dict[str, Any]: 生成结果包含文件路径和元数据
        """
        start_time = time.time()

        logger.info(f"开始生成回测报告: {request.strategy_name}")
        logger.info(f"请求ID: {request.request_id}")

        try:
            # 1. 预处理和验证
            self._validate_request(request)

            # 2. 初始化延迟加载的组件
            self._ensure_components_loaded()

            # 3. 生成图表
            chart_results = self._generate_charts(request)

            # 4. 生成各种格式的报告
            report_files = {}

            # 并行生成多种格式
            if len(request.config.output_formats) > 1:
                report_files = self._generate_reports_parallel(request, chart_results)
            else:
                # 单格式顺序生成
                for output_format in request.config.output_formats:
                    file_path = self._generate_single_format_report(
                        request, chart_results, output_format
                    )
                    report_files[output_format] = file_path

            # 5. 执行自动化分发
            distribution_results = self._handle_distribution(request, report_files)

            # 6. 构建返回结果
            generation_time = time.time() - start_time
            result = {
                'request_id': request.request_id,
                'strategy_name': request.strategy_name,
                'generation_timestamp': request.generation_timestamp.isoformat(),
                'generation_time_seconds': generation_time,
                'report_files': report_files,
                'chart_files': chart_results.get('chart_files', {}),
                'distribution_results': distribution_results,
                'performance_stats': {
                    'total_charts_generated': len(chart_results.get('chart_files', {})),
                    'chart_cache_usage': {
                        'hits': chart_results.get('cache_hits', 0),
                        'misses': chart_results.get('cache_misses', 0)
                    },
                    'template_cache_usage': {
                        'hits': self.performance_stats['template_cache_hits'],
                        'misses': self.performance_stats['template_cache_misses']
                    }
                },
                'config_used': asdict(request.config)
            }

            # 7. 更新性能统计
            self._update_performance_stats(generation_time)

            logger.info(f"回测报告生成完成: {request.strategy_name}")
            logger.info(f"生成时间: {generation_time:.2f}秒")
            logger.info(f"生成格式: {list(report_files.keys())}")

            return result

        except Exception as e:
            logger.error(f"生成回测报告失败: {e}")
            raise

    def _validate_request(self, request: ReportGenerationRequest):
        """验证请求参数"""
        if not request.strategy_name:
            raise ValueError("策略名称不能为空")

        if not request.evaluation_results:
            raise ValueError("评估结果不能为空")

        if not request.config.output_formats:
            raise ValueError("必须指定至少一种输出格式")

        # 验证输出格式
        supported_formats = {'html', 'pdf', 'excel', 'json', 'markdown'}
        for fmt in request.config.output_formats:
            if fmt.lower() not in supported_formats:
                raise ValueError(f"不支持的输出格式: {fmt}")

    def _ensure_components_loaded(self):
        """确保延迟加载的组件已初始化"""
        if self.visualization_engine is None:
            from reporting.visualization.chart_engine import ChartEngine
            self.visualization_engine = ChartEngine(
                cache_dir=self.cache_dir / "charts",
                config=self.base_config
            )

        if self.template_manager is None:
            from reporting.templates.template_manager import TemplateManager
            self.template_manager = TemplateManager(
                template_dir=self.template_dir,
                cache_dir=self.cache_dir / "templates"
            )

        if self.distribution_manager is None:
            from reporting.distribution.distribution_manager import DistributionManager
            self.distribution_manager = DistributionManager(
                config=self.base_config
            )

    def _generate_charts(self, request: ReportGenerationRequest) -> Dict[str, Any]:
        """生成图表"""
        chart_start_time = time.time()

        logger.info("开始生成图表")

        # 定义需要生成的图表
        chart_specs = self._get_chart_specifications(request)

        # 生成图表
        if request.config.parallel_chart_generation and len(chart_specs) > 1:
            chart_results = self._generate_charts_parallel(request, chart_specs)
        else:
            chart_results = self._generate_charts_sequential(request, chart_specs)

        chart_time = time.time() - chart_start_time
        chart_results['generation_time'] = chart_time

        logger.info(f"图表生成完成，耗时: {chart_time:.2f}秒")

        return chart_results

    def _get_chart_specifications(self, request: ReportGenerationRequest) -> List[Dict[str, Any]]:
        """获取图表规格说明"""
        specs = []
        evaluation_results = request.evaluation_results

        # 收益曲线图
        if request.config.include_performance_metrics:
            specs.append({
                'type': 'returns_curve',
                'title': '累计收益率曲线',
                'data_key': 'performance_metrics',
                'priority': 1
            })

            specs.append({
                'type': 'rolling_performance',
                'title': '滚动性能指标',
                'data_key': 'time_series_analysis',
                'priority': 2
            })

        # 风险分析图表
        if request.config.include_risk_analysis:
            specs.append({
                'type': 'drawdown_curve',
                'title': '回撤曲线',
                'data_key': 'risk_metrics',
                'priority': 1
            })

            specs.append({
                'type': 'risk_return_scatter',
                'title': '风险收益散点图',
                'data_key': 'risk_metrics',
                'priority': 2
            })

        # 基准比较图表
        if request.config.include_benchmark_comparison and 'benchmark_comparison' in evaluation_results:
            specs.append({
                'type': 'benchmark_comparison',
                'title': '基准比较',
                'data_key': 'benchmark_comparison',
                'priority': 1
            })

        # 因子分析图表
        if request.config.include_factor_analysis and 'factor_analysis' in evaluation_results:
            specs.extend([
                {
                    'type': 'factor_exposure',
                    'title': '因子暴露分析',
                    'data_key': 'factor_analysis',
                    'priority': 3
                },
                {
                    'type': 'factor_attribution',
                    'title': '因子归因分析',
                    'data_key': 'factor_analysis',
                    'priority': 3
                }
            ])

        # 持仓分析图表
        if request.config.include_position_analysis and 'position_analysis' in evaluation_results:
            specs.extend([
                {
                    'type': 'position_concentration',
                    'title': '持仓集中度分析',
                    'data_key': 'position_analysis',
                    'priority': 2
                },
                {
                    'type': 'turnover_analysis',
                    'title': '换手率分析',
                    'data_key': 'position_analysis',
                    'priority': 2
                }
            ])

        # 按优先级排序
        specs.sort(key=lambda x: x['priority'])

        return specs

    def _generate_charts_parallel(self, request: ReportGenerationRequest,
                                chart_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """并行生成图表"""
        chart_files = {}
        cache_hits = 0
        cache_misses = 0

        with ThreadPoolExecutor(max_workers=request.config.max_workers) as executor:
            # 提交图表生成任务
            future_to_spec = {
                executor.submit(
                    self.visualization_engine.generate_chart,
                    spec,
                    request.evaluation_results.get(spec['data_key'], {}),
                    request.config
                ): spec
                for spec in chart_specs
            }

            # 收集结果
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    chart_result = future.result()
                    chart_files[spec['type']] = chart_result['file_path']

                    if chart_result.get('from_cache', False):
                        cache_hits += 1
                    else:
                        cache_misses += 1

                    logger.debug(f"✅ 图表生成完成: {spec['title']}")

                except Exception as e:
                    logger.error(f"❌ 图表生成失败: {spec['title']} - {e}")
                    chart_files[spec['type']] = None

        return {
            'chart_files': chart_files,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses
        }

    def _generate_charts_sequential(self, request: ReportGenerationRequest,
                                  chart_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """顺序生成图表"""
        chart_files = {}
        cache_hits = 0
        cache_misses = 0

        for spec in chart_specs:
            try:
                chart_result = self.visualization_engine.generate_chart(
                    spec,
                    request.evaluation_results.get(spec['data_key'], {}),
                    request.config
                )

                chart_files[spec['type']] = chart_result['file_path']

                if chart_result.get('from_cache', False):
                    cache_hits += 1
                else:
                    cache_misses += 1

                logger.debug(f"✅ 图表生成完成: {spec['title']}")

            except Exception as e:
                logger.error(f"❌ 图表生成失败: {spec['title']} - {e}")
                chart_files[spec['type']] = None

        return {
            'chart_files': chart_files,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses
        }

    def _generate_reports_parallel(self, request: ReportGenerationRequest,
                                 chart_results: Dict[str, Any]) -> Dict[str, str]:
        """并行生成多格式报告"""
        report_files = {}

        with ThreadPoolExecutor(max_workers=len(request.config.output_formats)) as executor:
            # 提交报告生成任务
            future_to_format = {
                executor.submit(
                    self._generate_single_format_report,
                    request,
                    chart_results,
                    output_format
                ): output_format
                for output_format in request.config.output_formats
            }

            # 收集结果
            for future in as_completed(future_to_format):
                output_format = future_to_format[future]
                try:
                    file_path = future.result()
                    report_files[output_format] = file_path
                    logger.debug(f"✅ 报告生成完成: {output_format}")
                except Exception as e:
                    logger.error(f"❌ 报告生成失败: {output_format} - {e}")
                    report_files[output_format] = f"Error: {str(e)}"

        return report_files

    def _generate_single_format_report(self, request: ReportGenerationRequest,
                                     chart_results: Dict[str, Any],
                                     output_format: str) -> str:
        """生成单一格式的报告"""
        # 根据格式调用相应的生成器
        if output_format.lower() == 'html':
            from reporting.generators.html_generator import HTMLReportGenerator
            generator = HTMLReportGenerator(
                template_manager=self.template_manager,
                config=request.config
            )
        elif output_format.lower() == 'pdf':
            from reporting.generators.pdf_generator import PDFReportGenerator
            generator = PDFReportGenerator(
                template_manager=self.template_manager,
                config=request.config
            )
        elif output_format.lower() == 'excel':
            from reporting.generators.excel_generator import ExcelReportGenerator
            generator = ExcelReportGenerator(
                config=request.config
            )
        elif output_format.lower() == 'json':
            from reporting.generators.json_generator import JSONReportGenerator
            generator = JSONReportGenerator(
                config=request.config
            )
        elif output_format.lower() == 'markdown':
            from reporting.generators.markdown_generator import MarkdownReportGenerator
            generator = MarkdownReportGenerator(
                template_manager=self.template_manager,
                config=request.config
            )
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")

        return generator.generate_report(
            request.evaluation_results,
            chart_results.get('chart_files', {}),
            request.strategy_name,
            request.request_id
        )

    def _handle_distribution(self, request: ReportGenerationRequest,
                           report_files: Dict[str, str]) -> Dict[str, Any]:
        """处理报告分发"""
        if not (request.config.auto_email or request.config.auto_archive):
            return {'distribution_enabled': False}

        return self.distribution_manager.distribute_reports(
            report_files,
            request
        )

    def _update_performance_stats(self, generation_time: float):
        """更新性能统计"""
        self.performance_stats['total_reports'] += 1
        self.performance_stats['total_generation_time'] += generation_time
        self.performance_stats['avg_generation_time'] = (
            self.performance_stats['total_generation_time'] /
            self.performance_stats['total_reports']
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """获取引擎性能报告"""
        return {
            'engine_stats': self.performance_stats,
            'cache_stats': {
                'chart_cache_size': len(self.chart_cache),
                'template_cache_size': len(self.template_cache)
            },
            'system_resources': self._get_system_resources()
        }

    def _get_system_resources(self) -> Dict[str, Any]:
        """获取系统资源信息"""
        try:
            import psutil
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent,
                'disk_free_gb': psutil.disk_usage('.').free / 1024 / 1024 / 1024
            }
        except ImportError:
            return {'error': 'psutil not available'}

    def clear_cache(self):
        """清理缓存"""
        try:
            # 清理图表缓存文件
            chart_cache_dir = self.cache_dir / "charts"
            if chart_cache_dir.exists():
                for file in chart_cache_dir.glob("*"):
                    if file.is_file():
                        file.unlink()

            # 清理模板缓存文件
            template_cache_dir = self.cache_dir / "templates"
            if template_cache_dir.exists():
                for file in template_cache_dir.glob("*"):
                    if file.is_file():
                        file.unlink()

            # 清理内存缓存
            self.chart_cache.clear()
            self.template_cache.clear()

            logger.info("报告引擎缓存已清理")

        except Exception as e:
            logger.error(f"清理缓存失败: {e}")


# 全局报告引擎实例
report_engine = BacktestReportEngine()


def generate_backtest_report(strategy_name: str,
                           evaluation_results: Dict[str, Any],
                           config: Optional[ReportConfig] = None) -> Dict[str, Any]:
    """
    便捷的回测报告生成函数

    Args:
        strategy_name: 策略名称
        evaluation_results: 评估结果
        config: 报告配置

    Returns:
        Dict[str, Any]: 生成结果
    """
    request = ReportGenerationRequest(
        strategy_name=strategy_name,
        evaluation_results=evaluation_results,
        config=config or ReportConfig()
    )

    return report_engine.generate_report(request)