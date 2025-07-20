#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合选股测试系统命令行界面

提供命令行接口来运行综合选股测试，支持配置文件加载、参数覆盖、
进度报告和实时状态更新
"""

import argparse
import sys
import os
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import signal
import threading

from utils.logger import getLogger
from .stock_selection_tester import StockSelectionTester
from .test_config_manager import TestConfigManager, get_config_manager
from .test_result_validator import TestResultValidator, get_test_result_validator
from .enhanced_report_generator import EnhancedReportGenerator

logger = getLogger(__name__)


class ProgressReporter:
    """进度报告器"""
    
    def __init__(self):
        """初始化进度报告器"""
        self.start_time = None
        self.last_update_time = None
        self.update_interval = 10  # 10秒更新一次
        self.is_running = False
        self.current_status = "准备中"
        self.progress_data = {}
        
    def start(self):
        """开始进度报告"""
        self.start_time = datetime.now()
        self.last_update_time = self.start_time
        self.is_running = True
        self.current_status = "运行中"
        
        print(f"\nfrom config.config import get_config\n{'='*60}")
        print(f"综合选股测试开始 - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
    
    def update_progress(self, status: str, progress_data: Dict[str, Any] = None):
        """更新进度"""
        current_time = datetime.now()
        self.current_status = status
        
        if progress_data:
            self.progress_data.update(progress_data)
        
        # 检查是否需要更新显示
        if (current_time - self.last_update_time).total_seconds() >= self.update_interval:
            self._display_progress(current_time)
            self.last_update_time = current_time
    
    def _display_progress(self, current_time: datetime):
        """显示进度信息"""
        elapsed = (current_time - self.start_time).total_seconds()
        
        print(f"\n[{current_time.strftime('%H:%M:%S')}] {self.current_status}")
        print(f"运行时间: {elapsed:.0f}秒")
        
        if self.progress_data:
            for key, value in self.progress_data.items():
                if isinstance(value, (int, float)):
                    if key.endswith('_rate') or key.endswith('_ratio'):
                        print(f"  {key}: {value:.1%}")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
    
    def finish(self, success: bool = True):
        """结束进度报告"""
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*60}")
        if success:
            print(f"测试完成 - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"测试中断 - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总运行时间: {total_time:.0f}秒")
        print(f"{'='*60}\n")
        
        self.is_running = False


class TestCLI:
    """测试命令行界面"""
    
    def __init__(self):
        """初始化CLI"""
        self.config_manager = None
        self.tester = None
        self.validator = None
        self.reporter = None
        self.progress_reporter = ProgressReporter()
        self.interrupted = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到中断信号 {signum}，正在安全退出...")
        self.interrupted = True
        
        if self.tester:
            self.tester.stop_testing()
        
        self.progress_reporter.finish(success=False)
        sys.exit(1)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            description="综合选股测试系统",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  # 使用默认配置运行测试
  python cli.py run
  
  # 使用自定义配置文件
  python cli.py run --config custom_config.yaml
  
  # 覆盖特定参数
  python cli.py run --start-date 20240101 --end-date 20240331 --timeout 600
  
  # 只测试特定指标
  python cli.py run --indicators MA MACD KDJ
  
  # 只测试特定形态
  python cli.py run --patterns MA_GOLDEN_CROSS MACD_GOLDEN_CROSS
  
  # 验证配置文件
  python cli.py validate-config --config test_config.yaml
  
  # 生成默认配置文件
  python cli.py generate-config --output default_config.yaml
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # 运行测试命令
        run_parser = subparsers.add_parser('run', help='运行综合选股测试')
        run_parser.add_argument('--config', '-c', 
                               default='tests/comprehensive/test_config.yaml',
                               help='配置文件路径 (默认: tests/comprehensive/test_config.yaml)')
        run_parser.add_argument('--start-date', 
                               help='测试开始日期 (YYYYMMDD格式)')
        run_parser.add_argument('--end-date', 
                               help='测试结束日期 (YYYYMMDD格式)')
        run_parser.add_argument('--timeout', type=int,
                               help='超时时间（秒）')
        run_parser.add_argument('--max-workers', type=int,
                               help='最大工作线程数')
        run_parser.add_argument('--indicators', nargs='+',
                               help='要测试的指标列表')
        run_parser.add_argument('--patterns', nargs='+',
                               help='要测试的形态列表')
        run_parser.add_argument('--output-dir', 
                               help='报告输出目录')
        run_parser.add_argument('--output-formats', nargs='+',
                               choices=['json', 'csv', 'html'],
                               help='报告输出格式')
        run_parser.add_argument('--verbose', '-v', action='store_true',
                               help='详细输出')
        run_parser.add_argument('--dry-run', action='store_true',
                               help='试运行（不执行实际测试）')
        
        # 验证配置命令
        validate_parser = subparsers.add_parser('validate-config', help='验证配置文件')
        validate_parser.add_argument('--config', '-c', required=True,
                                   help='要验证的配置文件路径')
        
        # 生成配置命令
        generate_parser = subparsers.add_parser('generate-config', help='生成默认配置文件')
        generate_parser.add_argument('--output', '-o', required=True,
                                   help='输出配置文件路径')
        generate_parser.add_argument('--template', 
                                   choices=['minimal', 'comprehensive'],
                                   default='comprehensive',
                                   help='配置模板类型')
        
        # 状态查询命令
        status_parser = subparsers.add_parser('status', help='查询测试状态')
        status_parser.add_argument('--test-id', 
                                 help='测试ID')
        
        return parser
    
    def run_test(self, args) -> int:
        """运行测试"""
        try:
            # 加载配置
            self.config_manager = get_config_manager(args.config)
            config = self.config_manager.get_config()
            
            # 应用命令行参数覆盖
            self._apply_cli_overrides(config, args)
            
            # 验证配置
            validation_errors = self.config_manager.validate_config()
            if validation_errors:
                print("配置验证失败:")
                for error in validation_errors:
                    print(f"  - {error}")
                return 1
            
            if args.verbose:
                print("配置信息:")
                self._print_config_summary(config)
            
            if args.dry_run:
                print("试运行模式 - 配置验证通过，未执行实际测试")
                return 0
            
            # 初始化组件
            self.tester = StockSelectionTester()
            self.validator = get_test_result_validator()
            self.reporter = EnhancedReportGenerator()
            
            # 开始进度报告
            self.progress_reporter.start()
            
            # 运行测试
            test_results = self._run_comprehensive_test(config)
            
            if self.interrupted:
                return 1
            
            # 验证结果
            validation_result = self.validator.validate_test_results(test_results)
            
            # 生成报告
            self._generate_reports(test_results, validation_result, config, args)
            
            # 显示结果摘要
            self._print_results_summary(test_results, validation_result)
            
            self.progress_reporter.finish(success=True)
            
            return 0 if validation_result.overall_validation_passed else 2
            
        except KeyboardInterrupt:
            print("\n测试被用户中断")
            self.progress_reporter.finish(success=False)
            return 1
        except Exception as e:
            logger.error(f"运行测试失败: {e}")
            print(f"错误: {e}")
            self.progress_reporter.finish(success=False)
            return 1
    
    def _apply_cli_overrides(self, config, args):
        """应用命令行参数覆盖"""
        overrides = {}
        
        # 执行配置覆盖
        if args.timeout:
            overrides.setdefault('execution', {})['timeout_seconds'] = args.timeout
        if args.max_workers:
            overrides.setdefault('execution', {})['max_workers'] = args.max_workers
        
        # 测试范围覆盖
        if args.start_date or args.end_date:
            overrides.setdefault('test_scope', {}).setdefault('date_range', {})
            if args.start_date:
                overrides['test_scope']['date_range']['start_date'] = args.start_date
            if args.end_date:
                overrides['test_scope']['date_range']['end_date'] = args.end_date
        
        # 报告配置覆盖
        if args.output_formats:
            overrides.setdefault('reporting', {})['output_formats'] = args.output_formats
        
        # 指标和形态过滤
        if args.indicators:
            overrides['indicators_to_test'] = args.indicators
        if args.patterns:
            overrides['patterns_to_test'] = args.patterns
        
        # 应用覆盖
        if overrides:
            self.config_manager.update_config(overrides)
    
    def _run_comprehensive_test(self, config):
        """运行综合测试"""
        # 创建测试线程以便监控进度
        test_thread = threading.Thread(
            target=self._execute_test_with_monitoring,
            args=(config,)
        )
        test_thread.daemon = True
        test_thread.start()
        
        # 等待测试完成
        test_thread.join()
        
        if hasattr(self, '_test_results'):
            return self._test_results
        else:
            raise Exception("测试执行失败")
    
    def _execute_test_with_monitoring(self, config):
        """执行测试并监控进度"""
        try:
            self.progress_reporter.update_progress("初始化测试环境")
            
            # 运行测试
            self._test_results = self.tester.run_comprehensive_test(
                start_date=config.test_scope.date_range.start_date,
                end_date=config.test_scope.date_range.end_date,
                indicators_to_test=config.indicators_to_test,
                patterns_to_test=config.patterns_to_test,
                progress_callback=self._progress_callback
            )
            
        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            self._test_results = None
    
    def _progress_callback(self, status: str, progress_data: Dict[str, Any] = None):
        """进度回调函数"""
        self.progress_reporter.update_progress(status, progress_data)
    
    def _generate_reports(self, test_results, validation_result, config, args):
        """生成报告"""
        self.progress_reporter.update_progress("生成测试报告")
        
        output_dir = args.output_dir or "test_reports"
        output_formats = args.output_formats or config.reporting.output_formats
        
        # 生成测试报告
        for format_type in output_formats:
            try:
                if format_type == 'json':
                    self.reporter.generate_json_report(test_results, output_dir)
                elif format_type == 'csv':
                    self.reporter.generate_csv_report(test_results, output_dir)
                elif format_type == 'html':
                    self.reporter.generate_html_report(test_results, output_dir)
                
                print(f"已生成 {format_type.upper()} 报告")
                
            except Exception as e:
                logger.error(f"生成 {format_type} 报告失败: {e}")
                print(f"警告: 生成 {format_type} 报告失败: {e}")
        
        # 生成验证报告
        try:
            validation_report_path = os.path.join(output_dir, "validation_report.json")
            os.makedirs(output_dir, exist_ok=True)
            
            with open(validation_report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'validation_result': {
                        'test_id': validation_result.test_id,
                        'validation_time': validation_result.validation_time.isoformat(),
                        'overall_validation_passed': validation_result.overall_validation_passed,
                        'validation_score': validation_result.validation_score,
                        'summary_statistics': validation_result.summary_statistics,
                        'recommendations': validation_result.recommendations
                    }
                }, f, ensure_ascii=False, indent=2)
            
            print(f"已生成验证报告: {validation_report_path}")
            
        except Exception as e:
            logger.error(f"生成验证报告失败: {e}")
            print(f"警告: 生成验证报告失败: {e}")
    
    def _print_config_summary(self, config):
        """打印配置摘要"""
        print(f"  测试日期范围: {config.test_scope.date_range.start_date} - {config.test_scope.date_range.end_date}")
        print(f"  超时时间: {config.execution.timeout_seconds}秒")
        print(f"  最大工作线程: {config.execution.max_workers}")
        print(f"  股票池: {config.test_scope.stock_universe}")
        print(f"  最小成交量: {config.test_scope.min_volume:,}")
        print(f"  最小价格: {config.test_scope.min_price}")
        
        if config.indicators_to_test:
            print(f"  指定指标: {', '.join(config.indicators_to_test)}")
        if config.patterns_to_test:
            print(f"  指定形态: {', '.join(config.patterns_to_test)}")
        
        print()
    
    def _print_results_summary(self, test_results, validation_result):
        """打印结果摘要"""
        print(f"\n{'='*60}")
        print("测试结果摘要")
        print(f"{'='*60}")
        
        print(f"测试ID: {test_results.test_id}")
        print(f"测试时间: {test_results.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {test_results.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"执行时长: {(test_results.end_time - test_results.start_time).total_seconds():.0f}秒")
        print()
        
        print("测试统计:")
        print(f"  指标数量: {test_results.total_indicators_tested}")
        print(f"  形态数量: {test_results.total_patterns_tested}")
        print(f"  选中股票: {test_results.total_stocks_selected}")
        print(f"  验证次数: {test_results.total_verifications_performed}")
        print(f"  总体成功率: {test_results.overall_success_rate:.1%}")
        print()
        
        print("验证结果:")
        print(f"  验证通过: {'是' if validation_result.overall_validation_passed else '否'}")
        print(f"  验证评分: {validation_result.validation_score:.2f}")
        print(f"  有选股的形态: {validation_result.patterns_with_selections}/{validation_result.total_patterns}")
        print(f"  验证通过的形态: {validation_result.patterns_validation_passed}/{validation_result.total_patterns}")
        
        if validation_result.recommendations:
            print("\n改进建议:")
            for i, recommendation in enumerate(validation_result.recommendations[:5], 1):
                print(f"  {i}. {recommendation}")
        
        print(f"{'='*60}")
    
    def validate_config(self, args) -> int:
        """验证配置文件"""
        try:
            config_manager = get_config_manager(args.config)
            validation_errors = config_manager.validate_config()
            
            if validation_errors:
                print(f"配置文件 {args.config} 验证失败:")
                for error in validation_errors:
                    print(f"  - {error}")
                return 1
            else:
                print(f"配置文件 {args.config} 验证通过")
                return 0
                
        except Exception as e:
            print(f"验证配置文件失败: {e}")
            return 1
    
    def generate_config(self, args) -> int:
        """生成配置文件"""
        try:
            config_manager = TestConfigManager()
            config_manager.save_config(args.output)
            print(f"已生成配置文件: {args.output}")
            return 0
            
        except Exception as e:
            print(f"生成配置文件失败: {e}")
            return 1
    
    def show_status(self, args) -> int:
        """显示状态"""
        # 这里可以实现状态查询逻辑
        print("状态查询功能待实现")
        return 0
    
    def run(self, argv: List[str] = None) -> int:
        """运行CLI"""
        parser = self.create_parser()
        args = parser.parse_args(argv)
        
        if not args.command:
            parser.print_help()
            return 1
        
        # 设置日志级别
        if hasattr(args, 'verbose') and args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 执行命令
        if args.command == 'run':
            return self.run_test(args)
        elif args.command == 'validate-config':
            return self.validate_config(args)
        elif args.command == 'generate-config':
            return self.generate_config(args)
        elif args.command == 'status':
            return self.show_status(args)
        else:
            parser.print_help()
            return 1


def main():
    """主函数"""
    cli = TestCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())