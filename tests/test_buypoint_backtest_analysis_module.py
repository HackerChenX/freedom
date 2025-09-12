#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
买点回测分析模块全面测试脚本

测试买点回测分析模块的所有核心功能，包括：
- 增强回测引擎测试
- 买点识别系统测试  
- 回测评估系统测试
- 主控制器集成测试
确保100%功能正常运行
"""

import sys
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append('.')

def test_enhanced_backtest_engine():
    """测试增强回测引擎"""
    print("🔧 测试增强回测引擎...")
    
    try:
        from analysis.buypoints.enhanced_backtest_engine import (
            EnhancedBacktestEngine, BacktestConfig, BuyPointData
        )
        
        # 创建测试配置
        config = BacktestConfig(
            periods=['日线'],
            indicators=['MA', 'MACD'],
            lookback_days=100,
            parallel_workers=2
        )
        
        # 创建回测引擎
        engine = EnhancedBacktestEngine(config)
        
        # 创建测试买点数据
        test_buypoints = [
            BuyPointData(
                stock_code='000001',
                stock_name='平安银行',
                buypoint_date='20241201',
                expected_return=0.05,
                holding_period=10
            ),
            BuyPointData(
                stock_code='000002',
                stock_name='万科A',
                buypoint_date='20241202',
                expected_return=0.03,
                holding_period=15
            )
        ]
        
        # 运行回测
        summary = engine.run_backtest(test_buypoints)
        
        # 验证结果
        assert summary is not None, "回测汇总不能为空"
        assert summary.total_buypoints == len(test_buypoints), "买点数量不匹配"
        assert summary.execution_time > 0, "执行时间应大于0"
        
        # 获取性能报告
        performance_report = engine.get_performance_report()
        assert 'engine_stats' in performance_report, "性能报告应包含引擎统计"
        
        print("✅ 增强回测引擎测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 增强回测引擎测试失败: {e}")
        traceback.print_exc()
        return False

def test_enhanced_buypoint_detector():
    """测试增强买点识别系统"""
    print("🔍 测试增强买点识别系统...")
    
    try:
        from analysis.buypoints.enhanced_buypoint_detector import (
            EnhancedBuyPointDetector, BuyPointDetectionConfig, BuyPointType
        )
        
        # 创建测试配置
        config = BuyPointDetectionConfig(
            min_confidence=0.5,
            min_score=50.0,
            lookback_periods=20,
            detection_types=[BuyPointType.VOLUME_BREAKOUT, BuyPointType.PULLBACK_SUPPORT]
        )
        
        # 创建买点检测器
        detector = EnhancedBuyPointDetector(config)
        
        # 测试股票代码
        test_stocks = ['000001', '000002']
        
        # 检测买点
        signals = detector.detect_buypoints(test_stocks)
        
        # 验证结果
        assert isinstance(signals, list), "检测结果应为列表"
        
        # 验证信号结构
        for signal in signals:
            assert hasattr(signal, 'stock_code'), "信号应包含股票代码"
            assert hasattr(signal, 'confidence'), "信号应包含置信度"
            assert hasattr(signal, 'score'), "信号应包含评分"
            assert 0 <= signal.confidence <= 1, "置信度应在0-1之间"
            assert 0 <= signal.score <= 100, "评分应在0-100之间"
        
        print(f"✅ 增强买点识别系统测试通过，检测到 {len(signals)} 个信号")
        return True
        
    except Exception as e:
        print(f"❌ 增强买点识别系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_enhanced_backtest_evaluator():
    """测试增强回测评估系统"""
    print("📊 测试增强回测评估系统...")
    
    try:
        from analysis.buypoints.enhanced_backtest_evaluator import (
            EnhancedBacktestEvaluator, BacktestPerformanceMetrics, RiskMetrics
        )
        
        # 创建评估器
        evaluator = EnhancedBacktestEvaluator()
        
        # 创建模拟回测结果
        mock_results = [
            {'trade_id': 'trade_1', 'return': 0.05, 'date': '2024-12-01'},
            {'trade_id': 'trade_2', 'return': -0.02, 'date': '2024-12-02'},
            {'trade_id': 'trade_3', 'return': 0.03, 'date': '2024-12-03'},
            {'trade_id': 'trade_4', 'return': 0.01, 'date': '2024-12-04'},
            {'trade_id': 'trade_5', 'return': -0.01, 'date': '2024-12-05'}
        ]
        
        # 运行评估
        evaluation_result = evaluator.evaluate_backtest_results(mock_results)
        
        # 验证结果
        assert evaluation_result is not None, "评估结果不能为空"
        assert hasattr(evaluation_result, 'performance_metrics'), "应包含性能指标"
        assert hasattr(evaluation_result, 'risk_metrics'), "应包含风险指标"
        assert hasattr(evaluation_result, 'overall_rating'), "应包含综合评级"
        assert hasattr(evaluation_result, 'recommendations'), "应包含建议"
        
        # 验证性能指标
        perf_metrics = evaluation_result.performance_metrics
        assert isinstance(perf_metrics, BacktestPerformanceMetrics), "性能指标类型错误"
        assert perf_metrics.total_trades == len(mock_results), "交易数量不匹配"
        
        # 验证风险指标
        risk_metrics = evaluation_result.risk_metrics
        assert isinstance(risk_metrics, RiskMetrics), "风险指标类型错误"
        
        # 测试报告导出
        output_file = "reports/test_evaluation_report.json"
        export_success = evaluator.export_evaluation_report(evaluation_result, output_file)
        assert export_success, "报告导出应成功"
        
        print("✅ 增强回测评估系统测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 增强回测评估系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_buypoint_backtest_analysis_controller():
    """测试买点回测分析主控制器"""
    print("🎛️ 测试买点回测分析主控制器...")
    
    try:
        from analysis.buypoints.buypoint_backtest_analysis_controller import (
            BuyPointBacktestAnalysisController, AnalysisConfig
        )
        from analysis.buypoints.enhanced_backtest_engine import BuyPointData
        
        # 创建测试配置
        config = AnalysisConfig(
            enable_realtime_detection=True,
            enable_performance_evaluation=True,
            output_directory="reports/test_buypoint_analysis"
        )
        
        # 创建控制器
        controller = BuyPointBacktestAnalysisController(config)
        
        # 创建测试数据
        test_buypoints = [
            BuyPointData(
                stock_code='000001',
                stock_name='平安银行',
                buypoint_date='20241201'
            )
        ]
        
        test_stocks = ['000001', '000002']
        
        # 测试综合分析
        analysis_result = controller.run_comprehensive_analysis(
            buypoints=test_buypoints,
            stock_codes_for_detection=test_stocks
        )
        
        # 验证结果
        assert analysis_result is not None, "分析结果不能为空"
        assert hasattr(analysis_result, 'analysis_id'), "应包含分析ID"
        assert hasattr(analysis_result, 'backtest_summary'), "应包含回测汇总"
        assert hasattr(analysis_result, 'detected_signals'), "应包含检测信号"
        assert hasattr(analysis_result, 'recommendations'), "应包含建议"
        
        # 测试单独功能
        backtest_summary = controller.run_backtest_only(test_buypoints)
        assert backtest_summary is not None, "回测汇总不能为空"
        
        detected_signals = controller.run_detection_only(test_stocks)
        assert isinstance(detected_signals, list), "检测信号应为列表"
        
        # 测试统计信息
        stats = controller.get_analysis_statistics()
        assert 'analysis_stats' in stats, "统计信息应包含分析统计"
        assert 'engine_performance' in stats, "统计信息应包含引擎性能"
        
        # 测试系统健康检查
        health_status = controller.validate_system_health()
        assert 'overall_status' in health_status, "健康状态应包含总体状态"
        assert 'components' in health_status, "健康状态应包含组件状态"
        
        print("✅ 买点回测分析主控制器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 买点回测分析主控制器测试失败: {e}")
        traceback.print_exc()
        return False

def test_integration_workflow():
    """测试集成工作流"""
    print("🔄 测试集成工作流...")
    
    try:
        from analysis.buypoints.buypoint_backtest_analysis_controller import (
            BuyPointBacktestAnalysisController, AnalysisConfig
        )
        from analysis.buypoints.enhanced_backtest_engine import BuyPointData
        
        # 创建完整的工作流测试
        config = AnalysisConfig(
            enable_realtime_detection=True,
            enable_performance_evaluation=True,
            output_directory="reports/integration_test"
        )
        
        controller = BuyPointBacktestAnalysisController(config)
        
        # 模拟完整的分析流程
        buypoints = [
            BuyPointData(stock_code='000001', buypoint_date='20241201'),
            BuyPointData(stock_code='000002', buypoint_date='20241202'),
            BuyPointData(stock_code='000858', buypoint_date='20241203')
        ]
        
        detection_stocks = ['000001', '000002', '000858', '002415']
        
        # 运行完整分析
        start_time = time.time()
        result = controller.run_comprehensive_analysis(
            buypoints=buypoints,
            stock_codes_for_detection=detection_stocks
        )
        execution_time = time.time() - start_time
        
        # 验证工作流结果
        assert result.execution_time > 0, "执行时间应大于0"
        assert len(result.recommendations) > 0, "应生成建议"
        assert result.performance_metrics is not None, "应包含性能指标"
        
        # 验证文件输出
        assert os.path.exists(config.output_directory), "输出目录应存在"
        
        print(f"✅ 集成工作流测试通过，耗时: {execution_time:.2f}秒")
        return True
        
    except Exception as e:
        print(f"❌ 集成工作流测试失败: {e}")
        traceback.print_exc()
        return False

def test_performance_benchmarks():
    """测试性能基准"""
    print("⚡ 测试性能基准...")

    try:
        from analysis.buypoints.buypoint_backtest_analysis_controller import (
            BuyPointBacktestAnalysisController, AnalysisConfig
        )
        from analysis.buypoints.enhanced_backtest_engine import BuyPointData

        # 性能测试配置
        config = AnalysisConfig(
            enable_realtime_detection=False,  # 关闭实时检测以专注回测性能
            enable_performance_evaluation=False,
            output_directory="reports/performance_test"
        )

        controller = BuyPointBacktestAnalysisController(config)

        # 创建大量测试数据
        large_buypoints = []
        for i in range(50):  # 50个买点
            date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            large_buypoints.append(BuyPointData(
                stock_code=f'00000{i%10}',
                buypoint_date=date
            ))

        # 性能测试
        start_time = time.time()
        result = controller.run_backtest_only(large_buypoints)
        execution_time = time.time() - start_time

        # 性能验证
        assert execution_time < 60.0, f"大批量回测应在60秒内完成，实际: {execution_time:.2f}秒"
        assert result.total_buypoints == len(large_buypoints), "处理的买点数量应匹配"

        # 检测性能测试
        large_stock_list = [f'00000{i}' for i in range(20)]  # 20只股票

        start_time = time.time()
        signals = controller.run_detection_only(large_stock_list)
        detection_time = time.time() - start_time

        assert detection_time < 30.0, f"大批量检测应在30秒内完成，实际: {detection_time:.2f}秒"

        print(f"✅ 性能基准测试通过 - 回测: {execution_time:.2f}s, 检测: {detection_time:.2f}s")
        return True

    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    print("🛡️ 测试错误处理...")

    try:
        from analysis.buypoints.buypoint_backtest_analysis_controller import (
            BuyPointBacktestAnalysisController, AnalysisConfig
        )
        from analysis.buypoints.enhanced_backtest_engine import BuyPointData

        controller = BuyPointBacktestAnalysisController()

        # 测试空数据处理
        empty_result = controller.run_backtest_only([])
        assert empty_result is not None, "空数据应返回有效结果"
        assert empty_result.total_buypoints == 0, "空数据的买点数量应为0"

        # 测试无效股票代码
        invalid_signals = controller.run_detection_only(['INVALID_CODE'])
        assert isinstance(invalid_signals, list), "无效代码应返回空列表"

        # 测试无效买点数据
        invalid_buypoints = [
            BuyPointData(stock_code='', buypoint_date='invalid_date')
        ]

        try:
            controller.run_backtest_only(invalid_buypoints)
            # 应该能处理无效数据而不崩溃
        except Exception:
            pass  # 预期可能出现异常，但不应导致程序崩溃

        print("✅ 错误处理测试通过")
        return True

    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        traceback.print_exc()
        return False

def test_data_quality_validation():
    """测试数据质量验证"""
    print("🔍 测试数据质量验证...")

    try:
        from analysis.buypoints.enhanced_buypoint_detector import (
            EnhancedBuyPointDetector, BuyPointDetectionConfig
        )

        # 创建检测器
        config = BuyPointDetectionConfig(min_confidence=0.7, min_score=70.0)
        detector = EnhancedBuyPointDetector(config)

        # 测试质量过滤
        test_stocks = ['000001']  # 使用真实股票代码
        signals = detector.detect_buypoints(test_stocks)

        # 验证信号质量
        for signal in signals:
            assert signal.confidence >= config.min_confidence, f"信号置信度 {signal.confidence} 低于阈值 {config.min_confidence}"
            assert signal.score >= config.min_score, f"信号评分 {signal.score} 低于阈值 {config.min_score}"

        # 验证数据结构完整性
        for signal in signals:
            assert signal.stock_code, "股票代码不能为空"
            assert signal.detection_date, "检测日期不能为空"
            assert signal.buypoint_type, "买点类型不能为空"
            assert signal.quality, "买点质量不能为空"
            assert isinstance(signal.technical_indicators, dict), "技术指标应为字典"
            assert isinstance(signal.recommendations, list), "建议应为列表"

        print("✅ 数据质量验证测试通过")
        return True

    except Exception as e:
        print(f"❌ 数据质量验证测试失败: {e}")
        traceback.print_exc()
        return False

def test_configuration_flexibility():
    """测试配置灵活性"""
    print("⚙️ 测试配置灵活性...")

    try:
        from analysis.buypoints.enhanced_backtest_engine import BacktestConfig
        from analysis.buypoints.enhanced_buypoint_detector import BuyPointDetectionConfig, BuyPointType
        from analysis.buypoints.buypoint_backtest_analysis_controller import AnalysisConfig

        # 测试不同的回测配置
        configs = [
            BacktestConfig(periods=['日线'], indicators=['MA'], parallel_workers=1),
            BacktestConfig(periods=['日线', '60分钟'], indicators=['MA', 'MACD'], parallel_workers=2),
            BacktestConfig(periods=['日线'], indicators=['MA', 'MACD', 'RSI'], parallel_workers=4)
        ]

        for i, config in enumerate(configs):
            assert len(config.periods) > 0, f"配置{i+1}周期不能为空"
            assert len(config.indicators) > 0, f"配置{i+1}指标不能为空"
            assert config.parallel_workers > 0, f"配置{i+1}并行工作线程数应大于0"

        # 测试不同的检测配置
        detection_configs = [
            BuyPointDetectionConfig(detection_types=[BuyPointType.VOLUME_BREAKOUT]),
            BuyPointDetectionConfig(detection_types=[BuyPointType.PULLBACK_SUPPORT, BuyPointType.TREND_REVERSAL]),
            BuyPointDetectionConfig(detection_types=list(BuyPointType))
        ]

        for i, config in enumerate(detection_configs):
            assert len(config.detection_types) > 0, f"检测配置{i+1}类型不能为空"
            assert 0 <= config.min_confidence <= 1, f"检测配置{i+1}置信度阈值应在0-1之间"
            assert 0 <= config.min_score <= 100, f"检测配置{i+1}评分阈值应在0-100之间"

        # 测试分析配置
        analysis_configs = [
            AnalysisConfig(enable_realtime_detection=True, enable_performance_evaluation=True),
            AnalysisConfig(enable_realtime_detection=False, enable_performance_evaluation=True),
            AnalysisConfig(enable_realtime_detection=True, enable_performance_evaluation=False)
        ]

        for config in analysis_configs:
            assert hasattr(config, 'backtest_config'), "分析配置应包含回测配置"
            assert hasattr(config, 'detection_config'), "分析配置应包含检测配置"
            assert hasattr(config, 'output_directory'), "分析配置应包含输出目录"

        print("✅ 配置灵活性测试通过")
        return True

    except Exception as e:
        print(f"❌ 配置灵活性测试失败: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始买点回测分析模块全面测试")
    print("=" * 60)

    test_functions = [
        test_enhanced_backtest_engine,
        test_enhanced_buypoint_detector,
        test_enhanced_backtest_evaluator,
        test_buypoint_backtest_analysis_controller,
        test_integration_workflow,
        test_performance_benchmarks,
        test_error_handling,
        test_data_quality_validation,
        test_configuration_flexibility
    ]

    passed_tests = 0
    total_tests = len(test_functions)

    start_time = time.time()

    for i, test_func in enumerate(test_functions, 1):
        print(f"\n[{i}/{total_tests}] {test_func.__name__}")
        print("-" * 40)

        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ 测试 {test_func.__name__} 失败")
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 异常: {e}")
            traceback.print_exc()

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("🎯 买点回测分析模块测试总结")
    print("=" * 60)
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"❌ 失败测试: {total_tests - passed_tests}/{total_tests}")
    print(f"⏱️ 总耗时: {total_time:.2f}秒")
    print(f"📊 通过率: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！买点回测分析模块功能正常")
        return True
    else:
        print(f"\n⚠️ 有 {total_tests - passed_tests} 个测试失败，需要修复")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
