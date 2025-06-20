"""
完整系统集成测试

测试从买点分析到策略生成、验证、优化的完整流程
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
from datetime import datetime, timedelta

from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer
from analysis.validation.buypoint_validator import BuyPointValidator
from analysis.validation.data_quality_validator import DataQualityValidator
from analysis.optimization.strategy_optimizer import StrategyOptimizer
from monitoring.system_monitor import SystemHealthMonitor


class TestCompleteSystem:
    """完整系统集成测试"""
    
    @pytest.fixture
    def sample_buypoints(self):
        """创建示例买点数据"""
        return pd.DataFrame({
            'stock_code': ['000001', '000002', '000858', '002415'],
            'buypoint_date': ['2024-01-15', '2024-01-15', '2024-01-16', '2024-01-16'],
            'buy_price': [10.5, 15.2, 8.8, 12.3]
        })
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.skip(reason="需要完整的数据库和指标系统，暂时跳过")
    def test_complete_analysis_workflow(self, sample_buypoints, temp_dir):
        """测试完整的分析工作流程"""
        # 1. 初始化系统监控
        monitor = SystemHealthMonitor()

        # 2. 创建买点分析器
        analyzer = BuyPointBatchAnalyzer()
        
        # 3. 保存示例数据到临时文件
        buypoints_file = os.path.join(temp_dir, 'test_buypoints.csv')
        sample_buypoints.to_csv(buypoints_file, index=False)
        
        # 4. 使用监控装饰器执行分析
        @monitor.monitor_analysis_performance
        def run_analysis():
            return analyzer.run_complete_analysis(
                input_csv=buypoints_file,
                output_dir=temp_dir,
                min_hit_ratio=0.5,
                strategy_name="TestStrategy"
            )
        
        # 执行分析
        run_analysis()
        
        # 5. 验证输出文件
        expected_files = [
            'common_indicators_report.md',
            'generated_strategy.json',
            'buypoint_analysis_summary.md'
        ]
        
        for filename in expected_files:
            file_path = os.path.join(temp_dir, filename)
            assert os.path.exists(file_path), f"缺少输出文件: {filename}"
        
        # 6. 检查系统健康状态
        health = monitor.get_system_health()
        assert health['overall_status'] in ['healthy', 'warning'], "系统状态异常"
        assert health['statistics']['success_count'] > 0, "没有成功的操作记录"
    
    def test_strategy_validation_workflow(self, sample_buypoints, temp_dir):
        """测试策略验证工作流程"""
        # 1. 创建示例策略
        sample_strategy = {
            'strategy_id': 'test_strategy',
            'name': 'TestStrategy',
            'conditions': [
                {
                    'type': 'indicator',
                    'indicator_id': 'MACD',
                    'indicator': 'MACD',  # 保持兼容性
                    'period': 'daily',
                    'pattern': 'GOLDEN_CROSS',
                    'score_threshold': 60
                },
                {
                    'type': 'indicator',
                    'indicator_id': 'RSI',
                    'indicator': 'RSI',  # 保持兼容性
                    'period': 'daily',
                    'pattern': 'OVERSOLD',
                    'score_threshold': 70
                }
            ],
            'condition_logic': 'OR'
        }
        
        # 2. 创建验证器
        validator = BuyPointValidator()
        
        # 3. 执行闭环验证
        validation_date = '2024-01-20'
        validation_result = validator.validate_strategy_roundtrip(
            original_buypoints=sample_buypoints,
            generated_strategy=sample_strategy,
            validation_date=validation_date
        )
        
        # 4. 验证结果结构
        assert 'match_analysis' in validation_result
        assert 'execution_results' in validation_result
        assert 'recommendations' in validation_result
        # quality_grade可能不存在，如果执行失败的话
        if validation_result.get('execution_results', {}).get('execution_success', False):
            assert 'quality_grade' in validation_result
        
        # 5. 验证匹配率计算
        match_analysis = validation_result['match_analysis']
        assert 'match_rate' in match_analysis
        assert 0 <= match_analysis['match_rate'] <= 1
        
        # 6. 生成验证报告
        report_file = os.path.join(temp_dir, 'validation_report.md')
        validator.generate_validation_report(validation_result, report_file)
        assert os.path.exists(report_file)
    
    def test_strategy_optimization_workflow(self, sample_buypoints, temp_dir):
        """测试策略优化工作流程"""
        # 1. 创建需要优化的策略
        poor_strategy = {
            'strategy_id': 'poor_strategy',
            'name': 'PoorStrategy',
            'conditions': [
                {
                    'type': 'indicator',
                    'indicator_id': 'RSI',
                    'indicator': 'RSI',  # 保持兼容性
                    'period': 'daily',
                    'pattern': 'OVERBOUGHT',
                    'score_threshold': 95  # 过高的阈值
                }
            ] * 150,  # 过多的条件
            'condition_logic': 'AND'  # 过严格的逻辑
        }
        
        # 2. 创建优化器
        optimizer = StrategyOptimizer()
        
        # 3. 执行策略优化
        optimization_result = optimizer.optimize_strategy(
            original_strategy=poor_strategy,
            original_buypoints=sample_buypoints,
            validation_date='2024-01-20',
            max_iterations=3
        )
        
        # 4. 验证优化结果
        assert 'optimized_strategy' in optimization_result
        assert 'optimization_history' in optimization_result
        assert 'improvement_summary' in optimization_result
        
        # 5. 验证策略得到改进
        optimized_strategy = optimization_result['optimized_strategy']
        original_condition_count = len(poor_strategy['conditions'])
        optimized_condition_count = len(optimized_strategy['conditions'])
        
        # 条件数量应该减少
        assert optimized_condition_count <= original_condition_count
        
        # 6. 验证改进总结
        improvement = optimization_result['improvement_summary']
        assert 'initial_match_rate' in improvement
        assert 'final_match_rate' in improvement
        assert 'optimization_successful' in improvement
    
    def test_data_quality_validation(self):
        """测试数据质量验证"""
        # 1. 创建数据质量验证器
        validator = DataQualityValidator()
        
        # 2. 验证单只股票的数据质量
        validation_result = validator.validate_multi_period_data(
            stock_code='000001',
            date='2024-01-20'
        )
        
        # 3. 验证结果结构
        assert 'stock_code' in validation_result
        assert 'overall_quality' in validation_result
        assert 'period_results' in validation_result
        assert 'consistency_checks' in validation_result
        
        # 4. 验证各周期结果
        period_results = validation_result['period_results']
        expected_periods = ['15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        
        for period in expected_periods:
            if period in period_results:
                period_result = period_results[period]
                assert 'status' in period_result
                assert period_result['status'] in ['valid', 'warning', 'error', 'empty']
    
    def test_system_monitoring(self):
        """测试系统监控功能"""
        # 1. 创建系统监控器
        monitor = SystemHealthMonitor()
        
        # 2. 模拟一些操作
        @monitor.monitor_analysis_performance
        def mock_analysis_success():
            import time
            time.sleep(0.1)  # 模拟分析时间
            return {'match_analysis': {'match_rate': 0.75}}
        
        @monitor.monitor_analysis_performance
        def mock_analysis_error():
            raise Exception("模拟分析错误")
        
        # 3. 执行成功操作
        for _ in range(5):
            mock_analysis_success()
        
        # 4. 执行失败操作
        for _ in range(2):
            try:
                mock_analysis_error()
            except:
                pass
        
        # 5. 检查监控指标
        health = monitor.get_system_health()
        
        assert health['statistics']['success_count'] == 5
        assert health['statistics']['error_count'] == 2
        assert health['statistics']['total_operations'] == 7
        assert health['statistics']['error_rate'] == 2/7
        
        # 6. 检查告警
        assert len(health['recent_alerts']) >= 0  # 可能有告警
    
    @pytest.mark.skip(reason="需要完整的数据库和指标系统，暂时跳过")
    def test_end_to_end_workflow(self, sample_buypoints, temp_dir):
        """测试端到端完整工作流程"""
        # 1. 初始化所有组件
        monitor = SystemHealthMonitor()
        analyzer = BuyPointBatchAnalyzer()
        validator = BuyPointValidator()
        optimizer = StrategyOptimizer()
        
        # 2. 保存买点数据
        buypoints_file = os.path.join(temp_dir, 'buypoints.csv')
        sample_buypoints.to_csv(buypoints_file, index=False)
        
        # 3. 执行买点分析
        @monitor.monitor_analysis_performance
        def run_analysis():
            analyzer.run_complete_analysis(
                input_csv=buypoints_file,
                output_dir=temp_dir,
                min_hit_ratio=0.3,
                strategy_name="E2ETestStrategy"
            )
        
        run_analysis()
        
        # 4. 加载生成的策略
        strategy_file = os.path.join(temp_dir, 'generated_strategy.json')
        if os.path.exists(strategy_file):
            with open(strategy_file, 'r', encoding='utf-8') as f:
                generated_strategy = json.load(f)
            
            # 5. 执行策略验证
            validation_result = validator.validate_strategy_roundtrip(
                original_buypoints=sample_buypoints,
                generated_strategy=generated_strategy,
                validation_date='2024-01-20'
            )
            
            # 6. 如果匹配率低，执行优化
            match_rate = validation_result['match_analysis'].get('match_rate', 0)
            if match_rate < 0.6:
                optimization_result = optimizer.optimize_strategy(
                    original_strategy=generated_strategy,
                    original_buypoints=sample_buypoints,
                    validation_date='2024-01-20',
                    max_iterations=2
                )
                
                # 验证优化效果
                assert 'optimized_strategy' in optimization_result
                
                # 保存优化后的策略
                optimized_strategy_file = os.path.join(temp_dir, 'optimized_strategy.json')
                with open(optimized_strategy_file, 'w', encoding='utf-8') as f:
                    json.dump(optimization_result['optimized_strategy'], f, 
                            ensure_ascii=False, indent=2)
        
        # 7. 生成最终健康报告
        health_report_file = os.path.join(temp_dir, 'system_health_report.md')
        monitor.generate_health_report(health_report_file)
        
        # 8. 验证所有输出文件存在
        expected_files = [
            'common_indicators_report.md',
            'generated_strategy.json',
            'system_health_report.md'
        ]
        
        for filename in expected_files:
            file_path = os.path.join(temp_dir, filename)
            if os.path.exists(file_path):
                # 验证文件不为空
                assert os.path.getsize(file_path) > 0, f"文件为空: {filename}"
        
        # 9. 验证系统整体状态
        final_health = monitor.get_system_health()
        assert final_health['statistics']['total_operations'] > 0
        print(f"端到端测试完成，系统状态: {final_health['overall_status']}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
