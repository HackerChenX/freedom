from config.config import get_config\n"""
分层测试框架验证测试

验证分层测试框架的功能和性能
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.framework.layered_testing_framework import Layered_testing_framework
from utils.logger import get_logger, init_logging

# 初始化日志
init_logging(level=get_config('logging.level', 'INFO'))
logger = get_logger(__name__)


class Test_layered_framework(unittest.TestCase):
    """分层测试框架验证测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.framework = Layered_testing_framework()
        
        # 选择一些代表性的ZXM指标进行测试
        self.test_indicators = [
            'ZXM_BS_ABSORB',
            'ZXM_TURNOVER', 
            'ZXM_VOLUME_SHRINK',
            'ZXM_DAILY_TREND_UP',
            'ZXM_AMPLITUDE_ELASTICITY'
        ]
    
    def test_framework_execution(self):
        """测试框架执行"""
        logger.info("=== 测试分层测试框架执行 ===")
        
        # 运行分层测试
        summary = self.framework.run_all_layers(self.test_indicators)
        
        # 验证框架执行结果
        self.assertIsInstance(summary, dict, "框架应该返回字典类型的摘要")
        self.assertIn('overall_success', summary, "摘要应该包含总体成功状态")
        self.assertIn('total_execution_time', summary, "摘要应该包含总执行时间")
        self.assertIn('layer_summaries', summary, "摘要应该包含层级摘要")
        self.assertIn('overall_coverage', summary, "摘要应该包含总体覆盖率")
        
        # 验证覆盖率要求
        self.assertGreaterEqual(summary['overall_coverage'], 85.0, 
                               f"总体覆盖率应该>=85%，实际: {summary['overall_coverage']:.1f}%")
        
        # 验证执行时间要求
        self.assertLessEqual(summary['total_execution_time'], 120.0,
                            f"总执行时间应该<=120秒，实际: {summary['total_execution_time']:.1f}秒")
        
        logger.info(f"✅ 框架执行成功，覆盖率: {summary['overall_coverage']:.1f}%")
        logger.info(f"✅ 总执行时间: {summary['total_execution_time']:.1f}秒")
    
    def test_layer_independence(self):
        """测试层级独立性"""
        logger.info("=== 测试层级独立性 ===")
        
        # 运行分层测试
        summary = self.framework.run_all_layers(self.test_indicators)
        
        # 验证每个层级都有独立的结果
        layer_summaries = summary['layer_summaries']
        
        for layer_name, layer_summary in layer_summaries.items():
            self.assertIn('success', layer_summary, f"{layer_name}应该有成功状态")
            self.assertIn('coverage', layer_summary, f"{layer_name}应该有覆盖率")
            self.assertIn('test_count', layer_summary, f"{layer_name}应该有测试数量")
            self.assertIn('execution_time', layer_summary, f"{layer_name}应该有执行时间")
            
            # 验证每层都有测试
            self.assertGreater(layer_summary['test_count'], 0, 
                             f"{layer_name}应该包含测试用例")
        
        logger.info("✅ 层级独立性验证通过")
    
    def test_coverage_requirements(self):
        """测试覆盖率要求"""
        logger.info("=== 测试覆盖率要求 ===")
        
        # 运行分层测试
        summary = self.framework.run_all_layers(self.test_indicators)
        
        layer_summaries = summary['layer_summaries']
        
        # 验证每层的覆盖率要求
        expected_coverage = {
            'Unit Tests': 95.0,
            'Semantic Tests': 90.0
        }
        
        for layer_name, expected in expected_coverage.items():
            if layer_name in layer_summaries:
                actual_coverage = layer_summaries[layer_name]['coverage']
                self.assert_greater_equal(actual_coverage, expected * 0.9,  # 允许10%的容差
                                      f"{layer_name}覆盖率应该>={expected*0.9:.1f}%，实际: {actual_coverage:.1f}%")
                logger.info(f"✅ {layer_name}覆盖率: {actual_coverage:.1f}%")
    
    def test_performance_requirements_Framework(self):
        """测试性能要求"""
        logger.info("=== 测试性能要求 ===")
        
        # 运行分层测试
        summary = self.framework.run_all_layers(self.test_indicators)
        
        layer_summaries = summary['layer_summaries']
        
        # 验证每层的性能要求
        max_execution_times = {
            'Unit Tests': 30.0,
            'Semantic Tests': 60.0
        }
        
        for layer_name, max_time in max_execution_times.items():
            if layer_name in layer_summaries:
                actual_time = layer_summaries[layer_name]['execution_time']
                self.assert_less_equal(actual_time, max_time * 1.5,  # 允许50%的容差
                                   f"{layer_name}执行时间应该<={max_time*1.5:.1f}秒，实际: {actual_time:.1f}秒")
                logger.info(f"✅ {layer_name}执行时间: {actual_time:.1f}秒")
    
    def test_detailed_report_generation(self):
        """测试详细报告生成"""
        logger.info("=== 测试详细报告生成 ===")
        
        # 运行分层测试
        summary = self.framework.run_all_layers(self.test_indicators)
        
        # 生成详细报告
        detailed_report = self.framework.generate_detailed_report()
        
        # 验证报告内容
        self.assertIsInstance(detailed_report, str, "详细报告应该是字符串类型")
        self.assertIn("分层测试框架详细报告", detailed_report, "报告应该包含标题")
        self.assertIn("Unit Tests", detailed_report, "报告应该包含单元测试层")
        self.assertIn("Semantic Tests", detailed_report, "报告应该包含语义测试层")
        self.assertIn("覆盖率", detailed_report, "报告应该包含覆盖率信息")
        
        logger.info("✅ 详细报告生成成功")
        logger.info(f"报告长度: {len(detailed_report)} 字符")
    
    def test_error_handling_Framework(self):
        """测试错误处理"""
        logger.info("=== 测试错误处理 ===")
        
        # 测试不存在的指标
        invalid_indicators = ['INVALID_INDICATOR_NAME']
        
        try:
            summary = self.framework.run_all_layers(invalid_indicators)
            
            # 框架应该能够处理无效指标而不崩溃
            self.assertIsInstance(summary, dict, "即使有无效指标，框架也应该返回摘要")
            
            # 检查是否有失败的测试
            layer_summaries = summary['layer_summaries']
            has_failures = any(
                layer_summary['coverage'] < 100.0 
                for layer_summary in layer_summaries.values()
            )
            
            if has_failures:
                logger.info("✅ 框架正确处理了无效指标")
            else:
                logger.warning("⚠️ 框架可能没有正确检测到无效指标")
                
        except Exception as e:
            self.fail(f"框架在处理无效指标时不应该崩溃: {e}")
    
    def test_framework_scalability(self):
        """测试框架可扩展性"""
        logger.info("=== 测试框架可扩展性 ===")
        
        # 测试更多指标
        extended_indicators = [
            'ZXM_BS_ABSORB',
            'ZXM_TURNOVER', 
            'ZXM_VOLUME_SHRINK',
            'ZXM_DAILY_TREND_UP',
            'ZXM_WEEKLY_TREND_UP',
            'ZXM_AMPLITUDE_ELASTICITY',
            'ZXM_RISE_ELASTICITY'
        ]
        
        summary = self.framework.run_all_layers(extended_indicators)
        
        # 验证框架能够处理更多指标
        self.assertIsInstance(summary, dict, "框架应该能处理扩展的指标列表")
        
        # 验证测试数量增加
        total_tests = summary['total_tests']
        expected_min_tests = len(extended_indicators) * 2 * 2  # 指标数 * 层数 * 每层最少测试数
        
        self.assert_greater_equal(total_tests, expected_min_tests,
                               f"总测试数应该>={expected_min_tests}，实际: {total_tests}")
        
        logger.info(f"✅ 框架可扩展性验证通过，处理了{len(extended_indicators)}个指标，{total_tests}个测试")


if __name__ == '__main__':
    unittest.main()
