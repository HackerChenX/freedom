from config.unified_config_manager import get_config\n"""
自动化风险检测机制测试

验证自动化风险检测工具的功能和准确性
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.automated_risk_detection import Automated_risk_detector, Indicator_type, Risk_level
from utils.logger import get_logger, init_logging

# 初始化日志
init_logging(level=get_config('logging.level', 'INFO'))
logger = get_logger(__name__)


class Test_automated_risk_detection(unittest.TestCase):
    """自动化风险检测测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = Automated_risk_detector()
    
    def test_risk_detector_initialization(self):
        """测试风险检测器初始化"""
        logger.info("=== 测试风险检测器初始化 ===")
        
        self.assertIsNotNone(self.detector.analyzer, "分析器应该被正确初始化")
        self.assertIsInstance(self.detector.detection_results, dict, "检测结果应该是字典类型")
        
        logger.info("✅ 风险检测器初始化成功")
    
    def test_scan_zxm_indicators(self):
        """测试扫描ZXM指标"""
        logger.info("=== 测试扫描ZXM指标 ===")
        
        # 运行风险扫描
        results = self.detector.scan_all_indicators()
        
        # 验证扫描结果
        self.assertIsInstance(results, dict, "扫描结果应该是字典类型")
        self.assertGreater(len(results), 0, "应该扫描到指标")
        
        # 验证ZXM指标被正确识别
        zxm_indicators = [name for name in results.keys() if name.startswith('ZXM_')]
        self.assertGreater(len(zxm_indicators), 0, "应该识别到ZXM指标")
        
        logger.info(f"✅ 成功扫描 {len(results)} 个指标，其中 {len(zxm_indicators)} 个ZXM指标")
    
    def test_high_risk_indicator_detection(self):
        """测试高风险指标检测"""
        logger.info("=== 测试高风险指标检测 ===")
        
        # 运行风险扫描
        results = self.detector.scan_all_indicators()
        
        # 获取高风险指标
        high_risk_indicators = self.detector.get_high_risk_indicators()
        
        # 验证高风险指标检测
        self.assertIsInstance(high_risk_indicators, list, "高风险指标应该是列表类型")
        
        # 检查已知的高风险指标是否被正确识别
        known_high_risk = ['ZXM_BS_ABSORB', 'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK']
        detected_known_risks = [name for name in known_high_risk if name in results]
        
        if detected_known_risks:
            logger.info(f"✅ 正确识别已知高风险指标: {detected_known_risks}")
        
        # 验证风险等级分类
        risk_levels = {}
        for name, result in results.items():
            risk_level = result.risk_level.value
            if risk_level not in risk_levels:
                risk_levels[risk_level] = 0
            risk_levels[risk_level] += 1
        
        logger.info(f"风险等级分布: {risk_levels}")
        
        # 应该有不同的风险等级
        self.assertGreaterEqual(len(risk_levels), 2, "应该有多个风险等级")
    
    def test_indicator_type_classification(self):
        """测试指标类型分类"""
        logger.info("=== 测试指标类型分类 ===")
        
        # 运行风险扫描
        results = self.detector.scan_all_indicators()
        
        # 统计指标类型
        type_counts = {}
        for name, result in results.items():
            indicator_type = result.indicator_type.value
            if indicator_type not in type_counts:
                type_counts[indicator_type] = 0
            type_counts[indicator_type] += 1
        
        logger.info(f"指标类型分布: {type_counts}")
        
        # 验证类型分类
        self.assertGreater(len(type_counts), 1, "应该识别出多种指标类型")
        
        # 检查特定指标的类型分类
        specific_checks = {
            'ZXM_BS_ABSORB': IndicatorType.COUNT_TYPE,
            'ZXM_TURNOVER': IndicatorType.STATE_TYPE,
            'ZXM_VOLUME_SHRINK': IndicatorType.STATE_TYPE
        }
        
        for indicator_name, expected_type in specific_checks.items():
            if indicator_name in results:
                actual_type = results[indicator_name].indicator_type
                logger.info(f"指标 {indicator_name}: 期望类型 {expected_type.value}, 实际类型 {actual_type.value}")
    
    def test_signal_consistency_assessment(self):
        """测试信号一致性评估"""
        logger.info("=== 测试信号一致性评估 ===")
        
        # 运行风险扫描
        results = self.detector.scan_all_indicators()
        
        # 验证信号一致性评分
        for name, result in results.items():
            self.assert_is_instance(result.signal_consistency_score, float, 
                                f"{name}的信号一致性评分应该是浮点数")
            self.assert_greater_equal(result.signal_consistency_score, 0.0, 
                                  f"{name}的信号一致性评分应该>=0")
            self.assert_less_equal(result.signal_consistency_score, 100.0, 
                               f"{name}的信号一致性评分应该<=100")
        
        # 计算平均信号一致性评分
        avg_score = sum(r.signal_consistency_score for r in results.values()) / len(results)
        logger.info(f"平均信号一致性评分: {avg_score:.1f}")
        
        # 验证评分合理性
        self.assertGreater(avg_score, 50.0, "平均信号一致性评分应该>50")
    
    def test_risk_factor_identification(self):
        """测试风险因素识别"""
        logger.info("=== 测试风险因素识别 ===")
        
        # 运行风险扫描
        results = self.detector.scan_all_indicators()
        
        # 验证风险因素识别
        total_risk_factors = 0
        for name, result in results.items():
            self.assert_is_instance(result.risk_factors, list, 
                                f"{name}的风险因素应该是列表类型")
            total_risk_factors += len(result.risk_factors)
        
        logger.info(f"总共识别出 {total_risk_factors} 个风险因素")
        
        # 检查高风险指标是否有风险因素
        high_risk_indicators = [name for name, result in results.items() 
                               if result.risk_level == Risk_level.HIGH]
        
        for indicator_name in high_risk_indicators:
            result = results[indicator_name]
            if len(result.risk_factors) > 0:
                logger.info(f"高风险指标 {indicator_name} 的风险因素: {result.risk_factors}")
    
    def test_recommendation_generation(self):
        """测试建议生成"""
        logger.info("=== 测试建议生成 ===")
        
        # 运行风险扫描
        results = self.detector.scan_all_indicators()
        
        # 获取建议摘要
        recommendations_summary = self.detector.get_recommendations_summary()
        
        # 验证建议生成
        self.assertIsInstance(recommendations_summary, dict, "建议摘要应该是字典类型")
        
        total_recommendations = sum(len(recs) for recs in recommendations_summary.values())
        logger.info(f"总共生成 {total_recommendations} 条建议")
        
        # 验证高风险指标有建议
        high_risk_indicators = self.detector.get_high_risk_indicators()
        high_risk_with_recommendations = [name for name in high_risk_indicators 
                                        if name in recommendations_summary]
        
        if high_risk_with_recommendations:
            logger.info(f"有建议的高风险指标: {high_risk_with_recommendations}")
    
    def test_risk_report_generation(self):
        """测试风险报告生成"""
        logger.info("=== 测试风险报告生成 ===")
        
        # 运行风险扫描
        results = self.detector.scan_all_indicators()
        
        # 生成风险报告
        risk_report = self.detector.generate_risk_report()
        
        # 验证报告内容
        self.assertIsInstance(risk_report, str, "风险报告应该是字符串类型")
        self.assertIn("自动化风险检测报告", risk_report, "报告应该包含标题")
        self.assertIn("总体统计", risk_report, "报告应该包含统计信息")
        self.assertIn("总指标数", risk_report, "报告应该包含指标数量")
        
        # 验证报告长度
        self.assertGreater(len(risk_report), 100, "报告应该有足够的内容")
        
        logger.info(f"✅ 风险报告生成成功，长度: {len(risk_report)} 字符")
    
    def test_detection_accuracy(self):
        """测试检测准确性"""
        logger.info("=== 测试检测准确性 ===")
        
        # 运行风险扫描
        results = self.detector.scan_all_indicators()
        
        # 计算检测准确率
        accuracy = self.detector._calculate_accuracy()
        
        # 验证准确率
        self.assertIsInstance(accuracy, float, "准确率应该是浮点数")
        self.assertGreaterEqual(accuracy, 0.0, "准确率应该>=0")
        self.assertLessEqual(accuracy, 100.0, "准确率应该<=100")
        
        # 验证准确率达到要求
        self.assertGreaterEqual(accuracy, 80.0, f"检测准确率应该>=80%，实际: {accuracy:.1f}%")
        
        logger.info(f"✅ 检测准确率: {accuracy:.1f}%")
    
    def test_performance_requirements(self):
        """测试性能要求"""
        logger.info("=== 测试性能要求 ===")
        
        import time
        
        # 测试扫描性能
        start_time = time.time()
        results = self.detector.scan_all_indicators()
        scan_time = time.time() - start_time
        
        # 验证扫描时间
        self.assertLess(scan_time, 30.0, f"扫描时间应该<30秒，实际: {scan_time:.1f}秒")
        
        # 测试报告生成性能
        start_time = time.time()
        risk_report = self.detector.generate_risk_report()
        report_time = time.time() - start_time
        
        # 验证报告生成时间
        self.assertLess(report_time, 5.0, f"报告生成时间应该<5秒，实际: {report_time:.1f}秒")
        
        logger.info(f"✅ 性能测试通过 - 扫描: {scan_time:.1f}s, 报告: {report_time:.1f}s")


if __name__ == '__main__':
    unittest.main()
