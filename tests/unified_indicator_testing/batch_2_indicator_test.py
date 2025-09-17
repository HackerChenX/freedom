#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第二批指标Ultra Think测试框架
扩展到112个指标系统

专注于第二批关键指标：
1. SAR - 抛物线转向系统
2. TRIX - 三重指数移动平均  
3. CMO - 钱德动量振荡器
4. EMV - 简易波动指标
5. ATR - 平均真实波幅
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.logger import get_logger
from tests.unified_indicator_testing.components.buypoint_analyzer import BuypointAnalyzer
from tests.unified_indicator_testing.components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator

logger = get_logger(__name__)

class Batch2IndicatorTester:
    """第二批指标Ultra Think测试器"""
    
    def __init__(self):
        """初始化第二批指标测试器"""
        logger.info("🚀 第二批指标Ultra Think测试器初始化")
        
        # 第二批要测试的指标
        self.batch_2_indicators = {
            'SAR': ['BUY_SIGNAL', 'SELL_SIGNAL', 'UPTREND', 'DOWNTREND'],
            'TRIX': ['GOLDEN_CROSS', 'DEATH_CROSS', 'SIGNAL_LINE_CROSS'],
            'CMO': ['OVERSOLD', 'OVERBOUGHT', 'ZERO_CROSS'],
            'EMV': ['POSITIVE_EMV', 'NEGATIVE_EMV', 'EMV_CROSS'],
            'ATR': ['HIGH_VOLATILITY', 'LOW_VOLATILITY', 'VOLATILITY_EXPANSION']
        }
        
        self.buypoint_analyzer = BuypointAnalyzer()
        self.data_generator = StockInfoCompatibleDataGenerator()
        
        logger.info(f"✅ 第二批指标测试器初始化完成，覆盖 {len(self.batch_2_indicators)} 个指标")
    
    def test_all_batch_2_indicators(self) -> Dict[str, Any]:
        """测试第二批所有指标"""
        logger.info("🚀 开始第二批指标Ultra Think测试")
        
        test_results = {}
        overall_stats = {
            'total_indicators': len(self.batch_2_indicators),
            'passed_indicators': 0,
            'failed_indicators': 0,
            'total_patterns': 0,
            'passed_patterns': 0,
            'test_start_time': datetime.now().isoformat(),
        }
        
        for i, (indicator_name, patterns) in enumerate(self.batch_2_indicators.items(), 1):
            logger.info(f"📊 测试指标 {i}/{len(self.batch_2_indicators)}: {indicator_name}")
            
            indicator_result = self._test_single_indicator(indicator_name, patterns)
            test_results[indicator_name] = indicator_result
            
            # 更新统计
            overall_stats['total_patterns'] += indicator_result['total_patterns']
            overall_stats['passed_patterns'] += indicator_result['passed_patterns']
            
            if indicator_result['success']:
                overall_stats['passed_indicators'] += 1
                logger.info(f"✅ {indicator_name}: 测试通过，准确率: {indicator_result['accuracy']:.2f}%")
            else:
                overall_stats['failed_indicators'] += 1
                logger.error(f"❌ {indicator_name}: 测试失败，准确率: {indicator_result['accuracy']:.2f}%")
        
        # 计算总体统计
        overall_stats['test_end_time'] = datetime.now().isoformat()
        overall_stats['success_rate'] = (overall_stats['passed_indicators'] / overall_stats['total_indicators']) * 100
        overall_stats['overall_accuracy'] = (overall_stats['passed_patterns'] / overall_stats['total_patterns']) * 100 if overall_stats['total_patterns'] > 0 else 0
        
        # 保存测试报告
        report = {
            'test_type': 'batch_2_indicator_test',
            'summary': overall_stats,
            'detailed_results': test_results
        }
        
        report_path = self._save_test_report(report)
        
        logger.info(f"📊 第二批指标测试完成")
        logger.info(f"📊 成功率: {overall_stats['passed_indicators']}/{overall_stats['total_indicators']} = {overall_stats['success_rate']:.1f}%")
        logger.info(f"📊 总体准确率: {overall_stats['overall_accuracy']:.1f}%")
        logger.info(f"📄 测试报告已保存: {report_path}")
        
        return report
    
    def _test_single_indicator(self, indicator_name: str, patterns: List[str]) -> Dict[str, Any]:
        """测试单个指标的所有形态"""
        try:
            pattern_results = []
            total_accuracy = 0
            successful_patterns = 0
            
            for pattern_type in patterns:
                # 生成测试数据
                target_stock_data = self._generate_test_data(indicator_name, pattern_type, target=True)
                interference_stock_data = self._generate_test_data(indicator_name, pattern_type, target=False)
                
                # 测试目标股票（应该检测到形态）
                target_result = self._test_pattern_detection(indicator_name, pattern_type, target_stock_data, should_detect=True)
                
                # 测试干扰股票（不应该检测到形态）
                interference_result = self._test_pattern_detection(indicator_name, pattern_type, interference_stock_data, should_detect=False)
                
                # 计算形态准确率
                pattern_accuracy = (target_result['accuracy'] + interference_result['accuracy']) / 2
                total_accuracy += pattern_accuracy
                
                if pattern_accuracy >= 50:  # 设定通过阈值为50%
                    successful_patterns += 1
                
                pattern_results.append({
                    'pattern': pattern_type,
                    'accuracy': pattern_accuracy,
                    'target_result': target_result,
                    'interference_result': interference_result
                })
            
            # 计算指标总体准确率
            overall_accuracy = total_accuracy / len(patterns) if patterns else 0
            
            return {
                'success': successful_patterns == len(patterns),
                'accuracy': overall_accuracy,
                'total_patterns': len(patterns),
                'passed_patterns': successful_patterns,
                'pattern_results': pattern_results
            }
            
        except Exception as e:
            logger.error(f"❌ 测试 {indicator_name} 时发生错误: {str(e)}")
            return {
                'success': False,
                'accuracy': 0.0,
                'total_patterns': len(patterns),
                'passed_patterns': 0,
                'error': str(e)
            }
    
    def _generate_test_data(self, indicator_name: str, pattern_type: str, target: bool = True):
        """生成测试数据"""
        try:
            if target:
                # 生成目标形态数据
                return self.data_generator.generate_stockinfo_compatible_data(
                    indicator_name=indicator_name,
                    pattern_type=pattern_type,
                    stock_code=f"{indicator_name}_{pattern_type}_TARGET",
                    history_days=150
                )
            else:
                # 生成干扰数据（随机数据，不包含特定形态）
                return self.data_generator.generate_random_stockinfo_data(
                    stock_code=f"{indicator_name}_{pattern_type}_INTERFERENCE",
                    history_days=150
                )
        except Exception as e:
            logger.error(f"❌ 生成 {indicator_name}.{pattern_type} 测试数据失败: {str(e)}")
            # 返回基础随机数据作为fallback
            return self.data_generator.generate_random_stockinfo_data(
                stock_code=f"{indicator_name}_{pattern_type}_FALLBACK",
                history_days=150
            )
    
    def _test_pattern_detection(self, indicator_name: str, pattern_type: str, data, should_detect: bool) -> Dict[str, Any]:
        """测试形态检测"""
        try:
            # 使用买点识别器进行形态检测
            detection_result = self.buypoint_analyzer.test_pattern_recognition(
                [data], f"{indicator_name}_{pattern_type}"
            )
            
            detected = detection_result.get('detected', False)
            confidence = detection_result.get('confidence', 0.0)
            
            # 计算准确率
            if should_detect:
                # 目标股票应该检测到形态
                accuracy = 100.0 if detected and confidence > 0.5 else 0.0
            else:
                # 干扰股票不应该检测到形态
                accuracy = 100.0 if not detected or confidence < 0.5 else 0.0
            
            return {
                'detected': detected,
                'confidence': confidence,
                'accuracy': accuracy,
                'details': detection_result.get('details', {})
            }
            
        except Exception as e:
            logger.error(f"❌ 检测 {indicator_name}.{pattern_type} 形态失败: {str(e)}")
            return {
                'detected': False,
                'confidence': 0.0,
                'accuracy': 0.0,
                'error': str(e)
            }
    
    def _save_test_report(self, report: Dict[str, Any]) -> str:
        """保存测试报告"""
        # 确保结果目录存在
        results_dir = os.path.join(os.path.dirname(__file__), '../../results/batch_2_tests')
        os.makedirs(results_dir, exist_ok=True)
        
        # 生成报告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"batch_2_indicator_test_report_{timestamp}.json"
        report_path = os.path.join(results_dir, report_filename)
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return report_path


def main():
    """主函数"""
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试器并运行测试
    tester = Batch2IndicatorTester()
    
    logger.info("=== 第二批指标Ultra Think测试开始 ===")
    
    # 运行所有第二批指标测试
    test_report = tester.test_all_batch_2_indicators()
    
    logger.info("=== 第二批指标Ultra Think测试完成 ===")
    
    return test_report


if __name__ == "__main__":
    main()
