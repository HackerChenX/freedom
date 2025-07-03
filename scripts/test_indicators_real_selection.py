#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指标真实数据选股测试脚本

直接使用指标验证框架测试所有88个指标是否都能在真实数据上选到个股
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.indicator_validation_framework import IndicatorValidationFramework
from enums.indicator_enum import IndicatorEnum
from utils.logger import get_logger

logger = get_logger(__name__)

class IndicatorRealSelectionTester:
    """指标真实数据选股测试器"""
    
    def __init__(self):
        self.framework = IndicatorValidationFramework()
        self.results = {}
        
    def test_single_indicator_selection(self, indicator_name: str) -> Dict[str, Any]:
        """测试单个指标的选股效果"""
        try:
            logger.info(f"测试指标选股: {indicator_name}")
            
            # 使用指标验证框架测试
            result = self.framework.validate_single_indicator(indicator_name=indicator_name)
            
            # 检查验证结果
            if result:
                status = result.get('status', 'unknown')
                selected_count = result.get('selected_count', 0)
                selection_ratio = result.get('selection_ratio', 0.0)
                
                # 判断是否成功选股
                can_select_stocks = (
                    status == 'success' and 
                    selected_count > 0 and 
                    selection_ratio > 0
                )
                
                return {
                    "indicator": indicator_name,
                    "status": status,
                    "can_select_stocks": can_select_stocks,
                    "selected_count": selected_count,
                    "selection_ratio": selection_ratio,
                    "selected_stocks": result.get('selected_stocks', []),
                    "test_details": {
                        "validation_success": status == 'success',
                        "execution_time": result.get('execution_time', 0),
                        "stock_pool_size": result.get('stock_pool_size', 0)
                    }
                }
            else:
                return {
                    "indicator": indicator_name,
                    "status": "error",
                    "can_select_stocks": False,
                    "selected_count": 0,
                    "selection_ratio": 0.0,
                    "selected_stocks": [],
                    "error": "验证结果为空",
                    "test_details": {
                        "validation_success": False,
                        "execution_time": 0,
                        "stock_pool_size": 0
                    }
                }
                
        except Exception as e:
            logger.error(f"测试指标 {indicator_name} 选股失败: {str(e)}")
            return {
                "indicator": indicator_name,
                "status": "exception",
                "can_select_stocks": False,
                "strategy_conditions_count": 0,
                "score_range": {},
                "error": str(e),
                "test_details": {
                    "calculation_success": False,
                    "scoring_success": False,
                    "strategy_generation_success": False
                }
            }
    
    def run_comprehensive_test(self, test_limit: int = None) -> Dict[str, Any]:
        """运行综合测试"""
        logger.info("开始指标真实数据选股测试")
        
        # 获取所有指标
        all_indicators = [indicator.name for indicator in IndicatorEnum]
        if test_limit:
            all_indicators = all_indicators[:test_limit]
            
        logger.info(f"准备测试 {len(all_indicators)} 个指标")
        
        # 测试每个指标
        test_results = []
        successful_indicators = []
        failed_indicators = []
        can_select_indicators = []
        cannot_select_indicators = []
        
        for i, indicator_name in enumerate(all_indicators, 1):
            logger.info(f"进度: {i}/{len(all_indicators)} - 测试指标 {indicator_name}")
            
            result = self.test_single_indicator_selection(indicator_name)
            test_results.append(result)
            
            if result["status"] == "success":
                successful_indicators.append(indicator_name)
                if result["can_select_stocks"]:
                    can_select_indicators.append(indicator_name)
                else:
                    cannot_select_indicators.append(indicator_name)
            else:
                failed_indicators.append(indicator_name)
                cannot_select_indicators.append(indicator_name)
                
        # 汇总结果
        summary = {
            "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_indicators": len(all_indicators),
            "successful_indicators": len(successful_indicators),
            "failed_indicators": len(failed_indicators),
            "can_select_stocks": len(can_select_indicators),
            "cannot_select_stocks": len(cannot_select_indicators),
            "calculation_success_rate": len(successful_indicators) / len(all_indicators) * 100,
            "selection_success_rate": len(can_select_indicators) / len(all_indicators) * 100,
            "successful_indicator_list": successful_indicators,
            "failed_indicator_list": failed_indicators,
            "can_select_indicator_list": can_select_indicators,
            "cannot_select_indicator_list": cannot_select_indicators,
            "detailed_results": test_results
        }
        
        logger.info(f"测试完成: {len(can_select_indicators)}/{len(all_indicators)} 个指标可以选股")
        return summary
    
    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """保存测试结果"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"results/indicator_real_selection_test_{timestamp}.json"
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存JSON结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
        logger.info(f"测试结果已保存到: {output_file}")
        
        # 生成简要报告
        report_file = output_file.replace('.json', '_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("指标真实数据选股测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {results['test_time']}\n")
            f.write(f"测试指标数量: {results['total_indicators']}\n")
            f.write(f"计算成功指标: {results['successful_indicators']}\n")
            f.write(f"计算失败指标: {results['failed_indicators']}\n")
            f.write(f"可选股指标: {results['can_select_stocks']}\n")
            f.write(f"不可选股指标: {results['cannot_select_stocks']}\n")
            f.write(f"计算成功率: {results['calculation_success_rate']:.2f}%\n")
            f.write(f"选股成功率: {results['selection_success_rate']:.2f}%\n\n")
            
            f.write("可以选股的指标:\n")
            for indicator in results['can_select_indicator_list']:
                f.write(f"  ✅ {indicator}\n")
                
            f.write("\n不能选股的指标:\n")
            for indicator in results['cannot_select_indicator_list']:
                f.write(f"  ❌ {indicator}\n")
                
        logger.info(f"测试报告已保存到: {report_file}")

def main():
    """主函数"""
    try:
        tester = IndicatorRealSelectionTester()
        
        # 先测试前10个指标
        print("开始测试前10个指标的选股能力...")
        results = tester.run_comprehensive_test(test_limit=10)
        
        # 保存结果
        tester.save_results(results)
        
        # 打印摘要
        print(f"\n{'='*60}")
        print("指标真实数据选股测试结果摘要")
        print(f"{'='*60}")
        print(f"测试时间: {results['test_time']}")
        print(f"测试指标数量: {results['total_indicators']}")
        print(f"计算成功指标: {results['successful_indicators']}")
        print(f"计算失败指标: {results['failed_indicators']}")
        print(f"可选股指标: {results['can_select_stocks']}")
        print(f"不可选股指标: {results['cannot_select_stocks']}")
        print(f"计算成功率: {results['calculation_success_rate']:.2f}%")
        print(f"选股成功率: {results['selection_success_rate']:.2f}%")
        
        if results['selection_success_rate'] == 100:
            print("🎉 所有测试指标都能成功选股!")
        else:
            print(f"⚠️  有 {results['cannot_select_stocks']} 个指标无法选股")
            print("无法选股的指标:")
            for indicator in results['cannot_select_indicator_list']:
                print(f"  ❌ {indicator}")
                
        # 询问是否继续测试所有指标
        if results['selection_success_rate'] > 80:
            print(f"\n前10个指标选股成功率较高({results['selection_success_rate']:.1f}%)，建议继续测试所有88个指标")
        else:
            print(f"\n前10个指标选股成功率较低({results['selection_success_rate']:.1f}%)，建议先修复问题再测试全部指标")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 