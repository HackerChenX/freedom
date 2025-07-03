#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真实数据选股测试脚本

测试所有88个指标是否都能在真实股票数据上选到个股
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.clickhouse_db import get_clickhouse_db
from strategy.strategy_factory import StrategyFactory
from enums.indicator_enum import IndicatorEnum
from utils.logger import get_logger

logger = get_logger(__name__)

class RealStockSelectionTester:
    """真实数据选股测试器"""
    
    def __init__(self):
        self.db = get_clickhouse_db()
        self.strategy_factory = StrategyFactory()
        self.results = {}
        
    def get_test_stocks(self, limit: int = 50) -> List[str]:
        """获取测试用的股票列表"""
        try:
            # 获取有日线数据的股票
            stock_list = self.db.get_stock_list(limit=limit)
            if stock_list.empty:
                logger.warning("未找到股票数据")
                return []
                
            # 筛选有足够数据的股票
            valid_stocks = []
            for _, row in stock_list.iterrows():
                code = row['code']
                # 检查是否有足够的日线数据
                stock_data = self.db.get_stock_info(
                    stock_code=code,
                    level='日线',
                    limit=100
                )
                if stock_data.is_collection and len(stock_data) >= 50:
                    valid_stocks.append(code)
                    if len(valid_stocks) >= 20:  # 取前20只有足够数据的股票
                        break
                        
            logger.info(f"找到 {len(valid_stocks)} 只有足够数据的股票")
            return valid_stocks
            
        except Exception as e:
            logger.error(f"获取测试股票列表失败: {str(e)}")
            return []
    
    def test_single_indicator(self, indicator_name: str, test_stocks: List[str]) -> Dict[str, Any]:
        """测试单个指标的选股效果"""
        try:
            logger.info(f"测试指标: {indicator_name}")
            
            # 创建单指标策略
            strategy_config = {
                "name": f"test_{indicator_name}",
                "description": f"测试{indicator_name}指标",
                "conditions": [
                    {
                        "indicator": indicator_name,
                        "operator": ">",
                        "value": 60,
                        "weight": 1.0
                    }
                ]
            }
            
            strategy = self.strategy_factory.create_strategy(strategy_config)
            
            # 对每只股票进行选股测试
            selected_stocks = []
            error_stocks = []
            
            for stock_code in test_stocks:
                try:
                    # 获取股票数据
                    stock_data = self.db.get_stock_info(
                        stock_code=stock_code,
                        level='日线',
                        limit=100
                    )
                    
                    if not stock_data.is_collection or len(stock_data) < 50:
                        continue
                        
                    # 执行选股
                    result = strategy.select([stock_code])
                    
                    if result and len(result) > 0:
                        selected_stocks.extend(result)
                        
                except Exception as e:
                    error_stocks.append(stock_code)
                    logger.debug(f"股票 {stock_code} 测试失败: {str(e)}")
                    
            # 统计结果
            result = {
                "indicator": indicator_name,
                "total_tested": len(test_stocks),
                "selected_count": len(selected_stocks),
                "error_count": len(error_stocks),
                "success_rate": len(selected_stocks) / len(test_stocks) * 100 if test_stocks else 0,
                "selected_stocks": selected_stocks[:5],  # 只保存前5只
                "status": "success" if len(selected_stocks) > 0 else "no_selection"
            }
            
            logger.info(f"指标 {indicator_name} 测试完成: 选中 {len(selected_stocks)} 只股票")
            return result
            
        except Exception as e:
            logger.error(f"测试指标 {indicator_name} 失败: {str(e)}")
            return {
                "indicator": indicator_name,
                "total_tested": len(test_stocks),
                "selected_count": 0,
                "error_count": len(test_stocks),
                "success_rate": 0,
                "selected_stocks": [],
                "status": "error",
                "error": str(e)
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合测试"""
        logger.info("开始真实数据选股测试")
        
        # 获取测试股票
        test_stocks = self.get_test_stocks(limit=100)
        if not test_stocks:
            logger.error("无法获取测试股票数据")
            return {"error": "无法获取测试股票数据"}
            
        logger.info(f"使用 {len(test_stocks)} 只股票进行测试")
        
        # 获取所有指标
        all_indicators = [indicator.name for indicator in IndicatorEnum]
        logger.info(f"准备测试 {len(all_indicators)} 个指标")
        
        # 测试每个指标
        test_results = []
        successful_indicators = []
        failed_indicators = []
        
        for i, indicator_name in enumerate(all_indicators, 1):
            logger.info(f"进度: {i}/{len(all_indicators)} - 测试指标 {indicator_name}")
            
            result = self.test_single_indicator(indicator_name, test_stocks)
            test_results.append(result)
            
            if result["status"] == "success":
                successful_indicators.append(indicator_name)
            else:
                failed_indicators.append(indicator_name)
                
        # 汇总结果
        summary = {
            "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_indicators": len(all_indicators),
            "successful_indicators": len(successful_indicators),
            "failed_indicators": len(failed_indicators),
            "success_rate": len(successful_indicators) / len(all_indicators) * 100,
            "test_stocks_count": len(test_stocks),
            "test_stocks": test_stocks[:10],  # 只保存前10只
            "successful_indicator_list": successful_indicators,
            "failed_indicator_list": failed_indicators,
            "detailed_results": test_results
        }
        
        logger.info(f"测试完成: {len(successful_indicators)}/{len(all_indicators)} 个指标成功选股")
        return summary
    
    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """保存测试结果"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"results/real_stock_selection_test_{timestamp}.json"
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存JSON结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
        logger.info(f"测试结果已保存到: {output_file}")
        
        # 生成简要报告
        report_file = output_file.replace('.json', '_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("真实数据选股测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {results['test_time']}\n")
            f.write(f"测试股票数量: {results['test_stocks_count']}\n")
            f.write(f"测试指标数量: {results['total_indicators']}\n")
            f.write(f"成功选股指标: {results['successful_indicators']}\n")
            f.write(f"失败指标: {results['failed_indicators']}\n")
            f.write(f"成功率: {results['success_rate']:.2f}%\n\n")
            
            f.write("成功选股的指标:\n")
            for indicator in results['successful_indicator_list']:
                f.write(f"  ✅ {indicator}\n")
                
            f.write("\n未能选股的指标:\n")
            for indicator in results['failed_indicator_list']:
                f.write(f"  ❌ {indicator}\n")
                
        logger.info(f"测试报告已保存到: {report_file}")

def main():
    """主函数"""
    try:
        tester = RealStockSelectionTester()
        
        # 运行综合测试
        results = tester.run_comprehensive_test()
        
        if "error" in results:
            print(f"❌ 测试失败: {results['error']}")
            return
            
        # 保存结果
        tester.save_results(results)
        
        # 打印摘要
        print(f"\n{'='*60}")
        print("真实数据选股测试结果摘要")
        print(f"{'='*60}")
        print(f"测试时间: {results['test_time']}")
        print(f"测试股票数量: {results['test_stocks_count']}")
        print(f"测试指标数量: {results['total_indicators']}")
        print(f"成功选股指标: {results['successful_indicators']}")
        print(f"失败指标: {results['failed_indicators']}")
        print(f"成功率: {results['success_rate']:.2f}%")
        
        if results['success_rate'] == 100:
            print("🎉 所有指标都能成功选到股票!")
        else:
            print(f"⚠️  有 {results['failed_indicators']} 个指标未能选到股票")
            print("失败的指标:")
            for indicator in results['failed_indicator_list']:
                print(f"  ❌ {indicator}")
                
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 