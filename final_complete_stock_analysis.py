#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终完整股票分析验证
执行300005（探路者）和603359（东珠生态）的完整多周期买点分析
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bin.multi_period_buypoint_analyzer import MultiPeriodBuypointAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)

class FinalStockAnalysisTester:
    """最终股票分析测试器"""
    
    def __init__(self):
        self.test_stocks = ['300005', '603359']  # 探路者、东珠生态
        self.target_date = '2024-09-15'
        self.analyzer = None
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """运行完整的股票分析"""
        print("🎯 开始最终完整股票分析验证")
        print("=" * 60)
        
        results = {
            "test_type": "FINAL_COMPLETE_STOCK_ANALYSIS",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stocks_tested": self.test_stocks,
            "target_date": self.target_date,
            "analysis_results": {},
            "summary": {}
        }
        
        try:
            # 初始化分析器
            print("🔧 初始化多周期买点分析器")
            self.analyzer = MultiPeriodBuypointAnalyzer()
            print("  ✅ 分析器初始化成功")
            
            # 分析每只股票
            for stock_code in self.test_stocks:
                print(f"\n📊 分析股票: {stock_code}")
                stock_result = self.analyze_single_stock(stock_code)
                results["analysis_results"][stock_code] = stock_result
            
            # 计算总结
            self._calculate_summary(results)
            
        except Exception as e:
            results["error"] = str(e)
            print(f"❌ 分析失败: {e}")
        
        return results
    
    def analyze_single_stock(self, stock_code: str) -> Dict[str, Any]:
        """分析单只股票"""
        stock_result = {
            "stock_code": stock_code,
            "analysis_status": "PENDING",
            "data_coverage": {},
            "indicator_results": {},
            "buypoint_analysis": {},
            "strategy_signals": {},
            "issues": []
        }
        
        try:
            print(f"  🔍 开始分析 {stock_code}")
            
            # 1. 执行多周期买点分析
            analysis_result = self.analyzer.analyze_buypoint_signals(
                stock_code=stock_code,
                target_date=self.target_date
            )
            
            if analysis_result and "error" not in analysis_result:
                stock_result["analysis_status"] = "SUCCESS"
                
                # 2. 提取数据覆盖率信息
                if "data_coverage" in analysis_result:
                    stock_result["data_coverage"] = analysis_result["data_coverage"]
                    coverage_rate = analysis_result["data_coverage"].get("coverage_rate", 0)
                    print(f"    📊 数据覆盖率: {coverage_rate:.1%}")
                
                # 3. 提取指标计算结果
                if "indicator_results" in analysis_result:
                    stock_result["indicator_results"] = self._summarize_indicator_results(
                        analysis_result["indicator_results"]
                    )
                    success_rate = stock_result["indicator_results"].get("success_rate", 0)
                    print(f"    🔧 指标计算成功率: {success_rate:.1%}")
                
                # 4. 提取买点分析结果
                if "buypoint_analysis" in analysis_result:
                    stock_result["buypoint_analysis"] = analysis_result["buypoint_analysis"]
                    score = analysis_result["buypoint_analysis"].get("overall_score", 0)
                    print(f"    🎯 买点综合评分: {score:.1f}")
                
                # 5. 提取策略信号
                if "strategy_signals" in analysis_result:
                    stock_result["strategy_signals"] = self._summarize_strategy_signals(
                        analysis_result["strategy_signals"]
                    )
                    signal_count = len(stock_result["strategy_signals"].get("signals", []))
                    print(f"    📈 策略信号数量: {signal_count}")
                
                print(f"    ✅ {stock_code} 分析完成")
                
            else:
                stock_result["analysis_status"] = "FAILED"
                error_msg = analysis_result.get("error", "未知错误") if analysis_result else "分析结果为空"
                stock_result["issues"].append(f"分析失败: {error_msg}")
                print(f"    ❌ {stock_code} 分析失败: {error_msg}")
                
        except Exception as e:
            stock_result["analysis_status"] = "ERROR"
            stock_result["issues"].append(f"分析异常: {e}")
            print(f"    ❌ {stock_code} 分析异常: {e}")
        
        return stock_result
    
    def _summarize_indicator_results(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """总结指标计算结果"""
        summary = {
            "total_indicators": 0,
            "successful_indicators": 0,
            "failed_indicators": 0,
            "success_rate": 0.0,
            "period_coverage": {}
        }
        
        try:
            if "periods" in indicator_results:
                for period, period_data in indicator_results["periods"].items():
                    if "indicators" in period_data:
                        period_indicators = period_data["indicators"]
                        total = len(period_indicators)
                        successful = sum(1 for ind in period_indicators.values() 
                                       if ind.get("status") == "success")
                        
                        summary["period_coverage"][period] = {
                            "total": total,
                            "successful": successful,
                            "success_rate": successful / total if total > 0 else 0
                        }
                        
                        summary["total_indicators"] += total
                        summary["successful_indicators"] += successful
            
            summary["failed_indicators"] = summary["total_indicators"] - summary["successful_indicators"]
            summary["success_rate"] = (summary["successful_indicators"] / summary["total_indicators"] 
                                     if summary["total_indicators"] > 0 else 0)
            
        except Exception as e:
            summary["error"] = str(e)
        
        return summary
    
    def _summarize_strategy_signals(self, strategy_signals: Dict[str, Any]) -> Dict[str, Any]:
        """总结策略信号"""
        summary = {
            "total_strategies": 0,
            "active_signals": 0,
            "signals": [],
            "signal_strength": 0.0
        }
        
        try:
            if "strategies" in strategy_signals:
                strategies = strategy_signals["strategies"]
                summary["total_strategies"] = len(strategies)
                
                for strategy_name, strategy_data in strategies.items():
                    if strategy_data.get("signal") in ["BUY", "STRONG_BUY"]:
                        summary["active_signals"] += 1
                        summary["signals"].append({
                            "strategy": strategy_name,
                            "signal": strategy_data.get("signal"),
                            "strength": strategy_data.get("strength", 0),
                            "confidence": strategy_data.get("confidence", 0)
                        })
            
            # 计算平均信号强度
            if summary["signals"]:
                total_strength = sum(signal.get("strength", 0) for signal in summary["signals"])
                summary["signal_strength"] = total_strength / len(summary["signals"])
            
        except Exception as e:
            summary["error"] = str(e)
        
        return summary
    
    def _calculate_summary(self, results: Dict[str, Any]):
        """计算分析总结"""
        summary = results["summary"]
        
        # 统计分析状态
        total_stocks = len(results["analysis_results"])
        successful_analyses = sum(1 for result in results["analysis_results"].values() 
                                if result["analysis_status"] == "SUCCESS")
        
        summary["total_stocks"] = total_stocks
        summary["successful_analyses"] = successful_analyses
        summary["success_rate"] = successful_analyses / total_stocks if total_stocks > 0 else 0
        
        # 统计数据覆盖率
        coverage_rates = []
        indicator_success_rates = []
        buypoint_scores = []
        
        for stock_result in results["analysis_results"].values():
            if stock_result["analysis_status"] == "SUCCESS":
                # 数据覆盖率
                if "data_coverage" in stock_result and "coverage_rate" in stock_result["data_coverage"]:
                    coverage_rates.append(stock_result["data_coverage"]["coverage_rate"])
                
                # 指标成功率
                if "indicator_results" in stock_result and "success_rate" in stock_result["indicator_results"]:
                    indicator_success_rates.append(stock_result["indicator_results"]["success_rate"])
                
                # 买点评分
                if "buypoint_analysis" in stock_result and "overall_score" in stock_result["buypoint_analysis"]:
                    buypoint_scores.append(stock_result["buypoint_analysis"]["overall_score"])
        
        # 计算平均值
        summary["average_data_coverage"] = sum(coverage_rates) / len(coverage_rates) if coverage_rates else 0
        summary["average_indicator_success"] = sum(indicator_success_rates) / len(indicator_success_rates) if indicator_success_rates else 0
        summary["average_buypoint_score"] = sum(buypoint_scores) / len(buypoint_scores) if buypoint_scores else 0
        
        # 总体评估
        if summary["success_rate"] >= 1.0 and summary["average_data_coverage"] >= 0.8 and summary["average_indicator_success"] >= 0.8:
            summary["overall_status"] = "EXCELLENT"
        elif summary["success_rate"] >= 0.5 and summary["average_data_coverage"] >= 0.5:
            summary["overall_status"] = "GOOD"
        else:
            summary["overall_status"] = "NEEDS_IMPROVEMENT"

def main():
    """主函数"""
    print("🎯 最终完整股票分析验证")
    print("=" * 60)
    
    tester = FinalStockAnalysisTester()
    results = tester.run_complete_analysis()
    
    if "error" in results:
        print(f"❌ 分析失败: {results['error']}")
        return False
    
    # 显示分析结果
    print(f"\n📊 分析摘要:")
    summary = results["summary"]
    print(f"分析成功率: {summary['success_rate']:.1%} ({summary['successful_analyses']}/{summary['total_stocks']})")
    print(f"平均数据覆盖率: {summary['average_data_coverage']:.1%}")
    print(f"平均指标成功率: {summary['average_indicator_success']:.1%}")
    print(f"平均买点评分: {summary['average_buypoint_score']:.1f}")
    print(f"总体状态: {summary['overall_status']}")
    
    # 显示详细结果
    print(f"\n📋 详细分析结果:")
    for stock_code, stock_result in results["analysis_results"].items():
        status_icon = "✅" if stock_result["analysis_status"] == "SUCCESS" else "❌"
        print(f"  {status_icon} {stock_code}: {stock_result['analysis_status']}")
        
        if stock_result["analysis_status"] == "SUCCESS":
            if "data_coverage" in stock_result:
                coverage = stock_result["data_coverage"].get("coverage_rate", 0)
                print(f"    📊 数据覆盖率: {coverage:.1%}")
            
            if "indicator_results" in stock_result:
                success_rate = stock_result["indicator_results"].get("success_rate", 0)
                print(f"    🔧 指标成功率: {success_rate:.1%}")
            
            if "buypoint_analysis" in stock_result:
                score = stock_result["buypoint_analysis"].get("overall_score", 0)
                print(f"    🎯 买点评分: {score:.1f}")
            
            if "strategy_signals" in stock_result:
                signal_count = len(stock_result["strategy_signals"].get("signals", []))
                print(f"    📈 策略信号: {signal_count}个")
        
        if stock_result["issues"]:
            for issue in stock_result["issues"]:
                print(f"    ⚠️ {issue}")
    
    print(f"\n✅ 最终完整股票分析验证完成")
    
    # 判断是否达到验证标准
    verification_passed = (
        summary["success_rate"] >= 1.0 and  # 100%分析成功
        summary["average_data_coverage"] >= 0.8 and  # 80%数据覆盖
        summary["average_indicator_success"] >= 0.8  # 80%指标成功
    )
    
    if verification_passed:
        print("🎉 验证通过！系统达到生产级标准！")
    else:
        print("⚠️ 验证未完全通过，仍需改进")
    
    return verification_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
