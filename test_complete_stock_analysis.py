#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整股票分析验证测试
执行300005（探路者）和603359（东珠生态）的多周期买点分析全流程
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bin.multi_period_buypoint_analyzer import MultiPeriodBuypointAnalyzer
from db.services.multi_period_data_service import Period
from utils.logger import get_logger

logger = get_logger(__name__)

class CompleteStockAnalysisValidator:
    """完整股票分析验证器"""
    
    def __init__(self):
        self.analyzer = None
        self.test_stocks = [
            {"code": "300005", "name": "探路者"},
            {"code": "603359", "name": "东珠生态"}
        ]
        self.target_date = "2024-12-20"  # 使用最近的交易日
        self.analysis_results = {}
        
    def run_complete_analysis(self) -> Dict[str, Any]:
        """运行完整分析流程"""
        logger.info("🚀 开始完整股票分析验证")
        
        results = {
            "analysis_type": "COMPLETE_STOCK_ANALYSIS",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_date": self.target_date,
            "test_stocks": self.test_stocks,
            "phases": {},
            "summary": {
                "total_phases": 0,
                "successful_phases": 0,
                "failed_phases": 0,
                "critical_issues": [],
                "analysis_results": {}
            }
        }
        
        try:
            # 阶段1：系统初始化验证
            results["phases"]["phase1_initialization"] = self.phase1_system_initialization()
            
            # 阶段2：数据获取验证
            results["phases"]["phase2_data_validation"] = self.phase2_data_validation()
            
            # 阶段3：指标计算验证
            results["phases"]["phase3_indicator_calculation"] = self.phase3_indicator_calculation()
            
            # 阶段4：买点分析执行
            results["phases"]["phase4_buypoint_analysis"] = self.phase4_buypoint_analysis()
            
            # 阶段5：策略信号验证
            results["phases"]["phase5_strategy_validation"] = self.phase5_strategy_validation()
            
            # 阶段6：双向验证一致性
            results["phases"]["phase6_consistency_validation"] = self.phase6_consistency_validation()
            
            # 计算总结
            self._calculate_summary(results)
            
        except Exception as e:
            logger.error(f"完整分析流程异常: {e}")
            results["summary"]["critical_issues"].append(f"流程异常: {e}")
        
        return results
    
    def phase1_system_initialization(self) -> Dict[str, Any]:
        """阶段1：系统初始化验证"""
        logger.info("📋 阶段1：系统初始化验证")
        
        phase_result = {
            "phase_name": "系统初始化验证",
            "success": True,
            "start_time": time.time(),
            "details": {},
            "issues": []
        }
        
        try:
            # 初始化分析器
            self.analyzer = MultiPeriodBuypointAnalyzer()
            
            # 验证核心组件
            components = {
                "data_service": hasattr(self.analyzer, 'data_service') and self.analyzer.data_service is not None,
                "indicator_service": hasattr(self.analyzer, 'indicator_service') and self.analyzer.indicator_service is not None,
                "universal_calculator": hasattr(self.analyzer, 'universal_calculator') and self.analyzer.universal_calculator is not None,
                "buypoint_detector": hasattr(self.analyzer, 'buypoint_detector') and self.analyzer.buypoint_detector is not None,
                "indicator_registry": hasattr(self.analyzer, 'indicator_registry') and self.analyzer.indicator_registry is not None
            }
            
            # 验证配置和权重
            config_status = {
                "config_loaded": hasattr(self.analyzer, 'config') and bool(self.analyzer.config),
                "indicator_weights": hasattr(self.analyzer, 'indicator_weights') and len(self.analyzer.indicator_weights) > 0,
                "builtin_strategies": hasattr(self.analyzer, 'builtin_strategies') and len(self.analyzer.builtin_strategies) > 0,
                "all_indicators": hasattr(self.analyzer, 'all_indicators') and len(self.analyzer.all_indicators) > 0
            }
            
            phase_result["details"] = {
                "components": components,
                "config_status": config_status,
                "indicator_count": len(self.analyzer.all_indicators) if hasattr(self.analyzer, 'all_indicators') else 0,
                "strategy_count": len(self.analyzer.builtin_strategies) if hasattr(self.analyzer, 'builtin_strategies') else 0,
                "weight_count": len(self.analyzer.indicator_weights) if hasattr(self.analyzer, 'indicator_weights') else 0
            }
            
            # 检查失败项
            failed_components = [k for k, v in components.items() if not v]
            failed_configs = [k for k, v in config_status.items() if not v]
            
            if failed_components:
                phase_result["issues"].append(f"组件初始化失败: {failed_components}")
                phase_result["success"] = False
            
            if failed_configs:
                phase_result["issues"].append(f"配置加载失败: {failed_configs}")
                phase_result["success"] = False
            
            logger.info(f"系统初始化完成 - 指标:{phase_result['details']['indicator_count']}, 策略:{phase_result['details']['strategy_count']}")
            
        except Exception as e:
            phase_result["success"] = False
            phase_result["issues"].append(f"初始化异常: {e}")
            logger.error(f"系统初始化失败: {e}")
        
        phase_result["duration"] = time.time() - phase_result["start_time"]
        return phase_result
    
    def phase2_data_validation(self) -> Dict[str, Any]:
        """阶段2：数据获取验证"""
        logger.info("📊 阶段2：数据获取验证")
        
        phase_result = {
            "phase_name": "数据获取验证",
            "success": True,
            "start_time": time.time(),
            "details": {},
            "issues": []
        }
        
        try:
            data_coverage = {}
            
            for stock in self.test_stocks:
                stock_code = stock["code"]
                stock_name = stock["name"]
                
                logger.info(f"验证股票 {stock_code}({stock_name}) 数据覆盖率")
                
                # 获取多周期数据
                multi_period_data = self.analyzer.data_service.get_stock_multi_period_data(
                    stock_code=stock_code,
                    target_date=self.target_date,
                    periods=None,  # 获取全部周期
                    lookback_days=None
                )
                
                # 分析数据覆盖情况
                period_coverage = {}
                total_periods = 6
                available_periods = 0
                
                for period_name, period_data in multi_period_data.items():
                    if period_data is not None and not period_data.empty:
                        period_coverage[period_name] = {
                            "available": True,
                            "record_count": len(period_data),
                            "date_range": f"{period_data['date'].min()} ~ {period_data['date'].max()}"
                        }
                        available_periods += 1
                    else:
                        period_coverage[period_name] = {
                            "available": False,
                            "record_count": 0,
                            "date_range": "无数据"
                        }
                
                coverage_rate = (available_periods / total_periods) * 100
                
                data_coverage[stock_code] = {
                    "stock_name": stock_name,
                    "period_coverage": period_coverage,
                    "coverage_rate": f"{coverage_rate:.1f}%",
                    "available_periods": available_periods,
                    "total_periods": total_periods
                }
                
                logger.info(f"{stock_code} 数据覆盖率: {coverage_rate:.1f}% ({available_periods}/{total_periods})")
                
                # 检查数据质量
                if coverage_rate < 80:
                    phase_result["issues"].append(f"{stock_code} 数据覆盖率不足: {coverage_rate:.1f}%")
                    phase_result["success"] = False
            
            phase_result["details"]["data_coverage"] = data_coverage
            
        except Exception as e:
            phase_result["success"] = False
            phase_result["issues"].append(f"数据验证异常: {e}")
            logger.error(f"数据获取验证失败: {e}")
        
        phase_result["duration"] = time.time() - phase_result["start_time"]
        return phase_result
    
    def phase3_indicator_calculation(self) -> Dict[str, Any]:
        """阶段3：指标计算验证"""
        logger.info("🔢 阶段3：指标计算验证")
        
        phase_result = {
            "phase_name": "指标计算验证",
            "success": True,
            "start_time": time.time(),
            "details": {},
            "issues": []
        }
        
        try:
            indicator_results = {}
            
            for stock in self.test_stocks:
                stock_code = stock["code"]
                stock_name = stock["name"]
                
                logger.info(f"验证股票 {stock_code}({stock_name}) 指标计算")
                
                # 执行多周期指标计算
                multi_period_result = self.analyzer.universal_calculator.calculate_multi_period_indicators(
                    stock_code=stock_code,
                    target_date=self.target_date,
                    periods=None  # 全周期计算
                )
                
                # 分析指标计算结果
                if "analysis_matrix" in multi_period_result:
                    analysis_matrix = multi_period_result["analysis_matrix"]
                    
                    period_indicator_stats = {}
                    total_indicators = 0
                    successful_indicators = 0
                    
                    for period_name, indicators in analysis_matrix.items():
                        period_stats = {
                            "total_indicators": len(indicators),
                            "successful_indicators": 0,
                            "failed_indicators": 0,
                            "success_rate": "0%"
                        }
                        
                        for indicator_name, indicator_result in indicators.items():
                            total_indicators += 1
                            if indicator_result and indicator_result.get("status") == "SUCCESS":
                                period_stats["successful_indicators"] += 1
                                successful_indicators += 1
                            else:
                                period_stats["failed_indicators"] += 1
                        
                        if period_stats["total_indicators"] > 0:
                            success_rate = (period_stats["successful_indicators"] / period_stats["total_indicators"]) * 100
                            period_stats["success_rate"] = f"{success_rate:.1f}%"
                        
                        period_indicator_stats[period_name] = period_stats
                    
                    overall_success_rate = (successful_indicators / total_indicators * 100) if total_indicators > 0 else 0
                    
                    indicator_results[stock_code] = {
                        "stock_name": stock_name,
                        "period_stats": period_indicator_stats,
                        "overall_success_rate": f"{overall_success_rate:.1f}%",
                        "total_calculations": total_indicators,
                        "successful_calculations": successful_indicators
                    }
                    
                    logger.info(f"{stock_code} 指标计算成功率: {overall_success_rate:.1f}% ({successful_indicators}/{total_indicators})")
                    
                    # 检查计算质量
                    if overall_success_rate < 95:
                        phase_result["issues"].append(f"{stock_code} 指标计算成功率不足: {overall_success_rate:.1f}%")
                        phase_result["success"] = False
                
                else:
                    phase_result["issues"].append(f"{stock_code} 未获得分析矩阵结果")
                    phase_result["success"] = False
            
            phase_result["details"]["indicator_results"] = indicator_results
            
        except Exception as e:
            phase_result["success"] = False
            phase_result["issues"].append(f"指标计算异常: {e}")
            logger.error(f"指标计算验证失败: {e}")
        
        phase_result["duration"] = time.time() - phase_result["start_time"]
        return phase_result
    
    def phase4_buypoint_analysis(self) -> Dict[str, Any]:
        """阶段4：买点分析执行"""
        logger.info("🎯 阶段4：买点分析执行")
        
        phase_result = {
            "phase_name": "买点分析执行",
            "success": True,
            "start_time": time.time(),
            "details": {},
            "issues": []
        }
        
        try:
            buypoint_results = {}
            
            for stock in self.test_stocks:
                stock_code = stock["code"]
                stock_name = stock["name"]
                
                logger.info(f"执行股票 {stock_code}({stock_name}) 买点分析")
                
                # 执行买点分析
                analysis_result = self.analyzer.analyze_multi_period_buypoint(
                    stock_code=stock_code,
                    target_date=self.target_date
                )
                
                # 保存分析结果供后续验证使用
                self.analysis_results[stock_code] = analysis_result
                
                # 分析买点结果
                if analysis_result.get("status") == "SUCCESS":
                    buypoint_analysis = analysis_result.get("buypoint_analysis", {})
                    overall_score = analysis_result.get("overall_score", 0)
                    
                    buypoint_results[stock_code] = {
                        "stock_name": stock_name,
                        "analysis_status": "SUCCESS",
                        "overall_score": overall_score,
                        "buypoint_signals": buypoint_analysis.get("buypoint_signals", {}),
                        "period_comparison": buypoint_analysis.get("period_comparison", {}),
                        "recommendations": analysis_result.get("recommendations", {})
                    }
                    
                    logger.info(f"{stock_code} 买点分析完成 - 综合评分: {overall_score}")
                    
                else:
                    buypoint_results[stock_code] = {
                        "stock_name": stock_name,
                        "analysis_status": "FAILED",
                        "error_message": analysis_result.get("error", "未知错误")
                    }
                    
                    phase_result["issues"].append(f"{stock_code} 买点分析失败: {analysis_result.get('error', '未知错误')}")
                    phase_result["success"] = False
            
            phase_result["details"]["buypoint_results"] = buypoint_results
            
        except Exception as e:
            phase_result["success"] = False
            phase_result["issues"].append(f"买点分析异常: {e}")
            logger.error(f"买点分析执行失败: {e}")
        
        phase_result["duration"] = time.time() - phase_result["start_time"]
        return phase_result
    
    def phase5_strategy_validation(self) -> Dict[str, Any]:
        """阶段5：策略信号验证"""
        logger.info("📈 阶段5：策略信号验证")
        
        phase_result = {
            "phase_name": "策略信号验证",
            "success": True,
            "start_time": time.time(),
            "details": {},
            "issues": []
        }
        
        try:
            strategy_results = {}
            
            for stock in self.test_stocks:
                stock_code = stock["code"]
                stock_name = stock["name"]
                
                if stock_code in self.analysis_results:
                    analysis_result = self.analysis_results[stock_code]
                    
                    if analysis_result.get("status") == "SUCCESS":
                        strategy_analysis = analysis_result.get("strategy_analysis", {})
                        
                        if strategy_analysis:
                            strategy_signals = strategy_analysis.get("strategy_signals", {})
                            
                            strategy_results[stock_code] = {
                                "stock_name": stock_name,
                                "strategy_count": len(strategy_signals),
                                "strategy_signals": strategy_signals,
                                "strategy_summary": strategy_analysis.get("strategy_summary", {})
                            }
                            
                            logger.info(f"{stock_code} 策略信号验证完成 - 策略数量: {len(strategy_signals)}")
                            
                        else:
                            phase_result["issues"].append(f"{stock_code} 未获得策略分析结果")
                            phase_result["success"] = False
                    else:
                        phase_result["issues"].append(f"{stock_code} 分析失败，无法验证策略")
                        phase_result["success"] = False
                else:
                    phase_result["issues"].append(f"{stock_code} 缺少分析结果")
                    phase_result["success"] = False
            
            phase_result["details"]["strategy_results"] = strategy_results
            
        except Exception as e:
            phase_result["success"] = False
            phase_result["issues"].append(f"策略验证异常: {e}")
            logger.error(f"策略信号验证失败: {e}")
        
        phase_result["duration"] = time.time() - phase_result["start_time"]
        return phase_result
    
    def phase6_consistency_validation(self) -> Dict[str, Any]:
        """阶段6：双向验证一致性"""
        logger.info("🔄 阶段6：双向验证一致性")
        
        phase_result = {
            "phase_name": "双向验证一致性",
            "success": True,
            "start_time": time.time(),
            "details": {},
            "issues": []
        }
        
        try:
            consistency_results = {}
            
            for stock in self.test_stocks:
                stock_code = stock["code"]
                stock_name = stock["name"]
                
                logger.info(f"执行股票 {stock_code}({stock_name}) 双向验证")
                
                # 第二次独立分析
                second_analysis = self.analyzer.analyze_multi_period_buypoint(
                    stock_code=stock_code,
                    target_date=self.target_date
                )
                
                # 对比两次分析结果
                if stock_code in self.analysis_results and second_analysis.get("status") == "SUCCESS":
                    first_result = self.analysis_results[stock_code]
                    
                    # 对比综合评分
                    first_score = first_result.get("overall_score", 0)
                    second_score = second_analysis.get("overall_score", 0)
                    score_diff = abs(first_score - second_score)
                    score_consistency = score_diff <= 5  # 允许5分以内的差异
                    
                    # 对比买点信号
                    first_signals = first_result.get("buypoint_analysis", {}).get("buypoint_signals", {})
                    second_signals = second_analysis.get("buypoint_analysis", {}).get("buypoint_signals", {})
                    
                    signal_consistency = self._compare_signals(first_signals, second_signals)
                    
                    overall_consistency = score_consistency and signal_consistency >= 0.9
                    
                    consistency_results[stock_code] = {
                        "stock_name": stock_name,
                        "score_consistency": score_consistency,
                        "signal_consistency": f"{signal_consistency:.1%}",
                        "overall_consistency": overall_consistency,
                        "first_score": first_score,
                        "second_score": second_score,
                        "score_difference": score_diff
                    }
                    
                    logger.info(f"{stock_code} 一致性验证 - 评分差异:{score_diff:.1f}, 信号一致性:{signal_consistency:.1%}")
                    
                    if not overall_consistency:
                        phase_result["issues"].append(f"{stock_code} 一致性验证失败")
                        phase_result["success"] = False
                
                else:
                    phase_result["issues"].append(f"{stock_code} 第二次分析失败，无法验证一致性")
                    phase_result["success"] = False
            
            phase_result["details"]["consistency_results"] = consistency_results
            
        except Exception as e:
            phase_result["success"] = False
            phase_result["issues"].append(f"一致性验证异常: {e}")
            logger.error(f"双向验证一致性失败: {e}")
        
        phase_result["duration"] = time.time() - phase_result["start_time"]
        return phase_result
    
    def _compare_signals(self, signals1: Dict, signals2: Dict) -> float:
        """比较两组信号的一致性"""
        if not signals1 or not signals2:
            return 0.0
        
        common_periods = set(signals1.keys()) & set(signals2.keys())
        if not common_periods:
            return 0.0
        
        consistent_count = 0
        total_count = 0
        
        for period in common_periods:
            period_signals1 = signals1[period]
            period_signals2 = signals2[period]
            
            if isinstance(period_signals1, dict) and isinstance(period_signals2, dict):
                common_indicators = set(period_signals1.keys()) & set(period_signals2.keys())
                
                for indicator in common_indicators:
                    signal1 = period_signals1[indicator]
                    signal2 = period_signals2[indicator]
                    
                    if signal1 == signal2:
                        consistent_count += 1
                    total_count += 1
        
        return consistent_count / total_count if total_count > 0 else 0.0
    
    def _calculate_summary(self, results: Dict[str, Any]):
        """计算总结"""
        summary = results["summary"]
        
        for phase_name, phase_result in results["phases"].items():
            summary["total_phases"] += 1
            if phase_result.get("success", False):
                summary["successful_phases"] += 1
            else:
                summary["failed_phases"] += 1
                summary["critical_issues"].extend(phase_result.get("issues", []))
        
        # 保存分析结果
        summary["analysis_results"] = self.analysis_results
        
        # 计算成功率
        if summary["total_phases"] > 0:
            success_rate = (summary["successful_phases"] / summary["total_phases"]) * 100
            summary["success_rate"] = f"{success_rate:.1f}%"
        else:
            summary["success_rate"] = "0%"

def main():
    """主函数"""
    print("🚀 完整股票分析验证 - 300005（探路者）& 603359（东珠生态）")
    print("=" * 80)
    
    validator = CompleteStockAnalysisValidator()
    results = validator.run_complete_analysis()
    
    # 显示结果
    print(f"\n📊 分析摘要:")
    print(f"目标日期: {results['target_date']}")
    test_stocks_str = ', '.join([f"{s['code']}({s['name']})" for s in results['test_stocks']])
    print(f"测试股票: {test_stocks_str}")
    print(f"总阶段数: {results['summary']['total_phases']}")
    print(f"成功阶段: {results['summary']['successful_phases']}")
    print(f"失败阶段: {results['summary']['failed_phases']}")
    print(f"成功率: {results['summary']['success_rate']}")
    
    # 显示各阶段结果
    print(f"\n📋 各阶段执行结果:")
    for phase_name, phase_result in results['phases'].items():
        status = "✅ 成功" if phase_result['success'] else "❌ 失败"
        duration = phase_result.get('duration', 0)
        print(f"  {status} {phase_result['phase_name']} (耗时: {duration:.2f}秒)")
    
    # 显示关键问题
    if results['summary']['critical_issues']:
        print(f"\n❌ 关键问题 ({len(results['summary']['critical_issues'])}个):")
        for issue in results['summary']['critical_issues']:
            print(f"  ✗ {issue}")
    
    # 显示分析结果
    if results['summary']['analysis_results']:
        print(f"\n📈 股票分析结果:")
        for stock_code, analysis in results['summary']['analysis_results'].items():
            if analysis.get('status') == 'SUCCESS':
                score = analysis.get('overall_score', 0)
                print(f"  📊 {stock_code}: 综合评分 {score:.1f}")
            else:
                print(f"  ❌ {stock_code}: 分析失败")
    
    # 保存详细结果
    with open('complete_stock_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存到: complete_stock_analysis_results.json")
    
    # 返回成功状态
    return results['summary']['failed_phases'] == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
