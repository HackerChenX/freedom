#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略优化工具

根据验证结果优化买点策略，简化条件并提高执行效率
"""

import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class StrategyOptimizer:
    """策略优化器"""
    
    def __init__(self):
        """初始化优化器"""
        pass
        
    def optimize_strategy(self, strategy_file: str, optimization_config: dict = None) -> dict:
        """
        优化策略
        
        Args:
            strategy_file: 策略文件路径
            optimization_config: 优化配置
            
        Returns:
            dict: 优化结果
        """
        print(f"开始优化策略: {strategy_file}")
        
        # 默认优化配置
        if optimization_config is None:
            optimization_config = {
                "max_conditions": 100,        # 最大条件数
                "min_frequency": 3,           # 最小出现频率
                "keep_top_indicators": 20,    # 保留前N个指标
                "diversify_periods": True,    # 保持周期多样性
                "logic_optimization": True    # 逻辑优化
            }
        
        try:
            # 1. 加载原始策略
            original_strategy = self._load_strategy(strategy_file)
            
            # 2. 分析条件频率
            frequency_analysis = self._analyze_condition_frequency(original_strategy)
            
            # 3. 选择核心条件
            core_conditions = self._select_core_conditions(
                original_strategy, 
                frequency_analysis, 
                optimization_config
            )
            
            # 4. 生成优化策略
            optimized_strategy = self._generate_optimized_strategy(
                original_strategy, 
                core_conditions, 
                optimization_config
            )
            
            # 5. 生成优化报告
            optimization_results = {
                "original_strategy": {
                    "name": original_strategy.get("name", "未知"),
                    "condition_count": len(original_strategy.get("conditions", [])),
                    "logic": original_strategy.get("condition_logic", "OR")
                },
                "optimized_strategy": optimized_strategy,
                "optimization_summary": {
                    "original_conditions": len(original_strategy.get("conditions", [])),
                    "optimized_conditions": len(optimized_strategy.get("conditions", [])),
                    "reduction_rate": 1 - len(optimized_strategy.get("conditions", [])) / max(1, len(original_strategy.get("conditions", []))),
                    "optimization_time": datetime.now().isoformat()
                },
                "frequency_analysis": frequency_analysis,
                "optimization_status": "success"
            }
            
            print(f"策略优化完成，条件数从 {len(original_strategy.get('conditions', []))} 减少到 {len(optimized_strategy.get('conditions', []))}")
            return optimization_results
            
        except Exception as e:
            print(f"策略优化失败: {e}")
            return {
                "optimization_status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _load_strategy(self, strategy_file: str) -> dict:
        """加载策略文件"""
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
            
            print(f"成功加载策略: {strategy.get('name', '未知')}")
            return strategy
            
        except Exception as e:
            raise Exception(f"加载策略文件失败: {e}")
    
    def _analyze_condition_frequency(self, strategy: dict) -> dict:
        """分析条件频率"""
        conditions = strategy.get("conditions", [])
        
        analysis = {
            "indicator_frequency": {},
            "period_frequency": {},
            "pattern_frequency": {},
            "indicator_period_combinations": {}
        }
        
        # 统计指标频率
        indicators = [c.get("indicator", "unknown") for c in conditions]
        analysis["indicator_frequency"] = dict(Counter(indicators))
        
        # 统计周期频率
        periods = [c.get("period", "unknown") for c in conditions]
        analysis["period_frequency"] = dict(Counter(periods))
        
        # 统计形态频率
        patterns = [c.get("pattern", "unknown") for c in conditions]
        analysis["pattern_frequency"] = dict(Counter(patterns))
        
        # 统计指标-周期组合频率
        combinations = [f"{c.get('indicator', 'unknown')}_{c.get('period', 'unknown')}" for c in conditions]
        analysis["indicator_period_combinations"] = dict(Counter(combinations))
        
        return analysis
    
    def _select_core_conditions(self, strategy: dict, frequency_analysis: dict, config: dict) -> list:
        """选择核心条件"""
        conditions = strategy.get("conditions", [])
        
        # 1. 按指标频率排序，选择高频指标
        indicator_freq = frequency_analysis["indicator_frequency"]
        top_indicators = sorted(indicator_freq.items(), key=lambda x: x[1], reverse=True)
        top_indicators = [ind for ind, freq in top_indicators[:config.get("keep_top_indicators", 20)] 
                         if freq >= config.get("min_frequency", 3)]
        
        print(f"选择了 {len(top_indicators)} 个高频指标")
        
        # 2. 保持周期多样性
        if config.get("diversify_periods", True):
            period_freq = frequency_analysis["period_frequency"]
            important_periods = [period for period, freq in period_freq.items() 
                               if freq >= config.get("min_frequency", 3)]
        else:
            important_periods = list(frequency_analysis["period_frequency"].keys())
        
        print(f"保留了 {len(important_periods)} 个重要周期")
        
        # 3. 选择核心条件
        core_conditions = []
        for condition in conditions:
            indicator = condition.get("indicator", "")
            period = condition.get("period", "")
            
            # 检查是否为核心指标和重要周期
            if indicator in top_indicators and period in important_periods:
                core_conditions.append(condition)
                
                # 限制条件数量
                if len(core_conditions) >= config.get("max_conditions", 100):
                    break
        
        print(f"选择了 {len(core_conditions)} 个核心条件")
        return core_conditions
    
    def _generate_optimized_strategy(self, original_strategy: dict, core_conditions: list, config: dict) -> dict:
        """生成优化策略"""
        optimized_strategy = original_strategy.copy()
        
        # 更新策略信息
        optimized_strategy["name"] = f"{original_strategy.get('name', 'Strategy')}_Optimized"
        optimized_strategy["description"] = f"优化版策略：{original_strategy.get('description', '')}"
        optimized_strategy["version"] = "1.0_optimized"
        optimized_strategy["optimization_info"] = {
            "original_conditions": len(original_strategy.get("conditions", [])),
            "optimized_conditions": len(core_conditions),
            "optimization_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "optimization_method": "frequency_based_selection"
        }
        
        # 更新条件
        optimized_strategy["conditions"] = core_conditions
        
        # 逻辑优化
        if config.get("logic_optimization", True):
            # 如果条件数量较少，可以考虑使用AND逻辑
            if len(core_conditions) <= 20:
                optimized_strategy["condition_logic"] = "AND"
                print("条件数量较少，改为AND逻辑")
            else:
                optimized_strategy["condition_logic"] = "OR"
                print("保持OR逻辑")
        
        return optimized_strategy
    
    def save_optimized_strategy(self, optimization_results: dict, output_dir: Path) -> str:
        """保存优化后的策略"""
        if optimization_results["optimization_status"] != "success":
            raise Exception("优化失败，无法保存策略")
        
        optimized_strategy = optimization_results["optimized_strategy"]
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_file = output_dir / f"optimized_strategy_{timestamp}.json"
        
        # 保存策略文件
        with open(strategy_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_strategy, f, ensure_ascii=False, indent=2)
        
        print(f"优化策略已保存到: {strategy_file}")
        return str(strategy_file)
    
    def generate_optimization_report(self, optimization_results: dict, output_dir: Path) -> str:
        """生成优化报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"optimization_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("策略优化报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 原始策略信息
            original = optimization_results.get("original_strategy", {})
            f.write("原始策略:\n")
            f.write(f"  名称: {original.get('name', 'N/A')}\n")
            f.write(f"  条件数量: {original.get('condition_count', 0)}\n")
            f.write(f"  逻辑类型: {original.get('logic', 'N/A')}\n\n")
            
            # 优化策略信息
            optimized = optimization_results.get("optimized_strategy", {})
            f.write("优化策略:\n")
            f.write(f"  名称: {optimized.get('name', 'N/A')}\n")
            f.write(f"  条件数量: {len(optimized.get('conditions', []))}\n")
            f.write(f"  逻辑类型: {optimized.get('condition_logic', 'N/A')}\n\n")
            
            # 优化摘要
            summary = optimization_results.get("optimization_summary", {})
            f.write("优化摘要:\n")
            f.write(f"  原始条件数: {summary.get('original_conditions', 0)}\n")
            f.write(f"  优化条件数: {summary.get('optimized_conditions', 0)}\n")
            f.write(f"  减少比例: {summary.get('reduction_rate', 0):.1%}\n")
            f.write(f"  优化时间: {summary.get('optimization_time', 'N/A')}\n\n")
            
            # 频率分析
            frequency = optimization_results.get("frequency_analysis", {})
            indicator_freq = frequency.get("indicator_frequency", {})
            if indicator_freq:
                f.write("高频指标 (前10名):\n")
                sorted_indicators = sorted(indicator_freq.items(), key=lambda x: x[1], reverse=True)
                for i, (indicator, freq) in enumerate(sorted_indicators[:10], 1):
                    f.write(f"  {i:2d}. {indicator}: {freq} 次\n")
                f.write("\n")
            
            period_freq = frequency.get("period_frequency", {})
            if period_freq:
                f.write("周期分布:\n")
                for period, freq in sorted(period_freq.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {period}: {freq} 次\n")
        
        print(f"优化报告已保存到: {report_file}")
        return str(report_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="策略优化工具")
    parser.add_argument("--strategy", required=True, help="策略文件路径")
    parser.add_argument("--output", help="输出目录，默认为results/optimization")
    parser.add_argument("--max-conditions", type=int, default=100, help="最大条件数")
    parser.add_argument("--min-frequency", type=int, default=3, help="最小出现频率")
    parser.add_argument("--top-indicators", type=int, default=20, help="保留前N个指标")
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output) if args.output else Path("results/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 优化配置
    optimization_config = {
        "max_conditions": args.max_conditions,
        "min_frequency": args.min_frequency,
        "keep_top_indicators": args.top_indicators,
        "diversify_periods": True,
        "logic_optimization": True
    }
    
    # 创建优化器并执行优化
    optimizer = StrategyOptimizer()
    
    print(f"开始优化策略: {args.strategy}")
    print(f"最大条件数: {args.max_conditions}")
    print(f"最小频率: {args.min_frequency}")
    print(f"保留指标数: {args.top_indicators}")
    print("-" * 50)
    
    results = optimizer.optimize_strategy(args.strategy, optimization_config)
    
    if results["optimization_status"] == "success":
        # 保存优化策略
        strategy_file = optimizer.save_optimized_strategy(results, output_dir)
        
        # 生成优化报告
        report_file = optimizer.generate_optimization_report(results, output_dir)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"optimization_result_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 打印结果摘要
        print("\n" + "=" * 60)
        print("📊 优化结果摘要")
        print("=" * 60)
        
        original = results["original_strategy"]
        summary = results["optimization_summary"]
        
        print(f"原始策略: {original['name']}")
        print(f"原始条件数: {original['condition_count']}")
        print(f"优化条件数: {summary['optimized_conditions']}")
        print(f"减少比例: {summary['reduction_rate']:.1%}")
        print(f"新策略逻辑: {results['optimized_strategy']['condition_logic']}")
        
        print(f"\n文件输出:")
        print(f"  优化策略: {strategy_file}")
        print(f"  优化报告: {report_file}")
        print(f"  详细结果: {result_file}")
        
    else:
        print(f"优化失败: {results.get('error', '未知错误')}")


if __name__ == "__main__":
    main()
