#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整的买点策略验证工作流

包含策略生成、验证、优化、反向验证的完整流程
"""

import sys
import os
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CompleteValidationWorkflow:
    """完整验证工作流"""
    
    def __init__(self, output_base_dir: str = "results/complete_validation"):
        """初始化工作流"""
        self.output_base_dir = Path(output_base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workflow_dir = self.output_base_dir / f"workflow_{self.timestamp}"
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.analysis_dir = self.workflow_dir / "01_analysis"
        self.simple_validation_dir = self.workflow_dir / "02_simple_validation"
        self.optimization_dir = self.workflow_dir / "03_optimization"
        self.reverse_validation_dir = self.workflow_dir / "04_reverse_validation"
        self.comparison_dir = self.workflow_dir / "05_comparison"
        
        # 创建所有子目录
        for dir_path in [self.analysis_dir, self.simple_validation_dir, 
                        self.optimization_dir, self.reverse_validation_dir, self.comparison_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run_complete_workflow(self, buypoints_file: str, config: dict = None) -> dict:
        """
        运行完整验证工作流
        
        Args:
            buypoints_file: 买点数据文件
            config: 工作流配置
            
        Returns:
            dict: 工作流结果
        """
        print("🚀 开始完整验证工作流")
        print("=" * 60)
        
        # 默认配置
        if config is None:
            config = {
                "max_conditions": 50,
                "min_frequency": 5,
                "top_indicators": 15,
                "validation_date": "2024-12-01"
            }
        
        workflow_results = {
            "workflow_info": {
                "buypoints_file": buypoints_file,
                "start_time": datetime.now().isoformat(),
                "workflow_dir": str(self.workflow_dir),
                "config": config
            },
            "steps": {},
            "summary": {},
            "status": "running"
        }
        
        try:
            # 步骤1: 生成策略
            print("\n📊 步骤1: 生成买点策略")
            print("-" * 40)
            step1_result = self._step1_generate_strategy(buypoints_file)
            workflow_results["steps"]["step1_strategy_generation"] = step1_result
            
            if not step1_result["success"]:
                raise Exception("策略生成失败")
            
            strategy_file = step1_result["strategy_file"]
            
            # 步骤2: 简单验证
            print("\n🔍 步骤2: 策略结构验证")
            print("-" * 40)
            step2_result = self._step2_simple_validation(strategy_file)
            workflow_results["steps"]["step2_simple_validation"] = step2_result
            
            # 步骤3: 策略优化
            print("\n⚡ 步骤3: 策略优化")
            print("-" * 40)
            step3_result = self._step3_optimize_strategy(strategy_file, config)
            workflow_results["steps"]["step3_optimization"] = step3_result
            
            # 步骤4: 反向验证
            print("\n🔄 步骤4: 反向验证")
            print("-" * 40)
            step4_result = self._step4_reverse_validation(buypoints_file, strategy_file, step3_result)
            workflow_results["steps"]["step4_reverse_validation"] = step4_result
            
            # 步骤5: 结果对比
            print("\n📈 步骤5: 结果对比分析")
            print("-" * 40)
            step5_result = self._step5_comparison_analysis(workflow_results)
            workflow_results["steps"]["step5_comparison"] = step5_result
            
            # 生成最终摘要
            workflow_results["summary"] = self._generate_workflow_summary(workflow_results)
            workflow_results["status"] = "completed"
            workflow_results["workflow_info"]["end_time"] = datetime.now().isoformat()
            
            print("\n🎉 完整验证工作流完成！")
            return workflow_results
            
        except Exception as e:
            print(f"\n❌ 工作流执行失败: {e}")
            workflow_results["status"] = "failed"
            workflow_results["error"] = str(e)
            workflow_results["workflow_info"]["end_time"] = datetime.now().isoformat()
            return workflow_results
    
    def _step1_generate_strategy(self, buypoints_file: str) -> dict:
        """步骤1: 生成策略"""
        try:
            cmd = [
                "python", "bin/buypoint_batch_analyzer.py",
                "--input", buypoints_file,
                "--output", str(self.analysis_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                strategy_file = self.analysis_dir / "generated_strategy.json"
                if strategy_file.exists():
                    return {
                        "success": True,
                        "strategy_file": str(strategy_file),
                        "output": result.stdout,
                        "message": "策略生成成功"
                    }
                else:
                    return {
                        "success": False,
                        "error": "策略文件未生成",
                        "output": result.stdout,
                        "stderr": result.stderr
                    }
            else:
                return {
                    "success": False,
                    "error": f"命令执行失败，返回码: {result.returncode}",
                    "output": result.stdout,
                    "stderr": result.stderr
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"步骤1执行异常: {e}"
            }
    
    def _step2_simple_validation(self, strategy_file: str) -> dict:
        """步骤2: 简单验证"""
        try:
            cmd = [
                "python", "scripts/simple_strategy_validator.py",
                "--strategy", strategy_file,
                "--output", str(self.simple_validation_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "stderr": result.stderr if result.returncode != 0 else "",
                "message": "简单验证完成" if result.returncode == 0 else "简单验证失败"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"步骤2执行异常: {e}"
            }
    
    def _step3_optimize_strategy(self, strategy_file: str, config: dict) -> dict:
        """步骤3: 策略优化"""
        try:
            cmd = [
                "python", "scripts/optimize_strategy.py",
                "--strategy", strategy_file,
                "--output", str(self.optimization_dir),
                "--max-conditions", str(config.get("max_conditions", 50)),
                "--min-frequency", str(config.get("min_frequency", 5)),
                "--top-indicators", str(config.get("top_indicators", 15))
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            # 查找优化后的策略文件
            optimized_strategy_file = None
            if result.returncode == 0:
                for file in self.optimization_dir.glob("optimized_strategy_*.json"):
                    optimized_strategy_file = str(file)
                    break
            
            return {
                "success": result.returncode == 0,
                "optimized_strategy_file": optimized_strategy_file,
                "output": result.stdout,
                "stderr": result.stderr if result.returncode != 0 else "",
                "message": "策略优化完成" if result.returncode == 0 else "策略优化失败"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"步骤3执行异常: {e}"
            }
    
    def _step4_reverse_validation(self, buypoints_file: str, original_strategy_file: str, optimization_result: dict) -> dict:
        """步骤4: 反向验证"""
        results = {}
        
        # 验证原始策略
        try:
            cmd = [
                "python", "scripts/reverse_validation.py",
                "--buypoints", buypoints_file,
                "--strategy", original_strategy_file,
                "--output", str(self.reverse_validation_dir / "original")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            results["original_strategy"] = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "stderr": result.stderr if result.returncode != 0 else ""
            }
            
        except Exception as e:
            results["original_strategy"] = {
                "success": False,
                "error": f"原始策略验证异常: {e}"
            }
        
        # 验证优化策略
        if optimization_result.get("success") and optimization_result.get("optimized_strategy_file"):
            try:
                cmd = [
                    "python", "scripts/reverse_validation.py",
                    "--buypoints", buypoints_file,
                    "--strategy", optimization_result["optimized_strategy_file"],
                    "--output", str(self.reverse_validation_dir / "optimized")
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
                results["optimized_strategy"] = {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "stderr": result.stderr if result.returncode != 0 else ""
                }
                
            except Exception as e:
                results["optimized_strategy"] = {
                    "success": False,
                    "error": f"优化策略验证异常: {e}"
                }
        else:
            results["optimized_strategy"] = {
                "success": False,
                "error": "优化策略不可用"
            }
        
        return results
    
    def _step5_comparison_analysis(self, workflow_results: dict) -> dict:
        """步骤5: 结果对比分析"""
        try:
            comparison = {
                "strategy_comparison": {},
                "validation_comparison": {},
                "recommendations": []
            }
            
            # 策略对比
            step1 = workflow_results["steps"].get("step1_strategy_generation", {})
            step3 = workflow_results["steps"].get("step3_optimization", {})
            
            if step1.get("success") and step3.get("success"):
                # 加载策略文件进行对比
                try:
                    with open(step1["strategy_file"], 'r') as f:
                        original_strategy = json.load(f)
                    
                    if step3.get("optimized_strategy_file"):
                        with open(step3["optimized_strategy_file"], 'r') as f:
                            optimized_strategy = json.load(f)
                        
                        comparison["strategy_comparison"] = {
                            "original_conditions": len(original_strategy.get("conditions", [])),
                            "optimized_conditions": len(optimized_strategy.get("conditions", [])),
                            "reduction_rate": 1 - len(optimized_strategy.get("conditions", [])) / max(1, len(original_strategy.get("conditions", []))),
                            "original_logic": original_strategy.get("condition_logic", "OR"),
                            "optimized_logic": optimized_strategy.get("condition_logic", "OR")
                        }
                        
                except Exception as e:
                    comparison["strategy_comparison"]["error"] = f"策略对比失败: {e}"
            
            # 验证结果对比
            step4 = workflow_results["steps"].get("step4_reverse_validation", {})
            if step4:
                original_success = step4.get("original_strategy", {}).get("success", False)
                optimized_success = step4.get("optimized_strategy", {}).get("success", False)
                
                comparison["validation_comparison"] = {
                    "original_validation_success": original_success,
                    "optimized_validation_success": optimized_success,
                    "both_successful": original_success and optimized_success
                }
            
            # 生成建议
            if comparison["strategy_comparison"].get("reduction_rate", 0) > 0.8:
                comparison["recommendations"].append("策略优化效果显著，条件数量大幅减少")
            
            if comparison["validation_comparison"].get("both_successful", False):
                comparison["recommendations"].append("原始策略和优化策略都通过了反向验证")
            
            return {
                "success": True,
                "comparison": comparison,
                "message": "对比分析完成"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"对比分析异常: {e}"
            }
    
    def _generate_workflow_summary(self, workflow_results: dict) -> dict:
        """生成工作流摘要"""
        summary = {
            "total_steps": 5,
            "successful_steps": 0,
            "failed_steps": 0,
            "step_status": {},
            "key_metrics": {},
            "overall_success": True
        }
        
        # 统计步骤状态
        for step_name, step_result in workflow_results["steps"].items():
            if isinstance(step_result, dict):
                if step_result.get("success", False):
                    summary["successful_steps"] += 1
                    summary["step_status"][step_name] = "✅ 成功"
                else:
                    summary["failed_steps"] += 1
                    summary["step_status"][step_name] = "❌ 失败"
                    summary["overall_success"] = False
        
        # 提取关键指标
        strategy_comparison = workflow_results["steps"].get("step5_comparison", {}).get("comparison", {}).get("strategy_comparison", {})
        if strategy_comparison:
            summary["key_metrics"]["condition_reduction"] = strategy_comparison.get("reduction_rate", 0)
            summary["key_metrics"]["original_conditions"] = strategy_comparison.get("original_conditions", 0)
            summary["key_metrics"]["optimized_conditions"] = strategy_comparison.get("optimized_conditions", 0)
        
        return summary
    
    def save_workflow_results(self, workflow_results: dict):
        """保存工作流结果"""
        # 保存完整结果
        result_file = self.workflow_dir / "workflow_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(workflow_results, f, ensure_ascii=False, indent=2)
        
        # 生成摘要报告
        report_file = self.workflow_dir / "workflow_summary.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("完整验证工作流摘要报告\n")
            f.write("=" * 60 + "\n\n")
            
            workflow_info = workflow_results["workflow_info"]
            f.write(f"买点文件: {workflow_info['buypoints_file']}\n")
            f.write(f"开始时间: {workflow_info['start_time']}\n")
            f.write(f"结束时间: {workflow_info.get('end_time', 'N/A')}\n")
            f.write(f"工作流目录: {workflow_info['workflow_dir']}\n")
            f.write(f"执行状态: {workflow_results['status']}\n\n")
            
            summary = workflow_results.get("summary", {})
            f.write("执行摘要:\n")
            f.write(f"  总步骤数: {summary.get('total_steps', 0)}\n")
            f.write(f"  成功步骤: {summary.get('successful_steps', 0)}\n")
            f.write(f"  失败步骤: {summary.get('failed_steps', 0)}\n")
            f.write(f"  整体成功: {'是' if summary.get('overall_success', False) else '否'}\n\n")
            
            f.write("步骤状态:\n")
            for step_name, status in summary.get("step_status", {}).items():
                f.write(f"  {step_name}: {status}\n")
            
            key_metrics = summary.get("key_metrics", {})
            if key_metrics:
                f.write(f"\n关键指标:\n")
                if "original_conditions" in key_metrics:
                    f.write(f"  原始条件数: {key_metrics['original_conditions']}\n")
                if "optimized_conditions" in key_metrics:
                    f.write(f"  优化条件数: {key_metrics['optimized_conditions']}\n")
                if "condition_reduction" in key_metrics:
                    f.write(f"  条件减少率: {key_metrics['condition_reduction']:.1%}\n")
        
        print(f"\n📁 工作流结果已保存到: {self.workflow_dir}")
        print(f"  详细结果: {result_file}")
        print(f"  摘要报告: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="完整的买点策略验证工作流")
    parser.add_argument("--buypoints", required=True, help="买点数据文件")
    parser.add_argument("--output", help="输出目录，默认为results/complete_validation")
    parser.add_argument("--max-conditions", type=int, default=50, help="优化后最大条件数")
    parser.add_argument("--min-frequency", type=int, default=5, help="最小出现频率")
    parser.add_argument("--top-indicators", type=int, default=15, help="保留前N个指标")
    
    args = parser.parse_args()
    
    # 工作流配置
    config = {
        "max_conditions": args.max_conditions,
        "min_frequency": args.min_frequency,
        "top_indicators": args.top_indicators,
        "validation_date": "2024-12-01"
    }
    
    # 创建工作流并执行
    output_dir = args.output or "results/complete_validation"
    workflow = CompleteValidationWorkflow(output_dir)
    
    print(f"🚀 启动完整验证工作流")
    print(f"买点文件: {args.buypoints}")
    print(f"输出目录: {output_dir}")
    print(f"优化配置: 最大条件数={args.max_conditions}, 最小频率={args.min_frequency}, 保留指标数={args.top_indicators}")
    
    # 执行工作流
    results = workflow.run_complete_workflow(args.buypoints, config)
    
    # 保存结果
    workflow.save_workflow_results(results)
    
    # 打印最终摘要
    print("\n" + "=" * 60)
    print("🎯 工作流执行完成")
    print("=" * 60)
    
    summary = results.get("summary", {})
    print(f"执行状态: {results['status']}")
    print(f"成功步骤: {summary.get('successful_steps', 0)}/{summary.get('total_steps', 0)}")
    
    if results["status"] == "completed":
        key_metrics = summary.get("key_metrics", {})
        if key_metrics:
            print(f"策略优化: {key_metrics.get('original_conditions', 0)} → {key_metrics.get('optimized_conditions', 0)} 条件")
            print(f"减少比例: {key_metrics.get('condition_reduction', 0):.1%}")
    elif results["status"] == "failed":
        print(f"失败原因: {results.get('error', '未知错误')}")


if __name__ == "__main__":
    main()
