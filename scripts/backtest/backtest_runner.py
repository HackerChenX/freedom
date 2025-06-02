#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测运行模块

回测系统的主入口，整合各模块功能
"""

import os
import sys
import logging
import datetime
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, time_it
from scripts.backtest.data_manager import BacktestDataManager
from scripts.backtest.strategy_manager import StrategyManager
from scripts.backtest.pattern_analyzer import PatternAnalyzer

# 获取日志记录器
logger = get_logger(__name__)


class BacktestRunner:
    """
    回测运行器
    
    回测系统的主控制器，整合各模块功能
    """

    def __init__(self):
        """初始化回测运行器"""
        self.data_manager = BacktestDataManager()
        self.strategy_manager = StrategyManager()
        self.pattern_analyzer = PatternAnalyzer()
        
        # 回测结果
        self.results = {}
        
        logger.info("回测运行器初始化完成")
    
    def run_pattern_recognition(self, stock_code: str, start_date: str, end_date: str, 
                               pattern_ids: List[str] = None, min_strength: float = 0.6) -> Dict[str, Any]:
        """
        运行形态识别回测
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            pattern_ids: 形态ID列表
            min_strength: 最小形态强度
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        # 设置策略参数
        strategy = self.strategy_manager.get_strategy("pattern")
        if pattern_ids:
            strategy.set_patterns(pattern_ids)
        strategy.set_min_strength(min_strength)
        
        # 运行策略
        result = self.strategy_manager.run_strategy(
            strategy_id="pattern",
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        # 保存结果
        result_key = f"pattern_{stock_code}_{start_date}_{end_date}"
        self.results[result_key] = result
        
        return result
    
    def run_multi_period_analysis(self, stock_code: str, start_date: str, end_date: str,
                                 periods: List[str] = None, 
                                 pattern_combinations: List[Dict[str, Any]] = None,
                                 min_strength: float = 60.0) -> Dict[str, Any]:
        """
        运行多周期形态分析
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            periods: 周期列表
            pattern_combinations: 形态组合
            min_strength: 最小形态强度
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 设置策略参数
        strategy = self.strategy_manager.get_strategy("multi_period")
        if periods:
            strategy.set_periods(periods)
        if pattern_combinations:
            strategy.set_pattern_combinations(pattern_combinations)
        strategy.min_pattern_strength = min_strength
        
        # 运行策略
        result = self.strategy_manager.run_strategy(
            strategy_id="multi_period",
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        # 保存结果
        result_key = f"multi_period_{stock_code}_{start_date}_{end_date}"
        self.results[result_key] = result
        
        return result
    
    def run_zxm_analysis(self, stock_code: str, start_date: str, end_date: str,
                        periods: List[str] = None,
                        buy_score_threshold: float = 60.0) -> Dict[str, Any]:
        """
        运行ZXM形态分析
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            periods: 周期列表
            buy_score_threshold: 买入信号分数阈值
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 设置策略参数
        strategy = self.strategy_manager.get_strategy("zxm")
        if periods:
            strategy.set_periods(periods)
        strategy.buy_score_threshold = buy_score_threshold
        
        # 运行策略
        result = self.strategy_manager.run_strategy(
            strategy_id="zxm",
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        # 保存结果
        result_key = f"zxm_{stock_code}_{start_date}_{end_date}"
        self.results[result_key] = result
        
        return result
    
    def batch_run_pattern_recognition(self, stock_codes: List[str], start_date: str, end_date: str,
                                    pattern_ids: List[str] = None,
                                    min_strength: float = 0.6) -> List[Dict[str, Any]]:
        """
        批量运行形态识别
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            pattern_ids: 形态ID列表
            min_strength: 最小形态强度
            
        Returns:
            List[Dict[str, Any]]: 回测结果列表
        """
        # 设置策略参数
        strategy = self.strategy_manager.get_strategy("pattern")
        if pattern_ids:
            strategy.set_patterns(pattern_ids)
        strategy.set_min_strength(min_strength)
        
        # 批量运行策略
        results = self.strategy_manager.batch_run(
            strategy_id="pattern",
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date
        )
        
        # 保存结果
        for i, stock_code in enumerate(stock_codes):
            result_key = f"pattern_{stock_code}_{start_date}_{end_date}"
            self.results[result_key] = results[i]
        
        return results
    
    def batch_run_zxm_analysis(self, stock_codes: List[str], start_date: str, end_date: str,
                              periods: List[str] = None,
                              buy_score_threshold: float = 60.0) -> List[Dict[str, Any]]:
        """
        批量运行ZXM形态分析
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            periods: 周期列表
            buy_score_threshold: 买入信号分数阈值
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        # 设置策略参数
        strategy = self.strategy_manager.get_strategy("zxm")
        if periods:
            strategy.set_periods(periods)
        strategy.buy_score_threshold = buy_score_threshold
        
        # 批量运行策略
        results = self.strategy_manager.batch_run(
            strategy_id="zxm",
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date
        )
        
        # 保存结果
        for i, stock_code in enumerate(stock_codes):
            result_key = f"zxm_{stock_code}_{start_date}_{end_date}"
            self.results[result_key] = results[i]
        
        return results
    
    def generate_pattern_statistics(self, results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成形态统计信息
        
        Args:
            results: 回测结果列表，为None时使用所有已保存的结果
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if results is None:
            # 使用所有保存的形态识别结果
            results = [result for key, result in self.results.items() if key.startswith("pattern_")]
        
        # 统计形态出现频率
        pattern_stats = {}
        stock_with_patterns = set()
        
        for result in results:
            if not result.get("success", False):
                continue
                
            stock_code = result.get("stock_code", "")
            patterns = result.get("patterns", [])
            
            if patterns:
                stock_with_patterns.add(stock_code)
                
            for pattern in patterns:
                pattern_id = pattern.get("pattern_id", "")
                pattern_name = pattern.get("pattern_name", "")
                
                if not pattern_id:
                    continue
                    
                if pattern_id not in pattern_stats:
                    pattern_stats[pattern_id] = {
                        "pattern_id": pattern_id,
                        "pattern_name": pattern_name,
                        "count": 0,
                        "stocks": set(),
                        "total_strength": 0,
                        "occurrences": []
                    }
                
                # 更新统计信息
                pattern_stats[pattern_id]["count"] += 1
                pattern_stats[pattern_id]["stocks"].add(stock_code)
                pattern_stats[pattern_id]["total_strength"] += pattern.get("strength", 0)
                pattern_stats[pattern_id]["occurrences"].append({
                    "stock_code": stock_code,
                    "stock_name": result.get("stock_name", ""),
                    "strength": pattern.get("strength", 0),
                    "date": result.get("end_date", "")
                })
        
        # 计算平均强度
        for pattern_id, stats in pattern_stats.items():
            if stats["count"] > 0:
                stats["avg_strength"] = stats["total_strength"] / stats["count"]
            else:
                stats["avg_strength"] = 0
                
            # 转换股票集合为列表
            stats["stocks"] = list(stats["stocks"])
            stats["stock_count"] = len(stats["stocks"])
        
        # 计算整体统计信息
        total_stats = {
            "total_results": len(results),
            "stocks_with_patterns": len(stock_with_patterns),
            "total_pattern_occurrences": sum(stats["count"] for stats in pattern_stats.values()),
            "unique_patterns": len(pattern_stats),
            "pattern_stats": pattern_stats
        }
        
        return total_stats
    
    def generate_zxm_statistics(self, results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成ZXM分析统计信息
        
        Args:
            results: ZXM分析结果列表，为None时使用所有已保存的结果
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if results is None:
            # 使用所有保存的ZXM分析结果
            results = [result for key, result in self.results.items() if key.startswith("zxm_")]
        
        # 统计信号
        buy_signals = []
        score_distribution = {
            "0-20": 0,
            "20-40": 0,
            "40-60": 0,
            "60-80": 0,
            "80-100": 0
        }
        
        for result in results:
            if not result.get("success", False):
                continue
                
            stock_code = result.get("stock_code", "")
            composite_score = result.get("composite_score", 0)
            has_buy_signal = result.get("has_buy_signal", False)
            
            # 更新分数分布
            if composite_score < 20:
                score_distribution["0-20"] += 1
            elif composite_score < 40:
                score_distribution["20-40"] += 1
            elif composite_score < 60:
                score_distribution["40-60"] += 1
            elif composite_score < 80:
                score_distribution["60-80"] += 1
            else:
                score_distribution["80-100"] += 1
            
            # 记录买入信号
            if has_buy_signal:
                buy_signals.append({
                    "stock_code": stock_code,
                    "stock_name": result.get("stock_name", ""),
                    "composite_score": composite_score,
                    "composite_rating": result.get("composite_rating", ""),
                    "advice": result.get("advice", ""),
                    "date": result.get("end_date", "")
                })
        
        # 生成统计信息
        stats = {
            "total_results": len(results),
            "buy_signal_count": len(buy_signals),
            "buy_signals": buy_signals,
            "score_distribution": score_distribution
        }
        
        return stats
    
    def export_results_to_csv(self, file_path: str, results: List[Dict[str, Any]] = None):
        """
        导出回测结果到CSV
        
        Args:
            file_path: 文件路径
            results: 回测结果列表，为None时使用所有已保存的结果
        """
        if results is None:
            # 使用所有保存的结果
            results = list(self.results.values())
        
        # 准备导出数据
        export_data = []
        
        for result in results:
            if not result.get("success", False):
                continue
                
            # 基本信息
            row = {
                "股票代码": result.get("stock_code", ""),
                "股票名称": result.get("stock_name", ""),
                "策略名称": result.get("strategy_name", ""),
                "开始日期": result.get("start_date", ""),
                "结束日期": result.get("end_date", ""),
                "最后收盘价": result.get("last_close", "")
            }
            
            # 形态识别结果
            if "patterns" in result:
                patterns = result.get("patterns", [])
                row["形态数量"] = len(patterns)
                row["有形态"] = len(patterns) > 0
                
                if patterns:
                    strongest = max(patterns, key=lambda p: p.get("strength", 0))
                    row["最强形态ID"] = strongest.get("pattern_id", "")
                    row["最强形态名称"] = strongest.get("pattern_name", "")
                    row["最强形态强度"] = strongest.get("strength", 0)
                    row["最强形态描述"] = strongest.get("description", "")
            
            # ZXM分析结果
            if "composite_score" in result:
                row["ZXM综合得分"] = result.get("composite_score", 0)
                row["ZXM综合评级"] = result.get("composite_rating", "")
                row["操作建议"] = result.get("advice", "")
                row["买入信号"] = result.get("has_buy_signal", False)
            
            export_data.append(row)
        
        # 导出到CSV
        if export_data:
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            logger.info(f"回测结果已导出到: {file_path}")
        else:
            logger.warning("没有可导出的回测结果")
    
    def clear_results(self):
        """清除所有回测结果"""
        self.results.clear()
        logger.info("所有回测结果已清除")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="回测系统")
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 形态识别命令
    pattern_parser = subparsers.add_parser("pattern", help="形态识别回测")
    pattern_parser.add_argument("--stock", "-s", type=str, required=True, help="股票代码")
    pattern_parser.add_argument("--start-date", "-sd", type=str, required=True, help="开始日期")
    pattern_parser.add_argument("--end-date", "-ed", type=str, required=True, help="结束日期")
    pattern_parser.add_argument("--pattern-ids", "-p", type=str, nargs="+", help="形态ID列表")
    pattern_parser.add_argument("--min-strength", "-m", type=float, default=0.6, help="最小形态强度")
    
    # ZXM形态分析命令
    zxm_parser = subparsers.add_parser("zxm", help="ZXM形态分析")
    zxm_parser.add_argument("--stock", "-s", type=str, required=True, help="股票代码")
    zxm_parser.add_argument("--start-date", "-sd", type=str, required=True, help="开始日期")
    zxm_parser.add_argument("--end-date", "-ed", type=str, required=True, help="结束日期")
    zxm_parser.add_argument("--periods", "-p", type=str, nargs="+", 
                          default=["daily", "weekly", "monthly"], help="周期列表")
    zxm_parser.add_argument("--threshold", "-t", type=float, default=60.0, help="买入信号分数阈值")
    
    # 批量回测命令
    batch_parser = subparsers.add_parser("batch", help="批量回测")
    batch_parser.add_argument("--strategy", "-st", type=str, required=True, 
                            choices=["pattern", "zxm"], help="策略类型")
    batch_parser.add_argument("--stock-file", "-sf", type=str, required=True, help="股票列表文件")
    batch_parser.add_argument("--start-date", "-sd", type=str, required=True, help="开始日期")
    batch_parser.add_argument("--end-date", "-ed", type=str, required=True, help="结束日期")
    batch_parser.add_argument("--output", "-o", type=str, default="backtest_results.csv", 
                            help="输出文件路径")
    
    # 统计命令
    stats_parser = subparsers.add_parser("stats", help="生成统计信息")
    stats_parser.add_argument("--type", "-t", type=str, required=True, 
                            choices=["pattern", "zxm"], help="统计类型")
    stats_parser.add_argument("--input", "-i", type=str, help="输入文件路径")
    stats_parser.add_argument("--output", "-o", type=str, default="stats.json", help="输出文件路径")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化回测运行器
    runner = BacktestRunner()
    
    if args.command == "pattern":
        # 运行形态识别回测
        result = runner.run_pattern_recognition(
            stock_code=args.stock,
            start_date=args.start_date,
            end_date=args.end_date,
            pattern_ids=args.pattern_ids,
            min_strength=args.min_strength
        )
        
        # 输出结果
        print(f"\n股票 {args.stock} 的形态识别结果:")
        patterns = result.get("patterns", [])
        if patterns:
            print(f"发现 {len(patterns)} 个形态:")
            for pattern in patterns:
                print(f"- {pattern['pattern_name']}: 强度 {pattern['strength']:.2f}")
                print(f"  描述: {pattern['description']}")
        else:
            print("未发现符合条件的形态")
    
    elif args.command == "zxm":
        # 运行ZXM形态分析
        result = runner.run_zxm_analysis(
            stock_code=args.stock,
            start_date=args.start_date,
            end_date=args.end_date,
            periods=args.periods,
            buy_score_threshold=args.threshold
        )
        
        # 输出结果
        print(result.get("zxm_text", "ZXM分析失败"))
    
    elif args.command == "batch":
        # 读取股票列表
        try:
            with open(args.stock_file, "r") as f:
                stock_codes = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"读取股票列表文件失败: {e}")
            print(f"错误: 读取股票列表文件失败 - {e}")
            return
        
        print(f"开始批量回测 {len(stock_codes)} 只股票...")
        
        # 运行批量回测
        if args.strategy == "pattern":
            results = runner.batch_run_pattern_recognition(
                stock_codes=stock_codes,
                start_date=args.start_date,
                end_date=args.end_date
            )
        elif args.strategy == "zxm":
            results = runner.batch_run_zxm_analysis(
                stock_codes=stock_codes,
                start_date=args.start_date,
                end_date=args.end_date
            )
        
        # 导出结果
        runner.export_results_to_csv(args.output, results)
        print(f"回测完成，结果已导出到: {args.output}")
    
    elif args.command == "stats":
        # 生成统计信息
        if args.input:
            # 从文件读取结果
            try:
                df = pd.read_csv(args.input)
                results = df.to_dict("records")
            except Exception as e:
                logger.error(f"读取输入文件失败: {e}")
                print(f"错误: 读取输入文件失败 - {e}")
                return
        else:
            results = None  # 使用已保存的结果
        
        # 生成统计
        if args.type == "pattern":
            stats = runner.generate_pattern_statistics(results)
        elif args.type == "zxm":
            stats = runner.generate_zxm_statistics(results)
        
        # 输出统计信息
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已生成: {args.output}")
    
    else:
        print("未指定有效的命令，使用 -h 查看帮助")


if __name__ == "__main__":
    main() 