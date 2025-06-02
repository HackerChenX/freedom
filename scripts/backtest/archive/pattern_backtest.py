#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模式识别回测脚本

通过回测分析各种技术指标形态的有效性和预测能力
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import json
import argparse

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from analysis.pattern_recognition_analyzer import PatternRecognitionAnalyzer
from indicators.factory import IndicatorFactory
from indicators.base_indicator import PatternResult
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.path_utils import get_result_path, ensure_dir
from enums.kline_period import KlinePeriod

logger = get_logger(__name__)


class PatternBacktester:
    """
    形态回测器
    
    用于回测不同技术形态在股票买点的有效性，支持多周期、多指标的形态分析
    """
    
    def __init__(self, indicators: List[str] = None, periods: List[str] = None):
        """
        初始化形态回测器
        
        Args:
            indicators: 要分析的指标列表，如 ['MACD', 'KDJ', 'RSI']
            periods: 要分析的周期列表，如 ['DAILY', 'WEEKLY']
        """
        # 初始化指标列表和周期列表
        self.indicators = indicators or ["MACD", "KDJ", "RSI"]
        self.periods = periods or ["DAILY"]
        
        # 创建形态识别分析器
        self.analyzer = PatternRecognitionAnalyzer(
            indicators=self.indicators,
            periods=self.periods
        )
        
        # 初始化回测结果
        self.backtest_results = {
            "stocks": {},
            "patterns": {},
            "summary": {},
            "config": {
                "indicators": self.indicators,
                "periods": self.periods
            }
        }
        
        # 初始化数据库连接
        self.db = get_clickhouse_db()
        
        logger.info(f"初始化形态回测器，指标: {self.indicators}，周期: {self.periods}")

    def backtest_stock(self, stock_code: str, 
                      start_date: str, 
                      end_date: str,
                      forward_days: int = 5,
                      threshold: float = 0.02) -> Dict[str, Any]:
        """
        回测单个股票的形态表现
        
        Args:
            stock_code: 股票代码
            start_date: 回测开始日期
            end_date: 回测结束日期
            forward_days: 向前看多少天
            threshold: 涨幅阈值
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        # 获取股票名称
        stock_name = self.db.get_stock_name(stock_code)
        
        # 获取买点日期列表
        buy_dates = self._get_buy_dates(stock_code, start_date, end_date)
        
        if not buy_dates:
            logger.warning(f"股票 {stock_code} 在指定日期范围内没有买点")
            return {
                "stock_code": stock_code,
                "stock_name": stock_name,
                "buy_points": [],
                "patterns": {},
                "success_rate": 0,
                "avg_gain": 0
            }
        
        logger.info(f"股票 {stock_code} 共有 {len(buy_dates)} 个买点")
        
        # 买点分析结果
        buy_points = []
        
        # 形态统计
        pattern_stats = {}
        
        # 成功的买点数量
        success_count = 0
        total_gain = 0
        
        # 分析每个买点
        for buy_date in buy_dates:
            # 获取买点前后的数据
            pre_buy_date = (datetime.strptime(buy_date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
            post_buy_date = (datetime.strptime(buy_date, "%Y-%m-%d") + timedelta(days=forward_days + 10)).strftime("%Y-%m-%d")
            
            # 获取不同周期的数据
            period_data = {}
            for period in self.periods:
                try:
                    period_data[period] = self.db.get_kline_data(
                        stock_code=stock_code,
                        start_date=pre_buy_date,
                        end_date=post_buy_date,
                        period=period
                    )
                except Exception as e:
                    logger.error(f"获取 {stock_code} {period} 周期数据时出错: {e}")
            
            # 如果任何周期的数据为空，则跳过此买点
            if any(data is None or data.empty for data in period_data.values()):
                logger.warning(f"股票 {stock_code} 在买点 {buy_date} 的某些周期数据为空，跳过此买点")
                continue
            
            # 对齐数据到买点日期
            aligned_data = {}
            for period, data in period_data.items():
                # 找到买点日期在数据中的位置
                if isinstance(data.index, pd.DatetimeIndex):
                    buy_date_dt = pd.to_datetime(buy_date)
                    # 找到不晚于买点日期的最后一个日期
                    mask = data.index <= buy_date_dt
                    if not mask.any():
                        logger.warning(f"买点日期 {buy_date} 在 {period} 周期数据中不存在")
                        continue
                    
                    buy_idx = mask.sum() - 1
                    aligned_data[period] = data.iloc[:buy_idx+1]
                else:
                    # 非日期索引的处理
                    aligned_data[period] = data
            
            # 如果没有对齐的数据，则跳过此买点
            if not aligned_data:
                logger.warning(f"股票 {stock_code} 在买点 {buy_date} 没有对齐的数据，跳过此买点")
                continue
            
            # 使用形态识别分析器分析
            analysis_result = self.analyzer.analyze(
                data=aligned_data,
                stock_code=stock_code,
                stock_name=stock_name
            )
            
            # 获取买点后的实际涨跌幅
            future_change = self._calculate_future_change(
                stock_code=stock_code,
                buy_date=buy_date,
                forward_days=forward_days
            )
            
            # 判断买点是否成功
            is_success = future_change >= threshold
            
            if is_success:
                success_count += 1
                total_gain += future_change
            
            # 记录买点信息
            buy_point = {
                "date": buy_date,
                "future_change": future_change,
                "is_success": is_success,
                "score": analysis_result.get("scores", {}).get("total_score", 50),
                "patterns": {}
            }
            
            # 收集形态信息
            for period, period_data in analysis_result.get("periods", {}).items():
                buy_point["patterns"][period] = []
                
                for pattern in period_data.get("patterns", []):
                    pattern_id = pattern.get("pattern_id")
                    indicator_id = pattern.get("indicator_id")
                    display_name = pattern.get("display_name")
                    
                    # 添加到买点形态列表
                    buy_point["patterns"][period].append({
                        "pattern_id": pattern_id,
                        "indicator_id": indicator_id,
                        "display_name": display_name,
                        "strength": pattern.get("strength", 50)
                    })
                    
                    # 更新形态统计
                    pattern_key = f"{indicator_id}_{pattern_id}_{period}"
                    if pattern_key not in pattern_stats:
                        pattern_stats[pattern_key] = {
                            "indicator_id": indicator_id,
                            "pattern_id": pattern_id,
                            "display_name": display_name,
                            "period": period,
                            "count": 0,
                            "success_count": 0,
                            "total_gain": 0,
                            "avg_gain": 0,
                            "success_rate": 0
                        }
                    
                    pattern_stats[pattern_key]["count"] += 1
                    if is_success:
                        pattern_stats[pattern_key]["success_count"] += 1
                        pattern_stats[pattern_key]["total_gain"] += future_change
            
            buy_points.append(buy_point)
        
        # 计算形态的成功率和平均收益
        for pattern_key, stats in pattern_stats.items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["success_count"] / stats["count"]
                stats["avg_gain"] = stats["total_gain"] / stats["count"]
        
        # 计算总体成功率和平均收益
        overall_success_rate = success_count / len(buy_points) if buy_points else 0
        overall_avg_gain = total_gain / len(buy_points) if buy_points else 0
        
        # 生成回测结果
        result = {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "buy_points": buy_points,
            "patterns": list(pattern_stats.values()),
            "success_rate": overall_success_rate,
            "avg_gain": overall_avg_gain
        }
        
        # 更新回测结果
        self.backtest_results["stocks"][stock_code] = result
        
        # 更新全局形态统计
        self._update_global_pattern_stats(pattern_stats)
        
        return result

    def _get_buy_dates(self, stock_code: str, start_date: str, end_date: str) -> List[str]:
        """获取买点日期列表"""
        try:
            # 从数据库获取买点日期
            buy_dates = self.db.get_buy_dates(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            return buy_dates
        except Exception as e:
            logger.error(f"获取买点日期时出错: {e}")
            return []
    
    def _calculate_future_change(self, stock_code: str, buy_date: str, 
                               forward_days: int) -> float:
        """计算未来涨跌幅"""
        try:
            # 获取买点日期的收盘价
            buy_price_data = self.db.get_kline_data(
                stock_code=stock_code,
                start_date=buy_date,
                end_date=buy_date,
                period="DAILY"
            )
            
            if buy_price_data is None or buy_price_data.empty:
                logger.warning(f"无法获取 {stock_code} 在 {buy_date} 的价格数据")
                return 0
            
            buy_price = buy_price_data['close'].iloc[0]
            
            # 获取未来日期的收盘价
            future_date = (datetime.strptime(buy_date, "%Y-%m-%d") + timedelta(days=forward_days)).strftime("%Y-%m-%d")
            future_end_date = (datetime.strptime(future_date, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
            
            future_price_data = self.db.get_kline_data(
                stock_code=stock_code,
                start_date=buy_date,
                end_date=future_end_date,
                period="DAILY"
            )
            
            if future_price_data is None or future_price_data.empty:
                logger.warning(f"无法获取 {stock_code} 在 {future_date} 的价格数据")
                return 0
            
            # 找到不早于future_date的第一个日期
            future_date_dt = pd.to_datetime(future_date)
            future_price = None
            
            if isinstance(future_price_data.index, pd.DatetimeIndex):
                # 找到距离目标日期最近的行
                closest_idx = (future_price_data.index - future_date_dt).abs().argmin()
                future_price = future_price_data['close'].iloc[closest_idx]
            else:
                # 取最后一个价格作为未来价格
                future_price = future_price_data['close'].iloc[-1]
            
            # 计算涨跌幅
            if buy_price > 0:
                return (future_price - buy_price) / buy_price
            else:
                return 0
            
        except Exception as e:
            logger.error(f"计算未来涨跌幅时出错: {e}")
            return 0
    
    def _update_global_pattern_stats(self, pattern_stats: Dict[str, Dict[str, Any]]) -> None:
        """更新全局形态统计"""
        for pattern_key, stats in pattern_stats.items():
            if pattern_key not in self.backtest_results["patterns"]:
                self.backtest_results["patterns"][pattern_key] = {
                    "indicator_id": stats["indicator_id"],
                    "pattern_id": stats["pattern_id"],
                    "display_name": stats["display_name"],
                    "period": stats["period"],
                    "count": 0,
                    "success_count": 0,
                    "total_gain": 0,
                    "stocks": set()
                }
            
            global_stats = self.backtest_results["patterns"][pattern_key]
            global_stats["count"] += stats["count"]
            global_stats["success_count"] += stats["success_count"]
            global_stats["total_gain"] += stats["total_gain"]
            global_stats["stocks"].add(stats["stock_code"] if "stock_code" in stats else "unknown")
    
    def backtest_multiple_stocks(self, stock_codes: List[str], 
                               start_date: str, 
                               end_date: str,
                               forward_days: int = 5,
                               threshold: float = 0.02) -> Dict[str, Any]:
        """
        回测多个股票的形态表现
        
        Args:
            stock_codes: 股票代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            forward_days: 向前看多少天
            threshold: 涨幅阈值
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        # 清空之前的回测结果
        self.backtest_results = {
            "stocks": {},
            "patterns": {},
            "summary": {},
            "config": {
                "indicators": self.indicators,
                "periods": self.periods,
                "start_date": start_date,
                "end_date": end_date,
                "forward_days": forward_days,
                "threshold": threshold
            }
        }
        
        # 遍历股票列表进行回测
        for stock_code in stock_codes:
            try:
                logger.info(f"回测股票 {stock_code}")
                self.backtest_stock(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    forward_days=forward_days,
                    threshold=threshold
                )
            except Exception as e:
                logger.error(f"回测股票 {stock_code} 时出错: {e}")
        
        # 生成回测总结
        self._generate_summary()
        
        return self.backtest_results
    
    def _generate_summary(self) -> None:
        """生成回测总结"""
        # 计算全局形态的成功率和平均收益
        for pattern_key, stats in self.backtest_results["patterns"].items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["success_count"] / stats["count"]
                stats["avg_gain"] = stats["total_gain"] / stats["count"]
                # 将股票集合转换为列表，方便序列化
                stats["stocks"] = list(stats["stocks"])
        
        # 生成总体统计
        total_buy_points = sum(len(stock_result["buy_points"]) for stock_result in self.backtest_results["stocks"].values())
        total_success = sum(sum(1 for bp in stock_result["buy_points"] if bp["is_success"]) for stock_result in self.backtest_results["stocks"].values())
        total_gain = sum(sum(bp["future_change"] for bp in stock_result["buy_points"]) for stock_result in self.backtest_results["stocks"].values())
        
        overall_success_rate = total_success / total_buy_points if total_buy_points > 0 else 0
        overall_avg_gain = total_gain / total_buy_points if total_buy_points > 0 else 0
        
        # 按成功率排序的形态
        patterns_by_success_rate = sorted(
            self.backtest_results["patterns"].values(),
            key=lambda x: (x.get("success_rate", 0), x.get("count", 0)),
            reverse=True
        )
        
        # 按平均收益排序的形态
        patterns_by_avg_gain = sorted(
            self.backtest_results["patterns"].values(),
            key=lambda x: (x.get("avg_gain", 0), x.get("count", 0)),
            reverse=True
        )
        
        # 按计数排序的形态
        patterns_by_count = sorted(
            self.backtest_results["patterns"].values(),
            key=lambda x: x.get("count", 0),
            reverse=True
        )
        
        # 生成回测总结
        self.backtest_results["summary"] = {
            "total_stocks": len(self.backtest_results["stocks"]),
            "total_buy_points": total_buy_points,
            "total_success": total_success,
            "overall_success_rate": overall_success_rate,
            "overall_avg_gain": overall_avg_gain,
            "top_patterns_by_success_rate": patterns_by_success_rate[:10],
            "top_patterns_by_avg_gain": patterns_by_avg_gain[:10],
            "top_patterns_by_count": patterns_by_count[:10],
            "period_stats": self._calculate_period_stats(),
            "indicator_stats": self._calculate_indicator_stats()
        }
    
    def _calculate_period_stats(self) -> Dict[str, Dict[str, Any]]:
        """计算各周期的统计数据"""
        period_stats = {}
        
        for pattern in self.backtest_results["patterns"].values():
            period = pattern.get("period")
            if period not in period_stats:
                period_stats[period] = {
                    "count": 0,
                    "success_count": 0,
                    "total_gain": 0
                }
            
            period_stats[period]["count"] += pattern.get("count", 0)
            period_stats[period]["success_count"] += pattern.get("success_count", 0)
            period_stats[period]["total_gain"] += pattern.get("total_gain", 0)
        
        # 计算成功率和平均收益
        for period, stats in period_stats.items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["success_count"] / stats["count"]
                stats["avg_gain"] = stats["total_gain"] / stats["count"]
        
        return period_stats
    
    def _calculate_indicator_stats(self) -> Dict[str, Dict[str, Any]]:
        """计算各指标的统计数据"""
        indicator_stats = {}
        
        for pattern in self.backtest_results["patterns"].values():
            indicator_id = pattern.get("indicator_id")
            if indicator_id not in indicator_stats:
                indicator_stats[indicator_id] = {
                    "count": 0,
                    "success_count": 0,
                    "total_gain": 0
                }
            
            indicator_stats[indicator_id]["count"] += pattern.get("count", 0)
            indicator_stats[indicator_id]["success_count"] += pattern.get("success_count", 0)
            indicator_stats[indicator_id]["total_gain"] += pattern.get("total_gain", 0)
        
        # 计算成功率和平均收益
        for indicator_id, stats in indicator_stats.items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["success_count"] / stats["count"]
                stats["avg_gain"] = stats["total_gain"] / stats["count"]
        
        return indicator_stats


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="形态回测工具")
    
    parser.add_argument("--stocks", "-s", required=True, help="股票代码列表，多个股票用逗号分隔")
    parser.add_argument("--start", default=None, help="开始日期，格式：YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期，格式：YYYY-MM-DD")
    parser.add_argument("--periods", default="DAILY", help="周期列表，多个周期用逗号分隔")
    parser.add_argument("--indicators", default="MACD,KDJ,RSI", help="指标列表，多个指标用逗号分隔")
    parser.add_argument("--days", type=int, default=365, help="回测时间范围，默认为365天")
    parser.add_argument("--forward", type=int, default=5, help="向前看的天数，默认为5天")
    parser.add_argument("--threshold", type=float, default=0.02, help="收益率阈值，默认为0.02（2%）")
    parser.add_argument("--output", "-o", default=None, help="输出文件路径")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 解析参数
    stock_codes = [s.strip() for s in args.stocks.split(",")]
    
    # 解析周期列表
    periods = [p.strip() for p in args.periods.split(",")]
    
    # 解析指标列表
    indicators = [i.strip() for i in args.indicators.split(",")]
    
    # 如果没有指定开始日期，则使用当前日期减去指定天数
    if not args.start:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    else:
        start_date = args.start
    
    # 如果没有指定结束日期，则使用当前日期
    if not args.end:
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = args.end
    
    logger.info(f"开始回测 {len(stock_codes)} 只股票")
    logger.info(f"回测周期: {periods}")
    logger.info(f"回测指标: {indicators}")
    logger.info(f"回测时间: {start_date} 到 {end_date}")
    logger.info(f"向前看天数: {args.forward}")
    logger.info(f"收益率阈值: {args.threshold}")
    
    # 创建回测器
    backtester = PatternBacktester(indicators=indicators, periods=periods)
    
    # 运行回测
    if len(stock_codes) == 1:
        result = backtester.backtest_stock(
            stock_code=stock_codes[0],
            start_date=start_date,
            end_date=end_date,
            forward_days=args.forward,
            threshold=args.threshold
        )
    else:
        result = backtester.backtest_multiple_stocks(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            forward_days=args.forward,
            threshold=args.threshold
        )
    
    # 输出结果摘要
    print("\n" + "="*60)
    print("形态回测结果摘要")
    print("="*60)
    
    # 按平均收益率排序
    pattern_summary = result.get("pattern_summary", {})
    sorted_patterns = sorted(
        pattern_summary.items(), 
        key=lambda x: x[1]["avg_return"], 
        reverse=True
    )
    
    print(f"\n总共回测了 {len(pattern_summary)} 种形态")
    print(f"回测股票数量: {len(stock_codes)}")
    print(f"回测时间: {start_date} 到 {end_date}")
    print(f"向前看天数: {args.forward}")
    print(f"收益率阈值: {args.threshold * 100}%")
    
    print("\n表现最好的形态 (按平均收益率):")
    for i, (pattern_id, summary) in enumerate(sorted_patterns[:10]):
        print(f"{i+1}. {summary['display_name']} ({summary['indicator_id']})")
        print(f"   - 平均收益率: {summary['avg_return']*100:.2f}%")
        print(f"   - 胜率: {summary['win_rate']*100:.2f}%")
        print(f"   - 有效胜率: {summary['effective_win_rate']*100:.2f}%")
        print(f"   - 出现次数: {summary['occurrences']}")
        print(f"   - 出现股票数: {summary['stock_count']}")
        print()
    
    # 保存结果
    if args.output:
        output_path = args.output
    else:
        # 默认保存到results目录
        output_dir = get_result_path("pattern_backtest")
        ensure_dir(output_dir)
        output_path = os.path.join(
            output_dir, 
            f"pattern_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    backtester.save_results(output_path)
    print(f"\n回测结果已保存到: {output_path}")


if __name__ == "__main__":
    main() 