#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点策略验证工具

用于验证买点分析生成的选股策略的有效性
"""

import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from db.unified_data_manager import get_unified_data_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class BuypointstrategyvalidatorStrategy:
    """买点策略验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.data_manager = get_unified_data_manager()
        self.strategy_executor = StrategyExecutor()
        self.strategy_manager = StrategyManager()
        
    def validate_strategy_Strategy(self, strategy_file: str, validation_config: dict = None) -> dict:
        """
        验证买点策略
        
        Args:
            strategy_file: 策略文件路径
            validation_config: 验证配置
            
        Returns:
            dict: 验证结果
        """
        logger.info(f"开始验证策略: {strategy_file}")
        
        # 默认验证配置
        if validation_config is None:
            validation_config = {
                "validation_date": datetime.now().strftime("%Y-%m-%d"),
                "stock_pool_size": 100,  # 验证股票池大小
                "min_match_rate": 0.6,   # 最小匹配率
                "enable_backtest": True,  # 启用回测
                "backtest_days": 30      # 回测天数
            }
        
        try:
            # 1. 加载策略
            strategy = self._load_strategy_Validate_Buypoint_Strategy(strategy_file)
            
            # 2. 准备验证股票池
            stock_pool = self._prepare_stock_pool_Validate_Buypoint_Strategy(validation_config["stock_pool_size"])
            
            # 3. 执行策略选股
            selected_stocks = self._execute_strategy(strategy, stock_pool, validation_config["validation_date"])
            
            # 4. 分析选股结果
            selection_analysis = self._analyze_selection_results(selected_stocks, strategy)
            
            # 5. 回测验证（如果启用）
            backtest_results = None
            if validation_config.get("enable_backtest", False):
                backtest_results = self._run_backtest(
                    selected_stocks, 
                    validation_config["validation_date"],
                    validation_config["backtest_days"]
                )
            
            # 6. 生成验证报告
            validation_results = {
                "strategy_info": {
                    "name": strategy.get("name", "未知策略"),
                    "description": strategy.get("description", ""),
                    "condition_count": len(strategy.get("conditions", [])),
                    "validation_date": validation_config["validation_date"]
                },
                "selection_results": {
                    "total_pool_size": len(stock_pool),
                    "selected_count": len(selected_stocks) if selected_stocks is not None else 0,
                    "selection_rate": len(selected_stocks) / len(stock_pool) if selected_stocks is not None and len(stock_pool) > 0 else 0,
                    "selected_stocks": selected_stocks.to_dict('records') if selected_stocks is not None else []
                },
                "analysis": selection_analysis,
                "backtest": backtest_results,
                "validation_status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            # 7. 评估验证质量
            quality_assessment = self._assess_validation_quality(validation_results, validation_config)
            validation_results["quality_assessment"] = quality_assessment
            
            logger.info(f"策略验证完成，选出 {len(selected_stocks) if selected_stocks is not None else 0} 只股票")
            return validation_results
            
        except Exception as e:
            logger.error(f"策略验证失败: {e}")
            return {
                "validation_status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _load_strategy_Validate_Buypoint_Strategy(self, strategy_file: str) -> dict:
        """加载策略文件"""
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
            
            logger.info(f"成功加载策略: {strategy.get('name', '未知')}")
            return strategy
            
        except Exception as e:
            raise Exception(f"加载策略文件失败: {e}")
    
    def _prepare_stock_pool_Validate_Buypoint_Strategy(self, pool_size: int) -> list:
        """准备验证股票池"""
        try:
            # 获取活跃股票列表
            all_stocks = self.data_manager.get_all_stock_list()
            
            if len(all_stocks) == 0:
                raise Exception("无法获取股票列表")
            
            # 过滤掉ST股票和停牌股票
            filtered_stocks = []
            for stock in all_stocks:
                stock_code = stock.get('stock_code', '')
                stock_name = stock.get('stock_name', '')
                
                # 简单过滤规则
                if not stock_name.startswith('ST') and not stock_name.startswith('*ST'):
                    filtered_stocks.append(stock)
            
            # 限制股票池大小
            if len(filtered_stocks) > pool_size:
                # 随机选择或按市值排序选择
                import random
                random.shuffle(filtered_stocks)
                filtered_stocks = filtered_stocks[:pool_size]
            
            logger.info(f"准备验证股票池: {len(filtered_stocks)} 只股票")
            return filtered_stocks
            
        except Exception as e:
            raise Exception(f"准备股票池失败: {e}")
    
    def _execute_strategy(self, strategy: dict, stock_pool: list, validation_date: str) -> pd.DataFrame:
        """执行策略选股"""
        try:
            # 保存策略到临时文件
            temp_strategy_id = f"temp_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.strategy_manager.save_strategy(temp_strategy_id, strategy)
            
            # 准备股票代码列表
            stock_codes = [stock['stock_code'] for stock in stock_pool]
            
            # 执行策略
            def progress_callback_Strategy(progress, message):
                if progress % 0.1 < 0.01:  # 每10%打印一次
                    logger.info(f"策略执行进度: {progress:.1%} - {message}")
            
            selected_stocks = self.strategy_executor.execute_strategy_by_id(
                strategy_id=temp_strategy_id,
                strategy_manager=self.strategy_manager,
                end_date=validation_date,
                progress_callback=progress_callback
            )
            
            # 清理临时策略
            self.strategy_manager.delete_strategy(temp_strategy_id)
            
            return selected_stocks
            
        except Exception as e:
            raise Exception(f"执行策略失败: {e}")
    
    def _analyze_selection_results(self, selected_stocks: pd.DataFrame, strategy: dict) -> dict:
        """分析选股结果"""
        if selected_stocks is None or len(selected_stocks) == 0:
            return {
                "summary": "策略未选出任何股票",
                "industry_distribution": {},
                "score_distribution": {},
                "top_stocks": []
            }
        
        analysis = {}
        
        # 行业分布
        if 'industry' in selected_stocks.columns:
            industry_dist = selected_stocks['industry'].value_counts().to_dict()
            analysis["industry_distribution"] = industry_dist
        
        # 评分分布
        if 'score' in selected_stocks.columns:
            scores = selected_stocks['score']
            analysis["score_distribution"] = {
                "mean": float(scores.mean()),
                "median": float(scores.median()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max())
            }
        
        # 前10名股票
        top_stocks = selected_stocks.head(10).to_dict('records')
        analysis["top_stocks"] = top_stocks
        
        # 策略条件分析
        conditions = strategy.get("conditions", [])
        condition_summary = {
            "total_conditions": len(conditions),
            "indicator_types": {},
            "period_distribution": {}
        }
        
        for condition in conditions:
            indicator = condition.get("indicator", "unknown")
            period = condition.get("period", "unknown")
            
            condition_summary["indicator_types"][indicator] = condition_summary["indicator_types"].get(indicator, 0) + 1
            condition_summary["period_distribution"][period] = condition_summary["period_distribution"].get(period, 0) + 1
        
        analysis["condition_summary"] = condition_summary
        analysis["summary"] = f"策略选出 {len(selected_stocks)} 只股票，涵盖 {len(analysis.get('industry_distribution', {}))} 个行业"
        
        return analysis
    
    def _run_backtest(self, selected_stocks: pd.DataFrame, start_date: str, days: int) -> dict:
        """运行回测"""
        if selected_stocks is None or len(selected_stocks) == 0:
            return {"error": "没有选股结果，无法进行回测"}
        
        try:
            # 计算回测结束日期
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = start_dt + timedelta(days=days)
            end_date = end_dt.strftime("%Y-%m-%d")
            
            stock_codes = selected_stocks['stock_code'].tolist()
            
            # 获取回测期间的价格数据
            returns = []
            for stock_code in stock_codes[:20]:  # 限制回测股票数量
                try:
                    price_data = self.data_manager.get_stock_data(
                        stock_code=stock_code,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if len(price_data) >= 2:
                        start_price = price_data.iloc[0]['close']
                        end_price = price_data.iloc[-1]['close']
                        stock_return = (end_price - start_price) / start_price
                        returns.append(stock_return)
                        
                except Exception as e:
                    logger.warning(f"获取股票 {stock_code} 回测数据失败: {e}")
                    continue
            
            if not returns:
                return {"error": "无法获取回测数据"}
            
            # 计算回测统计
            avg_return = sum(returns) / len(returns)
            positive_count = sum(1 for r in returns if r > 0)
            win_rate = positive_count / len(returns)
            
            backtest_results = {
                "period": f"{start_date} 到 {end_date}",
                "tested_stocks": len(returns),
                "average_return": avg_return,
                "win_rate": win_rate,
                "positive_stocks": positive_count,
                "negative_stocks": len(returns) - positive_count,
                "returns": returns
            }
            
            logger.info(f"回测完成: 平均收益 {avg_return:.2%}, 胜率 {win_rate:.1%}")
            return backtest_results
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            return {"error": f"回测失败: {e}"}
    
    def _assess_validation_quality(self, results: dict, config: dict) -> dict:
        """评估验证质量"""
        assessment = {
            "overall_grade": "C",
            "scores": {},
            "recommendations": []
        }
        
        # 选股率评分
        selection_rate = results["selection_results"]["selection_rate"]
        if 0.01 <= selection_rate <= 0.1:  # 1%-10%的选股率比较合理
            assessment["scores"]["selection_rate"] = "A"
        elif 0.005 <= selection_rate <= 0.2:
            assessment["scores"]["selection_rate"] = "B"
        else:
            assessment["scores"]["selection_rate"] = "C"
            if selection_rate < 0.005:
                assessment["recommendations"].append("选股率过低，建议放宽策略条件")
            else:
                assessment["recommendations"].append("选股率过高，建议收紧策略条件")
        
        # 回测表现评分
        backtest = results.get("backtest")
        if backtest and "average_return" in backtest:
            avg_return = backtest["average_return"]
            win_rate = backtest["win_rate"]
            
            if avg_return > 0.05 and win_rate > 0.6:
                assessment["scores"]["backtest"] = "A"
            elif avg_return > 0 and win_rate > 0.5:
                assessment["scores"]["backtest"] = "B"
            else:
                assessment["scores"]["backtest"] = "C"
                assessment["recommendations"].append("回测表现不佳，建议优化策略条件")
        
        # 综合评分
        scores = list(assessment["scores"].values())
        if all(s == "A" for s in scores):
            assessment["overall_grade"] = "A"
        elif any(s == "A" for s in scores) and all(s in ["A", "B"] for s in scores):
            assessment["overall_grade"] = "B"
        else:
            assessment["overall_grade"] = "C"
        
        return assessment


def main_validatebuypointstrategy():
    """主函数"""
    parser = argparse.ArgumentParser(description="买点策略验证工具")
    parser.add_argument("--strategy", required=True, help="策略文件路径")
    parser.add_argument("--output", help="输出目录，默认为results/validation")
    parser.add_argument("--pool-size", type=int, default=100, help="验证股票池大小")
    parser.add_argument("--backtest-days", type=int, default=30, help="回测天数")
    parser.add_argument("--validation-date", help="验证日期，默认为今天")
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output) if args.output else Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置验证日期
    validation_date = args.validation_date or datetime.now().strftime("%Y-%m-%d")
    
    # 验证配置
    validation_config = {
        "validation_date": validation_date,
        "stock_pool_size": args.pool_size,
        "enable_backtest": True,
        "backtest_days": args.backtest_days
    }
    
    # 创建验证器并执行验证
    validator = BuyPointStrategyValidator_Strategy()
    
    print(f"开始验证策略: {args.strategy}")
    print(f"验证日期: {validation_date}")
    print(f"股票池大小: {args.pool_size}")
    print(f"回测天数: {args.backtest_days}")
    print("-" * 50)
    
    results = validator.validate_strategy_Strategy(args.strategy, validation_config)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"validation_result_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("📊 验证结果摘要")
    print("=" * 60)
    
    if results["validation_status"] == "success":
        selection = results["selection_results"]
        print(f"策略名称: {results['strategy_info']['name']}")
        print(f"条件数量: {results['strategy_info']['condition_count']}")
        print(f"股票池大小: {selection['total_pool_size']}")
        print(f"选出股票: {selection['selected_count']}")
        print(f"选股率: {selection['selection_rate']:.2%}")
        
        if results.get("backtest"):
            backtest = results["backtest"]
            if "average_return" in backtest:
                print(f"回测收益: {backtest['average_return']:.2%}")
                print(f"胜率: {backtest['win_rate']:.1%}")
        
        quality = results.get("quality_assessment", {})
        print(f"验证评级: {quality.get('overall_grade', 'N/A')}")
        
        if quality.get("recommendations"):
            print("\n建议:")
            for rec in quality["recommendations"]:
                print(f"  • {rec}")
    else:
        print(f"验证失败: {results.get('error', '未知错误')}")
    
    print(f"\n详细结果已保存到: {result_file}")


if __name__ == "__main__":
    mainValidatebuypointstrategy()
