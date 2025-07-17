#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略执行工具

用于执行买点分析生成的选股策略
"""

import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from db.unified_data_manager import get_unified_data_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class StrategyRunner:
    """策略执行器"""
    
    def __init__(self):
        """初始化执行器"""
        self.data_manager = get_unified_data_manager()
        self.strategy_executor = StrategyExecutor()
        self.strategy_manager = StrategyManager()
        
    def run_strategy(self, strategy_file: str, config: dict = None) -> dict:
        """
        执行策略
        
        Args:
            strategy_file: 策略文件路径
            config: 执行配置
            
        Returns:
            dict: 执行结果
        """
        logger.info(f"开始执行策略: {strategy_file}")
        
        # 默认配置
        if config is None:
            config = {
                "execution_date": datetime.now().strftime("%Y-%m-%d"),
                "stock_pool": "all",  # all, hs300, sz50, custom
                "max_results": 50,
                "min_score": 60,
                "export_format": ["csv", "json"]
            }
        
        try:
            # 1. 加载策略
            strategy = self._load_strategy_Run_Strategy(strategy_file)
            
            # 2. 准备股票池
            stock_pool = self._prepare_stock_pool_Run_Strategy(config.get("stock_pool", "all"))
            
            # 3. 执行策略
            selected_stocks = self._execute_strategy_Run_Strategy(strategy, config["execution_date"])
            
            # 4. 过滤和排序结果
            filtered_results = self._filter_results(
                selected_stocks, 
                config.get("max_results", 50),
                config.get("min_score", 60)
            )
            
            # 5. 生成执行报告
            execution_results = {
                "strategy_info": {
                    "name": strategy.get("name", "未知策略"),
                    "description": strategy.get("description", ""),
                    "condition_count": len(strategy.get("conditions", [])),
                    "execution_date": config["execution_date"]
                },
                "execution_summary": {
                    "total_conditions": len(strategy.get("conditions", [])),
                    "selected_count": len(filtered_results) if filtered_results is not None else 0,
                    "execution_time": datetime.now().isoformat()
                },
                "selected_stocks": filtered_results.to_dict('records') if filtered_results is not None else [],
                "execution_status": "success"
            }
            
            logger.info(f"策略执行完成，选出 {len(filtered_results) if filtered_results is not None else 0} 只股票")
            return execution_results
            
        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            return {
                "execution_status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _load_strategy_Run_Strategy(self, strategy_file: str) -> dict:
        """加载策略文件"""
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
            
            logger.info(f"成功加载策略: {strategy.get('name', '未知')}")
            return strategy
            
        except Exception as e:
            raise Exception(f"加载策略文件失败: {e}")
    
    def _prepare_stock_pool_Run_Strategy(self, pool_type: str) -> list:
        """准备股票池"""
        try:
            if pool_type == "all":
                # 获取所有A股
                all_stocks = self.data_manager.get_all_stock_list()
                logger.info(f"使用全市场股票池: {len(all_stocks)} 只股票")
                return all_stocks
            elif pool_type == "hs300":
                # 沪深300成分股
                hs300_stocks = self.data_manager.get_index_components("000300.SH")
                logger.info(f"使用沪深300股票池: {len(hs300_stocks)} 只股票")
                return hs300_stocks
            elif pool_type == "sz50":
                # 上证50成分股
                sz50_stocks = self.data_manager.get_index_components("000016.SH")
                logger.info(f"使用上证50股票池: {len(sz50_stocks)} 只股票")
                return sz50_stocks
            else:
                # 默认使用全市场
                all_stocks = self.data_manager.get_all_stock_list()
                logger.info(f"使用默认股票池: {len(all_stocks)} 只股票")
                return all_stocks
                
        except Exception as e:
            logger.warning(f"准备股票池失败，使用备用方案: {e}")
            # 备用方案：使用硬编码的股票列表
            backup_stocks = [
                {"stock_code": "000001", "stock_name": "平安银行"},
                {"stock_code": "000002", "stock_name": "万科A"},
                {"stock_code": "000858", "stock_name": "五粮液"},
                {"stock_code": "002415", "stock_name": "海康威视"},
                {"stock_code": "600036", "stock_name": "招商银行"},
                {"stock_code": "600519", "stock_name": "贵州茅台"},
                {"stock_code": "600887", "stock_name": "伊利股份"}
            ]
            logger.info(f"使用备用股票池: {len(backup_stocks)} 只股票")
            return backup_stocks
    
    def _execute_strategy_Run_Strategy(self, strategy: dict, execution_date: str) -> pd.DataFrame:
        """执行策略"""
        try:
            # 保存策略到临时文件
            temp_strategy_id = f"temp_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.strategy_manager.save_strategy(temp_strategy_id, strategy)
            
            # 执行策略
            def progress_callback_Strategy_Run_Strategy(progress, message):
                print(f"执行进度: {progress:.1%} - {message}")
            
            selected_stocks = self.strategy_executor.execute_strategy_by_id(
                strategy_id=temp_strategy_id,
                strategy_manager=self.strategy_manager,
                end_date=execution_date,
                progress_callback=progress_callback
            )
            
            # 清理临时策略
            self.strategy_manager.delete_strategy(temp_strategy_id)
            
            return selected_stocks
            
        except Exception as e:
            raise Exception(f"执行策略失败: {e}")
    
    def _filter_results(self, results: pd.DataFrame, max_results: int, min_score: float) -> pd.DataFrame:
        """过滤和排序结果"""
        if results is None or len(results) == 0:
            return pd.DataFrame()
        
        # 按评分过滤
        if 'score' in results.columns and min_score > 0:
            results = results[results['score'] >= min_score]
        
        # 按评分排序
        if 'score' in results.columns:
            results = results.sort_values('score', ascending=False)
        
        # 限制结果数量
        if max_results > 0 and len(results) > max_results:
            results = results.head(max_results)
        
        return results
    
    def export_results(self, results: dict, output_dir: Path, formats: list = None):
        """导出结果"""
        if formats is None:
            formats = ["csv", "json"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出JSON格式
        if "json" in formats:
            json_file = output_dir / f"strategy_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"JSON结果已保存到: {json_file}")
        
        # 导出CSV格式
        if "csv" in formats and results.get("selected_stocks"):
            csv_file = output_dir / f"selected_stocks_{timestamp}.csv"
            df = pd.DataFrame(results["selected_stocks"])
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"CSV结果已保存到: {csv_file}")
        
        # 导出简化报告
        report_file = output_dir / f"strategy_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("策略执行报告\n")
            f.write("=" * 60 + "\n\n")
            
            strategy_info = results.get("strategy_info", {})
            f.write(f"策略名称: {strategy_info.get('name', 'N/A')}\n")
            f.write(f"策略描述: {strategy_info.get('description', 'N/A')}\n")
            f.write(f"条件数量: {strategy_info.get('condition_count', 0)}\n")
            f.write(f"执行日期: {strategy_info.get('execution_date', 'N/A')}\n\n")
            
            summary = results.get("execution_summary", {})
            f.write(f"选出股票数量: {summary.get('selected_count', 0)}\n")
            f.write(f"执行时间: {summary.get('execution_time', 'N/A')}\n\n")
            
            if results.get("selected_stocks"):
                f.write("选股结果 (前10名):\n")
                f.write("-" * 40 + "\n")
                for i, stock in enumerate(results["selected_stocks"][:10], 1):
                    f.write(f"{i:2d}. {stock.get('stock_code', 'N/A')} {stock.get('stock_name', 'N/A')} "
                           f"评分: {stock.get('score', 0):.2f}\n")
        
        print(f"执行报告已保存到: {report_file}")


def main_runstrategy():
    """主函数"""
    parser = argparse.ArgumentParser(description="策略执行工具")
    parser.add_argument("--strategy", required=True, help="策略文件路径")
    parser.add_argument("--output", help="输出目录，默认为results/execution")
    parser.add_argument("--pool", choices=["all", "hs300", "sz50"], default="all", help="股票池类型")
    parser.add_argument("--max-results", type=int, default=50, help="最大结果数量")
    parser.add_argument("--min-score", type=float, default=60, help="最小评分")
    parser.add_argument("--date", help="执行日期，默认为今天")
    parser.add_argument("--format", nargs="+", choices=["csv", "json"], default=["csv", "json"], help="导出格式")
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output) if args.output else Path("results/execution")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置执行日期
    execution_date = args.date or datetime.now().strftime("%Y-%m-%d")
    
    # 执行配置
    config = {
        "execution_date": execution_date,
        "stock_pool": args.pool,
        "max_results": args.max_results,
        "min_score": args.min_score,
        "export_format": args.format
    }
    
    # 创建执行器并运行策略
    runner = StrategyRunner()
    
    print(f"开始执行策略: {args.strategy}")
    print(f"执行日期: {execution_date}")
    print(f"股票池: {args.pool}")
    print(f"最大结果数: {args.max_results}")
    print(f"最小评分: {args.min_score}")
    print("-" * 50)
    
    results = runner.run_strategy(args.strategy, config)
    
    # 导出结果
    runner.export_results(results, output_dir, args.format)
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("📊 执行结果摘要")
    print("=" * 60)
    
    if results["execution_status"] == "success":
        strategy_info = results["strategy_info"]
        summary = results["execution_summary"]
        
        print(f"策略名称: {strategy_info['name']}")
        print(f"条件数量: {strategy_info['condition_count']}")
        print(f"选出股票: {summary['selected_count']}")
        
        if results.get("selected_stocks"):
            print(f"\n前10名选股结果:")
            print("-" * 40)
            for i, stock in enumerate(results["selected_stocks"][:10], 1):
                print(f"{i:2d}. {stock.get('stock_code', 'N/A')} {stock.get('stock_name', 'N/A')} "
                     f"评分: {stock.get('score', 0):.2f}")
    else:
        print(f"执行失败: {results.get('error', '未知错误')}")
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    mainRunstrategy()
