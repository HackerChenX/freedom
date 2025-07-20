#!/usr/bin/env python3
"""
生产级高性能选股系统

集成所有性能优化的生产级选股主程序：
1. 批量数据库查询
2. 多进程并行计算
3. 向量化指标计算
4. 智能缓存策略

作者：AI Assistant
创建时间：2025-01-13
"""

import sys
import time
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from strategy.strategy_executor import UnifiedStrategyExecutor as get_high_performance_executor
from strategy.strategy_parser import StrategyParser
from db.unified_data_manager import get_unified_data_manager

logger = get_logger(__name__)

def get_all_stock_codes(target_date: str) -> List[str]:
    """
    获取所有股票代码
    
    Args:
        target_date: 目标日期
        
    Returns:
        List[str]: 股票代码列表
    """
    try:
        data_manager = get_unified_data_manager()
        
        # 使用优化的查询获取所有股票代码
        query = f"""
        SELECT DISTINCT code 
        FROM stock_info 
        WHERE date = '{target_date}' 
        AND level = '日线'
        ORDER BY code
        """
        
        from db.clickhouse_db import get_clickhouse_db
        db = get_clickhouse_db()
        result = db.execute(query)
        
        if result:
            # 处理不同类型的返回结果
            if isinstance(result, set):
                return list(result)
            elif isinstance(result, list):
                return [row[0] if isinstance(row, (list, tuple)) else row for row in result]
            else:
                return [result] if result else []
        else:
            logger.warning(f"未找到{target_date}的股票数据")
            return []
            
    except Exception as e:
        logger.error(f"获取股票代码列表失败: {e}")
        return []

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生产级高性能选股系统')
    parser.add_argument('--strategy', required=True, help='策略配置文件路径')
    parser.add_argument('--date', required=True, help='目标日期 (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=50, help='最大选股数量')
    parser.add_argument('--workers', type=int, help='并行工作进程数')
    parser.add_argument('--batch-size', type=int, default=100, help='批处理大小')
    parser.add_argument('--output', help='结果输出文件路径')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = time.time()
    
    try:
        logger.info(f"🚀 启动生产级高性能选股系统")
        logger.info(f"   策略配置: {args.strategy}")
        logger.info(f"   目标日期: {args.date}")
        logger.info(f"   选股限制: {args.limit}")
        
        # 1. 解析策略配置
        logger.info("📋 解析策略配置...")
        strategy_parser = StrategyParser()
        
        # 加载YAML文件
        import yaml
        with open(args.strategy, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        strategy_config = strategy_parser.parse_strategy(config_data)
        logger.info(f"   策略ID: {strategy_config.get('strategy', {}).get('id', 'Unknown')}")
        
        # 2. 获取股票代码列表
        logger.info("📊 获取股票代码列表...")
        stock_codes = get_all_stock_codes(args.date)
        if not stock_codes:
            logger.error("❌ 未找到任何股票数据")
            return
        logger.info(f"   股票总数: {len(stock_codes)}")
        
        # 3. 初始化高性能执行器
        executor_config = {}
        if args.workers:
            executor_config['max_workers'] = args.workers
        if args.batch_size:
            executor_config['batch_size'] = args.batch_size
            
        executor = get_high_performance_executor(**executor_config)
        
        # 4. 执行高性能选股
        logger.info("🧮 开始高性能选股计算...")
        selected_stocks = executor.execute_strategy(
            strategy_config=strategy_config,
            stock_codes=stock_codes,
            target_date=args.date,
            limit=args.limit
        )
        
        # 5. 输出结果
        execution_time = time.time() - start_time
        stats = executor.get_performance_stats()
        
        print(f"\n{'='*60}")
        print(f"🎯 生产级高性能选股完成")
        print(f"{'='*60}")
        print(f"总股票数量: {stats['total_stocks']:,}")
        print(f"处理股票数量: {stats['processed_stocks']:,}")
        print(f"选出股票数量: {stats['selected_stocks']:,}")
        print(f"选股成功率: {stats['selection_rate']:.2f}%")
        print(f"执行时间: {execution_time:.2f}秒")
        print(f"处理速度: {stats['stocks_per_second']:.1f} 股票/秒")
        print(f"缓存命中率: {stats['cache_hit_rate']:.1f}%")
        print(f"批量查询数: {stats['batch_queries']}")
        
        if selected_stocks:
            print(f"\n📈 选中股票:")
            print(f"{'代码':<10} {'XG值':<6} {'V11_EMA':<8} {'V12':<8} {'AA':<4} {'BB':<4}")
            print("-" * 50)
            for stock in selected_stocks[:20]:  # 显示前20只
                print(f"{stock['stock_code']:<10} {stock['xg_value']:<6} "
                      f"{stock['v11_ema']:<8.2f} {stock['v12']:<8.2f} "
                      f"{'✓' if stock['aa_signal'] else '✗':<4} "
                      f"{'✓' if stock['bb_signal'] else '✗':<4}")
            
            if len(selected_stocks) > 20:
                print(f"... 还有 {len(selected_stocks) - 20} 只股票")
        else:
            print(f"\n⚠️  未找到符合条件的股票")
        
        # 6. 保存结果
        if args.output or selected_stocks:
            output_file = args.output or f"high_performance_results_{args.date.replace('-', '')}.json"
            
            result_data = {
                'strategy_config': strategy_config,
                'execution_date': args.date,
                'performance_stats': stats,
                'execution_time': execution_time,
                'selected_stocks': selected_stocks
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"💾 结果已保存到: {output_file}")
        
        print(f"\n✅ 高性能选股系统执行完成!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断执行")
    except Exception as e:
        logger.error(f"❌ 系统执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        logger.info(f"总执行时间: {total_time:.2f}秒")

if __name__ == "__main__":
    main() 