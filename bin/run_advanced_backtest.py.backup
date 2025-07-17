#!/usr/bin/env python3
"""
高级回测系统

本模块实现高级回测功能，支持多策略、多周期和风险管理的回测分析。
"""

import os
import sys
import argparse
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# 使用依赖注入架构
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from utils.decorators import exception_handler, performance_monitor
from utils.logger import get_logger
from strategy.backtester import Backtester
from strategy.strategy_manager import Strategy_manager
from strategy.enhanced_base_strategy import Enhanced_base_strategy
from risk.portfolio_risk import Portfolio_risk_manager
from utils.date_utils import get_trading_day

logger = get_logger(__name__)

class AdvancedBacktestSystem:
    """高级回测系统"""
    
    def __init___14(self):
        """初始化高级回测系统"""
        self.container = get_container()
        self.data_access = self.get_service(Data_access_interface)
        self.strategy_manager = Strategy_manager()
        self.risk_manager = Portfolio_risk_manager()
        self.backtest_results = {}
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def load_stock_universe(self, stock_codes: Optional[List[str]] = None,
                           start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        加载股票池数据
        
        Args:
            stock_codes: 股票代码列表，如果为None则获取全部
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据字典
        """
        try:
            logger.info(f"加载股票池数据，股票数量: {len(stock_codes) if stock_codes else '全部'}")
            
            stock_data = {}
            
            if stock_codes is None:
                # 获取全部股票代码
                stock_codes = self.data_access.get_all_stock_codes()
                logger.info(f"获取到 {len(stock_codes)} 只股票")
            
            for i, code in enumerate(stock_codes, 1):
                try:
                    if i % 100 == 0:
                        logger.info(f"加载进度: {i}/{len(stock_codes)}")
                    
                    data = self.data_access.get_stock_data(
                        code=code,
                        start_date=start_date,
                        end_date=end_date,
                        level='日线'
                    )
                    
                    if not data.empty:
                        stock_data[code] = data
                        
                except Exception as e:
                    logger.warning(f"加载股票 {code} 数据失败: {e}")
                    continue
            
            logger.info(f"成功加载 {len(stock_data)} 只股票的数据")
            return stock_data
            
        except Exception as e:
            logger.error(f"加载股票池数据失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def prepare_backtest_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        准备回测配置
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            Dict[str, Any]: 回测配置
        """
        try:
            if config_file and os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"从文件加载回测配置: {config_file}")
            else:
                # 默认配置
                config = {
                    'initial_capital': 1000000,  # 初始资金100万
                    'commission_rate': 0.0003,   # 手续费率0.03%
                    'slippage_rate': 0.001,      # 滑点率0.1%
                    'max_position_size': 0.1,    # 最大单只股票仓位10%
                    'max_positions': 10,         # 最大持仓数量
                    'stop_loss_rate': 0.1,       # 止损率10%
                    'take_profit_rate': 0.2,     # 止盈率20%
                    'rebalance_frequency': 'weekly',  # 调仓频率
                    'benchmark': '000300.SH'     # 基准指数
                }
                logger.info("使用默认回测配置")
            
            return config
            
        except Exception as e:
            logger.error(f"准备回测配置失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=30.0)
    def run_single_strategy_backtest(self, strategy_name: str, 
                                   stock_data: Dict[str, pd.DataFrame],
                                   config: Dict[str, Any],
                                   start_date: str, end_date: str) -> Dict[str, Any]:
        """
        运行单策略回测
        
        Args:
            strategy_name: 策略名称
            stock_data: 股票数据
            config: 回测配置
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        try:
            logger.info(f"开始运行策略回测: {strategy_name}")
            
            # 创建策略实例
            strategy = self.strategy_manager.create_strategy(strategy_name, config)
            
            # 创建回测器
            backtester = Backtester(
                initial_capital=config['initial_capital'],
                commission_rate=config['commission_rate'],
                slippage_rate=config['slippage_rate']
            )
            
            # 运行回测
            backtest_result = backtester.run_backtest(
                strategy=strategy,
                stock_data=stock_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # 计算性能指标
            performance_metrics = self._calculate_performance_metrics(
                backtest_result, config.get('benchmark')
            )
            
            # 风险分析
            risk_analysis = self.risk_manager.analyze_portfolio_risk(
                backtest_result['portfolio_history']
            )
            
            result = {
                'strategy_name': strategy_name,
                'backtest_period': f"{start_date} - {end_date}",
                'backtest_result': backtest_result,
                'performance_metrics': performance_metrics,
                'risk_analysis': risk_analysis,
                'config': config
            }
            
            logger.info(f"策略 {strategy_name} 回测完成")
            return result
            
        except Exception as e:
            logger.error(f"运行策略回测失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=60.0)
    def run_multi_strategy_backtest(self, strategy_names: List[str],
                                  stock_data: Dict[str, pd.DataFrame],
                                  config: Dict[str, Any],
                                  start_date: str, end_date: str) -> Dict[str, Dict[str, Any]]:
        """
        运行多策略回测
        
        Args:
            strategy_names: 策略名称列表
            stock_data: 股票数据
            config: 回测配置
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Dict[str, Any]]: 多策略回测结果
        """
        results = {}
        
        logger.info(f"开始运行多策略回测，策略数量: {len(strategy_names)}")
        
        for i, strategy_name in enumerate(strategy_names, 1):
            try:
                logger.info(f"回测策略 {i}/{len(strategy_names)}: {strategy_name}")
                
                result = self.run_single_strategy_backtest(
                    strategy_name=strategy_name,
                    stock_data=stock_data,
                    config=config,
                    start_date=start_date,
                    end_date=end_date
                )
                
                results[strategy_name] = result
                
            except Exception as e:
                logger.error(f"策略 {strategy_name} 回测失败: {e}")
                continue
        
        # 生成策略对比分析
        comparison_analysis = self._generate_strategy_comparison(results)
        results['_comparison_analysis'] = comparison_analysis
        
        logger.info(f"多策略回测完成，成功回测 {len(results)-1} 个策略")
        return results
    
    def _calculate_performance_metrics(self, backtest_result: Dict[str, Any], 
                                     benchmark: Optional[str] = None) -> Dict[str, float]:
        """计算性能指标"""
        try:
            portfolio_values = backtest_result['portfolio_values']
            
            if len(portfolio_values) < 2:
                return {}
            
            # 计算收益率序列
            returns = portfolio_values.pct_change().dropna()
            
            # 基础指标
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
            volatility = returns.std() * (252 ** 0.5)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # 胜率
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': backtest_result.get('total_trades', 0),
                'profitable_trades': backtest_result.get('profitable_trades', 0)
            }
            
            # 如果有基准，计算相对指标
            if benchmark:
                try:
                    benchmark_data = self.data_access.get_stock_data(
                        code=benchmark,
                        start_date=portfolio_values.index[0].strftime('%Y-%m-%d'),
                        end_date=portfolio_values.index[-1].strftime('%Y-%m-%d')
                    )
                    
                    if not benchmark_data.empty:
                        benchmark_returns = benchmark_data['close'].pct_change().dropna()
                        benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
                        
                        metrics['alpha'] = annual_return - benchmark_annual_return
                        metrics['information_ratio'] = metrics['alpha'] / volatility if volatility > 0 else 0
                        
                except Exception as e:
                    logger.warning(f"计算基准指标失败: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
            return {}
    
    def _generate_strategy_comparison(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成策略对比分析"""
        try:
            if len(results) < 2:
                return {}
            
            comparison = {
                'strategy_ranking': [],
                'performance_summary': {},
                'best_strategies': {},
                'risk_comparison': {}
            }
            
            # 提取性能指标
            strategy_metrics = {}
            for strategy_name, result in results.items():
                if strategy_name.startswith('_'):
                    continue
                metrics = result.get('performance_metrics', {})
                strategy_metrics[strategy_name] = metrics
            
            # 按不同指标排序
            ranking_criteria = ['total_return', 'sharpe_ratio', 'annual_return', 'max_drawdown']
            
            for criterion in ranking_criteria:
                if criterion == 'max_drawdown':
                    # 最大回撤越小越好
                    sorted_strategies = sorted(
                        strategy_metrics.items(),
                        key=lambda x: x[1].get(criterion, -1),
                        reverse=True
                    )
                else:
                    # 其他指标越大越好
                    sorted_strategies = sorted(
                        strategy_metrics.items(),
                        key=lambda x: x[1].get(criterion, 0),
                        reverse=True
                    )
                
                comparison['best_strategies'][criterion] = sorted_strategies[0][0] if sorted_strategies else None
            
            # 综合排名（基于多个指标的加权平均）
            weighted_scores = {}
            weights = {'total_return': 0.3, 'sharpe_ratio': 0.4, 'max_drawdown': 0.3}
            
            for strategy_name, metrics in strategy_metrics.items():
                score = 0
                for metric, weight in weights.items():
                    value = metrics.get(metric, 0)
                    if metric == 'max_drawdown':
                        # 最大回撤转换为正向评分
                        normalized_value = max(0, 1 + value)  # 回撤是负值
                    else:
                        normalized_value = max(0, value)
                    score += normalized_value * weight
                
                weighted_scores[strategy_name] = score
            
            comparison['strategy_ranking'] = sorted(
                weighted_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"生成策略对比分析失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    def save_backtest_results(self, results: Dict[str, Any], 
                            output_file: str) -> None:
        """
        保存回测结果
        
        Args:
            results: 回测结果
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存为JSON格式（需要处理不可序列化的对象）
            serializable_results = self._make_serializable(results)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"回测结果已保存到: {output_file}")
            
            # 生成汇总报告
            summary_file = output_file.replace('.json', '_summary.txt')
            self._generate_backtest_summary(results, summary_file)
            
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
            raise
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def _generate_backtest_summary(self, results: Dict[str, Any], 
                                 summary_file: str) -> None:
        """生成回测汇总报告"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("高级回测系统汇总报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # 如果是多策略回测
                if '_comparison_analysis' in results:
                    comparison = results['_comparison_analysis']
                    strategy_count = len(results) - 1
                    
                    f.write(f"回测策略数量: {strategy_count}\n\n")
                    
                    # 策略排名
                    if comparison.get('strategy_ranking'):
                        f.write("策略综合排名:\n")
                        f.write("-" * 40 + "\n")
                        for i, (strategy, score) in enumerate(comparison['strategy_ranking'][:10], 1):
                            f.write(f"{i:2d}. {strategy}: {score:.4f}\n")
                        f.write("\n")
                    
                    # 各指标最佳策略
                    if comparison.get('best_strategies'):
                        f.write("各指标最佳策略:\n")
                        f.write("-" * 40 + "\n")
                        for metric, strategy in comparison['best_strategies'].items():
                            f.write(f"{metric}: {strategy}\n")
                        f.write("\n")
                    
                    # 详细性能指标
                    f.write("详细性能指标:\n")
                    f.write("-" * 40 + "\n")
                    for strategy_name, result in results.items():
                        if strategy_name.startswith('_'):
                            continue
                        
                        metrics = result.get('performance_metrics', {})
                        f.write(f"\n{strategy_name}:\n")
                        f.write(f"  总收益率: {metrics.get('total_return', 0):.2%}\n")
                        f.write(f"  年化收益率: {metrics.get('annual_return', 0):.2%}\n")
                        f.write(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}\n")
                        f.write(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}\n")
                        f.write(f"  胜率: {metrics.get('win_rate', 0):.2%}\n")
                        f.write(f"  交易次数: {metrics.get('total_trades', 0)}\n")
                
                else:
                    # 单策略回测
                    strategy_name = results.get('strategy_name', '未知策略')
                    metrics = results.get('performance_metrics', {})
                    
                    f.write(f"回测策略: {strategy_name}\n")
                    f.write(f"回测期间: {results.get('backtest_period', '未知')}\n\n")
                    
                    f.write("性能指标:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"总收益率: {metrics.get('total_return', 0):.2%}\n")
                    f.write(f"年化收益率: {metrics.get('annual_return', 0):.2%}\n")
                    f.write(f"波动率: {metrics.get('volatility', 0):.2%}\n")
                    f.write(f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f}\n")
                    f.write(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}\n")
                    f.write(f"胜率: {metrics.get('win_rate', 0):.2%}\n")
                    f.write(f"总交易次数: {metrics.get('total_trades', 0)}\n")
                    f.write(f"盈利交易次数: {metrics.get('profitable_trades', 0)}\n")
                    
                    if 'alpha' in metrics:
                        f.write(f"Alpha: {metrics['alpha']:.2%}\n")
                    if 'information_ratio' in metrics:
                        f.write(f"信息比率: {metrics['information_ratio']:.3f}\n")
                
            logger.info(f"回测汇总报告已生成: {summary_file}")
            
        except Exception as e:
            logger.error(f"生成回测汇总报告失败: {e}")

@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=300.0)
def main_28():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级回测系统')
    parser.add_argument('--strategies', type=str, nargs='+', 
                       help='策略名称列表')
    parser.add_argument('--codes', type=str, nargs='+',
                       help='股票代码列表（可选，默认全市场）')
    parser.add_argument('--start-date', type=str, 
                       default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                       help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='回测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--config', type=str,
                       help='回测配置文件路径')
    parser.add_argument('--output', type=str, 
                       default='data/result/advanced_backtest_results.json',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    try:
        # 初始化回测系统
        backtest_system = Advanced_backtest_system()
        
        # 准备回测配置
        config = backtest_system.prepare_backtest_config(args.config)
        
        # 加载股票数据
        logger.info("开始加载股票数据...")
        stock_data = backtest_system.load_stock_universe(
            stock_codes=args.codes,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if not stock_data:
            logger.error("未加载到任何股票数据")
            sys.exit(1)
        
        # 获取策略列表
        if args.strategies:
            strategy_names = args.strategies
        else:
            # 默认策略列表
            strategy_names = ['momentum_strategy', 'mean_reversion_strategy', 'breakout_strategy']
        
        logger.info(f"开始高级回测")
        logger.info(f"策略列表: {strategy_names}")
        logger.info(f"股票数量: {len(stock_data)}")
        logger.info(f"回测期间: {args.start_date} - {args.end_date}")
        
        # 运行回测
        if len(strategy_names) == 1:
            # 单策略回测
            results = backtest_system.run_single_strategy_backtest(
                strategy_name=strategy_names[0],
                stock_data=stock_data,
                config=config,
                start_date=args.start_date,
                end_date=args.end_date
            )
        else:
            # 多策略回测
            results = backtest_system.run_multi_strategy_backtest(
                strategy_names=strategy_names,
                stock_data=stock_data,
                config=config,
                start_date=args.start_date,
                end_date=args.end_date
            )
        
        # 保存结果
        backtest_system.save_backtest_results(results, args.output)
        
        logger.info("高级回测完成")
        
        # 输出简要结果
        if '_comparison_analysis' in results:
            comparison = results['_comparison_analysis']
            if comparison.get('strategy_ranking'):
                print(f"\n最佳策略: {comparison['strategy_ranking'][0][0]}")
                print(f"综合评分: {comparison['strategy_ranking'][0][1]:.4f}")
        else:
            metrics = results.get('performance_metrics', {})
            print(f"\n策略: {results.get('strategy_name', '未知')}")
            print(f"总收益率: {metrics.get('total_return', 0):.2%}")
            print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
        
    except Exception as e:
        logger.error(f"高级回测失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_28() 