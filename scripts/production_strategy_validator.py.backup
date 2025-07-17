#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生产环境策略验证工具

基于ClickHouse真实数据验证选股策略的可靠性
支持逐个指标验证和批量验证，以最新交易日期作为测试基准
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
import time
import warnings
import logging
import traceback
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IDataAccess
from utils.decorators import exception_handler, performance_monitor
from utils.logger import get_logger
from utils.date_utils import get_trading_day
from strategy.strategy_manager import StrategyManager
from strategy.backtester import Backtester
from risk.portfolio_risk import PortfolioRiskManager
from monitoring.performance_monitor import PerformanceMonitor

logger = get_logger(__name__)


class ProductionStrategyValidator:
    """生产环境策略验证器"""
    
    def __init__(self):
        """初始化生产环境策略验证器"""
        self.container = get_container()
        self.data_access = self.get_service(DataAccessInterface)
        self.strategy_manager = StrategyManager()
        self.risk_manager = PortfolioRiskManager()
        self.performance_monitor = PerformanceMonitor()
        self.validation_results = {}
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=30.0)
    def validate_strategy_performance(self, strategy_name: str, 
                                    validation_period: int = 30) -> Dict[str, Any]:
        """
        验证策略性能
        
        Args:
            strategy_name: 策略名称
            validation_period: 验证期间（天数）
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            logger.info(f"开始验证策略性能: {strategy_name}")
            
            # 获取验证期间
            end_date = get_trading_day()
            start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=validation_period*2)).strftime('%Y%m%d')
            
            # 加载策略
            strategy = self.strategy_manager.load_strategy(strategy_name)
            
            # 获取验证数据
            validation_data = self._load_validation_data(start_date, end_date)
            
            # 运行策略验证
            strategy_results = self._run_strategy_validation(strategy, validation_data, start_date, end_date)
            
            # 性能分析
            performance_metrics = self._analyze_strategy_performance(strategy_results)
            
            # 风险评估
            risk_assessment = self._assess_strategy_risk(strategy_results)
            
            # 稳定性检查
            stability_check = self._check_strategy_stability(strategy_results)
            
            # 合规性检查
            compliance_check = self._check_strategy_compliance(strategy, strategy_results)
            
            validation_result = {
                'strategy_name': strategy_name,
                'validation_period': f"{start_date} - {end_date}",
                'validation_timestamp': datetime.now().isoformat(),
                'strategy_results': strategy_results,
                'performance_metrics': performance_metrics,
                'risk_assessment': risk_assessment,
                'stability_check': stability_check,
                'compliance_check': compliance_check,
                'overall_status': self._determine_overall_status(
                    performance_metrics, risk_assessment, stability_check, compliance_check
                )
            }
            
            logger.info(f"策略 {strategy_name} 验证完成")
            return validation_result
            
        except Exception as e:
            logger.error(f"验证策略性能失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=15.0)
    def _load_validation_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """加载验证数据"""
        try:
            logger.info("加载验证数据")
            
            # 获取股票池
            stock_codes = self.data_access.get_active_stock_codes()
            
            validation_data = {}
            failed_loads = 0
            
            for i, code in enumerate(stock_codes[:100], 1):  # 限制为前100只股票进行验证
                try:
                    if i % 20 == 0:
                        logger.info(f"数据加载进度: {i}/{min(100, len(stock_codes))}")
                    
                    data = self.data_access.get_stock_data(
                        code=code,
                        start_date=datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d'),
                        end_date=datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d'),
                        level='日线'
                    )
                    
                    if not data.empty and len(data) >= 20:  # 确保有足够的数据
                        validation_data[code] = data
                    else:
                        failed_loads += 1
                        
                except Exception as e:
                    logger.warning(f"加载股票 {code} 数据失败: {e}")
                    failed_loads += 1
                    continue
            
            logger.info(f"成功加载 {len(validation_data)} 只股票数据，失败 {failed_loads} 只")
            
            if len(validation_data) < 10:
                raise ValueError("验证数据不足，无法进行有效验证")
            
            return validation_data
            
        except Exception as e:
            logger.error(f"加载验证数据失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=20.0)
    def _run_strategy_validation(self, strategy, validation_data: Dict[str, pd.DataFrame],
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """运行策略验证"""
        try:
            logger.info("运行策略验证")
            
            # 创建回测器
            backtester = Backtester(
                initial_capital=1000000,  # 100万初始资金
                commission_rate=0.0003,   # 0.03%手续费
                slippage_rate=0.001       # 0.1%滑点
            )
            
            # 运行回测
            backtest_result = backtester.run_backtest(
                strategy=strategy,
                stock_data=validation_data,
                start_date=datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d'),
                end_date=datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            )
            
            # 提取关键结果
            strategy_results = {
                'backtest_result': backtest_result,
                'portfolio_values': backtest_result.get('portfolio_values', pd.Series()),
                'trades': backtest_result.get('trades', []),
                'positions': backtest_result.get('positions', {}),
                'daily_returns': backtest_result.get('daily_returns', pd.Series()),
                'total_trades': len(backtest_result.get('trades', [])),
                'profitable_trades': len([t for t in backtest_result.get('trades', []) if t.get('profit', 0) > 0])
            }
            
            logger.info("策略验证运行完成")
            return strategy_results
            
        except Exception as e:
            logger.error(f"运行策略验证失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    def _analyze_strategy_performance(self, strategy_results: Dict[str, Any]) -> Dict[str, float]:
        """分析策略性能"""
        try:
            portfolio_values = strategy_results.get('portfolio_values', pd.Series())
            daily_returns = strategy_results.get('daily_returns', pd.Series())
            
            if portfolio_values.empty or len(portfolio_values) < 2:
                return {'error': 'insufficient_data'}
            
            # 基础性能指标
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
            
            if not daily_returns.empty:
                volatility = daily_returns.std() * (252 ** 0.5)
                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                
                # 最大回撤
                cumulative_returns = (1 + daily_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdowns.min()
                
                # 胜率
                win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
                win_rate = 0
            
            # 交易指标
            total_trades = strategy_results.get('total_trades', 0)
            profitable_trades = strategy_results.get('profitable_trades', 0)
            profit_ratio = profitable_trades / total_trades if total_trades > 0 else 0
            
            performance_metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'profit_ratio': profit_ratio,
                'avg_trade_return': total_return / total_trades if total_trades > 0 else 0
            }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"分析策略性能失败: {e}")
            return {'error': str(e)}
    
    @exception_handler(reraise=True)
    def _assess_strategy_risk(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估策略风险"""
        try:
            portfolio_values = strategy_results.get('portfolio_values', pd.Series())
            positions = strategy_results.get('positions', {})
            
            # 使用风险管理器进行风险评估
            risk_metrics = self.risk_manager.analyze_portfolio_risk(portfolio_values)
            
            # 持仓集中度风险
            if positions:
                position_values = list(positions.values())
                total_value = sum(position_values)
                concentration_risk = max(position_values) / total_value if total_value > 0 else 0
            else:
                concentration_risk = 0
            
            # 流动性风险评估
            liquidity_risk = self._assess_liquidity_risk(strategy_results)
            
            # 市场风险评估
            market_risk = self._assess_market_risk(strategy_results)
            
            risk_assessment = {
                'risk_metrics': risk_metrics,
                'concentration_risk': concentration_risk,
                'liquidity_risk': liquidity_risk,
                'market_risk': market_risk,
                'overall_risk_level': self._determine_risk_level(
                    risk_metrics, concentration_risk, liquidity_risk, market_risk
                )
            }
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"评估策略风险失败: {e}")
            return {'error': str(e)}
    
    def _assess_liquidity_risk(self, strategy_results: Dict[str, Any]) -> Dict[str, float]:
        """评估流动性风险"""
        try:
            trades = strategy_results.get('trades', [])
            
            # 计算平均交易规模
            trade_sizes = [trade.get('quantity', 0) for trade in trades]
            avg_trade_size = np.mean(trade_sizes) if trade_sizes else 0
            
            # 计算交易频率
            trade_frequency = len(trades) / 30 if trades else 0  # 每月交易次数
            
            return {
                'avg_trade_size': avg_trade_size,
                'trade_frequency': trade_frequency,
                'liquidity_score': min(100, max(0, 100 - avg_trade_size * trade_frequency / 1000))
            }
            
        except Exception as e:
            logger.error(f"评估流动性风险失败: {e}")
            return {'liquidity_score': 50}
    
    def _assess_market_risk(self, strategy_results: Dict[str, Any]) -> Dict[str, float]:
        """评估市场风险"""
        try:
            daily_returns = strategy_results.get('daily_returns', pd.Series())
            
            if daily_returns.empty:
                return {'market_beta': 0, 'market_correlation': 0}
            
            # 获取市场基准数据
            try:
                market_data = self.data_access.get_stock_data(
                    code='000300.SH',  # 沪深300作为基准
                    start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                if not market_data.empty:
                    market_returns = market_data['close'].pct_change().dropna()
                    
                    # 对齐时间序列
                    aligned_returns = daily_returns.align(market_returns, join='inner')
                    strategy_aligned, market_aligned = aligned_returns
                    
                    if len(strategy_aligned) > 10:
                        # 计算Beta
                        covariance = np.cov(strategy_aligned, market_aligned)[0, 1]
                        market_variance = np.var(market_aligned)
                        beta = covariance / market_variance if market_variance > 0 else 0
                        
                        # 计算相关性
                        correlation = np.corrcoef(strategy_aligned, market_aligned)[0, 1]
                        
                        return {
                            'market_beta': beta,
                            'market_correlation': correlation,
                            'systematic_risk': abs(beta) * 100
                        }
                        
            except Exception as e:
                logger.warning(f"获取市场基准数据失败: {e}")
            
            return {'market_beta': 0, 'market_correlation': 0, 'systematic_risk': 0}
            
        except Exception as e:
            logger.error(f"评估市场风险失败: {e}")
            return {'market_beta': 0, 'market_correlation': 0}
    
    def _determine_risk_level(self, risk_metrics: Dict, concentration_risk: float,
                            liquidity_risk: Dict, market_risk: Dict) -> str:
        """确定风险水平"""
        try:
            risk_score = 0
            
            # 集中度风险评分
            if concentration_risk > 0.3:
                risk_score += 30
            elif concentration_risk > 0.2:
                risk_score += 20
            elif concentration_risk > 0.1:
                risk_score += 10
            
            # 流动性风险评分
            liquidity_score = liquidity_risk.get('liquidity_score', 50)
            if liquidity_score < 30:
                risk_score += 25
            elif liquidity_score < 50:
                risk_score += 15
            elif liquidity_score < 70:
                risk_score += 10
            
            # 市场风险评分
            systematic_risk = market_risk.get('systematic_risk', 0)
            if systematic_risk > 150:
                risk_score += 25
            elif systematic_risk > 100:
                risk_score += 15
            elif systematic_risk > 50:
                risk_score += 10
            
            # 确定风险等级
            if risk_score >= 60:
                return "高风险"
            elif risk_score >= 40:
                return "中高风险"
            elif risk_score >= 20:
                return "中等风险"
            else:
                return "低风险"
                
        except Exception as e:
            logger.error(f"确定风险水平失败: {e}")
            return "未知风险"
    
    @exception_handler(reraise=True)
    def _check_strategy_stability(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """检查策略稳定性"""
        try:
            portfolio_values = strategy_results.get('portfolio_values', pd.Series())
            daily_returns = strategy_results.get('daily_returns', pd.Series())
            
            stability_metrics = {}
            
            if not portfolio_values.empty:
                # 收益稳定性
                returns_std = daily_returns.std() if not daily_returns.empty else 0
                stability_metrics['returns_stability'] = max(0, 100 - returns_std * 100)
                
                # 回撤稳定性
                cumulative_returns = (1 + daily_returns).cumprod() if not daily_returns.empty else pd.Series([1])
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                max_consecutive_loss_days = self._calculate_max_consecutive_loss_days(daily_returns)
                
                stability_metrics['drawdown_stability'] = max(0, 100 + drawdowns.min() * 100)
                stability_metrics['max_consecutive_loss_days'] = max_consecutive_loss_days
                
                # 交易稳定性
                trades = strategy_results.get('trades', [])
                if trades:
                    trade_returns = [trade.get('return', 0) for trade in trades]
                    trade_consistency = 100 - (np.std(trade_returns) * 100) if trade_returns else 0
                    stability_metrics['trade_consistency'] = max(0, min(100, trade_consistency))
                else:
                    stability_metrics['trade_consistency'] = 0
            
            # 综合稳定性评分
            if stability_metrics:
                avg_stability = np.mean(list(stability_metrics.values()))
                stability_level = self._get_stability_level(avg_stability)
            else:
                avg_stability = 0
                stability_level = "不稳定"
            
            return {
                'stability_metrics': stability_metrics,
                'overall_stability_score': avg_stability,
                'stability_level': stability_level
            }
            
        except Exception as e:
            logger.error(f"检查策略稳定性失败: {e}")
            return {'error': str(e)}
    
    def _calculate_max_consecutive_loss_days(self, daily_returns: pd.Series) -> int:
        """计算最大连续亏损天数"""
        try:
            if daily_returns.empty:
                return 0
            
            loss_days = daily_returns < 0
            max_consecutive = 0
            current_consecutive = 0
            
            for is_loss in loss_days:
                if is_loss:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            logger.error(f"计算最大连续亏损天数失败: {e}")
            return 0
    
    def _get_stability_level(self, stability_score: float) -> str:
        """获取稳定性等级"""
        if stability_score >= 80:
            return "非常稳定"
        elif stability_score >= 60:
            return "稳定"
        elif stability_score >= 40:
            return "一般稳定"
        elif stability_score >= 20:
            return "不太稳定"
        else:
            return "不稳定"
    
    @exception_handler(reraise=True)
    def _check_strategy_compliance(self, strategy, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """检查策略合规性"""
        try:
            compliance_results = {}
            
            # 检查持仓限制
            positions = strategy_results.get('positions', {})
            max_position_ratio = max(positions.values()) / sum(positions.values()) if positions else 0
            compliance_results['position_limit_check'] = {
                'max_position_ratio': max_position_ratio,
                'compliant': max_position_ratio <= 0.1,  # 单只股票不超过10%
                'limit': 0.1
            }
            
            # 检查交易频率
            trades = strategy_results.get('trades', [])
            daily_trade_count = len(trades) / 30  # 平均每日交易次数
            compliance_results['trading_frequency_check'] = {
                'daily_trade_count': daily_trade_count,
                'compliant': daily_trade_count <= 10,  # 每日交易不超过10次
                'limit': 10
            }
            
            # 检查风险控制
            portfolio_values = strategy_results.get('portfolio_values', pd.Series())
            if not portfolio_values.empty:
                max_loss = (portfolio_values.min() / portfolio_values.iloc[0]) - 1
                compliance_results['risk_control_check'] = {
                    'max_loss': max_loss,
                    'compliant': max_loss >= -0.2,  # 最大亏损不超过20%
                    'limit': -0.2
                }
            
            # 综合合规性
            all_compliant = all(
                check.get('compliant', False) 
                for check in compliance_results.values()
            )
            
            compliance_results['overall_compliance'] = {
                'compliant': all_compliant,
                'compliance_score': sum(
                    1 for check in compliance_results.values() 
                    if check.get('compliant', False)
                ) / len(compliance_results) * 100 if compliance_results else 0
            }
            
            return compliance_results
            
        except Exception as e:
            logger.error(f"检查策略合规性失败: {e}")
            return {'error': str(e)}
    
    def _determine_overall_status(self, performance_metrics: Dict, risk_assessment: Dict,
                                stability_check: Dict, compliance_check: Dict) -> Dict[str, Any]:
        """确定整体状态"""
        try:
            status_score = 0
            issues = []
            
            # 性能评分
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio > 1.5:
                status_score += 30
            elif sharpe_ratio > 1.0:
                status_score += 20
            elif sharpe_ratio > 0.5:
                status_score += 10
            else:
                issues.append("夏普比率偏低")
            
            # 风险评分
            risk_level = risk_assessment.get('overall_risk_level', '未知风险')
            if risk_level == '低风险':
                status_score += 25
            elif risk_level == '中等风险':
                status_score += 15
            elif risk_level == '中高风险':
                status_score += 5
            else:
                issues.append(f"风险水平较高: {risk_level}")
            
            # 稳定性评分
            stability_score = stability_check.get('overall_stability_score', 0)
            if stability_score > 80:
                status_score += 25
            elif stability_score > 60:
                status_score += 15
            elif stability_score > 40:
                status_score += 10
            else:
                issues.append("策略稳定性不足")
            
            # 合规性评分
            compliance_score = compliance_check.get('overall_compliance', {}).get('compliance_score', 0)
            if compliance_score == 100:
                status_score += 20
            elif compliance_score >= 80:
                status_score += 15
            elif compliance_score >= 60:
                status_score += 10
            else:
                issues.append("存在合规性问题")
            
            # 确定状态
            if status_score >= 80:
                status = "优秀"
            elif status_score >= 60:
                status = "良好"
            elif status_score >= 40:
                status = "一般"
            elif status_score >= 20:
                status = "需要改进"
            else:
                status = "不合格"
            
            return {
                'status': status,
                'status_score': status_score,
                'issues': issues,
                'recommendation': self._generate_recommendation_Production_Strategy_Validator(status, issues)
            }
            
        except Exception as e:
            logger.error(f"确定整体状态失败: {e}")
            return {'status': '评估失败', 'error': str(e)}
    
    def _generate_recommendation_Production_Strategy_Validator(self, status: str, issues: List[str]) -> str:
        """生成建议"""
        if status == "优秀":
            return "策略表现优秀，可以继续使用"
        elif status == "良好":
            return "策略表现良好，建议持续监控"
        elif status == "一般":
            return "策略表现一般，建议优化参数"
        elif status == "需要改进":
            return f"策略需要改进，主要问题: {', '.join(issues[:3])}"
        else:
            return f"策略不合格，不建议使用。问题: {', '.join(issues[:3])}"
    
    @exception_handler(reraise=True)
    def validate_multiple_strategies(self, strategy_names: List[str],
                                   validation_period: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        验证多个策略
        
        Args:
            strategy_names: 策略名称列表
            validation_period: 验证期间（天数）
            
        Returns:
            Dict[str, Dict[str, Any]]: 多策略验证结果
        """
        try:
            logger.info(f"开始验证多个策略，数量: {len(strategy_names)}")
            
            results = {}
            
            for i, strategy_name in enumerate(strategy_names, 1):
                try:
                    logger.info(f"验证策略 {i}/{len(strategy_names)}: {strategy_name}")
                    
                    result = self.validate_strategy_performance(strategy_name, validation_period)
                    results[strategy_name] = result
                    
                except Exception as e:
                    logger.error(f"验证策略 {strategy_name} 失败: {e}")
                    results[strategy_name] = {
                        'strategy_name': strategy_name,
                        'error': str(e),
                        'overall_status': {'status': '验证失败'}
                    }
                    continue
            
            # 生成对比分析
            comparison_analysis = self._generate_strategy_comparison_Production_Strategy_Validator(results)
            results['_comparison_analysis'] = comparison_analysis
            
            logger.info(f"多策略验证完成，成功验证 {len([r for r in results.values() if 'error' not in r])} 个策略")
            return results
            
        except Exception as e:
            logger.error(f"验证多个策略失败: {e}")
            raise
    
    def _generate_strategy_comparison_Production_Strategy_Validator(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成策略对比分析"""
        try:
            valid_results = {k: v for k, v in results.items() if 'error' not in v and not k.startswith('_')}
            
            if len(valid_results) < 2:
                return {}
            
            # 提取性能指标
            strategy_metrics = {}
            for strategy_name, result in valid_results.items():
                metrics = result.get('performance_metrics', {})
                overall_status = result.get('overall_status', {})
                
                strategy_metrics[strategy_name] = {
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'annual_return': metrics.get('annual_return', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'status_score': overall_status.get('status_score', 0),
                    'status': overall_status.get('status', '未知')
                }
            
            # 排名
            rankings = {
                'by_sharpe_ratio': sorted(strategy_metrics.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True),
                'by_annual_return': sorted(strategy_metrics.items(), key=lambda x: x[1]['annual_return'], reverse=True),
                'by_status_score': sorted(strategy_metrics.items(), key=lambda x: x[1]['status_score'], reverse=True)
            }
            
            # 最佳策略
            best_overall = rankings['by_status_score'][0] if rankings['by_status_score'] else None
            
            return {
                'strategy_count': len(valid_results),
                'rankings': rankings,
                'best_overall_strategy': best_overall[0] if best_overall else None,
                'summary_statistics': {
                    'avg_sharpe_ratio': np.mean([m['sharpe_ratio'] for m in strategy_metrics.values()]),
                    'avg_annual_return': np.mean([m['annual_return'] for m in strategy_metrics.values()]),
                    'excellent_strategies': len([m for m in strategy_metrics.values() if m['status'] == '优秀']),
                    'good_strategies': len([m for m in strategy_metrics.values() if m['status'] == '良好'])
                }
            }
            
        except Exception as e:
            logger.error(f"生成策略对比分析失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    def save_validation_results(self, results: Dict[str, Any], 
                              output_file: str) -> None:
        """
        保存验证结果
        
        Args:
            results: 验证结果
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 转换为可序列化格式
            serializable_results = self._make_serializable_Production_Strategy_Validator(results)
            
            # 保存为JSON格式
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"验证结果已保存到: {output_file}")
            
            # 生成汇总报告
            summary_file = output_file.replace('.json', '_summary.txt')
            self._generate_validation_summary(results, summary_file)
            
        except Exception as e:
            logger.error(f"保存验证结果失败: {e}")
            raise
    
    def _make_serializable_Production_Strategy_Validator(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable_Production_Strategy_Validator(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable_Production_Strategy_Validator(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    def _generate_validation_summary(self, results: Dict[str, Any], 
                                   summary_file: str) -> None:
        """生成验证汇总报告"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("生产环境策略验证汇总报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # 如果是多策略验证
                if '_comparison_analysis' in results:
                    comparison = results['_comparison_analysis']
                    strategy_count = len(results) - 1
                    
                    f.write(f"验证策略数量: {strategy_count}\n\n")
                    
                    # 最佳策略
                    best_strategy = comparison.get('best_overall_strategy')
                    if best_strategy:
                        f.write(f"最佳策略: {best_strategy}\n\n")
                    
                    # 策略状态统计
                    summary_stats = comparison.get('summary_statistics', {})
                    f.write("策略状态统计:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"优秀策略: {summary_stats.get('excellent_strategies', 0)} 个\n")
                    f.write(f"良好策略: {summary_stats.get('good_strategies', 0)} 个\n")
                    f.write(f"平均夏普比率: {summary_stats.get('avg_sharpe_ratio', 0):.3f}\n")
                    f.write(f"平均年化收益率: {summary_stats.get('avg_annual_return', 0):.2%}\n\n")
                    
                    # 详细验证结果
                    f.write("详细验证结果:\n")
                    f.write("-" * 40 + "\n")
                    for strategy_name, result in results.items():
                        if strategy_name.startswith('_'):
                            continue
                        
                        if 'error' in result:
                            f.write(f"\n{strategy_name}: 验证失败 - {result['error']}\n")
                            continue
                        
                        overall_status = result.get('overall_status', {})
                        performance = result.get('performance_metrics', {})
                        
                        f.write(f"\n{strategy_name}:\n")
                        f.write(f"  状态: {overall_status.get('status', '未知')}\n")
                        f.write(f"  评分: {overall_status.get('status_score', 0):.1f}\n")
                        f.write(f"  夏普比率: {performance.get('sharpe_ratio', 0):.3f}\n")
                        f.write(f"  年化收益率: {performance.get('annual_return', 0):.2%}\n")
                        f.write(f"  最大回撤: {performance.get('max_drawdown', 0):.2%}\n")
                        
                        issues = overall_status.get('issues', [])
                        if issues:
                            f.write(f"  问题: {', '.join(issues[:2])}\n")
                
                else:
                    # 单策略验证
                    strategy_name = results.get('strategy_name', '未知策略')
                    overall_status = results.get('overall_status', {})
                    performance = results.get('performance_metrics', {})
                    
                    f.write(f"验证策略: {strategy_name}\n")
                    f.write(f"验证期间: {results.get('validation_period', '未知')}\n\n")
                    
                    f.write("验证结果:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"整体状态: {overall_status.get('status', '未知')}\n")
                    f.write(f"状态评分: {overall_status.get('status_score', 0):.1f}\n")
                    f.write(f"夏普比率: {performance.get('sharpe_ratio', 0):.3f}\n")
                    f.write(f"年化收益率: {performance.get('annual_return', 0):.2%}\n")
                    f.write(f"最大回撤: {performance.get('max_drawdown', 0):.2%}\n")
                    f.write(f"胜率: {performance.get('win_rate', 0):.2%}\n")
                    
                    recommendation = overall_status.get('recommendation', '')
                    if recommendation:
                        f.write(f"\n建议: {recommendation}\n")
                
            logger.info(f"验证汇总报告已生成: {summary_file}")
            
        except Exception as e:
            logger.error(f"生成验证汇总报告失败: {e}")

@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=180.0)
def main_productionstrategyvalidator():
    """主函数"""
    parser = argparse.ArgumentParser(description='生产环境策略验证器')
    parser.add_argument('--strategies', type=str, nargs='+', 
                       help='策略名称列表')
    parser.add_argument('--period', type=int, default=30,
                       help='验证期间（天数）')
    parser.add_argument('--output', type=str, 
                       default='data/result/strategy_validation_results.json',
                       help='输出文件路径')
    parser.add_argument('--single', type=str,
                       help='验证单个策略')
    
    args = parser.parse_args()
    
    try:
        # 初始化验证器
        validator = ProductionStrategyValidator()
        
        # 获取策略列表
        if args.single:
            strategy_names = [args.single]
        elif args.strategies:
            strategy_names = args.strategies
        else:
            # 默认策略列表
            strategy_names = ['momentum_strategy', 'mean_reversion_strategy']
        
        logger.info(f"开始生产环境策略验证")
        logger.info(f"策略列表: {strategy_names}")
        logger.info(f"验证期间: {args.period} 天")
        
        # 运行验证
        if len(strategy_names) == 1:
            # 单策略验证
            results = validator.validate_strategy_performance(
                strategy_name=strategy_names[0],
                validation_period=args.period
            )
        else:
            # 多策略验证
            results = validator.validate_multiple_strategies(
                strategy_names=strategy_names,
                validation_period=args.period
            )
        
        # 保存结果
        validator.save_validation_results(results, args.output)
        
        logger.info("生产环境策略验证完成")
        
        # 输出简要结果
        if args.single:
            overall_status = results.get('overall_status', {})
            print(f"\n策略: {args.single}")
            print(f"状态: {overall_status.get('status', '未知')}")
            print(f"评分: {overall_status.get('status_score', 0):.1f}")
        else:
            comparison = results.get('_comparison_analysis', {})
            best_strategy = comparison.get('best_overall_strategy')
            if best_strategy:
                print(f"\n最佳策略: {best_strategy}")
        
    except Exception as e:
        logger.error(f"生产环境策略验证失败: {e}")
        print(f"验证失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    mainProductionstrategyvalidator() 