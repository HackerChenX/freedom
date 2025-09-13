#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
风险监控系统

提供全面的风险监控功能，包括：
1. 市场风险评估 - 系统性风险监控
2. 个股风险监控 - 单只股票风险评估
3. 组合风险管理 - 投资组合风险控制
4. 风险预警机制 - 实时风险预警
"""

import time
import threading
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import json
import math

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from indicators.complete_indicator_registry import get_indicator_registry

logger = get_logger(__name__)


class RiskLevel(Enum):
    """风险级别"""
    LOW = "低风险"
    MEDIUM = "中等风险"
    HIGH = "高风险"
    EXTREME = "极高风险"


class RiskType(Enum):
    """风险类型"""
    MARKET_RISK = "市场风险"
    LIQUIDITY_RISK = "流动性风险"
    VOLATILITY_RISK = "波动性风险"
    CONCENTRATION_RISK = "集中度风险"
    CORRELATION_RISK = "相关性风险"
    DRAWDOWN_RISK = "回撤风险"


@dataclass
class RiskMetrics:
    """风险指标数据类"""
    stock_code: str
    stock_name: str
    risk_type: str
    risk_level: str
    risk_score: float
    var_1d: float  # 1日风险价值
    var_5d: float  # 5日风险价值
    volatility: float  # 波动率
    beta: float  # 贝塔系数
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class PortfolioRisk:
    """组合风险数据类"""
    portfolio_id: str
    portfolio_name: str
    total_value: float
    risk_level: str
    portfolio_var: float  # 组合VaR
    portfolio_volatility: float  # 组合波动率
    portfolio_beta: float  # 组合贝塔
    concentration_risk: float  # 集中度风险
    correlation_risk: float  # 相关性风险
    positions: List[Dict[str, Any]]
    timestamp: datetime


class RiskCalculator:
    """风险计算器"""
    
    def __init__(self):
        """初始化风险计算器"""
        self.confidence_level = 0.95  # 置信水平
        self.lookback_period = 252  # 回望期（交易日）
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def calculate_var(self, returns: pd.Series, confidence_level: float = None) -> float:
        """
        计算风险价值(VaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            float: VaR值
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if len(returns) < 30:
            logger.warning("数据量不足，VaR计算可能不准确")
            return 0.0
        
        # 使用历史模拟法计算VaR
        sorted_returns = returns.sort_values()
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var_value = abs(sorted_returns.iloc[var_index])
        
        return var_value
    
    @exception_handler(reraise=True)
    def calculate_volatility(self, returns: pd.Series, annualized: bool = True) -> float:
        """
        计算波动率
        
        Args:
            returns: 收益率序列
            annualized: 是否年化
            
        Returns:
            float: 波动率
        """
        if len(returns) < 2:
            return 0.0
            
        volatility = returns.std()
        
        if annualized:
            volatility *= np.sqrt(252)  # 年化
            
        return volatility
    
    @exception_handler(reraise=True)
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        计算贝塔系数
        
        Args:
            stock_returns: 股票收益率
            market_returns: 市场收益率
            
        Returns:
            float: 贝塔系数
        """
        if len(stock_returns) < 30 or len(market_returns) < 30:
            return 1.0
            
        # 对齐数据
        aligned_data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 30:
            return 1.0
            
        covariance = aligned_data['stock'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        if market_variance == 0:
            return 1.0
            
        beta = covariance / market_variance
        return beta
    
    @exception_handler(reraise=True)
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        计算最大回撤
        
        Args:
            prices: 价格序列
            
        Returns:
            float: 最大回撤比例
        """
        if len(prices) < 2:
            return 0.0
            
        # 计算累计收益
        cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
        
        # 计算历史最高点
        running_max = cumulative.expanding().max()
        
        # 计算回撤
        drawdown = (cumulative - running_max) / running_max
        
        # 返回最大回撤
        max_drawdown = abs(drawdown.min())
        return max_drawdown
    
    @exception_handler(reraise=True)
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            float: 夏普比率
        """
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns.mean() * 252 - risk_free_rate  # 年化超额收益
        volatility = self.calculate_volatility(returns, annualized=True)
        
        if volatility == 0:
            return 0.0
            
        sharpe_ratio = excess_returns / volatility
        return sharpe_ratio


class MarketRiskAssessor:
    """市场风险评估器"""
    
    def __init__(self):
        """初始化市场风险评估器"""
        self.risk_calculator = RiskCalculator()
        self.container = get_container()
        
        # 尝试获取数据访问接口
        try:
            from db.interfaces.data_access_interface import DataAccessInterface
            self.data_access = self.container.resolve(DataAccessInterface)
        except Exception as e:
            logger.warning(f"无法获取数据访问接口: {e}")
            self.data_access = None
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=3.0)
    def assess_market_risk(self, market_index: str = "000001") -> Dict[str, Any]:
        """
        评估市场风险
        
        Args:
            market_index: 市场指数代码
            
        Returns:
            Dict[str, Any]: 市场风险评估结果
        """
        try:
            # 获取市场数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            market_data = self._get_market_data(market_index, start_date, end_date)
            
            if market_data is None or len(market_data) < 30:
                return self._create_default_market_risk()
            
            # 计算收益率
            returns = market_data['close'].pct_change().dropna()
            
            # 计算风险指标
            var_1d = self.risk_calculator.calculate_var(returns)
            var_5d = self.risk_calculator.calculate_var(returns) * np.sqrt(5)
            volatility = self.risk_calculator.calculate_volatility(returns)
            max_drawdown = self.risk_calculator.calculate_max_drawdown(market_data['close'])
            
            # 评估风险级别
            risk_level = self._assess_market_risk_level(volatility, var_1d, max_drawdown)
            
            # 计算风险评分
            risk_score = self._calculate_market_risk_score(volatility, var_1d, max_drawdown)
            
            return {
                'market_index': market_index,
                'risk_level': risk_level.value,
                'risk_score': risk_score,
                'var_1d': var_1d,
                'var_5d': var_5d,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'data_points': len(market_data),
                'assessment_time': datetime.now().isoformat(),
                'details': {
                    'recent_volatility': returns.tail(20).std() * np.sqrt(252),
                    'trend_direction': 'up' if returns.tail(5).mean() > 0 else 'down',
                    'volatility_trend': 'increasing' if returns.tail(20).std() > returns.tail(60).std() else 'decreasing'
                }
            }
            
        except Exception as e:
            logger.error(f"市场风险评估失败: {e}")
            return self._create_default_market_risk()
    
    def _get_market_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取市场数据"""
        if self.data_access is None:
            # 生成模拟数据
            return self._generate_mock_market_data()
        
        try:
            # 使用真实数据访问接口
            query = f"""
            SELECT date, open, high, low, close, volume
            FROM stock_info 
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
            """
            
            data = self.data_access.query_dataframe(query)
            return data
            
        except Exception as e:
            logger.warning(f"获取真实数据失败，使用模拟数据: {e}")
            return self._generate_mock_market_data()
    
    def _generate_mock_market_data(self) -> pd.DataFrame:
        """生成模拟市场数据"""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # 生成模拟价格数据
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [100.0]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        return data.tail(252)  # 返回最近一年的数据
    
    def _assess_market_risk_level(self, volatility: float, var_1d: float, max_drawdown: float) -> RiskLevel:
        """评估市场风险级别"""
        # 风险评分权重
        vol_weight = 0.4
        var_weight = 0.3
        dd_weight = 0.3
        
        # 标准化风险指标
        vol_score = min(volatility / 0.3, 1.0)  # 30%年化波动率为满分
        var_score = min(var_1d / 0.05, 1.0)     # 5%日VaR为满分
        dd_score = min(max_drawdown / 0.3, 1.0) # 30%最大回撤为满分
        
        # 计算综合风险评分
        total_score = vol_score * vol_weight + var_score * var_weight + dd_score * dd_weight
        
        if total_score < 0.3:
            return RiskLevel.LOW
        elif total_score < 0.6:
            return RiskLevel.MEDIUM
        elif total_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def _calculate_market_risk_score(self, volatility: float, var_1d: float, max_drawdown: float) -> float:
        """计算市场风险评分 (0-100)"""
        # 标准化并加权
        vol_score = min(volatility / 0.3 * 100, 100)
        var_score = min(var_1d / 0.05 * 100, 100)
        dd_score = min(max_drawdown / 0.3 * 100, 100)
        
        # 加权平均
        total_score = vol_score * 0.4 + var_score * 0.3 + dd_score * 0.3
        return round(total_score, 2)
    
    def _create_default_market_risk(self) -> Dict[str, Any]:
        """创建默认市场风险评估"""
        return {
            'market_index': '000001',
            'risk_level': RiskLevel.MEDIUM.value,
            'risk_score': 50.0,
            'var_1d': 0.02,
            'var_5d': 0.045,
            'volatility': 0.20,
            'max_drawdown': 0.15,
            'data_points': 0,
            'assessment_time': datetime.now().isoformat(),
            'details': {
                'recent_volatility': 0.22,
                'trend_direction': 'neutral',
                'volatility_trend': 'stable',
                'note': '使用默认风险评估值'
            }
        }


class StockRiskMonitor:
    """个股风险监控器"""

    def __init__(self):
        """初始化个股风险监控器"""
        self.risk_calculator = RiskCalculator()
        self.market_assessor = MarketRiskAssessor()
        self.container = get_container()

        # 尝试获取数据访问接口
        try:
            from db.interfaces.data_access_interface import DataAccessInterface
            self.data_access = self.container.resolve(DataAccessInterface)
        except Exception as e:
            logger.warning(f"无法获取数据访问接口: {e}")
            self.data_access = None

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def monitor_stock_risk(self, stock_code: str, stock_name: str = None) -> RiskMetrics:
        """
        监控个股风险

        Args:
            stock_code: 股票代码
            stock_name: 股票名称

        Returns:
            RiskMetrics: 风险指标
        """
        try:
            # 获取股票数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            stock_data = self._get_stock_data(stock_code, start_date, end_date)
            market_data = self._get_market_data("000001", start_date, end_date)

            if stock_data is None or len(stock_data) < 30:
                return self._create_default_risk_metrics(stock_code, stock_name)

            # 计算收益率
            stock_returns = stock_data['close'].pct_change().dropna()
            market_returns = market_data['close'].pct_change().dropna() if market_data is not None else None

            # 计算风险指标
            var_1d = self.risk_calculator.calculate_var(stock_returns)
            var_5d = self.risk_calculator.calculate_var(stock_returns) * np.sqrt(5)
            volatility = self.risk_calculator.calculate_volatility(stock_returns)
            max_drawdown = self.risk_calculator.calculate_max_drawdown(stock_data['close'])
            sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(stock_returns)

            # 计算贝塔系数
            beta = 1.0
            if market_returns is not None and len(market_returns) > 30:
                beta = self.risk_calculator.calculate_beta(stock_returns, market_returns)

            # 评估风险级别和类型
            risk_level = self._assess_stock_risk_level(volatility, var_1d, max_drawdown, beta)
            risk_type = self._determine_risk_type(volatility, var_1d, beta)

            # 计算风险评分
            risk_score = self._calculate_stock_risk_score(volatility, var_1d, max_drawdown, beta)

            return RiskMetrics(
                stock_code=stock_code,
                stock_name=stock_name or stock_code,
                risk_type=risk_type.value,
                risk_level=risk_level.value,
                risk_score=risk_score,
                var_1d=var_1d,
                var_5d=var_5d,
                volatility=volatility,
                beta=beta,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                timestamp=datetime.now(),
                details={
                    'recent_volatility': stock_returns.tail(20).std() * np.sqrt(252),
                    'price_trend': 'up' if stock_returns.tail(5).mean() > 0 else 'down',
                    'volatility_rank': self._calculate_volatility_rank(volatility),
                    'liquidity_score': self._assess_liquidity(stock_data),
                    'data_quality': 'good' if len(stock_data) > 200 else 'limited'
                }
            )

        except Exception as e:
            logger.error(f"个股风险监控失败 {stock_code}: {e}")
            return self._create_default_risk_metrics(stock_code, stock_name)

    def _get_stock_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        if self.data_access is None:
            return self._generate_mock_stock_data(stock_code)

        try:
            query = f"""
            SELECT date, open, high, low, close, volume, turnover_rate
            FROM stock_info
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
            """

            data = self.data_access.query_dataframe(query)
            return data

        except Exception as e:
            logger.warning(f"获取股票数据失败，使用模拟数据: {e}")
            return self._generate_mock_stock_data(stock_code)

    def _get_market_data(self, market_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取市场数据"""
        return self.market_assessor._get_market_data(market_code, start_date, end_date)

    def _generate_mock_stock_data(self, stock_code: str) -> pd.DataFrame:
        """生成模拟股票数据"""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        np.random.seed(hash(stock_code) % 1000)

        # 根据股票代码生成不同特征的数据
        base_volatility = 0.02 + (hash(stock_code) % 100) / 10000
        returns = np.random.normal(0.0003, base_volatility, len(dates))

        prices = [50.0 + (hash(stock_code) % 100)]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'volume': np.random.randint(100000, 5000000, len(dates)),
            'turnover_rate': np.random.uniform(0.5, 8.0, len(dates))
        })

        return data.tail(252)

    def _assess_stock_risk_level(self, volatility: float, var_1d: float,
                                max_drawdown: float, beta: float) -> RiskLevel:
        """评估个股风险级别"""
        # 多维度风险评分
        vol_score = min(volatility / 0.4, 1.0)      # 40%年化波动率为满分
        var_score = min(var_1d / 0.08, 1.0)         # 8%日VaR为满分
        dd_score = min(max_drawdown / 0.5, 1.0)     # 50%最大回撤为满分
        beta_score = min(abs(beta - 1) / 1.5, 1.0)  # 贝塔偏离1.5为满分

        # 加权计算
        total_score = vol_score * 0.3 + var_score * 0.3 + dd_score * 0.25 + beta_score * 0.15

        if total_score < 0.25:
            return RiskLevel.LOW
        elif total_score < 0.5:
            return RiskLevel.MEDIUM
        elif total_score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME

    def _determine_risk_type(self, volatility: float, var_1d: float, beta: float) -> RiskType:
        """确定主要风险类型"""
        if volatility > 0.35:
            return RiskType.VOLATILITY_RISK
        elif var_1d > 0.06:
            return RiskType.MARKET_RISK
        elif abs(beta) > 1.5:
            return RiskType.CORRELATION_RISK
        else:
            return RiskType.MARKET_RISK

    def _calculate_stock_risk_score(self, volatility: float, var_1d: float,
                                   max_drawdown: float, beta: float) -> float:
        """计算个股风险评分 (0-100)"""
        vol_score = min(volatility / 0.4 * 100, 100)
        var_score = min(var_1d / 0.08 * 100, 100)
        dd_score = min(max_drawdown / 0.5 * 100, 100)
        beta_score = min(abs(beta - 1) / 1.5 * 100, 100)

        total_score = vol_score * 0.3 + var_score * 0.3 + dd_score * 0.25 + beta_score * 0.15
        return round(total_score, 2)

    def _calculate_volatility_rank(self, volatility: float) -> str:
        """计算波动率排名"""
        if volatility < 0.15:
            return "低波动"
        elif volatility < 0.25:
            return "中等波动"
        elif volatility < 0.35:
            return "高波动"
        else:
            return "极高波动"

    def _assess_liquidity(self, stock_data: pd.DataFrame) -> float:
        """评估流动性评分"""
        if 'turnover_rate' in stock_data.columns:
            avg_turnover = stock_data['turnover_rate'].mean()
            return min(avg_turnover / 5.0 * 100, 100)  # 5%换手率为满分
        else:
            avg_volume = stock_data['volume'].mean()
            return min(avg_volume / 1000000 * 20, 100)  # 100万成交量对应20分

    def _create_default_risk_metrics(self, stock_code: str, stock_name: str = None) -> RiskMetrics:
        """创建默认风险指标"""
        return RiskMetrics(
            stock_code=stock_code,
            stock_name=stock_name or stock_code,
            risk_type=RiskType.MARKET_RISK.value,
            risk_level=RiskLevel.MEDIUM.value,
            risk_score=50.0,
            var_1d=0.03,
            var_5d=0.067,
            volatility=0.25,
            beta=1.0,
            max_drawdown=0.20,
            sharpe_ratio=0.5,
            timestamp=datetime.now(),
            details={
                'recent_volatility': 0.27,
                'price_trend': 'neutral',
                'volatility_rank': '中等波动',
                'liquidity_score': 60.0,
                'data_quality': 'default',
                'note': '使用默认风险指标'
            }
        )


class PortfolioRiskManager:
    """组合风险管理器"""

    def __init__(self):
        """初始化组合风险管理器"""
        self.stock_monitor = StockRiskMonitor()
        self.risk_calculator = RiskCalculator()

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def assess_portfolio_risk(self, portfolio: Dict[str, Any]) -> PortfolioRisk:
        """
        评估组合风险

        Args:
            portfolio: 组合信息 {
                'id': '组合ID',
                'name': '组合名称',
                'positions': [
                    {'code': '股票代码', 'name': '股票名称', 'weight': 权重, 'value': 市值},
                    ...
                ]
            }

        Returns:
            PortfolioRisk: 组合风险评估结果
        """
        try:
            positions = portfolio.get('positions', [])
            if not positions:
                return self._create_default_portfolio_risk(portfolio)

            # 计算组合总市值
            total_value = sum(pos.get('value', 0) for pos in positions)

            # 获取各股票的风险指标
            stock_risks = []
            for position in positions:
                stock_code = position.get('code')
                stock_name = position.get('name', stock_code)

                if stock_code:
                    risk_metrics = self.stock_monitor.monitor_stock_risk(stock_code, stock_name)
                    stock_risks.append({
                        'position': position,
                        'risk_metrics': risk_metrics
                    })

            # 计算组合风险指标
            portfolio_var = self._calculate_portfolio_var(stock_risks)
            portfolio_volatility = self._calculate_portfolio_volatility(stock_risks)
            portfolio_beta = self._calculate_portfolio_beta(stock_risks)
            concentration_risk = self._calculate_concentration_risk(positions, total_value)
            correlation_risk = self._calculate_correlation_risk(stock_risks)

            # 评估组合风险级别
            risk_level = self._assess_portfolio_risk_level(
                portfolio_var, portfolio_volatility, concentration_risk, correlation_risk
            )

            return PortfolioRisk(
                portfolio_id=portfolio.get('id', 'unknown'),
                portfolio_name=portfolio.get('name', '未命名组合'),
                total_value=total_value,
                risk_level=risk_level.value,
                portfolio_var=portfolio_var,
                portfolio_volatility=portfolio_volatility,
                portfolio_beta=portfolio_beta,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                positions=[
                    {
                        'code': sr['position']['code'],
                        'name': sr['position'].get('name', sr['position']['code']),
                        'weight': sr['position'].get('weight', 0),
                        'value': sr['position'].get('value', 0),
                        'risk_score': sr['risk_metrics'].risk_score,
                        'risk_level': sr['risk_metrics'].risk_level,
                        'beta': sr['risk_metrics'].beta,
                        'volatility': sr['risk_metrics'].volatility
                    }
                    for sr in stock_risks
                ],
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"组合风险评估失败: {e}")
            return self._create_default_portfolio_risk(portfolio)

    def _calculate_portfolio_var(self, stock_risks: List[Dict]) -> float:
        """计算组合VaR"""
        if not stock_risks:
            return 0.0

        # 简化计算：加权平均VaR
        total_weight = sum(sr['position'].get('weight', 0) for sr in stock_risks)
        if total_weight == 0:
            return 0.0

        weighted_var = sum(
            sr['risk_metrics'].var_1d * sr['position'].get('weight', 0)
            for sr in stock_risks
        ) / total_weight

        # 考虑分散化效应，降低10-30%
        diversification_factor = max(0.7, 1 - len(stock_risks) * 0.05)
        return weighted_var * diversification_factor

    def _calculate_portfolio_volatility(self, stock_risks: List[Dict]) -> float:
        """计算组合波动率"""
        if not stock_risks:
            return 0.0

        # 简化计算：加权平均波动率
        total_weight = sum(sr['position'].get('weight', 0) for sr in stock_risks)
        if total_weight == 0:
            return 0.0

        weighted_volatility = sum(
            sr['risk_metrics'].volatility * sr['position'].get('weight', 0)
            for sr in stock_risks
        ) / total_weight

        # 考虑分散化效应
        diversification_factor = max(0.8, 1 - len(stock_risks) * 0.03)
        return weighted_volatility * diversification_factor

    def _calculate_portfolio_beta(self, stock_risks: List[Dict]) -> float:
        """计算组合贝塔"""
        if not stock_risks:
            return 1.0

        total_weight = sum(sr['position'].get('weight', 0) for sr in stock_risks)
        if total_weight == 0:
            return 1.0

        weighted_beta = sum(
            sr['risk_metrics'].beta * sr['position'].get('weight', 0)
            for sr in stock_risks
        ) / total_weight

        return weighted_beta

    def _calculate_concentration_risk(self, positions: List[Dict], total_value: float) -> float:
        """计算集中度风险"""
        if not positions or total_value == 0:
            return 0.0

        # 计算赫芬达尔指数 (HHI)
        hhi = sum(
            (pos.get('value', 0) / total_value) ** 2
            for pos in positions
        )

        # 转换为风险评分 (0-100)
        # HHI范围：1/n (完全分散) 到 1 (完全集中)
        n = len(positions)
        min_hhi = 1 / n if n > 0 else 1
        concentration_score = (hhi - min_hhi) / (1 - min_hhi) * 100

        return round(concentration_score, 2)

    def _calculate_correlation_risk(self, stock_risks: List[Dict]) -> float:
        """计算相关性风险"""
        if len(stock_risks) < 2:
            return 0.0

        # 简化计算：基于行业和贝塔系数的相关性估计
        betas = [sr['risk_metrics'].beta for sr in stock_risks]
        beta_std = np.std(betas)

        # 贝塔系数标准差越小，相关性风险越高
        correlation_score = max(0, 100 - beta_std * 50)

        return round(correlation_score, 2)

    def _assess_portfolio_risk_level(self, portfolio_var: float, portfolio_volatility: float,
                                   concentration_risk: float, correlation_risk: float) -> RiskLevel:
        """评估组合风险级别"""
        # 多维度风险评分
        var_score = min(portfolio_var / 0.05, 1.0)           # 5%组合VaR为满分
        vol_score = min(portfolio_volatility / 0.3, 1.0)     # 30%组合波动率为满分
        conc_score = concentration_risk / 100                 # 集中度风险已经是0-100
        corr_score = correlation_risk / 100                   # 相关性风险已经是0-100

        # 加权计算
        total_score = var_score * 0.3 + vol_score * 0.3 + conc_score * 0.2 + corr_score * 0.2

        if total_score < 0.3:
            return RiskLevel.LOW
        elif total_score < 0.6:
            return RiskLevel.MEDIUM
        elif total_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME

    def _create_default_portfolio_risk(self, portfolio: Dict[str, Any]) -> PortfolioRisk:
        """创建默认组合风险"""
        return PortfolioRisk(
            portfolio_id=portfolio.get('id', 'unknown'),
            portfolio_name=portfolio.get('name', '未命名组合'),
            total_value=0.0,
            risk_level=RiskLevel.MEDIUM.value,
            portfolio_var=0.03,
            portfolio_volatility=0.20,
            portfolio_beta=1.0,
            concentration_risk=50.0,
            correlation_risk=60.0,
            positions=[],
            timestamp=datetime.now()
        )


class RiskMonitoringSystem:
    """风险监控系统主控制器"""

    def __init__(self):
        """初始化风险监控系统"""
        self.market_assessor = MarketRiskAssessor()
        self.stock_monitor = StockRiskMonitor()
        self.portfolio_manager = PortfolioRiskManager()

        # 监控配置
        self.monitoring_enabled = True
        self.monitoring_interval = 300  # 5分钟
        self.risk_thresholds = {
            'high_risk_score': 75.0,
            'extreme_risk_score': 90.0,
            'high_volatility': 0.35,
            'high_var': 0.06,
            'high_concentration': 80.0
        }

        # 监控线程
        self.monitoring_thread = None
        self.stop_event = threading.Event()

        logger.info("风险监控系统初始化完成")

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def comprehensive_risk_assessment(self, stocks: List[str], portfolios: List[Dict] = None) -> Dict[str, Any]:
        """
        综合风险评估

        Args:
            stocks: 股票代码列表
            portfolios: 组合列表

        Returns:
            Dict[str, Any]: 综合风险评估结果
        """
        assessment_start = time.time()

        # 市场风险评估
        market_risk = self.market_assessor.assess_market_risk()

        # 个股风险评估
        stock_risks = []
        for stock_code in stocks[:20]:  # 限制最多20只股票
            try:
                risk_metrics = self.stock_monitor.monitor_stock_risk(stock_code)
                stock_risks.append(risk_metrics)
            except Exception as e:
                logger.warning(f"股票 {stock_code} 风险评估失败: {e}")

        # 组合风险评估
        portfolio_risks = []
        if portfolios:
            for portfolio in portfolios[:10]:  # 限制最多10个组合
                try:
                    portfolio_risk = self.portfolio_manager.assess_portfolio_risk(portfolio)
                    portfolio_risks.append(portfolio_risk)
                except Exception as e:
                    logger.warning(f"组合风险评估失败: {e}")

        # 生成风险报告
        risk_report = self._generate_risk_report(market_risk, stock_risks, portfolio_risks)

        assessment_time = time.time() - assessment_start
        risk_report['assessment_duration'] = round(assessment_time, 2)

        logger.info(f"综合风险评估完成，耗时 {assessment_time:.2f} 秒")
        return risk_report

    def _generate_risk_report(self, market_risk: Dict, stock_risks: List[RiskMetrics],
                            portfolio_risks: List[PortfolioRisk]) -> Dict[str, Any]:
        """生成风险报告"""
        # 统计分析
        high_risk_stocks = [sr for sr in stock_risks if sr.risk_score >= self.risk_thresholds['high_risk_score']]
        extreme_risk_stocks = [sr for sr in stock_risks if sr.risk_score >= self.risk_thresholds['extreme_risk_score']]

        high_risk_portfolios = [pr for pr in portfolio_risks if pr.risk_level in ['高风险', '极高风险']]

        # 风险预警
        risk_alerts = []

        # 市场风险预警
        if market_risk['risk_score'] >= self.risk_thresholds['high_risk_score']:
            risk_alerts.append({
                'type': '市场风险预警',
                'level': '高风险' if market_risk['risk_score'] < 90 else '极高风险',
                'message': f"市场风险评分达到 {market_risk['risk_score']}，建议谨慎操作",
                'details': market_risk
            })

        # 个股风险预警
        for stock_risk in extreme_risk_stocks:
            risk_alerts.append({
                'type': '个股风险预警',
                'level': '极高风险',
                'message': f"股票 {stock_risk.stock_code} 风险评分达到 {stock_risk.risk_score}",
                'details': asdict(stock_risk)
            })

        # 组合风险预警
        for portfolio_risk in high_risk_portfolios:
            risk_alerts.append({
                'type': '组合风险预警',
                'level': portfolio_risk.risk_level,
                'message': f"组合 {portfolio_risk.portfolio_name} 风险级别为 {portfolio_risk.risk_level}",
                'details': asdict(portfolio_risk)
            })

        return {
            'assessment_time': datetime.now().isoformat(),
            'market_risk': market_risk,
            'stock_risks': {
                'total_count': len(stock_risks),
                'high_risk_count': len(high_risk_stocks),
                'extreme_risk_count': len(extreme_risk_stocks),
                'average_risk_score': round(np.mean([sr.risk_score for sr in stock_risks]), 2) if stock_risks else 0,
                'details': [asdict(sr) for sr in stock_risks]
            },
            'portfolio_risks': {
                'total_count': len(portfolio_risks),
                'high_risk_count': len(high_risk_portfolios),
                'details': [asdict(pr) for pr in portfolio_risks]
            },
            'risk_alerts': risk_alerts,
            'risk_summary': {
                'overall_risk_level': self._calculate_overall_risk_level(market_risk, stock_risks, portfolio_risks),
                'key_risks': self._identify_key_risks(market_risk, stock_risks, portfolio_risks),
                'recommendations': self._generate_recommendations(market_risk, stock_risks, portfolio_risks)
            }
        }

    def _calculate_overall_risk_level(self, market_risk: Dict, stock_risks: List[RiskMetrics],
                                    portfolio_risks: List[PortfolioRisk]) -> str:
        """计算整体风险级别"""
        risk_scores = []

        # 市场风险权重40%
        risk_scores.extend([market_risk['risk_score']] * 4)

        # 个股风险权重40%
        if stock_risks:
            avg_stock_risk = np.mean([sr.risk_score for sr in stock_risks])
            risk_scores.extend([avg_stock_risk] * 4)

        # 组合风险权重20%
        if portfolio_risks:
            # 将风险级别转换为评分
            portfolio_scores = []
            for pr in portfolio_risks:
                if pr.risk_level == '低风险':
                    portfolio_scores.append(25)
                elif pr.risk_level == '中等风险':
                    portfolio_scores.append(50)
                elif pr.risk_level == '高风险':
                    portfolio_scores.append(75)
                else:  # 极高风险
                    portfolio_scores.append(95)

            avg_portfolio_risk = np.mean(portfolio_scores)
            risk_scores.extend([avg_portfolio_risk] * 2)

        if not risk_scores:
            return '中等风险'

        overall_score = np.mean(risk_scores)

        if overall_score < 30:
            return '低风险'
        elif overall_score < 60:
            return '中等风险'
        elif overall_score < 80:
            return '高风险'
        else:
            return '极高风险'

    def _identify_key_risks(self, market_risk: Dict, stock_risks: List[RiskMetrics],
                          portfolio_risks: List[PortfolioRisk]) -> List[str]:
        """识别关键风险"""
        key_risks = []

        # 市场风险
        if market_risk['risk_score'] >= 70:
            key_risks.append(f"市场系统性风险较高 (评分: {market_risk['risk_score']})")

        if market_risk['volatility'] >= self.risk_thresholds['high_volatility']:
            key_risks.append(f"市场波动率过高 ({market_risk['volatility']:.1%})")

        # 个股风险
        high_vol_stocks = [sr for sr in stock_risks if sr.volatility >= self.risk_thresholds['high_volatility']]
        if high_vol_stocks:
            key_risks.append(f"{len(high_vol_stocks)} 只股票波动率过高")

        high_var_stocks = [sr for sr in stock_risks if sr.var_1d >= self.risk_thresholds['high_var']]
        if high_var_stocks:
            key_risks.append(f"{len(high_var_stocks)} 只股票VaR过高")

        # 组合风险
        high_conc_portfolios = [pr for pr in portfolio_risks if pr.concentration_risk >= self.risk_thresholds['high_concentration']]
        if high_conc_portfolios:
            key_risks.append(f"{len(high_conc_portfolios)} 个组合集中度风险过高")

        return key_risks

    def _generate_recommendations(self, market_risk: Dict, stock_risks: List[RiskMetrics],
                                portfolio_risks: List[PortfolioRisk]) -> List[str]:
        """生成风险管理建议"""
        recommendations = []

        # 基于市场风险的建议
        if market_risk['risk_score'] >= 80:
            recommendations.append("市场风险极高，建议减少仓位，增加现金比例")
        elif market_risk['risk_score'] >= 60:
            recommendations.append("市场风险较高，建议谨慎操作，控制仓位")

        # 基于个股风险的建议
        extreme_risk_stocks = [sr for sr in stock_risks if sr.risk_score >= 90]
        if extreme_risk_stocks:
            recommendations.append(f"建议减持或清仓极高风险股票: {', '.join([sr.stock_code for sr in extreme_risk_stocks[:3]])}")

        high_vol_stocks = [sr for sr in stock_risks if sr.volatility >= 0.4]
        if high_vol_stocks:
            recommendations.append("部分股票波动率过高，建议降低仓位或设置止损")

        # 基于组合风险的建议
        high_conc_portfolios = [pr for pr in portfolio_risks if pr.concentration_risk >= 80]
        if high_conc_portfolios:
            recommendations.append("组合集中度过高，建议增加分散化投资")

        high_corr_portfolios = [pr for pr in portfolio_risks if pr.correlation_risk >= 80]
        if high_corr_portfolios:
            recommendations.append("组合相关性风险较高，建议增加不同行业或风格的股票")

        if not recommendations:
            recommendations.append("当前风险水平可控，建议继续监控市场变化")

        return recommendations

    @exception_handler(reraise=True)
    def start_real_time_monitoring(self, stocks: List[str], portfolios: List[Dict] = None,
                                 interval: int = None):
        """
        启动实时风险监控

        Args:
            stocks: 监控的股票列表
            portfolios: 监控的组合列表
            interval: 监控间隔（秒）
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("风险监控已在运行中")
            return

        if interval:
            self.monitoring_interval = interval

        self.monitoring_enabled = True
        self.stop_event.clear()

        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(stocks, portfolios),
            daemon=True
        )
        self.monitoring_thread.start()

        logger.info(f"启动实时风险监控，监控间隔: {self.monitoring_interval} 秒")

    def stop_real_time_monitoring(self):
        """停止实时风险监控"""
        self.monitoring_enabled = False
        self.stop_event.set()

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)

        logger.info("实时风险监控已停止")

    def _monitoring_loop(self, stocks: List[str], portfolios: List[Dict] = None):
        """监控循环"""
        while self.monitoring_enabled and not self.stop_event.is_set():
            try:
                # 执行风险评估
                risk_report = self.comprehensive_risk_assessment(stocks, portfolios)

                # 处理风险预警
                self._process_risk_alerts(risk_report.get('risk_alerts', []))

                # 等待下次监控
                self.stop_event.wait(self.monitoring_interval)

            except Exception as e:
                logger.error(f"风险监控循环出错: {e}")
                self.stop_event.wait(60)  # 出错后等待1分钟再继续

    def _process_risk_alerts(self, risk_alerts: List[Dict]):
        """处理风险预警"""
        for alert in risk_alerts:
            if alert['level'] in ['极高风险', '高风险']:
                logger.warning(f"风险预警: {alert['message']}")
                # 这里可以集成邮件、短信等通知机制

    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'monitoring_interval': self.monitoring_interval,
            'thread_alive': self.monitoring_thread.is_alive() if self.monitoring_thread else False,
            'risk_thresholds': self.risk_thresholds,
            'last_check': datetime.now().isoformat()
        }
