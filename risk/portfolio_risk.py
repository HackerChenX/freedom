#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
组合风险评估系统

基于技术指标的组合风险评估，计算组合级别风险指标，提供风险预警
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import logging

from utils.logger import get_logger
from risk.warning_system import RiskWarningSystem, RiskLevel

logger = get_logger(__name__)


class PortfolioRiskType(Enum):
    """组合风险类型枚举"""
    CONCENTRATION = 0      # 集中度风险
    CORRELATION = 1        # 相关性风险
    DRAWDOWN = 2           # 回撤风险
    LIQUIDITY = 3          # 流动性风险
    SECTOR = 4             # 行业风险
    VOLATILITY = 5         # 波动率风险
    TECHNICAL = 6          # 技术指标风险


class PortfolioRisk:
    """
    组合风险评估系统
    
    基于技术指标评估组合风险，识别风险来源，提供风险缓解建议
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化组合风险评估系统
        
        Args:
            params: 配置参数字典，可包含以下键:
                - lookback_period: 回溯期，默认为60
                - risk_threshold: 高风险阈值，默认为70
                - extreme_threshold: 极端风险阈值，默认为90
                - correlation_threshold: 相关性警戒阈值，默认为0.7
                - max_concentration: 最大持仓集中度，默认为0.2 (20%)
                - risk_weights: 各风险维度权重，默认为均等权重
        """
        self._params = params or {}
        self._initialize_params()
        self.warning_system = RiskWarningSystem()
        
    def _initialize_params(self):
        """初始化参数，设置默认值"""
        # 基础参数
        self.lookback_period = self._params.get('lookback_period', 60)
        self.risk_threshold = self._params.get('risk_threshold', 70)
        self.extreme_threshold = self._params.get('extreme_threshold', 90)
        self.correlation_threshold = self._params.get('correlation_threshold', 0.7)
        self.max_concentration = self._params.get('max_concentration', 0.2)
        
        # 风险维度权重
        default_weights = {
            'concentration': 0.15,  # 集中度风险权重
            'correlation': 0.15,    # 相关性风险权重
            'drawdown': 0.2,        # 回撤风险权重
            'liquidity': 0.1,       # 流动性风险权重
            'sector': 0.1,          # 行业风险权重
            'volatility': 0.15,     # 波动率风险权重
            'technical': 0.15       # 技术指标风险权重
        }
        self.risk_weights = self._params.get('risk_weights', default_weights) 

    def evaluate_portfolio_risk(self, holdings: Dict[str, float], stock_data: Dict[str, pd.DataFrame],
                             sector_info: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        评估组合风险
        
        Args:
            holdings: 持仓字典，键为股票代码，值为持仓权重（占比）
            stock_data: 股票数据字典，键为股票代码，值为包含OHLCV数据的DataFrame
            sector_info: 行业信息字典，键为股票代码，值为行业名称
            
        Returns:
            包含各风险维度评分和综合风险评分的字典
        """
        # 验证输入数据
        if not holdings or not stock_data:
            logger.error("Empty holdings or stock data provided")
            raise ValueError("Holdings and stock data must not be empty")
        
        # 检查持仓总和是否接近1
        holdings_sum = sum(holdings.values())
        if abs(holdings_sum - 1.0) > 0.01:
            logger.warning(f"Holdings weights do not sum to 1 (sum: {holdings_sum:.2f}). Normalizing weights.")
            # 归一化持仓权重
            holdings = {code: weight / holdings_sum for code, weight in holdings.items()}
        
        # 检查所有持仓股票是否都有数据
        missing_data = [code for code in holdings if code not in stock_data]
        if missing_data:
            logger.error(f"Missing data for stocks: {missing_data}")
            raise ValueError(f"Missing stock data for: {missing_data}")
        
        # 计算各风险维度分数
        concentration_risk = self._calculate_concentration_risk(holdings)
        correlation_risk = self._calculate_correlation_risk(holdings, stock_data)
        drawdown_risk = self._calculate_drawdown_risk(holdings, stock_data)
        liquidity_risk = self._calculate_liquidity_risk(holdings, stock_data)
        sector_risk = self._calculate_sector_risk(holdings, sector_info) if sector_info else 0
        volatility_risk = self._calculate_volatility_risk(holdings, stock_data)
        technical_risk = self._calculate_technical_risk(holdings, stock_data)
        
        # 计算综合风险分数
        portfolio_risk_score = (
            self.risk_weights['concentration'] * concentration_risk +
            self.risk_weights['correlation'] * correlation_risk +
            self.risk_weights['drawdown'] * drawdown_risk +
            self.risk_weights['liquidity'] * liquidity_risk +
            self.risk_weights['sector'] * sector_risk +
            self.risk_weights['volatility'] * volatility_risk +
            self.risk_weights['technical'] * technical_risk
        )
        
        # 根据风险分数分级
        risk_level = self._get_risk_level(portfolio_risk_score)
        
        # 生成风险来源和缓解建议
        risk_sources = self._identify_risk_sources({
            'concentration': concentration_risk,
            'correlation': correlation_risk,
            'drawdown': drawdown_risk,
            'liquidity': liquidity_risk,
            'sector': sector_risk,
            'volatility': volatility_risk,
            'technical': technical_risk
        })
        
        mitigation_suggestions = self._generate_mitigation_suggestions(risk_sources, holdings, stock_data)
        
        # 返回结果
        return {
            'portfolio_risk_score': portfolio_risk_score,
            'risk_level': risk_level,
            'risk_details': {
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'drawdown_risk': drawdown_risk,
                'liquidity_risk': liquidity_risk,
                'sector_risk': sector_risk,
                'volatility_risk': volatility_risk,
                'technical_risk': technical_risk
            },
            'risk_sources': risk_sources,
            'mitigation_suggestions': mitigation_suggestions
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """根据风险分数获取风险级别"""
        if risk_score < 20:
            return RiskLevel.SAFE.name
        elif risk_score < 40:
            return RiskLevel.ATTENTION.name
        elif risk_score < 60:
            return RiskLevel.CAUTION.name
        elif risk_score < 75:
            return RiskLevel.WARNING.name
        elif risk_score < 90:
            return RiskLevel.DANGER.name
        else:
            return RiskLevel.EXTREME.name
    
    def _identify_risk_sources(self, risk_scores: Dict[str, float]) -> List[str]:
        """识别主要风险来源"""
        # 找出高于阈值的风险维度
        high_risk_dimensions = []
        for dimension, score in risk_scores.items():
            if score >= self.risk_threshold:
                if score >= self.extreme_threshold:
                    high_risk_dimensions.append(f"{dimension}(极高)")
                else:
                    high_risk_dimensions.append(f"{dimension}(高)")
        
        return high_risk_dimensions
    
    def _generate_mitigation_suggestions(self, risk_sources: List[str], 
                                        holdings: Dict[str, float], 
                                        stock_data: Dict[str, pd.DataFrame]) -> List[str]:
        """生成风险缓解建议"""
        suggestions = []
        
        # 根据风险来源生成相应建议
        for risk_source in risk_sources:
            dimension = risk_source.split('(')[0]
            
            if dimension == 'concentration':
                # 找出权重过高的股票
                high_concentration_stocks = [code for code, weight in holdings.items() if weight > self.max_concentration]
                if high_concentration_stocks:
                    suggestions.append(f"降低高集中度股票权重: {', '.join(high_concentration_stocks)}")
                suggestions.append("增加投资组合多样性，分散持仓到更多股票")
            
            elif dimension == 'correlation':
                suggestions.append("增加低相关性资产，降低组合内股票间的相关性")
                suggestions.append("考虑加入防御性资产，如黄金、债券或具有负相关性的ETF")
            
            elif dimension == 'drawdown':
                suggestions.append("设置更严格的止损策略，控制单只股票的最大回撤")
                suggestions.append("增加低波动性资产比例，提高组合稳定性")
            
            elif dimension == 'liquidity':
                suggestions.append("减少流动性较差的股票持仓比例")
                suggestions.append("避免在低流动性时段进行大额交易")
            
            elif dimension == 'sector':
                suggestions.append("降低对单一行业的过度暴露，增加行业多样性")
                suggestions.append("考虑在不同经济周期表现各异的行业进行配置")
            
            elif dimension == 'volatility':
                suggestions.append("增加低波动性股票比例，减少高波动性股票持仓")
                suggestions.append("考虑使用期权或其他衍生品进行波动率对冲")
            
            elif dimension == 'technical':
                # 分析各持仓股票的技术指标风险
                high_tech_risk_stocks = []
                for code, data in stock_data.items():
                    if code in holdings:
                        # 使用风险预警系统评估技术风险
                        risk_df = self.warning_system.classify_risk_level(data)
                        if not risk_df.empty and risk_df['risk_level'].iloc[-1] in [RiskLevel.WARNING.name, RiskLevel.DANGER.name, RiskLevel.EXTREME.name]:
                            high_tech_risk_stocks.append(code)
                
                if high_tech_risk_stocks:
                    suggestions.append(f"关注技术风险较高的股票: {', '.join(high_tech_risk_stocks)}")
                suggestions.append("对技术指标预警的股票设置更严格的止损位")
        
        return suggestions 

    def _calculate_concentration_risk(self, holdings: Dict[str, float]) -> float:
        """
        计算集中度风险
        
        Args:
            holdings: 持仓字典，键为股票代码，值为持仓权重
            
        Returns:
            集中度风险分数 (0-100)
        """
        # 计算赫芬达尔指数 (Herfindahl Index)，越大表示集中度越高
        herfindahl_index = sum(weight ** 2 for weight in holdings.values())
        
        # 标准化为0-100分（0.1为充分分散，1为完全集中）
        normalized_score = (herfindahl_index - 0.1) / 0.9 * 100
        
        # 限制在0-100范围内
        concentration_risk = max(0, min(100, normalized_score))
        
        # 考虑单一持仓过高的情况
        max_weight = max(holdings.values()) if holdings else 0
        if max_weight > self.max_concentration:
            # 根据超出程度增加风险分数
            excess_concentration = (max_weight - self.max_concentration) / (1 - self.max_concentration)
            concentration_risk += excess_concentration * 30
        
        return min(100, concentration_risk)
    
    def _calculate_correlation_risk(self, holdings: Dict[str, float], stock_data: Dict[str, pd.DataFrame]) -> float:
        """
        计算相关性风险
        
        Args:
            holdings: 持仓字典
            stock_data: 股票数据字典
            
        Returns:
            相关性风险分数 (0-100)
        """
        # 如果持仓少于2只股票，无法计算相关性
        if len(holdings) < 2:
            return 0
        
        # 提取收盘价数据
        prices = {}
        for code in holdings:
            if code in stock_data and 'close' in stock_data[code].columns:
                prices[code] = stock_data[code]['close']
        
        # 如果有效数据少于2只股票，无法计算相关性
        if len(prices) < 2:
            return 0
        
        # 创建价格DataFrame
        price_df = pd.DataFrame(prices)
        
        # 计算收益率
        returns_df = price_df.pct_change().dropna()
        
        # 计算相关性矩阵
        correlation_matrix = returns_df.corr()
        
        # 计算平均相关系数（排除对角线上的1）
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                correlations.append(correlation_matrix.iloc[i, j])
        
        avg_correlation = sum(correlations) / len(correlations) if correlations else 0
        
        # 计算高相关性比例（相关系数大于阈值的比例）
        high_correlation_count = sum(1 for corr in correlations if corr > self.correlation_threshold)
        high_correlation_ratio = high_correlation_count / len(correlations) if correlations else 0
        
        # 基于平均相关系数和高相关性比例计算风险分数
        # 相关系数范围从-1到1，转换为0-100分
        avg_correlation_score = (avg_correlation + 1) / 2 * 100
        high_correlation_score = high_correlation_ratio * 100
        
        # 综合评分（更重视高相关性比例）
        correlation_risk = avg_correlation_score * 0.4 + high_correlation_score * 0.6
        
        return correlation_risk
    
    def _calculate_drawdown_risk(self, holdings: Dict[str, float], stock_data: Dict[str, pd.DataFrame]) -> float:
        """
        计算回撤风险
        
        Args:
            holdings: 持仓字典
            stock_data: 股票数据字典
            
        Returns:
            回撤风险分数 (0-100)
        """
        # 计算每只股票的最大回撤
        max_drawdowns = {}
        for code, weight in holdings.items():
            if code in stock_data and 'close' in stock_data[code].columns:
                prices = stock_data[code]['close'].values
                max_drawdown = self._calculate_max_drawdown(prices)
                max_drawdowns[code] = max_drawdown
        
        # 计算加权平均最大回撤
        weighted_drawdown = 0
        total_weight = 0
        for code, drawdown in max_drawdowns.items():
            if code in holdings:
                weighted_drawdown += drawdown * holdings[code]
                total_weight += holdings[code]
        
        weighted_drawdown = weighted_drawdown / total_weight if total_weight > 0 else 0
        
        # 转换为风险分数 (0-100)
        # 假设20%回撤对应50分，40%回撤对应100分
        drawdown_risk = min(100, weighted_drawdown * 250)
        
        return drawdown_risk
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """
        计算最大回撤
        
        Args:
            prices: 价格数组
            
        Returns:
            最大回撤比例
        """
        # 如果价格数据不足，返回0
        if len(prices) < 2:
            return 0
        
        # 计算累计最大值
        max_so_far = prices[0]
        max_drawdown = 0
        
        for price in prices[1:]:
            if price > max_so_far:
                max_so_far = price
            else:
                drawdown = (max_so_far - price) / max_so_far
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_liquidity_risk(self, holdings: Dict[str, float], stock_data: Dict[str, pd.DataFrame]) -> float:
        """
        计算流动性风险
        
        Args:
            holdings: 持仓字典
            stock_data: 股票数据字典
            
        Returns:
            流动性风险分数 (0-100)
        """
        # 计算每只股票的平均成交量
        avg_volumes = {}
        for code, weight in holdings.items():
            if code in stock_data and 'volume' in stock_data[code].columns:
                # 使用最近N天的平均成交量
                recent_volumes = stock_data[code]['volume'].tail(self.lookback_period)
                avg_volume = recent_volumes.mean() if not recent_volumes.empty else 0
                avg_volumes[code] = avg_volume
        
        # 计算流动性风险
        liquidity_risks = {}
        for code, avg_volume in avg_volumes.items():
            # 这里假设日均成交量低于100万为低流动性，高于1000万为高流动性
            if avg_volume < 1_000_000:
                liquidity_risks[code] = 100  # 极低流动性
            elif avg_volume < 5_000_000:
                liquidity_risks[code] = 75  # 低流动性
            elif avg_volume < 10_000_000:
                liquidity_risks[code] = 50  # 中等流动性
            elif avg_volume < 50_000_000:
                liquidity_risks[code] = 25  # 高流动性
            else:
                liquidity_risks[code] = 0  # 极高流动性
        
        # 计算加权平均流动性风险
        weighted_liquidity_risk = 0
        total_weight = 0
        for code, risk in liquidity_risks.items():
            if code in holdings:
                weighted_liquidity_risk += risk * holdings[code]
                total_weight += holdings[code]
        
        weighted_liquidity_risk = weighted_liquidity_risk / total_weight if total_weight > 0 else 0
        
        return weighted_liquidity_risk
    
    def _calculate_sector_risk(self, holdings: Dict[str, float], sector_info: Dict[str, str]) -> float:
        """
        计算行业风险
        
        Args:
            holdings: 持仓字典
            sector_info: 行业信息字典
            
        Returns:
            行业风险分数 (0-100)
        """
        # 计算各行业权重
        sector_weights = {}
        for code, weight in holdings.items():
            if code in sector_info:
                sector = sector_info[code]
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        # 计算最大行业权重
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        
        # 计算赫芬达尔指数
        herfindahl_index = sum(weight ** 2 for weight in sector_weights.values())
        
        # 标准化为0-100分
        # 假设5个行业均等分布为理想状态（HI=0.2）
        normalized_hi_score = (herfindahl_index - 0.2) / 0.8 * 100
        normalized_hi_score = max(0, min(100, normalized_hi_score))
        
        # 考虑单一行业过高的情况
        # 假设单一行业占比超过40%开始有风险
        max_weight_score = 0
        if max_sector_weight > 0.4:
            max_weight_score = (max_sector_weight - 0.4) / 0.6 * 100
        
        # 综合评分
        sector_risk = normalized_hi_score * 0.6 + max_weight_score * 0.4
        
        return sector_risk
    
    def _calculate_volatility_risk(self, holdings: Dict[str, float], stock_data: Dict[str, pd.DataFrame]) -> float:
        """
        计算波动率风险
        
        Args:
            holdings: 持仓字典
            stock_data: 股票数据字典
            
        Returns:
            波动率风险分数 (0-100)
        """
        # 计算每只股票的波动率
        volatilities = {}
        for code, weight in holdings.items():
            if code in stock_data and 'close' in stock_data[code].columns:
                # 计算收益率
                returns = stock_data[code]['close'].pct_change().dropna()
                # 计算收益率标准差（波动率）
                if len(returns) > 0:
                    volatility = returns.std() * (252 ** 0.5)  # 年化波动率
                    volatilities[code] = volatility
        
        # 计算加权平均波动率
        weighted_volatility = 0
        total_weight = 0
        for code, volatility in volatilities.items():
            if code in holdings:
                weighted_volatility += volatility * holdings[code]
                total_weight += holdings[code]
        
        weighted_volatility = weighted_volatility / total_weight if total_weight > 0 else 0
        
        # 转换为风险分数 (0-100)
        # 假设20%波动率对应50分，40%波动率对应100分
        volatility_risk = min(100, weighted_volatility * 250)
        
        return volatility_risk
    
    def _calculate_technical_risk(self, holdings: Dict[str, float], stock_data: Dict[str, pd.DataFrame]) -> float:
        """
        计算基于技术指标的风险分数
        
        Args:
            holdings: 持仓字典
            stock_data: 股票数据字典
            
        Returns:
            技术指标风险分数 (0-100)
        """
        technical_risks = []
        
        for code, weight in holdings.items():
            if code in stock_data:
                df = stock_data[code]
                
                # 使用风险预警系统评估单只股票的风险
                risk_df = self.warning_system.classify_risk_level(df)
                
                # 获取最新的风险评分
                latest_risk = risk_df['risk_score'].iloc[-1] if not risk_df.empty else 50
                
                # 根据持仓权重加权
                technical_risks.append(latest_risk * weight)
        
        # 计算加权平均技术风险
        return sum(technical_risks) if technical_risks else 50
    
    def calculate_drawdown_indicator_correlation(self, portfolio_returns: pd.Series, 
                                             risk_scores: pd.Series,
                                             lookback_period: int = 60) -> Dict[str, float]:
        """
        计算组合回撤与技术指标风险的相关性
        
        Args:
            portfolio_returns: 组合日收益率序列
            risk_scores: 技术指标风险评分序列
            lookback_period: 回溯期
            
        Returns:
            相关性指标字典
        """
        if len(portfolio_returns) < lookback_period or len(risk_scores) < lookback_period:
            logger.warning(f"历史数据不足 {lookback_period} 个周期，无法准确计算相关性")
            return {'correlation': 0, 'predictive_power': 0, 'lead_lag_correlation': 0}
        
        # 计算组合累计收益曲线
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # 计算滚动最大值
        rolling_max = cumulative_returns.rolling(window=lookback_period, min_periods=1).max()
        
        # 计算回撤序列
        drawdown_series = (cumulative_returns / rolling_max - 1) * 100  # 转为百分比
        
        # 确保两个序列长度相同且索引对齐
        aligned_data = pd.DataFrame({
            'drawdown': drawdown_series,
            'risk_score': risk_scores
        }).dropna()
        
        if aligned_data.empty:
            return {'correlation': 0, 'predictive_power': 0, 'lead_lag_correlation': 0}
        
        # 1. 计算当期相关性
        current_correlation = aligned_data['drawdown'].corr(aligned_data['risk_score'])
        
        # 2. 计算风险评分对未来回撤的预测能力（前瞻性相关）
        # 风险评分提前1-5天的相关性
        forward_correlations = []
        for i in range(1, 6):
            if i >= len(aligned_data):
                break
            shifted_risk = aligned_data['risk_score'].shift(i)
            forward_corr = aligned_data['drawdown'].corr(shifted_risk)
            if not np.isnan(forward_corr):
                forward_correlations.append(forward_corr)
        
        # 平均前瞻性相关性
        predictive_power = np.mean(forward_correlations) if forward_correlations else 0
        
        # 3. 计算最大滞后相关性（寻找最佳预测窗口）
        max_lag_correlation = 0
        best_lag = 0
        
        for lag in range(1, 11):  # 测试1-10天的滞后
            if lag >= len(aligned_data):
                break
            lagged_risk = aligned_data['risk_score'].shift(lag)
            lag_corr = aligned_data['drawdown'].corr(lagged_risk)
            if not np.isnan(lag_corr) and abs(lag_corr) > abs(max_lag_correlation):
                max_lag_correlation = lag_corr
                best_lag = lag
        
        return {
            'correlation': current_correlation,
            'predictive_power': predictive_power,
            'lead_lag_correlation': max_lag_correlation,
            'best_prediction_lag': best_lag
        }
    
    def enhance_technical_risk_analysis(self, holdings: Dict[str, float], stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        增强的技术指标风险分析
        
        Args:
            holdings: 持仓字典
            stock_data: 股票数据字典
            
        Returns:
            细化的技术风险分析结果
        """
        # 初始化结果
        result = {
            'overall_technical_risk': 0,
            'high_risk_stocks': [],
            'risk_patterns': {},
            'risk_trends': {},
            'risk_clusters': {}
        }
        
        # 各类技术风险模式
        risk_patterns = {
            'bearish_trend': 0,
            'volatility_breakout': 0,
            'support_breakdown': 0,
            'volume_divergence': 0,
            'momentum_loss': 0,
            'negative_divergence': 0,
            'ma_breakdown': 0
        }
        
        # 风险趋势统计
        risk_trends = {
            'increasing': 0,
            'decreasing': 0,
            'stable': 0
        }
        
        # 各股票的风险评分和详情
        stock_risks = {}
        high_risk_weights = 0  # 高风险股票的总权重
        
        for code, weight in holdings.items():
            if code in stock_data:
                df = stock_data[code].copy()
                
                # 使用风险预警系统评估风险
                risk_df = self.warning_system.classify_risk_level(df)
                
                if risk_df.empty:
                    continue
                
                # 获取最新风险评分和等级
                latest_risk_score = risk_df['risk_score'].iloc[-1]
                latest_risk_level = risk_df['adjusted_risk_level'].iloc[-1]
                
                # 存储每只股票的风险信息
                stock_risks[code] = {
                    'risk_score': latest_risk_score,
                    'risk_level': latest_risk_level,
                    'weight': weight
                }
                
                # 加入高风险股票列表
                if latest_risk_score >= self.risk_threshold:
                    result['high_risk_stocks'].append({
                        'code': code,
                        'risk_score': latest_risk_score,
                        'risk_level': latest_risk_level,
                        'weight': weight
                    })
                    high_risk_weights += weight
                
                # 分析风险趋势
                if len(risk_df) >= 5:
                    risk_5d_ago = risk_df['risk_score'].iloc[-5]
                    risk_change = latest_risk_score - risk_5d_ago
                    
                    if risk_change > 5:
                        risk_trends['increasing'] += weight
                    elif risk_change < -5:
                        risk_trends['decreasing'] += weight
                    else:
                        risk_trends['stable'] += weight
                
                # 识别技术风险模式
                # 1. 下降趋势
                if 'close' in df.columns and len(df) >= 20:
                    ma20 = df['close'].rolling(window=20).mean()
                    ma50 = df['close'].rolling(window=50).mean() if len(df) >= 50 else None
                    
                    if ma20 is not None and not ma20.empty:
                        latest_close = df['close'].iloc[-1]
                        latest_ma20 = ma20.iloc[-1]
                        
                        if latest_close < latest_ma20:
                            risk_patterns['bearish_trend'] += weight
                        
                        if ma50 is not None and not ma50.empty:
                            latest_ma50 = ma50.iloc[-1]
                            if latest_ma20 < latest_ma50:
                                risk_patterns['ma_breakdown'] += weight
                
                # 2. 波动率突破
                if 'close' in df.columns and len(df) >= 20:
                    returns = df['close'].pct_change()
                    vol_20d = returns.rolling(window=20).std() * np.sqrt(252)  # 年化波动率
                    
                    if not vol_20d.empty:
                        latest_vol = vol_20d.iloc[-1]
                        avg_vol = vol_20d.iloc[-20:].mean()
                        
                        if latest_vol > avg_vol * 1.5:
                            risk_patterns['volatility_breakout'] += weight
                
                # 3. 支撑位击穿
                if all(x in df.columns for x in ['close', 'low']):
                    # 简化的支撑位计算 - 过去20天最低点
                    if len(df) >= 20:
                        support_level = df['low'].iloc[-20:].min()
                        latest_close = df['close'].iloc[-1]
                        
                        if latest_close < support_level:
                            risk_patterns['support_breakdown'] += weight
                
                # 4. 成交量背离
                if all(x in df.columns for x in ['close', 'volume']) and len(df) >= 10:
                    price_trend = df['close'].iloc[-5:].mean() > df['close'].iloc[-10:-5].mean()
                    volume_trend = df['volume'].iloc[-5:].mean() > df['volume'].iloc[-10:-5].mean()
                    
                    if price_trend and not volume_trend:
                        risk_patterns['volume_divergence'] += weight
                
                # 5. 动量损失
                if 'close' in df.columns and len(df) >= 14:
                    roc = (df['close'].iloc[-1] / df['close'].iloc[-14] - 1) * 100
                    if roc < -5:  # 14日下跌超过5%
                        risk_patterns['momentum_loss'] += weight
                
                # 6. 负面背离
                if 'close' in df.columns and len(df) >= 14:
                    # 简化的背离检测
                    if 'rsi' in df.columns:
                        price_higher = df['close'].iloc[-1] > df['close'].iloc[-5]
                        rsi_lower = df['rsi'].iloc[-1] < df['rsi'].iloc[-5]
                        
                        if price_higher and rsi_lower:
                            risk_patterns['negative_divergence'] += weight
        
        # 计算组合整体技术风险
        if stock_risks:
            result['overall_technical_risk'] = sum(info['risk_score'] * info['weight'] for info in stock_risks.values())
            
            # 归一化风险模式数据
            result['risk_patterns'] = {k: min(v * 100, 100) for k, v in risk_patterns.items()}
            result['risk_trends'] = {k: v * 100 for k, v in risk_trends.items()}
            
            # 计算风险集中度
            result['risk_concentration'] = high_risk_weights * 100  # 转为百分比
            
            # 识别风险聚类
            if result['high_risk_stocks']:
                # 按行业分组
                sector_risks = {}
                for stock in result['high_risk_stocks']:
                    sector = stock.get('sector', 'unknown')
                    if sector not in sector_risks:
                        sector_risks[sector] = 0
                    sector_risks[sector] += stock['weight']
                
                result['risk_clusters']['sector'] = sector_risks
                
                # 按风险模式分组
                pattern_clusters = {}
                for pattern, score in result['risk_patterns'].items():
                    if score > 25:  # 超过25%的组合暴露于此风险模式
                        pattern_clusters[pattern] = score
                
                result['risk_clusters']['patterns'] = pattern_clusters
        
        return result 