#!/usr/bin/env python3
"""
A股市场分析模块

本模块实现A股市场的综合分析功能，包括市场概况、行业分析、个股分析等。
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# 使用依赖注入架构
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from utils.decorators import exception_handler, performance_monitor
from utils.dependency_injection import get_logger
from utils.date_utils import get_trading_day
from analysis.engines.date_manager import Date_manager
from analysis.engines.complex_logic_processor import Complex_logic_processor

logger = getLogger(__name__)

class AstockMarketAnalyzer:
    """A股市场分析器"""
    
    def __init___117(self):
        """初始化A股市场分析器"""
        self.container = get_container()
        self.data_access = self.get_service(Data_access_interface)
        self.date_manager = Date_manager()
        self.logic_processor = Complex_logic_processor()
        self.analysis_cache = {}
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def get_market_overview(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取市场概况
        
        Args:
            date: 分析日期，默认为最新交易日
            
        Returns:
            Dict[str, Any]: 市场概况数据
        """
        try:
            if date is None:
                date = get_trading_day()
            
            logger.info(f"获取市场概况: {date}")
            
            # 获取市场指数数据
            market_indices = self._get_market_indices(date)
            
            # 获取个股统计
            stock_statistics = self._get_stock_statistics(date)
            
            # 获取行业分析
            industry_analysis = self._get_industry_analysis(date)
            
            # 获取资金流向
            capital_flow = self._get_capital_flow_analysis(date)
            
            # 获取市场情绪指标
            market_sentiment = self._get_market_sentiment(date)
            
            overview = {
                'analysis_date': date,
                'market_indices': market_indices,
                'stock_statistics': stock_statistics,
                'industry_analysis': industry_analysis,
                'capital_flow': capital_flow,
                'market_sentiment': market_sentiment,
                'summary': self._generate_market_summary(
                    market_indices, stock_statistics, industry_analysis
                )
            }
            
            logger.info("市场概况分析完成")
            return overview
            
        except Exception as e:
            logger.error(f"获取市场概况失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def _get_market_indices(self, date: str) -> Dict[str, Any]:
        """获取市场指数数据"""
        try:
            # 主要指数代码
            index_codes = [
                '000001.SH',  # 上证指数
                '399001.SZ',  # 深证成指
                '399006.SZ',  # 创业板指
                '000300.SH',  # 沪深300
                '000905.SH',  # 中证500
                '000852.SH'   # 中证1000
            ]
            
            indices_data = {}
            
            for code in index_codes:
                try:
                    # 获取指数数据
                    data = self.data_access.get_stock_data(
                        code=code,
                        start_date=(datetime.strptime(date, '%Y%m%d') - timedelta(days=30)).strftime('%Y-%m-%d'),
                        end_date=datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'),
                        level='日线'
                    )
                    
                    if not data.empty:
                        latest = data.iloc[-1]
                        prev = data.iloc[-2] if len(data) > 1 else latest
                        
                        indices_data[code] = {
                            'name': self._get_index_name(code),
                            'current_value': latest['close'],
                            'change': latest['close'] - prev['close'],
                            'change_pct': ((latest['close'] - prev['close']) / prev['close']) * 100,
                            'volume': latest['volume'],
                            'turnover': latest.get('turnover', 0)
                        }
                        
                except Exception as e:
                    logger.warning(f"获取指数 {code} 数据失败: {e}")
                    continue
            
            return indices_data
            
        except Exception as e:
            logger.error(f"获取市场指数数据失败: {e}")
            return {}
    
    def _get_index_name(self, code: str) -> str:
        """获取指数名称"""
        name_map = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指',
            '000300.SH': '沪深300',
            '000905.SH': '中证500',
            '000852.SH': '中证1000'
        }
        return name_map.get(code, code)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=8.0)
    def _get_stock_statistics(self, date: str) -> Dict[str, Any]:
        """获取个股统计数据"""
        try:
            # 获取当日所有股票数据
            all_stocks_data = self.data_access.get_market_data(
                date=datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'),
                level='日线'
            )
            
            if all_stocks_data.empty:
                return {}
            
            # 计算涨跌统计
            total_stocks = len(all_stocks_data)
            rising_stocks = len(all_stocks_data[all_stocks_data['change_pct'] > 0])
            falling_stocks = len(all_stocks_data[all_stocks_data['change_pct'] < 0])
            unchanged_stocks = total_stocks - rising_stocks - falling_stocks
            
            # 涨跌停统计
            limit_up = len(all_stocks_data[all_stocks_data['change_pct'] >= 9.8])
            limit_down = len(all_stocks_data[all_stocks_data['change_pct'] <= -9.8])
            
            # 成交量统计
            total_volume = all_stocks_data['volume'].sum()
            total_turnover = all_stocks_data['turnover'].sum() if 'turnover' in all_stocks_data.columns else 0
            
            # 平均涨跌幅
            avg_change_pct = all_stocks_data['change_pct'].mean()
            
            statistics = {
                'total_stocks': total_stocks,
                'rising_stocks': rising_stocks,
                'falling_stocks': falling_stocks,
                'unchanged_stocks': unchanged_stocks,
                'rising_ratio': rising_stocks / total_stocks * 100,
                'falling_ratio': falling_stocks / total_stocks * 100,
                'limit_up': limit_up,
                'limit_down': limit_down,
                'total_volume': total_volume,
                'total_turnover': total_turnover,
                'avg_change_pct': avg_change_pct,
                'market_breadth': (rising_stocks - falling_stocks) / total_stocks * 100
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"获取个股统计数据失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def _get_industry_analysis(self, date: str) -> Dict[str, Any]:
        """获取行业分析数据"""
        try:
            # 获取行业数据
            industry_data = self.data_access.get_industry_data(
                date=datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            )
            
            if industry_data.empty:
                return {}
            
            # 按行业分组统计
            industry_stats = industry_data.groupby('industry').agg({
                'change_pct': ['mean', 'count'],
                'volume': 'sum',
                'turnover': 'sum'
            }).round(2)
            
            # 整理数据结构
            industry_analysis = {}
            for industry in industry_stats.index:
                industry_analysis[industry] = {
                    'avg_change_pct': industry_stats.loc[industry, ('change_pct', 'mean')],
                    'stock_count': industry_stats.loc[industry, ('change_pct', 'count')],
                    'total_volume': industry_stats.loc[industry, ('volume', 'sum')],
                    'total_turnover': industry_stats.loc[industry, ('turnover', 'sum')]
                }
            
            # 排序获取涨幅最大和最小的行业
            sorted_industries = sorted(
                industry_analysis.items(),
                key=lambda x: x[1]['avg_change_pct'],
                reverse=True
            )
            
            return {
                'industry_performance': industry_analysis,
                'top_industries': sorted_industries[:10],
                'bottom_industries': sorted_industries[-10:],
                'total_industries': len(industry_analysis)
            }
            
        except Exception as e:
            logger.error(f"获取行业分析数据失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=6.0)
    def _get_capital_flow_analysis(self, date: str) -> Dict[str, Any]:
        """获取资金流向分析"""
        try:
            # 获取资金流向数据
            capital_flow_data = self.data_access.get_capital_flow_data(
                date=datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            )
            
            if capital_flow_data.empty:
                return {}
            
            # 计算主力资金流向
            main_inflow = capital_flow_data['main_inflow'].sum()
            main_outflow = capital_flow_data['main_outflow'].sum()
            net_main_flow = main_inflow - main_outflow
            
            # 计算散户资金流向
            retail_inflow = capital_flow_data['retail_inflow'].sum()
            retail_outflow = capital_flow_data['retail_outflow'].sum()
            net_retail_flow = retail_inflow - retail_outflow
            
            # 计算北向资金流向
            north_flow = capital_flow_data['north_flow'].sum() if 'north_flow' in capital_flow_data.columns else 0
            
            capital_flow = {
                'main_inflow': main_inflow,
                'main_outflow': main_outflow,
                'net_main_flow': net_main_flow,
                'retail_inflow': retail_inflow,
                'retail_outflow': retail_outflow,
                'net_retail_flow': net_retail_flow,
                'north_flow': north_flow,
                'total_inflow': main_inflow + retail_inflow,
                'total_outflow': main_outflow + retail_outflow
            }
            
            return capital_flow
            
        except Exception as e:
            logger.error(f"获取资金流向分析失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def _get_market_sentiment(self, date: str) -> Dict[str, Any]:
        """获取市场情绪指标"""
        try:
            # 获取市场情绪相关数据
            sentiment_data = self.data_access.get_market_sentiment_data(
                date=datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            )
            
            if sentiment_data.empty:
                return {}
            
            # 计算恐慌贪婪指数
            fear_greed_index = self._calculate_fear_greed_index(sentiment_data)
            
            # 计算市场热度
            market_heat = self._calculate_market_heat(sentiment_data)
            
            # 计算投资者情绪
            investor_sentiment = self._calculate_investor_sentiment(sentiment_data)
            
            sentiment = {
                'fear_greed_index': fear_greed_index,
                'market_heat': market_heat,
                'investor_sentiment': investor_sentiment,
                'sentiment_level': self._get_sentiment_level(fear_greed_index)
            }
            
            return sentiment
            
        except Exception as e:
            logger.error(f"获取市场情绪指标失败: {e}")
            return {}
    
    def _calculate_fear_greed_index(self, data: pd.DataFrame) -> float:
        """计算恐慌贪婪指数"""
        try:
            # 这里使用简化的计算方法
            # 实际应用中可以使用更复杂的算法
            volatility_score = min(data['volatility'].mean() * 10, 100)
            volume_score = min(data['volume_ratio'].mean() * 20, 100)
            momentum_score = min((data['change_pct'].mean() + 10) * 5, 100)
            
            # 加权平均
            fear_greed_index = (volatility_score * 0.3 + volume_score * 0.3 + momentum_score * 0.4)
            return max(0, min(100, fear_greed_index))
            
        except Exception as e:
            logger.error(f"计算恐慌贪婪指数失败: {e}")
            return 50.0  # 默认中性值
    
    def _calculate_market_heat(self, data: pd.DataFrame) -> float:
        """计算市场热度"""
        try:
            # 基于成交量和涨跌幅计算市场热度
            volume_heat = data['volume_ratio'].mean()
            price_heat = abs(data['change_pct'].mean())
            
            market_heat = (volume_heat + price_heat) / 2
            return max(0, min(100, market_heat))
            
        except Exception as e:
            logger.error(f"计算市场热度失败: {e}")
            return 50.0
    
    def _calculate_investor_sentiment(self, data: pd.DataFrame) -> str:
        """计算投资者情绪"""
        try:
            avg_change = data['change_pct'].mean()
            
            if avg_change > 2:
                return "极度乐观"
            elif avg_change > 1:
                return "乐观"
            elif avg_change > 0:
                return "谨慎乐观"
            elif avg_change > -1:
                return "谨慎悲观"
            elif avg_change > -2:
                return "悲观"
            else:
                return "极度悲观"
                
        except Exception as e:
            logger.error(f"计算投资者情绪失败: {e}")
            return "中性"
    
    def _get_sentiment_level(self, fear_greed_index: float) -> str:
        """获取情绪水平描述"""
        if fear_greed_index >= 80:
            return "极度贪婪"
        elif fear_greed_index >= 60:
            return "贪婪"
        elif fear_greed_index >= 40:
            return "中性"
        elif fear_greed_index >= 20:
            return "恐慌"
        else:
            return "极度恐慌"
    
    def _generate_market_summary(self, market_indices: Dict[str, Any],
                               stock_statistics: Dict[str, Any],
                               industry_analysis: Dict[str, Any]) -> Dict[str, str]:
        """生成市场总结"""
        try:
            summary = {}
            
            # 市场表现总结
            if market_indices:
                main_index = market_indices.get('000001.SH', {})
                if main_index:
                    change_pct = main_index.get('change_pct', 0)
                    if change_pct > 1:
                        summary['market_trend'] = "市场表现强势，主要指数大幅上涨"
                    elif change_pct > 0:
                        summary['market_trend'] = "市场表现平稳，主要指数小幅上涨"
                    elif change_pct > -1:
                        summary['market_trend'] = "市场表现疲软，主要指数小幅下跌"
                    else:
                        summary['market_trend'] = "市场表现低迷，主要指数大幅下跌"
            
            # 个股表现总结
            if stock_statistics:
                rising_ratio = stock_statistics.get('rising_ratio', 0)
                if rising_ratio > 70:
                    summary['stock_performance'] = "个股普涨，市场情绪积极"
                elif rising_ratio > 50:
                    summary['stock_performance'] = "个股涨多跌少，市场情绪乐观"
                elif rising_ratio > 30:
                    summary['stock_performance'] = "个股涨跌参半，市场情绪谨慎"
                else:
                    summary['stock_performance'] = "个股普跌，市场情绪悲观"
            
            # 行业表现总结
            if industry_analysis and industry_analysis.get('top_industries'):
                top_industry = industry_analysis['top_industries'][0]
                summary['industry_performance'] = f"行业表现分化，{top_industry[0]}领涨"
            
            return summary
            
        except Exception as e:
            logger.error(f"生成市场总结失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=15.0)
    def analyze_stock_selection_opportunities(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        分析选股机会
        
        Args:
            date: 分析日期
            
        Returns:
            Dict[str, Any]: 选股机会分析结果
        """
        try:
            if date is None:
                date = get_trading_day()
            
            logger.info(f"分析选股机会: {date}")
            
            # 获取市场数据
            market_data = self.data_access.get_market_data(
                date=datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'),
                level='日线'
            )
            
            if market_data.empty:
                return {}
            
            # 技术面选股
            technical_opportunities = self._find_technical_opportunities(market_data)
            
            # 基本面选股
            fundamental_opportunities = self._find_fundamental_opportunities(market_data)
            
            # 资金面选股
            capital_opportunities = self._find_capital_opportunities(market_data)
            
            # 主题概念选股
            theme_opportunities = self._find_theme_opportunities(market_data)
            
            opportunities = {
                'analysis_date': date,
                'technical_opportunities': technical_opportunities,
                'fundamental_opportunities': fundamental_opportunities,
                'capital_opportunities': capital_opportunities,
                'theme_opportunities': theme_opportunities,
                'comprehensive_ranking': self._rank_opportunities(
                    technical_opportunities, fundamental_opportunities, capital_opportunities
                )
            }
            
            logger.info("选股机会分析完成")
            return opportunities
            
        except Exception as e:
            logger.error(f"分析选股机会失败: {e}")
            raise
    
    def _find_technical_opportunities(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """寻找技术面机会"""
        try:
            opportunities = []
            
            # 技术指标筛选条件
            conditions = [
                (data['rsi'] < 30) & (data['change_pct'] > 0),  # RSI超卖反弹
                (data['macd_signal'] == 1) & (data['volume_ratio'] > 1.5),  # MACD金叉放量
                (data['bollinger_position'] < 0.2) & (data['change_pct'] > 2),  # 布林下轨反弹
            ]
            
            for i, condition in enumerate(conditions):
                filtered_data = data[condition]
                if not filtered_data.empty:
                    for _, row in filtered_data.head(10).iterrows():
                        opportunities.append({
                            'code': row['code'],
                            'name': row.get('name', ''),
                            'type': f'技术面机会{i+1}',
                            'reason': self._get_technical_reason(i),
                            'score': self._calculate_technical_score(row),
                            'current_price': row['close'],
                            'change_pct': row['change_pct']
                        })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"寻找技术面机会失败: {e}")
            return []
    
    def _find_fundamental_opportunities(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """寻找基本面机会"""
        try:
            opportunities = []
            
            # 基本面筛选条件
            if 'pe_ratio' in data.columns and 'pb_ratio' in data.columns:
                # 低估值股票
                undervalued = data[
                    (data['pe_ratio'] < 20) & 
                    (data['pb_ratio'] < 2) & 
                    (data['roe'] > 10)
                ]
                
                for _, row in undervalued.head(10).iterrows():
                    opportunities.append({
                        'code': row['code'],
                        'name': row.get('name', ''),
                        'type': '基本面机会',
                        'reason': '低估值高ROE',
                        'score': self._calculate_fundamental_score(row),
                        'current_price': row['close'],
                        'pe_ratio': row['pe_ratio'],
                        'pb_ratio': row['pb_ratio'],
                        'roe': row['roe']
                    })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"寻找基本面机会失败: {e}")
            return []
    
    def _find_capital_opportunities(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """寻找资金面机会"""
        try:
            opportunities = []
            
            # 资金流入筛选
            if 'main_inflow' in data.columns:
                capital_inflow = data[
                    (data['main_inflow'] > data['main_outflow']) &
                    (data['volume_ratio'] > 2)
                ]
                
                for _, row in capital_inflow.head(10).iterrows():
                    opportunities.append({
                        'code': row['code'],
                        'name': row.get('name', ''),
                        'type': '资金面机会',
                        'reason': '主力资金流入',
                        'score': self._calculate_capital_score(row),
                        'current_price': row['close'],
                        'main_inflow': row['main_inflow'],
                        'volume_ratio': row['volume_ratio']
                    })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"寻找资金面机会失败: {e}")
            return []
    
    def _find_theme_opportunities(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """寻找主题概念机会"""
        try:
            opportunities = []
            
            # 热门概念股
            if 'concept' in data.columns:
                concept_performance = data.groupby('concept')['change_pct'].mean().sort_values(ascending=False)
                
                for concept, avg_change in concept_performance.head(5).items():
                    concept_stocks = data[data['concept'] == concept].head(3)
                    
                    for _, row in concept_stocks.iterrows():
                        opportunities.append({
                            'code': row['code'],
                            'name': row.get('name', ''),
                            'type': '主题概念机会',
                            'reason': f'{concept}概念',
                            'score': self._calculate_theme_score(row),
                            'current_price': row['close'],
                            'concept': concept,
                            'concept_avg_change': avg_change
                        })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"寻找主题概念机会失败: {e}")
            return []
    
    def _get_technical_reason(self, index: int) -> str:
        """获取技术面原因描述"""
        reasons = [
            "RSI超卖反弹",
            "MACD金叉放量",
            "布林下轨反弹"
        ]
        return reasons[index] if index < len(reasons) else "技术面机会"
    
    def _calculate_technical_score(self, row: pd.Series) -> float:
        """计算技术面评分"""
        try:
            score = 0
            
            # RSI评分
            if 'rsi' in row and not pd.isna(row['rsi']):
                if row['rsi'] < 30:
                    score += 30
                elif row['rsi'] < 50:
                    score += 20
            
            # 成交量评分
            if 'volume_ratio' in row and not pd.isna(row['volume_ratio']):
                if row['volume_ratio'] > 2:
                    score += 25
                elif row['volume_ratio'] > 1.5:
                    score += 15
            
            # 涨幅评分
            if 'change_pct' in row and not pd.isna(row['change_pct']):
                if row['change_pct'] > 5:
                    score += 20
                elif row['change_pct'] > 2:
                    score += 15
                elif row['change_pct'] > 0:
                    score += 10
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"计算技术面评分失败: {e}")
            return 0
    
    def _calculate_fundamental_score(self, row: pd.Series) -> float:
        """计算基本面评分"""
        try:
            score = 0
            
            # PE评分
            if 'pe_ratio' in row and not pd.isna(row['pe_ratio']):
                if row['pe_ratio'] < 15:
                    score += 30
                elif row['pe_ratio'] < 25:
                    score += 20
            
            # PB评分
            if 'pb_ratio' in row and not pd.isna(row['pb_ratio']):
                if row['pb_ratio'] < 1.5:
                    score += 25
                elif row['pb_ratio'] < 2:
                    score += 15
            
            # ROE评分
            if 'roe' in row and not pd.isna(row['roe']):
                if row['roe'] > 15:
                    score += 25
                elif row['roe'] > 10:
                    score += 15
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"计算基本面评分失败: {e}")
            return 0
    
    def _calculate_capital_score(self, row: pd.Series) -> float:
        """计算资金面评分"""
        try:
            score = 0
            
            # 主力资金流入评分
            if 'main_inflow' in row and 'main_outflow' in row:
                net_flow = row['main_inflow'] - row['main_outflow']
                if net_flow > 0:
                    score += 40
            
            # 成交量评分
            if 'volume_ratio' in row and not pd.isna(row['volume_ratio']):
                if row['volume_ratio'] > 3:
                    score += 30
                elif row['volume_ratio'] > 2:
                    score += 20
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"计算资金面评分失败: {e}")
            return 0
    
    def _calculate_theme_score(self, row: pd.Series) -> float:
        """计算主题概念评分"""
        try:
            score = 0
            
            # 概念平均涨幅评分
            if 'concept_avg_change' in row and not pd.isna(row['concept_avg_change']):
                if row['concept_avg_change'] > 5:
                    score += 40
                elif row['concept_avg_change'] > 3:
                    score += 30
                elif row['concept_avg_change'] > 1:
                    score += 20
            
            # 个股涨幅评分
            if 'change_pct' in row and not pd.isna(row['change_pct']):
                if row['change_pct'] > 5:
                    score += 30
                elif row['change_pct'] > 3:
                    score += 20
                elif row['change_pct'] > 0:
                    score += 10
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"计算主题概念评分失败: {e}")
            return 0
    
    def _rank_opportunities(self, technical: List[Dict], fundamental: List[Dict], 
                          capital: List[Dict]) -> List[Dict[str, Any]]:
        """综合排名机会"""
        try:
            all_opportunities = technical + fundamental + capital
            
            # 按评分排序
            ranked_opportunities = sorted(
                all_opportunities,
                key=lambda x: x.get('score', 0),
                reverse=True
            )
            
            return ranked_opportunities[:20]  # 返回前20个机会
            
        except Exception as e:
            logger.error(f"综合排名机会失败: {e}")
            return []
    
    @exception_handler(reraise=True)
    def save_analysis_results(self, results: Dict[str, Any], 
                            output_file: str) -> None:
        """
        保存分析结果
        
        Args:
            results: 分析结果
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存为JSON格式
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"分析结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
            raise

def main_50():
    """主函数"""
    try:
        # 初始化分析器
        analyzer = AStock_market_analyzer()
        
        # 获取市场概况
        market_overview = analyzer.get_market_overview()
        
        # 分析选股机会
        opportunities = analyzer.analyze_stock_selection_opportunities()
        
        # 组合结果
        results = {
            'market_overview': market_overview,
            'opportunities': opportunities,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存结果
        analyzer.save_analysis_results(results, 'data/result/market_analysis_results.json')
        
        print("A股市场分析完成")
        
    except Exception as e:
        logger.error(f"A股市场分析失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_50()