#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.clickhouse_db import get_clickhouse_db, get_default_config
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from utils.path_utils import get_result_dir
from indicators.factory import IndicatorFactory
from analysis.market.market_dimension_analyzer import MarketDimensionAnalyzer
from analysis.buypoints.buypoint_dimension_analyzer import BuyPointDimensionAnalyzer

# 获取日志记录器
logger = get_logger(__name__)

class MultiDimensionAnalyzer:
    """
    多维度分析器 - 集成市场和买点分析功能，提供多维度综合分析
    
    支持对个股和市场进行多周期、多指标的综合分析，能够提取共性特征并生成分析报告
    """
    
    def __init__(self):
        """初始化多维度分析器"""
        logger.info("初始化多维度分析器")
        
        # 获取数据库连接
        config = get_default_config()
        self.ch_db = get_clickhouse_db(config=config)
        
        # 创建指标工厂
        self.indicator_factory = IndicatorFactory()
        
        # 创建市场分析器和买点分析器
        self.market_analyzer = MarketDimensionAnalyzer()
        self.buypoint_analyzer = BuyPointDimensionAnalyzer()
        
        # 存储分析结果
        self.analysis_results = {
            "single_stock_analysis": {},
            "stock_group_analysis": {},
            "market_correlation": {},
            "multi_period_analysis": {}
        }
        
        # 结果输出目录
        self.result_dir = get_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)
        
        logger.info("多维度分析器初始化完成")
    
    def analyze_single_stock(self, stock_code: str, date: str,
                          periods: List[str] = ['DAILY', 'WEEKLY'],
                          indicators: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        对单个股票进行多维度分析
        
        Args:
            stock_code: 股票代码
            date: 分析日期，格式YYYYMMDD
            periods: 分析周期列表
            indicators: 指标列表，如果为None则使用默认指标集
            
        Returns:
            Dict: 分析结果
        """
        logger.info(f"开始对股票 {stock_code} 进行多维度分析, 日期={date}")
        
        try:
            # 获取股票基本信息
            stock_info = self._get_stock_info(stock_code)
            
            if not stock_info:
                logger.warning(f"未找到股票 {stock_code} 的基本信息")
                return {}
            
            # 使用买点分析器进行多周期分析
            multi_period_result = self.buypoint_analyzer.analyze_multi_period(
                [stock_code], date, periods)
            
            # 使用默认指标集如果未提供
            if indicators is None:
                indicators = self._get_default_indicators()
            
            # 为每个周期分析指标信号
            indicator_results = {}
            
            for period in periods:
                # 获取日期范围
                start_date, end_date = self._get_period_date_range(date, period)
                
                # 分析指标信号
                indicator_result = self.buypoint_analyzer.analyze_indicator_signals(
                    [stock_code], start_date, end_date, indicators, period)
                
                indicator_results[period] = indicator_result
            
            # 获取市场状态
            market_state = self.market_analyzer.analyze_market_overview(date)
            
            # 获取行业分析
            industry = stock_info.get('industry', '')
            industry_analysis = None
            if industry:
                # 查询同行业股票
                industry_stocks = self._get_industry_stocks(industry, exclude=[stock_code])
                
                if industry_stocks:
                    # 使用买点分析器分析行业股票
                    start_date, end_date = self._get_period_date_range(date, 'DAILY')
                    industry_analysis = self.buypoint_analyzer.analyze_pattern_features(
                        industry_stocks, start_date, end_date, 'DAILY')
            
            # 构建综合结果
            result = {
                "stock_code": stock_code,
                "stock_name": stock_info.get('stock_name', ''),
                "industry": industry,
                "date": date,
                "periods": periods,
                "multi_period_analysis": multi_period_result,
                "indicator_analysis": indicator_results,
                "market_state": market_state,
                "industry_analysis": industry_analysis
            }
            
            # 提取关键特征
            result["key_features"] = self._extract_key_features(result)
            
            # 生成综合评估
            result["assessment"] = self._generate_stock_assessment(result)
            
            # 保存到分析结果
            self.analysis_results["single_stock_analysis"][stock_code] = result
            
            logger.info(f"股票 {stock_code} 的多维度分析完成")
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 时出错: {e}")
            return {}
            
    def _get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """获取股票基本信息"""
        try:
            sql = f"""
            SELECT stock_code, stock_name, industry, market, market_cap
            FROM stock_basic
            WHERE stock_code = '{stock_code}'
            LIMIT 1
            """
            result = self.ch_db.query_df(sql)
            
            if result.empty:
                return {}
                
            return result.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 基本信息时出错: {e}")
            return {}
    
    def _get_industry_stocks(self, industry: str, exclude: List[str] = None) -> List[str]:
        """获取行业股票列表"""
        try:
            exclude_condition = ""
            if exclude and len(exclude) > 0:
                exclude_str = "', '".join(exclude)
                exclude_condition = f" AND stock_code NOT IN ('{exclude_str}')"
                
            sql = f"""
            SELECT stock_code
            FROM stock_basic
            WHERE industry = '{industry}'{exclude_condition}
            LIMIT 50
            """
            result = self.ch_db.query_df(sql)
            
            if result.empty:
                return []
                
            return result['stock_code'].tolist()
            
        except Exception as e:
            logger.error(f"获取行业 {industry} 股票列表时出错: {e}")
            return []
    
    def _get_period_date_range(self, date: str, period: str) -> Tuple[str, str]:
        """根据周期获取日期范围"""
        date_obj = datetime.strptime(date, "%Y%m%d")
        
        if period == 'DAILY':
            # 日线取最近30天
            start_date = (date_obj - timedelta(days=30)).strftime("%Y%m%d")
            end_date = date
        elif period == 'WEEKLY':
            # 周线取最近12周
            start_date = (date_obj - timedelta(days=12*7)).strftime("%Y%m%d")
            end_date = date
        elif period == 'MONTHLY':
            # 月线取最近6个月
            start_date = (date_obj - timedelta(days=6*30)).strftime("%Y%m%d")
            end_date = date
        else:
            # 默认取30天
            start_date = (date_obj - timedelta(days=30)).strftime("%Y%m%d")
            end_date = date
            
        return start_date, end_date
    
    def _get_default_indicators(self) -> List[Dict[str, Any]]:
        """获取默认指标集"""
        return [
            {
                "name": "MACD",
                "id": "MACD",
                "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
            },
            {
                "name": "KDJ",
                "id": "KDJ",
                "parameters": {"k_period": 9, "d_period": 3}
            },
            {
                "name": "RSI",
                "id": "RSI",
                "parameters": {"period": 14}
            },
            {
                "name": "BOLL",
                "id": "BOLL",
                "parameters": {"period": 20, "std_dev": 2.0}
            },
            {
                "name": "MA",
                "id": "MA",
                "parameters": {"periods": [5, 10, 20, 30, 60]}
            }
        ] 

    def analyze_stock_group(self, stock_codes: List[str], date: str,
                          periods: List[str] = ['DAILY'],
                          indicators: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        对股票组进行多维度分析
        
        Args:
            stock_codes: 股票代码列表
            date: 分析日期，格式YYYYMMDD
            periods: 分析周期列表
            indicators: 指标列表，如果为None则使用默认指标集
            
        Returns:
            Dict: 分析结果
        """
        logger.info(f"开始对 {len(stock_codes)} 只股票进行组分析, 日期={date}")
        
        try:
            # 使用买点分析器进行多周期分析
            multi_period_result = self.buypoint_analyzer.analyze_multi_period(
                stock_codes, date, periods)
            
            # 使用默认指标集如果未提供
            if indicators is None:
                indicators = self._get_default_indicators()
            
            # 为每个周期分析指标信号
            indicator_results = {}
            
            for period in periods:
                # 获取日期范围
                start_date, end_date = self._get_period_date_range(date, period)
                
                # 分析指标信号
                indicator_result = self.buypoint_analyzer.analyze_indicator_signals(
                    stock_codes, start_date, end_date, indicators, period)
                
                indicator_results[period] = indicator_result
            
            # 获取市场状态
            market_state = self.market_analyzer.analyze_market_overview(date)
            
            # 获取行业分布
            industry_distribution = self._analyze_industry_distribution(stock_codes)
            
            # 构建综合结果
            result = {
                "stock_count": len(stock_codes),
                "date": date,
                "periods": periods,
                "multi_period_analysis": multi_period_result,
                "indicator_analysis": indicator_results,
                "market_state": market_state,
                "industry_distribution": industry_distribution
            }
            
            # 提取共性特征
            result["common_features"] = self._extract_common_features(result)
            
            # 生成综合评估
            result["assessment"] = self._generate_group_assessment(result)
            
            # 保存到分析结果
            group_id = f"group_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results["stock_group_analysis"][group_id] = result
            
            logger.info(f"股票组分析完成, 共 {len(stock_codes)} 只股票")
            return result
            
        except Exception as e:
            logger.error(f"分析股票组时出错: {e}")
            return {}
    
    def analyze_market_correlation(self, stock_codes: List[str], 
                                 end_date: str, days: int = 30) -> Dict[str, Any]:
        """
        分析股票与市场的相关性
        
        Args:
            stock_codes: 股票代码列表
            end_date: 结束日期，格式YYYYMMDD
            days: 分析的天数
            
        Returns:
            Dict: 相关性分析结果
        """
        logger.info(f"开始分析 {len(stock_codes)} 只股票与市场的相关性, 截止日期={end_date}, 时间范围={days}天")
        
        try:
            # 计算开始日期
            end_date_obj = datetime.strptime(end_date, "%Y%m%d")
            start_date_obj = end_date_obj - timedelta(days=days)
            start_date = start_date_obj.strftime("%Y%m%d")
            
            # 获取指数数据
            index_codes = ['000001.SH', '399001.SZ', '399006.SZ', '000300.SH', '000905.SH']
            index_data = {}
            
            # 获取各指数数据
            for code in index_codes:
                sql = f"""
                SELECT date, close
                FROM index_daily
                WHERE code = '{code}' AND date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date
                """
                df = self.ch_db.query_df(sql)
                
                if not df.empty:
                    # 计算日收益率
                    df['return'] = df['close'].pct_change()
                    index_data[code] = df.dropna()
            
            # 获取股票数据
            stock_returns = {}
            for stock_code in stock_codes:
                sql = f"""
                SELECT date, close
                FROM stock_daily
                WHERE stock_code = '{stock_code}' AND date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date
                """
                df = self.ch_db.query_df(sql)
                
                if not df.empty:
                    # 计算日收益率
                    df['return'] = df['close'].pct_change()
                    stock_returns[stock_code] = df.dropna()
            
            # 计算相关性
            correlations = {}
            
            for stock_code, stock_df in stock_returns.items():
                stock_corr = {}
                
                for index_code, index_df in index_data.items():
                    # 合并数据，按日期对齐
                    merged = pd.merge(
                        stock_df[['date', 'return']], 
                        index_df[['date', 'return']], 
                        on='date', 
                        suffixes=('_stock', '_index')
                    )
                    
                    if not merged.empty and len(merged) > 5:
                        # 计算相关系数
                        corr = merged['return_stock'].corr(merged['return_index'])
                        stock_corr[index_code] = corr
                
                correlations[stock_code] = stock_corr
            
            # 计算平均相关性
            avg_correlations = {}
            for index_code in index_data.keys():
                values = [stock_corr.get(index_code, np.nan) for stock_corr in correlations.values()]
                avg_correlations[index_code] = np.nanmean(values)
            
            # 构建结果
            result = {
                "start_date": start_date,
                "end_date": end_date,
                "days": days,
                "stock_count": len(stock_codes),
                "index_codes": index_codes,
                "correlations": correlations,
                "avg_correlations": avg_correlations
            }
            
            # 识别高相关和低相关股票
            high_corr_stocks = {}
            low_corr_stocks = {}
            
            for index_code in index_data.keys():
                # 高相关股票（相关系数 > 0.7）
                high_corr = [
                    stock_code for stock_code, corrs in correlations.items() 
                    if index_code in corrs and corrs[index_code] > 0.7
                ]
                high_corr_stocks[index_code] = high_corr
                
                # 低相关股票（相关系数 < 0.3）
                low_corr = [
                    stock_code for stock_code, corrs in correlations.items() 
                    if index_code in corrs and corrs[index_code] < 0.3
                ]
                low_corr_stocks[index_code] = low_corr
            
            result["high_correlation_stocks"] = high_corr_stocks
            result["low_correlation_stocks"] = low_corr_stocks
            
            # 保存到分析结果
            self.analysis_results["market_correlation"] = result
            
            logger.info(f"市场相关性分析完成")
            return result
            
        except Exception as e:
            logger.error(f"分析市场相关性时出错: {e}")
            return {}
    
    def _analyze_industry_distribution(self, stock_codes: List[str]) -> Dict[str, Any]:
        """分析股票的行业分布"""
        try:
            # 查询股票行业信息
            stock_codes_str = "', '".join(stock_codes)
            sql = f"""
            SELECT industry, COUNT(*) as count
            FROM stock_basic
            WHERE stock_code IN ('{stock_codes_str}')
            GROUP BY industry
            ORDER BY count DESC
            """
            result = self.ch_db.query_df(sql)
            
            if result.empty:
                return {}
            
            # 构建行业分布
            distribution = {}
            for _, row in result.iterrows():
                industry = row['industry'] if row['industry'] else "其他"
                count = int(row['count'])
                distribution[industry] = count
            
            # 计算行业分布百分比
            total = sum(distribution.values())
            percentages = {k: v / total for k, v in distribution.items()}
            
            return {
                "counts": distribution,
                "percentages": percentages
            }
            
        except Exception as e:
            logger.error(f"分析行业分布时出错: {e}")
            return {}
    
    def _extract_key_features(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取单个股票的关键特征"""
        key_features = []
        
        # 从多周期分析中提取特征
        multi_period = analysis_result.get("multi_period_analysis", {})
        common_features = multi_period.get("common_features", [])
        
        for feature in common_features:
            if isinstance(feature, dict):
                key_features.append({
                    "type": "multi_period",
                    "feature": feature.get("feature", ""),
                    "periods": feature.get("periods", []),
                    "frequency": feature.get("avg_frequency", 0),
                    "details": feature.get("details", [])
                })
        
        # 从指标分析中提取特征
        indicator_analysis = analysis_result.get("indicator_analysis", {})
        for period, period_analysis in indicator_analysis.items():
            common_signals = period_analysis.get("common_signals", [])
            
            for signal in common_signals:
                if signal["frequency"] > 0.5:  # 只选择高频信号
                    key_features.append({
                        "type": "indicator",
                        "period": period,
                        "indicator": signal.get("indicator", ""),
                        "signal": signal.get("signal", ""),
                        "frequency": signal.get("frequency", 0)
                    })
        
        # 按频率排序
        key_features.sort(key=lambda x: x.get("frequency", 0), reverse=True)
        
        return key_features
    
    def _extract_common_features(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取股票组的共性特征"""
        common_features = []
        
        # 从多周期分析中提取特征
        multi_period = analysis_result.get("multi_period_analysis", {})
        period_common_features = multi_period.get("common_features", [])
        
        for feature in period_common_features:
            if isinstance(feature, dict):
                common_features.append({
                    "type": "multi_period",
                    "feature": feature.get("feature", ""),
                    "periods": feature.get("periods", []),
                    "frequency": feature.get("avg_frequency", 0),
                    "details": feature.get("details", [])
                })
        
        # 从指标分析中提取特征
        indicator_analysis = analysis_result.get("indicator_analysis", {})
        for period, period_analysis in indicator_analysis.items():
            common_signals = period_analysis.get("common_signals", [])
            
            for signal in common_signals:
                if signal["frequency"] > 0.4:  # 共性特征可以设置较低的阈值
                    common_features.append({
                        "type": "indicator",
                        "period": period,
                        "indicator": signal.get("indicator", ""),
                        "signal": signal.get("signal", ""),
                        "frequency": signal.get("frequency", 0)
                    })
        
        # 从行业分布中提取特征
        industry_distribution = analysis_result.get("industry_distribution", {})
        percentages = industry_distribution.get("percentages", {})
        
        # 只选择占比较高的行业
        high_percentage_industries = [
            {"industry": k, "percentage": v}
            for k, v in percentages.items()
            if v > 0.2  # 占比超过20%的行业
        ]
        
        for industry in high_percentage_industries:
            common_features.append({
                "type": "industry",
                "feature": f"行业集中: {industry['industry']}",
                "frequency": industry["percentage"]
            })
        
        # 按频率排序
        common_features.sort(key=lambda x: x.get("frequency", 0), reverse=True)
        
        return common_features
    
    def _generate_stock_assessment(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成单个股票的综合评估"""
        assessment = {
            "stock_code": analysis_result.get("stock_code", ""),
            "stock_name": analysis_result.get("stock_name", ""),
            "date": analysis_result.get("date", ""),
            "summary": "",
            "trend_score": 0,
            "signal_score": 0,
            "overall_score": 0,
            "recommendation": ""
        }
        
        # 趋势评分
        trend_score = 0
        
        # 通过关键特征计算趋势得分
        key_features = analysis_result.get("key_features", [])
        for feature in key_features:
            if feature["type"] == "multi_period":
                feature_name = feature.get("feature", "").lower()
                if any(kw in feature_name for kw in ["上涨", "突破", "反弹", "金叉", "多头"]):
                    trend_score += feature.get("frequency", 0) * 100
                elif any(kw in feature_name for kw in ["下跌", "跌破", "回调", "死叉", "空头"]):
                    trend_score -= feature.get("frequency", 0) * 100
            
        # 信号评分
        signal_score = 0
        
        # 通过指标信号计算信号得分
        for feature in key_features:
            if feature["type"] == "indicator":
                signal = feature.get("signal", "").lower()
                if any(kw in signal for kw in ["bullish", "buy", "golden_cross", "oversold", "uptrend"]):
                    signal_score += feature.get("frequency", 0) * 100
                elif any(kw in signal for kw in ["bearish", "sell", "dead_cross", "overbought", "downtrend"]):
                    signal_score -= feature.get("frequency", 0) * 100
        
        # 市场状态修正
        market_state = analysis_result.get("market_state", {})
        market_status = market_state.get("market_status", "")
        
        market_correction = 0
        if market_status in ["强势上涨", "上涨"]:
            market_correction = 20
        elif market_status in ["下跌", "弱势下跌"]:
            market_correction = -20
        
        # 综合评分
        overall_score = (trend_score + signal_score) / 2 + market_correction
        
        # 更新评估结果
        assessment["trend_score"] = round(trend_score, 2)
        assessment["signal_score"] = round(signal_score, 2)
        assessment["overall_score"] = round(overall_score, 2)
        
        # 评级和建议
        if overall_score > 70:
            assessment["recommendation"] = "强烈推荐买入"
        elif overall_score > 30:
            assessment["recommendation"] = "建议买入"
        elif overall_score > 0:
            assessment["recommendation"] = "观望为主，可少量买入"
        elif overall_score > -30:
            assessment["recommendation"] = "观望为主，不建议买入"
        else:
            assessment["recommendation"] = "不建议买入"
        
        # 生成总结文字
        stock_name = analysis_result.get("stock_name", "")
        industry = analysis_result.get("industry", "")
        
        summary = f"{stock_name}({analysis_result.get('stock_code', '')})属于{industry}行业，"
        
        if overall_score > 30:
            summary += "各项技术指标表现良好，"
            if trend_score > 30:
                summary += "趋势特征向好，"
            if signal_score > 30:
                summary += "买入信号明确，"
            summary += f"建议 {assessment['recommendation']}。"
        elif overall_score < -30:
            summary += "各项技术指标表现不佳，"
            if trend_score < -30:
                summary += "趋势特征向下，"
            if signal_score < -30:
                summary += "卖出信号明确，"
            summary += f"建议 {assessment['recommendation']}。"
        else:
            summary += "技术指标表现中性，无明显买卖信号，"
            summary += f"建议 {assessment['recommendation']}。"
        
        assessment["summary"] = summary
        
        return assessment
    
    def _generate_group_assessment(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成股票组的综合评估"""
        assessment = {
            "stock_count": analysis_result.get("stock_count", 0),
            "date": analysis_result.get("date", ""),
            "summary": "",
            "common_characteristics": "",
            "recommendation": ""
        }
        
        # 提取共性特征
        common_features = analysis_result.get("common_features", [])
        
        # 计算特征趋势方向
        bullish_features = 0
        bearish_features = 0
        
        for feature in common_features:
            feature_name = str(feature.get("feature", "")).lower()
            signal = str(feature.get("signal", "")).lower()
            
            # 判断特征方向
            if any(kw in feature_name for kw in ["上涨", "突破", "反弹", "金叉", "多头"]) or \
               any(kw in signal for kw in ["bullish", "buy", "golden_cross", "oversold", "uptrend"]):
                bullish_features += 1
            elif any(kw in feature_name for kw in ["下跌", "跌破", "回调", "死叉", "空头"]) or \
                 any(kw in signal for kw in ["bearish", "sell", "dead_cross", "overbought", "downtrend"]):
                bearish_features += 1
        
        # 获取行业分布
        industry_distribution = analysis_result.get("industry_distribution", {})
        percentages = industry_distribution.get("percentages", {})
        
        # 找出主要行业
        main_industries = []
        for industry, percentage in percentages.items():
            if percentage > 0.2:  # 占比超过20%的行业
                main_industries.append(f"{industry}({percentage*100:.1f}%)")
        
        # 生成共性特征描述
        feature_desc = []
        for i, feature in enumerate(common_features[:5]):  # 只取前5个特征
            if feature["type"] == "multi_period":
                periods_str = ", ".join(feature.get("periods", []))
                feature_desc.append(
                    f"{i+1}. {feature.get('feature', '')}, 周期: {periods_str}, "
                    f"频率: {feature.get('frequency', 0)*100:.1f}%"
                )
            elif feature["type"] == "indicator":
                feature_desc.append(
                    f"{i+1}. {feature.get('indicator', '')}: {feature.get('signal', '')}, "
                    f"周期: {feature.get('period', '')}, "
                    f"频率: {feature.get('frequency', 0)*100:.1f}%"
                )
            elif feature["type"] == "industry":
                feature_desc.append(
                    f"{i+1}. {feature.get('feature', '')}, "
                    f"占比: {feature.get('frequency', 0)*100:.1f}%"
                )
        
        assessment["common_characteristics"] = "\n".join(feature_desc)
        
        # 生成建议
        if bullish_features > bearish_features * 2:
            assessment["recommendation"] = "整体看好，可重点关注"
        elif bullish_features > bearish_features:
            assessment["recommendation"] = "偏向看好，可适当关注"
        elif bearish_features > bullish_features * 2:
            assessment["recommendation"] = "整体看空，不建议关注"
        elif bearish_features > bullish_features:
            assessment["recommendation"] = "偏向看空，建议谨慎"
        else:
            assessment["recommendation"] = "中性，建议观望"
        
        # 生成总结文字
        summary = f"分析了{analysis_result.get('stock_count', 0)}只股票，"
        
        if main_industries:
            summary += f"主要集中在{', '.join(main_industries)}等行业，"
        
        if bullish_features > bearish_features:
            summary += f"共有{bullish_features}个看多特征和{bearish_features}个看空特征，"
            summary += "整体技术面偏向乐观，"
        elif bearish_features > bullish_features:
            summary += f"共有{bullish_features}个看多特征和{bearish_features}个看空特征，"
            summary += "整体技术面偏向谨慎，"
        else:
            summary += "技术特征方向不明确，"
        
        summary += f"{assessment['recommendation']}。"
        
        assessment["summary"] = summary
        
        return assessment
    
    def save_results(self, output_file: str, format_type: str = "json") -> None:
        """
        保存分析结果
        
        Args:
            output_file: 输出文件路径
            format_type: 输出格式类型，支持json和markdown
        """
        logger.info(f"保存多维度分析结果到: {output_file}")
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if format_type.lower() == "json":
                # 构建完整结果
                full_results = {
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis_results": self.analysis_results
                }
                
                # 保存到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(full_results, f, indent=2, ensure_ascii=False)
                
            elif format_type.lower() == "markdown":
                # 构建Markdown内容
                markdown = "# 多维度分析报告\n\n"
                markdown += f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # 添加单股分析内容
                single_stock = self.analysis_results.get("single_stock_analysis", {})
                if single_stock:
                    markdown += "## 个股分析\n\n"
                    
                    for stock_code, analysis in single_stock.items():
                        stock_name = analysis.get("stock_name", "")
                        assessment = analysis.get("assessment", {})
                        
                        markdown += f"### {stock_name}({stock_code})\n\n"
                        
                        # 综合评估
                        if assessment:
                            markdown += "#### 综合评估\n\n"
                            markdown += f"{assessment.get('summary', '')}\n\n"
                            markdown += f"趋势评分: {assessment.get('trend_score', 0)}\n\n"
                            markdown += f"信号评分: {assessment.get('signal_score', 0)}\n\n"
                            markdown += f"综合评分: {assessment.get('overall_score', 0)}\n\n"
                            markdown += f"建议: {assessment.get('recommendation', '')}\n\n"
                        
                        # 关键特征
                        key_features = analysis.get("key_features", [])
                        if key_features:
                            markdown += "#### 关键特征\n\n"
                            markdown += "| 特征类型 | 特征 | 频率 | 周期 |\n"
                            markdown += "| -------- | ---- | ---- | ---- |\n"
                            
                            for feature in key_features[:10]:  # 最多显示10个
                                feature_type = feature.get("type", "")
                                feature_name = feature.get("feature", "")
                                if not feature_name and "signal" in feature:
                                    feature_name = f"{feature.get('indicator', '')}: {feature.get('signal', '')}"
                                frequency = feature.get("frequency", 0)
                                periods = ", ".join(feature.get("periods", [feature.get("period", "")]))
                                
                                markdown += f"| {feature_type} | {feature_name} | {frequency*100:.1f}% | {periods} |\n"
                            
                            markdown += "\n"
                
                # 添加股票组分析内容
                stock_group = self.analysis_results.get("stock_group_analysis", {})
                if stock_group:
                    markdown += "## 股票组分析\n\n"
                    
                    for group_id, analysis in stock_group.items():
                        stock_count = analysis.get("stock_count", 0)
                        date = analysis.get("date", "")
                        assessment = analysis.get("assessment", {})
                        
                        markdown += f"### 股票组 {group_id}\n\n"
                        markdown += f"分析日期: {date}, 股票数量: {stock_count}\n\n"
                        
                        # 综合评估
                        if assessment:
                            markdown += "#### 综合评估\n\n"
                            markdown += f"{assessment.get('summary', '')}\n\n"
                            markdown += f"建议: {assessment.get('recommendation', '')}\n\n"
                        
                        # 共性特征
                        if "common_characteristics" in assessment:
                            markdown += "#### 共性特征\n\n"
                            markdown += f"```\n{assessment.get('common_characteristics', '')}\n```\n\n"
                        
                        # 行业分布
                        industry_distribution = analysis.get("industry_distribution", {})
                        percentages = industry_distribution.get("percentages", {})
                        
                        if percentages:
                            markdown += "#### 行业分布\n\n"
                            markdown += "| 行业 | 占比 |\n"
                            markdown += "| ---- | ---- |\n"
                            
                            for industry, percentage in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
                                markdown += f"| {industry} | {percentage*100:.1f}% |\n"
                            
                            markdown += "\n"
                
                # 添加市场相关性分析
                market_correlation = self.analysis_results.get("market_correlation", {})
                if market_correlation:
                    markdown += "## 市场相关性分析\n\n"
                    
                    start_date = market_correlation.get("start_date", "")
                    end_date = market_correlation.get("end_date", "")
                    days = market_correlation.get("days", 0)
                    
                    markdown += f"分析周期: {start_date} 至 {end_date}, 共 {days} 天\n\n"
                    
                    # 平均相关性
                    avg_correlations = market_correlation.get("avg_correlations", {})
                    if avg_correlations:
                        markdown += "### 平均相关性\n\n"
                        markdown += "| 指数 | 相关系数 |\n"
                        markdown += "| ---- | -------- |\n"
                        
                        for index_code, corr in sorted(avg_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                            markdown += f"| {index_code} | {corr:.4f} |\n"
                        
                        markdown += "\n"
                    
                    # 高相关股票
                    high_corr_stocks = market_correlation.get("high_correlation_stocks", {})
                    if high_corr_stocks:
                        markdown += "### 高相关股票\n\n"
                        
                        for index_code, stocks in high_corr_stocks.items():
                            if stocks:
                                markdown += f"#### {index_code}\n\n"
                                markdown += f"{', '.join(stocks[:20])}" + (", ..." if len(stocks) > 20 else "") + "\n\n"
                    
                    # 低相关股票
                    low_corr_stocks = market_correlation.get("low_correlation_stocks", {})
                    if low_corr_stocks:
                        markdown += "### 低相关股票\n\n"
                        
                        for index_code, stocks in low_corr_stocks.items():
                            if stocks:
                                markdown += f"#### {index_code}\n\n"
                                markdown += f"{', '.join(stocks[:20])}" + (", ..." if len(stocks) > 20 else "") + "\n\n"
                
                # 保存到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown)
            
            else:
                logger.error(f"不支持的输出格式: {format_type}")
                return
                
            logger.info(f"多维度分析结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存多维度分析结果时出错: {e}")
    
    def export_to_excel(self, output_file: str) -> None:
        """
        导出分析结果到Excel
        
        Args:
            output_file: 输出Excel文件路径
        """
        logger.info(f"导出多维度分析结果到Excel: {output_file}")
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 创建Excel写入器
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 导出单股分析结果
                single_stock = self.analysis_results.get("single_stock_analysis", {})
                if single_stock:
                    # 合并所有单股评估结果
                    assessments = []
                    for stock_code, analysis in single_stock.items():
                        assessment = analysis.get("assessment", {})
                        if assessment:
                            assessments.append(assessment)
                    
                    if assessments:
                        # 创建评估DataFrame
                        assessment_df = pd.DataFrame(assessments)
                        assessment_df.to_excel(writer, sheet_name="个股评估", index=False)
                    
                    # 合并所有单股关键特征
                    all_features = []
                    for stock_code, analysis in single_stock.items():
                        stock_name = analysis.get("stock_name", "")
                        key_features = analysis.get("key_features", [])
                        
                        for feature in key_features:
                            feature_type = feature.get("type", "")
                            feature_name = feature.get("feature", "")
                            if not feature_name and "signal" in feature:
                                feature_name = f"{feature.get('indicator', '')}: {feature.get('signal', '')}"
                            
                            all_features.append({
                                "stock_code": stock_code,
                                "stock_name": stock_name,
                                "feature_type": feature_type,
                                "feature": feature_name,
                                "frequency": feature.get("frequency", 0),
                                "period": feature.get("period", "")
                            })
                    
                    if all_features:
                        # 创建特征DataFrame
                        features_df = pd.DataFrame(all_features)
                        features_df.to_excel(writer, sheet_name="个股特征", index=False)
                
                # 导出股票组分析结果
                stock_group = self.analysis_results.get("stock_group_analysis", {})
                if stock_group:
                    # 合并所有股票组评估
                    group_assessments = []
                    for group_id, analysis in stock_group.items():
                        assessment = analysis.get("assessment", {}).copy()
                        assessment["group_id"] = group_id
                        assessment["stock_count"] = analysis.get("stock_count", 0)
                        assessment["date"] = analysis.get("date", "")
                        
                        group_assessments.append(assessment)
                    
                    if group_assessments:
                        # 创建股票组评估DataFrame
                        group_df = pd.DataFrame(group_assessments)
                        group_df.to_excel(writer, sheet_name="股票组评估", index=False)
                    
                    # 合并所有股票组共性特征
                    group_features = []
                    for group_id, analysis in stock_group.items():
                        common_features = analysis.get("common_features", [])
                        
                        for feature in common_features:
                            feature_type = feature.get("type", "")
                            feature_name = feature.get("feature", "")
                            if not feature_name and "signal" in feature:
                                feature_name = f"{feature.get('indicator', '')}: {feature.get('signal', '')}"
                            
                            group_features.append({
                                "group_id": group_id,
                                "feature_type": feature_type,
                                "feature": feature_name,
                                "frequency": feature.get("frequency", 0),
                                "period": feature.get("period", "")
                            })
                    
                    if group_features:
                        # 创建共性特征DataFrame
                        features_df = pd.DataFrame(group_features)
                        features_df.to_excel(writer, sheet_name="股票组共性特征", index=False)
                
                # 导出市场相关性分析
                market_correlation = self.analysis_results.get("market_correlation", {})
                if market_correlation:
                    # 平均相关性
                    avg_correlations = market_correlation.get("avg_correlations", {})
                    if avg_correlations:
                        avg_corr_data = []
                        for index_code, corr in avg_correlations.items():
                            avg_corr_data.append({
                                "index_code": index_code,
                                "correlation": corr
                            })
                        
                        if avg_corr_data:
                            avg_corr_df = pd.DataFrame(avg_corr_data)
                            avg_corr_df.to_excel(writer, sheet_name="平均相关性", index=False)
                    
                    # 所有股票相关性
                    correlations = market_correlation.get("correlations", {})
                    if correlations:
                        all_corr_data = []
                        for stock_code, stock_corr in correlations.items():
                            for index_code, corr in stock_corr.items():
                                all_corr_data.append({
                                    "stock_code": stock_code,
                                    "index_code": index_code,
                                    "correlation": corr
                                })
                        
                        if all_corr_data:
                            all_corr_df = pd.DataFrame(all_corr_data)
                            all_corr_df.to_excel(writer, sheet_name="股票相关性", index=False)
            
            logger.info(f"多维度分析结果已导出到Excel: {output_file}")
            
        except Exception as e:
            logger.error(f"导出多维度分析结果到Excel时出错: {e}")
            
    def clear_results(self) -> None:
        """清除所有分析结果"""
        logger.info("清除多维度分析结果")
        self.analysis_results = {
            "single_stock_analysis": {},
            "stock_group_analysis": {},
            "market_correlation": {},
            "multi_period_analysis": {}
        } 