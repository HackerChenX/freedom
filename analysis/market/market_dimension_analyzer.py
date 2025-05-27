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
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.clickhouse_db import get_clickhouse_db, get_default_config
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from utils.path_utils import get_result_dir
from indicators.factory import IndicatorFactory

# 获取日志记录器
logger = get_logger(__name__)

class MarketDimensionAnalyzer:
    """
    市场维度分析器 - 对整个市场进行多维度分析
    
    支持行业、概念、市值分布等多维度的市场分析，发现市场共性特征
    """
    
    def __init__(self):
        """初始化市场维度分析器"""
        logger.info("初始化市场维度分析器")
        
        # 获取数据库连接
        config = get_default_config()
        self.ch_db = get_clickhouse_db(config=config)
        
        # 创建指标工厂
        self.indicator_factory = IndicatorFactory()
        
        # 存储分析结果
        self.analysis_results = {
            "market_summary": {},
            "industry_analysis": {},
            "concept_analysis": {},
            "market_cap_analysis": {},
            "correlation_analysis": {}
        }
        
        # 结果输出目录
        self.result_dir = get_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)
        
        logger.info("市场维度分析器初始化完成")
    
    def analyze_market_overview(self, date: str = None) -> Dict[str, Any]:
        """
        分析市场整体概况
        
        Args:
            date: 分析日期，格式YYYYMMDD，默认为最近交易日
            
        Returns:
            Dict: 市场概况分析结果
        """
        logger.info(f"开始分析市场整体概况: {date or '最近交易日'}")
        
        # 如果未指定日期，获取最近交易日
        if date is None:
            date = self._get_latest_trade_date()
        
        try:
            # 获取市场整体数据
            sql = f"""
            SELECT 
                COUNT(*) as total_stocks,
                SUM(CASE WHEN price_change > 0 THEN 1 ELSE 0 END) as up_stocks,
                SUM(CASE WHEN price_change < 0 THEN 1 ELSE 0 END) as down_stocks,
                SUM(CASE WHEN price_change = 0 THEN 1 ELSE 0 END) as flat_stocks,
                AVG(price_change) as avg_change,
                SUM(volume) as total_volume,
                SUM(amount) as total_amount
            FROM stock_daily
            WHERE date = '{date}'
            """
            market_data = self.ch_db.query_df(sql)
            
            if market_data.empty:
                logger.warning(f"未找到日期 {date} 的市场数据")
                return {}
            
            # 计算涨跌比例
            up_ratio = market_data['up_stocks'].iloc[0] / market_data['total_stocks'].iloc[0]
            down_ratio = market_data['down_stocks'].iloc[0] / market_data['total_stocks'].iloc[0]
            
            # 获取涨停和跌停数量
            sql_limit = f"""
            SELECT 
                SUM(CASE WHEN price_change / pre_close >= 0.099 THEN 1 ELSE 0 END) as limit_up,
                SUM(CASE WHEN price_change / pre_close <= -0.099 THEN 1 ELSE 0 END) as limit_down
            FROM stock_daily
            WHERE date = '{date}'
            """
            limit_data = self.ch_db.query_df(sql_limit)
            
            # 计算市场强度指标
            market_strength = up_ratio - down_ratio
            
            # 构建结果
            result = {
                "date": date,
                "total_stocks": int(market_data['total_stocks'].iloc[0]),
                "up_stocks": int(market_data['up_stocks'].iloc[0]),
                "down_stocks": int(market_data['down_stocks'].iloc[0]),
                "flat_stocks": int(market_data['flat_stocks'].iloc[0]),
                "up_ratio": float(up_ratio),
                "down_ratio": float(down_ratio),
                "avg_change": float(market_data['avg_change'].iloc[0]),
                "total_volume": float(market_data['total_volume'].iloc[0]),
                "total_amount": float(market_data['total_amount'].iloc[0]),
                "limit_up": int(limit_data['limit_up'].iloc[0]) if not limit_data.empty else 0,
                "limit_down": int(limit_data['limit_down'].iloc[0]) if not limit_data.empty else 0,
                "market_strength": float(market_strength)
            }
            
            # 市场状态判断
            if market_strength > 0.3:
                result["market_status"] = "强势上涨"
            elif market_strength > 0.1:
                result["market_status"] = "上涨"
            elif market_strength > -0.1:
                result["market_status"] = "震荡"
            elif market_strength > -0.3:
                result["market_status"] = "下跌"
            else:
                result["market_status"] = "弱势下跌"
            
            # 保存到分析结果
            self.analysis_results["market_summary"] = result
            
            logger.info(f"市场整体概况分析完成: {result['market_status']}")
            return result
            
        except Exception as e:
            logger.error(f"分析市场整体概况时出错: {e}")
            return {}
    
    def analyze_industry_performance(self, date: str = None) -> Dict[str, Any]:
        """
        分析行业表现
        
        Args:
            date: 分析日期，格式YYYYMMDD，默认为最近交易日
            
        Returns:
            Dict: 行业表现分析结果
        """
        logger.info(f"开始分析行业表现: {date or '最近交易日'}")
        
        # 如果未指定日期，获取最近交易日
        if date is None:
            date = self._get_latest_trade_date()
        
        try:
            # 获取行业表现数据
            sql = f"""
            SELECT 
                industry,
                COUNT(*) as stock_count,
                AVG(price_change) as avg_change,
                SUM(CASE WHEN price_change > 0 THEN 1 ELSE 0 END) / COUNT(*) as up_ratio,
                AVG(volume) as avg_volume,
                AVG(amount) as avg_amount
            FROM stock_daily
            WHERE date = '{date}' AND industry != ''
            GROUP BY industry
            ORDER BY avg_change DESC
            """
            industry_data = self.ch_db.query_df(sql)
            
            if industry_data.empty:
                logger.warning(f"未找到日期 {date} 的行业数据")
                return {}
            
            # 构建结果
            industries = []
            for _, row in industry_data.iterrows():
                industries.append({
                    "industry": row['industry'],
                    "stock_count": int(row['stock_count']),
                    "avg_change": float(row['avg_change']),
                    "up_ratio": float(row['up_ratio']),
                    "avg_volume": float(row['avg_volume']),
                    "avg_amount": float(row['avg_amount'])
                })
            
            # 计算行业强弱排名
            sorted_industries = sorted(industries, key=lambda x: x['avg_change'], reverse=True)
            
            # 分类为强势、中性和弱势行业
            strong_industries = [ind for ind in sorted_industries if ind['avg_change'] > 1.0]
            neutral_industries = [ind for ind in sorted_industries if -1.0 <= ind['avg_change'] <= 1.0]
            weak_industries = [ind for ind in sorted_industries if ind['avg_change'] < -1.0]
            
            result = {
                "date": date,
                "all_industries": industries,
                "industry_count": len(industries),
                "strong_industries": strong_industries,
                "neutral_industries": neutral_industries,
                "weak_industries": weak_industries,
                "top_industries": sorted_industries[:5],
                "bottom_industries": sorted_industries[-5:] if len(sorted_industries) >= 5 else []
            }
            
            # 保存到分析结果
            self.analysis_results["industry_analysis"] = result
            
            logger.info(f"行业表现分析完成: 共 {len(industries)} 个行业")
            return result
            
        except Exception as e:
            logger.error(f"分析行业表现时出错: {e}")
            return {}
    
    def analyze_market_cap_distribution(self, date: str = None) -> Dict[str, Any]:
        """
        分析市值分布
        
        Args:
            date: 分析日期，格式YYYYMMDD，默认为最近交易日
            
        Returns:
            Dict: 市值分布分析结果
        """
        logger.info(f"开始分析市值分布: {date or '最近交易日'}")
        
        # 如果未指定日期，获取最近交易日
        if date is None:
            date = self._get_latest_trade_date()
        
        try:
            # 获取市值分布数据
            sql = f"""
            SELECT 
                CASE 
                    WHEN market_cap < 50 THEN '小市值'
                    WHEN market_cap < 200 THEN '中小市值'
                    WHEN market_cap < 500 THEN '中市值'
                    WHEN market_cap < 1000 THEN '中大市值'
                    ELSE '大市值'
                END as cap_group,
                COUNT(*) as stock_count,
                AVG(price_change) as avg_change,
                SUM(CASE WHEN price_change > 0 THEN 1 ELSE 0 END) / COUNT(*) as up_ratio,
                AVG(volume) as avg_volume,
                AVG(amount) as avg_amount
            FROM stock_daily
            WHERE date = '{date}' AND market_cap > 0
            GROUP BY cap_group
            ORDER BY 
                CASE 
                    WHEN cap_group = '小市值' THEN 1
                    WHEN cap_group = '中小市值' THEN 2
                    WHEN cap_group = '中市值' THEN 3
                    WHEN cap_group = '中大市值' THEN 4
                    WHEN cap_group = '大市值' THEN 5
                END
            """
            cap_data = self.ch_db.query_df(sql)
            
            if cap_data.empty:
                logger.warning(f"未找到日期 {date} 的市值数据")
                return {}
            
            # 构建结果
            cap_groups = []
            for _, row in cap_data.iterrows():
                cap_groups.append({
                    "cap_group": row['cap_group'],
                    "stock_count": int(row['stock_count']),
                    "avg_change": float(row['avg_change']),
                    "up_ratio": float(row['up_ratio']),
                    "avg_volume": float(row['avg_volume']),
                    "avg_amount": float(row['avg_amount'])
                })
            
            # 分析市值板块轮动
            result = {
                "date": date,
                "cap_groups": cap_groups,
                "leading_group": max(cap_groups, key=lambda x: x['avg_change'])['cap_group'],
                "lagging_group": min(cap_groups, key=lambda x: x['avg_change'])['cap_group']
            }
            
            # 保存到分析结果
            self.analysis_results["market_cap_analysis"] = result
            
            logger.info(f"市值分布分析完成: 领先板块 {result['leading_group']}")
            return result
            
        except Exception as e:
            logger.error(f"分析市值分布时出错: {e}")
            return {}
    
    def analyze_index_correlation(self, end_date: str = None, days: int = 30) -> Dict[str, Any]:
        """
        分析指数相关性
        
        Args:
            end_date: 结束日期，格式YYYYMMDD，默认为最近交易日
            days: 分析的天数，默认为30天
            
        Returns:
            Dict: 指数相关性分析结果
        """
        logger.info(f"开始分析指数相关性: {end_date or '最近交易日'}, 时间范围: {days}天")
        
        # 如果未指定日期，获取最近交易日
        if end_date is None:
            end_date = self._get_latest_trade_date()
            
        try:
            # 计算开始日期
            end_date_obj = datetime.strptime(end_date, "%Y%m%d")
            start_date_obj = end_date_obj - timedelta(days=days)
            start_date = start_date_obj.strftime("%Y%m%d")
            
            # 主要指数代码
            index_codes = ['000001.SH', '399001.SZ', '399006.SZ', '000016.SH', '000300.SH', '000905.SH']
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
                    index_data[code] = df
            
            # 如果数据不足，返回空结果
            if len(index_data) < 2:
                logger.warning("指数数据不足，无法计算相关性")
                return {}
            
            # 计算相关性矩阵
            index_returns = {}
            for code, df in index_data.items():
                if len(df) > 1:
                    # 计算日收益率
                    df['return'] = df['close'].pct_change()
                    index_returns[code] = df.dropna()
            
            # 合并数据
            return_data = pd.DataFrame()
            for code, df in index_returns.items():
                return_data[code] = df['return']
            
            # 计算相关性
            correlation_matrix = return_data.corr()
            
            # 构建结果
            result = {
                "start_date": start_date,
                "end_date": end_date,
                "days": days,
                "index_codes": index_codes,
                "correlation_matrix": correlation_matrix.to_dict()
            }
            
            # 提取高相关性和低相关性对
            correlations = []
            for i in range(len(index_codes)):
                for j in range(i+1, len(index_codes)):
                    code1 = index_codes[i]
                    code2 = index_codes[j]
                    if code1 in correlation_matrix.index and code2 in correlation_matrix.columns:
                        corr_value = correlation_matrix.loc[code1, code2]
                        correlations.append({
                            "index1": code1,
                            "index2": code2,
                            "correlation": float(corr_value)
                        })
            
            # 排序相关性
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            result["correlations"] = correlations
            result["high_correlation_pairs"] = [c for c in correlations if abs(c['correlation']) > 0.8]
            result["low_correlation_pairs"] = [c for c in correlations if abs(c['correlation']) < 0.5]
            
            # 保存到分析结果
            self.analysis_results["correlation_analysis"] = result
            
            logger.info(f"指数相关性分析完成: 分析了 {len(index_codes)} 个指数")
            return result
            
        except Exception as e:
            logger.error(f"分析指数相关性时出错: {e}")
            return {}
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        获取综合分析报告
        
        Returns:
            Dict: 综合分析报告
        """
        logger.info("生成市场综合分析报告")
        
        try:
            # 整合各模块分析结果
            report = {
                "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_summary": self.analysis_results.get("market_summary", {}),
                "industry_analysis": {
                    "top_industries": self.analysis_results.get("industry_analysis", {}).get("top_industries", []),
                    "bottom_industries": self.analysis_results.get("industry_analysis", {}).get("bottom_industries", [])
                },
                "market_cap_analysis": self.analysis_results.get("market_cap_analysis", {}),
                "correlation_analysis": {
                    "high_correlation_pairs": self.analysis_results.get("correlation_analysis", {}).get("high_correlation_pairs", []),
                    "low_correlation_pairs": self.analysis_results.get("correlation_analysis", {}).get("low_correlation_pairs", [])
                }
            }
            
            # 生成市场综合判断
            market_summary = self.analysis_results.get("market_summary", {})
            industry_analysis = self.analysis_results.get("industry_analysis", {})
            
            market_status = market_summary.get("market_status", "未知")
            market_strength = market_summary.get("market_strength", 0)
            strong_industries_count = len(industry_analysis.get("strong_industries", []))
            weak_industries_count = len(industry_analysis.get("weak_industries", []))
            
            # 市场整体判断
            if market_status in ["强势上涨", "上涨"] and strong_industries_count > weak_industries_count:
                market_judgment = "市场整体向好，行业普涨，建议积极参与"
            elif market_status == "震荡" and strong_industries_count > 0:
                market_judgment = "市场震荡，部分行业表现强势，建议关注强势板块"
            elif market_status in ["下跌", "弱势下跌"]:
                market_judgment = "市场疲软，建议减仓观望，关注超跌反弹机会"
            else:
                market_judgment = "市场无明显方向，建议谨慎操作，关注市场热点"
            
            report["market_judgment"] = market_judgment
            
            # 投资建议
            top_industries = industry_analysis.get("top_industries", [])
            cap_analysis = self.analysis_results.get("market_cap_analysis", {})
            leading_group = cap_analysis.get("leading_group", "")
            
            investment_suggestions = []
            
            # 基于行业表现的建议
            if top_industries:
                industries_str = ", ".join([ind["industry"] for ind in top_industries[:3]])
                investment_suggestions.append(f"重点关注表现强势的行业: {industries_str}")
            
            # 基于市值分布的建议
            if leading_group:
                investment_suggestions.append(f"当前市场偏好 {leading_group} 股票")
            
            # 基于市场状态的建议
            if market_status in ["强势上涨", "上涨"]:
                investment_suggestions.append("可考虑重点配置强势股和行业龙头")
            elif market_status == "震荡":
                investment_suggestions.append("适合波段操作，高抛低吸，控制仓位")
            else:
                investment_suggestions.append("降低仓位，保持现金，等待市场企稳")
            
            report["investment_suggestions"] = investment_suggestions
            
            logger.info("市场综合分析报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"生成市场综合分析报告时出错: {e}")
            return {}
    
    def save_results(self, output_file: str, format_type: str = "json") -> None:
        """
        保存分析结果
        
        Args:
            output_file: 输出文件路径
            format_type: 输出格式类型，支持json和markdown
        """
        logger.info(f"保存市场分析结果到: {output_file}")
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if format_type.lower() == "json":
                # 获取综合报告
                report = self.get_comprehensive_report()
                
                # 构建完整结果
                full_results = {
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "comprehensive_report": report,
                    "detailed_results": self.analysis_results
                }
                
                # 保存到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(full_results, f, indent=2, ensure_ascii=False)
                
            elif format_type.lower() == "markdown":
                # 获取综合报告
                report = self.get_comprehensive_report()
                
                # 构建Markdown内容
                markdown = "# 市场多维度分析报告\n\n"
                markdown += f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # 市场概况
                market_summary = self.analysis_results.get("market_summary", {})
                if market_summary:
                    markdown += "## 市场概况\n\n"
                    markdown += f"市场状态: **{market_summary.get('market_status', '未知')}**\n\n"
                    markdown += f"上涨家数: {market_summary.get('up_stocks', 0)}, "
                    markdown += f"下跌家数: {market_summary.get('down_stocks', 0)}, "
                    markdown += f"平盘家数: {market_summary.get('flat_stocks', 0)}\n\n"
                    markdown += f"平均涨幅: {market_summary.get('avg_change', 0):.2f}%\n\n"
                    markdown += f"涨停家数: {market_summary.get('limit_up', 0)}, "
                    markdown += f"跌停家数: {market_summary.get('limit_down', 0)}\n\n"
                
                # 行业分析
                industry_analysis = self.analysis_results.get("industry_analysis", {})
                if industry_analysis:
                    markdown += "## 行业分析\n\n"
                    
                    # 强势行业
                    top_industries = industry_analysis.get("top_industries", [])
                    if top_industries:
                        markdown += "### 表现最好的行业\n\n"
                        markdown += "| 行业 | 平均涨幅 | 上涨比例 | 股票数量 |\n"
                        markdown += "| ---- | -------- | -------- | -------- |\n"
                        
                        for ind in top_industries:
                            markdown += f"| {ind['industry']} | {ind['avg_change']:.2f}% | "
                            markdown += f"{ind['up_ratio']*100:.2f}% | {ind['stock_count']} |\n"
                        
                        markdown += "\n"
                    
                    # 弱势行业
                    bottom_industries = industry_analysis.get("bottom_industries", [])
                    if bottom_industries:
                        markdown += "### 表现最差的行业\n\n"
                        markdown += "| 行业 | 平均涨幅 | 上涨比例 | 股票数量 |\n"
                        markdown += "| ---- | -------- | -------- | -------- |\n"
                        
                        for ind in bottom_industries:
                            markdown += f"| {ind['industry']} | {ind['avg_change']:.2f}% | "
                            markdown += f"{ind['up_ratio']*100:.2f}% | {ind['stock_count']} |\n"
                        
                        markdown += "\n"
                
                # 市值分析
                cap_analysis = self.analysis_results.get("market_cap_analysis", {})
                if cap_analysis:
                    markdown += "## 市值分布分析\n\n"
                    
                    cap_groups = cap_analysis.get("cap_groups", [])
                    if cap_groups:
                        markdown += "| 市值分组 | 平均涨幅 | 上涨比例 | 股票数量 |\n"
                        markdown += "| -------- | -------- | -------- | -------- |\n"
                        
                        for cap in cap_groups:
                            markdown += f"| {cap['cap_group']} | {cap['avg_change']:.2f}% | "
                            markdown += f"{cap['up_ratio']*100:.2f}% | {cap['stock_count']} |\n"
                        
                        markdown += "\n"
                        
                        markdown += f"领先板块: **{cap_analysis.get('leading_group', '')}**\n\n"
                        markdown += f"落后板块: **{cap_analysis.get('lagging_group', '')}**\n\n"
                
                # 综合判断
                markdown += "## 综合判断\n\n"
                markdown += f"{report.get('market_judgment', '')}\n\n"
                
                # 投资建议
                markdown += "## 投资建议\n\n"
                for suggestion in report.get('investment_suggestions', []):
                    markdown += f"- {suggestion}\n"
                    
                # 保存到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown)
            
            else:
                logger.error(f"不支持的输出格式: {format_type}")
                return
                
            logger.info(f"市场分析结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存市场分析结果时出错: {e}")
    
    def _get_latest_trade_date(self) -> str:
        """
        获取最近的交易日
        
        Returns:
            str: 最近交易日，格式YYYYMMDD
        """
        try:
            sql = """
            SELECT MAX(date) as latest_date
            FROM trading_calendar
            WHERE is_open = 1 AND date <= CURRENT_DATE
            """
            result = self.ch_db.query_df(sql)
            
            if result.empty or pd.isna(result['latest_date'].iloc[0]):
                # 如果查询失败，返回今天的日期
                return datetime.now().strftime("%Y%m%d")
            
            # 将日期格式化为YYYYMMDD
            return pd.to_datetime(result['latest_date'].iloc[0]).strftime("%Y%m%d")
            
        except Exception as e:
            logger.error(f"获取最近交易日时出错: {e}")
            # 返回今天的日期作为后备
            return datetime.now().strftime("%Y%m%d") 