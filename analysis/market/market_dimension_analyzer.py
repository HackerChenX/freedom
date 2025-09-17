from analysis.base_analyzer import BaseAnalyzer
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

from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import Data_access_interface
from enums.kline_period import Kline_period
from utils.logger import get_logger
from utils.path_utils import get_result_dir
from indicators.complete_indicator_registry import complete_registry

# 获取日志记录器
logger = getLogger(__name__)

class MarketDimensionAnalyzer(BaseAnalyzer):
    """
    市场维度分析器 - 对整个市场进行多维度分析
    
    支持行业、概念、市值分布等多维度的市场分析，发现市场共性特征
    """
    
    def __init___119(self, data_access: Optional[Data_access_interface] = None):
        """
        初始化市场维度分析器
        
        Args:
            data_access: 数据访问接口，如果为None则从依赖注入容器获取
        """
        logger.info("初始化市场维度分析器")
        
        # 使用依赖注入架构
        self.data_access = data_access or get_service(Data_access_interface)
        
        # 使用统一指标注册系统
        self.indicator_registry = complete_registry
        
        # 存储分析结果
        self.analysis_results = {
            "market_summary": {},
            _analysis": {},
            "concept_analysis": {},
            "market_cap_analysis": {},
            "correlation_analysis": {}
        }
        
        # 结果输出目录
        self.result_dir = get_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)
        
        logger.info("市场维度分析器初始化完成")
    
    def analyze_market_overview(self, date: str) -> Dict[str, Any]:
        """
        分析市场概况
        
        Args:
            date: 分析日期
            
        Returns:
            Dict[str, Any]: 市场概况分析结果
        """
        logger.info(f"开始分析市场概况 - {date}")
        
        try:
            # 获取市场数据
            market_data = self.data_access.get_market_overview(date)
            
            if market_data.empty:
                logger.warning(f"未找到日期 {date} 的市场数据")
                return {}
            
            # 计算市场统计指标
            overview = {
                "date": date,
                "total_stocks": len(market_data),
                "up_stocks": len(market_data[market_data['change_pct'] > 0]),
                "down_stocks": len(market_data[market_data['change_pct'] < 0]),
                "flat_stocks": len(market_data[market_data['change_pct'] == 0]),
                "average_change": market_data['change_pct'].mean(),
                "median_change": market_data['change_pct'].median(),
                "max_change": market_data['change_pct'].max(),
                "min_change": market_data['change_pct'].min(),
                "total_volume": market_data['volume'].sum(),
                "total_amount": market_data['amount'].sum()
            }
            
            # 计算涨跌比例
            overview["up_ratio"] = overview["up_stocks"] / overview["total_stocks"] if overview["total_stocks"] > 0 else 0
            overview["down_ratio"] = overview["down_stocks"] / overview["total_stocks"] if overview["total_stocks"] > 0 else 0
            
            self.analysis_results["market_summary"] = overview
            
            logger.info(f"市场概况分析完成 - 总股票数: {overview['total_stocks']}, 涨跌比: {overview['up_ratio']:.2%}")
            return overview
            
        except Exception as e:
            logger.error(f"市场概况分析失败: {e}")
            return {}
    
    def analyze__distribution(self, date: str) -> Dict[str, Any]:
        """
        分析行业分布
        
        Args:
            date: 分析日期
            
        Returns:
            Dict[str, Any]: 行业分布分析结果
        """
        logger.info(f"开始分析行业分布 - {date}")
        
        try:
            # 获取行业数据
            _data = self.data_access.get__data(date)
            
            if _data.empty:
                logger.warning(f"未找到日期 {date} 的行业数据")
                return {}
            
            # 按行业统计
            _stats = {}
            for in _data[].unique():
                _stocks = _data[_data[] == ]
                
                stats = {
                    "stock_count": len(_stocks),
                    "up_count": len(_stocks[_stocks['change_pct'] > 0]),
                    "down_count": len(_stocks[_stocks['change_pct'] < 0]),
                    "average_change": _stocks['change_pct'].mean(),
                    "median_change": _stocks['change_pct'].median(),
                    "total_volume": _stocks['volume'].sum(),
                    "total_amount": _stocks['amount'].sum()
                }
                
                stats["up_ratio"] = stats["up_count"] / stats["stock_count"] if stats["stock_count"] > 0 else 0
                stats["down_ratio"] = stats["down_count"] / stats["stock_count"] if stats["stock_count"] > 0 else 0
                
                _stats[] = stats
            
            # 排序（按平均涨幅）
            sorted_industries = sorted(_stats.items(), 
                                     key=lambda x: x[1]['average_change'], 
                                     reverse=True)
            
            analysis_result = {
                "date": date,
                _count": len(_stats),
                _stats": dict(sorted_industries),
                "top_performers": sorted_industries[:10],
                "worst_performers": sorted_industries[-10:]
            }
            
            self.analysis_results[_analysis"] = analysis_result
            
            logger.info(f"行业分布分析完成 - 行业数: {analysis_result[_count']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"行业分布分析失败: {e}")
            return {}
    
    def analyze_market_cap_distribution(self, date: str) -> Dict[str, Any]:
        """
        分析市值分布
        
        Args:
            date: 分析日期
            
        Returns:
            Dict[str, Any]: 市值分布分析结果
        """
        logger.info(f"开始分析市值分布 - {date}")
        
        try:
            # 获取市值数据
            market_cap_data = self.data_access.get_market_cap_data(date)
            
            if market_cap_data.empty:
                logger.warning(f"未找到日期 {date} 的市值数据")
                return {}
            
            # 定义市值分类
            def classify_market_cap(market_cap):
                if market_cap >= 1000:  # 大于1000亿
                    return "超大盘"
                elif market_cap >= 300:  # 300-1000亿
                    return "大盘"
                elif market_cap >= 100:  # 100-300亿
                    return "中盘"
                elif market_cap >= 50:   # 50-100亿
                    return "小盘"
                else:                    # 小于50亿
                    return "微盘"
            
            # 添加市值分类
            market_cap_data['cap_category'] = market_cap_data['market_cap'].apply(classify_market_cap)
            
            # 按市值分类统计
            cap_stats = {}
            for category in market_cap_data['cap_category'].unique():
                category_stocks = market_cap_data[market_cap_data['cap_category'] == category]
                
                stats = {
                    "stock_count": len(category_stocks),
                    "up_count": len(category_stocks[category_stocks['change_pct'] > 0]),
                    "down_count": len(category_stocks[category_stocks['change_pct'] < 0]),
                    "average_change": category_stocks['change_pct'].mean(),
                    "median_change": category_stocks['change_pct'].median(),
                    "total_market_cap": category_stocks['market_cap'].sum(),
                    "average_market_cap": category_stocks['market_cap'].mean()
                }
                
                stats["up_ratio"] = stats["up_count"] / stats["stock_count"] if stats["stock_count"] > 0 else 0
                stats["down_ratio"] = stats["down_count"] / stats["stock_count"] if stats["stock_count"] > 0 else 0
                
                cap_stats[category] = stats
            
            analysis_result = {
                "date": date,
                "cap_categories": list(cap_stats.keys()),
                "cap_stats": cap_stats,
                "total_market_cap": market_cap_data['market_cap'].sum(),
                "average_market_cap": market_cap_data['market_cap'].mean(),
                "median_market_cap": market_cap_data['market_cap'].median()
            }
            
            self.analysis_results["market_cap_analysis"] = analysis_result
            
            logger.info(f"市值分布分析完成 - 总市值: {analysis_result['total_market_cap']:.2f}亿")
            return analysis_result
            
        except Exception as e:
            logger.error(f"市值分布分析失败: {e}")
            return {}
    
    def analyze_correlation_patterns(self, start_date: str, end_date: str, 
                                   sample_size: int = 100) -> Dict[str, Any]:
        """
        分析市场相关性模式
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            sample_size: 样本大小
            
        Returns:
            Dict[str, Any]: 相关性分析结果
        """
        logger.info(f"开始分析相关性模式 - {start_date} 到 {end_date}")
        
        try:
            # 获取样本股票数据
            sample_data = self.data_access.get_sample_stocks_data(
                start_date, end_date, sample_size
            )
            
            if sample_data.empty:
                logger.warning("未找到样本股票数据")
                return {}
            
            # 构建价格变化矩阵
            s = sample_data.pivot(
                index='date', 
                columns='code', 
                values='change_pct'
            )
            
            # 计算相关性矩阵
            correlation_matrix = s.corr()
            
            # 分析相关性统计
            correlation_stats = {
                "average_correlation": correlation_matrix.mean().mean(),
                "median_correlation": correlation_matrix.median().median(),
                "max_correlation": correlation_matrix.max().max(),
                "min_correlation": correlation_matrix.min().min(),
                "high_correlation_pairs": 0,
                "low_correlation_pairs": 0
            }
            
            # 统计高相关性和低相关性对数
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if corr_value > 0.7:
                        correlation_stats["high_correlation_pairs"] += 1
                    elif corr_value < 0.3:
                        correlation_stats["low_correlation_pairs"] += 1
            
            analysis_result = {
                "period": f"{start_date} 到 {end_date}",
                "sample_size": sample_size,
                "correlation_stats": correlation_stats,
                "correlation_matrix": correlation_matrix.to_dict()
            }
            
            self.analysis_results["correlation_analysis"] = analysis_result
            
            logger.info(f"相关性分析完成 - 平均相关性: {correlation_stats['average_correlation']:.4f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"相关性分析失败: {e}")
            return {}
    
    def generate_comprehensive_report(self, date: str) -> Dict[str, Any]:
        """
        生成综合分析报告
        
        Args:
            date: 分析日期
            
        Returns:
            Dict[str, Any]: 综合分析报告
        """
        logger.info(f"开始生成综合分析报告 - {date}")
        
        try:
            # 执行各项分析
            market_overview = self.analyze_market_overview(date)
            _analysis = self.analyze__distribution(date)
            market_cap_analysis = self.analyze_market_cap_distribution(date)
            
            # 生成综合报告
            comprehensive_report = {
                "analysis_date": date,
                "generation_time": datetime.now().isoformat(),
                "market_overview": market_overview,
                _analysis": _analysis,
                "market_cap_analysis": market_cap_analysis,
                "summary": self._generate_summary_Market_Dimension_Analyzer(market_overview, _analysis, market_cap_analysis)
            }
            
            # 保存报告
            self._save_report(comprehensive_report, date)
            
            logger.info(f"综合分析报告生成完成 - {date}")
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"生成综合分析报告失败: {e}")
            return {}
    
    def _generate_summary_Market_Dimension_Analyzer(self, market_overview: Dict[str, Any], 
                         _analysis: Dict[str, Any],
                         market_cap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成分析总结
        
        Args:
            market_overview: 市场概况
            _analysis: 行业分析
            market_cap_analysis: 市值分析
            
        Returns:
            Dict[str, Any]: 分析总结
        """
        summary = {
            "market_sentiment": "中性",
            "dominant_trends": [],
            "key_observations": [],
            "risk_factors": []
        }
        
        try:
            # 判断市场情绪
            if market_overview.get("up_ratio", 0) > 0.6:
                summary["market_sentiment"] = "乐观"
            elif market_overview.get("up_ratio", 0) < 0.4:
                summary["market_sentiment"] = "悲观"
            
            # 识别主导趋势
            if market_overview.get("average_change", 0) > 2:
                summary["dominant_trends"].append("强势上涨")
            elif market_overview.get("average_change", 0) < -2:
                summary["dominant_trends"].append("明显下跌")
            
            # 关键观察
            if _analysis.get(_count", 0) > 0:
                top_= _analysis.get("top_performers", [])
                if top_}"
                    )
            
            # 风险因素
            if market_overview.get("down_ratio", 0) > 0.7:
                summary["risk_factors"].append("市场普遍下跌")
            
        except Exception as e:
            logger.warning(f"生成分析总结时出现错误: {e}")
        
        return summary
    
    def _save_report(self, report: Dict[str, Any], date: str):
        """
        保存分析报告
        
        Args:
            report: 分析报告
            date: 日期
        """
        try:
            filename = f"market_analysis_{date.replace('-', '')}.json"
            filepath = os.path.join(self.result_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"分析报告已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存分析报告失败: {e}")
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """
        获取所有分析结果
        
        Returns:
            Dict[str, Any]: 分析结果
        """
        return self.analysis_results.copy()
    
    def clear_results_Market_Dimension_Analyzer(self):
        """清空分析结果"""
        self.analysis_results = {
            "market_summary": {},
            _analysis": {},
            "concept_analysis": {},
            "market_cap_analysis": {},
            "correlation_analysis": {}
        }
        logger.info("分析结果已清空") 