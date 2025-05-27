#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Union

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.clickhouse_db import get_clickhouse_db, get_default_config
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir
from indicators.factory import IndicatorFactory

# 获取日志记录器
logger = get_logger(__name__)

class MultiDimensionAnalyzer:
    """
    多维度分析器 - 对股票进行多个维度的技术分析
    
    支持多周期、多指标组合分析，评估股票在不同维度上的表现
    """
    
    def __init__(self):
        """初始化多维度分析器"""
        logger.info("初始化多维度分析器")
        
        # 获取数据库连接
        config = get_default_config()
        self.ch_db = get_clickhouse_db(config=config)
        
        # 创建指标工厂
        self.indicator_factory = IndicatorFactory()
        
        # 存储分析结果
        self.analysis_results = []
        
        # 结果输出目录
        self.result_dir = get_backtest_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)
        
        logger.info("多维度分析器初始化完成")
        
    def analyze_stock(self, stock_code: str, analysis_date: str = None, 
                      periods: List[KlinePeriod] = None, 
                      indicators: List[str] = None) -> Dict[str, Any]:
        """
        对单个股票进行多维度分析
        
        Args:
            stock_code: 股票代码
            analysis_date: 分析日期，默认为最近交易日
            periods: 要分析的周期列表，默认为日线、周线和60分钟线
            indicators: 要分析的指标列表，默认为所有基础指标
            
        Returns:
            Dict: 分析结果
        """
        # 设置默认值
        if periods is None:
            periods = [KlinePeriod.DAILY, KlinePeriod.WEEKLY, KlinePeriod.MIN_60]
            
        if indicators is None:
            indicators = ["MA", "MACD", "KDJ", "RSI", "BOLL", "VOL"]
            
        # 如果未指定日期，获取最近交易日
        if analysis_date is None:
            today = datetime.now().strftime("%Y%m%d")
            analysis_date = today
            
        logger.info(f"开始分析股票 {stock_code} 在 {analysis_date} 的多维度指标")
        
        # 创建结果字典
        result = {
            "code": stock_code,
            "name": "",
            "industry": "",
            "analysis_date": analysis_date,
            "dimensions": {}
        }
        
        # 分析各个周期
        for period in periods:
            period_result = self._analyze_period(stock_code, analysis_date, period, indicators)
            if period_result:
                # 只有首次获取股票名称和行业
                if not result["name"] and "name" in period_result:
                    result["name"] = period_result["name"]
                    result["industry"] = period_result.get("industry", "")
                    
                # 保存周期分析结果
                result["dimensions"][period.name] = period_result
        
        # 添加到分析结果
        self.analysis_results.append(result)
        
        return result 

    def _analyze_period(self, stock_code: str, analysis_date: str, 
                       period: KlinePeriod, indicators: List[str]) -> Dict[str, Any]:
        """
        分析指定周期的技术指标
        
        Args:
            stock_code: 股票代码
            analysis_date: 分析日期
            period: 周期类型
            indicators: 要分析的指标列表
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 计算日期范围
            analysis_date_obj = datetime.strptime(analysis_date, "%Y%m%d")
            start_date = (analysis_date_obj - timedelta(days=120)).strftime("%Y%m%d")
            end_date = analysis_date
            
            # 从数据库获取K线数据
            logger.info(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的 {period.name} 数据")
            data = self._get_kline_data(stock_code, period, start_date, end_date)
            
            if data is None or len(data) == 0:
                logger.warning(f"未找到 {stock_code} 的 {period.name} 数据")
                return {}
                
            # 计算指标
            result = {
                "name": data["name"].iloc[0] if "name" in data.columns else "",
                "industry": data["industry"].iloc[0] if "industry" in data.columns else "",
                "indicators": {},
                "signals": [],
                "patterns": []
            }
            
            # 找到分析日期对应的索引
            analysis_index = None
            for i, date in enumerate(data["date"]):
                date_str = pd.to_datetime(date).strftime("%Y%m%d")
                if date_str == analysis_date:
                    analysis_index = i
                    break
                    
            if analysis_index is None:
                # 如果找不到精确匹配，使用最后一个交易日
                logger.warning(f"未找到 {analysis_date} 的数据，使用最近交易日")
                analysis_index = len(data) - 1
                
            # 分析各个指标
            for indicator_name in indicators:
                indicator_result = self._calculate_indicator(
                    data, indicator_name, analysis_index)
                if indicator_result:
                    result["indicators"][indicator_name] = indicator_result
                    
                    # 收集信号
                    if "signals" in indicator_result:
                        for signal in indicator_result["signals"]:
                            if signal not in result["signals"]:
                                result["signals"].append(signal)
                                
                    # 收集形态
                    if "patterns" in indicator_result:
                        for pattern in indicator_result["patterns"]:
                            if pattern not in result["patterns"]:
                                result["patterns"].append(pattern)
            
            return result
            
        except Exception as e:
            logger.error(f"分析 {stock_code} 的 {period.name} 数据时出错: {e}")
            return {} 

    def _get_kline_data(self, stock_code: str, period: KlinePeriod, 
                       start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            stock_code: 股票代码
            period: 周期类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: K线数据
        """
        try:
            # 从数据库获取数据
            data = self.ch_db.get_stock_info(stock_code, period.value, start_date, end_date)
            
            if not data or len(data) == 0:
                logger.warning(f"未找到 {stock_code} 的 {period.name} 数据")
                return pd.DataFrame()
                
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=[
                'code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                'volume', 'turnover_rate', 'price_change', 'price_range', 'industry'
            ])
            
            # 转换日期列为日期类型
            df['date'] = pd.to_datetime(df['date'])
            
            # 排序数据
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {stock_code} 的 {period.name} 数据时出错: {e}")
            return pd.DataFrame()
            
    def _calculate_indicator(self, data: pd.DataFrame, indicator_name: str, 
                           analysis_index: int) -> Dict[str, Any]:
        """
        计算指标并进行分析
        
        Args:
            data: K线数据
            indicator_name: 指标名称
            analysis_index: 分析日期索引
            
        Returns:
            Dict: 指标分析结果
        """
        try:
            # 创建指标实例
            indicator = self.indicator_factory.create_indicator(indicator_name)
            
            # 计算指标值
            indicator_values = indicator.compute(data)
            
            # 获取分析日期的指标值
            result = {
                "values": {},
                "signals": [],
                "patterns": []
            }
            
            # 提取指标值
            for col in indicator_values.columns:
                if analysis_index < len(indicator_values[col]):
                    value = indicator_values[col].iloc[analysis_index]
                    # 转换numpy类型为Python原生类型
                    if isinstance(value, (np.integer, np.floating, np.bool_)):
                        value = value.item()
                    result["values"][col] = value
            
            # 生成信号
            signals = indicator.generate_signals(indicator_values)
            
            # 提取信号
            for col in signals.columns:
                if analysis_index < len(signals[col]) and signals[col].iloc[analysis_index]:
                    signal_name = f"{indicator_name}_{col}"
                    result["signals"].append(signal_name)
            
            # 识别形态
            patterns = self._identify_patterns(indicator_name, indicator_values, analysis_index)
            if patterns:
                result["patterns"] = patterns
            
            return result
            
        except Exception as e:
            logger.error(f"计算 {indicator_name} 指标时出错: {e}")
            return {} 

    def _identify_patterns(self, indicator_name: str, indicator_values: pd.DataFrame, 
                         analysis_index: int) -> List[str]:
        """
        识别技术形态
        
        Args:
            indicator_name: 指标名称
            indicator_values: 指标值
            analysis_index: 分析日期索引
            
        Returns:
            List[str]: 识别到的形态列表
        """
        patterns = []
        
        try:
            # 移动平均线形态
            if indicator_name == "MA":
                # MA5上穿MA10
                if (analysis_index > 0 and 
                    "MA5" in indicator_values.columns and 
                    "MA10" in indicator_values.columns):
                    
                    ma5_now = indicator_values["MA5"].iloc[analysis_index]
                    ma10_now = indicator_values["MA10"].iloc[analysis_index]
                    ma5_prev = indicator_values["MA5"].iloc[analysis_index-1]
                    ma10_prev = indicator_values["MA10"].iloc[analysis_index-1]
                    
                    if ma5_prev < ma10_prev and ma5_now > ma10_now:
                        patterns.append("MA5上穿MA10")
                        
                # MA5上穿MA20
                if (analysis_index > 0 and 
                    "MA5" in indicator_values.columns and 
                    "MA20" in indicator_values.columns):
                    
                    ma5_now = indicator_values["MA5"].iloc[analysis_index]
                    ma20_now = indicator_values["MA20"].iloc[analysis_index]
                    ma5_prev = indicator_values["MA5"].iloc[analysis_index-1]
                    ma20_prev = indicator_values["MA20"].iloc[analysis_index-1]
                    
                    if ma5_prev < ma20_prev and ma5_now > ma20_now:
                        patterns.append("MA5上穿MA20")
                        
                # 多头排列
                if ("MA5" in indicator_values.columns and 
                    "MA10" in indicator_values.columns and 
                    "MA20" in indicator_values.columns and
                    "MA60" in indicator_values.columns):
                    
                    ma5 = indicator_values["MA5"].iloc[analysis_index]
                    ma10 = indicator_values["MA10"].iloc[analysis_index]
                    ma20 = indicator_values["MA20"].iloc[analysis_index]
                    ma60 = indicator_values["MA60"].iloc[analysis_index]
                    
                    if ma5 > ma10 > ma20 > ma60:
                        patterns.append("均线多头排列")
            
            # MACD形态
            elif indicator_name == "MACD":
                # MACD金叉
                if (analysis_index > 0 and 
                    "DIF" in indicator_values.columns and 
                    "DEA" in indicator_values.columns):
                    
                    dif_now = indicator_values["DIF"].iloc[analysis_index]
                    dea_now = indicator_values["DEA"].iloc[analysis_index]
                    dif_prev = indicator_values["DIF"].iloc[analysis_index-1]
                    dea_prev = indicator_values["DEA"].iloc[analysis_index-1]
                    
                    if dif_prev < dea_prev and dif_now > dea_now:
                        patterns.append("MACD金叉")
                        
                # MACD底背离
                if (analysis_index > 10 and 
                    "DIF" in indicator_values.columns and 
                    "MACD" in indicator_values.columns):
                    
                    # 这里的MACD底背离判断逻辑比较复杂，简化处理
                    # 实际应用中应该寻找局部低点并比较
                    patterns.append("MACD底背离候选")
            
            # KDJ形态
            elif indicator_name == "KDJ":
                # KDJ金叉
                if (analysis_index > 0 and 
                    "K" in indicator_values.columns and 
                    "D" in indicator_values.columns):
                    
                    k_now = indicator_values["K"].iloc[analysis_index]
                    d_now = indicator_values["D"].iloc[analysis_index]
                    k_prev = indicator_values["K"].iloc[analysis_index-1]
                    d_prev = indicator_values["D"].iloc[analysis_index-1]
                    
                    if k_prev < d_prev and k_now > d_now:
                        patterns.append("KDJ金叉")
                        
                # KDJ超卖反弹
                if "K" in indicator_values.columns:
                    k_now = indicator_values["K"].iloc[analysis_index]
                    
                    if analysis_index > 1:
                        k_prev = indicator_values["K"].iloc[analysis_index-1]
                        k_prev2 = indicator_values["K"].iloc[analysis_index-2]
                        
                        if k_prev < 20 and k_now > k_prev and k_prev < k_prev2:
                            patterns.append("KDJ超卖反弹")
            
            # RSI形态
            elif indicator_name == "RSI":
                # RSI超卖反弹
                if "RSI6" in indicator_values.columns:
                    rsi_now = indicator_values["RSI6"].iloc[analysis_index]
                    
                    if analysis_index > 1:
                        rsi_prev = indicator_values["RSI6"].iloc[analysis_index-1]
                        
                        if rsi_prev < 20 and rsi_now > rsi_prev:
                            patterns.append("RSI超卖反弹")
                            
                # RSI背离
                # 实际应用中需要更复杂的逻辑
            
            # BOLL形态
            elif indicator_name == "BOLL":
                # 突破上轨
                if (analysis_index > 0 and 
                    "upper" in indicator_values.columns):
                    
                    upper_now = indicator_values["upper"].iloc[analysis_index]
                    upper_prev = indicator_values["upper"].iloc[analysis_index-1]
                    
                    # 需要价格数据，这里简化处理
                    patterns.append("BOLL通道分析")
            
            # 量价形态
            elif indicator_name == "VOL":
                # 需要价格和成交量数据，这里简化处理
                patterns.append("量价关系分析")
            
        except Exception as e:
            logger.error(f"识别 {indicator_name} 形态时出错: {e}")
            
        return patterns
        
    def analyze_multiple_stocks(self, stock_codes: List[str], analysis_date: str = None,
                              periods: List[KlinePeriod] = None,
                              indicators: List[str] = None) -> List[Dict[str, Any]]:
        """
        分析多个股票
        
        Args:
            stock_codes: 股票代码列表
            analysis_date: 分析日期
            periods: 要分析的周期列表
            indicators: 要分析的指标列表
            
        Returns:
            List[Dict]: 分析结果列表
        """
        results = []
        
        for stock_code in stock_codes:
            try:
                result = self.analyze_stock(stock_code, analysis_date, periods, indicators)
                if result and "dimensions" in result and result["dimensions"]:
                    results.append(result)
                    logger.info(f"成功分析股票: {stock_code}")
                else:
                    logger.warning(f"分析股票 {stock_code} 未获得有效结果")
            except Exception as e:
                logger.error(f"分析股票 {stock_code} 时出错: {e}")
        
        return results 

    def extract_common_features(self, analysis_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        从多个股票的分析结果中提取共性特征
        
        Args:
            analysis_results: 分析结果列表，如果为None则使用内部存储的结果
            
        Returns:
            Dict: 共性特征统计
        """
        if analysis_results is None:
            analysis_results = self.analysis_results
            
        if not analysis_results:
            logger.warning("没有分析结果可供提取共性特征")
            return {}
            
        # 统计各个周期的信号和形态
        period_stats = {}
        all_signals = []
        all_patterns = []
        
        # 遍历所有股票的分析结果
        for result in analysis_results:
            if "dimensions" not in result:
                continue
                
            # 遍历各个周期
            for period, period_data in result["dimensions"].items():
                if period not in period_stats:
                    period_stats[period] = {
                        "signals": {},
                        "patterns": {}
                    }
                    
                # 统计信号
                if "signals" in period_data:
                    for signal in period_data["signals"]:
                        if signal not in period_stats[period]["signals"]:
                            period_stats[period]["signals"][signal] = 0
                        period_stats[period]["signals"][signal] += 1
                        
                        if signal not in all_signals:
                            all_signals.append(signal)
                
                # 统计形态
                if "patterns" in period_data:
                    for pattern in period_data["patterns"]:
                        if pattern not in period_stats[period]["patterns"]:
                            period_stats[period]["patterns"][pattern] = 0
                        period_stats[period]["patterns"][pattern] += 1
                        
                        if pattern not in all_patterns:
                            all_patterns.append(pattern)
        
        # 计算共性特征百分比
        common_features = {
            "total_stocks": len(analysis_results),
            "periods": {},
            "top_signals": [],
            "top_patterns": []
        }
        
        # 处理各个周期的统计
        for period, stats in period_stats.items():
            common_features["periods"][period] = {
                "signals": [],
                "patterns": []
            }
            
            # 处理信号
            for signal, count in stats["signals"].items():
                percentage = count / len(analysis_results) * 100
                common_features["periods"][period]["signals"].append({
                    "signal": signal,
                    "count": count,
                    "percentage": percentage
                })
            
            # 排序信号
            common_features["periods"][period]["signals"].sort(
                key=lambda x: x["percentage"], reverse=True)
            
            # 处理形态
            for pattern, count in stats["patterns"].items():
                percentage = count / len(analysis_results) * 100
                common_features["periods"][period]["patterns"].append({
                    "pattern": pattern,
                    "count": count,
                    "percentage": percentage
                })
            
            # 排序形态
            common_features["periods"][period]["patterns"].sort(
                key=lambda x: x["percentage"], reverse=True)
        
        # 汇总所有周期的顶部信号和形态
        all_period_signals = []
        all_period_patterns = []
        
        for period, stats in common_features["periods"].items():
            for item in stats["signals"]:
                all_period_signals.append({
                    "period": period,
                    **item
                })
                
            for item in stats["patterns"]:
                all_period_patterns.append({
                    "period": period,
                    **item
                })
        
        # 排序并获取顶部信号和形态
        all_period_signals.sort(key=lambda x: x["percentage"], reverse=True)
        all_period_patterns.sort(key=lambda x: x["percentage"], reverse=True)
        
        common_features["top_signals"] = all_period_signals[:10]
        common_features["top_patterns"] = all_period_patterns[:10]
        
        return common_features
        
    def save_results(self, output_file: str, format_type: str = "json") -> None:
        """
        保存分析结果
        
        Args:
            output_file: 输出文件路径
            format_type: 输出格式类型，支持json和markdown
        """
        if not self.analysis_results:
            logger.warning("没有分析结果可供保存")
            return
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if format_type.lower() == "json":
            self._save_json_results(output_file)
        elif format_type.lower() == "markdown":
            self._save_markdown_results(output_file)
        else:
            logger.error(f"不支持的输出格式: {format_type}")
            
    def _save_json_results(self, output_file: str) -> None:
        """保存为JSON格式"""
        try:
            # 提取共性特征
            common_features = self.extract_common_features()
            
            # 构建完整结果
            full_results = {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stock_count": len(self.analysis_results),
                "stocks": self.analysis_results,
                "common_features": common_features
            }
            
            # 使用JSON编码器处理numpy类型
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif obj is None or obj == np.nan:
                        return None
                    return super(NumpyEncoder, self).default(obj)
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                
            logger.info(f"分析结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存JSON结果时出错: {e}")
            
    def _save_markdown_results(self, output_file: str) -> None:
        """保存为Markdown格式"""
        try:
            # 提取共性特征
            common_features = self.extract_common_features()
            
            # 构建Markdown内容
            markdown = "# 多维度分析结果\n\n"
            markdown += f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            markdown += f"分析股票数量: {len(self.analysis_results)}\n\n"
            
            # 添加共性特征
            markdown += "## 共性特征分析\n\n"
            
            # 添加顶部信号
            markdown += "### 顶部信号\n\n"
            markdown += "| 周期 | 信号 | 出现比例 | 出现次数 |\n"
            markdown += "| ---- | ---- | -------- | -------- |\n"
            
            for signal in common_features["top_signals"][:10]:
                markdown += f"| {signal['period']} | {signal['signal']} | "
                markdown += f"{signal['percentage']:.2f}% | {signal['count']} |\n"
            
            markdown += "\n"
            
            # 添加顶部形态
            markdown += "### 顶部形态\n\n"
            markdown += "| 周期 | 形态 | 出现比例 | 出现次数 |\n"
            markdown += "| ---- | ---- | -------- | -------- |\n"
            
            for pattern in common_features["top_patterns"][:10]:
                markdown += f"| {pattern['period']} | {pattern['pattern']} | "
                markdown += f"{pattern['percentage']:.2f}% | {pattern['count']} |\n"
            
            markdown += "\n"
            
            # 各周期详细分析
            markdown += "## 各周期分析详情\n\n"
            
            for period, data in common_features["periods"].items():
                markdown += f"### {period} 周期分析\n\n"
                
                # 信号统计
                if data["signals"]:
                    markdown += "#### 信号统计\n\n"
                    markdown += "| 信号 | 出现比例 | 出现次数 |\n"
                    markdown += "| ---- | -------- | -------- |\n"
                    
                    for signal in data["signals"]:
                        markdown += f"| {signal['signal']} | "
                        markdown += f"{signal['percentage']:.2f}% | {signal['count']} |\n"
                    
                    markdown += "\n"
                
                # 形态统计
                if data["patterns"]:
                    markdown += "#### 形态统计\n\n"
                    markdown += "| 形态 | 出现比例 | 出现次数 |\n"
                    markdown += "| ---- | -------- | -------- |\n"
                    
                    for pattern in data["patterns"]:
                        markdown += f"| {pattern['pattern']} | "
                        markdown += f"{pattern['percentage']:.2f}% | {pattern['count']} |\n"
                    
                    markdown += "\n"
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
                
            logger.info(f"分析结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存Markdown结果时出错: {e}")
            
# 主函数
def main():
    """命令行入口函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多维度股票分析工具")
    parser.add_argument('-i', '--input', required=True, help='输入文件，包含股票代码列表')
    parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    parser.add_argument('-d', '--date', help='分析日期，格式YYYYMMDD，默认为最近交易日')
    parser.add_argument('-f', '--format', choices=['json', 'markdown'], default='json',
                      help='输出格式，支持json和markdown，默认为json')
    
    args = parser.parse_args()
    
    # 读取股票代码
    stock_codes = []
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip()
                if code and not code.startswith('#'):
                    stock_codes.append(code)
    except Exception as e:
        logger.error(f"读取输入文件时出错: {e}")
        return
    
    if not stock_codes:
        logger.error("没有找到有效的股票代码")
        return
    
    # 创建分析器并执行分析
    analyzer = MultiDimensionAnalyzer()
    results = analyzer.analyze_multiple_stocks(stock_codes, args.date)
    
    # 保存结果
    analyzer.save_results(args.output, args.format)
    
if __name__ == "__main__":
    main() 