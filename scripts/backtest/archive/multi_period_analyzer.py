#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import datetime
import json

from formula import formula
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from utils.path_utils import get_stock_result_file, get_backtest_result_dir
from db.db_manager import DBManager
from indicators.factory import IndicatorFactory
from scripts.backtest.indicator_analysis import IndicatorAnalyzer

# 获取日志记录器
logger = get_logger(__name__)

class MultiPeriodAnalyzer:
    """
    多周期分析器
    分析多个周期的技术指标情况，包括15分钟、30分钟、60分钟、日线、周线、月线
    """
    
    def __init__(self):
        """初始化分析器"""
        self.db_manager = DBManager.get_instance()
        self.result_dir = get_backtest_result_dir()
        self.indicator_analyzer = IndicatorAnalyzer()
        
    def analyze_stock(self, code: str, buy_date: str, pattern_type: str = "", 
                     days_before: int = 10, days_after: int = 5) -> Dict[str, Any]:
        """
        分析单个股票多周期买点附近的技术指标
        
        Args:
            code: 股票代码
            buy_date: 买点日期，格式为YYYYMMDD
            pattern_type: 买点类型描述，如"回踩反弹"、"横盘突破"等
            days_before: 分析买点前几天的数据
            days_after: 分析买点后几天的数据
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 日期转换
            buy_date_obj = datetime.datetime.strptime(buy_date, "%Y%m%d")
            end_date = (buy_date_obj + datetime.timedelta(days=days_after)).strftime("%Y%m%d")
            start_date = (buy_date_obj - datetime.timedelta(days=days_before*2)).strftime("%Y%m%d")
            
            # 获取基础分析结果（日线）
            daily_result = self.indicator_analyzer.analyze_stock(
                code, buy_date, pattern_type, days_before, days_after
            )
            
            if not daily_result:
                logger.warning(f"未能获取股票 {code} 日线分析结果")
                return {}
                
            # 创建多周期结果
            result = {
                'code': code,
                'name': daily_result.get('name', ''),
                'industry': daily_result.get('industry', ''),
                'buy_date': buy_date,
                'pattern_type': pattern_type,
                'buy_price': daily_result.get('buy_price', 0),
                'periods': {
                    'daily': daily_result,
                    'min15': self._analyze_period(code, buy_date, KlinePeriod.MIN_15, start_date, end_date),
                    'min30': self._analyze_period(code, buy_date, KlinePeriod.MIN_30, start_date, end_date),
                    'min60': self._analyze_period(code, buy_date, KlinePeriod.MIN_60, start_date, end_date),
                    'weekly': self._analyze_period(code, buy_date, KlinePeriod.WEEKLY, start_date, end_date),
                    'monthly': self._analyze_period(code, buy_date, KlinePeriod.MONTHLY, start_date, end_date)
                }
            }
            
            # 提取跨周期共性
            result['cross_period_patterns'] = self._extract_cross_period_patterns(result['periods'])
            
            return result
                
        except Exception as e:
            logger.error(f"多周期分析过程中出错: {code} - {e}")
            return {}
            
    def _analyze_period(self, code: str, buy_date: str, period: KlinePeriod, 
                       start_date: str, end_date: str) -> Dict[str, Any]:
        """
        分析指定周期的技术指标
        
        Args:
            code: 股票代码
            buy_date: 买点日期
            period: 周期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 获取股票数据
            stock_data = formula.StockData(code, period, start=start_date, end=end_date)
            
            if not hasattr(stock_data, 'close') or len(stock_data.close) == 0:
                logger.warning(f"未获取到股票 {code} 周期 {period.name} 的数据")
                return {}
                
            # 准备数据DataFrame
            data = pd.DataFrame({
                'date': stock_data.history['date'],
                'open': stock_data.open,
                'high': stock_data.high,
                'low': stock_data.low,
                'close': stock_data.close,
                'volume': stock_data.volume if hasattr(stock_data, 'volume') else None
            })
            
            # 买点日期（转换为当前周期）
            buy_date_obj = datetime.datetime.strptime(buy_date, "%Y%m%d").date()
            
            # 在当前周期中找到对应的买点位置
            buy_index = None
            for i, date in enumerate(data['date']):
                if isinstance(date, str):
                    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                if date >= buy_date_obj:
                    buy_index = i
                    break
                    
            if buy_index is None:
                logger.warning(f"未找到股票 {code} 周期 {period.name} 买点日期 {buy_date} 的数据")
                return {}
                
            # 计算各种指标
            result = {
                'indicators': {}
            }
            
            # 计算MACD指标
            macd_indicator = IndicatorFactory.create_indicator("MACD")
            macd_result = macd_indicator.compute(data)
            
            diff = macd_result['DIF'].values
            dea = macd_result['DEA'].values
            macd = macd_result['MACD'].values
            
            result['indicators']['macd'] = {
                'diff': diff[buy_index],
                'dea': dea[buy_index],
                'macd': macd[buy_index],
                'diff_prev': diff[buy_index-1] if buy_index > 0 else None,
                'dea_prev': dea[buy_index-1] if buy_index > 0 else None,
                'macd_prev': macd[buy_index-1] if buy_index > 0 else None
            }
            
            # 计算KDJ指标
            kdj_indicator = IndicatorFactory.create_indicator("KDJ")
            kdj_result = kdj_indicator.compute(data)
            
            k = kdj_result['K'].values
            d = kdj_result['D'].values
            j = kdj_result['J'].values
            
            result['indicators']['kdj'] = {
                'k': k[buy_index],
                'd': d[buy_index],
                'j': j[buy_index],
                'k_prev': k[buy_index-1] if buy_index > 0 else None,
                'd_prev': d[buy_index-1] if buy_index > 0 else None,
                'j_prev': j[buy_index-1] if buy_index > 0 else None
            }
            
            # 计算RSI指标
            rsi_indicator = IndicatorFactory.create_indicator("RSI", periods=[6, 12, 24])
            rsi_result = rsi_indicator.compute(data)
            
            if 'RSI6' in rsi_result.columns:
                result['indicators']['rsi'] = {
                    'rsi6': rsi_result['RSI6'].values[buy_index],
                    'rsi12': rsi_result['RSI12'].values[buy_index] if 'RSI12' in rsi_result.columns else None,
                    'rsi24': rsi_result['RSI24'].values[buy_index] if 'RSI24' in rsi_result.columns else None
                }
            elif 'rsi' in rsi_result.columns:
                result['indicators']['rsi'] = {
                    'rsi': rsi_result['rsi'].values[buy_index]
                }
                
            # 计算均线
            ma_indicator = IndicatorFactory.create_indicator("MA", periods=[5, 10, 20, 30, 60])
            ma_result = ma_indicator.compute(data)
            
            result['indicators']['ma'] = {}
            for period in [5, 10, 20, 30, 60]:
                col_name = f'MA{period}'
                if col_name in ma_result.columns:
                    result['indicators']['ma'][f'ma{period}'] = ma_result[col_name].values[buy_index]
            
            # 价格信息
            result['price'] = {
                'open': data['open'].values[buy_index],
                'high': data['high'].values[buy_index],
                'low': data['low'].values[buy_index],
                'close': data['close'].values[buy_index]
            }
            
            # 添加交易信号识别
            result['signals'] = self._identify_signals(data, buy_index)
            
            return result
            
        except Exception as e:
            logger.error(f"分析周期 {period.name} 时出错: {code} - {e}")
            return {}
            
    def _identify_signals(self, data: pd.DataFrame, buy_index: int) -> List[str]:
        """
        识别交易信号
        
        Args:
            data: 数据
            buy_index: 买点索引
            
        Returns:
            List[str]: 信号列表
        """
        signals = []
        
        # 在这里添加各种信号识别逻辑
        
        return signals
        
    def _extract_cross_period_patterns(self, periods: Dict[str, Any]) -> List[str]:
        """
        提取跨周期共性
        
        Args:
            periods: 各周期分析结果
            
        Returns:
            List[str]: 跨周期共性列表
        """
        patterns = []
        
        # 在这里添加跨周期共性提取逻辑
        
        return patterns
        
    def analyze_multiple(self, stock_list: List[str], buy_dates: List[str], 
                       pattern_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        分析多个股票的多周期数据
        
        Args:
            stock_list: 股票代码列表
            buy_dates: 买点日期列表
            pattern_types: 买点类型列表
            
        Returns:
            List[Dict]: 分析结果列表
        """
        results = []
        
        if pattern_types is None:
            pattern_types = [""] * len(stock_list)
            
        for i, code in enumerate(stock_list):
            buy_date = buy_dates[i] if i < len(buy_dates) else buy_dates[-1]
            pattern_type = pattern_types[i] if i < len(pattern_types) else ""
            
            result = self.analyze_stock(code, buy_date, pattern_type)
            if result:
                results.append(result)
                
        return results
        
    def analyze_from_csv(self, csv_file: str) -> List[Dict[str, Any]]:
        """
        从CSV文件分析买点
        
        Args:
            csv_file: CSV文件路径
            
        Returns:
            List[Dict]: 分析结果列表
        """
        try:
            df = pd.read_csv(csv_file)
            
            if 'code' not in df.columns or 'date' not in df.columns:
                logger.error(f"CSV文件 {csv_file} 缺少必要的列: code, date")
                return []
                
            stock_list = df['code'].tolist()
            buy_dates = df['date'].astype(str).tolist()
            pattern_types = df['pattern_type'].tolist() if 'pattern_type' in df.columns else None
            
            return self.analyze_multiple(stock_list, buy_dates, pattern_types)
            
        except Exception as e:
            logger.error(f"从CSV文件 {csv_file} 分析买点时出错: {e}")
            return []
            
    def save_to_json(self, results: List[Dict[str, Any]], output_file: str = None) -> str:
        """
        将分析结果保存为JSON文件
        
        Args:
            results: 分析结果列表
            output_file: 输出文件路径
            
        Returns:
            str: 输出文件路径
        """
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.result_dir}/multi_period_analysis_{timestamp}.json"
            
        # 创建目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"多周期分析结果已保存到 {output_file}")
        return output_file
        
    def generate_report(self, results: List[Dict[str, Any]], output_file: str = None) -> str:
        """
        生成多周期分析报告
        
        Args:
            results: 分析结果列表
            output_file: 输出文件路径
            
        Returns:
            str: 输出文件路径
        """
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.result_dir}/multi_period_analysis_{timestamp}.md"
            
        # 创建目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 生成报告内容
        content = "# 多周期分析报告\n\n"
        
        for result in results:
            code = result.get('code', '')
            name = result.get('name', '')
            buy_date = result.get('buy_date', '')
            
            content += f"## {name}({code}) - {buy_date}\n\n"
            
            # 添加各周期分析结果
            periods = result.get('periods', {})
            for period_name, period_data in periods.items():
                if not period_data:
                    continue
                    
                content += f"### {self._get_period_name(period_name)}\n\n"
                
                # 价格信息
                price = period_data.get('price', {})
                if price:
                    content += "#### 价格信息\n\n"
                    content += f"- 开盘价: {price.get('open', 0):.2f}\n"
                    content += f"- 最高价: {price.get('high', 0):.2f}\n"
                    content += f"- 最低价: {price.get('low', 0):.2f}\n"
                    content += f"- 收盘价: {price.get('close', 0):.2f}\n\n"
                
                # 指标信息
                indicators = period_data.get('indicators', {})
                if indicators:
                    content += "#### 技术指标\n\n"
                    
                    # MACD
                    macd = indicators.get('macd', {})
                    if macd:
                        content += "##### MACD\n\n"
                        content += f"- DIF: {macd.get('diff', 0):.4f}\n"
                        content += f"- DEA: {macd.get('dea', 0):.4f}\n"
                        content += f"- MACD: {macd.get('macd', 0):.4f}\n\n"
                    
                    # KDJ
                    kdj = indicators.get('kdj', {})
                    if kdj:
                        content += "##### KDJ\n\n"
                        content += f"- K值: {kdj.get('k', 0):.4f}\n"
                        content += f"- D值: {kdj.get('d', 0):.4f}\n"
                        content += f"- J值: {kdj.get('j', 0):.4f}\n\n"
                    
                    # RSI
                    rsi = indicators.get('rsi', {})
                    if rsi:
                        content += "##### RSI\n\n"
                        if 'rsi6' in rsi:
                            rsi6_value = rsi.get('rsi6', 0)
                            content += f"- RSI6: {rsi6_value if rsi6_value is not None else 0:.4f}\n"
                        if 'rsi12' in rsi:
                            rsi12_value = rsi.get('rsi12', 0)
                            content += f"- RSI12: {rsi12_value if rsi12_value is not None else 0:.4f}\n"
                        if 'rsi24' in rsi:
                            rsi24_value = rsi.get('rsi24', 0)
                            content += f"- RSI24: {rsi24_value if rsi24_value is not None else 0:.4f}\n"
                        if 'rsi' in rsi:
                            rsi_value = rsi.get('rsi', 0)
                            content += f"- RSI: {rsi_value if rsi_value is not None else 0:.4f}\n"
                        content += "\n"
                    
                    # MA
                    ma = indicators.get('ma', {})
                    if ma:
                        content += "##### 均线\n\n"
                        for period in [5, 10, 20, 30, 60]:
                            ma_key = f'ma{period}'
                            if ma_key in ma:
                                content += f"- MA{period}: {ma.get(ma_key, 0):.4f}\n"
                        content += "\n"
                
                # 信号
                signals = period_data.get('signals', [])
                if signals:
                    content += "#### 交易信号\n\n"
                    for signal in signals:
                        content += f"- {signal}\n"
                    content += "\n"
            
            # 跨周期共性
            cross_period_patterns = result.get('cross_period_patterns', [])
            if cross_period_patterns:
                content += "### 跨周期共性\n\n"
                for pattern in cross_period_patterns:
                    content += f"- {pattern}\n"
                content += "\n"
            
            content += "---\n\n"
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"多周期分析报告已保存到 {output_file}")
        return output_file
        
    def _get_period_name(self, period: str) -> str:
        """
        获取周期的中文名称
        
        Args:
            period: 周期代码
            
        Returns:
            str: 周期中文名称
        """
        period_names = {
            'min15': '15分钟',
            'min30': '30分钟',
            'min60': '60分钟',
            'daily': '日线',
            'weekly': '周线',
            'monthly': '月线'
        }
        return period_names.get(period, period)


def analyze_multi_period(input_source, source_type="csv", output_file=None, report_file=None):
    """
    多周期分析入口函数
    
    Args:
        input_source: 输入源（CSV文件路径或股票列表）
        source_type: 输入源类型，"csv"或"list"
        output_file: 输出JSON文件路径
        report_file: 输出报告文件路径
        
    Returns:
        tuple: (JSON文件路径, 报告文件路径)
    """
    analyzer = MultiPeriodAnalyzer()
    
    if source_type == "csv":
        results = analyzer.analyze_from_csv(input_source)
    elif source_type == "list":
        stock_list, buy_dates, pattern_types = input_source
        results = analyzer.analyze_multiple(stock_list, buy_dates, pattern_types)
    else:
        logger.error(f"不支持的输入源类型: {source_type}")
        return None, None
        
    if not results:
        logger.error("没有分析结果")
        return None, None
        
    json_file = analyzer.save_to_json(results, output_file)
    md_file = analyzer.generate_report(results, report_file)
    
    return json_file, md_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="多周期技术指标分析")
    parser.add_argument("--input", required=True, help="输入CSV文件路径")
    parser.add_argument("--output", help="输出JSON文件路径")
    parser.add_argument("--report", help="输出报告文件路径")
    
    args = parser.parse_args()
    
    json_file, md_file = analyze_multi_period(args.input, "csv", args.output, args.report)
    
    if json_file and md_file:
        logger.info("多周期分析完成")
        logger.info(f"JSON结果: {json_file}")
        logger.info(f"报告结果: {md_file}")
    else:
        logger.error("多周期分析失败") 