#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import argparse
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir, get_multi_period_dir, get_strategies_dir
from scripts.backtest.indicator_analysis import IndicatorAnalyzer
from scripts.backtest.multi_period_analyzer import MultiPeriodAnalyzer
from scripts.backtest.report_to_strategy import convert_report_to_strategy

logger = get_logger(__name__)

class ComprehensiveBacktest:
    """
    综合回测类
    支持多周期、多股票、多买点分析，寻找共性指标特征
    """
    
    def __init__(self):
        """初始化综合回测"""
        self.indicator_analyzer = IndicatorAnalyzer()
        self.multi_period_analyzer = MultiPeriodAnalyzer()
        self.result_dir = get_backtest_result_dir()
        self.multi_period_dir = get_multi_period_dir()
        self.strategies_dir = get_strategies_dir()
        
        # 确保目录存在
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.multi_period_dir, exist_ok=True)
        os.makedirs(self.strategies_dir, exist_ok=True)
        
        # 存储分析结果
        self.analysis_results = []
        self.common_patterns = {}
        
    def analyze_from_csv(self, csv_file: str, days_before: int = 20, days_after: int = 10) -> None:
        """
        从CSV文件中批量分析股票买点
        
        Args:
            csv_file: CSV文件路径，包含code,date,pattern_type列
            days_before: 分析买点前几天的数据
            days_after: 分析买点后几天的数据
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            if 'code' not in df.columns or 'date' not in df.columns:
                logger.error(f"CSV文件 {csv_file} 缺少必要的列: code, date")
                return
            
            # 添加pattern_type列，如果不存在
            if 'pattern_type' not in df.columns:
                df['pattern_type'] = ""
            
            # 获取股票列表和日期列表
            stock_list = df['code'].astype(str).tolist()
            date_list = df['date'].astype(str).tolist()
            pattern_list = df['pattern_type'].tolist()
            
            # 分析每个股票
            logger.info(f"开始分析 {len(stock_list)} 只股票的买点")
            
            for i, (code, date, pattern) in enumerate(zip(stock_list, date_list, pattern_list)):
                logger.info(f"[{i+1}/{len(stock_list)}] 分析股票 {code} 在 {date} 的买点 ({pattern})")
                self._analyze_stock(code, date, pattern, days_before, days_after)
            
            # 提取共性特征
            self._extract_common_patterns()
            
        except Exception as e:
            logger.error(f"从CSV文件分析买点时出错: {e}")
    
    def _analyze_stock(self, code: str, date: str, pattern_type: str = "", 
                      days_before: int = 20, days_after: int = 10) -> None:
        """
        分析单个股票的买点
        
        Args:
            code: 股票代码
            date: 买点日期，格式为YYYYMMDD
            pattern_type: 买点类型
            days_before: 分析买点前几天的数据
            days_after: 分析买点后几天的数据
        """
        try:
            # 多周期分析
            result = self.multi_period_analyzer.analyze_stock(
                code, date, pattern_type, days_before, days_after
            )
            
            if result:
                self.analysis_results.append(result)
                logger.info(f"股票 {code} 分析完成")
            else:
                logger.warning(f"股票 {code} 分析失败")
                
        except Exception as e:
            logger.error(f"分析股票 {code} 时出错: {e}")
    
    def _extract_common_patterns(self) -> None:
        """
        提取所有分析结果中的共性特征
        """
        if not self.analysis_results:
            logger.warning("没有分析结果，无法提取共性特征")
            return
            
        logger.info("开始提取共性特征...")
        
        # 用于统计各种特征的出现频率
        pattern_stats = defaultdict(int)
        indicator_stats = defaultdict(lambda: defaultdict(int))
        period_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # 记录分析的总股票数
        total_stocks = len(self.analysis_results)
        
        # 遍历所有分析结果
        for result in self.analysis_results:
            code = result.get('code', '')
            
            # 统计跨周期共性
            for pattern in result.get('cross_period_patterns', []):
                pattern_stats[pattern] += 1
            
            # 统计各周期的技术指标特征
            periods = result.get('periods', {})
            for period_name, period_data in periods.items():
                if not period_data:
                    continue
                
                # 处理各种指标
                indicators = period_data.get('indicators', {})
                
                # MACD指标特征
                macd = indicators.get('macd', {})
                if macd:
                    # MACD金叉
                    diff = macd.get('diff', 0)
                    diff_prev = macd.get('diff_prev', 0)
                    dea = macd.get('dea', 0)
                    dea_prev = macd.get('dea_prev', 0)
                    
                    if diff_prev is not None and dea_prev is not None and diff is not None and dea is not None:
                        if diff_prev < dea_prev and diff > dea:
                            indicator_stats[period_name]['MACD金叉'] += 1
                            period_stats[period_name]['MACD']['金叉'].append(code)
                    
                    # MACD柱形图由负转正
                    macd_value = macd.get('macd', 0)
                    macd_prev = macd.get('macd_prev', 0)
                    
                    if macd_prev is not None and macd_value is not None:
                        if macd_prev < 0 and macd_value > 0:
                            indicator_stats[period_name]['MACD由负转正'] += 1
                            period_stats[period_name]['MACD']['由负转正'].append(code)
                
                # KDJ指标特征
                kdj = indicators.get('kdj', {})
                if kdj:
                    # KDJ金叉
                    k = kdj.get('k', 0)
                    k_prev = kdj.get('k_prev', 0)
                    d = kdj.get('d', 0)
                    d_prev = kdj.get('d_prev', 0)
                    
                    if k_prev is not None and d_prev is not None and k is not None and d is not None:
                        if k_prev < d_prev and k > d:
                            indicator_stats[period_name]['KDJ金叉'] += 1
                            period_stats[period_name]['KDJ']['金叉'].append(code)
                    
                    # J值超买超卖
                    j = kdj.get('j', 0)
                    if j is not None:
                        if j < 20:
                            indicator_stats[period_name]['KDJ超卖'] += 1
                            period_stats[period_name]['KDJ']['超卖'].append(code)
                        elif j > 80:
                            indicator_stats[period_name]['KDJ超买'] += 1
                            period_stats[period_name]['KDJ']['超买'].append(code)
                
                # RSI指标特征
                rsi = indicators.get('rsi', {})
                if rsi:
                    # RSI超买超卖
                    rsi6 = rsi.get('rsi6', None)
                    if rsi6 is not None:
                        if rsi6 < 30:
                            indicator_stats[period_name]['RSI超卖'] += 1
                            period_stats[period_name]['RSI']['超卖'].append(code)
                        elif rsi6 > 70:
                            indicator_stats[period_name]['RSI超买'] += 1
                            period_stats[period_name]['RSI']['超买'].append(code)
                
                # 均线系统特征
                ma = indicators.get('ma', {})
                if ma:
                    # 均线多头排列
                    ma5 = ma.get('ma5', None)
                    ma10 = ma.get('ma10', None)
                    ma20 = ma.get('ma20', None)
                    
                    if ma5 is not None and ma10 is not None and ma20 is not None:
                        if ma5 > ma10 > ma20:
                            indicator_stats[period_name]['均线多头排列'] += 1
                            period_stats[period_name]['MA']['多头排列'].append(code)
                    
                    # 价格与均线关系
                    close = period_data.get('price', {}).get('close', 0)
                    
                    if close and ma20:
                        ratio = close / ma20
                        if 0.98 <= ratio <= 1.02:
                            indicator_stats[period_name]['MA20支撑/压力'] += 1
                            period_stats[period_name]['MA']['MA20支撑'].append(code)
        
        # 计算共性特征（出现频率超过总数的50%）
        common_patterns = {}
        
        # 计算跨周期共性
        cross_period_common = {}
        for pattern, count in pattern_stats.items():
            if count / total_stocks >= 0.5:
                cross_period_common[pattern] = count / total_stocks
        
        # 计算各周期的共性指标
        period_common = {}
        for period, patterns in indicator_stats.items():
            period_common[period] = {}
            for pattern, count in patterns.items():
                if count / total_stocks >= 0.5:
                    period_common[period][pattern] = count / total_stocks
        
        # 保存结果
        self.common_patterns = {
            'cross_period': cross_period_common,
            'period': period_common,
            'details': period_stats,
            'total_stocks': total_stocks
        }
        
        logger.info("共性特征提取完成")
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        生成分析报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            str: 报告文件路径
        """
        if not self.analysis_results:
            logger.warning("没有分析结果，无法生成报告")
            return ""
            
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.result_dir}/综合回测分析_{timestamp}.md"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 生成报告内容
        content = "# 综合回测分析报告\n\n"
        
        # 1. 添加分析概况
        content += "## 分析概况\n\n"
        content += f"- 分析股票数量: {len(self.analysis_results)}\n"
        content += f"- 分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 2. 添加共性特征
        content += "## 共性特征\n\n"
        
        # 2.1 跨周期共性
        cross_period_common = self.common_patterns.get('cross_period', {})
        if cross_period_common:
            content += "### 跨周期共性\n\n"
            content += "| 特征 | 出现比例 |\n"
            content += "| ---- | -------- |\n"
            
            for pattern, ratio in sorted(cross_period_common.items(), key=lambda x: x[1], reverse=True):
                content += f"| {pattern} | {ratio*100:.2f}% |\n"
            
            content += "\n"
        
        # 2.2 各周期共性
        period_common = self.common_patterns.get('period', {})
        if period_common:
            content += "### 各周期共性\n\n"
            
            for period, patterns in period_common.items():
                if patterns:
                    period_name = self.multi_period_analyzer._get_period_name(period)
                    content += f"#### {period_name}\n\n"
                    content += "| 特征 | 出现比例 |\n"
                    content += "| ---- | -------- |\n"
                    
                    for pattern, ratio in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                        content += f"| {pattern} | {ratio*100:.2f}% |\n"
                    
                    content += "\n"
        
        # 3. 添加详细分析
        content += "## 个股分析\n\n"
        
        for i, result in enumerate(self.analysis_results):
            code = result.get('code', '')
            name = result.get('name', '')
            buy_date = result.get('buy_date', '')
            
            content += f"### {i+1}. {name}({code}) - {buy_date}\n\n"
            
            # 添加买点类型
            pattern_type = result.get('pattern_type', '')
            if pattern_type:
                content += f"买点类型: {pattern_type}\n\n"
            
            # 添加跨周期共性
            cross_patterns = result.get('cross_period_patterns', [])
            if cross_patterns:
                content += "#### 跨周期共性\n\n"
                for pattern in cross_patterns:
                    content += f"- {pattern}\n"
                content += "\n"
            
            # 添加各周期要点
            content += "#### 各周期要点\n\n"
            
            periods = result.get('periods', {})
            for period_name, period_data in periods.items():
                if not period_data or period_name == 'daily':
                    continue
                
                content += f"##### {self.multi_period_analyzer._get_period_name(period_name)}\n\n"
                
                # 关键指标摘要
                indicators = period_data.get('indicators', {})
                
                # MACD
                macd = indicators.get('macd', {})
                if macd:
                    diff = macd.get('diff', None)
                    dea = macd.get('dea', None)
                    macd_value = macd.get('macd', None)
                    
                    if None not in (diff, dea, macd_value):
                        if diff > dea:
                            content += f"- MACD: DIF位于DEA上方 (DIF={diff:.4f}, DEA={dea:.4f})\n"
                        else:
                            content += f"- MACD: DIF位于DEA下方 (DIF={diff:.4f}, DEA={dea:.4f})\n"
                        
                        if macd_value > 0:
                            content += f"- MACD柱形图为正 ({macd_value:.4f})\n"
                        else:
                            content += f"- MACD柱形图为负 ({macd_value:.4f})\n"
                
                # KDJ
                kdj = indicators.get('kdj', {})
                if kdj:
                    k = kdj.get('k', None)
                    d = kdj.get('d', None)
                    j = kdj.get('j', None)
                    
                    if None not in (k, d, j):
                        if k > d:
                            content += f"- KDJ: K线位于D线上方 (K={k:.2f}, D={d:.2f})\n"
                        else:
                            content += f"- KDJ: K线位于D线下方 (K={k:.2f}, D={d:.2f})\n"
                        
                        if j < 20:
                            content += f"- J值处于超卖区域 (J={j:.2f})\n"
                        elif j > 80:
                            content += f"- J值处于超买区域 (J={j:.2f})\n"
                        else:
                            content += f"- J值处于中性区域 (J={j:.2f})\n"
                
                content += "\n"
            
            # 添加日线详细分析
            daily_data = periods.get('daily', {})
            if daily_data:
                content += "#### 日线详细分析\n\n"
                
                # 价格信息
                price = daily_data.get('price', {})
                if price:
                    content += "##### 价格信息\n\n"
                    content += f"- 开盘价: {price.get('open', 0):.2f}\n"
                    content += f"- 最高价: {price.get('high', 0):.2f}\n"
                    content += f"- 最低价: {price.get('low', 0):.2f}\n"
                    content += f"- 收盘价: {price.get('close', 0):.2f}\n\n"
                
                # 技术指标
                indicators = daily_data.get('indicators', {})
                
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
                    content += f"- K值: {kdj.get('k', 0):.2f}\n"
                    content += f"- D值: {kdj.get('d', 0):.2f}\n"
                    content += f"- J值: {kdj.get('j', 0):.2f}\n\n"
                
                # 均线
                ma = indicators.get('ma', {})
                if ma:
                    content += "##### 均线\n\n"
                    for period in [5, 10, 20, 30, 60]:
                        ma_key = f'ma{period}'
                        if ma_key in ma:
                            content += f"- MA{period}: {ma.get(ma_key, 0):.2f}\n"
                    content += "\n"
            
            content += "---\n\n"
        
        # 4. 生成选股策略建议
        content += "## 选股策略建议\n\n"
        
        # 基于共性特征生成策略建议
        if cross_period_common or period_common:
            content += "根据分析结果，建议采用以下技术指标组合作为选股条件：\n\n"
            
            # 选取最显著的特征
            top_features = []
            
            # 跨周期共性
            for pattern, ratio in sorted(cross_period_common.items(), key=lambda x: x[1], reverse=True)[:3]:
                top_features.append(f"跨周期特征: {pattern} (出现比例: {ratio*100:.2f}%)")
            
            # 各周期特征
            for period, patterns in period_common.items():
                period_name = self.multi_period_analyzer._get_period_name(period)
                for pattern, ratio in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:2]:
                    top_features.append(f"{period_name}特征: {pattern} (出现比例: {ratio*100:.2f}%)")
            
            # 输出特征列表
            for i, feature in enumerate(top_features):
                content += f"{i+1}. {feature}\n"
            
            content += "\n建议将上述特征组合使用，特别关注跨周期共性特征，可以提高选股的准确性。\n\n"
        else:
            content += "分析结果中未发现显著的共性特征，建议增加样本数量或调整分析参数。\n\n"
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"综合分析报告已保存到 {output_file}")
        return output_file
    
    def generate_strategy(self, report_file: str, output_file: Optional[str] = None) -> str:
        """
        根据分析报告生成选股策略
        
        Args:
            report_file: 报告文件路径
            output_file: 输出文件路径
            
        Returns:
            str: 策略文件路径
        """
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.strategies_dir}/综合策略_{timestamp}.json"
            
        # 从报告中提取策略
        strategy_file = convert_report_to_strategy(report_file, os.path.dirname(output_file))
        
        if strategy_file:
            logger.info(f"选股策略已生成: {strategy_file}")
        else:
            logger.error("生成选股策略失败")
            
        return strategy_file
    
    def run(self, csv_file: str, days_before: int = 20, days_after: int = 10) -> Tuple[str, str]:
        """
        运行综合回测
        
        Args:
            csv_file: CSV文件路径
            days_before: 分析买点前几天的数据
            days_after: 分析买点后几天的数据
            
        Returns:
            Tuple[str, str]: (报告文件路径, 策略文件路径)
        """
        # 1. 分析股票
        self.analyze_from_csv(csv_file, days_before, days_after)
        
        if not self.analysis_results:
            logger.error("没有有效的分析结果")
            return "", ""
        
        # 2. 生成报告
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.result_dir}/综合回测分析_{timestamp}.md"
        report_file = self.generate_report(report_file)
        
        if not report_file:
            logger.error("生成报告失败")
            return "", ""
        
        # 3. 生成策略
        strategy_file = self.generate_strategy(report_file)
        
        return report_file, strategy_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合回测分析工具")
    
    parser.add_argument("--csv", required=True, help="包含股票代码和日期的CSV文件路径")
    parser.add_argument("--days-before", type=int, default=20, help="分析买点前几天的数据")
    parser.add_argument("--days-after", type=int, default=10, help="分析买点后几天的数据")
    parser.add_argument("--output-dir", help="输出目录路径")
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir:
        result_dir = args.output_dir
        os.makedirs(result_dir, exist_ok=True)
        
        # 创建子目录
        os.makedirs(f"{result_dir}/strategies", exist_ok=True)
        os.makedirs(f"{result_dir}/multi_period", exist_ok=True)
    
    # 运行综合回测
    backtest = ComprehensiveBacktest()
    report_file, strategy_file = backtest.run(args.csv, args.days_before, args.days_after)
    
    if report_file and strategy_file:
        logger.info("综合回测分析完成")
        logger.info(f"分析报告: {report_file}")
        logger.info(f"选股策略: {strategy_file}")
        
        print("\n=== 回测分析完成 ===")
        print(f"分析报告: {report_file}")
        print(f"选股策略: {strategy_file}")
    else:
        logger.error("综合回测分析失败")
        print("\n=== 回测分析失败 ===")
        sys.exit(1)


if __name__ == "__main__":
    main() 