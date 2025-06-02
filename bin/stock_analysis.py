#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import argparse
import pandas as pd
import numpy as np
import datetime
import json
from typing import Dict, List, Any, Optional

from formula import formula
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from utils.path_utils import get_stock_result_file, get_backtest_result_dir
from indicators.factory import IndicatorFactory
from scripts.backtest.indicator_analysis import IndicatorAnalyzer
from db.db_manager import DBManager

logger = get_logger(__name__)


class StockAnalysisReport:
    """
    股票分析报告生成器
    用于生成股票分析的Markdown报告
    """
    
    def __init__(self):
        """初始化报告生成器"""
        self.db_manager = DBManager.get_instance()
        self.indicator_analyzer = IndicatorAnalyzer()
        self.result_dir = get_backtest_result_dir()
        
    def analyze_stock(self, code: str, buy_date: str, output_file: str = None) -> str:
        """
        分析股票并生成Markdown报告
        
        Args:
            code: 股票代码
            buy_date: 买点日期，格式为YYYYMMDD
            output_file: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            str: 报告文件路径
        """
        try:
            # 1. 进行股票分析
            analysis_result = self.indicator_analyzer.analyze_stock(
                code, buy_date, days_before=20, days_after=10
            )
            
            if not analysis_result:
                logger.error(f"分析股票 {code} 买点指标时出错")
                return None
            
            # 2. 获取行业板块信息
            industry_analysis = self._analyze_industry(code, buy_date)
            
            # 3. 生成Markdown报告
            report_content = self._generate_markdown_report(
                analysis_result, industry_analysis
            )
            
            # 4. 保存报告到文件
            if output_file is None:
                # 使用默认文件名
                output_file = f"{self.result_dir}/{buy_date}_{code}_分析报告.md"
                
            # 创建目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"分析报告已保存到 {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"生成股票 {code} 分析报告时出错: {e}")
            return None
    
    def _analyze_industry(self, code: str, date: str) -> Dict[str, Any]:
        """
        分析股票所属行业情况
        
        Args:
            code: 股票代码
            date: 日期
            
        Returns:
            Dict: 行业分析结果
        """
        try:
            # 获取股票所属行业
            f = formula.Formula(code)
            industry = f.industry
            
            # 获取同行业股票
            query = f"SELECT code, name FROM stock_info WHERE industry = '{industry}'"
            industry_stocks = self.db_manager.db.client.execute(query)
            
            # 计算行业涨跌幅
            date_obj = datetime.datetime.strptime(date, "%Y%m%d")
            prev_date = (date_obj - datetime.timedelta(days=30)).strftime("%Y%m%d")
            
            industry_change = []
            for stock_code, stock_name in industry_stocks:
                try:
                    stock_f = formula.Formula(stock_code, start=prev_date, end=date)
                    if len(stock_f.dataDay.close) > 0:
                        start_price = stock_f.dataDay.close[0]
                        end_price = stock_f.dataDay.close[-1]
                        change_pct = (end_price / start_price - 1) * 100
                        industry_change.append({
                            'code': stock_code,
                            'name': stock_name,
                            'change_pct': change_pct
                        })
                except Exception as e:
                    logger.debug(f"处理行业股票 {stock_code} 时出错: {e}")
            
            # 计算行业平均涨跌幅
            if industry_change:
                avg_change = sum(item['change_pct'] for item in industry_change) / len(industry_change)
                # 排序
                industry_change.sort(key=lambda x: x['change_pct'], reverse=True)
                # 计算目标股票排名
                target_rank = -1
                for i, item in enumerate(industry_change):
                    if item['code'] == code:
                        target_rank = i + 1
                        break
            else:
                avg_change = 0
                target_rank = -1
            
            # 返回行业分析结果
            return {
                'industry': industry,
                'stock_count': len(industry_stocks),
                'avg_change': avg_change,
                'target_rank': target_rank,
                'total_rank': len(industry_change),
                'top_stocks': industry_change[:5] if len(industry_change) > 5 else industry_change
            }
            
        except Exception as e:
            logger.error(f"分析行业情况时出错: {e}")
            return {
                'industry': '未知',
                'stock_count': 0,
                'avg_change': 0,
                'target_rank': -1,
                'total_rank': 0,
                'top_stocks': []
            }
    
    def _generate_markdown_report(self, analysis_result: Dict[str, Any], 
                               industry_analysis: Dict[str, Any]) -> str:
        """
        生成Markdown格式的分析报告
        
        Args:
            analysis_result: 股票分析结果
            industry_analysis: 行业分析结果
            
        Returns:
            str: Markdown格式的报告内容
        """
        code = analysis_result['code']
        name = analysis_result['name']
        industry = analysis_result['industry']
        buy_date = analysis_result['buy_date']
        buy_price = analysis_result['buy_price']
        
        # 构建报告内容
        report = f"""# {name}({code}) 买点分析报告

## 基本信息

- **股票名称**: {name}
- **股票代码**: {code}
- **所属行业**: {industry}
- **买点日期**: {buy_date}
- **买点价格**: {buy_price:.2f}

## 技术指标分析

### 均线系统
"""
        # 添加均线分析
        ma_data = analysis_result['indicators'].get('ma', {})
        if ma_data:
            report += f"""
- **5日均线**: {ma_data.get('ma5', 0):.2f}
- **10日均线**: {ma_data.get('ma10', 0):.2f}
- **20日均线**: {ma_data.get('ma20', 0):.2f}
- **30日均线**: {ma_data.get('ma30', 0):.2f}
- **60日均线**: {ma_data.get('ma60', 0):.2f}
- **收盘价/5日均线**: {ma_data.get('close_ma5_ratio', 0):.4f}
- **收盘价/10日均线**: {ma_data.get('close_ma10_ratio', 0):.4f}
- **收盘价/20日均线**: {ma_data.get('close_ma20_ratio', 0):.4f}
"""

        # 添加KDJ分析
        kdj_data = analysis_result['indicators'].get('kdj', {})
        report += f"""
### KDJ指标

- **K值**: {kdj_data.get('k', 0):.2f}
- **D值**: {kdj_data.get('d', 0):.2f}
- **J值**: {kdj_data.get('j', 0):.2f}
- **K值变化**: {kdj_data.get('k_diff', 0):.2f}
- **D值变化**: {kdj_data.get('d_diff', 0):.2f}
"""

        # 添加MACD分析
        macd_data = analysis_result['indicators'].get('macd', {})
        report += f"""
### MACD指标

- **DIF**: {macd_data.get('diff', 0):.4f}
- **DEA**: {macd_data.get('dea', 0):.4f}
- **MACD**: {macd_data.get('macd', 0):.4f}
- **DIF变化**: {macd_data.get('diff_diff', 0):.4f}
- **DEA变化**: {macd_data.get('dea_diff', 0):.4f}
"""

        # 添加RSI分析
        rsi_data = analysis_result['indicators'].get('rsi', {})
        report += f"""
### RSI指标

- **RSI6**: {rsi_data.get('rsi6', 0):.2f}
- **RSI12**: {rsi_data.get('rsi12', 0):.2f}
- **RSI24**: {rsi_data.get('rsi24', 0):.2f}
"""

        # 添加BOLL分析
        boll_data = analysis_result['indicators'].get('boll', {})
        report += f"""
### BOLL指标

- **上轨**: {boll_data.get('upper', 0):.2f}
- **中轨**: {boll_data.get('middle', 0):.2f}
- **下轨**: {boll_data.get('lower', 0):.2f}
- **带宽**: {boll_data.get('width', 0):.4f}
- **位置**: {boll_data.get('close_position', 0):.4f}
"""

        # 添加成交量分析
        volume_data = analysis_result['indicators'].get('volume', {})
        report += f"""
### 成交量分析

- **成交量**: {volume_data.get('volume', 0):.0f}
- **5日均量**: {volume_data.get('volume_ma5', 0):.0f}
- **量比**: {volume_data.get('volume_ratio', 0):.2f}
- **前日量比**: {volume_data.get('volume_prev_ratio', 0):.2f}
"""

        # 添加技术形态
        patterns = analysis_result.get('patterns', [])
        report += f"""
## 技术形态分析

识别到的技术形态:
"""
        if patterns:
            for pattern in patterns:
                report += f"- {pattern}\n"
        else:
            report += "- 未识别到明显技术形态\n"

        # 添加行业分析
        report += f"""
## 行业板块分析

- **所属行业**: {industry_analysis['industry']}
- **行业股票数量**: {industry_analysis['stock_count']}
- **行业平均涨幅**: {industry_analysis['avg_change']:.2f}%
- **行业排名**: {industry_analysis['target_rank']}/{industry_analysis['total_rank']}
"""

        # 添加行业龙头股
        report += """
### 行业龙头股表现
| 排名 | 股票代码 | 股票名称 | 涨幅(%) |
|-----|---------|---------|--------|
"""
        for i, stock in enumerate(industry_analysis['top_stocks']):
            report += f"| {i+1} | {stock['code']} | {stock['name']} | {stock['change_pct']:.2f} |\n"

        # 添加策略建议
        report += f"""
## 策略建议

根据对{name}({code})的技术指标分析和行业分析，提出以下交易策略建议：

"""
        # 基于技术形态生成策略建议
        if 'KDJ金叉' in patterns or 'MACD金叉' in patterns:
            report += "- **建议操作**: 考虑买入\n"
            report += "- **理由**: 出现KDJ或MACD金叉，技术面向好\n"
        elif '均线多头排列' in patterns:
            report += "- **建议操作**: 考虑买入\n"
            report += "- **理由**: 均线呈多头排列，短期趋势向上\n"
        elif '回踩5日均线反弹' in patterns:
            report += "- **建议操作**: 考虑买入\n"
            report += "- **理由**: 股价回踩均线后反弹，是较好的买点\n"
        elif 'BOLL中轨上穿' in patterns:
            report += "- **建议操作**: 考虑买入\n"
            report += "- **理由**: 股价突破BOLL中轨，有望继续上行\n"
        else:
            report += "- **建议操作**: 观望\n"
            report += "- **理由**: 未出现明显的买入信号\n"
        
        # 基于行业分析添加建议
        if industry_analysis['target_rank'] <= 3 and industry_analysis['avg_change'] > 0:
            report += "- **行业分析**: 该股为行业龙头，行业整体向好，有望继续领涨\n"
        elif industry_analysis['target_rank'] <= industry_analysis['total_rank'] * 0.3 and industry_analysis['avg_change'] > 0:
            report += "- **行业分析**: 该股在行业中表现较好，行业整体向好\n"
        elif industry_analysis['avg_change'] > 0:
            report += "- **行业分析**: 行业整体向好，但该股表现一般\n"
        else:
            report += "- **行业分析**: 行业整体表现不佳，需谨慎操作\n"

        # 添加后记
        report += f"""
## 后记

本报告由系统自动生成，仅供参考，不构成投资建议。投资决策需结合市场环境、基本面分析、风险偏好等多方面因素。

报告生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        return report


def analyze_stock(args):
    """
    分析股票并生成报告
    
    Args:
        args: 命令行参数
    """
    try:
        report_generator = StockAnalysisReport()
        output_file = report_generator.analyze_stock(args.code, args.date, args.output)
        
        if output_file:
            logger.info(f"分析完成，报告已保存到: {output_file}")
            print(f"分析完成，报告已保存到: {output_file}")
        else:
            logger.error("分析失败")
            print("分析失败，请检查日志")
    
    except Exception as e:
        logger.error(f"分析股票时出错: {e}")
        print(f"分析股票时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='股票买点分析工具')
    parser.add_argument('--code', type=str, required=True, help='股票代码')
    parser.add_argument('--date', type=str, required=True, help='买点日期，格式为YYYYMMDD')
    parser.add_argument('--output', type=str, help='输出文件路径，默认为data/result/日期_股票代码_分析报告.md')
    
    args = parser.parse_args()
    
    analyze_stock(args) 