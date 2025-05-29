#!/usr/bin/env python3
"""
指标评分应用

对指定股票进行综合技术分析评分，提供买卖建议
"""

import sys
import os
import argparse
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.indicator_registry import indicator_registry
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from utils.date_utils import get_trading_day

logger = get_logger(__name__)


class StockScoreAnalyzer:
    """股票评分分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.db = get_clickhouse_db()
        
        # 默认指标配置
        self.default_config = [
            {'name': 'macd_score', 'weight': 1.5, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            {'name': 'kdj_score', 'weight': 1.2, 'n': 9, 'm1': 3, 'm2': 3},
            {'name': 'rsi_score', 'weight': 1.3, 'period': 14},
            {'name': 'boll_score', 'weight': 1.0, 'period': 20, 'std_dev': 2.0}
        ]
    
    def get_stock_data(self, stock_code: str, days: int = 100) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码
            days: 获取天数
            
        Returns:
            pd.DataFrame: 股票数据
        """
        end_date_str = get_trading_day()
        end_date = datetime.strptime(end_date_str, '%Y%m%d').strftime('%Y-%m-%d')
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days*2)).strftime('%Y-%m-%d')
        
        sql = f"""
        SELECT date, open, high, low, close, volume
        FROM stock_info 
        WHERE code = '{stock_code}'
        AND level = '日线'
        AND date >= '{start_date}'
        AND date <= '{end_date}'
        ORDER BY date
        """
        
        data = self.db.query(sql)
        if data.empty:
            raise ValueError(f"未找到股票 {stock_code} 的数据")
        
        # 重新映射列名
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        return data
    
    def get_stock_info(self, stock_code: str) -> dict:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            dict: 股票信息
        """
        sql = f"""
        SELECT code, name, industry
        FROM stock_info 
        WHERE code = '{stock_code}'
        AND level = '日线'
        LIMIT 1
        """
        
        result = self.db.query(sql)
        if result.empty:
            return {'code': stock_code, 'name': '未知', 'industry': '未知', 'market': '未知'}
        
        # 重新映射列名
        result.columns = ['code', 'name', 'industry']
        row = result.iloc[0]
        return {
            'code': row['code'],
            'name': row['name'], 
            'industry': row['industry'],
            'market': '未知'  # 数据库中没有market字段
        }
    
    def analyze_stock(self, stock_code: str, config: list = None) -> dict:
        """
        分析股票
        
        Args:
            stock_code: 股票代码
            config: 指标配置，如果为None则使用默认配置
            
        Returns:
            dict: 分析结果
        """
        if config is None:
            config = self.default_config
        
        # 获取股票数据和信息
        data = self.get_stock_data(stock_code)
        stock_info = self.get_stock_info(stock_code)
        
        # 创建评分管理器
        score_manager = indicator_registry.create_score_manager(config)
        
        # 计算综合评分
        result = score_manager.calculate_comprehensive_score(data)
        
        # 获取最新评分和信号
        latest_score = result['comprehensive_score'].iloc[-1]
        latest_signals = {}
        for signal_name, signal_series in result['comprehensive_signal'].items():
            latest_signals[signal_name] = signal_series.iloc[-1]
        
        # 生成交易建议
        recommendation = self._generate_recommendation(latest_score, latest_signals)
        
        # 分析趋势
        trend_analysis = self._analyze_trend(result['comprehensive_score'])
        
        return {
            'stock_info': stock_info,
            'latest_data': data.iloc[-1].to_dict(),
            'comprehensive_score': latest_score,
            'signals': latest_signals,
            'recommendation': recommendation,
            'trend_analysis': trend_analysis,
            'indicator_results': result['indicator_results'],
            'patterns': result['patterns'],
            'score_history': result['comprehensive_score'].tail(10).tolist()
        }
    
    def _generate_recommendation(self, score: float, signals: dict) -> dict:
        """
        生成交易建议
        
        Args:
            score: 综合评分
            signals: 信号字典
            
        Returns:
            dict: 交易建议
        """
        if signals.get('strong_buy', False):
            action = "强烈买入"
            confidence = "高"
            reason = f"综合评分{score:.1f}分，多项指标发出强烈买入信号"
        elif signals.get('buy', False):
            action = "买入"
            confidence = "中高" if score >= 70 else "中等"
            reason = f"综合评分{score:.1f}分，技术指标偏向看涨"
        elif signals.get('hold', False):
            action = "持有"
            confidence = "中等"
            reason = f"综合评分{score:.1f}分，技术指标中性，建议观望"
        elif signals.get('sell', False):
            action = "卖出"
            confidence = "中高" if score <= 30 else "中等"
            reason = f"综合评分{score:.1f}分，技术指标偏向看跌"
        elif signals.get('strong_sell', False):
            action = "强烈卖出"
            confidence = "高"
            reason = f"综合评分{score:.1f}分，多项指标发出强烈卖出信号"
        else:
            action = "观望"
            confidence = "低"
            reason = f"综合评分{score:.1f}分，信号不明确，建议观望"
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'score': score
        }
    
    def _analyze_trend(self, score_series: pd.Series) -> dict:
        """
        分析评分趋势
        
        Args:
            score_series: 评分序列
            
        Returns:
            dict: 趋势分析
        """
        recent_scores = score_series.tail(10)
        
        if len(recent_scores) < 3:
            return {'trend': '数据不足', 'strength': '未知'}
        
        # 计算趋势
        trend_slope = (recent_scores.iloc[-1] - recent_scores.iloc[0]) / len(recent_scores)
        
        if trend_slope > 2:
            trend = "强烈上升"
            strength = "强"
        elif trend_slope > 1:
            trend = "上升"
            strength = "中"
        elif trend_slope > 0.5:
            trend = "轻微上升"
            strength = "弱"
        elif trend_slope < -2:
            trend = "强烈下降"
            strength = "强"
        elif trend_slope < -1:
            trend = "下降"
            strength = "中"
        elif trend_slope < -0.5:
            trend = "轻微下降"
            strength = "弱"
        else:
            trend = "横盘整理"
            strength = "弱"
        
        # 计算波动性
        volatility = recent_scores.std()
        if volatility > 10:
            volatility_desc = "高波动"
        elif volatility > 5:
            volatility_desc = "中等波动"
        else:
            volatility_desc = "低波动"
        
        return {
            'trend': trend,
            'strength': strength,
            'volatility': volatility_desc,
            'slope': trend_slope
        }
    
    def print_analysis_report(self, analysis_result: dict):
        """
        打印分析报告
        
        Args:
            analysis_result: 分析结果
        """
        stock_info = analysis_result['stock_info']
        latest_data = analysis_result['latest_data']
        recommendation = analysis_result['recommendation']
        trend_analysis = analysis_result['trend_analysis']
        
        print("="*60)
        print(f"股票技术分析报告")
        print("="*60)
        print(f"股票代码: {stock_info['code']}")
        print(f"股票名称: {stock_info['name']}")
        print(f"所属行业: {stock_info['industry']}")
        print(f"交易市场: {stock_info['market']}")
        print(f"分析日期: {latest_data['date']}")
        print(f"最新价格: {latest_data['close']:.2f}")
        print(f"成交量: {latest_data['volume']:,}")
        
        print("\n" + "-"*40)
        print("综合技术评分")
        print("-"*40)
        print(f"综合评分: {analysis_result['comprehensive_score']:.1f}/100")
        print(f"交易建议: {recommendation['action']}")
        print(f"建议置信度: {recommendation['confidence']}")
        print(f"建议理由: {recommendation['reason']}")
        
        print("\n" + "-"*40)
        print("趋势分析")
        print("-"*40)
        print(f"评分趋势: {trend_analysis['trend']}")
        print(f"趋势强度: {trend_analysis['strength']}")
        print(f"波动特征: {trend_analysis['volatility']}")
        
        print("\n" + "-"*40)
        print("各指标详细评分")
        print("-"*40)
        for indicator_name, result in analysis_result['indicator_results'].items():
            latest_score = result['final_score'].iloc[-1]
            patterns = ', '.join(result['patterns']) if result['patterns'] else '无特殊形态'
            print(f"{indicator_name}: {latest_score:.1f}分 - {patterns}")
        
        print("\n" + "-"*40)
        print("识别的技术形态")
        print("-"*40)
        if analysis_result['patterns']:
            for pattern in analysis_result['patterns']:
                print(f"- {pattern}")
        else:
            print("未识别到特殊技术形态")
        
        print("\n" + "-"*40)
        print("最近10日评分走势")
        print("-"*40)
        score_history = analysis_result['score_history']
        for i, score in enumerate(score_history):
            print(f"第{i+1}日: {score:.1f}")
        
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='股票技术指标评分分析')
    parser.add_argument('stock_code', help='股票代码，如：000001.SZ')
    parser.add_argument('--days', type=int, default=100, help='分析天数，默认100天')
    parser.add_argument('--config', help='自定义指标配置文件路径（JSON格式）')
    parser.add_argument('--quiet', action='store_true', help='静默模式，只输出关键信息')
    
    args = parser.parse_args()
    
    try:
        analyzer = StockScoreAnalyzer()
        
        # 加载自定义配置
        config = None
        if args.config:
            import json
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # 分析股票
        logger.info(f"开始分析股票: {args.stock_code}")
        result = analyzer.analyze_stock(args.stock_code, config)
        
        if args.quiet:
            # 静默模式，只输出关键信息
            print(f"{args.stock_code},{result['stock_info']['name']},{result['comprehensive_score']:.1f},{result['recommendation']['action']}")
        else:
            # 完整报告
            analyzer.print_analysis_report(result)
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 