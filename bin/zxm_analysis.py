#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZXM体系分析命令行工具

基于ZXM体系分析股票买点和吸筹形态
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import json

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IDataAccess
from utils.decorators import exception_handler, performance_monitor
from utils.logger import get_logger
from indicators.complete_indicator_registry import complete_registry
from enums.indicator_types import IndicatorType, TimeFrame
from utils import path_utils
from utils import date_utils
from indicators.zxm.buy_point_indicators import ZXMBuyPointIndicator
from indicators.zxm.zxm_score_indicator import ZXMScoreIndicator
from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer

logger = get_logger(__name__)

class ZXMAnalysisSystem:
    """ZXM指标分析系统"""
    
    def __init__(self):
        """初始化ZXM分析系统"""
        self.container = get_container()
        self.data_access = self.get_service(DataAccessInterface)
        self.buypoint_analyzer = BuyPointAnalyzer()
        self.analysis_results = {}
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def load_stock_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载股票数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            logger.info(f"加载股票数据: {code}, {start_date} - {end_date}")
            
            # 使用数据访问接口获取数据
            stock_data = self.data_access.get_stock_data(
                code=code,
                start_date=start_date,
                end_date=end_date,
                level='日线'
            )
            
            if stock_data.empty:
                logger.warning(f"未找到股票 {code} 的数据")
                return pd.DataFrame()
                
            logger.info(f"成功加载 {len(stock_data)} 条数据")
            return stock_data
            
        except Exception as e:
            logger.error(f"加载股票数据失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def calculate_zxm_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算ZXM指标
        
        Args:
            data: 股票数据
            
        Returns:
            Dict[str, pd.Series]: ZXM指标结果
        """
        try:
            logger.info("开始计算ZXM指标")
            
            results = {}
            
            # 计算ZXM买点指标
            zxm_buypoint = ZXMBuyPointIndicator()
            buypoint_result = zxm_buypoint.calculate(data)
            
            if isinstance(buypoint_result, dict):
                results.update(buypoint_result)
            else:
                results['zxm_buypoint'] = buypoint_result
            
            # 计算ZXM评分指标
            zxm_score = ZXMScoreIndicator()
            score_result = zxm_score.calculate(data)
            
            if isinstance(score_result, dict):
                results.update(score_result)
            else:
                results['zxm_score'] = score_result
            
            logger.info(f"ZXM指标计算完成，生成了 {len(results)} 个指标")
            return results
            
        except Exception as e:
            logger.error(f"计算ZXM指标失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=8.0)
    def analyze_buy_signals(self, data: pd.DataFrame, 
                           zxm_indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        分析买入信号
        
        Args:
            data: 股票数据
            zxm_indicators: ZXM指标结果
            
        Returns:
            Dict[str, Any]: 买入信号分析结果
        """
        try:
            logger.info("开始分析买入信号")
            
            # 使用买点分析器分析信号
            buypoint_result = self.buypoint_analyzer.analyze_buypoints(
                data=data,
                indicators=zxm_indicators
            )
            
            # 提取关键信号信息
            signals = {
                'latest_signals': {},
                'signal_history': {},
                'signal_strength': {},
                'buy_points': []
            }
            
            # 获取最新信号
            for indicator_name, values in zxm_indicators.items():
                if not values.empty:
                    signals['latest_signals'][indicator_name] = values.iloc[-1]
                    signals['signal_history'][indicator_name] = values.tail(10).tolist()
                    
                    # 计算信号强度
                    if indicator_name.endswith('_score'):
                        recent_values = values.tail(5)
                        if len(recent_values) > 1:
                            trend = recent_values.iloc[-1] - recent_values.iloc[0]
                            signals['signal_strength'][indicator_name] = {
                                'current': recent_values.iloc[-1],
                                'trend': trend,
                                'volatility': recent_values.std()
                            }
            
            # 识别买点
            signals['buy_points'] = self._identify_buy_points(data, zxm_indicators)
            
            # 生成综合评估
            signals['comprehensive_assessment'] = self._generate_comprehensive_assessment(
                signals['latest_signals'], 
                signals['signal_strength']
            )
            
            logger.info("买入信号分析完成")
            return signals
            
        except Exception as e:
            logger.error(f"分析买入信号失败: {e}")
            raise
    
    def _identify_buy_points(self, data: pd.DataFrame, 
                           indicators: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """识别买点"""
        buy_points = []
        
        try:
            # 基于ZXM指标识别买点
            for i in range(len(data) - 10, len(data)):  # 检查最近10个交易日
                if i < 0:
                    continue
                    
                date = data.iloc[i]['date'] if 'date' in data.columns else data.index[i]
                price = data.iloc[i]['close']
                
                # 检查是否满足买点条件
                is_buy_point = False
                buy_signals = []
                
                for indicator_name, values in indicators.items():
                    if i < len(values) and not pd.isna(values.iloc[i]):
                        value = values.iloc[i]
                        
                        # ZXM买点条件判断
                        if indicator_name == 'zxm_buypoint' and value > 0.7:
                            is_buy_point = True
                            buy_signals.append(f"{indicator_name}信号强度: {value:.3f}")
                        elif indicator_name == 'zxm_score' and value > 80:
                            is_buy_point = True
                            buy_signals.append(f"{indicator_name}评分: {value:.1f}")
                
                if is_buy_point:
                    buy_points.append({
                        'date': str(date),
                        'price': price,
                        'signals': buy_signals,
                        'confidence': self._calculate_buy_point_confidence(indicators, i)
                    })
            
        except Exception as e:
            logger.error(f"识别买点失败: {e}")
        
        return buy_points
    
    def _calculate_buy_point_confidence(self, indicators: Dict[str, pd.Series], 
                                      index: int) -> float:
        """计算买点置信度"""
        try:
            confidence_scores = []
            
            for indicator_name, values in indicators.items():
                if index < len(values) and not pd.isna(values.iloc[index]):
                    value = values.iloc[index]
                    
                    if indicator_name == 'zxm_buypoint':
                        confidence_scores.append(min(value, 1.0))
                    elif indicator_name == 'zxm_score':
                        confidence_scores.append(min(value / 100.0, 1.0))
            
            return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
        except Exception as e:
            logger.error(f"计算买点置信度失败: {e}")
            return 0.0
    
    def _generate_comprehensive_assessment(self, latest_signals: Dict[str, float], 
                                         signal_strength: Dict[str, Dict]) -> Dict[str, Any]:
        """生成综合评估"""
        try:
            assessment = {
                'overall_score': 0.0,
                'recommendation': '观望',
                'confidence': '低',
                'key_factors': [],
                'risk_warnings': []
            }
            
            # 计算综合评分
            scores = []
            for indicator_name, value in latest_signals.items():
                if indicator_name.endswith('_score'):
                    scores.append(min(value / 100.0, 1.0))
                elif indicator_name.endswith('_buypoint'):
                    scores.append(min(value, 1.0))
            
            if scores:
                assessment['overall_score'] = sum(scores) / len(scores)
            
            # 生成建议
            overall_score = assessment['overall_score']
            if overall_score >= 0.8:
                assessment['recommendation'] = '强烈买入'
                assessment['confidence'] = '高'
                assessment['key_factors'].append('多项ZXM指标发出强烈买入信号')
            elif overall_score >= 0.6:
                assessment['recommendation'] = '买入'
                assessment['confidence'] = '中高'
                assessment['key_factors'].append('ZXM指标偏向看涨')
            elif overall_score >= 0.4:
                assessment['recommendation'] = '谨慎观望'
                assessment['confidence'] = '中等'
                assessment['key_factors'].append('ZXM指标信号中性')
            else:
                assessment['recommendation'] = '观望'
                assessment['confidence'] = '低'
                assessment['risk_warnings'].append('ZXM指标信号偏弱')
            
            # 检查信号强度趋势
            for indicator_name, strength in signal_strength.items():
                if strength['trend'] > 0.1:
                    assessment['key_factors'].append(f'{indicator_name}呈上升趋势')
                elif strength['trend'] < -0.1:
                    assessment['risk_warnings'].append(f'{indicator_name}呈下降趋势')
            
            return assessment
            
        except Exception as e:
            logger.error(f"生成综合评估失败: {e}")
            return {'overall_score': 0.0, 'recommendation': '数据错误', 'confidence': '无'}
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=20.0)
    def run_comprehensive_analysis_Analysis(self, stock_codes: List[str], 
                                 start_date: str, end_date: str) -> Dict[str, Dict[str, Any]]:
        """
        运行综合分析
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Dict[str, Any]]: 综合分析结果
        """
        results = {}
        
        logger.info(f"开始ZXM综合分析，股票数量: {len(stock_codes)}")
        
        for i, code in enumerate(stock_codes, 1):
            try:
                logger.info(f"分析股票 {i}/{len(stock_codes)}: {code}")
                
                # 加载股票数据
                data = self.load_stock_data(code, start_date, end_date)
                
                if data.empty:
                    logger.warning(f"跳过股票 {code}，无数据")
                    continue
                
                # 计算ZXM指标
                zxm_indicators = self.calculate_zxm_indicators(data)
                
                # 分析买入信号
                buy_signals = self.analyze_buy_signals(data, zxm_indicators)
                
                # 组装结果
                results[code] = {
                    'stock_info': {
                        'code': code,
                        'latest_price': data.iloc[-1]['close'] if not data.empty else 0,
                        'latest_date': str(data.iloc[-1]['date']) if 'date' in data.columns else str(data.index[-1])
                    },
                    'zxm_indicators': {k: v.iloc[-1] if not v.empty else 0 for k, v in zxm_indicators.items()},
                    'buy_signals': buy_signals,
                    'data_quality': {
                        'data_points': len(data),
                        'indicator_count': len(zxm_indicators),
                        'signal_count': len(buy_signals.get('buy_points', []))
                    }
                }
                
            except Exception as e:
                logger.error(f"分析股票 {code} 失败: {e}")
                continue
                
        logger.info(f"ZXM综合分析完成，处理了 {len(results)} 只股票")
        return results
    
    @exception_handler(reraise=True)
    def save_analysis_results_Analysis(self, results: Dict[str, Dict[str, Any]], 
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
            
            # 生成汇总报告
            summary_file = output_file.replace('.json', '_summary.txt')
            self._generate_analysis_summary(results, summary_file)
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
            raise
    
    def _generate_analysis_summary(self, results: Dict[str, Dict[str, Any]], 
                                 summary_file: str) -> None:
        """生成分析汇总报告"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("ZXM指标分析汇总报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"分析股票数量: {len(results)}\n\n")
                
                if results:
                    # 统计买入推荐的股票
                    strong_buy_stocks = []
                    buy_stocks = []
                    
                    for code, result in results.items():
                        recommendation = result['buy_signals']['comprehensive_assessment']['recommendation']
                        overall_score = result['buy_signals']['comprehensive_assessment']['overall_score']
                        
                        if recommendation == '强烈买入':
                            strong_buy_stocks.append((code, overall_score))
                        elif recommendation == '买入':
                            buy_stocks.append((code, overall_score))
                    
                    # 按评分排序
                    strong_buy_stocks.sort(key=lambda x: x[1], reverse=True)
                    buy_stocks.sort(key=lambda x: x[1], reverse=True)
                    
                    f.write("强烈买入推荐股票:\n")
                    f.write("-" * 40 + "\n")
                    if strong_buy_stocks:
                        for i, (code, score) in enumerate(strong_buy_stocks[:10], 1):
                            f.write(f"{i:2d}. {code}: {score:.3f}\n")
                    else:
                        f.write("无\n")
                    
                    f.write("\n买入推荐股票:\n")
                    f.write("-" * 40 + "\n")
                    if buy_stocks:
                        for i, (code, score) in enumerate(buy_stocks[:10], 1):
                            f.write(f"{i:2d}. {code}: {score:.3f}\n")
                    else:
                        f.write("无\n")
                    
                    f.write("\n详细分析结果:\n")
                    f.write("-" * 40 + "\n")
                    for code, result in list(results.items())[:10]:  # 只显示前10个
                        stock_info = result['stock_info']
                        assessment = result['buy_signals']['comprehensive_assessment']
                        
                        f.write(f"\n{code} ({stock_info['latest_price']:.2f}):\n")
                        f.write(f"  推荐: {assessment['recommendation']}\n")
                        f.write(f"  置信度: {assessment['confidence']}\n")
                        f.write(f"  综合评分: {assessment['overall_score']:.3f}\n")
                        
                        if assessment['key_factors']:
                            f.write(f"  关键因素: {', '.join(assessment['key_factors'])}\n")
                        if assessment['risk_warnings']:
                            f.write(f"  风险提示: {', '.join(assessment['risk_warnings'])}\n")
                
            logger.info(f"分析汇总报告已生成: {summary_file}")
            
        except Exception as e:
            logger.error(f"生成分析汇总报告失败: {e}")

@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=120.0)
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ZXM指标分析系统')
    parser.add_argument('--codes', type=str, nargs='+', 
                       help='股票代码列表')
    parser.add_argument('--start-date', type=str, 
                       default=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, 
                       default='data/result/zxm_analysis_results.json',
                       help='输出文件路径')
    parser.add_argument('--single', type=str,
                       help='分析单个股票代码')
    
    args = parser.parse_args()
    
    try:
        # 初始化ZXM分析系统
        zxm_system = ZXMAnalysisSystem()
        
        # 获取股票代码列表
        if args.single:
            stock_codes = [args.single]
        elif args.codes:
            stock_codes = args.codes
        else:
            # 默认使用一些示例股票
            stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '603359.SH']
        
        logger.info(f"开始ZXM指标分析")
        logger.info(f"股票代码: {stock_codes}")
        logger.info(f"时间范围: {args.start_date} - {args.end_date}")
        
        # 运行综合分析
        results = zxm_system.run_comprehensive_analysis_Analysis(
            stock_codes=stock_codes,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # 保存结果
        zxm_system.save_analysis_results_Analysis(results, args.output)
        
        # 如果是单个股票分析，输出详细结果
        if args.single and args.single in results:
            result = results[args.single]
            assessment = result['buy_signals']['comprehensive_assessment']
            
            print(f"\n{args.single} ZXM分析结果:")
            print(f"推荐操作: {assessment['recommendation']}")
            print(f"置信度: {assessment['confidence']}")
            print(f"综合评分: {assessment['overall_score']:.3f}")
            print(f"最新价格: {result['stock_info']['latest_price']:.2f}")
            
            if assessment['key_factors']:
                print(f"关键因素: {', '.join(assessment['key_factors'])}")
            if assessment['risk_warnings']:
                print(f"风险提示: {', '.join(assessment['risk_warnings'])}")
        
        logger.info("ZXM指标分析完成")
        
    except Exception as e:
        logger.error(f"ZXM指标分析失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 