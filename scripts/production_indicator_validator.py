#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生产环境指标验证工具

基于ClickHouse真实数据验证指标选股的可靠性
支持逐个指标验证和批量验证
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.clickhouse_db import get_clickhouse_db
from indicators.zxm.buy_point_indicators import (
    ZXMVolumeShrink, ZXMBSAbsorb, ZXMTurnover, 
    ZXMDailyMACD, ZXMMACallback
)
from indicators.enhanced_macd import EnhancedMACD
from indicators.enhanced_rsi import EnhancedRSI
from indicators.unified_ma import UnifiedMA
from utils.logger import get_logger

logger = get_logger(__name__)


class ProductionIndicatorValidator:
    """生产环境指标验证器"""
    
    def __init__(self):
        """初始化验证器"""
        try:
            self.db = get_clickhouse_db()
            logger.info("✅ 成功连接到ClickHouse数据库")
            
            # 初始化可用指标 - 只包含确实存在的指标
            self.available_indicators = {}
            
            # 尝试导入ZXM指标
            try:
                self.available_indicators['volume_shrink'] = ZXMVolumeShrink()
                self.available_indicators['bs_absorb'] = ZXMBSAbsorb()
                self.available_indicators['turnover'] = ZXMTurnover()
                self.available_indicators['daily_macd'] = ZXMDailyMACD()
                self.available_indicators['ma_callback'] = ZXMMACallback()
                logger.info("✅ ZXM指标加载成功")
            except Exception as e:
                logger.warning(f"⚠️ ZXM指标加载失败: {e}")
            
            # 尝试导入增强指标
            try:
                self.available_indicators['enhanced_macd'] = EnhancedMACD()
                logger.info("✅ EnhancedMACD指标加载成功")
            except Exception as e:
                logger.warning(f"⚠️ EnhancedMACD指标加载失败: {e}")
            
            try:
                self.available_indicators['enhanced_rsi'] = EnhancedRSI()
                logger.info("✅ EnhancedRSI指标加载成功")
            except Exception as e:
                logger.warning(f"⚠️ EnhancedRSI指标加载失败: {e}")
            
            try:
                self.available_indicators['unified_ma'] = UnifiedMA()
                logger.info("✅ UnifiedMA指标加载成功")
            except Exception as e:
                logger.warning(f"⚠️ UnifiedMA指标加载失败: {e}")
            
            if not self.available_indicators:
                raise Exception("没有可用的指标")
            
            logger.info(f"✅ 初始化完成，可用指标: {list(self.available_indicators.keys())}")
            
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            raise
    
    def get_latest_trade_date(self) -> str:
        """获取ClickHouse中最新的交易日期"""
        try:
            latest_date = self.db.get_stock_max_date()
            if latest_date:
                logger.info(f"📅 获取到最新交易日期: {latest_date}")
                return latest_date
            else:
                # 如果无法获取，使用当前日期的前一个工作日
                today = datetime.now()
                if today.weekday() == 0:  # 周一
                    latest_date = (today - timedelta(days=3)).strftime('%Y-%m-%d')
                elif today.weekday() == 6:  # 周日
                    latest_date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
                else:
                    latest_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
                
                logger.warning(f"⚠️ 无法从数据库获取最新日期，使用估算日期: {latest_date}")
                return latest_date
        except Exception as e:
            logger.error(f"❌ 获取最新交易日期失败: {e}")
            # 返回一个合理的默认日期
            return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    def get_stock_pool(self, test_date: str, max_stocks: int = 500) -> List[str]:
        """
        获取测试股票池
        
        Args:
            test_date: 测试日期
            max_stocks: 最大股票数量
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            logger.info(f"🔍 获取 {test_date} 的股票池...")
            
            # 获取指定日期的股票数据，过滤掉ST股票和价格异常的股票
            query = f"""
            SELECT DISTINCT code, name, close, volume
            FROM stock_info
            WHERE date = '{test_date}'
              AND level = '日线'
              AND close > 2.0
              AND close < 200.0
              AND volume > 1000
              AND name NOT LIKE '%%ST%%'
              AND name NOT LIKE '%%*%%'
            ORDER BY volume DESC
            LIMIT {max_stocks}
            """
            
            result = self.db.query(query)
            
            if result.empty:
                logger.warning(f"⚠️ 未找到 {test_date} 的股票数据")
                return []
            
            if 'code' in result.columns:
                stock_codes = result['code'].tolist()
            elif 'col_0' in result.columns:
                stock_codes = result['col_0'].tolist()
            else:
                logger.warning(f"⚠️ 无法找到股票代码列，可用列: {result.columns.tolist()}")
                return []
                
            logger.info(f"✅ 获取到 {len(stock_codes)} 只股票")
            
            return stock_codes
            
        except Exception as e:
            logger.error(f"❌ 获取股票池失败: {e}")
            return []
    
    def get_stock_data(self, stock_code: str, end_date: str, days: int = 100) -> pd.DataFrame:
        """
        获取股票历史数据
        
        Args:
            stock_code: 股票代码
            end_date: 结束日期
            days: 历史天数
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            # 计算开始日期（考虑到交易日）
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days*2)).strftime('%Y-%m-%d')
            
            query = f"""
            SELECT date, open, high, low, close, volume, turnover
            FROM stock_info
            WHERE code = '{stock_code}'
              AND level = '日线'
              AND date >= '{start_date}'
              AND date <= '{end_date}'
            ORDER BY date ASC
            """
            
            result = self.db.query(query)
            
            if result.empty:
                return pd.DataFrame()
            
            # 确保数据类型正确和列名映射
            if 'col_0' in result.columns:
                # 映射通用列名到实际列名
                column_mapping = {
                    'col_0': 'date',
                    'col_1': 'open', 
                    'col_2': 'high',
                    'col_3': 'low',
                    'col_4': 'close',
                    'col_5': 'volume',
                    'col_6': 'turnover'
                }
                result = result.rename(columns=column_mapping)
            
            if 'date' in result.columns:
                result['date'] = pd.to_datetime(result['date'])
                result = result.set_index('date')
            else:
                logger.warning(f"⚠️ 无法找到date列，可用列: {result.columns.tolist()}")
                return pd.DataFrame()
            
            # 只保留最近的指定天数
            if len(result) > days:
                result = result.tail(days)
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ 获取股票 {stock_code} 数据失败: {e}")
            return pd.DataFrame()
    
    def validate_single_indicator(self, indicator_name: str, test_date: str = None, 
                                 max_stocks: int = 500) -> Dict[str, Any]:
        """
        验证单个指标的选股效果
        
        Args:
            indicator_name: 指标名称
            test_date: 测试日期，None表示使用最新日期
            max_stocks: 最大测试股票数量
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if indicator_name not in self.available_indicators:
            raise ValueError(f"未知指标: {indicator_name}，可用指标: {list(self.available_indicators.keys())}")
        
        if test_date is None:
            test_date = self.get_latest_trade_date()
        
        logger.info(f"🚀 开始验证指标: {indicator_name}")
        logger.info(f"📅 测试日期: {test_date}")
        logger.info(f"📊 最大股票数: {max_stocks}")
        
        start_time = time.time()
        
        # 获取股票池
        stock_pool = self.get_stock_pool(test_date, max_stocks)
        if not stock_pool:
            return {
                'indicator_name': indicator_name,
                'test_date': test_date,
                'status': 'failed',
                'error': '无法获取股票池'
            }
        
        # 获取指标实例
        indicator = self.available_indicators[indicator_name]
        
        # 验证结果
        selected_stocks = []
        failed_stocks = []
        indicator_values = {}
        
        logger.info(f"🔄 开始处理 {len(stock_pool)} 只股票...")
        
        for i, stock_code in enumerate(stock_pool):
            try:
                if (i + 1) % 50 == 0:
                    logger.info(f"📈 进度: {i+1}/{len(stock_pool)} ({(i+1)/len(stock_pool)*100:.1f}%)")
                
                # 获取股票数据
                stock_data = self.get_stock_data(stock_code, test_date)
                
                if stock_data.empty or len(stock_data) < 20:
                    failed_stocks.append(stock_code)
                    continue
                
                # 计算指标
                result = indicator.calculate(stock_data)
                
                if result.empty:
                    failed_stocks.append(stock_code)
                    continue
                
                # 检查买入信号
                last_row = result.iloc[-1]
                
                # 根据不同指标类型判断买入信号
                has_buy_signal = False
                indicator_value = None
                
                if indicator_name in ['volume_shrink', 'bs_absorb', 'daily_macd', 'turnover', 'ma_callback']:
                    # ZXM指标通常有XG列表示买点信号
                    if 'XG' in result.columns:
                        has_buy_signal = bool(last_row['XG'])
                        indicator_value = last_row['XG']
                    elif 'buy_signal' in result.columns:
                        has_buy_signal = bool(last_row['buy_signal'])
                        indicator_value = last_row['buy_signal']
                else:
                    # 其他指标使用buy_signal列
                    if 'buy_signal' in result.columns:
                        has_buy_signal = bool(last_row['buy_signal'])
                        indicator_value = last_row['buy_signal']
                
                if has_buy_signal:
                    selected_stocks.append(stock_code)
                
                # 保存指标值用于分析 - 确保所有值都可以JSON序列化
                indicator_values[stock_code] = {
                    'has_signal': bool(has_buy_signal),
                    'value': float(indicator_value) if indicator_value is not None and not isinstance(indicator_value, bool) else (1.0 if indicator_value else 0.0),
                    'last_close': float(last_row.get('close', 0))
                }
                
            except Exception as e:
                logger.warning(f"⚠️ 处理股票 {stock_code} 失败: {e}")
                failed_stocks.append(stock_code)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 计算统计结果
        total_tested = len(stock_pool)
        success_count = len(indicator_values)
        selected_count = len(selected_stocks)
        failed_count = len(failed_stocks)
        
        selection_rate = selected_count / success_count if success_count > 0 else 0
        success_rate = success_count / total_tested if total_tested > 0 else 0
        
        result = {
            'indicator_name': indicator_name,
            'test_date': test_date,
            'status': 'success',
            'validation_time': datetime.now().isoformat(),
            'processing_time_seconds': round(processing_time, 2),
            'statistics': {
                'total_tested': total_tested,
                'success_processed': success_count,
                'failed_processed': failed_count,
                'selected_count': selected_count,
                'selection_rate': round(selection_rate, 4),
                'success_rate': round(success_rate, 4)
            },
            'selected_stocks': selected_stocks,
            'failed_stocks': failed_stocks[:10],  # 只保留前10个失败的股票
            'indicator_analysis': {
                'signal_distribution': self._analyze_signal_distribution(indicator_values),
                'top_signals': self._get_top_signals(indicator_values, 10)
            }
        }
        
        logger.info(f"✅ 指标 {indicator_name} 验证完成")
        logger.info(f"📊 选股结果: {selected_count}/{success_count} = {selection_rate:.2%}")
        logger.info(f"⏱️ 处理时间: {processing_time:.1f}秒")
        
        return result
    
    def validate_multiple_indicators(self, indicator_names: List[str], test_date: str = None,
                                   max_stocks: int = 500) -> Dict[str, Any]:
        """
        批量验证多个指标
        
        Args:
            indicator_names: 指标名称列表
            test_date: 测试日期
            max_stocks: 最大股票数量
            
        Returns:
            Dict[str, Any]: 批量验证结果
        """
        if test_date is None:
            test_date = self.get_latest_trade_date()
        
        logger.info(f"🚀 开始批量验证 {len(indicator_names)} 个指标")
        logger.info(f"📋 指标列表: {indicator_names}")
        
        start_time = time.time()
        
        # 验证每个指标
        individual_results = {}
        for indicator_name in indicator_names:
            try:
                logger.info(f"🔄 验证指标: {indicator_name}")
                result = self.validate_single_indicator(indicator_name, test_date, max_stocks)
                individual_results[indicator_name] = result
            except Exception as e:
                logger.error(f"❌ 验证指标 {indicator_name} 失败: {e}")
                individual_results[indicator_name] = {
                    'indicator_name': indicator_name,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 分析指标间的重叠度
        overlap_analysis = self._analyze_indicator_overlap(individual_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        batch_result = {
            'validation_type': 'batch',
            'test_date': test_date,
            'total_indicators': len(indicator_names),
            'validation_time': datetime.now().isoformat(),
            'total_processing_time_seconds': round(total_time, 2),
            'individual_results': individual_results,
            'overlap_analysis': overlap_analysis,
            'summary': self._generate_batch_summary(individual_results)
        }
        
        logger.info(f"✅ 批量验证完成，总耗时: {total_time:.1f}秒")
        
        return batch_result
    
    def _analyze_signal_distribution(self, indicator_values: Dict[str, Dict]) -> Dict[str, Any]:
        """分析信号分布"""
        signal_count = sum(1 for v in indicator_values.values() if v['has_signal'])
        total_count = len(indicator_values)
        
        return {
            'total_stocks': total_count,
            'signal_stocks': signal_count,
            'no_signal_stocks': total_count - signal_count,
            'signal_ratio': round(signal_count / total_count, 4) if total_count > 0 else 0
        }
    
    def _get_top_signals(self, indicator_values: Dict[str, Dict], top_n: int = 10) -> List[Dict[str, Any]]:
        """获取信号最强的股票"""
        signal_stocks = [(code, data) for code, data in indicator_values.items() if data['has_signal']]
        
        # 按最后收盘价排序（可以根据具体指标调整排序逻辑）
        signal_stocks.sort(key=lambda x: x[1]['last_close'], reverse=True)
        
        return [
            {
                'stock_code': code,
                'signal_value': data['value'],
                'last_close': data['last_close']
            }
            for code, data in signal_stocks[:top_n]
        ]
    
    def _analyze_indicator_overlap(self, individual_results: Dict[str, Dict]) -> Dict[str, Any]:
        """分析指标间的重叠度"""
        # 收集所有成功的指标的选股结果
        successful_indicators = {}
        for indicator_name, result in individual_results.items():
            if result.get('status') == 'success':
                successful_indicators[indicator_name] = set(result.get('selected_stocks', []))
        
        if len(successful_indicators) < 2:
            return {'message': '需要至少2个成功的指标才能分析重叠度'}
        
        # 计算两两重叠度
        overlap_matrix = {}
        indicator_names = list(successful_indicators.keys())
        
        for i, indicator1 in enumerate(indicator_names):
            overlap_matrix[indicator1] = {}
            for j, indicator2 in enumerate(indicator_names):
                if i == j:
                    overlap_matrix[indicator1][indicator2] = 1.0
                else:
                    stocks1 = successful_indicators[indicator1]
                    stocks2 = successful_indicators[indicator2]
                    
                    if len(stocks1) == 0 or len(stocks2) == 0:
                        overlap_ratio = 0.0
                    else:
                        intersection = len(stocks1 & stocks2)
                        union = len(stocks1 | stocks2)
                        overlap_ratio = intersection / union if union > 0 else 0
                    
                    overlap_matrix[indicator1][indicator2] = round(overlap_ratio, 4)
        
        # 找出所有指标的交集
        all_selected = set()
        for stocks in successful_indicators.values():
            all_selected.update(stocks)
        
        common_stocks = set(successful_indicators[indicator_names[0]])
        for indicator_name in indicator_names[1:]:
            common_stocks &= successful_indicators[indicator_name]
        
        return {
            'overlap_matrix': overlap_matrix,
            'total_unique_stocks': len(all_selected),
            'common_stocks': list(common_stocks),
            'common_stocks_count': len(common_stocks)
        }
    
    def _generate_batch_summary(self, individual_results: Dict[str, Dict]) -> Dict[str, Any]:
        """生成批量验证摘要"""
        successful_results = [r for r in individual_results.values() if r.get('status') == 'success']
        failed_results = [r for r in individual_results.values() if r.get('status') == 'failed']
        
        if not successful_results:
            return {
                'success_count': 0,
                'failed_count': len(failed_results),
                'message': '所有指标验证都失败了'
            }
        
        # 统计成功的指标
        selection_rates = [r['statistics']['selection_rate'] for r in successful_results]
        processing_times = [r['processing_time_seconds'] for r in successful_results]
        
        return {
            'success_count': len(successful_results),
            'failed_count': len(failed_results),
            'avg_selection_rate': round(np.mean(selection_rates), 4),
            'max_selection_rate': round(max(selection_rates), 4),
            'min_selection_rate': round(min(selection_rates), 4),
            'avg_processing_time': round(np.mean(processing_times), 2),
            'total_processing_time': round(sum(processing_times), 2),
            'best_indicator': max(successful_results, key=lambda x: x['statistics']['selection_rate'])['indicator_name'],
            'fastest_indicator': min(successful_results, key=lambda x: x['processing_time_seconds'])['indicator_name']
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results/validation") -> str:
        """
        保存验证结果
        
        Args:
            results: 验证结果
            output_dir: 输出目录
            
        Returns:
            str: 保存的文件路径
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if results.get('validation_type') == 'batch':
                filename = f"batch_validation_{timestamp}.json"
            else:
                indicator_name = results.get('indicator_name', 'unknown')
                filename = f"single_validation_{indicator_name}_{timestamp}.json"
            
            filepath = os.path.join(output_dir, filename)
            
            # 保存JSON结果
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 生成文本报告
            report_path = filepath.replace('.json', '.txt')
            self._generate_text_report(results, report_path)
            
            # 如果有选股结果，保存CSV
            csv_path = self._save_selection_csv(results, output_dir, timestamp)
            
            logger.info(f"✅ 验证结果已保存:")
            logger.info(f"  📄 JSON结果: {filepath}")
            logger.info(f"  📄 文本报告: {report_path}")
            if csv_path:
                logger.info(f"  📄 选股结果: {csv_path}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ 保存验证结果失败: {e}")
            return ""
    
    def _generate_text_report(self, results: Dict[str, Any], report_path: str):
        """生成文本格式的验证报告"""
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("生产环境指标验证报告\n")
                f.write("="*80 + "\n\n")
                
                if results.get('validation_type') == 'batch':
                    # 批量验证报告
                    f.write(f"验证类型: 批量验证\n")
                    f.write(f"测试日期: {results.get('test_date')}\n")
                    f.write(f"验证时间: {results.get('validation_time')}\n")
                    f.write(f"总处理时间: {results.get('total_processing_time_seconds')}秒\n\n")
                    
                    # 摘要信息
                    summary = results.get('summary', {})
                    f.write("验证摘要:\n")
                    f.write(f"  成功指标数: {summary.get('success_count')}\n")
                    f.write(f"  失败指标数: {summary.get('failed_count')}\n")
                    f.write(f"  平均选股率: {summary.get('avg_selection_rate', 0):.2%}\n")
                    f.write(f"  最佳指标: {summary.get('best_indicator')}\n")
                    f.write(f"  最快指标: {summary.get('fastest_indicator')}\n\n")
                    
                    # 各指标详情
                    f.write("各指标详细结果:\n")
                    f.write("-"*60 + "\n")
                    
                    for indicator_name, result in results.get('individual_results', {}).items():
                        f.write(f"\n指标: {indicator_name}\n")
                        if result.get('status') == 'success':
                            stats = result.get('statistics', {})
                            f.write(f"  状态: 成功\n")
                            f.write(f"  选股数量: {stats.get('selected_count')}\n")
                            f.write(f"  选股率: {stats.get('selection_rate', 0):.2%}\n")
                            f.write(f"  处理时间: {result.get('processing_time_seconds')}秒\n")
                        else:
                            f.write(f"  状态: 失败\n")
                            f.write(f"  错误: {result.get('error')}\n")
                
                else:
                    # 单个指标验证报告
                    f.write(f"验证类型: 单指标验证\n")
                    f.write(f"指标名称: {results.get('indicator_name')}\n")
                    f.write(f"测试日期: {results.get('test_date')}\n")
                    f.write(f"验证时间: {results.get('validation_time')}\n")
                    f.write(f"处理时间: {results.get('processing_time_seconds')}秒\n\n")
                    
                    if results.get('status') == 'success':
                        stats = results.get('statistics', {})
                        f.write("验证结果:\n")
                        f.write(f"  测试股票总数: {stats.get('total_tested')}\n")
                        f.write(f"  成功处理数: {stats.get('success_processed')}\n")
                        f.write(f"  失败处理数: {stats.get('failed_processed')}\n")
                        f.write(f"  选股数量: {stats.get('selected_count')}\n")
                        f.write(f"  选股率: {stats.get('selection_rate', 0):.2%}\n")
                        f.write(f"  成功率: {stats.get('success_rate', 0):.2%}\n\n")
                        
                        # 选股结果
                        selected_stocks = results.get('selected_stocks', [])
                        if selected_stocks:
                            f.write(f"选中股票 ({len(selected_stocks)}只):\n")
                            for i, stock in enumerate(selected_stocks[:20]):  # 只显示前20只
                                f.write(f"  {i+1:2d}. {stock}\n")
                            if len(selected_stocks) > 20:
                                f.write(f"  ... 还有{len(selected_stocks)-20}只股票\n")
                    else:
                        f.write(f"验证失败: {results.get('error')}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("报告生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                
        except Exception as e:
            logger.error(f"❌ 生成文本报告失败: {e}")
    
    def _save_selection_csv(self, results: Dict[str, Any], output_dir: str, timestamp: str) -> Optional[str]:
        """保存选股结果为CSV文件"""
        try:
            selected_stocks = []
            
            if results.get('validation_type') == 'batch':
                # 批量验证结果
                for indicator_name, result in results.get('individual_results', {}).items():
                    if result.get('status') == 'success':
                        for stock in result.get('selected_stocks', []):
                            selected_stocks.append({
                                'stock_code': stock,
                                'indicator': indicator_name
                            })
            else:
                # 单指标验证结果
                if results.get('status') == 'success':
                    indicator_name = results.get('indicator_name')
                    for stock in results.get('selected_stocks', []):
                        selected_stocks.append({
                            'stock_code': stock,
                            'indicator': indicator_name
                        })
            
            if not selected_stocks:
                return None
            
            # 保存CSV
            df = pd.DataFrame(selected_stocks)
            csv_path = os.path.join(output_dir, f"selection_results_{timestamp}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            return csv_path
            
        except Exception as e:
            logger.error(f"❌ 保存选股CSV失败: {e}")
            return None
    
    def list_available_indicators(self) -> List[str]:
        """列出所有可用的指标"""
        return list(self.available_indicators.keys())


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生产环境指标验证工具')
    parser.add_argument('--date', type=str, help='测试日期 (YYYY-MM-DD)，默认使用最新日期')
    parser.add_argument('--indicators', type=str, nargs='+', 
                       help='要验证的指标名称，多个指标用空格分隔')
    parser.add_argument('--max-stocks', type=int, default=500,
                       help='最大测试股票数量，默认500')
    parser.add_argument('--output', type=str, default='results/validation',
                       help='输出目录，默认results/validation')
    parser.add_argument('--list-indicators', action='store_true',
                       help='列出所有可用指标')
    
    args = parser.parse_args()
    
    try:
        # 初始化验证器
        validator = ProductionIndicatorValidator()
        
        # 列出可用指标
        if args.list_indicators:
            indicators = validator.list_available_indicators()
            print("可用指标:")
            for i, indicator in enumerate(indicators, 1):
                print(f"  {i:2d}. {indicator}")
            return
        
        # 验证指标参数
        if not args.indicators:
            print("请指定要验证的指标，使用 --list-indicators 查看可用指标")
            return
        
        available_indicators = validator.list_available_indicators()
        invalid_indicators = [ind for ind in args.indicators if ind not in available_indicators]
        if invalid_indicators:
            print(f"无效指标: {invalid_indicators}")
            print(f"可用指标: {available_indicators}")
            return
        
        # 执行验证
        if len(args.indicators) == 1:
            # 单指标验证
            results = validator.validate_single_indicator(
                indicator_name=args.indicators[0],
                test_date=args.date,
                max_stocks=args.max_stocks
            )
        else:
            # 批量验证
            results = validator.validate_multiple_indicators(
                indicator_names=args.indicators,
                test_date=args.date,
                max_stocks=args.max_stocks
            )
        
        # 保存结果
        output_path = validator.save_results(results, args.output)
        
        if output_path:
            print(f"\n✅ 验证完成，结果已保存到: {output_path}")
        else:
            print("\n❌ 验证失败")
            
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        print(f"❌ 程序执行失败: {e}")


if __name__ == '__main__':
    main() 