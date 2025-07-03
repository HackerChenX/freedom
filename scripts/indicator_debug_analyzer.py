#!/usr/bin/env python3
"""
指标调试分析脚本
深入分析无选股指标的计算结果和选股逻辑
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.production_indicator_validator import ProductionIndicatorValidator
from utils.logger import get_logger

class IndicatorDebugAnalyzer:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.validator = ProductionIndicatorValidator()
        
        # 选择几个典型的无选股指标进行深度分析
        self.debug_indicators = [
            'macd',      # 基础指标
            'boll',      # 基础指标
            'kdj',       # 基础指标
            'enhanced_rsi',  # 增强指标
            'composite',     # 复合指标
        ]
        
        # 测试股票
        self.test_stocks = ['600000', '600036', '000001', '000002', '000858']
    
    def analyze_indicator_details(self, indicator_name: str, stock_code: str) -> Dict[str, Any]:
        """深度分析单个指标在单只股票上的计算结果"""
        try:
            print(f"\n🔍 深度分析指标: {indicator_name} 在股票 {stock_code}")
            
            # 获取指标实例
            if indicator_name not in self.validator.available_indicators:
                return {'error': f'指标 {indicator_name} 不可用'}
            
            indicator = self.validator.available_indicators[indicator_name]
            
            # 获取股票数据
            test_date = self.validator.get_latest_trade_date()
            stock_data = self.validator.get_stock_data(stock_code, test_date, days=100)
            
            if stock_data.empty:
                return {'error': f'无法获取股票 {stock_code} 数据'}
            
            print(f"   📊 股票数据: {len(stock_data)} 行")
            print(f"   📅 数据范围: {stock_data.index[0]} 到 {stock_data.index[-1]}")
            print(f"   💰 最新价格: {stock_data['close'].iloc[-1]:.2f}")
            
            # 计算指标
            result = indicator.calculate(stock_data)
            
            if result.empty:
                return {'error': f'指标计算结果为空'}
            
            print(f"   📈 指标结果: {len(result)} 行")
            print(f"   🔢 结果列: {list(result.columns)}")
            
            # 分析最后几行的结果
            last_rows = result.tail(5)
            print(f"   📋 最后5行数据:")
            for i, (date, row) in enumerate(last_rows.iterrows()):
                print(f"      {date.strftime('%Y-%m-%d')}: {dict(row)}")
            
            # 检查买入信号
            last_row = result.iloc[-1]
            
            buy_signal_columns = ['buy_signal', 'XG', 'signal']
            buy_signal_found = False
            buy_signal_value = None
            
            for col in buy_signal_columns:
                if col in result.columns:
                    buy_signal_value = last_row[col]
                    buy_signal_found = True
                    print(f"   🎯 买入信号列 '{col}': {buy_signal_value}")
                    break
            
            if not buy_signal_found:
                print(f"   ⚠️ 未找到买入信号列，可用列: {list(result.columns)}")
            
            # 分析信号分布
            signal_stats = {}
            for col in buy_signal_columns:
                if col in result.columns:
                    signal_data = result[col]
                    signal_stats[col] = {
                        'total_count': len(signal_data),
                        'true_count': int(signal_data.sum()) if signal_data.dtype == bool else int((signal_data == 1).sum()),
                        'false_count': int((~signal_data).sum()) if signal_data.dtype == bool else int((signal_data == 0).sum()),
                        'unique_values': list(signal_data.unique())
                    }
                    print(f"   📊 信号统计 '{col}': {signal_stats[col]}")
            
            return {
                'indicator_name': indicator_name,
                'stock_code': stock_code,
                'stock_data_rows': len(stock_data),
                'result_rows': len(result),
                'result_columns': list(result.columns),
                'last_values': dict(last_row),
                'buy_signal_found': buy_signal_found,
                'buy_signal_value': buy_signal_value,
                'signal_stats': signal_stats,
                'latest_price': float(stock_data['close'].iloc[-1])
            }
            
        except Exception as e:
            error_msg = f"分析指标 {indicator_name} 在股票 {stock_code} 时出错: {e}"
            self.logger.error(error_msg)
            print(f"   ❌ {error_msg}")
            return {'error': error_msg}
    
    def analyze_all_debug_indicators(self):
        """分析所有调试指标"""
        print("🔬 开始深度调试分析")
        print("=" * 80)
        
        all_results = {}
        
        for indicator_name in self.debug_indicators:
            print(f"\n🎯 分析指标: {indicator_name}")
            print("-" * 50)
            
            indicator_results = {}
            
            for stock_code in self.test_stocks:
                result = self.analyze_indicator_details(indicator_name, stock_code)
                indicator_results[stock_code] = result
            
            all_results[indicator_name] = indicator_results
            
            # 总结该指标的问题
            self.summarize_indicator_issues(indicator_name, indicator_results)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"results/validation/indicator_debug_analysis_{timestamp}.json"
        
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 详细分析结果已保存到: {result_file}")
        
        # 生成优化建议
        self.generate_optimization_suggestions(all_results)
        
        return all_results
    
    def summarize_indicator_issues(self, indicator_name: str, results: Dict[str, Dict]):
        """总结单个指标的问题"""
        print(f"\n📋 指标 {indicator_name} 问题总结:")
        
        total_stocks = len(results)
        error_count = sum(1 for r in results.values() if 'error' in r)
        success_count = total_stocks - error_count
        
        if error_count > 0:
            print(f"   ❌ 计算错误: {error_count}/{total_stocks} 只股票")
            for stock, result in results.items():
                if 'error' in result:
                    print(f"      {stock}: {result['error']}")
        
        if success_count > 0:
            print(f"   ✅ 计算成功: {success_count}/{total_stocks} 只股票")
            
            # 分析买入信号
            signal_found_count = 0
            true_signal_count = 0
            
            for stock, result in results.items():
                if 'error' not in result:
                    if result.get('buy_signal_found'):
                        signal_found_count += 1
                        if result.get('buy_signal_value'):
                            true_signal_count += 1
            
            print(f"   🎯 找到买入信号列: {signal_found_count}/{success_count} 只股票")
            print(f"   🚀 触发买入信号: {true_signal_count}/{success_count} 只股票")
            
            if signal_found_count == 0:
                print(f"   ⚠️ 问题: 指标没有标准的买入信号列")
            elif true_signal_count == 0:
                print(f"   ⚠️ 问题: 所有股票的买入信号都为False")
    
    def generate_optimization_suggestions(self, all_results: Dict[str, Dict]):
        """生成优化建议"""
        print(f"\n🚀 优化建议总结")
        print("=" * 80)
        
        # 问题分类
        no_signal_column = []
        no_true_signals = []
        calculation_errors = []
        
        for indicator_name, results in all_results.items():
            has_signal_column = False
            has_true_signals = False
            has_errors = False
            
            for stock, result in results.items():
                if 'error' in result:
                    has_errors = True
                else:
                    if result.get('buy_signal_found'):
                        has_signal_column = True
                        if result.get('buy_signal_value'):
                            has_true_signals = True
            
            if has_errors:
                calculation_errors.append(indicator_name)
            elif not has_signal_column:
                no_signal_column.append(indicator_name)
            elif not has_true_signals:
                no_true_signals.append(indicator_name)
        
        print(f"\n📊 问题分类统计:")
        print(f"   🔧 计算错误: {len(calculation_errors)} 个指标")
        if calculation_errors:
            print(f"      {calculation_errors}")
        
        print(f"   📋 缺少信号列: {len(no_signal_column)} 个指标")
        if no_signal_column:
            print(f"      {no_signal_column}")
        
        print(f"   🎯 信号全为False: {len(no_true_signals)} 个指标")
        if no_true_signals:
            print(f"      {no_true_signals}")
        
        print(f"\n🔧 具体优化方案:")
        
        if calculation_errors:
            print(f"\n1. 修复计算错误 ({len(calculation_errors)}个):")
            print(f"   - 检查指标类的__init__和calculate方法")
            print(f"   - 确保数据预处理正确")
            print(f"   - 修复可能的除零错误或数据类型问题")
        
        if no_signal_column:
            print(f"\n2. 添加买入信号列 ({len(no_signal_column)}个):")
            print(f"   - 为指标添加'buy_signal'列")
            print(f"   - 定义合理的买入条件逻辑")
            print(f"   - 确保信号生成逻辑正确")
        
        if no_true_signals:
            print(f"\n3. 调整信号参数 ({len(no_true_signals)}个):")
            print(f"   - 放宽买入条件的阈值")
            print(f"   - 检查信号逻辑是否过于严格")
            print(f"   - 考虑市场环境调整参数")

if __name__ == "__main__":
    analyzer = IndicatorDebugAnalyzer()
    analyzer.analyze_all_debug_indicators() 