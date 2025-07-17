#!/usr/bin/env python3
"""
扩大股票池测试脚本
将测试股票池从5只扩展到50只，验证无选股指标的表现
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.production_indicator_validator import Production_indicator_validator
from utils.logger import get_logger

class ExpandedStockPoolTester:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.validator = Production_indicator_validator()
        
        # 无选股的37个指标
        self.no_selection_indicators = [
            # 第1批-基础指标
            'macd', 'boll', 'kdj',
            # 第2批-趋势指标
            'sar',
            # 第3批-成交量指标
            'emv',
            # 第4批-波动率指标
            'stock_vix',
            # 第5批-ZXM专业指标
            'amplitude_elasticity', 'rise_elasticity', 'elasticity_score', 'market_breadth',
            # 第6批-增强指标
            'enhanced_rsi', 'enhanced_dmi', 'enhanced_macd_trend', 'enhanced_trix', 
            'enhanced_kdj_osc', 'enhanced_obv', 'enhanced_stochrsi', 'enhanced_wr', 'enhanced_macd_root',
            # 第7批-复合形态指标
            'composite', 'unified_ma', 'chip_distribution', 'institutional_behavior', 
            'candlestick_patterns', 'advanced_candlestick', 'patterns',
            # 第8批-工具公式指标
            'fibonacci_tools', 'gann_tools', 'elliott_wave', 'kdj_condition', 'macd_condition',
            # 第9批-多周期指标
            'monthly_kdj_trend_up', 'monthly_macd', 'weekly_kdj_d_or_dea_trend_up', 
            'weekly_kdj_d_trend_up', 'weekly_macd',
            # 第10批-震荡指标
            'stochrsi'
        ]
        
        # 扩大的股票池 - 50只股票
        self.expanded_stock_pool = [
            # 原有的5只
            '600000', '600036', '000001', '000002', '000858',
            # 新增45只股票
            '600519', '000858', '002415', '300059', '600276',
            '002304', '600887', '000002', '600036', '000001',
            '002271', '600809', '600031', '000063', '002008',
            '000338', '600104', '002142', '600009', '000166',
            '002475', '600585', '000725', '600196', '002230',
            '600703', '000876', '600028', '002027', '600048',
            '000423', '600362', '002131', '600690', '000157',
            '600050', '000776', '600837', '002049', '600660',
            '000768', '600872', '002081', '600765', '000629',
            '600309', '000895', '600588', '002180', '600256'
        ]
    
    def test_expanded_pool(self):
        """使用扩大的股票池测试无选股指标"""
        print("🚀 开始扩大股票池测试")
        print("=" * 80)
        print(f"原股票池: 5只股票")
        print(f"扩大股票池: {len(self.expanded_stock_pool)}只股票")
        print(f"待测试指标: {len(self.no_selection_indicators)}个")
        
        results = {
            'test_info': {
                'original_pool_size': 5,
                'expanded_pool_size': len(self.expanded_stock_pool),
                'test_indicators': len(self.no_selection_indicators),
                'timestamp': datetime.now().isoformat()
            },
            'results': []
        }
        
        improved_indicators = []
        still_no_selection = []
        
        for i, indicator_name in enumerate(self.no_selection_indicators, 1):
            print(f"\n🔍 测试指标 {i}/{len(self.no_selection_indicators)}: {indicator_name}")
            
            try:
                # 使用扩大的股票池进行测试
                start_time = time.time()
                
                # 临时修改validator的股票池获取方法来使用我们的扩大股票池
                original_get_stock_pool = self.validator.get_stock_pool
                self.validator.get_stock_pool = lambda test_date, max_stocks: self.expanded_stock_pool[:max_stocks]
                
                try:
                    result = self.validator.validate_single_indicator(
                        indicator_name, 
                        max_stocks=50
                    )
                finally:
                    # 恢复原始方法
                    self.validator.get_stock_pool = original_get_stock_pool
                
                execution_time = time.time() - start_time
                
                selected_count = len(result.get('selected_stocks', []))
                
                indicator_result = {
                    'indicator_name': indicator_name,
                    'success': result.get('success', False),
                    'selected_count': selected_count,
                    'selected_stocks': result.get('selected_stocks', []),
                    'execution_time': execution_time,
                    'error_message': result.get('error_message')
                }
                
                results['results'].append(indicator_result)
                
                if selected_count > 0:
                    improved_indicators.append(indicator_name)
                    print(f"   ✅ 成功选出 {selected_count} 只股票: {result.get('selected_stocks', [])}")
                else:
                    still_no_selection.append(indicator_name)
                    print(f"   ❌ 仍无选股结果")
                    
            except Exception as e:
                self.logger.error(f"测试指标 {indicator_name} 时出错: {e}")
                print(f"   ❌ 测试失败: {e}")
                still_no_selection.append(indicator_name)
                
                results['results'].append({
                    'indicator_name': indicator_name,
                    'success': False,
                    'selected_count': 0,
                    'selected_stocks': [],
                    'execution_time': 0,
                    'error_message': str(e)
                })
        
        # 生成总结报告
        print(f"\n" + "=" * 80)
        print(f"📊 扩大股票池测试结果总结")
        print(f"=" * 80)
        
        improvement_rate = (len(improved_indicators) / len(self.no_selection_indicators)) * 100
        
        print(f"\n🎯 总体改善情况:")
        print(f"   测试指标总数: {len(self.no_selection_indicators)}")
        print(f"   获得选股能力: {len(improved_indicators)} 个")
        print(f"   仍无选股能力: {len(still_no_selection)} 个")
        print(f"   改善率: {improvement_rate:.1f}%")
        
        if improved_indicators:
            print(f"\n✅ 获得选股能力的指标 ({len(improved_indicators)}个):")
            for indicator in improved_indicators:
                result = next(r for r in results['results'] if r['indicator_name'] == indicator)
                print(f"   {indicator}: {result['selected_count']} 只股票")
        
        if still_no_selection:
            print(f"\n❌ 仍无选股能力的指标 ({len(still_no_selection)}个):")
            for indicator in still_no_selection:
                print(f"   {indicator}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"results/validation/expanded_pool_test_{timestamp}.json"
        
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {result_file}")
        
        return results, improved_indicators, still_no_selection

if __name__ == "__main__":
    tester = Expanded_stock_pool_tester()
    results, improved, still_no_selection = tester.test_expanded_pool() 