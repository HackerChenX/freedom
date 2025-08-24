#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD形态检测器 - 改进版

解决检测逻辑问题：
1. 使用最近的交易日而不是固定日期
2. 更灵活的形态检测逻辑
3. 增加调试信息
4. 提高检测成功率
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class ImprovedMacdPatternDetector:
    """改进的MACD形态检测器"""
    
    def __init__(self):
        """初始化检测器"""
        try:
            self.macd_indicator = MacdMacd()
            self.stock_data_service = get_stock_data_service()
            self.use_real_data = True
            print("✅ 使用真实MACD指标和数据服务")
        except Exception as e:
            print(f"⚠️ 初始化失败，使用模拟模式: {e}")
            self.macd_indicator = None
            self.stock_data_service = None
            self.use_real_data = False
    
    def test_macd_detection_logic(self, max_stocks: int = 1000) -> Dict[str, Any]:
        """测试MACD检测逻辑"""
        
        print("🔍 测试MACD形态检测逻辑")
        print("=" * 60)
        
        results = {
            'total_stocks_tested': 0,
            'stocks_with_data': 0,
            'macd_calculation_success': 0,
            'pattern_detection_results': {
                'GOLDEN_CROSS': [],
                'DEATH_CROSS': [],
                'MACD_ABOVE_ZERO_GOLDEN': [],
                'BEARISH_DIVERGENCE': []
            },
            'debug_info': [],
            'issues_found': []
        }
        
        if not self.use_real_data:
            results['issues_found'].append("无法使用真实数据，请检查数据服务")
            return results
        
        try:
            # 获取股票列表
            stock_codes = self.stock_data_service.get_stock_list(limit=max_stocks)
            results['total_stocks_tested'] = len(stock_codes)
            
            print(f"📊 获取到{len(stock_codes)}支股票，开始测试...")
            
            for i, stock_code in enumerate(stock_codes):
                if i % 50 == 0:
                    print(f"  进度: {i+1}/{len(stock_codes)} ({(i+1)/len(stock_codes)*100:.1f}%)")
                    # 显示当前找到的形态数量
                    current_counts = {k: len(v) for k, v in results['pattern_detection_results'].items()}
                    print(f"    当前找到: 金叉{current_counts['GOLDEN_CROSS']}, 死叉{current_counts['DEATH_CROSS']}, 零轴上金叉{current_counts['MACD_ABOVE_ZERO_GOLDEN']}, 看跌背离{current_counts['BEARISH_DIVERGENCE']}")
                
                try:
                    # 获取股票数据
                    df = self.stock_data_service.get_stock_data(stock_code, days=120)
                    
                    if df is None or len(df) < 60:
                        continue
                    
                    results['stocks_with_data'] += 1
                    
                    # 计算MACD
                    macd_result = self.macd_indicator.calculate(df)
                    
                    if macd_result is None or macd_result.empty:
                        continue
                    
                    results['macd_calculation_success'] += 1
                    
                    # 检测各种形态
                    detection_results = self._detect_all_patterns(stock_code, df, macd_result)
                    
                    # 记录检测结果
                    for pattern_id, detected in detection_results.items():
                        if detected:
                            results['pattern_detection_results'][pattern_id].append({
                                'stock_code': stock_code,
                                'detection_info': detected
                            })
                    
                    # 如果每个形态都找到了足够的股票，可以提前结束
                    if all(len(stocks) >= 10 for stocks in results['pattern_detection_results'].values()):
                        print(f"  ✅ 所有形态都找到了足够的股票（每个形态≥10支），提前结束测试")
                        break
                        
                except Exception as e:
                    results['debug_info'].append(f"{stock_code}: 处理失败 - {str(e)}")
                    continue
        
        except Exception as e:
            results['issues_found'].append(f"测试过程异常: {str(e)}")
        
        return results
    
    def _detect_all_patterns(self, stock_code: str, price_df: pd.DataFrame, macd_df: pd.DataFrame) -> Dict[str, Any]:
        """检测所有MACD形态"""
        
        results = {
            'GOLDEN_CROSS': None,
            'DEATH_CROSS': None,
            'MACD_ABOVE_ZERO_GOLDEN': None,
            'BEARISH_DIVERGENCE': None
        }
        
        try:
            # 确保数据有效
            if 'macd_line' not in macd_df.columns or 'signal_line' not in macd_df.columns:
                return results
            
            # 获取最近的数据（最后20天）
            recent_days = min(20, len(macd_df))
            recent_macd = macd_df.tail(recent_days).copy()
            recent_price = price_df.tail(recent_days).copy()
            
            if len(recent_macd) < 10:
                return results
            
            # 检测金叉
            golden_cross = self._detect_golden_cross(recent_macd)
            if golden_cross:
                results['GOLDEN_CROSS'] = golden_cross
            
            # 检测死叉
            death_cross = self._detect_death_cross(recent_macd)
            if death_cross:
                results['DEATH_CROSS'] = death_cross
            
            # 检测零轴上金叉
            above_zero_golden = self._detect_above_zero_golden(recent_macd)
            if above_zero_golden:
                results['MACD_ABOVE_ZERO_GOLDEN'] = above_zero_golden
            
            # 检测看跌背离
            bearish_divergence = self._detect_bearish_divergence(recent_price, recent_macd)
            if bearish_divergence:
                results['BEARISH_DIVERGENCE'] = bearish_divergence
        
        except Exception as e:
            pass
        
        return results
    
    def _detect_golden_cross(self, macd_df: pd.DataFrame) -> Optional[Dict]:
        """检测MACD金叉"""
        try:
            macd_line = macd_df['macd_line'].values
            signal_line = macd_df['signal_line'].values
            
            # 查找金叉点（MACD线上穿信号线）
            for i in range(1, len(macd_line)):
                if (macd_line[i-1] <= signal_line[i-1] and 
                    macd_line[i] > signal_line[i] and
                    abs(macd_line[i] - signal_line[i]) > 0.001):  # 避免噪音
                    
                    return {
                        'detection_date': macd_df.index[i],
                        'macd_value': macd_line[i],
                        'signal_value': signal_line[i],
                        'cross_strength': abs(macd_line[i] - signal_line[i]),
                        'pattern_type': 'GOLDEN_CROSS'
                    }
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_death_cross(self, macd_df: pd.DataFrame) -> Optional[Dict]:
        """检测MACD死叉"""
        try:
            macd_line = macd_df['macd_line'].values
            signal_line = macd_df['signal_line'].values
            
            # 查找死叉点（MACD线下穿信号线）
            for i in range(1, len(macd_line)):
                if (macd_line[i-1] >= signal_line[i-1] and 
                    macd_line[i] < signal_line[i] and
                    abs(macd_line[i] - signal_line[i]) > 0.001):  # 避免噪音
                    
                    return {
                        'detection_date': macd_df.index[i],
                        'macd_value': macd_line[i],
                        'signal_value': signal_line[i],
                        'cross_strength': abs(macd_line[i] - signal_line[i]),
                        'pattern_type': 'DEATH_CROSS'
                    }
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_above_zero_golden(self, macd_df: pd.DataFrame) -> Optional[Dict]:
        """检测零轴上方金叉"""
        try:
            macd_line = macd_df['macd_line'].values
            signal_line = macd_df['signal_line'].values
            
            # 查找零轴上方的金叉
            for i in range(1, len(macd_line)):
                if (macd_line[i] > 0 and signal_line[i] > 0 and  # 都在零轴上方
                    macd_line[i-1] <= signal_line[i-1] and 
                    macd_line[i] > signal_line[i] and
                    abs(macd_line[i] - signal_line[i]) > 0.001):
                    
                    return {
                        'detection_date': macd_df.index[i],
                        'macd_value': macd_line[i],
                        'signal_value': signal_line[i],
                        'cross_strength': abs(macd_line[i] - signal_line[i]),
                        'pattern_type': 'MACD_ABOVE_ZERO_GOLDEN'
                    }
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_bearish_divergence(self, price_df: pd.DataFrame, macd_df: pd.DataFrame) -> Optional[Dict]:
        """检测看跌背离"""
        try:
            if len(price_df) < 10 or len(macd_df) < 10:
                return None
            
            close_prices = price_df['close'].values
            macd_line = macd_df['macd_line'].values
            
            # 查找最近的价格高点和MACD高点
            recent_days = min(10, len(close_prices))
            recent_prices = close_prices[-recent_days:]
            recent_macd = macd_line[-recent_days:]
            
            # 找到价格和MACD的最高点位置
            price_high_idx = np.argmax(recent_prices)
            macd_high_idx = np.argmax(recent_macd)
            
            # 检查是否存在背离（价格新高但MACD未新高）
            if (price_high_idx >= recent_days - 3 and  # 价格高点在最近3天内
                macd_high_idx < recent_days - 3 and    # MACD高点不在最近3天内
                recent_prices[price_high_idx] > np.mean(recent_prices) * 1.02):  # 价格确实较高
                
                return {
                    'detection_date': macd_df.index[-recent_days + price_high_idx],
                    'price_high': recent_prices[price_high_idx],
                    'macd_at_price_high': recent_macd[price_high_idx],
                    'macd_high_value': recent_macd[macd_high_idx],
                    'divergence_strength': abs(price_high_idx - macd_high_idx),
                    'pattern_type': 'BEARISH_DIVERGENCE'
                }
            
            return None
            
        except Exception as e:
            return None
    
    def print_test_results(self, results: Dict[str, Any]):
        """打印测试结果"""
        print("\n" + "=" * 60)
        print("📊 MACD形态检测测试结果")
        print("=" * 60)
        
        print(f"📈 数据统计:")
        print(f"  测试股票总数: {results['total_stocks_tested']}")
        print(f"  有效数据股票: {results['stocks_with_data']}")
        print(f"  MACD计算成功: {results['macd_calculation_success']}")
        
        if results['macd_calculation_success'] > 0:
            success_rate = results['macd_calculation_success'] / results['stocks_with_data'] * 100
            print(f"  MACD计算成功率: {success_rate:.1f}%")
        
        print(f"\n🎯 形态检测结果:")
        pattern_names = {
            'GOLDEN_CROSS': 'MACD金叉',
            'DEATH_CROSS': 'MACD死叉', 
            'MACD_ABOVE_ZERO_GOLDEN': 'MACD零轴上金叉',
            'BEARISH_DIVERGENCE': 'MACD看跌背离'
        }
        
        total_patterns_found = 0
        for pattern_id, stocks in results['pattern_detection_results'].items():
            pattern_name = pattern_names.get(pattern_id, pattern_id)
            stock_count = len(stocks)
            total_patterns_found += stock_count
            
            status = "✅" if stock_count > 0 else "❌"
            print(f"  {status} {pattern_name}: {stock_count}支股票")
            
            # 显示前3个股票示例
            if stock_count > 0:
                for i, stock_info in enumerate(stocks[:3]):
                    stock_code = stock_info['stock_code']
                    detection_info = stock_info['detection_info']
                    print(f"    - {stock_code}: {detection_info.get('pattern_type', 'unknown')}")
        
        print(f"\n📊 总计找到: {total_patterns_found}个形态匹配")
        
        if results['issues_found']:
            print(f"\n⚠️ 发现的问题:")
            for issue in results['issues_found']:
                print(f"  - {issue}")
        
        if results['debug_info']:
            print(f"\n🔧 调试信息（前5条）:")
            for debug in results['debug_info'][:5]:
                print(f"  - {debug}")

def main():
    """主函数"""
    print("🔍 MACD形态检测逻辑测试")
    print("目标：验证在四千只股票中能否找到满足各种MACD形态的股票")
    
    # 创建检测器
    detector = ImprovedMacdPatternDetector()
    
    # 运行测试 - 扩大样本量到1000支股票
    results = detector.test_macd_detection_logic(max_stocks=1000)
    
    # 打印结果
    detector.print_test_results(results)
    
    # 给出建议
    total_found = sum(len(stocks) for stocks in results['pattern_detection_results'].values())
    
    if total_found > 0:
        print(f"\n🎉 检测逻辑正常工作！")
        print(f"✅ 在{results['macd_calculation_success']}支股票中找到了{total_found}个形态匹配")
        print(f"💡 建议：可以扩大测试范围到更多股票")
    else:
        print(f"\n⚠️ 检测逻辑需要调整")
        print(f"❌ 在{results['macd_calculation_success']}支股票中未找到形态匹配")
        print(f"💡 建议：检查MACD计算结果和形态检测条件")

if __name__ == "__main__":
    main()
