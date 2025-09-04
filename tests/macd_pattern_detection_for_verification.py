#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD技术形态检测系统 - 用于人工验证

目标：
- 检测MACD的4个核心技术形态
- 输出具体股票清单供人工验证
- 提供详细的技术指标数值
- 生成便于验证的报告格式
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class MacdPatternDetectionForVerification:
    """MACD技术形态检测系统 - 专为人工验证设计"""
    
    def __init__(self):
        """初始化检测系统"""
        self.macd_indicator = MacdMacd()
        self.stock_data_service = get_stock_data_service()
        
        # 检测参数
        self.target_date = "2025-05-12"
        self.max_stocks_to_check = 100
        self.patterns_to_find = 10  # 每个形态找10支股票
        
        # MACD技术形态定义
        self.macd_patterns = {
            'GOLDEN_CROSS': {
                'name': 'MACD金叉',
                'description': 'MACD线上穿信号线，买入信号',
                'detection_method': 'cross_detection',
                'stocks': []
            },
            'DEATH_CROSS': {
                'name': 'MACD死叉', 
                'description': 'MACD线下穿信号线，卖出信号',
                'detection_method': 'cross_detection',
                'stocks': []
            },
            'ABOVE_ZERO_GOLDEN': {
                'name': 'MACD零轴上金叉',
                'description': 'MACD线在零轴上方形成金叉，强买入信号',
                'detection_method': 'above_zero_cross',
                'stocks': []
            },
            'BEARISH_DIVERGENCE': {
                'name': 'MACD看跌背离',
                'description': '价格创新高但MACD未创新高，顶部信号',
                'detection_method': 'divergence_detection',
                'stocks': []
            }
        }
        
        print("✅ MACD技术形态检测系统初始化完成")
        print(f"📅 检测日期: {self.target_date}")
        print(f"🎯 目标: 每个形态找到{self.patterns_to_find}支股票")
    
    def detect_macd_patterns_for_stock(self, stock_code: str) -> Dict[str, Any]:
        """检测单支股票的MACD技术形态"""
        
        patterns_found = {
            'GOLDEN_CROSS': None,
            'DEATH_CROSS': None,
            'ABOVE_ZERO_GOLDEN': None,
            'BEARISH_DIVERGENCE': None
        }
        
        try:
            # 获取股票数据
            df = self.stock_data_service.get_stock_data(stock_code, days=120)
            
            if df is None or len(df) < 60:
                return patterns_found
            
            # 计算MACD
            macd_result = self.macd_indicator.calculate(df)
            
            if macd_result is None or macd_result.empty:
                return patterns_found
            
            # 查找目标日期附近的数据
            target_date_obj = pd.to_datetime(self.target_date).date()
            
            # 获取最近30天的数据进行形态检测
            recent_days = min(30, len(macd_result))
            recent_macd = macd_result.tail(recent_days)
            recent_price = df.tail(recent_days)
            
            if len(recent_macd) < 5:
                return patterns_found
            
            # 检测各种形态
            patterns_found['GOLDEN_CROSS'] = self._detect_golden_cross(
                stock_code, recent_macd, recent_price
            )
            
            patterns_found['DEATH_CROSS'] = self._detect_death_cross(
                stock_code, recent_macd, recent_price
            )
            
            patterns_found['ABOVE_ZERO_GOLDEN'] = self._detect_above_zero_golden(
                stock_code, recent_macd, recent_price
            )
            
            patterns_found['BEARISH_DIVERGENCE'] = self._detect_bearish_divergence(
                stock_code, recent_macd, recent_price
            )
            
        except Exception as e:
            pass
        
        return patterns_found
    
    def _detect_golden_cross(self, stock_code: str, macd_df: pd.DataFrame, price_df: pd.DataFrame) -> Optional[Dict]:
        """检测MACD金叉"""
        
        macd_values = macd_df['macd_line'].values
        signal_values = macd_df['macd_signal'].values
        
        # 查找金叉点
        for i in range(1, len(macd_values)):
            if (macd_values[i-1] <= signal_values[i-1] and 
                macd_values[i] > signal_values[i] and
                abs(macd_values[i] - signal_values[i]) > 0.001):  # 确保有明显的穿越
                
                detection_date = macd_df.index[i]
                price_at_cross = price_df.iloc[i]['close'] if i < len(price_df) else 0
                
                return {
                    'stock_code': stock_code,
                    'pattern_type': 'GOLDEN_CROSS',
                    'detection_date': str(detection_date.date()) if hasattr(detection_date, 'date') else str(detection_date),
                    'macd_line': float(macd_values[i]),
                    'signal_line': float(signal_values[i]),
                    'macd_histogram': float(macd_values[i] - signal_values[i]) * 2,
                    'cross_strength': float(abs(macd_values[i] - signal_values[i])),
                    'price_at_cross': float(price_at_cross),
                    'verification_points': [
                        f"MACD线({macd_values[i]:.6f}) > 信号线({signal_values[i]:.6f})",
                        f"前一日MACD线({macd_values[i-1]:.6f}) <= 信号线({signal_values[i-1]:.6f})",
                        f"穿越强度: {abs(macd_values[i] - signal_values[i]):.6f}",
                        f"股价: {price_at_cross:.2f}"
                    ]
                }
        
        return None
    
    def _detect_death_cross(self, stock_code: str, macd_df: pd.DataFrame, price_df: pd.DataFrame) -> Optional[Dict]:
        """检测MACD死叉"""
        
        macd_values = macd_df['macd_line'].values
        signal_values = macd_df['macd_signal'].values
        
        # 查找死叉点
        for i in range(1, len(macd_values)):
            if (macd_values[i-1] >= signal_values[i-1] and 
                macd_values[i] < signal_values[i] and
                abs(macd_values[i] - signal_values[i]) > 0.001):
                
                detection_date = macd_df.index[i]
                price_at_cross = price_df.iloc[i]['close'] if i < len(price_df) else 0
                
                return {
                    'stock_code': stock_code,
                    'pattern_type': 'DEATH_CROSS',
                    'detection_date': str(detection_date.date()) if hasattr(detection_date, 'date') else str(detection_date),
                    'macd_line': float(macd_values[i]),
                    'signal_line': float(signal_values[i]),
                    'macd_histogram': float(macd_values[i] - signal_values[i]) * 2,
                    'cross_strength': float(abs(macd_values[i] - signal_values[i])),
                    'price_at_cross': float(price_at_cross),
                    'verification_points': [
                        f"MACD线({macd_values[i]:.6f}) < 信号线({signal_values[i]:.6f})",
                        f"前一日MACD线({macd_values[i-1]:.6f}) >= 信号线({signal_values[i-1]:.6f})",
                        f"穿越强度: {abs(macd_values[i] - signal_values[i]):.6f}",
                        f"股价: {price_at_cross:.2f}"
                    ]
                }
        
        return None
    
    def _detect_above_zero_golden(self, stock_code: str, macd_df: pd.DataFrame, price_df: pd.DataFrame) -> Optional[Dict]:
        """检测MACD零轴上金叉"""
        
        macd_values = macd_df['macd_line'].values
        signal_values = macd_df['macd_signal'].values
        
        # 查找零轴上方的金叉
        for i in range(1, len(macd_values)):
            if (macd_values[i-1] <= signal_values[i-1] and 
                macd_values[i] > signal_values[i] and
                macd_values[i] > 0 and signal_values[i] > 0 and  # 都在零轴上方
                abs(macd_values[i] - signal_values[i]) > 0.001):
                
                detection_date = macd_df.index[i]
                price_at_cross = price_df.iloc[i]['close'] if i < len(price_df) else 0
                
                return {
                    'stock_code': stock_code,
                    'pattern_type': 'ABOVE_ZERO_GOLDEN',
                    'detection_date': str(detection_date.date()) if hasattr(detection_date, 'date') else str(detection_date),
                    'macd_line': float(macd_values[i]),
                    'signal_line': float(signal_values[i]),
                    'macd_histogram': float(macd_values[i] - signal_values[i]) * 2,
                    'cross_strength': float(abs(macd_values[i] - signal_values[i])),
                    'price_at_cross': float(price_at_cross),
                    'verification_points': [
                        f"MACD线({macd_values[i]:.6f}) > 0 (零轴上方)",
                        f"信号线({signal_values[i]:.6f}) > 0 (零轴上方)",
                        f"MACD线 > 信号线 (金叉)",
                        f"前一日MACD线({macd_values[i-1]:.6f}) <= 信号线({signal_values[i-1]:.6f})",
                        f"股价: {price_at_cross:.2f}"
                    ]
                }
        
        return None
    
    def _detect_bearish_divergence(self, stock_code: str, macd_df: pd.DataFrame, price_df: pd.DataFrame) -> Optional[Dict]:
        """检测MACD看跌背离"""
        
        if len(price_df) < 15:
            return None
        
        close_prices = price_df['close'].values
        macd_values = macd_df['macd_line'].values
        
        # 查找价格高点和MACD背离
        for i in range(10, len(close_prices)):
            # 检查是否是价格高点
            if (i >= 5 and i < len(close_prices) - 2 and
                close_prices[i] == max(close_prices[i-5:i+3]) and  # 局部高点
                close_prices[i] > close_prices[i-10] * 1.02):  # 比10天前高2%以上
                
                # 检查MACD是否创新高
                if i < len(macd_values):
                    macd_at_high = macd_values[i]
                    max_macd_before = max(macd_values[max(0, i-10):i]) if i >= 10 else max(macd_values[:i])
                    
                    # 如果MACD明显低于之前的高点，形成背离
                    if macd_at_high < max_macd_before * 0.85:
                        detection_date = macd_df.index[i] if i < len(macd_df) else macd_df.index[-1]
                        
                        return {
                            'stock_code': stock_code,
                            'pattern_type': 'BEARISH_DIVERGENCE',
                            'detection_date': str(detection_date.date()) if hasattr(detection_date, 'date') else str(detection_date),
                            'price_high': float(close_prices[i]),
                            'macd_at_high': float(macd_at_high),
                            'max_macd_before': float(max_macd_before),
                            'divergence_ratio': float(macd_at_high / max_macd_before) if max_macd_before != 0 else 0,
                            'price_at_cross': float(close_prices[i]),
                            'verification_points': [
                                f"价格创高点: {close_prices[i]:.2f}",
                                f"当前MACD: {macd_at_high:.6f}",
                                f"之前MACD最高: {max_macd_before:.6f}",
                                f"背离程度: {(1 - macd_at_high/max_macd_before)*100:.1f}%",
                                f"形成看跌背离信号"
                            ]
                        }
        
        return None
    
    def run_pattern_detection(self) -> Dict[str, Any]:
        """运行形态检测"""
        
        print("\n🎯 开始MACD技术形态检测")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # 获取股票列表
            stock_codes = self.stock_data_service.get_stock_list(limit=self.max_stocks_to_check)
            
            if not stock_codes:
                return {'error': '无法获取股票列表'}
            
            print(f"📋 获取到{len(stock_codes)}支股票，开始检测MACD技术形态...")
            
            stocks_processed = 0
            
            for stock_code in stock_codes:
                stocks_processed += 1
                
                if stocks_processed % 10 == 0:
                    print(f"  📊 已处理{stocks_processed}/{len(stock_codes)}支股票...")
                
                # 检测当前股票的形态
                patterns = self.detect_macd_patterns_for_stock(stock_code)
                
                # 记录检测到的形态
                for pattern_id, pattern_data in patterns.items():
                    if (pattern_data and 
                        len(self.macd_patterns[pattern_id]['stocks']) < self.patterns_to_find):
                        self.macd_patterns[pattern_id]['stocks'].append(pattern_data)
                
                # 检查是否所有形态都找到了足够的股票
                all_patterns_found = all(
                    len(pattern_info['stocks']) >= self.patterns_to_find
                    for pattern_info in self.macd_patterns.values()
                )
                
                if all_patterns_found:
                    print(f"✅ 所有形态都找到了{self.patterns_to_find}支股票，检测完成")
                    break
            
            end_time = time.time()
            
            # 生成检测结果
            detection_results = {
                'detection_timestamp': datetime.now().isoformat(),
                'detection_date': self.target_date,
                'stocks_processed': stocks_processed,
                'processing_time': end_time - start_time,
                'patterns_detected': {},
                'summary': {}
            }
            
            # 整理形态检测结果
            for pattern_id, pattern_info in self.macd_patterns.items():
                detection_results['patterns_detected'][pattern_id] = {
                    'pattern_name': pattern_info['name'],
                    'pattern_description': pattern_info['description'],
                    'stocks_found': len(pattern_info['stocks']),
                    'target_stocks': self.patterns_to_find,
                    'stocks_list': pattern_info['stocks']
                }
            
            # 生成汇总
            total_stocks_found = sum(len(p['stocks']) for p in self.macd_patterns.values())
            patterns_with_stocks = sum(1 for p in self.macd_patterns.values() if len(p['stocks']) > 0)
            
            detection_results['summary'] = {
                'total_patterns': len(self.macd_patterns),
                'patterns_with_stocks': patterns_with_stocks,
                'total_stocks_found': total_stocks_found,
                'detection_success_rate': patterns_with_stocks / len(self.macd_patterns),
                'average_stocks_per_pattern': total_stocks_found / len(self.macd_patterns)
            }
            
            return detection_results
            
        except Exception as e:
            return {'error': f'检测过程异常: {str(e)}'}
    
    def print_verification_report(self, results: Dict[str, Any]):
        """打印人工验证报告"""
        
        if 'error' in results:
            print(f"❌ 检测失败: {results['error']}")
            return
        
        print(f"\n📊 MACD技术形态检测结果")
        print("=" * 80)
        print(f"🕐 检测时间: {results['detection_timestamp']}")
        print(f"📅 检测日期: {results['detection_date']}")
        print(f"📈 处理股票: {results['stocks_processed']}支")
        print(f"⏱️ 处理耗时: {results['processing_time']:.1f}秒")
        
        summary = results['summary']
        print(f"\n📋 检测汇总:")
        print(f"  形态总数: {summary['total_patterns']}")
        print(f"  有股票的形态: {summary['patterns_with_stocks']}")
        print(f"  找到股票总数: {summary['total_stocks_found']}")
        print(f"  检测成功率: {summary['detection_success_rate']:.1%}")
        
        print(f"\n🎯 各形态检测结果:")
        print("=" * 80)
        
        for pattern_id, pattern_data in results['patterns_detected'].items():
            pattern_name = pattern_data['pattern_name']
            pattern_desc = pattern_data['pattern_description']
            stocks_found = pattern_data['stocks_found']
            target_stocks = pattern_data['target_stocks']
            
            status = "✅" if stocks_found >= target_stocks else "⚠️" if stocks_found > 0 else "❌"
            
            print(f"\n{status} {pattern_name} ({stocks_found}/{target_stocks}支股票)")
            print(f"   📝 {pattern_desc}")
            
            if stocks_found > 0:
                print(f"   📈 符合条件的股票:")
                
                for i, stock_info in enumerate(pattern_data['stocks_list'][:5], 1):  # 只显示前5支
                    stock_code = stock_info['stock_code']
                    detection_date = stock_info['detection_date']
                    price = stock_info.get('price_at_cross', 0)
                    
                    print(f"     {i}. {stock_code} - {detection_date} - 股价:{price:.2f}")
                    
                    # 显示验证要点
                    if 'verification_points' in stock_info:
                        for point in stock_info['verification_points'][:3]:  # 只显示前3个要点
                            print(f"        • {point}")
                
                if stocks_found > 5:
                    print(f"     ... 还有{stocks_found-5}支股票")
            
            print()
        
        print("=" * 80)
        print("💡 人工验证建议:")
        print("1. 使用股票软件查看上述股票在检测日期的MACD指标")
        print("2. 验证MACD线、信号线的数值是否匹配")
        print("3. 确认技术形态是否符合描述")
        print("4. 检查形态出现的时间点是否准确")
        print("5. 评估形态的技术分析意义是否正确")

def main():
    """主函数"""
    print("🎯 MACD技术形态检测系统 - 专为人工验证设计")
    print("检测MACD的4个核心技术形态，输出具体股票清单供人工验证")
    
    # 创建检测系统
    detector = MacdPatternDetectionForVerification()
    
    # 运行形态检测
    results = detector.run_pattern_detection()
    
    # 打印验证报告
    detector.print_verification_report(results)
    
    # 保存结果
    results_dir = Path("validation/pattern_detection_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    json_file = results_dir / "MACD_技术形态检测结果.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 检测结果已保存到: {json_file}")

if __name__ == "__main__":
    main()
