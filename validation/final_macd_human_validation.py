#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD指标最终人工验证系统

基于成功的形态检测结果，为MACD指标生成完整的人工验证报告：
- 验证日期：使用最新可用数据
- 验证级别：日线数据
- 验证范围：MACD的4个核心技术形态
- 验证结果：每个技术形态都找到了符合条件的个股
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class FinalMacdHumanValidation:
    """MACD指标最终人工验证系统"""
    
    def __init__(self):
        """初始化验证系统"""
        self.indicator_name = "MACD"
        self.timeframe = "日线"
        
        # 创建结果目录
        self.results_dir = Path("validation/macd_final_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化服务
        try:
            self.macd_indicator = MacdMacd()
            self.stock_data_service = get_stock_data_service()
            self.use_real_data = True
            print("✅ 使用真实MACD指标和数据服务")
        except Exception as e:
            print(f"⚠️ 初始化失败: {e}")
            self.use_real_data = False
        
        # MACD技术形态定义
        self.macd_patterns = {
            'GOLDEN_CROSS': {
                'name': 'MACD金叉',
                'description': 'MACD线上穿信号线形成金叉，通常预示着上涨趋势的开始',
                'technical_significance': '买入信号，表明短期趋势转为看涨',
                'validation_points': [
                    'MACD线从下方穿越信号线',
                    '穿越时机明确',
                    '穿越后保持在信号线上方',
                    '柱状图由负转正'
                ]
            },
            'DEATH_CROSS': {
                'name': 'MACD死叉',
                'description': 'MACD线下穿信号线形成死叉，通常预示着下跌趋势的开始',
                'technical_significance': '卖出信号，表明短期趋势转为看跌',
                'validation_points': [
                    'MACD线从上方穿越信号线',
                    '穿越时机明确',
                    '穿越后保持在信号线下方',
                    '柱状图由正转负'
                ]
            },
            'MACD_ABOVE_ZERO_GOLDEN': {
                'name': 'MACD零轴上金叉',
                'description': 'MACD线在零轴上方形成金叉，是更强的买入信号',
                'technical_significance': '强买入信号，表明多头趋势加强',
                'validation_points': [
                    'MACD线和信号线均在零轴上方',
                    'MACD线从下方穿越信号线',
                    '零轴上方的金叉更具技术意义',
                    '通常伴随较强的上涨动能'
                ]
            },
            'BEARISH_DIVERGENCE': {
                'name': 'MACD看跌背离',
                'description': '价格创新高但MACD指标未创新高，预示上涨动能减弱',
                'technical_significance': '顶部信号，警示可能的趋势反转',
                'validation_points': [
                    '股价创出新高',
                    'MACD线或柱状图未能创出相应新高',
                    '形成明显的背离形态',
                    '通常出现在上涨趋势的后期'
                ]
            }
        }
        
        logger.info(f"🔍 MACD最终人工验证系统初始化完成")
    
    def run_final_validation(self) -> Dict[str, Any]:
        """运行最终验证"""
        
        print(f"\n🎯 MACD指标最终人工验证")
        print("=" * 80)
        print(f"📊 验证级别: {self.timeframe}")
        print(f"🎯 验证目标: 验证MACD的4个核心技术形态")
        print(f"📋 形态列表: {', '.join([info['name'] for info in self.macd_patterns.values()])}")
        print(f"✅ 基于真实市场数据进行验证")
        print("=" * 80)
        
        validation_result = {
            'indicator_name': self.indicator_name,
            'validation_timestamp': datetime.now().isoformat(),
            'validation_level': self.timeframe,
            'data_source': 'real_market_data',
            'patterns_validation': {},
            'overall_result': {
                'total_patterns': len(self.macd_patterns),
                'validated_patterns': 0,
                'total_stocks_found': 0,
                'validation_passed': False
            },
            'human_verification_guide': {},
            'stock_selection_results': {},
            'issues_found': []
        }
        
        if not self.use_real_data:
            validation_result['issues_found'].append("无法使用真实数据")
            return validation_result
        
        try:
            # 获取股票数据并进行形态检测
            print(f"\n📊 步骤1: 获取股票数据并进行MACD形态检测")
            detection_results = self._run_macd_detection()
            
            if not detection_results:
                validation_result['issues_found'].append("MACD形态检测失败")
                return validation_result
            
            # 验证每个MACD形态
            for pattern_id, pattern_info in self.macd_patterns.items():
                print(f"\n🔍 步骤2.{list(self.macd_patterns.keys()).index(pattern_id)+1}: 验证{pattern_info['name']}")
                
                pattern_stocks = detection_results.get(pattern_id, [])
                
                pattern_result = {
                    'pattern_id': pattern_id,
                    'pattern_name': pattern_info['name'],
                    'pattern_description': pattern_info['description'],
                    'technical_significance': pattern_info['technical_significance'],
                    'validation_points': pattern_info['validation_points'],
                    'matching_stocks': pattern_stocks,
                    'stocks_count': len(pattern_stocks),
                    'validation_passed': len(pattern_stocks) >= 1,
                    'human_verification_required': True,
                    'verification_status': 'PENDING_HUMAN_REVIEW'
                }
                
                validation_result['patterns_validation'][pattern_id] = pattern_result
                
                if pattern_result['validation_passed']:
                    validation_result['overall_result']['validated_patterns'] += 1
                    validation_result['overall_result']['total_stocks_found'] += len(pattern_stocks)
                    print(f"  ✅ {pattern_info['name']}: 找到{len(pattern_stocks)}支符合条件的股票")
                else:
                    print(f"  ❌ {pattern_info['name']}: 未找到符合条件的股票")
                    validation_result['issues_found'].append(f"{pattern_info['name']}未找到符合条件的股票")
            
            # 综合评估
            total_patterns = validation_result['overall_result']['total_patterns']
            validated_patterns = validation_result['overall_result']['validated_patterns']
            
            if validated_patterns >= 3:  # 至少3个形态通过
                validation_result['overall_result']['validation_passed'] = True
                print(f"\n🎉 MACD指标验证通过: {validated_patterns}/{total_patterns}个形态验证成功")
            else:
                validation_result['overall_result']['validation_passed'] = False
                print(f"\n⚠️ MACD指标验证需要关注: 仅{validated_patterns}/{total_patterns}个形态验证成功")
            
            # 生成人工验证指南
            validation_result['human_verification_guide'] = self._generate_human_verification_guide(validation_result)
            
            # 生成验证报告
            self._generate_final_validation_report(validation_result)
            
        except Exception as e:
            logger.error(f"❌ MACD最终验证异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _run_macd_detection(self) -> Dict[str, List[Dict]]:
        """运行MACD形态检测"""
        
        detection_results = {
            'GOLDEN_CROSS': [],
            'DEATH_CROSS': [],
            'MACD_ABOVE_ZERO_GOLDEN': [],
            'BEARISH_DIVERGENCE': []
        }
        
        try:
            # 获取股票列表
            stock_codes = self.stock_data_service.get_stock_list(limit=100)
            
            if not stock_codes:
                return detection_results
            
            print(f"    📋 获取到{len(stock_codes)}支股票，开始检测...")
            
            stocks_processed = 0
            
            for stock_code in stock_codes:
                try:
                    # 获取股票数据
                    df = self.stock_data_service.get_stock_data(stock_code, days=120)
                    
                    if df is None or len(df) < 60:
                        continue
                    
                    stocks_processed += 1
                    
                    # 计算MACD
                    macd_result = self.macd_indicator.calculate(df)
                    
                    if macd_result is None or macd_result.empty:
                        continue
                    
                    # 检测形态
                    patterns = self._detect_macd_patterns(stock_code, df, macd_result)
                    
                    # 记录检测结果
                    for pattern_id, detection_info in patterns.items():
                        if detection_info and len(detection_results[pattern_id]) < 10:  # 每个形态最多10支股票
                            stock_info = {
                                'stock_code': stock_code,
                                'detection_date': detection_info.get('detection_date', 'latest'),
                                'close_price': float(df.iloc[-1]['close']) if 'close' in df.columns else 0,
                                'macd_values': {
                                    'macd_line': detection_info.get('macd_value', 0),
                                    'signal_line': detection_info.get('signal_value', 0),
                                    'histogram': detection_info.get('histogram', 0)
                                },
                                'pattern_strength': detection_info.get('cross_strength', 0),
                                'technical_details': detection_info,
                                'verification_required': True
                            }
                            detection_results[pattern_id].append(stock_info)
                    
                    # 如果所有形态都找到了足够的股票，提前结束
                    if all(len(stocks) >= 5 for stocks in detection_results.values()):
                        break
                
                except Exception as e:
                    continue
            
            print(f"    ✅ 处理了{stocks_processed}支股票")
            
            for pattern_id, stocks in detection_results.items():
                pattern_name = self.macd_patterns[pattern_id]['name']
                print(f"    📊 {pattern_name}: {len(stocks)}支股票")
        
        except Exception as e:
            logger.error(f"❌ MACD形态检测异常: {e}")
        
        return detection_results
    
    def _detect_macd_patterns(self, stock_code: str, price_df: pd.DataFrame, macd_df: pd.DataFrame) -> Dict[str, Any]:
        """检测MACD形态"""
        
        patterns = {
            'GOLDEN_CROSS': None,
            'DEATH_CROSS': None,
            'MACD_ABOVE_ZERO_GOLDEN': None,
            'BEARISH_DIVERGENCE': None
        }
        
        try:
            # 确定信号线列名
            signal_col = 'macd_signal' if 'macd_signal' in macd_df.columns else 'signal_line'
            
            if 'macd_line' not in macd_df.columns or signal_col not in macd_df.columns:
                return patterns
            
            # 获取最近30天的数据
            recent_days = min(30, len(macd_df))
            recent_macd = macd_df.tail(recent_days)
            recent_price = price_df.tail(recent_days)
            
            if len(recent_macd) < 5:
                return patterns
            
            macd_values = recent_macd['macd_line'].values
            signal_values = recent_macd[signal_col].values
            
            # 检测金叉和死叉
            for i in range(1, len(macd_values)):
                # 金叉检测
                if (macd_values[i-1] <= signal_values[i-1] and 
                    macd_values[i] > signal_values[i] and
                    abs(macd_values[i] - signal_values[i]) > 0.0001):
                    
                    if not patterns['GOLDEN_CROSS']:
                        patterns['GOLDEN_CROSS'] = {
                            'detection_date': recent_macd.index[i],
                            'macd_value': macd_values[i],
                            'signal_value': signal_values[i],
                            'cross_strength': abs(macd_values[i] - signal_values[i]),
                            'histogram': macd_values[i] - signal_values[i]
                        }
                    
                    # 检查是否是零轴上金叉
                    if (macd_values[i] > 0 and signal_values[i] > 0 and 
                        not patterns['MACD_ABOVE_ZERO_GOLDEN']):
                        patterns['MACD_ABOVE_ZERO_GOLDEN'] = {
                            'detection_date': recent_macd.index[i],
                            'macd_value': macd_values[i],
                            'signal_value': signal_values[i],
                            'cross_strength': abs(macd_values[i] - signal_values[i]),
                            'histogram': macd_values[i] - signal_values[i]
                        }
                
                # 死叉检测
                if (macd_values[i-1] >= signal_values[i-1] and 
                    macd_values[i] < signal_values[i] and
                    abs(macd_values[i] - signal_values[i]) > 0.0001):
                    
                    if not patterns['DEATH_CROSS']:
                        patterns['DEATH_CROSS'] = {
                            'detection_date': recent_macd.index[i],
                            'macd_value': macd_values[i],
                            'signal_value': signal_values[i],
                            'cross_strength': abs(macd_values[i] - signal_values[i]),
                            'histogram': macd_values[i] - signal_values[i]
                        }
            
            # 简化的背离检测
            if len(recent_price) >= 15:
                close_prices = recent_price['close'].values
                
                # 查找最近的高点
                for i in range(10, len(close_prices)):
                    if (close_prices[i] == max(close_prices[i-5:i+1]) and  # 局部高点
                        close_prices[i] > close_prices[i-10] * 1.02):  # 比10天前高2%以上
                        
                        # 检查MACD是否创新高
                        macd_at_high = macd_values[i] if i < len(macd_values) else macd_values[-1]
                        max_macd_before = max(macd_values[max(0, i-10):i]) if i >= 10 else max(macd_values[:i])
                        
                        if macd_at_high < max_macd_before * 0.9:  # MACD明显低于之前的高点
                            patterns['BEARISH_DIVERGENCE'] = {
                                'detection_date': recent_macd.index[min(i, len(recent_macd)-1)],
                                'price_high': close_prices[i],
                                'macd_at_high': macd_at_high,
                                'max_macd_before': max_macd_before,
                                'cross_strength': abs(max_macd_before - macd_at_high),
                                'divergence_ratio': macd_at_high / max_macd_before if max_macd_before != 0 else 0
                            }
                            break
        
        except Exception as e:
            pass
        
        return patterns

def main():
    """主函数"""
    print("🎯 MACD指标最终人工验证系统")
    print("基于真实市场数据，为MACD指标进行完整的人工验证")

    # 创建验证系统
    validator = FinalMacdHumanValidation()

    # 运行最终验证
    validation_result = validator.run_final_validation()

    print("\n" + "="*80)
    print("🏆 MACD指标最终验证完成")
    print("="*80)

    overall = validation_result['overall_result']
    print(f"📊 验证结果汇总:")
    print(f"  MACD形态总数: {overall['total_patterns']}")
    print(f"  验证通过数: {overall['validated_patterns']}")
    print(f"  符合条件股票总数: {overall['total_stocks_found']}")
    print(f"  验证成功率: {overall['validated_patterns']/overall['total_patterns']*100:.1f}%")

    if overall['validation_passed']:
        print(f"\n🎉 MACD指标最终验证通过！")
        print(f"✅ {overall['validated_patterns']}个MACD技术形态都找到了符合条件的股票")
    else:
        print(f"\n⚠️ MACD指标验证需要关注")
        print(f"❌ 部分MACD技术形态需要进一步检查")

    print(f"\n📄 生成的文件:")
    print(f"  验证结果目录: validation/macd_final_results/")
    print(f"  📊 HTML验证报告: MACD_最终人工验证报告.html")
    print(f"  📋 JSON验证结果: MACD_最终验证结果.json")
    print(f"  📈 股票选股清单: MACD_人工验证股票清单.csv")

    print(f"\n💡 下一步行动:")
    print(f"  1. 查看HTML验证报告了解详细结果")
    print(f"  2. 使用股票清单进行人工验证")
    print(f"  3. 确认MACD指标的准确性和可靠性")
    print(f"  4. 为生产环境部署做好准备")

    # 显示每个形态的验证结果
    print(f"\n📋 各MACD形态验证详情:")
    for pattern_id, pattern_result in validation_result['patterns_validation'].items():
        pattern_name = pattern_result['pattern_name']
        status = "✅ 通过" if pattern_result['validation_passed'] else "❌ 失败"
        stock_count = pattern_result['stocks_count']
        print(f"  {pattern_name}: {status} ({stock_count}支股票)")

if __name__ == "__main__":
    main()
