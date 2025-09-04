#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD指标阶段2形态识别验证

基于RSI项目成功经验，验证MACD指标关键形态识别功能
重点验证金叉、死叉、背离、零轴穿越、柱状图变化等核心形态
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from utils.technical_utils import calculate_macd_Utils
    from clickhouse_driver import Client
except ImportError as e:
    print(f"导入错误: {e}")

class MACDStage2PatternValidator:
    """MACD阶段2形态识别验证器"""
    
    def __init__(self):
        """初始化阶段2验证器"""
        self.validator_name = "MACD阶段2形态识别验证器"
        self.macd_indicator = MacdMacd()
        
        # 连接数据库
        self.client = Client(
            host='localhost',
            port=9000,
            database='stock',
            user='default',
            password='123456'
        )
        
        # 阶段2验证配置
        self.stage2_config = {
            'pattern_accuracy_target': 0.90,  # 90%形态识别准确率
            'golden_cross_target': 0.85,      # 85%金叉识别准确率
            'death_cross_target': 0.85,       # 85%死叉识别准确率
            'divergence_target': 0.80,        # 80%背离识别准确率
            'zero_cross_target': 0.85,        # 85%零轴穿越识别准确率
            'histogram_target': 0.80,         # 80%柱状图变化识别准确率
            'overall_score_target': 85.0      # 85分总体评分目标
        }
        
        print(f"✅ {self.validator_name}初始化完成")
        print(f"🎯 验证MACD关键形态识别功能")
    
    def get_real_market_data(self, stock_code: str = '000001', days: int = 200) -> Optional[pd.DataFrame]:
        """获取真实市场数据"""
        
        try:
            # 查询可用股票
            available_query = """
            SELECT code, COUNT(*) as data_count
            FROM stock_info
            WHERE level = '日线'
            AND date >= '2024-01-01'
            GROUP BY code
            HAVING data_count >= 200
            ORDER BY data_count DESC
            LIMIT 10
            """
            
            available_result = self.client.execute(available_query)
            
            if available_result:
                available_stocks = [row[0] for row in available_result]
                print(f"📊 可用股票: {available_stocks[:5]}...")
                
                # 使用第一个可用股票
                if available_stocks:
                    stock_code = available_stocks[0]
                    print(f"📈 使用股票: {stock_code}")
            
            # 查询股票数据
            query = f"""
            SELECT date, open, high, low, close, volume
            FROM stock_info
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '2024-01-01'
            AND date <= '2025-12-31'
            ORDER BY date ASC
            LIMIT {days}
            """
            
            result = self.client.execute(query)
            
            if result:
                df = pd.DataFrame(result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df_sorted = df.sort_values('date').reset_index(drop=True)
                print(f"📊 获取到{len(df_sorted)}条真实市场数据")
                return df_sorted
            else:
                print(f"❌ 股票{stock_code}无数据")
                return None
                
        except Exception as e:
            print(f"❌ 获取真实市场数据失败: {e}")
            return None
    
    def create_synthetic_pattern_data(self, pattern_type: str) -> pd.DataFrame:
        """创建合成形态数据"""
        
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        
        if pattern_type == 'golden_cross':
            # 创建金叉形态：快线从下向上穿越慢线
            base_price = 100
            # 前半段：快线在慢线下方
            trend1 = np.linspace(-5, -1, 50)
            # 后半段：快线穿越到慢线上方
            trend2 = np.linspace(-1, 3, 50)
            trend = np.concatenate([trend1, trend2])
            
        elif pattern_type == 'death_cross':
            # 创建死叉形态：快线从上向下穿越慢线
            base_price = 100
            # 前半段：快线在慢线上方
            trend1 = np.linspace(3, 1, 50)
            # 后半段：快线穿越到慢线下方
            trend2 = np.linspace(1, -3, 50)
            trend = np.concatenate([trend1, trend2])
            
        elif pattern_type == 'bullish_divergence':
            # 创建牛市背离：价格下跌但MACD上升
            base_price = 100
            # 价格下跌趋势
            price_trend = np.linspace(0, -10, 100)
            # 但在后半段有轻微反弹，形成背离
            price_trend[70:] += np.linspace(0, 2, 30)
            trend = price_trend
            
        elif pattern_type == 'bearish_divergence':
            # 创建熊市背离：价格上涨但MACD下降
            base_price = 100
            # 价格上涨趋势
            price_trend = np.linspace(0, 10, 100)
            # 但在后半段有轻微回调，形成背离
            price_trend[70:] -= np.linspace(0, 3, 30)
            trend = price_trend
            
        elif pattern_type == 'zero_cross_up':
            # 创建零轴向上穿越
            base_price = 100
            # 从负值区域穿越到正值区域
            trend = np.linspace(-3, 3, 100)
            
        elif pattern_type == 'zero_cross_down':
            # 创建零轴向下穿越
            base_price = 100
            # 从正值区域穿越到负值区域
            trend = np.linspace(3, -3, 100)
            
        else:
            # 默认：平稳趋势
            base_price = 100
            trend = np.zeros(100)
        
        # 添加随机噪声
        noise = np.random.normal(0, 0.5, 100)
        prices = base_price + trend + noise
        
        # 确保价格合理性
        prices = np.maximum(prices, 50)
        
        synthetic_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        return synthetic_data
    
    def test_golden_cross_detection(self) -> Dict[str, Any]:
        """测试金叉检测"""
        
        print(f"\n🔧 测试MACD金叉检测")
        print("=" * 60)
        
        golden_cross_result = {
            'test_type': 'GOLDEN_CROSS_DETECTION',
            'timestamp': datetime.now().isoformat(),
            'test_cases': [],
            'detection_accuracy': 0.0,
            'status': 'IN_PROGRESS'
        }
        
        try:
            # 测试用例1: 合成金叉数据
            print(f"📊 测试用例1: 合成金叉数据")
            
            synthetic_data = self.create_synthetic_pattern_data('golden_cross')
            
            # 获取MACD形态
            patterns_result = self.macd_indicator.get_patterns_Macd(synthetic_data)
            
            if patterns_result is not None and not patterns_result.empty:
                # 检查是否检测到金叉
                if 'GOLDEN_CROSS' in patterns_result.columns:
                    golden_cross_signals = patterns_result['GOLDEN_CROSS'].dropna()
                    detected_signals = golden_cross_signals[golden_cross_signals == True]
                    
                    detection_rate = len(detected_signals) / len(golden_cross_signals) if len(golden_cross_signals) > 0 else 0
                    
                    golden_cross_result['test_cases'].append({
                        'case': 'synthetic_golden_cross',
                        'total_signals': len(golden_cross_signals),
                        'detected_signals': len(detected_signals),
                        'detection_rate': detection_rate,
                        'status': 'PASSED' if detection_rate >= 0.3 else 'FAILED'  # 至少30%检测率
                    })
                    
                    print(f"  ✅ 检测到{len(detected_signals)}个金叉信号，检测率{detection_rate:.1%}")
                else:
                    golden_cross_result['test_cases'].append({
                        'case': 'synthetic_golden_cross',
                        'status': 'NO_PATTERN_COLUMN',
                        'detection_rate': 0.0
                    })
                    print(f"  ❌ 未找到GOLDEN_CROSS列")
            else:
                golden_cross_result['test_cases'].append({
                    'case': 'synthetic_golden_cross',
                    'status': 'NO_PATTERNS_RESULT',
                    'detection_rate': 0.0
                })
                print(f"  ❌ 无形态识别结果")
            
            # 测试用例2: 真实市场数据
            print(f"\n📊 测试用例2: 真实市场数据")
            
            real_data = self.get_real_market_data('000001', 150)
            
            if real_data is not None and len(real_data) >= 100:
                # 获取MACD形态
                real_patterns_result = self.macd_indicator.get_patterns_Macd(real_data)
                
                if real_patterns_result is not None and not real_patterns_result.empty:
                    if 'GOLDEN_CROSS' in real_patterns_result.columns:
                        real_golden_cross = real_patterns_result['GOLDEN_CROSS'].dropna()
                        real_detected = real_golden_cross[real_golden_cross == True]
                        
                        # 验证金叉的技术正确性
                        macd_data = self.macd_indicator._calculate_macd(real_data)
                        
                        if macd_data is not None and 'macd_line' in macd_data.columns and 'macd_signal' in macd_data.columns:
                            dif = macd_data['macd_line']
                            dea = macd_data['macd_signal']
                            
                            # 手动验证金叉：DIF从下方穿越DEA
                            manual_golden_cross = []
                            for i in range(1, len(dif)):
                                if (dif.iloc[i-1] <= dea.iloc[i-1] and 
                                    dif.iloc[i] > dea.iloc[i] and 
                                    not pd.isna(dif.iloc[i]) and not pd.isna(dea.iloc[i])):
                                    manual_golden_cross.append(i)
                            
                            # 计算准确率
                            if len(manual_golden_cross) > 0:
                                # 检查系统检测的金叉是否与手动验证一致
                                accuracy_count = 0
                                for idx in real_detected.index:
                                    if idx in manual_golden_cross or any(abs(idx - mgc) <= 2 for mgc in manual_golden_cross):
                                        accuracy_count += 1
                                
                                accuracy = accuracy_count / len(real_detected) if len(real_detected) > 0 else 0
                            else:
                                accuracy = 0
                            
                            golden_cross_result['test_cases'].append({
                                'case': 'real_market_data',
                                'detected_signals': len(real_detected),
                                'manual_golden_cross': len(manual_golden_cross),
                                'accuracy': accuracy,
                                'status': 'PASSED' if accuracy >= 0.7 else 'FAILED'
                            })
                            
                            print(f"  ✅ 系统检测{len(real_detected)}个，手动验证{len(manual_golden_cross)}个，准确率{accuracy:.1%}")
                        else:
                            print(f"  ❌ 无法获取MACD计算数据进行验证")
                    else:
                        print(f"  ❌ 真实数据中未找到GOLDEN_CROSS列")
                else:
                    print(f"  ❌ 真实数据无形态识别结果")
            else:
                print(f"  ❌ 无法获取足够的真实市场数据")
            
            # 计算总体检测准确率
            accuracies = [case.get('accuracy', case.get('detection_rate', 0)) for case in golden_cross_result['test_cases']]
            golden_cross_result['detection_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else 0
            
            if golden_cross_result['detection_accuracy'] >= self.stage2_config['golden_cross_target']:
                golden_cross_result['status'] = 'PASSED'
                print(f"\n✅ 金叉检测测试通过: {golden_cross_result['detection_accuracy']:.1%}")
            else:
                golden_cross_result['status'] = 'FAILED'
                print(f"\n❌ 金叉检测测试失败: {golden_cross_result['detection_accuracy']:.1%}")
        
        except Exception as e:
            golden_cross_result['status'] = 'ERROR'
            golden_cross_result['error'] = str(e)
            print(f"❌ 金叉检测测试异常: {e}")
        
        return golden_cross_result
    
    def test_death_cross_detection(self) -> Dict[str, Any]:
        """测试死叉检测"""
        
        print(f"\n🔧 测试MACD死叉检测")
        print("=" * 60)
        
        death_cross_result = {
            'test_type': 'DEATH_CROSS_DETECTION',
            'timestamp': datetime.now().isoformat(),
            'test_cases': [],
            'detection_accuracy': 0.0,
            'status': 'IN_PROGRESS'
        }
        
        try:
            # 测试用例1: 合成死叉数据
            print(f"📊 测试用例1: 合成死叉数据")
            
            synthetic_data = self.create_synthetic_pattern_data('death_cross')
            
            # 获取MACD形态
            patterns_result = self.macd_indicator.get_patterns_Macd(synthetic_data)
            
            if patterns_result is not None and not patterns_result.empty:
                # 检查是否检测到死叉
                if 'DEATH_CROSS' in patterns_result.columns:
                    death_cross_signals = patterns_result['DEATH_CROSS'].dropna()
                    detected_signals = death_cross_signals[death_cross_signals == True]
                    
                    detection_rate = len(detected_signals) / len(death_cross_signals) if len(death_cross_signals) > 0 else 0
                    
                    death_cross_result['test_cases'].append({
                        'case': 'synthetic_death_cross',
                        'total_signals': len(death_cross_signals),
                        'detected_signals': len(detected_signals),
                        'detection_rate': detection_rate,
                        'status': 'PASSED' if detection_rate >= 0.3 else 'FAILED'
                    })
                    
                    print(f"  ✅ 检测到{len(detected_signals)}个死叉信号，检测率{detection_rate:.1%}")
                else:
                    death_cross_result['test_cases'].append({
                        'case': 'synthetic_death_cross',
                        'status': 'NO_PATTERN_COLUMN',
                        'detection_rate': 0.0
                    })
                    print(f"  ❌ 未找到DEATH_CROSS列")
            else:
                print(f"  ❌ 无形态识别结果")
            
            # 测试用例2: 真实市场数据验证
            print(f"\n📊 测试用例2: 真实市场数据验证")
            
            real_data = self.get_real_market_data('000001', 150)
            
            if real_data is not None and len(real_data) >= 100:
                real_patterns_result = self.macd_indicator.get_patterns_Macd(real_data)
                
                if real_patterns_result is not None and not real_patterns_result.empty:
                    if 'DEATH_CROSS' in real_patterns_result.columns:
                        real_death_cross = real_patterns_result['DEATH_CROSS'].dropna()
                        real_detected = real_death_cross[real_death_cross == True]
                        
                        # 手动验证死叉
                        macd_data = self.macd_indicator._calculate_macd(real_data)
                        
                        if macd_data is not None and 'macd_line' in macd_data.columns and 'macd_signal' in macd_data.columns:
                            dif = macd_data['macd_line']
                            dea = macd_data['macd_signal']
                            
                            # 手动验证死叉：DIF从上方穿越DEA
                            manual_death_cross = []
                            for i in range(1, len(dif)):
                                if (dif.iloc[i-1] >= dea.iloc[i-1] and 
                                    dif.iloc[i] < dea.iloc[i] and 
                                    not pd.isna(dif.iloc[i]) and not pd.isna(dea.iloc[i])):
                                    manual_death_cross.append(i)
                            
                            # 计算准确率
                            if len(manual_death_cross) > 0:
                                accuracy_count = 0
                                for idx in real_detected.index:
                                    if idx in manual_death_cross or any(abs(idx - mdc) <= 2 for mdc in manual_death_cross):
                                        accuracy_count += 1
                                
                                accuracy = accuracy_count / len(real_detected) if len(real_detected) > 0 else 0
                            else:
                                accuracy = 0
                            
                            death_cross_result['test_cases'].append({
                                'case': 'real_market_data',
                                'detected_signals': len(real_detected),
                                'manual_death_cross': len(manual_death_cross),
                                'accuracy': accuracy,
                                'status': 'PASSED' if accuracy >= 0.7 else 'FAILED'
                            })
                            
                            print(f"  ✅ 系统检测{len(real_detected)}个，手动验证{len(manual_death_cross)}个，准确率{accuracy:.1%}")
                        else:
                            print(f"  ❌ 无法获取MACD计算数据进行验证")
                    else:
                        print(f"  ❌ 真实数据中未找到DEATH_CROSS列")
                else:
                    print(f"  ❌ 真实数据无形态识别结果")
            else:
                print(f"  ❌ 无法获取足够的真实市场数据")
            
            # 计算总体检测准确率
            accuracies = [case.get('accuracy', case.get('detection_rate', 0)) for case in death_cross_result['test_cases']]
            death_cross_result['detection_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else 0
            
            if death_cross_result['detection_accuracy'] >= self.stage2_config['death_cross_target']:
                death_cross_result['status'] = 'PASSED'
                print(f"\n✅ 死叉检测测试通过: {death_cross_result['detection_accuracy']:.1%}")
            else:
                death_cross_result['status'] = 'FAILED'
                print(f"\n❌ 死叉检测测试失败: {death_cross_result['detection_accuracy']:.1%}")
        
        except Exception as e:
            death_cross_result['status'] = 'ERROR'
            death_cross_result['error'] = str(e)
            print(f"❌ 死叉检测测试异常: {e}")
        
        return death_cross_result

    def test_divergence_detection(self) -> Dict[str, Any]:
        """测试背离检测"""

        print(f"\n🔧 测试MACD背离检测")
        print("=" * 60)

        divergence_result = {
            'test_type': 'DIVERGENCE_DETECTION',
            'timestamp': datetime.now().isoformat(),
            'test_cases': [],
            'detection_accuracy': 0.0,
            'status': 'IN_PROGRESS'
        }

        try:
            # 测试牛市背离
            print(f"📊 测试用例1: 牛市背离检测")

            bullish_data = self.create_synthetic_pattern_data('bullish_divergence')
            patterns_result = self.macd_indicator.get_patterns_Macd(bullish_data)

            if patterns_result is not None and not patterns_result.empty:
                # 检查背离相关列
                divergence_columns = [col for col in patterns_result.columns if 'DIVERGENCE' in col]

                if divergence_columns:
                    detected_count = 0
                    for col in divergence_columns:
                        signals = patterns_result[col].dropna()
                        detected = signals[signals == True]
                        detected_count += len(detected)

                    detection_rate = min(detected_count / 10, 1.0)  # 最多期望10个信号

                    divergence_result['test_cases'].append({
                        'case': 'bullish_divergence',
                        'detected_signals': detected_count,
                        'detection_rate': detection_rate,
                        'status': 'PASSED' if detection_rate >= 0.2 else 'FAILED'
                    })

                    print(f"  ✅ 检测到{detected_count}个牛市背离信号，检测率{detection_rate:.1%}")
                else:
                    divergence_result['test_cases'].append({
                        'case': 'bullish_divergence',
                        'status': 'NO_DIVERGENCE_COLUMNS',
                        'detection_rate': 0.0
                    })
                    print(f"  ❌ 未找到背离相关列")

            # 测试熊市背离
            print(f"\n📊 测试用例2: 熊市背离检测")

            bearish_data = self.create_synthetic_pattern_data('bearish_divergence')
            patterns_result2 = self.macd_indicator.get_patterns_Macd(bearish_data)

            if patterns_result2 is not None and not patterns_result2.empty:
                divergence_columns = [col for col in patterns_result2.columns if 'DIVERGENCE' in col]

                if divergence_columns:
                    detected_count = 0
                    for col in divergence_columns:
                        signals = patterns_result2[col].dropna()
                        detected = signals[signals == True]
                        detected_count += len(detected)

                    detection_rate = min(detected_count / 10, 1.0)

                    divergence_result['test_cases'].append({
                        'case': 'bearish_divergence',
                        'detected_signals': detected_count,
                        'detection_rate': detection_rate,
                        'status': 'PASSED' if detection_rate >= 0.2 else 'FAILED'
                    })

                    print(f"  ✅ 检测到{detected_count}个熊市背离信号，检测率{detection_rate:.1%}")
                else:
                    print(f"  ❌ 未找到背离相关列")

            # 计算总体背离检测准确率
            detection_rates = [case.get('detection_rate', 0) for case in divergence_result['test_cases']]
            divergence_result['detection_accuracy'] = sum(detection_rates) / len(detection_rates) if detection_rates else 0

            if divergence_result['detection_accuracy'] >= self.stage2_config['divergence_target']:
                divergence_result['status'] = 'PASSED'
                print(f"\n✅ 背离检测测试通过: {divergence_result['detection_accuracy']:.1%}")
            else:
                divergence_result['status'] = 'FAILED'
                print(f"\n❌ 背离检测测试失败: {divergence_result['detection_accuracy']:.1%}")

        except Exception as e:
            divergence_result['status'] = 'ERROR'
            divergence_result['error'] = str(e)
            print(f"❌ 背离检测测试异常: {e}")

        return divergence_result

    def test_zero_line_cross_detection(self) -> Dict[str, Any]:
        """测试零轴穿越检测"""

        print(f"\n🔧 测试MACD零轴穿越检测")
        print("=" * 60)

        zero_cross_result = {
            'test_type': 'ZERO_LINE_CROSS_DETECTION',
            'timestamp': datetime.now().isoformat(),
            'test_cases': [],
            'detection_accuracy': 0.0,
            'status': 'IN_PROGRESS'
        }

        try:
            # 测试向上穿越零轴
            print(f"📊 测试用例1: 向上穿越零轴")

            up_cross_data = self.create_synthetic_pattern_data('zero_cross_up')
            patterns_result = self.macd_indicator.get_patterns_Macd(up_cross_data)

            if patterns_result is not None and not patterns_result.empty:
                # 检查零轴穿越相关列
                zero_columns = [col for col in patterns_result.columns if 'ZERO_CROSS' in col]

                if zero_columns:
                    detected_count = 0
                    for col in zero_columns:
                        signals = patterns_result[col].dropna()
                        detected = signals[signals == True]
                        detected_count += len(detected)

                    detection_rate = min(detected_count / 5, 1.0)  # 期望5个左右信号

                    zero_cross_result['test_cases'].append({
                        'case': 'zero_cross_up',
                        'detected_signals': detected_count,
                        'detection_rate': detection_rate,
                        'status': 'PASSED' if detection_rate >= 0.4 else 'FAILED'
                    })

                    print(f"  ✅ 检测到{detected_count}个向上穿越信号，检测率{detection_rate:.1%}")
                else:
                    zero_cross_result['test_cases'].append({
                        'case': 'zero_cross_up',
                        'status': 'NO_ZERO_COLUMNS',
                        'detection_rate': 0.0
                    })
                    print(f"  ❌ 未找到零轴相关列")

            # 测试向下穿越零轴
            print(f"\n📊 测试用例2: 向下穿越零轴")

            down_cross_data = self.create_synthetic_pattern_data('zero_cross_down')
            patterns_result2 = self.macd_indicator.get_patterns_Macd(down_cross_data)

            if patterns_result2 is not None and not patterns_result2.empty:
                zero_columns = [col for col in patterns_result2.columns if 'ZERO_CROSS' in col]

                if zero_columns:
                    detected_count = 0
                    for col in zero_columns:
                        signals = patterns_result2[col].dropna()
                        detected = signals[signals == True]
                        detected_count += len(detected)

                    detection_rate = min(detected_count / 5, 1.0)

                    zero_cross_result['test_cases'].append({
                        'case': 'zero_cross_down',
                        'detected_signals': detected_count,
                        'detection_rate': detection_rate,
                        'status': 'PASSED' if detection_rate >= 0.4 else 'FAILED'
                    })

                    print(f"  ✅ 检测到{detected_count}个向下穿越信号，检测率{detection_rate:.1%}")
                else:
                    print(f"  ❌ 未找到零轴相关列")

            # 计算总体零轴穿越检测准确率
            detection_rates = [case.get('detection_rate', 0) for case in zero_cross_result['test_cases']]
            zero_cross_result['detection_accuracy'] = sum(detection_rates) / len(detection_rates) if detection_rates else 0

            if zero_cross_result['detection_accuracy'] >= self.stage2_config['zero_cross_target']:
                zero_cross_result['status'] = 'PASSED'
                print(f"\n✅ 零轴穿越检测测试通过: {zero_cross_result['detection_accuracy']:.1%}")
            else:
                zero_cross_result['status'] = 'FAILED'
                print(f"\n❌ 零轴穿越检测测试失败: {zero_cross_result['detection_accuracy']:.1%}")

        except Exception as e:
            zero_cross_result['status'] = 'ERROR'
            zero_cross_result['error'] = str(e)
            print(f"❌ 零轴穿越检测测试异常: {e}")

        return zero_cross_result

    def test_histogram_pattern_detection(self) -> Dict[str, Any]:
        """测试柱状图形态检测"""

        print(f"\n🔧 测试MACD柱状图形态检测")
        print("=" * 60)

        histogram_result = {
            'test_type': 'HISTOGRAM_PATTERN_DETECTION',
            'timestamp': datetime.now().isoformat(),
            'test_cases': [],
            'detection_accuracy': 0.0,
            'status': 'IN_PROGRESS'
        }

        try:
            # 使用真实市场数据测试柱状图形态
            real_data = self.get_real_market_data('000001', 150)

            if real_data is not None and len(real_data) >= 100:
                # 获取MACD计算结果
                macd_data = self.macd_indicator._calculate_macd(real_data)

                if macd_data is not None and 'macd_histogram' in macd_data.columns:
                    histogram = macd_data['macd_histogram'].dropna()

                    if len(histogram) > 10:
                        # 分析柱状图变化模式
                        histogram_changes = []
                        for i in range(1, len(histogram)):
                            if not pd.isna(histogram.iloc[i]) and not pd.isna(histogram.iloc[i-1]):
                                if histogram.iloc[i] > histogram.iloc[i-1]:
                                    histogram_changes.append('increasing')
                                elif histogram.iloc[i] < histogram.iloc[i-1]:
                                    histogram_changes.append('decreasing')
                                else:
                                    histogram_changes.append('stable')

                        # 统计变化模式
                        increasing_count = histogram_changes.count('increasing')
                        decreasing_count = histogram_changes.count('decreasing')
                        stable_count = histogram_changes.count('stable')

                        total_changes = len(histogram_changes)

                        # 检查形态识别结果
                        patterns_result = self.macd_indicator.get_patterns_Macd(real_data)

                        if patterns_result is not None and not patterns_result.empty:
                            # 检查柱状图相关形态
                            histogram_columns = [col for col in patterns_result.columns if 'HISTOGRAM' in col]

                            detected_patterns = 0
                            for col in histogram_columns:
                                signals = patterns_result[col].dropna()
                                detected = signals[signals == True]
                                detected_patterns += len(detected)

                            # 计算检测准确率（基于变化的复杂性）
                            expected_patterns = max(increasing_count, decreasing_count) // 5  # 期望检测到主要变化的1/5
                            detection_rate = min(detected_patterns / max(expected_patterns, 1), 1.0)

                            histogram_result['test_cases'].append({
                                'case': 'real_market_histogram',
                                'total_changes': total_changes,
                                'increasing_count': increasing_count,
                                'decreasing_count': decreasing_count,
                                'detected_patterns': detected_patterns,
                                'detection_rate': detection_rate,
                                'status': 'PASSED' if detection_rate >= 0.3 else 'FAILED'
                            })

                            print(f"  ✅ 柱状图变化{total_changes}次，检测到{detected_patterns}个形态，检测率{detection_rate:.1%}")
                        else:
                            print(f"  ❌ 无形态识别结果")
                    else:
                        print(f"  ❌ 柱状图数据不足")
                else:
                    print(f"  ❌ 无法获取柱状图数据")
            else:
                print(f"  ❌ 无法获取真实市场数据")

            # 计算总体柱状图检测准确率
            detection_rates = [case.get('detection_rate', 0) for case in histogram_result['test_cases']]
            histogram_result['detection_accuracy'] = sum(detection_rates) / len(detection_rates) if detection_rates else 0

            if histogram_result['detection_accuracy'] >= self.stage2_config['histogram_target']:
                histogram_result['status'] = 'PASSED'
                print(f"\n✅ 柱状图形态检测测试通过: {histogram_result['detection_accuracy']:.1%}")
            else:
                histogram_result['status'] = 'FAILED'
                print(f"\n❌ 柱状图形态检测测试失败: {histogram_result['detection_accuracy']:.1%}")

        except Exception as e:
            histogram_result['status'] = 'ERROR'
            histogram_result['error'] = str(e)
            print(f"❌ 柱状图形态检测测试异常: {e}")

        return histogram_result

    def run_complete_stage2_validation(self) -> Dict[str, Any]:
        """运行完整的阶段2验证"""

        print(f"\n🎯 MACD指标阶段2形态识别验证")
        print("基于RSI项目成功经验，验证MACD关键形态识别功能")
        print("=" * 80)

        stage2_results = {
            'validation_type': 'MACD_STAGE2_PATTERN_VALIDATION',
            'validator': self.validator_name,
            'start_time': datetime.now().isoformat(),
            'stage2_config': self.stage2_config,
            'golden_cross_detection': {},
            'death_cross_detection': {},
            'divergence_detection': {},
            'zero_line_cross_detection': {},
            'histogram_pattern_detection': {},
            'overall_assessment': {},
            'final_status': 'IN_PROGRESS'
        }

        try:
            # 1. 金叉检测测试
            golden_cross_result = self.test_golden_cross_detection()
            stage2_results['golden_cross_detection'] = golden_cross_result

            # 2. 死叉检测测试
            death_cross_result = self.test_death_cross_detection()
            stage2_results['death_cross_detection'] = death_cross_result

            # 3. 背离检测测试
            divergence_result = self.test_divergence_detection()
            stage2_results['divergence_detection'] = divergence_result

            # 4. 零轴穿越检测测试
            zero_cross_result = self.test_zero_line_cross_detection()
            stage2_results['zero_line_cross_detection'] = zero_cross_result

            # 5. 柱状图形态检测测试
            histogram_result = self.test_histogram_pattern_detection()
            stage2_results['histogram_pattern_detection'] = histogram_result

            # 6. 总体评估
            overall_assessment = self._assess_stage2_results(
                golden_cross_result, death_cross_result, divergence_result,
                zero_cross_result, histogram_result
            )
            stage2_results['overall_assessment'] = overall_assessment
            stage2_results['final_status'] = overall_assessment['final_status']

            stage2_results['end_time'] = datetime.now().isoformat()

            print(f"\n🏆 MACD阶段2验证完成")
            print(f"最终状态: {stage2_results['final_status']}")
            print(f"总体评分: {overall_assessment.get('total_score', 0):.1f}/100")

        except Exception as e:
            stage2_results['final_status'] = 'ERROR'
            stage2_results['error'] = str(e)
            print(f"❌ 阶段2验证异常: {e}")

        # 保存结果
        self._save_stage2_results(stage2_results)

        return stage2_results

    def _assess_stage2_results(self, golden_cross_result: Dict, death_cross_result: Dict,
                              divergence_result: Dict, zero_cross_result: Dict,
                              histogram_result: Dict) -> Dict[str, Any]:
        """评估阶段2结果"""

        assessment = {
            'assessment_type': 'STAGE2_PATTERN_ASSESSMENT',
            'individual_scores': {},
            'total_score': 0.0,
            'pattern_accuracy': 0.0,
            'final_status': 'UNKNOWN'
        }

        # 评估各项测试
        tests = {
            'golden_cross_detection': golden_cross_result,
            'death_cross_detection': death_cross_result,
            'divergence_detection': divergence_result,
            'zero_line_cross_detection': zero_cross_result,
            'histogram_pattern_detection': histogram_result
        }

        total_score = 0.0
        total_accuracy = 0.0

        for test_name, test_result in tests.items():
            test_status = test_result.get('status', 'UNKNOWN')
            test_accuracy = test_result.get('detection_accuracy', 0.0)

            if test_status == 'PASSED':
                score = 90 + (test_accuracy * 10)  # 90-100分
            elif test_status == 'FAILED':
                score = 50 + (test_accuracy * 30)  # 50-80分
            else:
                score = 30  # ERROR状态

            assessment['individual_scores'][test_name] = {
                'status': test_status,
                'accuracy': test_accuracy,
                'score': score
            }

            total_score += score
            total_accuracy += test_accuracy

        assessment['total_score'] = total_score / len(tests)
        assessment['pattern_accuracy'] = total_accuracy / len(tests)

        # 确定最终状态
        if (assessment['total_score'] >= self.stage2_config['overall_score_target'] and
            assessment['pattern_accuracy'] >= self.stage2_config['pattern_accuracy_target']):
            assessment['final_status'] = 'PASSED'
        elif assessment['total_score'] >= 70:
            assessment['final_status'] = 'CONDITIONAL_PASS'
        else:
            assessment['final_status'] = 'FAILED'

        return assessment

    def _save_stage2_results(self, results: Dict[str, Any]):
        """保存阶段2结果"""

        results_dir = Path("validation/macd_validation_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"MACD阶段2验证结果_{timestamp}.json"

        # 转换numpy类型
        def convert_types(obj):
            if isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        converted_results = convert_types(results)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, ensure_ascii=False, indent=2)

        print(f"\n📄 阶段2验证结果已保存: {results_file}")

def main():
    """主函数"""
    print("🎯 MACD指标阶段2形态识别验证")
    print("基于RSI项目成功经验，验证MACD关键形态识别功能")

    # 创建阶段2验证器
    validator = MACDStage2PatternValidator()

    # 运行完整的阶段2验证
    results = validator.run_complete_stage2_validation()

    # 显示结果摘要
    print(f"\n📊 MACD阶段2验证结果摘要")
    print("=" * 80)

    if 'overall_assessment' in results:
        assessment = results['overall_assessment']
        print(f"总体评分: {assessment.get('total_score', 0):.1f}/100")
        print(f"形态准确率: {assessment.get('pattern_accuracy', 0):.1%}")
        print(f"最终状态: {assessment.get('final_status', 'UNKNOWN')}")

        print(f"\n📋 各项测试结果:")
        for test_name, test_score in assessment.get('individual_scores', {}).items():
            status_icon = "✅" if test_score['status'] == 'PASSED' else "⚠️" if 'CONDITIONAL' in test_score['status'] else "❌"
            print(f"  {status_icon} {test_name}: {test_score['status']} (准确率{test_score['accuracy']:.1%}, 评分{test_score['score']:.1f})")

if __name__ == "__main__":
    main()
