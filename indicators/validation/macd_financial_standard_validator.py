#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACD指标专业金融标准验证器

本模块用于验证MACD指标的计算是否符合传统金融公式和行业标准，
包括DIF、DEA、MACD柱状图的计算准确性，以及技术形态识别能力。

验证标准：
1. DIF = EMA12 - EMA26 (快线减慢线)
2. DEA = EMA9(DIF) (DIF的9周期指数移动平均)
3. MACD = 2 * (DIF - DEA) (柱状图)
4. 金叉：DIF上穿DEA
5. 死叉：DIF下穿DEA
6. 零轴突破：DIF穿越零轴
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

from indicators.macd import MacdMacd
from utils.logger import get_logger

logger = get_logger(__name__)


class MACDFinancialStandardValidator:
    """MACD指标专业金融标准验证器"""
    
    def __init__(self):
        self.macd_indicator = MacdMacd()
        self.validation_results = {}
        
    def create_standard_test_data(self, length: int = 100) -> pd.DataFrame:
        """
        创建标准测试数据
        
        Args:
            length: 数据长度
            
        Returns:
            pd.DataFrame: 标准测试数据
        """
        # 创建模拟股价数据，包含趋势和波动
        np.random.seed(42)  # 确保结果可重现
        
        base_price = 100
        trend = np.linspace(0, 20, length)  # 上升趋势
        noise = np.random.normal(0, 2, length)  # 随机波动
        
        close_prices = base_price + trend + noise
        
        # 确保价格为正数
        close_prices = np.maximum(close_prices, 1.0)
        
        # 生成其他OHLC数据
        high_prices = close_prices * (1 + np.random.uniform(0, 0.02, length))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.02, length))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        volume = np.random.randint(1000, 10000, length)
        
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        return data
    
    def calculate_reference_macd(self, data: pd.DataFrame, 
                                fast_period: int = 12, 
                                slow_period: int = 26, 
                                signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        计算参考MACD值（使用标准金融公式）
        
        Args:
            data: 价格数据
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            Dict[str, pd.Series]: 包含DIF、DEA、MACD的字典
        """
        close = data['close']
        
        # 计算EMA
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        
        # 计算DIF (快线减慢线)
        dif = ema_fast - ema_slow
        
        # 计算DEA (DIF的信号线)
        dea = dif.ewm(span=signal_period).mean()
        
        # 计算MACD柱状图
        macd_histogram = 2 * (dif - dea)
        
        return {
            'DIF': dif,
            'DEA': dea,
            'MACD': macd_histogram
        }
    
    def validate_calculation_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证MACD计算准确性
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("开始验证MACD计算准确性...")
        
        # 获取我们的MACD计算结果
        our_result = self.macd_indicator.calculate(data)
        
        # 计算参考MACD值
        reference_result = self.calculate_reference_macd(data)
        
        # 验证结果
        validation_result = {
            'test_name': 'MACD计算准确性验证',
            'data_length': len(data),
            'accuracy_scores': {},
            'max_deviations': {},
            'mean_deviations': {},
            'passed': True,
            'details': []
        }
        
        # 检查每个指标的准确性
        for indicator_name in ['DIF', 'DEA', 'MACD']:
            if indicator_name in our_result.columns:
                our_values = our_result[indicator_name].dropna()
                ref_values = reference_result[indicator_name].dropna()
                
                # 确保长度一致
                min_length = min(len(our_values), len(ref_values))
                our_values = our_values.iloc[-min_length:]
                ref_values = ref_values.iloc[-min_length:]
                
                # 计算偏差
                deviations = np.abs(our_values - ref_values)
                max_deviation = deviations.max()
                mean_deviation = deviations.mean()
                
                # 计算准确性分数 (基于相对误差)
                relative_errors = deviations / (np.abs(ref_values) + 1e-8)
                accuracy_score = 1.0 - relative_errors.mean()
                
                validation_result['accuracy_scores'][indicator_name] = accuracy_score
                validation_result['max_deviations'][indicator_name] = max_deviation
                validation_result['mean_deviations'][indicator_name] = mean_deviation
                
                # 判断是否通过 (准确性 > 95%)
                if accuracy_score < 0.95:
                    validation_result['passed'] = False
                    validation_result['details'].append(
                        f"{indicator_name}准确性不足: {accuracy_score:.3f} < 0.95"
                    )
                else:
                    validation_result['details'].append(
                        f"{indicator_name}准确性验证通过: {accuracy_score:.3f}"
                    )
            else:
                validation_result['passed'] = False
                validation_result['details'].append(f"缺少{indicator_name}指标")
        
        return validation_result
    
    def validate_signal_generation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证MACD信号生成能力
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("开始验证MACD信号生成能力...")
        
        # 获取信号
        signals = self.macd_indicator.get_signals(data)
        
        # 获取MACD计算结果
        macd_result = self.macd_indicator.calculate(data)
        
        validation_result = {
            'test_name': 'MACD信号生成验证',
            'signal_count': {
                'buy_signals': signals['buy_signal'].sum() if 'buy_signal' in signals.columns else 0,
                'sell_signals': signals['sell_signal'].sum() if 'sell_signal' in signals.columns else 0
            },
            'signal_quality': {},
            'passed': True,
            'details': []
        }
        
        # 验证信号的合理性
        if 'buy_signal' in signals.columns and 'sell_signal' in signals.columns:
            # 检查信号不会同时出现
            simultaneous_signals = (signals['buy_signal'] & signals['sell_signal']).sum()
            if simultaneous_signals > 0:
                validation_result['passed'] = False
                validation_result['details'].append(f"发现{simultaneous_signals}个同时买卖信号")
            
            # 检查信号频率是否合理 (不应过于频繁)
            total_signals = validation_result['signal_count']['buy_signals'] + validation_result['signal_count']['sell_signals']
            signal_frequency = total_signals / len(data)
            
            if signal_frequency > 0.3:  # 信号频率不应超过30%
                validation_result['passed'] = False
                validation_result['details'].append(f"信号频率过高: {signal_frequency:.3f}")
            else:
                validation_result['details'].append(f"信号频率合理: {signal_frequency:.3f}")
        
        return validation_result
    
    def validate_pattern_recognition(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证MACD技术形态识别能力
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("开始验证MACD技术形态识别能力...")
        
        # 获取形态识别结果
        try:
            patterns = self.macd_indicator.get_patterns_Macd(data)
        except Exception as e:
            return {
                'test_name': 'MACD技术形态识别验证',
                'passed': False,
                'details': [f"形态识别失败: {e}"]
            }
        
        validation_result = {
            'test_name': 'MACD技术形态识别验证',
            'pattern_count': len(patterns.columns) if hasattr(patterns, 'columns') else 0,
            'recognized_patterns': list(patterns.columns) if hasattr(patterns, 'columns') else [],
            'passed': True,
            'details': []
        }
        
        # 检查关键形态是否存在
        expected_patterns = ['GOLDEN_CROSS', 'DEATH_CROSS', 'MACD_ZERO_CROSS_ABOVE', 'MACD_ZERO_CROSS_BELOW']
        
        for pattern in expected_patterns:
            if pattern in validation_result['recognized_patterns']:
                validation_result['details'].append(f"✅ 识别到{pattern}形态")
            else:
                validation_result['details'].append(f"⚠️ 未识别到{pattern}形态")
        
        return validation_result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        运行MACD指标的全面验证
        
        Returns:
            Dict[str, Any]: 完整验证结果
        """
        logger.info("开始MACD指标专业金融标准全面验证...")
        
        # 创建测试数据
        test_data = self.create_standard_test_data(100)
        
        # 执行各项验证
        results = {
            'validation_time': datetime.now().isoformat(),
            'test_data_length': len(test_data),
            'tests': {}
        }
        
        # 1. 计算准确性验证
        results['tests']['calculation_accuracy'] = self.validate_calculation_accuracy(test_data)
        
        # 2. 信号生成验证
        results['tests']['signal_generation'] = self.validate_signal_generation(test_data)
        
        # 3. 技术形态识别验证
        results['tests']['pattern_recognition'] = self.validate_pattern_recognition(test_data)
        
        # 计算总体通过率
        passed_tests = sum(1 for test in results['tests'].values() if test['passed'])
        total_tests = len(results['tests'])
        
        results['overall_score'] = passed_tests / total_tests
        results['overall_passed'] = results['overall_score'] >= 0.8  # 80%通过率
        
        logger.info(f"MACD验证完成，总体通过率: {results['overall_score']:.1%}")
        
        return results


if __name__ == "__main__":
    # 运行MACD专业金融标准验证
    validator = MACDFinancialStandardValidator()
    results = validator.run_comprehensive_validation()
    
    print("=== MACD指标专业金融标准验证报告 ===")
    print(f"验证时间: {results['validation_time']}")
    print(f"测试数据长度: {results['test_data_length']}")
    print(f"总体通过率: {results['overall_score']:.1%}")
    print(f"总体结果: {'✅ 通过' if results['overall_passed'] else '❌ 未通过'}")
    
    for test_name, test_result in results['tests'].items():
        print(f"\n📊 {test_result['test_name']}:")
        print(f"  结果: {'✅ 通过' if test_result['passed'] else '❌ 未通过'}")
        for detail in test_result['details']:
            print(f"  - {detail}")
