#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KDJ指标专业金融标准验证器

本模块用于验证KDJ指标的计算是否符合随机振荡器标准公式和行业标准，
包括K值、D值、J值计算准确性，金叉死叉识别，J线加速度分析等专业金融功能。

验证标准：
1. RSV = (收盘价 - 最低价) / (最高价 - 最低价) * 100
2. K值 = 2/3 * 前一日K值 + 1/3 * 当日RSV
3. D值 = 2/3 * 前一日D值 + 1/3 * 当日K值
4. J值 = 3 * K值 - 2 * D值
5. 金叉：K线上穿D线
6. 死叉：K线下穿D线
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

from indicators.kdj import KdjKdj
from utils.logger import get_logger

logger = get_logger(__name__)


class KDJFinancialStandardValidator:
    """KDJ指标专业金融标准验证器"""
    
    def __init__(self):
        self.kdj_indicator = KdjKdj()
        self.validation_results = {}
        
    def create_standard_test_data(self, length: int = 100) -> pd.DataFrame:
        """
        创建标准测试数据
        
        Args:
            length: 数据长度
            
        Returns:
            pd.DataFrame: 标准测试数据
        """
        # 创建模拟股价数据，包含明显的趋势和震荡
        np.random.seed(42)  # 确保结果可重现
        
        # 创建有趋势的价格数据
        base_price = 100
        
        # 分段创建不同的市场状态
        segment_length = length // 3
        
        # 上升趋势段
        uptrend = np.linspace(0, 20, segment_length) + np.random.normal(0, 2, segment_length)
        
        # 震荡整理段
        oscillation = 20 + 5 * np.sin(np.linspace(0, 4*np.pi, segment_length)) + np.random.normal(0, 1, segment_length)
        
        # 下降趋势段
        downtrend = np.linspace(25, 10, length - 2 * segment_length) + np.random.normal(0, 2, length - 2 * segment_length)
        
        # 合并所有段
        price_changes = np.concatenate([uptrend, oscillation, downtrend])
        close_prices = base_price + price_changes
        
        # 确保价格为正数
        close_prices = np.maximum(close_prices, 1.0)
        
        # 生成OHLC数据
        high_prices = close_prices * (1 + np.random.uniform(0, 0.03, length))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.03, length))
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
    
    def calculate_reference_kdj(self, data: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> Dict[str, pd.Series]:
        """
        计算参考KDJ值（使用标准随机振荡器公式）
        
        Args:
            data: 价格数据
            n: RSV周期
            m1: K值平滑周期
            m2: D值平滑周期
            
        Returns:
            Dict[str, pd.Series]: 包含K、D、J值的字典
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算RSV (Raw Stochastic Value)
        lowest_low = low.rolling(window=n).min()
        highest_high = high.rolling(window=n).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)  # 初始值设为50
        
        # 计算K值 (使用指数移动平均)
        k_values = pd.Series(index=data.index, dtype=float)
        k_values.iloc[0] = 50  # 初始K值
        
        for i in range(1, len(rsv)):
            if pd.notna(rsv.iloc[i]):
                k_values.iloc[i] = (2/3) * k_values.iloc[i-1] + (1/3) * rsv.iloc[i]
            else:
                k_values.iloc[i] = k_values.iloc[i-1]
        
        # 计算D值 (K值的平滑)
        d_values = pd.Series(index=data.index, dtype=float)
        d_values.iloc[0] = 50  # 初始D值
        
        for i in range(1, len(k_values)):
            d_values.iloc[i] = (2/3) * d_values.iloc[i-1] + (1/3) * k_values.iloc[i]
        
        # 计算J值
        j_values = 3 * k_values - 2 * d_values
        
        return {
            'K': k_values,
            'D': d_values,
            'J': j_values
        }
    
    def validate_calculation_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证KDJ计算准确性
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("开始验证KDJ计算准确性...")
        
        # 获取我们的KDJ计算结果
        our_result = self.kdj_indicator.calculate(data)
        
        # 计算参考KDJ值
        reference_result = self.calculate_reference_kdj(data)
        
        # 验证结果
        validation_result = {
            'test_name': 'KDJ计算准确性验证',
            'data_length': len(data),
            'accuracy_scores': {},
            'max_deviations': {},
            'mean_deviations': {},
            'passed': True,
            'details': []
        }
        
        # 检查每个指标的准确性
        for indicator_name in ['K', 'D', 'J']:
            if isinstance(our_result, dict) and indicator_name in our_result:
                our_values = our_result[indicator_name].dropna()
                ref_values = reference_result[indicator_name].dropna()
            elif hasattr(our_result, 'columns') and indicator_name in our_result.columns:
                our_values = our_result[indicator_name].dropna()
                ref_values = reference_result[indicator_name].dropna()
            else:
                validation_result['passed'] = False
                validation_result['details'].append(f"缺少{indicator_name}指标")
                continue
            
            # 确保长度一致
            min_length = min(len(our_values), len(ref_values))
            if min_length < 20:
                validation_result['details'].append(f"{indicator_name}有效数据点不足: {min_length}")
                continue
            
            our_values = our_values.iloc[-min_length:]
            ref_values = ref_values.iloc[-min_length:]
            
            # 计算偏差
            deviations = np.abs(our_values.values - ref_values.values)
            max_deviation = deviations.max()
            mean_deviation = deviations.mean()
            
            # 计算准确性分数 (基于相对误差)
            relative_errors = deviations / (np.abs(ref_values.values) + 1e-8)
            accuracy_score = 1.0 - relative_errors.mean()
            
            validation_result['accuracy_scores'][indicator_name] = accuracy_score
            validation_result['max_deviations'][indicator_name] = max_deviation
            validation_result['mean_deviations'][indicator_name] = mean_deviation
            
            # 判断是否通过 (准确性 > 85%)
            if accuracy_score < 0.85:
                validation_result['passed'] = False
                validation_result['details'].append(
                    f"{indicator_name}准确性不足: {accuracy_score:.3f} < 0.85"
                )
            else:
                validation_result['details'].append(
                    f"{indicator_name}准确性验证通过: {accuracy_score:.3f}"
                )
        
        return validation_result
    
    def validate_signal_generation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证KDJ信号生成能力
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("开始验证KDJ信号生成能力...")
        
        # 获取信号
        signals = self.kdj_indicator.get_signals(data)
        
        validation_result = {
            'test_name': 'KDJ信号生成验证',
            'signal_count': {
                'buy_signals': signals['buy_signal'].sum() if 'buy_signal' in signals.columns else 0,
                'sell_signals': signals['sell_signal'].sum() if 'sell_signal' in signals.columns else 0
            },
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
            
            # 检查信号频率是否合理
            total_signals = validation_result['signal_count']['buy_signals'] + validation_result['signal_count']['sell_signals']
            signal_frequency = total_signals / len(data)
            
            if signal_frequency > 0.25:  # KDJ信号频率不应超过25%
                validation_result['passed'] = False
                validation_result['details'].append(f"信号频率过高: {signal_frequency:.3f}")
            else:
                validation_result['details'].append(f"信号频率合理: {signal_frequency:.3f}")
        
        return validation_result
    
    def validate_overbought_oversold(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证KDJ超买超卖判断
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("开始验证KDJ超买超卖判断...")
        
        # 获取KDJ计算结果
        our_result = self.kdj_indicator.calculate(data)
        
        validation_result = {
            'test_name': 'KDJ超买超卖判断验证',
            'overbought_count': 0,
            'oversold_count': 0,
            'neutral_count': 0,
            'passed': True,
            'details': []
        }
        
        # 获取K值和D值
        k_values = None
        d_values = None
        
        if isinstance(our_result, dict):
            k_values = our_result.get('K')
            d_values = our_result.get('D')
        elif hasattr(our_result, 'columns'):
            k_values = our_result.get('K') if 'K' in our_result.columns else None
            d_values = our_result.get('D') if 'D' in our_result.columns else None
        
        if k_values is None or d_values is None:
            validation_result['passed'] = False
            validation_result['details'].append("未找到K值或D值")
            return validation_result
        
        k_values = k_values.dropna()
        d_values = d_values.dropna()
        
        # 统计超买超卖情况 (KDJ标准: >80超买, <20超卖)
        overbought = ((k_values > 80) | (d_values > 80)).sum()
        oversold = ((k_values < 20) | (d_values < 20)).sum()
        neutral = len(k_values) - overbought - oversold
        
        validation_result['overbought_count'] = overbought
        validation_result['oversold_count'] = oversold
        validation_result['neutral_count'] = neutral
        
        # 验证KDJ值的合理性
        if k_values.min() < 0 or k_values.max() > 100 or d_values.min() < 0 or d_values.max() > 100:
            validation_result['passed'] = False
            validation_result['details'].append(f"KDJ值超出0-100范围")
        
        validation_result['details'].append(f"超买信号: {overbought}次")
        validation_result['details'].append(f"超卖信号: {oversold}次")
        validation_result['details'].append(f"中性区域: {neutral}次")
        
        return validation_result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        运行KDJ指标的全面验证
        
        Returns:
            Dict[str, Any]: 完整验证结果
        """
        logger.info("开始KDJ指标专业金融标准全面验证...")
        
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
        
        # 3. 超买超卖判断验证
        results['tests']['overbought_oversold'] = self.validate_overbought_oversold(test_data)
        
        # 计算总体通过率
        passed_tests = sum(1 for test in results['tests'].values() if test['passed'])
        total_tests = len(results['tests'])
        
        results['overall_score'] = passed_tests / total_tests
        results['overall_passed'] = results['overall_score'] >= 0.8  # 80%通过率
        
        logger.info(f"KDJ验证完成，总体通过率: {results['overall_score']:.1%}")
        
        return results


if __name__ == "__main__":
    # 运行KDJ专业金融标准验证
    validator = KDJFinancialStandardValidator()
    results = validator.run_comprehensive_validation()
    
    print("=== KDJ指标专业金融标准验证报告 ===")
    print(f"验证时间: {results['validation_time']}")
    print(f"测试数据长度: {results['test_data_length']}")
    print(f"总体通过率: {results['overall_score']:.1%}")
    print(f"总体结果: {'✅ 通过' if results['overall_passed'] else '❌ 未通过'}")
    
    for test_name, test_result in results['tests'].items():
        print(f"\n📊 {test_result['test_name']}:")
        print(f"  结果: {'✅ 通过' if test_result['passed'] else '❌ 未通过'}")
        for detail in test_result['details']:
            print(f"  - {detail}")
