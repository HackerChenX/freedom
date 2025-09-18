#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSI指标专业金融标准验证器

本模块用于验证RSI指标的计算是否符合Wilder平滑算法和行业标准，
包括超买超卖判断、背离检测等专业金融功能。

验证标准：
1. Wilder平滑算法：RS = Average Gain / Average Loss
2. RSI = 100 - (100 / (1 + RS))
3. 超买区域：RSI > 70
4. 超卖区域：RSI < 30
5. 中性区域：30 <= RSI <= 70
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

from indicators.rsi import RsiRsi
from utils.logger import get_logger

logger = get_logger(__name__)


class RSIFinancialStandardValidator:
    """RSI指标专业金融标准验证器"""
    
    def __init__(self):
        self.rsi_indicator = RsiRsi()
        self.validation_results = {}
        
    def create_standard_test_data(self, length: int = 100) -> pd.DataFrame:
        """
        创建标准测试数据
        
        Args:
            length: 数据长度
            
        Returns:
            pd.DataFrame: 标准测试数据
        """
        # 创建模拟股价数据，包含明显的超买超卖区域
        np.random.seed(42)  # 确保结果可重现
        
        # 创建有趋势的价格数据
        base_price = 100
        
        # 分段创建不同的市场状态
        segment_length = length // 4
        
        # 上升趋势段
        uptrend = np.linspace(0, 15, segment_length) + np.random.normal(0, 1, segment_length)
        
        # 横盘整理段
        sideways = np.random.normal(15, 2, segment_length)
        
        # 下降趋势段
        downtrend = np.linspace(15, 5, segment_length) + np.random.normal(0, 1, segment_length)
        
        # 反弹段
        rebound = np.linspace(5, 12, length - 3 * segment_length) + np.random.normal(0, 1, length - 3 * segment_length)
        
        # 合并所有段
        price_changes = np.concatenate([uptrend, sideways, downtrend, rebound])
        close_prices = base_price + price_changes
        
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
    
    def calculate_reference_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算参考RSI值（使用标准Wilder平滑算法）
        
        Args:
            data: 价格数据
            period: RSI周期
            
        Returns:
            pd.Series: RSI值
        """
        close = data['close']
        
        # 计算价格变化
        delta = close.diff()
        
        # 分离涨跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Wilder平滑算法
        # 第一个值使用简单平均
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 从第二个周期开始使用Wilder平滑
        for i in range(period, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def validate_calculation_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证RSI计算准确性
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("开始验证RSI计算准确性...")
        
        # 获取我们的RSI计算结果
        our_result = self.rsi_indicator.calculate(data)
        
        # 计算参考RSI值
        reference_rsi = self.calculate_reference_rsi(data)
        
        # 验证结果
        validation_result = {
            'test_name': 'RSI计算准确性验证',
            'data_length': len(data),
            'accuracy_score': 0.0,
            'max_deviation': 0.0,
            'mean_deviation': 0.0,
            'passed': False,
            'details': []
        }
        
        # 检查RSI列是否存在 (处理Dict和DataFrame两种格式)
        rsi_column = None
        our_rsi = None

        if isinstance(our_result, dict):
            # Dict格式
            for key, value in our_result.items():
                if 'rsi' in key.lower() or 'RSI' in key:
                    rsi_column = key
                    our_rsi = value
                    break
        else:
            # DataFrame格式
            for col in our_result.columns:
                if 'rsi' in col.lower() or 'RSI' in col:
                    rsi_column = col
                    our_rsi = our_result[col]
                    break

        if rsi_column is None or our_rsi is None:
            validation_result['details'].append("未找到RSI计算结果列")
            return validation_result
        
        # 获取有效数据进行比较
        our_rsi = our_rsi.dropna()
        ref_rsi = reference_rsi.dropna()
        
        # 确保长度一致
        min_length = min(len(our_rsi), len(ref_rsi))
        if min_length < 20:
            validation_result['details'].append(f"有效数据点不足: {min_length}")
            return validation_result
        
        our_rsi = our_rsi.iloc[-min_length:]
        ref_rsi = ref_rsi.iloc[-min_length:]
        
        # 计算偏差
        deviations = np.abs(our_rsi.values - ref_rsi.values)
        max_deviation = deviations.max()
        mean_deviation = deviations.mean()
        
        # 计算准确性分数 (基于相对误差)
        relative_errors = deviations / (np.abs(ref_rsi.values) + 1e-8)
        accuracy_score = 1.0 - relative_errors.mean()
        
        validation_result['accuracy_score'] = accuracy_score
        validation_result['max_deviation'] = max_deviation
        validation_result['mean_deviation'] = mean_deviation
        
        # 判断是否通过 (准确性 > 90%)
        if accuracy_score > 0.90:
            validation_result['passed'] = True
            validation_result['details'].append(f"RSI准确性验证通过: {accuracy_score:.3f}")
        else:
            validation_result['details'].append(f"RSI准确性不足: {accuracy_score:.3f} < 0.90")
        
        return validation_result
    
    def validate_overbought_oversold(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证RSI超买超卖判断
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("开始验证RSI超买超卖判断...")
        
        # 获取RSI计算结果
        our_result = self.rsi_indicator.calculate(data)
        
        validation_result = {
            'test_name': 'RSI超买超卖判断验证',
            'overbought_count': 0,
            'oversold_count': 0,
            'neutral_count': 0,
            'signal_quality': {},
            'passed': True,
            'details': []
        }
        
        # 找到RSI列 (处理Dict和DataFrame两种格式)
        rsi_column = None
        rsi_values = None

        if isinstance(our_result, dict):
            # Dict格式
            for key, value in our_result.items():
                if 'rsi' in key.lower() or 'RSI' in key:
                    rsi_column = key
                    rsi_values = value.dropna()
                    break
        else:
            # DataFrame格式
            for col in our_result.columns:
                if 'rsi' in col.lower() or 'RSI' in col:
                    rsi_column = col
                    rsi_values = our_result[col].dropna()
                    break

        if rsi_column is None or rsi_values is None:
            validation_result['passed'] = False
            validation_result['details'].append("未找到RSI计算结果列")
            return validation_result
        
        # 统计超买超卖情况
        overbought = (rsi_values > 70).sum()
        oversold = (rsi_values < 30).sum()
        neutral = ((rsi_values >= 30) & (rsi_values <= 70)).sum()
        
        validation_result['overbought_count'] = overbought
        validation_result['oversold_count'] = oversold
        validation_result['neutral_count'] = neutral
        
        # 验证RSI值的合理性
        if rsi_values.min() < 0 or rsi_values.max() > 100:
            validation_result['passed'] = False
            validation_result['details'].append(f"RSI值超出0-100范围: {rsi_values.min():.2f} - {rsi_values.max():.2f}")
        
        # 验证超买超卖区域的合理性
        total_signals = overbought + oversold
        signal_ratio = total_signals / len(rsi_values)
        
        if signal_ratio > 0.4:  # 超买超卖信号不应过于频繁
            validation_result['passed'] = False
            validation_result['details'].append(f"超买超卖信号过于频繁: {signal_ratio:.3f}")
        else:
            validation_result['details'].append(f"超买超卖信号频率合理: {signal_ratio:.3f}")
        
        validation_result['details'].append(f"超买信号: {overbought}次")
        validation_result['details'].append(f"超卖信号: {oversold}次")
        validation_result['details'].append(f"中性区域: {neutral}次")
        
        return validation_result
    
    def validate_signal_generation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证RSI信号生成能力
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        logger.info("开始验证RSI信号生成能力...")
        
        # 获取信号
        signals = self.rsi_indicator.get_signals(data)
        
        validation_result = {
            'test_name': 'RSI信号生成验证',
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
            
            if signal_frequency > 0.3:  # 信号频率不应超过30%
                validation_result['passed'] = False
                validation_result['details'].append(f"信号频率过高: {signal_frequency:.3f}")
            else:
                validation_result['details'].append(f"信号频率合理: {signal_frequency:.3f}")
        
        return validation_result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        运行RSI指标的全面验证
        
        Returns:
            Dict[str, Any]: 完整验证结果
        """
        logger.info("开始RSI指标专业金融标准全面验证...")
        
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
        
        # 2. 超买超卖判断验证
        results['tests']['overbought_oversold'] = self.validate_overbought_oversold(test_data)
        
        # 3. 信号生成验证
        results['tests']['signal_generation'] = self.validate_signal_generation(test_data)
        
        # 计算总体通过率
        passed_tests = sum(1 for test in results['tests'].values() if test['passed'])
        total_tests = len(results['tests'])
        
        results['overall_score'] = passed_tests / total_tests
        results['overall_passed'] = results['overall_score'] >= 0.8  # 80%通过率
        
        logger.info(f"RSI验证完成，总体通过率: {results['overall_score']:.1%}")
        
        return results


if __name__ == "__main__":
    # 运行RSI专业金融标准验证
    validator = RSIFinancialStandardValidator()
    results = validator.run_comprehensive_validation()
    
    print("=== RSI指标专业金融标准验证报告 ===")
    print(f"验证时间: {results['validation_time']}")
    print(f"测试数据长度: {results['test_data_length']}")
    print(f"总体通过率: {results['overall_score']:.1%}")
    print(f"总体结果: {'✅ 通过' if results['overall_passed'] else '❌ 未通过'}")
    
    for test_name, test_result in results['tests'].items():
        print(f"\n📊 {test_result['test_name']}:")
        print(f"  结果: {'✅ 通过' if test_result['passed'] else '❌ 未通过'}")
        for detail in test_result['details']:
            print(f"  - {detail}")
