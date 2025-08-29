#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MA指标阶段3: 形态识别验证

验证MA趋势识别、金叉死叉信号、多周期MA交叉等
"""

import sys
import os
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class MAPatternRecognition:
    """MA指标形态识别验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.verification_name = "MA形态识别验证"
        self.start_time = datetime.now()
        
        # 形态识别验证标准
        self.pattern_standards = {
            'trend_identification': 95.0,     # 趋势识别要求95%
            'crossover_signals': 95.0,        # 交叉信号要求95%
            'multi_period_analysis': 95.0,    # 多周期分析要求95%
            'pattern_accuracy': 95.0,         # 形态准确性要求95%
            'signal_quality': 95.0,           # 信号质量要求95%
            'target_score': 95.0
        }
        
        logger.info(f"✅ {self.verification_name}初始化完成")
        logger.info(f"🎯 目标: 验证MA形态识别能力")
    
    def run_pattern_recognition_verification(self) -> Dict[str, Any]:
        """运行形态识别验证"""
        logger.info("🚀 开始MA形态识别验证")
        
        verification_results = {
            'verification_session': {
                'name': self.verification_name,
                'start_time': self.start_time.isoformat(),
                'standards': self.pattern_standards
            },
            'trend_identification_tests': {},
            'crossover_signal_tests': {},
            'multi_period_tests': {},
            'pattern_accuracy_tests': {},
            'signal_quality_tests': {},
            'final_assessment': {},
            'final_status': 'IN_PROGRESS'
        }
        
        try:
            # 导入MA指标
            from indicators.ma import MaMa
            ma = MaMa()
            
            # 测试1: 趋势识别测试
            logger.info("📈 测试1: 趋势识别测试")
            trend_result = self._test_trend_identification(ma)
            verification_results['trend_identification_tests'] = trend_result
            
            # 测试2: 交叉信号测试
            logger.info("⚡ 测试2: 交叉信号测试")
            crossover_result = self._test_crossover_signals(ma)
            verification_results['crossover_signal_tests'] = crossover_result
            
            # 测试3: 多周期分析测试
            logger.info("🔄 测试3: 多周期分析测试")
            multi_period_result = self._test_multi_period_analysis(ma)
            verification_results['multi_period_tests'] = multi_period_result
            
            # 测试4: 形态准确性测试
            logger.info("🎯 测试4: 形态准确性测试")
            pattern_accuracy_result = self._test_pattern_accuracy(ma)
            verification_results['pattern_accuracy_tests'] = pattern_accuracy_result
            
            # 测试5: 信号质量测试
            logger.info("✨ 测试5: 信号质量测试")
            signal_quality_result = self._test_signal_quality(ma)
            verification_results['signal_quality_tests'] = signal_quality_result
            
            # 最终评估
            logger.info("📊 最终评估")
            final_assessment = self._generate_final_assessment(verification_results)
            verification_results['final_assessment'] = final_assessment
            
            # 确定最终状态
            final_status = self._determine_final_status(final_assessment)
            verification_results['final_status'] = final_status
            
            logger.info("✅ MA形态识别验证完成")
            return verification_results
            
        except Exception as e:
            logger.error(f"❌ 验证过程中发生异常: {e}")
            verification_results['final_status'] = 'ERROR'
            verification_results['error'] = str(e)
            verification_results['traceback'] = traceback.format_exc()
            return verification_results
    
    def _test_trend_identification(self, ma) -> Dict[str, Any]:
        """测试趋势识别"""
        logger.info("📈 测试MA趋势识别...")
        
        trend_result = {
            'uptrend_test': {},
            'downtrend_test': {},
            'sideways_test': {},
            'trend_change_test': {},
            'overall_score': 0.0
        }
        
        try:
            # 测试1: 上升趋势识别
            uptrend_test = self._test_uptrend_identification(ma)
            trend_result['uptrend_test'] = uptrend_test
            
            # 测试2: 下降趋势识别
            downtrend_test = self._test_downtrend_identification(ma)
            trend_result['downtrend_test'] = downtrend_test
            
            # 测试3: 横盘趋势识别
            sideways_test = self._test_sideways_identification(ma)
            trend_result['sideways_test'] = sideways_test
            
            # 测试4: 趋势变化识别
            trend_change_test = self._test_trend_change_identification(ma)
            trend_result['trend_change_test'] = trend_change_test
            
            # 计算总体评分
            scores = [
                uptrend_test.get('score', 0),
                downtrend_test.get('score', 0),
                sideways_test.get('score', 0),
                trend_change_test.get('score', 0)
            ]
            trend_result['overall_score'] = sum(scores) / len(scores)
            
            logger.info(f"✅ 趋势识别测试完成: {trend_result['overall_score']:.1f}分")
            return trend_result
            
        except Exception as e:
            logger.error(f"❌ 趋势识别测试失败: {e}")
            trend_result['error'] = str(e)
            return trend_result
    
    def _test_uptrend_identification(self, ma) -> Dict[str, Any]:
        """测试上升趋势识别"""
        # 创建明显的上升趋势数据
        uptrend_data = self._create_uptrend_data(50)
        
        try:
            ma.set_parameters(period=10, ma_type='SMA')
            result = ma.calculate(uptrend_data)
            
            if result is not None and 'ma' in result.columns:
                ma_values = result['ma'].dropna()
                
                if len(ma_values) > 10:
                    # 检查MA是否呈上升趋势
                    ma_diff = ma_values.diff().dropna()
                    positive_changes = (ma_diff > 0).sum()
                    total_changes = len(ma_diff)
                    
                    uptrend_ratio = positive_changes / total_changes if total_changes > 0 else 0
                    
                    # 评分：上升趋势比例越高分数越高
                    score = uptrend_ratio * 100
                    
                    return {
                        'uptrend_ratio': uptrend_ratio,
                        'positive_changes': positive_changes,
                        'total_changes': total_changes,
                        'identified_correctly': uptrend_ratio > 0.7,
                        'score': score
                    }
            
            return {'score': 0, 'error': '无法计算MA值'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_downtrend_identification(self, ma) -> Dict[str, Any]:
        """测试下降趋势识别"""
        # 创建明显的下降趋势数据
        downtrend_data = self._create_downtrend_data(50)
        
        try:
            ma.set_parameters(period=10, ma_type='SMA')
            result = ma.calculate(downtrend_data)
            
            if result is not None and 'ma' in result.columns:
                ma_values = result['ma'].dropna()
                
                if len(ma_values) > 10:
                    # 检查MA是否呈下降趋势
                    ma_diff = ma_values.diff().dropna()
                    negative_changes = (ma_diff < 0).sum()
                    total_changes = len(ma_diff)
                    
                    downtrend_ratio = negative_changes / total_changes if total_changes > 0 else 0
                    
                    # 评分：下降趋势比例越高分数越高
                    score = downtrend_ratio * 100
                    
                    return {
                        'downtrend_ratio': downtrend_ratio,
                        'negative_changes': negative_changes,
                        'total_changes': total_changes,
                        'identified_correctly': downtrend_ratio > 0.7,
                        'score': score
                    }
            
            return {'score': 0, 'error': '无法计算MA值'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_sideways_identification(self, ma) -> Dict[str, Any]:
        """测试横盘趋势识别"""
        # 创建横盘震荡数据
        sideways_data = self._create_sideways_data(50)
        
        try:
            ma.set_parameters(period=10, ma_type='SMA')
            result = ma.calculate(sideways_data)
            
            if result is not None and 'ma' in result.columns:
                ma_values = result['ma'].dropna()
                
                if len(ma_values) > 10:
                    # 检查MA的波动性
                    ma_std = ma_values.std()
                    ma_mean = ma_values.mean()
                    
                    # 横盘的特征：标准差相对较小
                    volatility_ratio = ma_std / ma_mean if ma_mean != 0 else 0
                    
                    # 评分：波动性越小（越横盘）分数越高
                    if volatility_ratio < 0.02:  # 2%以内的波动认为是横盘
                        score = 100
                    elif volatility_ratio < 0.05:  # 5%以内给部分分数
                        score = 80
                    else:
                        score = max(0, 80 - (volatility_ratio - 0.05) * 1000)
                    
                    return {
                        'volatility_ratio': volatility_ratio,
                        'ma_std': ma_std,
                        'ma_mean': ma_mean,
                        'identified_correctly': volatility_ratio < 0.05,
                        'score': score
                    }
            
            return {'score': 0, 'error': '无法计算MA值'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_trend_change_identification(self, ma) -> Dict[str, Any]:
        """测试趋势变化识别"""
        # 创建趋势变化数据（先上升后下降）
        trend_change_data = self._create_trend_change_data(60)
        
        try:
            ma.set_parameters(period=10, ma_type='SMA')
            result = ma.calculate(trend_change_data)
            
            if result is not None and 'ma' in result.columns:
                ma_values = result['ma'].dropna()
                
                if len(ma_values) > 20:
                    # 分析前半段和后半段的趋势
                    mid_point = len(ma_values) // 2
                    first_half = ma_values.iloc[:mid_point]
                    second_half = ma_values.iloc[mid_point:]
                    
                    # 前半段应该上升
                    first_trend = (first_half.iloc[-1] - first_half.iloc[0]) / first_half.iloc[0]
                    # 后半段应该下降
                    second_trend = (second_half.iloc[-1] - second_half.iloc[0]) / second_half.iloc[0]
                    
                    # 评分：能否识别趋势变化
                    trend_change_detected = first_trend > 0.02 and second_trend < -0.02
                    score = 100 if trend_change_detected else 50
                    
                    return {
                        'first_half_trend': first_trend,
                        'second_half_trend': second_trend,
                        'trend_change_detected': trend_change_detected,
                        'score': score
                    }
            
            return {'score': 0, 'error': '无法计算MA值'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_crossover_signals(self, ma) -> Dict[str, Any]:
        """测试交叉信号"""
        logger.info("⚡ 测试MA交叉信号...")
        
        crossover_result = {
            'golden_cross_test': {},
            'death_cross_test': {},
            'signal_timing_test': {},
            'overall_score': 0.0
        }
        
        try:
            # 测试1: 金叉信号测试
            golden_cross_test = self._test_golden_cross(ma)
            crossover_result['golden_cross_test'] = golden_cross_test
            
            # 测试2: 死叉信号测试
            death_cross_test = self._test_death_cross(ma)
            crossover_result['death_cross_test'] = death_cross_test
            
            # 测试3: 信号时机测试
            signal_timing_test = self._test_signal_timing(ma)
            crossover_result['signal_timing_test'] = signal_timing_test
            
            # 计算总体评分
            scores = [
                golden_cross_test.get('score', 0),
                death_cross_test.get('score', 0),
                signal_timing_test.get('score', 0)
            ]
            crossover_result['overall_score'] = sum(scores) / len(scores)
            
            logger.info(f"✅ 交叉信号测试完成: {crossover_result['overall_score']:.1f}分")
            return crossover_result
            
        except Exception as e:
            logger.error(f"❌ 交叉信号测试失败: {e}")
            crossover_result['error'] = str(e)
            return crossover_result
    
    def _test_golden_cross(self, ma) -> Dict[str, Any]:
        """测试金叉信号"""
        # 创建金叉场景数据
        golden_cross_data = self._create_golden_cross_data(40)
        
        try:
            # 使用双MA系统：5日和20日
            ma.set_parameters(periods=[5, 20], ma_type='SMA')
            result = ma.calculate(golden_cross_data)
            
            if result is not None and 'MA5' in result.columns and 'MA20' in result.columns:
                ma5 = result['MA5'].dropna()
                ma20 = result['MA20'].dropna()
                
                if len(ma5) > 10 and len(ma20) > 10:
                    # 检查是否发生金叉（短期MA上穿长期MA）
                    # 找到有效数据的共同索引
                    common_index = ma5.index.intersection(ma20.index)
                    if len(common_index) > 10:
                        ma5_common = ma5.loc[common_index]
                        ma20_common = ma20.loc[common_index]
                        
                        # 检查最后是否短期MA在长期MA之上
                        final_golden = ma5_common.iloc[-1] > ma20_common.iloc[-1]
                        
                        # 检查是否有交叉发生
                        crossover_detected = False
                        for i in range(1, len(ma5_common)):
                            if (ma5_common.iloc[i-1] <= ma20_common.iloc[i-1] and 
                                ma5_common.iloc[i] > ma20_common.iloc[i]):
                                crossover_detected = True
                                break
                        
                        score = 100 if (final_golden and crossover_detected) else 70 if final_golden else 30
                        
                        return {
                            'final_golden_cross': final_golden,
                            'crossover_detected': crossover_detected,
                            'ma5_final': ma5_common.iloc[-1],
                            'ma20_final': ma20_common.iloc[-1],
                            'score': score
                        }
            
            return {'score': 0, 'error': '无法计算双MA值'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_death_cross(self, ma) -> Dict[str, Any]:
        """测试死叉信号"""
        # 创建死叉场景数据
        death_cross_data = self._create_death_cross_data(40)
        
        try:
            # 使用双MA系统：5日和20日
            ma.set_parameters(periods=[5, 20], ma_type='SMA')
            result = ma.calculate(death_cross_data)
            
            if result is not None and 'MA5' in result.columns and 'MA20' in result.columns:
                ma5 = result['MA5'].dropna()
                ma20 = result['MA20'].dropna()
                
                if len(ma5) > 10 and len(ma20) > 10:
                    # 检查是否发生死叉（短期MA下穿长期MA）
                    common_index = ma5.index.intersection(ma20.index)
                    if len(common_index) > 10:
                        ma5_common = ma5.loc[common_index]
                        ma20_common = ma20.loc[common_index]
                        
                        # 检查最后是否短期MA在长期MA之下
                        final_death = ma5_common.iloc[-1] < ma20_common.iloc[-1]
                        
                        # 检查是否有交叉发生
                        crossover_detected = False
                        for i in range(1, len(ma5_common)):
                            if (ma5_common.iloc[i-1] >= ma20_common.iloc[i-1] and 
                                ma5_common.iloc[i] < ma20_common.iloc[i]):
                                crossover_detected = True
                                break
                        
                        score = 100 if (final_death and crossover_detected) else 70 if final_death else 30
                        
                        return {
                            'final_death_cross': final_death,
                            'crossover_detected': crossover_detected,
                            'ma5_final': ma5_common.iloc[-1],
                            'ma20_final': ma20_common.iloc[-1],
                            'score': score
                        }
            
            return {'score': 0, 'error': '无法计算双MA值'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_signal_timing(self, ma) -> Dict[str, Any]:
        """测试信号时机"""
        # 创建标准测试数据
        timing_data = self._create_standard_test_data(50)
        
        try:
            ma.set_parameters(period=10, ma_type='SMA')
            result = ma.calculate(timing_data)
            
            if result is not None and 'buy_signal' in result.columns and 'sell_signal' in result.columns:
                buy_signals = result['buy_signal'].sum()
                sell_signals = result['sell_signal'].sum()
                
                # 评分：有合理数量的信号
                total_signals = buy_signals + sell_signals
                signal_ratio = total_signals / len(result) if len(result) > 0 else 0
                
                # 合理的信号频率：5%-20%
                if 0.05 <= signal_ratio <= 0.20:
                    score = 100
                elif 0.02 <= signal_ratio <= 0.30:
                    score = 80
                else:
                    score = 50
                
                return {
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'total_signals': total_signals,
                    'signal_ratio': signal_ratio,
                    'reasonable_timing': 0.05 <= signal_ratio <= 0.20,
                    'score': score
                }
            
            return {'score': 0, 'error': '缺少信号列'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_multi_period_analysis(self, ma) -> Dict[str, Any]:
        """测试多周期分析"""
        logger.info("🔄 测试MA多周期分析...")
        
        # 创建测试数据
        test_data = self._create_standard_test_data(100)
        
        try:
            # 设置多周期参数
            ma.set_parameters(periods=[5, 10, 20, 50], ma_type='SMA')
            result = ma.calculate(test_data)
            
            if result is not None:
                # 检查是否有所有周期的MA
                expected_columns = ['MA5', 'MA10', 'MA20', 'MA50']
                has_all_periods = all(col in result.columns for col in expected_columns)
                
                if has_all_periods:
                    # 检查多周期关系的合理性
                    valid_data = result.dropna(subset=expected_columns)
                    
                    if len(valid_data) > 10:
                        # 检查短期MA比长期MA更敏感（波动性更大）
                        ma5_std = valid_data['MA5'].std()
                        ma50_std = valid_data['MA50'].std()
                        
                        sensitivity_correct = ma5_std > ma50_std
                        
                        # 检查MA的排列关系在趋势中的合理性
                        final_values = valid_data.iloc[-1]
                        ma_values = [final_values['MA5'], final_values['MA10'], final_values['MA20'], final_values['MA50']]
                        
                        # 计算评分
                        score = 0
                        if has_all_periods:
                            score += 40
                        if sensitivity_correct:
                            score += 30
                        if len(valid_data) > 10:
                            score += 30
                        
                        return {
                            'has_all_periods': has_all_periods,
                            'sensitivity_correct': sensitivity_correct,
                            'valid_data_points': len(valid_data),
                            'ma5_volatility': ma5_std,
                            'ma50_volatility': ma50_std,
                            'final_ma_values': ma_values,
                            'score': score
                        }
            
            return {'score': 0, 'error': '多周期计算失败'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_pattern_accuracy(self, ma) -> Dict[str, Any]:
        """测试形态准确性"""
        logger.info("🎯 测试MA形态准确性...")
        
        # 简化的形态准确性测试
        test_data = self._create_standard_test_data(50)
        
        try:
            ma.set_parameters(period=20, ma_type='SMA')
            result = ma.calculate(test_data)
            
            if result is not None and 'ma' in result.columns:
                ma_values = result['ma'].dropna()
                
                if len(ma_values) > 10:
                    # 检查MA的平滑性（相邻值不应该有剧烈跳跃）
                    ma_diff = ma_values.diff().dropna()
                    max_change = ma_diff.abs().max()
                    mean_price = ma_values.mean()
                    
                    # 最大变化不应该超过平均价格的5%
                    smoothness_ratio = max_change / mean_price if mean_price > 0 else 0
                    
                    # 评分：越平滑分数越高
                    if smoothness_ratio < 0.02:
                        score = 100
                    elif smoothness_ratio < 0.05:
                        score = 80
                    else:
                        score = max(0, 80 - (smoothness_ratio - 0.05) * 1000)
                    
                    return {
                        'smoothness_ratio': smoothness_ratio,
                        'max_change': max_change,
                        'mean_price': mean_price,
                        'pattern_accurate': smoothness_ratio < 0.05,
                        'score': score
                    }
            
            return {'score': 0, 'error': '无法计算MA值'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _test_signal_quality(self, ma) -> Dict[str, Any]:
        """测试信号质量"""
        logger.info("✨ 测试MA信号质量...")
        
        # 创建测试数据
        test_data = self._create_standard_test_data(60)
        
        try:
            ma.set_parameters(period=15, ma_type='SMA')
            result = ma.calculate(test_data)
            
            if result is not None:
                # 检查信号列是否存在
                signal_columns = ['buy_signal', 'sell_signal', 'hold_signal']
                has_signals = all(col in result.columns for col in signal_columns)
                
                if has_signals:
                    buy_count = result['buy_signal'].sum()
                    sell_count = result['sell_signal'].sum()
                    hold_count = result['hold_signal'].sum()
                    
                    total_signals = buy_count + sell_count + hold_count
                    
                    # 评分：信号分布合理性
                    score = 0
                    if has_signals:
                        score += 50
                    if total_signals > 0:
                        score += 30
                    if buy_count > 0 and sell_count > 0:  # 有买卖信号
                        score += 20
                    
                    return {
                        'has_signals': has_signals,
                        'buy_signals': buy_count,
                        'sell_signals': sell_count,
                        'hold_signals': hold_count,
                        'total_signals': total_signals,
                        'signal_quality_good': total_signals > 0,
                        'score': score
                    }
            
            return {'score': 0, 'error': '缺少信号列'}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}
    
    def _create_uptrend_data(self, size: int) -> pd.DataFrame:
        """创建上升趋势数据"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
        base_price = 100
        prices = []
        
        for i in range(size):
            # 创建明显的上升趋势
            trend_component = i * 0.5  # 每天上涨0.5
            noise = np.random.normal(0, 0.5)  # 小幅随机波动
            price = base_price + trend_component + noise
            prices.append(max(price, 1))
        
        return self._create_dataframe_from_prices(dates, prices)
    
    def _create_downtrend_data(self, size: int) -> pd.DataFrame:
        """创建下降趋势数据"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
        base_price = 120
        prices = []
        
        for i in range(size):
            # 创建明显的下降趋势
            trend_component = -i * 0.4  # 每天下跌0.4
            noise = np.random.normal(0, 0.5)  # 小幅随机波动
            price = base_price + trend_component + noise
            prices.append(max(price, 1))
        
        return self._create_dataframe_from_prices(dates, prices)
    
    def _create_sideways_data(self, size: int) -> pd.DataFrame:
        """创建横盘数据"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
        base_price = 100
        prices = []
        
        for i in range(size):
            # 创建横盘震荡
            noise = np.random.normal(0, 1)  # 随机波动
            price = base_price + noise
            prices.append(max(price, 1))
        
        return self._create_dataframe_from_prices(dates, prices)
    
    def _create_trend_change_data(self, size: int) -> pd.DataFrame:
        """创建趋势变化数据"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
        base_price = 100
        prices = []
        mid_point = size // 2
        
        for i in range(size):
            if i < mid_point:
                # 前半段上升
                trend_component = i * 0.3
            else:
                # 后半段下降
                peak_value = (mid_point - 1) * 0.3
                trend_component = peak_value - (i - mid_point) * 0.4
            
            noise = np.random.normal(0, 0.3)
            price = base_price + trend_component + noise
            prices.append(max(price, 1))
        
        return self._create_dataframe_from_prices(dates, prices)
    
    def _create_golden_cross_data(self, size: int) -> pd.DataFrame:
        """创建金叉场景数据"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
        base_price = 100
        prices = []
        
        for i in range(size):
            if i < size // 2:
                # 前半段缓慢下跌
                trend_component = -i * 0.1
            else:
                # 后半段快速上涨（形成金叉）
                trend_component = -(size // 2) * 0.1 + (i - size // 2) * 0.8
            
            noise = np.random.normal(0, 0.3)
            price = base_price + trend_component + noise
            prices.append(max(price, 1))
        
        return self._create_dataframe_from_prices(dates, prices)
    
    def _create_death_cross_data(self, size: int) -> pd.DataFrame:
        """创建死叉场景数据"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
        base_price = 120
        prices = []
        
        for i in range(size):
            if i < size // 2:
                # 前半段缓慢上涨
                trend_component = i * 0.1
            else:
                # 后半段快速下跌（形成死叉）
                trend_component = (size // 2) * 0.1 - (i - size // 2) * 0.8
            
            noise = np.random.normal(0, 0.3)
            price = base_price + trend_component + noise
            prices.append(max(price, 1))
        
        return self._create_dataframe_from_prices(dates, prices)
    
    def _create_dataframe_from_prices(self, dates, prices) -> pd.DataFrame:
        """从价格列表创建DataFrame"""
        highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
        
        return pd.DataFrame({
            'date': dates,
            'code': ['TEST'] * len(prices),
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(prices))
        })
    
    def _create_standard_test_data(self, size: int) -> pd.DataFrame:
        """创建标准测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='D')
        np.random.seed(42)
        
        base_price = 100
        price_changes = np.random.normal(0.1, 2, size)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change / 100)
            prices.append(max(new_price, 1))
        
        return self._create_dataframe_from_prices(dates, prices)
    
    def _generate_final_assessment(self, verification_results: Dict) -> Dict[str, Any]:
        """生成最终评估"""
        trend_score = verification_results.get('trend_identification_tests', {}).get('overall_score', 0)
        crossover_score = verification_results.get('crossover_signal_tests', {}).get('overall_score', 0)
        multi_period_score = verification_results.get('multi_period_tests', {}).get('score', 0)
        pattern_accuracy_score = verification_results.get('pattern_accuracy_tests', {}).get('score', 0)
        signal_quality_score = verification_results.get('signal_quality_tests', {}).get('score', 0)
        
        overall_score = (trend_score + crossover_score + multi_period_score + pattern_accuracy_score + signal_quality_score) / 5
        
        return {
            'trend_identification_score': trend_score,
            'crossover_signals_score': crossover_score,
            'multi_period_analysis_score': multi_period_score,
            'pattern_accuracy_score': pattern_accuracy_score,
            'signal_quality_score': signal_quality_score,
            'overall_score': overall_score,
            'pattern_recognition_complete': overall_score >= 95.0,
            'target_achieved': overall_score >= 95.0
        }
    
    def _determine_final_status(self, final_assessment: Dict) -> str:
        """确定最终状态"""
        overall_score = final_assessment.get('overall_score', 0)
        
        if overall_score >= 95.0:
            return 'PATTERN_RECOGNITION_COMPLETE'
        elif overall_score >= 85.0:
            return 'PATTERN_RECOGNITION_MOSTLY_COMPLETE'
        else:
            return 'PATTERN_RECOGNITION_NEEDS_IMPROVEMENT'


def main():
    """主函数"""
    print("🚀 启动MA指标阶段3: 形态识别验证")
    print("验证MA趋势识别、金叉死叉信号、多周期MA交叉等")
    print("=" * 80)
    
    try:
        # 创建验证器
        verifier = MAPatternRecognition()
        
        # 运行形态识别验证
        results = verifier.run_pattern_recognition_verification()
        
        # 输出验证摘要
        print(f"\n📊 验证摘要:")
        print(f"最终状态: {results['final_status']}")
        
        if 'final_assessment' in results:
            assessment = results['final_assessment']
            print(f"趋势识别评分: {assessment.get('trend_identification_score', 0):.1f}/100")
            print(f"交叉信号评分: {assessment.get('crossover_signals_score', 0):.1f}/100")
            print(f"多周期分析评分: {assessment.get('multi_period_analysis_score', 0):.1f}/100")
            print(f"形态准确性评分: {assessment.get('pattern_accuracy_score', 0):.1f}/100")
            print(f"信号质量评分: {assessment.get('signal_quality_score', 0):.1f}/100")
            print(f"总体评分: {assessment.get('overall_score', 0):.1f}/100")
            print(f"形态识别完整: {'✅ 是' if assessment.get('pattern_recognition_complete', False) else '❌ 否'}")
            print(f"目标达成: {'✅ 是' if assessment.get('target_achieved', False) else '❌ 否'}")
        
        if results['final_status'] == 'PATTERN_RECOGNITION_COMPLETE':
            print("🎉 MA形态识别验证通过!")
            return 0
        else:
            print("⚠️ MA形态识别需要进一步优化")
            return 1
            
    except Exception as e:
        logger.error(f"💥 验证执行失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
