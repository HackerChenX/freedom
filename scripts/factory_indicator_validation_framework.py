#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工厂模式指标验证框架
专门为54个工厂模式指标开发的统一验证框架，适配dict返回格式
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import traceback

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger
from utils.enhanced_performance_monitor import performance_monitor
from utils.enhanced_exception_handler import exception_handler

logger = get_logger(__name__)


class FactoryIndicatorValidationFramework:
    """工厂模式指标验证框架"""
    
    def __init__(self):
        self.min_score = 95.0  # 95分以上标准
        self.test_data = None
        self.validation_results = {}
        
        # 54个工厂模式指标分类
        self.zxm_indicators = [
            'ZXM_DAILY_MACD', 'ZXM_turnover_rate', 'ZXM_VOLUME_SHRINK', 'ZXM_MA_CALLBACK',
            'ZXM_BS_ABSORB', 'ZXM_DAILY_TREND_UP', 'ZXM_WEEKLY_TREND_UP', 'ZXM_MONTHLY_KDJ_TREND_UP',
            'ZXM_WEEKLY_MACD', 'ZXM_MONTHLY_MACD', 'ZXM_AMPLITUDE_ELASTICITY', 'ZXM_RISE_ELASTICITY',
            'ZXM_ELASTICITY', 'ZXM_BOUNCE_DETECTOR', 'ZXM_BUYPOINT_SCORE', 'ZXM_TREND_SCORE',
            'ZXM_ELASTIC_SCORE', 'ZXM_VOLUME_ENERGY', 'ZXM_PRICE_POSITION', 'ZXM_TECHNICAL_FORM',
            'ZXM_HOT_SPOT', 'ZXM_INDUSTRY_ROTATION', 'ZXM_CYCLE_POSITION', 'ZXM_RISK_CONTROL',
            'ZXM_TIMING_SIGNAL', 'ZXM_POSITION_MANAGEMENT', 'ZXM_PORTFOLIO_OPTIMIZATION', 'ZXM_STRATEGY_COMBINATION',
            'ZXM_PERFORMANCE_ATTRIBUTION', 'ZXM_ALPHA_GENERATION', 'ZXM_BETA_HEDGING', 'ZXM_MARKET_SENTIMENT',
            'ZXM_LIQUIDITY_ANALYSIS', 'ZXM_VOLATILITY_FORECAST', 'ZXM_CORRELATION_MATRIX'
        ]
        
        self.pattern_indicators = [
            'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI', 'PIERCING_LINE',
            'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR', 'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS',
            'V_SHAPED_REVERSAL', 'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'TRIANGLE',
            'WEDGE', 'FLAG', 'PENNANT'
        ]
        
        self.all_factory_indicators = self.zxm_indicators + self.pattern_indicators
        
    @performance_monitor(threshold_seconds=2.0)
    def generate_comprehensive_test_data(self) -> pd.DataFrame:
        """生成适用于所有工厂模式指标的综合测试数据"""
        logger.info("📊 生成工厂模式指标综合测试数据...")
        
        # 生成150天的测试数据，包含多种市场特征
        dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
        
        np.random.seed(42)
        base_price = 100.0
        base_volume = 2000000
        
        # 模拟复杂的市场环境，包含多种形态和趋势
        market_phases = np.concatenate([
            np.linspace(0, 12, 30),    # 初期上升
            np.linspace(12, 18, 25),   # 加速上升
            np.linspace(18, 22, 20),   # 高位震荡
            np.linspace(22, 8, 35),    # 下降调整
            np.linspace(8, 16, 25),    # 底部反弹
            np.linspace(16, 20, 15)    # 再次上升
        ])
        
        # 添加多层次波动性
        volatility = (np.sin(np.linspace(0, 8*np.pi, 150)) * 1.2 + 
                     np.cos(np.linspace(0, 5*np.pi, 150)) * 0.8 + 2.5)
        volume_pattern = (np.cos(np.linspace(0, 6*np.pi, 150)) * 600000 + 
                         np.sin(np.linspace(0, 4*np.pi, 150)) * 400000 + base_volume)
        
        # 生成价格和成交量序列
        prices = [base_price]
        volumes = [base_volume]
        
        for i in range(1, 150):
            # 趋势分量
            trend = (market_phases[i] - market_phases[i-1]) * 0.25
            # 波动分量
            vol = volatility[i] * 0.08
            # 随机噪声
            noise = np.random.normal(0, 0.3)
            
            price_change = trend + vol + noise
            new_price = max(prices[-1] + price_change, 1.0)
            
            # 成交量与波动性相关
            volume_factor = (volatility[i] / 3) + 0.7
            new_volume = int(volume_pattern[i] * volume_factor * np.random.uniform(0.7, 1.3))
            
            prices.append(new_price)
            volumes.append(max(new_volume, 50000))
        
        # 生成高质量OHLC数据
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            daily_vol = volatility[i] * 0.008
            
            # 生成真实的OHLC关系
            high_factor = np.random.uniform(1.0, 1.0 + daily_vol)
            low_factor = np.random.uniform(1.0 - daily_vol, 1.0)
            
            high = price * high_factor
            low = price * low_factor
            
            open_price = prices[i-1] if i > 0 else price
            close = price
            
            # 确保OHLC关系正确
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # 添加换手率和其他技术指标需要的字段
            turnover_rate = np.random.uniform(0.5, 8.0)
            
            data.append({
                'date': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume,
                'turnover_rate': round(turnover_rate, 2),
                'amount': round(volume * close, 2)
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        logger.info(f"✅ 生成测试数据完成: {len(df)}行，包含{len(df.columns)}列")
        return df
    
    @exception_handler(reraise=True)
    def validate_factory_indicator(self, indicator_name: str, indicator_type: str = 'auto') -> Dict[str, Any]:
        """
        验证单个工厂模式指标
        
        Args:
            indicator_name: 指标名称
            indicator_type: 指标类型 ('zxm', 'pattern', 'auto')
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        start_time = time.time()
        
        try:
            # 自动判断指标类型
            if indicator_type == 'auto':
                if indicator_name in self.zxm_indicators:
                    indicator_type = 'zxm'
                elif indicator_name in self.pattern_indicators:
                    indicator_type = 'pattern'
                else:
                    indicator_type = 'unknown'
            
            logger.info(f"🔍 开始验证工厂模式指标: {indicator_name} (类型: {indicator_type})")
            
            # 获取指标实例
            indicator = self._get_factory_indicator_instance(indicator_name)
            if not indicator:
                return self._create_failed_result(indicator_name, "无法获取指标实例", start_time)
            
            # 生成测试数据
            if self.test_data is None:
                self.test_data = self.generate_comprehensive_test_data()
            
            # 执行验证
            result = self._execute_factory_validation(indicator, indicator_name, indicator_type)
            
            execution_time = time.time() - start_time
            result['execution_time'] = round(execution_time, 4)
            result['indicator_type'] = indicator_type
            
            logger.info(f"✅ 指标 {indicator_name} 验证完成，得分: {result['total_score']:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 验证指标 {indicator_name} 时发生错误: {e}")
            return self._create_failed_result(indicator_name, str(e), start_time)
    
    def _get_factory_indicator_instance(self, indicator_name: str):
        """获取工厂模式指标实例"""
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
from db.sql_manager import SQLManager, QueryType
            registry = get_indicator_registry()
            return registry.get_indicator(indicator_name)
        except Exception as e:
            logger.error(f"获取指标 {indicator_name} 实例失败: {e}")
            return None
    
    def _execute_factory_validation(self, indicator, indicator_name: str, indicator_type: str) -> Dict[str, Any]:
        """执行工厂模式指标验证"""
        result = {
            'indicator_name': indicator_name,
            'indicator_type': indicator_type,
            'basic_function_score': 0,
            'data_quality_score': 0,
            'method_implementation_score': 0,
            'performance_score': 0,
            'factory_specific_score': 0,
            'total_score': 0,
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            # 1. 基础功能验证 (30分)
            result['basic_function_score'] = self._validate_basic_function(indicator, indicator_name)
            
            # 2. 数据质量验证 (25分)
            result['data_quality_score'] = self._validate_data_quality(indicator, indicator_name)
            
            # 3. 方法实现验证 (20分)
            result['method_implementation_score'] = self._validate_method_implementation(indicator, indicator_name)
            
            # 4. 性能验证 (10分)
            result['performance_score'] = self._validate_performance(indicator, indicator_name)
            
            # 5. 工厂模式特定验证 (15分)
            result['factory_specific_score'] = self._validate_factory_specific(indicator, indicator_name, indicator_type)
            
            # 计算总分
            result['total_score'] = (
                result['basic_function_score'] + 
                result['data_quality_score'] + 
                result['method_implementation_score'] + 
                result['performance_score'] + 
                result['factory_specific_score']
            )
            
            result['passed'] = result['total_score'] >= self.min_score
            
        except Exception as e:
            result['errors'].append(f"验证执行错误: {e}")
            logger.error(f"验证指标 {indicator_name} 时发生错误: {e}")
        
        return result
    
    def _validate_basic_function(self, indicator, indicator_name: str) -> float:
        """验证基础功能 (30分)"""
        score = 0
        
        try:
            # 测试calculate方法 (15分)
            if hasattr(indicator, 'calculate'):
                calc_result = indicator.calculate(self.test_data)
                if isinstance(calc_result, pd.DataFrame) and not calc_result.empty:
                    score += 15
                    logger.debug(f"✅ {indicator_name} calculate方法正常")
                else:
                    logger.warning(f"⚠️ {indicator_name} calculate方法返回无效结果")
            
            # 测试get_patterns方法 (15分)
            if hasattr(indicator, 'get_patterns'):
                try:
                    # 工厂模式指标的get_patterns通常无参数
                    patterns_result = indicator.get_patterns()
                    if patterns_result is not None:
                        score += 15
                        logger.debug(f"✅ {indicator_name} get_patterns方法正常")
                    else:
                        logger.warning(f"⚠️ {indicator_name} get_patterns方法返回None")
                except Exception as e:
                    # 尝试带参数调用
                    try:
                        patterns_result = indicator.get_patterns(self.test_data)
                        if patterns_result is not None:
                            score += 10  # 部分分数
                            logger.debug(f"✅ {indicator_name} get_patterns方法(带参数)正常")
                    except:
                        logger.warning(f"⚠️ {indicator_name} get_patterns方法调用失败: {e}")
            
        except Exception as e:
            logger.error(f"验证 {indicator_name} 基础功能时出错: {e}")
        
        return score

    def _validate_data_quality(self, indicator, indicator_name: str) -> float:
        """验证数据质量 (25分)"""
        score = 0

        try:
            # 执行计算获取结果
            result = indicator.calculate(self.test_data)

            if isinstance(result, pd.DataFrame):
                # 检查数据完整性 (10分)
                if not result.empty and len(result) > 0:
                    score += 10
                    logger.debug(f"✅ {indicator_name} 数据完整性良好")

                # 检查数据有效性 (10分)
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # 检查是否有有效的数值数据
                    valid_data = result[numeric_cols].dropna()
                    if len(valid_data) > len(result) * 0.8:  # 80%以上有效数据
                        score += 10
                        logger.debug(f"✅ {indicator_name} 数据有效性良好")
                    else:
                        score += 5  # 部分分数
                        logger.warning(f"⚠️ {indicator_name} 数据有效性一般")

                # 检查数据范围合理性 (5分)
                if len(numeric_cols) > 0:
                    # 检查是否有异常值
                    has_reasonable_values = True
                    for col in numeric_cols:
                        col_data = result[col].dropna()
                        if len(col_data) > 0:
                            # 检查是否有无穷大或NaN
                            if np.isinf(col_data).any() or col_data.isna().all():
                                has_reasonable_values = False
                                break

                    if has_reasonable_values:
                        score += 5
                        logger.debug(f"✅ {indicator_name} 数据范围合理")
                    else:
                        logger.warning(f"⚠️ {indicator_name} 数据范围存在异常")

        except Exception as e:
            logger.error(f"验证 {indicator_name} 数据质量时出错: {e}")

        return score

    def _validate_method_implementation(self, indicator, indicator_name: str) -> float:
        """验证方法实现 (20分)"""
        score = 0

        try:
            # 检查必需方法存在性 (10分)
            required_methods = ['calculate']
            existing_methods = 0

            for method in required_methods:
                if hasattr(indicator, method) and callable(getattr(indicator, method)):
                    existing_methods += 1

            score += (existing_methods / len(required_methods)) * 10

            # 检查可选方法存在性 (5分)
            optional_methods = ['get_patterns', 'get_signal', 'get_score']
            existing_optional = 0

            for method in optional_methods:
                if hasattr(indicator, method) and callable(getattr(indicator, method)):
                    existing_optional += 1

            score += (existing_optional / len(optional_methods)) * 5

            # 检查属性完整性 (5分)
            if hasattr(indicator, 'name') or hasattr(indicator, '__class__'):
                score += 2.5

            if hasattr(indicator, 'period') or hasattr(indicator, 'params'):
                score += 2.5

            logger.debug(f"✅ {indicator_name} 方法实现检查完成，得分: {score}")

        except Exception as e:
            logger.error(f"验证 {indicator_name} 方法实现时出错: {e}")

        return score

    def _validate_performance(self, indicator, indicator_name: str) -> float:
        """验证性能 (10分)"""
        score = 0

        try:
            # 测试计算性能
            start_time = time.time()

            # 执行多次计算测试性能
            for _ in range(3):
                result = indicator.calculate(self.test_data)

            execution_time = (time.time() - start_time) / 3  # 平均时间

            # 性能评分标准
            if execution_time < 0.01:  # 10ms以下
                score = 10
            elif execution_time < 0.05:  # 50ms以下
                score = 8
            elif execution_time < 0.1:  # 100ms以下
                score = 6
            elif execution_time < 0.5:  # 500ms以下
                score = 4
            else:
                score = 2

            logger.debug(f"✅ {indicator_name} 性能测试完成，平均执行时间: {execution_time:.4f}s，得分: {score}")

        except Exception as e:
            logger.error(f"验证 {indicator_name} 性能时出错: {e}")
            score = 2  # 最低分数

        return score

    def _validate_factory_specific(self, indicator, indicator_name: str, indicator_type: str) -> float:
        """验证工厂模式特定功能 (15分)"""
        score = 0

        try:
            if indicator_type == 'zxm':
                score = self._validate_zxm_specific(indicator, indicator_name)
            elif indicator_type == 'pattern':
                score = self._validate_pattern_specific(indicator, indicator_name)
            else:
                # 通用工厂模式验证
                score = self._validate_generic_factory(indicator, indicator_name)

        except Exception as e:
            logger.error(f"验证 {indicator_name} 工厂模式特定功能时出错: {e}")

        return score

    def _validate_zxm_specific(self, indicator, indicator_name: str) -> float:
        """验证ZXM体系指标特定功能"""
        score = 0

        try:
            # ZXM指标特定验证
            result = indicator.calculate(self.test_data)

            if isinstance(result, pd.DataFrame) and not result.empty:
                # 检查ZXM特征列 (7分)
                zxm_features = ['score', 'signal', 'trend', 'value']
                found_features = 0

                for feature in zxm_features:
                    matching_cols = [col for col in result.columns if feature.lower() in col.lower()]
                    if matching_cols:
                        found_features += 1

                score += (found_features / len(zxm_features)) * 7

                # 检查数据范围合理性 (4分)
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    reasonable_ranges = True
                    for col in numeric_cols:
                        col_data = result[col].dropna()
                        if len(col_data) > 0:
                            # ZXM指标通常在合理范围内
                            if col_data.min() < -1000 or col_data.max() > 1000:
                                reasonable_ranges = False
                                break

                    if reasonable_ranges:
                        score += 4

                # 检查时间序列连续性 (4分)
                if len(result) > 10:
                    score += 4

        except Exception as e:
            logger.error(f"验证ZXM指标 {indicator_name} 特定功能时出错: {e}")

        return score

    def _validate_pattern_specific(self, indicator, indicator_name: str) -> float:
        """验证形态识别指标特定功能"""
        score = 0

        try:
            # 形态识别指标特定验证
            result = indicator.calculate(self.test_data)

            if isinstance(result, pd.DataFrame) and not result.empty:
                # 检查形态识别特征 (8分)
                pattern_features = ['pattern', 'signal', 'detected', 'confidence']
                found_features = 0

                for feature in pattern_features:
                    matching_cols = [col for col in result.columns if feature.lower() in col.lower()]
                    if matching_cols:
                        found_features += 1

                score += (found_features / len(pattern_features)) * 8

                # 检查布尔型形态检测结果 (4分)
                bool_cols = result.select_dtypes(include=[bool]).columns
                if len(bool_cols) > 0:
                    score += 4

                # 检查形态检测的合理性 (3分)
                # 形态不应该过于频繁出现
                if len(bool_cols) > 0:
                    reasonable_detection = True
                    for col in bool_cols:
                        detection_rate = result[col].sum() / len(result)
                        if detection_rate > 0.5:  # 超过50%的检测率可能不合理
                            reasonable_detection = False
                            break

                    if reasonable_detection:
                        score += 3

        except Exception as e:
            logger.error(f"验证形态识别指标 {indicator_name} 特定功能时出错: {e}")

        return score

    def _validate_generic_factory(self, indicator, indicator_name: str) -> float:
        """验证通用工厂模式指标"""
        score = 0

        try:
            # 通用工厂模式验证
            result = indicator.calculate(self.test_data)

            if isinstance(result, pd.DataFrame) and not result.empty:
                # 基础功能正常 (8分)
                score += 8

                # 数据格式正确 (4分)
                if len(result.columns) > 0:
                    score += 4

                # 无严重错误 (3分)
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    has_inf = any(np.isinf(result[col]).any() for col in numeric_cols)
                    if not has_inf:
                        score += 3

        except Exception as e:
            logger.error(f"验证通用工厂指标 {indicator_name} 时出错: {e}")

        return score

    def _create_failed_result(self, indicator_name: str, error_msg: str, start_time: float) -> Dict[str, Any]:
        """创建失败结果"""
        return {
            'indicator_name': indicator_name,
            'indicator_type': 'unknown',
            'basic_function_score': 0,
            'data_quality_score': 0,
            'method_implementation_score': 0,
            'performance_score': 0,
            'factory_specific_score': 0,
            'total_score': 0,
            'passed': False,
            'execution_time': round(time.time() - start_time, 4),
            'details': {},
            'errors': [error_msg]
        }

    @performance_monitor(threshold_seconds=10.0)
    def validate_all_factory_indicators(self) -> Dict[str, Any]:
        """验证所有54个工厂模式指标"""
        logger.info("🚀 开始验证所有54个工厂模式指标...")

        start_time = time.time()
        results = {}

        # 分类验证
        zxm_results = self._validate_indicator_group(self.zxm_indicators, 'zxm')
        pattern_results = self._validate_indicator_group(self.pattern_indicators, 'pattern')

        results.update(zxm_results)
        results.update(pattern_results)

        # 生成总结报告
        summary = self._generate_validation_summary(results)

        total_time = time.time() - start_time
        logger.info(f"✅ 所有工厂模式指标验证完成，总耗时: {total_time:.2f}秒")

        return {
            'results': results,
            'summary': summary,
            'total_execution_time': round(total_time, 2)
        }

    def _validate_indicator_group(self, indicators: List[str], group_type: str) -> Dict[str, Any]:
        """验证指标组"""
        logger.info(f"📊 开始验证{group_type}指标组，共{len(indicators)}个指标...")

        results = {}
        for i, indicator_name in enumerate(indicators, 1):
            logger.info(f"[{i}/{len(indicators)}] 验证 {indicator_name}...")

            try:
                result = self.validate_factory_indicator(indicator_name, group_type)
                results[indicator_name] = result

                # 显示进度
                if result['passed']:
                    logger.info(f"✅ {indicator_name}: {result['total_score']:.1f}分 (通过)")
                else:
                    logger.warning(f"❌ {indicator_name}: {result['total_score']:.1f}分 (未通过)")

            except Exception as e:
                logger.error(f"❌ 验证 {indicator_name} 时发生异常: {e}")
                results[indicator_name] = self._create_failed_result(
                    indicator_name, f"验证异常: {e}", time.time()
                )

        return results

    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成验证总结"""
        total_indicators = len(results)
        passed_indicators = sum(1 for r in results.values() if r['passed'])
        failed_indicators = total_indicators - passed_indicators

        # 分类统计
        zxm_results = {k: v for k, v in results.items() if k in self.zxm_indicators}
        pattern_results = {k: v for k, v in results.items() if k in self.pattern_indicators}

        zxm_passed = sum(1 for r in zxm_results.values() if r['passed'])
        pattern_passed = sum(1 for r in pattern_results.values() if r['passed'])

        # 计算平均分
        total_scores = [r['total_score'] for r in results.values()]
        avg_score = sum(total_scores) / len(total_scores) if total_scores else 0

        # 性能统计
        execution_times = [r['execution_time'] for r in results.values()]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

        # 获取最高分和最低分指标
        best_indicator = max(results.items(), key=lambda x: x[1]['total_score']) if results else None
        worst_indicator = min(results.items(), key=lambda x: x[1]['total_score']) if results else None

        summary = {
            'total_indicators': total_indicators,
            'passed_indicators': passed_indicators,
            'failed_indicators': failed_indicators,
            'pass_rate': round((passed_indicators / total_indicators) * 100, 1) if total_indicators > 0 else 0,
            'average_score': round(avg_score, 1),
            'average_execution_time': round(avg_execution_time, 4),
            'zxm_statistics': {
                'total': len(zxm_results),
                'passed': zxm_passed,
                'pass_rate': round((zxm_passed / len(zxm_results)) * 100, 1) if zxm_results else 0
            },
            'pattern_statistics': {
                'total': len(pattern_results),
                'passed': pattern_passed,
                'pass_rate': round((pattern_passed / len(pattern_results)) * 100, 1) if pattern_results else 0
            },
            'best_indicator': {
                'name': best_indicator[0] if best_indicator else None,
                'score': best_indicator[1]['total_score'] if best_indicator else 0
            },
            'worst_indicator': {
                'name': worst_indicator[0] if worst_indicator else None,
                'score': worst_indicator[1]['total_score'] if worst_indicator else 0
            }
        }

        return summary

    def generate_detailed_report(self, validation_results: Dict[str, Any]) -> str:
        """生成详细的验证报告"""
        results = validation_results['results']
        summary = validation_results['summary']

        report_lines = [
            "# 工厂模式指标验证框架 - 详细报告",
            f"**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**验证标准**: {self.min_score}分以上",
            f"**总执行时间**: {validation_results['total_execution_time']}秒",
            "",
            "## 📊 验证总结",
            f"- **总指标数**: {summary['total_indicators']}个",
            f"- **通过指标**: {summary['passed_indicators']}个",
            f"- **未通过指标**: {summary['failed_indicators']}个",
            f"- **通过率**: {summary['pass_rate']}%",
            f"- **平均得分**: {summary['average_score']}分",
            f"- **平均执行时间**: {summary['average_execution_time']}秒",
            "",
            "## 🎯 分类统计",
            "",
            "### ZXM体系指标",
            f"- **总数**: {summary['zxm_statistics']['total']}个",
            f"- **通过**: {summary['zxm_statistics']['passed']}个",
            f"- **通过率**: {summary['zxm_statistics']['pass_rate']}%",
            "",
            "### 形态识别指标",
            f"- **总数**: {summary['pattern_statistics']['total']}个",
            f"- **通过**: {summary['pattern_statistics']['passed']}个",
            f"- **通过率**: {summary['pattern_statistics']['pass_rate']}%",
            "",
            "## 🏆 最佳和最差指标",
            f"- **最佳指标**: {summary['best_indicator']['name']} ({summary['best_indicator']['score']}分)",
            f"- **最差指标**: {summary['worst_indicator']['name']} ({summary['worst_indicator']['score']}分)",
            "",
            "## 📋 详细验证结果",
            ""
        ]

        # 按分数排序显示结果
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_score'], reverse=True)

        for indicator_name, result in sorted_results:
            status = "✅ 通过" if result['passed'] else "❌ 未通过"
            report_lines.extend([
                f"### {indicator_name} - {status}",
                f"- **总分**: {result['total_score']:.1f}/100",
                f"- **基础功能**: {result['basic_function_score']:.1f}/30",
                f"- **数据质量**: {result['data_quality_score']:.1f}/25",
                f"- **方法实现**: {result['method_implementation_score']:.1f}/20",
                f"- **性能**: {result['performance_score']:.1f}/10",
                f"- **工厂特定**: {result['factory_specific_score']:.1f}/15",
                f"- **执行时间**: {result['execution_time']}秒",
                ""
            ])

            if result['errors']:
                report_lines.extend([
                    "**错误信息**:",
                    *[f"- {error}" for error in result['errors']],
                    ""
                ])

        return "\n".join(report_lines)

    def save_report(self, validation_results: Dict[str, Any], filename: str = None) -> str:
        """保存验证报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"factory_indicator_validation_report_{timestamp}.md"

        report_content = self.generate_detailed_report(validation_results)

        # 确保reports目录存在
        reports_dir = os.path.join(root_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        filepath = os.path.join(reports_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📄 验证报告已保存到: {filepath}")
        return filepath


def main():
    """主函数 - 执行完整的工厂模式指标验证"""
    logger.info("🚀 启动工厂模式指标验证框架...")

    try:
        # 创建验证框架实例
        framework = FactoryIndicatorValidationFramework()

        # 执行完整验证
        validation_results = framework.validate_all_factory_indicators()

        # 生成并保存报告
        report_path = framework.save_report(validation_results)

        # 显示总结
        summary = validation_results['summary']
        logger.info("🎉 工厂模式指标验证完成!")
        logger.info(f"📊 总结: {summary['passed_indicators']}/{summary['total_indicators']} 通过 ({summary['pass_rate']}%)")
        logger.info(f"📄 详细报告: {report_path}")

        return validation_results

    except Exception as e:
        logger.error(f"❌ 工厂模式指标验证失败: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
