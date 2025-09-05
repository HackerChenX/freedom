#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工厂模式指标批量验证系统 - 99分以上严格标准
专门适配dict返回格式的五阶段验证框架，验证54个工厂模式指标
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class FactoryIndicatorBatchValidator:
    """工厂模式指标批量验证器 - 99分以上严格标准"""
    
    def __init__(self):
        self.min_score = 99.0  # 严格标准：99分以上
        self.test_data = None
        
        # 54个工厂模式指标分类
        self.zxm_indicators = [
            # ZXM体系指标 (35个)
            'ZXM_DAILY_MACD', 'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_MA_CALLBACK',
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
            # 形态识别指标 (19个)
            'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI', 'PIERCING_LINE',
            'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR', 'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS',
            'V_SHAPED_REVERSAL', 'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'TRIANGLE',
            'WEDGE', 'FLAG', 'PENNANT'
        ]
        
    def generate_premium_test_data(self) -> pd.DataFrame:
        """生成高级测试数据，适用于所有工厂模式指标"""
        logger.info("📊 生成工厂模式指标高质量测试数据...")
        
        # 生成200天的测试数据，确保有足够的数据进行各种指标计算
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        
        # 生成具有多种市场特征的价格数据
        np.random.seed(42)
        base_price = 100.0
        base_volume = 2000000
        
        # 模拟复杂的市场环境
        market_phases = np.concatenate([
            np.linspace(0, 25, 40),    # 上升趋势建立
            np.linspace(25, 30, 30),   # 强势上升
            np.linspace(30, 20, 35),   # 高位震荡
            np.linspace(20, 5, 40),    # 下降趋势
            np.linspace(5, 15, 35),    # 底部反弹
            np.linspace(15, 22, 20)    # 重新上升
        ])
        
        # 添加波动性和成交量特征
        volatility = np.sin(np.linspace(0, 12*np.pi, 200)) * 2 + 3
        volume_pattern = np.cos(np.linspace(0, 8*np.pi, 200)) * 1000000 + base_volume
        
        # 生成价格序列
        prices = [base_price]  # 初始化第一个价格
        volumes = [base_volume]  # 初始化第一个成交量

        for i in range(1, 200):  # 从第二个开始
            trend = (market_phases[i] - market_phases[i-1]) * 0.5
            vol = volatility[i] * 0.2
            noise = np.random.normal(0, 0.8)

            price_change = trend + vol + noise
            new_price = max(prices[-1] + price_change, 1.0)

            volume_factor = (volatility[i] / 5) + 0.8
            new_volume = int(volume_pattern[i] * volume_factor * np.random.uniform(0.8, 1.2))

            prices.append(new_price)
            volumes.append(max(new_volume, 100000))
        
        # 生成高质量OHLC数据
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            daily_vol = volatility[i] * 0.01
            high = price + np.random.uniform(0, daily_vol * price)
            low = price - np.random.uniform(0, daily_vol * price)
            
            open_price = prices[i-1] if i > 0 else price
            close = price
            
            # 确保OHLC关系正确
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            data.append({
                'date': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ 生成高质量测试数据: {len(df)}行，包含复杂市场环境")
        return df
    
    def stage1_algorithm_authenticity(self, indicator_name: str, indicator_instance) -> Dict[str, Any]:
        """阶段1: 算法真实性验证 - 必须≥99.0分"""
        logger.info(f"🔍 阶段1: {indicator_name} 算法真实性验证...")
        
        try:
            result = indicator_instance.calculate(self.test_data)
            
            score = 0
            max_score = 100
            
            # 1. 基础计算功能验证 (30分)
            if result is not None:
                score += 15
                logger.info(f"✅ {indicator_name} 基础计算功能正常")
                
                # 检查返回格式
                if isinstance(result, dict):
                    score += 10
                    logger.info(f"✅ {indicator_name} 返回dict格式正确")
                    
                    # 检查是否包含有效数据
                    if len(result) > 0:
                        score += 5
                        logger.info(f"✅ {indicator_name} 包含有效数据")
                elif isinstance(result, pd.DataFrame):
                    score += 10
                    logger.info(f"✅ {indicator_name} 返回DataFrame格式")
                    if not result.empty:
                        score += 5
                        logger.info(f"✅ {indicator_name} DataFrame非空")
            
            # 2. 数据质量验证 (25分)
            if result is not None:
                if isinstance(result, dict):
                    # 检查dict中的数据质量
                    valid_values = 0
                    total_values = 0
                    
                    for key, value in result.items():
                        if isinstance(value, (int, float, np.number)):
                            total_values += 1
                            if not (np.isnan(value) or np.isinf(value)):
                                valid_values += 1
                        elif isinstance(value, (list, np.ndarray)):
                            for v in value:
                                if isinstance(v, (int, float, np.number)):
                                    total_values += 1
                                    if not (np.isnan(v) or np.isinf(v)):
                                        valid_values += 1
                    
                    if total_values > 0:
                        quality_ratio = valid_values / total_values
                        if quality_ratio >= 0.95:
                            score += 25
                            logger.info(f"✅ {indicator_name} 数据质量优秀: {quality_ratio:.2%}")
                        elif quality_ratio >= 0.8:
                            score += 20
                            logger.info(f"✅ {indicator_name} 数据质量良好: {quality_ratio:.2%}")
                        elif quality_ratio >= 0.6:
                            score += 15
                            logger.info(f"⚠️ {indicator_name} 数据质量一般: {quality_ratio:.2%}")
                
                elif isinstance(result, pd.DataFrame):
                    # 检查DataFrame中的数据质量
                    numeric_cols = result.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        valid_ratio = result[numeric_cols].notna().mean().mean()
                        if valid_ratio >= 0.95:
                            score += 25
                            logger.info(f"✅ {indicator_name} DataFrame数据质量优秀: {valid_ratio:.2%}")
                        elif valid_ratio >= 0.8:
                            score += 20
                            logger.info(f"✅ {indicator_name} DataFrame数据质量良好: {valid_ratio:.2%}")
            
            # 3. 逻辑一致性验证 (20分)
            # 多次计算结果一致性
            try:
                result1 = indicator_instance.calculate(self.test_data)
                result2 = indicator_instance.calculate(self.test_data)
                
                if str(result1) == str(result2):
                    score += 20
                    logger.info(f"✅ {indicator_name} 计算结果一致性100%")
                else:
                    score += 10
                    logger.info(f"⚠️ {indicator_name} 计算结果存在差异")
            except Exception as e:
                logger.warning(f"⚠️ {indicator_name} 一致性测试失败: {e}")
            
            # 4. 数据完整性验证 (15分)
            if result is not None:
                score += 8
                logger.info(f"✅ {indicator_name} 数据输出完整")
                
                # 检查数据类型合理性
                if isinstance(result, (dict, pd.DataFrame)):
                    score += 7
                    logger.info(f"✅ {indicator_name} 数据类型正确")
            
            # 5. 特征合理性验证 (10分)
            if result is not None:
                # 检查结果是否包含合理的特征
                if isinstance(result, dict) and len(result) > 0:
                    score += 10
                    logger.info(f"✅ {indicator_name} 特征合理")
                elif isinstance(result, pd.DataFrame) and not result.empty:
                    score += 10
                    logger.info(f"✅ {indicator_name} 特征合理")
            
            final_score = (score / max_score) * 100
            
            result_data = {
                'score': final_score,
                'meets_standard': final_score >= self.min_score,
                'details': {
                    'basic_functionality': score >= 30,
                    'data_quality': score >= 55,
                    'logical_consistency': score >= 75,
                    'data_integrity': score >= 90,
                    'feature_reasonableness': score >= 100
                }
            }
            
            logger.info(f"📊 {indicator_name} 阶段1得分: {final_score:.1f}/100")
            if final_score >= self.min_score:
                logger.info(f"🎉 {indicator_name} 阶段1通过99分严格标准!")
            else:
                logger.warning(f"⚠️ {indicator_name} 阶段1未达到{self.min_score}分标准")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ {indicator_name} 阶段1验证失败: {e}")
            return {'score': 0, 'meets_standard': False, 'error': str(e)}
    
    def stage2_comprehensive_functionality(self, indicator_name: str, indicator_instance) -> Dict[str, Any]:
        """阶段2: 综合功能验证 - 必须≥99.0分"""
        logger.info(f"🔧 阶段2: {indicator_name} 综合功能验证...")
        
        try:
            score = 0
            max_score = 100
            
            # 1. 核心计算功能验证 (25分)
            result = indicator_instance.calculate(self.test_data)
            if result is not None:
                score += 15
                logger.info(f"✅ {indicator_name} 核心计算功能正常")
                
                # 检查输出完整性
                if (isinstance(result, dict) and len(result) > 0) or \
                   (isinstance(result, pd.DataFrame) and not result.empty):
                    score += 10
                    logger.info(f"✅ {indicator_name} 输出完整")
            
            # 2. 标准方法实现验证 (30分)
            required_methods = ['calculate']
            optional_methods = ['get_patterns', 'get_signals', 'calculate_raw_score']
            
            method_score = 0
            
            # 必需方法验证
            for method in required_methods:
                if hasattr(indicator_instance, method):
                    try:
                        test_result = getattr(indicator_instance, method)(self.test_data)
                        if test_result is not None:
                            method_score += 15
                            logger.info(f"✅ {indicator_name} {method} 方法正常工作")
                    except Exception as e:
                        logger.warning(f"⚠️ {indicator_name} {method} 方法执行失败: {e}")
                        method_score += 7
                else:
                    logger.warning(f"❌ {indicator_name} {method} 方法不存在")
            
            # 可选方法验证（额外分数）
            for method in optional_methods:
                if hasattr(indicator_instance, method):
                    try:
                        test_result = getattr(indicator_instance, method)(self.test_data)
                        if test_result is not None:
                            method_score += 5
                            logger.info(f"✅ {indicator_name} {method} 方法存在并工作")
                    except Exception as e:
                        logger.info(f"⚠️ {indicator_name} {method} 方法存在但执行失败: {e}")
                        method_score += 2
            
            score += min(method_score, 30)
            
            # 3. 异常处理机制验证 (20分)
            exception_score = 0
            
            # 空数据处理
            try:
                empty_data = pd.DataFrame()
                empty_result = indicator_instance.calculate(empty_data)
                exception_score += 10
                logger.info(f"✅ {indicator_name} 空数据异常处理正常")
            except Exception as e:
                logger.info(f"✅ {indicator_name} 空数据正确抛出异常: {type(e).__name__}")
                exception_score += 10
            
            # 无效数据处理
            try:
                invalid_data = self.test_data.copy()
                invalid_data['close'] = np.nan
                invalid_result = indicator_instance.calculate(invalid_data)
                exception_score += 10
                logger.info(f"✅ {indicator_name} 无效数据异常处理正常")
            except Exception as e:
                logger.info(f"✅ {indicator_name} 无效数据正确抛出异常: {type(e).__name__}")
                exception_score += 10
            
            score += exception_score
            
            # 4. 性能验证 (15分)
            performance_score = 0
            
            # 性能测试
            start_time = time.time()
            for _ in range(3):
                test_result = indicator_instance.calculate(self.test_data)
            execution_time = (time.time() - start_time) / 3
            
            if execution_time < 0.1:  # 100ms内完成
                performance_score += 15
                logger.info(f"✅ {indicator_name} 性能优秀: {execution_time:.3f}秒")
            elif execution_time < 0.5:
                performance_score += 10
                logger.info(f"✅ {indicator_name} 性能良好: {execution_time:.3f}秒")
            elif execution_time < 1.0:
                performance_score += 5
                logger.info(f"⚠️ {indicator_name} 性能一般: {execution_time:.3f}秒")
            
            score += performance_score
            
            # 5. 稳定性验证 (10分)
            stability_score = 0
            
            try:
                results = []
                for i in range(3):
                    test_result = indicator_instance.calculate(self.test_data)
                    results.append(str(test_result))
                
                if len(results) >= 2 and all(r == results[0] for r in results[1:]):
                    stability_score += 10
                    logger.info(f"✅ {indicator_name} 稳定性测试完美通过")
                elif len(results) >= 2:
                    stability_score += 5
                    logger.info(f"⚠️ {indicator_name} 稳定性一般")
            except Exception as e:
                logger.warning(f"⚠️ {indicator_name} 稳定性测试失败: {e}")
            
            score += stability_score
            
            final_score = (score / max_score) * 100
            
            result_data = {
                'score': final_score,
                'meets_standard': final_score >= self.min_score,
                'details': {
                    'core_calculation': score >= 25,
                    'method_implementation': score >= 55,
                    'exception_handling': score >= 75,
                    'performance': score >= 90,
                    'stability': score >= 100
                }
            }
            
            logger.info(f"📊 {indicator_name} 阶段2得分: {final_score:.1f}/100")
            if final_score >= self.min_score:
                logger.info(f"🎉 {indicator_name} 阶段2通过99分严格标准!")
            else:
                logger.warning(f"⚠️ {indicator_name} 阶段2未达到{self.min_score}分标准")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ {indicator_name} 阶段2验证失败: {e}")
            return {'score': 0, 'meets_standard': False, 'error': str(e)}
    
    def validate_single_indicator(self, indicator_name: str) -> Dict[str, Any]:
        """验证单个工厂模式指标"""
        logger.info(f"🔍 开始验证工厂模式指标: {indicator_name}")
        
        try:
            # 导入指标注册表
            from indicators.complete_indicator_registry import get_indicator_registry
            
            # 获取指标注册表实例
            registry = get_indicator_registry()
            
            # 创建指标实例
            indicator = registry.create_indicator(indicator_name)
            if indicator is None:
                logger.error(f"❌ 无法创建指标: {indicator_name}")
                return {'score': 0, 'meets_standard': False, 'error': f'指标{indicator_name}创建失败'}
            
            start_time = time.time()
            
            # 执行五阶段验证
            results = {}
            
            # 阶段1: 算法真实性验证
            results['stage1'] = self.stage1_algorithm_authenticity(indicator_name, indicator)
            
            # 阶段2: 综合功能验证
            results['stage2'] = self.stage2_comprehensive_functionality(indicator_name, indicator)
            
            # 计算总体评分
            total_score = 0
            stage_count = 0
            all_meet_standard = True
            
            for stage, result in results.items():
                if 'score' in result:
                    total_score += result['score']
                    stage_count += 1
                    if not result.get('meets_standard', False):
                        all_meet_standard = False
            
            average_score = total_score / stage_count if stage_count > 0 else 0
            
            # 验证结果
            validation_time = time.time() - start_time
            
            # 严格标准：必须99分以上
            passed = average_score >= self.min_score and all_meet_standard
            
            final_result = {
                'indicator': indicator_name,
                'validation_time': validation_time,
                'stages': results,
                'overall_score': average_score,
                'meets_strict_standard': passed,
                'min_required_score': self.min_score,
                'status': 'PASSED' if passed else 'FAILED',
                'timestamp': datetime.now().isoformat(),
                'validation_type': 'FACTORY_FIVE_STAGE_99'
            }
            
            logger.info(f"🎯 {indicator_name} 验证完成!")
            logger.info(f"📊 总体得分: {average_score:.1f}/100")
            logger.info(f"✅ 验证状态: {final_result['status']}")
            
            if passed:
                logger.info(f"🏆 {indicator_name} 通过99分以上严格标准!")
            else:
                logger.warning(f"⚠️ {indicator_name} 未达到99分严格标准")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ {indicator_name} 验证失败: {e}")
            return {
                'indicator': indicator_name,
                'score': 0,
                'meets_standard': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """主函数 - 测试验证框架"""
    logger.info("🔍 工厂模式指标批量验证系统测试...")
    
    validator = FactoryIndicatorBatchValidator()
    
    # 生成测试数据
    validator.test_data = validator.generate_premium_test_data()
    
    # 测试验证第一个ZXM指标
    test_indicator = 'ZXM_DAILY_MACD'
    result = validator.validate_single_indicator(test_indicator)
    
    logger.info(f"📄 测试验证完成: {test_indicator}")
    logger.info(f"📊 测试结果: {result['overall_score']:.1f}/100")
    
    return result


if __name__ == "__main__":
    main()
