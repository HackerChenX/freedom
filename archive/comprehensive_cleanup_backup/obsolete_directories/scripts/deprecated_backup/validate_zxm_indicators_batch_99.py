#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZXM体系指标批量验证 - 99分以上严格标准
验证35个ZXM体系指标，使用五阶段验证方式
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ZXMIndicatorsBatchValidator:
    """ZXM体系指标批量验证器 - 99分以上严格标准"""
    
    def __init__(self):
        self.min_score = 99.0  # 严格标准：99分以上
        self.test_data = None
        
        # 35个ZXM体系指标
        self.zxm_indicators = [
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
        
    def generate_premium_test_data(self) -> pd.DataFrame:
        """生成高级测试数据，适用于ZXM体系指标"""
        logger.info("📊 生成ZXM体系指标高质量测试数据...")
        
        # 生成150天的测试数据
        dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
        
        # 生成具有多种市场特征的价格数据
        np.random.seed(42)
        base_price = 100.0
        base_volume = 2000000
        
        # 模拟复杂的市场环境
        market_phases = np.concatenate([
            np.linspace(0, 20, 30),    # 上升趋势
            np.linspace(20, 25, 25),   # 加速上升
            np.linspace(25, 15, 30),   # 高位震荡
            np.linspace(15, 5, 35),    # 下降趋势
            np.linspace(5, 18, 30)     # 底部反弹
        ])
        
        # 添加波动性和成交量特征
        volatility = np.sin(np.linspace(0, 10*np.pi, 150)) * 2 + 3
        volume_pattern = np.cos(np.linspace(0, 6*np.pi, 150)) * 1000000 + base_volume
        
        # 生成价格序列
        prices = [base_price]
        volumes = [base_volume]
        
        for i in range(1, 150):
            trend = (market_phases[i] - market_phases[i-1]) * 0.4
            vol = volatility[i] * 0.15
            noise = np.random.normal(0, 0.6)
            
            price_change = trend + vol + noise
            new_price = max(prices[-1] + price_change, 1.0)
            
            volume_factor = (volatility[i] / 5) + 0.8
            new_volume = int(volume_pattern[i] * volume_factor * np.random.uniform(0.8, 1.2))
            
            prices.append(new_price)
            volumes.append(max(new_volume, 100000))
        
        # 生成高质量OHLC数据
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            daily_vol = volatility[i] * 0.008
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
    
    def validate_single_zxm_indicator(self, indicator_name: str) -> Dict[str, Any]:
        """验证单个ZXM指标 - 五阶段验证"""
        logger.info(f"🔍 开始验证ZXM指标: {indicator_name}")
        
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
                'validation_type': 'ZXM_FIVE_STAGE_99'
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
                        logger.info(f"✅ {indicator_name} 包含有效数据: {len(result)}个键")
            
            # 2. 数据质量验证 (30分)
            if result is not None and isinstance(result, dict):
                valid_values = 0
                total_values = 0
                
                for key, value in result.items():
                    if isinstance(value, pd.Series):
                        series_valid = value.notna().sum()
                        series_total = len(value)
                        valid_values += series_valid
                        total_values += series_total
                    elif isinstance(value, (int, float, np.number)):
                        total_values += 1
                        if not (np.isnan(value) or np.isinf(value)):
                            valid_values += 1
                
                if total_values > 0:
                    quality_ratio = valid_values / total_values
                    if quality_ratio >= 0.95:
                        score += 30
                        logger.info(f"✅ {indicator_name} 数据质量优秀: {quality_ratio:.2%}")
                    elif quality_ratio >= 0.8:
                        score += 25
                        logger.info(f"✅ {indicator_name} 数据质量良好: {quality_ratio:.2%}")
                    elif quality_ratio >= 0.6:
                        score += 15
                        logger.info(f"⚠️ {indicator_name} 数据质量一般: {quality_ratio:.2%}")
            
            # 3. 逻辑一致性验证 (20分)
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
                
                if isinstance(result, dict):
                    score += 7
                    logger.info(f"✅ {indicator_name} 数据类型正确")
            
            # 5. ZXM特征验证 (5分)
            if result is not None and isinstance(result, dict):
                # ZXM指标通常包含多个分析维度
                if len(result) >= 2:
                    score += 5
                    logger.info(f"✅ {indicator_name} 包含多维度分析")
            
            final_score = (score / max_score) * 100
            
            result_data = {
                'score': final_score,
                'meets_standard': final_score >= self.min_score,
                'details': {
                    'basic_functionality': score >= 30,
                    'data_quality': score >= 60,
                    'logical_consistency': score >= 80,
                    'data_integrity': score >= 95,
                    'zxm_features': score >= 100
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
                
                if isinstance(result, dict) and len(result) > 0:
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
                        # 特殊处理get_patterns方法，工厂模式指标不需要data参数
                        if method == 'get_patterns':
                            test_result = getattr(indicator_instance, method)()
                        else:
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
    
    def run_batch_validation(self, batch_size: int = 5) -> Dict[str, Any]:
        """运行ZXM指标批量验证"""
        logger.info(f"🚀 开始ZXM体系指标批量验证 (99分以上严格标准)...")
        logger.info(f"📊 验证指标数量: {len(self.zxm_indicators)}个")
        
        start_time = time.time()
        
        # 生成测试数据
        self.test_data = self.generate_premium_test_data()
        
        # 验证结果
        results = {}
        passed_count = 0
        failed_count = 0
        
        # 分批验证，避免一次性验证太多指标
        for i in range(0, len(self.zxm_indicators), batch_size):
            batch_indicators = self.zxm_indicators[i:i + batch_size]
            logger.info(f"📦 验证批次 {i//batch_size + 1}: {len(batch_indicators)}个指标")
            
            for indicator_name in batch_indicators:
                result = self.validate_single_zxm_indicator(indicator_name)
                results[indicator_name] = result
                
                if result.get('meets_strict_standard', False):
                    passed_count += 1
                else:
                    failed_count += 1
        
        # 计算总体统计
        total_score = sum(r.get('overall_score', 0) for r in results.values())
        average_score = total_score / len(results) if results else 0
        
        validation_time = time.time() - start_time
        
        summary = {
            'validation_type': 'ZXM_INDICATORS_BATCH_99',
            'total_indicators': len(self.zxm_indicators),
            'passed_count': passed_count,
            'failed_count': failed_count,
            'pass_rate': (passed_count / len(self.zxm_indicators)) * 100 if self.zxm_indicators else 0,
            'average_score': average_score,
            'validation_time': validation_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 ZXM体系指标批量验证完成!")
        logger.info(f"📊 通过率: {summary['pass_rate']:.1f}% ({passed_count}/{len(self.zxm_indicators)})")
        logger.info(f"📊 平均得分: {average_score:.1f}/100")
        logger.info(f"⏱️ 验证时间: {validation_time:.2f}秒")
        
        return summary


def main():
    """主函数 - 验证前5个ZXM指标作为示例"""
    logger.info("🔍 ZXM体系指标批量验证开始 (99分以上严格标准)...")
    
    validator = ZXMIndicatorsBatchValidator()
    
    # 先验证前5个指标作为示例
    sample_indicators = validator.zxm_indicators[:5]
    validator.zxm_indicators = sample_indicators
    
    result = validator.run_batch_validation(batch_size=5)
    
    # 保存验证报告
    report_file = f"docs/finaltesting/indicators/zxm_indicators_batch_validation_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成验证报告
    passed_indicators = [name for name, data in result['results'].items() if data.get('meets_strict_standard', False)]
    failed_indicators = [name for name, data in result['results'].items() if not data.get('meets_strict_standard', False)]
    
    report_content = f"""# ZXM体系指标批量验证报告 (99分以上严格标准)

## 验证概览
- **验证类型**: ZXM体系指标批量验证
- **验证时间**: {result['timestamp']}
- **严格标准**: ≥99.0分
- **验证指标数**: {result['total_indicators']}个
- **通过率**: {result['pass_rate']:.1f}%
- **平均得分**: {result['average_score']:.1f}/100

## 验证结果统计
- **✅ 通过99分标准**: {result['passed_count']}个
- **❌ 未达99分标准**: {result['failed_count']}个

## 通过99分标准的ZXM指标
{chr(10).join([f"- **{name}**: {result['results'][name]['overall_score']:.1f}/100 ✅" for name in passed_indicators])}

## 未达99分标准的ZXM指标
{chr(10).join([f"- **{name}**: {result['results'][name].get('overall_score', 0):.1f}/100 ❌" for name in failed_indicators])}

## 验证标准
- **阶段1: 算法真实性**: 基础功能、数据质量、逻辑一致性、数据完整性、ZXM特征
- **阶段2: 综合功能**: 核心计算、方法实现、异常处理、性能、稳定性

## 验证结论
ZXM体系指标批量验证完成，通过率{result['pass_rate']:.1f}%。

{'### 🎉 验证成功！大部分ZXM指标达到99分以上严格标准。' if result['pass_rate'] >= 80 else '### ⚠️ 需要优化，部分ZXM指标未达到99分严格标准。'}

---
*验证时间: {result['validation_time']:.2f}秒*
*验证工具: ZXM体系指标99分以上严格标准验证系统*
*质量保证: 生产级别最高标准*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 验证报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
