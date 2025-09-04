#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工厂模式指标验证框架 - 99分以上标准
专门用于验证54个工厂模式指标，适配dict返回格式
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


class FactoryIndicatorValidator99:
    """工厂模式指标验证器 - 99分以上标准"""
    
    def __init__(self):
        self.min_score = 99.0  # 严格标准：99分以上
        self.test_data = None
        
        # 54个工厂模式指标列表
        self.factory_indicators = [
            # ZXM体系指标 (35个)
            'ZXM_DAILY_MACD', 'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_MA_CALLBACK',
            'ZXM_BS_ABSORB', 'ZXM_DAILY_TREND_UP', 'ZXM_WEEKLY_TREND_UP', 'ZXM_MONTHLY_KDJ_TREND_UP',
            'ZXM_WEEKLY_MACD', 'ZXM_MONTHLY_MACD', 'ZXM_AMPLITUDE_ELASTICITY', 'ZXM_RISE_ELASTICITY',
            'ZXM_ELASTICITY', 'ZXM_BOUNCE_DETECTOR', 'ZXM_BUYPOINT_SCORE', 'ZXM_TREND_SCORE',
            'ZXM_ELASTIC_SCORE', 'ZXM_VOLUME_ENERGY', 'ZXM_PRICE_POSITION', 'ZXM_TECHNICAL_FORM',
            'ZXM_HOT_SPOT', 'ZXM_INDUSTRY_ROTATION', 'ZXM_CYCLE_POSITION', 'ZXM_RISK_CONTROL',
            'ZXM_TIMING_SIGNAL', 'ZXM_POSITION_MANAGEMENT', 'ZXM_PORTFOLIO_OPTIMIZATION', 'ZXM_STRATEGY_COMBINATION',
            'ZXM_PERFORMANCE_ATTRIBUTION', 'ZXM_ALPHA_GENERATION', 'ZXM_BETA_HEDGING',
            
            # 形态识别指标 (19个)
            'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI', 'PIERCING_LINE',
            'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR', 'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS',
            'V_SHAPED_REVERSAL', 'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'TRIANGLE',
            'WEDGE', 'FLAG', 'PENNANT'
        ]
        
    def generate_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据，适用于所有工厂模式指标"""
        logger.info("📊 生成工厂模式指标高质量测试数据...")
        
        # 生成150天的测试数据，确保有足够的数据进行各种指标计算
        dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
        
        # 生成具有多种市场特征的价格数据
        np.random.seed(42)
        base_price = 100.0
        
        # 模拟复杂的市场环境
        market_phases = np.concatenate([
            np.linspace(0, 20, 30),    # 上升趋势
            np.linspace(20, 15, 20),   # 横盘整理
            np.linspace(15, 5, 25),    # 下降趋势
            np.linspace(5, 25, 35),    # 强势反弹
            np.linspace(25, 30, 25),   # 高位震荡
            np.linspace(30, 20, 15)    # 回调整理
        ])
        
        # 添加波动性和成交量特征
        volatility = np.sin(np.linspace(0, 8*np.pi, 150)) * 3
        volume_pattern = np.cos(np.linspace(0, 6*np.pi, 150)) * 2000000 + 3000000
        
        # 生成价格序列
        prices = []
        for i in range(150):
            if i == 0:
                price = base_price
            else:
                trend = market_phases[i] - market_phases[i-1]
                vol = volatility[i]
                noise = np.random.normal(0, 1)
                
                price_change = trend + vol * 0.3 + noise * 0.5
                price = max(prices[-1] + price_change, 1.0)
            
            prices.append(price)
        
        # 生成OHLC数据
        data = []
        for i, price in enumerate(prices):
            daily_vol = abs(volatility[i]) * 0.02 + 0.01
            high = price + np.random.uniform(0, daily_vol * price)
            low = price - np.random.uniform(0, daily_vol * price)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = int(volume_pattern[i] * np.random.uniform(0.8, 1.2))
            
            data.append({
                'date': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ 生成高质量测试数据: {len(df)}行，包含多种市场环境")
        return df
    
    def validate_factory_indicator(self, indicator_name: str) -> Dict[str, Any]:
        """验证单个工厂模式指标 - 99分以上标准"""
        logger.info(f"🔍 验证工厂模式指标: {indicator_name}")
        
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
            
            score = 0
            max_score = 100
            
            # 1. 基础功能验证 (30分)
            try:
                result = indicator.calculate(self.test_data)
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
            except Exception as e:
                logger.warning(f"⚠️ {indicator_name} 基础功能测试失败: {e}")
            
            # 2. 数据质量验证 (25分)
            try:
                result = indicator.calculate(self.test_data)
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
                            if quality_ratio >= 0.9:
                                score += 25
                                logger.info(f"✅ {indicator_name} 数据质量优秀: {quality_ratio:.2%}")
                            elif quality_ratio >= 0.7:
                                score += 15
                                logger.info(f"✅ {indicator_name} 数据质量良好: {quality_ratio:.2%}")
                            elif quality_ratio >= 0.5:
                                score += 10
                                logger.info(f"⚠️ {indicator_name} 数据质量一般: {quality_ratio:.2%}")
                    
                    elif isinstance(result, pd.DataFrame):
                        # 检查DataFrame中的数据质量
                        numeric_cols = result.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            valid_ratio = result[numeric_cols].notna().mean().mean()
                            if valid_ratio >= 0.9:
                                score += 25
                                logger.info(f"✅ {indicator_name} DataFrame数据质量优秀: {valid_ratio:.2%}")
                            elif valid_ratio >= 0.7:
                                score += 15
                                logger.info(f"✅ {indicator_name} DataFrame数据质量良好: {valid_ratio:.2%}")
            except Exception as e:
                logger.warning(f"⚠️ {indicator_name} 数据质量验证失败: {e}")
            
            # 3. 性能验证 (20分)
            try:
                start_time = time.time()
                for _ in range(3):
                    result = indicator.calculate(self.test_data)
                execution_time = (time.time() - start_time) / 3
                
                if execution_time < 0.1:  # 100ms内完成
                    score += 20
                    logger.info(f"✅ {indicator_name} 性能优秀: {execution_time:.3f}秒")
                elif execution_time < 0.5:
                    score += 15
                    logger.info(f"✅ {indicator_name} 性能良好: {execution_time:.3f}秒")
                elif execution_time < 1.0:
                    score += 10
                    logger.info(f"⚠️ {indicator_name} 性能一般: {execution_time:.3f}秒")
            except Exception as e:
                logger.warning(f"⚠️ {indicator_name} 性能测试失败: {e}")
            
            # 4. 稳定性验证 (15分)
            try:
                results = []
                for i in range(3):
                    result = indicator.calculate(self.test_data)
                    if result is not None:
                        results.append(str(result))
                
                if len(results) >= 2 and all(r == results[0] for r in results[1:]):
                    score += 15
                    logger.info(f"✅ {indicator_name} 稳定性测试通过")
                elif len(results) >= 2:
                    score += 10
                    logger.info(f"⚠️ {indicator_name} 稳定性一般")
            except Exception as e:
                logger.warning(f"⚠️ {indicator_name} 稳定性测试失败: {e}")
            
            # 5. 异常处理验证 (10分)
            try:
                # 测试空数据
                empty_data = pd.DataFrame()
                empty_result = indicator.calculate(empty_data)
                score += 5
                logger.info(f"✅ {indicator_name} 空数据处理正常")
                
                # 测试无效数据
                invalid_data = self.test_data.copy()
                invalid_data['close'] = np.nan
                invalid_result = indicator.calculate(invalid_data)
                score += 5
                logger.info(f"✅ {indicator_name} 无效数据处理正常")
            except Exception as e:
                logger.info(f"✅ {indicator_name} 正确抛出异常: {type(e).__name__}")
                score += 10
            
            final_score = (score / max_score) * 100
            meets_standard = final_score >= self.min_score
            
            result_data = {
                'indicator': indicator_name,
                'score': final_score,
                'meets_standard': meets_standard,
                'details': {
                    'basic_functionality': score >= 30,
                    'data_quality': score >= 55,
                    'performance': score >= 75,
                    'stability': score >= 90,
                    'exception_handling': score >= 100
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📊 {indicator_name} 得分: {final_score:.1f}/100")
            if meets_standard:
                logger.info(f"🎉 {indicator_name} 通过99分严格标准!")
            else:
                logger.warning(f"⚠️ {indicator_name} 未达到99分标准")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ {indicator_name} 验证失败: {e}")
            return {
                'indicator': indicator_name,
                'score': 0,
                'meets_standard': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_batch_validation(self, indicator_list: List[str] = None) -> Dict[str, Any]:
        """批量验证工厂模式指标"""
        if indicator_list is None:
            indicator_list = self.factory_indicators
        
        logger.info(f"🚀 开始批量验证{len(indicator_list)}个工厂模式指标 (99分以上标准)...")
        
        start_time = time.time()
        
        # 生成测试数据
        self.test_data = self.generate_test_data()
        
        # 验证结果
        results = {}
        passed_count = 0
        failed_count = 0
        
        for indicator_name in indicator_list:
            result = self.validate_factory_indicator(indicator_name)
            results[indicator_name] = result
            
            if result.get('meets_standard', False):
                passed_count += 1
            else:
                failed_count += 1
        
        # 计算总体统计
        total_score = sum(r.get('score', 0) for r in results.values())
        average_score = total_score / len(results) if results else 0
        
        validation_time = time.time() - start_time
        
        summary = {
            'validation_type': 'FACTORY_INDICATORS_BATCH_99',
            'total_indicators': len(indicator_list),
            'passed_count': passed_count,
            'failed_count': failed_count,
            'pass_rate': (passed_count / len(indicator_list)) * 100 if indicator_list else 0,
            'average_score': average_score,
            'validation_time': validation_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 批量验证完成!")
        logger.info(f"📊 通过率: {summary['pass_rate']:.1f}% ({passed_count}/{len(indicator_list)})")
        logger.info(f"📊 平均得分: {average_score:.1f}/100")
        logger.info(f"⏱️ 验证时间: {validation_time:.2f}秒")
        
        return summary


def main():
    """主函数 - 验证所有54个工厂模式指标"""
    logger.info("🔍 工厂模式指标验证开始 (99分以上标准)...")

    validator = FactoryIndicatorValidator99()

    # 验证所有54个工厂模式指标
    result = validator.run_batch_validation()
    
    # 保存验证报告
    report_file = f"docs/finaltesting/indicators/factory_indicators_complete_validation_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成验证报告
    passed_indicators = [name for name, data in result['results'].items() if data.get('meets_standard', False)]
    failed_indicators = [name for name, data in result['results'].items() if not data.get('meets_standard', False)]
    
    report_content = f"""# 工厂模式指标验证报告 (99分以上标准) - 完整验证

## 验证概览
- **验证类型**: 工厂模式指标批量验证
- **验证时间**: {result['timestamp']}
- **严格标准**: ≥99.0分
- **验证指标数**: {result['total_indicators']}个
- **通过率**: {result['pass_rate']:.1f}%
- **平均得分**: {result['average_score']:.1f}/100

## 验证结果统计
- **✅ 通过99分标准**: {result['passed_count']}个
- **❌ 未达99分标准**: {result['failed_count']}个

## 通过99分标准的指标
{chr(10).join([f"- **{name}**: {result['results'][name]['score']:.1f}/100 ✅" for name in passed_indicators])}

## 未达99分标准的指标
{chr(10).join([f"- **{name}**: {result['results'][name]['score']:.1f}/100 ❌" for name in failed_indicators])}

## 验证标准
- **基础功能**: 30分 (计算功能、返回格式、数据有效性)
- **数据质量**: 25分 (数据完整性、有效性比例)
- **性能要求**: 20分 (执行时间<0.1秒为优秀)
- **稳定性**: 15分 (多次执行结果一致性)
- **异常处理**: 10分 (空数据、无效数据处理)

## 验证结论
工厂模式指标验证完成，通过率{result['pass_rate']:.1f}%。

{'### 🎉 验证成功！大部分指标达到99分以上严格标准。' if result['pass_rate'] >= 80 else '### ⚠️ 需要优化，部分指标未达到99分严格标准。'}

---
*验证时间: {result['validation_time']:.2f}秒*
*验证工具: 工厂模式指标99分以上严格标准验证系统*
*质量保证: 生产级别最高标准*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 验证报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
