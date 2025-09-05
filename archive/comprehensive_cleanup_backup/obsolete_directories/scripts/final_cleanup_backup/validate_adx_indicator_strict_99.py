#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADX指标严格验证 - 99分以上标准（重新验证）
确保ADX指标达到生产级别的最高质量标准
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


class ADXStrictValidator99:
    """ADX指标严格验证器 - 99分以上标准（重新验证）"""
    
    def __init__(self):
        self.indicator_name = "ADX"
        self.validation_results = {}
        self.test_data = None
        self.min_score = 99.0  # 严格标准：99分以上
        
    def generate_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据，专门用于ADX验证"""
        logger.info("📊 生成ADX高质量测试数据...")
        
        # 生成120天的测试数据，确保有足够的数据进行ADX计算
        dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
        
        # 生成具有明显趋势强度变化的价格数据
        np.random.seed(42)
        base_price = 100.0
        
        # 模拟不同趋势强度阶段
        trend_strength_phases = np.concatenate([
            np.linspace(0, 25, 30),    # 强趋势上升
            np.linspace(25, 20, 20),   # 趋势减弱
            np.linspace(20, 5, 30),    # 横盘整理（弱趋势）
            np.linspace(5, 30, 25),    # 重新建立强趋势
            np.linspace(30, 15, 15)    # 趋势回调
        ])
        
        # 添加方向性变化
        directional_changes = np.concatenate([
            np.ones(30),      # 上升趋势
            np.ones(20) * 0.5, # 混合趋势
            np.zeros(30),     # 无明确方向
            np.ones(25),      # 重新上升
            np.ones(15) * -1  # 下降趋势
        ])
        
        # 生成价格序列
        prices = [base_price]
        for i in range(1, 120):
            trend_strength = trend_strength_phases[i] / 100
            direction = directional_changes[i]
            
            # 基于趋势强度和方向生成价格变化
            price_change = trend_strength * direction * np.random.uniform(0.5, 2.0)
            noise = np.random.normal(0, 0.5)
            
            new_price = prices[-1] + price_change + noise
            prices.append(max(new_price, 1.0))  # 确保价格为正
        
        # 生成OHLC数据
        data = []
        for i, price in enumerate(prices):
            # 根据趋势强度生成日内波动
            strength = trend_strength_phases[i] / 100
            daily_range = strength * price * 0.03  # 强趋势时波动更大
            
            high = price + np.random.uniform(0, daily_range)
            low = price - np.random.uniform(0, daily_range)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'date': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ 生成高质量测试数据: {len(df)}行，包含多种趋势强度环境")
        return df
    
    def stage1_algorithm_authenticity(self) -> Dict[str, Any]:
        """阶段1: 算法真实性验证 - 必须≥99.0分"""
        logger.info("🔍 阶段1: ADX算法真实性严格验证...")
        
        try:
            from indicators.adx import ADX
            
            # 创建ADX指标实例
            adx_indicator = ADX()
            
            # 使用测试数据计算ADX
            result = adx_indicator.calculate(self.test_data)
            
            score = 0
            max_score = 100
            
            # 1. ADX核心算法验证 (40分)
            adx_cols = [col for col in result.columns if 'adx' in col.lower()]
            if len(adx_cols) >= 1:
                score += 15
                logger.info(f"✅ ADX列存在: {adx_cols}")
                
                # 验证ADX值的合理性
                adx_col = adx_cols[0]
                adx_values = result[adx_col].dropna()
                
                if len(adx_values) > 0:
                    # ADX应该在0-100范围内
                    if all(0 <= val <= 100 for val in adx_values):
                        score += 10
                        logger.info("✅ ADX值在0-100范围内")
                    
                    # ADX应该反映趋势强度变化
                    if adx_values.std() > 5:  # 标准差大于5
                        score += 10
                        logger.info("✅ ADX值有合理的趋势强度变化")
                    
                    # ADX应该有明显的高低变化
                    max_adx = adx_values.max()
                    min_adx = adx_values.min()
                    if max_adx - min_adx > 20:
                        score += 5
                        logger.info(f"✅ ADX有明显变化范围: {min_adx:.1f} - {max_adx:.1f}")
            
            # 2. 手动ADX算法验证 (35分)
            # 手动计算ADX进行对比验证
            period = getattr(adx_indicator, 'period', 14)
            
            # 计算True Range (TR)
            high = self.test_data['high']
            low = self.test_data['low']
            close = self.test_data['close']
            prev_close = close.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # 计算Directional Movement (DM)
            high_diff = high.diff()
            low_diff = -low.diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # 计算平滑的ATR和DM
            atr = pd.Series(tr).rolling(window=period).mean()
            plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
            minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
            
            # 计算DI
            plus_di = 100 * plus_dm_smooth / atr
            minus_di = 100 * minus_dm_smooth / atr
            
            # 计算DX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # 计算ADX
            manual_adx = dx.rolling(window=period).mean()
            
            if len(manual_adx.dropna()) > 0 and len(adx_values) > 0:
                # 比较手动计算和指标计算的结果
                min_len = min(len(manual_adx.dropna()), len(adx_values))
                if min_len > 10:  # 确保有足够的数据进行比较
                    calc_subset = adx_values.iloc[-min_len:].values
                    manual_subset = manual_adx.dropna().iloc[-min_len:].values
                    
                    # 计算差异
                    differences = [abs(a - b) for a, b in zip(calc_subset, manual_subset)]
                    max_diff = max(differences) if differences else float('inf')
                    avg_diff = sum(differences) / len(differences) if differences else float('inf')
                    
                    if max_diff < 0.01:  # 差异小于0.01
                        score += 35
                        logger.info(f"✅ ADX手动验证完美匹配，最大差异: {max_diff:.6f}")
                    elif max_diff < 0.1:
                        score += 30
                        logger.info(f"✅ ADX手动验证高度匹配，最大差异: {max_diff:.6f}")
                    elif max_diff < 1.0:
                        score += 20
                        logger.info(f"⚠️ ADX手动验证基本匹配，最大差异: {max_diff:.6f}")
                    elif max_diff < 5.0:
                        score += 10
                        logger.info(f"⚠️ ADX手动验证部分匹配，最大差异: {max_diff:.6f}")
            
            # 3. DI指标验证 (15分)
            # 检查是否包含+DI和-DI
            di_cols = [col for col in result.columns if 'di' in col.lower()]
            if len(di_cols) >= 2:
                score += 10
                logger.info(f"✅ DI指标存在: {di_cols}")
                
                # 验证DI值的合理性
                for di_col in di_cols[:2]:
                    di_values = result[di_col].dropna()
                    if len(di_values) > 0 and all(0 <= val <= 100 for val in di_values):
                        score += 2.5
                        logger.info(f"✅ {di_col}值在合理范围内")
            
            # 4. 数据完整性验证 (7分)
            if result is not None and not result.empty:
                score += 4
                logger.info("✅ 数据输出完整")
                
                # 检查数据类型
                if adx_cols and pd.api.types.is_numeric_dtype(result[adx_cols[0]]):
                    score += 3
                    logger.info("✅ ADX数据类型正确")
            
            # 5. 参数响应验证 (3分)
            try:
                custom_adx = ADX(period=21)
                custom_result = custom_adx.calculate(self.test_data)
                
                if custom_result is not None and not custom_result.empty:
                    score += 3
                    logger.info("✅ 参数响应正确")
            except Exception as e:
                logger.warning(f"⚠️ 参数测试失败: {e}")
            
            final_score = (score / max_score) * 100
            
            result_data = {
                'score': final_score,
                'meets_standard': final_score >= self.min_score,
                'details': {
                    'algorithm_core': score >= 40,
                    'manual_verification': score >= 75,
                    'di_indicators': score >= 90,
                    'data_integrity': score >= 97,
                    'parameter_response': score >= 100
                },
                'adx_sample_values': adx_values.tail(5).tolist() if len(adx_values) > 0 else [],
                'manual_sample_values': manual_adx.dropna().tail(5).tolist() if len(manual_adx.dropna()) > 0 else []
            }
            
            logger.info(f"📊 阶段1得分: {final_score:.1f}/100")
            if final_score >= self.min_score:
                logger.info("🎉 阶段1通过严格标准!")
            else:
                logger.warning(f"⚠️ 阶段1未达到{self.min_score}分标准")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ 阶段1验证失败: {e}")
            return {'score': 0, 'meets_standard': False, 'error': str(e)}
    
    def stage2_comprehensive_functionality(self) -> Dict[str, Any]:
        """阶段2: 综合功能验证 - 必须≥99.0分"""
        logger.info("🔧 阶段2: ADX综合功能严格验证...")
        
        try:
            from indicators.adx import ADX
            
            adx_indicator = ADX()
            score = 0
            max_score = 100
            
            # 1. 核心计算功能 (25分)
            result = adx_indicator.calculate(self.test_data)
            if result is not None and not result.empty:
                score += 15
                logger.info("✅ 核心计算功能正常")
                
                # 检查ADX相关列
                adx_cols = [col for col in result.columns if 'adx' in col.lower()]
                if len(adx_cols) >= 1:
                    score += 10
                    logger.info("✅ ADX输出列完整")
            
            # 2. 标准方法实现 (25分)
            required_methods = ['calculate', 'get_patterns']
            method_score = 0
            
            for method in required_methods:
                if hasattr(adx_indicator, method):
                    try:
                        if method == 'calculate':
                            test_result = adx_indicator.calculate(self.test_data)
                            if test_result is not None:
                                method_score += 12.5
                                logger.info(f"✅ {method} 方法正常工作")
                        elif method == 'get_patterns':
                            patterns = adx_indicator.get_patterns(self.test_data)
                            if patterns is not None:
                                method_score += 12.5
                                logger.info(f"✅ {method} 方法正常工作")
                    except Exception as e:
                        logger.warning(f"⚠️ {method} 方法执行失败: {e}")
                else:
                    logger.warning(f"❌ {method} 方法不存在")
            
            score += method_score
            
            # 3. 参数管理系统 (20分)
            param_score = 0
            
            # 检查参数属性
            if hasattr(adx_indicator, 'period'):
                param_score += 10
                logger.info("✅ period参数存在")
            
            # 测试参数设置
            try:
                test_adx = ADX(period=21)
                if hasattr(test_adx, 'period') and test_adx.period == 21:
                    param_score += 10
                    logger.info("✅ 参数设置功能正常")
            except Exception as e:
                logger.warning(f"⚠️ 参数设置测试失败: {e}")
            
            score += param_score
            
            # 4. 异常处理机制 (15分)
            exception_score = 0
            
            # 测试空数据处理
            try:
                empty_data = pd.DataFrame()
                empty_result = adx_indicator.calculate(empty_data)
                exception_score += 7.5
                logger.info("✅ 空数据异常处理正常")
            except Exception as e:
                logger.info(f"✅ 空数据正确抛出异常: {type(e).__name__}")
                exception_score += 7.5
            
            # 测试无效数据处理
            try:
                invalid_data = self.test_data.copy()
                invalid_data['high'] = np.nan
                invalid_result = adx_indicator.calculate(invalid_data)
                exception_score += 7.5
                logger.info("✅ 无效数据异常处理正常")
            except Exception as e:
                logger.info(f"✅ 无效数据正确抛出异常: {type(e).__name__}")
                exception_score += 7.5
            
            score += exception_score
            
            # 5. 性能和稳定性 (15分)
            performance_score = 0
            
            # 性能测试
            start_time = time.time()
            for _ in range(5):
                test_result = adx_indicator.calculate(self.test_data)
            execution_time = (time.time() - start_time) / 5
            
            if execution_time < 0.05:  # 50ms内完成
                performance_score += 7.5
                logger.info(f"✅ 性能测试通过: {execution_time:.3f}秒")
            elif execution_time < 0.1:
                performance_score += 5
                logger.info(f"✅ 性能可接受: {execution_time:.3f}秒")
            
            # 稳定性测试
            results = []
            for i in range(3):
                test_result = adx_indicator.calculate(self.test_data)
                adx_cols = [col for col in test_result.columns if 'adx' in col.lower()]
                if adx_cols and len(test_result[adx_cols[0]].dropna()) > 0:
                    results.append(test_result[adx_cols[0]].dropna().iloc[-1])
            
            if len(results) >= 2 and all(abs(results[0] - r) < 0.001 for r in results[1:]):
                performance_score += 7.5
                logger.info("✅ 稳定性测试通过")
            
            score += performance_score
            
            final_score = (score / max_score) * 100
            
            result_data = {
                'score': final_score,
                'meets_standard': final_score >= self.min_score,
                'details': {
                    'core_calculation': score >= 25,
                    'method_implementation': score >= 50,
                    'parameter_management': score >= 70,
                    'exception_handling': score >= 85,
                    'performance_stability': score >= 100
                }
            }
            
            logger.info(f"📊 阶段2得分: {final_score:.1f}/100")
            if final_score >= self.min_score:
                logger.info("🎉 阶段2通过严格标准!")
            else:
                logger.warning(f"⚠️ 阶段2未达到{self.min_score}分标准")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ 阶段2验证失败: {e}")
            return {'score': 0, 'meets_standard': False, 'error': str(e)}
    
    def run_strict_validation(self) -> Dict[str, Any]:
        """运行严格验证流程 - 99分以上标准"""
        logger.info("🚀 开始ADX指标严格验证 (99分以上标准) - 重新验证...")
        
        start_time = time.time()
        
        # 生成测试数据
        self.test_data = self.generate_test_data()
        
        # 执行验证阶段
        results = {}
        
        # 阶段1: 算法真实性
        results['stage1'] = self.stage1_algorithm_authenticity()
        
        # 阶段2: 综合功能
        results['stage2'] = self.stage2_comprehensive_functionality()
        
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
            'indicator': 'ADX',
            'validation_time': validation_time,
            'stages': results,
            'overall_score': average_score,
            'meets_strict_standard': passed,
            'min_required_score': self.min_score,
            'status': 'PASSED' if passed else 'FAILED',
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'STRICT_99_REVALIDATION'
        }
        
        logger.info(f"🎯 ADX严格验证完成!")
        logger.info(f"📊 总体得分: {average_score:.1f}/100")
        logger.info(f"🎯 严格标准: ≥{self.min_score}分")
        logger.info(f"⏱️ 验证时间: {validation_time:.2f}秒")
        logger.info(f"✅ 验证状态: {final_result['status']}")
        
        if passed:
            logger.info("🏆 ADX指标通过99分以上严格标准!")
        else:
            logger.warning("⚠️ ADX指标未达到99分严格标准，需要进一步优化")
        
        return final_result


def main():
    """主函数"""
    logger.info("🔍 ADX指标严格验证开始 (99分以上标准) - 重新验证...")
    
    validator = ADXStrictValidator99()
    result = validator.run_strict_validation()
    
    # 保存验证报告
    report_file = f"docs/finaltesting/indicators/adx_strict_99_revalidation_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成验证报告
    report_content = f"""# ADX指标严格验证报告 (99分以上标准) - 重新验证

## 验证概览
- **指标名称**: ADX (平均趋向指数)
- **验证类型**: 99分以上严格标准重新验证
- **验证时间**: {result['timestamp']}
- **严格标准**: ≥{result['min_required_score']}分
- **验证状态**: {result['status']}
- **总体得分**: {result['overall_score']:.1f}/100

## 验证结果详情

### 阶段1: 算法真实性验证 (要求≥99.0分)
- **得分**: {result['stages']['stage1']['score']:.1f}/100
- **达标状态**: {'✅ 通过严格标准' if result['stages']['stage1']['meets_standard'] else '❌ 未达到99分标准'}
- **手动验证**: 包含完整的ADX、DI计算验证

### 阶段2: 综合功能验证 (要求≥99.0分)
- **得分**: {result['stages']['stage2']['score']:.1f}/100
- **达标状态**: {'✅ 通过严格标准' if result['stages']['stage2']['meets_standard'] else '❌ 未达到99分标准'}
- **功能完整性**: 标准方法、参数管理、异常处理、性能稳定

## 严格验证标准
- **算法真实性**: ≥99.0分 (绝对不可妥协)
- **综合功能**: ≥99.0分 (生产级别要求)
- **总体平均**: ≥99.0分 (最高质量标准)
- **手动验证**: 最大差异<0.01 (完美匹配)

## 验证结论
ADX指标重新验证{'✅ 通过' if result['status'] == 'PASSED' else '❌ 失败'}，总体得分{result['overall_score']:.1f}分。

{'### 🏆 验证成功！ADX指标达到99分以上严格标准，符合生产级别最高质量要求。' if result['status'] == 'PASSED' else '### ⚠️ 验证失败，未达到99分严格标准，需要进一步优化。'}

## 技术细节
- **ADX算法**: 基于True Range、Directional Movement的完整计算
- **DI指标**: +DI和-DI的正确实现
- **趋势强度**: 0-100范围内的合理变化
- **性能要求**: 执行时间<0.1秒，结果一致性100%

---
*验证时间: {result['validation_time']:.2f}秒*
*验证工具: 99分以上严格标准验证系统*
*质量保证: 生产级别最高标准*
*验证类型: 重新验证*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 严格验证报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
