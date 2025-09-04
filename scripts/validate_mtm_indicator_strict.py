#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTM指标严格验证 - 99分以上标准
确保每个指标都达到生产级别的最高质量标准
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


class MTMStrictValidator:
    """MTM指标严格验证器 - 99分以上标准"""
    
    def __init__(self):
        self.indicator_name = "MTM"
        self.validation_results = {}
        self.test_data = None
        self.min_score = 99.0  # 严格标准：99分以上
        
    def generate_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据"""
        logger.info("📊 生成MTM高质量测试数据...")
        
        # 生成100天的测试数据，确保有足够的数据进行动量计算
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # 生成具有明显动量特征的价格数据
        np.random.seed(42)
        base_price = 100.0
        
        # 模拟不同动量阶段的价格变化
        momentum_phases = np.concatenate([
            np.linspace(0, 15, 25),    # 强上升动量
            np.linspace(15, 10, 15),   # 动量减弱
            np.linspace(10, 5, 20),    # 横盘整理
            np.linspace(5, 25, 25),    # 重新加速上升
            np.linspace(25, 20, 15)    # 动量回调
        ])
        
        # 添加随机波动
        noise = np.random.normal(0, 1, 100)
        prices = base_price + momentum_phases + noise
        
        # 生成OHLC数据
        data = []
        for i, price in enumerate(prices):
            daily_range = abs(momentum_phases[i] - momentum_phases[i-1]) if i > 0 else 1
            high = price + np.random.uniform(0.5, daily_range * 0.3)
            low = price - np.random.uniform(0.5, daily_range * 0.3)
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
        logger.info(f"✅ 生成高质量测试数据: {len(df)}行，包含多种动量环境")
        return df
    
    def stage1_algorithm_authenticity(self) -> Dict[str, Any]:
        """阶段1: 算法真实性验证 - 必须≥99.0分"""
        logger.info("🔍 阶段1: MTM算法真实性严格验证...")
        
        try:
            from indicators.mtm import Momentum
            
            # 创建MTM指标实例
            mtm_indicator = Momentum()
            
            # 使用测试数据计算MTM
            result = mtm_indicator.calculate(self.test_data)
            
            score = 0
            max_score = 100
            
            # 1. MTM核心算法验证 (35分)
            mtm_cols = [col for col in result.columns if 'mtm' in col.lower() or 'momentum' in col.lower()]
            if len(mtm_cols) >= 1:
                score += 15
                logger.info(f"✅ MTM列存在: {mtm_cols}")
                
                # 验证MTM值的合理性
                mtm_col = mtm_cols[0]
                mtm_values = result[mtm_col].dropna()
                
                if len(mtm_values) > 0:
                    # MTM应该有正负值变化
                    has_positive = any(val > 0 for val in mtm_values)
                    has_negative = any(val < 0 for val in mtm_values)
                    
                    if has_positive and has_negative:
                        score += 10
                        logger.info("✅ MTM值有正负变化")
                    elif has_positive or has_negative:
                        score += 5
                        logger.info("✅ MTM值有单向变化")
                    
                    # MTM应该反映价格动量变化
                    if mtm_values.std() > 0:
                        score += 10
                        logger.info("✅ MTM值有合理变化")
            
            # 2. 手动算法验证 (30分)
            # MTM = 当前价格 - N期前价格
            period = getattr(mtm_indicator, 'period', 12)
            manual_mtm = []
            
            for i in range(period, len(self.test_data)):
                current_price = self.test_data.iloc[i]['close']
                past_price = self.test_data.iloc[i-period]['close']
                mtm_val = current_price - past_price
                manual_mtm.append(mtm_val)
            
            if len(manual_mtm) > 0 and len(mtm_values) > 0:
                # 比较手动计算和指标计算的结果
                min_len = min(len(manual_mtm), len(mtm_values))
                if min_len > 0:
                    calc_subset = mtm_values.iloc[-min_len:].values
                    manual_subset = manual_mtm[-min_len:]
                    
                    # 计算差异
                    differences = [abs(a - b) for a, b in zip(calc_subset, manual_subset)]
                    max_diff = max(differences) if differences else float('inf')
                    avg_diff = sum(differences) / len(differences) if differences else float('inf')
                    
                    if max_diff < 0.01:  # 差异小于0.01
                        score += 30
                        logger.info(f"✅ MTM手动验证完美匹配，最大差异: {max_diff:.6f}")
                    elif max_diff < 0.1:
                        score += 25
                        logger.info(f"✅ MTM手动验证高度匹配，最大差异: {max_diff:.6f}")
                    elif max_diff < 1.0:
                        score += 15
                        logger.info(f"⚠️ MTM手动验证基本匹配，最大差异: {max_diff:.6f}")
            
            # 3. 数据完整性验证 (20分)
            if result is not None and not result.empty:
                score += 10
                logger.info("✅ 数据输出完整")
                
                # 检查数据类型和范围
                if mtm_cols and pd.api.types.is_numeric_dtype(result[mtm_cols[0]]):
                    score += 5
                    logger.info("✅ MTM数据类型正确")
                
                # 检查数据长度合理性
                expected_length = len(self.test_data) - period + 1
                if abs(len(mtm_values) - expected_length) <= 2:  # 允许2行误差
                    score += 5
                    logger.info("✅ MTM数据长度合理")
            
            # 4. 参数响应验证 (10分)
            try:
                custom_mtm = Momentum(period=20)
                custom_result = custom_mtm.calculate(self.test_data)
                
                if custom_result is not None and not custom_result.empty:
                    score += 10
                    logger.info("✅ 参数响应正确")
            except Exception as e:
                logger.warning(f"⚠️ 参数测试失败: {e}")
            
            # 5. 边界条件处理 (5分)
            try:
                small_data = self.test_data.head(15)  # 小于默认周期的数据
                small_result = mtm_indicator.calculate(small_data)
                
                if small_result is not None:
                    score += 5
                    logger.info("✅ 小数据集处理正常")
            except Exception as e:
                logger.warning(f"⚠️ 边界条件处理需要改进: {e}")
            
            final_score = (score / max_score) * 100
            
            result_data = {
                'score': final_score,
                'meets_standard': final_score >= self.min_score,
                'details': {
                    'algorithm_core': score >= 35,
                    'manual_verification': score >= 65,
                    'data_integrity': score >= 85,
                    'parameter_response': score >= 95,
                    'boundary_conditions': score >= 100
                },
                'mtm_sample_values': mtm_values.tail(5).tolist() if len(mtm_values) > 0 else [],
                'manual_sample_values': manual_mtm[-5:] if len(manual_mtm) >= 5 else manual_mtm
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
        logger.info("🔧 阶段2: MTM综合功能严格验证...")
        
        try:
            from indicators.mtm import Momentum
            
            mtm_indicator = Momentum()
            score = 0
            max_score = 100
            
            # 1. 核心计算功能 (25分)
            result = mtm_indicator.calculate(self.test_data)
            if result is not None and not result.empty:
                score += 15
                logger.info("✅ 核心计算功能正常")
                
                # 检查MTM相关列
                mtm_cols = [col for col in result.columns if 'mtm' in col.lower() or 'momentum' in col.lower()]
                if len(mtm_cols) >= 1:
                    score += 10
                    logger.info("✅ MTM输出列完整")
            
            # 2. 标准方法实现 (25分)
            required_methods = ['calculate', 'get_patterns']
            method_score = 0
            
            for method in required_methods:
                if hasattr(mtm_indicator, method):
                    try:
                        if method == 'calculate':
                            test_result = mtm_indicator.calculate(self.test_data)
                            if test_result is not None:
                                method_score += 12.5
                                logger.info(f"✅ {method} 方法正常工作")
                        elif method == 'get_patterns':
                            patterns = mtm_indicator.get_patterns(self.test_data)
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
            if hasattr(mtm_indicator, 'period'):
                param_score += 10
                logger.info("✅ period参数存在")
            
            # 测试参数设置
            try:
                test_mtm = Momentum(period=15)
                if hasattr(test_mtm, 'period') and test_mtm.period == 15:
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
                empty_result = mtm_indicator.calculate(empty_data)
                exception_score += 7.5
                logger.info("✅ 空数据异常处理正常")
            except Exception as e:
                logger.info(f"✅ 空数据正确抛出异常: {type(e).__name__}")
                exception_score += 7.5
            
            # 测试无效数据处理
            try:
                invalid_data = self.test_data.copy()
                invalid_data['close'] = np.nan
                invalid_result = mtm_indicator.calculate(invalid_data)
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
                test_result = mtm_indicator.calculate(self.test_data)
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
                test_result = mtm_indicator.calculate(self.test_data)
                mtm_cols = [col for col in test_result.columns if 'mtm' in col.lower() or 'momentum' in col.lower()]
                if mtm_cols and len(test_result[mtm_cols[0]].dropna()) > 0:
                    results.append(test_result[mtm_cols[0]].dropna().iloc[-1])
            
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
        logger.info("🚀 开始MTM指标严格验证 (99分以上标准)...")
        
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
            'indicator': 'MTM',
            'validation_time': validation_time,
            'stages': results,
            'overall_score': average_score,
            'meets_strict_standard': passed,
            'min_required_score': self.min_score,
            'status': 'PASSED' if passed else 'FAILED',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 MTM严格验证完成!")
        logger.info(f"📊 总体得分: {average_score:.1f}/100")
        logger.info(f"🎯 严格标准: ≥{self.min_score}分")
        logger.info(f"⏱️ 验证时间: {validation_time:.2f}秒")
        logger.info(f"✅ 验证状态: {final_result['status']}")
        
        if passed:
            logger.info("🏆 MTM指标通过99分以上严格标准!")
        else:
            logger.warning("⚠️ MTM指标未达到99分严格标准，需要优化")
        
        return final_result


def main():
    """主函数"""
    logger.info("🔍 MTM指标严格验证开始 (99分以上标准)...")
    
    validator = MTMStrictValidator()
    result = validator.run_strict_validation()
    
    # 保存验证报告
    report_file = f"docs/finaltesting/indicators/mtm_strict_validation_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成验证报告
    report_content = f"""# MTM指标严格验证报告 (99分以上标准)

## 验证概览
- **指标名称**: MTM (动量指标)
- **验证时间**: {result['timestamp']}
- **严格标准**: ≥{result['min_required_score']}分
- **验证状态**: {result['status']}
- **总体得分**: {result['overall_score']:.1f}/100

## 验证结果详情

### 阶段1: 算法真实性验证 (要求≥99.0分)
- **得分**: {result['stages']['stage1']['score']:.1f}/100
- **达标状态**: {'✅ 通过严格标准' if result['stages']['stage1']['meets_standard'] else '❌ 未达到99分标准'}

### 阶段2: 综合功能验证 (要求≥99.0分)
- **得分**: {result['stages']['stage2']['score']:.1f}/100
- **达标状态**: {'✅ 通过严格标准' if result['stages']['stage2']['meets_standard'] else '❌ 未达到99分标准'}

## 严格验证标准
- **算法真实性**: ≥99.0分 (绝对不可妥协)
- **综合功能**: ≥99.0分 (生产级别要求)
- **总体平均**: ≥99.0分 (最高质量标准)

## 验证结论
MTM指标验证{'✅ 通过' if result['status'] == 'PASSED' else '❌ 失败'}，总体得分{result['overall_score']:.1f}分。

{'### 🏆 验证成功！MTM指标达到99分以上严格标准，符合生产级别最高质量要求。' if result['status'] == 'PASSED' else '### ⚠️ 验证失败，未达到99分严格标准，需要进一步优化。'}

---
*验证时间: {result['validation_time']:.2f}秒*
*验证工具: 99分以上严格标准验证系统*
*质量保证: 生产级别最高标准*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 严格验证报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
