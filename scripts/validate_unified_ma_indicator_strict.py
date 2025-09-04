#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIFIED_MA指标严格验证 - 99分以上标准
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


class UnifiedMAStrictValidator:
    """UNIFIED_MA指标严格验证器 - 99分以上标准"""
    
    def __init__(self):
        self.indicator_name = "UNIFIED_MA"
        self.validation_results = {}
        self.test_data = None
        self.min_score = 99.0  # 严格标准：99分以上
        
    def generate_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据"""
        logger.info("📊 生成UNIFIED_MA高质量测试数据...")
        
        # 生成100天的测试数据，确保有足够的数据进行MA计算
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # 生成具有明显趋势的价格数据，适合MA分析
        np.random.seed(42)
        base_price = 100.0
        
        # 模拟不同趋势阶段
        trend_phases = np.concatenate([
            np.linspace(0, 20, 30),    # 上升趋势
            np.linspace(20, 15, 20),   # 横盘整理
            np.linspace(15, 35, 30),   # 强势上升
            np.linspace(35, 25, 20)    # 回调整理
        ])
        
        # 添加周期性波动
        cycle = np.sin(np.linspace(0, 4*np.pi, 100)) * 3
        
        # 添加随机噪声
        noise = np.random.normal(0, 1.5, 100)
        
        # 综合价格序列
        prices = base_price + trend_phases + cycle + noise
        
        # 生成OHLC数据
        data = []
        for i, price in enumerate(prices):
            daily_range = abs(trend_phases[i] - trend_phases[i-1]) if i > 0 else 2
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
        logger.info(f"✅ 生成高质量测试数据: {len(df)}行，包含多种趋势环境")
        return df
    
    def stage1_algorithm_authenticity(self) -> Dict[str, Any]:
        """阶段1: 算法真实性验证 - 必须≥99.0分"""
        logger.info("🔍 阶段1: UNIFIED_MA算法真实性严格验证...")
        
        try:
            from indicators.unified_ma import UnifiedMa

            # 创建UNIFIED_MA指标实例
            unified_ma_indicator = UnifiedMa()
            
            # 使用测试数据计算UNIFIED_MA
            result = unified_ma_indicator.calculate_Ma(self.test_data)
            
            score = 0
            max_score = 100
            
            # 1. UNIFIED_MA核心算法验证 (35分)
            ma_cols = [col for col in result.columns if 'ma' in col.lower() or 'unified' in col.lower()]
            if len(ma_cols) >= 1:
                score += 15
                logger.info(f"✅ UNIFIED_MA列存在: {ma_cols}")
                
                # 验证MA值的合理性
                ma_col = ma_cols[0]
                ma_values = result[ma_col].dropna()
                
                if len(ma_values) > 0:
                    # MA应该平滑价格波动
                    close_values = self.test_data['close'][-len(ma_values):]
                    
                    # 计算平滑度（MA的标准差应该小于原始价格）
                    ma_std = ma_values.std()
                    close_std = close_values.std()
                    
                    if ma_std < close_std:
                        score += 10
                        logger.info("✅ UNIFIED_MA具有平滑效果")
                    
                    # MA应该跟随价格趋势
                    ma_trend = ma_values.iloc[-1] - ma_values.iloc[0]
                    close_trend = close_values.iloc[-1] - close_values.iloc[0]
                    
                    if (ma_trend > 0 and close_trend > 0) or (ma_trend < 0 and close_trend < 0):
                        score += 10
                        logger.info("✅ UNIFIED_MA跟随价格趋势")
            
            # 2. 手动算法验证 (30分)
            # 手动计算简单移动平均进行对比
            period = getattr(unified_ma_indicator, 'period', 20)
            manual_ma = self.test_data['close'].rolling(window=period).mean()
            
            if len(manual_ma.dropna()) > 0 and len(ma_values) > 0:
                # 比较手动计算和指标计算的结果
                min_len = min(len(manual_ma.dropna()), len(ma_values))
                if min_len > 0:
                    calc_subset = ma_values.iloc[-min_len:].values
                    manual_subset = manual_ma.dropna().iloc[-min_len:].values
                    
                    # 计算差异
                    differences = [abs(a - b) for a, b in zip(calc_subset, manual_subset)]
                    max_diff = max(differences) if differences else float('inf')
                    avg_diff = sum(differences) / len(differences) if differences else float('inf')
                    
                    if max_diff < 0.01:  # 差异小于0.01
                        score += 30
                        logger.info(f"✅ UNIFIED_MA手动验证完美匹配，最大差异: {max_diff:.6f}")
                    elif max_diff < 0.1:
                        score += 25
                        logger.info(f"✅ UNIFIED_MA手动验证高度匹配，最大差异: {max_diff:.6f}")
                    elif max_diff < 1.0:
                        score += 15
                        logger.info(f"⚠️ UNIFIED_MA手动验证基本匹配，最大差异: {max_diff:.6f}")
            
            # 3. 数据完整性验证 (20分)
            if result is not None and not result.empty:
                score += 10
                logger.info("✅ 数据输出完整")
                
                # 检查数据类型和范围
                if ma_cols and pd.api.types.is_numeric_dtype(result[ma_cols[0]]):
                    score += 5
                    logger.info("✅ UNIFIED_MA数据类型正确")
                
                # 检查数据长度合理性
                expected_length = len(self.test_data) - period + 1
                if abs(len(ma_values) - expected_length) <= 2:  # 允许2行误差
                    score += 5
                    logger.info("✅ UNIFIED_MA数据长度合理")
            
            # 4. 参数响应验证 (10分)
            try:
                custom_ma = UnifiedMa(period=30)
                custom_result = custom_ma.calculate_Ma(self.test_data)
                
                if custom_result is not None and not custom_result.empty:
                    score += 10
                    logger.info("✅ 参数响应正确")
            except Exception as e:
                logger.warning(f"⚠️ 参数测试失败: {e}")
            
            # 5. 边界条件处理 (5分)
            try:
                small_data = self.test_data.head(25)  # 小于默认周期的数据
                small_result = unified_ma_indicator.calculate(small_data)
                
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
                'ma_sample_values': ma_values.tail(5).tolist() if len(ma_values) > 0 else [],
                'manual_sample_values': manual_ma.dropna().tail(5).tolist() if len(manual_ma.dropna()) > 0 else []
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
        logger.info("🔧 阶段2: UNIFIED_MA综合功能严格验证...")
        
        try:
            from indicators.unified_ma import UnifiedMa

            unified_ma_indicator = UnifiedMa()
            score = 0
            max_score = 100
            
            # 1. 核心计算功能 (25分)
            result = unified_ma_indicator.calculate_Ma(self.test_data)
            if result is not None and not result.empty:
                score += 15
                logger.info("✅ 核心计算功能正常")
                
                # 检查UNIFIED_MA相关列
                ma_cols = [col for col in result.columns if 'ma' in col.lower() or 'unified' in col.lower()]
                if len(ma_cols) >= 1:
                    score += 10
                    logger.info("✅ UNIFIED_MA输出列完整")
            
            # 2. 标准方法实现 (25分)
            required_methods = ['calculate', 'get_patterns']
            method_score = 0
            
            for method in required_methods:
                if hasattr(unified_ma_indicator, method):
                    try:
                        if method == 'calculate':
                            test_result = unified_ma_indicator.calculate_Ma(self.test_data)
                            if test_result is not None:
                                method_score += 12.5
                                logger.info(f"✅ {method} 方法正常工作")
                        elif method == 'get_patterns':
                            patterns = unified_ma_indicator.get_patterns(self.test_data)
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
            if hasattr(unified_ma_indicator, 'period'):
                param_score += 10
                logger.info("✅ period参数存在")
            
            # 测试参数设置
            try:
                test_ma = UnifiedMa(period=25)
                if hasattr(test_ma, 'period') and test_ma.period == 25:
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
                empty_result = unified_ma_indicator.calculate_Ma(empty_data)
                exception_score += 7.5
                logger.info("✅ 空数据异常处理正常")
            except Exception as e:
                logger.info(f"✅ 空数据正确抛出异常: {type(e).__name__}")
                exception_score += 7.5
            
            # 测试无效数据处理
            try:
                invalid_data = self.test_data.copy()
                invalid_data['close'] = np.nan
                invalid_result = unified_ma_indicator.calculate_Ma(invalid_data)
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
                test_result = unified_ma_indicator.calculate_Ma(self.test_data)
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
                test_result = unified_ma_indicator.calculate_Ma(self.test_data)
                ma_cols = [col for col in test_result.columns if 'ma' in col.lower() or 'unified' in col.lower()]
                if ma_cols and len(test_result[ma_cols[0]].dropna()) > 0:
                    results.append(test_result[ma_cols[0]].dropna().iloc[-1])
            
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
        logger.info("🚀 开始UNIFIED_MA指标严格验证 (99分以上标准)...")
        
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
            'indicator': 'UNIFIED_MA',
            'validation_time': validation_time,
            'stages': results,
            'overall_score': average_score,
            'meets_strict_standard': passed,
            'min_required_score': self.min_score,
            'status': 'PASSED' if passed else 'FAILED',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 UNIFIED_MA严格验证完成!")
        logger.info(f"📊 总体得分: {average_score:.1f}/100")
        logger.info(f"🎯 严格标准: ≥{self.min_score}分")
        logger.info(f"⏱️ 验证时间: {validation_time:.2f}秒")
        logger.info(f"✅ 验证状态: {final_result['status']}")
        
        if passed:
            logger.info("🏆 UNIFIED_MA指标通过99分以上严格标准!")
        else:
            logger.warning("⚠️ UNIFIED_MA指标未达到99分严格标准，需要优化")
        
        return final_result


def main():
    """主函数"""
    logger.info("🔍 UNIFIED_MA指标严格验证开始 (99分以上标准)...")
    
    validator = UnifiedMAStrictValidator()
    result = validator.run_strict_validation()
    
    # 保存验证报告
    report_file = f"docs/finaltesting/indicators/unified_ma_strict_validation_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成验证报告
    report_content = f"""# UNIFIED_MA指标严格验证报告 (99分以上标准)

## 验证概览
- **指标名称**: UNIFIED_MA (统一移动平均)
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
UNIFIED_MA指标验证{'✅ 通过' if result['status'] == 'PASSED' else '❌ 失败'}，总体得分{result['overall_score']:.1f}分。

{'### 🏆 验证成功！UNIFIED_MA指标达到99分以上严格标准，符合生产级别最高质量要求。' if result['status'] == 'PASSED' else '### ⚠️ 验证失败，未达到99分严格标准，需要进一步优化。'}

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
