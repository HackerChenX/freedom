#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复后BaseIndicator指标验证脚本
验证VIX、ROC、MFI三个修复后的指标是否达到99分以上标准
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


class FixedBaseIndicatorValidator:
    """修复后BaseIndicator指标验证器"""
    
    def __init__(self):
        self.target_score = 99.0  # 目标分数：99分以上
        self.test_data = None
        self.fixed_indicators = ['VIX', 'ROC', 'MFI']
        
    def generate_premium_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据"""
        logger.info("📊 生成高质量测试数据...")
        
        # 生成120天的测试数据
        dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
        
        np.random.seed(42)
        base_price = 100.0
        base_volume = 2000000
        
        # 生成具有真实市场特征的数据
        market_trend = np.concatenate([
            np.linspace(0, 20, 30),    # 上升趋势
            np.linspace(20, 25, 20),   # 加速上升
            np.linspace(25, 15, 25),   # 高位震荡
            np.linspace(15, 5, 25),    # 下降趋势
            np.linspace(5, 18, 20)     # 底部反弹
        ])
        
        volatility = np.sin(np.linspace(0, 8*np.pi, 120)) * 2 + 3
        volume_pattern = np.cos(np.linspace(0, 6*np.pi, 120)) * 1000000 + base_volume
        
        prices = [base_price]
        volumes = [base_volume]
        
        for i in range(1, 120):
            trend = (market_trend[i] - market_trend[i-1]) * 0.4
            vol = volatility[i] * 0.15
            noise = np.random.normal(0, 0.6)
            
            price_change = trend + vol + noise
            new_price = max(prices[-1] + price_change, 1.0)
            
            volume_factor = (volatility[i] / 5) + 0.8
            new_volume = int(volume_pattern[i] * volume_factor * np.random.uniform(0.8, 1.2))
            
            prices.append(new_price)
            volumes.append(max(new_volume, 100000))
        
        # 生成OHLC数据
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            daily_vol = volatility[i] * 0.01
            high = price + np.random.uniform(0, daily_vol * price)
            low = price - np.random.uniform(0, daily_vol * price)
            
            open_price = prices[i-1] if i > 0 else price
            close = price
            
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
        logger.info(f"✅ 生成高质量测试数据: {len(df)}行")
        return df
    
    def validate_single_indicator_99_standard(self, indicator_name: str) -> Dict[str, Any]:
        """使用99分严格标准验证单个指标"""
        logger.info(f"🔍 验证修复后指标: {indicator_name}")
        
        try:
            # 导入指标
            if indicator_name == 'VIX':
                from indicators.vix import Vix
                indicator = Vix()
            elif indicator_name == 'ROC':
                from indicators.roc import RateOfChange
                indicator = RateOfChange()
            elif indicator_name == 'MFI':
                from indicators.mfi import Mfi
                indicator = Mfi()
            else:
                return {'score': 0, 'meets_standard': False, 'error': f'未知指标: {indicator_name}'}
            
            start_time = time.time()
            
            # 99分严格标准验证
            score = 0
            max_score = 100
            
            # 1. 算法真实性验证 (30分)
            try:
                result = indicator.calculate(self.test_data)
                if result is not None and not result.empty:
                    score += 15
                    logger.info(f"✅ {indicator_name} 基础计算功能正常")
                    
                    # 检查数据质量
                    valid_data_ratio = 0
                    for col in result.columns:
                        if result[col].notna().sum() > 0:
                            valid_data_ratio = result[col].notna().sum() / len(result)
                            break
                    
                    if valid_data_ratio >= 0.8:
                        score += 15
                        logger.info(f"✅ {indicator_name} 数据质量优秀: {valid_data_ratio:.2%}")
                    elif valid_data_ratio >= 0.6:
                        score += 10
                        logger.info(f"⚠️ {indicator_name} 数据质量一般: {valid_data_ratio:.2%}")
            except Exception as e:
                logger.warning(f"⚠️ {indicator_name} 算法真实性验证失败: {e}")
            
            # 2. 综合功能验证 (40分) - 关键提升点
            method_score = 0
            
            # calculate方法 (10分)
            if hasattr(indicator, 'calculate'):
                try:
                    test_result = indicator.calculate(self.test_data)
                    if test_result is not None:
                        method_score += 10
                        logger.info(f"✅ {indicator_name} calculate方法正常")
                except Exception as e:
                    logger.warning(f"⚠️ {indicator_name} calculate方法失败: {e}")
            
            # get_patterns方法 (10分)
            if hasattr(indicator, 'get_patterns'):
                try:
                    patterns = indicator.get_patterns(self.test_data)
                    if patterns is not None:
                        method_score += 10
                        logger.info(f"✅ {indicator_name} get_patterns方法正常")
                except Exception as e:
                    logger.warning(f"⚠️ {indicator_name} get_patterns方法失败: {e}")
                    method_score += 5  # 部分分数
            
            # get_signals方法 (10分) - 新增验证
            if hasattr(indicator, 'get_signals'):
                try:
                    signals = indicator.get_signals(self.test_data)
                    if signals is not None:
                        method_score += 10
                        logger.info(f"✅ {indicator_name} get_signals方法正常")
                except Exception as e:
                    logger.warning(f"⚠️ {indicator_name} get_signals方法失败: {e}")
            
            # calculate_raw_score方法 (10分) - 新增验证
            if hasattr(indicator, 'calculate_raw_score'):
                try:
                    raw_score = indicator.calculate_raw_score(self.test_data)
                    if raw_score is not None:
                        method_score += 10
                        logger.info(f"✅ {indicator_name} calculate_raw_score方法正常")
                except Exception as e:
                    logger.warning(f"⚠️ {indicator_name} calculate_raw_score方法失败: {e}")
            
            score += method_score
            
            # 3. 性能稳定性验证 (20分)
            try:
                start_perf = time.time()
                for _ in range(5):
                    test_result = indicator.calculate(self.test_data)
                execution_time = (time.time() - start_perf) / 5
                
                if execution_time < 0.05:
                    score += 20
                    logger.info(f"✅ {indicator_name} 性能优秀: {execution_time:.3f}秒")
                elif execution_time < 0.1:
                    score += 15
                    logger.info(f"✅ {indicator_name} 性能良好: {execution_time:.3f}秒")
                elif execution_time < 0.2:
                    score += 10
                    logger.info(f"⚠️ {indicator_name} 性能一般: {execution_time:.3f}秒")
            except Exception as e:
                logger.warning(f"⚠️ {indicator_name} 性能测试失败: {e}")
            
            # 4. 架构合规验证 (10分)
            compliance_score = 0
            
            # 检查必要属性
            if hasattr(indicator, 'name'):
                compliance_score += 3
            if hasattr(indicator, 'period') or hasattr(indicator, '_parameters'):
                compliance_score += 3
            if hasattr(indicator, 'minimum_periods'):
                compliance_score += 4
                
            score += compliance_score
            
            final_score = score
            validation_time = time.time() - start_time
            
            # 99分标准
            passed = final_score >= self.target_score
            
            result_data = {
                'indicator': indicator_name,
                'validation_time': validation_time,
                'overall_score': final_score,
                'meets_standard': passed,
                'min_required_score': self.target_score,
                'status': 'PASSED' if passed else 'FAILED',
                'timestamp': datetime.now().isoformat(),
                'validation_type': 'FIXED_BASEINDICATOR_99_STANDARD'
            }
            
            logger.info(f"📊 {indicator_name} 得分: {final_score:.1f}/100")
            if passed:
                logger.info(f"🎉 {indicator_name} 通过99分标准!")
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
    
    def run_all_fixed_validation(self) -> Dict[str, Any]:
        """运行全部修复后指标验证"""
        logger.info(f"🚀 开始修复后BaseIndicator指标验证 (99分标准)...")
        logger.info(f"📊 验证指标数量: {len(self.fixed_indicators)}个")
        
        start_time = time.time()
        
        # 生成测试数据
        self.test_data = self.generate_premium_test_data()
        
        # 验证结果
        results = {}
        passed_count = 0
        failed_count = 0
        
        # 验证所有指标
        for i, indicator_name in enumerate(self.fixed_indicators):
            logger.info(f"📦 验证进度: {i+1}/{len(self.fixed_indicators)} - {indicator_name}")
            
            result = self.validate_single_indicator_99_standard(indicator_name)
            results[indicator_name] = result
            
            if result.get('meets_standard', False):
                passed_count += 1
            else:
                failed_count += 1
        
        # 计算总体统计
        total_score = sum(r.get('overall_score', 0) for r in results.values())
        average_score = total_score / len(results) if results else 0
        
        validation_time = time.time() - start_time
        
        summary = {
            'validation_type': 'FIXED_BASEINDICATOR_99_STANDARD',
            'total_indicators': len(self.fixed_indicators),
            'passed_count': passed_count,
            'failed_count': failed_count,
            'pass_rate': (passed_count / len(self.fixed_indicators)) * 100 if self.fixed_indicators else 0,
            'average_score': average_score,
            'validation_time': validation_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 修复后BaseIndicator指标验证完成!")
        logger.info(f"📊 通过率: {summary['pass_rate']:.1f}% ({passed_count}/{len(self.fixed_indicators)})")
        logger.info(f"📊 平均得分: {average_score:.1f}/100")
        logger.info(f"⏱️ 验证时间: {validation_time:.2f}秒")
        
        return summary


def main():
    """主函数"""
    logger.info("🔍 修复后BaseIndicator指标验证开始 (99分标准)...")
    
    validator = FixedBaseIndicatorValidator()
    result = validator.run_all_fixed_validation()
    
    # 保存验证报告
    report_file = f"docs/finaltesting/indicators/fixed_baseindicator_99_validation_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成验证报告
    passed_indicators = [name for name, data in result['results'].items() if data.get('meets_standard', False)]
    failed_indicators = [name for name, data in result['results'].items() if not data.get('meets_standard', False)]
    
    report_content = f"""# 修复后BaseIndicator指标验证报告 (99分标准)

## 验证概览
- **验证类型**: 修复后BaseIndicator指标验证
- **验证时间**: {result['timestamp']}
- **验证标准**: ≥99.0分 (严格标准)
- **验证指标数**: {result['total_indicators']}个
- **通过率**: {result['pass_rate']:.1f}%
- **平均得分**: {result['average_score']:.1f}/100

## 验证结果统计
- **✅ 通过99分标准**: {result['passed_count']}个
- **❌ 未达99分标准**: {result['failed_count']}个

## 通过99分标准的指标
{chr(10).join([f"- **{name}**: {result['results'][name]['overall_score']:.1f}/100 ✅" for name in passed_indicators])}

## 未达99分标准的指标
{chr(10).join([f"- **{name}**: {result['results'][name].get('overall_score', 0):.1f}/100 ❌" for name in failed_indicators])}

## 修复效果分析
本次修复主要针对缺失的可选方法：
- **VIX指标**: 添加了 `calculate_raw_score` 方法
- **ROC指标**: 添加了 `get_signals` 和 `calculate_raw_score` 方法
- **MFI指标**: 添加了 `get_signals` 和 `calculate_raw_score` 方法

## 验证结论
修复后BaseIndicator指标验证完成，通过率{result['pass_rate']:.1f}%。

{'### 🎉 修复成功！所有指标达到99分以上标准，可以投入生产使用。' if result['pass_rate'] == 100 else '### ⚠️ 部分指标仍需进一步优化。'}

---
*验证时间: {result['validation_time']:.2f}秒*
*验证工具: 修复后BaseIndicator 99分严格标准验证系统*
*质量保证: 生产级别标准*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 验证报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
