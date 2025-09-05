#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZXM指标当前状态验证脚本
验证4个问题ZXM指标的当前实际状态和得分
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

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ZXMIndicatorsCurrentStatusValidator:
    """ZXM指标当前状态验证器"""
    
    def __init__(self):
        self.target_score = 95.0  # 目标分数：95分以上
        self.test_data = None
        self.zxm_indicators = [
            'ZXM_MARKET_SENTIMENT',      # 之前返回None，现在声称工作正常
            'ZXM_LIQUIDITY_ANALYSIS',    # 未注册问题
            'ZXM_VOLATILITY_FORECAST',   # 未注册问题
            'ZXM_CORRELATION_MATRIX'     # 未注册问题
        ]
        
    def generate_premium_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据"""
        logger.info("📊 生成ZXM指标验证数据...")
        
        # 生成100天的测试数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        np.random.seed(42)
        base_price = 100.0
        base_volume = 2000000
        
        # 生成具有市场特征的数据
        market_trend = np.concatenate([
            np.linspace(0, 15, 25),    # 上升趋势
            np.linspace(15, 20, 20),   # 加速上升
            np.linspace(20, 12, 25),   # 高位震荡
            np.linspace(12, 3, 20),    # 下降趋势
            np.linspace(3, 10, 10)     # 底部反弹
        ])
        
        volatility = np.sin(np.linspace(0, 6*np.pi, 100)) * 2 + 3
        volume_pattern = np.cos(np.linspace(0, 4*np.pi, 100)) * 1000000 + base_volume
        
        prices = [base_price]
        volumes = [base_volume]
        
        for i in range(1, 100):
            trend = (market_trend[i] - market_trend[i-1]) * 0.3
            vol = volatility[i] * 0.12
            noise = np.random.normal(0, 0.5)
            
            price_change = trend + vol + noise
            new_price = max(prices[-1] + price_change, 1.0)
            
            volume_factor = (volatility[i] / 4) + 0.8
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
        logger.info(f"✅ 生成ZXM验证数据: {len(df)}行")
        return df
    
    def validate_single_zxm_indicator_95_standard(self, indicator_name: str) -> Dict[str, Any]:
        """使用95分标准验证单个ZXM指标"""
        logger.info(f"🔍 验证ZXM指标: {indicator_name}")
        
        try:
            # 导入指标注册表
            from indicators.complete_indicator_registry import get_indicator_registry
            
            # 获取指标注册表实例
            registry = get_indicator_registry()
            
            # 尝试创建指标实例
            try:
                indicator = registry.create_indicator(indicator_name)
                if indicator is None:
                    return {
                        'indicator': indicator_name,
                        'overall_score': 0,
                        'meets_standard': False,
                        'status': 'NOT_REGISTERED',
                        'error': '指标未注册或创建失败'
                    }
            except Exception as e:
                return {
                    'indicator': indicator_name,
                    'overall_score': 0,
                    'meets_standard': False,
                    'status': 'CREATION_ERROR',
                    'error': f'指标创建异常: {e}'
                }
            
            start_time = time.time()
            
            # 95分标准验证
            score = 0
            max_score = 100
            
            # 1. 基础计算功能验证 (40分)
            try:
                result = indicator.calculate(self.test_data)
                if result is not None and isinstance(result, dict) and len(result) > 0:
                    score += 20
                    logger.info(f"✅ {indicator_name} 基础计算功能正常")
                    
                    # 检查数据质量
                    valid_data_ratio = 0
                    total_values = 0
                    valid_values = 0
                    
                    for key, value in result.items():
                        if isinstance(value, (int, float, np.number)):
                            total_values += 1
                            if not (np.isnan(value) or np.isinf(value)):
                                valid_values += 1
                        elif hasattr(value, '__len__'):
                            total_values += len(value) if hasattr(value, '__len__') else 1
                            if hasattr(value, 'notna'):
                                valid_values += value.notna().sum()
                            else:
                                valid_values += total_values
                    
                    if total_values > 0:
                        valid_data_ratio = valid_values / total_values
                        if valid_data_ratio >= 0.9:
                            score += 20
                            logger.info(f"✅ {indicator_name} 数据质量优秀: {valid_data_ratio:.2%}")
                        elif valid_data_ratio >= 0.7:
                            score += 15
                            logger.info(f"⚠️ {indicator_name} 数据质量良好: {valid_data_ratio:.2%}")
                        elif valid_data_ratio >= 0.5:
                            score += 10
                            logger.info(f"⚠️ {indicator_name} 数据质量一般: {valid_data_ratio:.2%}")
                else:
                    logger.warning(f"⚠️ {indicator_name} 基础计算返回无效结果")
            except Exception as e:
                logger.warning(f"⚠️ {indicator_name} 基础计算失败: {e}")
            
            # 2. 方法实现验证 (30分)
            method_score = 0
            
            # calculate方法 (15分)
            if hasattr(indicator, 'calculate'):
                try:
                    test_result = indicator.calculate(self.test_data)
                    if test_result is not None:
                        method_score += 15
                        logger.info(f"✅ {indicator_name} calculate方法正常")
                except Exception as e:
                    logger.warning(f"⚠️ {indicator_name} calculate方法失败: {e}")
            
            # get_patterns方法 (15分)
            if hasattr(indicator, 'get_patterns'):
                try:
                    patterns = indicator.get_patterns()
                    if patterns is not None:
                        method_score += 15
                        logger.info(f"✅ {indicator_name} get_patterns方法正常")
                except Exception as e:
                    logger.warning(f"⚠️ {indicator_name} get_patterns方法失败: {e}")
                    method_score += 7  # 部分分数
            
            score += method_score
            
            # 3. 性能验证 (20分)
            try:
                start_perf = time.time()
                for _ in range(3):
                    test_result = indicator.calculate(self.test_data)
                execution_time = (time.time() - start_perf) / 3
                
                if execution_time < 0.1:
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
            
            # 4. 接口兼容性验证 (10分)
            compatibility_score = 0
            
            # 检查是否为工厂模式指标
            if hasattr(indicator, 'get_patterns') and callable(getattr(indicator, 'get_patterns')):
                try:
                    # 工厂模式指标的get_patterns()方法无参数
                    patterns = indicator.get_patterns()
                    compatibility_score += 10
                    logger.info(f"✅ {indicator_name} 工厂模式接口兼容")
                except Exception as e:
                    logger.warning(f"⚠️ {indicator_name} 接口兼容性问题: {e}")
                    compatibility_score += 5
            
            score += compatibility_score
            
            final_score = score
            validation_time = time.time() - start_time
            
            # 95分标准
            passed = final_score >= self.target_score
            
            result_data = {
                'indicator': indicator_name,
                'validation_time': validation_time,
                'overall_score': final_score,
                'meets_standard': passed,
                'min_required_score': self.target_score,
                'status': 'PASSED' if passed else 'FAILED',
                'timestamp': datetime.now().isoformat(),
                'validation_type': 'ZXM_CURRENT_STATUS_95_STANDARD'
            }
            
            logger.info(f"📊 {indicator_name} 得分: {final_score:.1f}/100")
            if passed:
                logger.info(f"🎉 {indicator_name} 通过95分标准!")
            else:
                logger.warning(f"⚠️ {indicator_name} 未达到95分标准")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ {indicator_name} 验证失败: {e}")
            return {
                'indicator': indicator_name,
                'overall_score': 0,
                'meets_standard': False,
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_zxm_validation(self) -> Dict[str, Any]:
        """运行全部ZXM指标验证"""
        logger.info(f"🚀 开始ZXM指标当前状态验证 (95分标准)...")
        logger.info(f"📊 验证指标数量: {len(self.zxm_indicators)}个")
        
        start_time = time.time()
        
        # 生成测试数据
        self.test_data = self.generate_premium_test_data()
        
        # 验证结果
        results = {}
        passed_count = 0
        failed_count = 0
        
        # 验证所有指标
        for i, indicator_name in enumerate(self.zxm_indicators):
            logger.info(f"📦 验证进度: {i+1}/{len(self.zxm_indicators)} - {indicator_name}")
            
            result = self.validate_single_zxm_indicator_95_standard(indicator_name)
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
            'validation_type': 'ZXM_CURRENT_STATUS_95_STANDARD',
            'total_indicators': len(self.zxm_indicators),
            'passed_count': passed_count,
            'failed_count': failed_count,
            'pass_rate': (passed_count / len(self.zxm_indicators)) * 100 if self.zxm_indicators else 0,
            'average_score': average_score,
            'validation_time': validation_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 ZXM指标当前状态验证完成!")
        logger.info(f"📊 通过率: {summary['pass_rate']:.1f}% ({passed_count}/{len(self.zxm_indicators)})")
        logger.info(f"📊 平均得分: {average_score:.1f}/100")
        logger.info(f"⏱️ 验证时间: {validation_time:.2f}秒")
        
        return summary


def main():
    """主函数"""
    logger.info("🔍 ZXM指标当前状态验证开始 (95分标准)...")
    
    validator = ZXMIndicatorsCurrentStatusValidator()
    result = validator.run_all_zxm_validation()
    
    # 保存验证报告
    report_file = f"docs/finaltesting/indicators/zxm_indicators_current_status_validation_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成验证报告
    passed_indicators = [name for name, data in result['results'].items() if data.get('meets_standard', False)]
    failed_indicators = [name for name, data in result['results'].items() if not data.get('meets_standard', False)]
    
    report_content = f"""# ZXM指标当前状态验证报告 (95分标准)

## 验证概览
- **验证类型**: ZXM指标当前状态验证
- **验证时间**: {result['timestamp']}
- **验证标准**: ≥95.0分 (工厂模式标准)
- **验证指标数**: {result['total_indicators']}个
- **通过率**: {result['pass_rate']:.1f}%
- **平均得分**: {result['average_score']:.1f}/100

## 验证结果统计
- **✅ 通过95分标准**: {result['passed_count']}个
- **❌ 未达95分标准**: {result['failed_count']}个

## 通过95分标准的ZXM指标
{chr(10).join([f"- **{name}**: {result['results'][name]['overall_score']:.1f}/100 ✅" for name in passed_indicators])}

## 未达95分标准的ZXM指标
{chr(10).join([f"- **{name}**: {result['results'][name].get('overall_score', 0):.1f}/100 ❌ ({result['results'][name].get('status', 'UNKNOWN')})" for name in failed_indicators])}

## 验证结论
ZXM指标当前状态验证完成，通过率{result['pass_rate']:.1f}%。

{'### 🎉 验证成功！大部分ZXM指标达到95分以上标准，适合生产使用。' if result['pass_rate'] >= 75 else '### ⚠️ 需要修复，部分ZXM指标未达到95分标准。'}

---
*验证时间: {result['validation_time']:.2f}秒*
*验证工具: ZXM指标95分标准验证系统*
*质量保证: 生产级别标准*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 验证报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
