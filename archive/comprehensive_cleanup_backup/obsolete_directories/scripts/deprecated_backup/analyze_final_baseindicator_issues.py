#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终BaseIndicator指标问题分析脚本
分析SYNERGY和UNIFIED_MA两个指标的具体问题，制定修复方案
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


class FinalBaseIndicatorAnalyzer:
    """最终BaseIndicator指标分析器"""
    
    def __init__(self):
        self.target_score = 99.0  # BaseIndicator目标分数：99分以上
        self.test_data = None
        
        # 最后2个需要修复的BaseIndicator指标
        self.final_indicators = ['SYNERGY', 'UNIFIED_MA']
        
        # 当前得分记录
        self.current_scores = {
            'SYNERGY': 73.8,      # 需要大幅提升
            'UNIFIED_MA': 88.8    # 需要进一步优化
        }
        
    def generate_premium_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据，适合BaseIndicator验证"""
        logger.info("📊 生成BaseIndicator指标测试数据...")
        
        # 生成200天的测试数据，确保足够的历史数据
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        
        np.random.seed(42)
        base_price = 100.0
        base_volume = 1500000
        
        # 生成具有趋势和周期性的价格数据
        trend = np.linspace(0, 20, 200)  # 长期上升趋势
        cycle = 10 * np.sin(np.linspace(0, 8*np.pi, 200))  # 周期性波动
        noise = np.random.normal(0, 2, 200)  # 随机噪声
        
        price_changes = trend + cycle + noise
        
        # 生成价格序列
        prices = [base_price]
        volumes = []
        
        for i in range(1, 200):
            new_price = max(prices[-1] + price_changes[i] * 0.5, 1.0)
            prices.append(new_price)
            
            # 成交量与价格变化相关
            price_change_pct = abs(price_changes[i]) / prices[-1]
            volume_factor = 1 + price_change_pct * 2
            new_volume = int(base_volume * volume_factor * np.random.uniform(0.8, 1.2))
            volumes.append(max(new_volume, 100000))
        
        volumes.append(base_volume)  # 为第一天添加成交量
        
        # 生成OHLC数据
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            daily_volatility = abs(price_changes[i]) * 0.01
            
            # 生成OHLC
            high_factor = np.random.uniform(1.0, 1.0 + daily_volatility)
            low_factor = np.random.uniform(1.0 - daily_volatility, 1.0)
            
            high = price * high_factor
            low = price * low_factor
            
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
        logger.info(f"✅ 生成BaseIndicator测试数据: {len(df)}行")
        return df
    
    def analyze_single_baseindicator_detailed(self, indicator_name: str) -> Dict[str, Any]:
        """详细分析单个BaseIndicator指标的问题"""
        logger.info(f"🔍 详细分析BaseIndicator指标: {indicator_name}")
        
        try:
            # 导入指标注册表
            from indicators.complete_indicator_registry import get_indicator_registry
            
            # 获取指标注册表实例
            registry = get_indicator_registry()
            
            # 创建指标实例
            indicator = registry.create_indicator(indicator_name)
            if indicator is None:
                return {'error': f'指标{indicator_name}创建失败'}
            
            analysis = {
                'indicator': indicator_name,
                'current_score': self.current_scores.get(indicator_name, 0),
                'target_score': self.target_score,
                'issues': [],
                'scores': {},
                'recommendations': []
            }
            
            # 1. 基础计算功能分析 (40分)
            try:
                result = indicator.calculate(self.test_data)
                if result is not None and isinstance(result, dict) and len(result) > 0:
                    analysis['scores']['basic_calculation'] = 20
                    
                    # 检查数据质量
                    valid_data_ratio = 0
                    total_values = 0
                    valid_values = 0
                    
                    for key, value in result.items():
                        if isinstance(value, (int, float, np.number)):
                            total_values += 1
                            if not (np.isnan(value) or np.isinf(value)):
                                valid_values += 1
                        elif hasattr(value, '__len__') and hasattr(value, 'notna'):
                            total_values += len(value)
                            valid_values += value.notna().sum()
                        else:
                            total_values += 1
                            valid_values += 1
                    
                    if total_values > 0:
                        valid_data_ratio = valid_values / total_values
                        if valid_data_ratio >= 0.98:
                            analysis['scores']['data_quality'] = 20
                        elif valid_data_ratio >= 0.95:
                            analysis['scores']['data_quality'] = 15
                        elif valid_data_ratio >= 0.9:
                            analysis['scores']['data_quality'] = 10
                        else:
                            analysis['scores']['data_quality'] = 5
                            analysis['issues'].append(f'数据质量{valid_data_ratio:.1%}，需要提升到98%以上')
                else:
                    analysis['scores']['basic_calculation'] = 0
                    analysis['issues'].append('基础计算返回无效结果')
            except Exception as e:
                analysis['scores']['basic_calculation'] = 0
                analysis['issues'].append(f'基础计算失败: {e}')
            
            # 2. 算法真实性分析 (30分) - BaseIndicator关键要求
            algorithm_score = 0
            
            # 检查是否有真实的数学计算
            if result is not None and isinstance(result, dict):
                # 检查数值合理性
                numeric_values = [v for v in result.values() if isinstance(v, (int, float, np.number))]
                if numeric_values:
                    # 检查数值范围合理性
                    reasonable_values = [v for v in numeric_values if not (np.isnan(v) or np.isinf(v)) and abs(v) < 1e6]
                    if len(reasonable_values) / len(numeric_values) >= 0.95:
                        algorithm_score += 15
                    else:
                        analysis['issues'].append('数值范围不合理，可能存在计算错误')
                
                # 检查计算复杂度
                if len(result) >= 5:
                    algorithm_score += 15
                elif len(result) >= 3:
                    algorithm_score += 10
                else:
                    analysis['issues'].append('计算结果过于简单，缺少复杂算法')
            
            analysis['scores']['algorithm_authenticity'] = algorithm_score
            
            # 3. 方法实现分析 (20分)
            method_score = 0
            
            # calculate方法
            if hasattr(indicator, 'calculate'):
                method_score += 10
            
            # get_patterns方法
            if hasattr(indicator, 'get_patterns'):
                try:
                    patterns = indicator.get_patterns()
                    if patterns is not None and isinstance(patterns, dict):
                        method_score += 10
                    else:
                        method_score += 5
                        analysis['issues'].append('get_patterns返回结果不完整')
                except Exception as e:
                    method_score += 5
                    analysis['issues'].append(f'get_patterns方法异常: {e}')
            
            analysis['scores']['method_implementation'] = method_score
            
            # 4. 性能分析 (9分)
            try:
                start_perf = time.time()
                for _ in range(5):
                    test_result = indicator.calculate(self.test_data)
                execution_time = (time.time() - start_perf) / 5
                
                if execution_time < 0.05:
                    analysis['scores']['performance'] = 9
                elif execution_time < 0.1:
                    analysis['scores']['performance'] = 7
                elif execution_time < 0.5:
                    analysis['scores']['performance'] = 5
                else:
                    analysis['scores']['performance'] = 3
                    analysis['issues'].append(f'性能较慢: {execution_time:.3f}秒')
            except Exception as e:
                analysis['scores']['performance'] = 0
                analysis['issues'].append(f'性能测试失败: {e}')
            
            # 计算总分
            total_score = sum(analysis['scores'].values())
            analysis['total_score'] = total_score
            analysis['score_gap'] = self.target_score - total_score
            
            # 生成改进建议
            if total_score < self.target_score:
                if analysis['scores'].get('algorithm_authenticity', 0) < 25:
                    analysis['recommendations'].append('增强算法真实性：实现更复杂的数学计算')
                if analysis['scores'].get('data_quality', 0) < 20:
                    analysis['recommendations'].append('提升数据质量：确保98%以上的有效数据')
                if analysis['scores'].get('method_implementation', 0) < 20:
                    analysis['recommendations'].append('完善方法实现：优化calculate和get_patterns方法')
                if analysis['scores'].get('performance', 0) < 9:
                    analysis['recommendations'].append('优化性能：减少计算时间到50ms以下')
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 分析{indicator_name}失败: {e}")
            return {'error': str(e)}
    
    def run_final_baseindicator_analysis(self) -> Dict[str, Any]:
        """运行最终BaseIndicator指标分析"""
        logger.info(f"🚀 开始最终BaseIndicator指标分析...")
        logger.info(f"📊 分析指标数量: {len(self.final_indicators)}个")
        
        start_time = time.time()
        
        # 生成测试数据
        self.test_data = self.generate_premium_test_data()
        
        # 分析结果
        results = {}
        
        # 分析所有指标
        for i, indicator_name in enumerate(self.final_indicators):
            logger.info(f"📦 分析进度: {i+1}/{len(self.final_indicators)} - {indicator_name}")
            
            result = self.analyze_single_baseindicator_detailed(indicator_name)
            results[indicator_name] = result
        
        analysis_time = time.time() - start_time
        
        # 计算统计信息
        total_scores = [r.get('total_score', 0) for r in results.values() if 'total_score' in r]
        average_score = sum(total_scores) / len(total_scores) if total_scores else 0
        
        summary = {
            'analysis_type': 'FINAL_BASEINDICATOR_ANALYSIS',
            'total_indicators': len(self.final_indicators),
            'average_score': average_score,
            'target_score': self.target_score,
            'score_gap': self.target_score - average_score,
            'analysis_time': analysis_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 最终BaseIndicator指标分析完成!")
        logger.info(f"📊 平均得分: {average_score:.1f}/100")
        logger.info(f"📊 目标差距: {self.target_score - average_score:.1f}分")
        logger.info(f"⏱️ 分析时间: {analysis_time:.2f}秒")
        
        return summary


def main():
    """主函数"""
    logger.info("🔍 最终BaseIndicator指标分析开始...")
    
    analyzer = FinalBaseIndicatorAnalyzer()
    result = analyzer.run_final_baseindicator_analysis()
    
    # 保存分析报告
    report_file = f"docs/finaltesting/indicators/final_baseindicator_analysis_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成分析报告
    report_content = f"""# 最终BaseIndicator指标分析报告

## 分析概览
- **分析类型**: 最终BaseIndicator指标分析
- **分析时间**: {result['timestamp']}
- **目标**: 分析SYNERGY和UNIFIED_MA指标问题，制定99分修复方案
- **分析指标数**: {result['total_indicators']}个

## 分析结果统计
- **平均得分**: {result['average_score']:.1f}/100
- **目标得分**: {result['target_score']:.1f}/100
- **得分差距**: {result['score_gap']:.1f}分

## 指标详细分析

### SYNERGY指标分析
{f"- **当前得分**: {result['results']['SYNERGY'].get('total_score', 0):.1f}/100" if 'SYNERGY' in result['results'] else "- **状态**: 分析失败"}
{f"- **主要问题**: {', '.join(result['results']['SYNERGY'].get('issues', []))}" if 'SYNERGY' in result['results'] and result['results']['SYNERGY'].get('issues') else ""}
{f"- **改进建议**: {', '.join(result['results']['SYNERGY'].get('recommendations', []))}" if 'SYNERGY' in result['results'] and result['results']['SYNERGY'].get('recommendations') else ""}

### UNIFIED_MA指标分析
{f"- **当前得分**: {result['results']['UNIFIED_MA'].get('total_score', 0):.1f}/100" if 'UNIFIED_MA' in result['results'] else "- **状态**: 分析失败"}
{f"- **主要问题**: {', '.join(result['results']['UNIFIED_MA'].get('issues', []))}" if 'UNIFIED_MA' in result['results'] and result['results']['UNIFIED_MA'].get('issues') else ""}
{f"- **改进建议**: {', '.join(result['results']['UNIFIED_MA'].get('recommendations', []))}" if 'UNIFIED_MA' in result['results'] and result['results']['UNIFIED_MA'].get('recommendations') else ""}

## 修复策略

### 总体修复方向
1. **算法真实性增强**: 实现更复杂的数学计算逻辑
2. **数据质量提升**: 确保98%以上的有效数据
3. **方法完善**: 优化calculate和get_patterns方法
4. **性能优化**: 将计算时间控制在50ms以下

### 下一步行动
1. 实施针对性的算法重构
2. 优化数据处理逻辑
3. 重新运行99分标准验证
4. 确认所有BaseIndicator指标达到99分以上标准

---
*分析时间: {result['analysis_time']:.2f}秒*
*分析工具: 最终BaseIndicator指标分析系统*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 分析报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
