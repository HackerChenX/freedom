#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
形态识别指标问题分析脚本
分析19个形态识别指标为什么都获得85分而不是95分以上，找出具体的改进点
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


class PatternIndicatorsIssuesAnalyzer:
    """形态识别指标问题分析器"""
    
    def __init__(self):
        self.target_score = 95.0  # 目标分数：95分以上
        self.current_score = 85.0  # 当前分数：85分
        self.test_data = None
        
        # 19个形态识别指标
        self.pattern_indicators = [
            'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI', 'PIERCING_LINE',
            'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR', 'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS',
            'V_SHAPED_REVERSAL', 'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'TRIANGLE',
            'WEDGE', 'FLAG', 'PENNANT'
        ]
        
    def generate_premium_test_data(self) -> pd.DataFrame:
        """生成高质量测试数据，特别适合形态识别"""
        logger.info("📊 生成形态识别指标测试数据...")
        
        # 生成100天的测试数据，包含各种形态特征
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        np.random.seed(42)
        base_price = 100.0
        base_volume = 1500000
        
        # 生成包含各种形态的价格数据
        pattern_phases = np.concatenate([
            np.linspace(0, 10, 20),    # 上升趋势（可能形成锤子线等）
            np.linspace(10, 15, 15),   # 加速上升（可能形成流星线）
            np.linspace(15, 12, 20),   # 高位震荡（可能形成十字星）
            np.linspace(12, 3, 25),    # 下降趋势（可能形成吞没形态）
            np.linspace(3, 8, 20)      # 底部反弹（可能形成早晨之星）
        ])
        
        # 添加形态特征的波动性
        volatility = np.sin(np.linspace(0, 6*np.pi, 100)) * 1.5 + 2.0
        volume_pattern = np.cos(np.linspace(0, 4*np.pi, 100)) * 500000 + base_volume
        
        # 生成价格序列
        prices = [base_price]
        volumes = [base_volume]
        
        for i in range(1, 100):
            trend = (pattern_phases[i] - pattern_phases[i-1]) * 0.3
            vol = volatility[i] * 0.1
            noise = np.random.normal(0, 0.4)
            
            price_change = trend + vol + noise
            new_price = max(prices[-1] + price_change, 1.0)
            
            volume_factor = (volatility[i] / 3) + 0.8
            new_volume = int(volume_pattern[i] * volume_factor * np.random.uniform(0.8, 1.2))
            
            prices.append(new_price)
            volumes.append(max(new_volume, 100000))
        
        # 生成高质量OHLC数据，特别适合形态识别
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            daily_vol = volatility[i] * 0.01
            
            # 生成更真实的OHLC，有利于形态识别
            high_factor = np.random.uniform(1.0, 1.0 + daily_vol)
            low_factor = np.random.uniform(1.0 - daily_vol, 1.0)
            
            high = price * high_factor
            low = price * low_factor
            
            open_price = prices[i-1] if i > 0 else price
            close = price
            
            # 确保OHLC关系正确
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # 偶尔生成特殊形态（如十字星、锤子线等）
            if i % 15 == 0:  # 每15天可能出现特殊形态
                if np.random.random() > 0.5:
                    # 十字星形态：开盘价接近收盘价
                    close = open_price + np.random.uniform(-0.1, 0.1)
                    high = max(open_price, close) + abs(open_price - close) * 2
                    low = min(open_price, close) - abs(open_price - close) * 2
            
            data.append({
                'date': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ 生成形态识别测试数据: {len(df)}行")
        return df
    
    def analyze_single_pattern_indicator_detailed(self, indicator_name: str) -> Dict[str, Any]:
        """详细分析单个形态识别指标的问题"""
        logger.info(f"🔍 详细分析形态识别指标: {indicator_name}")
        
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
                        elif hasattr(value, '__len__'):
                            total_values += len(value) if hasattr(value, '__len__') else 1
                            if hasattr(value, 'notna'):
                                valid_values += value.notna().sum()
                            else:
                                valid_values += total_values
                    
                    if total_values > 0:
                        valid_data_ratio = valid_values / total_values
                        if valid_data_ratio >= 0.95:
                            analysis['scores']['data_quality'] = 20
                        elif valid_data_ratio >= 0.9:
                            analysis['scores']['data_quality'] = 15
                            analysis['issues'].append(f'数据质量{valid_data_ratio:.1%}，需要提升到95%以上')
                        else:
                            analysis['scores']['data_quality'] = 10
                            analysis['issues'].append(f'数据质量{valid_data_ratio:.1%}，严重不足')
                else:
                    analysis['scores']['basic_calculation'] = 0
                    analysis['issues'].append('基础计算返回无效结果')
            except Exception as e:
                analysis['scores']['basic_calculation'] = 0
                analysis['issues'].append(f'基础计算失败: {e}')
            
            # 2. 形态识别特征分析 (30分) - 关键失分点
            pattern_score = 0
            
            # 检查是否包含形态识别相关的键
            if result is not None and isinstance(result, dict):
                pattern_keys = ['pattern', 'signal', 'strength', 'confidence', 'detected', 'formation']
                found_pattern_keys = [key for key in result.keys() if any(pk in key.lower() for pk in pattern_keys)]
                
                if len(found_pattern_keys) >= 3:
                    pattern_score += 15
                elif len(found_pattern_keys) >= 1:
                    pattern_score += 10
                    analysis['issues'].append(f'形态识别键不够丰富，只有{found_pattern_keys}')
                else:
                    analysis['issues'].append('缺少形态识别相关的键')
                
                # 检查形态识别的具体性
                specific_patterns = ['bullish', 'bearish', 'reversal', 'continuation', 'breakout']
                found_specific = [key for key in result.keys() if any(sp in str(result[key]).lower() for sp in specific_patterns)]
                
                if len(found_specific) >= 2:
                    pattern_score += 15
                elif len(found_specific) >= 1:
                    pattern_score += 10
                    analysis['issues'].append('形态识别特征不够具体')
                else:
                    analysis['issues'].append('缺少具体的形态识别特征')
            
            analysis['scores']['pattern_features'] = pattern_score
            
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
            
            # 4. 性能分析 (10分)
            try:
                start_perf = time.time()
                for _ in range(3):
                    test_result = indicator.calculate(self.test_data)
                execution_time = (time.time() - start_perf) / 3
                
                if execution_time < 0.1:
                    analysis['scores']['performance'] = 10
                elif execution_time < 0.5:
                    analysis['scores']['performance'] = 7
                else:
                    analysis['scores']['performance'] = 5
                    analysis['issues'].append(f'性能较慢: {execution_time:.3f}秒')
            except Exception as e:
                analysis['scores']['performance'] = 0
                analysis['issues'].append(f'性能测试失败: {e}')
            
            # 计算总分
            total_score = sum(analysis['scores'].values())
            analysis['total_score'] = total_score
            analysis['target_score'] = self.target_score
            analysis['score_gap'] = self.target_score - total_score
            
            # 生成改进建议
            if total_score < self.target_score:
                if analysis['scores'].get('pattern_features', 0) < 25:
                    analysis['recommendations'].append('增强形态识别特征：添加更多pattern、signal、confidence等键')
                if analysis['scores'].get('data_quality', 0) < 20:
                    analysis['recommendations'].append('提升数据质量：确保95%以上的有效数据')
                if analysis['scores'].get('method_implementation', 0) < 20:
                    analysis['recommendations'].append('完善方法实现：优化get_patterns方法')
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 分析{indicator_name}失败: {e}")
            return {'error': str(e)}
    
    def run_all_pattern_analysis(self) -> Dict[str, Any]:
        """运行全部形态识别指标分析"""
        logger.info(f"🚀 开始形态识别指标问题分析...")
        logger.info(f"📊 分析指标数量: {len(self.pattern_indicators)}个")
        
        start_time = time.time()
        
        # 生成测试数据
        self.test_data = self.generate_premium_test_data()
        
        # 分析结果
        results = {}
        common_issues = {}
        
        # 分析所有指标
        for i, indicator_name in enumerate(self.pattern_indicators):
            logger.info(f"📦 分析进度: {i+1}/{len(self.pattern_indicators)} - {indicator_name}")
            
            result = self.analyze_single_pattern_indicator_detailed(indicator_name)
            results[indicator_name] = result
            
            # 统计常见问题
            if 'issues' in result:
                for issue in result['issues']:
                    if issue not in common_issues:
                        common_issues[issue] = 0
                    common_issues[issue] += 1
        
        analysis_time = time.time() - start_time
        
        # 计算统计信息
        total_scores = [r.get('total_score', 0) for r in results.values() if 'total_score' in r]
        average_score = sum(total_scores) / len(total_scores) if total_scores else 0
        
        summary = {
            'analysis_type': 'PATTERN_INDICATORS_ISSUES_ANALYSIS',
            'total_indicators': len(self.pattern_indicators),
            'average_score': average_score,
            'target_score': self.target_score,
            'score_gap': self.target_score - average_score,
            'common_issues': common_issues,
            'analysis_time': analysis_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 形态识别指标问题分析完成!")
        logger.info(f"📊 平均得分: {average_score:.1f}/100")
        logger.info(f"📊 目标差距: {self.target_score - average_score:.1f}分")
        logger.info(f"⏱️ 分析时间: {analysis_time:.2f}秒")
        
        return summary


def main():
    """主函数"""
    logger.info("🔍 形态识别指标问题分析开始...")
    
    analyzer = PatternIndicatorsIssuesAnalyzer()
    result = analyzer.run_all_pattern_analysis()
    
    # 保存分析报告
    report_file = f"docs/finaltesting/indicators/pattern_indicators_issues_analysis.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 生成分析报告
    common_issues_text = "\n".join([f"- **{issue}**: {count}个指标" for issue, count in result['common_issues'].items()])
    
    report_content = f"""# 形态识别指标问题分析报告

## 分析概览
- **分析类型**: 形态识别指标问题分析
- **分析时间**: {result['timestamp']}
- **目标**: 分析19个形态识别指标为什么获得85分而不是95分以上
- **分析指标数**: {result['total_indicators']}个

## 分析结果统计
- **平均得分**: {result['average_score']:.1f}/100
- **目标得分**: {result['target_score']:.1f}/100
- **得分差距**: {result['score_gap']:.1f}分

## 常见问题统计
{common_issues_text}

## 主要改进方向
基于分析结果，形态识别指标的主要改进方向：

1. **增强形态识别特征**: 添加更多pattern、signal、confidence等键
2. **提升数据质量**: 确保95%以上的有效数据
3. **完善方法实现**: 优化get_patterns方法
4. **增加形态特异性**: 添加具体的形态识别特征

## 下一步行动
1. 实施具体的形态识别特征增强
2. 优化数据质量处理逻辑
3. 重新运行95分标准验证
4. 确认所有形态识别指标达到95分以上标准

---
*分析时间: {result['analysis_time']:.2f}秒*
*分析工具: 形态识别指标问题分析系统*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 问题分析报告已保存: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
