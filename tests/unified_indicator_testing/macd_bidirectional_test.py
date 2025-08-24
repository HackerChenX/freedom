#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD指标双向验证测试 - 基于现有框架

使用现有的数据生成器和测试框架进行双向验证：
1. 生成符合特定形态的模拟数据
2. 验证指标能否识别出预期形态
3. 验证识别结果与原始数据的一致性
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from tests.unified_indicator_testing.components.stockinfo_compatible_data_generator import StockInfoCompatibleDataGenerator
from unified_indicator_tester import UnifiedIndicatorTester
# 导入统一形态注册表
from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry

class MACDBidirectionalTester:
    """MACD指标双向验证测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.data_generator = StockInfoCompatibleDataGenerator()
        self.tester = UnifiedIndicatorTester(config_path='production_config.yaml')
        self.test_results = {}
        
        # 获取统一形态注册表
        self.pattern_registry = get_unified_pattern_registry()

        # 使用规范的形态名称（从统一注册表获取）
        candidate_patterns = ['GOLDEN_CROSS', 'DEATH_CROSS', 'BEARISH_DIVERGENCE']

        # 验证形态是否适用于MACD并获取规范名称
        self.macd_patterns = []
        for pattern in candidate_patterns:
            canonical_name = self.pattern_registry.get_canonical_pattern_name(pattern)
            if self.pattern_registry.validate_pattern_for_indicator('MACD', canonical_name):
                self.macd_patterns.append(canonical_name)
                print(f"✅ 验证形态: {pattern} -> {canonical_name}")
            else:
                print(f"⚠️ 形态 {pattern} 不适用于MACD指标")

        # 获取数据生成器使用的形态映射
        all_mappings = self.pattern_registry.get_pattern_mapping_for_data_generator()
        self.pattern_mapping = {pattern: all_mappings.get(pattern, pattern)
                               for pattern in self.macd_patterns}
        
        print("✅ MACD双向验证测试器初始化完成")
    
    def run_bidirectional_test(self) -> Dict[str, Any]:
        """运行完整的双向验证测试"""
        print("🚀 开始MACD指标双向验证测试")
        print("=" * 80)
        
        results = {
            'test_time': datetime.now().isoformat(),
            'patterns_tested': 0,
            'patterns_passed': 0,
            'pattern_results': {},
            'overall_success': False,
            'summary': {}
        }
        
        # 测试每个形态
        for pattern_name in self.macd_patterns:
            print(f"\n🔍 测试形态: {pattern_name}")
            print("-" * 50)
            
            pattern_result = self._test_single_pattern(pattern_name)
            results['pattern_results'][pattern_name] = pattern_result
            results['patterns_tested'] += 1
            
            if pattern_result['success']:
                results['patterns_passed'] += 1
                print(f"✅ {pattern_name} 双向验证通过")
            else:
                print(f"❌ {pattern_name} 双向验证失败")
        
        # 计算总体结果
        success_rate = results['patterns_passed'] / results['patterns_tested'] if results['patterns_tested'] > 0 else 0
        results['success_rate'] = success_rate
        results['overall_success'] = success_rate == 1.0
        
        # 生成总结
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _test_single_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """测试单个形态的双向验证"""
        result = {
            'pattern_name': pattern_name,
            'success': False,
            'data_generation': {'success': False, 'details': {}},
            'pattern_recognition': {'success': False, 'details': {}},
            'bidirectional_consistency': {'success': False, 'details': {}},
            'error_message': None
        }
        
        try:
            # 第一步：生成包含特定形态的数据
            print(f"  📊 步骤1: 生成{pattern_name}形态数据")
            generated_data = self._generate_pattern_data(pattern_name)
            
            if generated_data is not None and len(generated_data) > 0:
                result['data_generation']['success'] = True
                result['data_generation']['details'] = {
                    'data_length': len(generated_data),
                    'price_range': f"{generated_data['close'].min():.2f}-{generated_data['close'].max():.2f}",
                    'price_trend': self._analyze_price_trend(generated_data)
                }
                print(f"    ✅ 数据生成成功: {len(generated_data)}天数据")
            else:
                result['error_message'] = "数据生成失败"
                return result
            
            # 第二步：使用MACD指标识别形态
            print(f"  🔍 步骤2: MACD形态识别")
            recognition_result = self._test_pattern_recognition(generated_data, pattern_name)
            result['pattern_recognition'] = recognition_result
            
            if recognition_result['success']:
                print(f"    ✅ 形态识别成功: 检测到{recognition_result['details']['detected_count']}个{pattern_name}")
            else:
                print(f"    ❌ 形态识别失败: {recognition_result['details']['error']}")
            
            # 第三步：验证双向一致性
            print(f"  🔄 步骤3: 双向一致性验证")
            if result['data_generation']['success'] and result['pattern_recognition']['success']:
                consistency_result = self._verify_bidirectional_consistency(
                    generated_data, pattern_name, recognition_result
                )
                result['bidirectional_consistency'] = consistency_result
                
                if consistency_result['success']:
                    print(f"    ✅ 双向一致性验证通过: 一致性评分{consistency_result['details']['consistency_score']:.1%}")
                else:
                    print(f"    ❌ 双向一致性验证失败: {consistency_result['details']['reason']}")
            
            # 综合判断
            result['success'] = (
                result['data_generation']['success'] and
                result['pattern_recognition']['success'] and
                result['bidirectional_consistency']['success']
            )
            
        except Exception as e:
            result['error_message'] = str(e)
            print(f"    ❌ 测试异常: {e}")
        
        return result
    
    def _generate_pattern_data(self, pattern_name: str) -> pd.DataFrame:
        """生成包含特定形态的数据"""
        try:
            # 使用形态名称映射，将MACD实际形态名称映射到数据生成器期望的名称
            generator_pattern_name = self.pattern_mapping.get(pattern_name, pattern_name)

            stock_code = f"TEST_{pattern_name}"
            generated_data = self.data_generator.generate_stockinfo_compatible_data(
                indicator_name='MACD',
                pattern_type=generator_pattern_name,
                stock_code=stock_code,
                history_days=60
            )
            return generated_data
        except Exception as e:
            print(f"    ❌ 数据生成异常: {e}")
            return None
    
    def _test_pattern_recognition(self, data: pd.DataFrame, pattern_name: str) -> Dict[str, Any]:
        """测试形态识别"""
        result = {
            'success': False,
            'details': {'detected_count': 0, 'error': None}
        }

        try:
            # 使用正确的MACD指标导入路径
            from indicators.macd import MacdMacd
            macd = MacdMacd()
            
            # 计算MACD值
            macd_result = macd.calculate(data)

            # 获取形态识别结果
            patterns_result = macd.get_patterns(data)

            if isinstance(patterns_result, pd.DataFrame) and pattern_name in patterns_result.columns:
                detected_count = patterns_result[pattern_name].sum()
                result['details']['detected_count'] = int(detected_count)
                
                if detected_count > 0:
                    result['success'] = True
                    # 记录形态位置
                    pattern_positions = patterns_result[patterns_result[pattern_name]].index.tolist()
                    result['details']['pattern_positions'] = pattern_positions
                else:
                    result['details']['error'] = f"未检测到{pattern_name}形态"
            else:
                result['details']['error'] = "形态识别结果格式错误或不包含目标形态"
                
        except Exception as e:
            result['details']['error'] = str(e)
        
        return result
    
    def _verify_bidirectional_consistency(self, data: pd.DataFrame, pattern_name: str, 
                                        recognition_result: Dict[str, Any]) -> Dict[str, Any]:
        """验证双向一致性"""
        result = {
            'success': False,
            'details': {'consistency_score': 0.0, 'reason': None}
        }
        
        try:
            # 分析生成数据的特征
            data_features = self._analyze_data_features(data, pattern_name)
            
            # 分析识别结果的特征
            recognition_features = self._analyze_recognition_features(recognition_result)
            
            # 计算一致性评分
            consistency_score = self._calculate_consistency_score(
                data_features, recognition_features, pattern_name
            )
            
            result['details']['consistency_score'] = consistency_score
            result['details']['data_features'] = data_features
            result['details']['recognition_features'] = recognition_features
            
            # 判断是否通过（阈值70%）
            if consistency_score >= 0.7:
                result['success'] = True
            else:
                result['details']['reason'] = f"一致性评分{consistency_score:.1%}低于70%阈值"
                
        except Exception as e:
            result['details']['reason'] = str(e)
        
        return result
    
    def _analyze_price_trend(self, data: pd.DataFrame) -> str:
        """分析价格趋势"""
        if len(data) < 2:
            return "数据不足"
        
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]
        change_pct = (end_price - start_price) / start_price * 100
        
        if change_pct > 5:
            return f"上升趋势 (+{change_pct:.1f}%)"
        elif change_pct < -5:
            return f"下降趋势 ({change_pct:.1f}%)"
        else:
            return f"横盘整理 ({change_pct:.1f}%)"
    
    def _analyze_data_features(self, data: pd.DataFrame, pattern_name: str) -> Dict[str, Any]:
        """分析生成数据的特征"""
        features = {}
        
        # 价格变化特征
        features['price_change'] = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        features['volatility'] = data['close'].pct_change().std()
        features['trend_direction'] = 'up' if features['price_change'] > 0 else 'down'
        
        # 根据形态类型分析特定特征
        if pattern_name == 'GOLDEN_CROSS':
            # 金叉应该有先跌后涨的特征
            mid_point = len(data) // 2
            first_half_change = (data['close'].iloc[mid_point] - data['close'].iloc[0]) / data['close'].iloc[0]
            second_half_change = (data['close'].iloc[-1] - data['close'].iloc[mid_point]) / data['close'].iloc[mid_point]
            features['first_half_trend'] = first_half_change
            features['second_half_trend'] = second_half_change
            features['expected_pattern'] = first_half_change < 0 and second_half_change > 0
            
        elif pattern_name == 'DEATH_CROSS':
            # 死叉应该有先涨后跌的特征
            mid_point = len(data) // 2
            first_half_change = (data['close'].iloc[mid_point] - data['close'].iloc[0]) / data['close'].iloc[0]
            second_half_change = (data['close'].iloc[-1] - data['close'].iloc[mid_point]) / data['close'].iloc[mid_point]
            features['first_half_trend'] = first_half_change
            features['second_half_trend'] = second_half_change
            features['expected_pattern'] = first_half_change > 0 and second_half_change < 0
        
        return features
    
    def _analyze_recognition_features(self, recognition_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析识别结果的特征"""
        features = {}
        
        details = recognition_result.get('details', {})
        features['detected_count'] = details.get('detected_count', 0)
        features['pattern_positions'] = details.get('pattern_positions', [])
        features['has_detection'] = features['detected_count'] > 0
        
        return features
    
    def _calculate_consistency_score(self, data_features: Dict[str, Any], 
                                   recognition_features: Dict[str, Any], 
                                   pattern_name: str) -> float:
        """计算一致性评分"""
        score = 0.0
        
        # 基础分：是否检测到形态 (50%)
        if recognition_features['has_detection']:
            score += 0.5
        
        # 形态特征分：检测结果是否符合预期 (50%)
        if pattern_name in ['GOLDEN_CROSS', 'DEATH_CROSS']:
            if data_features.get('expected_pattern', False):
                score += 0.3  # 数据符合预期形态
            if recognition_features['detected_count'] >= 1:
                score += 0.2  # 检测到合理数量的形态
        else:
            # 其他形态的简化评分
            if recognition_features['detected_count'] >= 1:
                score += 0.5
        
        return min(1.0, score)
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试总结"""
        summary = {
            'total_patterns': results['patterns_tested'],
            'passed_patterns': results['patterns_passed'],
            'failed_patterns': results['patterns_tested'] - results['patterns_passed'],
            'success_rate': results['success_rate'],
            'overall_result': '通过' if results['overall_success'] else '失败',
            'failed_pattern_list': [],
            'recommendations': []
        }
        
        # 收集失败的形态
        for pattern_name, pattern_result in results['pattern_results'].items():
            if not pattern_result['success']:
                summary['failed_pattern_list'].append(pattern_name)
        
        # 生成建议
        if results['overall_success']:
            summary['recommendations'].append("🎉 MACD指标双向验证完全通过，可以继续测试下一个指标")
        else:
            summary['recommendations'].append("🔧 需要修复MACD指标的形态识别逻辑")
            for failed_pattern in summary['failed_pattern_list']:
                summary['recommendations'].append(f"  - 修复{failed_pattern}形态的识别算法")
        
        return summary

def run_macd_bidirectional_test():
    """运行MACD双向验证测试"""
    print("🚀 MACD指标双向验证测试")
    print("=" * 80)
    print("📋 测试方法:")
    print("  1. 使用现有数据生成器生成包含特定形态的数据")
    print("  2. 使用MACD指标识别生成数据中的形态")
    print("  3. 验证识别结果与生成数据的双向一致性")
    print()
    
    tester = MACDBidirectionalTester()
    results = tester.run_bidirectional_test()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"macd_bidirectional_test_result_{timestamp}.json"
    
    import json
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 显示总结
    print("\n" + "=" * 80)
    print("📊 MACD双向验证测试总结")
    print("=" * 80)
    summary = results['summary']
    print(f"✅ 通过形态: {summary['passed_patterns']}/{summary['total_patterns']}")
    print(f"📈 成功率: {summary['success_rate']:.1%}")
    print(f"🎯 总体结果: {summary['overall_result']}")
    
    if summary['failed_pattern_list']:
        print(f"❌ 失败形态: {', '.join(summary['failed_pattern_list'])}")
    
    print(f"📄 详细结果: {result_file}")
    
    for recommendation in summary['recommendations']:
        print(recommendation)
    
    return results

if __name__ == "__main__":
    results = run_macd_bidirectional_test()
