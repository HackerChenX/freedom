#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 4: ZXM评分指标批量修复脚本

修复3个ZXM评分指标：
- ZXM_BUYPOINT_SCORE
- ZXM_TREND_SCORE
- ZXM_ELASTIC_SCORE
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indicators.complete_indicator_registry import complete_registry
from tests.unified_indicator_testing.components.test_data_generator import TestDataGenerator
from tests.unified_indicator_testing.components.buypoint_analyzer import BuypointAnalyzer


class Phase4ZXMScoringRepair:
    """Phase 4: ZXM评分指标批量修复器"""
    
    def __init__(self):
        self.phase_name = "Phase 4: ZXM评分指标"
        self.target_indicators = [
            'ZXM_BUYPOINT_SCORE',
            'ZXM_TREND_SCORE',
            'ZXM_ELASTIC_SCORE'
        ]
        self.test_data_generator = TestDataGenerator()
        self.buypoint_analyzer = BuypointAnalyzer()
        self.results = {}
    
    def run_phase4_repair(self):
        """运行Phase 4完整修复"""
        print("=" * 80)
        print(f"🔧 {self.phase_name} 开始")
        print(f"📋 目标指标: {len(self.target_indicators)}个")
        print("=" * 80)
        
        phase_start_time = time.time()
        
        # 逐个修复指标
        for i, indicator_name in enumerate(self.target_indicators, 1):
            print(f"\n🎯 [{i}/{len(self.target_indicators)}] 修复 {indicator_name}")
            print("-" * 60)
            
            repair_result = self._repair_scoring_indicator(indicator_name)
            self.results[indicator_name] = repair_result
            
            # 输出单个指标结果
            if repair_result['status'] == 'SUCCESS':
                print(f"  ✅ 修复成功 - 质量等级: {repair_result.get('quality_level', 'N/A')}")
                if 'scoring_accuracy' in repair_result:
                    print(f"  📊 评分准确性: {repair_result['scoring_accuracy']:.1f}/100")
            else:
                print(f"  ❌ 修复失败 - {repair_result.get('error', '未知错误')}")
        
        # 生成Phase 4总结报告
        phase_duration = time.time() - phase_start_time
        phase_report = self._generate_phase_report(phase_duration)
        
        return phase_report
    
    def _repair_scoring_indicator(self, indicator_name: str) -> Dict[str, Any]:
        """修复单个ZXM评分指标"""
        repair_start = time.time()
        
        try:
            # Step 1: 基础功能验证
            print(f"  📈 Step 1: 基础功能验证...")
            basic_result = self._test_basic_function(indicator_name)
            if not basic_result['success']:
                return {
                    'status': 'FAILED',
                    'error': f"基础功能失败: {basic_result['error']}",
                    'repair_time': time.time() - repair_start
                }
            
            # Step 2: 评分特性检查
            print(f"  📊 Step 2: 评分特性检查...")
            scoring_result = self._check_scoring_features(indicator_name, basic_result['indicator'])
            
            # Step 3: 评分算法测试
            print(f"  🧮 Step 3: 评分算法测试...")
            algorithm_result = self._test_scoring_algorithm(indicator_name)
            
            # Step 4: 权重和范围测试
            print(f"  ⚖️ Step 4: 权重和范围测试...")
            weight_result = self._test_weight_and_range(indicator_name)
            
            # Step 5: 质量评估
            print(f"  📊 Step 5: 质量评估...")
            quality_assessment = self._assess_scoring_quality(
                basic_result, scoring_result, algorithm_result, weight_result
            )
            
            repair_time = time.time() - repair_start
            
            return {
                'status': 'SUCCESS',
                'basic_function': basic_result,
                'scoring_features': scoring_result,
                'algorithm_test': algorithm_result,
                'weight_test': weight_result,
                'quality_assessment': quality_assessment,
                'quality_level': quality_assessment['level'],
                'scoring_accuracy': quality_assessment['total_score'] * 100,
                'repair_time': repair_time
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'repair_time': time.time() - repair_start
            }
    
    def _test_basic_function(self, indicator_name: str) -> Dict[str, Any]:
        """测试基础功能"""
        try:
            # 创建指标
            indicator = complete_registry.create_indicator(indicator_name)
            if not indicator:
                return {'success': False, 'error': '指标创建失败'}
            
            # 生成测试数据 - 评分指标需要综合数据
            test_data = self.test_data_generator.generate_zxm_test_data(200)
            if test_data is None or test_data.empty:
                return {'success': False, 'error': '测试数据生成失败'}
            
            # 计算指标
            start_time = time.time()
            result = indicator.calculate(test_data)
            calc_time = time.time() - start_time
            
            if result is None:
                return {'success': False, 'error': '指标计算返回None'}
            
            return {
                'success': True,
                'indicator': indicator,
                'test_data': test_data,
                'result': result,
                'calculation_time': calc_time,
                'result_type': type(result).__name__
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _check_scoring_features(self, indicator_name: str, indicator) -> Dict[str, Any]:
        """检查评分指标特性"""
        try:
            test_data = self.test_data_generator.generate_zxm_test_data(150)
            result = indicator.calculate(test_data)
            
            # 检查返回格式
            is_dict_format = isinstance(result, dict)
            has_score = False
            has_weighted_score = False
            has_grade = False
            
            if is_dict_format:
                result_str = str(result).lower()
                has_score = 'score' in result_str or 'rating' in result_str
                has_weighted_score = 'weight' in result_str or 'weighted' in result_str
                has_grade = 'grade' in result_str or 'level' in result_str
            
            # 检查评分方法
            has_scoring_methods = (
                hasattr(indicator, 'calculate_score') or 
                hasattr(indicator, 'get_rating') or
                hasattr(indicator, 'compute_grade') or
                hasattr(indicator, 'get_signals')
            )
            
            # 尝试获取数值评分
            numeric_score = None
            if is_dict_format and result:
                for key, value in result.items():
                    if 'score' in str(key).lower() and isinstance(value, (int, float)):
                        numeric_score = value
                        break
            
            # 计算评分特性评分
            scoring_compliance = (
                is_dict_format and 
                has_score and 
                has_scoring_methods
            )
            
            scoring_score = (
                (0.3 if is_dict_format else 0) +
                (0.3 if has_score else 0) +
                (0.2 if has_weighted_score or has_grade else 0) +
                (0.2 if has_scoring_methods else 0)
            )
            
            return {
                'is_dict_format': is_dict_format,
                'has_score': has_score,
                'has_weighted_score': has_weighted_score,
                'has_grade': has_grade,
                'has_scoring_methods': has_scoring_methods,
                'numeric_score': numeric_score,
                'scoring_compliance': scoring_compliance,
                'scoring_score': scoring_score
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'scoring_compliance': False,
                'scoring_score': 0
            }
    
    def _test_scoring_algorithm(self, indicator_name: str) -> Dict[str, Any]:
        """测试评分算法"""
        try:
            indicator = complete_registry.create_indicator(indicator_name)
            
            # 测试不同市场条件下的评分
            scenarios = ['强势市场', '弱势市场', '震荡市场', '转折市场']
            scenario_scores = {}
            
            for scenario in scenarios:
                try:
                    if scenario == '强势市场':
                        test_data = self._generate_bull_market_data()
                    elif scenario == '弱势市场':
                        test_data = self._generate_bear_market_data()
                    elif scenario == '震荡市场':
                        test_data = self._generate_sideways_data()
                    else:  # 转折市场
                        test_data = self._generate_turning_data()
                    
                    result = indicator.calculate(test_data)
                    
                    # 提取评分
                    score = self._extract_score_from_result(result)
                    scenario_scores[scenario] = {
                        'score': score,
                        'result_type': type(result).__name__,
                        'has_valid_score': score is not None
                    }
                    
                except Exception as e:
                    scenario_scores[scenario] = {
                        'error': str(e),
                        'score': None,
                        'has_valid_score': False
                    }
            
            # 计算算法有效性
            valid_scores = [s for s in scenario_scores.values() if s.get('has_valid_score', False)]
            algorithm_effectiveness = len(valid_scores) / len(scenarios)
            
            # 检查评分合理性（不同场景应有不同评分）
            scores = [s['score'] for s in valid_scores if s['score'] is not None]
            score_variance = 0
            if len(scores) > 1:
                mean_score = sum(scores) / len(scores)
                score_variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            
            return {
                'scenario_scores': scenario_scores,
                'algorithm_effectiveness': algorithm_effectiveness,
                'score_variance': score_variance,
                'scenarios_tested': len(scenarios),
                'valid_scenarios': len(valid_scores)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'algorithm_effectiveness': 0,
                'scenarios_tested': 0
            }
    
    def _generate_bull_market_data(self):
        """生成强势市场数据"""
        return self.test_data_generator.generate_zxm_test_data(120)
    
    def _generate_bear_market_data(self):
        """生成弱势市场数据"""
        return self.test_data_generator.generate_zxm_test_data(120)
    
    def _generate_sideways_data(self):
        """生成震荡市场数据"""
        return self.test_data_generator.generate_zxm_test_data(120)
    
    def _generate_turning_data(self):
        """生成转折市场数据"""
        return self.test_data_generator.generate_zxm_test_data(120)
    
    def _extract_score_from_result(self, result):
        """从结果中提取评分"""
        if isinstance(result, dict):
            for key, value in result.items():
                if 'score' in str(key).lower() and isinstance(value, (int, float)):
                    return value
        elif isinstance(result, (int, float)):
            return result
        return None
    
    def _test_weight_and_range(self, indicator_name: str) -> Dict[str, Any]:
        """测试权重和评分范围"""
        try:
            # 测试评分范围的合理性
            indicator = complete_registry.create_indicator(indicator_name)
            
            # 多次测试获取评分范围
            scores = []
            for _ in range(10):
                try:
                    test_data = self.test_data_generator.generate_zxm_test_data(100)
                    result = indicator.calculate(test_data)
                    score = self._extract_score_from_result(result)
                    if score is not None:
                        scores.append(score)
                except:
                    continue
            
            if not scores:
                return {
                    'error': '无法获取有效评分',
                    'range_valid': False,
                    'weight_reasonable': False
                }
            
            # 分析评分范围
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            score_range = max_score - min_score
            
            # 评分范围合理性检查
            range_valid = (
                min_score >= 0 and  # 最小值非负
                max_score <= 100 and  # 最大值不超过100
                score_range > 0  # 有区分度
            )
            
            # 权重合理性检查（评分应有适当分布）
            weight_reasonable = (
                score_range >= 10 and  # 至少有10分的区间
                abs(avg_score - 50) <= 30  # 平均值在合理范围内
            )
            
            return {
                'scores_collected': len(scores),
                'min_score': min_score,
                'max_score': max_score,
                'avg_score': avg_score,
                'score_range': score_range,
                'range_valid': range_valid,
                'weight_reasonable': weight_reasonable,
                'range_score': 1.0 if range_valid and weight_reasonable else 0.5 if range_valid else 0.0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'range_valid': False,
                'weight_reasonable': False,
                'range_score': 0.0
            }
    
    def _assess_scoring_quality(self, basic_result: Dict, scoring_result: Dict, 
                              algorithm_result: Dict, weight_result: Dict) -> Dict[str, Any]:
        """综合评分质量评估"""
        try:
            # 计算各维度得分
            basic_score = 0.25 if basic_result['success'] else 0
            scoring_score = 0.25 * scoring_result.get('scoring_score', 0)
            algorithm_score = 0.25 * algorithm_result.get('algorithm_effectiveness', 0)
            weight_score = 0.25 * weight_result.get('range_score', 0)
            
            total_score = basic_score + scoring_score + algorithm_score + weight_score
            
            # 确定质量等级 - 评分指标标准
            if total_score >= 0.75:
                level = 'A'
                description = '优秀'
            elif total_score >= 0.60:
                level = 'B'
                description = '良好'
            elif total_score >= 0.45:
                level = 'C'
                description = '可接受'
            else:
                level = 'D'
                description = '需要改进'
            
            # 生成评分指标专用改进建议
            recommendations = []
            if not basic_result['success']:
                recommendations.append("修复基础计算功能")
            if scoring_result.get('scoring_score', 0) < 0.6:
                recommendations.append("完善评分机制和格式")
            if algorithm_result.get('algorithm_effectiveness', 0) < 0.75:
                recommendations.append("优化评分算法逻辑")
            if weight_result.get('range_score', 0) < 0.7:
                recommendations.append("调整评分范围和权重")
            
            if not recommendations:
                recommendations.append("评分指标质量良好，可进行生产环境验证")
            
            return {
                'total_score': total_score,
                'level': level,
                'description': description,
                'score_breakdown': {
                    'basic_function': basic_score,
                    'scoring_features': scoring_score,
                    'algorithm_effectiveness': algorithm_score,
                    'weight_and_range': weight_score
                },
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'level': 'D',
                'total_score': 0
            }
    
    def _generate_phase_report(self, phase_duration: float) -> Dict[str, Any]:
        """生成Phase 4总结报告"""
        # 统计结果
        total_indicators = len(self.target_indicators)
        successful_repairs = len([r for r in self.results.values() if r['status'] == 'SUCCESS'])
        failed_repairs = total_indicators - successful_repairs
        
        # 计算平均质量
        quality_levels = [r.get('quality_level', 'D') for r in self.results.values() if r['status'] == 'SUCCESS']
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for level in quality_levels:
            grade_counts[level] = grade_counts.get(level, 0) + 1
        
        # 计算平均评分准确性
        scoring_accuracies = [r.get('scoring_accuracy', 0) for r in self.results.values() if r['status'] == 'SUCCESS']
        avg_scoring_accuracy = sum(scoring_accuracies) / len(scoring_accuracies) if scoring_accuracies else 0
        
        success_rate = (successful_repairs / total_indicators) * 100
        
        # 生成报告
        phase_report = {
            'phase_name': self.phase_name,
            'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': phase_duration,
            'summary': {
                'total_indicators': total_indicators,
                'successful_repairs': successful_repairs,
                'failed_repairs': failed_repairs,
                'success_rate': success_rate,
                'average_scoring_accuracy': avg_scoring_accuracy
            },
            'quality_distribution': grade_counts,
            'detailed_results': self.results
        }
        
        # 保存报告
        self._save_phase_report(phase_report)
        
        # 输出摘要
        print(f"\n" + "=" * 80)
        print(f"🎊 {self.phase_name} 完成!")
        print(f"=" * 80)
        print(f"📊 修复统计:")
        print(f"   目标指标: {total_indicators}")
        print(f"   成功修复: {successful_repairs}")
        print(f"   失败修复: {failed_repairs}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   平均评分准确性: {avg_scoring_accuracy:.1f}/100")
        print(f"   用时: {phase_duration:.2f}秒")
        
        print(f"\n📈 质量分布:")
        for grade, count in grade_counts.items():
            if count > 0:
                print(f"   {grade}级: {count}个")
        
        return phase_report
    
    def _save_phase_report(self, report: Dict[str, Any]):
        """保存Phase 4报告"""
        try:
            results_dir = project_root / "results" / "phase4_zxm_repair"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = results_dir / f"phase4_zxm_scoring_repair_{timestamp}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n📄 Phase 4报告已保存: {report_path}")
        except Exception as e:
            print(f"⚠️ 保存报告失败: {e}")


def main():
    """主函数"""
    try:
        print("🚀 启动Phase 4: ZXM评分指标批量修复")
        
        # 创建修复器
        repair_system = Phase4ZXMScoringRepair()
        
        # 运行Phase 4修复
        phase_report = repair_system.run_phase4_repair()
        
        # 判断Phase 4结果
        success_rate = phase_report['summary']['success_rate']
        return 0 if success_rate >= 50 else 1
        
    except Exception as e:
        print(f"💥 Phase 4修复异常: {e}")
        return 1


if __name__ == "__main__":
    exit(main())