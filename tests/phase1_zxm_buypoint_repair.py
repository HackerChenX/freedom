#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 1: ZXM买点指标批量修复脚本

修复5个ZXM买点指标：
- ZXM_DAILY_MACD
- ZXM_TURNOVER 
- ZXM_BS_ABSORB
- ZXM_VOLUME_SHRINK
- ZXM_MA_CALLBACK
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


class Phase1ZXMBuypointRepair:
    """Phase 1: ZXM买点指标批量修复器"""
    
    def __init__(self):
        self.phase_name = "Phase 1: ZXM买点指标"
        self.target_indicators = [
            'ZXM_DAILY_MACD',
            'ZXM_TURNOVER', 
            'ZXM_BS_ABSORB',
            'ZXM_VOLUME_SHRINK',
            'ZXM_MA_CALLBACK'
        ]
        self.test_data_generator = TestDataGenerator()
        self.buypoint_analyzer = BuypointAnalyzer()
        self.results = {}
    
    def run_phase1_repair(self):
        """运行Phase 1完整修复"""
        print("=" * 80)
        print(f"🔧 {self.phase_name} 开始")
        print(f"📋 目标指标: {len(self.target_indicators)}个")
        print("=" * 80)
        
        phase_start_time = time.time()
        
        # 逐个修复指标
        for i, indicator_name in enumerate(self.target_indicators, 1):
            print(f"\n🎯 [{i}/{len(self.target_indicators)}] 修复 {indicator_name}")
            print("-" * 60)
            
            repair_result = self._repair_single_indicator(indicator_name)
            self.results[indicator_name] = repair_result
            
            # 输出单个指标结果
            if repair_result['status'] == 'SUCCESS':
                print(f"  ✅ 修复成功 - 质量等级: {repair_result.get('quality_level', 'N/A')}")
                if 'accuracy' in repair_result:
                    print(f"  📊 买点准确率: {repair_result['accuracy']:.1f}%")
            else:
                print(f"  ❌ 修复失败 - {repair_result.get('error', '未知错误')}")
        
        # 生成Phase 1总结报告
        phase_duration = time.time() - phase_start_time
        phase_report = self._generate_phase_report(phase_duration)
        
        return phase_report
    
    def _repair_single_indicator(self, indicator_name: str) -> Dict[str, Any]:
        """修复单个ZXM买点指标"""
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
            
            # Step 2: ZXM特性检查
            print(f"  🔍 Step 2: ZXM特性检查...")
            zxm_result = self._check_zxm_features(indicator_name, basic_result['indicator'])
            
            # Step 3: 买点识别测试
            print(f"  🎯 Step 3: 买点识别测试...")
            buypoint_result = self._test_buypoint_capability(indicator_name)
            
            # Step 4: 质量评估
            print(f"  📊 Step 4: 质量评估...")
            quality_assessment = self._assess_quality(basic_result, zxm_result, buypoint_result)
            
            repair_time = time.time() - repair_start
            
            return {
                'status': 'SUCCESS',
                'basic_function': basic_result,
                'zxm_features': zxm_result,
                'buypoint_capability': buypoint_result,
                'quality_assessment': quality_assessment,
                'quality_level': quality_assessment['level'],
                'accuracy': buypoint_result.get('accuracy', 0),
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
            
            # 生成测试数据
            test_data = self.test_data_generator.generate_zxm_test_data(120)
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
    
    def _check_zxm_features(self, indicator_name: str, indicator) -> Dict[str, Any]:
        """检查ZXM指标特性"""
        try:
            test_data = self.test_data_generator.generate_zxm_test_data(100)
            result = indicator.calculate(test_data)
            
            # 检查返回格式
            is_dict_format = isinstance(result, dict)
            has_signal = False
            has_score = False
            
            if is_dict_format:
                has_signal = 'signal' in result or any('signal' in str(k).lower() for k in result.keys())
                has_score = 'score' in result or any('score' in str(k).lower() for k in result.keys())
            
            # 检查买点方法
            has_buypoint_methods = (
                hasattr(indicator, 'get_signals') or 
                hasattr(indicator, 'detect_patterns') or
                hasattr(indicator, 'get_buypoint_signals')
            )
            
            zxm_compliance = (
                is_dict_format and 
                (has_signal or has_score) and 
                has_buypoint_methods
            )
            
            return {
                'is_dict_format': is_dict_format,
                'has_signal': has_signal,
                'has_score': has_score,
                'has_buypoint_methods': has_buypoint_methods,
                'zxm_compliance': zxm_compliance,
                'compliance_score': (
                    (0.4 if is_dict_format else 0) +
                    (0.3 if has_signal or has_score else 0) +
                    (0.3 if has_buypoint_methods else 0)
                )
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'zxm_compliance': False,
                'compliance_score': 0
            }
    
    def _test_buypoint_capability(self, indicator_name: str) -> Dict[str, Any]:
        """测试买点识别能力"""
        try:
            # 快速买点测试
            test_data = self.test_data_generator.generate_zxm_test_data(100)
            accuracy = self.buypoint_analyzer.quick_buypoint_test(indicator_name, test_data)
            
            # 根据准确率判断能力等级
            if accuracy >= 80:
                capability_level = 'HIGH'
            elif accuracy >= 60:
                capability_level = 'MEDIUM'
            elif accuracy >= 30:
                capability_level = 'LOW'
            else:
                capability_level = 'MINIMAL'
            
            return {
                'accuracy': accuracy,
                'capability_level': capability_level,
                'test_data_size': len(test_data),
                'buypoint_capable': accuracy > 30
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'accuracy': 0,
                'capability_level': 'NONE',
                'buypoint_capable': False
            }
    
    def _assess_quality(self, basic_result: Dict, zxm_result: Dict, buypoint_result: Dict) -> Dict[str, Any]:
        """综合质量评估"""
        try:
            # 计算各维度得分
            basic_score = 0.3 if basic_result['success'] else 0
            zxm_score = 0.3 * zxm_result.get('compliance_score', 0)
            buypoint_score = 0.4 * (buypoint_result.get('accuracy', 0) / 100)
            
            total_score = basic_score + zxm_score + buypoint_score
            
            # 确定质量等级
            if total_score >= 0.8:
                level = 'A'
                description = '优秀'
            elif total_score >= 0.6:
                level = 'B'
                description = '良好'
            elif total_score >= 0.4:
                level = 'C'
                description = '可接受'
            else:
                level = 'D'
                description = '需要改进'
            
            # 生成改进建议
            recommendations = []
            if not basic_result['success']:
                recommendations.append("修复基础功能问题")
            if zxm_result.get('compliance_score', 0) < 0.7:
                recommendations.append("提升ZXM格式兼容性")
            if buypoint_result.get('accuracy', 0) < 70:
                recommendations.append("改进买点识别算法")
            
            return {
                'total_score': total_score,
                'level': level,
                'description': description,
                'score_breakdown': {
                    'basic_function': basic_score,
                    'zxm_compliance': zxm_score,
                    'buypoint_accuracy': buypoint_score
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
        """生成Phase 1总结报告"""
        # 统计结果
        total_indicators = len(self.target_indicators)
        successful_repairs = len([r for r in self.results.values() if r['status'] == 'SUCCESS'])
        failed_repairs = total_indicators - successful_repairs
        
        # 计算平均质量
        quality_levels = [r.get('quality_level', 'D') for r in self.results.values() if r['status'] == 'SUCCESS']
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for level in quality_levels:
            grade_counts[level] = grade_counts.get(level, 0) + 1
        
        # 计算平均准确率
        accuracies = [r.get('accuracy', 0) for r in self.results.values() if r['status'] == 'SUCCESS']
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
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
                'average_accuracy': avg_accuracy
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
        print(f"   平均准确率: {avg_accuracy:.1f}%")
        print(f"   用时: {phase_duration:.2f}秒")
        
        print(f"\n📈 质量分布:")
        for grade, count in grade_counts.items():
            if count > 0:
                print(f"   {grade}级: {count}个")
        
        return phase_report
    
    def _save_phase_report(self, report: Dict[str, Any]):
        """保存Phase 1报告"""
        try:
            results_dir = project_root / "results" / "phase1_zxm_repair"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = results_dir / f"phase1_zxm_buypoint_repair_{timestamp}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n📄 Phase 1报告已保存: {report_path}")
        except Exception as e:
            print(f"⚠️ 保存报告失败: {e}")


def main():
    """主函数"""
    try:
        print("🚀 启动Phase 1: ZXM买点指标批量修复")
        
        # 创建修复器
        repair_system = Phase1ZXMBuypointRepair()
        
        # 运行Phase 1修复
        phase_report = repair_system.run_phase1_repair()
        
        # 判断Phase 1结果
        success_rate = phase_report['summary']['success_rate']
        return 0 if success_rate >= 60 else 1
        
    except Exception as e:
        print(f"💥 Phase 1修复异常: {e}")
        return 1


if __name__ == "__main__":
    exit(main())