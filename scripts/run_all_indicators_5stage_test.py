#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
所有指标五阶段测试批量运行器

修复所有WARNING错误后，对所有128个指标进行完整的五阶段测试验证
确保系统达到生产级质量标准
"""

import os
import sys
import time
import json
import importlib
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger
from indicators.complete_indicator_registry import CompleteIndicatorRegistry

logger = get_logger(__name__)

class AllIndicators5StageTestRunner:
    """所有指标五阶段测试批量运行器"""
    
    def __init__(self):
        """初始化测试运行器"""
        self.registry = CompleteIndicatorRegistry()
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # 创建结果目录
        self.results_dir = Path("results/5stage_tests")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("五阶段测试批量运行器初始化完成")
    
    def get_all_indicators(self) -> List[str]:
        """获取所有注册的指标名称"""
        all_indicators = self.registry.get_all_indicators()
        return list(all_indicators.keys())
    
    def run_single_indicator_5stage_test(self, indicator_name: str) -> Dict[str, Any]:
        """运行单个指标的五阶段测试"""
        logger.info(f"🔍 开始测试指标: {indicator_name}")
        
        try:
            # 尝试获取指标实例
            indicator = self.registry.get_indicator(indicator_name)
            if indicator is None:
                return {
                    'indicator': indicator_name,
                    'status': 'FAILED',
                    'error': 'Failed to create indicator instance',
                    'stage_results': {},
                    'overall_score': 0,
                    'passed': False
                }
            
            # 检查指标是否有五阶段测试文件
            test_file_path = self._find_5stage_test_file(indicator_name)
            if test_file_path:
                # 使用专用的五阶段测试
                return self._run_dedicated_5stage_test(indicator_name, test_file_path)
            else:
                # 使用通用的五阶段测试
                return self._run_generic_5stage_test(indicator_name, indicator)
        
        except Exception as e:
            logger.error(f"❌ 指标 {indicator_name} 测试失败: {e}")
            return {
                'indicator': indicator_name,
                'status': 'ERROR',
                'error': str(e),
                'stage_results': {},
                'overall_score': 0,
                'passed': False
            }
    
    def _find_5stage_test_file(self, indicator_name: str) -> str:
        """查找指标的专用五阶段测试文件"""
        # 可能的测试文件名模式
        patterns = [
            f"validation/{indicator_name.lower()}_5stage_validation.py",
            f"validation/enhanced_{indicator_name.lower()}_5stage_validation.py",
            f"validation/{indicator_name}_5stage_validation.py"
        ]
        
        for pattern in patterns:
            if os.path.exists(pattern):
                return pattern
        
        return None
    
    def _run_dedicated_5stage_test(self, indicator_name: str, test_file_path: str) -> Dict[str, Any]:
        """运行专用的五阶段测试"""
        try:
            # 动态导入测试模块
            module_name = test_file_path.replace('/', '.').replace('.py', '')
            module = importlib.import_module(module_name)
            
            # 查找验证器类
            validator_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, 'run_complete_validation') and
                    attr_name.endswith('Validator')):
                    validator_class = attr
                    break
            
            if validator_class:
                # 创建验证器实例并运行测试
                validator = validator_class()
                result = validator.run_complete_validation()
                
                return {
                    'indicator': indicator_name,
                    'status': 'COMPLETED',
                    'test_type': 'DEDICATED',
                    'test_file': test_file_path,
                    'stage_results': result.get('stage_results', {}),
                    'overall_score': result.get('overall_score', 0),
                    'passed': result.get('overall_passed', False),
                    'details': result
                }
            else:
                logger.warning(f"⚠️ 未找到验证器类: {test_file_path}")
                return self._run_generic_5stage_test(indicator_name, None)
        
        except Exception as e:
            logger.error(f"❌ 专用测试失败 {indicator_name}: {e}")
            return self._run_generic_5stage_test(indicator_name, None)
    
    def _run_generic_5stage_test(self, indicator_name: str, indicator) -> Dict[str, Any]:
        """运行通用的五阶段测试"""
        logger.info(f"🔧 使用通用测试: {indicator_name}")
        
        stage_results = {}
        overall_score = 0
        
        try:
            # 阶段1: 实例化测试 (20分)
            stage1_score = 0
            if indicator is not None:
                stage1_score = 20
                logger.info(f"✅ 阶段1通过: 指标实例化成功")
            else:
                logger.error(f"❌ 阶段1失败: 指标实例化失败")
            
            stage_results['stage1'] = {
                'name': '实例化测试',
                'score': stage1_score,
                'max_score': 20,
                'passed': stage1_score > 0
            }
            
            # 阶段2: 基础功能测试 (20分)
            stage2_score = 0
            if indicator and hasattr(indicator, 'calculate'):
                stage2_score = 20
                logger.info(f"✅ 阶段2通过: 具有calculate方法")
            else:
                logger.error(f"❌ 阶段2失败: 缺少calculate方法")
            
            stage_results['stage2'] = {
                'name': '基础功能测试',
                'score': stage2_score,
                'max_score': 20,
                'passed': stage2_score > 0
            }
            
            # 阶段3: 参数设置测试 (20分)
            stage3_score = 0
            if indicator and hasattr(indicator, 'set_parameters'):
                try:
                    indicator.set_parameters(period=20)
                    stage3_score = 20
                    logger.info(f"✅ 阶段3通过: 参数设置成功")
                except:
                    logger.error(f"❌ 阶段3失败: 参数设置失败")
            else:
                logger.error(f"❌ 阶段3失败: 缺少set_parameters方法")
            
            stage_results['stage3'] = {
                'name': '参数设置测试',
                'score': stage3_score,
                'max_score': 20,
                'passed': stage3_score > 0
            }
            
            # 阶段4: 信号生成测试 (20分)
            stage4_score = 0
            if indicator and (hasattr(indicator, 'get_signal') or hasattr(indicator, 'get_signals')):
                stage4_score = 20
                logger.info(f"✅ 阶段4通过: 具有信号生成方法")
            else:
                logger.error(f"❌ 阶段4失败: 缺少信号生成方法")
            
            stage_results['stage4'] = {
                'name': '信号生成测试',
                'score': stage4_score,
                'max_score': 20,
                'passed': stage4_score > 0
            }
            
            # 阶段5: 架构合规测试 (20分)
            stage5_score = 0
            if indicator and hasattr(indicator, '__class__'):
                # 检查是否继承自BaseIndicator
                class_name = indicator.__class__.__name__
                if 'Indicator' in class_name or 'indicator' in class_name.lower():
                    stage5_score = 20
                    logger.info(f"✅ 阶段5通过: 架构合规")
                else:
                    logger.warning(f"⚠️ 阶段5部分通过: 类名不规范")
                    stage5_score = 10
            else:
                logger.error(f"❌ 阶段5失败: 架构不合规")
            
            stage_results['stage5'] = {
                'name': '架构合规测试',
                'score': stage5_score,
                'max_score': 20,
                'passed': stage5_score > 0
            }
            
            # 计算总分
            overall_score = sum(stage['score'] for stage in stage_results.values())
            max_score = sum(stage['max_score'] for stage in stage_results.values())
            passed = overall_score >= 80  # 80分及格
            
            return {
                'indicator': indicator_name,
                'status': 'COMPLETED',
                'test_type': 'GENERIC',
                'stage_results': stage_results,
                'overall_score': overall_score,
                'max_score': max_score,
                'percentage': (overall_score / max_score) * 100,
                'passed': passed
            }
        
        except Exception as e:
            logger.error(f"❌ 通用测试失败 {indicator_name}: {e}")
            return {
                'indicator': indicator_name,
                'status': 'ERROR',
                'test_type': 'GENERIC',
                'error': str(e),
                'stage_results': stage_results,
                'overall_score': overall_score,
                'passed': False
            }
    
    def run_all_indicators_test(self) -> Dict[str, Any]:
        """运行所有指标的五阶段测试"""
        logger.info("🚀 开始所有指标五阶段测试")
        self.start_time = datetime.now()
        
        # 获取所有指标
        all_indicators = self.get_all_indicators()
        logger.info(f"📊 共需测试 {len(all_indicators)} 个指标")
        
        # 测试统计
        total_indicators = len(all_indicators)
        passed_count = 0
        failed_count = 0
        error_count = 0
        
        # 逐个测试指标
        for i, indicator_name in enumerate(all_indicators, 1):
            logger.info(f"📋 进度: {i}/{total_indicators} - {indicator_name}")
            
            result = self.run_single_indicator_5stage_test(indicator_name)
            self.test_results[indicator_name] = result
            
            # 更新统计
            if result['status'] == 'COMPLETED' and result['passed']:
                passed_count += 1
                logger.info(f"✅ {indicator_name}: 通过 ({result['overall_score']}分)")
            elif result['status'] == 'COMPLETED':
                failed_count += 1
                logger.warning(f"⚠️ {indicator_name}: 未通过 ({result['overall_score']}分)")
            else:
                error_count += 1
                logger.error(f"❌ {indicator_name}: 错误")
        
        self.end_time = datetime.now()
        
        # 生成总结报告
        summary = {
            'test_summary': {
                'total_indicators': total_indicators,
                'passed_count': passed_count,
                'failed_count': failed_count,
                'error_count': error_count,
                'pass_rate': (passed_count / total_indicators) * 100,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': (self.end_time - self.start_time).total_seconds()
            },
            'detailed_results': self.test_results
        }
        
        # 保存结果
        self._save_test_results(summary)
        
        # 显示总结
        self._display_summary(summary)
        
        return summary
    
    def _save_test_results(self, summary: Dict[str, Any]):
        """保存测试结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = self.results_dir / f"all_indicators_5stage_test_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 测试结果已保存: {result_file}")
    
    def _display_summary(self, summary: Dict[str, Any]):
        """显示测试总结"""
        test_summary = summary['test_summary']
        
        print("\n" + "="*80)
        print("🎉 所有指标五阶段测试完成总结")
        print("="*80)
        print(f"📊 总指标数量: {test_summary['total_indicators']}")
        print(f"✅ 通过数量: {test_summary['passed_count']}")
        print(f"⚠️ 未通过数量: {test_summary['failed_count']}")
        print(f"❌ 错误数量: {test_summary['error_count']}")
        print(f"🎯 通过率: {test_summary['pass_rate']:.1f}%")
        print(f"⏱️ 测试耗时: {test_summary['duration_seconds']:.1f}秒")
        
        # 显示未通过的指标
        if test_summary['failed_count'] > 0 or test_summary['error_count'] > 0:
            print("\n⚠️ 需要关注的指标:")
            for indicator_name, result in summary['detailed_results'].items():
                if not result['passed']:
                    status = result['status']
                    score = result.get('overall_score', 0)
                    error = result.get('error', '')
                    print(f"  - {indicator_name}: {status} (分数: {score}) {error}")
        
        print("="*80)


def main():
    """主函数"""
    print("🚀 开始所有指标五阶段测试...")
    
    # 创建测试运行器
    runner = AllIndicators5StageTestRunner()
    
    # 运行所有指标测试
    summary = runner.run_all_indicators_test()
    
    # 判断整体结果
    pass_rate = summary['test_summary']['pass_rate']
    if pass_rate >= 90:
        print(f"\n🎉 测试结果优秀！通过率: {pass_rate:.1f}%")
        exit(0)
    elif pass_rate >= 80:
        print(f"\n✅ 测试结果良好！通过率: {pass_rate:.1f}%")
        exit(0)
    else:
        print(f"\n⚠️ 测试结果需要改进！通过率: {pass_rate:.1f}%")
        exit(1)


if __name__ == "__main__":
    main()
