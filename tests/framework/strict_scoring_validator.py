#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格评分验证框架

确保所有指标验证都达到99分以上标准，严格禁止降低标准
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StageRequirement:
    """阶段要求"""
    name: str
    min_score: float
    requires_real_data: bool
    real_data_percentage: float
    description: str


class StrictScoringValidator:
    """严格评分验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.stage_requirements = {
            'stage1': StageRequirement(
                name='算法差异预分析',
                min_score=99.0,
                requires_real_data=False,
                real_data_percentage=0.0,
                description='使用标准测试用例验证算法准确性'
            ),
            'stage2': StageRequirement(
                name='基础功能验证',
                min_score=99.0,
                requires_real_data=False,
                real_data_percentage=0.0,
                description='验证参数管理、错误处理等基础功能'
            ),
            'stage3': StageRequirement(
                name='形态识别验证',
                min_score=99.0,
                requires_real_data=True,
                real_data_percentage=50.0,
                description='至少50%使用真实数据验证形态识别能力'
            ),
            'stage4': StageRequirement(
                name='架构合规性验证',
                min_score=99.0,
                requires_real_data=False,
                real_data_percentage=0.0,
                description='验证分层架构、依赖注入等架构合规性'
            ),
            'stage5': StageRequirement(
                name='生产就绪性验证',
                min_score=99.0,
                requires_real_data=True,
                real_data_percentage=100.0,
                description='100%使用真实数据验证生产就绪性'
            )
        }
        
        self.final_requirements = {
            'min_stage_score': 99.0,
            'min_average_score': 99.5,
            'max_failed_stages': 0,
            'requires_all_stages_pass': True
        }
        
        logger.info("✅ 严格评分验证器初始化完成")
        logger.info(f"🎯 所有阶段最低要求: {self.final_requirements['min_stage_score']}分")
        logger.info(f"🎯 平均评分最低要求: {self.final_requirements['min_average_score']}分")
    
    def validate_stage_score(self, stage_key: str, score: float, 
                           uses_real_data: bool = False, 
                           real_data_percentage: float = 0.0) -> bool:
        """验证阶段评分"""
        if stage_key not in self.stage_requirements:
            raise ValidationError(f"未知阶段: {stage_key}")
        
        requirement = self.stage_requirements[stage_key]
        
        # 1. 检查评分是否达标
        if score < requirement.min_score:
            raise ValidationError(
                f"❌ {requirement.name}评分不达标: "
                f"{score:.1f}分 < {requirement.min_score}分"
            )
        
        # 2. 检查真实数据使用要求
        if requirement.requires_real_data:
            if not uses_real_data:
                raise ValidationError(
                    f"❌ {requirement.name}必须使用真实数据，"
                    f"要求真实数据占比: {requirement.real_data_percentage}%"
                )
            
            if real_data_percentage < requirement.real_data_percentage:
                raise ValidationError(
                    f"❌ {requirement.name}真实数据占比不足: "
                    f"{real_data_percentage}% < {requirement.real_data_percentage}%"
                )
        
        logger.info(f"✅ {requirement.name}验证通过: {score:.1f}分")
        return True
    
    def validate_final_status(self, indicator_name: str, 
                            stage_results: Dict[str, Dict]) -> bool:
        """验证最终PASSED状态"""
        logger.info(f"🔍 验证{indicator_name}最终PASSED状态...")
        
        scores = []
        failed_stages = []
        
        # 1. 验证每个阶段
        for stage_key, requirement in self.stage_requirements.items():
            if stage_key not in stage_results:
                failed_stages.append(f"{requirement.name}(缺失)")
                continue
            
            stage_result = stage_results[stage_key]
            score = stage_result.get('overall_score', 0)
            scores.append(score)
            
            try:
                uses_real_data = stage_result.get('uses_real_data', False)
                real_data_percentage = stage_result.get('real_data_percentage', 0.0)
                
                self.validate_stage_score(
                    stage_key, score, uses_real_data, real_data_percentage
                )
            except ValidationError as e:
                failed_stages.append(str(e))
        
        # 2. 检查失败阶段数量
        if len(failed_stages) > self.final_requirements['max_failed_stages']:
            raise ValidationError(
                f"❌ {indicator_name}有{len(failed_stages)}个阶段未通过验证: "
                f"{failed_stages}"
            )
        
        # 3. 检查平均评分
        if scores:
            average_score = sum(scores) / len(scores)
            if average_score < self.final_requirements['min_average_score']:
                raise ValidationError(
                    f"❌ {indicator_name}平均评分不达标: "
                    f"{average_score:.1f}分 < {self.final_requirements['min_average_score']}分"
                )
        else:
            raise ValidationError(f"❌ {indicator_name}没有有效的阶段评分")
        
        # 4. 生成验证报告
        validation_report = self._generate_validation_report(
            indicator_name, stage_results, scores, average_score
        )
        
        logger.info(f"✅ {indicator_name}通过严格评分验证")
        logger.info(f"📊 平均评分: {average_score:.1f}分")
        
        return True
    
    def _generate_validation_report(self, indicator_name: str, 
                                  stage_results: Dict, scores: List[float], 
                                  average_score: float) -> Dict[str, Any]:
        """生成验证报告"""
        report = {
            'indicator_name': indicator_name,
            'validation_time': datetime.now().isoformat(),
            'validation_passed': True,
            'average_score': average_score,
            'stage_scores': scores,
            'stage_details': {},
            'requirements_met': {
                'min_stage_score': all(s >= self.final_requirements['min_stage_score'] for s in scores),
                'min_average_score': average_score >= self.final_requirements['min_average_score'],
                'all_stages_pass': len(stage_results) == len(self.stage_requirements)
            }
        }
        
        for stage_key, requirement in self.stage_requirements.items():
            if stage_key in stage_results:
                stage_result = stage_results[stage_key]
                report['stage_details'][stage_key] = {
                    'name': requirement.name,
                    'score': stage_result.get('overall_score', 0),
                    'required_score': requirement.min_score,
                    'uses_real_data': stage_result.get('uses_real_data', False),
                    'real_data_required': requirement.requires_real_data,
                    'real_data_percentage': stage_result.get('real_data_percentage', 0.0),
                    'required_real_data_percentage': requirement.real_data_percentage,
                    'passed': stage_result.get('overall_score', 0) >= requirement.min_score
                }
        
        return report
    
    def create_stage_validator(self, stage_key: str):
        """创建阶段验证器"""
        if stage_key not in self.stage_requirements:
            raise ValueError(f"未知阶段: {stage_key}")
        
        requirement = self.stage_requirements[stage_key]
        
        class StageValidator:
            def __init__(self, requirement: StageRequirement):
                self.requirement = requirement
            
            def validate_result(self, result: Dict[str, Any]) -> bool:
                """验证阶段结果"""
                score = result.get('overall_score', 0)
                uses_real_data = result.get('uses_real_data', False)
                real_data_percentage = result.get('real_data_percentage', 0.0)
                
                validator = StrictScoringValidator()
                return validator.validate_stage_score(
                    stage_key, score, uses_real_data, real_data_percentage
                )
            
            def get_requirements(self) -> Dict[str, Any]:
                """获取阶段要求"""
                return {
                    'name': self.requirement.name,
                    'min_score': self.requirement.min_score,
                    'requires_real_data': self.requirement.requires_real_data,
                    'real_data_percentage': self.requirement.real_data_percentage,
                    'description': self.requirement.description
                }
        
        return StageValidator(requirement)
    
    def get_all_requirements(self) -> Dict[str, Any]:
        """获取所有验证要求"""
        return {
            'stage_requirements': {
                k: {
                    'name': v.name,
                    'min_score': v.min_score,
                    'requires_real_data': v.requires_real_data,
                    'real_data_percentage': v.real_data_percentage,
                    'description': v.description
                }
                for k, v in self.stage_requirements.items()
            },
            'final_requirements': self.final_requirements
        }


class ValidationError(Exception):
    """验证错误异常"""
    pass


def validate_indicator_passed_status(indicator_name: str, 
                                   stage_results: Dict[str, Dict]) -> bool:
    """验证指标PASSED状态"""
    validator = StrictScoringValidator()
    return validator.validate_final_status(indicator_name, stage_results)


def ensure_stage_score_compliance(stage_key: str, score: float, 
                                uses_real_data: bool = False,
                                real_data_percentage: float = 0.0) -> bool:
    """确保阶段评分合规"""
    validator = StrictScoringValidator()
    return validator.validate_stage_score(
        stage_key, score, uses_real_data, real_data_percentage
    )


def get_stage_requirements(stage_key: str) -> Dict[str, Any]:
    """获取阶段要求"""
    validator = StrictScoringValidator()
    stage_validator = validator.create_stage_validator(stage_key)
    return stage_validator.get_requirements()


if __name__ == "__main__":
    # 测试严格评分验证
    validator = StrictScoringValidator()
    
    # 测试阶段验证
    try:
        validator.validate_stage_score('stage1', 99.5)
        print("✅ 阶段1评分验证通过")
    except ValidationError as e:
        print(f"❌ 阶段1评分验证失败: {e}")
    
    # 测试最终状态验证
    test_results = {
        'stage1': {'overall_score': 99.2, 'uses_real_data': False},
        'stage2': {'overall_score': 99.1, 'uses_real_data': False},
        'stage3': {'overall_score': 99.3, 'uses_real_data': True, 'real_data_percentage': 60.0},
        'stage4': {'overall_score': 99.0, 'uses_real_data': False},
        'stage5': {'overall_score': 99.4, 'uses_real_data': True, 'real_data_percentage': 100.0}
    }
    
    try:
        validator.validate_final_status('TEST_INDICATOR', test_results)
        print("✅ 最终状态验证通过")
    except ValidationError as e:
        print(f"❌ 最终状态验证失败: {e}")
    
    # 输出所有要求
    requirements = validator.get_all_requirements()
    print(f"📋 验证要求: {requirements}")
