#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准化验证流程模板

为所有指标提供统一的验证流程模板，确保验证标准一致性
"""

import sys
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from utils.dependency_injection import get_logger
from tests.framework.real_data_validator import RealDataValidator, ensure_real_data_usage
from tests.framework.strict_scoring_validator import StrictScoringValidator, ValidationError

logger = get_logger(__name__)


class StandardizedIndicatorValidator(ABC):
    """标准化指标验证器基类"""
    
    def __init__(self, indicator_name: str):
        """初始化验证器"""
        self.indicator_name = indicator_name
        self.start_time = datetime.now()
        self.real_data_validator = RealDataValidator()
        self.scoring_validator = StrictScoringValidator()
        
        logger.info(f"✅ {indicator_name}标准化验证器初始化完成")
        logger.info(f"🎯 严格标准: 所有阶段≥99分，平均≥99.5分，真实数据验证")
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """运行完整的5阶段验证"""
        logger.info(f"🚀 开始{self.indicator_name}完整验证流程")
        
        validation_results = {
            'indicator_name': self.indicator_name,
            'validation_start_time': self.start_time.isoformat(),
            'validation_standards': self.scoring_validator.get_all_requirements(),
            'stage_results': {},
            'final_assessment': {},
            'final_status': 'IN_PROGRESS'
        }
        
        try:
            # 阶段1: 算法差异预分析
            logger.info("📊 阶段1: 算法差异预分析")
            stage1_result = self.run_stage1_algorithm_analysis()
            self._validate_stage_result('stage1', stage1_result)
            validation_results['stage_results']['stage1'] = stage1_result
            
            # 阶段2: 基础功能验证
            logger.info("🔧 阶段2: 基础功能验证")
            stage2_result = self.run_stage2_basic_functionality()
            self._validate_stage_result('stage2', stage2_result)
            validation_results['stage_results']['stage2'] = stage2_result
            
            # 阶段3: 形态识别验证
            logger.info("🎯 阶段3: 形态识别验证")
            stage3_result = self.run_stage3_pattern_recognition()
            self._validate_stage_result('stage3', stage3_result)
            validation_results['stage_results']['stage3'] = stage3_result
            
            # 阶段4: 架构合规性验证
            logger.info("🏗️ 阶段4: 架构合规性验证")
            stage4_result = self.run_stage4_architecture_compliance()
            self._validate_stage_result('stage4', stage4_result)
            validation_results['stage_results']['stage4'] = stage4_result
            
            # 阶段5: 生产就绪性验证
            logger.info("🚀 阶段5: 生产就绪性验证")
            stage5_result = self.run_stage5_production_readiness()
            self._validate_stage_result('stage5', stage5_result)
            validation_results['stage_results']['stage5'] = stage5_result
            
            # 最终评估
            logger.info("📊 最终评估")
            final_assessment = self._generate_final_assessment(validation_results['stage_results'])
            validation_results['final_assessment'] = final_assessment
            
            # 验证最终PASSED状态
            self.scoring_validator.validate_final_status(
                self.indicator_name, validation_results['stage_results']
            )
            
            validation_results['final_status'] = 'PASSED_ARCHITECTURE_COMPLIANT'
            validation_results['validation_end_time'] = datetime.now().isoformat()
            
            logger.info(f"✅ {self.indicator_name}完整验证流程通过")
            return validation_results
            
        except ValidationError as e:
            logger.error(f"❌ {self.indicator_name}验证失败: {e}")
            validation_results['final_status'] = 'VALIDATION_FAILED'
            validation_results['error'] = str(e)
            validation_results['validation_end_time'] = datetime.now().isoformat()
            return validation_results
            
        except Exception as e:
            logger.error(f"💥 {self.indicator_name}验证异常: {e}")
            validation_results['final_status'] = 'ERROR'
            validation_results['error'] = str(e)
            validation_results['traceback'] = traceback.format_exc()
            validation_results['validation_end_time'] = datetime.now().isoformat()
            return validation_results
    
    def _validate_stage_result(self, stage_key: str, stage_result: Dict[str, Any]):
        """验证阶段结果"""
        score = stage_result.get('overall_score', 0)
        uses_real_data = stage_result.get('uses_real_data', False)
        real_data_percentage = stage_result.get('real_data_percentage', 0.0)
        
        self.scoring_validator.validate_stage_score(
            stage_key, score, uses_real_data, real_data_percentage
        )
    
    def _generate_final_assessment(self, stage_results: Dict[str, Dict]) -> Dict[str, Any]:
        """生成最终评估"""
        scores = [result.get('overall_score', 0) for result in stage_results.values()]
        average_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'stage_scores': {k: v.get('overall_score', 0) for k, v in stage_results.items()},
            'average_score': average_score,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'all_stages_passed': all(s >= 99.0 for s in scores),
            'average_score_passed': average_score >= 99.5,
            'real_data_compliance': self._check_real_data_compliance(stage_results),
            'final_passed': average_score >= 99.5 and all(s >= 99.0 for s in scores)
        }
    
    def _check_real_data_compliance(self, stage_results: Dict[str, Dict]) -> Dict[str, bool]:
        """检查真实数据合规性"""
        compliance = {}
        
        # 阶段3需要50%真实数据
        if 'stage3' in stage_results:
            stage3 = stage_results['stage3']
            compliance['stage3'] = (
                stage3.get('uses_real_data', False) and 
                stage3.get('real_data_percentage', 0) >= 50.0
            )
        
        # 阶段5需要100%真实数据
        if 'stage5' in stage_results:
            stage5 = stage_results['stage5']
            compliance['stage5'] = (
                stage5.get('uses_real_data', False) and 
                stage5.get('real_data_percentage', 0) >= 100.0
            )
        
        return compliance
    
    # 抽象方法 - 子类必须实现
    @abstractmethod
    def run_stage1_algorithm_analysis(self) -> Dict[str, Any]:
        """阶段1: 算法差异预分析"""
        pass
    
    @abstractmethod
    def run_stage2_basic_functionality(self) -> Dict[str, Any]:
        """阶段2: 基础功能验证"""
        pass
    
    @abstractmethod
    def run_stage3_pattern_recognition(self) -> Dict[str, Any]:
        """阶段3: 形态识别验证"""
        pass
    
    @abstractmethod
    def run_stage4_architecture_compliance(self) -> Dict[str, Any]:
        """阶段4: 架构合规性验证"""
        pass
    
    @abstractmethod
    def run_stage5_production_readiness(self) -> Dict[str, Any]:
        """阶段5: 生产就绪性验证"""
        pass
    
    # 辅助方法
    def get_real_data(self, limit: int = 10000) -> Any:
        """获取真实数据"""
        return self.real_data_validator.get_real_stock_data(limit=limit)
    
    def validate_real_data(self, data: Any) -> bool:
        """验证真实数据"""
        return self.real_data_validator.validate_real_data(data)
    
    def create_stage_result(self, overall_score: float, 
                          uses_real_data: bool = False,
                          real_data_percentage: float = 0.0,
                          **kwargs) -> Dict[str, Any]:
        """创建标准化阶段结果"""
        result = {
            'overall_score': overall_score,
            'uses_real_data': uses_real_data,
            'real_data_percentage': real_data_percentage,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        return result


class IndicatorValidationTemplate:
    """指标验证模板"""
    
    @staticmethod
    def create_validator_class(indicator_name: str) -> type:
        """创建指标验证器类"""
        
        class SpecificIndicatorValidator(StandardizedIndicatorValidator):
            def __init__(self):
                super().__init__(indicator_name)
            
            def run_stage1_algorithm_analysis(self) -> Dict[str, Any]:
                """阶段1: 算法差异预分析 - 需要子类实现"""
                raise NotImplementedError(f"{indicator_name}需要实现阶段1验证")
            
            def run_stage2_basic_functionality(self) -> Dict[str, Any]:
                """阶段2: 基础功能验证 - 需要子类实现"""
                raise NotImplementedError(f"{indicator_name}需要实现阶段2验证")
            
            def run_stage3_pattern_recognition(self) -> Dict[str, Any]:
                """阶段3: 形态识别验证 - 需要子类实现"""
                raise NotImplementedError(f"{indicator_name}需要实现阶段3验证")
            
            def run_stage4_architecture_compliance(self) -> Dict[str, Any]:
                """阶段4: 架构合规性验证 - 需要子类实现"""
                raise NotImplementedError(f"{indicator_name}需要实现阶段4验证")
            
            def run_stage5_production_readiness(self) -> Dict[str, Any]:
                """阶段5: 生产就绪性验证 - 需要子类实现"""
                raise NotImplementedError(f"{indicator_name}需要实现阶段5验证")
        
        return SpecificIndicatorValidator


def create_standardized_validator(indicator_name: str) -> StandardizedIndicatorValidator:
    """创建标准化验证器"""
    validator_class = IndicatorValidationTemplate.create_validator_class(indicator_name)
    return validator_class()


def validate_indicator_with_strict_standards(indicator_name: str, 
                                           validator: StandardizedIndicatorValidator) -> bool:
    """使用严格标准验证指标"""
    try:
        results = validator.run_complete_validation()
        
        if results['final_status'] == 'PASSED_ARCHITECTURE_COMPLIANT':
            logger.info(f"🎉 {indicator_name}通过严格标准验证")
            return True
        else:
            logger.error(f"❌ {indicator_name}未通过严格标准验证: {results['final_status']}")
            return False
            
    except Exception as e:
        logger.error(f"💥 {indicator_name}验证过程异常: {e}")
        return False


if __name__ == "__main__":
    # 测试标准化验证模板
    try:
        validator = create_standardized_validator("TEST_INDICATOR")
        print("✅ 标准化验证模板创建成功")
        
        # 获取验证要求
        requirements = validator.scoring_validator.get_all_requirements()
        print(f"📋 验证要求: {requirements}")
        
    except Exception as e:
        print(f"❌ 标准化验证模板测试失败: {e}")
