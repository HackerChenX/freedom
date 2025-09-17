#!/usr/bin/env python3
"""
L4核心服务层智能合规性评估
正确理解L4层特点：不同指标有不同入口是正常的，重点是消除功能重复
"""

import os
import ast
import re
from typing import Dict, List, Any, Set
from utils.logger import get_logger

logger = get_logger(__name__)


class L4IntelligentComplianceAssessment:
    """L4核心服务层智能合规性评估"""
    
    def __init__(self):
        self.assessment_results = {}
        self.functional_duplicates = []
        self.base_class_compliance = {}
        
    def execute_intelligent_assessment(self):
        """执行智能合规性评估"""
        logger.info("🎯 开始L4核心服务层智能合规性评估")
        logger.info("正确理解L4层特点：不同指标有不同入口是正常的")
        
        # 第1步：评估基础类合规性
        base_class_score = self._assess_base_class_compliance()
        
        # 第2步：评估功能重复问题
        functional_duplicate_score = self._assess_functional_duplicates()
        
        # 第3步：评估架构扩展性
        extensibility_score = self._assess_architecture_extensibility()
        
        # 第4步：评估分层架构合规性
        layered_architecture_score = self._assess_layered_architecture()
        
        # 计算总体评分
        overall_score = (
            base_class_score * 0.3 +
            functional_duplicate_score * 0.3 +
            extensibility_score * 0.2 +
            layered_architecture_score * 0.2
        )
        
        # 生成评估报告
        self._generate_assessment_report(
            base_class_score,
            functional_duplicate_score,
            extensibility_score,
            layered_architecture_score,
            overall_score
        )
        
        logger.info("✅ L4核心服务层智能合规性评估完成")
    
    def _assess_base_class_compliance(self) -> float:
        """评估基础类合规性"""
        logger.info("第1步：评估基础类合规性")
        
        # 评估BaseIndicator使用情况
        indicator_compliance = self._assess_indicator_base_class_usage()
        
        # 评估BaseStrategy使用情况
        strategy_compliance = self._assess_strategy_base_class_usage()
        
        # 评估BaseAnalyzer使用情况
        analyzer_compliance = self._assess_analyzer_base_class_usage()
        
        # 计算综合评分
        base_class_score = (indicator_compliance + strategy_compliance + analyzer_compliance) / 3
        
        self.base_class_compliance = {
            'indicator_compliance': indicator_compliance,
            'strategy_compliance': strategy_compliance,
            'analyzer_compliance': analyzer_compliance,
            'overall_score': base_class_score
        }
        
        logger.info(f"  基础类合规性评分: {base_class_score:.1f}/100")
        return base_class_score
    
    def _assess_indicator_base_class_usage(self) -> float:
        """评估指标基础类使用情况"""
        indicators_dir = 'indicators/'
        total_indicators = 0
        compliant_indicators = 0
        
        if os.path.exists(indicators_dir):
            for root, dirs, files in os.walk(indicators_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__') and file != 'base_indicator.py':
                        file_path = os.path.join(root, file)
                        total_indicators += 1
                        
                        if self._check_indicator_inheritance(file_path):
                            compliant_indicators += 1
        
        if total_indicators == 0:
            return 100.0
        
        compliance_rate = (compliant_indicators / total_indicators) * 100
        logger.info(f"    指标基础类合规性: {compliance_rate:.1f}% ({compliant_indicators}/{total_indicators})")
        return compliance_rate
    
    def _check_indicator_inheritance(self, file_path: str) -> bool:
        """检查指标是否正确继承BaseIndicator"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否有指标类定义
            if not re.search(r'class\s+\w+.*Indicator', content, re.IGNORECASE):
                return True  # 不是指标文件，不需要检查
            
            # 检查是否继承BaseIndicator
            if 'BaseIndicator' in content:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id == 'BaseIndicator':
                                return True
                            elif isinstance(base, ast.Attribute) and base.attr == 'BaseIndicator':
                                return True
            
            return False
        
        except Exception:
            return False
    
    def _assess_strategy_base_class_usage(self) -> float:
        """评估策略基础类使用情况"""
        strategy_dir = 'strategy/'
        total_strategies = 0
        compliant_strategies = 0
        
        if os.path.exists(strategy_dir):
            for root, dirs, files in os.walk(strategy_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        file_path = os.path.join(root, file)
                        
                        if self._is_strategy_file(file_path):
                            total_strategies += 1
                            
                            if self._check_strategy_inheritance(file_path):
                                compliant_strategies += 1
        
        if total_strategies == 0:
            return 100.0
        
        compliance_rate = (compliant_strategies / total_strategies) * 100
        logger.info(f"    策略基础类合规性: {compliance_rate:.1f}% ({compliant_strategies}/{total_strategies})")
        return compliance_rate
    
    def _is_strategy_file(self, file_path: str) -> bool:
        """判断是否是策略文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return bool(re.search(r'class\s+\w+.*Strategy', content, re.IGNORECASE))
        
        except Exception:
            return False
    
    def _check_strategy_inheritance(self, file_path: str) -> bool:
        """检查策略是否正确继承BaseStrategy"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否继承BaseStrategy或UnifiedBaseStrategy
            if 'BaseStrategy' in content or 'UnifiedBaseStrategy' in content:
                return True
            
            return False
        
        except Exception:
            return False
    
    def _assess_analyzer_base_class_usage(self) -> float:
        """评估分析器基础类使用情况"""
        analysis_dir = 'analysis/'
        total_analyzers = 0
        compliant_analyzers = 0
        
        if os.path.exists(analysis_dir):
            for root, dirs, files in os.walk(analysis_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        file_path = os.path.join(root, file)
                        
                        if self._is_analyzer_file(file_path):
                            total_analyzers += 1
                            
                            if self._check_analyzer_inheritance(file_path):
                                compliant_analyzers += 1
        
        if total_analyzers == 0:
            return 100.0
        
        compliance_rate = (compliant_analyzers / total_analyzers) * 100
        logger.info(f"    分析器基础类合规性: {compliance_rate:.1f}% ({compliant_analyzers}/{total_analyzers})")
        return compliance_rate
    
    def _is_analyzer_file(self, file_path: str) -> bool:
        """判断是否是分析器文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return bool(re.search(r'class\s+\w+.*Analyzer', content, re.IGNORECASE))
        
        except Exception:
            return False
    
    def _check_analyzer_inheritance(self, file_path: str) -> bool:
        """检查分析器是否正确继承BaseAnalyzer"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否继承BaseAnalyzer
            if 'BaseAnalyzer' in content:
                return True
            
            return False
        
        except Exception:
            return False
    
    def _assess_functional_duplicates(self) -> float:
        """评估功能重复问题"""
        logger.info("第2步：评估功能重复问题")
        
        # 识别真正的功能重复
        duplicates = self._identify_real_functional_duplicates()
        
        # 计算评分（基于重复数量）
        if len(duplicates) == 0:
            score = 100.0
        elif len(duplicates) <= 3:
            score = 85.0
        elif len(duplicates) <= 6:
            score = 70.0
        else:
            score = max(50.0, 100.0 - len(duplicates) * 5)
        
        self.functional_duplicates = duplicates
        
        logger.info(f"  功能重复评分: {score:.1f}/100 (发现{len(duplicates)}个重复)")
        return score
    
    def _identify_real_functional_duplicates(self) -> List[Dict[str, Any]]:
        """识别真正的功能重复"""
        duplicates = []
        
        # 检查已知的功能重复模式
        duplicate_patterns = [
            {
                'name': 'MACD指标重复实现',
                'pattern': r'class\s+.*MACD.*',
                'directory': 'indicators/',
                'exclude_files': ['macd.py']  # 保留标准实现
            },
            {
                'name': 'RSI指标重复实现',
                'pattern': r'class\s+.*RSI.*',
                'directory': 'indicators/',
                'exclude_files': ['rsi.py']
            },
            {
                'name': '买点分析器重复实现',
                'pattern': r'class\s+.*Buypoint.*Analyzer',
                'directory': 'analysis/',
                'exclude_files': ['parallel_buypoint_analyzer.py']
            }
        ]
        
        for pattern_info in duplicate_patterns:
            matching_files = self._find_matching_files(pattern_info)
            if len(matching_files) > 1:
                duplicates.append({
                    'name': pattern_info['name'],
                    'files': matching_files,
                    'count': len(matching_files)
                })
        
        return duplicates
    
    def _find_matching_files(self, pattern_info: Dict[str, Any]) -> List[str]:
        """查找匹配模式的文件"""
        matching_files = []
        directory = pattern_info['directory']
        pattern = pattern_info['pattern']
        exclude_files = pattern_info.get('exclude_files', [])
        
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py') and file not in exclude_files:
                        file_path = os.path.join(root, file)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            if re.search(pattern, content, re.IGNORECASE):
                                matching_files.append(file_path)
                        
                        except Exception:
                            continue
        
        return matching_files
    
    def _assess_architecture_extensibility(self) -> float:
        """评估架构扩展性"""
        logger.info("第3步：评估架构扩展性")
        
        # 检查接口设计质量
        interface_quality = self._evaluate_interface_quality()
        
        # 检查组件化程度
        componentization = self._evaluate_componentization()
        
        # 检查依赖注入使用
        dependency_injection = self._evaluate_dependency_injection_usage()
        
        extensibility_score = (interface_quality + componentization + dependency_injection) / 3
        
        logger.info(f"  架构扩展性评分: {extensibility_score:.1f}/100")
        return extensibility_score
    
    def _evaluate_interface_quality(self) -> float:
        """评估接口设计质量"""
        # 简化评估：检查是否有抽象基类
        abstract_classes = 0
        total_classes = 0
        
        for directory in ['indicators/', 'strategy/', 'analysis/']:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                if 'ABC' in content or 'abstractmethod' in content:
                                    abstract_classes += 1
                                
                                if 'class ' in content:
                                    total_classes += 1
                            
                            except Exception:
                                continue
        
        if total_classes == 0:
            return 80.0
        
        interface_ratio = (abstract_classes / total_classes) * 100
        return min(90.0, 60.0 + interface_ratio * 2)
    
    def _evaluate_componentization(self) -> float:
        """评估组件化程度"""
        # 检查组件化模式的使用
        component_patterns = ['Factory', 'Registry', 'Manager', 'Service', 'Engine']
        pattern_count = 0
        
        for directory in ['indicators/', 'strategy/', 'analysis/']:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py'):
                            for pattern in component_patterns:
                                if pattern.lower() in file.lower():
                                    pattern_count += 1
                                    break
        
        # 基于组件化模式使用情况评分
        if pattern_count >= 15:
            return 90.0
        elif pattern_count >= 8:
            return 75.0
        else:
            return 60.0
    
    def _evaluate_dependency_injection_usage(self) -> float:
        """评估依赖注入使用情况"""
        di_usage_count = 0
        total_files = 0
        
        for directory in ['indicators/', 'strategy/', 'analysis/']:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            total_files += 1
                            
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                if 'container.resolve' in content or 'get_service' in content:
                                    di_usage_count += 1
                            
                            except Exception:
                                continue
        
        if total_files == 0:
            return 80.0
        
        di_ratio = (di_usage_count / total_files) * 100
        return min(95.0, 50.0 + di_ratio * 2)
    
    def _assess_layered_architecture(self) -> float:
        """评估分层架构合规性"""
        logger.info("第4步：评估分层架构合规性")
        
        # 检查跨层调用违规
        cross_layer_violations = self._check_cross_layer_violations()
        
        # 检查类职责过多问题
        responsibility_violations = self._check_responsibility_violations()
        
        # 计算评分
        total_violations = len(cross_layer_violations) + len(responsibility_violations)
        
        if total_violations == 0:
            score = 100.0
        elif total_violations <= 5:
            score = 90.0
        elif total_violations <= 10:
            score = 80.0
        else:
            score = max(60.0, 100.0 - total_violations * 3)
        
        logger.info(f"  分层架构合规性评分: {score:.1f}/100 (发现{total_violations}个违规)")
        return score
    
    def _check_cross_layer_violations(self) -> List[str]:
        """检查跨层调用违规"""
        violations = []
        
        # L4层不应该直接调用L2层
        l4_directories = ['indicators/', 'strategy/', 'analysis/']
        
        for directory in l4_directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                # 检查是否直接导入L2层
                                if 'from db.enhanced_connection_pool' in content:
                                    violations.append(f"跨层调用违规: {file_path} 直接调用L2层")
                                
                                # 检查是否直接导入L1层数据库
                                if 'from db.clickhouse_db' in content:
                                    violations.append(f"跨层调用违规: {file_path} 直接调用L1层")
                            
                            except Exception:
                                continue
        
        return violations
    
    def _check_responsibility_violations(self) -> List[str]:
        """检查类职责过多问题"""
        violations = []
        
        for directory in ['indicators/', 'strategy/', 'analysis/']:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                tree = ast.parse(content)
                                
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.ClassDef):
                                        method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                                        
                                        # L4层核心服务类可以有更多方法，但需要合理性说明
                                        if method_count > 30:
                                            violations.append(f"职责过多: {file_path}:{node.name} 有{method_count}个方法")
                                        elif method_count > 20:
                                            # 检查是否有合理性说明
                                            class_docstring = ast.get_docstring(node)
                                            if not class_docstring or '合理性' not in class_docstring:
                                                violations.append(f"职责说明缺失: {file_path}:{node.name} 有{method_count}个方法但缺少合理性说明")
                            
                            except Exception:
                                continue
        
        return violations
    
    def _generate_assessment_report(self, base_class_score: float, functional_duplicate_score: float,
                                  extensibility_score: float, layered_architecture_score: float,
                                  overall_score: float):
        """生成评估报告"""
        print("\n" + "="*80)
        print("🎯 L4核心服务层智能合规性评估报告")
        print("正确理解L4层特点：不同指标有不同入口是正常的")
        print("="*80)
        
        print(f"\n📊 详细评分:")
        print(f"  基础类合规性: {base_class_score:.1f}/100")
        print(f"  功能重复控制: {functional_duplicate_score:.1f}/100")
        print(f"  架构扩展性: {extensibility_score:.1f}/100")
        print(f"  分层架构合规性: {layered_architecture_score:.1f}/100")
        
        print(f"\n🏆 总体评分: {overall_score:.1f}/100")
        
        if overall_score >= 90:
            grade = "A+"
            status = "EXCELLENT"
        elif overall_score >= 80:
            grade = "A"
            status = "GOOD"
        elif overall_score >= 70:
            grade = "B"
            status = "ACCEPTABLE"
        else:
            grade = "C"
            status = "NEEDS_IMPROVEMENT"
        
        print(f"评级: {grade} ({status})")
        
        print(f"\n📋 基础类合规性详情:")
        for key, value in self.base_class_compliance.items():
            if key != 'overall_score':
                print(f"  {key}: {value:.1f}%")
        
        if self.functional_duplicates:
            print(f"\n⚠️ 发现的功能重复:")
            for i, duplicate in enumerate(self.functional_duplicates[:5], 1):
                print(f"  {i}. {duplicate['name']}: {duplicate['count']}个文件")
        
        print(f"\n🎯 优化建议:")
        if base_class_score < 80:
            print("  • 继续完善基础类继承体系")
        if functional_duplicate_score < 80:
            print("  • 消除剩余的功能重复实现")
        if extensibility_score < 80:
            print("  • 加强接口设计和组件化")
        if layered_architecture_score < 80:
            print("  • 修复分层架构违规问题")
        
        print("="*80)


def main():
    """主函数"""
    try:
        assessment = L4IntelligentComplianceAssessment()
        assessment.execute_intelligent_assessment()
        return 0
        
    except Exception as e:
        logger.error(f"L4智能合规性评估异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
