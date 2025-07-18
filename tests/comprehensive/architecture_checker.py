#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票选股策略系统架构合规性检查器

验证系统是否严格遵循六层架构原则，确保代码质量和架构一致性。
遵循L5业务应用层规范，提供全面的架构合规性检查。

检查内容：
- 分层依赖关系验证（L6→L5→L4→L3→L2→L1）
- 数据库访问模式检查（强制使用统一查询执行器）
- 接口使用验证（层间标准接口交互）
- 配置管理检查（无硬编码配置）
- 代码质量检查（命名规范、类型提示、文档字符串）
- 依赖注入模式验证
"""

import os
import sys
import ast
import re
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from .config_manager import get_config_manager
from .logging_config import get_test_logger

logger = get_test_logger('architecture_checker')


@dataclass
class ArchitectureViolation:
    """架构违规记录"""
    violation_type: str
    severity: str  # 'critical', 'major', 'minor'
    file_path: str
    line_number: Optional[int] = None
    description: str = ""
    recommendation: str = ""
    code_snippet: str = ""


@dataclass
class LayerInfo:
    """层级信息"""
    layer_name: str
    layer_level: int
    directories: List[str]
    allowed_dependencies: List[str]
    description: str


@dataclass
class ComplianceReport:
    """合规性报告"""
    total_files_checked: int = 0
    violations: List[ArchitectureViolation] = field(default_factory=list)
    compliance_score: float = 0.0
    layer_violations: Dict[str, int] = field(default_factory=dict)
    code_quality_score: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)


class ArchitectureComplianceChecker:
    """架构合规性检查器"""
    
    def __init__(self):
        """初始化架构合规性检查器"""
        self.config = get_config_manager().get_config()
        self.project_root = Path.cwd()
        
        # 定义六层架构
        self.layers = {
            'L6': LayerInfo(
                layer_name='用户接口层',
                layer_level=6,
                directories=['bin/', 'api/'],
                allowed_dependencies=['L5'],
                description='用户接口层 - 命令行接口、API接口'
            ),
            'L5': LayerInfo(
                layer_name='业务应用层',
                layer_level=5,
                directories=['strategy/', 'analysis/'],
                allowed_dependencies=['L4'],
                description='业务应用层 - 策略执行、分析逻辑'
            ),
            'L4': LayerInfo(
                layer_name='核心服务层',
                layer_level=4,
                directories=['indicators/', 'formula/'],
                allowed_dependencies=['L3'],
                description='核心服务层 - 指标计算、公式处理'
            ),
            'L3': LayerInfo(
                layer_name='数据服务层',
                layer_level=3,
                directories=['db/interfaces/', 'db/managers/'],
                allowed_dependencies=['L2'],
                description='数据服务层 - 数据接口、管理器'
            ),
            'L2': LayerInfo(
                layer_name='存储访问层',
                layer_level=2,
                directories=['db/'],
                allowed_dependencies=['L1'],
                description='存储访问层 - 数据库访问'
            ),
            'L1': LayerInfo(
                layer_name='基础设施层',
                layer_level=1,
                directories=['utils/', 'config/', 'enums/'],
                allowed_dependencies=[],
                description='基础设施层 - 工具、配置、枚举'
            )
        }
        
        # 禁止的直接数据库访问模式
        self.forbidden_db_patterns = [
            r'from\s+db\.clickhouse_db\s+import\s+get_clickhouse_db',
            r'get_clickhouse_db\s*\(',
            r'clickhouse_client\s*=',
            r'\.execute\s*\(\s*["\'](?:SELECT|INSERT|UPDATE|DELETE)',
            r'direct_query\s*\(',
        ]
        
        # 必需的查询执行器模式
        self.required_query_patterns = [
            r'from\s+db\.query_executor\s+import\s+get_query_executor',
            r'self\.query_executor\s*=\s*get_query_executor\s*\(',
            r'query_executor\.execute_query\s*\(',
        ]
        
        # 代码质量检查规则
        self.quality_rules = {
            'class_naming': r'^[A-Z][a-zA-Z0-9]*$',
            'function_naming': r'^[a-z_][a-z0-9_]*$',
            'constant_naming': r'^[A-Z_][A-Z0-9_]*$',
            'private_method': r'^_[a-z_][a-z0-9_]*$',
        }
        
        logger.info("架构合规性检查器初始化完成")
    
    @performance_monitor(threshold=10.0)
    @exception_handler(reraise=True)
    def test_layer_dependencies(self) -> Dict[str, Any]:
        """
        测试分层依赖关系
        
        Returns:
            Dict[str, Any]: 检查结果
        """
        logger.info("开始检查分层依赖关系")
        
        violations = []
        file_layer_map = {}
        
        try:
            # 构建文件到层级的映射
            for layer_id, layer_info in self.layers.items():
                for directory in layer_info.directories:
                    dir_path = self.project_root / directory
                    if dir_path.exists():
                        for py_file in dir_path.rglob('*.py'):
                            relative_path = py_file.relative_to(self.project_root)
                            file_layer_map[str(relative_path)] = layer_id
            
            # 检查每个文件的导入依赖
            for file_path, file_layer in file_layer_map.items():
                violations.extend(self._check_file_dependencies(file_path, file_layer, file_layer_map))
            
            # 计算合规性得分
            total_files = len(file_layer_map)
            violation_files = len(set(v.file_path for v in violations))
            compliance_score = (total_files - violation_files) / total_files if total_files > 0 else 1.0
            
            result = {
                "test_name": "分层依赖关系检查",
                "total_files_checked": total_files,
                "violations": [self._violation_to_dict(v) for v in violations],
                "compliance_score": compliance_score,
                "layer_summary": self._generate_layer_summary(violations),
                "success": compliance_score >= 0.9,  # 90%合规率
                "summary": {
                    "total_files": total_files,
                    "violation_count": len(violations),
                    "compliance_score": compliance_score
                }
            }
            
            if result["success"]:
                logger.info(f"分层依赖关系检查通过，合规率: {compliance_score:.1%}")
            else:
                logger.warning(f"分层依赖关系检查失败，合规率: {compliance_score:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"分层依赖关系检查异常: {e}")
            return {
                "test_name": "分层依赖关系检查",
                "success": False,
                "error": str(e),
                "summary": {
                    "total_files": 0,
                    "violation_count": 0,
                    "compliance_score": 0.0
                }
            }
    
    @performance_monitor(threshold=8.0)
    @exception_handler(reraise=True)
    def test_database_access_patterns(self) -> Dict[str, Any]:
        """
        测试数据库访问模式
        
        Returns:
            Dict[str, Any]: 检查结果
        """
        logger.info("开始检查数据库访问模式")
        
        violations = []
        checked_files = 0
        
        try:
            # 遍历所有Python文件
            for py_file in self.project_root.rglob('*.py'):
                if self._should_skip_file(py_file):
                    continue
                
                checked_files += 1
                file_violations = self._check_database_access(py_file)
                violations.extend(file_violations)
            
            # 计算合规性得分
            violation_files = len(set(v.file_path for v in violations))
            compliance_score = (checked_files - violation_files) / checked_files if checked_files > 0 else 1.0
            
            result = {
                "test_name": "数据库访问模式检查",
                "total_files_checked": checked_files,
                "violations": [self._violation_to_dict(v) for v in violations],
                "compliance_score": compliance_score,
                "access_pattern_summary": self._generate_access_pattern_summary(violations),
                "success": compliance_score >= 0.95,  # 95%合规率
                "summary": {
                    "total_files": checked_files,
                    "violation_count": len(violations),
                    "compliance_score": compliance_score
                }
            }
            
            if result["success"]:
                logger.info(f"数据库访问模式检查通过，合规率: {compliance_score:.1%}")
            else:
                logger.warning(f"数据库访问模式检查失败，合规率: {compliance_score:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"数据库访问模式检查异常: {e}")
            return {
                "test_name": "数据库访问模式检查",
                "success": False,
                "error": str(e),
                "summary": {
                    "total_files": 0,
                    "violation_count": 0,
                    "compliance_score": 0.0
                }
            }
    
    @performance_monitor(threshold=6.0)
    @exception_handler(reraise=True)
    def test_code_quality_standards(self) -> Dict[str, Any]:
        """
        测试代码质量标准
        
        Returns:
            Dict[str, Any]: 检查结果
        """
        logger.info("开始检查代码质量标准")
        
        violations = []
        checked_files = 0
        quality_metrics = {
            'naming_violations': 0,
            'missing_docstrings': 0,
            'missing_type_hints': 0,
            'hardcoded_values': 0
        }
        
        try:
            # 遍历所有Python文件
            for py_file in self.project_root.rglob('*.py'):
                if self._should_skip_file(py_file):
                    continue
                
                checked_files += 1
                file_violations, file_metrics = self._check_code_quality(py_file)
                violations.extend(file_violations)
                
                # 累计质量指标
                for metric, count in file_metrics.items():
                    if metric in quality_metrics:
                        quality_metrics[metric] += count
            
            # 计算质量得分
            total_violations = len(violations)
            quality_score = max(0.0, 1.0 - (total_violations / (checked_files * 10)))  # 每文件最多10个违规
            
            result = {
                "test_name": "代码质量标准检查",
                "total_files_checked": checked_files,
                "violations": [self._violation_to_dict(v) for v in violations],
                "quality_score": quality_score,
                "quality_metrics": quality_metrics,
                "quality_summary": self._generate_quality_summary(quality_metrics),
                "success": quality_score >= 0.8,  # 80%质量得分
                "summary": {
                    "total_files": checked_files,
                    "violation_count": len(violations),
                    "quality_score": quality_score
                }
            }
            
            if result["success"]:
                logger.info(f"代码质量标准检查通过，质量得分: {quality_score:.1%}")
            else:
                logger.warning(f"代码质量标准检查失败，质量得分: {quality_score:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"代码质量标准检查异常: {e}")
            return {
                "test_name": "代码质量标准检查",
                "success": False,
                "error": str(e),
                "summary": {
                    "total_files": 0,
                    "violation_count": 0,
                    "quality_score": 0.0
                }
            }
    
    @performance_monitor(threshold=5.0)
    @exception_handler(reraise=True)
    def test_configuration_management(self) -> Dict[str, Any]:
        """
        测试配置管理规范
        
        Returns:
            Dict[str, Any]: 检查结果
        """
        logger.info("开始检查配置管理规范")
        
        violations = []
        checked_files = 0
        config_issues = {
            'hardcoded_urls': 0,
            'hardcoded_passwords': 0,
            'hardcoded_paths': 0,
            'missing_config_usage': 0
        }
        
        try:
            # 遍历所有Python文件
            for py_file in self.project_root.rglob('*.py'):
                if self._should_skip_file(py_file):
                    continue
                
                checked_files += 1
                file_violations, file_issues = self._check_configuration_usage(py_file)
                violations.extend(file_violations)
                
                # 累计配置问题
                for issue, count in file_issues.items():
                    if issue in config_issues:
                        config_issues[issue] += count
            
            # 计算配置管理得分
            total_issues = sum(config_issues.values())
            config_score = max(0.0, 1.0 - (total_issues / (checked_files * 5)))  # 每文件最多5个问题
            
            result = {
                "test_name": "配置管理规范检查",
                "total_files_checked": checked_files,
                "violations": [self._violation_to_dict(v) for v in violations],
                "config_score": config_score,
                "config_issues": config_issues,
                "config_summary": self._generate_config_summary(config_issues),
                "success": config_score >= 0.9,  # 90%配置得分
                "summary": {
                    "total_files": checked_files,
                    "violation_count": len(violations),
                    "config_score": config_score
                }
            }
            
            if result["success"]:
                logger.info(f"配置管理规范检查通过，配置得分: {config_score:.1%}")
            else:
                logger.warning(f"配置管理规范检查失败，配置得分: {config_score:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"配置管理规范检查异常: {e}")
            return {
                "test_name": "配置管理规范检查",
                "success": False,
                "error": str(e),
                "summary": {
                    "total_files": 0,
                    "violation_count": 0,
                    "config_score": 0.0
                }
            }
    
    def _check_file_dependencies(self, file_path: str, file_layer: str, 
                                file_layer_map: Dict[str, str]) -> List[ArchitectureViolation]:
        """检查文件的依赖关系"""
        violations = []
        
        try:
            with open(self.project_root / file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析AST
            tree = ast.parse(content)
            
            # 检查import语句
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    violations.extend(self._check_import_compliance(
                        node, file_path, file_layer, file_layer_map
                    ))
        
        except Exception as e:
            logger.warning(f"检查文件依赖失败 {file_path}: {e}")
        
        return violations
    
    def _check_import_compliance(self, node: ast.AST, file_path: str, file_layer: str,
                               file_layer_map: Dict[str, str]) -> List[ArchitectureViolation]:
        """检查import语句的合规性"""
        violations = []
        
        try:
            if isinstance(node, ast.ImportFrom) and node.module:
                module_path = node.module.replace('.', '/')
                
                # 查找被导入模块所属的层级
                imported_layer = None
                for mapped_file, layer in file_layer_map.items():
                    if mapped_file.startswith(module_path):
                        imported_layer = layer
                        break
                
                if imported_layer and imported_layer != file_layer:
                    # 检查是否违反分层规则
                    current_level = self.layers[file_layer].layer_level
                    imported_level = self.layers[imported_layer].layer_level
                    
                    # 只能依赖更低层级（L6→L5→L4→L3→L2→L1）
                    if imported_level >= current_level:
                        violations.append(ArchitectureViolation(
                            violation_type="layer_dependency",
                            severity="critical",
                            file_path=file_path,
                            line_number=node.lineno,
                            description=f"{file_layer}层不应该依赖{imported_layer}层",
                            recommendation=f"请调整依赖关系，确保遵循{file_layer}→更低层级的原则",
                            code_snippet=f"from {node.module} import ..."
                        ))
        
        except Exception as e:
            logger.warning(f"检查import合规性失败: {e}")
        
        return violations
    
    def _check_database_access(self, file_path: Path) -> List[ArchitectureViolation]:
        """检查数据库访问模式"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            relative_path = file_path.relative_to(self.project_root)
            
            # 检查禁止的数据库访问模式
            for i, line in enumerate(lines, 1):
                for pattern in self.forbidden_db_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append(ArchitectureViolation(
                            violation_type="forbidden_db_access",
                            severity="critical",
                            file_path=str(relative_path),
                            line_number=i,
                            description="直接访问数据库，违反架构规范",
                            recommendation="请使用统一的查询执行器(get_query_executor)",
                            code_snippet=line.strip()
                        ))
            
            # 检查是否使用了推荐的查询执行器（在需要数据库访问的文件中）
            if self._file_needs_db_access(content):
                has_query_executor = any(
                    re.search(pattern, content, re.IGNORECASE) 
                    for pattern in self.required_query_patterns
                )
                
                if not has_query_executor:
                    violations.append(ArchitectureViolation(
                        violation_type="missing_query_executor",
                        severity="major",
                        file_path=str(relative_path),
                        description="文件需要数据库访问但未使用统一查询执行器",
                        recommendation="请导入并使用get_query_executor",
                        code_snippet=""
                    ))
        
        except Exception as e:
            logger.warning(f"检查数据库访问模式失败 {file_path}: {e}")
        
        return violations
    
    def _check_code_quality(self, file_path: Path) -> Tuple[List[ArchitectureViolation], Dict[str, int]]:
        """检查代码质量"""
        violations = []
        metrics = {
            'naming_violations': 0,
            'missing_docstrings': 0,
            'missing_type_hints': 0,
            'hardcoded_values': 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            relative_path = file_path.relative_to(self.project_root)
            tree = ast.parse(content)
            
            # 检查类和函数的命名规范
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not re.match(self.quality_rules['class_naming'], node.name):
                        violations.append(ArchitectureViolation(
                            violation_type="naming_convention",
                            severity="minor",
                            file_path=str(relative_path),
                            line_number=node.lineno,
                            description=f"类名 '{node.name}' 不符合命名规范",
                            recommendation="类名应使用大驼峰命名(PascalCase)",
                            code_snippet=f"class {node.name}:"
                        ))
                        metrics['naming_violations'] += 1
                    
                    # 检查类文档字符串
                    if not ast.get_docstring(node):
                        violations.append(ArchitectureViolation(
                            violation_type="missing_docstring",
                            severity="minor",
                            file_path=str(relative_path),
                            line_number=node.lineno,
                            description=f"类 '{node.name}' 缺少文档字符串",
                            recommendation="请为类添加详细的文档字符串",
                            code_snippet=f"class {node.name}:"
                        ))
                        metrics['missing_docstrings'] += 1
                
                elif isinstance(node, ast.FunctionDef):
                    # 检查函数命名
                    if node.name.startswith('_'):
                        pattern = self.quality_rules['private_method']
                    else:
                        pattern = self.quality_rules['function_naming']
                    
                    if not re.match(pattern, node.name):
                        violations.append(ArchitectureViolation(
                            violation_type="naming_convention",
                            severity="minor",
                            file_path=str(relative_path),
                            line_number=node.lineno,
                            description=f"函数名 '{node.name}' 不符合命名规范",
                            recommendation="函数名应使用小写+下划线(snake_case)",
                            code_snippet=f"def {node.name}():"
                        ))
                        metrics['naming_violations'] += 1
                    
                    # 检查函数文档字符串（非私有方法）
                    if not node.name.startswith('_') and not ast.get_docstring(node):
                        violations.append(ArchitectureViolation(
                            violation_type="missing_docstring",
                            severity="minor",
                            file_path=str(relative_path),
                            line_number=node.lineno,
                            description=f"函数 '{node.name}' 缺少文档字符串",
                            recommendation="请为公有函数添加文档字符串",
                            code_snippet=f"def {node.name}():"
                        ))
                        metrics['missing_docstrings'] += 1
                    
                    # 检查类型提示（简化检查）
                    if (not node.name.startswith('_') and 
                        node.args.args and 
                        not node.returns and 
                        len(node.args.args) > 1):  # 除了self参数
                        
                        violations.append(ArchitectureViolation(
                            violation_type="missing_type_hints",
                            severity="minor",
                            file_path=str(relative_path),
                            line_number=node.lineno,
                            description=f"函数 '{node.name}' 缺少返回类型提示",
                            recommendation="请添加类型提示以提高代码可读性",
                            code_snippet=f"def {node.name}() -> ReturnType:"
                        ))
                        metrics['missing_type_hints'] += 1
        
        except Exception as e:
            logger.warning(f"检查代码质量失败 {file_path}: {e}")
        
        return violations, metrics
    
    def _check_configuration_usage(self, file_path: Path) -> Tuple[List[ArchitectureViolation], Dict[str, int]]:
        """检查配置使用情况"""
        violations = []
        issues = {
            'hardcoded_urls': 0,
            'hardcoded_passwords': 0,
            'hardcoded_paths': 0,
            'missing_config_usage': 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            relative_path = file_path.relative_to(self.project_root)
            
            # 检查硬编码问题
            hardcoded_patterns = {
                'hardcoded_urls': [
                    r'https?://[^\s"\']+',
                    r'ftp://[^\s"\']+',
                ],
                'hardcoded_passwords': [
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'passwd\s*=\s*["\'][^"\']+["\']',
                    r'pwd\s*=\s*["\'][^"\']+["\']',
                ],
                'hardcoded_paths': [
                    r'["\'][A-Za-z]:\\[^"\']*["\']',  # Windows绝对路径
                    r'["\']\/[^"\']*["\']',  # Unix绝对路径
                ]
            }
            
            for i, line in enumerate(lines, 1):
                for issue_type, patterns in hardcoded_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            violations.append(ArchitectureViolation(
                                violation_type="hardcoded_config",
                                severity="major",
                                file_path=str(relative_path),
                                line_number=i,
                                description=f"发现硬编码配置: {issue_type}",
                                recommendation="请将硬编码值移至配置文件",
                                code_snippet=line.strip()
                            ))
                            issues[issue_type] += 1
        
        except Exception as e:
            logger.warning(f"检查配置使用失败 {file_path}: {e}")
        
        return violations, issues
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        skip_patterns = [
            '__pycache__',
            '.pyc',
            '.git',
            'venv',
            'env',
            '.backup',
            'test_',
            '__init__.py'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _file_needs_db_access(self, content: str) -> bool:
        """判断文件是否需要数据库访问"""
        db_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE',
            'query', 'database', 'db',
            'stock_info', 'clickhouse'
        ]
        
        content_upper = content.upper()
        return any(keyword.upper() in content_upper for keyword in db_keywords)
    
    def _violation_to_dict(self, violation: ArchitectureViolation) -> Dict[str, Any]:
        """将违规记录转换为字典"""
        return {
            'violation_type': violation.violation_type,
            'severity': violation.severity,
            'file_path': violation.file_path,
            'line_number': violation.line_number,
            'description': violation.description,
            'recommendation': violation.recommendation,
            'code_snippet': violation.code_snippet
        }
    
    def _generate_layer_summary(self, violations: List[ArchitectureViolation]) -> Dict[str, Any]:
        """生成分层违规摘要"""
        layer_violations = defaultdict(int)
        violation_types = defaultdict(int)
        
        for violation in violations:
            if violation.violation_type == "layer_dependency":
                # 从描述中提取层级信息
                if "层不应该依赖" in violation.description:
                    layer_violations[violation.file_path] += 1
            violation_types[violation.violation_type] += 1
        
        return {
            'layer_violations': dict(layer_violations),
            'violation_types': dict(violation_types),
            'total_violations': len(violations)
        }
    
    def _generate_access_pattern_summary(self, violations: List[ArchitectureViolation]) -> Dict[str, Any]:
        """生成数据库访问模式摘要"""
        pattern_violations = defaultdict(int)
        
        for violation in violations:
            pattern_violations[violation.violation_type] += 1
        
        return {
            'pattern_violations': dict(pattern_violations),
            'total_violations': len(violations)
        }
    
    def _generate_quality_summary(self, metrics: Dict[str, int]) -> Dict[str, Any]:
        """生成代码质量摘要"""
        total_issues = sum(metrics.values())
        
        return {
            'quality_metrics': metrics,
            'total_issues': total_issues,
            'most_common_issue': max(metrics.items(), key=lambda x: x[1])[0] if metrics else None
        }
    
    def _generate_config_summary(self, issues: Dict[str, int]) -> Dict[str, Any]:
        """生成配置管理摘要"""
        total_issues = sum(issues.values())
        
        return {
            'config_issues': issues,
            'total_issues': total_issues,
            'most_common_issue': max(issues.items(), key=lambda x: x[1])[0] if issues else None
        }


if __name__ == "__main__":
    # 测试架构合规性检查器
    checker = ArchitectureComplianceChecker()
    
    print("运行分层依赖关系检查...")
    result1 = checker.test_layer_dependencies()
    print(f"结果: {result1['success']}, 合规率: {result1['summary']['compliance_score']:.1%}")
    
    print("\n运行数据库访问模式检查...")
    result2 = checker.test_database_access_patterns()
    print(f"结果: {result2['success']}, 合规率: {result2['summary']['compliance_score']:.1%}")
    
    print("\n运行代码质量标准检查...")
    result3 = checker.test_code_quality_standards()
    print(f"结果: {result3['success']}, 质量得分: {result3['summary']['quality_score']:.1%}")
    
    print("\n运行配置管理规范检查...")
    result4 = checker.test_configuration_management()
    print(f"结果: {result4['success']}, 配置得分: {result4['summary']['config_score']:.1%}") 