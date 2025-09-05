"""
架构合规性验证器
严格按照六层架构规范，检查代码是否符合架构要求
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """架构层类型"""
    L1_INFRASTRUCTURE = "L1_基础设施层"
    L2_STORAGE = "L2_存储访问层"
    L3_DATA_SERVICE = "L3_数据服务层"
    L4_CORE_SERVICE = "L4_核心服务层"
    L5_BUSINESS = "L5_业务应用层"
    L6_INTERFACE = "L6_用户接口层"


@dataclass
class LayerDefinition:
    """层定义"""
    layer_type: LayerType
    directories: List[str]
    allowed_dependencies: List[LayerType]
    description: str


@dataclass
class ViolationReport:
    """违规报告"""
    file_path: str
    line_number: int
    violation_type: str
    description: str
    severity: str
    suggestion: str


class ArchitectureValidator:
    """
    架构合规性验证器
    
    检查代码是否符合六层架构规范
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.violations: List[ViolationReport] = []
        
        # 定义六层架构
        self.layers = {
            LayerType.L1_INFRASTRUCTURE: LayerDefinition(
                layer_type=LayerType.L1_INFRASTRUCTURE,
                directories=["utils/", "config/", "enums/"],
                allowed_dependencies=[],
                description="基础设施层：工具函数、配置、枚举"
            ),
            LayerType.L2_STORAGE: LayerDefinition(
                layer_type=LayerType.L2_STORAGE,
                directories=["db/enhanced_connection_pool.py", "db/clickhouse_db.py"],
                allowed_dependencies=[LayerType.L1_INFRASTRUCTURE],
                description="存储访问层：数据库连接和基础访问"
            ),
            LayerType.L3_DATA_SERVICE: LayerDefinition(
                layer_type=LayerType.L3_DATA_SERVICE,
                directories=["db/interfaces/", "db/managers/"],
                allowed_dependencies=[LayerType.L1_INFRASTRUCTURE, LayerType.L2_STORAGE],
                description="数据服务层：数据访问接口和管理器"
            ),
            LayerType.L4_CORE_SERVICE: LayerDefinition(
                layer_type=LayerType.L4_CORE_SERVICE,
                directories=["indicators/", "formula/"],
                allowed_dependencies=[LayerType.L1_INFRASTRUCTURE, LayerType.L2_STORAGE, LayerType.L3_DATA_SERVICE],
                description="核心服务层：技术指标和公式计算"
            ),
            LayerType.L5_BUSINESS: LayerDefinition(
                layer_type=LayerType.L5_BUSINESS,
                directories=["strategy/", "analysis/"],
                allowed_dependencies=[LayerType.L1_INFRASTRUCTURE, LayerType.L2_STORAGE, 
                                    LayerType.L3_DATA_SERVICE, LayerType.L4_CORE_SERVICE],
                description="业务应用层：策略和分析逻辑"
            ),
            LayerType.L6_INTERFACE: LayerDefinition(
                layer_type=LayerType.L6_INTERFACE,
                directories=["bin/", "api/"],
                allowed_dependencies=[LayerType.L1_INFRASTRUCTURE, LayerType.L2_STORAGE, 
                                    LayerType.L3_DATA_SERVICE, LayerType.L4_CORE_SERVICE, LayerType.L5_BUSINESS],
                description="用户接口层：命令行工具和API接口"
            )
        }
        
        # 禁止的导入模式
        self.forbidden_patterns = [
            # 直接数据库依赖
            (r"from db\.clickhouse_db import", "禁止直接导入数据库实现"),
            (r"from db\.enhanced_connection_pool import.*ClickHouseConnectionPool", "禁止直接导入连接池"),
            
            # 通配符导入
            (r"from .* import \*", "禁止通配符导入"),
            
            # 相对导入
            (r"from \.\..*import", "禁止相对导入"),
            
            # 跨层调用检查将在具体方法中实现
        ]
    
    def get_file_layer(self, file_path: str) -> Optional[LayerType]:
        """
        获取文件所属的架构层
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[LayerType]: 架构层类型
        """
        file_path = file_path.replace("\\", "/")  # 统一路径分隔符
        
        for layer_type, layer_def in self.layers.items():
            for directory in layer_def.directories:
                if file_path.startswith(directory):
                    return layer_type
        
        return None
    
    def is_import_allowed(self, from_layer: LayerType, to_layer: LayerType) -> bool:
        """
        检查导入是否被允许
        
        Args:
            from_layer: 源层
            to_layer: 目标层
            
        Returns:
            bool: 是否允许
        """
        if from_layer == to_layer:
            return True  # 同层导入允许
        
        layer_def = self.layers.get(from_layer)
        if not layer_def:
            return False
        
        return to_layer in layer_def.allowed_dependencies
    
    def extract_imports_from_file(self, file_path: Path) -> List[Tuple[str, int]]:
        """
        从文件中提取导入语句
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Tuple[str, int]]: (导入语句, 行号) 列表
        """
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用AST解析导入
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((f"import {alias.name}", node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    import_stmt = f"from {module} import {', '.join(names)}"
                    imports.append((import_stmt, node.lineno))
        
        except Exception as e:
            logger.warning(f"解析文件 {file_path} 失败: {e}")
        
        return imports
    
    def validate_file(self, file_path: Path) -> List[ViolationReport]:
        """
        验证单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[ViolationReport]: 违规报告列表
        """
        violations = []
        relative_path = str(file_path.relative_to(self.project_root))
        
        # 获取文件所属层
        file_layer = self.get_file_layer(relative_path)
        if not file_layer:
            return violations  # 不在架构层中的文件跳过检查
        
        # 提取导入语句
        imports = self.extract_imports_from_file(file_path)
        
        for import_stmt, line_number in imports:
            # 检查禁止的导入模式
            for pattern, description in self.forbidden_patterns:
                if re.search(pattern, import_stmt):
                    violations.append(ViolationReport(
                        file_path=relative_path,
                        line_number=line_number,
                        violation_type="FORBIDDEN_IMPORT",
                        description=f"{description}: {import_stmt}",
                        severity="ERROR",
                        suggestion="使用依赖注入或接口导入"
                    ))
            
            # 检查跨层调用
            target_layer = self._get_import_target_layer(import_stmt)
            if target_layer and not self.is_import_allowed(file_layer, target_layer):
                violations.append(ViolationReport(
                    file_path=relative_path,
                    line_number=line_number,
                    violation_type="LAYER_VIOLATION",
                    description=f"违反分层架构: {file_layer.value} 不能导入 {target_layer.value}",
                    severity="ERROR",
                    suggestion=f"通过依赖注入或使用 {file_layer.value} 允许的层"
                ))
        
        return violations
    
    def _get_import_target_layer(self, import_stmt: str) -> Optional[LayerType]:
        """
        获取导入语句的目标层
        
        Args:
            import_stmt: 导入语句
            
        Returns:
            Optional[LayerType]: 目标层类型
        """
        # 提取模块路径
        if import_stmt.startswith("from "):
            match = re.match(r"from\s+([^\s]+)\s+import", import_stmt)
            if match:
                module_path = match.group(1)
            else:
                return None
        elif import_stmt.startswith("import "):
            match = re.match(r"import\s+([^\s]+)", import_stmt)
            if match:
                module_path = match.group(1)
            else:
                return None
        else:
            return None
        
        # 转换为文件路径格式
        file_path = module_path.replace(".", "/")
        
        return self.get_file_layer(file_path)
    
    def validate_project(self) -> Dict[str, any]:
        """
        验证整个项目
        
        Returns:
            Dict[str, any]: 验证结果
        """
        logger.info("开始架构合规性验证...")
        
        self.violations.clear()
        
        # 遍历所有Python文件
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            # 跳过测试文件和临时文件
            if any(skip in str(file_path) for skip in ["test_", "__pycache__", ".git", "venv"]):
                continue
            
            file_violations = self.validate_file(file_path)
            self.violations.extend(file_violations)
        
        # 统计结果
        error_count = sum(1 for v in self.violations if v.severity == "ERROR")
        warning_count = sum(1 for v in self.violations if v.severity == "WARNING")
        
        result = {
            "total_files": len(python_files),
            "total_violations": len(self.violations),
            "error_count": error_count,
            "warning_count": warning_count,
            "violations": self.violations,
            "is_compliant": error_count == 0
        }
        
        logger.info(f"架构验证完成: {result['total_files']} 个文件, "
                   f"{error_count} 个错误, {warning_count} 个警告")
        
        return result
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        生成验证报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        result = self.validate_project()
        
        report_lines = [
            "# 架构合规性验证报告",
            "",
            f"**验证时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**验证文件数**: {result['total_files']}",
            f"**违规总数**: {result['total_violations']}",
            f"**错误数**: {result['error_count']}",
            f"**警告数**: {result['warning_count']}",
            f"**合规状态**: {'✅ 合规' if result['is_compliant'] else '❌ 不合规'}",
            "",
            "## 违规详情",
            ""
        ]
        
        if not self.violations:
            report_lines.append("🎉 恭喜！没有发现架构违规问题。")
        else:
            # 按文件分组显示违规
            violations_by_file = {}
            for violation in self.violations:
                if violation.file_path not in violations_by_file:
                    violations_by_file[violation.file_path] = []
                violations_by_file[violation.file_path].append(violation)
            
            for file_path, file_violations in violations_by_file.items():
                report_lines.append(f"### {file_path}")
                report_lines.append("")
                
                for violation in file_violations:
                    severity_icon = "🔴" if violation.severity == "ERROR" else "🟡"
                    report_lines.extend([
                        f"{severity_icon} **第{violation.line_number}行** - {violation.violation_type}",
                        f"   - **问题**: {violation.description}",
                        f"   - **建议**: {violation.suggestion}",
                        ""
                    ])
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"验证报告已保存到: {output_file}")
        
        return report_content


def validate_architecture(project_root: str = ".", output_file: Optional[str] = None) -> bool:
    """
    验证项目架构合规性
    
    Args:
        project_root: 项目根目录
        output_file: 输出报告文件
        
    Returns:
        bool: 是否合规
    """
    validator = ArchitectureValidator(project_root)
    result = validator.validate_project()
    
    if output_file:
        validator.generate_report(output_file)
    
    return result['is_compliant']


# 导出主要类和函数
__all__ = [
    'ArchitectureValidator',
    'LayerType',
    'ViolationReport',
    'validate_architecture'
]
