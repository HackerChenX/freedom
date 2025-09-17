"""
指标继承合规性持续监控
定期检查所有指标的继承合规性，确保质量标准
"""

import os
import ast
import re
from typing import Dict, List, Any
from datetime import datetime


class InheritanceComplianceMonitor:
    """指标继承合规性监控器"""

    def __init__(self):
        self.monitoring_results = {}

    def run_compliance_check(self) -> Dict[str, Any]:
        """运行合规性检查"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_indicators": 0,
            "compliant_indicators": 0,
            "compliance_rate": 0.0,
            "issues_found": [],
            "recommendations": [],
        }

        # 发现所有指标文件
        indicator_files = self._discover_indicator_files()
        results["total_indicators"] = len(indicator_files)

        # 检查每个指标的合规性
        compliant_count = 0
        for file_path in indicator_files:
            if self._check_single_indicator_compliance(file_path, results):
                compliant_count += 1

        results["compliant_indicators"] = compliant_count
        results["compliance_rate"] = (compliant_count / max(len(indicator_files), 1)) * 100

        # 生成建议
        self._generate_recommendations(results)

        return results

    def _discover_indicator_files(self) -> List[str]:
        """发现指标文件"""
        indicator_files = []
        indicators_dir = "indicators/"

        if os.path.exists(indicators_dir):
            for root, dirs, files in os.walk(indicators_dir):
                for file in files:
                    if (
                        file.endswith(".py")
                        and not file.startswith("__")
                        and file not in ["base_indicator.py", "indicator_template.py"]
                    ):

                        file_path = os.path.join(root, file)
                        if self._is_indicator_file(file_path):
                            indicator_files.append(file_path)

        return indicator_files

    def _is_indicator_file(self, file_path: str) -> bool:
        """判断是否是指标文件"""
        try:
            with open(
                file_path, "r", encoding="utf-8"
            ) as f:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                content = f.read()
            return bool(re.search(r"class\s+\w*[Ii]ndicator\w*", content))
        except Exception:
            return False

    def _check_single_indicator_compliance(self, file_path: str, results: Dict[str, Any]) -> bool:
        """检查单个指标的合规性"""
        try:
            with open(
                file_path, "r", encoding="utf-8"
            ) as f:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                content = f.read()

            # 检查基本要求
            has_base_import = "from indicators.base_indicator import BaseIndicator" in content

            if not has_base_import:
                results["issues_found"].append({"file": file_path, "issue": "缺少BaseIndicator导入"})
                return False

            # AST分析
            try:
                tree = ast.parse(content)
                return self._analyze_compliance_ast(tree, file_path, results)
            except SyntaxError:
                results["issues_found"].append({"file": file_path, "issue": "语法错误"})
                return False

        except Exception:
            results["issues_found"].append({"file": file_path, "issue": "文件读取失败"})
            return False

    def _analyze_compliance_ast(self, tree: ast.AST, file_path: str, results: Dict[str, Any]) -> bool:
        """AST分析合规性"""
        required_methods = {"calculate", "get_signal"}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if re.search(r"[Ii]ndicator", node.name):
                    # 检查继承
                    inherits_base = any(
                        (isinstance(base, ast.Name) and base.id == "BaseIndicator")
                        or (isinstance(base, ast.Attribute) and base.attr == "BaseIndicator")
                        for base in node.bases
                    )

                    if not inherits_base:
                        results["issues_found"].append({"file": file_path, "issue": f"{node.name}未继承BaseIndicator"})
                        return False

                    # 检查方法实现
                    implemented_methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}

                    missing_methods = required_methods - implemented_methods
                    if missing_methods:
                        results["issues_found"].append(
                            {"file": file_path, "issue": f"{node.name}缺少方法: {list(missing_methods)}"}
                        )
                        return False

                    return True

        return False

    def _generate_recommendations(self, results: Dict[str, Any]):
        """生成改进建议"""
        if results["compliance_rate"] < 95:  # TODO: 将魔法数字提取到配置中
            results["recommendations"].append("建议修复所有继承合规性问题以达到A+级标准")

        if len(results["issues_found"]) > 0:
            results["recommendations"].append("建议优先修复语法错误和缺少导入的问题")

        if results["compliance_rate"] >= 95:  # TODO: 将魔法数字提取到配置中
            results["recommendations"].append("继承合规性已达到A+级标准，建议保持")


# 使用示例
if __name__ == "__main__":
    monitor = InheritanceComplianceMonitor()
    results = monitor.run_compliance_check()

    print(f"继承合规性监控结果:")
    print(f"合规率: {results['compliance_rate']:.1f}%")
    print(f"发现问题: {len(results['issues_found'])}个")
