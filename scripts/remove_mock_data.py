"""
系统模拟数据移除脚本
扫描并移除系统中所有模拟数据相关代码
"""

import os
import re
import ast
import shutil
from typing import List, Dict, Tuple, Set
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class MockDataRemovalTool:
    """模拟数据移除工具"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup" / f"mock_removal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 模拟数据相关的关键词
        self.mock_keywords = [
            '_create_mock_data',
            'MockDataAccess',
            'MockDataAccessManager',
            'mock_data',
            'simulate_data',
            'fake_data',
            'dummy_data',
            'test_data_generator',
            '_generate_mock',
            'create_fake',
            'simulate_',
        ]

        # 需要检查的文件类型
        self.file_patterns = ['**/*.py']

        # 排除的目录
        self.exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv'}

    def scan_mock_data_usage(self) -> Dict[str, List[Dict]]:
        """
        扫描系统中的模拟数据使用情况

        Returns:
            包含文件路径和模拟数据使用信息的字典
        """
        results = {}

        for pattern in self.file_patterns:
            for file_path in self.project_root.glob(pattern):
                # 跳过排除的目录
                if any(exclude_dir in file_path.parts for exclude_dir in self.exclude_dirs):
                    continue

                mock_usages = self._analyze_file_for_mock_data(file_path)
                if mock_usages:
                    results[str(file_path)] = mock_usages

        return results

    def _analyze_file_for_mock_data(self, file_path: Path) -> List[Dict]:
        """分析单个文件中的模拟数据使用"""
        mock_usages = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # 使用AST分析
            try:
                tree = ast.parse(content)
                mock_usages.extend(self._analyze_ast_for_mock_data(tree, lines))
            except SyntaxError:
                logger.warning(f"无法解析文件 {file_path}，跳过AST分析")

            # 使用正则表达式分析
            mock_usages.extend(self._analyze_regex_for_mock_data(content, lines))

        except Exception as e:
            logger.error(f"分析文件失败 {file_path}: {e}")

        return mock_usages

    def _analyze_ast_for_mock_data(self, tree: ast.AST, lines: List[str]) -> List[Dict]:
        """使用AST分析模拟数据"""
        mock_usages = []

        class MockDataVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                if any(keyword in node.name.lower() for keyword in self.mock_keywords):
                    mock_usages.append({
                        'type': 'function_definition',
                        'name': node.name,
                        'line': node.lineno,
                        'content': lines[node.lineno - 1].strip() if node.lineno <= len(lines) else '',
                        'severity': 'high'
                    })
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                if any(keyword in node.name.lower() for keyword in self.mock_keywords):
                    mock_usages.append({
                        'type': 'class_definition',
                        'name': node.name,
                        'line': node.lineno,
                        'content': lines[node.lineno - 1].strip() if node.lineno <= len(lines) else '',
                        'severity': 'high'
                    })
                self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if any(keyword in node.func.id.lower() for keyword in self.mock_keywords):
                        mock_usages.append({
                            'type': 'function_call',
                            'name': node.func.id,
                            'line': node.lineno,
                            'content': lines[node.lineno - 1].strip() if node.lineno <= len(lines) else '',
                            'severity': 'medium'
                        })
                elif isinstance(node.func, ast.Attribute):
                    if any(keyword in node.func.attr.lower() for keyword in self.mock_keywords):
                        mock_usages.append({
                            'type': 'method_call',
                            'name': node.func.attr,
                            'line': node.lineno,
                            'content': lines[node.lineno - 1].strip() if node.lineno <= len(lines) else '',
                            'severity': 'medium'
                        })
                self.generic_visit(node)

        visitor = MockDataVisitor()
        visitor.visit(tree)

        return mock_usages

    def _analyze_regex_for_mock_data(self, content: str, lines: List[str]) -> List[Dict]:
        """使用正则表达式分析模拟数据"""
        mock_usages = []

        # 检查每行是否包含模拟数据关键词
        for line_num, line in enumerate(lines, 1):
            for keyword in self.mock_keywords:
                if keyword.lower() in line.lower():
                    mock_usages.append({
                        'type': 'keyword_match',
                        'keyword': keyword,
                        'line': line_num,
                        'content': line.strip(),
                        'severity': 'low'
                    })

        return mock_usages

    def generate_removal_plan(self, scan_results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """生成移除计划"""
        removal_plan = {}

        for file_path, mock_usages in scan_results.items():
            file_plan = []

            # 按严重性分组
            high_severity = [usage for usage in mock_usages if usage['severity'] == 'high']
            medium_severity = [usage for usage in mock_usages if usage['severity'] == 'medium']
            low_severity = [usage for usage in mock_usages if usage['severity'] == 'low']

            # 生成移除策略
            if high_severity:
                file_plan.append({
                    'action': 'remove_definitions',
                    'items': high_severity,
                    'description': '移除模拟数据类和函数定义'
                })

            if medium_severity:
                file_plan.append({
                    'action': 'replace_calls',
                    'items': medium_severity,
                    'description': '替换模拟数据调用为真实数据调用'
                })

            if low_severity:
                file_plan.append({
                    'action': 'review_manually',
                    'items': low_severity,
                    'description': '手动审查模拟数据引用'
                })

            if file_plan:
                removal_plan[file_path] = file_plan

        return removal_plan

    def execute_removal_plan(self, removal_plan: Dict[str, List[Dict]], dry_run: bool = True) -> Dict[str, Dict]:
        """执行移除计划"""
        results = {}

        if not dry_run:
            # 创建备份目录
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建备份目录: {self.backup_dir}")

        for file_path, file_plan in removal_plan.items():
            try:
                result = self._process_file_removal(file_path, file_plan, dry_run)
                results[file_path] = result
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                results[file_path] = {'status': 'error', 'message': str(e)}

        return results

    def _process_file_removal(self, file_path: str, file_plan: List[Dict], dry_run: bool) -> Dict:
        """处理单个文件的模拟数据移除"""
        if not dry_run:
            # 备份原文件
            backup_path = self.backup_dir / Path(file_path).name
            shutil.copy2(file_path, backup_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
            lines = original_content.split('\n')

        modified_lines = lines.copy()
        modifications = []

        for plan_item in file_plan:
            action = plan_item['action']
            items = plan_item['items']

            if action == 'remove_definitions':
                for item in items:
                    line_num = item['line'] - 1  # 转换为0索引
                    if line_num < len(modified_lines):
                        # 标记为移除
                        modified_lines[line_num] = f"# REMOVED: {modified_lines[line_num]}"
                        modifications.append(f"行{item['line']}: 移除{item['type']} '{item['name']}'")

            elif action == 'replace_calls':
                for item in items:
                    line_num = item['line'] - 1
                    if line_num < len(modified_lines):
                        original_line = modified_lines[line_num]
                        # 简单替换策略
                        new_line = self._replace_mock_call_with_real_data(original_line, item)
                        modified_lines[line_num] = new_line
                        modifications.append(f"行{item['line']}: 替换调用 '{item['name']}'")

        if not dry_run and modifications:
            # 写入修改后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(modified_lines))

        return {
            'status': 'success' if modifications else 'no_changes',
            'modifications': modifications,
            'dry_run': dry_run
        }

    def _replace_mock_call_with_real_data(self, line: str, item: Dict) -> str:
        """替换模拟数据调用为真实数据调用"""
        # 这里实现具体的替换逻辑
        mock_name = item['name']

        # 常见的替换映射
        replacements = {
            '_create_mock_data': 'self.data_access.get_stock_data',
            'MockDataAccess': 'self.data_access',
            'mock_data': 'real_data',
            'simulate_data': 'self.data_access.get_stock_data',
        }

        new_line = line
        for mock_term, real_term in replacements.items():
            if mock_term in line:
                new_line = line.replace(mock_term, real_term)
                new_line = f"{new_line}  # TODO: 验证真实数据调用参数"
                break

        return new_line

    def generate_report(self, scan_results: Dict[str, List[Dict]],
                       removal_plan: Dict[str, List[Dict]],
                       execution_results: Dict[str, Dict]) -> str:
        """生成移除报告"""
        report_lines = [
            "# 模拟数据移除报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 扫描结果摘要",
            f"- 扫描文件数: {len(scan_results)}",
            f"- 发现模拟数据使用的文件数: {len([f for f, usages in scan_results.items() if usages])}",
            f"- 总计模拟数据使用数: {sum(len(usages) for usages in scan_results.values())}",
            "",
            "## 详细分析",
        ]

        for file_path, mock_usages in scan_results.items():
            if not mock_usages:
                continue

            report_lines.extend([
                f"### {file_path}",
                f"发现 {len(mock_usages)} 处模拟数据使用:",
                ""
            ])

            for usage in mock_usages:
                report_lines.append(
                    f"- 行{usage['line']}: {usage['type']} `{usage.get('name', usage.get('keyword', 'N/A'))}` "
                    f"({usage['severity']})"
                )

            report_lines.append("")

        # 移除计划
        report_lines.extend([
            "## 移除计划",
            f"计划处理 {len(removal_plan)} 个文件:",
            ""
        ])

        for file_path, file_plan in removal_plan.items():
            report_lines.extend([
                f"### {file_path}",
                ""
            ])
            for plan_item in file_plan:
                report_lines.append(f"- {plan_item['action']}: {plan_item['description']} ({len(plan_item['items'])} 项)")
            report_lines.append("")

        # 执行结果
        if execution_results:
            report_lines.extend([
                "## 执行结果",
                ""
            ])

            success_count = len([r for r in execution_results.values() if r['status'] == 'success'])
            report_lines.extend([
                f"- 成功处理: {success_count} 个文件",
                f"- 总计修改: {sum(len(r.get('modifications', [])) for r in execution_results.values())} 处",
                ""
            ])

        return '\n'.join(report_lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='模拟数据移除工具')
    parser.add_argument('--project-root', default='.', help='项目根目录')
    parser.add_argument('--dry-run', action='store_true', help='试运行模式')
    parser.add_argument('--output', help='报告输出文件')

    args = parser.parse_args()

    # 创建工具实例
    tool = MockDataRemovalTool(args.project_root)

    # 扫描模拟数据使用
    logger.info("开始扫描模拟数据使用...")
    scan_results = tool.scan_mock_data_usage()

    # 生成移除计划
    logger.info("生成移除计划...")
    removal_plan = tool.generate_removal_plan(scan_results)

    # 执行移除计划
    logger.info(f"执行移除计划 (dry_run={args.dry_run})...")
    execution_results = tool.execute_removal_plan(removal_plan, dry_run=args.dry_run)

    # 生成报告
    report = tool.generate_report(scan_results, removal_plan, execution_results)

    # 输出报告
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"报告已保存到: {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()