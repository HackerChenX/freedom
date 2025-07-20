#!/usr/bin/env python3
"""
项目全面清理和架构合规检查工具

功能：
1. 识别无用文件（备份文件、临时文件、过时脚本）
2. 检查架构分层违规
3. 生成清理报告
4. 执行安全清理操作
"""

import os
import sys
import json
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import defaultdict


class ComprehensiveCleanupChecker:
    """全面清理和合规检查器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'cleanup_analysis': {},
            'architecture_violations': {},
            'statistics': {},
            'recommendations': []
        }
        
        # 六层架构定义
        self.layers = {
            'L6': {'name': '用户接口层', 'dirs': ['bin/', 'api/']},
            'L5': {'name': '业务应用层', 'dirs': ['strategy/', 'analysis/']},
            'L4': {'name': '核心服务层', 'dirs': ['indicators/', 'formula/']},
            'L3': {'name': '数据服务层', 'dirs': ['db/interfaces/', 'db/managers/', 'db/services/']},
            'L2': {'name': '存储访问层', 'dirs': ['db/']},
            'L1': {'name': '基础设施层', 'dirs': ['utils/', 'config/', 'enums/']}
        }
        
        # 无用文件模式
        self.useless_patterns = {
            'backup_files': ['*.backup', '*.bak', '*.old', '*.orig', '*~'],
            'temp_files': ['*.tmp', '*.temp', '*.swp', '.DS_Store'],
            'cache_files': ['__pycache__', '*.pyc', '*.pyo'],
            'log_files': ['*.log'],
            'test_results': ['*_results_*.json', '*_report_*.json', '*_test_*.json']
        }
    
    def analyze_useless_files(self) -> Dict[str, List[str]]:
        """分析无用文件"""
        print("🔍 分析无用文件...")
        
        useless_files = defaultdict(list)
        
        # 1. 备份文件和临时文件
        for category, patterns in self.useless_patterns.items():
            for pattern in patterns:
                for file_path in self.root_dir.rglob(pattern):
                    if self._should_keep_file(file_path):
                        continue
                    useless_files[category].append(str(file_path.relative_to(self.root_dir)))
        
        # 2. 重复功能脚本
        duplicate_scripts = self._find_duplicate_scripts()
        useless_files['duplicate_scripts'] = duplicate_scripts
        
        # 3. 空目录
        empty_dirs = self._find_empty_directories()
        useless_files['empty_directories'] = empty_dirs
        
        # 4. 过时的测试结果文件
        outdated_results = self._find_outdated_test_results()
        useless_files['outdated_test_results'] = outdated_results
        
        self.report['cleanup_analysis'] = dict(useless_files)
        return dict(useless_files)
    
    def check_architecture_violations(self) -> Dict[str, List[Dict[str, Any]]]:
        """检查架构违规"""
        print("🏗️ 检查架构违规...")
        
        violations = {
            'cross_layer_violations': [],
            'misplaced_files': [],
            'forbidden_imports': []
        }
        
        # 检查跨层依赖
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            file_layer = self._get_file_layer(py_file)
            if not file_layer:
                violations['misplaced_files'].append({
                    'file': str(py_file.relative_to(self.root_dir)),
                    'reason': '文件不在任何架构层中',
                    'suggested_location': self._suggest_file_location(py_file)
                })
                continue
            
            imports = self._extract_imports(py_file)
            for import_stmt in imports:
                import_layer = self._get_import_layer(import_stmt)
                if import_layer and self._is_forbidden_dependency(file_layer, import_layer):
                    violations['cross_layer_violations'].append({
                        'file': str(py_file.relative_to(self.root_dir)),
                        'file_layer': file_layer,
                        'import_statement': import_stmt,
                        'import_layer': import_layer,
                        'violation_type': f'{self.layers[file_layer]["name"]} 不应直接依赖 {self.layers[import_layer]["name"]}'
                    })
        
        self.report['architecture_violations'] = violations
        return violations
    
    def generate_cleanup_report(self) -> Dict[str, Any]:
        """生成清理报告"""
        print("📊 生成清理报告...")
        
        # 统计信息
        total_useless_files = sum(len(files) for files in self.report['cleanup_analysis'].values())
        total_violations = sum(len(violations) for violations in self.report['architecture_violations'].values())
        
        self.report['statistics'] = {
            'total_useless_files': total_useless_files,
            'total_architecture_violations': total_violations,
            'disk_space_to_free': self._calculate_disk_space(),
            'files_by_category': {
                category: len(files) 
                for category, files in self.report['cleanup_analysis'].items()
            }
        }
        
        # 生成建议
        self._generate_recommendations()
        
        return self.report
    
    def execute_safe_cleanup(self, dry_run: bool = True) -> Dict[str, Any]:
        """执行安全清理操作"""
        print(f"🧹 执行清理操作 (dry_run={dry_run})...")
        
        cleanup_results = {
            'deleted_files': [],
            'moved_files': [],
            'errors': [],
            'disk_space_freed': 0
        }
        
        if not dry_run:
            # 创建备份目录
            backup_dir = self.root_dir / 'archive' / 'cleanup_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 删除无用文件
        for category, files in self.report['cleanup_analysis'].items():
            if category in ['backup_files', 'temp_files', 'cache_files', 'outdated_test_results']:
                for file_path in files:
                    full_path = self.root_dir / file_path
                    try:
                        if full_path.exists():
                            if not dry_run:
                                if full_path.is_file():
                                    cleanup_results['disk_space_freed'] += full_path.stat().st_size
                                    full_path.unlink()
                                elif full_path.is_dir():
                                    shutil.rmtree(full_path)
                            cleanup_results['deleted_files'].append(file_path)
                    except Exception as e:
                        cleanup_results['errors'].append(f"删除 {file_path} 失败: {str(e)}")
        
        # 移动违规文件到正确位置
        for violation in self.report['architecture_violations']['misplaced_files']:
            file_path = violation['file']
            suggested_location = violation.get('suggested_location')
            if suggested_location:
                try:
                    if not dry_run:
                        src = self.root_dir / file_path
                        dst = self.root_dir / suggested_location
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), str(dst))
                    cleanup_results['moved_files'].append({
                        'from': file_path,
                        'to': suggested_location
                    })
                except Exception as e:
                    cleanup_results['errors'].append(f"移动 {file_path} 失败: {str(e)}")
        
        return cleanup_results
    
    def _should_keep_file(self, file_path: Path) -> bool:
        """判断是否应该保留文件"""
        # 保留重要的备份文件
        if 'final_migration_backup' in str(file_path):
            return True
        if 'singleton_fix_backup' in str(file_path):
            return True
        
        # 保留 Docker 和 ClickHouse 相关文件
        if 'docker/' in str(file_path) or 'clickhouse/' in str(file_path):
            return True
        
        # 保留 venv 目录
        if 'venv/' in str(file_path):
            return True
        
        return False
    
    def _find_duplicate_scripts(self) -> List[str]:
        """查找重复功能的脚本"""
        duplicate_scripts = []
        
        # 查找明显的重复脚本
        script_patterns = [
            ('test_', 'test_.*_fix\\.py$'),
            ('debug_', 'debug_.*\\.py$'),
            ('fix_', 'fix_.*\\.py$'),
            ('run_', 'run_.*_test\\.py$')
        ]
        
        for prefix, pattern in script_patterns:
            matching_files = []
            for py_file in self.root_dir.rglob("*.py"):
                if re.search(pattern, py_file.name):
                    matching_files.append(py_file)
            
            if len(matching_files) > 3:  # 如果同类脚本超过3个，可能有重复
                for file_path in matching_files[3:]:  # 保留前3个，其余标记为重复
                    duplicate_scripts.append(str(file_path.relative_to(self.root_dir)))
        
        return duplicate_scripts
    
    def _find_empty_directories(self) -> List[str]:
        """查找空目录"""
        empty_dirs = []
        for dir_path in self.root_dir.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                # 排除一些应该保留的空目录
                if not any(keep in str(dir_path) for keep in ['__pycache__', '.git', 'venv', 'docker']):
                    empty_dirs.append(str(dir_path.relative_to(self.root_dir)))
        return empty_dirs
    
    def _find_outdated_test_results(self) -> List[str]:
        """查找过时的测试结果文件"""
        outdated_files = []
        cutoff_date = datetime.now().timestamp() - (30 * 24 * 3600)  # 30天前
        
        for file_path in self.root_dir.rglob("*"):
            if file_path.is_file():
                # 检查文件名模式
                if any(pattern in file_path.name for pattern in ['_results_', '_report_', '_test_']):
                    if file_path.suffix in ['.json', '.txt', '.md']:
                        try:
                            if file_path.stat().st_mtime < cutoff_date:
                                outdated_files.append(str(file_path.relative_to(self.root_dir)))
                        except OSError:
                            pass
        
        return outdated_files

    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件检查"""
        skip_patterns = [
            '__pycache__', '.git', 'venv', 'docker', 'archive',
            '.pytest_cache', '.coverage', 'node_modules'
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _get_file_layer(self, file_path: Path) -> str:
        """获取文件所属的架构层"""
        relative_path = str(file_path.relative_to(self.root_dir))

        for layer, info in self.layers.items():
            for dir_pattern in info['dirs']:
                if relative_path.startswith(dir_pattern):
                    return layer
        return None

    def _get_import_layer(self, import_stmt: str) -> str:
        """获取导入语句所属的架构层"""
        for layer, info in self.layers.items():
            for dir_pattern in info['dirs']:
                if import_stmt.startswith(dir_pattern.rstrip('/')):
                    return layer
        return None

    def _is_forbidden_dependency(self, file_layer: str, import_layer: str) -> bool:
        """检查是否为禁止的依赖关系"""
        layer_order = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        file_level = layer_order.index(file_layer)
        import_level = layer_order.index(import_layer)

        # 上层不能依赖下层（跨层依赖）
        return file_level > import_level and (file_level - import_level) > 1

    def _extract_imports(self, file_path: Path) -> List[str]:
        """提取文件中的导入语句"""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('from ') and ' import ' in line:
                        module = line.split('from ')[1].split(' import ')[0].strip()
                        imports.append(module)
                    elif line.startswith('import '):
                        module = line.split('import ')[1].split(' as ')[0].split('.')[0].strip()
                        imports.append(module)
        except (UnicodeDecodeError, FileNotFoundError):
            pass
        return imports

    def _suggest_file_location(self, file_path: Path) -> str:
        """建议文件的正确位置"""
        file_name = file_path.name.lower()

        # 根据文件名模式建议位置
        if 'test' in file_name:
            return f"tests/{file_path.name}"
        elif 'config' in file_name:
            return f"config/{file_path.name}"
        elif 'util' in file_name or 'helper' in file_name:
            return f"utils/{file_path.name}"
        elif 'indicator' in file_name:
            return f"indicators/{file_path.name}"
        elif 'strategy' in file_name:
            return f"strategy/{file_path.name}"
        elif 'analysis' in file_name or 'analyzer' in file_name:
            return f"analysis/{file_path.name}"
        elif file_name.startswith('run_') or file_name.startswith('main'):
            return f"bin/{file_path.name}"
        else:
            return f"utils/{file_path.name}"

    def _calculate_disk_space(self) -> int:
        """计算可释放的磁盘空间"""
        total_size = 0
        for category, files in self.report['cleanup_analysis'].items():
            if category in ['backup_files', 'temp_files', 'cache_files', 'outdated_test_results']:
                for file_path in files:
                    full_path = self.root_dir / file_path
                    try:
                        if full_path.exists() and full_path.is_file():
                            total_size += full_path.stat().st_size
                    except OSError:
                        pass
        return total_size

    def _generate_recommendations(self):
        """生成清理和优化建议"""
        recommendations = []

        # 文件清理建议
        total_useless = self.report['statistics']['total_useless_files']
        if total_useless > 0:
            recommendations.append({
                'type': 'cleanup',
                'priority': 'high',
                'title': f'清理 {total_useless} 个无用文件',
                'description': f'可释放约 {self.report["statistics"]["disk_space_to_free"] / 1024 / 1024:.1f} MB 磁盘空间',
                'action': '运行清理脚本删除备份文件、临时文件和过时的测试结果'
            })

        # 架构违规建议
        violations = self.report['architecture_violations']
        if violations['cross_layer_violations']:
            recommendations.append({
                'type': 'architecture',
                'priority': 'critical',
                'title': f'修复 {len(violations["cross_layer_violations"])} 个跨层依赖违规',
                'description': '业务层直接依赖基础设施层，违反了分层架构原则',
                'action': '使用依赖注入模式，通过服务接口访问底层功能'
            })

        if violations['misplaced_files']:
            recommendations.append({
                'type': 'architecture',
                'priority': 'medium',
                'title': f'重新组织 {len(violations["misplaced_files"])} 个错位文件',
                'description': '文件未按照架构分层正确放置',
                'action': '将文件移动到建议的目录位置'
            })

        # 代码重复建议
        duplicate_scripts = self.report['cleanup_analysis'].get('duplicate_scripts', [])
        if duplicate_scripts:
            recommendations.append({
                'type': 'refactoring',
                'priority': 'medium',
                'title': f'合并 {len(duplicate_scripts)} 个重复脚本',
                'description': '存在功能重复的脚本文件',
                'action': '分析脚本功能，合并重复代码，保留最优实现'
            })

        self.report['recommendations'] = recommendations


def main():
    """主函数"""
    print("🚀 项目全面清理和架构合规检查工具")
    print("=" * 60)

    root_dir = os.getcwd()
    checker = ComprehensiveCleanupChecker(root_dir)

    # 1. 分析无用文件
    useless_files = checker.analyze_useless_files()

    # 2. 检查架构违规
    violations = checker.check_architecture_violations()

    # 3. 生成报告
    report = checker.generate_cleanup_report()

    # 4. 输出统计信息
    print(f"\n📊 清理分析结果:")
    print(f"  无用文件总数: {report['statistics']['total_useless_files']}")
    print(f"  架构违规总数: {report['statistics']['total_architecture_violations']}")
    print(f"  可释放空间: {report['statistics']['disk_space_to_free'] / 1024 / 1024:.1f} MB")

    print(f"\n📋 文件分类统计:")
    for category, count in report['statistics']['files_by_category'].items():
        if count > 0:
            print(f"  {category}: {count} 个文件")

    print(f"\n💡 优化建议:")
    for i, rec in enumerate(report['recommendations'], 1):
        priority_icon = "🔴" if rec['priority'] == 'critical' else "🟡" if rec['priority'] == 'high' else "🟢"
        print(f"  {i}. {priority_icon} {rec['title']}")
        print(f"     {rec['description']}")
        print(f"     建议: {rec['action']}")
        print()

    # 5. 保存详细报告
    report_file = f'comprehensive_cleanup_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 详细报告已保存到: {report_file}")

    # 6. 询问是否执行清理
    if input("\n是否执行清理操作？(y/N): ").lower() == 'y':
        dry_run = input("是否先进行模拟运行？(Y/n): ").lower() != 'n'
        cleanup_results = checker.execute_safe_cleanup(dry_run=dry_run)

        print(f"\n🧹 清理结果:")
        print(f"  删除文件: {len(cleanup_results['deleted_files'])}")
        print(f"  移动文件: {len(cleanup_results['moved_files'])}")
        print(f"  错误数量: {len(cleanup_results['errors'])}")
        if not dry_run:
            print(f"  释放空间: {cleanup_results['disk_space_freed'] / 1024 / 1024:.1f} MB")

        if cleanup_results['errors']:
            print(f"\n❌ 清理错误:")
            for error in cleanup_results['errors'][:5]:  # 只显示前5个错误
                print(f"  {error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
