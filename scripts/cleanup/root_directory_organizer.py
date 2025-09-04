#!/usr/bin/env python3
"""
根目录文件整理工具

专门清理和整理根目录下散落的文件，按照项目架构规范重新组织
"""

import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any


class RootDirectoryOrganizer:
    """根目录文件整理器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.backup_dir = self.root_dir / 'archive' / 'root_cleanup_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 文件分类规则
        self.file_categories = {
            'reports': {
                'patterns': [
                    r'.*_report.*\.(json|md|txt)$',
                    r'.*_analysis.*\.(json|md|txt)$',
                    r'.*_summary.*\.(json|md|txt)$',
                    r'.*_completion.*\.(json|md|txt)$',
                    r'.*_optimization.*\.(json|md|txt)$',
                    r'.*_performance.*\.(json|md|txt)$',
                    r'.*_test.*\.(json|md|txt)$',
                    r'.*_validation.*\.(json|md|txt)$'
                ],
                'target_dir': 'reports/generated'
            },
            'test_scripts': {
                'patterns': [
                    r'^test_.*\.py$',
                    r'^debug_.*\.py$',
                    r'^demo_.*\.py$',
                    r'^simple_.*\.py$',
                    r'^comprehensive_.*\.py$'
                ],
                'target_dir': 'tests/scripts'
            },
            'strategy_files': {
                'patterns': [
                    r'.*strategy.*\.py$',
                    r'.*absorb.*\.py$',
                    r'.*macd.*\.py$',
                    r'.*kdj.*\.py$'
                ],
                'target_dir': 'strategy/implementations'
            },
            'fix_scripts': {
                'patterns': [
                    r'^fix_.*\.py$',
                    r'^migrate_.*\.py$',
                    r'^upgrade_.*\.py$',
                    r'^verify_.*\.py$'
                ],
                'target_dir': 'tools/fixes'
            },
            'validation_scripts': {
                'patterns': [
                    r'.*validator.*\.py$',
                    r'.*verification.*\.py$',
                    r'.*validate.*\.py$'
                ],
                'target_dir': 'tools/validation'
            },
            'data_files': {
                'patterns': [
                    r'.*\.json$',
                    r'.*\.csv$',
                    r'.*\.xml$',
                    r'.*\.png$'
                ],
                'target_dir': 'data/generated'
            },
            'temp_results': {
                'patterns': [
                    r'.*_results_.*\.json$',
                    r'.*_20250\d+_\d+\.json$',
                    r'.*_20250\d+_\d+\.txt$',
                    r'.*_20250\d+_\d+\.md$'
                ],
                'target_dir': 'data/temp_results'
            },
            'documentation': {
                'patterns': [
                    r'.*\.md$',
                    r'README.*',
                    r'.*指南.*\.md$',
                    r'.*报告.*\.md$'
                ],
                'target_dir': 'docs/generated'
            }
        }
        
        # 需要保留在根目录的重要文件
        self.keep_in_root = {
            'README.md',
            'requirements.txt',
            'pyproject.toml',
            'pytest.ini',
            'docker-compose.yml',
            '__init__.py',
            '.gitignore',
            'LICENSE',
            'CHANGELOG.md'
        }
    
    def analyze_root_files(self) -> Dict[str, Any]:
        """分析根目录文件"""
        print("🔍 分析根目录文件...")
        
        analysis = {
            'total_files': 0,
            'categorized_files': {},
            'uncategorized_files': [],
            'files_to_keep': [],
            'files_to_move': {},
            'files_to_delete': []
        }
        
        # 获取根目录下的所有文件（不包括子目录）
        root_files = [f for f in self.root_dir.iterdir() if f.is_file()]
        analysis['total_files'] = len(root_files)
        
        for file_path in root_files:
            file_name = file_path.name
            
            # 检查是否需要保留在根目录
            if file_name in self.keep_in_root:
                analysis['files_to_keep'].append(file_name)
                continue
            
            # 尝试分类文件
            categorized = False
            for category, config in self.file_categories.items():
                for pattern in config['patterns']:
                    if re.match(pattern, file_name, re.IGNORECASE):
                        if category not in analysis['categorized_files']:
                            analysis['categorized_files'][category] = []
                        analysis['categorized_files'][category].append(file_name)
                        analysis['files_to_move'][file_name] = config['target_dir']
                        categorized = True
                        break
                if categorized:
                    break
            
            if not categorized:
                # 检查是否为临时文件或可删除文件
                if self._is_deletable_file(file_name):
                    analysis['files_to_delete'].append(file_name)
                else:
                    analysis['uncategorized_files'].append(file_name)
        
        return analysis
    
    def organize_files(self, analysis: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """整理文件"""
        print(f"📁 整理根目录文件 (dry_run={dry_run})...")
        
        results = {
            'moved_files': [],
            'deleted_files': [],
            'created_directories': [],
            'errors': []
        }
        
        if not dry_run:
            # 创建备份目录
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动文件到正确位置
        for file_name, target_dir in analysis['files_to_move'].items():
            try:
                src_path = self.root_dir / file_name
                dst_dir = self.root_dir / target_dir
                dst_path = dst_dir / file_name
                
                if not dry_run:
                    # 创建目标目录
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    if target_dir not in [str(p.relative_to(self.root_dir)) for p in results['created_directories']]:
                        results['created_directories'].append(dst_dir)
                    
                    # 备份原文件
                    backup_path = self.backup_dir / file_name
                    shutil.copy2(src_path, backup_path)
                    
                    # 移动文件
                    shutil.move(str(src_path), str(dst_path))
                
                results['moved_files'].append({
                    'file': file_name,
                    'from': '.',
                    'to': target_dir
                })
                
            except Exception as e:
                results['errors'].append(f"移动 {file_name} 失败: {str(e)}")
        
        # 删除临时文件
        for file_name in analysis['files_to_delete']:
            try:
                src_path = self.root_dir / file_name
                
                if not dry_run:
                    # 备份后删除
                    backup_path = self.backup_dir / file_name
                    shutil.copy2(src_path, backup_path)
                    src_path.unlink()
                
                results['deleted_files'].append(file_name)
                
            except Exception as e:
                results['errors'].append(f"删除 {file_name} 失败: {str(e)}")
        
        return results
    
    def create_directory_structure(self) -> List[str]:
        """创建标准目录结构"""
        print("🏗️ 创建标准目录结构...")
        
        standard_dirs = [
            'reports/generated',
            'tests/scripts',
            'strategy/implementations',
            'tools/fixes',
            'tools/validation',
            'data/generated',
            'data/temp_results',
            'docs/generated',
            'archive/deprecated',
            'archive/temp'
        ]
        
        created_dirs = []
        for dir_path in standard_dirs:
            full_path = self.root_dir / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(dir_path)
        
        return created_dirs
    
    def _is_deletable_file(self, file_name: str) -> bool:
        """判断文件是否可以删除"""
        deletable_patterns = [
            r'.*\.pyc$',
            r'.*\.pyo$',
            r'.*~$',
            r'.*\.tmp$',
            r'.*\.temp$',
            r'\.DS_Store$',
            r'Thumbs\.db$',
            r'.*_backup$',
            r'.*\.bak$'
        ]
        
        for pattern in deletable_patterns:
            if re.match(pattern, file_name, re.IGNORECASE):
                return True
        
        return False
    
    def generate_organization_report(self, analysis: Dict[str, Any], results: Dict[str, Any]) -> str:
        """生成整理报告"""
        report = f"""# 根目录文件整理报告

**执行时间**: {datetime.now().isoformat()}

## 📊 整理统计

- **根目录文件总数**: {analysis['total_files']} 个
- **保留在根目录**: {len(analysis['files_to_keep'])} 个
- **移动到子目录**: {len(results['moved_files'])} 个
- **删除临时文件**: {len(results['deleted_files'])} 个
- **未分类文件**: {len(analysis['uncategorized_files'])} 个

## 📁 文件分类结果

"""
        
        for category, files in analysis['categorized_files'].items():
            target_dir = self.file_categories[category]['target_dir']
            report += f"### {category} ({len(files)} 个文件)\n"
            report += f"**目标目录**: `{target_dir}`\n\n"
            for file_name in files[:5]:  # 只显示前5个
                report += f"- {file_name}\n"
            if len(files) > 5:
                report += f"- ... 还有 {len(files) - 5} 个文件\n"
            report += "\n"
        
        report += "## 🗂️ 保留在根目录的文件\n\n"
        for file_name in analysis['files_to_keep']:
            report += f"- {file_name}\n"
        
        if analysis['uncategorized_files']:
            report += "\n## ❓ 未分类文件\n\n"
            report += "以下文件需要手动处理：\n\n"
            for file_name in analysis['uncategorized_files']:
                report += f"- {file_name}\n"
        
        if results['errors']:
            report += "\n## ❌ 处理错误\n\n"
            for error in results['errors']:
                report += f"- {error}\n"
        
        report += f"\n## 📁 备份位置\n\n"
        report += f"原文件已备份到: `{self.backup_dir}`\n"
        
        return report


def main():
    """主函数"""
    print("🚀 根目录文件整理工具")
    print("=" * 50)
    
    organizer = RootDirectoryOrganizer('.')
    
    # 1. 分析根目录文件
    analysis = organizer.analyze_root_files()
    
    print(f"\n📊 分析结果:")
    print(f"  根目录文件总数: {analysis['total_files']}")
    print(f"  需要移动的文件: {len(analysis['files_to_move'])}")
    print(f"  需要删除的文件: {len(analysis['files_to_delete'])}")
    print(f"  保留在根目录: {len(analysis['files_to_keep'])}")
    print(f"  未分类文件: {len(analysis['uncategorized_files'])}")
    
    if analysis['uncategorized_files']:
        print(f"\n❓ 未分类文件:")
        for file_name in analysis['uncategorized_files'][:10]:
            print(f"    {file_name}")
        if len(analysis['uncategorized_files']) > 10:
            print(f"    ... 还有 {len(analysis['uncategorized_files']) - 10} 个")
    
    print(f"\n📁 文件分类:")
    for category, files in analysis['categorized_files'].items():
        target_dir = organizer.file_categories[category]['target_dir']
        print(f"  {category}: {len(files)} 个文件 -> {target_dir}")
    
    # 2. 询问是否执行整理
    if input("\n是否执行文件整理？(y/N): ").lower() == 'y':
        dry_run = input("是否先进行模拟运行？(Y/n): ").lower() != 'n'
        
        # 3. 创建目录结构
        created_dirs = organizer.create_directory_structure()
        if created_dirs:
            print(f"\n🏗️ 创建了 {len(created_dirs)} 个目录")
        
        # 4. 执行文件整理
        results = organizer.organize_files(analysis, dry_run=dry_run)
        
        print(f"\n📁 整理结果:")
        print(f"  移动文件: {len(results['moved_files'])}")
        print(f"  删除文件: {len(results['deleted_files'])}")
        print(f"  创建目录: {len(results['created_directories'])}")
        print(f"  错误数量: {len(results['errors'])}")
        
        if results['errors']:
            print(f"\n❌ 错误信息:")
            for error in results['errors'][:5]:
                print(f"  {error}")
        
        # 5. 生成报告
        report = organizer.generate_organization_report(analysis, results)
        report_file = f'root_directory_organization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 整理报告已保存到: {report_file}")
        
        if not dry_run:
            print(f"📁 文件备份位置: {organizer.backup_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
