#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
项目文件重新组织：按照六层架构规范整理项目文件
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set
import json
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ProjectReorganizer:
    """项目文件重新组织器"""
    
    def __init__(self):
        self.root_path = Path(root_dir)
        self.backup_path = self.root_path / "archive" / f"reorganization_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cleanup_report = []
        
        # 六层架构目录结构
        self.target_structure = {
            # L6: 用户接口层
            'bin': ['bin'],
            'api': ['api'],
            
            # L5: 业务应用层
            'strategy': ['strategy'],
            'analysis': ['analysis'],
            
            # L4: 核心服务层
            'indicators': ['indicators'],
            'formula': ['formula'],
            
            # L3: 数据服务层
            'db': ['db'],
            
            # L2: 存储访问层 (包含在db中)
            
            # L1: 基础设施层
            'utils': ['utils'],
            'config': ['config'],
            'enums': ['enums'],
            
            # 支撑目录
            'docs': ['docs', 'doc'],
            'tests': ['tests', 'validation', 'unit'],
            'scripts': ['scripts', 'tools'],
            'data': ['data'],
            'logs': ['logs'],
            'results': ['results', 'output', 'test_results', 'test_reports', 'validation_results'],
            'examples': ['examples'],
            'models': ['models'],
            'monitoring': ['monitoring'],
            'risk': ['risk'],
            'crawler': ['crawler'],
            'deployment': ['deployment'],
            'integration': ['integration'],
            'performance': ['performance']
        }
    
    def identify_obsolete_files(self) -> Dict[str, List[str]]:
        """识别过时和重复的文件"""
        logger.info("🔍 识别过时和重复的文件...")
        
        obsolete_files = {
            'backup_files': [],
            'duplicate_files': [],
            'test_files': [],
            'temp_files': [],
            'cache_files': [],
            'log_files': [],
            'old_versions': []
        }
        
        # 扫描所有文件
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file():
                file_name = file_path.name
                relative_path = str(file_path.relative_to(self.root_path))
                
                # 备份文件
                if any(keyword in file_name for keyword in ['.backup', '.bak', '_backup', 'backup_']):
                    obsolete_files['backup_files'].append(relative_path)
                
                # 缓存文件
                elif any(keyword in file_name for keyword in ['__pycache__', '.pyc', '.pyo']):
                    obsolete_files['cache_files'].append(relative_path)
                
                # 临时文件
                elif any(keyword in file_name for keyword in ['.tmp', '.temp', '_temp', 'temp_']):
                    obsolete_files['temp_files'].append(relative_path)
                
                # 测试文件（在根目录的）
                elif file_name.startswith('test_') and file_path.parent == self.root_path:
                    obsolete_files['test_files'].append(relative_path)
                
                # 日志文件（在根目录的）
                elif file_name.endswith('.log') and file_path.parent == self.root_path:
                    obsolete_files['log_files'].append(relative_path)
                
                # 旧版本文件
                elif any(keyword in file_name for keyword in ['_old', '_deprecated', '_legacy']):
                    obsolete_files['old_versions'].append(relative_path)
        
        # 报告发现的过时文件
        total_obsolete = sum(len(files) for files in obsolete_files.values())
        logger.info(f"  📊 发现过时文件总数: {total_obsolete}")
        
        for category, files in obsolete_files.items():
            if files:
                logger.info(f"    - {category}: {len(files)}个")
        
        return obsolete_files
    
    def identify_duplicate_functionality(self) -> Dict[str, List[str]]:
        """识别功能重复的文件"""
        logger.info("🔍 识别功能重复的文件...")
        
        duplicate_groups = {
            'data_managers': [],
            'performance_monitors': [],
            'config_managers': [],
            'test_frameworks': [],
            'crawlers': [],
            'validators': []
        }
        
        # 扫描特定模式的重复文件
        patterns = {
            'data_managers': ['data_manager', 'db_manager', 'database_manager'],
            'performance_monitors': ['performance_monitor', 'system_monitor'],
            'config_managers': ['config_manager', 'database_config'],
            'test_frameworks': ['test_', 'validator_', 'validation_'],
            'crawlers': ['crawler', 'spider'],
            'validators': ['validator', 'validation']
        }
        
        for category, keywords in patterns.items():
            found_files = []
            for keyword in keywords:
                for file_path in self.root_path.rglob(f'*{keyword}*.py'):
                    if file_path.is_file():
                        found_files.append(str(file_path.relative_to(self.root_path)))
            
            if len(found_files) > 1:
                duplicate_groups[category] = found_files
                logger.info(f"    - {category}: {len(found_files)}个重复文件")
        
        return duplicate_groups
    
    def create_backup(self, files_to_backup: List[str]):
        """创建备份"""
        logger.info("💾 创建备份...")
        
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in files_to_backup:
            source = self.root_path / file_path
            if source.exists():
                target = self.backup_path / file_path
                target.parent.mkdir(parents=True, exist_ok=True)
                
                if source.is_file():
                    shutil.copy2(source, target)
                elif source.is_dir():
                    shutil.copytree(source, target, dirs_exist_ok=True)
        
        logger.info(f"  ✅ 备份已创建: {self.backup_path}")
    
    def clean_obsolete_files(self, obsolete_files: Dict[str, List[str]]):
        """清理过时文件"""
        logger.info("🧹 清理过时文件...")
        
        cleaned_count = 0
        
        # 安全清理的文件类型
        safe_to_clean = ['backup_files', 'cache_files', 'temp_files']
        
        for category in safe_to_clean:
            files = obsolete_files.get(category, [])
            for file_path in files:
                full_path = self.root_path / file_path
                try:
                    if full_path.exists():
                        if full_path.is_file():
                            full_path.unlink()
                        elif full_path.is_dir():
                            shutil.rmtree(full_path)
                        cleaned_count += 1
                        self.cleanup_report.append(f"已删除 {category}: {file_path}")
                except Exception as e:
                    logger.warning(f"  ⚠️ 删除失败 {file_path}: {e}")
        
        logger.info(f"  ✅ 已清理 {cleaned_count} 个过时文件")
    
    def reorganize_directories(self):
        """重新组织目录结构"""
        logger.info("📁 重新组织目录结构...")
        
        # 创建标准目录结构
        standard_dirs = [
            'bin', 'api',                    # L6
            'strategy', 'analysis',          # L5
            'indicators', 'formula',         # L4
            'db',                           # L3
            'utils', 'config', 'enums',     # L1
            'docs', 'tests', 'scripts',     # 支撑
            'data', 'logs', 'results',
            'examples', 'models', 'monitoring',
            'risk', 'crawler', 'deployment'
        ]
        
        for dir_name in standard_dirs:
            dir_path = self.root_path / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"  📁 创建目录: {dir_name}")
        
        # 合并分散的目录
        merge_operations = [
            (['doc'], 'docs'),
            (['validation', 'unit'], 'tests'),
            (['tools'], 'scripts'),
            (['test_results', 'test_reports', 'validation_results', 'output'], 'results')
        ]
        
        for source_dirs, target_dir in merge_operations:
            target_path = self.root_path / target_dir
            
            for source_dir in source_dirs:
                source_path = self.root_path / source_dir
                if source_path.exists() and source_path.is_dir():
                    # 移动文件到目标目录
                    for item in source_path.iterdir():
                        target_item = target_path / item.name
                        if not target_item.exists():
                            shutil.move(str(item), str(target_item))
                            logger.info(f"  📦 移动: {source_dir}/{item.name} -> {target_dir}/{item.name}")
                    
                    # 删除空的源目录
                    if not any(source_path.iterdir()):
                        source_path.rmdir()
                        logger.info(f"  🗑️ 删除空目录: {source_dir}")
    
    def consolidate_duplicate_files(self, duplicate_groups: Dict[str, List[str]]):
        """整合重复文件"""
        logger.info("🔄 整合重复文件...")
        
        # 对于每个重复组，保留最新或最完整的版本
        for category, files in duplicate_groups.items():
            if len(files) <= 1:
                continue
            
            logger.info(f"  🔍 处理 {category} 重复文件:")
            
            # 简单策略：保留文件名最短的（通常是主要版本）
            files_with_size = []
            for file_path in files:
                full_path = self.root_path / file_path
                if full_path.exists():
                    size = full_path.stat().st_size if full_path.is_file() else 0
                    files_with_size.append((file_path, size, len(file_path)))
            
            if files_with_size:
                # 按文件大小和路径长度排序，保留最大最短的
                files_with_size.sort(key=lambda x: (-x[1], x[2]))
                keep_file = files_with_size[0][0]
                
                logger.info(f"    ✅ 保留: {keep_file}")
                
                # 移动其他文件到archive
                for file_path, _, _ in files_with_size[1:]:
                    source = self.root_path / file_path
                    if source.exists():
                        archive_path = self.backup_path / "duplicates" / file_path
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(source), str(archive_path))
                        logger.info(f"    📦 归档: {file_path}")
                        self.cleanup_report.append(f"归档重复文件: {file_path}")
    
    def generate_cleanup_report(self):
        """生成清理报告"""
        logger.info("📊 生成清理报告...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'backup_location': str(self.backup_path),
            'cleanup_actions': self.cleanup_report,
            'summary': {
                'total_actions': len(self.cleanup_report),
                'backup_created': self.backup_path.exists(),
                'reorganization_completed': True
            }
        }
        
        report_file = self.root_path / 'docs' / 'project_reorganization_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  📄 清理报告已生成: {report_file}")
        
        # 生成Markdown报告
        md_report = self.root_path / 'docs' / 'project_reorganization_report.md'
        with open(md_report, 'w', encoding='utf-8') as f:
            f.write(f"""# 项目文件重新组织报告

## 📊 执行概要

- **执行时间**: {report['timestamp']}
- **备份位置**: {report['backup_location']}
- **总操作数**: {report['summary']['total_actions']}
- **备份状态**: {'✅ 已创建' if report['summary']['backup_created'] else '❌ 未创建'}
- **重组状态**: {'✅ 已完成' if report['summary']['reorganization_completed'] else '❌ 未完成'}

## 🗂️ 目标架构结构

### 六层架构分层
```
L6: 用户接口层 (bin/, api/)
L5: 业务应用层 (strategy/, analysis/)
L4: 核心服务层 (indicators/, formula/)
L3: 数据服务层 (db/)
L2: 存储访问层 (包含在db中)
L1: 基础设施层 (utils/, config/, enums/)
```

### 支撑目录
- docs/ - 文档
- tests/ - 测试文件
- scripts/ - 脚本工具
- data/ - 数据文件
- logs/ - 日志文件
- results/ - 结果输出

## 📋 清理操作详情

""")
            
            for i, action in enumerate(self.cleanup_report, 1):
                f.write(f"{i}. {action}\n")
            
            f.write(f"""
## 🎯 重组效果

通过本次重组，项目结构更加清晰：

1. **架构合规**: 严格按照六层架构组织文件
2. **消除冗余**: 删除过时和重复文件
3. **提高可维护性**: 统一的目录结构
4. **便于开发**: 清晰的文件组织

## 📚 使用建议

1. **新文件创建**: 严格按照架构分层放置文件
2. **定期清理**: 定期运行清理脚本
3. **备份恢复**: 如需恢复可从备份目录获取
4. **持续维护**: 保持项目结构的整洁

---
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
        
        logger.info(f"  📄 Markdown报告已生成: {md_report}")


def main():
    """主函数"""
    try:
        reorganizer = ProjectReorganizer()
        
        logger.info("🚀 开始项目文件重新组织...")
        logger.info("=" * 80)
        
        # 1. 识别过时文件
        obsolete_files = reorganizer.identify_obsolete_files()
        
        # 2. 识别重复文件
        duplicate_groups = reorganizer.identify_duplicate_functionality()
        
        # 3. 创建备份
        all_files_to_backup = []
        for files in obsolete_files.values():
            all_files_to_backup.extend(files)
        for files in duplicate_groups.values():
            all_files_to_backup.extend(files)
        
        if all_files_to_backup:
            reorganizer.create_backup(all_files_to_backup)
        
        # 4. 清理过时文件
        reorganizer.clean_obsolete_files(obsolete_files)
        
        # 5. 重新组织目录结构
        reorganizer.reorganize_directories()
        
        # 6. 整合重复文件
        reorganizer.consolidate_duplicate_files(duplicate_groups)
        
        # 7. 生成报告
        reorganizer.generate_cleanup_report()
        
        logger.info("=" * 80)
        logger.info("🎉 项目文件重新组织完成！")
        logger.info(f"📦 备份位置: {reorganizer.backup_path}")
        logger.info("📄 详细报告: docs/project_reorganization_report.md")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 项目重组过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
