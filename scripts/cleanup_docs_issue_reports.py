#!/usr/bin/env python3
"""
清理docs目录中的问题修复报告和问题总结
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Set
from datetime import datetime
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class DocsIssueReportsCleaner:
    """docs目录问题报告清理器"""
    
    def __init__(self):
        self.docs_root = Path("docs")
        self.backup_dir = Path("archive/docs_cleanup_backup")
        
        # 需要清理的问题报告和总结文件模式
        self.issue_report_patterns = [
            # 问题修复报告
            r".*修复.*报告.*\.md$",
            r".*问题.*修复.*\.md$", 
            r".*fix.*report.*\.md$",
            r".*issue.*fix.*\.md$",
            r".*bug.*fix.*\.md$",
            
            # 问题总结
            r".*问题.*总结.*\.md$",
            r".*issue.*summary.*\.md$",
            r".*问题.*汇总.*\.md$",
            
            # 验证报告（大量重复的验证报告）
            r".*validation_report\.md$",
            r".*验证.*报告.*\.md$",
            r".*测试.*修复.*\.md$",
            
            # 修复相关总结
            r".*修复.*总结.*\.md$",
            r".*修复.*经验.*\.md$",
            r".*修复.*清单.*\.md$",
            
            # 进度报告和状态报告
            r".*进度.*总结.*\.md$",
            r".*进度.*报告.*\.md$",
            r".*状态.*报告.*\.md$",
            r".*项目.*总结.*报告.*\.md$",
            
            # 质量保证报告
            r".*质量.*保证.*报告.*\.md$",
            r".*最终.*报告.*\.md$",
            
            # 特定的问题文件
            r".*修复脚本和验证报告清单.*\.md$",
            r".*技术指标修复项目总结报告.*\.md$",
            r".*技术指标验证工作最终总结报告.*\.md$",
            r".*技术指标验证进度.*\.md$",
            r".*最终质量保证报告.*\.md$",
            r".*最终项目状态报告.*\.md$",
            r".*项目总结报告.*\.md$"
        ]
        
        # 需要保留的重要文档模式（不删除）
        self.keep_patterns = [
            r".*README.*\.md$",
            r".*guide.*\.md$", 
            r".*指南.*\.md$",
            r".*标准.*\.md$",
            r".*规范.*\.md$",
            r".*架构.*\.md$",
            r".*设计.*\.md$",
            r".*需求.*\.md$",
            r".*api.*\.md$",
            r".*reference.*\.md$",
            r".*methodology.*\.md$",
            r".*implementation.*\.md$",
            r".*quick.*start.*\.md$",
            r".*user.*guide.*\.md$"
        ]
        
        # 需要完全清理的目录
        self.cleanup_directories = [
            "docs/finaltesting/indicators"  # 大量重复的验证报告
        ]
    
    def create_backup_dir(self):
        """创建备份目录"""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
            logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def should_keep_file(self, file_path: Path) -> bool:
        """检查文件是否应该保留"""
        filename = file_path.name
        
        # 检查是否匹配保留模式
        for pattern in self.keep_patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return True
        
        return False
    
    def should_cleanup_file(self, file_path: Path) -> bool:
        """检查文件是否应该清理"""
        filename = file_path.name
        
        # 首先检查是否应该保留
        if self.should_keep_file(file_path):
            return False
        
        # 检查是否匹配清理模式
        for pattern in self.issue_report_patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return True
        
        return False
    
    def scan_for_cleanup(self) -> List[Path]:
        """扫描需要清理的文件"""
        cleanup_files = []
        
        logger.info("🔍 扫描docs目录中的问题报告和总结文件...")
        
        # 扫描所有markdown文件
        for md_file in self.docs_root.rglob("*.md"):
            if self.should_cleanup_file(md_file):
                cleanup_files.append(md_file)
        
        # 添加需要完全清理的目录
        for cleanup_dir in self.cleanup_directories:
            cleanup_dir_path = Path(cleanup_dir)
            if cleanup_dir_path.exists():
                for file_path in cleanup_dir_path.rglob("*"):
                    if file_path.is_file():
                        cleanup_files.append(file_path)
        
        return cleanup_files
    
    def move_to_backup(self, file_path: Path) -> bool:
        """移动文件到备份目录"""
        try:
            # 保持相对路径结构
            relative_path = file_path.relative_to(Path("."))
            backup_path = self.backup_dir / relative_path
            
            # 创建父目录
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果备份文件已存在，添加时间戳
            if backup_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = backup_path.name.split('.')
                if len(name_parts) > 1:
                    new_name = f"{'.'.join(name_parts[:-1])}_{timestamp}.{name_parts[-1]}"
                else:
                    new_name = f"{backup_path.name}_{timestamp}"
                backup_path = backup_path.parent / new_name
            
            # 移动文件
            shutil.move(str(file_path), str(backup_path))
            logger.debug(f"📦 移动到备份: {file_path} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动失败 {file_path}: {e}")
            return False
    
    def clean_empty_directories(self):
        """清理空目录"""
        empty_dirs = []
        
        for root, dirs, files in os.walk(self.docs_root, topdown=False):
            root_path = Path(root)
            
            # 跳过根目录
            if root_path == self.docs_root:
                continue
            
            # 检查是否为空目录
            if not files and not dirs:
                empty_dirs.append(root_path)
        
        # 移除空目录
        for empty_dir in empty_dirs:
            try:
                empty_dir.rmdir()
                logger.info(f"🗑️ 删除空目录: {empty_dir}")
            except Exception as e:
                logger.warning(f"⚠️ 删除空目录失败 {empty_dir}: {e}")
    
    def clean_docs(self, dry_run: bool = True):
        """清理docs目录"""
        logger.info(f"🧹 开始清理docs目录问题报告... (dry_run={dry_run})")
        
        cleanup_files = self.scan_for_cleanup()
        
        if not cleanup_files:
            logger.info("✅ 没有发现需要清理的问题报告文件")
            return
        
        logger.info(f"📋 发现 {len(cleanup_files)} 个需要清理的文件:")
        
        # 按目录分组显示
        files_by_dir = {}
        for file_path in cleanup_files:
            dir_name = str(file_path.parent)
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(file_path.name)
        
        for dir_name, files in files_by_dir.items():
            logger.info(f"  📂 {dir_name}: {len(files)} 个文件")
            for file_name in files[:3]:  # 只显示前3个
                logger.info(f"    📄 {file_name}")
            if len(files) > 3:
                logger.info(f"    ... 还有 {len(files) - 3} 个文件")
        
        if dry_run:
            logger.info("🔍 这是预览模式，没有实际删除文件")
            return
        
        # 创建备份目录
        self.create_backup_dir()
        
        # 执行清理
        moved_count = 0
        for file_path in cleanup_files:
            if self.move_to_backup(file_path):
                moved_count += 1
        
        # 清理空目录
        self.clean_empty_directories()
        
        logger.info(f"✅ 清理完成! 移动了 {moved_count} 个文件到备份目录")
        logger.info(f"📁 备份位置: {self.backup_dir}")
    
    def generate_cleanup_report(self, cleanup_files: List[Path]) -> str:
        """生成清理报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"scripts/docs_cleanup_report_{timestamp}.md"
        
        report_content = f"""# docs目录问题报告清理报告
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 清理统计
- **总文件数**: {len(cleanup_files)}
- **备份位置**: {self.backup_dir}

## 📂 清理文件列表
"""
        
        # 按目录分组
        files_by_dir = {}
        for file_path in cleanup_files:
            dir_name = str(file_path.parent)
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(file_path.name)
        
        for dir_name, files in files_by_dir.items():
            report_content += f"\n### {dir_name} ({len(files)} 个文件)\n"
            for file_name in files:
                report_content += f"- 📄 `{file_name}`\n"
        
        report_content += f"""
## 🔄 恢复指导
如需恢复文件，可以从备份目录 `{self.backup_dir}` 中找到对应文件。

### 恢复命令示例:
```bash
# 恢复单个文件
cp {self.backup_dir}/docs/[path]/[filename] docs/[path]/

# 恢复整个目录
cp -r {self.backup_dir}/docs/[directory]/* docs/[directory]/
```

## ⚠️ 注意事项
- 备份文件保留在 `{self.backup_dir}` 目录中
- 建议在确认系统正常运行后再删除备份
- 如有问题，请及时从备份恢复
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 生成清理报告: {report_path}")
        return report_path

def main():
    """主函数"""
    logger.info("🧹 开始清理docs目录问题报告...")
    
    cleaner = DocsIssueReportsCleaner()
    
    # 首先预览
    logger.info("=" * 60)
    logger.info("🔍 预览模式 - 扫描需要清理的问题报告")
    logger.info("=" * 60)
    
    cleanup_files = cleaner.scan_for_cleanup()
    
    # 显示预览
    cleaner.clean_docs(dry_run=True)
    
    # 生成详细报告
    if cleanup_files:
        report_path = cleaner.generate_cleanup_report(cleanup_files)
        
        response = input(f"\n是否执行清理? (将清理 {len(cleanup_files)} 个文件) (y/N): ").strip().lower()
        
        if response == 'y':
            logger.info("=" * 60)
            logger.info("🧹 执行清理")
            logger.info("=" * 60)
            
            cleaner.clean_docs(dry_run=False)
            logger.info(f"📄 详细报告已保存: {report_path}")
        else:
            logger.info("❌ 用户取消清理")
            logger.info(f"📄 预览报告已保存: {report_path}")

if __name__ == "__main__":
    main()
