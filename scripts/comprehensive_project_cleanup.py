#!/usr/bin/env python3
"""
综合项目清理器 - 清理所有剩余的过时文件和脚本
避免引起误解，保持项目整洁
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class ComprehensiveProjectCleaner:
    """综合项目清理器"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.backup_dir = Path("archive/comprehensive_cleanup_backup")
        
        # 绝对不能删除的核心目录和文件
        self.protected_paths = {
            # 核心功能目录
            "indicators/", "strategy/", "db/", "utils/", "config/",
            "enums/", "models/", "formula/", "api/", "analysis/",
            "crawler/", "monitoring/", "risk/",
            
            # 重要配置文件
            "README.md", "__init__.py", "requirements.txt", 
            "pyproject.toml", "pytest.ini", "docker-compose.yml",
            ".gitignore", "database.yaml",
            
            # 重要脚本
            "scripts/intelligent_project_cleaner.py",
            "scripts/cleanup_docs_issue_reports.py",
            "scripts/final_docs_cleanup.py",
            "scripts/comprehensive_project_cleanup.py"
        }
        
        # 需要清理的过时脚本（明确指定）
        self.obsolete_scripts = {
            # bin目录中的验证脚本（功能重复）
            "bin/validate_buypoint_strategy.py",
            "bin/validate_indicators.py", 
            "bin/verify_complete_strategy.py",
            "bin/verify_strategy.py",
            "bin/validate_indicators_closed_loop.py",
            "bin/validate_strategy.py",
            
            # scripts/fixes目录（临时修复脚本）
            "scripts/fixes/final_architecture_fix.py",
            "scripts/fixes/fix_method_signatures.py",
            "scripts/fixes/quick_architecture_fix.py",
            "scripts/fixes/fix_architecture_violations.py",
            "scripts/fixes/fix_system_issues.py",
            "scripts/fixes/fix_table_names.py",
            "scripts/fixes/fix_decorator_parameters.py",
            "scripts/fixes/migrate_to_unified_data_manager.py",
            "scripts/fixes/verify_architecture_fixes.py",
            
            # tests中的修复脚本
            "tests/comprehensive/fix_issues.py"
        }
        
        # 需要清理的过时目录
        self.obsolete_directories = {
            "scripts/fixes/",  # 临时修复脚本目录
            "scripts/deprecated_backup/",  # 已备份的废弃脚本
            "scripts/final_cleanup_backup/",  # 最终清理备份
            "backup_obsolete_files/",  # 过时文件备份
            "test_reports/"  # 空的测试报告目录
        }
        
        # 需要清理的文件模式
        self.cleanup_patterns = [
            # 临时文件
            r".*\.tmp$", r".*\.temp$", r".*\.bak$", r".*\.old$",
            r".*\.backup$", r".*~$", r".*\.swp$", r".*\.swo$",
            
            # 测试输出文件
            r"test_output.*", r".*_test_result.*", r".*_debug_.*",
            r"debug_.*\.py$", r"temp_test_.*",
            
            # 重复备份文件
            r".*_backup_\d+.*", r".*_bak_\d+.*", r".*_old_\d+.*",
            r".*\.backup\.\d+$", r".*_copy\d*\.py$",
            
            # 过时的验证脚本
            r"validate_.*_strict\.py$",
            r"validate_.*_optimization_99\.py$",
            r"validate_.*_five_stage_99\.py$",
            r"validate_.*_deep_optimization.*\.py$"
        ]
    
    def create_backup_dir(self):
        """创建备份目录"""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
            logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def is_protected(self, path: Path) -> bool:
        """检查路径是否受保护"""
        path_str = str(path)
        
        # 检查是否是受保护的路径
        for protected in self.protected_paths:
            if path_str.startswith(protected) or protected in path_str:
                return True
        
        # 检查是否是重要的核心文件
        if path.name in ["__init__.py", "README.md", "requirements.txt"]:
            return True
        
        return False
    
    def should_cleanup_file(self, file_path: Path) -> bool:
        """检查文件是否应该清理"""
        path_str = str(file_path)
        filename = file_path.name
        
        # 检查是否是明确指定的过时脚本
        if path_str in self.obsolete_scripts:
            return True
        
        # 检查是否匹配清理模式
        for pattern in self.cleanup_patterns:
            if re.match(pattern, filename):
                return True
        
        return False
    
    def should_cleanup_directory(self, dir_path: Path) -> bool:
        """检查目录是否应该清理"""
        path_str = str(dir_path) + "/"
        
        # 检查是否是明确指定的过时目录
        if path_str in self.obsolete_directories:
            return True
        
        # 检查是否是空目录（除了受保护的）
        if not self.is_protected(dir_path):
            try:
                # 检查目录是否为空（忽略隐藏文件）
                contents = [f for f in dir_path.iterdir() if not f.name.startswith('.')]
                if not contents:
                    return True
            except (OSError, PermissionError):
                pass
        
        return False
    
    def scan_for_cleanup(self) -> Dict[str, List[Path]]:
        """扫描需要清理的文件和目录"""
        cleanup_items = {
            "obsolete_scripts": [],
            "obsolete_directories": [],
            "temp_files": [],
            "backup_files": [],
            "test_outputs": [],
            "empty_dirs": []
        }
        
        logger.info("🔍 扫描项目中的过时文件和目录...")
        
        # 扫描文件
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                # 跳过受保护的文件
                if self.is_protected(file_path):
                    continue
                
                # 检查是否应该清理
                if self.should_cleanup_file(file_path):
                    path_str = str(file_path)
                    
                    if path_str in self.obsolete_scripts:
                        cleanup_items["obsolete_scripts"].append(file_path)
                    elif any(re.match(pattern, file_path.name) for pattern in self.cleanup_patterns):
                        # 根据模式分类
                        if any(keyword in file_path.name for keyword in ["temp", "tmp"]):
                            cleanup_items["temp_files"].append(file_path)
                        elif any(keyword in file_path.name for keyword in ["backup", "bak", "old"]):
                            cleanup_items["backup_files"].append(file_path)
                        elif any(keyword in file_path.name for keyword in ["test", "debug"]):
                            cleanup_items["test_outputs"].append(file_path)
        
        # 扫描目录（从深层开始，避免删除父目录后子目录不存在）
        all_dirs = list(self.project_root.rglob("*"))
        all_dirs = [d for d in all_dirs if d.is_dir()]
        all_dirs.sort(key=lambda x: len(str(x)), reverse=True)
        
        for dir_path in all_dirs:
            # 跳过受保护的目录
            if self.is_protected(dir_path):
                continue
            
            # 检查是否应该清理
            if self.should_cleanup_directory(dir_path):
                path_str = str(dir_path) + "/"
                
                if path_str in self.obsolete_directories:
                    cleanup_items["obsolete_directories"].append(dir_path)
                else:
                    cleanup_items["empty_dirs"].append(dir_path)
        
        return cleanup_items
    
    def move_to_backup(self, path: Path, category: str) -> bool:
        """移动文件或目录到备份"""
        try:
            # 保持相对路径结构
            relative_path = path.relative_to(self.project_root)
            backup_path = self.backup_dir / category / relative_path
            
            # 创建父目录
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果备份已存在，添加时间戳
            if backup_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if path.is_dir():
                    backup_path = backup_path.parent / f"{backup_path.name}_{timestamp}"
                else:
                    name_parts = backup_path.name.split('.')
                    if len(name_parts) > 1:
                        new_name = f"{'.'.join(name_parts[:-1])}_{timestamp}.{name_parts[-1]}"
                    else:
                        new_name = f"{backup_path.name}_{timestamp}"
                    backup_path = backup_path.parent / new_name
            
            # 移动文件或目录
            shutil.move(str(path), str(backup_path))
            logger.debug(f"📦 移动到备份: {path} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动失败 {path}: {e}")
            return False
    
    def clean_project(self, dry_run: bool = True):
        """清理项目"""
        logger.info(f"🧹 开始综合项目清理... (dry_run={dry_run})")
        
        cleanup_items = self.scan_for_cleanup()
        
        # 统计清理项目
        total_items = sum(len(items) for items in cleanup_items.values())
        
        if total_items == 0:
            logger.info("✅ 没有发现需要清理的文件或目录")
            return
        
        logger.info(f"📋 发现 {total_items} 个需要清理的项目:")
        
        for category, items in cleanup_items.items():
            if items:
                logger.info(f"  📂 {category}: {len(items)} 个")
                for item in items[:3]:  # 只显示前3个
                    item_type = "📁" if item.is_dir() else "📄"
                    logger.info(f"    {item_type} {item}")
                if len(items) > 3:
                    logger.info(f"    ... 还有 {len(items) - 3} 个")
        
        if dry_run:
            logger.info("🔍 这是预览模式，没有实际删除文件")
            return cleanup_items
        
        # 创建备份目录
        self.create_backup_dir()
        
        # 执行清理
        moved_count = 0
        for category, items in cleanup_items.items():
            if items:
                logger.info(f"🗂️ 清理 {category} ({len(items)} 个)...")
                for item in items:
                    if self.move_to_backup(item, category):
                        moved_count += 1
        
        logger.info(f"✅ 综合清理完成! 移动了 {moved_count} 个项目到备份目录")
        logger.info(f"📁 备份位置: {self.backup_dir}")
        
        return cleanup_items

def main():
    """主函数"""
    logger.info("🧹 开始综合项目清理...")
    
    cleaner = ComprehensiveProjectCleaner()
    
    # 首先预览
    logger.info("=" * 60)
    logger.info("🔍 预览模式 - 扫描需要清理的过时文件")
    logger.info("=" * 60)
    
    cleanup_items = cleaner.clean_project(dry_run=True)
    
    if cleanup_items and sum(len(items) for items in cleanup_items.values()) > 0:
        total_items = sum(len(items) for items in cleanup_items.values())
        response = input(f"\n是否执行清理? (将清理 {total_items} 个项目) (y/N): ").strip().lower()
        
        if response == 'y':
            logger.info("=" * 60)
            logger.info("🧹 执行清理")
            logger.info("=" * 60)
            
            cleaner.clean_project(dry_run=False)
        else:
            logger.info("❌ 用户取消清理")

if __name__ == "__main__":
    main()
