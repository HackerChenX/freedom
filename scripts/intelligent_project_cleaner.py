#!/usr/bin/env python3
"""
智能项目清理器
扫描并清理整个项目中的过时文件和目录
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Set, Dict, Tuple
from datetime import datetime
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class IntelligentProjectCleaner:
    """智能项目清理器"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.backup_dir = Path("archive/intelligent_cleanup_backup")
        
        # 绝对不能删除的核心目录
        self.protected_directories = {
            ".git", "venv", "__pycache__", ".pytest_cache",
            "indicators", "strategy", "db", "utils", "config", 
            "enums", "models", "formula", "api", "bin",
            "analysis", "crawler", "monitoring", "risk"
        }
        
        # 绝对不能删除的核心文件
        self.protected_files = {
            "README.md", "__init__.py", "requirements.txt", 
            "pyproject.toml", "pytest.ini", "docker-compose.yml",
            ".gitignore", ".env", "config.py", "database.yaml"
        }
        
        # 需要清理的文件模式
        self.cleanup_patterns = {
            # 临时文件
            "temp_files": [
                r".*\.tmp$", r".*\.temp$", r".*\.bak$", r".*\.old$",
                r".*\.backup$", r".*~$", r".*\.swp$", r".*\.swo$"
            ],
            
            # 重复备份文件
            "backup_files": [
                r".*_backup_\d+.*", r".*_bak_\d+.*", r".*_old_\d+.*",
                r".*\.backup\.\d+$", r".*_copy\d*\.py$"
            ],
            
            # 测试输出文件
            "test_outputs": [
                r"test_output.*", r".*_test_result.*", r".*_debug_.*",
                r"debug_.*\.py$", r"temp_test_.*"
            ],
            
            # 日志文件（保留最近的）
            "old_logs": [
                r".*\.log\.\d+$", r".*\.log\.old$"
            ]
        }
        
        # 需要清理的特定目录内容
        self.cleanup_directories = {
            "logs": {
                "keep_recent": 10,  # 保留最近10个日志文件
                "patterns": [r".*\.log$"]
            },
            "output": {
                "keep_recent": 5,
                "patterns": [r".*"]
            },
            "test_reports": {
                "keep_recent": 5,
                "patterns": [r".*\.html$", r".*\.xml$", r".*\.json$"]
            }
        }
        
        # 需要清理的过时脚本（基于您手动删除的模式）
        self.obsolete_script_patterns = [
            r".*_strict\.py$",
            r".*_optimization_99\.py$", 
            r".*_five_stage_99\.py$",
            r".*_deep_optimization.*\.py$",
            r"enhance_.*\.py$",
            r"implement_.*_enhancement\.py$",
            r"precise_.*_check\.py$",
            r"smart_.*_matching\.py$",
            r"correct_.*_analysis\.py$",
            r"quick_validate_.*\.py$",
            r"comprehensive_.*_check\.py$",
            r"project_.*_reorganization\.py$"
        ]
        
        # 需要清理的过时测试文件
        self.obsolete_test_patterns = [
            r".*_comprehensive_diagnosis\.py$",
            r".*_final_architecture_compliant_validation\.py$",
            r".*_production_readiness_validation\.py$",
            r".*_advanced_optimization\.py$",
            r".*_final_optimization\.py$",
            r".*_human_validation\.py$",
            r".*_pattern_detection.*\.py$",
            r".*_pattern_detector\.py$",
            r"large_scale_.*\.py$",
            r"optimized_.*_pattern_detection\.py$"
        ]
    
    def create_backup_dir(self):
        """创建备份目录"""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
            logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def is_protected(self, path: Path) -> bool:
        """检查路径是否受保护"""
        # 检查是否是受保护的目录
        if path.is_dir() and path.name in self.protected_directories:
            return True
        
        # 检查是否是受保护的文件
        if path.is_file() and path.name in self.protected_files:
            return True
        
        # 检查是否在受保护的目录中
        for protected_dir in self.protected_directories:
            if protected_dir in str(path):
                return True
        
        return False
    
    def matches_cleanup_pattern(self, path: Path) -> Tuple[bool, str]:
        """检查文件是否匹配清理模式"""
        filename = path.name
        
        # 检查临时文件模式
        for pattern in self.cleanup_patterns["temp_files"]:
            if re.match(pattern, filename):
                return True, "temp_files"
        
        # 检查备份文件模式
        for pattern in self.cleanup_patterns["backup_files"]:
            if re.match(pattern, filename):
                return True, "backup_files"
        
        # 检查测试输出文件模式
        for pattern in self.cleanup_patterns["test_outputs"]:
            if re.match(pattern, filename):
                return True, "test_outputs"
        
        # 检查过时脚本模式
        if str(path).startswith("scripts/"):
            for pattern in self.obsolete_script_patterns:
                if re.match(pattern, filename):
                    return True, "obsolete_scripts"
        
        # 检查过时测试文件模式
        if str(path).startswith("tests/"):
            for pattern in self.obsolete_test_patterns:
                if re.match(pattern, filename):
                    return True, "obsolete_tests"
        
        return False, ""
    
    def scan_for_cleanup(self) -> Dict[str, List[Path]]:
        """扫描需要清理的文件"""
        cleanup_files = {
            "temp_files": [],
            "backup_files": [],
            "test_outputs": [],
            "obsolete_scripts": [],
            "obsolete_tests": [],
            "old_logs": [],
            "empty_dirs": []
        }
        
        logger.info("🔍 扫描项目文件...")
        
        # 扫描所有文件
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # 跳过受保护的目录
            if self.is_protected(root_path):
                continue
            
            # 检查文件
            for file in files:
                file_path = root_path / file
                
                # 跳过受保护的文件
                if self.is_protected(file_path):
                    continue
                
                # 检查是否匹配清理模式
                should_cleanup, category = self.matches_cleanup_pattern(file_path)
                if should_cleanup:
                    cleanup_files[category].append(file_path)
            
            # 检查空目录
            if not files and not dirs and root_path != self.project_root:
                if not self.is_protected(root_path):
                    cleanup_files["empty_dirs"].append(root_path)
        
        # 处理特定目录的清理
        self._scan_specific_directories(cleanup_files)
        
        return cleanup_files
    
    def _scan_specific_directories(self, cleanup_files: Dict[str, List[Path]]):
        """扫描特定目录的清理需求"""
        for dir_name, config in self.cleanup_directories.items():
            dir_path = self.project_root / dir_name
            
            if not dir_path.exists():
                continue
            
            files = []
            for pattern in config["patterns"]:
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file() and re.match(pattern, file_path.name):
                        files.append(file_path)
            
            # 按修改时间排序，保留最新的文件
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 标记需要清理的旧文件
            if len(files) > config["keep_recent"]:
                old_files = files[config["keep_recent"]:]
                cleanup_files["old_logs"].extend(old_files)
    
    def move_to_backup(self, file_path: Path, category: str) -> bool:
        """移动文件到备份目录"""
        try:
            # 创建分类备份目录
            category_backup_dir = self.backup_dir / category
            category_backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 保持相对路径结构
            relative_path = file_path.relative_to(self.project_root)
            backup_path = category_backup_dir / relative_path
            
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
            if file_path.is_dir():
                shutil.copytree(str(file_path), str(backup_path))
                shutil.rmtree(str(file_path))
            else:
                shutil.move(str(file_path), str(backup_path))
            
            logger.debug(f"📦 移动到备份: {file_path} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 移动失败 {file_path}: {e}")
            return False
    
    def clean_project(self, cleanup_files: Dict[str, List[Path]], dry_run: bool = True):
        """清理项目"""
        logger.info(f"🧹 开始清理项目... (dry_run={dry_run})")

        # 统计清理文件
        total_files = sum(len(files) for files in cleanup_files.values())

        if total_files == 0:
            logger.info("✅ 没有发现需要清理的文件")
            return

        if dry_run:
            logger.info("🔍 这是预览模式，没有实际删除文件")
            return

        # 创建备份目录
        self.create_backup_dir()

        # 执行清理
        moved_count = 0
        failed_count = 0

        for category, files in cleanup_files.items():
            if not files:
                continue

            logger.info(f"🗂️ 清理 {category} ({len(files)} 个文件)...")

            for file_path in files:
                if self.move_to_backup(file_path, category):
                    moved_count += 1
                else:
                    failed_count += 1

        logger.info(f"✅ 清理完成!")
        logger.info(f"  📦 成功移动: {moved_count} 个文件")
        if failed_count > 0:
            logger.warning(f"  ❌ 移动失败: {failed_count} 个文件")
        logger.info(f"  📁 备份位置: {self.backup_dir}")

    def generate_cleanup_report(self, cleanup_files: Dict[str, List[Path]]) -> str:
        """生成清理报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"scripts/cleanup_report_{timestamp}.md"

        total_files = sum(len(files) for files in cleanup_files.values())

        report_content = f"""# 项目清理报告
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 清理统计
- **总文件数**: {total_files}
- **备份位置**: {self.backup_dir}

## 📂 分类详情
"""

        for category, files in cleanup_files.items():
            if files:
                report_content += f"\n### {category} ({len(files)} 个文件)\n"
                for file_path in files:
                    file_type = "📁" if file_path.is_dir() else "📄"
                    report_content += f"- {file_type} `{file_path}`\n"

        report_content += f"""
## 🔄 恢复指导
如需恢复文件，可以从备份目录 `{self.backup_dir}` 中找到对应文件。

### 恢复命令示例:
```bash
# 恢复单个文件
cp {self.backup_dir}/[category]/[relative_path] [original_path]

# 恢复整个分类
cp -r {self.backup_dir}/[category]/* ./
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

    def check_disk_space(self) -> Dict[str, float]:
        """检查磁盘空间"""
        import shutil

        total, used, free = shutil.disk_usage(self.project_root)

        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "usage_percent": (used / total) * 100
        }

    def estimate_cleanup_size(self, cleanup_files: Dict[str, List[Path]]) -> float:
        """估算清理文件的总大小"""
        total_size = 0

        for files in cleanup_files.values():
            for file_path in files:
                try:
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                    elif file_path.is_dir():
                        for sub_file in file_path.rglob("*"):
                            if sub_file.is_file():
                                total_size += sub_file.stat().st_size
                except (OSError, PermissionError):
                    continue

        return total_size / (1024**2)  # 返回MB

def main():
    """主函数"""
    logger.info("🧹 开始智能项目清理...")

    cleaner = IntelligentProjectCleaner()

    # 检查磁盘空间
    disk_info = cleaner.check_disk_space()
    logger.info(f"💾 磁盘空间: {disk_info['free_gb']:.1f}GB 可用 "
                f"({disk_info['usage_percent']:.1f}% 已使用)")

    # 首先预览
    logger.info("=" * 60)
    logger.info("🔍 预览模式 - 扫描需要清理的文件")
    logger.info("=" * 60)

    cleanup_files = cleaner.scan_for_cleanup()

    # 估算清理大小
    cleanup_size_mb = cleaner.estimate_cleanup_size(cleanup_files)
    logger.info(f"📏 预计清理大小: {cleanup_size_mb:.1f} MB")

    # 显示清理预览
    total_files = sum(len(files) for files in cleanup_files.values())

    if total_files == 0:
        logger.info("✅ 没有发现需要清理的文件")
        return

    logger.info(f"📋 发现 {total_files} 个需要清理的文件:")

    for category, files in cleanup_files.items():
        if files:
            logger.info(f"  📂 {category}: {len(files)} 个文件")
            for file_path in files[:3]:  # 只显示前3个
                file_type = "📁" if file_path.is_dir() else "📄"
                logger.info(f"    {file_type} {file_path}")
            if len(files) > 3:
                logger.info(f"    ... 还有 {len(files) - 3} 个文件")

    # 生成详细报告
    report_path = cleaner.generate_cleanup_report(cleanup_files)

    response = input(f"\n是否执行清理? (预计释放 {cleanup_size_mb:.1f}MB 空间) (y/N): ").strip().lower()

    if response == 'y':
        logger.info("=" * 60)
        logger.info("🧹 执行清理")
        logger.info("=" * 60)

        cleaner.clean_project(cleanup_files, dry_run=False)

        # 再次检查磁盘空间
        new_disk_info = cleaner.check_disk_space()
        freed_space = new_disk_info['free_gb'] - disk_info['free_gb']
        logger.info(f"💾 清理后磁盘空间: {new_disk_info['free_gb']:.1f}GB 可用 "
                    f"(释放了 {freed_space:.1f}GB)")

        logger.info(f"📄 详细报告已保存: {report_path}")
    else:
        logger.info("❌ 用户取消清理")
        logger.info(f"📄 预览报告已保存: {report_path}")

if __name__ == "__main__":
    main()
