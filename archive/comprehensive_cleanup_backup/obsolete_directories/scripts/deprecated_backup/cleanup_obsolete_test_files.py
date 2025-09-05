#!/usr/bin/env python3
"""
清理过时的测试脚本和文档文件
删除冗余和过时的文件，保留核心测试功能
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Set
import json
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class ObsoleteTestFileCleaner:
    """过时测试文件清理器"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.backup_dir = self.root_path / "backup_obsolete_files"
        self.cleanup_report = {
            'timestamp': datetime.now().isoformat(),
            'deleted_files': [],
            'deleted_directories': [],
            'preserved_files': [],
            'errors': []
        }
        
    def get_obsolete_patterns(self) -> List[str]:
        """获取过时文件模式"""
        return [
            # 过时的验证器文件
            "*_adjusted_validator.py",
            "*_fixed_validator.py", 
            "*_quick_validator.py",
            "*_5stage_validation.py",
            "*_fixed_revalidator.py",
            "*_standardized_validator.py",
            "*_mock_validator.py",
            
            # 过时的调试文件
            "debug_*.py",
            "trace_*.py",
            "verify_*.py",
            
            # 过时的修复文件
            "fix_*.py",
            "repair_*.py",
            "*_fix.py",
            "*_repair.py",
            
            # 过时的阶段测试文件
            "*_stage*.py",
            "phase*_*.py",
            
            # 过时的结果目录
            "*_results/",
            "*_validation_results/",
            "corrected_results/",
            "fixed_pattern_results/",
            "optimized_pattern_results/",
            "professional_fix_results/",
            "final_fixed_results/",
            "pattern_detection_results/",
            
            # 过时的报告文件
            "*.md",
            "*.html",
            "*.csv",
            "*.json",
            
            # 临时和缓存文件
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            ".pytest_cache/",
            
            # 过时的特定文件
            "human_validation_system.py",
            "individual_*_repair.py",
            "*_algorithm_*.py",
            "*_pattern_debug.py",
            "*_production_diagnosis.py",
        ]
    
    def get_preserved_patterns(self) -> List[str]:
        """获取需要保留的文件模式"""
        return [
            # 核心测试框架
            "tests/test_indicators/test_*.py",
            "tests/unit/test_*.py",
            "tests/comprehensive/",
            "tests/unified_indicator_testing/comprehensive_indicator_test.py",
            "tests/unified_indicator_testing/run_unified_tests.py",
            "tests/unified_indicator_testing/config.yaml",
            
            # 重要的集成测试
            "tests/integration/system_integration_test.py",
            "tests/run_updated_comprehensive_tests.py",
            
            # 核心配置和工具
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/helper/",
            "tests/framework/",
            
            # 重要的脚本
            "scripts/final_system_verification.py",
            "scripts/comprehensive_indicator_tester.py",
            
            # 文档（选择性保留）
            "tests/README*.md",
            "tests/unified_indicator_testing/README.md",
        ]
    
    def should_preserve_file(self, file_path: Path) -> bool:
        """判断文件是否应该保留"""
        file_str = str(file_path.relative_to(self.root_path))
        
        # 检查保留模式
        preserved_patterns = self.get_preserved_patterns()
        for pattern in preserved_patterns:
            if file_path.match(pattern) or file_str.startswith(pattern.rstrip('*')):
                return True
        
        # 特殊保留规则
        if file_path.name in ['__init__.py', 'conftest.py']:
            return True
            
        if 'comprehensive_indicator_test.py' in file_path.name:
            return True
            
        if 'run_unified_tests.py' in file_path.name:
            return True
            
        return False
    
    def should_delete_file(self, file_path: Path) -> bool:
        """判断文件是否应该删除"""
        if self.should_preserve_file(file_path):
            return False
            
        file_str = str(file_path.relative_to(self.root_path))
        obsolete_patterns = self.get_obsolete_patterns()
        
        for pattern in obsolete_patterns:
            if file_path.match(pattern) or pattern.rstrip('/') in file_str:
                return True
        
        # 特殊删除规则
        if any(keyword in file_path.name.lower() for keyword in [
            'debug', 'fix', 'repair', 'stage', 'phase', 'validator', 
            'revalidator', 'adjusted', 'obsolete', 'deprecated'
        ]):
            return True
            
        return False
    
    def create_backup(self, file_path: Path):
        """创建文件备份"""
        try:
            relative_path = file_path.relative_to(self.root_path)
            backup_path = self.backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.is_file():
                shutil.copy2(file_path, backup_path)
            elif file_path.is_dir():
                shutil.copytree(file_path, backup_path, dirs_exist_ok=True)
                
        except Exception as e:
            logger.warning(f"备份失败 {file_path}: {e}")
    
    def clean_obsolete_files(self, dry_run: bool = True) -> dict:
        """清理过时文件"""
        logger.info(f"开始清理过时测试文件 (dry_run={dry_run})...")
        
        if not dry_run:
            self.backup_dir.mkdir(exist_ok=True)
        
        # 遍历tests目录
        tests_dir = self.root_path / "tests"
        if not tests_dir.exists():
            logger.error("tests目录不存在")
            return self.cleanup_report
        
        files_to_delete = []
        dirs_to_delete = []
        
        # 收集要删除的文件
        for item in tests_dir.rglob("*"):
            if self.should_delete_file(item):
                if item.is_file():
                    files_to_delete.append(item)
                elif item.is_dir() and not any(self.should_preserve_file(child) for child in item.rglob("*")):
                    dirs_to_delete.append(item)
            elif self.should_preserve_file(item):
                self.cleanup_report['preserved_files'].append(str(item.relative_to(self.root_path)))
        
        # 删除文件
        for file_path in files_to_delete:
            try:
                if not dry_run:
                    self.create_backup(file_path)
                    file_path.unlink()
                    
                self.cleanup_report['deleted_files'].append(str(file_path.relative_to(self.root_path)))
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}删除文件: {file_path.relative_to(self.root_path)}")
                
            except Exception as e:
                error_msg = f"删除文件失败 {file_path}: {e}"
                self.cleanup_report['errors'].append(error_msg)
                logger.error(error_msg)
        
        # 删除空目录
        for dir_path in sorted(dirs_to_delete, key=lambda x: len(str(x)), reverse=True):
            try:
                if not dry_run:
                    if dir_path.exists() and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        
                self.cleanup_report['deleted_directories'].append(str(dir_path.relative_to(self.root_path)))
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}删除目录: {dir_path.relative_to(self.root_path)}")
                
            except Exception as e:
                error_msg = f"删除目录失败 {dir_path}: {e}"
                self.cleanup_report['errors'].append(error_msg)
                logger.error(error_msg)
        
        return self.cleanup_report
    
    def generate_report(self) -> str:
        """生成清理报告"""
        report = f"""
# 过时测试文件清理报告

**清理时间**: {self.cleanup_report['timestamp']}

## 📊 清理统计

- **删除文件数**: {len(self.cleanup_report['deleted_files'])}
- **删除目录数**: {len(self.cleanup_report['deleted_directories'])}
- **保留文件数**: {len(self.cleanup_report['preserved_files'])}
- **错误数量**: {len(self.cleanup_report['errors'])}

## 🗑️ 删除的文件 (前20个)

"""
        for file_path in self.cleanup_report['deleted_files'][:20]:
            report += f"- {file_path}\n"
        
        if len(self.cleanup_report['deleted_files']) > 20:
            report += f"- ... 还有 {len(self.cleanup_report['deleted_files']) - 20} 个文件\n"
        
        report += f"""
## 📁 删除的目录

"""
        for dir_path in self.cleanup_report['deleted_directories']:
            report += f"- {dir_path}\n"
        
        if self.cleanup_report['errors']:
            report += f"""
## ❌ 错误信息

"""
            for error in self.cleanup_report['errors']:
                report += f"- {error}\n"
        
        return report
    
    def save_report(self) -> str:
        """保存清理报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.root_path / f"cleanup_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        
        # 保存JSON格式的详细报告
        json_file = self.root_path / f"cleanup_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.cleanup_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"清理报告已保存: {report_file}")
        return str(report_file)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="清理过时的测试文件")
    parser.add_argument('--dry-run', action='store_true', help='预览模式，不实际删除文件')
    parser.add_argument('--force', action='store_true', help='强制执行清理')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.force:
        print("⚠️  这将删除大量过时的测试文件！")
        print("   建议先运行 --dry-run 预览要删除的文件")
        print("   确认无误后使用 --force 执行实际清理")
        return 1
    
    cleaner = ObsoleteTestFileCleaner(root_dir)
    
    print(f"🧹 开始清理过时测试文件...")
    print(f"   模式: {'预览模式' if args.dry_run else '实际清理'}")
    
    # 执行清理
    report = cleaner.clean_obsolete_files(dry_run=args.dry_run)
    
    # 生成报告
    report_file = cleaner.save_report()
    
    # 显示摘要
    print(f"\n📊 清理完成:")
    print(f"  删除文件: {len(report['deleted_files'])}")
    print(f"  删除目录: {len(report['deleted_directories'])}")
    print(f"  保留文件: {len(report['preserved_files'])}")
    print(f"  错误数量: {len(report['errors'])}")
    print(f"  报告文件: {report_file}")
    
    if args.dry_run:
        print(f"\n💡 这是预览模式，没有实际删除文件")
        print(f"   确认无误后请使用 --force 参数执行实际清理")
    
    return 0 if len(report['errors']) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
