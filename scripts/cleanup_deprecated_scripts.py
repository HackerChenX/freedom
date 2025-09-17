#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
弃用脚本清理工具

清理已被统一架构替代的重复和弃用脚本
遵循六层架构规范，确保系统稳定性
"""

import os
import sys
import shutil
from datetime import datetime
from typing import List, Dict, Set
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import getLogger

logger = getLogger(__name__)


class DeprecatedScriptCleaner:
    """弃用脚本清理器"""
    
    def __init__(self):
        """初始化清理器"""
        self.root_dir = Path(root_dir)
        self.backup_dir = self.root_dir / 'archive' / 'deprecated_scripts' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 需要删除的重复执行器文件
        self.deprecated_executors = [
            'strategy/optimized_strategy_executor.py',
            'strategy/memory_aware_strategy_executor.py', 
            'strategy/high_performance_executor.py',
            'strategy/enhanced_strategy_executor.py',
            'strategy/large_scale_memory_optimizer.py',
            'strategy/batch_data_optimizer.py'
        ]
        
        # 需要删除的重复脚本文件
        self.deprecated_scripts = [
            'scripts/test_unified_executor.py',  # 已被comprehensive_system_test.py替代
            'scripts/test_enhanced_validation.py',  # 已被comprehensive_system_test.py替代
            'scripts/production_indicator_tester.py',  # 功能重复
            'scripts/utils/ultimate_compliance_fix.py',  # 临时修复脚本
            'scripts/utils/advanced_quality_refactor.py',  # 临时修复脚本
            'scripts/utils/fix_layer_violations.py',  # 临时修复脚本
            'scripts/utils/final_query_fix.py',  # 临时修复脚本
            'scripts/utils/massive_compliance_fix.py',  # 临时修复脚本
            'scripts/utils/precise_query_fix.py'  # 临时修复脚本
        ]
        
        # 需要删除的备份文件
        self.backup_files_patterns = [
            '*.backup',
            '*.quality_backup',
            '*.sql_migration_backup',
            '*.bak'
        ]
        
        # 需要删除的临时文件
        self.temp_files = [
            'improved_architecture_check.py',
            'improved_architecture_check.py.backup',
            'SCRIPT_NORMALIZATION_PLAN.md'
        ]
        
        # 保护的核心文件（不能删除）
        self.protected_files = {
            'strategy/strategy_executor.py',  # 统一执行器
            'scripts/comprehensive_system_test.py',  # 综合测试
            'scripts/migrate_strategy_configs.py',  # 配置迁移
            'analysis/enhanced_closed_loop_validator.py',  # 增强验证器
            'utils/strategy_validator.py',  # 统一验证器
            'utils/strategy_config_migrator.py'  # 配置迁移器
        }
        
        logger.info("弃用脚本清理器初始化完成")
    
    def cleanup_deprecated_files(self) -> Dict[str, int]:
        """清理弃用文件"""
        logger.info("=" * 60)
        logger.info("开始清理弃用脚本和文件")
        logger.info("=" * 60)
        
        cleanup_stats = {
            'deprecated_executors': 0,
            'deprecated_scripts': 0,
            'backup_files': 0,
            'temp_files': 0,
            'total_cleaned': 0,
            'backup_created': 0
        }
        
        try:
            # 1. 清理重复执行器
            cleanup_stats['deprecated_executors'] = self._cleanup_deprecated_executors()
            
            # 2. 清理重复脚本
            cleanup_stats['deprecated_scripts'] = self._cleanup_deprecated_scripts()
            
            # 3. 清理备份文件
            cleanup_stats['backup_files'] = self._cleanup_backup_files()
            
            # 4. 清理临时文件
            cleanup_stats['temp_files'] = self._cleanup_temp_files()
            
            # 5. 计算总计
            cleanup_stats['total_cleaned'] = (
                cleanup_stats['deprecated_executors'] +
                cleanup_stats['deprecated_scripts'] +
                cleanup_stats['backup_files'] +
                cleanup_stats['temp_files']
            )
            
            # 6. 生成清理报告
            self._generate_cleanup_report(cleanup_stats)
            
            logger.info(f"清理完成，共清理 {cleanup_stats['total_cleaned']} 个文件")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"清理过程失败: {e}")
            return cleanup_stats
    
    def _cleanup_deprecated_executors(self) -> int:
        """清理弃用的执行器"""
        logger.info("清理弃用的策略执行器...")
        cleaned_count = 0
        
        for executor_path in self.deprecated_executors:
            full_path = self.root_dir / executor_path
            if full_path.exists():
                try:
                    # 备份文件
                    backup_path = self.backup_dir / executor_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full_path, backup_path)
                    
                    # 删除原文件
                    full_path.unlink()
                    cleaned_count += 1
                    logger.info(f"已删除弃用执行器: {executor_path}")
                    
                except Exception as e:
                    logger.error(f"删除执行器失败: {executor_path}, 错误: {e}")
        
        return cleaned_count
    
    def _cleanup_deprecated_scripts(self) -> int:
        """清理弃用的脚本"""
        logger.info("清理弃用的脚本文件...")
        cleaned_count = 0
        
        for script_path in self.deprecated_scripts:
            full_path = self.root_dir / script_path
            if full_path.exists():
                try:
                    # 备份文件
                    backup_path = self.backup_dir / script_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full_path, backup_path)
                    
                    # 删除原文件
                    full_path.unlink()
                    cleaned_count += 1
                    logger.info(f"已删除弃用脚本: {script_path}")
                    
                except Exception as e:
                    logger.error(f"删除脚本失败: {script_path}, 错误: {e}")
        
        return cleaned_count
    
    def _cleanup_backup_files(self) -> int:
        """清理备份文件"""
        logger.info("清理备份文件...")
        cleaned_count = 0
        
        for pattern in self.backup_files_patterns:
            for backup_file in self.root_dir.rglob(pattern):
                # 跳过我们自己创建的备份目录
                if str(self.backup_dir) in str(backup_file):
                    continue
                
                # 跳过隐藏目录和特殊目录
                if any(part.startswith('.') for part in backup_file.parts):
                    continue
                if any(part in ['__pycache__', 'venv', 'node_modules'] for part in backup_file.parts):
                    continue
                
                try:
                    # 备份到我们的归档目录
                    relative_path = backup_file.relative_to(self.root_dir)
                    backup_path = self.backup_dir / relative_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, backup_path)
                    
                    # 删除原文件
                    backup_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"已删除备份文件: {relative_path}")
                    
                except Exception as e:
                    logger.warning(f"删除备份文件失败: {backup_file}, 错误: {e}")
        
        return cleaned_count
    
    def _cleanup_temp_files(self) -> int:
        """清理临时文件"""
        logger.info("清理临时文件...")
        cleaned_count = 0
        
        for temp_file in self.temp_files:
            full_path = self.root_dir / temp_file
            if full_path.exists():
                try:
                    # 备份文件
                    backup_path = self.backup_dir / temp_file
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full_path, backup_path)
                    
                    # 删除原文件
                    full_path.unlink()
                    cleaned_count += 1
                    logger.info(f"已删除临时文件: {temp_file}")
                    
                except Exception as e:
                    logger.error(f"删除临时文件失败: {temp_file}, 错误: {e}")
        
        return cleaned_count
    
    def _generate_cleanup_report(self, cleanup_stats: Dict[str, int]):
        """生成清理报告"""
        report_path = self.backup_dir / 'cleanup_report.txt'
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("弃用脚本清理报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"清理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("清理统计:\n")
                f.write(f"- 弃用执行器: {cleanup_stats['deprecated_executors']} 个\n")
                f.write(f"- 弃用脚本: {cleanup_stats['deprecated_scripts']} 个\n")
                f.write(f"- 备份文件: {cleanup_stats['backup_files']} 个\n")
                f.write(f"- 临时文件: {cleanup_stats['temp_files']} 个\n")
                f.write(f"- 总计: {cleanup_stats['total_cleaned']} 个\n\n")
                
                f.write("已删除的弃用执行器:\n")
                for executor in self.deprecated_executors:
                    if (self.root_dir / executor).exists() == False:
                        f.write(f"- {executor}\n")
                
                f.write("\n已删除的弃用脚本:\n")
                for script in self.deprecated_scripts:
                    if (self.root_dir / script).exists() == False:
                        f.write(f"- {script}\n")
                
                f.write("\n保护的核心文件:\n")
                for protected in self.protected_files:
                    f.write(f"- {protected}\n")
                
                f.write(f"\n备份位置: {self.backup_dir}\n")
            
            logger.info(f"清理报告已生成: {report_path}")
            
        except Exception as e:
            logger.error(f"生成清理报告失败: {e}")
    
    def verify_core_files(self) -> bool:
        """验证核心文件完整性"""
        logger.info("验证核心文件完整性...")
        
        missing_files = []
        for core_file in self.protected_files:
            full_path = self.root_dir / core_file
            if not full_path.exists():
                missing_files.append(core_file)
        
        if missing_files:
            logger.error(f"发现缺失的核心文件: {missing_files}")
            return False
        
        logger.info("核心文件完整性验证通过")
        return True
    
    def update_imports(self) -> int:
        """更新导入引用"""
        logger.info("更新导入引用...")
        updated_count = 0
        
        # 导入映射表
        import_mappings = {
            'from strategy.strategy_executor import UnifiedStrategyExecutor as': 'from strategy.strategy_executor import UnifiedStrategyExecutor as',
from db.sql_manager import SQLManager, QueryType
            'from strategy.strategy_executor import UnifiedStrategyExecutor as': 'from strategy.strategy_executor import UnifiedStrategyExecutor as',
from db.sql_manager import SQLManager, QueryType
            'from strategy.strategy_executor import UnifiedStrategyExecutor as': 'from strategy.strategy_executor import UnifiedStrategyExecutor as',
from db.sql_manager import SQLManager, QueryType
            'from strategy.strategy_executor import UnifiedStrategyExecutor as': 'from strategy.strategy_executor import UnifiedStrategyExecutor as',
from db.sql_manager import SQLManager, QueryType
            'UnifiedStrategyExecutor': 'UnifiedStrategyExecutor',
            'UnifiedStrategyExecutor': 'UnifiedStrategyExecutor',
            'UnifiedStrategyExecutor': 'UnifiedStrategyExecutor',
            'UnifiedStrategyExecutor': 'UnifiedStrategyExecutor'
        }
        
        # 查找需要更新的Python文件
        for py_file in self.root_dir.rglob('*.py'):
            # 跳过备份目录和特殊目录
            if str(self.backup_dir) in str(py_file):
                continue
            if any(part.startswith('.') for part in py_file.parts):
                continue
            if any(part in ['__pycache__', 'venv', 'archive'] for part in py_file.parts):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 应用导入映射
                for old_import, new_import in import_mappings.items():
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                
                # 如果内容有变化，写回文件
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    updated_count += 1
                    logger.debug(f"已更新导入引用: {py_file.relative_to(self.root_dir)}")
                
            except Exception as e:
                logger.warning(f"更新导入引用失败: {py_file}, 错误: {e}")
        
        logger.info(f"导入引用更新完成，共更新 {updated_count} 个文件")
        return updated_count


def main():
    """主函数"""
    logger.info("开始弃用脚本清理...")
    
    cleaner = DeprecatedScriptCleaner()
    
    # 1. 验证核心文件完整性
    if not cleaner.verify_core_files():
        logger.error("核心文件完整性验证失败，停止清理")
        return 1
    
    # 2. 执行清理
    cleanup_stats = cleaner.cleanup_deprecated_files()
    
    # 3. 更新导入引用
    updated_imports = cleaner.update_imports()
    
    # 4. 再次验证核心文件
    if not cleaner.verify_core_files():
        logger.error("清理后核心文件完整性验证失败")
        return 1
    
    # 5. 总结
    logger.info("=" * 60)
    logger.info("弃用脚本清理总结")
    logger.info("=" * 60)
    logger.info(f"清理文件总数: {cleanup_stats['total_cleaned']}")
    logger.info(f"更新导入引用: {updated_imports} 个文件")
    logger.info(f"备份位置: {cleaner.backup_dir}")
    logger.info("✅ 弃用脚本清理完成！")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
