#!/usr/bin/env python3
"""
配置管理统一修复脚本
解决配置管理入口过多的问题，建立清晰的单一入口原则

修复目标:
1. 明确标准入口: config/unified_config_manager.py
2. 废弃冗余入口: config/__init__.py, config/config.py
3. 保留兼容入口: config/database_config_manager.py (仅用于数据库配置)
4. 统一所有配置访问到标准入口
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Set
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManagementUnifier:
    """配置管理统一器"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_dir = self.project_root / "config"
        self.backup_dir = self.project_root / "backup" / "config_unification_20240916"
        
        # 标准入口定义
        self.standard_entry = "config.unified_config_manager"
        self.standard_functions = ["get_config", "get_config_manager"]
        
        # 要废弃的入口
        self.deprecated_entries = [
            "config.__init__",
            "config.config"
        ]
        
        # 保留的兼容入口
        self.compatible_entries = [
            "config.database_config_manager",  # 仅用于数据库配置
            "config.unified_database_config"   # 数据库专用配置
        ]
        
        # 统计信息
        self.stats = {
            'files_processed': 0,
            'imports_fixed': 0,
            'deprecated_files_moved': 0,
            'errors': []
        }
    
    def run_unification(self):
        """执行配置管理统一"""
        logger.info("🚀 开始配置管理统一修复...")
        
        try:
            # 1. 创建备份目录
            self._create_backup_directory()
            
            # 2. 分析当前配置文件使用情况
            usage_analysis = self._analyze_config_usage()
            
            # 3. 移动废弃的配置文件
            self._move_deprecated_files()
            
            # 4. 修复所有导入语句
            self._fix_import_statements()
            
            # 5. 创建清晰的配置入口文档
            self._create_config_entry_documentation()
            
            # 6. 验证修复结果
            self._validate_fixes()
            
            # 7. 输出统计报告
            self._print_statistics()
            
            logger.info("✅ 配置管理统一修复完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置管理统一修复失败: {e}")
            self.stats['errors'].append(str(e))
            return False
    
    def _create_backup_directory(self):
        """创建备份目录"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def _analyze_config_usage(self) -> Dict[str, List[str]]:
        """分析配置文件使用情况"""
        logger.info("🔍 分析配置文件使用情况...")
        
        usage = {
            'config.__init__': [],
            'config.config': [],
            'config.unified_config_manager': [],
            'config.database_config_manager': []
        }
        
        # 扫描所有Python文件
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # 检查各种导入模式
                for entry in usage.keys():
                    patterns = [
                        f"from {entry} import",
                        f"import {entry}",
                        f"{entry}."
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            usage[entry].append(str(py_file))
                            break
                            
            except Exception as e:
                logger.warning(f"⚠️  无法读取文件 {py_file}: {e}")
        
        # 输出分析结果
        for entry, files in usage.items():
            logger.info(f"📊 {entry}: {len(files)} 个文件使用")
        
        return usage
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        skip_patterns = [
            "__pycache__",
            ".git",
            "backup",
            "archive",
            ".venv",
            "venv"
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _move_deprecated_files(self):
        """移动废弃的配置文件"""
        logger.info("📦 移动废弃的配置文件...")
        
        deprecated_files = [
            self.config_dir / "__init__.py",
            self.config_dir / "config.py"
        ]
        
        for file_path in deprecated_files:
            if file_path.exists():
                backup_path = self.backup_dir / file_path.name
                shutil.copy2(file_path, backup_path)
                logger.info(f"📦 备份文件: {file_path} -> {backup_path}")
                
                # 移动到备份目录
                file_path.unlink()
                logger.info(f"🗑️  移除废弃文件: {file_path}")
                self.stats['deprecated_files_moved'] += 1
    
    def _fix_import_statements(self):
        """修复所有导入语句"""
        logger.info("🔧 修复导入语句...")
        
        # 导入替换规则
        replacement_rules = {
            # 废弃的导入 -> 标准导入
            "from config.unified_config_manager import get_config": "from config.unified_config_manager import get_config",
            "from config.unified_config_manager import get_config": "from config.unified_config_manager import get_config",
            "from config.unified_config_manager import get_config": "from config.unified_config_manager import get_config",
            "from config.unified_config_manager import get_config_manager": "from config.unified_config_manager import get_config_manager",
            "from config.unified_config_manager import get_config": "from config.unified_config_manager import get_config",
            "get_config": "get_config",
            
            # 数据库配置保持不变（已经是正确的）
            # "from config.database_config_manager import" 保持不变
            # "from config.unified_database_config import" 保持不变
        }
        
        # 扫描并修复所有Python文件
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                original_content = content
                
                # 应用替换规则
                for old_import, new_import in replacement_rules.items():
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        logger.info(f"🔧 修复导入: {py_file.name} - {old_import} -> {new_import}")
                        self.stats['imports_fixed'] += 1
                
                # 如果内容有变化，写回文件
                if content != original_content:
                    py_file.write_text(content, encoding='utf-8')
                    self.stats['files_processed'] += 1
                    
            except Exception as e:
                error_msg = f"修复文件 {py_file} 失败: {e}"
                logger.error(f"❌ {error_msg}")
                self.stats['errors'].append(error_msg)
    
    def _create_config_entry_documentation(self):
        """创建清晰的配置入口文档"""
        logger.info("📝 创建配置入口文档...")
        
        doc_content = '''# 配置管理入口说明

## 🎯 标准入口 (推荐使用)

### 主要入口: `config.unified_config_manager`
```python
from config.unified_config_manager import get_config, get_config_manager

# 获取配置值
database_host = get_config('database.host', 'localhost')
app_config = get_config('application')

# 获取配置管理器实例
config_manager = get_config_manager()
config_manager.reload_config()
```

## 🔧 兼容入口 (特定用途)

### 数据库配置: `config.database_config_manager`
```python
from config.database_config_manager import DatabaseConfigManager

# 仅用于数据库配置访问
db_manager = DatabaseConfigManager()
db_config = db_manager.get_config()
```

### 统一数据库配置: `config.unified_database_config`
```python
from config.unified_database_config import get_unified_database_config

# 获取统一数据库配置
db_config = get_unified_database_config()
```

## ❌ 已废弃的入口 (不要使用)

- ~~`config.__init__`~~ - 已移除
- ~~`config.config`~~ - 已移除

## 📋 使用建议

1. **新代码**: 统一使用 `config.unified_config_manager`
2. **数据库配置**: 使用 `config.database_config_manager` 或 `config.unified_database_config`
3. **旧代码**: 逐步迁移到标准入口

## 🔍 配置文件位置

- 主配置: `config/application.yaml`
- 数据库配置: `config/database.yaml`
- 其他配置: `config/` 目录下的相应文件

---
更新时间: 2024-09-16
维护者: AI Assistant
'''
        
        doc_path = self.config_dir / "CONFIG_ENTRY_GUIDE.md"
        doc_path.write_text(doc_content, encoding='utf-8')
        logger.info(f"📝 创建配置入口文档: {doc_path}")
    
    def _validate_fixes(self):
        """验证修复结果"""
        logger.info("✅ 验证修复结果...")
        
        # 检查是否还有废弃的导入
        deprecated_imports = [
            "from config.unified_config_manager import get_config",
            "from config.config import",
            "from config.unified_config_manager import get_config"
        ]
        
        remaining_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for deprecated in deprecated_imports:
                    if deprecated in content:
                        remaining_issues.append(f"{py_file}: {deprecated}")
                        
            except Exception as e:
                logger.warning(f"⚠️  验证文件 {py_file} 失败: {e}")
        
        if remaining_issues:
            logger.warning(f"⚠️  发现 {len(remaining_issues)} 个未修复的导入:")
            for issue in remaining_issues[:10]:  # 只显示前10个
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ 所有废弃导入已修复")
    
    def _print_statistics(self):
        """输出统计报告"""
        logger.info("📊 配置管理统一修复统计报告:")
        logger.info(f"   - 处理文件数: {self.stats['files_processed']}")
        logger.info(f"   - 修复导入数: {self.stats['imports_fixed']}")
        logger.info(f"   - 移动废弃文件数: {self.stats['deprecated_files_moved']}")
        logger.info(f"   - 错误数: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            logger.error("❌ 错误详情:")
            for error in self.stats['errors'][:5]:  # 只显示前5个错误
                logger.error(f"   - {error}")


def main():
    """主函数"""
    print("🚀 配置管理统一修复工具")
    print("=" * 50)
    
    unifier = ConfigManagementUnifier()
    success = unifier.run_unification()
    
    if success:
        print("\n✅ 配置管理统一修复成功完成!")
        print("\n📋 修复成果:")
        print("   1. ✅ 建立了清晰的单一入口原则")
        print("   2. ✅ 移除了冗余的配置入口")
        print("   3. ✅ 保留了必要的兼容入口")
        print("   4. ✅ 修复了所有废弃的导入语句")
        print("   5. ✅ 创建了配置入口使用指南")
        print("\n🎯 标准入口: config.unified_config_manager")
        print("🔧 兼容入口: config.database_config_manager (仅数据库配置)")
        print("📝 使用指南: config/CONFIG_ENTRY_GUIDE.md")
    else:
        print("\n❌ 配置管理统一修复失败!")
        print("请检查错误日志并手动修复问题")
    
    return success


if __name__ == "__main__":
    main()
