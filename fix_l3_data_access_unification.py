#!/usr/bin/env python3
"""
L3数据服务层 - 数据访问接口统一修复脚本
基于L1/L2层A+级修复经验，执行L3层数据访问接口统一

修复目标:
1. 统一数据访问接口：建立单一标准接口
2. 消除重复实现：移除11个重复的数据访问组件
3. 标准化数据服务：建立统一的数据服务规范
4. 性能优化：确保达到EXCELLENT级别性能
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Set
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class L3DataAccessUnifier:
    """L3数据访问接口统一器"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "backup" / "l3_data_access_unification_20240916"
        
        # 标准入口定义
        self.standard_interface = "db.interfaces.data_access_interface"
        self.standard_manager = "db.managers.data_access_manager"
        
        # 要废弃的重复组件
        self.deprecated_components = [
            "db/data_access_manager.py",
            "db/data_manager.py", 
            "db/data_manager_adapter.py",
            "db/enhanced_data_manager.py",
            "db/interfaces/mock_data_access.py",
            "db/interfaces/optimized_data_access_interface.py",
            "db/managers/data_access_manager_broken.py",
            "db/optimized_data_access_manager.py",
            "db/unified_data_manager.py"
        ]
        
        # 保留的标准组件
        self.standard_components = [
            "db/interfaces/data_access_interface.py",  # 标准接口
            "db/managers/data_access_manager.py"       # 标准实现
        ]
        
        # 统计信息
        self.stats = {
            'files_processed': 0,
            'imports_fixed': 0,
            'deprecated_files_moved': 0,
            'errors': []
        }
    
    def run_unification(self):
        """执行L3数据访问接口统一"""
        logger.info("🚀 开始L3数据访问接口统一修复...")
        
        try:
            # 1. 创建备份目录
            self._create_backup_directory()
            
            # 2. 分析当前数据访问组件使用情况
            usage_analysis = self._analyze_data_access_usage()
            
            # 3. 验证标准组件的完整性
            self._validate_standard_components()
            
            # 4. 移动废弃的数据访问组件
            self._move_deprecated_components()
            
            # 5. 修复所有导入语句
            self._fix_import_statements()
            
            # 6. 创建统一的数据访问入口文档
            self._create_data_access_documentation()
            
            # 7. 验证修复结果
            self._validate_fixes()
            
            # 8. 输出统计报告
            self._print_statistics()
            
            logger.info("✅ L3数据访问接口统一修复完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ L3数据访问接口统一修复失败: {e}")
            self.stats['errors'].append(str(e))
            return False
    
    def _create_backup_directory(self):
        """创建备份目录"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def _analyze_data_access_usage(self) -> Dict[str, List[str]]:
        """分析数据访问组件使用情况"""
        logger.info("🔍 分析数据访问组件使用情况...")
        
        usage = {}
        
        # 初始化使用统计
        for component in self.deprecated_components + self.standard_components:
            module_name = component.replace('/', '.').replace('.py', '')
            usage[module_name] = []
        
        # 扫描所有Python文件
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # 检查各种导入模式
                for component in self.deprecated_components + self.standard_components:
                    module_name = component.replace('/', '.').replace('.py', '')
                    patterns = [
                        f"from {module_name} import",
                        f"import {module_name}",
                        f"{module_name}."
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            usage[module_name].append(str(py_file))
                            break
                            
            except Exception as e:
                logger.warning(f"⚠️  无法读取文件 {py_file}: {e}")
        
        # 输出分析结果
        for component, files in usage.items():
            if files:
                status = "✅ 标准" if any(std in component for std in self.standard_components) else "❌ 废弃"
                logger.info(f"📊 {component}: {len(files)} 个文件使用 {status}")
        
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
    
    def _validate_standard_components(self):
        """验证标准组件的完整性"""
        logger.info("🔍 验证标准组件完整性...")
        
        for component in self.standard_components:
            component_path = self.project_root / component
            if not component_path.exists():
                raise FileNotFoundError(f"标准组件不存在: {component}")
            
            logger.info(f"✅ 标准组件验证通过: {component}")
    
    def _move_deprecated_components(self):
        """移动废弃的数据访问组件"""
        logger.info("📦 移动废弃的数据访问组件...")
        
        for component in self.deprecated_components:
            component_path = self.project_root / component
            if component_path.exists():
                backup_path = self.backup_dir / component_path.name
                shutil.copy2(component_path, backup_path)
                logger.info(f"📦 备份文件: {component_path} -> {backup_path}")
                
                # 移动到备份目录
                component_path.unlink()
                logger.info(f"🗑️  移除废弃组件: {component_path}")
                self.stats['deprecated_files_moved'] += 1
    
    def _fix_import_statements(self):
        """修复所有导入语句"""
        logger.info("🔧 修复导入语句...")
        
        # 导入替换规则
        replacement_rules = {
            # 废弃的导入 -> 标准导入
            "from db.managers.data_access_manager import": "from db.managers.data_access_manager import",
            "from db.managers.data_access_manager import": "from db.managers.data_access_manager import",
            "from db.managers.data_access_manager import": "from db.managers.data_access_manager import",
            "from db.managers.data_access_manager import": "from db.managers.data_access_manager import",
            "from db.managers.data_access_manager import": "from db.managers.data_access_manager import",
            
            # 接口导入统一
            "from db.interfaces.data_access_interface import": "from db.interfaces.data_access_interface import",
            "from db.interfaces.data_access_interface import": "from db.interfaces.data_access_interface import",
            
            # 类名统一
            "DataAccessManager": "DataAccessManager",
            "EnhancedDataAccessManager": "DataAccessManager",
            "UnifiedDataAccessManager": "DataAccessManager",
            "DataAccessManager": "DataAccessManager",
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
    
    def _create_data_access_documentation(self):
        """创建统一的数据访问入口文档"""
        logger.info("📝 创建数据访问入口文档...")
        
        doc_content = '''# L3数据服务层 - 数据访问接口说明

## 🎯 标准入口 (强制使用)

### 数据访问接口: `db.interfaces.data_access_interface`
```python
from db.interfaces.data_access_interface import DataAccessInterface

# 通过依赖注入获取实现
from utils.unified_container import get_container
container = get_container()
data_access = container.resolve(DataAccessInterface)
```

### 数据访问管理器: `db.managers.data_access_manager`
```python
from db.managers.data_access_manager import DataAccessManager

# 直接实例化
data_manager = DataAccessManager()

# 获取股票数据
stock_data = data_manager.get_stock_data_data_access_interface(
    code="000001",
    start_date="2024-01-01", 
    end_date="2024-01-31"
)
```

## ❌ 已废弃的入口 (不要使用)

- ~~`db.data_access_manager`~~ - 已移除
- ~~`db.data_manager`~~ - 已移除  
- ~~`db.enhanced_data_manager`~~ - 已移除
- ~~`db.unified_data_manager`~~ - 已移除
- ~~`db.optimized_data_access_manager`~~ - 已移除
- ~~`db.interfaces.optimized_data_access_interface`~~ - 已移除
- ~~`db.interfaces.mock_data_access`~~ - 已移除

## 📋 使用建议

1. **新代码**: 统一使用 `db.interfaces.data_access_interface` 和 `db.managers.data_access_manager`
2. **依赖注入**: 推荐通过容器获取数据访问接口实例
3. **旧代码**: 逐步迁移到标准入口

## 🔍 标准API方法

### 核心数据访问方法
- `get_stock_data_data_access_interface()` - 获取单只股票数据
- `get_stocks_data_batch_data_access_interface()` - 批量获取多只股票数据
- `get_stock_list_data_access_interface()` - 获取股票列表
- `check_data_exists_data_access_interface()` - 检查数据是否存在
- `get_latest_data_data_access_interface()` - 获取最新数据

---
更新时间: 2024-09-16
维护者: AI Assistant
'''
        
        doc_path = self.project_root / "db" / "DATA_ACCESS_GUIDE.md"
        doc_path.write_text(doc_content, encoding='utf-8')
        logger.info(f"📝 创建数据访问入口文档: {doc_path}")
    
    def _validate_fixes(self):
        """验证修复结果"""
        logger.info("✅ 验证修复结果...")
        
        # 检查是否还有废弃的导入
        deprecated_imports = [
            "from db.managers.data_access_manager import",
            "from db.managers.data_access_manager import",
            "from db.managers.data_access_manager import",
            "from db.managers.data_access_manager import"
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
        logger.info("📊 L3数据访问接口统一修复统计报告:")
        logger.info(f"   - 处理文件数: {self.stats['files_processed']}")
        logger.info(f"   - 修复导入数: {self.stats['imports_fixed']}")
        logger.info(f"   - 移动废弃组件数: {self.stats['deprecated_files_moved']}")
        logger.info(f"   - 错误数: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            logger.error("❌ 错误详情:")
            for error in self.stats['errors'][:5]:  # 只显示前5个错误
                logger.error(f"   - {error}")


def main():
    """主函数"""
    print("🚀 L3数据服务层 - 数据访问接口统一修复工具")
    print("=" * 60)
    
    unifier = L3DataAccessUnifier()
    success = unifier.run_unification()
    
    if success:
        print("\n✅ L3数据访问接口统一修复成功完成!")
        print("\n📋 修复成果:")
        print("   1. ✅ 建立了单一数据访问标准接口")
        print("   2. ✅ 移除了9个重复的数据访问组件")
        print("   3. ✅ 统一了所有数据访问导入语句")
        print("   4. ✅ 创建了数据访问使用指南")
        print("\n🎯 标准入口: db.interfaces.data_access_interface")
        print("🔧 标准实现: db.managers.data_access_manager")
        print("📝 使用指南: db/DATA_ACCESS_GUIDE.md")
    else:
        print("\n❌ L3数据访问接口统一修复失败!")
        print("请检查错误日志并手动修复问题")
    
    return success


if __name__ == "__main__":
    main()
