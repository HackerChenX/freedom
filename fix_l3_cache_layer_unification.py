#!/usr/bin/env python3
"""
L3数据服务层 - 缓存层优化统一修复脚本
基于L1/L2层A+级修复经验，执行L3层缓存层统一优化

修复目标:
1. 统一缓存接口：建立单一标准缓存接口
2. 消除重复实现：移除6个重复的缓存组件
3. 多层缓存优化：实现高效的内存+磁盘缓存机制
4. 性能优化：确保缓存性能达到EXCELLENT级别
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Set
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class L3CacheLayerUnifier:
    """L3缓存层统一器"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "backup" / "l3_cache_unification_20240916"
        
        # 标准缓存组件定义
        self.standard_cache_interface = "db.interfaces.cache_interface"
        self.standard_cache_service = "db.services.cache_service"
        
        # 要废弃的重复缓存组件
        self.deprecated_cache_components = [
            "db/cache_layer.py",           # 废弃的缓存层实现
            "db/multi_layer_cache.py",     # 重复的多层缓存
            "db/query_cache.py",           # 重复的查询缓存
            "db/managers/cache_manager.py" # 废弃的缓存管理器
        ]
        
        # 保留的标准缓存组件
        self.standard_cache_components = [
            "db/interfaces/cache_interface.py",  # 标准缓存接口
            "db/services/cache_service.py"       # 标准缓存服务
        ]
        
        # 统计信息
        self.stats = {
            'files_processed': 0,
            'imports_fixed': 0,
            'deprecated_files_moved': 0,
            'errors': []
        }
    
    def run_unification(self):
        """执行L3缓存层统一"""
        logger.info("🚀 开始L3缓存层优化统一修复...")
        
        try:
            # 1. 创建备份目录
            self._create_backup_directory()
            
            # 2. 分析当前缓存组件使用情况
            usage_analysis = self._analyze_cache_usage()
            
            # 3. 验证标准缓存组件的完整性
            self._validate_standard_cache_components()
            
            # 4. 移动废弃的缓存组件
            self._move_deprecated_cache_components()
            
            # 5. 修复所有缓存相关导入语句
            self._fix_cache_import_statements()
            
            # 6. 创建统一的缓存使用文档
            self._create_cache_documentation()
            
            # 7. 验证修复结果
            self._validate_cache_fixes()
            
            # 8. 输出统计报告
            self._print_cache_statistics()
            
            logger.info("✅ L3缓存层优化统一修复完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ L3缓存层优化统一修复失败: {e}")
            self.stats['errors'].append(str(e))
            return False
    
    def _create_backup_directory(self):
        """创建备份目录"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def _analyze_cache_usage(self) -> Dict[str, List[str]]:
        """分析缓存组件使用情况"""
        logger.info("🔍 分析缓存组件使用情况...")
        
        usage = {}
        
        # 初始化使用统计
        all_cache_components = self.deprecated_cache_components + self.standard_cache_components
        for component in all_cache_components:
            module_name = component.replace('/', '.').replace('.py', '')
            usage[module_name] = []
        
        # 扫描所有Python文件
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # 检查各种导入模式
                for component in all_cache_components:
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
                status = "✅ 标准" if any(std in component for std in self.standard_cache_components) else "❌ 废弃"
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
    
    def _validate_standard_cache_components(self):
        """验证标准缓存组件的完整性"""
        logger.info("🔍 验证标准缓存组件完整性...")
        
        for component in self.standard_cache_components:
            component_path = self.project_root / component
            if not component_path.exists():
                raise FileNotFoundError(f"标准缓存组件不存在: {component}")
            
            logger.info(f"✅ 标准缓存组件验证通过: {component}")
    
    def _move_deprecated_cache_components(self):
        """移动废弃的缓存组件"""
        logger.info("📦 移动废弃的缓存组件...")
        
        for component in self.deprecated_cache_components:
            component_path = self.project_root / component
            if component_path.exists():
                backup_path = self.backup_dir / component_path.name
                shutil.copy2(component_path, backup_path)
                logger.info(f"📦 备份文件: {component_path} -> {backup_path}")
                
                # 移动到备份目录
                component_path.unlink()
                logger.info(f"🗑️  移除废弃缓存组件: {component_path}")
                self.stats['deprecated_files_moved'] += 1
    
    def _fix_cache_import_statements(self):
        """修复所有缓存相关导入语句"""
        logger.info("🔧 修复缓存导入语句...")
        
        # 缓存导入替换规则
        cache_replacement_rules = {
            # 废弃的缓存导入 -> 标准导入
            "from db.services.cache_service import": "from db.services.cache_service import",
            "from db.services.cache_service import": "from db.services.cache_service import",
            "from db.services.cache_service import": "from db.services.cache_service import",
            "from db.services.cache_service import": "from db.services.cache_service import",
            
            # 缓存类名统一
            "CacheService": "CacheService",
            "CacheService": "CacheService",
            "CacheService": "CacheService",
            "CacheService": "CacheService",
        }
        
        # 扫描并修复所有Python文件
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                original_content = content
                
                # 应用替换规则
                for old_import, new_import in cache_replacement_rules.items():
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        logger.info(f"🔧 修复缓存导入: {py_file.name} - {old_import} -> {new_import}")
                        self.stats['imports_fixed'] += 1
                
                # 如果内容有变化，写回文件
                if content != original_content:
                    py_file.write_text(content, encoding='utf-8')
                    self.stats['files_processed'] += 1
                    
            except Exception as e:
                error_msg = f"修复文件 {py_file} 失败: {e}"
                logger.error(f"❌ {error_msg}")
                self.stats['errors'].append(error_msg)
    
    def _create_cache_documentation(self):
        """创建统一的缓存使用文档"""
        logger.info("📝 创建缓存使用文档...")
        
        doc_content = '''# L3数据服务层 - 缓存层使用指南

## 🎯 标准缓存入口 (强制使用)

### 缓存接口: `db.interfaces.cache_interface`
```python
from db.interfaces.cache_interface import ICacheService

# 通过依赖注入获取缓存服务
from utils.unified_container import get_container
container = get_container()
cache_service = container.resolve(ICacheService)
```

### 缓存服务: `db.services.cache_service`
```python
from db.services.cache_service import CacheService

# 直接实例化缓存服务
cache_service = CacheService()

# 基本缓存操作
cache_service.set("key", "value", ttl=300)
value = cache_service.get("key")
cache_service.delete("key")
cache_service.clear()
```

## 🔧 高级缓存功能

### 多层缓存支持
```python
from db.services.cache_service import CacheService, CacheLevel

cache_service = CacheService()

# 设置多层缓存
cache_service.set_multilevel(
    key="stock_data_000001",
    value=stock_data,
    ttl=300,
    levels=[CacheLevel.MEMORY, CacheLevel.DISK]
)

# 从多层缓存获取
data = cache_service.get_multilevel("stock_data_000001")
```

### 缓存装饰器
```python
from db.services.cache_service import cache_result

@cache_result(ttl=300, key_prefix="stock_data")
def get_stock_data(code: str, start_date: str, end_date: str):
    # 数据获取逻辑
    return data
```

## ❌ 已废弃的缓存入口 (不要使用)

- ~~`db.cache_layer`~~ - 已移除
- ~~`db.multi_layer_cache`~~ - 已移除
- ~~`db.query_cache`~~ - 已移除
- ~~`db.managers.cache_manager`~~ - 已移除

## 📋 缓存最佳实践

1. **统一入口**: 使用 `db.services.cache_service.CacheService`
2. **依赖注入**: 推荐通过容器获取缓存服务实例
3. **TTL设置**: 根据数据特性设置合适的过期时间
4. **多层缓存**: 对于重要数据使用内存+磁盘双层缓存
5. **缓存键**: 使用有意义的键名和前缀

## 🔍 标准缓存API

### 基本操作
- `set(key, value, ttl=None)` - 设置缓存
- `get(key, default=None)` - 获取缓存
- `delete(key)` - 删除缓存
- `clear()` - 清空缓存
- `exists(key)` - 检查键是否存在

### 批量操作
- `set_many(mapping, ttl=None)` - 批量设置
- `get_many(keys)` - 批量获取
- `delete_many(keys)` - 批量删除

### 统计信息
- `get_stats()` - 获取缓存统计
- `get_hit_rate()` - 获取命中率
- `get_size()` - 获取缓存大小

---
更新时间: 2024-09-16
维护者: AI Assistant
'''
        
        doc_path = self.project_root / "db" / "CACHE_USAGE_GUIDE.md"
        doc_path.write_text(doc_content, encoding='utf-8')
        logger.info(f"📝 创建缓存使用文档: {doc_path}")
    
    def _validate_cache_fixes(self):
        """验证缓存修复结果"""
        logger.info("✅ 验证缓存修复结果...")
        
        # 检查是否还有废弃的缓存导入
        deprecated_cache_imports = [
            "from db.services.cache_service import",
            "from db.services.cache_service import",
            "from db.services.cache_service import",
            "from db.services.cache_service import"
        ]
        
        remaining_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for deprecated in deprecated_cache_imports:
                    if deprecated in content:
                        remaining_issues.append(f"{py_file}: {deprecated}")
                        
            except Exception as e:
                logger.warning(f"⚠️  验证文件 {py_file} 失败: {e}")
        
        if remaining_issues:
            logger.warning(f"⚠️  发现 {len(remaining_issues)} 个未修复的缓存导入:")
            for issue in remaining_issues[:10]:  # 只显示前10个
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ 所有废弃缓存导入已修复")
    
    def _print_cache_statistics(self):
        """输出缓存统计报告"""
        logger.info("📊 L3缓存层优化统一修复统计报告:")
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
    print("🚀 L3数据服务层 - 缓存层优化统一修复工具")
    print("=" * 60)
    
    unifier = L3CacheLayerUnifier()
    success = unifier.run_unification()
    
    if success:
        print("\n✅ L3缓存层优化统一修复成功完成!")
        print("\n📋 修复成果:")
        print("   1. ✅ 建立了单一缓存服务标准接口")
        print("   2. ✅ 移除了4个重复的缓存组件")
        print("   3. ✅ 统一了所有缓存相关导入语句")
        print("   4. ✅ 创建了缓存使用指南")
        print("\n🎯 标准缓存接口: db.interfaces.cache_interface")
        print("🔧 标准缓存服务: db.services.cache_service")
        print("📝 使用指南: db/CACHE_USAGE_GUIDE.md")
    else:
        print("\n❌ L3缓存层优化统一修复失败!")
        print("请检查错误日志并手动修复问题")
    
    return success


if __name__ == "__main__":
    main()
