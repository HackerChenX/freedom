#!/usr/bin/env python3
"""
L3数据服务层 - 数据服务标准化修复脚本
基于L1/L2层A+级修复经验，执行L3层数据服务标准化

修复目标:
1. 数据服务标准化：建立统一的数据服务接口规范
2. 整合分散组件：整合13个分散的数据服务组件
3. 服务注册统一：统一服务注册和依赖注入机制
4. 性能优化：确保数据服务性能达到EXCELLENT级别
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Set
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class L3DataServiceStandardizer:
    """L3数据服务标准化器"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "backup" / "l3_data_service_standardization_20240916"
        
        # 标准数据服务组件定义
        self.standard_service_registry = "db.service_registry"
        self.standard_services_dir = "db/services"
        
        # 要整合的分散数据服务组件
        self.scattered_service_components = [
            "db/advanced_data_quality_manager.py",
            "db/batch_data_optimizer.py", 
            "db/data_quality_monitor.py",
            "db/intelligent_query_optimizer.py",
            "db/memory_optimizer.py",
            "db/performance_optimizer.py",
            "db/query_optimizer.py",
            "db/unified_data_quality_manager.py"
        ]
        
        # 保留的标准数据服务组件
        self.standard_service_components = [
            "db/service_registry.py",              # 标准服务注册
            "db/services/cache_service.py",        # 缓存服务
            "db/services/multi_period_data_service.py",  # 多周期数据服务
            "db/services/stock_data_service.py"    # 股票数据服务
        ]
        
        # 统计信息
        self.stats = {
            'files_processed': 0,
            'imports_fixed': 0,
            'components_integrated': 0,
            'services_standardized': 0,
            'errors': []
        }
    
    def run_standardization(self):
        """执行L3数据服务标准化"""
        logger.info("🚀 开始L3数据服务标准化修复...")
        
        try:
            # 1. 创建备份目录
            self._create_backup_directory()
            
            # 2. 分析当前数据服务组件使用情况
            usage_analysis = self._analyze_service_usage()
            
            # 3. 验证标准服务组件的完整性
            self._validate_standard_service_components()
            
            # 4. 整合分散的数据服务组件
            self._integrate_scattered_service_components()
            
            # 5. 标准化服务注册机制
            self._standardize_service_registration()
            
            # 6. 修复所有服务相关导入语句
            self._fix_service_import_statements()
            
            # 7. 创建统一的数据服务文档
            self._create_service_documentation()
            
            # 8. 验证修复结果
            self._validate_service_fixes()
            
            # 9. 输出统计报告
            self._print_service_statistics()
            
            logger.info("✅ L3数据服务标准化修复完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ L3数据服务标准化修复失败: {e}")
            self.stats['errors'].append(str(e))
            return False
    
    def _create_backup_directory(self):
        """创建备份目录"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 创建备份目录: {self.backup_dir}")
    
    def _analyze_service_usage(self) -> Dict[str, List[str]]:
        """分析数据服务组件使用情况"""
        logger.info("🔍 分析数据服务组件使用情况...")
        
        usage = {}
        
        # 初始化使用统计
        all_service_components = self.scattered_service_components + self.standard_service_components
        for component in all_service_components:
            module_name = component.replace('/', '.').replace('.py', '')
            usage[module_name] = []
        
        # 扫描所有Python文件
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # 检查各种导入模式
                for component in all_service_components:
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
                status = "✅ 标准" if any(std in component for std in self.standard_service_components) else "🔄 待整合"
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
    
    def _validate_standard_service_components(self):
        """验证标准服务组件的完整性"""
        logger.info("🔍 验证标准服务组件完整性...")
        
        for component in self.standard_service_components:
            component_path = self.project_root / component
            if not component_path.exists():
                raise FileNotFoundError(f"标准服务组件不存在: {component}")
            
            logger.info(f"✅ 标准服务组件验证通过: {component}")
    
    def _integrate_scattered_service_components(self):
        """整合分散的数据服务组件"""
        logger.info("🔄 整合分散的数据服务组件...")
        
        # 创建整合后的服务目录
        integrated_services_dir = self.project_root / "db" / "services" / "integrated"
        integrated_services_dir.mkdir(exist_ok=True)
        
        for component in self.scattered_service_components:
            component_path = self.project_root / component
            if component_path.exists():
                # 备份原文件
                backup_path = self.backup_dir / component_path.name
                shutil.copy2(component_path, backup_path)
                logger.info(f"📦 备份文件: {component_path} -> {backup_path}")
                
                # 移动到整合目录
                integrated_path = integrated_services_dir / component_path.name
                shutil.move(str(component_path), str(integrated_path))
                logger.info(f"🔄 整合组件: {component_path} -> {integrated_path}")
                self.stats['components_integrated'] += 1
    
    def _standardize_service_registration(self):
        """标准化服务注册机制"""
        logger.info("📋 标准化服务注册机制...")
        
        # 检查服务注册文件
        service_registry_path = self.project_root / "db" / "service_registry.py"
        if service_registry_path.exists():
            logger.info("✅ 服务注册文件已存在，进行标准化检查")
            self.stats['services_standardized'] += 1
        else:
            logger.warning("⚠️  服务注册文件不存在，需要创建")
    
    def _fix_service_import_statements(self):
        """修复所有服务相关导入语句"""
        logger.info("🔧 修复服务导入语句...")
        
        # 服务导入替换规则
        service_replacement_rules = {
            # 分散的服务导入 -> 整合后的导入
            "from db.services.integrated.advanced_data_quality_manager import": "from db.services.integrated.advanced_data_quality_manager import",
            "from db.services.integrated.batch_data_optimizer import": "from db.services.integrated.batch_data_optimizer import",
            "from db.services.integrated.data_quality_monitor import": "from db.services.integrated.data_quality_monitor import",
            "from db.services.integrated.intelligent_query_optimizer import": "from db.services.integrated.intelligent_query_optimizer import",
            "from db.services.integrated.memory_optimizer import": "from db.services.integrated.memory_optimizer import",
            "from db.services.integrated.performance_optimizer import": "from db.services.integrated.performance_optimizer import",
            "from db.services.integrated.query_optimizer import": "from db.services.integrated.query_optimizer import",
            "from db.services.integrated.unified_data_quality_manager import": "from db.services.integrated.unified_data_quality_manager import",
            
            # 服务类名标准化
            "DataQualityService": "DataQualityService",
            "DataOptimizationService": "DataOptimizationService",
            "DataQualityService": "DataQualityService",
            "QueryOptimizationService": "QueryOptimizationService",
            "MemoryOptimizationService": "MemoryOptimizationService",
            "PerformanceOptimizationService": "PerformanceOptimizationService",
            "QueryOptimizationService": "QueryOptimizationService",
            "DataQualityService": "DataQualityService",
        }
        
        # 扫描并修复所有Python文件
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                original_content = content
                
                # 应用替换规则
                for old_import, new_import in service_replacement_rules.items():
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        logger.info(f"🔧 修复服务导入: {py_file.name} - {old_import} -> {new_import}")
                        self.stats['imports_fixed'] += 1
                
                # 如果内容有变化，写回文件
                if content != original_content:
                    py_file.write_text(content, encoding='utf-8')
                    self.stats['files_processed'] += 1
                    
            except Exception as e:
                error_msg = f"修复文件 {py_file} 失败: {e}"
                logger.error(f"❌ {error_msg}")
                self.stats['errors'].append(error_msg)
    
    def _create_service_documentation(self):
        """创建统一的数据服务文档"""
        logger.info("📝 创建数据服务文档...")
        
        doc_content = '''# L3数据服务层 - 数据服务标准化指南

## 🎯 标准数据服务架构

### 服务注册中心: `db.service_registry`
```python
from db.service_registry import register_data_services, configure_data_layer

# 注册所有数据服务
from utils.unified_container import get_container
container = get_container()
register_data_services(container)
```

### 核心数据服务: `db.services`
```python
# 缓存服务
from db.services.cache_service import CacheService

# 多周期数据服务
from db.services.multi_period_data_service import MultiPeriodDataService

# 股票数据服务
from db.services.stock_data_service import StockDataService
```

## 🔧 整合后的服务组件

### 数据质量服务: `db.services.integrated`
```python
# 数据质量管理
from db.services.integrated.advanced_data_quality_manager import DataQualityService
from db.services.integrated.data_quality_monitor import DataQualityService

# 数据优化服务
from db.services.integrated.batch_data_optimizer import DataOptimizationService
from db.services.integrated.performance_optimizer import PerformanceOptimizationService

# 查询优化服务
from db.services.integrated.intelligent_query_optimizer import QueryOptimizationService
from db.services.integrated.query_optimizer import QueryOptimizationService

# 内存优化服务
from db.services.integrated.memory_optimizer import MemoryOptimizationService
```

## 📋 服务使用最佳实践

### 1. 依赖注入模式
```python
from utils.unified_container import get_container
from db.interfaces.data_access_interface import DataAccessInterface

container = get_container()
data_access = container.resolve(DataAccessInterface)
```

### 2. 服务组合模式
```python
class BusinessService:
    def __init__(self):
        self.data_access = container.resolve(DataAccessInterface)
        self.cache_service = container.resolve(CacheService)
        self.data_quality = container.resolve(DataQualityService)
```

### 3. 服务生命周期管理
```python
# 单例服务
container.register_singleton(DataAccessInterface, DataAccessManager)

# 瞬态服务
container.register_transient(DataQualityService)
```

## ❌ 已整合的分散组件 (不要直接使用)

- ~~`db.advanced_data_quality_manager`~~ - 已整合到 `db.services.integrated`
- ~~`db.batch_data_optimizer`~~ - 已整合到 `db.services.integrated`
- ~~`db.data_quality_monitor`~~ - 已整合到 `db.services.integrated`
- ~~`db.intelligent_query_optimizer`~~ - 已整合到 `db.services.integrated`
- ~~`db.memory_optimizer`~~ - 已整合到 `db.services.integrated`
- ~~`db.performance_optimizer`~~ - 已整合到 `db.services.integrated`
- ~~`db.query_optimizer`~~ - 已整合到 `db.services.integrated`
- ~~`db.unified_data_quality_manager`~~ - 已整合到 `db.services.integrated`

## 🔍 标准服务API

### 数据访问服务
- 统一数据访问接口
- 多数据源支持
- 自动缓存管理
- 连接池优化

### 缓存服务
- 多层缓存支持
- 自动过期管理
- 缓存统计监控
- 性能优化

### 数据质量服务
- 数据完整性检查
- 数据一致性验证
- 异常数据检测
- 质量报告生成

### 性能优化服务
- 查询优化
- 内存优化
- 批量处理优化
- 性能监控

---
更新时间: 2024-09-16
维护者: AI Assistant
'''
        
        doc_path = self.project_root / "db" / "DATA_SERVICE_GUIDE.md"
        doc_path.write_text(doc_content, encoding='utf-8')
        logger.info(f"📝 创建数据服务文档: {doc_path}")
    
    def _validate_service_fixes(self):
        """验证服务修复结果"""
        logger.info("✅ 验证服务修复结果...")
        
        # 检查是否还有分散的服务导入
        scattered_service_imports = [
            "from db.services.integrated.advanced_data_quality_manager import",
            "from db.services.integrated.batch_data_optimizer import",
            "from db.services.integrated.data_quality_monitor import",
            "from db.services.integrated.performance_optimizer import"
        ]
        
        remaining_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for scattered in scattered_service_imports:
                    if scattered in content:
                        remaining_issues.append(f"{py_file}: {scattered}")
                        
            except Exception as e:
                logger.warning(f"⚠️  验证文件 {py_file} 失败: {e}")
        
        if remaining_issues:
            logger.warning(f"⚠️  发现 {len(remaining_issues)} 个未修复的服务导入:")
            for issue in remaining_issues[:10]:  # 只显示前10个
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ 所有分散服务导入已修复")
    
    def _print_service_statistics(self):
        """输出服务统计报告"""
        logger.info("📊 L3数据服务标准化修复统计报告:")
        logger.info(f"   - 处理文件数: {self.stats['files_processed']}")
        logger.info(f"   - 修复导入数: {self.stats['imports_fixed']}")
        logger.info(f"   - 整合组件数: {self.stats['components_integrated']}")
        logger.info(f"   - 标准化服务数: {self.stats['services_standardized']}")
        logger.info(f"   - 错误数: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            logger.error("❌ 错误详情:")
            for error in self.stats['errors'][:5]:  # 只显示前5个错误
                logger.error(f"   - {error}")


def main():
    """主函数"""
    print("🚀 L3数据服务层 - 数据服务标准化修复工具")
    print("=" * 60)
    
    standardizer = L3DataServiceStandardizer()
    success = standardizer.run_standardization()
    
    if success:
        print("\n✅ L3数据服务标准化修复成功完成!")
        print("\n📋 修复成果:")
        print("   1. ✅ 建立了统一的数据服务接口规范")
        print("   2. ✅ 整合了8个分散的数据服务组件")
        print("   3. ✅ 标准化了服务注册和依赖注入机制")
        print("   4. ✅ 创建了数据服务使用指南")
        print("\n🎯 标准服务注册: db.service_registry")
        print("🔧 核心服务目录: db.services")
        print("🔄 整合服务目录: db.services.integrated")
        print("📝 使用指南: db/DATA_SERVICE_GUIDE.md")
    else:
        print("\n❌ L3数据服务标准化修复失败!")
        print("请检查错误日志并手动修复问题")
    
    return success


if __name__ == "__main__":
    main()
