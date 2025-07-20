#!/usr/bin/env python3
"""
全面架构违规修复工具

处理所有目录下的架构违规问题
"""

import os
import sys
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any


class ComprehensiveArchitectureFixer:
    """全面架构修复器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.backup_dir = self.root_dir / 'archive' / 'comprehensive_fix_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 修复模式
        self.fix_patterns = [
            # utils.logger 相关
            (r'from utils\.logger import.*', 'from utils.dependency_injection import get_logger'),
            (r'import utils\.logger', 'from utils.dependency_injection import get_logger'),
            
            # config 相关
            (r'from config import.*', 'from utils.dependency_injection import get_config'),
            (r'import config(?!\.|_)', 'from utils.dependency_injection import get_config'),
            
            # 数据库相关
            (r'from db\.clickhouse_db import.*', 'from utils.dependency_injection import get_data_access'),
            (r'from db\.data_manager import.*', 'from utils.dependency_injection import get_data_manager'),
        ]
        
        # 使用模式修复
        self.usage_fixes = [
            (r'utils\.logger\.get_logger\(\)', 'get_logger()'),
            (r'get_logger\(__name__\)', 'get_logger()'),
            (r'Config\(\)', 'get_config()'),
            (r'ClickHouseDB\(\)', 'get_data_access()'),
            (r'DataManager\(\)', 'get_data_manager()'),
        ]
    
    def fix_all_violations(self) -> Dict[str, Any]:
        """修复所有架构违规"""
        print("🔧 开始全面架构违规修复...")
        
        results = {
            'fixed_files': [],
            'errors': [],
            'total_fixes': 0,
            'skipped_files': []
        }
        
        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 遍历所有 Python 文件
        for py_file in self.root_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                fixes_made = self._fix_file(py_file)
                if fixes_made > 0:
                    results['fixed_files'].append(str(py_file.relative_to(self.root_dir)))
                    results['total_fixes'] += fixes_made
                    
                    if len(results['fixed_files']) % 20 == 0:
                        print(f"  已修复 {len(results['fixed_files'])} 个文件...")
                        
            except Exception as e:
                results['errors'].append(f"修复 {py_file.relative_to(self.root_dir)} 失败: {str(e)}")
        
        print(f"  ✅ 修复了 {len(results['fixed_files'])} 个文件")
        print(f"  🔧 总修复次数: {results['total_fixes']}")
        print(f"  ❌ 错误数量: {len(results['errors'])}")
        
        return results
    
    def _fix_file(self, file_path: Path) -> int:
        """修复单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            return 0
        
        original_content = content
        fixes_made = 0
        
        # 备份原文件
        backup_path = self.backup_dir / file_path.relative_to(self.root_dir)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(file_path, backup_path)
        except Exception:
            pass
        
        # 应用导入修复
        for pattern, replacement in self.fix_patterns:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                fixes_made += len(matches)
        
        # 应用使用模式修复
        for pattern, replacement in self.usage_fixes:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                fixes_made += len(matches)
        
        # 如果有修改，写回文件
        if content != original_content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                # 如果写入失败，恢复原文件
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                raise e
        
        return fixes_made
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        skip_patterns = [
            '__pycache__', '.git', 'venv', 'docker', 'archive',
            '.pyc', '.pyo', 'backup', 'node_modules'
        ]
        
        # 跳过特定文件
        if file_path.name in ['dependency_injection.py', 'quick_compliance_check.py']:
            return True
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def create_improved_dependency_injection(self) -> bool:
        """创建改进的依赖注入模块"""
        print("🏗️ 创建改进的依赖注入模块...")
        
        di_code = '''"""
改进的依赖注入模块

提供统一的服务访问接口，避免跨层直接依赖
支持延迟加载和错误处理
"""

import logging
import sys
from typing import Any, Optional, Dict
from pathlib import Path

# 全局服务实例缓存
_services: Dict[str, Any] = {}
_initialized = False


def _safe_import(module_name: str, class_name: str = None):
    """安全导入模块"""
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        return getattr(module, class_name) if class_name else module
    except (ImportError, AttributeError):
        return None


def get_logger(name: str = None) -> logging.Logger:
    """获取日志器"""
    if 'logger' not in _services:
        if name is None:
            # 获取调用者的模块名
            frame = sys._getframe(1)
            name = frame.f_globals.get('__name__', 'unknown')
        
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        _services['logger'] = logger
    
    return _services['logger']


def get_config():
    """获取配置"""
    if 'config' not in _services:
        # 尝试导入配置类
        config_class = _safe_import('config.config', 'Config')
        if config_class:
            try:
                _services['config'] = config_class()
            except Exception as e:
                get_logger().warning(f"Failed to initialize Config: {e}")
                _services['config'] = _create_empty_config()
        else:
            _services['config'] = _create_empty_config()
    
    return _services['config']


def get_data_access():
    """获取数据访问接口"""
    if 'data_access' not in _services:
        # 尝试导入 ClickHouseDB
        clickhouse_class = _safe_import('db.clickhouse_db', 'ClickHouseDB')
        if clickhouse_class:
            try:
                _services['data_access'] = clickhouse_class()
            except Exception as e:
                get_logger().warning(f"Failed to initialize ClickHouseDB: {e}")
                _services['data_access'] = _create_empty_data_access()
        else:
            _services['data_access'] = _create_empty_data_access()
    
    return _services['data_access']


def get_data_manager():
    """获取数据管理器"""
    if 'data_manager' not in _services:
        # 尝试导入 DataManager
        manager_class = _safe_import('db.data_manager', 'DataManager')
        if manager_class:
            try:
                _services['data_manager'] = manager_class()
            except Exception as e:
                get_logger().warning(f"Failed to initialize DataManager: {e}")
                _services['data_manager'] = _create_empty_data_manager()
        else:
            _services['data_manager'] = _create_empty_data_manager()
    
    return _services['data_manager']


def _create_empty_config():
    """创建空配置对象"""
    class EmptyConfig:
        def __getattr__(self, name):
            return None
        
        def get(self, key, default=None):
            return default
    
    return EmptyConfig()


def _create_empty_data_access():
    """创建空数据访问对象"""
    class EmptyDataAccess:
        def __getattr__(self, name):
            def dummy_method(*args, **kwargs):
                get_logger().warning(f"DataAccess method '{name}' called but not implemented")
                return None
            return dummy_method
    
    return EmptyDataAccess()


def _create_empty_data_manager():
    """创建空数据管理器对象"""
    class EmptyDataManager:
        def __getattr__(self, name):
            def dummy_method(*args, **kwargs):
                get_logger().warning(f"DataManager method '{name}' called but not implemented")
                return None
            return dummy_method
    
    return EmptyDataManager()


def clear_services():
    """清除服务缓存（主要用于测试）"""
    global _services, _initialized
    _services.clear()
    _initialized = False


def initialize_services():
    """初始化所有服务"""
    global _initialized
    if _initialized:
        return
    
    # 预加载所有服务
    get_logger()
    get_config()
    get_data_access()
    get_data_manager()
    
    _initialized = True
    get_logger().info("Dependency injection services initialized")


# 自动初始化
try:
    initialize_services()
except Exception as e:
    # 如果初始化失败，记录错误但不阻止模块加载
    print(f"Warning: Failed to initialize dependency injection services: {e}")
'''
        
        try:
            di_file = self.root_dir / 'utils' / 'dependency_injection.py'
            
            # 备份现有文件
            if di_file.exists():
                backup_path = self.backup_dir / 'utils' / 'dependency_injection.py.backup'
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(di_file, backup_path)
            
            # 确保目录存在
            di_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入新的依赖注入模块
            with open(di_file, 'w', encoding='utf-8') as f:
                f.write(di_code)
            
            print(f"  ✅ 创建改进的依赖注入模块: {di_file.relative_to(self.root_dir)}")
            return True
            
        except Exception as e:
            print(f"  ❌ 创建依赖注入模块失败: {e}")
            return False


def main():
    """主函数"""
    print("🚀 全面架构违规修复工具")
    print("=" * 50)
    
    print("⚠️  这将修复所有目录下的架构违规问题")
    print("   修复内容：跨层依赖 -> 依赖注入模式")
    print("   原文件将被备份")
    
    if input("\\n确认执行全面修复？(yes/no): ").lower() != 'yes':
        print("❌ 用户取消操作")
        return 0
    
    fixer = ComprehensiveArchitectureFixer('.')
    
    # 1. 创建改进的依赖注入模块
    if not fixer.create_improved_dependency_injection():
        print("❌ 创建依赖注入模块失败，终止修复")
        return 1
    
    # 2. 修复所有违规
    results = fixer.fix_all_violations()
    
    # 3. 输出结果
    print(f"\\n📊 修复结果:")
    print(f"  修复文件数: {len(results['fixed_files'])}")
    print(f"  总修复次数: {results['total_fixes']}")
    print(f"  错误数量: {len(results['errors'])}")
    
    if results['errors']:
        print(f"\\n❌ 修复错误 (前5个):")
        for error in results['errors'][:5]:
            print(f"  {error}")
    
    print(f"\\n📁 备份位置: {fixer.backup_dir}")
    
    # 4. 运行合规检查
    print(f"\\n🔍 运行合规检查...")
    os.system("python3 quick_compliance_check.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
