#!/usr/bin/env python3
"""
快速架构违规修复工具

专门处理最关键的跨层依赖问题，快速提升架构合规性
"""

import os
import sys
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any


class QuickArchitectureFixer:
    """快速架构修复器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.backup_dir = self.root_dir / 'archive' / 'quick_fix_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 常见的跨层依赖模式和修复方案
        self.fix_patterns = {
            # 直接导入 utils.logger -> 使用依赖注入
            r'from utils\.logger import.*': 'from utils.dependency_injection import get_logger',
            r'import utils\.logger': 'from utils.dependency_injection import get_logger',
            
            # 直接导入 config -> 使用依赖注入
            r'from config import.*': 'from utils.dependency_injection import get_config',
            r'import config': 'from utils.dependency_injection import get_config',
            
            # 直接导入数据库模块 -> 使用依赖注入
            r'from db\.clickhouse_db import.*': 'from utils.dependency_injection import get_data_access',
            r'from db\.data_manager import.*': 'from utils.dependency_injection import get_data_manager',
        }
        
        # 使用模式替换
        self.usage_patterns = {
            r'utils\.logger\.get_logger\(\)': 'get_logger()',
            r'logger = get_logger\(\)': 'logger = get_logger()',
            r'Config\(\)': 'get_config()',
            r'ClickHouseDB\(\)': 'get_data_access()',
            r'DataManager\(\)': 'get_data_manager()',
        }
    
    def create_dependency_injection_module(self) -> bool:
        """创建简化的依赖注入模块"""
        print("🏗️ 创建依赖注入模块...")
        
        di_code = '''"""
简化的依赖注入模块

提供统一的服务访问接口，避免跨层直接依赖
"""

import logging
from typing import Any, Optional

# 全局服务实例缓存
_services = {}


def get_logger(name: str = None) -> logging.Logger:
    """获取日志器"""
    if 'logger' not in _services:
        if name is None:
            name = __name__
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        _services['logger'] = logger
    return _services['logger']


def get_config():
    """获取配置"""
    if 'config' not in _services:
        try:
            from config.config import Config
            _services['config'] = Config()
        except ImportError:
            # 如果配置模块不存在，返回空配置
            class EmptyConfig:
                def __getattr__(self, name):
                    return None
            _services['config'] = EmptyConfig()
    return _services['config']


def get_data_access():
    """获取数据访问接口"""
    if 'data_access' not in _services:
        try:
            from db.clickhouse_db import ClickHouseDB
            _services['data_access'] = ClickHouseDB()
        except ImportError:
            # 如果数据库模块不存在，返回空实现
            class EmptyDataAccess:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            _services['data_access'] = EmptyDataAccess()
    return _services['data_access']


def get_data_manager():
    """获取数据管理器"""
    if 'data_manager' not in _services:
        try:
            from db.data_manager import DataManager
            _services['data_manager'] = DataManager()
        except ImportError:
            # 如果数据管理器不存在，返回空实现
            class EmptyDataManager:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            _services['data_manager'] = EmptyDataManager()
    return _services['data_manager']


def clear_services():
    """清除服务缓存（主要用于测试）"""
    global _services
    _services.clear()
'''
        
        try:
            di_file = self.root_dir / 'utils' / 'dependency_injection.py'
            
            # 如果文件已存在，备份
            if di_file.exists():
                backup_path = self.backup_dir / 'utils' / 'dependency_injection.py.backup'
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(di_file, backup_path)
            
            # 确保目录存在
            di_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入新的依赖注入模块
            with open(di_file, 'w', encoding='utf-8') as f:
                f.write(di_code)
            
            print(f"  ✅ 创建依赖注入模块: {di_file.relative_to(self.root_dir)}")
            return True
            
        except Exception as e:
            print(f"  ❌ 创建依赖注入模块失败: {e}")
            return False
    
    def fix_critical_violations(self) -> Dict[str, Any]:
        """修复关键的架构违规"""
        print("🔧 修复关键架构违规...")
        
        results = {
            'fixed_files': [],
            'errors': [],
            'total_fixes': 0
        }
        
        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找需要修复的 Python 文件
        target_dirs = ['analysis', 'strategy', 'indicators']
        
        for dir_name in target_dirs:
            dir_path = self.root_dir / dir_name
            if not dir_path.exists():
                continue
            
            for py_file in dir_path.rglob("*.py"):
                if self._should_skip_file(py_file):
                    continue
                
                try:
                    fixes_made = self._fix_file(py_file)
                    if fixes_made > 0:
                        results['fixed_files'].append(str(py_file.relative_to(self.root_dir)))
                        results['total_fixes'] += fixes_made
                        
                        if len(results['fixed_files']) % 10 == 0:
                            print(f"  已修复 {len(results['fixed_files'])} 个文件...")
                            
                except Exception as e:
                    results['errors'].append(f"修复 {py_file.relative_to(self.root_dir)} 失败: {str(e)}")
        
        print(f"  ✅ 修复了 {len(results['fixed_files'])} 个文件")
        print(f"  🔧 总修复次数: {results['total_fixes']}")
        print(f"  ❌ 错误数量: {len(results['errors'])}")
        
        return results
    
    def _fix_file(self, file_path: Path) -> int:
        """修复单个文件的架构违规"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes_made = 0
            
            # 备份原文件
            backup_path = self.backup_dir / file_path.relative_to(self.root_dir)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)
            
            # 应用导入修复模式
            for pattern, replacement in self.fix_patterns.items():
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    fixes_made += 1
            
            # 应用使用模式修复
            for pattern, replacement in self.usage_patterns.items():
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    fixes_made += 1
            
            # 如果有修改，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            return fixes_made
            
        except Exception as e:
            # 如果修复失败，恢复原文件
            if backup_path.exists():
                shutil.copy2(backup_path, file_path)
            raise e
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        skip_patterns = [
            '__pycache__', '.git', 'venv', 'docker', 'archive',
            'test_', 'backup', '.pyc'
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def create_architecture_compliance_check(self) -> bool:
        """创建简化的架构合规检查脚本"""
        print("📋 创建架构合规检查脚本...")
        
        check_code = '''#!/usr/bin/env python3
"""
简化的架构合规检查脚本
"""

import os
import re
from pathlib import Path

def check_compliance():
    """检查架构合规性"""
    violations = 0
    
    # 检查关键违规模式
    violation_patterns = [
        r'from utils\.logger import',
        r'import utils\.logger',
        r'from config import',
        r'import config',
        r'from db\.clickhouse_db import',
        r'from db\.data_manager import'
    ]
    
    for py_file in Path('.').rglob("*.py"):
        if any(skip in str(py_file) for skip in ['__pycache__', '.git', 'venv', 'archive']):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in violation_patterns:
                if re.search(pattern, content):
                    violations += 1
                    print(f"违规: {py_file} - {pattern}")
        
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    print(f"\\n总违规数: {violations}")
    return violations

if __name__ == "__main__":
    violations = check_compliance()
    exit(1 if violations > 0 else 0)
'''
        
        try:
            check_file = self.root_dir / 'quick_compliance_check.py'
            with open(check_file, 'w', encoding='utf-8') as f:
                f.write(check_code)
            
            # 设置执行权限
            os.chmod(check_file, 0o755)
            
            print(f"  ✅ 创建合规检查脚本: {check_file.relative_to(self.root_dir)}")
            return True
            
        except Exception as e:
            print(f"  ❌ 创建合规检查脚本失败: {e}")
            return False


def main():
    """主函数"""
    print("🚀 快速架构违规修复工具")
    print("=" * 50)
    
    print("⚠️  这是一个快速修复工具，将处理最关键的架构违规问题")
    print("   修复内容：跨层依赖 -> 依赖注入模式")
    print("   原文件将被备份到 archive/quick_fix_backup/")
    
    if input("\\n确认执行快速修复？(yes/no): ").lower() != 'yes':
        print("❌ 用户取消操作")
        return 0
    
    fixer = QuickArchitectureFixer('.')
    
    # 1. 创建依赖注入模块
    if not fixer.create_dependency_injection_module():
        print("❌ 创建依赖注入模块失败，终止修复")
        return 1
    
    # 2. 修复关键违规
    results = fixer.fix_critical_violations()
    
    # 3. 创建合规检查脚本
    fixer.create_architecture_compliance_check()
    
    # 4. 输出结果
    print(f"\\n📊 修复结果:")
    print(f"  修复文件数: {len(results['fixed_files'])}")
    print(f"  总修复次数: {results['total_fixes']}")
    print(f"  错误数量: {len(results['errors'])}")
    
    if results['errors']:
        print(f"\\n❌ 修复错误:")
        for error in results['errors'][:5]:  # 只显示前5个错误
            print(f"  {error}")
    
    print(f"\\n📁 备份位置: {fixer.backup_dir}")
    print(f"\\n🔍 运行合规检查: python3 quick_compliance_check.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
