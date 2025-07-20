#!/usr/bin/env python3
"""
架构违规修复工具

基于架构合规检查报告，修复跨层依赖违规和文件错位问题
"""

import os
import sys
import json
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set


class ArchitectureViolationFixer:
    """架构违规修复器"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.backup_dir = self.root_dir / 'archive' / 'architecture_fix_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 六层架构定义
        self.layers = {
            'L6': {'name': '用户接口层', 'dirs': ['bin/', 'api/']},
            'L5': {'name': '业务应用层', 'dirs': ['strategy/', 'analysis/']},
            'L4': {'name': '核心服务层', 'dirs': ['indicators/', 'formula/']},
            'L3': {'name': '数据服务层', 'dirs': ['db/interfaces/', 'db/managers/', 'db/services/']},
            'L2': {'name': '存储访问层', 'dirs': ['db/']},
            'L1': {'name': '基础设施层', 'dirs': ['utils/', 'config/', 'enums/']}
        }
        
        # 依赖注入模式映射
        self.dependency_injection_patterns = {
            'utils.logger': 'from utils.dependency_injection import get_logger',
            'config': 'from utils.dependency_injection import get_config',
            'db.clickhouse_db': 'from utils.dependency_injection import get_data_access',
            'db.data_manager': 'from utils.dependency_injection import get_data_manager'
        }
    
    def create_backup(self) -> bool:
        """创建备份目录"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建架构修复备份目录: {self.backup_dir}")
            return True
        except Exception as e:
            print(f"❌ 创建备份目录失败: {e}")
            return False
    
    def fix_cross_layer_violations(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """修复跨层依赖违规"""
        print("\n🔧 修复跨层依赖违规...")
        
        results = {'fixed': [], 'errors': [], 'skipped': []}
        
        # 按文件分组违规
        violations_by_file = {}
        for violation in violations:
            file_path = violation['file']
            if file_path not in violations_by_file:
                violations_by_file[file_path] = []
            violations_by_file[file_path].append(violation)
        
        for file_path, file_violations in violations_by_file.items():
            try:
                if self._fix_file_dependencies(file_path, file_violations):
                    results['fixed'].append(file_path)
                else:
                    results['skipped'].append(file_path)
            except Exception as e:
                results['errors'].append(f"修复 {file_path} 失败: {str(e)}")
        
        print(f"  ✅ 修复了 {len(results['fixed'])} 个文件的依赖违规")
        print(f"  ⚠️  跳过了 {len(results['skipped'])} 个文件")
        print(f"  ❌ 修复失败 {len(results['errors'])} 个文件")
        
        return results
    
    def fix_misplaced_files(self, misplaced_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """修复文件错位问题"""
        print("\n📁 修复文件错位问题...")
        
        results = {'moved': [], 'errors': [], 'skipped': []}
        
        for file_info in misplaced_files:
            file_path = file_info['file']
            suggested_location = file_info.get('suggested_location')
            
            if not suggested_location:
                results['skipped'].append(file_path)
                continue
            
            try:
                if self._move_file_safely(file_path, suggested_location):
                    results['moved'].append({
                        'from': file_path,
                        'to': suggested_location
                    })
                else:
                    results['skipped'].append(file_path)
            except Exception as e:
                results['errors'].append(f"移动 {file_path} 失败: {str(e)}")
        
        print(f"  ✅ 移动了 {len(results['moved'])} 个文件")
        print(f"  ⚠️  跳过了 {len(results['skipped'])} 个文件")
        print(f"  ❌ 移动失败 {len(results['errors'])} 个文件")
        
        return results
    
    def create_dependency_injection_interfaces(self) -> Dict[str, Any]:
        """创建依赖注入接口"""
        print("\n🏗️ 创建依赖注入接口...")
        
        results = {'created': [], 'errors': []}
        
        # 创建依赖注入容器
        di_container_code = '''"""
依赖注入容器

提供统一的服务访问接口，避免跨层直接依赖
"""

from typing import Any, Dict, Type, Optional
import logging
from config.config import Config
from utils.logger import get_logger


class ServiceContainer:
    """服务容器"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(self, service_type: str, service_instance: Any, singleton: bool = True):
        """注册服务"""
        if singleton:
            self._singletons[service_type] = service_instance
        else:
            self._services[service_type] = service_instance
    
    def get(self, service_type: str) -> Any:
        """获取服务实例"""
        if service_type in self._singletons:
            return self._singletons[service_type]
        elif service_type in self._services:
            return self._services[service_type]
        else:
            raise ValueError(f"Service {service_type} not registered")


# 全局容器实例
_container = ServiceContainer()


def get_container() -> ServiceContainer:
    """获取服务容器"""
    return _container


def get_logger(name: str = __name__) -> logging.Logger:
    """获取日志器"""
    try:
        return _container.get('logger')
    except ValueError:
        # 如果未注册，返回默认日志器
        return logging.getLogger(name)


def get_config() -> Config:
    """获取配置"""
    try:
        return _container.get('config')
    except ValueError:
        # 如果未注册，返回默认配置
        return Config()


def get_data_access():
    """获取数据访问接口"""
    try:
        return _container.get('data_access')
    except ValueError:
        # 如果未注册，延迟导入
        from db.clickhouse_db import ClickHouseDB
        return ClickHouseDB()


def get_data_manager():
    """获取数据管理器"""
    try:
        return _container.get('data_manager')
    except ValueError:
        # 如果未注册，延迟导入
        from db.data_manager import DataManager
        return DataManager()


def initialize_services():
    """初始化服务"""
    # 注册基础服务
    _container.register('logger', get_logger())
    _container.register('config', Config())
    
    # 注册数据服务
    from db.clickhouse_db import ClickHouseDB
    from db.data_manager import DataManager
    
    _container.register('data_access', ClickHouseDB())
    _container.register('data_manager', DataManager())
'''
        
        try:
            di_file = self.root_dir / 'utils' / 'dependency_injection.py'
            if not di_file.exists():
                with open(di_file, 'w', encoding='utf-8') as f:
                    f.write(di_container_code)
                results['created'].append(str(di_file.relative_to(self.root_dir)))
                print(f"  ✅ 创建依赖注入容器: {di_file.relative_to(self.root_dir)}")
        except Exception as e:
            results['errors'].append(f"创建依赖注入容器失败: {str(e)}")
        
        return results
    
    def _fix_file_dependencies(self, file_path: str, violations: List[Dict[str, Any]]) -> bool:
        """修复单个文件的依赖违规"""
        full_path = self.root_dir / file_path
        
        if not full_path.exists():
            return False
        
        # 备份原文件
        backup_path = self.backup_dir / file_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full_path, backup_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修复每个违规的导入
            for violation in violations:
                import_stmt = violation['import_statement']
                
                # 检查是否可以用依赖注入替换
                if import_stmt in self.dependency_injection_patterns:
                    new_import = self.dependency_injection_patterns[import_stmt]
                    
                    # 替换导入语句
                    old_patterns = [
                        f"from {import_stmt} import",
                        f"import {import_stmt}",
                        f"from {import_stmt}."
                    ]
                    
                    for pattern in old_patterns:
                        if pattern in content:
                            # 简单替换为依赖注入模式
                            content = content.replace(pattern, f"# {pattern}  # 已替换为依赖注入")
                    
                    # 在文件开头添加新的导入
                    if new_import not in content:
                        import_section = content.split('\n\n')[0]  # 获取导入部分
                        content = content.replace(import_section, import_section + '\n' + new_import)
            
            # 如果内容有变化，写回文件
            if content != original_content:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
        except Exception as e:
            # 如果修复失败，恢复原文件
            shutil.copy2(backup_path, full_path)
            raise e
        
        return False
    
    def _move_file_safely(self, file_path: str, suggested_location: str) -> bool:
        """安全地移动文件"""
        src_path = self.root_dir / file_path
        dst_path = self.root_dir / suggested_location
        
        if not src_path.exists():
            return False
        
        # 检查目标位置是否已存在文件
        if dst_path.exists():
            print(f"    ⚠️  目标位置已存在文件: {suggested_location}")
            return False
        
        # 检查是否为重要文件
        if self._is_important_file(file_path):
            print(f"    ⚠️  跳过重要文件: {file_path}")
            return False
        
        try:
            # 创建目标目录
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 备份原文件
            backup_path = self.backup_dir / file_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, backup_path)
            
            # 移动文件
            shutil.move(str(src_path), str(dst_path))
            
            print(f"    ✅ 移动: {file_path} -> {suggested_location}")
            return True
            
        except Exception as e:
            print(f"    ❌ 移动失败: {file_path} -> {suggested_location}: {e}")
            return False
    
    def _is_important_file(self, file_path: str) -> bool:
        """判断是否为重要文件"""
        important_patterns = [
            '__init__.py',
            'main.py',
            'config.py',
            'requirements.txt',
            'setup.py',
            'README'
        ]
        
        file_name = Path(file_path).name
        return any(pattern in file_name for pattern in important_patterns)


def main():
    """主函数"""
    print("🚀 架构违规修复工具")
    print("=" * 50)
    
    # 查找架构合规报告
    report_file = 'architecture_compliance_report.json'
    if not Path(report_file).exists():
        print(f"❌ 未找到架构合规报告: {report_file}")
        print("请先运行架构检查工具生成报告")
        return 1
    
    # 加载报告
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    violations = report.get('cross_layer_violations', [])
    misplaced_files = report.get('misplaced_files', [])
    
    print(f"📊 发现问题:")
    print(f"  跨层依赖违规: {len(violations)} 个")
    print(f"  文件错位问题: {len(misplaced_files)} 个")
    
    if len(violations) == 0 and len(misplaced_files) == 0:
        print("✅ 未发现需要修复的架构违规")
        return 0
    
    # 确认执行修复
    print(f"\n⚠️  即将执行架构违规修复")
    print(f"   这将修改文件导入语句和移动文件位置")
    print(f"   原文件将被备份")
    
    if input("\n确认执行修复？(yes/no): ").lower() != 'yes':
        print("❌ 用户取消操作")
        return 0
    
    # 执行修复
    fixer = ArchitectureViolationFixer('.')
    
    if not fixer.create_backup():
        return 1
    
    results = {}
    
    # 1. 创建依赖注入接口
    results['dependency_injection'] = fixer.create_dependency_injection_interfaces()
    
    # 2. 修复跨层依赖违规
    if violations:
        results['cross_layer_fixes'] = fixer.fix_cross_layer_violations(violations)
    
    # 3. 修复文件错位问题
    if misplaced_files:
        results['file_moves'] = fixer.fix_misplaced_files(misplaced_files)
    
    # 生成修复总结
    print(f"\n📊 修复总结:")
    
    if 'dependency_injection' in results:
        di_result = results['dependency_injection']
        print(f"  创建接口文件: {len(di_result['created'])} 个")
    
    if 'cross_layer_fixes' in results:
        fix_result = results['cross_layer_fixes']
        print(f"  修复依赖违规: {len(fix_result['fixed'])} 个文件")
    
    if 'file_moves' in results:
        move_result = results['file_moves']
        print(f"  移动错位文件: {len(move_result['moved'])} 个文件")
    
    # 保存修复报告
    fix_report = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'backup_location': str(fixer.backup_dir)
    }
    
    report_file = f'architecture_fix_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(fix_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 修复报告已保存到: {report_file}")
    print(f"📁 备份文件位置: {fixer.backup_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
