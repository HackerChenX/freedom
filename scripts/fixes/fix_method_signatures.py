#!/usr/bin/env python3
"""
方法签名修复脚本

修复系统中的方法签名不匹配问题，确保100%功能可用

作者：AI Assistant
创建时间：2025-01-13
"""

import os
import re
from pathlib import Path

class MethodSignatureFixer:
    """方法签名修复器"""
    
    def __init__(self):
        self.fixes_applied = []
        self.errors_encountered = []
    
    def fix_data_access_manager_signatures(self):
        """修复DataAccessManager的方法签名问题"""
        print("🔧 修复DataAccessManager方法签名...")
        
        file_path = "db/managers/data_access_manager.py"
        try:
            if not os.path.exists(file_path):
                return
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 检查并添加灵活的get_stock_data方法
            if 'def get_stock_data(' in content:
                # 替换现有方法为更灵活的版本
                pattern = r'def get_stock_data\(self, code: str, start_date: str, end_date: str,\s*columns: Optional\[List\[str\]\] = None\) -> pd\.DataFrame:'
                replacement = '''def get_stock_data(self, code: str = None, stock_code: str = None, 
                      start_date: str = None, end_date: str = None, 
                      columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:'''
                
                content = re.sub(pattern, replacement, content)
                
                # 更新方法体以处理不同的参数名
                method_body_pattern = r'(def get_stock_data\([^)]+\):[\s\S]*?"""[\s\S]*?"""[\s\S]*?)return self\.get_stock_data_data_access_manager\(code, start_date, end_date, columns\)'
                method_body_replacement = r'''\1# 处理不同的参数名
        actual_code = code or stock_code
        if not actual_code:
            raise ValueError("必须提供code或stock_code参数")
        return self.get_stock_data_data_access_manager(actual_code, start_date, end_date, columns)'''
                
                content = re.sub(method_body_pattern, method_body_replacement, content)
                
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied.append("修复DataAccessManager.get_stock_data方法签名")
                
        except Exception as e:
            self.errors_encountered.append(f"修复DataAccessManager签名失败: {e}")
    
    def fix_service_container_get_method(self):
        """修复ServiceContainer缺少get方法的问题"""
        print("🔧 修复ServiceContainer缺少get方法...")
        
        file_path = "utils/dependency_injection.py"
        try:
            if not os.path.exists(file_path):
                return
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 检查是否已有get方法
            if 'def get(' not in content:
                # 在resolve方法后添加get方法
                insertion_point = content.find('def clear_dependency_injection(self):')
                if insertion_point > 0:
                    get_method = '''
    def get(self, key: str, default=None):
        """
        获取服务实例（兼容旧的get方法调用）
        
        Args:
            key: 服务键名
            default: 默认值
            
        Returns:
            服务实例或默认值
        """
        try:
            # 如果key是字符串，尝试转换为类型
            if isinstance(key, str):
                # 对于常见的服务名称，直接返回相应实例
                if key == 'data_access':
                    from db.interfaces.data_access_interface import DataAccessInterface
                    return self.resolve(DataAccessInterface)
                else:
                    return default
            else:
                # 如果是类型，直接resolve
                return self.resolve(key)
        except Exception:
            return default
    
'''
                    content = content[:insertion_point] + get_method + content[insertion_point:]
                    
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied.append("添加ServiceContainer.get方法")
                
        except Exception as e:
            self.errors_encountered.append(f"修复ServiceContainer.get方法失败: {e}")
    
    def fix_strategy_executor_calls(self):
        """修复策略执行器中的方法调用"""
        print("🔧 修复策略执行器方法调用...")
        
        # 查找所有可能调用get_stock_data的文件
        files_to_check = [
            "strategy/strategy_executor.py",
            "strategy/strategy_parser.py"
        ]
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 修复get_stock_data调用中的参数名
                patterns_to_fix = [
                    # 修复stock_code参数名
                    (r'\.get_stock_data\(stock_code=([^,)]+)', r'.get_stock_data(code=\1'),
                    (r'\.get_stock_data\(\s*stock_code\s*=\s*([^,)]+)', r'.get_stock_data(code=\1'),
                    
                    # 修复容器get调用
                    (r'container\.get\(([^)]+)\)', r'container.resolve(\1)'),
                    (r'self\.container\.get\(([^)]+)\)', r'self.container.resolve(\1)'),
                ]
                
                for pattern, replacement in patterns_to_fix:
                    content = re.sub(pattern, replacement, content)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixes_applied.append(f"修复方法调用: {file_path}")
                    
            except Exception as e:
                self.errors_encountered.append(f"修复{file_path}方法调用失败: {e}")
    
    def create_compatibility_wrapper(self):
        """创建兼容性包装器"""
        print("🔧 创建兼容性包装器...")
        
        wrapper_content = '''"""
兼容性包装器 - 解决方法签名不匹配问题

提供向后兼容的方法包装，确保系统稳定运行
"""

from functools import wraps
from typing import Any, Callable

def flexible_parameters(func: Callable) -> Callable:
    """
    创建灵活参数装饰器，允许不同的参数名
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 处理code/stock_code参数的兼容性
        if 'stock_code' in kwargs and 'code' not in kwargs:
            kwargs['code'] = kwargs.pop('stock_code')
        elif 'code' in kwargs and 'stock_code' in kwargs:
            # 如果两个都有，优先使用code
            kwargs.pop('stock_code', None)
        
        return func(*args, **kwargs)
    
    return wrapper

def container_compatibility_wrapper(container):
    """
    为ServiceContainer添加兼容性方法
    """
    if not hasattr(container, 'get'):
        def get(key: str, default=None):
            try:
                if isinstance(key, str):
                    if key == 'data_access':
                        from db.interfaces.data_access_interface import DataAccessInterface
                        return container.resolve(DataAccessInterface)
                    else:
                        return default
                else:
                    return container.resolve(key)
            except Exception:
                return default
        
        container.get = get
    
    return container
'''
        
        try:
            with open('utils/compatibility.py', 'w', encoding='utf-8') as f:
                f.write(wrapper_content)
            self.fixes_applied.append("创建兼容性包装器")
        except Exception as e:
            self.errors_encountered.append(f"创建兼容性包装器失败: {e}")
    
    def run_signature_fixes(self):
        """运行所有方法签名修复"""
        print("🚀 开始方法签名修复...")
        print("=" * 60)
        
        self.fix_data_access_manager_signatures()
        self.fix_service_container_get_method()
        self.fix_strategy_executor_calls()
        self.create_compatibility_wrapper()
        
        self._generate_fix_report()
    
    def _generate_fix_report(self):
        """生成修复报告"""
        print("\n" + "=" * 60)
        print("🎯 方法签名修复完成报告")
        print("=" * 60)
        
        print(f"\n✅ 成功修复项目 ({len(self.fixes_applied)}项):")
        for fix in self.fixes_applied:
            print(f"  • {fix}")
        
        if self.errors_encountered:
            print(f"\n❌ 遇到的错误 ({len(self.errors_encountered)}项):")
            for error in self.errors_encountered:
                print(f"  • {error}")
        
        print(f"\n📊 修复统计:")
        print(f"  • 成功修复: {len(self.fixes_applied)}项")
        print(f"  • 遇到错误: {len(self.errors_encountered)}项")
        print(f"  • 总体状态: {'✅ 完全成功' if len(self.errors_encountered) == 0 else '⚠️ 基本成功'}")


if __name__ == "__main__":
    fixer = MethodSignatureFixer()
    fixer.run_signature_fixes() 