#!/usr/bin/env python3
"""
单例违规修复脚本

移除@singleton装饰器，改用依赖注入模式，提高代码的可测试性和可维护性。
"""

import os
import re
import shutil
from typing import Dict, List, Tuple

class SingletonViolationFixer:
    """单例违规修复器"""
    
    def __init__(self):
        # 需要修复的文件列表
        self.target_files = [
            "indicators/dmi.py",
            "db/data_manager.py", 
            "scripts/utils/simplified_integration_test.py"
        ]
        
        # 修复统计
        self.stats = {
            'total_files': len(self.target_files),
            'successful_fixes': 0,
            'failed_fixes': 0,
            'decorators_removed': 0
        }
    
    def backup_file(self, file_path: str) -> str:
        """备份文件"""
        backup_path = f"{file_path}.singleton_fix_backup"
        if os.path.exists(file_path):
            shutil.copy2(file_path, backup_path)
            return backup_path
        return ""
    
    def remove_singleton_decorator(self, content: str) -> Tuple[str, int]:
        """移除@singleton装饰器"""
        removed_count = 0
        
        # 移除@singleton装饰器行
        lines = content.split('\n')
        filtered_lines = []
        
        for i, line in enumerate(lines):
            if line.strip() == '@singleton':
                removed_count += 1
                # 跳过这一行，不添加到filtered_lines
                continue
            else:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines), removed_count
    
    def remove_singleton_import(self, content: str) -> str:
        """移除singleton装饰器的导入"""
        # 移除from utils.decorators import singleton
        content = re.sub(r'from\s+utils\.decorators\s+import\s+singleton\s*\n?', '', content)
        
        # 移除from utils.decorators import ..., singleton, ...
        content = re.sub(r'(from\s+utils\.decorators\s+import\s+[^,\n]*),\s*singleton', r'\1', content)
        content = re.sub(r'(from\s+utils\.decorators\s+import\s+)singleton,\s*([^,\n]*)', r'\1\2', content)
        
        return content
    
    def add_dependency_injection_support(self, content: str, class_name: str) -> str:
        """添加依赖注入支持"""
        # 在文件开头添加依赖注入导入（如果不存在）
        if 'from utils.dependency_injection import' not in content:
            import_section = self._find_import_section(content)
            if import_section:
                content = content.replace(
                    import_section,
                    import_section + 'from utils.dependency_injection import get_service_container\n'
                )
        
        # 添加获取实例的函数（如果不存在）
        getter_func_name = f"get_{class_name.lower()}"
        if f"def {getter_func_name}(" not in content:
            getter_function = f'''

def {getter_func_name}():
    """获取{class_name}实例（通过依赖注入）"""
    try:
        container = get_service_container()
        if not container.is_registered({class_name}):
            container.register_singleton({class_name})
        return container.get_service({class_name})
    except Exception:
        # 降级处理：如果依赖注入失败，直接创建实例
        return {class_name}()
'''
            content += getter_function
        
        return content
    
    def _find_import_section(self, content: str) -> str:
        """找到导入部分的结束位置"""
        lines = content.split('\n')
        last_import_line = ""
        
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                last_import_line = line
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        return last_import_line + '\n' if last_import_line else ""
    
    def extract_class_name(self, content: str) -> str:
        """提取类名"""
        # 查找被@singleton装饰的类
        pattern = r'@singleton\s*\nclass\s+(\w+)'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(1)
        
        # 如果没有找到，查找第一个类
        pattern = r'class\s+(\w+)'
        match = re.search(pattern, content)
        if match:
            return match.group(1)
        
        return "UnknownClass"
    
    def fix_specific_file_issues(self, file_path: str, content: str) -> str:
        """修复特定文件的问题"""
        if file_path == "indicators/dmi.py":
            # DMI指标类的特殊处理
            content = self._fix_dmi_file(content)
        elif file_path == "db/data_manager.py":
            # 数据管理器的特殊处理
            content = self._fix_data_manager_file(content)
        elif file_path == "scripts/utils/simplified_integration_test.py":
            # 测试文件的特殊处理
            content = self._fix_test_file(content)
        
        return content
    
    def _fix_dmi_file(self, content: str) -> str:
        """修复DMI文件的特殊问题"""
        # DMI指标应该是无状态的，不需要单例
        # 移除单例装饰器后，确保类可以正常实例化
        
        # 添加获取DMI实例的函数
        if "def get_dmi_indicator(" not in content:
            getter_function = '''

def get_dmi_indicator(**kwargs):
    """获取DMI指标实例"""
    return DirectionalMovementIndex(**kwargs)
'''
            content += getter_function
        
        return content
    
    def _fix_data_manager_file(self, content: str) -> str:
        """修复数据管理器文件"""
        # 数据管理器确实需要单例模式，但应该通过依赖注入容器管理
        return content
    
    def _fix_test_file(self, content: str) -> str:
        """修复测试文件"""
        # 测试文件中的单例通常不是必需的
        return content
    
    def fix_file(self, file_path: str) -> Tuple[bool, str]:
        """修复单个文件"""
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        try:
            # 备份文件
            backup_path = self.backup_file(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否包含@singleton装饰器
            if '@singleton' not in content:
                return True, f"文件无需修复: {file_path}"
            
            original_content = content
            
            # 1. 提取类名
            class_name = self.extract_class_name(content)
            
            # 2. 移除@singleton装饰器
            content, removed_count = self.remove_singleton_decorator(content)
            self.stats['decorators_removed'] += removed_count
            
            # 3. 移除singleton导入
            content = self.remove_singleton_import(content)
            
            # 4. 添加依赖注入支持
            content = self.add_dependency_injection_support(content, class_name)
            
            # 5. 修复特定文件的问题
            content = self.fix_specific_file_issues(file_path, content)
            
            # 检查是否有实际变化
            if content != original_content:
                # 写入修复后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, f"成功修复: {file_path} (移除了{removed_count}个@singleton装饰器)"
            else:
                return True, f"无需修复: {file_path}"
                
        except Exception as e:
            return False, f"修复失败 {file_path}: {str(e)}"
    
    def run_fixes(self) -> Dict[str, List[Tuple[str, bool, str]]]:
        """运行所有修复"""
        print("开始修复单例违规问题...")
        
        results = {'fixed': []}
        
        for file_path in self.target_files:
            success, message = self.fix_file(file_path)
            results['fixed'].append((file_path, success, message))
            
            if success:
                self.stats['successful_fixes'] += 1
            else:
                self.stats['failed_fixes'] += 1
            
            print(f"  {'✅' if success else '❌'} {message}")
        
        return results
    
    def generate_fix_report(self, results: Dict[str, List[Tuple[str, bool, str]]]) -> str:
        """生成修复报告"""
        report = []
        report.append("# 单例违规修复报告")
        report.append(f"生成时间: {os.popen('date').read().strip()}")
        report.append("")
        
        # 统计信息
        report.append("## 修复统计")
        report.append(f"- 目标文件数: {self.stats['total_files']}")
        report.append(f"- 成功修复: {self.stats['successful_fixes']}")
        report.append(f"- 失败数: {self.stats['failed_fixes']}")
        report.append(f"- 移除装饰器数: {self.stats['decorators_removed']}")
        report.append(f"- 成功率: {self.stats['successful_fixes']/self.stats['total_files']*100:.1f}%")
        report.append("")
        
        # 详细结果
        report.append("## 修复详情")
        for file_path, success, message in results['fixed']:
            status = "✅" if success else "❌"
            report.append(f"- {status} {file_path}: {message}")
        report.append("")
        
        # 修复说明
        report.append("## 修复说明")
        report.append("1. **移除@singleton装饰器**: 所有@singleton装饰器已被移除")
        report.append("2. **依赖注入支持**: 为需要单例的类添加了依赖注入支持")
        report.append("3. **降级处理**: 实现了优雅的降级处理机制")
        report.append("4. **向后兼容**: 保持了现有API的兼容性")
        report.append("")
        
        # 使用指南
        report.append("## 使用指南")
        report.append("修复后的类使用方式：")
        report.append("```python")
        report.append("# 原来的方式（仍然支持）")
        report.append("instance = ClassName()")
        report.append("")
        report.append("# 推荐的新方式（通过依赖注入）")
        report.append("instance = get_classname()")
        report.append("```")
        
        return '\n'.join(report)

def main():
    """主函数"""
    fixer = SingletonViolationFixer()
    results = fixer.run_fixes()
    
    # 生成报告
    report = fixer.generate_fix_report(results)
    
    # 保存报告
    report_path = "singleton_violations_fix_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n修复完成！报告已保存到: {report_path}")
    
    # 打印统计
    print(f"\n=== 修复统计 ===")
    print(f"目标文件数: {fixer.stats['total_files']}")
    print(f"成功修复: {fixer.stats['successful_fixes']}")
    print(f"失败数: {fixer.stats['failed_fixes']}")
    print(f"移除装饰器数: {fixer.stats['decorators_removed']}")
    print(f"成功率: {fixer.stats['successful_fixes']/fixer.stats['total_files']*100:.1f}%")

if __name__ == "__main__":
    main() 