#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复剩余的数据库兼容性问题
扫描并修复所有包含不存在字段的代码
"""

import os
import re
import shutil
from typing import List, Dict, Any
from datetime import datetime
from db.sql_manager import SQLManager, QueryType

class RemainingDBIssuesFixer:
    """剩余数据库问题修复器"""
    
    def __init__(self):
        self.backup_dir = f"backup/remaining_db_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.invalid_fields = ['price_change', 'price_range', 'industry']
        self.fixed_files = []
        
    def fix_all_remaining_issues(self) -> Dict[str, Any]:
        """修复所有剩余的数据库兼容性问题"""
        print("🔧 开始修复剩余的数据库兼容性问题")
        print("=" * 50)
        
        # 创建备份目录
        os.makedirs(self.backup_dir, exist_ok=True)
        
        results = {
            "fix_type": "REMAINING_DB_COMPATIBILITY_FIX",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fixes_applied": {},
            "summary": {}
        }
        
        # 扫描并修复所有Python文件
        python_files = self.find_python_files_with_db_issues()
        
        for file_path in python_files:
            print(f"📝 修复文件: {file_path}")
            fix_result = self.fix_file_db_issues(file_path)
            if fix_result["modifications"]:
                results["fixes_applied"][file_path] = fix_result
        
        # 计算总结
        self._calculate_summary(results)
        
        return results
    
    def find_python_files_with_db_issues(self) -> List[str]:
        """查找包含数据库问题的Python文件"""
        problem_files = []
        
        # 扫描目录
        scan_dirs = ['db', 'utils', 'analysis', 'indicators', 'strategy']
        
        for scan_dir in scan_dirs:
            if os.path.exists(scan_dir):
                for root, dirs, files in os.walk(scan_dir):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            if self.file_contains_invalid_fields(file_path):
                                problem_files.append(file_path)
        
        return problem_files
    
    def file_contains_invalid_fields(self, file_path: str) -> bool:
        """检查文件是否包含无效字段"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for invalid_field in self.invalid_fields:
                # 检查各种可能的引用方式
                patterns = [
                    rf"['\"]?{invalid_field}['\"]?",
                    rf"{invalid_field}\s*[=:,]",
                    rf"\.{invalid_field}",
                    rf"{invalid_field}\s*\)",
                ]
                
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def fix_file_db_issues(self, file_path: str) -> Dict[str, Any]:
        """修复单个文件的数据库问题"""
        fix_result = {
            "file_path": file_path,
            "backup_created": False,
            "modifications": [],
            "issues_fixed": 0
        }
        
        try:
            # 读取原文件
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 创建备份
            backup_path = os.path.join(self.backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            fix_result["backup_created"] = True
            
            # 修复内容
            modified_content = original_content
            
            # 1. 修复字典键中的无效字段
            modified_content, dict_fixes = self.fix_dictionary_keys(modified_content)
            if dict_fixes > 0:
                fix_result["modifications"].append(f"修复字典键: {dict_fixes}处")
                fix_result["issues_fixed"] += dict_fixes
            
            # 2. 修复列表中的无效字段
            modified_content, list_fixes = self.fix_list_items(modified_content)
            if list_fixes > 0:
                fix_result["modifications"].append(f"修复列表项: {list_fixes}处")
                fix_result["issues_fixed"] += list_fixes
            
            # 3. 修复SQL查询中的无效字段
            modified_content, sql_fixes = self.fix_sql_queries(modified_content)
            if sql_fixes > 0:
                fix_result["modifications"].append(f"修复SQL查询: {sql_fixes}处")
                fix_result["issues_fixed"] += sql_fixes
            
            # 4. 修复变量赋值中的无效字段
            modified_content, var_fixes = self.fix_variable_assignments(modified_content)
            if var_fixes > 0:
                fix_result["modifications"].append(f"修复变量赋值: {var_fixes}处")
                fix_result["issues_fixed"] += var_fixes
            
            # 5. 修复函数参数中的无效字段
            modified_content, param_fixes = self.fix_function_parameters(modified_content)
            if param_fixes > 0:
                fix_result["modifications"].append(f"修复函数参数: {param_fixes}处")
                fix_result["issues_fixed"] += param_fixes
            
            # 如果有修改，写入文件
            if modified_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                self.fixed_files.append(file_path)
                print(f"  ✅ 已修复: {len(fix_result['modifications'])}项修改")
            else:
                print(f"  ℹ️ 无需修改")
            
        except Exception as e:
            fix_result["error"] = str(e)
            print(f"  ❌ 修复失败: {e}")
        
        return fix_result
    
    def fix_dictionary_keys(self, content: str) -> tuple:
        """修复字典键中的无效字段"""
        fixes = 0
        
        for invalid_field in self.invalid_fields:
            # 修复字典键
            patterns = [
                (rf"['\"]?{invalid_field}['\"]?\s*:\s*[^,}}]+,?\s*", ""),
                (rf",\s*['\"]?{invalid_field}['\"]?\s*:\s*[^,}}]+", ""),
            ]
            
            for pattern, replacement in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                    fixes += 1
        
        return content, fixes
    
    def fix_list_items(self, content: str) -> tuple:
        """修复列表项中的无效字段"""
        fixes = 0
        
        for invalid_field in self.invalid_fields:
            # 修复列表项
            patterns = [
                (rf"['\"]?{invalid_field}['\"]?,?\s*", ""),
                (rf",\s*['\"]?{invalid_field}['\"]?", ""),
            ]
            
            for pattern, replacement in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                    fixes += 1
        
        return content, fixes
    
    def fix_sql_queries(self, content: str) -> tuple:
        """修复SQL查询中的无效字段"""
        fixes = 0
        
        for invalid_field in self.invalid_fields:
            # 修复SELECT语句
            patterns = [
                (rf"SELECT\s+([^FROM]*),\s*{invalid_field}\s*([^FROM]*)\s+FROM", r"SELECT \1\2 FROM"),
                (rf"SELECT\s+{invalid_field}\s*,\s*([^FROM]*)\s+FROM", r"SELECT \1 FROM"),
                (rf"SELECT\s+([^FROM]*)\s*{invalid_field}\s*([^FROM]*)\s+FROM", r"SELECT \1\2 FROM"),
            ]
            
            for pattern, replacement in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                    fixes += 1
        
        return content, fixes
    
    def fix_variable_assignments(self, content: str) -> tuple:
        """修复变量赋值中的无效字段"""
        fixes = 0
        
        for invalid_field in self.invalid_fields:
            # 修复变量赋值
            patterns = [
                (rf"params\[\s*['\"]?{invalid_field}['\"]?\s*\]\s*=\s*[^;\n]+", ""),
                (rf"{invalid_field}\s*=\s*[^;\n]+", ""),
            ]
            
            for pattern, replacement in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                    fixes += 1
        
        return content, fixes
    
    def fix_function_parameters(self, content: str) -> tuple:
        """修复函数参数中的无效字段"""
        fixes = 0
        
        for invalid_field in self.invalid_fields:
            # 修复函数参数
            patterns = [
                (rf"def\s+\w+\([^)]*{invalid_field}[^)]*\):", lambda m: m.group(0).replace(invalid_field, "")),
                (rf"def\s+\w+\([^)]*,\s*{invalid_field}\s*[=:]?[^,)]*", lambda m: m.group(0).replace(f", {invalid_field}", "")),
            ]
            
            for pattern, replacement in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    if callable(replacement):
                        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                    else:
                        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                    fixes += 1
        
        return content, fixes
    
    def _calculate_summary(self, results: Dict[str, Any]):
        """计算修复总结"""
        summary = results["summary"]
        
        total_files = len(results["fixes_applied"])
        total_fixes = sum(fix_result["issues_fixed"] for fix_result in results["fixes_applied"].values())
        
        summary["files_fixed"] = total_files
        summary["total_fixes"] = total_fixes
        summary["backup_directory"] = self.backup_dir
        summary["fixed_files"] = self.fixed_files

def main():
    """主函数"""
    print("🔧 剩余数据库兼容性问题修复")
    print("=" * 50)
    
    fixer = RemainingDBIssuesFixer()
    results = fixer.fix_all_remaining_issues()
    
    # 显示修复结果
    print(f"\n📊 修复摘要:")
    print(f"修复文件数: {results['summary']['files_fixed']}")
    print(f"总修复数: {results['summary']['total_fixes']}")
    
    # 显示修复详情
    if results["fixes_applied"]:
        print(f"\n📋 修复详情:")
        for file_path, fix_result in results["fixes_applied"].items():
            print(f"  ✅ {file_path}:")
            for mod in fix_result["modifications"]:
                print(f"    - {mod}")
    
    print(f"\n💾 备份目录: {results['summary']['backup_directory']}")
    print(f"\n✅ 剩余数据库兼容性问题修复完成")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
