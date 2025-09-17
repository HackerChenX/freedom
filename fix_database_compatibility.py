#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复数据库兼容性问题
移除不存在的字段：price_change, price_range, industry
"""

import os
import re
import shutil
from typing import List, Dict, Any
from datetime import datetime
from db.sql_manager import SQLManager, QueryType

class DatabaseCompatibilityFixer:
    """数据库兼容性修复器"""
    
    def __init__(self):
        self.backup_dir = f"backup/db_compatibility_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.fixed_files = []
        self.fix_stats = {
            'files_processed': 0,
            'files_modified': 0,
            'queries_fixed': 0,
            'fields_removed': 0
        }
        
        # 不存在的字段
        self.invalid_fields = ['price_change', 'price_range', 'industry']
        
        # 需要修复的文件列表
        self.target_files = [
            'analysis/buypoints/analyze_buypoints.py',
            'db/managers/data_access_manager.py',
            'db/sql_manager.py',
            'db/query_optimizer.py',
            'utils/unified_query_builder.py'
        ]
    
    def fix_all_compatibility_issues(self) -> Dict[str, Any]:
        """修复所有数据库兼容性问题"""
        print("🔧 开始修复数据库兼容性问题")
        print("=" * 50)
        
        # 创建备份目录
        os.makedirs(self.backup_dir, exist_ok=True)
        
        results = {
            "fix_type": "DATABASE_COMPATIBILITY_FIX",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "invalid_fields": self.invalid_fields,
            "fixes_applied": {},
            "summary": {}
        }
        
        # 修复各个文件
        for file_path in self.target_files:
            if os.path.exists(file_path):
                print(f"📝 修复文件: {file_path}")
                fix_result = self.fix_file(file_path)
                results["fixes_applied"][file_path] = fix_result
            else:
                print(f"⚠️ 文件不存在: {file_path}")
        
        # 删除废弃的买点分析文件
        results["fixes_applied"]["deprecated_cleanup"] = self.cleanup_deprecated_files()
        
        # 计算总结
        self._calculate_summary(results)
        
        return results
    
    def fix_file(self, file_path: str) -> Dict[str, Any]:
        """修复单个文件"""
        fix_result = {
            "file_path": file_path,
            "backup_created": False,
            "modifications": [],
            "queries_fixed": 0,
            "fields_removed": 0
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
            
            # 1. 移除SELECT语句中的无效字段
            modified_content, select_fixes = self.fix_select_statements(modified_content)
            if select_fixes > 0:
                fix_result["modifications"].append(f"修复SELECT语句中的无效字段: {select_fixes}处")
                fix_result["queries_fixed"] += select_fixes
            
            # 2. 移除WHERE条件中的无效字段
            modified_content, where_fixes = self.fix_where_conditions(modified_content)
            if where_fixes > 0:
                fix_result["modifications"].append(f"修复WHERE条件中的无效字段: {where_fixes}处")
                fix_result["queries_fixed"] += where_fixes
            
            # 3. 移除ORDER BY中的无效字段
            modified_content, order_fixes = self.fix_order_by_clauses(modified_content)
            if order_fixes > 0:
                fix_result["modifications"].append(f"修复ORDER BY中的无效字段: {order_fixes}处")
                fix_result["queries_fixed"] += order_fixes
            
            # 4. 修复字段映射和配置
            modified_content, mapping_fixes = self.fix_field_mappings(modified_content)
            if mapping_fixes > 0:
                fix_result["modifications"].append(f"修复字段映射配置: {mapping_fixes}处")
                fix_result["fields_removed"] += mapping_fixes
            
            # 如果有修改，写入文件
            if modified_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                self.fixed_files.append(file_path)
                self.fix_stats['files_modified'] += 1
                print(f"  ✅ 已修复: {len(fix_result['modifications'])}项修改")
            else:
                print(f"  ℹ️ 无需修改")
            
            self.fix_stats['files_processed'] += 1
            self.fix_stats['queries_fixed'] += fix_result['queries_fixed']
            self.fix_stats['fields_removed'] += fix_result['fields_removed']
            
        except Exception as e:
            fix_result["error"] = str(e)
            print(f"  ❌ 修复失败: {e}")
        
        return fix_result
    
    def fix_select_statements(self, content: str) -> tuple:
        """修复SELECT语句中的无效字段"""
        fixes = 0
        
        # 匹配SELECT语句中的字段列表
        select_pattern = r'SELECT\s+([^FROM]+)\s+FROM'
        
        def fix_select_fields(match):
            nonlocal fixes
            fields_part = match.group(1)
            
            # 检查是否包含无效字段
            has_invalid = any(field in fields_part for field in self.invalid_fields)
            if not has_invalid:
                return match.group(0)
            
            # 移除无效字段
            lines = fields_part.split('\n')
            fixed_lines = []
            
            for line in lines:
                line_fixed = line
                for invalid_field in self.invalid_fields:
                    # 移除字段（包括前后的逗号和空格）
                    patterns = [
                        rf',\s*{invalid_field}\s*',  # 中间的字段
                        rf'{invalid_field}\s*,\s*',  # 开头的字段
                        rf'\s*{invalid_field}\s*$',  # 结尾的字段
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, line_fixed, re.IGNORECASE):
                            line_fixed = re.sub(pattern, '', line_fixed, flags=re.IGNORECASE)
                            break
                
                # 清理多余的逗号和空格
                line_fixed = re.sub(r',\s*,', ',', line_fixed)
                line_fixed = re.sub(r'^\s*,\s*', '', line_fixed)
                line_fixed = re.sub(r'\s*,\s*$', '', line_fixed)
                
                if line_fixed.strip():
                    fixed_lines.append(line_fixed)
            
            if len(fixed_lines) != len(lines) or any(line != orig for line, orig in zip(fixed_lines, lines)):
                fixes += 1
            
            return f"SELECT {' '.join(fixed_lines)} FROM"
        
        modified_content = re.sub(select_pattern, fix_select_fields, content, flags=re.IGNORECASE | re.DOTALL)
        return modified_content, fixes
    
    def fix_where_conditions(self, content: str) -> tuple:
        """修复WHERE条件中的无效字段"""
        fixes = 0
        
        for invalid_field in self.invalid_fields:
            # 匹配WHERE条件中的无效字段
            patterns = [
                rf'AND\s+{invalid_field}\s*[!=<>]+[^AND\s]*',
                rf'WHERE\s+{invalid_field}\s*[!=<>]+[^AND\s]*\s*AND',
                rf'{invalid_field}\s*[!=<>]+[^AND\s]*\s*AND',
                rf'AND\s+{invalid_field}\s+IS\s+(NOT\s+)?NULL',
                rf'{invalid_field}\s*!=\s*[\'"][^\'"]*[\'"]',
            ]
            
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    content = re.sub(pattern, '', content, flags=re.IGNORECASE)
                    fixes += 1
        
        return content, fixes
    
    def fix_order_by_clauses(self, content: str) -> tuple:
        """修复ORDER BY中的无效字段"""
        fixes = 0
        
        for invalid_field in self.invalid_fields:
            # 匹配ORDER BY中的无效字段
            patterns = [
                rf'ORDER\s+BY\s+{invalid_field}\s*,?',
                rf',\s*{invalid_field}\s*(ASC|DESC)?\s*,?',
            ]
            
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    content = re.sub(pattern, '', content, flags=re.IGNORECASE)
                    fixes += 1
        
        return content, fixes
    
    def fix_field_mappings(self, content: str) -> tuple:
        """修复字段映射和配置"""
        fixes = 0
        
        for invalid_field in self.invalid_fields:
            # 移除字段映射中的无效字段
            patterns = [
                rf"['\"]?{invalid_field}['\"]?\s*:\s*['\"][^'\"]*['\"],?\s*",
                rf"'{invalid_field}':\s*'[^']*',?\s*",
                rf'"{invalid_field}":\s*"[^"]*",?\s*',
                rf"'{invalid_field}',?\s*",
                rf'"{invalid_field}",?\s*',
            ]
            
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    content = re.sub(pattern, '', content, flags=re.IGNORECASE)
                    fixes += 1
        
        return content, fixes
    
    def cleanup_deprecated_files(self) -> Dict[str, Any]:
        """清理废弃文件"""
        cleanup_result = {
            "action": "cleanup_deprecated_files",
            "files_removed": [],
            "files_backed_up": []
        }
        
        # 备份并删除有问题的买点分析文件
        deprecated_file = 'analysis/buypoints/analyze_buypoints.py'
        
        if os.path.exists(deprecated_file):
            # 创建备份
            backup_path = os.path.join(self.backup_dir, 'deprecated_analyze_buypoints.py')
            shutil.copy2(deprecated_file, backup_path)
            cleanup_result["files_backed_up"].append(deprecated_file)
            
            # 删除原文件
            os.remove(deprecated_file)
            cleanup_result["files_removed"].append(deprecated_file)
            
            print(f"🗑️ 已删除废弃文件: {deprecated_file}")
            print(f"📦 备份位置: {backup_path}")
        
        return cleanup_result
    
    def _calculate_summary(self, results: Dict[str, Any]):
        """计算修复总结"""
        summary = results["summary"]
        summary.update(self.fix_stats)
        summary["backup_directory"] = self.backup_dir
        summary["fixed_files"] = self.fixed_files
        
        # 计算修复率
        if summary["files_processed"] > 0:
            fix_rate = (summary["files_modified"] / summary["files_processed"]) * 100
            summary["fix_rate"] = f"{fix_rate:.1f}%"
        else:
            summary["fix_rate"] = "0%"

def main():
    """主函数"""
    print("🔧 数据库兼容性修复")
    print("=" * 50)
    
    fixer = DatabaseCompatibilityFixer()
    results = fixer.fix_all_compatibility_issues()
    
    # 显示修复结果
    print(f"\n📊 修复摘要:")
    print(f"处理文件: {results['summary']['files_processed']}")
    print(f"修改文件: {results['summary']['files_modified']}")
    print(f"修复查询: {results['summary']['queries_fixed']}")
    print(f"移除字段: {results['summary']['fields_removed']}")
    print(f"修复率: {results['summary']['fix_rate']}")
    
    # 显示修复详情
    print(f"\n📋 修复详情:")
    for file_path, fix_result in results["fixes_applied"].items():
        if isinstance(fix_result, dict) and "modifications" in fix_result:
            if fix_result["modifications"]:
                print(f"  ✅ {file_path}:")
                for mod in fix_result["modifications"]:
                    print(f"    - {mod}")
            else:
                print(f"  ℹ️ {file_path}: 无需修改")
    
    print(f"\n💾 备份目录: {results['summary']['backup_directory']}")
    print(f"\n✅ 数据库兼容性修复完成")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
