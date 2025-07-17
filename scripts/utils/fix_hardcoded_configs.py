#!/usr/bin/env python3
"""
自动修复硬编码配置问题

扫描代码中的硬编码配置，并替换为统一配置管理系统的调用。
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class HardcodedConfigFixerFix_Hardcoded_Configs:
    """硬编码配置修复器"""
    
    def __init__(self):
        self.fixed_files = []
        self.patterns = self._init_patterns()
        self.replacements = self._init_replacements()
    
    def _init_patterns(self) -> Dict[str, re.Pattern]:
        """初始化匹配模式"""
        return {
            'host_localhost': re.compile(r"host\s*=\s*['\"]localhost['\"]"),
            'host_127': re.compile(r"host\s*=\s*['\"]127\.0\.0\.1['\"]"),
            'port_8123': re.compile(r"port\s*=\s*8123"),
            'port_6379': re.compile(r"port\s*=\s*6379"),
            'database_stock': re.compile(r"database\s*=\s*['\"]stock_\w+['\"]"),
            'database_test': re.compile(r"database\s*=\s*['\"]test_\w+['\"]"),
            'password_empty': re.compile(r"password\s*=\s*['\"]['\"]"),
            'username_default': re.compile(r"username\s*=\s*['\"]default['\"]"),
            'timeout_30': re.compile(r"timeout\s*=\s*30"),
            'timeout_60': re.compile(r"timeout\s*=\s*60"),
            'level_daily': re.compile(r"level\s*=\s*['\"]日线['\"]"),
            'level_weekly': re.compile(r"level\s*=\s*['\"]周线['\"]"),
            'level_monthly': re.compile(r"level\s*=\s*['\"]月线['\"]"),
        }
    
    def _init_replacements(self) -> Dict[str, str]:
        """初始化替换模式"""
        return {
            'host_localhost': "host=get_config_value('database.host', 'localhost')",
            'host_127': "host=get_config_value('database.host', '127.0.0.1')",
            'port_8123': "port=get_config_value('database.port', 8123)",
            'port_6379': "port=get_config_value('redis.port', 6379)",
            'database_stock': "database=get_config_value('database.database', 'stock_data')",
            'database_test': "database=get_config_value('test.test_db_database', 'test_stock_data')",
            'password_empty': "password=get_config_value('database.password', '')",
            'username_default': "username=get_config_value('database.username', 'default')",
            'timeout_30': "timeout=get_config_value('database.timeout', 30)",
            'timeout_60': "timeout=get_config_value('test.test_timeout', 60)",
            'level_daily': "level=get_config_value('system.default_level', '日线')",
            'level_weekly': "level=get_config_value('system.default_level', '周线')",
            'level_monthly': "level=get_config_value('system.default_level', '月线')",
        }
    
    def scan_file(self, file_path: str) -> List[Tuple[str, int, str]]:
        """扫描文件中的硬编码配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Tuple[str, int, str]]: 发现的问题列表 (类型, 行号, 内容)
        """
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                for pattern_name, pattern in self.patterns.items():
                    if pattern.search(line):
                        issues.append((pattern_name, line_num, line.strip()))
        
        except Exception as e:
            logger.error(f"扫描文件失败 {file_path}: {e}")
        
        return issues
    
    def fix_file_fix_hardcoded_configs(self, file_path: str) -> bool:
        """修复文件中的硬编码配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否有修改
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # 检查是否需要添加导入
            needs_import = False
            for pattern_name, pattern in self.patterns.items():
                if pattern.search(content):
                    needs_import = True
                    break
            
            # 添加导入语句
            if needs_import and 'from config.unified_config import get_config_value' not in content:
                # 找到合适的位置插入导入语句
                import_pos = self._find_import_position(content)
                if import_pos is not None:
                    import_line = "from config.unified_config import get_config_value\n"
                    content = content[:import_pos] + import_line + content[import_pos:]
                    modified = True
            
            # 应用替换
            for pattern_name, pattern in self.patterns.items():
                if pattern_name in self.replacements:
                    new_content = pattern.sub(self.replacements[pattern_name], content)
                    if new_content != content:
                        content = new_content
                        modified = True
            
            # 如果有修改，写回文件
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"修复文件: {file_path}")
                self.fixed_files.append(file_path)
                return True
        
        except Exception as e:
            logger.error(f"修复文件失败 {file_path}: {e}")
        
        return False
    
    def _find_import_position(self, content: str) -> int:
        """找到合适的导入位置
        
        Args:
            content: 文件内容
            
        Returns:
            int: 插入位置，如果找不到则返回None
        """
        lines = content.split('\n')
        
        # 找到最后一个import语句的位置
        last_import_line = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                last_import_line = i
        
        if last_import_line >= 0:
            # 在最后一个import语句后插入
            pos = 0
            for i in range(last_import_line + 1):
                pos += len(lines[i]) + 1  # +1 for newline
            return pos
        
        # 如果没有找到import语句，在文件开头插入
        return 0
    
    def scan_directory_fix_hardcoded_configs(self, directory: str, extensions: Set[str] = None) -> Dict[str, List[Tuple[str, int, str]]]:
        """扫描目录中的硬编码配置
        
        Args:
            directory: 目录路径
            extensions: 文件扩展名集合
            
        Returns:
            Dict[str, List[Tuple[str, int, str]]]: 文件路径到问题列表的映射
        """
        if extensions is None:
            extensions = {'.py'}
        
        issues = {}
        
        for root, dirs, files in os.walk(directory):
            # 跳过特定目录
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'venv', 'env'}]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    file_issues = self.scan_file(file_path)
                    if file_issues:
                        issues[file_path] = file_issues
        
        return issues
    
    def fix_directory(self, directory: str, extensions: Set[str] = None) -> int:
        """修复目录中的硬编码配置
        
        Args:
            directory: 目录路径
            extensions: 文件扩展名集合
            
        Returns:
            int: 修复的文件数量
        """
        if extensions is None:
            extensions = {'.py'}
        
        fixed_count = 0
        
        for root, dirs, files in os.walk(directory):
            # 跳过特定目录
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'venv', 'env'}]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    if self.fix_file(file_path):
                        fixed_count += 1
        
        return fixed_count
    
    def generate_report_fix_hardcoded_configs(self, issues: Dict[str, List[Tuple[str, int, str]]]) -> str:
        """生成扫描报告
        
        Args:
            issues: 问题字典
            
        Returns:
            str: 报告内容
        """
        report = ["硬编码配置扫描报告", "=" * 50, ""]
        
        total_issues = sum(len(file_issues) for file_issues in issues.values())
        report.append(f"总计发现 {total_issues} 个硬编码配置问题")
        report.append(f"涉及 {len(issues)} 个文件")
        report.append("")
        
        for file_path, file_issues in issues.items():
            report.append(f"文件: {file_path}")
            report.append("-" * 40)
            
            for pattern_name, line_num, content in file_issues:
                report.append(f"  行 {line_num}: {pattern_name}")
                report.append(f"    {content}")
            
            report.append("")
        
        return "\n".join(report)

def main_fix_hardcoded_configs():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python fix_hardcoded_configs.py <scan|fix> [directory]")
        sys.exit(1)
    
    action = sys.argv[1]
    directory = sys.argv[2] if len(sys.argv) > 2 else root_dir
    
    fixer = HardcodedConfigFixer()
    
    if action == 'scan':
        print(f"扫描目录: {directory}")
        issues = fixer.scan_directory(directory)
        
        if issues:
            report = fixer.generate_report(issues)
            print(report)
            
            # 保存报告
            report_file = os.path.join(root_dir, 'hardcoded_config_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存到: {report_file}")
        else:
            print("未发现硬编码配置问题")
    
    elif action == 'fix':
        print(f"修复目录: {directory}")
        fixed_count = fixer.fix_directory(directory)
        
        if fixed_count > 0:
            print(f"成功修复 {fixed_count} 个文件")
            print("修复的文件:")
            for file_path in fixer.fixed_files:
                print(f"  - {file_path}")
        else:
            print("未发现需要修复的文件")
    
    else:
        print("无效的操作，请使用 'scan' 或 'fix'")
        sys.exit(1)

if __name__ == "__main__":
    main_fix_hardcoded_configs() 