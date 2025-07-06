#!/usr/bin/env python3
"""
自动修复剩余硬编码配置问题
"""
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from config.unified_config import UnifiedConfigManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardcodedConfigFixer:
    """硬编码配置修复器"""
    
    def __init__(self):
        self.config_manager = UnifiedConfigManager()
        self.fixes_applied = 0
        self.files_modified = set()
        
        # 硬编码配置模式和替换规则
        self.config_patterns = {
            # ClickHouse连接配置
            r"host\s*=\s*['\"]localhost['\"]": "host=self.config_manager.get_database_config().get('host', 'localhost')",
            r"port\s*=\s*8123": "port=self.config_manager.get_database_config().get('port', 8123)",
            r"port\s*=\s*9000": "port=self.config_manager.get_database_config().get('port', 9000)",
            r"database\s*=\s*['\"]stock['\"]": "database=self.config_manager.get_database_config().get('database', 'stock')",
            r"user\s*=\s*['\"]default['\"]": "user=self.config_manager.get_database_config().get('user', 'default')",
            r"password\s*=\s*['\"]['\"]": "password=self.config_manager.get_database_config().get('password', '')",
            
            # Redis连接配置
            r"redis\.Redis\(\s*host\s*=\s*['\"]localhost['\"]": "redis.Redis(host=self.config_manager.get_redis_config().get('host', 'localhost')",
            r"redis\.Redis\(\s*port\s*=\s*6379": "redis.Redis(port=self.config_manager.get_redis_config().get('port', 6379)",
            
            # 测试配置
            r"test_host\s*=\s*['\"]localhost['\"]": "test_host=self.config_manager.get_test_config().get('host', 'localhost')",
            r"test_port\s*=\s*8123": "test_port=self.config_manager.get_test_config().get('port', 8123)",
            r"test_database\s*=\s*['\"]stock['\"]": "test_database=self.config_manager.get_test_config().get('database', 'stock')",
        }
        
        # 需要添加的导入语句
        self.import_statements = {
            "from config.unified_config import UnifiedConfigManager",
            "import os",
            "import sys"
        }
    
    def fix_file(self, file_path: str) -> bool:
        """修复单个文件的硬编码配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # 检查是否需要添加配置管理器导入
            needs_config_import = False
            
            # 应用配置模式替换
            for pattern, replacement in self.config_patterns.items():
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True
                    needs_config_import = True
            
            # 如果需要配置管理器，添加相关代码
            if needs_config_import:
                content = self._add_config_manager_setup(content, file_path)
                modified = True
            
            # 如果内容被修改，写回文件
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_modified.add(file_path)
                logger.info(f"修复文件: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"修复文件 {file_path} 失败: {e}")
            return False
    
    def _add_config_manager_setup(self, content: str, file_path: str) -> str:
        """添加配置管理器设置代码"""
        lines = content.split('\n')
        
        # 查找导入区域的结束位置
        import_end_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_end_idx = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        # 检查是否已经有配置管理器导入
        has_config_import = any('UnifiedConfigManager' in line for line in lines)
        
        if not has_config_import:
            # 添加必要的导入
            new_imports = [
                "from config.unified_config import UnifiedConfigManager"
            ]
            
            # 在导入区域末尾添加新导入
            for imp in reversed(new_imports):
                lines.insert(import_end_idx, imp)
                import_end_idx += 1
            
            # 添加空行分隔
            lines.insert(import_end_idx, "")
        
        # 查找类定义或主要逻辑开始位置，添加配置管理器初始化
        for i, line in enumerate(lines):
            if (line.strip().startswith('class ') or 
                line.strip().startswith('def main') or
                line.strip().startswith('if __name__')):
                
                # 在类或函数内部添加配置管理器初始化
                if line.strip().startswith('class '):
                    # 查找__init__方法
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith('def __init__'):
                            # 在__init__方法中添加配置管理器
                            init_indent = len(lines[j]) - len(lines[j].lstrip())
                            config_line = " " * (init_indent + 4) + "self.config_manager = UnifiedConfigManager()"
                            
                            # 查找__init__方法的结束位置
                            for k in range(j + 1, len(lines)):
                                if (lines[k].strip() and 
                                    not lines[k].startswith(' ' * (init_indent + 4))):
                                    lines.insert(k, config_line)
                                    break
                            break
                elif line.strip().startswith('def main') or line.strip().startswith('if __name__'):
                    # 在主函数开始处添加配置管理器
                    func_indent = len(line) - len(line.lstrip())
                    config_line = " " * (func_indent + 4) + "config_manager = UnifiedConfigManager()"
                    lines.insert(i + 1, config_line)
                break
        
        return '\n'.join(lines)
    
    def scan_and_fix_files(self) -> None:
        """扫描并修复所有文件的硬编码配置"""
        logger.info("开始扫描和修复硬编码配置...")
        
        # 目标文件列表（从检查结果中获取）
        target_files = [
            "tests/end_to_end/real_data_performance_test.py",
            "tests/optimization/comprehensive_optimization_test.py", 
            "tests/performance/simple_concurrent_test.py",
            "tests/performance/concurrent_optimization_test.py",
            "tests/performance/advanced_concurrent_test.py",
            "tests/performance/batch_processing_test.py",
            "tests/performance/memory_optimization_test.py",
            "scripts/simple_clickhouse_test.py",
            "scripts/production_database_test.py",
            "scripts/clickhouse_connection_summary.py"
        ]
        
        for file_path in target_files:
            full_path = os.path.join(root_dir, file_path)
            if os.path.exists(full_path):
                if self.fix_file(full_path):
                    self.fixes_applied += 1
            else:
                logger.warning(f"文件不存在: {full_path}")
        
        logger.info(f"修复完成，共修复 {self.fixes_applied} 个文件")
        logger.info(f"修改的文件: {list(self.files_modified)}")

def main():
    """主函数"""
    logger.info("开始硬编码配置修复...")
    
    fixer = HardcodedConfigFixer()
    fixer.scan_and_fix_files()
    
    logger.info("硬编码配置修复完成！")
    
    # 运行架构检查验证修复效果
    logger.info("运行架构检查验证修复效果...")
    os.system("python simple_architecture_check.py")

if __name__ == "__main__":
    main() 