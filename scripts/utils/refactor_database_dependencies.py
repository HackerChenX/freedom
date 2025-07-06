#!/usr/bin/env python3
"""
数据库依赖重构脚本

自动化重构直接数据库依赖，将get_service(Data_access_interface)调用替换为依赖注入模式。
解决122个文件的直接数据库依赖问题。

Author: System Architecture Team
Date: 2025-01-15
Version: 1.0
"""

import os
import re
import sys
from typing import List, Dict, Set, Tuple
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.path_utils import get_project_root

logger = get_logger(__name__)


class DatabaseDependencyRefactor:
    """数据库依赖重构器"""
    
    def __init__(self):
        """初始化重构器"""
        self.project_root = get_project_root()
        self.files_to_refactor: List[str] = []
        self.refactor_stats = {
            'total_files': 0,
            'successful_refactors': 0,
            'failed_refactors': 0,
            'imports_replaced': 0,
            'calls_replaced': 0
        }
        
        # 重构模式定义
        self.import_patterns = [
            r'from db\.clickhouse_db from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access
            r'from db\.clickhouse_db from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access
            r'from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access
        ]
        
        self.call_patterns = [
            r'db\s*=\s*get_clickhouse_db\(\)',
            r'self\.db\s*=\s*get_clickhouse_db\(\)',
            r'cls\.db\s*=\s*get_clickhouse_db\(\)',
            r'get_clickhouse_db\(\)'
        ]
    
    def find_files_to_refactor(self) -> List[str]:
        """查找需要重构的文件"""
        files_with_db_deps = []
        
        # 遍历项目文件
        for root, dirs, files in os.walk(self.project_root):
            # 跳过某些目录
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.pytest_cache']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if self._file_has_db_dependency(file_path):
                        files_with_db_deps.append(file_path)
        
        self.files_to_refactor = files_with_db_deps
        logger.info(f"发现{len(files_with_db_deps)}个需要重构的文件")
        return files_with_db_deps
    
    def _file_has_db_dependency(self, file_path: str) -> bool:
        """检查文件是否有数据库依赖"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查导入语句
            for pattern in self.import_patterns:
                if re.search(pattern, content):
                    return True
            
            # 检查函数调用
            for pattern in self.call_patterns:
                if re.search(pattern, content):
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"检查文件失败: {file_path}, 错误: {e}")
            return False
    
    def refactor_file(self, file_path: str) -> bool:
        """重构单个文件"""
        try:
            logger.info(f"重构文件: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            content = original_content
            
            # 1. 替换导入语句
            content, import_count = self._replace_imports(content)
            
            # 2. 替换数据库调用
            content, call_count = self._replace_db_calls(content, file_path)
            
            # 3. 添加必要的导入
            content = self._add_required_imports(content)
            
            # 4. 写入文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.refactor_stats['imports_replaced'] += import_count
                self.refactor_stats['calls_replaced'] += call_count
                self.refactor_stats['successful_refactors'] += 1
                
                logger.info(f"重构完成: {file_path} (导入:{import_count}, 调用:{call_count})")
                return True
            else:
                logger.info(f"文件无需修改: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"重构文件失败: {file_path}, 错误: {e}")
            self.refactor_stats['failed_refactors'] += 1
            return False
    
    def _replace_imports(self, content: str) -> Tuple[str, int]:
        """替换导入语句"""
        import_count = 0
        
        # 替换clickhouse_db导入
        patterns_replacements = [
            (r'from db\.clickhouse_db from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access
             'from utils.dependency_injection import get_service\nfrom db.interfaces.data_access_interface import IDataAccess'),
            (r'from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access
             'from utils.dependency_injection import get_service\nfrom db.interfaces.data_access_interface import IDataAccess')
        ]
        
        for pattern, replacement in patterns_replacements:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                import_count += len(matches)
        
        return content, import_count
    
    def _replace_db_calls(self, content: str, file_path: str) -> Tuple[str, int]:
        """替换数据库调用"""
        call_count = 0
        
        # 检查文件类型，确定替换策略
        if self._is_class_file(content):
            content, count = self._replace_class_db_calls(content)
            call_count += count
        else:
            content, count = self._replace_function_db_calls(content)
            call_count += count
        
        return content, call_count
    
    def _is_class_file(self, content: str) -> bool:
        """检查是否是类文件"""
        return bool(re.search(r'class\s+\w+.*:', content))
    
    def _replace_class_db_calls(self, content: str) -> Tuple[str, int]:
        """替换类中的数据库调用"""
        call_count = 0
        
        # 替换类级别的数据库初始化
        patterns_replacements = [
            (r'self\.db\s*=\s*get_clickhouse_db\(\)',
             'self.data_access = get_container().resolve(IDataAccess)'),
            (r'cls\.db\s*=\s*get_clickhouse_db\(\)',
             'cls.data_access = get_container().resolve(IDataAccess)')
        ]
        
        for pattern, replacement in patterns_replacements:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                call_count += len(matches)
        
        # 替换数据库方法调用
        content = re.sub(r'self\.db\.query\(', 'self.data_access.execute_query(', content)
        content = re.sub(r'cls\.db\.query\(', 'cls.data_access.execute_query(', content)
        
        return content, call_count
    
    def _replace_function_db_calls(self, content: str) -> Tuple[str, int]:
        """替换函数中的数据库调用"""
        call_count = 0
        
        # 替换函数级别的数据库调用
        pattern = r'db\s*=\s*get_clickhouse_db\(\)'
        replacement = 'data_access = get_container().resolve(IDataAccess)'
        
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            call_count += len(matches)
        
        # 替换数据库方法调用
        content = re.sub(r'\bdb\.query\(', 'data_access.execute_query(', content)
        
        return content, call_count
    
    def _add_required_imports(self, content: str) -> str:
        """添加必要的导入语句"""
        lines = content.split('\n')
        
        # 查找导入区域
        import_end_idx = 0
        for i, line in enumerate(lines):
            if (line.strip().startswith('import ') or 
                line.strip().startswith('from ') or
                line.strip() == '' or
                line.strip().startswith('#')):
                import_end_idx = i
            else:
                break
        
        # 检查是否已有必要的导入
        has_container_import = any('from db.container import' in line for line in lines)
        has_interface_import = any('from db.interfaces.data_access_interface import' in line for line in lines)
        
        # 添加缺失的导入
        new_imports = []
        if not has_container_import:
            new_imports.append('from utils.dependency_injection import get_service')
        if not has_interface_import:
            new_imports.append('from db.interfaces.data_access_interface import IDataAccess')
        
        if new_imports:
            # 在导入区域末尾插入新导入
            for import_stmt in reversed(new_imports):
                lines.insert(import_end_idx + 1, import_stmt)
        
        return '\n'.join(lines)
    
    def refactor_all_files(self) -> Dict[str, int]:
        """重构所有文件"""
        logger.info("开始批量重构数据库依赖...")
        
        # 查找需要重构的文件
        files_to_refactor = self.find_files_to_refactor()
        self.refactor_stats['total_files'] = len(files_to_refactor)
        
        # 逐个重构文件
        for file_path in files_to_refactor:
            self.refactor_file(file_path)
        
        # 输出统计信息
        logger.info("重构完成！统计信息:")
        logger.info(f"  总文件数: {self.refactor_stats['total_files']}")
        logger.info(f"  成功重构: {self.refactor_stats['successful_refactors']}")
        logger.info(f"  失败重构: {self.refactor_stats['failed_refactors']}")
        logger.info(f"  导入替换: {self.refactor_stats['imports_replaced']}")
        logger.info(f"  调用替换: {self.refactor_stats['calls_replaced']}")
        
        return self.refactor_stats
    
    def generate_refactor_report(self) -> str:
        """生成重构报告"""
        report = f"""
# 数据库依赖重构报告

## 重构统计

- **总文件数**: {self.refactor_stats['total_files']}
- **成功重构**: {self.refactor_stats['successful_refactors']}
- **失败重构**: {self.refactor_stats['failed_refactors']}
- **导入语句替换**: {self.refactor_stats['imports_replaced']}
- **函数调用替换**: {self.refactor_stats['calls_replaced']}

## 重构内容

### 导入语句替换
```python
# 替换前
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access

# 替换后
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access
```

### 数据库调用替换
```python
# 替换前
db = get_service(Data_access_interface)
result = db.query(sql)

# 替换后
data_access = get_container().resolve(IData_access)
result = data_access.execute_query(sql)
```

### 类中的数据库调用替换
```python
# 替换前
self.data_access = get_container().resolve(IData_access)
data = self.data_access.execute_query(sql)

# 替换后
self.data_access = get_container().resolve(IData_access)
data = self.data_access.execute_query(sql)
```

## 重构效果

- ✅ 消除了直接数据库依赖
- ✅ 引入了依赖注入模式
- ✅ 提高了代码的可测试性
- ✅ 增强了系统的可维护性
- ✅ 符合SOLID原则中的依赖倒置原则

"""
        return report


def main():
    """主函数"""
    logger.info("启动数据库依赖重构脚本...")
    
    try:
        # 创建重构器
        refactor = Database_dependency_refactor()
        
        # 执行重构
        stats = refactor.refactor_all_files()
        
        # 生成报告
        report = refactor.generate_refactor_report()
        
        # 保存报告
        report_path = os.path.join(get_project_root(), 'reports', 'database_dependency_refactor_report.md')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"重构报告已保存到: {report_path}")
        
        # 返回成功状态
        if stats['failed_refactors'] == 0:
            logger.info("所有文件重构成功！")
            return 0
        else:
            logger.warning(f"有{stats['failed_refactors']}个文件重构失败")
            return 1
            
    except Exception as e:
        logger.error(f"重构过程发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 