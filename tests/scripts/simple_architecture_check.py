#!/usr/bin/env python3
"""
简化架构合规性检查脚本

检查系统是否符合六层架构规范，包括：
1. 直接数据库依赖
2. 通配符导入
3. 全局单例
4. 硬编码配置
5. SQL语句分散
6. 分层架构违规
"""

import os
import re
import ast
from typing import Dict, List, Set, Tuple
from pathlib import Path

class SimpleArchitectureChecker:
    """简化架构检查器"""
    
    def __init__(self):
        self.violations = {
            'direct_db_dependency': [],
            'wildcard_imports': [],
            'global_singletons': [],
            'hardcoded_configs': [],
            'sql_scattered': [],
            'layer_violations': []
        }
        
        # 层级定义
        self.layer_mapping = {
            'L1': ['config', 'utils', 'enums', 'db/clickhouse_db.py', 'db/enhanced_connection_pool.py'],
            'L2': ['db'],
            'L3': ['db/sql_manager.py', 'db/query_executor.py', 'db/unified_data_manager.py'],
            'L4': ['indicators', 'formula'],
            'L5': ['analysis', 'strategy'],
            'L6': ['bin', 'scripts', 'tests', 'examples', 'crawler']
        }
        
        # 排除的目录
        self.excluded_dirs = {
            'venv', '__pycache__', '.git', 'node_modules', 
            'logs', 'cache', 'tmp', 'metadata', 'store', '.venv'
        }
    
    def check_direct_db_dependency(self) -> List[str]:
        """检查直接数据库依赖"""
        violations = []
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # 检查直接导入ClickhouseDB
                        if 'from db.clickhouse_db import ClickhouseDB' in content:
                            violations.append(file_path)
                    except Exception:
                        continue
        
        return violations
    
    def check_wildcard_imports(self) -> List[str]:
        """检查通配符导入"""
        violations = []
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # 检查通配符导入
                        if re.search(r'from\s+\w+\s+import\s+\*', content):
                            violations.append(file_path)
                    except Exception:
                        continue
        
        return violations
    
    def check_global_singletons(self) -> List[str]:
        """检查全局单例"""
        violations = []
        singleton_patterns = [
            r'_instance\s*=\s*None',
            r'def\s+__new__\s*\(.*?\):',
            r'class\s+\w+.*Singleton',
            r'@singleton',
            r'if\s+not\s+hasattr\(.*?,\s*[\'"]_instance[\'"].*?\):'
        ]
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # 检查单例模式
                        for pattern in singleton_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                violations.append(file_path)
                                break
                    except Exception:
                        continue
        
        return violations
    
    def check_hardcoded_configs(self) -> List[str]:
        """检查硬编码配置"""
        violations = []
        
        # 改进的硬编码检测模式
        config_patterns = [
            r'host\s*=\s*[\'"][^\'"\s]+[\'"]',
            r'port\s*=\s*\d+',
            r'user\s*=\s*[\'"][^\'"\s]+[\'"]',
            r'password\s*=\s*[\'"][^\'"\s]+[\'"]',
            r'database\s*=\s*[\'"][^\'"\s]+[\'"]',
            r'redis_host\s*=\s*[\'"][^\'"\s]+[\'"]',
            r'redis_port\s*=\s*\d+',
            r'timeout\s*=\s*\d+',
            r'max_connections\s*=\s*\d+',
            r'batch_size\s*=\s*\d+',
        ]
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # 检查硬编码配置
                        for pattern in config_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                # 排除f字符串中的变量和枚举值
                                line_content = content[max(0, match.start()-50):match.end()+50]
                                if (not re.search(r'f[\'"].*?\{.*?\}.*?[\'"]', line_content) and
                                    'enum' not in line_content.lower() and
                                    'Enum' not in line_content and
                                    not re.search(r'[\'"].*?\{.*?\}.*?[\'"]', line_content)):
                                    violations.append(f"{file_path}: {match.group()}")
                                    break
                    except Exception:
                        continue
        
        return violations
    
    def check_sql_scattered(self) -> List[str]:
        """检查SQL语句分散（更新版）"""
        violations = []
        
        # 更新的SQL检测模式，排除已迁移的查询
        sql_patterns = [
            r'SELECT\s+.*?\s+FROM\s+\w+',
            r'INSERT\s+INTO\s+\w+',
            r'UPDATE\s+\w+\s+SET',
            r'DELETE\s+FROM\s+\w+',
            r'query_dataframe\s*\(',
            r'execute\s*\(\s*[\'"].*?SELECT.*?[\'"]',
            r'conn\.query\s*\(',
        ]
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # 检查是否已迁移到统一SQL管理系统
                        is_migrated = (
                            'from db.query_executor import get_query_executor' in content or
                            'from db.sql_manager import QueryType' in content or
from db.sql_manager import SQLManager, QueryType
                            'query_executor.get_stock_data' in content or
                            'query_executor.execute_query' in content
                        )
                        
                        # 如果已迁移，检查是否还有遗留的SQL
                        has_legacy_sql = False
                        if is_migrated:
                            # 只检查遗留的直接SQL查询
                            legacy_patterns = [
                                r'conn\.query_dataframe\s*\(\s*[\'"]SELECT.*?[\'"]',
                                r'execute\s*\(\s*[\'"].*?SELECT.*?[\'"]',
                                r'query\s*\(\s*[\'"].*?SELECT.*?[\'"]'
                            ]
                            
                            for pattern in legacy_patterns:
                                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                                    has_legacy_sql = True
                                    break
                        else:
                            # 如果未迁移，检查是否包含SQL
                            for pattern in sql_patterns:
                                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                                    has_legacy_sql = True
                                    break
                        
                        if has_legacy_sql:
                            violations.append(file_path.replace('./', ''))
                    except Exception:
                        continue
        
        return violations
    
    def check_layer_violations(self) -> List[str]:
        """检查分层架构违规"""
        violations = []
        
        # 构建文件到层级的映射
        file_to_layer = {}
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file).replace('./', '')
                    
                    # 确定文件所属层级
                    layer = self._get_file_layer(file_path)
                    if layer:
                        file_to_layer[file_path] = layer
        
        # 检查依赖关系
        for file_path, layer in file_to_layer.items():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 分析导入语句
                imports = self._extract_imports(content)
                
                for import_path in imports:
                    # 找到被导入文件的层级
                    imported_layer = self._get_import_layer(import_path, file_to_layer)
                    
                    if imported_layer and self._is_layer_violation(layer, imported_layer):
                        violations.append(f"{file_path} ({layer}) -> {import_path} ({imported_layer})")
                        
            except Exception:
                continue
        
        return violations
    
    def _get_file_layer(self, file_path: str) -> str:
        """获取文件所属层级"""
        for layer, patterns in self.layer_mapping.items():
            for pattern in patterns:
                if file_path.startswith(pattern):
                    return layer
        return None
    
    def _get_import_layer(self, import_path: str, file_to_layer: Dict[str, str]) -> str:
        """获取导入路径的层级"""
        # 转换导入路径为文件路径
        possible_paths = [
            import_path.replace('.', '/') + '.py',
            import_path.replace('.', '/') + '/__init__.py'
        ]
        
        for path in possible_paths:
            if path in file_to_layer:
                return file_to_layer[path]
        
        # 根据导入路径模式判断层级
        for layer, patterns in self.layer_mapping.items():
            for pattern in patterns:
                if import_path.startswith(pattern.replace('/', '.')):
                    return layer
        
        return None
    
    def _extract_imports(self, content: str) -> List[str]:
        """提取导入语句"""
        imports = []
        
        # 匹配import语句
        import_patterns = [
            r'from\s+([\w\.]+)\s+import',
            r'import\s+([\w\.]+)'
        ]
        
        for pattern in import_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                import_path = match.group(1)
                if not import_path.startswith('.'):  # 排除相对导入
                    imports.append(import_path)
        
        return imports
    
    def _is_layer_violation(self, from_layer: str, to_layer: str) -> bool:
        """检查是否违反分层规则"""
        layer_order = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        
        try:
            from_index = layer_order.index(from_layer)
            to_index = layer_order.index(to_layer)
            
            # 上层只能依赖相邻下层
            return from_index < to_index or (from_index - to_index) > 1
        except ValueError:
            return False
    
    def run_all_checks(self) -> Dict[str, List[str]]:
        """运行所有检查"""
        print("开始架构合规性检查...")
        
        print("检查直接数据库依赖...")
        self.violations['direct_db_dependency'] = self.check_direct_db_dependency()
        
        print("检查通配符导入...")
        self.violations['wildcard_imports'] = self.check_wildcard_imports()
        
        print("检查全局单例...")
        self.violations['global_singletons'] = self.check_global_singletons()
        
        print("检查硬编码配置...")
        self.violations['hardcoded_configs'] = self.check_hardcoded_configs()
        
        print("检查SQL语句分散...")
        self.violations['sql_scattered'] = self.check_sql_scattered()
        
        print("检查分层架构违规...")
        self.violations['layer_violations'] = self.check_layer_violations()
        
        return self.violations
    
    def print_results(self):
        """打印检查结果"""
        print("\n=== 架构合规性检查结果 ===\n")
        
        total_violations = 0
        
        # 1. 直接数据库依赖
        db_violations = len(self.violations['direct_db_dependency'])
        total_violations += db_violations
        print(f"1. 直接数据库依赖违规: {db_violations}个")
        if db_violations > 0:
            for violation in self.violations['direct_db_dependency'][:5]:
                print(f"   - {violation}")
            if db_violations > 5:
                print(f"   ... 还有 {db_violations - 5} 个")
        
        # 2. 通配符导入
        wildcard_violations = len(self.violations['wildcard_imports'])
        total_violations += wildcard_violations
        print(f"\n2. 通配符导入违规: {wildcard_violations}个")
        if wildcard_violations > 0:
            for violation in self.violations['wildcard_imports'][:5]:
                print(f"   - {violation}")
            if wildcard_violations > 5:
                print(f"   ... 还有 {wildcard_violations - 5} 个")
        
        # 3. 全局单例
        singleton_violations = len(self.violations['global_singletons'])
        total_violations += singleton_violations
        print(f"\n3. 全局单例违规: {singleton_violations}个")
        if singleton_violations > 0:
            for violation in self.violations['global_singletons'][:5]:
                print(f"   - {violation}")
            if singleton_violations > 5:
                print(f"   ... 还有 {singleton_violations - 5} 个")
        
        # 4. 硬编码配置
        config_violations = len(self.violations['hardcoded_configs'])
        total_violations += config_violations
        print(f"\n4. 硬编码配置违规: {config_violations}个")
        if config_violations > 0:
            for violation in self.violations['hardcoded_configs'][:5]:
                print(f"   - {violation}")
            if config_violations > 5:
                print(f"   ... 还有 {config_violations - 5} 个")
        
        # 5. SQL语句分散
        sql_violations = len(self.violations['sql_scattered'])
        total_violations += sql_violations
        print(f"\n5. SQL语句分散违规: {sql_violations}个")
        if sql_violations > 0:
            for violation in self.violations['sql_scattered']:
                print(f"   - {violation}")
        
        # 6. 分层架构违规
        layer_violations = len(self.violations['layer_violations'])
        total_violations += layer_violations
        print(f"\n6. 分层架构违规: {layer_violations}个")
        if layer_violations > 0:
            for violation in self.violations['layer_violations'][:5]:
                print(f"   - {violation}")
            if layer_violations > 5:
                print(f"   ... 还有 {layer_violations - 5} 个")
        
        # 总结
        print(f"\n总计发现 {total_violations} 个架构违规问题")
        
        if total_violations == 0:
            print("🎉 系统完全符合架构规范！")
        elif total_violations <= 5:
            print("✅ 系统基本符合架构规范，仅有少量问题需要修复")
        elif total_violations <= 20:
            print("⚠️ 系统架构部分合规，需要重点关注和修复")
        else:
            print("❌ 系统存在严重架构违规，需要立即修复")

def main_simple_architecture_check():
    """主函数"""
    checker = SimpleArchitectureChecker()
    checker.run_all_checks()
    checker.print_results()

if __name__ == "__main__":
    main_simple_architecture_check() 