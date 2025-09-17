#!/usr/bin/env python3
"""
统一配置管理迁移脚本

将分散在各个文件中的硬编码配置迁移到统一的配置管理系统中。
解决项目中配置管理混乱的问题。

Author: System Architecture Team
Date: 2025-07-16
Version: 1.0
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedConfigMigrator:
    """统一配置管理迁移器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.config_issues = {
            'hardcoded_configs': [],
            'duplicate_configs': [],
            'missing_defaults': []
        }
        
        self.migration_stats = {
            'files_scanned': 0,
            'configs_found': 0,
            'configs_migrated': 0,
            'files_updated': 0
        }
        
        # 硬编码配置模式
        self.hardcoded_patterns = {
            'database': {
                r"['\"]host['\"][\s]*[:=][\s]*['\"]localhost['\"]": 'database.host',
                r"['\"]port['\"][\s]*[:=][\s]*9000": 'database.port',
                r"['\"]password['\"][\s]*[:=][\s]*['\"]123456['\"]": 'database.password',
                r"['\"]user['\"][\s]*[:=][\s]*['\"]default['\"]": 'database.user',
                r"['\"]database['\"][\s]*[:=][\s]*['\"]stock['\"]": 'database.name'
            },
            'cache': {
                r"max_size[\s]*=[\s]*100": 'cache.max_size',
                r"ttl[\s]*=[\s]*3600": 'cache.ttl',
                r"cache_size[\s]*=[\s]*\d+": 'cache.size'
            },
            'performance': {
                r"timeout[\s]*=[\s]*30": 'performance.timeout',
                r"max_connections[\s]*=[\s]*20": 'performance.max_connections',
                r"pool_size[\s]*=[\s]*\d+": 'performance.pool_size'
            }
        }
        
        # 配置替换映射
        self.config_replacements = {}
        
    def analyze_config_issues(self):
        """分析配置问题"""
        logger.info("开始分析配置问题...")
        
        # 扫描Python文件
        python_files = list(self.root_dir.rglob("*.py"))
        python_files = [f for f in python_files if not any(exclude in str(f) for exclude in [
            '__pycache__', '.git', 'venv', '.pytest_cache', 'node_modules'
        ])]
        
        logger.info(f"扫描 {len(python_files)} 个Python文件")
        
        for file_path in python_files:
            self._analyze_file_configs(file_path)
            self.migration_stats['files_scanned'] += 1
        
        # 生成分析报告
        self._generate_analysis_report()
        
    def _analyze_file_configs(self, file_path: Path):
        """分析单个文件的配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查硬编码配置
            for category, patterns in self.hardcoded_patterns.items():
                for pattern, config_key in patterns.items():
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        self.config_issues['hardcoded_configs'].append({
                            'file': str(file_path),
                            'category': category,
                            'pattern': pattern,
                            'config_key': config_key,
                            'matches': len(matches)
                        })
                        self.migration_stats['configs_found'] += len(matches)
                        
        except Exception as e:
            logger.warning(f"分析文件失败: {file_path}, 错误: {e}")
    
    def migrate_configurations(self):
        """迁移配置"""
        logger.info("开始迁移配置...")
        
        # 创建统一配置文件
        self._create_unified_config()
        
        # 迁移硬编码配置
        self._migrate_hardcoded_configs()
        
        # 生成迁移报告
        self._generate_migration_report()
        
    def _create_unified_config(self):
        """创建统一配置文件"""
        config_dir = self.root_dir / 'config'
        config_dir.mkdir(exist_ok=True)
        
        unified_config = {
            "database": {
                get_config('database.host'),
                get_config('database.port'),
                get_config('database.user'), 
                get_config('database.password'),
                "name": "stock",
                "timeout": 30,
                "compression": True
            },
            "cache": {
                "max_size": 100,
                "ttl": 3600,
                "size": 1000,
                "enabled": True
            },
            "performance": {
                "timeout": 30,
                "max_connections": 20,
                "pool_size": 10,
                "batch_size": 1000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/app.log"
            }
        }
        
        config_file = config_dir / 'unified_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(unified_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"创建统一配置文件: {config_file}")
        
    def _migrate_hardcoded_configs(self):
        """迁移硬编码配置"""
        migrated_files = set()
        
        for config_issue in self.config_issues['hardcoded_configs']:
            file_path = Path(config_issue['file'])
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 替换硬编码配置
                config_key = config_issue['config_key']
                pattern = config_issue['pattern']
                
                # 生成配置调用
                config_call = f"get_config('{config_key}')"
                
                # 执行替换
                content = re.sub(pattern, f"get_config('{config_key}')", content, flags=re.IGNORECASE)
                
                # 添加配置导入
                if content != original_content:
                    content = self._add_config_import(content)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    migrated_files.add(str(file_path))
                    self.migration_stats['configs_migrated'] += 1
                    
            except Exception as e:
                logger.error(f"迁移文件失败: {file_path}, 错误: {e}")
        
        self.migration_stats['files_updated'] = len(migrated_files)
        logger.info(f"成功迁移 {len(migrated_files)} 个文件的配置")
        
    def _add_config_import(self, content: str) -> str:
        """添加配置导入语句"""
        lines = content.split('\n')
        
        # 检查是否已有配置导入
        has_config_import = any('from config.unified_config_manager import get_config' in line for line in lines)
        
        if not has_config_import:
            # 找到导入区域
            import_end_idx = 0
            for i, line in enumerate(lines):
                if (line.strip().startswith('import ') or 
                    line.strip().startswith('from ') or
                    line.strip() == '' or
                    line.strip().startswith('#')):
                    import_end_idx = i
                else:
                    break
            
            # 添加配置导入
            lines.insert(import_end_idx + 1, 'from config.unified_config_manager import get_config')
        
        return '\n'.join(lines)
    
    def _generate_analysis_report(self):
        """生成分析报告"""
        report = {
            'summary': {
                'files_scanned': self.migration_stats['files_scanned'],
                'configs_found': self.migration_stats['configs_found'],
                'hardcoded_configs': len(self.config_issues['hardcoded_configs'])
            },
            'issues': self.config_issues
        }
        
        report_file = self.root_dir / 'config_analysis_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置分析报告生成: {report_file}")
        
        # 输出统计信息
        logger.info(f"配置分析完成:")
        logger.info(f"  扫描文件: {self.migration_stats['files_scanned']} 个")
        logger.info(f"  发现硬编码配置: {self.migration_stats['configs_found']} 个")
        logger.info(f"  涉及文件: {len(set(issue['file'] for issue in self.config_issues['hardcoded_configs']))} 个")
        
    def _generate_migration_report(self):
        """生成迁移报告"""
        report = {
            'summary': self.migration_stats,
            'migrated_configs': self.config_issues['hardcoded_configs']
        }
        
        report_file = self.root_dir / 'config_migration_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置迁移报告生成: {report_file}")
        
        # 输出统计信息
        logger.info(f"配置迁移完成:")
        logger.info(f"  迁移配置: {self.migration_stats['configs_migrated']} 个")
        logger.info(f"  更新文件: {self.migration_stats['files_updated']} 个")
        
    def create_config_manager(self):
        """创建配置管理器"""
        config_manager_code = '''#!/usr/bin/env python3
"""
统一配置管理器

提供统一的配置访问接口，支持环境变量覆盖、配置验证等功能。
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache
from db.sql_manager import SQLManager, QueryType


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._config: Optional[Dict] = None
        self._config_file = Path(__file__).parent / 'unified_config.json'
    
    @lru_cache(maxsize=1)
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self._config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self._config_file}")
        
        with open(self._config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 环境变量覆盖
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        # 数据库配置
        if 'CLICKHOUSE_HOST' in os.environ:
            config['database']['host'] = os.environ['CLICKHOUSE_HOST']
        if 'CLICKHOUSE_PORT' in os.environ:
            config['database']['port'] = int(os.environ['CLICKHOUSE_PORT'])
        if 'CLICKHOUSE_USER' in os.environ:
            config['database']['user'] = os.environ['CLICKHOUSE_USER']
        if 'CLICKHOUSE_PASSWORD' in os.environ:
            config['database']['password'] = os.environ['CLICKHOUSE_PASSWORD']
        if 'CLICKHOUSE_DATABASE' in os.environ:
            config['database']['name'] = os.environ['CLICKHOUSE_DATABASE']
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套访问，如 'database.host'
            default: 默认值
        
        Returns:
            配置值
        """
        if self._config is None:
            self._config = self._load_config()
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload(self):
        """重新加载配置"""
        self._config = None
        self._load_config.cache_clear()
'''
        
        config_manager_file = self.root_dir / 'config' / '__init__.py'
        with open(config_manager_file, 'w', encoding='utf-8') as f:
            f.write(config_manager_code)
        
        logger.info(f"创建配置管理器: {config_manager_file}")


def main_unified_config_migration():
    """主函数"""
    migrator = UnifiedConfigMigrator()
    
    # 分析配置问题
    migrator.analyze_config_issues()
    
    # 迁移配置
    migrator.migrate_configurations()
    
    # 创建配置管理器
    migrator.create_config_manager()
    
    logger.info("统一配置管理迁移完成!")


if __name__ == "__main__":
    main_unified_config_migration() 