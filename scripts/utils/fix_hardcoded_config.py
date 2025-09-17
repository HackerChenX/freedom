#!/usr/bin/env python3
"""
硬编码配置修复脚本

将硬编码的配置项迁移到配置文件管理
"""

import os
import re
import json
from typing import Dict, List, Tuple

class HardcodedConfigFixer:
    """硬编码配置修复器"""
    
    def __init__(self):
        # 需要修复的文件
        self.target_files = [
            "crawler/integration/deployment_manager.py"
        ]
        
        # 配置项映射
        self.config_mappings = {
            'REDIS_PORT=6379': {
                'config_key': 'redis.port',
                'default_value': 6379,
                'description': 'Redis服务器端口'
            },
            'CLICKHOUSE_PORT=8123': {
                'config_key': 'clickhouse.port',
                'default_value': 8123,
                'description': 'ClickHouse HTTP端口'
            },
            'CLICKHOUSE_PORT=9000': {
                'config_key': 'clickhouse.native_port',
                'default_value': 9000,
                'description': 'ClickHouse原生端口'
            }
        }
        
        # 修复统计
        self.stats = {
            'total_files': len(self.target_files),
            'successful_fixes': 0,
            'failed_fixes': 0,
            'configs_migrated': 0
        }
    
    def create_docker_config(self) -> Dict[str, any]:
        """创建Docker相关配置"""
        return {
            'docker': {
                'redis': {
                    'image': 'redis:7-alpine',
                    'port': 6379,
                    'volumes': ['redis_data:/data'],
                    'restart': 'unless-stopped'
                },
                'clickhouse': {
                    'image': 'clickhouse/clickhouse-server:latest',
                    'http_port': 8123,
                    'native_port': 9000,
                    'volumes': ['clickhouse_data:/var/lib/clickhouse'],
                    'database': 'stock_crawler',
                    get_config('database.user'),
                    'password': '',
                    'restart': 'unless-stopped'
                },
                'crawler': {
                    'max_workers': 5,
                    'use_proxy': False,
                    'restart': 'unless-stopped'
                }
            }
        }
    
    def fix_deployment_manager(self, content: str) -> Tuple[str, int]:
        """修复部署管理器文件"""
        fixed_count = 0
        
        # 在文件开头添加配置导入
        if 'from config.unified_config_manager import get_config' not in content:
            # 找到导入部分
            import_lines = []
            content_lines = content.split('\n')
            import_end_idx = 0
            
            for i, line in enumerate(content_lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_end_idx = i
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            # 在导入部分添加配置导入
            content_lines.insert(import_end_idx + 1, 'from config.unified_config_manager import get_config')
            content = '\n'.join(content_lines)
        
        # 修复硬编码的端口配置
        replacements = [
            # Redis端口
            (r'- REDIS_PORT=6379', '- REDIS_PORT=${REDIS_PORT:-6379}'),
            (r'"6379:6379"', '"${REDIS_PORT:-6379}:6379"'),
            
            # ClickHouse端口
            (r'- CLICKHOUSE_PORT=8123', '- CLICKHOUSE_PORT=${CLICKHOUSE_HTTP_PORT:-8123}'),
            (r'"8123:8123"', '"${CLICKHOUSE_HTTP_PORT:-8123}:8123"'),
            (r'"9000:9000"', '"${CLICKHOUSE_NATIVE_PORT:-9000}:9000"'),
            
            # 其他硬编码配置
            (r'- MAX_WORKERS=5', '- MAX_WORKERS=${MAX_WORKERS:-5}'),
            (r'- USE_PROXY=false', '- USE_PROXY=${USE_PROXY:-false}'),
        ]
        
        for pattern, replacement in replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixed_count += 1
        
        # 添加配置获取方法
        if 'def get_docker_config(' not in content:
            config_method = '''
    def get_docker_config(self) -> Dict[str, any]:
        """获取Docker配置"""
        try:
            config = get_config()
            return config.get('docker', {
                'redis': {'port': 6379},
                'clickhouse': {'http_port': 8123, 'native_port': 9000},
                'crawler': {'max_workers': 5, 'use_proxy': False}
            })
        except Exception as e:
            logger.warning(f"获取Docker配置失败，使用默认配置: {e}")
            return {
                'redis': {'port': 6379},
                'clickhouse': {'http_port': 8123, 'native_port': 9000},
                'crawler': {'max_workers': 5, 'use_proxy': False}
            }
'''
            # 在类的末尾添加方法
            content = content.replace(
                "volumes:\n  redis_data:\n  clickhouse_data:\n'''",
                "volumes:\n  redis_data:\n  clickhouse_data:\n'''" + config_method
            )
        
        return content, fixed_count
    
    def create_docker_config_file(self) -> str:
        """创建Docker配置文件"""
        config_path = "config/docker_config.json"
        
        # 确保配置目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 创建配置内容
        config_content = self.create_docker_config()
        
        # 写入配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_content, f, indent=2, ensure_ascii=False)
        
        return config_path
    
    def fix_file(self, file_path: str) -> Tuple[bool, str]:
        """修复单个文件"""
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixed_count = 0
            
            # 根据文件类型进行不同的修复
            if file_path == "crawler/integration/deployment_manager.py":
                content, fixed_count = self.fix_deployment_manager(content)
            
            # 检查是否有实际变化
            if content != original_content:
                # 备份原文件
                backup_path = f"{file_path}.hardcoded_fix_backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # 写入修复后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.stats['configs_migrated'] += fixed_count
                return True, f"成功修复: {file_path} (迁移了{fixed_count}个配置项)"
            else:
                return True, f"无需修复: {file_path}"
                
        except Exception as e:
            return False, f"修复失败 {file_path}: {str(e)}"
    
    def run_fixes(self) -> Dict[str, List[Tuple[str, bool, str]]]:
        """运行所有修复"""
        print("开始修复硬编码配置问题...")
        
        results = {'fixed': []}
        
        # 创建Docker配置文件
        try:
            config_path = self.create_docker_config_file()
            print(f"  ✅ 创建配置文件: {config_path}")
        except Exception as e:
            print(f"  ❌ 创建配置文件失败: {e}")
        
        # 修复文件
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
        report.append("# 硬编码配置修复报告")
        report.append(f"生成时间: {os.popen('date').read().strip()}")
        report.append("")
        
        # 统计信息
        report.append("## 修复统计")
        report.append(f"- 目标文件数: {self.stats['total_files']}")
        report.append(f"- 成功修复: {self.stats['successful_fixes']}")
        report.append(f"- 失败数: {self.stats['failed_fixes']}")
        report.append(f"- 迁移配置项数: {self.stats['configs_migrated']}")
        report.append(f"- 成功率: {self.stats['successful_fixes']/self.stats['total_files']*100:.1f}%")
        report.append("")
        
        # 详细结果
        report.append("## 修复详情")
        for file_path, success, message in results['fixed']:
            status = "✅" if success else "❌"
            report.append(f"- {status} {file_path}: {message}")
        report.append("")
        
        # 配置说明
        report.append("## 配置说明")
        report.append("### Docker配置文件")
        report.append("创建了 `config/docker_config.json` 文件，包含以下配置:")
        report.append("- **Redis配置**: 端口、镜像、重启策略等")
        report.append("- **ClickHouse配置**: HTTP端口、原生端口、数据库设置等")
        report.append("- **爬虫配置**: 最大工作线程数、代理设置等")
        report.append("")
        
        report.append("### 环境变量支持")
        report.append("修复后的docker-compose.yml支持以下环境变量:")
        report.append("- `REDIS_PORT`: Redis端口 (默认: 6379)")
        report.append("- `CLICKHOUSE_HTTP_PORT`: ClickHouse HTTP端口 (默认: 8123)")
        report.append("- `CLICKHOUSE_NATIVE_PORT`: ClickHouse原生端口 (默认: 9000)")
        report.append("- `MAX_WORKERS`: 最大工作线程数 (默认: 5)")
        report.append("- `USE_PROXY`: 是否使用代理 (默认: false)")
        report.append("")
        
        # 使用指南
        report.append("## 使用指南")
        report.append("### 1. 环境变量设置")
        report.append("```bash")
        report.append("# 设置自定义端口")
        report.append("export REDIS_PORT=6380")
        report.append("export CLICKHOUSE_HTTP_PORT=8124")
        report.append("")
        report.append("# 启动服务")
        report.append("docker-compose up -d")
        report.append("```")
        report.append("")
        
        report.append("### 2. 配置文件修改")
        report.append("直接修改 `config/docker_config.json` 文件中的配置项。")
        report.append("")
        
        report.append("### 3. 代码中获取配置")
        report.append("```python")
        report.append("# 获取Docker配置")
        report.append("docker_config = deployment_manager.get_docker_config()")
        report.append("redis_port = docker_config['redis']['port']")
        report.append("```")
        
        return '\n'.join(report)

def main_fix_hardcoded_config():
    """主函数"""
    fixer = HardcodedConfigFixer()
    results = fixer.run_fixes()
    
    # 生成报告
    report = fixer.generate_fix_report(results)
    
    # 保存报告
    report_path = "hardcoded_config_fix_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n修复完成！报告已保存到: {report_path}")
    
    # 打印统计
    print(f"\n=== 修复统计 ===")
    print(f"目标文件数: {fixer.stats['total_files']}")
    print(f"成功修复: {fixer.stats['successful_fixes']}")
    print(f"失败数: {fixer.stats['failed_fixes']}")
    print(f"迁移配置项数: {fixer.stats['configs_migrated']}")
    print(f"成功率: {fixer.stats['successful_fixes']/fixer.stats['total_files']*100:.1f}%")

if __name__ == "__main__":
    main_fix_hardcoded_config() 