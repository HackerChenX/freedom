#!/usr/bin/env python3
"""
简单的硬编码配置修复脚本
"""
import os
import re
import sys
import logging

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_hardcoded_configs():
    """修复硬编码配置"""
    logger.info("开始修复硬编码配置...")
    
    # 简单的替换规则
    replacements = [
        # ClickHouse配置
        (r"host\s*=\s*['\"]localhost['\"]", "host=os.getenv('DB_HOST', 'localhost')"),
        (r"port\s*=\s*8123", "port=int(os.getenv('DB_PORT', '8123'))"),
        (r"port\s*=\s*9000", "port=int(os.getenv('DB_PORT', '9000'))"),
        (r"database\s*=\s*['\"]stock['\"]", "database=os.getenv('DB_DATABASE', 'stock')"),
        (r"user\s*=\s*['\"]default['\"]", "user=os.getenv('DB_USER', 'default')"),
        (r"password\s*=\s*['\"]['\"]", "password=os.getenv('DB_PASSWORD', '')"),
    ]
    
    # 需要修复的文件
    target_files = [
        "tests/end_to_end/real_data_performance_test.py",
        "tests/optimization/comprehensive_optimization_test.py",
        "tests/performance/simple_concurrent_test.py",
        "tests/performance/concurrent_optimization_test.py",
        "scripts/simple_clickhouse_test.py",
        "scripts/production_database_test.py",
        "scripts/clickhouse_connection_summary.py"
    ]
    
    files_fixed = 0
    
    for file_path in target_files:
        full_path = os.path.join(root_dir, file_path)
        if not os.path.exists(full_path):
            logger.warning(f"文件不存在: {full_path}")
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # 应用替换规则
            for pattern, replacement in replacements:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True
            
            # 如果修改了内容，添加os导入并写回文件
            if modified:
                # 检查是否已经有os导入
                if 'import os' not in content:
                    # 在第一行添加os导入
                    lines = content.split('\n')
                    # 找到第一个import语句的位置
                    import_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            import_idx = i
                            break
                    
                    lines.insert(import_idx, 'import os')
                    content = '\n'.join(lines)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"修复文件: {file_path}")
                files_fixed += 1
            
        except Exception as e:
            logger.error(f"修复文件 {file_path} 失败: {e}")
    
    logger.info(f"修复完成，共修复 {files_fixed} 个文件")
    
    # 运行架构检查验证修复效果
    logger.info("运行架构检查验证修复效果...")
    os.system("python simple_architecture_check.py")

if __name__ == "__main__":
    fix_hardcoded_configs() 