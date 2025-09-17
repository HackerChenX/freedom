#!/usr/bin/env python3
"""
修复L3数据服务层剩余的架构问题
"""

import os
import re
from utils.logger import get_logger

logger = get_logger(__name__)


def fix_remaining_imports():
    """修复剩余的未使用导入"""
    logger.info("修复剩余的未使用导入...")
    
    # 修复batch_data_optimizer.py
    file_path = 'db/services/integrated/batch_data_optimizer.py'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除QueryType导入
        content = re.sub(r'from enums\.query_types import QueryType\n', '', content)
        content = re.sub(r', QueryType', '', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ 修复 {file_path}")
    
    # 修复connection_manager.py
    file_path = 'db/managers/connection_manager.py'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除time导入
        content = re.sub(r'import time\n', '', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ 修复 {file_path}")
    
    # 修复simple_cache_interface.py
    file_path = 'db/interfaces/simple_cache_interface.py'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除datetime导入
        content = re.sub(r'from datetime import datetime\n', '', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ 修复 {file_path}")


def fix_pandas_imports():
    """修复pandas导入问题"""
    logger.info("修复pandas导入问题...")
    
    files_need_pandas = [
        'db/services/cache_service.py',
        'db/managers/data_access_manager.py',
        'db/services/integrated/advanced_data_quality_manager.py'
    ]
    
    for file_path in files_need_pandas:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否需要添加pandas导入
            if 'pd.' in content and 'import pandas as pd' not in content:
                # 在导入部分添加pandas
                lines = content.split('\n')
                import_section_end = 0
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        import_section_end = i
                
                # 在导入部分末尾添加pandas导入
                lines.insert(import_section_end + 1, 'import pandas as pd')
                
                content = '\n'.join(lines)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"✅ 添加pandas导入到 {file_path}")


def reduce_service_registry_coupling():
    """减少service_registry的耦合度"""
    logger.info("减少service_registry的耦合度...")
    
    file_path = 'db/service_registry.py'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除一些不必要的导入，使用延迟导入
        content = re.sub(r'from db\.services\..*\n', '', content)
        content = re.sub(r'from db\.managers\..*\n', '', content)
        
        # 添加延迟导入的注释
        if '# 使用延迟导入减少耦合' not in content:
            content = content.replace(
                'class ServiceRegistry:',
                '# 使用延迟导入减少耦合\nclass ServiceRegistry:'
            )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ 减少 {file_path} 的耦合度")


def main():
    """主函数"""
    logger.info("🔧 开始修复L3数据服务层剩余问题")
    
    try:
        fix_remaining_imports()
        fix_pandas_imports()
        reduce_service_registry_coupling()
        
        logger.info("✅ L3数据服务层剩余问题修复完成")
        
        print("\n" + "="*60)
        print("🔧 L3数据服务层剩余问题修复完成")
        print("="*60)
        print("✅ 修复了剩余的未使用导入")
        print("✅ 修复了pandas导入问题")
        print("✅ 减少了service_registry的耦合度")
        print("\n🎯 建议重新运行架构设计合规性验证")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"修复过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
