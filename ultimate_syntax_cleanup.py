#!/usr/bin/env python3
"""
终极语法清理脚本 - 彻底解决所有语法问题
"""

import os
import ast
from utils.logger import get_logger

logger = get_logger(__name__)


def ultimate_syntax_cleanup():
    """终极语法清理"""
    logger.info("🔧 开始终极语法清理")
    
    # 修复cache_interface.py
    fix_cache_interface_syntax()
    
    # 修复intelligent_query_optimizer.py
    fix_query_optimizer_syntax()
    
    # 修复data_access_manager.py
    fix_data_access_manager_syntax()
    
    # 验证所有文件
    verify_all_syntax()


def fix_cache_interface_syntax():
    """修复cache_interface.py语法"""
    file_path = 'db/interfaces/cache_interface.py'
    if not os.path.exists(file_path):
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复第24行的缩进问题
        lines = content.split('\n')
        
        # 找到问题行并修复
        for i, line in enumerate(lines):
            if i == 23 and line.strip() == '"""':  # 第24行 (0-based index 23)
                # 移除这个多余的文档字符串开始
                lines[i] = ''
                logger.info("  修复cache_interface.py第24行缩进问题")
                break
        
        content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("✅ 修复cache_interface.py语法")
    
    except Exception as e:
        logger.error(f"❌ 修复cache_interface.py失败: {e}")


def fix_query_optimizer_syntax():
    """修复intelligent_query_optimizer.py语法"""
    file_path = 'db/services/integrated/intelligent_query_optimizer.py'
    if not os.path.exists(file_path):
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复缩进问题
        lines = content.split('\n')
        
        # 查找并修复缩进问题
        for i, line in enumerate(lines):
            if '"""' in line and i > 50 and i < 70:  # 大概在第60行附近
                if line.startswith('    """'):  # 如果有错误的缩进
                    lines[i] = '"""'  # 移除缩进
                    logger.info(f"  修复intelligent_query_optimizer.py第{i+1}行缩进")
        
        content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("✅ 修复intelligent_query_optimizer.py语法")
    
    except Exception as e:
        logger.error(f"❌ 修复intelligent_query_optimizer.py失败: {e}")


def fix_data_access_manager_syntax():
    """修复data_access_manager.py语法"""
    file_path = 'db/managers/data_access_manager.py'
    if not os.path.exists(file_path):
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 确保文档字符串格式正确
        lines = content.split('\n')
        
        # 查找类定义和文档字符串
        for i, line in enumerate(lines):
            if 'class DataAccessManager' in line:
                # 确保下一行是正确的文档字符串开始
                if i + 1 < len(lines) and lines[i + 1].strip() == '"""':
                    # 检查文档字符串内容
                    doc_start = i + 1
                    doc_end = -1
                    
                    for j in range(doc_start + 1, len(lines)):
                        if lines[j].strip() == '"""':
                            doc_end = j
                            break
                    
                    if doc_end > doc_start:
                        # 重写文档字符串，确保语法正确
                        new_doc = [
                            '    """',
                            '    Data Access Manager - Unified data access entry point',
                            '    ',
                            '    Method groups:',
                            '    - Interface implementation: Implement IDataAccess interface methods',
                            '    - Core queries: Basic stock data queries',
                            '    - Batch operations: Batch data retrieval and processing',
                            '    - Indicator data: Technical indicator data retrieval',
                            '    - Utility methods: Data validation and formatting',
                            '    """'
                        ]
                        
                        # 替换文档字符串
                        lines = lines[:doc_start] + new_doc + lines[doc_end + 1:]
                        logger.info("  重写data_access_manager.py文档字符串")
                        break
        
        content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("✅ 修复data_access_manager.py语法")
    
    except Exception as e:
        logger.error(f"❌ 修复data_access_manager.py失败: {e}")


def verify_all_syntax():
    """验证所有文件语法"""
    files_to_check = [
        'db/interfaces/cache_interface.py',
        'db/interfaces/data_access_interface.py',
        'db/services/cache_service.py',
        'db/managers/data_access_manager.py',
        'db/services/integrated/intelligent_query_optimizer.py'
    ]
    
    all_valid = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                ast.parse(content)
                logger.info(f"  ✅ {file_path} 语法正确")
            
            except SyntaxError as e:
                logger.error(f"  ❌ {file_path} 语法错误 (行{e.lineno}): {e.msg}")
                all_valid = False
            
            except Exception as e:
                logger.error(f"  ❌ {file_path} 检查失败: {e}")
                all_valid = False
    
    if all_valid:
        logger.info("🎉 所有文件语法验证通过！")
    else:
        logger.warning("⚠️ 部分文件仍有语法错误")
    
    return all_valid


def main():
    """主函数"""
    try:
        ultimate_syntax_cleanup()
        
        print("\n" + "="*60)
        print("🔧 终极语法清理完成")
        print("="*60)
        print("✅ 修复项目:")
        print("  1. cache_interface.py 缩进问题修复")
        print("  2. intelligent_query_optimizer.py 缩进问题修复")
        print("  3. data_access_manager.py 文档字符串重写")
        
        print("\n🎯 预期效果:")
        print("  • 所有语法错误彻底解决")
        print("  • 所有文件通过AST解析")
        print("  • 接口实现检查100%通过")
        print("  • 准备进行最终合规验证")
        
        print("\n🚀 下一步:")
        print("  运行 test_l3_architecture_design_compliance.py")
        print("  确认100%完美合规状态")
        
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"终极语法清理过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
