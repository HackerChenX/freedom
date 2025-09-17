#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第一阶段：修复关键语法错误
系统性修复影响指标注册的58个语法错误
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fix_common_syntax_errors():
    """修复常见的语法错误"""
    
    fixes_applied = 0
    files_fixed = 0
    
    # 定义需要修复的文件和错误模式
    error_patterns = [
        # 1. 修复空变量赋值
        (r'(\s+)=\s+([^=\n]+)', r'\1temp_var = \2'),
        
        # 2. 修复缺少逗号的语法错误
        (r'(\w+)\s+(\w+)\s*FROM', r'\1, \2 FROM'),
        
        # 3. 修复缺少变量名的赋值
        (r'(\s+)=\s*\(([^)]+)\)\s*\*\s*([^,\n]+)', r'\1score_change = (\2) * \3'),
        
        # 4. 修复字符串字面量错误
        (r'unterminated string literal', ''),
        
        # 5. 修复缺少引号的字符串
        (r"(\s+return\s+)([^'\"\n]+)(\s*#.*)?$", r'\1"\2"\3'),
    ]
    
    # 需要修复的文件列表
    target_files = [
        'indicators/psy.py',
        'indicators/enhanced_trix.py', 
        'indicators/cmo.py',
        'indicators/obv.py',
        'indicators/vol.py',
        'indicators/vr.py',
        'indicators/vosc.py',
        'indicators/pvt.py',
        'indicators/chaikin.py',
        'indicators/force_index.py',
        'indicators/vix.py',
        'indicators/pattern_recognition/v_shaped_reversal.py',
        'indicators/pattern_recognition/rectangle.py',
        'indicators/enhanced_macd.py',
        'indicators/elliott_wave.py',
        'indicators/mtm.py',
        'indicators/composite.py',
        'indicators/zxm/zxm_abstract_methods_mixin.py',
        'indicators/zxm/institutional_behavior.py',
    ]
    
    print("🔧 开始修复语法错误...")
    
    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"  ⚠️ 文件不存在: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            file_fixes = 0
            
            # 应用修复模式
            for pattern, replacement in error_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                    file_fixes += len(matches)
            
            # 特定文件的特殊修复
            if 'psy.py' in file_path:
                content = fix_psy_specific_errors(content)
                file_fixes += 1
            elif 'enhanced_trix.py' in file_path:
                content = fix_enhanced_trix_errors(content)
                file_fixes += 1
            elif 'cmo.py' in file_path:
                content = fix_cmo_errors(content)
                file_fixes += 1
            elif 'v_shaped_reversal.py' in file_path:
                content = fix_v_shaped_reversal_errors(content)
                file_fixes += 1
            elif 'rectangle.py' in file_path:
                content = fix_rectangle_errors(content)
                file_fixes += 1
            
            # 如果有修改，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                files_fixed += 1
                fixes_applied += file_fixes
                print(f"  ✅ 修复文件: {file_path} ({file_fixes}个修复)")
            else:
                print(f"  ℹ️ 无需修复: {file_path}")
                
        except Exception as e:
            print(f"  ❌ 修复失败: {file_path} - {e}")
    
    print(f"\n📊 修复摘要:")
    print(f"  修复文件数: {files_fixed}")
    print(f"  总修复数: {fixes_applied}")
    
    return files_fixed, fixes_applied

def fix_psy_specific_errors(content: str) -> str:
    """修复PSY指标的特定错误"""
    # 修复第391行的语法错误
    content = re.sub(
        r'(\s+)return\s+([^"\'\n]+)(\s*#.*)?$',
        r'\1return "\2"\3',
        content,
        flags=re.MULTILINE
    )
    return content

def fix_enhanced_trix_errors(content: str) -> str:
    """修复ENHANCED_TRIX指标的特定错误"""
    # 修复第351行的语法错误
    content = re.sub(
        r'(\s+)=\s+([^=\n]+)(\s*#.*)?$',
        r'\1trix_value = \2\3',
        content,
        flags=re.MULTILINE
    )
    return content

def fix_cmo_errors(content: str) -> str:
    """修复CMO指标的特定错误"""
    # 修复第345行的语法错误 - 缺少逗号
    content = re.sub(
        r'(\w+)\s+(\w+)(\s*[,\)])',
        r'\1, \2\3',
        content
    )
    return content

def fix_v_shaped_reversal_errors(content: str) -> str:
    """修复V型反转指标的特定错误"""
    # 修复第110行的未终止字符串字面量
    content = re.sub(
        r'(["\'])([^"\']*?)$',
        r'\1\2\1',
        content,
        flags=re.MULTILINE
    )
    return content

def fix_rectangle_errors(content: str) -> str:
    """修复矩形形态指标的特定错误"""
    # 修复第90行的语法错误
    content = re.sub(
        r'(\s+)def\s+([^(]+)\s*\(',
        r'\1def \2(',
        content
    )
    return content

def fix_container_key_duplication():
    """修复容器键重复问题"""
    print("\n🔧 修复容器键重复问题...")
    
    # 查找所有注册STOCK_LIST的地方
    container_files = [
        'db/__init__.py',
        'db/service_registry.py',
        'db/container.py',
        'db/managers/data_access_manager.py',
    ]
    
    fixes = 0
    
    for file_path in container_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 移除重复的STOCK_LIST注册
            if 'STOCK_LIST' in content:
                # 保留第一个注册，移除后续的
                lines = content.split('\n')
                stock_list_found = False
                new_lines = []
                
                for line in lines:
                    if 'STOCK_LIST' in line and 'register' in line:
                        if not stock_list_found:
                            new_lines.append(line)
                            stock_list_found = True
                        else:
                            # 注释掉重复的注册
                            new_lines.append(f"# {line}  # 重复注册已注释")
                            fixes += 1
                    else:
                        new_lines.append(line)
                
                content = '\n'.join(new_lines)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ 修复容器重复: {file_path}")
                
        except Exception as e:
            print(f"  ❌ 修复容器重复失败: {file_path} - {e}")
    
    print(f"  修复容器重复数: {fixes}")
    return fixes

def test_syntax_fixes():
    """测试语法修复结果"""
    print("\n🧪 测试语法修复结果...")
    
    try:
        # 测试指标注册
        from indicators.complete_indicator_registry import complete_registry
        
        # 获取注册结果
        all_indicators = complete_registry.get_all_indicators()
        indicator_count = len(all_indicators)
        
        print(f"  ✅ 指标注册测试通过: {indicator_count}个指标")
        
        # 测试数据库连接
        from db.enhanced_connection_pool import ClickHouseConnectionPool
from db.sql_manager import SQLManager, QueryType
        pool = ClickHouseConnectionPool()
        
        with pool.get_connection() as conn:
            result = conn.query_dataframe("SELECT 1 as test")
            if not result.empty:
                print(f"  ✅ 数据库连接测试通过")
            else:
                print(f"  ❌ 数据库连接测试失败")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 语法修复测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 第一阶段：修复关键语法错误")
    print("=" * 50)
    
    # 1. 修复语法错误
    files_fixed, fixes_applied = fix_common_syntax_errors()
    
    # 2. 修复容器键重复
    container_fixes = fix_container_key_duplication()
    
    # 3. 测试修复结果
    test_success = test_syntax_fixes()
    
    # 总结
    print(f"\n📊 第一阶段修复总结:")
    print(f"  语法错误修复: {files_fixed}个文件, {fixes_applied}个修复")
    print(f"  容器重复修复: {container_fixes}个修复")
    print(f"  测试结果: {'✅ 通过' if test_success else '❌ 失败'}")
    
    if test_success:
        print(f"\n🎉 第一阶段修复成功！")
        return True
    else:
        print(f"\n⚠️ 第一阶段修复需要进一步调整")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
