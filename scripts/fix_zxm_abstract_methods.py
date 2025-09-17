#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复ZXM指标抽象方法问题

解决"Can't instantiate abstract class"错误
"""

import os
import re
from pathlib import Path
from typing import List, Dict

def find_zxm_files_with_abstract_issues() -> List[str]:
    """查找有抽象方法问题的ZXM指标文件"""
    zxm_dir = Path("indicators/zxm")
    problem_files = []
    
    # 需要修复的文件列表（基于错误日志）
    problem_indicators = [
        "elasticity_indicators.py",  # ZXM_AMPLITUDE_ELASTICITY, ZXM_RISE_ELASTICITY, ZXM_ELASTICITY
        "buy_point_indicators.py",   # ZXM_BUYPOINT_SCORE
        "score_indicators.py",       # ZXM_ELASTIC_SCORE
        "market_breadth.py",         # ZXM_VOLUME_ENERGY, ZXM_MARKET_SENTIMENT
        "selection_model.py",        # ZXM_TECHNICAL_FORM
        "trend_indicators.py",       # ZXM_DAILY_TREND_UP, ZXM_WEEKLY_TREND_UP, ZXM_MONTHLY_KDJ_TREND_UP
    ]
    
    for file_name in problem_indicators:
        file_path = zxm_dir / file_name
        if file_path.exists():
            problem_files.append(str(file_path))
    
    return problem_files

def fix_zxm_file(file_path: str) -> bool:
    """修复单个ZXM文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经导入了ZXMAbstractMethodsMixin
        if "ZXMAbstractMethodsMixin" in content:
            print(f"✅ {file_path} 已经修复过了")
            return True
        
        # 添加导入
        import_pattern = r'(from indicators\.base\.minimum_periods_mixin import MinimumPeriodsMixin\n)'
        if re.search(import_pattern, content):
            content = re.sub(
                import_pattern,
                r'\1from indicators.zxm.zxm_abstract_methods_mixin import ZXMAbstractMethodsMixin\n',
from db.sql_manager import SQLManager, QueryType
                content
            )
        else:
            # 如果没有找到MinimumPeriodsMixin导入，在其他导入后添加
            import_pattern = r'(from utils\.dependency_injection import get_logger\n)'
            if re.search(import_pattern, content):
                content = re.sub(
                    import_pattern,
                    r'\1from indicators.zxm.zxm_abstract_methods_mixin import ZXMAbstractMethodsMixin\n',
from db.sql_manager import SQLManager, QueryType
                    content
                )
        
        # 修复类定义，添加ZXMAbstractMethodsMixin
        class_patterns = [
            # 匹配各种可能的类定义模式
            r'(class \w+\(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin)\):',
            r'(class \w+\(BaseIndicator, PatternSignalMixin)\):',
            r'(class \w+\(BaseIndicator, MinimumPeriodsMixin)\):',
            r'(class \w+\(BaseIndicator)\):',
        ]
        
        for pattern in class_patterns:
            if re.search(pattern, content):
                content = re.sub(
                    pattern,
                    r'\1, ZXMAbstractMethodsMixin):',
                    content
                )
                break
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 修复完成: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ 修复失败 {file_path}: {e}")
        return False

def main():
    """主函数"""
    print("🔧 开始修复ZXM指标抽象方法问题...")
    
    # 查找需要修复的文件
    problem_files = find_zxm_files_with_abstract_issues()
    
    if not problem_files:
        print("✅ 没有发现需要修复的ZXM文件")
        return
    
    print(f"📋 发现 {len(problem_files)} 个需要修复的文件:")
    for file_path in problem_files:
        print(f"  - {file_path}")
    
    # 修复文件
    success_count = 0
    for file_path in problem_files:
        if fix_zxm_file(file_path):
            success_count += 1
    
    print(f"\n🎉 修复完成: {success_count}/{len(problem_files)} 个文件修复成功")
    
    if success_count == len(problem_files):
        print("✅ 所有ZXM指标抽象方法问题已修复！")
    else:
        print("⚠️ 部分文件修复失败，请手动检查")

if __name__ == "__main__":
    main()
