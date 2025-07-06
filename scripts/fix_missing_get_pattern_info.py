#!/usr/bin/env python3
"""
批量修复缺失get_pattern_info方法的脚本
"""

import os
import re
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.get_logger(__name__)

def get_pattern_info_template_Info():
    """返回get_pattern_info方法的模板"""
    return '''
    def get_pattern_info_Fix_Missing_Get_Pattern_Info_Fix_Missing_Get_Pattern_Info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
'''

def has_get_pattern_info_method(file_content: str) -> bool:
    """检查文件是否已有get_pattern_info方法"""
    return 'def get_pattern_info_Fix_Missing_Get_Pattern_Info_Fix_Missing_Get_Pattern_Info(' in file_content

def add_get_pattern_info_method(file_path: str) -> bool:
    """为指定文件添加get_pattern_info方法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已有该方法
        if has_get_pattern_info_method(content):
            logger.debug(f"文件 {file_path} 已有get_pattern_info方法，跳过")
            return False
        
        # 检查是否是指标类文件
        if not ('class ' in content and 'BaseIndicator' in content):
            logger.debug(f"文件 {file_path} 不是指标类文件，跳过")
            return False

        # 跳过注册表文件
        if 'indicator_registry.py' in file_path:
            logger.debug(f"跳过注册表文件: {file_path}")
            return False
        
        # 找到类的结束位置（最后一个方法后）
        lines = content.split('\n')
        insert_position = -1
        
        # 寻找最后一个方法定义的位置
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                # 找到最后一个非空非注释行
                insert_position = i + 1
                break
        
        if insert_position == -1:
            logger.warning(f"无法确定插入位置: {file_path}")
            return False
        
        # 插入get_pattern_info方法
        method_lines = get_pattern_info_template_Info().split('\n')
        lines[insert_position:insert_position] = method_lines
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"✅ 已为 {file_path} 添加get_pattern_info方法")
        return True
        
    except Exception as e:
        logger.error(f"❌ 处理文件 {file_path} 时出错: {e}")
        return False

def find_indicator_files():
    """查找所有指标文件"""
    indicator_files = []
    
    # 搜索indicators目录下的所有Python文件
    indicators_dir = Path('indicators')
    if indicators_dir.exists():
        for py_file in indicators_dir.rglob('*.py'):
            if py_file.name != '__init__.py':
                indicator_files.append(str(py_file))
    
    return indicator_files

def main_fixmissinggetpatterninfo():
    """主函数"""
    logger.info("开始批量修复缺失get_pattern_info方法的问题...")
    
    # 查找所有指标文件
    indicator_files = find_indicator_files()
    logger.info(f"找到 {len(indicator_files)} 个指标文件")
    
    # 统计修复结果
    success_count = 0
    total_count = len(indicator_files)
    
    # 逐个处理文件
    for file_path in indicator_files:
        if add_get_pattern_info_method(file_path):
            success_count += 1
    
    # 输出结果
    logger.info(f"\n📊 批量修复结果:")
    logger.info(f"  总文件数: {total_count}")
    logger.info(f"  成功修复: {success_count}")
    logger.info(f"  修复率: {(success_count/total_count)*100:.1f}%")
    
    if success_count > 0:
        logger.info(f"\n🎉 成功为 {success_count} 个指标文件添加了get_pattern_info方法！")
    else:
        logger.info(f"\n✅ 所有指标文件都已有get_pattern_info方法")

if __name__ == "__main__":
    main_fixmissinggetpatterninfo()
