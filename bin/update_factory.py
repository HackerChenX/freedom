#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指标工厂更新工具

根据已生成的技术指标模块，更新指标工厂文件，添加新的指标导入和注册
"""

import os
import sys
import re
from typing import List, Dict

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger
from utils.path_utils import get_indicators_dir
from enums.indicator_types import IndicatorType

logger = get_logger(__name__)


def get_all_indicator_modules() -> List[str]:
    """
    获取所有技术指标模块名称
    
    Returns:
        List[str]: 技术指标模块名称列表
    """
    indicators_dir = get_indicators_dir()
    module_names = []
    
    for file in os.listdir(indicators_dir):
        if (file.endswith('.py') and 
            not file.startswith('__') and 
            file != 'base_indicator.py' and 
            file != 'common.py' and 
            file != 'factory.py'):
            module_names.append(file[:-3])  # 去掉.py后缀
    
    return module_names


def get_indicator_class_names() -> Dict[str, str]:
    """
    获取所有技术指标类名和对应的模块名
    
    Returns:
        Dict[str, str]: 键为指标类名（大写），值为模块名（小写）
    """
    result = {}
    
    # 从IndicatorType枚举中获取所有指标类名
    for indicator in IndicatorType:
        if isinstance(indicator.value, str):  # 跳过auto()生成的枚举值
            indicator_name = indicator.name
            module_name = indicator_name.lower()
            result[indicator_name] = module_name
    
    return result


def update_factory_imports(factory_file: str, indicator_modules: Dict[str, str]) -> bool:
    """
    更新工厂文件的导入语句
    
    Args:
        factory_file: 工厂文件路径
        indicator_modules: 指标类名和模块名映射
        
    Returns:
        bool: 是否更新成功
    """
    with open(factory_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找import部分
    import_section_match = re.search(r'import.*?from indicators\.base_indicator.*?\n\n', content, re.DOTALL)
    if not import_section_match:
        logger.error("无法在工厂文件中找到导入部分")
        return False
    
    import_section = import_section_match.group(0)
    
    # 检查哪些导入语句需要添加
    existing_imports = re.findall(r'from indicators\.(\w+) import (\w+)', content)
    existing_modules = {module: class_name for module, class_name in existing_imports}
    
    new_imports = []
    for class_name, module_name in indicator_modules.items():
        if module_name not in existing_modules:
            new_imports.append(f"from indicators.{module_name} import {class_name}")
    
    if not new_imports:
        logger.info("工厂文件已包含所有指标导入语句，无需更新")
        return False
    
    # 在导入部分末尾添加新的导入语句
    updated_imports = import_section.rstrip() + "\n" + "\n".join(new_imports) + "\n\n"
    updated_content = content.replace(import_section, updated_imports)
    
    with open(factory_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    logger.info(f"已添加 {len(new_imports)} 个指标导入语句到工厂文件")
    return True


def update_factory_registrations(factory_file: str, indicator_modules: Dict[str, str]) -> bool:
    """
    更新工厂文件的指标注册语句
    
    Args:
        factory_file: 工厂文件路径
        indicator_modules: 指标类名和模块名映射
        
    Returns:
        bool: 是否更新成功
    """
    with open(factory_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找注册部分
    registration_section_match = re.search(r'# 注册内置指标.*?$', content, re.DOTALL | re.MULTILINE)
    if not registration_section_match:
        logger.error("无法在工厂文件中找到注册部分")
        return False
    
    registration_section = registration_section_match.group(0)
    
    # 检查哪些注册语句需要添加
    existing_registrations = re.findall(r'IndicatorFactory\.register_indicator\("(\w+)",\s*(\w+)\)', content)
    existing_indicators = {indicator: class_name for indicator, class_name in existing_registrations}
    
    new_registrations = []
    for class_name in indicator_modules.keys():
        if class_name not in existing_indicators:
            new_registrations.append(f'IndicatorFactory.register_indicator("{class_name}", {class_name})')
    
    if not new_registrations:
        logger.info("工厂文件已包含所有指标注册语句，无需更新")
        return False
    
    # 在注册部分末尾添加新的注册语句
    updated_registrations = registration_section + "\n" + "\n".join(new_registrations)
    updated_content = content.replace(registration_section, updated_registrations)
    
    with open(factory_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    logger.info(f"已添加 {len(new_registrations)} 个指标注册语句到工厂文件")
    return True


def update_create_indicator_method(factory_file: str, indicator_modules: Dict[str, str]) -> bool:
    """
    更新工厂文件的create_indicator方法
    
    Args:
        factory_file: 工厂文件路径
        indicator_modules: 指标类名和模块名映射
        
    Returns:
        bool: 是否更新成功
    """
    with open(factory_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到create_indicator方法
    create_method_match = re.search(r'def create_indicator.*?:.*?return.*?$', content, re.DOTALL)
    if not create_method_match:
        logger.error("无法在工厂文件中找到create_indicator方法")
        return False
    
    method_content = create_method_match.group(0)
    
    # 检查哪些指标尚未添加到create_indicator方法中
    existing_indicators = re.findall(r'indicator_type == IndicatorType\.(\w+)', method_content)
    
    new_entries = []
    for class_name, module_name in indicator_modules.items():
        if class_name not in existing_indicators:
            new_entries.append(f'''        elif indicator_type == IndicatorType.{class_name}:
            from indicators.{module_name} import {class_name}
            return {class_name}(**kwargs)''')
    
    if not new_entries:
        logger.info("create_indicator方法已包含所有指标，无需更新")
        return False
    
    # 找到最后一个elif语句
    last_elif_pos = method_content.rfind('elif')
    if last_elif_pos != -1:
        # 找到该elif语句所在行的结束位置
        line_end = method_content.find('\n', last_elif_pos)
        if line_end != -1:
            # 在此位置插入新条目
            updated_method = (
                method_content[:line_end+1] + 
                '\n' + '\n'.join(new_entries) + 
                method_content[line_end:]
            )
            
            # 更新整个文件内容
            updated_content = content.replace(method_content, updated_method)
            
            with open(factory_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"已添加 {len(new_entries)} 个指标到create_indicator方法")
            return True
    
    logger.info("create_indicator方法已包含所有指标，无需更新")
    return False


def main():
    """主函数"""
    logger.info("开始更新指标工厂文件")
    
    # 获取所有指标模块
    indicator_modules = get_indicator_class_names()
    logger.info(f"发现 {len(indicator_modules)} 个技术指标")
    
    factory_file = os.path.join(get_indicators_dir(), 'factory.py')
    if not os.path.exists(factory_file):
        logger.error(f"工厂文件不存在: {factory_file}")
        return
    
    # 更新导入语句
    import_updated = update_factory_imports(factory_file, indicator_modules)
    
    # 更新注册语句
    reg_updated = update_factory_registrations(factory_file, indicator_modules)
    
    # 更新create_indicator方法
    method_updated = update_create_indicator_method(factory_file, indicator_modules)
    
    if import_updated or reg_updated or method_updated:
        logger.info("指标工厂文件更新成功")
    else:
        logger.info("指标工厂文件无需更新")


if __name__ == "__main__":
    main() 