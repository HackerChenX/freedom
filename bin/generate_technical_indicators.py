#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标生成工具

根据技术指标大全文档，自动生成增量的技术指标模块框架代码
"""

import os
import sys
import re
from collections import defaultdict

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger
from utils.path_utils import get_indicators_dir, get_doc_dir, ensure_dir_exists

logger = get_logger(__name__)


class IndicatorTemplate:
    """技术指标模板生成器"""
    
    def __init__(self, name, full_name, category, description):
        self.name = name  # 短名称，如'cci'
        self.full_name = full_name  # 全名，如'顺势指标(CCI)'
        self.category = category  # 分类，如'震荡类指标'
        self.description = description  # 描述信息
        
    def generate_code(self):
        """生成指标代码模板"""
        class_name = self.name.upper()
        
        # 生成模板代码
        code = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
{self.full_name}

{self.description}
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class {class_name}(BaseIndicator):
    """
    {self.full_name} ({self.name.upper()})
    
    分类：{self.category}
    描述：{self.description}
    """
    
    def __init__(self, period: int = 14):
        """
        初始化{self.full_name}指标
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__()
        self.period = period
        self.name = "{self.name.upper()}"
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算{self.full_name}指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了{self.name.upper()}指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # TODO: 实现{self.full_name}计算逻辑
        # 示例计算逻辑，请根据实际指标修改
        df_copy[f'{self.name.upper()}'] = None
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成{self.full_name}指标交易信号
        
        Args:
            df: 包含价格数据和{self.name.upper()}指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值
                oversold: 超卖阈值
                
        Returns:
            添加了信号列的DataFrame:
            - {self.name.lower()}_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = [f'{self.name.upper()}']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        overbought = kwargs.get('overbought', 80)  # 超买阈值
        oversold = kwargs.get('oversold', 20)  # 超卖阈值
        
        # TODO: 实现信号生成逻辑
        # 示例信号逻辑，请根据实际指标修改
        df_copy[f'{self.name.lower()}_signal'] = 0
        
        # 示例：超卖区域上穿信号线为买入信号
        # df_copy.loc[crossover(df_copy[f'{self.name.upper()}'], oversold), f'{self.name.lower()}_signal'] = 1
        
        # 示例：超买区域下穿信号线为卖出信号
        # df_copy.loc[crossunder(df_copy[f'{self.name.upper()}'], overbought), f'{self.name.lower()}_signal'] = -1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制{self.full_name}指标图表
        
        Args:
            df: 包含{self.name.upper()}指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = [f'{self.name.upper()}']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制{self.name.upper()}指标线
        ax.plot(df.index, df[f'{self.name.upper()}'], label=f'{self.full_name}')
        
        # TODO: 根据指标类型添加适当的参考线或其他元素
        # 示例：添加超买超卖参考线
        # ax.axhline(y=overbought, color='r', linestyle='--', alpha=0.3)
        # ax.axhline(y=oversold, color='g', linestyle='--', alpha=0.3)
        
        ax.set_ylabel(f'{self.full_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

'''
        return code
        
    def generate_enum_entry(self):
        """生成枚举类型条目"""
        return f'    {self.name.upper()} = "{self.name.upper()}"  # {self.full_name}'
        
    def generate_factory_entry(self):
        """生成工厂方法条目"""
        return f'''        elif indicator_type == IndicatorType.{self.name.upper()}:
            from indicators.{self.name.lower()} import {self.name.upper()}
            return {self.name.upper()}(**kwargs)'''


def extract_indicators_from_doc():
    """从技术指标大全文档中提取指标信息"""
    doc_path = os.path.join(get_doc_dir(), "股票技术分析指标大全.md")
    
    if not os.path.exists(doc_path):
        logger.error(f"文档不存在: {doc_path}")
        return []
    
    indicators = []
    current_category = ""
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 提取各类指标章节
    sections = re.split(r'## [一二三四五六七八九十]+、', content)[1:]
    category_titles = re.findall(r'## [一二三四五六七八九十]+、(.*?)$', content, re.MULTILINE)
    
    for idx, (title, section) in enumerate(zip(category_titles, sections)):
        category = title.strip()
        
        # 提取各个指标
        indicator_blocks = re.split(r'### \d+\. ', section)[1:]
        indicator_names = re.findall(r'### \d+\. (.*?)$', section, re.MULTILINE)
        
        for name, block in zip(indicator_names, indicator_blocks):
            # 提取指标名称和简称
            match = re.search(r'(.*?)\((.*?)\)', name)
            if match:
                full_name = name.strip()
                short_name = match.group(2).lower()
                
                # 提取描述（取第一个特点或应用作为描述）
                description_match = re.search(r'\*\*特点\*\*：(.*?)$|\*\*应用\*\*：(.*?)$|\*\*计算\*\*：(.*?)$', 
                                             block, re.MULTILINE)
                description = ""
                if description_match:
                    for group in description_match.groups():
                        if group:
                            description = group.strip()
                            break
                
                indicators.append({
                    'short_name': short_name,
                    'full_name': full_name,
                    'category': category,
                    'description': description
                })
    
    return indicators


def get_existing_indicators():
    """获取已存在的技术指标模块"""
    indicators_dir = get_indicators_dir()
    existing = []
    
    for file in os.listdir(indicators_dir):
        if file.endswith('.py') and not file.startswith('__') and file != 'base_indicator.py' and file != 'common.py' and file != 'factory.py':
            existing.append(file[:-3])  # 去掉.py后缀
    
    return existing


def update_enum_file(new_indicators):
    """更新技术指标枚举文件"""
    enum_dir = os.path.join(root_dir, 'enums')
    enum_file = os.path.join(enum_dir, 'indicator_types.py')
    
    ensure_dir_exists(enum_dir)
    
    # 读取现有枚举文件内容
    existing_content = ""
    if os.path.exists(enum_file):
        with open(enum_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    
    # 如果文件不存在或不包含枚举类定义，创建新文件
    if not existing_content or 'class IndicatorType' not in existing_content:
        enum_code = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标类型枚举

定义系统支持的各种技术指标类型
"""

from enum import Enum, auto


class IndicatorType(str, Enum):
    """技术指标类型枚举"""
    
    # 现有指标
    MA = "MA"  # 移动平均线
    EMA = "EMA"  # 指数移动平均线
    MACD = "MACD"  # 平滑异同移动平均线
    KDJ = "KDJ"  # 随机指标
    RSI = "RSI"  # 相对强弱指标
    BOLL = "BOLL"  # 布林带
    
    # 新增指标
'''
        
        # 添加新指标
        for indicator in new_indicators:
            template = IndicatorTemplate(
                indicator['short_name'],
                indicator['full_name'],
                indicator['category'],
                indicator['description']
            )
            enum_code += template.generate_enum_entry() + '\n'
        
        with open(enum_file, 'w', encoding='utf-8') as f:
            f.write(enum_code)
        
        logger.info(f"已创建枚举文件: {enum_file}")
    else:
        # 已存在枚举文件，追加新指标
        enum_class_match = re.search(r'class IndicatorType.*?:(.*?)(?=\n\n|\Z)', existing_content, re.DOTALL)
        if enum_class_match:
            enum_class_content = enum_class_match.group(1)
            
            # 检查哪些指标尚未添加
            existing_indicators = re.findall(r'(\w+)\s*=', enum_class_content)
            
            new_entries = []
            for indicator in new_indicators:
                if indicator['short_name'].upper() not in existing_indicators:
                    template = IndicatorTemplate(
                        indicator['short_name'],
                        indicator['full_name'],
                        indicator['category'],
                        indicator['description']
                    )
                    new_entries.append(template.generate_enum_entry())
            
            if new_entries:
                # 在枚举类的最后添加新条目
                updated_content = existing_content.replace(
                    'class IndicatorType', 
                    'class IndicatorType', 
                    1
                )
                
                # 找到类定义的结束位置
                class_end = updated_content.find('\n\n', updated_content.find('class IndicatorType'))
                if class_end == -1:
                    class_end = len(updated_content)
                
                # 插入新条目
                updated_content = (
                    updated_content[:class_end] + 
                    '\n    # 新增指标\n' + 
                    '\n'.join(new_entries) + 
                    '\n' + 
                    updated_content[class_end:]
                )
                
                with open(enum_file, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                logger.info(f"已更新枚举文件: {enum_file}")
            else:
                logger.info("枚举文件中已包含所有指标，无需更新")


def update_factory_file(new_indicators):
    """更新指标工厂文件"""
    factory_file = os.path.join(get_indicators_dir(), 'factory.py')
    
    if not os.path.exists(factory_file):
        logger.error(f"工厂文件不存在: {factory_file}")
        return
    
    with open(factory_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到create_indicator方法
    create_method_match = re.search(r'def create_indicator.*?:.*?return.*?$', content, re.DOTALL)
    if not create_method_match:
        logger.error("无法在工厂文件中找到create_indicator方法")
        return
    
    method_content = create_method_match.group(0)
    
    # 检查哪些指标尚未添加到工厂方法中
    existing_indicators = re.findall(r'IndicatorType\.(\w+)', method_content)
    
    new_entries = []
    for indicator in new_indicators:
        if indicator['short_name'].upper() not in existing_indicators:
            template = IndicatorTemplate(
                indicator['short_name'],
                indicator['full_name'],
                indicator['category'],
                indicator['description']
            )
            new_entries.append(template.generate_factory_entry())
    
    if new_entries:
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
                
                logger.info(f"已更新工厂文件: {factory_file}")
                return
    
    logger.info("工厂文件中已包含所有指标，无需更新")


def generate_indicator_modules(indicators):
    """生成技术指标模块文件"""
    indicators_dir = get_indicators_dir()
    
    for indicator in indicators:
        file_path = os.path.join(indicators_dir, f"{indicator['short_name'].lower()}.py")
        
        # 如果文件已存在，跳过
        if os.path.exists(file_path):
            logger.info(f"指标文件已存在，跳过: {file_path}")
            continue
        
        # 生成代码
        template = IndicatorTemplate(
            indicator['short_name'],
            indicator['full_name'],
            indicator['category'],
            indicator['description']
        )
        code = template.generate_code()
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"已生成指标文件: {file_path}")


def main():
    """主函数"""
    logger.info("开始生成技术指标模块")
    
    # 获取文档中的指标
    indicators = extract_indicators_from_doc()
    logger.info(f"从文档中提取到 {len(indicators)} 个技术指标")
    
    # 获取已存在的指标
    existing = get_existing_indicators()
    logger.info(f"系统中已存在 {len(existing)} 个技术指标")
    
    # 找出需要新增的指标
    new_indicators = [ind for ind in indicators if ind['short_name'].lower() not in existing]
    logger.info(f"需要新增 {len(new_indicators)} 个技术指标")
    
    if not new_indicators:
        logger.info("没有需要新增的指标，退出")
        return
    
    # 更新枚举文件
    update_enum_file(new_indicators)
    
    # 更新工厂文件
    update_factory_file(new_indicators)
    
    # 生成指标模块
    generate_indicator_modules(new_indicators)
    
    logger.info("技术指标模块生成完成")


if __name__ == "__main__":
    main() 