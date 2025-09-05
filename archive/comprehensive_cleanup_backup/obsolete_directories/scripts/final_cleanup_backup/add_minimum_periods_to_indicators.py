#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为所有技术指标添加最小周期要求的工具

自动为所有指标类添加minimum_periods属性，确保：
1. 每个指标明确定义最少数据周期要求
2. 基于指标参数智能计算最小周期
3. 提供统一的数据验证机制
4. 支持双向验证的数据窗口确定
"""

import os
import sys
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

class IndicatorMinimumPeriodsAdder:
    """为指标添加最小周期要求的工具类"""
    
    def __init__(self, indicators_dir: str = "/Users/hacker/PycharmProjects/freedom/indicators"):
        """初始化工具"""
        self.indicators_dir = Path(indicators_dir)
        self.processed_files = []
        self.skipped_files = []
        self.errors = []
        
        # 指标特定的最小周期映射
        self.indicator_periods_mapping = {
            'macd': {'formula': 'slow_period + signal_period + 5', 'default': 40},
            'rsi': {'formula': 'period + 6', 'default': 20},
            'kdj': {'formula': 'k_period + d_period + 6', 'default': 18},
            'boll': {'formula': 'period + 5', 'default': 25},
            'ma': {'formula': 'period + 5', 'default': 25},
            'ema': {'formula': 'period + 5', 'default': 25},
            'atr': {'formula': 'period + 6', 'default': 20},
            'cci': {'formula': 'period + 5', 'default': 25},
            'mfi': {'formula': 'period + 6', 'default': 20},
            'obv': {'formula': '10', 'default': 10},
            'stochrsi': {'formula': 'rsi_period + stoch_period + 5', 'default': 25},
            'aroon': {'formula': 'period + 5', 'default': 30},
            'ichimoku': {'formula': 'max(conversion_period, base_period, leading_span_b_period) + 5', 'default': 55},
            'sar': {'formula': '15', 'default': 15},
            'adx': {'formula': 'period + 6', 'default': 20},
            'wma': {'formula': 'period + 5', 'default': 25},
            'vortex': {'formula': 'period + 6', 'default': 20},
            'emv': {'formula': 'period + 6', 'default': 20},
            'trix': {'formula': 'period * 3 + 5', 'default': 35},
            'cmo': {'formula': 'period + 6', 'default': 20},
            'roc': {'formula': 'period + 3', 'default': 15},
            'kc': {'formula': 'period + atr_period + 5', 'default': 25},
            'vix': {'formula': '30', 'default': 30},
            'volume_ratio': {'formula': '15', 'default': 15},
            'enhanced_cci': {'formula': 'period + 10', 'default': 30},
            'enhanced_dmi': {'formula': 'period + 11', 'default': 25},
        }
    
    def find_indicator_files(self) -> List[Path]:
        """查找所有指标文件"""
        indicator_files = []
        
        # 查找indicators目录下的所有.py文件
        for file_path in self.indicators_dir.rglob("*.py"):
            # 跳过__init__.py和base目录
            if file_path.name == "__init__.py" or "base" in file_path.parts:
                continue
            
            # 检查是否包含指标类定义
            if self._contains_indicator_class(file_path):
                indicator_files.append(file_path)
        
        return indicator_files
    
    def _contains_indicator_class(self, file_path: Path) -> bool:
        """检查文件是否包含指标类定义"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 查找继承自BaseIndicator的类
            pattern = r'class\s+\w+.*BaseIndicator'
            return bool(re.search(pattern, content))
            
        except Exception as e:
            self.errors.append(f"读取文件{file_path}失败: {e}")
            return False
    
    def analyze_indicator_file(self, file_path: Path) -> Dict:
        """分析指标文件，提取关键信息"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'file_path': file_path,
                'class_name': None,
                'has_minimum_periods': False,
                'has_mixin': False,
                'parameters': {},
                'suggested_periods': None
            }
            
            # 查找类名
            class_match = re.search(r'class\s+(\w+).*BaseIndicator', content)
            if class_match:
                analysis['class_name'] = class_match.group(1)
            
            # 检查是否已有minimum_periods属性
            analysis['has_minimum_periods'] = 'minimum_periods' in content
            
            # 检查是否已导入MinimumPeriodsMixin
            analysis['has_mixin'] = 'MinimumPeriodsMixin' in content
            
            # 提取参数信息
            analysis['parameters'] = self._extract_parameters(content)
            
            # 计算建议的最小周期
            analysis['suggested_periods'] = self._calculate_suggested_periods(
                file_path.stem, analysis['parameters']
            )
            
            return analysis
            
        except Exception as e:
            self.errors.append(f"分析文件{file_path}失败: {e}")
            return None
    
    def _extract_parameters(self, content: str) -> Dict:
        """从代码中提取参数信息"""
        parameters = {}
        
        # 查找__init__方法中的参数
        init_match = re.search(r'def __init__\(self[^)]*\):', content, re.DOTALL)
        if init_match:
            init_def = init_match.group(0)
            
            # 提取参数名和默认值
            param_pattern = r'(\w+):\s*\w+\s*=\s*(\d+)'
            for match in re.finditer(param_pattern, init_def):
                param_name, default_value = match.groups()
                if 'period' in param_name.lower() or param_name.lower() in ['window', 'length', 'span']:
                    parameters[param_name] = int(default_value)
        
        return parameters
    
    def _calculate_suggested_periods(self, indicator_name: str, parameters: Dict) -> int:
        """计算建议的最小周期数"""
        # 标准化指标名称
        normalized_name = indicator_name.lower().replace('_', '').replace('indicator', '')
        
        # 查找匹配的映射
        for key, config in self.indicator_periods_mapping.items():
            if key in normalized_name or normalized_name.startswith(key):
                try:
                    # 尝试使用公式计算
                    formula = config['formula']
                    
                    # 替换公式中的参数
                    for param_name, param_value in parameters.items():
                        formula = formula.replace(param_name, str(param_value))
                    
                    # 处理max函数
                    if 'max(' in formula:
                        # 简单处理max函数
                        max_match = re.search(r'max\(([^)]+)\)', formula)
                        if max_match:
                            values = [int(x.strip()) for x in max_match.group(1).split(',') if x.strip().isdigit()]
                            if values:
                                formula = formula.replace(max_match.group(0), str(max(values)))
                    
                    # 计算结果
                    if formula.isdigit():
                        return int(formula)
                    else:
                        # 尝试简单的数学表达式
                        try:
                            return eval(formula)
                        except:
                            return config['default']
                            
                except:
                    return config['default']
        
        # 如果没有找到匹配，使用通用计算
        if parameters:
            max_period = max(parameters.values())
            return max_period + max(10, max_period // 2)
        else:
            return 30  # 默认值
    
    def generate_minimum_periods_code(self, analysis: Dict) -> str:
        """生成minimum_periods属性的代码"""
        suggested_periods = analysis['suggested_periods']
        parameters = analysis['parameters']
        
        # 生成计算逻辑的注释
        if parameters:
            param_desc = ', '.join([f"{k}({v})" for k, v in parameters.items()])
            logic_desc = f"基于参数 {param_desc} 计算"
        else:
            logic_desc = "使用默认值"
        
        code = f'''
    @property
    def minimum_periods(self) -> int:
        """
        {analysis['class_name']}指标所需的最少数据周期数
        
        计算逻辑：{logic_desc}
        
        Returns:
            int: 最少需要的数据周期数
        """'''
        
        if parameters:
            # 生成动态计算代码
            param_lines = []
            for param_name in parameters.keys():
                param_lines.append(f"        {param_name} = self._parameters.get('{param_name}', {parameters[param_name]})")
            
            code += '\n' + '\n'.join(param_lines)
            
            # 生成计算表达式
            if len(parameters) == 1:
                param_name = list(parameters.keys())[0]
                code += f"\n        return {param_name} + max(10, {param_name} // 2)"
            else:
                param_names = list(parameters.keys())
                code += f"\n        return max({', '.join(param_names)}) + 10"
        else:
            # 使用固定值
            code += f"\n        return {suggested_periods}"
        
        return code
    
    def add_mixin_import(self, content: str) -> str:
        """添加MinimumPeriodsMixin导入"""
        # 查找现有的导入语句
        import_pattern = r'from indicators\.base\.\w+ import \w+'
        
        if re.search(import_pattern, content):
            # 在现有的base导入后添加
            def replace_func(match):
                return match.group(0) + '\nfrom indicators.base.minimum_periods_mixin import MinimumPeriodsMixin'
            
            return re.sub(import_pattern, replace_func, content, count=1)
        else:
            # 在BaseIndicator导入后添加
            base_import_pattern = r'from indicators\.base_indicator import BaseIndicator'
            if re.search(base_import_pattern, content):
                return re.sub(
                    base_import_pattern,
                    r'from indicators.base_indicator import BaseIndicator\nfrom indicators.base.minimum_periods_mixin import MinimumPeriodsMixin',
                    content
                )
        
        return content
    
    def add_mixin_to_class(self, content: str, class_name: str) -> str:
        """将MinimumPeriodsMixin添加到类继承中"""
        # 查找类定义
        class_pattern = f'class {class_name}\\([^)]+\\):'
        
        def replace_func(match):
            class_def = match.group(0)
            if 'MinimumPeriodsMixin' not in class_def:
                # 在最后一个继承类后添加
                return class_def.replace('):', ', MinimumPeriodsMixin):')
            return class_def
        
        return re.sub(class_pattern, replace_func, content)
    
    def process_indicator_file(self, file_path: Path) -> bool:
        """处理单个指标文件"""
        try:
            # 分析文件
            analysis = self.analyze_indicator_file(file_path)
            if not analysis or not analysis['class_name']:
                self.skipped_files.append(f"{file_path}: 无法分析")
                return False
            
            # 如果已经有minimum_periods，跳过
            if analysis['has_minimum_periods']:
                self.skipped_files.append(f"{file_path}: 已有minimum_periods")
                return False
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 添加导入
            if not analysis['has_mixin']:
                content = self.add_mixin_import(content)
                content = self.add_mixin_to_class(content, analysis['class_name'])
            
            # 生成minimum_periods代码
            periods_code = self.generate_minimum_periods_code(analysis)
            
            # 查找合适的插入位置（在__init__方法后）
            init_end_pattern = r'(\n\s+self\.is_available = True\n)'
            if re.search(init_end_pattern, content):
                content = re.sub(init_end_pattern, r'\1' + periods_code + '\n', content)
            else:
                # 如果找不到标准位置，在类的最后添加
                class_end_pattern = f'(class {analysis["class_name"]}.*?\n)(.*?)(\n\nclass|\n\ndef|\Z)'
                if re.search(class_end_pattern, content, re.DOTALL):
                    content = re.sub(
                        class_end_pattern,
                        r'\1\2' + periods_code + r'\3',
                        content,
                        flags=re.DOTALL
                    )
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.processed_files.append(f"{file_path}: 添加minimum_periods({analysis['suggested_periods']})")
            return True
            
        except Exception as e:
            self.errors.append(f"处理文件{file_path}失败: {e}")
            return False
    
    def process_all_indicators(self) -> Dict:
        """处理所有指标文件"""
        print("🔍 查找指标文件...")
        indicator_files = self.find_indicator_files()
        
        print(f"📋 找到{len(indicator_files)}个指标文件")
        
        results = {
            'total_files': len(indicator_files),
            'processed': 0,
            'skipped': 0,
            'errors': 0
        }
        
        for file_path in indicator_files:
            print(f"🔧 处理: {file_path.name}")
            
            if self.process_indicator_file(file_path):
                results['processed'] += 1
            else:
                results['skipped'] += 1
        
        results['errors'] = len(self.errors)
        
        return results
    
    def print_summary(self, results: Dict):
        """打印处理结果汇总"""
        print("\n" + "="*60)
        print("📊 处理结果汇总")
        print("="*60)
        
        print(f"📁 总文件数: {results['total_files']}")
        print(f"✅ 已处理: {results['processed']}")
        print(f"⏭️ 已跳过: {results['skipped']}")
        print(f"❌ 错误数: {results['errors']}")
        
        if self.processed_files:
            print(f"\n✅ 已处理的文件:")
            for file_info in self.processed_files:
                print(f"  {file_info}")
        
        if self.skipped_files:
            print(f"\n⏭️ 跳过的文件:")
            for file_info in self.skipped_files:
                print(f"  {file_info}")
        
        if self.errors:
            print(f"\n❌ 错误信息:")
            for error in self.errors:
                print(f"  {error}")

def main():
    """主函数"""
    print("🚀 为技术指标添加最小周期要求")
    print("="*60)
    
    # 创建处理器
    processor = IndicatorMinimumPeriodsAdder()
    
    # 处理所有指标
    results = processor.process_all_indicators()
    
    # 打印汇总
    processor.print_summary(results)
    
    print(f"\n💡 使用说明:")
    print(f"  - 每个指标现在都有minimum_periods属性")
    print(f"  - 可以通过indicator.minimum_periods获取最小周期要求")
    print(f"  - 验证器可以使用这个值确定数据窗口大小")

if __name__ == "__main__":
    main()
