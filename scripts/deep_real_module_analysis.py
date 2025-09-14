#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度分析real模块导入问题

找出所有导致"No module named 'real'"错误的根本原因
"""

import os
import sys
import importlib
import traceback
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)

class DeepRealModuleAnalyzer:
    """深度real模块分析器"""
    
    def __init__(self):
        self.root_dir = Path(root_dir)
        self.indicators_dir = self.root_dir / "indicators"
        self.problematic_indicators = []
        
    def analyze_indicator_imports(self):
        """分析每个指标的导入问题"""
        logger.info("🔍 开始深度分析指标导入问题...")
        
        # 获取所有指标注册信息
        try:
            from indicators.complete_indicator_registry import CompleteIndicatorRegistry
            registry = CompleteIndicatorRegistry()
            all_indicators = registry.get_all_indicators()
            
            logger.info(f"📊 总共注册了 {len(all_indicators)} 个指标")
            
            # 逐个测试指标实例化
            for name, indicator_class in all_indicators.items():
                self._test_indicator_instantiation(name, indicator_class)
                
        except Exception as e:
            logger.error(f"❌ 获取指标注册表失败: {e}")
            traceback.print_exc()
    
    def _test_indicator_instantiation(self, name: str, indicator_class):
        """测试单个指标的实例化"""
        try:
            # 尝试实例化指标
            if indicator_class is None:
                logger.warning(f"⚠️ {name}: 指标类为None")
                return
            
            # 尝试不同的实例化方式
            instance = None
            error_msg = None
            
            # 方式1: 无参数
            try:
                instance = indicator_class()
            except Exception as e1:
                error_msg = str(e1)
                
                # 方式2: 带period参数
                try:
                    instance = indicator_class(period=20)
                except Exception as e2:
                    error_msg = str(e2)
                    
                    # 方式3: 带其他常见参数
                    try:
                        instance = indicator_class(n=9, m1=3, m2=3)
                    except Exception as e3:
                        error_msg = str(e3)
            
            if instance is None:
                if "No module named 'real'" in error_msg:
                    self.problematic_indicators.append({
                        'name': name,
                        'class': indicator_class,
                        'error': error_msg,
                        'type': 'real_module_error'
                    })
                    logger.warning(f"🔴 {name}: No module named 'real' - {error_msg}")
                else:
                    logger.warning(f"⚠️ {name}: 其他实例化错误 - {error_msg}")
            else:
                logger.debug(f"✅ {name}: 实例化成功")
                
        except Exception as e:
            logger.error(f"❌ {name}: 测试失败 - {e}")
    
    def analyze_problematic_classes(self):
        """分析有问题的指标类"""
        logger.info("🔍 分析有问题的指标类...")
        
        for indicator_info in self.problematic_indicators:
            name = indicator_info['name']
            indicator_class = indicator_info['class']
            
            logger.info(f"🔍 分析 {name}...")
            
            # 获取类的模块信息
            try:
                module = indicator_class.__module__
                file_path = None
                
                if hasattr(indicator_class, '__file__'):
                    file_path = indicator_class.__file__
                else:
                    # 尝试获取模块文件路径
                    try:
                        module_obj = importlib.import_module(module)
                        if hasattr(module_obj, '__file__'):
                            file_path = module_obj.__file__
                    except:
                        pass
                
                logger.info(f"  📁 模块: {module}")
                logger.info(f"  📄 文件: {file_path}")
                
                # 检查类的源代码
                if file_path and os.path.exists(file_path):
                    self._analyze_class_source(name, file_path)
                
            except Exception as e:
                logger.error(f"  ❌ 分析 {name} 失败: {e}")
    
    def _analyze_class_source(self, name: str, file_path: str):
        """分析类的源代码"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找real相关的导入
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'real' in line.lower() and ('import' in line or 'from' in line):
                    logger.info(f"  🔍 第{i}行: {line.strip()}")
            
            # 查找__init__方法中的问题
            if 'def __init__' in content:
                logger.info(f"  📝 {name} 有__init__方法")
                
                # 提取__init__方法内容
                init_start = content.find('def __init__')
                if init_start != -1:
                    # 找到下一个方法或类的结束
                    remaining = content[init_start:]
                    init_lines = []
                    indent_level = None
                    
                    for line in remaining.split('\n'):
                        if line.strip().startswith('def __init__'):
                            indent_level = len(line) - len(line.lstrip())
                            init_lines.append(line)
                        elif indent_level is not None:
                            current_indent = len(line) - len(line.lstrip())
                            if line.strip() and current_indent <= indent_level and not line.strip().startswith('#'):
                                break
                            init_lines.append(line)
                    
                    init_content = '\n'.join(init_lines)
                    if 'real' in init_content.lower():
                        logger.info(f"  🔍 __init__方法中包含real相关代码:")
                        for line in init_lines:
                            if 'real' in line.lower():
                                logger.info(f"    {line.strip()}")
            
        except Exception as e:
            logger.error(f"  ❌ 分析源代码失败: {e}")
    
    def generate_fix_suggestions(self):
        """生成修复建议"""
        logger.info("💡 生成修复建议...")
        
        if not self.problematic_indicators:
            logger.info("✅ 没有发现real模块相关问题")
            return
        
        logger.info(f"🔴 发现 {len(self.problematic_indicators)} 个有real模块问题的指标:")
        
        for indicator_info in self.problematic_indicators:
            name = indicator_info['name']
            logger.info(f"  - {name}")
        
        logger.info("\n💡 修复建议:")
        logger.info("1. 检查这些指标类的__init__方法")
        logger.info("2. 将'from real import'改为'from indicators.real import'")
        logger.info("3. 将'import real'改为'import indicators.real as real'")
        logger.info("4. 确保indicators/real.py文件存在且完整")
    
    def run_analysis(self):
        """运行完整分析"""
        logger.info("🚀 开始深度real模块分析...")
        
        # 1. 分析指标导入问题
        self.analyze_indicator_imports()
        
        # 2. 分析有问题的类
        self.analyze_problematic_classes()
        
        # 3. 生成修复建议
        self.generate_fix_suggestions()
        
        logger.info("🎉 深度分析完成")
        
        return {
            'total_indicators': len(self.problematic_indicators),
            'problematic_indicators': [info['name'] for info in self.problematic_indicators]
        }

def main():
    """主函数"""
    print("🚀 开始深度real模块分析...")
    
    analyzer = DeepRealModuleAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n" + "="*80)
    print("🎉 深度分析完成总结")
    print("="*80)
    print(f"📊 有问题的指标数: {results['total_indicators']}")
    
    if results['problematic_indicators']:
        print("🔴 有问题的指标:")
        for name in results['problematic_indicators']:
            print(f"  - {name}")
    else:
        print("✅ 没有发现real模块相关问题")
    
    print("="*80)
    
    return 0

if __name__ == "__main__":
    exit(main())
