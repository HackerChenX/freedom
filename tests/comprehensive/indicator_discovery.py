#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标发现系统

自动发现和加载系统中所有可用的技术指标
支持动态指标注册和形态验证
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from utils.logger import getLogger
from indicators.pattern_registry import get_pattern_registry, PatternRegistry

logger = getLogger(__name__)


@dataclass
class IndicatorInfo:
    """指标信息"""
    name: str
    class_name: str
    module_path: str
    indicator_class: Type
    patterns: List[str]
    pattern_count: int
    is_loadable: bool
    error_message: Optional[str] = None


class IndicatorDiscovery:
    """指标发现系统"""
    
    def __init__(self):
        """初始化指标发现系统"""
        self.pattern_registry = get_pattern_registry()
        self.discovered_indicators = {}
        self.failed_indicators = {}
        self.indicator_modules = {}
        
        # 指标基类名称列表
        self.base_indicator_classes = [
            'BaseIndicator',
            'TrendIndicatorBase', 
            'OscillatorIndicatorBase',
            'VolumeIndicatorBase',
            'PatternSignalMixin'
        ]
        
        logger.info("指标发现系统初始化完成")
    
    def discover_all_indicators(self) -> List[IndicatorInfo]:
        """
        发现所有可用指标
        
        Returns:
            List[IndicatorInfo]: 指标信息列表
        """
        logger.info("开始发现所有可用指标...")
        
        # 清空之前的发现结果
        self.discovered_indicators.clear()
        self.failed_indicators.clear()
        
        # 发现indicators目录中的所有指标
        indicators_path = Path("indicators")
        if not indicators_path.exists():
            logger.error("indicators目录不存在")
            return []
        
        # 扫描主indicators目录
        self._scan_directory(indicators_path)
        
        # 扫描子目录
        for subdir in ['trend', 'oscillator', 'volume', 'pattern', 'zxm']:
            subdir_path = indicators_path / subdir
            if subdir_path.exists():
                self._scan_directory(subdir_path)
        
        # 验证和加载指标
        self._validate_and_load_indicators()
        
        # 注册指标形态
        self._register_indicator_patterns()
        
        # 生成指标信息列表
        indicator_infos = list(self.discovered_indicators.values())
        
        logger.info(f"指标发现完成: 成功{len(indicator_infos)}个, 失败{len(self.failed_indicators)}个")
        
        # 打印发现结果摘要
        self._print_discovery_summary(indicator_infos)
        
        return indicator_infos
    
    def _scan_directory(self, directory: Path) -> None:
        """
        扫描目录中的Python文件
        
        Args:
            directory: 要扫描的目录
        """
        logger.debug(f"扫描目录: {directory}")
        
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("__") or py_file.name.endswith(".backup"):
                continue
            
            module_name = self._get_module_name(py_file)
            if module_name:
                self._discover_indicators_in_module(module_name, py_file)
    
    def _get_module_name(self, py_file: Path) -> Optional[str]:
        """
        获取Python文件的模块名
        
        Args:
            py_file: Python文件路径
            
        Returns:
            Optional[str]: 模块名
        """
        try:
            # 将文件路径转换为模块路径
            relative_path = py_file.relative_to(Path.cwd())
            module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
            module_name = ".".join(module_parts)
            return module_name
        except Exception as e:
            logger.debug(f"获取模块名失败 {py_file}: {e}")
            return None
    
    def _discover_indicators_in_module(self, module_name: str, py_file: Path) -> None:
        """
        在模块中发现指标类
        
        Args:
            module_name: 模块名
            py_file: Python文件路径
        """
        try:
            # 导入模块
            module = importlib.import_module(module_name)
            self.indicator_modules[module_name] = module
            
            # 检查模块中的所有类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_indicator_class(obj, module):
                    self._add_discovered_indicator(name, obj, module_name)
                    
        except Exception as e:
            logger.debug(f"导入模块失败 {module_name}: {e}")
            self.failed_indicators[module_name] = str(e)
    
    def _is_indicator_class(self, cls: Type, module) -> bool:
        """
        判断是否为指标类
        
        Args:
            cls: 类对象
            module: 模块对象
            
        Returns:
            bool: 是否为指标类
        """
        # 检查类是否定义在当前模块中
        if cls.__module__ != module.__name__:
            return False
        
        # 检查是否继承自基础指标类
        base_classes = [base.__name__ for base in cls.__bases__]
        mro_classes = [base.__name__ for base in cls.__mro__]
        
        # 检查是否有指标相关的基类
        indicator_keywords = ['Indicator', 'Signal', 'Pattern']
        has_indicator_base = any(
            any(keyword in base_name for keyword in indicator_keywords)
            for base_name in base_classes + mro_classes
        )
        
        # 检查是否有指标相关的方法
        methods = [method for method in dir(cls) if not method.startswith('_')]
        indicator_methods = ['calculate', 'get_patterns', 'register_patterns']
        has_indicator_methods = any(method in methods for method in indicator_methods)
        
        return has_indicator_base or has_indicator_methods
    
    def _add_discovered_indicator(self, class_name: str, indicator_class: Type, module_name: str) -> None:
        """
        添加发现的指标
        
        Args:
            class_name: 类名
            indicator_class: 指标类
            module_name: 模块名
        """
        # 生成指标名称（去除常见后缀）
        indicator_name = self._normalize_indicator_name(class_name)
        
        logger.debug(f"发现指标: {indicator_name} ({class_name}) 在 {module_name}")
        
        # 创建指标信息（暂时不加载形态信息）
        indicator_info = IndicatorInfo(
            name=indicator_name,
            class_name=class_name,
            module_path=module_name,
            indicator_class=indicator_class,
            patterns=[],
            pattern_count=0,
            is_loadable=True
        )
        
        self.discovered_indicators[indicator_name] = indicator_info
    
    def _normalize_indicator_name(self, class_name: str) -> str:
        """
        规范化指标名称
        
        Args:
            class_name: 类名
            
        Returns:
            str: 规范化的指标名称
        """
        # 移除常见的后缀
        suffixes_to_remove = ['Indicator', 'Index', 'Oscillator', 'Signal']
        name = class_name
        
        for suffix in suffixes_to_remove:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[:-len(suffix)]
                break
        
        # 处理重复的名称（如RsiRsi -> RSI）
        parts = []
        current_part = ""
        
        for char in name:
            if char.isupper() and current_part:
                parts.append(current_part)
                current_part = char
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part)
        
        # 去重相邻的相同部分
        unique_parts = []
        for part in parts:
            if not unique_parts or part.upper() != unique_parts[-1].upper():
                unique_parts.append(part)
        
        return "".join(unique_parts).upper()
    
    def _validate_and_load_indicators(self) -> None:
        """验证和加载指标"""
        logger.info("验证和加载指标...")
        
        for indicator_name, indicator_info in list(self.discovered_indicators.items()):
            try:
                # 尝试实例化指标
                indicator_instance = self._create_indicator_instance(indicator_info.indicator_class)
                
                if indicator_instance is None:
                    indicator_info.is_loadable = False
                    indicator_info.error_message = "无法实例化指标"
                    continue
                
                # 检查指标是否有必要的方法
                required_methods = ['calculate']
                missing_methods = []
                
                for method in required_methods:
                    if not hasattr(indicator_instance, method):
                        missing_methods.append(method)
                
                if missing_methods:
                    indicator_info.is_loadable = False
                    indicator_info.error_message = f"缺少必要方法: {missing_methods}"
                    continue
                
                logger.debug(f"指标 {indicator_name} 验证成功")
                
            except Exception as e:
                logger.debug(f"验证指标 {indicator_name} 失败: {e}")
                indicator_info.is_loadable = False
                indicator_info.error_message = str(e)
    
    def _create_indicator_instance(self, indicator_class: Type) -> Optional[Any]:
        """
        创建指标实例
        
        Args:
            indicator_class: 指标类
            
        Returns:
            Optional[Any]: 指标实例
        """
        try:
            # 尝试不同的实例化方式
            
            # 1. 无参数实例化
            try:
                return indicator_class()
            except TypeError:
                pass
            
            # 2. 使用常见的默认参数
            try:
                # 尝试常见的参数组合
                common_params = [
                    {'period': 14},
                    {'n': 14},
                    {'window': 14},
                    {'length': 14},
                    {'periods': 14},
                    {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                    {'k_period': 9, 'd_period': 3, 'j_period': 3},
                ]
                
                for params in common_params:
                    try:
                        return indicator_class(**params)
                    except (TypeError, ValueError):
                        continue
            except:
                pass
            
            # 3. 检查构造函数签名并提供默认值
            try:
                sig = inspect.signature(indicator_class.__init__)
                params = {}
                
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    if param.default != inspect.Parameter.empty:
                        continue  # 有默认值，跳过
                    
                    # 为常见参数提供默认值
                    if param_name in ['period', 'n', 'window', 'length']:
                        params[param_name] = 14
                    elif param_name in ['fast_period', 'fast']:
                        params[param_name] = 12
                    elif param_name in ['slow_period', 'slow']:
                        params[param_name] = 26
                    elif param_name in ['signal_period', 'signal']:
                        params[param_name] = 9
                    elif param_name in ['k_period', 'k']:
                        params[param_name] = 9
                    elif param_name in ['d_period', 'd']:
                        params[param_name] = 3
                    elif param_name in ['j_period', 'j']:
                        params[param_name] = 3
                
                return indicator_class(**params)
                
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"创建指标实例失败: {e}")
            return None
    
    def _register_indicator_patterns(self) -> None:
        """注册指标形态"""
        logger.info("注册指标形态...")
        
        for indicator_name, indicator_info in self.discovered_indicators.items():
            if not indicator_info.is_loadable:
                continue
            
            try:
                # 创建指标实例
                indicator_instance = self._create_indicator_instance(indicator_info.indicator_class)
                if indicator_instance is None:
                    continue
                
                # 尝试注册形态
                if hasattr(indicator_instance, 'register_patterns'):
                    try:
                        indicator_instance.register_patterns()
                        logger.debug(f"注册 {indicator_name} 的形态")
                    except Exception as e:
                        logger.debug(f"注册 {indicator_name} 形态失败: {e}")
                elif hasattr(indicator_instance, '_register_patterns'):
                    try:
                        indicator_instance._register_patterns()
                        logger.debug(f"注册 {indicator_name} 的形态")
                    except Exception as e:
                        logger.debug(f"注册 {indicator_name} 形态失败: {e}")
                
                # 获取已注册的形态
                patterns = self.pattern_registry.get_patterns_by_indicator(indicator_name)
                indicator_info.patterns = patterns
                indicator_info.pattern_count = len(patterns)
                
                if not patterns:
                    # 如果没有注册形态，创建默认形态
                    default_patterns = self._create_default_patterns(indicator_name)
                    indicator_info.patterns = default_patterns
                    indicator_info.pattern_count = len(default_patterns)
                
            except Exception as e:
                logger.debug(f"处理指标 {indicator_name} 的形态时出错: {e}")
    
    def _create_default_patterns(self, indicator_name: str) -> List[str]:
        """
        为指标创建默认形态
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            List[str]: 默认形态列表
        """
        default_patterns = [
            f"{indicator_name}_BULLISH",
            f"{indicator_name}_BEARISH",
            f"{indicator_name}_NEUTRAL"
        ]
        
        # 注册默认形态到形态注册表
        try:
            from indicators.pattern_registry import PatternTypePatternRegistry
            
            pattern_configs = [
                (f"{indicator_name}_BULLISH", "看涨信号", PatternTypePatternRegistry.BULLISH),
                (f"{indicator_name}_BEARISH", "看跌信号", PatternTypePatternRegistry.BEARISH),
                (f"{indicator_name}_NEUTRAL", "中性信号", PatternTypePatternRegistry.NEUTRAL),
            ]
            
            for pattern_id, display_name, pattern_type in pattern_configs:
                self.pattern_registry.register_pattern_registry(
                    pattern_id=pattern_id,
                    display_name=display_name,
                    indicator_id=indicator_name,
                    pattern_type=pattern_type,
                    allow_override=True
                )
            
            logger.debug(f"为 {indicator_name} 创建默认形态")
            
        except Exception as e:
            logger.debug(f"创建默认形态失败 {indicator_name}: {e}")
        
        return default_patterns
    
    def _print_discovery_summary(self, indicator_infos: List[IndicatorInfo]) -> None:
        """
        打印发现结果摘要
        
        Args:
            indicator_infos: 指标信息列表
        """
        logger.info("=== 指标发现结果摘要 ===")
        logger.info(f"总计发现指标: {len(indicator_infos)}")
        
        # 按类别统计
        loadable_count = sum(1 for info in indicator_infos if info.is_loadable)
        with_patterns_count = sum(1 for info in indicator_infos if info.pattern_count > 0)
        total_patterns = sum(info.pattern_count for info in indicator_infos)
        
        logger.info(f"可加载指标: {loadable_count}")
        logger.info(f"有形态指标: {with_patterns_count}")
        logger.info(f"总形态数量: {total_patterns}")
        
        # 打印前10个指标的详细信息
        logger.info("=== 指标详细信息（前10个）===")
        for i, info in enumerate(indicator_infos[:10]):
            status = "✓" if info.is_loadable else "✗"
            logger.info(f"{status} {info.name}: {info.pattern_count}个形态 ({info.class_name})")
        
        if len(indicator_infos) > 10:
            logger.info(f"... 还有 {len(indicator_infos) - 10} 个指标")
        
        # 打印失败的指标
        if self.failed_indicators:
            logger.info("=== 加载失败的模块 ===")
            for module_name, error in list(self.failed_indicators.items())[:5]:
                logger.info(f"✗ {module_name}: {error}")
    
    def get_indicator_by_name(self, indicator_name: str) -> Optional[IndicatorInfo]:
        """
        根据名称获取指标信息
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            Optional[IndicatorInfo]: 指标信息
        """
        return self.discovered_indicators.get(indicator_name.upper())
    
    def load_indicator(self, indicator_name: str) -> Optional[Any]:
        """
        加载指标实例
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            Optional[Any]: 指标实例
        """
        indicator_info = self.get_indicator_by_name(indicator_name)
        if not indicator_info or not indicator_info.is_loadable:
            return None
        
        try:
            return self._create_indicator_instance(indicator_info.indicator_class)
        except Exception as e:
            logger.error(f"加载指标 {indicator_name} 失败: {e}")
            return None
    
    def get_indicator_patterns(self, indicator_name: str) -> List[str]:
        """
        获取指标的所有形态
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            List[str]: 形态列表
        """
        indicator_info = self.get_indicator_by_name(indicator_name)
        if not indicator_info:
            return []
        
        return indicator_info.patterns
    
    def get_all_indicator_names(self) -> List[str]:
        """
        获取所有指标名称
        
        Returns:
            List[str]: 指标名称列表
        """
        return [info.name for info in self.discovered_indicators.values() if info.is_loadable]
    
    def get_indicators_with_patterns(self) -> List[IndicatorInfo]:
        """
        获取有形态的指标列表
        
        Returns:
            List[IndicatorInfo]: 有形态的指标信息列表
        """
        return [
            info for info in self.discovered_indicators.values() 
            if info.is_loadable and info.pattern_count > 0
        ]
    
    def get_discovery_statistics(self) -> Dict[str, Any]:
        """
        获取发现统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        indicator_infos = list(self.discovered_indicators.values())
        
        return {
            'total_discovered': len(indicator_infos),
            'loadable_indicators': sum(1 for info in indicator_infos if info.is_loadable),
            'indicators_with_patterns': sum(1 for info in indicator_infos if info.pattern_count > 0),
            'total_patterns': sum(info.pattern_count for info in indicator_infos),
            'failed_modules': len(self.failed_indicators),
            'average_patterns_per_indicator': (
                sum(info.pattern_count for info in indicator_infos) / len(indicator_infos)
                if indicator_infos else 0
            )
        }


def main():
    """测试指标发现系统"""
    discovery = IndicatorDiscovery()
    
    print("开始发现指标...")
    indicators = discovery.discover_all_indicators()
    
    print(f"\n发现结果:")
    print(f"总计: {len(indicators)} 个指标")
    
    stats = discovery.get_discovery_statistics()
    print(f"可加载: {stats['loadable_indicators']} 个")
    print(f"有形态: {stats['indicators_with_patterns']} 个")
    print(f"总形态: {stats['total_patterns']} 个")
    
    print(f"\n指标列表:")
    for info in indicators[:20]:  # 显示前20个
        status = "✓" if info.is_loadable else "✗"
        print(f"{status} {info.name}: {info.pattern_count}个形态")


if __name__ == "__main__":
    main()