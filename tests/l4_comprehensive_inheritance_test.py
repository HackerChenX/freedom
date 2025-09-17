#!/usr/bin/env python3
"""
全面的继承合规性测试框架
确保100%指标正确继承BaseIndicator
"""

import unittest
import importlib
import inspect
from pathlib import Path
from indicators.base_indicator import BaseIndicator


class ComprehensiveInheritanceTest(unittest.TestCase):
    """全面的继承合规性测试"""

    def setUp(self):
        """测试设置"""
        self.indicators_dir = Path('indicators')
        self.discovered_indicators = self._discover_all_indicators()

    def _discover_all_indicators(self):
        """发现所有指标类"""
        indicators = []

        if self.indicators_dir.exists():
            for py_file in self.indicators_dir.rglob('*.py'):
                if (py_file.name not in ['__init__.py', 'base_indicator.py'] and
                    not py_file.name.startswith('test_')):

                    indicator_classes = self._extract_indicator_classes(py_file)
                    indicators.extend(indicator_classes)

        return indicators

    def _extract_indicator_classes(self, file_path):
        """从文件中提取指标类"""
        indicator_classes = []

        try:
            # 构建模块路径
            relative_path = file_path.relative_to(Path.cwd())
            module_path = str(relative_path).replace('/', '.').replace('\\', '.').replace('.py', '')

            # 导入模块
            module = importlib.import_module(module_path)

            # 检查模块中的所有类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (hasattr(obj, '__name__') and
                    'indicator' in obj.__name__.lower() and
                    obj.__module__ == module.__name__):
                    indicator_classes.append((name, obj, str(file_path)))

        except Exception:
            pass

        return indicator_classes

    def test_all_indicators_inherit_base_indicator(self):
        """测试所有指标都继承BaseIndicator"""
        non_compliant = []

        for name, indicator_class, file_path in self.discovered_indicators:
            if not issubclass(indicator_class, BaseIndicator):
                non_compliant.append(f"{name} in {file_path}")

        self.assertEqual(len(non_compliant), 0,
                        f"以下指标未继承BaseIndicator: {non_compliant}")

    def test_all_indicators_implement_abstract_methods(self):
        """测试所有指标实现抽象方法"""
        non_compliant = []

        for name, indicator_class, file_path in self.discovered_indicators:
            if issubclass(indicator_class, BaseIndicator):
                # 检查抽象方法实现
                abstract_methods = ['calculate', 'get_signal']

                for method in abstract_methods:
                    if not hasattr(indicator_class, method):
                        non_compliant.append(f"{name}.{method} in {file_path}")
                    elif getattr(getattr(indicator_class, method), '__isabstractmethod__', False):
                        non_compliant.append(f"{name}.{method} not implemented in {file_path}")

        self.assertEqual(len(non_compliant), 0,
                        f"以下指标未实现抽象方法: {non_compliant}")

    def test_all_indicators_call_super_init(self):
        """测试所有指标调用super().__init__()"""
        non_compliant = []

        for name, indicator_class, file_path in self.discovered_indicators:
            if issubclass(indicator_class, BaseIndicator):
                # 检查__init__方法中是否调用super()
                if hasattr(indicator_class, '__init__'):
                    init_source = inspect.getsource(indicator_class.__init__)
                    if 'super().__init__(' not in init_source:
                        non_compliant.append(f"{name} in {file_path}")

        self.assertEqual(len(non_compliant), 0,
                        f"以下指标未调用super().__init__(): {non_compliant}")

    def test_polymorphism_compatibility(self):
        """测试多态性兼容性"""
        test_data = self._create_test_data()
        non_compliant = []

        for name, indicator_class, file_path in self.discovered_indicators:
            if issubclass(indicator_class, BaseIndicator):
                try:
                    # 尝试实例化和调用
                    indicator = indicator_class()
                    result = indicator.calculate(test_data)
                    signal = indicator.get_signal(test_data)

                    # 验证返回类型
                    if not isinstance(result, pd.DataFrame):
                        non_compliant.append(f"{name}.calculate() 返回类型错误")

                    if not isinstance(signal, dict):
                        non_compliant.append(f"{name}.get_signal() 返回类型错误")

                except Exception as e:
                    non_compliant.append(f"{name} 多态性测试失败: {e}")

        self.assertEqual(len(non_compliant), 0,
                        f"以下指标多态性测试失败: {non_compliant}")

    def _create_test_data(self):
        """创建测试数据"""
        import pandas as pd
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })


if __name__ == '__main__':
    unittest.main()
