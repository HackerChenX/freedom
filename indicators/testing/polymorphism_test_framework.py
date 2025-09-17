"""
多态性测试框架
验证所有指标类能够通过BaseIndicator接口正确调用
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Type
from indicators.base_indicator import BaseIndicator
import importlib
import os


class PolymorphismTestFramework:
    """多态性测试框架"""

    def __init__(self):
        self.test_results = {}
        self.discovered_indicators = []

    def discover_all_indicators(self) -> List[Type[BaseIndicator]]:
        """发现所有指标类"""
        indicator_classes = []
        indicators_dir = "indicators/"

        if os.path.exists(indicators_dir):
            for root, dirs, files in os.walk(indicators_dir):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        if file not in ["base_indicator.py", "indicator_template.py"]:
                            try:
                                classes = self._extract_indicator_classes_from_file(os.path.join(root, file))
                                indicator_classes.extend(classes)
                            except Exception:
                                continue

        self.discovered_indicators = indicator_classes
        return indicator_classes

    def _extract_indicator_classes_from_file(self, file_path: str) -> List[Type[BaseIndicator]]:
        """从文件中提取指标类"""
        classes = []

        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("temp_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找BaseIndicator的子类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseIndicator) and attr != BaseIndicator:
                    classes.append(attr)

        except Exception:
            pass

        return classes

    def test_polymorphism(self) -> Dict[str, Any]:
        """测试多态性"""
        indicator_classes = self.discover_all_indicators()

        test_results = {
            "total_indicators": len(indicator_classes),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "pass_rate": 0.0,
        }

        # 创建测试数据
        test_data = self._create_test_data()

        for indicator_class in indicator_classes:
            result = self._test_single_indicator_polymorphism(indicator_class, test_data)
            test_results["test_details"].append(result)

            if result["passed"]:
                test_results["passed_tests"] += 1
            else:
                test_results["failed_tests"] += 1

        test_results["pass_rate"] = test_results["passed_tests"] / max(test_results["total_indicators"], 1) * 100

        return test_results

    def _create_test_data(self) -> pd.DataFrame:
        """创建测试数据"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")  # TODO: 将魔法数字提取到配置中

        data = pd.DataFrame(
            {
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 105,  # TODO: 将魔法数字提取到配置中
                "low": np.random.randn(100).cumsum() + 95,  # TODO: 将魔法数字提取到配置中
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(
                    1000, 10000, 100
                ),  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            },
            index=dates,
        )

        return data

    def _test_single_indicator_polymorphism(
        self, indicator_class: Type[BaseIndicator], test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """测试单个指标的多态性"""
        result = {"indicator_class": indicator_class.__name__, "passed": False, "errors": [], "interface_tests": {}}

        try:
            # 通过BaseIndicator接口创建实例
            indicator: BaseIndicator = indicator_class()

            # 测试calculate方法
            try:
                calc_result = indicator.calculate(test_data)
                result["interface_tests"]["calculate"] = True
                assert isinstance(calc_result, pd.DataFrame), "calculate应返回DataFrame"
            except Exception as e:
                result["interface_tests"]["calculate"] = False
                result["errors"].append(f"calculate方法测试失败: {e}")

            # 测试get_signal方法
            try:
                signal_result = indicator.get_signal(test_data)
                result["interface_tests"]["get_signal"] = True
                assert isinstance(signal_result, dict), "get_signal应返回字典"
                assert "signal" in signal_result, "信号结果应包含signal字段"
            except Exception as e:
                result["interface_tests"]["get_signal"] = False
                result["errors"].append(f"get_signal方法测试失败: {e}")

            # 测试get_patterns方法
            try:
                patterns_result = indicator.get_patterns(test_data)
                result["interface_tests"]["get_patterns"] = True
                assert isinstance(patterns_result, list), "get_patterns应返回列表"
            except Exception as e:
                result["interface_tests"]["get_patterns"] = False
                result["errors"].append(f"get_patterns方法测试失败: {e}")

            # 判断整体是否通过
            result["passed"] = all(result["interface_tests"].values())

        except Exception as e:
            result["errors"].append(f"指标实例化失败: {e}")

        return result


# 使用示例
if __name__ == "__main__":
    framework = PolymorphismTestFramework()
    results = framework.test_polymorphism()

    print(f"多态性测试结果:")
    print(f"总指标数: {results['total_indicators']}")
    print(f"通过测试: {results['passed_tests']}")
    print(f"失败测试: {results['failed_tests']}")
    print(f"通过率: {results['pass_rate']:.1f}%")
