"""
指标自动发现机制
自动扫描indicators目录，发现并注册新的指标类
"""

import os
import importlib
import inspect
from pathlib import Path
from typing import List, Type
from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorAutoDiscovery:
    """指标自动发现器"""

    def __init__(self):
        self.discovered_indicators = {}

    def discover_indicators(self, base_path: str = 'indicators') -> List[Type[BaseIndicator]]:
        """自动发现指标"""
        indicators = []
        base_dir = Path(base_path)

        if not base_dir.exists():
            return indicators

        for py_file in base_dir.rglob('*.py'):
            if (py_file.name not in ['__init__.py', 'base_indicator.py'] and
                not py_file.name.startswith('test_')):

                indicator_classes = self._extract_indicator_classes(py_file)
                indicators.extend(indicator_classes)

        logger.info(f"自动发现了{len(indicators)}个指标类")
        return indicators

    def _extract_indicator_classes(self, file_path: Path) -> List[Type[BaseIndicator]]:
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
                if (issubclass(obj, BaseIndicator) and
                    obj != BaseIndicator and
                    obj.__module__ == module.__name__):
                    indicator_classes.append(obj)

        except Exception as e:
            logger.debug(f"提取指标类失败 {file_path}: {e}")

        return indicator_classes


# 全局自动发现器实例
auto_discovery = IndicatorAutoDiscovery()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指标值"""
        if not self.validate_data(data):
            raise ValueError("输入数据不符合要求")

        result = self.preprocess_data(data).copy()
        # TODO: 实现具体的指标计算逻辑
        result[f'{self.name}_value'] = result['close'].rolling(window=getattr(self, 'period', 20)).mean()

        result = self.postprocess_result(result)
        self._result = result
        return result

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取交易信号"""
        if data.empty:
            return {'signal': 'hold', 'strength': 0.0, 'timestamp': None}

        return {
            'signal': 'hold',
            'strength': 0.0,
            'timestamp': data.index[-1] if hasattr(data, 'index') else None,
            'price': data['close'].iloc[-1] if 'close' in data.columns else 0,
            'indicator': self.name
        }
