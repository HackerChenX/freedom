"""
统一指标管理器
防止指标重复实现，提供统一的指标注册和管理机制
"""

from typing import Dict, List, Any, Type, Optional
from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedIndicatorManager:
    """统一指标管理器"""

    def __init__(self):
        self._registered_indicators: Dict[str, Type[BaseIndicator]] = {}
        self._indicator_aliases: Dict[str, str] = {}
        self._indicator_metadata: Dict[str, Dict[str, Any]] = {}

    def register_indicator(self, name: str, indicator_class: Type[BaseIndicator],
                          aliases: List[str] = None, metadata: Dict[str, Any] = None):
        """注册指标"""
        if name in self._registered_indicators:
            logger.warning(f"指标 {name} 已经注册，将被覆盖")

        # 验证指标类
        if not issubclass(indicator_class, BaseIndicator):
            raise ValueError(f"指标类 {indicator_class} 必须继承自 BaseIndicator")

        self._registered_indicators[name] = indicator_class

        # 注册别名
        if aliases:
            for alias in aliases:
                if alias in self._indicator_aliases:
                    logger.warning(f"指标别名 {alias} 已经存在，将被覆盖")
                self._indicator_aliases[alias] = name

        # 保存元数据
        if metadata:
            self._indicator_metadata[name] = metadata

        logger.info(f"指标 {name} 注册成功")

    def get_indicator(self, name: str) -> Optional[Type[BaseIndicator]]:
        """获取指标类"""
        # 检查别名
        if name in self._indicator_aliases:
            name = self._indicator_aliases[name]

        return self._registered_indicators.get(name)

    def list_indicators(self) -> List[str]:
        """列出所有注册的指标"""
        return list(self._registered_indicators.keys())

    def check_duplicates(self) -> Dict[str, List[str]]:
        """检查重复指标"""
        duplicates = {}

        # 基于功能相似性检测重复
        for name1, class1 in self._registered_indicators.items():
            for name2, class2 in self._registered_indicators.items():
                if name1 != name2 and self._is_similar_indicator(class1, class2):
                    if name1 not in duplicates:
                        duplicates[name1] = []
                    duplicates[name1].append(name2)

        return duplicates

    def _is_similar_indicator(self, class1: Type[BaseIndicator], class2: Type[BaseIndicator]) -> bool:
        """判断两个指标是否相似"""
        # 简化实现：基于类名相似性
        name1 = class1.__name__.lower()
        name2 = class2.__name__.lower()

        # 检查是否包含相同的核心关键词
        keywords = ['macd', 'rsi', 'ma', 'ema', 'sma', 'bollinger', 'kdj', 'atr']

        for keyword in keywords:
            if keyword in name1 and keyword in name2:
                return True

        return False

    def get_indicator_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """获取指标元数据"""
        if name in self._indicator_aliases:
            name = self._indicator_aliases[name]

        return self._indicator_metadata.get(name)

    def unregister_indicator(self, name: str) -> bool:
        """注销指标"""
        if name in self._registered_indicators:
            del self._registered_indicators[name]

            # 删除相关别名
            aliases_to_remove = [alias for alias, target in self._indicator_aliases.items() if target == name]
            for alias in aliases_to_remove:
                del self._indicator_aliases[alias]

            # 删除元数据
            if name in self._indicator_metadata:
                del self._indicator_metadata[name]

            logger.info(f"指标 {name} 注销成功")
            return True

        return False


# 全局指标管理器实例
unified_indicator_manager = UnifiedIndicatorManager()

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
