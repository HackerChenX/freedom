"""
L4核心服务层 - 统一指标管理器

基于L1L2L3成功修复经验的生产级指标管理器实现
确保单一入口原则、严格依赖注入、完整向后兼容性
"""

import importlib
from typing import Dict, List, Any, Type, Optional, Union
from threading import Lock
from utils.container import container
from utils.logger import get_logger
from indicators.base_indicator import BaseIndicator
from indicators.core.indicator_manager_interface import (
    IIndicatorManager, 
    IIndicatorFactory, 
    IIndicatorRegistry,
    DependencyInjectionError,
    IndicatorRegistrationError
)

logger = get_logger(__name__)


class UnifiedIndicatorManager(IIndicatorManager, IIndicatorFactory, IIndicatorRegistry):
    """
    统一指标管理器 - L4层唯一入口
    
    基于L3层成功经验设计的生产级指标管理器
    实现单一入口原则、严格依赖注入、完整功能覆盖
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化统一指标管理器"""
        if hasattr(self, '_initialized'):
            return
            
        # 严格依赖注入 - 不允许兜底逻辑
        try:
            self.data_access = container.resolve("DataAccessInterface")
            self.cache_service = container.resolve("ICacheService")
            self.logger_service = container.resolve("Logger") or logger
        except Exception as e:
            raise DependencyInjectionError(f"依赖注入失败: {e}")
        
        # 验证依赖注入成功
        if not self.data_access:
            raise DependencyInjectionError("DataAccessInterface服务未注册")
        if not self.cache_service:
            raise DependencyInjectionError("ICacheService服务未注册")
        
        # 初始化内部状态
        self._indicators: Dict[str, str] = {}  # name -> class_path
        self._indicator_classes: Dict[str, Type[BaseIndicator]] = {}  # name -> class
        self._indicator_aliases: Dict[str, str] = {}  # alias -> name
        self._indicator_metadata: Dict[str, Dict[str, Any]] = {}  # name -> metadata
        self._failed_indicators: List[str] = []
        self._registration_log: List[str] = []
        
        # 统计信息
        self._stats = {
            'total_registered': 0,
            'total_failed': 0,
            'success_rate': 0.0
        }
        
        # 集成现有CompleteIndicatorRegistry
        self._integrate_existing_registry()
        
        self._initialized = True
        logger.info("统一指标管理器初始化完成")
    
    def _integrate_existing_registry(self):
        """集成现有的CompleteIndicatorRegistry"""
        try:
            from indicators.complete_indicator_registry import CompleteIndicatorRegistry

            # 创建现有注册表实例并执行注册
            existing_registry = CompleteIndicatorRegistry()
            registered_count = existing_registry.register_all_indicators()

            # 获取已注册的指标
            if hasattr(existing_registry, '_indicators'):
                self._indicators.update(existing_registry._indicators)
                logger.info(f"集成现有注册表: {len(existing_registry._indicators)}个指标")

                # 更新统计
                self._stats['total_registered'] = len(existing_registry._indicators)

            # 获取失败记录
            if hasattr(existing_registry, '_failed_indicators'):
                self._failed_indicators.extend(existing_registry._failed_indicators)
                self._stats['total_failed'] = len(existing_registry._failed_indicators)

            # 更新成功率
            self._update_success_rate()

        except Exception as e:
            logger.warning(f"集成现有注册表失败: {e}")
    
    # ==================== IIndicatorManager接口实现 ====================
    
    def register_indicator(self, name: str, indicator_class: Type[BaseIndicator], 
                          aliases: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """注册指标类"""
        try:
            # 验证指标类
            if not self.validate_indicator_class(indicator_class):
                raise IndicatorRegistrationError(f"指标类验证失败: {indicator_class}")
            
            # 检查重复注册
            if name in self._indicator_classes:
                logger.warning(f"指标 {name} 已注册，将被覆盖")
            
            # 注册指标类
            self._indicator_classes[name] = indicator_class
            self._indicators[name] = f"{indicator_class.__module__}.{indicator_class.__name__}"
            
            # 注册别名
            if aliases:
                for alias in aliases:
                    if alias in self._indicator_aliases:
                        logger.warning(f"别名 {alias} 已存在，将被覆盖")
                    self._indicator_aliases[alias] = name
            
            # 保存元数据
            if metadata:
                self._indicator_metadata[name] = metadata
            
            # 更新统计
            self._stats['total_registered'] += 1
            self._update_success_rate()
            
            # 记录日志
            self._registration_log.append(f"✅ 成功注册: {name}")
            logger.info(f"指标 {name} 注册成功")
            
            return True
            
        except Exception as e:
            self._failed_indicators.append(f"{name}: {e}")
            self._stats['total_failed'] += 1
            self._update_success_rate()
            logger.error(f"指标 {name} 注册失败: {e}")
            return False
    
    def get_indicator(self, name: str) -> Optional[Type[BaseIndicator]]:
        """获取指标类"""
        # 检查别名
        if name in self._indicator_aliases:
            name = self._indicator_aliases[name]
        
        # 从缓存获取
        if name in self._indicator_classes:
            return self._indicator_classes[name]
        
        # 动态加载
        if name in self._indicators:
            try:
                class_path = self._indicators[name]
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                indicator_class = getattr(module, class_name)
                
                # 缓存类
                self._indicator_classes[name] = indicator_class
                return indicator_class
                
            except Exception as e:
                logger.error(f"动态加载指标 {name} 失败: {e}")
                return None
        
        return None
    
    def create_indicator(self, name: str, **kwargs) -> BaseIndicator:
        """创建指标实例"""
        indicator_class = self.get_indicator(name)
        if indicator_class is None:
            raise IndicatorRegistrationError(f"未注册的指标: {name}")
        
        try:
            return indicator_class(**kwargs)
        except Exception as e:
            raise IndicatorRegistrationError(f"创建指标 {name} 实例失败: {e}")
    
    def list_indicators(self) -> List[str]:
        """列出所有已注册指标"""
        return list(self._indicators.keys())
    
    def get_indicator_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """获取指标元数据"""
        return self._indicator_metadata.get(name)
    
    def is_registered(self, name: str) -> bool:
        """检查指标是否已注册"""
        return name in self._indicators or name in self._indicator_aliases
    
    def get_registration_stats(self) -> Dict[str, Any]:
        """获取注册统计信息"""
        return self._stats.copy()
    
    def validate_indicator_class(self, indicator_class: Type[BaseIndicator]) -> bool:
        """验证指标类是否符合规范"""
        try:
            # 检查是否继承BaseIndicator
            if not issubclass(indicator_class, BaseIndicator):
                return False
            
            # 检查必需的抽象方法
            required_methods = ['calculate', 'get_signal']
            for method in required_methods:
                if not hasattr(indicator_class, method):
                    return False
            
            return True
            
        except Exception:
            return False
    
    # ==================== IIndicatorFactory接口实现 ====================
    
    def create_indicator_instance(self, name: str, **kwargs) -> BaseIndicator:
        """创建指标实例"""
        return self.create_indicator(name, **kwargs)
    
    def batch_create_indicators(self, indicator_configs: List[Dict[str, Any]]) -> List[BaseIndicator]:
        """批量创建指标实例"""
        indicators = []
        for config in indicator_configs:
            name = config.get('name')
            params = {k: v for k, v in config.items() if k != 'name'}
            
            try:
                indicator = self.create_indicator(name, **params)
                indicators.append(indicator)
            except Exception as e:
                logger.error(f"批量创建指标 {name} 失败: {e}")
        
        return indicators
    
    # ==================== IIndicatorRegistry接口实现 ====================
    
    def register_indicator_safe(self, indicator_class: Type[BaseIndicator], name: str) -> bool:
        """安全注册指标"""
        return self.register_indicator(name, indicator_class)
    
    def get_registered_indicators(self) -> Dict[str, str]:
        """获取所有已注册指标"""
        return self._indicators.copy()
    
    def get_failed_indicators(self) -> List[str]:
        """获取注册失败的指标"""
        return self._failed_indicators.copy()
    
    # ==================== 内部辅助方法 ====================
    
    def _update_success_rate(self):
        """更新成功率"""
        total = self._stats['total_registered'] + self._stats['total_failed']
        if total > 0:
            self._stats['success_rate'] = (self._stats['total_registered'] / total) * 100
        else:
            self._stats['success_rate'] = 0.0


    def register_all_indicators(self) -> int:
        """
        注册所有88+个指标 - 向后兼容方法

        Returns:
            int: 成功注册的指标数量
        """
        try:
            from indicators.complete_indicator_registry import CompleteIndicatorRegistry

            # 创建临时注册表实例来执行批量注册
            temp_registry = CompleteIndicatorRegistry()
            registered_count = temp_registry.register_all_indicators()

            # 同步注册结果到统一管理器
            if hasattr(temp_registry, '_indicators'):
                self._indicators.update(temp_registry._indicators)
            if hasattr(temp_registry, '_failed_indicators'):
                self._failed_indicators.extend(temp_registry._failed_indicators)

            # 更新统计
            self._stats['total_registered'] = len(self._indicators)
            self._stats['total_failed'] = len(self._failed_indicators)
            self._update_success_rate()

            logger.info(f"批量注册完成: 成功 {registered_count} 个指标")
            return registered_count

        except Exception as e:
            logger.error(f"批量注册失败: {e}")
            return 0


# ==================== 向后兼容性包装器 ====================

class CompatibilityWrapper:
    """向后兼容性包装器"""

    def __init__(self, manager: UnifiedIndicatorManager):
        self._manager = manager

    def __getattr__(self, name):
        """代理所有属性访问到统一管理器"""
        return getattr(self._manager, name)


# 创建全局单例实例
unified_indicator_manager = UnifiedIndicatorManager()

# 向后兼容性别名 - 确保100%兼容性
indicator_manager = CompatibilityWrapper(unified_indicator_manager)
complete_registry = CompatibilityWrapper(unified_indicator_manager)

# 导出
__all__ = [
    'UnifiedIndicatorManager',
    'unified_indicator_manager',
    'indicator_manager',
    'complete_registry',
    'CompatibilityWrapper'
]
