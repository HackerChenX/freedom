"""
策略工厂模块

提供创建和管理选股策略的工厂类
"""

import importlib
from typing import Dict, List, Any, Optional, Type

from strategy.base_strategy import BaseStrategy
from strategy.momentum_strategy import MomentumStrategy
from strategy.rebound_strategy import ReboundStrategy
from strategy.breakout_strategy import BreakoutStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


class StrategyFactory:
    """
    策略工厂类
    
    用于创建和管理选股策略
    """
    
    # 已注册的策略类
    _strategies: Dict[str, Type[BaseStrategy]] = {}
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        注册策略类
        
        Args:
            name: 策略名称
            strategy_class: 策略类
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"策略类必须继承自BaseStrategy，但提供的是{strategy_class.__name__}")
        
        cls._strategies[name.lower()] = strategy_class
        logger.debug(f"注册策略类: {name}")
    
    @classmethod
    def create_strategy(cls, name: str, **kwargs) -> BaseStrategy:
        """
        创建策略实例
        
        Args:
            name: 策略名称
            kwargs: 传递给策略构造函数的参数
            
        Returns:
            BaseStrategy: 策略实例
            
        Raises:
            ValueError: 如果策略名称未注册
        """
        name = name.lower()
        if name not in cls._strategies:
            raise ValueError(f"未注册的策略名称: {name}")
        
        strategy_class = cls._strategies[name]
        try:
            strategy = strategy_class(**kwargs)
            
            # 如果有额外参数，设置到策略中
            if kwargs:
                strategy.set_parameters(kwargs)
                
            return strategy
        except Exception as e:
            logger.error(f"创建策略 {name} 时出错: {e}")
            raise
    
    @classmethod
    def get_registered_strategies(cls) -> List[str]:
        """
        获取已注册的策略名称列表
        
        Returns:
            List[str]: 已注册的策略名称列表
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[BaseStrategy]]:
        """
        获取策略类
        
        Args:
            name: 策略名称
            
        Returns:
            Optional[Type[BaseStrategy]]: 策略类，如果不存在则返回None
        """
        return cls._strategies.get(name.lower())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        检查策略是否已注册
        
        Args:
            name: 策略名称
            
        Returns:
            bool: 是否已注册
        """
        return name.lower() in cls._strategies
    
    @classmethod
    def load_strategies_from_module(cls, module_name: str) -> int:
        """
        从模块中加载策略类
        
        Args:
            module_name: 模块名称
            
        Returns:
            int: 加载的策略类数量
        """
        try:
            module = importlib.import_module(module_name)
            count = 0
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseStrategy) and 
                    attr != BaseStrategy):
                    # 使用类名作为策略名称
                    cls.register_strategy(attr_name, attr)
                    count += 1
            
            return count
        except Exception as e:
            logger.error(f"从模块 {module_name} 加载策略类时出错: {e}")
            return 0
            
    @classmethod
    def create_strategy_from_analysis(cls, analysis_result: Dict, strategy_type: str = None) -> BaseStrategy:
        """
        根据分析结果创建策略
        
        Args:
            analysis_result: 回测分析结果
            strategy_type: 策略类型，如"回踩反弹"、"横盘突破"等，如果为None则根据分析结果自动判断
            
        Returns:
            BaseStrategy: 策略实例
        """
        if not analysis_result or 'common_patterns' not in analysis_result:
            raise ValueError("无效的分析结果")
            
        # 如果未指定策略类型，则根据共性特征自动判断
        if strategy_type is None:
            common_patterns = analysis_result.get('common_patterns', [])
            if not common_patterns:
                raise ValueError("分析结果中没有找到共性特征")
                
            # 根据最常见的形态特征判断策略类型
            pattern_map = {
                'KDJ金叉': 'momentum_strategy',
                'MACD金叉': 'momentum_strategy',
                '回踩5日均线反弹': 'rebound_strategy',
                'MA5上穿MA10': 'rebound_strategy',
                '均线多头排列': 'momentum_strategy',
                'BOLL中轨上穿': 'breakout_strategy',
                '成交量放大': 'breakout_strategy'
            }
            
            # 计算每种策略类型的权重
            strategy_weights = {
                'momentum_strategy': 0,
                'rebound_strategy': 0,
                'breakout_strategy': 0
            }
            
            for pattern in common_patterns:
                pattern_name = pattern['pattern']
                if pattern_name in pattern_map:
                    strategy_weights[pattern_map[pattern_name]] += pattern['ratio']
            
            # 选择权重最高的策略类型
            strategy_type = max(strategy_weights, key=strategy_weights.get)
            logger.info(f"根据分析结果自动选择策略类型: {strategy_type}")
        
        # 创建策略实例
        if strategy_type == 'rebound_strategy' or strategy_type == '回踩反弹':
            return cls.create_strategy('回踩反弹')
        elif strategy_type == 'breakout_strategy' or strategy_type == '横盘突破':
            return cls.create_strategy('横盘突破')
        elif strategy_type == 'momentum_strategy' or strategy_type == '动量策略':
            return cls.create_strategy('动量策略')
        else:
            raise ValueError(f"不支持的策略类型: {strategy_type}")


# 注册内置策略
StrategyFactory.register_strategy("动量策略", MomentumStrategy)
StrategyFactory.register_strategy("回踩反弹", ReboundStrategy)
StrategyFactory.register_strategy("横盘突破", BreakoutStrategy) 