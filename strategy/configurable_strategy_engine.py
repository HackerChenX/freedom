from utils.container import container
from strategy.unified_base_strategy import UnifiedBaseStrategy
#!/usr/bin/env python3
"""
可配置策略引擎
基于配置文件的动态策略组合系统，支持灵活的策略定义和组合
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger
from strategy.strategy_parser import StrategyParser
from strategy.enhanced_strategy_config_engine import EnhancedStrategyConfigEngine
from strategy.strategy_executor import StrategyExecutor
from indicators.complete_indicator_registry import CompleteIndicatorRegistry
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

class ConfigurableStrategyEngine:
    """可配置策略引擎"""
    
    def __init__(self, config_dir: str = "config/strategies"):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化可配置策略引擎
        
        Args:
            config_dir: 策略配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.strategy_parser = StrategyParser()
        self.config_engine = EnhancedStrategyConfigEngine()
        self.executor = StrategyExecutor()
        self.indicator_registry = CompleteIndicatorRegistry()
        
        # 加载所有策略配置
        self.available_strategies = {}
        self.strategy_combinations = {}
        self._load_all_strategies()
        
        logger.info(f"可配置策略引擎初始化完成，加载 {len(self.available_strategies)} 个策略")
    
    def _load_all_strategies(self):
        """加载所有策略配置文件"""
        if not self.config_dir.exists():
            logger.warning(f"策略配置目录不存在: {self.config_dir}")
            return
        
        # 加载单个策略配置
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                strategy_config = self._load_strategy_config(config_file)
                if strategy_config:
                    strategy_id = strategy_config.get('strategy', {}).get('id', config_file.stem)
                    self.available_strategies[strategy_id] = {
                        'config': strategy_config,
                        'file_path': str(config_file),
                        'loaded_at': datetime.now(),
                        'enabled': True
                    }
                    logger.info(f"加载策略配置: {strategy_id}")
            except Exception as e:
                logger.error(f"加载策略配置失败 {config_file}: {e}")
        
        # 加载策略组合配置
        combination_file = self.config_dir / "strategy_combinations.yaml"
        if combination_file.exists():
            try:
                with open(combination_file, 'r', encoding='utf-8') as f:
                    self.strategy_combinations = yaml.safe_load(f) or {}
                logger.info(f"加载策略组合配置: {len(self.strategy_combinations)} 个组合")
            except Exception as e:
                logger.error(f"加载策略组合配置失败: {e}")
    
    def _load_strategy_config(self, config_file: Path) -> Optional[Dict[str, Any]]:
        """加载单个策略配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.yaml':
                    return yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {config_file}")
                    return None
        except Exception as e:
            logger.error(f"读取配置文件失败 {config_file}: {e}")
            return None
    
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """获取所有可用策略"""
        return {
            strategy_id: {
                'name': config['config'].get('strategy', {}).get('name', strategy_id),
                'description': config['config'].get('strategy', {}).get('description', ''),
                'version': config['config'].get('strategy', {}).get('version', '1.0'),
                'category': config['config'].get('strategy', {}).get('category', 'unknown'),
                'enabled': config['enabled'],
                'file_path': config['file_path']
            }
            for strategy_id, config in self.available_strategies.items()
        }
    
    def get_strategy_combinations(self) -> Dict[str, Dict[str, Any]]:
        """获取策略组合配置"""
        return self.strategy_combinations
    
    def create_dynamic_strategy(self, strategy_config: Dict[str, Any]) -> 'DynamicStrategy':
        """根据配置创建动态策略"""
        return DynamicStrategy(strategy_config, self.indicator_registry)
    
    def create_combination_strategy(self, combination_id: str) -> 'CombinationStrategy':
        """创建组合策略"""
        if combination_id not in self.strategy_combinations:
            raise ValueError(f"策略组合不存在: {combination_id}")
        
        combination_config = self.strategy_combinations[combination_id]
        return CombinationStrategy(combination_config, self)
    
    def execute_strategy(self, strategy_id: str, stock_codes: List[str], 
                        target_date: str, **kwargs) -> List[Dict[str, Any]]:
        """执行单个策略"""
        if strategy_id not in self.available_strategies:
            raise ValueError(f"策略不存在: {strategy_id}")
        
        strategy_config = self.available_strategies[strategy_id]['config']
        dynamic_strategy = self.create_dynamic_strategy(strategy_config)
        
        return dynamic_strategy.select_stocks(stock_codes, target_date, **kwargs)
    
    def execute_combination(self, combination_id: str, stock_codes: List[str], 
                           target_date: str, **kwargs) -> List[Dict[str, Any]]:
        """执行策略组合"""
        combination_strategy = self.create_combination_strategy(combination_id)
        return combination_strategy.select_stocks(stock_codes, target_date, **kwargs)
    
    def reload_strategies(self):
        """重新加载所有策略配置"""
        self.available_strategies.clear()
        self.strategy_combinations.clear()
        self._load_all_strategies()
        logger.info("策略配置重新加载完成")

class DynamicStrategy:
    """动态策略类，基于配置文件创建"""
    
    def __init__(self, config: Dict[str, Any], indicator_registry: CompleteIndicatorRegistry):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.config = config
        self.indicator_registry = indicator_registry
        self.strategy_info = config.get('strategy', {})
        self.name = self.strategy_info.get('name', 'Unknown Strategy')
        
    def select_stocks(self, stock_codes: List[str], target_date: str, **kwargs) -> List[Dict[str, Any]]:
        """根据配置执行选股"""
        try:
            # 解析技术指标配置
            indicators_config = self.config.get('technical_indicators', {})
            primary_indicators = indicators_config.get('primary_indicators', [])
            
            # 解析选股参数
            selection_params = self.config.get('selection_parameters', {})
            max_selections = selection_params.get('max_selections', 50)
            min_score = selection_params.get('min_score', 0.6)
            
            # 解析过滤条件
            filters = self.config.get('filters', {})
            
            selected_stocks = []
            
            for stock_code in stock_codes:
                try:
                    # 计算股票评分
                    score = self._calculate_stock_score(stock_code, target_date, primary_indicators)
                    
                    # 应用过滤条件
                    if self._apply_filters(stock_code, target_date, filters):
                        if score >= min_score:
                            selected_stocks.append({
                                'stock_code': stock_code,
                                'score': score,
                                'strategy': self.name,
                                'selection_reason': 'config_based_selection',
                                'indicators_used': [ind.get('indicator_id') for ind in primary_indicators]
                            })
                
                except Exception as e:
                    logger.debug(f"处理股票 {stock_code} 失败: {e}")
                    continue
            
            # 排序并限制数量
            selected_stocks.sort(key=lambda x: x['score'], reverse=True)
            return selected_stocks[:max_selections]
            
        except Exception as e:
            logger.error(f"动态策略执行失败: {e}")
            return []
    
    def _calculate_stock_score(self, stock_code: str, target_date: str, 
                              indicators_config: List[Dict[str, Any]]) -> float:
        """计算股票评分"""
        total_score = 0.0
        total_weight = 0.0
        
        for indicator_config in indicators_config:
            try:
                indicator_id = indicator_config.get('indicator_id')
                conditions = indicator_config.get('conditions', [])
                weight = indicator_config.get('weight', 1.0)
                
                # 计算指标评分
                indicator_score = self._evaluate_indicator_conditions(
                    stock_code, target_date, indicator_id, conditions
                )
                
                total_score += indicator_score * weight
                total_weight += weight
                
            except Exception as e:
                logger.debug(f"计算指标 {indicator_id} 评分失败: {e}")
                continue
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_indicator_conditions(self, stock_code: str, target_date: str, 
                                     indicator_id: str, conditions: List[Dict[str, Any]]) -> float:
        """评估指标条件"""
        # 简化实现，实际应该根据具体指标和条件进行计算
        # 这里返回基础评分
        if indicator_id in ['KDJ', 'MACD', 'RSI']:
            return 0.7  # 基础评分
        return 0.5
    
    def _apply_filters(self, stock_code: str, target_date: str, 
                      filters: Dict[str, Any]) -> bool:
        """应用过滤条件"""
        # 简化实现，实际应该根据具体过滤条件进行判断
        return True

class CombinationStrategy:
    """组合策略类"""
    
    def __init__(self, combination_config: Dict[str, Any], engine: ConfigurableStrategyEngine):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self.config = combination_config
        self.engine = engine
        self.name = combination_config.get('name', 'Unknown Combination')
        
    def select_stocks(self, stock_codes: List[str], target_date: str, **kwargs) -> List[Dict[str, Any]]:
        """执行组合策略选股"""
        strategies = self.config.get('strategies', [])
        combination_method = self.config.get('combination_method', 'union')
        
        all_results = []
        
        # 执行各个子策略
        for strategy_config in strategies:
            strategy_id = strategy_config.get('strategy_id')
            weight = strategy_config.get('weight', 1.0)
            
            if strategy_id in self.engine.available_strategies:
                try:
                    results = self.engine.execute_strategy(strategy_id, stock_codes, target_date)
                    
                    # 应用权重
                    for result in results:
                        result['score'] *= weight
                        result['combination_strategy'] = self.name
                    
                    all_results.extend(results)
                    
                except Exception as e:
                    logger.error(f"执行子策略 {strategy_id} 失败: {e}")
        
        # 根据组合方法合并结果
        if combination_method == 'union':
            return self._union_results(all_results)
        elif combination_method == 'intersection':
            return self._intersection_results(all_results)
        elif combination_method == 'weighted_average':
            return self._weighted_average_results(all_results)
        else:
            return all_results
    
    def _union_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并结果（并集）"""
        stock_scores = {}
        
        for result in all_results:
            stock_code = result['stock_code']
            if stock_code not in stock_scores:
                stock_scores[stock_code] = result
            else:
                # 取最高评分
                if result['score'] > stock_scores[stock_code]['score']:
                    stock_scores[stock_code] = result
        
        results = list(stock_scores.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _intersection_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """交集结果"""
        # 简化实现
        return self._union_results(all_results)
    
    def _weighted_average_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """加权平均结果"""
        stock_scores = {}
        stock_counts = {}
        
        for result in all_results:
            stock_code = result['stock_code']
            if stock_code not in stock_scores:
                stock_scores[stock_code] = result.copy()
                stock_counts[stock_code] = 1
            else:
                stock_scores[stock_code]['score'] += result['score']
                stock_counts[stock_code] += 1
        
        # 计算平均分
        for stock_code in stock_scores:
            stock_scores[stock_code]['score'] /= stock_counts[stock_code]
        
        results = list(stock_scores.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
