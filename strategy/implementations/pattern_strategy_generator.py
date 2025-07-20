#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
形态策略生成器

为系统中所有技术指标的所有形态模式自动生成独立的选股策略。

核心功能：
1. 自动识别指标支持的形态模式
2. 为每个形态生成独立的选股策略
3. 智能参数配置和优化
4. 策略有效性预验证

Author: AI Assistant
Date: 2025-07-19
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from indicators.complete_indicator_registry import complete_registry

logger = get_logger(__name__)


class PatternStrategyGenerator:
    """形态策略生成器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化策略生成器
        
        Args:
            config: 生成配置
        """
        self.config = config or self._get_default_config()
        self.indicator_registry = complete_registry
        
        # 策略模板
        self.strategy_templates = self._load_strategy_templates()
        
        # 生成统计
        self.generation_stats = {
            'total_indicators': 0,
            'total_patterns': 0,
            'generated_strategies': 0,
            'failed_generations': 0,
            'validated_strategies': 0
        }
        
        logger.info("⚙️ 形态策略生成器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'generation': {
                'include_all_patterns': True,
                'validate_strategies': True,
                'optimize_parameters': True,
                'generate_combinations': False
            },
            'strategy_params': {
                'default_threshold': 0.6,
                'default_weight': 1.0,
                'min_score': 0.5,
                'max_selections_ratio': 0.1
            },
            'validation': {
                'sample_size': 100,
                'min_success_rate': 0.1,
                'require_selections': True
            },
            'output': {
                'save_strategies': True,
                'strategy_file': 'data/generated_strategies.json',
                'template_file': 'data/strategy_templates.json'
            }
        }
    
    def _load_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载策略模板"""
        try:
            templates = {
                'trend_following': {
                    'description': '趋势跟踪策略模板',
                    'conditions': [
                        {
                            'type': 'indicator_signal',
                            'operator': '>',
                            'threshold': 0.6,
                            'weight': 1.0
                        }
                    ],
                    'selection_criteria': {
                        'min_score': 0.6,
                        'sort_by': 'score',
                        'sort_order': 'desc'
                    }
                },
                'mean_reversion': {
                    'description': '均值回归策略模板',
                    'conditions': [
                        {
                            'type': 'indicator_signal',
                            'operator': '<',
                            'threshold': 0.3,
                            'weight': 1.0
                        }
                    ],
                    'selection_criteria': {
                        'min_score': 0.5,
                        'sort_by': 'score',
                        'sort_order': 'asc'
                    }
                },
                'momentum': {
                    'description': '动量策略模板',
                    'conditions': [
                        {
                            'type': 'indicator_signal',
                            'operator': '>',
                            'threshold': 0.7,
                            'weight': 1.2
                        }
                    ],
                    'selection_criteria': {
                        'min_score': 0.7,
                        'sort_by': 'score',
                        'sort_order': 'desc'
                    }
                },
                'oversold_bounce': {
                    'description': '超卖反弹策略模板',
                    'conditions': [
                        {
                            'type': 'indicator_signal',
                            'operator': '<',
                            'threshold': 0.2,
                            'weight': 1.0
                        }
                    ],
                    'selection_criteria': {
                        'min_score': 0.4,
                        'sort_by': 'score',
                        'sort_order': 'asc'
                    }
                }
            }
            
            return templates
            
        except Exception as e:
            logger.error(f"❌ 加载策略模板失败: {e}")
            return {}
    
    @performance_monitor(threshold=30.0)
    @exception_handler(reraise=True)
    def generate_all_strategies(self) -> List[Dict[str, Any]]:
        """
        生成所有指标形态策略
        
        Returns:
            List[Dict[str, Any]]: 生成的策略列表
        """
        try:
            logger.info("🎯 开始生成所有指标形态策略")
            
            # 获取所有指标
            indicators = self._get_all_indicators()
            self.generation_stats['total_indicators'] = len(indicators)
            
            # 生成策略
            all_strategies = []
            
            for indicator_name in indicators:
                try:
                    # 获取指标的形态
                    patterns = self._get_indicator_patterns(indicator_name)
                    self.generation_stats['total_patterns'] += len(patterns)
                    
                    # 为每个形态生成策略
                    for pattern in patterns:
                        strategy = self._generate_pattern_strategy(indicator_name, pattern)
                        if strategy:
                            all_strategies.append(strategy)
                            self.generation_stats['generated_strategies'] += 1
                        else:
                            self.generation_stats['failed_generations'] += 1
                            
                except Exception as e:
                    logger.warning(f"⚠️ 生成指标 {indicator_name} 策略失败: {e}")
                    self.generation_stats['failed_generations'] += 1
                    continue
            
            # 验证策略
            if self.config['generation']['validate_strategies']:
                validated_strategies = self._validate_strategies(all_strategies)
                self.generation_stats['validated_strategies'] = len(validated_strategies)
            else:
                validated_strategies = all_strategies
            
            # 保存策略
            if self.config['output']['save_strategies']:
                self._save_strategies(validated_strategies)
            
            logger.info(f"✅ 策略生成完成: 生成{len(all_strategies)}个, 验证通过{len(validated_strategies)}个")
            return validated_strategies
            
        except Exception as e:
            logger.error(f"❌ 策略生成失败: {e}")
            raise
    
    def _get_all_indicators(self) -> List[str]:
        """获取所有指标"""
        try:
            # 确保指标已注册
            self.indicator_registry.register_all_indicators()
            
            # 获取已注册的指标
            indicators = self.indicator_registry.get_all_indicators()
            
            logger.info(f"📊 发现指标: {len(indicators)}个")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 获取指标列表失败: {e}")
            return []
    
    def _get_indicator_patterns(self, indicator_name: str) -> List[str]:
        """获取指标的形态列表"""
        try:
            # 获取指标实例
            indicator = self.indicator_registry.get_indicator(indicator_name)
            if not indicator:
                logger.warning(f"⚠️ 未找到指标: {indicator_name}")
                return []
            
            patterns = []
            
            # 方法1: 检查get_pattern_info方法
            if hasattr(indicator, 'get_pattern_info'):
                try:
                    pattern_info = indicator.get_pattern_info()
                    if isinstance(pattern_info, dict):
                        patterns.extend(pattern_info.keys())
                    elif isinstance(pattern_info, list):
                        patterns.extend(pattern_info)
                except Exception as e:
                    logger.warning(f"⚠️ 获取 {indicator_name} 形态信息失败: {e}")
            
            # 方法2: 检查identify_patterns方法
            if hasattr(indicator, 'identify_patterns'):
                try:
                    # 使用示例数据测试
                    sample_data = self._get_sample_data()
                    if sample_data is not None:
                        pattern_result = indicator.identify_patterns(sample_data)
                        if isinstance(pattern_result, pd.DataFrame):
                            patterns.extend(pattern_result.columns.tolist())
                        elif isinstance(pattern_result, list):
                            patterns.extend(pattern_result)
                except Exception:
                    pass  # 忽略示例测试错误
            
            # 方法3: 基于指标类型推断默认形态
            if not patterns:
                patterns = self._infer_default_patterns(indicator_name, indicator)
            
            # 去重并过滤
            patterns = list(set(patterns))
            patterns = [p for p in patterns if p and isinstance(p, str)]
            
            logger.debug(f"📋 指标 {indicator_name} 形态: {patterns}")
            return patterns
            
        except Exception as e:
            logger.error(f"❌ 获取指标 {indicator_name} 形态失败: {e}")
            return []
    
    def _infer_default_patterns(self, indicator_name: str, indicator) -> List[str]:
        """推断默认形态"""
        try:
            # 基于指标名称推断
            name_lower = indicator_name.lower()
            
            if any(trend in name_lower for trend in ['ma', 'ema', 'trend', 'dmi', 'adx']):
                return ['trend_up', 'trend_down', 'trend_stable']
            elif any(osc in name_lower for osc in ['rsi', 'kdj', 'cci', 'wr']):
                return ['overbought', 'oversold', 'neutral']
            elif any(vol in name_lower for vol in ['volume', 'vol', 'obv', 'mfi']):
                return ['volume_surge', 'volume_dry', 'volume_normal']
            elif any(mom in name_lower for mom in ['macd', 'momentum', 'roc']):
                return ['bullish_signal', 'bearish_signal', 'neutral_signal']
            else:
                return ['signal_positive', 'signal_negative', 'signal_neutral']
                
        except Exception as e:
            logger.warning(f"⚠️ 推断默认形态失败: {e}")
            return ['default_signal']
    
    def _get_sample_data(self) -> Optional[pd.DataFrame]:
        """获取示例数据"""
        try:
            # 创建示例数据
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            n = len(dates)
            
            # 模拟股价数据
            np.random.seed(42)
            close_prices = 100 + np.cumsum(np.random.randn(n) * 0.02)
            
            sample_data = pd.DataFrame({
                'datetime': dates,
                'open': close_prices * (1 + np.random.randn(n) * 0.001),
                'high': close_prices * (1 + np.abs(np.random.randn(n)) * 0.01),
                'low': close_prices * (1 - np.abs(np.random.randn(n)) * 0.01),
                'close': close_prices,
                'volume': np.random.randint(1000000, 10000000, n),
                'amount': close_prices * np.random.randint(1000000, 10000000, n)
            })
            
            return sample_data
            
        except Exception as e:
            logger.warning(f"⚠️ 生成示例数据失败: {e}")
            return None
    
    def _generate_pattern_strategy(self, indicator_name: str, pattern: str) -> Optional[Dict[str, Any]]:
        """生成单个形态策略"""
        try:
            # 选择策略模板
            template = self._select_strategy_template(indicator_name, pattern)
            
            # 生成策略ID和名称
            strategy_id = f"{indicator_name}_{pattern}_strategy"
            strategy_name = f"{indicator_name} {pattern} 选股策略"
            
            # 构建策略
            strategy = {
                'id': strategy_id,
                'name': strategy_name,
                'description': f"基于{indicator_name}指标的{pattern}形态选股策略",
                'indicator': indicator_name,
                'pattern': pattern,
                'template': template['description'],
                'conditions': self._build_strategy_conditions(indicator_name, pattern, template),
                'selection_criteria': self._build_selection_criteria(template),
                'parameters': self._build_strategy_parameters(indicator_name, pattern),
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'generator_version': '1.0',
                    'auto_generated': True
                }
            }
            
            return strategy
            
        except Exception as e:
            logger.error(f"❌ 生成策略失败 {indicator_name}_{pattern}: {e}")
            return None
    
    def _select_strategy_template(self, indicator_name: str, pattern: str) -> Dict[str, Any]:
        """选择策略模板"""
        try:
            name_lower = indicator_name.lower()
            pattern_lower = pattern.lower()
            
            # 基于指标和形态选择模板
            if any(trend in name_lower for trend in ['ma', 'ema', 'trend']) and 'up' in pattern_lower:
                return self.strategy_templates['trend_following']
            elif any(osc in name_lower for osc in ['rsi', 'wr']) and 'oversold' in pattern_lower:
                return self.strategy_templates['oversold_bounce']
            elif 'momentum' in name_lower or 'macd' in name_lower:
                return self.strategy_templates['momentum']
            elif 'down' in pattern_lower or 'bear' in pattern_lower:
                return self.strategy_templates['mean_reversion']
            else:
                return self.strategy_templates['trend_following']  # 默认模板
                
        except Exception as e:
            logger.warning(f"⚠️ 选择策略模板失败: {e}")
            return self.strategy_templates['trend_following']
    
    def _build_strategy_conditions(self, indicator_name: str, pattern: str, 
                                 template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """构建策略条件"""
        try:
            conditions = []
            
            for condition_template in template['conditions']:
                condition = {
                    'indicator': indicator_name,
                    'pattern': pattern,
                    'operator': condition_template['operator'],
                    'threshold': condition_template['threshold'],
                    'weight': condition_template['weight'],
                    'type': 'pattern_match'
                }
                conditions.append(condition)
            
            return conditions
            
        except Exception as e:
            logger.error(f"❌ 构建策略条件失败: {e}")
            return []
    
    def _build_selection_criteria(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """构建选择标准"""
        try:
            criteria = template['selection_criteria'].copy()
            
            # 添加默认参数
            criteria.update({
                'max_selections': int(1000 * self.config['strategy_params']['max_selections_ratio']),
                'min_score': self.config['strategy_params']['min_score']
            })
            
            return criteria
            
        except Exception as e:
            logger.error(f"❌ 构建选择标准失败: {e}")
            return {'min_score': 0.5, 'max_selections': 100}
    
    def _build_strategy_parameters(self, indicator_name: str, pattern: str) -> Dict[str, Any]:
        """构建策略参数"""
        try:
            return {
                'indicator_params': {},
                'pattern_params': {},
                'optimization_params': {
                    'enable_optimization': self.config['generation']['optimize_parameters'],
                    'optimization_target': 'sharpe_ratio'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 构建策略参数失败: {e}")
            return {}
    
    def _validate_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证策略有效性"""
        try:
            logger.info(f"🔍 开始验证 {len(strategies)} 个策略")
            
            validated_strategies = []
            
            for strategy in strategies:
                try:
                    if self._validate_single_strategy(strategy):
                        validated_strategies.append(strategy)
                        
                except Exception as e:
                    logger.warning(f"⚠️ 验证策略 {strategy['id']} 失败: {e}")
                    continue
            
            logger.info(f"✅ 策略验证完成: {len(validated_strategies)}/{len(strategies)} 通过")
            return validated_strategies
            
        except Exception as e:
            logger.error(f"❌ 策略验证失败: {e}")
            return strategies
    
    def _validate_single_strategy(self, strategy: Dict[str, Any]) -> bool:
        """验证单个策略"""
        try:
            # 基本结构验证
            required_fields = ['id', 'name', 'indicator', 'pattern', 'conditions']
            for field in required_fields:
                if field not in strategy:
                    logger.warning(f"⚠️ 策略 {strategy.get('id', 'unknown')} 缺少字段: {field}")
                    return False
            
            # 条件验证
            if not strategy['conditions']:
                logger.warning(f"⚠️ 策略 {strategy['id']} 没有条件")
                return False
            
            # 指标存在性验证
            indicator_name = strategy['indicator']
            if not self.indicator_registry.get_indicator(indicator_name):
                logger.warning(f"⚠️ 策略 {strategy['id']} 指标不存在: {indicator_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 验证策略失败: {e}")
            return False
    
    def _save_strategies(self, strategies: List[Dict[str, Any]]):
        """保存策略到文件"""
        try:
            output_file = self.config['output']['strategy_file']
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(strategies, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 策略已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存策略失败: {e}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """获取生成统计"""
        return self.generation_stats.copy()

    def _get_all_indicators_with_patterns(self) -> Dict[str, List[str]]:
        """获取所有指标及其形态"""
        try:
            indicator_patterns = {}

            # 获取所有指标
            indicators = self._get_all_indicators()

            for indicator_name in indicators:
                patterns = self._get_indicator_patterns(indicator_name)
                if patterns:
                    indicator_patterns[indicator_name] = patterns

            return indicator_patterns

        except Exception as e:
            logger.error(f"❌ 获取指标形态失败: {e}")
            return {}
