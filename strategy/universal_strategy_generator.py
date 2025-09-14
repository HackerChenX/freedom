#!/usr/bin/env python3
"""
通用策略生成器 - 符合六层架构的生产级实现
L5: 业务应用层 - 策略生成业务逻辑
"""

import sys
import os
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.unified_container import container
from analysis.universal_buypoint_analyzer import UniversalBuyPointAnalyzer

logger = get_logger(__name__)

class UniversalStrategyGenerator:
    """
    通用策略生成器
    
    基于买点分析结果动态生成策略配置文件
    符合六层架构设计
    """
    
    def __init__(self, config_dir: str = "config/strategies"):
        """初始化策略生成器"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 依赖注入获取买点分析器
        self.buypoint_analyzer = UniversalBuyPointAnalyzer()
        
        logger.info("通用策略生成器初始化完成")
    
    def generate_strategy_from_buypoint(self, stock_code: str, target_date: str,
                                      strategy_name: Optional[str] = None,
                                      timeframes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        基于买点分析生成策略配置
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
            strategy_name: 策略名称，默认自动生成
            timeframes: 时间周期列表
            
        Returns:
            Dict: 生成的策略配置
        """
        logger.info(f"开始为 {stock_code} ({target_date}) 生成策略配置")
        
        try:
            # 1. 执行买点分析
            buypoint_result = self.buypoint_analyzer.analyze_buypoint(
                stock_code, target_date, timeframes
            )
            
            if not buypoint_result.get('success', False):
                logger.error(f"买点分析失败: {buypoint_result.get('error_message', '未知错误')}")
                return {}
            
            # 2. 生成策略配置
            strategy_config = self._create_strategy_config(
                buypoint_result, strategy_name
            )
            
            # 3. 保存策略配置文件
            config_file = self._save_strategy_config(strategy_config)
            
            logger.info(f"策略配置生成完成: {config_file}")
            
            return {
                'strategy_config': strategy_config,
                'config_file': str(config_file),
                'buypoint_analysis': buypoint_result,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"策略生成失败: {e}")
            return {'success': False, 'error_message': str(e)}
    
    def _create_strategy_config(self, buypoint_result: Dict[str, Any],
                               strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """创建策略配置"""
        stock_code = buypoint_result['stock_code']
        target_date = buypoint_result['target_date']
        pattern_matches = buypoint_result['pattern_matches']
        
        if strategy_name is None:
            strategy_name = f"auto_generated_{stock_code}_{target_date}_strategy"
        
        # 基础策略配置
        strategy_config = {
            "strategy": {
                "id": strategy_name,
                "name": f"自动生成-{stock_code}策略",
                "description": f"基于{stock_code}在{target_date}的买点分析自动生成",
                "version": "1.0",
                "category": "auto_generated",
                "target_stock": stock_code,
                "target_date": target_date,
                "generation_timestamp": datetime.now().isoformat()
            },
            "technical_indicators": {
                "primary_indicators": [],
                "combination_logic": "OR"  # 使用OR逻辑提高匹配概率
            },
            "selection_parameters": {
                "max_selections": 50,
                "min_score": 0.3,
                "ranking_method": "composite_score"
            },
            "metadata": {
                "source_analysis": {
                    "total_indicators": buypoint_result.get('total_indicators', 0),
                    "total_patterns": buypoint_result.get('total_patterns', 0),
                    "buypoint_score": buypoint_result.get('buypoint_score', 0.0)
                }
            }
        }
        
        # 基于形态匹配生成指标配置
        indicator_configs = self._generate_indicator_configs_from_patterns(pattern_matches)
        strategy_config["technical_indicators"]["primary_indicators"] = indicator_configs
        
        return strategy_config
    
    def _generate_indicator_configs_from_patterns(self, 
                                                 pattern_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于形态匹配生成指标配置"""
        indicator_configs = []
        
        # 按指标分组形态
        indicator_groups = {}
        for pattern in pattern_matches:
            pattern_id = pattern.get('pattern_id', '')
            
            # 从形态ID中提取指标名称
            indicator_name = self._extract_indicator_name(pattern_id)
            
            if indicator_name not in indicator_groups:
                indicator_groups[indicator_name] = []
            indicator_groups[indicator_name].append(pattern)
        
        # 为每个指标生成配置
        for indicator_name, patterns in indicator_groups.items():
            config = self._create_indicator_config(indicator_name, patterns)
            if config:
                indicator_configs.append(config)
        
        logger.info(f"生成了 {len(indicator_configs)} 个指标配置")
        return indicator_configs
    
    def _extract_indicator_name(self, pattern_id: str) -> str:
        """从形态ID中提取指标名称"""
        # 常见的指标前缀
        indicator_prefixes = [
            'KDJ', 'RSI', 'MACD', 'BOLL', 'MA', 'EMA', 'DMI', 'ADX',
            'ATR', 'CCI', 'WR', 'ROC', 'CMO', 'STOCHRSI', 'OBV', 'MFI',
            'AROON', 'SAR', 'TRIX', 'VR', 'PSY', 'ZXM'
        ]
        
        pattern_upper = pattern_id.upper()
        
        for prefix in indicator_prefixes:
            if pattern_upper.startswith(prefix):
                return prefix
        
        # 如果没有匹配到已知前缀，返回第一个下划线前的部分
        if '_' in pattern_id:
            return pattern_id.split('_')[0].upper()
        
        return 'UNKNOWN'
    
    def _create_indicator_config(self, indicator_name: str, 
                                patterns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """为指标创建配置"""
        if not patterns:
            return None
        
        # 计算平均匹配强度作为权重
        avg_strength = sum(p.get('match_strength', 0) for p in patterns) / len(patterns)
        
        # 基础配置模板
        config = {
            "indicator_id": indicator_name,
            "parameters": self._get_default_parameters(indicator_name),
            "conditions": self._generate_conditions_from_patterns(patterns),
            "weight": min(avg_strength, 0.5),  # 限制最大权重
            "metadata": {
                "source_patterns": [p.get('pattern_id') for p in patterns],
                "pattern_count": len(patterns),
                "avg_match_strength": avg_strength
            }
        }
        
        return config
    
    def _get_default_parameters(self, indicator_name: str) -> Dict[str, Any]:
        """获取指标的默认参数"""
        default_params = {
            'KDJ': {'k_period': 9, 'd_period': 3, 'j_multiplier': 3},
            'RSI': {'period': 14},
            'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'BOLL': {'period': 20, 'std_dev': 2},
            'MA': {'periods': [5, 10, 20, 60]},
            'EMA': {'periods': [12, 26]},
            'DMI': {'period': 14},
            'ADX': {'period': 14},
            'ATR': {'period': 14},
            'CCI': {'period': 14},
            'WR': {'period': 14},
            'ROC': {'period': 12},
            'CMO': {'period': 14},
            'STOCHRSI': {'period': 14, 'k_period': 3, 'd_period': 3},
            'OBV': {},
            'MFI': {'period': 14},
            'AROON': {'period': 14},
            'SAR': {'acceleration': 0.02, 'maximum': 0.2},
            'TRIX': {'period': 14},
            'VR': {'period': 26},
            'PSY': {'period': 12}
        }
        
        return default_params.get(indicator_name, {})
    
    def _generate_conditions_from_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于形态生成条件"""
        conditions = []
        
        for pattern in patterns:
            # 基于形态匹配强度生成宽松的条件
            match_strength = pattern.get('match_strength', 0.5)
            
            # 生成通用条件
            condition = {
                "field": "signal_strength",
                "operator": ">",
                "value": max(0.3, match_strength - 0.2),  # 比原始强度稍微宽松
                "weight": match_strength,
                "source_pattern": pattern.get('pattern_id')
            }
            
            conditions.append(condition)
        
        return conditions
    
    def _save_strategy_config(self, strategy_config: Dict[str, Any]) -> Path:
        """保存策略配置文件"""
        strategy_id = strategy_config['strategy']['id']
        config_file = self.config_dir / f"{strategy_id}.yaml"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(strategy_config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"策略配置已保存: {config_file}")
            return config_file
            
        except Exception as e:
            logger.error(f"保存策略配置失败: {e}")
            raise
    
    def batch_generate_strategies(self, stock_list: List[Dict[str, str]],
                                 timeframes: Optional[List[str]] = None) -> Dict[str, Any]:
        """批量生成策略配置"""
        logger.info(f"开始批量生成 {len(stock_list)} 个策略配置")
        
        results = {
            'success_count': 0,
            'failed_count': 0,
            'results': [],
            'summary': {}
        }
        
        for stock_info in stock_list:
            stock_code = stock_info.get('code')
            target_date = stock_info.get('date')
            
            if not stock_code or not target_date:
                logger.warning(f"跳过无效的股票信息: {stock_info}")
                results['failed_count'] += 1
                continue
            
            try:
                result = self.generate_strategy_from_buypoint(
                    stock_code, target_date, timeframes=timeframes
                )
                
                if result.get('success', False):
                    results['success_count'] += 1
                    logger.info(f"✅ {stock_code} ({target_date}) 策略生成成功")
                else:
                    results['failed_count'] += 1
                    logger.warning(f"❌ {stock_code} ({target_date}) 策略生成失败")
                
                results['results'].append({
                    'stock_code': stock_code,
                    'target_date': target_date,
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"处理 {stock_code} 时发生错误: {e}")
                results['failed_count'] += 1
        
        # 生成汇总信息
        results['summary'] = {
            'total_processed': len(stock_list),
            'success_rate': results['success_count'] / len(stock_list) * 100,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"批量生成完成: 成功 {results['success_count']}, "
                   f"失败 {results['failed_count']}, "
                   f"成功率 {results['summary']['success_rate']:.1f}%")
        
        return results
