#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
形态注册表管理器

管理技术指标形态的注册、验证和查询
确保所有指标的形态都正确注册到PatternRegistry中
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from datetime import datetime

from utils.logger import getLogger
from indicators.pattern_registry import (
    get_pattern_registry, PatternRegistry, 
    PatternTypePatternRegistry, PatternStrengthPatternRegistry, PatternPolarity
)
from .indicator_discovery import IndicatorDiscovery, IndicatorInfo

logger = getLogger(__name__)


@dataclass
class PatternInfo:
    """形态信息"""
    pattern_id: str
    display_name: str
    indicator_id: str
    pattern_type: str
    description: str
    score_impact: float
    polarity: str
    is_registered: bool = False
    registration_source: str = ""


@dataclass
class IndicatorPatternSummary:
    """指标形态摘要"""
    indicator_name: str
    total_patterns: int
    registered_patterns: int
    missing_patterns: List[str] = field(default_factory=list)
    duplicate_patterns: List[str] = field(default_factory=list)
    pattern_details: List[PatternInfo] = field(default_factory=list)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    total_indicators: int
    total_patterns: int
    registered_patterns: int
    missing_patterns: int
    duplicate_patterns: int
    validation_errors: List[str] = field(default_factory=list)
    indicator_summaries: List[IndicatorPatternSummary] = field(default_factory=list)


class PatternRegistryManager:
    """形态注册表管理器"""
    
    def __init__(self, indicator_discovery: Optional[IndicatorDiscovery] = None):
        """
        初始化形态注册表管理器
        
        Args:
            indicator_discovery: 指标发现系统实例
        """
        self.pattern_registry = get_pattern_registry()
        self.indicator_discovery = indicator_discovery or IndicatorDiscovery()
        
        # 缓存数据
        self.indicator_patterns_cache = {}
        self.pattern_info_cache = {}
        self.validation_cache = None
        
        logger.info("形态注册表管理器初始化完成")
    
    def ensure_patterns_registered(self, indicator_name: str) -> bool:
        """
        确保指定指标的所有形态都已注册
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            bool: 是否成功注册所有形态
        """
        logger.info(f"确保指标 {indicator_name} 的形态已注册...")
        
        try:
            # 获取指标信息
            indicator_info = self.indicator_discovery.get_indicator_by_name(indicator_name)
            if not indicator_info or not indicator_info.is_loadable:
                logger.warning(f"指标 {indicator_name} 不可用")
                return False
            
            # 加载指标实例
            indicator_instance = self.indicator_discovery.load_indicator(indicator_name)
            if not indicator_instance:
                logger.warning(f"无法加载指标 {indicator_name}")
                return False
            
            # 尝试注册形态
            registration_success = False
            
            # 方法1: 调用register_patterns方法
            if hasattr(indicator_instance, 'register_patterns'):
                try:
                    indicator_instance.register_patterns()
                    registration_success = True
                    logger.debug(f"通过register_patterns注册 {indicator_name} 的形态")
                except Exception as e:
                    logger.debug(f"register_patterns方法失败 {indicator_name}: {e}")
            
            # 方法2: 调用_register_patterns方法
            if not registration_success and hasattr(indicator_instance, '_register_patterns'):
                try:
                    indicator_instance._register_patterns()
                    registration_success = True
                    logger.debug(f"通过_register_patterns注册 {indicator_name} 的形态")
                except Exception as e:
                    logger.debug(f"_register_patterns方法失败 {indicator_name}: {e}")
            
            # 方法3: 手动创建默认形态
            if not registration_success:
                self._create_default_patterns_for_indicator(indicator_name)
                registration_success = True
                logger.debug(f"为 {indicator_name} 创建默认形态")
            
            # 验证注册结果
            registered_patterns = self.pattern_registry.get_patterns_by_indicator(indicator_name)
            if registered_patterns:
                logger.info(f"指标 {indicator_name} 成功注册 {len(registered_patterns)} 个形态")
                return True
            else:
                logger.warning(f"指标 {indicator_name} 没有注册任何形态")
                return False
                
        except Exception as e:
            logger.error(f"注册指标 {indicator_name} 的形态时出错: {e}")
            return False
    
    def _create_default_patterns_for_indicator(self, indicator_name: str) -> None:
        """
        为指标创建默认形态
        
        Args:
            indicator_name: 指标名称
        """
        try:
            # 根据指标类型创建不同的默认形态
            default_patterns = self._get_default_patterns_for_indicator(indicator_name)
            
            for pattern_config in default_patterns:
                self.pattern_registry.register_pattern_registry(
                    pattern_id=pattern_config['pattern_id'],
                    display_name=pattern_config['display_name'],
                    indicator_id=indicator_name,
                    pattern_type=pattern_config['pattern_type'],
                    description=pattern_config.get('description', ''),
                    score_impact=pattern_config.get('score_impact', 0.0),
                    allow_override=True
                )
            
            logger.debug(f"为 {indicator_name} 创建了 {len(default_patterns)} 个默认形态")
            
        except Exception as e:
            logger.error(f"为 {indicator_name} 创建默认形态失败: {e}")
    
    def _get_default_patterns_for_indicator(self, indicator_name: str) -> List[Dict[str, Any]]:
        """
        获取指标的默认形态配置
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            List[Dict[str, Any]]: 默认形态配置列表
        """
        # 基础形态
        base_patterns = [
            {
                'pattern_id': f"{indicator_name}_BULLISH",
                'display_name': f"{indicator_name}看涨信号",
                'pattern_type': PatternTypePatternRegistry.BULLISH,
                'description': f"{indicator_name}指标显示看涨信号",
                'score_impact': 10.0
            },
            {
                'pattern_id': f"{indicator_name}_BEARISH",
                'display_name': f"{indicator_name}看跌信号",
                'pattern_type': PatternTypePatternRegistry.BEARISH,
                'description': f"{indicator_name}指标显示看跌信号",
                'score_impact': -10.0
            }
        ]
        
        # 根据指标类型添加特定形态
        indicator_upper = indicator_name.upper()
        
        # 趋势指标
        if any(keyword in indicator_upper for keyword in ['MA', 'EMA', 'SMA', 'WMA', 'TREND']):
            base_patterns.extend([
                {
                    'pattern_id': f"{indicator_name}_GOLDEN_CROSS",
                    'display_name': f"{indicator_name}金叉",
                    'pattern_type': PatternTypePatternRegistry.BULLISH,
                    'description': f"{indicator_name}短期线上穿长期线",
                    'score_impact': 15.0
                },
                {
                    'pattern_id': f"{indicator_name}_DEATH_CROSS",
                    'display_name': f"{indicator_name}死叉",
                    'pattern_type': PatternTypePatternRegistry.BEARISH,
                    'description': f"{indicator_name}短期线下穿长期线",
                    'score_impact': -15.0
                }
            ])
        
        # 震荡指标
        elif any(keyword in indicator_upper for keyword in ['RSI', 'KDJ', 'CCI', 'WR', 'STOCH']):
            base_patterns.extend([
                {
                    'pattern_id': f"{indicator_name}_OVERSOLD",
                    'display_name': f"{indicator_name}超卖",
                    'pattern_type': PatternTypePatternRegistry.BULLISH,
                    'description': f"{indicator_name}指标进入超卖区域",
                    'score_impact': 12.0
                },
                {
                    'pattern_id': f"{indicator_name}_OVERBOUGHT",
                    'display_name': f"{indicator_name}超买",
                    'pattern_type': PatternTypePatternRegistry.BEARISH,
                    'description': f"{indicator_name}指标进入超买区域",
                    'score_impact': -12.0
                }
            ])
        
        # MACD特殊形态
        elif 'MACD' in indicator_upper:
            base_patterns.extend([
                {
                    'pattern_id': f"{indicator_name}_GOLDEN_CROSS",
                    'display_name': f"{indicator_name}金叉",
                    'pattern_type': PatternTypePatternRegistry.BULLISH,
                    'description': f"MACD DIF上穿DEA",
                    'score_impact': 15.0
                },
                {
                    'pattern_id': f"{indicator_name}_DEATH_CROSS",
                    'display_name': f"{indicator_name}死叉",
                    'pattern_type': PatternTypePatternRegistry.BEARISH,
                    'description': f"MACD DIF下穿DEA",
                    'score_impact': -15.0
                },
                {
                    'pattern_id': f"{indicator_name}_HISTOGRAM_POSITIVE",
                    'display_name': f"{indicator_name}柱状线转正",
                    'pattern_type': PatternTypePatternRegistry.BULLISH,
                    'description': f"MACD柱状线由负转正",
                    'score_impact': 8.0
                }
            ])
        
        # 成交量指标
        elif any(keyword in indicator_upper for keyword in ['VOL', 'OBV', 'VR', 'VOLUME']):
            base_patterns.extend([
                {
                    'pattern_id': f"{indicator_name}_VOLUME_SURGE",
                    'display_name': f"{indicator_name}放量",
                    'pattern_type': PatternTypePatternRegistry.BULLISH,
                    'description': f"{indicator_name}成交量显著放大",
                    'score_impact': 8.0
                },
                {
                    'pattern_id': f"{indicator_name}_VOLUME_SHRINK",
                    'display_name': f"{indicator_name}缩量",
                    'pattern_type': PatternTypePatternRegistry.NEUTRAL,
                    'description': f"{indicator_name}成交量萎缩",
                    'score_impact': 0.0
                }
            ])
        
        return base_patterns
    
    def get_all_patterns_by_indicator(self, indicator_name: str) -> List[PatternInfo]:
        """
        获取指标的所有已注册形态信息
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            List[PatternInfo]: 形态信息列表
        """
        if indicator_name in self.pattern_info_cache:
            return self.pattern_info_cache[indicator_name]
        
        pattern_infos = []
        pattern_ids = self.pattern_registry.get_patterns_by_indicator(indicator_name)
        
        for pattern_id in pattern_ids:
            pattern_data = self.pattern_registry.get_pattern(pattern_id)
            if pattern_data:
                pattern_info = PatternInfo(
                    pattern_id=pattern_id,
                    display_name=pattern_data.get('display_name', pattern_id),
                    indicator_id=indicator_name,
                    pattern_type=str(pattern_data.get('pattern_type', '')),
                    description=pattern_data.get('description', ''),
                    score_impact=pattern_data.get('score_impact', 0.0),
                    polarity=str(pattern_data.get('polarity', '')),
                    is_registered=True,
                    registration_source='PatternRegistry'
                )
                pattern_infos.append(pattern_info)
        
        # 缓存结果
        self.pattern_info_cache[indicator_name] = pattern_infos
        return pattern_infos
    
    def validate_pattern_registration(self) -> ValidationResult:
        """
        验证形态注册的完整性
        
        Returns:
            ValidationResult: 验证结果
        """
        if self.validation_cache:
            return self.validation_cache
        
        logger.info("开始验证形态注册完整性...")
        
        # 发现所有指标
        indicators = self.indicator_discovery.discover_all_indicators()
        loadable_indicators = [info for info in indicators if info.is_loadable]
        
        validation_result = ValidationResult(
            is_valid=True,
            total_indicators=len(loadable_indicators),
            total_patterns=0,
            registered_patterns=0,
            missing_patterns=0,
            duplicate_patterns=0
        )
        
        # 检查每个指标的形态注册情况
        for indicator_info in loadable_indicators:
            indicator_name = indicator_info.name
            
            try:
                # 确保形态已注册
                self.ensure_patterns_registered(indicator_name)
                
                # 获取注册的形态
                registered_patterns = self.pattern_registry.get_patterns_by_indicator(indicator_name)
                pattern_infos = self.get_all_patterns_by_indicator(indicator_name)
                
                # 创建指标摘要
                indicator_summary = IndicatorPatternSummary(
                    indicator_name=indicator_name,
                    total_patterns=len(registered_patterns),
                    registered_patterns=len(registered_patterns),
                    pattern_details=pattern_infos
                )
                
                validation_result.indicator_summaries.append(indicator_summary)
                validation_result.total_patterns += len(registered_patterns)
                validation_result.registered_patterns += len(registered_patterns)
                
                # 检查是否有足够的形态
                if len(registered_patterns) == 0:
                    validation_result.missing_patterns += 1
                    validation_result.is_valid = False
                    validation_result.validation_errors.append(
                        f"指标 {indicator_name} 没有注册任何形态"
                    )
                
            except Exception as e:
                logger.error(f"验证指标 {indicator_name} 时出错: {e}")
                validation_result.validation_errors.append(
                    f"验证指标 {indicator_name} 失败: {str(e)}"
                )
                validation_result.is_valid = False
        
        # 检查重复形态
        self._check_duplicate_patterns(validation_result)
        
        # 缓存验证结果
        self.validation_cache = validation_result
        
        logger.info(f"形态注册验证完成: {validation_result.registered_patterns}/{validation_result.total_patterns} 个形态已注册")
        
        return validation_result
    
    def _check_duplicate_patterns(self, validation_result: ValidationResult) -> None:
        """
        检查重复的形态
        
        Args:
            validation_result: 验证结果对象
        """
        pattern_counts = defaultdict(int)
        
        # 统计所有形态ID的出现次数
        for pattern_id in self.pattern_registry.get_all_pattern_ids():
            pattern_counts[pattern_id] += 1
        
        # 找出重复的形态
        duplicates = [pattern_id for pattern_id, count in pattern_counts.items() if count > 1]
        
        if duplicates:
            validation_result.duplicate_patterns = len(duplicates)
            validation_result.is_valid = False
            validation_result.validation_errors.append(
                f"发现 {len(duplicates)} 个重复形态: {duplicates[:5]}..."
            )
    
    def generate_pattern_inventory(self) -> Dict[str, Any]:
        """
        生成形态清单
        
        Returns:
            Dict[str, Any]: 形态清单
        """
        logger.info("生成形态清单...")
        
        validation_result = self.validate_pattern_registration()
        
        inventory = {
            'generation_time': datetime.now().isoformat(),
            'summary': {
                'total_indicators': validation_result.total_indicators,
                'total_patterns': validation_result.total_patterns,
                'registered_patterns': validation_result.registered_patterns,
                'missing_patterns': validation_result.missing_patterns,
                'duplicate_patterns': validation_result.duplicate_patterns,
                'validation_status': 'PASS' if validation_result.is_valid else 'FAIL'
            },
            'indicators': {},
            'pattern_types': defaultdict(int),
            'validation_errors': validation_result.validation_errors
        }
        
        # 按指标组织形态信息
        for indicator_summary in validation_result.indicator_summaries:
            indicator_data = {
                'indicator_name': indicator_summary.indicator_name,
                'total_patterns': indicator_summary.total_patterns,
                'patterns': []
            }
            
            for pattern_info in indicator_summary.pattern_details:
                pattern_data = {
                    'pattern_id': pattern_info.pattern_id,
                    'display_name': pattern_info.display_name,
                    'pattern_type': pattern_info.pattern_type,
                    'description': pattern_info.description,
                    'score_impact': pattern_info.score_impact,
                    'polarity': pattern_info.polarity
                }
                indicator_data['patterns'].append(pattern_data)
                
                # 统计形态类型
                inventory['pattern_types'][pattern_info.pattern_type] += 1
            
            inventory['indicators'][indicator_summary.indicator_name] = indicator_data
        
        # 转换defaultdict为普通dict
        inventory['pattern_types'] = dict(inventory['pattern_types'])
        
        logger.info(f"形态清单生成完成: {len(inventory['indicators'])} 个指标")
        
        return inventory
    
    def export_pattern_inventory(self, file_path: str) -> None:
        """
        导出形态清单到文件
        
        Args:
            file_path: 导出文件路径
        """
        inventory = self.generate_pattern_inventory()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(inventory, f, ensure_ascii=False, indent=2)
        
        logger.info(f"形态清单已导出到: {file_path}")
    
    def get_patterns_by_type(self, pattern_type: PatternTypePatternRegistry) -> List[str]:
        """
        根据形态类型获取形态列表
        
        Args:
            pattern_type: 形态类型
            
        Returns:
            List[str]: 形态ID列表
        """
        all_patterns = self.pattern_registry.get_all_patterns()
        matching_patterns = []
        
        for pattern_id, pattern_data in all_patterns.items():
            if pattern_data.get('pattern_type') == pattern_type:
                matching_patterns.append(pattern_id)
        
        return matching_patterns
    
    def get_patterns_by_polarity(self, polarity: PatternPolarity) -> List[str]:
        """
        根据极性获取形态列表
        
        Args:
            polarity: 形态极性
            
        Returns:
            List[str]: 形态ID列表
        """
        return self.pattern_registry.get_patterns_by_polarity(polarity)
    
    def get_buypoint_suitable_patterns(self) -> List[str]:
        """
        获取适合买点分析的形态列表
        
        Returns:
            List[str]: 适合买点分析的形态ID列表
        """
        # 获取正面和中性形态
        positive_patterns = self.get_patterns_by_polarity(PatternPolarity.POSITIVE)
        neutral_patterns = self.get_patterns_by_polarity(PatternPolarity.NEUTRAL)
        
        return positive_patterns + neutral_patterns
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.indicator_patterns_cache.clear()
        self.pattern_info_cache.clear()
        self.validation_cache = None
        logger.debug("形态注册表管理器缓存已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        validation_result = self.validate_pattern_registration()
        
        # 按形态类型统计
        pattern_type_stats = defaultdict(int)
        polarity_stats = defaultdict(int)
        
        for pattern_id in self.pattern_registry.get_all_pattern_ids():
            pattern_data = self.pattern_registry.get_pattern(pattern_id)
            if pattern_data:
                pattern_type = pattern_data.get('pattern_type')
                polarity = pattern_data.get('polarity')
                
                if pattern_type:
                    pattern_type_stats[str(pattern_type)] += 1
                if polarity:
                    polarity_stats[str(polarity)] += 1
        
        return {
            'total_indicators': validation_result.total_indicators,
            'total_patterns': validation_result.total_patterns,
            'registered_patterns': validation_result.registered_patterns,
            'missing_patterns': validation_result.missing_patterns,
            'duplicate_patterns': validation_result.duplicate_patterns,
            'validation_status': 'PASS' if validation_result.is_valid else 'FAIL',
            'pattern_type_distribution': dict(pattern_type_stats),
            'polarity_distribution': dict(polarity_stats),
            'buypoint_suitable_patterns': len(self.get_buypoint_suitable_patterns())
        }


def main():
    """测试形态注册表管理器"""
    print("初始化形态注册表管理器...")
    
    # 创建指标发现系统
    discovery = IndicatorDiscovery()
    indicators = discovery.discover_all_indicators()
    print(f"发现 {len(indicators)} 个指标")
    
    # 创建形态注册表管理器
    manager = PatternRegistryManager(discovery)
    
    # 验证形态注册
    print("\n验证形态注册...")
    validation_result = manager.validate_pattern_registration()
    
    print(f"验证结果: {'通过' if validation_result.is_valid else '失败'}")
    print(f"总指标: {validation_result.total_indicators}")
    print(f"总形态: {validation_result.total_patterns}")
    print(f"已注册: {validation_result.registered_patterns}")
    
    if validation_result.validation_errors:
        print(f"\n验证错误:")
        for error in validation_result.validation_errors[:5]:
            print(f"  - {error}")
    
    # 生成形态清单
    print("\n生成形态清单...")
    inventory = manager.generate_pattern_inventory()
    
    print(f"清单摘要:")
    print(f"  指标数量: {inventory['summary']['total_indicators']}")
    print(f"  形态数量: {inventory['summary']['total_patterns']}")
    print(f"  形态类型分布: {inventory['pattern_types']}")
    
    # 导出清单
    output_file = f"pattern_inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manager.export_pattern_inventory(output_file)
    print(f"\n形态清单已导出到: {output_file}")


if __name__ == "__main__":
    main()